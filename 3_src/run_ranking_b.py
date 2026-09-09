#!/usr/bin/env python3
"""Ranking A/B 單一交付入口：先產生可審核命令計畫，明確授權後才執行。

預設 profile 是 ``rankB_robust``，也預設為 dry-run。這個入口本身只使用
Python 標準庫，因此即使目前機器沒有 SAM3/SAMURAI 環境、權重或解包後資料，仍會
列出完整命令與 fail-closed 前置檢查；它不會把「產生了命令」誤報成「推論已完成」。

常用方式：

  python 3_src/run_ranking_b.py
  python 3_src/run_ranking_b.py --profile rankB_robust --frames-root TEST_FC \
      --sample sample_submisson.csv --sam3-ckpt sam3.pt \
      --samurai-dir SAMURAI --samurai-ckpt sam2.1_hiera_large.pt
  # 前置檢查全部通過後，才加：
  python 3_src/run_ranking_b.py ... --execute

offline two-pass crop 無論 profile 為何，都只有顯式傳入
``--allow-offline-two-pass`` 才會進入命令計畫與執行。``rankA_best`` 的 profile intent
是 enabled；若沒有此旗標，dry-run 會標成 BLOCK，execute 則拒絕啟動。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


SRC_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SRC_ROOT.parent
DEFAULT_CONFIG = SRC_ROOT / "configs" / "ranking_profiles.json"
SUBMISSION_COLUMNS = ["ID", "x", "y", "width", "height"]
LEVEL_ORDER = {"OK": 0, "WARN": 1, "BLOCK": 2}


@dataclass(frozen=True)
class Finding:
    level: str
    check: str
    detail: str


@dataclass(frozen=True)
class PlanStep:
    name: str
    kind: str
    argv: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    condition: str = "always"
    detail: str = ""


def _path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def _executable_arg(value: str | Path) -> Path:
    """保留 venv Python symlink，也接受 ``python3`` 這類 PATH command。

    不可用 ``Path.resolve()``：uv 的 ``venv/bin/python`` 指向 shared base
    interpreter，resolve 後會靜默丟掉 venv 及其 site-packages。
    """
    raw = str(value)
    expanded = Path(raw).expanduser()
    if expanded.is_absolute() or expanded.parent != Path("."):
        return Path(os.path.abspath(str(expanded)))
    found = shutil.which(raw)
    return Path(os.path.abspath(found)) if found else Path(os.path.abspath(str(expanded)))


def load_profiles(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """讀取並驗證 semantic profile；刻意不依賴 YAML/Pydantic。"""
    try:
        doc = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"找不到 profile 設定檔：{config_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"profile JSON 無效：{config_path}:{exc.lineno}: {exc.msg}") from exc
    if doc.get("schema_version") != 1:
        raise ValueError(f"不支援 schema_version={doc.get('schema_version')!r}（目前只接受 1）")
    profiles = doc.get("profiles")
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError("profile JSON 缺非空 profiles object")
    if doc.get("default_profile") not in profiles:
        raise ValueError("default_profile 不在 profiles 中")
    for name, profile in profiles.items():
        _validate_profile(name, profile)
    return doc


def _validate_profile(name: str, profile: Any) -> None:
    if not isinstance(profile, dict):
        raise ValueError(f"{name}: profile 必須是 object")
    for section in ("tracking", "offline_two_pass_crop", "postprocess", "validation"):
        if not isinstance(profile.get(section), dict):
            raise ValueError(f"{name}: 缺 object `{section}`")
    tracking = profile["tracking"]
    if tracking.get("primary_backend") != "sam3" or tracking.get("source_backend") != "samurai":
        raise ValueError(f"{name}: 目前入口只支援 primary=sam3、source=samurai")
    if not isinstance(tracking.get("samurai_reset_between_sequences"), bool):
        raise ValueError(f"{name}: samurai_reset_between_sequences 必須是 boolean")
    if not isinstance(tracking.get("sam3_eval"), bool):
        raise ValueError(f"{name}: sam3_eval 必須是 boolean")
    crop = profile["offline_two_pass_crop"]
    if crop.get("profile_intent") not in {"enabled", "disabled"}:
        raise ValueError(f"{name}: offline two-pass profile_intent 必須是 enabled/disabled")
    try:
        area = float(crop["area_fraction_max"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{name}: area_fraction_max 必須是數值") from exc
    if not 0.0 < area <= 1.0:
        raise ValueError(f"{name}: area_fraction_max 必須在 (0,1]")
    _validate_window_consensus(name, crop)
    post = profile["postprocess"]
    if post.get("coordinate_correction") not in {"both", "top-only", "none"}:
        raise ValueError(f"{name}: coordinate_correction 無效")
    if post.get("quality_head") not in {"v056", "none"}:
        raise ValueError(f"{name}: quality_head 無效")
    if post.get("selector", "none") not in {"v2", "none"}:
        raise ValueError(f"{name}: selector 無效（v2/none）")
    if not isinstance(post.get("third_leg_rescue", False), bool):
        raise ValueError(f"{name}: third_leg_rescue 必須是布林")
    if post.get("first_frame_policy") != "preserve_init":
        raise ValueError(f"{name}: 此入口只接受 first_frame_policy=preserve_init")
    if not isinstance(post.get("frozen_splice_k"), int) or post["frozen_splice_k"] < 1:
        raise ValueError(f"{name}: frozen_splice_k 必須是正整數")
    validation = profile["validation"]
    if validation.get("sample_submission_required") is not True:
        raise ValueError(f"{name}: 交付 profile 必須 sample_submission_required=true")


def _validate_window_consensus(name: str, crop: dict[str, Any]) -> None:
    """crop 窗三方共識（D101）。缺這個 key ＝ 單一窗，行為與 08-31 前完全相同。"""
    wc = crop.get("window_consensus")
    if wc is None:
        return
    if not isinstance(wc, dict):
        raise ValueError(f"{name}: window_consensus 必須是 object")
    if not isinstance(wc.get("enabled"), bool):
        raise ValueError(f"{name}: window_consensus.enabled 必須是 boolean")
    if not wc["enabled"]:
        return
    if crop.get("profile_intent") != "enabled":
        raise ValueError(f"{name}: window_consensus 需要 profile_intent=enabled（窗共識只作用在 crop 階段）")
    variants = wc.get("variants")
    if not isinstance(variants, list) or len(variants) < 3:
        # N=2 時 medoid 恆等於現任（sum 對兩者是同一個 IoU 值，平手取第一個）
        # ⇒ 兩組窗測不到任何東西。這是 run_ensemble_medoid.py docstring 的明文契約，
        #   也是 08-30 v084 有 40 支序列等於沒被驗證的原因。
        raise ValueError(f"{name}: window_consensus.variants 至少要三組（N=2 時 medoid 恆等於現任）")
    tags: list[str] = []
    for index, variant in enumerate(variants):
        if not isinstance(variant, dict):
            raise ValueError(f"{name}: window_consensus.variants[{index}] 必須是 object")
        tag = variant.get("tag")
        if not isinstance(tag, str) or not re.fullmatch(r"[A-Za-z0-9]+", tag or ""):
            raise ValueError(f"{name}: variants[{index}].tag 必須是英數字串（會進檔名與步驟名）")
        if tag in tags:
            raise ValueError(f"{name}: window_consensus 有重複 tag {tag!r}")
        tags.append(tag)
        if not isinstance(variant.get("envelope_extra"), bool):
            raise ValueError(f"{name}: variants[{tag}].envelope_extra 必須是 boolean")
        segments = variant.get("segments")
        if not isinstance(segments, int) or isinstance(segments, bool) or segments < 1:
            raise ValueError(f"{name}: variants[{tag}].segments 必須是 ≥1 的整數")
    if wc.get("source_leg_variant") not in tags:
        raise ValueError(f"{name}: window_consensus.source_leg_variant 必須是 {tags} 之一")
    if wc.get("medoid_tie_breaker") != "first_variant":
        # medoid 平手時取第一個輸入。把它寫進設定檔是為了讓「variants 的順序有語意」
        # 這件事無法被靜默改掉——重排 variants 會改變輸出。
        raise ValueError(f"{name}: window_consensus.medoid_tie_breaker 目前只支援 'first_variant'")


def window_consensus(profile: dict[str, Any]) -> dict[str, Any] | None:
    """回傳啟用中的 window_consensus 設定；未設定或 enabled=false 時回 None。"""
    wc = profile["offline_two_pass_crop"].get("window_consensus")
    if isinstance(wc, dict) and wc.get("enabled"):
        return wc
    return None


def effective_two_pass(profile: dict[str, Any], allow_offline_two_pass: bool) -> bool:
    """唯一啟用條件是顯式 CLI 授權；profile intent 本身不構成授權。"""
    return bool(allow_offline_two_pass)


def _track_command(
    python: Path,
    frames_root: Path,
    out_dir: Path,
    backend: str,
    sample: Path | None,
    sam3_ckpt: Path,
    samurai_dir: Path,
    samurai_ckpt: Path,
    samurai_reset: bool,
    sam3_eval: bool,
    source_revision: str,
    seq_list: Path | None = None,
) -> list[str]:
    argv = [
        str(python), str(SRC_ROOT / "track_t1.py"),
        "--frames-root", str(frames_root),
        "--out-dir", str(out_dir),
        "--backend", backend,
    ]
    if seq_list is not None:
        argv += ["--seq-list", str(seq_list)]
    if sample is not None:
        argv += ["--sample-csv", str(sample)]
    if backend == "sam3":
        argv += ["--sam3-ckpt", str(sam3_ckpt)]
        if sam3_eval:
            argv.append("--sam3-eval")
    else:
        argv += [
            "--samurai-dir", str(samurai_dir),
            "--ckpt", str(samurai_ckpt),
        ]
        if samurai_reset:
            argv.append("--samurai-reset-kf")
        else:
            # track_t1 的安全預設已改成每序列 reset。Ranking A 要重現歷史 E02
            # 跨序列 KF 狀態時必須顯式 opt-in；省略旗標已不再代表 legacy 行為。
            argv.append("--samurai-legacy-cross-seq-kf")
    if source_revision:
        argv += ["--source-revision", source_revision]
    return argv


def build_plan(
    profile_name: str,
    profile: dict[str, Any],
    *,
    frames_root: Path,
    sample: Path,
    work_dir: Path,
    out: Path,
    sam3_python: Path,
    samurai_python: Path,
    sam3_ckpt: Path,
    samurai_dir: Path,
    samurai_ckpt: Path,
    qhead_weights: Path,
    selector_weights: Path,
    allow_offline_two_pass: bool,
    sam3_source_revision: str = "",
    samurai_source_revision: str = "",
) -> tuple[list[PlanStep], dict[str, Path | bool]]:
    """把 semantic keys 映射成現有 CLI；首幀策略由 runner 內建，非猜旗標。"""
    post = profile["postprocess"]
    reset = bool(profile["tracking"]["samurai_reset_between_sequences"])
    sam3_eval = bool(profile["tracking"]["sam3_eval"])
    use_crop = effective_two_pass(profile, allow_offline_two_pass)
    full_primary_dir = work_dir / "full_sam3"
    full_source_dir = work_dir / "full_samurai"
    full_primary = full_primary_dir / "submission.csv"
    full_source = full_source_dir / "submission.csv"
    crop_root = work_dir / "offline_two_pass" / "frames"
    crop_meta = work_dir / "offline_two_pass" / "crop_meta.json"
    crop_seqs = work_dir / "offline_two_pass" / "crop_seqs.txt"
    merged_primary = work_dir / "offline_two_pass" / "main_merged.csv"
    merged_source = work_dir / "offline_two_pass" / "source_merged.csv"
    pre_first = work_dir / "final_before_first_frame_restore.csv"
    py_path = str(SRC_ROOT)
    steps = [
        PlanStep(
            "track-full-source-samurai", "command",
            _track_command(
                samurai_python, frames_root, full_source_dir, "samurai", sample,
                sam3_ckpt, samurai_dir, samurai_ckpt, reset, sam3_eval,
                samurai_source_revision,
            ),
            detail="全圖 source；rankB_robust 必帶 --samurai-reset-kf",
        ),
        PlanStep(
            "track-full-primary-sam3", "command",
            _track_command(
                sam3_python, frames_root, full_primary_dir, "sam3", sample,
                sam3_ckpt, samurai_dir, samurai_ckpt, reset, sam3_eval,
                sam3_source_revision,
            ),
            detail="全圖 primary",
        ),
        PlanStep(
            "reject-base-fallbacks", "internal",
            detail="讀兩份 diagnostics.json；任何 error/fallback 皆停止",
        ),
    ]
    main_csv, source_csv = full_primary, full_source
    crop_diagnostics: list[list[str]] = []
    variant_paths: dict[str, dict[str, str]] = {}
    if use_crop:
        wc = window_consensus(profile)
        if wc is None:
            area = profile["offline_two_pass_crop"]["area_fraction_max"]
            steps += [
                PlanStep(
                    "offline-two-pass-prep", "command",
                    [
                        str(sam3_python), "-m", "hsot.crop_rerun", "prep",
                        "--frames-root", str(frames_root),
                        "--base-csv", str(full_primary),
                        "--envelope-extra", str(full_source),
                        "--area-frac-max", str(area),
                        "--out-root", str(crop_root),
                        "--meta", str(crop_meta),
                    ],
                    {"PYTHONPATH": py_path},
                    detail="非因果：以完整未來軌跡決定固定 crop 窗",
                ),
                PlanStep(
                    "materialize-crop-sequence-list", "internal",
                    detail=f"從 {crop_meta} 產生 {crop_seqs}",
                ),
                PlanStep(
                    "track-crop-primary-sam3", "command",
                    _track_command(
                        sam3_python, crop_root, work_dir / "crop_sam3", "sam3", None,
                        sam3_ckpt, samurai_dir, samurai_ckpt, reset, sam3_eval,
                        sam3_source_revision, crop_seqs,
                    ),
                    condition="crop_selected",
                ),
                PlanStep(
                    "merge-crop-primary", "command",
                    [
                        str(sam3_python), "-m", "hsot.crop_rerun", "merge",
                        "--base-csv", str(full_primary),
                        "--crop-csv", str(work_dir / "crop_sam3" / "submission.csv"),
                        "--meta", str(crop_meta), "--out", str(merged_primary),
                    ],
                    {"PYTHONPATH": py_path},
                    condition="crop_selected",
                ),
                PlanStep(
                    "track-crop-source-samurai", "command",
                    _track_command(
                        samurai_python, crop_root, work_dir / "crop_samurai", "samurai", None,
                        sam3_ckpt, samurai_dir, samurai_ckpt, reset, sam3_eval,
                        samurai_source_revision, crop_seqs,
                    ),
                    condition="crop_selected",
                ),
                PlanStep(
                    "merge-crop-source", "command",
                    [
                        str(samurai_python), "-m", "hsot.crop_rerun", "merge",
                        "--base-csv", str(full_source),
                        "--crop-csv", str(work_dir / "crop_samurai" / "submission.csv"),
                        "--meta", str(crop_meta), "--out", str(merged_source),
                    ],
                    {"PYTHONPATH": py_path},
                    condition="crop_selected",
                ),
                PlanStep(
                    "reject-crop-fallbacks", "internal", condition="crop_selected",
                    detail="讀兩份 crop diagnostics.json；任何 error/fallback 皆停止",
                ),
                PlanStep(
                    "no-crop-passthrough", "internal", condition="no_crop_selected",
                    detail="若新資料沒有合格小目標，原子複製兩份 full CSV 作為 merged 輸入",
                ),
            ]
            crop_diagnostics = [
                ["", str(work_dir / "crop_sam3" / "diagnostics.json")],
                ["", str(work_dir / "crop_samurai" / "diagnostics.json")],
            ]
        else:
            # ── crop 窗三方共識（D101）──────────────────────────────────────
            # 只有 SAM3 主腿走共識：D100 已把 0.0097 落差歸因到 SAM3 腿的 crop 階段，
            # SAMURAI 腿（每序列重置）v087 實測 +0.00064 ⇒ 沿用 source_leg_variant 的窗。
            # 三組窗都只用**本次 run 自己的 full 輸出**算 ⇒ 9/7 可複製、不依賴歷史檔案。
            area = profile["offline_two_pass_crop"]["area_fraction_max"]
            two_pass = work_dir / "offline_two_pass"
            source_tag = str(wc["source_leg_variant"])
            merged_variants: list[Path] = []
            for variant in wc["variants"]:
                tag = str(variant["tag"])
                v_root = two_pass / f"frames_{tag}"
                v_meta = two_pass / f"crop_meta_{tag}.json"
                v_seqs = two_pass / f"crop_seqs_{tag}.txt"
                v_track = work_dir / f"crop_sam3_{tag}"
                v_merged = two_pass / f"main_merged_{tag}.csv"
                merged_variants.append(v_merged)
                variant_paths[tag] = {
                    "crop_root": str(v_root), "crop_meta": str(v_meta),
                    "crop_seqs": str(v_seqs), "merged_primary": str(v_merged),
                }
                crop_diagnostics.append([tag, str(v_track / "diagnostics.json")])
                prep_argv = [
                    str(sam3_python), "-m", "hsot.crop_rerun", "prep",
                    "--frames-root", str(frames_root),
                    "--base-csv", str(full_primary),
                ]
                if variant["envelope_extra"]:
                    prep_argv += ["--envelope-extra", str(full_source)]
                prep_argv += [
                    "--area-frac-max", str(area),
                    "--out-root", str(v_root),
                    "--meta", str(v_meta),
                ]
                if int(variant["segments"]) != 1:
                    # segments=1 是 crop_rerun 的預設；不發旗標讓「現行窗」那組
                    # 與 08-31 前的單一窗指令逐字相同（位元級可比）。
                    prep_argv += ["--segments", str(int(variant["segments"]))]
                steps += [
                    PlanStep(
                        f"offline-two-pass-prep-{tag}", "command", prep_argv,
                        {"PYTHONPATH": py_path},
                        detail=(f"窗 {tag}：非因果，以完整未來軌跡決定固定 crop 窗"
                                f"（envelope_extra={bool(variant['envelope_extra'])}、"
                                f"segments={int(variant['segments'])}）"
                                + (f"；{variant['note']}" if variant.get("note") else "")),
                    ),
                    PlanStep(
                        f"materialize-crop-sequence-list-{tag}", "internal",
                        detail=f"從 {v_meta} 產生 {v_seqs}",
                    ),
                    PlanStep(
                        f"track-crop-primary-sam3-{tag}", "command",
                        _track_command(
                            sam3_python, v_root, v_track, "sam3", None,
                            sam3_ckpt, samurai_dir, samurai_ckpt, reset, sam3_eval,
                            sam3_source_revision, v_seqs,
                        ),
                        condition=f"crop_selected:{tag}",
                    ),
                    PlanStep(
                        f"merge-crop-primary-{tag}", "command",
                        [
                            str(sam3_python), "-m", "hsot.crop_rerun", "merge",
                            "--base-csv", str(full_primary),
                            "--crop-csv", str(v_track / "submission.csv"),
                            "--meta", str(v_meta), "--out", str(v_merged),
                        ],
                        {"PYTHONPATH": py_path},
                        condition=f"crop_selected:{tag}",
                    ),
                    PlanStep(
                        f"no-crop-passthrough-{tag}", "internal",
                        condition=f"no_crop_selected:{tag}",
                        detail=f"窗 {tag} 沒有合格小目標時，原子複製 full-SAM3 當作本組 merged 輸入",
                    ),
                ]
            steps.append(PlanStep(
                "medoid-crop-primary", "command",
                [
                    str(sam3_python), str(SRC_ROOT / "run_ensemble_medoid.py"),
                    *[str(m) for m in merged_variants],
                    "--out", str(merged_primary),
                    "--report", str(two_pass / "medoid_report.txt"),
                ],
                detail=("逐幀 medoid（與其他各框 IoU 總和最大者）；不合成新框（守 E33）。"
                        "⚠️ 第一個輸入＝現任，平手時勝出 ⇒ variants 的順序有語意，不可重排。"),
            ))
            # SAMURAI source 腿只跑 source_leg_variant 的窗（＝現行窗）。
            src_meta = Path(variant_paths[source_tag]["crop_meta"])
            src_root = Path(variant_paths[source_tag]["crop_root"])
            src_seqs = Path(variant_paths[source_tag]["crop_seqs"])
            crop_diagnostics.append([source_tag, str(work_dir / "crop_samurai" / "diagnostics.json")])
            steps += [
                PlanStep(
                    "track-crop-source-samurai", "command",
                    _track_command(
                        samurai_python, src_root, work_dir / "crop_samurai", "samurai", None,
                        sam3_ckpt, samurai_dir, samurai_ckpt, reset, sam3_eval,
                        samurai_source_revision, src_seqs,
                    ),
                    condition=f"crop_selected:{source_tag}",
                ),
                PlanStep(
                    "merge-crop-source", "command",
                    [
                        str(samurai_python), "-m", "hsot.crop_rerun", "merge",
                        "--base-csv", str(full_source),
                        "--crop-csv", str(work_dir / "crop_samurai" / "submission.csv"),
                        "--meta", str(src_meta), "--out", str(merged_source),
                    ],
                    {"PYTHONPATH": py_path},
                    condition=f"crop_selected:{source_tag}",
                ),
                PlanStep(
                    "no-crop-passthrough-source", "internal",
                    condition=f"no_crop_selected:{source_tag}",
                    detail="原子複製 full-SAMURAI 當作 source merged 輸入",
                ),
                PlanStep(
                    "reject-crop-fallbacks", "internal",
                    detail="讀所有實際跑過的 crop diagnostics.json；任何 error/fallback 皆停止",
                ),
            ]
        main_csv, source_csv = merged_primary, merged_source

    finalize = [
        str(samurai_python), str(SRC_ROOT / "finalize_submission.py"),
        "--main", str(main_csv), "--source", str(source_csv),
        "--sample", str(sample), "--out", str(pre_first),
        "--corr", str(post["coordinate_correction"]),
        "--K", str(post["frozen_splice_k"]),
        "--qhead", str(post["quality_head"]),
    ]
    if post["quality_head"] == "v056":
        finalize += ["--qhead-weights", str(qhead_weights)]
    if post.get("selector", "none") == "v2":
        # 固化權重（export_selector_weights.py 於 405 crop-merged 配對 fit）；
        # 9/7 沒有 405 快取可重 fit，只載入不訓練。
        finalize += ["--selector", "v2", "--selector-weights", str(selector_weights)]
    if post.get("third_leg_rescue", False):
        # 死區救援來源＝full-frame SAM3 腿（crop profile 下 crop 前的那一腿）。
        finalize += ["--third-leg", str(full_primary)]
    steps += [
        PlanStep("finalize", "command", finalize),
        PlanStep(
            "preserve-first-frame-and-validate", "internal",
            detail=(f"從每支 init_rect.txt 還原首幀，按 sample 順序原子寫入 {out}；"
                    "再驗 exact ID set、duplicate、finite、positive、首幀 exact init"),
        ),
    ]
    paths: dict[str, Path | bool] = {
        "full_primary": full_primary,
        "full_source": full_source,
        "crop_meta": crop_meta,
        "crop_seqs": crop_seqs,
        "merged_primary": merged_primary,
        "merged_source": merged_source,
        "pre_first": pre_first,
        "out": out,
        "use_crop": use_crop,
        "work_dir": work_dir,
        "crop_diagnostics": crop_diagnostics,
        "crop_variants": variant_paths,
        "window_consensus": window_consensus(profile) is not None and use_crop,
    }
    return steps, paths


def _sample_contract(sample: Path) -> tuple[dict[str, list[str]], list[str]]:
    errors: list[str] = []
    groups: dict[str, list[str]] = {}
    frames_by_seq: dict[str, list[tuple[int, int]]] = {}
    try:
        with sample.open(newline="", encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames != SUBMISSION_COLUMNS:
                return {}, [f"sample 欄位 {reader.fieldnames} != {SUBMISSION_COLUMNS}"]
            seen: set[str] = set()
            for row_no, row in enumerate(reader, 2):
                ident = row.get("ID", "")
                if ident in seen:
                    errors.append(f"sample 重複 ID（line {row_no}）：{ident}")
                    continue
                seen.add(ident)
                try:
                    seq, frame = ident.rsplit("_", 1)
                    int(frame)
                except (ValueError, AttributeError):
                    errors.append(f"sample ID 無法拆成 <seq>_<int>（line {row_no}）：{ident!r}")
                    continue
                groups.setdefault(seq, []).append(ident)
                frames_by_seq.setdefault(seq, []).append((int(frame), row_no))
    except (OSError, UnicodeError) as exc:
        errors.append(f"sample 讀取失敗：{exc}")
    if not groups and not errors:
        errors.append("sample 無資料列")

    # 🚨 檔內列序是隱性契約，且兩端不一致（09-04 稽核 Q02）：
    #   track_t1.py 的 rows_from_boxes 用 enumerate 把第 i 個追蹤結果綁到 sample 檔內第 i 列；
    #   finalize_submission.py 的 frozen_runs／splice／corr／第三腿救援一律用 sorted(fm)＝數字遞增。
    # 若某支序列在檔內是字典序（seq_1, seq_10, seq_100, …, seq_2）而非數字遞增，
    # preflight 與 finalize 的五道驗證會全部通過（ID 集合、唯一性、筆數、w/h>0、首幀 exact init 都對），
    # 但每一列的框都綁到錯的幀 ⇒ 交出一份自我檢查全過、時序卻整個錯位的結果。
    # Ranking B 無 GT、無 LB 回饋 ⇒ 當天不會有任何訊號。**fail closed，不自動排序**
    # （sample 若真的用非 1-based 全域 frame ID，自動排序反而會做錯，見 track_t1.py:80-83）。
    for seq, pairs in sorted(frames_by_seq.items()):
        for (a, _), (b, line_b) in zip(pairs, pairs[1:]):
            if b <= a:
                errors.append(
                    f"sample 序列 {seq} 的 frame ID 未嚴格遞增（line {line_b}：{a} → {b}）"
                    "——檔內列序即為 tracker 的輸出順序，非遞增會讓每一列綁到錯的幀")
                break
    return groups, errors


def _resolve_executable(value: Path) -> Path | None:
    raw = str(value)
    if value.is_file():
        return Path(os.path.abspath(raw))
    found = shutil.which(raw)
    return Path(os.path.abspath(found)) if found else None


def _module_probe(python: Path, modules: Iterable[str]) -> tuple[bool, str]:
    names = list(modules)
    code = (
        "import importlib.util,json; "
        f"m={json.dumps(names)}; "
        "missing=[x for x in m if importlib.util.find_spec(x) is None]; "
        "print(','.join(missing)); raise SystemExit(bool(missing))"
    )
    try:
        proc = subprocess.run(
            [str(python), "-c", code], text=True, capture_output=True, timeout=15,
            cwd=str(REPO_ROOT), check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    missing = proc.stdout.strip() or proc.stderr.strip()
    return proc.returncode == 0, missing


def _source_has_flags(path: Path, flags: Iterable[str]) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return list(flags)
    return [flag for flag in flags if flag not in text]


def preflight(
    profile_name: str,
    profile: dict[str, Any],
    *,
    frames_root: Path,
    sample: Path,
    work_dir: Path,
    out: Path,
    sam3_python: Path,
    samurai_python: Path,
    sam3_ckpt: Path,
    samurai_dir: Path,
    samurai_ckpt: Path,
    qhead_weights: Path,
    selector_weights: Path,
    allow_offline_two_pass: bool,
    execute: bool,
    resume_existing_work: bool,
    sam3_source_revision: str = "",
    samurai_source_revision: str = "",
) -> list[Finding]:
    findings: list[Finding] = []

    def add(level: str, check: str, detail: str) -> None:
        findings.append(Finding(level, check, detail))

    intent = profile["offline_two_pass_crop"]["profile_intent"]
    if intent == "enabled" and not allow_offline_two_pass:
        add("BLOCK", "offline-two-pass-gate",
            f"{profile_name} 的 profile intent=enabled，但未顯式傳 --allow-offline-two-pass")
    elif intent == "disabled" and allow_offline_two_pass:
        # 09-10：這裡原本只是 WARN，但實測旗標會把 crop 打開（計畫 5 步 → 13 步）
        # ⇒ 「嚴格因果的對照檔」會靜默變成兩趟版本，而那正是它存在的唯一理由。
        # 交付給主辦方時，順手帶上旗標是完全可能的（主檔就要帶）⇒ 改成 fail-closed。
        add("BLOCK", "offline-two-pass-gate",
            f"{profile_name} 的 profile intent=disabled（嚴格單趟因果），"
            "不可與 --allow-offline-two-pass 併用——該旗標會啟用 crop 兩趟路徑，"
            "使這個 profile 失去它唯一的用途。要兩趟請改用 rankB_deliver_v090。")
    elif allow_offline_two_pass:
        add("WARN", "offline-two-pass-gate", "已顯式授權；會使用完整未來預測軌跡回套重跑")
    else:
        add("OK", "offline-two-pass-gate", "未啟用；維持因果 full-frame 管線")

    scripts = [SRC_ROOT / "track_t1.py", SRC_ROOT / "finalize_submission.py"]
    wc = window_consensus(profile) if effective_two_pass(profile, allow_offline_two_pass) else None
    if allow_offline_two_pass:
        scripts.append(SRC_ROOT / "hsot" / "crop_rerun.py")
    if wc is not None:
        scripts.append(SRC_ROOT / "run_ensemble_medoid.py")
    missing_scripts = [str(p) for p in scripts if not p.is_file()]
    add("BLOCK" if missing_scripts else "OK", "entrypoint-files",
        f"缺少：{missing_scripts}" if missing_scripts else "track/finalize 所需程式存在")

    track_missing = _source_has_flags(
        SRC_ROOT / "track_t1.py",
        ["--frames-root", "--out-dir", "--backend", "--sample-csv", "--samurai-reset-kf",
         "--samurai-legacy-cross-seq-kf", "--sam3-eval", "--source-revision"],
    )
    final_missing = _source_has_flags(
        SRC_ROOT / "finalize_submission.py",
        ["--main", "--source", "--sample", "--out", "--corr", "--K", "--qhead"],
    )
    missing_iface = track_missing + final_missing
    add("BLOCK" if missing_iface else "OK", "cli-interface",
        f"現有腳本缺 runner mapping 需要的旗標：{missing_iface}" if missing_iface
        else "semantic keys 可映射到現有 track/finalize CLI；首幀由 runner 自行保護")

    if wc is not None:
        tags = [str(v["tag"]) for v in wc["variants"]]
        add("WARN", "crop-window-consensus",
            f"窗共識啟用：{'／'.join(tags)}（第一組＝medoid 平手時的現任，順序有語意）；"
            f"SAMURAI source 腿用窗 {wc['source_leg_variant']}；多兩組 crop-SAM3 ≈ +80 分鐘")
    else:
        add("OK", "crop-window-consensus", "單一 crop 窗（08-31 前行為）")

    missing_revisions = [name for name, value in (
        ("sam3", sam3_source_revision), ("samurai", samurai_source_revision)) if not value]
    add("WARN" if missing_revisions else "OK", "source-revisions",
        (f"未固定 {missing_revisions} source revision；不可把產物當跨機可 resume 正式 run"
         if missing_revisions else
         f"SAM3={sam3_source_revision}；SAMURAI={samurai_source_revision}"))

    groups: dict[str, list[str]] = {}
    if not sample.is_file():
        add("BLOCK", "sample-submission", f"必填 sample 不存在：{sample}")
    else:
        groups, errors = _sample_contract(sample)
        add("BLOCK" if errors else "OK", "sample-submission",
            "; ".join(errors[:5]) if errors else
            f"schema/ID 唯一性通過：{sum(map(len, groups.values()))} 列、{len(groups)} 支序列")

    # 09-07：官方 ranking/ 佈局把模態放在上層資料夾名（HSI-NIR-Falsecolor/<seq>/），序列目錄本身
    # 沒有 nir-/rednir-/vis- 前綴。finalize 的 quality_head_v1.modality() 對無前綴名一律回 VIS
    # ⇒ 若主辦方把 --frames-root 直接指到官方資料夾（跳過 prep），所有檢查照過、
    #    RedNIR 品質頭永不觸發、兩顆頭的模態 one-hot 全錯，而且沒有任何訊號。這是最後一條靜默跑錯的路。
    if groups:
        prefixed = [s for s in groups if s.split("-", 1)[0] in ("nir", "rednir", "vis") and "-" in s]
        if not prefixed:
            add("BLOCK", "modality-prefix",
                f"{len(groups)} 支序列沒有任何一支帶 nir-/rednir-/vis- 前綴（例 {sorted(groups)[:3]}）"
                "——後處理靠前綴判模態，缺前綴會全部當 VIS。請用 prep_rankingb_frames_v1.py 從官方"
                "模態資料夾產生 frames-root（它會自動加前綴），不要直接指向官方資料夾")
        elif len(prefixed) != len(groups):
            add("WARN", "modality-prefix",
                f"{len(groups) - len(prefixed)} 支序列缺模態前綴，會被當 VIS："
                f"{sorted(set(groups) - set(prefixed))[:3]}")
        else:
            add("OK", "modality-prefix", f"{len(groups)} 支序列皆帶模態前綴")

    if not frames_root.is_dir():
        add("BLOCK", "frames-root", f"解包後資料目錄不存在：{frames_root}")
    elif groups:
        missing_seq: list[str] = []
        missing_init: list[str] = []
        count_diff: list[str] = []
        bad_init: list[str] = []
        bad_stem: list[str] = []
        for seq, ids in groups.items():
            seq_dir = frames_root / seq
            if not seq_dir.is_dir():
                missing_seq.append(seq)
                continue
            init_path = seq_dir / "init_rect.txt"
            if not init_path.is_file():
                missing_init.append(seq)
            frame_paths = sorted(seq_dir.glob("*.jpg")) + sorted(seq_dir.glob("*.jpeg"))
            n_frames = len(frame_paths)
            if n_frames != len(ids):
                count_diff.append(f"{seq}:{n_frames}!={len(ids)}")
            # 09-05 稽核 G15：init_rect 的座標慣例是**唯一會燒掉整輪 4.5 小時的失敗型態**
            # ——所有既有檢查（序列數、init 存在、幀數）都會通過，推論照跑，交出來的東西全錯。
            # 無 GT 可比，但「框必須落在影像內」這條與慣例無關、且能抓到 (x1,y1,x2,y2) 被
            # 誤當成 (x,y,w,h)：那時 x+w 幾乎必然超出影像寬度。
            if init_path.is_file() and frame_paths:
                try:
                    x, y, w, h = _read_init(init_path)
                except RuntimeError as exc:
                    bad_init.append(f"{seq}:{exc}")
                else:
                    dims = _image_size(frame_paths[0])
                    if dims is not None:
                        iw, ih = dims
                        if x + w > iw + 1 or y + h > ih + 1 or x < -1 or y < -1:
                            bad_init.append(
                                f"{seq}: init_rect ({x:g},{y:g},{w:g},{h:g}) 超出影像 {iw}x{ih}"
                                "——最可能是格式被當成 (x,y,w,h) 但實際是 (x1,y1,x2,y2)")
            # 檔名 stem 必須可 int()：track_t1 依排序後的位置對應 sample 列序，
            # 若命名是 frame_001a.jpg 之類，排序結果與數字序不同 ⇒ 同 Q02 的錯位問題。
            for fp in frame_paths[:3]:
                try:
                    int(fp.stem)
                except ValueError:
                    bad_stem.append(f"{seq}/{fp.name}")
                    break
        problems = []
        if missing_seq:
            problems.append(f"缺序列 {len(missing_seq)}，例 {missing_seq[:3]}")
        if missing_init:
            problems.append(f"缺 init_rect {len(missing_init)}，例 {missing_init[:3]}")
        if count_diff:
            problems.append(f"幀數不符 {len(count_diff)}，例 {count_diff[:3]}")
        if bad_init:
            problems.append(f"init_rect 座標異常 {len(bad_init)}，例 {bad_init[:2]}")
        if bad_stem:
            problems.append(f"幀檔名 stem 非純數字 {len(bad_stem)}，例 {bad_stem[:3]}"
                            "（排序順序會與數字序不同）")
        add("BLOCK" if problems else "OK", "frames-root",
            "；".join(problems) if problems else f"{len(groups)} 支資料、init、幀數皆與 sample 對齊")
    else:
        add("WARN", "frames-root", "目錄存在，但 sample 無有效序列可做逐支對齊")

    for label, raw, modules in (
        ("sam3-python", sam3_python, ("numpy", "pandas", "torch", "sam3")),
        ("samurai-python", samurai_python, ("numpy", "pandas", "torch", "sam2")),
    ):
        exe = _resolve_executable(raw)
        if exe is None:
            add("BLOCK", label, f"Python executable 不存在：{raw}")
            continue
        ok, detail = _module_probe(exe, modules)
        add("OK" if ok else "BLOCK", label,
            f"{exe}：模組探針通過" if ok else f"{exe}：缺模組或探針失敗 {detail!r}")

    for label, path in (
        ("sam3-checkpoint", sam3_ckpt),
        ("samurai-checkpoint", samurai_ckpt),
    ):
        add("OK" if path.is_file() and path.stat().st_size > 0 else "BLOCK", label,
            f"存在：{path}" if path.is_file() and path.stat().st_size > 0 else f"不存在/空檔：{path}")
    samurai_ok = samurai_dir.is_dir() and (samurai_dir / "sam2").is_dir()
    add("OK" if samurai_ok else "BLOCK", "samurai-source",
        f"存在：{samurai_dir}" if samurai_ok else f"需含 sam2/ 的 SAMURAI clone：{samurai_dir}")
    if profile["postprocess"]["quality_head"] == "v056":
        add("OK" if qhead_weights.is_file() else "BLOCK", "qhead-weights",
            f"存在：{qhead_weights}" if qhead_weights.is_file() else f"v056 權重不存在：{qhead_weights}")
    if profile["postprocess"].get("selector", "none") == "v2":
        add("OK" if selector_weights.is_file() else "BLOCK", "selector-weights",
            f"存在：{selector_weights}" if selector_weights.is_file()
            else f"selector-v2 權重不存在：{selector_weights}（跑 export_selector_weights.py）")

    work_is_file = work_dir.exists() and not work_dir.is_dir()
    occupied = work_dir.is_dir() and any(work_dir.iterdir())
    out_exists = out.exists()
    if work_is_file:
        add("BLOCK", "output-collision", f"work-dir 是檔案而非目錄：{work_dir}")
    elif occupied or out_exists:
        level = "WARN" if resume_existing_work else ("BLOCK" if execute else "WARN")
        add(level, "output-collision",
            f"work/output 已存在（work_nonempty={occupied}, out_exists={out_exists}）；"
            + ("已顯式 --resume-existing-work，仍會逐段驗證" if resume_existing_work
               else "execute 時請換 --work-dir/--out，或明示 --resume-existing-work"))
    else:
        add("OK", "output-collision", "不會覆寫既有 run 產物")
    return findings


def _image_size(path: Path) -> tuple[int, int] | None:
    """只讀 JPEG header 取寬高。本入口刻意只依賴標準庫（無 Pillow），讀不出來就回 None。"""
    try:
        with path.open("rb") as fh:
            if fh.read(2) != b"\xff\xd8":
                return None
            while True:
                b = fh.read(1)
                while b and b != b"\xff":
                    b = fh.read(1)
                marker = fh.read(1)
                while marker == b"\xff":
                    marker = fh.read(1)
                if not marker:
                    return None
                if marker[0] in (0xD8, 0xD9) or 0xD0 <= marker[0] <= 0xD7:
                    continue
                seg = fh.read(2)
                if len(seg) < 2:
                    return None
                length = int.from_bytes(seg, "big")
                if marker[0] in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                                 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                    body = fh.read(5)
                    if len(body) < 5:
                        return None
                    height = int.from_bytes(body[1:3], "big")
                    width = int.from_bytes(body[3:5], "big")
                    return width, height
                fh.seek(length - 2, 1)
    except OSError:
        return None


def _read_init(path: Path) -> list[float]:
    try:
        tokens = [x for x in re.split(r"[,\s]+", path.read_text().strip()) if x]
        values = [float(x) for x in tokens[:4]]
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"init_rect 無法讀取：{path}: {exc}") from exc
    if len(values) != 4 or not all(math.isfinite(v) for v in values):
        raise RuntimeError(f"init_rect 必須含四個 finite 數值：{path}")
    if values[2] <= 0 or values[3] <= 0:
        raise RuntimeError(f"init_rect width/height 必須 >0：{path}: {values}")
    return values


def _read_unique_submission(
    path: Path, *, validate_boxes: bool = True,
) -> tuple[list[dict[str, str]], list[str]]:
    """讀取 unique ID CSV。

    官方 sample 的座標欄只是零值 placeholder，不能套 production box 的 w/h>0 驗證；
    但仍要求標準欄位與 unique ID。這與 finalize_submission.load(...,
    validate_boxes=False) 的角色分流一致。
    """
    errors: list[str] = []
    rows: list[dict[str, str]] = []
    try:
        with path.open(newline="", encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames != SUBMISSION_COLUMNS:
                return [], [f"欄位 {reader.fieldnames} != {SUBMISSION_COLUMNS}"]
            seen: set[str] = set()
            for line, row in enumerate(reader, 2):
                ident = row["ID"]
                if ident in seen:
                    errors.append(f"重複 ID line {line}: {ident}")
                seen.add(ident)
                for key in SUBMISSION_COLUMNS[1:]:
                    try:
                        value = float(row[key])
                    except (TypeError, ValueError):
                        errors.append(f"非數值 {ident}.{key}={row[key]!r}")
                        continue
                    if validate_boxes and not math.isfinite(value):
                        errors.append(f"非 finite {ident}.{key}={row[key]!r}")
                    if validate_boxes and key in {"x", "y"} and value < 0:
                        errors.append(f"負座標 {ident}.{key}={row[key]!r}")
                    if validate_boxes and key in {"width", "height"} and value <= 0:
                        errors.append(f"非正 {ident}.{key}={row[key]!r}")
                rows.append(dict(row))
    except OSError as exc:
        errors.append(f"讀取失敗：{path}: {exc}")
    return rows, errors


def _fmt_number(value: float) -> str:
    return format(value, ".15g")


def preserve_first_frames_and_validate(
    source_csv: Path,
    out_csv: Path,
    sample_csv: Path,
    frames_root: Path,
) -> dict[str, int]:
    """runner-native 首幀護欄；不依賴 finalize/track 未定案的旗標名稱。"""
    sample_rows, sample_errors = _read_unique_submission(sample_csv, validate_boxes=False)
    pred_rows, pred_errors = _read_unique_submission(source_csv)
    errors = sample_errors + pred_errors
    sample_ids = [row["ID"] for row in sample_rows]
    pred_by_id = {row["ID"]: row for row in pred_rows}
    if set(sample_ids) != set(pred_by_id):
        missing = sorted(set(sample_ids) - set(pred_by_id))
        extra = sorted(set(pred_by_id) - set(sample_ids))
        errors.append(f"exact ID set 不符：缺 {len(missing)} {missing[:3]}；多 {len(extra)} {extra[:3]}")
    if errors:
        raise RuntimeError("final CSV 驗證失敗：\n  - " + "\n  - ".join(errors[:20]))

    first_by_seq: dict[str, tuple[int, str]] = {}
    for ident in sample_ids:
        seq, frame_raw = ident.rsplit("_", 1)
        frame = int(frame_raw)
        if seq not in first_by_seq or frame < first_by_seq[seq][0]:
            first_by_seq[seq] = (frame, ident)
    restored = 0
    for seq, (_, ident) in first_by_seq.items():
        init = _read_init(frames_root / seq / "init_rect.txt")
        row = pred_by_id[ident]
        for key, value in zip(SUBMISSION_COLUMNS[1:], init):
            row[key] = _fmt_number(value)
        restored += 1

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=out_csv.name + ".", suffix=".part", dir=out_csv.parent)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=SUBMISSION_COLUMNS, lineterminator="\n")
            writer.writeheader()
            for ident in sample_ids:
                writer.writerow(pred_by_id[ident])
        os.replace(tmp_name, out_csv)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise

    final_rows, final_errors = _read_unique_submission(out_csv)
    if final_errors or [r["ID"] for r in final_rows] != sample_ids:
        raise RuntimeError("原子寫入後驗證失敗：" + "; ".join(final_errors))
    final_by_id = {r["ID"]: r for r in final_rows}
    for seq, (_, ident) in first_by_seq.items():
        init = _read_init(frames_root / seq / "init_rect.txt")
        got = [float(final_by_id[ident][k]) for k in SUBMISSION_COLUMNS[1:]]
        if got != init:
            raise RuntimeError(f"{seq} 首幀未 exact preserve：got={got}, init={init}")
    return {"rows": len(final_rows), "sequences": len(first_by_seq), "restored": restored}


def _reject_diagnostic_fallbacks(paths: Iterable[Path]) -> None:
    failures: list[str] = []
    for path in paths:
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            failures.append(f"diagnostics 不存在/無效：{path}: {exc}")
            continue
        for seq, diag in doc.items():
            if seq == "_meta" or not isinstance(diag, dict):
                continue
            if "error" in diag or "fallback" in diag or "fallback_error" in diag:
                failures.append(f"{path.parent.name}/{seq}: {diag}")
    if failures:
        raise RuntimeError("tracker 有 error/fallback，依 profile 拒絕交付：\n  - " + "\n  - ".join(failures[:20]))


def _atomic_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=target.name + ".", suffix=".part", dir=target.parent)
    os.close(fd)
    try:
        shutil.copyfile(source, tmp_name)
        os.replace(tmp_name, target)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _run_command(step: PlanStep) -> None:
    env = os.environ.copy()
    env.update(step.env)
    print(f"\n>>> {step.name}\n{render_command(step)}", flush=True)
    subprocess.run(step.argv, env=env, cwd=str(REPO_ROOT), check=True)


def execute_plan(
    steps: list[PlanStep], paths: dict[str, Path | bool], *,
    profile: dict[str, Any], sample: Path, frames_root: Path,
) -> None:
    work_dir = Path(paths["work_dir"])
    work_dir.mkdir(parents=True, exist_ok=True)
    # tag ""＝單一窗（08-31 前的行為）；窗共識時每組窗各有自己的 tag 與選中狀態。
    crop_selected: dict[str, bool] = {}
    for step in steps:
        kind, _, cond_tag = step.condition.partition(":")
        if kind in {"crop_selected", "no_crop_selected"}:
            state = crop_selected.get(cond_tag)
            want = kind == "crop_selected"
            if state is not want:
                print(f"SKIP {step.name}（crop_selected[{cond_tag!r}]={state}）")
                continue
        if step.kind == "command":
            _run_command(step)
            continue
        print(f"\n>>> {step.name}\n{step.detail}", flush=True)
        if step.name == "reject-base-fallbacks":
            if profile["validation"]["reject_tracker_fallback"]:
                _reject_diagnostic_fallbacks([
                    Path(paths["full_primary"]).parent / "diagnostics.json",
                    Path(paths["full_source"]).parent / "diagnostics.json",
                ])
        elif step.name.startswith("materialize-crop-sequence-list"):
            tag = step.name[len("materialize-crop-sequence-list"):].lstrip("-")
            slot = paths["crop_variants"][tag] if tag else paths  # type: ignore[index]
            meta_path = Path(slot["crop_meta"])
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if not isinstance(meta, dict):
                raise RuntimeError(f"crop meta 不是 object：{meta_path}")
            seqs = sorted(meta)
            seq_path = Path(slot["crop_seqs"])
            seq_path.parent.mkdir(parents=True, exist_ok=True)
            seq_path.write_text("".join(f"{seq}\n" for seq in seqs), encoding="utf-8")
            crop_selected[tag] = bool(seqs)
            print(f"crop{f' 窗 {tag}' if tag else ''} 選中 {len(seqs)} 支")
        elif step.name == "reject-crop-fallbacks":
            if profile["validation"]["reject_tracker_fallback"]:
                # 只檢查「真的跑過」的那些腿——沒跑的腿不會有 diagnostics.json，
                # 而 _reject_diagnostic_fallbacks 把檔案不存在也算失敗。
                _reject_diagnostic_fallbacks([
                    Path(path) for tag, path in paths["crop_diagnostics"]  # type: ignore[union-attr]
                    if crop_selected.get(tag) is True
                ])
        elif step.name == "no-crop-passthrough":
            _atomic_copy(Path(paths["full_primary"]), Path(paths["merged_primary"]))
            _atomic_copy(Path(paths["full_source"]), Path(paths["merged_source"]))
        elif step.name == "no-crop-passthrough-source":
            _atomic_copy(Path(paths["full_source"]), Path(paths["merged_source"]))
        elif step.name.startswith("no-crop-passthrough-"):
            tag = step.name[len("no-crop-passthrough-"):]
            _atomic_copy(Path(paths["full_primary"]),
                         Path(paths["crop_variants"][tag]["merged_primary"]))  # type: ignore[index]
        elif step.name == "preserve-first-frame-and-validate":
            stats = preserve_first_frames_and_validate(
                Path(paths["pre_first"]), Path(paths["out"]), sample, frames_root,
            )
            print(f"final 驗證通過：{stats['rows']} 列、{stats['sequences']} 支；"
                  f"首幀還原 {stats['restored']} 支 → {paths['out']}")
        else:
            raise RuntimeError(f"未知 internal step：{step.name}")


def render_command(step: PlanStep) -> str:
    prefix = ""
    if step.env:
        prefix = " ".join(f"{k}={shlex.quote(v)}" for k, v in sorted(step.env.items())) + " "
    return prefix + shlex.join(step.argv)


def print_human_plan(
    profile_name: str,
    profile: dict[str, Any],
    findings: list[Finding],
    steps: list[PlanStep],
    execute: bool,
) -> None:
    print(f"Profile: {profile_name}")
    print(profile.get("description", ""))
    post = profile["postprocess"]
    print("Semantic policy: "
        f"qhead={post['quality_head']} | corr={post['coordinate_correction']} | "
        f"first_frame={post['first_frame_policy']} | "
        f"samurai_reset={profile['tracking']['samurai_reset_between_sequences']} | "
        f"sam3_eval={profile['tracking']['sam3_eval']} | "
        f"selector={post.get('selector', 'none')} | "
        f"third_leg={post.get('third_leg_rescue', False)}")
    wc = profile["offline_two_pass_crop"].get("window_consensus")
    if isinstance(wc, dict) and wc.get("enabled"):
        print("Crop window consensus: "
              + "／".join(
                  f"{v['tag']}(extra={bool(v['envelope_extra'])},segments={int(v['segments'])})"
                  for v in wc["variants"])
              + f" → 逐幀 medoid｜source 腿用窗 {wc['source_leg_variant']}"
              + "（僅在 --allow-offline-two-pass 時生效）")
    print("\nPreflight:")
    for finding in findings:
        print(f"  [{finding.level:<5}] {finding.check}: {finding.detail}")
    print("\nCommand plan:")
    for index, step in enumerate(steps, 1):
        cond = "" if step.condition == "always" else f" if={step.condition}"
        print(f"  {index:02d}. {step.name} [{step.kind}{cond}]")
        if step.kind == "command":
            print(f"      {render_command(step)}")
        elif step.detail:
            print(f"      {step.detail}")
    blocked = sum(f.level == "BLOCK" for f in findings)
    if execute:
        print(f"\nExecution requested: {'BLOCKED' if blocked else 'READY'}")
    else:
        print(f"\nDRY-RUN ONLY（BLOCK={blocked}）；未執行推論、未建立產物。前置檢查全過後加 --execute。")


def _default_path(env_name: str, fallback: Path) -> str:
    return os.environ.get(env_name, str(fallback))


def make_parser(default_profile: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG), help="ranking_profiles.json")
    ap.add_argument("--profile", default=default_profile, help="rankB_robust（預設）或 rankA_best")
    ap.add_argument("--list-profiles", action="store_true")
    ap.add_argument("--frames-root", default=str(REPO_ROOT / "1_data/processed/test_fc"))
    ap.add_argument("--sample", default=str(REPO_ROOT / "1_data/raw/sample_submisson.csv"),
                    help="必填語意；預設指向 repo 內官方 sample")
    ap.add_argument("--work-dir", help="預設 5_outputs/ranking_runs/<profile>")
    ap.add_argument("--out", help="最終 submission；預設 <work-dir>/final.csv")
    ap.add_argument("--sam3-python", default=_default_path("SAM3_PYTHON", Path(sys.executable)))
    ap.add_argument("--samurai-python", default=_default_path("SAMURAI_PYTHON", Path(sys.executable)))
    ap.add_argument("--sam3-ckpt", default=_default_path(
        "SAM3_CKPT", REPO_ROOT / "4_models/pretrained/sam3.pt"))
    ap.add_argument("--samurai-dir", default=_default_path(
        "SAMURAI_DIR", REPO_ROOT / "4_models/vendor/samurai"))
    ap.add_argument("--samurai-ckpt", default=_default_path(
        "SAMURAI_CKPT", REPO_ROOT / "4_models/pretrained/sam2.1_hiera_large.pt"))
    ap.add_argument("--sam3-source-revision", default="",
                    help="SAM3 git commit/revision；正式跨機 resume run 應明示")
    ap.add_argument("--samurai-source-revision", default="",
                    help="SAMURAI git commit/revision；正式跨機 resume run 應明示")
    ap.add_argument("--qhead-weights", default=str(SRC_ROOT / "hsot/qhead_weights_v056.npz"))
    ap.add_argument("--selector-weights", default=str(SRC_ROOT / "hsot/selector_weights_v2.npz"),
                    help="selector-v2 固化權重（D085）；由 export_selector_weights.py 產生")
    ap.add_argument("--allow-offline-two-pass", action="store_true",
                    help="顯式承擔 OPE/因果合規風險並啟用 two-pass crop；沒有此旗標永不啟用")
    ap.add_argument("--execute", action="store_true",
                    help="前置檢查全部通過才執行；預設永遠只是 dry-run")
    ap.add_argument("--resume-existing-work", action="store_true",
                    help="顯式允許沿用非空 work-dir；不會刪檔，仍檢查 diagnostics/final CSV")
    ap.add_argument("--format", choices=("human", "json"), default="human")
    return ap


def main(argv: list[str] | None = None) -> int:
    # 先只讀 --config，讓自訂設定檔也能提供 default_profile。
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", default=str(DEFAULT_CONFIG))
    bootstrap_args, _ = bootstrap.parse_known_args(argv)
    try:
        config_path = _path(bootstrap_args.config)
        doc = load_profiles(config_path)
    except ValueError as exc:
        print(f"配置錯誤：{exc}", file=sys.stderr)
        return 2
    parser = make_parser(doc["default_profile"])
    args = parser.parse_args(argv)
    # parser 已接受的 --config 必須與 bootstrap 一致；重新載入避免相對路徑語意含糊。
    try:
        doc = load_profiles(_path(args.config))
    except ValueError as exc:
        parser.error(str(exc))
    if args.list_profiles:
        for name, profile in doc["profiles"].items():
            print(f"{name}: {profile.get('description', '')}")
        return 0
    if args.profile not in doc["profiles"]:
        parser.error(f"未知 profile {args.profile!r}；可用：{', '.join(doc['profiles'])}")
    profile = doc["profiles"][args.profile]
    frames_root = _path(args.frames_root)
    sample = _path(args.sample)
    work_dir = _path(args.work_dir or REPO_ROOT / "5_outputs/ranking_runs" / args.profile)
    out = _path(args.out or work_dir / "final.csv")
    kwargs = {
        "frames_root": frames_root,
        "sample": sample,
        "work_dir": work_dir,
        "out": out,
        "sam3_python": _executable_arg(args.sam3_python),
        "samurai_python": _executable_arg(args.samurai_python),
        "sam3_ckpt": _path(args.sam3_ckpt),
        "samurai_dir": _path(args.samurai_dir),
        "samurai_ckpt": _path(args.samurai_ckpt),
        "qhead_weights": _path(args.qhead_weights),
        "selector_weights": _path(args.selector_weights),
        "allow_offline_two_pass": bool(args.allow_offline_two_pass),
        "sam3_source_revision": args.sam3_source_revision,
        "samurai_source_revision": args.samurai_source_revision,
    }
    steps, paths = build_plan(args.profile, profile, **kwargs)
    findings = preflight(
        args.profile, profile, **kwargs,
        execute=bool(args.execute), resume_existing_work=bool(args.resume_existing_work),
    )
    if args.format == "json":
        payload = {
            "profile": args.profile,
            "semantic_profile": profile,
            "dry_run": not args.execute,
            "execution_ready": not any(f.level == "BLOCK" for f in findings),
            "findings": [asdict(f) for f in findings],
            "steps": [asdict(s) for s in steps],
            "paths": {k: str(v) if isinstance(v, Path) else v for k, v in paths.items()},
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print_human_plan(args.profile, profile, findings, steps, bool(args.execute))
    if not args.execute:
        return 0
    blocked = [f for f in findings if f.level == "BLOCK"]
    if blocked:
        print("\n拒絕執行：前置檢查有 BLOCK。", file=sys.stderr)
        return 2
    try:
        execute_plan(steps, paths, profile=profile, sample=sample, frames_root=frames_root)
    except (OSError, RuntimeError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        print(f"\n執行失敗（未宣告完成）：{exc}", file=sys.stderr)
        return 2
    print(f"\n完成：{out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
