#!/usr/bin/env python3
"""G2 訓練包（merge_ckpt / smoke_compare / train_g2 純函式）CPU 單元測試。

torch 相關測試以 importorskip＋stub 防護（同 test_train_tracker_g1.py）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from train_tracker_g1 import clip_dataset as cd  # noqa: E402
from train_tracker_g1 import smoke_compare as sc  # noqa: E402
from train_tracker_g1 import train_g2 as g2  # noqa: E402


# ─── 純 python：fold 切分與 usable 規則 ───────────────────────────────

def _mk_clip(seq, fold=0, min_iou=0.1, failing=8, start=0, length=16):
    ids = [f"{seq}_{100 + start + i}" for i in range(length)]
    return {"sequence": seq, "modality": seq.split("-")[0], "start_position": start,
            "length": length, "frame_ids": ids, "mean_iou": 0.3,
            "min_iou": min_iou, "failing_frames": failing, "frozen_frames": 0,
            "fold": fold}


def _mk_gt(clips, invalid_ids=()):
    gt = {}
    for c in clips:
        for fid in c["frame_ids"]:
            gt[fid] = (0.0, 0.0, 0.0, 0.0) if fid in invalid_ids else (5.0, 5.0, 10.0, 10.0)
    return gt


def test_parse_folds_and_range_check():
    assert g2.parse_folds("0,1,2,3") == {0, 1, 2, 3}
    with pytest.raises(AssertionError):
        g2.parse_folds("4,5")


def test_split_by_folds_disjoint_and_identity_kept():
    clips = [_mk_clip("vis-a", fold=0),
             _mk_clip("vis-id", fold=1, min_iou=0.9),  # identity：G2 要保留
             _mk_clip("rednir-h", fold=4)]
    gt = _mk_gt(clips)
    train, hold = g2.split_by_folds(clips, gt, {0, 1}, {4}, t=8, stride=2)
    assert {c["sequence"] for c in train} == {"vis-a", "vis-id"}
    assert [c["sequence"] for c in hold] == ["rednir-h"]
    with pytest.raises(AssertionError):
        g2.split_by_folds(clips, gt, {0, 1}, {1, 4}, t=8, stride=2)  # 相交


def test_split_excludes_unusable_first_frame():
    bad = _mk_clip("vis-bad", fold=0)
    ok = _mk_clip("vis-ok", fold=0)
    hold = _mk_clip("rednir-h", fold=4)
    gt = _mk_gt([bad, ok, hold], invalid_ids={bad["frame_ids"][0]})
    train, _ = g2.split_by_folds([bad, ok, hold], gt, {0}, {4}, t=8, stride=2)
    assert [c["sequence"] for c in train] == ["vis-ok"]


def test_sample_holdout_deterministic_and_capped():
    clips = [_mk_clip(f"vis-s{i:03d}", fold=4, start=0) for i in range(30)]
    a = g2.sample_holdout(clips, seed=42, cap=10)
    b = g2.sample_holdout(clips, seed=42, cap=10)
    assert len(a) == 10 and [c["sequence"] for c in a] == [c["sequence"] for c in b]
    assert g2.sample_holdout(clips, seed=42, cap=100) == sorted(
        clips, key=lambda c: (c["sequence"], c["start_position"]))


def _mk_smoke_world(tmp_path, seqs, n_frames=20, gt_rows=None, n_valid=None):
    """假 405 目錄＋假 GT：每序列 n_frames 張空 jpg、GT 行數/有效數可各別覆寫。"""
    gt = {}
    for s in seqs:
        (tmp_path / s).mkdir(exist_ok=True)
        for i in range(n_frames):
            (tmp_path / s / f"{i:05d}.jpg").touch()
        rows = (gt_rows or {}).get(s, n_frames)
        valid = (n_valid or {}).get(s, rows)
        for i in range(rows):
            gt[f"{s}_{i + 1}"] = ((5.0, 5.0, 10.0, 10.0) if i < valid
                                  else (0.0, 0.0, 0.0, 0.0))
    return gt


def test_build_smoke_plan_balanced_modality(tmp_path):
    seqs = ("vis-a", "vis-h1", "vis-h2", "rednir-a", "rednir-h1", "rednir-h2")
    gt = _mk_smoke_world(tmp_path, seqs)
    clips = [_mk_clip("vis-a", failing=9), _mk_clip("rednir-a", failing=5)]
    plan = g2.build_smoke_plan(clips, tmp_path, gt, n_failing=2)
    assert plan["failing"] == ["vis-a", "rednir-a"]  # failing 總和降冪
    # 健康＝2 vis＋1 rednir（modality 平衡是災難閘判準的一部分）
    assert plan["healthy"] == ["vis-h1", "vis-h2", "rednir-h1"]
    assert plan["all"] == plan["failing"] + plan["healthy"]
    assert plan["substituted"] == []


def test_build_smoke_plan_substitutes_on_gt_mismatch(tmp_path):
    seqs = ("vis-a", "vis-b", "vis-h1", "vis-h2", "vis-h3", "rednir-h1")
    # vis-a：GT 多 5 行（duplicate block 型）；vis-h1：有效 GT 不足
    gt = _mk_smoke_world(tmp_path, seqs,
                         gt_rows={"vis-a": 25},
                         n_valid={"vis-h1": g2.SMOKE_MIN_VALID_GT - 1})
    clips = [_mk_clip("vis-a", failing=9), _mk_clip("vis-b", failing=5)]
    plan = g2.build_smoke_plan(clips, tmp_path, gt, n_failing=1)
    assert plan["failing"] == ["vis-b"]  # vis-a 行數不合 → 順位替補
    assert plan["healthy"] == ["vis-h2", "vis-h3", "rednir-h1"]  # vis-h1 被替補
    reasons = {s["sequence"]: s["reason"] for s in plan["substituted"]}
    assert "vis-a" in reasons and "gt_rows=25" in reasons["vis-a"]
    assert "vis-h1" in reasons and "valid_gt" in reasons["vis-h1"]


def test_build_smoke_plan_fails_closed_when_short(tmp_path):
    seqs = ("vis-a", "vis-h1", "rednir-h1")
    gt = _mk_smoke_world(tmp_path, seqs, gt_rows={"vis-h1": 3})
    clips = [_mk_clip("vis-a", failing=9)]
    with pytest.raises(AssertionError, match="合格候選不足"):
        g2.build_smoke_plan(clips, tmp_path, gt, n_failing=1)  # 健康 vis 湊不滿 2


def test_decide_g2_go_nogo_abort():
    log_go = [{"step": 0, "loss": [2.0], "iou": [0.5]},
              {"step": 200, "loss": [1.0], "iou": [0.6]}]
    v, d = g2.decide_g2(log_go, best_iou=0.6, best_step=200, final_step=200)
    assert v == "GO" and d["iou_gain"] == pytest.approx(0.1)

    v, _ = g2.decide_g2(log_go, best_iou=0.505, best_step=200, final_step=200)
    assert v == "NO_GO"  # gain 0.005 < 0.02

    log_flat = [{"step": 0, "loss": [2.0], "iou": [0.5]},
                {"step": 200, "loss": [1.99], "iou": [0.5]}]
    v, d = g2.decide_g2(log_flat, best_iou=0.5, best_step=0, final_step=200)
    assert v == "ABORT" and d["rel_drop_at_abort_check"] < 0.02


def test_group_breakdown_modality_and_identity():
    clips = [_mk_clip("vis-a", min_iou=0.9), _mk_clip("vis-b", min_iou=0.1),
             _mk_clip("rednir-c", min_iou=0.1)]
    br = g2.group_breakdown(clips, [0.5, 0.4, 0.3], [0.6, 0.7, 0.5])
    assert br["modality:vis"]["n"] == 2 and br["modality:rednir"]["n"] == 1
    assert br["identity"]["n"] == 1 and br["non_identity"]["n"] == 2
    assert br["identity"]["frozen"] == pytest.approx(0.5)
    assert br["non_identity"]["best"] == pytest.approx(0.6)


# ─── 純 python：smoke_compare ────────────────────────────────────────

def _write_csv(path, rows):
    path.write_text("ID,x,y,width,height\n" +
                    "\n".join(f"{i},{x},{y},{w},{h}" for i, x, y, w, h in rows) + "\n")


def test_seq_of_handles_underscore_names():
    assert sc.seq_of("vis-L_person_12345") == "vis-L_person"
    assert sc.seq_of("rednir-ball&mirror10_7") == "rednir-ball&mirror10"


def test_smoke_compare_ok_and_catastrophe(tmp_path):
    gt_rows = [(f"vis-h_{i}", 10, 10, 20, 20) for i in range(1, 5)]
    gt_rows += [(f"vis-f_{i}", 30, 30, 10, 10) for i in range(1, 5)]
    _write_csv(tmp_path / "gt.csv", gt_rows)
    gt = cd.load_gt_csv(tmp_path / "gt.csv")
    plan = {"failing": ["vis-f"], "healthy": ["vis-h"]}

    perfect = {i: (x, y, w, h) for i, x, y, w, h in gt_rows}
    off = dict(perfect)
    for i in range(1, 5):  # 健康序列 IoU 掉到 0 ⇒ 災難
        off[f"vis-h_{i}"] = (200.0, 200.0, 5.0, 5.0)

    ok = sc.compare(perfect, perfect, gt, plan)
    assert ok["verdict"] == "OK" and ok["catastrophe_sequences"] == []
    assert ok["per_sequence"][0]["delta"] == pytest.approx(0.0)

    cat = sc.compare(off, perfect, gt, plan)
    assert cat["verdict"] == "CATASTROPHE"
    assert cat["catastrophe_sequences"] == ["vis-h"]
    # failing 序列惡化不觸發災難（照實記錄、不設限）
    off2 = dict(perfect)
    for i in range(1, 5):
        off2[f"vis-f_{i}"] = (200.0, 200.0, 5.0, 5.0)
    assert sc.compare(off2, perfect, gt, plan)["verdict"] == "OK"


def test_smoke_compare_missing_frame_counts_as_zero(tmp_path):
    _write_csv(tmp_path / "gt.csv", [("vis-h_1", 10, 10, 20, 20),
                                     ("vis-h_2", 10, 10, 20, 20)])
    gt = cd.load_gt_csv(tmp_path / "gt.csv")
    full = {"vis-h_1": (10.0, 10.0, 20.0, 20.0), "vis-h_2": (10.0, 10.0, 20.0, 20.0)}
    missing = {"vis-h_1": (10.0, 10.0, 20.0, 20.0)}  # 掉一幀
    per = sc.per_seq_mean_iou(missing, gt, ["vis-h"])
    assert per["vis-h"] == pytest.approx(0.5)
    assert sc.per_seq_mean_iou(full, gt, ["vis-h"])["vis-h"] == pytest.approx(1.0)


def test_load_submission_csv_roundtrip(tmp_path):
    _write_csv(tmp_path / "sub.csv", [("vis-a_1", 1.5, 2.5, 3.0, 4.0)])
    boxes = sc.load_submission_csv(tmp_path / "sub.csv")
    assert boxes["vis-a_1"] == (1.5, 2.5, 3.0, 4.0)


# ─── torch 區（無 torch ⇒ skip，不紅）─────────────────────────────────

torch = pytest.importorskip("torch")
if not hasattr(torch, "tensor"):
    # 同 test_train_tracker_g1.py 的防護：全量 run 時 test_track_t1_backends.py
    # （字母序在前）可能已塞輕量 torch stub。不可依賴 g1 測試檔先跑過（單獨跑
    # 本檔或部分選集時 stub 仍在），自帶同款 pop-and-reload。
    sys.modules.pop("torch", None)
    torch = pytest.importorskip("torch")
    if not hasattr(torch, "tensor"):  # pragma: no cover
        pytest.skip("真 torch 不可用（只有 stub）", allow_module_level=True)

from train_tracker_g1 import merge_ckpt as mc  # noqa: E402


def _fake_full_sd():
    return {
        "detector.backbone.w": torch.arange(4, dtype=torch.float32),
        "detector.head.b": torch.ones(2),
        "tracker.transformer.w": torch.zeros(3),
        "tracker.sam_mask_decoder.w": torch.zeros(2),
    }


def _fake_tracker_sd():
    return {"transformer.w": torch.tensor([1.0, 2.0, 3.0]),
            "sam_mask_decoder.w": torch.tensor([4.0, 5.0])}


def test_merge_state_dicts_swaps_tracker_only():
    base = _fake_full_sd()
    merged = mc.merge_state_dicts(base, _fake_tracker_sd())
    assert torch.equal(merged["tracker.transformer.w"], torch.tensor([1.0, 2.0, 3.0]))
    assert merged["detector.backbone.w"] is base["detector.backbone.w"]  # 原物件直傳
    assert set(merged) == set(base)


def test_merge_asserts_full_coverage_and_no_unexpected():
    base = _fake_full_sd()
    with pytest.raises(AssertionError, match="覆蓋不全"):
        mc.merge_state_dicts(base, {"transformer.w": torch.zeros(3)})
    sd = _fake_tracker_sd()
    sd["extra.w"] = torch.zeros(1)
    with pytest.raises(AssertionError, match="unexpected"):
        mc.merge_state_dicts(base, sd)


def test_merge_rejects_backbone_keys(tmp_path):
    p = tmp_path / "bad.pt"
    torch.save({"tracker_state_dict": {"backbone.x": torch.zeros(1)}}, p)
    with pytest.raises(AssertionError, match="backbone"):
        mc.load_tracker_sd(p)


@pytest.mark.parametrize("wrapped", [False, True])
def test_merge_cli_end_to_end_both_wrappers(tmp_path, wrapped):
    base_sd = _fake_full_sd()
    base_p = tmp_path / "sam3.pt"
    torch.save({"model": base_sd} if wrapped else base_sd, base_p)
    trk_p = tmp_path / "tracker_best.pt"
    torch.save({"tracker_state_dict": _fake_tracker_sd(), "step": 7}, trk_p)
    out_p = tmp_path / "merged.pt"

    rc = mc.main(["--base", str(base_p), "--tracker-ckpt", str(trk_p),
                  "--out", str(out_p)])  # --verify 預設開
    assert rc == 0
    flat, rewrapped = mc.load_full_ckpt(out_p)
    assert rewrapped == wrapped  # 輸出與輸入同構
    assert torch.equal(flat["tracker.transformer.w"], torch.tensor([1.0, 2.0, 3.0]))
    assert torch.equal(flat["detector.head.b"], torch.ones(2))  # 非 tracker 位元級同
    side = json.loads((tmp_path / "merged.pt.provenance.json").read_text())
    assert side["verified"] and side["n_tracker_keys"] == 2
    assert side["tracker_ckpt_meta"]["step"] == 7
