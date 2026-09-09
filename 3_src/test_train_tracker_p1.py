#!/usr/bin/env python3
"""P1 精度線（D105）的 CPU 回歸：uniform clip 索引、Stage A/B 判準表、arm 選擇、lr 排程、smoke 計畫規則。
torch 相關（凍結範圍）只能在機上第一次 backward 前驗（launch 腳本有 3 行自檢）；此處純 CPU。"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_uniform_clips as bu  # noqa: E402
from train_tracker_g1 import precision_compare as pc  # noqa: E402
from train_tracker_g1 import train_p1 as p1  # noqa: E402
# sot_trainable 在模組層 import torch；本機無 torch ⇒ 該測試以 importorskip 於機上執行


# ─── build_uniform_clips ────────────────────────────────────────────────────
def _contract_rows(seq, n, fold, invalid_positions=()):
    mod = seq.split("-", 1)[0]
    return [{"ID": f"{seq}_{1000 + i}", "sequence": seq, "position": str(i), "modality": mod,
             "capture_group": seq.split("-", 1)[1], "fold": str(fold),
             "gt_valid": "0" if i in invalid_positions else "1",
             "selected_block_start": "0", "selected_block_end": str(n)} for i in range(n)]


def _write_contract(tmp_path, rows):
    p = tmp_path / "frame_contract.csv"
    with p.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return p


def _gt_for(rows):
    gt = {}
    for r in rows:
        gt[r["ID"]] = (0.0, 0.0, 0.0, 0.0) if r["gt_valid"] == "0" else (10.0, 10.0, 20.0, 20.0)
    return gt


def test_window_starts_grid():
    assert bu.window_starts(15) == []
    assert bu.window_starts(16) == [0]
    assert bu.window_starts(40) == [0, 16]
    assert bu.window_starts(48) == [0, 16, 32]


def test_uniform_clips_schema_and_valid_filter(tmp_path):
    rows = _contract_rows("vis-a", 48, fold=2) + _contract_rows("nir-b", 32, fold=4, invalid_positions=range(0, 12))
    by_seq = bu.load_frame_contract(_write_contract(tmp_path, rows))
    clips = bu.build_uniform_clips(by_seq, _gt_for(rows), None)
    # vis-a: 3 窗；nir-b: 第一窗只有 4 幀有效（<8）被丟、第二窗保留
    assert [(c["sequence"], c["start_position"]) for c in clips] == \
        [("nir-b", 16), ("vis-a", 0), ("vis-a", 16), ("vis-a", 32)]
    c = clips[1]
    assert c["length"] == 16 and c["fold"] == 2 and c["modality"] == "vis"
    assert c["frame_ids"] == [f"vis-a_{1000 + i}" for i in range(16)]
    assert c["mean_iou"] is None and c["min_iou"] is None and c["failing_frames"] is None
    doc = bu.summarize(clips, has_pred_stats=False)
    assert doc["n_clips"] == 4 and doc["clips_per_fold"] == {"2": 3, "4": 1}
    assert doc["has_pred_stats"] is False and "identity_rate" not in doc


def test_uniform_clips_pred_stats_and_frozen(tmp_path):
    rows = _contract_rows("rednir-c", 16, fold=1)
    by_seq = bu.load_frame_contract(_write_contract(tmp_path, rows))
    gt = _gt_for(rows)
    pred = {r["ID"]: (10.0, 10.0, 20.0, 20.0) for r in rows}          # 全部命中
    pred[rows[3]["ID"]] = (50.0, 50.0, 5.0, 5.0)                          # 一幀失敗
    clips = bu.build_uniform_clips(by_seq, gt, pred)
    assert len(clips) == 1
    c = clips[0]
    assert c["failing_frames"] == 1 and math.isclose(c["min_iou"], 0.0)
    assert c["frozen_frames"] > 0  # 連續相同框 ⇒ 凍結計數 > 0
    doc = bu.summarize(clips, has_pred_stats=True)
    assert doc["has_pred_stats"] and doc["identity_clips"] == 0


def test_uniform_clips_fail_closed_on_gt_mismatch(tmp_path):
    rows = _contract_rows("vis-d", 16, fold=0)
    by_seq = bu.load_frame_contract(_write_contract(tmp_path, rows))
    gt = _gt_for(rows)
    gt[rows[0]["ID"]] = (0.0, 0.0, 0.0, 0.0)  # contract 說有效、GT 說無效
    with pytest.raises(AssertionError):
        bu.build_uniform_clips(by_seq, gt, None)


@pytest.mark.parametrize("box", [
    (0.0, 94.0, 20.0, 24.0),      # 貼左緣：x=0（實例 nir-ball_31，09-04 讓 P1 首跑當場死掉）
    (314.0, 0.0, 21.0, 18.0),     # 貼上緣：y=0（實例 rednir-ball&mirror10_18398）
    (0.0, 0.0, 15.0, 9.0),        # 左上角：x=y=0
    (-3.0, 88.0, 24.0, 23.0),     # 框超出左邊界：x 為負，官方標註允許
])
def test_edge_touching_gt_boxes_are_valid(tmp_path, box):
    """x/y 可為 0 或負；判準只看 w>0、h>0 ＋ 全為有限值。

    判準必須與契約端 prep_oof405_frames.py:57 `valid_box` 逐字相同。
    舊碼用 all(v > 0 for v in g)，把 1,352 幀貼邊的合法框誤判為無效，
    與 contract 的 gt_valid=1 衝突 ⇒ fail-closed 斷言在 GPU 上才爆。
    """
    rows = _contract_rows("nir-ball", 16, fold=0)
    by_seq = bu.load_frame_contract(_write_contract(tmp_path, rows))
    gt = _gt_for(rows)
    gt[rows[0]["ID"]] = box
    clips = bu.build_uniform_clips(by_seq, gt, None)
    assert len(clips) == 1, "貼邊的合法框不該讓整個 clip 被丟掉"


@pytest.mark.parametrize("box", [
    (10.0, 10.0, 0.0, 20.0),          # w=0 ⇒ 真無效
    (10.0, 10.0, 20.0, -1.0),         # h<0 ⇒ 真無效
    (float("nan"), 10.0, 20.0, 20.0),  # 非有限值 ⇒ 真無效
    (10.0, float("inf"), 20.0, 20.0),
])
def test_degenerate_gt_boxes_still_rejected(tmp_path, box):
    """修正不得放寬真正無效的框——contract 說有效時仍須 fail closed。"""
    rows = _contract_rows("nir-ball", 16, fold=0)
    by_seq = bu.load_frame_contract(_write_contract(tmp_path, rows))
    gt = _gt_for(rows)
    gt[rows[0]["ID"]] = box
    with pytest.raises(AssertionError):
        bu.build_uniform_clips(by_seq, gt, None)


# ─── train_p1：lr 排程、Stage A、smoke 計畫規則 ─────────────────────────────
def test_cosine_lr_shape():
    base = 1e-4
    assert math.isclose(p1.cosine_lr(1, 3000, base), base / p1.LR_WARMUP_STEPS)
    assert math.isclose(p1.cosine_lr(p1.LR_WARMUP_STEPS, 3000, base), base)
    mid = p1.cosine_lr((3000 + p1.LR_WARMUP_STEPS) // 2, 3000, base)
    assert base * 0.5 < mid < base * 0.6
    assert math.isclose(p1.cosine_lr(3000, 3000, base), base * p1.LR_FINAL_FRAC)


def _log(*points):
    return [{"step": s, "loss": [l], "iou": [i]} for s, l, i in points]


def test_decide_stage_a_all_branches():
    v, d = p1.decide_stage_a(_log((0, 1.0, 0.80), (250, 0.99, 0.81)), 0.81, 250, 250)
    assert v == "ABORT" and d["rel_drop_at_abort_check"] < 0.02
    v, d = p1.decide_stage_a(_log((0, 1.0, 0.80), (250, 0.5, 0.805)), 0.805, 250, 3000)
    assert v == "FAIL_A" and math.isclose(d["clip_gain"], 0.005)
    v, d = p1.decide_stage_a(_log((0, 1.0, 0.80), (250, 0.5, 0.82)), 0.82, 250, 3000)
    assert v == "PASS_A" and math.isclose(d["clip_gain"], 0.02)
    v, _ = p1.decide_stage_a(_log((0, 1.0, 0.80), (50, 0.5, 0.85)), 0.85, 50, 50)
    assert v == "FAIL_A"  # 未跑滿 100 步不得 PASS


def test_evenly_spaced_rule():
    assert p1.evenly_spaced(["b", "a", "c"], 5) == ["a", "b", "c"]
    assert p1.evenly_spaced(list("abcdefg"), 4) == ["a", "c", "e", "g"]
    assert p1.evenly_spaced(list("abcdefg"), 1) == ["a"]
    assert p1.evenly_spaced(list("abc"), 0) == []


def test_build_precision_smoke_plan_selection(tmp_path):
    frames = tmp_path / "frames"
    clips, gt = [], {}
    def mk(seq, n, fold, valid_n=None, jpgs=None):
        (frames / seq).mkdir(parents=True)
        for i in range(n if jpgs is None else jpgs):
            (frames / seq / f"{i:05d}.jpg").write_bytes(b"x")
        for i in range(n):
            gt[f"{seq}_{i+1}"] = (1.0, 1.0, 5.0, 5.0) if (valid_n is None or i < valid_n) else (0.0, 0.0, 0.0, 0.0)
        clips.append({"sequence": seq, "modality": seq.split("-")[0], "fold": fold,
                      "start_position": 0, "length": 16, "frame_ids": [f"{seq}_{i+1}" for i in range(16)],
                      "mean_iou": None, "min_iou": None, "failing_frames": None, "frozen_frames": None})
    for k in "abcdefg":            # 7 支 vis 合格 → 等距取 4
        mk(f"vis-{k}", 40, 4)
    mk("nir-x", 40, 4); mk("nir-y", 40, 4)          # 2 支 nir → 全取
    mk("nir-z", 40, 4, jpgs=39)                     # GT 行數 ≠ jpg → 剔除
    mk("rednir-p", 40, 4, valid_n=10)               # 有效 GT <16 → 剔除
    mk("rednir-q", 40, 3)                           # 非 holdout fold → 不算候選
    plan = p1.build_precision_smoke_plan(clips, frames, gt, {4}, per_modality=4)
    assert plan["by_modality"]["vis"] == ["vis-a", "vis-c", "vis-e", "vis-g"]
    assert plan["by_modality"]["nir"] == ["nir-x", "nir-y"]
    assert plan["by_modality"]["rednir"] == []
    assert {s["sequence"] for s in plan["substituted"]} == {"nir-z", "rednir-p"}
    assert plan["all"] == ["vis-a", "vis-c", "vis-e", "vis-g", "nir-x", "nir-y"]


def test_group_breakdown_skips_identity_without_stats():
    clips = [{"modality": "vis", "min_iou": None}, {"modality": "nir", "min_iou": None}]
    out = p1.group_breakdown_p1(clips, [0.8, 0.7], [0.82, 0.71])
    assert "modality:vis" in out and "identity" not in out and "identity_grouping" in out
    clips2 = [{"modality": "vis", "min_iou": 0.9}, {"modality": "vis", "min_iou": 0.1}]
    out2 = p1.group_breakdown_p1(clips2, [0.8, 0.3], [0.82, 0.5])
    assert out2["identity"]["n"] == 1 and out2["non_identity"]["n"] == 1


# ─── precision_compare：Stage B 判準表、arm 選擇 ────────────────────────────
def _toy(plan_seqs, frozen_iou, trained_iou, n=20):
    gt, fr, tr = {}, {}, {}
    for s in plan_seqs:
        for i in range(n):
            fid = f"{s}_{i+1}"
            gt[fid] = (10.0, 10.0, 20.0, 20.0)
            fr[fid] = _box_with_iou(frozen_iou[s])
            tr[fid] = _box_with_iou(trained_iou[s])
    return gt, fr, tr


def _box_with_iou(target):
    """與 GT (10,10,20,20) 有指定 IoU 的同心縮放框（IoU=r² 當 r<1 ⇒ r=sqrt(target)）。"""
    if target >= 1.0:
        return (10.0, 10.0, 20.0, 20.0)
    if target <= 0.0:
        return (200.0, 200.0, 5.0, 5.0)
    r = math.sqrt(target)
    w = 20.0 * r
    return (20.0 - w / 2, 20.0 - w / 2, w, w)


@pytest.mark.parametrize("trained,expected", [
    ({"vis-a": 0.86, "nir-b": 0.86}, "GO"),          # pooled Δ +0.06
    ({"vis-a": 0.802, "nir-b": 0.802}, "GREY"),      # pooled Δ +0.002
    ({"vis-a": 0.79, "nir-b": 0.79}, "NO_GO"),       # pooled Δ −0.01
    ({"vis-a": 0.90, "nir-b": 0.70}, "CATASTROPHE"), # 一支 −0.10，儘管 pooled 持平
])
def test_stage_b_verdict_table(trained, expected):
    seqs = ["vis-a", "nir-b"]
    gt, fr, tr = _toy(seqs, {"vis-a": 0.80, "nir-b": 0.80}, trained)
    res = pc.compare(tr, fr, gt, {"all": seqs}, clip_gain=0.02)
    assert res["verdict"] == expected, res


def test_stage_b_requires_stage_a():
    seqs = ["vis-a"]
    gt, fr, tr = _toy(seqs, {"vis-a": 0.80}, {"vis-a": 0.90})
    assert pc.compare(tr, fr, gt, {"all": seqs}, clip_gain=0.005)["verdict"] == "NO_GO_A"


def test_pooled_is_frame_weighted_not_seq_mean():
    gt, fr, tr = {}, {}, {}
    for i in range(90):  # 長序列 90 幀持平
        fid = f"vis-long_{i+1}"; gt[fid] = (10.0, 10.0, 20.0, 20.0); fr[fid] = tr[fid] = gt[fid]
    for i in range(10):  # 短序列 10 幀 +0.5
        fid = f"vis-short_{i+1}"; gt[fid] = (10.0, 10.0, 20.0, 20.0)
        fr[fid] = _box_with_iou(0.4); tr[fid] = _box_with_iou(0.9)
    res = pc.compare(tr, fr, gt, {"all": ["vis-long", "vis-short"]}, clip_gain=0.02)
    assert math.isclose(res["pooled_delta"], 0.05, abs_tol=1e-6)  # 0.5 × 10/100
    assert res["diagnostics"]["tracked_share_frozen"] == 0.9


def test_select_arm_rules():
    A = {"verdict": "GO", "pooled_delta": 0.010}
    B = {"verdict": "GO", "pooled_delta": 0.011}
    assert pc.select_arm({"A": A, "B": B})["chosen"] == "A"          # 差 <0.002 → A
    B2 = {"verdict": "GO", "pooled_delta": 0.020}
    assert pc.select_arm({"A": A, "B": B2})["chosen"] == "B"         # 明顯較高 → B
    assert pc.select_arm({"A": {"verdict": "GREY", "pooled_delta": 0.004}, "B": B2})["chosen"] == "B"
    assert pc.select_arm({"A": {"verdict": "GREY", "pooled_delta": 0.004},
                          "B": {"verdict": "NO_GO", "pooled_delta": -0.01}})["chosen"] is None


def test_select_cli_roundtrip(tmp_path):
    a = tmp_path / "a.json"; b = tmp_path / "b.json"; out = tmp_path / "sel.json"
    a.write_text(json.dumps({"arm": "A", "verdict": "GO", "pooled_delta": 0.02}))
    b.write_text(json.dumps({"arm": "B", "verdict": "GREY", "pooled_delta": 0.001}))
    assert pc.main(["--select", str(a), str(b), "--out", str(out)]) == 0
    assert json.loads(out.read_text())["chosen"] == "A"


# ─── sot_trainable：scope 表（純表格，不需 torch）───────────────────────────
def test_scope_param_ranges_table():
    pytest.importorskip("torch")
    from train_tracker_g1 import sot_trainable as st
    assert set(st.SCOPES) == {"decoder", "tracker"}
    assert st.SCOPE_PARAM_RANGES["tracker"] == (st.TRAINABLE_PARAM_MIN, st.TRAINABLE_PARAM_MAX)
    lo, hi = st.SCOPE_PARAM_RANGES["decoder"]
    assert lo < 4_200_000 < hi                      # D088 實測 sam_mask_decoder 4.2M
    assert hi < st.TRAINABLE_PARAM_MIN              # 兩個範圍不重疊，凍錯範圍必被抓到
