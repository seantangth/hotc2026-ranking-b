#!/usr/bin/env python3
"""G1 訓練包 CPU 單元測試（不需 GPU、不需 sam3、不讀 405 影像）。

torch 相關測試以 importorskip 防護：無 torch 的機器降級 skip、絕不變紅。
純 python 部分（clip 選取規則、GT parsing、box 幾何）在任何機器都真跑。
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from train_tracker_g1 import clip_dataset as cd  # noqa: E402
from train_tracker_g1 import losses as L  # noqa: E402


# ─── 純 python：box 幾何 ───────────────────────────────────────────────

def test_box_iou_identical():
    assert L.box_iou_xywh((10, 10, 20, 20), (10, 10, 20, 20)) == pytest.approx(1.0)


def test_box_iou_disjoint():
    assert L.box_iou_xywh((0, 0, 10, 10), (20, 20, 5, 5)) == 0.0


def test_box_iou_half_overlap():
    # 兩個 10×10、水平位移 5：inter=50, union=150
    assert L.box_iou_xywh((0, 0, 10, 10), (5, 0, 10, 10)) == pytest.approx(50 / 150)


def test_box_iou_degenerate_returns_zero():
    assert L.box_iou_xywh((0, 0, 0, 10), (0, 0, 10, 10)) == 0.0


# ─── 純 python：GT csv、採樣、有效性 ──────────────────────────────────

def test_load_gt_csv(tmp_path):
    p = tmp_path / "gt.csv"
    p.write_text("ID,x,y,width,height\nvis-a_1,10,20,30,40\nvis-a_2,11,21,31,41\n")
    gt = cd.load_gt_csv(p)
    assert gt["vis-a_1"] == (10.0, 20.0, 30.0, 40.0)
    assert len(gt) == 2


def test_sample_positions_stride2_covers_clip():
    assert cd.sample_positions(0, 16, 8, 2) == [0, 2, 4, 6, 8, 10, 12, 14]


def test_sample_positions_rejects_overrun():
    with pytest.raises(AssertionError):
        cd.sample_positions(0, 16, 9, 2)


def test_frame_valid_matches_gt_gt0_rule():
    assert cd.frame_valid((1, 1, 5, 5))
    assert not cd.frame_valid((0, 1, 5, 5))  # build_hard_clips: np.all(gt > 0)
    assert not cd.frame_valid(None)


# ─── 純 python：G1 clip 選取規則 ──────────────────────────────────────

def _mk_clip(seq, min_iou=0.1, mean_iou=0.3, failing=8, start=0, length=16):
    ids = [f"{seq}_{100 + start + i}" for i in range(length)]
    return {"sequence": seq, "modality": seq.split("-")[0], "start_position": start,
            "length": length, "frame_ids": ids, "mean_iou": mean_iou,
            "min_iou": min_iou, "failing_frames": failing, "frozen_frames": 0,
            "fold": 0}


def _mk_gt(clips, invalid_ids=()):
    gt = {}
    for c in clips:
        for fid in c["frame_ids"]:
            gt[fid] = (0.0, 0.0, 0.0, 0.0) if fid in invalid_ids else (5.0, 5.0, 10.0, 10.0)
    return gt


def test_select_excludes_identity_clips():
    clips = [_mk_clip("vis-id", min_iou=0.9), _mk_clip("vis-a"), _mk_clip("vis-b")]
    gt = _mk_gt(clips)
    chosen = cd.select_g1_clips(clips, gt, n=2)
    assert {c["sequence"] for c in chosen} == {"vis-a", "vis-b"}


def test_select_one_clip_per_sequence_and_ranking():
    clips = [
        _mk_clip("vis-a", failing=9, start=0),
        _mk_clip("vis-a", failing=8, start=20),   # 同序列第二支：不取
        _mk_clip("vis-b", failing=5),
        _mk_clip("rednir-c", failing=7),
    ]
    gt = _mk_gt(clips)
    chosen = cd.select_g1_clips(clips, gt, n=3)
    assert [c["sequence"] for c in chosen] == ["vis-a", "rednir-c", "vis-b"]
    assert chosen[0]["start_position"] == 0  # failing 較多的那支


def test_select_excludes_invalid_first_frame():
    bad = _mk_clip("vis-bad")
    ok = _mk_clip("vis-ok")
    gt = _mk_gt([bad, ok], invalid_ids={bad["frame_ids"][0]})
    chosen = cd.select_g1_clips([bad, ok], gt, n=1)
    assert chosen[0]["sequence"] == "vis-ok"


def test_select_is_deterministic():
    clips = [_mk_clip(f"vis-s{i}", failing=8, mean_iou=0.3) for i in range(6)]
    gt = _mk_gt(clips)
    a = [c["sequence"] for c in cd.select_g1_clips(clips, gt, n=4)]
    b = [c["sequence"] for c in cd.select_g1_clips(clips, gt, n=4)]
    assert a == b == sorted(a)[:4]  # tie 時以 sequence 名保序


def test_select_insufficient_raises():
    clips = [_mk_clip("vis-a")]
    with pytest.raises(ValueError):
        cd.select_g1_clips(clips, _mk_gt(clips), n=10)


def test_select_against_real_index_if_present():
    """真 JSON 在本機存在時順手驗 schema 假設（CI/乾淨機自動跳過）。"""
    real = (Path(__file__).parent.parent / "5_outputs" / "d079_gate0_20260828"
            / "hard_clips_vis_rednir.json")
    if not real.exists():
        pytest.skip("hard clips JSON 不在本機")
    index = cd.load_clip_index(real)
    c0 = index["clips"][0]
    for key in ("sequence", "start_position", "length", "frame_ids",
                "mean_iou", "min_iou", "failing_frames"):
        assert key in c0, f"schema 變了：缺 {key}"
    assert len(c0["frame_ids"]) == c0["length"] == 16


# ─── torch 區（無 torch ⇒ skip，不紅）─────────────────────────────────

torch = pytest.importorskip("torch")
if not hasattr(torch, "tensor"):
    # 全量 run 時 test_track_t1_backends.py（字母序在前）可能已在 sys.modules
    # 塞了輕量 torch stub（它測的是自家邏輯、其 track_t1 綁定在 import 時已完成，
    # 不受此處影響）。本檔需要真 torch：pop 掉 stub 重載；真 torch 不存在則
    # importorskip 整檔 skip、絕不紅。
    # ⚠️ 副作用假設：此後 sys.modules["torch"] 是真 torch——字母序排在本檔之後
    # 的測試檔會看到真 torch 而非 stub。今日 3_src 沒有這樣的檔；新增排序在
    # test_train_tracker_g1 之後且依賴 stub 的測試檔時，此假設失效。
    sys.modules.pop("torch", None)
    torch = pytest.importorskip("torch")
    if not hasattr(torch, "tensor"):  # pragma: no cover —— 防雙重 stub
        pytest.skip("真 torch 不可用（只有 stub）", allow_module_level=True)


def test_projection_targets_geometry():
    tx, ty = L.projection_targets(torch.tensor([[2.0, 3.0, 4.0, 5.0]]), size=16)
    assert tx.shape == ty.shape == (1, 16)
    assert tx[0, 2:6].tolist() == [1.0] * 4 and float(tx[0].sum()) == 4.0
    assert ty[0, 3:8].tolist() == [1.0] * 5 and float(ty[0].sum()) == 5.0


def test_projection_loss_perfect_vs_wrong():
    s = 32
    gt = torch.tensor([[8.0, 8.0, 12.0, 12.0]])
    valid = [True]
    good = torch.full((1, 1, s, s), -12.0)
    good[0, 0, 8:20, 8:20] = 12.0
    wrong = torch.full((1, 1, s, s), -12.0)
    wrong[0, 0, 0:4, 24:32] = 12.0
    l_good, parts = L.projection_loss(good, gt, valid)
    l_wrong, _ = L.projection_loss(wrong, gt, valid)
    assert parts["n_valid"] == 1
    assert float(l_good) < 0.05 < float(l_wrong)
    assert l_good.requires_grad is False  # 輸入無 grad 時不憑空造 graph


def test_projection_loss_gradient_flows():
    logits = torch.zeros(2, 1, 16, 16, requires_grad=True)
    gt = torch.tensor([[2.0, 2.0, 6.0, 6.0], [4.0, 4.0, 8.0, 8.0]])
    loss, _ = L.projection_loss(logits, gt, [True, True])
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_projection_loss_respects_valid_mask():
    logits = torch.zeros(2, 1, 16, 16, requires_grad=True)
    gt = torch.tensor([[2.0, 2.0, 6.0, 6.0], [0.0, 0.0, 0.0, 0.0]])
    loss_one, parts = L.projection_loss(logits, gt, [True, False])
    assert parts["n_valid"] == 1
    loss_zero, parts0 = L.projection_loss(logits, gt, [False, False])
    assert parts0["n_valid"] == 0 and float(loss_zero.detach()) == 0.0
    loss_zero.backward()  # 全 invalid 也要能 backward（graph 連通）
    assert float(loss_one) > 0.0


def test_mask_to_box_matches_track_t1_semantics():
    m = torch.zeros(10, 10, dtype=torch.bool)
    m[3:7, 2:9] = True  # ys 3..6, xs 2..8
    assert L.mask_to_box_xywh(m) == (2.0, 3.0, 7.0, 4.0)  # x2=max+1（E33）
    assert L.mask_to_box_xywh(torch.zeros(4, 4, dtype=torch.bool)) is None


def test_iou_head_proxy_loss_values():
    s = 32
    logits = torch.full((1, 1, s, s), -12.0)
    logits[0, 0, 8:20, 8:20] = 12.0  # tight box = (8,8,12,12)
    gt = torch.tensor([[8.0, 8.0, 12.0, 12.0]])
    perfect = torch.tensor([1.0])
    bad = torch.tensor([0.0])
    l_perfect, parts = L.iou_head_proxy_loss(perfect, logits, gt, [True])
    l_bad, _ = L.iou_head_proxy_loss(bad, logits, gt, [True])
    assert float(l_perfect) == pytest.approx(0.0, abs=1e-6)
    assert float(l_bad) == pytest.approx(1.0, abs=1e-6)
    assert parts["n_valid"] == 1


def test_g1_total_loss_combines_and_tolerates_missing_iou():
    logits = torch.zeros(1, 1, 16, 16, requires_grad=True)
    gt = torch.tensor([[2.0, 2.0, 6.0, 6.0]])
    with_iou, parts = L.g1_total_loss(logits, torch.tensor([0.5]), gt, [True])
    without, parts2 = L.g1_total_loss(logits, None, gt, [True])
    assert "iou_head_mse" in parts and "iou_head_mse" not in parts2
    assert float(with_iou) >= float(without)
    with_iou.backward()
    assert logits.grad is not None


# ─── torch：synthetic dataset ────────────────────────────────────────

def test_synthetic_dataset_shapes_and_alignment():
    ds = cd.ClipDataset.make_synthetic(n_clips=3, t=8, stride=1, image_size=64)
    sample = ds[2]
    assert sample["images"].shape == (8, 3, 64, 64)
    assert sample["gt_boxes_model"].shape == (8, 4)
    assert sample["valid"].shape == (8,) and bool(sample["valid"][0])
    assert sample["init_box_rel"].shape == (1, 4)
    assert float(sample["init_box_rel"].max()) <= 1.0
    # 合成影像的白方塊應與 GT box 對齊：mask_to_box(影像>閾值) ≈ GT
    img0 = sample["images"][0].mean(dim=0)  # 灰階
    box = L.mask_to_box_xywh(img0 > 0.5)  # normalize 後白=~1.4、灰=~-0.5
    gt0 = tuple(float(v) for v in sample["gt_boxes_model"][0])
    assert L.box_iou_xywh(box, gt0) > 0.9


def test_synthetic_dataset_has_invalid_frame_case():
    ds = cd.ClipDataset.make_synthetic(n_clips=3, t=8, stride=1, image_size=64)
    sample = ds[1]  # make_synthetic 對 ci=1 埋了一幀無效 GT
    assert not bool(sample["valid"].all())


def test_synthetic_selection_pipeline_end_to_end():
    ds = cd.ClipDataset.make_synthetic(n_clips=3, t=8, stride=1, image_size=64)
    chosen = cd.select_g1_clips(ds.clips, ds.gt_by_id, n=2, t=8, stride=1)
    names = {c["sequence"] for c in chosen}
    assert "vis-synth0" not in names  # identity（min_iou=0.9）被排除
    assert len(names) == 2
