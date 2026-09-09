#!/usr/bin/env python3.14
"""track_t1.py 雙 backend 離線回歸測試（純 CPU，無需 torch/GPU/資料）。

動 track_t1.py 就必須跑這支——它是 Ranking B 一鍵重現交付物。除了本次明確修正的
「每序列 Kalman 預設隔離」外，兩 backend 共用的 box 行為仍須回歸不變。

核心斷言：**兩個 backend 餵同一串 mask 必須產出逐幀完全相同的 box**
——證明 backend 之後的一切（mask→box、空 mask 沿用前框、首幀 init）完全共用，
E15 的對照確實是單變因隔離（只換 tracker）。

另驗三個「不報錯只做錯事」的 SAM3 介面陷阱（見 _iter_masks_sam3 docstring）。

用法：python3.14 test_track_t1_backends.py
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np

# --- stub torch：測的是我方邏輯，不是 torch ---------------------------------
if "torch" not in sys.modules:
    class _Ctx:
        def __enter__(self): return self
        def __exit__(self, *a): return False

    torch = types.ModuleType("torch")
    torch.bfloat16 = "bfloat16"
    torch.inference_mode = lambda *a, **k: _Ctx()
    torch.autocast = lambda *a, **k: _Ctx()
    torch.manual_seed = lambda s: None
    torch.cuda = types.SimpleNamespace(empty_cache=lambda: None)
    sys.modules["torch"] = torch

sys.path.insert(0, str(Path(__file__).parent))
import track_t1  # noqa: E402


class FakeT:
    """最小張量替身：支援 [i]、> v、.cpu().numpy()。"""

    def __init__(self, a): self.a = np.asarray(a)
    def __getitem__(self, i): return FakeT(self.a[i])
    def __gt__(self, v): return FakeT(self.a > v)
    def cpu(self): return self
    def numpy(self): return self.a


def make_masks(h=40, w=60):
    """5 幀：3 幀有物件（位置遞移）、1 幀全空（測沿用前框）、1 幀回來。"""
    seq = []
    for cx in (10, 14, None, 22, 26):
        m = np.zeros((h, w), dtype=bool)
        if cx is not None:
            m[8:16, cx:cx + 9] = True
        seq.append(m)
    return seq


class FakeSamuraiPredictor:
    """SAM2/SAMURAI 介面：propagate 回傳 3 值。"""

    def __init__(self, masks): self.masks = masks; self.calls = {}

    def init_state(self, video_path, offload_video_to_cpu=False):
        self.calls["offload"] = offload_video_to_cpu
        return {"v": video_path}

    def add_new_points_or_box(self, state, box=None, frame_idx=None, obj_id=None):
        self.calls["box"] = box

    def propagate_in_video(self, state):
        for i, m in enumerate(self.masks):
            yield i, [0], FakeT(m[None, None])

    def reset_state(self, state): self.calls["reset"] = True


class FakeSam3Predictor:
    """SAM3 介面：box 須正規化、propagate 三參數無預設且回傳 5 值。"""

    def __init__(self, masks): self.masks = masks; self.calls = {}

    def init_state(self, video_path, offload_video_to_cpu=False, async_loading_frames=False):
        self.calls["offload"] = offload_video_to_cpu
        self.calls["async"] = async_loading_frames
        return {"v": video_path}

    def add_new_points_or_box(self, inference_state=None, frame_idx=None, obj_id=None, box=None):
        self.calls["box"] = np.asarray(box)

    def propagate_in_video(self, state, start_frame_idx=None, max_frame_num_to_track="MISSING",
                           reverse=None, propagate_preflight=False, tqdm_disable=False):
        self.calls["max_frames"] = max_frame_num_to_track
        self.calls["start"] = start_frame_idx
        self.calls["preflight"] = propagate_preflight
        for i, m in enumerate(self.masks):
            yield i, [0], FakeT(m[None]), FakeT(m[None]), [0.9]

    def reset_state(self, state): self.calls["reset"] = True


def write_frames(d: Path, n: int, w: int, h: int):
    from PIL import Image
    d.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        Image.new("RGB", (w, h), (i * 7 % 256, 40, 90)).save(d / f"{i:05d}.jpg", quality=90)


def main() -> int:
    import tempfile

    H, W = 40, 60
    masks = make_masks(H, W)
    init = [10.0, 8.0, 9.0, 8.0]
    tmp = Path(tempfile.mkdtemp(prefix="t1bk_"))
    seq_dir = tmp / "nir-fake1"
    write_frames(seq_dir, len(masks), W, H)

    fails = []

    # --- 1. samurai 路徑 ---------------------------------------------------
    p1 = FakeSamuraiPredictor(masks)
    b1, raw1, d1 = track_t1.track_sequence(p1, seq_dir, init, want_masks=True, backend="samurai")

    if p1.calls.get("offload") is not True:
        fails.append("samurai: offload_video_to_cpu 應為 True（E02 固定行為）")
    if p1.calls.get("box") != [10.0, 8.0, 19.0, 16.0]:
        fails.append(f"samurai: box 應為絕對 xyxy [10,8,19,16]，實得 {p1.calls.get('box')}")
    if b1.get(0) != init:
        fails.append(f"samurai: 位置 0 必須照抄 init，實得 {b1.get(0)}")

    # --- 2. sam3 路徑 ------------------------------------------------------
    p2 = FakeSam3Predictor(masks)
    b2, raw2, d2 = track_t1.track_sequence(p2, seq_dir, init, want_masks=True, backend="sam3")

    box_sent = p2.calls.get("box")
    expect_rel = np.array([[10 / W, 8 / H, 19 / W, 16 / H]], dtype=np.float32)
    if box_sent is None or box_sent.shape != (1, 4):
        fails.append(f"sam3: box 應為 shape (1,4) 的 2D array，實得 {None if box_sent is None else box_sent.shape}")
    elif not np.allclose(box_sent, expect_rel, atol=1e-6):
        fails.append(f"sam3: box 未正規化（陷阱 1）——期望 {expect_rel.tolist()}，實得 {box_sent.tolist()}")
    if p2.calls.get("max_frames") is not None:
        fails.append(f"sam3: max_frame_num_to_track 必須為 None 才追到底（陷阱 3，範例的 240 會截斷 "
                     f"nir 平均 892 幀），實得 {p2.calls.get('max_frames')!r}")
    if p2.calls.get("preflight") is not True:
        fails.append("sam3: propagate_preflight 應為 True（官方範例）")
    if p2.calls.get("offload") is not True:
        fails.append("sam3: offload_video_to_cpu 應為 True（與 E02 對齊）")
    if p2.calls.get("async") is not True:
        fails.append("sam3: async_loading_frames 預設應為 True（載入與 GPU forward 重疊）")

    # async 可關閉（懷疑幀錯位時排除變因用）
    p3 = FakeSam3Predictor(masks)
    b3, _, _ = track_t1.track_sequence(p3, seq_dir, init, want_masks=False,
                                       backend="sam3", sam3_async=False)
    if p3.calls.get("async") is not False:
        fails.append("sam3: --sam3-no-async-load 應能關閉背景載入")
    if {k: v for k, v in b3.items()} != {k: v for k, v in b2.items()}:
        fails.append("sam3: 開關 async 不應改變輸出（只影響載入時機）")

    # --- 3. 核心：兩 backend 逐幀輸出必須完全相同 --------------------------
    if set(b1) != set(b2):
        fails.append(f"兩 backend 幀位置集合不同：{sorted(set(b1) ^ set(b2))}")
    else:
        diff = [k for k in b1 if b1[k] != b2[k]]
        if diff:
            fails.append(f"兩 backend 有 {len(diff)} 幀 box 不同（單變因隔離破功）："
                         + "; ".join(f"pos{k} {b1[k]} vs {b2[k]}" for k in diff[:3]))

    # --- 4. 空 mask 沿用前框 + empty 計數 ----------------------------------
    if d1["empty_mask_frames"] != 1 or d2["empty_mask_frames"] != 1:
        fails.append(f"空 mask 計數應為 1，實得 samurai={d1['empty_mask_frames']} sam3={d2['empty_mask_frames']}")
    if b1.get(2) != b1.get(1):
        fails.append(f"空 mask 幀應沿用前框：pos2 {b1.get(2)} 應等於 pos1 {b1.get(1)}")

    # --- 4b. samurai Kalman 預設隔離；legacy 才能 opt-in 跨序列狀態 --------
    p4 = FakeSamuraiPredictor(masks)
    p4.kf_mean, p4.kf_covariance, p4.stable_frames = "髒狀態", "髒狀態", 200
    track_t1.track_sequence(p4, seq_dir, init, want_masks=False, backend="samurai")
    if not (p4.kf_mean is None and p4.kf_covariance is None and p4.stable_frames == 0):
        fails.append("samurai: 預設必須重置 Kalman，隔離每支序列")

    p5 = FakeSamuraiPredictor(masks)
    p5.kf_mean, p5.kf_covariance, p5.stable_frames = "髒狀態", "髒狀態", 200
    track_t1.track_sequence(p5, seq_dir, init, want_masks=False, backend="samurai",
                            samurai_reset_kf=False)
    if p5.stable_frames != 200:
        fails.append("samurai: legacy cross-seq opt-in 應保留 Kalman 狀態")

    p6 = FakeSamuraiPredictor(masks)
    p6.kf_mean, p6.kf_covariance, p6.stable_frames = "髒狀態", "髒狀態", 200
    track_t1.track_sequence(p6, seq_dir, init, want_masks=False, backend="samurai",
                            samurai_reset_kf=True)
    if not (p6.kf_mean is None and p6.kf_covariance is None and p6.stable_frames == 0):
        fails.append("samurai: 舊 --samurai-reset-kf 相容路徑仍應清空 Kalman 狀態")

    # --- 5. mask 快取形狀 --------------------------------------------------
    for tag, raw in (("samurai", raw1), ("sam3", raw2)):
        if raw is None or len(raw["masks"]) != len(masks):
            fails.append(f"{tag}: mask 快取應有 {len(masks)} 幀")
        elif raw["masks"][0].shape != (H, W):
            fails.append(f"{tag}: mask 快取形狀應為 ({H},{W})，實得 {raw['masks'][0].shape}")

    print(f"samurai boxes: {[b1[k] for k in sorted(b1)]}")
    print(f"sam3    boxes: {[b2[k] for k in sorted(b2)]}")
    print()
    if fails:
        print(f"❌ {len(fails)} 項失敗：")
        for f in fails:
            print(f"  - {f}")
        return 1
    print("✅ 全部通過：samurai 預設隔離 + sam3 三陷阱已擋 + 兩 backend 單變因隔離成立")
    return 0


if __name__ == "__main__":
    sys.exit(main())
