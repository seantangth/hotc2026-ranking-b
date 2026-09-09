#!/usr/bin/env python3
"""G1 hard-clips 載入器。

資料血緣（全部實查於 2026-08-29，勿憑記憶改動）：
* clips index = 5_outputs/d079_gate0_20260828/hard_clips_vis_rednir.json，
  由 3_src/build_hard_clips.py 產生。每支 clip 的欄位：sequence / modality /
  start_position / length(=16) / frame_ids / mean_iou / min_iou /
  failing_frames / frozen_frames / fold。
* ``start_position`` 是**序列內 0-based 幀位置**（build_hard_clips.py:113-116
  的 ``lo``，索引 oof contract 順序＝prep_oof405_frames.py 的 sorted jpg 順序）
  ⇒ frames_root/<seq>/ 下 sorted(*.jpg) 的第 start_position+i 張，就是
  frame_ids[i] 那一幀。**frames_root 必須由 prep_oof405_frames.py 產生**，
  否則此對齊不成立（launch 腳本因此重跑完整 prep，含 167174 行 contract 驗證）。
* clip JSON **不含 GT box**：GT 從 1_data/raw/2026training.csv 讀
  （``ID,x,y,width,height``，ID = frame_ids 的格式）。
* GT 無效幀＝x,y,w,h 任一 ≤ 0（build_hard_clips.py 的 ``gt > 0``）；
  clip 只保證 ≥8 幀有效 ⇒ 下游一律帶 per-frame valid mask。
* identity clip ≡ ``min_iou >= 0.5``（build_hard_clips.py:131 的定義；
  JSON 頂層的 identity_clips 只是計數、沒有 per-clip 布林欄位）。

影像前處理逐行對照 sam3/model/utils/sam2_utils.py 的 _load_img_as_tensor ＋
loader 端 normalize：PIL convert("RGB") → resize((S,S))（PIL 預設 BICUBIC）
→ /255 → CHW → −0.5 → /0.5。訓練與推論同構，避免 domain gap。
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Optional

IDENTITY_MIN_IOU = 0.5  # build_hard_clips.py 的 FAIL_IOU；min_iou >= 此值 ⇒ identity clip
IMG_MEAN = 0.5
IMG_STD = 0.5


def load_clip_index(json_path) -> dict:
    with open(json_path, "r", encoding="utf-8") as f:
        index = json.load(f)
    assert "clips" in index and index["clips"], f"{json_path} 缺 clips"
    return index


def load_gt_csv(csv_path) -> dict:
    """2026training.csv → {frame_id: (x, y, w, h)}。標頭 ID,x,y,width,height。"""
    gt = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        assert header[0].strip().lower() == "id", f"非預期標頭 {header}"
        for row in reader:
            if not row or not row[0]:
                continue
            gt[row[0]] = tuple(float(v) for v in row[1:5])
    return gt


def sample_positions(start: int, clip_len: int, t: int, stride: int) -> list:
    """clip 內取 T 幀：start, start+stride, ...；必須落在 clip 範圍內。"""
    assert (t - 1) * stride < clip_len, f"T={t}×stride={stride} 超出 clip_len={clip_len}"
    return [start + i * stride for i in range(t)]


def frame_valid(box: Optional[tuple]) -> bool:
    """GT 有效性，同 build_hard_clips.py 的 ``np.all(gt > 0)``。缺行視為無效。"""
    return box is not None and all(v > 0 for v in box)


def select_g1_clips(clips: list, gt_by_id: dict, n: int = 10,
                    t: int = 8, stride: int = 2) -> list:
    """G1 overfit 用 clip 選取——確定性規則，寫死（事前定案 2026-08-29）：

    1. 排除 identity clips（min_iou >= 0.5：tracker 已全程跟對，梯度資訊少）。
    2. 排除「採樣後首幀 GT 無效」或「採樣後有效幀 < 6」的 clip
       （首幀是 box prompt 的來源，必須有效）。
    3. 依 (failing_frames 降冪, mean_iou 升冪, sequence, start_position) 排序
       ——失敗幀最多、整體最差的優先；後兩鍵保證確定性。
    4. 每個 sequence 最多取 1 支（多樣性），取前 n 支。
    """
    def usable(c):
        if c["min_iou"] >= IDENTITY_MIN_IOU:
            return False
        pos = sample_positions(0, c["length"], t, stride)  # clip 內相對位置
        boxes = [gt_by_id.get(c["frame_ids"][p]) for p in pos]
        if not frame_valid(boxes[0]):
            return False
        return sum(1 for b in boxes if frame_valid(b)) >= 6

    ranked = sorted(
        (c for c in clips if usable(c)),
        key=lambda c: (-c["failing_frames"], c["mean_iou"], c["sequence"], c["start_position"]),
    )
    chosen, seen_seq = [], set()
    for c in ranked:
        if c["sequence"] in seen_seq:
            continue
        chosen.append(c)
        seen_seq.add(c["sequence"])
        if len(chosen) == n:
            break
    if len(chosen) < n:
        raise ValueError(f"可用 clip 不足：要 {n} 支只選到 {len(chosen)}")
    return chosen


def _load_frame_tensor(img_path, image_size: int):
    """單幀載入。逐行對照 sam2_utils._load_img_as_tensor ＋ normalize。
    回傳 (tensor [3,S,S] float32 已 normalize, 原圖 W, 原圖 H)。"""
    import numpy as np
    import torch
    from PIL import Image

    img_pil = Image.open(img_path)
    img_np = np.array(img_pil.convert("RGB").resize((image_size, image_size)))
    if img_np.dtype != np.uint8:
        raise RuntimeError(f"Unknown image dtype: {img_np.dtype} on {img_path}")
    img = torch.from_numpy(img_np / 255.0).permute(2, 0, 1).float()
    video_width, video_height = img_pil.size
    img = (img - IMG_MEAN) / IMG_STD
    return img, video_width, video_height


class ClipDataset:
    """G1 clip dataset。real 模式讀磁碟；synthetic 模式不碰磁碟（CPU 單元測試）。

    __getitem__ 回傳 dict：
      images          float32 [T,3,S,S]（已 normalize）
      gt_boxes_model  float32 [T,4] xywh，模型座標（原圖 → S×S 的非等比縮放）
      valid           bool    [T]（GT 有效幀）
      init_box_rel    float32 [1,4] xyxy 正規化（首幀 GT，供 box prompt；
                      對照 track_t1.py:177 的 rel_box 慣例）
      meta            dict（sequence / start_position / positions / frame_ids）
    """

    def __init__(self, clips: list, gt_by_id: dict, frames_root=None,
                 t: int = 8, stride: int = 2, image_size: int = 1008,
                 synthetic: bool = False, synthetic_seed: int = 0):
        self.clips = clips
        self.gt_by_id = gt_by_id
        self.frames_root = Path(frames_root) if frames_root else None
        self.t = t
        self.stride = stride
        self.image_size = image_size
        self.synthetic = synthetic
        self.synthetic_seed = synthetic_seed
        if not synthetic:
            assert self.frames_root is not None, "real 模式需要 frames_root"

    @classmethod
    def make_synthetic(cls, n_clips: int = 3, t: int = 8, stride: int = 1,
                       image_size: int = 64, seed: int = 0):
        """合成資料：灰底上水平移動的白方塊，GT 精確已知。clip/GT 結構與
        real 模式同構（含一支 identity、一幀無效 GT，覆蓋選取規則分支）。"""
        clip_len = t * stride if stride > 1 else max(t, 16)
        clips, gt = [], {}
        for ci in range(n_clips):
            seq = f"vis-synth{ci}"
            ids = [f"{seq}_{100 + i}" for i in range(clip_len)]
            for i, fid in enumerate(ids):
                x = 4.0 + 2.0 * i + ci
                gt[fid] = (x, 10.0 + ci, 12.0, 12.0)
            if ci == 1 and clip_len > 3:
                gt[ids[3]] = (0.0, 0.0, 0.0, 0.0)  # 無效 GT 幀
            clips.append({
                "sequence": seq, "modality": "vis", "start_position": 0,
                "length": clip_len, "frame_ids": ids,
                "mean_iou": 0.9 if ci == 0 else 0.3,   # ci=0 是 identity
                "min_iou": 0.9 if ci == 0 else 0.1,
                "failing_frames": 0 if ci == 0 else clip_len // 2,
                "frozen_frames": 0, "fold": 0,
            })
        return cls(clips, gt, frames_root=None, t=t, stride=stride,
                   image_size=image_size, synthetic=True, synthetic_seed=seed)

    def __len__(self):
        return len(self.clips)

    def _frame_paths(self, clip, positions):
        seq_dir = self.frames_root / clip["sequence"]
        frames = sorted(seq_dir.glob("*.jp*g"))
        assert frames, f"{seq_dir} 無影像"
        paths = []
        for p in positions:
            idx = clip["start_position"] + p
            assert idx < len(frames), (
                f"{clip['sequence']}: start_position+{p}={idx} 超出 {len(frames)} 張"
            )
            paths.append(frames[idx])
        return paths

    def _synthetic_frame(self, box_xywh):
        import numpy as np
        import torch

        s = self.image_size
        img = np.full((s, s, 3), 64, dtype=np.uint8)
        if frame_valid(box_xywh):
            x, y, w, h = (int(round(v)) for v in box_xywh)
            img[max(y, 0):min(y + h, s), max(x, 0):min(x + w, s)] = 220
        t = torch.from_numpy(img / 255.0).permute(2, 0, 1).float()
        return (t - IMG_MEAN) / IMG_STD

    def __getitem__(self, i: int) -> dict:
        import torch

        clip = self.clips[i]
        positions = sample_positions(0, clip["length"], self.t, self.stride)
        frame_ids = [clip["frame_ids"][p] for p in positions]
        boxes_img = [self.gt_by_id.get(fid) for fid in frame_ids]
        valid = [frame_valid(b) for b in boxes_img]
        assert valid[0], f"{clip['sequence']}: 首幀 GT 無效，select_g1_clips 應已排除"

        s = self.image_size
        if self.synthetic:
            # 合成影像座標＝模型座標（本來就是 S×S 畫布），W=H=S。
            images = torch.stack([self._synthetic_frame(b) for b in boxes_img])
            w_img = h_img = float(s)
        else:
            tensors, w_img, h_img = [], None, None
            for p_path in self._frame_paths(clip, positions):
                t_img, w0, h0 = _load_frame_tensor(p_path, s)
                if w_img is None:
                    w_img, h_img = float(w0), float(h0)
                else:
                    assert (float(w0), float(h0)) == (w_img, h_img), \
                        f"{clip['sequence']} 內幀尺寸不一致"
                tensors.append(t_img)
            images = torch.stack(tensors)

        sx, sy = s / w_img, s / h_img
        gt_model = torch.zeros(self.t, 4, dtype=torch.float32)
        for j, b in enumerate(boxes_img):
            if valid[j]:
                gt_model[j] = torch.tensor(
                    [b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy], dtype=torch.float32)

        x, y, w, h = boxes_img[0]
        init_box_rel = torch.tensor(
            [[x / w_img, y / h_img, (x + w) / w_img, (y + h) / h_img]],
            dtype=torch.float32)
        assert float(init_box_rel.max()) <= 1.5, f"init box 正規化異常 {init_box_rel.tolist()}"

        return {
            "images": images,
            "gt_boxes_model": gt_model,
            "valid": torch.tensor(valid, dtype=torch.bool),
            "init_box_rel": init_box_rel,
            "meta": {"sequence": clip["sequence"],
                     "start_position": clip["start_position"],
                     "positions": positions, "frame_ids": frame_ids},
        }
