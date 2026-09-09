#!/usr/bin/env python3
"""E16 teacher 偽 mask 生成器:GT box 逐(抽)幀 prompt SAM2 → 高品質偽 mask 訓練標籤。

前測(probe_result.json)已驗證:box-tightness mean 0.857、三模態均勻 → GATE PASS。
本工具把它產品化:對訓練序列抽幀生成 teacher mask,只保留 tightness ≥ 門檻的幀
(前測顯示過濾掉 <0.8 的 20.3% 後品質更高)。

輸出:每序列一個 npz——frames(抽中幀位置)、boxes(GT)、tightness、
masks(packbits)——供 E16 adapter 訓練直接當監督。

用法(在已裝 samurai/sam2 的機器上):
  python3 teacher_gen.py --frames-root <假色根> --gt-csv 2026training.csv \
      --seq-list <清單> --samurai-dir ~/samurai --ckpt <sam2.1-L> \
      --stride 5 --min-tightness 0.8 --out-dir teacher_masks
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


def box_iou(a, b) -> float:
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx2, by2 = b[0] + b[2], b[1] + b[3]
    ix = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    iy = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = ix * iy
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def main() -> None:
    import torch
    from PIL import Image

    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-root", required=True)
    ap.add_argument("--gt-csv", required=True)
    ap.add_argument("--seq-list", required=True)
    ap.add_argument("--samurai-dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--min-tightness", type=float, default=0.8)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    os.chdir(Path(args.samurai_dir) / "sam2")
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    model = build_sam2("configs/sam2.1/sam2.1_hiera_l.yaml", str(Path(args.ckpt).resolve()), device="cuda:0")
    predictor = SAM2ImagePredictor(model)

    gt = pd.read_csv(args.gt_csv)
    gt.columns = ["ID", "x", "y", "w", "h"]
    parts = gt["ID"].str.rsplit("_", n=1, expand=True)
    gt["seq"], gt["frame"] = parts[0], parts[1].astype(int)

    out_root = Path(args.out_dir); out_root.mkdir(parents=True, exist_ok=True)
    seqs = [s.strip() for s in Path(args.seq_list).read_text().split() if s.strip()]
    for si, seq in enumerate(seqs):
        out_f = out_root / f"{seq}.npz"
        if out_f.exists():
            continue
        jpgs = sorted((Path(args.frames_root) / seq).glob("*.jpg"))
        rows = gt[gt["seq"] == seq].sort_values("frame").reset_index(drop=True)
        n = min(len(jpgs), len(rows))
        keep_pos, keep_boxes, keep_tight, keep_masks = [], [], [], []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for pos in range(0, n, args.stride):
                r = rows.iloc[pos]
                g = [float(r["x"]), float(r["y"]), float(r["w"]), float(r["h"])]
                if min(g) <= 0:  # 官方無效幀(遮蔽/邊界)跳過
                    continue
                img = np.array(Image.open(jpgs[pos]).convert("RGB"))
                predictor.set_image(img)
                bx = np.array([g[0], g[1], g[0] + g[2], g[1] + g[3]])
                masks, scores, _ = predictor.predict(box=bx, multimask_output=False)
                m = masks[0].astype(bool)
                ys, xs = np.nonzero(m)
                if len(xs) == 0:
                    continue
                pb = [float(xs.min()), float(ys.min()), float(xs.max() + 1 - xs.min()), float(ys.max() + 1 - ys.min())]
                t = box_iou(pb, g)
                if t < args.min_tightness:
                    continue
                keep_pos.append(pos); keep_boxes.append(g); keep_tight.append(t); keep_masks.append(m)
        if keep_masks:
            M = np.stack(keep_masks)
            np.savez_compressed(out_f, frames=np.array(keep_pos), boxes=np.array(keep_boxes, np.float32),
                                tightness=np.array(keep_tight, np.float32),
                                bits=np.packbits(M, axis=None), n=M.shape[0], height=M.shape[1], width=M.shape[2])
        print(f"[{si+1}/{len(seqs)}] {seq}: 抽 {len(range(0, n, args.stride))} 幀 → 合格 {len(keep_masks)}(tightness≥{args.min_tightness})")
    print("TEACHER-GEN-DONE")


if __name__ == "__main__":
    main()
