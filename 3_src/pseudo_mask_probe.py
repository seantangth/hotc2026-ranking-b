#!/usr/bin/env python3
"""E16 前測：偽 mask 品質探針（原定 W2,提前執行）。

問題:E16(可學光譜 adapter + 偽 mask 自訓練)的第一個 gate 是
「用 GT box 逐幀 prompt SAM2 產生的偽 mask,其最緊外接框對 GT box 的
IoU(box-tightness)平均 ≥ 0.80」——不達標代表偽標籤品質不足以監督訓練
(garbage in garbage out),E16 直接 no-go,省下整條訓練線的投資。

做法:每序列均勻抽 K 幀(含首幀),SAM2ImagePredictor + GT box prompt,
取最高信心 mask → 最緊外接框 vs GT box IoU。純 image 模式,不碰 video/記憶。

用法(在已裝 samurai/sam2 的機器上):
  python pseudo_mask_probe.py --frames-root ~/t1_data --gt-csv ~/2026training.csv \
      --seq-list ~/val_split_v1.txt --samurai-dir ~/samurai \
      --ckpt ~/ckpt/sam2.1_hiera_large.pt --per-seq 8 --out probe_result.json
"""
from __future__ import annotations

import argparse
import json
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-root", required=True)
    ap.add_argument("--gt-csv", required=True)
    ap.add_argument("--seq-list", required=True)
    ap.add_argument("--samurai-dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--per-seq", type=int, default=8)
    ap.add_argument("--out", default="probe_result.json")
    args = ap.parse_args()

    import torch
    from PIL import Image

    os.chdir(Path(args.samurai_dir) / "sam2")
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    model = build_sam2("configs/sam2.1/sam2.1_hiera_l.yaml", str(Path(args.ckpt).resolve()), device="cuda:0")
    predictor = SAM2ImagePredictor(model)

    gt = pd.read_csv(args.gt_csv)
    gt.columns = ["ID", "x", "y", "w", "h"]
    parts = gt["ID"].str.rsplit("_", n=1, expand=True)
    gt["seq"], gt["frame"] = parts[0], parts[1].astype(int)

    seqs = [s.strip() for s in Path(args.seq_list).read_text().split() if s.strip()]
    frames_root = Path(args.frames_root)
    per_seq = {}
    all_ious = []
    for si, seq in enumerate(seqs):
        seq_dir = frames_root / seq
        jpgs = sorted(seq_dir.glob("*.jpg"))
        rows = gt[gt["seq"] == seq].sort_values("frame").reset_index(drop=True)
        n = min(len(jpgs), len(rows))
        if n == 0:
            per_seq[seq] = {"error": "no frames or no gt"}
            continue
        # 均勻抽樣(含首幀);跳過 GT 無效幀(任一值 ≤0,官方遮蔽/邊界語意)
        picks = sorted(set(np.linspace(0, n - 1, args.per_seq, dtype=int).tolist()))
        ious = []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for pos in picks:
                r = rows.iloc[pos]
                g = [float(r["x"]), float(r["y"]), float(r["w"]), float(r["h"])]
                if min(g) <= 0:
                    continue
                img = np.array(Image.open(jpgs[pos]).convert("RGB"))
                predictor.set_image(img)
                box_xyxy = np.array([g[0], g[1], g[0] + g[2], g[1] + g[3]])
                masks, scores, _ = predictor.predict(box=box_xyxy, multimask_output=False)
                m = masks[0].astype(bool)
                ys, xs = np.nonzero(m)
                if len(xs) == 0:
                    ious.append(0.0)
                    continue
                pb = [float(xs.min()), float(ys.min()), float(xs.max() + 1 - xs.min()), float(ys.max() + 1 - ys.min())]
                ious.append(box_iou(pb, g))
        mod = "rednir" if seq.startswith("rednir-") else ("nir" if seq.startswith("nir-") else "vis")
        per_seq[seq] = {"n_probed": len(ious), "mean_iou": float(np.mean(ious)) if ious else None, "modality": mod}
        all_ious += [(mod, v) for v in ious]
        print(f"[{si+1}/{len(seqs)}] {seq}: mean box-tightness {np.mean(ious):.3f} ({len(ious)} 幀)" if ious else f"[{si+1}/{len(seqs)}] {seq}: 無有效幀")

    vals = [v for _, v in all_ious]
    by_mod = {m: [v for mm, v in all_ious if mm == m] for m in ("vis", "nir", "rednir")}
    summary = {
        "overall_mean": float(np.mean(vals)),
        "overall_median": float(np.median(vals)),
        "frac_below_0.5": float(np.mean([v < 0.5 for v in vals])),
        "frac_below_0.8": float(np.mean([v < 0.8 for v in vals])),
        "by_modality_mean": {m: (float(np.mean(x)) if x else None) for m, x in by_mod.items()},
        "n_frames_probed": len(vals),
        "gate": "PASS(≥0.80)" if np.mean(vals) >= 0.80 else "FAIL(<0.80)",
        "per_seq": per_seq,
    }
    Path(args.out).write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(f"\n=== E16 偽 mask 品質前測 ===")
    print(f"overall mean {summary['overall_mean']:.4f} | median {summary['overall_median']:.4f} | "
          f"<0.8 佔 {summary['frac_below_0.8']:.1%} | 模態 {summary['by_modality_mean']}")
    print(f"GATE: {summary['gate']} → {args.out}")


if __name__ == "__main__":
    main()
