#!/usr/bin/env python3
"""E12 mask→box 精修消融（v2 路線 D032；純 CPU，離線吃 track_t1.py 的 mask 快取）。

變體（D033 紀律：個位數候選、不網格搜索）：
  base      nonzero 外接（重建 E02 邏輯 —— 回歸不變量，應與 base-csv 逐幀一致）
  cc        最大連通域主體選擇
  close3_cc binary_closing(3x3) 後取最大連通域
  close5_cc binary_closing(5x5) 後取最大連通域
  cc_ema    cc + 框尺寸時序 EMA（alpha=0.5，僅 w/h；中心跟隨當幀 mask）
  cc_pad1s  cc + 小目標序列（GT 首幀 sqrt(w*h)<32）各邊外擴 1px

與 E02 對齊的不變式：首幀恆 init（不精修）、空 mask 幀沿用該變體前一框、
缺幀尾端沿用最後框。

用法：
  python3.14 -m hsot.box_refine --masks-dir out_t1/masks --base-csv out_t1/submission.csv \
      --gt-csv 1_data/raw/2026training.csv --out-dir 5_outputs/e12_refine
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

from . import eval as ev

STABLE_THRESHOLD = 0.75  # base AUC ≥ 此值的序列視為「穩定組」，精修不得使其退步


def load_cache(npz_path: Path):
    z = np.load(npz_path)
    n = int(z["n"])
    if n == 0:
        return None
    h, w = int(z["height"]), int(z["width"])
    masks = np.unpackbits(z["bits"], count=n * h * w).reshape(n, h, w).astype(bool)
    return masks, z["boxes"], z["empty"]


def nonzero_box(mask: np.ndarray):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1 - xs.min()), float(ys.max() + 1 - ys.min())]


def largest_cc(mask: np.ndarray) -> np.ndarray:
    lab, k = ndimage.label(mask)
    if k <= 1:
        return mask
    sizes = ndimage.sum_labels(np.ones_like(lab), lab, index=range(1, k + 1))
    return lab == (int(np.argmax(sizes)) + 1)


def variant_mask(mask: np.ndarray, variant: str) -> np.ndarray:
    if variant.startswith("close3"):
        mask = ndimage.binary_closing(mask, structure=np.ones((3, 3), bool))
    elif variant.startswith("close5"):
        mask = ndimage.binary_closing(mask, structure=np.ones((5, 5), bool))
    if variant == "base":
        return mask
    return largest_cc(mask)


def refine_sequence(masks, empty, init_box, n_expect, variant, small: bool, hw):
    """回傳 n_expect 個 box。位置 0 恆 init；空 mask 沿前框；尾端補最後框。"""
    H, W = hw
    boxes = [list(init_box)]
    prev = list(init_box)
    ema_wh = None
    for pos in range(1, len(masks)):
        if empty[pos]:
            box = list(prev)
        else:
            b = nonzero_box(variant_mask(masks[pos], variant))
            box = list(prev) if b is None else b
            if b is not None:
                if variant == "cc_ema":
                    cx, cy = box[0] + box[2] / 2, box[1] + box[3] / 2
                    ema_wh = [box[2], box[3]] if ema_wh is None else \
                        [0.5 * box[2] + 0.5 * ema_wh[0], 0.5 * box[3] + 0.5 * ema_wh[1]]
                    box = [cx - ema_wh[0] / 2, cy - ema_wh[1] / 2, ema_wh[0], ema_wh[1]]
                if variant == "cc_pad1s" and small:
                    box = [max(0.0, box[0] - 1), max(0.0, box[1] - 1),
                           min(W - max(0.0, box[0] - 1), box[2] + 2),
                           min(H - max(0.0, box[1] - 1), box[3] + 2)]
        boxes.append(box)
        prev = box
    while len(boxes) < n_expect:
        boxes.append(list(boxes[-1]))
    return boxes[:n_expect]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--masks-dir", required=True)
    ap.add_argument("--base-csv", required=True, help="track_t1.py 的 submission.csv（幀號權威來源）")
    ap.add_argument("--gt-csv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--variants", default="base,cc,close3_cc,close5_cc,cc_ema,cc_pad1s")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    base = pd.read_csv(args.base_csv)
    parts = base["ID"].str.rsplit("_", n=1, expand=True)
    base["seq"], base["frame"] = parts[0], parts[1].astype(int)

    gt = pd.read_csv(args.gt_csv)
    gt.columns = ["ID", "x", "y", "w", "h"]
    gp = gt["ID"].str.rsplit("_", n=1, expand=True)
    gt["seq"], gt["frame"] = gp[0], gp[1].astype(int)
    first = gt.sort_values("frame").groupby("seq").first()
    small_seqs = set(first.index[np.sqrt(first["w"] * first["h"]) < 32])

    seqs = sorted(base["seq"].unique())
    variants = args.variants.split(",")
    per_variant_rows = {v: [] for v in variants}
    for seq in seqs:
        sub = base[base["seq"] == seq].sort_values("frame")
        ids = sub["ID"].tolist()
        init_box = sub.iloc[0][["x", "y", "width", "height"]].astype(float).tolist()
        cache = load_cache(Path(args.masks_dir) / f"{seq}.npz")
        if cache is None:  # 無 mask（單幀序列等）：所有變體照抄 base
            for v in variants:
                per_variant_rows[v] += list(sub[["ID", "x", "y", "width", "height"]].itertuples(index=False, name=None))
            continue
        masks, _, empty = cache
        hw = masks.shape[1:]
        for v in variants:
            boxes = refine_sequence(masks, empty, init_box, len(ids), v, seq in small_seqs, hw)
            per_variant_rows[v] += [(ids[i], *boxes[i]) for i in range(len(ids))]

    report = {}
    per_seq_tables = {}
    for v in variants:
        df = pd.DataFrame(per_variant_rows[v], columns=["ID", "x", "y", "width", "height"])
        csv_path = out / f"sub_{v}.csv"
        df.to_csv(csv_path, index=False)
        r = ev.evaluate(csv_path, args.gt_csv, seqs)
        report[v] = {"pooled_auc": r["pooled"]["auc"], "dp20": r["pooled"]["dp20"]}
        per_seq_tables[v] = {s: d["auc"] for s, d in r["per_seq"].items()}

    base_auc = report["base"]["pooled_auc"]
    base_seq = per_seq_tables["base"]
    stable = {s for s, a in base_seq.items() if a >= STABLE_THRESHOLD}
    print(f"\n=== E12 消融（{len(seqs)} 序列；base pooled {base_auc:.5f}）===")
    print(f"{'variant':<10} {'pooled':>8} {'Δ':>8}  {'改善':>4} {'退步':>4}  {'穩定組最壞Δ':>10}")
    for v in variants:
        d = report[v]["pooled_auc"] - base_auc
        deltas = {s: per_seq_tables[v][s] - base_seq[s] for s in seqs}
        imp = sum(1 for x in deltas.values() if x > 0.002)
        reg = sum(1 for x in deltas.values() if x < -0.002)
        worst_stable = min((deltas[s] for s in stable), default=0.0)
        print(f"{v:<10} {report[v]['pooled_auc']:>8.5f} {d:>+8.5f}  {imp:>4} {reg:>4}  {worst_stable:>+10.5f}")
        report[v]["delta"] = d
        report[v]["improved"] = imp
        report[v]["regressed"] = reg
        report[v]["worst_stable_delta"] = worst_stable
        report[v]["per_seq_delta"] = {s: round(deltas[s], 5) for s in sorted(deltas, key=deltas.get)}

    (out / "e12_report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(f"\n報表 → {out}/e12_report.json（含逐序列 delta 全表）")


if __name__ == "__main__":
    main()
