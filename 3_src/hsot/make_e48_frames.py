#!/usr/bin/env python3
"""E48 樣本清單。執行前生成；cap 與序列寫死在本檔，與 DESIGN 一致。"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "3_src" / "peft" / "e48_frames.json"

LESION = ["nir-fake_orange", "nir-pingpong", "vis-S_jump2", "nir-yo_yo"]
CONTROL = ["vis-officefan2"]
CAP_WRONG = 80
CAP_GOOD = 80
CAP_CTRL_GOOD = 40


def iou(a, b):
    iw = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def frozen_runs(fm):
    ks = sorted(fm)
    o, r = {}, 0
    for i, k in enumerate(ks):
        r = r + 1 if i > 0 and np.allclose(fm[k], fm[ks[i - 1]]) else 0
        o[k] = r
    return o


def load_boxes(path, seqs):
    d = defaultdict(dict)
    with open(path) as f:
        for r in csv.DictReader(f):
            s, fr = r["ID"].rsplit("_", 1)
            if s not in seqs:
                continue
            w, h = float(r["width"]), float(r["height"])
            if w <= 0 or h <= 0:
                continue
            d[s][int(fr)] = [float(r["x"]), float(r["y"]), w, h]
    return d


def take_uniform(xs, n):
    if len(xs) <= n:
        return list(xs)
    idx = np.linspace(0, len(xs) - 1, num=n, dtype=int)
    return [xs[i] for i in idx]


def main():
    seqs = set(LESION + CONTROL)
    pred = load_boxes(ROOT / "5_outputs/e46_sam3_crop_scores_20260819/val_merged.csv", seqs)
    gt = load_boxes(ROOT / "1_data/raw/2026training.csv", seqs)
    out = {"seqs": {}, "caps": {"wrong": CAP_WRONG, "good": CAP_GOOD, "ctrl_good": CAP_CTRL_GOOD},
           "policy": {"disagree_tau": 0.30, "A": "e46_crop_sam3"}, "dropped": {}}
    for seq in LESION + CONTROL:
        g, p = gt[seq], pred[seq]
        first = min(g)
        init = g[first]
        frz = frozen_runs({k: np.array(v) for k, v in p.items()})
        wrong, good = [], []
        for f in sorted(set(g) & set(p)):
            if f == first:
                continue
            ia = iou(np.array(p[f]), np.array(g[f]))
            frozen = frz.get(f, 0) >= 2
            if ia < 0.1 and not frozen:
                wrong.append(f)
            elif ia >= 0.5:
                good.append(f)
        is_ctrl = seq in CONTROL
        w_use = take_uniform(wrong, CAP_WRONG if not is_ctrl else 0)
        g_use = take_uniform(good, CAP_CTRL_GOOD if is_ctrl else CAP_GOOD)
        used = set(w_use + g_use)
        gt_d = {str(f): g[f] for f in used}
        gt_d[str(first)] = init
        pred_d = {str(f): p[f] for f in used}
        out["seqs"][seq] = {
            "group": "control" if is_ctrl else "lesion",
            "first": first,
            "init_box": init,
            "alive_wrong": w_use,
            "alive_good": g_use,
            "gt": gt_d,
            "pred": pred_d,
        }
        out["dropped"][seq] = {
            "wrong_total": len(wrong), "wrong_used": len(w_use),
            "good_total": len(good), "good_used": len(g_use),
        }
        print(f"{seq:24s} {out['seqs'][seq]['group']:7s} wrong {len(wrong):4d}->{len(w_use):3d}  "
              f"good {len(good):4d}->{len(g_use):3d}  first={first}")
    n = sum(len(s["alive_wrong"]) + len(s["alive_good"]) for s in out["seqs"].values())
    out["n_probe_frames"] = n
    print("n_probe_frames", n)
    OUT.write_text(json.dumps(out, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
