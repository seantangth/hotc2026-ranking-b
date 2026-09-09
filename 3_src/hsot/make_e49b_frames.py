#!/usr/bin/env python3
"""E49b 全序列連續窗。規則寫死在 E49B_FULLSEQ_DESIGN。"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "3_src" / "peft" / "e49b_frames.json"

SEQS = ["nir-fake_orange", "nir-pingpong", "vis-S_jump2", "nir-yo_yo", "vis-officefan2"]
CONTROL = {"vis-officefan2"}


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


def main():
    seqs = set(SEQS)
    pred = load_boxes(ROOT / "5_outputs/e46_sam3_crop_scores_20260819/val_merged.csv", seqs)
    gt = load_boxes(ROOT / "1_data/raw/2026training.csv", seqs)
    out = {
        "seqs": {},
        "policy": {
            "disagree_tau": 0.30, "agree_tau": 0.50, "K": 3, "M": 1,
            "A": "e46_crop_sam3", "mode": "hysteresis_run_splice",
        },
    }
    n = 0
    for seq in SEQS:
        g, p = gt[seq], pred[seq]
        first = min(g)
        init = g[first]
        common = sorted((set(g) & set(p)) - {first})
        assert common[-1] - common[0] + 1 == len(common), seq
        frz = frozen_runs({k: np.array(v) for k, v in p.items()})
        out["seqs"][seq] = {
            "group": "control" if seq in CONTROL else "lesion",
            "first": first,
            "init_box": init,
            "windows": [{"id": "full", "kind": "full", "frames": common}],
            "gt": {str(f): g[f] for f in common + [first]},
            "pred": {str(f): p[f] for f in common},
            "frozen": {str(f): int(frz.get(f, 0)) for f in common},
        }
        n += len(common)
        print(f"{seq:24s} {out['seqs'][seq]['group']:7s} {common[0]}-{common[-1]} n={len(common)}")
    out["n_probe_frames"] = n
    print("n_probe_frames", n)
    OUT.write_text(json.dumps(out, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
