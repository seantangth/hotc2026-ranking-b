#!/usr/bin/env python3
"""E51 時間線樣本。區間寫死，與 DESIGN 一致。"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "3_src" / "peft" / "e51_frames.json"

LOCKED = {
    "nir-fake_orange": (3282, 3695),
    "nir-pingpong": (11933, 12350),
    "nir-yo_yo": (21959, 22519),
    "vis-S_jump2": (113921, 114063),
    "vis-officefan2": (90686, 90765),
}
CONTROL = {"vis-officefan2"}


def iou(a, b):
    iw = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


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
    seqs = set(LOCKED)
    pred = load_boxes(ROOT / "5_outputs/e46_sam3_crop_scores_20260819/val_merged.csv", seqs)
    gt = load_boxes(ROOT / "1_data/raw/2026training.csv", seqs)
    out = {
        "seqs": {},
        "policy": {
            "disagree_tau": 0.30, "agree_tau": 0.50,
            "A": "e46_crop_sam3", "mode": "update_on_agree_then_perframe_splice",
        },
        "locked": {s: list(v) for s, v in LOCKED.items()},
    }
    n = 0
    for seq, (lo, hi) in LOCKED.items():
        g, p = gt[seq], pred[seq]
        first = min(g)
        frames = [f for f in sorted(set(g) & set(p)) if lo <= f <= hi and f != first]
        assert frames, seq
        kinds = {}
        ng = nw = 0
        for f in frames:
            ia = iou(p[f], g[f])
            if ia >= 0.5:
                kinds[str(f)] = "alive_good"; ng += 1
            elif ia < 0.1:
                kinds[str(f)] = "alive_wrong"; nw += 1
            else:
                kinds[str(f)] = "mid"
        out["seqs"][seq] = {
            "group": "control" if seq in CONTROL else "lesion",
            "first": first,
            "init_box": g[first],
            "windows": [{"id": "tl0", "kind": "timeline", "frames": frames}],
            "kind": kinds,
            "gt": {str(f): g[f] for f in frames + [first]},
            "pred": {str(f): p[f] for f in frames},
        }
        n += len(frames)
        holes = sum(1 for a, b in zip(frames, frames[1:]) if b != a + 1)
        print(f"{seq:22s} {frames[0]}-{frames[-1]} n={len(frames)} holes={holes} good={ng} wrong={nw}")
    out["n_probe_frames"] = n
    print("n_probe_frames", n)
    OUT.write_text(json.dumps(out, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
