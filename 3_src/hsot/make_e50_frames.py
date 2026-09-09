#!/usr/bin/env python3
"""E50 獨立連續窗。區間寫死，與 DESIGN 一致，禁止重算挑段。"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "3_src" / "peft" / "e50_frames.json"
E49 = ROOT / "3_src" / "peft" / "e49_frames.json"

# 執行前寫死（E49B_FULLSEQ 取消後的獨立窗）
LOCKED = {
    "nir-fake_orange": {"wrong": (3932, 3981), "good": (3346, 3409)},
    "nir-pingpong": {"wrong": (12938, 12974), "good": (12274, 12337)},
    "vis-S_jump2": {"wrong": (114009, 114063), "good": (113921, 113956)},
    "nir-yo_yo": {"wrong": (22520, 22599), "good": (21959, 22022)},
    "vis-officefan2": {"good": (90766, 90845)},
}
CONTROL = {"vis-officefan2"}


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


def span(lo_hi):
    lo, hi = lo_hi
    xs = list(range(lo, hi + 1))
    assert xs[-1] - xs[0] + 1 == len(xs)
    return xs


def main():
    seqs = set(LOCKED)
    pred = load_boxes(ROOT / "5_outputs/e46_sam3_crop_scores_20260819/val_merged.csv", seqs)
    gt = load_boxes(ROOT / "1_data/raw/2026training.csv", seqs)
    e49_used = defaultdict(set)
    if E49.exists():
        e49 = json.loads(E49.read_text())
        for seq, c in e49["seqs"].items():
            for w in c.get("windows") or []:
                e49_used[seq].update(w["frames"])
            e49_used[seq].update(c.get("alive_wrong") or [])
            e49_used[seq].update(c.get("alive_good") or [])

    out = {
        "seqs": {},
        "policy": {
            "disagree_tau": 0.30, "agree_tau": 0.50, "K": 8, "M": 1,
            "A": "e46_crop_sam3", "mode": "hysteresis_run_splice",
        },
        "locked": {s: {k: list(v) for k, v in d.items()} for s, d in LOCKED.items()},
    }
    n = 0
    for seq, spec in LOCKED.items():
        g, p = gt[seq], pred[seq]
        first = min(g)
        init = g[first]
        windows = []
        used = []
        if "wrong" in spec:
            fr = span(spec["wrong"])
            windows.append({"id": "wrong0", "kind": "alive_wrong", "frames": fr})
            used.extend(fr)
        fr = span(spec["good"])
        windows.append({"id": "good0", "kind": "alive_good", "frames": fr})
        used.extend(fr)
        overlap = set(used) & e49_used[seq]
        assert not overlap, (seq, sorted(overlap)[:10], len(overlap))
        missing = [f for f in used if f not in g or f not in p]
        assert not missing, (seq, missing[:5])
        out["seqs"][seq] = {
            "group": "control" if seq in CONTROL else "lesion",
            "first": first,
            "init_box": init,
            "windows": windows,
            "gt": {str(f): g[f] for f in set(used) | {first}},
            "pred": {str(f): p[f] for f in used},
        }
        n += len(set(used))
        print(seq, " ".join(f"{w['kind']} {w['frames'][0]}-{w['frames'][-1]} n={len(w['frames'])}" for w in windows),
              "overlap_e49=0")
    out["n_probe_frames"] = n
    print("n_probe_frames", n)
    OUT.write_text(json.dumps(out, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
