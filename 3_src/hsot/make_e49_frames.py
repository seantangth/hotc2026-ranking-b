#!/usr/bin/env python3
"""E49 連續窗樣本。規則寫死在 DESIGN；本檔只執行那些數字。"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "3_src" / "peft" / "e49_frames.json"
E48 = ROOT / "3_src" / "peft" / "e48_frames.json"

LESION = ["nir-fake_orange", "nir-pingpong", "vis-S_jump2", "nir-yo_yo"]
CONTROL = ["vis-officefan2"]
CAP_WRONG = 80
CAP_GOOD = 64
CAP_CTRL_GOOD = 80


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


def runs_of(frames):
    if not frames:
        return []
    frames = sorted(frames)
    out, s, prev = [], frames[0], frames[0]
    for f in frames[1:]:
        if f == prev + 1:
            prev = f
        else:
            out.append((s, prev, prev - s + 1))
            s = prev = f
    out.append((s, prev, prev - s + 1))
    return sorted(out, key=lambda x: -x[2])


def prefix(run, cap):
    lo, hi, n = run
    take = min(n, cap)
    return list(range(lo, lo + take))


def main():
    seqs = set(LESION + CONTROL)
    pred = load_boxes(ROOT / "5_outputs/e46_sam3_crop_scores_20260819/val_merged.csv", seqs)
    gt = load_boxes(ROOT / "1_data/raw/2026training.csv", seqs)
    e48_used = defaultdict(set)
    if E48.exists():
        e48 = json.loads(E48.read_text())
        for seq, c in e48["seqs"].items():
            e48_used[seq].update(c.get("alive_wrong") or [])
            e48_used[seq].update(c.get("alive_good") or [])

    out = {
        "seqs": {},
        "caps": {"wrong": CAP_WRONG, "good": CAP_GOOD, "ctrl_good": CAP_CTRL_GOOD},
        "policy": {
            "disagree_tau": 0.30,
            "agree_tau": 0.50,
            "K": 3,
            "M": 1,
            "A": "e46_crop_sam3",
            "mode": "hysteresis_run_splice",
        },
        "windows_meta": {},
    }
    n_probe = 0
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
        wr, gr = runs_of(wrong), runs_of(good)
        windows = []
        if not is_ctrl:
            assert wr, f"{seq} 無 alive_wrong 段"
            w_frames = prefix(wr[0], CAP_WRONG)
            assert w_frames[-1] - w_frames[0] + 1 == len(w_frames)
            windows.append({"id": "wrong0", "kind": "alive_wrong", "frames": w_frames})
        assert gr, f"{seq} 無 alive_good 段"
        g_cap = CAP_CTRL_GOOD if is_ctrl else CAP_GOOD
        g_frames = prefix(gr[0], g_cap)
        assert g_frames[-1] - g_frames[0] + 1 == len(g_frames)
        windows.append({"id": "good0", "kind": "alive_good", "frames": g_frames})

        used = []
        for w in windows:
            used.extend(w["frames"])
        used_set = set(used)
        gt_d = {str(f): g[f] for f in used_set}
        gt_d[str(first)] = init
        pred_d = {str(f): p[f] for f in used_set}
        overlap = sorted(used_set & e48_used[seq])
        out["seqs"][seq] = {
            "group": "control" if is_ctrl else "lesion",
            "first": first,
            "init_box": init,
            "windows": windows,
            "gt": gt_d,
            "pred": pred_d,
        }
        out["windows_meta"][seq] = {
            "wrong_total": len(wrong),
            "good_total": len(good),
            "wrong_longest": wr[0] if wr else None,
            "good_longest": gr[0] if gr else None,
            "n_overlap_e48": len(overlap),
            "n_heldout": len(used_set) - len(overlap),
        }
        n_seq = len(used_set)
        n_probe += n_seq
        print(
            f"{seq:24s} {out['seqs'][seq]['group']:7s} "
            + " ".join(
                f"{w['kind']} {w['frames'][0]}-{w['frames'][-1]} n={len(w['frames'])}"
                for w in windows
            )
            + f"  overlap_e48={len(overlap)} heldout={n_seq - len(overlap)}"
        )
    out["n_probe_frames"] = n_probe
    print("n_probe_frames", n_probe)
    OUT.write_text(json.dumps(out, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
