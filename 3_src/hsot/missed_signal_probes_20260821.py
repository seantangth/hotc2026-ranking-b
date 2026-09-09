#!/usr/bin/env python3
"""08-21 零 GPU 漏挖前測：crop 貼邊、首幀 mask 吃第二實例、同幀 init vs SAM 框。"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]


def load_csv(p):
    d = defaultdict(dict)
    with open(p) as f:
        for r in csv.DictReader(f):
            s, fr = r["ID"].rsplit("_", 1)
            d[s][int(fr)] = np.array(
                [float(r["x"]), float(r["y"]), float(r["width"]), float(r["height"])]
            )
    return d


def load_gt():
    seqs = {s.strip() for s in (ROOT / "1_data/val_split_v1.txt").read_text().splitlines() if s.strip()}
    d = defaultdict(dict)
    with open(ROOT / "1_data/raw/2026training.csv") as f:
        for r in csv.DictReader(f):
            s, fr = r["ID"].rsplit("_", 1)
            if s not in seqs:
                continue
            w, h = float(r["width"]), float(r["height"])
            if w <= 0 or h <= 0:
                continue
            d[s][int(fr)] = np.array([float(r["x"]), float(r["y"]), w, h])
    return d


def iou(a, b):
    iw = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def margins(box, win):
    x, y, w, h = box
    x1, y1, ww, hh = win["x1"], win["y1"], win["w"], win["h"]
    x2, y2 = x1 + ww, y1 + hh
    return {
        "left": float(x - x1),
        "top": float(y - y1),
        "right": float(x2 - (x + w)),
        "bottom": float(y2 - (y + h)),
    }


def probe_crop_margin(gt, pred, windows, touch_px=4.0):
    rows = []
    iou_touch, iou_free = [], []
    for seq, win in windows.items():
        if seq not in pred or seq not in gt:
            continue
        for f, b in pred[seq].items():
            if f not in gt[seq]:
                continue
            m = margins(b, win)
            mn = min(m.values())
            touch = mn < touch_px
            v = iou(b, gt[seq][f])
            rec = {"seq": seq, "frame": f, "min_margin": mn, "touch": touch, "iou": v, **m}
            rows.append(rec)
            (iou_touch if touch else iou_free).append(v)
    def stats(xs):
        a = np.asarray(xs, float)
        return None if a.size == 0 else {
            "n": int(a.size),
            "mean": float(a.mean()),
            "p50": float(np.median(a)),
            "lt01": float((a < 0.1).mean()),
        }
    return {
        "touch_px": touch_px,
        "n": len(rows),
        "touch_frac": float(np.mean([r["touch"] for r in rows])) if rows else None,
        "iou_touch": stats(iou_touch),
        "iou_free": stats(iou_free),
        "delta_mean_touch_minus_free": (
            None if not iou_touch or not iou_free
            else float(np.mean(iou_touch) - np.mean(iou_free))
        ),
        "worst_touch_seqs": sorted(
            (
                {
                    "seq": s,
                    "n_touch": int(sum(1 for r in rows if r["seq"] == s and r["touch"])),
                    "n": int(sum(1 for r in rows if r["seq"] == s)),
                    "iou_touch_mean": float(np.mean([r["iou"] for r in rows if r["seq"] == s and r["touch"]]))
                    if any(r["seq"] == s and r["touch"] for r in rows) else None,
                }
                for s in {r["seq"] for r in rows}
            ),
            key=lambda z: -(z["n_touch"] or 0),
        )[:8],
    }


def unpack_mask0(npz_path):
    z = np.load(npz_path)
    n, h, w = int(z["n"]), int(z["height"]), int(z["width"])
    bits = np.unpackbits(z["bits"])[: n * h * w].reshape(n, h, w).astype(bool)
    return bits[0], z["boxes"][0], bool(z["empty"][0]), h, w


def probe_frame0_mask(gt):
    mask_dir = ROOT / "5_outputs/t1_rerun_20260805/masks"
    out = []
    for p in sorted(mask_dir.glob("*.npz")):
        seq = p.stem
        if seq not in gt:
            continue
        m0, box0, empty, H, W = unpack_mask0(p)
        # init = first GT frame (usually 1)
        f0 = min(gt[seq])
        g = gt[seq][f0]
        labeled, ncc = ndimage.label(m0)
        # mass inside vs outside GT (1px dilated)
        gx1, gy1 = int(np.floor(g[0])), int(np.floor(g[1]))
        gx2, gy2 = int(np.ceil(g[0] + g[2])), int(np.ceil(g[1] + g[3]))
        gx1, gy1 = max(0, gx1), max(0, gy1)
        gx2, gy2 = min(W, gx2), min(H, gy2)
        inside = np.zeros_like(m0)
        inside[gy1:gy2, gx1:gx2] = True
        area = max(int(m0.sum()), 1)
        outside_frac = float((m0 & ~inside).sum() / area)
        # largest CC overlap with GT
        extra_cc = 0
        extra_area = 0
        for cid in range(1, ncc + 1):
            cc = labeled == cid
            if not (cc & inside).any() and cc.sum() >= 8:
                extra_cc += 1
                extra_area += int(cc.sum())
        sam_vs_init = {
            "dw": float(box0[2] - g[2]),
            "dh": float(box0[3] - g[3]),
            "dx": float(box0[0] - g[0]),
            "dy": float(box0[1] - g[1]),
            "iou": float(iou(box0, g)),
        }
        out.append({
            "seq": seq,
            "empty0": empty,
            "n_cc": int(ncc),
            "outside_frac": outside_frac,
            "extra_cc": extra_cc,
            "extra_area": extra_area,
            "mod": seq.split("-", 1)[0],
            **{f"box_{k}": v for k, v in sam_vs_init.items()},
        })
    extra = [r for r in out if r["extra_cc"] > 0]
    leak = [r for r in out if r["outside_frac"] > 0.15]
    dw = np.array([r["box_dw"] for r in out])
    dh = np.array([r["box_dh"] for r in out])
    by_mod = {}
    for mod in ("nir", "rednir", "vis"):
        sub = [r for r in out if r["mod"] == mod]
        if not sub:
            continue
        by_mod[mod] = {
            "n": len(sub),
            "dw_med": float(np.median([r["box_dw"] for r in sub])),
            "dh_med": float(np.median([r["box_dh"] for r in sub])),
            "dx_med": float(np.median([r["box_dx"] for r in sub])),
            "dy_med": float(np.median([r["box_dy"] for r in sub])),
            "iou_med": float(np.median([r["box_iou"] for r in sub])),
        }
    return {
        "n": len(out),
        "n_extra_cc": len(extra),
        "n_outside15": len(leak),
        "extra_seqs": [r["seq"] for r in extra],
        "leak_seqs": [r["seq"] for r in sorted(leak, key=lambda z: -z["outside_frac"])[:12]],
        "dw_med": float(np.median(dw)) if len(dw) else None,
        "dh_med": float(np.median(dh)) if len(dh) else None,
        "dw_mean": float(dw.mean()) if len(dw) else None,
        "dh_mean": float(dh.mean()) if len(dh) else None,
        "frac_dw_near_m1": float(np.mean(np.abs(dw + 1) < 0.51)) if len(dw) else None,
        "frac_dh_near_m1": float(np.mean(np.abs(dh + 1) < 0.51)) if len(dh) else None,
        "by_mod": by_mod,
        "switchy_extra": [r["seq"] for r in extra if any(k in r["seq"] for k in
                         ("drone", "herbs", "jump", "leaves", "pingpong", "yo_yo", "runner"))],
    }


def main():
    gt = load_gt()
    e46 = load_csv(ROOT / "5_outputs/e46_sam3_crop_scores_20260819/val_merged.csv")
    wins = json.loads((ROOT / "5_outputs/e46_sam3_crop_scores_20260819/val_meta.json").read_text())
    crop = probe_crop_margin(gt, e46, wins, touch_px=4.0)
    crop8 = probe_crop_margin(gt, e46, wins, touch_px=8.0)
    f0 = probe_frame0_mask(gt)
    summary = {"crop_margin_4px": crop, "crop_margin_8px": crop8, "frame0_mask_and_init": f0}
    out = ROOT / "5_outputs/missed_signal_probes_20260821.json"
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
