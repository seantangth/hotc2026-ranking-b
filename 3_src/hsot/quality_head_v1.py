#!/usr/bin/env python3
"""E41：獨立第三者投票（ViPT）＋可學品質頭 v0（框特徵）。

品質頭不動 SAM：用 val 上 A/B 框 vs GT IoU 學「誰比較好」，套到 test 的
v029 vs v012+corr。LB 是裁決（D040）。本地只做災難檢查與逐序列診斷。
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "5_outputs" / "e41_third_and_qhead_20260817"
CORR_TOP, CORR_LEFT = 1.0, 1.0
P_STAR = 0.561


def load_csv(p):
    d = defaultdict(dict)
    order = []
    with open(p) as f:
        for r in csv.DictReader(f):
            s, fr = r["ID"].rsplit("_", 1)
            d[s][int(fr)] = np.array(
                [float(r["x"]), float(r["y"]), float(r["width"]), float(r["height"])],
                dtype=np.float64,
            )
            order.append(r["ID"])
    return d, order


def apply_corr(d, top=CORR_TOP, left=CORR_LEFT):
    out = defaultdict(dict)
    for s, fm in d.items():
        for f, b in fm.items():
            x, y, w, h = b
            nx, ny = max(0.0, x - left), max(0.0, y - top)
            out[s][f] = np.array([nx, ny, w + (x - nx), h + (y - ny)])
    return out


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


def modality(s):
    if s.startswith("nir-"):
        return 0
    if s.startswith("rednir-"):
        return 1
    return 2


def feats(a, b, prev_a, prev_b, run_a, run_b, frac, mod):
    ia = iou(a, b)
    area_a = max(a[2] * a[3], 1e-6)
    area_b = max(b[2] * b[3], 1e-6)
    asp_a = a[2] / max(a[3], 1e-6)
    asp_b = b[2] / max(b[3], 1e-6)
    ca = (a[0] + a[2] / 2, a[1] + a[3] / 2)
    cb = (b[0] + b[2] / 2, b[1] + b[3] / 2)
    cdist = np.hypot(ca[0] - cb[0], ca[1] - cb[1]) / max(np.sqrt(area_a), 1.0)
    if prev_a is None:
        va = vb = np.zeros(4)
    else:
        va = a - prev_a
        vb = b - prev_b
    return np.array(
        [
            ia,
            np.log(area_a),
            np.log(area_b),
            np.log(area_a / area_b),
            asp_a,
            asp_b,
            cdist,
            va[0],
            va[1],
            va[2],
            va[3],
            vb[0],
            vb[1],
            vb[2],
            vb[3],
            float(run_a),
            float(run_b),
            frac,
            float(mod == 0),
            float(mod == 1),
            float(mod == 2),
            a[2],
            a[3],
            b[2],
            b[3],
        ],
        dtype=np.float64,
    )


def collect_val(A, B, G, C=None):
    rows = []
    for s in sorted(set(A) & set(B) & set(G)):
        ks = sorted(set(A[s]) & set(B[s]) & set(G[s]))
        if not ks:
            continue
        ra, rb = frozen_runs(A[s]), frozen_runs(B[s])
        n = max(len(ks), 1)
        prev_a = prev_b = None
        for i, f in enumerate(ks):
            g = G[s][f]
            if g.min() <= 0:
                prev_a, prev_b = A[s][f], B[s][f]
                continue
            ia, ib = iou(A[s][f], g), iou(B[s][f], g)
            x = feats(A[s][f], B[s][f], prev_a, prev_b, ra[f], rb[f], i / n, modality(s))
            rec = {
                "seq": s,
                "frame": f,
                "x": x,
                "ia": ia,
                "ib": ib,
                "y": 1 if ia > ib else 0,
                "tie": abs(ia - ib) < 1e-9,
                "iab": iou(A[s][f], B[s][f]),
                "mod": modality(s),
            }
            if C is not None and s in C and f in C[s]:
                rec["iac"] = iou(A[s][f], C[s][f])
                rec["ibc"] = iou(B[s][f], C[s][f])
                rec["ic"] = iou(C[s][f], g)
            rows.append(rec)
            prev_a, prev_b = A[s][f], B[s][f]
    return rows


def fit_logit(X, y, l2=1e-2, steps=200, lr=0.3):
    """L2 logistic regression, numpy only."""
    w = np.zeros(X.shape[1])
    Xn = (X - X.mean(0)) / (X.std(0) + 1e-6)
    for _ in range(steps):
        z = np.clip(Xn @ w, -20, 20)
        p = 1 / (1 + np.exp(-z))
        g = Xn.T @ (p - y) / len(y) + l2 * w
        w -= lr * g
    return w, X.mean(0), X.std(0) + 1e-6


def predict_p(X, w, mu, sd):
    z = np.clip(((X - mu) / sd) @ w, -20, 20)
    return 1 / (1 + np.exp(-z))


def seq_holdout_acc(rows, disagree_only=True, margin=0.0):
    seqs = sorted({r["seq"] for r in rows})
    correct = total = 0
    deltas = []
    per_seq = []
    for hold in seqs:
        tr = [r for r in rows if r["seq"] != hold and not r["tie"]]
        te = [r for r in rows if r["seq"] == hold]
        if disagree_only:
            te = [r for r in te if r["iab"] < 0.5 and not r["tie"]]
        if not tr or not te:
            continue
        X = np.stack([r["x"] for r in tr])
        y = np.array([r["y"] for r in tr], dtype=np.float64)
        w, mu, sd = fit_logit(X, y)
        Xt = np.stack([r["x"] for r in te])
        p = predict_p(Xt, w, mu, sd)
        pick_a = p >= 0.5 + margin
        ok = 0
        dsum = 0.0
        for r, pa in zip(te, pick_a):
            pred_iou = r["ia"] if pa else r["ib"]
            dsum += pred_iou - r["ia"]
            if not r["tie"]:
                ok += int(pa == (r["y"] == 1))
                total += 1
                correct += int(pa == (r["y"] == 1))
        per_seq.append((hold, ok / max(len(te), 1), dsum / max(len(te), 1), len(te)))
        deltas.append(dsum)
    acc = correct / total if total else float("nan")
    return acc, total, float(np.sum(deltas)), per_seq


def write_sub(order, boxes, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ID", "x", "y", "width", "height"])
        for i in order:
            s, fr = i.rsplit("_", 1)
            b = boxes[s][int(fr)]
            w.writerow([i, f"{b[0]:.4f}", f"{b[1]:.4f}", f"{b[2]:.4f}", f"{b[3]:.4f}"])


def validate(sample, out):
    s = {r["ID"] for r in csv.DictReader(open(sample))}
    o = [r["ID"] for r in csv.DictReader(open(out))]
    if set(o) != s or len(o) != len(s):
        raise SystemExit(f"exact-set fail {out}: {len(o)} vs {len(s)}")
    for r in csv.DictReader(open(out)):
        x, y, w, h = map(float, (r["x"], r["y"], r["width"], r["height"]))
        if not (np.isfinite([x, y, w, h]).all() and w > 0 and h > 0 and x >= 0 and y >= 0):
            raise SystemExit(f"bad box {r['ID']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-test", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    A, _ = load_csv(ROOT / "5_outputs/e15_sam3_20260806/submission_val65.csv")
    B, _ = load_csv(ROOT / "5_outputs/t1_rerun_20260805/submission.csv")
    G, _ = load_csv(ROOT / "1_data/raw/2026training.csv")
    Cn, _ = load_csv(ROOT / "5_outputs/t2_bench/local_orig_nir.csv")
    Cr, _ = load_csv(ROOT / "5_outputs/t2_bench/local_orig_rednir.csv")
    C = defaultdict(dict)
    for src in (Cn, Cr):
        for s, fm in src.items():
            C[s].update(fm)

    rows = collect_val(A, B, G, C)
    dis = [r for r in rows if r["iab"] < 0.5 and not r["tie"]]
    report = {"n_val": len(rows), "n_disagree": len(dis), "p_star": P_STAR}

    # ViPT vote
    both_c = [r for r in dis if "iac" in r]
    if both_c:
        vote_a = np.array([r["iac"] > r["ibc"] for r in both_c])
        truth_a = np.array([r["y"] == 1 for r in both_c])
        report["vipt_majority_acc"] = float((vote_a == truth_a).mean())
        report["vipt_n"] = len(both_c)
        gated = [r for r in both_c if abs(r["iac"] - r["ibc"]) >= 0.05]
        if gated:
            va = np.array([r["iac"] > r["ibc"] for r in gated])
            ta = np.array([r["y"] == 1 for r in gated])
            report["vipt_gated05_acc"] = float((va == ta).mean())
            report["vipt_gated05_n"] = len(gated)

    acc, n, dsum, per = seq_holdout_acc(rows, disagree_only=True, margin=0.0)
    report["qhead_holdout_acc"] = acc
    report["qhead_holdout_n"] = n
    report["qhead_holdout_sum_delta_vs_A"] = dsum
    report["qhead_beats_pstar"] = bool(acc >= P_STAR) if n else False
    report["qhead_worst_seq"] = sorted(per, key=lambda t: t[2])[:8]
    report["qhead_best_seq"] = sorted(per, key=lambda t: t[2])[-8:]

    acc2, n2, d2, _ = seq_holdout_acc(rows, disagree_only=True, margin=0.10)
    report["qhead_margin10_acc"] = acc2
    report["qhead_margin10_n"] = n2
    report["qhead_margin10_sum_delta_vs_A"] = d2

    (OUT / "val_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))

    if not args.build_test:
        return

    main_raw, order = load_csv(ROOT / "5_outputs/submissions/sub_v029_v027_left1px.csv")
    src_raw, _ = load_csv(ROOT / "5_outputs/submissions/sub_v012_e23b_sam21_ablation.csv")
    src = apply_corr(src_raw)
    # v029 already has corr
    main = main_raw
    base049, order049 = load_csv(ROOT / "5_outputs/submissions/sub_v049_src_v012_K6.csv")

    tr = [r for r in rows if not r["tie"]]
    X = np.stack([r["x"] for r in tr])
    y = np.array([r["y"] for r in tr], dtype=np.float64)
    w, mu, sd = fit_logit(X, y)
    np.savez(OUT / "qhead_weights.npz", w=w, mu=mu, sd=sd)

    variants = {
        "v051_qhead_disagree": {"margin": 0.0, "iab_max": 0.5, "mods": {0, 1, 2}, "base": "v049"},
        "v052_qhead_margin10": {"margin": 0.10, "iab_max": 0.5, "mods": {0, 1, 2}, "base": "v049"},
        "v053_qhead_rednir": {"margin": 0.0, "iab_max": 0.5, "mods": {1}, "base": "v049"},
        "v054_qhead_allalive": {"margin": 0.05, "iab_max": 0.8, "mods": {0, 1, 2}, "base": "v049"},
        "v055_qhead_from_v029": {"margin": 0.0, "iab_max": 0.5, "mods": {0, 1, 2}, "base": "v029"},
    }

    sample = ROOT / "1_data/raw/sample_submisson.csv"
    stats = {}
    for name, cfg in variants.items():
        base = dict(base049 if cfg["base"] == "v049" else main)
        # deepcopy boxes
        outb = {s: {f: v.copy() for f, v in fm.items()} for s, fm in base.items()}
        ra = {s: frozen_runs(main[s]) for s in main}
        rb = {s: frozen_runs(src[s]) for s in src}
        nrep = 0
        for s in outb:
            if s not in src or s not in main:
                continue
            ks = sorted(set(outb[s]) & set(main[s]) & set(src[s]))
            n = max(len(ks), 1)
            prev_a = prev_b = None
            for i, f in enumerate(ks):
                a, b = main[s][f], src[s][f]
                if modality(s) not in cfg["mods"]:
                    prev_a, prev_b = a, b
                    continue
                if ra[s][f] >= 6:
                    prev_a, prev_b = a, b
                    continue  # leave freeze-splice alone
                if rb[s][f] >= 2:
                    prev_a, prev_b = a, b
                    continue
                iab = iou(a, b)
                if iab >= cfg["iab_max"]:
                    prev_a, prev_b = a, b
                    continue
                x = feats(a, b, prev_a, prev_b, ra[s][f], rb[s][f], i / n, modality(s))
                p = float(predict_p(x[None, :], w, mu, sd)[0])
                # p = P(A better). take B only if confident A is worse
                if p < 0.5 - cfg["margin"]:
                    outb[s][f] = b.copy()
                    nrep += 1
                prev_a, prev_b = a, b
        dest = OUT / f"sub_{name}.csv"
        write_sub(order049, outb, dest)
        validate(sample, dest)
        stats[name] = {"replaced": nrep, "path": str(dest)}
        print(name, nrep, dest)

    (OUT / "test_build.json").write_text(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
