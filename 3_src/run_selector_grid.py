#!/usr/bin/env python3
"""Run the selector 2x2 attribution grid on a formal 405 exact-pair cache.

    {logistic, ridge-d} x {v056 gate, wide gate}

Why a grid and not just the new head: E41 measured the label-space analogue on
LB and found the *gate* alone moved the result from +0.00079 (RedNIR-only) to
-0.006..-0.012 (all modalities).  Changing target/loss and gate together would
make a win unattributable and a loss undiagnosable, so all four cells run
through the same cross-fit.

Pre-registered (see hsot/selector_v2.py, fixed before any number existed):
  primary candidate = ridge-d x wide, on the CROP pair only
  accept iff pooled dAUC >= +0.002 AND >= 4/5 folds positive AND worst >= -0.003
The other three cells are attribution, not selection.

Everything here is CPU and reads only the cached predictions, so iterating costs
nothing -- that is the whole point of having paid for the cache once.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import csv

import oof405_crossfit as oof
from hsot import selector_v2 as S
from hsot.quality_head_v1 import fit_logit, predict_p


def objectness_features(dataset, paths: list[str]) -> tuple[np.ndarray, list[str]]:
    """SAM3's own per-frame objectness, plus how it moves -- a model-internal
    confidence signal that the 25-D geometry vector cannot express.

    Not a revival of D075: that judged obj_score as a *hand-carved A-vs-B gate*
    on 23 live frames.  Here it is one feature among many inside a gain-weighted
    regression, which is a different use and outside that closure's scope.

    Later paths override earlier ones, so pass [full_leg, crop_leg]: the crop leg
    only covers the cropped frames and its score is the one the deployed primary
    actually produced there.
    """
    score = {}
    for path in paths:
        if not path:
            continue
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh):
                score[row["ID"]] = float(row["obj_score"])
    missing = [i for i in dataset.ids if i not in score]
    if missing:
        raise SystemExit(
            f"obj_scores missing {len(missing)} IDs (first {missing[0]}); refusing to impute")

    raw = np.asarray([score[i] for i in dataset.ids], dtype=np.float64)
    if not np.isfinite(raw).all():
        raise SystemExit("obj_scores contain non-finite values")
    # Squash: the logit's tail runs to ~24 and would dominate a standardised fit.
    sq = np.sign(raw) * np.log1p(np.abs(raw))
    delta_prev = np.zeros_like(sq)
    run_min = np.zeros_like(sq)
    for indices in dataset.sequence_indices.values():
        seq = sq[indices]
        delta_prev[indices[1:]] = seq[1:] - seq[:-1]
        # rolling min over the last 5 frames: a recent collapse in objectness is
        # the shape that precedes a lost target
        for k, idx in enumerate(indices):
            run_min[idx] = seq[max(0, k - 4):k + 1].min()
    return (np.column_stack([sq, delta_prev, run_min, sq - run_min]),
            ["objscore", "objscore_d1", "objscore_min5", "objscore_minus_min5"])


def build_gates(dataset, main, source, main_runs, source_runs, K):
    """The two gates under test.

    v056: the hand-carved deployment gate (RedNIR only, both alive, disagreeing).
    wide: every frame the selector could legally decide -- valid GT, not a first
    frame, and the source is not itself frozen (a frozen source has nothing to
    offer).  No modality restriction: whether RedNIR-only was essential or was an
    artefact of the old target is precisely what this grid tests.
    """
    valid_gt = np.all(dataset.gt > 0, axis=1)
    v056 = oof._qhead_gate(dataset, main, source, main_runs, source_runs, K) & valid_gt
    wide = valid_gt & (~dataset.first_mask) & (source_runs < oof.SPLICE_SRC_MAX_RUN)
    return {"v056": v056, "wide": wide}


def run_logistic(dataset, spliced, main, source, features, gate, delta):
    """crossfit-v1's rule (binary 'who wins', unweighted) on an arbitrary gate."""
    non_tie = np.abs(delta) >= S.TIE_EPS
    out, selected = spliced.copy(), np.zeros(len(spliced), dtype=bool)
    models = []
    for heldout in range(oof.N_FOLDS):
        train = gate & (dataset.folds != heldout) & non_tie
        y = (delta[train] < 0).astype(np.float64)  # 1 = A better
        if len(y) < 50 or y.sum() < 2 or (len(y) - y.sum()) < 2:
            raise SystemExit(f"logistic fold {heldout}: degenerate train set n={len(y)}")
        w, mu, sd = fit_logit(features[train], y, l2=1e-2)
        idx = np.flatnonzero(gate & (dataset.folds == heldout))
        if len(idx):
            take = idx[predict_p(features[idx], w, mu, sd) < 0.5 - oof.QHEAD_MARGIN]
            out[take] = source[take]
            selected[take] = True
        models.append({"heldout_fold": heldout, "n_train": int(len(y))})
    out[dataset.first_mask] = dataset.raw_main[dataset.first_mask]
    selected[dataset.first_mask] = False
    return out, selected, models


def run_ridge(dataset, spliced, main, source, features, gate, delta):
    """selector-v2: regression on d, weighted by |d|, tau chosen on train folds."""
    models = S.fit_fold_models(features, delta, dataset.folds, gate, oof.N_FOLDS)
    out, selected = S.apply_oof(
        spliced, source, features, dataset.folds, gate,
        dataset.first_mask, dataset.raw_main, models)
    return out, selected, [m.to_doc() for m in models]


def score(dataset, boxes):
    ious = oof._ious(boxes, dataset.gt)
    pooled = oof._auc_from_ious(ious)
    per_fold = {}
    for f in range(oof.N_FOLDS):
        mask = dataset.folds == f
        per_fold[f] = oof._auc_from_ious(ious[mask])
    return pooled, per_fold


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contracts", required=True)
    ap.add_argument("--main", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--pair-profile", default="rankB_robust_crop",
                    choices=sorted(oof.PAIR_SCOPES))
    ap.add_argument("--main-diagnostics")
    ap.add_argument("--source-diagnostics")
    ap.add_argument("--main-crop-diagnostics")
    ap.add_argument("--source-crop-diagnostics")
    ap.add_argument("--crop-meta")
    ap.add_argument("--corr", default="both", choices=("none", "top-only", "both"))
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--obj-scores", nargs="*", default=[],
                    help="obj_scores.csv paths, full leg first then crop leg")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    dataset = oof.load_dataset(
        args.contracts, args.main, args.source,
        args.main_diagnostics, args.source_diagnostics,
        pair_profile=args.pair_profile,
        main_crop_diagnostics=args.main_crop_diagnostics,
        source_crop_diagnostics=args.source_crop_diagnostics,
        crop_meta=args.crop_meta,
    )
    main_b = oof.apply_correction(dataset.raw_main, dataset.first_mask, args.corr)
    source_b = oof.apply_correction(dataset.raw_source, dataset.first_mask, args.corr)
    spliced, splice_sel = oof.frozen_splice(
        main_b, source_b, dataset.sequence_indices, dataset.first_mask, args.K)

    features, main_runs, source_runs = oof._qhead_feature_cache(dataset, main_b, source_b)
    feature_names = list(oof.FEATURE_NAMES)
    if args.obj_scores:
        extra, extra_names = objectness_features(dataset, args.obj_scores)
        features = np.hstack([features, extra])
        feature_names += extra_names
        print(f'features: {len(oof.FEATURE_NAMES)} 幾何 + {len(extra_names)} objectness'
              f' = {features.shape[1]}')
    gates = build_gates(dataset, main_b, source_b, main_runs, source_runs, args.K)

    # d is measured against the SPLICED base: the selector only ever decides
    # frames the splice left alone, so learning against raw main would re-earn
    # the splice's own swaps as if they were free.
    iou_base = oof._ious(spliced, dataset.gt)
    iou_src = oof._ious(source_b, dataset.gt)
    delta = np.where(np.all(dataset.gt > 0, axis=1), iou_src - iou_base, 0.0)

    base_pooled, base_folds = score(dataset, spliced)
    print(f"pair={args.pair_profile} corr={args.corr} K={args.K}")
    print(f"baseline (splice only) pooled AUC = {base_pooled:.5f}\n")

    results = {"pair_profile": args.pair_profile, "corr": args.corr, "K": args.K,
               "baseline_pooled_auc": base_pooled,
               "baseline_fold_auc": {str(k): v for k, v in base_folds.items()},
               "pre_registered_gate": {
                   "pooled_min": 0.002, "min_positive_folds": 4, "worst_fold_min": -0.003,
                   "primary_candidate": "ridge-d x wide, crop pair only"},
               "features": None, "cells": {}}

    print(f"{'cell':22s} {'pooled':>9s} {'dAUC':>9s} {'+folds':>7s} {'worst':>9s} {'swaps':>7s}")
    for head_name, runner in (("logistic", run_logistic), ("ridge-d", run_ridge)):
        for gate_name, gate in gates.items():
            try:
                out, selected, models = runner(
                    dataset, spliced, main_b, source_b, features, gate, delta)
            except (SystemExit, ValueError) as exc:
                print(f"{head_name+' x '+gate_name:22s} FAILED: {exc}")
                results["cells"][f"{head_name}__{gate_name}"] = {"error": str(exc)}
                continue
            pooled, per_fold = score(dataset, out)
            fold_d = {f: per_fold[f] - base_folds[f] for f in per_fold}
            pos = sum(1 for v in fold_d.values() if v > 0)
            worst = min(fold_d.values())
            cell = f"{head_name} x {gate_name}"
            print(f"{cell:22s} {pooled:9.5f} {pooled-base_pooled:+9.5f} "
                  f"{pos:5d}/5 {worst:+9.5f} {int(selected.sum()):7d}")
            results["cells"][f"{head_name}__{gate_name}"] = {
                "pooled_auc": pooled, "delta_vs_splice_only": pooled - base_pooled,
                "positive_folds": pos, "worst_fold_delta": worst,
                "n_selected": int(selected.sum()),
                "fold_delta": {str(k): v for k, v in fold_d.items()},
                "gate_rows": int(gate.sum()), "models": models,
            }

    primary = results["cells"].get("ridge-d__wide", {})
    passed = (
        args.pair_profile == "rankB_robust_crop"
        and primary.get("delta_vs_splice_only", -1) >= 0.002
        and primary.get("positive_folds", 0) >= 4
        and primary.get("worst_fold_delta", -1) >= -0.003
    )
    results["primary_candidate_passes_pre_registered_gate"] = passed
    print(f"\n事前判準（pooled>=+0.002 且 >=4/5 folds 正 且 worst>=-0.003）"
          f"，主候選 ridge-d x wide：{'✅ 通過' if passed else '❌ 未通過'}")
    if args.pair_profile != "rankB_robust_crop":
        print("⚠️ 非 crop 配對：本結果僅供管線試車與歸因，不作為採用依據（D044）")

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    print(f"outputs={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
