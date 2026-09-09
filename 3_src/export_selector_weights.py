#!/usr/bin/env python3
"""Freeze the selector-v2 head (D085) into a weights file for delivery.

Why this exists
---------------
`apply_selector_to_test75.py` fits on the 405 cache at apply time. That is fine
for a leaderboard experiment but impossible on 9/7: refitting needs the full
405-sequence prediction cache (~6.5 A100-h), which we will not have during the
three-day Ranking B window.

The head is a linear model — `(w, mu, sd)`, exactly the shape the v056 quality
head is already shipped as (`hsot/qhead_weights_v056.npz`). So we fit ONCE here,
against the frozen 405 crop-merged pair, and ship the coefficients. On 9/7
`finalize_submission.py --selector` loads them and only evaluates.

The fit is nothing but the fit inside `apply_selector_to_test75.py`; that path
is the one that byte-reproduced sub_v072, so keep them in sync.

Provenance (inputs, hashes, tau, feature count) goes into a sidecar json so the
delivered weights can be traced back to the cache that produced them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import oof405_crossfit as oof  # noqa: E402
from hsot import selector_v2 as S  # noqa: E402

DEFAULT_OUT = Path(__file__).resolve().parent / "hsot" / "selector_weights_v2.npz"
# LB-selected operating point: tau sweep 0.00/0.02/0.05/0.08/0.15 measured
# 0.71532/0.71552/0.71550/0.71507/0.71457 (v073/v075/v072/v076/v074).
# Plateau is [0.00, 0.05]; 0.05 is the delivered constant -- 0.02's +0.00002 is
# noise and re-picking on it would be fitting the leaderboard.
DELIVERED_TAU = 0.05


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contracts", required=True)
    ap.add_argument("--train-main", required=True,
                    help="405 crop-MERGED primary csv (offline_two_pass/main_merged.csv)")
    ap.add_argument("--train-source", required=True,
                    help="405 crop-MERGED source csv (offline_two_pass/source_merged.csv)")
    ap.add_argument("--train-main-diagnostics", required=True)
    ap.add_argument("--train-source-diagnostics", required=True)
    ap.add_argument("--train-main-crop-diagnostics", required=True)
    ap.add_argument("--train-source-crop-diagnostics", required=True)
    ap.add_argument("--train-crop-meta", required=True)
    ap.add_argument("--pair-profile", default="rankB_robust_crop",
                    choices=sorted(oof.PAIR_SCOPES))
    ap.add_argument("--corr", default="both", choices=("none", "top-only", "both"))
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--tau", type=float, default=DELIVERED_TAU)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    a = ap.parse_args(argv)

    train = oof.load_dataset(
        a.contracts, a.train_main, a.train_source,
        a.train_main_diagnostics, a.train_source_diagnostics,
        pair_profile=a.pair_profile,
        main_crop_diagnostics=a.train_main_crop_diagnostics,
        source_crop_diagnostics=a.train_source_crop_diagnostics,
        crop_meta=a.train_crop_meta,
    )
    tr_main = oof.apply_correction(train.raw_main, train.first_mask, a.corr)
    tr_src = oof.apply_correction(train.raw_source, train.first_mask, a.corr)
    tr_spliced, _ = oof.frozen_splice(
        tr_main, tr_src, train.sequence_indices, train.first_mask, a.K)
    tr_feat, _, tr_src_runs = oof._qhead_feature_cache(train, tr_main, tr_src)

    valid = np.all(train.gt > 0, axis=1)
    delta = np.where(
        valid,
        oof._ious(tr_src, train.gt) - oof._ious(tr_spliced, train.gt), 0.0)
    eligible = valid & (~train.first_mask) & (tr_src_runs < oof.SPLICE_SRC_MAX_RUN)
    fit = eligible & (np.abs(delta) >= S.TIE_EPS)

    X, y = tr_feat[fit], delta[fit]
    weights = np.minimum(np.abs(y), S.WEIGHT_CAP)
    w, mu, sd = S.fit_weighted_ridge(X, y, weights)
    eff_n = S.effective_sample_size(weights)
    train_tau, train_gain = S.choose_tau(S.predict_delta(X, w, mu, sd), y, int(fit.sum()))

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, w=w, mu=mu, sd=sd, tau=np.float64(a.tau))
    meta = {
        "schema_version": 1,
        "head": "selector-v2 (gain-weighted ridge on dIoU, D085)",
        "tau_delivered": a.tau,
        "tau_chosen_on_train": train_tau,
        "train_pooled_gain": train_gain,
        "n_fit_rows": int(fit.sum()),
        "effective_n": eff_n,
        "n_features": int(X.shape[1]),
        "pair_profile": a.pair_profile,
        "corr": a.corr,
        "K": a.K,
        "inputs": {
            "train_main": {"path": str(Path(a.train_main).resolve()),
                           "sha256": _sha256(Path(a.train_main))},
            "train_source": {"path": str(Path(a.train_source).resolve()),
                             "sha256": _sha256(Path(a.train_source))},
            "contracts": str(Path(a.contracts).resolve()),
        },
        "weights_sha256": _sha256(out),
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"fit rows={meta['n_fit_rows']} eff_n={eff_n:.0f} feats={X.shape[1]} "
          f"train-tau={train_tau} delivered-tau={a.tau}")
    print(f"wrote {out} (+ .json provenance)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
