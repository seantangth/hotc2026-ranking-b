#!/usr/bin/env python3
"""Fit selector-v2 on all 405 train sequences, apply to test75, emit a Kaggle CSV.

Sean's standing rule (08-28, and the project's own D040): **Kaggle is the only
ground truth.**  The 405 grouped OOF is a disaster check, not a gain gate, so a
head that fails the OOF still gets its shot on the leaderboard.

This is also a *cleaner* test than the OOF: test75 is genuinely held out from the
405 training set, so there is one fit on all 405 and one application to test75 --
no folds, no tau chosen on the thing being scored.

Deployment order matches production exactly: corr -> frozen splice K -> selector,
with first frames restored to init at the end.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import oof405_crossfit as oof  # noqa: E402
from hsot import selector_v2 as S  # noqa: E402

SUB_COLS = ("ID", "x", "y", "width", "height")


def read_boxes(path: Path) -> tuple[list[str], np.ndarray]:
    ids, rows = [], []
    with path.open(newline="") as fh:
        for r in csv.DictReader(fh):
            ids.append(r["ID"])
            rows.append([float(r[c]) for c in SUB_COLS[1:]])
    return ids, np.asarray(rows, dtype=np.float64)


def structure(ids: list[str]):
    """sequence_indices / modalities / first_mask, derived from IDs alone.

    No GT is involved: the feature extractor is prediction-only by contract, and
    test75 has no labels anyway.
    """
    seq_of, order = [], {}
    for i, ident in enumerate(ids):
        seq = ident.rsplit("_", 1)[0]
        seq_of.append(seq)
        order.setdefault(seq, []).append(i)
    sequence_indices = {s: np.asarray(v, dtype=np.int64) for s, v in order.items()}
    first_mask = np.zeros(len(ids), dtype=bool)
    for v in sequence_indices.values():
        first_mask[v[0]] = True
    modalities = [s.split("-", 1)[0] for s in seq_of]
    unknown = sorted({m for m in modalities} - {"nir", "rednir", "vis"})
    if unknown:
        raise SystemExit(f"unknown modality prefixes: {unknown}")
    return sequence_indices, modalities, first_mask


class _Shim:
    """Minimal duck-type of oof.Dataset for the prediction-only feature cache."""

    def __init__(self, ids, sequence_indices, modalities):
        self.ids = ids
        self.sequence_indices = sequence_indices
        self.modalities = modalities


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contracts", required=True, help="405 contracts dir (for fitting)")
    ap.add_argument("--train-main", required=True)
    ap.add_argument("--train-source", required=True)
    ap.add_argument("--test-main", required=True)
    ap.add_argument("--test-source", required=True)
    ap.add_argument("--train-main-diagnostics")
    ap.add_argument("--train-source-diagnostics")
    ap.add_argument("--train-pair-profile", default="rankB_robust",
                    choices=sorted(oof.PAIR_SCOPES))
    ap.add_argument("--train-main-crop-diagnostics")
    ap.add_argument("--train-source-crop-diagnostics")
    ap.add_argument("--train-crop-meta")
    ap.add_argument("--sample", required=True)
    ap.add_argument("--corr", default="both", choices=("none", "top-only", "both"))
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--tau", type=float, default=None,
                    help="override the train-chosen tau. Sean 08-28: Kaggle is the only\n"
                         "ground truth, so tau is swept on the LB rather than trusted from\n"
                         "the train fit -- which under-estimated this head by 20x.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    # ---- fit on all 405 -----------------------------------------------------
    train = oof.load_dataset(
        args.contracts, args.train_main, args.train_source,
        args.train_main_diagnostics, args.train_source_diagnostics,
        pair_profile=args.train_pair_profile,
        main_crop_diagnostics=args.train_main_crop_diagnostics,
        source_crop_diagnostics=args.train_source_crop_diagnostics,
        crop_meta=args.train_crop_meta,
    )
    tr_main = oof.apply_correction(train.raw_main, train.first_mask, args.corr)
    tr_src = oof.apply_correction(train.raw_source, train.first_mask, args.corr)
    tr_spliced, _ = oof.frozen_splice(
        tr_main, tr_src, train.sequence_indices, train.first_mask, args.K)
    tr_feat, _, tr_src_runs = oof._qhead_feature_cache(train, tr_main, tr_src)

    valid = np.all(train.gt > 0, axis=1)
    delta = np.where(
        valid,
        oof._ious(tr_src, train.gt) - oof._ious(tr_spliced, train.gt), 0.0)
    eligible = valid & (~train.first_mask) & (tr_src_runs < oof.SPLICE_SRC_MAX_RUN)
    non_tie = np.abs(delta) >= S.TIE_EPS

    fit = eligible & non_tie
    X, y = tr_feat[fit], delta[fit]
    weights = np.minimum(np.abs(y), S.WEIGHT_CAP)
    eff_n = S.effective_sample_size(weights)
    w, mu, sd = S.fit_weighted_ridge(X, y, weights)
    tau, train_gain = S.choose_tau(
        S.predict_delta(X, w, mu, sd), y, int(fit.sum()))
    print(f"fit on 405: n={len(y)} eff_n={eff_n:.0f} "
          f"({eff_n/X.shape[1]:.1f} per feature) train-chosen tau={tau} "
          f"train pooled gain={train_gain:+.5f}")
    if args.tau is not None:
        print(f"tau overridden for LB sweep: {tau} -> {args.tau}")
        tau = args.tau

    # ---- apply to test75 ----------------------------------------------------
    ids_m, te_main_raw = read_boxes(Path(args.test_main))
    ids_s, te_src_raw = read_boxes(Path(args.test_source))
    if ids_m != ids_s:
        raise SystemExit("test main/source ID order differs")
    seq_idx, modalities, first_mask = structure(ids_m)

    te_main = oof.apply_correction(te_main_raw, first_mask, args.corr)
    te_src = oof.apply_correction(te_src_raw, first_mask, args.corr)
    te_spliced, splice_sel = oof.frozen_splice(
        te_main, te_src, seq_idx, first_mask, args.K)

    shim = _Shim(ids_m, seq_idx, modalities)
    te_feat, _, te_src_runs = oof._qhead_feature_cache(shim, te_main, te_src)
    te_gate = (~first_mask) & (te_src_runs < oof.SPLICE_SRC_MAX_RUN)

    pred = S.predict_delta(te_feat[te_gate], w, mu, sd)
    idx = np.flatnonzero(te_gate)
    take = idx[pred >= tau]
    out = te_spliced.copy()
    out[take] = te_src[take]
    out[first_mask] = te_main_raw[first_mask]  # preserve_init, un-corrected
    print(f"test75: splice {int(splice_sel.sum())} 幀, selector 再換 {len(take)} 幀 "
          f"({100*len(take)/len(ids_m):.1f}%)")

    # ---- write in canonical sample order ------------------------------------
    sample_ids = [r["ID"] for r in csv.DictReader(Path(args.sample).open(newline=""))]
    if set(sample_ids) != set(ids_m):
        raise SystemExit("prediction ID set != sample ID set")
    by_id = dict(zip(ids_m, out, strict=True))
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with tmp.open("w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(SUB_COLS)
        for ident in sample_ids:
            b = by_id[ident]
            if not np.isfinite(b).all() or b[2] <= 0 or b[3] <= 0 or b[0] < 0 or b[1] < 0:
                raise SystemExit(f"invalid box for {ident}: {b}")
            wr.writerow([ident, *(f"{v:.15g}" for v in b)])
    tmp.replace(dest)
    print(f"wrote {len(sample_ids)} rows -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
