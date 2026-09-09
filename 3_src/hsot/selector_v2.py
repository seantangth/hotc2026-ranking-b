#!/usr/bin/env python3
"""selector-v2: gain-weighted ΔIoU regression to choose between the two trackers.

Why a new head at all
---------------------
The official metric is Success AUC, which equals the mean per-frame IoU, so
swapping frame i from A to B changes the score by exactly

    (IoU(B_i, GT_i) - IoU(A_i, GT_i)) / N  =  d_i / N

The Bayes-optimal rule for that objective is therefore "swap iff E[d] > 0" --
a *regression* on d.  The v056-shaped head (`crossfit-v1`) instead fit an
unweighted logistic on ``y = 1[IoU_A > IoU_B]``, i.e. it optimised P(d > 0).
That is the wrong functional: a frame where A wins by 0.001 and a frame where B
wins by 0.4 contribute equally to its loss, and near-ties vastly outnumber
rescues, so the fit is dominated by frames whose decision does not matter.
D083's fresh grouped smoke measured that head at -0.00231 vs splice-only.

The frozen splice (K=6) is the hand-coded special case of the correct rule:
frames where A is frozen are exactly the frames where |d| is enormous and the
decision is trivially safe, which is why splice earned +0.0059 while every
classifier-shaped head died.  selector-v2 is that same rule with a learned
decision boundary instead of a hand-carved one.

What is deliberately held fixed
-------------------------------
The 25-D feature vector is unchanged from crossfit-v1.  Changing target, loss
and features at once would confound attribution, and the gap audit's finding was
specifically that the old result "只否證 25D logistic + 舊 gate", not the
selector class.  If the crop-pair result misses the gate, the next axis to move
is features (appearance / objectness terms) -- not target or loss.

Pre-registered acceptance gate (fixed here before any number exists; a result
only counts on the **crop pair** for the **primary candidate** ridge-d x wide):

    pooled Δ >= +0.002  AND  >= 4/5 folds positive  AND  worst fold >= -0.003

OOF discipline
--------------
* Fold models are fit on four folds and applied to the fifth, never both.
* τ is chosen inside the training folds only; the heldout number is reported for
  that train-chosen τ.  Best-heldout-τ is never reported as a result.
* The selector runs on the **post-splice** base, mirroring deployment order
  (finalize → splice K → selector), so splice's own swaps are not re-learned as
  free wins.
* First frames are excluded from training and from deployment: preserve_init is
  a delivery invariant and one leaked swap fails the strict validator.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

# τ candidates, fixed before seeing data.  τ is a threshold on predicted d, so
# τ=0 is the Bayes rule and larger τ trades recall for precision.
TAU_GRID: tuple[float, ...] = (0.00, 0.02, 0.05, 0.10, 0.15)

# Ridge penalty on standardised features.  Not tuned per fold -- a swept
# regulariser would be another selection surface to leak through.
RIDGE_LAMBDA = 1.0

# |d| weights concentrate effective sample size onto the rescue tail.  Capping
# keeps a handful of huge-|d| frames from becoming the entire fit; the cap is
# recorded in the report so the choice is auditable rather than silent.
WEIGHT_CAP = 0.5

# Below this |d| the frame is a tie: its decision cannot move the score, and
# including it only adds label noise at ~zero weight.
TIE_EPS = 1e-9


@dataclass(frozen=True)
class SelectorModel:
    """One fold's fitted state.  `tau` was chosen on `train_folds` only."""
    heldout_fold: int
    train_folds: tuple[int, ...]
    n_train: int
    effective_n: float
    tau: float
    train_gain_at_tau: float
    w: np.ndarray
    mu: np.ndarray
    sd: np.ndarray

    def to_doc(self) -> dict[str, Any]:
        return {
            "heldout_fold": self.heldout_fold,
            "train_folds": list(self.train_folds),
            "n_train": self.n_train,
            "effective_n": round(self.effective_n, 2),
            "effective_n_per_feature": round(self.effective_n / max(len(self.w), 1), 2),
            "tau": self.tau,
            "train_gain_at_tau": self.train_gain_at_tau,
            "weights": [round(float(v), 6) for v in self.w],
        }


def fit_weighted_ridge(
    X: np.ndarray, y: np.ndarray, weights: np.ndarray, lam: float = RIDGE_LAMBDA,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Weighted ridge with an unpenalised intercept, on standardised features."""
    if X.ndim != 2 or y.shape != (len(X),) or weights.shape != (len(X),):
        raise ValueError("fit_weighted_ridge: inconsistent shapes")
    if not np.isfinite(X).all() or not np.isfinite(y).all() or not np.isfinite(weights).all():
        raise ValueError("fit_weighted_ridge: non-finite input")
    if np.any(weights < 0):
        raise ValueError("fit_weighted_ridge: negative weight")

    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd <= 0] = 1.0
    Z = np.hstack([(X - mu) / sd, np.ones((len(X), 1))])

    sw = np.sqrt(weights)[:, None]
    Zw, yw = Z * sw, y * np.sqrt(weights)
    penalty = lam * np.eye(Z.shape[1])
    penalty[-1, -1] = 0.0  # never shrink the intercept
    w = np.linalg.solve(Zw.T @ Zw + penalty, Zw.T @ yw)
    if not np.isfinite(w).all():
        raise ValueError("fit_weighted_ridge: non-finite solution")
    return w, mu, sd


def predict_delta(X: np.ndarray, w: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """Predicted d = IoU(B,GT) - IoU(A,GT); positive means B is worth taking.

    Note: numpy 2.x on Apple Accelerate raises spurious FP warnings from ``@``
    even for well-conditioned float64 inputs (verified against einsum and a
    manual dot loop, agreeing to 1.8e-15).  Production runs on Linux/OpenBLAS
    and does not see them, so the warnings are left unsuppressed -- suppressing
    them here would also hide a real one.  The finiteness check below is what
    actually guards the output.
    """
    Z = np.hstack([(X - mu) / sd, np.ones((len(X), 1))])
    out = Z @ w
    if not np.isfinite(out).all():
        raise ValueError("predict_delta produced non-finite predictions")
    return out


def effective_sample_size(weights: np.ndarray) -> float:
    """Kish effective n = (Σw)² / Σw².

    With |d| weighting most mass sits on the rescue tail, so the nominal row
    count badly overstates how much the fit actually saw.  Reported per fold so
    an over-concentrated fit is visible instead of silently overfitting.
    """
    s1 = float(weights.sum())
    s2 = float((weights ** 2).sum())
    return 0.0 if s2 <= 0 else (s1 * s1) / s2


def _pooled_gain(delta: np.ndarray, take: np.ndarray, n_rows: int) -> float:
    """AUC change from swapping the selected rows, in pooled-mean-IoU units."""
    return float(delta[take].sum() / n_rows) if n_rows else 0.0


def choose_tau(
    pred: np.ndarray, delta: np.ndarray, n_rows: int, tau_grid: Sequence[float] = TAU_GRID,
) -> tuple[float, float]:
    """Pick τ maximising pooled gain **on the rows handed in** (train folds only).

    Ties break toward the larger τ: fewer swaps for the same modelled gain is the
    lower-variance choice on unseen data.
    """
    best_tau, best_gain = float(tau_grid[0]), -np.inf
    for tau in tau_grid:
        take = pred >= tau
        gain = _pooled_gain(delta, take, n_rows)
        if gain > best_gain or (gain == best_gain and tau > best_tau):
            best_tau, best_gain = float(tau), gain
    return best_tau, float(best_gain)


def fit_fold_models(
    features: np.ndarray,
    delta: np.ndarray,
    folds: np.ndarray,
    eligible: np.ndarray,
    n_folds: int,
    weight_cap: float = WEIGHT_CAP,
    tau_grid: Sequence[float] = TAU_GRID,
) -> list[SelectorModel]:
    """Fit one model per heldout fold on the other four.

    `eligible` is the training/deployment gate: rows with valid GT, both trackers
    alive, not a first frame.  Ties (|d| < TIE_EPS) carry no signal and no weight,
    so they are dropped rather than fed in at weight ~0.
    """
    models: list[SelectorModel] = []
    n_features = features.shape[1]
    for heldout in range(n_folds):
        train_mask = eligible & (folds != heldout) & (np.abs(delta) >= TIE_EPS)
        seen = tuple(sorted(set(folds[train_mask].tolist())))
        expected = tuple(f for f in range(n_folds) if f != heldout)
        if seen != expected:
            raise ValueError(
                f"selector fold {heldout}: train rows do not cover all four train folds; "
                f"got {seen}, required {expected}")
        X = features[train_mask]
        y = delta[train_mask]
        weights = np.minimum(np.abs(y), weight_cap)
        eff_n = effective_sample_size(weights)
        if len(y) < 50 or eff_n < 10:
            raise ValueError(
                f"selector fold {heldout}: degenerate training set n={len(y)}, eff_n={eff_n:.1f}")
        w, mu, sd = fit_weighted_ridge(X, y, weights)
        # τ is chosen on these same training rows -- never on the heldout fold.
        tau, train_gain = choose_tau(
            predict_delta(X, w, mu, sd), y, int(train_mask.sum()), tau_grid)
        models.append(SelectorModel(
            heldout_fold=heldout, train_folds=expected, n_train=len(y), effective_n=eff_n,
            tau=tau, train_gain_at_tau=train_gain, w=w, mu=mu, sd=sd,
        ))
        if w.shape != (n_features + 1,):
            raise ValueError(f"selector fold {heldout}: unexpected weight shape {w.shape}")
    return models


def apply_oof(
    base: np.ndarray,
    source: np.ndarray,
    features: np.ndarray,
    folds: np.ndarray,
    eligible: np.ndarray,
    first_mask: np.ndarray,
    raw_main: np.ndarray,
    models: Sequence[SelectorModel],
) -> tuple[np.ndarray, np.ndarray]:
    """Apply each fold's model to its own heldout fold and assemble OOF output.

    `base` is the post-splice prediction: deployment order is
    finalize → splice K → selector, so the selector only ever decides frames the
    splice left alone.
    """
    out = base.copy()
    selected = np.zeros(len(base), dtype=bool)
    for model in models:
        idx = np.flatnonzero(eligible & (folds == model.heldout_fold))
        if not len(idx):
            continue  # a fold with no eligible rows contributes 0, not an error
        take = idx[predict_delta(features[idx], model.w, model.mu, model.sd) >= model.tau]
        out[take] = source[take]
        selected[take] = True
    # preserve_init is a hard delivery invariant; restore unconditionally.
    out[first_mask] = raw_main[first_mask]
    selected[first_mask] = False
    return out, selected
