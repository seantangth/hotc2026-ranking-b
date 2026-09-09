#!/usr/bin/env python3
"""selector-v2 regression tests: the OOF discipline must be enforced, not documented."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from hsot import selector_v2 as S  # noqa: E402

N_FOLDS = 5


def _synthetic(n_per_fold: int = 400, seed: int = 0):
    """A pair where B is better exactly when feature 0 is large.

    Mimics the real shape: a mass of near-ties plus a small high-|d| rescue tail,
    which is what makes the unweighted classifier fail and the weighting matter.
    """
    rng = np.random.default_rng(seed)
    n = n_per_fold * N_FOLDS
    folds = np.repeat(np.arange(N_FOLDS), n_per_fold)
    x0 = rng.normal(size=n)
    noise = rng.normal(scale=0.05, size=n)
    features = np.column_stack([x0, rng.normal(size=(n, 4))])
    # Real shape: a mass of near-ties, plus BOTH tails -- swapping rescues when
    # x0 is high and does damage when x0 is low.  A one-sided synthetic would
    # let a constant-positive predictor look good, which is the exact failure
    # mode |d| weighting can induce (see test_gain_weighting_...).
    delta = np.where(x0 > 1.5, 0.40 + noise,
                     np.where(x0 < -1.5, -0.40 + noise, noise * 0.02))
    eligible = np.ones(n, dtype=bool)
    return features, delta, folds, eligible


def test_weighted_ridge_recovers_the_signal_and_ignores_noise_features():
    features, delta, folds, eligible = _synthetic()
    w, mu, sd = S.fit_weighted_ridge(
        features, delta, np.minimum(np.abs(delta), S.WEIGHT_CAP))
    # feature 0 carries all the signal; the four noise columns must stay small
    assert abs(w[0]) > 5 * max(abs(v) for v in w[1:5])

    # What matters for the objective is *ranking*, not linear fit quality: the
    # rule is "swap iff predicted d is high", so the test is whether rescues
    # outrank damage.  Demanding a high Pearson r against a step-shaped target
    # would only measure that a linear model cannot draw a step.
    pred = S.predict_delta(features, w, mu, sd)
    rescue, damage = delta > 0.2, delta < -0.2
    assert pred[rescue].mean() > pred[damage].mean()
    assert pred[rescue].min() > pred[damage].max(), "rescues must rank above damage"


def test_gain_weighting_makes_the_near_zero_region_sloppy_but_tau_absorbs_it():
    """|d| weighting is correct for the objective and has a known side effect.

    Ties get ~zero weight, so the fit barely constrains predictions near d=0 and
    can carry a large intercept.  That is acceptable *because* a mis-decision
    there costs ~|d|≈0 -- but only if τ selection can still refuse to swap.  This
    pins that safety property: on data where swapping is uniformly harmful, the
    chosen τ must fire on nothing rather than swapping everything.
    """
    features, delta, folds, eligible = _synthetic()
    harmful = -np.abs(delta)  # every swap loses
    models = S.fit_fold_models(features, harmful, folds, eligible, N_FOLDS)
    n = len(delta)
    base = np.tile(np.array([1.0, 1.0, 1.0, 1.0]), (n, 1))
    source = np.tile(np.array([2.0, 2.0, 2.0, 2.0]), (n, 1))
    _, selected = S.apply_oof(
        base, source, features, folds, eligible,
        np.zeros(n, bool), base.copy(), models)
    assert selected.sum() == 0, "τ must refuse to swap when swapping only ever loses"


def test_effective_n_exposes_weight_concentration():
    # uniform weights: eff_n == n
    assert S.effective_sample_size(np.ones(100)) == pytest.approx(100.0)
    # one dominant weight: eff_n collapses toward 1
    w = np.concatenate([[1000.0], np.full(999, 1e-3)])
    assert S.effective_sample_size(w) < 2.0
    assert S.effective_sample_size(np.zeros(10)) == 0.0


def test_tau_is_chosen_on_the_rows_given_and_prefers_fewer_swaps_on_ties():
    # swapping helps only above 0.10, so a higher tau must win
    pred = np.array([0.01, 0.05, 0.12, 0.20])
    delta = np.array([-0.10, -0.10, 0.30, 0.30])
    tau, gain = S.choose_tau(pred, delta, n_rows=4)
    assert tau == 0.10
    assert gain == pytest.approx((0.30 + 0.30) / 4)

    # exact tie in gain across tau: the larger tau (fewer swaps) is taken
    pred_tie = np.array([0.20, 0.20])
    delta_tie = np.array([0.10, 0.10])
    tau_tie, _ = S.choose_tau(pred_tie, delta_tie, n_rows=2)
    assert tau_tie == max(S.TAU_GRID)


def test_fold_models_never_see_their_own_heldout_fold():
    features, delta, folds, eligible = _synthetic()
    models = S.fit_fold_models(features, delta, folds, eligible, N_FOLDS)
    assert len(models) == N_FOLDS
    for model in models:
        assert model.heldout_fold not in model.train_folds
        assert model.train_folds == tuple(
            f for f in range(N_FOLDS) if f != model.heldout_fold)
        assert model.tau in S.TAU_GRID
        assert model.effective_n > 0


def test_fold_model_fit_is_identical_with_the_heldout_fold_perturbed():
    """The strongest available leak check: corrupt a fold, its model must not move."""
    features, delta, folds, eligible = _synthetic()
    base = S.fit_fold_models(features, delta, folds, eligible, N_FOLDS)

    poisoned = delta.copy()
    poisoned[folds == 2] = -5.0  # nonsense labels in fold 2 only
    after = S.fit_fold_models(features, poisoned, folds, eligible, N_FOLDS)

    m_base = next(m for m in base if m.heldout_fold == 2)
    m_after = next(m for m in after if m.heldout_fold == 2)
    assert np.allclose(m_base.w, m_after.w)
    assert m_base.tau == m_after.tau
    # every other fold *does* train on fold 2, so those must change
    other_base = next(m for m in base if m.heldout_fold == 0)
    other_after = next(m for m in after if m.heldout_fold == 0)
    assert not np.allclose(other_base.w, other_after.w)


def test_apply_oof_restores_first_frames_and_never_selects_them():
    features, delta, folds, eligible = _synthetic()
    n = len(delta)
    models = S.fit_fold_models(features, delta, folds, eligible, N_FOLDS)

    base = np.tile(np.array([10.0, 10.0, 5.0, 5.0]), (n, 1))
    source = np.tile(np.array([20.0, 20.0, 6.0, 6.0]), (n, 1))
    raw_main = np.tile(np.array([1.0, 2.0, 3.0, 4.0]), (n, 1))
    first_mask = np.zeros(n, dtype=bool)
    first_mask[::400] = True  # one first frame per synthetic sequence

    out, selected = S.apply_oof(
        base, source, features, folds, eligible, first_mask, raw_main, models)
    assert np.array_equal(out[first_mask], raw_main[first_mask])
    assert not selected[first_mask].any()
    assert selected.sum() > 0, "the selector should fire somewhere on this signal"
    # untouched rows keep the spliced base, never a blend
    untouched = ~selected & ~first_mask
    assert np.array_equal(out[untouched], base[untouched])


def test_apply_oof_uses_the_post_splice_base_not_raw_main():
    """Deployment order is finalize → splice → selector; rows the selector skips
    must keep the *spliced* box, otherwise splice's gains are silently reverted."""
    features, delta, folds, eligible = _synthetic()
    n = len(delta)
    models = S.fit_fold_models(features, delta, folds, eligible, N_FOLDS)
    base = np.tile(np.array([7.0, 7.0, 7.0, 7.0]), (n, 1))       # "spliced"
    raw_main = np.tile(np.array([1.0, 1.0, 1.0, 1.0]), (n, 1))   # pre-splice
    source = np.tile(np.array([9.0, 9.0, 9.0, 9.0]), (n, 1))
    out, selected = S.apply_oof(
        base, source, features, folds, eligible, np.zeros(n, bool), raw_main, models)
    assert np.array_equal(out[~selected], base[~selected])
    assert not np.array_equal(out[~selected][0], raw_main[0])


def test_degenerate_training_set_fails_closed():
    features, delta, folds, eligible = _synthetic(n_per_fold=5)
    with pytest.raises(ValueError, match="degenerate training set"):
        S.fit_fold_models(features, delta, folds, eligible, N_FOLDS)


def test_missing_train_fold_coverage_fails_closed():
    features, delta, folds, eligible = _synthetic()
    eligible = eligible.copy()
    eligible[folds == 3] = False  # fold 3 can never be trained on
    with pytest.raises(ValueError, match="do not cover all four train folds"):
        S.fit_fold_models(features, delta, folds, eligible, N_FOLDS)


def test_weight_cap_bounds_the_influence_of_a_single_rescue_frame():
    features, delta, folds, eligible = _synthetic()
    outlier = delta.copy()
    outlier[0] = 50.0  # one absurd frame
    capped = S.fit_fold_models(features, outlier, folds, eligible, N_FOLDS)
    clean = S.fit_fold_models(features, delta, folds, eligible, N_FOLDS)
    # fold 0 holds out the outlier, so its model must be untouched by it
    m_capped = next(m for m in capped if m.heldout_fold == 0)
    m_clean = next(m for m in clean if m.heldout_fold == 0)
    assert np.allclose(m_capped.w, m_clean.w)
    # folds that do train on it stay finite and bounded thanks to the cap
    m_other = next(m for m in capped if m.heldout_fold == 1)
    assert np.isfinite(m_other.w).all()
    assert m_other.effective_n > 1.0


def test_ties_are_dropped_rather_than_trained_on_at_zero_weight():
    features, delta, folds, eligible = _synthetic()
    ties = np.zeros_like(delta)
    ties[np.abs(delta) > 0.2] = delta[np.abs(delta) > 0.2]
    models = S.fit_fold_models(features, ties, folds, eligible, N_FOLDS)
    n_non_tie = int((np.abs(ties) >= S.TIE_EPS).sum())
    assert all(m.n_train < n_non_tie for m in models)
    assert all(m.n_train > 0 for m in models)
