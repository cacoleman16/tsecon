"""Regression tests for three confirmed audit findings in the LP family.

1. ``lp(cumulative="both")`` reported an inconsistent standard error under
   the (then-)default lag-augmented HC1 inference: the cumulated impulse
   ``sum_(j=0..h) shock_(t+j)`` shares *future* shocks across base times up
   to ``h`` apart, which past-lag augmentation cannot project out, so the
   score is serially correlated and HC1 misses it (measured: 0.507 coverage
   at a nominal 95%, h=12, flat in T). The fix: the default ``se=None``
   resolves to HAC for this mode, the result stamps ``se_method``, and an
   explicit ``se="lag_augmented"`` with ``cumulative="both"`` raises.
   ``lp_state`` shares defect and fix.

2. ``lp(se="hac", band=...)`` computed ``cov_se_max_rel_diff`` in Rust and
   dropped it at the binding while the model card promised "the largest
   relative gap is reported". The fix returns it from every banded route.

3. ``smooth_lp``'s default CV lambda grid was the absolute ladder
   1e-2..1e6, so rescaling the shock walked the CV optimum off the grid
   (``lambda_used`` pinned at an endpoint, the unit-normalized IRF changed
   character). The fix anchors the default ladder to the spline block of the
   stacked X'X, making the selection invariant to rescaling either series.
"""

import numpy as np
import pytest

import tsecon


def _no_propagation_dgp(seed=0, n=400):
    """y_t = s_t + eta_t: under cumulative="both" the truth is exactly 1."""
    rng = np.random.default_rng(seed)
    s = rng.standard_normal(n)
    y = s + rng.standard_normal(n)
    return y, s


# --------------------------------------------------------------------------
# Finding 1: cumulative="both" inference
# --------------------------------------------------------------------------
def test_lp_both_default_resolves_to_hac_and_stamps_se_method():
    y, s = _no_propagation_dgp()
    r = tsecon.lp(y, s, horizons=8, n_lag_controls=4, cumulative="both")
    assert r["se_method"] == "hac"
    explicit = tsecon.lp(
        y, s, horizons=8, n_lag_controls=4, cumulative="both", se="hac"
    )
    np.testing.assert_array_equal(r["irf"], explicit["irf"])
    np.testing.assert_array_equal(r["se"], explicit["se"])
    # The defect's signature was a FLAT se path (~0.55% spread) against a
    # sampling sd growing ~2.7x by h=8; a covariance that carries the MA(h)
    # overlap must grow materially with the horizon on this DGP.
    se = np.asarray(r["se"])
    assert se[8] > 1.5 * se[0], f"se path still flat: {se}"


def test_lp_default_se_is_unchanged_outside_both_mode():
    y, s = _no_propagation_dgp()
    for cum in (None, False, True, "none", "outcome"):
        r = tsecon.lp(y, s, horizons=4, n_lag_controls=4, cumulative=cum)
        assert r["se_method"] == "lag_augmented", cum
    la = tsecon.lp(y, s, horizons=4, n_lag_controls=4)
    ex = tsecon.lp(y, s, horizons=4, n_lag_controls=4, se="lag_augmented")
    np.testing.assert_array_equal(la["irf"], ex["irf"])
    np.testing.assert_array_equal(la["se"], ex["se"])
    assert tsecon.lp(y, s, horizons=4, se="hac")["se_method"] == "hac"


def test_lp_both_with_explicit_lag_augmented_raises_and_teaches():
    y, s = _no_propagation_dgp(n=200)
    with pytest.raises(ValueError, match="statistically invalid"):
        tsecon.lp(y, s, horizons=6, cumulative="both", se="lag_augmented")
    # The error must hand the user the way out.
    with pytest.raises(ValueError, match='se="hac"'):
        tsecon.lp(y, s, horizons=6, cumulative="both", se="lag_augmented")


def test_lp_state_shares_the_both_mode_fix():
    y, s = _no_propagation_dgp(seed=3)
    rng = np.random.default_rng(99)
    ind = (rng.random(len(y)) < 0.5).astype(float)
    r = tsecon.lp_state(
        y, s, ind, horizons=6, n_lag_controls=4, cumulative="both"
    )
    assert r["se_method"] == "hac"
    assert np.asarray(r["se_state1"])[6] > 1.3 * np.asarray(r["se_state1"])[0]
    with pytest.raises(ValueError, match="statistically invalid"):
        tsecon.lp_state(
            y, s, ind, horizons=6, cumulative="both", se="lag_augmented"
        )
    plain = tsecon.lp_state(y, s, ind, horizons=6, n_lag_controls=4)
    assert plain["se_method"] == "lag_augmented"


def test_lp_both_bands_ride_on_hac_ses():
    # All band routes reuse the pointwise se; pre-fix a sup-t band at h=12
    # was NARROWER than an honest pointwise interval. Post-fix the band must
    # be built on the HAC path (se_method stamped, widths grow with h).
    y, s = _no_propagation_dgp(seed=5)
    r = tsecon.lp(
        y, s, horizons=8, n_lag_controls=4, cumulative="both",
        band="sup-t", band_n_sim=20_000,
    )
    assert r["se_method"] == "hac"
    half = (np.asarray(r["upper"]) - np.asarray(r["lower"])) / 2.0
    assert half[8] > 1.5 * half[0]


# --------------------------------------------------------------------------
# Finding 2: cov_se_max_rel_diff is returned
# --------------------------------------------------------------------------
def _persistent_dgp(seed=20260817, n=300):
    rng = np.random.default_rng(seed)
    shock = rng.standard_normal(n)
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = 0.6 * y[t - 1] + shock[t] + 0.5 * rng.standard_normal()
    return y, shock


def test_hac_sup_t_band_reports_the_cov_se_gap():
    y, shock = _persistent_dgp()
    r = tsecon.lp(
        y, shock, horizons=8, n_lag_controls=4, se="hac",
        band="sup-t", band_n_sim=20_000,
    )
    gap = r["cov_se_max_rel_diff"]
    assert gap is not None and np.isfinite(gap) and gap >= 0.0
    # One common Bartlett bandwidth (H+p) serves the whole covariance while
    # the reported se uses maxlags = h+p, so the gap is reconstructible from
    # the public surface and materially non-zero on this design.
    per_h = np.asarray(
        tsecon.lp(y, shock, horizons=8, n_lag_controls=4, se="hac")["se"]
    )
    common = np.asarray(
        tsecon.lp(y, shock, horizons=8, n_lag_controls=4, se="hac", maxlags=12)["se"]
    )
    expected = float(np.max(np.abs(common - per_h) / per_h))
    assert gap == pytest.approx(expected, rel=1e-10)
    assert gap > 0.01


def test_lag_augmented_sup_t_gap_is_machine_noise():
    y, shock = _persistent_dgp(seed=7)
    r = tsecon.lp(
        y, shock, horizons=8, n_lag_controls=4,
        band="sup-t", band_n_sim=20_000,
    )
    assert r["cov_se_max_rel_diff"] is not None
    assert r["cov_se_max_rel_diff"] < 1e-10


def test_smooth_lp_sup_t_reports_the_gap_too():
    y, shock = _persistent_dgp(seed=11)
    r = tsecon.smooth_lp(
        y, shock, horizons=8, n_lag_controls=4, lam=10.0,
        band="sup-t", band_n_sim=20_000,
    )
    # Smooth LP's band covariance IS the delta-method matrix behind se.
    assert r["cov_se_max_rel_diff"] is not None
    assert r["cov_se_max_rel_diff"] < 1e-10


def test_closed_form_routes_return_none_for_the_gap():
    y, shock = _persistent_dgp(seed=13)
    assert (
        tsecon.lp(y, shock, horizons=6, band="sidak")["cov_se_max_rel_diff"]
        is None
    )
    rng = np.random.default_rng(1)
    ind = (rng.random(len(y)) < 0.5).astype(float)
    rs = tsecon.lp_state(y, shock, ind, horizons=6, band="bonferroni")
    assert rs["cov_se_max_rel_diff_state1"] is None
    assert rs["cov_se_max_rel_diff_state0"] is None


# --------------------------------------------------------------------------
# Finding 3: default CV grid is scale-relative
# --------------------------------------------------------------------------
def _smooth_dgp(seed=0, n=250):
    rng = np.random.default_rng(seed)
    shock = rng.standard_normal(n)
    y = np.zeros(n)
    for t in range(n):
        y[t] = sum(0.85 ** h * shock[t - h] for h in range(min(t, 24) + 1))
    y += 1.5 * rng.standard_normal(n)
    return y, shock


def test_default_cv_lambda_selection_is_scale_invariant():
    y, shock = _smooth_dgp()
    base = tsecon.smooth_lp(y, shock, horizons=12, n_lag_controls=4, lam="cv")
    lam0, irf0 = float(base["lambda_used"]), np.asarray(base["irf"])
    grid0 = np.asarray(base["cv_grid"])
    assert len(grid0) == 17
    # Pre-fix, shock*100 pinned lambda_used at the absolute grid's maximum
    # (1e6) and moved the unit-normalized IRF materially; shock*1e4 landed on
    # an interior-but-wrong lambda. Post-fix, over eight decades on each
    # axis: lambda tracks the units exactly and the unit IRF does not move.
    for c in (1e-4, 1e-2, 1e2, 1e4):
        r = tsecon.smooth_lp(
            y, shock * c, horizons=12, n_lag_controls=4, lam="cv"
        )
        assert float(r["lambda_used"]) == pytest.approx(lam0 * c * c, rel=1e-9)
        np.testing.assert_allclose(np.asarray(r["irf"]) * c, irf0, rtol=1e-7)

        r = tsecon.smooth_lp(
            y * c, shock, horizons=12, n_lag_controls=4, lam="cv"
        )
        assert float(r["lambda_used"]) == pytest.approx(lam0, rel=1e-9)
        np.testing.assert_allclose(np.asarray(r["irf"]) / c, irf0, rtol=1e-7)


def test_explicit_lambda_grid_stays_absolute():
    y, shock = _smooth_dgp(seed=4)
    grid = [1.0, 10.0, 100.0]
    r = tsecon.smooth_lp(
        y, shock, horizons=10, n_lag_controls=4, lam="cv", lambda_grid=grid
    )
    np.testing.assert_array_equal(np.asarray(r["cv_grid"]), grid)
    assert float(r["lambda_used"]) in grid
