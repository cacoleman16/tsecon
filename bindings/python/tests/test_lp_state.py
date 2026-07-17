"""Structural tests for the state-dependent (interacted) local-projection
binding `tsecon.lp_state` (Ramey & Zubairy 2018).

There is no state-dependent golden in fixtures/, so these mirror the crate's
own property test (crates/tsecon-lp/tests/properties.rs): on a two-regime DGP
whose shock impact is twice as large in state 1 as in state 0, lp_state must
recover a state-1 impact response that significantly exceeds the state-0 one,
and roughly recover the 2x-vs-1x truth.
"""
import numpy as np
import pytest
import tsecon


def _two_regime_dgp(n=800, burn=100, rho=0.9, seed=31337):
    """s_t = rho s_{t-1} + mult_t e_t with mult = 2 in (lagged) state 1 else 1;
    y_t = s_t + 0.5 w_t. The regime is predetermined (blocks of 25 periods) so
    the lagged indicator governs the impact multiplier."""
    rng = np.random.default_rng(seed)
    total = n + burn
    # Predetermined, balanced regime: alternating blocks of 25 periods.
    regime = ((np.arange(total) // 25) % 2).astype(float)
    e = rng.standard_normal(total)
    w = rng.standard_normal(total)
    s = 0.0
    y = np.empty(total)
    for t in range(total):
        mult = 2.0 if (t > 0 and regime[t - 1] == 1.0) else 1.0
        s = rho * s + mult * e[t]
        y[t] = s + 0.5 * w[t]
    return y[burn:], e[burn:], regime[burn:]


def test_lp_state_high_regime_impact_is_larger():
    # The impact effect (h=0) is 2x in state 1: recover b1 significantly > b0.
    y, e, ind = _two_regime_dgp()
    r = tsecon.lp_state(y, e, ind, horizons=4, n_lag_controls=4)

    b1, b0 = r["irf_state1"][0], r["irf_state0"][0]
    se = np.hypot(r["se_state1"][0], r["se_state0"][0])
    tstat = (b1 - b0) / se
    assert b1 > b0 and tstat > 2.0, f"b1={b1}, b0={b0}, t={tstat}"
    # Loose level check that we recovered roughly the 2-vs-1 truth.
    assert 1.4 < b1 < 2.7, f"state-1 impact {b1} far from 2"
    assert 0.5 < b0 < 1.5, f"state-0 impact {b0} far from 1"


def test_lp_state_shapes_and_keys():
    y, e, ind = _two_regime_dgp()
    r = tsecon.lp_state(y, e, ind, horizons=6, n_lag_controls=4)
    assert set(r) >= {"horizons", "irf_state1", "se_state1", "irf_state0", "se_state0"}
    np.testing.assert_array_equal(np.asarray(r["horizons"]), np.arange(7))
    for k in ("irf_state1", "se_state1", "irf_state0", "se_state0"):
        assert len(r[k]) == 7, k
    # Standard errors are strictly positive at every horizon and in both regimes.
    assert (np.asarray(r["se_state1"]) > 0).all()
    assert (np.asarray(r["se_state0"]) > 0).all()


def test_lp_state_hac_se_path_runs():
    # The plain Newey-West HAC spec also produces positive SEs and the same
    # regime ordering. NOTE: lag augmentation (the default) adds an extra lag
    # of the shock to the regression, so its point IRFs legitimately differ
    # from the plain-HAC spec at horizons >= 1 — only the impact response
    # (h=0, the contemporaneous coefficient) coincides across the two specs.
    y, e, ind = _two_regime_dgp()
    la = tsecon.lp_state(y, e, ind, horizons=4, n_lag_controls=4)
    hac = tsecon.lp_state(y, e, ind, horizons=4, n_lag_controls=4, se="hac")
    # Impact responses agree; later horizons differ by construction.
    np.testing.assert_allclose(hac["irf_state1"][0], la["irf_state1"][0], rtol=1e-9)
    np.testing.assert_allclose(hac["irf_state0"][0], la["irf_state0"][0], rtol=1e-9)
    assert (np.asarray(hac["se_state1"]) > 0).all()
    assert (np.asarray(hac["se_state0"]) > 0).all()
    assert hac["irf_state1"][0] > hac["irf_state0"][0]


def test_lp_state_rejects_non_binary_indicator():
    # The indicator must be 0/1; a fractional value teaches an error.
    y, e, ind = _two_regime_dgp()
    bad = ind.copy()
    bad[10] = 0.5
    with pytest.raises(ValueError):
        tsecon.lp_state(y, e, bad, horizons=4, n_lag_controls=4)
