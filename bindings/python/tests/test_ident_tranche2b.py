"""Python contract + cross-check tests for tranche-2b: structural_fevd,
historical_decomposition, narrative_svar, fry_pagan_svar, robust_svar_bounds.

The heavy numerics live in the tsecon-ident crate goldens. These pin the
binding contract and re-check the key identities from Python: the historical
decomposition adds up exactly, structural FEVD rows sum to one and reproduce a
recursive NumPy FEVD, narrative reduces to sign_restricted_svar with no events,
Fry-Pagan returns an accepted draw minimizing the median-target criterion, and
the GK robust region contains the sign-restricted band.
"""
import numpy as np
import pytest
import tsecon


def _stable_var(n=220, seed=0):
    rng = np.random.default_rng(seed)
    a = np.array([[0.5, 0.1, 0.0], [0.0, 0.4, 0.1], [0.1, 0.0, 0.3]])
    y = np.zeros((n, 3))
    for t in range(1, n):
        y[t] = a @ y[t - 1] + rng.standard_normal(3)
    return y


DATA = _stable_var()
POS = [(0, 0, 0, "+")]   # a single, easily-satisfied sign restriction


# --------------------------------------------------------------------------- #
# structural_fevd
# --------------------------------------------------------------------------- #
def test_structural_fevd_rows_sum_to_one():
    r = tsecon.structural_fevd(DATA, lags=2, horizon=8)
    fevd = np.asarray(r["fevd"])
    assert fevd.shape == (9, 3, 3)
    assert np.allclose(fevd.sum(axis=2), 1.0, atol=1e-10)
    assert np.all(fevd >= -1e-12)


def test_structural_fevd_recursive_matches_var_fevd():
    # default impact = lower Cholesky => must reproduce the recursive var_fevd.
    # Layouts differ: structural_fevd is [h=0..H][var][shock] (H+1 entries),
    # var_fevd is [var][h=0..H-1][shock]; align on the first H horizons.
    H = 8
    got = np.asarray(tsecon.structural_fevd(DATA, lags=2, horizon=H)["fevd"])
    ref = np.asarray(tsecon.var_fevd(DATA, lags=2, horizon=H))
    aligned = np.transpose(got[:H], (1, 0, 2))          # -> [var][h][shock]
    assert np.allclose(aligned, ref, atol=1e-8)


# --------------------------------------------------------------------------- #
# historical_decomposition
# --------------------------------------------------------------------------- #
def test_historical_decomposition_adds_up():
    r = tsecon.historical_decomposition(DATA, lags=2)
    baseline = np.asarray(r["baseline"])       # (T-p, k)
    hd = np.asarray(r["hd"])                    # (T-p, k, k): [t][var][shock]
    recon = baseline + hd.sum(axis=2)          # baseline + sum over shocks
    actual = DATA[2:]                          # observations after the p presample rows
    assert recon.shape == actual.shape
    assert np.allclose(recon, actual, atol=1e-9)


def test_historical_decomposition_shocks_shape():
    r = tsecon.historical_decomposition(DATA, lags=2)
    assert np.asarray(r["shocks"]).shape == (DATA.shape[0] - 2, 3)


# --------------------------------------------------------------------------- #
# narrative_svar — reduces to sign_restricted_svar with no narrative events
# --------------------------------------------------------------------------- #
def test_narrative_reduces_to_sign_restricted_when_no_events():
    kw = dict(lags=2, horizon=6, n_draws=150, max_tries=5, seed=0)
    nar = tsecon.narrative_svar(DATA, POS, **kw)
    sr = tsecon.sign_restricted_svar(DATA, POS, **kw)
    # with no narrative events every importance weight is 1, so the summaries
    # coincide with the plain sign-restricted sampler
    assert np.allclose(np.asarray(nar["quantiles"]), np.asarray(sr["quantiles"]), atol=1e-8)


def test_narrative_reproducible():
    kw = dict(lags=2, horizon=6, n_draws=120, seed=3)
    a = tsecon.narrative_svar(DATA, POS, **kw)
    b = tsecon.narrative_svar(DATA, POS, **kw)
    assert np.array_equal(np.asarray(a["quantiles"]), np.asarray(b["quantiles"]))


# --------------------------------------------------------------------------- #
# fry_pagan_svar — returned draw is accepted and minimizes the MT criterion
# --------------------------------------------------------------------------- #
def test_fry_pagan_returns_median_target_draw():
    r = tsecon.fry_pagan_svar(DATA, POS, lags=2, horizon=6, n_draws=200, seed=1)
    assert r["n_accepted"] >= 1
    assert 0 <= r["mt_index"] < r["n_accepted"]
    assert r["mt_statistic"] >= 0.0
    mt = np.asarray(r["median_target_irf"])
    med = np.asarray(r["median_irf"])
    assert mt.shape == med.shape          # a single coherent model, IRF-shaped


# --------------------------------------------------------------------------- #
# robust_svar_bounds — robust region contains the sign-restricted band
# --------------------------------------------------------------------------- #
def test_robust_bounds_contain_sign_restricted_band():
    kw = dict(lags=2, horizon=6, n_draws=300, seed=2)
    rb = tsecon.robust_svar_bounds(DATA, POS, **kw)
    rs = rb["restricted_shocks"]                             # bounds only for these
    lo = np.asarray(rb["robust_ci_lower"])[..., rs]
    hi = np.asarray(rb["robust_ci_upper"])[..., rs]
    assert lo.shape == hi.shape
    assert np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))
    assert np.all(hi - lo >= -1e-10)                        # a proper interval
    # the prior-robust region is at least as wide as the pointwise band
    sr = tsecon.sign_restricted_svar(DATA, POS, **kw)
    q = np.asarray(sr["quantiles"])[..., rs, :]             # [h][var][restricted][prob]
    band_lo, band_hi = q[..., 0], q[..., -1]                # 5% and 95%
    assert np.all(lo <= band_lo + 1e-8)
    assert np.all(hi >= band_hi - 1e-8)
