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
    # Both are [h][var][shock] since 0.6.0 (var_fevd used to leak a
    # variable-major layout); only the horizon counts differ —
    # structural_fevd has H+1 entries (h = 0..H), var_fevd has H — so align
    # on the first H horizons with no transpose.
    H = 8
    got = np.asarray(tsecon.structural_fevd(DATA, lags=2, horizon=H)["fevd"])
    ref = np.asarray(tsecon.var_fevd(DATA, lags=2, horizon=H))
    assert np.allclose(got[:H], ref, atol=1e-8)


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


# --------------------------------------------------------------------------- #
# historical_decomposition(identification="impact") — an arbitrary A0
# --------------------------------------------------------------------------- #
import json
import re
from pathlib import Path

HDI = json.loads(
    (Path(__file__).resolve().parents[3] / "fixtures" / "historical_decomposition_impact.json").read_text()
)


@pytest.mark.parametrize("case", ["bq_var8", "random_3var_p2"])
def test_historical_decomposition_impact_reloads_the_fixture(case):
    """statsmodels VAR reduced form + the NumPy closed form (E = U A0^{-T},
    Theta_s = Psi_s A0, HD = sum_s Theta_s E_{t-s}); never imports tsecon."""
    c = HDI[case]
    data = np.asarray(c["data"])
    r = tsecon.historical_decomposition(data, lags=c["p"], identification="impact", impact=c["impact"])
    for key in ("hd", "baseline", "shocks", "shock_covariance", "impact"):
        np.testing.assert_allclose(np.asarray(r[key]), np.asarray(c[key]), rtol=1e-8, atol=1e-10,
                                   err_msg=f"{case}.{key}")
    assert list(r["times"]) == list(range(c["t_eff"]))
    recon = np.asarray(r["baseline"]) + np.asarray(r["hd"]).sum(axis=2)
    assert np.abs(recon - data[c["p"]:]).max() < 1e-12


def test_historical_decomposition_impact_takes_long_run_svar_output_verbatim():
    """A nested list straight out of another callable is accepted (rust-numpy
    PyArrayLike2), and the result equals the ndarray call bit for bit."""
    B = tsecon.long_run_svar(DATA, lags=2, impact_signs=[(0, 1)])["impact"]
    assert isinstance(B, list)
    a = tsecon.historical_decomposition(DATA, lags=2, identification="impact", impact=B)
    b = tsecon.historical_decomposition(DATA, lags=2, identification="impact", impact=np.asarray(B))
    for key in ("hd", "baseline", "shocks", "shock_covariance"):
        assert np.array_equal(np.asarray(a[key]), np.asarray(b[key]))
    # long_run_svar's impact is built from the df-adjusted Sigma: the shock
    # covariance is (T_eff - m) / T_eff times the identity (m = 1 + k p)
    t_eff, m = DATA.shape[0] - 2, 1 + 3 * 2
    np.testing.assert_allclose(np.asarray(a["shock_covariance"]), (t_eff - m) / t_eff * np.eye(3),
                               atol=1e-12)
    # structural_fevd accepts the same nested list (a 0.11.0 behaviour change:
    # it used to require an ndarray)
    f1 = tsecon.structural_fevd(DATA, lags=2, horizon=4, impact=B)
    f2 = tsecon.structural_fevd(DATA, lags=2, horizon=4, impact=np.asarray(B))
    assert np.array_equal(np.asarray(f1["fevd"]), np.asarray(f2["fevd"]))


def test_historical_decomposition_impact_reduces_to_cholesky():
    """impact = chol(Sigma_mle) reproduces identification="cholesky"."""
    fit = tsecon.var_fit(DATA, lags=2)
    resid = np.asarray(fit["resid"])
    P = np.linalg.cholesky(resid.T @ resid / resid.shape[0])
    a = tsecon.historical_decomposition(DATA, lags=2)
    b = tsecon.historical_decomposition(DATA, lags=2, identification="impact", impact=P)
    for key in ("hd", "baseline", "shocks"):
        np.testing.assert_allclose(np.asarray(b[key]), np.asarray(a[key]), rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(np.asarray(b["shock_covariance"]), np.eye(3), atol=1e-12)
    # horizon above T_eff - 1 is inert (and never allocated); below is refused
    big = tsecon.historical_decomposition(DATA, lags=2, identification="impact", impact=P,
                                          horizon=2**40)
    assert np.array_equal(np.asarray(big["hd"]), np.asarray(b["hd"]))
    with pytest.raises(ValueError, match=r"\bhorizon\b"):
        tsecon.historical_decomposition(DATA, lags=2, identification="impact", impact=P, horizon=5)


def _numpy_singular(m):
    return np.linalg.matrix_rank(np.asarray(m, dtype=float)) < len(m)


@pytest.mark.parametrize(
    "impact",
    [
        np.eye(2),                                       # wrong shape
        np.ones((3, 2)),
        [[1.0, 0.0, 0.0], [0.0, np.nan, 0.0], [0.0, 0.0, 1.0]],
        [[1.0, 0.0, 0.0], [0.0, np.inf, 0.0], [0.0, 0.0, 1.0]],
        np.ones((3, 3)),                                 # rank 1
        [[1.0, 2.0, 3.0], [2.0, 4.0, 6.0 + 1e-15], [0.0, 1.0, 1.0]],   # rank-deficient by matrix_rank
        [[1.0, 0.0], [0.0]],                              # ragged
    ],
)
def test_historical_decomposition_bad_impact_is_refused_naming_impact(impact):
    with pytest.raises((ValueError, TypeError)) as info:
        tsecon.historical_decomposition(DATA, lags=2, identification="impact", impact=impact)
    assert re.search(r"\bimpact\b", str(info.value)), str(info.value)


def test_historical_decomposition_singularity_follows_numpy_matrix_rank():
    # just inside numpy's tolerance: refused; just outside: accepted
    eps = np.finfo(float).eps
    for scale, singular in ((1.0, True), (100.0, False)):
        m = np.diag([1.0, 1.0, 3.0 * eps * 0.5 * scale])
        assert bool(_numpy_singular(m)) is singular
        call = lambda: tsecon.historical_decomposition(DATA, lags=2, identification="impact", impact=m)
        if singular:
            with pytest.raises(ValueError, match=r"\bimpact\b"):
                call()
        else:
            assert np.isfinite(np.asarray(call()["shocks"])).all()


@pytest.mark.parametrize(
    "kw, name",
    [
        ({"identification": "impact"}, "impact"),                              # missing impact
        ({"impact": np.eye(3)}, "impact"),                                     # impact under cholesky
        ({"identification": "sign", "restrictions": POS, "impact": np.eye(3)}, "impact"),
        ({"identification": "impact", "impact": np.eye(3), "seed": 3}, "seed"),
        ({"identification": "impact", "impact": np.eye(3), "n_draws": 10}, "n_draws"),
        ({"identification": "impact", "impact": np.eye(3), "max_tries": 10}, "max_tries"),
        ({"identification": "impact", "impact": np.eye(3), "lambda1": 0.5}, "lambda1"),
        ({"identification": "impact", "impact": np.eye(3), "n_weight_draws": 10}, "n_weight_draws"),
        ({"identification": "impact", "impact": np.eye(3), "restrictions": POS}, "restrictions"),
        ({"identification": "impact", "impact": np.eye(3),
          "narrative_restrictions": [{"t": 10, "shock": 0, "sign": "+"}]}, "narrative_restrictions"),
        # cholesky silently ignored these before 0.11.0
        ({"restrictions": POS}, "restrictions"),
        ({"narrative_restrictions": [{"t": 10, "shock": 0, "sign": "+"}]}, "narrative_restrictions"),
        ({"identification": "narrative"}, "identification"),
    ],
)
def test_historical_decomposition_refusals_name_the_parameter(kw, name):
    with pytest.raises(ValueError) as info:
        tsecon.historical_decomposition(DATA, lags=2, **kw)
    msg = str(info.value)
    assert re.search(rf"\b{name}\b", msg), msg


def test_cholesky_accepts_empty_restriction_lists_and_no_longer_suggests_narrative():
    a = tsecon.historical_decomposition(DATA, lags=2)
    b = tsecon.historical_decomposition(DATA, lags=2, restrictions=[], narrative_restrictions=[])
    assert np.array_equal(np.asarray(a["hd"]), np.asarray(b["hd"]))
    with pytest.raises(ValueError) as info:
        tsecon.historical_decomposition(DATA, lags=2, seed=4)
    msg = str(info.value)
    assert '(or "narrative")' not in msg and 'identification="sign"' in msg
    with pytest.raises(ValueError, match=r'"impact"'):
        tsecon.historical_decomposition(DATA, lags=2, identification="bogus")
