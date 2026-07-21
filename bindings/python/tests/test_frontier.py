"""Binding tests for the frontier slice: quantile/GaR, functional shocks,
Bai-Perron breaks, and smooth local projections.

Golden pins reuse the crate fixtures (statsmodels QuantReg for the quantile
family — a genuine independent package; brute-force NumPy enumeration for the
break DP; scipy-BSpline/NumPy normal equations for smooth LP; numpy.linalg.eigh
for the FPCA). Structural tests cover what a fixture cannot: the FLP scenario
reconstruction identity, break recovery on a fresh DGP, and the smooth-LP
lambda->0 internal-consistency limit against the shipped lp().
"""
import json
from pathlib import Path

import numpy as np
import pytest

import tsecon

FIXTURES = Path(__file__).resolve().parents[3] / "fixtures"


def _load(name):
    return json.loads((FIXTURES / name).read_text())


# --------------------------------------------------------------------------- #
# quantile regression — pinned to statsmodels QuantReg
# --------------------------------------------------------------------------- #
QF = _load("tsecon-quantile.json")


@pytest.mark.parametrize("case", QF["qreg"], ids=lambda c: c["name"])
def test_quantile_regression_matches_statsmodels(case):
    y = np.asarray(case["y"], float)
    X = np.column_stack([np.asarray(c, float) for c in case["columns"]])
    taus = np.array([f["tau"] for f in case["fits"]])
    out = tsecon.quantile_regression(y, X, taus)
    params = np.asarray(out["params"])
    bse = np.asarray(out["bse"])
    for i, f in enumerate(case["fits"]):
        np.testing.assert_allclose(params[i], f["params"], rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(bse[i], f["bse"], rtol=1e-6, atol=1e-8)


def test_growth_at_risk_orders_quantiles_and_reports_crossing():
    rng = np.random.default_rng(3)
    n = 250
    cond = rng.standard_normal(n)
    # location-scale DGP: the condition moves the spread, the ABG mechanism
    y = 0.3 * cond + (1.0 + 0.6 * np.abs(cond)) * rng.standard_normal(n)
    g = tsecon.growth_at_risk(
        y, np.column_stack([cond]), horizon=1,
        taus=np.array([0.1, 0.25, 0.5, 0.75, 0.9]), rearrange=True,
    )
    fitted = np.asarray(g["fitted"])
    # after rearrangement the conditional quantiles are monotone at every t
    assert np.all(np.diff(fitted, axis=0) >= -1e-12)
    assert len(np.asarray(g["current"])) == 5


# --------------------------------------------------------------------------- #
# Bai-Perron — pinned to the brute-force fixture + fresh-DGP recovery
# --------------------------------------------------------------------------- #
BF = _load("tsecon-breaks.json")


def test_bai_perron_matches_the_bruteforce_fixture():
    c = BF["bai_perron_case"]
    y = np.asarray(c["y"], float)
    x = np.column_stack([np.asarray(col, float) for col in c["x"]])
    out = tsecon.bai_perron(y, x, max_breaks=c["max_breaks"], trim=c["trim"])
    assert int(out["n_breaks"]) == c["n_breaks"]
    np.testing.assert_array_equal(np.asarray(out["break_dates"], int), c["break_dates"])
    np.testing.assert_allclose(np.asarray(out["ssr_path"]), c["ssr_path"], rtol=1e-8)


def test_bai_perron_selects_zero_breaks_on_stable_data():
    c = BF["bai_perron_null_case"]
    y = np.asarray(c["y"], float)
    x = np.column_stack([np.asarray(col, float) for col in c["x"]])
    out = tsecon.bai_perron(y, x, max_breaks=c["max_breaks"], trim=c["trim"])
    assert int(out["n_breaks"]) == c["n_breaks"]


def test_bai_perron_recovers_fresh_mean_shifts():
    rng = np.random.default_rng(11)
    y = np.concatenate(
        [rng.standard_normal(90), rng.standard_normal(90) + 2.5,
         rng.standard_normal(90) - 1.5]
    )
    out = tsecon.bai_perron(y, np.ones((270, 1)), max_breaks=4, trim=0.15)
    assert int(out["n_breaks"]) == 2
    dates = np.asarray(out["break_dates"], int)
    assert abs(dates[0] - 89) <= 2 and abs(dates[1] - 179) <= 2


def test_sup_f_detects_and_locates_a_break():
    rng = np.random.default_rng(5)
    y = np.concatenate([rng.standard_normal(100), rng.standard_normal(100) + 1.5])
    out = tsecon.sup_f_test(y, np.ones((200, 1)), trim=0.15)
    assert out["p_value"] < 0.01
    assert abs(int(out["break_date"]) - 99) <= 3


def test_sup_f_null_is_not_rejected_wildly():
    rng = np.random.default_rng(7)
    rejections = 0
    for _ in range(30):
        y = rng.standard_normal(200)
        out = tsecon.sup_f_test(y, np.ones((200, 1)), trim=0.15)
        rejections += out["p_value"] < 0.05
    assert rejections <= 6  # ~5% nominal; generous MC slack at 30 reps


# --------------------------------------------------------------------------- #
# functional shocks — FPCA golden + the scenario reconstruction identity
# --------------------------------------------------------------------------- #
FF = _load("tsecon-funcshock.json")


@pytest.mark.parametrize("i", range(len(FF["fpca"])))
def test_functional_pca_matches_numpy_eigh(i):
    c = FF["fpca"][i]
    curves = np.asarray(c["curves"], float)
    out = tsecon.functional_pca(curves, n_factors=c["n_factors"])
    np.testing.assert_allclose(
        np.asarray(out["eigenfunctions"]), c["eigenfunctions"], rtol=1e-8, atol=1e-10
    )
    np.testing.assert_allclose(np.asarray(out["scores"]), c["scores"], rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(np.asarray(out["explained"]), c["explained"], rtol=1e-8)


def test_flp_scenario_reconstruction_identity():
    """A scenario equal to the j-th eigenfunction must reproduce the j-th
    score's coefficient path exactly — the identity that makes the functional
    machinery trustworthy rather than approximate."""
    rng = np.random.default_rng(2)
    n, M, K = 260, 10, 3
    curves = rng.standard_normal((n, M)) @ np.diag(np.linspace(1.5, 0.3, M))
    y = np.zeros(n)
    sc_true = tsecon.functional_pca(curves, n_factors=K)
    scores = np.asarray(sc_true["scores"])
    for t in range(1, n):
        y[t] = 0.4 * y[t - 1] + 0.7 * scores[t, 0] - 0.3 * scores[t, 1] + rng.standard_normal()

    eig = np.asarray(sc_true["eigenfunctions"])
    joint = tsecon.flp(y, scores, horizons=4, n_lag_controls=1)
    betas = np.asarray(joint["betas"])  # [h][k]
    for j in range(K):
        scen = tsecon.flp_scenario(
            y, curves, eig[j], n_factors=K, horizons=4, n_lag_controls=1
        )
        np.testing.assert_allclose(
            np.asarray(scen["response"]), betas[:, j], rtol=1e-8, atol=1e-10
        )


# --------------------------------------------------------------------------- #
# smooth LP — golden + the lambda -> 0 internal-consistency limit
# --------------------------------------------------------------------------- #
SF = _load("smoothlp.json")


@pytest.mark.parametrize("case", ["case_a", "case_b"])
def test_smooth_lp_matches_the_scipy_numpy_golden(case):
    c = SF[case]
    y = np.asarray(c["y"], float)
    e = np.asarray(c["e"], float)
    for ref in c["smooth"]:  # a list of {lambda, theta, irf, se} entries
        out = tsecon.smooth_lp(
            y, e, horizons=c["horizons"], n_lag_controls=c["n_lag_controls"],
            lam=ref["lambda"], degree=c["degree"], n_basis=c["n_basis"],
            penalty_order=c["penalty_order"], hac_maxlags=c["hac_bandwidth"],
        )
        np.testing.assert_allclose(np.asarray(out["irf"]), ref["irf"], rtol=1e-7, atol=1e-9)
        np.testing.assert_allclose(np.asarray(out["se"]), ref["se"], rtol=1e-7, atol=1e-9)


def test_smooth_lp_lambda_zero_reproduces_raw_lp():
    """lam=0 with the interpolating basis is EXACTLY the per-horizon LP on the
    same (HAC, non-augmented) design — machine-precision internal consistency.
    The comparison uses lp(se="hac") deliberately: the lag-augmented default
    adds an extra lag to the design, which changes the point estimates."""
    rng = np.random.default_rng(9)
    n = 260
    shock = rng.standard_normal(n)
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = 0.5 * y[t - 1] + 0.8 * shock[t] + rng.standard_normal()
    sl = tsecon.smooth_lp(y, shock, horizons=8, n_lag_controls=2, lam=0.0)
    raw = tsecon.lp(y, shock, horizons=8, n_lag_controls=2, se="hac")
    np.testing.assert_allclose(
        np.asarray(sl["irf"]), np.asarray(raw["irf"]), rtol=1e-9, atol=1e-12
    )


def test_smooth_lp_large_lambda_flattens_curvature():
    rng = np.random.default_rng(13)
    n = 260
    shock = rng.standard_normal(n)
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = 0.5 * y[t - 1] + 0.8 * shock[t] + rng.standard_normal()
    tight = tsecon.smooth_lp(y, shock, horizons=10, n_lag_controls=2, lam=1e8)
    loose = tsecon.smooth_lp(y, shock, horizons=10, n_lag_controls=2, lam=0.0)
    dd = lambda v: np.abs(np.diff(np.asarray(v), n=2)).max()
    assert dd(tight["irf"]) < dd(loose["irf"]) * 0.5  # penalty visibly smooths
