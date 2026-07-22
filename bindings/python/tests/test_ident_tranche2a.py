"""Python contract + parity tests for tranche-2a: panel_unit_root (LLC/IPS/
Fisher), bvar_ssvs (spike-and-slab Gibbs), zero_sign_svar (RWZ 2010).

Heavy numerics live in the crate goldens. These pin the binding contract and,
where an independent reference runs here, cross-check: Fisher combination vs
statsmodels per-unit ADF p-values, and the recursive special case of
zero_sign_svar vs var_irf(orth=True). Includes a regression for the large-T
SSVS crash (inverse-incomplete-gamma non-convergence at large shape).
"""
import numpy as np
import pytest
import tsecon


def _var2(n, k=2, seed=0):
    rng = np.random.default_rng(seed)
    a = np.array([[0.5, 0.1], [0.0, 0.4]])
    y = np.zeros((n, k))
    for t in range(1, n):
        y[t] = a @ y[t - 1] + rng.standard_normal(k)
    return y


# --------------------------------------------------------------------------- #
# panel_unit_root
# --------------------------------------------------------------------------- #
def _panel(N, T, seed, stationary):
    rng = np.random.default_rng(seed)
    out = np.zeros((N, T))
    for i in range(N):
        e = rng.standard_normal(T)
        if stationary:
            for t in range(1, T):
                out[i, t] = 0.3 * out[i, t - 1] + e[t]
        else:
            out[i] = np.cumsum(e)
    return out


@pytest.mark.parametrize("test", ["ips", "llc", "fisher"])
def test_panel_unit_root_contract(test):
    r = tsecon.panel_unit_root(_panel(8, 80, 0, stationary=False), test=test, lags=1)
    assert {"statistic", "p_value", "per_unit_tstat", "n_units"} <= set(r)
    assert r["n_units"] == 8
    assert 0.0 <= r["p_value"] <= 1.0


def test_panel_unit_root_rejects_stationary_holds_unit_root():
    # a stationary panel should reject the unit-root null; an I(1) panel should not
    st = tsecon.panel_unit_root(_panel(12, 120, 1, stationary=True), test="ips", lags=1)
    ur = tsecon.panel_unit_root(_panel(12, 120, 2, stationary=False), test="ips", lags=1)
    assert st["p_value"] < 0.05
    assert ur["p_value"] > 0.10


def test_panel_fisher_matches_statsmodels_combination():
    sm = pytest.importorskip("statsmodels.tsa.stattools")
    from scipy.stats import chi2

    panel = _panel(10, 150, 3, stationary=False)
    got = tsecon.panel_unit_root(panel, test="fisher", lags=2, regression="c")
    # independent Fisher combination of per-unit ADF p-values
    ps = [sm.adfuller(panel[i], maxlag=2, autolag=None, regression="c")[1] for i in range(10)]
    mw = -2.0 * np.sum(np.log(ps))
    p_mw = chi2.sf(mw, 2 * 10)
    assert got["statistic"] == pytest.approx(mw, rel=1e-6)
    assert got["p_value"] == pytest.approx(p_mw, rel=1e-6)


def test_panel_unit_root_accepts_list_of_uneven_series():
    rng = np.random.default_rng(9)
    units = [np.cumsum(rng.standard_normal(T)) for T in (90, 110, 130)]
    r = tsecon.panel_unit_root(units, test="fisher", lags=1)
    assert r["n_units"] == 3


# --------------------------------------------------------------------------- #
# bvar_ssvs
# --------------------------------------------------------------------------- #
def test_bvar_ssvs_contract_and_reproducibility():
    kw = dict(lags=2, n_draws=400, burn=100, seed=5)
    a = tsecon.bvar_ssvs(_var2(300), **kw)
    b = tsecon.bvar_ssvs(_var2(300), **kw)
    assert {"inclusion_prob", "coef_mean", "sigma_mean", "irf_draws"} <= set(a)
    ip = np.asarray(a["inclusion_prob"])
    assert ip.shape == (5, 2)                       # (1 + k*p) x k
    assert np.all((ip >= 0.0) & (ip <= 1.0))
    # same seed => identical draws
    assert np.array_equal(np.asarray(a["coef_mean"]), np.asarray(b["coef_mean"]))
    assert np.array_equal(ip, np.asarray(b["inclusion_prob"]))


@pytest.mark.parametrize("T", [1500, 4000])
def test_bvar_ssvs_large_T_does_not_crash(T):
    # regression: inverse-incomplete-gamma non-convergence at shape = gamma_a + T/2
    r = tsecon.bvar_ssvs(_var2(T, seed=1), lags=2, n_draws=200, burn=50, seed=1)
    assert np.all(np.isfinite(np.asarray(r["coef_mean"])))


# --------------------------------------------------------------------------- #
# zero_sign_svar — recursive special case vs var_irf(orth=True)
# --------------------------------------------------------------------------- #
def test_zero_sign_recursive_special_case_recovers_cholesky_structure():
    y = _var2(300, k=2, seed=7)
    # strict-upper impact zero + no signs => rotation pinned to Q=I at every
    # posterior draw, so the band posterior tracks the recursive Cholesky. (The
    # machine-precision per-draw identity lives in the crate golden; here the
    # reduced form is drawn from the NIW posterior, so the median only
    # approximates the OLS Cholesky within posterior scatter.)
    r = tsecon.zero_sign_svar(y, [], [(0, 1, 0)], lags=2, horizon=8, n_draws=400, seed=0)
    q = np.asarray(r["quantiles"])            # [h][var][shock][prob], probs[2]=0.50
    # the imposed impact zero is enforced exactly at every quantile
    assert np.allclose(q[0, 0, 1, :], 0.0, atol=1e-10)
    # the posterior median recovers the recursive-Cholesky structure (loose:
    # posterior scatter, not a machine-precision identity)
    median = q[:, :, :, 2]
    chol = np.asarray(tsecon.var_irf(y, lags=2, horizon=8, orth=True))
    assert np.max(np.abs(median - chol)) < 0.05


def test_zero_sign_reproducible():
    y = _var2(300, k=2, seed=7)
    kw = dict(lags=2, horizon=8, n_draws=80, seed=0)
    a = tsecon.zero_sign_svar(y, [], [(0, 1, 0)], **kw)
    b = tsecon.zero_sign_svar(y, [], [(0, 1, 0)], **kw)
    assert np.array_equal(np.asarray(a["quantiles"]), np.asarray(b["quantiles"]))


def test_zero_sign_with_signs_runs():
    y = _var2(300, k=2, seed=3)
    r = tsecon.zero_sign_svar(
        y, [(0, 0, 0, "+")], [(0, 1, 0)], lags=2, horizon=6, n_draws=100, seed=1
    )
    assert r["diagnostics"]["accepted"] >= 1
    assert abs(float(np.sum(np.asarray(r["weights"]))) - 1.0) < 1e-8
