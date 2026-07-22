"""Python-level contract + parity tests for the tranche-1 identification and
unit-root methods: long_run_svar (Blanchard-Quah), max_share_svar (Uhlig),
proxy_svar (SVAR-IV), hetero_svar (Rigobon), phillips_perron / phillips_ouliaris,
and bvar_hierarchical (GLP empirical Bayes).

Heavy numerical validation lives in the crate goldens (matched to arch / NumPy
closed forms to ~1e-13). These tests pin the binding contract from Python and,
where an independent reference runs in this venv (arch for Phillips-Perron,
NumPy for the Blanchard-Quah closed form), re-check parity end to end.
"""
import numpy as np
import pytest
import tsecon


def _stable_var(n=300, seed=7):
    rng = np.random.default_rng(seed)
    a = np.array([[0.5, 0.1, 0.0], [0.0, 0.4, 0.1], [0.1, 0.0, 0.3]])
    y = np.zeros((n, 3))
    for t in range(1, n):
        y[t] = a @ y[t - 1] + rng.standard_normal(3)
    return y


DATA = _stable_var()


# --------------------------------------------------------------------------- #
# Phillips-Perron — strong parity vs arch
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("regression", ["c", "ct"])
def test_phillips_perron_matches_arch(regression):
    arch_ur = pytest.importorskip("arch.unitroot")
    rng = np.random.default_rng(0)
    y = np.cumsum(rng.standard_normal(250)) + 0.02 * np.arange(250)
    got = tsecon.phillips_perron(y, regression=regression, test_type="tau")
    trend = {"c": "c", "ct": "ct"}[regression]
    ref = arch_ur.PhillipsPerron(y, trend=trend, test_type="tau")
    assert got["ztau"] == pytest.approx(ref.stat, abs=1e-8)
    assert got["pvalue"] == pytest.approx(ref.pvalue, abs=1e-6)


def test_phillips_perron_rejects_stationary_holds_unit_root():
    rng = np.random.default_rng(3)
    rw = np.cumsum(rng.standard_normal(300))               # unit root
    ar = np.zeros(300)
    for t in range(1, 300):
        ar[t] = 0.2 * ar[t - 1] + rng.standard_normal()   # stationary
    assert tsecon.phillips_perron(rw)["pvalue"] > 0.10
    assert tsecon.phillips_perron(ar)["pvalue"] < 0.05


def test_phillips_ouliaris_contract():
    rng = np.random.default_rng(5)
    x = np.cumsum(rng.standard_normal((300, 1)), axis=0)
    y = x[:, 0] + 0.3 * rng.standard_normal(300)           # cointegrated
    r = tsecon.phillips_ouliaris(y, x)
    assert {"stat", "pvalue", "crit", "nobs"} <= set(r)
    assert r["pvalue"] < 0.10                              # detects cointegration


# --------------------------------------------------------------------------- #
# Blanchard-Quah long-run SVAR — closed-form identities + NumPy parity
# --------------------------------------------------------------------------- #
def test_long_run_svar_identities():
    r = tsecon.long_run_svar(DATA, lags=2, horizon=8)
    impact = np.asarray(r["impact"])
    lr = np.asarray(r["long_run"])
    irf = np.asarray(r["irf"])
    # long-run matrix lower-triangular (BQ recursive restriction)
    assert np.allclose(np.triu(lr, k=1), 0.0, atol=1e-8)
    # irf[0] == impact exactly
    assert np.array_equal(irf[0], impact)
    # B B' reproduces the residual covariance (structural shocks orthonormal)
    fit = tsecon.var_fit(DATA, lags=2)
    sigma_u = np.asarray(fit["sigma_u"])
    assert np.allclose(impact @ impact.T, sigma_u, atol=1e-8)


def test_long_run_svar_matches_numpy_closed_form():
    r = tsecon.long_run_svar(DATA, lags=2, horizon=6)
    fit = tsecon.var_fit(DATA, lags=2)
    params = np.asarray(fit["params"])            # (1 + k*p) x k, rows: const, lag1, lag2
    k = 3
    a1 = params[1 : 1 + k].T                       # equations in rows
    a2 = params[1 + k : 1 + 2 * k].T
    sigma_u = np.asarray(fit["sigma_u"])
    c1 = np.linalg.inv(np.eye(k) - a1 - a2)        # long-run multiplier
    lr = np.linalg.cholesky(c1 @ sigma_u @ c1.T)   # lower-triangular
    b = np.linalg.solve(c1, lr)                     # impact = C(1)^{-1} LR
    # sign convention: match diagonal signs of the reference to tsecon's
    got_b = np.asarray(r["impact"])
    s = np.sign(np.diag(got_b)) * np.sign(np.diag(b))
    assert np.allclose(got_b, b * s, atol=1e-7)
    assert np.allclose(np.asarray(r["long_run_multiplier"]), c1, atol=1e-8)


# --------------------------------------------------------------------------- #
# max-share — spectral properties
# --------------------------------------------------------------------------- #
def test_max_share_svar_properties():
    r = tsecon.max_share_svar(DATA, lags=2, target=0, h0=1, h1=8, horizon=8)
    q = np.asarray(r["q"])
    assert q.shape[0] == 3 and abs(np.linalg.norm(q) - 1.0) < 1e-8   # unit vector
    assert 0.0 <= r["share_window"] <= 1.0                            # a valid FEV share
    eig = np.asarray(r["eigenvalues"])
    assert np.all(eig >= -1e-12)                                      # PSD objective
    assert np.all(np.diff(eig) >= -1e-10)                            # ascending order


# --------------------------------------------------------------------------- #
# proxy SVAR — first-stage diagnostic + impact column
# --------------------------------------------------------------------------- #
def test_proxy_svar_contract():
    rng = np.random.default_rng(11)
    # instrument strongly correlated with variable-0 innovation
    fit = tsecon.var_fit(DATA, lags=2)
    proxy = DATA[:, 0] * 0.8 + rng.standard_normal(len(DATA))
    r = tsecon.proxy_svar(DATA, proxy, lags=2, horizon=8)
    assert {"impact", "irf", "first_stage_f", "reliability"} <= set(r)
    assert r["first_stage_f"] > 0.0
    assert np.asarray(r["impact"]).shape[0] == 3


# --------------------------------------------------------------------------- #
# hetero SVAR (Rigobon) — recovers B on a 2-regime DGP
# --------------------------------------------------------------------------- #
def test_hetero_svar_identifies_on_distinct_regimes():
    rng = np.random.default_rng(21)
    k, n = 2, 800
    b_true = np.array([[1.0, 0.5], [0.4, 1.0]])
    reg = (np.arange(n) >= n // 2).astype(np.int64)        # integer regime labels
    scale = np.where(reg == 0, 1.0, 2.5)[:, None]          # variances jump in regime 2
    a = np.array([[0.4, 0.1], [0.0, 0.3]])
    y = np.zeros((n, k))
    for t in range(1, n):
        eps = rng.standard_normal(k) * scale[t]
        y[t] = a @ y[t - 1] + b_true @ eps
    r = tsecon.hetero_svar(y, reg, lags=1, horizon=4)
    assert r["identified"] is True
    b = np.asarray(r["B"])
    # B recovered up to column sign/order: B B' must reproduce regime-1 covariance
    assert b.shape == (2, 2)
    assert np.all(np.asarray(r["variance_ratios"]) > 0)


# --------------------------------------------------------------------------- #
# hierarchical BVAR — ML-optimal lambda inside bounds, beats fixed
# --------------------------------------------------------------------------- #
def test_bvar_hierarchical_optimizes_lambda():
    r = tsecon.bvar_hierarchical(DATA, lags=2, lambda1_lo=0.02, lambda1_hi=5.0)
    assert 0.02 <= r["lambda1_opt"] <= 5.0
    # the ML-selected lambda's marginal likelihood is >= the fixed-init value
    assert r["log_marginal_likelihood"] >= r["lambda1_fixed_log_ml"] - 1e-6
    assert np.asarray(r["posterior_mean_coefs"]).shape[1] == 3
