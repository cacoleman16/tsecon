"""Tests for the three new Phase-3/4 crate bindings: GAS score-driven
volatility, mean-group / CCE-MG panel estimators, and DFM nowcasting.

panel_mean_group is checked tightly against the crate's independent
statsmodels golden (fixtures/tsecon-panelts.json); GAS and the DFM nowcast
are exercised structurally here (their tight numeric goldens live in the
crates' own Rust tests).
"""
import json
from pathlib import Path

import numpy as np
import pytest
import tsecon

FIXTURES = Path(__file__).parents[3] / "fixtures"
PANEL = json.loads((FIXTURES / "tsecon-panelts.json").read_text())


# ----------------------------------------------------- GAS volatility
def _garch_like(n=800, seed=0):
    rng = np.random.default_rng(seed)
    y = np.empty(n)
    h = 1.0
    for t in range(n):
        h = 0.05 + 0.08 * (y[t - 1] ** 2 if t else 1.0) + 0.90 * h
        y[t] = np.sqrt(h) * rng.standard_normal()
    return y


def test_gas_gaussian_fit_is_sane():
    y = _garch_like(seed=1)
    r = tsecon.gas_volatility(y, density="gaussian", horizon=5)
    assert np.all(np.asarray(r["variance"]) > 0)
    assert len(r["variance"]) == len(y)
    assert np.isfinite(r["loglik"])
    assert 0.0 <= r["b"] < 1.0
    assert r["next_variance"] > 0
    fc = np.asarray(r["forecast"])
    assert fc.shape == (5,) and np.all(fc > 0)


def test_gas_student_t_estimates_dof():
    y = _garch_like(seed=2)
    r = tsecon.gas_volatility(y, density="student_t")
    assert r["nu"] > 2.0  # valid Student-t degrees of freedom
    assert np.isfinite(r["loglik"])
    assert np.all(np.asarray(r["variance"]) > 0)


def test_gas_rejects_unknown_density():
    with pytest.raises(ValueError):
        tsecon.gas_volatility(np.zeros(50), density="cauchy")


# ------------------------------------------------ mean-group panel
def _panel_units():
    ys = [np.array(u) for u in PANEL["y"]]                     # N vectors
    x = PANEL["x"]                                             # [K][N][T]
    n = PANEL["design"]["N"]
    xs = [np.column_stack([x[0][i], x[1][i]]) for i in range(n)]
    return ys, xs


def test_mean_group_matches_golden():
    ys, xs = _panel_units()
    mg = tsecon.panel_mean_group(ys, xs, method="mg")
    np.testing.assert_allclose(mg["coef"], PANEL["mg"]["coef"], atol=1e-9)
    np.testing.assert_allclose(mg["se"], PANEL["mg"]["se"], atol=1e-9)
    np.testing.assert_allclose(mg["tstat"], PANEL["mg"]["tstat"], atol=1e-9)
    assert mg["n_units"] == PANEL["design"]["N"]
    assert mg["k"] == PANEL["design"]["K"]


def test_cce_mean_group_matches_golden_and_beats_mg():
    ys, xs = _panel_units()
    mg = tsecon.panel_mean_group(ys, xs, method="mg")
    cce = tsecon.panel_mean_group(ys, xs, method="cce")
    np.testing.assert_allclose(cce["coef"], PANEL["cce"]["coef"], atol=1e-9)
    # CCE-MG purges the common factor, so it is closer to the true mean slopes
    # than plain MG (the whole point of Pesaran 2006).
    truth = np.array(PANEL["true_mean_slopes"])
    err_mg = np.abs(np.array(mg["coef"]) - truth).sum()
    err_cce = np.abs(np.array(cce["coef"]) - truth).sum()
    assert err_cce < err_mg


# ---------------------------------------------------- DFM nowcast
def _factor_panel(n=160, big_n=12, seed=3):
    rng = np.random.default_rng(seed)
    f = np.zeros(n)
    for t in range(1, n):
        f[t] = 1.2 * f[t - 1] - 0.4 * f[t - 2] if t > 1 else 0.7 * f[t - 1]
        f[t] += rng.standard_normal()
    load = rng.uniform(0.5, 1.5, big_n)
    x = np.outer(f, load) + 0.5 * rng.standard_normal((n, big_n))
    return x, f


def test_dfm_nowcast_balanced_panel():
    x, f = _factor_panel()
    res = tsecon.dfm_nowcast(x, n_factors=1, factor_order=2)
    assert len(res["nowcast"]) == x.shape[1]
    assert np.all(np.isfinite(res["nowcast"]))
    assert np.isfinite(res["loglik"])
    fac = np.array(res["smoothed_factors"])
    assert fac.shape == (x.shape[0], 1)
    # The smoothed factor tracks the true factor (up to sign/scale).
    assert abs(np.corrcoef(fac[:, 0], f)[0, 1]) > 0.85


def test_dfm_nowcast_handles_ragged_edge():
    x, _ = _factor_panel(seed=4)
    ragged = x.copy()
    ragged[-1, :6] = np.nan  # half the series unobserved at the edge
    res = tsecon.dfm_nowcast(ragged, n_factors=1, factor_order=2)
    assert np.all(np.isfinite(res["nowcast"]))  # nowcast still complete
    assert len(res["nowcast"]) == x.shape[1]


def test_dfm_nowcast_mle_fits():
    # The one-step MLE path produces a valid nowcast whose smoothed factor
    # tracks the truth. (The tight loglik-maximization check lives in the
    # crate's Rust tests, which compare on a common data scaling; two_step and
    # mle standardize/centre differently, so their loglik values are not
    # directly comparable here.) Small panel + factor_order=1 keeps the
    # debug-build MLE fit quick.
    x, f = _factor_panel(n=90, big_n=5, seed=6)
    mle = tsecon.dfm_nowcast(x, n_factors=1, factor_order=1, method="mle")
    assert np.isfinite(mle["fit_loglik"])
    assert np.all(np.isfinite(mle["nowcast"]))
    fac = np.array(mle["smoothed_factors"])
    assert fac.shape == (x.shape[0], 1)
    assert abs(np.corrcoef(fac[:, 0], f)[0, 1]) > 0.85


def test_dfm_mle_rejects_multiple_factors():
    x, _ = _factor_panel(n=100, big_n=6, seed=7)
    with pytest.raises(ValueError):
        tsecon.dfm_nowcast(x, n_factors=2, factor_order=2, method="mle")
