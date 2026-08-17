"""Tests for prophet_lite — run with pytest from the shared venv:

    /home/user/tsecon/.venv/bin/python -m pytest lab/prophet_lite/tests.py -q

Everything is seeded; tolerances are stated next to each assertion with the
reasoning.  Note on the tau direction: tau is the LAPLACE PRIOR SCALE
(Prophet's changepoint_prior_scale), so the L1 weight is lam = sigma^2/tau —
tau -> 0 means an infinite penalty (NO active changepoints) and tau large
means a vanishing penalty (many active).  The path test below asserts that
monotonicity in the prior-scale parametrization.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prophet_lite import fit  # noqa: E402
from prophet_lite.model import fourier_features, lasso_cd  # noqa: E402


# ----------------------------------------------------------------------------
# DGP helpers (seeded)
# ----------------------------------------------------------------------------

def piecewise_trend_dgp(n=300, cp=120, base=10.0, slope1=0.5, slope2=-0.3,
                        noise=0.5, seed=7):
    rng = np.random.default_rng(seed)
    u = np.arange(n, dtype=float)
    trend = base + slope1 * np.minimum(u, cp) + slope2 * np.maximum(u - cp, 0.0)
    return trend + rng.normal(0.0, noise, n), trend


def seasonal_dgp(n=240, period=12.0, K=3, seed=11, noise=0.1):
    rng = np.random.default_rng(seed)
    u = np.arange(n, dtype=float)
    beta = np.array([1.2, -0.7, 0.4, 0.3, -0.2, 0.1])  # sin1,cos1,...,sin3,cos3
    seas = fourier_features(u, period, K) @ beta
    trend = 5.0 + 0.02 * u
    return trend + seas + rng.normal(0.0, noise, n), trend, seas, beta


# ----------------------------------------------------------------------------
# 1. trend recovery: changepoint located, slopes recovered
# ----------------------------------------------------------------------------

def test_trend_changepoint_location_and_slopes():
    y, _ = piecewise_trend_dgp()
    res = fit(y)  # integer index, no seasonality

    # changepoint located: the largest |delta| candidate sits near u=120.
    # Candidate grid spacing is 0.8*300/25 = 9.6 indices -> tolerance 15
    # allows the mass to land on the nearest or next-nearest candidate.
    j = int(np.argmax(np.abs(res["delta"])))
    cp_index = res["changepoint_indices"][j]
    assert abs(cp_index - 120) <= 15, f"changepoint found at {cp_index}, truth 120"

    # slopes recovered: numerical slope of the fitted trend component,
    # averaged over windows well away from the kink (tolerance 0.05 against
    # true slopes 0.5 / -0.3; noise sd 0.5 over 80-120 point windows).
    trend = res.components()["trend"]
    d = np.diff(trend)
    assert abs(d[20:100].mean() - 0.5) < 0.05
    assert abs(d[160:280].mean() - (-0.3)) < 0.05


# ----------------------------------------------------------------------------
# 2. seasonality recovery
# ----------------------------------------------------------------------------

def test_seasonality_fourier_recovery():
    y, _, seas_true, beta_true = seasonal_dgp()
    res = fit(y, [(12, 3)])  # explicit (period, K) pair, integer index

    # Fourier coefficients: fitted betas are in scaled-y units; original-
    # scale coefficients are beta * y_scale (unit-amplitude sin/cos basis).
    beta_hat = np.asarray(res["beta_season"]) * res["y_scale"]
    assert np.max(np.abs(beta_hat - beta_true)) < 0.1, beta_hat

    # component reconstruction to tolerance (noise sd 0.1, n=240)
    seas_hat = res.components()["seasonal"]
    rmse = np.sqrt(np.mean((seas_hat - seas_true) ** 2))
    assert rmse < 0.08, f"seasonal RMSE {rmse:.4f}"


# ----------------------------------------------------------------------------
# 3. interval calibration on the DGP (generous band, seeded)
# ----------------------------------------------------------------------------

def test_interval_coverage_roughly_nominal():
    n, h, M = 400, 40, 500
    cp, base, s1, s2, noise = 250, 20.0, 0.30, -0.10, 0.8
    rng = np.random.default_rng(21)
    u = np.arange(n + h, dtype=float)
    seas_full = 2.0 * np.sin(2 * np.pi * u / 20.0) + 1.0 * np.cos(4 * np.pi * u / 20.0)
    trend_full = base + s1 * np.minimum(u, cp) + s2 * np.maximum(u - cp, 0.0)
    signal = trend_full + seas_full
    y = signal[:n] + rng.normal(0.0, noise, n)

    res = fit(y, [(20, 2)])
    fc = res.forecast(h, level=[0.8, 0.95], n_draws=2000, seed=3)

    lo80, hi80 = fc["lower"]["0.8"], fc["upper"]["0.8"]
    lo95, hi95 = fc["lower"]["0.95"], fc["upper"]["0.95"]

    # sanity: 95% band strictly contains the 80% band on average
    assert np.all(lo95 <= lo80) and np.all(hi95 >= hi80)

    # Monte Carlo future truths from the DGP (trend continues, no new breaks)
    truth = signal[n:][None, :] + rng.normal(0.0, noise, (M, h))
    cov80 = np.mean((truth >= lo80[None, :]) & (truth <= hi80[None, :]))
    cov95 = np.mean((truth >= lo95[None, :]) & (truth <= hi95[None, :]))

    # Generous bands: the changepoint bootstrap adds trend variance the
    # no-new-break truth does not have, so mild over-coverage is expected
    # (a known Prophet property, noted in README.md).  We require rough
    # calibration, not exactness.
    assert 0.60 <= cov80 <= 0.995, f"80% coverage {cov80:.3f}"
    assert 0.80 <= cov95 <= 1.0, f"95% coverage {cov95:.3f}"


# ----------------------------------------------------------------------------
# 4. L1 path in the prior scale tau
# ----------------------------------------------------------------------------

def test_l1_path_in_tau():
    # two genuine changepoints, moderate noise
    rng = np.random.default_rng(5)
    n = 300
    u = np.arange(n, dtype=float)
    trend = (5.0 + 0.4 * np.minimum(u, 100)
             - 0.4 * np.clip(u - 100, 0.0, 100.0)
             + 0.3 * np.maximum(u - 200, 0.0))
    y = trend + rng.normal(0.0, 0.8, n)

    n_active = {tau: fit(y, tau=tau)["n_active"] for tau in (1e-4, 0.05, 10.0)}

    # tau -> 0: lam = sigma^2/tau -> inf: essentially no active changepoints
    assert n_active[1e-4] <= 2, n_active
    # tau large: lam -> 0: most of the 25 candidates go active
    assert n_active[10.0] >= 10, n_active
    # monotone along the path (widely separated taus; exact zeros from CD)
    assert n_active[1e-4] <= n_active[0.05] <= n_active[10.0], n_active


def test_lasso_cd_matches_kkt_on_toy():
    # tiny independent check of the exact solver on a random toy problem
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 8))
    beta = np.array([2.0, 0.0, -1.5, 0.0, 0.0, 1.0, 0.0, 0.0])
    y = X @ beta + rng.normal(0.0, 0.1, 50)
    lam = 5.0
    d, _, ok = lasso_cd(X, y, lam)
    assert ok
    g = X.T @ (y - X @ d)
    active = d != 0
    assert np.all(np.abs(g[~active]) <= lam + 1e-6)
    assert np.allclose(g[active], lam * np.sign(d[active]), atol=1e-6)


# ----------------------------------------------------------------------------
# 5. deterministic seeding & API plumbing
# ----------------------------------------------------------------------------

def test_predictive_draws_deterministic_seeding():
    y, _ = piecewise_trend_dgp(seed=3)
    res = fit(y)
    a = res.predictive_draws(20, n_draws=64, seed=42)
    b = res.predictive_draws(20, n_draws=64, seed=42)
    c = res.predictive_draws(20, n_draws=64, seed=43)
    assert a.shape == (64, 20)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    # forecast intervals reuse the same generator: seed-stable too
    f1 = res.forecast(20, n_draws=256, seed=1)
    f2 = res.forecast(20, n_draws=256, seed=1)
    assert np.array_equal(f1["lower"]["0.8"], f2["lower"]["0.8"])


def test_daily_dates_auto_seasonality_and_dict_roundtrip():
    import pandas as pd
    n = 800
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    u = np.arange(n, dtype=float)
    rng = np.random.default_rng(9)
    y = (10.0 + 0.01 * u + 3.0 * np.sin(2 * np.pi * u / 365.25)
         + 0.8 * np.sin(2 * np.pi * u / 7.0) + rng.normal(0.0, 0.3, n))
    res = fit(y, dates)
    # daily index, span >= 2y and >= 2w: yearly (K=10) and weekly (K=3) on
    assert set(res["seasonalities"]) == {"yearly", "weekly"}
    assert res["seasonalities"]["yearly"]["K"] == 10
    assert res["seasonalities"]["weekly"]["K"] == 3

    comp = res.components()
    yr = comp["seasonal_yearly"]
    # injected yearly amplitude 3 recovered within 10%
    assert abs(0.5 * (yr.max() - yr.min()) - 3.0) < 0.3

    fc = res.forecast(14, level=0.8, n_draws=200, seed=0)
    assert fc["dates"] is not None and len(fc["dates"]) == 14
    assert np.all(np.diff(fc["dates"]).astype("timedelta64[D]").astype(int) == 1)

    # plain-dict friendliness: forecasting works from a bare dict copy
    from prophet_lite import forecast_from_result
    bare = dict(res)
    fc2 = forecast_from_result(bare, 14, level=0.8, n_draws=200, seed=0)
    assert np.allclose(fc["mean"], fc2["mean"])


def test_extra_regressors_recovered_and_required():
    rng = np.random.default_rng(13)
    n = 250
    u = np.arange(n, dtype=float)
    x = rng.normal(size=(n, 1))
    y = 3.0 + 0.05 * u + 2.5 * x[:, 0] + rng.normal(0.0, 0.2, n)
    res = fit(y, X=x)
    # coefficient on the ORIGINAL regressor = beta_extra*y_scale/x_std
    b = float(res["beta_extra"][0]) * res["y_scale"] / float(res["x_std"][0])
    assert abs(b - 2.5) < 0.1, b
    with pytest.raises(ValueError):
        res.forecast(5)  # X_future required
    xf = rng.normal(size=(5, 1))
    fc = res.forecast(5, X_future=xf, n_draws=100, seed=0)
    assert fc["mean"].shape == (5,)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
