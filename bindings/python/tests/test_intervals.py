"""Interval API audit tests: every confidence/credible band the bindings
produce must match mean +/- z * se at the *requested* coverage, and
coverage must be caller-configurable.

Two-sided normal multipliers used here come from scipy.stats.norm.ppf so
the assertions pin the exact quantile math (e.g. alpha=0.05 -> 1.9600,
alpha=0.10 -> 1.6449, alpha=0.32 -> 0.9945).
"""
import json
from pathlib import Path

import numpy as np
import pytest
import tsecon
from scipy.stats import norm

FIXTURES = Path(__file__).parents[3] / "fixtures"
DIAG = json.loads((FIXTURES / "diagnostics.json").read_text())
NILE = np.array(DIAG["nile"])
VARFX = json.loads((FIXTURES / "var.json").read_text())
MACRO = np.array(VARFX["data_100dlog_gdp_cons_inv"])


def z(alpha):
    """Two-sided normal multiplier for a 1-alpha interval."""
    return norm.ppf(1.0 - alpha / 2.0)


# ---------------------------------------------------------------------------
# var_forecast: alpha flows through to the interval construction
# ---------------------------------------------------------------------------

def test_var_forecast_endpoints_are_mean_pm_z_se_at_requested_alpha():
    base = tsecon.var_forecast(MACRO, lags=2, steps=8, alpha=0.05)
    point = np.array(base["point"])
    # Recover the implied standard errors from the 95% band...
    se = (np.array(base["upper"]) - point) / z(0.05)
    assert (se > 0).all()
    # ...and demand every other alpha reproduce point +/- z(alpha) * se.
    for alpha in (0.01, 0.10, 0.32, 0.50):
        fc = tsecon.var_forecast(MACRO, lags=2, steps=8, alpha=alpha)
        np.testing.assert_allclose(fc["point"], point, rtol=1e-12)
        np.testing.assert_allclose(fc["upper"], point + z(alpha) * se, rtol=1e-10)
        np.testing.assert_allclose(fc["lower"], point - z(alpha) * se, rtol=1e-10)


def test_var_forecast_bands_symmetric_and_widths_ordered():
    fc32 = tsecon.var_forecast(MACRO, lags=2, steps=8, alpha=0.32)
    fc05 = tsecon.var_forecast(MACRO, lags=2, steps=8, alpha=0.05)
    for fc in (fc32, fc05):
        lo, hi, pt = np.array(fc["lower"]), np.array(fc["upper"]), np.array(fc["point"])
        np.testing.assert_allclose((lo + hi) / 2.0, pt, rtol=1e-10)
    w32 = np.array(fc32["upper"]) - np.array(fc32["lower"])
    w05 = np.array(fc05["upper"]) - np.array(fc05["lower"])
    assert (w32 < w05).all()  # 68% band strictly narrower than 95%
    # The width ratio is exactly the quantile ratio (same se cancels).
    np.testing.assert_allclose(w32 / w05, z(0.32) / z(0.05), rtol=1e-10)


def test_var_forecast_rejects_invalid_alpha():
    for bad in (0.0, 1.0, -0.1, 1.7):
        with pytest.raises(ValueError, match="alpha"):
            tsecon.var_forecast(MACRO, lags=2, steps=8, alpha=bad)


# ---------------------------------------------------------------------------
# arima_fit: conf_alpha round-trip
# ---------------------------------------------------------------------------

def test_arima_conf_alpha_round_trip():
    r = tsecon.arima_fit(NILE, p=1, d=0, q=1, constant=True,
                         forecast_steps=12, conf_alpha=0.10)
    mean, se = r["forecast_mean"], r["forecast_se"]
    assert r["conf_alpha"] == 0.10
    np.testing.assert_allclose(r["forecast_lower"], mean - z(0.10) * se, rtol=1e-12)
    np.testing.assert_allclose(r["forecast_upper"], mean + z(0.10) * se, rtol=1e-12)


def test_arima_conf_alpha_coverage_is_configurable():
    runs = {
        alpha: tsecon.arima_fit(NILE, p=1, d=0, q=1, constant=True,
                                forecast_steps=6, conf_alpha=alpha)
        for alpha in (0.05, 0.32)
    }
    for alpha, r in runs.items():
        np.testing.assert_allclose(
            r["forecast_upper"] - r["forecast_lower"],
            2.0 * z(alpha) * r["forecast_se"], rtol=1e-12)
    w68 = runs[0.32]["forecast_upper"] - runs[0.32]["forecast_lower"]
    w95 = runs[0.05]["forecast_upper"] - runs[0.05]["forecast_lower"]
    assert (w68 < w95).all()


def test_arima_conf_alpha_default_none_adds_no_bands():
    r = tsecon.arima_fit(NILE, p=1, d=0, q=1, constant=True, forecast_steps=6)
    assert "forecast_lower" not in r and "forecast_upper" not in r
    assert "forecast_mean" in r and "forecast_se" in r


def test_arima_conf_alpha_validation():
    with pytest.raises(ValueError, match="forecast_steps"):
        tsecon.arima_fit(NILE, p=1, d=0, q=1, conf_alpha=0.05)  # no forecast
    for bad in (0.0, 1.0, -0.2, 2.0):
        with pytest.raises(ValueError, match="alpha"):
            tsecon.arima_fit(NILE, p=1, d=0, q=1, forecast_steps=4, conf_alpha=bad)


# ---------------------------------------------------------------------------
# Coverage semantics of the SE-producing (interval-free) APIs
# ---------------------------------------------------------------------------

def test_acf_bartlett_se_are_ses_not_bands():
    """acf returns raw Bartlett SEs; the caller picks the multiplier, so
    coverage is configurable by construction. Pin se(r_1) = 1/sqrt(n)."""
    r = tsecon.acf(NILE, nlags=5)
    assert r["bartlett_se"][0] == 0.0
    assert r["bartlett_se"][1] == pytest.approx(1.0 / np.sqrt(len(NILE)), rel=1e-12)
    assert (np.diff(r["bartlett_se"][1:]) >= -1e-15).all()  # Bartlett widening


def test_garch_variance_forecast_is_point_path_only():
    """garch_fit's variance forecast is a point path: no lower/upper keys
    may appear (no interval is defined for it)."""
    ret = np.array(json.loads((FIXTURES / "garch.json").read_text())["returns"])
    r = tsecon.garch_fit(ret, vol="garch", mean="zero", forecast_horizon=5)
    assert "variance_forecast" in r
    assert not any("lower" in k or "upper" in k for k in r.keys())


def test_bvar_irf_draws_quantile_bands_scale_with_coverage():
    """Credible bands from raw draws: the caller-chosen quantile pair sets
    the coverage, so a 90% band must contain the 68% band pointwise."""
    data = np.array(json.loads((FIXTURES / "bvar_niw.json").read_text())["data"])
    draws = np.array(tsecon.bvar_irf_draws(data, lags=2, horizon=8, n_draws=200, seed=11))
    lo90, hi90 = np.quantile(draws, [0.05, 0.95], axis=0)
    lo68, hi68 = np.quantile(draws, [0.16, 0.84], axis=0)
    assert (lo90 <= lo68).all() and (hi68 <= hi90).all()
    med = np.quantile(draws, 0.5, axis=0)
    assert (lo90 <= med).all() and (med <= hi90).all()


# --- cumulative IRF views (added when wiring the cumulative flag) ---
import json as _json
from pathlib import Path as _Path
_FIX = _Path(__file__).parents[3] / "fixtures"
_MACRO = np.array(_json.loads((_FIX / "var.json").read_text())["data_100dlog_gdp_cons_inv"])
_BVARDATA = np.array(_json.loads((_FIX / "bvar_niw.json").read_text())["data"])


def test_var_irf_cumulative_is_running_sum():
    level = np.array(tsecon.var_irf(_MACRO, lags=2, horizon=10, orth=True, cumulative=False))
    cum = np.array(tsecon.var_irf(_MACRO, lags=2, horizon=10, orth=True, cumulative=True))
    np.testing.assert_allclose(cum, np.cumsum(level, axis=0), atol=1e-12)


def test_bvar_irf_draws_cumulative_bands_not_summed():
    # Cumulative credible bands come from cumulating each draw then quantiling —
    # they must NOT equal the cumulative sum of the level bands.
    lvl = np.array(tsecon.bvar_irf_draws(_BVARDATA, lags=2, horizon=8, n_draws=600, seed=1))
    cum = np.array(tsecon.bvar_irf_draws(_BVARDATA, lags=2, horizon=8, n_draws=600, seed=1, cumulative=True))
    # The MEAN is linear, so cumulate-then-mean == cumsum of means exactly.
    np.testing.assert_allclose(cum.mean(0), np.cumsum(lvl.mean(0), axis=0), atol=1e-10)
    # But the 90% band width of the cumulative is NOT the cumsum of the level
    # band widths — because the summed responses are correlated across horizons.
    # That correlation is exactly why you must cumulate the draws, not the bands.
    lvl_w = np.quantile(lvl, 0.95, 0) - np.quantile(lvl, 0.05, 0)
    cum_w = np.quantile(cum, 0.95, 0) - np.quantile(cum, 0.05, 0)
    assert not np.allclose(cum_w, np.cumsum(lvl_w, axis=0), atol=1e-6)
