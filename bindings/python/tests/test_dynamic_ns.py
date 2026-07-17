"""Golden + property tests for the dynamic Nelson-Siegel binding.

Dynamic Nelson-Siegel is Diebold & Li's (2006) two-step estimator: fit the
three Nelson-Siegel factors cross-sectionally for every date in a yield panel
at a fixed decay lambda, then fit an independent AR(1) to each factor series
for a one-step-ahead curve forecast.

Reference values are in fixtures/termstructure.json (statsmodels OLS on the
Nelson-Siegel loadings at the Diebold-Li 2006 monthly lambda=0.0609). Row 100
of yields_panel is the single-date golden (yields_date100 / ns_fit_factors /
ns_fit_rsquared), which anchors the per-date cross-sectional fit exactly.
"""
import json
from pathlib import Path

import numpy as np
import tsecon

FIXTURES = Path(__file__).parents[3] / "fixtures"
TS = json.loads((FIXTURES / "termstructure.json").read_text())
MAT = np.array(TS["maturities"])
PANEL = np.array(TS["yields_panel"])  # T x n_maturities
LAM = TS["lambda"]


def _ns_loadings(maturities, lam):
    """The three Nelson-Siegel loading columns [level, slope, curvature]."""
    t = np.asarray(maturities, dtype=float)
    g = (1.0 - np.exp(-lam * t)) / (lam * t)
    h = g - np.exp(-lam * t)
    return np.column_stack([np.ones_like(t), g, h])


def test_dynamic_ns_shapes_and_lambda():
    fit = tsecon.dynamic_ns(PANEL, MAT, decay=LAM)
    factors = np.array(fit["factors"])
    assert factors.shape == (PANEL.shape[0], 3)
    assert len(fit["rsquared"]) == PANEL.shape[0]
    assert len(fit["level"]) == PANEL.shape[0]
    assert len(fit["slope"]) == PANEL.shape[0]
    assert len(fit["curvature"]) == PANEL.shape[0]
    np.testing.assert_allclose(fit["maturities"], MAT, atol=1e-12)
    assert abs(fit["lambda"] - LAM) < 1e-12
    # The level/slope/curvature series are exactly the factor columns.
    np.testing.assert_allclose(fit["level"], factors[:, 0], atol=1e-12)
    np.testing.assert_allclose(fit["slope"], factors[:, 1], atol=1e-12)
    np.testing.assert_allclose(fit["curvature"], factors[:, 2], atol=1e-12)


def test_dynamic_ns_matches_single_date_golden():
    # Row 100 of the panel is the standalone Nelson-Siegel golden.
    fit = tsecon.dynamic_ns(PANEL, MAT, decay=LAM)
    factors = np.array(fit["factors"])
    np.testing.assert_allclose(factors[100], TS["ns_fit_factors"], atol=1e-8)
    assert abs(fit["rsquared"][100] - TS["ns_fit_rsquared"]) < 1e-8


def test_dynamic_ns_level_tracks_long_yield_and_reconstructs():
    fit = tsecon.dynamic_ns(PANEL, MAT, decay=LAM)
    # The level factor is the long-rate proxy: it tracks the longest-maturity
    # yield across the panel.
    long_yield = PANEL[:, int(MAT.argmax())]
    corr = np.corrcoef(np.array(fit["level"]), long_yield)[0, 1]
    assert corr > 0.9
    # The cross-sectional fit reconstructs each curve well: median R^2 > 0.9.
    assert np.median(np.array(fit["rsquared"])) > 0.9
    # Every per-date R^2 is a genuine (<= 1) fit.
    assert np.all(np.array(fit["rsquared"]) <= 1.0 + 1e-9)


def test_dynamic_ns_one_step_forecast():
    fit = tsecon.dynamic_ns(PANEL, MAT, decay=LAM)
    fc = fit["forecast"]
    fc_factors = np.array(fc["factors"])
    fc_yields = np.array(fc["yields"])
    assert fc_factors.shape == (3,)
    assert fc_yields.shape == (len(MAT),)
    # The forecast yields are the forecast factors mapped through the NS
    # loadings on the same maturity grid.
    loadings = _ns_loadings(MAT, LAM)
    np.testing.assert_allclose(fc_yields, loadings @ fc_factors, atol=1e-8)
    # AR(1) coefficients: one intercept/phi per factor.
    assert len(fc["ar1_intercept"]) == 3
    assert len(fc["ar1_phi"]) == 3
    # The level factor is highly persistent (Diebold-Li 2006).
    assert fc["ar1_phi"][0] > 0.9
    # Each factor forecast equals its AR(1) one-step forecast from the last
    # observed factor: fhat = c + phi * last.
    last = np.array(fit["factors"])[-1]
    manual = np.array(fc["ar1_intercept"]) + np.array(fc["ar1_phi"]) * last
    np.testing.assert_allclose(fc_factors, manual, atol=1e-8)
