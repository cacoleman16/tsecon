"""Golden tests for the term-structure bindings: Nelson-Siegel (Diebold-Li)
and Svensson yield-curve fits.

Reference values are in fixtures/termstructure.json (statsmodels OLS on the
Nelson-Siegel loadings at the Diebold-Li 2006 monthly lambda=0.0609).
"""
import json
from pathlib import Path

import numpy as np
import tsecon

FIXTURES = Path(__file__).parents[3] / "fixtures"
TS = json.loads((FIXTURES / "termstructure.json").read_text())
MAT = np.array(TS["maturities"])
YLD = np.array(TS["yields_date100"])
LAM = TS["lambda"]


def test_nelson_siegel_matches_ols_golden():
    fit = tsecon.nelson_siegel(MAT, YLD, decay=LAM)
    np.testing.assert_allclose(fit["factors"], TS["ns_fit_factors"], atol=1e-8)
    assert abs(fit["level"] - TS["ns_fit_factors"][0]) < 1e-8
    assert abs(fit["slope"] - TS["ns_fit_factors"][1]) < 1e-8
    assert abs(fit["curvature"] - TS["ns_fit_factors"][2]) < 1e-8
    assert abs(fit["rsquared"] - TS["ns_fit_rsquared"]) < 1e-8
    assert abs(fit["lambda"] - LAM) < 1e-12
    assert len(fit["residuals"]) == len(MAT)


def test_nelson_siegel_optimal_lambda_improves_fit():
    # Estimating the decay by NLS can only do at least as well as the fixed
    # Diebold-Li lambda on the same curve.
    fixed = tsecon.nelson_siegel(MAT, YLD, decay=LAM)
    opt = tsecon.nelson_siegel(MAT, YLD, decay=LAM, optimal_lambda=True)
    assert opt["rsquared"] >= fixed["rsquared"] - 1e-9
    assert opt["lambda"] > 0.0


def test_svensson_nests_nelson_siegel():
    # With a second decay far from the first, Svensson (4 factors) fits at
    # least as well as Nelson-Siegel (3 factors) on the same curve.
    ns = tsecon.nelson_siegel(MAT, YLD, decay=LAM)
    sv = tsecon.svensson(MAT, YLD, LAM, LAM * 3.0)
    assert len(sv["factors"]) == 4
    assert sv["rsquared"] >= ns["rsquared"] - 1e-9
