"""Golden tests for the depth bindings: realized volatility / HAR-RV,
Diebold-Yilmaz connectedness, and the PCA factor model.

Reference values are in fixtures/{realized,connect,favar}.json, generated
by fixtures/generate_depth_fixtures.py from statsmodels / numpy.
"""
import json
from pathlib import Path

import numpy as np
import tsecon

FIXTURES = Path(__file__).parents[3] / "fixtures"
REAL = json.loads((FIXTURES / "realized.json").read_text())
CONN = json.loads((FIXTURES / "connect.json").read_text())
FAV = json.loads((FIXTURES / "favar.json").read_text())


# --------------------------------------------- realized volatility / HAR
def test_realized_measures_match_bns():
    small = np.array(REAL["measures_small"]["returns"])
    m = tsecon.realized_measures(small)
    assert abs(m["rv"] - REAL["measures_small"]["rv"]) < 1e-12
    assert abs(m["bipower"] - REAL["measures_small"]["bipower"]) < 1e-12
    # Jump component is the truncated RV - BV.
    assert m["jump"] >= 0.0
    assert abs(m["jump"] - max(m["rv"] - m["bipower"], 0.0)) < 1e-12


def test_har_rv_matches_statsmodels_hac():
    rv = np.array(REAL["rv_series"])
    fit = tsecon.har_rv(rv, start=REAL["har"]["start"])
    np.testing.assert_allclose(fit["params"], REAL["har"]["params"], atol=1e-8)
    np.testing.assert_allclose(fit["bse"], REAL["har"]["bse"], atol=1e-8)
    assert abs(fit["rsquared"] - REAL["har"]["rsquared"]) < 1e-8
    assert len(fit["params"]) == 4  # const + daily + weekly + monthly


# -------------------------------------------------------- connectedness
def test_connectedness_matches_gfevd_golden():
    data = np.array(CONN["data"]).T  # fixture stores columns
    res = tsecon.connectedness(data, lags=CONN["lags"], horizon=CONN["horizon"])
    assert abs(res["total"] - CONN["total_connectedness"]) < 1e-9
    np.testing.assert_allclose(res["to_others"], CONN["to_others"], atol=1e-9)
    np.testing.assert_allclose(res["from_others"], CONN["from_others"], atol=1e-9)
    np.testing.assert_allclose(res["gfevd"], CONN["gfevd_normalized"], atol=1e-9)
    # Each GFEVD row is a normalized variance decomposition -> sums to 1.
    for row in res["gfevd"]:
        assert abs(sum(row) - 1.0) < 1e-9
    # Net is to - from, and sums to zero across the system.
    net = np.array(res["net"])
    np.testing.assert_allclose(
        net, np.array(res["to_others"]) - np.array(res["from_others"]), atol=1e-9
    )
    assert abs(net.sum()) < 1e-9


# ---------------------------------------------------------- factor model
def test_factor_model_matches_pca():
    xs = np.array(FAV["X_standardized"]).T  # n x big_n, already standardized
    res = tsecon.factor_model(xs, n_factors=FAV["true_r"], kmax=8)
    # Eigenvalues match numpy SVD (S^2 / n).
    np.testing.assert_allclose(res["eigenvalues"], FAV["eigenvalues"], atol=1e-6)
    # Factors are identified up to sign -> compare magnitudes.
    factors = np.array(res["factors"])
    np.testing.assert_allclose(np.abs(factors[:, 0]), FAV["pc1_abs"], atol=1e-5)
    np.testing.assert_allclose(np.abs(factors[:, 1]), FAV["pc2_abs"], atol=1e-5)
    loadings = np.array(res["loadings"])
    np.testing.assert_allclose(
        np.abs(loadings[:, 0]), FAV["loadings_pc1_abs"], atol=1e-5
    )
    # The Ahn-Horenstein eigenvalue ratio recovers the true factor count;
    # Bai-Ng over-selects in this small cross-section (N=24), by design.
    assert res["er"] == FAV["true_r"]
    assert res["icp2"] >= FAV["true_r"]
