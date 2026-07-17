"""Golden tests for the MIDAS and multivariate-GARCH bindings."""
import json
from pathlib import Path

import numpy as np
import pytest
import tsecon

FIX = Path(__file__).parents[3] / "fixtures"
MIDAS = json.loads((FIX / "midas.json").read_text())
MG = json.loads((FIX / "mgarch.json").read_text())


def test_midas_weights_match_fixture():
    wg = MIDAS["weight_goldens"]
    np.testing.assert_allclose(
        tsecon.midas_weights("exp_almon", 0.1, -0.05, 6), wg["exp_almon_0.1_-0.05_K6"], atol=1e-10)
    np.testing.assert_allclose(
        tsecon.midas_weights("beta", 2.0, 3.0, 10), wg["beta_2_3_K10"], atol=1e-10)
    for w in [tsecon.midas_weights("exp_almon", 0.2, -0.1, 8), tsecon.midas_weights("beta", 1.5, 4.0, 6)]:
        assert abs(np.sum(w) - 1.0) < 1e-12  # normalized


def test_umidas_matches_ols():
    y = np.array(MIDAS["y"])
    X = np.array(MIDAS["X_stacked"]).T   # (nobs, K)
    r = tsecon.umidas(y, X, se_type="nonrobust")
    fx = MIDAS["umidas_ols"]
    np.testing.assert_allclose(r["params"], fx["params"], rtol=1e-8)
    assert r["rsquared"] == pytest.approx(fx["rsquared"], rel=1e-8)


def test_ccc_garch_runs():
    returns = np.array(MG["returns"]).T  # (T, k)
    r = tsecon.ccc_garch(returns)
    C = np.array(r["correlation"])
    assert C.shape == (3, 3)
    np.testing.assert_allclose(np.diag(C), 1.0, atol=1e-8)   # unit diagonal
    assert np.allclose(C, C.T)                               # symmetric
    assert (np.linalg.eigvalsh(C) > 0).all()                 # positive definite
    assert np.isfinite(r["loglik"])


def test_dcc_garch_recovers_persistence():
    returns = np.array(MG["returns"]).T
    r = tsecon.dcc_garch(returns)
    assert 0 <= r["a"] and 0 <= r["b"] and r["a"] + r["b"] < 1  # valid DCC
    # Simulation recovery: true a=0.03, b=0.95 => persistence ~0.98.
    assert abs((r["a"] + r["b"]) - (MG["true"]["a_dcc"] + MG["true"]["b_dcc"])) < 0.06
    Clast = np.array(r["correlation_last"])
    assert (np.linalg.eigvalsh(Clast) > 0).all()
