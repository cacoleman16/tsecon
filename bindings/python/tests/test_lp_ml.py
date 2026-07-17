"""Golden tests for the local-projections and penalized-regression bindings,
against the same fixtures the Rust crates validate on (statsmodels/
linearmodels for LP, sklearn for penalized regression)."""
import json
from pathlib import Path

import numpy as np
import pytest
import tsecon

FIX = Path(__file__).parents[3] / "fixtures"
LP = json.loads((FIX / "lp.json").read_text())
ML = json.loads((FIX / "ml.json").read_text())


def test_lp_hac_matches_statsmodels():
    y, e = np.array(LP["y"]), np.array(LP["e"])
    r = tsecon.lp(y, e, horizons=8, n_lag_controls=LP["n_lag_controls"], se="hac")
    for case in LP["ols_lp"]:
        h = case["h"]
        assert r["irf"][h] == pytest.approx(case["beta"], rel=1e-9), f"h={h}"
        assert r["se"][h] == pytest.approx(case["se_hac"], rel=1e-7), f"h={h}"


def test_lp_lag_augmented_covers_true_irf():
    # The default (lag-augmented) inference must cover the true IRF 0.9^h.
    y, e = np.array(LP["y"]), np.array(LP["e"])
    true_irf = np.array(LP["true_irf"])
    r = tsecon.lp(y, e, horizons=8, n_lag_controls=LP["n_lag_controls"])  # default se
    for h in range(len(true_irf)):
        assert abs(r["irf"][h] - true_irf[h]) < 4 * r["se"][h] + 0.05, f"h={h}"


def test_lp_iv_matches_linearmodels():
    y, x, z = np.array(LP["y"]), np.array(LP["x"]), np.array(LP["z"])
    r = tsecon.lp_iv(y, x, z, horizons=8, n_lag_controls=LP["n_lag_controls"])
    for case in LP["iv_lp"]:
        h = case["h"]
        assert r["irf"][h] == pytest.approx(case["beta"], rel=1e-7), f"h={h}"
        assert r["se"][h] == pytest.approx(case["se_kernel"], rel=1e-5), f"h={h}"
    assert (np.asarray(r["first_stage_f"]) > 0).all()


def test_lp_cumulative_differs_from_level():
    y, e = np.array(LP["y"]), np.array(LP["e"])
    lvl = tsecon.lp(y, e, horizons=8, n_lag_controls=4)
    cum = tsecon.lp(y, e, horizons=8, n_lag_controls=4, cumulative=True)
    # cumulative point IRF ~ running sum of level IRF, but the SEs are NOT the
    # cumulative sum of level SEs (that is the whole point of cumulating the LHS)
    np.testing.assert_allclose(cum["irf"], np.cumsum(lvl["irf"]), rtol=0.05)
    assert not np.allclose(cum["se"], np.cumsum(lvl["se"]), atol=1e-6)


def test_ridge_lasso_elasticnet_match_sklearn():
    X = np.array(ML["X_standardized"])
    y = np.array(ML["y_centered"])
    for case in ML["cases"]:
        p = case["params"]
        if case["name"].startswith("ridge"):
            coef = tsecon.ridge(X, y, alpha=p["alpha"])
        elif case["name"].startswith("lasso"):
            coef = tsecon.lasso(X, y, alpha=p["alpha"])["coef"]
        else:  # elastic net
            coef = tsecon.elastic_net(X, y, alpha=p["alpha"], l1_ratio=p["l1_ratio"])["coef"]
        np.testing.assert_allclose(coef, case["coef"], atol=1e-6, err_msg=case["name"])
