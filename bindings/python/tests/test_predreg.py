"""Golden tests for the predictive-regression / IVX bindings against the
crate's documented-formula golden (fixtures/predreg.json).

The statistical correctness of IVX (uniform-over-persistence size) is
established by the crate's own Monte-Carlo property tests; here we check the
Python surface reproduces the published point estimates and Wald statistics.
"""
import json
from pathlib import Path

import numpy as np
import tsecon

FIXTURES = Path(__file__).parents[3] / "fixtures"
PR = json.loads((FIXTURES / "predreg.json").read_text())


def test_scalar_predictive_regression_matches_golden():
    sc = PR["scalar"]
    r = np.array(sc["r"])
    x = np.array(sc["x"])
    res = tsecon.predictive_regression(r, x)  # defaults cz=-1, alpha=0.95

    # OLS predictive regression.
    assert abs(res["ols"]["beta"] - sc["ols"]["beta_ols"]) < 1e-6
    assert abs(res["ols"]["se"] - sc["ols"]["se"]) < 1e-6
    assert abs(res["ols"]["tstat"] - sc["ols"]["tstat"]) < 1e-6

    # Stambaugh bias correction.
    stb = sc["stambaugh"]
    assert abs(res["stambaugh"]["beta_corrected"] - stb["beta_corrected"]) < 1e-6
    assert abs(res["stambaugh"]["bias_term"] - stb["bias_term"]) < 1e-6
    assert abs(res["stambaugh"]["rho_ols"] - stb["rho_ols"]) < 1e-6
    # The correction pulls the biased OLS slope toward zero here.
    assert abs(res["stambaugh"]["beta_corrected"]) < abs(res["ols"]["beta"])

    # IVX estimator + Wald test.
    iv = sc["ivx"]
    assert abs(res["ivx"]["beta_ivx"] - iv["beta_ivx"]) < 1e-6
    assert abs(res["ivx"]["wald"] - iv["wald"]) < 1e-5
    assert abs(res["ivx"]["rz"] - iv["Rz"]) < 1e-9
    assert res["ivx"]["pvalue"] < 0.001  # strongly significant on this design


def test_multi_ivx_joint_test_matches_golden():
    mu = PR["multi"]
    r = np.array(mu["r"])
    xs = np.column_stack([mu["x1"], mu["x2"]])
    res = tsecon.ivx_test(r, xs)
    np.testing.assert_allclose(res["beta_ivx"], mu["ivx"]["beta_ivx"], atol=1e-6)
    assert abs(res["wald"] - mu["ivx"]["wald"]) < 1e-5
    assert abs(res["pvalue"] - mu["ivx"]["pvalue"]) < 1e-5
    assert res["nregressors"] == 2
