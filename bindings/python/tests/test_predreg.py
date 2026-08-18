"""Golden tests for the predictive-regression / IVX bindings against the
crate's documented-formula golden (fixtures/predreg.json).

The statistical correctness of IVX (uniform-over-persistence size) is
established by the crate's own Monte-Carlo property tests; here we check the
Python surface reproduces the published point estimates and Wald statistics.
"""
import json
from pathlib import Path

import numpy as np
import pytest
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


# --------------------------------------------------------------------------- #
# ivx_test joint="bonferroni" — the many-predictor escape hatch (audit 3-4/6)
# --------------------------------------------------------------------------- #
def _stambaugh_panel(seed=3, n=300, k=4, rho=0.99, cue=-0.9, beta=None):
    """k persistent predictors; endogeneity carried by predictor 0."""
    rng = np.random.default_rng(seed)
    e = rng.standard_normal((n, k))
    x = np.zeros((n, k))
    for t in range(1, n):
        x[t] = rho * x[t - 1] + e[t]
    u = cue * e[:, 0] + np.sqrt(1 - cue * cue) * rng.standard_normal(n)
    r = u.copy()
    if beta is not None:
        r[1:] += x[:-1] @ np.asarray(beta)
    return r, x


def test_ivx_test_bonferroni_is_the_scalar_tests_combined():
    """Each wald_scalar[j] must be exactly predictive_regression's ivx wald on
    column j, and the joint p-value exactly min(1, k * min_j p_j) — the
    union-intersection construction, nothing more."""
    r, x = _stambaugh_panel()
    k = x.shape[1]
    res = tsecon.ivx_test(r, x, joint="bonferroni")
    assert res["joint"] == "bonferroni"
    for j in range(k):
        scalar = tsecon.predictive_regression(r, x[:, j])["ivx"]
        assert res["wald_scalar"][j] == scalar["wald"]
        assert abs(res["pvalue_scalar"][j] - scalar["pvalue"]) < 1e-15
    assert res["wald"] == max(res["wald_scalar"])
    assert abs(res["pvalue"] - min(1.0, k * min(res["pvalue_scalar"]))) < 1e-15
    # The slope vector is still the joint IVX estimator (shared with chi2 mode).
    chi2 = tsecon.ivx_test(r, x)
    np.testing.assert_array_equal(res["beta_ivx"], chi2["beta_ivx"])
    assert res["nobs"] == chi2["nobs"] and res["rz"] == chi2["rz"]


def test_ivx_test_default_key_set_is_unchanged():
    """joint="chi2" (the default) must return exactly the historical keys —
    the bonferroni extras appear only when asked for."""
    r, x = _stambaugh_panel(k=2)
    assert set(tsecon.ivx_test(r, x)) == {
        "beta_ivx", "wald", "pvalue", "rz", "nregressors", "nobs",
    }
    assert set(tsecon.ivx_test(r, x, joint="bonferroni")) == {
        "beta_ivx", "wald", "pvalue", "rz", "nregressors", "nobs",
        "wald_scalar", "pvalue_scalar", "joint",
    }


def test_ivx_test_bonferroni_specializes_to_scalar_at_k1():
    r, x = _stambaugh_panel(k=1)
    res = tsecon.ivx_test(r, x, joint="bonferroni")
    scalar = tsecon.predictive_regression(r, x[:, 0])["ivx"]
    assert res["wald"] == scalar["wald"]
    assert abs(res["pvalue"] - scalar["pvalue"]) < 1e-15


def test_ivx_test_unknown_joint_raises_with_both_options_named():
    r, x = _stambaugh_panel(k=2)
    with pytest.raises(ValueError, match="bonferroni"):
        tsecon.ivx_test(r, x, joint="hotelling")
