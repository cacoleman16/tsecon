"""`tsecon.arima_fit(method="css")` — conditional-sum-of-squares ARIMA (0.11;
crate tsecon-arima `fit_css`).

Re-pins fixtures/arima_css.json (AR(2) against statsmodels AutoReg; the
ARMA/SARIMA/MA/ARIMA optimum against a SciPy Levenberg-Marquardt reference
whose objective is pinned to statsforecast's port of R's ARIMA_CSS), checks
the new `method`/`nobs` keys on both methods, that the default path's keys
and values are those of the exact-MLE fit, the CSS drift-uncertainty band's
use of the CSS Var(c_hat), the refusals, the ARIMAResults facade, and the
docstring/stub contracts.
"""
import ast
import json
import re
from pathlib import Path

import numpy as np
import pytest
import tsecon
from tsecon.results._arima import ARIMAResults

REPO = Path(__file__).resolve().parents[3]
FIX = json.loads((REPO / "fixtures" / "arima_css.json").read_text(encoding="utf-8"))
PYI = REPO / "bindings" / "python" / "python" / "tsecon" / "__init__.pyi"


def _stub_doc(name):
    tree = ast.parse(PYI.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_docstring(node) or ""
    raise AssertionError(f"{name} not in the stub")


def _tokens(text):
    return set(re.findall(r"`([A-Za-z_][A-Za-z_0-9]*)`", text or ""))


def _kw(case):
    p, d, q = case["order"]
    kw = dict(p=p, d=d, q=q, constant=case["constant"], method="css")
    if case["seasonal"] is not None:
        kw["seasonal"] = tuple(case["seasonal"])
    return kw


def test_ar2_css_is_autoreg_ols():
    c = FIX["ar_autoreg"]
    out = tsecon.arima_fit(np.asarray(c["y"]), p=2, method="css")
    assert out["method"] == "css"
    assert out["param_names"] == c["param_names"]
    np.testing.assert_allclose(out["params"], c["params"], rtol=1e-6)
    assert out["params"][-1] == pytest.approx(c["sigma2"], rel=1e-9)
    for key in ("loglik", "aic", "bic"):
        assert out[key] == pytest.approx(c[key], rel=1e-9), key
    assert out["nobs"] == c["nobs"] == len(c["y"]) - 2


@pytest.mark.parametrize("case", FIX["cases"], ids=lambda c: c["name"])
def test_css_optimum_matches_the_reference(case):
    y = np.asarray(case["y"])
    out = tsecon.arima_fit(y, **_kw(case))
    o = case["optimum"]
    assert out["converged"]
    assert out["param_names"] == case["param_names"]
    np.testing.assert_allclose(out["params"], o["params"], rtol=1e-6)
    for key in ("loglik", "aic", "bic"):
        assert out[key] == pytest.approx(o[key], rel=1e-9), key
    np.testing.assert_allclose(out["bse"], o["bse"], rtol=1e-4)
    assert out["nobs"] == o["nobs"]
    assert out["cov_ok"] and np.all(out["se_valid"])
    # sigma2 at the optimum is the statsforecast arima_css value there.
    sf = next(a for a in case["fixed_point_statsforecast"] if a["point"] == "optimum")
    assert out["params"][-1] == pytest.approx(sf["sigma2"], rel=1e-9)


def test_nobs_counts_the_likelihood_observations_for_both_methods():
    case = next(c for c in FIX["cases"] if c["name"] == "sarima100_100_4_c")
    y = np.asarray(case["y"])
    kw = _kw(case)
    css = tsecon.arima_fit(y, **kw)
    kw["method"] = "mle"
    mle = tsecon.arima_fit(y, **kw)
    assert mle["nobs"] == len(y)  # d = D = 0
    assert css["nobs"] == len(y) - (1 + 4 * 1)  # less p + s*P
    rw = np.cumsum(np.random.default_rng(2).standard_normal(120))
    m = tsecon.arima_fit(rw, p=1, d=1, q=0)
    c = tsecon.arima_fit(rw, p=1, d=1, q=0, method="css")
    assert (m["nobs"], c["nobs"]) == (119, 118)
    assert len(m["residuals"]) == len(c["residuals"]) == 119  # Kalman residuals either way


def test_default_method_is_the_exact_mle_and_only_adds_two_keys():
    y = np.asarray(FIX["cases"][0]["y"])
    a = tsecon.arima_fit(y, p=1, q=1, forecast_steps=4, conf_alpha=0.1)
    b = tsecon.arima_fit(y, p=1, q=1, forecast_steps=4, conf_alpha=0.1, method="mle")
    assert a["method"] == "mle" and isinstance(a["nobs"], int)
    assert set(a) == set(b)
    for k in a:
        if isinstance(a[k], np.ndarray):
            np.testing.assert_array_equal(a[k], b[k])
        else:
            assert a[k] == b[k] or (a[k] is None and b[k] is None), k
    # auto_arima's documented key list is untouched: the keys live in arima_fit.
    auto = tsecon.auto_arima(y[:150], max_p=1, max_q=1)
    assert "method" not in auto and "nobs" not in auto


def test_css_and_mle_differ_but_agree_asymptotically():
    case = next(c for c in FIX["cases"] if c["name"] == "arma11_c")
    y = np.asarray(case["y"])
    css = tsecon.arima_fit(y, p=1, q=1, method="css")
    mle = tsecon.arima_fit(y, p=1, q=1)
    assert not np.array_equal(css["params"], mle["params"])
    np.testing.assert_allclose(css["params"][1:3], mle["params"][1:3], atol=0.05)


def test_css_drift_band_uses_the_css_var_c_hat():
    rng = np.random.default_rng(4)
    y = np.cumsum(0.3 + rng.standard_normal(80))
    h = 12
    out = tsecon.arima_fit(y, p=0, d=1, q=0, method="css", forecast_steps=h, drift_uncertainty=True)
    base = tsecon.arima_fit(y, p=0, d=1, q=0, method="css", forecast_steps=h)
    var_c = np.asarray(out["param_cov"])[0, 0]
    steps = np.arange(1, h + 1)
    # ARIMA(0,1,0)+c: d yhat / d c = h, so se_h^2 = se_known^2 + h^2 Var(c_hat).
    np.testing.assert_allclose(out["forecast_se"] ** 2, base["forecast_se"] ** 2 + steps**2 * var_c,
                               rtol=1e-9)
    # The CSS Var(c_hat) of a random walk with drift is sigma2 / n_c.
    n_c = out["nobs"]
    assert n_c == len(y) - 1
    assert var_c == pytest.approx(out["params"][-1] / n_c, rel=1e-5)
    np.testing.assert_array_equal(out["forecast_mean"], base["forecast_mean"])


def test_unknown_method_is_refused_by_name():
    y = np.asarray(FIX["cases"][0]["y"])
    with pytest.raises(ValueError, match=r'unknown method "cs".*"mle".*"css"'):
        tsecon.arima_fit(y, method="cs")


def test_css_on_too_short_a_series_says_so():
    with pytest.raises(ValueError, match=r'method="css"\).*conditional-sum-of-squares'):
        tsecon.arima_fit(np.arange(5.0) ** 1.5, p=2, q=1, method="css")


def test_existing_refusals_still_apply_under_css():
    y = np.asarray(FIX["cases"][0]["y"])
    with pytest.raises(ValueError, match="drift_uncertainty"):
        tsecon.arima_fit(y, p=1, method="css", drift_uncertainty=True)
    with pytest.raises(ValueError, match="constant"):
        tsecon.arima_fit(y, p=1, method="css", forecast_steps=3, constant=False, drift_uncertainty=True)


# --------------------------------------------------------------- facade
def test_facade_nobs_label_and_dict_contract_under_css():
    case = next(c for c in FIX["cases"] if c["name"] == "arima111")
    y = np.asarray(case["y"])
    raw = tsecon.arima_fit(y, 1, 1, 1, constant=False, method="css", forecast_steps=3)
    res = ARIMAResults.fit(y, 1, 1, 1, constant=False, method="css", forecast_steps=3)
    assert set(res) == set(raw)
    assert res.nobs == res["nobs"] == raw["nobs"] == len(y) - 2
    assert res.method == "css"
    text = res.summary()
    assert "by CSS" in text and "cond. log-likelihood" in text
    assert f"{res['loglik']:.4f}" in text
    mle = ARIMAResults.fit(y, 1, 1, 1, constant=False)
    assert mle.method == "mle" and mle.nobs == mle["nobs"] == len(y) - 1
    assert "cond. log-likelihood" not in mle.summary() and "by CSS" not in mle.summary()


# --------------------------------------------------------------- contracts
def test_every_returned_key_is_named_in_both_docstrings():
    case = FIX["cases"][0]
    y = np.asarray(case["y"])
    keys = set(tsecon.arima_fit(y, p=1, q=1, method="css", forecast_steps=3, conf_alpha=0.1,
                                drift_uncertainty=True))
    assert {"method", "nobs"} <= keys
    assert not keys - _tokens(tsecon.arima_fit.__doc__)
    assert not keys - _tokens(_stub_doc("arima_fit"))


def test_docs_state_the_css_conventions():
    for doc in (tsecon.arima_fit.__doc__, _stub_doc("arima_fit")):
        flat = re.sub(r"\s+", " ", doc)
        assert "CONDITIONAL" in flat
        assert "statsforecast" in flat
        assert "-650.97" in flat and "-649.34" in flat
        assert "Var(c_hat)" in flat
