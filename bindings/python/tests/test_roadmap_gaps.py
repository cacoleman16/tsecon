"""Golden tests for the roadmap-extension bindings: recession probability
(E8), survey expectations (E6), and long memory (E7).

recession and survey are checked against statsmodels goldens; the
long-memory GPH / local-Whittle estimators against documented-formula
goldens (the crates' Rust tests carry the same targets).
"""
import json
from pathlib import Path

import numpy as np
import tsecon

FIXTURES = Path(__file__).parents[3] / "fixtures"
REC = json.loads((FIXTURES / "tsecon-recession.json").read_text())
SUR = json.loads((FIXTURES / "tsecon-survey.json").read_text())
LM = json.loads((FIXTURES / "longmemory.json").read_text())


# ----------------------------------------------------- recession probit/logit
def _rec_design():
    y = np.array(REC["y"])
    x = np.column_stack([REC["const"], REC["spread"], REC["lead"]])
    return y, x


def test_recession_probit_matches_statsmodels():
    y, x = _rec_design()
    r = tsecon.recession_probit(y, x, link="probit")
    g = REC["probit"]
    np.testing.assert_allclose(r["params"], g["params"], atol=1e-5)
    np.testing.assert_allclose(r["bse"], g["bse"], atol=1e-5)
    assert abs(r["loglik"] - g["llf"]) < 1e-4
    assert abs(r["pseudo_r2"] - g["prsquared"]) < 1e-5
    np.testing.assert_allclose(r["probabilities"], g["fitted"], atol=1e-5)


def test_recession_logit_matches_statsmodels():
    y, x = _rec_design()
    r = tsecon.recession_probit(y, x, link="logit")
    g = REC["logit"]
    np.testing.assert_allclose(r["params"], g["params"], atol=1e-5)
    np.testing.assert_allclose(r["bse"], g["bse"], atol=1e-5)
    assert abs(r["loglik"] - g["llf"]) < 1e-4


# ----------------------------------------------------- Coibion-Gorodnichenko
def test_cg_regression_matches_statsmodels_hac():
    cg = SUR["cg"]
    r = tsecon.cg_regression(
        np.array(cg["errors"]), np.array(cg["revisions"]),
        maxlags=cg["maxlags"], use_correction=cg["use_correction"],
    )
    assert abs(r["intercept"] - cg["intercept"]) < 1e-7
    assert abs(r["slope"] - cg["slope"]) < 1e-7
    assert abs(r["se_slope"] - cg["se_slope"]) < 1e-7
    # implied_rigidity = slope / (1 + slope)
    assert abs(r["implied_rigidity"] - cg["slope"] / (1 + cg["slope"])) < 1e-9


def test_forecast_efficiency_matches_statsmodels_hac():
    ef = SUR["efficiency"]
    regs = np.array(ef["regressors"]).T  # fixture stores (k, T); binding wants (T, k)
    r = tsecon.forecast_efficiency(
        np.array(ef["errors"]), regs,
        maxlags=ef["maxlags"], use_correction=ef["use_correction"],
    )
    np.testing.assert_allclose(r["params"], ef["params"], atol=1e-7)
    np.testing.assert_allclose(r["bse"], ef["bse"], atol=1e-7)
    assert r["wald"] >= 0 and 0.0 <= r["wald_pvalue"] <= 1.0


# ---------------------------------------------------------------- long memory
def test_frac_diff_matches_golden():
    case = LM["fracdiff"]["cases"][0]
    out = tsecon.frac_diff(np.array(case["x"]), case["d"])
    np.testing.assert_allclose(out, case["frac_diff"], atol=1e-10)
    # (1 - L)^1 is the ordinary first difference (leading term = x[0]).
    x = np.arange(1.0, 11.0)
    fd1 = np.asarray(tsecon.frac_diff(x, 1.0))
    np.testing.assert_allclose(fd1[1:], np.diff(x), atol=1e-10)


def test_long_memory_d_gph_and_whittle_match_golden():
    sp = LM["semiparametric"]
    x = np.array(sp["x"])
    m = sp["m"]
    gph = tsecon.long_memory_d(x, m=m, method="gph")
    assert abs(gph["d"] - sp["gph"]["d"]) < 1e-6
    assert abs(gph["se"] - sp["gph"]["se"]) < 1e-6
    lw = tsecon.long_memory_d(x, m=m, method="local_whittle")
    assert abs(lw["d"] - sp["whittle"]["d"]) < 1e-5
    # Both recover a genuinely long-memory series (d well above 0).
    assert gph["d"] > 0.1 and lw["d"] > 0.1
