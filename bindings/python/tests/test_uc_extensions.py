"""Golden and behavioral tests for the 0.11.0 follow-ups of
``unobserved_components``: the stationary (Lyapunov) cycle initialization
``cycle_init="stationary"``.

Re-pins ``fixtures/uc_extensions.json`` (generator
``fixtures/generate_uc_extensions_fixtures.py``: statsmodels' own block
``Initialization`` for the stationary cycle, at fixed parameters at 1e-8,
and the two-optimizer MLE) through the Python surface, cross-checks
statsmodels live where it is installed (the cycle located through the
public ``state_names``, never private attributes), and exercises the
refusals: every one names the offending parameter.
"""
import json
from pathlib import Path

import numpy as np
import pytest

import tsecon

FIX = Path(__file__).parents[3] / "fixtures"
UX = json.loads((FIX / "uc_extensions.json").read_text())

TOL = 1e-8
DIFFUSE_SMOOTH_ROWS = 2


def _y(case):
    return np.asarray(UX["data"]["sim_y" if case["data"] == "sim" else case["data"]], float)


def _kwargs(spec):
    kw = {}
    for k, v in spec.items():
        if k == "exog":
            kw["exog"] = np.asarray(UX["data"][v], float)
        elif k == "cycle_period_bounds":
            kw[k] = list(v)
        elif k == "freq_seasonal":
            kw["freq_seasonal"] = [d["period"] for d in v]
            kw["freq_seasonal_harmonics"] = [d["harmonics"] for d in v]
        else:
            kw[k] = v
    return kw


def _nan_close(a, e, tol=TOL):
    a, e = np.asarray(a, float), np.asarray(e, float)
    assert a.shape == e.shape
    m = np.isnan(e)
    assert np.all(np.isnan(a[m]))
    np.testing.assert_allclose(a[~m], e[~m], rtol=tol, atol=tol)


def _smoothed_var_close(a, e, block, d):
    a, e = np.asarray(a, float), np.asarray(e, float)
    relaxed = max(float(block.get("smoother_spread", 0.0)), TOL)
    cut = min(d + DIFFUSE_SMOOTH_ROWS + 1, len(a))
    _nan_close(a[:cut], e[:cut], tol=relaxed)
    _nan_close(a[cut:], e[cut:], tol=TOL)


def _fixed(case):
    kw = _kwargs(case["spec"])
    h = len(case.get("forecast", []))
    if "exog" in kw and h:
        kw["forecast_exog"] = np.asarray(UX["data"]["sim_x_forecast"], float)[:h]
    return tsecon.unobserved_components(_y(case), fixed_params=case["params"], forecast_steps=h, **kw)


def _check_fixed(r, block):
    assert r["k_states"] == block["k_states"]
    assert r["nobs_diffuse"] == block["nobs_diffuse"]
    assert r["k_diffuse"] == block["k_diffuse"]
    assert r["k_params"] + r["k_diffuse"] == block["df_model"]
    assert list(r["param_names"]) == block["param_names"]
    assert r["loglik"] == pytest.approx(block["loglike"], rel=TOL, abs=TOL)
    assert r["aic"] == pytest.approx(block["aic"], rel=TOL)
    assert r["bic"] == pytest.approx(block["bic"], rel=TOL)
    if "filtered_state" in block:
        for key in ("filtered_state", "filtered_state_var", "smoothed_state"):
            _nan_close(r[key], block[key])
        _smoothed_var_close(r["smoothed_state_var"], block["smoothed_state_var"], block, r["nobs_diffuse"])
    _nan_close(r["fitted"], block["fitted"])
    _nan_close(r["resid"], block["resid"])
    d = r["nobs_diffuse"]
    sr = np.asarray(block["std_resid"], float)
    got = np.asarray(r["std_resid"], float)
    missing = np.isnan(np.asarray(block["resid"], float))
    assert np.all(np.isnan(got[:d])) and np.all(np.isnan(got[missing]))
    obs = ~missing
    obs[:d] = False
    _nan_close(got[obs], sr[obs])
    if "forecast" in block:
        _nan_close(r["forecast"], block["forecast"])
        _nan_close(r["forecast_var"], block["forecast_var"])
    comp = block.get("components", {})
    for sm_name, key in (("level", "level"), ("trend", "slope"), ("seasonal", "seasonal"), ("cycle", "cycle")):
        if sm_name in comp:
            _nan_close(r[key], comp[sm_name]["smoothed"])
            _smoothed_var_close(r[key + "_var"], comp[sm_name]["smoothed_var"], block, d)
            _nan_close(r["filtered_" + key], comp[sm_name]["filtered"])
            _nan_close(r["filtered_" + key + "_var"], comp[sm_name]["filtered_var"])


def _check_mle(r, case):
    best = case["best"]
    assert r["estimated"]
    assert list(r["param_names"]) == case["param_names"]
    assert max(case["statsmodels"]["llf"], case["scipy"]["llf"]) == best["llf"]
    assert r["loglik"] >= best["llf"] - 1e-5
    scale = np.max(np.abs(best["params"]))
    np.testing.assert_allclose(r["params"], best["params"], rtol=2e-3, atol=2e-3 * 1e-3 * scale)
    assert r["at_boundary"] == best["at_boundary"]
    for i, e in enumerate(best["se_conditional"]):
        if np.isnan(e):
            assert np.isnan(r["se"][i]), (i, r["se"][i])
        else:
            assert r["se"][i] == pytest.approx(e, rel=2e-2)
    assert r["nobs_diffuse"] == best["nobs_diffuse"] and r["k_diffuse"] == best["k_diffuse"]


# ------------------------------------------------------ stationary cycle
@pytest.mark.parametrize("case", UX["stationary_cycle"], ids=lambda c: c["name"])
def test_stationary_cycle_fixed_parameters(case):
    r = _fixed(case)
    _check_fixed(r, case)
    # The cycle pair is the stationary block: two states fewer are diffuse.
    assert r["k_diffuse"] == r["k_states"] - 2


def test_stationary_cycle_likelihood_is_not_the_diffuse_one():
    case = UX["stationary_cycle"][0]
    assert case["name"] == "lgdp_llevel"
    kw = _kwargs(case["spec"])
    kw.pop("cycle_init")
    stat = _fixed(case)
    diff = tsecon.unobserved_components(_y(case), fixed_params=case["params"], **kw)
    assert diff["loglik"] == pytest.approx(case["diffuse_cycle"]["loglike"], rel=TOL)
    assert diff["nobs_diffuse"] == case["diffuse_cycle"]["nobs_diffuse"] == 3
    assert stat["nobs_diffuse"] == 1
    # -454.98 against -444.16 at the SAME parameters: a different likelihood.
    assert stat["loglik"] < diff["loglik"] - 10
    # "Stationary" in any case is accepted.
    kw2 = dict(kw, cycle_init=" Stationary ")
    again = tsecon.unobserved_components(_y(case), fixed_params=case["params"], **kw2)
    assert again["loglik"] == stat["loglik"]


@pytest.mark.parametrize("case", UX["stationary_cycle_mle"], ids=lambda c: c["name"])
def test_stationary_cycle_mle(case):
    r = tsecon.unobserved_components(_y(case), **_kwargs(case["spec"]))
    _check_mle(r, case)
    # sigma2.irregular piles up at exactly zero on GDP; the cycle variance
    # does not, so the frequency and damping keep finite standard errors.
    names = list(r["param_names"])
    assert r["at_boundary"][names.index("sigma2.irregular")]
    assert np.isfinite(r["se"][names.index("frequency.cycle")])


def test_stationary_cycle_lambda_sweep_is_flat():
    y = np.asarray(UX["data"]["lgdp"], float)
    lls = []
    for s in UX["stationary_cycle_lambda_sweep"]:
        r = tsecon.unobserved_components(y, level="strend", cycle=True, damped_cycle=True, stochastic_cycle=True,
                                         cycle_period_bounds=[2.0, 1e9], cycle_init="stationary",
                                         fixed_params=s["params"])
        assert r["loglik"] == pytest.approx(s["stationary"], rel=TOL)
        lls.append(r["loglik"])
    assert max(lls) - min(lls) < 0.2


def test_stationary_cycle_matches_statsmodels_live():
    pytest.importorskip("statsmodels")
    from statsmodels.tsa.statespace.initialization import Initialization
    from statsmodels.tsa.statespace.structural import UnobservedComponents

    class StationaryCycleUC(UnobservedComponents):
        def initialize_default(self, approximate_diffuse_variance=None):
            super().initialize_default(approximate_diffuse_variance)
            k = self.k_states
            off = list(self.state_names).index("cycle")       # public, not _k_*_states
            init = Initialization(k)
            if off:
                init.set((0, off), "diffuse")
            init.set((off, off + 2), "stationary")
            if k > off + 2:
                init.set((off + 2, k), "diffuse")
            self.ssm.initialization = init

    nile = np.asarray(UX["data"]["nile"], float)
    p = np.array([12000.0, 800.0, 2500.0, 2 * np.pi / 10, 0.6])
    spec = dict(level="llevel", cycle=True, damped_cycle=True, stochastic_cycle=True)
    ref = StationaryCycleUC(nile, use_exact_diffuse=True, **spec).smooth(p)
    r = tsecon.unobserved_components(nile, cycle_init="stationary", fixed_params=list(p), forecast_steps=3, **spec)
    assert r["loglik"] == pytest.approx(ref.llf, rel=1e-10)
    assert r["nobs_diffuse"] == ref.nobs_diffuse and r["k_diffuse"] == ref.k_diffuse_states
    assert r["aic"] == pytest.approx(ref.aic, rel=1e-10)
    np.testing.assert_allclose(r["cycle"], ref.cycle.smoothed, rtol=1e-8, atol=1e-8)
    fc = ref.get_forecast(3)
    np.testing.assert_allclose(r["forecast"], fc.predicted_mean, rtol=1e-10)
    np.testing.assert_allclose(r["forecast_var"], fc.var_pred_mean, rtol=1e-10)


def test_pandas_series_input_is_coerced():
    pd = pytest.importorskip("pandas")
    case = UX["stationary_cycle"][1]
    y = pd.Series(_y(case), index=pd.RangeIndex(1871, 1871 + len(_y(case))))
    r = tsecon.unobserved_components(y, fixed_params=case["params"], **_kwargs(case["spec"]))
    assert r["loglik"] == pytest.approx(case["loglike"], rel=TOL)


# --------------------------------------------------------------- refusals
def _raises(match, **kw):
    y = np.asarray(UX["data"]["nile"], float)
    with pytest.raises(ValueError, match=match):
        tsecon.unobserved_components(y, **kw)


def test_cycle_init_refusals_name_the_parameter():
    damped = dict(cycle=True, damped_cycle=True, stochastic_cycle=True)
    _raises(r"cycle_init was given but cycle=False", cycle_init="stationary")
    _raises(r"cycle_init was given but cycle=False", cycle_init="diffuse")
    _raises(r'cycle_init = "bogus".*"diffuse".*"stationary"', cycle_init="bogus", **damped)
    _raises(r"requires damped_cycle = true", cycle=True, stochastic_cycle=True, cycle_init="stationary")
    _raises(r"requires damped_cycle = true", cycle=True, damped_cycle=False, stochastic_cycle=True,
            cycle_init="stationary")
    _raises(r"requires stochastic_cycle = true", cycle=True, damped_cycle=True, stochastic_cycle=False,
            cycle_init="stationary")
    _raises(r"requires stochastic_cycle = true", cycle=True, damped_cycle=True, cycle_init="stationary")
    _raises(r"fixed_params\[4\] \(damping.cycle\) = 1",
            cycle_init="stationary", fixed_params=[1.0, 1.0, 1.0, 0.5, 1.0], **damped)
    # The diffuse cycle still accepts an undamped (damping 1) fixed value.
    y = np.asarray(UX["data"]["nile"], float)
    r = tsecon.unobserved_components(y, cycle_init="diffuse", fixed_params=[1.0, 1.0, 1.0, 0.5, 1.0], **damped)
    assert np.isfinite(r["loglik"])
    with pytest.raises(TypeError):
        tsecon.unobserved_components(y, cycle_init=1, **damped)


def test_defaults_render_and_keys_are_documented():
    import inspect
    import re

    sig = inspect.signature(tsecon.unobserved_components)
    assert sig.parameters["cycle_init"].default is None
    assert all(p.default is not Ellipsis for p in sig.parameters.values())
    tokens = set(re.findall(r"`([A-Za-z_][A-Za-z_0-9]*)`", tsecon.unobserved_components.__doc__ or ""))
    r = _fixed(UX["stationary_cycle"][0])
    assert not set(r.keys()) - tokens
    stub = (Path(tsecon.__file__).parent / "__init__.pyi").read_text(encoding="utf-8")
    i = stub.index("def unobserved_components(")
    body = stub[i:stub.index("\ndef ", i + 10)]
    assert "cycle_init: str | None = None" in body
    stub_tokens = set(re.findall(r"`([A-Za-z_][A-Za-z_0-9]*)`", body))
    assert not set(r.keys()) - stub_tokens
