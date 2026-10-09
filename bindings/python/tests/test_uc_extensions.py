"""Golden and behavioral tests for the 0.11.0 follow-ups of
``unobserved_components``: the stationary (Lyapunov) cycle initialization
``cycle_init="stationary"``, the autoregressive component
``autoregressive=p``, and the regressors in the state vector
``mle_regression=False``.

Re-pins ``fixtures/uc_extensions.json`` (generator
``fixtures/generate_uc_extensions_fixtures.py``: statsmodels' own
``autoregressive=`` and its block ``Initialization`` for the stationary
cycle, at fixed parameters at 1e-8, and the two-optimizer MLE) through the
Python surface, cross-checks
statsmodels live where it is installed (the cycle located through the
public ``state_names``, never private attributes), exercises the
refusals (every one names the offending parameter), and runs the model
card's new example blocks, checking that they print what they claim.
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
SPREAD_FACTOR = 10.0


_SERIES = {"sim": "sim_y", "sim2": "sim2_y", "sim2_missing": "sim2_y_missing"}


def _y(case):
    return np.asarray(UX["data"][_SERIES.get(case["data"], case["data"])], float)


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
    """Smoothed variances: inside the diffuse period the tolerance is the
    fixture's ``smoother_spread`` (how far statsmodels' OWN univariate and
    conventional smoothers disagree there) times ``SPREAD_FACTOR``.

    The spread measures how strongly those rows amplify rounding, and a
    different platform's rounding passes through the same amplification:
    on ``state_regression[3]`` (spread 6.3e-8, every other case about 1e-14)
    the macOS arm64 wheel lands 1.6e-7 absolute / 4.9e-7 relative from
    statsmodels, while the x86_64 wheels stay within one spread. Every case
    with a negligible spread keeps ``TOL``."""
    a, e = np.asarray(a, float), np.asarray(e, float)
    relaxed = max(SPREAD_FACTOR * float(block.get("smoother_spread", 0.0)), TOL)
    cut = min(d + DIFFUSE_SMOOTH_ROWS + 1, len(a))
    _nan_close(a[:cut], e[:cut], tol=relaxed)
    _nan_close(a[cut:], e[cut:], tol=TOL)


def _fixed(case):
    kw = _kwargs(case["spec"])
    h = len(case.get("forecast", []))
    if "exog" in kw and h:
        kw["forecast_exog"] = np.asarray(UX["data"][case["spec"]["exog"] + "_forecast"], float)[:h]
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
    for sm_name, key in (("level", "level"), ("trend", "slope"), ("seasonal", "seasonal"), ("cycle", "cycle"),
                         ("autoregressive", "autoregressive")):
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


# ----------------------------------------------------------- AR component
@pytest.mark.parametrize("case", UX["autoregressive"], ids=lambda c: c["name"])
def test_ar_component_fixed_parameters(case):
    r = _fixed(case)
    _check_fixed(r, case)
    p = case["spec"]["autoregressive"]
    assert r["k_diffuse"] == r["k_states"] - p
    assert list(r["state_names"])[-p:] == [f"ar.L{j}" for j in range(1, p + 1)]
    assert "autoregressive" in case["components"]
    st = np.asarray(r["smoothed_state"])
    np.testing.assert_allclose(r["autoregressive"], st[:, r["k_states"] - p])


@pytest.mark.parametrize("case", UX["autoregressive_mle"], ids=lambda c: c["name"])
def test_ar_component_mle(case):
    r = tsecon.unobserved_components(_y(case), **_kwargs(case["spec"]))
    _check_mle(r, case)
    if case["name"] == "infl_llevel_ar2":
        # statsmodels' own L-BFGS stops 1.89 points short on this one.
        assert r["loglik"] > case["statsmodels"]["llf"] + 1.8


def test_ar_component_matches_statsmodels_live():
    pytest.importorskip("statsmodels")
    from statsmodels.tsa.statespace.structural import UnobservedComponents

    nile = np.asarray(UX["data"]["nile"], float)
    p = np.array([9000.0, 1000.0, 5000.0, 0.3, 0.2])
    ref = UnobservedComponents(nile, "llevel", autoregressive=2, use_exact_diffuse=True).smooth(p)
    r = tsecon.unobserved_components(nile, autoregressive=2, fixed_params=list(p), forecast_steps=4)
    assert r["loglik"] == pytest.approx(ref.llf, rel=1e-10)
    assert r["aic"] == pytest.approx(ref.aic, rel=1e-10) and r["k_diffuse"] == ref.k_diffuse_states == 1
    assert list(r["param_names"]) == list(ref.model.param_names)
    np.testing.assert_allclose(r["autoregressive"], ref.autoregressive.smoothed, rtol=1e-8, atol=1e-6)
    np.testing.assert_allclose(r["filtered_autoregressive_var"], ref.autoregressive.filtered_cov, rtol=1e-8)
    fc = ref.get_forecast(4)
    np.testing.assert_allclose(r["forecast"], fc.predicted_mean, rtol=1e-10)
    np.testing.assert_allclose(r["forecast_var"], fc.var_pred_mean, rtol=1e-10)


def test_ar_refusals_name_the_parameter():
    _raises(r"autoregressive = 0.*pass None", autoregressive=0)
    _raises(r"autoregressive = 101 with 100 observations", autoregressive=101)
    _raises(r"at least k_states \+ 2 = 102 observations", autoregressive=99)
    _raises(r"autoregressive", autoregressive=-1)
    _raises(r"autoregressive", autoregressive=2**63)
    _raises(r"fixed_params\[3\] \(ar.L1\).*partial autocorrelation at lag 1 is 1.2",
            autoregressive=1, fixed_params=[8000.0, 1400.0, 6000.0, 1.2])
    _raises(r"fixed_params\[3\] \(ar.L1\).*partial autocorrelation at lag 1 is 1",
            autoregressive=1, fixed_params=[8000.0, 1400.0, 6000.0, 1.0])
    _raises(r"cannot resolve observation", autoregressive=1, fixed_params=[8000.0, 1400.0, 6000.0, 1.0 - 1e-15])
    # The same skip with no unit root: every variance is zero, so after the
    # first observation the model is deterministic. The refusal says so
    # rather than blaming the (stationary) AR coefficient.
    _raises(r"fixed_params = \[0\.0, 0\.0, 0\.0, 0\.5\]: the filter cannot resolve observation 1.*"
            r"every variance that reaches the observation is zero",
            autoregressive=1, fixed_params=[0.0, 0.0, 0.0, 0.5])
    y = np.asarray(UX["data"]["nile"], float)
    with pytest.raises(TypeError):
        tsecon.unobserved_components(y, autoregressive=1.5)


def test_a_state_dimension_too_large_to_store_is_refused_by_name():
    """`autoregressive` (like `seasonal`) at most len(y) does not bound the
    len(y) * k_states^2 cells of each of the four stored covariance
    sequences: an order near the length of a long series used to end in an
    allocator abort. It is refused by name before anything is built."""
    y = np.random.default_rng(3).normal(size=20_000)
    for kw, needle in ((dict(autoregressive=10_000), "autoregressive = 10000 costs 10000 states"),
                       (dict(seasonal=10_000), "seasonal = 10000 costs 9999 states")):
        with pytest.raises(ValueError, match=needle + r".*beyond the 67108864 accepted"):
            tsecon.unobserved_components(y, **kw)


def test_ar_refuses_where_statsmodels_returns_numbers():
    """statsmodels has no stationarity check: at a unit or explosive root it
    returns a finite number (recorded in the fixture as evidence, not as a
    golden). tsecon refuses there, and agrees with statsmodels right next to
    the boundary, where the polynomial is still stationary."""
    y = np.asarray(UX["data"]["nile"], float)
    for c in UX["autoregressive_nonstationary_statsmodels"]:
        phi = c["params"][-1]
        assert np.isfinite(c["statsmodels_loglike"])
        if phi >= 1.0:
            with pytest.raises(ValueError, match=r"fixed_params\[3\] \(ar.L1\)"):
                tsecon.unobserved_components(y, autoregressive=1, fixed_params=c["params"])
        else:
            r = tsecon.unobserved_components(y, autoregressive=1, fixed_params=c["params"])
            assert r["loglik"] == pytest.approx(c["statsmodels_loglike"], rel=1e-10)


# --------------------------------------------- regressors in the state
def _regression_close(r, rc, block):
    d = r["nobs_diffuse"]
    for key, ref in (("regression_coefficients", "smoothed"),
                     ("filtered_regression_coefficients", "filtered"),
                     ("filtered_regression_coefficients_var", "filtered_var")):
        _nan_close(r[key], rc[ref])
    _smoothed_var_close(r["regression_coefficients_var"], rc["smoothed_var"], block, d)
    _nan_close(r["regression_coefficients_cov"], rc["smoothed_cov_last"])


@pytest.mark.parametrize("case", UX["state_regression"], ids=lambda c: c["name"])
def test_state_regression_fixed_parameters(case):
    r = _fixed(case)
    _check_fixed(r, case)
    _regression_close(r, case["components"]["regression_coefficients"], case)
    assert not any(n.startswith("beta") for n in r["param_names"])
    assert list(r["state_names"])[-2:] == ["beta.x1", "beta.x2"]
    beta = np.asarray(r["regression_coefficients"])
    assert beta.shape == (r["nobs"], 2)
    np.testing.assert_allclose(beta, np.tile(beta[-1], (r["nobs"], 1)), rtol=1e-9, atol=1e-12)


def test_white_noise_plus_state_regression_is_ols():
    cf = UX["state_regression_closed_form"]
    y = np.asarray(UX["data"]["sim2_y"], float)
    X = np.asarray(UX["data"]["sim2_x"], float)
    Xf = np.asarray(UX["data"]["sim2_x_forecast"], float)
    r = tsecon.unobserved_components(y, level="ntrend", exog=X, mle_regression=False,
                                     fixed_params=[cf["sigma2"]], forecast_steps=6, forecast_exog=Xf)
    assert r["loglik"] == pytest.approx(cf["loglike"], rel=1e-10)
    np.testing.assert_allclose(np.asarray(r["regression_coefficients"])[[0, -1]], [cf["beta_ols"]] * 2, rtol=1e-10)
    np.testing.assert_allclose(r["regression_coefficients_cov"], cf["cov"], rtol=1e-10)
    np.testing.assert_allclose(r["forecast"], cf["forecast"], rtol=1e-10)
    np.testing.assert_allclose(r["forecast_var"], cf["forecast_var"], rtol=1e-10)
    # Independently, with numpy only.
    np.testing.assert_allclose(cf["beta_ols"], np.linalg.lstsq(X, y, rcond=None)[0], rtol=1e-12)


@pytest.mark.parametrize("c", UX["state_regression_identity"], ids=lambda c: f'{c["level"]}-ar{c["autoregressive"]}')
def test_state_and_mle_regression_identity(c):
    y = np.asarray(UX["data"]["sim2_y"], float)
    X = np.asarray(UX["data"]["sim2_x"], float)
    kw = dict(level=c["level"], exog=X)
    if c["autoregressive"]:
        kw["autoregressive"] = c["autoregressive"]
    st = tsecon.unobserved_components(y, mle_regression=False, fixed_params=c["params"], **kw)
    beta = list(np.asarray(st["regression_coefficients"])[-1])
    ml = tsecon.unobserved_components(y, fixed_params=c["params"] + beta, **kw)
    half = 0.5 * np.linalg.slogdet(np.asarray(st["regression_coefficients_cov"]))[1]
    assert st["loglik"] == pytest.approx(c["loglike_state"], rel=TOL)
    assert ml["loglik"] == pytest.approx(c["loglike_mle_at_gls"], rel=TOL)
    assert abs(st["loglik"] - ml["loglik"] - half) < 1e-10


def test_state_regression_loglik_depends_on_regressor_units():
    u = UX["state_regression_units"]
    y = np.asarray(UX["data"]["sim2_y"], float)
    X = np.asarray(UX["data"]["sim2_x"], float)
    Xc = X.copy()
    Xc[:, 0] *= u["scale"]
    a = tsecon.unobserved_components(y, exog=X, mle_regression=False, fixed_params=u["params"])
    b = tsecon.unobserved_components(y, exog=Xc, mle_regression=False, fixed_params=u["params"])
    assert a["loglik"] == pytest.approx(u["loglike_state"], rel=TOL)
    assert b["loglik"] == pytest.approx(u["loglike_state_scaled"], rel=TOL)
    assert a["loglik"] - b["loglik"] == pytest.approx(np.log(u["scale"]), abs=1e-10)
    m = tsecon.unobserved_components(y, exog=X, fixed_params=[0.8, 0.3, 0.5, -0.3])
    mc = tsecon.unobserved_components(y, exog=Xc, fixed_params=[0.8, 0.3, 0.5 / u["scale"], -0.3])
    assert m["loglik"] == pytest.approx(mc["loglik"], abs=1e-10)


@pytest.mark.parametrize("case", UX["state_regression_mle"], ids=lambda c: c["name"])
def test_state_regression_mle(case):
    r = tsecon.unobserved_components(_y(case), **_kwargs(case["spec"]))
    _check_mle(r, case)
    np.testing.assert_allclose(np.asarray(r["regression_coefficients"])[-1],
                               case["best"]["regression_coefficients"], rtol=2e-3)


def test_state_regression_matches_statsmodels_live():
    pytest.importorskip("statsmodels")
    from statsmodels.tsa.statespace.structural import UnobservedComponents

    y = np.asarray(UX["data"]["sim2_y_missing"], float)
    X = np.asarray(UX["data"]["sim2_x"], float)
    Xf = np.asarray(UX["data"]["sim2_x_forecast"], float)
    p = np.array([0.6, 0.2])
    ref = UnobservedComponents(y, "llevel", exog=X, mle_regression=False, use_exact_diffuse=True).smooth(p)
    r = tsecon.unobserved_components(y, exog=X, mle_regression=False, fixed_params=list(p),
                                     forecast_steps=3, forecast_exog=Xf[:3])
    assert r["loglik"] == pytest.approx(ref.llf, rel=1e-10)
    assert r["aic"] == pytest.approx(ref.aic, rel=1e-10)
    np.testing.assert_allclose(np.asarray(r["regression_coefficients"]), ref.regression_coefficients.smoothed.T,
                               rtol=1e-8, atol=1e-10)
    fc = ref.get_forecast(3, exog=Xf[:3])
    np.testing.assert_allclose(r["forecast"], fc.predicted_mean, rtol=1e-10)
    np.testing.assert_allclose(r["forecast_var"], fc.var_pred_mean, rtol=1e-10)


def test_state_regression_accepts_pandas():
    pd = pytest.importorskip("pandas")
    case = UX["state_regression"][0]
    idx = pd.RangeIndex(len(_y(case)))
    y = pd.Series(_y(case), index=idx)
    X = pd.DataFrame(np.asarray(UX["data"]["sim2_x"], float), index=idx, columns=["a", "b"])
    Xf = pd.DataFrame(np.asarray(UX["data"]["sim2_x_forecast"], float)[:4], columns=["a", "b"])
    r = tsecon.unobserved_components(y, exog=X, mle_regression=False, fixed_params=case["params"],
                                     forecast_steps=4, forecast_exog=Xf)
    assert r["loglik"] == pytest.approx(case["loglike"], rel=TOL)
    _nan_close(r["forecast"], case["forecast"])


def test_state_regression_refusals_name_the_parameter():
    y = np.asarray(UX["data"]["sim2_y"], float)
    X = np.asarray(UX["data"]["sim2_x"], float)
    for flag in (False, True):
        with pytest.raises(ValueError, match="mle_regression was given but exog is None"):
            tsecon.unobserved_components(y, mle_regression=flag)
    with pytest.raises(ValueError, match="forecast_steps = 3 with 2 regressors requires forecast_exog"):
        tsecon.unobserved_components(y, exog=X, mle_regression=False, forecast_steps=3)
    Xn = X.copy()
    Xn[7, 1] = np.nan
    with pytest.raises(ValueError, match="exog column 1 contains a NaN"):
        tsecon.unobserved_components(y, exog=Xn, mle_regression=False)
    with pytest.raises(ValueError, match=r"fixed_params has length 4 but the specification has 2"):
        tsecon.unobserved_components(y, exog=X, mle_regression=False, fixed_params=[0.8, 0.3, 0.5, -0.3])


def test_state_regression_refuses_regressor_scales_the_filter_cannot_resolve():
    """The coefficient states are diffuse and loaded by x_t: with one column
    eight orders of magnitude from the other (and from the level's unit
    loading) the exact-diffuse filter cannot resolve them in floating
    point, and the diffuse period would run to the end of the sample with
    roundoff coefficients. Refused by name, fixed and estimated alike;
    inside the resolvable range the units identity holds."""
    y = np.asarray(UX["data"]["sim2_y"], float)
    X = np.asarray(UX["data"]["sim2_x"], float)
    base = tsecon.unobserved_components(y, exog=X, mle_regression=False, fixed_params=[0.8, 0.3])
    for c in (1e-4, 1e4):
        Xc = X.copy()
        Xc[:, 0] *= c
        r = tsecon.unobserved_components(y, exog=Xc, mle_regression=False, fixed_params=[0.8, 0.3])
        assert r["nobs_diffuse"] == base["nobs_diffuse"] == 3
        assert r["loglik"] == pytest.approx(base["loglik"] - np.log(c), abs=1e-7)
    for c in (1e-8, 1e8):
        Xc = X.copy()
        Xc[:, 0] *= c
        for kw in (dict(fixed_params=[0.8, 0.3]), dict(n_starts=1)):
            with pytest.raises(ValueError, match=r"exog: under mle_regression = false.*would end at observation "
                                                 r"120 instead of 3.*Rescale the columns"):
                tsecon.unobserved_components(y, exog=Xc, mle_regression=False, **kw)


# ------------------------------------------------------ stationary cycle
@pytest.mark.parametrize("case", UX["stationary_cycle"], ids=lambda c: c["name"])
def test_stationary_cycle_fixed_parameters(case):
    r = _fixed(case)
    _check_fixed(r, case)
    # The cycle pair (and any AR block) is stationary: that many states
    # fewer are diffuse.
    assert r["k_diffuse"] == r["k_states"] - 2 - case["spec"].get("autoregressive", 0)


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
    assert sig.parameters["autoregressive"].default is None
    assert sig.parameters["mle_regression"].default is None
    names = list(sig.parameters)
    assert names.index("autoregressive") == names.index("n_starts") + 1
    assert all(p.default is not Ellipsis for p in sig.parameters.values())
    tokens = set(re.findall(r"`([A-Za-z_][A-Za-z_0-9]*)`", tsecon.unobserved_components.__doc__ or ""))
    r = _fixed(UX["stationary_cycle"][0])
    assert not set(r.keys()) - tokens
    # Every component key is present whether or not the component is.
    for k in ("autoregressive", "autoregressive_var", "filtered_autoregressive", "filtered_autoregressive_var"):
        assert k in r and r[k] is None
    for k in ("regression_coefficients", "regression_coefficients_var", "filtered_regression_coefficients",
              "filtered_regression_coefficients_var", "regression_coefficients_cov"):
        assert k in r and r[k] is None
    ra = _fixed(UX["autoregressive"][0])
    assert set(ra.keys()) == set(r.keys())
    assert ra["autoregressive"] is not None and ra["cycle"] is None
    rs = _fixed(UX["state_regression"][0])
    assert set(rs.keys()) == set(r.keys())
    assert rs["regression_coefficients_cov"] is not None
    # Under MLE regression the coefficients are parameters, not states.
    rm = _fixed(UX["autoregressive"][-1])
    assert "exog" in UX["autoregressive"][-1]["spec"] and rm["regression_coefficients"] is None
    stub = (Path(tsecon.__file__).parent / "__init__.pyi").read_text(encoding="utf-8")
    i = stub.index("def unobserved_components(")
    body = stub[i:stub.index("\ndef ", i + 10)]
    assert "cycle_init: str | None = None" in body
    stub_tokens = set(re.findall(r"`([A-Za-z_][A-Za-z_0-9]*)`", body))
    assert not set(r.keys()) - stub_tokens


# ------------------------------------------------------------ model card
def test_model_card_examples_print_what_they_claim():
    """Every 0.11.0 example block in the card runs and prints its ``# {...}``
    lines verbatim. The round-14 test runs only the FIRST block containing
    ``unobserved_components(nile``; these sit after it and are run here."""
    pytest.importorskip("statsmodels.api")
    import re
    import subprocess
    import sys

    card = (Path(__file__).parents[3] / "docs" / "reference" / "model-cards"
            / "unobserved-components.md").read_text(encoding="utf-8")
    blocks = re.findall(r"```python\n(.*?)```", card, re.S)
    first = next(i for i, b in enumerate(blocks) if "unobserved_components(nile" in b)
    new = [b for b in blocks[first + 1:]
           if any(k in b for k in ("autoregressive=", "cycle_init=", "mle_regression="))]
    assert len(new) == 3, [b[:60] for b in new]
    for block in new:
        claimed = [line[2:] for line in block.splitlines() if line.startswith("# {")]
        assert claimed, block
        out = subprocess.run([sys.executable, "-c", block], capture_output=True, text=True)
        assert out.returncode == 0, out.stderr[-600:]
        printed = out.stdout.splitlines()
        for line in claimed:
            assert line in printed, (line, printed)
