"""Golden and behavioral tests for the structural time-series bindings
(``unobserved_components``, ``tvp_regression``).

Re-pins ``fixtures/uc.json`` (see ``fixtures/generate_uc_fixtures.py`` for
the honest grading: fixed-parameter quantities are an independent
statsmodels golden at 1e-8, the MLE a two-optimizer target, the TVP
regression a statsmodels ``MLEModel`` transcription plus the ``RecursiveLS``
zero-variance limit) through the Python surface, and additionally calls
statsmodels directly for a live cross-check; exercises the sentinel
refusals (inert options raise when passed), teaching errors, determinism,
missing data, and the docstring / stub contract.
"""
import json
import re
from pathlib import Path

import numpy as np
import pytest

import tsecon

FIX = Path(__file__).parents[3] / "fixtures"
UC = json.loads((FIX / "uc.json").read_text())

TOL = 1e-8


def _spec_kwargs(spec):
    kw = {}
    for k, v in spec.items():
        if k == "exog":
            kw["exog"] = np.asarray(UC["sim"]["x"], float)
        elif k == "cycle_period_bounds":
            kw["cycle_period_bounds"] = list(v)
        elif k == "freq_seasonal":
            kw["freq_seasonal"] = [d["period"] for d in v]
            if any("harmonics" in d for d in v):
                kw["freq_seasonal_harmonics"] = [d.get("harmonics", int(np.floor(d["period"] / 2))) for d in v]
        else:
            kw[k] = v
    return kw


def _nan_close(a, e, tol=TOL):
    a, e = np.asarray(a, float), np.asarray(e, float)
    assert a.shape == e.shape
    m = np.isnan(e)
    assert np.all(np.isnan(a[m]))
    np.testing.assert_allclose(a[~m], e[~m], rtol=tol, atol=tol)


# Rows past the end of the diffuse period on which the exact-diffuse
# smoother's conditioning is still visible.
DIFFUSE_SMOOTH_ROWS = 2


def _smoothed_var_close(a, e, block, d, what=""):
    """Smoothed variances: 1e-8 everywhere except inside the diffuse period,
    where the tolerance is the fixture's ``smoother_spread`` — the distance
    between statsmodels' OWN univariate and conventional smoothers on this
    model at these parameters (zero on all but two of the 32 blocks; 4.5e-3
    on the eight-diffuse-state combination, whose *filters* still agree to
    2.9e-11). tsecon must be at least as close to the reference as the
    reference is to itself."""
    a, e = np.asarray(a, float), np.asarray(e, float)
    assert a.shape == e.shape, what
    relaxed = max(float(block.get("smoother_spread", 0.0)), TOL)
    cut = min(d + DIFFUSE_SMOOTH_ROWS + 1, len(a))
    _nan_close(a[:cut], e[:cut], tol=relaxed)
    _nan_close(a[cut:], e[cut:], tol=TOL)


def _check_fixed(r, block):
    assert r["k_states"] == block["k_states"]
    assert r["nobs_diffuse"] == block["nobs_diffuse"]
    assert list(r["param_names"]) == block["param_names"]
    assert r["loglik"] == pytest.approx(block["loglike"], rel=TOL, abs=TOL)
    assert r["aic"] == pytest.approx(block["aic"], rel=TOL)
    assert r["bic"] == pytest.approx(block["bic"], rel=TOL)
    for key in ("filtered_state", "filtered_state_var", "smoothed_state"):
        _nan_close(r[key], block[key])
    _smoothed_var_close(r["smoothed_state_var"], block["smoothed_state_var"], block,
                        r["nobs_diffuse"], "smoothed_state_var")
    _nan_close(r["fitted"], block["fitted"])
    _nan_close(r["resid"], block["resid"])
    # statsmodels writes 0.0 for the standardized residual inside the
    # diffuse period and at a missing period; tsecon writes NaN at both
    # (no finite prediction variance / no prediction error exists). The
    # missing periods are those whose reference prediction error is NaN.
    sr = np.asarray(block["std_resid"], float)
    got_sr = np.asarray(r["std_resid"], float)
    d = r["nobs_diffuse"]
    missing = np.isnan(np.asarray(block["resid"], float))
    assert np.all(np.isnan(got_sr[:d])), "std_resid must be NaN inside the diffuse period"
    assert np.all(np.isnan(got_sr[missing])), "std_resid must be NaN at missing periods"
    obs = ~missing
    obs[:d] = False
    _nan_close(got_sr[obs], sr[obs])
    if "forecast" in block:
        _nan_close(r["forecast"], block["forecast"])
        _nan_close(r["forecast_var"], block["forecast_var"])
    comp = block.get("components", {})
    for sm_name, key in (("level", "level"), ("trend", "slope"), ("seasonal", "seasonal"), ("cycle", "cycle")):
        if sm_name in comp:
            _nan_close(r[key], comp[sm_name]["smoothed"])
            # A component variance is a sum of state variances: same
            # diffuse-period treatment.
            _smoothed_var_close(r[key + "_var"], comp[sm_name]["smoothed_var"], block,
                                r["nobs_diffuse"], key + "_var")
            _nan_close(r["filtered_" + key], comp[sm_name]["filtered"])
            _nan_close(r["filtered_" + key + "_var"], comp[sm_name]["filtered_var"])
    if "freq_seasonal" in comp:
        assert len(r["freq_seasonal"]) == len(comp["freq_seasonal"])
        for got, exp in zip(r["freq_seasonal"], comp["freq_seasonal"]):
            _nan_close(got, exp["smoothed"])


# ------------------------------------------------------------------- Nile
def test_nile_fixed_at_durbin_koopman_values():
    nile = np.asarray(UC["nile"]["y"], float)
    block = UC["nile"]["fixed"]
    r = tsecon.unobserved_components(nile, fixed_params=block["params"], forecast_steps=10)
    assert r["estimated"] is False and r["n_iter"] == 0
    assert np.all(np.isnan(r["se"]))
    _check_fixed(r, block)
    assert r["trend_specification"] == "local level"
    assert r["slope"] is None and r["seasonal"] is None and r["cycle"] is None
    assert r["freq_seasonal"] == []


def test_nile_mle_reproduces_durbin_koopman_and_both_optimizers():
    nile = np.asarray(UC["nile"]["y"], float)
    r = tsecon.unobserved_components(nile)
    mle = UC["nile"]["mle"]
    best = mle["best"]
    assert r["estimated"] and r["converged"]
    assert r["loglik"] >= best["llf"] - 1e-5
    assert max(mle["statsmodels"]["llf"], mle["scipy"]["llf"]) == best["llf"]
    np.testing.assert_allclose(r["params"], best["params"], rtol=2e-3)
    # se_conditional, not se_approx: tsecon inverts the observed information
    # over the non-boundary parameters only. Nothing is at a boundary here,
    # so the two reference numbers coincide.
    np.testing.assert_allclose(r["se"], best["se_conditional"], rtol=2e-2)
    np.testing.assert_allclose(best["se_conditional"], best["se_approx"], rtol=1e-10)
    dk = UC["nile"]["dk_params"]                    # 15099, 1469.1 as printed
    assert abs(r["params"][0] - dk[0]) < 1.0 and abs(r["params"][1] - dk[1]) < 0.2
    assert r["at_boundary"] == [False, False]
    assert r["nobs_diffuse"] == 1 and r["k_diffuse"] == 1


def test_nile_matches_statsmodels_live():
    sm = pytest.importorskip("statsmodels.api")
    from statsmodels.tsa.statespace.structural import UnobservedComponents
    nile = np.asarray(UC["nile"]["y"], float)
    p = [12000.0, 2000.0]
    ref = UnobservedComponents(nile, "llevel", use_exact_diffuse=True).smooth(np.array(p))
    r = tsecon.unobserved_components(nile, fixed_params=p, forecast_steps=3)
    assert r["loglik"] == pytest.approx(ref.llf, rel=1e-10)
    np.testing.assert_allclose(r["level"], ref.level.smoothed, rtol=1e-10)
    np.testing.assert_allclose(r["level_var"], ref.level.smoothed_cov, rtol=1e-10)
    fc = ref.get_forecast(3)
    np.testing.assert_allclose(r["forecast"], fc.predicted_mean, rtol=1e-10)
    np.testing.assert_allclose(r["forecast_var"], fc.var_pred_mean, rtol=1e-10)


# -------------------------------------------------------- fixed parameters
@pytest.mark.parametrize("case", UC["cases"], ids=lambda c: c["name"])
def test_fixed_parameter_component_combinations(case):
    y = np.asarray(UC["sim"]["y"], float)
    kw = _spec_kwargs(case["spec"])
    if "exog" in kw:
        kw["forecast_exog"] = np.asarray(UC["sim"]["x_forecast"], float)
    r = tsecon.unobserved_components(y, fixed_params=case["params"], forecast_steps=8, **kw)
    _check_fixed(r, case)


@pytest.mark.parametrize("case", UC["missing"], ids=lambda c: c["name"])
def test_missing_observations(case):
    y = np.asarray(UC["sim"]["y_missing"], float)
    # An interior run, two isolated holes, and a missing tail (so the
    # forecast origin itself is unobserved): 12 of 120.
    assert np.isnan(y).sum() == 12
    assert np.isnan(y[10:15]).all() and np.isnan(y[115:120]).all()
    kw = _spec_kwargs(case["spec"])
    if "exog" in kw:
        kw["forecast_exog"] = np.asarray(UC["sim"]["x_forecast"], float)
    r = tsecon.unobserved_components(y, fixed_params=case["params"], forecast_steps=8, **kw)
    assert r["nobs_observed"] == 120 - 12 and r["nobs"] == 120
    _check_fixed(r, case)
    # Missing periods: prediction exists, residuals do not.
    assert np.isfinite(r["fitted"][40]) and np.isnan(r["resid"][40]) and np.isnan(r["std_resid"][40])


# ---------------------------------------------------------------------- MLE
@pytest.mark.parametrize("case", UC["mle_cases"], ids=lambda c: c["name"])
def test_mle_reaches_the_better_of_two_optimizers(case):
    y = np.asarray(UC["sim"]["y"], float)
    r = tsecon.unobserved_components(y, **_spec_kwargs(case["spec"]))
    best = case["best"]
    assert r["estimated"]
    assert list(r["param_names"]) == case["param_names"]
    assert r["loglik"] >= best["llf"] - 1e-5
    if r["loglik"] - best["llf"] < 1e-3:
        scale = np.max(np.abs(best["params"]))
        np.testing.assert_allclose(r["params"], best["params"], rtol=2e-3, atol=2e-6 * scale)
        assert r["at_boundary"] == best["at_boundary"]
        for i, e in enumerate(best["se_conditional"]):
            if not np.isnan(e) and not r["at_boundary"][i]:
                assert r["se"][i] == pytest.approx(e, rel=2e-2)
    for i, b in enumerate(r["at_boundary"]):
        assert np.isnan(r["se"][i]) == b or (not b and np.isfinite(r["se"][i]))


def test_seatbelts_bsm_pile_up_and_law_effect():
    """Harvey-Durbin (1986) basic structural model on R ``datasets::Seatbelts``
    (fetched live -- the series is GPL-distributed and not vendored): the
    fixture holds the statsmodels/SciPy optimum only; the fit must reach it,
    flag the slope and seasonal pile-ups, and agree with a live statsmodels
    evaluation at the optimum."""
    sb = UC["seatbelts"]
    try:
        import statsmodels.api as sm
        df = sm.datasets.get_rdataset("Seatbelts", "datasets").data
    except Exception as exc:  # network / cache unavailable
        pytest.skip(f"Rdatasets Seatbelts unreachable in this environment: {exc}")
    y = np.log(np.asarray(df["drivers"], float))
    x = np.column_stack([np.log(df["PetrolPrice"]), np.log(df["kms"]), df["law"]]).astype(float)
    assert len(y) == sb["nobs"] == 192
    est = tsecon.unobserved_components(y, level="lltrend", seasonal=12, exog=x, forecast_steps=12,
                                       forecast_exog=np.tile(x[-1], (12, 1)))
    best = sb["mle"]["best"]
    assert est["loglik"] >= best["llf"] - 1e-5
    names_p = list(est["param_names"])
    assert names_p == sb["mle"]["param_names"]
    assert est["k_states"] == sb["k_states"] == 13 and est["nobs_diffuse"] == sb["nobs_diffuse"]
    # Pile-up: statsmodels' optimum has sigma2.trend ~ 3e-21 and sigma2.seasonal ~ 4e-19.
    assert best["params"][names_p.index("sigma2.trend")] < 1e-12
    assert best["params"][names_p.index("sigma2.seasonal")] < 1e-12
    assert est["at_boundary"][names_p.index("sigma2.trend")] and est["at_boundary"][names_p.index("sigma2.seasonal")]
    assert np.isnan(est["se"][names_p.index("sigma2.trend")])
    assert est["at_boundary"] == best["at_boundary"]
    for nm in ("sigma2.irregular", "sigma2.level", "beta.x1", "beta.x2", "beta.x3"):
        i = names_p.index(nm)
        assert not est["at_boundary"][i]
        assert est["params"][i] == pytest.approx(best["params"][i], rel=2e-3, abs=1e-6)
        # Conditional on the two piled-up variances being exactly zero —
        # which is what a pile-up means. statsmodels' own se_approx inverts
        # the full Hessian instead, and that Hessian is indefinite here (its
        # bse for sigma2.trend is NaN), so the two differ by a factor of 26
        # on sigma2.level: a difference of definition, recorded in the
        # fixture as both numbers.
        assert est["se"][i] == pytest.approx(best["se_conditional"][i], rel=2e-2)
    law = est["params"][names_p.index("beta.x3")]
    assert -0.35 < law < -0.15                      # a sizable reduction; no published value asserted
    assert len(est["forecast"]) == 12
    # Live statsmodels at the fixture optimum, fixed parameters, 1e-8.
    from statsmodels.tsa.statespace.structural import UnobservedComponents
    ref = UnobservedComponents(y, level="lltrend", seasonal=12, exog=x, use_exact_diffuse=True).smooth(np.asarray(best["params"]))
    r = tsecon.unobserved_components(y, level="lltrend", seasonal=12, exog=x, fixed_params=best["params"])
    assert r["loglik"] == pytest.approx(ref.llf, rel=1e-8)
    np.testing.assert_allclose(r["level"], ref.level.smoothed, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(r["seasonal"], ref.seasonal.smoothed, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(r["filtered_state"], ref.filtered_state.T, rtol=1e-8, atol=1e-8)


# ---------------------------------------------------------------------- TVP
def test_tvp_fixed_and_missing_match_the_statsmodels_transcription():
    tv = UC["tvp"]
    y, x = np.asarray(tv["y"], float), np.asarray(tv["x"], float)
    for block, yy in ((tv["fixed"], y), (tv["missing"], np.asarray(tv["y_missing"], float))):
        r = tsecon.tvp_regression(yy, x, fixed_params=block["params"])
        assert r["k"] == 3 and list(r["coef_names"]) == ["const", "x1", "x2"]
        assert r["loglik"] == pytest.approx(block["loglike"], rel=TOL)
        assert r["aic"] == pytest.approx(block["aic"], rel=TOL) and r["bic"] == pytest.approx(block["bic"], rel=TOL)
        for key in ("beta_filtered", "beta_filtered_var", "beta_smoothed"):
            _nan_close(r[key], block[key])
        _smoothed_var_close(r["beta_smoothed_var"], block["beta_smoothed_var"], block,
                            r["nobs_diffuse"], "beta_smoothed_var")
        _nan_close(r["fitted"], block["fitted"])
        _nan_close(r["resid"], block["resid"])
        d = r["nobs_diffuse"]
        assert d == block["nobs_diffuse"]
        obs = ~np.isnan(np.asarray(block["resid"], float))
        obs[:d] = False
        _nan_close(np.asarray(r["std_resid"], float)[obs],
                   np.asarray(block["std_resid"], float)[obs])


def test_tvp_zero_state_variance_is_recursive_least_squares():
    tv = UC["tvp"]
    y, x = np.asarray(tv["y"], float), np.asarray(tv["x"], float)
    rls = tv["rls"]
    r = tsecon.tvp_regression(y, x, fixed_params=[rls["scale"], 0.0, 0.0, 0.0])
    assert r["pile_up"] == [True, True, True]
    assert r["loglik"] == pytest.approx(rls["llf"], rel=TOL)
    np.testing.assert_allclose(r["beta_filtered"], rls["filtered_coefficients"], rtol=TOL, atol=TOL)
    np.testing.assert_allclose(r["beta_filtered"][-1], rls["params"], rtol=TOL)
    # Live: statsmodels RecursiveLS on the same design.
    try:
        from statsmodels.regression.recursive_ls import RecursiveLS
    except ImportError:  # pragma: no cover
        return
    X = np.column_stack([np.ones(len(y)), x])
    ref = RecursiveLS(y, X).fit()
    np.testing.assert_allclose(r["beta_filtered"], ref.recursive_coefficients.filtered.T, rtol=1e-9, atol=1e-9)


def test_tvp_mle_two_optimizers_and_pile_up_flag():
    tv = UC["tvp"]
    y, x = np.asarray(tv["y"], float), np.asarray(tv["x"], float)
    r = tsecon.tvp_regression(y, x)
    best = tv["mle"]["best"]
    assert r["loglik"] >= best["llf"] - 1e-5
    assert r["pile_up"] == [False, False, True] and r["at_boundary"] == [False, False, False, True]
    assert np.isnan(r["se"][3]) and np.all(np.isfinite(r["se"][:3]))
    np.testing.assert_allclose(r["params"][:3], best["params"][:3], rtol=2e-3)
    np.testing.assert_allclose(r["se"][:3], best["se_conditional"][:3], rtol=2e-2)
    assert r["at_boundary"] == best["at_boundary"]
    assert list(r["param_names"]) == ["sigma2.irregular", "sigma2.const", "sigma2.x1", "sigma2.x2"]
    assert r["sigma2_eps"] == r["params"][0] and list(r["sigma2_beta"]) == list(r["params"][1:])
    # Truth: sigma2_eps = 1, sigma2_beta = [0.01, 0.0025, 0] — recovered in
    # order of magnitude, the zero one flagged.
    truth = tv["truth"]
    assert 0.7 < r["sigma2_eps"] / truth["sigma2_eps"] < 1.4
    assert r["sigma2_beta"][0] < 0.05 and r["sigma2_beta"][1] < 0.02


# ---------------------------------------------------------------- refusals
def test_inert_options_raise_when_passed_explicitly():
    y = np.asarray(UC["sim"]["y"], float)
    with pytest.raises(ValueError, match="stochastic_seasonal was given but seasonal is None"):
        tsecon.unobserved_components(y, stochastic_seasonal=True)
    with pytest.raises(ValueError, match="freq_seasonal_harmonics was given but freq_seasonal is None"):
        tsecon.unobserved_components(y, freq_seasonal_harmonics=[2])
    with pytest.raises(ValueError, match="stochastic_freq_seasonal was given but freq_seasonal is None"):
        tsecon.unobserved_components(y, stochastic_freq_seasonal=[True])
    with pytest.raises(ValueError, match="damped_cycle was given but cycle=False"):
        tsecon.unobserved_components(y, damped_cycle=True)
    with pytest.raises(ValueError, match="stochastic_cycle was given but cycle=False"):
        tsecon.unobserved_components(y, stochastic_cycle=True)
    with pytest.raises(ValueError, match="cycle_period_bounds was given but cycle=False"):
        tsecon.unobserved_components(y, cycle_period_bounds=[4, 20])
    with pytest.raises(ValueError, match="forecast_exog was given but forecast_steps = 0"):
        tsecon.unobserved_components(y, forecast_exog=np.ones((3, 1)))
    with pytest.raises(ValueError, match="forecast_exog was given but the model has no regressors"):
        tsecon.unobserved_components(y, forecast_steps=3, forecast_exog=np.ones((3, 1)))


def test_teaching_errors_name_the_argument():
    y = np.asarray(UC["sim"]["y"], float)
    x = np.asarray(UC["sim"]["x"], float)
    with pytest.raises(ValueError, match='level = "bogus"'):
        tsecon.unobserved_components(y, level="bogus")
    with pytest.raises(ValueError, match="seasonal = 1"):
        tsecon.unobserved_components(y, seasonal=1)
    with pytest.raises(ValueError, match=r"freq_seasonal\[0\]\.harmonics = 7"):
        tsecon.unobserved_components(y, freq_seasonal=[12], freq_seasonal_harmonics=[7])
    with pytest.raises(ValueError, match="freq_seasonal_harmonics has 2 entries but freq_seasonal has 1"):
        tsecon.unobserved_components(y, freq_seasonal=[12], freq_seasonal_harmonics=[2, 3])
    with pytest.raises(ValueError, match=r"cycle_period_bounds = \(1, 10\)"):
        tsecon.unobserved_components(y, cycle=True, cycle_period_bounds=[1, 10])
    with pytest.raises(ValueError, match="cycle_period_bounds must be \\[min, max\\]"):
        tsecon.unobserved_components(y, cycle=True, cycle_period_bounds=[4])
    with pytest.raises(ValueError, match="exog column 0 has length 119"):
        tsecon.unobserved_components(y, exog=x[:-1])
    with pytest.raises(ValueError, match="exog column 1 is identically zero"):
        tsecon.unobserved_components(y, exog=np.column_stack([x[:, 0], np.zeros(len(y))]))
    with pytest.raises(ValueError, match='level = "fixed intercept"'):
        tsecon.unobserved_components(y, level="fixed intercept")
    with pytest.raises(ValueError, match="forecast_steps = 3 with 2 regressors requires forecast_exog"):
        tsecon.unobserved_components(y, exog=x, forecast_steps=3)
    with pytest.raises(ValueError, match="fixed_params has length 1 but the specification has 2 parameters"):
        tsecon.unobserved_components(y, fixed_params=[1.0])
    with pytest.raises(ValueError, match=r"fixed_params\[1\] \(sigma2.level\) = -1"):
        tsecon.unobserved_components(y, fixed_params=[1.0, -1.0])
    with pytest.raises(ValueError, match="n_starts = 0"):
        tsecon.unobserved_components(y, n_starts=0)
    with pytest.raises(ValueError, match="y is constant"):
        tsecon.unobserved_components(np.ones(30))
    with pytest.raises(ValueError, match="y is empty"):
        tsecon.unobserved_components(np.array([]))
    with pytest.raises(ValueError, match="y contains an infinity"):
        tsecon.unobserved_components(np.array([1.0, np.inf, 2.0, 3.0, 4.0]))
    with pytest.raises(ValueError, match="x column 0 contains a NaN"):
        tsecon.tvp_regression(y, np.full((len(y), 1), np.nan))
    with pytest.raises(ValueError, match="x column 0 has length 119"):
        tsecon.tvp_regression(y, x[:-1])
    with pytest.raises(ValueError, match="x has no columns and constant = false"):
        tsecon.tvp_regression(y, np.empty((len(y), 0)), constant=False)
    with pytest.raises(ValueError, match="fixed_params has length 2"):
        tsecon.tvp_regression(y, x, fixed_params=[1.0, 0.0])
    with pytest.raises(ValueError, match=r"fixed_params\[0\] = 0 \(sigma2_eps\)"):
        tsecon.tvp_regression(y, x, fixed_params=[0.0, 0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="at least k \\+ 2"):
        tsecon.tvp_regression(y[:4], x[:4])
    with pytest.raises(ValueError):
        tsecon.unobserved_components(y, seasonal=-4)


def test_counts_that_size_an_allocation_are_refused_not_aborted():
    """The wrapper only refuses integer counts at or above 2**48; everything
    below has to be caught in Rust, before the allocation. Each of these
    aborted the allocator or allocated a state per unit of a user integer
    before the bounds were added."""
    y = np.asarray(UC["sim"]["y"], float)
    with pytest.raises(ValueError, match="seasonal = 10000000 with 120 observations"):
        tsecon.unobserved_components(y, seasonal=10 ** 7)
    with pytest.raises(ValueError, match=r"freq_seasonal\[0\]\.period = 10000000 with 120 observations"):
        tsecon.unobserved_components(y, freq_seasonal=[1e7])
    with pytest.raises(ValueError, match="forecast_steps = 1000000000000: at most 100000"):
        tsecon.unobserved_components(y, forecast_steps=10 ** 12)
    with pytest.raises(ValueError, match="at most 64 starting values"):
        tsecon.unobserved_components(y, n_starts=2 ** 40)
    with pytest.raises(ValueError, match="at most 64 starting values"):
        tsecon.tvp_regression(y, np.asarray(UC["sim"]["x"], float), n_starts=2 ** 40)


# ------------------------------------------------------------ behaviour
def test_determinism_and_scale_invariance():
    y = np.asarray(UC["sim"]["y"], float)
    a = tsecon.unobserved_components(y, level="lltrend", seasonal=4, forecast_steps=4)
    b = tsecon.unobserved_components(y, level="lltrend", seasonal=4, forecast_steps=4)
    np.testing.assert_array_equal(a["params"], b["params"])
    np.testing.assert_array_equal(a["level"], b["level"])
    c = tsecon.unobserved_components(1000.0 * y, level="lltrend", seasonal=4, forecast_steps=4)
    want = np.asarray(a["params"]) * 1e6
    # atol floors the comparison for a variance that piles up at zero: the
    # relative difference of two numerical zeros says nothing.
    # rtol 1e-7: the two runs standardize by scales that differ in the last
    # ulp (`var(1000 y)` is not exactly `1e6 var(y)`), measured deviation
    # 1.6e-9 here.
    np.testing.assert_allclose(c["params"], want, rtol=1e-7, atol=1e-10 * np.max(np.abs(want)))
    # Exact-diffuse likelihood: the shift is -(n - nobs_diffuse) ln c, not
    # -n ln c — the diffuse contributions -(ln 2pi + ln F_inf)/2 never touch
    # the units of y.
    assert c["nobs_diffuse"] == a["nobs_diffuse"]
    shift = (len(y) - a["nobs_diffuse"]) * np.log(1000.0)
    assert c["loglik"] == pytest.approx(a["loglik"] - shift, rel=1e-9)
    np.testing.assert_allclose(c["forecast"], np.asarray(a["forecast"]) * 1000.0, rtol=1e-9)
    assert a["at_boundary"] == c["at_boundary"]
    t1 = tsecon.tvp_regression(y, np.asarray(UC["sim"]["x"], float))
    t2 = tsecon.tvp_regression(y, np.asarray(UC["sim"]["x"], float))
    np.testing.assert_array_equal(t1["params"], t2["params"])


def test_components_and_keys_are_consistent():
    y = np.asarray(UC["sim"]["y"], float)
    r = tsecon.unobserved_components(y, level="lltrend", seasonal=4, cycle=True, damped_cycle=True,
                                     stochastic_cycle=True, freq_seasonal=[12], freq_seasonal_harmonics=[2],
                                     fixed_params=[0.8, 0.3, 0.02, 0.1, 0.05, 0.2, 0.6, 0.85])
    assert list(r["state_names"])[:3] == ["level", "trend", "seasonal"]
    assert r["k_states"] == 2 + 3 + 4 + 2
    st = np.asarray(r["smoothed_state"])
    np.testing.assert_allclose(r["level"], st[:, 0])
    np.testing.assert_allclose(r["slope"], st[:, 1])
    np.testing.assert_allclose(r["seasonal"], st[:, 2])
    np.testing.assert_allclose(r["freq_seasonal"][0], st[:, 5] + st[:, 7])
    np.testing.assert_allclose(r["cycle"], st[:, 9])
    sv = np.asarray(r["smoothed_state_var"])
    np.testing.assert_allclose(r["freq_seasonal_var"][0], sv[:, 5] + sv[:, 7])
    # y ≈ level + seasonal + freq + cycle + irregular: the smoothed sum tracks y.
    recon = np.asarray(r["level"]) + np.asarray(r["seasonal"]) + np.asarray(r["freq_seasonal"][0]) + np.asarray(r["cycle"])
    assert np.corrcoef(recon, y)[0, 1] > 0.95
    # Fitted plus residual is y at observed periods.
    np.testing.assert_allclose(np.asarray(r["fitted"]) + np.asarray(r["resid"]), y, rtol=1e-12)


def _doc_tokens(fn):
    return set(re.findall(r"`([A-Za-z_][A-Za-z_0-9]*)`", fn.__doc__ or ""))


def test_docstrings_name_every_returned_key_and_stub_matches():
    y = np.asarray(UC["sim"]["y"], float)
    r = tsecon.unobserved_components(y, level="lltrend", seasonal=4, fixed_params=[0.8, 0.3, 0.02, 0.1], forecast_steps=2)
    missing = set(r.keys()) - _doc_tokens(tsecon.unobserved_components)
    assert not missing, sorted(missing)
    t = tsecon.tvp_regression(y, np.asarray(UC["sim"]["x"], float), fixed_params=[1.0, 0.0, 0.0, 0.0])
    missing = set(t.keys()) - _doc_tokens(tsecon.tvp_regression)
    assert not missing, sorted(missing)
    stub = (Path(tsecon.__file__).parent / "__init__.pyi").read_text(encoding="utf-8")
    for name in ("unobserved_components", "tvp_regression"):
        assert re.search(rf"^def {name}\(", stub, re.MULTILINE)
    import inspect
    sig = inspect.signature(tsecon.unobserved_components)
    assert all(p.default is not Ellipsis for p in sig.parameters.values())
    assert sig.parameters["level"].default == "llevel" and sig.parameters["n_starts"].default == 3
