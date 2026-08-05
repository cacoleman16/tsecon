"""Regression tests for the four defects the interval-coverage audit found.

Each section is written as if the fix did not exist: the assertions are the
ones that would have *failed* before it landed, not a restatement of the new
code. The four defects, and what proves each one is closed:

1. `ols` had no leverage-corrected standard errors. hc1's `n/(n-k)` factor is
   flat in leverage and barely moves; hc2/hc3 divide by `(1-h_t)^{1,2}` and are
   what a small sample with an influential point actually needs. Proof: parity
   with statsmodels HC2/HC3, and the elementwise ordering hc3 >= hc2 >= hc0.

2. `iv_gmm(weight="hac")` was a SILENT NO-OP. `bandwidth` defaulted to 0.0, and
   a Bartlett kernel truncated at zero lags *is* the White estimator, so it
   returned results bit-identical to `weight="robust"` while the caller
   believed they had serial-correlation robustness. Proof: the two now DIFFER.

3. `iv_gmm` reported no first-stage diagnostic, and a naive one fabricates an
   F of ~1e33 when the instruments reproduce a regressor exactly. Proof: the
   entry is present when defined and OMITTED when it is not.

4. `arima_fit`'s h-step forecast se omitted the estimated drift's own
   uncertainty (which grows like h^2) and measurably under-covered. Proof: the
   opt-in correction matches the closed form while the default path is
   unchanged to machine precision.

Caveats these tests deliberately encode: a `first_stage` entry may be missing
without the fit having failed, so entries are indexed by `regressor` and never
by position; and `drift_uncertainty` is opt-in because the two forecast
standard errors are different estimands, not a right and a wrong one.
"""

import math

import numpy as np
import pytest

import tsecon

# --------------------------------------------------------------------------- #
# 1. ols: hc2 / hc3
# --------------------------------------------------------------------------- #
# The audit's own design: T=25 with a chi2(1) regressor, so a couple of points
# carry most of the leverage and the HC ladder actually separates.
OLS_T = 25


def _leverage_design(seed: int = 0, t: int = OLS_T):
    rng = np.random.default_rng(seed)
    x = rng.chisquare(1, t)
    xmat = np.column_stack([np.ones(t), x])
    # Heteroskedastic by construction: the noise scale rises with x.
    y = 1.0 + 0.5 * x + rng.standard_normal(t) * (0.5 + x)
    return y, xmat


def test_ols_accepts_hc2_and_hc3():
    y, x = _leverage_design()
    for se_type in ("hc2", "hc3"):
        fit = tsecon.ols(y, x, se_type=se_type)
        assert fit["se_type"] == se_type
        assert fit["bse"].shape == (x.shape[1],)
        assert np.all(np.isfinite(fit["bse"])) and np.all(fit["bse"] > 0)


def test_hc2_and_hc3_match_statsmodels():
    sm = pytest.importorskip("statsmodels.api")
    y, x = _leverage_design()
    ref = sm.OLS(y, x).fit()
    for se_type in ("hc2", "hc3"):
        got = tsecon.ols(y, x, se_type=se_type)["bse"]
        want = ref.get_robustcov_results(cov_type=se_type.upper()).bse
        np.testing.assert_allclose(got, want, rtol=0, atol=1e-12)


def test_hc_ladder_is_ordered_by_leverage_weight():
    """hc3 >= hc2 >= hc0 elementwise: the weights 1/(1-h)^{2,1,0} are ordered
    and >= 1, so each rung inflates the sandwich meat by a psd amount."""
    for seed in range(4):
        y, x = _leverage_design(seed=seed, t=30)
        hc0, hc2, hc3 = (tsecon.ols(y, x, se_type=t)["bse"] for t in ("hc0", "hc2", "hc3"))
        assert np.all(hc3 >= hc2), seed
        assert np.all(hc2 >= hc0), seed


def test_leverage_correction_is_not_hc1_by_another_name():
    """The point of the fix. hc1 rescales hc0 by the same scalar
    sqrt(n/(n-k)) for every coefficient -- it cannot know which observation is
    influential. hc3's inflation varies coefficient by coefficient because it
    is driven by leverage, and on this design it is materially larger."""
    y, x = _leverage_design()
    hc0 = tsecon.ols(y, x, se_type="hc0")["bse"]
    hc1 = tsecon.ols(y, x, se_type="hc1")["bse"]
    hc3 = tsecon.ols(y, x, se_type="hc3")["bse"]

    flat = math.sqrt(OLS_T / (OLS_T - x.shape[1]))
    np.testing.assert_allclose(hc1 / hc0, flat, rtol=1e-12)  # identical per coef
    assert np.all(hc3 > hc1)
    assert len(set(np.round(hc3 / hc0, 6))) > 1  # leverage-dependent, not flat
    assert hc3[1] > 1.10 * hc1[1]  # the slope carries the influential points


def test_unknown_se_type_message_now_lists_hc2_and_hc3():
    y, x = _leverage_design()
    with pytest.raises(ValueError) as exc:
        tsecon.ols(y, x, se_type="hc4")
    msg = str(exc.value)
    assert "hc4" in msg
    assert "hc2" in msg and "hc3" in msg


def test_unit_leverage_is_refused_not_reported_as_a_huge_se():
    """An observation with leverage numerically 1 has a residual of exactly 0
    by construction and an infinite HC2/HC3 weight. Returning a near-infinite
    standard error would look like a computed answer; it is not one."""
    rng = np.random.default_rng(5)
    t = 12
    dummy = np.zeros(t)
    dummy[3] = 1.0  # a singleton dummy -> h_3 == 1
    x = np.column_stack([np.ones(t), rng.standard_normal(t), dummy])
    y = rng.standard_normal(t)

    for se_type in ("hc2", "hc3"):
        with pytest.raises(ValueError) as exc:
            tsecon.ols(y, x, se_type=se_type)
        msg = str(exc.value)
        assert "leverage" in msg
        assert "3" in msg  # names the offending observation

    # hc0 has no leverage weight, so it still returns finite numbers here.
    assert np.all(np.isfinite(tsecon.ols(y, x, se_type="hc0")["bse"]))


# --------------------------------------------------------------------------- #
# 2. iv_gmm: the HAC bandwidth no-op
# --------------------------------------------------------------------------- #
GMM_N = 200


def _ar1_error_iv(seed: int = 11, n: int = GMM_N, phi: float = 0.8):
    """One endogenous regressor, two excluded instruments, AR(1) errors --
    the design where HAC and White weighting must not agree."""
    rng = np.random.default_rng(seed)
    z1 = rng.standard_normal(n)
    z2 = rng.standard_normal(n)
    e = rng.standard_normal(n)
    u = np.zeros(n)
    for t in range(1, n):
        u[t] = phi * u[t - 1] + e[t]
    x = 0.8 * z1 + 0.5 * z2 + 0.6 * u + rng.standard_normal(n)
    y = 1.0 + 2.0 * x + u
    return (
        np.column_stack([np.ones(n), x]),
        np.column_stack([np.ones(n), z1, z2]),
        y,
    )


def test_hac_weighting_is_not_identical_to_robust():
    """THE BUG. bandwidth defaulted to 0.0, Bartlett-at-zero-lags is White, so
    weight="hac" used to return results bit-identical to weight="robust". The
    assertion is that they now DIFFER."""
    x, z, y = _ar1_error_iv()
    robust = tsecon.iv_gmm(x, z, y, method="2step", weight="robust")
    hac = tsecon.iv_gmm(x, z, y, method="2step", weight="hac")

    delta = np.max(np.abs(np.asarray(hac["bse"]) - np.asarray(robust["bse"])))
    assert delta > 1e-8, "weight='hac' is still returning the White estimator"


def test_hac_bandwidth_is_reported_and_follows_the_newey_west_rule():
    x, z, y = _ar1_error_iv()
    hac = tsecon.iv_gmm(x, z, y, method="2step", weight="hac")
    expected = math.floor(4.0 * (GMM_N / 100.0) ** (2.0 / 9.0))
    assert expected == 4  # pins the rule itself, not just its reimplementation
    assert hac["hac_bandwidth"] == pytest.approx(float(expected))


def test_robust_reports_no_bandwidth():
    x, z, y = _ar1_error_iv()
    robust = tsecon.iv_gmm(x, z, y, method="2step", weight="robust")
    assert "hac_bandwidth" in robust
    assert robust["hac_bandwidth"] is None


def test_explicit_bandwidth_is_honoured_and_reported():
    x, z, y = _ar1_error_iv()
    hac = tsecon.iv_gmm(x, z, y, method="2step", weight="hac", bandwidth=10.0)
    assert hac["hac_bandwidth"] == pytest.approx(10.0)
    auto = tsecon.iv_gmm(x, z, y, method="2step", weight="hac")
    # More lags is a different truncation, hence a different sandwich.
    assert np.max(np.abs(np.asarray(hac["bse"]) - np.asarray(auto["bse"]))) > 1e-10


def test_explicit_zero_bandwidth_raises_and_explains_that_it_is_white():
    x, z, y = _ar1_error_iv()
    with pytest.raises(ValueError) as exc:
        tsecon.iv_gmm(x, z, y, method="2step", weight="hac", bandwidth=0.0)
    msg = str(exc.value)
    assert "no-op" in msg
    assert "White" in msg
    assert "Bartlett" in msg
    assert "robust" in msg  # says what it silently degenerates to


def test_two_stage_least_squares_refuses_a_weight_argument():
    """2SLS fixes its weight at (Z'Z/n)^-1 by construction, so accepting a
    weight was the same silent no-op the bandwidth default was."""
    x, z, y = _ar1_error_iv()
    with pytest.raises(ValueError) as exc:
        tsecon.iv_gmm(x, z, y, method="2sls", weight="hac")
    msg = str(exc.value)
    assert "2sls" in msg
    assert "Z'Z" in msg
    # 2SLS with the default weight is still fine, and reports no bandwidth.
    assert tsecon.iv_gmm(x, z, y, method="2sls")["hac_bandwidth"] is None


def test_iterated_gmm_also_resolves_the_automatic_bandwidth():
    x, z, y = _ar1_error_iv()
    fit = tsecon.iv_gmm(x, z, y, method="iterated", weight="hac")
    assert fit["hac_bandwidth"] == pytest.approx(4.0)
    assert fit["steps"] > 2


# --------------------------------------------------------------------------- #
# 3. iv_gmm: first_stage
# --------------------------------------------------------------------------- #
FIRST_STAGE_KEYS = {"regressor", "fstat", "dof_num", "dof_den", "pval"}


def _mixed_exog_endog(seed: int = 7, n: int = 150):
    """X = [const, w, x] with only x endogenous; Z = [const, w, z1, z2]."""
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(n)
    z1 = rng.standard_normal(n)
    z2 = rng.standard_normal(n)
    u = rng.standard_normal(n)
    x = 0.9 * z1 + 0.4 * z2 + 0.7 * u + rng.standard_normal(n)
    y = 1.0 + 0.5 * w + 2.0 * x + u
    return (
        np.column_stack([np.ones(n), w, x]),
        np.column_stack([np.ones(n), w, z1, z2]),
        y,
    )


def test_first_stage_present_for_an_endogenous_regressor():
    x, z, y = _mixed_exog_endog()
    fit = tsecon.iv_gmm(x, z, y, method="2step", weight="robust")
    fs = fit["first_stage"]
    assert isinstance(fs, list) and len(fs) == 1

    entry = fs[0]
    assert set(entry) == FIRST_STAGE_KEYS
    # Indexed by regressor position in X, NOT by position in this list: the
    # endogenous column is X[:, 2], and the two exogenous ones got no entry.
    assert entry["regressor"] == 2
    assert entry["dof_num"] == 2  # two excluded instruments
    assert entry["dof_den"] == len(y) - z.shape[1]
    assert entry["fstat"] > 10.0
    assert 0.0 <= entry["pval"] <= 1.0
    assert np.isfinite(entry["fstat"])


def test_first_stage_is_empty_for_a_fully_exogenous_model():
    """Every regressor is its own instrument, so there is no first stage to
    report. An empty list is the honest answer, not a failure."""
    rng = np.random.default_rng(3)
    n = 120
    w = rng.standard_normal(n)
    z1 = rng.standard_normal(n)
    x = np.column_stack([np.ones(n), w])
    z = np.column_stack([np.ones(n), w, z1])
    y = 1.0 + 0.5 * w + rng.standard_normal(n)

    fit = tsecon.iv_gmm(x, z, y, method="2step", weight="robust")
    assert fit["first_stage"] == []
    # A missing entry is not a failed fit -- the estimates are still there.
    assert np.all(np.isfinite(fit["params"]))
    assert np.all(np.isfinite(fit["bse"]))


def test_exact_linear_combination_yields_no_entry_not_a_fabricated_f():
    """When the instruments reproduce a regressor exactly the first-stage
    residual is 0 and the F is a 0/0 that floating point renders as ~1e33.
    Publishing that number as a weak-instrument diagnostic is worse than
    publishing nothing, so the entry is omitted."""
    rng = np.random.default_rng(3)
    n = 120
    z1 = rng.standard_normal(n)
    z2 = rng.standard_normal(n)
    x_exact = 0.7 * z1 - 1.3 * z2  # exactly spanned by the instruments
    x = np.column_stack([np.ones(n), x_exact])
    z = np.column_stack([np.ones(n), z1, z2])
    y = 1.0 + 2.0 * x_exact + rng.standard_normal(n)

    fit = tsecon.iv_gmm(x, z, y, method="2step", weight="robust")
    assert fit["first_stage"] == []
    for entry in fit["first_stage"]:  # defensive: nothing astronomically large
        assert entry["fstat"] < 1e12


def test_first_stage_entries_must_be_looked_up_by_regressor():
    """With two endogenous regressors there is one entry each, and the list
    index is not the regressor index. CAVEAT these numbers do not discharge:
    both can clear 10 while the system is under-identified -- this is a
    per-regressor diagnostic, not a joint weak-identification test."""
    rng = np.random.default_rng(7)
    n = 200
    z1, z2, z3 = (rng.standard_normal(n) for _ in range(3))
    u = rng.standard_normal(n)
    x1 = 0.9 * z1 + 0.4 * z2 + 0.7 * u + rng.standard_normal(n)
    x2 = 0.8 * z2 + 0.5 * z3 + 0.6 * u + rng.standard_normal(n)
    x = np.column_stack([np.ones(n), x1, x2])
    z = np.column_stack([np.ones(n), z1, z2, z3])
    y = 1.0 + 2.0 * x1 + x2 + u

    fs = tsecon.iv_gmm(x, z, y, method="2step", weight="robust")["first_stage"]
    by_regressor = {e["regressor"]: e for e in fs}
    assert set(by_regressor) == {1, 2}  # the constant is exogenous -> no entry
    assert 0 not in by_regressor
    for entry in fs:
        assert set(entry) == FIRST_STAGE_KEYS
        assert entry["dof_num"] == 3  # three excluded instruments
        assert np.isfinite(entry["fstat"])


# --------------------------------------------------------------------------- #
# 4. arima_fit: drift uncertainty and parameter covariance
# --------------------------------------------------------------------------- #
ARIMA_T = 60
ARIMA_H = 24


def _random_walk_with_drift(seed: int = 2, t: int = ARIMA_T, mu: float = 0.3):
    rng = np.random.default_rng(seed)
    return np.cumsum(mu + rng.standard_normal(t))


def _sigma(fit):
    idx = list(fit["param_names"]).index("sigma2")
    return math.sqrt(fit["params"][idx])


def test_default_forecast_se_is_unchanged_and_still_sigma_root_h():
    """The pre-fix expectation, asserted exactly. drift_uncertainty defaults to
    False so the default path stays bit-identical and keeps matching the
    statsmodels get_forecast golden -- the two are different estimands, not a
    right and a wrong one."""
    y = _random_walk_with_drift()
    fit = tsecon.arima_fit(y, 0, 1, 0, constant=True, forecast_steps=ARIMA_H)
    assert fit["drift_uncertainty"] is False

    steps = np.arange(1, ARIMA_H + 1)
    np.testing.assert_allclose(
        fit["forecast_se"], _sigma(fit) * np.sqrt(steps), rtol=0, atol=1e-12
    )
    # Passing the flag explicitly as False must not perturb a single bit.
    explicit = tsecon.arima_fit(
        y, 0, 1, 0, constant=True, forecast_steps=ARIMA_H, drift_uncertainty=False
    )
    np.testing.assert_array_equal(explicit["forecast_se"], fit["forecast_se"])


def test_drift_uncertainty_widens_the_band():
    y = _random_walk_with_drift()
    kw = dict(constant=True, forecast_steps=ARIMA_H, conf_alpha=0.05)
    base = tsecon.arima_fit(y, 0, 1, 0, **kw)
    wide = tsecon.arima_fit(y, 0, 1, 0, drift_uncertainty=True, **kw)

    assert wide["drift_uncertainty"] is True
    assert np.all(wide["forecast_se"] > base["forecast_se"])
    base_width = base["forecast_upper"] - base["forecast_lower"]
    wide_width = wide["forecast_upper"] - wide["forecast_lower"]
    assert np.all(wide_width > base_width)
    # The h^2 term compounds, so the gap grows with the horizon.
    ratio = np.asarray(wide["forecast_se"]) / np.asarray(base["forecast_se"])
    assert ratio[-1] > ratio[0]
    # The point forecast is a different object and must not move.
    np.testing.assert_allclose(wide["forecast_mean"], base["forecast_mean"], rtol=0, atol=1e-12)


def test_drift_corrected_se_matches_the_closed_form():
    """For ARIMA(0,1,0) with a constant the h-step forecast is
    y_T + h*mu_hat, and Var(mu_hat) = sigma^2/(T-1), so the corrected se is
    sigma * sqrt(h + h^2/(T-1))."""
    y = _random_walk_with_drift()
    fit = tsecon.arima_fit(
        y, 0, 1, 0, constant=True, forecast_steps=ARIMA_H, drift_uncertainty=True
    )
    steps = np.arange(1, ARIMA_H + 1, dtype=float)
    closed_form = _sigma(fit) * np.sqrt(steps + steps**2 / (ARIMA_T - 1))
    np.testing.assert_allclose(fit["forecast_se"], closed_form, rtol=1e-6)


def test_drift_uncertainty_requires_a_forecast():
    y = _random_walk_with_drift()
    with pytest.raises(ValueError) as exc:
        tsecon.arima_fit(y, 0, 1, 0, constant=True, forecast_steps=0, drift_uncertainty=True)
    assert "forecast_steps" in str(exc.value)


def test_drift_uncertainty_requires_a_constant():
    """With constant=False there is no estimated drift, so the correction would
    be identically zero -- silently returning the uncorrected se would make the
    flag a no-op of exactly the kind defect (2) was."""
    y = _random_walk_with_drift()
    with pytest.raises(ValueError) as exc:
        tsecon.arima_fit(
            y, 0, 1, 0, constant=False, forecast_steps=6, drift_uncertainty=True
        )
    assert "constant" in str(exc.value)


def _assert_param_cov_contract(fit):
    """ARIMA previously reported no parameter standard errors at all. Either
    the triple is present and internally consistent, or it refuses honestly."""
    for key in ("bse", "param_cov", "cov_ok"):
        assert key in fit, key
    k = len(fit["params"])

    if not fit["cov_ok"]:
        assert fit["bse"] is None and fit["param_cov"] is None
        assert "cov_error" in fit and fit["cov_error"]
        return

    cov = np.asarray(fit["param_cov"])
    bse = np.asarray(fit["bse"])
    assert cov.shape == (k, k)
    np.testing.assert_array_equal(cov, cov.T)  # symmetric, not just close
    assert np.all(np.diag(cov) >= 0.0)
    assert bse.shape == (k,)
    np.testing.assert_allclose(bse, np.sqrt(np.diag(cov)), rtol=0, atol=1e-12)
    assert np.all(np.isfinite(bse))


def test_arima_reports_parameter_standard_errors():
    y = _random_walk_with_drift()
    fit = tsecon.arima_fit(y, 0, 1, 0, constant=True, forecast_steps=ARIMA_H)
    assert fit["cov_ok"] is True
    _assert_param_cov_contract(fit)
    assert list(fit["param_names"]) == ["const", "sigma2"]


def test_param_cov_contract_holds_across_specifications():
    rng = np.random.default_rng(4)
    n = 150
    e = rng.standard_normal(n)
    arma = np.zeros(n)
    for t in range(1, n):
        arma[t] = 0.6 * arma[t - 1] + e[t] + 0.3 * e[t - 1]

    for y, order in ((arma, (1, 0, 1)), (arma, (1, 0, 0)), (_random_walk_with_drift(), (0, 1, 0))):
        fit = tsecon.arima_fit(y, *order, forecast_steps=0)
        _assert_param_cov_contract(fit)
