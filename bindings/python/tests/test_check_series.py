"""Tests for tsecon.check_series — the one-call diagnostic battery.

Routing tests use fixture-free seeded DGPs (the frontier-test style): each
canonical series shape must land in the right family and produce a
recommendation that routes to callables that actually exist. Contract tests
pin the report's dict shape: JSON-serializability, non-empty evidence,
teaching errors, and the exact top-level key sets. A Monte Carlo size check
verifies the battery's test families do not over-reject on pure noise —
the multiple-testing stance is only honest if the per-family sizes hold.
"""
import functools
import json

import numpy as np
import pytest

import tsecon


# --------------------------------------------------------------------------- #
# seeded DGP reports, built once and shared
# --------------------------------------------------------------------------- #
@functools.lru_cache(maxsize=None)
def wn_report():
    return tsecon.check_series(np.random.default_rng(0).standard_normal(300))


@functools.lru_cache(maxsize=None)
def rw_report():
    return tsecon.check_series(np.cumsum(np.random.default_rng(1).standard_normal(400)))


@functools.lru_cache(maxsize=None)
def ar1_report():
    rng = np.random.default_rng(2)
    e = rng.standard_normal(400)
    y = np.zeros(400)
    for t in range(1, 400):
        y[t] = 0.7 * y[t - 1] + e[t]
    return tsecon.check_series(y)


@functools.lru_cache(maxsize=None)
def garch_report():
    rng = np.random.default_rng(3)
    n = 1000
    z = rng.standard_normal(n)
    sig2 = np.empty(n)
    eps = np.empty(n)
    sig2[0] = 0.1 / (1 - 0.15 - 0.8)
    eps[0] = np.sqrt(sig2[0]) * z[0]
    for t in range(1, n):
        sig2[t] = 0.1 + 0.15 * eps[t - 1] ** 2 + 0.8 * sig2[t - 1]
        eps[t] = np.sqrt(sig2[t]) * z[t]
    return tsecon.check_series(eps)


BREAK_TRUTH = 149  # last observation of regime 1


@functools.lru_cache(maxsize=None)
def break_report():
    rng = np.random.default_rng(4)
    y = rng.standard_normal(600)
    y[BREAK_TRUTH + 1 :] += 1.0
    return tsecon.check_series(y)


@functools.lru_cache(maxsize=None)
def frac_report():
    shocks = np.random.default_rng(5).standard_normal(600)
    return tsecon.check_series(tsecon.frac_integrate(shocks, 0.3))


@functools.lru_cache(maxsize=None)
def seasonal_report():
    rng = np.random.default_rng(7)
    t = np.arange(240)
    y = 2.0 * np.sin(2 * np.pi * t / 12) + rng.standard_normal(240)
    return tsecon.check_series(y, seasonal_period=12)


@functools.lru_cache(maxsize=None)
def coint_report():
    rng = np.random.default_rng(14)
    n = 400
    x = np.cumsum(rng.standard_normal(n))
    u = np.zeros(n)
    e = rng.standard_normal(n)
    for t in range(1, n):
        u[t] = 0.6 * u[t - 1] + e[t]
    return tsecon.check_series(np.column_stack([x, x + u]))


@functools.lru_cache(maxsize=None)
def var3_report():
    rng = np.random.default_rng(9)
    A = np.array([[0.5, 0.1, 0.0], [0.0, 0.4, 0.1], [0.1, 0.0, 0.3]])
    data = np.zeros((500, 3))
    for t in range(1, 500):
        data[t] = A @ data[t - 1] + rng.standard_normal(3)
    return tsecon.check_series(data)


@functools.lru_cache(maxsize=None)
def mixed_report():
    rng = np.random.default_rng(10)
    a = np.zeros(400)
    e = rng.standard_normal(400)
    for t in range(1, 400):
        a[t] = 0.5 * a[t - 1] + e[t]
    b = np.cumsum(rng.standard_normal(400))
    return tsecon.check_series(np.column_stack([a, b]))


@functools.lru_cache(maxsize=None)
def wide_report():
    return tsecon.check_series(np.random.default_rng(11).standard_normal((300, 13)))


@functools.lru_cache(maxsize=None)
def trend_report():
    # y = 0.008*t + noise lands in the Conflict quadrant ('Detrend'): the
    # battery must not run the downstream families on the trending level.
    rng = np.random.default_rng(0)
    return tsecon.check_series(0.008 * np.arange(200) + rng.standard_normal(200))


@functools.lru_cache(maxsize=None)
def wide_i1_report():
    # 13 independent random walks: johansen's trace critical values stop at
    # k=12, so the rank comes back undetermined (None), not zero.
    steps = np.random.default_rng(2).standard_normal((300, 13))
    return tsecon.check_series(np.cumsum(steps, axis=0))


@functools.lru_cache(maxsize=None)
def collinear_report():
    # two identical random walks: a classic beginner input the VAR step
    # cannot estimate — the battery must report, not crash.
    rw = np.cumsum(np.random.default_rng(5).standard_normal(300))
    return tsecon.check_series(np.column_stack([rw, rw]))


def all_reports():
    return [
        wn_report(), rw_report(), ar1_report(), garch_report(), break_report(),
        frac_report(), seasonal_report(), coint_report(), var3_report(),
        mixed_report(), wide_report(), trend_report(), wide_i1_report(),
        collinear_report(),
    ]


def topics(report):
    return [r["topic"] for r in report["recommendations"]]


def rec_by_topic(report, topic):
    matches = [r for r in report["recommendations"] if r["topic"] == topic]
    assert matches, f"no '{topic}' recommendation; got {topics(report)}"
    return matches[0]


# --------------------------------------------------------------------------- #
# univariate routing
# --------------------------------------------------------------------------- #
def test_white_noise_gets_the_clean_series_entry():
    rec = rec_by_topic(wn_report(), "no_red_flags")
    # the expected-false-alarm arithmetic is spelled out, not corrected away
    assert "expected" in (rec["suggestion"] + rec["finding"]).lower() or (
        rec["evidence"]["expected_false_rejections"] > 0
    )


def test_white_noise_quadrant_and_analysis_scale():
    rep = wn_report()
    assert rep["stationarity"]["quadrant"] == "Stationary"
    assert rep["analysis_scale"]["scale"] == "level"


def test_multiple_testing_arithmetic_is_shown_not_applied():
    rep = wn_report()
    mt = rep["multiple_testing"]
    assert mt["n_tests"] == len(rep["tests_run"])
    # the expected-false-alarm count is over TRUE-NULL tests only: ADF's null
    # (a unit root) is false on a clean stationary series, so an ADF rejection
    # there is correct, not a false alarm
    n_true_null = sum(
        1 for t in rep["tests_run"] if not t["name"].startswith("adf")
    )
    assert mt["n_true_null"] == n_true_null
    assert mt["n_true_null"] == mt["n_tests"] - 1  # exactly one ADF ran
    assert mt["expected_false_rejections"] == pytest.approx(
        mt["n_true_null"] * mt["alpha"]
    )
    assert "ADF" in mt["note"]
    for entry in rep["tests_run"]:
        assert set(entry) == {"name", "pvalue", "alpha"}


def test_random_walk_differences_downstream_analysis():
    rep = rw_report()
    assert rep["stationarity"]["quadrant"] == "UnitRoot"
    assert rep["analysis_scale"]["scale"] == "first_difference"
    assert rep["serial_correlation"]["computed_on"] == "first_difference"
    assert rep["arch_effects"]["computed_on"] == "first_difference"


def test_random_walk_unit_root_recommendation_echoes_the_evidence():
    rep = rw_report()
    rec = rec_by_topic(rep, "unit_root")
    assert rec["evidence"]["adf_p_value"] == rep["stationarity"]["adf_p_value"]
    assert rec["evidence"]["kpss_p_value"] == rep["stationarity"]["kpss_p_value"]
    assert "arima_fit" in rec["functions"]


def test_random_walk_flags_the_persistent_regressor_hazard():
    rec = rec_by_topic(rw_report(), "persistent_regressor")
    assert "predictive_regression" in rec["functions"]
    assert "ivx_test" in rec["functions"]


def test_random_walk_long_memory_reads_d_against_the_unit_root():
    lm = rw_report()["long_memory"]
    assert lm["computed_on"] == "level"
    assert 0.5 < lm["gph"]["d"] < 1.5  # d near 1 for a random walk
    assert lm["gph_on_differences"] is not None
    assert "unit" in lm["joint_interpretation"].lower()


def test_ar1_suggested_orders_are_sane():
    orders = ar1_report()["serial_correlation"]["suggested_arma_orders"]
    assert 1 <= orders["p"] <= 3  # PACF cuts at 1 for an AR(1)
    assert orders["q"] == 0


def test_ar1_arma_recommendation_is_hedged_and_routes_to_arima_fit():
    rec = rec_by_topic(ar1_report(), "arma_orders")
    assert "arima_fit" in rec["functions"]
    assert "starting point" in rec["suggestion"].lower()


def test_garch_dgp_fires_the_arch_recommendation():
    rec = rec_by_topic(garch_report(), "arch")
    assert "garch_fit" in rec["functions"]
    assert garch_report()["arch_effects"]["p_value"] < 0.05


def test_garch_fat_tails_escalate_to_t_innovations():
    rep = garch_report()
    assert rep["normality"]["rejected"]  # GARCH aggregates to fat tails
    assert "dist='t'" in rec_by_topic(rep, "arch")["suggestion"]


def test_broken_mean_fires_the_break_recommendation():
    rep = break_report()
    # ADF still rejects here (the shift is moderate and off-center), KPSS
    # rejects too: the Conflict quadrant draws 'Detrend', so the scan runs on
    # the OLS-detrended level — where a genuine mean shift still stands out
    # (whereas a deterministic trend left in place would fake breaks).
    assert rep["analysis_scale"]["scale"] == "detrended_level"
    assert rep["breaks"]["computed_on"] == "detrended_level"
    rec = rec_by_topic(rep, "structural_break")
    assert "bai_perron" in rec["functions"]
    assert rec["evidence"]["sup_f_p_value"] < 0.05


def test_broken_mean_dates_are_recovered():
    bp = break_report()["breaks"]["bai_perron"]
    assert bp is not None and bp["n_breaks"] >= 1
    assert abs(bp["break_dates"][0] - BREAK_TRUTH) <= 5


def test_break_report_carries_the_homogeneous_ci_caveat():
    breaks = break_report()["breaks"]
    assert "homogeneous" in breaks["caveat"]


def test_fractional_series_fires_the_long_memory_recommendation():
    rep = frac_report()
    rec = rec_by_topic(rep, "long_memory")
    assert "frac_diff" in rec["functions"]
    assert "long_memory_d" in rec["functions"]
    assert 0.1 < rep["long_memory"]["gph"]["d"] < 0.7  # truth d = 0.3


def test_seasonal_evidence_is_reported_at_the_given_period():
    seas = seasonal_report()["seasonality"]
    assert seas["seasonal_period"] == 12
    assert [e["lag"] for e in seas["acf_at_seasonal_lags"]] == [12, 24, 36]
    assert seas["periodogram_ordinate"]["period"] == pytest.approx(12.0, abs=1.0)


def test_seasonal_recommendation_is_honest_about_the_gap():
    rec = rec_by_topic(seasonal_report(), "seasonality")
    assert "no seasonal ARIMA" in rec["suggestion"]
    assert "roadmap" in rec["suggestion"]


def test_lags_argument_controls_the_ljung_box_horizon():
    rep = tsecon.check_series(
        np.random.default_rng(0).standard_normal(300), lags=15
    )
    assert rep["serial_correlation"]["ljung_box"]["lags"][-1] == 15


def test_outlier_screen_flags_an_injected_spike():
    y = np.random.default_rng(1).standard_normal(200)
    y[50] = 12.0
    out = tsecon.check_series(y)["outliers"]
    assert out["count"] >= 1
    assert 50 in out["indices"]


# --------------------------------------------------------------------------- #
# multivariate routing
# --------------------------------------------------------------------------- #
def test_cointegrated_pair_routes_to_vecm():
    rep = coint_report()
    assert all(s["verdict"] == "UnitRoot" for s in rep["per_series"])
    assert rep["cointegration"]["rank"] >= 1
    rec = rec_by_topic(rep, "cointegration")
    assert "vecm" in rec["functions"]
    assert "johansen" in rec["functions"]


def test_cointegrated_stability_note_points_at_the_vecm_route():
    stab = coint_report()["stability"]
    assert stab["is_stable"] is False  # a levels VAR keeps the unit root
    assert "vecm" in stab["note"]


def test_stationary_var3_routes_to_var_fit_with_a_small_lag():
    rep = var3_report()
    rec = rec_by_topic(rep, "var")
    assert {"var_fit", "var_irf", "var_fevd"} <= set(rec["functions"])
    assert rep["var_lag_selection"]["selected_by_bic"] <= 2  # truth is VAR(1)
    assert rep["stability"]["is_stable"] is True
    assert rep["stability"]["min_root"] > 1.0


def test_var3_offers_connectedness_for_spillover_questions():
    rec = rec_by_topic(var3_report(), "spillovers")
    assert rec["functions"] == ["connectedness"]


def test_multivariate_always_mentions_local_projections():
    rec = rec_by_topic(var3_report(), "single_shock_irf")
    assert {"lp", "lp_iv"} <= set(rec["functions"])


def test_mixed_integration_pair_gets_a_warning():
    rep = mixed_report()
    verdicts = {s["verdict"] for s in rep["per_series"]}
    assert "Stationary" in verdicts and "UnitRoot" in verdicts
    rec = rec_by_topic(rep, "mixed_integration")
    assert "check_stationarity" in rec["functions"]
    assert "skipped_reason" in rep["cointegration"]  # johansen needs all-I(1)


def test_wide_panel_gets_the_dimensionality_note():
    rec = rec_by_topic(wide_report(), "dimensionality")
    assert {"favar", "factor_model"} <= set(rec["functions"])


def test_per_series_entries_are_compact():
    rep = wide_report()
    assert len(rep["per_series"]) == 13
    for entry in rep["per_series"]:
        assert set(entry) == {"index", "verdict", "recommendation"}


# --------------------------------------------------------------------------- #
# cross-report contract
# --------------------------------------------------------------------------- #
def test_every_recommended_function_exists():
    for rep in all_reports():
        for rec in rep["recommendations"]:
            for name in rec["functions"]:
                assert hasattr(tsecon, name), (
                    f"recommendation '{rec['topic']}' routes to a function "
                    f"that does not exist: tsecon.{name}"
                )


def test_every_recommendation_has_nonempty_evidence():
    for rep in all_reports():
        for rec in rep["recommendations"]:
            assert isinstance(rec["evidence"], dict) and rec["evidence"], (
                f"'{rec['topic']}' has empty evidence"
            )
            assert set(rec) == {
                "topic", "finding", "evidence", "suggestion", "functions",
                "caveat",
            }


def test_reports_are_json_serializable():
    for rep in all_reports():
        json.dumps(rep)


def test_nan_input_raises_a_teaching_error_with_the_count():
    y = np.random.default_rng(0).standard_normal(100)
    y[[7, 40, 41]] = np.nan
    with pytest.raises(ValueError, match="3 NaN"):
        tsecon.check_series(y)
    with pytest.raises(ValueError, match="index 7"):
        tsecon.check_series(y)


def test_3d_input_raises():
    with pytest.raises(ValueError, match="3D"):
        tsecon.check_series(np.zeros((4, 4, 4)))


def test_too_short_raises_with_a_teaching_message():
    with pytest.raises(ValueError, match="at least 20"):
        tsecon.check_series(np.arange(10.0))


def test_plain_python_list_input_works():
    values = np.random.default_rng(42).standard_normal(120).tolist()
    rep = tsecon.check_series(values)
    assert rep["kind"] == "univariate" and rep["n"] == 120


def test_single_column_squeezes_to_univariate():
    data = np.random.default_rng(0).standard_normal((150, 1))
    rep = tsecon.check_series(data)
    assert rep["kind"] == "univariate" and rep["n"] == 150


def test_family_sizes_hold_on_white_noise():
    """200 seeded white-noise draws: every family whose null is true must
    reject at close to the nominal 5% — within [0, 0.12], a loose 3-sigma
    binomial bound. ADF is excluded because its null (a unit root) is false
    for white noise. This is what makes the multiple-testing arithmetic in
    the report honest."""
    families = {"kpss": 0, "ljung_box": 0, "arch_lm": 0, "jarque_bera": 0,
                "sup_f_test": 0}
    n_draws = 200
    rng = np.random.default_rng(20260721)
    for _ in range(n_draws):
        rep = tsecon.check_series(rng.standard_normal(200))
        for entry in rep["tests_run"]:
            base = entry["name"].split(" ")[0]
            if base in families and entry["pvalue"] < 0.05:
                families[base] += 1
    for name, count in families.items():
        rate = count / n_draws
        assert 0.0 <= rate <= 0.12, f"{name} rejects at {rate:.3f} on noise"


# --------------------------------------------------------------------------- #
# regression tests for the adversarial-review fixes
# --------------------------------------------------------------------------- #
def test_conflict_quadrant_detrends_the_analysis_object():
    rep = trend_report()
    assert rep["stationarity"]["recommendation"] == "Detrend"
    assert rep["analysis_scale"]["scale"] == "detrended_level"
    assert "trend" in rep["analysis_scale"]["rationale"]
    for family in ("serial_correlation", "arch_effects", "normality", "breaks"):
        assert rep[family]["computed_on"] == "detrended_level"
    assert rep["long_memory"]["computed_on"] == "detrended_level"


def test_pure_trend_emits_no_trend_artifact_recommendations():
    # Before the Detrend branch existed this DGP fired structural_break,
    # arma_orders, arch, AND long_memory — all artifacts of the raw trend.
    tps = topics(trend_report())
    assert "trend_or_break" in tps
    for artifact in ("structural_break", "arma_orders", "arch", "long_memory"):
        assert artifact not in tps, f"trend-artifact rec '{artifact}' fired"


def test_alpha_outside_the_kpss_clamp_raises_a_teaching_error():
    y = np.random.default_rng(0).standard_normal(100)
    for bad in (0.01, 0.005, 0.2):
        with pytest.raises(ValueError, match=r"0\.01, 0\.10"):
            tsecon.check_series(y, alpha=bad)
    # the boundary that CAN flip both tests still works
    assert tsecon.check_series(y, alpha=0.10)["alpha"] == pytest.approx(0.10)


@pytest.mark.parametrize("n", [20, 24, 40])
def test_short_even_lengths_do_not_crash_the_pacf(n):
    # the compiled pacf needs nlags < n/2 STRICTLY; min(20, m // 2) violated
    # that for even analysis objects of length 20..40
    rep = tsecon.check_series(np.random.default_rng(0).standard_normal(n))
    assert rep["kind"] == "univariate" and rep["n"] == n


def test_even_length_differences_do_not_crash_the_pacf():
    # a Difference verdict at n=41 makes the analysis object an even m=40
    y = np.cumsum(np.random.default_rng(3).standard_normal(41))
    rep = tsecon.check_series(y)
    assert rep["kind"] == "univariate"
    assert rep["analysis_scale"]["scale"] == "first_difference"


def test_inconclusive_quadrant_hedges_the_persistence_finding():
    rep = tsecon.check_series(np.random.default_rng(7).standard_normal(20))
    assert rep["stationarity"]["quadrant"] == "Inconclusive"
    rec = rec_by_topic(rep, "persistent_regressor")
    assert rec["finding"].startswith("IF")
    assert "cannot tell" in rec["finding"]
    assert rec["evidence"]["quadrant"] == "Inconclusive"


def test_unit_root_quadrant_keeps_the_flat_persistence_finding():
    rec = rec_by_topic(rw_report(), "persistent_regressor")
    assert rec["finding"] == "the series is highly persistent"


def test_undetermined_johansen_rank_is_not_reported_as_rank_zero():
    rep = wide_i1_report()
    assert rep["integration_summary"]["all_i1"] is True
    assert "skipped_reason" in rep["cointegration"]
    assert "12" in rep["cointegration"]["skipped_reason"]
    # the old bug: rank=None fell into difference_first, asserting a Johansen
    # rank-0 result that never existed
    assert "difference_first" not in topics(rep)
    rec = rec_by_topic(rep, "cointegration_undetermined")
    assert rec["evidence"]["rank"] is None
    assert "johansen" in rec["functions"]
    assert rep["var_lag_selection"]["scale"] == "level"


def test_collinear_panel_reports_instead_of_crashing():
    rep = collinear_report()
    assert "skipped_reason" in rep["var_lag_selection"]
    assert "collinear" in rep["var_lag_selection"]["skipped_reason"]
    assert rep["stability"] == {"skipped_reason": "no VAR was fit"}


def test_gph_far_below_one_is_not_called_consistent_with_a_unit_root():
    # fractional d=0.7: the level GPH d-hat sits significantly below 1 and
    # the joint interpretation must say so rather than endorse d ~ 1
    shocks = np.random.default_rng(6).standard_normal(2000)
    rep = tsecon.check_series(tsecon.frac_integrate(shocks, 0.7))
    assert rep["stationarity"]["quadrant"] == "UnitRoot"
    gph = rep["long_memory"]["gph"]
    assert abs(gph["d"] - 1.0) > 1.64 * gph["se"]
    text = rep["long_memory"]["joint_interpretation"]
    assert "significantly below 1" in text
    assert "near 1" not in text


def test_gph_near_one_reads_as_consistent_with_the_unit_root():
    text = rw_report()["long_memory"]["joint_interpretation"]
    assert "indistinguishable from 1" in text


def test_univariate_top_level_keys_snapshot():
    assert set(wn_report()) == {
        "kind", "n", "alpha", "descriptives", "outliers", "stationarity",
        "analysis_scale", "serial_correlation", "arch_effects", "normality",
        "breaks", "long_memory", "seasonality", "tests_run",
        "multiple_testing", "recommendations",
    }


def test_multivariate_top_level_keys_snapshot():
    assert set(var3_report()) == {
        "kind", "n", "k", "alpha", "per_series", "integration_summary",
        "cointegration", "var_lag_selection", "stability", "tests_run",
        "multiple_testing", "recommendations",
    }


# --------------------------------------------------------------------------- #
# post-review regressions (verifier catch + minor findings)
# --------------------------------------------------------------------------- #
def test_seasonal_period_is_validated():
    y = np.random.default_rng(0).standard_normal(80)
    with pytest.raises(ValueError, match="seasonal_period"):
        tsecon.check_series(y, seasonal_period=-12)
    with pytest.raises(ValueError, match="seasonal_period"):
        tsecon.check_series(y, seasonal_period=1)
    with pytest.raises(ValueError, match="two full seasonal cycles"):
        tsecon.check_series(y, seasonal_period=60)


def test_clean_series_with_seasonal_period_keeps_no_red_flags():
    # the informational seasonality entry must not block the clean bill
    rng = np.random.default_rng(7)
    rep = tsecon.check_series(rng.standard_normal(240), seasonal_period=12)
    tps = [r["topic"] for r in rep["recommendations"]]
    assert "seasonality" in tps
    assert "no_red_flags" in tps


def test_arch_suggestion_names_the_differenced_object_on_i1_series():
    # I(1) level whose increments carry GARCH: the battery analyzes the first
    # differences, so the garch_fit target must be the differences, not y
    rng = np.random.default_rng(3)
    n = 600
    e, h = np.zeros(n), np.ones(n)
    z = rng.standard_normal(n)
    for t in range(1, n):
        h[t] = 0.1 + 0.15 * e[t - 1] ** 2 + 0.80 * h[t - 1]
        e[t] = np.sqrt(h[t]) * z[t]
    rep = tsecon.check_series(np.cumsum(e))
    assert rep["analysis_scale"]["scale"] == "first_difference"
    rec = rec_by_topic(rep, "arch")
    assert "garch_fit(np.diff(y)" in rec["suggestion"]
    assert "garch_fit(y)" not in rec["suggestion"]


def test_vecm_k_ar_diff_matches_the_stated_rule():
    # the scale_note states k_ar_diff = selected_lag - 1; the rec must agree
    # (floored at 0 — vecm accepts k_ar_diff=0)
    rep = coint_report()
    sel = rep["var_lag_selection"]["selected_by_bic"]
    rec = rec_by_topic(rep, "cointegration")
    assert f"k_ar_diff={max(sel - 1, 0)}" in rec["suggestion"]


def test_johansen_states_its_deterministic_case():
    co = coint_report()["cointegration"]
    assert "det_order=0" in co["method"]
    rec = rec_by_topic(coint_report(), "cointegration")
    assert "small samples" in rec["caveat"]


def test_multivariate_lags_zero_names_the_cap_not_dof():
    rng = np.random.default_rng(5)
    rep = tsecon.check_series(rng.standard_normal((120, 2)), lags=0)
    assert "caps the search" in rep["var_lag_selection"]["skipped_reason"]


def test_outlier_spike_is_cross_referenced_in_the_normality_caveat():
    rng = np.random.default_rng(11)
    y = rng.standard_normal(300)
    y[150] += 30.0
    rep = tsecon.check_series(y)
    assert rep["outliers"]["count"] >= 1
    assert 150 in rep["outliers"]["indices"]
    fired = [
        r for r in rep["recommendations"] if r["topic"] in ("normality", "arch")
    ]
    assert fired, "a 30-sigma spike must trip JB or ARCH-LM"
    assert any("outlier" in r["caveat"] for r in fired)
