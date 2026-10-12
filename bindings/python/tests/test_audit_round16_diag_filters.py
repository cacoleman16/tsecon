"""Regression pins for audit round 16: diagnostics, filters and their kin.

Each test pins one confirmed cell of the round-16 house sweep:

* A8 — `stl`/`mstl` iteration counts (`inner_iter`, `outer_iter`,
  `iterate`) are fixed loop counts; 2**31 used to run for about a day. They
  are refused by name against a LOESS work budget before any smoothing.
* A9 — `reset_test(max_power=2**31)` built power columns until the allocator
  aborted; it is refused at once, naming `max_power`.
* The pre-existing `arch_lm`/`adf` allocation aborts at lag orders the sample
  admits but no machine holds (and the same in `zivot_andrews`' break-date
  scan); refused by name before any lag column.
* B1/B2 — refusals that named no keyword or an engine-internal one now name
  the Python keyword (`nlags`, `lags`, `max_lags`, `maxlag`, `period`,
  `lrv_bandwidth`, `lrv_lags`), including the stationarity workflows that
  take no lag keyword.
* B5 — `check_series` validates `lags` and `max_breaks` up front.
* C1 — `dsge_solve`'s docstring no longer promises an indeterminate verdict.

Only numpy and pytest are needed (the CI wheel jobs install nothing else).
"""
import math
import re
import time

import numpy as np
import pytest

import tsecon

STL_BUDGET = 2**32
LAG_CELLS, LAG_WORK = 2**26, 2**35
BG_CELLS, BG_WORK = 2**26, 2**34


def _seasonal(n=96, seed=4):
    t = np.arange(n)
    rng = np.random.default_rng(seed)
    return 10.0 + np.sin(2 * np.pi * t / 12) + 0.3 * rng.standard_normal(n) + 0.02 * t


def _walk(n, seed=5):
    return np.cumsum(np.random.default_rng(seed).standard_normal(n))


def _ar1(n, seed=0, phi=0.5):
    e = np.random.default_rng(seed).standard_normal(n)
    y = np.empty(n)
    y[0] = e[0]
    for t in range(1, n):
        y[t] = phi * y[t - 1] + e[t]
    return y


def _quick(fn, *args, **kwargs):
    """Call `fn`, expecting a ValueError, and return (message, seconds)."""
    t0 = time.perf_counter()
    with pytest.raises(ValueError) as exc:
        fn(*args, **kwargs)
    return str(exc.value), time.perf_counter() - t0


def _stl_pass_units(n, period, seasonal, low_pass, trend):
    """The documented LOESS units of one STL inner pass (unit jumps)."""
    k = -(-n // period)
    points = n + 1
    return (
        (points + 3 * period) * min(seasonal, k)
        + points * min(low_pass, n)
        + points * min(trend, n)
        + 12 * n
    )


# ----------------------------------------------------------------- A8: STL


@pytest.mark.parametrize("count", [2**31, 2**40])
def test_stl_inner_iter_is_budgeted(count):
    y = _seasonal()
    msg, dt = _quick(
        tsecon.stl, y, period=12, seasonal=7, trend=23, low_pass=13,
        inner_iter=count, outer_iter=0,
    )
    units = _stl_pass_units(96, 12, 7, 13, 23)
    best = max(32, STL_BUDGET // units)
    assert msg.startswith(f"stl: inner_iter = {count} is past the STL work budget"), msg
    assert msg.endswith(f"pass inner_iter <= {best}"), msg
    assert dt < 5.0


@pytest.mark.parametrize("count", [2**31, 2**40])
def test_stl_outer_iter_is_budgeted(count):
    y = _seasonal()
    msg, dt = _quick(
        tsecon.stl, y, period=12, seasonal=7, trend=23, low_pass=13,
        inner_iter=2, outer_iter=count,
    )
    units = _stl_pass_units(96, 12, 7, 13, 23)
    # 2 (outer_iter + 1) passes within the budget.
    best = STL_BUDGET // (2 * units) - 1
    assert msg.startswith(f"stl: outer_iter = {count} is past the STL work budget"), msg
    assert msg.endswith(f"pass outer_iter <= {best}"), msg
    assert dt < 5.0


@pytest.mark.parametrize(
    "kwargs, param",
    [
        ({"inner_iter": 2**31, "outer_iter": 0}, "inner_iter"),
        ({"inner_iter": 2, "outer_iter": 2**31}, "outer_iter"),
    ],
)
def test_mstl_forwarded_iteration_counts_are_budgeted(kwargs, param):
    y = _seasonal()
    msg, dt = _quick(
        tsecon.mstl, y, periods=[12], windows=[7], iterate=2, trend=23,
        low_pass=13, **kwargs,
    )
    assert msg.startswith(f"mstl: {param} = {kwargs[param]} is past the STL work budget"), msg
    assert f"pass {param} <= " in msg
    assert dt < 5.0


def test_mstl_iterate_is_budgeted_with_several_periods():
    t = np.arange(240)
    y = 5 + np.sin(2 * np.pi * t / 7) + 0.5 * np.cos(2 * np.pi * t / 30) + 0.01 * t
    msg, dt = _quick(tsecon.mstl, y, periods=[7, 30], iterate=2**31)
    assert msg.startswith("mstl: iterate = 2147483648 is past the STL work budget"), msg
    assert dt < 5.0
    # With one period MSTL runs one round whatever `iterate` says
    # (statsmodels semantics), so the same count costs nothing there.
    r = tsecon.mstl(y, periods=[7], iterate=2**31)
    assert r["iterate"] == 1


def test_stl_boundary_overshoot_names_the_raised_count():
    # One step past the largest admissible outer_iter: inner_iter = 2 also
    # "overshoots" by one step (2 > 1), but the refusal names the count the
    # caller raised, not its small companion.
    y = _seasonal()
    units = _stl_pass_units(96, 12, 7, 13, 23)
    best = STL_BUDGET // (2 * units) - 1
    kw = dict(period=12, seasonal=7, trend=23, low_pass=13, inner_iter=2)
    msg, _ = _quick(tsecon.stl, y, outer_iter=best + 1, **kw)
    assert msg.startswith(f"stl: outer_iter = {best + 1} is past"), msg
    assert msg.endswith(f"pass outer_iter <= {best}"), msg
    # MSTL: one round past the largest admissible iterate names iterate,
    # not the default inner_iter = 5.
    t = np.arange(240)
    y2 = 5 + np.sin(2 * np.pi * t / 7) + 0.5 * np.cos(2 * np.pi * t / 30) + 0.01 * t
    msg, _ = _quick(tsecon.mstl, y2, periods=[7, 30], iterate=2**31)
    top = int(re.search(r"pass iterate <= (\d+)$", msg).group(1))
    msg, _ = _quick(tsecon.mstl, y2, periods=[7, 30], iterate=top + 1)
    assert msg.startswith(f"mstl: iterate = {top + 1} is past"), msg
    assert msg.endswith(f"pass iterate <= {top}"), msg


def test_stl_counts_that_only_fit_together_quote_the_pass_budget():
    # No single count fits with the other held fixed: the refusal says to
    # reduce them together (never a "<= 0" no count can take).
    y = _seasonal()
    units = _stl_pass_units(96, 12, 7, 13, 23)
    msg, _ = _quick(
        tsecon.stl, y, period=12, seasonal=7, trend=23, low_pass=13,
        inner_iter=2**20, outer_iter=2**20,
    )
    assert msg.endswith(
        "no single count fits with the others held fixed, so reduce them together "
        f"to inner_iter * (outer_iter + 1) <= {STL_BUDGET // units}, or to the default "
        "region of at most 32 inner passes per fit over at most two rounds"
    ), msg
    assert "<= 0" not in msg


def test_stl_counts_within_the_budget_still_run():
    y = _seasonal()
    base = tsecon.stl(y, period=12)
    assert base["config"]["inner_iter"] == 5 and base["config"]["outer_iter"] == 0
    robust = tsecon.stl(y, period=12, robust=True)
    assert robust["config"]["inner_iter"] == 2 and robust["config"]["outer_iter"] == 15
    # Past the 32 free passes but far inside the budget.
    r = tsecon.stl(y, period=12, inner_iter=40, outer_iter=3)
    assert r["config"]["inner_iter"] == 40
    np.testing.assert_allclose(r["seasonal"] + r["trend"] + r["resid"], y, rtol=0, atol=1e-12)


# ---------------------------------------------------------------- A9: RESET


@pytest.mark.parametrize("max_power", [199, 5000, 2**22, 2**31, 2**40])
def test_reset_max_power_is_refused_before_any_power_column(max_power):
    rng = np.random.default_rng(0)
    y = rng.standard_normal(200)
    x = np.column_stack([np.ones(200), rng.standard_normal(200)])
    msg, dt = _quick(tsecon.reset_test, y, x, max_power=max_power)
    assert msg.startswith(f"max_power = {max_power} adds {max_power - 1} fitted-value powers"), msg
    # n > k + max_power - 1 with n = 200, k = 2.
    assert msg.endswith("pass max_power <= 198"), msg
    assert dt < 5.0


def test_reset_power_overflow_names_max_power():
    # |yhat| about 1e40: yhat^8 overflows. It used to reach the OLS engine as
    # an inf "in the design column", blaming the data.
    rng = np.random.default_rng(0)
    x = np.column_stack([np.ones(60), rng.standard_normal(60)])
    y = (2.0 + x[:, 1] + 0.1 * rng.standard_normal(60)) * 1e40
    msg, _ = _quick(tsecon.reset_test, y, x, max_power=12)
    m = re.match(r"max_power = 12: the fitted-value power yhat\^(\d+) overflows to infinity", msg)
    assert m, msg
    power = int(m.group(1))
    assert msg.endswith(
        f"pass max_power <= {power - 1}, or rescale y (RESET is invariant to the scale of y)"
    ), msg
    # Rescaled, the same test runs and gives the same statistic.
    a = tsecon.reset_test(y, x, max_power=3)["fstat"]
    b = tsecon.reset_test(y / 1e40, x, max_power=3)["fstat"]
    assert a == pytest.approx(b, rel=1e-8)


def test_reset_max_power_past_the_auxiliary_budget_is_refused_by_name():
    n = 100_000
    rng = np.random.default_rng(1)
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    y = x[:, 1] + rng.standard_normal(n)
    msg, dt = _quick(tsecon.reset_test, y, x, max_power=30_000)
    best = int(re.search(r"pass max_power <= (\d+)$", msg).group(1))

    def fits(p):
        return n * p <= BG_CELLS and n * p * p // 2 + 6 * p**3 <= BG_WORK

    # p = k + max_power - 1 regressors.
    assert fits(2 + best - 1) and not fits(2 + best), best
    assert "past both the memory budget" in msg
    assert dt < 5.0


# ---------------------------------------------- arch_lm / adf allocation aborts


def test_arch_lm_order_past_the_design_budget_is_refused_by_name():
    n = 1_000_000
    e = np.random.default_rng(3).standard_normal(n)
    msg, dt = _quick(tsecon.arch_lm, e, nlags=400_000)
    assert msg.startswith("arch_lm: nlags = 400000 needs a lag design of 240000000000 entries"), msg
    best = int(re.search(r"pass nlags <= (\d+), or a shorter series$", msg).group(1))
    assert (n - best) * best <= LAG_CELLS < (n - best - 1) * (best + 1)
    assert dt < 5.0


def test_adf_fixed_lag_past_the_design_budget_is_refused_by_name():
    n = 1_000_000
    y = _walk(n)
    msg, dt = _quick(tsecon.adf, y, maxlag=400_000, autolag=None)
    assert msg.startswith("adf: maxlag = 400000 needs a lag design of"), msg
    best = int(re.search(r"pass maxlag <= (\d+), or a shorter series$", msg).group(1))
    # (n - 1 - p) (p + 2) entries for regression "c".
    assert (n - 1 - best) * (best + 2) <= LAG_CELLS < (n - 2 - best) * (best + 3)
    assert dt < 5.0


def test_adf_default_search_is_refused_only_when_its_design_exceeds_memory():
    n = 2_000_000
    y = _walk(n, 9)
    default = math.ceil(12 * (n / 100) ** 0.25)
    assert default == 143
    msg, dt = _quick(tsecon.adf, y)
    assert msg.startswith(f"adf: the default maxlag = {default} needs a lag design of"), msg
    assert "past the memory budget of 2^26" in msg
    assert dt < 5.0


def test_adf_search_past_the_work_budget_is_refused_by_name():
    n = 20_000
    y = _walk(n, 7)
    msg, dt = _quick(tsecon.adf, y, regression="ct", maxlag=3000)
    assert "past the work budget of 2^35 multiply-adds" in msg, msg
    best = int(re.search(r"pass maxlag <= (\d+), or a shorter series$", msg).group(1))

    def work(p):  # a search over 0..=p from 3 fixed columns, plus the refit
        rows, k = n - 1 - p, 3 + p
        return rows * sum(j * j for j in range(3, k + 1)) + (n - 1) * k * k

    assert work(best) <= LAG_WORK < work(best + 1)
    assert dt < 5.0


def _za_scan(n, s, cands, lag):
    """The documented Zivot-Andrews break-date scan: one (n-1-lag) x (s+lag)
    design refit at each of `cands` candidate breaks."""
    rows, k = n - 1 - lag, s + lag
    return rows * k, cands * rows * k * k


def test_zivot_andrews_fixed_lag_past_the_design_budget_is_refused_by_name():
    # lags = 14000 is inside the trimmed window (int(n*trim) - 1 = 14999) on
    # 1e5 observations, but the scan's design is 1.2e9 entries (9.6 GB): it
    # used to abort the interpreter in the allocator.
    n = 100_000
    y = _walk(n, 21)
    msg, dt = _quick(tsecon.zivot_andrews, y, autolag=None, lags=14_000)
    assert msg.startswith("zivot_andrews: lags = 14000 needs a lag design of"), msg
    assert "past both the memory budget" in msg, msg
    best = int(re.search(r"pass lags <= (\d+), or a shorter series$", msg).group(1))
    cands, free = n - 2 * 15_000, math.ceil(12 * (n / 100) ** 0.25)

    def admissible(p):  # lags up to the default cap are exempt from the work budget
        cells, work = _za_scan(n, 4, cands, p)
        return cells <= LAG_CELLS and (p <= free or work <= LAG_WORK)

    # At this length the scan past the default lag (68) is over the work
    # budget, so the default cap is the longest lag that fits.
    assert admissible(best) and not admissible(best + 1)
    assert best == free == 68
    assert dt < 5.0


def test_zivot_andrews_scan_work_is_budgeted_past_the_default_lag():
    # n = 3000: lags = 400 fits the window and memory, but the scan would
    # take about 20 minutes. Lags up to the default cap (29) are exempt.
    n = 3000
    y = _walk(n, 23)
    msg, dt = _quick(tsecon.zivot_andrews, y, regression="ct", autolag=None, lags=400)
    assert msg.startswith("zivot_andrews: lags = 400 needs a lag design of"), msg
    assert "past the work budget of 2^35 multiply-adds" in msg, msg
    best = int(re.search(r"pass lags <= (\d+), or a shorter series$", msg).group(1))
    assert best > math.ceil(12 * (n / 100) ** 0.25)
    assert _za_scan(n, 5, 2100, best)[1] <= LAG_WORK < _za_scan(n, 5, 2100, best + 1)[1]
    assert dt < 5.0


def test_zivot_andrews_names_the_keyword_that_set_an_out_of_window_lag():
    # n = 50, trim 0.15: the break regressions need lags <= int(7.5) - 1 = 6.
    y = _walk(50, 3)
    msg, _ = _quick(tsecon.zivot_andrews, y, autolag=None, lags=7)
    assert msg.startswith("zivot_andrews: lags = 7 is invalid for a series of length 50"), msg
    assert msg.endswith("; pass lags <= 6"), msg
    # The truncated-Schwert default int(12 (0.5)^(1/4)) = 10 names the default.
    msg, _ = _quick(tsecon.zivot_andrews, y, autolag=None)
    assert msg.startswith("zivot_andrews: the default lags = 10"), msg
    assert msg.endswith("; pass lags <= 6"), msg
    # A searched lag outside the window names max_lags, the keyword that
    # admitted it: an AR(12) in the differences makes the t-stat search keep
    # its cap.
    e = np.random.default_rng(0).standard_normal(400)
    d = np.zeros(400)
    for t in range(12, 400):
        d[t] = 0.85 * d[t - 12] + e[t]
    msg, _ = _quick(tsecon.zivot_andrews, np.cumsum(d)[:80], autolag="t-stat", max_lags=12)
    assert msg.startswith(
        "zivot_andrews: lags = 12, chosen by the lag search capped at max_lags = 12, "
        "is invalid for a series of length 80"
    ), msg
    assert msg.endswith("; pass max_lags <= 11"), msg


@pytest.mark.parametrize("call, kw", [("check_stationarity", {}), ("ndiffs", {"test": "adf"})])
def test_stationarity_workflows_explain_a_default_lag_refusal(call, kw):
    # Their ADF leg runs the default search and they take no lag keyword:
    # past the memory budget (n = 7e5) the refusal says so.
    y = _walk(700_000, 25)
    msg, dt = _quick(getattr(tsecon, call), y, **kw)
    assert msg.startswith(f"{call}: adf: the default maxlag = "), msg
    assert msg.endswith(
        f"{call} itself takes no lag keyword (its ADF leg runs the default lag "
        "search), so supply a shorter series, or call tsecon.adf with an explicit maxlag"
    ), msg
    assert dt < 5.0


# --------------------------------------------------- B1/B2: keyword-named refusals


def test_arch_lm_names_nlags():
    e = np.random.default_rng(0).standard_normal(500)
    for nlags in (250, 600, 2**31):
        msg, _ = _quick(tsecon.arch_lm, e, nlags=nlags)
        assert msg.startswith(f"arch_lm: nlags = {nlags} is too long for a series of length 500"), msg
        assert msg.endswith("pass nlags <= 249"), msg
    assert tsecon.arch_lm(e, nlags=249)["df"] == 249


@pytest.mark.parametrize("fn", [tsecon.dfgls, tsecon.ng_perron])
def test_gls_unit_root_tests_name_lags_and_max_lags(fn):
    y = _walk(200)
    msg, _ = _quick(fn, y, lags=2**31)
    assert msg.startswith(f"{fn.__name__}: lags = 2147483648 is too long for a series of length 200"), msg
    assert msg.endswith("pass lags <= 98"), msg
    msg, _ = _quick(fn, y, max_lags=2**31)
    assert msg.startswith(f"{fn.__name__}: max_lags = 2147483648 is too long"), msg
    assert msg.endswith("pass max_lags <= 98"), msg
    assert fn(y, lags=98)["used_lag"] == 98


def test_adf_names_maxlag():
    y = _ar1(200)
    for autolag in (None, "aic"):
        msg, _ = _quick(tsecon.adf, y, maxlag=2**31, autolag=autolag)
        assert msg.startswith("adf: maxlag = 2147483648 is too long for a series of length 200"), msg
        assert msg.endswith("pass maxlag <= 98"), msg
    # regression "n" on an even n: the statsmodels bound n//2 - 1 = 99 leaves
    # no residual degree of freedom; it used to read "series has 200
    # observations but needs at least 201".
    msg, _ = _quick(tsecon.adf, y, regression="n", maxlag=99, autolag=None)
    assert msg.endswith("pass maxlag <= 98"), msg


def test_zivot_andrews_names_max_lags_and_lags():
    y = _walk(200)
    msg, _ = _quick(tsecon.zivot_andrews, y, max_lags=2**31)
    assert msg.startswith("zivot_andrews: max_lags = 2147483648 is too long for a series of length 200"), msg
    assert msg.endswith("pass max_lags <= 97"), msg
    msg, _ = _quick(tsecon.zivot_andrews, y, autolag=None, lags=40)
    assert msg.startswith("zivot_andrews: lags = 40 is invalid for a series of length 200"), msg


def test_panel_unit_root_names_its_keywords():
    pout = np.array([_ar1(40, i) for i in range(5)]) + np.arange(5)[:, None]
    msg, _ = _quick(tsecon.panel_unit_root, pout, max_lags=2**31)
    assert msg.startswith("per-unit ADF failed for unit 0: adf: max_lags = 2147483648 is too long"), msg
    msg, _ = _quick(tsecon.panel_unit_root, pout, lags=2**31)
    assert msg.startswith("per-unit ADF failed for unit 0: adf: lags = 2147483648 is too long"), msg
    for bw in (-1.0, float("nan"), float("inf")):
        msg, _ = _quick(tsecon.panel_unit_root, pout, test="llc", lags=1, lrv_bandwidth=bw)
        assert re.match(r"lrv_bandwidth = \S+ is invalid: requires a finite value >= 0", msg), msg
    assert tsecon.panel_unit_root(pout, test="llc", lags=1, lrv_bandwidth=2.0)["n_units"] == 5


@pytest.mark.parametrize("call", ["stl", "seasonal_strength", "nsdiffs"])
def test_stl_period_refusals_name_period(call):
    y = _seasonal()
    kw = {"max_d": 1} if call == "nsdiffs" else {}
    for period in (49, 60, 2**31):
        msg, _ = _quick(getattr(tsecon, call), y, period=period, **kw)
        assert msg.startswith(
            f"stl: period = {period} needs at least 2 * period = {2 * period} observations "
            "but the series has 96"
        ), msg
        assert msg.endswith("pass period <= 48, or supply a longer series"), msg


def test_forecast_comparisons_name_lrv_lags():
    rng = np.random.default_rng(2)
    a, b, c, d = (rng.standard_normal(200) for _ in range(4))
    for fn, args in ((tsecon.cw_test, (a, b, c, d)), (tsecon.gw_test, (a**2, b**2))):
        msg, _ = _quick(fn, *args, lrv_lags=2**31)
        assert "lrv_lags = 2147483648 (the Bartlett long-run-variance lag truncation)" in msg, msg
        assert msg.endswith("pass lrv_lags <= 199"), msg


# ------------------------------------------------------ B5: check_series counts


@pytest.mark.parametrize("lags", [-1, 0, 94, 2**62])
def test_check_series_1d_lags_out_of_range_is_refused(lags):
    y = _seasonal()
    with pytest.raises(ValueError, match=rf"check_series: lags={lags} is out of range: for 1D input"):
        tsecon.check_series(y, lags=lags)


@pytest.mark.parametrize("lags", [2.7, 5.0, "5"])
def test_check_series_lags_must_be_an_integer(lags):
    with pytest.raises(TypeError, match=r"check_series: lags=.* but this parameter takes an integer"):
        tsecon.check_series(_seasonal(), lags=lags)


def test_check_series_1d_lags_at_the_bound_runs():
    y = _seasonal()
    rep = tsecon.check_series(y, lags=93)  # n - 3
    lb = rep["serial_correlation"]["ljung_box"]["lags"]
    assert int(lb[-1]) == 93
    rep = tsecon.check_series(y, lags=np.int64(4))
    assert int(rep["serial_correlation"]["ljung_box"]["lags"][-1]) == 4


def test_check_series_2d_lags_is_a_cap_computed_in_closed_form():
    rng = np.random.default_rng(5)
    data = rng.standard_normal((120, 2))
    t0 = time.perf_counter()
    rep = tsecon.check_series(data, lags=2**62)
    assert time.perf_counter() - t0 < 30.0
    # The largest p with 120 - p - (2p + 1) >= 10 is 36.
    assert rep["var_lag_selection"]["lags_tried"][-1] == 36
    capped = tsecon.check_series(data, lags=3)
    assert capped["var_lag_selection"]["lags_tried"] == [1, 2, 3]
    for bad in (-1, 0):
        with pytest.raises(ValueError, match=rf"lags={bad} is out of range: for 2D input"):
            tsecon.check_series(data, lags=bad)


def _broken():
    # A mean shift on [100, 200) that the battery's sup-F scan rejects
    # (p = 0.02 on the detrended level), so bai_perron runs.
    t = np.arange(300)
    return np.random.default_rng(1).standard_normal(300) + np.where((t >= 100) & (t < 200), 0.6, 0.0)


@pytest.mark.parametrize("max_breaks", [-1, 0, 11])
@pytest.mark.parametrize("series", ["quiet", "broken"])
def test_check_series_max_breaks_is_refused_whatever_the_data(max_breaks, series):
    y = np.random.default_rng(0).standard_normal(300) if series == "quiet" else _broken()
    with pytest.raises(ValueError, match=rf"check_series: max_breaks={max_breaks} is out of range"):
        tsecon.check_series(y, max_breaks=max_breaks)


def test_check_series_max_breaks_type_and_valid_value():
    with pytest.raises(TypeError, match=r"max_breaks=2.0 is of type float"):
        tsecon.check_series(_broken(), max_breaks=2.0)
    rep = tsecon.check_series(_broken(), max_breaks=2)
    assert rep["breaks"]["sup_f"]["rejected"]
    assert rep["breaks"]["bai_perron"]["n_breaks"] <= 2


# --------------------------------------------------------- C1: dsge_solve doc


def test_dsge_solve_docstring_says_non_unique_regimes_raise():
    doc = tsecon.dsge_solve.__doc__
    assert "unique / indeterminate / no stable solution" not in doc
    assert "raises ValueError" in doc
    a = np.eye(2)
    b = np.array([[0.9, 0.1], [0.2, 1.5]])
    c = np.array([[1.0], [0.0]])
    assert tsecon.dsge_solve(a, b, c, 1)["verdict"].startswith("unique stable solution")
    for n_pre, regime in ((0, "indeterminate"), (2, "no stable solution")):
        with pytest.raises(ValueError, match=f"Blanchard-Kahn: {regime}"):
            tsecon.dsge_solve(a, b, c, n_pre)
