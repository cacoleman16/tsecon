"""Regression pins for audit round 16, forecasting and volatility group.

* A3 — a forecast horizon of 2**31 or 2**40 aborted the interpreter in
  ``garch_fit``, ``ccc_garch``, ``dcc_garch``, ``gas_volatility``,
  ``theta_forecast`` and ``conformal_forecast(method="enbpi")``: each is now
  capped at 1,000,000 steps (as ``arima_fit`` and ``ets_fit`` are), and the
  multivariate forecasts additionally at a 2^28-value working set for their
  ``(horizon, k, k)`` nested lists, refused by name before anything is
  fitted.
* ``arima_fit`` refuses absurd ARMA orders before anything is allocated
  (they ran for hours or exhausted memory), naming p, q and seasonal.
* A4 (share) — ``conformal_forecast`` / ``conformal_backtest`` with
  ``method="enbpi"`` size the bootstrap ensemble before any stream is drawn:
  ``n_boot=2**31`` aborted the process and ``2**40`` gave an unnamed RNG
  error.
* B1 / B2 (share) — refusals that named no keyword or an engine-internal one
  now name the Python keyword and its largest admissible value:
  ``conformal_*`` ``calib`` / ``n_eval`` / ``lags``, ``backtest``
  ``insample_period``, ``dcc_test`` ``lags``, ``auto_arima``
  ``seasonal_period`` / ``D`` / ``max_P`` / ``max_Q``.
* H16-14 — the too-few-observations refusal of a seasonal ARIMA names the
  seasonal specification instead of "Lower p, d, or q".

Every refusal here comes before any allocation, so each is exercised in
process. The repro scripts are under the audit's abort-sweep directory.
"""
import re

import numpy as np
import pytest

import tsecon

CAP = 1_000_000
BIG = [CAP + 1, 2**31, 2**40]


def _garch_returns(n=300, k=1, seed=0):
    rng = np.random.default_rng(seed)
    out = np.zeros((n, k))
    s2 = np.ones(k)
    eps = np.zeros(k)
    for t in range(n):
        s2 = 0.1 + 0.1 * eps**2 + 0.8 * s2
        eps = np.sqrt(s2) * rng.standard_normal(k)
        out[t] = eps
    return out[:, 0] if k == 1 else out


def _pos(n=200, seed=1):
    rng = np.random.default_rng(seed)
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = 0.6 * y[t - 1] + rng.standard_normal()
    return 10.0 + y


def _seasonal(n=96, period=12, seed=2):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return 10.0 + 3.0 * np.sin(2 * np.pi * t / period) + 0.5 * rng.standard_normal(n)


# --------------------------------------------------------------- A3: horizons


@pytest.mark.parametrize("h", BIG)
def test_garch_fit_forecast_horizon_capped(h):
    with pytest.raises(ValueError, match=r"forecast_horizon = \d+.*forecast_horizon <= 1000000"):
        tsecon.garch_fit(_garch_returns(), forecast_horizon=h)


def test_garch_fit_forecast_horizon_at_cap_runs():
    r = tsecon.garch_fit(_garch_returns(), forecast_horizon=CAP)
    v = r["variance_forecast"]
    assert v.shape == (CAP,)
    omega, alpha, beta = r["params"]
    assert v[-1] == pytest.approx(omega / (1.0 - alpha - beta), rel=1e-9)


@pytest.mark.parametrize("h", BIG)
def test_gas_volatility_horizon_capped(h):
    with pytest.raises(ValueError, match=r"horizon = \d+.*horizon <= 1000000"):
        tsecon.gas_volatility(_garch_returns(), horizon=h)


def test_gas_volatility_horizon_at_cap_runs():
    r = tsecon.gas_volatility(_garch_returns(), horizon=CAP)
    f = r["forecast"]
    assert f.shape == (CAP,)
    # The GAS projection converges to omega / (1 - b).
    assert f[-1] == pytest.approx(r["omega"] / (1.0 - r["b"]), rel=1e-9)


@pytest.mark.parametrize("h", BIG)
def test_theta_forecast_steps_capped(h):
    with pytest.raises(ValueError, match=r"Theta method: steps = \d+ is too large.*steps can be at most 1000000"):
        tsecon.theta_forecast(_seasonal(), steps=h, period=12)


def test_theta_forecast_at_cap_runs():
    assert tsecon.theta_forecast(_pos(), steps=CAP).shape == (CAP,)


@pytest.mark.parametrize("h", BIG)
def test_conformal_enbpi_horizon_capped(h):
    with pytest.raises(ValueError, match=r"EnbPI: horizon = \d+ is too large.*horizon can be at most 1000000"):
        tsecon.conformal_forecast(_pos(), method="enbpi", base="ar", lags=2, n_boot=10, seed=1, horizon=h)


@pytest.mark.parametrize("h", BIG)
@pytest.mark.parametrize("fn", ["ccc_garch", "dcc_garch"])
def test_multivariate_forecast_horizon_capped(fn, h):
    ret = _garch_returns(k=2)
    with pytest.raises(ValueError, match=r"forecast_horizon = \d+ is too large for a forecast of 2 series.*at most 1000000"):
        getattr(tsecon, fn)(ret, forecast_horizon=h)


def _mgarch_per_step(paths, k):
    """The documented per-step working set (tsecon-mgarch forecast.rs)."""
    mat = 8 * -(-k // 8) * k + 8
    return paths * (mat + 5 * k * k + 11 * k) + 6 * k + 11


@pytest.mark.parametrize("fn,paths", [("ccc_garch", 1), ("dcc_garch", 2)])
def test_multivariate_forecast_budget_scales_with_k(fn, paths):
    # Ten series: the (horizon, k, k) nested lists bind before the step cap.
    # The check runs before the fit, so random data suffices.
    ret = np.random.default_rng(3).standard_normal((300, 10))
    max_h = min(CAP, (1 << 28) // _mgarch_per_step(paths, 10))
    assert max_h < CAP
    with pytest.raises(ValueError, match=rf"with k = 10 series forecast_horizon can be at most {max_h}$"):
        getattr(tsecon, fn)(ret, forecast_horizon=max_h + 1)


def test_multivariate_forecast_below_cap_still_runs():
    ret = _garch_returns(k=2)
    r = tsecon.dcc_garch(ret, forecast_horizon=50)
    assert np.asarray(r["correlation_forecast"]).shape == (50, 2, 2)
    r = tsecon.ccc_garch(ret, forecast_horizon=50)
    assert np.asarray(r["covariance_forecast"]).shape == (50, 2, 2)


# --------------------------------------------------- arima_fit order budget


def _ar1(n=200, seed=4):
    rng = np.random.default_rng(seed)
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = 0.5 * y[t - 1] + rng.standard_normal()
    return y


@pytest.mark.parametrize(
    "kw",
    [
        dict(p=1, seasonal=(1, 0, 0, 199)),
        dict(p=1, seasonal=(50, 0, 0, 4)),
        dict(p=150),
        dict(p=0, q=150),
    ],
)
def test_arima_fit_absurd_orders_refused_by_name(kw):
    with pytest.raises(ValueError, match=r"^the ARIMA orders \(p, q and seasonal=\(P, D, Q, s\)\) are too large to estimate") as exc:
        tsecon.arima_fit(_ar1(), **kw)
    msg = str(exc.value)
    assert "lower p or q" in msg
    assert "the state dimension can be at most" in msg


def test_arima_fit_memory_budget_refused_by_name():
    # p = 600 on 1000 observations: 1.08e9 filter values per evaluation.
    with pytest.raises(ValueError, match=r"are too large to estimate.*memory budget of 2\^28 values"):
        tsecon.arima_fit(_ar1(1000), p=600)


def test_arima_fit_ordinary_orders_unaffected():
    r = tsecon.arima_fit(_ar1(), p=2, q=1)
    assert len(r["params"]) == 5 and np.isfinite(r["loglik"])


def test_arima_conformal_base_reports_the_budget():
    # The "arima" conformal base fits through the same engine; the refusal
    # names the conformal keyword, order, ahead of the engine's own text.
    with pytest.raises(ValueError, match=r"base=\"arima\": order = \(150, 0, 0\) is too large to estimate.*are too large to estimate"):
        tsecon.conformal_forecast(_pos(), base="arima", order=(150, 0, 0), calib=20)


# -------------------------------------------------- H16-14: seasonal remedy


def test_css_seasonal_refusal_names_seasonal():
    with pytest.raises(ValueError) as exc:
        tsecon.arima_fit(_ar1(), p=1, seasonal=(1, 0, 0, 199), method="css")
    msg = str(exc.value)
    assert "needs at least 205 observations" in msg
    assert "seasonal=(P, D, Q, s)" in msg and "Lower p, d, or q" not in msg
    with pytest.raises(ValueError, match=r"seasonal=\(P, D, Q, s\)"):
        tsecon.arima_fit(_ar1(), p=1, seasonal=(50, 0, 0, 4), method="css")


def test_seasonal_differencing_refusal_names_seasonal():
    with pytest.raises(ValueError) as exc:
        tsecon.arima_fit(_ar1(), p=0, seasonal=(0, 3, 0, 70))
    msg = str(exc.value)
    assert "seasonal differencing order D" in msg and "seasonal=(P, D, Q, s)" in msg


def test_non_seasonal_short_sample_wording_unchanged():
    with pytest.raises(ValueError, match=r"Lower p, d, or q, or supply a longer series\.$"):
        tsecon.arima_fit(_ar1()[:3], p=1, method="css")


# ----------------------------------------------------- A4 share: EnbPI n_boot


def _enbpi_max_boot(n, lags, reserve, steps):
    """The documented EnbPI budget (tsecon-forecast conformal.rs enbpi_cost)."""
    rows = n - reserve - lags
    l = lags + 1
    work = rows * l * (l + 1) // 2 + l**3 + rows * l + steps * (rows + l)
    values = -(-rows // 8) + l + 18
    fixed = 4 * rows + l * l + 4 * steps
    return min((1 << 34) // work, ((1 << 28) - fixed) // values)


@pytest.mark.parametrize("n_boot", [2**31, 2**40])
def test_conformal_forecast_enbpi_n_boot_budget(n_boot):
    max_b = _enbpi_max_boot(200, 2, 0, 1)
    with pytest.raises(ValueError, match=rf"EnbPI: n_boot = {n_boot} is too large.*n_boot can be at most {max_b} here"):
        tsecon.conformal_forecast(_pos(), method="enbpi", base="ar", lags=2, n_boot=n_boot, seed=1)
    with pytest.raises(ValueError, match=rf"n_boot can be at most {max_b} here"):
        tsecon.conformal_forecast(_pos(), method="enbpi", base="ar", lags=2, n_boot=max_b + 1, seed=1)


@pytest.mark.parametrize("n_boot", [2**31, 2**40])
def test_conformal_backtest_enbpi_n_boot_budget(n_boot):
    max_b = _enbpi_max_boot(200, 2, 20, 20)
    with pytest.raises(ValueError, match=rf"EnbPI: n_boot = {n_boot} is too large.*n_boot can be at most {max_b} here"):
        tsecon.conformal_backtest(_pos(), method="enbpi", base="ar", lags=2, n_boot=n_boot, seed=1, n_eval=20)


def test_enbpi_long_horizon_names_n_boot_or_horizon():
    # At the step cap the prediction steps dominate: 1e6 steps * 199 rows
    # per model leaves room for a few dozen models...
    y = _pos(200)
    max_b = _enbpi_max_boot(200, 1, 0, CAP)
    assert 2 <= max_b < 200
    with pytest.raises(ValueError, match=rf"n_boot = 200 is too large.*n_boot can be at most {max_b} here"):
        tsecon.conformal_forecast(y, method="enbpi", base="ar", n_boot=200, horizon=CAP)
    # ...and on a long series not even two models fit: the refusal names
    # the horizon and the longest one that does.
    y = _pos(20_002)
    rows, l = 20_001, 2
    base_work = rows * l * (l + 1) // 2 + l**3 + rows * l
    max_h = ((1 << 34) // 25 - base_work) // (rows + l)
    with pytest.raises(ValueError, match=rf"EnbPI: horizon = {CAP} is too large.*horizon can be at most {max_h} here"):
        tsecon.conformal_forecast(y, method="enbpi", base="ar", n_boot=25, horizon=CAP)


def test_enbpi_ordinary_call_unaffected():
    r = tsecon.conformal_forecast(_pos(), method="enbpi", base="ar", lags=2, n_boot=30, seed=1, horizon=3)
    assert r["n_boot"] == 30 and len(r["mean"]) == 3


# ------------------------------------------------ B1 / B2 share: keywords


@pytest.mark.parametrize(
    "call,pattern",
    [
        (lambda y: tsecon.conformal_backtest(y, horizon=1, calib=2**31, n_eval=20),
         r"split conformal \(online\): calib = 2147483648 is too large.*calib can be at most 178 here"),
        (lambda y: tsecon.conformal_backtest(y, horizon=1, calib=30, n_eval=2**31),
         r"split conformal \(online\): n_eval = 2147483648 is too large.*n_eval can be at most 168 here"),
        (lambda y: tsecon.conformal_backtest(y, method="enbpi", base="ar", lags=2**31, n_boot=10, seed=1, n_eval=20),
         r"EnbPI: lags = 2147483648 is too large.*lags can be at most 89 here"),
        (lambda y: tsecon.conformal_backtest(y, method="enbpi", base="ar", n_boot=10, seed=1, n_eval=2**31),
         r"EnbPI: n_eval = 2147483648 is too large.*n_eval can be at most 196 here"),
        (lambda y: tsecon.conformal_forecast(y, horizon=2, calib=2**31),
         r"split conformal: calib = 2147483648 is too large.*calib can be at most 197 here"),
        (lambda y: tsecon.conformal_forecast(y, method="aci", n_eval=2**31, calib=30),
         r"adaptive conformal inference: n_eval = 2147483648 is too large.*n_eval can be at most 168 here"),
        (lambda y: tsecon.conformal_forecast(y, base="ar", lags=2**31, calib=30),
         r"AR least-squares forecast: lags = 2147483648 is too large.*training window has 170.*lags can be at most 84 here"),
        (lambda y: tsecon.conformal_forecast(y, method="enbpi", base="ar", lags=2**31, n_boot=10, seed=1),
         r"EnbPI: lags = 2147483648 is too large.*lags can be at most 99 here"),
        (lambda y: tsecon.conformal_forecast(y, horizon=300),
         r"split conformal: horizon = 300 is too large.*horizon can be at most 149 here"),
    ],
)
def test_conformal_size_refusals_name_the_keyword(call, pattern):
    with pytest.raises(ValueError, match=pattern) as exc:
        call(_pos())
    assert "seasonal period" not in str(exc.value)


def test_backtest_insample_period_named():
    y = _pos(200)
    with pytest.raises(ValueError, match=r"backtest: insample_period = 2147483648 is too large.*insample_period can be at most 99 here"):
        tsecon.backtest(y, train=100, horizon=2, refit_every=5, insample_period=2**31)
    with pytest.raises(ValueError, match=r"backtest: insample_period = 0 is invalid"):
        tsecon.backtest(y, train=100, horizon=2, refit_every=5, insample_period=0)
    # The largest admissible period still runs.
    r = tsecon.backtest(y, train=100, horizon=2, refit_every=5, insample_period=99)
    assert r["n_origins"] == 99


def test_dcc_test_lags_named():
    # Two series, one pair: (500 - lags) stacked rows must leave one residual
    # degree of freedom for the lags + 1 coefficients, so lags <= 249.
    ret = _garch_returns(500, k=2)
    with pytest.raises(ValueError, match=r"lags = 2147483648 is invalid for 500 observations: .*\(T - lags\) \* k\(k-1\)/2 >= lags \+ 2.*lags can be at most 249 here"):
        tsecon.dcc_test(ret, lags=2**31)
    for lags in (250, 498):
        # 498 used to pass the size check and fail as a singular X'X.
        with pytest.raises(ValueError, match=rf"lags = {lags} is invalid.*at most 249 here"):
            tsecon.dcc_test(ret, lags=lags)
    r = tsecon.dcc_test(ret, lags=249)
    assert (r["df"], r["n_stacked"]) == (250, 251)
    assert np.isfinite(r["stat"]) and 0.0 <= r["p_value"] <= 1.0


def test_auto_arima_seasonal_period_and_D_named():
    with pytest.raises(ValueError, match=r"seasonal_period = 2147483648 is too long for a series of 200 observations.*seasonal_period can be at most 100 here"):
        tsecon.auto_arima(_ar1(), max_p=2, max_q=2, forecast_steps=2, seasonal_period=2**31)
    with pytest.raises(ValueError, match=r"D = 2147483648 seasonal differences at seasonal_period = 12.*D can be at most 7 here"):
        tsecon.auto_arima(_seasonal(), seasonal_period=12, max_p=1, max_q=1, max_P=1, max_Q=1, D=2**31)
    with pytest.raises(ValueError, match=r"seasonal_period = 96 is not shorter than the series"):
        tsecon.auto_arima(_seasonal(), seasonal_period=96, D=1)


@pytest.mark.parametrize("kw,name", [(dict(max_P=2**31), "max_P"), (dict(max_Q=2**31), "max_Q"), (dict(max_P=7), "max_P")])
def test_auto_arima_seasonal_caps_use_python_names(kw, name):
    with pytest.raises(ValueError, match=rf"^{name} = \d+ is beyond the cap of 6.*{name} can be at most 6$") as exc:
        tsecon.auto_arima(_seasonal(), seasonal_period=12, max_p=1, max_q=1, **kw)
    assert "max_seasonal" not in str(exc.value)


# ------------------------------------------- review: residual keyword gaps


@pytest.mark.parametrize(
    "call",
    [
        lambda y: tsecon.conformal_forecast(y, horizon=0),
        lambda y: tsecon.conformal_forecast(y, method="aci", horizon=0),
        lambda y: tsecon.conformal_forecast(y, method="enbpi", base="ar", horizon=0),
        lambda y: tsecon.conformal_backtest(y, horizon=0),
    ],
)
def test_conformal_zero_horizon_names_horizon(call):
    # The shared step check used to report the conformal horizon as "steps".
    with pytest.raises(ValueError, match=r"^horizon = 0 is invalid"):
        call(_pos())


def test_theta_zero_steps_still_names_steps():
    with pytest.raises(ValueError, match=r"^steps = 0 is invalid"):
        tsecon.theta_forecast(_pos(), steps=0)


@pytest.mark.parametrize("s", [96, 97, 1000])
def test_auto_arima_fixed_D0_period_the_series_never_completes(s):
    # D = 0 fixed and a period the 96 observations never complete: no
    # seasonal lag is in sample. It used to run every seasonal candidate
    # (state dimension >= s + 1) and was still searching after 290 s.
    with pytest.raises(ValueError, match=rf"seasonal_period = {s} is not shorter than the series \(96 observations\).*seasonal_period can be at most 95 here"):
        tsecon.auto_arima(_seasonal(), seasonal_period=s, D=0, max_p=1, max_q=1)


@pytest.mark.parametrize(
    "call,pattern,ok",
    [
        # The grid fits, but its first training window (200 - 2 - 197 + 1 =
        # 2 observations) starves the default Theta base, which needs 4.
        (lambda y, v: tsecon.conformal_forecast(y, horizon=2, calib=v),
         r"split conformal: calib = 197 is too large: the grid's first training window holds n - horizon - calib \+ 1 = 2 observations, but the base forecaster \(Theta method\) needs at least 4, so calib can be at most 195 here",
         195),
        (lambda y, v: tsecon.conformal_backtest(y, horizon=1, calib=30, n_eval=v),
         r"split conformal \(online\): n_eval = 168 is too large: .*holds n - horizon - calib - n_eval \+ 1 = 2 observations.*n_eval can be at most 166 here",
         166),
        (lambda y, v: tsecon.conformal_forecast(y, method="aci", n_eval=v, calib=30),
         r"adaptive conformal inference: n_eval = 168 is too large.*n_eval can be at most 166 here",
         166),
        # The grid bound quoted for horizon=300 (149) starves the base next.
        (lambda y, v: tsecon.conformal_forecast(y, horizon=v),
         r"split conformal: horizon = 149 is too large.*horizon can be at most 147 here",
         147),
    ],
)
def test_conformal_starved_base_window_names_the_grid_keyword(call, pattern, ok):
    m = re.search(r"= (\d+) is too large", pattern)
    with pytest.raises(ValueError, match=pattern) as exc:
        call(_pos(), int(m.group(1)))
    assert "seasonal period" not in str(exc.value)
    # The quoted maximum runs.
    call(_pos(), ok)
