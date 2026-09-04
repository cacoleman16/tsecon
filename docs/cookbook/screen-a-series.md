# Screen a new series before you model it

Someone hands you a column of numbers. Before any model, you want the same
opening moves a careful practitioner makes: is it stationary, is it
autocorrelated, does the volatility cluster, are the tails fat, did something
break, is it seasonal? `tsecon.check_series` runs all of that in one call and
ends with an **ordered list of recommendations** that name the functions to
call next.

## The recipe

```python
import numpy as np, tsecon

rng = np.random.default_rng(12)                       # 30 years of monthly data
n = 360
season = 2.0 * np.sin(2 * np.pi * np.arange(n) / 12)
y = np.cumsum(0.02 + rng.standard_normal(n)) + season  # drift + unit root + seasonality

rep = tsecon.check_series(y, seasonal_period=12)

print("stationarity :", rep["stationarity"]["quadrant"], "->",
      rep["stationarity"]["recommendation"])
print("tests ran on :", rep["analysis_scale"]["scale"])
print("Ljung-Box p  :", round(rep["serial_correlation"]["ljung_box"]["lb_pvalue"][-1], 4))
print("ARCH-LM p    :", round(rep["arch_effects"]["p_value"], 4))
print("Jarque-Bera p:", round(rep["normality"]["p_value"], 4))
print("sup-F break  :", rep["breaks"]["sup_f"]["rejected"])
print("expected false alarms:", rep["multiple_testing"]["expected_false_rejections"])
```

```
stationarity : UnitRoot -> Difference
tests ran on : first_difference
Ljung-Box p  : 0.0
ARCH-LM p    : 0.8599
Jarque-Bera p: 0.38
sup-F break  : False
expected false alarms: 0.25
```

## Reading it

**`analysis_scale` is the line to read second.** The ADF+KPSS quadrant decides
which object every downstream test sees: a `Difference` verdict moves the
dependence, ARCH, normality, and break families onto `np.diff(y)`. Every family
also carries its own `computed_on`, so a p-value is never ambiguous about which
series it describes — that ambiguity is the single most common way a diagnostic
battery misleads.

**`multiple_testing` is the line most batteries omit.** Six tests ran; five of
them have nulls that actually hold here, so about 0.25 spurious rejections are
expected by construction. `tsecon` reports that arithmetic instead of silently
applying a correction you did not ask for.

## The to-do list

The report ends in `recommendations`, ordered, each with the finding, the
suggested next call, and the functions that implement it:

```python
for i, rec in enumerate(rep["recommendations"], 1):     # the ordered to-do list
    print(f"{i}. [{rec['topic']}] {rec['finding']}")
    print(f"   -> {rec['suggestion']}")
    print(f"   functions: {', '.join(rec['functions'])}")
```

```
1. [unit_root] quadrant 'UnitRoot': the tests agree the series looks I(1)
   -> Model in first differences — arima_fit(y, d=1, ...) differences internally — and re-run check_series on the differences to pick the short-run orders.
   functions: check_stationarity, arima_fit
2. [persistent_regressor] the series is highly persistent
   -> If this series will be a REGRESSOR (e.g. predicting returns), standard t-tests are size-distorted under persistence: use predictive_regression (Stambaugh + IVX) for one predictor or ivx_test for a joint test.
   functions: predictive_regression, ivx_test
3. [arma_orders] Ljung-Box rejects whiteness on the first_difference (p=2.97e-31)
   -> Start from arima_fit(y, p=1, d=1, q=0) and compare AIC/BIC over a small order grid — the ACF/PACF cutoff heuristics are starting points, not a verdict.
   functions: arima_fit, acf, pacf
4. [seasonality] seasonal evidence reported at period 12
   -> Plain speech: model it directly with arima_fit(..., seasonal=(P, D, Q, s)), letting nsdiffs pick D first; or decompose with mstl (multiple periods) or stl (one) and model the seasonally adjusted series. The one real gap is X-13ARIMA-SEATS, which needs the external Census binary and so stays out of scope — deseasonalize elsewhere if you need X-13 specifically.
   functions: acf, periodogram, nsdiffs, arima_fit, mstl, stl
```

All four findings are real properties of the simulated series, and note that
recommendation 4 names the one thing the library *cannot* do — X-13ARIMA-SEATS,
which needs the external Census binary — while routing you to what it can. That
is deliberate: a battery that only advertises its own coverage is not a
screening report.

## Gotchas

- **Pass `seasonal_period` when you know it** (12 monthly, 4 quarterly).
  Without it you get a heuristic `detected_period` — the argmax periodogram
  ordinate — which is not a test.
- **Hand it a 2-D `(n, k)` array** and it switches to the multivariate battery:
  per-series integration verdicts, Johansen cointegration, and VAR lag
  selection with a stability check.
- **It is a screening report, not a verdict.** Near the unit circle, stationary
  and integrated processes are nearly observationally equivalent in finite
  samples; no battery can manufacture information the data do not contain.
- Want the whole thing rendered instead of walked?
  `tsecon.results.check_series(y).summary()` returns the same dict plus a
  printable report and `.plot_diagnostics()`.

## See also

- Model card: [check_series, the one-call diagnostic battery](../reference/model-cards/check-series.md)
- Guide: [1. Thinking in Time Series](../guide/01-foundations.md) ·
  [2. Exploring and Diagnosing](../guide/02-exploration-and-diagnostics.md)
- [Which model when?](../which-model-when.md) — this battery is that page's
  decision table, executable
- Recipes: [Test whether a series has a unit root](unit-root-test.md) ·
  [Model fat-tailed volatility with GARCH](garch-fat-tails.md)
