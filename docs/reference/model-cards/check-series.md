# Model card — check_series, the one-call diagnostic battery

**Family:** `check_series` (and the rendering facade
`tsecon.results.check_series`, which returns the same dict plus `.summary()`
and `.plot_diagnostics()`)

The first hour with a series, in one call. `check_series` runs the library's
diagnostic families — descriptives and an outlier screen, the ADF+KPSS
confirmatory quadrant, Ljung-Box/ACF/PACF serial correlation, ARCH-LM,
Jarque-Bera, an intercept-only sup-F / Bai-Perron mean-shift scan, GPH long
memory, and seasonality evidence — and ends in an ordered list of
**recommendations** that route to concrete tsecon calls. Handed a 2D `(n, k)`
panel, it switches to the multivariate battery: per-series integration
verdicts, Johansen cointegration, and VAR lag selection with a stability
check. It is a pure-Python composition layer over individually validated
compiled primitives; there is no new statistics in it, only the workflow.

## What it is — and what it is not

It is a **screening report**: the same opening moves a careful practitioner
makes, run in the right order, on the right transformation of the data, with
the assumptions and caveats attached to each verdict. It is **not a substitute
for judgment**. Near the unit circle, stationary and integrated processes are
nearly observationally equivalent in finite samples (Cochrane, 1991) — no
battery, however complete, can manufacture the information the data do not
contain. Every recommendation is a *starting point* that names the follow-up
functions and the assumption that could overturn it; the intended workflow is
run the battery, read the caveats, then interrogate the branches yourself.
It inspects one dataset — it cannot know your research question.

## Key arguments and defaults

| Argument | Default | Notes |
|---|---|---|
| `data` | — | 1D array → univariate battery; 2D `(n, k)`, k ≥ 2 → multivariate (an `(n, 1)` column squeezes to univariate). Plain lists are coerced. |
| `seasonal_period` | `None` | A *known* period `s` (12 for monthly, 4 for quarterly) unlocks real seasonal evidence: the ACF at lags `s, 2s, 3s` plus the periodogram ordinate at frequency `1/s`. Without it you only get a heuristic `detected_period`. |
| `lags` | `None` | Ljung-Box horizon; `None` → `min(10, n_analysis // 5)`. |
| `alpha` | `0.05` | One significance level for every family. Must lie in `(0.01, 0.10]` — the KPSS p-value is interpolated from a table clamped to `[0.01, 0.10]`, so anything outside that range could never be served honestly (a teaching error explains this if you try). |
| `max_breaks` | `5` | Passed to `bai_perron` when the break scan escalates. |
| `trim` | `0.15` | Trimming fraction for the sup-F search and Bai-Perron segments. |

## How to read the report

**Walk the families.** The report is a plain JSON-serializable dict, one key
per family (`descriptives`, `outliers`, `stationarity`, `analysis_scale`,
`serial_correlation`, `arch_effects`, `normality`, `breaks`, `long_memory`,
`seasonality`; multivariate: `per_series`, `integration_summary`,
`cointegration`, `var_lag_selection`, `stability`), then `tests_run`,
`multiple_testing`, and `recommendations`. The `.summary()` render walks them
in the same order.

**The `analysis_scale` logic.** The ADF+KPSS quadrant decides the object every
downstream test sees. Only a `Difference` verdict moves the
dependence/ARCH/normality/break families onto `np.diff(y)`; a `Conflict`
verdict detrends instead and routes to the trend-vs-break follow-ups. Each
family records `computed_on` so there is never ambiguity about which series a
p-value describes — the single most common way batteries mislead. Two families
deliberately stay off the differences: GPH long memory reads the *level*
(detrended first when the verdict says trend) — d near 1 there *is* the
unit-root reading, and d re-estimated on the differences flags
over-differencing when it goes significantly negative — and the outlier
screen, which is an unconditional check on the level.

**Recommendations are evidence–functions–caveat triples.** Each entry carries
a `topic`, a `finding` stated in words, the `evidence` dict of the exact
statistics that fired it, a `suggestion` naming concrete calls, the
`functions` list (every name is a real tsecon callable — a test enforces
this), and a `caveat` stating the assumption that could overturn the routing.
An entry with no caveat would be a verdict; these are deliberately not
verdicts.

**The multiple-testing footer is a design stance.** Every hypothesis test the
battery ran lands in `tests_run`, and the report closes with the arithmetic:
`n_tests × alpha` expected false rejections on a series with nothing wrong
(Elder and Kennedy, 2001, make the classroom version of this point about
unit-root testing strategy). The p-values are **never silently corrected** —
blanket Bonferroni on a battery of correlated tests would misstate the
evidence in the other direction. Families are shown, the arithmetic is shown,
and a clean series gets an explicit `no_red_flags` entry restating it, so a
"significant" stray rejection is read against its expected count.

## Failure modes

- **The Perron trap, inherited.** A stationary series with a broken mean tests
  as `UnitRoot` (Perron, 1989) — the battery then differences, and a mean
  shift in the level becomes a single spike in the differences that the break
  scan cannot see. The report's own tell is the long-memory family: GPH going
  significantly *negative* on the differences flags the over-differencing.
  Take the hint and run `bai_perron` on the level (the runnable example below
  does exactly this).
- **The break scan is intercept-only.** It is a mean-shift scan with Bai
  (1997) homogeneous-case confidence intervals, assuming serially uncorrelated
  errors under the null; strong autocorrelation inflates the sup-F and an
  unremoved trend masquerades as a cascade of mean shifts. Breaks in dynamics
  or variance need `chow_test` / `bai_perron` with your own regressors.
- **Heuristic order suggestions.** The suggested `(p, q)` comes from
  ACF/PACF cutoff runs, capped at 5 — a Box-Jenkins starting point for an
  `arima_fit` + IC comparison, not a model choice. Mixed ARMA signatures
  defeat cutoff logic by construction.
- **Seasonality is evidence, not a test.** Without `seasonal_period` you get
  an argmax-periodogram `detected_period` heuristic; with it you get ACF and
  periodogram evidence but no HEGY/Canova-Hansen test — and tsecon ships no
  seasonal ARIMA and no X-13 (roadmap), which the recommendation states
  plainly.
- **One dataset, no question.** The battery cannot know whether the series is
  an outcome or a future regressor, what the loss function is, or which
  regimes matter. It screens; it does not decide.

## Validated against

`check_series` is a composition layer, and its validation is honest about
that: every component it calls is individually validated on its own card —
ADF/KPSS/`check_stationarity`, `ljung_box`/`acf`/`pacf`, `arch_lm`,
`jarque_bera` (statsmodels/SciPy goldens; see the
[diagnostics card](diagnostics.md)), `sup_f_test`/`bai_perron`
([structural breaks](structural-breaks.md)), GPH via `long_memory_d`
([long memory](long-memory.md)), `periodogram` ([spectral](spectral.md)), and
`johansen`/VAR selection ([cointegration & regimes](cointegration-regime.md),
[VAR/SVAR](var-svar.md)). What the battery itself adds — the routing — is
validated in
[`bindings/python/tests/test_check_series.py`](../../../bindings/python/tests/test_check_series.py):
seeded-DGP recovery tests (a random walk routes to differencing, a GARCH DGP
fires the ARCH recommendation, a broken mean recovers its dates, a
cointegrated pair routes to `vecm`, a stationary VAR(3) to `var_fit`, …), a
Monte-Carlo **white-noise size check** confirming the per-family rejection
rates on pure noise stay near nominal (the multiple-testing footer is only
honest if they do), and report-contract tests pinning the exact key sets,
JSON-serializability, and that every recommended function exists. The
`.summary()` render is snapshot-tested in
[`test_results_check.py`](../../../bindings/python/tests/test_results_check.py).

## References

- Box, G. E. P. & Jenkins, G. M. (1970). *Time Series Analysis: Forecasting
  and Control.* Holden-Day. (The identify–estimate–diagnose loop the battery
  automates the first pass of.)
- Elder, J. & Kennedy, P. E. (2001). "Testing for Unit Roots: What Should
  Students Be Taught?" *Journal of Economic Education* 32(2). (Test-strategy
  discipline: decide what you are testing before you test.)
- Cochrane, J. H. (1991). "A critique of the application of unit root tests."
  *Journal of Economic Dynamics and Control* 15(2). (Near-observational
  equivalence — the limit no battery escapes.)
- Iglewicz, B. & Hoaglin, D. C. (1993). *How to Detect and Handle Outliers.*
  ASQC Quality Press. (The modified z-score behind the outlier screen.)
- Perron, P. (1989). "The Great Crash, the Oil Price Shock, and the Unit Root
  Hypothesis." *Econometrica* 57(6). (Why the example below fools the
  quadrant.)

See the guide: [Exploring and Diagnosing a Series](../../guide/02-exploration-and-diagnostics.md),
and [Which model when](../../which-model-when.md) — this battery is that page's
decision table as an executable.

## Runnable example

The centerpiece: a **broken-mean unit-root lookalike** — a stationary AR(1)
whose mean jumps mid-sample, the exact series Perron (1989) warned about.

```python
import numpy as np
import tsecon
from tsecon.results import check_series   # same battery, plus .summary()

# A Perron (1989) trap: a stationary AR(1) whose mean jumps mid-sample —
# the classic unit-root lookalike.
rng = np.random.default_rng(42)
n = 400
mu = np.where(np.arange(n) < 200, 0.0, 3.0)        # mean shifts 0 -> 3 at t = 200
y = np.zeros(n)
for t in range(1, n):
    y[t] = mu[t] + 0.5 * (y[t - 1] - mu[t - 1]) + rng.standard_normal()

report = check_series(y)                  # a dict, plus .summary()/.plot_diagnostics()
print(report.summary())
```

Output:

```
====================================================================
check_series — univariate, n=400, alpha=0.05
====================================================================
mean +1.4909    sd 1.9094    skew +0.142    ex.kurt -0.844
min -2.4924    max +6.4412    outliers 0
  outlier screen: modified z-score (median/MAD, |z|>3.5, Iglewicz-
  Hoaglin)
--------------------------------------------------------------------
Stationarity (ADF + KPSS) — on level
  quadrant UnitRoot -> recommendation Difference
  adf    stat -1.2093   p 0.67
  kpss   stat +2.7729   p 0.01
  analysis scale: first_difference
    The ADF+KPSS quadrant is 'UnitRoot' and the workflow says to
    difference, so all downstream dependence/ARCH/normality/break
    tests run on the first differences (n=399).
--------------------------------------------------------------------
Serial correlation — on first_difference
  ljung_box (lags 1..10)   stat 32.1292   p 0.000381
  significant ACF lags: 1, 2, 3
  significant PACF lags: 1, 2, 3, 4, 5, 6, 9, 11, 15, 20
  suggested starting orders   p 0   q 3
--------------------------------------------------------------------
ARCH effects — on first_difference
  arch_lm   stat 4.7462   p 0.314   not rejected
--------------------------------------------------------------------
Normality (Jarque-Bera) — on first_difference
  jarque_bera   stat 0.9002   p 0.638   not rejected
  skew -0.102   ex.kurt +0.112
--------------------------------------------------------------------
Structural breaks — on first_difference
  sup_f_test   stat 0.1414   p 1   break_date 248
  bai_perron skipped: sup-F did not reject at alpha=0.05
--------------------------------------------------------------------
Long memory (GPH) — on level
  gph (level)   d +0.5558   se 0.1434   m 20
  gph (differences)   d -0.5694   se 0.1471   m 19
  GPH on the level gives d=0.556 (se 0.143), significantly below 1
  at the 5% level — the unit-root verdict's integer difference may
  be the wrong filter (a fractional d is plausible); read the
  differences next. On the first differences d=-0.569 (se 0.147):
  significantly negative — integer differencing may be too much
  (over-differencing); a fractional filter frac_diff(y, d) with d ≈
  0.43 on the level is the alternative.
--------------------------------------------------------------------
Seasonality — on first_difference
  detected_period 3.44
  detected_period is simply the argmax periodogram ordinate — a
  heuristic, not a test. Check the ACF at that lag before believing
  it; pass seasonal_period to get seasonal evidence.
--------------------------------------------------------------------
Recommendations
--------------------------------------------------------------------
 1. unit_root
    quadrant 'UnitRoot': the tests agree the series looks I(1)
    -> Model in first differences — arima_fit(y, d=1, ...)
       differences internally — and re-run check_series on the
       differences to pick the short-run orders.
    functions: check_stationarity, arima_fit
    caveat: Near the unit circle, stationary and integrated
    processes are nearly observationally equivalent in finite
    samples — ADF/KPSS organize the evidence, they cannot
    manufacture information.
--------------------------------------------------------------------
 2. persistent_regressor
    the series is highly persistent
    -> If this series will be a REGRESSOR (e.g. predicting returns),
       standard t-tests are size-distorted under persistence: use
       predictive_regression (Stambaugh + IVX) for one predictor or
       ivx_test for a joint test.
    functions: predictive_regression, ivx_test
    caveat: Only relevant when the series sits on the right-hand
    side; irrelevant for modeling it as the outcome.
--------------------------------------------------------------------
 3. arma_orders
    Ljung-Box rejects whiteness on the first_difference (p=0.000381)
    -> Start from arima_fit(y, p=0, d=1, q=3) and compare AIC/BIC
       over a small order grid — the ACF/PACF cutoff heuristics are
       starting points, not a verdict.
    functions: arima_fit, acf, pacf
    caveat: Order heuristics assume a clean cutoff pattern; mixed
    ARMA signatures need the IC comparison to disambiguate.
--------------------------------------------------------------------
 4. long_memory
    GPH d=-0.569 (se 0.147) on the differences is significantly
    negative — integer differencing looks like too much
    -> The level may be fractionally integrated with d<1: apply
       frac_diff(y, d=0.43) to the level instead of a first
       difference, then model the filtered series.
    functions: long_memory_d, frac_diff
    caveat: GPH is a small-bandwidth semiparametric estimator:
    short-memory AR dynamics bias it upward, and level shifts and
    unremoved deterministic trends masquerade as long memory. Cross-
    check with long_memory_d(method='local_whittle') and the break
    scan.
====================================================================
6 hypothesis tests at alpha=0.05 - expect ~0.25 false alarms from the 5 true-null tests on a clean series.
====================================================================
```

Read it the way the card promised. The quadrant was fooled — Perron's point —
and the battery differenced. But the report carries its own tell: GPH on the
differences is *significantly negative* (recommendation 4), the
over-differencing flag. That is the screening report saying "my differencing
verdict may be wrong; interrogate the level." Doing so is one call:

```python
bp = tsecon.bai_perron(y, np.ones((n, 1)))
print("breaks:", bp["n_breaks"], "at", bp["break_dates"],
      " regime means:", [round(p[0], 2) for p in bp["params"]])
for name, segment in [("first regime ", y[:200]), ("second regime", y[200:])]:
    verdict = tsecon.check_stationarity(segment)
    print(name, verdict["quadrant"], "->", verdict["recommendation"])
```

```
breaks: 1 at [199]  regime means: [-0.06, 3.04]
first regime  Stationary -> Proceed
second regime Stationary -> Proceed
```

One mean shift at t = 199, regime means −0.06 and 3.04, and each regime is
comfortably stationary: not a unit root at all, but a break — the judgment
call the battery flags and you make.

**A short multivariate example.** A cointegrated pair — two I(1) series
sharing one stochastic trend — must *not* be differenced into a VAR:

```python
rng = np.random.default_rng(7)
w = np.cumsum(rng.standard_normal(300))            # one shared stochastic trend
pair = np.column_stack([w + 0.5 * rng.standard_normal(300),
                        0.8 * w + 0.5 * rng.standard_normal(300)])
m = tsecon.check_series(pair)                      # kind == "multivariate"
print([s["verdict"] for s in m["per_series"]])
print(m["cointegration"]["interpretation"])
print([r["topic"] for r in m["recommendations"]])
print(m["recommendations"][0]["suggestion"])
```

```
['UnitRoot', 'UnitRoot']
trace rank = 1: 1 cointegrating relation(s) — differencing everything would discard the error-correction terms; route through vecm
['cointegration', 'single_shock_irf']
Model the system as a VECM: vecm(data, k_ar_diff=1, coint_rank=1). Differencing everything would throw away the error-correction terms that tie the levels together.
```

Both series draw `UnitRoot` verdicts, Johansen finds rank 1, and the routing
lands where the textbook says it should: a VECM, not a differenced VAR — with
the lag-sensitivity caveat attached.
