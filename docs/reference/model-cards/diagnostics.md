# Model card — Diagnostics and the stationarity workflow

**Family:** `acf`, `pacf`, `ljung_box`, `jarque_bera`, `arch_lm`, `adf`, `kpss`,
`check_stationarity` — plus the seasonal workflow: `stl`, `seasonal_strength`,
`nsdiffs` (see the [dedicated section](#stl-decomposition-and-the-seasonal-workflow)
at the end of this card) and the trend-cycle workflow: `hamilton_filter`'s
inference surface, `bn_decomposition`, `bn_filter` (see the
[dedicated section](#trend-cycle-decomposition-the-hamilton-filters-inference-and-the-beveridge-nelson-family))

The first hour with any series. Before you fit a model you need to know how
persistent the data are, what lag structure they carry, and whether they must
be differenced. After you fit one, the same battery tells you whether the
residuals are the white noise the model assumed. These are cheap, standard,
and the mistakes people make with them are equally standard — this card is
about avoiding those.

| Function | What it answers |
|----------|-----------------|
| `acf` / `pacf` | How persistent is the series, and what AR/MA order does it suggest? |
| `ljung_box` | Is there *any* linear autocorrelation left, jointly across lags? |
| `jarque_bera` | Are the (residual) innovations Gaussian? |
| `arch_lm` | Is there conditional heteroskedasticity (volatility clustering)? |
| `adf` / `kpss` | Is there a unit root? (opposite nulls — read them together) |
| `check_stationarity` | The ADF + KPSS confirmatory quadrant, with a recommendation |

## What it estimates

- **`acf(y)`** — the autocorrelation function ρ(k) = Corr(yₜ, yₜ₋ₖ) for
  k = 0…nlags, with Bartlett standard errors for the "is this spike real?"
  bands. **`pacf(y)`** — the partial autocorrelations, the correlation at lag k
  after projecting out lags 1…k−1 (Yule-Walker or OLS).
- **`ljung_box(y)`** — the portmanteau statistic Q = n(n+2)Σρ̂(k)²/(n−k),
  which is χ²(nlags) under the white-noise null, plus the Box-Pierce variant.
- **`jarque_bera(x)`** — a χ²(2) test built from sample skewness and excess
  kurtosis; the null is normality.
- **`arch_lm(resid)`** — Engle's LM test: regress squared residuals on their
  own lags and test joint significance (null: no ARCH).
- **`adf(y)`** — the Augmented Dickey-Fuller t-statistic for a unit root
  (null: unit root), with MacKinnon response-surface p-values.
- **`kpss(y)`** — the KPSS statistic (null: **stationary**), the deliberate
  mirror of ADF.
- **`check_stationarity(y)`** — runs both, places the series in the
  ADF×KPSS confirmatory quadrant, and returns a plain-language recommendation.

## Assumptions

- **ACF/PACF and Ljung-Box** describe *linear* dependence. A series can be
  serially dependent through its variance (GARCH) or nonlinearly while showing
  a flat ACF — a clean Ljung-Box is not a clean bill of health; pair it with
  `arch_lm`.
- **Ljung-Box on model residuals** should have its degrees of freedom reduced
  by the number of estimated ARMA parameters. This function returns the raw
  χ²(lag) p-values; for an ARMA(p,q) fit, compare against χ² with lag−p−q df.
- **Jarque-Bera** is asymptotic and over-rejects in small samples; with a few
  hundred observations a "significant" p-value often just means fat tails, not
  a broken model.
- **ADF** assumes the only nonstationarity is a unit root — a deterministic
  trend must be modeled through `regression="ct"`, or ADF will confound trend
  with a root. **KPSS** assumes the alternative is a unit root.
- Both unit-root tests have low power near the boundary (φ close to 1): a
  near-unit-root stationary series and a true random walk look alike in
  samples of a few hundred. This is why you run both.

## When to use

- **Always, first.** ACF/PACF and `check_stationarity` are the opening move on
  any univariate series — they tell you whether to difference and roughly what
  order to fit.
- **ACF geometric decay + PACF cutoff at lag p** → an AR(p); the mirror image
  (PACF decay, ACF cutoff) → an MA(q). This is Box-Jenkins identification.
- **Ljung-Box / Jarque-Bera / ARCH-LM after fitting** — the residual battery.
  A surviving Ljung-Box rejection means the mean model is under-specified; a
  surviving ARCH-LM rejection means you need a volatility model (see the
  realized-vol and GARCH cards).
- Use `check_stationarity` rather than ADF alone — running one test and
  ignoring its complement is the single most common unit-root mistake.

## Key arguments and defaults

| Call | Argument | Default | Notes |
|------|----------|---------|-------|
| `acf` | `nlags` | `20` | number of lags returned (plus lag 0) |
| | `adjusted` | `False` | `True` uses the n−k divisor (less biased, higher variance) |
| `pacf` | `nlags` | `20` | |
| | `method` | `"yw"` | Yule-Walker; `"ols"` for the regression estimator |
| `ljung_box` | `nlags` | `10` | statistic reported for each lag 1…nlags |
| `arch_lm` | `nlags` | `4` | number of squared-residual lags |
| `adf` | `regression` | `"c"` | `"c"` constant, `"ct"` constant+trend, `"n"` none |
| | `autolag` | `"aic"` | lag selection; or pass `maxlag` directly |
| `kpss` | `regression` | `"c"` | `"c"` level-stationary, `"ct"` trend-stationary |
| | `nlags` | `None` | `None` → automatic (Hobijn-Franses-Ooms) bandwidth |
| `check_stationarity` | `alpha` | `0.05` | significance level for both underlying tests |

## How to read the output

- **`acf`** → `{"acf", "bartlett_se"}`, both length `nlags+1` (index 0 is the
  trivial ρ(0)=1). A spike outside ±1.96·`bartlett_se[k]` is significant at 5%.
  **`pacf`** returns a bare array of the same length.
- **`ljung_box`** → `{"lags", "lb_stat", "lb_pvalue", "bp_stat", "bp_pvalue"}`,
  one entry per lag. Small `lb_pvalue` ⇒ reject white noise. Prefer the
  Ljung-Box (`lb_*`) columns; Box-Pierce is the older, less accurate variant.
- **`jarque_bera`** → `{"statistic", "p_value", "skewness", "kurtosis", "n"}`.
  Note `kurtosis` is the raw (not excess) value — 3 is Gaussian.
- **`arch_lm`** → `{"statistic", "p_value", "df", "nobs"}`. Small `p_value` ⇒
  volatility clustering.
- **`adf`** → `{"statistic", "p_value", "used_lag", "nobs", "crit"}`, where
  `crit` is a dict of the 1/5/10% critical values. Small `p_value` ⇒ **reject**
  the unit root (series looks stationary).
- **`kpss`** → `{"statistic", "p_value", "lags"}`. `p_value` is clipped to the
  tabulated `[0.01, 0.10]` range; **small** `p_value` ⇒ **reject** stationarity.
- **`check_stationarity`** → `quadrant` ∈ {`Stationary`, `UnitRoot`, `Conflict`,
  `Inconclusive`}, a `recommendation` (`Proceed` / `Difference` / `Detrend`),
  a plain-language `interpretation`, and the raw test statistics/p-values.

## Failure modes

- **Reading ADF alone.** A failure to reject a unit root is *not* evidence of
  one — it may just be low power. `check_stationarity` exists to force the
  confirmatory reading; act on the `quadrant`, not a single p-value.
- **Trend mistaken for a root.** A trend-stationary series fed to `adf` with
  the default `regression="c"` will look like a unit root. Use `"ct"` when a
  deterministic trend is plausible, and `kpss(..., regression="ct")` to match.
- **Clean Ljung-Box, dirty variance.** Linear-autocorrelation tests miss ARCH.
  Always run `arch_lm` on residuals before declaring them white noise.
- **Ljung-Box df on residuals.** These functions do not subtract estimated
  parameters from the degrees of freedom; over-optimistic p-values result if
  you read them naïvely on ARMA residuals (see Assumptions).
- **Jarque-Bera in large samples** rejects on economically trivial fat tails —
  inspect skewness and kurtosis, do not stop at the p-value.

## Validated against

`statsmodels` to tight tolerance: `acf`/`pacf` (`acf`, `pacf`), Ljung-Box and
Box-Pierce (`acorr_ljungbox`), ARCH-LM (`het_arch`), Jarque-Bera, and the ADF
and KPSS statistics with MacKinnon (2010) p-value response surfaces and the
Hobijn-Franses-Ooms automatic KPSS bandwidth; `scipy.stats` for the
distributional pieces. The golden values are pinned in
[`fixtures/diagnostics.json`](../../../fixtures/diagnostics.json) and
[`fixtures/unitroot.json`](../../../fixtures/unitroot.json).

## References

- Ljung, G. & Box, G. (1978). "On a Measure of Lack of Fit in Time Series
  Models." *Biometrika* 65.
- Jarque, C. & Bera, A. (1980). "Efficient tests for normality,
  homoscedasticity and serial independence." *Economics Letters* 6.
- Engle, R. (1982). "Autoregressive Conditional Heteroscedasticity."
  *Econometrica* 50.
- Dickey, D. & Fuller, W. (1979). "Distribution of the Estimators for
  Autoregressive Time Series with a Unit Root." *JASA* 74.
- Kwiatkowski, Phillips, Schmidt & Shin (1992). "Testing the null hypothesis
  of stationarity against the alternative of a unit root." *J. Econometrics* 54.
- MacKinnon, J. (2010). "Critical Values for Cointegration Tests." Queen's
  Economics Department WP 1227.

See the guide: [Exploring and Diagnosing a Series](../../guide/02-exploration-and-diagnostics.md).

## Runnable example

```python
import numpy as np
import tsecon

rng = np.random.default_rng(0)
walk = np.cumsum(rng.standard_normal(300))          # a random walk (unit root)

# 1. Is it white noise? Ljung-Box portmanteau on the levels.
lb = tsecon.ljung_box(walk, nlags=10)
print("Ljung-Box p at lag 10:", round(lb["lb_pvalue"][-1], 4))     # ~0 -> not white noise

# 2. ACF and PACF shape (Box-Jenkins identification).
r = tsecon.acf(walk, nlags=10)          # dict: acf, bartlett_se
p = tsecon.pacf(walk, nlags=10)         # array; method "yw" (default) or "ols"
print("acf(1):", round(r["acf"][1], 3), " pacf(1):", round(p[1], 3))

# 3. The confirmatory stationarity workflow: ADF (H0: unit root) + KPSS (H0: stationary).
rep = tsecon.check_stationarity(walk)
print(rep["quadrant"], "->", rep["recommendation"])                # UnitRoot -> Difference

# 4. Re-run on the differences; they should now look stationary.
print("after differencing:", tsecon.check_stationarity(np.diff(walk))["recommendation"])

# 5. Post-fit residual checks: normality and conditional heteroskedasticity.
resid = rng.standard_normal(300)
print("Jarque-Bera p:", round(tsecon.jarque_bera(resid)["p_value"], 3))
print("ARCH-LM p:", round(tsecon.arch_lm(resid, nlags=5)["p_value"], 3))

# The individual unit-root tests are available directly with their p-values.
print("ADF p:", round(tsecon.adf(walk)["p_value"], 3),
      " KPSS p:", round(tsecon.kpss(walk)["p_value"], 3))
```

Expected output:

```
Ljung-Box p at lag 10: 0.0
acf(1): 0.971  pacf(1): 0.971
UnitRoot -> Difference
after differencing: Proceed
Jarque-Bera p: 0.001
ARCH-LM p: 0.175
ADF p: 0.841  KPSS p: 0.01
```

---

## STL decomposition and the seasonal workflow

**Family:** `stl`, `mstl`, `seasonal_strength`, `nsdiffs`

Season-Trend decomposition using LOESS (Cleveland, Cleveland, McRae &
Terpenning 1990) — the workhorse for exploratory seasonal adjustment outside
official statistics — plus the two advisors built on it: the
Wang-Smith-Hyndman strength-of-seasonality measures and the
Hyndman-Khandakar `nsdiffs` seasonal-differencing rule.

### What it estimates

- **`stl(y, period, ...)`** — the additive decomposition
  `y = seasonal + trend + resid`. The inner loop LOESS-smooths each
  *cycle-subseries* (all Januaries, all Februaries, …), low-passes the result
  so the seasonal averages ~0 over every cycle, and LOESS-smooths the
  deseasonalized series into the trend; the optional outer loop downweights
  outliers with bisquare robustness weights on the remainder. This is the
  netlib Fortran `stl.f` semantics, matched to `statsmodels.tsa.seasonal.STL`
  elementwise.
- **`seasonal_strength(y, period)`** — from a default STL fit:
  `strength_seasonal = max(0, 1 − var(resid)/var(seasonal + resid))` and the
  analogous `strength_trend` (sample variances). Near 1: the component
  dominates; near 0: absent.
- **`nsdiffs(y, period, alpha=0.05, max_d=1)`** — the number of seasonal
  differences `D`: difference at the seasonal lag while
  `seasonal_strength >= 0.64`, capped at `max_d` (the
  `forecast::nsdiffs(test="seas")` rule). `alpha` is validated but unused —
  the rule is threshold-based, not a hypothesis test (forecast ignores it for
  this test too).

### Assumptions and when to use

- **Additive components.** STL decomposes additively; for multiplicative
  seasonality (amplitude growing with the level) log-transform first —
  `box_cox_lambda` tells you whether the log is defensible.
- **One fixed integer period** (12 monthly, 4 quarterly) per component.
  Multiple seasonalities (hourly data with daily *and* weekly cycles) are
  `mstl`'s job — see its section below; non-integer seasonalities need STR
  (roadmap).
- **Use `stl` before a SARIMA fit** (does the seasonal look stable? how big is
  it relative to the noise?), to seasonally adjust for eyeballing turning
  points, or to feed `resid` to outlier screens. `robust=True` when the series
  has suspected outliers — the weights returned tell you which points the fit
  ignored.
- **Use `nsdiffs` + `ndiffs`, in that order** (seasonal difference first, then
  the regular difference on the seasonally-differenced series) to pick
  SARIMA's `(d, D)` the way auto-arima procedures do.

### Key arguments and defaults (mirror statsmodels exactly)

| Argument | Default | Notes |
|----------|---------|-------|
| `seasonal` | `7` | seasonal LOESS window; odd, ≥ 3; larger → smoother, more nearly periodic seasonal |
| `trend` | `None` | odd, > period; `None` → smallest odd ≥ `1.5·period/(1−1.5/seasonal)` |
| `low_pass` | `None` | odd, > period; `None` → smallest odd > period |
| `seasonal_deg`/`trend_deg`/`low_pass_deg` | `1` | LOESS degree, 0 or 1 |
| `robust` | `False` | `True` runs bisquare outer iterations |
| `*_jump` | `1` | evaluate the LOESS every jump-th point, interpolate between (speedup) |
| `inner_iter`/`outer_iter` | `None` | `None` → 2/15 if `robust` else 5/0 (Cleveland et al. §3.3) |

Requires `n >= 2·period` (R's `stl()` bound; statsmodels silently misbehaves
below it) and `period >= 2`.

### How to read the output

- **`stl`** → `{"seasonal", "trend", "resid", "weights", "period", "config"}`.
  The identity `y = seasonal + trend + resid` holds exactly; `weights` are all
  1 unless the outer loop ran (0 = ignored as an outlier); `config` reports
  every resolved window/degree/jump and the iteration counts actually used.
- **`seasonal_strength`** → `{"seasonal_strength", "trend_strength", "period"}`.
- **`nsdiffs`** → `{"d", "stop", "steps", ...}` in the `ndiffs` house style:
  per-order evidence in `steps`, and `stop` says *why* the sequence ended
  (`WeakSeasonality` is the intended exit; `MaxD`/`TooShort` mean `d` is a
  floor, not a verdict; `Constant` means the seasonality was deterministic and
  is gone).

### Failure modes

- **Seasonal window too small.** `seasonal=7` lets the seasonal pattern evolve;
  if you believe the pattern is fixed, use a large odd window (e.g. 51+ —
  "periodic-ish") or the seasonal barely smooths and noise leaks into it.
- **Trend window ≤ period is rejected** — it would let the trend absorb the
  seasonal. The default rule exists precisely to prevent that leakage.
- **Robust weights all ≈ 1 under `robust=True`** just means no outliers — not
  a failure. Conversely a weight of 0 on a *real* event (a strike, a
  recession trough) means the decomposition is describing the series without
  that event; check `weights` before interpreting `resid`.
- **`nsdiffs` on seasonally adjusted data** (e.g. most US macro releases)
  correctly returns `D=0`; running it is still worthwhile as a check that the
  adjustment did its job.
- **Fewer than ~4 full cycles saturates the strength rule** (audit round 6,
  measured on pure white noise, period 12): with only 2 cycles (n = 24) the
  STL cycle-subseries interpolates noise straight into the seasonal component
  and `seasonal_strength` is 1.000 on *every* draw — `nsdiffs` flags **D = 1
  on 100% of white-noise series at n ≤ 28**, 38% at n = 48 (four cycles), 2%
  at n = 72, 0% by n = 120. R's `forecast::nsdiffs` behaves identically (the
  rule is matched, not mis-implemented), and the `stop="TooShort"` marker
  warns in the *other* direction (d as a floor). With under ~4–6 cycles,
  treat `D = 1` as "not enough data to tell", not as evidence of seasonality.
- **A constant series raises** from `seasonal_strength` (the variance-ratio
  measure is undefined there — the ratio of the decomposition's float-noise
  variances is implementation noise; audit round 6 measured ≈ 0.61–0.67 on
  flat lines before the guard). `nsdiffs` and `check_series` already
  special-case constants themselves.

### Validated against

`statsmodels.tsa.seasonal.STL` 0.14.6 **elementwise** (seasonal, trend, resid,
robustness weights) on CO2 monthly, 100·log US real GDP quarterly, and a
seeded synthetic monthly series, across defaults / `robust=True` / a large
seasonal window / `seasonal_deg=0` / non-unit jumps / explicit inner-outer
counts, at 1e-8 tolerance (observed agreement ~1e-12; the algorithm is a
deterministic port of the same Fortran). The strength measures and the 0.64
rule have no reference implementation in the test environment (R-only), so
they are graded honestly as documented-formula/rule transcriptions computed
from statsmodels components — see the header of
[`fixtures/generate_stl_fixtures.py`](../../../fixtures/generate_stl_fixtures.py).
Pinned in [`fixtures/stl.json`](../../../fixtures/stl.json).

### References

- Cleveland, R. B., Cleveland, W. S., McRae, J. E. & Terpenning, I. (1990).
  "STL: A Seasonal-Trend Decomposition Procedure Based on Loess."
  *Journal of Official Statistics* 6, 3–73.
- Wang, X., Smith, K. & Hyndman, R. (2006). "Characteristic-based clustering
  for time series data." *Data Mining and Knowledge Discovery* 13, 335–364.
- Hyndman, R. & Khandakar, Y. (2008). "Automatic time series forecasting:
  the forecast package for R." *JSS* 27(3).
- Hyndman, R. & Athanasopoulos, G. *Forecasting: Principles and Practice*
  (3rd ed.), §3.6 (STL), §4.3 (strength), §9.9 (`nsdiffs`).

### Runnable example

```python
import numpy as np
import tsecon

t = np.arange(240, dtype=float)
y = 10 + 0.05*t + 3*np.sin(2*np.pi*t/12) + 0.4*np.sin(t*0.7134)

r = tsecon.stl(y, 12, robust=True)                  # statsmodels-exact STL
print("trend window:", r["config"]["trend"],        # default rule -> 23
      "inner/outer:", r["config"]["inner_iter"], r["config"]["outer_iter"])

s = tsecon.seasonal_strength(y, 12)
print("seasonal strength:", round(s["seasonal_strength"], 3))

d = tsecon.nsdiffs(y, 12)                           # SARIMA's D
print("nsdiffs D =", d["d"], "| stop:", d["stop"])
```

Expected output:

```
trend window: 23 inner/outer: 2 15
seasonal strength: 0.978
nsdiffs D = 1 | stop: WeakSeasonality
```

---

## MSTL: decomposition with multiple seasonal cycles

**Family:** `mstl`

Multiple Seasonal-Trend decomposition using LOESS (Bandara, Hyndman &
Bergmeir 2021) — STL iterated over several seasonal periods, for series
whose seasonality has more than one layer: hourly load with a daily *and* a
weekly cycle (`periods=[24, 168]`), daily sales with weekly and annual
cycles, and so on. Matches `statsmodels.tsa.seasonal.MSTL` elementwise.

### What it estimates

The additive decomposition
`y = seasonal_1 + … + seasonal_K + trend + resid`, one seasonal component
per period. Periods are sorted ascending and any period ≥ n/2 is dropped
(statsmodels warns; here the drop is reported in `dropped_periods`). Each
of `iterate` rounds (default 2; forced to 1 for a single period) cycles
over the periods, re-running STL at each period on the series
deseasonalized of all the *other* components — so each seasonal is
re-extracted with the competing cycles removed, which is what lets nested
cycles (24 inside 168) separate cleanly. `trend` and the robustness
`weights` come from the final STL pass; `resid` is what's left.

### Assumptions and when to use

- **Additive components**, as with `stl`: log-transform first for
  multiplicative seasonality. statsmodels' `lmbda`/Box-Cox option is
  deliberately **not implemented** — pre-transform `y` yourself
  (`box_cox_lambda` advises on the exponent).
- **Distinct integer periods**, each < n/2. A single period degenerates to
  plain `stl` with seasonal window 11 (bit-for-bit — tested), so `mstl` is
  a safe default entry point for seasonal decomposition generally.
- **Use it before modeling multi-seasonal data** (which cycle dominates?
  is the weekly pattern stable?), to seasonally adjust at several
  frequencies at once, or to feed per-cycle strengths into a seasonality
  triage. For one ordinary monthly/quarterly cycle, `stl` gives you finer
  control (its `seasonal` window default 7 vs MSTL's 11).

### Key arguments and defaults (mirror statsmodels exactly)

| Argument | Default | Notes |
|----------|---------|-------|
| `periods` | required | sequence of observations-per-cycle, e.g. `[24, 168]`; a single period is `[12]`; sorted ascending internally |
| `windows` | `None` | per-period seasonal LOESS window (odd, ≥ 3), paired with the same-index period; `None` → the paper's rule 7 + 4·k over the *sorted* periods: 11, 15, 19, … |
| `iterate` | `2` | refinement rounds over all periods; 1 is faster and usually close; forced to 1 when only one period survives |
| STL kwargs | as in `stl` | `trend`, `low_pass`, degrees, `robust`, jumps, `inner_iter`/`outer_iter` are forwarded unchanged to **every** per-period STL pass |

Deliberate safe-side refusals where statsmodels crashes or degrades
silently: empty `periods`, duplicate periods, `iterate=0`, and "every
period was dropped" are teaching `ValueError`s.

### How to read the output

- `{"seasonal", "trend", "resid", "weights", "periods", "windows",
  "iterate", "dropped_periods", "seasonal_strength"}`.
- **`seasonal`** is a dict keyed `"seasonal_<period>"` in ascending-period
  order; the components plus `trend` plus `resid` reconstruct `y` exactly.
- **`periods`/`windows`** are the *resolved* values (sorted, post-drop) —
  check `dropped_periods` whenever n is short relative to the longest
  cycle: a silently absent component changes the meaning of `trend`.
- **`seasonal_strength`** gives the Wang-Smith-Hyndman strength of each
  component against the shared remainder (same guarded formula as
  `seasonal_strength`); it is `None` for a constant input series, where
  the variance ratio would be float noise.
- **`weights`** are the final pass's bisquare robustness weights (all 1
  unless `robust=True`/`outer_iter>0`).

### Failure modes

- **A period ≥ n/2 vanishes by design** (with `dropped_periods` saying
  so): fewer than two full cycles cannot be told from trend. If the long
  cycle is the one you care about, you need more data, not different
  windows.
- **Close periods compete.** Periods like 28 and 30 have nearly identical
  frequencies at short n; MSTL will split their energy arbitrarily.
  Merge them or fix one of them by prior knowledge.
- **Leakage between cycles at iterate=1.** With strongly nested cycles
  the first pass extracts the short cycle from a series still carrying
  the long one; the second round (the default) cleans this up. If
  components look contaminated, raise `iterate`, not the windows.
- **The trend window binds across all periods.** A forwarded `trend`
  window must exceed the *longest* period (each pass validates it); the
  default rule re-resolves per pass, which is almost always what you
  want.
- **Duplicate periods are refused** rather than silently producing two
  components of the same period (which statsmodels does).

### Validated against

`statsmodels.tsa.seasonal.MSTL` 0.14.6 **elementwise** (trend, every
per-period seasonal, resid, robustness weights) on a seeded two-seasonal
hourly-like series (24/168), a seeded three-seasonal awkward-period series
(5/12/31), the degenerate single-period case, and a dropped-period case —
across default and explicit (unsorted) windows, `robust`, forwarded
`stl_kwargs` including `inner_iter`/`outer_iter`, and `iterate` 1–4, at
1e-8 tolerance (observed ≤ ~5e-11 on components; the algorithm drives the
same netlib STL core our `stl` pins). Grade: **strong third-party golden
(statsmodels MSTL, elementwise)**. The single-period case is additionally
required to reproduce tsecon's own `stl` **bitwise** — internal
consistency, graded separately. Pinned in
[`fixtures/mstl.json`](../../../fixtures/mstl.json); provenance in
[`fixtures/generate_mstl_fixtures.py`](../../../fixtures/generate_mstl_fixtures.py).

### References

- Bandara, K., Hyndman, R. J. & Bergmeir, C. (2021). "MSTL: A
  Seasonal-Trend Decomposition Algorithm for Time Series with Multiple
  Seasonal Patterns." arXiv:2107.13462.
- Cleveland, R. B., Cleveland, W. S., McRae, J. E. & Terpenning, I. (1990).
  "STL: A Seasonal-Trend Decomposition Procedure Based on Loess."
  *Journal of Official Statistics* 6, 3–73.
- Hyndman, R. & Athanasopoulos, G. *Forecasting: Principles and Practice*
  (3rd ed.), §12.1 (complex seasonality).

### Runnable example

```python
import numpy as np
import tsecon

t = np.arange(24 * 7 * 6, dtype=float)                # 6 weeks hourly
y = (20 + 0.01*t + 3*np.sin(2*np.pi*t/24)            # daily cycle
     + 5*np.sin(2*np.pi*t/168) + 0.3*np.sin(t*0.91)) # weekly cycle + wobble

r = tsecon.mstl(y, [168, 24])                         # order doesn't matter
print("periods:", r["periods"], "| windows:", r["windows"],
      "| dropped:", r["dropped_periods"])
for k, v in r["seasonal_strength"].items():
    print(k, "strength:", round(v, 3))

recon = sum(np.asarray(s) for s in r["seasonal"].values()) \
    + r["trend"] + r["resid"]
print("reconstructs y:", bool(np.allclose(recon, y)))
```

Expected output:

```
periods: [24, 168] | windows: [11, 15] | dropped: []
seasonal_24 strength: 0.991
seasonal_168 strength: 0.997
reconstructs y: True
```

## Trend-cycle decomposition: the Hamilton filter's inference and the Beveridge-Nelson family

**Family:** `hamilton_filter` (extended: `method`, `se`), `bn_decomposition`,
`bn_filter`

Three ways to split a drifting macro series into trend and cycle, each with
a different discipline on the trend. The Hamilton (2018) regression filter
is the recommended replacement for HP filtering; the classic
Beveridge-Nelson (1981) decomposition defines the trend as the long-horizon
conditional expectation of an estimated ARIMA; the Kamber-Morley-Wong
(2018) BN *filter* keeps the BN definition but pins the signal-to-noise
ratio, which is what turns the classic BN's famously tiny cycle into an
intuitive output gap. When these fixtures were generated statsmodels shipped
**none of these** — which is why the `hamilton_filter` and `bn_decomposition`
goldens below are formula transcriptions rather than reference runs — and the
absence is pinned by a canary in `fixtures/bn_filters.json`. That canary has
since fired on one half: **statsmodels 0.15.0 added
`tsa.filters.api.hamilton_filter`** (the decomposition only — no standard
errors, no `method="random_walk"`), so the canary test now runs a live,
version-gated cross-check of our full cycle/trend decomposition against it —
**measured max abs 4.2e-14** on first contact, asserted at 1e-10.
**statsmodels still ships no BN decomposition in any form** (re-verified live
by the same test), which shapes how the BN pair is validated below.

### What they estimate

- **`hamilton_filter(y, h, p, method="regression")`** — OLS of `y_t` on
  `[1, y_{t-h}, …, y_{t-h-p+1}]`; `cycle` = residual, `trend` = fitted
  value. `method="random_walk"` is the short-sample variant Hamilton
  recommends when the regression sample is thin: `cycle = y_t − y_{t−h}`
  (the population regression under a random-walk null; no coefficients).
  Frequency-aware defaults (the horizon spans two years, the lags one):

  | frequency | `h` | `p` |
  |-----------|-----|-----|
  | quarterly |  8  |  4  |
  | monthly   | 24  | 12  |
  | annual    |  2  |  1  |

- **`hamilton_filter(..., se="hac")`** — Newey-West standard errors on the
  regression coefficients, through the library's single HAC engine
  (`tsecon-hac`). The residual `v_t` is an `h`-step-ahead forecast error
  observed at overlapping horizons, so it is serially correlated **by
  construction** — MA(`h−1`) under a correctly specified model — and
  classical OLS standard errors are simply wrong for this regression
  (Hamilton's own tables use Newey-West). The default bandwidth is the
  **h-overlap rule `maxlags = h`**: it covers the known MA(`h−1`)
  correlation with one lag of slack, where generic plug-in rules
  (`0.75·n^{1/3}` ≈ 4 at n ≈ 200) can land *below* `h−1` and truncate
  autocorrelation known to exist. `se="nonrobust"` is provided as the
  comparison point.
- **`bn_decomposition(y, p, q)`** — classic BN from `ARIMA(p, 1, q)` with
  constant, fit by the library's exact MLE (default `p=2, q=2`, the
  Morley-Nelson-Zivot 2003 US-GDP spec). The trend is the long-horizon
  conditional expectation net of deterministic growth — algebraically a
  **random walk with drift in the series' own innovations**,
  `Δτ_t = μ + ψ(1)·ε_t`, where `ψ(1) = θ(1)/φ(1)` is the long-run
  multiplier (the cumulative impulse response — the permanent effect of a
  unit shock). Cycle: `c_t = y_t − τ_t = −e1′F(I−F)⁻¹X_t` in the ARMA
  companion form (Morley 2002), with conditional (zero-presample)
  innovations. Passing `ar`/`ma`/`drift` decomposes at fixed (e.g.
  published) coefficients instead of fitting.
- **`bn_filter(y, p, delta, demean)`** — Kamber-Morley-Wong: AR(`p`) on
  demeaned growth with `Σφ` **fixed at `ρ = 1 − 1/√δ`** (a Bayesian ridge
  on the Dickey-Fuller form with their `N(0, 0.5/j²)` shrinkage prior),
  `δ` selected by the paper's amplitude-to-noise criterion (first local
  maximum of `var(cycle)/mean(residual²)` on the grid `d0=0.01, dt=0.0005`)
  or imposed. Baseline `p=12` for quarterly data; `cycle_se` is the
  reference code's fixed error band (95% band `cycle ± 1.96·cycle_se`).

### When to use which

- **Hamilton** for a regression-based cycle with no model of the trend at
  all — robust to the exact ARIMA form, loses `h+p−1` observations, and
  produces a cycle whose interpretation ("what was not predictable two
  years out") differs from a band-pass or BN gap.
- **Classic BN** when you want the trend/cycle split *implied by the
  series' own estimated dynamics*. Expect a small, choppy cycle on US-GDP-
  like series — that is the honest answer of the freely estimated model
  (Stock-Watson 1988; MNZ 2003), not a bug. The `long_run_multiplier` is
  itself the economically interesting number (>1: shocks are amplified
  into the trend; <1: partly transitory).
- **KMW `bn_filter`** when you want an *output gap* — large, persistent,
  intuitive — while keeping the BN definition of trend. The pinned δ is a
  judgment (that trend shocks contribute a small share of forecast-error
  variance); the amplitude-to-noise criterion makes it data-driven but it
  remains a discipline imposed, not discovered. On the fixture's simulated
  drifting series the KMW cycle variance is **37.6×** the classic BN's —
  the paper's headline contrast, reproduced and asserted.

### Failure modes

- **`se=None` on the Hamilton regression is not neutral** — reading the
  plain OLS `beta` t-statistics off a hand-rolled covariance understates
  uncertainty badly at `h=8` (overlap correlation). Ask for `se="hac"`.
- **`bn_decomposition` refuses unit-circle fits.** An MA root numerically
  on the unit circle (the classic boundary pile-up, common when `q` is too
  generous and AR/MA roots nearly cancel — the fixture's simulated series
  does exactly this at `(2,2)`) makes the innovation recursion unreliable;
  the error says to lower `q`. Likewise a nonstationary AR (`φ(1) ≈ 0`
  after differencing usually means over-differencing).
- **`bn_filter` needs `p ≥ 2` and `n ≥ 2p+3`**, and its automatic δ search
  errors out (rather than walking forever) if the amplitude-to-noise
  ratio never peaks — impose a fixed `delta` there.
- **The KMW cycle depends on the demeaning choice.** `demean="sm"` (the
  baseline) attributes the full-sample mean growth to trend; on samples
  with a structural growth slowdown the authors' later work uses dynamic
  demeaning (not implemented here — the 2018 baseline is).

### Validated against (grades, with measured numbers)

- **`bn_filter` — grade: reference-run (R).** Pinned against actual runs
  of the authors' own replication code (bnfiltering.com lineage: Ben
  Wong's MATLAB, R conversion by Luke Hartigan, updated by James Morley —
  as packaged at `github.com/kletts/bnfilter@8af7924`, sourced at fixture
  generation, not vendored) at the KMW-2018 baseline options
  (`delta_select=1`, `ib=FALSE`, `d0=0.01`, `dt=0.0005`, fixed bands), on
  100·log US real GDP and a seeded simulated series, four cases spanning
  auto/fixed δ, sample-mean/no demeaning, `p ∈ {8, 12}`. Rust matches the
  R runs elementwise at **≤ 2.9e-15** (cycle) / **≤ 1.6e-15** (AR), the
  automatic δ lands on the identical grid point, and `cycle_se` /
  amplitude-to-noise are pinned at 1e-8. The generator additionally
  re-implements the whole procedure in NumPy and asserts agreement with R
  at 1e-9 before writing, so the stored numbers are simultaneously a
  reference run and a two-implementation cross-check. Honest caveats:
  (a) the packaged code is the authors' current (2022–2025-refined)
  lineage run at its 2018-baseline settings, not a bit-frozen 2017
  snapshot — it includes the shrinkage prior the refined code applies on
  all paths; (b) `kletts/bnfilter` is a re-packaging of the
  bnfiltering.com code, not the authors' own repository. US-GDP auto δ
  comes out 0.2295 on the macrodata sample (KMW report ≈ 0.24 on theirs).
- **`bn_decomposition` — grade: documented-formula transcription with a
  genuine statsmodels pin on ψ(1), plus exact identities.** statsmodels
  has no BN decomposition, so trend/cycle/innovations are pinned against
  an independent NumPy transcription of the Morley-2002 companion
  computation (three cases: MNZ ARIMA(2,1,2) coefficients on GDP, fixed
  ARMA(1,1) and AR(2) on the simulated series) — Rust matches at
  **≤ 2.4e-16**. The number that *defines* the decomposition, ψ(1), IS
  third-party checkable: it equals the cumulative sum of statsmodels'
  `arma_impulse_response`, asserted at generation (< 1e-8) and re-pinned
  in the crate and binding tests at 1e-7. The identities are asserted on
  the library's own output: `trend + cycle` reconstructs `y[1:]` (≤ 1
  ulp), `Δtrend = μ + ψ(1)·ε` at 1e-9, and ARIMA(0,1,1) reproduces the
  textbook `c_t = −θε_t`, `ψ(1) = 1+θ` exactly. The fit path (library
  MLE vs statsmodels MLE of the same spec) lands within **1.4e-4** on
  ψ(1) and **6.8e-6** on the drift for the GDP ARIMA(2,1,2).
- **`hamilton_filter` — grade: independent package (statsmodels), now on
  both legs.** Since statsmodels 0.15.0 the *decomposition* has a
  third-party reference it could not have had when it shipped: the
  version-gated canary test pins our `cycle`/`trend` against
  `tsa.filters.api.hamilton_filter` at **≤ 4.2e-14** max abs (asserted
  1e-10). The *inference* surface still has no counterpart to compare
  against — statsmodels' filter returns cycle/trend only — but the filter
  is literally OLS, so its coefficient inference is statsmodels territory
  anyway: `OLS(...).fit(cov_type="HAC", cov_kwds={"maxlags": …,
  "use_correction": …})` on the identical design pins `bse`/`tvalues`
  for nonrobust and three HAC settings (including the `maxlags = h = 8`
  default). Measured agreement **≤ 2.9e-8** (bse) / **≤ 6.8e-8**
  (tvalues), pinned at 1e-6 — the design is raw *levels* of a trending
  series, so the two solvers (statsmodels pinv vs `tsecon-hac` refined
  Cholesky) agree to ~1e-8 here rather than the engine's 1e-10 on its
  own calmer goldens. The decomposition and `beta` are asserted
  **bit-identical** with and without `se` (the defaults-unchanged
  guarantee), and `hamilton_filter(y)` still reproduces the original
  `fixtures/filters.json` golden.

### References

- Hamilton, J. D. (2018). "Why You Should Never Use the Hodrick-Prescott
  Filter." *Review of Economics and Statistics* 100(5), 831–843.
- Beveridge, S. & Nelson, C. R. (1981). "A New Approach to Decomposition of
  Economic Time Series into Permanent and Transitory Components…" *Journal
  of Monetary Economics* 7(2), 151–174.
- Morley, J. C. (2002). "A state-space approach to calculating the
  Beveridge-Nelson decomposition." *Economics Letters* 75(1), 123–127.
- Morley, J. C., Nelson, C. R. & Zivot, E. (2003). "Why Are the
  Beveridge-Nelson and Unobserved-Components Decompositions of GDP So
  Different?" *Review of Economics and Statistics* 85(2), 235–243.
- Kamber, G., Morley, J. & Wong, B. (2018). "Intuitive and Reliable
  Estimates of the Output Gap from a Beveridge-Nelson Filter." *Review of
  Economics and Statistics* 100(3), 550–566. Replication code:
  bnfiltering.com (R conversion by Luke Hartigan).
- Newey, W. K. & West, K. D. (1987). "A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix."
  *Econometrica* 55(3), 703–708.

### Runnable example

```python
import numpy as np
import tsecon

# A synthetic quarterly log-level series, built as the ARIMA(2,1,2)+drift that
# bn_decomposition fits by default: phi = (0.6, -0.2), theta = (0.3, 0.1), so
# the true long-run multiplier is psi(1) = theta(1)/phi(1) = 1.4/0.6 = 2.33.
n = 240
e = np.random.default_rng(11).standard_normal(n)
dy = np.zeros(n)
for t in range(2, n):
    dy[t] = (0.8 + 0.6 * dy[t - 1] - 0.2 * dy[t - 2]
             + e[t] + 0.3 * e[t - 1] + 0.1 * e[t - 2])
y = 700.0 + np.cumsum(dy)                  # quarterly log-level, 100*log units

ham = tsecon.hamilton_filter(y, se="hac")  # h=8, p=4, NW maxlags=8
print("beta_1 t-stat (HAC):", round(ham["tvalues"][1], 2))

classic = tsecon.bn_decomposition(y)       # ARIMA(2,1,2)+c, exact MLE
print("psi(1):", round(classic["long_run_multiplier"], 2), " (truth 2.33)",
      "| cycle sd:", round(np.std(classic["cycle"]), 2))

gap = tsecon.bn_filter(y)                  # KMW, p=12, auto delta
print("delta:", round(gap["delta"], 3),
      "| gap sd:", round(np.std(gap["cycle"]), 2),
      "| band:", round(1.96 * gap["cycle_se"], 2))
```

Expected output:

```
beta_1 t-stat (HAC): 4.58
psi(1): 2.38  (truth 2.33) | cycle sd: 1.35
delta: 0.504 | gap sd: 2.24 | band: 2.68
```

Read the three lines together. The Hamilton regression finds a real `y_{t−8}`
coefficient, and the HAC bandwidth is what keeps that t-statistic honest under
the overlapping-horizon residual — the same fit reports t = 5.26 with
`se="nonrobust"`, which is the number you should not quote. The classic BN
decomposition recovers the DGP's long-run multiplier (2.38 against a truth of
2.33), and that `psi(1) > 1` is exactly why its cycle is the small one: sd 1.35
against the BN *filter*'s 2.24 on the same series, the famous BN "tiny cycle"
reproduced on data whose ψ(1) we know. The KMW filter pins the signal-to-noise
ratio rather than inheriting it (`delta = 0.504` from the automatic search) and
returns a gap with a standard error, so the ±2.68 band is the one number the
classic decomposition cannot give you.
