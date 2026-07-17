# Model card — Diagnostics and the stationarity workflow

**Family:** `acf`, `pacf`, `ljung_box`, `jarque_bera`, `arch_lm`, `adf`, `kpss`,
`check_stationarity`

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
