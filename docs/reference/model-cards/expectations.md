# Model card — Survey expectations

`cg_regression` · `forecast_efficiency` · `forecast_disagreement`

Tools for the survey-expectations literature: they take a panel of forecasts —
or the mean forecast and the realized outcome — and ask whether forecasters use
information efficiently, and how much they disagree. The first two are ordinary
least squares with a Newey-West (HAC) covariance, because forecast errors are
serially correlated by construction (overlapping horizons); the third is a
purely descriptive cross-sectional dispersion measure. Reach for this family
when you have survey expectations (SPF, Michigan, Consensus, central-bank
surveys) and want to test rationality or measure information rigidity.

---

## `cg_regression` — Coibion-Gorodnichenko information-rigidity regression

**What it estimates.** The slope of the *mean forecast error* on the *mean
forecast revision* (Coibion & Gorodnichenko 2015):
`error_t = c + beta·revision_t + u_t`, fit by OLS with a HAC covariance. Under
full-information rational expectations the current revision carries no
information about the future error, so `beta = 0`. A **positive** slope is the
signature of sticky or noisy information — forecasters underreact, so revisions
predict errors. The estimator also reports the **implied degree of information
rigidity** `implied_rigidity = slope / (1 + slope)`, the fraction of agents who
do not update (sticky-information λ) or the Kalman-gain complement (noisy
information) — both maps give the same `beta/(1+beta)`.

**Assumptions.** The error and revision are the *consensus* (mean across
forecasters) series, aligned so that `error_t` is the outcome minus the
forecast and `revision_t` is this period's forecast minus last period's, both
for the same fixed horizon. Errors are serially correlated (overlapping
forecasts) but stationary with summable autocovariances — that is exactly what
the HAC covariance is there to handle. The implied-rigidity map assumes the
sticky/noisy-information model that motivates the regression.

**When to use (and when not).** Use it to test the full-information rational
expectations null on consensus forecasts and to quantify underreaction. It is a
*consensus-level* test: pooling individual forecasters instead tends to reveal
the opposite sign (overreaction) and is a different exercise. Do not read a
positive slope as proof of any single micro-founded model — sticky information,
noisy information, and rational inattention all imply `beta > 0`.

**Key arguments and defaults (and why).** `maxlags=None` picks the Newey-West
rule-of-thumb bandwidth `floor(4·(n/100)^(2/9))` (reported back in `maxlags`);
set an integer to match a target horizon (a common choice is the forecast
horizon in periods). `use_correction=True` applies the statsmodels small-sample
`T/(T−k)` scaling to the HAC covariance — leave it on to match statsmodels'
default, turn it off for the textbook asymptotic form.

**How to read the output.** `slope` is `beta` (the object of interest) with its
HAC `se_slope`, `t_slope`, `p_slope`; `intercept`/`se_intercept` are the mean
bias. `implied_rigidity` translates the slope into the fraction of stale
information. `r_squared`, `maxlags` (the bandwidth actually used), and `nobs`
round it out. A significantly positive `slope` rejects FIRE toward
underreaction; a slope indistinguishable from zero is consistent with full
information.

**Failure modes.** Reading `implied_rigidity` when the slope is negative (the
formula still returns a number, but the sticky/noisy-information story does not
apply); using individual rather than consensus forecasts and misinterpreting
the sign; too small a `maxlags` for a long overlapping horizon, which
understates the standard errors.

**Validated against.** statsmodels `OLS(...).fit(cov_type="HAC",
cov_kwds={"maxlags": L, "use_correction": ...}, use_t=False)` — the same OLS-HAC
estimand from an independent implementation, matched to ~1e-7; `implied_rigidity
= slope/(1+slope)` is a documented closed form (`fixtures/tsecon-survey.json`).

**References.** Coibion & Gorodnichenko (2015, *AER* 105:2644-2678); Newey &
West (1987, HAC).

```python
import numpy as np, tsecon

rng = np.random.default_rng(0)
n = 200
# Positively-autocorrelated revision; persistent error that loads on it.
rev = np.zeros(n); u = np.zeros(n)
for t in range(1, n):
    rev[t] = 0.5 * rev[t - 1] + rng.normal(0.0, 1.0)
    u[t]   = 0.6 * u[t - 1] + rng.normal(0.0, 0.8)
err = 0.1 + 0.7 * rev + u                        # true CG slope = 0.7

cg = tsecon.cg_regression(err, rev)              # maxlags=None -> Newey-West rule
print(f"slope {cg['slope']:.3f}  (HAC se {cg['se_slope']:.3f}, p {cg['p_slope']:.4f}), "
      f"maxlags {cg['maxlags']}")
print(f"implied information rigidity = {cg['implied_rigidity']:.3f}")
# slope 0.699  (HAC se 0.085, p 0.0000), maxlags 4
# implied information rigidity = 0.411
```

---

## `forecast_efficiency` — Mincer-Zarnowitz rationality test

**What it estimates.** A Mincer-Zarnowitz efficiency (rationality) test in
error-on-regressor form: regress the forecast error on a constant and one or
more `regressors` — typically the forecast itself, but any information set known
at forecast time — then jointly test by a HAC Wald statistic that **all**
coefficients (intercept and slopes) are zero. Rational forecasts leave errors
that are mean-zero and unpredictable from information available when the
forecast was made, so a rational forecaster produces coefficients that are all
zero.

**Assumptions.** The regressors are in the forecaster's information set at the
time of the forecast (otherwise a non-zero coefficient is not evidence of
irrationality). Errors are serially correlated but stationary — the HAC
covariance handles the overlap. `regressors` is a `T×k` array (one column per
conditioning variable); a constant is added internally, so do not include a
column of ones.

**When to use (and when not).** Use it to test unbiasedness and efficiency of a
forecast against a chosen information set. The classic MZ specification
regresses the *outcome* on the forecast and tests `(intercept, slope) = (0, 1)`;
this crate uses the equivalent *error*-on-forecast form and tests all
coefficients equal to zero, which is algebraically the same restriction and
avoids a non-standard hypothesis vector. Do not include regressors the
forecaster could not have seen, and remember a rejection tells you the forecast
is inefficient with respect to *these* regressors, not why.

**Key arguments and defaults (and why).** `maxlags=None` and
`use_correction=True` mean the same Newey-West bandwidth rule and small-sample
scaling as `cg_regression`. Choose `regressors` deliberately: the forecast alone
gives the canonical MZ test; adding lagged errors or macro variables tests
efficiency against a richer information set.

**How to read the output.** `wald` is the chi-square statistic with `wald_df`
degrees of freedom (equal to the number of coefficients, `k+1`) and
`wald_pvalue`; a small p-value **rejects** rationality. `params` are the
coefficients (intercept first, then the regressor slopes) with `bse`, `tvalues`,
`pvalues` for a per-coefficient look at where any inefficiency comes from, plus
`r_squared`. A large `wald_pvalue` — as below — is consistent with an efficient
forecast.

**Failure modes.** Including a column of ones in `regressors` (the constant is
already added, so this makes the design rank-deficient); testing against
information the forecaster lacked; too short a HAC bandwidth for long overlapping
horizons, which shrinks the Wald p-value spuriously.

**Validated against.** statsmodels — the OLS-HAC coefficients/standard errors
and the `wald_test` chi-square from the same covariance, matched to ~1e-7
(`fixtures/tsecon-survey.json`).

**References.** Mincer & Zarnowitz (1969); Newey & West (1987, HAC).

```python
import numpy as np, tsecon

rng = np.random.default_rng(1)
n = 200
fc = np.cumsum(rng.normal(0.0, 1.0, n)) * 0.3       # the forecasts
e  = np.zeros(n)
for t in range(1, n):
    e[t] = 0.4 * e[t - 1] + rng.normal(0.0, 1.0)     # near-rational error
err = 0.15 + 0.05 * fc + e

mz = tsecon.forecast_efficiency(err, fc[:, None])    # regressors is T x k
print(f"Wald = {mz['wald']:.2f} on {mz['wald_df']} df, p = {mz['wald_pvalue']:.3f}")
print("coeffs [const, forecast]:", np.round(mz["params"], 3))
# Wald = 2.53 on 2 df, p = 0.282
# coeffs [const, forecast]: [-0.178 -0.039]
```

---

## `forecast_disagreement` — cross-sectional dispersion of a forecaster panel

**What it estimates.** Per-period disagreement across forecasters: given a
`panel` (a list of cross-sections, one array of individual forecasts per period,
ragged allowed), it returns the standard deviation, the 25/50/75 percentiles,
and the interquartile range of the forecasts *within each period*. Disagreement
is the standard empirical proxy for the dispersion of beliefs and, in
sticky/noisy-information models, moves with the degree of information rigidity.

**Assumptions.** Purely descriptive — no distributional assumption. Each
period's cross-section is treated independently; periods may have different
numbers of forecasters (the ragged panel is supported and `counts` records the
size used).

**When to use (and when not).** Use it to build a disagreement time series from
individual survey responses, to pair with `cg_regression` (theory ties the two
together), or as an uncertainty proxy in a downstream regression. It measures
*dispersion across forecasters*, not the *uncertainty of any one* forecaster
(which needs density forecasts); do not conflate the two.

**Key arguments and defaults (and why).** `ddof=1` gives the **sample** standard
deviation (divide by `count − 1`), the natural default for a finite panel of
forecasters; set `ddof=0` for the population standard deviation (numpy's
`np.std` default). The quartiles and IQR use numpy's linear-interpolation
percentiles and are unaffected by `ddof`.

**How to read the output.** Each of `std`, `p25`, `p50`, `p75`, `iqr` is an
array with one entry per period; `counts` is the number of forecasters in each
period. Rising `std` or `iqr` over time means widening disagreement. The IQR is
the robust companion to `std` — prefer it when a few extreme forecasters would
otherwise dominate the standard deviation.

**Failure modes.** Setting `ddof` at least as large as a period's cross-section
size (the `count − ddof` divisor would be non-positive — the call raises); a
single-forecaster period with `ddof=1` (zero degrees of freedom); reading `std`
as forecaster-level uncertainty rather than cross-forecaster spread.

**Validated against.** numpy — `np.std(ddof=...)` for the standard deviation and
`np.percentile(method="linear")` for the quartiles, with `iqr = p75 − p25` a
documented closed form (`fixtures/tsecon-survey.json`).

**References.** Mankiw, Reis & Wolfers (2004, disagreement); Zarnowitz &
Lambros (1987).

```python
import numpy as np, tsecon

rng = np.random.default_rng(2)
# Three periods of ~40 forecasters, each period more dispersed than the last.
panel = [rng.normal(2.0, s, 40) for s in (0.4, 0.8, 1.5)]
dis = tsecon.forecast_disagreement(panel)            # ddof=1 (sample std)
print("std:", np.round(dis["std"], 3))
print("IQR:", np.round(dis["iqr"], 3), " counts:", list(dis["counts"]))
# std: [0.38  0.862 1.437]
# IQR: [0.506 1.38  2.09 ]  counts: [40, 40, 40]
```
