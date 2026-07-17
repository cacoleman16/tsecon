# Model card — Realized volatility

**Family:** `realized_measures`, `har_rv`, `realized_quarticity`,
`tripower_quarticity`, `bns_jump_test`, `realized_range`

Measuring and forecasting volatility from high-frequency data. Where GARCH
*models* latent volatility from daily returns, realized measures *observe* it:
sum intra-day squared returns to estimate a day's integrated variance, use
jump-robust variants to separate the smooth diffusion from discrete jumps, test
whether a given day contained a jump, and forecast realized variance with the
HAR model that has become the field's workhorse. When only OHLC bars are
available, range estimators recover much of the same signal.

| Function | Role |
|----------|------|
| `realized_measures` | RV, bipower variation, and the jump component for one day |
| `realized_quarticity` | Integrated-quarticity estimate (RV's own standard error scale) |
| `tripower_quarticity` | Jump-robust quarticity |
| `bns_jump_test` | Barndorff-Nielsen-Shephard ratio jump test |
| `har_rv` | HAR-RV forecasting regression (Corsi 2009) |
| `realized_range` | Range-based variance from OHLC bars |

## What it estimates

- **`realized_measures(returns)`** — from one day of intraday returns: the
  **realized variance** RV = Σrᵢ² (a consistent estimate of the day's
  integrated variance plus jumps), the jump-robust **bipower variation** BV
  (integrated variance only), and the **jump** component max(RV − BV, 0).
- **`realized_quarticity(returns)`** — RQ = (n/3)Σrᵢ⁴, the integrated
  quarticity that sets the scale of RV's sampling error. **`tripower_quarticity`**
  is the jump-robust counterpart, used to make the jump test robust to jumps in
  the variance-of-variance.
- **`bns_jump_test(returns)`** — the Barndorff-Nielsen-Shephard ratio statistic
  (with the Huang-Tauchen 2005 refinement): a standardized measure of the gap
  between RV and BV; large positive values signal a jump occurred that day.
- **`har_rv(rv)`** — the Corsi (2009) Heterogeneous AutoRegressive model:
  regress RV_t on a constant and the **daily**, **weekly**, and **monthly**
  averages of past RV, with HAC standard errors. Its cascade of horizons
  captures RV's long memory with three regressors.
- **`realized_range(high, low)`** — the Parkinson (or Garman-Klass, given open
  and close) range estimator of variance from OHLC bars — far more efficient
  than close-to-close when intraday returns are unavailable.

## Assumptions

- **`realized_measures`, quarticity, and the jump test each take one day's
  intraday returns** — a 1-D array of the intra-period log returns (e.g. 78
  five-minute returns). They return per-day scalars; loop over days to build a
  series.
- **`har_rv` takes a *series* of daily RV**, not intraday returns. It needs at
  least ~a month of history (the monthly component averages 22 days) plus the
  `start` burn-in; `nobs` in the output reflects the usable rows.
- **Sampling frequency is a bias-variance tradeoff.** Too-fine sampling lets
  market-microstructure noise inflate RV; the classic 5-minute grid is a
  common compromise. These estimators assume you have already chosen a sensible
  grid — they do not implement noise-robust (two-scale / pre-averaging)
  corrections.
- **Jump separation is asymptotic.** RV − BV is a noisy jump proxy in finite
  samples; the `bns_jump_test` ratio is the disciplined way to decide whether a
  day's gap is a real jump rather than sampling noise.
- **Range estimators assume continuous trading and no drift** within the bar;
  Garman-Klass additionally uses the open and close and is more efficient when
  those are reliable.

## When to use

- **`realized_measures`** — the daily volatility proxy for any high-frequency
  dataset, and the input to HAR forecasting.
- **`bns_jump_test` (+ `tripower_quarticity`)** — to flag jump days before
  modeling, or to build a jump indicator / separate continuous and jump
  variation for a HAR-CJ style regression.
- **`realized_quarticity`** — to attach a standard error to RV or to construct
  the jump-test denominator by hand.
- **`har_rv`** — the default realized-volatility *forecast*: simple, robust,
  hard to beat, and interpretable (daily/weekly/monthly loadings).
- **`realized_range`** — when you only have OHLC bars (most historical equity
  and FX data), recovering most of the efficiency of true realized variance.

## Key arguments and defaults

| Call | Argument | Default | Notes |
|------|----------|---------|-------|
| `realized_measures` | `returns` | — | one day's intraday returns |
| `har_rv` | `rv` | — | a series of daily realized variance |
| | `start` | `22` | burn-in (needs the monthly window) |
| | `variant` | `"level"` | `"level"`, `"log"`, or `"sqrt"` |
| | `hac_maxlags` | `5` | Newey-West lags on the HAR SEs |
| | `use_correction` | `False` | small-sample HAC correction |
| `realized_range` | `method` | `"parkinson"` | or `"garman_klass"` (needs `open`, `close`) |
| | `open` / `close` | `None` | required for Garman-Klass |

## How to read the output

- **`realized_measures`** → `{"rv", "bipower", "jump"}`, all scalars in
  variance (squared-return) units. `jump = max(rv − bipower, 0)`; a `jump` of 0
  means no jump was detected that day. On a small quiet sample `bipower` can
  exceed `rv`, which is why the jump is floored at 0.
- **`realized_quarticity`, `tripower_quarticity`** → scalars.
- **`bns_jump_test`** → `{"ratio"}`. The `ratio` is (asymptotically standard
  normal under no jump) large and **positive** on jump days; compare to a normal
  quantile (e.g. > 1.96 flags a jump at 5%). Negative or small values indicate
  no jump.
- **`har_rv`** → `{"params", "bse", "tvalues", "rsquared", "nobs"}` with
  `params` ordered **[const, daily, weekly, monthly]** and HAC `bse`. The
  persistence shows up as positive daily+weekly+monthly loadings; `variant`
  controls whether the regression is in levels, logs, or square roots (logs
  keep RV positive and tame outliers).
- **`realized_range`** → a scalar variance.

## Failure modes

- **Feeding a daily RV series to `realized_measures`** (or intraday returns to
  `har_rv`). The two operate at different granularities:
  `realized_measures`/quarticity/`bns_jump_test` consume *one day of intraday
  returns*; `har_rv` consumes *a series of daily RV*.
- **Too-fine sampling.** Below a few minutes, microstructure noise biases RV
  upward without a noise-robust estimator (not implemented here). Stick to a
  ~5-minute grid or coarser unless you handle noise separately.
- **Reading RV − BV as a jump without the test.** The difference is noisy; use
  `bns_jump_test` to decide.
- **HAR in levels with heavy-tailed RV.** A few volatile days dominate the
  least-squares fit; `variant="log"` is the common, better-behaved default.
- **Garman-Klass without open/close.** It requires all of high/low/open/close;
  with only high and low, use Parkinson.

## Validated against

`har_rv` is validated as an OLS regression with Newey-West HAC SEs against
`statsmodels`; `realized_measures`, `realized_quarticity`,
`tripower_quarticity`, `bns_jump_test`, and `realized_range` reproduce the
documented Barndorff-Nielsen-Shephard (2002, 2004), Huang-Tauchen (2005),
Corsi (2009), Parkinson (1980), and Garman-Klass (1980) measure definitions.
Golden values are pinned in
[`fixtures/realized.json`](../../../fixtures/realized.json).

## References

- Barndorff-Nielsen, O. & Shephard, N. (2002). "Econometric analysis of
  realized volatility." *JRSS-B* 64.
- Barndorff-Nielsen, O. & Shephard, N. (2004). "Power and Bipower Variation
  with Stochastic Volatility and Jumps." *J. Financial Econometrics* 2.
- Huang, X. & Tauchen, G. (2005). "The Relative Contribution of Jumps to Total
  Price Variance." *J. Financial Econometrics* 3.
- Corsi, F. (2009). "A Simple Approximate Long-Memory Model of Realized
  Volatility." *J. Financial Econometrics* 7.
- Parkinson, M. (1980). "The Extreme Value Method for Estimating the Variance
  of the Rate of Return." *J. Business* 53.
- Garman, M. & Klass, M. (1980). "On the Estimation of Security Price
  Volatilities from Historical Data." *J. Business* 53.

See the guide: [Volatility: GARCH and Risk](../../guide/06-volatility.md).

## Runnable example

```python
import numpy as np
import tsecon

rng = np.random.default_rng(21)

# ---- one trading day of intraday returns (e.g. 78 five-minute log returns) ----
intraday = 0.001 * rng.standard_normal(78)

# 1. Realized variance, jump-robust bipower variation, and the jump component.
rm = tsecon.realized_measures(intraday)
print("RV:", format(rm["rv"], ".2e"), " bipower:", format(rm["bipower"], ".2e"),
      " jump:", format(rm["jump"], ".2e"))

# 2. Integrated-quarticity estimators (the scale for RV's own standard error).
print("RQ:", format(tsecon.realized_quarticity(intraday), ".2e"),
      " tripower (jump-robust):", format(tsecon.tripower_quarticity(intraday), ".2e"))

# 3. BNS ratio jump test on a day WITH an injected jump: a large positive ratio flags it.
jumpy = intraday.copy(); jumpy[40] += 0.02
print("BNS ratio, no jump:", round(tsecon.bns_jump_test(intraday)["ratio"], 3),
      " with jump:", round(tsecon.bns_jump_test(jumpy)["ratio"], 3))

# ---- a persistent daily realized-variance series ----
days = 500
rv = np.empty(days); rv[0] = 1e-4
for t in range(1, days):
    rv[t] = 0.6 * rv[t - 1] + 0.4 * abs(rng.standard_normal() * 1e-4) + 1e-6

# 4. HAR-RV (Corsi): RV_t on its daily, weekly, and monthly averages, HAC SEs.
har = tsecon.har_rv(rv, variant="log")
print("HAR params [const, daily, weekly, monthly]:", np.round(har["params"], 3),
      " R^2:", round(har["rsquared"], 3))

# 5. Range-based variance from OHLC bars (no intraday returns needed).
n = 250
high = 1 + 0.01 * np.abs(rng.standard_normal(n)); low = 1 - 0.01 * np.abs(rng.standard_normal(n))
op = 1 + 0.005 * rng.standard_normal(n); cl = 1 + 0.005 * rng.standard_normal(n)
print("Parkinson:", round(tsecon.realized_range(high, low), 4),
      " Garman-Klass:", round(tsecon.realized_range(high, low, method="garman_klass",
                                                    open=op, close=cl), 4))
```

Expected output:

```
RV: 6.34e-05  bipower: 7.53e-05  jump: 0.00e+00
RQ: 3.21e-09  tripower (jump-robust): 7.25e-09
BNS ratio, no jump: -1.877  with jump: 8.306
HAR params [const, daily, weekly, monthly]: [-4.197  0.641  0.03  -0.112]  R^2: 0.419
Parkinson: 0.0284  Garman-Klass: 0.0353
```
