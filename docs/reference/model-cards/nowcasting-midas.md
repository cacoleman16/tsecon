# Model card — Nowcasting and mixed frequencies

**Family:** `dfm_nowcast`, `dfm_news`, `midas_weights`, `umidas`,
`weighted_midas`

Reading the economy in real time. The target (quarterly GDP) arrives late and
rarely; the indicators (monthly, weekly, daily) arrive early and often, on a
staggered "ragged edge." Two complementary toolkits handle this: **MIDAS**
regressions map many high-frequency lags onto one low-frequency target through
a parsimonious weight function, and **dynamic factor models** extract a common
factor from a whole panel and let the Kalman filter fill the ragged edge. The
`dfm_news` decomposition then attributes each nowcast revision to the specific
data release that caused it.

| Function | Role |
|----------|------|
| `midas_weights` | The MIDAS weight kernels (exp-Almon, beta) |
| `umidas` | Unrestricted MIDAS — one OLS coefficient per HF lag |
| `weighted_midas` | Parsimonious MIDAS — NLS fit of a weight shape |
| `dfm_nowcast` | Two-step dynamic-factor nowcast with a ragged edge |
| `dfm_news` | Decompose a nowcast revision into per-datapoint news |

## What it estimates

- **`midas_weights(scheme, θ₁, θ₂, k)`** — the k MIDAS weights that sum to 1,
  either the two-parameter exponential-Almon or beta kernel. These are the
  shapes that let a handful of parameters describe many high-frequency lags.
- **`umidas(y, hf_lags)`** — the unrestricted MIDAS regression: with the
  frequency mismatch modest, just put each high-frequency lag in as its own OLS
  regressor. No weight function, but many parameters.
- **`weighted_midas(y, hf_lags)`** — the restricted MIDAS: fit an intercept, a
  single slope, and the two weight-shape parameters by nonlinear least squares,
  so the high-frequency lag polynomial is described by four numbers regardless
  of how many lags there are.
- **`dfm_nowcast(data)`** — a two-step (Doz-Giannone-Reichlin 2011) dynamic
  factor model: PCA for the loadings, a Kalman filter/smoother for the factor
  that handles missing cells, and a projection back onto each series to produce
  the nowcast at the ragged edge.
- **`dfm_news(old_vintage, new_vintage)`** — the Banbura-Modugno (2014) news
  decomposition: it splits the change in the target nowcast between two data
  vintages into a sum of *weight × news* contributions, one per newly released
  datapoint, so you can say "the nowcast rose 0.3pp, mostly on the payrolls
  release."

## Assumptions

- **DFM: a single common factor drives comovement.** The panel should be
  genuinely co-moving; series with idiosyncratic dynamics contribute little and
  can be dropped. Data are treated as standardized internally, but you are
  responsible for transforming each series to stationarity first (growth rates,
  differences).
- **Ragged edge = NaN.** Missing/not-yet-released cells are encoded as `NaN`
  and handled by the Kalman measurement update using only observed rows. The
  ragged edge belongs at the *end* (recent periods), which is where releases
  lag.
- **`dfm_news` requires nested vintages.** Every cell observed in
  `old_vintage` must still be observed (and unchanged) in `new_vintage` — the
  new vintage is a *superset* of the old. News is the surprise in *newly
  revealed* cells relative to the model's prior forecast of them; a changed or
  vanished old value violates the decomposition and raises.
- **MIDAS weights sum to one**, so scale is carried by the slope. The
  exp-Almon and beta kernels are flexible but unimodal — they cannot represent
  arbitrary lag shapes; that is the price of parsimony (and the reason `umidas`
  exists).
- **`weighted_midas` is a nonlinear fit** and can be sensitive to starting
  values in poorly identified problems; it reports whether it converged.

## When to use

- **`umidas`** when the frequency ratio is small (monthly→quarterly, k a
  handful) and you have enough data to estimate one coefficient per lag.
- **`weighted_midas`** when the ratio is large (daily→quarterly) or lags are
  many, so an unrestricted regression would overfit — the weight function
  regularizes the lag polynomial.
- **`midas_weights`** to inspect or plot the kernel shapes, or to build a MIDAS
  regressor by hand.
- **`dfm_nowcast`** when you have a *panel* of indicators and want one nowcast
  that pools their signal and respects the ragged edge.
- **`dfm_news`** to explain *why* a nowcast moved between two release dates —
  the standard "what did we learn from today's data?" decomposition.

## Key arguments and defaults

| Call | Argument | Default | Notes |
|------|----------|---------|-------|
| `midas_weights` | `scheme`, `θ₁`, `θ₂`, `k` | — | `"exp_almon"` or `"beta"`; k lags |
| `umidas` | `se_type` | `"hac"` | robust SEs on the OLS coefficients |
| | `maxlags` | `None` | HAC bandwidth (auto if `None`) |
| `weighted_midas` | `scheme` | `"exp_almon"` | or `"beta"` |
| | `weight_start` | `None` | `(θ₁, θ₂)` NLS starting values |
| `dfm_nowcast` | `n_factors` | `1` | common factors extracted |
| | `factor_order` | `2` | AR order of the factor process |
| `dfm_news` | `target_series` | `0` | column index of the nowcast target |
| | `target_period` | `None` | period to nowcast (default: last) |
| | `n_factors` / `factor_order` | `1` / `2` | as in `dfm_nowcast` |

## How to read the output

- **`midas_weights`** → a bare array of length k summing to 1.
- **`umidas`** → `{"params", "bse", "rsquared"}`; `params` has one intercept
  plus one coefficient per high-frequency lag (length K+1), with HAC `bse`.
- **`weighted_midas`** → `{"scheme", "intercept", "slope", "weight_params",
  "weights", "fitted", "residuals", "ssr", "rsquared", "converged",
  "iterations"}`. Check `converged`; read the estimated lag shape off
  `weights`.
- **`dfm_nowcast`** → `{"nowcast", "edge_factor", "loglik",
  "smoothed_factors", "n_factors", "factor_order"}`. `nowcast` is one value per
  series at the ragged edge (in the series' own standardized-then-restored
  units); `smoothed_factors` is the factor path.
- **`dfm_news`** → `{"target_series", "target_period", "old_nowcast",
  "new_nowcast", "total_revision", "contributions"}`. `total_revision =
  new_nowcast − old_nowcast`, and `contributions` is a list of dicts
  (`series`, `period`, `actual`, `forecast`, `news`, `weight`, `contribution`)
  whose `contribution` values sum to `total_revision`. Sort by `abs(contribution)`
  to find the release that moved the nowcast.

## Failure modes

- **Non-stationary inputs to the DFM.** Feeding levels (nominal GDP, price
  indices) instead of growth rates produces a factor dominated by trends and a
  meaningless nowcast. Transform to stationarity first.
- **Ragged edge in the wrong place, or non-NaN missings.** Missing cells must
  be `NaN`; a sentinel like 0 or −999 will be read as data.
- **Non-nested vintages in `dfm_news`.** If the new vintage drops or changes a
  cell the old one had, the call raises ("a cell observed in the old vintage is
  missing or changed in the new vintage"). Build the new vintage by *adding*
  releases to the old, never by revising past values in the same call.
- **Unrestricted MIDAS overfitting.** With a large frequency ratio, `umidas`
  has too many parameters and fits noise; switch to `weighted_midas`.
- **`weighted_midas` not converging.** Poorly identified weight shapes can stall
  the NLS; supply `weight_start` near a sensible decay, and check `converged`
  before trusting the fit.

## Validated against

The DFM Kalman/state-space step is reference-exact (~1e-8) against a documented
single-factor dynamic-factor model in the `statsmodels` `DynamicFactor` layout;
the two-step nowcast and the Banbura-Modugno (2014) news decomposition are
documented-formula goldens (an independent NumPy Kalman smoother), with the
contributions verified to sum to the total revision. U-MIDAS is validated as
OLS against a documented reference, and the exp-Almon / beta weight formulas
and the weighted-MIDAS NLS against their published definitions. Fixtures:
[`fixtures/tsecon-nowcast.json`](../../../fixtures/tsecon-nowcast.json),
[`fixtures/nowcast_news.json`](../../../fixtures/nowcast_news.json),
[`fixtures/midas.json`](../../../fixtures/midas.json).

## References

- Ghysels, E., Santa-Clara, P. & Valkanov, R. (2004). "The MIDAS touch: Mixed
  data sampling regression models." Working paper.
- Ghysels, E., Sinko, A. & Valkanov, R. (2007). "MIDAS Regressions: Further
  Results and New Directions." *Econometric Reviews* 26.
- Doz, C., Giannone, D. & Reichlin, L. (2011). "A two-step estimator for large
  approximate dynamic factor models." *J. Econometrics* 164.
- Bańbura, M. & Modugno, M. (2014). "Maximum Likelihood Estimation of Factor
  Models on Datasets with Arbitrary Pattern of Missing Data." *J. Applied
  Econometrics* 29.
- Bańbura, Giannone, Modugno & Reichlin (2013). "Now-Casting and the Real-Time
  Data Flow." *Handbook of Economic Forecasting* 2.

See the guide: [Nowcasting and Mixed Frequencies](../../guide/11-nowcasting.md).

## Runnable example

```python
import numpy as np
import tsecon

rng = np.random.default_rng(7)

# ---------------------------- MIDAS ----------------------------
# 1. The weight kernels themselves (they sum to 1).
w = tsecon.midas_weights("exp_almon", 0.0, -0.2, 6)
print("exp-Almon weights sum to:", round(w.sum(), 6))

# Build a mixed-frequency regression: low-frequency y on K high-frequency lags.
nobs, K = 120, 6
HF = rng.standard_normal((nobs, K))
y = 0.5 + 2.0 * (HF @ w) + 0.3 * rng.standard_normal(nobs)

# 2. U-MIDAS: unrestricted, one OLS coefficient per high-frequency lag.
um = tsecon.umidas(y, HF, se_type="hac")
print("U-MIDAS R^2:", round(um["rsquared"], 3), " params:", um["params"].shape[0])

# 3. Weighted MIDAS: parsimonious NLS fit of the weight shape.
wm = tsecon.weighted_midas(y, HF, scheme="exp_almon")
print("weighted-MIDAS R^2:", round(wm["rsquared"], 3),
      " converged:", wm["converged"])

# ------------------------ DFM nowcasting ------------------------
T, Nser = 150, 5
f = np.cumsum(rng.standard_normal(T)) * 0.3
data = np.outer(f, rng.uniform(0.5, 1.5, Nser)) + rng.standard_normal((T, Nser)) * 0.5

# 4. Nowcast from a panel with a ragged edge (NaN = not-yet-released).
new_v = data.copy()
new_v[-1, 3] = np.nan; new_v[-1, 4] = np.nan          # two series lag by a period
now = tsecon.dfm_nowcast(new_v, n_factors=1, factor_order=1)
print("nowcast (5 series):", np.round(now["nowcast"], 2))

# 5. News decomposition: attribute a nowcast revision to each new datapoint.
old_v = new_v.copy()
old_v[-1, 0] = np.nan; old_v[-1, 1] = np.nan; old_v[-1, 2] = np.nan  # older, sparser vintage
old_v[-2, 3] = np.nan; old_v[-2, 4] = np.nan
news = tsecon.dfm_news(old_v, new_v, target_series=0, n_factors=1, factor_order=1)
print("revision:", round(news["total_revision"], 3),
      " from", len(news["contributions"]), "datapoints")
top = max(news["contributions"], key=lambda c: abs(c["contribution"]))
print("largest contribution from series", top["series"],
      "=", round(top["contribution"], 4))
```

Expected output:

```
exp-Almon weights sum to: 1.0
U-MIDAS R^2: 0.936  params: 7
weighted-MIDAS R^2: 0.934  converged: True
nowcast (5 series): [7.57 3.47 2.73 4.15 3.46]
revision: 0.088  from 5 datapoints
largest contribution from series 4 = 0.0759
```
