# Model card — Nowcasting and mixed frequencies

**Family:** `dfm_nowcast`, `dfm_smooth`, `dfm_news`, `midas_weights`,
`umidas`, `adl_midas`, `weighted_midas`

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
| `adl_midas` | ADL-MIDAS — U-MIDAS plus lags of the target itself (0.11) |
| `weighted_midas` | Parsimonious MIDAS — NLS fit of a weight shape |
| `dfm_nowcast` | Two-step dynamic-factor nowcast with a ragged edge |
| `dfm_smooth` | Re-smooth any data vintage at fixed DFM parameters (0.11) |
| `dfm_news` | Decompose a nowcast revision into per-datapoint news |

## What it estimates

- **`midas_weights(scheme, θ₁, θ₂, k)`** — the k MIDAS weights that sum to 1,
  either the two-parameter exponential-Almon or beta kernel. These are the
  shapes that let a handful of parameters describe many high-frequency lags.
- **`umidas(y, hf_lags)`** — the unrestricted MIDAS regression: with the
  frequency mismatch modest, just put each high-frequency lag in as its own OLS
  regressor. No weight function, but many parameters.
- **`adl_midas(y, hf_lags, ar_lags=1)`** — the autoregressive-distributed-lag
  MIDAS (Andreou, Ghysels & Kourtellos 2013): `umidas` plus `ar_lags` lags of
  the low-frequency target itself, `y_t = c + Σ_p ρ_p y_{t−p} + Σ_k b_k
  hf_lags[t, k−1] + u_t`, fitted by OLS over the rows `t = P..n−1` (the first
  `P = ar_lags` rows serve only as lags). Same `(y, hf_lags)` pair as
  `umidas`; the lagged-`y` columns are built from `y`.
- **`weighted_midas(y, hf_lags)`** — the restricted MIDAS: fit an intercept, a
  single slope, and the two weight-shape parameters by nonlinear least squares,
  so the high-frequency lag polynomial is described by four numbers regardless
  of how many lags there are.
- **`dfm_nowcast(data)`** — a two-step (Doz-Giannone-Reichlin 2011) dynamic
  factor model: PCA for the loadings, a Kalman filter/smoother for the factor
  that handles missing cells, and a projection back onto each series to produce
  the nowcast at the ragged edge.
- **`dfm_smooth(data, loadings, factor_ar, idiosyncratic, factor_cov=None,
  center=None, scale=None)`** — the Kalman filter/smoother of the dynamic factor
  model at **given** parameters, no estimation: the standardized panel
  `z = (data − center)/scale`, `z_t = Λf_t + e_t`, `f_t = A_1 f_{t−1} + … +
  A_p f_{t−p} + η_t`, stationary initialization — statsmodels'
  `DynamicFactor(error_order=0)` exactly (which fixes `factor_cov` to the
  identity, as `factor_cov=None` does). It is the vintage re-smoothing step of
  a nowcasting round: estimate once with `dfm_nowcast`, then re-smooth every
  new release with the fitted model's six keys.
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
- **ADL-MIDAS alignment.** Row `t` of `hf_lags` belongs to low-frequency
  period `t`, most recent first: column 1 is the **last** high-frequency
  observation inside period `t` (high-frequency lag 0), column 2 the one
  before it, and so on — the convention of the crate's stacked-design builder.
  That is why `adl_midas` names the coefficients `hf1..hfK`, not
  `hf.L1..hf.LK`: an `L1` would claim a lag the column does not have.
- **`dfm_smooth` takes the parameters as true.** It is a filter, not an
  estimator: standard errors of the parameters do not enter the smoothed
  state covariances, and `factor_ar` must describe a stationary factor VAR
  (the stationary initialization needs one; a non-stationary one is refused).
- **`weighted_midas` is a nonlinear fit** and can be sensitive to starting
  values in poorly identified problems; it reports whether it converged.

## When to use

- **`umidas`** when the frequency ratio is small (monthly→quarterly, k a
  handful) and you have enough data to estimate one coefficient per lag.
- **`adl_midas`** whenever the target is persistent — quarterly GDP growth,
  inflation. Without its own lags a MIDAS regression lets the high-frequency
  indicator take credit for the target's autocorrelation; plain MIDAS is a
  straw-man benchmark for macro series.
- **`weighted_midas`** when the ratio is large (daily→quarterly) or lags are
  many, so an unrestricted regression would overfit — the weight function
  regularizes the lag polynomial.
- **`midas_weights`** to inspect or plot the kernel shapes, or to build a MIDAS
  regressor by hand.
- **`dfm_nowcast`** when you have a *panel* of indicators and want one nowcast
  that pools their signal and respects the ragged edge.
- **`dfm_smooth`** between re-estimations: re-smooth each new data vintage at
  the parameters of the last fit (the institutional routine — parameters are
  re-estimated monthly or quarterly, the smoother runs at every release), or
  smooth at parameters estimated elsewhere (a statsmodels `DynamicFactor`
  fit, a published model).
- **`dfm_news`** to explain *why* a nowcast moved between two release dates —
  the standard "what did we learn from today's data?" decomposition.

## Key arguments and defaults

| Call | Argument | Default | Notes |
|------|----------|---------|-------|
| `midas_weights` | `scheme`, `θ₁`, `θ₂`, `k` | — | `"exp_almon"` or `"beta"`; k lags |
| `umidas` | `se_type` | `"hac"` | robust SEs on the OLS coefficients |
| | `maxlags` | `None` | HAC bandwidth (auto if `None`) |
| `adl_midas` | `ar_lags` | `1` | lags of the target; `0` is refused (that model is `umidas`) |
| | `se_type` | `"hac"` | Bartlett/Newey-West with the `n/(n−k)` factor, or `"nonrobust"` |
| | `maxlags` | `None` | `floor(4 (nobs/100)^(2/9))` on the **fitted** rows `nobs = n − P`; refused under `"nonrobust"` |
| `weighted_midas` | `scheme` | `"exp_almon"` | or `"beta"` |
| | `weight_start` | `None` | `(θ₁, θ₂)` NLS starting values |
| `dfm_nowcast` | `n_factors` | `1` | common factors extracted |
| | `factor_order` | `2` | AR order of the factor process |
| `dfm_smooth` | `factor_cov` | `None` | identity (statsmodels' normalization) |
| | `center` / `scale` | `None` | zeros / ones: smooth the raw panel |
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
- **`dfm_nowcast`** → `{"nowcast", "edge_factor", "loglik", "fit_loglik",
  "smoothed_factors", "n_factors", "factor_order", "loadings", "factor_ar",
  "factor_cov", "idiosyncratic", "center", "scale"}`. `nowcast` is one value per
  series at the ragged edge (in the series' own standardized-then-restored
  units); `smoothed_factors` is the factor path; `loglik` is the filtering
  pass's log-likelihood on the full ragged panel, while `fit_loglik` is the
  estimation pass's log-likelihood on the balanced sub-panel (the rows before
  the first ragged edge — both `method` routes estimate there), so the two
  differ whenever the panel actually has a ragged edge.

  The last six keys (0.6 — previously computed in Rust but never bound) are
  **the fitted model itself**, on both `method` routes: `loadings` (Λ, N×r),
  `factor_ar` (r×rp, the stacked factor-VAR blocks `[A_1 | … | A_p]`),
  `factor_cov` (Q, r×r; fixed to 1 on the `"mle"` route for identification),
  `idiosyncratic` (length N — the diagonal of the observation covariance H),
  and `center`/`scale` (length N — the training column means and ddof-0
  standard deviations; `scale` is all ones on the `"mle"` route, which fits
  the centred raw scale with the loadings carrying the factor scale). The
  factor-to-series mapping is exact and pinned in the test suite:
  `nowcast == center + scale * (loadings @ edge_factor)` (that is literally
  how the binding computes it), and
  `center + scale * (smoothed_factors @ loadings.T)` is the common-component
  fit of the balanced panel — its per-series standardized residual variance
  is what `idiosyncratic` estimates. With these keys the state space
  `z_t = Λf_t + e_t`, `f_t = A_1 f_{t−1} + … + A_p f_{t−p} + η_t`,
  `e ~ N(0, diag(idiosyncratic))`, `η ~ N(0, factor_cov)` is fully
  reproducible from the results dict alone — and re-smoothable with
  `dfm_smooth` (below). With `method="mle"`
  the dict additionally carries `converged` and `iterations`: `converged` is
  the certificate of the optimizer stage whose point is reported (the BFGS
  polish's gradient-norm test when the polish improves on the Nelder-Mead
  optimum — the usual case — else Nelder-Mead's simplex-tolerance test), and
  `iterations` is the total across both stages; `converged=False` means the
  reported parameters are the best point found within the budget, not a
  certified optimum. The default `method="two_step"` runs no iterative
  optimizer (PCA + OLS + one Kalman pass are closed-form), so it carries
  neither key — the absence is the honest statement that a convergence
  certificate does not apply, not a gap.
- **`dfm_smooth`** → `{"loglik", "smoothed_factors", "smoothed_state",
  "smoothed_state_cov", "edge_factor", "nowcast", "n_factors",
  "factor_order"}`. `smoothed_factors` is a `(T, r)` ndarray over the **full**
  panel — not the same object as `dfm_nowcast`'s `smoothed_factors`, a
  `(T_b, r)` nested list from the balanced training block only;
  `smoothed_state` (`(T, rp)`) is the whole lag-stacked state and
  `smoothed_state_cov` (`(T, rp, rp)`) its posterior covariance, whose leading
  `r × r` block is the factors' uncertainty (it is largest at the ragged
  edge). `edge_factor` is the last row of `smoothed_factors` and
  `nowcast = center + scale * (loadings @ edge_factor)` — the model's level of
  every series at the last period, observed or not. `loglik` is the Gaussian
  log-likelihood of the **standardized** panel `z`, with no Jacobian term for
  the standardization (subtract `Σ_i n_obs,i · ln scale_i` to put it on the
  raw scale). Re-smoothing a fitted `dfm_nowcast` on its own panel returns its
  `loglik`, `edge_factor` and `nowcast` **bit for bit**: `dfm_nowcast`
  computes its edge nowcast through the same routine.
- **`adl_midas`** → `{"params", "param_names", "bse", "tvalues", "rsquared",
  "nobs", "residuals", "first_index", "se_type", "maxlags"}`, with
  `param_names = ["const", "y.L1".."y.LP", "hf1".."hfK"]`, `nobs = n − P`,
  `residuals` aligned with `y[P:]` (`first_index = P` is the index of the
  first fitted row) and `maxlags` the HAC bandwidth actually used (`None`
  under `"nonrobust"`).
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
- **ADL-MIDAS bandwidth from the wrong row count.** `umidas` applies the
  Newey-West rule to its own rows; `adl_midas` loses `P` rows to the lags and
  applies the rule to the rows it fits. At `n = 275`, `P = 4` that is 4 lags,
  not 5 — the case the golden uses to catch a copied rule.
- **Reading `dfm_smooth`'s `loglik` on the raw scale.** It is the
  standardized panel's likelihood; two calls with different `scale` are not
  comparable without the Jacobian term above.
- **Unrestricted MIDAS overfitting.** With a large frequency ratio, `umidas`
  has too many parameters and fits noise; switch to `weighted_midas`.
- **`weighted_midas` not converging.** Poorly identified weight shapes can stall
  the NLS; supply `weight_start` near a sensible decay, and check `converged`
  before trusting the fit.

**What it costs.** `dfm_smooth` keeps sequences of `T × (rp)²` covariance
cells and the dense `N × N` observation covariance, each capped at 2^26
cells, and its work is about `T (2m³ + 20Nm² + N² + 1000N + 12000) + 8m³`
units (`m = rp`; the one-off term is the stationary initialization), capped
at 2^34. A unit measured 0.38–1.19 ns across fourteen shapes, so a call at the
cap takes about 20 s at most: 12.1 s at `T = 8000`, `N = 1000`, `m = 2` (0.97
of the cap), 6.3 s at `T = 4000`. `adl_midas` caps its `nobs × (1 + P + K)`
design at 2^26 cells and its work, about `nobs k² (min(maxlags, nobs − 1) + 2)`
units with `k = 1 + P + K`, at 2^33; a unit measured 0.65–2.16 ns, 13.2 s at
the cap itself (`n = 100000`, `K = 6`, `maxlags = 1340`). Past a cap either
call is refused before anything is built, naming what sizes it (`ar_lags`, or
`maxlags` when the bandwidth alone is too wide). The timings are recorded by
`lab/audit/wave11/dfm_smooth_cost.py` and `lab/audit/wave11/adl_midas_cost.py`
(output in `lab/audit/wave11/out/`).

## Validated against

The DFM Kalman/state-space step is reference-exact (~1e-8) against a documented
single-factor dynamic-factor model in the `statsmodels` `DynamicFactor` layout;
the two-step nowcast and the Banbura-Modugno (2014) news decomposition are
documented-formula goldens (an independent NumPy Kalman smoother), with the
contributions verified to sum to the total revision. U-MIDAS is validated as
OLS against a documented reference, and the exp-Almon / beta weight formulas
and the weighted-MIDAS NLS against their published definitions.

`dfm_smooth` (0.11) is pinned against statsmodels 0.15.0
`DynamicFactor(k_factors=2, factor_order=2, error_order=0).smooth(params)` on
a ragged panel with an interior gap, against statsmodels' generic
`KalmanSmoother` for a non-identity `factor_cov`, and on a standardized
`r = 1, p = 3` panel with its edge nowcast: log-likelihood 1e-8 relative,
smoothed states and covariances 1e-8 absolute (achieved 5.2e-13 and 6.0e-11 —
statsmodels' filter and the crate's univariate one agree to roundoff, not
bitwise). The round trip with `dfm_nowcast` is bitwise on both estimation
routes. `adl_midas` (0.11) is pinned against statsmodels `OLS` on a design
built independently with `pandas.shift` — params, residuals, R², and bse /
t-values under nonrobust and HAC at the default and at `maxlags = 8` — at
1e-10 (achieved 3.2e-14), which also pins the parameter order. Fixtures:
[`fixtures/tsecon-nowcast.json`](../../../fixtures/tsecon-nowcast.json),
[`fixtures/nowcast_news.json`](../../../fixtures/nowcast_news.json),
[`fixtures/midas.json`](../../../fixtures/midas.json),
[`fixtures/dfm_smooth.json`](../../../fixtures/dfm_smooth.json),
[`fixtures/adl_midas.json`](../../../fixtures/adl_midas.json).

## References

- Ghysels, E., Santa-Clara, P. & Valkanov, R. (2004). "The MIDAS touch: Mixed
  data sampling regression models." Working paper.
- Ghysels, E., Sinko, A. & Valkanov, R. (2007). "MIDAS Regressions: Further
  Results and New Directions." *Econometric Reviews* 26.
- Andreou, E., Ghysels, E. & Kourtellos, A. (2013). "Should Macroeconomic
  Forecasters Use Daily Financial Data and How?" *J. Business & Economic
  Statistics* 31.
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

# 6. Re-smooth a vintage with the fitted model: no re-estimation.
model = {k: now[k] for k in ("loadings", "factor_ar", "factor_cov",
                             "idiosyncratic", "center", "scale")}
same = tsecon.dfm_smooth(new_v, **model)
print("re-smooth == dfm_nowcast:", same["loglik"] == now["loglik"],
      np.array_equal(same["nowcast"], now["nowcast"]))
newer = new_v.copy()
newer[-1, 3] = data[-1, 3]                            # series 3's release arrives
upd = tsecon.dfm_smooth(newer, **model)
print("nowcast of series 4:", round(now["nowcast"][4], 3), "->",
      round(upd["nowcast"][4], 3))
print("factor sd at the edge:",
      round(float(np.sqrt(same["smoothed_state_cov"][-1, 0, 0])), 4), "->",
      round(float(np.sqrt(upd["smoothed_state_cov"][-1, 0, 0])), 4))

# 7. ADL-MIDAS: U-MIDAS plus the target's own lag.
yq = np.zeros(nobs)
for t in range(nobs):
    yq[t] = 0.5 + (0.6 * yq[t - 1] if t else 0.0) + 2.0 * (HF[t] @ w) + 0.3 * rng.standard_normal()
adl = tsecon.adl_midas(yq, HF, ar_lags=1)
print("ADL-MIDAS:", adl["param_names"][:3], np.round(adl["params"][:3], 3),
      " nobs:", adl["nobs"], " maxlags:", adl["maxlags"])
```

Expected output:

```
exp-Almon weights sum to: 1.0
U-MIDAS R^2: 0.936  params: 7
weighted-MIDAS R^2: 0.934  converged: True
nowcast (5 series): [7.57 3.47 2.73 4.15 3.46]
revision: 0.088  from 5 datapoints
largest contribution from series 4 = 0.0759
re-smooth == dfm_nowcast: True True
nowcast of series 4: 3.461 -> 3.406
factor sd at the edge: 0.2798 -> 0.2525
ADL-MIDAS: ['const', 'y.L1', 'hf1'] [0.499 0.621 1.053]  nobs: 119  maxlags: 4
```

Step 6 is the vintage workflow: the re-smooth reproduces `dfm_nowcast` bit
for bit, and when series 3's last value is released the edge factor's
posterior standard deviation falls from 0.2798 to 0.2525 and every series'
nowcast moves with it — at the fitted parameters, with nothing re-estimated.
