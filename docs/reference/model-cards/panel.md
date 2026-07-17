# Model card — Panel time series

**Family:** `panel_fe`, `panel_lp`, `mean_group_var`, `panel_mean_group`,
`panel_pmg`

Many entities, each observed over time. The methods here span the two ends of
the panel spectrum: **pooled** estimators that assume a common slope and
difference out fixed effects (`panel_fe`, `panel_lp`), and **heterogeneous**
estimators that let every unit have its own dynamics and then average or pool
carefully across them (`mean_group_var`, `panel_mean_group`, `panel_pmg`). The
recurring theme is honest inference: cross-sectional and serial correlation
both bias naïve standard errors, so the defaults reach for Driscoll-Kraay and
cluster covariances.

| Function | Slope assumption | Delivers |
|----------|------------------|----------|
| `panel_fe` | common | Fixed-effects OLS with robust SEs |
| `panel_lp` | common | Panel local-projection IRF of a common shock |
| `mean_group_var` | heterogeneous | Mean-group panel VAR + orthogonalized IRFs |
| `panel_mean_group` | heterogeneous | Mean-group / CCE-MG average slope |
| `panel_pmg` | pooled long run, free short run | Pooled Mean Group ARDL(1,1) |

## What it estimates

- **`panel_fe(outcome, regressors)`** — the within (fixed-effects) estimator:
  entity means are swept out and a common slope vector is estimated by OLS,
  with clustered or Driscoll-Kraay standard errors. `outcome` is N×T;
  `regressors` is k×N×T.
- **`panel_lp(outcome, shock)`** — a panel local projection: at each horizon h,
  regress the h-step-ahead outcome on a **common** shock with entity fixed
  effects, tracing a dynamic causal response averaged across units.
- **`mean_group_var(entities)`** — fits a separate VAR to each entity's Tᵢ×k
  matrix and averages the coefficients and orthogonalized IRFs (Pesaran-Smith
  1995). Robust to slope heterogeneity that a pooled panel VAR would bias.
- **`panel_mean_group(ys, xs)`** — the mean-group estimator: per-unit OLS
  slopes averaged across units, with the cross-unit standard deviation giving
  the standard error. `method="cce"` adds Pesaran (2006) common-correlated-
  effects terms (cross-sectional averages) to purge a common factor.
- **`panel_pmg(ys, xs)`** — the Pooled Mean Group ARDL(1,1) estimator
  (Pesaran-Shin-Smith 1999): the **long-run** coefficient θ is pooled (common
  across units) by maximum likelihood, while the error-correction speed and
  short-run dynamics stay unit-specific.

## Assumptions

- **`panel_fe` / `panel_lp` assume a common slope.** If the true response
  differs across units, the pooled estimate is a variance-weighted average that
  need not equal the cross-sectional mean effect — reach for the mean-group
  estimators instead.
- **Cross-sectional dependence.** With a common shock or common factor, errors
  are correlated across entities at each date; cluster-by-entity SEs do not
  address this. `se_type="driscoll_kraay"` is the default for `panel_lp`
  precisely because it is robust to both serial and cross-sectional
  correlation.
- **`panel_pmg` requires a genuine ARDL / error-correction structure**: the
  long-run regressors must be non-degenerate and not collinear across the panel
  once short-run dynamics are partialled out, or θ is not identified (the
  estimator raises rather than returning a meaningless number). Feed it *level*
  series with real dynamics, not, say, a shock and its own lag.
- **Mean-group estimators need enough time per unit** to estimate each unit's
  regression; they trade the efficiency of pooling for robustness to
  heterogeneity, and are noisy when Tᵢ is small.
- **`panel_mean_group(method="mg")` is a static regression** of y on
  contemporaneous x — its average slope is *not* the ARDL long-run coefficient.
  Use `panel_pmg` when the object of interest is a common long-run relationship.

## When to use

- **`panel_fe`** — the workhorse when you believe in a common slope and want
  clustered or Driscoll-Kraay inference (e.g. the effect of a policy variable
  across countries).
- **`panel_lp`** — dynamic causal responses to a *common* shock (a global oil
  or monetary shock hitting many countries), fixed effects for level
  differences, Driscoll-Kraay bands.
- **`mean_group_var`** — impulse responses in a heterogeneous panel where a
  pooled VAR would be misspecified.
- **`panel_mean_group`** — the average marginal effect across heterogeneous
  units; `method="cce"` when an unobserved common factor contaminates OLS.
- **`panel_pmg`** — long-run equilibrium relationships (growth-savings,
  consumption-income) where theory says the long run is common but adjustment
  speeds differ by country.

## Key arguments and defaults

| Call | Argument | Default | Notes |
|------|----------|---------|-------|
| `panel_fe` | `se_type` | `"cluster"` | `"nonrobust"`, `"cluster"` (by entity), `"driscoll_kraay"` |
| | `bandwidth` | `4.0` | Driscoll-Kraay kernel bandwidth |
| `panel_lp` | `horizon` | `8` | IRF horizons |
| | `n_lag_controls` | `2` | lags of outcome/shock included as controls |
| | `se_type` | `"driscoll_kraay"` | robust to cross-sectional dependence |
| | `cumulative` | `False` | `True` for cumulative IRFs |
| | `jackknife` | `False` | leave-one-entity-out bias reduction |
| `mean_group_var` | `lags` | `1` | per-entity VAR order |
| | `trend` | `"c"` | deterministic terms |
| | `horizon` / `response` / `impulse` | `10` / `0` / `0` | IRF horizon and the response/shock variable indices |
| `panel_mean_group` | `method` | `"mg"` | or `"cce"` (common-correlated-effects) |
| `panel_pmg` | — | — | `ys`/`xs` per-unit level series and Tᵢ×k regressor matrices |

## How to read the output

- **`panel_fe`** → `{"params", "bse", "tvalues", "se_type"}`, one entry per
  regressor. The stamped `se_type` tells you which covariance produced `bse`.
- **`panel_lp`** → `{"irf", "se", "nobs"}`, each length `horizon+1`; plot `irf`
  ±1.96·`se`. `irf[0]` is the impact response.
- **`mean_group_var`** → per-entity-averaged `intercept`, `coefs`
  (lags × neqs × neqs) and their SEs, plus `orth_irfs`
  (horizon+1 × response × shock) with SEs and a convenience `irf_path`
  (the `response`/`impulse` cell) and `irf_path_se`. Also `n_entities`,
  `neqs`, `lags`.
- **`panel_mean_group`** → `{"coef", "se", "tstat", "coef_per_unit",
  "n_units", "k"}`. `coef_per_unit` (n_units × k) lets you inspect the spread
  of individual slopes behind the average.
- **`panel_pmg`** → `{"theta", "theta_se", "phi_bar", "phi", "sigma2",
  "loglik", "iterations", "n_units", "k"}`. `theta` is the pooled long-run
  coefficient; `phi_bar` is the average error-correction speed (negative and
  bounded by −1 for stable adjustment); `phi` is the per-unit speed vector.

## Failure modes

- **Pooling heterogeneous slopes.** `panel_fe` on data with genuinely
  different unit responses returns a hard-to-interpret weighted average. If a
  Hausman-style comparison of pooled vs mean-group estimates diverges, trust
  the mean-group one.
- **Cluster SEs under cross-sectional dependence.** With a common shock,
  `se_type="cluster"` understates uncertainty. Use `driscoll_kraay`.
- **`panel_pmg` collinearity error.** If the partialled long-run regressors are
  collinear across the panel, θ is unidentified and the call raises. This is a
  correct refusal, not a bug — supply level regressors with real, non-redundant
  long-run variation.
- **Small Tᵢ with mean-group.** Per-unit regressions become unstable and the
  cross-unit average inherits the noise; prefer pooling (with heterogeneity
  tested) when time series are short.
- **Reading `panel_mean_group(method="mg")` as a long run.** It is a static
  average slope; the ARDL long run comes from `panel_pmg`.

## Validated against

`panel_fe` matches `linearmodels` `PanelOLS` for the within estimator under
nonrobust, cluster-by-entity, and Driscoll-Kraay (Bartlett kernel) covariances.
`panel_lp` is a documented-formula golden built on the same within-plus-DK
machinery with a known simulated IRF. `mean_group_var`, `panel_mean_group`
(MG and CCE-MG), and `panel_pmg` are documented-formula goldens reproducing the
Pesaran-Smith (1995), Pesaran (2006), and Pesaran-Shin-Smith (1999)
estimating equations, and are additionally property-validated: on data with a
known common long run, PMG recovers it and pools far more tightly than a free
mean-group of per-unit long runs. Fixtures:
[`fixtures/panel.json`](../../../fixtures/panel.json),
[`fixtures/tsecon-panelts.json`](../../../fixtures/tsecon-panelts.json),
[`fixtures/pmg.json`](../../../fixtures/pmg.json).

## References

- Pesaran, M. H. & Smith, R. (1995). "Estimating long-run relationships from
  dynamic heterogeneous panels." *J. Econometrics* 68.
- Pesaran, M. H., Shin, Y. & Smith, R. (1999). "Pooled Mean Group Estimation of
  Dynamic Heterogeneous Panels." *JASA* 94.
- Pesaran, M. H. (2006). "Estimation and Inference in Large Heterogeneous
  Panels with a Multifactor Error Structure." *Econometrica* 74.
- Driscoll, J. & Kraay, A. (1998). "Consistent Covariance Matrix Estimation
  with Spatially Dependent Panel Data." *Rev. Econ. Stat.* 80.
- Jordà, Ò. (2005). "Estimation and Inference of Impulse Responses by Local
  Projections." *AER* 95.

See the guide: [Panel Time Series](../../guide/14-panel-time-series.md).

## Runnable example

```python
import numpy as np
import tsecon

rng = np.random.default_rng(88)
N, T = 20, 100

# ---- a balanced panel with entity fixed effects and a common observed shock ----
shock = rng.standard_normal(T)
alpha = rng.normal(0, 2.0, N)                 # entity fixed effects
psi = 0.8 * 0.6 ** np.arange(8)               # true dynamic response to the shock
y = np.empty((N, T))
for i in range(N):
    u = np.empty(T); u[0] = rng.standard_normal()
    for t in range(1, T):
        u[t] = 0.3 * u[t - 1] + rng.standard_normal()
    y[i] = alpha[i] + np.convolve(shock, psi)[:T] + u + 0.3 * rng.standard_normal(T)

# 1. Fixed-effects panel OLS. outcome is N x T; regressors is k x N x T.
s0 = np.tile(shock, (N, 1))
s1 = np.tile(np.r_[0.0, shock[:-1]], (N, 1))
regressors = np.stack([s0, s1])               # 2 x N x T
fe = tsecon.panel_fe(y, regressors, se_type="driscoll_kraay")
print("FE params:", np.round(fe["params"], 3), " (Driscoll-Kraay SEs)")

# 2. Panel local projection of the common shock (dynamic causal response).
plp = tsecon.panel_lp(y, shock, horizon=8, se_type="driscoll_kraay")
print("panel-LP IRF h=0..2:", np.round(plp["irf"][:3], 3))

# 3. Mean-group panel VAR (Pesaran-Smith): per-entity VARs, averaged.
entities = [np.column_stack([y[i], np.r_[0.0, shock[:-1]]]) for i in range(N)]
mg = tsecon.mean_group_var(entities, lags=2, horizon=8)
print("MG-VAR orthogonalized IRF path h=0..2:", np.round(mg["irf_path"][:3], 3))

# ---- a heterogeneous ARDL(1,1) panel with a COMMON long run (for MG / PMG) ----
theta0 = np.array([1.5, -0.8])
def sim_unit():
    lam = rng.uniform(0.2, 0.7); mu = rng.normal(0.5, 1.0)
    d0 = rng.normal([0.6, -0.3], [0.25, 0.25]); d1 = theta0 * (1 - lam) - d0
    burn, tt = 50, 90 + 50; K = 2
    x = np.empty((tt, K)); rho = rng.uniform(0.3, 0.6, K); xm = rng.normal(0, 1, K); x[0] = xm
    for t in range(1, tt):
        x[t] = xm * (1 - rho) + rho * x[t - 1] + rng.normal(0, 1, K)
    yy = np.empty(tt); yy[0] = mu / (1 - lam)
    for t in range(1, tt):
        yy[t] = mu + lam * yy[t - 1] + d0 @ x[t] + d1 @ x[t - 1] + rng.normal(0, 0.5)
    return yy[burn:], x[burn:]
ys = []; xs = []
for _ in range(25):
    yy, xx = sim_unit(); ys.append(yy); xs.append(xx)

# 4. Mean-group / CCE-MG estimator: the average of per-unit static slopes.
#    (A static contemporaneous regression, so this is NOT the ARDL long run.)
mgest = tsecon.panel_mean_group(ys, xs, method="mg")
print("MG average slope:", np.round(mgest["coef"], 3), " t:", np.round(mgest["tstat"], 2))

# 5. Pooled Mean Group: pool the long-run coefficient, keep short-run dynamics
#    free. This IS the estimator that targets the common long run of the DGP.
pmg = tsecon.panel_pmg(ys, xs)
print("PMG long-run theta:", np.round(pmg["theta"], 3),
      " (true", theta0, "),  adjustment speed phi_bar:", round(pmg["phi_bar"], 3))
```

Expected output:

```
FE params: [0.827 0.454]  (Driscoll-Kraay SEs)
panel-LP IRF h=0..2: [0.819 0.383 0.303]
MG-VAR orthogonalized IRF path h=0..2: [1.289 0.424 0.205]
MG average slope: [ 0.878 -0.418]  t: [ 25.53 -10.53]
PMG long-run theta: [ 1.5 -0.8]  (true [ 1.5 -0.8] ),  adjustment speed phi_bar: -0.583
```
