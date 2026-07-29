# Driscoll-Kraay and clustered standard errors for a panel local projection

Clustering by entity is the reflex. On a panel hit by a **common** shock it is
the wrong reflex, and it does not fail loudly — it fails by making your standard
errors *smaller*. `tsecon.panel_lp` exposes the choice as `se_type`, and this
recipe shows what each one is actually correcting.

## The recipe

```python
import numpy as np, tsecon

rng = np.random.default_rng(5)
N, T = 30, 120                                       # 30 countries, 120 quarters
shock = rng.standard_normal(T)                       # ONE common shock, same for everyone
common = rng.standard_normal(T)                      # an unobserved common factor
alpha = rng.normal(0, 1, N)                          # entity fixed effects
y = np.zeros((N, T))                                 # outcome is N x T
for i in range(N):
    for t in range(1, T):
        y[i, t] = (alpha[i] + 0.5 * y[i, t - 1] + 0.8 * shock[t]
                   + 0.9 * common[t] + rng.standard_normal())

for se in ("nonrobust", "cluster", "driscoll_kraay"):
    r = tsecon.panel_lp(y, shock, horizon=6, n_lag_controls=2, se_type=se)
    band = " ".join(f"{s:.3f}" for s in r["se"][:4])
    tstat = r["irf"][2] / r["se"][2]
    print(f"{se:>15}   se(h=0..3) {band}   t at h=2 {tstat:6.2f}")
```

```
      nonrobust   se(h=0..3) 0.026 0.032 0.033 0.034   t at h=2   9.31
        cluster   se(h=0..3) 0.021 0.022 0.022 0.023   t at h=2  14.18
 driscoll_kraay   se(h=0..3) 0.086 0.142 0.142 0.128   t at h=2   2.18
```

## Reading it

The point estimates are identical across all three rows — only the uncertainty
changes. What changes is dramatic:

- **`cluster` made things worse.** Clustering by entity allows arbitrary
  correlation *within* a country over time, and assumes independence *across*
  countries. This panel violates exactly that assumption: a common factor hits
  every country in the same quarter. The t-statistic goes *up*, from 9.3 to
  14.2. Nothing warns you.
- **`driscoll_kraay` is four to six times wider** and drops the t-statistic to
  2.2. It
  clusters by *time* and applies a HAC kernel across periods, so it absorbs both
  the cross-sectional dependence and the serial correlation the local projection
  guarantees.

When the regressor is a single common shock — one number per period, shared by
every unit — the effective sample size is closer to `T` than to `N × T`.
Driscoll-Kraay is the estimator that knows this. Reach for `cluster` when your
shock varies across entities and cross-sectional dependence is genuinely
implausible.

## Building the band

```python
r = tsecon.panel_lp(y, shock, horizon=6, n_lag_controls=2, se_type="driscoll_kraay")
for h in range(7):
    lo = r["irf"][h] - 1.96 * r["se"][h]
    hi = r["irf"][h] + 1.96 * r["se"][h]
    print(f"h={h}   {r['irf'][h]:+.3f}   [{lo:+.3f}, {hi:+.3f}]")
```

```
h=0   +0.731   [+0.563, +0.899]
h=1   +0.515   [+0.237, +0.794]
h=2   +0.310   [+0.032, +0.588]
h=3   +0.239   [-0.012, +0.490]
h=4   +0.140   [-0.141, +0.421]
h=5   +0.226   [-0.084, +0.535]
h=6   +0.125   [-0.129, +0.380]
```

Under Driscoll-Kraay the response is significant through `h=2` and
indistinguishable from zero afterwards. Under clustered standard errors every
horizon would have looked significant.

## Gotchas

- **Shapes.** `outcome` is `N × T` (entities by time); `shock` is length `T`.
  `tsecon.panel_fe` takes the same `outcome` plus `regressors` as `k × N × T`
  and offers the same `se_type` menu.
- **`bandwidth`** sets the Driscoll-Kraay kernel width. Like any HAC bandwidth
  it moves the answer; report it.
- **`jackknife=True`** applies the Dhaene-Jochmans half-panel correction for
  Nickell bias — worth using when `T` is short and lagged dependent variables
  are in the controls.
- **`cumulative=True`** switches to the Ramey-Zubairy cumulated-outcome
  convention.
- Slopes genuinely heterogeneous across units? A pooled estimator is the wrong
  model, not the wrong standard error — see `tsecon.panel_mean_group` and
  `tsecon.panel_pmg`.

## See also

- Model card: [Panel time series](../reference/model-cards/panel.md)
- Guide: [14. Panel Time Series](../guide/14-panel-time-series.md) ·
  [9. Local Projections](../guide/09-local-projections.md)
- Recipes: [Get HAC standard errors on a regression](hac-standard-errors.md) ·
  [Estimate a fiscal multiplier from an instrumented shock](fiscal-multiplier.md)
