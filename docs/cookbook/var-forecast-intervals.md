# Forecast a VAR with prediction intervals

`tsecon.var_forecast` produces iterated multi-step forecasts for every variable
in the system plus symmetric asymptotic intervals, in one call, from the same
`(data, lags)` arguments every other VAR function takes. No fitted object to
carry around.

## The recipe

```python
import numpy as np, tsecon

rng = np.random.default_rng(0)                      # same toy system as the IRF recipe
A = np.array([[0.55, 0.10], [0.35, 0.60]])
L = np.array([[1.00, 0.00], [0.50, 0.80]])
y = np.zeros((240, 2))
for t in range(1, 240):
    y[t] = A @ y[t - 1] + L @ rng.standard_normal(2)

fit = tsecon.var_fit(y, lags=2)
print("stable:", fit["is_stable"], " min_root:", round(fit["min_root"], 3),
      " bic:", round(fit["bic"], 3))

f = tsecon.var_forecast(y, lags=2, steps=8, alpha=0.10)     # 90% intervals
point, lower, upper = (np.array(f[k]) for k in ("point", "lower", "upper"))
for h in range(8):
    print(f"h={h+1}   y1 {point[h,0]:+.3f} [{lower[h,0]:+.3f}, {upper[h,0]:+.3f}]"
          f"   y2 {point[h,1]:+.3f} [{lower[h,1]:+.3f}, {upper[h,1]:+.3f}]")
```

```
stable: True  min_root: 1.554  bic: -0.233
h=1   y1 -0.888 [-2.516, +0.741]   y2 -1.163 [-2.750, +0.424]
h=2   y1 -0.702 [-2.627, +1.222]   y2 -1.033 [-3.063, +0.996]
h=3   y1 -0.566 [-2.631, +1.498]   y2 -0.865 [-3.211, +1.481]
h=4   y1 -0.469 [-2.599, +1.661]   y2 -0.705 [-3.252, +1.842]
h=5   y1 -0.402 [-2.562, +1.758]   y2 -0.571 [-3.234, +2.092]
h=6   y1 -0.357 [-2.530, +1.815]   y2 -0.467 [-3.193, +2.258]
h=7   y1 -0.328 [-2.506, +1.850]   y2 -0.391 [-3.148, +2.365]
h=8   y1 -0.310 [-2.489, +1.870]   y2 -0.338 [-3.110, +2.434]
```

## Reading it

`point`, `lower`, and `upper` are each `steps × k`, so column `j` is variable
`j`'s path. The forecasts decay toward the unconditional mean — that is what a
stable VAR does, and it is why `fit["is_stable"]` is worth printing *before*
the forecast rather than after. **Read `is_stable`, not `max_root`**: these are
reciprocal characteristic roots, so stability requires the *smallest* to exceed
1, and `max_root` stays above 1 even for an explosive system.

The intervals widen with the horizon and then stop widening, converging to the
unconditional forecast-error variance. Interval width, not the point path, is
what a forecast consumer should look at first.

## Choosing the coverage

`alpha` is the non-coverage, so `alpha=0.05` is 95% and `alpha=0.32` is the 68%
("one sigma") band:

```python
for a in (0.32, 0.05):
    w = np.array(tsecon.var_forecast(y, lags=2, steps=8, alpha=a)["upper"])[:, 0] - point[:, 0]
    print(f"alpha={a}  half-width at h=1: {w[0]:.3f}   at h=8: {w[7]:.3f}")
```

```
alpha=0.32  half-width at h=1: 0.984   at h=8: 1.318
alpha=0.05  half-width at h=1: 1.940   at h=8: 2.597
```

## Gotchas

- **These intervals price innovation uncertainty only.** They are
  `point ± z·se` where `se` comes from the residual covariance propagated
  forward; they do not add parameter-estimation uncertainty, and they are
  symmetric and Gaussian. For a fan chart with genuine posterior spread, draw
  from a [Bayesian VAR](../reference/model-cards/bayesian.md).
- **Pick `lags` deliberately.** Compare `aic` / `bic` / `hqic` from `var_fit`
  over a *common* effective sample — comparing ICs fitted on different samples
  flips rankings.
- **Do not forecast I(1) levels casually.** Difference first, or use
  `tsecon.johansen` / `tsecon.vecm` if the series trend together.
- Want to know whether this VAR actually forecasts better than a naive rule?
  That is `tsecon.backtest` plus `tsecon.dm_test`.

## See also

- Model card: [VAR / SVAR](../reference/model-cards/var-svar.md) ·
  [Forecasting](../reference/model-cards/forecasting.md)
- Guide: [5. Forecasting](../guide/05-forecasting.md) ·
  [7. Systems (VAR, Factors)](../guide/07-multivariate.md)
- Recipes: [Put confidence bands on a VAR impulse response](var-irf-bands.md)
