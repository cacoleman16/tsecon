# Get HAC (Newey-West) standard errors on a regression

Serially correlated errors do not bias OLS coefficients — they wreck the
standard errors. In `tsecon` the whole robust ladder is one keyword on
`tsecon.ols`: `se_type="nonrobust"`, `"hc0"`, `"hc1"`, or `"hac"`. The HAC rung
is the Bartlett-kernel Newey-West estimator and matches statsmodels'
`cov_type="HAC"` to 1e-10.

## The recipe

```python
import numpy as np, tsecon

rng = np.random.default_rng(1)                     # persistent regressor, persistent errors
n = 300
x = np.zeros(n)
e = np.zeros(n)
for t in range(1, n):
    x[t] = 0.8 * x[t - 1] + rng.standard_normal()
    e[t] = 0.7 * e[t - 1] + rng.standard_normal()
y = 1.0 + 0.5 * x + e                              # true slope = 0.5
X = np.column_stack([np.ones(n), x])               # add your own constant column

for se in ("nonrobust", "hc1", "hac"):
    r = tsecon.ols(y, X, se_type=se)
    print(f"{se:>9}   beta {r['params'][1]:.4f}   se {r['bse'][1]:.4f}   t {r['tvalues'][1]:5.2f}")
```

```
nonrobust   beta 0.5298   se 0.0423   t 12.53
      hc1   beta 0.5298   se 0.0441   t 12.02
      hac   beta 0.5298   se 0.0681   t  7.78
```

## Reading it

Three things to internalise, all visible in three lines of output.

1. **The coefficient never moves.** Robust standard errors reweight
   *uncertainty*, not estimates. If a package changes your point estimate when
   you change `se_type`, something else is going on.
2. **`hc1` does almost nothing here.** White-type corrections fix
   heteroskedasticity and assume *independent* errors. On time series data they
   are just as overconfident as `nonrobust`. Reaching for "robust standard
   errors" out of cross-sectional habit is the single most common inference
   mistake in applied time series.
3. **HAC is 60% wider**, and the t-statistic falls from 12.5 to 7.8. In a
   borderline case that is the difference between a published finding and a
   null.

`tsecon.ols` returns `params`, `bse`, `tvalues`, and the echoed `se_type` — no
p-values, deliberately, because the right reference distribution depends on the
inference you are doing. `2 * scipy.stats.norm.sf(abs(t))` is the usual normal
approximation if you want one.

## Choosing the bandwidth

`maxlags=None` (the default) uses the Newey-West rule of thumb. Set it
explicitly when you know the persistence in your data outruns the rule:

```python
print("Newey-West rule of thumb :", round(tsecon.ols(y, X, se_type="hac")["bse"][1], 4))
print("maxlags=12               :", round(tsecon.ols(y, X, se_type="hac", maxlags=12)["bse"][1], 4))
```

```
Newey-West rule of thumb : 0.0681
maxlags=12               : 0.0756
```

Widening the window raises the standard error by another 11%. That sensitivity
is real and it is why **a HAC standard error without its kernel and bandwidth is
not a reportable number** — say "Newey-West, Bartlett kernel, 12 lags" in the
table note. When you replicate someone else's HAC t-statistic and it does not
match, the bandwidth is the first suspect, not the estimator.

## Gotchas

- **`x` is used exactly as given.** `tsecon.ols` adds no constant; build the
  design matrix yourself with `np.column_stack([np.ones(n), ...])`.
- `use_correction` toggles the small-sample $n/(n-k)$ factor. Packages differ on
  this default; it is another reason two HAC numbers can disagree.
- HAC undercovers in small samples even when the bandwidth is right — see the
  coverage experiment and the fixed-b / EWC discussion in the guide chapter
  below.
- Estimating a *long-run variance* on its own rather than a regression? That is
  `tsecon.long_run_variance`.

## See also

- Guide: [3. Honest Inference](../guide/03-inference-toolkit.md) — the robust
  standard-error ladder, kernels, bandwidths, and a Monte Carlo coverage check
- Model card: [Diagnostics and the stationarity workflow](../reference/model-cards/diagnostics.md)
- Recipes: [Driscoll-Kraay and clustered SEs for a panel local projection](panel-lp-standard-errors.md) ·
  [Estimate a fiscal multiplier from an instrumented shock](fiscal-multiplier.md)
