# Get HAC (Newey-West) standard errors on a regression

Serially correlated errors do not bias OLS coefficients — they wreck the
standard errors. In `tsecon` the whole robust ladder is one keyword on
`tsecon.ols`: `se_type="nonrobust"`, `"hc0"`, `"hc1"`, `"hc2"`, `"hc3"`, or
`"hac"`. This recipe is about the last one. The HAC rung is the Bartlett-kernel
Newey-West estimator and matches statsmodels' `cov_type="HAC"` to 1e-10 when
`use_correction` is matched — the two libraries default that flag opposite
ways (see [Gotchas](#gotchas)); the
`hc*` rungs fix heteroskedasticity only, and the
[section below](#if-the-problem-is-heteroskedasticity-not-serial-correlation)
shows why they are no help here and where they are indispensable instead.

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
2. **`hc1` does almost nothing here** — and neither does any other HC rung.
   White-type corrections fix heteroskedasticity and assume *independent*
   errors. On time series data they are just as overconfident as `nonrobust`.
   Reaching for "robust standard errors" out of cross-sectional habit is the
   single most common inference mistake in applied time series.
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

## If the problem is heteroskedasticity, not serial correlation

Everything above assumes you diagnosed the disease correctly. If the errors are
*independent* but unequally scaled, HAC is the wrong tool and the leverage-
corrected `hc2`/`hc3` rungs are the right one. Run both claims on the same page:

```python
print("hc3 on the recipe design :", round(tsecon.ols(y, X, se_type="hc3")["bse"][1], 4))

g = np.random.default_rng(7)                        # short sample, NO serial correlation
m = 25
xs = g.chisquare(1.0, m)                            # right-skewed -> a few high-leverage points
ys = 1.0 + 2.0 * xs + xs * g.standard_normal(m)     # sd(e|x) = x
Xs = np.column_stack([np.ones(m), xs])

for se in ("nonrobust", "hc1", "hc2", "hc3", "hac"):
    print(f"{se:>9}   se {tsecon.ols(ys, Xs, se_type=se)['bse'][1]:.4f}")
```

```
hc3 on the recipe design : 0.0446
nonrobust   se 0.2327
      hc1   se 0.4805
      hc2   se 0.5450
      hc3   se 0.6496
      hac   se 0.4662
```

The first line closes the loop on point 2 above: climbing to the top of the HC
ladder on the serially correlated design moves the standard error from `hc1`'s
0.0441 to 0.0446, about 1%, while `hac` needs 0.0681. HC is not a partial fix
for autocorrelation.

The second block is the mirror image. `hc1` scales every squared residual by the
same $n/(n-k)$; `hc2` and `hc3` divide each one by $1-h_t$ and $(1-h_t)^2$,
where $h_t$ is that observation's leverage — so only they can tell that a
handful of points are carrying the design. Here `hc3` is 35% wider than `hc1`,
and the [coverage audit](../examples/interval-coverage.md#regression-standard-errors)
measures what that width is worth: on this design at $n=25$, `hc1` covers 0.68
against a nominal 0.95 while `hc3` covers 0.86.

Note the last row. `hac` is *narrower* than `hc1` here, because it carries no
leverage correction at all — at zero lags it reduces to `hc0`, never to `hc2` or
`hc3`. Neither family contains the other. Pick by which assumption your data
violates: serial correlation → `hac`; heteroskedasticity with a short sample or
influential points → `hc3`; both → `hac`, and treat it as a floor.

## Gotchas

- **`x` is used exactly as given.** `tsecon.ols` adds no constant; build the
  design matrix yourself with `np.column_stack([np.ones(n), ...])`.
- `use_correction` toggles the small-sample $n/(n-k)$ factor, and the packages
  default it **opposite ways**: `tsecon` defaults it **on** (the finite-sample
  choice), statsmodels `cov_type="HAC"` defaults it **off** (its
  `cov_hac_simple` helper, confusingly, defaults on). So
  `tsecon.ols(..., se_type="hac")` and a bare statsmodels
  `.fit(cov_type="HAC", ...)` disagree by exactly $\sqrt{n/(n-k)}$ — 4.3% at
  $n=25, k=2$ — while matching either flag setting agrees to 1e-10. Pass
  `use_correction=False` to reproduce a default statsmodels call. It is the
  first thing to check, after the bandwidth, when two HAC numbers disagree.
- HAC undercovers in small samples even when the bandwidth is right — see the
  coverage experiment and the fixed-b / EWC discussion in the guide chapter
  below.
- `use_correction` does *not* apply to `hc2`/`hc3`: their correction is
  per-observation, through leverage, not a single scalar. And if some
  observation's leverage is numerically 1, HC2/HC3 are undefined — `tsecon`
  raises rather than handing back a near-infinite standard error.
- Estimating a *long-run variance* on its own rather than a regression? That is
  `tsecon.long_run_variance`.

## See also

- Guide: [3. Honest Inference](../guide/03-inference-toolkit.md#the-robust-standard-error-ladder) —
  the full robust standard-error ladder (which rung at which sample size),
  kernels, bandwidths, and a Monte Carlo coverage check
- Model card: [Diagnostics and the stationarity workflow](../reference/model-cards/diagnostics.md)
- Recipes: [Driscoll-Kraay and clustered SEs for a panel local projection](panel-lp-standard-errors.md) ·
  [Estimate a fiscal multiplier from an instrumented shock](fiscal-multiplier.md)
