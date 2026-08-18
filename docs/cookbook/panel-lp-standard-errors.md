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

## Short panels: correcting Nickell bias without lying about the uncertainty

With `T = 120` above, dynamic-panel (Nickell) bias is negligible. Shorten the
panel and it is not: fixed effects + dynamics + short `T` biases the IRF by
`O(h/T)` — growing with the horizon — and no `se_type` fixes a biased point
estimate. `panel_lp` has two half-panel corrections, and they treat the
*standard errors* very differently:

```python
rng = np.random.default_rng(3)
N, T = 30, 40                                        # a SHORT panel this time
shock = rng.standard_normal(T)
alpha = rng.normal(0, 1, N)
y = np.zeros((N, T))
for i in range(N):
    for t in range(1, T):
        y[i, t] = (alpha[i] + 0.8 * y[i, t - 1] + 0.8 * shock[t]
                   + rng.standard_normal())

common = dict(horizon=4, n_lag_controls=1, se_type="driscoll_kraay", bandwidth=2.0)
fe  = tsecon.panel_lp(y, shock, **common)
dj  = tsecon.panel_lp(y, shock, jackknife=True, **common)
spj = tsecon.panel_lp(y, shock, bias_correction="spj", **common)
print("true                       " + " ".join(f"{0.8*0.8**h:+.3f}" for h in range(5)))
for r, label in ((fe, "fe"), (dj, "dj"), (spj, "spj")):
    irf = " ".join(f"{v:+.3f}" for v in r["irf"])
    se  = " ".join(f"{v:.3f}" for v in r["se"])
    print(f"{label:>4} ({r['bias_correction']:>15})  irf {irf}   se {se}")
```

```
true                       +0.800 +0.640 +0.512 +0.410 +0.328
  fe (           none)  irf +0.751 +0.512 +0.376 +0.300 +0.064   se 0.030 0.079 0.083 0.154 0.153
  dj (dhaene_jochmans)  irf +0.752 +0.539 +0.436 +0.399 +0.346   se 0.030 0.079 0.083 0.154 0.153
 spj (            spj)  irf +0.752 +0.546 +0.441 +0.369 +0.266   se 0.033 0.098 0.117 0.188 0.238
```

Reading it (one illustrative draw — the *rates* live in the
[panel model card](../reference/model-cards/panel.md)'s Monte Carlo table):

- **The uncorrected FE row sinks below the truth as `h` grows** — at `h=4` it
  reports 0.064 against a true 0.328. That is the horizon-amplified Nickell
  bias, and it shrinks with `T`, not with `N`.
- **`jackknife=True` (Dhaene-Jochmans) moves the points and nothing else.**
  Its `se` row is bit-identical to FE's by construction: the plug-in SE is kept
  on an asymptotic-equivalence argument that has *not* arrived at `T=40`. The
  round-2 audit measured the cost: at `N=20, T=60, h=8` the correction removes
  the bias but inflates the estimator's dispersion ~36% while `se` doesn't
  move, costing 8pp of coverage (0.880 → 0.804, recovering by `T≈240`).
- **`bias_correction="spj"` (Mei-Sheng-Shi 2026) moves the points *and*
  recomputes the SEs** for the corrected estimator (residuals at the corrected
  coefficients, jackknife-adjusted scores, per their reference
  implementation) — here 0.153 → 0.238 at `h=4`, wider exactly where the
  correction is doing the most work. In the card's seeded Monte Carlo the SPJ
  route cuts the `h=2` bias by ~4-15x at `T∈{20,40}` and covers no worse than
  FE — but at `T=20` *neither* reaches the nominal 95% with Driscoll-Kraay
  standard errors (0.74 FE vs 0.82 SPJ at `h=2`): DK itself is a short-`T`
  approximation, the same caveat this page's `bandwidth` gotcha carries. Do
  not read "bias-corrected" as "exact bands at `T=20`".

The two corrections are the same `2·full − ½(half₁+half₂)` combination with
different bookkeeping (the DJ halves are self-contained windows; the MSS/SPJ
halves keep full-panel leads and lags and split the usable rows at their
median), so they coincide in the points only in the degenerate
no-lags-at-`h=0`-even-`T` case and differ everywhere else. Ask for one or the
other; asking for both raises.

## Gotchas

- **Shapes.** `outcome` is `N × T` (entities by time); `shock` is length `T`.
  `tsecon.panel_fe` takes the same `outcome` plus `regressors` as `k × N × T`
  and offers the same `se_type` menu.
- **`bandwidth`** sets the Driscoll-Kraay kernel width. Like any HAC bandwidth
  it moves the answer; report it.
- **`jackknife=True`** applies the Dhaene-Jochmans half-panel correction for
  Nickell bias to the *point estimates only* — the reported `se` is the
  unchanged full-sample one, which under-states the corrected estimator's
  dispersion at short `T` (measured: −8pp of coverage at `N=20, T=60, h=8`).
  When `T` is short — exactly where a correction matters — prefer
  `bias_correction="spj"`, which recomputes the SEs for the corrected
  estimator; see the section above and the
  [panel model card](../reference/model-cards/panel.md).
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
