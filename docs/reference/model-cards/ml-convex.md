# Model card — L1 trend filtering and componentwise L2 boosting

**Family:** `l1_trend_filter`, `boosting`

Two deterministic estimators from the machine-learning module's "convex and
greedy" corner. `l1_trend_filter` is Kim, Koh & Boyd's (2009) answer to the
Hodrick-Prescott filter: swap HP's squared penalty on second differences for
an L1 penalty and the trend becomes *piecewise linear*, with the kinks —
the knots — chosen by the data the way the LASSO chooses variables; the same
function computes the HP filter itself under `penalty="l2"`, so the two
trends can be compared on one objective. `boosting` is componentwise L2
boosting (Bühlmann & Yu 2003; Bühlmann 2006), the engine of R's
`mboost::glmboost`: a slow-learning variable selector that adds one
shrunken single-column least-squares fit per step, which macro forecasters
read as sequential ARDL building (Ng 2014). Both are seedless and
deterministic; neither forms an n×n matrix.

| Function | Role |
|----------|------|
| `l1_trend_filter` | Piecewise-linear (or -constant) trend with data-chosen knots; HP filter under `penalty="l2"` |
| `boosting` | Componentwise L2 boosting with corrected-AIC stopping; full coefficient / selection / df paths |

## What it estimates

- **`l1_trend_filter(y, lam, order=2, penalty="l1")`** — the trend `x`
  minimizing

      (1/2)·‖y − x‖² + lam·‖D x‖₁              (penalty="l1")
      (1/2)·‖y − x‖² + (lam/2)·‖D x‖²          (penalty="l2")

  with `D` the `order`-th difference operator. Under L1, most `order`-th
  differences of the trend are *exactly* zero: `order=2` gives a
  piecewise-linear trend (the Kim-Koh-Boyd filter), `order=1` a
  piecewise-constant one (the fused LASSO on the level / total-variation
  denoising). The indices where the difference is nonzero are the `knots`.
  Under L2 with `order=2` the minimizer is the Hodrick-Prescott trend for
  the *same* `lam` — `‖y − x‖² + lam‖Dx‖²` and `(1/2)‖y − x‖² +
  (lam/2)‖Dx‖²` have identical minimizers — so `lam=1600` is quarterly HP,
  and the suite asserts equality with `hp_filter` at 1e-10.
- **`boosting(x, y, learning_rate=0.1, n_steps=500, stop="aic")`** — from
  `F_0 = 0`, each step regresses the current residual on every column of
  `x` separately, keeps the column with the smallest residual sum of
  squares, and adds `learning_rate` times that least-squares fit to the
  model. After `m` steps the fit is `B_m y` for the boosting operator
  `B_m = B_{m−1} + ν H_j (I − B_{m−1})`, `H_j = x_j x_jᵀ / x_jᵀx_j`, whose
  trace is the effective degrees of freedom in Bühlmann's (2006) corrected
  AIC, `AIC_c(m) = log(RSS_m/n) + (1 + df_m/n)/(1 − (df_m + 2)/n)`.
  `stop="aic"` reports the minimizing step; the whole path is returned
  regardless.

## Assumptions

- **No intercept, no standardization** — as everywhere in the penalized
  family: `boosting` fits the objective on the design exactly as passed.
  Center `y` and center (typically standardize) the columns of `x` first;
  a nonzero mean in `y` is absorbed into the slopes and every coefficient
  is wrong. `l1_trend_filter` needs no centering (the trend carries the
  level).
- **`lam` scales with the data.** The trend-filter objective scales as
  `y²`, so `lam` scales with `y` — a `lam` tuned on one series does not
  transfer to another with different units. Scan `lam` *down from
  `lam_max`* (returned), the smallest value at which the L1 trend collapses
  to the least-squares polynomial of degree `order − 1`; below it, knots
  appear one at a time. Under L2 the usual HP conventions apply (1600
  quarterly, 129600 monthly, 6.25 annual).
- **`lam` is not comparable across penalties.** An L1 `lam` is a bound on
  the dual variable; an L2 `lam` is a smoothness weight. They share the
  data-fit term and nothing else.
- **The boosting operator is tracked exactly, not approximately.** `B_m`
  is kept in a rank-`m` factored form, `B_m = Σ ν x_{j_i} w_iᵀ`, so its
  trace is the same number a dense n×n update produces, to rounding
  (pinned at 1e-12 against a dense transcription). What is *not* done is
  materializing `B_m`; the fit at the chosen step is recomputed as
  `x @ coef`, which the dense-operator fit reproduces at 3.6e-15.
- **Boosting ties go to the smallest column index**, and there is no
  randomness anywhere: the `selected` sequence is a deterministic function
  of the inputs (seedless). Zero-norm columns are never selectable.

## When to use

- **`l1_trend_filter(order=2)`** when you want a trend that is linear
  between a few dates and want the *dates* chosen by the data — growth
  regimes, productivity-slowdown narratives, "the trend changed in 2008"
  arguments — rather than HP's smooth curve that spreads every change over
  years. Kim-Koh-Boyd's recommendation: read the knots as regime dates and
  the slopes as growth rates.
- **`l1_trend_filter(order=1)`** for level shifts (piecewise-constant
  means): a fast fused-LASSO segmentation to compare against the
  structural-breaks module's tests.
- **`l1_trend_filter(penalty="l2")`** when you want HP on the same
  footing as the L1 trend — same `lam` scale, same output keys, the same
  certificate — for a side-by-side.
- **`boosting`** as a slow, interpretable variable selector in a
  data-rich regression (many lags, many predictors): the selection *order*
  is the story, and `df_path`/`aic_path` show where the corrected AIC stops
  adding. With `learning_rate=1.0` it is unshrunk forward stagewise; with
  `learning_rate → 0` and many steps it converges to OLS on the columns it
  ever selects.

## Key arguments and defaults

| Call | Argument | Default | Notes |
|------|----------|---------|-------|
| `l1_trend_filter` | `lam` | — (required) | penalty weight, ≥ 0; scan down from the returned `lam_max` |
| | `order` | `2` | `2` piecewise-linear, `1` piecewise-constant; anything else raises |
| | `penalty` | `"l1"` | `"l1"` trend filtering, `"l2"` Hodrick-Prescott |
| | `tol` | `None` (→ `1e-8`) | relative duality gap at which the L1 solver stops; **inert under `"l2"`** and raises if passed there |
| | `max_iter` | `None` (→ `10000`) | interior-point Newton-step budget (20–60 typical); same sentinel rule |
| `boosting` | `learning_rate` | `0.1` | step shrinkage ν ∈ (0, 1]; outside that range raises |
| | `n_steps` | `500` | iterations run (≥ 1); the AIC step is searched inside them |
| | `stop` | `"aic"` | `"aic"` reports the corrected-AIC minimizer, `"none"` the last step |
| | `x_test` | `None` | optional `n_test × p` matrix scored with the reported coefficients |

## How to read the output

- **`l1_trend_filter`** → `{"trend", "cycle", "knots", "n_knots",
  "duality_gap", "objective", "converged", "n_iter", "lam_max"}`.
  `cycle = y − trend`. `knots` are indices into the `order`-th differences
  (`0..n−order`) where `|(D·trend)_i|` exceeds `max(1e-6·max|D y|,
  1e-12·max|y|)`; a knot at index `i` for `order=2` is a slope change
  between observations `i+1` and `i+2`. Under L1 the inactive differences
  are zero to rounding, so the count is meaningful; under L2 no
  difference is ever exactly zero and nearly every index is listed.
  **`duality_gap` is a certificate**: the objective at the returned trend
  minus a dual-feasible dual objective, so `objective − optimum ≤
  duality_gap` by weak duality — a number the user can check, not a
  solver's opinion. `converged` is `duality_gap ≤ tol·objective`; it is
  always `True` on the closed-form paths (`"l2"`, `lam=0`, `lam ≥ lam_max`,
  all with `n_iter=0`).
- **`boosting`** → `{"coef", "coef_path", "selected", "rss_path",
  "df_path", "aic_path", "best_step", "fitted", "predicted"}`. `best_step`
  is a **0-based index into the path arrays** (the model after
  `best_step + 1` iterations); `coef == coef_path[best_step]`. `selected[m]`
  is the column added at step `m`; `df_path[m] = tr(B_m)`; `aic_path`
  entries where `df_m + 2 ≥ n` are `+inf` (never selected). `predicted` is
  `None` unless `x_test` was given.

## Failure modes

- **Passing a raw `y`/`x` with a mean to `boosting`.** No intercept is fit;
  the slopes absorb the mean. Symptom: the first selected column is
  whichever best mimics a constant, and every coefficient is off. Fix:
  center `y`, standardize `x`.
- **Reading `lam` across series or penalties.** The objective scales as
  `y²`; use the returned `lam_max` as the yardstick (`lam = 0.05·lam_max`
  transfers, `lam = 24.7` does not). An L1 `lam` says nothing about the
  right HP `lam`.
- **A `tol` below the certificate's floating-point floor.** The
  certificate is evaluated in floating point and its floor is about 1e-11
  relative (the L1 term pays `lam` on the rounding residue of every
  inactive difference; larger for larger `lam²·n`). A `tol` below it cannot
  be certified: the stall detector ends the loop after ten iterations
  without a 1% improvement (about 40 total) and the call returns
  `converged=False` with the honest gap — the trend it returns is still
  the converged one to 1e-8. The default `1e-8` sits two to three decades
  above the floor. `max_iter` is likewise honest: a starved budget
  returns the last iterate with `converged=False` and its larger gap.
- **Exactly `lam = lam_max`.** The problem is degenerate there (one
  constraint active with a zero multiplier) and no solver's gap pins the
  trend tightly — by strong convexity a relative gap `g` only bounds the
  trend error by `√(2·g·objective)`. The crate returns the closed-form
  polynomial at `lam ≥ lam_max` using its own `lam_max`; a `lam` a
  rounding error below it goes through the solver and lands within ~1e-7
  of the polynomial. Use `lam ≥ 1.01·lam_max` if you mean "the polynomial".
- **`tol` / `max_iter` with `penalty="l2"`.** The L2 trend is one banded
  closed-form solve: nothing iterates. Passing either explicitly raises
  naming the kwarg and the fix (the sentinel convention of audit round 10)
  rather than being silently ignored.
- **Boosting `df_path` is not monotone.** The boosting operator is not
  symmetric, so `tr(B_m)` can dip by ~1e-5 near the saturated
  `learning_rate=1` limit. It is not a bug: the dense transcription agrees
  to 1e-15.
- **`aic_path` running to the last step.** With a small `learning_rate`
  the corrected AIC may still be falling at `n_steps`; then `best_step ==
  n_steps − 1` and the model is under-fit. Raise `n_steps` (cost is
  `O(n·(p + m))` per step, 0.09 s for `n=500, p=50, 500 steps`).

## Relation to the HP filter and the boosted HP filter

The three trends sit on one objective, `(1/2)‖y − x‖² + lam·φ(D₂x)`:

- **HP** is `φ = (1/2)‖·‖²` — a linear smoother, `x = (I + lam D₂ᵀD₂)⁻¹y`,
  which spreads every change in slope over many periods and, on a
  stochastic trend, under-smooths at the ends and leaves a residual with
  spurious cycles (Hamilton 2018). `l1_trend_filter(penalty="l2")` *is*
  this filter, same `lam`.
- **L1 trend filtering** is `φ = ‖·‖₁` — the same data-fit term, a
  nonlinear estimator: piecewise linear, sparse in slope changes, with the
  change dates estimated. It answers Hamilton's critique differently from
  the Hamilton filter: instead of abandoning the smoother it changes the
  penalty's geometry, and reports the fit's suboptimality as a certificate.
- **Boosted HP** (Phillips & Shi 2021; roadmap Module 1, not shipped)
  keeps `φ = (1/2)‖·‖²` and *iterates* HP on its own residual — an L2
  boosting of the HP smoother, `x_(m) = x_(m−1) + S(y − x_(m−1))` with `S`
  the HP operator — stopping by a BIC or ADF rule. It removes the
  under-smoothing of stochastic trends by re-applying the linear smoother;
  L1 trend filtering removes it by sparsity. The `boosting` function on
  this card is the componentwise (regression) version of the same
  L2-boosting idea: the base learner is a single column instead of the HP
  smoother, and the corrected AIC plays the role of Phillips-Shi's BIC
  stop. When a boosted HP lands in the filters module it will reuse this
  card's operator-trace bookkeeping.

## Validated against

Graded per leg, honestly:

- **`l1_trend_filter`, `penalty="l1"`** — (1) an **optimality
  certificate** re-derived from scratch in the Rust and Python tests for the
  crate's own trend on all 14 fixture cases (dual variable recovered from
  the residual by `order` cumulative sums, clipped into the dual box,
  `P(x) − G(v)` an upper bound by weak duality; asserted ≤ 1e-8 relative,
  achieved ≤ 3.3e-10 on the closed-form polynomial cases, ≤ 5.8e-11 on the
  interior-point cases, 1e-15 on the order-1 cases); (2) **cvxpy +
  Clarabel 0.11.1** (cvxpy 1.9.2; an independent interior-point solver) converged at 1e-14,
  each reference's own relative gap recorded in the fixture (1.6e-16 to
  1.6e-12): trends agree at 1e-8 absolute, achieved 1.4e-10; (3) the
  closed-form limits — `lam_max` at 1e-10 relative and the `np.polyfit`
  polynomial at `lam ≥ lam_max` at 1e-8 (achieved 1.4e-10), `lam → 0`
  returning the data at 1e-10.
- **`l1_trend_filter`, `penalty="l2"`** — the dense
  `np.linalg.solve(I + lam DᵀD, y)` closed form at 1e-10 (achieved
  1.7e-12) and the cross-surface identity with `hp_filter` at 1e-10
  (achieved 1.4e-12 across the suite's four series and three `lam` values).
- **`boosting`** — a **transcription** grade, not a third-party run: an
  independent dense NumPy transcription of the published algorithm with
  the boosting operator formed explicitly (so its trace is exact by
  construction) pins `coef_path`, `df_path`, `aic_path` at 1e-12 (achieved
  6.7e-16 / 2.7e-15 / 1.6e-15) and `selected` / `best_step` exactly on five
  cases; R mboost `glmboost`, the roadmap's target, is not runnable in the
  build environment, and cross-checking against it is an open follow-up.
  Properties: RSS nonincreasing; the small-step / many-step limit
  reproduces OLS on the selected support at 3.3e-14; AIC stopping recovers
  a sparse truth's support.
- **Timing** (release wheel): `l1_trend_filter` at `n = 10000` in 0.12 s
  (49 interior-point iterations; the L2 form in 4 ms); `boosting` at
  `n = 500, p = 50, n_steps = 500` in 0.09 s.

Fixture: [`fixtures/convex.json`](../../../fixtures/convex.json)
(generator `generate_convex_fixtures.py`, seeded, never imports tsecon).

## References

- Kim, S.-J., Koh, K., Boyd, S. & Gorinevsky, D. (2009). "ℓ₁ Trend
  Filtering." *SIAM Review* 51(2).
- Tibshirani, R. J. (2014). "Adaptive piecewise polynomial estimation via
  trend filtering." *Annals of Statistics* 42(1).
- Hodrick, R. & Prescott, E. (1997). "Postwar U.S. Business Cycles: An
  Empirical Investigation." *JMCB* 29(1).
- Phillips, P. C. B. & Shi, Z. (2021). "Boosting: Why You Can Use the HP
  Filter." *International Economic Review* 62(2).
- Bühlmann, P. & Yu, B. (2003). "Boosting with the L2 loss: regression and
  classification." *JASA* 98(462).
- Bühlmann, P. (2006). "Boosting for high-dimensional linear models."
  *Annals of Statistics* 34(2).
- Ng, S. (2014). "Viewpoint: Boosting recessions." *Canadian Journal of
  Economics* 47(1).

See also: [Penalized regression and leakage-safe validation](machine-learning.md).

## Runnable example

```python
import numpy as np
import tsecon

rng = np.random.default_rng(3)
n = 160
t = np.arange(n)
# A trend whose slope changes at t = 60 and t = 110, plus noise.
slope = np.where(t < 60, 0.20, np.where(t < 110, -0.10, 0.15))
y = np.cumsum(slope) + 0.6 * rng.standard_normal(n)

# 1. L1 trend filtering: scan lam down from lam_max (the polynomial limit).
lam_max = tsecon.l1_trend_filter(y, 1.0)["lam_max"]
fit = tsecon.l1_trend_filter(y, lam=0.05 * lam_max)
print("knots:", fit["knots"].tolist(), " converged:", fit["converged"],
      " relative gap: %.1e" % (fit["duality_gap"] / fit["objective"]))

# 2. The squared penalty is the Hodrick-Prescott filter under HP's own lam.
hp = tsecon.l1_trend_filter(y, lam=1600.0, penalty="l2")
print("HP identity, max |diff|: %.1e"
      % np.max(np.abs(hp["trend"] - tsecon.hp_filter(y, lamb=1600.0)["trend"])))

# 3. Componentwise L2 boosting on a sparse truth (centered y, standardized X).
X = rng.standard_normal((120, 10))
X = (X - X.mean(0)) / X.std(0)
beta = np.array([2.0, -1.5, 1.0, 0, 0, 0, 0, 0, 0, 0])
yb = X @ beta + 0.5 * rng.standard_normal(120)
yb -= yb.mean()
b = tsecon.boosting(X, yb, learning_rate=0.1, n_steps=500)
print("AIC-chosen step:", b["best_step"], " df: %.2f" % b["df_path"][b["best_step"]],
      " nonzero columns:", np.flatnonzero(b["coef"]).tolist())
print("coef[:3]:", np.round(b["coef"][:3], 3))
```

Expected output:

```
knots: [52, 57, 106, 111]  converged: True  relative gap: 6.1e-12
HP identity, max |diff|: 2.4e-12
AIC-chosen step: 141  df: 4.16  nonzero columns: [0, 1, 2, 3, 4, 7]
coef[:3]: [ 2.039 -1.505  0.981]
```

The two true slope changes (at 60 and 110) show up as knot *pairs* two
observations apart — the L1 solution splits a kink across neighbouring
second differences when the data cannot resolve the exact date — and the
boosted model keeps the three true signals near their values (2, −1.5, 1)
while three noise columns carry small coefficients the AIC could not
reject; a slower `learning_rate` or the LASSO path's BIC is the usual
tightening.
