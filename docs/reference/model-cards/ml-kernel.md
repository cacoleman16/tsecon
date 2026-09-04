# Model card — Kernel methods: kernel ridge and nonparametric regression

**Family:** `kernel_ridge`, `kernel_regression`

The two kernel-method workhorses for nonlinear conditional means: a
tuning-light nonlinear *predictor* (kernel ridge regression, exact or with
random Fourier features for long samples) and a *smoother* you can read
(Nadaraya-Watson and local-linear regression in up to three regressors). Both
reproduce their reference implementations exactly — scikit-learn's
`KernelRidge` and statsmodels' `KernelReg` — and the smoother adds the one
thing the Python ecosystem lacks for time-series regressors: a bandwidth
selector that does not chase serially correlated noise, and a Rust core that
makes cross-validating a 2,000-observation smoother a matter of seconds
rather than hours.

| Function | Role |
|----------|------|
| `kernel_ridge` | Kernelized ridge (rbf / laplacian / polynomial / linear); exact `O(n^3)` dual solve or the Rahimi-Recht random-Fourier-feature approximation |
| `kernel_regression` | Nadaraya-Watson (`"nadaraya_watson"`) and local-linear (`"local_linear"`) product-Gaussian smoothing with fixed, leave-one-out, or leave-block-out bandwidths |

## What it estimates

- **`kernel_ridge(x, y, alpha, kernel, ...)`** — the minimizer of
  ‖y − f(x)‖² + α‖f‖²_H over the reproducing-kernel Hilbert space of the
  kernel, whose representer solution `f(x) = Σᵢ aᵢ k(x, xᵢ)` has dual
  coefficients `(K + αI) a = y`. This is scikit-learn's `KernelRidge`
  objective — **no 1/n factor and no intercept**, the same scale as `ridge`
  (with the linear kernel the two are the same fit). Kernels, in
  scikit-learn's exact parameterization: rbf `exp(−γ‖x−y‖²)`, laplacian
  `exp(−γ‖x−y‖₁)`, polynomial `(γ⟨x,y⟩ + coef0)^degree`, linear `⟨x,y⟩`;
  `gamma=None` is scikit-learn's `1 / n_features`. With `rff_features=D` the
  rbf kernel is replaced by its `D`-dimensional random Fourier feature map
  `z(x) = √(2/D) cos(Wx + b)`, `W ~ N(0, 2γI)`, `b ~ U[0, 2π)` (Rahimi &
  Recht 2007), drawn from a Philox stream keyed by `seed`, and ridge is
  solved in the primal — `O(nD²)` instead of `O(n³)`, converging to the exact
  fit as `D` grows.
- **`kernel_regression(x, y, bandwidth, kind, ...)`** — the conditional mean
  `E[y | x]` for `x` of one to three columns with the product Gaussian kernel
  `K_h(xᵢ − x) = Πⱼ φ((xᵢⱼ − xⱼ)/hⱼ) / Πⱼ hⱼ`. `"nadaraya_watson"` is the
  local constant `Σᵢ K_h(xᵢ − x) yᵢ / Σᵢ K_h(xᵢ − x)`; `"local_linear"` (the
  default) is the intercept of the kernel-weighted least squares of `y` on
  `[1, xᵢ − x]`, solved through the pseudoinverse exactly as statsmodels'
  `KernelReg(reg_type="ll")` does. Local linear has no boundary bias and a
  bias that does not depend on the design density (Fan 1992), which is why
  it is the default. The fit is a linear smoother `ŷ = S y`; `effective_df`
  is `tr(S)`.
- **Bandwidth selection.** `bandwidth_method="loo_cv"` minimizes the
  leave-one-out least-squares criterion `n⁻¹ Σᵢ (yᵢ − ĝ₋ᵢ(xᵢ))²`
  (statsmodels' `cv_loo`). `"block_cv"` minimizes the **leave-block-out**
  criterion of Chu & Marron (1991) and Hart & Vieu (1990): predicting `yᵢ`
  drops the `2·block + 1` observations with `|j − i| ≤ block` (default
  `block = ⌈n^{1/3}⌉`), so neighbours whose errors are correlated with `eᵢ`
  never vote on `yᵢ`. Selection is deterministic: a 21-point log-spaced grid
  on a common multiple of the Scott reference `1.06·sd(xⱼ)·n^{−1/(4+k)}` over
  `[0.05, 20]`, golden-section refinement of the bracket around the grid
  minimum, and for `k ≥ 2` per-column coordinate refinement (a wide
  per-column grid, then narrow golden-section polishing rounds per column
  and along the common scale until a round stops improving). It is not
  statsmodels' Nelder-Mead path; on the fixture it reaches a criterion no
  worse than `fmin`'s on all four cases (equal to 1e-9 relative on three,
  strictly better on the fourth).

## Assumptions

- **`kernel_ridge` fits no intercept and does not standardize.** Center `y`
  unless the kernel can absorb a level (rbf/laplacian and polynomial with
  `coef0 > 0` can; linear cannot), and put the columns of `x` on comparable
  scales — every kernel here is a function of Euclidean or `L1` distances or
  inner products, so a column measured in thousands dominates the kernel.
- **`alpha` is on scikit-learn's `KernelRidge`/`Ridge` scale** (no 1/n).
- **`kernel_regression` is a product kernel with one bandwidth per column**,
  in each column's own units (the Gaussian kernel's standard deviation).
  Columns are not rescaled; the reference bandwidth the search starts from
  is proportional to each column's spread, so differing scales are handled
  by the bandwidths, not by the user.
- **Regressors, not the series' own lags, are the intended use of leave-one-out.**
  For a nonlinear autoregression (`x = y_{t−1}`) or any regression with
  serially correlated errors, leave-one-out undersmooths — the smoother
  reproduces the correlated neighbours' noise, and leave-one-out cannot see
  that because the neighbours are still in the training set (Hart 1991;
  Opsomer, Wang & Yang 2001). Use `bandwidth_method="block_cv"`.
- **`k ≤ 3`.** A product-kernel smoother in more dimensions needs samples
  that grow exponentially in `k` and is uninformative at econometric sizes;
  the call refuses and points to `kernel_ridge`.
- **Gaussian kernel only** for `kernel_regression`. It is the kernel
  statsmodels validates against, and its unbounded support keeps every local
  fit defined. Compact-support kernels are deferred: statsmodels' `tricube`
  gives points outside the support *full* weight (a bug in the reference),
  so there is nothing honest to pin one to.

## When to use

- **`kernel_ridge`** as the nonlinear benchmark in a forecasting horse race —
  the "tuning-light nonlinear baseline" of Goulet Coulombe et al. (2022):
  two hyperparameters (`alpha`, `gamma`), a convex problem, no local optima.
  Use `rff_features` once `n` is in the thousands and the `n × n` kernel
  matrix (and its `O(n³)` solve) is the bottleneck; `D` in the low thousands
  is typically within a few percent of the exact fit.
- **`kernel_regression`** when you want to *see* the conditional mean — a
  nonlinear Phillips curve, a threshold-like response, an Engel curve — with
  an interpretable smoothing parameter and a degrees-of-freedom count, and
  when the regressor count is one to three.
- **`"block_cv"`** whenever the observations are ordered in time and the
  errors may be autocorrelated — which is every nonlinear autoregression.
  `"loo_cv"` is the right comparison to statsmodels and the right choice for
  independent errors.
- **`"fixed"`** to reproduce a published bandwidth or to draw the smoother
  at several bandwidths and read `effective_df` at each.

## Key arguments and defaults

| Call | Argument | Default | Notes |
|------|----------|---------|-------|
| `kernel_ridge` | `alpha` | `1.0` | RKHS-norm penalty; `0` is the interpolating fit and is refused when `K` is not positive definite (scikit-learn silently falls back to least squares there) |
| | `kernel` | `"rbf"` | `"rbf"`, `"laplacian"`, `"polynomial"`, `"linear"` |
| | `gamma` | `None` → `1 / n_features` | refused with `"linear"` (it has no width) |
| | `degree` / `coef0` | `3` / `1.0` | polynomial only; non-default values are refused with other kernels |
| | `rff_features` | `None` (exact) | `D` random Fourier features; rbf only, refused otherwise |
| | `seed` | `0` | RFF draws (Philox); a non-zero seed is refused in exact mode, where nothing is drawn |
| | `x_test` | `None` | adds `predicted` |
| `kernel_regression` | `bandwidth` | `None` | required under `"fixed"`: a positive scalar (broadcast) or one value per column; refused under the CV methods |
| | `kind` | `"local_linear"` | or `"nadaraya_watson"` |
| | `kernel` | `"gaussian"` | the only accepted value in this slice |
| | `bandwidth_method` | `"fixed"` | `"fixed"`, `"loo_cv"`, `"block_cv"` |
| | `block` | `None` → `⌈n^{1/3}⌉` | leave-block-out half-width (`block_cv` only; refused elsewhere; `0` is refused — that is `"loo_cv"`) |
| | `x_test` | `None` | adds `predicted` |

## How to read the output

- **`kernel_ridge`** → `{"dual_coef" | "coef", "fitted", "predicted"?,
  "kernel", "gamma", "n_rff_features"}`. `dual_coef` (exact mode) is
  scikit-learn's `dual_coef_`, the `aᵢ` of `f(x) = Σᵢ aᵢ k(x, xᵢ)`; `coef`
  (RFF mode) holds the `D` primal weights on the feature map. `gamma` is the
  resolved width (`None` for linear); `n_rff_features` is `None` in exact
  mode. `predicted` appears only when `x_test` is given.
- **`kernel_regression`** → `{"fitted", "predicted"?, "bandwidth",
  "bandwidth_method", "block", "cv_criterion", "effective_df", "kind",
  "kernel", "bandwidth_at_boundary", "n_criterion_evaluations"}`.
  `bandwidth` is the resolved per-column vector. `cv_criterion` is the
  leave-one-out criterion under `"fixed"` and `"loo_cv"` (statsmodels'
  `cv_loo` at that bandwidth) and the leave-block-out criterion under
  `"block_cv"`. `effective_df = tr(S)` runs from `k + 1` (local linear) or
  `1` (Nadaraya-Watson) at huge bandwidths to `n` at tiny ones — read it the
  way you read a regression's parameter count. `bandwidth_at_boundary` is
  the honesty flag: `True` means a selected bandwidth ended on a wall of the
  search range (within 1%) — the criterion was still falling at the edge, so
  the reported bandwidth is the search's limit, not an interior optimum.
  That is what a target with no detectable signal produces (the criterion
  wants `h → ∞`, the global fit) and it is proven to fire in the test suite
  on pure noise. `predicted` is `NaN` for a test point so far from every
  training row that all kernel weights underflow.

## Failure modes

- **Leave-one-out on a nonlinear autoregression.** With `x = y_{t−1}` and
  `y` an AR process, `"loo_cv"` selects a tiny bandwidth and an
  `effective_df` near `n`: the smoother is interpolating the noise. On the
  test suite's AR(1)-error design (`ρ = 0.9`, `n = 200`), leave-one-out picks
  `h ≈ 0.03` — the search's lower wall, `bandwidth_at_boundary = True` — on
  every one of ten seeds, while `"block_cv"` (`block = 10`) picks `0.3` to
  `13` and is wider on 10 of 10. The fix is the method, not a larger sample.
- **A block that does not cover the correlation length.** The default
  `block = ⌈n^{1/3}⌉` is a rate, not a diagnosis: the criterion only stops
  seeing the correlated noise once the error autocorrelation at lag `block`
  is negligible. In the runnable example below (`ρ = 0.8`, `n = 300`,
  `block = 7`, `0.8⁷ ≈ 0.21`) block-CV still lands at `effective_df ≈ 26`
  for a target with three or four degrees of freedom in it — far better than
  leave-one-out's 83, but not the answer either. Look at the residual
  autocorrelation and set `block` where it dies out.
- **A conflicting argument.** `bandwidth` with a CV method, `block` with
  `"fixed"`/`"loo_cv"`, `gamma` with `"linear"`, `rff_features` with a
  non-rbf kernel, `seed` in exact mode, `degree`/`coef0` off their defaults
  with a non-polynomial kernel: every one is refused with an error naming
  the argument, the mode that would use it, and the fix. Nothing is
  silently ignored.
- **`alpha = 0` with duplicate or near-duplicate rows** (or a polynomial /
  linear kernel of rank `p < n`): `K` is singular and the exact solve
  refuses, naming `alpha`. scikit-learn warns and returns a least-squares
  solution of a different problem; tsecon does not.
- **Unscaled columns in `kernel_ridge`.** The rbf distance is dominated by
  the widest column and the fit ignores the others. Standardize first.
- **Too few observations.** The smoother needs `k + 1` rows (local linear)
  or one row (Nadaraya-Watson) left after the exclusion window is removed;
  the error states the exact minimum in the house wording (`insufficient
  data: 3 observations, at least 4 required`).

## Validated against

- **`kernel_ridge`: independent package.** scikit-learn 1.9.0
  `KernelRidge` — `dual_coef_`, `predict(X)` and `predict(X_test)` for two
  parameterizations of each of the four kernels (eight cases, `n = 60`,
  `p = 3`, 15 test rows), asserted at 1e-8, **achieved 2.2e-12**. The
  random-Fourier-feature mode is a Monte-Carlo object and is not
  golden-pinned: the crate's property tests pin seeded determinism (same
  seed bit-identical, different seeds differ) and convergence to the exact
  fit — RMSE against the exact fitted values 0.307 → 0.049 → 0.016 at
  `D = 20, 200, 2000` on a seeded `n = 150` design.
- **`kernel_regression` fitted values and the LOO criterion: independent
  package.** statsmodels 0.15.0 `KernelReg(reg_type="lc"|"ll",
  var_type="c"*k)` at fixed bandwidths — twelve cases (`k = 1`: three
  bandwidths × both estimators, `n = 100`; `k = 2`: three bandwidth pairs ×
  both, `n = 90`), `fit()` at the training rows and at `x_test` asserted at
  1e-8, **achieved 6.7e-15**; `cv_loo(bw, func)` asserted at 1e-10,
  **achieved 3.0e-15**. statsmodels' own `bw="cv_ls"` optimum is used as a
  property target only: the criterion our search reaches is no worse on all
  four (series, estimator) cases, and the criterion at statsmodels' optimum
  is reproduced at 1e-10.
- **The leave-block-out criterion and `effective_df`: documented-formula
  transcription.** No package computes either; the generator transcribes
  the Chu-Marron criterion and `tr(S)` into NumPy and pins them at 1e-10
  (**achieved 2.7e-15 and 3.6e-15**). The transcription is grounded where it
  overlaps statsmodels: with `l = 0` it reproduces `cv_loo`, and its local
  fits reproduce `fit()`, both asserted at 1e-12 at generation time and
  recorded in the fixture's `_meta.transcription_checks`.
- **Properties** (`kernel_properties.rs`, 14 tests): the dual residual
  `(K + αI)a − y` is zero for every kernel; the linear kernel reproduces
  `ridge`; `effective_df` hits `1`/`k+1` at `h = 10⁴` and `n` at `h = 10⁻³`;
  the wide-bandwidth limits are the global mean and OLS; the selected
  bandwidth is a local minimum of its criterion (±5%); block-CV is wider
  than LOO on 10/10 AR(1) seeds; the boundary flag fires on noise and not
  on signal; every teaching error names its argument.

Fixture: [`fixtures/kernel.json`](../../../fixtures/kernel.json), generated
by `fixtures/generate_kernel_fixtures.py` (never imports tsecon).

## Speed (measured)

statsmodels' `KernelReg` evaluates each local fit in a Python loop over the
prediction points, and `cv_loo` re-slices the sample `n` times, so both are
`O(n²)` in Python. Measured in this environment (4 cores, best of several
runs, `n = 2000`, `k = 1`, local linear, fixed bandwidth):

| Task (`n = 2000`) | statsmodels 0.15.0 | tsecon | ratio |
|---|---|---|---|
| local-linear fit at a fixed bandwidth, `k = 1` (tsecon's one call also returns the LOO criterion and `tr(S)`) | 0.59 s (`fit()`) | 0.16 s | 3.6× |
| the LOO criterion at one bandwidth, `k = 1` | 0.97 s (`cv_loo`) | 0.16 s (the same call) | 5.9× |
| Nadaraya-Watson fit / LOO criterion, `k = 1` | 0.37 s / 0.59 s | 0.085 s | 4.4× / 7.0× |
| full leave-one-out bandwidth search, `k = 1` | 22.0 s (`bw="cv_ls"`, Nelder-Mead, ~23 evaluations) | 6.5 s (48 evaluations) | 3.4× — the same optimum: `h = 0.1793`, criterion `0.08865414` on both |
| full leave-block-out search, `k = 1`, `block = 13` | no counterpart | 5.7 s | — |
| local-linear fit at a fixed bandwidth, `k = 2` | 1.28 s | 0.24 s | 5.4× |
| full leave-one-out bandwidth search, `k = 2` | 59.4 s (~50 evaluations) | 48.4 s (362 evaluations, including the per-column polishing rounds) | 1.2× — the same optimum to five digits, identical criterion `0.09495446` |
| `kernel_ridge`, rbf, `p = 3`, exact solve | scikit-learn 1.9.0 `KernelRidge`: 13.8 s | 0.48 s (agreement 2e-14) | 29× — scikit-learn's figure is dominated by its LAPACK backend in this environment; both are `O(n³)` |
| `kernel_ridge`, rbf, random Fourier features `D = 500` | no counterpart | 0.43 s (RMSE 0.046 against the exact fit) | — |

Read the smoother rows honestly: both implementations are bound by the same
`n²` kernel evaluations (four million `exp` calls per criterion pass), so the
per-pass gain is a constant 4–7×, not an order of magnitude, and the
search-level gain is smaller still because statsmodels' Nelder-Mead needs
fewer criterion evaluations than the global grid plus golden section (which
buys robustness to spurious local minima and the boundary diagnostic, not
speed). Wall-clock times varied by ±50% between runs on this shared host;
the ratios were stable. The one thing that is not a constant factor is the
leave-block-out criterion, which statsmodels does not have at any speed.

The bandwidth search costs one criterion evaluation per grid point or
golden-section step; `n_criterion_evaluations` in the output is the count.

## References

- Nadaraya, E. (1964). "On estimating regression." *Theory of Probability
  and its Applications* 9; Watson, G. (1964). "Smooth regression analysis."
  *Sankhyā A* 26.
- Fan, J. (1992). "Design-adaptive nonparametric regression." *JASA* 87;
  Fan, J. & Gijbels, I. (1996). *Local Polynomial Modelling and Its
  Applications*. Chapman & Hall.
- Li, Q. & Racine, J. (2007). *Nonparametric Econometrics: Theory and
  Practice*. Princeton, ch. 2 (the `KernelReg` conventions).
- Chu, C.-K. & Marron, J. S. (1991). "Comparison of two bandwidth selectors
  with dependent errors." *Annals of Statistics* 19; Hart, J. & Vieu, P.
  (1990). "Data-driven bandwidth choice for density estimation based on
  dependent data." *Annals of Statistics* 18.
- Opsomer, J., Wang, Y. & Yang, Y. (2001). "Nonparametric regression with
  correlated errors." *Statistical Science* 16.
- Hastie, T. & Tibshirani, R. (1990). *Generalized Additive Models*, sec.
  3.5 (effective degrees of freedom).
- Rahimi, A. & Recht, B. (2007). "Random features for large-scale kernel
  machines." *NeurIPS* 20.
- Goulet Coulombe, P., Leroux, M., Stevanovic, D. & Surprenant, S. (2022).
  "How is machine learning useful for macroeconomic forecasting?" *Journal
  of Applied Econometrics* 37.

See the guide: [Machine Learning for Time Series](../../guide/12-machine-learning.md).

## Runnable example

```python
import numpy as np
import tsecon

rng = np.random.default_rng(7)
n = 300
x = np.linspace(-3, 3, n)
# A nonlinear mean with AR(1) errors: the classic case where leave-one-out
# cross-validation undersmooths.
e = np.zeros(n)
for t in range(1, n):
    e[t] = 0.8 * e[t - 1] + 0.4 * rng.standard_normal()
y = np.sin(x) + e

# 1. Local-linear smoothing at a fixed bandwidth (statsmodels KernelReg "ll").
fixed = tsecon.kernel_regression(x, y, bandwidth=0.4)
print("fixed h=0.4: effective df", round(fixed["effective_df"], 1),
      " LOO criterion", round(fixed["cv_criterion"], 4))

# 2. Leave-one-out vs leave-block-out bandwidth selection.
loo = tsecon.kernel_regression(x, y, bandwidth_method="loo_cv")
blk = tsecon.kernel_regression(x, y, bandwidth_method="block_cv")
print("loo_cv:   h", round(loo["bandwidth"][0], 3), " df", round(loo["effective_df"], 1),
      " at boundary:", loo["bandwidth_at_boundary"])
print("block_cv: h", round(blk["bandwidth"][0], 3), " df", round(blk["effective_df"], 1),
      " block", blk["block"])

# 3. Kernel ridge as a nonlinear predictor, exact and with random Fourier
#    features, with out-of-sample predictions.
X = rng.standard_normal((400, 3))
Y = np.sin(X[:, 0]) + 0.5 * X[:, 1] ** 2 - X[:, 2] + 0.3 * rng.standard_normal(400)
Xt = rng.standard_normal((50, 3))
exact = tsecon.kernel_ridge(X, Y, alpha=0.5, x_test=Xt)
rff = tsecon.kernel_ridge(X, Y, alpha=0.5, x_test=Xt, rff_features=1000, seed=1)
print("krr gamma", round(exact["gamma"], 3), " rff features", rff["n_rff_features"])
print("rff vs exact prediction rmse",
      round(float(np.sqrt(np.mean((rff["predicted"] - exact["predicted"]) ** 2))), 3))
```

Expected output:

```
fixed h=0.4: effective df 7.4  LOO criterion 0.3195
loo_cv:   h 0.029  df 82.8  at boundary: True
block_cv: h 0.096  df 26.4  block 7
krr gamma 0.333  rff features 1000
rff vs exact prediction rmse 0.036
```
