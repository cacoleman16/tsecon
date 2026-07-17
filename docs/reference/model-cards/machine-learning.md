# Model card — Penalized regression and leakage-safe validation

**Family:** `ridge`, `lasso`, `elastic_net`, `adaptive_lasso`, `lasso_path`,
`cv_splits`

Shrinkage and sparsity for the high-dimensional, many-predictor regressions
that show up in nowcasting, factor selection, and predictor screening — plus
the one piece of machinery that makes machine learning on time series honest:
cross-validation splits that never let the future leak into the past. The
penalized estimators reproduce scikit-learn's objectives exactly; the value
here is a fast Rust core and a validation scheme built for sequential data.

| Function | Role |
|----------|------|
| `ridge` | L2 shrinkage (dense, closed form) |
| `lasso` | L1 shrinkage (sparse, coordinate descent) |
| `elastic_net` | L1+L2 blend via `l1_ratio` |
| `adaptive_lasso` | Weighted L1 with the oracle property (Zou 2006) |
| `lasso_path` | The full regularization path with AIC/BIC selection |
| `cv_splits` | Leakage-safe expanding / rolling / purged-k-fold indices |

## What it estimates

- **`ridge(x, y, alpha)`** — minimizes ‖y − Xβ‖² + α‖β‖², the closed-form
  ridge solution. Shrinks all coefficients toward zero; never sets them
  exactly to zero.
- **`lasso(x, y, alpha)`** — minimizes ‖y − Xβ‖²/(2n) + α‖β‖₁ by coordinate
  descent. Produces exact zeros, so it selects.
- **`elastic_net(x, y, alpha, l1_ratio)`** — the α[ l1_ratio·‖β‖₁ +
  ½(1−l1_ratio)·‖β‖² ] penalty; `l1_ratio=1` is lasso, `0` is ridge. Handles
  correlated predictor groups better than pure lasso.
- **`adaptive_lasso(x, y, alpha)`** — a two-stage lasso whose per-coefficient
  L1 weights are 1/|β̂|^γ from an initial fit, giving Zou's (2006) oracle
  property: consistent selection *and* √n-normal estimation of the non-zeros.
- **`lasso_path(x, y)`** — the elastic-net solution across a decreasing grid of
  λ, with residual sum of squares, degrees of freedom, and AIC/BIC at each
  knot, returning the AIC- and BIC-optimal indices.
- **`cv_splits(n, ...)`** — index sets for time-series cross-validation:
  expanding or rolling origins, or de Prado's purged k-fold with an embargo for
  overlapping-label problems. It returns indices only; you do the fitting.

## Assumptions

- **No intercept, no standardization.** These estimators fit the objective on
  the design *exactly as passed* — they match scikit-learn with
  `fit_intercept=False`. The penalty is not scale-invariant, so **center `y`
  and standardize the columns of `X` yourself** before calling, or the fit is
  biased and the penalty falls unevenly across predictors. This is the most
  important line on this card.
- **`alpha` is on scikit-learn's scale**, not glmnet's λ. For `lasso`/
  `elastic_net` the least-squares term carries the 1/(2n) factor; for `ridge`
  it does not (the `Ridge` convention).
- **AIC/BIC in `lasso_path`** use the active-set size as the degrees of
  freedom (the Zou-Hastie-Tibshirani result for the lasso) — a heuristic that
  is standard but not exact under heavy correlation.
- **`cv_splits` guarantees no forward leakage**: every test index is strictly
  later than its training block (expanding/rolling), and purged k-fold removes
  `purge` observations around each test fold plus an `embargo` after it. It
  does *not* protect against leakage you introduce elsewhere (e.g. scaling on
  the full sample before splitting).

## When to use

- **`ridge`** when predictors are many and collinear and you want stable,
  dense coefficients (all retained, just shrunk).
- **`lasso`** when you want a sparse, interpretable subset — predictor
  screening, nowcasting indicator selection.
- **`elastic_net`** when predictors come in correlated groups and pure lasso
  arbitrarily keeps one and drops the rest.
- **`adaptive_lasso`** when you need the *selected set itself* to be
  trustworthy (inference after selection), not just good prediction.
- **`lasso_path`** when you want AIC/BIC to pick λ instead of cross-validation
  — fast, and it returns the whole coefficient trajectory for a plot.
- **`cv_splits`** for *any* hyperparameter tuning or model comparison on
  sequential data. Ordinary k-fold shuffles the future into the training set
  and reports fantasy accuracy.

## Key arguments and defaults

| Call | Argument | Default | Notes |
|------|----------|---------|-------|
| `ridge` | `alpha` | — (required) | L2 strength; larger ⇒ more shrinkage |
| `lasso` | `alpha` | — (required) | L1 strength |
| | `tol` / `max_iter` | `1e-8` / `100000` | coordinate-descent stopping |
| `elastic_net` | `l1_ratio` | `0.5` | 1 = lasso, 0 = ridge |
| `adaptive_lasso` | `l1_ratio` | `1.0` | pure adaptive-L1 by default |
| | `gamma` | `1.0` | weight exponent 1/|β̂|^γ |
| `lasso_path` | `l1_ratio` | `1.0` | pure lasso path |
| | `n_lambdas` | `100` | grid resolution |
| | `eps` | `1e-3` | λ_min / λ_max ratio |
| `cv_splits` | `scheme` | `"expanding"` | `expanding`, `rolling`, `purged_kfold` |
| | `train` | `0` | initial/fixed train length (0 = auto for expanding) |
| | `horizon` | `1` | test-block length |
| | `step` | `1` | origin increment |
| | `k` / `purge` / `embargo` | `5` / `0` / `0` | purged-k-fold controls |

## How to read the output

- **`ridge`** → a bare coefficient array of length k. **`lasso`**,
  **`elastic_net`**, **`adaptive_lasso`** → `{"coef", "n_iter", "max_change"}`;
  `max_change` is the final coordinate-descent step size (a convergence check).
  Count `np.sum(coef != 0)` for the selected-set size.
- **`lasso_path`** → `{"lambdas", "coefs", "rss", "df", "aic", "bic",
  "aic_best", "bic_best"}`. `coefs` is `n_lambdas × k`; `*_best` are the row
  indices of the AIC- and BIC-optimal fits. `path["coefs"][path["bic_best"]]`
  is the BIC-selected coefficient vector.
- **`cv_splits`** → a list of `{"train": [...], "test": [...]}` index dicts.
  Iterate, slice your arrays with the indices, fit on `train`, score on `test`.

## Failure modes

- **Passing raw `X`/`y` with an intercept in the data.** Because no intercept
  is fit, a non-zero mean in `y` is absorbed into the slopes and every
  coefficient is wrong. Symptom: coefficients that make no sense and a
  suspiciously poor fit. Fix: center `y`, standardize `X`.
- **Un-standardized predictors.** The penalty then depends on each column's
  units — a predictor measured in thousands is penalized far less than one in
  units. Always standardize before penalizing.
- **glmnet-scale `alpha`.** Values tuned in R's glmnet will not transfer; use
  scikit-learn's scale or tune via `cv_splits`.
- **Ordinary k-fold on time series.** Shuffled folds leak the future and
  produce optimistic CV error. Use `cv_splits`.
- **Overlapping labels without purging.** If your target at t depends on data
  through t+h (multi-period returns), adjacent train/test points share
  information; use `scheme="purged_kfold"` with `purge≥h` and a positive
  `embargo`.

## Validated against

`ridge`, `lasso`, and `elastic_net` reproduce scikit-learn's `Ridge`, `Lasso`,
and `ElasticNet` (with `fit_intercept=False`) — verified to match coefficient
for coefficient in the test suite. `adaptive_lasso` and `lasso_path` are
validated against documented Zou (2006) / Zou-Hastie-Tibshirani (2007)
formulas, and `cv_splits` against the leakage invariants (every test index
strictly post-dates its training block; purge/embargo gaps enforced).
Fixtures: [`fixtures/ml.json`](../../../fixtures/ml.json) and
[`fixtures/predreg.json`](../../../fixtures/predreg.json).

## References

- Hoerl, A. & Kennard, R. (1970). "Ridge Regression." *Technometrics* 12.
- Tibshirani, R. (1996). "Regression Shrinkage and Selection via the Lasso."
  *JRSS-B* 58.
- Zou, H. & Hastie, T. (2005). "Regularization and variable selection via the
  elastic net." *JRSS-B* 67.
- Zou, H. (2006). "The Adaptive Lasso and Its Oracle Properties." *JASA* 101.
- de Prado, M. L. (2018). *Advances in Financial Machine Learning*, ch. 7
  (purged k-fold, embargo).

See the guide: [Machine Learning for Time Series](../../guide/12-machine-learning.md).

## Runnable example

```python
import numpy as np
import tsecon

rng = np.random.default_rng(2)
n, p = 120, 8
X = rng.standard_normal((n, p))
beta = np.array([3.0, -2.0, 0.0, 0.0, 1.5, 0.0, 0.0, 0.0])   # sparse truth
y = X @ beta + 0.5 * rng.standard_normal(n)

# These estimators fit NO intercept and do NOT standardize (scikit-learn's
# fit_intercept=False). Center y (and typically standardize X) first.
Xc = (X - X.mean(0)) / X.std(0)
yc = y - y.mean()

# 1. Ridge: dense shrinkage (returns a coefficient array).
b_ridge = tsecon.ridge(Xc, yc, alpha=1.0)

# 2. Lasso: sparse selection (returns a dict).
lasso = tsecon.lasso(Xc, yc, alpha=0.1)
print("lasso nonzeros:", int(np.sum(lasso["coef"] != 0)), "/", p)

# 3. Elastic net: ridge-lasso blend via l1_ratio.
en = tsecon.elastic_net(Xc, yc, alpha=0.1, l1_ratio=0.5)

# 4. Adaptive lasso: data-driven weights, oracle selection.
al = tsecon.adaptive_lasso(Xc, yc, alpha=0.1)
print("adaptive-lasso nonzeros:", int(np.sum(al["coef"] != 0)), "/", p)

# 5. The full elastic-net path with AIC/BIC model selection.
path = tsecon.lasso_path(Xc, yc)
b_bic = path["coefs"][path["bic_best"]]
print("BIC-selected lambda index:", path["bic_best"],
      " nonzeros:", int(np.sum(b_bic != 0)))

# 6. Leakage-safe cross-validation splits for tuning alpha on sequential data.
splits = tsecon.cv_splits(n, scheme="expanding", train=40, horizon=10, step=10)
print("n folds:", len(splits), " fold 0 test:", splits[0]["test"][:3], "...")

# Purged k-fold with an embargo (de Prado) for overlapping-label problems.
purged = tsecon.cv_splits(n, scheme="purged_kfold", k=5, purge=2, embargo=2)
print("purged folds:", len(purged))
```

Expected output:

```
lasso nonzeros: 3 / 8
adaptive-lasso nonzeros: 3 / 8
BIC-selected lambda index: 57  nonzeros: 1
n folds: 8  fold 0 test: [40, 41, 42] ...
purged folds: 5
```
