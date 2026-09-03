# Model card — Regression trees and random forests

**Family:** `regression_tree`, `random_forest`

The native tree ensemble of the machine-learning module: a CART regression
tree that reproduces scikit-learn's `DecisionTreeRegressor` exactly, and a
regression forest built from it with the pieces a time-series user needs and
scikit-learn does not offer — block and stationary bootstrap resampling, an
out-of-bag error that is *labelled* optimistic rather than sold as honest,
quantile regression forests for growth-at-risk style density forecasts, and
grouped block-permutation importance. The forest is the strongest
off-the-shelf nonlinear benchmark for macro forecasting (Medeiros,
Vasconcelos, Veiga & Zilberman 2021), and the honest caveat travels with it:
trees rarely beat regularized linear models for smooth, low-signal problems
on short samples.

| Function | Role |
|----------|------|
| `regression_tree` | CART with squared-error splits, scikit-learn conventions (golden-pinned) |
| `random_forest` | Bagged trees with iid / block / stationary resampling, OOB, quantiles, importance |

## What it estimates

- **`regression_tree(x, y, max_depth=None, min_samples_leaf=1,
  min_samples_split=2, x_test=None)`** — the best-split CART tree: every
  internal node picks the (feature, threshold) pair that most reduces the
  squared error, thresholds are midpoints between adjacent sorted distinct
  values, a split must leave `min_samples_leaf` rows on both sides, and
  leaves predict the training mean. Returns the tree's fit, its size, the
  impurity-based feature importance, and the sorted list of splits.
- **`random_forest(x, y, n_trees=500, max_features="third", max_depth=None,
  min_samples_leaf=5, bootstrap="iid", block_length=None, seed=0,
  x_test=None, quantiles=None, importance="none", importance_groups=None,
  permutation_block=None, n_permutations=None)`** — Breiman's (2001) forest:
  each tree is grown on a row resample (drawn rows act as multiplicity
  weights, rows never drawn are that tree's *out-of-bag* rows), visiting
  `max_features` random columns per node, and the forest averages the trees.
  With `quantiles`, the forest also returns Meinshausen's (2006) conditional
  quantiles: every training row is dropped down every tree, and the
  conditional distribution at a test point is the tree-averaged empirical
  distribution of the targets sharing its leaves. With `importance`, it
  returns either the mean normalized impurity decrease or the grouped
  block-permutation importance (the out-of-bag MSE increase when a unit's
  columns are permuted in contiguous blocks).

## Assumptions

- **Rows are the sampling unit.** The forest models `y_t` from the row
  `x_t` alone; lags are yours to build as columns. Nothing in the forest
  knows the row order except the resampling scheme and the importance
  permutation.
- **`bootstrap="iid"` assumes independent rows.** It is scikit-learn's
  behaviour and the right default for cross-sectional data; on a time
  series it throws away the serial dependence every tree should see.
  `"block"` (Künsch 1989) and `"stationary"` (Politis & Romano 1994) keep
  it: in the property suite a lag-1 autocorrelation of 0.90 in the
  original series becomes -0.01 after iid resampling but 0.84
  (moving block, length 20) and 0.84 (stationary, mean length 20).
- **`min_samples_leaf` bounds the leaf size, not the leaf weight.** As in
  scikit-learn, the count is of *distinct* rows, so a bootstrap row drawn
  three times still counts once.
- **Ties.** When two features induce the same partition of a node the
  lowest feature index wins. scikit-learn breaks the same tie by its
  private RNG's feature-visit order, which is why the golden fixture stores
  only settings proven tie-free (see below) and why unbounded depth with
  `min_samples_leaf=1` — where every two-row node ties on every feature —
  is not golden-reproducible against scikit-learn.

## When to use

- **`regression_tree`** as a diagnostic: the splits are readable, the
  importance is exact, and the tree is the building block you can compare
  against scikit-learn line by line.
- **`random_forest`** as the nonlinear benchmark in a forecasting horse race
  against `ridge`/`lasso`/`elastic_net` on the same lag-augmented design,
  scored with a pseudo-out-of-sample loop (`cv_splits`, `backtest`) — not
  with `oob_mse`.
- **`quantiles=`** when the object of interest is a conditional
  distribution (growth-at-risk, tail forecasts); pair the output with the
  quantile scores and CRPS of the forecast-evaluation module.
- **`importance="block_permutation"` with `importance_groups`** when the
  design holds several lags of the same variables and the question is
  *which variables* matter — the groups are the unit, and the lags of one
  variable travel together.

## Key arguments and defaults

| Call | Argument | Default | Notes |
|------|----------|---------|-------|
| `regression_tree` | `max_depth` | `None` | unbounded |
| | `min_samples_leaf` / `min_samples_split` | `1` / `2` | scikit-learn's defaults |
| `random_forest` | `n_trees` | `500` | about 0.3 s at n = 500, p = 10 with the defaults |
| | `max_features` | `"third"` | `max(1, p // 3)` (Breiman's regression rule); `"sqrt"`, `"all"`, or an int |
| | `min_samples_leaf` | `5` | larger than the tree's default: smoother leaves for noisy targets |
| | `bootstrap` | `"iid"` | `"block"` / `"stationary"` need `block_length`; `"none"` grows every tree on every row (no OOB) |
| | `block_length` | `None` | required for block/stationary, refused otherwise; `optimal_block_length(y)["stationary"]` is a reasonable start |
| | `seed` | `0` | one Philox substream per tree: bit-identical at any thread count |
| | `quantiles` | `None` | strictly inside (0, 1), increasing; needs `x_test` |
| | `importance` | `"none"` | `"impurity"` or `"block_permutation"` |
| | `importance_groups` | `None` | integer label per column; needs an `importance`; pass a list of ints (a label vector, not data) |
| | `permutation_block` | `None` | block_permutation only; `None` = `ceil(n ** (1/3))`, `1` = single-row |
| | `n_permutations` | `None` | block_permutation only; `None` = 10 |

## How to read the output

- **`regression_tree`** → `fitted`, `predicted` (or `None`), `n_nodes`,
  `n_leaves`, `depth` (root = 0, a stump is 1), `feature_importance`
  (normalized to one; zeros if the tree never split), `splits` (a list of
  `[feature, threshold]` sorted by feature then threshold — count how often
  each feature appears to see where the tree spends its splits).
- **`random_forest`** → `fitted` (in-sample: every tree, in-bag rows
  included — it is *not* a test error), `predicted`, `oob_prediction` (NaN
  for a row that was in-bag in every tree; `None` under
  `bootstrap="none"`), `oob_mse`, `importance` plus
  `importance_groups_resolved` (the label each entry refers to — the sorted
  distinct labels, or `0..p-1`), `quantile_predictions` (`(m, len(quantiles))`,
  never crossing), `n_trees`, `max_features_resolved`.
- **Impurity importance** sums to one and is a share of the total impurity
  decrease; **block-permutation importance** is an MSE increase in the
  units of `y**2` and can be negative for an irrelevant unit. They are not
  on the same scale.

## Failure modes

- **Quoting `oob_mse` as the forecast error of a time-series forest.** An
  out-of-bag row's temporal neighbours are in-bag in most of the trees
  that score it; with persistent predictors and autocorrelated errors those
  neighbours carry the row's own error, so the out-of-bag MSE is
  optimistic. Measured in the property suite on the *same* forest
  (persistent logistic-AR(0.9) predictors, fit on 400 rows, scored on the
  next 200): OOB/POOS MSE ratio 0.84 with iid errors and
  0.70 with AR(0.9) errors — the autocorrelation roughly doubles the
  optimism. Report pseudo-out-of-sample metrics; use `oob_mse` to compare
  hyper-parameters, not to state accuracy.
- **Reading an inflated importance for a persistent irrelevant predictor as
  relevance.** When the relevant predictors are persistent, an irrelevant
  persistent series is correlated with them in-sample and the forest uses
  it as a time proxy. Measured (Friedman #1 on five relevant columns plus
  two lags of an irrelevant AR(0.95) series, out-of-bag MSE increase,
  mean of four seeds): the irrelevant unit scores about zero (-0.02)
  when the relevant columns are iid but 0.11 when they are persistent
  (the strongest relevant column scores about 5-9). **Block permutation
  does not remove this.** For a row-wise forest scored row-wise, the mean
  importance depends only on which row each permuted value comes from,
  and single-row and block permutations both pair a row with an
  essentially uniform other row: grouped single-row 0.114 vs
  grouped block(20) 0.108 on the persistent design, the same
  within Monte-Carlo noise. What block permutation buys is a permuted
  design that is still a plausible series (and a different variance of
  the estimate), not a smaller number. What *grouping* buys is real: the
  lags of a variable are permuted together, so the forest cannot route
  around a scrambled lag through its near-collinear neighbour — permuting
  the two lags separately scores 0.093 on the same design,
  i.e. per-lag permutation *dilutes* a variable's importance. The remedy
  for the inflation is a control comparison (the iid-design number above)
  or conditional importance, which this release does not ship.
- **Impurity importance on mixed designs.** It favours columns with many
  distinct values and credits whichever member of a correlated group the
  split happened to pick; prefer the permutation importance when columns
  are correlated.
- **Small `n` with the default `min_samples_leaf=5`.** A forest needs at
  least `2 * min_samples_leaf` rows to make a single split; below that the
  call refuses with `insufficient data: {got} observations, at least
  {needed} required` rather than returning a constant predictor.
- **Reading the quantile band as calibrated.** The quantile forest reads
  its quantiles off the empirical distribution of the training targets in
  the shared leaves, and that distribution mixes targets from neighbouring
  leaves whose conditional means differ — so the band is **conservative**,
  the more so the larger the signal is relative to the noise: the 10-90
  band covers 0.88 of the test targets in the property suite (noise sd
  0.5-2.5 against a Friedman signal) and 0.96 in the runnable example
  below (noise sd 1 against a signal spanning ~25 units, 300 training
  rows). Calibrate it on a pseudo-out-of-sample window before quoting a
  coverage, and treat `0.01`/`0.99` on a few hundred rows as order
  statistics of a handful of points.

## Validated against

- **`regression_tree`: independent package.** scikit-learn 1.9.0
  `DecisionTreeRegressor(criterion="squared_error", splitter="best",
  max_features=None)` on float32-representable Friedman #1 data
  (`n=300`, `p=8`, 120 test rows), eight `(max_depth, min_samples_leaf,
  min_samples_split)` settings. Asserted: training fit and test predictions
  at 1e-12, `n_nodes`/`n_leaves`/`depth` exact, `feature_importances_` at
  1e-10, the sorted split multiset — features exact, thresholds 1e-12.
  Achieved: predictions 1.1e-14, importances 3.3e-14, thresholds
  exact (0). The generator refits every stored case under five other
  `random_state` values and asserts the same tree, which proves no
  RNG-dependent tie-break was exercised; two candidate settings that did
  exercise one are recorded in the fixture's `_meta.excluded_settings`.
  The features are rounded to float32 because scikit-learn grows and
  predicts in float32.
- **`random_forest`, single-tree bridge: exact.** `random_forest(
  bootstrap="none", max_features="all", n_trees=1)` reproduces
  `regression_tree` bit-for-bit on every fixture setting (and at the
  tie-heavy unbounded/`min_samples_leaf=1` setting), so the forest's tree
  grower is the golden-pinned one.
- **`random_forest`, full forest: property / Monte-Carlo grade** (its
  randomness is tsecon's own Philox stream, so no third-party golden can
  exist). Measured and asserted with margins in
  [`trees_properties.rs`](../../../crates/tsecon-ml/tests/trees_properties.rs):
  same seed bit-identical, different seed differs, 1/3/8 rayon threads and
  the global pool identical; Friedman #1 out-of-sample R² 0.79 (bar
  0.70; scikit-learn's forest at the same settings measures ~0.77) with
  the iid out-of-bag R² within 0.05 of it; the resampling autocorrelation
  numbers, the out-of-bag optimism ratios, and the importance numbers
  quoted above; quantile forest on iid heteroskedastic data: q10-q90
  band coverage 0.88 of 400 test targets for a nominal 0.80 — the forest
  is **conservative**, as Meinshausen reports for small leaves, because
  the leaf-weighted distribution mixes targets from neighbouring leaves
  with different conditional means (asserted in [0.75, 0.95]; binomial
  sd 0.02) — q05-q95 coverage 0.943, quantiles never cross (exact by
  construction), median-vs-mean correlation 0.995;
  impurity and permutation importance both rank the five relevant
  Friedman columns first.

Fixture: [`fixtures/trees.json`](../../../fixtures/trees.json) (generator
[`generate_trees_fixtures.py`](../../../fixtures/generate_trees_fixtures.py)).

## References

- Breiman, L., Friedman, J., Olshen, R. & Stone, C. (1984). *Classification
  and Regression Trees*. Wadsworth.
- Breiman, L. (2001). "Random Forests." *Machine Learning* 45.
- Künsch, H. (1989). "The Jackknife and the Bootstrap for General Stationary
  Observations." *Annals of Statistics* 17.
- Politis, D. & Romano, J. (1994). "The Stationary Bootstrap." *JASA* 89.
- Meinshausen, N. (2006). "Quantile Regression Forests." *JMLR* 7.
- Strobl, C., Boulesteix, A.-L., Kneib, T., Augustin, T. & Zeileis, A.
  (2008). "Conditional variable importance for random forests." *BMC
  Bioinformatics* 9.
- Medeiros, M., Vasconcelos, G., Veiga, Á. & Zilberman, E. (2021).
  "Forecasting Inflation in a Data-Rich Environment: The Benefits of Machine
  Learning Methods." *JBES* 39.

See the guide: [Machine Learning for Time Series](../../guide/12-machine-learning.md).

## Runnable example

```python
import numpy as np
import tsecon

rng = np.random.default_rng(3)
n, p = 400, 8
X = rng.uniform(size=(n, p))
y = (10 * np.sin(np.pi * X[:, 0] * X[:, 1]) + 20 * (X[:, 2] - 0.5) ** 2
     + 10 * X[:, 3] + 5 * X[:, 4] + rng.standard_normal(n))
Xtr, ytr, Xte, yte = X[:300], y[:300], X[300:], y[300:]

# 1. A shallow tree: readable splits, exact impurity importance.
tree = tsecon.regression_tree(Xtr, ytr, max_depth=3, x_test=Xte)
print("tree leaves:", tree["n_leaves"], " splits on feature 3:",
      sum(f == 3 for f, _ in tree["splits"]))

# 2. The forest, scored honestly out of sample AND by its (optimistic) OOB.
rf = tsecon.random_forest(Xtr, ytr, n_trees=300, x_test=Xte,
                          quantiles=[0.1, 0.5, 0.9], importance="impurity")
oos_r2 = 1 - np.mean((yte - rf["predicted"]) ** 2) / np.var(yte)
print("out-of-sample R2: %.2f   oob mse: %.2f" % (oos_r2, rf["oob_mse"]))
print("top-3 impurity importance:", np.argsort(rf["importance"])[-3:][::-1])

# 3. Quantile forest: coverage of the 10-90 band on the test rows.
q = rf["quantile_predictions"]
print("q10-q90 coverage: %.2f" % np.mean((q[:, 0] <= yte) & (yte <= q[:, 2])))

# 4. Block resampling for a time series (rows in time order).
rf_block = tsecon.random_forest(Xtr, ytr, n_trees=100, bootstrap="stationary",
                                block_length=10, x_test=Xte)
print("stationary-bootstrap forest predicts", rf_block["predicted"].shape)
```

Expected output:

```
tree leaves: 8  splits on feature 3: 3
out-of-sample R2: 0.72   oob mse: 9.00
top-3 impurity importance: [3 1 0]
q10-q90 coverage: 0.96
stationary-bootstrap forest predicts (100,)
```
