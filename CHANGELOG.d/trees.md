### Added

- **Regression trees and random forests, native Rust (`regression_tree`,
  `random_forest`; roadmap Module 10, Tier 2 "Trees, forests, and
  boosting", "Interpretation", and Tier 3 "Quantile regression forests").**
  `regression_tree` is CART with scikit-learn's best-split conventions —
  midpoint thresholds, the 1e-7 tie window, `min_samples_leaf` on both
  sides, leaves at the training mean, impurity-based importance — and is
  **golden-pinned to scikit-learn 1.9.0 `DecisionTreeRegressor`** on eight
  `(max_depth, min_samples_leaf, min_samples_split)` settings: predictions
  at 1e-12 (achieved 1.1e-14), `n_nodes`/`n_leaves`/`depth` exact,
  `feature_importances_` at 1e-10 (achieved 3.3e-14), the sorted split
  multiset with thresholds at 1e-12 (achieved 0). Exact matching is
  possible because the fixture generator proves every stored case tie-free
  by refitting under five sklearn `random_state` values (sklearn breaks an
  exact tie by its private RNG's feature-visit order); the two candidate
  settings that exercised a tie-break are recorded in the fixture rather
  than silently dropped. `random_forest` grows those trees on iid,
  moving-block (Künsch) or stationary (Politis-Romano) row resamples —
  `block_length` required for the block schemes and refused for
  `"iid"`/`"none"` — with `max_features` in {"sqrt", "third", "all", int},
  one Philox substream per tree so the fit is bit-identical at any rayon
  thread count (asserted at 1/3/8 threads), out-of-bag predictions and
  MSE, Meinshausen (2006) quantile regression forests (`quantiles=`), and
  impurity or grouped block-permutation importance (`importance_groups`
  names the unit each column belongs to; `permutation_block` shuffles
  contiguous row blocks). `random_forest(bootstrap="none",
  max_features="all", n_trees=1)` reproduces `regression_tree`
  bit-for-bit, which is how the forest inherits the golden; the full
  forest is **property / Monte-Carlo grade**, measured and asserted with
  margins: Friedman #1 out-of-sample R² 0.79 (bar 0.70); block and
  stationary resampling keep an AR(0.9) series' lag-1 autocorrelation at
  0.84 where iid resampling leaves −0.01; **out-of-bag error is optimistic
  on time series** — on the same forest with persistent predictors the
  OOB/POOS MSE ratio is 0.84 under iid errors and 0.70 under AR(0.9)
  errors (the trap the roadmap names; the card says to report
  pseudo-out-of-sample metrics); the q10-q90 quantile band covers 0.88 of
  iid test targets for a nominal 0.80 (conservative, as Meinshausen
  reports for small leaves) and quantiles never cross; importance
  recovers the five relevant Friedman columns under both schemes. The
  default call (`n_trees=500`, n=500, p=10) takes about 0.3 s.
  **Honest downgrade, stated on the card:** the roadmap's claim that block
  permutation gives a persistent irrelevant predictor a *smaller*
  importance than single-row permutation does not hold for a row-wise
  forest scored with a row-wise loss — the mean importance depends only
  on which row each permuted value comes from, and both permutations
  pair a row with an essentially uniform other row. Measured: an
  irrelevant AR(0.95) unit scores about zero (-0.02) when the relevant
  predictors are iid but 0.11 when they are persistent (the forest uses it as a time
  proxy), and grouped single-row (0.114) vs grouped block (0.108)
  permutation agree within noise, so the test asserts the inflation and
  the agreement, not the ordering; what grouping *does* fix — per-lag
  permutation diluting a variable's importance across collinear lags
  (0.093 vs 0.114) — is measured too. Teaching errors: NaN/inf refused
  naming the array; `insufficient data: {got} observations, at least
  {needed} required`; unknown string options list the accepted values;
  `quantiles` outside (0, 1) or unsorted name the fix;
  `importance_groups` of the wrong length names both lengths; the
  inert-kwarg sentinels (`block_length`, `importance_groups`,
  `permutation_block`, `n_permutations`) refuse where they would do
  nothing. Rust: `tsecon_ml::{regression_tree, random_forest,
  resample_indices, TreeOptions, ForestOptions, MaxFeatures, Resampling,
  Importance}`; new `MlError::InsufficientData` / `InvalidBlockLength`.
  Fixture `fixtures/trees.json`; card
  `docs/reference/model-cards/ml-trees.md`.
