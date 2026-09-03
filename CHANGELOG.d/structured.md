### Added

- **`group_lasso` — group LASSO (Yuan & Lin 2006) and sparse-group LASSO
  (Simon, Friedman, Hastie & Tibshirani 2013) with a readable optimality
  certificate.** Block coordinate descent on
  `(1/(2n))||y - Xb||^2 + alpha*[(1 - l1_ratio)*sum_g w_g||b_g||_2 +
  l1_ratio*||b||_1]` — the crate's `lasso` scaling, so `l1_ratio=1` *is*
  `lasso` and `alpha` stays on scikit-learn's scale. Each block is solved
  by proximal gradient with the exact per-block Lipschitz constant
  `lambda_max(X_g'X_g)/n` and Simon et al.'s group-zero test, the trap the
  roadmap names (a wrong constant or prox order "converges smoothly to the
  wrong answer"), and `converged` is gated on the subgradient KKT residual
  the fit returns as `kkt_violation` — a self-certificate that is rigorous
  for this convex problem. `groups` takes any integer labels, contiguous
  or not (integer arrays pass through coercion untouched);
  `group_weights` is `"sqrt_size"` (Yuan-Lin), `"none"`, or a per-group
  array. Returns `coef`, `n_iter`, `converged`, `active_groups`,
  `active_set`, `objective`, `kkt_violation`, `max_rel_change`,
  `alpha_max`. Validation: an **independent** KKT evaluation on every
  fixture case asserted ≤ 1e-8 and achieved 2.3e-13 (primary grade); cross-
  package agreement with **skglm 0.5** (`GroupLasso`, `WeightedL1GroupL2`
  + `GroupBCD`, same 1/(2n) objective, ten cases over two designs and three
  weight conventions) asserted 1e-8 and achieved 1.5e-12, bounded by
  skglm's own recorded KKT residual (5.7e-13); reductions to `lasso`
  (`l1_ratio=1`; singleton groups; weighted singletons ≡ rescaled `lasso`)
  at 1e-8, achieved ~1e-13; `alpha_max` at 1e-12; scale equivariance over
  sixteen decades at both shipped tolerances; the `converged=False` flag
  proven to fire with an honestly larger residual.
- **`post_lasso` — post-LASSO OLS (Belloni & Chernozhukov 2013), with no
  standard errors by design.** LASSO / elastic net on `elastic_net`'s
  objective, then the minimum-norm OLS refit on the selected columns.
  Returns `support`, `coef_lasso`, `coef_ols`, `n_selected`, `rss`. The
  docstring and card say why nothing resembling a standard error is
  returned (the selection event depends on the same sample; naive OLS
  standard errors after selection are invalid) and route inference to
  `pds_lasso`. Refit pinned to scikit-learn
  `LinearRegression(fit_intercept=False)` on the scikit-learn support at
  1e-10, achieved 8.4e-15; support exact.
- **`pds_lasso` — post-double-selection (Belloni, Chernozhukov & Hansen
  2014) for a treatment coefficient among high-dimensional controls, with
  Newey-West HAC inference from the shared HAC engine.** LASSO of `y` on
  `x` and of `d` on `x` (`alpha="bic"` = the per-equation BIC pick along
  `lasso_path`'s grid, or one float), union of supports, OLS of `y` on
  `[d, x_union]` with Bartlett HAC (`hac_lags=None` → the Newey-West rule
  `floor(4 (n/100)^(2/9))`; `0` → classical standard errors); the HAC
  covariance carries `n/(n-k)` and `p_value`/`conf_int` use the normal
  (statsmodels `use_correction=True`, `use_t=False`). Returns `coef`,
  `se`, `t_stat`, `p_value`, `conf_int`, `support_y`, `support_d`,
  `union_support`, `n_controls_selected`, `alpha_y`, `alpha_d`,
  `hac_lags_resolved`. Exact leg pinned to statsmodels HAC / nonrobust OLS
  on the forced-full and BIC-selected unions at 1e-8 relative, achieved
  9.8e-15 (`p_value` 1e-12). **Coverage is Monte-Carlo grade** — R `hdm`
  and Stata `pdslasso` are not runnable in the reference environment — and
  the failure it exists to fix is measured rather than asserted: on the
  seeded design (n = 400, p = 40 AR(1) controls, AR(1) errors ρ = 0.3, four
  confounders loading γ = ±1 on `d` and β = ±0.15 on `y`, 300 replications)
  the single-selection interval covers **0.003** at a nominal 0.95
  while PDS covers **0.950**, within Monte-Carlo noise of the
  infeasible oracle (**0.953**). A second, more persistent cell
  (n = 200, ρ = 0.5) shows the HAC engine's own small-sample shortfall —
  oracle **0.930**, PDS **0.903**, single **0.153** — so
  the card can separate what selection costs from what Newey-West costs.
  (The two 300-replication cells are an `#[ignore]`d release-mode test;
  the always-on cell — n = 200, p = 16, 80 replications — asserts the
  same ordering on every `cargo test`: PDS 0.938, oracle 0.963, single
  selection 0.075.)
- New teaching errors on the slice: NaN/inf refused naming the array
  (`x`, `y`, `d`); `groups` length mismatches name the expected and
  received sizes; `l1_ratio` outside `[0, 1]`; unknown `group_weights` /
  `alpha` strings list the accepted values; custom weights of the wrong
  length or not positive; negative `hac_lags` names the three valid
  choices; the house insufficiency wording (`insufficient data: {got}
  observations, at least {needed} required`) when a refit has no residual
  degrees of freedom; a singular `[d, X_union]` design surfaces the HAC
  engine's error rather than a panic. `MlError` gains `InsufficientData`
  and `Hac(HacError)`; `tsecon-ml` now depends on `tsecon-hac`.
- `_coerce._EXEMPT` gains `group_lasso: {groups}`, and the integer-
  parameter audit in `test_coerce.py` now scans every `src/*.rs` binding
  file rather than `lib.rs` alone.
