# Adversarial-input matrix — summary

171 callables, 4596 cells, 60 s wall (this run).

| outcome | cells |
|---|---:|
| refusal | 3572 |
| ok | 958 |
| ALLOC-ABORT | 63 |
| HANG | 2 |
| CRASH | 1 |

## Every cell that was not a refusal, a success, or a skip

| callable | cell | outcome | detail |
|---|---|---|---|
| `arima_fit` | `kw:forecast_steps:int=2^31` | ALLOC-ABORT | rc=-6 bytes=17179869184 |
| `auto_arima` | `kw:forecast_steps:int=2^31` | ALLOC-ABORT | rc=-6 bytes=17179869184 |
| `boosting` | `kw:n_steps:int=2^31` | ALLOC-ABORT | rc=-6 bytes=51539607552 |
| `bootstrap_indices` | `arg0:int=2^31` | ALLOC-ABORT | rc=-6 bytes=17179869184 |
| `bvar_hierarchical` | `kw:n_grid:int=2^31` | ALLOC-ABORT | rc=-6 bytes=17179869184 |
| `bvar_irf_draws` | `kw:horizon:int=2^31` | ALLOC-ABORT | rc=-6 bytes=120259084344 |
| `bvar_irf_draws` | `kw:n_draws:int=2^31` | ALLOC-ABORT | rc=-6 bytes=51539607552 |
| `bvar_ssvs` | `kw:horizon:int=2^31` | ALLOC-ABORT | rc=-6 bytes=120259084344 |
| `bvar_ssvs` | `kw:n_chains:int=2^31` | ALLOC-ABORT | rc=-6 bytes=154618822656 |
| `bvar_ssvs` | `kw:n_draws:int=2^31` | ALLOC-ABORT | rc=-6 bytes=17179868784 |
| `ccc_garch` | `kw:forecast_horizon:int=2^31` | ALLOC-ABORT | rc=-6 bytes=17179869184 |
| `connectedness` | `kw:horizon:int=2^31` | ALLOC-ABORT | rc=-6 bytes=120259084344 |
| `cv_splits` | `arg0:int=2^31` | ALLOC-ABORT | rc=-6 bytes=176 |
| `dcc_garch` | `kw:forecast_horizon:int=2^31` | ALLOC-ABORT | rc=-6 bytes=17179869184 |
| `favar` | `kw:horizon:int=2^31` | CRASH | rc=-6 |
| `flp` | `kw:horizons:int=2^31` | ALLOC-ABORT | rc=-6 bytes=17179869192 |
| `flp_scenario` | `kw:horizons:int=2^31` | ALLOC-ABORT | rc=-6 bytes=17179869192 |
| `fry_pagan_svar` | `kw:horizon:int=2^31` | ALLOC-ABORT | rc=-6 bytes=120259084344 |
| `fry_pagan_svar` | `kw:n_draws:int=2^31` | ALLOC-ABORT | rc=-6 bytes=154618822656 |
| `fvar_scenario` | `kw:horizon:int=2^31` | ALLOC-ABORT | rc=-6 bytes=120259084344 |
| `garch_fit` | `kw:forecast_horizon:int=2^31` | ALLOC-ABORT | rc=-6 bytes=17179869184 |
| `gas_volatility` | `kw:horizon:int=2^31` | ALLOC-ABORT | rc=-6 bytes=17179869184 |
| `hansen_seo_test` | `kw:n_boot:int=2^31` | ALLOC-ABORT | rc=-6 bytes=154618822656 |
| `hetero_svar` | `kw:horizon:int=2^31` | ALLOC-ABORT | rc=-6 bytes=120259084344 |
| `historical_decomposition` | `kw:horizon:int=2^31` | ALLOC-ABORT | rc=-6 bytes=120259084344 |
| `kernel_ridge` | `kw:rff_features:int=2^31` | ALLOC-ABORT | rc=-6 bytes=51539607552 |
| `lasso_path` | `kw:n_lambdas:int=2^31` | ALLOC-ABORT | rc=-6 bytes=17179869184 |
| `long_run_svar` | `kw:horizon:int=2^31` | ALLOC-ABORT | rc=-6 bytes=120259084344 |
| `lp` | `kw:horizons:int=2^31` | ALLOC-ABORT | rc=-6 bytes=17179869192 |
| `lp_iv` | `kw:horizons:int=2^31` | ALLOC-ABORT | rc=-6 bytes=17179869192 |
| `lp_multiplier` | `kw:horizons:int=2^31` | ALLOC-ABORT | rc=-6 bytes=17179869192 |
| `lp_state` | `kw:horizons:int=2^31` | ALLOC-ABORT | rc=-6 bytes=17179869192 |
| `max_share_svar` | `kw:horizon:int=2^31` | ALLOC-ABORT | rc=-6 bytes=120259084344 |
| `mean_group_var` | `kw:horizon:int=2^31` | ALLOC-ABORT | rc=-6 bytes=120259084344 |
| `midas_weights` | `arg3:int=2^31` | ALLOC-ABORT | rc=-6 bytes=17179869184 |
| `mlp_regression` | `kw:hidden:ilist_2^31` | ALLOC-ABORT | rc=-6 bytes=68719476744 |
| `mlp_regression` | `kw:max_epochs:int=2^31` | ALLOC-ABORT | rc=-6 bytes=17179869184 |
| `mlp_regression` | `kw:n_seeds:int=2^31` | ALLOC-ABORT | rc=-6 bytes=154618822656 |
| `mstl` | `kw:iterate:int=2^31` | HANG | deadline 15.0s |
| `narrative_svar` | `kw:horizon:int=2^31` | ALLOC-ABORT | rc=-6 bytes=120259084344 |
| `narrative_svar` | `kw:n_draws:int=2^31` | ALLOC-ABORT | rc=-6 bytes=154618822656 |
| `nongaussian_svar` | `kw:horizon:int=2^31` | ALLOC-ABORT | rc=-6 bytes=120259084344 |
| `philox_uniforms` | `arg1:int=2^31` | ALLOC-ABORT | rc=-6 bytes=17179869184 |
| `proxy_ar_sets` | `kw:horizon:int=2^31` | ALLOC-ABORT | rc=-6 bytes=120259084344 |
| `proxy_svar` | `kw:horizon:int=2^31` | ALLOC-ABORT | rc=-6 bytes=120259084344 |
| `proxy_svar_bands` | `kw:horizon:int=2^31` | ALLOC-ABORT | rc=-6 bytes=120259084344 |
| `proxy_svar_bands` | `kw:n_boot:int=2^31` | ALLOC-ABORT | rc=-6 bytes=154618822656 |
| `quantile_lp` | `kw:horizons:int=2^31` | ALLOC-ABORT | rc=-6 bytes=17179869192 |
| `random_forest` | `kw:n_trees:int=2^31` | ALLOC-ABORT | rc=-6 bytes=154618822728 |
| `reset_test` | `kw:max_power:int=2^31` | HANG | deadline 15.0s |
| `robust_svar_bounds` | `kw:horizon:int=2^31` | ALLOC-ABORT | rc=-6 bytes=463856468184 |
| `robust_svar_bounds` | `kw:n_draws:int=2^31` | ALLOC-ABORT | rc=-6 bytes=154618822656 |
| `setar_test` | `kw:n_boot:int=2^31` | ALLOC-ABORT | rc=-6 bytes=154618822656 |
| `sign_restricted_svar` | `kw:horizon:int=2^31` | ALLOC-ABORT | rc=-6 bytes=120259084344 |
| `sign_restricted_svar` | `kw:n_draws:int=2^31` | ALLOC-ABORT | rc=-6 bytes=154618822656 |
| `star` | `kw:n_c:int=2^31` | ALLOC-ABORT | rc=-6 bytes=17179869184 |
| `star` | `kw:n_gamma:int=2^31` | ALLOC-ABORT | rc=-6 bytes=17179869184 |
| `structural_fevd` | `kw:horizon:int=2^31` | ALLOC-ABORT | rc=-6 bytes=120259084344 |
| `theta_forecast` | `arg1:int=2^31` | ALLOC-ABORT | rc=-6 bytes=17179869184 |
| `threshold_var_test` | `kw:n_boot:int=2^31` | ALLOC-ABORT | rc=-6 bytes=154618822656 |
| `threshold_vecm` | `kw:n_grid_beta:int=2^31` | ALLOC-ABORT | rc=-6 bytes=17179869184 |
| `var_fevd` | `kw:horizon:int=2^31` | ALLOC-ABORT | rc=-6 bytes=120259084288 |
| `var_irf` | `kw:horizon:int=2^31` | ALLOC-ABORT | rc=-6 bytes=120259084344 |
| `var_irf_bands` | `kw:horizon:int=2^31` | ALLOC-ABORT | rc=-6 bytes=120259084344 |
| `zero_sign_svar` | `kw:horizon:int=2^31` | ALLOC-ABORT | rc=-6 bytes=120259084344 |
| `zero_sign_svar` | `kw:n_draws:int=2^31` | ALLOC-ABORT | rc=-6 bytes=154618822656 |

## Largest RSS deltas (top 25 cells)

| callable | cell | outcome | RSS delta (MB) | seconds |
|---|---|---|---:|---:|
| `auto_arima` | `kw:max_p:int=0` | ok | 1.1 | 0.581 |
| `auto_arima` | `kw:seasonal_period:int=2` | ok | 0.4 | 6.948 |
| `auto_arima` | `arg0:nan_all` | refusal | 0.3 | 0.001 |
| `auto_arima` | `arg0:empty` | refusal | 0.2 | 0.0 |
| `auto_arima` | `arg0:one` | refusal | 0.1 | 0.0 |
| `auto_arima` | `kw:max_p:int=1` | ok | 0.1 | 0.919 |
| `acf` | `arg0:nan_all` | refusal | 0.0 | 0.0 |
| `acf` | `arg0:nan_one` | refusal | 0.0 | 0.0 |
| `acf` | `arg0:inf_one` | refusal | 0.0 | 0.0 |
| `acf` | `arg0:empty` | refusal | 0.0 | 0.0 |
| `acf` | `arg0:one` | refusal | 0.0 | 0.0 |
| `acf` | `arg0:str` | refusal | 0.0 | 0.0 |
| `acf` | `kw:nlags:int=0` | refusal | 0.0 | 0.0 |
| `acf` | `kw:nlags:int=1` | ok | 0.0 | 0.0 |
| `acf` | `kw:nlags:int=2` | ok | 0.0 | 0.0 |
| `acf` | `kw:nlags:int=neg1` | refusal | 0.0 | 0.0 |
| `acf` | `kw:nlags:int=2^31` | refusal | 0.0 | 0.0 |
| `acf` | `kw:nlags:int=2^63` | refusal | 0.0 | 0.0 |
| `acf` | `kw:nlags:int=2^64` | refusal | 0.0 | 0.0 |
| `accuracy` | `arg0:nan_all` | refusal | 0.0 | 0.0 |
| `accuracy` | `arg0:nan_one` | refusal | 0.0 | 0.0 |
| `accuracy` | `arg0:inf_one` | refusal | 0.0 | 0.0 |
| `accuracy` | `arg0:empty` | refusal | 0.0 | 0.0 |
| `accuracy` | `arg0:one` | refusal | 0.0 | 0.0 |
| `accuracy` | `arg0:str` | refusal | 0.0 | 0.0 |

## Slowest cells (top 25, completed)

| callable | cell | outcome | seconds |
|---|---|---|---:|
| `auto_arima` | `kw:seasonal_period:int=2` | ok | 6.948 |
| `auto_arima` | `kw:max_D:int=1` | ok | 2.099 |
| `auto_arima` | `kw:max_D:int=2` | ok | 2.071 |
| `auto_arima` | `kw:max_D:int=2^31` | ok | 2.07 |
| `auto_arima` | `kw:forecast_steps:int=2` | ok | 2.043 |
| `auto_arima` | `kw:max_d:int=2` | ok | 2.04 |
| `auto_arima` | `kw:max_D:int=0` | ok | 2.037 |
| `auto_arima` | `kw:forecast_steps:int=0` | ok | 2.037 |
| `auto_arima` | `kw:forecast_steps:int=1` | ok | 2.036 |
| `auto_arima` | `kw:max_d:int=2^31` | ok | 2.034 |
| `auto_arima` | `kw:max_d:int=1` | ok | 2.03 |
| `auto_arima` | `kw:max_d:int=0` | ok | 1.957 |
| `auto_arima` | `kw:max_q:int=2` | ok | 1.952 |
| `auto_arima` | `kw:seasonal_period:int=0` | ok | 1.93 |
| `auto_arima` | `kw:max_Q:int=2` | ok | 1.918 |
| `auto_arima` | `kw:max_order:int=1` | ok | 1.917 |
| `auto_arima` | `kw:max_P:int=2` | ok | 1.915 |
| `auto_arima` | `kw:max_order:int=2` | ok | 1.914 |
| `auto_arima` | `kw:max_order:int=2^31` | ok | 1.914 |
| `auto_arima` | `kw:max_order:int=0` | ok | 1.912 |
| `auto_arima` | `kw:max_P:int=0` | ok | 1.907 |
| `auto_arima` | `kw:max_p:int=2` | ok | 1.903 |
| `auto_arima` | `kw:max_P:int=1` | ok | 1.899 |
| `auto_arima` | `kw:max_Q:int=0` | ok | 1.897 |
| `auto_arima` | `kw:max_Q:int=1` | ok | 1.897 |
