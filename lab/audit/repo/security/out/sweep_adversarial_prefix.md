# Adversarial-input matrix — summary

173 callables, 4769 cells, 264 s wall (this run).

| outcome | cells |
|---|---:|
| refusal | 3400 |
| ok | 1175 |
| PANIC | 113 |
| ALLOC-ABORT | 66 |
| HANG | 14 |
| CRASH | 1 |

## Every cell that was not a refusal, a success, or a skip

| callable | cell | outcome | detail |
|---|---|---|---|
| `arch_lm` | `kw:nlags:int=2^63` | PANIC | capacity overflow |
| `arima_fit` | `big:T=1e5` | HANG | deadline 45.0s |
| `arima_fit` | `kw:forecast_steps:int=2^31` | ALLOC-ABORT | rc=-6 bytes=17179869184 |
| `arima_fit` | `kw:forecast_steps:int=2^63` | PANIC | capacity overflow |
| `auto_arima` | `big:T=1e5` | HANG | deadline 45.0s |
| `auto_arima` | `kw:forecast_steps:int=2^31` | ALLOC-ABORT | rc=-6 bytes=17179869184 |
| `auto_arima` | `kw:forecast_steps:int=2^63` | PANIC | capacity overflow |
| `auto_arima` | `kw:seasonal_period:int=2^63` | PANIC | index out of bounds: the len is 200 but the index is 9223372036854775808 |
| `bai_perron` | `big:T=1e5` | ALLOC-ABORT | rc=-6 bytes=601272 |
| `bk_filter` | `kw:k:int=2^63` | PANIC | index out of bounds: the len is 1 but the index is 9223372036854775808 |
| `bn_decomposition` | `big:T=1e5` | HANG | deadline 45.0s |
| `bn_filter` | `kw:p:int=2^63` | PANIC | capacity overflow |
| `boosting` | `kw:n_steps:int=2^31` | ALLOC-ABORT | rc=-6 bytes=51539607552 |
| `boosting` | `kw:n_steps:int=2^63` | PANIC | capacity overflow |
| `bootstrap_indices` | `arg0:int=2^31` | ALLOC-ABORT | rc=-6 bytes=17179869184 |
| `bootstrap_indices` | `arg0:int=2^63` | PANIC | capacity overflow |
| `bvar_fit` | `kw:lags:int=2^31` | PANIC | called `Result::unwrap()` on an `Err` value: AllocError { layout: Layout { size: 154618822848, align: 64 (1 << 6) } } |
| `bvar_fit` | `kw:lags:int=2^63` | PANIC | called `Result::unwrap()` on an `Err` value: CapacityOverflow |
| `bvar_fit` | `kw:scale_ar:int=2^63` | PANIC | called `Result::unwrap()` on an `Err` value: CapacityOverflow |
| `bvar_hierarchical` | `kw:lags:int=2^31` | PANIC | called `Result::unwrap()` on an `Err` value: AllocError { layout: Layout { size: 154618822848, align: 64 (1 << 6) } } |
| `bvar_hierarchical` | `kw:lags:int=2^63` | PANIC | called `Result::unwrap()` on an `Err` value: CapacityOverflow |
| `bvar_hierarchical` | `kw:n_grid:int=2^31` | ALLOC-ABORT | rc=-6 bytes=17179869184 |
| `bvar_hierarchical` | `kw:n_grid:int=2^63` | PANIC | capacity overflow |
| `bvar_hierarchical` | `kw:scale_ar:int=2^63` | PANIC | called `Result::unwrap()` on an `Err` value: CapacityOverflow |
| `bvar_irf_draws` | `kw:horizon:int=2^31` | ALLOC-ABORT | rc=-6 bytes=120259084344 |
| `bvar_irf_draws` | `kw:horizon:int=2^63` | PANIC | capacity overflow |
| `bvar_irf_draws` | `kw:lags:int=2^31` | PANIC | called `Result::unwrap()` on an `Err` value: AllocError { layout: Layout { size: 154618822848, align: 64 (1 << 6) } } |
| `bvar_irf_draws` | `kw:lags:int=2^63` | PANIC | called `Result::unwrap()` on an `Err` value: CapacityOverflow |
| `bvar_irf_draws` | `kw:n_draws:int=2^31` | ALLOC-ABORT | rc=-6 bytes=51539607552 |
| `bvar_irf_draws` | `kw:n_draws:int=2^63` | PANIC | capacity overflow |
| `bvar_irf_draws` | `kw:scale_ar:int=2^63` | PANIC | called `Result::unwrap()` on an `Err` value: CapacityOverflow |
| `bvar_ssvs` | `kw:horizon:int=2^31` | ALLOC-ABORT | rc=-6 bytes=120259084344 |
| `bvar_ssvs` | `kw:horizon:int=2^63` | PANIC | capacity overflow |
| `bvar_ssvs` | `kw:lags:int=2^63` | PANIC | called `Result::unwrap()` on an `Err` value: CapacityOverflow |
| `bvar_ssvs` | `kw:n_chains:int=2^31` | ALLOC-ABORT | rc=-6 bytes=154618822656 |
| `bvar_ssvs` | `kw:n_draws:int=2^31` | ALLOC-ABORT | rc=-6 bytes=17179868784 |
| `bvar_ssvs` | `kw:n_draws:int=2^63` | PANIC | capacity overflow |
| `ccc_garch` | `kw:forecast_horizon:int=2^31` | ALLOC-ABORT | rc=-6 bytes=17179869184 |
| `ccc_garch` | `kw:forecast_horizon:int=2^63` | PANIC | capacity overflow |
| `ccc_garch` | `kw:p:int=2^63` | PANIC | capacity overflow |
| `ccc_garch` | `kw:q:int=2^63` | PANIC | capacity overflow |
| `conformal_backtest` | `big:T=1e5` | HANG | deadline 45.0s |
| `conformal_backtest` | `kw:period:int=2^63` | PANIC | capacity overflow |
| `conformal_forecast` | `big:T=1e5` | HANG | deadline 45.0s |
| `conformal_forecast` | `kw:period:int=2^63` | PANIC | capacity overflow |
| `connectedness` | `kw:horizon:int=2^31` | ALLOC-ABORT | rc=-6 bytes=120259084344 |
| `connectedness` | `kw:horizon:int=2^63` | PANIC | capacity overflow |
| `connectedness` | `kw:lags:int=2^63` | PANIC | Assertion failed at /root/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/faer-0.24.4/src/mat/matref.rs:819:3
Assertion failed: row_start <= self.nrows()
- |
| `copula_select` | `big:T=1e5` | HANG | deadline 45.0s |
| `cv_splits` | `arg0:int=2^31` | ALLOC-ABORT | rc=-6 bytes=1792 |
| `cv_splits` | `arg0:int=2^63` | ALLOC-ABORT | rc=-6 bytes=176 |
| `cv_splits` | `big:T=1e5` | ALLOC-ABORT | rc=-6 bytes=510512 |
| `dcc_garch` | `kw:forecast_horizon:int=2^31` | ALLOC-ABORT | rc=-6 bytes=17179869184 |
| `dcc_garch` | `kw:forecast_horizon:int=2^63` | PANIC | capacity overflow |
| `dcc_garch` | `kw:p:int=2^63` | PANIC | capacity overflow |
| `dcc_garch` | `kw:q:int=2^63` | PANIC | capacity overflow |
| `dcc_test` | `kw:p:int=2^63` | PANIC | capacity overflow |
| `dcc_test` | `kw:q:int=2^63` | PANIC | capacity overflow |
| `dfm_news` | `kw:factor_order:int=2^63` | PANIC | Assertion failed at /root/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/faer-0.24.4/src/mat/matref.rs:819:3
Assertion failed: row_start <= self.nrows()
- |
| `dfm_nowcast` | `kw:factor_order:int=2^63` | PANIC | Assertion failed at /root/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/faer-0.24.4/src/mat/matref.rs:819:3
Assertion failed: row_start <= self.nrows()
- |
| `echo_state_network` | `kw:reservoir_size:int=2^31` | PANIC | called `Result::unwrap()` on an `Err` value: CapacityOverflow |
| `echo_state_network` | `kw:reservoir_size:int=2^63` | PANIC | called `Result::unwrap()` on an `Err` value: CapacityOverflow |
| `favar` | `kw:horizon:int=2^31` | CRASH | rc=-6 |
| `favar` | `kw:horizon:int=2^63` | PANIC | capacity overflow |
| `favar` | `kw:lags:int=2^63` | PANIC | Assertion failed at /root/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/faer-0.24.4/src/mat/matref.rs:819:3
Assertion failed: row_start <= self.nrows()
- |
| `flp` | `kw:horizons:int=2^31` | ALLOC-ABORT | rc=-6 bytes=17179869192 |
| `flp` | `kw:horizons:int=2^63` | PANIC | capacity overflow |
| `flp_scenario` | `kw:horizons:int=2^31` | ALLOC-ABORT | rc=-6 bytes=17179869192 |
| `flp_scenario` | `kw:horizons:int=2^63` | PANIC | capacity overflow |
| `fry_pagan_svar` | `kw:horizon:int=2^31` | ALLOC-ABORT | rc=-6 bytes=120259084344 |
| `fry_pagan_svar` | `kw:horizon:int=2^63` | PANIC | capacity overflow |
| `fry_pagan_svar` | `kw:lags:int=2^31` | PANIC | called `Result::unwrap()` on an `Err` value: AllocError { layout: Layout { size: 154618822848, align: 64 (1 << 6) } } |
| `fry_pagan_svar` | `kw:lags:int=2^63` | PANIC | called `Result::unwrap()` on an `Err` value: CapacityOverflow |
| `fry_pagan_svar` | `kw:n_draws:int=2^31` | ALLOC-ABORT | rc=-6 bytes=154618822656 |
| `fvar_scenario` | `kw:horizon:int=2^31` | ALLOC-ABORT | rc=-6 bytes=120259084344 |
| `fvar_scenario` | `kw:horizon:int=2^63` | PANIC | capacity overflow |
| `fvar_scenario` | `kw:lags:int=2^63` | PANIC | Assertion failed at /root/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/faer-0.24.4/src/mat/matref.rs:819:3
Assertion failed: row_start <= self.nrows()
- |
| `garch_fit` | `kw:forecast_horizon:int=2^31` | ALLOC-ABORT | rc=-6 bytes=17179869184 |
| `garch_fit` | `kw:forecast_horizon:int=2^63` | PANIC | capacity overflow |
| `garch_fit` | `kw:p:int=2^63` | PANIC | capacity overflow |
| `garch_fit` | `kw:q:int=2^63` | PANIC | capacity overflow |
| `gas_volatility` | `kw:horizon:int=2^31` | ALLOC-ABORT | rc=-6 bytes=17179869184 |
| `gas_volatility` | `kw:horizon:int=2^63` | PANIC | capacity overflow |
| `hamilton_filter` | `kw:p:int=2^63` | PANIC | capacity overflow |
| `hansen_seo_test` | `kw:n_boot:int=2^31` | ALLOC-ABORT | rc=-6 bytes=154618822656 |
| `hetero_svar` | `kw:horizon:int=2^31` | ALLOC-ABORT | rc=-6 bytes=120259084344 |
| `hetero_svar` | `kw:horizon:int=2^63` | PANIC | capacity overflow |
| `hetero_svar` | `kw:lags:int=2^63` | PANIC | Assertion failed at /root/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/faer-0.24.4/src/mat/matref.rs:819:3
Assertion failed: row_start <= self.nrows()
- |
| `historical_decomposition` | `big:T=1e5` | HANG | deadline 45.0s |
| `historical_decomposition` | `kw:horizon:int=2^31` | ALLOC-ABORT | rc=-6 bytes=120259084344 |
| `historical_decomposition` | `kw:horizon:int=2^63` | PANIC | capacity overflow |
| `kernel_regression` | `big:T=1e5` | HANG | deadline 45.0s |
| `kernel_ridge` | `big:T=1e5` | PANIC | called `Result::unwrap()` on an `Err` value: AllocError { layout: Layout { size: 80000000000, align: 64 (1 << 6) } } |
| `kernel_ridge` | `kw:rff_features:int=2^31` | ALLOC-ABORT | rc=-6 bytes=51539607552 |
| `kernel_ridge` | `kw:rff_features:int=2^63` | PANIC | capacity overflow |
| `lasso_path` | `kw:n_lambdas:int=2^31` | ALLOC-ABORT | rc=-6 bytes=17179869184 |
| `lasso_path` | `kw:n_lambdas:int=2^63` | PANIC | capacity overflow |
| `long_run_svar` | `kw:horizon:int=2^31` | ALLOC-ABORT | rc=-6 bytes=120259084344 |
| `long_run_svar` | `kw:horizon:int=2^63` | PANIC | capacity overflow |
| `long_run_svar` | `kw:lags:int=2^63` | PANIC | Assertion failed at /root/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/faer-0.24.4/src/mat/matref.rs:819:3
Assertion failed: row_start <= self.nrows()
- |
| `lp` | `kw:horizons:int=2^31` | ALLOC-ABORT | rc=-6 bytes=17179869192 |
| `lp` | `kw:horizons:int=2^63` | PANIC | capacity overflow |
| `lp_iv` | `kw:horizons:int=2^31` | ALLOC-ABORT | rc=-6 bytes=17179869192 |
| `lp_iv` | `kw:horizons:int=2^63` | PANIC | capacity overflow |
| `lp_multiplier` | `kw:horizons:int=2^31` | ALLOC-ABORT | rc=-6 bytes=17179869192 |
| `lp_multiplier` | `kw:horizons:int=2^63` | PANIC | capacity overflow |
| `lp_state` | `kw:horizons:int=2^31` | ALLOC-ABORT | rc=-6 bytes=17179869192 |
| `lp_state` | `kw:horizons:int=2^63` | PANIC | capacity overflow |
| `max_share_svar` | `kw:horizon:int=2^31` | ALLOC-ABORT | rc=-6 bytes=120259084344 |
| `max_share_svar` | `kw:horizon:int=2^63` | PANIC | capacity overflow |
| `max_share_svar` | `kw:lags:int=2^63` | PANIC | Assertion failed at /root/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/faer-0.24.4/src/mat/matref.rs:819:3
Assertion failed: row_start <= self.nrows()
- |
| `mcmc_diagnostics` | `big:T=1e5` | HANG | deadline 45.0s |
| `mean_group_var` | `kw:horizon:int=2^31` | ALLOC-ABORT | rc=-6 bytes=120259084344 |
| `mean_group_var` | `kw:horizon:int=2^63` | PANIC | capacity overflow |
| `midas_weights` | `arg3:int=2^31` | ALLOC-ABORT | rc=-6 bytes=17179869184 |
| `midas_weights` | `arg3:int=2^63` | PANIC | capacity overflow |
| `mlp_regression` | `kw:hidden:ilist_2^31` | ALLOC-ABORT | rc=-6 bytes=68719476744 |
| `mlp_regression` | `kw:max_epochs:int=2^31` | ALLOC-ABORT | rc=-6 bytes=17179869184 |
| `mlp_regression` | `kw:max_epochs:int=2^63` | PANIC | capacity overflow |
| `mlp_regression` | `kw:n_seeds:int=2^31` | ALLOC-ABORT | rc=-6 bytes=154618822656 |
| `mstl` | `kw:iterate:int=2^31` | HANG | deadline 15.0s |
| `mstl` | `kw:iterate:int=2^63` | HANG | deadline 15.0s |
| `narrative_svar` | `kw:horizon:int=2^31` | ALLOC-ABORT | rc=-6 bytes=120259084344 |
| `narrative_svar` | `kw:horizon:int=2^63` | PANIC | capacity overflow |
| `narrative_svar` | `kw:n_draws:int=2^31` | ALLOC-ABORT | rc=-6 bytes=154618822656 |
| `nongaussian_svar` | `kw:horizon:int=2^31` | ALLOC-ABORT | rc=-6 bytes=120259084344 |
| `nongaussian_svar` | `kw:horizon:int=2^63` | PANIC | capacity overflow |
| `nongaussian_svar` | `kw:lags:int=2^63` | PANIC | Assertion failed at /root/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/faer-0.24.4/src/mat/matref.rs:819:3
Assertion failed: row_start <= self.nrows()
- |
| `nsdiffs` | `arg1:int=2^63` | PANIC | index out of bounds: the len is 200 but the index is 9223372036854775808 |
| `pacf` | `kw:nlags:int=2^63` | PANIC | capacity overflow |
| `philox_uniforms` | `arg1:int=2^31` | ALLOC-ABORT | rc=-6 bytes=17179869184 |
| `philox_uniforms` | `arg1:int=2^63` | PANIC | capacity overflow |
| `proxy_ar_sets` | `kw:horizon:int=2^31` | ALLOC-ABORT | rc=-6 bytes=120259084344 |
| `proxy_ar_sets` | `kw:horizon:int=2^63` | PANIC | capacity overflow |
| `proxy_ar_sets` | `kw:lags:int=2^63` | PANIC | Assertion failed at /root/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/faer-0.24.4/src/mat/matref.rs:819:3
Assertion failed: row_start <= self.nrows()
- |
| `proxy_first_stage` | `kw:lags:int=2^63` | PANIC | Assertion failed at /root/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/faer-0.24.4/src/mat/matref.rs:819:3
Assertion failed: row_start <= self.nrows()
- |
| `proxy_svar` | `kw:horizon:int=2^31` | ALLOC-ABORT | rc=-6 bytes=120259084344 |
| `proxy_svar` | `kw:horizon:int=2^63` | PANIC | capacity overflow |
| `proxy_svar` | `kw:lags:int=2^63` | PANIC | Assertion failed at /root/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/faer-0.24.4/src/mat/matref.rs:819:3
Assertion failed: row_start <= self.nrows()
- |
| `proxy_svar_bands` | `kw:horizon:int=2^31` | ALLOC-ABORT | rc=-6 bytes=120259084344 |
| `proxy_svar_bands` | `kw:horizon:int=2^63` | PANIC | capacity overflow |
| `proxy_svar_bands` | `kw:n_boot:int=2^31` | ALLOC-ABORT | rc=-6 bytes=154618822656 |
| `quantile_lp` | `kw:horizons:int=2^31` | ALLOC-ABORT | rc=-6 bytes=17179869192 |
| `quantile_lp` | `kw:horizons:int=2^63` | PANIC | capacity overflow |
| `random_forest` | `kw:n_trees:int=2^31` | ALLOC-ABORT | rc=-6 bytes=154618822728 |
| `reset_test` | `kw:max_power:int=2^31` | HANG | deadline 15.0s |
| `reset_test` | `kw:max_power:int=2^63` | HANG | deadline 15.0s |
| `robust_svar_bounds` | `kw:horizon:int=2^31` | ALLOC-ABORT | rc=-6 bytes=463856468184 |
| `robust_svar_bounds` | `kw:horizon:int=2^63` | PANIC | capacity overflow |
| `robust_svar_bounds` | `kw:lags:int=2^31` | PANIC | called `Result::unwrap()` on an `Err` value: AllocError { layout: Layout { size: 154618822848, align: 64 (1 << 6) } } |
| `robust_svar_bounds` | `kw:lags:int=2^63` | PANIC | called `Result::unwrap()` on an `Err` value: CapacityOverflow |
| `robust_svar_bounds` | `kw:n_draws:int=2^31` | ALLOC-ABORT | rc=-6 bytes=154618822656 |
| `seasonal_strength` | `arg1:int=2^63` | PANIC | index out of bounds: the len is 200 but the index is 9223372036854775808 |
| `setar_test` | `kw:n_boot:int=2^31` | ALLOC-ABORT | rc=-6 bytes=154618822656 |
| `sign_restricted_svar` | `kw:horizon:int=2^31` | ALLOC-ABORT | rc=-6 bytes=120259084344 |
| `sign_restricted_svar` | `kw:horizon:int=2^63` | PANIC | capacity overflow |
| `sign_restricted_svar` | `kw:lags:int=2^31` | PANIC | called `Result::unwrap()` on an `Err` value: AllocError { layout: Layout { size: 154618822848, align: 64 (1 << 6) } } |
| `sign_restricted_svar` | `kw:lags:int=2^63` | PANIC | called `Result::unwrap()` on an `Err` value: CapacityOverflow |
| `sign_restricted_svar` | `kw:n_draws:int=2^31` | ALLOC-ABORT | rc=-6 bytes=154618822656 |
| `star` | `kw:n_c:int=2^31` | ALLOC-ABORT | rc=-6 bytes=17179869184 |
| `star` | `kw:n_c:int=2^63` | PANIC | capacity overflow |
| `star` | `kw:n_gamma:int=2^31` | ALLOC-ABORT | rc=-6 bytes=17179869184 |
| `star` | `kw:n_gamma:int=2^63` | PANIC | capacity overflow |
| `stl` | `arg1:int=2^63` | PANIC | index out of bounds: the len is 200 but the index is 9223372036854775808 |
| `structural_fevd` | `kw:horizon:int=2^31` | ALLOC-ABORT | rc=-6 bytes=120259084344 |
| `structural_fevd` | `kw:horizon:int=2^63` | PANIC | capacity overflow |
| `structural_fevd` | `kw:lags:int=2^63` | PANIC | Assertion failed at /root/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/faer-0.24.4/src/mat/matref.rs:819:3
Assertion failed: row_start <= self.nrows()
- |
| `theta_forecast` | `arg1:int=2^31` | ALLOC-ABORT | rc=-6 bytes=17179869184 |
| `theta_forecast` | `arg1:int=2^63` | PANIC | capacity overflow |
| `theta_forecast` | `kw:period:int=2^63` | PANIC | capacity overflow |
| `threshold_var_test` | `kw:n_boot:int=2^31` | ALLOC-ABORT | rc=-6 bytes=154618822656 |
| `threshold_vecm` | `kw:n_grid_beta:int=2^31` | ALLOC-ABORT | rc=-6 bytes=17179869184 |
| `threshold_vecm` | `kw:n_grid_beta:int=2^63` | PANIC | capacity overflow |
| `var_backtest` | `kw:dq_lags:int=2^63` | PANIC | capacity overflow |
| `var_fevd` | `kw:horizon:int=2^31` | ALLOC-ABORT | rc=-6 bytes=120259084288 |
| `var_fevd` | `kw:horizon:int=2^63` | PANIC | capacity overflow |
| `var_fevd` | `kw:lags:int=2^63` | PANIC | Assertion failed at /root/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/faer-0.24.4/src/mat/matref.rs:819:3
Assertion failed: row_start <= self.nrows()
- |
| `var_fit` | `kw:lags:int=2^63` | PANIC | Assertion failed at /root/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/faer-0.24.4/src/mat/matref.rs:819:3
Assertion failed: row_start <= self.nrows()
- |
| `var_forecast` | `kw:lags:int=2^63` | PANIC | Assertion failed at /root/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/faer-0.24.4/src/mat/matref.rs:819:3
Assertion failed: row_start <= self.nrows()
- |
| `var_forecast` | `kw:steps:int=2^31` | PANIC | called `Result::unwrap()` on an `Err` value: AllocError { layout: Layout { size: 51539607552, align: 64 (1 << 6) } } |
| `var_forecast` | `kw:steps:int=2^63` | PANIC | called `Result::unwrap()` on an `Err` value: CapacityOverflow |
| `var_granger` | `kw:lags:int=2^63` | PANIC | Assertion failed at /root/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/faer-0.24.4/src/mat/matref.rs:819:3
Assertion failed: row_start <= self.nrows()
- |
| `var_irf` | `kw:horizon:int=2^31` | ALLOC-ABORT | rc=-6 bytes=120259084344 |
| `var_irf` | `kw:horizon:int=2^63` | PANIC | capacity overflow |
| `var_irf` | `kw:lags:int=2^63` | PANIC | Assertion failed at /root/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/faer-0.24.4/src/mat/matref.rs:819:3
Assertion failed: row_start <= self.nrows()
- |
| `var_irf_bands` | `kw:horizon:int=2^31` | ALLOC-ABORT | rc=-6 bytes=120259084344 |
| `var_irf_bands` | `kw:horizon:int=2^63` | PANIC | capacity overflow |
| `var_irf_bands` | `kw:lags:int=2^63` | PANIC | Assertion failed at /root/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/faer-0.24.4/src/mat/matref.rs:819:3
Assertion failed: row_start <= self.nrows()
- |
| `zero_sign_svar` | `kw:horizon:int=2^31` | ALLOC-ABORT | rc=-6 bytes=120259084344 |
| `zero_sign_svar` | `kw:horizon:int=2^63` | PANIC | capacity overflow |
| `zero_sign_svar` | `kw:lags:int=2^31` | PANIC | called `Result::unwrap()` on an `Err` value: AllocError { layout: Layout { size: 154618822848, align: 64 (1 << 6) } } |
| `zero_sign_svar` | `kw:lags:int=2^63` | PANIC | called `Result::unwrap()` on an `Err` value: CapacityOverflow |
| `zero_sign_svar` | `kw:n_draws:int=2^31` | ALLOC-ABORT | rc=-6 bytes=154618822656 |
| `zivot_andrews` | `big:T=1e5` | HANG | deadline 45.0s |

## Largest RSS deltas (top 25 cells)

| callable | cell | outcome | RSS delta (MB) | seconds |
|---|---|---|---:|---:|
| `dfm_news` | `big:T=1e5` | ok | 244.5 | 5.077 |
| `acm_term_premium` | `big:T=1e5` | ok | 227.9 | 0.598 |
| `dfm_nowcast` | `big:T=1e5` | ok | 209.2 | 1.952 |
| `dcc_garch` | `big:T=1e5` | ok | 126.5 | 18.582 |
| `ccc_garch` | `big:T=1e5` | ok | 52.8 | 5.431 |
| `panel_lp` | `big:T=1e5` | ok | 52.0 | 1.31 |
| `engle_granger` | `big:T=1e5` | ok | 46.7 | 7.913 |
| `echo_state_network` | `big:T=1e5` | ok | 45.0 | 0.209 |
| `check_series` | `big:T=1e5` | ok | 44.7 | 14.944 |
| `lp_did` | `big:T=1e5` | ok | 43.3 | 0.345 |
| `adf` | `big:T=1e5` | ok | 42.8 | 11.156 |
| `check_stationarity` | `big:T=1e5` | ok | 41.7 | 9.464 |
| `dfgls` | `big:T=1e5` | ok | 40.8 | 8.19 |
| `ng_perron` | `big:T=1e5` | ok | 38.5 | 8.461 |
| `lp_multiplier` | `big:T=1e5` | ok | 26.6 | 1.596 |
| `favar` | `big:T=1e5` | ok | 24.1 | 0.087 |
| `local_level_smooth` | `big:T=1e5` | ok | 19.5 | 0.319 |
| `panel_fe` | `big:T=1e5` | ok | 17.0 | 0.097 |
| `var_fit` | `big:T=1e5` | ok | 17.0 | 0.097 |
| `lp_state` | `big:T=1e5` | ok | 9.8 | 0.399 |
| `mean_group_var` | `big:T=1e5` | ok | 6.1 | 0.042 |
| `lp_iv` | `big:T=1e5` | ok | 5.4 | 0.658 |
| `copula_fit` | `big:T=1e5` | ok | 2.7 | 31.333 |
| `acf` | `arg0:nan_all` | refusal | 0.0 | 0.0 |
| `acf` | `arg0:nan_one` | refusal | 0.0 | 0.0 |

## Slowest cells (top 25, completed)

| callable | cell | outcome | seconds |
|---|---|---|---:|
| `cf_filter` | `big:T=1e5` | ok | 34.071 |
| `copula_fit` | `big:T=1e5` | ok | 31.333 |
| `quantile_lp` | `big:T=1e5` | ok | 28.487 |
| `dcc_garch` | `big:T=1e5` | ok | 18.582 |
| `check_series` | `big:T=1e5` | ok | 14.944 |
| `auto_arima` | `kw:seasonal_period:int=2` | ok | 11.237 |
| `bn_filter` | `big:T=1e5` | ok | 11.23 |
| `adf` | `big:T=1e5` | ok | 11.156 |
| `check_stationarity` | `big:T=1e5` | ok | 9.464 |
| `ng_perron` | `big:T=1e5` | ok | 8.461 |
| `dfgls` | `big:T=1e5` | ok | 8.19 |
| `engle_granger` | `big:T=1e5` | ok | 7.913 |
| `dcs_local_level` | `big:T=1e5` | ok | 7.071 |
| `gmm_nonlinear` | `big:T=1e5` | ok | 6.217 |
| `backtest` | `big:T=1e5` | ok | 6.001 |
| `ccc_garch` | `big:T=1e5` | ok | 5.431 |
| `dfm_news` | `big:T=1e5` | ok | 5.077 |
| `auto_arima` | `kw:max_order:int=0` | ok | 3.977 |
| `auto_arima` | `kw:max_order:int=1` | ok | 3.824 |
| `auto_arima` | `kw:max_Q:int=1` | ok | 3.77 |
| `auto_arima` | `kw:max_P:int=2` | ok | 3.582 |
| `frac_diff` | `big:T=1e5` | ok | 3.565 |
| `auto_arima` | `kw:max_Q:int=2` | ok | 3.564 |
| `frac_integrate` | `big:T=1e5` | ok | 3.534 |
| `auto_arima` | `kw:max_Q:int=0` | ok | 3.524 |
