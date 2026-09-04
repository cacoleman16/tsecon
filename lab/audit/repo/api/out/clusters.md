### Parameter-name clusters

| concept | spelling | n functions | functions |
|---|---|---|---|
| randomness | `seed` | 21 | `bootstrap_indices`, `bvar_irf_draws`, `bvar_ssvs`, `conformal_backtest`, `conformal_forecast`, `echo_state_network`, `fry_pagan_svar`, `hansen_seo_test`, `historical_decomposition`, `kernel_ridge`, `mlp_regression`, `narrative_svar` … (+9) |
| randomness | `band_seed` | 4 | `lp`, `smooth_lp`, `var_forecast`, `var_irf_bands` |
| randomness | `rf_seed` | 1 | `proxy_ar_sets` |
| replication count | `n_draws` | 8 | `bvar_irf_draws`, `bvar_ssvs`, `fry_pagan_svar`, `historical_decomposition`, `narrative_svar`, `robust_svar_bounds`, `sign_restricted_svar`, `zero_sign_svar` |
| replication count | `n_boot` | 7 | `conformal_backtest`, `conformal_forecast`, `hansen_seo_test`, `proxy_svar_bands`, `setar_test`, `threshold_var_test`, `var_irf_bands` |
| replication count | `max_tries` | 5 | `fry_pagan_svar`, `historical_decomposition`, `narrative_svar`, `sign_restricted_svar`, `zero_sign_svar` |
| replication count | `band_n_sim` | 4 | `lp`, `smooth_lp`, `var_forecast`, `var_irf_bands` |
| replication count | `n_grid` | 3 | `bvar_hierarchical`, `hansen_seo_test`, `threshold_var_test` |
| replication count | `n_eval` | 2 | `conformal_backtest`, `conformal_forecast` |
| replication count | `n_weight_draws` | 2 | `historical_decomposition`, `narrative_svar` |
| replication count | `burn` | 1 | `bvar_ssvs` |
| replication count | `n_c` | 1 | `star` |
| replication count | `n_chains` | 1 | `bvar_ssvs` |
| replication count | `n_gamma` | 1 | `star` |
| replication count | `n_grid_beta` | 1 | `threshold_vecm` |
| replication count | `n_grid_gamma` | 1 | `threshold_vecm` |
| replication count | `n_lambdas` | 1 | `lasso_path` |
| replication count | `n_permutations` | 1 | `random_forest` |
| replication count | `n_seeds` | 1 | `mlp_regression` |
| replication count | `n_trees` | 1 | `random_forest` |
| replication count | `thin` | 1 | `bvar_ssvs` |
| confidence level | `alpha` | 24 | `adaptive_lasso`, `auto_arima`, `check_series`, `check_stationarity`, `conformal_backtest`, `conformal_forecast`, `elastic_net`, `group_lasso`, `ivx_test`, `kernel_ridge`, `lasso`, `mlp_regression` … (+12) |
| confidence level | `band_alpha` | 6 | `lp`, `lp_iv`, `lp_multiplier`, `lp_state`, `panel_lp`, `smooth_lp` |
| confidence level | `conf_alpha` | 2 | `arima_fit`, `auto_arima` |
| confidence level | `level` | 1 | `ou_fit` |
| penalty strength | `alpha` | 24 | `adaptive_lasso`, `auto_arima`, `check_series`, `check_stationarity`, `conformal_backtest`, `conformal_forecast`, `elastic_net`, `group_lasso`, `ivx_test`, `kernel_ridge`, `lasso`, `mlp_regression` … (+12) |
| penalty strength | `lambda1` | 9 | `bvar_fit`, `bvar_irf_draws`, `fry_pagan_svar`, `historical_decomposition`, `narrative_svar`, `robust_svar_bounds`, `sign_restricted_svar`, `svensson`, `zero_sign_svar` |
| penalty strength | `l1_ratio` | 5 | `adaptive_lasso`, `elastic_net`, `group_lasso`, `lasso_path`, `post_lasso` |
| penalty strength | `lambda0` | 3 | `bvar_fit`, `bvar_hierarchical`, `bvar_irf_draws` |
| penalty strength | `lambda3` | 3 | `bvar_fit`, `bvar_hierarchical`, `bvar_irf_draws` |
| penalty strength | `lam` | 2 | `l1_trend_filter`, `smooth_lp` |
| penalty strength | `lamb` | 1 | `hp_filter` |
| penalty strength | `lambda1_hi` | 1 | `bvar_hierarchical` |
| penalty strength | `lambda1_init` | 1 | `bvar_hierarchical` |
| penalty strength | `lambda1_lo` | 1 | `bvar_hierarchical` |
| penalty strength | `mu` | 1 | `spread_zscore` |
| penalty strength | `penalty` | 1 | `l1_trend_filter` |
| penalty strength | `ridge_alpha` | 1 | `echo_state_network` |
| lag / order | `lags` | 38 | `bvar_fit`, `bvar_hierarchical`, `bvar_irf_draws`, `bvar_ssvs`, `check_series`, `conformal_backtest`, `conformal_forecast`, `connectedness`, `dcc_test`, `dfgls`, `favar`, `fry_pagan_svar` … (+26) |
| lag / order | `p` | 16 | `arima_fit`, `bn_decomposition`, `bn_filter`, `bootstrap_indices`, `ccc_garch`, `dcc_garch`, `dcc_test`, `garch_fit`, `hamilton_filter`, `setar`, `setar_test`, `star` … (+4) |
| lag / order | `n_lag_controls` | 9 | `flp`, `flp_scenario`, `lp`, `lp_iv`, `lp_multiplier`, `lp_state`, `panel_lp`, `quantile_lp`, `smooth_lp` |
| lag / order | `maxlags` | 8 | `cg_regression`, `forecast_efficiency`, `hamilton_filter`, `lp`, `lp_multiplier`, `lp_state`, `ols`, `umidas` |
| lag / order | `delay` | 7 | `setar`, `setar_test`, `star`, `star_eval`, `star_test`, `threshold_var`, `threshold_var_test` |
| lag / order | `q` | 6 | `arima_fit`, `bn_decomposition`, `ccc_garch`, `dcc_garch`, `dcc_test`, `garch_fit` |
| lag / order | `d` | 5 | `arima_fit`, `auto_arima`, `frac_diff`, `frac_integrate`, `pds_lasso` |
| lag / order | `nlags` | 5 | `acf`, `arch_lm`, `kpss`, `ljung_box`, `pacf` |
| lag / order | `delays` | 4 | `setar`, `star`, `star_test`, `threshold_var` |
| lag / order | `hac_maxlags` | 4 | `flp`, `flp_scenario`, `har_rv`, `smooth_lp` |
| lag / order | `k_ar_diff` | 4 | `hansen_seo_test`, `johansen`, `threshold_vecm`, `vecm` |
| lag / order | `max_lags` | 4 | `dfgls`, `ng_perron`, `panel_unit_root`, `zivot_andrews` |
| lag / order | `order` | 4 | `conformal_backtest`, `conformal_forecast`, `l1_trend_filter`, `markov_switching_ar` |
| lag / order | `hac_lags` | 3 | `pds_lasso`, `proxy_ar_sets`, `proxy_first_stage` |
| lag / order | `max_d` | 3 | `auto_arima`, `ndiffs`, `nsdiffs` |
| lag / order | `factor_order` | 2 | `dfm_news`, `dfm_nowcast` |
| lag / order | `lrv_lags` | 2 | `cw_test`, `gw_test` |
| lag / order | `maxlag` | 2 | `adf`, `engle_granger` |
| lag / order | `D` | 1 | `auto_arima` |
| lag / order | `ar` | 1 | `bn_decomposition` |
| lag / order | `ma` | 1 | `bn_decomposition` |
| lag / order | `max_D` | 1 | `auto_arima` |
| lag / order | `max_P` | 1 | `auto_arima` |
| lag / order | `max_Q` | 1 | `auto_arima` |
| lag / order | `max_order` | 1 | `auto_arima` |
| lag / order | `max_p` | 1 | `auto_arima` |
| lag / order | `max_q` | 1 | `auto_arima` |
| horizon | `horizon` | 30 | `backtest`, `bvar_irf_draws`, `bvar_ssvs`, `conformal_backtest`, `conformal_forecast`, `connectedness`, `cv_splits`, `favar`, `fry_pagan_svar`, `fvar_scenario`, `gas_volatility`, `growth_at_risk` … (+18) |
| horizon | `horizons` | 8 | `flp`, `flp_scenario`, `lp`, `lp_iv`, `lp_multiplier`, `lp_state`, `quantile_lp`, `smooth_lp` |
| horizon | `forecast_horizon` | 3 | `ccc_garch`, `dcc_garch`, `garch_fit` |
| horizon | `forecast_steps` | 2 | `arima_fit`, `auto_arima` |
| horizon | `h` | 2 | `dm_test`, `hamilton_filter` |
| horizon | `steps` | 2 | `theta_forecast`, `var_forecast` |
| horizon | `h1` | 1 | `max_share_svar` |
| horizon | `n_steps` | 1 | `boosting` |
| horizon | `post_window` | 1 | `lp_did` |
| horizon | `pre_window` | 1 | `lp_did` |
| trend / deterministic | `trend` | 23 | `connectedness`, `engle_granger`, `favar`, `hetero_svar`, `long_run_svar`, `max_share_svar`, `mean_group_var`, `mstl`, `ng_perron`, `nongaussian_svar`, `phillips_ouliaris`, `proxy_ar_sets` … (+11) |
| trend / deterministic | `constant` | 6 | `arima_fit`, `setar`, `star`, `star_eval`, `threshold_var`, `threshold_var_test` |
| trend / deterministic | `regression` | 6 | `adf`, `dfgls`, `kpss`, `panel_unit_root`, `phillips_perron`, `zivot_andrews` |
| trend / deterministic | `drift` | 2 | `bn_decomposition`, `cf_filter` |
| trend / deterministic | `deterministic` | 1 | `vecm` |
| trend / deterministic | `first_season` | 1 | `vecm` |
| trend / deterministic | `intercept` | 1 | `ar_loglik` |
| trend / deterministic | `seasons` | 1 | `vecm` |
| standard-error type | `bandwidth` | 6 | `iv_gmm`, `kernel_regression`, `long_run_variance`, `panel_fe`, `panel_lp`, `phillips_ouliaris` |
| standard-error type | `use_correction` | 5 | `cg_regression`, `forecast_efficiency`, `hamilton_filter`, `har_rv`, `ols` |
| standard-error type | `se` | 4 | `hamilton_filter`, `lp`, `lp_state`, `quantile_regression` |
| standard-error type | `se_type` | 4 | `ols`, `panel_fe`, `panel_lp`, `umidas` |
| standard-error type | `kernel` | 3 | `kernel_regression`, `kernel_ridge`, `long_run_variance` |
| standard-error type | `robust` | 2 | `mstl`, `stl` |
| tolerance / iterations | `max_iter` | 13 | `adaptive_lasso`, `bvar_hierarchical`, `elastic_net`, `group_lasso`, `iv_gmm`, `l1_trend_filter`, `lasso`, `lasso_path`, `markov_switching_ar`, `nongaussian_svar`, `panel_pmg`, `pds_lasso` … (+1) |
| tolerance / iterations | `tol` | 13 | `adaptive_lasso`, `bvar_hierarchical`, `elastic_net`, `group_lasso`, `iv_gmm`, `l1_trend_filter`, `lasso`, `lasso_path`, `markov_switching_ar`, `nongaussian_svar`, `panel_pmg`, `pds_lasso` … (+1) |
| tolerance / iterations | `inner_iter` | 2 | `mstl`, `stl` |
| tolerance / iterations | `outer_iter` | 2 | `mstl`, `stl` |
| tolerance / iterations | `max_epochs` | 1 | `mlp_regression` |
| tolerance / iterations | `patience` | 1 | `mlp_regression` |
| method selector (string) | `trend` | 23 | `connectedness`, `engle_granger`, `favar`, `hetero_svar`, `long_run_svar`, `max_share_svar`, `mean_group_var`, `mstl`, `ng_perron`, `nongaussian_svar`, `phillips_ouliaris`, `proxy_ar_sets` … (+11) |
| method selector (string) | `method` | 14 | `box_cox_lambda`, `conformal_backtest`, `conformal_forecast`, `copula_fit`, `copula_select`, `dfgls`, `dfm_nowcast`, `hamilton_filter`, `iv_gmm`, `long_memory_d`, `pacf`, `panel_mean_group` … (+2) |
| method selector (string) | `band` | 8 | `lp`, `lp_iv`, `lp_multiplier`, `lp_state`, `panel_lp`, `smooth_lp`, `var_forecast`, `var_irf_bands` |
| method selector (string) | `cumulative` | 7 | `bvar_irf_draws`, `lp`, `lp_iv`, `lp_state`, `panel_lp`, `var_irf`, `var_irf_bands` |
| method selector (string) | `regression` | 6 | `adf`, `dfgls`, `kpss`, `panel_unit_root`, `phillips_perron`, `zivot_andrews` |
| method selector (string) | `mean` | 4 | `ccc_garch`, `dcc_garch`, `dcc_test`, `garch_fit` |
| method selector (string) | `scheme` | 4 | `bootstrap_indices`, `cv_splits`, `midas_weights`, `weighted_midas` |
| method selector (string) | `se_type` | 4 | `ols`, `panel_fe`, `panel_lp`, `umidas` |
| method selector (string) | `vol` | 4 | `ccc_garch`, `dcc_garch`, `dcc_test`, `garch_fit` |
| method selector (string) | `window` | 4 | `backtest`, `coherence`, `periodogram`, `welch` |
| method selector (string) | `autolag` | 3 | `adf`, `engle_granger`, `zivot_andrews` |
| method selector (string) | `kernel` | 3 | `kernel_regression`, `kernel_ridge`, `long_run_variance` |
| method selector (string) | `test` | 3 | `heteroskedasticity_test`, `ndiffs`, `panel_unit_root` |
| method selector (string) | `univariate_dist` | 3 | `ccc_garch`, `dcc_garch`, `dcc_test` |
| method selector (string) | `band_scope` | 2 | `var_forecast`, `var_irf_bands` |
| method selector (string) | `base` | 2 | `conformal_backtest`, `conformal_forecast` |
| method selector (string) | `calib` | 2 | `conformal_backtest`, `conformal_forecast` |
| method selector (string) | `dist` | 2 | `dcc_garch`, `garch_fit` |
| method selector (string) | `ic` | 2 | `auto_arima`, `setar` |
| method selector (string) | `mode` | 2 | `conformal_backtest`, `conformal_forecast` |
| method selector (string) | `model` | 2 | `star`, `star_eval` |
| method selector (string) | `test_type` | 2 | `phillips_ouliaris`, `phillips_perron` |
| method selector (string) | `variant` | 2 | `dcc_garch`, `har_rv` |
| method selector (string) | `weight` | 2 | `gmm_nonlinear`, `iv_gmm` |
| method selector (string) | `activation` | 1 | `mlp_regression` |
| method selector (string) | `bandwidth_method` | 1 | `kernel_regression` |
| method selector (string) | `bootstrap` | 1 | `random_forest` |
| method selector (string) | `deterministic` | 1 | `vecm` |
| method selector (string) | `family` | 1 | `copula_fit` |
| method selector (string) | `forecaster` | 1 | `backtest` |
| method selector (string) | `group_weights` | 1 | `group_lasso` |
| method selector (string) | `hyperprior` | 1 | `bvar_hierarchical` |
| method selector (string) | `identification` | 1 | `historical_decomposition` |
| method selector (string) | `importance` | 1 | `random_forest` |
| method selector (string) | `kind` | 1 | `kernel_regression` |
| method selector (string) | `loss` | 1 | `dm_test` |
| method selector (string) | `max_features` | 1 | `random_forest` |
| method selector (string) | `penalty` | 1 | `l1_trend_filter` |
| method selector (string) | `policy` | 1 | `favar` |
| method selector (string) | `sign_normalization` | 1 | `hetero_svar` |
| method selector (string) | `solver` | 1 | `mlp_regression` |
| method selector (string) | `split` | 1 | `chow_test` |
| method selector (string) | `stop` | 1 | `boosting` |
| data (first positional) | `y` | 82 | `acf`, `adaptive_lasso`, `adf`, `ar_loglik`, `arima_fit`, `auto_arima`, `backtest`, `bai_perron`, `bk_filter`, `bn_decomposition`, `bn_filter`, `boosting` … (+70) |
| data (first positional) | `x` | 38 | `adaptive_lasso`, `bai_perron`, `boosting`, `chow_test`, `coherence`, `cusum_test`, `echo_state_network`, `elastic_net`, `frac_diff`, `frac_integrate`, `group_lasso`, `heteroskedasticity_test` … (+26) |
| data (first positional) | `data` | 37 | `bvar_fit`, `bvar_hierarchical`, `bvar_irf_draws`, `bvar_ssvs`, `check_series`, `connectedness`, `dfm_nowcast`, `engle_granger`, `factor_model`, `fry_pagan_svar`, `hansen_seo_test`, `hetero_svar` … (+25) |
| data (first positional) | `returns` | 7 | `bns_jump_test`, `ccc_garch`, `dcc_garch`, `dcc_test`, `realized_measures`, `realized_quarticity`, `tripower_quarticity` |
| data (first positional) | `d` | 5 | `arima_fit`, `auto_arima`, `frac_diff`, `frac_integrate`, `pds_lasso` |
| data (first positional) | `maturities` | 5 | `acm_term_premium`, `afns_adjustment`, `dynamic_ns`, `nelson_siegel`, `svensson` |
| data (first positional) | `shock` | 5 | `lp`, `lp_state`, `panel_lp`, `quantile_lp`, `smooth_lp` |
| data (first positional) | `proxy` | 4 | `proxy_ar_sets`, `proxy_first_stage`, `proxy_svar`, `proxy_svar_bands` |
| data (first positional) | `curves` | 3 | `flp_scenario`, `functional_pca`, `fvar_scenario` |
| data (first positional) | `high` | 3 | `bk_filter`, `cf_filter`, `realized_range` |
| data (first positional) | `impulse` | 3 | `lp_iv`, `lp_multiplier`, `mean_group_var` |
| data (first positional) | `low` | 3 | `bk_filter`, `cf_filter`, `realized_range` |
| data (first positional) | `outcome` | 3 | `lp_did`, `panel_fe`, `panel_lp` |
| data (first positional) | `panel` | 3 | `dynamic_ns`, `favar`, `forecast_disagreement` |
| data (first positional) | `xs` | 3 | `ivx_test`, `panel_mean_group`, `panel_pmg` |
| data (first positional) | `yields` | 3 | `acm_term_premium`, `nelson_siegel`, `svensson` |
| data (first positional) | `hf_lags` | 2 | `umidas`, `weighted_midas` |
| data (first positional) | `instrument` | 2 | `lp_iv`, `lp_multiplier` |
| data (first positional) | `r` | 2 | `ivx_test`, `predictive_regression` |
| data (first positional) | `regressors` | 2 | `forecast_efficiency`, `panel_fe` |
| data (first positional) | `target` | 2 | `fry_pagan_svar`, `max_share_svar` |
| data (first positional) | `u` | 2 | `copula_fit`, `copula_select` |
| data (first positional) | `ys` | 2 | `panel_mean_group`, `panel_pmg` |
| data (first positional) | `actual` | 1 | `accuracy` |
| data (first positional) | `chains` | 1 | `mcmc_diagnostics` |
| data (first positional) | `conditions` | 1 | `growth_at_risk` |
| data (first positional) | `e1` | 1 | `dm_test` |
| data (first positional) | `e2` | 1 | `dm_test` |
| data (first positional) | `e_large` | 1 | `cw_test` |
| data (first positional) | `e_small` | 1 | `cw_test` |
| data (first positional) | `entities` | 1 | `mean_group_var` |
| data (first positional) | `forecast` | 1 | `accuracy` |
| data (first positional) | `insample` | 1 | `accuracy` |
| data (first positional) | `loss1` | 1 | `gw_test` |
| data (first positional) | `loss2` | 1 | `gw_test` |
| data (first positional) | `new_vintage` | 1 | `dfm_news` |
| data (first positional) | `obj` | 1 | `summarize` |
| data (first positional) | `old_vintage` | 1 | `dfm_news` |
| data (first positional) | `resid` | 1 | `arch_lm` |
| data (first positional) | `returns_or_hits` | 1 | `var_backtest` |
| data (first positional) | `rv` | 1 | `har_rv` |
| data (first positional) | `scores` | 1 | `flp` |
| data (first positional) | `state_indicator` | 1 | `lp_state` |
| data (first positional) | `treatment` | 1 | `lp_did` |
| data (first positional) | `yhat_large` | 1 | `cw_test` |
| data (first positional) | `yhat_small` | 1 | `cw_test` |
| data (first positional) | `z` | 1 | `iv_gmm` |
| prediction input | `x_test` | 7 | `boosting`, `echo_state_network`, `kernel_regression`, `kernel_ridge`, `mlp_regression`, `random_forest`, `regression_tree` |
| prediction input | `test` | 3 | `heteroskedasticity_test`, `ndiffs`, `panel_unit_root` |
| train/test split | `scheme` | 4 | `bootstrap_indices`, `cv_splits`, `midas_weights`, `weighted_midas` |
| train/test split | `window` | 4 | `backtest`, `coherence`, `periodogram`, `welch` |
| train/test split | `n_eval` | 2 | `conformal_backtest`, `conformal_forecast` |
| train/test split | `train` | 2 | `backtest`, `cv_splits` |
| train/test split | `validation_fraction` | 1 | `mlp_regression` |

### Return-key clusters (top level)

| concept | spelling | n functions | functions |
|---|---|---|---|
| standard errors | `se` | 16 | `copula_fit`, `flp`, `flp_scenario`, `long_memory_d`, `lp`, `lp_did`, `lp_iv`, `lp_multiplier`, `panel_lp`, `panel_mean_group`, `pds_lasso`, `proxy_first_stage` … (+4) |
| standard errors | `bse` | 12 | `arima_fit`, `auto_arima`, `bai_perron`, `forecast_efficiency`, `growth_at_risk`, `har_rv`, `iv_gmm`, `ols`, `panel_fe`, `quantile_regression`, `recession_probit`, `umidas` |
| standard errors | `se_valid` | 8 | `arima_fit`, `auto_arima`, `copula_fit`, `garch_fit`, `gev_fit`, `gpd_fit`, `star`, `star_eval` |
| standard errors | `se_type` | 4 | `lp_did`, `ols`, `panel_fe`, `panel_lp` |
| standard errors | `bse_high` | 3 | `setar`, `threshold_var`, `threshold_vecm` |
| standard errors | `bse_low` | 3 | `setar`, `threshold_var`, `threshold_vecm` |
| standard errors | `bse_linear` | 2 | `star`, `star_eval` |
| standard errors | `bse_nonlinear` | 2 | `star`, `star_eval` |
| standard errors | `kappa_se` | 2 | `dcs_local_level`, `ou_fit` |
| standard errors | `se_c` | 2 | `star`, `star_eval` |
| standard errors | `se_gamma` | 2 | `star`, `star_eval` |
| standard errors | `se_method` | 2 | `lp`, `lp_state` |
| standard errors | `se_xi` | 2 | `gev_fit`, `gpd_fit` |
| standard errors | `bartlett_se` | 1 | `acf` |
| standard errors | `bse_powell` | 1 | `growth_at_risk` |
| standard errors | `c_se` | 1 | `ou_fit` |
| standard errors | `coefs_se` | 1 | `mean_group_var` |
| standard errors | `cycle_se` | 1 | `bn_filter` |
| standard errors | `forecast_se` | 1 | `arima_fit` |
| standard errors | `intercept_se` | 1 | `mean_group_var` |
| standard errors | `irf_path_se` | 1 | `mean_group_var` |
| standard errors | `mu_se` | 1 | `ou_fit` |
| standard errors | `nu_se` | 1 | `dcs_local_level` |
| standard errors | `orth_irfs_se` | 1 | `mean_group_var` |
| standard errors | `phi_se` | 1 | `ou_fit` |
| standard errors | `scale_se` | 1 | `dcs_local_level` |
| standard errors | `se_asymptotic` | 1 | `long_memory_d` |
| standard errors | `se_beta` | 1 | `gpd_fit` |
| standard errors | `se_intercept` | 1 | `cg_regression` |
| standard errors | `se_mle` | 1 | `garch_fit` |
| standard errors | `se_mu` | 1 | `gev_fit` |
| standard errors | `se_raw` | 1 | `smooth_lp` |
| standard errors | `se_regression` | 1 | `long_memory_d` |
| standard errors | `se_rho` | 1 | `copula_fit` |
| standard errors | `se_robust` | 1 | `garch_fit` |
| standard errors | `se_sigma` | 1 | `gev_fit` |
| standard errors | `se_slope` | 1 | `cg_regression` |
| standard errors | `se_state0` | 1 | `lp_state` |
| standard errors | `se_state1` | 1 | `lp_state` |
| standard errors | `sigma_se` | 1 | `ou_fit` |
| standard errors | `theta_se` | 1 | `panel_pmg` |
| p-values | `p_value` | 16 | `adf`, `arch_lm`, `cw_test`, `dcc_test`, `dfgls`, `dm_test`, `gw_test`, `hansen_seo_test`, `jarque_bera`, `kpss`, `panel_unit_root`, `pds_lasso` … (+4) |
| p-values | `pvalue` | 8 | `chow_test`, `engle_granger`, `heteroskedasticity_test`, `ivx_test`, `phillips_ouliaris`, `phillips_perron`, `reset_test`, `zivot_andrews` |
| p-values | `adf_p_value` | 1 | `check_stationarity` |
| p-values | `bp_pvalue` | 1 | `ljung_box` |
| p-values | `f_pvalue` | 1 | `heteroskedasticity_test` |
| p-values | `h1_p_value` | 1 | `star_test` |
| p-values | `h2_p_value` | 1 | `star_test` |
| p-values | `h3_p_value` | 1 | `star_test` |
| p-values | `kpss_p_value` | 1 | `check_stationarity` |
| p-values | `lb_pvalue` | 1 | `ljung_box` |
| p-values | `lm3_f_p_value` | 1 | `star_test` |
| p-values | `lm3_p_value` | 1 | `star_test` |
| p-values | `p` | 1 | `dsge_solve` |
| p-values | `p_cc` | 1 | `var_backtest` |
| p-values | `p_dq` | 1 | `var_backtest` |
| p-values | `p_ind` | 1 | `var_backtest` |
| p-values | `p_slope` | 1 | `cg_regression` |
| p-values | `p_tail` | 1 | `gpd_fit` |
| p-values | `p_uc` | 1 | `var_backtest` |
| p-values | `per_unit_pvalue` | 1 | `panel_unit_root` |
| p-values | `pvalues` | 1 | `forecast_efficiency` |
| p-values | `wald_pvalue` | 1 | `forecast_efficiency` |
| coefficients | `params` | 17 | `arima_fit`, `auto_arima`, `bai_perron`, `copula_fit`, `favar`, `forecast_efficiency`, `garch_fit`, `gmm_nonlinear`, `growth_at_risk`, `har_rv`, `iv_gmm`, `ols` … (+5) |
| coefficients | `alpha` | 11 | `check_series`, `check_stationarity`, `conformal_backtest`, `conformal_forecast`, `ndiffs`, `nsdiffs`, `proxy_svar_bands`, `robust_svar_bounds`, `var_backtest`, `var_irf_bands`, `vecm` |
| coefficients | `coef` | 8 | `adaptive_lasso`, `boosting`, `elastic_net`, `group_lasso`, `lasso`, `lp_did`, `panel_mean_group`, `pds_lasso` |
| coefficients | `weights` | 8 | `flp_scenario`, `fvar_scenario`, `mlp_regression`, `mstl`, `narrative_svar`, `stl`, `weighted_midas`, `zero_sign_svar` |
| coefficients | `beta` | 7 | `acm_term_premium`, `gpd_fit`, `hamilton_filter`, `hansen_seo_test`, `proxy_first_stage`, `threshold_vecm`, `vecm` |
| coefficients | `a` | 4 | `acm_term_premium`, `dcc_garch`, `gas_volatility`, `summarize` |
| coefficients | `param_names` | 4 | `arima_fit`, `auto_arima`, `copula_fit`, `garch_fit` |
| coefficients | `ar` | 3 | `bn_decomposition`, `bn_filter`, `markov_switching_ar` |
| coefficients | `b` | 3 | `dcc_garch`, `gas_volatility`, `summarize` |
| coefficients | `gamma` | 3 | `kernel_ridge`, `star`, `vecm` |
| coefficients | `params_high` | 3 | `setar`, `threshold_var`, `threshold_vecm` |
| coefficients | `params_low` | 3 | `setar`, `threshold_var`, `threshold_vecm` |
| coefficients | `phi` | 3 | `acm_term_premium`, `ou_fit`, `panel_pmg` |
| coefficients | `B` | 2 | `acm_term_premium`, `hetero_svar` |
| coefficients | `betas` | 2 | `flp`, `flp_scenario` |
| coefficients | `coefs` | 2 | `lasso_path`, `mean_group_var` |
| coefficients | `loadings` | 2 | `dfm_nowcast`, `factor_model` |
| coefficients | `params_linear` | 2 | `star`, `star_eval` |
| coefficients | `params_nonlinear` | 2 | `star`, `star_eval` |
| coefficients | `posterior_mean_coefs` | 2 | `bvar_fit`, `bvar_hierarchical` |
| coefficients | `theta` | 2 | `panel_pmg`, `smooth_lp` |
| coefficients | `beta_grid` | 1 | `threshold_vecm` |
| coefficients | `beta_ivx` | 1 | `ivx_test` |
| coefficients | `beta_linear` | 1 | `threshold_vecm` |
| coefficients | `coef_lasso` | 1 | `post_lasso` |
| coefficients | `coef_mean` | 1 | `bvar_ssvs` |
| coefficients | `coef_ols` | 1 | `post_lasso` |
| coefficients | `coef_path` | 1 | `boosting` |
| coefficients | `coef_per_unit` | 1 | `panel_mean_group` |
| coefficients | `coint_coefs` | 1 | `engle_granger` |
| coefficients | `delta` | 1 | `bn_filter` |
| coefficients | `det_coef` | 1 | `vecm` |
| coefficients | `dual_coef` | 1 | `kernel_ridge` |
| coefficients | `factor_loadings` | 1 | `acm_term_premium` |
| coefficients | `ma` | 1 | `bn_decomposition` |
| coefficients | `omega` | 1 | `gas_volatility` |
| coefficients | `params_named` | 1 | `garch_fit` |
| coefficients | `readout` | 1 | `echo_state_network` |
| fitted values | `fitted` | 12 | `acm_term_premium`, `boosting`, `echo_state_network`, `growth_at_risk`, `kernel_regression`, `kernel_ridge`, `mlp_regression`, `random_forest`, `regression_tree`, `spread_zscore`, `var_fit`, `weighted_midas` |
| fitted values | `trend` | 10 | `bn_decomposition`, `bn_filter`, `cf_filter`, `dfgls`, `hamilton_filter`, `hp_filter`, `l1_trend_filter`, `mstl`, `ng_perron`, `stl` |
| fitted values | `cycle` | 7 | `bk_filter`, `bn_decomposition`, `bn_filter`, `cf_filter`, `hamilton_filter`, `hp_filter`, `l1_trend_filter` |
| fitted values | `predicted` | 5 | `boosting`, `echo_state_network`, `mlp_regression`, `random_forest`, `regression_tree` |
| fitted values | `point` | 3 | `proxy_svar_bands`, `var_forecast`, `var_irf_bands` |
| fitted values | `forecast` | 2 | `dynamic_ns`, `gas_volatility` |
| fitted values | `mean` | 2 | `conformal_backtest`, `conformal_forecast` |
| fitted values | `filtered_prob` | 1 | `markov_switching_ar` |
| fitted values | `filtered_state` | 1 | `local_level_smooth` |
| fitted values | `filtered_state_var` | 1 | `local_level_smooth` |
| fitted values | `forecasts` | 1 | `backtest` |
| fitted values | `nowcast` | 1 | `dfm_nowcast` |
| fitted values | `smoothed_factors` | 1 | `dfm_nowcast` |
| fitted values | `smoothed_prob` | 1 | `markov_switching_ar` |
| fitted values | `smoothed_prob_last_regime` | 1 | `markov_switching_ar` |
| fitted values | `smoothed_state` | 1 | `local_level_smooth` |
| fitted values | `smoothed_state_var` | 1 | `local_level_smooth` |
| residuals | `residuals` | 6 | `arima_fit`, `auto_arima`, `iv_gmm`, `nelson_siegel`, `svensson`, `weighted_midas` |
| residuals | `resid` | 5 | `dcs_local_level`, `engle_granger`, `mstl`, `stl`, `var_fit` |
| residuals | `innovations` | 1 | `bn_decomposition` |
| residuals | `std_resid` | 1 | `gas_volatility` |
| log-likelihood | `loglik` | 19 | `arima_fit`, `auto_arima`, `bn_decomposition`, `ccc_garch`, `copula_fit`, `dcc_garch`, `dcs_local_level`, `dfm_nowcast`, `garch_fit`, `gas_volatility`, `gev_fit`, `gpd_fit` … (+7) |
| log-likelihood | `llf` | 4 | `threshold_var`, `threshold_vecm`, `var_fit`, `vecm` |
| information criteria | `aic` | 13 | `arima_fit`, `auto_arima`, `bn_decomposition`, `copula_fit`, `dcs_local_level`, `garch_fit`, `gas_volatility`, `lasso_path`, `setar`, `star`, `star_eval`, `threshold_var` … (+1) |
| information criteria | `bic` | 13 | `arima_fit`, `auto_arima`, `bn_decomposition`, `copula_fit`, `dcs_local_level`, `garch_fit`, `gas_volatility`, `lasso_path`, `setar`, `star`, `star_eval`, `threshold_var` … (+1) |
| information criteria | `ic` | 2 | `auto_arima`, `setar` |
| information criteria | `aic_path` | 1 | `boosting` |
| information criteria | `aicc` | 1 | `auto_arima` |
| information criteria | `hqic` | 1 | `var_fit` |
| convergence | `converged` | 23 | `arima_fit`, `auto_arima`, `bn_decomposition`, `bvar_hierarchical`, `copula_fit`, `dcc_garch`, `dcs_local_level`, `garch_fit`, `gas_volatility`, `gev_fit`, `gmm_nonlinear`, `gpd_fit` … (+11) |
| convergence | `iterations` | 7 | `dcs_local_level`, `gas_volatility`, `gmm_nonlinear`, `markov_switching_ar`, `panel_pmg`, `quantile_regression`, `weighted_midas` |
| convergence | `n_iter` | 6 | `adaptive_lasso`, `elastic_net`, `group_lasso`, `l1_trend_filter`, `lasso`, `nongaussian_svar` |
| convergence | `boundary` | 3 | `arima_fit`, `auto_arima`, `garch_fit` |
| convergence | `boundary_note` | 3 | `arima_fit`, `auto_arima`, `garch_fit` |
| convergence | `cov_ok` | 2 | `arima_fit`, `auto_arima` |
| convergence | `fevals` | 2 | `gmm_nonlinear`, `star` |
| convergence | `at_bound` | 1 | `box_cox_lambda` |
| convergence | `bandwidth_at_boundary` | 1 | `kernel_regression` |
| convergence | `best_epoch` | 1 | `mlp_regression` |
| convergence | `budget_exhausted` | 1 | `auto_arima` |
| convergence | `n_accepted` | 1 | `fry_pagan_svar` |
| convergence | `n_criterion_evaluations` | 1 | `kernel_regression` |
| convergence | `n_evals` | 1 | `bvar_hierarchical` |
| confidence intervals | `alpha` | 11 | `check_series`, `check_stationarity`, `conformal_backtest`, `conformal_forecast`, `ndiffs`, `nsdiffs`, `proxy_svar_bands`, `robust_svar_bounds`, `var_backtest`, `var_irf_bands`, `vecm` |
| confidence intervals | `lower` | 6 | `box_cox_lambda`, `conformal_backtest`, `conformal_forecast`, `proxy_svar_bands`, `var_forecast`, `var_irf_bands` |
| confidence intervals | `upper` | 6 | `box_cox_lambda`, `conformal_backtest`, `conformal_forecast`, `proxy_svar_bands`, `var_forecast`, `var_irf_bands` |
| confidence intervals | `quantiles` | 3 | `narrative_svar`, `sign_restricted_svar`, `zero_sign_svar` |
| confidence intervals | `set_max` | 3 | `narrative_svar`, `sign_restricted_svar`, `zero_sign_svar` |
| confidence intervals | `set_min` | 3 | `narrative_svar`, `sign_restricted_svar`, `zero_sign_svar` |
| confidence intervals | `band` | 2 | `var_forecast`, `var_irf_bands` |
| confidence intervals | `bound_lower` | 1 | `cusum_test` |
| confidence intervals | `bound_upper` | 1 | `cusum_test` |
| confidence intervals | `ci_lower_90` | 1 | `bai_perron` |
| confidence intervals | `ci_lower_95` | 1 | `bai_perron` |
| confidence intervals | `ci_scale` | 1 | `bai_perron` |
| confidence intervals | `ci_upper_90` | 1 | `bai_perron` |
| confidence intervals | `ci_upper_95` | 1 | `bai_perron` |
| confidence intervals | `conf_alpha` | 1 | `arima_fit` |
| confidence intervals | `conf_int` | 1 | `pds_lasso` |
| confidence intervals | `forecast_lower` | 1 | `arima_fit` |
| confidence intervals | `forecast_upper` | 1 | `arima_fit` |
| confidence intervals | `half_life_ci` | 1 | `ou_fit` |
| confidence intervals | `lower_efron` | 1 | `proxy_svar_bands` |
| confidence intervals | `lower_quantiles` | 1 | `robust_svar_bounds` |
| confidence intervals | `q_lower` | 1 | `conformal_forecast` |
| confidence intervals | `q_upper` | 1 | `conformal_forecast` |
| confidence intervals | `robust_ci_lower` | 1 | `robust_svar_bounds` |
| confidence intervals | `robust_ci_upper` | 1 | `robust_svar_bounds` |
| confidence intervals | `tail_lower` | 1 | `copula_fit` |
| confidence intervals | `tail_upper` | 1 | `copula_fit` |
| confidence intervals | `upper_efron` | 1 | `proxy_svar_bands` |
| confidence intervals | `upper_quantiles` | 1 | `robust_svar_bounds` |
| R-squared | `rsquared` | 6 | `dynamic_ns`, `har_rv`, `nelson_siegel`, `svensson`, `umidas`, `weighted_midas` |
| R-squared | `r_squared` | 2 | `cg_regression`, `forecast_efficiency` |
| R-squared | `rx_rsquared` | 1 | `acm_term_premium` |
| R-squared | `short_rate_rsquared` | 1 | `acm_term_premium` |
| R-squared | `var_rsquared` | 1 | `acm_term_premium` |
| R-squared | `yield_rsquared` | 1 | `acm_term_premium` |
| sample-size echo | `nobs` | 27 | `adf`, `arch_lm`, `cg_regression`, `dcc_test`, `dfgls`, `engle_granger`, `flp`, `hansen_seo_test`, `har_rv`, `iv_gmm`, `ivx_test`, `lp_did` … (+15) |
| sample-size echo | `n` | 6 | `box_cox_lambda`, `check_series`, `copula_fit`, `gpd_fit`, `jarque_bera`, `var_backtest` |
| sample-size echo | `neqs` | 5 | `hansen_seo_test`, `mean_group_var`, `threshold_var`, `threshold_var_test`, `threshold_vecm` |
| sample-size echo | `n_factors` | 4 | `acm_term_premium`, `dfm_nowcast`, `favar`, `flp` |
| sample-size echo | `n_proxy` | 4 | `proxy_ar_sets`, `proxy_first_stage`, `proxy_svar`, `proxy_svar_bands` |
| sample-size echo | `n_regressors` | 4 | `hansen_seo_test`, `threshold_var`, `threshold_var_test`, `threshold_vecm` |
| sample-size echo | `n_units` | 3 | `panel_mean_group`, `panel_pmg`, `panel_unit_root` |
| sample-size echo | `n_vars` | 3 | `engle_granger`, `hetero_svar`, `phillips_ouliaris` |
| sample-size echo | `n_obs` | 2 | `dcs_local_level`, `ou_fit` |
| sample-size echo | `n_train` | 2 | `echo_state_network`, `mlp_regression` |
| sample-size echo | `adf_nobs` | 1 | `engle_granger` |
| sample-size echo | `n_breaks` | 1 | `bai_perron` |
| sample-size echo | `n_calib` | 1 | `conformal_forecast` |
| sample-size echo | `n_controls_selected` | 1 | `pds_lasso` |
| sample-size echo | `n_endog` | 1 | `favar` |
| sample-size echo | `n_eval` | 1 | `conformal_backtest` |
| sample-size echo | `n_knots` | 1 | `l1_trend_filter` |
| sample-size echo | `n_maxima` | 1 | `gev_fit` |
| sample-size echo | `n_models` | 1 | `auto_arima` |
| sample-size echo | `n_origins` | 1 | `backtest` |
| sample-size echo | `n_parameters` | 1 | `mlp_regression` |
| sample-size echo | `n_stacked` | 1 | `dcc_test` |
| sample-size echo | `n_used` | 1 | `proxy_svar_bands` |
| sample-size echo | `n_validation` | 1 | `mlp_regression` |
| sample-size echo | `n_washout` | 1 | `echo_state_network` |
| sample-size echo | `nobs_per_h` | 1 | `lp_multiplier` |
| lag echo | `lags` | 8 | `dcc_test`, `hetero_svar`, `kpss`, `ljung_box`, `mean_group_var`, `phillips_ouliaris`, `phillips_perron`, `zivot_andrews` |
| lag echo | `delay` | 6 | `setar`, `setar_test`, `star`, `star_test`, `threshold_var`, `threshold_var_test` |
| lag echo | `used_lag` | 4 | `adf`, `dfgls`, `engle_granger`, `ng_perron` |
| lag echo | `hac_lags` | 2 | `growth_at_risk`, `proxy_first_stage` |
| lag echo | `k_ar_diff` | 2 | `hansen_seo_test`, `threshold_vecm` |
| lag echo | `max_d` | 2 | `ndiffs`, `nsdiffs` |
| lag echo | `order` | 2 | `auto_arima`, `nongaussian_svar` |
| lag echo | `factor_order` | 1 | `dfm_nowcast` |
| lag echo | `hac_lags_resolved` | 1 | `pds_lasso` |
| lag echo | `maxlags` | 1 | `cg_regression` |
| lag echo | `seasonal_order` | 1 | `auto_arima` |
| variance / covariance | `sigma2` | 8 | `acm_term_premium`, `bn_decomposition`, `ccc_garch`, `dcc_garch`, `panel_pmg`, `setar`, `star`, `star_eval` |
| variance / covariance | `sigma` | 7 | `acm_term_premium`, `cusum_test`, `gev_fit`, `ou_fit`, `spread_zscore`, `threshold_var`, `threshold_vecm` |
| variance / covariance | `sigma_u` | 3 | `favar`, `var_fit`, `vecm` |
| variance / covariance | `variance_forecast` | 3 | `ccc_garch`, `dcc_garch`, `garch_fit` |
| variance / covariance | `correlation` | 2 | `ccc_garch`, `dcc_garch` |
| variance / covariance | `covariance` | 2 | `ccc_garch`, `dcc_garch` |
| variance / covariance | `covariance_forecast` | 2 | `ccc_garch`, `dcc_garch` |
| variance / covariance | `param_cov` | 2 | `arima_fit`, `auto_arima` |
| variance / covariance | `scale` | 2 | `dcs_local_level`, `dfm_nowcast` |
| variance / covariance | `sigma_posterior_mean` | 2 | `bvar_fit`, `bvar_hierarchical` |
| variance / covariance | `correlation_forecast` | 1 | `dcc_garch` |
| variance / covariance | `correlation_last` | 1 | `dcc_garch` |
| variance / covariance | `cov` | 1 | `iv_gmm` |
| variance / covariance | `covs` | 1 | `flp` |
| variance / covariance | `factor_cov` | 1 | `dfm_nowcast` |
| variance / covariance | `omega_bar` | 1 | `bvar_fit` |
| variance / covariance | `qbar` | 1 | `dcc_garch` |
| variance / covariance | `s_bar` | 1 | `bvar_fit` |
| variance / covariance | `sigma2_high` | 1 | `setar` |
| variance / covariance | `sigma2_low` | 1 | `setar` |
| variance / covariance | `sigma_high` | 1 | `threshold_var` |
| variance / covariance | `sigma_low` | 1 | `threshold_var` |
| variance / covariance | `sigma_mean` | 1 | `bvar_ssvs` |
| variance / covariance | `sigma_regime1` | 1 | `hetero_svar` |
| variance / covariance | `sigma_regime2` | 1 | `hetero_svar` |
| variance / covariance | `sigma_se` | 1 | `ou_fit` |
| variance / covariance | `variance` | 1 | `gas_volatility` |
| variance / covariance | `variances` | 1 | `markov_switching_ar` |
| test statistic | `stat` | 9 | `dcc_test`, `engle_granger`, `hansen_seo_test`, `phillips_ouliaris`, `phillips_perron`, `setar_test`, `sup_f_test`, `threshold_var_test`, `zivot_andrews` |
| test statistic | `statistic` | 8 | `adf`, `arch_lm`, `dfgls`, `heteroskedasticity_test`, `jarque_bera`, `kpss`, `panel_unit_root`, `var_granger` |
| test statistic | `tvalues` | 5 | `forecast_efficiency`, `har_rv`, `ols`, `panel_fe`, `quantile_regression` |
| test statistic | `fstat` | 3 | `chow_test`, `heteroskedasticity_test`, `reset_test` |
| test statistic | `q` | 3 | `dsge_solve`, `max_share_svar`, `star_test` |
| test statistic | `wald` | 2 | `forecast_efficiency`, `ivx_test` |
| test statistic | `adf_statistic` | 1 | `check_stationarity` |
| test statistic | `ar_bound_stat` | 1 | `proxy_ar_sets` |
| test statistic | `bp_stat` | 1 | `ljung_box` |
| test statistic | `cw_stat` | 1 | `cw_test` |
| test statistic | `dm_stat` | 1 | `dm_test` |
| test statistic | `dq_stat` | 1 | `var_backtest` |
| test statistic | `gw_stat` | 1 | `gw_test` |
| test statistic | `h1_f_stat` | 1 | `star_test` |
| test statistic | `h2_f_stat` | 1 | `star_test` |
| test statistic | `h3_f_stat` | 1 | `star_test` |
| test statistic | `hln_stat` | 1 | `dm_test` |
| test statistic | `j_stat` | 1 | `iv_gmm` |
| test statistic | `kpss_statistic` | 1 | `check_stationarity` |
| test statistic | `lb_stat` | 1 | `ljung_box` |
| test statistic | `lm3_f_stat` | 1 | `star_test` |
| test statistic | `lm3_stat` | 1 | `star_test` |
| test statistic | `max_eig_stat` | 1 | `johansen` |
| test statistic | `mt_statistic` | 1 | `fry_pagan_svar` |
| test statistic | `t_stat` | 1 | `pds_lasso` |
| test statistic | `trace_stat` | 1 | `johansen` |
| test statistic | `tstat` | 1 | `panel_mean_group` |
| critical values | `crit` | 7 | `adf`, `dfgls`, `engle_granger`, `ng_perron`, `phillips_ouliaris`, `phillips_perron`, `zivot_andrews` |
| critical values | `sup_f_crit` | 1 | `bai_perron` |

### Return-key clusters (nested, `function.key`)

| concept | spelling | n functions | functions |
|---|---|---|---|
| standard errors | `se` | 3 | `predictive_regression.ols`, `predictive_regression.stambaugh`, `proxy_svar.first_stage` |
| p-values | `p_value` | 2 | `check_series.arch_effects`, `check_series.normality` |
| p-values | `pvalue` | 2 | `hetero_svar.covariance_equality`, `predictive_regression.ivx` |
| p-values | `adf_p_value` | 1 | `check_series.stationarity` |
| p-values | `kpss_p_value` | 1 | `check_series.stationarity` |
| coefficients | `alpha` | 4 | `auto_arima.d_test`, `check_series.multiple_testing`, `check_series.stationarity`, `predictive_regression.ols` |
| coefficients | `beta` | 2 | `predictive_regression.ols`, `proxy_svar.first_stage` |
| coefficients | `beta_corrected` | 1 | `predictive_regression.stambaugh` |
| coefficients | `beta_ivx` | 1 | `predictive_regression.ivx` |
| coefficients | `beta_ols` | 1 | `predictive_regression.stambaugh` |
| coefficients | `omega` | 1 | `garch_fit.params_named` |
| fitted values | `mean` | 1 | `check_series.descriptives` |
| fitted values | `trend` | 1 | `stl.config` |
| confidence intervals | `alpha` | 4 | `auto_arima.d_test`, `check_series.multiple_testing`, `check_series.stationarity`, `predictive_regression.ols` |
| sample-size echo | `n_proxy` | 1 | `proxy_svar.first_stage` |
| sample-size echo | `nobs` | 1 | `check_series.arch_effects` |
| lag echo | `hac_lags` | 1 | `proxy_svar.first_stage` |
| lag echo | `max_d` | 1 | `auto_arima.d_test` |
| variance / covariance | `scale` | 1 | `check_series.analysis_scale` |
| test statistic | `statistic` | 3 | `check_series.arch_effects`, `check_series.normality`, `hetero_svar.covariance_equality` |
| test statistic | `adf_statistic` | 1 | `check_series.stationarity` |
| test statistic | `kpss_statistic` | 1 | `check_series.stationarity` |
| test statistic | `tstat` | 1 | `predictive_regression.ols` |
| test statistic | `wald` | 1 | `predictive_regression.ivx` |
