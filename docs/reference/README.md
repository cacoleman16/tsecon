# Reference

Two complementary references for the `tsecon` API.

## [API reference](api.md)

The complete callable surface — every function's signature and one-line
contract, generated directly from the type stub so it never drifts from the
shipped module. Start here when you know the function and need its arguments.

## [Results objects](results.md)

`tsecon.results` — an opt-in layer of `dict` subclasses that render themselves:
`.summary()` output an economist can read, plus the standard plots. Because they
subclass `dict`, the plain-dict contract every estimator already returns is
preserved exactly; adopting them is additive.

## [Testing & validation](testing.md)

The single map of how the library is tested: nine tiers from Rust goldens to a
published-result replication, what each can and cannot prove, the commands to
run them, and an honest list of what is *not* covered. Start here if you are
deciding whether to trust a number this library produced.

## [Validation matrix](validation-matrix.md)

What every method family is validated *against* — the reference implementation
(statsmodels, SciPy, `arch`, `linearmodels`, scikit-learn, ArviZ) or the
documented closed form, with the fixture, the test, and the tolerance for each.
Correctness, made auditable. Its runtime companion is the
[Monte Carlo suite](../examples/monte-carlo.md), which verifies the statistical
*properties* (size, coverage, consistency) that a fixture match cannot.

## Model cards

One card per method family, each with the same anatomy: **what it estimates ·
assumptions · when to use (and when not) · key arguments and defaults (and why)
· how to read the output · failure modes · what it's validated against ·
references · a runnable example.** Start here when you want to know whether a
method fits your problem and how to trust its output.

| Family | Functions |
|---|---|
| [Diagnostics](model-cards/diagnostics.md) | `acf`, `pacf`, `ljung_box`, `jarque_bera`, `arch_lm`, `adf`, `kpss`, `check_stationarity` |
| [ARIMA](model-cards/arima.md) | `arima_fit`, `ar_loglik` |
| [Spectral analysis](model-cards/spectral.md) | `periodogram`, `welch`, `coherence` |
| [Volatility](model-cards/volatility.md) | `garch_fit`, `gas_volatility`, `ccc_garch`, `dcc_garch` |
| [VAR / SVAR](model-cards/var-svar.md) | `var_fit`, `var_irf`, `var_fevd`, `var_granger`, `var_forecast`, `sign_restricted_svar`, `favar`, `connectedness` |
| [Local projections](model-cards/local-projections.md) | `lp`, `lp_iv`, `lp_multiplier`, `lp_state`, `smooth_lp` |
| [Bayesian](model-cards/bayesian.md) | `bvar_fit`, `bvar_irf_draws`, `mcmc_diagnostics` |
| [GMM](model-cards/gmm.md) | `iv_gmm`, `gmm_nonlinear` |
| [Cointegration & regimes](model-cards/cointegration-regime.md) | `johansen`, `vecm`, `markov_switching_ar` |
| [Forecasting](model-cards/forecasting.md) | `backtest`, `dm_test`, `cw_test`, `gw_test`, `theta_forecast`, `accuracy` |
| [Machine learning](model-cards/machine-learning.md) | `ridge`, `lasso`, `elastic_net`, `adaptive_lasso`, `lasso_path`, `cv_splits` |
| [Panel](model-cards/panel.md) | `panel_fe`, `panel_lp`, `mean_group_var`, `panel_mean_group`, `panel_pmg` |
| [Nowcasting & MIDAS](model-cards/nowcasting-midas.md) | `dfm_nowcast`, `dfm_news`, `midas_weights`, `umidas`, `weighted_midas` |
| [Term structure](model-cards/term-structure.md) | `nelson_siegel`, `svensson`, `dynamic_ns` |
| [Realized volatility](model-cards/realized-vol.md) | `realized_measures`, `har_rv`, `realized_quarticity`, `tripower_quarticity`, `bns_jump_test`, `realized_range` |
| [Predictive regressions & IVX](model-cards/predictive-regressions.md) | `predictive_regression`, `ivx_test` |
| [Recession probability](model-cards/recession.md) | `recession_probit` |
| [Survey expectations](model-cards/expectations.md) | `cg_regression`, `forecast_efficiency`, `forecast_disagreement` |
| [Long memory](model-cards/long-memory.md) | `frac_diff`, `frac_integrate`, `long_memory_d` |
| [Specification & diagnostic tests](model-cards/specification-tests.md) | `heteroskedasticity_test`, `reset_test`, `chow_test`, `cusum_test` |
| [Arbitrage-free Nelson-Siegel](model-cards/afns.md) | `afns_adjustment` |
| [DSGE (linear RE solver)](model-cards/dsge.md) | `dsge_solve` |
| [Quantile regression & growth-at-risk](model-cards/quantile.md) | `quantile_regression`, `quantile_lp`, `growth_at_risk` |
| [Functional shocks (FVAR/FLP)](model-cards/functional-shocks.md) | `functional_pca`, `flp`, `flp_scenario`, `fvar_scenario` |
| [Structural breaks](model-cards/structural-breaks.md) | `bai_perron`, `sup_f_test` |
