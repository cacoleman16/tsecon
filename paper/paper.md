---
title: 'tsecon: A Rust-core Python library for time-series econometrics'
tags:
  - Python
  - Rust
  - econometrics
  - time series
  - macroeconomics
  - vector autoregression
  - local projections
  - Bayesian VAR
  - nowcasting
authors:
  - name: Chase Coleman
    affiliation: 1
affiliations:
  - name: Independent Researcher, United States
    index: 1
date: 17 July 2026
# Draft — before JOSS submission, confirm author metadata (ORCID, affiliation)
# and reset `date` to the submission date: the paper describes 0.7.0, whose
# measured test and count figures were taken on 30 August 2026, so the current
# placeholder predates the artifact it documents.
bibliography: paper.bib
---

# Summary

`tsecon` is a Python library for applied time-series econometrics with a
compiled Rust core. It brings the methods that empirical macroeconomists and
financial econometricians actually use — structural VARs, local projections,
Bayesian VARs, dynamic-factor nowcasting and MIDAS mixed-frequency regression,
cointegration, threshold and smooth-transition dynamics, volatility models,
and predictive-regression inference — under a single package with one shared
numerical engine and one consistent grammar. The user-facing API is ordinary
Python and NumPy; the estimation kernels (state-space filtering, bootstrap and
posterior resampling, sign-restriction rotation sampling, spectral transforms,
optimization) are implemented in Rust and exposed through `PyO3` and built into
portable wheels with `maturin` [@pyo3; @maturin]. The release described here,
**0.7.0**, exposes **162 functions** organized as a task-oriented facade over
**43 Rust crates**, and NumPy is the only required runtime dependency.

The design goal is not another forecasting toolkit but a maintained, fast, and
internally consistent home for the *inference and identification* work that is
most of empirical macroeconomics: recovering structural impulse responses,
quantifying their uncertainty with resampling, and testing forecasts and
predictability hypotheses under the persistence and heteroskedasticity that
real macro-financial data exhibit.

# Statement of need

Python's incumbent econometrics tools are `statsmodels` [@seabold2010] and, for
volatility, `arch` [@sheppard_arch]. Both are excellent within their scope, and
`statsmodels` 0.15.0 narrowed the distance: it added a single-equation
`LocalProjections`, the Hamilton regression filter, and a Diebold–Mariano test,
and its `DynamicFactorMQ` remains the reference mixed-frequency nowcaster in
Python. What is still absent from it is most of the identification and
inference layer — there is no Bayesian VAR, no structural identification beyond
Cholesky and the short-run A/B `SVAR` (no sign restrictions, proxies, long-run
zeros, heteroskedasticity- or non-Gaussianity-based schemes), no MIDAS, no
threshold or smooth-transition models, and no persistence-robust
predictive-regression inference. Those methods live instead in unmaintained,
non-interoperating packages in other languages — R's
`vars`/`svars`/`lpirfs`/`BVAR`, hand-passed MATLAB replication archives, and
Dynare. The fast Python forecasting libraries, by contrast, have no notion of
identification, structural inference, or the hypothesis tests economists report.
There is no single, maintained, fast Python package that covers structural
identification, local projections, Bayesian VARs, and nowcasting together. That
gap is `tsecon`'s reason to exist.

Concretely, `tsecon` ships the following, with the state of the Python
alternative named in each case:

- **The local-projection family** (`lp`, `lp_iv`, `lp_state`, `panel_lp`,
  `quantile_lp`, `smooth_lp`, `lp_did`) with lag-augmented inference as the
  default, following the recommendation of @montielolea2021, plus LP-IV and
  state-dependent regimes [@jorda2005]. `statsmodels`' new `LocalProjections`
  covers the single-equation HAC case only.
- **Structural VAR identification beyond Cholesky** — `sign_restricted_svar`
  implements sign-restricted Bayesian SVARs via Haar rotation sampling, a
  standard tool of structural macro analysis [@kilian2017], alongside proxy,
  narrative, long-run, max-share, heteroskedasticity- and
  non-Gaussianity-based schemes (`proxy_svar`, `narrative_svar`,
  `long_run_svar`, `max_share_svar`, `hetero_svar`, `nongaussian_svar`).
  `statsmodels`' `SVAR` offers short-run A/B restrictions only.
- **Bayesian VARs** (`bvar_fit`, `bvar_irf_draws`) with a Minnesota / normal-
  inverse-Wishart prior and posterior impulse-response draws [@giannone2015].
- **Nowcasting and mixed frequency** — the MIDAS family (`midas_weights`,
  `umidas`, `weighted_midas`), which `statsmodels` does not implement, next to
  dynamic-factor nowcasting (`dfm_nowcast`) with two-step and one-step
  Gaussian-MLE estimation and a ragged edge [@dozgiannone2012] and the
  Bańbura–Modugno news decomposition (`dfm_news`). This is the one area where
  `statsmodels` is a genuine alternative rather than a gap: `DynamicFactorMQ`
  fits a mixed-frequency state space, which `tsecon` does not, and its results
  expose their own news decomposition.
- **Persistence-robust predictive-regression inference** — `predictive_regression`
  applies the Stambaugh bias correction and `ivx_test` implements the IVX Wald
  test, whose size is robust across the near-unit-root region [@kostakis2015].
- **Threshold and smooth-transition dynamics**, new in 0.7.0 — Hansen–Seo
  threshold cointegration (`threshold_vecm`, `hansen_seo_test`)
  [@hansenseo2002], a two-regime threshold VAR (`threshold_var`,
  `threshold_var_test`), and LSTAR/ESTAR smooth transition with the Teräsvirta
  modeling cycle (`star`, `star_test`) [@terasvirta1994]. None of the reference
  libraries this project validates against implements them; R's `tsDyn` is the
  usual recourse, and the grade at which this family ships is stated below.

Because these estimators lean heavily on simulation — bootstrap confidence
bands, posterior draws, rotation sampling, Monte Carlo studies — a compiled,
multithreaded core is not a micro-optimization; it changes what users attempt.
`tsecon` uses a counter-based Philox generator that is bit-compatible with
NumPy's, so every bootstrap, MCMC, and rotation draw is reproducible at any
thread count. Rayon-parallel resampling makes large wild/block bootstraps the
comfortable default rather than an overnight job.

# Functionality

The 162 functions span the applied workflow end to end:

- **Diagnostics and exploration**: `acf`, `pacf`, `ljung_box`, `jarque_bera`,
  `arch_lm`; the unit-root battery (`adf`, `kpss`, `dfgls`, `phillips_perron`,
  `ng_perron`, `zivot_andrews`, the `ndiffs`/`nsdiffs` differencing advisors,
  and the `check_stationarity` verdict); spectral analysis (`periodogram`,
  `welch`, `coherence`); the one-call `check_series` battery that runs the
  families and returns evidence-backed model recommendations; a
  specification-test battery (`heteroskedasticity_test`, `reset_test`,
  `chow_test`, `cusum_test`); and trend–cycle filters (`hp_filter`, `bk_filter`,
  `cf_filter`, `hamilton_filter`, and the Beveridge–Nelson pair
  `bn_decomposition`/`bn_filter`, for which `statsmodels` has no counterpart).
- **Univariate and volatility models**: `arima_fit` (exact MLE, seasonal via
  `seasonal=(P, D, Q, s)`, with `auto_arima` order selection), the GARCH
  family (`garch_fit`, GJR/EGARCH), score-driven volatility (`gas_volatility`),
  multivariate GARCH (`ccc_garch`, `dcc_garch`), realized measures (`har_rv`,
  `realized_measures`, `bns_jump_test`), long memory (`frac_diff`,
  `long_memory_d`), and Markov-switching (`markov_switching_ar`).
- **Multivariate and structural**: `var_fit` with `var_irf`, `var_fevd`,
  `var_granger`, `var_forecast`; cointegration (`johansen`; `vecm` across every
  `statsmodels` deterministic case and with centered seasonal dummies;
  `engle_granger`, `phillips_ouliaris`); factor models and `favar`; and
  Diebold–Yilmaz `connectedness`.
- **Nonlinear and regime dynamics**: threshold autoregressions (`setar`,
  `setar_test`); smooth-transition LSTAR/ESTAR models by concentrated
  nonlinear least squares with the Teräsvirta modeling cycle (`star`,
  `star_eval`, `star_test`) [@terasvirta1994]; and the multivariate threshold
  pair — a two-regime threshold VAR with a robust sup-Wald test
  (`threshold_var`, `threshold_var_test`) and Hansen–Seo threshold
  cointegration with a fixed-regressor-bootstrap sup-LM test
  (`threshold_vecm`, `hansen_seo_test`) [@hansenseo2002].
- **Forecast evaluation**: `dm_test`, `cw_test`, `gw_test`, `theta_forecast`,
  a leakage-checked rolling/expanding `backtest` engine, and distribution-free
  prediction intervals by split, EnbPI, and adaptive conformal inference
  (`conformal_forecast`, `conformal_backtest`) — with no `statsmodels`
  counterpart.
- **Distributions, breaks, and smoothing**: extreme-value tails and copulas
  (`gpd_fit`, `gev_fit`, `copula_fit`, `copula_select`); quantile regression
  and quantile local projections (`quantile_regression`, `quantile_lp`) with
  the growth-at-risk workflow of @adrian2019 (`growth_at_risk`); functional shocks
  — FPCA of curve-valued shocks with functional local projections and scenario
  analysis (`functional_pca`, `flp`, `flp_scenario`, `fvar_scenario`)
  [@inoue2021]; multiple structural breaks by dynamic programming with
  break-date confidence intervals (`bai_perron`, `sup_f_test`)
  [@baiperron1998]; and smooth local projections (`smooth_lp`)
  [@barnichon2019].
- **Machine learning for econometrics**: penalized regression (`ridge`,
  `lasso`, `elastic_net`, `adaptive_lasso`, `lasso_path`) with leakage-safe
  time-series cross-validation (`cv_splits`).
- **Panel, term structure, and structural-economic models**: the mean-group /
  CCE-MG / PMG panel trio (`panel_mean_group`, `panel_pmg`), panel local
  projections (`panel_lp`); the Nelson–Siegel / Svensson yield curve
  (`nelson_siegel`, `svensson`, `dynamic_ns`) and its arbitrage-free adjustment
  (`afns_adjustment`) [@christensen2011]; GMM/IV-GMM (`iv_gmm`,
  `gmm_nonlinear`); survey-expectations tools (`cg_regression`,
  `forecast_efficiency`); recession-probability models (`recession_probit`);
  and a linear rational-expectations solver (`dsge_solve`) that returns the
  Blanchard–Kahn saddle-path solution [@blanchardkahn1980].

Robust inference is served by a single library-wide HAC implementation that
eighteen of the crates consume, so the same kernel and the same automatic
bandwidth rule stand behind `ols(se_type="hac")`, `lp(se="hac")`,
`hamilton_filter(se="hac")`, `umidas`, and the panel estimators alike, and
identical settings cannot yield different p-values in different modules. The
regression estimators take the nonrobust / HC0–HC3 / HAC menu through
`se_type=`; the projection and filter estimators take `se=` with the methods
appropriate to them (lag-augmented LP inference by default, Driscoll–Kraay and
cluster in the panels).

# Validation-first design

`tsecon`'s central engineering discipline is that no estimator is included
without a named golden validation target — a published table, a reference
implementation, a documented closed form, or a Monte-Carlo size/power check.
The Rust core carries a large unit and integration suite of **1610 passing
`#[test]` cases** (1393 integration tests in `crates/*/tests/`, 217 unit tests
in `src/`); with 54 documentation tests the workspace total is 1664 passing
Rust tests, with 9 explicitly ignored. The Python layer adds a conformance
suite of 1312 tests across 93 files whose fixtures are gated against
`statsmodels`, `arch`, `linearmodels`, `scikit-learn`, SciPy, and `ArviZ`. The
replication fixtures *are* the integration test suite.

Which target each family got is recorded per estimator rather than averaged
away, because the grades genuinely differ. The threshold and smooth-transition
family is the clearest case: no third-party implementation was reachable from
the build container — R's `tsDyn`, the natural reference, could not be
installed because CRAN was unreachable through its egress proxy, and none of
the Python reference libraries implements these estimators — so
`threshold_vecm`, `hansen_seo_test`, `threshold_var`, `threshold_var_test`,
and the `star` family are validated
against an independent NumPy transcription of the published closed forms,
pinned at 1e-10, plus seeded Monte-Carlo size, power, and parameter-recovery
checks. A `tsDyn` reference run remains outstanding, and the validation matrix
says so on the row rather than in a footnote.

This discipline paid off in an unusual way: building against reference
implementations surfaced four genuine defects in the *references* themselves
(a duplicated standard-error column and an early optimizer stop in one library,
a non-converged fit and an operator-precedence slip in a tail-dependence
formula in another), which were then corrected at the fixture source.
Determinism is a first-class property — results are identical at any thread
count because the parallel RNG substreams are reproducible — so
simulation-based confidence bands and posterior summaries are exactly
replayable across machines.

The core is built on a small set of pure-Rust numerical foundations (`faer`
for dense linear algebra, `rayon` for parallelism, `rustfft` for spectral
transforms); everything statistical and econometric is implemented from
scratch. Because these foundations are BLAS-free, the wheels are static and
install without a system numerical stack.

# Acknowledgements

`tsecon` builds on the open-source scientific Python and Rust ecosystems,
including NumPy, and validates against `statsmodels` [@seabold2010] and `arch`
[@sheppard_arch]; the Python bindings and cross-platform wheels are produced
with `PyO3` and `maturin` [@pyo3; @maturin]. We thank the maintainers of those
projects, whose reference implementations served as validation targets
throughout development.

# References
