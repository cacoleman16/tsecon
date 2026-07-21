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
# Draft — confirm author metadata (ORCID, affiliation) before JOSS submission.
bibliography: paper.bib
---

# Summary

`tsecon` is a Python library for applied time-series econometrics with a
compiled Rust core. It brings the methods that empirical macroeconomists and
financial econometricians actually use — structural VARs, local projections,
Bayesian VARs, mixed-frequency nowcasting, cointegration, volatility models,
and predictive-regression inference — under a single package with one shared
numerical engine and one consistent grammar. The user-facing API is ordinary
Python and NumPy; the estimation kernels (state-space filtering, bootstrap and
posterior resampling, sign-restriction rotation sampling, spectral transforms,
optimization) are implemented in Rust and exposed through `PyO3` and built into
portable wheels with `maturin` [@pyo3; @maturin]. The current release exposes
**104 functions** organized as a task-oriented facade over **40 Rust crates**,
and NumPy is the only required runtime dependency.

The design goal is not another forecasting toolkit but a maintained, fast, and
internally consistent home for the *inference and identification* work that is
most of empirical macroeconomics: recovering structural impulse responses,
quantifying their uncertainty with resampling, and testing forecasts and
predictability hypotheses under the persistence and heteroskedasticity that
real macro-financial data exhibit.

# Statement of need

Python's incumbent econometrics tools are `statsmodels` [@seabold2010] and, for
volatility, `arch` [@sheppard_arch]. Both are excellent within their scope, but
the modern macro toolkit is either missing from them or scattered across
unmaintained, non-interoperating packages in other languages — R's
`vars`/`svars`/`lpirfs`/`BVAR`, hand-passed MATLAB replication archives, and
Dynare. The fast Python forecasting libraries, by contrast, have no notion of
identification, structural inference, or the hypothesis tests economists report.
There is no single, maintained, fast Python package that covers structural
identification, local projections, Bayesian VARs, and nowcasting together. That
gap is `tsecon`'s reason to exist.

Concretely, `tsecon` ships capabilities that Python has lacked a unified home
for:

- **Local projections** (`lp`, `lp_iv`, `lp_state`) with lag-augmented
  inference as the default, following the recommendation of
  @montielolea2021, plus LP-IV and state-dependent regimes [@jorda2005].
- **Structural VAR identification beyond Cholesky** — `sign_restricted_svar`
  implements sign-restricted Bayesian SVARs via Haar rotation sampling, a
  standard tool of structural macro analysis [@kilian2017].
- **Bayesian VARs** (`bvar_fit`, `bvar_irf_draws`) with a Minnesota / normal-
  inverse-Wishart prior and posterior impulse-response draws [@giannone2015].
- **Nowcasting and mixed frequency** — dynamic-factor nowcasting
  (`dfm_nowcast`) with two-step and one-step Gaussian-MLE estimation and a
  ragged edge [@dozgiannone2012], the Bańbura–Modugno news decomposition
  (`dfm_news`), and the MIDAS family (`midas_weights`, `umidas`,
  `weighted_midas`).
- **Persistence-robust predictive-regression inference** — `predictive_regression`
  applies the Stambaugh bias correction and `ivx_test` implements the IVX Wald
  test, whose size is robust across the near-unit-root region [@kostakis2015].

Because these estimators lean heavily on simulation — bootstrap confidence
bands, posterior draws, rotation sampling, Monte Carlo studies — a compiled,
multithreaded core is not a micro-optimization; it changes what users attempt.
`tsecon` uses a counter-based Philox generator that is bit-compatible with
NumPy's, so every bootstrap, MCMC, and rotation draw is reproducible at any
thread count. Rayon-parallel resampling makes large wild/block bootstraps the
comfortable default rather than an overnight job.

# Functionality

The 104 functions span the applied workflow end to end:

- **Diagnostics and exploration**: `acf`, `pacf`, `ljung_box`, `jarque_bera`,
  `arch_lm`; the unit-root workflow `adf`, `kpss`, `check_stationarity`;
  spectral analysis (`periodogram`, `welch`, `coherence`); and a specification-
  test battery (`heteroskedasticity_test`, `reset_test`, `chow_test`,
  `cusum_test`).
- **Univariate and volatility models**: `arima_fit` (exact MLE), the GARCH
  family (`garch_fit`, GJR/EGARCH), score-driven volatility (`gas_volatility`),
  multivariate GARCH (`ccc_garch`, `dcc_garch`), realized measures (`har_rv`,
  `realized_measures`, `bns_jump_test`), long memory (`frac_diff`,
  `long_memory_d`), and Markov-switching (`markov_switching_ar`).
- **Multivariate and structural**: `var_fit` with `var_irf`, `var_fevd`,
  `var_granger`, `var_forecast`; cointegration (`johansen`, `vecm`); factor
  models and `favar`; and Diebold–Yilmaz `connectedness`.
- **Forecast evaluation**: `dm_test`, `cw_test`, `gw_test`, `theta_forecast`,
  and a leakage-checked rolling/expanding `backtest` engine.
- **Distributions, breaks, and smoothing**: quantile regression and quantile
  local projections (`quantile_regression`, `quantile_lp`) with the
  growth-at-risk workflow of @adrian2019 (`growth_at_risk`); functional shocks
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

Every regression-based estimator exposes a uniform `se_type=` argument (from
nonrobust through HC0–HC3, HAC/Newey–West with automatic bandwidth, and
bootstrap), served by one library-wide HAC implementation, so identical
settings can never yield different p-values in different modules.

# Validation-first design

`tsecon`'s central engineering discipline is that no estimator is included
without a named golden validation target — a published table, a reference
implementation, or both. The Rust core carries a large unit and integration
suite (more than 800 `#[test]` cases) and the Python layer adds a conformance
suite (380 tests collected) whose fixtures are gated against `statsmodels`,
`arch`, `linearmodels`, `scikit-learn`, SciPy, and `ArviZ`. The replication
fixtures *are* the integration test suite.

This discipline paid off in an unusual way: building against reference
implementations surfaced three genuine defects in the *references* themselves
(a duplicated standard-error column and an early optimizer stop in one library,
a non-converged fit in another), which were then corrected at the fixture
source. Determinism is a first-class property — results are identical at any
thread count because the parallel RNG substreams are reproducible — so
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
