# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/); versioning follows the
pre-1.0 policy in [ROADMAP.md](ROADMAP.md) (minor = breaking allowed, patch =
fixes) until 1.0, then strict [SemVer](https://semver.org/).

## [Unreleased]

Pre-1.0, under active development. The name is settled — the package publishes
to PyPI as `tsecon` at the first tagged release.

### Added — foundations and first model classes
- **Foundations**: Philox RNG (bit-identical to NumPy), special functions and
  the distribution zoo, structured linear algebra (Levinson-Durbin, Toeplitz,
  discrete Lyapunov), the resampling/bootstrap engine, the exact-diffuse
  linear-Gaussian state-space (Kalman) engine, the numerical optimizer suite
  with the Monahan stationarity transform, and the HAC/robust-inference module.
- **Diagnostics**: ACF/PACF, Ljung-Box, Jarque-Bera, ARCH-LM; the full
  unit-root workflow (ADF with MacKinnon p-values, KPSS, `check_stationarity`);
  the semiparametric Phillips family — `phillips_perron` (Z-tau/Z-alpha
  unit-root test) and `phillips_ouliaris` (single-equation residual
  cointegration test), matched to `arch` to < 1e-10 with MacKinnon
  response-surface p-values; spectral analysis (periodogram, Welch, coherence).
- **One-call battery**: `check_series` — the Module 01 flagship — runs the
  diagnostic families in order (outlier screen, ADF+KPSS quadrant with
  analysis-scale routing, Ljung-Box/ACF/PACF, ARCH-LM, Jarque-Bera, a
  sup-F/Bai-Perron mean-shift scan, GPH long memory, seasonality evidence;
  for a 2D panel: per-series integration, Johansen, and VAR lag selection
  with a stability check) and ends in recommendations that route to concrete
  tsecon calls — every hypothesis test on the record in `tests_run` with the
  multiple-testing arithmetic shown, never silently corrected.
  `tsecon.results.check_series` adds `.summary()` and `.plot_diagnostics()`.
- **Univariate models**: exact-MLE ARIMA; GARCH/GJR/EGARCH with normal and
  Student-t QMLE, Bollerslev-Wooldridge robust standard errors, and a fused
  allocation-free likelihood with analytic gradients; GAS/DCS score-driven
  volatility (Gaussian and Student-t); Markov-switching AR; trend-cycle
  filters (HP, one-sided HP, Baxter-King, Christiano-Fitzgerald, Hamilton);
  long memory (fractional differencing/integration, GPH, local Whittle).
- **Multivariate and structural**: reduced-form VAR with IRF/FEVD/Granger/
  forecasting, frequentist IRF confidence bands (`var_irf_bands` — Lütkepohl
  (1990) delta-method SEs validated against statsmodels to machine precision,
  and a Kilian (1998) residual bootstrap with optional bias correction), and an
  honest stability block (`is_stable`/`min_root`); sign-restricted Bayesian
  SVARs; `zero_sign_svar` — the corrected Rubio-Ramírez-Waggoner-Zha (2010) /
  Arias-Rubio-Ramírez-Waggoner (2018) **zero + sign** restricted SVAR (a
  superset of the sign-only sampler that reproduces the recursive Cholesky IRF
  as its degenerate impact-only-zero corner, with the weight-invariant
  identified-set envelope as the prior-robust deliverable); and four closed-form
  point-identification schemes —
  `long_run_svar` (Blanchard-Quah long-run restrictions), `max_share_svar`
  (Uhlig/Francis maximum-FEV-share and Barsky-Sims news shocks), `proxy_svar`
  (external-instrument SVAR-IV with a first-stage-F report and NaN-window
  handling), and `hetero_svar` (Rigobon two-regime identification through
  heteroskedasticity, with a Box's-M covariance-equality gate); FAVAR;
  Diebold-Yilmaz connectedness; the PCA factor model with Bai-Ng selection;
  Johansen cointegration and VECM; multivariate GARCH (CCC/DCC).
- **Post-identification and prior-robust SVAR tools**: a layer that *takes* an
  identification (any impact matrix `A0`, or a sign-restricted set) and answers
  what comes after — `structural_fevd` (forecast-error variance decomposition
  for an arbitrary structural `A0`, the gap the recursive-only `var_fevd`
  leaves; reproduces `var_fevd`/statsmodels exactly for the Cholesky case,
  rows sum to 1 by the rotation-invariant-denominator identity);
  `historical_decomposition` (per-`(time, variable, shock)` contributions with
  the exact `y = baseline + Σ_j hd` adding-up identity, in a Cholesky point mode
  and an importance-weighted sign-identified set mode); `fry_pagan_svar`
  (Fry-Pagan 2011 median-target — the single accepted, coherent draw closest to
  the pointwise-median band, the answer to "medians mix models");
  `robust_svar_bounds` (Giacomini-Kitagawa 2021 prior-robust identified-set
  bounds via the Gafarov-Meier-Montiel-Olea 2018 active-set closed form, exact
  for a single restricted shock and a conservative marginal outer bound for
  jointly-restricted shocks — removing the Haar-prior artifact that pointwise
  sign-restricted bands carry); and `narrative_svar` (Antolín-Díaz-Rubio-Ramírez
  2018 narrative sign restrictions — shock-sign and "most/least important
  contributor" episode statements imposed by importance-reweighting with weight
  `1/P̂(N|S)`, reporting `ess`/`min_ptilde`; a strict superset of
  `sign_restricted_svar` that reproduces it bit-for-bit with no narrative
  restrictions).
- **Local projections**: `lp` (lag-augmented inference by default), `lp_iv`
  with a per-horizon first-stage F, state-dependent `lp_state`, a three-valued
  `cumulative` mode, and `lp_multiplier` — the one-step Ramey-Zubairy integral
  multiplier as a first-class entry point (because outcome-only cumulation is
  a cumulative IRF, not a multiplier).
- **Bayesian**: a Minnesota-NIW Bayesian VAR with closed-form posterior,
  posterior impulse-response draws, and ArviZ-exact convergence diagnostics;
  `bvar_hierarchical` — empirical-Bayes (ML-II / GLP MAP-II) selection of
  the prior tightness by maximizing the Giannone-Lenza-Primiceri (2015)
  marginal likelihood, then refitting the posterior at the optimum; and
  `bvar_ssvs` — the George-Sun-Ni (2008) spike-and-slab **stochastic-search
  variable selection** BVAR (a four-block Gibbs sampler returning per-coefficient
  posterior inclusion probabilities, optional error-precision selection, and
  Cholesky-orthogonalized IRF draws), MC-recovery-validated on a sparse VAR.
- **Forecasting and evaluation**: Diebold-Mariano (HLN), Clark-West,
  Giacomini-White, Theta, accuracy measures, and the rolling/expanding
  backtest engine.
- **GMM**: linear IV-GMM (2SLS/two-step/iterated, Hansen J) and nonlinear GMM
  with Python-callback moment conditions.
- **Predictive regressions**: OLS/Stambaugh/IVX in one call plus the joint
  IVX test — Monte-Carlo-validated to hold size through an exact unit root.
- **Panels**: fixed effects with clustered/Driscoll-Kraay SEs, panel LP,
  mean-group VAR, the heterogeneous-panel trio (mean group, CCE-MG, PMG), and
  `panel_unit_root` — the three first-generation panel unit-root tests
  (Levin-Lin-Chu, Im-Pesaran-Shin, Fisher/Maddala-Wu-Choi) of the joint
  unit-root null, validated to R `plm::purtest` (and, for Fisher, statsmodels).
- **Nowcasting and mixed frequencies**: MIDAS (weights/U-MIDAS/weighted),
  DFM nowcasting (two-step and exact one-step MLE) with a ragged edge and the
  Bańbura-Modugno news decomposition.
- **Term structure**: Nelson-Siegel, Svensson, dynamic Nelson-Siegel, and the
  arbitrage-free (AFNS) yield adjustment of Christensen-Diebold-Rudebusch.
- **Applied-macro extensions**: recession-probability models (static and
  Kauppi-Saikkonen dynamic probit/logit); survey-expectations tools
  (Coibion-Gorodnichenko, Mincer-Zarnowitz, disagreement); the specification
  and stability battery (White, Koenker-Breusch-Pagan, RESET, Chow, CUSUM);
  and a linear rational-expectations (DSGE-lite) solver via Blanchard-Kahn.
- **Python layer**: maturin mixed layout (`tsecon._core` + a pure-Python
  package); the opt-in `tsecon.results` rendering layer — `dict` subclasses
  with `.summary()`/`.plot_*()` that preserve the plain-dict contract;
  complete type stubs with `py.typed`.
- **Configurable inference**: a uniform `se_type=` on regression estimators;
  configurable interval coverage; cumulative IRF views.
- **Evidence beyond fixtures**: a seeded Monte Carlo validation suite (size /
  coverage / consistency) and frontier experiments (LP vs VAR; weak-IV LP-IV);
  a 25-case cross-library parity benchmark harness (statsmodels, SciPy,
  scikit-learn, `arch`) that gates CI; two replications of published results —
  Ramey-Zubairy (2018) government-spending multipliers and Estrella-Mishkin
  (1998) yield-curve recession prediction — running offline from committed
  public data; Rust and Python coverage tooling.
- **Docs**: a 15-chapter teaching guide, model cards for every estimator
  family, a generated API reference with a drift guard, a validation matrix,
  a testing-and-validation map, migration guides (statsmodels/R/Stata) with a
  Rosetta glossary, a worked figure gallery, and an interactive demo.
- **Packaging**: complete `pyproject.toml` metadata, abi3-py39 wheels tested
  on Python 3.9 and 3.13 in CI, GitHub Actions CI (Rust gates, a three-OS
  wheel matrix, a mypy stub check, and an evidence job running the Monte Carlo
  suites and the parity gate), and a tag-triggered release pipeline with PyPI
  trusted publishing.

### Removed
- **Data-fetching loaders** (`tsecon.datasets`): built, then deliberately
  removed before release. A library that hardcodes external data URLs owns
  their breakage (the widely-cited FRED-MD URL had already moved), so tsecon
  ships no network code — the only runtime dependency is NumPy, and the
  replications run on small public datasets committed to the repository.

Every estimator is validated against a reference implementation (statsmodels,
SciPy, NumPy, `arch`, `linearmodels`, scikit-learn, ArviZ) or a documented
closed form in the test suite.
