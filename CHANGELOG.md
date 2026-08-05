# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/); versioning follows the
pre-1.0 policy in [ROADMAP.md](ROADMAP.md) (minor = breaking allowed, patch =
fixes) until 1.0, then strict [SemVer](https://semver.org/).

## [Unreleased]

Nothing yet.

## [0.2.0] - 2026-08-05

An interval-coverage audit took the library's interval-producing surfaces,
pointed each at its own nominal level, and measured what fraction of intervals
actually covered. It found four real defects. This release fixes them. **One
fix is breaking**: it changes numbers callers were already getting, and
changing those numbers is the point of the release.

### Changed — BREAKING
- **`iv_gmm(weight="hac")` was a silent no-op, and its standard errors will
  move.** `bandwidth` defaulted to `0.0`, and a Bartlett kernel truncated at
  zero lags *is* the White estimator — so `weight="hac"` returned results
  bit-identical to `weight="robust"` (max |Δ se| = 0.000e+00 over 3000
  replications) while the caller believed they had bought serial-correlation
  robustness. `bandwidth` now defaults to `None`, which selects the
  Newey-West rule of thumb `floor(4 * (n/100)^(2/9))`; an **explicit**
  `bandwidth=0.0` now raises instead of silently degrading to White; and the
  truncation actually used comes back as `hac_bandwidth`. On an AR(1)-error
  design at n = 250 the slope standard error moves from 0.10522 (`robust`, and
  the old `hac`) to 0.09393 (`hac`, automatic bandwidth 4) or 0.09228 (`hac`,
  `bandwidth=10`). If you passed `weight="hac"` before, you were reporting
  White standard errors under a HAC label and your numbers will change.
  **This does not restore coverage.** The audit measured 0.868 ± 0.006 against
  a nominal 0.95 at `bandwidth=10`, and the automatic rule picks *fewer* lags
  (4 at T = 250). A defensible default is not a remedy: those intervals are
  still too narrow, and the fix is that you can now see and set the bandwidth
  rather than that the bandwidth is now right.
- **`iv_gmm(method="2sls", weight="hac")` now raises.** 2SLS fixes its weight
  matrix at `(Z'Z/n)^-1` by construction, so accepting a weight argument there
  was the same silent no-op in a second place.

### Added — inference
- **`ols(se_type="hc2")` and `ols(se_type="hc3")`** — the leverage-corrected
  heteroskedasticity-robust standard errors, matched to statsmodels HC2/HC3 to
  2.96e-15 on the audit's own T = 25 chi2(1)-regressor high-leverage design. On
  that design tsecon's own slope standard error runs 0.1749746 (nonrobust),
  0.1751173 (hc0), 0.1825724 (hc1), 0.2095910 (hc2), 0.2629148 (hc3): HC3 is
  44% larger than HC1, which is exactly the leverage correction that hc1's
  `n/(n-k)` factor does not buy. An observation whose leverage is numerically
  equal to 1 is refused, not returned as a near-infinite standard error.
- **`iv_gmm` returns `first_stage`** — a list of dicts with keys `regressor`,
  `fstat`, `dof_num`, `dof_den`, `pval`: a heteroskedasticity-robust
  per-regressor first-stage F. Entries are **omitted** where the statistic is
  undefined (an exogenous regressor, no excluded instruments, a regressor the
  instruments reproduce exactly, rank-deficient `Z`, a non-finite statistic),
  so the list can be shorter than the regressor count and must be indexed by
  `regressor`, never by position; a missing entry is not a failed fit. **With
  two or more endogenous regressors this is not a weak-identification test** —
  every regressor can clear 10 while the system is under-identified, because
  the instruments may predict only one common combination of them. The right
  objects are Angrist-Pischke (per regressor) and Cragg-Donald /
  Kleibergen-Paap against Stock-Yogo (joint), and none of those are
  implemented. Even with a single endogenous regressor, F > 10 is not a safety
  threshold: the audit measured 0.915 coverage at a median first-stage F of
  10.5.
- **`arima_fit(drift_uncertainty=True)`** — with `d >= 1` and `constant=True`
  the h-step forecast contains an estimated drift whose uncertainty grows like
  h², and the default omits it entirely (the reported se is exactly
  `sigma*sqrt(h)`). Opt in and the se matches the closed form
  `sigma*sqrt(h + h²/(T-1))` to 5.22e-09. It is **opt-in** so the default path
  stays bit-identical and keeps matching the statsmodels `get_forecast` golden
  at 1e-6: the two are different estimands, not a right one and a wrong one.
- **`arima_fit` returns `bse`, `param_cov`, and `cov_ok`** — parameter standard
  errors from the observed information (statsmodels `cov_type="approx"`).
  ARIMA previously reported no parameter standard errors at all. `bse` and
  `param_cov` are `None` with `cov_ok=False` when that matrix is too
  ill-conditioned to invert honestly, rather than reporting a number the
  numerics do not support.

### Added — ergonomics
- **Forgiving input**: every estimator now accepts a pandas `DataFrame`/`Series`
  (or any `.to_numpy` array-like), off-dtype/non-contiguous float arrays,
  **integer and boolean arrays** (data read as `int`, a 0/1 dummy, a `y > 0`
  mask), and a **plain list of numbers**. All are converted to `float64` at the
  boundary instead of raising. Coercion is *parameter-aware*, so it never
  touches an argument that is not data: the four audited integer label/index
  parameters (`hetero_svar.regime_labels`, `var_granger.caused`/`causing`,
  `favar.slow_indices`), restriction-tuple specs, tuple-valued options, and
  callables pass through untouched, and ragged panel lists keep their container
  while each per-unit array is converted. A *nested* Python list is
  deliberately left alone — `[(0, 1), (0, 2)]` is a restriction spec and
  `[[1.0, 2.0], [3.0, 4.0]]` is a matrix, and the values are indistinguishable.
- **Errors that teach**: the messages a first run is most likely to hit now say
  what happened, why, and what to try, with the offending numbers included. For
  example, too few observations for the requested lags now reports the required
  row count and the degrees-of-freedom arithmetic behind it, and suggests a
  concrete smaller `lags`. A wrong-rank array argument reports the shapes it
  received instead of the low-level `'ndarray' object is not an instance of
  'ndarray'`.
- **`tsecon.summarize(result)`**: a uniform, opt-in renderer for *any* function's
  output. Plain dicts get a generic aligned `.summary()`; the six bespoke
  `tsecon.results` objects pass through unchanged. Still a `dict` subclass, so
  the plain-data contract is preserved.
- **Cookbook**: short single-task recipes under `docs/cookbook/`, each a
  self-contained page with executed output.

### Added — estimators
- `ndiffs` — how many differences a series needs, with the per-order test
  evidence rather than just the integer.
- `box_cox_lambda` — variance-stabilising Box-Cox lambda (MLE, matched to
  `scipy.stats.boxcox_normmax`; Guerrero as a documented-formula alternative).
- `engle_granger` — the two-step cointegration test now returns p-values and
  critical values from the MacKinnon response surfaces, not just the statistic.

### Fixed

The four defects the interval-coverage audit found, with measured coverage
before and after (nominal 0.95 throughout):

- **`ols` robust intervals under-covered badly under high leverage.** On the
  T = 25 chi2(1) design: **hc1 0.682 → hc3 0.863**. Fixed by adding hc2/hc3.
  hc1 is still available and is still the wrong choice on that design; hc3 does
  not reach nominal there either, and the honest reading is that a T = 25
  tail-heavy design is hard, not that hc3 solves it.
- **`arima_fit` forecast intervals ignored drift uncertainty.** With `d >= 1`
  and `constant=True`, at h = 24 and T = 60: **0.902 → 0.945** with
  `drift_uncertainty=True`.
- **`iv_gmm(weight="hac")` never applied a HAC correction** (bit-identical to
  `weight="robust"`, max |Δ se| = 0.000e+00 over 3000 replications). See the
  breaking note above. **Not fixed by this release:** coverage on the audited
  AR(1)-error design is 0.868 at `bandwidth=10`, and the new automatic default
  picks fewer lags than that.
- **`iv_gmm` reported no first-stage evidence at all.** Now returns
  `first_stage`, with the caveats above — this is a diagnostic, not a
  weak-identification test.

### Not in this release

Named so they are not read into the above: SARIMA seasonal orders `(P, D, Q, s)`;
Anderson-Rubin and other weak-IV-robust confidence sets for `iv_gmm`;
Angrist-Pischke, Cragg-Donald, and Kleibergen-Paap statistics; and simultaneous
(joint) bands anywhere — every band in the library is pointwise.

## [0.1.0] - 2026-07-23

First tagged release, published to PyPI as `tsecon`. Pre-1.0 and under active
development: minor versions may make breaking changes, patch versions are
fixes, until 1.0.

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
- **Statistical (non-Gaussian) SVAR identification**: `nongaussian_svar` —
  independent-component / ICA identification (Lanne-Meitz-Saikkonen 2017;
  Gouriéroux-Monfort-Renne 2017) that point-identifies the *whole* structural
  impact matrix from the reduced-form residuals alone — no sign, zero, long-run,
  proxy, or variance-regime restriction — by exploiting the mutual independence
  and non-Gaussianity of the shocks. A deterministic symmetric FastICA fixed
  point (Hyvärinen log-cosh contrast, identity-initialized — bit-reproducible, no
  RNG) rotates the whitened residuals to maximal non-Gaussianity and returns
  `B = Σ_u^{1/2} Q`; column sign and order are conventions, and the scheme
  **fails under Gaussianity** — a boundary the returned `shock_kurtosis`
  diagnostic flags (near zero ⇒ weakly identified). Validated by an independent
  NumPy FastICA golden (itself cross-checked to `sklearn.decomposition.FastICA`)
  for the core, plus seeded Monte-Carlo recovery of the true `B` up to sign and
  permutation.
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
