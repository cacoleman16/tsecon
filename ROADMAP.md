# Roadmap — A High-Performance Time Series Econometrics Library

**A Rust-core, Python-first library that unifies the full suite of time series econometrics — classical, Bayesian, structural, and machine-learning — with speed built for Monte Carlo work, accuracy validated against the published literature, and documentation that teaches which model to use when.**

This is the master plan. Each module has a detailed specification in [docs/roadmap/](docs/roadmap/), produced from an exhaustive research pass over the field (12 domain inventories totaling ~830 methods, plus an adversarial completeness review). Start here; drill into the module docs for the full inventories, implementation notes, numerical traps, and validation targets.

---

## 0. Current build status

*Snapshot as of the latest commit. The plan below is the destination; this is where the code actually is.*

**Phases 0–1 complete; Phases 2–4 substantially landed.** 41 Rust crates implemented (~52,000 source lines), **1001 Rust tests + 491 Python tests, all green and golden-fixture-gated** against statsmodels, SciPy, NumPy, `arch`, `linearmodels`, `scikit-learn`, `ArviZ`, and `scipy.signal`. Everything installs and runs from Python today. The repository builds from a clean clone (CI-verified), with `cargo fmt`/`clippy`/`test` and a build-and-test-the-wheel matrix (Linux/macOS/Windows) on every push, plus a tag-triggered PyPI release pipeline with trusted publishing. A **hosted docs site** (mkdocs-material: quickstart, the "which model when" guide, model cards for every estimator, an API reference, migration guides, and the figure gallery) builds under `mkdocs build --strict`. The tree is **release-audited**: no secrets in history, a 100%-permissive dependency tree (zero copyleft), full dual MIT/Apache license texts shipped in the wheel, and a third-party-license inventory.

**Built and callable from Python now** (121 functions): the diagnostic battery (ACF/PACF, Ljung-Box, Jarque-Bera, ARCH-LM), the full unit-root workflow (ADF, KPSS, `check_stationarity`, plus `phillips_perron` and the `phillips_ouliaris` residual cointegration test), and the one-call **`check_series` diagnostic battery with model recommendations** (the Module 01 flagship — evidence in families, the multiple-testing arithmetic shown, every suggestion routing to shipped functions); robust standard errors (HAC/Newey-West with a uniform `se_type=`); the bootstrap family; the exact-diffuse Kalman filter/smoother; **ARIMA** (exact MLE + forecasting), **GARCH/GJR/EGARCH** (QMLE + robust SEs), and **GAS/DCS** score-driven volatility (`gas_volatility` — Gaussian & Student-t, Creal-Koopman-Lucas 2013); **VAR** (IRF/FEVD/Granger/forecast, cumulative IRF views, and `var_irf_bands` — frequentist asymptotic/bootstrap IRF confidence bands); trend-cycle filters; forecast evaluation (Diebold-Mariano, Clark-West, Giacomini-White, Theta, accuracy measures) and the **rolling/expanding backtest engine** (`backtest`); a **Bayesian VAR** — the Minnesota-NIW conjugate prior with posterior IRF draws and convergence diagnostics, plus `bvar_hierarchical` (Giannone-Lenza-Primiceri 2015 empirical-Bayes ML selection of the shrinkage hyperparameters); the dedicated **local projections** module (`lp`/`lp_iv`/`lp_multiplier`/`lp_state`/`smooth_lp` — lag-augmented default, LP-IV, the Ramey-Zubairy integral fiscal multiplier, state-dependent regimes, and Barnichon-Brownlees smooth LP); **penalized regression** (ridge/lasso/elastic-net, `adaptive_lasso`, the AIC/BIC-selected `lasso_path`) with **leakage-safe time-series CV** (`cv_splits`); a full **structural-identification suite** — sign-restricted Bayesian SVARs (`sign_restricted_svar`, the headline differentiator), long-run/Blanchard-Quah (`long_run_svar`), max-share/max-FEV news shocks (`max_share_svar` — Uhlig 2004/Barsky-Sims), proxy/external-instrument SVAR-IV (`proxy_svar` — Stock-Watson/Mertens-Ravn/Gertler-Karadi), and identification-through-heteroskedasticity (`hetero_svar` — Rigobon 2003), zero+sign restrictions (`zero_sign_svar` — RWZ 2010), narrative sign restrictions (`narrative_svar` — Antolin-Diaz-Rubio-Ramirez 2018), the Fry-Pagan median-target rotation (`fry_pagan_svar`), Giacomini-Kitagawa prior-robust bounds (`robust_svar_bounds`), general-impact structural FEVD (`structural_fevd`), and historical decomposition (`historical_decomposition`); **GMM** (`iv_gmm` — 2SLS/two-step/iterated with the Hansen J-test, and `gmm_nonlinear` — custom moment conditions via a Python callback); **predictive regressions** (`predictive_regression`/`ivx_test` — Stambaugh bias correction and the persistence-robust IVX Wald test); **panel** methods (`panel_fe`/`panel_lp`, the `mean_group_var` mean-group panel VAR, and `panel_mean_group`/`panel_pmg` — the Pesaran-Smith MG, Pesaran CCE-MG, and PMG trio); **spectral analysis** (`periodogram`/`welch`/`coherence`); **cointegration** (`johansen`/`vecm`); **Markov-switching AR** (`markov_switching_ar`); **MIDAS** mixed-frequency regressions (`midas_weights`/`umidas`/`weighted_midas`) and **DFM nowcasting** (`dfm_nowcast` — two-step Doz-Giannone-Reichlin *or* the exact one-step Gaussian MLE, with a ragged edge, plus `dfm_news` — the Bańbura-Modugno news decomposition); **multivariate GARCH** (`ccc_garch`/`dcc_garch`); **realized volatility** (`realized_measures`, `har_rv`, `realized_quarticity`, `tripower_quarticity`, `bns_jump_test`, `realized_range`); **Diebold-Yilmaz connectedness** (`connectedness`); the **PCA factor model** (`factor_model`) and the two-step **FAVAR** (`favar`); the **Nelson-Siegel/Svensson** yield curve (`nelson_siegel`/`svensson`/`dynamic_ns`); **recession-probability** models (`recession_probit` — static & dynamic probit/logit); the **survey-expectations** toolkit (`cg_regression`/`forecast_efficiency` — Coibion-Gorodnichenko & Mincer-Zarnowitz); **long-memory** tools (`frac_diff`/`long_memory_d` — fractional differencing, GPH, local Whittle); the **specification & diagnostic tests** battery (`heteroskedasticity_test` — White & Koenker-Breusch-Pagan, `reset_test`, `chow_test`, `cusum_test` — Brown-Durbin-Evans); the **arbitrage-free Nelson-Siegel** yield adjustment (`afns_adjustment` — Christensen-Diebold-Rudebusch 2011); a **linear rational-expectations (DSGE-lite) solver** (`dsge_solve` — the Blanchard-Kahn saddle-path solution); **quantile regression, quantile local projections, and growth-at-risk** (`quantile_regression`/`quantile_lp`/`growth_at_risk` — Koenker-Bassett by IRLS with Powell sandwich SEs and the Adrian-Boyarchenko-Giannone GaR workflow, with monotone quantile rearrangement); **functional shocks** (`functional_pca`/`flp`/`flp_scenario`/`fvar_scenario` — FPCA of curve-valued shocks with functional local projections and FVAR scenario responses, in the spirit of Inoue-Rossi 2021); **multiple structural breaks** (`bai_perron` — the Bai-Perron dynamic-programming estimator with information-criterion selection and break-date confidence intervals — plus `sup_f_test` — the Andrews sup-F with Hansen 1997 p-values); and **smooth local projections** (`smooth_lp` — Barnichon-Brownlees penalized B-spline LP).

**The Python layer** (maturin mixed layout — the compiled extension installs as the private `tsecon._core`, with the `tsecon` package re-exporting it so pure-Python submodules can sit beside the Rust core): **`tsecon.results`** — an opt-in rendering layer of `dict` *subclasses*, so `res["params"]`, `json`, and `pickle` keep working while `.summary()` and `.plot_*()` become available (VAR/LP/GARCH/ARIMA/predictive-regression/DSGE; matplotlib is an optional `plots` extra, lazily imported). The library ships **no data loaders and makes no network calls** — `import tsecon` depends only on NumPy — a deliberate boundary: data acquisition against external URLs is a maintenance liability better left to specialist tools, so tsecon takes plain arrays and does the econometrics. Real-data workflows in the gallery run on small public datasets committed to the repo.

**Built in Rust, awaiting Python bindings** (validated, committed): the Haar rotation kernel and non-Bayesian identification internals (Module 06, exposed through `sign_restricted_svar`); the `smooth_fixed` fixed-parameter DFM state-space entry point (Module 08, used internally by `dfm_nowcast`). The built-but-unbound backlog is otherwise drained.

**Documentation shipped**: a 15-chapter [teaching guide](docs/guide/README.md) (beginner to research-grade, including nonlinear dynamics), a worked figure [gallery](docs/examples/README.md) (base, advanced, structural, depth, and roadmap-extension sections), model cards for every estimator family, a [validation matrix](docs/reference/validation-matrix.md) naming what each family is checked against, this roadmap's 15 module specs, and an interactive demo. **Evidence beyond fixtures**: a seeded [Monte Carlo validation suite](docs/examples/monte-carlo.md) for the statistical properties a golden match cannot prove (IVX holds its 5% size at an exact unit root where the naive OLS t-test rejects 27.8%), [frontier experiments](docs/examples/monte-carlo-frontier.md) on the comparative questions (LP vs VAR bias/variance; what a weak instrument actually breaks), a parity-first [benchmark harness](benchmarks/) that verifies cross-library estimate agreement *before* timing anything, and two replications of published results — [Ramey-Zubairy (2018)](docs/examples/replication-ramey-zubairy.md) government-spending multipliers (0.64-0.74 vs their published 0.6-0.8) and [Estrella-Mishkin (1998)](docs/examples/replication-yield-curve-recession.md) yield-curve recession prediction (a term-spread probit coefficient of -0.58, z=-9.6). Community files (CONTRIBUTING, CODE_OF_CONDUCT, GOVERNANCE, CITATION.cff) and a JOSS paper draft are in place.

**Extensions landed** (beyond the original brief, §10): panel time series — mean-group / CCE-MG / PMG (E1); the Nelson-Siegel term-structure family and its **arbitrage-free (AFNS)** yield adjustment — Christensen-Diebold-Rudebusch (E2); predictive regressions & IVX (E3); GMM / IV-GMM (E4); the **linear rational-expectations DSGE-lite solver** — Blanchard-Kahn saddle path (E5); **survey expectations** — Coibion-Gorodnichenko & Mincer-Zarnowitz (E6); **fractional integration / long memory** — fractional differencing, GPH, local Whittle (E7); **recession-probability models** — static & Kauppi-Saikkonen dynamic probit/logit (E8); and the **specification & diagnostic tests** battery — White, Koenker-Breusch-Pagan, Ramsey RESET, Chow, Brown-Durbin-Evans CUSUM (E9). A **frontier slice** followed (July 2026): **quantile regression & growth-at-risk**, **functional shocks (FVAR/FLP)** — the first E10 (functional time series) landing — **Bai-Perron multiple structural breaks**, and **smooth local projections**.

**Next up**: the remaining §9 adoption polish (head-to-head explainer chapters, a public speed dashboard, more published replications). Data loaders were built and then **deliberately removed** — a library that hardcodes external data URLs owns their breakage (FRED had already moved the canonical FRED-MD file), so tsecon stays data-agnostic and the replications run on committed public datasets instead. The **library name is resolved: `tsecon` stays** — verified available on PyPI, so the codename is now the name. With the DSGE-lite solver (E5), the AFNS arbitrage-free term structure (extending E2), and the specification-test battery (E9) now landed and bound, the remaining tiered work is the richer affine term-structure models (JSZ/ACM) and E11–E12 (climate econometrics, extremal dependence), deliberately future work rather than v1 gaps — E10 (functional time series) now has its first landing in the functional-shock (FVAR/FLP) module. The DFM nowcaster (two-step *and* one-step MLE), its news/update decomposition, and the docs/UX/adoption layer have all landed. GAS score-driven volatility, DFM nowcasting + news decomposition, the panel trio (mean-group / CCE-MG / **PMG**), IV-GMM + nonlinear GMM, **predictive regressions & IVX (E3)** — validated by a Monte-Carlo size test that holds nominal across ρ∈{0.9,…,1.0} including the unit root — realized-volatility measures, FAVAR, and Diebold-Yilmaz connectedness are all now built and bound.

**Validation-first paid off repeatedly**: the discipline surfaced three genuine defects in *reference* implementations (a duplicated-SE column and an early optimizer stop in `arch`, a non-converged `statsmodels` fit) and two fixture-precision ceilings (panel, forecast-eval) that were fixed at the source — plus a repo-breaking `.gitignore` that had silently excluded a whole module, caught only because CI tests a fresh checkout.

---

## 1. Vision and the wedge

Time series econometrics has no centralized home. The situation today:

- **statsmodels.tsa** is the closest thing Python has — but it is slow for simulation work, has stagnated on everything structural, and offers no SVAR identification beyond Cholesky, no local projections, no BVARs, no nowcasting, no mixed frequency.
- **The macro toolkit lives in fragments**: R's `vars`/`svars`/`lpirfs`/`BVAR`/`midasr`, Matlab replication zips passed hand-to-hand (BEAR, the VAR Toolbox, JSZ/ACM term-structure code), Dynare for Bayesian estimation. None of it interoperates; much of it is unmaintained; almost none of it is fast.
- **The fast Python libraries** (Nixtla's statsforecast, darts) are forecasting-competition tools — they have no notion of identification, inference, or structural analysis, which is most of what economists actually do.

**The wedge**: modern empirical macroeconomics — SVAR identification, local projections, Bayesian VARs, nowcasting — has *no maintained, fast, unified Python home*. That is the library's beachhead. Around it we build the full field: diagnostics, univariate models, volatility, forecasting evaluation, and econometrics-first machine learning, all on one shared numerical core.

**Why a low-level core matters**: bootstrap inference, sign-restriction rotation sampling, posterior simulation, particle filtering, and Monte Carlo studies are embarrassingly parallel and compute-bound. A compiled, multithreaded core with reproducible parallel RNG turns hours into seconds — and changes what users attempt (5,000-draw wild bootstraps as the *default*, not a luxury).

## 2. Guiding principles

1. **Correctness is gated, not aspirational.** No estimator merges without a named golden validation target: a published table, a reference implementation, or both. The replication gallery *is* the integration test suite.
2. **Speed is a feature users can feel.** Rayon-parallel Monte Carlo/bootstrap/rotation sampling everywhere; published benchmark suite; deterministic results at any thread count.
3. **The API is one grammar.** Every model: `Spec → fit()/sample() → Results`, with rich results objects (`.summary()`, `.irf()`, `.forecast()`, publication-ready output). Learn one model, know them all.
4. **Documentation teaches judgment, not just syntax.** The flagship deliverable is the "which model when" guide and per-estimator model cards (assumptions, failure modes, alternatives). Error messages teach.
5. **Scope discipline is survival.** ~830 candidate methods were inventoried; abandoned-library post-mortems (Kats, PyFlux) show exactly how feature sprawl kills. Tiering is public, v1 surface area is capped, and frontier items are gated on validation targets.
6. **Sensible loud defaults.** The statistically-recommended choice is the default (e.g., lag-augmented inference for local projections per Montiel Olea & Plagborg-Møller 2021), and deviations from older conventions are flagged in output, with references.
7. **Robust inference is a first-class option everywhere.** Every regression-based estimator exposes a uniform `se_type=` argument — nonrobust, HC0–HC3, HAC (Newey-West/Andrews/QS with automatic bandwidth), EWC fixed-b (the LLSW 2018 recommendation), cluster/Driscoll-Kraay where panels apply, and bootstrap (wild/block via the shared resampling engine) — with the method always stamped on the reported intervals. One HAC implementation serves the whole library, so identical settings can never produce different p-values in different modules.

## 3. Architecture at a glance

Full detail: [docs/roadmap/00-architecture.md](docs/roadmap/00-architecture.md)

**Language verdict: Rust** (over Fortran and C++), bound to Python with **PyO3 + maturin**.

- *Why Rust*: memory/thread safety for aggressive parallelism (rayon), first-class wheel tooling (maturin builds abi3 wheels for every platform in CI), a healthy contributor pipeline, cargo's dependency story, and a maturing numerics stack. Fortran still wins raw-kernel ergonomics for dense array math, but loses badly on distribution (f2py/meson wheel pain), safety in parallel code, and the contributor pool — a community library lives and dies on contributors.
- *Linear algebra*: **faer** (pure-Rust, competitive with MKL for the relevant sizes) as primary backend — pure-Rust matters because static, BLAS-free wheels install anywhere, including pyodide/WASM eventually.
- *Accepted infrastructure dependencies* (pure-Rust, static-wheel-preserving, in the same category as faer — foundational numerical primitives that are not econometrics and not worth reimplementing): **faer** (dense linear algebra), **rayon** (parallelism), and **rustfft** (the FFT, used by the spectral module). Everything statistical and econometric is implemented from scratch; the runtime footprint stays tiny (numpy is the only Python dependency).
- *RNG*: counter-based **Philox**, bit-compatible with NumPy's, giving reproducible parallel streams — every bootstrap/MCMC/rotation draw is replayable at any thread count, and common random numbers for SMM come free.
- *The shared state-space engine* is the single most load-bearing component: one linear-Gaussian SSM implementation (exact diffuse initialization, univariate treatment of multivariate observations, Durbin-Koopman and precision-based simulation smoothers as interchangeable backends, EM) serves ARIMA, ETS, unobserved components, TVP, DFM, MF-VAR, nowcasting, and Bayesian state sampling.
- *Data layer*: Arrow-native; pandas, polars, and NumPy all first-class in and out.
- *Covariates everywhere*: one exogenous-regressor contract shared by every model family (ARIMAX/regARIMA, VARX, GARCH-X, MIDAS, LP controls, factor-augmented regressions, ML forecasters): index-aligned ingestion with explicit alignment diagnostics, deterministic-terms builders (trends, seasonal dummies, Fourier terms, holidays, interventions), and — the part most libraries fumble — an explicit future-covariate interface for forecasting that distinguishes known-future values (calendars, policy paths), scenario paths (fan out forecasts over user-supplied paths), and auxiliary-model forecasts (with uncertainty propagated), plus leakage checks in the backtesting engine so future covariate information can never silently enter a pseudo-out-of-sample exercise.
- *Optimization*: in-house quasi-Newton suite (BFGS/L-BFGS-B, strong-Wolfe), Nelder-Mead, trust-region; constrained-domain reparameterization toolkit (stationarity/invertibility/positivity); analytic gradients first, complex-step and dual-number AD as fallbacks.
- *Build-vs-buy*: wrap the Census **X-13ARIMA-SEATS** binary (never reimplement); embed **nuts-rs** for HMC/NUTS; wrap XGBoost/LightGBM behind adapters outside the core wheel.

Crate layout: a cargo workspace with domain crates (`core-ssm`, `core-rng`, `core-linalg`, `models-arima`, `models-garch`, `models-var`, ...) and one thin `python-bindings` crate; the Python package is a facade organized by task, not by crate.

## 4. Module map

| # | Module | One-line scope | Spec |
|---|--------|----------------|------|
| 00 | **Systems architecture** | Language, numerics, FFI, RNG, SSM engine, packaging, testing policy | [00-architecture.md](docs/roadmap/00-architecture.md) |
| 01 | **Diagnostics & exploration** | ACF/PACF, unit roots, seasonality, breaks, filters, spectral, X-13, `check_series()` | [01-diagnostics-exploration.md](docs/roadmap/01-diagnostics-exploration.md) |
| 02 | **Univariate models** | ARIMA/SARIMA family, ETS, unobserved components, TVP, Markov-switching, threshold/STAR, long memory | [02-univariate.md](docs/roadmap/02-univariate.md) |
| 03 | **Volatility & risk** | GARCH zoo, stochastic volatility, multivariate GARCH, realized measures, VaR/ES + backtests | [03-volatility.md](docs/roadmap/03-volatility.md) |
| 04 | **Multivariate models** | VAR/VARX/VARMA, VECM/cointegration, factor models (the DFM home), nonlinear VARs, connectedness | [04-multivariate.md](docs/roadmap/04-multivariate.md) |
| 05 | **Bayesian time series** | BVAR priors (Minnesota → GLP hierarchical → shrinkage), large BVARs, SV, TVP, samplers, marginal likelihoods | [05-bayesian.md](docs/roadmap/05-bayesian.md) |
| 06 | **Structural identification** | Cholesky, long-run, sign/zero+sign, narrative, max-share, heteroskedasticity, non-Gaussianity, proxy/internal IV, set-ID robust Bayes | [06-identification.md](docs/roadmap/06-identification.md) |
| 07 | **Local projections** | LP, LP-IV, lag-augmented inference, smooth/state-dependent/panel LP, LP-DiD, LP-vs-VAR dual reporting | [07-local-projections.md](docs/roadmap/07-local-projections.md) |
| 08 | **Nowcasting & mixed frequency** | MIDAS family, DFM nowcasting facade, MF-VAR/MF-BVAR, vintages, ragged edges, news decomposition | [08-nowcasting-mixed-frequency.md](docs/roadmap/08-nowcasting-mixed-frequency.md) |
| 09 | **Forecasting & evaluation** | Forecast objects, backtesting engine, accuracy/comparison tests, density evaluation, combination, conformal, reconciliation | [09-forecasting-evaluation.md](docs/roadmap/09-forecasting-evaluation.md) |
| 10 | **Machine learning** | Penalized regression, regularized VAR solvers, trees/boosting, TS cross-validation, DML, GP-SSM, interpretation | [10-machine-learning.md](docs/roadmap/10-machine-learning.md) |
| 11 | **Docs, UX & adoption** | Competitive audits, Diátaxis docs, model cards, replication gallery, datasets, migration guides, naming, governance | [11-docs-ux-adoption.md](docs/roadmap/11-docs-ux-adoption.md) |
| 12 | **Extensions (beyond the brief)** | Panel TS, term structure, predictive regressions, GMM/SMM, DSGE-lite, survey expectations, and more | [12-extensions.md](docs/roadmap/12-extensions.md) |
| 13 | **Visualization** | Publication-ready-by-default figures: style system, IRF panel grids, fan charts, diagnostic dashboards, visual regression CI | [13-visualization.md](docs/roadmap/13-visualization.md) |
| 14 | **Packaging & distribution** | The PyPI release pipeline: cross-platform abi3 wheels, sdist, type stubs, trusted publishing, conda-forge, provenance | [14-packaging-distribution.md](docs/roadmap/14-packaging-distribution.md) |

Headline differentiators — the modules nothing else offers as a maintained, unified stack: **06 (identification), 07 (local projections), 05 (Bayesian), 08 (nowcasting)**. They are also the modules with the deepest cross-dependencies, which is why the shared infrastructure comes first.

## 5. Shared infrastructure and the ownership map

The completeness review found ~80–100 items claimed by two or more domains — six subtly different wild bootstraps is how a library ships contradictory p-values. Every overlapping capability gets exactly one owner; everyone else consumes.

| Capability | Owner | Principal consumers |
|---|---|---|
| Resampling/bootstrap engine (iid, wild, block, stationary, sieve, subsampling, grid bootstrap; block-length selection; RNG substream contract) | foundations (00) | every module |
| HAC / long-run variance / fixed-b / EWC inference — one library-wide default policy | foundations (00) | 01, 04, 07, 09, E4 |
| Linear-Gaussian state-space engine + simulation smoothers + EM | foundations (00) | 02, 04, 05, 08, E2, E5 |
| Typed IRF object + generalized-IRF (KPP) engine — incl. the cumulative view with correctly cumulated uncertainty (draw-wise cumulation for Bayesian/bootstrap; joint-covariance delta method for asymptotic; direct cumulative-outcome regression for LP) | foundations (00), GIRF contributed by 04 | 04, 05, 06, 07 |
| Factor-model estimation core (PCA/EM/QML, factor-number criteria) | foundations (00) | 04 (owner of DFM), 08, 10 |
| Quantile-regression solver (interior-point/ADMM, dependent-data inference, rearrangement) | foundations (00) | 03 (CAViaR), 04 (QVAR), 07 (quantile LP), 08 (MIDAS-QR), 09 |
| Critical-value engine (response surfaces + on-demand simulation, cached & versioned) | foundations (00) | 01, 02, 04 |
| Innovation-distribution zoo (t, GED, skew-t, Johnson SU, NIG, GH; densities/quantiles/samplers/score+CRPS hooks) | foundations (00) | 02, 03, 05, 09 |
| Time-index / calendar / frequency / holiday engine; vintage data store | foundations (00) | 01, 08, 09, 11 |
| Exogenous-regressor (covariate) contract: aligned ingestion, deterministic-terms builders, future-path/scenario interface for forecasting, leakage checks in backtesting | foundations (00) | every model module — 02 (ARIMAX/regARIMA, transfer functions), 03 (GARCH-X), 04 (VARX), 07 (LP controls), 08 (MIDAS, bridge equations), 10 (ML feature pipelines) |
| Temporal disaggregation & benchmarking (Chow-Lin, Denton, ...) | foundations (00) | 08 (primary API), 01, 02 |
| Haar-rotation / restriction-algebra kernel | foundations (00) | 05, 06 (one sign-restriction sampler, not two) |
| DFM implementation | 04 | 08 (nowcasting facade), 10 |
| Granger-causality tooling | 04 | 01 (re-export) |
| STL/MSTL; X-13ARIMA-SEATS wrapper | 01 | 02 |
| MIDAS weighting machinery | 08 | 03 (GARCH-MIDAS, DCC-MIDAS) |
| Forecast-comparison tests (DM/GW/MCS/SPA), density evaluation (PITs, scores), combination, **conformal prediction** | 09 | 03, 05, 08, 10 |
| Penalized-regression solvers + time-series cross-validation | 10 | 04 (regularized VARs), 08 (ML nowcasting) |
| Structural-VAR restrictions layer (frequentist + Bayesian backends) | 06 | 05 (supplies samplers/priors), 04 (supplies reduced form) |
| Likelihood & information-criterion conventions (one written contract) | foundations (00) | every module — otherwise AIC is incomparable across classes |
| Missing-data policy (one document; NaN-facing, SSM-filtering internally, per-estimator declared behavior) | foundations (00) | every module |
| Plotting: tidy-data contract (foundations) + the style system and chart catalog (Module 13 — publication-ready defaults, themes, export presets, visual regression CI) | foundations (00) + 13 | every results object's `.plot_*()` methods |

## 6. Scale and tiering policy

The research inventories total **831 methods before deduplication** (the completeness review estimates 80–100 cross-domain overlaps, resolved by the ownership map in §5). No credible v1 ships that. The tiering is public and enforced:

- **Tier 1 — Core**: v1-blocking. Has golden values in statsmodels/R/published tables. 279 items as inventoried — expect this to shrink meaningfully after deduplication and v1-cap triage; the Phase 1–2 gates in §7 define the true v1 surface.
- **Tier 2 — Standard**: expected of a serious library; ships in point releases after v1. 224 items.
- **Tier 3 — Advanced**: the differentiators; each needs a named validation target before work starts. 221 items.
- **Tier 4 — Frontier**: research-grade (2018–2026); gated on a paper table or reference implementation, often lives in `contrib` first. 107 items.
- **Contrib tier**: defensible individually, collectively how libraries drown (bilinear models, GARMA, echo-state networks, matrix profile, intermittent demand...). Lives outside the release-blocking path via the extension API.
- **Non-goals**: documented explicitly (see §12).

## 7. Phased build plan

Phases are dependency-ordered. Each has a **validation gate** — the phase is done when the gate passes, not when the code exists. Rough sizing assumes 2–3 experienced contributors; calendar estimates are honest guesses, not commitments.

### Phase 0 — Foundations (the bedrock) · ~4–6 months
Workspace scaffolding; faer-backed linalg + Sylvester/Lyapunov/Toeplitz solvers; Philox RNG (NumPy bit-compatible) + seeding API; PyO3/maturin skeleton with CI wheels on Linux/macOS/Windows; the SSM engine (KF/smoother, exact diffuse init, univariate filtering, both simulation smoothers, EM); optimizer suite + reparameterization toolkit; HAC/LRV module; bootstrap engine; distribution zoo; time-index/calendar engine; the covariate contract (aligned ingestion, deterministic-terms builders, future-path interface) — designed here so every model built in Phase 1 onward accepts exogenous regressors through the same API; forecast + IRF objects; golden-value harness; benchmark rig.
**Gate:** Kalman filter/smoother matches Durbin-Koopman worked examples and statsmodels to near machine precision; RNG streams bit-match NumPy Philox; abi3 wheels install clean on all three OSes; parallel bootstrap reproduces identically at 1 and N threads.

### Phase 1 — The classical core · ~6–9 months (parallelizable across 4 tracks)
- **Diagnostics track (01):** ACF/PACF/CCF with correct bands, portmanteau tests, full unit-root battery (ADF/PP/KPSS/DF-GLS/Ng-Perron/Zivot-Andrews/HEGY), cointegration tests, Bai-Perron, seasonality tests, STL/MSTL, HP/Hamilton/BK/CF filters, spectral basics, Box-Cox, `check_series()`.
- **Univariate track (02):** ARIMA/SARIMA/regARIMA exact MLE via SSM, CSS, auto-ARIMA, ETS taxonomy + AutoETS, unobserved components, TVP regression, missing-data handling, simulation engine.
- **Volatility track (03):** ARCH/GARCH/EGARCH/GJR/TGARCH + distributions, QMLE with robust SEs, forecasting, ARCH-LM/sign-bias diagnostics, VaR/ES + Kupiec/Christoffersen backtests, HAR-RV.
- **Multivariate track (04):** VAR estimation/lag selection/stability, Granger causality, Cholesky IRF/FEVD with delta-method + bootstrap bands (incl. Kilian bootstrap-after-bootstrap), VARX, Johansen VECM, forecasting.
- **Forecasting track (09):** forecast objects, fixed/rolling/expanding backtesting engine, core accuracy measures, DM test, benchmark zoo (naive/seasonal-naive/drift/Theta).
- **Visualization track (13):** the style system and themes, time series line plot with recession shading, ACF/PACF panels, fan charts, the residual diagnostic dashboard, export presets, and the visual regression harness — so every Phase 1 model ships with publication-ready figures from day one.
**Gate:** the parity battery — airline-model ARIMA matches R `arima()` to 6+ digits; GARCH(1,1) matches the `arch` package on S&P data; Johansen matches `urca`; unit-root p-values match published response surfaces; Lütkepohl (2005) textbook VAR examples reproduce exactly; benchmark suite shows the Monte Carlo speed story publicly.

### Phase 2 — Structural macro (the differentiators) · ~6–9 months
- **Visualization depth (13):** the IRF panel grid (shared with LP dual reporting), FEVD/historical-decomposition charts, regime plots — the structural figures land with the structural models.
- **Identification (06):** the unified restrictions layer — recursive, non-recursive A/B, Blanchard-Quah long-run, max-share, sign (Uhlig/RWZ), zero+sign (ARW 2018), narrative (AR 2018), proxy-SVAR with weak-IV-robust inference (MSW), internal instruments, heteroskedasticity (Rigobon, Markov-switching, GARCH-based), non-Gaussianity/ICA; identification diagnostics; the "which scheme when" guide.
- **Local projections (07):** LP/LP-IV with lag-augmented default inference, HAC/EWC options, sup-t bands, smooth LP, state-dependent LP, panel LP, cumulative multipliers, LP-DiD, LP-vs-VAR dual reporting.
- **Bayesian (05):** Minnesota/NIW/dummy-observation priors, GLP hierarchical hyperpriors, large BVARs, common SV and TVP-BVAR-SV (with the Del Negro-Primiceri correction), steady-state BVAR, conditional forecasts, marginal likelihoods, convergence diagnostics, Geweke getting-it-right CI tests.
**Gate:** the replication gallery opens — Blanchard-Quah (1989), Uhlig (2005), Gertler-Karadi (2015), Ramey-Zubairy (2018) multipliers, Giannone-Lenza-Primiceri (2015), Primiceri (2005, corrected), Kilian (2009) oil VAR, Antolín-Díaz & Rubio-Ramírez (2018) — each reproduced as an executable doc that runs in CI.

### Phase 3 — Nowcasting and the production layer · ~5–7 months
- **Nowcasting (08):** DFM nowcasting facade on 04's EM implementation (block structure, ragged edges), news decomposition, vintage store + release calendar, pseudo-real-time evaluation harness, MIDAS family (Almon/beta/U-MIDAS/ADL-MIDAS with leads), MF-VAR (stacked + state-space), MF-BVAR (Schorfheide-Song).
- **Forecast evaluation depth (09):** GW/Clark-West/MCS/SPA, fluctuation tests, PIT/Berkowitz density evaluation, combination (equal-weights → Bates-Granger → opinion pools), conformal prediction (ACI/EnbPI), conditional/scenario forecasts (Waggoner-Zha), fan charts, forecasting-under-breaks (Pesaran-Timmermann/PPP 2013).
- **Multivariate depth (04):** FAVAR, connectedness (Diebold-Yilmaz), SVEC, MS-VAR/TVAR/STVAR with simulated GIRFs.
**Gate:** Banbura-Modugno news decomposition reproduces; a FRED-MD real-time nowcasting exercise matches published pseudo-real-time RMSEs; Schorfheide-Song replication; M4 benchmark harness runs end to end.

### Phase 4 — Depth and v1.0 · ~5–7 months
- **Volatility depth (03):** DCC/cDCC/ADCC/BEKK/GO-GARCH with correct two-step inference, realized measures + jump tests + noise-robust estimators, Realized GARCH/HEAVY, SV via MCMC (KSC/Omori) and particle filters, GAS models, EVT tails, ES backtests, GARCH-MIDAS.
- **ML (10):** penalized solvers (+ regularized VARs in 04), native random forest/boosting with TS-aware resampling, factor/diffusion-index forecasting, DML with dependent data, GP-SSM, interpretation tooling, contamination-aware benchmark harness.
- **Univariate depth (02):** Markov-switching, SETAR/STAR with test batteries, ARFIMA/long memory, structural-break workflows.
**v1.0 gate:** API freeze + deprecation policy in force; model cards for every core estimator; the "which model when" guide complete; replication gallery ≥ 15 papers; public benchmark dashboard; tiering published; conda-forge feedstock live.

### Packaging workstream (Module 14) — cross-cutting, first release after Phase 1
Packaging is not a phase but a workstream that ships an installable artifact as early as there is a usable core, then matures to a mature release process by v1.0. Milestones: **`0.0.x` first public release** right after Phase 1 (complete `pyproject.toml` metadata, cross-platform abi3 wheel matrix + sdist via `maturin-action`/`cibuildwheel`, hand-written type stubs + `py.typed`, tag-triggered GitHub Actions release with PyPI **trusted publishing**, dry-run on Test PyPI first) so anyone can `pip install <name>` the alpha; **`0.x` maturation** (conda-forge feedstock, `show_versions()` provenance, optional-dependency extras, wheel-level clean-environment CI smoke tests, changelog + semver policy, docs site) through Phases 2–4; **v1.0** carries the API-stability guarantees and build attestations. Full plan: [14-packaging-distribution.md](docs/roadmap/14-packaging-distribution.md). Prerequisite: resolve the real library name (Module 11) before the first upload — the first PyPI publish claims the name permanently.

### Phase 5 — Extensions (post-1.0, demand-ordered)
E1 panel time series → E2 term structure → E3 predictive regressions → E4 GMM/SMM/indirect inference (see [12-extensions.md](docs/roadmap/12-extensions.md)); the DSGE-lite decision (E5) executes here; E6–E8 follow; companion packages (deep-learning adapters, causal-panel suite) spin up under separate versioning.

### Continuous (every phase)
Docs written with the code (a feature without its model card is unfinished); golden-value tests accumulate monotonically; benchmark regression tracking; quarterly frontier-watchlist triage; community/extension API from Phase 3.

## 8. Validation and quality strategy

- **Golden-value harness**: every estimator pinned against statsmodels/R/Matlab reference output or published tables, with explicit numerical tolerances and a written tolerance policy (condition-number-aware).
- **Replication gallery as CI**: the famous-paper reproductions are executable documentation *and* integration tests — docs drift is test failure.
- **Simulation-based calibration** for every Bayesian sampler; Geweke "getting it right" joint-distribution tests in CI.
- **Property-based testing** (proptest) for invariants: stationarity-region reparameterizations, filter/smoother consistency, forecast-variance monotonicity, IRF equivalences (LP = VAR under the Plagborg-Møller-Wolf conditions).
- **Monte Carlo size/power suites** for the test batteries, checked against published tables (this is where the fast core pays for itself — these suites are feasible in CI).
- **Cross-implementation determinism**: same seed ⇒ same result at any thread count, guaranteed by the counter-based RNG design.

## 9. Documentation and adoption strategy

Full detail: [docs/roadmap/11-docs-ux-adoption.md](docs/roadmap/11-docs-ux-adoption.md)

- **Diátaxis architecture**: tutorials / how-to / reference / explanation, versioned and executed.
- **The "which model when" guide** — the flagship: symptom-driven entry points ("my series is persistent and I need an impulse response", "I have quarterly GDP and monthly indicators"), decision trees with escape hatches, each leaf linking to a model card and a runnable example on a bundled dataset.
- **Model cards for every estimator**: assumptions, when to use, failure modes, defaults and why, references, and the validation targets it passes.
- **Head-to-head explainers**: LP vs VAR, BVAR priors compared, the GARCH zoo, identification schemes compared.
- **Migration guides**: from statsmodels, R, Stata/Matlab, with a cross-package Rosetta glossary.
- **Datasets policy (settled)**: tsecon ships no loaders and makes no network calls — data acquisition belongs to specialist tools, and hardcoded provider URLs are a maintenance liability the core refuses to own. Replications run on small public datasets committed to the repo; any richer loader suite (FRED-MD/QD vintages, shock series, licensing audit) is companion-package territory.
- **Error messages that teach**: "the AR polynomial root is 1.002 — your series may be nonstationary; see the unit-root workflow (link)".
- **Naming**: RESOLVED — the library ships as `tsecon`. PyPI availability was verified (the name is unregistered), so the working codename became the name; it is what goes in the JOSS paper title.
- **Marketing = benchmarks + accuracy badges**: a public speed dashboard vs statsmodels/R and a per-estimator "validated against X" badge program; JOSS paper early, JSS paper at maturity; course-pack alignment for teaching adoption.

## 10. What was added beyond the original brief

The completeness review added, beyond everything discussed in the initial scoping:

1. **Panel time series econometrics** (Pedroni/Westerlund/PMG-ARDL/CCE) — E1
2. **Term-structure econometrics** (Diebold-Li, AFNS, JSZ affine models, ACM term premia, shadow rates) — E2
3. **Predictive-regression inference** (Stambaugh, Campbell-Yogo, IVX) — E3
4. **GMM / SMM / indirect inference / IRF-matching** — E4
5. **A DSGE-lite estimation layer** (gensys/Klein + SMC), decided explicitly rather than left ambiguous — E5
6. **Survey-expectations toolkit** (SPF structures, Coibion-Gorodnichenko regressions) — E6
7. **Fractional cointegration** (FCVAR) — E7
8. **Recession-probability models** (dynamic probit/logit) — E8
9. **High-frequency completions** (jump tests, noise-robust realized measures) — folded into Module 03
10. **Forecasting under structural breaks** (Pesaran-Pick-Pranovich) — folded into Module 09
11. **Bootstrap completions** (subsampling, dependent wild, grid bootstrap) — folded into foundations
12. **Bundled datasets & loaders with licensing audit** — built, then deliberately removed from the core (no-network boundary); future home is a companion package
13. Specification tests, functional time series, climate econometrics (docs deliverable), extremal dependence — E9–E12, deferred

## 11. Companion packages (separate wheels, separate cadence)

- **Deep learning & foundation models**: N-BEATS/N-HiTS/DeepAR/TFT adapters and Chronos/TimesFM/Moirai/TimeGPT integrations. Fast-churning APIs and heavy dependencies contradict the static-wheel promise; the *contamination-aware benchmark harness* stays in core (that part is durable econometrics).
- **Causal panels**: synthetic control (classic/augmented), synthetic DiD, matrix completion, conformal counterfactuals — a different literature with credible existing Python entrants. LP-DiD and IPW/AIPW-LP stay in core because they are IRF-native.
- **Gradient-boosting adapters** (XGBoost/LightGBM): wrap-don't-own, outside the core wheel.

## 12. Non-goals

Full DSGE tooling beyond the E5 minimal layer (higher-order perturbation, OccBin, projection — Dynare interop is the answer); reimplementing X-13ARIMA-SEATS; a plotting framework (tidy returns + optional matplotlib convenience); a deep-learning framework; high-dimensional vine copulas in v1 (pyvinecopulib interop documented); supply-chain/retail forecasting features driving API design; GPU in v1 (designed-for, not shipped).

## 13. Risks and mitigations

| Risk | Mitigation |
|---|---|
| **Scope drowning** (the Kats/PyFlux failure mode) | Public tiering, hard v1 cap, contrib tier + extension API, validation-target gating for every frontier item |
| **Silent statistical wrongness** (worse than crashing) | Golden-value gating, replication-gallery CI, SBC for samplers, Monte Carlo size/power suites |
| **The Bayesian/structural overlap forks** (two sign-restriction samplers disagree) | Single ownership map (§5), one rotation kernel in foundations |
| **Wheel/packaging pain erodes trust** | Pure-Rust core (no BLAS linkage), abi3 wheels, conda-forge from day one, X-13 binary as isolated workstream with STL fallback |
| **Bus factor / burnout** | Governance plan early, JOSS visibility, small-and-finished beats broad-and-broken, funding applications (NumFOCUS affiliation) at v1 |
| **statsmodels inertia** ("good enough") | The wedge is what statsmodels *doesn't have* (06/07/05/08); benchmarks + replication gallery make switching legible; migration guides lower the cost |

## 14. Success metrics

- **v1**: parity battery green; ≥ 15 executable paper replications; benchmark dashboard showing order-of-magnitude Monte Carlo speedups; model cards for 100% of core estimators.
- **Year 1 post-v1**: appears in a published paper's replication files not written by the authors of the library; ≥ 3 external contributors with merged estimators; a graduate course adopts it.
- **Steady state**: the default answer to "how do I estimate a proxy SVAR / run LP-IV / nowcast GDP in Python."

## 15. Document index

| Doc | Contents |
|---|---|
| [00-architecture.md](docs/roadmap/00-architecture.md) | Language verdict, numerical stack, SSM engine, RNG, bootstrap engine, covariate contract, packaging, testing |
| [01-diagnostics-exploration.md](docs/roadmap/01-diagnostics-exploration.md) | The base layer: test batteries, unit roots, breaks, filters, spectral, seasonal adjustment, `check_series()` |
| [02-univariate.md](docs/roadmap/02-univariate.md) | ARIMA/ETS/UC/TVP, Markov-switching, threshold/STAR, long memory, count/duration models |
| [03-volatility.md](docs/roadmap/03-volatility.md) | GARCH zoo, SV, multivariate GARCH, realized measures + high-frequency additions, VaR/ES backtesting |
| [04-multivariate.md](docs/roadmap/04-multivariate.md) | VAR/VECM/VARMA, factor models (DFM home), nonlinear VARs, connectedness, IRF inference machinery |
| [05-bayesian.md](docs/roadmap/05-bayesian.md) | BVAR prior families, large BVARs, SV/TVP, samplers, marginal likelihoods, sampler-correctness CI |
| [06-identification.md](docs/roadmap/06-identification.md) | The unified structural-VAR restrictions layer, frequentist + Bayesian backends, scheme decision guide |
| [07-local-projections.md](docs/roadmap/07-local-projections.md) | LP/LP-IV, lag-augmented default inference, state-dependent/panel/smooth LP, LP-DiD, LP-vs-VAR dual reporting |
| [08-nowcasting-mixed-frequency.md](docs/roadmap/08-nowcasting-mixed-frequency.md) | MIDAS family, DFM nowcasting facade, MF-VAR/MF-BVAR, vintages, release calendar, news decomposition |
| [09-forecasting-evaluation.md](docs/roadmap/09-forecasting-evaluation.md) | Backtesting engine, accuracy/comparison tests, density evaluation, combination, conformal, forecasting under breaks |
| [10-machine-learning.md](docs/roadmap/10-machine-learning.md) | Penalized solvers, TS cross-validation, trees/boosting, DML, GP-SSM, interpretation, scope rulings |
| [11-docs-ux-adoption.md](docs/roadmap/11-docs-ux-adoption.md) | Competitive audits, docs architecture, model cards, bundled datasets, migration guides, naming, governance |
| [12-extensions.md](docs/roadmap/12-extensions.md) | The beyond-the-brief additions E1–E12 |
| [13-visualization.md](docs/roadmap/13-visualization.md) | The style system, chart catalog (IRF grids, fan charts, dashboards), export presets, visual regression CI |
