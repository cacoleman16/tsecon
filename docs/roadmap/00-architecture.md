# Module 00 — Systems Architecture

> Part of the time series econometrics library roadmap. Master plan: [ROADMAP.md](../../ROADMAP.md).

**This module is the substrate everything else stands on: the language and toolchain verdict (Rust, PyO3, maturin), the numerical linear algebra stack, the reproducible parallel RNG and replication engine, the shared state-space and optimization machinery, the data/time-index layer, the API contract every model family follows, and the build, packaging, testing, and validation strategy. Unlike the domain modules, it also owns the master plan's "foundations" — the bootstrap engine, HAC inference, the IRF result object, the factor-model core, the critical-value engine, the innovation-distribution zoo, and the other shared statistical services that every other module consumes rather than reimplements.**

## Purpose and scope

The foundational decision is settled here: the numerical core is written in **Rust**, exposed to Python through PyO3 with maturin-built wheels. Fortran is rejected for the core despite its numerical pedigree — f2py is fragile, there is no MSVC Fortran so Windows support is a chronic problem (SciPy has been actively translating Fortran out of its tree since roughly 2023 for exactly this reason), fpm/meson wheel tooling is immature, and the Fortran contributor pool is shrinking while Rust is where new scientific-systems contributors are arriving. C++ with pybind11 works but offers no unified build and dependency story and no memory safety. Rust's cargo gives reproducible builds with integrated tests, benchmarks, and docs; maturin builds manylinux/macOS/Windows wheels in one CI action, with polars, pydantic-core, tokenizers, ruff, and cryptography as precedent at massive scale; and rayon plus the borrow checker eliminate the data-race class that plagues OpenMP Fortran in heavily parallel bootstrap and Monte Carlo code. The acknowledged cost — more verbose generic numerics and a younger linear algebra ecosystem — is mitigated by adopting faer as the dense LA engine. The verdict is validated empirically before full commitment: a two-week spike prototyping the Kalman filter and GARCH MLE in Rust/faer and Fortran, benchmarked against statsmodels.

This module defines what the master plan's ownership map calls "foundations": the linear-Gaussian state-space engine (exact diffuse initialization, univariate filtering, interchangeable Durbin–Koopman and precision-based simulation smoothers, EM), the resampling/bootstrap engine with its parallel RNG substream contract, HAC/long-run-variance inference with one library-wide default policy, the typed IRF result object and generalized-IRF engine, the factor-model estimation core, the fast quantile-regression solver, the critical-value engine, the innovation-distribution zoo, the time-index/calendar/vintage data layer, the exogenous-regressor (covariate) contract, temporal disaggregation utilities, the Haar-rotation/restriction-algebra kernel, the numerical optimizers, and the deterministic-terms toolkit. It also owns two decisions the master plan has ratified: the plotting layer (results objects return tidy data plus `.plot()` convenience methods, with matplotlib as an optional dependency — no hard plotting dependency in core) and the missing-data policy (one written policy document; NaN-based user-facing representation; state-space filtering internally; per-estimator declared behavior of error, skip, or filter).

Its users are twofold. Library developers in every other module build on these primitives — no domain module ships its own filter, optimizer, bootstrap loop, or RNG. End users touch this module indirectly through its guarantees: thread-count-invariant seeded results, wheels that install anywhere with only NumPy as a runtime dependency, a stable versioned persistence format, and a validation gallery tying outputs to published tables. The relation to other modules is strictly one-directional: this module exposes infrastructure and consumes nothing statistical from downstream, with the single exception of the X-13ARIMA-SEATS wrapper and STL/MSTL (owned by diagnostics), for which this module supplies the binary-distribution packaging workstream.

## Where existing tools fall short

- **statsmodels**: single-threaded almost everywhere; bootstrap and Monte Carlo must be hand-rolled in Python loops; the Kalman code is Cython but defaults to approximate ("big kappa") diffuse initialization, lacks steady-state gain freezing, and the state-space, ARIMA, and VAR stacks are partially duplicated with inconsistent APIs and NaN policies; pickle-based persistence breaks across versions; releases are slow and Bayesian support is minimal.
- **arch (Sheppard)**: excellent univariate volatility and bootstrap correctness, but the bootstrap loops run at Python speed, multivariate volatility (DCC/BEKK) is absent, and there is no shared state-space or Bayesian infrastructure.
- **R ecosystem fragmentation**: vars, BVAR, bvarsv, lpirfs, midasr, KFAS, rugarch, and nowcasting each have their own data conventions, seed handling, and results objects; combining them for one paper means writing glue code and trusting five maintainers; almost all are single-threaded and none share a filtering engine.
- **Dynare/MATLAB**: strong estimation tooling (steady-state Kalman, exact diffuse filters, doubling algorithms) locked behind a MATLAB license and a monolithic DSGE-centric interface; not usable as a general library.
- **Parallel reproducibility**: no existing package guarantees thread-count-invariant, seed-reproducible parallel bootstrap or Monte Carlo results; it is ad hoc everywhere (R's parallel seeds, joblib chunking artifacts).
- **Precision-based state sampling**: no package exposes Chan–Jeliazkov banded-Cholesky state sampling as a reusable primitive, despite it underpinning a decade of large-BVAR/TVP literature — every paper re-implements it.
- **Mixed-frequency and real-time data**: ragged-edge handling is bolted on everywhere; statsmodels barely supports it, R splits it across midasr/nowcasting/mfbvar with incompatible conventions; vintage data as a first-class object exists nowhere.
- **Validation opacity**: none of the major packages ship a public replication gallery tying outputs to published tables, and cross-package log-likelihood discrepancies (diffuse constants, GARCH presample conventions) are undocumented, wasting enormous researcher time.
- **X-13ARIMA-SEATS in Python**: statsmodels requires a user-located binary with poor errors; R's seasonal package shows binaries can be bundled with good UX, but Python users have nothing comparable.
- **Serialization and production deployment**: no time series econometrics library offers a stable, versioned, pickle-independent persistence format, which blocks central-bank production use.

## Inventory

### Tier 1 — Core (v1-blocking)

#### Language and toolchain

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Core language verdict: Rust (PyO3/maturin), not Fortran or C++ | All numerical kernels in Rust, exposed via PyO3; Fortran retained only as a wrapped legacy binary where unavoidable (X-13). | Low | Rationale: cargo tooling, maturin one-action multi-platform wheels (polars/ruff precedent), growing contributor pool, rayon + borrow checker eliminating OpenMP-style data races. Cost (verbose generics, younger LA ecosystem) mitigated by faer. Validate via a 2-week spike: Kalman filter + GARCH MLE in Rust/faer vs Fortran, benchmarked against statsmodels, before committing. |

#### Linear algebra

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Dense linear algebra: faer as primary backend | Pure-Rust SIMD/rayon LA engine (matmul, pivoted QR, LLT/LDLT/Bunch-Kaufman Cholesky, LU, SVD, eigendecompositions) as the default; no system BLAS. | Medium | Within ~1–2x of MKL/OpenBLAS at econometric sizes (k=2–200), faster at tiny sizes; enables fully static wheels, identical numerics across platforms, no BLAS-vs-rayon thread oversubscription. Reject nalgebra (fixed-size graphics focus) and ndarray-linalg (LAPACK dependency, spotty maintenance) as engines; ndarray survives only as an FFI view type. Keep a `lapack` cargo feature for verification, never the default wheel. Trap: verify faer exposes real Schur/Hessenberg forms needed by the Sylvester suite; if not, implement the QR-algorithm Schur step in-house. Validate every decomposition against LAPACK (dgeqp3, dpotrf, dgesdd, dhseqr) on random plus pathological (Hilbert, near-singular) matrices at 1e-12 relative tolerance. |
| Internal array convention (column-major kernels, ndarray views) | One documented memory-layout convention: kernels operate on f64 column-major (T×k, Fortran order); conversion happens exactly once at the Python boundary. | Medium | NumPy defaults to C-order, faer is column-major-native: convert or accept strided views once on entry, never inside loops; expose a zero-copy fast path for Fortran-ordered input. Trap: silent transposes inside per-replication bootstrap loops are the #1 hidden performance bug in numpy-based econometrics code — encode layout in the type system (newtype `ColMajor<T>`) so the compiler catches it. |
| Sylvester/Lyapunov/Riccati solver suite | Discrete Lyapunov for unconditional state covariances (VAR companion form, SSM initialization), Sylvester for quadratic-form expectations, Riccati/doubling for steady-state Kalman gains. | High | Never use the vec-Kronecker formula beyond k≈15 — O(k^6) and ill-conditioned. Implement Bartels–Stewart (1972) on real Schur form (the 2x2 block handling for complex pairs is the fiddly part), Hammarling (1982) returning the Cholesky factor directly (needed for square-root filters), and the structure-preserving doubling algorithm for the discrete Riccati equation (Chu et al. 2005; what Dynare uses). Validate against scipy `solve_discrete_lyapunov`/`solve_discrete_are` and SLICOT SB03MD/SB02MD, including eigenvalues near the unit circle where naive methods lose all digits. |
| Toeplitz and AR-structure fast solvers | O(n^2) solvers exploiting stationary autocovariance structure: Levinson–Durbin (with PACF byproduct), Trench inverse, innovations algorithm for MA/ARMA prediction. | Medium | Brockwell–Davis (1991) ch. 5 for the innovations algorithm. Trap: Levinson–Durbin is numerically unstable for near-nonstationary processes — detect pacf magnitudes approaching 1 and fall back to Cholesky. Validate PACF/Yule–Walker against statsmodels.tsa.stattools and Brockwell–Davis textbook tables. |
| Positive-definiteness hygiene utilities | Robust Cholesky with escalating jitter, Higham (1988/2002) nearest-correlation/nearest-PSD projection, log-determinant via Cholesky, symmetrization helpers. | Medium | Every covariance the library emits passes one central symmetrize-and-verify path (0.5(P+P') then LLT with jitter ladder 1e-12→1e-8, logged when triggered). Newton-accelerated nearest correlation (Qi–Sun 2006) for DCC/copula work. Never compute log-det via determinant — always 2·Σ log(diag(L)). Validate nearest-correlation against R `Matrix::nearPD` and Higham's published test matrices. |

#### FFI / Python bindings

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| PyO3 + rust-numpy zero-copy boundary | Accept `PyReadonlyArray2<f64>` zero-copy views, release the GIL for all computation over ~1ms, return owned arrays allocated once via `IntoPyArray`. | Medium | Validate and borrow under the GIL, then `allow_threads` for the compute; handle non-contiguous or wrong-dtype input by a single explicit copy with a documented warning, never per-element strided access in hot loops; never touch Python objects inside the detached closure (the type system enforces this). Trap: holding an array borrow across a detach point while Python mutates it — rust-numpy borrow flags catch most cases; add tests that mutate arrays from another thread. |
| Thin binding crate / Python-free core | Exactly one PyO3 crate that only converts types and maps errors; all domain crates compile and test without Python. | Low | Keeps R (extendr), Julia, and WASM doors open. Error strategy: thiserror enums in core → a small Python exception hierarchy (TsError, ConvergenceError, SpecificationError, NumericalError) preserving the Rust error chain. Anti-pattern to avoid: the "thick binding" where MC/bootstrap orchestration lives in Python (statsmodels' problem) — orchestration must live in Rust or GIL/callback overhead destroys the performance pillar. |

#### Parallelism

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| rayon-based `par_replicate` engine | One shared `par_replicate(n_reps, seed, task)` primitive that all bootstrap CIs, Monte Carlo experiments, permutation tests, and multistart grids run through, with per-replication RNG streams. | Medium | Each replication derives its RNG deterministically from (seed, rep_index) via counter-based RNG so results are identical for any thread count. One global pool exposed as `set_num_threads()`, defaulting to physical cores; force inner kernels sequential inside parallel outer loops to prevent oversubscription (the classic numpy+joblib pathology). Trap: rayon pools do not survive fork() — detect fork and raise a clear error; document spawn. Validate: bootstrap CIs bit-identical for 1, 4, 16 threads in CI. |

#### RNG and reproducibility

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Counter-based RNG core: Philox4x32-10, bit-compatible with NumPy | Default generator: stateless counter-based design (Salmon et al. 2011) with O(1) jump-ahead and trivially independent streams keyed by (seed, stream_id), reproducible regardless of scheduling. | Medium | Implement in-house (~150 lines) rather than depend on a thin third-party crate; PCG64 offered for single-stream speed. Trap: NumPy's counter/key packing and uint64 assembly from two uint32 draws are specific — replicate exactly. Validate bit-for-bit against `numpy.random.Philox` raw uint64 output with golden vectors locked in CI. |
| Distribution sampling layer with explicit compatibility contract | Normal/gamma/Student-t/Dirichlet/Wishart/MVN samplers with a documented policy: bitwise NumPy compatibility at the uniform level, statistical equivalence at the distribution level. | Medium | rand_distr's ziggurat tables differ from NumPy's, so draws will not match bitwise even from identical uniforms — port NumPy's exact tables only for the standard normal (feasible, high value) and document everything else as statistically equivalent. Wishart via Bartlett decomposition (Smith–Hocking 1972), never summed outer products; MVN via cached Cholesky with eigendecomposition fallback for singular covariances. Validate with KS/chi-square batteries plus exact moment checks at n=1e7. |
| Seeding API mirroring NumPy SeedSequence semantics | Python-facing seeding accepting an int, SeedSequence, or Generator, spawning hierarchical child streams (per chain, per replication, per equation) with no correlation. | Low | Implement SeedSequence hash-mixing spawn logic (or call numpy's once at setup and pass derived keys into Rust). Code-review discipline: no core function ever creates entropy from the OS; every draw is traceable to the user's seed. Results objects record full seed material so any published table is reproducible from the saved results file. |

#### Optimization

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| In-house quasi-Newton suite: BFGS, L-BFGS, L-BFGS-B | The workhorse MLE optimizers; L-BFGS-B bound constraints cover variance-positivity boxes; BFGS supplies the inverse-Hessian estimate users take SEs from (with warnings). | High | The single highest-risk infrastructure item: nearly every "my estimates differ from statsmodels" report will trace here. argmin's traits are good inspiration but its line searches are not battle-hardened — implement in-house with Moré–Thuente line search and Byrd–Lu–Nocedal–Zhu (1995) projected-gradient machinery. Always optimize on internally standardized data and rescale (how arch avoids scale-dependent failures). Validate on the Moré–Garbow–Hillstrom (1981) test set, then end-to-end: GARCH(1,1) on Bollerslev–Ghysels DM/USD against the Fiorentini–Calzolari–Panattoni benchmark; ARMA on Box–Jenkins series against statsmodels/gretl. |
| Derivative-free optimizers: adaptive Nelder–Mead and differential evolution | Nelder–Mead with Gao–Han (2012) adaptive parameters for rough/kinked likelihoods; DE (Storn–Price 1997) plus Sobol multistart for multimodal objectives (Markov-switching, GARCH-M, threshold models). | Medium | Match scipy's Nelder–Mead termination semantics for easy cross-checking. Multistart is a first-class citizen — `fit(method='multistart', n_starts=...)` running in parallel via the replication engine; no mainstream library makes this the ergonomic default despite threshold/MS models routinely having local optima. Validate DE against `scipy.optimize.differential_evolution` on standard test functions. |
| Constrained reparameterization toolkit | Typed bijections between constrained model space and unconstrained optimizer space: AR stationarity via PACF transform (Monahan 1984), VAR stationarity via Ansley–Kohn (1986), positivity via softplus/exp, correlations via tanh/hyperspherical, simplex via stick-breaking. | Medium | Each transform provides forward, inverse, log-Jacobian (omitting the Jacobian correction for Bayesian priors on transformed space is a classic silent bug), and delta-method Jacobian for SEs. Trap: reparameterization pushes boundary optima to infinity in the working space — detect divergence and report "parameter at boundary" honestly instead of a fake interior optimum (statsmodels often fails silently here). |
| Covariance/inference machinery at the optimum | Numerical Hessian with Richardson extrapolation, OPG, robust sandwich (QMLE), and HAC (Newey–West, Andrews) score covariance — one shared module for every frequentist model. | Medium | The classic bug: SEs computed in unconstrained space without the Jacobian chain back to natural parameters — the transform objects own the delta method so it cannot be skipped. Bollerslev–Wooldridge (1992) QMLE sandwich is the GARCH default. Numerical Hessians of noisy (simulated) likelihoods need step sizes tied to objective noise — expose and document. Validate SEs against arch (GARCH), statsmodels (ARMA/SSM), and Bollerslev–Ghysels published tables. |

#### State-space engine

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Unified linear-Gaussian state-space representation | One representation (Z, d, H, T, c, R, Q; time-varying capable, missing-data aware) with filter/smoother/loglik/forecast/simulate, on which ARIMA, UC, DFM, TVP-VAR, MF-VAR, and structural models are all built. | High | Durbin–Koopman (2012) notation and algorithms exactly, so docs cite chapter and verse. Matrices stored as constant or time-varying with a change-point tape, not T naive copies; all downstream models compile into this representation — no per-model filter forks (statsmodels' duplication causes bug divergence); filter output is a reusable arena to avoid allocation in MC loops. Validate against KFAS (R) and statsmodels on the Nile data, DK2012 local-linear-trend examples, and Harvey (1989) UC results. |
| Univariate (sequential) treatment of multivariate observations | Koopman–Durbin (2000) sequential processing: update states one scalar observation at a time, eliminating the F_t inversion and handling arbitrary per-period missingness for free. | High | The DEFAULT filter path, not an option: faster (no k×k solve), more stable (no explicit inverse), and makes ragged-edge mixed-frequency data trivial. Requires diagonal H — apply and cache the LDL' pre-transformation when H is non-diagonal. Trap: the diffuse variant has subtle F_inf ordering issues — follow Durbin–Koopman (2012) §6.4 precisely. Validate: loglik identical (1e-10) to the standard multivariate filter on full data; against KFAS under per-series missingness patterns. |
| Exact diffuse initialization (Koopman 1997) | Exact treatment of nonstationary states via the F_inf/F_star two-matrix recursion, replacing the kappa=1e6 approximate-diffuse hack. | High | Approximate diffuse (statsmodels' historical default) contaminates the likelihood with the arbitrary kappa; exact diffuse is what KFAS and SsfPack do. Corner cases: rank-deficient F_inf at collapse time; collapsing to standard recursions at the right step. The diffuse loglik omits a constant — document the convention (Francke, Koopman & de Vos 2010) because cross-package loglik comparisons hit this constantly. Validate against KFAS numerically and DK2012 table values for the Nile model with diffuse level. |
| Simulation smoother (Durbin–Koopman 2002) | Mean-correction draws of the full state path given data — the core primitive for Bayesian DFM, TVP-VAR, UC-SV, and SV estimation; called millions of times inside Gibbs loops. | High | DK2002 beats Carter–Kohn/Frühwirth-Schnatter in speed and robustness (no per-step covariance draws). Zero-allocation and filter-workspace reuse matter more here than anywhere else. Must support exact-diffuse initial conditions (DK2002 §3; many implementations skip this and silently bias initial-state draws). Validate: repeated-draw moments converge to analytic smoother output (property test); cross-check statsmodels `simulation_smoother` and KFAS `simulateSSM`. |

#### Shared statistical foundations (owned here per the master ownership map)

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Resampling/bootstrap engine | One library-wide engine: iid, wild, moving/circular block, stationary (Politis–Romano 1994), and sieve bootstrap, with automatic block-length selection (Politis–White 2004) and the parallel RNG substream contract. | High | All schemes run through `par_replicate` with per-replication substreams, so every bootstrap in the library is thread-count-invariant by construction. Block-length selection ships as the default, not an exercise for the user. Validate against arch's bootstrap module and published coverage/simulation studies; CIs bit-identical across thread counts. |
| HAC / long-run variance / fixed-b / EWC inference | Kernel HAC estimators (Newey–West 1987, Andrews 1991, quadratic spectral) plus fixed-b asymptotics (Kiefer–Vogelsang 2005) and equal-weighted cosine inference (Lazarus–Lewis–Stock–Watson 2018), with ONE library-wide default policy. | Medium | The default policy (LLSW 2018 recommendations) is written down once and used by every estimator — no per-module bandwidth folklore. Validate against R sandwich and statsmodels; fixed-b critical values against Kiefer–Vogelsang published tables. |
| Typed IRF result object + generalized-IRF engine | The single IRF result type (point, bands, method metadata) and the generalized-IRF machinery (Koop–Pesaran–Potter 1996; Pesaran–Shin 1998) that multivariate, identification, and LP modules all emit. | Medium | Bands always carry their method (asymptotic/bootstrap/Bayesian) as metadata. Validate against Lütkepohl (2005) IRF tables and vars (R). |
| Factor-model estimation core | PCA, EM, and QML factor estimation (Doz–Giannone–Reichlin 2012) with Bai–Ng (2002), Ahn–Horenstein (2013), and Onatski (2010) factor-number criteria. | High | This is the estimation core only — the multivariate module owns the single DFM implementation built on it; nowcasting consumes that DFM. Validate against Stock–Watson datasets and R dfms/nowcasting packages. |
| Critical-value engine | Response-surface critical values (MacKinnon 1996/2010) plus on-demand null simulation, cached and versioned, serving unit-root/cointegration/stability tests across modules. | Medium | Simulated critical values are cached with (test, spec, T, seed, library version) keys so results are stable across sessions. Validate response surfaces against MacKinnon's published values. |
| Innovation-distribution zoo | Normal, Student-t, GED, Hansen (1994) skew-t, Fernandez–Steel (1998), Johnson SU, NIG, and GH skew-t: densities, quantiles, samplers, score and CRPS hooks. | Medium | One implementation consumed by volatility, forecasting, and Bayesian modules; every distribution exposes analytic scores where they exist. Trap: skew-t parameterizations differ across papers and packages — pin one and document the mapping. Validate densities/quantiles against rugarch and published moments. |
| Deterministic-terms toolkit | Constants, trends, seasonal dummies, and break dummies with one convention (naming, ordering, orthogonalization) shared by every model family. | Low | Sounds trivial; inconsistent deterministic-term handling is a recurring source of cross-package discrepancies in unit-root and VAR output. Validate by exact agreement with statsmodels/vars conventions in golden fixtures. |

#### Data layer

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Arrow-native data ingestion | Accept input via Arrow C data/stream interfaces and the numpy protocol so polars and pandas both work zero-copy where dtypes allow; never depend on pandas internals. | Medium | Use arro3/nanoarrow-style minimal FFI rather than the heavy arrow-rs tree if binary size matters. Reality check: most numeric columns arrive as f64 numpy arrays anyway — the Arrow path mainly buys clean polars interop. pandas stays optional; core requires only numpy. Output: numpy plus a thin results wrapper with lazy `.to_pandas()`/`.to_polars()`. |
| Time index engine | Internal representation of regular-frequency indexes (D/B/W/M/Q/A with anchoring), mixed-frequency alignment for MIDAS/MF-VAR/nowcasting, and ragged-edge tracking. | High | Do NOT rebuild pandas' datetime machinery: represent as (frequency enum, i64 period ordinals), exactly pandas PeriodIndex ordinals, losslessly convertible. The hard parts: quarter anchoring (Q-DEC vs fiscal), week anchoring, business-day calendars with holiday support, explicit high-to-low aggregation rules (mean/sum/last/end-of-period), per-series ragged-edge masks. Silent misalignment is the deadliest bug class in applied macro — raise on ambiguity, never auto-guess. |
| Missing-data representation and policy | NaN-based missingness at the user boundary, converted to explicit per-period observation masks inside the SSM/estimation core; ratified policy: one written policy document, per-estimator declared behavior (error/skip/filter) — never silent listwise deletion. | Medium | Distinguish three cases in the type system: internal holes, ragged edge (not yet released), and mixed-frequency structural missingness — the sequential filter treats all three, but evaluation logic must not (a "forecast" of a hole differs from a nowcast). statsmodels' inconsistent NaN policies across submodules are a known pain point; a single documented policy is a selling feature. |
| Real-time vintage data store | ALFRED-style real-time data matrices as a first-class object, recording what was known when. | Medium | Owned here as shared infrastructure; nowcasting builds its release-calendar/news layer on it and forecasting-evaluation uses it for honest pseudo-out-of-sample exercises. Validate round-trips against ALFRED extracts. |
| Exogenous-regressor (covariate) contract | One shared convention for covariates across every model family (regARIMA, VARX, GARCH-X, LP controls, MIDAS, ML pipelines): index-aligned ingestion with loud alignment diagnostics, integration with the deterministic-terms toolkit, and a future-covariate interface for forecasting that distinguishes known-future values, user-supplied scenario paths (fanning forecasts over them), and auxiliary-model forecasts with uncertainty propagated. | Medium | Ratified master-plan decision; designed in Phase 0 so no model family grows its own `xreg` convention. Must document the regression-with-ARMA-errors vs transfer-function-ARMAX distinction once, centrally (the perennial statsmodels-vs-R trap). The backtesting engine enforces leakage checks: future covariate values can never silently enter a pseudo-out-of-sample exercise. Validate alignment semantics against R `arima(xreg=)` and `forecast::forecast(xreg=)` golden fixtures. |

#### API design

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| API verdict: Model→fit()→Results with spec objects, thin sklearn adapters | Primary API: `m = VAR(data, lags=2, trend='c'); res = m.fit(); res.irf(...)` — explicit model objects, immutable Results with lazy cached properties. R-formula strings rejected as primary interface; sklearn fit/predict only as an adapter. | Medium | Time series specs (lag orders, deterministic terms, identification schemes) do not map onto formulas or sklearn's X,y; the two-step pattern is what the audience knows. Improvements over statsmodels: typed keyword-only spec arguments validated at construction; lazy cached Results properties; every Results has `.summary()`, `.to_frame()`, and consistent naming across families (statsmodels' inconsistency is its most-cited usability failure). Confidence intervals always carry their method (asymptotic/bootstrap/Bayesian) stamped on them. |
| Unified forecast object | One Forecast type shared by every model: point, interval bands with method metadata, density where available, simulated path draws on demand, conditional/scenario hooks. | Medium | Designed once so ARIMA, VAR, GARCH, SSM, and Bayesian models emit the same shape — the forecasting-evaluation module consumes any model's output through it. Simulated paths route through `par_replicate` and the seeded RNG. Conditional forecasts via Waggoner–Zha (1999) on the SSM engine. This uniformity is exactly what the fragmented R ecosystem lacks. |
| Plotting layer: tidy data + optional matplotlib (ratified) | Results objects return tidy data frames for any plot; `.plot()` convenience methods import matplotlib lazily; no hard plotting dependency in core. | Low | Ratified master-plan decision. Every plottable quantity is available as tidy data first so users of any plotting stack (matplotlib, plotnine, altair) are first-class; `.plot()` raises a clear ImportError naming the `[plots]` extra when matplotlib is absent. |

#### Testing and validation

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Golden-value validation harness | Pinned Docker images run scripted R (vars, KFAS, rugarch, BVAR, midasr) and statsmodels/arch on canonical datasets, emitting versioned JSON fixtures; Rust and Python suites assert against them with tiered tolerances. | Medium | Tolerance tiers: 1e-10 relative for same-algorithm deterministic quantities; 1e-6 for logliks at fixed parameters; 1e-4 parameter-scaled for optimizer endpoints — compare loglik at each package's optimum when packages disagree. Canonical targets: NIST StRD, Box–Jenkins series, Nelson–Plosser, Lütkepohl (2005) VAR tables, Bollerslev–Ghysels + FCP GARCH benchmark, DK2012 Nile, Stock–Watson factor data, Kilian (2009) oil VAR. Fixture-generation scripts live in the repo so disagreements can be re-litigated. |
| Numerical tolerance policy and condition-aware assertions | A written, enforced policy: what accuracy each layer promises, how tolerances scale with condition numbers and T, and a shared `assert_close(rtol, atol, context)` reporting ULP distance and condition estimates on failure. | Low | Prevents both failure modes of numerical suites: tolerances too loose to catch anything, and flaky tests ratcheted looser until meaningless. No bare float equality; cross-platform tests use tolerance bands; same-platform regression tests may pin exact bits for integer/RNG paths; every loosened tolerance requires a justification comment. Known-hard cases (near-unit-root, near-singular) live in a quarantined suite with documented expected accuracy. |
| CI matrix and sanitizer/fuzzing layer | GitHub Actions matrix: {Linux x86_64/aarch64, macOS arm64/x86_64, Windows x64} × {Python 3.10–3.13} × {oldest, latest numpy}; cargo test + clippy(-D warnings) + rustfmt + Miri on unsafe-containing crates; cargo-fuzz on index/frequency parsing. | Medium | The numpy min/max axis catches the 2.x ABI drift that broke half the ecosystem in 2024. Address/thread sanitizers on the binding crate weekly (PyO3+asan needs a special build dance). Test the actual built wheel (install from artifact, run the Python suite), not just `maturin develop` — abi3/LTO/feature-flag differences have shipped broken binaries in well-run projects. Doctests run in CI so teaching docs never rot. |

#### Packaging and distribution

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| maturin abi3 wheels, static everything, tiny footprint | abi3-py310 stable-ABI wheels (one per platform for all Python versions), manylinux_2_28 x86_64+aarch64, macOS arm64 (+x86_64), Windows x64; runtime dependency: numpy only. | Medium | The pure-Rust LA decision pays off here: no vendored OpenBLAS means few-MB wheels, no rpath surgery, no BLAS thread-pool conflicts. abi3 cuts the build matrix ~5x. Thin-LTO and stripped symbols for release; a debug-symbols artifact for profiling. sdist must build with only a Rust toolchain (document MSRV). pandas, polars, matplotlib, formulaic: optional extras, never hard requirements. |
| X-13ARIMA-SEATS binary distribution workstream | Package and distribute the official Census X-13 binaries per platform (the x13binary model from R), with auto-fetch fallback and clear errors; the wrapper itself lives in diagnostics. | Medium | A distinct packaging workstream, not part of the main wheel: verify current Census redistribution terms, bundle or auto-fetch per platform, and pair with the native STL-based fallback (owned by diagnostics) so seasonal adjustment degrades gracefully when the binary is unavailable. Improves on statsmodels' locate-your-own-binary UX; R's seasonal package is the precedent. |

#### Repository and governance

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Cargo workspace and crate layout | One repo, one workspace: ts-core (traits, RNG, errors), ts-linalg, ts-optimize, ts-ssm, ts-arima, ts-var, ts-volatility, ts-bayes, ts-tests, and exactly one PyO3 crate ts-python; the Python package wraps ts-python plus pure-Python sugar. | Low | Dependency DAG flows one way (domain crates → ts-ssm/ts-linalg → ts-core); no domain crate depends on another (shared needs get promoted to core). Feature flags for `lapack` and `serde`. The Python package versions independently; internal crates stay 0.x and unpublished until APIs stabilize. Python API follows semver with a one-minor-release deprecation cycle; Rust internals explicitly unstable pre-1.0. |
| Documentation-as-product: executable guides with doctested output | Sphinx/Jupyter-book docs where every teaching page ("level shifts? seasonal? volatility clustering? → use X" decision trees) executes real code in CI, plus a validated-results gallery reproducing published tables. | Medium | The replication gallery IS a golden-value test that renders as documentation (Lütkepohl, Kilian–Lütkepohl, DK2012). myst-nb executed notebooks cached in CI; numpydoc docstrings with doctests. A WASM/pyodide build is a plausible later step (the no-BLAS stack keeps it feasible). Documentation gets the same review bar as code from the first PR — retrofitting docs culture fails. |

### Tier 2 — Standard (expected of a serious library)

#### Language and toolchain

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| unsafe-code policy and MSRV discipline | `#![forbid(unsafe_code)]` in all domain crates; unsafe permitted only in the FFI crate and audited SIMD kernels; MSRV pinned ~9 months behind stable. | Low | Enforce with `#[deny(unsafe_op_in_unsafe_fn)]`, cargo-geiger in CI, and Miri on any unsafe-containing crate; MSRV in the Cargo.toml rust-version field, tested in CI. A mostly-safe codebase with a short unsafe audit surface is a real selling point for central-bank security review. |

#### FFI / Python bindings

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Cooperative cancellation and progress reporting | Long estimations (MCMC chains, 10k-rep bootstraps) respond to Ctrl-C in a notebook and optionally report progress, without per-iteration GIL costs. | Medium | AtomicBool cancellation token; a coordinator thread re-acquires the GIL every ~200ms to call PyErr_CheckSignals and flips the token; workers poll the atomic. Progress via a lock-free counter read by an optional Python callback thread (tqdm). Trap: calling into Python from rayon workers deadlocks — never do it. On cancellation return completed chains/replications as a typed PartialResult, matching arch/PyMC expectations. |

#### Parallelism

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Deterministic parallel reductions | Fixed-order chunked pairwise summation for log-likelihood accumulation and MC averaging, so results do not depend on thread count or scheduling. | Medium | Naive rayon `reduce()` gives thread-count-dependent results, breaking reproducibility claims and golden tests. Fixed-size blocks by index, summed pairwise — also improves accuracy (O(log n) vs O(n) error). Kahan/Neumaier compensated summation offered for T>1e6. A genuine differentiator: no mainstream econometrics package guarantees thread-count-invariant results. |

#### RNG and reproducibility

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Reproducibility manifest on every stochastic result | Fitted models and simulations carry metadata: library version, seed material, RNG algorithm, thread-count-independence flag, platform, options — serialized with the result. | Low | Cheap to build, huge for the academic audience (journals increasingly require replication packages). Include `results.replication_stub()` emitting the minimal reproducing snippet. Warning: guarantee reproducibility across thread counts and within a minor version; explicitly do NOT promise bit-identity across OS/CPU (libm and FMA differences make that a false promise — say so in docs). |

#### Optimization

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Trust-region Newton (Steihaug-CG / dogleg) | For state-space MLE with analytic or complex-step Hessians; converges more reliably near boundaries (small variances) than line-search quasi-Newton. | High | Conn–Gould–Toint (2000) is the reference. Particularly valuable for unobserved-components models where a variance heading to zero (pile-up) wrecks BFGS curvature updates. Validate against MATLAB fminunc trust-region results on published UC estimates (Harvey 1989 examples). |
| EM algorithm infrastructure | Generic EM/ECM loop with convergence monitoring and acceleration, instantiated for DFMs (Watson–Engle 1983; Bańbura–Modugno 2014 with missing data), Markov-switching (Hamilton 1990), and t-errors. | Medium | Shares the E-step with the Kalman smoother engine. Include SQUAREM/Anderson acceleration (Varadhan–Roland 2008) — EM for DFMs is notoriously slow and acceleration is a cheap 3–10x. Debug builds assert monotone log-likelihood every iteration (catches E-step bugs immediately). Validate DFM-EM against Bańbura–Modugno published results and R nowcasting/dfms. |

#### State-space engine

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Square-root and Joseph-form filter variants | Cholesky-based square-root covariance filtering for numerically hostile cases (TVP models with near-zero state variances, long samples), plus Joseph-stabilized updates in the standard filter. | High | The standard update P − KFK' can go indefinite; Joseph form (I−KZ)P(I−KZ)'+KHK' is cheap insurance and should be the default. Full square-root filtering (Morf–Kailath; DK2012 §6.3) via QR-updates over faer is the escalation path, shipped as opt-in `stability='high'`. Validate: filtered covariances stay PSD on the notorious TVP-regression examples where naive filters fail. |

#### Shared statistical foundations

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Extended resampling schemes | Subsampling (Politis–Romano 1994) with convergence-rate estimation, dependent wild bootstrap (Shao 2010), tapered block bootstrap (Paparoditis–Politis 2001), fast double/nested bootstrap, and the Hansen (1999) grid bootstrap for AR roots near unity. | High | Absorbed per the completeness review: these ride on the same engine and substream contract as the core schemes. The grid bootstrap is the canonical fix for near-unit-root AR inference — validate against Hansen (1999, REStat) published grid-bootstrap confidence intervals; subsampling rate estimation against Politis–Romano–Wolf (1999) examples. |
| Fast quantile-regression solver | Interior-point (Portnoy–Koenker 1997) and ADMM solvers with dependent-data inference and monotone rearrangement (Chernozhukov–Fernández-Val–Galichon 2010). | High | Consumed by quantile-flavored models across modules (CAViaR, quantile LPs, growth-at-risk). Validate coefficients and inference against R quantreg on canonical datasets. |
| Temporal disaggregation and benchmarking utilities | Chow–Lin (1971), Denton (1971), Fernandez (1981), and Litterman (1983) methods for interpolating low-frequency series with high-frequency indicators. | Medium | Small, self-contained, heavily used in production statistics. Validate against R tempdisagg outputs on its reference examples. |
| Haar-rotation / restriction-algebra kernel | Uniform-Haar orthogonal matrix draws (QR of Gaussian with sign fix) and the sign/zero-restriction algebra of Rubio-Ramírez–Waggoner–Zha (2010) and Arias–Rubio-Ramírez–Waggoner (2018). | Medium | The kernel only — the identification module owns the structural-VAR logic built on it. Trap: naive QR without the sign-fix does not give the Haar measure. Validated end-to-end through identification's Uhlig (2005) replication. |

#### API design

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Model serialization with versioned schema | Save/load of fitted results to a stable versioned format (serde-backed binary or JSON+npz hybrid), independent of pickle, with an explicit cross-version compatibility policy. | Medium | Pickle across versions is a chronic statsmodels wound. The file embeds (schema_version, library_version, reproducibility manifest); newer schemas error clearly on load; older schemas load within a major version. Pickle keeps working via `__getstate__` delegating to the stable format. `res.save('model.tse')`/`load` ships in v1 — central-bank production users ask immediately. |

#### Testing and validation

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Property-based testing with proptest | Randomized invariance tests: transform round-trips, likelihood invariance under scaling where theory says so, filter-vs-smoother consistency, PSD-ness of every emitted covariance, simulate→estimate recovery, gradient-vs-complex-step agreement. | Medium | Strategies generate valid parameters via the reparameterization toolkit (stationarity/positivity by construction — the transforms get free testing). Key properties: smoother variance ≤ filter variance elementwise; simulation-smoother moments → analytic moments; sequential loglik == multivariate loglik; VAR IRFs on own simulated data converge to truth as T grows. Shrinking gives minimal failing cases. Reduced iterations per-PR, full nightly. |
| Benchmark suite: criterion.rs micro + asv-style macro benchmarks | Two-layer performance CI: criterion for Rust kernels (filter step, GARCH recursion, Cholesky) with historical tracking; Python end-to-end benchmarks (GARCH fit at T=5000, 10k-rep bootstrap, BVAR posterior) against statsmodels/arch/R baselines. | Medium | Publish comparative numbers and keep them honest by re-running competitors in the same pinned Docker environment (stale comparisons destroy credibility). Guard against noisy runners: dedicated hardware or criterion's statistical change detection, alerting on >10% sustained regressions rather than failing PRs on noise. Launch headline claims: Kalman-based ARIMA MLE, bootstrap IRF bands, and DFM-EM each ≥10x statsmodels single-thread. |

#### Packaging and distribution

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| conda-forge feedstock from day one | A conda-forge package tracking PyPI releases; much of the target audience (academia, central banks with locked-down IT) lives on conda. | Low | Rust builds on conda-forge are routine (polars precedent). Gotcha: conda-forge builds from sdist with its own toolchain and no network — keep the sdist self-contained, no git-dependency crates. Also ship an offline-installable docs bundle; air-gapped central-bank environments are a real deployment target. |

#### Data layer / build-vs-buy

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Business-day/holiday calendars: minimal in-house core | Weekday plus user-supplied-holiday-list calendars implemented natively; interop with pandas calendar objects at the boundary; no exchange-calendar dependency in core. | Low | Econometrics needs alignment and lag arithmetic on business days, not exchange session times. Accept a holiday date array or a pandas CustomBusinessDay at the Python layer, compiled to a bitmap calendar in Rust. Full exchange calendars remain the user's problem via optional pandas interop. |

### Tier 3 — Advanced (differentiators)

#### Parallelism

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| SIMD strategy: pulp/portable-simd kernels for scalar recursions | Explicit SIMD only where autovectorization fails: GARCH variance recursions vectorized across bootstrap replications (not time), likelihood point-wise terms, filtering innovations. | High | Time-recursive loops have loop-carried dependencies — the correct trick is batching 4/8 replications in SIMD lanes (structure-of-arrays layout). Use pulp for runtime AVX2/AVX-512/NEON dispatch so wheels stay portable; never compile with `-C target-cpu=native`. Measure first with criterion — most wins come from memory layout, not intrinsics. Trap: FMA contraction changes last-ulp results — pin `mul_add` usage explicitly per kernel. |

#### Optimization

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Differentiation strategy: analytic first, complex-step second, dual-number third, Enzyme watched | A per-model-class gradient policy, not one AD hammer: hand-coded analytic gradients for hot likelihoods, complex-step for verification and Hessians, forward-mode duals for small research models, audited finite differences as fallback. | High | Policy: GARCH — analytic recursive derivatives (Fiorentini–Calzolari–Panattoni 1996); linear-Gaussian SSM — exact score via the Koopman (1992)/Koopman–Shephard smoother identity (one filter+smoother pass yields the whole gradient); ARMA — CSS analytic gradient plus exact-MLE score via the SSM route. Complex-step (Squire–Trapp 1998) gives machine-precision derivatives but requires kernels generic over a complex-like scalar and free of abs/max/branches — audit for holomorphicity. Enzyme is nightly-only: track, do not depend. CI property test: every analytic gradient vs complex-step at 1e-12 over random parameter points. |

#### State-space engine

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Steady-state filter detection and gain freezing | Detect convergence of P_t in time-invariant models and switch to fixed-gain recursions, reducing per-step cost to matrix-vector work. | Medium | 5–50x speedups on long series (T>1e4) and inside MLE loops. Test the P_{t+1}−P_t norm at ~1e-12 relative sustained over several steps, or solve the Riccati equation by doubling and verify. Traps: missing observations reset convergence; time-varying matrices disable it — the detector must be aware of the change-point tape. Dynare and SsfPack do this; statsmodels does not (a concrete headline benchmark win). Validate loglik identical to the non-switching filter at 1e-9. |
| Nonlinear/non-Gaussian filtering layer: EKF, UKF, particle filters | Extended and unscented Kalman filters plus bootstrap/auxiliary particle filters with systematic resampling and log-sum-exp weights, for SV models, second-order DSGE, and non-Gaussian UC models. | Research-grade | Systematic resampling (lowest variance of the simple schemes), adaptive resampling on ESS threshold, all weights in log space (naive normalization underflows within tens of periods). Counter-based RNG makes the PF loglik a deterministic function of (params, seed) — required for correlated pseudo-marginal methods (Deligiannidis et al. 2018). Tempered particle filter of Herbst–Schorfheide (2019) for DSGE likelihoods. Validate: PF loglik on a linear-Gaussian model converges to the Kalman loglik; SV estimates vs Kim–Shephard–Chib (1998). |

#### Testing and validation

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Simulation-based calibration (SBC) for all Bayesian samplers | Talts et al. (2018) rank-uniformity checks as scheduled CI: draw from the prior, simulate data, run the sampler, verify posterior rank uniformity — catches subtle sampler bias no unit test finds. | High | Apply to every Gibbs step and the NUTS integration: BVAR posteriors, simulation smoother inside Gibbs, SV samplers, MS-model FFBS. Use Modrák et al. (2023) refinements (ECDF-difference plots, gamma statistic) for automated pass/fail; nightly/weekly job on a beefy runner, not per-PR. Geweke (2004) joint-distribution tests as the cheaper per-PR smoke check. No econometrics library does CI-integrated SBC — "every sampler is SBC-validated" is a credibility differentiator aimed at central banks. |

#### Build-vs-buy

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| NUTS/HMC: adopt/embed nuts-rs, keep Gibbs/FFBS in-house | Build gradient-based MCMC on nuts-rs (the Rust NUTS engine behind PyMC's nutpie, with mass-matrix and normalizing-flow adaptation) rather than writing a new sampler. | Medium | A correct NUTS (multinomial sampling, divergence handling, adaptation per Hoffman–Gelman 2014 and Betancourt 2017) is a year of subtle bugs; nuts-rs is battle-tested and MIT-licensed. The in-house value is the Gibbs infrastructure — conjugate BVAR steps, FFBS, simulation smoother, Chan precision samplers — which no general PPL provides at acceptable speed. Gradients come from the differentiation strategy above. Validate embedded NUTS with SBC and against Stan posteriors on shared models (eight-schools, a small BVAR). |

### Tier 4 — Frontier (research-grade, each gated on a named validation target)

#### Parallelism

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| GPU as a designed-for future option (not v1) | Keep the replication engine and filter APIs backend-agnostic so batched-GPU execution (thousands of Kalman/particle filters in parallel) can be added later without API breaks. | Research-grade | Candidates by 2026: cudarc (CUDA), wgpu/CubeCL (portable). Valuable targets: batched small-matrix Kalman filters for SMC/particle MCMC and massive MC studies (literature shows 50–100x for batched filtering). Do NOT attempt in v1 — it doubles CI complexity and the CPU story must be excellent first. Concrete v1 obligation: kernels free of hidden global state, expressed over slices, so GPU is a code addition, not a rewrite. Gate: batched-GPU filter logliks match the CPU Kalman filter at tolerance on the DK2012 Nile benchmark before any GPU code merges. |

#### State-space engine

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Precision-based (banded sparse) state-space sampling | Chan–Jeliazkov (2009): exploit the block-banded precision matrix of the joint state vector, sampling entire paths via banded Cholesky — often much faster than Kalman-based simulation smoothing for TVP-VARs and UC models. | High | The workhorse of the Chan (2017+) large-Bayesian-VAR literature; requires an in-house banded/block-tridiagonal Cholesky kernel (bandwidth small and fixed). The master plan commits to it as an interchangeable backend beside the DK2002 smoother behind one API, with docs on when each wins (precision wins for high-dimensional states with a thin temporal band). No general library exposes this as a primitive — R users hand-roll it per paper. Gate: identical posteriors (up to MC error) to the DK2002 sampler on a TVP regression. |

## Frontier watchlist

- Correlated pseudo-marginal MCMC and tempered particle filtering for DSGE/SV likelihoods (Deligiannidis, Doucet & Pitt 2018; Herbst & Schorfheide 2019) — enabled by the deterministic seeded likelihood estimates the RNG contract already provides.
- Enzyme (LLVM reverse-mode AD for Rust, 2023–2026): exact gradients of compiled Kalman/particle likelihoods without hand derivation — still nightly-only; track and adopt when stable.
- Koopman–Shephard smoother-based exact score as the default gradient for all state-space MLE (one smoother pass for the entire gradient), with complex-step verification in CI.
- Normalizing-flow (Fisher-divergence) mass-matrix adaptation in nuts-rs (2024) — inherit as nuts-rs matures rather than reimplementing.
- WASM/pyodide builds of the full engine for in-browser executable teaching documentation — made feasible by the pure-Rust no-BLAS stack.
- Vintage-aware real-time data objects wired into forecast evaluation so pseudo-out-of-sample exercises are honest by construction (evaluation logic owned by forecasting-evaluation; the vintage store lives here).

## Implementation warnings

- **Diffuse initialization conventions differ across packages**: approximate-diffuse (big kappa) and exact-diffuse likelihoods differ by constants and precision, so logliks are not comparable across statsmodels/KFAS/Dynare. Pick exact diffuse, document the omitted diffuse constant, and never let golden tests compare mismatched conventions.
- **Never invert F_t in the Kalman filter** or form explicit matrix inverses anywhere: use univariate sequential processing or Cholesky solves; use Joseph-form covariance updates and symmetrize P_t each step, or covariances go indefinite on ill-conditioned problems and downstream Cholesky calls panic intermittently.
- **Floating-point reduction order**: rayon `reduce()` gives thread-count-dependent likelihoods and bootstrap statistics; without fixed-chunk pairwise summation the library's own reproducibility promise is false and golden tests flake.
- **Cross-platform bit-identity is unattainable** (libm differences, FMA contraction, SIMD width): promise bit-identity per-platform per-version only, write tolerance-based cross-platform tests, and decide `mul_add` usage explicitly per kernel rather than letting the compiler choose.
- **rand_distr's normal sampler is not bitwise NumPy-compatible** even from identical uniform streams (different ziggurat tables): declare the compatibility level per distribution explicitly, or users validating against numpy-based simulation studies will file "wrong results" bugs that are just different valid draws.
- **rayon thread pools do not survive fork()**: Python multiprocessing with the fork start method (Linux default before 3.14) deadlocks after the first parallel call — detect fork and error clearly; document spawn.
- **GARCH presample/burn-in conventions** (initial variance = sample variance vs unconditional vs backcast) differ across arch/rugarch/EViews/G@RCH and move estimates in the third decimal: implement the alternatives as named options and pin each golden test to its matching convention.
- **Optimizer endpoint comparisons are the wrong golden test**: different valid optimizers stop at different points near a flat optimum — compare log-likelihood values at each package's reported optimum, standardize data internally before optimizing, and rescale results.
- **SEs after constrained reparameterization**: the Hessian lives in the unconstrained space; forgetting the Jacobian chain rule delivers SEs on the working scale — a widespread silent bug. Make transform objects own the delta method. Likewise, Bayesian priors on transformed parameters require log-Jacobian terms in the posterior.
- **Parameters at boundaries** (variance pile-up in UC models, GARCH persistence near 1, near-unit roots): unconstrained reparameterizations send working parameters to ±∞ and quasi-Newton curvature collapses — detect divergence, report boundary diagnostics honestly, and quarantine near-unit-root accuracy expectations in tests.
- **Lyapunov/Sylvester via vec-Kronecker is O(k^6) and ill-conditioned** near the unit circle — Bartels–Stewart/Hammarling/doubling only; the real-Schur 2x2 block handling is where implementations go subtly wrong, so validate against SLICOT/scipy on eigenvalue-near-one cases.
- **Missing observations must skip the measurement update** (or drop rows in sequential processing), never be imputed as zeros or handled by inflating H — and steady-state gain freezing must be disabled and reset around missing patches and time-varying system matrices.
- **Complex-step differentiation silently breaks on non-holomorphic operations** (abs, max, value comparisons, mid-computation `.re` extraction): audit every kernel used with complex scalars, and property-test complex-step against analytic gradients wherever both exist.
- **Particle filter weights must live in log space** with log-sum-exp normalization and systematic resampling; naive normalized weights underflow within tens of periods, and unseeded resampling breaks pseudo-marginal MCMC validity.
- **Time-index alignment must fail loudly**: pandas quarter anchoring (Q-DEC vs Q-MAR), week anchoring, and end-vs-start-of-period stamping silently shift data one period when mismatched — one period of misalignment reverses Granger-causality conclusions and is invisible in output.
- **PyO3/GIL discipline**: any Python-object access inside `allow_threads` sections, or Python callbacks invoked from rayon workers, causes deadlocks or unsoundness — keep the boundary thin, poll atomics for cancellation, and re-acquire the GIL only on a single coordinator thread.
- **Test the released wheel artifact, not just the development build**: abi3, LTO, stripped symbols, and feature-flag differences between `maturin develop` and release wheels have shipped broken binaries in well-run projects; numpy oldest and newest must both be in the install-test matrix.
- **EM implementations must assert monotone log-likelihood every iteration in debug builds** — a decreasing EM objective is the earliest and often only symptom of an E-step covariance bug, and acceleration schemes (SQUAREM) can mask it if unchecked.

## Dependencies and shared infrastructure

This module is the provider side of the ownership map. It **exposes** to every other module:

- The Philox-based reproducible parallel RNG, SeedSequence-style seeding, and the `par_replicate` substream contract (consumed by every module).
- The unified forecast object (point/interval/density/path) — consumed by every model family and by forecasting-evaluation.
- The golden-value validation harness, tolerance policy, and benchmark infrastructure — every module's fixtures run through it.
- The linear-Gaussian state-space engine (exact diffuse, sequential filtering, DK2002 and precision-based simulation smoothers, EM) — consumed by ARIMA/UC, multivariate (DFM, TVP-VAR, MF-VAR), bayesian, and nowcasting modules.
- The bootstrap/resampling engine (all schemes and block-length selection) — consumed by every module that reports bootstrap inference; the LP module and forecasting-evaluation are heavy users.
- HAC/long-run-variance/fixed-b/EWC inference with the single library-wide default policy.
- The typed IRF result object and generalized-IRF engine — consumed by multivariate, identification, and LP.
- The factor-model estimation core — the multivariate module builds the single DFM implementation on it; nowcasting consumes that DFM.
- The quantile-regression solver, critical-value engine, innovation-distribution zoo, deterministic-terms toolkit, temporal disaggregation utilities, and the Haar-rotation/restriction-algebra kernel (identification builds SVAR restriction logic on the kernel; bayesian supplies priors/samplers on top of the simulation smoothers and nuts-rs embedding).
- The time-index/calendar/frequency engine, missing-data policy, and the real-time vintage data store — nowcasting builds its release-calendar/news layer on the vintage store; forecasting-evaluation uses it for honest out-of-sample exercises.
- The exogenous-regressor (covariate) contract — every model family that accepts covariates (regARIMA, VARX, GARCH-X, LP controls, MIDAS, ML pipelines) ingests them through the same aligned, leakage-checked interface, and forecasts with them through the same known-future/scenario/auxiliary-forecast distinction.
- The optimizer suite (quasi-Newton, derivative-free, trust-region, EM, multistart) and the reparameterization/inference machinery.

It **consumes** (or coordinates with) from other modules:

- **diagnostics** — owns STL/MSTL and the X-13ARIMA-SEATS wrapper. This module supplies only the X-13 binary-distribution packaging workstream (x13binary model) and the wheel/conda plumbing; the wrapper logic, spec-file generation, and the native STL fallback live in diagnostics.
- **forecasting-evaluation** — owns forecast-comparison tests (DM/GW/MCS/SPA/Reality Check), density-forecast evaluation, forecast combination, and conformal prediction; the unified forecast object defined here is shaped to feed those consumers.
- **multivariate** — owns the single DFM implementation and Granger-causality tooling built on this module's factor core and SSM engine.
- **identification** — owns the unified structural-VAR module (frequentist and Bayesian backends) built on the Haar-rotation/restriction-algebra kernel.
- **ML** — owns penalized-regression solvers and time-series cross-validation; this module does not duplicate them.
- **nowcasting** — owns MIDAS weighting machinery and the vintage/release-calendar/news layer built on the vintage store.
- **LP module** — owns everything local-projection, consuming the bootstrap engine, HAC policy, and IRF object.

## Validation gallery

- **NumPy Philox golden vectors** — raw uint64 output bit-for-bit identical to `numpy.random.Philox`, locked in CI.
- **NIST StRD certified regression values** — OLS results to certified digits.
- **Box–Jenkins airline and Series A–G** — ARIMA estimates and logliks matching statsmodels/gretl at tiered tolerances.
- **Bollerslev–Ghysels DM/USD + Fiorentini–Calzolari–Panattoni benchmark** — GARCH(1,1) parameters, loglik, and QMLE standard errors matching the published FCP benchmark.
- **Durbin–Koopman (2012) Nile examples** — exact-diffuse filter/smoother output and loglik matching KFAS numerically and DK2012 table values.
- **Lütkepohl (2005) West German investment/income/consumption VAR** — coefficient, IRF, and FEVD tables from the book reproduced.
- **Kilian (2009) oil VAR** — structural IRFs reproduced (jointly with the identification module).
- **Nelson–Plosser unit-root dataset** — test statistics against MacKinnon response-surface critical values.
- **Stock–Watson factor datasets** — factor estimates and criteria (Bai–Ng 2002) matching published results and R dfms.
- **Hansen (1999, REStat) grid bootstrap** — published near-unit-root AR confidence intervals reproduced.
- **SLICOT SB03MD/SB02MD and scipy** — Lyapunov/Riccati solutions on eigenvalue-near-one test matrices at reference accuracy.
- **Moré–Garbow–Hillstrom (1981) test set** — optimizer convergence behavior matching scipy reference runs.
- **Higham nearest-correlation test matrices** — projections matching R `Matrix::nearPD`.
- **Bańbura–Modugno (2014) DFM-EM** — nowcasting estimates matching published results and R nowcasting/dfms.
- **Stan cross-checks (eight-schools, small BVAR)** — embedded nuts-rs posteriors matching Stan, plus SBC rank-uniformity passes.
- **Kim–Shephard–Chib (1998) stochastic volatility** — particle-filter and MCMC estimates matching published results.
