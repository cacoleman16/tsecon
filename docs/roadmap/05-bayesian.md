# Module 05 — Bayesian Time Series

> Part of the time series econometrics library roadmap. Master plan: [ROADMAP.md](../../ROADMAP.md).

**This module is the library's Bayesian macroeconometrics engine: the full reduced-form BVAR stack (Minnesota through hierarchical Giannone-Lenza-Primiceri priors, global-local shrinkage, stochastic volatility, time-varying parameters, regime switching, VECM, panel, mixed-frequency, and large systems), together with the posterior-sampling infrastructure — composable Gibbs blocks, adaptive Metropolis, HMC/NUTS, SMC, a validated marginal-likelihood layer, and modern convergence diagnostics — that every Bayesian model in the library, including the identification module's Bayesian structural backends, runs on. Its defining commitment is correctness: only corrected samplers ship (Del Negro-Primiceri 2015; the Carriero-Chan-Clark-Marcellino 2022 corrigendum), and every sampler passes Geweke (2004) joint-distribution tests and simulation-based calibration in continuous integration.**

## Purpose and scope

The module covers everything reduced-form Bayesian in multivariate time series: conjugate and hierarchical BVAR priors with closed-form marginal likelihoods, dummy-observation machinery, shrinkage families (SSVS, horseshoe, Normal-Gamma, Dirichlet-Laplace), stochastic volatility in common-factor, Cholesky, factor, and outlier-robust forms, TVP and Markov-switching dynamics, steady-state and error-correction parameterizations, panel and factor-augmented extensions, mixed-frequency state-space VARs, conditional forecasting, and Bayesian model averaging. Around the models sits the posterior-computation platform: a composable Gibbs-block scheduler, adaptive random-walk Metropolis and slice samplers, gradient-based NUTS (adopting the nuts-rs engine per the architecture ruling), particle Gibbs with ancestor sampling, sequential Monte Carlo, a marginal-likelihood suite that refuses to ship known-broken estimators, and Vehtari et al. (2021)-grade convergence diagnostics on by default.

The primary users are macroeconomic forecasters and policy institutions — the central-bank workflow of hierarchical-prior BVARs, steady-state models, conditional forecasts, and real-time density forecasting is the design center — plus applied researchers estimating TVP, regime-switching, and large-N systems, and methodologists who need a trustworthy sampler platform rather than a model zoo. The Rust core matters here more than anywhere else in the library: hyperparameter search, recursive out-of-sample exercises with hundreds of re-estimations, SMC, and prior-sensitivity sweeps are exactly the workloads that are painfully slow in R and Matlab and embarrassingly parallel given reproducible RNG substreams.

Relative to the rest of the library, this module is a supplier as much as a product. The identification module owns all structural restriction and rotation logic (sign, zero+sign, narrative, penalty-function, explicit structural priors, proxies, robust Bayes, structural scenario analysis) and consumes this module's reduced-form posteriors, priors, samplers, and marginal likelihoods through its Bayesian backend. The foundations layer supplies the linear-Gaussian state-space engine whose simulation smoothers (Carter-Kohn, Durbin-Koopman, precision-based) this module drives inside its Gibbs blocks, plus the parallel RNG, the IRF engine, and the forecast object. Forecast-density scoring and combination live in forecasting-evaluation; local projections — including Bayesian LPs — live in the LP module.

## Where existing tools fall short

- statsmodels has essentially no Bayesian VAR capability at all; Python users today stitch together PyMC (poorly suited to structural-identification workflows and simulation smoothers) or abandoned one-off GitHub repos. There is no Python home for BVARs.
- R `BVAR` (Kuschnig-Vashold) implements only the Giannone-Lenza-Primiceri hierarchical NIW model: no stochastic volatility, no TVP, no sign/zero/narrative/proxy identification, no VECM.
- R `bvartools` covers SSVS, BVEC, and some SV, but is slow (R loops around C++ kernels), has limited identification, no hierarchical hyperparameter estimation, and sparse guidance on which model to use when.
- R `bsvars` / `bsvarSIGNs` (Woźniak) are fast and modern but deliberately narrow (structural-identification focus): no GLP hierarchical priors, limited forecasting and conditional-forecast infrastructure, no mixed frequency, no panel.
- Matlab BEAR (ECB) is the broadest single toolbox but is Matlab-licensed, a GUI/script hybrid, slow for Monte Carlo at scale, hard to embed in pipelines, and thin on diagnostics (single-chain culture, no R-hat or ESS by default).
- Dynare's BVAR tools are legacy Sims/Zha code oriented toward DSGE comparison; MS-SBVAR is powerful but nearly unmaintained and notoriously hard to use.
- No existing package implements the Carriero-Clark-Marcellino (2022) corrigendum consistently — corrected and uncorrected large-BVAR-SV samplers circulate side by side, and users cannot tell which they are running.
- Convergence diagnostics are an afterthought throughout econometrics tooling: single chains, no rank-normalized R-hat, no bulk/tail ESS, no joint-distribution sampler tests in CI. ArviZ-level diagnostics integrated with econometric models do not exist.
- Marginal-likelihood computation is scattered and often wrong in practice: harmonic-mean estimators are still common in circulating code, and no package offers a unified, validated layer (Chib, bridge sampling, cross-entropy, SMC) with latent-variable-aware warnings.
- Conditional forecasting and structural scenario analysis are missing from every open-source package except (partially) BEAR — yet these are the features central-bank users ask for first.
- Speed: R and Matlab implementations make hyperparameter search, real-time out-of-sample exercises, SMC, and prior-sensitivity analysis painfully slow, and none exploit multi-core parallelism with reproducible RNG streams.
- Documentation teaching which prior, model, and identification scheme to use when — including the known pathologies (Haar-prior informativeness, Cholesky-SV ordering dependence, TVP overfitting) — exists only in handbook chapters and folklore, not in any package.

## Inventory

Structural-identification items from the research inventory (sign, zero+sign, narrative, penalty-function, Baumeister-Hamilton, proxy, robust Bayes, structural scenario analysis) are owned by the identification module and appear under Dependencies below, as do the foundations-owned state-space simulation smoothers, the parallel RNG layer, the batched IRF/forecast engine, density-forecast evaluation, prediction pools, and Bayesian local projections.

### Tier 1 — Core (v1-blocking)

**Conjugate and classic priors**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Minnesota/Litterman prior BVAR | Workhorse shrinkage prior toward univariate random walks (white noise for stationary data), with tightness, cross-variable, and lag-decay hyperparameters. Default starting point for any macro forecasting VAR. | Low | Ship both the original fixed-Sigma Litterman form (equation-by-equation ridge) and the modern NIW-embedded version. Scale factors use residual variances from univariate AR fits — document exactly which convention (AR(1) vs AR(p), with/without intercept); packages differ and results are sensitive. Refs: Litterman (1986), Doan-Litterman-Sims (1984). Validate: R `BVAR` and Matlab BEAR with identical hyperparameters; conjugate-case posterior means to machine precision. |
| Natural-conjugate Normal-inverse-Wishart (NIW) BVAR | Fully analytical posterior under Kronecker prior variance: matric-variate-t coefficients, closed-form marginal likelihood. Computational backbone for hyperparameter optimization and large systems. | Low | Never form the NK x NK covariance: matrix-normal identities, Cholesky log-determinants, QR/Cholesky solves. Inverse-Wishart parameterization conventions (scale vs rate; mean exists only if dof > n+1) are the single most common source of cross-package discrepancies. Refs: Kadiyala-Karlsson (1997), Karlsson (2013 Handbook chapter). Validate: closed-form ML and moments vs BEAR and vs brute-force numerical integration on a 2-variable toy model. |
| Independent Normal-Wishart BVAR (Gibbs) | Breaks the Kronecker restriction so coefficient prior variances differ arbitrarily across equations (true asymmetric Minnesota); two-block Gibbs. | Low | Coefficients given Sigma is a large GLS draw — O(N^3 p^3) if done naively; implement the per-equation factorization. Ref: Koop-Korobilis (2010 primer). Validate: `bvartools` and BEAR; Gibbs output must match the analytic NIW special case when priors are made Kronecker. |
| Dummy-observation priors: sum-of-coefficients and single-unit-root | Artificial observations expressing unit-root and cointegration beliefs; standard components of the Sims-Zha and GLP prior stacks. | Low | Implement as data augmentation appended to (Y, X), preserving conjugacy; unit-test exact algebraic equivalence with the direct prior-moment formulation. Trap: the single-unit-root dummy interacts with the intercept and can dominate in small samples — document scaling by initial-condition means. Refs: Doan-Litterman-Sims (1984), Sims (1993), Sims-Zha (1998). Validate: GLP (2015) replication files. |

**Hierarchical priors and hyperparameter selection**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Hierarchical prior with marginal-likelihood hyperparameter selection (Giannone-Lenza-Primiceri 2015) | Treats Minnesota/sum-of-coefficients/dummy-initial tightness as hyperparameters with hyperpriors, exploiting the closed-form NIW marginal likelihood. The modern default for BVAR estimation. | Medium | RW-MH on transformed (log/logit) hyperparameters, or BFGS maximization for the empirical-Bayes mode. Traps: keep ALL normalizing constants (they matter when comparing hyperparameter values), guard log-dets with Cholesky, impose the GLP gamma hyperpriors exactly. Ref: Giannone-Lenza-Primiceri (2015 ReStat). Validate: replicate GLP Table 1 and forecast RMSEs via R `BVAR` and the authors' Matlab code. |

**Large BVARs and scalability**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Large BVAR à la Bańbura-Giannone-Reichlin (2010) | Conjugate BVAR on 20-130 variables with shrinkage tightened as dimension grows; large BVARs beat factor models for forecasting and structural analysis. | Medium | Everything via the dummy-observation/NIW closed form; tightness by their in-sample-fit targeting or GLP marginal likelihood. Use Cholesky downdating, float64 throughout; keep the posterior coefficient covariance factored — forming it explicitly is the memory bottleneck. Ref: Bańbura-Giannone-Reichlin (2010 JAE). Validate: their published RMSE tables (FRED-based dataset) and BEAR's large-VAR mode. |

**Stochastic volatility and fat tails**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| BVAR with Cholesky multivariate stochastic volatility (Cogley-Sargent / CCM-style) | Full multivariate SV via triangular factorization with independent log-volatility random walks — the standard BVAR-SV specification for medium VARs. | High | Gibbs blocks: coefficients given vols (GLS), free covariance elements, log-vols via KSC mixture + FFBS or precision sampler. CRITICAL: results depend on variable ordering through the Cholesky factorization; offer the order-invariant alternative of Chan-Koop-Yu (2024 JBES) or common/factor SV, and document the issue prominently. Refs: Cogley-Sargent (2005), Clark (2011 JBES), Carriero-Clark-Marcellino (2015). Validate: Clark (2011) replication and BEAR's BVAR-SV. |

**Time-varying parameters and regime switching**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| TVP-BVAR with SV (Primiceri 2005, Del Negro-Primiceri 2015 correction) | Drifting coefficients, drifting contemporaneous relations, and stochastic volatility — the canonical model for evolving monetary transmission. | High | Multi-block Gibbs: coefficient states (FFBS or precision sampler), simultaneous-relation states, SV via KSC mixture, innovation covariances. CRITICAL: the original Primiceri (2005) block ordering is incorrect with the KSC mixture — draw mixture indicators in the order established by Del Negro-Primiceri (2015 ReStud corrigendum); implement only the corrected sampler. Expose the training-sample prior calibration exactly. Validate: replicate Primiceri's Figures 1-4 via the Del Negro-Primiceri corrected code. |

**Structural identification (BSVAR) — baseline output**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Recursive (Cholesky) identification with full posterior IRF/FEVD/HD suite | Baseline structural analysis: impulse responses, forecast-error variance decompositions, and historical decompositions computed draw-by-draw from the reduced-form posterior. Day-one output for every user. | Low | Compute IRFs per posterior draw via the foundations IRF engine (companion-form recursions); report pointwise quantile bands AND clearly labeled joint bands (Tier 3 item). Historical decompositions must add up: unit-test that shock contributions plus the initial-condition component reconstruct the data exactly per draw. Restriction/rotation logic beyond the recursive baseline lives in the identification module. Ref: Kilian-Lütkepohl (2017 textbook). Validate: IRFs against statsmodels/vars (frequentist point) and BEAR (Bayesian bands). |

**State-space samplers**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| KSC/Omori mixture sampler for stochastic volatility | Linearizes the SV model by approximating the log-chi-squared error with a discrete normal mixture, enabling Gaussian FFBS/precision draws of log-volatilities plus mixture-indicator Gibbs steps. | Medium | The mixture is an APPROXIMATION: offer the Kim-Shephard-Chib importance-reweighting correction for exactness; default to the 10-point Omori et al. (2007) mixture. Handle near-zero residuals with the standard offset constant carefully — document its value, results can be sensitive. Refs: Kim-Shephard-Chib (1998 ReStud), Omori-Chib-Shephard-Nakajima (2007 JoE). Validate: R `stochvol` posterior moments. |

**MCMC infrastructure**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Adaptive MCMC toolkit: blocked Gibbs, adaptive RW-MH, slice sampling | The generic infrastructure layer: composable Gibbs blocks, Haario-style adaptive RW-MH with Robbins-Monro scaling targeting 0.234/0.44 acceptance, slice samplers for awkward univariate conditionals (dof, hyperparameters). | Medium | Adaptation must satisfy diminishing adaptation (freeze or decay after burn-in) or ergodicity breaks — document and enforce. Design the block API so users can swap/extend conditionals without touching the scheduler; this composability makes the library a platform rather than a model zoo. Refs: Haario-Saksman-Tamminen (2001), Roberts-Rosenthal (2009 JCGS), Neal (2003). |

**Diagnostics and posterior checks**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Convergence diagnostics: rank-normalized split R-hat, bulk/tail ESS, multi-chain defaults | Modern MCMC quality control (Vehtari et al. 2021) integrated by default — a capability econometrics packages almost universally lack. | Medium | Default to 4 chains from overdispersed starts; refuse to report posterior summaries without R-hat unless the user opts out; per-function ESS for IRFs (functions of draws), not just raw parameters; MCSE and Geyer (1992) initial-monotone-sequence ESS. Ref: Vehtari-Gelman-Simpson-Carpenter-Bürkner (2021 Bayesian Analysis). Validate: numerically against ArviZ and the R `posterior` package on shared draws. |
| Geweke "getting it right" simulator-consistency tests as CI | Joint-distribution tests detecting coding errors in any Gibbs sampler by comparing marginal-conditional and successive-conditional simulators — the best defense against silent-wrong-posterior failures (CCM 2019 and Primiceri 2005 both shipped with such bugs). | Medium | Run Geweke (2004 JASA) tests plus simulation-based calibration (Talts et al. 2018) in continuous integration on tiny models, for every sampler in the library. An internal-quality item with roadmap status: it is what will make this library more trustworthy than the incumbents. |

### Tier 2 — Standard (expected of a serious library)

**Cointegration and steady state**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Steady-state BVAR (Villani 2009) | Parameterizes the VAR in terms of its unconditional mean so informative priors go directly on steady states (inflation target, trend growth). Heavily used at central banks; long-horizon forecasts converge to interpretable values. | Medium | Three-block Gibbs on (dynamics, steady state, Sigma); the steady-state draw is GLS on deterministic terms filtered through the lag polynomial. Trap: the conditional requires the lag polynomial at 1 — near-unit-root draws make it ill-conditioned; guard with stationarity truncation or a proper prior. Ref: Villani (2009 JAE). Validate: R `mfbvar` steady-state implementation and published Riksbank forecast exercises. |
| Bayesian VECM (Koop-León-González-Strachan 2010) | Error-correction model with a proper prior on the cointegrating space (Grassmann manifold), avoiding the pathologies of flat priors on normalized beta. | High | Linear normalization with flat priors puts infinite mass at the boundary; use the KLS parameterization with orthonormal beta and a matrix angular central Gaussian prior, via parameter-expanded Gibbs. Rank selection via marginal likelihoods (Savage-Dickey where available). Refs: Koop-León-González-Strachan (2010 JoE); Villani (2005). Validate: R `bvartools` BVEC on the Lütkepohl datasets. |

**Global-local shrinkage and variable selection**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| SSVS prior for VARs (George-Sun-Ni 2008) | Spike-and-slab mixture on each VAR coefficient (optionally covariance elements) giving posterior inclusion probabilities; data-driven restriction search in medium VARs. | Medium | Gibbs with Bernoulli indicator draws; mixing degrades badly beyond ~15 variables — document and cap. The spike/slab scale ratio drives everything; provide the semiautomatic default scaled by OLS standard errors as in the original paper. Ref: George-Sun-Ni (2008 JoE). Validate: `bvartools` SSVS on the original 6-variable example. |

**Stochastic volatility and fat tails**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| VAR with common stochastic volatility (Carriero-Clark-Marcellino 2016) | A single scalar volatility factor scales the whole covariance matrix, preserving Kronecker/conjugate structure — the cheapest time-varying volatility for big BVARs, order-invariant by construction. | Medium | Conditional on the volatility path, coefficients stay conjugate (GLS-weighted dummy observations); the log-volatility path via independence-MH or a KSC-mixture step on a univariate state equation. Ref: Carriero-Clark-Marcellino (2016 JAE). Validate: their replication files; shutting off SV must recover the homoskedastic NIW posterior exactly. |

**Forecasting tools**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Conditional forecasts (Waggoner-Zha 1999; Bańbura-Giannone-Lenza 2015) | Forecasts conditional on hard or soft paths for subsets of variables (e.g., oil price follows the futures curve), with full parameter and shock uncertainty. Non-negotiable for policy institutions. | Medium | Waggoner-Zha Gibbs sampling of constrained shocks per parameter draw (hard conditions); soft conditions via truncated distributions; for large systems the Kalman-based simulation-smoother approach of Bańbura-Giannone-Lenza (2015 IJF). Trap: distinguish "conditional on a future path with all shocks free" from "specific shocks drive the path" — the latter is structural scenario analysis (identification module). Validate: BEAR's conditional-forecast module. |
| Entropic tilting and soft conditioning of predictive densities | Exponential tilting imposes moment conditions (survey means, nowcasts) on posterior-predictive simulations. Scoring and calibration evaluation of the resulting densities lives in forecasting-evaluation. | Medium | Tilting weights solved by convex optimization. Predictive simulations must integrate parameter uncertainty — never tilt plug-in predictive densities. Refs: Robertson-Tallman-Whiteman (2005); Krüger-Clark-Ravazzolo (2017 JBES). Validate: Krüger-Clark-Ravazzolo published applications; scores cross-checked via R `scoringRules` through the forecasting-evaluation module. |

**Factor and panel extensions**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Bayesian FAVAR (Bernanke-Boivin-Eliasz 2005) | Factor-augmented VAR with joint Bayesian estimation of factors and VAR dynamics via Gibbs — structural analysis with hundreds of information variables. | High | Gibbs alternates factor draws (linear state space given loadings) with loadings and VAR blocks; identify factors via loading normalizations (upper-triangular block = identity). Also offer the two-step PCA variant (much faster, nearly identical IRFs), consuming the foundations factor-estimation core. Ref: Bernanke-Boivin-Eliasz (2005 QJE). Validate: their replication (monetary policy IRFs on 120 series). |

**Model averaging and comparison**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Bayesian model averaging and dynamic model averaging over VARs | Marginal-likelihood-weighted averaging across lag lengths, variable sets, and priors; DMA with forgetting-factor time-varying weights for real-time forecasting. Score-based prediction pools live in forecasting-evaluation. | Medium | Static BMA is nearly free given closed-form conjugate MLs — 2^K enumeration for small K, MC3 search beyond. DMA per Raftery-Kárný-Ettler (2010) and Koop-Korobilis (2012 IER). Trap: BMA weights are extremely sensitive to prior scale constants — document Lindley-paradox behavior. Validate: Koop-Korobilis published inflation-forecasting results. |

**Marginal likelihood and model comparison**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Marginal likelihood suite: Chib, Chib-Jeliazkov, bridge sampling, Geweke MHM, cross-entropy | A unified, honest marginal-likelihood layer: analytic for conjugate models; Chib (1995) for Gibbs; Chib-Jeliazkov (2001) for MH blocks; bridge sampling as the robust general method; truncated-Gaussian MHM and Chan-Eisenstat cross-entropy IS for latent-variable models. | High | NEVER ship the raw Newton-Raftery harmonic mean (infinite variance); Geweke's truncated MHM only with truncation diagnostics. CRITICAL: for latent-state models (SV, TVP), Chib-style estimates on the conditional likelihood are wrong — the integrated likelihood is required (Chan-Eisenstat 2015 JAE; Chan-Grant 2015 documents the biases). Bridge sampling per Meng-Wong (1996), Gronau et al. (2017). Validate: every estimator against the closed-form conjugate ML. |
| Savage-Dickey density ratios and posterior odds tools | Cheap Bayes factors for point restrictions nested in a sampled model (e.g., Granger-causality restrictions, cointegration rank in some parameterizations) from a single MCMC run. | Low | Evaluate Rao-Blackwellized posterior and prior densities at the restriction — conditional densities averaged over draws, never kernel estimates when a conjugate conditional exists. Trap: the prior density at the point must be exactly the sampling prior, including truncations. Refs: Verdinelli-Wasserman (1995); Koop (2003 textbook). |

**Diagnostics and posterior checks**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Prior and posterior predictive checks | Prior predictive simulation (what a prior implies for data paths, IRFs, spectra) and posterior predictive p-values for chosen statistics. Doubles as the pedagogical layer teaching users what priors do. | Low | Cheap given fast simulation; the design task is a good default battery of test statistics per model class. Prior-implied IRF distributions should be a first-class plot for any set-identified analysis (the Baumeister-Hamilton lesson), exposed through the identification module. Refs: Gelman-Meng-Stern (1996); Geweke (2004). |

**MCMC infrastructure**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Stationarity truncation and explosive-draw policy | Optional rejection of explosive companion-matrix draws (standard in TVP-VARs following Cogley-Sargent). Must be a documented, explicit choice: truncation changes the prior, the posterior, and the marginal likelihood. | Medium | Accept-reject within the state/coefficient draw with an iteration cap and a warning on low acceptance — near-unit-root data can make the truncated region tiny, silently biasing results (a known replication headache in Primiceri-style models). Report the fraction rejected. Refs: Cogley-Sargent (2005) appendix; Koop-Potter (2011). |

### Tier 3 — Advanced (differentiators)

**Hierarchical priors and hyperparameter selection**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Prior on the long run (Giannone-Lenza-Primiceri 2019) | Replaces ad hoc sum-of-coefficients dummies with a disciplined prior on the long-run behavior of user-specified linear combinations (great ratios). | Medium | Dummy observations built from a user-supplied loading matrix H; per-combination tightness folds into the GLP hierarchical machinery. Trap: H must be full rank and results are sensitive to row scaling — normalize and document. Ref: Giannone-Lenza-Primiceri (2019 ReStat). Validate: their replication files (5-variable US system). |
| Asymmetric conjugate prior (Chan 2022) | Reparameterized (equation-by-equation, triangularized) conjugate prior allowing genuine cross-variable Minnesota shrinkage while retaining a closed-form marginal likelihood — GLP-style hyperparameter optimization at 100+ variable scale. | Medium | Works in the structural-form recursive parameterization; each equation gets an independent normal-inverse-gamma posterior, ML is a product of univariate t densities. Trap: results depend on variable ordering through the triangularization — expose ordering and document. Ref: Chan (2022 Journal of Econometrics). Validate: Chan's Matlab code for the 25-variable application. |

**Global-local shrinkage and variable selection**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Horseshoe prior BVAR | Global-local shrinkage with heavy-tailed local scales: aggressive shrinkage of noise with minimal shrinkage of signal, no discrete search. Best-performing automatic shrinkage prior in several forecasting horse races. | Medium | Local/global scales via the Makalic-Schmidt (2016) inverse-gamma auxiliary representation or slice sampling; coefficient draws via the fast Bhattacharya-Chakraborty-Mallick (2016) algorithm when coefficients outnumber observations. Trap: half-Cauchy scales produce occasional huge draws — work in logs and cap without distorting the posterior. Refs: Carvalho-Polson-Scott (2010); Follett-Yu (2019), Cross-Hou-Poon (2020 IJF). Validate: Cross-Hou-Poon replication results. |
| Normal-Gamma and Dirichlet-Laplace shrinkage priors | Alternative global-local families (Normal-Gamma per Huber-Feldkircher grouping; Dirichlet-Laplace), shipped behind a unified "shrinkage family" interface with the horseshoe so users can horse-race priors. | Medium | Normal-Gamma requires a robust GIG sampler (Devroye 2014 / Hörmann-Leydold) — naive GIG samplers fail silently for extreme parameter values, a classic bug. Refs: Griffin-Brown (2010); Huber-Feldkircher (2019 JBES) for the VAR grouping (own lags vs cross lags vs deterministics). Validate: R `bayesianVARs` (Gruber-Kastner). |

**Large BVARs and scalability**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Corrected triangular algorithm for large BVAR-SV (CCM 2019 + 2022 corrigendum) | Equation-by-equation estimation making N-large VARs with full SV feasible (O(N^4) to per-equation) — but the original 2019 sampler conditioned incorrectly and yields the wrong posterior. | High | Implement ONLY the corrected algorithm: Carriero-Clark-Marcellino (2019 JoE) with the Carriero-Chan-Clark-Marcellino (2022 JoE) corrigendum — the valid conditional must account for later equations' residuals depending on earlier equations' coefficients. Much circulating Matlab code still runs the invalid sampler: a differentiating correctness point. Validate: the corrigendum's replication files and brute-force full-system Gibbs on a small model. |

**Stochastic volatility and fat tails**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Factor stochastic volatility BVAR (Kastner-Huber 2020) | Volatility driven by a few latent SV factors plus idiosyncratic SV — scalable, order-invariant, captures volatility comovement. The serious alternative to Cholesky SV for large systems. | High | Interweave loadings/factor draws with univariate SV updates; use ASIS for the loadings-scale identification. Traps: factor sign/scale identification (fix loading signs or post-process) and number-of-factors selection. Refs: Kastner-Huber (2020 Journal of Forecasting); Kastner-Frühwirth-Schnatter (2014). Validate: R `factorstochvol` + `bayesianVARs`. |
| Fat tails: t-errors, outlier states, and SV-with-outliers (SVO) | Student-t measurement errors via latent scale mixtures, plus the Stock-Watson (2016) outlier-state formulation adapted to VARs. Essential post-COVID: SV alone catastrophically misreads March-2020-type observations. | Medium | t-errors: inverse-gamma latent scales; the dof step is slow-mixing for small dof — MH on a log grid or slice sampler. SVO: discrete-mixture outlier indicator with Bernoulli occurrence probability. Refs: Jacquier-Polson-Rossi (2004); Chan (2020 JBES); Carriero-Clark-Marcellino-Mertens (2022 ReStat). Validate: CCMM (2022) replication files. |
| COVID volatility scaling (Lenza-Primiceri 2022) | Explicit treatment of pandemic observations: common volatility scale factors on the COVID months with decay back to normal. Cheap, transparent, and what many policy institutions actually adopted. | Low | Three parameters multiply residual standard deviations; conjugacy preserved conditional on the scales (MH or grid step); supports the "downweight, don't drop" recommendation. Ref: Lenza-Primiceri (2022 JAE). Validate: their replication files (large US monthly VAR). |

**Time-varying parameters and regime switching**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Non-centered TVP parameterization with shrinkage on state variances | Frühwirth-Schnatter-Wagner reparameterization turning "does this coefficient drift?" into variable selection on square-root state-innovation scales, enabling shrinkage toward constant parameters. Fixes chronic TVP-VAR overfitting. | High | The non-centered form makes sqrt(state variance) a regression coefficient with sign ambiguity — randomize signs each sweep to avoid boundary sticking. Combine with the dynamic horseshoe (Kowal-Matteson-Ruppert 2019 JRSS-B) as the frontier variant. Refs: Frühwirth-Schnatter-Wagner (2010 JoE); Bitto-Frühwirth-Schnatter (2019 JoE); Huber-Koop-Onorante (2021 JBES). Validate: R `shrinkTVP`. |
| Markov-switching BVAR/BSVAR (Sims-Zha 2006) | Discrete regime changes in coefficients and/or variances with Markov transitions; the variance-switching version doubles as an identification device. | High | Hamilton filter + backward regime sampling; label switching handled by permutation sampling or identification constraints (e.g., ordered variances) — post-hoc relabeling is fragile. Posterior typically multimodal: multi-chain diagnostics essential; SMC (Bognanni-Herbst 2018) is the robust estimation route. Refs: Sims-Zha (2006 AER), Sims-Waggoner-Zha (2008 JoE). Validate: Sims-Waggoner-Zha replication and Dynare's MS-SBVAR module. |
| Fast TVP approximations: forgetting factors / dynamic model switching (Koop-Korobilis) | Kalman-filter TVP-VAR with forgetting factors and exponentially weighted covariance — no MCMC, seconds on large systems. The pragmatic real-time tool when full Bayes is too slow. | Medium | Pure filtering recursions; "estimation" is a grid search over forgetting factors scored by predictive likelihood. An excellent speed benchmark for the library. Ref: Koop-Korobilis (2013 JoE). Validate: their published Matlab code (TVP-VAR with DMA over VAR dimensions). |
| Regime-dependent and threshold BVARs (threshold/smooth-transition) | Nonlinearity via observed-variable thresholds (TVAR) or smooth transition (ST-VAR): state-dependent fiscal multipliers, financial-stress asymmetries. | High | Threshold and delay parameters have irregular likelihoods — griddy Gibbs or MH on a fine grid; regime-specific IRFs are generalized IRFs (Koop-Pesaran-Potter 1996) requiring forward simulation, cheap on the fast engine. Refs: Chen-Lee (1995); Auerbach-Gorodnichenko (2012). Validate: regime-dependent multipliers against published ST-VAR replications. |

**Structural identification (BSVAR) — posterior summaries**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Joint/simultaneous credible bands for IRFs (Inoue-Kilian 2022) | Pointwise quantile bands understate joint uncertainty about the IRF as a curve; joint credible sets fix the well-documented miscoverage. Should be the labeled default alternative in plots. | Medium | Implement (a) Inoue-Kilian (2022 JoE) HPD sets over whole-IRF-vector draws and (b) calibrated sup-t bands (adjust the pointwise level until joint coverage attains 1-alpha). Cheap given stored draws. Also relevant: Plagborg-Møller (2019 QE) for direct Bayesian inference on IRFs. Validate: coverage by simulation against known DGPs. |

**Forecasting tools**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Mixed-frequency BVAR (Schorfheide-Song 2015) | Monthly/quarterly VAR in state-space form with quarterly series as latent monthly under temporal-aggregation constraints — the Bayesian nowcasting workhorse. | High | Deterministic aggregation rows in the state space; use the precision-based sampler for latent monthly states (much faster than Kalman FFBS at these dimensions). Trap: real-time ragged edges require careful missing-data handling in the filter; release-calendar/news layers come from the nowcasting module. Ref: Schorfheide-Song (2015 JBES). Validate: R `mfbvar` and Schorfheide-Song replication files. |

**Factor and panel extensions**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Bayesian panel VAR with hierarchical pooling (Jarociński 2010) | Cross-country VARs with exchangeable hierarchical priors shrinking unit-specific coefficients toward a common mean, with the pooling degree estimated from data. Standard demand from central-bank multi-country teams. | High | Gibbs on unit coefficients, common mean, and the shrinkage variance (inverse-gamma or half-t hyperprior). Include the Canova-Ciccarelli factor-coefficient approach as an alternative. Refs: Jarociński (2010 JAE); Canova-Ciccarelli (2009 IER). Validate: BEAR's panel VAR module. |

**Model averaging and comparison**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| DSGE-VAR (Del Negro-Schorfheide 2004) — GATED | Uses a DSGE model's implied population moments as dummy observations for a VAR prior, with tightness lambda estimated by marginal likelihood; both a forecasting device and a DSGE misspecification diagnostic. CONDITIONAL: gated on the master plan's separate DSGE-lite extension decision (linear RE solver + SMC). | High | In v1 scope only if the gate opens; regardless, accept a user-supplied state-space or moment function rather than building a DSGE solver here. Posterior over lambda on a grid via closed-form conjugate MLs; structural shocks via the DSGE rotation. Refs: Del Negro-Schorfheide (2004 IER); Del Negro-Schorfheide-Smets-Wouters (2007). Validate: Dynare's dsge-var implementation. |

**State-space samplers**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| ASIS / interweaving strategies for SV and TVP blocks | Ancillarity-sufficiency interweaving alternates centered and non-centered parameterizations within each sweep, dramatically improving mixing of volatility-of-volatility and state-variance parameters where naive Gibbs stalls. | Medium | Essentially free extra steps once both parameterizations are coded; largest gains for SV persistence/variance and TVP state variances. Refs: Yu-Meng (2011 JCGS), Kastner-Frühwirth-Schnatter (2014 CSDA). Validate: mixing (ESS per second) against R `stochvol`, which is built around ASIS. |

**MCMC infrastructure**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| HMC/NUTS for structural and hyperparameter blocks | Gradient-based sampling for non-conjugate blocks: GLP hyperparameters, SV parameter blocks, and the structural-matrix posteriors the identification module drives. Marginalizing states analytically and running NUTS on the rest is often the most robust estimator. | High | Adopt/embed nuts-rs per the architecture ruling. Requires reverse-mode autodiff through Cholesky factorizations and the Kalman filter/banded solvers (manual adjoints for the linear-algebra kernels; the Cholesky pullback is standard but easy to get wrong — test against finite differences). Refs: Hoffman-Gelman (2014 JMLR); Betancourt (2017) for divergence diagnostics. Validate: posterior agreement with Stan on small models. |
| Particle Gibbs with ancestor sampling (PGAS) | Conditional-SMC state draws inside Gibbs for genuinely nonlinear/non-Gaussian state blocks (SV-in-mean, threshold latent states, non-Gaussian measurement). The escape hatch when no linearization exists. | Research-grade | Plain particle Gibbs suffers path degeneracy on long samples; ancestor sampling (Lindsten-Jordan-Schön 2014 JMLR) is essential. Keep particle counts modest (100-500) and monitor update rates of early-sample states. Ref: Andrieu-Doucet-Holenstein (2010 JRSS-B). Validate: univariate SV model against the exact KSC-corrected sampler. |

### Tier 4 — Frontier (research-grade, each gated on a named validation target)

**Frontier model classes**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Shadow-rate and censored-variable BVARs | Treats the policy rate as censored at the ELB with a latent shadow rate, keeping the VAR well-specified through zero-rate periods. Essential for post-2008 US/EA/Japan monetary VARs. | High | Data augmentation: sample the latent shadow rate from its truncated conditional. Trap: high-dimensional truncated-normal sampling needs a robust method (Botev 2017 minimax tilting or carefully monitored Gibbs). Refs: Johannsen-Mertens (2021), Carriero-Clark-Marcellino-Mertens (2021). GATE: reproduce the Carriero et al. replication files. |
| Bayesian nonparametric VARs: BART-VAR and GP-VAR | Conditional-mean nonlinearity via Bayesian Additive Regression Trees or Gaussian-process regressions inside a VAR — asymmetries and tail nonlinearities for growth-at-risk and inflation tail forecasting. | Research-grade | Equation-by-equation BART with triangularized covariance and SV; needs a fast native BART sampler (Chipman-George-McCulloch backfitting or particle) — a significant engineering subproject well suited to Rust. Refs: Huber-Rossini (2022 JoE), Clark-Huber-Koop-Marcellino-Pfarrhofer (2023 JAE); Hauzenberger-Huber-Marcellino-Petz (2024) for GP-VAR. GATE: reproduce the authors' R-code tail-risk forecasting results. |

**Large BVARs and scalability**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Variational Bayes for huge BVARs | Structured mean-field approximations for VARs with hundreds of variables or dense TVP structures where MCMC is infeasible; increasingly used in nowcasting platforms. | High | Natural-gradient coordinate-ascent VI exploiting conjugate blocks; always pair with Pareto-k importance-sampling diagnostics and an MCMC cross-validation mode on subsampled systems — VI understates posterior variance; document this honestly. Refs: Chan-Yu (2022 JoE), Gefang-Koop-Poon (2023). GATE: match Chan-Yu's reported accuracy benchmarks. |
| Bayesian compressed and random-projection VARs | Random-projection compression of the regressor space with BMA over projections — a huge system reduced to many small conjugate problems; strong forecasting record and trivially parallel. | Medium | Draw sparse random projections, estimate small conjugate VARs, average by ML weights; embarrassingly parallel — a natural Rust showcase. Ref: Koop-Korobilis-Pettenuzzo (2019 JoE). GATE: reproduce their macro forecasting application results. |

**MCMC infrastructure**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Sequential Monte Carlo (SMC) samplers | Likelihood-tempering particle samplers handling multimodal posteriors (MS-VARs, sign-restricted sets, DSGE-VAR lambda), yielding the marginal likelihood as a byproduct; embarrassingly parallel — a headline use case for a Rust engine. | High | Adaptive tempering schedule (target ESS ratio), mutation via a few RW-MH or HMC steps per stage, stratified/systematic resampling. Verify the marginal-likelihood byproduct against the conjugate closed form. Refs: Herbst-Schorfheide (2014 JAE; 2015 book), Bognanni-Herbst (2018 JAE), Del Moral-Doucet-Jasra (2006). GATE: reproduce the Herbst-Schorfheide replication. |

**Robust and set-identification frontier**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Identification via heteroskedasticity and non-Gaussianity (Bayesian) | Point identification of the full structural rotation from volatility regime changes (MS-variance SVAR) or non-Gaussian independent shocks — lets users test sign restrictions rather than impose them. This module owns the MS-variance and non-Gaussian reduced-form samplers; the identification module exposes the scheme in its unified API. | Research-grade | Heteroskedasticity: Markov-switching variances with the structural matrix constant across regimes (Lütkepohl-Woźniak 2020 JEDC; Brunnermeier-Palia-Sastry-Sims 2021 AER); shock label/ordering ambiguity remains — normalize by variance ordering. Non-Gaussianity: t- or mixture-distributed independent shocks (Anttonen-Lanne-Luoto 2023; Braun 2023 ReStud). GATE: match R `bsvars` (Woźniak), the only maintained implementation. |

## Frontier watchlist

Source frontier entries not carried as Tier 4 rows above, with their dispositions:

- Robust (multiple-prior) Bayes for set-identified SVARs (Giacomini-Kitagawa 2021 Econometrica; Giacomini-Kitagawa-Read extensions) — identification module owns; this module supplies the reduced-form posterior draws it aggregates over.
- Full Bayesian proxy SVARs and proxies combined with sign restrictions via importance sampling (Arias-Rubio-Ramírez-Waggoner 2021 JoE) — identification module owns.
- Structural scenario analysis with shock-specific conditioning and KL plausibility metrics (Antolín-Díaz, Petrella, Rubio-Ramírez 2021 JME) — identification module owns; consumes this module's conditional-forecast engine.
- Bayesian local projections with VAR-based priors (Ferreira-Miranda-Agrippino-Ricco) and Bayesian smooth LPs (Tanaka 2020) — LP module owns; this module supplies the auxiliary-VAR prior and marginal-likelihood tightness selection.
- Order-invariant large BVAR-SV specifications (Chan-Koop-Yu 2024) — currently a documented alternative inside the Cholesky-SV item; candidate for promotion to a first-class Tier 3 item once the JBES implementation is validated.
- Dynamic shrinkage processes / dynamic horseshoe priors for TVPs (Kowal-Matteson-Ruppert 2019; Huber-Koop-Onorante 2021) — flagged as the frontier variant of the Tier 3 non-centered TVP item.
- SV-with-outliers and pandemic volatility scaling (Carriero-Clark-Marcellino-Mertens 2022; Lenza-Primiceri 2022) — already scheduled at Tier 3.
- Asymmetric conjugate priors at 100+ variables (Chan 2022) — already scheduled at Tier 3.
- Joint (simultaneous) credible sets for IRFs (Inoue-Kilian 2022; Plagborg-Møller 2019) — already scheduled at Tier 3.
- Entropic tilting and soft conditioning toward surveys and nowcasts (Krüger-Clark-Ravazzolo 2017) — already scheduled at Tier 2.
- Mixed-frequency BVARs with real-time ragged-edge handling (Schorfheide-Song 2015 lineage, precision-sampler implementations) — already scheduled at Tier 3; real-time vintage integration via foundations and nowcasting.

## Implementation warnings

The "easy to get statistically or numerically wrong" list. Every item below is a documented failure mode in published code or circulating implementations.

1. **Wrong-posterior bugs have shipped in the most-cited samplers in this literature.** Primiceri (2005) block ordering (corrected by Del Negro-Primiceri 2015) and the Carriero-Clark-Marcellino (2019) triangular algorithm (corrected in the 2022 corrigendum) both produced plausible-looking but wrong posteriors for years. Implement only the corrected versions, and run Geweke (2004) joint-distribution tests plus simulation-based calibration in CI for every sampler — conditional-independence errors in Gibbs blocks are silent.
2. **Haar rotation draws via QR must normalize the diagonal of R to be positive**; without it the rotation distribution is not uniform and sign-restriction posteriors are silently wrong. (Enforced in the foundations Haar kernel; this module must never bypass it.)
3. **Zero+sign algorithms (ARRW 2018) require volume-element/Jacobian importance weights**; the unweighted version — common in circulating code — samples from an undocumented prior. Narrative restrictions (AD-RR 2018) likewise require reweighting by the inverse acceptance probability estimated per draw. Always monitor and report importance-weight ESS. (Identification module owns the algorithms; the warning binds the draw streams this module supplies.)
4. **The uniform-Haar "agnostic" prior is informative about impulse responses and elasticities** (Baumeister-Hamilton 2015). Sign-restriction interfaces should ship prior-implied IRF plots by default, and note that the number of rotation draws per reduced-form draw changes the effective prior.
5. **Never ship the untruncated Newton-Raftery harmonic-mean marginal-likelihood estimator** (infinite variance). For latent-variable models (SV, TVP), Chib-style or MHM estimators built on the conditional-on-states likelihood are biased — the integrated likelihood is required (Chan-Grant 2015); use Chan-Eisenstat cross-entropy importance sampling, particle filters, or SMC tempering constants.
6. **Inverse-Wishart parameterization conventions** (scale vs rate, degrees-of-freedom offsets) differ across papers and packages and are the leading cause of failed cross-validation against published results. Fix one convention, document it, and provide translation helpers.
7. **Exploit Kronecker/matrix-normal structure everywhere**: never form the NK x NK posterior covariance; all log-determinants via Cholesky; no explicit matrix inverses anywhere; tested jitter policies for near-PSD matrices; banded (not generic sparse) Cholesky for precision samplers, since state precisions are block-banded.
8. **The KSC/Omori mixture for SV is an approximation** — provide the importance-reweighting correction and document the offset constant for near-zero residuals; results can be sensitive to both.
9. **Cholesky-based multivariate SV makes the reduced-form covariance depend on variable ordering** — an econometric, not numerical, pitfall users constantly miss. Offer order-invariant alternatives (common SV, factor SV, Chan-Koop-Yu 2024) and warn in the documentation.
10. **Truncating explosive/nonstationary draws is a change of prior, not a numerical detail**: it biases marginal likelihoods and can silently reject almost all draws in near-unit-root data. Make it explicit, capped, and reported.
11. **Simulation smoothers**: square-root Kalman filtering, exact diffuse initialization (never the big-kappa hack, which corrupts likelihood values), and rank-aware backward recursions for singular state covariances. Carter-Kohn with a singular state-innovation covariance is a classic crash-or-silently-wrong site.
12. **Dummy-observation and direct prior-moment implementations must be unit-tested for exact algebraic equivalence**, including degrees-of-freedom bookkeeping — off-by-one dof errors reproduce published numbers approximately and are therefore hard to catch.
13. **Predictive densities must integrate parameter uncertainty** by simulating the full posterior predictive; plug-in-posterior-mean predictive densities overstate calibration and are a pervasive error in forecast-evaluation code.
14. **Markov-switching models**: handle label switching by permutation samplers or enforced identification constraints, and expect multimodality — single-chain MCMC results on MS-VARs are untrustworthy; prefer SMC or many dispersed chains.
15. **Adaptive MCMC must respect diminishing adaptation** (freeze or decay adaptation) or the chain is not ergodic; this is easy to break when users compose custom blocks.
16. **GIG sampling (Normal-Gamma priors) and high-dimensional truncated-normal sampling (shadow rates, soft conditions) both have naive implementations that fail silently** in extreme-parameter regions — use Devroye/Hörmann-Leydold GIG and Botev minimax tilting or carefully monitored Gibbs for TMVN.
17. **Reproducibility across parallel chains, bootstrap, and SMC requires counter-based RNGs keyed by (seed, chain, draw, block)**; sharing stateful RNGs across threads makes results scheduler-dependent and unreproducible — fatal for a library whose pitch includes replication.
18. **Hyperparameter optimization or MH over marginal likelihoods must retain ALL normalizing constants** of the closed-form ML — dropping "constant" terms that depend on hyperparameters (a common shortcut) corrupts both optimization and posterior weighting.

## Dependencies and shared infrastructure

**Consumed from foundations:**

- Linear-Gaussian state-space engine — Carter-Kohn FFBS (Carter-Kohn 1994; Frühwirth-Schnatter 1994), the Durbin-Koopman (2002) mean-correction simulation smoother, and the precision-based sampler (Chan-Jeliazkov 2009; McCausland-Miller-Pelletier 2011) as interchangeable backends. This module drives them inside every TVP/SV/mixed-frequency Gibbs block and needs: square-root filtering, exact diffuse initialization (no big-kappa), rank-aware backward recursions for singular state-innovation covariances, banded (LAPACK dpbtrf-style) Cholesky for the precision path, and draw-for-draw distributional equivalence tests across backends. Precision-based sampling should be the default engine.
- Philox counter-based parallel RNG — substream contract seed = f(user_seed, chain_id, draw_id, block_id) for bitwise-reproducible multi-chain MCMC, SMC particles, and compressed-VAR projections (Salmon et al. 2011).
- Typed IRF result object + generalized-IRF engine and companion-form forecast kernels — this module needs the engine batched over thousands of posterior draws (BLAS-3 over stacked 3D tensors, direct recursion rather than eigendecomposition) and supplies deterministic mappings from (reduced-form draw, identification object) to (IRF, FEVD, HD) that all identification schemes reuse.
- Haar-rotation/restriction-algebra kernel — consumed indirectly through the identification module's Bayesian backend; positive-diagonal QR normalization enforced at the kernel level.
- Innovation-distribution zoo — Student-t and skewed families for fat-tailed measurement errors and SV-t blocks.
- Numerical optimizers — BFGS for the GLP empirical-Bayes mode; convex solvers for entropic tilting.
- Deterministic-terms toolkit, time-index/calendar engine, and the real-time vintage data store (for recursive out-of-sample and real-time exercises).
- Unified forecast object (point/interval/density/path) and the golden-value validation harness.
- Exogenous-regressor (covariate) contract — exogenous blocks in BVARs (deterministic terms, dummy variables such as COVID indicators, exogenous foreign blocks in small-open-economy VARs) enter through the shared aligned interface, and Bayesian forecasting with exogenous variables uses its known-future/scenario-path machinery, which conditional-forecast and structural-scenario tooling reuses.

**Consumed from / supplied to the identification module** (identification owns restriction, rotation, and reweighting logic; this module supplies reduced-form posterior draw streams, priors, samplers, conditional-forecast machinery, and marginal likelihoods to its Bayesian backend):

- Sign-restricted BSVAR (Uhlig 2005; Rubio-Ramírez-Waggoner-Zha 2010) — identification-owned; consumes this module's NIW/hierarchical posteriors and the Haar kernel.
- Zero + sign restrictions via importance sampling (Arias-Rubio-Ramírez-Waggoner 2018) — identification-owned; importance-weight ESS monitoring is a shared contract.
- Penalty-function sign restrictions (Mountford-Uhlig 2009) — identification-owned, with the documented health warnings.
- Baumeister-Hamilton explicit-prior BSVAR (2015 Econometrica; 2019 AER) — identification-owned; consumes this module's NUTS/MH machinery for the structural-matrix posterior and the prior-implied-IRF plotting layer.
- Narrative sign restrictions (Antolín-Díaz & Rubio-Ramírez 2018) — identification-owned.
- Proxy/IV BSVAR (Caldara-Herbst 2019; Arias-Rubio-Ramírez-Waggoner 2021) — identification-owned; consumes this module's MH-within-Gibbs samplers.
- Robust Bayes for set-identified SVARs (Giacomini-Kitagawa 2021) — identification-owned; consumes reduced-form draw streams for bound aggregation.
- Structural scenario analysis (Antolín-Díaz, Petrella, Rubio-Ramírez 2021) — identification-owned; consumes this module's Waggoner-Zha conditional-forecast engine and KL/modesty statistics hooks.

**Consumed from forecasting-evaluation:**

- Density-forecast evaluation (log scores, CRPS, PITs, Amisano-Giacomini comparisons) — this module emits full posterior-predictive simulations; scoring, calibration tests, and comparisons live there (validated against R `scoringRules`).
- Forecast combination and prediction pools — Geweke-Amisano (2011 JoE) optimal pools and time-varying-weight pools (Del Negro-Hasegawa-Schorfheide 2016 JoE) are forecasting-evaluation items; this module documents that pooling densities is not BMA and hands off its predictive densities.

**Consumed from other modules:**

- LP module — Bayesian local projections (Ferreira-Miranda-Agrippino-Ricco; Tanaka 2020) are LP-owned; this module exports the auxiliary-VAR prior construction and marginal-likelihood tightness selection they need.
- multivariate — the single DFM implementation, reused for two-step FAVAR factor extraction.
- nowcasting — release-calendar and news-decomposition layers for real-time mixed-frequency BVAR exercises.

**Exposed to other modules:**

- Reduced-form BVAR posterior draw streams, priors (Minnesota/NIW/GLP/shrinkage families), and closed-form marginal likelihoods — the substrate for identification's Bayesian backend and nowcasting's mixed-frequency work.
- The composable Gibbs-block scheduler, adaptive MCMC, NUTS (nuts-rs), SMC, and PGAS engines — available to any module that ships a sampler.
- The marginal-likelihood suite and Savage-Dickey tools for model comparison anywhere in the library.
- Convergence diagnostics (R-hat, bulk/tail ESS, MCSE) and the Geweke (2004)/SBC CI harness — mandatory for every sampler library-wide.
- The Waggoner-Zha/Bańbura-Giannone-Lenza conditional-forecast engine, consumed by identification's structural scenario analysis.
- KSC/Omori and ASIS stochastic-volatility machinery for volatility work elsewhere in the library.

## Validation gallery

- Giannone-Lenza-Primiceri (2015 ReStat) Table 1 and forecast RMSEs — hyperparameter posteriors and forecasts must match R `BVAR` and the authors' Matlab code.
- Primiceri (2005) Figures 1-4 — must reproduce via the Del Negro-Primiceri (2015) corrected sampler and calibration.
- Carriero-Chan-Clark-Marcellino (2022 JoE) corrigendum replication files — corrected triangular large-BVAR-SV posterior must match, and must agree with brute-force full-system Gibbs on a small model.
- Bańbura-Giannone-Reichlin (2010 JAE) published RMSE tables — large-BVAR forecasting on the FRED-based dataset must match.
- Closed-form NIW benchmark — posterior moments and marginal likelihood on a 2-variable toy model must match brute-force numerical integration to machine precision; every ML estimator (Chib, bridge, cross-entropy, SMC constants) must recover it.
- Clark (2011 JBES) BVAR-SV replication and BEAR's BVAR-SV — medium-VAR stochastic-volatility posteriors must match.
- R `stochvol` — KSC/Omori posterior moments and ASIS mixing (ESS per second) benchmarks.
- Chan (2022) 25-variable asymmetric-conjugate application — must match Chan's posted Matlab code.
- Carriero-Clark-Marcellino-Mertens (2022 ReStat) and Lenza-Primiceri (2022 JAE) replication files — SVO and COVID-scaling results must match.
- Schorfheide-Song (2015 JBES) replication files and R `mfbvar` — mixed-frequency and steady-state posteriors must match.
- Bernanke-Boivin-Eliasz (2005 QJE) — FAVAR monetary-policy IRFs on 120 series must match.
- Koop-Korobilis (2013 JoE) Matlab code — forgetting-factor TVP-VAR/DMA forecasts must match, with runtime as the speed benchmark.
- Herbst-Schorfheide (2014 JAE) replication — SMC posterior and marginal-likelihood estimates must match.
- ArviZ / R `posterior` — R-hat, bulk/tail ESS, and MCSE values must agree numerically on shared draws.
- Geweke (2004) joint-distribution tests and simulation-based calibration (Talts et al. 2018) — every sampler must pass in CI on tiny models; this is a permanent gate, not a one-time target.
