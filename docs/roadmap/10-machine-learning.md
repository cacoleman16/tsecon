# Module 10 — Machine Learning

> Part of the time series econometrics library roadmap. Master plan: [ROADMAP.md](../../ROADMAP.md).

**This module is the library's bridge between econometrics and the machine-learning toolchain: it owns the penalized-regression solver stack (ridge, LASSO, elastic net, adaptive, group, sparse-group, with information-criterion tuning), dependence-aware cross-validation with purging and embargo, native random forests and boosting, factor/diffusion-index forecasting facades, double/debiased machine learning adapted to serially dependent data, Gaussian processes in state-space form, and interpretation tooling that respects autocorrelation. The guiding scope opinion is to own the econometrics — tuning, selection, inference, and honest evaluation under dependence — and interoperate with the existing ML zoo (sklearn, XGBoost, LightGBM, torch) rather than reimplement it.**

## Purpose and scope

The module covers everything a data-rich forecaster or applied econometrician needs to use machine-learning methods without violating the statistics: leakage-safe pipelines in which every transform is fit only on training folds; path-based penalized regression matching glmnet conventions exactly; time-series cross-validation with horizon-scaled embargoes; native tree ensembles with block-bootstrap resampling; nonparametric regression with autocorrelation-corrected smoothing selection; post-selection and Neyman-orthogonal inference (desparsified LASSO, post-double-selection, DML with block cross-fitting and HAC scores); and interpretation methods (block-permutation importance, lag-grouped Shapley, ALE) that do not fabricate dynamically impossible feature vectors. Its users are macro forecasters running data-rich horse races, central-bank staff building nowcast-adjacent pipelines, and applied researchers who need valid inference on a low-dimensional parameter surrounded by high-dimensional nuisances.

Three scope rulings from the master plan bound the module. First, deep-learning forecaster adapters (N-BEATS, N-HiTS, DeepAR, TFT) and foundation-model adapters (Chronos, TimesFM, Moirai, Lag-Llama, TimeGPT) move out of core into a separate optional companion package; the contamination-aware benchmark harness stays in core, because auditing zero-shot claims against training-corpus contamination and real-time vintages is durable econometrics even as the models churn. Second, the causal-panel suite — classic and augmented synthetic control, elastic-net SC, synthetic difference-in-differences, matrix completion (MC-NNM), and conformal counterfactual inference — moves to a companion package built on this module's solvers; LP-DiD and IPW/AIPW local projections stay in the local-projections module. Third, the library wraps rather than owns gradient-boosted trees and torch models: XGBoost/LightGBM/torch adapters live outside the core wheels behind feature flags. In addition, the matrix profile, echo state networks, and S-H-ESD anomaly scoring are demoted to the contrib tier.

Relative to the rest of the library, this module is a supplier as much as a consumer. Its penalized-regression solvers and TS-CV splitters are the engines behind the multivariate module's regularized VARs (Basu-Michailidis, VARX-L, HLag), the nowcasting module's sg-LASSO-MIDAS regressions, and the LP module's high-dimensional local projections — those estimators live in their owning modules and are listed here only as dependencies. Factor estimation and number-of-factors criteria live in foundations; this module owns the forecasting facades built on them (diffusion indexes, targeted predictors, 3PRF). Conformal prediction is consumed from forecasting-evaluation, which owns the single model-agnostic implementation.

## Where existing tools fall short

- statsmodels has essentially no ML layer: `fit_regularized` returns a single elastic-net point fit with no path algorithm, no time-series-aware tuning, no post-selection inference, no group or structured penalties, and no penalized VARs.
- sklearn's `TimeSeriesSplit` lacks purging, embargo, and horizon-aware gaps; its CV, permutation importance, and conformal-adjacent tools all assume exchangeability, so out-of-the-box sklearn pipelines quietly leak or overstate accuracy on dependent data.
- The best tools in this domain are R-only, loop-heavy, and mutually incompatible — BigVAR, midasml, MacroRF, gets, desla, bigtime, synthdid, augsynth, gsynth, tso — with no Python port of most of them and no unified API anywhere.
- No mainstream library in any language ships maintained implementations of Basu-Michailidis sparse VARs, HLag, or high-dimensional Granger causality tests (this library places them in the multivariate module, on top of this module's solvers).
- Factor tooling is fragmented: Bai-Ng criteria are half-implemented across packages with inconsistent standardization conventions, and targeted predictors and the three-pass regression filter have no mainstream implementation at all.
- DoubleML and EconML implement iid cross-fitting only; there is no packaged DML for dependent data (block cross-fitting with embargoes, HAC score variances) anywhere.
- Interpretation tooling (shap, sklearn permutation importance) ignores serial dependence and lag-group structure, producing dynamically impossible perturbations and inflated importances for persistent predictors.
- Conformal prediction for time series (EnbPI, ACI, SPCI) lives only in research repos and ML-ops libraries with no statistical documentation or dependence-aware defaults (owned by forecasting-evaluation in this library).
- Foundation TS models ship with demo notebooks, not econometrics-grade evaluation: no DM/Clark-West/MCS testing, no real-time vintage handling, and no training-set contamination checks, so their macro claims are hard to audit.
- GP libraries (GPy, GPflow, GPyTorch) do not expose the O(n) state-space representation in a stats-oriented API, and no econometrics package connects GP kernels to structural time-series components.
- Nonparametric regression in Python is slow and dependence-naive: statsmodels `KernelReg` is O(n²) without binning, and no package implements autocorrelation-corrected smoothing-parameter selection (Krivobokova-Kauermann).
- IC-based LASSO tuning with correct degrees of freedom, EBIC for p>n, and HAC-aware tuning exist only in scattered R code (midasml being the notable exception); Python users default to leaky CV.
- MATLAB/Dynare, the default stack for many central-bank users, has effectively none of this domain, forcing an R+MATLAB+Python toolchain patchwork.

## Inventory

### Tier 1 — Core (v1-blocking)

**Infrastructure and interop**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Leakage-safe TS pipeline + sklearn/torch interop layer | Pipeline protocol where every transform (scaling, differencing, lag creation, imputation) is fit only on training folds, plus thin adapters so any sklearn regressor or torch model drops into the library's forecasting, backtesting, and DML machinery. | Medium | The central scope opinion: own the econometrics, interoperate with the ML zoo. Rust `Forecaster` trait with a Python protocol mirror (fit/predict/predict_interval/update). Trap: silent leakage via full-sample standardization or target transforms — the most common bug in published ML-macro replications. Validate by property tests: perturbing post-fold-boundary data must not change fitted transforms. |

**Penalized regression and tuning**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Time-series cross-validation suite (rolling/expanding origin, blocked K-fold with purging/embargo, hv-block) | Dependence-aware splitters for tuning and evaluation — the backbone every ML estimator in the library uses. | Medium | Racine (2000) hv-block; Bergmeir, Hyndman & Koo (2018) show ordinary K-fold IS valid for purely autoregressive models with iid errors — document the nuance rather than blanket-banning K-fold. Embargo width must scale with horizon h (h-step errors are MA(h-1)). Validate index sets against sktime/sklearn `TimeSeriesSplit` where comparable; add the purge/embargo they lack. |
| Ridge regression with SVD path and GCV | L2-penalized forecasting regression: the workhorse for dense-signal macro problems and the readout for reservoir computing. | Low | One thin SVD, then the whole lambda path is O(np) per lambda. GCV, IC, and rolling-origin CV tuning. Hoerl & Kennard (1970); macro evidence Smeekes & Wijler (2018, IJF). Validate against glmnet(alpha=0) and sklearn Ridge to 1e-8; watch intercept/standardization conventions. |
| LASSO via coordinate descent with strong rules | L1-penalized regression, the default variable-selection tool for data-rich macro forecasting. | Medium | Covariance-updating coordinate descent with active sets, sequential strong rules (Tibshirani et al. 2012), warm starts on a log-spaced grid from data-derived lambda_max. Trap: match glmnet's 1/(2n) objective scaling and standardization exactly or lambdas silently misalign. Dependence theory: Medeiros & Mendes (2016, JoE); Kock & Callot (2015, JoE). Validate full coefficient paths against glmnet to 1e-6. |
| Elastic net | Convex L1/L2 combination; handles the highly collinear lag blocks where pure LASSO arbitrarily drops one of two correlated lags. | Low | Trivial extension of the CD engine. Zou & Hastie (2005, JRSS-B). Tune (alpha, lambda) on a 2-D grid with warm starts along lambda. Validate against glmnet; replicate the elastic-net column of Medeiros et al. (2021). |
| Adaptive LASSO | Two-step LASSO with data-driven penalty weights achieving oracle selection; the theoretically preferred variant for time series. | Low | Zou (2006, JASA); time-series validity Medeiros & Mendes (2016). Implement as penalty-factor reweighting in the CD engine. Trap: first-stage estimator when p>n (use ridge or LASSO, never OLS); document the gamma exponent default (1). Validate against glmnet with `penalty.factor`. |
| IC-based tuning (BIC/AIC/EBIC/WIC) for penalized estimators | Selects lambda by information criteria instead of CV — much faster for Monte Carlo work and preferred in the TS literature (BIC-tuned LASSO). | Medium | df = nonzero count is exact for LASSO (Zou, Hastie & Tibshirani 2007, AoS); diverging-p correction Wang, Li & Leng (2009, JRSS-B); EBIC for p>n Chen & Chen (2008, Biometrika). Trap: nonzero-count df is only approximate for elastic net (use trace of the ridge-part hat matrix on the active set). Validate selected models against published Medeiros et al. (2021) choices. |

**Factor models and dimension reduction**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Stock-Watson diffusion-index forecasting facade | Forecast with factor-augmented ARDL on principal-component factors from a large standardized panel — the canonical data-rich benchmark every ML paper compares against. | Medium | Stock & Watson (2002, JASA; 2002, JBES). Estimation core (thin-SVD PCA, EM for unbalanced panels, deterministic sign fixing) is CONSUMED from foundations; this module owns the forecasting facade and its leakage-safe pipeline wiring. Validate against FRED-MD factor estimates in McCracken & Ng (2016, JBES) and statsmodels DynamicFactorMQ where applicable. |

### Tier 2 — Standard (expected of a serious library)

**Penalized regression and tuning**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Group LASSO for lag blocks | Selects whole variables (all lags of a predictor as one group) — the natural structure for ARDL/VAR forecasting equations. | Medium | Yuan & Lin (2006, JRSS-B). Block coordinate descent; either orthonormalize within groups or use MM with per-block Lipschitz constants — mixing these up gives silently wrong solutions. Expose "lag-block" and "variable-block" group builders. Validate against R gglasso/grpreg. |
| Sparse-group LASSO (SGL) | Group plus within-group sparsity; needed standalone and as the engine for the nowcasting module's sg-LASSO-MIDAS. | Medium | Simon, Friedman, Hastie & Tibshirani (2013, JCGS). Proximal/block CD with the two-level soft-threshold prox. Trap: convergence tolerance interacts with group scaling — document the sqrt(group size) weight convention. Validate against R SGL and midasml. |
| Post-LASSO OLS | Refit OLS on the LASSO-selected support to remove shrinkage bias, standard before using coefficients economically. | Low | Belloni & Chernozhukov (2013, Bernoulli). Emit a loud warning that naive OLS standard errors after selection are invalid — point users to desparsified LASSO or PDS. Trivial numerics; the API and documentation design is the real work. |
| L1 trend filtering | Piecewise-linear trend estimation via L1 penalty on second differences — a modern alternative to HP filtering with automatic knot selection. | Medium | Kim, Koh & Boyd (2009, SIAM Review). Specialized ADMM or primal-dual interior point on the banded system — exploit bandedness for O(n) per iteration. Validate against R glmgen/genlasso. Document the relation to the boosted HP filter (Phillips & Shi 2021) in the filtering/decomposition module. |

**Factor models and dimension reduction**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Targeted predictors | Pre-select predictors by hard/soft thresholding (marginal t-stats or LASSO) before factor extraction, so factors target the forecast variable rather than panel variance. | Medium | Bai & Ng (2008, JoE). Composable pipeline: screen → PCA → ARDL, all inside the leakage-safe CV. Validate against the paper's inflation-forecasting improvements over plain diffusion indexes. |
| Factor-augmented regressions / FAVAR bridge | Plug estimated factors into forecasting regressions and VARs; this module supplies factor construction and pipelines, the multivariate module supplies VAR dynamics and IRFs. | Medium | Bernanke, Boivin & Eliasz (2005, QJE). Main issue is generated-regressor uncertainty: document when it is asymptotically negligible (sqrt(T)/N → 0, Bai & Ng 2006) and offer bootstrap otherwise. Validate IRFs against the published BBE figures. |

**Trees, forests, and boosting**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Random forest with TS-aware resampling (native Rust) | Regression forest with optional block/stationary bootstrap resampling and lag-feature tooling; the strongest off-the-shelf nonlinear benchmark for inflation and macro forecasting. | High | Breiman (2001); macro evidence Medeiros, Vasconcelos, Veiga & Zilberman (2021, JBES) — RF wins for US inflation, driven by nonlinearity per Goulet Coulombe et al. (2022, JAE). Native implementation is required anyway for MRF and importance work; offer Politis-Romano stationary bootstrap. Trap: OOB error is optimistic under autocorrelation — always surface POOS metrics instead. Validate iid behavior against ranger/sklearn, then replicate Medeiros et al. table results on their public data. Docs must be honest: trees rarely beat regularized linear models for smooth low-signal problems and short samples. |
| Componentwise L2 boosting (boosted ARDL) | Gradient boosting with single-predictor/single-lag least-squares base learners — a slow-learning variable selector econometricians read as sequential ARDL building. | Medium | Bühlmann & Yu (2003, JASA); Bühlmann (2006, AoS); macro application Ng (2014, CJE). Early stopping via corrected-df AIC or POOS. Cheap, deterministic, very fast in Rust. Validate against R mboost `glmboost`. |
| Gradient-boosted trees adapter (XGBoost/LightGBM) | First-class adapter exposing GBTs inside the library's TS-CV, backtesting, and interpretation machinery. Wrap-don't-own: never reimplement histogram GBM. | Low | Adapters live outside the core wheels per the master plan. Leakage-safe tuning via rolling-origin CV; expose monotonicity constraints for econ priors. Validation is of the harness (identical predictions to direct library calls), not the algorithm. Docs carry honest evidence on when GBTs beat linear macro benchmarks (rarely for point forecasts of smooth aggregates; more often under asymmetric/quantile loss). |
| Bagging for predictor selection (Inoue-Kilian) | Bootstrap-aggregated pretest/hard-threshold forecasts; a simple, well-cited ensemble that often matches fancier methods. | Low | Inoue & Kilian (2008, JASA). Use block bootstrap under dependence (from foundations). Good pedagogical bridge in docs from OLS to ML. Validate on their CPI inflation application. |

**Neural**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Minimal native MLP forecaster | Single/two-hidden-layer feedforward net on lagged inputs with seed-ensembling — the "NN" entry in macro horse races, and the only neural net implemented natively. | Medium | As benchmarked in Medeiros et al. (2021) and Goulet Coulombe et al. (2022, JAE). Adam/L-BFGS, early stopping on a temporal validation split, ensemble over ~10 seeds. Document nondeterminism honestly (BLAS threading). Validate: match sklearn MLPRegressor on fixed seeds/architecture, then replicate published macro rankings qualitatively. |

**Nonparametric regression**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Kernel regression (Nadaraya-Watson, local linear/polynomial) | Classic nonparametric conditional-mean estimation for nonlinear autoregressions; local linear is the default for boundary-bias reasons. | Medium | Fan & Gijbels (1996). Bandwidth under dependence: leave-block-out CV or plug-in with correlation correction — ordinary leave-one-out CV undersmooths badly with autocorrelated errors. Linear binning + FFT for O(n log n). Validate against the R np package (painfully slow — a speed win to advertise) and statsmodels KernelReg. |
| Penalized splines with AR-error smoothing selection | B-spline bases with difference penalties for trends, seasonality, and semiparametric terms; mixed-model representation gives REML-based smoothness selection. | Medium | Eilers & Marx (1996, Statistical Science). Critical trap: autocorrelated errors make GCV/REML drastically undersmooth — implement the AR(p)-error correction of Krivobokova & Kauermann (2007, JASA). Banded penalized least squares, O(n). Validate against mgcv `gam`/`gamm`. |
| Kernel ridge regression with random Fourier features | Kernelized ridge for nonlinear forecasting; RFF keeps it O(n) for long samples. A strong, tuning-light nonlinear baseline. | Low | Rahimi & Recht (2007) for RFF. Exact KRR is a Cholesky solve reusing ridge code; used as a nonlinear benchmark in Goulet Coulombe et al. (2022). Validate against sklearn KernelRidge. |

**Causal ML and DML**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Post-double-selection (PDS) LASSO for treatment coefficients | Belloni-Chernozhukov-Hansen selection of controls for inference on a treatment/policy coefficient, extended with HAC inference for time-series controls (lags, deterministics). | Medium | Belloni, Chernozhukov & Hansen (2014, ReStud). Union-of-supports then OLS with robust/HAC SEs; never penalize the treatment or mandatory controls (amelioration set). Validate against Stata pdslasso/dsregress and R hdm on their examples. |

**Interpretation**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Block-permutation and conditional variable importance | Permutation importance with lags permuted in contiguous blocks (preserving autocorrelation) and conditional importance for correlated predictors. | Medium | Conditional importance: Strobl et al. (2008, BMC Bioinformatics). Trap: naive single-observation permutation creates dynamically impossible feature vectors and overstates importance of persistent predictors. Group all lags of one variable into one importance unit by default. Validate by simulation: recover known relevant variables in sparse nonlinear DGPs. |
| Partial dependence and ALE plots for TS features | One- and two-dimensional effect plots of predictors/lags on forecasts; ALE is the default because it stays honest under the strong feature correlation of lagged designs. | Low | ALE: Apley & Zhu (2020, JRSS-B). PD plots mislead when lags are correlated (extrapolation into empty regions) — docs push ALE first. Cheap over the unified Forecaster protocol. Validate against R ALEPlot/iml on shared models. |

**Anomaly detection and exploratory**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Intervention-type outlier detection (Chen-Liu AO/LS/TC) | Joint estimation of ARIMA parameters and additive outliers, level shifts, and temporary changes — the econometric standard for pre-cleaning series (indispensable post-COVID). | High | Chen & Liu (1993, JASA). Iterative detect-adjust-reestimate loop; critical-value defaults depend on sample size (follow TRAMO conventions; consume foundations' critical-value engine). The R tso package is the validation target (and is slow — another speed win). Cross-reference COVID-handling guidance (Lenza & Primiceri 2022, JAE); coordinate ownership with the diagnostics/seasonal-adjustment module. |

### Tier 3 — Advanced (differentiators)

**Penalized regression and tuning**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Square-root LASSO | Pivotal LASSO whose theoretically valid lambda does not depend on the unknown error variance — attractive for automated Monte Carlo pipelines. | Medium | Belloni, Chernozhukov & Wang (2011, Biometrika). Solvable by scaled-lasso iteration (Sun & Zhang 2012) reusing the CD engine. Validate against R RPtests/scalreg. |

**Factor models and dimension reduction**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Three-pass regression filter (3PRF) / PLS forecasting | Supervised factor extraction via proxy regressions; nests PLS as a special case. Use when relevant factors are dominated in the panel's covariance by irrelevant ones. | Medium | Kelly & Pruitt (2015, JoE). Three OLS passes — trivial numerics, subtle bookkeeping (auto-proxy construction). Include vanilla PLS (NIPALS/SIMPLS). Validate 3PRF against the authors' MATLAB outputs and PLS against sklearn on iid data. |

**Trees, forests, and boosting**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Quantile regression forests / distributional forests | Forest-based conditional quantile and density forecasting; the ML route to growth-at-risk analyses. | Medium | Meinshausen (2006, JMLR). Reuse the native forest, store leaf training weights. Cross-reference density-forecast evaluation in forecasting-evaluation (CRPS, quantile scores) and Adrian, Boyarchenko & Giannone (2019). Validate against R quantregForest / sklearn-quantile on iid data. |

**Shrinkage vs sparsity**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Complete subset regressions (CSR) | Equal-weight combination of forecasts from all (or sampled) k-variable subsets; a robust dense-ish combination method with a strong empirical record. | Medium | Elliott, Gargano & Timmermann (2013, JoE). Combinatorial explosion handled by random subset sampling; embarrassingly parallel — a showcase for Rust speed. Validate against the CSR benchmark numbers reported in Medeiros et al. (2021). |
| Dynamic model averaging/selection (DMA/DMS) | Online Bayesian averaging over TVP regressions with forgetting factors; popular at central banks for inflation forecasting with evolving predictor relevance. | High | Raftery, Kárný & Ettler (2010, Technometrics); Koop & Korobilis (2012, IER). Kalman-filter bank with forgetting — a thin facade over foundations' state-space engine, coordinated with the Bayesian/TVP module. Trap: numerical underflow in model probabilities (accumulate in log space). Validate against Koop-Korobilis MATLAB output. |

**Nonparametric regression**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Gaussian processes with state-space (SDE) representation | GP regression for trend + seasonality where Matern/quasi-periodic kernels convert to linear SDEs run through the Kalman filter — exact O(n) inference tied to the library's state-space engine. | High | Hartikainen & Särkkä (2010); periodic expansion Solin & Särkkä (2014). Matern 1/2, 3/2, 5/2 are exact SDEs; periodic requires truncated harmonic expansion (default J≈6, documented). Hyperparameters by marginal likelihood with filter-based gradients; square-root Kalman filtering for stability. Validate small-n posteriors against exact GP (GPy/GPflow) to 1e-6; keep exact O(n³) GP as fallback for arbitrary kernels. |

**Causal ML and DML**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Double/debiased ML (DML) with dependent-data cross-fitting | Neyman-orthogonal estimation of low-dimensional causal parameters (partially linear, IV/LATE) with ML nuisances, using cross-fitting valid under serial correlation (contiguous block folds with embargo buffers). | High | Chernozhukov et al. (2018, Econometrics Journal). TS adaptation: block folds with embargo margins sized by the mixing/horizon structure, HAC variance for the orthogonal score; docs must state that dependent-data theory is still developing. Validate iid behavior against DoubleML exactly, then TS coverage by simulation against known DGPs. Any registered learner can be a nuisance model via the interop layer. |

**Interpretation**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Lag-grouped Shapley attributions (TreeSHAP interop + grouped Shapley) | Shapley decompositions where coalitions are variables (all their lags together) rather than individual lag features, with exact TreeSHAP for native forests and interop for wrapped models. | High | Lundberg & Lee (2017, NeurIPS); TreeSHAP (Lundberg et al. 2020, Nature MI). Trap: interventional Shapley marginalizes over impossible lag combinations under strong autocorrelation — offer conditional variants and document the interventional/conditional distinction prominently. Grouped (coalition) Shapley over lag blocks is the default output. Validate TreeSHAP against the shap package on identical trees. |

**Model selection (GETS)**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| GETS model selection and indicator saturation (IIS/SIS) | General-to-specific automated selection with impulse- and step-indicator saturation, doubling as outlier and structural-break detection with controlled gauge — the Hendry-school alternative to LASSO. | High | Hendry, Johansen & Santos (2008); software reference Pretis, Reade & Sucarrat (2018, JSS, R gets). Multi-path search with diagnostic-test guards; block-partitioning for saturation (more candidate indicators than observations). Trap: gauge/potency calibration depends on the significance level — replicate the gets defaults. Validate selection paths against gets on its documented examples. |

**Neural and foundation models**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Contamination-aware benchmark harness for external forecasters | Econometrics-grade evaluation harness for wrapped deep and foundation models: DM/Clark-West/MCS testing (via forecasting-evaluation), real-time vintages (via nowcasting's vintage layer), and training-corpus contamination flags with post-cutoff evaluation windows. Retained in core by scope ruling; the model adapters themselves (Chronos, TimesFM, Moirai, Lag-Llama, TimeGPT, N-BEATS, DeepAR, TFT) live in the companion package. | Medium | Chronos (Ansari et al. 2024), TimesFM (Das et al. 2024 ICML), Moirai (Woo et al. 2024 ICML), Lag-Llama (Rasul et al. 2024), TimeGPT (Garza et al. 2023, API-only). Honest docs: zero-shot evidence on small-T quarterly macro is mixed and frequently fails to beat AR(1)/BVAR; contamination of public series (FRED, M-competitions) undermines many claimed wins. Flag series plausibly in training corpora; prefer post-cutoff windows; guard against silent resampling/scaling defaults inside wrapped libraries. TimeGPT is a network call — document data-privacy implications for central-bank users. |

### Tier 4 — Frontier (research-grade, each gated on a named validation target)

**Penalized regression and tuning**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Desparsified/debiased LASSO inference under serial dependence | Honest confidence intervals and tests on individual coefficients in high-dimensional time series regressions — the inference layer missing from every Python ML library. | High | van de Geer et al. (2014, AoS) adapted to time series by Adamek, Smeekes & Wilms (2023, JoE). Nodewise lasso regressions for the precision matrix plus HAC/long-run variance of the moment terms. Traps: nodewise lambda tuning drives coverage; HAC bandwidth choice matters. Gate: reproduce the desla R package and the paper's simulation coverage tables. |

**High-dimensional selection under factors**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Factor-adjusted regularized selection (FarmSelect) | Removes estimated common factors before running LASSO on idiosyncratic components, fixing the failure of sparsity methods under pervasive cross-correlation — directly relevant to macro panels. | Medium | Fan, Ke & Wang (2020, JoE). Composable from the foundations PCA core and this module's LASSO engine. Trap: factor-number misspecification propagates; report selection under a range of k. Gate: reproduce the FarmSelect R package. |

**Trees, forests, and boosting**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Macroeconomic Random Forest (MRF) | Random forest whose leaves contain linear time-varying-coefficient macro equations (trees model coefficient evolution) — interpretable "generalized time-varying parameters" with tree flexibility. | Research-grade | Goulet Coulombe (2024, Journal of Applied Econometrics). Key ingredients: ridge shrinkage inside leaves, random-walk regularization of coefficient paths, blocked subsampling. Almost no library has this (only the author's MacroRF R/Python code). Substantial custom splitting logic — budget accordingly. Gate: reproduce MacroRF outputs and the paper's GTVP plots. |

**Shrinkage vs sparsity**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Illusion-of-sparsity diagnostic (GLP spike-and-slab) | Hierarchical spike-and-slab regression treating the inclusion probability q as unknown, reporting a posterior over sparsity — the packaged arbiter of the LASSO-vs-ridge/Bayesian-shrinkage choice. | Research-grade | Giannone, Lenza & Primiceri (2021, Econometrica). Gibbs sampler over (q, R²-linked slab variance, inclusion indicators); mind the label-switching-free parameterization and R²-based prior scaling. Gate: reproduce posterior sparsity distributions across the paper's six applications (public replication code). |
| Random subspace / random projection forecasting | Average forecasts over random low-dimensional projections or subsets of predictors; a theoretically grounded dense alternative to factor models. | Medium | Boot & Nibbering (2019, JoE). Gaussian and sparse (Achlioptas) projections; tuning = subspace dimension via POOS. Cheap on the ridge/OLS engines. Gate: reproduce the paper's FRED-MD results. |

### Contrib tier (demoted by scope ruling)

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Matrix profile (STOMP/SCRIMP++) | All-pairs z-normalized subsequence distance profile for motif discovery, discord (anomaly) detection, and regime-change hints — purely exploratory. | Medium | STOMP: Zhu et al. (2016, ICDM); SCRIMP++ (2018) for anytime computation. Traps: near-constant windows blow up z-normalized distance (epsilon guard + flag); streaming dot-product recursions accumulate FP error — refresh via FFT recomputation. Validate elementwise against stumpy. |
| Echo state networks / reservoir computing | Fixed random recurrent reservoir with a ridge-trained readout: nonlinear dynamics at linear-regression cost, deterministic given seed, with emerging macro evidence. | Medium | Jaeger (2001); practical guide Lukoševičius (2012); macro/mixed-frequency application Ballarin et al. (2024, IJF). Reuses the ridge engine. Traps: spectral-radius rescaling, washout discard, leak-rate tuning via POOS. Validate against reservoirpy on NARMA10 and the IJF paper's setup. |
| Residual-based anomaly scoring (S-H-ESD, robust thresholds) | Model/STL-residual anomaly flagging with robust MAD thresholds and seasonal-hybrid generalized-ESD tests; a quick-look data-quality screen. | Low | Generalized ESD: Rosner (1983, Technometrics); seasonal-hybrid variant Hochenbaum et al. (2017). Deliberately simple utility; consumes STL from the diagnostics module. Validate against the anomalize/AnomalyDetection R packages. |

## Frontier watchlist

Frontier items from the research inventory that live elsewhere per the ownership map and scope rulings, tracked here because they consume this module's engines:

- sg-LASSO-MIDAS machine-learning regressions with HAC-based inference (Babii, Ghysels & Striaukas 2022, JBES; 2024, JFEC Granger tests) — owned by the nowcasting module; runs on this module's SGL solver. Only midasml (R) has it today.
- High-dimensional local projection inference (Adamek, Smeekes & Wilms 2024, The Econometrics Journal) and generic orthogonal-LP with cross-fit ML nuisances — owned by the LP module; consumes this module's desparsified LASSO and DML machinery.
- High-dimensional Granger causality via post-double-selection (Hecq, Margaritella & Smeekes 2023, JFEC) — owned by the multivariate module; consumes this module's PDS-LASSO engine.
- Unified modern panel-causal inference — synthetic DiD (Arkhangelsky et al. 2021, AER), matrix completion MC-NNM (Athey et al. 2021, JASA), conformal counterfactuals (Chernozhukov, Wüthrich & Zhu 2021, JASA) — companion package, built on this module's elastic-net/ridge solvers and foundations' bootstrap engine.
- Adaptive/sequential conformal prediction for dependent data — EnbPI (Xu & Xie 2021), ACI (Gibbs & Candès 2021), SPCI (Xu & Xie 2023) — owned by forecasting-evaluation as the single model-agnostic implementation, wrapping any Forecaster from this module's protocol.
- Foundation time-series model adapters (Chronos, TimesFM, Moirai, Lag-Llama, TimeGPT) — companion package; evaluated through this module's contamination-aware benchmark harness (Tier 3).

## Implementation warnings

- Cross-validation leakage is the domain's cardinal sin: standardization, differencing, imputation, and target transforms must be fit within training folds only, and blocked folds need an embargo of at least h-1 periods because h-step errors are MA(h-1) even under correct specification. Do not over-correct, though: K-fold is provably fine for pure AR models with iid errors (Bergmeir-Hyndman-Koo 2018) — document the nuance.
- Match glmnet's conventions exactly — 1/(2n) loss scaling, standardization including/excluding the intercept, lambda-path construction from lambda_max — or every published lambda in the literature will silently mean something different in this library. Validate full coefficient paths to 1e-6.
- IC-based tuning needs the right degrees of freedom: the nonzero count is exact for LASSO (Zou-Hastie-Tibshirani 2007) but not for elastic net or adaptive variants; for p>n use EBIC, or BIC will overselect catastrophically.
- Never report naive OLS standard errors after any selection step; route users to desparsified LASSO, PDS, or explicitly labeled "no-inference" post-LASSO output.
- Structured penalties have subtle prox math: group LASSO requires groupwise orthonormalization or correct per-block Lipschitz constants, and HLag's nested prox must be applied in the Jenatton tree order — the wrong order converges smoothly to the wrong answer with no error raised. (The HLag estimator lives in the multivariate module, but the prox primitive ships from this module's solver stack.)
- Random forest OOB error is optimistic under autocorrelation, and iid bootstrap breaks temporal dependence: default to block/stationary bootstrap resampling options and always surface pseudo-out-of-sample metrics rather than OOB in reports.
- Factor estimation: standardize the panel, compute factors via thin SVD (never eigendecomposition of an explicitly formed covariance), fix factor signs deterministically for reproducibility, and expose the kmax bound because Bai-Ng criteria are sensitive to it.
- State-space GPs need square-root/Cholesky-form Kalman filtering and kernel-matrix jitter; the periodic kernel is only a truncated harmonic expansion — document the truncation order and its effect on the marginal likelihood.
- Synthetic control (companion package, but the warnings bind its use of this module's solvers): the outer V-optimization is non-convex (different optimizers give different published-looking weights — document which is matched), inner simplex QPs are degenerate with large donor pools (offer the penalized variant), and SDID replication requires the exact synthdid regularization constant zeta = ((N_treat · T_post)^(1/4)) · sigma-hat with exact period bookkeeping — off-by-one errors silently break Prop-99 replication.
- HAC choices propagate everywhere: Newey-West bandwidth materially changes sg-LASSO tuning, DML score variances, and LP inference; nested forecast comparisons need Clark-West rather than Diebold-Mariano; and DM tests at horizon h need HAC lags of at least h-1.
- Conformal methods lose exchangeability under dependence: plain split conformal undercovers, so default to ACI/EnbPI, state clearly that coverage is marginal (not conditional), and expose the ACI learning rate because it drives interval-width volatility. (Implementation owned by forecasting-evaluation.)
- Interpretation under autocorrelation: permuting individual lag features creates dynamically impossible inputs and inflates the importance of persistent predictors — use block permutation and lag-grouped (coalition) attributions, and document interventional vs conditional Shapley explicitly.
- Neural and foundation models are nondeterministic across BLAS thread counts and GPUs: promise statistical reproducibility (seed ensembles) rather than bitwise reproducibility, and treat benchmark wins on public series as suspect until checked for training-corpus contamination and evaluated on post-cutoff windows.
- Nonparametric smoothing with autocorrelated errors: leave-one-out CV and uncorrected GCV/REML undersmooth drastically (the smoother chases serially correlated noise) — implement correlation-corrected criteria (Krivobokova-Kauermann 2007) and leave-block-out bandwidth selection as defaults.
- Matrix profile numerics: guard z-normalized distances against near-zero-variance windows, and periodically refresh streaming dot-product recursions via FFT recomputation to contain floating-point drift.
- Soft-impute/nuclear-norm solvers (companion package) must alternate correctly with fixed-effect updates and use warm-started lambda paths; randomized SVD speedups can be inaccurate exactly at the rank that matters — validate ranks against exact SVD on subsamples.

## Dependencies and shared infrastructure

**Consumed from foundations**

- Resampling/bootstrap engine (block/stationary bootstrap with block-length selection, parallel RNG substreams) — powers RF resampling, Inoue-Kilian bagging, and generated-regressor bootstraps.
- HAC/long-run-variance inference with the library-wide default policy — desparsified LASSO, PDS, DML score variances, and HAC-aware tuning all defer to it.
- Factor-model estimation core (PCA/EM/QML) and number-of-factors criteria (Bai & Ng 2002; Ahn & Horenstein 2013; Onatski 2010) — this module's diffusion-index, targeted-predictor, 3PRF, FarmSelect, and FAVAR facades consume it; the criteria item from the research inventory is owned there, with kmax and standardization conventions exposed per the warnings above.
- Linear-Gaussian state-space engine (Kalman filtering/smoothing, square-root forms) — backs GP-SSM and the DMA/DMS filter bank.
- Critical-value engine — Chen-Liu sample-size-dependent critical values.
- Numerical optimizers, Philox-based reproducible parallel RNG, the unified forecast object, and the golden-value validation harness — used throughout.
- Exogenous-regressor (covariate) contract — the leakage-safe feature pipeline builds on it: lagged-feature construction, covariate alignment, and purging/embargo logic all consume the shared aligned interface, and multi-step ML forecasts with covariates inherit its known-future/scenario/auxiliary-forecast distinction instead of silently assuming covariates are known ahead.

**Consumed from other modules**

- forecasting-evaluation: conformal prediction (EnbPI/ACI/SPCI — the single model-agnostic implementation wraps any Forecaster from this module); forecast combination/stacking (Granger & Ramanathan 1984 constrained LS; the stacking item from the research inventory lives there, fed by this module's out-of-fold rolling-origin predictions); DM/Clark-West/MCS tests and density-forecast scoring consumed by the benchmark harness and QRF.
- nowcasting: real-time vintage/release-calendar layer for the contamination-aware harness's vintage-aware evaluation.
- diagnostics: STL residuals for the contrib anomaly-scoring utility; ownership coordination on Chen-Liu intervention detection.
- multivariate: VAR dynamics and IRF machinery behind the FAVAR bridge.

**Reassigned by the ownership map (pointer entries from this module's research inventory)**

- Sparse VAR (Basu & Michailidis 2015), VARX-L structured-penalty VARs (Nicholson, Matteson & Bien 2017, IJF), HLag hierarchical lag-order VARs (Nicholson, Wilms, Bien & Matteson 2020, JMLR), and HD Granger causality (Hecq, Margaritella & Smeekes 2023, JFEC) → multivariate module, consuming this module's CD engine, structured prox operators (including the Jenatton nested prox), and PDS logic.
- sg-LASSO-MIDAS (Babii, Ghysels & Striaukas 2022, JBES) → nowcasting module, consuming this module's SGL solver.
- High-dimensional / ML-controlled local projections (Adamek, Smeekes & Wilms 2024) → LP module, consuming desparsified LASSO and DML cross-fitting from here.
- Synthetic-control family (Abadie, Diamond & Hainmueller 2010; Doudchenko & Imbens 2016; Ben-Michael, Feller & Rothstein 2021; Arkhangelsky et al. 2021; Athey et al. 2021; Xu 2017; Chernozhukov, Wüthrich & Zhu 2021) → causal-panel companion package, consuming this module's ridge/elastic-net engines and foundations' bootstrap.
- Deep and foundation forecaster adapters → companion package, consuming this module's Forecaster protocol and benchmark harness.

**Exposed to others**

- The penalized-regression solver stack: coordinate-descent engine with strong rules and warm starts, structured prox operators (group, sparse-group, hierarchical/Jenatton), penalty-factor reweighting, lambda-path conventions, and IC tuning — consumed by multivariate (regularized VARs), nowcasting (sg-LASSO-MIDAS), LP (HD-LP), and the causal-panel companion.
- Time-series cross-validation splitters (rolling/expanding origin, purging/embargo, hv-block) — the tuning backbone for every module that fits anything by CV.
- The Forecaster trait/protocol and sklearn/torch interop layer — the registration point for external learners across backtesting, DML, and interpretation.
- Native random forest and boosting engines (reused by QRF and MRF) and the interpretation toolkit (block-permutation importance, lag-grouped Shapley, ALE) usable over any registered forecaster.

## Validation gallery

- glmnet coefficient paths — LASSO, elastic-net, and adaptive-LASSO full paths must match to 1e-6 (ridge to 1e-8 vs glmnet/sklearn), including objective scaling and standardization conventions.
- Medeiros, Vasconcelos, Veiga & Zilberman (2021, JBES) inflation benchmarks — RF, elastic-net, and CSR columns must be replicated on their public data.
- McCracken & Ng (2016, JBES) FRED-MD — diffusion-index factor estimates and downstream Stock-Watson forecasts must match published values (factor core via foundations).
- Adamek, Smeekes & Wilms (2023, JoE) / desla — desparsified-LASSO coverage must reproduce the R package and the paper's simulation coverage tables.
- Goulet Coulombe (2024, JAE) MacroRF — MRF outputs and the paper's GTVP plots must be reproduced.
- Giannone, Lenza & Primiceri (2021, Econometrica) — posterior sparsity distributions across all six applications must match the public replication code.
- DoubleML — iid-mode DML must match the reference package exactly; dependent-data mode must hit simulated coverage on known DGPs.
- Stata pdslasso / R hdm — PDS-LASSO treatment-coefficient estimates and SEs on their documented examples.
- R mboost glmboost — componentwise L2 boosting paths and AIC stopping.
- mgcv gam/gamm — P-spline fits with AR-error-corrected smoothing selection on matched setups.
- GPy/GPflow exact GPs — state-space GP posteriors must match exact GP posteriors to 1e-6 at small n.
- R gets — GETS/IIS/SIS selection paths on the package's documented examples.
- R tso — Chen-Liu outlier detection classifications and adjusted series (at better speed).
- shap package — TreeSHAP attributions on identical trees; R ALEPlot/iml for ALE curves.
- Boot & Nibbering (2019, JoE) — random-subspace FRED-MD results (Tier 4 gate).
