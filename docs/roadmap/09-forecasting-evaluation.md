# Module 09 — Forecasting and Evaluation

> Part of the time series econometrics library roadmap. Master plan: [ROADMAP.md](../../ROADMAP.md).

**This module is the production layer of the library: it defines the forecast object every model returns, runs the pseudo-out-of-sample backtests that generate forecast records, scores those records with the full battery of point and probabilistic accuracy measures, tests differences in predictive ability with the scheme-appropriate econometric test, combines forecasts and densities, calibrates and reconciles them, and renders the whole exercise as a reproducible evaluation report. Its central design commitment — encoded in the API, not just the documentation — is that the out-of-sample scheme, the loss function, and the comparison test must be mutually consistent, a discipline no existing library enforces.**

## Purpose and scope

The module covers everything that happens after a model is specified: multi-step forecasting strategy (direct versus iterated), transformation-aware back-transformation, the unified backtesting engine over fixed, rolling, and expanding schemes, point accuracy measures (scale-dependent, percentage, scaled, quantile), proper scoring rules for densities (CRPS, log score, weighted and multivariate scores), pairwise and multiple-model comparison tests (Diebold-Mariano through the Model Confidence Set), rationality and efficiency tests, density calibration diagnostics (PITs, Berkowitz, Knüppel, Rossi-Sekhposyan), point and density forecast combination, conformal prediction, bootstrap prediction intervals, conditional and scenario forecasting, hierarchical and temporal reconciliation, forecasting under structural breaks, and the benchmark models that anchor every skill score.

Its users span the library's whole audience. Applied macroeconomists and central-bank staff run recursive backtests, Clark-West tests against random-walk benchmarks, fan charts, conditional forecasts, and real-time vintage evaluations. Forecasting practitioners in industry run rolling-origin cross-validation over large panels, MASE/RMSSE leaderboards, combination schemes, conformal intervals, and MinT reconciliation. Researchers get the comparison-test stack (West asymptotics, Clark-McCracken bootstraps, SPA, MCS, fluctuation tests) that today exists only in scattered author code, wired to a backtest engine fast enough to make bootstrap-heavy inference routine.

Relative to the rest of the library, this module is a hub: every model-producing module (univariate, multivariate, volatility, bayesian, ML, nowcasting) returns this module's forecast object and is evaluated by this module's machinery. Per the master ownership map, it owns forecast objects, the backtesting engine, all accuracy measures, all forecast-comparison tests (volatility re-exports DM/GW/MCS/SPA), all density-forecast evaluation (bayesian and volatility consume it), forecast combination, the single model-agnostic conformal-prediction implementation, and reconciliation. It consumes the foundations resampling engine for all bootstrap mechanics, the HAC engine for all long-run variances, and the state-space engine for conditional forecasting.

## Where existing tools fall short

- **statsmodels** has no unified backtesting or pseudo-out-of-sample API at all — users hand-roll rolling loops that are slow and leakage-prone — and lacks Diebold-Mariano, Clark-West, Giacomini-White, SPA, MCS, fluctuation tests, density evaluation, combination, reconciliation, and conformal prediction entirely.
- **R forecast/fable** have the best point/interval workflow (tsCV, `accuracy()`, distribution-valued forecasts) but almost none of the econometric comparison-test stack: `dm.test` with HLN is roughly the ceiling — no Clark-West or Clark-McCracken, no West asymptotics, no Giacomini-White, no SPA/MCS, no fluctuation tests, no PIT-based density tests, no conditional forecasting — and R-level speed makes large bootstrap comparison exercises painful.
- **arch** (Kevin Sheppard) has good SPA/StepM/MCS and bootstrap primitives, but they are disconnected from any forecasting workflow: no backtest engine feeding them, no loss-function layer, no nested-model tests, no density scores.
- **Nixtla statsforecast/hierarchicalforecast** are fast and production-minded but evaluation is metrics-only: no statistical tests of predictive ability whatsoever, no density evaluation beyond quantile loss, no rationality tests, no real-time vintages, no scenario analysis.
- **scoringRules** (R) is the gold standard for proper scores but stops at scoring — no comparison tests, no backtesting, no combination — and there is no Python equivalent of comparable quality and coverage.
- **No library anywhere ties the OOS scheme to test validity.** The recursive-vs-rolling / West-vs-Giacomini-White distinction is ignored by every implementation, so practitioners routinely run DM tests whose asymptotics do not apply (nested models, recursive schemes).
- Conditional and scenario forecasting (Waggoner-Zha, Bańbura-Giannone-Lenza, structural scenarios, entropic tilting) lives only in Matlab toolboxes (BEAR, Dynare, replication zips) and internal central-bank code.
- Real-time vintage-aware evaluation (Croushore-Stark) is unsupported everywhere; evaluation-vintage choices in published work are undocumented and irreproducible.
- Density-forecast evaluation is fragmented across rugarch/GAS/author code; the estimation-robust Rossi-Sekhposyan and Knüppel tests exist only as replication archives, unintegrated with any backtest engine.
- Density combination (optimal pools, generalized pools, DMA, BPS, entropic tilting) has no maintained home in any language; eDMA and opera cover fragments.
- Conformal prediction for time series is siloed in ML packages (MAPIE, author repos) with no connection to econometric models, backtesting schemes, or comparison tests.
- Instability-aware evaluation (Giacomini-Rossi fluctuation, Rossi-Sekhposyan rationality-under-instability) exists only as Barbara Rossi's Matlab code.
- Multiple-testing discipline is absent from workflows: no tool tracks the universe of specifications actually tried, and multi-horizon joint tests (Quaedvlieg) are implemented nowhere.
- Probabilistic and cross-temporal reconciliation each have essentially one academic R implementation (Panagiotelis et al.; FoReco), none in Python.
- The "which loss / which test / which combination method when" pedagogy — elicitability, the combination puzzle, MAPE pathologies, proper-vs-improper weighted scores — is missing from all documentation.

## Inventory

Source priorities map to tiers: core → Tier 1, standard → Tier 2, advanced → Tier 3, frontier → Tier 4. Blocked/hv-block/K-fold cross-validation for dependent data is reassigned to the ML module per the ownership map (see Dependencies). The M5/retail intermittent-demand harness is contrib-tier per master ruling; the M4/OWA machinery stays. The "Forecasting under structural breaks" category in Tier 3 is a mandated addition.

### Tier 1 — Core (v1-blocking)

**Forecast objects and multi-step strategy**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Typed forecast objects (point / interval / density / path) | A first-class Forecast container carrying the full predictive distribution per horizon (analytic, quantile-grid, or draw-based); every model returns it, every evaluation function consumes it. | Medium | Design after fable + distributional: forecasts are distributions, points are functionals of them. Store the (origin, horizon, target_date) index triple explicitly — most evaluation bugs are misaligned origins. Lazy conversion between representations; draws in contiguous [n_draws x h] arrays for cache-friendly scoring. Validate round-tripping (draws → quantiles → interval score) against scoringRules and fable. |
| Direct vs iterated multi-step forecasting | Unified support for iterated (recursing a one-step model) and direct (horizon-specific projection) strategies, plus hybrids; iterated usually wins with well-specified AR, direct is robust to misspecification. | Medium | Marcellino, Stock & Watson (2006). Direct h-step errors are MA(h-1) by construction: downstream tests must use HAC with bandwidth ≥ h-1, automatically. Iterated density forecasts require simulating the recursion for nonlinear models, not recursing means. Validate against MSW (2006) tables on the 170-series US dataset and lpirfs/statsmodels direct-projection output. |
| Transformation-aware back-transformation with bias adjustment | Automatic handling of forecasting in logs/Box-Cox/differences with correct level-space median/mean semantics and cumulated-difference variances. | Medium | exp of the log-space point forecast is the median, not the mean; the mean needs the +0.5·sigma_h² correction (Granger-Newbold). Copy fable's design: transformations live in the model spec and the distribution is back-transformed, not the point. Cumulating differenced forecasts must cumulate covariances. Test back-transformed draw moments against analytic lognormal formulas. |

**Backtesting engine and OOS schemes**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Unified pseudo-out-of-sample backtesting engine (fixed / rolling / expanding) | The centerpiece API: iterate forecast origins over a sample split, refit per scheme, forecast horizons 1..H, collect a tidy origin-by-horizon evaluation object. | Medium | The scheme determines which comparison test is valid (recursive → West/Clark-West; fixed-width rolling → Giacomini-White): the backtest object records scheme and P/R and downstream tests read it. All preprocessing (scaling, seasonal adjustment, transformation choice, tuning) must be forced inside the training window — full-sample preprocessing is the most common leakage bug. Validate mechanics against fable stretch_tsibble/tsCV and Nixtla statsforecast cross_validation. |
| Refit cadence, warm starts, and parallel origin execution | Controls for re-estimating every k origins vs filter-through updating, warm-started optimizers, and multi-core execution across origins and replications. | Medium | Kalman-filter models filter through new observations without refitting — orders of magnitude faster, and what central banks actually do. Warm starts risk tracking a local optimum: periodically cold-start and compare. Counter-based parallel RNG (Philox, from foundations) for bit-reproducibility at any thread count. Benchmark target: full recursive AR/ETS backtest over M4 (100k series) in minutes, beating statsforecast. |

**Cross-validation for dependent data**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Rolling-origin evaluation / time series cross-validation (tsCV) | One-command "evaluate this model with h-step rolling-origin CV" convenience layer over the backtesting engine — the dependent-data analogue of K-fold and the default model-selection tool. | Low | Hyndman & Athanasopoulos (fpp3) rolling-origin design. Keep minimum training size explicit; return per-horizon error matrices (origins × horizons) for MSE-vs-horizon plots. Fold-construction primitives are shared with the ML module's CV splitters. Validate numerically against forecast::tsCV on AR simulations. |

**Point accuracy measures**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Scale-dependent measures (ME, MSE, RMSE, MAE, MdAE) | The basic loss battery with correct aggregation across origins/horizons/series and standard errors on the loss estimates themselves. | Low | Trivial math, nontrivial API: always return per-horizon and per-series breakdowns; never average RMSE across series of different scales (document why; point to MASE/RMSSE). Report HAC standard errors on mean loss. Validate against fabletools::accuracy() and Nixtla utilsforecast. |
| Percentage errors (MAPE, sMAPE) with guardrails | Included for compatibility with practice and the M-competitions, wrapped with warnings about known pathologies. | Low | MAPE explodes near zero and penalizes over-forecasts asymmetrically; sMAPE is not actually symmetric — implement the M4 definition (200·abs(e)/(abs(y)+abs(yhat))) and raise on zero denominators rather than returning inf. Goodwin & Lawton (1999), Hyndman & Koehler (2006). Validate sMAPE against M4 published values (Naive2 sMAPE = 13.564). |
| Scaled errors (MASE, RMSSE) | Scale-free errors normalized by in-sample (seasonal) naive MAE/MSE — the recommended default for cross-series comparison and the official M4/M5 metrics. | Low | Hyndman & Koehler (2006). The scaling denominator uses the TRAINING sample only, at the series' seasonal period; constant training series give zero denominators (NaN with warning). RMSSE is the M5 metric. Validate against M4/M5 official evaluation code and published scores. |
| Quantile / pinball loss and weighted quantile aggregation | Pinball loss for individual quantiles and weighted aggregation across a quantile grid — the standard for quantile-based probabilistic evaluation. | Low | rho_tau(e) = (tau − 1{e<0})·e; averaging over a fine grid approximates CRPS/2 — document the connection. Crossed quantile sets silently corrupt the aggregate: repair via monotone rearrangement first. Validate against M5 uncertainty evaluation code and scoringRules::qs. |

**Probabilistic scoring rules**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| CRPS with analytic and ensemble estimators | The workhorse proper score for full predictive distributions, in closed form for common distributions and as sample estimators for draws. | Medium | Closed forms per Jordan, Krüger & Lerch (2019) — also the validation target, to machine precision. NRG vs PWM ensemble estimators differ at O(1/m); expose the fair CRPS (Ferro 2014) finite-ensemble correction. Sort-based O(m log m) per observation; naive O(m²) double sums are slow and biased at small m. Negatively oriented (lower is better) library-wide. |
| Log score / predictive likelihood | Log predictive density at the realization — the local proper score, the Bayesian default, input to weighted-likelihood tests and pooling. | Low | Trivial for analytic densities; the trap is draws: KDE on MCMC output is badly biased in the tails — implement the mixture-of-parameters estimator per Krüger, Lerch, Thorarinsdottir & Gneiting (2021). Guard −inf when the realization falls outside draw support. Validate against scoringRules::logs_sample. |
| Interval (Winkler) score and coverage diagnostics | Proper score for central (1−alpha) intervals (width plus 2/alpha violation penalty), with empirical coverage and average-width reports. | Low | Winkler (1972); Gneiting & Raftery (2007, eq. 43). Pin the alpha convention in one place — off-by-alpha/2 bugs are common. Include unconditional (binomial) and conditional (Christoffersen 1998 Markov) coverage tests: correct average coverage can hide clustered violations. Validate against scoringRules and Christoffersen's published example. |

**Pairwise forecast comparison tests**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Diebold-Mariano test with HLN correction | The default test of equal predictive accuracy between two non-nested forecast streams under any loss. | Low | Diebold & Mariano (1995); make the Harvey-Leybourne-Newbold (1997) correction and t(T−1) critical values the default, not an option. Truncated-uniform LRV at lag h−1 can go negative — fall back to Bartlett with a warning. Refuse (or warn loudly) on nested models under recursive schemes — degenerate; route to Clark-West. Include multivariate DM (Mariano & Preve 2012). Validate against forecast::dm.test and multDM; see Diebold (2015) for intended use. |
| Clark-West test for nested models | Adjusted-MSPE t-test for nested comparisons, correcting the noise term that biases naive DM against the larger model. | Low | Clark & West (2006; 2007). adj-MSPE_t = e1²_t − e2²_t + (f1_t − f2_t)²; HAC t-test, standard normal critical values (slightly undersized). One-sided by construction — print the one-sided p-value by default. Validate on Welch-Goyal return-predictability replications and Todd Clark's published code output. |

**Rationality and efficiency tests**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Mincer-Zarnowitz rationality/efficiency regressions | Regression of realizations on forecasts; joint test a=0, b=1, plus orthogonality extensions with information-set variables. | Low | Mincer & Zarnowitz (1969). Multi-step: HAC, with Hodrick (1992)-style standard errors offered for long horizons. HAC-Wald over-rejects in small samples — offer wild/block bootstrap p-values. Include the weak-efficiency variant on forecast errors (Patton-Timmermann framing). Validate on classic SPF evaluations (Croushore). |

**Density evaluation and calibration**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| PIT computation, histograms, and uniformity diagnostics | Probability integral transforms through predictive CDFs — the foundational density-calibration diagnostic (uniform iid under correct calibration at h=1). | Medium | Diebold, Gunther & Tay (1998). Draw-based forecasts: empirical CDF with randomization between adjacent draws; discrete/mixed outcomes need randomized PITs (Czado, Gneiting & Held 2009); h>1 PITs are serially dependent under the null — test only uniformity and say so. Default 10 bins with binomial bands, plus KS/CvM with block-bootstrap p-values. Validate against rugarch/scoringRules examples and the DGT exchange-rate application. |
| Berkowitz LR test | Inverse-normal-transformed PITs tested for zero mean, unit variance, no AR(1) — a compact, higher-power calibration test for small samples. | Low | Berkowitz (2001). LR3 and tail-focused LR2 variants. Clip inverse-normal transforms of PITs at exactly 0/1 (infinite otherwise) with warning. For h>1 the AR(1) independence component is invalid — auto-drop it and test only N(0,1) marginals. Validate against Berkowitz's published example and rugarch/GAS. |

**Point forecast combination**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Simple combinations: equal weights, median, trimmed/winsorized means | The robust baselines that are extremely hard to beat (the forecast combination puzzle); median and trimmed means guard against outlier forecasts. | Low | Estimated "optimal" weights routinely lose to 1/N because weight-estimation error swamps the gains — Smith & Wallis (2009); Claeskens et al. (2016). Docs must lead with this; default to equal weights unless N is small and the sample long. Validate against Stock & Watson (2004) combination results. |
| Bates-Granger inverse-MSE weights | Weights proportional to inverse error (co)variance — the original 1969 rule; the diagonal version is the practical default. | Low | Bates & Granger (1969). Full covariance weights explode under near-collinear forecasts — default to diagonal, offer Ledoit-Wolf shrinkage for the full version; support rolling and discounted variance estimation. Validate against Timmermann (2006) Handbook chapter tables. |
| Granger-Ramanathan regression combinations (incl. constrained and NNLS) | OLS of realizations on forecasts for combination weights — unconstrained, sum-to-one, and nonnegativity-constrained (NNLS) variants. | Low | Granger & Ramanathan (1984). Collinearity makes unconstrained weights garbage; the convex-combination constraint via NNLS/QP is the robust production choice and scales to hundreds of forecasters — Conflitti, De Mol & Giannone (2015). Validate constrained weights against their ECB SPF application. |

**Benchmark models**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Benchmark zoo: naive, seasonal naive, drift, mean, random walk with drift | The mandatory baselines with correct analytic prediction intervals; auto-included in every evaluation report; macro reports default to RMSE ratios vs the random walk. | Low | Interval formulas per fpp3 5.5 (naive sigma_h = sigma·sqrt(h); seasonal naive sigma·sqrt(floor((h−1)/m)+1)). Must be exactly right — they anchor every skill score. M4's Naive2 (seasonal-naive-after-seasonal-adjustment) reimplemented exactly, validated to M4 published numbers. Document the Atkeson & Ohanian (2001) four-quarter RW inflation benchmark as a recipe. |
| Theta method (and optimized/dynamic variants) | The M3-winning method — equivalent to SES with drift — still a top-tier univariate benchmark, cheap and shockingly hard to beat. | Medium | Assimakopoulos & Nikolopoulos (2000); implement via the Hyndman & Billah (2003) SES-with-drift equivalence. Seasonally adjust first (multiplicative classical decomposition), then re-seasonalize — matching M4's Theta requires the exact 90% autocorrelation seasonality test. Validate against forecast::thetaf and M4's published Theta sMAPE (12.309). |

### Tier 2 — Standard (expected of a serious library)

**Forecast objects and multi-step strategy**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Path forecasts and joint multi-horizon evaluation | Joint predictive distribution across horizons 1..H (simulated paths) for path-dependent questions, simultaneous path bands, and path scoring. | Medium | Jordà & Marcellino (2010): Scheffé-type simultaneous bands from the cross-horizon error covariance; pointwise bands badly undercover the whole path. Score paths with energy score or joint log score; store paths, don't marginalize prematurely. Validate band coverage against Jordà-Marcellino Monte Carlo and cross-check vs Wolf & Wunderli (2015) bootstrap joint bands. |

**Point accuracy measures**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Relative measures and skill scores (Theil's U, relative RMSE, OWA) | Ratios of model loss to a benchmark's loss, including M4's Overall Weighted Average (mean of relative sMAPE and relative MASE vs Naive2). | Low | Build in the macro convention of RMSE ratios vs a random walk (Atkeson & Ohanian 2001 style) as the default macro report. Ratios of averages vs averages of ratios differ — follow the M4 competition definition exactly. Validate OWA against the M4 leaderboard (winning hybrid OWA = 0.821). |

**Probabilistic scoring rules**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Event-probability evaluation: Brier score, RPS, ROC/AUC, CORP reliability diagrams | Evaluation of event-probability forecasts (recession, rate hike): Brier with calibration/resolution decomposition, RPS for ordered categories, ROC, and isotonic reliability diagrams. | Medium | Brier (1950); Murphy (1973) decomposition. Implement the CORP approach of Dimitriadis, Gneiting & Jordan (2021): PAV-based reliability diagrams instead of arbitrary binning, with miscalibration/discrimination/uncertainty decomposition. Serial dependence invalidates naive calibration inference — block-bootstrap bands. Validate against the R reliabilitydiag package. |

**Pairwise forecast comparison tests**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Giacomini-White conditional predictive ability test | Tests whether one forecast METHOD (model + window + everything) beats another conditionally on available information; valid for nested models under fixed-width rolling estimation; can say WHEN each method wins. | Medium | Giacomini & White (2006). Wald statistic on instrumented loss differentials (constant + lagged dL); chi-square(q) limit; requires non-vanishing estimation error, i.e. FIXED-width rolling windows — the API must check the backtest scheme and refuse recursive schemes with an explanatory error. Ship the conditional decision rule. Validate against murphydiagram/afmtools and the paper's DM exchange-rate example. |
| Forecast encompassing tests | Tests whether forecast A contains all useful information in B (lambda = 0 in the combination regression) — the decision tool for whether combining adds value. | Low | Chong & Hendry (1986); HLN (1998) small-sample-corrected statistic on d_t = e1_t(e1_t − e2_t). HAC for multi-step; ENC-NEW for nested models. Tie output to the combination module ("encompassing rejected → consider combining, suggested weight lambda-hat"). Validate against the HLN (1998) published example. |

**Multiple-model comparison and data snooping**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| White Reality Check and Hansen SPA test | Tests whether the best of a large model universe beats a benchmark, correcting for search; SPA studentizes and defuses irrelevant bad models. | High | White (2000); Hansen (2005) with consistent/lower/upper p-values — report all three. Stationary bootstrap (Politis & Romano 1994) of the loss-differential panel with automatic block length (Politis & White 2004; Patton-Politis-White 2009 correction), via the foundations resampling engine. The null re-centering (Hansen's sqrt(2 log log T) threshold) must match the paper exactly. Validate against arch.bootstrap.SPA and Hansen's Ox code on the same loss matrix. |
| Model Confidence Set (MCS) | Constructs the set of models containing the best with given confidence via sequential bootstrap elimination — the modern way to report "these 4 of 30 are statistically indistinguishable at the top". | High | Hansen, Lunde & Nason (2011). T_R and T_max statistics with semi-quadratic elimination; MCS p-values are running maxima; O(m²) loss-differential matrices computed incrementally for large m. Require an explicit seed; document loss- and seed-dependence. Validate against arch.bootstrap.MCS and the R MCS package on the HLN (2011) empirical example. |

**Rationality and efficiency tests**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Coibion-Gorodnichenko information-rigidity regressions | Regression of ex-post forecast errors on ex-ante revisions; the slope maps to information rigidity in sticky/noisy-information models. | Low | Coibion & Gorodnichenko (2015). Simple OLS with HAC; the object construction (consensus vs individual, fixed-event revision alignment) is where users err — provide a dedicated survey-panel adapter; individual-level versions need panel clustering (pair with Bordalo et al. 2020 overreaction regressions in docs). Validate against CG's published SPF CPI-inflation coefficient (~1.2). |

**Density evaluation and calibration**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Amisano-Giacomini weighted likelihood-ratio comparison | DM-type test on (weighted) log-score differentials for comparing two density forecasts, with weights emphasizing tails or center. | Medium | Amisano & Giacomini (2007). Region-selective weights must use the propriety-safe censored-likelihood construction of Diks, Panchenko & van Dijk (2011) — the original indicator weighting can spuriously favor densities that shift mass out of the region. Reuses the DM HAC/HLN machinery. Validate against DPvD's published comparison tables. |

**Point forecast combination**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Time-varying combination weights (discounted MSFE, rolling, regime-switching) | Weights that adapt over time: discounted past performance, rolling Bates-Granger, and Markov-switching/threshold weights. | Medium | Stock & Watson (2004) discounted MSFE (delta in {0.9, 0.95, 1}) — cheap, robust, the default adaptive option. Elliott & Timmermann (2005) regime-switching weights depend on the regime-switching module — flag the dependency. Validate discounted-MSFE combination against Stock-Watson's seven-country output growth results. |

**Density combination and pooling**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Linear opinion pools with optimal weights (Geweke-Amisano, Hall-Mitchell) | Convex mixtures of predictive densities with weights maximizing past log score — the canonical density-combination method for macro. | Medium | Hall & Mitchell (2007); Geweke & Amisano (2011) — the pool typically outperforms even the best component. Maximize sum log(sum_j w_j f_j(y_t)) over the simplex — concave; EM (mixture interpretation) or projected Newton; log-sum-exp stabilization essential. Validate against the Geweke-Amisano S&P 500 application and overlapping eDMA/DynamicPools output. |
| Bayesian model averaging and predictive-likelihood weighting | Posterior-model-probability weights and the more forecast-relevant predictive-likelihood weights over a training window. | Medium | Standard BMA weights are dominated by in-sample fit; Eklund & Karlsson (2007) argue for predictive likelihood. Log-marginal differences of hundreds → softmax overflow: always work in log space with a max-shift. Marginal likelihoods come from the bayesian module; the weighting layer here is generic over any model exposing log predictive scores. Validate on Wright (2008) BMA inflation-forecasting results. |

**Conformal prediction**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Split/sequential conformal baseline for time series | Distribution-free intervals from calibration-set residual quantiles on a trailing window — the simple entry point, with honest caveats about exchangeability. | Medium | Vovk et al. (2005) foundations. Exchangeability fails for time series: plain split conformal has no finite-sample guarantee and undercovers under shift — surface this and default to adaptive variants for nonstationary data. Use conformalized quantile regression (Romano, Patterson & Candès 2019) as the default score for heteroskedastic series. Validate coverage in simulation against MAPIE's time-series module behavior. |

**Bootstrap prediction intervals**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Residual-based bootstrap prediction intervals with parameter uncertainty | AR/ARMA(-GARCH) intervals from resampled residual paths that also resample parameter estimates — fixing plug-in Gaussian undercoverage. | Medium | Thombs & Schucany (1989); implement Pascual, Romo & Ruiz (2004; 2006 for GARCH) as the default (no backward representation, includes estimation uncertainty). Bias-correct AR coefficients first (Kilian 1998 bootstrap-after-bootstrap) or intervals inherit small-sample bias. Resampling via the foundations engine. Validate coverage against PRR (2004) Monte Carlo tables and R forecast bootstrap intervals. |

**Conditional and scenario forecasting**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Conditional forecasts in VARs (Waggoner-Zha, Bańbura-Giannone-Lenza) | Forecasts conditional on hard future paths of some variables — the central-bank scenario workhorse. | High | Waggoner & Zha (1999) exact Gibbs sampler; Bańbura, Giannone & Lenza (2015) recast conditioning as state-space missing data solved with the simulation smoother (foundations engine) — implement BGL as the engine, WZ as validation. Long conditioning horizons create ill-conditioned constraint systems: QR, never normal equations. Report the distribution of constrained shocks as a scenario-plausibility diagnostic. Validate against the ECB BEAR toolbox and WZ's example. |
| Fan charts (two-piece normal and quantile-path parameterizations) | Bank-of-England-style asymmetric fan charts from (mode, uncertainty, skew) per horizon, plus generic quantile fans from any forecast object. | Low | Britton, Fisher & Whitley (1998); two-piece normal quantile math per Wallis (2004). The BoE skew parameter has multiple conventions — implement Wallis's and document. Deliver as a distribution type and a plotting helper. Validate quantiles against published BoE fan chart parameter tables. |

**Hierarchical and temporal reconciliation**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Aggregation-constraint tooling: bottom-up, top-down, middle-out | Infrastructure for forecast hierarchies (summing matrix S) with the classical single-level approaches as baselines. | Low | fpp3 ch. 11 framing. The summing matrix is sparse from day one (retail hierarchies exceed 10^5 nodes). Top-down proportions (Gross-Sohl 1990) documented as biased below the top level. Validate structures against hts/fabletools and Nixtla hierarchicalforecast. |
| MinT (minimum-trace) reconciliation | Optimal linear reconciliation minimizing total error variance given the base-forecast error covariance — the modern default (with shrinkage covariance). | Medium | Wickramasuriya, Athanasopoulos & Hyndman (2019). G = (S'W⁻¹S)⁻¹S'W⁻¹ with Schäfer-Strimmer-shrunk residual covariance; never form explicit inverses — sparse SPD Cholesky; wls_struct/wls_var/ols fallbacks. Implement the nonnegative variant (Wickramasuriya et al. 2020) for the negative-forecast wart. Validate against fabletools::min_trace and Nixtla hierarchicalforecast on the Australian tourism dataset. |

**Benchmark models and competition machinery**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Competition evaluation harness (M4 OWA, Naive2) | Exact reimplementation of the M4 evaluation pipeline so users and CI can reproduce leaderboard numbers — the credibility anchor for the whole evaluation stack. | Medium | Makridakis, Spiliotis & Assimakopoulos (2020). Ship M1/M3/M4 datasets as optional downloads and a one-command "reproduce M4 benchmarks" script; CI asserts published scores to 3 decimals — this doubles as an integration test of naive/Theta/ETS/backtesting at once. Per master ruling the M5/retail intermittent-demand harness (WRMSSE/WSPL pipeline) is contrib-tier — Nixtla serves it well — while the RMSSE/pinball metrics themselves remain in Tier 1. |

**Production utilities**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Quantile-crossing repair via monotone rearrangement | Sorts crossed quantile forecasts into a valid monotone quantile function — provably never hurts accuracy; needed whenever quantiles are forecast separately. | Low | Chernozhukov, Fernández-Val & Galichon (2010): rearrangement weakly improves every quantile loss. O(q log q) per (origin, horizon). Apply automatically (with a log message) before computing interval/pinball scores from quantile grids — prevents silent corruption of probabilistic evaluation. |
| Evaluation reports, leaderboards, and "which-test-when" guidance layer | One-command evaluation report: per-horizon losses with HAC standard errors, benchmark ratios, scheme-appropriate DM/CW tests auto-selected, calibration diagnostics, MCS over the model set. | Low | The report engine encodes the decision tree (nested? → Clark-West; rolling fixed window? → GW; many models? → MCS not pairwise DM; density forecast? → PIT + Berkowitz + CRPS). Embed seeds, scheme metadata, and library version for reproducibility. The accompanying documentation chapter mapping each situation to the right test is as valuable as the code. |

### Tier 3 — Advanced (differentiators)

**Forecast objects and multi-step strategy**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Fixed-event forecast support | Handling of fixed-event forecasts ("GDP growth for calendar 2026") and their revision sequences — the natural format of surveys (SPF, Consensus). | Low | Nordhaus (1987) weak-efficiency test: revisions uncorrelated with lagged revisions. Requires an (event, announcement date) index; include fixed-horizon↔fixed-event approximate conversion (Dovern, Fritsche & Slacalek 2012 weighting). Validate against published Nordhaus-test results on SPF data (Clements 2014). |

**Backtesting engine and OOS schemes**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Real-time data vintage evaluation | Backtesting against real-time vintages (ALFRED-style): the forecaster at origin t sees only data as published at t; evaluation targets first-release, second-release, or latest actuals. | High | Croushore & Stark (2001) — model rankings can flip with the vintage used for estimation and for actuals. Consumes the foundations vintage store (observation × publication triangle, ragged edges); this module owns the mandatory "actuals policy" parameter in every evaluation call. Almost no open library supports this; a killer feature for central banks. Validate against Croushore-Stark replication files and the Philadelphia Fed real-time dataset. |
| Data-snooping audit trail / evaluation registry | Records every specification evaluated in a project so multiple-comparison corrections (RC/SPA/StepM/MCS) run over the true universe of models tried, not the survivors. | Medium | White (2000) framed data snooping as the central OOS credibility problem; the practical failure is running SPA over 3 finalists after trying 300 specs. Lightweight registry (hash of spec + loss series persisted per experiment) plus workflow docs. Low numerical risk; high design value; no existing package does this. |

**Point accuracy measures**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Elicitability framework and Murphy diagrams | Enforces that the loss matches the functional forecast (mean↔MSE, median↔MAE, quantile↔pinball, expectile↔asymmetric squared); Murphy diagrams show dominance across all consistent scoring functions. | Medium | Gneiting (2011) — evaluating a median forecast with MSE is a category error the API should flag (forecast objects know their functional). Murphy diagrams per Ehm, Gneiting, Jordan & Krüger (2016): one forecast dominates iff its elementary-score curve is below everywhere. Validate against the R murphydiagram package. Resolves "wins on MAE, loses on MSE" confusion; earns a documentation chapter. |

**Probabilistic scoring rules**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Threshold- and quantile-weighted scores (twCRPS, censored/conditional likelihood) | Proper scores emphasizing a region of interest (left tail for downside risk, GDP-at-risk) without destroying propriety. | Medium | Gneiting & Ranjan (2011) twCRPS; Diks, Panchenko & van Dijk (2011) censored/conditional likelihood. Critical: indicator-weighted log scores are NOT proper (they reward moving mass out of the region) — weighted log-score tests must use the DPvD censored construction, enforced in the API. Validate twCRPS against scoringRules and reproduce a DPvD empirical table. |
| Multivariate scores: energy score, variogram score, Dawid-Sebastiani | Proper scores for joint (multivariate or path) forecast distributions — required where marginal scores miss dependence errors. | Medium | Energy score (Gneiting & Raftery 2007), sample estimator O(m²d) — SIMD-friendly; variogram score of order p (Scheuerer & Hamill 2015) is more sensitive to misspecified correlation; Dawid-Sebastiani for Gaussian-ish cases. Document the energy score's low power against dependence misspecification (Pinson & Tastu 2013). Validate against scoringRules multivariate functions. |

**Pairwise forecast comparison tests**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| West (1996) / McCracken asymptotics for parameter-estimation error | Correct asymptotic variance for OOS moments when forecasts come from estimated models and the null concerns population-level predictive ability; adjusts DM/MZ-type statistics via pi = lim P/R. | High | West (1996); West & McCracken (1998) for regression-based tests. Requires model-supplied score/Jacobian objects — design the model trait accordingly. The correction vanishes when P/R → 0 or when the same loss is used for estimation and evaluation — document these escape hatches. Essentially unimplemented anywhere; validate against West-McCracken published Monte Carlo tables. |
| Clark-McCracken MSE-F and ENC-NEW with bootstrap critical values | Higher-power nested-comparison tests with nonstandard limiting distributions (functionals of Brownian motion in pi), via asymptotic tables or a restricted-DGP bootstrap. | High | Clark & McCracken (2001) ENC-NEW; McCracken (2007) MSE-F tables indexed by pi and extra regressors; Clark & McCracken (2005) fixed-regressor bootstrap. Implement table interpolation (via the foundations critical-value engine) and the restricted bootstrap that re-runs the whole recursive OOS exercise — the Monte-Carlo-speed showcase for the Rust core. Validate against McCracken's published tables and Clark's code output. |

**Multiple-model comparison and data snooping**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Romano-Wolf StepM stepwise multiple testing | Stepwise bootstrap controlling family-wise error while identifying WHICH models beat the benchmark — more informative than RC/SPA. | Medium | Romano & Wolf (2005); Hsu, Hsu & Kuan (2010) SPA-adapted version. Reuses the RC/SPA bootstrap engine; the stepwise max-t construction is straightforward once that exists. Expose FDR alternatives (Barras, Scaillet & Wermers 2010 style) for very large universes. Validate against the rwolf Stata routine and Romano-Wolf's published simulations. |

**Instability and time-varying performance**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Giacomini-Rossi fluctuation test | Rolling DM statistic against sup-type critical-value bands — detects WHEN relative performance broke down instead of averaging it away. | Medium | Giacomini & Rossi (2010). DM over centered rolling windows of size m = mu·P; critical values depend on mu — hard-code their Table 1 with interpolation. The fluctuation plot with bands is the main deliverable; pair with Rossi's (2013) one-time-reversal variant. Only exists as Barbara Rossi's Matlab code. Validate against her replication files for the ECB exchange-rate example. |
| Rossi-Sekhposyan rationality and comparison under instability | Fluctuation-rationality tests (rolling MZ Wald with sup critical values) and instability-robust predictive-ability tests with power against episodic predictability. | Medium | Rossi & Sekhposyan (2016); Rossi (2021) is the survey to structure the docs chapter. Reuses the fluctuation-test scaffolding; critical values from tabulated sup-Wald functionals. Validate against Rossi's posted Matlab toolbox on the SPF application. |

**Forecasting under structural breaks**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Optimal estimation-window selection under breaks | Chooses how much pre-break data to include in estimation, trading the bias of contaminated data against the variance of short windows — often the whole sample forecasts better even with a known break. | Medium | Pesaran & Timmermann (2007). Implement trade-off-based window choice and cross-validation-based selection over candidate windows, integrated with the backtest engine so window choice is itself evaluated out of sample. Validate against the Pesaran-Pick-Pranovich simulation designs. |
| Robust weighting and averaging over estimation windows | Forecasts averaged over multiple estimation windows, or observations down-weighted by exponential/robust weights — insurance against breaks of unknown timing and size. | Medium | Pesaran, Pick & Pranovich (2013) optimal-weights results; Pesaran & Timmermann (2007) AveW averaging. Implement exponential down-weighting and average-over-windows as first-class estimation options in the backtest engine. Validate against the Pesaran-Pick-Pranovich (2013) simulation designs — the named golden target for this category. |
| Post-break shrinkage forecasts | Shrinks post-break parameter estimates toward full-sample (or pre-break) estimates, trading break-induced bias against post-break estimation noise. | Medium | Shrinkage between post-break and full-sample estimators with weights tied to estimated break size and post-break sample length; connects to the combination layer (window-based forecasts are just another pool). Validate in the Pesaran-Pick-Pranovich simulation designs against the paper's reported MSFE rankings. |
| Break-aware forecast design hooks | Integration layer so detected breaks (from the diagnostics module's break tests) flow automatically into forecast design: window selection, down-weighting, and shrinkage triggered by estimated break dates. | Low | Consumes break dates/confidence sets from diagnostics (Bai-Perron-style output); emits window/weighting recommendations with the estimation-uncertainty caveats documented (act on estimated breaks only when large). Validated end-to-end within the Pesaran-Pick-Pranovich designs. |

**Rationality and efficiency tests**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Patton-Timmermann monotonicity and bounds tests | Tests of internal-consistency properties any rational forecast must satisfy regardless of the unknown loss: MSE nondecreasing in horizon, revision-variance and covariance bounds — for surveys and central banks. | Medium | Patton & Timmermann (2012); inequalities via Wolak (1989) or the paper's bootstrap moment-inequality approach. Needs multi-horizon forecast panels. Companion result: optimal forecasts under asymmetric loss are biased (Patton & Timmermann 2007) — MZ rejections are not proof of irrationality. Validate against their Greenbook application results. |
| Elliott-Komunjer-Timmermann flexible-loss rationality test | Jointly estimates the forecaster's loss-asymmetry parameter (lin-lin/quad-quad family) by GMM and tests rationality given that loss — separates "irrational" from "asymmetric loss". | Medium | Elliott, Komunjer & Timmermann (2005; 2008 application). GMM with information-set instruments; J-test conditional on estimated asymmetry; report the asymmetry parameter with CI. Weak-instrument issues when forecast errors are nearly unpredictable — warn. Validate against EKT (2008) published SPF/Greenbook estimates. |

**Density evaluation and calibration**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Knüppel moment-based calibration test | Tests PIT uniformity via raw moments jointly with a dependence-robust covariance — works for multi-step forecasts where Berkowitz's independence assumption fails. | Medium | Knüppel (2015). GMM-type Wald on the first 4 standardized PIT moments with HAC covariance; good size with overlapping multi-step forecasts — exactly the empirically relevant case. Straightforward on top of the foundations HAC engine. Validate against Knüppel's published Monte Carlo and his Bundesbank code. |

**Point forecast combination**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Shrinkage combinations: toward equal weights, egalitarian ridge/LASSO | Weight estimators shrunk toward 1/N — the theoretically motivated response to the combination puzzle; LASSO variants select members and shrink survivors toward equality. | Medium | Stock & Watson (2004) shrinkage combinations; Diebold & Shin (2019) egalitarian LASSO (two-step select-then-shrink). Penalized regression centered at the equal-weight vector (solver from the ML module); cross-validate the penalty on a rolling basis — no lookahead. Validate against Diebold-Shin's Eurozone survey application. |
| Online aggregation: EWA, Hedge, ML-Poly, adaptive online learning | Prediction-with-expert-advice algorithms with finite-sample regret guarantees — no stationarity assumptions; ideal for production streams and mixed econometric+ML pools. | Medium | Cesa-Bianchi & Lugosi (2006) theory; implement EWA/Hedge, ML-Poly, and gradient-trick versions following the R opera package (Gaillard & Goude) — the validation target. Learning-rate tuning is the pitfall; ML-Poly's adaptive rates avoid it. Report cumulative regret vs best expert and best convex combination. Bridges to the e-value monitoring item. |
| Complete subset regressions (CSR) | Averages forecasts from all (or sampled) k-variable subsets of a predictor set — beats single-model selection in noisy environments. | Medium | Elliott, Gargano & Timmermann (2013). n-choose-k explodes: sample subsets uniformly beyond ~10k combinations (loses little). Embarrassingly parallel — a Rust showcase. Choose k by rolling OOS. Validate against EGT's equity-premium application (Welch-Goyal data). |

**Density combination and pooling**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Generalized pools: logarithmic, beta-transformed, calibrated | Beyond-linear pooling: log pools (sharper), beta-transformed linear pools fixing linear-pool overdispersion, generalized families with time-varying tilting. | Medium | Gneiting & Ranjan (2013) — linear pools of calibrated densities are systematically overdispersed; BLP recalibrates. Kapetanios, Mitchell, Price & Fawcett (2015) generalized pools. Log-pool normalizing constants need adaptive quadrature for non-Gaussian components. Validate BLP against Gneiting-Ranjan's simulation examples. |
| Dynamic model averaging and selection (DMA/DMS) | Kalman-style recursive updating of model probabilities with forgetting factors — parameters and model weights both drift; popular for inflation and commodities. | Medium | Raftery, Kárný & Ettler (2010); Koop & Korobilis (2012). Two forgetting factors (alpha for probabilities, lambda for parameters); numerically simple but sensitive — offer grid search over forgetting with honest OOS evaluation. Validate against the eDMA R package reproducing Koop-Korobilis inflation results. |

**Conformal prediction**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Adaptive conformal inference (ACI / AgACI) | Online update of the miscoverage level by gradient steps on realized coverage errors — restores long-run coverage under arbitrary distribution shift; the practical default for streams. | Medium | Gibbs & Candès (2021): alpha_{t+1} = alpha_t + gamma(alpha − err_t); the step size matters — implement AgACI (Zaffran et al. 2022), aggregating over gamma with online learning (reuses the EWA module). Guard alpha_t leaving (0,1) → infinite intervals; report interval-width paths. Validate against the authors' released code on their election/stock examples. |
| EnbPI (ensemble batch prediction intervals) | Conformal-style intervals from leave-one-out ensemble residuals of bootstrap models — no data splitting or exchangeability requirement (needs error stationarity); suited to ML regressors. | Medium | Xu & Xie (2021). Fit B bootstrap models, aggregate LOO residuals, slide the residual window forward. Approximate marginal coverage under strongly mixing errors — state the assumption. B fits × rolling origins is compute-heavy: parallelize in the core. Validate against the authors' Python release and MAPIE's EnbPI. |

**Bootstrap prediction intervals**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Sieve and block bootstrap prediction intervals (model-free) | Intervals without trusting a parametric model: AR-sieve for linear processes, block bootstrap for general stationary series; bagged prediction paths for arbitrary point forecasters. | Medium | Bühlmann (1997) sieve; Pan & Politis (2016) is the definitive treatment — follow their taxonomy (forward vs backward, fitted vs predictive residuals; predictive residuals fix undercoverage). Block length via Politis-White (foundations engine). Validate against Pan-Politis simulation coverage tables. |

**Conditional and scenario forecasting**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Entropic tilting of predictive distributions | Minimally reweights forecast draws (KL-closest distribution) to satisfy moment conditions — e.g., tilt BVAR draws toward a survey nowcast or a market rate path; soft, model-agnostic conditioning. | Medium | Robertson, Tallman & Whiteman (2005); Krüger, Clark & Ravazzolo (2017) for tilting toward SPF. Solve the convex dual by Newton; overflow is the trap — center the moment functions and use log-sum-exp; monitor effective sample size of tilted weights and warn on collapse. Validate against Krüger-Clark-Ravazzolo replication output. |

**Hierarchical and temporal reconciliation**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Temporal and cross-temporal reconciliation | Coherence across temporal aggregation levels (monthly summing to quarterly/annual — THieF) and jointly across hierarchy and time — directly relevant to macro (monthly indicators vs quarterly GDP). | High | Athanasopoulos, Hyndman, Kourentzes & Petropoulos (2017) temporal hierarchies; Di Fonzo & Girolimetto (2023) cross-temporal framework — their FoReco R package is the validation target. Exploit Kronecker identities rather than materializing the cross-temporal structure. Macro showcase: reconcile monthly nowcasts with quarterly model forecasts. |

### Tier 4 — Frontier (research-grade, each gated on a named validation target)

**Pairwise forecast comparison tests**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Multi-horizon comparison: uniform and average SPA (Quaedvlieg) | Jointly compares two models across horizons 1..H with family-wise error control over the horizon dimension — replaces H separate DM tests. | High | Quaedvlieg (2021): uniform SPA (min-t over horizons) and average SPA with a moving-block bootstrap of the joint loss-differential distribution. The bootstrap must resample loss-differential VECTORS (all horizons together) to preserve cross-horizon dependence. Gate: match Quaedvlieg's replication code (JBES site). |
| Anytime-valid / e-value sequential forecast comparison | E-process tests of forecast superiority valid under optional stopping and continuous monitoring — the right tool for dashboards that check "is the new model better yet?" weekly. | High | Henzi & Ziegel (2022); Choe & Ramdas (2023); Arnold, Henzi & Ziegel (2023) for general losses. Build the e-process as a product of conditional bets (mixture/plug-in lambda); report anytime-valid confidence sequences on the mean loss differential. Gate: match the eprob/epredict R code from the papers' authors. |

**Density evaluation and calibration**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Rossi-Sekhposyan estimation-robust density tests | KS and CvM tests on PITs whose critical values account for parameter estimation error and the OOS scheme — the correct-size version of PIT tests for model-based densities. | High | Rossi & Sekhposyan (2019). Critical values via simulation/bootstrap replicating the estimation scheme — expensive, hence a flagship use of the fast Rust backtest engine. Gate: match their QE replication files. |
| Recalibration: isotonic distributional regression and beta-recalibration | Post-processing that maps miscalibrated predictive CDFs into calibrated ones using past PITs — cheap accuracy gains in production. | High | Henzi, Ziegel & Gneiting (2021) IDR (PAV-based, tuning-free); beta-calibration of PITs (Gneiting & Ranjan 2013). IDR fitting is O(n²) worst case, much faster with clever PAV — a Rust showcase. Lookahead trap: recalibration maps must be fit on a rolling basis inside the backtest. Gate: match the isodistrreg R package. |

**Point forecast combination**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Feature-based meta-learning combination (FFORMA-style) | Learns combination weights as a function of time-series features via gradient boosting — M4 runner-up; the industrial approach for many heterogeneous series. | High | Montero-Manso, Athanasopoulos, Hyndman & Talagala (2020). Needs a tsfeatures/catch22-equivalent extractor (Rust) and an XGBoost-like learner with a custom softmax-weighted-loss objective (scope as an optional ML-module/Python dependency). Gate: reproduce FFORMA's published M4 OWA (0.838) within tolerance. |

**Density combination and pooling**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Bayesian predictive synthesis (BPS) | Decision-theoretic generalization of pooling: agent densities enter a dynamic latent-factor synthesis function that learns bias, miscalibration, and inter-forecaster dependence — state of the art in density combination. | Research-grade | McAlinn & West (2019); McAlinn, Aastveit, Nakajima & West (2020) multivariate. MCMC over dynamic latent agent-coefficient states (FFBS within Gibbs, via the foundations state-space engine) — computationally heavy; exactly what a fast Rust FFBS core makes practical. Gate: match McAlinn-West's posted replication code on their US inflation application. |

**Conformal prediction**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| SPCI (sequential predictive conformal inference) | Improves EnbPI by fitting a quantile regression (random forest) on past conformity scores to exploit residual serial dependence — materially narrower intervals at the same coverage. | High | Xu & Xie (2023). Requires a quantile random forest on lagged residuals refit each step — expensive; offer a lightweight quantile-autoregression fallback (foundations QR solver). Gate: match the authors' code. |
| Nonexchangeable / weighted conformal prediction | Conformal with fixed decaying weights on past residuals and explicit coverage-gap bounds in terms of total-variation drift — principled recency weighting for slowly drifting processes. | High | Barber, Candès, Ramdas & Tibshirani (2023); Tibshirani et al. (2019) covariate-shift weighting. Implementation is weighted quantiles of conformity scores — easy; the value is exposing the coverage-gap diagnostic. Gate: match the paper's simulation designs. |
| Conformal PID control | Frames online interval calibration as a control problem (P/I/D terms on coverage error plus a "scorecaster" forecasting the conformity-score quantile) — the state of the art for volatile streams. | High | Angelopoulos, Candès & Tibshirani (2023). The integrator term guarantees long-run coverage for ANY score sequence; the scorecaster (a simple AR or Theta model on scores — this library has those) adds anticipation. Gate: match their released code on the paper's M4/electricity examples. |

**Conditional and scenario forecasting**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Structural scenario analysis (choice of driving shocks) | Conditional forecasts where the user specifies WHICH structural shocks implement the conditioning path — "rates high because of demand" vs "because of policy" — plus scenario-plausibility metrics. | High | Antolín-Díaz, Petrella & Rubio-Ramírez (2021). Requires an identified SVAR (identification module); conditioning restricted to a shock subset follows from Gaussian conditioning on a linear map of shocks; KL divergence between scenario and unconditional forecast measures plausibility. Gate: match their replication files. No open library implements this. |

**Hierarchical and temporal reconciliation**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Probabilistic forecast reconciliation | Reconciling full predictive distributions (not just means) across a hierarchy via score-optimal projection of sample paths or Bayesian conditioning — needed once users want coherent intervals. | High | Panagiotelis, Gamakumara, Athanasopoulos & Hyndman (2023) probabilistic coherence and projections of sample paths (reconcile draws through G, preserving dependence via Schaake-shuffle-style reordering); Corani et al. (2021) Bayesian variant. Cheap once MinT's G exists; evaluate with energy/variogram scores. Gate: match the papers' tourism/electricity applications. |

## Frontier watchlist

Frontier themes from the research already housed in lower tiers rather than Tier 4 rows:

- CORP reliability diagrams and PAV-based score decompositions (Dimitriadis, Gneiting & Jordan 2021) — shipped inside the Tier 2 event-probability suite.
- Murphy diagrams and elicitability-based dominance analysis (Ehm, Gneiting, Jordan & Krüger 2016) — Tier 3, point accuracy measures.
- Fair finite-ensemble CRPS (Ferro 2014) and mixture-of-parameters log-score estimation for MCMC densities (Krüger et al. 2021) — small details that change published rankings, folded into the Tier 1 CRPS/log-score items.
- Adaptive conformal inference and AgACI (Gibbs & Candès 2021; Zaffran et al. 2022) and EnbPI (Xu & Xie 2021) — Tier 3 conformal items; SPCI, nonexchangeable conformal, and conformal PID hold the Tier 4 rows.
- Egalitarian LASSO shrinkage combinations (Diebold & Shin 2019) — Tier 3, point forecast combination.
- Online expert aggregation with regret guarantees (ML-Poly/BOA per Gaillard-Goude-Wintenberger, the opera package) integrated with econometric forecast streams — Tier 3, point forecast combination.
- Cross-temporal reconciliation (Di Fonzo & Girolimetto 2023) — Tier 3, reconciliation; the probabilistic variant holds the Tier 4 row.

## Implementation warnings

- **DM-test long-run variance.** The textbook rectangular kernel truncated at h−1 lags frequently yields negative variance estimates in small samples — detect and fall back to Bartlett/HLN with a warning; always apply the HLN small-sample correction and t(T−1) critical values by default.
- **Nested models under recursive schemes.** Never let users run a plain DM test there — the statistic is degenerate (loss differential → 0 under the null). Route to Clark-West/Clark-McCracken, and encode the scheme-to-test validity map (recursive → West/CW; fixed rolling window → Giacomini-White) in the API, not just the docs.
- **Multi-step regression-based tests.** Direct h-step errors are MA(h−1) by construction: Mincer-Zarnowitz, encompassing, and CG regressions need HAC with bandwidth ≥ h−1, and HAC-Wald tests over-reject badly in small samples — bootstrap p-values as the default for h > 4.
- **Leakage is the #1 silent backtest bug.** Scalers, seasonal adjustment, transformation choice, hyperparameter tuning, recalibration maps, and combination-weight estimation must ALL be fit inside the training window at each origin — design the API so full-sample preprocessing is impossible, not merely discouraged.
- **SPA/MCS bootstrap details.** Results are wrong unless (i) losses are re-centered exactly per Hansen (2005) (the sqrt(2 log log T) threshold), (ii) the stationary/moving-block bootstrap resamples entire loss-differential vectors across models (and horizons) to preserve cross-sectional dependence, and (iii) block length follows Politis-White (2004) with the Patton-Politis-White (2009) correction.
- **Proper-scoring traps.** An indicator-weighted log score is NOT proper (use the Diks-Panchenko-van Dijk censored/conditional likelihood); linear pools of calibrated densities are systematically overdispersed (Gneiting-Ranjan 2013); pick one score orientation (lower = better) library-wide and never mix.
- **Draw-based score estimation.** KDE-based log scores from MCMC output are severely biased in the tails (use the mixture-of-parameters estimator); naive O(m²) CRPS double sums are slow and biased at small m (use sort-based O(m log m) forms plus the fair correction); guard log(0) when realizations fall outside draw support.
- **PIT-based tests.** Parameter estimation error distorts test sizes (use Rossi-Sekhposyan or at least say so); PITs of multi-step forecasts are serially dependent under the null, so independence components (Berkowitz AR term, Ljung-Box on PITs) must be dropped for h > 1; discrete/mixed outcomes require randomized PITs; clip inverse-normal transforms of PITs at 0/1.
- **Percentage and scaled errors.** MAPE/sMAPE division-by-zero must raise or NaN loudly — never silently return inf that averages away; the MASE/RMSSE denominator uses the training-sample naive error with the correct seasonal period, and constant training windows give zero denominators.
- **Combination weights.** Full Bates-Granger covariance weights and unconstrained Granger-Ramanathan OLS explode under near-collinear forecasts — default to diagonal/constrained (simplex) versions with shrinkage toward 1/N, and estimate weights on a rolling basis inside the backtest (weight estimation is itself a model choice subject to snooping).
- **Optimal pools and tilting.** Log-score weight optimization needs log-sum-exp stabilization when component densities differ by orders of magnitude; entropic-tilting Newton solvers overflow unless moment functions are centered — always monitor the effective sample size of tilted/pooled weights and warn on collapse.
- **Conditional forecasting.** Waggoner-Zha constraint systems become ill-conditioned for long conditioning horizons — solve via QR or the simulation smoother (Bańbura-Giannone-Lenza), never normal equations; report implied structural-shock magnitudes so users see when a scenario is statistically absurd.
- **MinT reconciliation.** Never invert the residual covariance directly — Schäfer-Strimmer shrinkage is mandatory for wide hierarchies and all solves should use sparse SPD Cholesky on S'W⁻¹S; watch for negative reconciled values on nonnegative series.
- **Conformal for time series.** Exchangeability fails, so plain split conformal carries NO finite-sample guarantee — default to adaptive variants for drifting data, guard ACI's alpha_t leaving (0,1) (infinite intervals), and surface empirical rolling coverage alongside the intervals.
- **Quantile crossing.** Quantile sets from separate regressions or combinations frequently cross — apply monotone rearrangement (Chernozhukov et al. 2010) before computing any interval or pinball score, or probabilistic evaluation is silently corrupted.
- **Back-transformation bias.** exp of a log-space mean forecast is the median in levels; cumulating differenced forecasts requires cumulating forecast-error covariances — make transformations part of the model spec so the library, not the user, handles Jacobians and bias corrections.
- **Reproducibility of bootstrap/simulation p-values** (SPA, MCS, CM bootstrap, RS tests): require explicit seeds, use counter-based parallel RNG so results are identical across thread counts, and embed seed + scheme metadata in every evaluation report.
- **Evaluation-sample bookkeeping.** Off-by-one errors in aligning (origin, horizon, target date) are endemic — a typed index that makes "forecast made at t for t+h" unambiguous, with explicit handling of ragged edges and missing actuals, prevents an entire class of wrong published numbers.
- **Real-time evaluation.** Model rankings can reverse depending on whether first-release, second-release, or latest-vintage data serve as actuals (Croushore-Stark 2001) — force the user to state the actuals policy explicitly rather than defaulting silently to latest data.

## Dependencies and shared infrastructure

**Consumed from foundations:**

- **Resampling/bootstrap engine** — stationary/moving-block/wild bootstrap with Politis-White block-length selection (Patton-Politis-White correction) and parallel RNG substreams; powers DM/SPA/MCS/StepM/Quaedvlieg bootstraps, bootstrap prediction intervals, and block-bootstrap calibration bands.
- **HAC/long-run-variance engine (one library-wide default policy)** — every loss-differential test, regression-based rationality test, and Knüppel covariance.
- **Innovation-distribution zoo** — analytic forecast distributions and their closed-form CRPS/log-score/quantile hooks.
- **Linear-Gaussian state-space engine (simulation smoothers, FFBS)** — Bańbura-Giannone-Lenza conditional forecasting, DMA updating, BPS posterior simulation.
- **Critical-value engine (response surfaces + cached null simulation)** — McCracken MSE-F/ENC-NEW tables, Giacomini-Rossi fluctuation bands, Rossi-Sekhposyan sup-Wald and density-test critical values.
- **Real-time vintage data store** — the observation-by-publication triangle behind vintage-aware backtesting; this module owns the actuals-policy evaluation layer on top.
- **Fast quantile-regression solver + monotone rearrangement** — quantile forecasts, SPCI fallback, crossing repair.
- **Philox-based reproducible parallel RNG** — bit-reproducible parallel-origin backtests and bootstrap p-values.
- **Exogenous-regressor (covariate) contract** — this module is the contract's enforcement point: the backtesting engine runs its leakage checks so future covariate values can never silently enter a pseudo-out-of-sample exercise, and covariate-dependent forecasts declare at each origin whether future regressor values are known-future, scenario-supplied, or auxiliary-forecast (each implying a different honest evaluation design).
- **Time-index/calendar/frequency engine** — (origin, horizon, target_date) and (event, announcement date) bookkeeping.

**Consumed from other modules:**

- **ML module: time-series cross-validation splitters** — blocked, hv-block (Racine 2000), and K-fold CV for dependent data (Bergmeir, Hyndman & Koo 2018 nuance included) are owned by ML per the master map; this module consumes the fold-construction primitives so its rolling-origin tsCV and ML's tuning CV share one implementation. Also: penalized-regression solvers (egalitarian LASSO, CSR) and the gradient-boosting learner behind FFORMA.
- **bayesian module** — marginal and predictive likelihoods for BMA weighting; MCMC samplers underlying BPS.
- **identification module** — identified SVARs for structural scenario analysis.
- **regime-switching/nonlinear module** — regime processes behind regime-switching combination weights.
- **diagnostics module** — structural-break detection output feeding the break-aware forecast design hooks; STL/MSTL machinery behind Naive2-style seasonal adjustment where shared.
- **nowcasting module** — release calendars complementing vintage evaluation for point-in-time information sets.

**Exposed to other modules:**

- **The unified forecast object** (point/interval/density/path) — the type every model module returns; mandated library-wide.
- **The backtesting engine and evaluation objects with scheme metadata** — consumed by every model module's "evaluate" verb; the scheme-to-test validity map travels with the object.
- **All forecast-comparison tests** — volatility re-exports DM/GW/MCS/SPA for loss-based volatility model comparison.
- **All density-forecast evaluation** (PITs, scoring rules, calibration tests) — consumed by bayesian and volatility.
- **Forecast combination and density pooling** — consumed by nowcasting and multivariate model suites.
- **The single model-agnostic conformal-prediction implementation** — univariate and ML modules point here rather than reimplementing.
- **Reconciliation (MinT and relatives)** — exposed for hierarchical applications across modules.
- **The golden-value validation harness integration** — the M4 reproduction script doubles as a cross-module integration test (naive/Theta/ETS/backtesting/scoring at once).

## Validation gallery

- **scoringRules (Jordan, Krüger & Lerch 2019, JSS)** — closed-form CRPS and log scores for all supported distributions, and sample-based estimators, must match to machine precision.
- **M4 competition published results (Makridakis, Spiliotis & Assimakopoulos 2020)** — Naive2 sMAPE 13.564, Theta sMAPE 12.309, winning OWA 0.821, FFORMA OWA 0.838; CI asserts to 3 decimals.
- **arch (Kevin Sheppard) SPA/MCS + Hansen, Lunde & Nason (2011)** — SPA p-values (consistent/lower/upper) and MCS membership on identical loss matrices must match arch and the HLN empirical example.
- **forecast::dm.test and multDM** — DM/HLN statistics and multivariate DM on identical loss series.
- **Marcellino, Stock & Watson (2006)** — direct-vs-iterated empirical tables on the 170-series US dataset.
- **Welch-Goyal / Clark-West replications** — CW statistics on the equity-premium dataset match published values and Todd Clark's code output.
- **McCracken (2007) critical-value tables and Clark-McCracken bootstrap output** — MSE-F/ENC-NEW table interpolation and restricted-bootstrap p-values.
- **Barbara Rossi's Matlab toolboxes** — Giacomini-Rossi fluctuation test (ECB exchange-rate example) and Rossi-Sekhposyan rationality and density tests against posted replication files.
- **Pesaran, Pick & Pranovich (2013) simulation designs** — MSFE rankings of window selection, robust weighting, and post-break shrinkage reproduce the paper's Monte Carlo results.
- **Croushore-Stark (2001) replication files / Philadelphia Fed real-time dataset** — vintage-aware evaluation reproduces documented ranking reversals across actuals policies.
- **Geweke & Amisano (2011) S&P 500 pools and Krüger-Clark-Ravazzolo (2017) tilting output** — optimal pool weights and entropically tilted moments match replication output.
- **opera (Gaillard & Goude)** — EWA/ML-Poly weights and cumulative regret trajectories match the R package.
- **Wickramasuriya, Athanasopoulos & Hyndman (2019) / fabletools / Nixtla hierarchicalforecast** — MinT reconciled forecasts on the Australian tourism dataset; FoReco for cross-temporal cases.
- **BEAR toolbox (ECB) conditional forecasts** — Waggoner-Zha and BGL conditional distributions match on the WZ example.
- **Conformal authors' code (Gibbs-Candès ACI; Xu-Xie EnbPI/SPCI; Angelopoulos et al. PID)** — coverage and width paths on the papers' examples.
- **Quaedvlieg (2021) replication code (JBES)** — uniform/average multi-horizon SPA statistics and rejection decisions.
