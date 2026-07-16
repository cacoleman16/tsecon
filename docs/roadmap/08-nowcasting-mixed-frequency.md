# Module 08 — Nowcasting and Mixed Frequency

> Part of the time series econometrics library roadmap. Master plan: [ROADMAP.md](../../ROADMAP.md).

**This module is the library's headline differentiator: a complete, fast, statistically honest nowcasting stack — bridge equations, the full MIDAS family, mixed-frequency VARs, and the dynamic-factor nowcasting facade — built on a single mixed-frequency data model with first-class release calendars, vintage awareness, news decomposition, and a leakage-proof pseudo-real-time evaluation harness. No maintained end-to-end nowcasting stack exists in Python today; every central bank and forecasting shop rebuilds this plumbing from scratch. Shipping it, validated against the NY Fed model, Schorfheide-Song, and the R reference packages, and an order of magnitude faster, is the single most visible thing this library can do.**

## Purpose and scope

This module covers the estimation and real-time operation of models that mix sampling frequencies: bridge equations that complete the current quarter with auxiliary indicator forecasts; the MIDAS regression family in all its weighting schemes (Almon, exponential Almon, beta, unrestricted, ADL, leads, regime-switching, Bayesian, quantile, penalized); mixed-frequency VARs in both stacked (Ghysels) and state-space (Schorfheide-Song) form; and the nowcasting facade over the library's dynamic factor model — block loading structures, ragged-edge handling, news decomposition, and release-calendar-driven updating. It also owns the user-facing temporal disaggregation API (Chow-Lin, Denton, and their state-space generalizations) and the module-level layer that turns raw data vintages into honest backtests.

The users are central-bank and ministry nowcasting teams, market economists tracking GDP in real time, and researchers doing mixed-frequency econometrics. Their defining problems are not exotic estimators but infrastructure: which series was known on which date, at which vintage, with what publication lag; how a quarterly target loads on latent monthly states; and why this morning's release moved the nowcast. The module treats these as first-class objects — the release calendar, the vintage-aware information set, the news decomposition — rather than leaving them as user-side scripting.

Relative to the rest of the library, this module is a heavy consumer and a demanding customer. The Kalman/state-space machinery (missing data, exact diffuse initialization, simulation smoothing, EM, collapsing) lives in foundations, and this module is the primary driver of its requirements; the DFM estimation core lives in the multivariate module and is wrapped here; MF-BVAR samplers come from the Bayesian module; penalized solvers and temporal cross-validation come from ML; forecast-comparison and density-scoring machinery comes from forecasting-evaluation. What this module owns outright is the MIDAS weighting machinery (which volatility's GARCH-MIDAS consumes), MF-VAR model classes, news decomposition, the release-calendar layer, and the pseudo-real-time evaluation harness.

## Where existing tools fall short

- **statsmodels `DynamicFactorMQ`** handles only the monthly+quarterly pair — no weekly/daily data and no calendar-aware aggregation, so ADS/WEI-class models are impossible; its EM is Python-loop slow on large panels; it has `news()` but no vintage store, no release calendar, no pseudo-real-time harness, and no stochastic-volatility, Markov-switching, or outlier-robust variants.
- **R `midasr`** is frequentist-only, has fragile NLS with no built-in multistart, an API built around a cryptic `mls()` embedding that trips new users, no quantile MIDAS, no Bayesian or penalized MIDAS, and no connection between lead specification and release calendars.
- **R `nowcasting` / `nowcastDFM`** hard-code one NY Fed-style architecture, are slow (pure-R EM), have been intermittently archived from CRAN, and offer no Bayesian options and limited news-decomposition flexibility.
- **Bayesian MF-VAR tooling** is essentially R `mfbvar` (limited priors, monthly+quarterly only, development stalled) plus authors' Matlab code (Schorfheide-Song, Brave-Butters-Justiniano). There is no maintained Python implementation of Schorfheide-Song at all.
- **No package unifies bridge equations, MIDAS, MF-VAR, and DFM behind one data model** — every model family carries its own ad hoc alignment code, so the cross-model comparison practitioners actually want requires bespoke research code.
- **Real-time vintage management is nobody's feature**: ALFRED/ECB-RTDB fetching, vintage-stamped storage, release-calendar simulation, and leakage-safe recursive evaluation are reimplemented from scratch in every central-bank shop.
- **News decomposition exists only for Gaussian DFMs** (statsmodels, NY Fed code); no tool provides news decomposition for MF-BVARs, cleanly separates data revisions from new releases, or attributes revisions for ML nowcasts.
- **Temporal disaggregation in Python is essentially absent**: `tempdisagg` is R; Python ports are unmaintained toys. Denton-Cholette, Chow-Lin with boundary handling, and multivariate reconciliation have no serious Python home.
- **GARCH-MIDAS exists only as R `mfGARCH`**; Python `arch` has nothing; quantile MIDAS and Bayesian MIDAS exist only as paper replication code.
- **The NY Fed Matlab replication code — the de facto benchmark — is archived, dataset-hard-coded, and slow.** "NY Fed model, general data, 10-100x faster, with tests" is an achievable and highly visible bar to clear.
- **COVID-era outlier handling is ad hoc everywhere**: no library offers t-errors, outlier states, or Lenza-Primiceri-style volatility scaling as first-class estimation options for nowcasting models.
- **Mixed-frequency Granger causality tests** (Ghysels-Hill-Motegi) and stacked MF-VAR structural analysis have no maintained implementation in any language.
- **Evaluation practice is weak everywhere**: nothing wires Clark-West or fluctuation tests to a vintage store, nothing tracks accuracy as a function of days-to-release, and nothing forces the first-release vs latest-vintage "actuals" choice explicitly.

## Inventory

Priorities map to tiers: core → Tier 1, standard → Tier 2, advanced → Tier 3, frontier → Tier 4. State-space infrastructure items and the classical temporal disaggregation engines are reassigned by the ownership map and appear under [Dependencies and shared infrastructure](#dependencies-and-shared-infrastructure).

### Tier 1 — Core (v1-blocking)

**Real-time infrastructure**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Release calendar and vintage integration layer | First-class release calendars (which series is published when, with what publication lag) wired to the foundations vintage store, so a nowcast date deterministically implies an information set. Every nowcasting exercise lives or dies on this plumbing. | medium | Croushore & Stark (2001, JoE) for the real-time dataset concept. Series metadata must carry frequency, stock/flow type, transformation, SA status, and publication-lag profile; ALFRED and ECB real-time DB connectors as optional I/O. No open library integrates this with models — the single biggest gap in the field. Validate calendar simulation against the NY Fed nowcast's stylized release ordering. |
| Pseudo real-time (recursive vintage) evaluation harness | Automated backtesting: for each hypothetical nowcast date, reconstruct the exact information set (true vintages or simulated ragged edges from publication lags), re-estimate or update models, and record nowcasts against declared "actuals". Built jointly with the forecasting-evaluation module. | medium | Design pattern in Giannone-Reichlin-Small (2008) and Banbura et al. (2013, Handbook). Force the user to declare which release is truth (first, second, latest) — rankings change with this choice. Guard against leakage: all standardization, factor extraction, tuning, and selection must happen inside each vintage loop. Provide within-quarter nowcast trajectories (RMSE by days-to-release). Validate by reproducing the RMSE-vs-horizon curves in Giannone-Reichlin-Small (2008) and Banbura-Modugno (2014). |

**Temporal aggregation**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Stock/flow aggregation mapping with Mariano-Murasawa triangle weights | The measurement-equation link between latent high-frequency series and observed low-frequency ones: averages/sums for flows, point-in-time sampling for stocks, and the (1,2,3,2,1)/3 triangle for quarterly growth rates of monthly log-differences. | medium | Mariano & Murasawa (2003, JAE). Triangle weights are exact for arithmetic aggregation of levels but an approximation for log-differences (geometric vs arithmetic mean) — document loudly and offer exact level-space aggregation. API must require declaring aggregation type per series; silently applying flow weights to a stock (e.g., the unemployment rate) is the classic silent error. Support general frequency ratios (m = 3, m = 12, irregular weekly). Validate measurement rows against statsmodels `DynamicFactorMQ` internals and the published Mariano-Murasawa coincident index. |

**Bridge models**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Bridge equations | Quarterly regressions of GDP growth on quarterly aggregates of monthly indicators, with missing months of the current quarter filled by auxiliary AR/RW forecasts. The oldest and still ubiquitous institutional nowcasting tool. | low | Baffigi, Golinelli & Parigi (2004, IJF); Schumacher (2016) for bridge-vs-MIDAS. Estimation is easy; the value is the surrounding machinery: automatic indicator-completion models, release-calendar-aware aggregation, and pooling across many single-indicator bridges (Kuzin, Marcellino & Schumacher 2013 show pooling beats selection). Validate against published euro-area bridge RMSEs and ECB bridge-suite conventions. |

**MIDAS family**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| MIDAS with Almon (PDL) polynomial weights | Low-frequency variable regressed on many high-frequency lags with weights restricted to a low-order polynomial; linear in parameters, estimable by OLS. The pedagogical and practical entry point. | low | Ghysels, Santa-Clara & Valkanov (2004); Almon (1965). Implement as a design-matrix transform with endpoint (tail) restriction support. The frequency-alignment embedding (`mls` in midasr) is where bugs live: off-by-one in high-frequency lag indexing. Validate coefficient-for-coefficient against R midasr (Ghysels, Kvedaras & Zemlys 2016, JSS). |
| Exponential Almon MIDAS (NLS) | Weights proportional to exp(θ₁k + θ₂k²): flexible declining patterns with two parameters. The signature Ghysels specification. | medium | Ghysels, Santa-Clara & Valkanov (2005, JFE; 2006, JoE). Compute weights in log-space and normalize to sum to one (naive exp overflows for daily lags); impose θ₂ ≤ 0 for declining/identified shapes; the NLS objective is multimodal — mandatory multistart plus profiling out the linear slope. Document the normalization convention (weights sum to 1, separate slope β) — Matlab toolbox and midasr conventions differ. Validate against midasr with matched convention. |
| Beta-lag MIDAS | Weights from the normalized beta density over lag fractions; hump-shaped or monotone with two or three parameters. Popular for daily financial data. | medium | Ghysels, Sinko & Valkanov (2007). Evaluate the beta kernel on a (k+ε)/(K+1) grid to avoid 0^negative at endpoints; enforce θ > 0; near-flat gradient regions stall NLS — supply analytic gradients and multistart. Third "non-zero last lag" parameter as an option. Validate against midasr's `nbeta`/`nbetaMT`. |
| U-MIDAS (unrestricted) | OLS on individual high-frequency lags without a weight function; dominates restricted MIDAS when the frequency mismatch is small (monthly-quarterly). | low | Foroni, Marcellino & Schumacher (2015, JRSS-A). Trivial estimation; the library's job is guidance — warn/refuse when the frequency ratio m is large (daily-quarterly means parameter explosion) and route users to restricted MIDAS or sg-LASSO. Built-in AIC/BIC lag selection. Validate against the FMS (2015) simulation table patterns and midasr. |
| ADL-MIDAS | MIDAS augmented with autoregressive low-frequency dynamics — the practically relevant version; plain MIDAS without AR terms is a straw man for macro. | medium | Andreou, Ghysels & Kourtellos (2013, JBES); Clements & Galvao (2008, JBES) for the common-factor way of adding AR terms without spurious seasonal patterns. Implement both the CG common-factor AR-MIDAS and the direct ADL-MIDAS, and document the difference — it trips up practitioners constantly. Validate against Clements-Galvao published US output-growth results. |
| MIDAS with leads | Uses high-frequency observations from inside the reference low-frequency period — exactly the mid-quarter nowcasting situation. | medium | Andreou, Ghysels & Kourtellos (2013); Clements & Galvao (2009). The lead count must be driven by the release calendar, not naive index arithmetic — off-by-one lead errors are the most common MIDAS bug in applied work. Tie lead specification to the vintage/calendar layer: users specify a nowcast date, the library derives available leads per indicator. Validate against AGK (2013) daily-financial-data results. |

**Mixed-frequency VAR**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Bayesian MF-VAR (Schorfheide-Song) | The canonical Bayesian mixed-frequency VAR: Minnesota-prior VAR at monthly frequency, quarterly observables via aggregation states, Gibbs sampling alternating simulation smoother and standard BVAR posterior draws. The headline model for central-bank users. | high | Schorfheide & Song (2015, JBES). Blocked Gibbs: states given parameters via the simulation smoother (reduced-rank innovation covariance — see the foundations requirements below), parameters given states via the Normal-inverse-Wishart Minnesota posterior (samplers from the Bayesian module). Compute the marginal data density for prior tuning. Companion: Schorfheide & Song (2024, IJCB) pandemic update for what breaks with COVID data. Validate against the authors' Matlab replication and R mfbvar (Ankargren & Yang) on the same dataset — target 10x+ speed via univariate filtering and precision samplers. |

**Dynamic factor nowcasting**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Mixed-frequency DFM with block structure (NY Fed architecture) | The flagship nowcasting model: monthly+quarterly panel, global/soft/real/labor factor blocks with restricted loadings, AR(1) idiosyncratics, EM estimation under arbitrary missingness, Kalman-based nowcast updates. The single model most users will run. This module owns the facade; the DFM estimation core is consumed from the multivariate module. | high | Bok, Caratelli, Giannone, Sbordone & Tambalotti (2018, Annual Review of Economics) and the NY Fed Nowcasting_Code; statsmodels `DynamicFactorMQ` (Fulton) is the Python benchmark. Block restrictions enter the EM M-step as linear restrictions on loadings (Banbura-Modugno appendix). AR(1) idiosyncratics: augment states rather than quasi-difference — simpler with missing data. Sign-fix factor normalization for run-to-run reproducibility. Validate: reproduce NY Fed replication-code nowcast paths to 3+ decimals and `DynamicFactorMQ` log-likelihoods; beat both on speed by an order of magnitude. |
| News decomposition of nowcast revisions | Decomposes the change in a nowcast between two vintages into per-release contributions ("news" = release minus model expectation, times a Kalman-gain-based weight). The killer feature for institutional communication: "why did the nowcast move today?" | high | Banbura & Modugno (2014, JAE); Banbura, Giannone, Modugno & Reichlin (2013, Handbook of Economic Forecasting). Exact only when parameters are identical across the two information sets; when re-estimated, report the residual "parameter revision" term separately (statsmodels `news()` does this — match it). Impact weights come from projecting the nowcast on the news vector via smoother output. Revisions to previously published values need treatment separate from newly released values. Validate against statsmodels `DynamicFactorMQ.news()` and the NY Fed's published news-impact bar charts. |

**Evaluation**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Vintage-wired nowcast evaluation suite | Point-nowcast accuracy testing tailored to nowcasting: Diebold-Mariano with correct caveats, Clark-West for nested comparisons, Giacomini-White conditional tests, Giacomini-Rossi fluctuation tests, all wired to the vintage harness. Test statistics are consumed from forecasting-evaluation; this module owns the vintage and days-to-release integration. | medium | Diebold & Mariano (1995); Clark & West (2007, JoE) — most nowcast comparisons are nested (indicator model vs AR benchmark), so CW must be the advertised default; Giacomini & White (2006); Giacomini & Rossi (2010, JAE). Small evaluation samples are the norm — ship fixed-b/HAR corrections (Coroneo & Iacone 2020 for exactly this setting). Value-add is the integration: vintages, actuals choice, days-to-release conditioning. |

### Tier 2 — Standard (expected of a serious library)

**MIDAS family**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Step-weighting and simple schemes | Piecewise-constant (step) weights and other simple parameterizations; useful baselines and what some institutions actually run. | low | Forsberg & Ghysels (2007) for step functions; Corsi (2009) HAR is a step-MIDAS special case (cross-reference the volatility module). Cheap once the alignment machinery exists. |
| Factor-MIDAS | MIDAS regressions on estimated factors from a monthly panel — dimension reduction plus MIDAS aggregation for ragged-edge nowcasting. | medium | Marcellino & Schumacher (2010, OBES). Three variants (basic, smoothed, U-MIDAS on factors); factor estimation must respect the ragged edge (EM or vertical realignment per vintage). Document the two-step generated-regressor caveat for inference. Validate against Marcellino-Schumacher German GDP results. |
| MIDAS nowcast pooling and combination | Systematic combination (equal-weight, BIC-weight, MSE-discounted) of many single-indicator MIDAS/bridge nowcasts — in practice beats most single models and is what institutions run. Combination-weight machinery is consumed from forecasting-evaluation. | low | Kuzin, Marcellino & Schumacher (2013, JAE) for pooling vs selection; Timmermann (2006) for weights; discounted MSFE weights per Andreou-Ghysels-Kourtellos (2013). Ties directly into the evaluation harness. Validate against KMS pooling results patterns. |

**Mixed-frequency VAR**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Stacked (blocking) mixed-frequency VAR | Ghysels' observation-driven MF-VAR: stack the m intra-period high-frequency observations as separate elements of a low-frequency vector; OLS-estimable, no latent states, delivers mixed-frequency IRFs. | medium | Ghysels (2016, JoE). Parameter count explodes with m and lags — provide a Bayesian shrinkage option (Minnesota-style on the stacked system). Intra-period ordering conventions determine IRF interpretation; make ordering explicit in the API and document within-period response conventions. Validate against Ghysels' published bivariate examples. |
| State-space MF-VAR (classical ML) | Parameter-driven MF-VAR: the VAR runs at high frequency, low-frequency observables load on latent states via aggregation constraints; ML via Kalman filter. | high | Mariano & Murasawa (2010, Oxford Bulletin); Foroni & Marcellino (2013 survey). The ML surface is multimodal for medium systems — EM warm starts then quasi-Newton. State dimension grows with m and VAR lags; exploit transition-matrix sparsity. Mostly a stepping stone to the Bayesian version, but needed for frequentist users. |
| Conditional forecasting / scenario nowcasts | Nowcasts conditional on paths or values for subsets of variables ("given oil prices this month, what is GDP?") via Kalman conditioning; also the mechanism that turns any state-space model into a ragged-edge nowcaster. | medium | Banbura, Giannone & Lenza (2015, IJF); Waggoner & Zha (1999) for the distributional version. Implement both hard and soft (distributional) conditioning. Shared across MF-VAR and DFM — design once on the foundations state-space engine. Validate against BGL euro-area scenario exercises. |

**Dynamic factor nowcasting**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Two-step DFM nowcasting (Giannone-Reichlin-Small / Doz-Giannone-Reichlin) | PCA factors on a balanced panel, VAR on factors, then Kalman filtering for the ragged edge — the original "nowcasting" model and still the fastest credible DFM estimator. | medium | Giannone, Reichlin & Small (2008, JME); Doz, Giannone & Reichlin (2011, JoE) for two-step consistency; Doz-Giannone-Reichlin (2012, REStat) for QML. Keep as the fast initializer for EM and for huge panels. Pitfall: standardization and PCA must be done per-vintage in real-time use. Validate against the GRS (2008) US exercise and Doz et al. simulation tables. |

**Big data and ML nowcasting**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Targeted predictors and preselection | Hard/soft-thresholding of a large indicator panel before factor extraction or regression — choose predictors informative for the target, not just common variation. | low | Bai & Ng (2008, JoE). t-stat thresholding plus LARS/lasso preselection; selection must happen inside the pseudo-real-time loop. Also implement Ferrara & Simoni (2023, JBES) pre-testing guidance for when alternative data (Google) actually helps. |
| Penalized-regression nowcasting with temporal CV | Direct high-dimensional regression nowcasts on large mixed-frequency panels with leakage-safe tuning; the sane ML baseline. Solvers and blocked CV consumed from the ML module. | medium | De Mol, Giannone & Reichlin (2008, JoE) — the ridge/lasso vs PCA equivalence results frame the docs; Giannone, Lenza & Primiceri (2021, Econometrica) "illusion of sparsity" shapes defaults (dense/ridge-type priors as default, sparsity as a hypothesis to test). Blocked/rolling-origin CV only; iid K-fold on time series is a documented bug, not an option. Validate against DGR (2008) US panel results. |
| Tree-ensemble nowcasting on vintage panels | RF/GBM nowcasts using ragged-edge features engineered from the vintage store (missingness indicators, days-since-release, factor summaries) — competitive in nonlinear episodes. | medium | Medeiros, Vasconcelos, Veiga & Zilberman (2021, JBES) as the RF-in-macro template. Wrap existing engines (lightgbm/sklearn) rather than reimplement; the library's value-add is vintage-store feature engineering and a protocol object guaranteeing pseudo-real-time honesty (retrain schedule, embargoes). Provide permutation importance mapped back to releases. |

**Temporal disaggregation utilities**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| State-space temporal disaggregation (Proietti) | Casts Chow-Lin-type disaggregation in state-space form, allowing dynamic regressions, logs, ragged edges, and extrapolation beyond the last benchmark — strictly generalizes the classical utilities. | medium | Proietti (2006, Econometrics Journal). Falls out nearly free once the missing-data Kalman filter and cumulator states exist. Cumulator-state construction (Harvey 1989) is the key device; watch the diffuse initialization of the cumulator. Validate: replicate exact Chow-Lin results as a special case, then Proietti's published examples. |

**Documentation-as-feature**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Model-selection and specification guidance layer | Executable guidance: given the user's data shape (frequencies, N, T, ragged-edge profile, target), recommend and benchmark the appropriate model family (bridge vs MIDAS vs U-MIDAS vs DFM vs MF-BVAR) with citations. | medium | Encode the empirical consensus: U-MIDAS beats restricted MIDAS for small m (Foroni-Marcellino-Schumacher 2015); pooling beats selection (Kuzin et al. 2013); DFM vs MIDAS comparisons in Marcellino-Schumacher (2010) and Kuzin et al. (2011, JAE); BVAR matches DFM (Cimadomo et al. 2022). A diagnostic report plus auto-benchmark harness, not magic AutoML. This is documentation the field actually lacks. |

### Tier 3 — Advanced (differentiators)

**MIDAS family**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Smooth-transition and asymmetric MIDAS | Regime-dependent or sign-asymmetric responses to high-frequency indicators (e.g., financial conditions matter more in downturns). | medium | Galvao (2013, IJF) smooth-transition MIDAS; Ghysels, Sinko & Valkanov (2007) asymmetric specifications. NLS with transition parameters is badly behaved — grid the transition slope/location, profile the rest. Validate against Galvao's UK/US results qualitatively. |
| Markov-switching MIDAS (MS-MIDAS) | MIDAS with Markov-switching intercept/slope/variance for regime-aware nowcasting and joint recession-probability output. | high | Guerin & Marcellino (2013, JBES). EM/Hamilton filter with a MIDAS NLS inner loop; label switching and multimodality — fix regimes by mean ordering. Cross-reference the regime-switching module for the filter. Validate against the Guerin-Marcellino US GDP application. |
| Bayesian MIDAS (including penalized variants) | Bayesian MIDAS: priors on weight parameters, stochastic volatility in errors, and shrinkage/selection across many indicators (BMIDAS with group priors). | high | Pettenuzzo, Timmermann & Valkanov (2016, JoE) for BMIDAS with SV; Mogliani & Simoni (2021, JoE) for Bayesian penalized MIDAS with group-lasso-type priors and adaptive spike-and-slab. MCMC over exp-Almon parameters needs careful reparameterization (log-scale, bounded transforms). A genuine differentiator: no open implementation exists in Python or R beyond the authors' replication code. Validate against the Mogliani-Simoni replication files (Banque de France). |
| MIDAS quantile regression (Growth-at-Risk nowcasting) | Quantile regressions with MIDAS-aggregated high-frequency regressors — the tool for nowcasting distributional tail risk with daily/weekly financial conditions. | high | Ghysels, Plazzi & Valkanov (2016, JF); Adams, Adrian, Boyarchenko & Giannone (2021, IJF); cross-reference Adrian-Boyarchenko-Giannone (2019, AER). Check-loss with nonlinear weight parameters is nonsmooth+nonconvex — profile/grid the weight parameters with linear-programming inner QR solves (foundations' solver), or use a smoothed check function. Enforce/report quantile crossing; offer rearrangement (Chernozhukov et al. 2010). Validate against published US GaR term structures. |
| Reverse and RU-MIDAS | Predicting a high-frequency variable using low-frequency ones (e.g., monthly indicators informed by quarterly GDP releases), completing the frequency-direction matrix. | medium | Foroni, Guerin & Marcellino (2018, IJF). Mostly a re-indexing of the alignment machinery; each high-frequency sub-period gets its own equation. Useful for interpolation and high-frequency monitoring conditioned on official releases. |

**Mixed-frequency VAR**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Mixed-frequency Granger causality tests | Causality tests between series at different frequencies without temporal aggregation bias (aggregation famously creates spurious causality). | medium | Ghysels, Hill & Motegi (2016, JoE; 2020, JoE max-test version). Wald tests on stacked MF-VAR coefficients; the max-test variant handles parameter proliferation. Bootstrap critical values needed in realistic samples — leverage foundations' bootstrap engine. Almost no software implements this in any language; genuine differentiator. |
| MF-BVAR with stochastic volatility | Adds SV (and optionally fat tails/outlier states) to the MF-BVAR — essential for realistic density nowcasts and post-2020 data. | high | Carriero, Clark & Marcellino (2015, JRSS-A); Ankargren, Unosson & Yang (2020) as an alternative. KSC (Kim-Shephard-Chib 1998) mixture sampler for SV; watch the corrected triangular sampler issue (Carriero et al. 2022 correction) in equation-by-equation estimation. Validate against CCM (2015) US results and mfbvar's SV mode. |
| Large mixed-frequency BVARs | MF-BVARs with 20-130 variables as a DFM alternative, with conditional-forecast-based nowcast updating; shown to match or beat DFM nowcasts. | high | Cimadomo, Giannone, Lenza, Monti & Sokol (2022, JoE); Brave, Butters & Justiniano (2019, IJF) Chicago Fed MF-BVAR. Key devices: hierarchical prior-tightness selection (Giannone-Lenza-Primiceri 2015), conditional forecasting via Kalman (Banbura-Giannone-Lenza 2015) to absorb the ragged edge. Scaling: equation-by-equation triangularized sampling; dense-matrix memory is the binding constraint. Validate against Brave-Butters-Justiniano published RMSEs (their code is public). |

**Dynamic factor nowcasting**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Markov-switching DFM for recession nowcasting | DFM whose factor mean/variance switches with a latent Markov state — real-time recession probabilities alongside activity nowcasts (Euro-STING / Chauvet tradition). | high | Chauvet (1998, IER); Camacho & Perez-Quiros (2010, JAE) Euro-STING; Camacho, Perez-Quiros & Poncela (2018) survey. Kim (1994) approximate filter for MS state-space with missing data; exact filtering is infeasible — document the approximation. Label switching; initialize from business-cycle dating. Validate against published Chauvet-Piger real-time recession probabilities and the Euro-STING replication. |
| DFM with stochastic volatility and time-varying long-run growth | Drifting mean growth (secular stagnation) plus SV — materially better density nowcasts, avoids systematic bias when trend growth shifts. | high | Antolin-Diaz, Drechsel & Petrella (2017, REStat); Marcellino, Porqueddu & Venditti (2016, JBES) for SV in nowcasting DFMs. Bayesian estimation (Gibbs with KSC SV blocks); the random-walk mean-growth state needs a tight prior or it soaks up cycles. Validate against the ADP (2017) replication code (publicly available). |
| Outlier-robust / pandemic-proof DFM estimation | Options that keep DFM/MF-VAR estimation sane with COVID-scale observations: outlier dummies, Student-t idiosyncratics, explicit outlier states, or volatility regimes. Post-2020 this is effectively mandatory. | high | Ng (2021, NBER); Antolin-Diaz, Drechsel & Petrella (2024, JAE); Lenza & Primiceri (2022, JAE) for the VAR analog. t-errors via scale-mixture augmentation in EM/Gibbs. Default behavior matters: silently Gaussian-fitting 2020Q2 produces garbage factors — detect extreme standardized observations and warn with suggested options. Validate against ADP (2024) results. |
| Weekly/daily activity indexes (ADS / WEI class) | High-frequency latent activity indexes mixing daily/weekly/monthly/quarterly data — the Aruoba-Diebold-Scotti index and Weekly Economic Index architecture. | high | Aruoba, Diebold & Scotti (2009, JBES); Lewis, Mertens, Stock & Trivedi (2022, JAE). The hard part is the calendar: weeks per month/quarter are irregular (52/53-week years), so aggregator states must be built from an actual date calendar, not fixed ratios — design the aggregation layer date-aware from day one. State dimension gets large at daily frequency; univariate filtering essential. Validate against published ADS values (Philadelphia Fed) and WEI vintages (Dallas Fed/NY Fed). |

**Real-time infrastructure**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Data-revision models (news vs noise) and revision-aware nowcasting | Models of the measurement process linking preliminary releases to final values, enabling nowcasts of the "true"/final value and joint modeling of revisions. | high | Jacobs & van Norden (2011, JoE); Kishor & Koenig (2012, JBES) for VAR estimation with data subject to revision; Anesti, Galvao & Miranda-Agrippino (2022, JAE) release-augmented DFM ("Uncertain Kingdom") as the frontier version. State-space with multiple release equations per variable; news-vs-noise identification is fragile — implement the JvN identification diagnostics. Validate against the UK ONS revision triangles used in Anesti et al. |

**Temporal disaggregation utilities**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Multivariate temporal disaggregation with contemporaneous constraints | Simultaneous disaggregation of several series under cross-sectional accounting identities (components sum to aggregate) — national-accounts-consistent monthly GDP components. | high | Di Fonzo & Marini (2011); Rossi (1982). QP/GLS formulation with sparse constraint matrices. Niche but valued by central-bank users; no Python implementation exists. Validate against tempdisagg's multivariate examples and Eurostat case studies. |

### Tier 4 — Frontier (research-grade, each gated on a named validation target)

**MIDAS family**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| sg-LASSO MIDAS / machine-learning MIDAS | High-dimensional MIDAS: Legendre-polynomial weight bases with sparse-group LASSO across indicators (groups) and lag polynomials (within-group) — MIDAS with hundreds of high-frequency predictors. | high | Babii, Ghysels & Striaukas (2022, JBES); companion HAC-based inference for sg-LASSO (Babii-Ghysels-Striaukas 2023). Proximal-gradient solver with group structure (ML module); tuning by temporally-blocked CV only. Legendre basis conditioning is much better than raw Almon for long lags. **Gate:** match R midasml (Striaukas) and the paper's US GDP nowcasting exercise. |

**Mixed-frequency VAR**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Nonparametric / BART mixed-frequency VAR | MF-VAR with Bayesian Additive Regression Trees replacing linear conditional means — state-of-the-art density nowcasting in turbulent periods. | research-grade | Huber, Koop, Onorante, Pfarrhofer & Schreiner (2023, JoE). Gibbs alternates BART fits (Chipman-George-McCulloch backfitting) with state simulation; expensive — the speed pillar matters here. Filtering with nonparametric conditional means requires the paper's mixed conditional-posterior tricks. **Gate:** reproduce the paper's published predictive-score rankings for euro-area pandemic nowcasts (exact MCMC replication is infeasible). |

**Dynamic factor nowcasting**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Sparse dynamic factor models | Sparsity-inducing penalties/priors on loadings for interpretability (which series drive which factor); recent EM-with-lasso implementations make this practical. Extends the multivariate module's DFM core. | high | Mosley, Gibberd & Eckley (2023/2024, sparseDFM); Kaufmann & Schumacher (2019, JoE) for Bayesian sparse-loading identification. EM with adaptive-lasso M-step; sparsity is only meaningful up to rotation — enforce identification first. **Gate:** match sparseDFM package outputs. |
| Quantile factor models and distributional nowcasting | Factor structures in conditional quantiles and nowcasting entire predictive distributions of GDP — the frontier of "nowcasting at risk". | research-grade | Chen, Dolado & Gonzalo (2021, Econometrica); Adams-Adrian-Boyarchenko-Giannone (2021, IJF); Carriero, Clark & Marcellino (2024) on nowcasting tail risk. Iterative quantile-PCA estimation; inference is delicate. Position as research-grade with clear caveats. **Gate:** reproduce published US GaR/tail-risk term structures. |

**Big data and ML nowcasting**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Neural-network nowcasting (LSTM-class) | Sequence models trained on vintage panels; shown competitive at UNCTAD and several central banks. Optional extension module, not core. | medium | Hopp (2022, UNCTAD; nowcast_lstm package). Danger zone: tiny effective macro samples make deep nets fragile — docs must say so; ensembling across seeds mandatory. Delegate training to torch behind an optional dependency. **Gate:** match nowcast_lstm's published comparisons. |
| Text and alternative-data indicators | Ingestion utilities for news-sentiment indices, Google Trends, mobility, and payments data as high-frequency indicators feeding any model in the library. | medium | Ashwin, Kalamara & Saiz (2024, JAE); Ferrara & Simoni (2023); Barbaglia, Consoli & Manzan (2024). The library provides indicator-side plumbing (frequency conversion, vintage-stamping of scraped data), not NLP. Emphasize vintage-stamping: alt-data series are silently revised, which invalidates naive backtests. **Gate:** reproduce the Ashwin-Kalamara-Saiz euro-area text-indicator exercise pattern. |
| Shapley-based news attribution for ML nowcasts | The Kalman-news analog for non-state-space models: attribute nowcast revisions between vintages to individual releases via Shapley values on the information-set difference. | research-grade | Emerging practice (ECB/BIS working papers 2021-2024 on interpretable ML nowcasting). Exact Shapley over release subsets is exponential — sampling-based approximation with the release calendar defining the coalition structure. No package anywhere does this; unifies "news" communication across model classes. **Gate:** exact agreement with Kalman news decomposition when applied to a linear-Gaussian model. |

## Frontier watchlist

Frontier items from the research sweep not carried as Tier 4 rows, kept on watch:

- Release-augmented DFM jointly nowcasting GDP and its future revisions (Anesti-Galvao-Miranda-Agrippino 2022 JAE "Uncertain Kingdom") — the upgrade path for the Tier 3 data-revision models.
- Calendar-exact weekly state-level activity indices (Baumeister-Leiva-Leon-Sims 2024) with proper 52/53-week handling — extension of the Tier 3 ADS/WEI item.
- Bayesian penalized MIDAS with group spike-and-slab priors (Mogliani-Simoni 2021 JoE) — the gating reference for the Tier 3 Bayesian MIDAS item.
- Mixed-frequency Granger causality max-tests with bootstrap inference (Ghysels-Hill-Motegi 2016, 2020) — the max-test extension of the Tier 3 test suite.
- Illusion-of-sparsity diagnostics: spike-and-slab posterior over sparse vs dense predictive models as a built-in check (Giannone-Lenza-Primiceri 2021 Econometrica).
- Pandemic-mode MF-VAR reweighting and time-varying-volatility fixes (Schorfheide-Song 2024 IJCB update; Lenza-Primiceri 2022 JAE).
- Outlier-robust nowcasting DFMs with heterogeneous dynamics and secular trends (Antolin-Diaz-Drechsel-Petrella 2024 JAE; Ng 2021) — tracked as the reference bar for the Tier 3 outlier-robust item.

## Implementation warnings

The "easy to get statistically or numerically wrong" list. All of these are load-bearing.

1. **EM for DFMs:** the smoothed lag-one cross-covariance E[f_t f_{t-1}'] must come from the smoother's dedicated recursion (Watson-Engle 1983), not from products of smoothed means — getting it wrong biases loadings silently while the model still "runs". Assert monotone log-likelihood every EM iteration as a built-in invariant.
2. **Missing-data likelihoods:** the Gaussian constant must use the number of series actually observed at each t; a fixed-N constant silently corrupts every likelihood-based comparison and information criterion across specifications with different missingness.
3. **Never zero-fill missing observations** "with large measurement variance" as the default mechanism; use row selection or univariate filtering. Big-kappa diffuse initialization (1e7) is similarly toxic — use exact diffuse initialization and solve stationary blocks via a Schur-based Lyapunov solver.
4. **Reduced-rank state innovations:** mixed-frequency state vectors contain zero-innovation aggregator/lag states, so the state innovation covariance is reduced-rank. Simulation smoothers and any code that inverts Q will fail or must use the R·Q·R' selection-matrix formulation. Adding tiny jitter noise changes posteriors — avoid it.
5. **Mariano-Murasawa triangle aggregation** of log-differences is an approximation (arithmetic vs geometric aggregation); document it and offer exact level aggregation. The API must force the user to declare stock vs flow per series — applying flow weights to a stock variable is the most common silent nowcasting bug.
6. **Exponential-Almon and beta MIDAS weights:** compute in log-space and normalize (naive exp overflows with daily lags); the NLS objective is multimodal — multistart plus profiling out linear parameters is mandatory, not optional. Weight-normalization conventions differ across midasr, Matlab toolboxes, and papers — pick one, document it, and match conventions in validation tests.
7. **Real-time leakage is the cardinal sin:** standardization, PCA/factor extraction, hyperparameter tuning, variable selection, and ML training must all occur inside each vintage of the pseudo-real-time loop. Full-sample standardization before recursive evaluation is the single most common bug in published nowcasting comparisons.
8. **MIDAS leads and ragged-edge alignment** must be driven by an explicit release calendar; index-arithmetic "lead = 2" code produces off-by-one information sets that flatter backtests.
9. **News decomposition is exact only when parameters are held fixed** across the two vintages; when models are re-estimated, report a separate "parameter revision" remainder (as statsmodels does), and treat revisions to previously published data separately from genuinely new releases.
10. **Nested out-of-sample comparisons** (indicator model vs AR benchmark — i.e., almost all nowcast evaluations) invalidate standard Diebold-Mariano; default to Clark-West and provide small-sample/fixed-b corrections. Nowcast evaluation windows are short, so also ship fluctuation tests rather than only full-sample averages.
11. **Chow-Lin rho estimation routinely hits the unit boundary** — detect, warn, and fall back to Fernandez; disaggregated output must satisfy aggregation constraints exactly by construction, not by post-hoc rescaling. Denton must be the Denton-Cholette proportional-first-difference variant by default (the original has a known initial-period artifact).
12. **Calendar reality:** weeks per month/quarter are irregular (52/53-week years, trading days); any weekly/daily mixing must be built on an actual date calendar. Hard-coding ratios other than 3 months/quarter will corrupt WEI-class models.
13. **COVID-scale observations** (2020Q2 is roughly 15-sigma) destroy Gaussian ML/EM estimates: detect extreme standardized observations and surface outlier-robust options (t-errors, outlier dummies, volatility scaling) with loud warnings rather than silently fitting.
14. **Factor sign/rotation indeterminacy:** fix normalization (e.g., positive loading on a named series, or an identity block) and seed EM initialization so results are bit-reproducible across runs — EM can converge to observationally equivalent rotations and users will report "random" results.
15. **Performance traps:** multivariate Kalman filtering is O(N³) per period — univariate (sequential) filtering, per-missingness-pattern steady-state caching, and Jungbacker-Koopman collapsing are what make the bootstrap/Monte Carlo pillars real. Retrofitting is painful; the filter must be built around them from day one (a requirement placed on foundations).
16. **Data-model hygiene:** track units and transformations (q/q vs annualized, SA vs NSA, levels vs log-diffs) as series metadata; mixed conventions inside one panel is a classic source of nonsense factors that no numerical care can fix.
17. **Evaluation "truth" is ambiguous with revised data:** force an explicit choice of first release vs later vintage as actuals — model rankings demonstrably flip with this choice (Croushore-Stark 2001), and a silent default hides it.

## Dependencies and shared infrastructure

### Consumed from foundations (this module is the primary requirements driver for the state-space engine)

- **Linear-Gaussian state-space engine.** Nowcasting is the most demanding customer of this engine; the following requirements originate here and are v1-blocking for foundations:
  - *Arbitrary missing observations* via row selection of the observation equation (never zero-filling), with the log-likelihood constant using the observed dimension n_t each period. Univariate/sequential filtering (Koopman & Durbin 2000) as the default path — it handles missingness for free and is 5-50x faster for large N. Joseph-form or square-root covariance updates. Reference: Durbin & Koopman (2012, ch. 4.10); validate against statsmodels KalmanFilter with NaNs and the DK Nile examples.
  - *Exact diffuse initialization* (Koopman 1997; Durbin & Koopman 2012, ch. 5): Pinf/Pstar recursions combined with univariate filtering (Koopman-Durbin 2003); Schur-based Lyapunov solver for stationary blocks. The kappa=1e7 hack loses ~7 digits exactly in the mixed-frequency setups this module targets. Validate against KFAS (R).
  - *Simulation smoother with degenerate states*: Durbin & Koopman (2002, Biometrika) mean-correction algorithm written against the R·Q·R' selection-matrix form (never requiring Q invertible), plus the precision-based sampler (Chan & Jeliazkov 2009) as an interchangeable backend — often faster for MF-VARs. Validate state-draw moments against Schorfheide-Song replication code.
  - *EM for state-space models with arbitrary missing data* (Banbura & Modugno 2014, JAE; Watson & Engle 1983 sufficient statistics), including the restricted M-step (GLS-type) for block loading structures and aggregation constraints, and the lag-one cross-covariance from the dedicated smoother recursion. Monotone log-likelihood monitored every iteration. Validate against statsmodels DynamicFactorMQ and the Banbura-Modugno euro-area replication.
  - *Observation-vector collapsing* (Jungbacker & Koopman 2015, Econometrics Journal) with per-missingness-pattern transform caching — this plus univariate filtering is what makes "NY Fed model in milliseconds" feasible. Validate: identical likelihood to the uncollapsed filter to ~1e-10.
  - *Steady-state filter detection and caching* (Harvey 1989), cached per recurring missingness pattern (e.g., the repeating monthly pattern within a quarter); convergence tolerance on ‖P_t − P_{t−1}‖ chosen safely, since a wrong tolerance silently biases likelihoods.
- **Real-time vintage data store** (Croushore & Stark 2001, JoE). This module states the requirements: a 3-D panel (series × observation-period × vintage-date) with lazy views; series metadata carrying frequency, stock/flow type, transformation, SA status, and publication-lag profile; ALFRED (St. Louis Fed) and ECB real-time DB connectors as optional I/O. This module builds its release-calendar and news layers on top.
- **Temporal disaggregation and benchmarking utilities** — Chow-Lin (Chow & Lin 1971, REStat), Fernandez (1981), Litterman (1983, JBES), Denton (1971, JASA) / Denton-Cholette (Cholette 1984). Engines live in foundations; this module owns the user-facing API and imposes the policy requirements: ML estimation of rho on the aggregated model with unit-boundary detection and automatic Fernandez fallback (with a warning); exact constraint satisfaction via the BLUE distribution formula (no post-hoc scaling); Denton-Cholette PFD as the default variant; sparse banded linear algebra. Validation: R tempdisagg (Sax & Steiner 2013) shipped examples and Dagum & Cholette (2006) book examples.
- **Time-index/calendar/frequency/holiday engine** — date-exact aggregator construction (52/53-week years, trading days) for weekly/daily activity indexes and general frequency alignment.
- **Fast quantile-regression solver** — inner solves for quantile MIDAS and quantile factor models, including monotone rearrangement.
- **Bootstrap engine and critical-value engine** — bootstrap critical values for mixed-frequency Granger max-tests and small-sample evaluation corrections.
- **Numerical optimizers** — multistart NLS with analytic gradients and profiling support for MIDAS weight functions; EM-warm-started quasi-Newton for state-space MF-VAR ML.
- **Philox-based parallel RNG, the unified forecast object (point/interval/density/path), and the golden-value validation harness** — as for every module.
- **Exogenous-regressor (covariate) contract** — MIDAS regressors and bridge-equation indicators are high-frequency covariates ingested through the shared aligned interface (mixed-frequency alignment supplied by the time-index engine); scenario nowcasts ("what if next week's indicator prints X") reuse its scenario-path machinery, and the leakage checks compose with the vintage store so only data available at the nowcast date enters pseudo-real-time exercises.

### Consumed from other modules

- **multivariate:** the single DFM implementation (PCA/EM/QML estimation core, factor-number criteria via foundations). This module wraps it with block loading restrictions, mixed-frequency measurement rows, vintage handling, the release calendar, and news decomposition — it does not reimplement factor estimation.
- **bayesian:** MF-BVAR samplers — Minnesota Normal-inverse-Wishart posteriors, KSC stochastic-volatility blocks (with the Carriero et al. 2022 triangular-sampler correction), hierarchical prior-tightness selection (Giannone-Lenza-Primiceri 2015), marginal-data-density computation.
- **ML:** penalized-regression solvers (lasso/EN/ridge, sparse-group lasso proximal-gradient) and time-series cross-validation (blocked/rolling-origin) for penalized and sg-LASSO MIDAS nowcasting; wrapped tree/boosting engines.
- **forecasting-evaluation:** forecast-comparison tests (Diebold-Mariano, Clark-West, Giacomini-White, Giacomini-Rossi fluctuation), density evaluation (log score, CRPS, PIT histograms, Rossi-Sekhposyan 2019 calibration tests), combination weights and linear/log pools (Geweke & Amisano 2011; Aastveit, Gerdrup, Jore & Thorsrud 2014, JBES), conformal prediction. This module wires them to vintages, days-to-release conditioning, and the explicit actuals choice, and co-owns the pseudo-real-time harness.
- **regime-switching:** Hamilton/Kim (1994) filters consumed by MS-MIDAS and MS-DFM.

### Exposed to other modules

- **MIDAS weighting machinery** (all schemes: Almon, exponential Almon, beta, step, Legendre bases; log-space evaluation, normalization conventions, analytic gradients) — consumed by the volatility module's GARCH-MIDAS (Engle, Ghysels & Sohn 2013, REStat; Conrad & Kleen 2020, JAE; validated there against R mfGARCH), which lives in the volatility module per the ownership map.
- **Release-calendar layer and pseudo-real-time evaluation harness** — available to any module that wants honest recursive backtests on revised data.
- **News decomposition API** — a model-agnostic revision-attribution interface (Kalman news for state-space models, Shapley news for ML models) usable by multivariate and Bayesian model classes.
- **Conditional forecasting facade** (hard and soft conditioning) over the foundations state-space engine — reused by multivariate and Bayesian VAR workflows.
- **Temporal disaggregation user-facing API** over the foundations engines.

## Validation gallery

- **NY Fed Nowcasting_Code (Bok et al. 2018)** — reproduce published nowcast paths for their sample to 3+ decimals, and beat the archived Matlab code on speed by an order of magnitude.
- **statsmodels DynamicFactorMQ (Fulton)** — match log-likelihoods, smoothed factors, and `news()` decompositions including the parameter-revision remainder.
- **Schorfheide & Song (2015, JBES) replication + R mfbvar** — match MF-BVAR posterior moments on the same dataset; target 10x+ speed.
- **R midasr (Ghysels, Kvedaras & Zemlys 2016, JSS)** — coefficient-for-coefficient agreement for Almon, exponential Almon (matched normalization convention), beta, and U-MIDAS.
- **Banbura & Modugno (2014, JAE) euro-area replication** — EM estimates under arbitrary missingness and the news decomposition.
- **Giannone, Reichlin & Small (2008, JME)** — reproduce the RMSE-vs-horizon (days-to-release) curves from the pseudo-real-time harness.
- **R tempdisagg (Sax & Steiner 2013)** — numerical agreement on shipped Chow-Lin/Denton-Cholette examples (through the foundations engines, via this module's API).
- **ADS index (Philadelphia Fed) and WEI vintages (Dallas Fed/NY Fed)** — reproduce published weekly/daily index values with calendar-exact aggregation.
- **Brave, Butters & Justiniano (2019, IJF)** — match the Chicago Fed MF-BVAR published RMSEs (public code).
- **R midasml (Striaukas)** — sg-LASSO MIDAS estimates and the Babii-Ghysels-Striaukas (2022) US GDP nowcasting exercise.
- **Mogliani & Simoni (2021, JoE) replication files (Banque de France)** — Bayesian penalized MIDAS posteriors.
- **Antolin-Diaz, Drechsel & Petrella (2017, REStat) replication code** — DFM with time-varying long-run growth and SV.
- **Chauvet-Piger real-time recession probabilities** — MS-DFM recession nowcasts.
- **KFAS (R) and Durbin-Koopman Nile examples** — exact diffuse and missing-data filtering (jointly with foundations).
