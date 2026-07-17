# Module 04 — Multivariate Models

> Part of the time series econometrics library roadmap. Master plan: [ROADMAP.md](../../ROADMAP.md).

**This module is the library's multivariate workhorse: reduced-form VARs and their entire frequentist ecosystem — lag selection, diagnostics, Granger causality, IRF/FEVD machinery with modern bootstrap inference, forecasting, VARX/VARMA, cointegration and VECMs, dynamic factor models and FAVARs, large/regularized VARs, nonlinear and time-varying VARs, panel and global VARs, connectedness analysis, and quantile VARs — built on a Rust core fast enough to make best-practice inference (Kilian double bootstrap, KPP generalized IRFs, moving-block bootstrap) the default rather than a luxury.**

## Purpose and scope

This module covers everything a frequentist macroeconometrician or empirical-finance researcher does with systems of time series: estimating reduced-form VARs, VARX, and identified VARMA models; testing for and modeling cointegration (Engle-Granger through Johansen, VECMs, threshold and broken-trend variants, ARDL bounds); extracting factors from large panels and running FAVARs; regularizing high-dimensional VARs; fitting regime-switching, threshold, and smooth-transition VARs with proper generalized impulse responses; estimating panel and global VARs; and computing Diebold-Yilmaz connectedness. It owns the library's single dynamic factor model implementation (the nowcasting module layers vintage and news machinery on top of it) and the Granger-causality toolkit (re-exported by the diagnostics module).

Its users span applied macro researchers replicating and extending published VAR studies, central-bank and IMF staff running conditional forecasts, historical decompositions, GVARs, and DFM-based pipelines, and empirical-finance researchers doing connectedness, price discovery, and realized-volatility (VHAR) work. A stated pillar of the project is served here: the module must not just compute, but guide — warning when persistence invalidates long-horizon asymptotics, when ARCH effects invalidate iid bootstraps, and making sensitivity analysis (Cholesky orderings, band types, GFEVD normalizations) first-class API options.

Relative to the rest of the library: this module produces reduced-form estimates and forecast objects; structural identification (SVAR/SVEC restriction and rotation machinery) lives in the identification module, which consumes this module's VAR/VECM fits. IRF results flow through the foundations-owned typed IRF object, and this module contributes the Koop-Pesaran-Potter generalized-IRF simulation engine to foundations for library-wide reuse. Bootstrap resampling, HAC/long-run variance estimation, the state-space engine, the factor-estimation core, and the Monte Carlo executor are all consumed from foundations; penalized solvers and time-series cross-validation come from the ML module.

## Where existing tools fall short

- **statsmodels VAR** offers only asymptotic and plain residual-bootstrap IRF bands: no Kilian bias-corrected bootstrap-after-bootstrap, no wild or moving-block bootstrap, no simultaneous (sup-t) bands, no historical decomposition, and no Toda-Yamamoto helper — users hand-roll all of these.
- **statsmodels VECM** lacks restricted alpha/beta LR testing (the switching algorithm), structural VECM, Bartlett corrections, and bootstrap rank tests, and has limited deterministic-case handling; **R urca** still ships Osterwald-Lenum table lookups instead of MacKinnon-Haug-Michelis response-surface p-values.
- The **R ecosystem is fragmented** across vars, urca, tsDyn, svars, ConnectednessApproach, frequencyConnectedness, BigVAR, etc., with mutually inconsistent conventions (Sigma degrees of freedom, deterministic-term cases, interval types) and no shared IRF/bootstrap engine — and R loops make serious bootstrap/Monte Carlo work painfully slow.
- **VARMA is effectively unusable everywhere**: statsmodels VARMAX ignores identification (echelon form/Kronecker indices) and is slow; R MTS is fragile and unmaintained; no package makes identified VARMA routine.
- **FAVAR, GVAR, panel VAR, and interacted VAR** live in one-off MATLAB/Stata/GAUSS code (BBE replication files, GVAR Toolbox, Stata pvar, IVAR toolbox) with no maintained, tested Python implementation at all.
- **No mainstream library implements heteroskedasticity-valid bootstrap theory for VARs** (Brüggemann-Jentsch-Trenkler 2016 moving-block bootstrap; Gonçalves-Kilian wild bootstrap), even though iid residual bootstraps are formally invalid for most monthly/financial applications — including all FEVD and Cholesky-IRF bands.
- **Nonlinear VAR inference is an afterthought**: tsDyn fits TVAR/TVECM but GIRF bands, regime-dependent IRF inference, and threshold-test bootstraps are partial, slow, and thinly documented; MS-VAR has no maintained home in either R or Python (MSBVAR is archived; statsmodels MarkovAutoregression is univariate).
- **Connectedness/spillover analysis** — one of the most-used multivariate methods in empirical finance — has no Python home, and the R packages disagree on GFEVD normalization conventions without saying so.
- **Missing-data handling** (EM/Kalman) exists only inside DFM code paths; plain VARs/VECMs with ragged edges or internal missing values are unsupported everywhere.
- **No package teaches model choice**: nothing guides users among band types (pointwise vs sup-t, percentile vs Hall), VAR-in-levels vs VECM, GIRF vs Cholesky, or warns when persistence invalidates long-horizon asymptotics — a stated pillar of this project.
- **Quantile VARs, mixed-frequency VARs, frequentist TVP-VARs, and time-varying Granger causality** exist only as paper-specific replication code.

## Inventory

Difficulty scale: low / medium / high / research-grade. Items reassigned by the shared-infrastructure ownership map appear under [Dependencies and shared infrastructure](#dependencies-and-shared-infrastructure) rather than in these tables.

### Tier 1 — Core (v1-blocking)

**VAR core**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| VAR(p) estimation (multivariate LS / equation-by-equation OLS, ML) | Workhorse reduced-form vector autoregression; the substrate for nearly everything else in this domain — forecasting, causality, and the reduced form behind structural analysis. | low | One shared regressor matrix Z; estimate all equations with a single QR (or Cholesky of Z'Z) — never invert explicitly. Offer both ML (divide by T) and OLS-df (T−Kp−1) covariance conventions and document the default; this mismatch is the #1 cause of statsmodels-vs-R-vars replication failures. Lütkepohl (2005) ch. 3. Validate against Lütkepohl's West German investment/income/consumption (E1) worked examples; cross-check statsmodels VAR and R vars::VAR to machine precision. |
| Lag-order selection (AIC, BIC/SC, HQ, FPE, sequential LR) | Information-criteria and testing-based choice of p; a mandatory pre-step for essentially every VAR application. | low | Hold the estimation sample fixed across candidate p (drop max-p initial observations) or criteria are not comparable — packages differ here and it changes the selected p. Include the small-sample LR correction (Sims 1980). Lütkepohl (2005) ch. 4. Validate against vars::VARselect and statsmodels select_order on the same fixed sample. |
| VAR diagnostics suite (stability, Portmanteau, Breusch-Godfrey LM, multivariate ARCH-LM, Doornik-Hansen/Jarque-Bera normality) | Post-estimation residual and stability checks; practitioners run these on every fitted VAR before trusting inference. | medium | Stability via companion-matrix eigenvalue moduli. Portmanteau needs the adjusted (Ljung-Box-type) small-sample version with df = K²(h−p). Doornik-Hansen (2008) normality requires the specific square-root decomposition of the correlation matrix — Cholesky silently changes the statistic. Validate against vars::serial.test, arch.test, normality.test. |
| Granger causality and instantaneous causality tests (Wald/F) | Block-exclusion Wald/F tests for predictive causality between variable groups, plus instantaneous causality via residual covariance restrictions. This module owns Granger tooling library-wide (diagnostics re-exports). | low | Offer HC/HAC-robust Wald variants (rarely available elsewhere). Support group-to-group testing, not just pairwise, matching vars::causality. Pitfall: F vs chi-square small-sample versions differ across packages. Lütkepohl (2005) ch. 3.6. Validate against vars::causality and Stata vargranger. |
| Toda-Yamamoto lag-augmented Granger causality | Causality testing valid with unit roots/cointegration of unknown order: fit VAR(p+d_max) but Wald-test only the first p lag blocks. | low | The classic user error is restricting all p+d_max lags; the extra d_max lags must remain untested. Automate d_max via unit-root pretests with user override. Toda & Yamamoto (1995). No mainstream library has this built in — validate against manual constructions in published replication files (Zapata-Rambaldi 1997 examples). |

**IRF/FEVD machinery**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Orthogonalized (Cholesky) IRFs and FEVD | Recursive-identification impulse responses and forecast error variance decompositions; the default structural output every user expects. (Full SVAR restriction machinery lives in the identification module.) | low | Compute MA coefficients by iterating the Phi recursion, never via eigendecomposition of the non-normal companion matrix. Make ordering-dependence loud in docs and ship an "all orderings" sensitivity helper. Sims (1980); Lütkepohl (2005) ch. 2. Validate against statsmodels irf/fevd and Lütkepohl textbook tables. |
| Generalized IRFs and generalized FEVD (Pesaran-Shin) | Order-invariant impulse responses/decompositions; also the input to Diebold-Yilmaz connectedness. | low | GFEVD rows do not sum to one; implement both row-normalization (Diebold-Yilmaz convention) and the Lanne-Nyberg (2016) normalization, and label which is used. Pesaran & Shin (1998). Validate against R ConnectednessApproach internals and the Pesaran-Shin worked example. |
| Cumulative IRFs with correctly cumulated uncertainty | A `cumulative=True` view on every IRF (level responses to a shock when the model is in differences; running totals generally) where the *bands* are cumulated correctly, not just the point path. | medium | Cumulating the point IRF is a partial sum; cumulating the uncertainty is not — the summed responses are correlated across horizons. Three correct routes, one per inference type: (i) simulation/bootstrap/Bayesian: cumulate *within each draw*, then take quantiles of the cumulated draws; (ii) frequentist asymptotic: delta method on the partial-sum transformation using the joint covariance of the MA coefficients (Lütkepohl 2005 ch. 3.7); (iii) LP: estimate the cumulated outcome directly (the Ramey-Zubairy one-step convention — see Module 07). Summing per-horizon standard errors, or bands, is always wrong and the docs must say so. Validate: cumulative orth IRFs against statsmodels `irf(...).cum_effects` and bootstrap-band coverage against a Monte Carlo. |
| Asymptotic (delta-method) IRF and FEVD confidence bands | Closed-form Gaussian bands from the asymptotic distribution of VAR coefficients; the fast baseline inference. | medium | The Lütkepohl (1990; 2005 ch. 3.7) analytic derivatives involving duplication/elimination matrices and Kronecker products are the classic bug source — unit-test every formula against central-difference numerical derivatives. FEVD bands additionally need the covariance between coefficient and Sigma estimates. Validate against vars::irf(boot=FALSE) and JMulTi asymptotic bands. |
| Residual (iid) bootstrap IRF bands: Efron percentile, Hall, studentized | Standard recursive-design residual bootstrap for IRF confidence intervals; the default in R vars. | medium | Recenter residuals before resampling; document initial-condition treatment (fixed presample vs randomized) — it changes results. Efron vs Hall percentile intervals differ materially at long horizons; expose both plus symmetric-t. Kilian & Lütkepohl (2017) ch. 12. Validate coverage against Kilian (1998) Monte Carlo designs; point-match vars::irf(boot=TRUE) given identical resampling. Consumes the foundations bootstrap engine. |
| Kilian bootstrap-after-bootstrap bias-corrected IRF bands | Double bootstrap that estimates small-sample coefficient bias, corrects, then bootstraps the corrected model; accepted best-practice frequentist bands for persistent VARs. | medium | Must implement Kilian's stationarity correction: if bias-adjusted coefficients become explosive, shrink the correction until the largest companion root is inside the unit circle — omitting this breaks everything. Computationally heavy (B1×B2 fits): exactly where the Rust core wins; counter-based RNG for reproducible parallel streams. Kilian (1998, REStat). Validate against Kilian's published Monte Carlo coverage and the implementation inside R lpirfs. |
| Historical decomposition | Decomposes each variable's observed path into cumulative contributions of each structural shock plus initial conditions and deterministic terms; a staple of central-bank storytelling. | medium | Enforce the exact adding-up identity (data = shock contributions + initial-condition component + deterministic component) as a runtime check; packages disagree on baseline definitions, so state it precisely. Burbidge & Harrison (1985); Kilian & Lütkepohl (2017) ch. 4. statsmodels lacks this entirely. Validate against R svars::hd and JMulTi. |

**Forecasting**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| VAR forecasting: iterated point forecasts, asymptotic and simulation/bootstrap intervals, density forecasts | Multi-step forecasting with honest uncertainty, including parameter-uncertainty-adjusted intervals and simulated predictive densities (emitted via the unified forecast object; evaluation lives in forecasting-evaluation). | medium | Asymptotic MSE matrices per Lütkepohl (2005) ch. 3.5 including the parameter-estimation correction term (statsmodels omits it in places). Bootstrap intervals via backward-representation or bias-corrected resampling (Pascual-Romo-Ruiz 2004 logic extended to VARs). Validate point forecasts against statsmodels/vars exactly; intervals via Monte Carlo coverage. |

**VARX & VARMA**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| VARX / VAR with exogenous variables and dynamic multipliers | VAR augmented with contemporaneous and lagged exogenous inputs; produces dynamic multipliers (responses to observed inputs) distinct from shock IRFs. | medium | Keep multipliers to X and IRFs to endogenous shocks as separate, clearly named outputs — conflating them is a common user confusion. Multiplier bands via delta method or bootstrap. Lütkepohl (2005) ch. 10; Tsay (2014). Validate against MTS::VARX and Stata var with exog. |

**Cointegration & VECM**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Engle-Granger two-step and Phillips-Ouliaris residual cointegration tests | Single-equation residual-based cointegration testing; the pedagogical and practical entry point to cointegration. | low | Use MacKinnon (1996/2010) response-surface p-values (depend on regressor count and deterministic terms), not stale table lookups. Phillips-Ouliaris Z-alpha/Z-t need long-run variance estimation with kernel/bandwidth choices exposed (consumed from foundations). Validate against urca::ca.po, aTSA, and MacKinnon's published numerical values. |
| Johansen cointegration rank tests (trace and max-eigenvalue) with the five deterministic cases | System ML rank determination; the canonical cointegration methodology every user will demand on day one. | high | Solve the generalized eigenproblem via Cholesky of S11 (symmetrized) for numerical stability; sort eigenvalues descending. Ship MacKinnon-Haug-Michelis (1999) response-surface p-values (urca still uses tables) plus the Johansen (2002) Bartlett small-sample correction. The mapping of the 5 deterministic cases across packages (urca ecdet, statsmodels det_order, EViews cases 1-5) is inconsistent — provide an explicit translation table in docs. Validate against Johansen & Juselius (1990) Danish/Finnish money-demand results and urca::ca.jo. |
| VECM estimation (reduced-rank regression / Johansen ML) with mapping to level-VAR representation | Full VECM estimation given rank r, plus conversion to the implied VAR in levels for IRFs, FEVD, and forecasting under cointegration. | medium | Beta is identified only up to rotation: apply Phillips triangular normalization by default before reporting standard errors (standard errors on unnormalized beta are meaningless). IRFs from VECMs do not die out — document the permanent effects. Johansen (1995); Lütkepohl (2005) ch. 6-7. Validate against urca::cajorls + vec2var and statsmodels VECM on the Lütkepohl-Krätzig (2004) examples. |
| Restriction testing on alpha and beta: weak exogeneity, restricted cointegration vectors, joint restrictions (switching algorithm) | LR tests for hypotheses like "money is weakly exogenous" or "PPP holds in the cointegrating space"; the substantive payoff of the Johansen framework. | high | Linear restrictions H·phi on beta and A·psi on alpha via eigenvalue problems; joint restrictions need the Johansen-Juselius switching algorithm, which has local optima — use multiple starts and verify the LR statistic is non-negative and monotone across iterations. Boswijk & Doornik (2004) is the best implementation survey. Validate against urca::blrtest/alrtest/ablrtest and PcGive/CATS in RATS published output (Juselius 2006 textbook examples). |

**Factor models & FAVAR**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| QML/EM dynamic factor model with arbitrary missing data and mixed frequency (Doz-Giannone-Reichlin 2012; Banbura-Modugno 2014) | Full EM estimation handling ragged edges, mixed monthly/quarterly panels, and factor blocks; the engine behind institutional nowcasts. This is the library's single DFM implementation, owned here; nowcasting consumes it. | high | Use Durbin-Koopman univariate treatment of multivariate observations or square-root filtering for time-varying observation dimension (via the foundations state-space engine); enforce and assert EM log-likelihood monotonicity; quarterly variables enter via the Mariano-Murasawa (2003) 5-lag aggregation constraint. DGR (2012, REStat); Banbura & Modugno (2014, JAE). Validate against statsmodels DynamicFactorMQ (match it, then beat it on speed) and FRBNY nowcast outputs. |

### Tier 2 — Standard (expected of a serious library)

**VAR core**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Restricted / subset VAR (EGLS) with linear parameter restrictions | Zero and linear restrictions on VAR coefficients estimated by feasible GLS; prunes overparameterized systems and imposes theory-driven exclusions. | medium | EGLS with restriction matrix R (vec(B)=R·gamma), per Lütkepohl (2005) ch. 5; also top-down/bottom-up sequential elimination search as in JMulTi. Pitfall: after restriction, delta-method IRF covariances must use the restricted asymptotic covariance. Validate against JMulTi subset VAR output and R vars::restrict. |

**IRF/FEVD machinery**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Analytical small-sample bias correction (Pope / Nicholls-Pope) | Closed-form first-order bias formula for VAR coefficients; a cheap alternative to bootstrap bias estimation. | medium | Pope (1990) generalizes Nicholls-Pope (1988) to VAR(p) via companion form; requires solving a discrete Lyapunov equation (Bartels-Stewart, not a vectorized Kronecker inverse, for large K·p). Apply the same stationarity shrinkage as Kilian. Validate against R BVAR-adjacent implementations and by Monte Carlo (visibly smaller bias at AR roots near 0.95). |
| Wild bootstrap for IRFs (Gonçalves-Kilian) | Recursive-design wild bootstrap robust to conditional heteroskedasticity of unknown form; essential for financial and monthly macro data. | medium | Multiply each residual *vector* by one scalar draw (Rademacher or Mammen) to preserve contemporaneous cross-equation dependence — per-element draws destroy the covariance and are a common bug. Gonçalves & Kilian (2004, JoE). Implement the recursive-design variant (fixed-design has different validity properties). Validate via Monte Carlo coverage under GARCH errors reproducing their tables. |

**Breaks & stability**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Structural stability tests for VARs (bootstrapped Chow, OLS-CUSUM/MOSUM, Nyblom fluctuation) | Detecting parameter instability in fitted VARs; a standard robustness requirement for published work. | medium | Chow-type tests in VARs have badly distorted asymptotic sizes — bootstrap the null distribution as in Candelon & Lütkepohl (2001). Reference implementation: vars::stability via strucchange. Validate against vars/strucchange output on the Canada dataset used in Pfaff (2008). |
| VAR with exogenous/known breaks (shift dummies, partial structural change) | Deterministic level/trend shifts and regime dummies interacted with coefficient subsets; the pragmatic tool for COVID-type discontinuities. | low | Straightforward regressor engineering, but IRF machinery must know which coefficients are regime-specific, and lag selection/diagnostics must condition on the dummies. Include the Lenza & Primiceri (2022) COVID dummy/volatility-rescaling treatment as a documented recipe. Validate by construction against manual dummy-augmented OLS. |

**Cointegration & VECM**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Efficient single-equation cointegrating regression: FM-OLS, DOLS, CCR | Asymptotically efficient estimators of a single cointegrating vector with valid normal inference; standard in applied energy/finance work. | medium | FM-OLS (Phillips-Hansen 1990) needs the one-sided long-run covariance Delta; DOLS (Stock-Watson 1993) needs lead/lag selection and HAC standard errors; CCR (Park 1992). Kernel and bandwidth defaults must be documented — they drive results. Validate against R cointReg and EViews (the de facto benchmark applied users compare to). |
| Permanent-transitory decompositions: Gonzalo-Granger, Gonzalo-Ng, multivariate Beveridge-Nelson | Decomposes cointegrated systems into common permanent components and transitory dynamics; used in price discovery (finance) and trend-cycle analysis. | medium | Gonzalo-Granger (1995) common-factor weights alpha_perp; Gonzalo-Ng (2001) orthogonalized P-T shocks; multivariate Beveridge-Nelson via the VECM MA representation. Price-discovery metrics (Hasbrouck information shares, component shares) are a cheap, high-demand add-on for finance users. Validate against published price-discovery examples (Hasbrouck 1995 setup) and R apt/pdshare-style packages. |
| ARDL bounds testing for level relationships (Pesaran-Shin-Smith) | Single-equation testing for a long-run relationship without pre-classifying regressors as I(0)/I(1); wildly popular in applied work. | medium | F- and t-bounds with Kripfganz & Schneider (2020) response-surface critical values (finite-sample, superior to the PSS 2001 asymptotic tables); careful per-variable lag selection; report the error-correction reparameterization with correct standard errors. Validate against Stata ardl (Kripfganz-Schneider), the de facto standard. |

**Factor models & FAVAR**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Dynamic factor model, two-step estimation (Doz-Giannone-Reichlin 2011) | PCA factors refined by a Kalman smoother given a VAR law of motion for factors; fast, robust, standard for nowcasting pipelines. | medium | Steps: PCA → VAR on factors → Kalman smoothing treating loadings/variances as known. Consistency holds for large N,T despite misspecification. Doz, Giannone & Reichlin (2011, JoE). Validate against the New York Fed nowcasting code (public MATLAB/Python) intermediate outputs. |
| FAVAR (Bernanke-Boivin-Eliasz): two-step PCA with slow/fast rotation and joint estimation | VAR augmented with estimated factors so monetary-policy IRFs can be traced on hundreds of series; a named requirement for this library. | high | Two-step: extract factors from the full panel, purge the policy rate from "fast" factors using slow-variable factors (the BBE rotation), run VAR(factors, FFR), map IRFs back through loadings. Bands must account for factor-estimation uncertainty — bootstrap the whole two-step pipeline. Bernanke, Boivin & Eliasz (2005, QJE). Validate by replicating BBE Figure 2 IRFs (their dataset is public). |

**Large & regularized VARs**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Reduced-rank VAR and canonical-correlation-based index models | Rank-restricted coefficient matrices (RRVAR) as classical dimension reduction; includes serial-correlation common features (SCCF). | medium | Estimation by canonical correlations between y_t and lags (Velu, Reinsel & Wichern 1986; Reinsel-Velu 1998); SCCF/common-cycle tests per Vahid & Engle (1993). Validate against Reinsel-Velu textbook examples and GAUSS/R code from the common-features literature. |

**Network & connectedness**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Diebold-Yilmaz connectedness (spillover) tables, indices, rolling analysis | GFEVD-based directional spillover measures and network graphs; enormously popular in empirical finance (thousands of citations, no first-class Python home). | medium | Generalized FEVD at horizon H with row normalization; total/directional/net/pairwise measures; rolling windows with bootstrap or asymptotic bands. Sensitivity to VAR lag, horizon, and window must be first-class API options. Diebold & Yilmaz (2009, 2012, 2014). Validate against the R ConnectednessApproach/Spillover packages and the DY 2012 published spillover tables. |

### Tier 3 — Advanced (differentiators)

**IRF/FEVD machinery**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Moving-block bootstrap for VAR inference (Brüggemann-Jentsch-Trenkler) | Blockwise resampling of residual vectors; the only bootstrap proven valid for joint (coefficients, Sigma) inference — hence FEVD and structural IRF bands — under conditional heteroskedasticity. | high | Key result: even the wild bootstrap is invalid for Sigma-dependent statistics (Cholesky IRFs, FEVD) under conditional heteroskedasticity; MBB is valid. Needs block-length selection (rule-of-thumb ≈ 5.03·T^0.25 plus user override) and residual recentering within the block scheme. Brüggemann, Jentsch & Trenkler (2016, JoE). Almost no library has this; validate against the authors' simulation designs. |
| Simultaneous confidence bands: sup-t, Bonferroni, projection (Montiel Olea & Plagborg-Møller) | Bands with joint coverage across horizons, replacing the misleading pointwise bands users habitually over-interpret. | medium | Sup-t via plug-in Gaussian quantile of the max-\|t\| statistic or bootstrap; it is the narrowest band with correct joint coverage. Offer as a one-flag alternative on every IRF plot. Montiel Olea & Plagborg-Møller (2019, JAE). Validate against their MATLAB replication code. |

**Forecasting**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Conditional forecasts and scenario analysis (hard and soft conditioning) | Forecasts constrained so selected variables follow assumed paths (e.g., an oil-price scenario); heavily used in policy institutions. | high | Implement via stacked linear-Gaussian conditioning (Doan-Litterman-Sims 1984; Waggoner-Zha 1999 algebra in a frequentist plug-in sense) or Kalman smoother on the state-space form (Banbura, Giannone & Lenza 2015), which scales better and handles soft conditions. Pitfall: the distribution of conditional forecasts under structural vs reduced-form shocks differs — document. Validate against the BEAR toolbox / ECB BGL replication. |

**Breaks & stability**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Multiple structural breaks in multivariate systems (Qu-Perron) | Estimation and testing of multiple unknown break dates in systems of equations, allowing breaks in coefficients and/or covariance. | high | Dynamic-programming break-date search with quasi-ML objective; allow restricted break configurations (breaks in some equations only). Numerically demanding — a good Rust target. Qu & Perron (2007, Econometrica). Validate against the authors' GAUSS code on their US term-structure example. |

**VARX & VARMA**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| VARMA estimation with echelon-form / Kronecker-index identification | Parsimonious VARMA(p,q) as the theoretically closed class (marginalization/aggregation of VARs); needed for DSGE-consistent modeling and better small-sample forecasting. | high | Unidentified without canonical structure: echelon form with Kronecker indices via the Hannan-Kavalieris/Poskitt procedure, Hannan-Rissanen long-AR two-stage initial estimates, then exact ML via state-space Kalman filter. Enforce invertibility by flipping MA roots. Hannan & Deistler (1988); Lütkepohl (2005) ch. 12; Athanasopoulos & Vahid (2008). Validate against R MTS and Gretl VARMA on Tsay's examples; beat statsmodels VARMAX on speed and identifiability handling. |

**Cointegration & VECM**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Threshold cointegration / threshold VECM (Hansen-Seo) | Two-regime VECM where error correction switches by the lagged equilibrium error; standard for asymmetric adjustment (interest rates, law of one price). | high | Joint grid search over cointegrating parameter and threshold with trimming (≥5-10% of observations per regime); sup-LM test with fixed-regressor and residual bootstrap null distributions. Hansen & Seo (2002, JoE). Validate against tsDyn::TVECM and TVECM.HStest reproducing the Hansen-Seo term-structure application. |
| Cointegration with structural breaks (Johansen-Mosconi-Nielsen; Gregory-Hansen; Saikkonen-Lütkepohl) | Rank testing and single-equation cointegration tests allowing broken deterministic trends or a broken cointegrating vector. | high | JMN (2000) modifies the Johansen likelihood with broken trends and supplies new response-surface critical values; Gregory-Hansen (1996) is the single-equation sup-ADF analogue; include Saikkonen-Lütkepohl (2000) GLS-detrended rank tests, popular in Europe via JMulTi. Validate against JMulTi and the papers' tabulated critical values. |

**Factor models & FAVAR**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Generalized dynamic factor model (Forni-Hallin-Lippi-Reichlin), spectral/dynamic PCA | Frequency-domain factor extraction allowing dynamic loading structure; the other main factor tradition, with one-sided filtering variants for forecasting. | high | Dynamic eigenanalysis of the smoothed spectral density matrix; the two-sided filter is unusable at sample ends — implement the one-sided version (Forni et al. 2005/2017). Hallin-Liska (2007) criterion for the number of dynamic factors. Validate against the authors' MATLAB code and R GDFM implementations. |

**Large & regularized VARs**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Lasso/elastic-net VAR with hierarchical lag penalties (HLag / BigVAR) | Sparse estimation for VARs with dozens-to-hundreds of variables where OLS is infeasible or noisy; frequentist counterpart to big BVARs. VAR-specific wrapper — solvers and time-ordered CV consumed from the ML module. | high | Proximal-gradient/coordinate descent with structured penalties (componentwise, own/other, elementwise HLag) that shrink distant lags more; rolling-window cross-validation respecting time order (never k-fold shuffle). Basu & Michailidis (2015, AoS) for theory; Nicholson, Wilms, Bien & Matteson (2020, JMLR) for HLag. Validate against R BigVAR numerically. |
| VHAR (vector heterogeneous autoregression) | Multivariate HAR for realized-volatility panels: a VAR(22) restricted to daily/weekly/monthly averages; the standard model for multivariate realized volatility. | low | Implement as a restricted VAR via the fixed aggregation matrix (so all IRF/forecast machinery is inherited free), with optional lasso on the HAR terms. Corsi (2009) univariate; Cubadda, Guardabascio & Hecq (2017) vector version. Validate against R bvhar (frequentist VHAR) and HARModel outputs. |

**Nonlinear & time-varying VARs**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Markov-switching VAR (MS-VAR, Krolzig taxonomy: MSI/MSA/MSH variants) | VARs whose intercepts, coefficients, and/or covariances switch with a hidden Markov chain; standard for recession dynamics and regime-dependent policy effects. | high | Hamilton filter + Kim smoother inside EM; compute in log-space or with per-period scaling to avoid underflow; handle label switching with an ordering constraint (regime 1 = low mean or low variance); EM has local maxima — multiple random + moment-based starts, and watch for degenerate regimes (one regime capturing a single outlier). Regime-dependent and KPP generalized IRFs. Hamilton (1989); Krolzig (1997). Validate against Krolzig's MSVAR (Ox) published output and MS_Regress (MATLAB) examples; EM-information-matrix standard errors are delicate — offer a parametric bootstrap. |
| Threshold VAR (TVAR) with Tsay/Hansen threshold tests and simulated GIRFs | Regime switching triggered by an observed variable crossing a threshold (e.g., credit spread, inflation); the workhorse for asymmetry questions in fiscal/financial macro. | high | Grid search over threshold (and delay) with 10-15% trimming; sup-Wald/sup-LR tests bootstrapped under the linear null (Hansen 1996 fixed-regressor bootstrap; Tsay 1998 multivariate test). Nonlinear IRFs must be Koop-Pesaran-Potter (1996) GIRFs: simulate paths with and without a shock, averaging over histories and future shocks, with history-conditional bands. Validate against tsDyn::TVAR and published GIRFs in Balke (2000). |
| Smooth-transition VAR (STVAR, Auerbach-Gorodnichenko style) | Logistic smooth weighting between regimes (recession/expansion); dominant in the state-dependent fiscal-multiplier literature. | high | The transition slope gamma is weakly identified — AG fix it by calibrating the fraction of time in recession; expose that convention plus profile-likelihood estimation with a documented warning. Estimation by NLS/QML with regime-weighted covariances; GIRFs by simulation as in KPP (1996). Auerbach & Gorodnichenko (2012, AEJ:Policy). Validate against the AG replication files reproducing their multipliers. |
| Koop-Pesaran-Potter generalized IRF engine for nonlinear models | A single simulation engine computing history- and shock-dependent GIRFs with Monte Carlo integration over histories and future shocks, reused by TVAR/STVAR/MSVAR/IVAR. Built on the foundations typed-IRF object; this module contributes the engine to foundations for library-wide reuse. | high | Design as shared infrastructure: given any model exposing simulate(state, shocks), compute GIRF(h, shock, history-set) with antithetic variates and common random numbers for variance reduction — a major Rust speed showcase (R implementations take hours). Koop, Pesaran & Potter (1996, JoE). Validate against tsDyn GIRFs and published STVAR/TVAR GIRF figures (Auerbach-Gorodnichenko 2012; Balke 2000). |

**Panel & global VARs**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Panel VAR (GMM: Holtz-Eakin-Newey-Rosen / Arellano-Bond; fixed-effects LSDV with bias correction) | VARs on cross-section-by-time panels (countries, firms) with fixed effects; the standard tool for micro-macro dynamic questions. | high | Nickell bias makes LSDV inconsistent for small T: implement forward-orthogonal-deviation (Helmert) transformation + system GMM, with instrument-count collapsing to avoid proliferation bias, Hansen-J and AR(2) diagnostics; panel-specific IRFs/FEVD with GMM-based bands. Holtz-Eakin, Newey & Rosen (1988); Abrigo & Love (2016, Stata Journal). Validate against Stata pvar and pvarirf on the Abrigo-Love example datasets. |
| GVAR (Global VAR, Pesaran et al.) | Links many country VARX* models via trade-weighted foreign variables and solves the stacked global system; the standard international-spillover framework at central banks and the IMF. | high | Per-country VECMX* with weakly exogenous foreign "star" variables (test weak exogeneity), link matrices from (possibly time-varying) trade weights, stack and solve the global companion form; GIRFs are the natural impulse concept. Pesaran, Schuermann & Weiner (2004, JBES); Dees, di Mauro, Pesaran & Smith (2007, JAE). Validate against the Smith-Galesi GVAR Toolbox 2.0 (MATLAB) on the DdPS dataset — reproducing its GIRFs is the acceptance test. |

**Mixed frequency & misc**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Mixed-frequency VAR (stacked/blocked frequentist MF-VAR) | VARs mixing monthly and quarterly variables without aggregation, in observation-driven (Ghysels stacking) form; complements the state-space DFM route. | high | Stack high-frequency observations within the low-frequency period as separate rows (Ghysels 2016, JoE); dimension grows fast, so pair with HLag-type shrinkage; IRFs at both frequencies need careful labeling. The alternative state-space MF-VAR with Kalman-EM shares infrastructure with the DFM. Validate against Ghysels' MATLAB replication code, with the (Bayesian) mfbvar package point estimates as sanity checks. |

### Tier 4 — Frontier (research-grade, each gated on a named validation target)

**IRF/FEVD machinery**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Uniformly valid IRF inference with possible unit roots (Inoue-Kilian) | Bootstrap procedures whose coverage is uniform over stationary, local-to-unity, and unit-root regions; fixes the known failure of standard bands for persistent data at long horizons. | research-grade | Inoue & Kilian (2020, JoE); complement with lag-augmentation ideas (Dolado-Lütkepohl 1996; Toda-Yamamoto logic applied to IRFs). At minimum, emit a documented warning when the largest companion root exceeds ~0.97 and horizons are long, pointing users to these methods. Gate: reproduce the paper's Monte Carlo designs. |
| Functional VAR (fVAR): distributions as state variables | Joins macro aggregates with entire cross-sectional distributions (income, firm size, beliefs) in one VAR: the distribution is a function-valued variable, so IRFs answer "how does a monetary shock move the whole earnings distribution?" — the reduced-form bridge between heterogeneous-agent models and time series data. | research-grade | Compress each period's cross-sectional density into a finite sieve basis (splines/orthogonal polynomials on the log-density, as in Chang, Chen & Schorfheide's heterogeneity fVAR), stack the coefficients with the aggregates in a (B)VAR — shrinkage is essential since the state is large, so this rides on the Module 05 Minnesota machinery. Key traps: density-positivity and integrate-to-one constraints on responses (impose via the log-density parameterization); basis-dimension selection; sampling error in the per-period density estimates enters as measurement error (state-space treatment). Functional IRFs plot as response *curves* per horizon — needs the Module 13 heatmap/small-multiple treatment plus the cumulative view (cumulate draw-wise, as everywhere). Gate: reproduce the Chang-Chen-Schorfheide earnings-distribution IRFs from their replication code. |

**VARX & VARMA**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| VARMA via scalar-component models (SCM) | Athanasopoulos-Vahid alternative identification of VARMA through scalar components; shown to out-forecast VARs on macro data. | research-grade | Sequence of canonical-correlation tests to find SCM orders, then FIML with rotation restrictions. Delicate testing cascade; automate with a clear audit trail. Athanasopoulos & Vahid (2008, JBES). Gate: reproduce their published forecast-comparison results. |

**Cointegration & VECM**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Bootstrap cointegration rank testing under heteroskedasticity (Cavaliere-Rahbek-Taylor) | Wild-bootstrap Johansen tests robust to unconditional/conditional heteroskedasticity, where asymptotic tests over-reject badly. | high | Bootstrap under the restricted rank-r estimate (crucial — the unrestricted bootstrap is invalid); wild-bootstrap the VECM residuals; iterate over ranks sequentially. Cavaliere, Rahbek & Taylor (2012, Econometrica; 2014, ET). No mainstream package has it. Gate: reproduce their reported Monte Carlo sizes. |

**Factor models & FAVAR**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Nonstationary dynamic factor models (Barigozzi-Lippi-Luciani) | Factor models with I(1) factors and cointegration among factors; connects big-data factor analysis with trend/common-cycle macro. | research-grade | Estimation of I(1) factors and their VECM; delicate rank determination at the factor level. Barigozzi, Lippi & Luciani (2021, JoE) and companion papers. Post-v1. Gate: reproduce the authors' replication code output. |

**Large & regularized VARs**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Inference after regularization: de-sparsified lasso VAR and high-dimensional Granger causality | Honest tests (e.g., Granger causality) in high-dimensional VARs via post-double-selection or debiased lasso. | research-grade | Post-double-selection Wald tests per Hecq, Margaritella & Smeekes (2023, JBES); bootstrap for sparse VARs per Krampe, Kreiss & Paparoditis (2021). Easy to get wrong: naive post-lasso standard errors are invalid. Gate: reproduce the HMS replication code. |
| Factor-adjusted sparse VAR / networks (FNETS) | Models the panel as common factors plus a sparse idiosyncratic VAR; produces network estimates robust to pervasive comovement. | research-grade | Two-stage: GDFM-type factor removal, then regularized VAR/precision-matrix estimation on idiosyncratic parts (solvers from the ML module). Barigozzi, Cho & Owens (2024, JBES). Gate: match the R fnets package. |

**Nonlinear & time-varying VARs**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Interacted VAR (IVAR) | VAR coefficients interacted with observed conditioning variables (e.g., policy responses conditioned on debt levels); increasingly common in policy analysis, available in almost no library. | medium | OLS with interaction regressors; state-conditional IRFs evaluated at chosen values of the interaction variable, bands by bootstrap holding the state path fixed vs simulated (document both). Towbin & Weber (2013, JDE); Sá, Towbin & Wieladek (2014). Gate: match the authors' MATLAB IVAR toolbox output. |
| Kernel/rolling frequentist TVP-VAR (Giraitis-Kapetanios-Yates) | Time-varying-parameter VAR by kernel-weighted least squares — the frequentist answer to Primiceri's Bayesian TVP-VAR-SV, without MCMC. | high | Kernel weights over time with bandwidth ≈ H·T^0.5; cross-validated bandwidth; time-varying covariance by kernel-smoothed residual outer products; IRFs at each t with bootstrap bands. Document the frequentist-vs-Bayesian split: Primiceri (2005) with stochastic volatility belongs in the Bayesian module. Giraitis, Kapetanios & Yates (2014, JoE; 2018 extension). Gate: reproduce their replication code. |
| Time-varying Granger causality (recursive evolving windows, Shi-Phillips-Hurn) | Dates when Granger causality switches on/off using sup-Wald sequences over forward/rolling/recursive windows; increasingly cited in applied finance. | medium | Lag-augmented (Toda-Yamamoto) Wald statistics over expanding-window families with bootstrapped critical-value sequences controlling family-wise size. Shi, Phillips & Hurn (2018; 2020, JBES). Gate: match the authors' R/MATLAB code and R tvgc-style implementations. |

**Network & connectedness**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Frequency-domain connectedness (Barunik-Krehlik) | Decomposes DY connectedness into short/medium/long-run frequency bands; state of the art in the spillover literature. | high | Spectral decomposition of the GFEVD via the frequency response of the VAR MA representation, integrated over frequency bands; check that band contributions sum to the aggregate DY measure as an internal invariant. Barunik & Krehlik (2018, J. Financial Econometrics). Gate: match the R frequencyConnectedness package. |
| High-dimensional / elastic-net connectedness and TVP connectedness | Connectedness for hundreds of nodes via regularized VAR estimation, and time-varying connectedness via TVP-VAR filtering rather than rolling windows. | high | Elastic-net VAR then GFEVD per Demirer, Diebold, Liu & Yilmaz (2018, JAE); TVP connectedness via forgetting-factor Kalman VAR per Antonakakis, Chatziantoniou & Gabauer (2020). Gate: match ConnectednessApproach, which implements both. |

**Quantile VARs**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Quantile VAR and quantile IRFs (growth-at-risk machinery) | VAR estimated at conditional quantiles for tail-risk dynamics, quantile impulse responses, and quantile spillovers; core of the growth-at-risk agenda. | high | Equation-by-equation quantile regression on VAR regressors (solver from foundations); quantile IRFs via the pseudo-companion recursion (document the strong "quantile stays fixed along the path" assumption) or simulation from the full quantile process; stress-test scenarios per Chavleishvili & Manganelli (2024, J. Applied Econometrics / ECB QVAR papers). White, Kim & Manganelli (2015, JoE); Montes-Rojas (2019). Gate: match the ECB QVAR replication code and R quantile-connectedness implementations (Ando, Greenwood-Nimmo & Shin 2022 for the spillover variant). |

**Mixed frequency & misc**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Network/spatial VAR with known adjacency (network autoregression) | Parsimonious VARs whose coefficient matrices are structured by a known network/spatial weight matrix (W·y_{t−1} terms); scales to thousands of nodes. | medium | NAR of Zhu, Pan, Li, Liu & Wang (2017, AoS) and grouped/community extensions; estimation is constrained least squares but inference under network asymptotics differs. Distinguish clearly from Diebold-Yilmaz (estimated network) in docs. Gate: match the authors' R code. |

## Frontier watchlist

- **Matrix- and tensor-valued autoregressions** for panels with two-way structure (Chen, Xiao & Yang 2021, JoE) — watch-list item; no implementation commitment yet.
- **LP-vs-VAR equivalence tooling** (Plagborg-Møller & Wolf 2021) — a documentation/diagnostic bridge to the local-projections module, not a new estimator; build jointly with the LP module.

All other entries on the research sweep's frontier list (sup-t bands, BJT moving-block bootstrap, Inoue-Kilian uniform inference, Cavaliere-Rahbek-Taylor bootstrap rank tests, HLag with debiased post-lasso inference, FNETS, frequency-domain/TVP/quantile connectedness, QVAR stress testing, time-varying Granger causality, kernel TVP-VARs, nonstationary DFMs, stacked MF-VARs, network autoregression) already appear in the Tier 3 and Tier 4 tables above.

## Implementation warnings

The "easy to get statistically or numerically wrong" list. Every item below is a known failure mode observed in existing packages or home-rolled code.

1. **Sigma-hat degrees-of-freedom convention** (divide by T vs T−Kp−1) silently differs across statsmodels, R vars, and EViews and propagates into every information-criterion value, IRF band, and FEVD — pick a default, expose both, and print which is in use.
2. **Never compute MA coefficients via eigendecomposition of the companion matrix** (it is non-normal and often defective near repeated roots); iterate the Phi recursion. Never form explicit matrix inverses — use QR/Cholesky solves throughout.
3. **Delta-method IRF/FEVD covariances** involving duplication/elimination matrices and Kronecker products are the classic bug farm: unit-test every analytic gradient against central-difference numerical derivatives on random stable VARs.
4. **Bootstrap hygiene**: recenter residuals; document fixed-vs-resampled initial conditions; decide and document the policy for explosive bootstrap draws (redraw vs keep) because it changes band width; Efron percentile, Hall percentile, and symmetric intervals differ materially at long horizons and must be labeled.
5. **Kilian bias correction requires the stationarity safeguard**: if the corrected coefficient matrix is explosive, shrink the correction until the largest companion root is strictly inside the unit circle — omitting this is a known silent failure in home-rolled code.
6. **iid residual bootstraps are invalid under conditional heteroskedasticity**, and even wild bootstraps are invalid for Sigma-dependent statistics (Cholesky IRFs, FEVD); the Brüggemann-Jentsch-Trenkler MBB is the correct default for financial/monthly data — at minimum, warn when ARCH-LM rejects.
7. **Wild bootstrap must multiply the entire residual vector at time t by one scalar draw** to preserve contemporaneous cross-correlation; per-element draws destroy Sigma and give absurdly narrow bands.
8. **Johansen**: build the eigenproblem symmetrically via Cholesky of S11; use MacKinnon-Haug-Michelis (1999) response surfaces for p-values; offer the Johansen (2002) Bartlett correction; and publish an explicit translation table for the five deterministic cases, because urca/statsmodels/EViews all label them differently.
9. **Alpha and beta are identified only up to rotation**: normalize (Phillips triangular) before reporting standard errors; the switching algorithm for joint alpha/beta restrictions has local optima — use multiple starts and assert LR statistics are non-negative.
10. **Hamilton filter / EM for MS-VAR**: compute in log-space or with per-period scaling to prevent underflow; handle label switching with an explicit regime-ordering constraint; guard against degenerate regimes (one regime absorbing a single outlier drives a variance to zero); always verify EM likelihood monotonicity as a runtime assertion.
11. **Kalman/EM DFM**: use Durbin-Koopman univariate treatment of observations or square-root filtering for time-varying observation dimension with missing data; fix PCA factor sign/scale conventions or results are irreproducible across runs and BLAS backends.
12. **GFEVD rows do not sum to one** — the row-normalization choice (Diebold-Yilmaz vs Lanne-Nyberg) changes connectedness numbers; make it an explicit argument, never a hidden default.
13. **Historical decompositions must satisfy the exact adding-up identity** (shocks + initial conditions + deterministics = data) as a runtime check; packages disagree on baseline definitions and users notice.
14. **Threshold/STVAR grid searches** need trimming (10-15% per regime) and bootstrapped null distributions for sup-tests (nuisance parameters are unidentified under the null — chi-square critical values are wrong); the STVAR transition slope gamma is weakly identified and should default to the calibrated-duration convention with a warning.
15. **Long-horizon IRF inference with the largest companion root near unity**: pointwise Gaussian and standard bootstrap bands undercover badly; detect (root > ~0.97 and horizon large) and steer users to Inoue-Kilian/lag-augmented procedures in the warning text.
16. **Information criteria must be computed on the same effective sample** for all candidate lag lengths, and the penalty normalization (per-observation vs total) must match the documented convention or cross-package comparisons fail.
17. **Parallel bootstrap/Monte Carlo must use counter-based RNG streams** so results are bit-identical across thread counts; and never silently drop failed replications (singular fits, non-converged EM) — report them, since silent dropping biases the bootstrap distribution.
18. **Validate everything against published numbers, not just other libraries**: Lütkepohl (2005) E1 examples, Johansen-Juselius (1990), Bernanke-Boivin-Eliasz (2005), Hansen-Seo (2002), Auerbach-Gorodnichenko (2012), Diebold-Yilmaz (2012), and the GVAR Toolbox — because other libraries share bugs (e.g., table-based Johansen p-values).

## Dependencies and shared infrastructure

### Consumed from foundations

- **Resampling/bootstrap engine** (iid/wild/block/stationary + block-length selection + parallel RNG substreams) — CONSUMED. This module supplies the VAR-specific recursive-design schemes (residual, Kilian double, Gonçalves-Kilian wild, BJT moving-block) as plug-ins on that engine; it needs residual recentering hooks, per-replication failure capture, and the explosive-draw policy switch.
- **Monte Carlo / bootstrap execution engine** (parallel, reproducible, resumable; originally inventoried here) — CONSUMED. Needs counter-based (Philox) RNG for bit-identical results across thread counts, transparent failure reporting (never silent dropping), and streaming quantiles; the target is ≥100x R vars on Kilian double-bootstrap bands.
- **Exogenous-regressor (covariate) contract** — VARX ingestion, dynamic-multiplier computation, and VARX forecasting (which requires future exogenous paths) all flow through the shared aligned, leakage-checked interface with its known-future/scenario/auxiliary-forecast distinction; conditional-forecast scenarios reuse the same scenario-path objects.
- **HAC / long-run variance utilities** (Bartlett/QS/Parzen kernels, Andrews 1991 and Newey-West 1994 bandwidths, Andrews-Monahan prewhitening with the 0.97 eigenvalue cap, fixed-b options; originally inventoried here) — CONSUMED by FM-OLS, Phillips-Ouliaris, robust Wald tests, and connectedness. Every downstream replication hinges on matching kernel/bandwidth conventions, so all of them must be exposed explicitly.
- **Typed IRF result object and generalized-IRF engine** — CONSUMED and extended: this module *contributes* the Koop-Pesaran-Potter simulation GIRF engine (Tier 3) to foundations, built on the shared IRF object, so TVAR/STVAR/MSVAR/IVAR and other modules' nonlinear models all use one engine.
- **Linear-Gaussian state-space engine** (Durbin-Koopman univariate filtering, square-root/simulation smoothers, EM) — CONSUMED by the DFM, VARMA exact ML, conditional forecasting, and missing-data VAR estimation.
- **Factor-model estimation core** (PCA/EM/QML; Bai-Ng 2002, Ahn-Horenstein 2013, Onatski 2010 criteria; originally inventoried here as "static approximate factor models") — CONSUMED. This module needs deterministic sign/scale conventions and the criteria suite; it layers the DFM, FAVAR, and GDFM model classes on top. FRED-MD/McCracken-Ng (2016) validation of the static core lives with foundations; this module re-verifies through its DFM tests.
- **Critical-value engine** (response surfaces + on-demand null simulation) — CONSUMED for MacKinnon (1996/2010), MacKinnon-Haug-Michelis (1999), Kripfganz-Schneider (2020), JMN broken-trend, and Hansen-Seo bootstrap critical values.
- **Fast quantile-regression solver** — CONSUMED by the quantile VAR (Tier 4).
- **Deterministic-terms toolkit, time-index/calendar engine, numerical optimizers** — CONSUMED throughout (the five Johansen cases, break dummies, mixed-frequency alignment, EM/NLS/scoring loops).

### Consumed from other modules

- **ML module**: penalized-regression solvers and time-series cross-validation — consumed by the HLag/BigVAR wrapper (Tier 3), FNETS, elastic-net connectedness, and MF-VAR shrinkage. This module keeps only the VAR-specific penalty structures and wrappers.
- **Identification module**: all structural VAR/VEC identification — restriction and rotation logic, sign restrictions, and the SVEC common-trends (King-Plosser-Stock-Watson; Breitung-Brüggemann-Lütkepohl 2004) permanent/transitory machinery originally inventoried here — lives there, with both frequentist and Bayesian backends. This module supplies the reduced-form VAR/VECM fits and Granger-representation components (alpha_perp, beta_perp, the long-run impact matrix Xi) it needs, and points users there for Cholesky-beyond identification.
- **Forecasting-evaluation module**: density-forecast evaluation, forecast-comparison tests, and combination — this module only *produces* forecasts through the unified forecast object.
- **Nowcasting module**: owns the vintage/release-calendar/news layer, including the Banbura-Modugno (2014, sec. 3) nowcast news decomposition originally inventoried here; this module exposes the Kalman-gain and smoother internals of its DFM so nowcasting can compute per-release news without reimplementing the model.
- **LP module**: everything local-projection; the LP-vs-VAR equivalence bridge (Plagborg-Møller & Wolf 2021) is a joint documentation deliverable.

### Exposed to other modules

- The **single DFM implementation** (two-step and QML/EM, mixed frequency, arbitrary missing data) — consumed by nowcasting.
- **Granger-causality tooling** (standard, Toda-Yamamoto, time-varying) — re-exported by diagnostics.
- **Reduced-form VAR/VECM/VARMA fit objects** with companion-form, MA-representation, and Granger-representation accessors — consumed by the identification module and the LP module's comparison tooling.
- **GFEVD/connectedness kernels** — reused by the network-connectedness variants and available to any module needing order-invariant decompositions.
- All forecasts emitted through the library-wide **unified forecast object**; all stochastic procedures on the **Philox RNG substream contract**; all reference numbers wired into the **golden-value validation harness**.

## Validation gallery

Golden targets this module must reproduce before the corresponding tier ships:

- **Lütkepohl (2005) E1 dataset** (West German investment/income/consumption) — VAR estimation, lag selection, diagnostics, Cholesky IRF/FEVD, and asymptotic bands must match the textbook tables and statsmodels/R vars to machine precision.
- **Johansen & Juselius (1990)** Danish/Finnish money demand — Johansen trace/max-eigenvalue statistics, VECM estimates, and alpha/beta restriction tests must match the published results and urca::ca.jo/blrtest output (with MHM response-surface p-values, not table lookups).
- **Kilian (1998, REStat) Monte Carlo designs** — bootstrap-after-bootstrap IRF bands must reproduce published coverage rates; single-threaded and multi-threaded runs must match bit-for-bit.
- **Bernanke, Boivin & Eliasz (2005, QJE) Figure 2** — FAVAR monetary-policy IRFs on the public BBE dataset, including two-step-pipeline bootstrap bands.
- **Hansen & Seo (2002, JoE) term-structure application** — TVECM estimates and sup-LM bootstrap test matching tsDyn::TVECM and the paper.
- **Auerbach & Gorodnichenko (2012, AEJ:Policy) replication files** — STVAR state-dependent fiscal multipliers via the KPP GIRF engine.
- **Diebold & Yilmaz (2012) spillover tables** — connectedness measures matching the published tables and R ConnectednessApproach, with the normalization convention stated.
- **GVAR Toolbox 2.0 (Smith-Galesi) on the DdPS dataset** — reproducing its GIRFs is the GVAR acceptance test.
- **FRBNY nowcast outputs / statsmodels DynamicFactorMQ** — DFM parameter estimates, smoothed factors, and log-likelihoods must match before the speed claims are made.
- **Lütkepohl-Krätzig (2004) / JMulTi examples** — subset VAR, historical decomposition, Saikkonen-Lütkepohl tests, and (jointly with the identification module) the ch. 4 Canadian labor-market SVEC exercise.
