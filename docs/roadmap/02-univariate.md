# Module 02 — Univariate Models

> Part of the time series econometrics library roadmap. Master plan: [ROADMAP.md](../../ROADMAP.md).

**This module is the library's workhorse layer: univariate conditional-mean models and nonlinear/time-varying dynamics, built on a single exact-MLE-via-state-space estimation path. It delivers the full ARMA/ARIMA/SARIMA stack with R-parity likelihoods, the complete ETS taxonomy, a reliable auto-ARIMA, Bai-Perron structural breaks, Markov-switching, and the threshold/STAR families — the models practitioners fit every day, implemented fast enough to bootstrap and Monte Carlo at scale, and validated against published numbers rather than other packages' defaults.**

## Purpose and scope

Module 02 covers everything a single series' conditional mean can do: linear ARMA-family models and their seasonal, integrated, fractionally integrated, and regression-augmented extensions; exponential smoothing from Holt-Winters through the full Hyndman ETS state-space taxonomy; Harvey-style unobserved-components models; time-varying-parameter regression; structural-break estimation and testing; Markov-switching, threshold, and smooth-transition dynamics; count and duration models; and the forecasting infrastructure (multi-step predictive distributions, a unified simulation API) that every model must expose. The estimation backbone is deliberate: exact Gaussian MLE through the shared linear-Gaussian state-space engine, with CSS, Yule-Walker, Burg, Hannan-Rissanen, and Whittle as fast or specialized alternatives, and a documented likelihood-convention contract so results reconcile with R, statsmodels, and STAMP to the digit.

The users are applied macroeconomists and central-bank staff (SARIMA, regARIMA, UC models, breaks, Markov-switching, TVP), forecasting practitioners (ETS, auto-ARIMA, Theta, TBATS, HAR), and researchers in nonlinear time series (SETAR/STAR, MARX, regime tests). For them this module is the front door to the library: the first fit, the first forecast, the first diagnostic report. Its performance ceiling — how fast one Kalman-filter likelihood evaluates, how cheaply one model refits — determines whether the library's bootstrap, Monte Carlo, and rolling-origin promises are real.

Relative to the rest of the roadmap, this module owns model specification, estimation, and the model-specific parts of inference and forecasting. It consumes the foundations state-space engine (exact diffuse initialization, univariate filtering, simulation smoother, EM), the bootstrap engine, the critical-value engine, the innovation-distribution zoo, and the deterministic-terms/calendar toolkit; it points to diagnostics for STL/MSTL, the HP/band-pass/Hamilton/Beveridge-Nelson filter suite, and the X-13ARIMA-SEATS wrapper; and it hands its fitted models to forecasting-evaluation for comparison tests and to the multivariate and volatility modules as building blocks.

## Where existing tools fall short

- **Speed and scale.** statsmodels' Kalman-filter-based SARIMAX/UC/ETS is Cython-assisted but still slow for Monte Carlo and bootstrap work; every mainstream tool is single-threaded per fit, with no batch-estimation API, no steady-state or Chandrasekhar acceleration, and no reproducible parallel-RNG story.
- **No maintained Python auto-ARIMA.** pmdarima is effectively unmaintained and breaks with modern numpy; statsforecast is fast but diverges from exact-MLE R results and has limited model coverage; R forecast/fable remain the gold standard but are R-only.
- **Bai-Perron multiple structural breaks do not exist in Python.** R strucchange and mbreaks cover them, but with confusing heteroskedasticity/autocorrelation option handling relative to the original paper's configurations.
- **Threshold and smooth-transition models have no credible Python home.** R tsDyn is aging, its inference options are thin, and STAR estimation is fragile everywhere, with no packaged Teräsvirta specification cycle.
- **Markov-switching is unreliable everywhere.** statsmodels MarkovAutoregression is fragile (initialization, EM, boundary variances), lacks Markov-switching state space (the Kim filter), has limited TVTP support — and no mainstream package anywhere ships tests for the number of regimes (Hansen 1992; Carrasco-Hu-Ploberger 2014).
- **Long memory is scattered.** No exact Sowell ARFIMA MLE in Python; R arfima and fracdiff disagree with each other on real series; GPH, local Whittle, and ELW estimators live in abandoned packages with inconsistent bandwidth conventions.
- **Unobserved components stop at the basics.** statsmodels lacks Harvey-Trimbur higher-order cycles, correlated-component UC (Morley-Nelson-Zivot), and STAMP-quality auxiliary-residual diagnostics.
- **Outlier detection (Chen-Liu AO/LS/TC/IO)** exists only via R tsoutliers or the Census X-13 binary; nothing native, fast, and scriptable in Python.
- **Count and duration time series (INAR, INGARCH, GLARMA, ACD)** have no credible Python implementations; R tscount is good but slow, ACDm is barely maintained, and none produce integer-coherent forecast distributions by default.
- **TVP tooling is fragmented.** Stock-Watson median-unbiased variance estimation, DMA/DMS, and shrinkage-prior TVP (shrinkTVP-class) exist only as author MATLAB code or single-purpose R packages, never unified with the classical Kalman-filter route.
- **Prediction intervals are systematically too narrow.** Across ecosystems, intervals ignore parameter uncertainty, and nonlinear models (MS/TAR/STAR) are routinely given naive Gaussian or iterated plug-in forecasts that are provably biased beyond one step.
- **Likelihood-convention chaos.** Diffuse initialization, likelihood constants, and concentrated-vs-full conventions differ silently across R, statsmodels, STAMP, and EViews; no package publishes its conventions precisely, making cross-validation of results needlessly painful.
- **Nobody teaches which model when.** R's fpp3 book is the closest but is method-limited and R-bound; no library connects diagnostics (e.g., a rejected linearity test) to a recommended model family.

## Inventory

Source priorities map to tiers: core → Tier 1, standard → Tier 2, advanced → Tier 3, frontier → Tier 4. Items reassigned by the master-plan ownership map appear under [Dependencies and shared infrastructure](#dependencies-and-shared-infrastructure); items demoted by scope rulings appear in the [Contrib tier](#contrib-tier) below.

### Tier 1 — Core (v1-blocking)

**ARMA family**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| AR(p) autoregression | Workhorse linear persistence model; used for benchmarks, prewhitening, lag-augmented inference, and as a building block everywhere. | Low | OLS, Yule-Walker, Burg, and exact MLE (state space); companion-form utilities (roots, stationarity check, IRF/MA(∞) weights); Andrews-Chen (1994) approximately median-unbiased and Kilian (1998) bootstrap-after-bootstrap bias corrections. Validate vs. statsmodels AutoReg and R `ar()`; document that R's Yule-Walker default is biased toward stationarity near unit roots. |
| MA(q) and ARMA(p,q) | General linear short-memory model; the canonical Box-Jenkins object. | Medium | Exact Gaussian MLE via the Harvey/Jones state-space form; enforce stationarity/invertibility with the PACF reparameterization (Monahan 1984, building on Jones 1980) — the admissible region is not a box, so naive coefficient bounds fail for p,q > 1. Starting values via Hannan-Rissanen. Validate log-likelihood and estimates against R `arima()` and statsmodels SARIMAX on Box-Jenkins (1976) Series A–J. |
| ARIMA(p,d,q) | Integrated ARMA for nonstationary series; the default univariate forecasting model in practice. | Medium | Two implementations: difference-then-fit (CSS/exact on differenced data) and levels state space with exact diffuse initialization (handles missing data in levels correctly). Forecasts must integrate back with correct cumulative variance; warn that information criteria are not comparable across different d. Validate against R `arima()` on classic datasets (e.g., Series A) to 6+ digits. |
| SARIMA (p,d,q)(P,D,Q)_s | Multiplicative seasonal ARIMA; standard for monthly/quarterly macro and the airline-model workhorse. | Medium | Polynomial convolution of seasonal and nonseasonal lag polynomials; keep the transition matrix sparse (state dimension p+sP+d+sD can be large — exploit structure, use univariate filtering). Canonical validation: airline model ARIMA(0,1,1)(0,1,1)_12 on log AirPassengers, matching Box-Jenkins (1976) and R `arima()` θ estimates and log-likelihood exactly. |
| Regression with ARMA errors / ARIMAX (regARIMA) | Deterministic or stochastic regressors with ARMA disturbances; the backbone of intervention analysis, calendar adjustment, and X-13-style pre-adjustment. | Medium | Implement the regression-with-ARMA-errors parameterization (as in R `arima(xreg=)` and X-13 regARIMA), NOT the transfer-function ARMAX with lagged y — and document the difference loudly (statsmodels-vs-R confusion here is a perennial user trap). GLS-type concentrated estimation of betas inside the Kalman filter. Validate vs. R `arima(xreg=)` and Census X-13 regARIMA output. |

**Estimation algorithms**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Conditional sum of squares (CSS) | Fast approximate ARMA estimation; standard for large T and as a warm start for exact MLE. | Low | Match R `method="CSS"` and statsmodels conventions exactly (treatment of initial residuals, scaling of the objective). Document that CSS and exact MLE differ noticeably in small samples and near the unit circle — a top source of "your package disagrees with R" bug reports. |
| Yule-Walker and Burg estimation | Moment/lattice-based AR estimation; Burg is preferred for spectral work and near-unit-root series. | Low | Levinson-Durbin recursion for Yule-Walker; Burg via lattice recursion. Yule-Walker guarantees stationarity but is badly biased near unit roots; Burg much less so (Percival-Walden 1993). Validate vs. R `ar.yw`/`ar.burg`. |
| Hannan-Rissanen three-stage procedure | Long-AR regression-based ARMA estimation; the standard starting-value generator for MLE. | Medium | Hannan-Rissanen (1982): long AR (order by AIC), regress on lagged y and lagged residuals, then third-stage bias correction — without it, estimates are inconsistent as starting values for near-cancellation models. Also implement the innovations algorithm (Brockwell-Davis 1991) for pure-MA starts. |
| Stationarity/invertibility reparameterization layer | Bijective transform between unconstrained optimizer space and stationary/invertible ARMA coefficients. | Low | Monahan (1984) PACF transform (tanh of unconstrained parameters → partial autocorrelations → coefficients), implemented with its exact Jacobian for analytic gradients. This single utility prevents most ARMA optimizer failures; make it a documented public API. |

**State-space model building** (on top of the foundations engine)

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| User-defined state-space model API | Lets researchers specify arbitrary (time-varying) system matrices and get filtering/smoothing/MLE/Bayes for free — the extensibility story of the library. | Medium | Mimic the good parts of statsmodels MLEModel and the KFAS formula interface, but with compiled callbacks or matrix-buffer specification to avoid Python-callback overhead in the hot loop; time-varying matrices stored as strided arrays. This API decision constrains everything — design early, jointly with the foundations engine team. |
| Smoothed auxiliary-residual diagnostics | Standardized smoothed disturbances used to flag additive outliers (measurement) vs. level shifts (state); Harvey-Koopman (1992). | Medium | Built on the Koopman (1993) disturbance smoother exposed by the foundations engine. Validate vs. KFAS/STAMP on the Nile series (documented 1899 level break). |

**Exponential smoothing**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| SES, Holt, Holt-Winters (additive/multiplicative) | The methods practitioners actually use daily; must match textbook and R behavior. | Low | Offer both heuristic initialization (matching R `HoltWinters`) and full-optimization initialization (matching `forecast::ets`) — the differences between these are a top user-confusion source; document explicitly. |
| Full ETS state-space taxonomy | All 30 error/trend/seasonal combinations with proper likelihood and forecast distributions; the theoretically grounded exponential smoothing. | High | Hyndman-Koehler-Snyder-Grose (2002); Hyndman et al. (2008). Pitfalls: the admissible parameter region is larger than the [0,1] box (ch. 10 eigenvalue conditions); multiplicative-error models are undefined for nonpositive data; forecast variances are class-specific (Class 1 analytic, Class 2 approximations, Class 3 simulate). Validate vs. `forecast::ets` on M3 (published MASE/sMAPE per method) and unit-test each class's variance formula. |
| AutoETS model selection | Automatic ETS selection by AICc; the standard automatic benchmark alongside auto-ARIMA. | Medium | AICc over the admissible taxonomy with damped variants; restrict unstable combinations (multiplicative trend off by default, matching `forecast::ets`). Validate aggregate M3 accuracy vs. published Hyndman results. |

**Unobserved components**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Local level and local linear trend | Harvey's basic structural blocks; the pedagogical and practical entry to state space. | Medium | Include smooth-trend (integrated random walk) and damped-slope variants; signal-to-noise ratio q parameterization for teaching. Canonical validation: Nile data (Durbin-Koopman 2012) exact variance estimates; UK seatbelt data for the full BSM. |
| Stochastic seasonality (dummy and trigonometric) | Time-varying seasonal component within UC models. | Medium | Trigonometric form with common or frequency-specific variances. Validate vs. KFAS/STAMP on the UK seatbelt dataset (Harvey-Durbin 1986). |
| Basic Structural Model with regressors and interventions | Complete Harvey-style UC model with explanatory variables and break dummies; the STAMP experience in Python. | Medium | Harvey (1989); Durbin-Koopman (2012). Auxiliary-residual diagnostics for automatic detection of level shifts/outliers integrate here. Validate the seatbelt-law intervention effect vs. published Harvey-Durbin (1986) estimates. |

**Time-varying parameters**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| TVP regression via Kalman filter | The basic drifting-coefficients regression (random-walk coefficients); time-varying Phillips curves, betas, pass-through. | Medium | MLE of variance ratios on the log scale. The central pitfall is pile-up: the MLE of a small state variance hits zero with high probability — detect boundary solutions and point users to Stock-Watson (1998) or Bayesian options. Validate vs. statsmodels custom SSM and R dlm. |

**Structural breaks and stability**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Bai-Perron multiple structural breaks | Estimation and testing of multiple unknown break dates in regression/AR models; the single most-requested break tool. | High | Bai-Perron (1998; 2003). Dynamic programming over precomputed segment SSRs (numerically stable recursive-residual/QR updating, never naive rank-one accumulation); trimming ε; supF(k), UDmax/WDmax, sequential supF(l+1|l); Bai (1997) asymmetric CIs for break dates; heteroskedasticity/autocorrelation options change critical values (ship Bai-Perron response-surface CVs via the critical-value engine). Replicate their US ex-post real interest rate example exactly (breaks 1972:3, 1980:3) and cross-check vs. R strucchange and mbreaks. Python has nothing for this. |

**Regime switching**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Markov-switching AR (Hamilton model) | Regime-dependent mean/variance/dynamics; the canonical recession-dating and nonlinear-macro model. | High | Hamilton (1989). Hamilton filter in the log domain (underflow otherwise); Kim (1994) smoother (classic off-by-one bugs); EM (Hamilton 1990) for warm starts, then quasi-Newton; multistart mandatory (multimodal likelihood); variance floors to prevent unbounded likelihood as a regime variance → 0; label-switching normalization (e.g., μ₁ < μ₂). Validate: replicate Hamilton's 1951–84 GNP estimates (μ₀ ≈ 1.16, μ₁ ≈ −0.36, p₀₀ ≈ 0.90, p₁₁ ≈ 0.75) and the smoothed-probabilities figure. statsmodels' version is fragile — reliability here wins users. |

**Missing data**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Missing-data handling in estimation and smoothing | Exact likelihood with arbitrary missing patterns plus model-based interpolation; a first-class feature, not an afterthought. | Medium | Skip the Kalman measurement update at missing points (do not zero-weight); the log-likelihood counts only observed terms. For ARIMA with gaps, estimate in levels state space with exact diffuse initialization — never difference through NaNs. Smoothed-state interpolation with proper MSEs. Validate vs. the Durbin-Koopman ch. 4 missing-data Nile example and KFAS. |

**Model selection and diagnostics**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Auto-ARIMA (Hyndman-Khandakar stepwise) | Automatic ARIMA order selection; the single most used function in R forecast — table stakes for adoption. | High | Hyndman-Khandakar (2008): choose d by successive KPSS and D by seasonal strength or OCSB; then stepwise search over (p,q,P,Q) by AICc within fixed (d,D) — IC are not comparable across differencing orders. Must survive fit failures gracefully, reject roots near the unit circle, and offer exhaustive parallel search (a speed showcase). Validate: near-identical model choices to `forecast::auto.arima` across M3; beat pmdarima (unmaintained, numpy-incompatible) on reliability. |
| Information criteria and likelihood conventions | AIC/AICc/BIC/HQ with exactly documented likelihood constants and effective sample sizes. | Low | Publish the exact log-likelihood definition per model (constants included, diffuse terms excluded, concentrated vs. full) so users can reconcile with R/statsmodels/STAMP to the digit. Cross-package likelihood-convention mismatch is the top source of spurious "bug" reports in this space. |
| Residual diagnostics suite | Ljung-Box/Box-Pierce with df corrections, ACF/PACF with correct bands, EACF, normality and ARCH-LM screens, standardized-residual plots. | Low | Ljung-Box df must subtract fitted ARMA parameters (and it is invalid on ARMAX residuals without adjustment); ACF bands under the estimated-parameter null differ from 1/√T (Bartlett formula option). EACF (Tsay-Tiao 1984) for order identification. Return a structured diagnostics report tied to docs on remedial actions. |

**Forecasting infrastructure**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Multi-step forecast distributions (analytic + simulation) | Correct point forecasts and predictive distributions for every model, including nonlinear ones. | Medium | Analytic MSE recursions for linear/state-space models; for MS/TAR/STAR, iterated plug-in of conditional means is BIASED beyond h = 1 — multi-step must integrate over regime/state paths by simulation or exact Chapman-Kolmogorov (MS case). Make the simulation path the well-tested default for nonlinear classes; document with a worked bias example. |
| Unified simulation engine | One `.simulate()` API across all models for Monte Carlo studies, bootstrap, and predictive simulation. | Medium | Counter-based RNG (Philox, consumed from foundations) so parallel streams are reproducible independent of thread count; burn-in defaults with stationary-distribution initialization where available (exact for AR via a Gaussian stationary draw); allocate-once buffers for repeated simulation. Every other domain's bootstrap tools will call this — spec it early. |

### Tier 2 — Standard (expected of a serious library)

**Long memory**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Fast fractional differencing operator | Utility to apply (1−L)^d; needed for ARFIMA, fractional cointegration, and long-memory simulation. | Low | FFT-based circular convolution (Jensen-Nielsen 2014) is O(T log T) and exact; naive truncated filters are O(T²) and inaccurate. Also provide fractional-Gaussian-noise and ARFIMA simulators via Davies-Harte circulant embedding. |
| GPH log-periodogram estimator of d | Semiparametric memory estimation from low frequencies; quick diagnostic for long memory. | Low | Geweke-Porter-Hudak (1983); include the bias-reduced variant (Andrews-Guggenberger 2003). Bandwidth m dominates results — provide defaults (T^0.5, T^0.65) plus a sensitivity plot. Validate vs. R `fracdiff::fdGPH` and published Monte Carlo. |
| HAR model (heterogeneous autoregression) | Corsi (2009) daily/weekly/monthly aggregated AR; the de facto standard for realized-volatility conditional-mean forecasting. | Low | OLS with overlapping-average regressors; HAC/Newey-West standard errors (via the foundations HAC policy), log and sqrt variants, HAR-with-jumps and semi-variance extensions (Patton-Sheppard 2015). Validate vs. R HARModel/highfrequency. Cheap to implement, high user demand — good early win. |

**Exponential smoothing and decomposition**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Theta method and optimized/dynamic Theta | M3 winner; equivalent to SES with drift (Hyndman-Billah 2003); cheap and strong benchmark. | Low | Implement classic Theta, optimized Theta, and dynamic optimized Theta (Fiorucci et al. 2016). Validate vs. the forecTheta R package and published M3 numbers. |

**Unobserved components**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Stochastic cycles incl. higher-order (Harvey-Trimbur) | Model-based business-cycle extraction with smooth band-pass properties; central-bank staple. | Medium | Harvey (1989) cycle; Harvey-Trimbur (2003) nth-order cycles. Constrain frequency λ to (2π/period_max, 2π/period_min) and damping ρ to (0,1) via transforms. Validate vs. STAMP and Trimbur's published US GDP cycle estimates. Nobody in Python ships higher-order cycles. |

**Structural breaks and stability**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Andrews sup-W/QLR and Andrews-Ploberger exp/ave tests | Single unknown-break tests; standard robustness-table entries. | Medium | Andrews (1993, with the 2003 corrected critical values — use the corrected tables); Andrews-Ploberger (1994); Hansen (1997) approximate asymptotic p-value functions. Validate vs. `strucchange::Fstats` and Hansen's published p-value coefficients. |
| CUSUM/MOSUM and generalized fluctuation tests | Recursive/moving-sum stability diagnostics; also the basis for monitoring. | Medium | Brown-Durbin-Evans (1975); OLS-based CUSUM (Ploberger-Krämer 1992); the strucchange empirical-fluctuation-process framework (Zeileis et al. 2002) is a clean design to emulate. Include the real-time monitoring variant (Chu-Stinchcombe-White 1996). |
| Nyblom-Hansen parameter stability tests | LM tests of constancy against random-walk drift; cheap default diagnostic after any regression/ARMA fit. | Low | Nyblom (1989); Hansen (1992 JPE) individual and joint versions. Trivial to compute from scores; report with every fit as part of the "which model when" documentation story. |

**Regime switching**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| MS regression and MS with switching variances | Switching intercepts/coefficients/variances in general regressions; covers "moving intercept" demands. | Medium | Same filter infrastructure as MS-AR; expose which parameters switch via a spec object. Validate vs. MSwM (R) and Kim-Nelson (1999) examples. |

**Threshold and smooth transition**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| SETAR/TAR estimation with threshold inference | Self-exciting threshold AR; regime switches when a lagged value crosses a threshold. Standard nonlinear benchmark (sunspots, unemployment asymmetry). | Medium | Tong-Lim (1980). Concentrated LS: grid over threshold candidates = order statistics of the threshold variable with 10–15% trimming and over delay d; per-candidate OLS via updating. Hansen (1997; 2000) likelihood-ratio-based confidence sets for the threshold (nonstandard, asymmetric). Validate vs. Hansen's published US GNP/unemployment applications and R tsDyn. |
| Threshold effect tests (Hansen sup-test, Tsay test) | Testing linearity against threshold alternatives with unidentified-nuisance-parameter corrections. | Medium | Hansen (1996) fixed-regressor/wild bootstrap of the sup-LM/sup-F (embarrassingly parallel — showcase speed); Tsay (1989) arranged-autoregression F-test as a cheap screen. Never report naive chi-squared p-values. |
| STAR models (LSTAR/ESTAR) with the Teräsvirta modeling cycle | Smooth-transition AR; the standard for smooth regime change (real exchange rates ESTAR, business cycles LSTAR). | High | Teräsvirta (1994) full cycle: LM linearity tests against STAR, LSTAR-vs-ESTAR selection via nested F-tests on the Taylor expansion, then NLS. Pitfalls: γ is poorly identified when large — standardize the transition variable by its sample SD and estimate log(γ); grid-search (γ, c) before NLS; a near-flat likelihood in the γ direction means report profile CIs. ESTAR has additional identification pathologies near unit roots (relevant to KSS-test users). Validate vs. tsDyn and Teräsvirta's published applications. |
| Nonlinearity test battery | Portmanteau of linearity tests users expect before fitting nonlinear models: BDS, Keenan, Tsay, RESET, Teräsvirta NN, White NN. | Medium | BDS (Brock et al. 1996) needs the O(T²) correlation integral in compiled code with the exact small-sample epsilon conventions of the original C code (R tseries and fNonlinear disagree; match Kanzler 1999 benchmark values). Teräsvirta et al. (1993) V23 test. Present as a coherent report object teaching which rejection points to which model family. |

**Deterministic terms, intervention, and outliers**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Intervention analysis (Box-Tiao) | Pulse/step/ramp effects with dynamic (rational-lag) responses; policy-evaluation classic. | Medium | Box-Tiao (1975). Implement as transfer-function regressors within regARIMA. Validate vs. their Los Angeles ozone application and R tfarima/SCA output. |
| Automatic outlier detection (AO/LS/TC/IO, Chen-Liu) | Iterative detection and joint estimation of additive outliers, level shifts, transitory changes, and innovational outliers in ARIMA models. | High | Chen-Liu (1993); Chang-Tiao-Chen (1988). Pitfalls: masking/swamping (require a joint re-estimation pass); the critical value must grow with T (X-13 default rule); LS vs. AO are indistinguishable at the sample end; IO is fragile in nonstationary models (TRAMO disables it by default — follow that). Validate vs. R tsoutliers and TRAMO on published examples. Also expose the state-space auxiliary-residual route (Harvey-Koopman 1992) as an alternative. |

### Tier 3 — Advanced (differentiators)

**ARMA family**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Transfer function / dynamic regression (Box-Jenkins rational lags) | Rational distributed-lag response of y to inputs x with ARMA noise; intervention dynamics and leading-indicator models. | High | Identification via prewhitening and the cross-correlation function; estimation by exact MLE in state-space form. Numerically fragile when denominator roots are near the unit circle. Canonical benchmark: Box-Jenkins gas furnace data; validate vs. SCA or the R tfarima package. |
| Periodic AR/ARMA (PAR) | Season-dependent AR coefficients; seasonal macro series whose dynamics differ by quarter/month. | Medium | Season-by-season OLS or the stacked VAR-of-seasons representation; test PAR vs. constant-coefficient AR (Franses-Paap 2004). Watch the periodic stationarity condition (product of companion matrices). Validate vs. R partsm/pear. |

**Long memory**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| ARFIMA exact MLE (Sowell) | Fractionally integrated ARMA for long-memory series (inflation, realized volatility, interest-rate spreads). | High | Sowell (1992) exact autocovariances involve hypergeometric functions with severe cancellation as d → 0.5; use stable recursions/high precision, then Durbin-Levinson or Trench for the Gaussian likelihood (O(T²)); offer Whittle as the fast alternative. Validate vs. the R arfima package and Doornik-Ooms (2003) Ox results — fracdiff and arfima disagree on some series, so validate against published estimates, not package output alone. |
| Local Whittle and Exact Local Whittle (ELW) | Efficient semiparametric d estimation valid over a wide d range; the modern default for memory estimation. | Medium | Robinson (1995) LW; Shimotsu-Phillips (2005) ELW valid for nonstationary d; Shimotsu (2010) two-step feasible ELW with unknown mean/trend. The objective can have boundary issues — optimize d on a grid, then refine. Validate vs. Shimotsu's published MATLAB results (extended Nelson-Plosser application). |

**Estimation algorithms**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Whittle (frequency-domain) likelihood | O(T log T) approximate likelihood; the fast path for very long series and ARFIMA. | Medium | Periodogram via FFT; sum over Fourier frequencies excluding zero (handles unknown mean). Document small-T bias relative to exact MLE. Good target for SIMD/parallel speed claims; validate asymptotic equivalence to exact MLE numerically. |
| Bias-corrected AR estimation (median-unbiased, bootstrap) | Corrects downward persistence bias; critical for IRFs and half-life estimates in small macro samples. | Medium | Andrews (1993) exactly median-unbiased for AR(1) with trend; Andrews-Chen (1994) for AR(p); Kilian (1998) bootstrap-after-bootstrap with stationarity adjustment. Validate vs. published Andrews (1993) tables. Nobody in Python ships this; heavily used in the PPP/half-life literatures. |
| Bayesian estimation for ARIMA/UC/TVP | Posterior inference with priors; needed by central-bank users and for models where MLE piles up at boundaries. | High | Gibbs samplers built on the foundations simulation smoother (Frühwirth-Schnatter 1994; Durbin-Koopman 2002); NUTS via autodiff of the KF log-likelihood for ARMA. Interoperate with ArviZ for diagnostics; coordinate priors/samplers with the bayesian module. Validate the local-level Nile posterior vs. published results and R bssm. |

**State-space infrastructure extensions**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Analytic score for state-space likelihoods | Exact gradients of the KF log-likelihood; makes MLE fast/reliable and enables NUTS. | High | Koopman-Shephard (1992) score via smoothed moments, or reverse-mode autodiff through the filter (Rust: enzyme/num-dual, or a hand-written adjoint). Finite differences near variance boundaries produce spurious convergence — analytic gradients are a genuine differentiator vs. statsmodels. Coordinate with the foundations engine. |
| Extended/iterated EKF hooks | Covers Box-Cox/exponential measurement links and nonlinear ETS classes without full particle filtering. | Medium | EKF/UKF as approximations with clear warnings; full particle-filter machinery belongs to the volatility/nonlinear-filtering domain — keep an interface boundary. |
| Kim filter for Markov-switching state space | Approximate filter for regime-switching state-space models (MS unobserved components, MS dynamic-factor precursors). | High | Kim (1994) collapsing filter + Kim smoother. Classic bugs: mixed indices in the collapse step and wrong initial regime probabilities (use the ergodic distribution). Validate against Kim-Nelson (1999) book examples (their GAUSS results are published). |

**Exponential smoothing and decomposition**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| TBATS / BATS | Trigonometric seasonality + Box-Cox + ARMA errors for multiple/non-integer seasonalities (daily data with weekly and annual cycles). | High | De Livera-Hyndman-Snyder (2011). Large state dimension — needs the fast KF core; Python's existing tbats package is prohibitively slow, so a compiled TBATS is a visible differentiator. Validate vs. `forecast::tbats` forecasts and log-likelihood. |

**Unobserved components**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| UC with correlated components (UC-UR / Morley-Nelson-Zivot) | Allows trend-cycle innovation correlation; reconciles UC and Beveridge-Nelson decompositions of US GDP. | Medium | Morley-Nelson-Zivot (2003). Identification is delicate (requires an AR(2)+ cycle); replicate their US GDP correlation estimate (≈ −0.9). A well-known result users will test on day one. |

**Time-varying parameters**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Stock-Watson median-unbiased TVP variance estimation | Solves the pile-up problem for the coefficient-drift variance; canonical in trend-inflation work. | Medium | Stock-Watson (1998) mapping from Nyblom/qLL-type statistics to λ; ship their lookup table. Replicate their G7 trend-growth application. Absent from every mainstream library. |
| Dynamic model averaging/selection (DMA/DMS) | Online averaging over TVP models with forgetting factors; popular for inflation forecasting. | Medium | Raftery-Karny-Ettler (2010); Koop-Korobilis (2012). Cheap recursions (no MCMC); forgetting-factor grid. Validate vs. Koop-Korobilis MATLAB results on US inflation. R eDMA exists; Python has nothing maintained. |
| Score-driven (GAS/DCS) location and trend models | Observation-driven time-varying level/mean with heavy tails; robust filters that automatically downweight outliers. | High | Creal-Koopman-Lucas (2013); Harvey (2013) DCS-t location; Harvey-Luati (2014) robust local level. Likelihood is closed-form (fast); pitfalls are scaling choices (inverse Fisher vs. sqrt) and filter invertibility (Blasques et al. 2018 conditions). Validate vs. the GAS package (R) and Harvey-Luati published examples. |

**Structural breaks and stability**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Elliott-Müller qLL test | Efficient test against very general persistent parameter variation; more powerful than supF against smooth change. | Medium | Elliott-Müller (2006). Simple to compute (GLS-detrending-style recursions), but critical values must be shipped (critical-value engine). Barely exists outside the authors' code. |

**Regime switching**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Time-varying transition probability MS (TVTP) | Transition probabilities driven by covariates (spreads, leading indicators); standard in applied business-cycle work. | Medium | Filardo (1994); Diebold-Lee-Weinbach (1994). Logistic link on transition rows; EM becomes weighted logit M-steps. Validate vs. Filardo's published IP results. |
| Tests for the number of regimes | Deciding 1 vs. 2 (or k vs. k+1) regimes; nonstandard because nuisance parameters vanish under the null (Davies problem). | Research-grade | Hansen (1992) standardized LR bound (simulation-heavy); Carrasco-Hu-Ploberger (2014) information-matrix-type optimal test (much cheaper — ship as default); Qu-Zhuo (2021) LR asymptotics. Never use chi-squared critical values — enforce this in the API. Validate vs. published CHP MATLAB output. |
| Bayesian MS estimation with permutation sampling | Posterior inference for MS models; sidesteps boundary/multimodality MLE pathologies. | High | Frühwirth-Schnatter (2006): forward-filter backward-sample for states, random permutation to handle label switching, post-hoc identification. Validate marginal likelihoods vs. published bridge-sampling results. |

**Threshold and smooth transition**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Multiple-regime and time-varying STAR (MRSTAR, TV-STAR) | Combines smooth regime switching with structural change; asymmetry and instability jointly. | High | van Dijk-Teräsvirta-Franses (2002) is the roadmap. Estimation compounds all STAR pathologies; ship with strong defaults and diagnostics. |

**Count and duration models**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| INAR(p) integer autoregression | Thinning-operator AR for low counts (crime, epidemiology, transactions); preserves integer support. | Medium | Al-Osh & Alzaid (1987); Du-Li (1991). CLS and exact MLE via convolution recursions (dynamic programming over the count support); forecasts must be integer-coherent distributions, not Gaussian intervals. Validate vs. published Westgren gold-particle series results. |
| INGARCH / autoregressive conditional Poisson and NegBin | GARCH-analogue dynamics for conditional count means; the modern default for count time series. | Medium | Ferland-Latour-Oraichi (2006); Fokianos-Rahbek-Tjøstheim (2009) for asymptotics; the log-linear variant handles covariates and negative dependence. Analytic score/information — fast MLE. Validate vs. R tscount (JSS 2017), including its published campylobacter/E. coli examples. |
| GLARMA models | GLM with ARMA-type serial dependence in a latent linear predictor; flexible count/binary time series regression. | Medium | Davis-Dunsmuir-Streett (2003). Newton-Raphson with Pearson or score residual recursions; watch nonstationarity of the recursion for some parameterizations. Validate vs. the R glarma package (JSS). |
| ACD models (Engle-Russell) and log-ACD | Autoregressive conditional duration for irregularly spaced transaction data; microstructure staple. | Medium | Engle-Russell (1998); log-ACD (Bauwens-Giot 2000) avoids positivity constraints. Exponential QMLE is consistent under misspecification (mirrors GARCH QMLE theory); require diurnal-seasonality adjustment (cubic spline on time of day) as a preprocessing step in the API. Validate vs. ACDm (R) and Engle-Russell's IBM results. Coordinate with the volatility domain (shared recursion machinery). |

### Tier 4 — Frontier (research-grade, each gated on a named validation target)

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Robust exponential smoothing | Outlier-resistant smoothing for contaminated operational data. | Medium | Gelper-Fried-Croux (2010) Huberized updates; alternatively the score-driven t-error local level (Harvey-Luati 2014), which is more principled — offer both, cross-referenced. Gate: reproduce Gelper-Fried-Croux (2010) simulation results. |
| l1 trend filtering | Piecewise-linear trend estimation; sparse alternative to HP producing interpretable kink dates. | Medium | Kim-Koh-Boyd (2009); specialized ADMM or primal-dual interior point on banded systems; cross-validated λ. Rarely present in econometrics libraries. Gate: match Kim-Koh-Boyd (2009) reference solutions (cvxpy cross-check). |
| Bayesian TVP with global-local shrinkage (shrinkTVP-style) | Modern TVP estimation that shrinks non-varying coefficients to constancy; the 2018+ standard in applied Bayesian macro. | Research-grade | Bitto-Frühwirth-Schnatter (2019) double gamma; Cadonna et al. (2020) triple gamma; dynamic horseshoe (Kowal-Matteson-Ruppert 2019). Requires ancillarity-sufficiency interweaving (ASIS) for the sqrt-variance parameterization to mix. Gate: match R shrinkTVP (JSS 2021) posteriors. |
| Kernel/local-constant deterministic TVP-AR | Nonparametric smoothly varying AR coefficients without stochastic-process assumptions. | Medium | Giraitis-Kapetanios-Yates (2014). Kernel-weighted least squares with bandwidth ~ T^0.5; bootstrap bands. Almost no packaged implementation anywhere. Gate: reproduce Giraitis-Kapetanios-Yates (2014) empirical results. |
| Continuous-record break inference (Casini-Perron) | Better finite-sample confidence sets for break dates, especially small breaks where Bai CIs undercover. | Research-grade | Casini-Perron (2021+). Feasible density of the break-date estimator under continuous-record asymptotics. A genuine 2020s differentiator. Gate: validate vs. their R/MATLAB replication files. |
| Mixed causal-noncausal AR (MAR/MARX) | AR models with roots inside the unit circle interpreted as forward-looking components; captures locally explosive bubble episodes. | Research-grade | Lanne-Saikkonen (2011); Gouriéroux-Zakoïan (2017); Hecq-Lieb-Telg MARX (2016). Requires non-Gaussian MLE (t or alpha-stable errors — the Gaussian likelihood cannot identify root location); forecasting via simulation (Hecq-Voisin 2021). Strong appeal to bubble/commodity researchers. Gate: validate vs. the R MARX package. |
| Functional-coefficient AR (FAR/FCAR) | AR coefficients as nonparametric functions of a state variable; nests TAR/STAR smoothly. | High | Chen-Tsay (1993); Cai-Fan-Yao (2000). Local-linear estimation with bandwidth by modified multifold CV. Position as an exploratory tool for choosing between TAR and STAR. Gate: reproduce Cai-Fan-Yao (2000) examples. |
| Zero-inflated and hurdle dynamic count models | Counts with excess zeros (operational risk, rare events). | High | ZI-INGARCH variants (Zhu 2012). EM over the mixture; identifiability of inflation vs. low mean is delicate. Ship after base INGARCH is validated. Gate: reproduce Zhu (2012) applications. |
| Hawkes self-exciting point processes (univariate) | Event-arrival clustering (defaults, trades, extremes); increasingly expected by finance users. | High | Ogata (1988) thinning simulation; the exponential-kernel likelihood has an O(T) recursion (Ozaki 1979); power-law kernels need approximation. Compensator-based residual diagnostics (time-rescaling theorem). Coordinate scope with the volatility/microstructure domain. Gate: match published Ozaki/Ogata example likelihoods. |
| Continuous-time ARMA (CARMA) for irregular sampling | Models for genuinely unequally spaced observations (some finance/commodities applications). | Research-grade | Brockwell (2001). State space with matrix exponentials between observation times; ill-conditioning of the exponential map is the main trap. Low demand; late-roadmap. Gate: reproduce Brockwell (2001) reference fits. |

### Contrib tier

Demoted by master-plan scope rulings; kept as documented, community-maintainable extras.

- **Bilinear models** — Granger-Andersen (1978), Subba Rao (1981); minimal BL(p,q,m,k) via CSS; rarely forecast-competitive, included for completeness/teaching.
- **Random coefficient autoregression (RCA)** — Nicholls-Quinn (1982) QMLE; theoretical bridge to ARCH; document the E log|φ_t| < 0 stationarity condition (allows locally explosive paths).
- **GARMA / Gegenbauer seasonal long memory** — Gray-Zhang-Woodward (1989); Whittle estimation as the tractable route; only after ARFIMA infrastructure is solid; validate vs. the R garma package.
- **Sticky HDP-HMM** — Fox-Sudderth-Jordan-Willsky (2011) nonparametric regime count via beam sampling or weak-limit Gibbs; experimental.
- **Croston, SBA, and TSB intermittent-demand methods** — Croston (1972), Syntetos-Boylan (2005), Teunter-Syntetos-Babai (2011); near-free additions; document that Croston has no coherent underlying stochastic model and link to INAR/INGARCH.

## Frontier watchlist

Frontier items from the research sweep not tabled above, kept on watch:

- Boosted HP filter (Phillips-Shi 2021) and the Hamilton (2018) regression filter with automatic stopping/lag rules — diagnostics-owned; this module links its trend/cycle docs there.
- BN filter with automatic signal-to-noise selection for intuitive output gaps (Kamber-Morley-Wong 2018, and their 2025 refinement) — diagnostics-owned; this module supplies the underlying ARMA/companion-form machinery.
- Conformal prediction for dependent data (EnbPI, Xu-Xie 2021; adaptive conformal inference, Gibbs-Candès 2021; SPCI, Xu-Xie 2023) — forecasting-evaluation-owned model-agnostic interval layer; this module's forecasters must plug in cleanly.
- Chandrasekhar recursions + steady-state switching + univariate filtering as the default KF fast path for Monte Carlo work (Herbst 2015) — foundations-owned; known for a decade, implemented almost nowhere, and the single biggest speed lever for this module.
- ES-RNN-style hybrid hooks (Smyl 2020) as the bridge from ETS to ML forecasting — joint design with the ML module.
- FFT-based exact fractional differencing (Jensen-Nielsen 2014) feeding a modern two-step ELW long-memory pipeline (Shimotsu 2010) — integration target across the Tier 2/Tier 3 long-memory rows.

## Implementation warnings

The "easy to get statistically or numerically wrong" list. Every item here has burned an existing package.

1. **The ARMA stationarity/invertibility region is not a box.** Enforce constraints via the Monahan (1984) PACF transform, never coefficient bounds; optimizer failures and non-invertible fits in existing packages nearly all trace to this.
2. **Exact MLE vs. CSS give different answers** — materially so near unit roots and in small samples. Validation against R requires matching the method AND the likelihood-constant conventions to the digit; publish the conventions.
3. **Use exact diffuse Kalman initialization** (Koopman 1997; Durbin-Koopman 2012), never the large-kappa approximation; and exclude diffuse-period terms from the log-likelihood consistently, or information criteria and cross-package parity silently break.
4. **Variance-parameter pile-up at zero** (MA unit root, TVP state variances): the MLE sits on the boundary with high probability even when the true variance is positive; log-scale parameterization hides it. Detect boundary solutions, warn, and offer Stock-Watson (1998) or Bayesian alternatives.
5. **Nonlinear likelihoods (MS, STAR, MAR) are multimodal.** Single-start quasi-Newton silently returns local optima; require multistart plus structured warm starts (EM for MS, (γ, c) grids for STAR) as the default, not an option.
6. **The Hamilton filter must run with log-domain scaling** (underflow on long series); Kim-smoother off-by-one indexing is a classic bug — regression-test smoothed probabilities against Hamilton's published 1989 GNP figures.
7. **Mixture/MS likelihoods are unbounded** as any regime variance → 0: impose variance floors or priors and report boundary hits instead of returning degenerate "estimates."
8. **The Davies problem is everywhere in this domain.** Threshold, STAR, and MS tests have nuisance parameters unidentified under the null — never expose chi-squared p-values for them; ship sup-test critical values, the Hansen (1996) bootstrap, and CHP-type tests.
9. **Bai-Perron results are sensitive to trimming and the heteroskedasticity/autocorrelation configuration flags.** Replicate the original configurations exactly, and compute segment SSRs with numerically stable recursive updating (naive rank-one accumulation drifts on long samples).
10. **ARFIMA: Sowell autocovariances suffer catastrophic cancellation as d → 0.5** (use stable recursions/extended precision); naive truncated fractional differencing is both O(T²) and inaccurate — use FFT-based exact differencing (Jensen-Nielsen 2014).
11. **ETS: multiplicative-error/seasonal models are undefined for nonpositive data**; the admissible parameter region is NOT the [0,1] box (eigenvalue conditions, Hyndman et al. 2008 ch. 10); forecast-variance formulas are class-specific and Class 3 has no closed form — simulate.
12. **Missing data: skip the Kalman measurement update entirely at missing points** (zero-weighting is wrong), count only observed terms in the likelihood, and never difference through NaNs — fit ARIMA in levels with diffuse initialization instead.
13. **Multi-step forecasts for nonlinear models:** iterating one-step conditional means (plug-in) is biased for MS/TAR/STAR beyond h = 1; predictive distributions must integrate over regime/state paths by simulation or exact Chapman-Kolmogorov recursions.
14. **Default analytic prediction intervals ignore parameter uncertainty and are too narrow** — dramatically so near unit roots and for T < 100. Ship bootstrap intervals and say so in the docs where users will see it.
15. **Box-Cox/log back-transformation:** exponentiating the mean forecast yields the median — make bias-adjusted mean vs. median an explicit, documented choice.
16. **Ljung-Box degrees of freedom must subtract the number of fitted ARMA parameters,** and the test is not valid unadjusted on residuals from models with exogenous regressors — get this right or diagnostics mislead every user.
17. **Automatic order selection: information criteria are not comparable across different d, D** (the data change). Select differencing first via unit-root/seasonal-strength tests, then compare IC within fixed (d, D) — and make the search robust to individual fit failures.
18. **Outlier detection: masking/swamping require iterative joint re-estimation;** the detection critical value must grow with T; level shifts and additive outliers are indistinguishable at the sample end; innovational outliers are fragile in nonstationary models (TRAMO disables them — follow suit).
19. **Estimation-method defaults differ silently across ecosystems** (R `ar()` = Yule-Walker, biased toward stationarity; Burg vs. OLS vs. MLE matter near unit roots): document the default prominently and offer all methods.
20. **Gradients: finite differences near constrained boundaries produce spurious convergence declarations.** Invest in analytic KF score recursions or autodiff from day one — it is both a correctness and a speed differentiator.
21. **Monte Carlo reproducibility: use counter-based RNGs (Philox/Threefry) with per-task streams** so results are bit-identical regardless of thread count or scheduling.
22. **Validate against published numbers, not other libraries' defaults:** Nile local-level variances (Durbin-Koopman 2012), Hamilton (1989) GNP MS-AR estimates, the airline model, the Box-Jenkins gas furnace transfer function, Bai-Perron real-interest-rate breaks, the Morley-Nelson-Zivot UC-UR correlation, and M3 aggregate accuracy for ETS/auto-ARIMA — cross-package agreement can just mean shared bugs.

## Dependencies and shared infrastructure

**Consumed from foundations:**

- **Linear-Gaussian state-space engine** (exact diffuse initialization, univariate filtering of multivariate observations, Durbin-Koopman and precision-based simulation smoothers, EM) — the estimation backbone for ARMA exact MLE, ETS, UC, TVP, and MS state space. This module additionally needs the Koopman (1993) disturbance smoother exposed for auxiliary-residual diagnostics, and the Chandrasekhar/steady-state fast path for Monte Carlo throughput.
- **Resampling/bootstrap engine** — Hansen (1996) fixed-regressor/wild bootstraps for threshold tests, Kilian (1998) bootstrap-after-bootstrap, and all simulation-based null distributions.
- **Bootstrap prediction-interval engine** (forecasting-owned) — this module supplies the model-specific resampling hooks (Thombs-Schucany 1990 backward AR bootstrap; Pascual-Romo-Ruiz 2004 for ARIMA) and the headline near-unit-root undercoverage documentation example.
- **Critical-value engine** — Bai-Perron response-surface CVs, Andrews (1993/2003) corrected tables, Hansen (1997) p-value functions, Elliott-Müller (2006) CVs, Hansen (2000) threshold-CI tabulations.
- **Innovation-distribution zoo** — heavy-tailed/skewed innovations for GAS/DCS models, non-Gaussian MLE for MARX, ETS simulation, and count-model mixing distributions.
- **Deterministic-terms toolkit and calendar/holiday engine** — trends, seasonal dummies, Fourier terms, and X-13/TRAMO-exact trading-day/Easter/leap-year regressors so regARIMA pre-adjustment is reproducible natively.
- **Temporal disaggregation and benchmarking** (Chow-Lin, Denton, Fernandez, Litterman) — this module documents and cross-links them for its national-accounts users.
- **Exogenous-regressor (covariate) contract** — regARIMA/ARIMAX, transfer functions, intervention analysis, and TVP regressions ingest covariates through the shared aligned, leakage-checked interface; forecasting with covariates flows through its known-future / scenario-path / auxiliary-forecast distinction. This module contributes the central documentation of the regression-with-ARMA-errors vs transfer-function-ARMAX distinction (the perennial statsmodels-vs-R trap).
- **Numerical optimizers**, **Philox-based reproducible parallel RNG**, the **unified forecast object**, and the **golden-value validation harness** — used by every row above.
- **HAC/long-run-variance inference** — standard errors for HAR regressions and break/stability tests, under the library-wide default policy.
- **Box-Cox back-transform machinery** (foundations/forecasting-owned) — this module integrates Guerrero (1993) λ selection into its pipelines and surfaces the mean-vs-median forecast choice.
- **Rolling-origin CV harness** (forecasting-owned) — this module's fits must be cheap and parallel enough to refit thousands of times inside it.

**Consumed from other modules:**

- **diagnostics** — STL/MSTL decomposition; the HP/one-sided HP/boosted HP, Baxter-King/Christiano-Fitzgerald/Butterworth, Hamilton, and Beveridge-Nelson/BN filter suite; and the X-13ARIMA-SEATS wrapper. This module's regARIMA internals must match X-13 conventions so the wrapper's pre-adjustment is reproducible natively.
- **forecasting-evaluation** — forecast-comparison tests (DM/GW/MCS/SPA), density-forecast evaluation, forecast combination, and conformal prediction; this module's forecasters emit the unified forecast object those tools expect.
- **bayesian** — priors and samplers for the Bayesian ARIMA/UC/TVP and Bayesian MS rows; MCMC diagnostics interoperate with ArviZ.
- **volatility** — interface boundary for particle filtering and shared conditional-recursion machinery (ACD mirrors GARCH QMLE theory).

**Exposed to other modules:**

- The unified `.simulate()` engine every bootstrap and Monte Carlo tool calls.
- Companion-form utilities, the Monahan transform, and the fast fractional-differencing operator (consumed by multivariate/fractional cointegration).
- Hamilton/Kim regime-switching filters (precursors to MS-DFM work in multivariate/nowcasting).
- Fitted ARMA/ETS/auto-ARIMA models as standard benchmarks for the forecasting-evaluation and ML modules.
- Structural-break and stability tests reused by the multivariate and LP modules.
- The structured residual-diagnostics report object, wired to "which model when" documentation.

## Validation gallery

- **Nile local level (Durbin-Koopman 2012)** — σ²_ε ≈ 15099, σ²_η ≈ 1469.1 must match, KFAS output element-by-element, and auxiliary residuals must flag the documented 1899 level break.
- **Airline model on log AirPassengers (Box-Jenkins 1976)** — ARIMA(0,1,1)(0,1,1)_12 θ estimates and log-likelihood must match R `arima()` exactly; Series A–J for the wider ARMA stack to 6+ digits.
- **Hamilton (1989) GNP MS-AR** — μ₀ ≈ 1.16, μ₁ ≈ −0.36, p₀₀ ≈ 0.90, p₁₁ ≈ 0.75, and the smoothed-probabilities figure must reproduce.
- **Bai-Perron (1998, 2003) US ex-post real interest rate** — break dates 1972:3 and 1980:3 with matching supF statistics and confidence intervals; cross-checked vs. strucchange/mbreaks.
- **Morley-Nelson-Zivot (2003) UC-UR** — US GDP trend-cycle innovation correlation ≈ −0.9.
- **M3 competition aggregates** — ETS and auto-ARIMA must land near-identical model choices and published MASE/sMAPE to `forecast::ets` / `forecast::auto.arima` (Hyndman-Khandakar 2008).
- **Box-Jenkins gas furnace** — transfer-function estimates vs. SCA/tfarima.
- **UK seatbelt data (Harvey-Durbin 1986)** — BSM intervention effect and seasonal component vs. published estimates and STAMP.
- **Kim-Nelson (1999) book examples** — Kim filter/smoother and MS regressions vs. their published GAUSS results.
- **Shimotsu (2010) extended Nelson-Plosser** — two-step ELW estimates vs. published MATLAB results; Doornik-Ooms (2003) for Sowell ARFIMA.
- **Kanzler (1999) BDS benchmark values** — exact small-sample epsilon conventions of the original C code.
- **Stock-Watson (1998) G7 trend growth** — median-unbiased TVP variance estimates and lookup-table mapping.
- **R tscount (JSS 2017) campylobacter/E. coli examples** — INGARCH estimates; Engle-Russell (1998) IBM durations for ACD.
