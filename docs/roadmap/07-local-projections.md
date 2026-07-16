# Module 07 — Local Projections

> Part of the time series econometrics library roadmap. Master plan: [ROADMAP.md](../../ROADMAP.md).

**This module is the library's local-projection engine: per-horizon impulse-response estimation (Jordà 2005) and everything the modern empirical-macro literature has built on top of it — LP-IV, joint and simultaneous inference, one-step fiscal multipliers, state-dependent and quantile variants, panel and LP-DiD designs, and LP-VAR dual reporting. Nothing maintained exists in Python; the best available tool anywhere (R's `lpirfs`) is inference-thin and a decade behind the theory. This is a headline differentiator module: a Rust core makes the bootstrap- and simulation-heavy inference that the literature now demands fast enough to be the default rather than a luxury.**

## Purpose and scope

Local projections estimate impulse responses by running a separate regression of the horizon-`h` outcome on an impulse variable and controls, one regression per horizon, instead of committing to a full dynamic system and iterating it forward. Since Jordà (2005), LPs have become the workhorse of applied macroeconomics — fiscal multipliers, monetary transmission, credit-cycle dynamics, growth-at-risk — because they are robust to misspecification, extend trivially to instrumental variables and nonlinearities, and travel well to panel data. This module owns everything local-projection: point estimation, identification (internal instruments, LP-IV), the full inference stack (lag-augmented, HAC/HAR, bootstrap, joint cross-horizon, simultaneous bands, weak-IV-robust sets), cumulative multipliers, efficiency-improving variants (smooth, Bayesian, GLS, bias-corrected), state-dependent and quantile LPs, panel LP with its bias corrections, LP-DiD, and the LP-VAR equivalence machinery.

The intended users are applied macroeconomists replicating and extending the Ramey–Zubairy / Gertler–Karadi style of evidence, central-bank staff producing policy IRFs and multipliers, and panel researchers who want event-study estimates free of two-way-fixed-effects pathologies. The design commitment that shapes the whole module is that inference — not point estimation — is where existing tools fail, so the joint cross-horizon covariance of the entire IRF vector is a first-class internal object from day one: simultaneous bands, path Wald tests, significance bands, multiplier delta methods, and counterfactual policy calculations all fall out of it.

This module must also make one policy decision loudly and document it everywhere: **the default inference mode is lag-augmented LP with heteroskedasticity-robust (EHW) standard errors, per Montiel Olea & Plagborg-Møller (2021)** — uniformly valid across persistence (including unit roots) and horizon length, and simpler than HAC. HAC/HAR inference remains fully supported and is required whenever the impulse regressor is not innovation-like, but it is the explicit fallback, never the silent default. Relative to its neighbors: the module consumes the HAC/EWC, bootstrap, and quantile-regression engines from foundations, consumes VAR estimation from the multivariate module for dual reporting, and leaves structural-VAR identification to the identification module. Per the master ruling, LP-DiD and IPW/AIPW-LP stay in scope here; the broader causal-panel suite lives in a companion package.

## Where existing tools fall short

- **statsmodels has no local projections at all.** Python users hand-roll per-horizon OLS with Newey–West and routinely get sample alignment, bandwidth growth with horizon, and multiplier construction wrong. There is no serious Python LP package, period — this domain is a green field.
- **R `lpirfs` (the current best) is inference-thin.** Newey–West-only standard errors (no EWC/fixed-b HAR, no lag-augmented EHW default per Montiel Olea & Plagborg-Møller 2021), no joint cross-horizon covariance, hence no sup-t simultaneous bands or path Wald tests; its default HAC bandwidth = `h` is undocumented folklore.
- **`lpirfs` is also missing the modern applied toolkit:** no one-step cumulative-multiplier IV (the Ramey–Zubairy standard), no weak-IV diagnostics beyond a basic F, no Anderson–Rubin confidence sets, no LP-DiD, no smooth/Bayesian/GLS LP — and its R-loop bootstraps are too slow for serious Monte Carlo work.
- **Stata's built-in `lpirf` (Stata 18) covers only linear LP with basic HAC.** State dependence, IV multipliers, and panel variants require hand-written code or the separate user-contributed `lpdid` package; nothing produces simultaneous bands.
- **The gold-standard numbers live in one-off replication archives** (Ramey–Zubairy Stata/MATLAB files, Barnichon–Brownlees MATLAB, Montiel Olea–Plagborg-Møller MATLAB, McKay–Wolf code) — correct but non-reusable, unmaintained, and mutually inconsistent in conventions (sample alignment, normalization, bandwidths).
- **No package anywhere implements:** LP-GLS (Lusompa 2023), significance bands (Inoue–Jordà–Kuersteiner), time-varying LP (Inoue–Rossi–Wang), AIPW doubly-robust LP, LP-VAR shrinkage (Li–Plagborg-Møller–Wolf), bias-corrected LP (Herbst–Johannsen), panel Nickell corrections for LP, McKay–Wolf counterfactuals, or Barnichon–Mesters optimal policy.
- **No existing tool fits LP and VAR from a single specification and reports both**, despite Plagborg-Møller–Wolf equivalence being the organizing result of the field — users juggle two packages with different conventions and cannot tell specification error from estimator noise.
- **Joint (simultaneous) inference is essentially absent from all mainstream tools.** Published papers overwhelmingly present pointwise bands that readers interpret as joint — a library that makes sup-t bands the default plot would move empirical practice.
- **Dynare has nothing for LP or reduced-form IRF inference.** Central-bank users who estimate DSGEs have no bridge from LP evidence to model-based counterfactuals (McKay–Wolf closes this gap conceptually; no software does).

## Inventory

Difficulty scale: low / medium / high / research-grade. Two source items — kernel HAC estimation (Newey–West 1987/1994, Andrews 1991) and HAR fixed-b/EWC inference (Lazarus–Lewis–Stock–Watson 2018, Kiefer–Vogelsang 2005) — are owned by foundations under the master ownership map and appear under [Dependencies and shared infrastructure](#dependencies-and-shared-infrastructure) with the LP-specific policy this module layers on top.

### Tier 1 — Core (v1-blocking)

**Core estimation**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Baseline local projections | Per-horizon OLS of `y_{t+h}` on an impulse variable plus controls, yielding IRFs without committing to a full dynamic system. The workhorse flexible IRF estimator every other item builds on. | Low | Per-horizon OLS via QR — never normal equations. Store per-horizon scores/residuals so the joint cross-horizon covariance (sup-t bands, Wald tests) comes free. Expose sample policy explicitly: common sample across horizons (Ramey convention) vs. maximal per-horizon sample — results differ. Support unit-effect and one-SD normalization. Jordà (2005). Validate: R `lpirfs::lp_lin` and Stata 18 `lpirf`; reproduce Jordà (2005) figures; match `lpirfs` vignette numbers to 3+ decimals. |

**Identification**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| LP with internal instrument / recursive identification | Structural IRFs by ordering an observed shock or proxy first among controls, or contemporaneous controls mimicking Cholesky ordering. Recommended default when a shock series is available but possibly noisy. | Low | Regress `y_{t+h}` on the shock with lags of the shock and all system variables as controls; equivalence to recursive SVAR identification is Plagborg-Møller & Wolf (2021). Under classical measurement error in the proxy, unit-effect-normalized IRFs remain valid — normalize on the impact response of the policy indicator. Validate: LP IRFs vs. large-`p` VAR Cholesky IRFs on simulated ARMA DGPs (asymptotic coincidence), plus a Gertler–Karadi (2015)-style application. |
| LP-IV (external instruments) | Per-horizon 2SLS where the endogenous impulse variable is instrumented by a narrative or high-frequency external instrument. The dominant identification approach in modern empirical macro. | Medium | Lead-lag exogeneity (Stock & Watson 2018) requires controls that absorb autocorrelation in `z` — enforce identical control sets across stages. Report per-horizon first-stage effective F (Montiel Olea & Pflueger 2013, HAC-robust variant). HAC/HAR in the 2SLS sandwich. Stock & Watson (2018); Ramey & Zubairy (2018). Validate: Ramey–Zubairy replication files (military news; linear multiplier ≈0.6–0.7 at 2–4y) and Gertler–Karadi FF4 surprises — these numbers are the credibility bar. |

**Inference: standard errors**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Lag-augmented LP inference | Augment the horizon-`h` regression with lags of all variables so the score is approximately a martingale difference; plain EHW standard errors are then valid uniformly over persistence (including unit roots) and at long horizons. **The library's default inference mode.** | Medium | Key insight: with lag augmentation and an innovation-like impulse, no HAC is needed; if the impulse regressor is not a shock, HAC is still required — make the inference mode explicit in the API, never silent. Pairs naturally with the wild bootstrap in small samples. Montiel Olea & Plagborg-Møller (2021). Validate: their replication code (coverage Monte Carlos across a persistence/horizon grid; Gertler–Karadi application). |

**Inference: confidence bands**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Joint multi-horizon estimation and IRF covariance | Estimate all horizons as a stacked GMM/SUR system and compute the joint HAC/EWC covariance of the entire IRF vector. Prerequisite for simultaneous bands, cross-horizon Wald tests (path = 0, peak timing, shape restrictions), and multiplier delta methods. | Medium | Stack per-horizon moment conditions sharing observations; one long-run variance of the stacked score captures cross-horizon correlation from overlapping samples. Pitfall: unbalanced samples across horizons (later horizons lose observations) — use a common sample or handle unbalanced stacking explicitly. Jordà (2009); Montiel Olea & Plagborg-Møller (2019). Validate: reproduce the MOP (2019) application bands exactly. |
| sup-t simultaneous confidence bands | Simultaneous bands with exact asymptotic joint coverage — narrower than Bonferroni, honest unlike pointwise. The correct object for "is the IRF significant over horizons 4–16" questions; almost no mainstream package offers them. | Medium | Plug-in sup-t: simulate max-\|t\| from the multivariate normal with the estimated IRF correlation matrix; also bootstrap and Bayesian-calibration variants. Requires the joint covariance item above. With bootstrap, use studentized sup statistics. Montiel Olea & Plagborg-Møller (2019). Validate: their MATLAB replication code. |

**Inference: bootstrap**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Bootstrap inference for LP | Resampling-based CIs and bands: wild bootstrap on lag-augmented scores, moving-block bootstrap on data tuples, and model-based VAR bootstrap for comparison studies. A headline speed use-case for the Rust core. | High | Naive per-horizon iid residual resampling is invalid (residuals are dependent within and across horizons). Implement: (i) wild bootstrap on lag-augmented LP (Montiel Olea & Plagborg-Møller 2021 variant); (ii) MBB with data-driven block length, percentile-t (studentized) — percentile alone has poor coverage; (iii) VAR-bootstrap-then-LP à la Kilian & Kim (2011). Built on the foundations bootstrap engine with Philox counter-based RNG for thread-count-independent reproducibility. Validate: replicate the Kilian–Kim coverage tables. |

**Policy analysis & multipliers**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Cumulative IRFs and fiscal multipliers via one-step IV | Horizon-`H` multiplier estimated in a single IV regression of cumulated output on cumulated spending instrumented by the shock — the modern standard for fiscal multipliers, with correct inference built in. | Medium | One-step IV: `Σ_{j≤h} y_{t+j}` on `Σ_{j≤h} g_{t+j}` instrumented by the shock, HAC SEs — **not** the ratio of two separately estimated cumulative IRFs with the delta method (offer that only as a labeled comparison; results differ materially). Handle unit conversion (Gordon–Krenn transformation, shares of GDP) as first-class options. Ramey & Zubairy (2018); Ramey (2016). Validate: RZ published multipliers, linear and state-dependent, from their Stata replication files — **the** validation target for the whole module. |

**Nonlinear & state-dependent LP**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| State-dependent LP with regime indicator | Interact all regressors with a lagged binary state indicator (recession/expansion, slack) for regime-specific IRFs and multipliers, including the IV version for state-dependent fiscal multipliers. | Medium | Interact everything — impulse, controls, deterministics — with `I_{t-1}`; the IV version interacts the instrument with the state. Use a lagged state to limit endogeneity, but document the deeper issue: validity requires the state not to respond to the shock, else the estimand changes (Gonçalves, Herrera, Kilian & Pesavento 2021/2024) — emit a diagnostic warning. State-specific multipliers via one-step cumulative IV within regime. Ramey & Zubairy (2018). Validate: RZ state-dependent multipliers from their code. |

**Panel & micro-causal LP**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Panel LP with fixed effects | Within-estimator LP for country/firm/household panels with a full SE menu (unit cluster, two-way, Driscoll–Kraay for cross-sectional dependence), including panel LP-IV as used throughout the Jordà–Schularick–Taylor macrohistory literature. | Medium | Careful handling of unbalanced panels and gaps once outcomes are shifted `h` periods. Small-N cluster corrections (CR2/CR3, wild cluster bootstrap); Driscoll–Kraay needs large T. Jordà, Schularick & Taylor (2015 onward). Validate: `lpirfs::lp_lin_panel` vignette and a JST macrohistory replication. |

### Tier 2 — Standard (expected of a serious library)

**Inference: confidence bands**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Significance bands for LP | Bands constructed under the null of zero response (analogous to ACF significance bands) — the right tool for "is there any response at all," complementary to confidence bands. | Low | Inoue, Jordà & Kuersteiner (2023, FRBSF WP) give the LP construction accounting for serial dependence under the null. Trivial on top of the stacked-covariance machinery; large pedagogical payoff for the documentation pillar. Validate: their R replication code. |

**Efficiency & shrinkage**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Smooth local projections (SLP) | All horizons estimated jointly with the IRF in a B-spline basis and a ridge penalty on r-th differences, shrinking toward a polynomial. Cuts LP variance dramatically while keeping flexibility; currently lives only in unmaintained MATLAB code. | High | Penalty applies only to IRF coefficients, not controls. Tune λ by blocked (time-series-aware) cross-validation — iid k-fold overfits under autocorrelation. Post-shrinkage SEs are not honest; provide bootstrap bands with explicit caveats. Watch B-spline endpoint behavior and knot placement across the horizon grid. Barnichon & Brownlees (2019). Validate: their MATLAB replication (monetary application, penalty path). |

**Nonlinear & state-dependent LP**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Smooth-transition LP | Logistic-weighted regime mixing `F(z_t)` applied to all regressors, giving IRFs that vary smoothly with the cycle. The most-used nonlinear LP variant after the dummy interaction. | Medium | `F(z) = exp(−γz)/(1+exp(−γz))` on a standardized state variable; γ typically calibrated (AG use 1.5) — expose as a parameter with optional estimation and show sensitivity in docs. Pitfalls: centered-moving-average or two-sided-filter state variables introduce look-ahead (the standard AG critique) — warn; near-collinearity between the two weighted regressor blocks. Auerbach & Gorodnichenko (2012, 2013). Validate: match `lpirfs::lp_nl`, then match AG's original results. |
| Sign- and size-dependent LP | Asymmetric responses to positive vs. negative shocks (censored regressors `max(s,0)`/`min(s,0)`) and size nonlinearity via polynomial shock terms. Standard in the monetary and fiscal asymmetry literatures. | Medium | Asymmetry tests need the joint covariance of both branches (reuse the stacked machinery). Document estimand caveats from Kolesár & Plagborg-Møller (2025) — what weighted average a misspecified nonlinear LP recovers. Tenreyro & Thwaites (2016); Ben Zeev, Ramey & Zubairy (2023). Validate: Tenreyro–Thwaites replication. |

**Panel & micro-causal LP**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| LP-DiD | Event-study difference-in-differences via LP with a clean-control condition: regress `h`-horizon outcome changes on treatment switches using only not-yet/never-treated comparisons, avoiding the negative-weight pathologies of TWFE event studies. | High | Implement the clean-control sample restriction, optional reweighting to an equally-weighted ATT, pre-event horizons (`h < 0`) for pre-trend display, absorbing vs. non-absorbing treatment, and an IV variant. In core scope per master ruling; the broader causal-panel suite lives in a companion package. Dube, Girardi, Jordà & Taylor (2023, NBER WP 31184). Validate: their Stata `lpdid` package output on staggered-adoption simulations; cross-check against Callaway–Sant'Anna where designs overlap. |

**LP-VAR relationship**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| LP-VAR equivalence tooling and dual reporting | LPs and VARs estimate the same population IRFs; fit both from a single model specification and overlay them, turning the organizing insight of the modern literature into a diagnostic (divergence signals lag-length/specification problems). | Medium | Single spec object → {LP IRF, iterated VAR(p) IRF, large-`p` VAR IRF} with shared identification (Cholesky, internal/external instrument). VAR estimation is consumed from the multivariate module. Document the finite-sample bias-variance tradeoff. Plagborg-Møller & Wolf (2021). Validate: reproduce PMW simulation figures; verify small-`h` numerical near-equivalence with matched lag structures. |

**Forecasting**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Direct multi-step forecasting via LP | The LP regression at horizon `h` is exactly the direct `h`-step forecast; expose a forecasting API with direct-vs-iterated comparison and forecast-path fan charts reusing the sup-t band machinery. | Low | Canonical evidence on direct vs. iterated: Marcellino, Stock & Watson (2006). Direct forecasts are non-monotone across `h` — offer optional smoothing/reconciliation across horizons. Per-horizon forecast comparison via the forecasting-evaluation module's Diebold–Mariano implementation. Validate: reproduce MSW-style rankings on a FRED-MD subset. |

### Tier 3 — Advanced (differentiators)

**Identification**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Weak-instrument-robust LP-IV inference (Anderson–Rubin grid inversion) | AR-type test inversion per horizon (and for cumulative multipliers) giving confidence sets valid under weak instruments — sets may be unbounded or disjoint. Narrative instruments are frequently weak, so this is not optional for a serious library. | High | Grid over β; regress `y_{t+h} − β·x_t` on `z_t` + controls with HAC/HAR; invert. Extend to the multiplier ratio equation (Ramey–Zubairy report AR-style intervals for state-dependent multipliers). The API must represent unbounded/disjoint sets honestly — never silently truncate to a finite interval. Report Montiel Olea–Pflueger effective F alongside. Anderson & Rubin (1949) adapted per Stock & Watson (2018); Ramey & Zubairy (2018) appendix. Validate: RZ appendix intervals. |

**Efficiency & shrinkage**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Bayesian local projections (BLP) | Full Bayesian LP: VAR-based priors that place the estimator on the LP-VAR bias-variance frontier, with explicit modeling of the MA(h) error structure. No general-purpose implementation exists anywhere. | Research-grade | Two flavors: roughness-penalty priors on the IRF path (Tanaka 2020) and VAR-prior BLP (Ferreira, Miranda-Agrippino & Ricco, REStat). Gibbs sampling over stacked horizons is compute-heavy — a strong fit for the Rust core. Pitfall: the pseudo-likelihood with overlapping MA errors is misspecified; follow the paper's treatment exactly. Validate: FMR replication applications. |
| LP-GLS efficient estimation | Exploits the known MA(h) structure of LP errors via a recursive GLS transformation using earlier-horizon estimates, recovering large efficiency gains over OLS-HAC. Published in Quantitative Economics; implemented in no mainstream package. | Research-grade | Recursive Cochrane–Orcutt-style transformation: the horizon-`h` error depends on earlier-horizon IRF coefficients; estimation error propagates across horizons, so use the paper's wild-bootstrap inference rather than plug-in SEs. Lusompa (2023). Validate: his replication code on the Gertler–Karadi and Ramey datasets. |
| Small-sample bias correction | LP point estimates are biased in finite samples (analogue of Kendall AR bias), with bias growing in horizon and persistence; analytical corrections exist and should be a user-toggleable option. | Medium | Herbst & Johannsen (2024, Journal of Econometrics) derive the bias and a feasible correction; interacts with lag augmentation. Cheap given the core regression infrastructure. Validate: their simulation tables and empirical illustration. |

**Nonlinear & state-dependent LP**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Quantile LP / growth-at-risk dynamics | Per-horizon quantile regressions giving IRFs of conditional quantiles — the dynamic extension of Adrian–Boyarchenko–Giannone growth-at-risk, tracing how shocks move tail risk. | High | Koenker–Bassett per horizon and τ (solver consumed from foundations); inference via moving-block bootstrap or HAC-robust QR sandwich (Powell kernel bandwidth). Offer monotone rearrangement (Chernozhukov, Fernández-Val & Galichon 2010) to fix quantile crossing across τ and `h`. Linnemann & Winkler (2016); Ruzicka (2021); Adrian, Boyarchenko & Giannone (2019) as the one-step special case. Validate: Linnemann–Winkler fiscal quantile IRFs; ABG one-step results. |

**Panel & micro-causal LP**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Nickell bias in panel LP and jackknife corrections | FE panel LP with predetermined regressors suffers dynamic-panel bias that grows with horizon, plus a Stambaugh-type bias from persistent regressors. Corrections exist but are in no package. | High | Bias is O(1/T) but horizon-amplified (effectively O(h/T)); implement the half-panel jackknife (Dhaene & Jochmans 2015) applied per horizon and the analytical corrections of Mei, Sheng & Shi ("Nickell Meets Stambaugh"). Emit warnings when T is small relative to the max horizon. Validate: MSS Monte Carlo designs. |
| IPW and doubly-robust (AIPW) LP | Treat policy as a treatment: model the propensity of intervention, inverse-probability-weight the LP to remove allocation bias, or combine with outcome regression for double robustness. | High | Propensity via probit/logit (or ML — see the DML item); per-horizon weighted LP; AIPW per Angrist, Jordà & Kuersteiner (2018). Pitfalls: weight trimming/truncation policy, overlap diagnostics, and inference accounting for estimated weights — stack the propensity and outcome moment conditions or bootstrap the entire two-step. In core scope per master ruling. Jordà & Taylor (2016); AJK (2018). Validate: both papers' replication files. |

**LP-VAR relationship**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| LP-VAR shrinkage / model averaging | Estimators interpolating between low-bias LP and low-variance VAR-iterated IRFs; the LPW study over thousands of DGPs shows intermediate estimators dominate in MSE. A genuine differentiator no package offers. | Research-grade | Penalized LP shrinking toward the VAR(p)-implied IRF, tuned by blocked CV or Stein-type risk criteria; Bayesian LP is the Bayesian route to the same frontier. Honest post-shrinkage inference is unsolved — ship bootstrap bands with loud caveats. Li, Plagborg-Møller & Wolf (2024, Journal of Econometrics). Validate: port the LPW DGP battery as a permanent test suite — it doubles as the library's Monte Carlo benchmark. |

### Tier 4 — Frontier (research-grade, each gated on a named validation target)

**LP-VAR relationship**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Double robustness of LP / bias-aware VAR bands ("Unpleasant VARithmetic") | Theory showing LP with sufficient lag controls is doubly robust to misspecification while VAR bias does not vanish even with wide bands — the intellectual justification for the lag-augmented LP default, plus bias-aware honest CIs for the VAR comparator. | Medium | Implementation is mostly defaults-plus-diagnostics: lag-augmentation depth guidance and Armstrong–Kolesár-style bias-aware bands for the VAR comparator. Montiel Olea, Plagborg-Møller, Qian & Wolf (2024, NBER WP 32495). Gate: reproduce the paper's simulations. |

**Frontier robustness**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Time-varying LP in unstable environments | Kernel-weighted/rolling LP delivering IRFs as functions of calendar time, with inference and stability tests — captures drift in transmission mechanisms (pre/post-1980s monetary policy, ZLB). | Research-grade | Nonparametric time-weighting with bandwidth selection; joint inference over (time, horizon) surfaces; boundary bias at sample endpoints. Inoue, Rossi & Wang (2024, Journal of Econometrics). Gate: their replication (time-varying monetary IRFs). |
| High-dimensional and ML controls in LP (double selection, time-series DML) | Lasso double-selection and double/debiased ML adapted to LP for many-control settings (large macro panels, granular micro data) and for ML-based propensities in IPW-LP. | Research-grade | Belloni–Chernozhukov–Hansen double selection per horizon; DML cross-fitting must respect temporal ordering (blocked folds, no leakage across the `h`-step overlap); post-selection HAC inference is delicate. Framework: Chernozhukov et al. (2018). Penalized solvers and time-series CV consumed from the ML module. No canonical package exists — pure differentiator. Gate: designed-in simulation coverage studies in the test battery. |

**Policy analysis & multipliers**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Policy counterfactuals from estimated IRFs | Combine empirically identified IRFs to multiple policy instruments/shocks to construct counterfactual policy-rule outcomes robust to the Lucas critique under stated conditions — bridges the LP toolkit to Dynare-style policy questions. | Research-grade | Needs joint estimation of an IRF matrix (several outcomes × several policy shocks) with full covariance; the counterfactual is a linear combination solving the policy-rule restriction; bands via delta method or joint bootstrap. McKay & Wolf (2023, Econometrica). Gate: McKay–Wolf replication code. |
| Sufficient-statistics optimal policy on estimated IRFs | Compute optimal policy adjustments by minimizing a loss subject to LP/IV-estimated IRFs of targets to policy shocks (the Phillips multiplier as a special case) — turns the IRF engine into a policy-evaluation tool for central-bank users. | High | GMM on stacked IRF estimates plus quadratic programming; inference flows from the joint IRF covariance. Barnichon & Mesters (2023, AER); Barnichon & Mesters (2021, JME). Gate: both papers' replication files. |

**Nonlinear & state-dependent LP**

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| IRF heterogeneity decomposition for nonlinear LP | Kitagawa–Oaxaca–Blinder-style decomposition of state-dependent IRF differences into direct, indirect, and composition components — answers *why* responses differ across regimes, not just whether. | High | Requires the state-dependent LP machinery plus counterfactual reweighting of covariate distributions across regimes. Cloyne, Jordà & Taylor (2023, NBER WP 30864). Gate: their replication code. |

## Frontier watchlist

Most of the source inventory's frontier list is already scheduled above — lag-augmented EHW as the default engine (Tier 1), significance bands and LP-DiD (Tier 2), LP-GLS, Bayesian LP, LP-VAR shrinkage, quantile LP with rearrangement, panel bias corrections, and the Herbst–Johannsen correction (Tier 3). The residual watchlist:

- **Nonlinear-world estimand diagnostics** — report what weighted average of heterogeneous responses a given (possibly misspecified) nonlinear LP recovers (Kolesár & Plagborg-Møller 2025); ship as a diagnostic attached to nonlinear LP results.
- **State-dependence validity checks** — automated warnings when the conditioning state plausibly responds to the shock, changing the estimand (Gonçalves, Herrera, Kilian & Pesavento 2021/2024).
- **The Li–Plagborg-Møller–Wolf (2024) DGP battery as a public benchmark product** — beyond serving as this module's internal test suite, publish it as a reusable Monte Carlo benchmark for IRF estimators.

## Implementation warnings

- **LP residuals are MA(h) by construction**, even under correct specification. Serial correlation in LP residuals is not a specification failure and must not trigger naive GLS "fixes"; efficiency gains require the Lusompa recursive construction, not textbook Cochrane–Orcutt.
- **Sample alignment is the silent killer.** Horizon `h` loses the last `h` observations, so per-horizon maximal samples differ across `h`. Ramey–Zubairy fix a common sample; `lpirfs` does not by default. Results differ visibly. Choose a deliberate default, expose both, and document which convention every validation target uses.
- **Never pair a fixed-b/EWC variance estimator with N(0,1) critical values.** The nonstandard (t_ν or Kiefer–Vogelsang) critical values *are* the size correction. Thread degrees of freedom through CIs, bands, and Wald tests.
- **Lag augmentation makes EHW (non-HAC) SEs valid only when the impulse regressor is innovation-like.** If users put a persistent observable as the impulse, HAC is still required. Make the inference mode an explicit, validated API choice — never inferred silently.
- **Cumulative multipliers must be estimated by one-step IV on cumulated sums** (Ramey–Zubairy 2018). The ratio of separately estimated cumulative IRFs with delta-method SEs is a different, inferior estimator that can differ materially — offer it only as a labeled comparison.
- **Percentile bootstrap for LP has poor coverage** (Kilian–Kim 2011); implement studentized (percentile-t) intervals. Never resample per-horizon residuals iid — residuals are dependent within and across horizons; resample blocks of data tuples or use the wild bootstrap on lag-augmented scores.
- **Simultaneous bands require the joint covariance of the IRF vector**, including cross-horizon terms induced by overlapping samples. Computing per-horizon SEs and assuming independence produces bands that are wrong in both width and shape.
- **Weak-IV AR confidence sets can be unbounded or disjoint** — the return type must represent this honestly; silently reporting a huge finite interval is a statistical lie. Report the HAC-robust effective F (Montiel Olea–Pflueger) per horizon by default.
- **State-dependent LP:** interacting with a contemporaneous or shock-responsive state changes the estimand (Gonçalves, Herrera, Kilian & Pesavento 2021/2024); default to lagged states and emit a warning with a documentation link when the state is plausibly endogenous. Smooth-transition results are highly sensitive to γ and to look-ahead in filtered state variables (the standard Auerbach–Gorodnichenko critique) — surface both in diagnostics.
- **Panel LP:** Nickell bias grows with horizon (effectively O(h/T)); Driscoll–Kraay requires large T; small-N clusters need the wild cluster bootstrap or CR2/CR3. Emit warnings keyed to panel dimensions relative to the max horizon.
- **Near-unit-root data invalidate standard LP inference at long horizons** (`h` a nontrivial fraction of T) unless lag-augmented. Do not let users request `h = T/2` with textbook Newey–West SEs without a warning.
- **All cross-validation for smooth/shrinkage LP must be blocked in time.** IID k-fold leaks the MA(h) overlap and systematically undersmooths.
- **Numerics:** per-horizon regressions share almost identical regressor matrices (shifted outcomes) — exploit shared QR factorizations for order-of-magnitude speedups, but only when samples truly coincide. Never form X'X (the condition number squares). Handle dummy-interacted designs where subsample columns are all-zero.
- **Reproducibility:** bootstrap and sup-t simulation draws must use counter-based parallel RNG so results are bit-identical across thread counts — Monte Carlo speed is a design pillar, and non-reproducible parallelism would poison validation.
- **Shifting outcomes creates missing values at the sample edge.** Silently dropping rows yields horizon-varying samples and subtly non-comparable IRFs — make the missing-data policy explicit and logged.
- **Normalization conventions differ across the validation corpus** (unit-effect vs. one-SD, impact normalization under proxy measurement error, sign conventions). Store the convention in the results object and print it, or users will "replicate" papers off by a scale factor.
- **HAC bandwidth policy:** the folklore bandwidth = `h` (the `lpirfs` default) is ad hoc — implement it for replication but document it; the bandwidth must grow with horizon since errors are MA(h). Known size distortions at long horizons motivate HAR/fixed-b and lag augmentation instead.
- **Validation discipline:** every estimator ships with a test that reproduces a published number (Ramey–Zubairy multipliers, Gertler–Karadi IRFs, Auerbach–Gorodnichenko 2013, Barnichon–Brownlees, Montiel Olea–Plagborg-Møller 2019/2021 replication output, `lpirfs` vignettes) to 3+ decimals. Replication fidelity against Ramey's codes and `lpirfs` is the bar the whole library is marketed on.

## Dependencies and shared infrastructure

**Consumed from foundations:**

- **HAC / long-run variance / fixed-b / EWC inference engine.** The per-horizon and stacked LP sandwiches call the shared kernel-LRV estimators (Bartlett/QS, Newey–West 1987/1994 and Andrews 1991 automatic bandwidths, Andrews–Monahan prewhitening) and the EWC/fixed-b machinery (Lazarus, Lewis, Stock & Watson 2018 with ν = 0.4·T^(2/3) and t_ν critical values; Kiefer–Vogelsang 2005 fixed-b). This module layers LP-specific policy on top: bandwidth must grow with horizon (errors are MA(h)); the `lpirfs` bandwidth = `h` folklore is implemented for replication only; fixed-b degrees of freedom are threaded through all CIs, bands, and Wald tests. Validation of the engines themselves (match Stata `newey` and R `sandwich::NeweyWest` to machine precision; LLSW replication tables) lives with foundations.
- **Bootstrap engine.** Wild/moving-block/stationary resampling with data-driven block-length selection and the Philox parallel-RNG substream contract; this module supplies the LP-specific schemes (wild-on-lag-augmented-scores, tuple MBB percentile-t, VAR-bootstrap-then-LP) as plugins.
- **Fast quantile-regression solver** (interior point/ADMM, dependent-data inference, monotone rearrangement) for quantile LP.
- **Typed IRF result object and generalized-IRF engine.** All LP output flows through the shared IRF object, which must carry: normalization convention, sample-alignment policy, inference mode (lag-augmented EHW vs. HAC/HAR vs. bootstrap), and the joint cross-horizon covariance.
- **Critical-value engine** for Kiefer–Vogelsang fixed-b and simulated sup-t quantiles (cached and versioned).
- Philox-based reproducible parallel RNG, numerical optimizers, the deterministic-terms toolkit, the unified forecast object, and the golden-value validation harness (as all modules do).
- **Exogenous-regressor (covariate) contract.** LP control sets, shock series, and external instruments are all covariates in the shared sense: ingested through the aligned, leakage-checked interface (instruments and narrative shock series routinely cover shorter samples than the outcome data — alignment diagnostics catch the mismatch), with control-set construction composing with the deterministic-terms toolkit.

**Consumed from other modules:**

- **multivariate:** VAR estimation for LP-VAR dual reporting, the VAR comparator in shrinkage/model-averaging estimators, and VAR-bootstrap DGPs.
- **identification:** restriction/rotation logic where LP and SVAR share an identification scheme in dual-reporting mode.
- **ML:** penalized-regression solvers and time-series cross-validation (blocked folds) for DML-LP, double selection, and smooth-LP tuning.
- **forecasting-evaluation:** Diebold–Mariano and related forecast-comparison tests for the direct-vs-iterated forecasting API.

**Exposed to others:**

- Per-horizon and jointly-estimated IRFs (with full joint covariance) via the shared IRF object — consumed by identification (LP-side estimates), forecasting-evaluation (direct forecasts), and any module plotting IRFs with sup-t bands.
- The multiplier result type (one-step IV cumulative multipliers with honest inference).
- The Li–Plagborg-Møller–Wolf DGP battery as a library-wide Monte Carlo benchmark suite.

## Validation gallery

- **Ramey & Zubairy (2018, JPE) replication files** — linear and state-dependent fiscal multipliers (linear ≈0.6–0.7 at 2–4y horizons) from their Stata code, including the appendix AR-style intervals. The single most important target for the module.
- **Jordà (2005, AER)** — baseline LP figures must be reproduced.
- **R `lpirfs`** — `lp_lin`, `lp_nl` (Auerbach–Gorodnichenko), and `lp_lin_panel` vignette numbers matched to 3+ decimals; Stata 18 `lpirf` cross-checked on the same data.
- **Montiel Olea & Plagborg-Møller (2021) replication code** — lag-augmented coverage Monte Carlos across the persistence/horizon grid and the Gertler–Karadi application.
- **Montiel Olea & Plagborg-Møller (2019) MATLAB code** — sup-t simultaneous bands and the joint-covariance application reproduced exactly.
- **Kilian & Kim (2011, REStat) coverage tables** — the canonical bootstrap-coverage benchmark for LP intervals.
- **Gertler & Karadi (2015) FF4 high-frequency application** — LP-IV and internal-instrument IRFs.
- **Barnichon & Brownlees (2019) MATLAB replication** — smooth LP monetary application and penalty path.
- **Dube, Girardi, Jordà & Taylor Stata `lpdid`** — LP-DiD output on staggered-adoption simulations, cross-checked against Callaway–Sant'Anna where designs overlap.
- **Lusompa (2023) replication code** — LP-GLS on the Gertler–Karadi and Ramey datasets.
- **Li, Plagborg-Møller & Wolf (2024) DGP battery** — ported wholesale as a permanent test suite and public benchmark.
- **Inoue, Jordà & Kuersteiner (2023) R code** — significance bands.
- **McKay & Wolf (2023) and Barnichon & Mesters (2021, 2023) replication files** — Tier 4 gates for policy counterfactuals and optimal-policy sufficient statistics.
