# Module 06 — Structural Identification

> Part of the time series econometrics library roadmap. Master plan: [ROADMAP.md](../../ROADMAP.md).

**This module is the library's unified structural-VAR layer: every scheme for turning reduced-form dynamics into causal shocks — point identification (recursive, non-recursive, long-run, max-share), set identification (sign, zero, narrative, elasticity, and FEVD restrictions), statistical identification (heteroskedasticity and non-Gaussianity), and external/internal instruments — with matched frequentist and Bayesian inference backends, mandatory identification diagnostics, and a composable restriction algebra that lets any mix of restriction types constrain the same rotation space. It is a headline differentiator of the library: no maintained Python home for modern SVAR identification exists today.**

## Purpose and scope

Structural identification is where applied macroeconometrics actually lives. A reduced-form VAR is a forecasting device; the questions users pay for — what does a monetary tightening do to output, how large is the fiscal multiplier, what share of business cycles is demand — require mapping reduced-form innovations into economically interpretable shocks. This module owns that mapping end to end: the restriction schemes, the estimators, the samplers, and — critically — the inference that goes with each scheme, in both frequentist and Bayesian forms. The Bayesian module supplies priors and posterior samplers; this module owns the restriction and rotation logic, so a single implementation (built on the foundations rotation/restriction-algebra kernel) serves both backends and prevents two subtly different sign-restriction samplers from ever coexisting.

The audience is applied macroeconomists, central-bank staff, and PhD students who today stitch together statsmodels (Cholesky only, fragile beyond that), half a dozen incompatible R packages, and one-off MATLAB replication code. The module's design center is honesty about identification: every fit prints its diagnostics — generic-identification rank checks, acceptance rates, importance-weight effective sample sizes, instrument-strength report cards, prior-vs-posterior overlays — because in set-identified and instrument-based settings the diagnostics are the inference. Speed is a second design center: sign-, narrative-, and zero-restriction loops are embarrassingly parallel, and a Rust kernel with cached companion-form recursions plausibly buys two to three orders of magnitude over interpreted single-threaded R/MATLAB loops, changing what inference is feasible (for example, narrative restrictions with respectable effective sample sizes).

Relative to the rest of the library: the IRF/FEVD/historical-decomposition machinery is the foundations IRF object; Haar rotation sampling and restriction algebra live in foundations; LP-IV estimation mechanics live in the local-projections module (this module supplies the instruments and the structural interpretation); the Bayesian module supplies reduced-form priors and samplers. An explicit documentation deliverable of this module is the identification-scheme decision guide — "which scheme when" — walking users from question (monetary, fiscal, oil, technology, uncertainty) through data situation (instruments available? volatility regimes? credible zeros?) to a recommended scheme, its maintained assumptions, and the diagnostics that would falsify it.

## Where existing tools fall short

- **statsmodels** covers only Cholesky and basic A/B/long-run SVAR with fragile optimization: no sign restrictions, no proxy SVAR, no set-identification inference, no heteroskedasticity or non-Gaussian identification, and its SVAR error bands ignore the Waggoner-Zha normalization problem.
- **The R ecosystem is fragmented beyond repair**: `vars` (point ID), `svars` (statistical ID), `VARsignR`/`bsvarSIGNs` (sign/narrative), `BVAR`/`bvartools` (Bayesian reduced form), and `lpirfs` (LP) have incompatible APIs, inconsistent normalizations, and no shared validation suite; nothing combines restriction types.
- **The correct Arias-Rubio-Ramirez-Waggoner zero+sign algorithm** with importance weights exists essentially only in the authors' MATLAB code and a few ports; many packages still ship the naive zeroing that ARW (2018) proved distorts inference in published papers.
- **Weak-instrument-robust proxy-SVAR confidence sets** (Montiel Olea-Stock-Watson 2021) are implemented in no mainstream package; almost all applied proxy-SVAR work still reports invalid delta-method or wild-bootstrap bands (Jentsch-Lunsford showed the wild bootstrap is inconsistent here).
- **Giacomini-Kitagawa robust Bayes** for set-identified SVARs exists only as author MATLAB code; no package lets users see how much of their "posterior" is Haar-prior artifact versus data.
- **Frequentist inference on sign-identified sets** (Gafarov-Meier-Montiel Olea; Granziera-Moon-Schorfheide) has zero general-purpose implementations.
- **Identification diagnostics are absent everywhere**: no package prints RWZ (2010) rank/generic-identification checks, acceptance rates, importance-weight ESS, max-share eigengaps, instrument-strength report cards, or prior-vs-posterior overlays by default.
- **Speed**: sign- and narrative-restriction loops are embarrassingly parallel, yet R/MATLAB implementations are single-threaded and recompute IRFs per draw; a Rust kernel with cached companion recursions plausibly buys 100–1000x.
- **High-frequency identification has no software home at all**: surprise construction, GSS/Swanson factor rotations, Jarocinski-Karadi information-effect separation, and Rigobon-Sack event heteroskedasticity are all bespoke replication code.
- **Validation culture is missing**: essentially no package ships regression tests against canonical published IRFs (Uhlig 2005, Mertens-Ravn 2013, Gertler-Karadi 2015, Baumeister-Hamilton 2019, Antolin-Diaz-Rubio-Ramirez 2018) — exactly what central-bank users need to trust a new library.
- **MATLAB's BEAR toolbox** (ECB) is the closest thing to comprehensive, but it is MATLAB-licensed, monolithic, hard to script or extend, and weak on frequentist set-identified and weak-IV-robust inference; Dynare covers DSGE, not SVAR identification.

## Inventory

Difficulty ratings are from the research inventory (low / medium / high / research-grade). Three inventory items — the Haar-uniform rotation kernel, the structural IRF/FEVD/historical-decomposition engine, and LP-IV estimation mechanics — are owned elsewhere per the master-plan ownership map and appear under [Dependencies and shared infrastructure](#dependencies-and-shared-infrastructure).

### Tier 1 — Core (v1-blocking)

#### Point identification

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Recursive / Cholesky short-run identification | Lower-triangular impact matrix from the Cholesky factor of the reduced-form covariance; the default ordering-based scheme in most applied monetary VARs. | Low | Cholesky of Sigma_u; expose both unit-standard-deviation and unit-effect normalizations plus cheap ordering sweeps. Sims (1980); Christiano-Eichenbaum-Evans (1999). Trap: near-singular Sigma after over-differencing — pivoted Cholesky or fail loudly. Validate against statsmodels/R `vars` IRFs and the CEE (1999) handbook figures. |
| Non-recursive exclusion restrictions (A/B/AB-model) | General contemporaneous zero restrictions on A (simultaneous relations) and B (shock loadings) estimated by FIML; covers Sims (1986) and Bernanke (1986) systems. | Medium | Concentrated Gaussian likelihood with scoring and analytic derivatives (Amisano-Giannini 1997); check local identification via Jacobian rank. Traps: multiple local optima and sign flips — Waggoner-Zha (2003) likelihood-preserving normalization plus mandatory multiple random starts. Validate against R `vars::SVAR` and the JMulTi examples in Luetkepohl (2005). |
| Blanchard-Quah long-run restrictions | Zeros on the long-run cumulative impact matrix (e.g., demand shocks have no permanent output effect); the canonical supply/demand decomposition. | Medium | Impact matrix B = A(1) chol(A(1)^-1 Sigma A(1)^-1'). Blanchard-Quah (1989 AER). Traps: fragile when A(1) is near-singular under near-unit roots (Faust-Leeper 1997 critique — document it); cumulate IRFs correctly with differenced variables. Validate against the BQ (1989) published IRFs and statsmodels SVAR long-run mode. |
| Max-share / FEV-maximizing identification | Identify the shock explaining the maximum forecast-error-variance share of a target variable at horizon h (or summed horizons); used for technology news, TFP, MFP shocks. | Medium | Rayleigh-quotient eigenproblem: principal eigenvector of sum_h (e_i' C_h P)'(e_i' C_h P). Faust (1998); Uhlig (2003); Francis-Owyang-Roush-DiCecio (2014 REStat); Barsky-Sims (2011 JME). Traps: eigenvector sign is arbitrary (normalize by impact); eigenvalue near-degeneracy means weak identification (report the eigengap). Validate against the Barsky-Sims replication and R `bsvars` max-share examples. |

#### Set identification

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Sign restrictions: rejection sampling (Uhlig) | Accept orthogonal rotations whose IRFs satisfy sign restrictions over specified horizons; the baseline agnostic identification of monetary shocks. | Medium | Per reduced-form draw, draw Q from the foundations Haar kernel, check restrictions with early exit on first violation, parallelize across draws (Rust sweet spot). Uhlig (2005 JME). Traps: acceptance rate decays exponentially in restriction count — report it; pointwise posterior medians mix models (pair with Fry-Pagan). Validate against Uhlig (2005) Figure 6 and R `VARsignR`/`bsvarSIGNs`. |
| Zero + sign restrictions (Arias-Rubio-Ramirez-Waggoner) | The correct algorithm for combining zero restrictions with sign restrictions, sampling from the conditionally uniform distribution over admissible rotations. | High | Draw Q column-by-column in the null spaces of stacked zero-restriction matrices, then apply ARW volume-element importance weights; naive Mountford-Uhlig style zeroing samples the wrong distribution. Arias, Rubio-Ramirez, Waggoner (2018 Econometrica). Monitor importance-weight ESS. Validate against the authors' MATLAB toolbox and their replication of the Uhlig penalty application. |

#### External instruments

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Proxy SVAR / external instruments (SVAR-IV) | Identify shock columns from external instruments correlated with the target shock and uncorrelated with others; the modern applied default for monetary and tax shocks. | Medium | Single proxy: b_1 proportional to E[u z] with unit-effect normalization; multiple proxies/shocks via the Mertens-Ravn (2013 AER) closed form or minimum distance. Traps: unit-effect normalization explodes when the normalizing impact is near zero; align censored proxy samples as Mertens-Ravn do. Validate against Mertens-Ravn (2013) tax multipliers and Gertler-Karadi (2015) IRFs. |
| Weak-instrument-robust proxy-SVAR inference | Anderson-Rubin-type confidence sets for IRFs valid under weak proxies; the correct default because many published proxies are weak. | High | Invert AR statistics horizon-by-horizon: the set solves a quadratic inequality and can be an interval, a union of two rays, or the whole real line — render all three honestly; report robust first-stage F and heteroskedasticity-robust variants. Montiel Olea, Stock, Watson (2021 JoE). Validate against their Gertler-Karadi worked example. No mainstream package implements this. |
| Moving-block bootstrap for proxy SVARs | Valid frequentist bands for proxy-SVAR IRFs; replaces the wild bootstrap, which is inconsistent here. | Medium | Jointly resample residuals and proxies in recentered blocks via the foundations block-bootstrap engine. Jentsch-Lunsford (2019 AER comment; 2021 JBES). Trap: block-length selection — expose rules and sensitivity. Validate against Jentsch-Lunsford's corrected Mertens-Ravn intervals (they widen substantially). |
| Internal instruments: instrument ordered first in the VAR | Order the instrument/shock series first in a Cholesky VAR; consistent even under noninvertibility and equivalent to LP-IV estimands. | Low | Make internal-vs-external instrument a one-argument switch; docs teach the Plagborg-Moller & Wolf (2021 Econometrica) equivalence — VAR and LP estimate the same population IRFs, differing in finite-sample bias-variance. Hugely used since Ramey (2011). Validate against Plagborg-Moller-Wolf simulation results and Ramey handbook chapter figures. |

### Tier 2 — Standard (expected of a serious library)

#### Point identification

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Sims-Zha Bayesian structural VAR | Bayesian estimation of overidentified structural-form VARs with priors on structural parameters; the central-bank workhorse for non-recursive monetary systems. | High | Waggoner-Zha (2003a JEDC) Gibbs sampler drawing A0 columns from conditional Gaussians on unit spheres; Sims-Zha (1998 IER) prior. Trap: A0 columns identified only up to sign — impose likelihood-preserving normalization every draw. Validate against Sims-Zha replication files and the `szbvar` routines in R `MSBVAR` (archived). |
| Global identification rank conditions (RWZ) | Necessary and sufficient rank conditions for exact/global identification of restricted SVARs; tells the user whether their restriction set identifies before estimating. | Medium | Rubio-Ramirez, Waggoner, Zha (2010 REStud) Theorems 1–7: rank of transformed restriction matrices at a random parameter point (generic check). Trap: evaluate at several random points to avoid measure-zero false positives. A major documentation differentiator — no mainstream package exposes this. |
| Overidentification LR tests for AB-models | Likelihood-ratio test of overidentifying short-run restrictions against the just-identified model. | Low | LR = T(log det Sigma_r − log det Sigma_u), chi-square with dof = #restrictions − n(n−1)/2. Amisano-Giannini (1997). Trap: small-sample size distortion — offer bootstrap p-values. Validate against JMulTi/R `vars` output. |
| Combined short-run + long-run restrictions | Mixing contemporaneous zeros with long-run zeros (Gali 1992 IS-LM system). | Medium | Cast as a nonlinear equation system in the impact matrix; solve by Newton or via the ARW (2018) zero-restriction machinery applied to the stacked [impact; long-run] matrix. Gali (1992 QJE). Validate against Gali's published IRFs; statsmodels cannot do this at all. |
| Structural VECM / common-trends identification | Identification via cointegration structure: r transitory and n−r permanent shocks with restrictions allocated between blocks (King-Plosser-Stock-Watson). | High | Granger representation: Xi = beta_perp (alpha_perp' Gamma beta_perp)^-1 alpha_perp'; needs r(r−1)/2 plus transitory-block restrictions. Trap: rank(Xi) = n−r caps permanent shocks at n−r — enforce automatically. King-Plosser-Stock-Watson (1991 AER); Luetkepohl (2005 ch. 9). Validate against the KPSW (1991) six-variable results and JMulTi SVEC. |

#### Set identification

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Sign restrictions: penalty-function approach (Uhlig) | Choose the single rotation minimizing a penalty over sign violations/rewards; delivers a point-identified "most agreeable" shock. | Medium | Optimize over the unit sphere (single shock) with the Uhlig (2005) penalty f(x) = x for violations, scaled by IRF standard deviations; polar parameterization or manifold optimization on the sphere. Trap: penalty results are not draws from the identified set — document that inference differs from the rejection method. Validate against Uhlig (2005) penalty-based figures and Danne's `VARsignR`. |
| Fry-Pagan median-target summarization | Single rotation whose IRFs are closest to the pointwise posterior medians; fixes the "median IRFs mix different structural models" problem. | Low | Minimize squared standardized distance between candidate-draw IRFs and pointwise medians over accepted draws. Fry-Pagan (2011 JEL). Trivial once draws are stored; make it the documented default companion to any sign-restriction output. |

#### Statistical identification

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Identification via regime heteroskedasticity (Rigobon) | Identify simultaneous relations from shifts in shock variances across known regimes (crisis vs. calm), holding impact coefficients constant. | Medium | Two-regime case: joint diagonalization of Sigma_1, Sigma_2 via generalized eigendecomposition; identification requires distinct relative variances — test and report. Traps: shock labeling is arbitrary (sort by relative variance and document); misspecified regime dates bias everything. Rigobon (2003 REStat); Lanne-Luetkepohl (2008 JMCB). Validate against R `svars::id.cv`. |
| ICA-style semiparametric estimators (fastICA, distance covariance, CvM) | Estimate the rotation making shocks maximally independent without a parametric density. | Medium | Matteson-Tsay (2017 JASA) distance covariance; Herwartz (2018) CvM with bootstrap inference; fastICA as a fast default with multiple starts. Traps: whitening must use the same covariance estimator as the VAR stage; distance covariance is O(T^2) — Rust makes exact computation feasible where R packages subsample. Validate against R `svars::id.dc`. |
| Shock labeling machinery | Resolve permutation/sign indeterminacy of statistically identified shocks and attach economic labels via correlations with external series or sign patterns. | Medium | Hungarian-algorithm assignment maximizing absolute correlation with reference shocks or minimizing distance to a target impact pattern; report label stability across bootstrap/MCMC draws; Bayesian labeling via sign-pattern posterior probabilities (Anttonen-Lanne-Luoto 2024). Easy to get silently wrong — make unlabeled output impossible to plot without a warning. |

#### External instruments and high-frequency identification

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| High-frequency surprise construction toolkit | Build monetary-policy surprises from futures in event windows and rotate them into interpretable factors (target/path/LSAP) for use as instruments. | Medium | 30-minute window changes in FF futures with Kuttner (2001) day-in-month scaling, ED futures; Gurkaynak-Sack-Swanson (2005 IJCB) factor rotation; Swanson (2021 JME) three-factor identification via zero restrictions on pre-ZLB behavior; ship standard published surprise datasets (licenses checked) plus the Gertler-Karadi (2015 AEJ:Macro) monthly-aggregation conventions. Validate: reproduce posted GK and Swanson factor series to correlation > 0.99. |
| Event-day heteroskedasticity identification (Rigobon-Sack) | Identify policy effects from the jump in surprise variance on announcement days versus control days; robust to announcement-day background noise that biases event studies. | Medium | IV/GMM on the difference of covariance matrices between event and non-event samples; equivalent to IV with dummy-interacted instruments. Trap: negative-definite covariance differences in small samples — report and bound. Rigobon-Sack (2003 QJE; 2004 JME). Validate against Rigobon-Sack (2004) asset-price responses. |
| Instrument diagnostics: relevance, exogeneity, stability | First-stage strength (robust/effective F), overidentification-based exogeneity tests with multiple proxies, and time-stability tests of instrument relevance. | Medium | Effective first-stage F (Montiel Olea-Pflueger style adapted to proxy SVARs); Mertens-Ravn overID when #proxies > #shocks; rolling relevance per Hoesch-Rossi-Sekhposyan (2023 JAE) evidence that Romer-Romer and GK instrument strength varies over time. Ship as a mandatory "identification report card" printed with every proxy-SVAR fit. |

#### Application conventions

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Narrative shock conventions (Romer-Romer, Ramey news, tax narratives) | Canonical narrative series and their standard usage patterns: Romer-Romer monetary and tax shocks, Ramey defense news, Ramey-Zubairy military spending news. | Low | Documented loaders for published series (RR 2004 AER monetary; RR 2010 AER tax; Ramey 2011 QJE; Ramey-Zubairy 2018 JPE) with vintage/version metadata, plus recipes: direct regressors (Jorda LP), proxies, or ordered-first internal instruments. Validate: reproduce the Ramey (2016 Handbook) comparison figures. Low difficulty, huge user value. |
| Blanchard-Perotti fiscal identification preset | Fiscal SVAR with institutional output elasticities of taxes (2.08) fixed a priori and spending predetermined; the fiscal-multiplier benchmark. | Low | A calibrated non-recursive AB-model with selected A elements user-fixed rather than estimated; support quarter-dependent elasticities and anticipated-policy caveats. Blanchard-Perotti (2002 QJE). Validate against their multipliers and later corrected estimates; docs contrast with Ramey news and Mertens-Ravn proxies as a "which fiscal ID when" guide. |

### Tier 3 — Advanced (differentiators)

#### Point identification

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Medium-run / spectral restrictions | Identify a shock by its contribution over a frequency band or medium horizon rather than at zero or infinite horizon. | Medium | Maximize variance contribution integrated over a spectral band [2pi/32, 2pi/6] using the VAR spectral density; reduces to a symmetric eigenproblem. Trap: numerical integration of the spectrum — Gauss-Legendre nodes, verify Parseval consistency with the FEVD. Uhlig (2004, "What moves GNP?"). |
| Business-cycle anatomy frequency-domain max-share | Max-share over business-cycle frequencies (6–32 quarters) in the spectral domain; the "main business cycle shock." | Medium | Same eigenproblem with a spectral-band weighting matrix; must match the paper's exact frequency band and integration scheme to replicate. Angeletos-Collard-Dellas (2020 AER). Validate against their posted replication IRFs. Almost no library ships this; cheap once the spectral kernel exists. |

#### Set identification

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Narrative sign restrictions | Restrict the sign of specific structural shocks and/or their historical-decomposition contribution in named episodes (e.g., Volcker 1979); sharpens sign-identified sets dramatically. | High | Rejection plus importance sampling with weight 1/Pr(narrative constraints hold given the draw), estimated by simulating shocks. Trap: heavy-tailed weights — report ESS and use large simulation counts. Antolin-Diaz & Rubio-Ramirez (2018 AER). Validate against their monetary and oil applications; cross-check with R `bsvarSIGNs` (2024), the only package implementation. |
| Elasticity / magnitude / ranking bounds | Inequality restrictions beyond signs: bounds on impact elasticities (Kilian-Murphy oil supply), relative magnitudes, and shock rankings. | Medium | General linear/nonlinear inequality checks on functions of (B, Q) inside the accept/reject loop; support ratios of IRF elements with care when denominators near zero. Kilian-Murphy (2012 JEEA; 2014 JAE); Amir-Ahmadi & Drautzburg (2021 QE) for rankings. Validate against the Kilian-Murphy (2014) oil-market posterior. |
| Explicit structural priors (Baumeister-Hamilton) | Full Bayesian inference with priors directly on structural parameters (e.g., supply/demand elasticities) instead of the Haar prior on rotations; the principled answer to the "uniform prior is informative" critique. | High | Random-walk Metropolis on A0 parameterized by economically meaningful elasticities with truncated Student-t priors; analytic conditional posteriors for lag coefficients and variances. Trap: prior-to-posterior updating for set-identified parameters never vanishes asymptotically — ship prior/posterior overlay plots by default. Baumeister-Hamilton (2015 Econometrica; 2019 AER oil). Validate against their posted oil-market posterior quantiles. |

#### Statistical identification

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Markov-switching variance SVAR | Latent regime-switching in structural shock variances identifies the impact matrix without economic restrictions; economic restrictions become testable. | High | EM or Bayesian estimation with the Hamilton filter; B constant across states, diagonal Lambda_s state-dependent. Traps: likelihood multimodality (many EM starts), label switching across states and shocks, weak identification when relative variances are similar — implement the Luetkepohl et al. identification test. Lanne-Luetkepohl-Maciejowska (2010 JEDC); Herwartz-Luetkepohl (2014 JoE). Validate against the Herwartz-Luetkepohl (2014) US monetary system results. |
| Smooth-transition and GARCH covariance identification | Identification from smoothly time-varying variances (ST-SVAR) or conditionally heteroskedastic shocks (GARCH-SVAR). | High | ST: Luetkepohl-Netsunajev (2017) logistic transition in Lambda(t). GARCH: Normandin-Phaneuf (2004 JME); Lanne-Saikkonen (2007) — GO-GARCH-like structure, ML with univariate GARCH(1,1) per shock. Traps: GARCH likelihood ill-conditioned in small samples; transition-speed parameter poorly identified (profile it). Validate against R `svars::id.st` and the Luetkepohl-Netsunajev replication. |
| Non-Gaussian maximum likelihood SVAR | Mutually independent non-Gaussian shocks identify B up to permutation/scale/sign (ICA theorem); ML with parametric shock densities. | High | Identification fails if more than one shock is Gaussian (Comon 1994) — test Gaussianity per shock and warn. Lanne-Meitz-Saikkonen (2017 JoE) asymptotics; pseudo-ML consistency conditions in Gourieroux-Monfort-Renne (2017 REStud). Traps: unbounded/multimodal likelihood (df near boundary); permutation-sign-scale indeterminacy — fixed diagonal scaling plus systematic relabeling. Validate against R `svars::id.ngml`. |

#### External instruments and high-frequency identification

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Bayesian proxy SVAR | Posterior inference treating the proxy equation as part of the likelihood, or via zero restrictions in an augmented system. | High | Caldara-Herbst (2019 AEJ:Macro): proxy = relevance × shock + noise, Metropolis or SMC — the prior on relevance matters when the proxy is weak, so prior sensitivity is required. Arias-Rubio-Ramirez-Waggoner (2021 JME): proxy-augmented SVAR with zero/sign restrictions via their importance-sampling machinery. Validate against the Caldara-Herbst monetary application. Giacomini-Kitagawa-Read (2022 JoE) robust-Bayes weak-proxy version is a frontier add-on. |
| Central-bank information-effect handling | Separate pure policy shocks from information shocks: sign restrictions on rate/stock-price surprise comovement and Greenbook-orthogonalized instruments. | Medium | Jarocinski-Karadi (2020 AEJ:Macro): high-frequency comovement sign restrictions inside a VAR containing both surprises (reuses the sign-restriction kernel), plus their "poor man's" median split. Miranda-Agrippino-Ricco (2021 AEJ:Macro): project surprises on central-bank forecasts/revisions, use the residual as instrument. Validate against both papers' IRFs. Docs present this as the default modern monetary-ID recipe. |

#### Combination frameworks and shared machinery

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Combining identification schemes (proxy + sign/zero + narrative + heteroskedasticity) | Unified framework in which any mix of restriction types constrains the same rotation space; the module's architectural centerpiece. | High | Restrictions as composable predicates/weights on (reduced-form draw, Q): zeros shape the Q-sampling null spaces (ARW), signs/narratives/FEVD bounds act as accept-reject or importance weights, proxies enter as moment conditions or augmented blocks. Braun-Bruggemann (2023 JoE) combine proxies with signs; Schlaak-Rieth-Podstawski (2023 JAE) combine proxy with heteroskedasticity. Trap: each combination changes the effective prior — always emit prior-vs-posterior and ESS diagnostics. This composability does not exist anywhere today. |
| Identified-set optimization kernel | Min/max of smooth functionals (IRF at horizon h, FEVD share) over rotations subject to sign/zero constraints; powers Giacomini-Kitagawa bounds and frequentist set inference. | High | One restricted column: sphere-intersect-half-spaces with analytic active-set solutions (Gafarov-Meier-Montiel Olea 2018); full Q: manifold optimization (Cayley/geodesic steps on O(n)) with many random Haar starts; verify global optima against dense random rotation sampling. This kernel plus Rust parallelism is a genuine moat. |
| Fundamentalness/invertibility diagnostics | Tests for whether structural shocks are recoverable from the VAR's variables — the maintained assumption behind all internal SVAR identification. | Medium | Forni-Gambetti (2014 JME) sufficient-information test: regress candidate shocks/orthogonalized residuals on lagged factor-model principal components, F-test. Docs teach the fixes: add factors/FAVAR, or use internal instruments, which remain valid under noninvertibility per Plagborg-Moller & Wolf. Also report the Wolf (2020 AEJ:Macro) "masquerading shocks" diagnostic for sign restrictions. |

### Tier 4 — Frontier (research-grade, each gated on a named validation target)

#### Set identification and robust inference

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| FEVD and unconditional-variance restrictions | Bound the forecast-error-variance share a shock explains (e.g., monetary shocks explain <30% of output variance) to shrink identified sets. | Medium | FEVD-share inequality checks in the rejection loop with cached companion-power recursions per reduced-form draw. Volpicella (2022 JBES); also supports Plagborg-Moller & Wolf (2022 JPE) instrumented variance-decomposition bounds. No mainstream package has this. Gate: reproduce the Volpicella (2022) application. |
| Robust Bayes for set-identified SVARs (Giacomini-Kitagawa) | Prior-robust posterior bounds over the class of all priors consistent with the identified set; separates information in the data from the prior on rotations. | Research-grade | Per reduced-form draw, min/max the IRF over admissible Q (nonconvex constrained optimization on the orthogonal group/sphere — many random starts plus analytic active-set solutions where available), then aggregate bounds across draws. Giacomini-Kitagawa (2021 Econometrica); Giacomini-Kitagawa-Read (2021 survey). Headline differentiator — no general-purpose package ships this. Gate: match the authors' MATLAB code results. |
| Frequentist inference on sign-identified sets | Confidence sets for identified sets of IRFs under sign/zero restrictions: delta-method bounds and projection inference. | Research-grade | Gafarov-Meier-Montiel Olea (2018 JoE): closed-form active-set IRF bounds with delta-method CIs — watch nondifferentiability when the active set switches (check regularity). Granziera-Moon-Schorfheide (2018 QE): moment-inequality projection, conservative but uniformly valid. Moon-Schorfheide (2012 Econometrica) for Bayes/frequentist divergence under set ID. Gate: reproduce Gafarov-Meier-Montiel Olea's posted examples. |

#### Statistical identification

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Stochastic-volatility and time-varying-volatility identification | Identification from continuous volatility variation without discrete regimes: SV-SVAR and Lewis's moment-based TVV estimator. | Research-grade | Bertsche-Braun (2022 JBES): SV per structural shock via MCMC or EIS-ML. Lewis (2021 REStud): GMM on autocovariances of squared reduced-form residuals — frequentist, no regime dating; needs HAC-robust weighting and careful moment selection. Trap: weak identification when volatility paths comove. Nobody ships this. Gate: reproduce the Lewis (2021) fiscal application. |
| Higher-moment GMM identification (coskewness/cokurtosis) | GMM on third/fourth-order cross-moment conditions of structural shocks; allows overidentification tests and mixing with economic restrictions. | High | Keweloh (2021 JBES): conditions E[e_i^2 e_j] = 0, E[e_i^3 e_j] = 0, etc., continuously-updated GMM with analytic derivatives; Guay (2021 JoE): cumulant selection and rank tests for identification strength. Traps: fourth moments extremely noisy in small T; weight-matrix conditioning — offer an identity-weighted first step and moment-selection diagnostics. Gate: match Keweloh's replication files. |
| Statistical-identification diagnostics and robustness tests | Test battery deciding whether data-driven identification is trustworthy: variance-regime equality tests, shock normality/independence tests, robustness of higher-moment ID. | High | LR/Wald tests for distinct relative variances; Jarque-Bera and multivariate normality per shock; the Montiel Olea-Plagborg-Moller-Qian (2022 AEA P&P) caution that higher-moment ID rests on strong exclusion-type independence assumptions — implement their diagnostic framing; Drautzburg-Wright (2023 REStud) independence-relaxing bounds as the robust alternative. Documentation must teach when NOT to trust statistical ID. Gate: reproduce the Drautzburg-Wright (2023) bounds application. |
| Identification through structural breaks in the covariance | Use documented breaks (known dates) in both coefficients and covariances to identify structural parameters, allowing impact responses to change across regimes. | High | Bacchiocchi-Fanelli (2015 OBES): generalizes Rigobon by letting B shift subject to cross-regime restrictions; ML needs careful parameter mapping. Trap: break dates are treated as known — offer sensitivity sweeps. Gate: reproduce the Bacchiocchi-Fanelli monetary application. |

#### External instruments

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Bounds under imperfect/contaminated proxies | Set identification when the instrument may be endogenous or only event-constrained. | Research-grade | Ludvigson-Ma-Ng (2021, "Shock Restricted SVARs"): inequality constraints tying structural shocks to external events (magnitude/correlation constraints) yield identified sets — implement via the rejection kernel with shock-path constraints (reuses the narrative machinery). Trap: constraints on realized shock paths depend on estimated reduced-form residuals — propagate that uncertainty. Gate: reproduce the Ludvigson-Ma-Ng uncertainty-shock application. |

#### Applications and beyond linear SVARs

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Structural scenario analysis and conditional forecasting with structural shocks | Conditional-on-path forecasts implemented via identified structural shocks with minimal-distortion shock selection; the central-bank stress-testing workhorse. | Medium | Antolin-Diaz, Petrella, Rubio-Ramirez (2021 JME): choose future shock distributions to satisfy path constraints while penalizing KL deviation from unconditionality; reduces to Gaussian conditioning per draw. Trap: report which shocks "drive" the scenario. Gate: reproduce their replication, with Waggoner-Zha (1999) conditional forecasts as the special case. |
| Nonlinear/state-dependent structural estimands and counterfactual tooling | State-dependent identification (fiscal multipliers in recessions), robustness of linear IRF estimands under nonlinearity, and shock-based policy counterfactuals. | Research-grade | Ramey-Zubairy (2018 JPE) state-dependent LP-IV as the applied baseline (LP mechanics consumed from the LP module); Goncalves-Herrera-Kilian-Pesavento (2021 JoE) on when state-dependent LPs identify meaningful objects; linear-estimands-as-weighted-averages results for the docs; McKay-Wolf (2023 Econometrica) counterfactuals from combinations of identified shocks as a downstream consumer of every ID scheme in the library. v1 scope: LP-based state dependence plus documentation. Gate: reproduce Ramey-Zubairy (2018) state-dependent multipliers. |

## Frontier watchlist

- Robust-Bayes treatment of narrative sign restrictions with honest importance-weight/ESS handling — Giacomini-Kitagawa-Read (2022 JoE) extension of the Tier 3 narrative machinery.
- Robust Bayes for proxy SVARs under weak-proxy ambiguity — Giacomini-Kitagawa-Read (2022 JoE), flagged as the frontier add-on to the Tier 3 Bayesian proxy SVAR.
- Non-Gaussian separation of Fed announcement shocks — Jarocinski (2024 JME), a statistical-ID upgrade to the information-effect recipe.
- Unified LP/VAR equivalence tooling as a first-class diagnostic layer — Plagborg-Moller & Wolf (2021 Econometrica), built jointly with the LP module.
- McKay-Wolf (2023 Econometrica) policy counterfactuals as a full standalone consumer of every identification scheme (v1 ships documentation only; see the Tier 4 nonlinear item).

## Implementation warnings

The "easy to get statistically or numerically wrong" list. Every item here has bitten a published paper or a shipping package.

1. **Haar sampling.** QR-based orthogonal draws are uniform ONLY if the R factor's diagonal is normalized to be positive (Stewart 1980; Mezzadri 2007); raw LAPACK QR gives non-Haar draws and silently biases every sign-restriction posterior. Uniform Givens angles are NOT Haar for n >= 3.
2. **Zero + sign restrictions.** Imposing zeros by construction and then sign-checking WITHOUT the Arias-Rubio-Ramirez-Waggoner (2018) volume-element importance weights samples the wrong distribution — this exact error appeared in published papers. Make the corrected algorithm the only code path.
3. **The Haar prior is informative.** The uniform prior on rotations is informative about IRFs (Baumeister-Hamilton 2015): never present sign-restriction posteriors without prior-vs-posterior overlays and, ideally, Giacomini-Kitagawa robust bounds alongside.
4. **Median IRFs mix models.** Pointwise posterior median IRFs under set identification mix mutually inconsistent structural models (Fry-Pagan 2011); ship median-target rotations and label pointwise bands honestly.
5. **Proxy-SVAR inference.** The Mertens-Ravn wild bootstrap is asymptotically invalid (Jentsch-Lunsford 2019); use the moving-block bootstrap or MSW weak-IV-robust sets. Delta-method bands are additionally invalid under weak proxies — check effective first-stage strength before reporting them.
6. **Unit-effect normalization fragility.** The normalization divides by an impact coefficient that can be near zero in some draws, producing exploding IRF quantiles; detect and report normalization fragility (this afflicts sign-restricted and proxy models alike).
7. **Long-run identification fragility.** Blanchard-Quah identification is fragile to near-unit roots: A(1)^-1 amplifies small-sample bias enormously (Faust-Leeper 1997); prefer the VECM formulation when cointegration is plausible and warn when eigenvalue moduli approach 1.
8. **Likelihood indeterminacy.** Likelihood-based structural estimation (AB-models, Sims-Zha, Markov-switching, non-Gaussian ML) suffers sign/permutation/scale indeterminacy and multimodality: always run many random starts and apply likelihood-preserving normalization (Waggoner-Zha 2003) — naive diagonal normalization distorts error bands.
9. **Statistical ID fails quietly.** Heteroskedasticity- and non-Gaussianity-based identification fails when regimes have similar relative variances or more than one shock is Gaussian: identification-strength tests must run automatically and gate the output.
10. **Narrative importance weights are heavy-tailed.** Report effective sample size and refuse to summarize posteriors when ESS is tiny — otherwise results are dominated by a handful of draws.
11. **Acceptance rates decay exponentially** in the number of sign restrictions; report them (an identification diagnostic in itself) and use early-exit restriction checking with cached MA-coefficient recursions — never recompute IRFs from scratch per Q draw.
12. **IRF computation.** Use companion-form recursions, not eigendecomposition (companion matrices are frequently defective); historical decompositions must satisfy the adding-up identity including the initial-condition term — unit-test the identity to machine precision.
13. **Nondifferentiable set bounds.** Frequentist bounds on identified sets are nondifferentiable where the active set of binding sign restrictions switches; delta-method CIs (Gafarov-Meier-Montiel Olea) require regularity checks, and projection methods are uniformly valid but conservative — implement both and say so.
14. **Reproducibility under parallelism.** Accept-reject across threads must use counter-based/jump-ahead RNG streams keyed by draw index so results are bitwise reproducible regardless of thread count — essential for the validation suite.
15. **Small-sample behavior of moment objectives.** Distance-covariance objectives are O(T^2) and fourth-moment estimates are extremely noisy; document small-sample behavior and provide bootstrap inference rather than asymptotics for T < 300.
16. **High-frequency surprise conventions.** Kuttner day-in-month scaling for FF futures, unscheduled-meeting handling, and monthly-aggregation conventions (Gertler-Karadi) all change results materially; ship exact published conventions with versioned datasets and reproduce published series to correlation > 0.99 in CI.

## Dependencies and shared infrastructure

**Consumed from foundations:**

- **Haar-rotation/restriction-algebra kernel** (CONSUMED; this module is its primary consumer and spec driver). Needs: QR-of-Gaussian Haar draws with the R-diagonal sign normalization, Householder single-column draws on the sphere for one-shock schemes, column-by-column null-space draws for zero restrictions, and uniformity unit tests via moments of Q entries. A single shared kernel is what prevents two subtly different sign-restriction samplers from existing in the library.
- **IRF/FEVD/historical-decomposition engine** (CONSUMED; the typed IRF result object and generalized-IRF engine live in foundations). Needs: companion-form recursions (never eigendecomposition), per-draw caching of reduced-form MA coefficients so restriction checking costs only matrix-vector work per Q, early-exit hooks for restriction evaluation, and the historical-decomposition adding-up identity enforced to machine precision. Validation: FEVD/HD digit-for-digit against R `vars` and the BEAR toolbox.
- **Resampling/bootstrap engine** (CONSUMED): moving-block bootstrap primitives and block-length selection for proxy-SVAR inference; bootstrap p-values for overidentification tests.
- **Philox-based reproducible parallel RNG** (CONSUMED): counter-based substreams keyed by draw index for bitwise-reproducible accept-reject across any thread count.
- **Exogenous-regressor (covariate) contract** (CONSUMED): proxy/instrument series — which typically cover shorter samples than the VAR data and contain missing values — are aligned to estimation samples through the shared covariate interface with loud alignment diagnostics, so sample-mismatch between instrument and VAR residuals (a classic proxy-SVAR bug) is caught at ingestion rather than silently truncated.
- **Critical-value engine, numerical optimizers, deterministic-terms toolkit, innovation-distribution zoo** (CONSUMED): manifold/constrained optimizers for penalty and identified-set problems; Student-t and skewed densities for non-Gaussian ML.

**Consumed from other modules:**

- **Bayesian module**: reduced-form priors and posterior samplers (Minnesota-family, SMC, Gibbs infrastructure). This module owns everything from the reduced-form draw onward: restrictions, rotations, importance weighting, and structural summaries.
- **Multivariate module**: reduced-form VAR/VECM estimation objects that all identification schemes attach to.
- **LP module**: LP-IV estimation mechanics (2SLS per horizon with Newey-West or lag-augmented inference, weak-IV robust AR bands per horizon; Stock-Watson 2018 EJ; Jorda 2005; Montiel Olea-Plagborg-Moller 2021 Ecta) — the inventory's LP-IV structural-inference item lives there. This module supplies instruments, shock series, and the structural-interpretation and VAR/LP-equivalence documentation, and validates jointly against R `lpirfs` and Ramey-Zubairy (2018) multipliers. State-dependent LP mechanics for the Tier 4 nonlinear item are likewise consumed from the LP module.
- **Golden-value validation harness** (library-wide): every canonical replication below runs in CI.

**Exposed to other modules:**

- Identified structural models: labeled shock series, rotation draws with weights, structural IRF/FEVD/HD results in the unified forecast/IRF objects — consumed by the LP module (instrument supply), nowcasting (structural interpretation), and the McKay-Wolf counterfactual tooling.
- The identification report card (rank checks, acceptance rates, ESS, instrument strength, normalization fragility) as a reusable diagnostic pattern.
- High-frequency surprise datasets and narrative shock series with vintage metadata (backed by the foundations real-time vintage store).
- The identification-scheme decision guide ("which scheme when") as a flagship documentation deliverable.

## Validation gallery

Golden targets this module must reproduce in CI before the corresponding feature ships:

- **Uhlig (2005) Figure 6** — rejection-sampling sign-restricted monetary IRFs match the published figure; penalty-function figures match the penalty variant.
- **Arias-Rubio-Ramirez-Waggoner (2018) MATLAB toolbox** — zero+sign importance-weighted draws replicate their corrected Uhlig penalty application.
- **Mertens-Ravn (2013) tax multipliers** — proxy-SVAR point estimates match; Jentsch-Lunsford corrected moving-block-bootstrap intervals match (and widen substantially versus the invalid wild bootstrap).
- **Gertler-Karadi (2015) IRFs and the Montiel Olea-Stock-Watson (2021) worked example** — proxy-SVAR IRFs and weak-IV-robust AR confidence sets match.
- **Baumeister-Hamilton (2019) oil-market posterior quantiles** — structural-prior Bayesian inference matches their posted results.
- **Antolin-Diaz & Rubio-Ramirez (2018) monetary and oil applications** — narrative-restriction posteriors match, cross-checked against R `bsvarSIGNs`.
- **Blanchard-Quah (1989) published IRFs** — long-run identification matches, cross-checked against statsmodels' long-run mode.
- **King-Plosser-Stock-Watson (1991) six-variable results** — structural VECM matches, cross-checked against JMulTi SVEC.
- **Barsky-Sims (2011) news-shock replication** — max-share identification matches, cross-checked against R `bsvars`.
- **Angeletos-Collard-Dellas (2020) posted replication IRFs** — frequency-domain max-share matches their band and integration scheme.
- **R `svars` outputs (`id.cv`, `id.st`, `id.dc`, `id.ngml`)** — heteroskedasticity, smooth-transition, distance-covariance, and non-Gaussian ML estimators match.
- **Gurkaynak-Sack-Swanson / Swanson / Gertler-Karadi surprise series** — constructed factors and monthly aggregates correlate > 0.99 with posted series.
- **Ramey (2016 Handbook) comparison figures** — narrative-shock usage recipes reproduce the handbook comparisons.
- **Giacomini-Kitagawa (2021) MATLAB results** — robust-Bayes posterior bounds match (Tier 4 gate).
- **Lewis (2021) fiscal application** — time-varying-volatility GMM estimator matches (Tier 4 gate).
- **CEE (1999) handbook figures and R `vars`/BEAR toolbox output** — recursive IRFs, FEVDs, and historical decompositions match digit-for-digit.
