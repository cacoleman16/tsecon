# Module 03 — Volatility and Risk

> Part of the time series econometrics library roadmap. Master plan: [ROADMAP.md](../../ROADMAP.md).

**This module delivers the complete conditional-variance and tail-risk stack: the univariate GARCH family and its asymmetric, long-memory, component, score-driven, and Markov-switching extensions; stochastic volatility by MCMC and particle methods; realized-measure construction from tick data through HAR, Realized GARCH, and HEAVY models; multivariate volatility from CCC/DCC through BEKK, factor structures, and vast-dimensional composite-likelihood estimators; and a unified VaR/ES forecasting and backtesting layer — all on a Rust core whose parallel simulation, bootstrap, and MCMC throughput makes workflows interactive that are overnight jobs in existing tools.**

## Purpose and scope

The module covers everything between a return series and a defensible risk number. On the modeling side that means the univariate ARCH/GARCH core (Engle 1982; Bollerslev 1986) and its asymmetric (EGARCH, GJR, TGARCH, APARCH), long-memory (FIGARCH, FIEGARCH), component and multi-frequency (CGARCH, Spline-GARCH, GARCH-MIDAS, MF2-GARCH), score-driven (GAS, Beta-t-EGARCH), regime-switching (MS-GARCH), and stochastic-volatility branches; on the data side, a high-frequency pipeline that turns raw ticks into cleaned, noise-robust, jump-tested realized measures feeding HAR, Realized GARCH, and HEAVY models; on the cross-sectional side, multivariate volatility from CCC and DCC (with correct two-step inference) through cDCC, BEKK, DECO, GO-GARCH, and composite-likelihood methods for thousands of assets; and on the decision side, a unified VaR/ES layer with the full modern backtesting battery, from Kupiec to Fissler-Ziegel joint scoring.

The audiences are risk managers producing regulatory VaR/ES, empirical asset-pricing researchers estimating and comparing volatility models, and macro-finance economists linking volatility to low-frequency drivers. The design premise is that a fitted volatility model, its forecasts, its risk measures, and their backtests are one connected object model — the fragmentation of this workflow across a dozen mutually incompatible packages is the single largest gap in existing tooling.

Relative to the rest of the library: innovation distributions, bootstrap machinery, quantile-regression solvers, and optimizers are consumed from foundations; forecast-comparison tests (DM/GW/MCS/SPA) and density-forecast evaluation are consumed from and re-exported via forecasting-evaluation; MIDAS weighting utilities come from nowcasting. One explicit scope ruling: vine and factor copulas for high-dimensional dependence are **deferred** — the module ships static and GAS-dynamic copulas with GARCH margins and documents interoperability with pyvinecopulib for vine structures.

## Where existing tools fall short

- statsmodels has essentially no volatility modeling; the entire Python ecosystem rests on `arch`, which is univariate-only — there is no production-quality DCC, BEKK, GO-GARCH, DECO, or copula-GARCH anywhere in Python.
- `arch` lacks GARCH-in-mean, Realized GARCH, HEAVY, GARCH-MIDAS, Markov-switching GARCH, GAS/Beta-t-EGARCH, stochastic volatility, and all ES backtesting; its forecast API for asymmetric models leans on simulation with limited distributional output.
- `rugarch`/`rmgarch` are effectively in maintenance mode, single-threaded on the expensive paths (simulation, rolling backtests), GPL-licensed, and `rmgarch` covers only DCC/GO-GARCH/copula-GARCH — no BEKK, no DECO, no realized-covariance models; documented bugs in cDCC and in two-step standard errors persist.
- The field is fragmented across a dozen niche packages (stochvol, MSGARCH, GAS, mfGARCH, highfrequency, esback, MCS, VineCopula), each with different data structures, so a GARCH fit cannot flow into an ES backtest or a model confidence set without glue code; no unified forecast/backtest object model exists anywhere.
- Two-step DCC standard errors that ignore first-stage estimation uncertainty are the norm in shipped software; correct stacked-GMM inference (Engle-Sheppard) is essentially unavailable outside author code.
- Modern ES and tail evaluation (Fissler-Ziegel joint scores, Murphy diagrams, Nolde-Ziegel comparative backtests, ESR tests) exists only in small R packages disconnected from any fitting engine, and not at all in Python.
- Vast-dimensional methods — composite-likelihood DCC (Pakel et al. 2021) and DCC with nonlinear shrinkage (Engle-Ledoit-Wolf 2019) — exist only as author MATLAB code despite being the state of practice at quant funds.
- Realized-measure construction (realized kernels, pre-averaging, semivariances, refresh-time synchronization) is separated from the models that consume the measures; Python has no maintained realized-kernel implementation at all.
- Cross-package replication is poor: `arch`, `rugarch`, and the Oxford MFE toolbox give different answers for the same GARCH(1,1) because of undocumented variance-initialization, distribution-parameterization (Hansen vs Fernandez-Steel skew-t), and constraint conventions — no package ships a published-results validation suite.
- Monte Carlo and bootstrap workflows (bootstrap prediction intervals, MCS over 100 models, boundary-robust bootstrap inference, particle MCMC for SV) are prohibitively slow in interpreted implementations — precisely where a Rust core with parallel path simulation changes what is feasible.
- No existing tool teaches model selection; `rugarch`'s docs are reference-only and `arch`'s are API-focused, so the "which volatility model when" knowledge lives in survey papers rather than software documentation.

## Inventory

### Tier 1 — Core (v1-blocking)

#### Univariate GARCH core

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| ARCH(q) (Engle 1982) | Variance as a linear function of q lagged squared residuals; mostly pedagogical but the foundation of the stack and used in LM testing. | Low | Simple recursion + QMLE; ship together with the ARCH-LM test. Engle (1982). Validate against Python `arch` and `rugarch` with identical variance initialization (backcast) — initialization is the usual source of cross-package discrepancy. |
| GARCH(p,q) (Bollerslev 1986) | The workhorse conditional-variance model; everything else in the domain builds on its recursion, forecasting, and simulation machinery. | Medium | O(T) recursion; QMLE with Bollerslev-Wooldridge sandwich SEs; analytic derivatives per Fiorentini-Calzolari-Panattoni (1996). Traps: variance initialization (backcast, unconditional, user-set), positivity/stationarity constraints, flat likelihood near beta = 1. Validate GARCH(1,1) on S&P 500 against `arch`, `rugarch`, and the MFE MATLAB toolbox to 6 decimals; reproduce Bollerslev (1986) inflation example. |
| IGARCH & RiskMetrics EWMA | Integrated GARCH (alpha + beta = 1) and the fixed-lambda EWMA used throughout industry risk systems. | Low | Impose the unit-sum constraint via reparameterization; unconditional variance does not exist, so variance targeting must auto-disable and multi-step forecasts are flat + linear. Lambda = 0.94 daily / 0.97 monthly (J.P. Morgan RiskMetrics Technical Document 1996); also implement RiskMetrics 2006 (Zumbach) long-memory EWMA. Validate EWMA VaR against the RM technical document tables. |
| QMLE engine with robust SEs and analytic derivatives | Shared estimation core: constrained/transformed optimization, analytic gradients, Hessian + OPG + Bollerslev-Wooldridge (1992) sandwich covariance. | High | Default to sandwich SEs (Hessian-only is wrong under non-normal errors). Multi-start over persistence and alpha share; scale-check inputs and warn on decimal-unit returns; delta-method SEs on the natural scale. Boundary detection (alpha near 0, persistence near 1) must warn that standard asymptotics fail (Andrews 2001). Validate SEs against `arch` (robust) and Fiorentini-Calzolari-Panattoni (1996) derivative benchmarks. |
| Multi-step forecasting engine | h-step variance and return-distribution forecasts: closed-form where they exist (GARCH, GJR under symmetric z), simulation otherwise, plus filtered historical simulation and bootstrap prediction intervals. | Medium | Analytic recursions need E[z^2 1(z<0)] and E&#124;z&#124; under the actual innovation law — precompute per distribution. Bootstrap prediction intervals per Pascual-Romo-Ruiz (2006). Simulation must reuse the fitted state (last h, last eps) exactly as `arch` does. Validate: analytic vs 10M-path simulation agreement for GARCH/GJR; EGARCH forecasts vs `arch`'s simulation method. |
| Parallel simulation / Monte Carlo primitives | Vectorized, multithreaded path simulation for any fitted model with common random numbers and antithetic variates; the speed pillar for bootstrap backtests and MC studies. | Medium | Rust rayon over paths; counter-based RNG (Philox, from foundations) so paths are reproducible and CRN across models is trivial. Burn-in control, variance-explosion caps near IGARCH, exact re-seeding from Python. Benchmark target: >100x `rugarch` ugarchsim throughput on 1e5 paths. |
| Residual diagnostics battery | Post-fit adequacy tests every referee expects: ARCH-LM, Ljung-Box on z and z^2, Nyblom (1989) stability, Engle-Ng (1993) sign-bias. | Low | Ljung-Box on standardized residuals of a fitted model has a nonstandard distribution — offer the Li-Mak (1994) corrected test, which `rugarch` omits doing properly. Nyblom per-parameter and joint. Validate against `rugarch`'s show() diagnostic block. |

#### Asymmetric & nonlinear GARCH

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| EGARCH (Nelson 1991) | Log-variance specification with asymmetric response to signed shocks; no positivity constraints, the standard choice when leverage matters and constraints bind. | Medium | Recursion in log h with g(z) = theta·z + gamma·(&#124;z&#124; − E&#124;z&#124;); E&#124;z&#124; is distribution-specific (sqrt(2/pi) normal; closed forms for t/GED/skew-t). Clamp log h against overflow. Multi-step forecasts have no closed form — simulate (E[h] ≠ exp(E[log h])). Nelson (1991). Validate against `arch` on S&P 500 and `rugarch` (parameterization order differs — document the mapping). |
| GJR-GARCH (Glosten-Jagannathan-Runkle 1993) | Threshold GARCH with an indicator-weighted squared shock for negative returns; the most-used asymmetric model in applied finance. | Low | Stationarity: alpha + beta + gamma·P(z<0) < 1, with P(z<0) = 0.5 only for symmetric z — compute under skew-t when used. Multi-step forecasts need E[z^2 1(z<0)]. Glosten-Jagannathan-Runkle (1993). Validate vs `arch` GJR and Engle-Ng (1993) news-impact comparisons. |
| TGARCH / ZARCH (Zakoian 1994) | Threshold model on the conditional standard deviation; more outlier-robust than squared-shock recursions. | Medium | Dynamics in sigma, not sigma^2; likelihood non-differentiable at zero residuals — use subgradient-safe optimization or smooth &#124;x&#124; slightly. Forecasts require moments of &#124;z&#124; and z·1(z<0). Zakoian (1994). Validate vs `rugarch` fGARCH (TGARCH submodel) and `arch`'s ZARCH. |

#### Realized volatility models

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| HAR-RV (Corsi 2009) | Heterogeneous autoregression of RV on daily/weekly/monthly averages; OLS-estimable and still the benchmark to beat for RV forecasting. | Low | OLS with HAC or Patton-Sheppard WLS; log and sqrt transforms with bias-corrected back-transformation; direct vs iterated multi-horizon forecasting both needed. Corsi (2009). Validate coefficients on Oxford-Man SPX RV against published HAR tables (Bollerslev-Patton-Quaedvlieg 2016, Table 1). |

#### Multivariate volatility

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| CCC-GARCH (Bollerslev 1990) | Constant conditional correlation: univariate GARCH per asset plus a fixed correlation matrix; base case for DCC and the null in constant-correlation tests. | Medium | Two-step estimation (univariate QML, then correlation of standardized residuals). Include the Engle-Sheppard (2001) constant-correlation test. Bollerslev (1990). Validate vs `rmgarch` and the MFE toolbox. |
| DCC-GARCH (Engle 2002) with correct two-step inference | Dynamic conditional correlation — the most-used multivariate volatility model in existence; getting estimation, targeting, and SEs right (where `rmgarch` is sloppy) is a flagship deliverable. | High | Q recursion with correlation targeting; normalize Q to R every step; log-likelihood via Cholesky, never explicit inverses/determinants. Two-step SEs must stack first-stage moment conditions (Engle-Sheppard 2001; Engle 2002). t-DCC of Pesaran-Pesaran for t errors. Validate vs `rmgarch` dccfit and MFE dcc() — document where and why they differ (targeting, df handling). |

#### Risk measurement (VaR/ES)

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| VaR and ES forecasting layer | Unified API producing h-step VaR/ES from any fitted model via parametric quantiles, Cornish-Fisher, FHS, or EVT tails — the object backtests consume. | Medium | Fix sign conventions once (losses positive; convention stored on the object). Multi-step VaR is not sqrt(h)-scalable under GARCH — simulate the h-period return distribution. ES under skew-t needs accurate partial expectations (quadrature fallback). Validate one-step parametric VaR/ES against `rugarch` and Kuester-Mittnik-Paolella (2006) backtest-study numbers. |
| VaR backtests: Kupiec, Christoffersen, duration, Dynamic Quantile | Unconditional coverage (Kupiec 1995), independence + conditional coverage (Christoffersen 1998), duration-based (Christoffersen-Pelletier 2004), and the DQ regression test (Engle-Manganelli 2004). | Low | LR tests are unreliable with few violations (T=250 at 1%): provide exact binomial and Monte Carlo p-values (Dufour 2006 randomization for discrete statistics). DQ: OLS on the hit sequence with lagged hits and VaR, chi-square. Validate against `rugarch` VaRTest and the GAS package's BacktestVaR. |

#### Backtesting & forecast evaluation

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Patton robust losses + Mincer-Zarnowitz | MSE- and QLIKE-class losses consistent for ranking under noisy volatility proxies (Patton 2011), plus MZ level/log regressions — the correct way to compare volatility forecasts. | Low | Only the Patton (2011) b-parameterized family (MSE b=0, QLIKE b=−2) is proxy-robust — refuse or warn on MAE and R2-on-sd with proxies. QLIKE differentials are heavy-tailed; pair with HAC. Validate loss values and rankings against Patton's MATLAB code and Bollerslev-Patton-Quaedvlieg (2016) tables. Comparison tests themselves (DM/GW/MCS) are consumed from forecasting-evaluation. |
| Volatility model selection guide (docs deliverable) | Opinionated documentation mapping data situation to model: asymmetry tests, long-memory diagnostics, macro drivers, system dimension — "which model when" applied to volatility. | Low | Decision-tree docs backed by runnable examples: sign-bias test → GJR/EGARCH; persistence near 1 + breaks → MS-GARCH or components; RV available → HAR/Realized GARCH/HEAVY (cite Hansen-Lunde 2005 and its realized-era reversal). Includes a replication gallery (Bollerslev 1986; Nelson 1991; Engle 2002; Corsi 2009; Hansen-Huang-Shek 2012) that doubles as the validation suite. |

### Tier 2 — Standard (expected of a serious library)

#### Univariate GARCH core

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| GARCH-X (exogenous variance regressors) | GARCH with covariates (realized measures, macro indicators, dummies) in the variance equation; the standard applied tool for event studies and spillovers. | Low | Nonnegativity of h is no longer guaranteed by parameter constraints alone — constrain x ≥ 0 with positive coefficients, use exp(gamma'x) multiplicatively, or clamp with a warning. Asymptotics in Han-Kristensen (2014). Validate against `rugarch` external.regressors. |

#### Asymmetric & nonlinear GARCH

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| APARCH (Ding-Granger-Engle 1993) | Asymmetric power ARCH with estimated power delta and leverage; nests GARCH, TGARCH, GJR variants and captures the Taylor effect. | Medium | Gradient of &#124;eps&#124;^delta ill-conditioned near zero; delta weakly identified in small samples — offer fixed-delta profiling. Ding-Granger-Engle (1993). Validate vs `arch` APARCH and G@RCH (Laurent) published output. |
| NAGARCH / AVGARCH / GQARCH cluster | Nonlinear-asymmetric GARCH (Engle-Ng 1993, used in Duan option pricing), absolute-value GARCH (Taylor 1986/Schwert 1989), and Sentana's quadratic GARCH — cheap once the fGARCH skeleton exists. | Low | Implement as parameter restrictions of a generic power/shift recursion. NAGARCH stationarity: beta + alpha(1 + theta^2) < 1. Engle-Ng (1993); Sentana (1995). Validate vs `rugarch` fGARCH submodels. |
| News impact curve + Engle-Ng sign-bias tests | NIC plots h as a function of the lagged shock at fixed h; the standard asymmetry visualization plus the companion LM tests for spec choice. | Low | Evaluate at h = unconditional variance by convention. Engle-Ng (1993). One-liner method on every fitted model; headline docs example. Validate curves against `rugarch` newsimpact(). |
| GARCH-in-mean (Engle-Lilien-Robins 1987) | Conditional variance (or sd, or log) enters the mean equation, testing risk-return tradeoffs; notably missing from Python's `arch`. | Medium | Mean and variance parameters no longer block-separable — joint estimation; chain rule through h into the mean for analytic derivatives. Offer variance, sd, and log-variance in-mean forms. Engle-Lilien-Robins (1987). Validate vs `rugarch` archm=TRUE. |

#### Long-memory volatility

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| FIGARCH (Baillie-Bollerslev-Mikkelsen 1996) | Fractionally integrated GARCH with hyperbolic decay of shocks' impact on variance; the standard parametric long-memory volatility model. | High | Expand the ARCH(infinity) representation truncated at ≥1000 lags; compute lambda weights once by recursion, then O(T·L) SIMD-friendly filtering. Use Conrad-Haag (2006) positivity conditions — the original BBM conditions are neither necessary nor sufficient. d and beta weakly jointly identified; presample values matter (long burn-in). Expose HYGARCH (Davidson 2004). Validate vs `arch` FIGARCH and Ox G@RCH output on DEM/USD from Baillie-Bollerslev-Mikkelsen (1996). |

#### Component & multi-frequency volatility

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Component GARCH / CGARCH (Engle-Lee 1999) + asymmetric variant | Additive decomposition into a slow-moving long-run component and a mean-reverting short-run component. | Medium | Constraints: long-run AR rho > transitory alpha + beta, both stationary; reparameterize to enforce ordering or the components swap identities (within-fit label switching). Engle-Lee (1999); asymmetric variant per Christoffersen et al. (2008). Validate vs `rugarch` csGARCH; expose component paths as first-class outputs. |
| Volatility decomposition & persistence toolkit | Unified post-fit API for component extraction, unconditional volatility, persistence, half-life, and annualization — the numbers practitioners quote. | Low | Half-life = log(0.5)/log(persistence); document per-model persistence definitions (alpha+beta; alpha+beta+gamma/2; EGARCH beta). Common component-plot interface across CGARCH/Spline/GARCH-MIDAS/MF2. Validate half-lives and unconditional vols against `arch` summary output. |

#### High-frequency and realized-measure construction

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Realized measure construction suite | From 5-min/tick data to daily measures: realized variance, subsampled/averaged RV, realized quarticity, and realized semivariances (Barndorff-Nielsen-Kinnebrock-Shephard 2010) — inputs for every realized model downstream. | High | Pitfalls: overnight-return treatment (add vs scale — make explicit), calendar/TZ alignment, sparse-sampling defaults. Validate against the R highfrequency package and the Oxford-Man Realized Library's published values. |
| Microstructure-noise-robust estimators | Two-scale RV (Zhang-Mykland-Aït-Sahalia 2005), realized kernels (Barndorff-Nielsen-Hansen-Lunde-Shephard 2008), and pre-averaging (Jacod et al. 2009) for RV estimation from noisy tick data. | High | Realized kernel with Parzen weights and the BNHLS (2008/2009) bandwidth rule; end-point jittering; pre-averaging window per Jacod et al. (2009). Python has no maintained realized-kernel implementation at all. Validate against R highfrequency and the BNHLS empirical tables. |
| Jump tests | Barndorff-Nielsen-Shephard (2006) bipower-variation ratio test, Lee-Mykland (2008) intraday jump detection, and Aït-Sahalia-Jacod (2009) power-variation jump activity test; required inputs for HAR-J/CHAR. | Medium | Bipower (BNS 2004) with the small-sample correction; Lee-Mykland requires intraday periodicity filtering first or the test is badly corrupted; ASJ needs careful power/truncation choices. Validate against R highfrequency and BNS/Lee-Mykland published rejection rates. |
| Tick-data cleaning conventions (BNHLS 2009) | The standard step-by-step tick and quote cleaning rules (out-of-session ticks, zero/negative prices, bounce-back outliers, same-timestamp aggregation) — every realized measure is only as good as its cleaning. | Medium | Implement the Barndorff-Nielsen-Hansen-Lunde-Shephard (2009) rules as a documented, configurable pipeline with per-rule deletion counts reported. Validate deletion counts and resulting RV against R highfrequency's cleaning functions and the BNHLS empirical tables. |
| Range-based volatility estimators | OHLC daily variance estimators (Parkinson 1980; Garman-Klass 1980; Rogers-Satchell 1991; Yang-Zhang 2000), 5-8x more efficient than squared returns; cheap realized-measure substitutes for HEAVY/Realized GARCH. | Low | Yang-Zhang (2000) handles overnight drift; document each estimator's zero-drift assumptions. Validate against TTR/highfrequency in R. |

#### Realized volatility models

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Realized GARCH (Hansen-Huang-Shek 2012) and Realized EGARCH (Hansen-Huang 2016) | Joint model of returns and a realized measure with a measurement equation linking RM to latent variance; the standard way to inject high-frequency information into GARCH-type filtering. | High | Joint likelihood must include the measurement-equation density (log sigma_u^2 term) — a classic bug is maximizing the partial likelihood only. Leverage function tau(z) = tau1·z + tau2·(z^2 − 1). Realized EGARCH allows multiple measures. Hansen-Huang-Shek (2012); Hansen-Huang (2016). Validate vs `rugarch` realGARCH and the HHS replication tables for SPY. |
| HEAVY models (Shephard-Sheppard 2010) | Two-equation system where realized measures drive the conditional variance of returns and their own dynamics; robust forecasting with quasi-closed-form multi-step forecasts. | Medium | Two Gaussian/QML equations estimable separately; multi-step forecasts via the companion VARMA form. Shephard-Sheppard (2010). Validate against Sheppard's MFE toolbox HEAVY code and the paper's OMX/SPY results. |

#### Multivariate volatility

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Corrected DCC / cDCC (Aielli 2013) | Fixes the inconsistency of DCC's correlation-targeting estimator via a reformulated Q recursion; the recommended default, with DCC kept for comparability. | Medium | Q*^{1/2} scaling inside the recursion makes targeting consistent; iterate the intercept estimate. Aielli (2013). Validate vs `rmgarch`'s cDCC — but cross-check with Aielli's own simulations, since `rmgarch`'s cDCC has had documented bugs. |
| Asymmetric DCC (Cappiello-Engle-Sheppard 2006) | Adds joint-bad-news terms to correlation dynamics; correlations rise after joint negative shocks — essential for equity portfolios and contagion studies. | Medium | AG-DCC generalization with asymmetry matrices; targeting involves E[n n'] of the negative-part vectors. Cappiello-Engle-Sheppard (2006). Validate vs `rmgarch` aDCC and the paper's international equity/bond results. |
| BEKK (Engle-Kroner 1995): scalar, diagonal, full | MGARCH with guaranteed PSD covariances by construction; the standard for small systems (2-5 assets) and volatility-spillover testing. | High | Identification: C lower-triangular with positive diagonal, sign convention a11, g11 > 0; full BEKK likelihood is multimodal — multi-start required; O(N^2) parameters restrict practical N. Covariance targeting via vec/vech mapping. Engle-Kroner (1995). Validate scalar/diagonal against the MFE toolbox; R has no reliable BEKK (mgarchBEKK is abandoned) — a real gap to fill. |

#### Innovation distributions & semiparametric tails

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Filtered historical simulation and EVT-POT tails (McNeil-Frey 2000) | Empirical distribution of standardized residuals (FHS) and generalized-Pareto tail fitting on them for extreme quantiles; the industry-standard VaR pipelines. | Medium | FHS: resample standardized residuals through the fitted variance recursion (Barone-Adesi-Giannopoulos-Vosper 1999). EVT: GPD MLE above a threshold (10% exceedances default), quantile/ES formulas from McNeil-Frey (2000); built-in threshold-sensitivity plots. Parametric innovation laws themselves come from foundations. Validate against the McNeil-Frey tables and R's evir/QRM. |

#### Estimation & inference machinery

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Variance targeting estimation | Replace the variance intercept with (1 − persistence)·sample variance, reducing dimension and stabilizing estimation; ubiquitous in multivariate models. | Low | Changes the asymptotic distribution — two-step GMM-style covariance per Francq-Horvath-Zakoian (2011), not plain QMLE SEs. Invalid under IGARCH; auto-disable with a message. Validate that targeted and free estimates converge on long simulated samples. |

### Tier 3 — Advanced (differentiators)

#### Asymmetric & nonlinear GARCH

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Hentschel fGARCH omnibus family (1995) | Single Box-Cox/shift parameterization nesting GARCH, EGARCH, GJR, TGARCH, APARCH, NAGARCH; enables encompassing tests of which asymmetry form the data prefer. | Medium | Hentschel (1995). Implementing this first and deriving named models as restrictions guarantees internal consistency and slashes duplication — this is how `rugarch` does it. Validation: fGARCH restricted fits must equal the dedicated implementations to machine precision. |
| Smooth-transition GARCH (ST-GARCH) | Logistic/exponential smooth transition between volatility regimes; between GJR's hard threshold and MS-GARCH's latent states. | Medium | Transition slope poorly identified when large (approaches an indicator) — profile or bound it. Hagerud (1997); Gonzalez-Rivera (1998). Few reference implementations; validate via simulation recovery study rather than another package. |
| Markov-switching GARCH (Haas-Mittnik-Paolella 2004; Gray 1996; Klaassen 2002) | Latent-regime GARCH capturing breaks in volatility level and persistence; fixes the upward persistence bias under breaks. | High | Exact likelihood is path-dependent (K^T paths). Implement the Haas-Mittnik-Paolella (2004) parallel-recursion spec (exact), with Gray (1996) and Klaassen (2002) collapsing approximations as options. Hamilton filter + smoothed regime probabilities; label switching in Bayesian mode. Validate against the R MSGARCH package (Ardia et al. 2019, JSS) replication tables. |
| Multiplicative-component intraday GARCH (Engle-Sokalska 2012) | Decomposes high-frequency return volatility into daily, diurnal, and intraday GARCH components; the standard for modeling intraday returns directly. | Medium | Requires an intraday periodicity estimate first (see periodicity item). Engle-Sokalska (2012). Validate against `rugarch` mcsGARCH; watch date/session alignment and half-days. |

#### Long-memory volatility

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| FIEGARCH (Bollerslev-Mikkelsen 1996) | Fractional integration in log-variance with EGARCH-type asymmetry; long memory plus leverage without positivity constraints. | High | Fractional differencing on the log-variance ARMA polynomial; truncation and burn-in issues as in FIGARCH. Bollerslev-Mikkelsen (1996). Almost no maintained implementation exists (`rugarch` dropped it; `arch` lacks it) — validate against the original paper's S&P 500 estimates and Ox G@RCH. |

#### Component & multi-frequency volatility

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Spline-GARCH (Engle-Rangel 2008) | Multiplicative decomposition where an exponential quadratic spline captures low-frequency volatility; used to link slow volatility to macro conditions. | Medium | Knot count by BIC; constrain the unit GARCH (mean-one) for identification. Engle-Rangel (2008). No mainstream package has it — validate against the paper's cross-country tables. Shares infrastructure with GARCH-MIDAS. |
| GARCH-MIDAS (Engle-Ghysels-Sohn 2013) | Multiplicative model with the long-run component driven by low-frequency variables through MIDAS beta-lag weights; the standard "does the macro economy drive volatility?" tool. | Medium | Beta-weight polynomial normalized to sum to one; restricted w1 = 1 monotone version is standard; rolling and fixed-span low-frequency windows both needed; short-run GARCH must be mean-one. MIDAS weighting utilities consumed from the nowcasting module. Engle-Ghysels-Sohn (2013). Validate against the R mfGARCH package (Conrad-Kleen 2020, JAE replication). |

#### Score-driven (GAS) models

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| GAS / score-driven volatility (Creal-Koopman-Lucas 2013; Harvey 2013 Beta-t-EGARCH) | Variance updates by the scaled score of the conditional likelihood; with Student-t errors yields Beta-t-EGARCH, which downweights outliers and dominates GARCH in turbulent samples. | Medium | Generic GAS(1,1) with scaling choices (identity, inverse-info, inverse-sqrt-info); Beta-t-EGARCH (Harvey-Chakravarty 2008; Harvey 2013) as flagship with leverage extension. Check invertibility/filter-stability post-fit (Blasques-Koopman-Lucas 2018). Validate vs the R GAS package (Ardia-Boudt-Catania 2019, JSS) and Harvey's published FTSE results. |

#### Stochastic volatility

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Log-normal SV via MCMC (Kim-Shephard-Chib 1998; ASIS) | Latent AR(1) log-volatility estimated by the KSC mixture sampler; the canonical parameter-driven alternative to GARCH and the base for macro SV blocks (TVP-VAR-SV). | High | Approximate log chi-square(1) with the Omori et al. (2007) 10-component mixture; sample h jointly via sparse tridiagonal precision draws; ASIS interweaving of centered/noncentered parameterizations (Kastner-Frühwirth-Schnatter 2014) — without it, mixing on (phi, sigma_eta) is catastrophically slow. SV-t via scale mixtures. Validate posterior means/quantiles against the R stochvol package (Kastner 2016, JSS) with matched priors. |
| SV with leverage (Omori-Chib-Shephard-Nakajima 2007) | Correlated return and log-volatility innovations capture the leverage effect within SV; needed for any serious equity application. | High | The mixture approximation must be extended to the bivariate case exactly as in Omori et al. (2007) — naive reuse of KSC weights biases rho. Validate against stochvol's svlsample and the paper's TOPIX results. |

#### High-frequency and realized-measure construction

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Refresh-time sampling and the multivariate realized kernel | Synchronization of non-synchronous tick data across assets plus the multivariate realized kernel (Barndorff-Nielsen-Hansen-Lunde-Shephard 2011); the upstream requirement for realized semicovariances and multivariate HEAVY. | High | Refresh-time grid construction, then the PSD-guaranteed multivariate kernel with joint bandwidth. BNHLS (2011). Validate against R highfrequency's refresh-time and rKernelCov output. |
| Intraday periodicity estimation and filtering | Diurnal volatility pattern estimation (flexible Fourier form or robust nonparametric) needed before any intraday GARCH or jump detection; skipping it corrupts intraday jump tests badly. | Medium | Andersen-Bollerslev (1997) FFF; robust weighted-SD/TML versions from Boudt-Croux-Laurent (2011) to avoid jump contamination. Validate against highfrequency's spotVol. |

#### Realized volatility models

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| HAR extensions: HAR-J, CHAR, SHAR, HARQ, log-HAR | Jump-separated (Andersen-Bollerslev-Diebold 2007), semivariance-based (Patton-Sheppard 2015), and measurement-error-adjusted (HARQ) variants that dominate plain HAR out of sample. | Low | HARQ interacts the daily-lag coefficient with sqrt(RQ) — requires realized quarticity from the measure suite; guard against negative fitted variances (insanity filter as in BPQ). Patton-Sheppard (2015); Bollerslev-Patton-Quaedvlieg (2016). Validate against the BPQ replication files. No mainstream package bundles these — easy differentiator. |
| Multiplicative Error Model (MEM, Engle 2002) | GARCH-style dynamics for any nonnegative process (RV, range, volume, durations); the general framework behind HEAVY and realized modeling. | Medium | Exponential/Gamma QMLE — Gamma QML point estimates coincide with GARCH applied to the square root of the series; asymmetric signed-return terms standard. Engle (2002); Engle-Gallo (2006). Validate vs Cipollini-Engle-Gallo replication material. |

#### Multivariate volatility

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| DECO (Engle-Kelly 2012) | Dynamic equicorrelation — a single time-varying correlation across all pairs; closed-form inverse/determinant makes it feasible and surprisingly competitive for large equity panels. | Medium | Equicorrelation must stay in (−1/(N−1), 1); closed-form determinant/inverse gives O(N) likelihood. Engle-Kelly (2012). Validate vs the paper's results; almost no open implementation exists. |
| GO-GARCH / O-GARCH with ICA estimation | Orthogonal factor structure: univariate GARCH on statistically independent linear combinations; scales well, closed-form aggregation, popular for risk decomposition. | High | Mixing matrix by FastICA per Broda-Paolella (2009) or ML per van der Weide (2002); ICA sign/order indeterminacy must be pinned for reproducibility. Alexander (2001); van der Weide (2002). Validate vs `rmgarch` gogarchfit with NIG factors. |
| Factor GARCH (Engle-Ng-Rothschild 1990) and Rotated ARCH (Noureldin-Shephard-Sheppard 2014) | Observable/latent factor structures for conditional covariances; RARCH rotates returns by the unconditional covariance square root so scalar dynamics fit better — a clean modern baseline for medium N. | High | RARCH: rotate by Sigma^{−1/2}, fit scalar/diagonal BEKK-type dynamics on rotated returns; targeting is automatic. Engle-Ng-Rothschild (1990); Noureldin-Shephard-Sheppard (2014). Validate vs Sheppard's MFE code. |
| DCC-MIDAS (Colacito-Engle-Ghysels 2011) | Long-run/short-run decomposition of correlations with the long-run driven by low-frequency averages; the correlation analogue of GARCH-MIDAS. | Medium | Long-run correlation from MIDAS-weighted sample correlations of standardized residuals, projected to PSD; short-run DCC around it. MIDAS weights from nowcasting. Colacito-Engle-Ghysels (2011). Validate against the authors' replication; no Python implementation exists. |

#### Copula-based dependence

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Copula-GARCH (static and time-varying, Patton 2006) | Univariate GARCH margins plus a copula (Gaussian, t, Clayton, Gumbel, SJC) for dependence with tail asymmetry beyond correlation; the standard bivariate/small-N risk and contagion tool. | High | Two-step IFM estimation; SEs must account for margin estimation (Patton 2006 or bootstrap). Time-varying parameters via Patton's ARMA-type evolution or GAS. PIT transforms must use the fitted skewed distributions' CDFs — numerical inversion accuracy matters. Patton (2006); Jondeau-Rockinger (2006). Validate vs `rmgarch` cgarchfit and Patton's MATLAB copula toolbox. Vine/factor copulas are deferred (see scope). |

#### Estimation & inference machinery

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Bayesian GARCH estimation | Adaptive MCMC / griddy Gibbs posteriors for GARCH-family models: exact finite-sample uncertainty, density forecasts with parameter uncertainty, regularization for weakly identified specs (FIGARCH d, MS-GARCH). | Medium | Adaptive random-walk MH on transformed parameters; priors respecting the stationarity region. Ardia (2008). Validate posterior means vs bayesGARCH (R) with matched priors; expose marginal likelihood (bridge sampling) for model comparison. |

#### Risk measurement (VaR/ES)

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| ES backtests: McNeil-Frey, Acerbi-Szekely, conditional calibration, ESR | The modern ES backtesting toolkit: exceedance-residual bootstrap (McNeil-Frey 2000), Acerbi-Szekely (2014) Z1/Z2 with simulated critical values, Nolde-Ziegel (2017) conditional calibration, and the regression-based ESR test (Bayer-Dimitriadis 2022). | Medium | AS tests need model-implied distributions to simulate critical values — the API must let fitted models supply samplers. ESR: joint (VaR, ES) regression via FZ-loss minimization, numerically delicate — follow Bayer-Dimitriadis (2022) and validate against their R package esback. Basel traffic-light utilities as a convenience. |
| Joint VaR-ES elicitability scoring (Fissler-Ziegel) and Murphy diagrams | Strictly consistent joint scores for (VaR, ES) pairs enabling DM-style comparison of tail-risk models, plus Murphy diagrams for dominance across all consistent scores. | Medium | ES alone is not elicitable — only jointly with VaR (Fissler-Ziegel 2016); default to the FZ0 (0-homogeneous) loss of Patton-Ziegel-Chen (2019). Murphy diagrams per Ehm-Gneiting-Jordan-Krüger (2016). Validate score values against the esback/murphydiagram R packages. Nothing in Python offers this. |
| CAViaR (Engle-Manganelli 2004) | Direct autoregressive quantile models (symmetric absolute value, asymmetric slope, indirect GARCH) by nonlinear quantile regression — semiparametric VaR without a distributional assumption. | Medium | Non-smooth objective: use the Engle-Manganelli grid+simplex+quasi-Newton multi-start scheme or smoothed quantile loss (solver from foundations); SEs via the paper's k-nearest-neighbor bandwidth estimator. Engle-Manganelli (2004). Validate against their published GM/S&P parameter tables — a standard replication target. |

### Tier 4 — Frontier (research-grade, each gated on a named validation target)

#### Component & multi-frequency volatility

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| MF2-GARCH (Conrad-Engle 2025) | Multiplicative two-component GARCH where the long-run component is driven by smoothed forecast errors of the short-run component; captures volatility cycles without external MIDAS data and beats GARCH at long horizons. | Medium | Closed-form multi-step forecasts exist — implement them; they are the headline feature. Conrad-Engle (2025, JAE). Gate: reproduce the authors' replication files (Conrad's MATLAB/R code). Cheap differentiator once GARCH-MIDAS scaffolding exists. |

#### Score-driven (GAS) models

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Score-driven correlation and copula dynamics (GAS-DCC, GAS copulas) | GAS updates for correlation matrices and copula parameters; the modern alternative to DCC dynamics with better outlier robustness under fat tails. | High | Parameterize correlations via hyperspherical or Fisher-z transforms to stay in the valid set; the t-copula score requires careful matrix calculus. Creal-Koopman-Lucas (2011); Oh-Patton (2018). Gate: match the R GAS package in low dimensions; simulation recovery in higher dimensions. |

#### Stochastic volatility

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| SV with jumps in returns and/or volatility (Eraker-Johannes-Polson 2003) | Compound-Poisson jumps in returns (SVJ) and volatility (SVCJ); required for crisis samples and the bridge to option pricing. | Research-grade | MCMC with data augmentation for jump times/sizes; jump intensity and jump-size variance weakly identified — priors matter and must be documented. Eraker-Johannes-Polson (2003). Gate: reproduce EJP's S&P 500/Nasdaq posterior tables; simulation recovery essential. |
| Particle filters and particle MCMC for nonlinear SV | Bootstrap/auxiliary particle filters for likelihood evaluation in SV beyond the mixture trick (leverage + jumps + fat tails), with PMMH for full Bayes and smooth resampling for point estimation. | Research-grade | PF likelihoods are noisy and discontinuous in parameters — never hand them to a derivative-based optimizer; use PMMH (Andrieu-Doucet-Holenstein 2010) tuned so likelihood-estimate variance is about 1 at the mode, or Malik-Pitt (2011) continuous resampling for MLE. Systematic resampling, log-space weights, ESS-triggered resampling. A headline Rust speed feature. Gate: filtered volatility must match KSC Gibbs output on models where both apply. |

#### Realized volatility models

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Rough volatility forecasting (Gatheral-Jaisson-Rosenbaum 2018) | Log-RV modeled as fractional Brownian motion with Hurst H around 0.1; simple, strong RV forecasts connecting econometrics to rough-vol option pricing. | Medium | Forecast is a weighted integral of past log-RV (discretized kernel weights); estimate H by the scaling regression on q-order moments. Trap: measurement noise in RV biases H downward — document. Gatheral-Jaisson-Rosenbaum (2018). Gate: reproduce their Oxford-Man results; no econometrics package ships this. |

#### Multivariate volatility

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Multivariate HEAVY and realized covariance models (incl. semicovariances) | Realized-covariance-driven conditional covariance dynamics (multivariate HEAVY, Realized DCC/BEKK) plus realized semicovariance decompositions of asymmetric comovement. | High | Wishart-type measurement densities; ensure PSD via matrix square-root or Cholesky dynamics; refresh-time synchronization and the multivariate kernel (Tier 3) required upstream. Noureldin-Shephard-Sheppard (2012); Bollerslev-Li-Patton-Quaedvlieg (2020). Gate: reproduce the Noureldin-Shephard-Sheppard replication files. |
| Vast-dimensional DCC: composite likelihood + nonlinear shrinkage | DCC-type estimation for hundreds to thousands of assets via composite (pairwise) likelihood and nonlinear-shrinkage correlation targeting; the state of the art for large portfolio risk. | Research-grade | Composite likelihood over contiguous or random pairs (Pakel-Shephard-Sheppard-Engle 2021) avoids O(N^3) likelihoods and large-N targeting bias; DCC-NL uses Ledoit-Wolf nonlinear shrinkage as the target (Engle-Ledoit-Wolf 2019). Only author MATLAB code exists — Rust parallelism is decisive. Gate: reproduce the Engle-Ledoit-Wolf simulation designs and out-of-sample minimum-variance portfolio results. |

#### Estimation & inference machinery

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Bootstrap and boundary-robust inference for GARCH | Residual-based bootstrap for parameter CIs, volatility-path bands, and test critical values, with corrections when parameters sit near the boundary (alpha near 0) where Gaussian asymptotics fail. | High | Bootstrap must re-filter: resample standardized residuals, rebuild returns through the recursion, re-estimate. Naive percentile intervals fail at boundaries — use the modified bootstrap of Cavaliere-Nielsen-Pedersen-Rahbek (2022). Compute-heavy: parallel re-estimation in Rust is the selling point. Gate: reproduce coverage in a Monte Carlo study matching the Cavaliere et al. (2022) designs. |

#### Risk measurement (VaR/ES)

| Method | What it is / when to use it | Difficulty | Implementation notes and validation target |
|---|---|---|---|
| Dynamic joint VaR-ES models (Taylor 2019; Patton-Ziegel-Chen 2019) | Score-driven/semiparametric models filtering VaR and ES jointly (ES-CAViaR via asymmetric-Laplace quasi-likelihood; FZ-loss-minimizing GAS one-factor models); the research frontier of tail-risk forecasting, absent from all mainstream libraries. | High | Taylor (2019): AL density with time-varying scale ties ES to VaR; Patton-Ziegel-Chen (2019): minimize FZ0 loss with GAS dynamics. Optimization is non-smooth-ish — multi-start plus derivative-free polish. Gate: reproduce Taylor's and PZC's published out-of-sample FZ losses on their equity datasets. |

## Frontier watchlist

- Heavy-tailed realized-measure GAS models (Opschoor et al. 2021) — score-driven covariance dynamics driven by realized measures with fat-tailed measurement densities.
- SMC^2 and fully adaptive sequential estimation for SV with leverage and jumps — the natural extension of the Tier 4 particle-MCMC stack once PMMH lands.
- Factor and vine copulas for high dimensions (Oh-Patton 2017, 2018; Dissmann et al. 2013) — deferred by scope ruling; ship static and GAS copulas first and document pyvinecopulib interop for vine structures.
- ML hybrids as optional extras: HARNet (Reisenhofer et al. 2022), GARCH-LSTM residual hybrids, and amortized variational deep SV — valuable as benchmarks, but kept out of the statistical core.

## Implementation warnings

- **Return scaling.** Optimizers fail or find flat regions when returns are in decimals; fit on percent returns internally (the well-known `arch` scale warning) and make results provably scale-equivariant with a CI test.
- **Variance-recursion initialization.** EWMA backcast vs unconditional vs first-k average visibly changes estimates for T < 1000 and is the number-one cause of cross-package disagreement — pick `arch`'s backcast as the default, expose the alternatives, and document exactly.
- **Constraints and boundaries.** Enforce positivity/stationarity via reparameterization or constrained optimization, but report SEs on the natural scale via the delta method; detect boundary solutions (alpha = 0, persistence = 1) and warn that standard asymptotics are invalid there (Andrews 2001; Cavaliere et al. 2022).
- **Standard errors.** Default to Bollerslev-Wooldridge (1992) sandwich SEs — Hessian-only SEs are wrong under non-Gaussian innovations; implement analytic gradients (Fiorentini-Calzolari-Panattoni 1996) because numerically differentiating a T-long recursion accumulates error and is slow.
- **EGARCH.** Clamp log-variance against overflow; E|z| constants are distribution-specific; multi-step variance forecasts require simulation or a lognormal correction, since E[h] ≠ exp(E[log h]).
- **Asymmetric-model forecasts.** Multi-step forecasts need P(z<0) and partial moments E[z^2 1(z<0)] under the actual innovation law — hardcoding 0.5 breaks skew-t GJR forecasts silently.
- **FIGARCH.** Truncate the ARCH(infinity) weights at ≥1000 lags with long presample burn-in; use Conrad-Haag (2006) positivity conditions, not the commonly copied BBM conditions; expect weak joint identification of d and beta.
- **Skew-t confusion is endemic.** Hansen (1994) and Fernandez-Steel (1998) are different distributions both called "skew-t" (`rugarch`'s sstd is Fernandez-Steel); implement both, name them unambiguously, and unit-test quantiles against published tables.
- **Student-t standardization.** Standardize by sqrt((nu−2)/nu) with nu > 2 enforced through a smooth transform; several published replications fail because of the unstandardized-t convention in older software.
- **Variance targeting.** Targeting changes the asymptotic covariance (Francq-Horvath-Zakoian 2011) — reusing plain QMLE SEs after targeting is wrong; targeting is invalid under IGARCH (auto-disable).
- **DCC.** The correlation-targeting intercept estimator is inconsistent (Aielli 2013) — ship cDCC and say why; renormalize Q to R every step; compute log-likelihoods via Cholesky factorizations, never explicit inverses/determinants; two-step SEs must stack first-stage moment conditions.
- **BEKK.** Requires sign normalizations (diagonal of C positive, a11, g11 > 0) or the likelihood has reflection multimodality; always multi-start full BEKK.
- **SV via MCMC.** Use the Omori et al. (2007) 10-component mixture (bivariate extension under leverage) and ASIS interweaving — naive centered-only Gibbs on (phi, sigma_eta) mixes so slowly that results look converged but are not; monitor effective sample size per parameter.
- **Particle filters.** PF likelihoods are noisy and discontinuous in parameters: never feed them to derivative-based optimizers; use PMMH with likelihood-variance tuning, or Malik-Pitt (2011) continuous resampling for MLE; keep all weights in log space with log-sum-exp.
- **Proxy-based forecast comparison.** Only Patton (2011) robust losses (MSE/QLIKE families) rank correctly under proxy noise — warn loudly on MAE and "R2 on sd"; QLIKE differentials are heavy-tailed, so pair DM tests with HAC and consider bootstrap p-values.
- **VaR/ES conventions.** Sign and tail conventions (losses positive vs returns negative, left vs right tail) cause silent backtest bugs — fix one convention on the risk object and convert at the boundaries; with 1% VaR on 250 observations provide exact/Monte Carlo p-values, because LR asymptotics are unreliable with ~2.5 expected violations.
- **Bootstrap for GARCH.** Must be residual-based with re-filtering through the variance recursion (never resample raw returns iid); simulation engines need burn-in plus explosion caps when persistence is near one — cap and count discarded paths rather than crashing.
- **Realized measures.** Overnight-return handling (add vs rescale), timezone/session alignment, and kernel bandwidth choice move daily RV by tens of percent — pin documented defaults and validate against the Oxford-Man Realized Library; Realized GARCH likelihoods must include the measurement-equation term or parameters are biased.
- **Factor-method reproducibility.** ICA in GO-GARCH has sign/permutation indeterminacy — fix an ordering convention and seed policy or users get different "fits" on every run.

## Dependencies and shared infrastructure

**Consumed from foundations:**

- **Innovation-distribution zoo** (normal, Student-t, GED, Hansen skew-t, Fernandez-Steel, Johnson SU, NIG, GH skew-t). This module additionally needs unit-variance standardization conventions and analytic partial moments — E|z|, E[z^2 1(z<c)], P(z<0) — per distribution, which the multi-step forecasting engine and asymmetric-model recursions require; plus exact (piecewise where needed) quantile functions for parametric VaR/ES.
- **Resampling/bootstrap engine** with the parallel RNG substream contract — FHS resampling, bootstrap prediction intervals, and the boundary-robust bootstrap designs all run on it.
- **Philox-based reproducible parallel RNG** — path simulation, common random numbers, antithetics, MCMC/PMCMC seeding.
- **Exogenous-regressor (covariate) contract** — GARCH-X variance regressors and MIDAS-filtered long-run components ingest external series through the shared aligned interface; multi-step variance forecasts with covariates require future covariate paths, handled by its known-future/scenario/auxiliary-forecast distinction rather than ad hoc per-model conventions.
- **Numerical optimizers** — constrained/transformed QMLE, multi-start orchestration, derivative-free polish for non-smooth objectives.
- **Fast quantile-regression solver** — CAViaR estimation and ESR-type joint (VaR, ES) regressions.
- **HAC/long-run-variance inference** — MZ regressions and heavy-tailed QLIKE loss differentials.
- **Critical-value engine** — cached simulated critical values for Acerbi-Szekely ES backtests and Dufour-style Monte Carlo p-values.
- **Linear-Gaussian state-space engine** (precision-based simulation smoother) — the KSC/Omori mixture samplers for SV draw latent log-volatility through it.
- **Time-index/calendar/frequency engine** — intraday session alignment, timezone handling, half-days, overnight-return conventions in the realized-measure pipeline.

**Consumed from other modules:**

- **forecasting-evaluation:** Diebold-Mariano, Giacomini-White, Model Confidence Set, SPA, and Reality Check tests (the source inventory's forecast-comparison items live there; this module consumes and re-exports them so volatility horse races run without glue code), plus density-forecast evaluation (PITs, scores) for GARCH/SV density forecasts.
- **nowcasting:** MIDAS beta-lag weighting utilities, consumed by GARCH-MIDAS and DCC-MIDAS.
- **Unified forecast object** (library-wide) — all variance, VaR/ES, and covariance forecasts publish through it.
- **Golden-value validation harness** (library-wide) — the validation gallery below runs on it.

**Exposed to other modules:**

- Fitted conditional-variance and covariance paths, standardized residuals, and PIT transforms — consumed by forecasting-evaluation (density evaluation) and by identification (heteroskedasticity-based approaches where applicable).
- The realized-measure construction pipeline (cleaned tick data through RV/kernels/semivariances) — usable by any module needing high-frequency volatility features (nowcasting, ML).
- The VaR/ES risk object with fixed sign/tail conventions — the standard interface for tail-risk work anywhere in the library.
- The SV blocks (log-volatility samplers) — reusable inside bayesian's TVP-VAR-SV machinery.

## Validation gallery

- **Bollerslev (1986) UK inflation GARCH** — original parameter estimates reproduced.
- **GARCH(1,1) on S&P 500 daily returns** — match `arch`, `rugarch`, and Kevin Sheppard's MFE MATLAB toolbox to 6 decimals under identical backcast initialization.
- **RiskMetrics Technical Document (1996)** — EWMA VaR tables reproduced.
- **Baillie-Bollerslev-Mikkelsen (1996) DEM/USD FIGARCH** — match published estimates and Ox G@RCH output.
- **Hansen-Huang-Shek (2012) SPY Realized GARCH** — replication tables matched, including the measurement-equation likelihood term.
- **Bollerslev-Patton-Quaedvlieg (2016) HAR/HARQ tables** — coefficients and out-of-sample losses on Oxford-Man SPX RV.
- **BNHLS (2008/2009) realized kernel empirical tables** — kernel RV and tick-cleaning deletion counts vs the R highfrequency package.
- **Ardia et al. (2019, JSS) MSGARCH replication tables** — Markov-switching GARCH estimates and forecasts.
- **Kastner (2016, JSS) stochvol posteriors** — SV posterior means/quantiles with matched priors; Omori et al. (2007) TOPIX results for the leverage extension.
- **Engle-Manganelli (2004) CAViaR GM/S&P parameter tables** — the standard replication target for quantile-based VaR.
- **Kuester-Mittnik-Paolella (2006) VaR backtest study** — one-step VaR/ES numbers across distributions and methods.
- **Conrad-Kleen (2020, JAE) mfGARCH replication** — GARCH-MIDAS estimates.
- **Conrad-Engle (2025, JAE) MF2-GARCH replication files** — closed-form multi-step forecasts.
- **Engle-Ledoit-Wolf (2019) DCC-NL simulations and minimum-variance portfolios** — vast-dimensional gate.
- **Taylor (2019) / Patton-Ziegel-Chen (2019) out-of-sample FZ losses** — dynamic joint VaR-ES gate.
