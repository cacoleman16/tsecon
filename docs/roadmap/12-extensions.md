# Module 12 — Extensions: Beyond the Core Brief

> Part of the time series econometrics library roadmap. Master plan: [ROADMAP.md](../../ROADMAP.md).

**This module collects the additions that were not in the original brief but that an adversarial completeness review identified as necessary for a genuinely end-to-end time series econometrics framework.** Each extension is a candidate future module with its own scope, priority, and validation targets. Four other review findings were folded directly into existing modules rather than listed here: high-frequency/realized-measure construction (→ [Module 03](03-volatility.md)), forecasting under structural breaks (→ [Module 09](09-forecasting-evaluation.md)), bootstrap-engine completions such as subsampling and the Hansen grid bootstrap (→ [Module 00](00-architecture.md) foundations), and bundled datasets/loaders (→ [Module 11](11-docs-ux-adoption.md)).

---

## High priority

### E1 — Panel time series econometrics

The nonstationary-panel workflow that sits between the panel unit-root tests (Module 01) and panel VAR/panel LP (Modules 04/07), none of which any expert domain owned:

| Method | Reference | Notes |
|---|---|---|
| Panel cointegration tests | Pedroni 1999/2004; Kao 1999; Westerlund 2007 ECM tests | Westerlund bootstrap under cross-sectional dependence is the hard part |
| Panel FMOLS / DOLS | Pedroni 2000/2001 | Group-mean and pooled variants |
| Mean Group & Pooled Mean Group ARDL | Pesaran-Smith 1995; Pesaran-Shin-Smith 1999 | The workhorse for cross-country heterogeneous panels |
| Common Correlated Effects (CCE-MG/CCE-P) | Pesaran 2006; Chudik-Pesaran 2015 dynamic CCE | Rank conditions with few time periods are a known trap |
| Interactive fixed effects | Bai 2009 | Iterated PCA + least squares |
| Cross-section dependence diagnostics | Pesaran CD test; Bailey-Kapetanios-Pesaran CSD exponent | Gate for choosing first- vs second-generation panel tests |
| Half-panel jackknife bias corrections | Dhaene-Jochmans 2015 | Beyond the LP context already covered in Module 07 |

**Why:** bread-and-butter for central-bank cross-country work; currently forces users into Stata (`xtpmg`, `xtwest`, `xtdcce2`) or fragile R code. **Validate against:** Stata `xtwest`/`xtpmg`/`xtdcce2` output; Pesaran (2006) and Chudik-Pesaran (2015) Monte Carlo tables.

### E2 — Term structure and yield-curve econometrics

The single most common model family among the library's stated target users (central-bank researchers), absent from every maintained Python or R package:

| Method | Reference | Notes |
|---|---|---|
| Dynamic Nelson-Siegel in state space | Diebold-Li 2006 | Direct fit on the shared SSM engine |
| Arbitrage-free Nelson-Siegel (AFNS) | Christensen-Diebold-Rudebusch 2011 | Closed-form yield-adjustment term |
| Svensson / Nelson-Siegel-Svensson curves | Svensson 1994 | Cross-sectional fitting utilities |
| Gaussian affine term structure models with JSZ normalization | Joslin-Singleton-Zhu 2011 | Concentrated likelihood kills rotation indeterminacy — nearly self-checking MLE |
| Regression-based term premia | Adrian-Crump-Moench 2013 | Validate against the NY Fed's published ACM term-premium series |
| Shadow-rate models | Krippner 2013; Wu-Xia 2016 | Extended Kalman filter divergence is the known trap; validate against published Wu-Xia shadow-rate series |
| Bond return predictability factors | Cochrane-Piazzesi 2005 | Ties into E3 |

**Traps:** likelihood flatness in market-price-of-risk parameters; near-unit-root level factor. **Why:** everyone currently passes around Matlab code from the ACM and JSZ replication files.

### E3 — Predictive-regression inference with persistent regressors

One of the most-run regressions in empirical finance/macro-finance, where naive OLS t-statistics are badly oversized:

| Method | Reference | Notes |
|---|---|---|
| Stambaugh bias correction | Stambaugh 1999 | |
| Bonferroni Q-tests | Campbell-Yogo 2006 | Validate against published confidence intervals |
| IVX estimation and Wald inference | Kostakis-Magdalinos-Stamatogiannis 2015 | Robust across the whole persistence spectrum, trivially fast; only scattered author code exists |
| Long-horizon regression inference | Hodrick 1992 SEs; Valkanov 2003 rescaled t | |
| Nearly-optimal tests | Elliott-Müller-Watson 2015 | Optional advanced tier |

Slots naturally beside the SADF/bubble-testing machinery in Module 01. **Validate against:** KMS (2015) empirical tables.

### E4 — GMM, simulated method of moments, and indirect inference

General moment-based estimation underpins Euler-equation asset pricing, IRF-matching estimation of DSGE-lite models, heteroskedasticity-based SVAR identification, and models with intractable likelihoods (MS-GARCH). Python has no credible time-series GMM. The compiled parallel core makes SMM/indirect inference genuinely fast — exactly this library's comparative advantage:

| Method | Reference | Notes |
|---|---|---|
| Time-series GMM (2-step, iterated, CUE) | Hansen 1982 | HAC/prewhitened weighting; bandwidth choice inside the weighting-matrix iteration is a trap |
| Weak-identification-robust inference | Stock-Wright 2000 S-sets; Kleibergen 2005 K/CLR | |
| Simulated method of moments | Duffie-Singleton 1993; Lee-Ingram 1991 | Common random numbers via Philox substreams eliminate chatter — elegant fit with the RNG design |
| Indirect inference | Gouriéroux-Monfort-Renault 1993 | |
| Efficient method of moments (EMM) | Gallant-Tauchen 1996 | Advanced tier |
| Impulse-response matching | Christiano-Eichenbaum-Evans 2005; Guerron-Quintana-Inoue-Kilian 2017 | Weak-identification-robust IRF matching consumes the LP/VAR modules |

**Validate against:** Hansen-Singleton-style textbook replications; CEE (2005) IRF-matching estimates.

---

## Medium priority

### E5 — DSGE-lite estimation layer (an explicit scope decision)

The mission statement names "Dynare's estimation tools," and Module 05 lists DSGE-VAR — which is unbuildable without a linear rational-expectations solver that no module owned. **Decision: build the minimal, sharply-scoped version; everything else is a documented non-goal with Dynare interop as the answer.**

In scope: linear RE solvers (Sims 2002 `gensys`; Klein 2000; Blanchard-Kahn), first-order perturbation, mapping solved models into the shared state-space engine, Bayesian estimation à la Herbst-Schorfheide (2015) using the already-planned SMC and particle-filter machinery, prior/posterior IRF and variance-decomposition reporting. Out of scope (documented non-goals): higher-order perturbation, OccBin, projection methods.

**Validate against:** the Herbst-Schorfheide NK-model posterior; Dynare output on the Smets-Wouters model. **Unlocks:** DSGE-VAR (Del Negro-Schorfheide 2004) in Module 05.

### E6 — Survey-expectations toolkit

Expectations data are now a core input to macro identification (information effects, Module 06) and entropic tilting (Module 05), but nothing covers ingesting and modeling the survey data itself. Fixed-event forecast structures break every standard evaluation tool in Module 09 and need deliberate support.

In scope: SPF/consensus data structures (fixed-event and fixed-horizon panels), probability-bin density fitting (Engelberg-Manski-Williams 2009), disagreement and ex-ante uncertainty decompositions (Lahiri-Sheng 2010), panel forecast-rationality tests with appropriate covariance structures (Keane-Runkle 1990), micro-level Coibion-Gorodnichenko information-rigidity regressions, fixed-event-to-fixed-horizon conversion (Dovern-Fritsche-Slacalek 2012).

**Validate against:** Coibion-Gorodnichenko (2015 AER) coefficient estimates; Philadelphia Fed SPF documentation conventions.

### E7 — Fractional cointegration

The library commits heavily to long memory (Sowell MLE, exact local Whittle, fast fractional differencing) and to cointegration (Johansen) — but not to their intersection, the standard framework for realized-volatility comovement and interest-parity applications.

In scope: FCVAR estimation and rank testing (Johansen 2008; Johansen-Nielsen 2012), narrow-band least squares fractional cointegrating regressions (Robinson-Marinucci 2003), Nielsen-Shimotsu (2007) rank determination via exact local Whittle. The fast fractional-differencing kernel already planned does the heavy lifting.

**Traps:** identification restrictions on d and b; initial-values sensitivity of the profile likelihood. **Validate against:** the Nielsen-Popiel Matlab/R FCVAR implementation.

### E8 — Dynamic discrete-outcome models (recession probability)

Recession-probability modeling is among the most common applied tasks for the target user base and sits naturally beside Bry-Boschan dating (Module 01) and MS-DFM recession nowcasting (Module 08).

In scope: dynamic probit/logit with lagged latent or lagged response dynamics (Kauppi-Saikkonen 2008; Chauvet-Potter 2005 Bayesian variants), autoregressive conditional hazard models, correctly simulated multi-step recession probabilities (a common user error when done analytically), evaluation via the Brier/ROC machinery of Module 09.

**Canonical replication:** Estrella-Mishkin (1998) yield-curve probit; Kauppi-Saikkonen published U.S. recession results.

---

## Lower priority / deferred

### E9 — Consistent specification tests

Omnibus complements to the directional (portmanteau/LM) diagnostics of Module 01: Bierens (1990) ICM tests, Escanciano (2006) generalized spectral and Cramér-von Mises tests for conditional-mean/quantile specification, Hong (1996) spectral tests, all with wild-bootstrap p-values. Essentially unavailable outside author code. Validate against Escanciano's published size/power tables.

### E10 — Functional time series

Curve-valued series (yield curves, intraday load/return curves): functional PCA, functional AR(1) (Bosq 2000), dynamic FPCA (Hörmann-Kidziński-Hallin 2015). A real subfield, but a thinner audience needing its own data structures — **explicitly deferred to the extension API** rather than silently absent. E2 covers the main economic use case meanwhile. Validate against R `ftsa`.

### E11 — Climate econometrics (docs + datasets deliverable, not new estimators)

The methods are ~95% re-exports of tools already planned: robust trend inference, cointegration of forcing and temperature in state-space form (Pretis 2020 energy-balance-model equivalence), detection-and-attribution regressions with HAC/fixed-b inference, indicator saturation (IIS/SIS, Module 10). Ship as a documentation chapter plus HadCRUT/forcing datasets. Validates against Pretis (2020, JoE) and Estrada-Perron attribution papers. Cheap; opens a new user community.

### E12 — Heavy tails and extremal dependence

Tail-index estimation under serial dependence (Hill with block/cluster inference), the extremogram and cross-extremogram (Davis-Mikosch 2009) with stationary-bootstrap bands, extremal-index estimation. Small module, natural companion to the quantilogram (Module 01); absent everywhere in Python. Validate against R `extremogram` and Davis-Mikosch published examples.

---

## Sequencing guidance

E1–E4 are high-priority because they serve the library's core constituency directly and have clean validation targets; schedule them immediately after the Phase 3 modules stabilize. E5 should be decided (not necessarily built) early because Module 05's DSGE-VAR promise depends on it. E6–E8 ride on infrastructure that exists by Phase 3. E9–E12 wait for the extension API and community demand.
