# Research-contribution scan — where tsecon can contribute to the literature

> **Working document.** Produced by a four-lens web scan (methods frontier,
> applied gaps, software venues, frontier forecasting) run 2026-08-17, then
> synthesized and ranked. Lives in `docs/roadmap/` and is excluded from the
> published site. **Every citation below was collected from search metadata
> (direct arXiv/RePEc fetches were egress-blocked) and must be re-verified
> against the primary source before any submission is drafted.**

**Synthesized from four literature scans (methods-frontier, applied-gaps, software-venues, frontier-forecasting) · 2026-08-17**

**Grounding notes (verified against the repo, not the reports):** tsecon's public git history begins **2026-07-17** (50 commits, all within the last month). This means **JOSS's hard gate of 6+ months of distributed public history is not met until ~mid-January 2027**, and the "evidence of research use" gate is not yet met either. This inverts the naive plan: the *replication/paper* opportunities below are not just publications — they are the research-use evidence JOSS requires. Also verified in-repo: `panel_lp`, `lp`/`lp_iv`/`lp_state`/`smooth_lp`/`quantile_lp`, `proxy_svar`, `sign_restricted_svar`, `narrative_svar`, `robust_svar_bounds` (Giacomini-Kitagawa already shipped), BVAR (Minnesota/GLP/SSVS), MIDAS, `dfm_nowcast`+`dfm_news`, quantile regression + GaR, DM/GW/CW tests, backtest engine, bootstrap engine, a Ramey-Zubairy replication, Monte Carlo coverage suites, and a JOSS paper draft. EVT and conformal (ACI/EnbPI) are on the roadmap but unshipped. *(Status as of 0.8.0 — repository audit, September 2026: EVT shipped in 0.3.0 (`gpd_fit`/`gev_fit`) and conformal split/EnbPI/ACI in 0.5.0; of the ranked items, #1 SPJ shipped in 0.3.0, #2 LP-DiD core in 0.4.0, #4(b) the Gertler-Karadi replication in 0.4.0, and #5's core in 0.5.0. The rest is re-verified row by row in `docs/roadmap/_repo_audit/ledger.md` (RC19-*).)* **Caveat:** all four scans report that direct fetches of arxiv.org/RePEc were proxy-blocked; citations below came from search metadata and fetchable mirrors and must be spot-verified before any submission is drafted.

Effort scale: **S** = days–2 weeks · **M** = 2–8 weeks · **L** = a quarter or more. Ranking = (impact × fit) / effort.

---

## 1. Ranked Top-10 Contribution Opportunities

### 1. Split-panel-jackknife (Nickell-corrected) panel LP — *S effort, first-Python, correctness fix to shipped code*
- **What:** FE panel LPs carry an asymptotic Nickell bias that invalidates t-tests even at large T; the split-panel jackknife restores valid inference. Add an SPJ option + bias-aware SEs to the existing `panel_lp`.
- **Citations:** Mei, Sheng & Shi, *J. International Economics* 2026 (https://www.sciencedirect.com/science/article/abs/pii/S0022199625001679; arXiv 2302.13455). Reference code is **R-only** (https://github.com/zhentaoshi/panel-local-projection).
- **Why tsecon:** `panel_lp` already ships; this is a small estimator delta with an R golden fixture available; it converts an already-shipped function from "known-biased" to "state of the art." Published-journal theory (not a working paper), so low scientific risk.
- **Effort:** S. **Deliverable:** software feature + model-card update; a headline bullet in the JOSS paper's state-of-the-field section.

### 2. LP-DiD (Dube-Girardi-Jordà-Taylor) on the panel-LP core — *M effort, first-Python, largest applied audience*
- **What:** LP event-study difference-in-differences with clean-control conditions (avoids negative weights), non-absorbing treatment, pooled ATT. Optional stretch: doubly robust LP-DiD (AIPW), which has **no public code anywhere**.
- **Citations:** Dube, Girardi, Jordà & Taylor, NBER w31184 rev. 2025 (https://www.nber.org/papers/w31184); Uhr & Moura, arXiv 2604.27035 (2026). Incumbents: Stata `lpdid` (https://github.com/danielegirardi/lpdid), R port (https://alexcardazzi.github.io/lpdid_vignette.html) — **no Python implementation** per Report 1's search.
- **Why tsecon:** The DiD user base is far larger than the macro-LP user base — this is the single biggest adoption lever in the list. Stata `lpdid` provides exact golden fixtures, matching tsecon's validation-first policy. Roadmap Module 07 already names LP-DiD as in-scope core.
- **Effort:** M (core LP-DiD); DR variant adds M and is **speculative** (2026 preprint, unverified, no reference implementation to validate against — sequence it second). **Deliverable:** software feature; optionally a short arXiv note "LP-DiD in Python" that doubles as JOSS research-use evidence.

### 3. Joint-across-horizon LP inference: sup-t bands, significance bands, wild block bootstrap — *M effort, first-Python, closes the field's flagged inference gap*
- **What:** Pointwise LP bands understate joint uncertainty across horizons; cumulative-LP SEs need the joint covariance. Implement simultaneous (sup-t) bands, Jordà significance bands, and the wild block bootstrap for LP/LP-IV.
- **Citations:** Inoue, Jordà & Kuersteiner, forthcoming *Econometrics Journal* (FRBSF WP 2024-29, https://www.frbsf.org/wp-content/uploads/wp2024-29.pdf; arXiv 2306.03073); Jordà & Gadea, FRBSF WP 2025-21 (https://www.frbsf.org/wp-content/uploads/wp2025-21.pdf); Jordà, CEPR DP18271 (https://ideas.repec.org/p/cpr/ceprdp/18271.html); benchmark practice: Montiel Olea, Plagborg-Møller, Qian & Wolf primer (https://economics.mit.edu/sites/default/files/2025-03/lp_var_primer.pdf). Author code is **Stata/Matlab only**.
- **Why tsecon:** The LP suite and the wild/block bootstrap engine already exist; Rust makes bootstrap-of-bands cheap; the Monte-Carlo coverage audit this literature explicitly demands is tsecon's existing house infrastructure. Roadmap already lists sup-t bands as Module 07 scope.
- **Effort:** M. **Deliverable:** software feature + a published MC coverage-audit doc; feeds directly into opportunity #6.

### 4. Weak-proxy robust inference + Gertler-Karadi "proxy zoo" replication — *M effort; one workstream, two deliverables*
- **What:** (a) Feature: add the Angelini-Cavaliere-Fanelli moving-block-bootstrap proxy-strength pre-test, weak-IV-robust (AR-type/MSW) confidence sets, and the Lewis-Mertens generalized first-stage test to the shipped `proxy_svar`. (b) Paper: an independent Python replication of Gertler-Karadi (2015) plus the Doko Tchatoka-Haque finding that GK output effects vanish post-1984 under weak-identification-robust inference, run across the 8-proxy "proxy zoo" grid.
- **Citations:** Angelini, Cavaliere & Fanelli, *J. Econometrics* 2024 (https://www.sciencedirect.com/science/article/pii/S0304407623003202); Lewis & Mertens, FRBNY SR1020 (https://www.newyorkfed.org/medialibrary/media/research/staff_reports/sr1020.pdf, Matlab-only); Montiel Olea-Stock-Watson (https://www.princeton.edu/~mwatson/papers/JOE_Publication_SVARIV.pdf); Doko Tchatoka & Haque, *Economic Record* 2024 (https://onlinelibrary.wiley.com/doi/10.1111/1475-4932.12801); "Proxy Zoo" (https://arxiv.org/pdf/2601.11195). Nothing exists in Python or R.
- **Why tsecon:** `proxy_svar` ships already; MSW weak-IV robust inference is named in the roadmap; the JCRE Kilian-replication precedent (Ryan & Michieka 2025, https://jcr-econ.org/not-all-oil-price-shocks-are-alike-replication/) shows "reimplement in an open ecosystem + extend + robust inference" is an accepted publication type. This is the fastest credible path to a **peer-reviewed citation using tsecon** — exactly what JOSS's research-use gate needs.
- **Effort:** M (feature) + M (replication paper). **Deliverable:** software feature + JCRE (or JAE replication section) submission.

### 5. Conformal prediction module over all tsecon forecasters — *M effort, roadmapped, broadest non-econ audience*
- **What:** Model-free `conformal` module — split-CP, ACI, decaying-step-size online CP, EnbPI, SPCI-style residual-quantile regression, multi-horizon variants — wrapping any tsecon forecaster (ARIMA/VAR/GARCH/DFM/backtest engine). Skip deep-net methods (HopCPT).
- **Citations:** EnbPI (https://arxiv.org/pdf/2010.09107); SPCI (https://arxiv.org/pdf/2212.03463); Angelopoulos-Barber-Bates ICML 2024 decaying step sizes (https://dl.acm.org/doi/10.5555/3692070.3692135); NeurIPS 2025 tutorial (https://neurips.cc/media/neurips-2025/Slides/118881.pdf); survey (https://arxiv.org/pdf/2601.18509). Reference implementations MIT/BSD (EnbPI repo, MAPIE) — license-safe to read.
- **Why tsecon:** Already in the Module 09 roadmap; the algorithms are simple residual-tracking loops (ideal Rust targets); no econometrics library wraps conformal around classical estimators; and tsecon's MC coverage-audit harness is *literally* the finite-sample-coverage validation this literature requires. Strong JOSS "state of the field" differentiator with an audience beyond economics.
- **Effort:** M. **Deliverable:** software feature + coverage-audit gallery page; candidate for a standalone SciPy Proceedings talk.

### 6. Independent Monte-Carlo verification of the LP-vs-VAR primer — *S/M effort, fastest publishable artifact*
- **What:** Reproduce and extend the coverage/bias-variance simulation claims of the NBER Macro Annual primer ("LP CIs robust; short-lag VAR CIs severely undercover") from its MATLAB-only code, across broader DGPs/horizons/bootstrap variants.
- **Citations:** Montiel Olea, Plagborg-Møller, Qian & Wolf, NBER Macro Annual 2025 (https://www.nber.org/papers/w33871; https://arxiv.org/abs/2503.17144); companion "Unpleasant VARithmetic" (https://arxiv.org/abs/2405.09509); MATLAB code (https://github.com/ckwolf92/lp_var_nberma).
- **Why tsecon:** The repo already contains LP-vs-VAR "frontier experiments"; LP + lag-augmented inference + BVAR live in one library; Rust makes the full MC grid cheap. The primer is becoming field-wide guidance — an independent verification is exactly the I4R robustness-reproduction genre, and a Python port becomes *the* reference implementation.
- **Effort:** S/M (leverages existing experiments; grows with #3). **Deliverable:** I4R discussion paper (EconStor) + executable-doc gallery entry.

### 7. Fast "soft" sign-restriction posterior sampling — *M/L effort, first-Python, the Rust speed story made literal*
- **What:** Replace accept-reject sign-restriction sampling (which collapses under many/tight restrictions) with the Read-Zhu smooth-penalty MCMC target; integrate with the already-shipped narrative restrictions and Giacomini-Kitagawa `robust_svar_bounds`.
- **Citations:** Read & Zhu, RBA RDP 2025-03 (https://www.rba.gov.au/publications/rdp/2025/2025-03/introduction.html; arXiv 2603.27088, journal-accepted); Giacomini & Kitagawa, *Econometrica* 2021 (https://onlinelibrary.wiley.com/doi/full/10.3982/ECTA16773); Giacomini-Kitagawa-Read JBES narrative critique (https://www.tandfonline.com/doi/full/10.1080/07350015.2022.2115496). No public package found; R `bsvarSIGNs` does accept-reject only.
- **Why tsecon:** The entire sign/zero/narrative/robust-bounds stack is shipped — this is a sampler swap, not a new module, and the paper's motivation *is* computational cost. Benchmark against RBA code; replicate an Antolín-Díaz-Rubio-Ramírez-style narrative result under prior-robust inference as the demo. Note honestly: the Woźniak `bsvars`/`bsvarSIGNs` group is the likely reviewer pool — position as first Python-native unified frequentist+Bayesian suite, not as beating bsvars at Bayesian computation.
- **Effort:** M/L. **Deliverable:** software feature + benchmark note; the narrative-robustness replication is a possible JBES/JAE comment (**speculative** as a standalone paper).

### 8. Growth-at-Risk robustness horse-race (quantile GaR vs EVT tails) — *M effort, IMF/central-bank adoption channel*
- **What:** Implement EVT (POT/GPD, GEV — already on the roadmap's build-next list with scipy goldens) and run an out-of-sample horse-race: Adrian-Boyarchenko-Giannone quantile GaR vs the Adrian-Sasaki-Wang EVT-based tail estimator vs calibrated quantiles, with coverage-audited intervals.
- **Citations:** Adrian, Sasaki & Wang, *J. Econometrics* 2026 (https://arxiv.org/abs/2508.00263; https://www.sciencedirect.com/science/article/pii/S0304407626000564) — Adrian critiquing his own canonical AER 2019 framework; calibrated quantile GaR (https://arxiv.org/pdf/2411.00520); live policy use: IMF GFSR Apr 2025 ch.1 (https://www.imf.org/-/media/files/publications/gfsr/2025/april/english/ch1.pdf).
- **Why tsecon:** `quantile_regression` + `growth_at_risk` shipped; EVT is next on the roadmap anyway; the contested quantity (tail-quantile coverage) is what tsecon's MC audit machinery measures. No open Python benchmark exists; doubles as an IMF/central-bank adoption driver.
- **Effort:** M (EVT feature + horse-race study). **Deliverable:** software feature + replication/horse-race paper (IJCB, JCRE, or JAE replication section).

### 9. Bayesian quantile VAR via multivariate asymmetric-Laplace likelihood — *M/L effort, near-greenfield, connects three shipped modules*
- **What:** MAL-likelihood QVAR (Gibbs via the AL scale-mixture-of-normals representation), composable with the shipped Minnesota priors; quantile IRFs feeding the GaR module.
- **Citations:** Iacopini, Poon, Rossini & Zhu, JEDC 2023 (https://arxiv.org/abs/2209.01910); time-varying-volatility extension (https://arxiv.org/abs/2211.16121); ECB WP 2983 (https://www.ecb.europa.eu/pub/pdf/scpwps/ecb.wp2983~ad23b7c8e2.en.pdf). No packaged implementation in any language — author replication code only.
- **Why tsecon:** BVAR + quantile solver + GaR already shipped; the sampler is textbook-level. **Honesty flag:** no incumbent package means no golden fixture — validation must run through simulation-based calibration and MC coverage instead, a deliberate exception to tsecon's fixtures-first policy that should be stated as such. Related first-Python quantile-LP structural identification (Ruzicka, AEA 2025, https://www.aeaweb.org/conference/2025/program/paper/fBArK44e) is **speculative** (unpublished WP) — defer.
- **Effort:** M/L. **Deliverable:** software feature (a genuine first); later a JSS-style module paper.

### 10. Climate-damages reconciliation: Bilal-Känzig vs Nath-Ramey-Klenow in one framework — *L effort, highest visibility, highest execution risk*
- **What:** The two designs disagree by an order of magnitude on the SCC (>$1,200/ton vs ~$80): global-temperature time-series shocks vs country-panel lag-augmented LPs. Run both specifications through one validated LP/LP-IV/VAR engine with uniform HAR inference and lag augmentation; map the sensitivity surface over identification/horizon/level-vs-growth choices.
- **Citations:** Bilal & Känzig, QJE 2026 (https://www.nber.org/papers/w32450; code https://github.com/dkaenzig/global-temperature-shocks); Nath, Ramey & Klenow (https://www.nber.org/papers/w32761, rev. at AER); CBO reliance (https://www.cbo.gov/system/files/2025-02/61186-Climate-GDP.pdf). No independent reconciling replication found at I4R/JCRE.
- **Why tsecon:** `lp` (lag-augmented default), `panel_lp` (+#1's SPJ correction — directly relevant to NRK's panel design), and uniform HAC/HAR inference in one engine is precisely the apparatus needed. **Honesty flag: speculative** — largest data/execution burden in this list, both papers are moving targets (AER revision in progress), and the deliverable's value depends on the sensitivity analysis actually discriminating between designs. Do it *after* #1/#3 ship, which it depends on.
- **Effort:** L. **Deliverable:** I4R robustness reproduction or JCRE article — the highest-citation-potential single item here.

### Honorable mentions (tracked, not ranked)
- **GSULP Bayesian joint-LP systems** (Huber-Matthes-Pfarrhofer, https://arxiv.org/abs/2410.17105): first-anywhere opportunity — the authors' repo is verifiably empty as of 2026-08-17 — but that also means it could be filled by them at any time; heavy GP-MCMC build. Speculative.
- **Generalised-Bayes outlier-robust Kalman update** (Duran-Martin et al., https://arxiv.org/abs/2405.05646): closed-form, S/M effort, drop-in for the DFM/nowcasting stack; strong candidate to attach to any nowcasting workstream.
- **Distribution-generic score-driven (GAS) engine** incl. van Heel invertibility diagnostics absent from all packages (Holý, R Journal 2026, https://journal.r-project.org/articles/RJ-2026-002/): first golden-fixture-validated Python GAS suite; L effort; shares a "score engine" primitive with robust filtering.
- **HD-LP (desparsified lasso)** (Adamek-Smeekes-Wilms, https://arxiv.org/abs/2209.03218; R `desla`): first-Python, rayon-friendly; niche audience — later.
- **NY Fed Nowcast 2.0 / SR1152 replication** (https://www.newyorkfed.org/research/staff_reports/sr1152): publishable and adoption-relevant, but the t-errors+TVP DFM is what broke every prior open port — L, high risk.
- **TSFM contamination-safe benchmark harness** (leakage findings: https://arxiv.org/abs/2510.13654; GIFT-Eval https://arxiv.org/abs/2410.10393): tsecon's DM/GW/MCS tests + leakage-aware splits fill a real gap, but it is partly covered by the roadmap's Module 10 harness already.

---

## 2. Do Next Quarter (Q4 2026): three items

**A. "LP inference hardening" release — items #1 + #3 shipped together as tsecon 0.3.**
First steps: (i) generate golden fixtures from the Mei-Sheng-Shi R repo (`zhentaoshi/panel-local-projection`) for FE and SPJ panel LP; (ii) implement significance bands and sup-t bands on `lp`/`lp_iv` using the existing bootstrap engine, wild-block variant per Jordà-Gadea; (iii) run and publish an MC coverage audit (pointwise vs joint, across the primer's DGPs — this simultaneously seeds item #6); (iv) spot-verify the Inoue-Jordà-Kuersteiner and Jordà-Gadea papers directly (proxy blocked the scans' fetches).

**B. LP-DiD (item #2), core estimator only.**
First steps: (i) install Stata `lpdid` or use its published example output to build golden fixtures; (ii) implement clean-control LP-DiD on the `panel_lp` core (pooled ATT, non-absorbing treatment); (iii) validate against the Stata and R implementations on the same dataset; (iv) write the model card + a worked gallery example. Defer the doubly-robust variant until the Uhr-Moura preprint is verified and stable.

**C. Proxy-SVAR workstream (item #4): feature first, replication started.**
First steps: (i) implement the MSW weak-IV-robust confidence set and Montiel Olea-Pflueger/Lewis-Mertens first-stage diagnostics on `proxy_svar`; (ii) reproduce Gertler-Karadi 2015 baseline as an executable gallery doc (the RZ2018 replication template already exists in-repo); (iii) extend to the post-1984 sample split per Doko Tchatoka-Haque; (iv) draft the JCRE submission following the Ryan-Michieka oil-shock template. Target: submission by end of Q1 2027 — timed so a peer-reviewed use-case exists when the JOSS window opens.

*Why these three:* they are the highest (impact×fit)/effort items, they share infrastructure, and together they manufacture the two things the venue analysis says tsecon lacks — time (the JOSS 6-month clock runs regardless) and research-use evidence (B and C produce it).

---

## 3. Venue Map

| Venue | Role for tsecon | Gates / costs | Timing |
|---|---|---|---|
| **arXiv (econ.EM / stat.CO)** | Immediate preprint for every paper-shaped deliverable (#4, #6, #8, #10); establishes priority on first-anywhere features (#9) | None | Now, continuously |
| **JOSS** | Primary archival software paper (whole-library) | **6-month public-history gate: not met until ~Jan 2027** (repo public 2026-07-17); research-use evidence required (aspirational statements insufficient — fulfilled by #4/#6/#8 outputs); 2026 five-section format (statement of need / state of the field / software design / research impact / **mandatory AI-usage disclosure**, non-disclosure = desk rejection); state-of-the-field must engage `bsvars`/`bsvarSIGNs`/`lpirfs`/`BVAR` | Submit **Q1-Q2 2027**; expect ~6-10 months to publication (pydynpd precedent) |
| **SoftwareX** | No-gate fallback or complement if a 2026 archival DOI is wanted | ~$1,560 APC; 3,000-word template; lower econ/stats prestige | Usable now |
| **JSS** | Prestige software paper, later — scope to **one module** (LP suite or SVAR identification suite) with full math + exact replication scripts, not the 126-function surface | ~1-2 years calendar time; full reproducibility of every figure/table; JOSS+JSS on one codebase permitted with disclosure | Submit 2027, after the module in question is feature-complete |
| **JCRE** | Replication articles #4 (GK proxy) and #8 (GaR); direct precedent for "reimplement in open ecosystem + extend data + robust inference" (Ryan-Michieka 2025) | Peer-reviewed, OA, no fees | First submission target Q1 2027 |
| **I4R (EconStor DP series)** | Robustness reproductions of AER/QJE-tier results: #6 (LP-vs-VAR primer), #10 (climate); Replication Games = co-authorship + community entry | Scope covers AER, AEJ:Macro, QJE-tier | #6 feasible H1 2027; #10 later |
| **JAE replication section** | Alternative home for #4/#8 if JCRE fit is poor | Mandatory data archive | As papers mature |
| **IJCB / Econometrics & Statistics / JSCS** | #8 (GaR, policy audience); methods+software hybrids like #9 (bvhar precedent at JSCS) | Standard review | 2027 |
| **SciPy Proceedings** | Visibility for the conformal module (#5) and the Rust-core architecture | Low bar, not archival prestige | Next SciPy CFP |
| **R Journal** | **Skip** unless an extendr R wrapper ships (out of scope) | — | — |

**Sequencing logic:** ship features (#1-#3, #5) and post arXiv notes now → replications (#4, #6, #8) through JCRE/I4R in Q1-Q2 2027, generating peer-reviewed research use → JOSS whole-library submission the moment both gates clear (~Q1-Q2 2027) → JSS scoped module paper as the prestige capstone in late 2027+. Before drafting anything: re-verify every citation above against the primary source — all four scans were working from search snippets due to proxy egress blocks, and several 2026 arXiv IDs and forthcoming-journal claims are unconfirmed.
