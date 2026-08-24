# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/); versioning follows the
pre-1.0 policy in [ROADMAP.md](ROADMAP.md) (minor = breaking allowed, patch =
fixes) until 1.0, then strict [SemVer](https://semver.org/).

## [Unreleased]

### Added

- **`proxy_ar_sets(rf_method="second_order_bc")`** — the roadmap-note-21
  follow-up on `second_order`'s residual ~2pp at `h=12`: the same seeded
  second-order simulation with the coefficient draws centred at Pope (1990)
  bias-corrected coefficients (Kilian stationarity shrinkage), implemented as
  `tsecon_ident::proxy_ar::pope_bias_corrected_coefs`. Measured on the same
  500 seeded replications as `second_order`: the only arm at-or-above nominal
  at every horizon on both DGPs (`h=12`: 0.889 → 0.964 → **0.982** on the
  card VAR(2); 0.830 → 0.932 → **0.966** on the routine VAR(1)), at a further
  width price (median ~1.8x the delta width at `h=12`, vs ~1.45x) — a
  conservative floor, not a calibration, and documented as such. Boundedness
  is bit-identical across all three `rf_method`s; the default remains
  `"delta"`. Crate tests pin the AR(1) closed form `-(1+3a)/T` exactly and
  the experiment harness's NumPy transcription at 1e-10.

### Measured

- **The interval-coverage audit now measures the five families it previously
  listed as unmeasured** — a new registry module
  (`docs/examples/coverage/proxy_garch_tail.py`, 13 probes; the registry
  grows 50 → 63) covering `growth_at_risk` (`bse` vs `bse_powell` on an
  exact-truth overlap design), `proxy_svar_bands` (moving-block Hall/Efron
  and the wild reproduction arm), `proxy_ar_sets` (all three `rf_method`s,
  paired), `garch_fit` (`se_mle` vs `se_robust` under Gaussian and t(5)
  QMLE), `flp`/`flp_scenario` (the generated-regressor warning, priced
  against the external-score and `w'beta` exempt routes), plus verified
  no-interval rows for `nongaussian_svar` and the GARCH `variance_forecast`
  (key-set tripwires). Headline findings are on the
  [interval-coverage page](docs/examples/interval-coverage.md).

## [0.4.0] - 2026-08-18

### Added — the weak-proxy workstream

- **`proxy_first_stage`** — the Montiel Olea-Pflueger effective first-stage F
  (equal to the robust F in the just-identified single-proxy case, Windmeijer
  2025) under classical/HC1/HAC-Bartlett variance, with the τ-based
  noncentral-χ² critical values (**23.11** for the conventional 10%-bias bar —
  not the folklore 10), the implied worst-case-bias bound `tau_bound`, and
  weak-verdict flags; stamped into every `proxy_svar` result as `first_stage`.
  Pinned against statsmodels (rtol 1e-9) and scipy's `ncx2.ppf` (1e-6);
  reproduces the published `weakivtest` table (37.418/23.109/15.062).
- **Gertler-Karadi (2015) replication** — `proxy_svar` upgraded to a
  published-result golden on the authors' AEJ dataset
  (`fixtures/gertler_karadi.csv`): first-stage F **21.5499 vs the paper's
  21.55** and robust F 17.50 vs ~17.5, the Figure-1 IRF shapes under the
  +20bp normalization, the wild-vs-moving-block band contrast (GK's
  significance pattern reproduced under their own asymptotically invalid
  method, mostly undone by the valid Jentsch-Lunsford bands), and the
  Doko Tchatoka-Haque post-1984 weakening reproduced end to end — with the
  new diagnostic flagging that even GK's baseline effective F fails the MOP
  τ=10% bar while passing folklore F>10.

### Added — estimators

- **`acm_term_premium`** (Adrian-Crump-Moench 2013): the three-step
  regression-based Gaussian affine term-structure estimator — PCA factors,
  factor VAR(1), excess-return regressions with convexity-adjusted λ₀/λ₁,
  and affine recursions decomposing fitted yields into risk-neutral yields
  and the term premium. Golden-gated at 1e-8 against an independent NumPy
  pipeline (measured ≤1e-11) on a simulated affine DGP (known-truth premium
  recovery corr 0.98) and the vendored 1961–2014 GSW panel; validated
  against the NY Fed's published ACMTP10 (corr 0.985, RMSE 0.31pp; the
  documented estimation-sample sensitivity demonstrated, not hidden).
- **`copula_fit` / `copula_select` / `pseudo_obs`** — static bivariate
  copulas (new crate `tsecon-copula`): Gaussian/Student-t/Clayton/Gumbel/
  Frank by full MLE (observed-information SEs) or Kendall-tau inversion
  (t's ν profiled), AIC/BIC selection with a teaching verdict, closed-form
  tau maps (Frank via exact Debye D1) and tail-dependence coefficients.
  Golden-pinned to statsmodels 0.14.6 densities/CDFs at 1e-10, a
  scipy-polished MLE of the statsmodels log-density (statsmodels exposes no
  copula MLE), kendalltau at 1e-15, and Owen's-T/quad for the CDFs
  statsmodels lacks. Validation-first found a **fourth reference defect**:
  statsmodels' `StudentTCopula.dependence_tail` operator-precedence bug
  (0.1438 where the true Demarta-McNeil value is 0.2532) — recorded, the
  correct closed form pinned by numeric copula limits. Bivariate this
  slice; d>2, rotations, and dynamic copulas deferred and stated.
- **`lp_did`** — LP-DiD (Dube-Girardi-Jordà-Taylor): clean-control
  event-study difference-in-differences on the panel core — not-yet/never-
  treated/stabilized controls, pre-trend horizons, equally-weighted-ATT
  reweighting, pooled ATTs, non-absorbing treatments, entity-clustered SEs
  in the authors' fixest convention. **Reference-run golden**: pinned at
  1e-10 against an actual R/fixest execution of the authors' example code
  (`fixtures/lpdid.json`; independent NumPy cross-check at 5.3e-15), with
  the clean-control condition asserted in CI — a naive all-controls variant
  loses 56.5% of a heterogeneous-cohort effect where LP-DiD lands within
  0.1%. The first Python implementation.
- **`scale_ar`** — the GLP-exact residual-scale convention on
  `bvar_fit`/`bvar_irf_draws`/`bvar_hierarchical` (default 4, unchanged and
  verified bit-identical to 0.3.0 by an out-of-band wheel-hash comparison):
  `scale_ar=1` matches Giannone-Lenza-Primiceri's own `setpriors.m`. With
  it, the GLP *design* replication becomes a **point replication**: on
  GLP's own Stock-Watson panel (now committed from the public FRBNY mirror
  with its redistribution notice kept) the selected tightness lands on the
  published Figure-1 modes — 0.420 vs 0.449±0.03 (small VAR), 0.1716 vs
  0.172±0.01 (medium) — where the AR(4) default selects 0.260/0.142.

### Fixed — audit round 7 (docs/roadmap/22-audit-round-7-findings.md)

- **`garch_fit` no longer returns silent all-NaN standard errors at a
  parameter boundary** (open since round 1; 24 of 50 boundary-battery fits).
  The SE path now uses a reduced Hessian over the interior directions with
  per-parameter `se_valid`/`boundary` flags, a `boundary_note`, and an
  exposed `converged` — 24/50 silent NaN rows → 0, all flagged with finite
  interior SEs; the arch-pinned fixture cases are bit-identical pre/post.
- **`garch_fit` and `dcs_local_level` estimation is now scale-adaptive**
  (standardize-and-map-back, the arch `rescale` trick): cross-scale
  disagreements 0/320 on the well-identified battery (was 93/320 on the
  boundary-attracted one, now 75/320 — every one flag-covered,
  loglik-equivalent ridge landings); power-of-two rescalings commute
  **bit-exactly**, pinned as a same-run invariant. The finder half caught
  the week-old DCS-Laplace path converging to unit-dependent points
  (11/20 seeded series, κ moving up to 57%, all certified converged) —
  fixed by the same route, 11/20 → 4/20 irreducible kink-surface rounding.
- **Nelder-Mead's mixed-scale initial simplex — the brief's oldest
  theoretical item, realized and fixed**: a near-zero coordinate's simplex
  edge could fall below `x_tol`, pre-converging before iteration 1; edges
  are now floored at scipy's `zdelt` (bit-identical vertices for
  |x0| ≥ 0.005).
- **Negative integer arguments now raise a teaching `ValueError`
  library-wide** (naming the function and parameter, chaining the original)
  instead of PyO3's raw `OverflowError: can't convert negative int to
  unsigned` — fixed once at the `_coerce` choke-point.
- Cosmetic: `forecasting.md` still called `gpd_fit` unshipped; a stale
  printed digit in the volatility card.

### Fixed — caught by CI (Windows) after the 0.4.0 freeze

- **The Student-t log-density constant is now cancellation-safe at large
  `nu`** (new `tsecon_stats::special::ln_gamma_half_ratio`, asymptotic
  branch above `x = 1e3`; used by `StudentT::ln_pdf`, the Hansen skew-t,
  and both `tsecon-gas` models). The literal `lnΓ((ν+1)/2) − lnΓ(ν/2)`
  difference turns into rounding noise as `nu` rides toward the Gaussian
  boundary — measured: a clean-data `dcs_local_level(density="t")` fit
  reported `loglik = +54230` on a series whose Gaussian log-likelihood is
  `−744`, and the optimizer climbed that noise to `nu ≈ 1e16`. Post-fix
  the t fit's loglik matches the Gaussian fit's to `1e-12` on the clean
  fixtures (the mathematical nesting bound), pinned by new Rust and
  Python regression tests. Interior optima (Nile `nu = 20.3`,
  contaminated fixtures `nu ≈ 2`) are bit-identical pre/post: the literal
  difference is kept below the seam.
- **`converged` is now deterministic on the `nu → ∞` Gaussian ridge**
  (`dcs_local_level` and `gas_volatility`, Student-t density). Whether
  the simplex happens to collapse on that flat ridge is a libm rounding
  accident — windows-latest CI certified `converged=True` on the exact
  fixture where Linux reported `False`, failing the documented
  "honest flag" contract. Past `tsecon_gas::kernel::NU_GAUSSIAN_RIDGE`
  (`nu > 1e3`, far above any genuine interior optimum observed and far
  below any tolerance stop on the stable ridge) the flag is forced
  `False` on every platform; the estimates themselves are untouched.

## [0.3.0] - 2026-08-18

### Added — the replication gallery, opened wide

Five published-paper replications on public data committed to the repository,
each upgrading a validation grade that previously rested on a transcription or
an internal oracle. Every page follows the Ramey-Zubairy pattern: a vendored
CSV with full attribution, an executable doc, and a CI test pinning the
published numbers at stated tolerances.

- **Hamilton (1989) — `markov_switching_ar` upgraded to a dual golden.** The
  founding Markov-switching paper, run offline on Hamilton's own GNP-growth
  series (`fixtures/hamilton_gnp.csv`, vendored from statsmodels' test suite):
  regime means +1.165/−0.344 vs the published +1.16/−0.36, persistences
  0.902/0.763 vs 0.9049/0.7550, ~10/~4-quarter expected durations — while a
  statsmodels cross-fit on identical data (itself pinned to its E-views
  benchmark) agrees to ≤0.016 on every parameter and produces bit-identical
  NBER recession calls (120/131 quarters). The CI guard also deliberately pins
  the binding's non-exposure of the common AR coefficients so the comparison
  extends the day that key appears.
- **Uhlig (2005) — `sign_restricted_svar` upgraded from property-only to a
  published-result replication** on the paper's own monthly dataset
  (`fixtures/uhlig2005.csv`, via VARsignR): VAR(12), his K=5 restriction set —
  **no price puzzle** (deflator 84% quantile negative through month 60) and
  the **ambiguous output response** (68% band straddling zero at every
  horizon 6–60, within the paper's ~±0.2% range) reproduce with a 5.9%
  acceptance rate in ~3 s. Sampler-lineage differences (pure-sign rejection
  matched; Minnesota-NIW vs flat posterior) documented, not hidden.
- **Bai & Perron (2003) — `bai_perron` validated on the paper's own
  application** (`fixtures/realint_bai_perron.csv`, the US ex-post real
  interest rate 1961Q1–1986Q3): break dates **exact at every reported
  partition size** (1972Q3/1980Q3; +1966Q4 at m=3), segment means at
  published rounding, the published 5% sequential critical values verbatim,
  the SSR path matching R strucchange to every printed digit and the
  classical supF sequence matching Perron's own mbreaks to 3 d.p. (both run
  during development for corroboration). Honest gaps stated: the paper's
  HAC-robust supF (which selects 3 breaks where classical F and their own
  BIC select 2) and heterogeneity-robust CIs are out of scope.
- **Hansen (1999) — `setar` anchored to a published fit of real data**
  (`fixtures/sunspots_tong.csv`, the Wolf sunspots 1700–1988 under the
  Ghaddar-Tong transform): common-order p=11 SETAR — delay exact (d=2),
  threshold 7.4234 printing as the published 7.4 (identified up to one order
  statistic), and the seeded bootstrap linearity test rejecting at 0.02–0.03
  vs the paper's ~0.03. The regime-specific-order Tong-Lim fits the common-p
  design cannot express are named, not fudged.
- **GLP (2015) — `bvar_hierarchical` grounded in the published application
  *design*** (`fixtures/glp_smallvar.csv` — macrodata stand-ins, **not**
  GLP's Stock-Watson panel; stated in bold everywhere): their
  transformations, sample span, five lags, and Gamma hyperprior (verified
  against the authors' own `setpriors.m`, publicly mirrored) reproduce the
  paper's behavioural claims — tightness of a few tenths, looser than the
  0.2 folklore for the small VAR, tightening as the cross-section grows,
  dominance over fixed references. A development-time run on GLP's own data
  attributes the remaining level gap entirely to the documented AR(4)-vs-AR(1)
  residual-scale convention: switching it reproduces their published
  Figure-1 modes (0.449/0.172).

### Added — risk backtesting

- **`var_backtest`** — Kupiec (1995) unconditional coverage, Christoffersen
  (1998) independence and conditional coverage (with 0·ln 0 continuity for
  empty Markov cells), and the Engle-Manganelli (2004) dynamic quantile test,
  with a teaching verdict. One documented sign convention (return scale;
  violation = return < VaR); pre-computed hit sequences accepted. The DQ
  degrees of freedom are the **design rank**: a constant VaR path (collinear
  with the intercept) is dropped with honest df — statsmodels' pinv would
  silently miscount it — and unidentifiable designs raise. Golden-pinned to
  first-principles closed forms and a statsmodels-OLS DQ construction
  (`fixtures/var_backtest.json`), including a hand-derived n=250/5 case
  (LR_uc = 6.0715) and Jorion's J.P. Morgan-1998 example (LR_uc = 3.91);
  seeded size/power suites assert ≈-nominal size and the LR_ind/DQ-vs-Kupiec
  separation on clustered violations.

### Added — robust filtering (the lab's first graduation)

- **`dcs_local_level(y, density="t"|"laplace"|"gaussian")`** — the
  score-driven robust local level (Harvey 2013; Harvey-Luati 2014),
  graduated from the lab's strongest result: a bounded redescending t-score
  (−23%/−31% level RMSE vs the Gaussian control at 5/10% additive
  contamination, zero clean-data tax, the gain-collapse mechanism asserted),
  a Laplace sign filter, and a Gaussian case that is exactly the
  steady-state Kalman local level — golden-pinned to statsmodels
  `UnobservedComponents('llevel')` through the derived mapping
  `kappa = p/(1+p)`, `p = (q+sqrt(q²+4q))/2`, `scale² = σ²_ε/(1−kappa)`
  (loglik 1e-8, path 1e-6, fitted params 1e-4 vs a scipy same-criterion MLE)
  in `fixtures/tsecon-dcs.json`; t/Laplace are MC-recovery graded (no
  runnable reference exists — stated). Returns observed-information SEs and
  an honest `converged` (t on clean Gaussian data reports non-convergence at
  the ν → ∞ boundary rather than certifying it).

### Added — split-panel jackknife for panel local projections

- **`panel_lp(bias_correction="spj")`** — the Mei-Sheng-Shi (2026, J. Int.
  Economics) split-panel jackknife: median-split halves with full-panel
  leads/lags, `2F−(A+B)/2` corrected points, and — unlike the existing
  Dhaene-Jochmans `jackknife` flag, which the round-2 audit measured costing
  8pp of coverage by keeping the plug-in SE — **standard errors recomputed
  for the corrected estimator** (adjusted-score cluster/Driscoll-Kraay
  sandwiches per the authors' pLP reference implementation, transcribed
  verbatim; `nonrobust` refused). Transcription golden at 1e-10
  (`fixtures/panel_spj.json`) + seeded MC: at T=20, h=2 the FE Nickell bias
  −0.137 falls to +0.009 and coverage improves 0.74 → 0.82 (neither reaches
  nominal at T=20 — documented, with the residual attributed to
  Driscoll-Kraay's own short-T approximation). `jackknife=True` keeps its
  exact semantics (`bias_correction="dj"` alias); combining both raises;
  the result stamps `se_type`/`cumulative`/`jackknife`/`bias_correction`.

### Added — the interval-coverage audit closes the unmeasured-seven list

- The registry grows **40 → 50 outputs (21 → 28 functions)**: `quantile_lp`
  (near nominal on the canonical identified-iid-shock design — the card's
  transferred overlap warning does not bind there; the persistent-regressor
  hazard measured with and without the whitening lag controls, 0.909 vs
  0.758), `panel_lp` Driscoll-Kraay over (N, T) (T drives it, N does not;
  the naive cluster covariance on the same draws covers 0.20 — the DK
  default is doing real work), the SPJ correction re-measured at 8× the
  card's replications (bias removal corroborated exactly; the short-T
  coverage gain is +2.5pp, smaller than the card's 300-rep point numbers
  suggested — card softened), the official post-fix `lp(cumulative="both")`
  numbers (0.507 → **0.920** at T=400, h=12), FAVAR two-step bands priced
  against a true-factor oracle (**0.673** at N=20, T=800 — more T makes the
  generated-regressor cost worse, matching the guide's warning), `umidas`
  HAC intervals (slopes hold; the intercept 0.829 under AR(1) errors — the
  disclosed kernel regime), and verified ships-no-interval rows for
  `weighted_midas`/`dfm_nowcast`/`nelson_siegel` (key-set tripwire, so a
  future interval cannot appear unmeasured). `run_all.py --markdown` now
  emits the page's tables from the harvested results — nothing on the page
  is typed by hand, and `check_page.py` enforces it.

### Added — measured repairs for the two open inference problems

- **`proxy_ar_sets(..., rf_method="second_order", rf_draws=, rf_seed=)`** —
  second-order reduced-form propagation (`psi_reduced_form_cov_mc`: seeded
  antithetic coefficient draws through the exact MA recursion, equal to the
  delta method to first order plus the horizon-growing convexity the audit's
  one-sided misses traced to). Measured on the audit's own harnesses (500
  reps, estimand validated at T=200k): h=12 coverage **0.889 → 0.964** on the
  card DGP and **0.830 → 0.932** on the harder VAR(1), at ~1.15×/~1.45×
  median width (h=8/h=12), with weak-instrument boundedness bit-identical.
  The default stays `"delta"` (a default flip needs its own audit round);
  the card carries the six-arm comparison, including the structurally doomed
  bootstrap-critical-values direction (it *raises* the unbounded share —
  the bootstrap truth is the under-persistent fit itself).
- **`ivx_test(..., joint="bonferroni")`** — a joint verdict built from the
  scalar IVX tests (reject at level/k), leaning on the scalar test's measured
  deep-tail calibration. Measured over a 64-cell k×ρ×δ×n grid (2000
  reps/cell): size **0.011–0.059 everywhere**, where the chi-square default
  reaches 0.28–0.34 at k=8 — with power ~equal on sparse alternatives and
  ~20% lower on diffuse ones (stated on the card). The demeaned-variance and
  FM-normaliser directions were verified and discarded (worse and inert,
  respectively); the wild bootstrap fixes k=8 but breaks k=1 — all recorded
  in `docs/roadmap/21-long-horizon-and-joint-inference.md` with the seeded
  harnesses committed under `docs/examples/coverage/experiments/`. The
  default stays `"chi2"`; the card's advice now routes many-predictor joint
  tests to `"bonferroni"` instead of the old `alpha=0.50` workaround.

### Added — the unit-root battery completed (DF-GLS, Zivot-Andrews)

- **`dfgls`** — the DF-GLS unit-root test (Elliott-Rothenberg-Stock 1996):
  GLS detrending at the ERS local alternative (c̄ = −7 constant / −13.5 trend),
  AIC/BIC/t-stat lag selection on the GLS-detrended series, and the trendless
  ADF regression — statistic, selected lag, and nobs match `arch.unitroot.DFGLS`
  to < 1e-12 on `fixtures/dfgls.json` (Nile, seeded walks, trend-stationary,
  noise, fixed-lag/max-lags paths). P-values and critical values transcribe
  arch's DF-GLS response surfaces bit-for-bit, with the constants re-exported
  in the fixture so upstream drift is visible (attribution recorded in
  `THIRD-PARTY-LICENSES.md`, which now also documents the pre-existing
  MacKinnon-surface transcriptions). The GLS-detrending engine is shared
  internal machinery the Ng-Perron M-tests will reuse.
- **`zivot_andrews(y, regression=, trim=, max_lags=, autolag=, lags=)`** — the
  Zivot-Andrews (1992) unit-root test with one endogenous break (intercept /
  trend / both): minimum-t over trimmed candidate break dates, statsmodels'
  Baum single up-front autolag convention matched exactly (break-dummy timing,
  `int(n*trim)` trimming, the `"t"`-model ramp quirk — all documented in the
  module), `break_index` reported as the last pre-break observation.
  P-values/critical values interpolate the statsmodels-simulated null table
  (transcribed, BSD-3). Golden-pinned in `fixtures/zivot_andrews.json`:
  24 cases at 1e-10 relative on the statistic with break/lag exact; arch 8.0.0
  agrees on every expressible case (same Baum lineage — corroboration, not an
  independent derivation). The card flags the break-only-under-the-alternative
  criticism and points to Lee-Strazicich.

### Added — STL and the seasonal workflow

- **`stl(y, period, ...)`** — Cleveland et al. (1990) seasonal-trend
  decomposition using LOESS, a statement-for-statement port of the netlib
  `stl.f` semantics statsmodels preserves: cycle-subseries seasonal LOESS,
  3×MA + LOESS low pass, bisquare outer robustness loop, and the jump speedups
  replicated exactly. Parameters, defaults (trend/low-pass window rules; 2/15
  vs 5/0 inner/outer under `robust`) and the resolved `config` mirror
  `statsmodels.tsa.seasonal.STL`; pinned elementwise at 1e-8 (observed ~1e-12)
  across 20 cases in `fixtures/stl.json`. Requires `n >= 2*period` — R's
  `stl()` bound, where statsmodels silently misbehaves.
- **`seasonal_strength(y, period)`** — the Wang-Smith-Hyndman trend/seasonal
  strength measures from the STL fit, and **`nsdiffs(y, period)`** — the
  Hyndman-Khandakar seasonal-differencing advisor (D = 1 iff seasonal strength
  ≥ 0.64, the `forecast::nsdiffs(test="seas")` rule), in the `ndiffs` house
  style: per-order evidence, stop reason, teaching interpretation.

### Added — threshold autoregression

- **`setar(y, p, delay=1, trim=0.15, delays=None, ic="aic", constant=True)`** —
  the two-regime self-exciting threshold AR (Tong-Lim 1980) by concentrated
  least squares (Hansen 1997): trimmed order-statistic threshold grid with a
  `max(k+1, ceil(trim·n))` per-regime minimum, optional joint delay search on
  a common sample, per-regime coefficients with classical SEs, pooled and
  per-regime variances, AIC/BIC, and the full SSR profile. Golden-pinned at
  1e-10 against an independent NumPy transcription of the published algorithm
  (`fixtures/setar.json` — graded honestly: no third-party SETAR reference
  runs in the test environment); MC-validated threshold recovery (median
  absolute error 0.008 at T=400 over 200 seeded replications, reported on the
  card).
- **`setar_test(y, p, delay=1, trim=0.15, n_boot=499, seed=0)`** — the Hansen
  (1996) sup-F linearity test with the fixed-regressor wild bootstrap; the
  threshold is unidentified under the null (the Davies problem), so no
  chi-squared p-value exists or is reported. Rayon-parallel, one seeded Philox
  substream per replication — bit-identical at any thread count. Statistic
  golden-pinned; null rejection rate 0.08 at nominal 5% over 200 seeded series.

### Added — EVT tails (the first E11 slice)

- **`gpd_fit`** / **`gev_fit`** — extreme value theory, in the new
  `tsecon-evt` crate. Peaks-over-threshold GPD MLE over a default 0.90-quantile
  threshold with observed-information SEs and McNeil-Frey (2000) VaR/ES at the
  empirical exceedance rate; GEV MLE on block maxima (pre-computed, or series +
  `block_size`) with return levels. A shared `ln(1+ξx)/ξ` kernel switches to
  the documented Gumbel/exponential-limit branch below |ξ| = 1e-8 (continuity
  property-tested); the ξ ≤ −0.5 irregularity is reported but not certified
  (`se_valid`), and ξ ≤ −1 non-existence (uniform tails) is documented and
  tested. Golden-pinned to polished `scipy.stats.genpareto`/`genextreme` fits
  in `fixtures/tsecon-evt.json`: params at 1e-6, log-likelihood at the optimum
  at 1e-10, SEs at 1e-4, VaR/ES/return levels at 1e-5, on t(3), exponential
  (ξ = 0 boundary), negative-ξ, and real absolute-log-return data.

### Added — repository (not part of the wheel)

- **`lab/`** — a frontier-methods research bench outside the released surface:
  a from-scratch Taylor-Letham decomposable forecaster, asymmetric-Laplace
  score-driven quantiles, DCS robust local levels, LAD-ARMA, and a seeded
  five-experiment study against the shipped baselines with the losses reported
  as prominently as the wins (`lab/REPORT.md`). Graduation candidates named:
  the DCS-t robust local level and a VaR backtest battery.
- **`docs/roadmap/19-research-contributions.md`** — a ranked
  research-contribution scan (ten opportunities, a next-quarter shortlist, a
  venue map), sequenced against the JOSS six-month public-history gate.

### Added — seasonal ARIMA

- **`arima_fit(..., seasonal=(P, D, Q, s))`** — the multiplicative
  SARIMA(p,d,q)(P,D,Q)_s, closing the gap the docs have been honest about
  since 0.1.0. The seasonal and regular lag polynomials are multiplied into a
  single dense ARMA and run through the existing exact-MLE state-space engine;
  seasonal differencing follows the statsmodels order (seasonal first, then
  regular; `simple_differencing=True` semantics, `d + D*s` observations lost);
  forecasts are re-cumulated to levels through the augmented state — one
  cumulator per regular difference, one `s`-long delay line per seasonal
  difference — so the reported forecast variance is the exact cumulative one,
  seasonal stages included (the pure seasonal random walk reproduces
  `se_h = sigma * sqrt(ceil(h/s))` bit-for-bit). Seasonal parameters are named
  statsmodels-style (`ar.S.L12`, `ma.S.L12`), stationarity/invertibility is
  enforced per factor polynomial through the same Monahan transform, and
  Hannan-Rissanen starting values gain the seasonal lags. Golden-pinned to
  statsmodels `SARIMAX` in `fixtures/sarima.json`: the airline model
  `(0,1,1)(0,1,1)_12` on the real log Series G (fixed-parameter log-likelihood
  at 1e-8 relative, fit at the textbook `theta ~ -0.40`, `Theta ~ -0.56`,
  `cov_type='approx'` standard errors at 1e-4, 24-step levels forecasts against
  the statsmodels levels state-space form at 1e-6), a quarterly
  SAR(1)x(1)_4-with-constant, and the mixed `(1,1,1)(1,1,1)_4`. The Rust
  `ArimaSpec` gains `seasonal(P, D, Q, s)` and the results object
  `seasonal_ar()` / `seasonal_ma()`; CSS (`fit_css`) conditions on the expanded
  `p + s*P` presample.

### Fixed — audit findings (rounds 2–5)

- **`panel_fe` / `panel_lp` now refuse a design the fixed effects have
  absorbed** instead of returning a publishable t-statistic for it. The old
  guard was a Cholesky positive-definiteness test that fired only when the
  within-demeaned residue was bit-exactly zero: entity constants stored as
  ordinary doubles (log land area, a share in `[0, 1]`) slipped through, and
  the default cluster covariance turned the `O(1e-16)` residue into
  t-statistics that reached nominal 5% significance in 19.2% of audit draws —
  a statistic that moved when a constant was added to the data. The guard is
  now two scale-invariant checks (per-column absorption of the demeaned norm
  relative to the raw norm, plus the numpy/linearmodels singular-value rank
  criterion), matching the linearmodels `AbsorbingEffectError` refusal the
  docstring promises — and, unlike its predecessor, it is monotone: a merely
  ill-conditioned near-duplicate still fits while an exact duplicate raises.
  `panel_lp` additionally rejects an infeasible `max_horizon` up front as a
  sample-size error, rather than tripping over whatever symptom the shrunken
  per-horizon window produced first.
- **`engle_granger` raises instead of panicking when the sample has no more
  rows than step-1 design columns** — the `(0, k)` / `(1, k)` shapes escaped
  `except Exception` as `PanicException` through the Python boundary.
- **`proxy_svar_bands`'s proxy alignment can no longer panic** when `lags`
  exceeds the observation count (the same `PanicException` class).
- **`proxy_ar_sets(hac_lags=...)` with the default `variance="hc0"` now
  raises** instead of silently ignoring the argument — output used to be
  bit-identical with and without it. `hac_lags` parameterizes only the
  `variance="hac"` route, and the structural-identification model card now
  says so instead of implying `hac_lags` itself switches the estimator.
- **The `bvar_fit` model card had its two shrinkage dials swapped**: it called
  `lambda0` "overall tightness" and `lambda1` the "own-lag scale", while in
  the implementation (deliberately, and consistently with the conjugate
  Kronecker form) `lambda0` scales only the intercept prior and `lambda1` is
  the overall tightness on every lag coefficient, own and cross alike. A
  card-following user lowering `lambda0` for shrinkage got a pinned intercept
  and untouched dynamics. The card now matches the code, and states that the
  conjugate form cannot express the classic cross-variable `lambda2`.
- **The ARIMA model card mis-stated `arima_fit`'s defaults** (`p`, `d`, `q`
  "all default 0", `constant = False`): the shipped defaults are `p = 1` and
  `constant = True`, so with `d >= 1` the default fits a drift. The card now
  matches the code and says to turn the constant off deliberately.
- **`VARResults.irf_bands` and `var_irf_bands` now share the same default
  `band_seed` (20260807)**: the facade defaulted to 0, so the two documented
  routes to the same sup-t band returned different critical values for
  identical inputs. A regression test pins the two routes bit-identical at
  their defaults. Also swept the sup-t release's own doc surface:
  `testing.md` no longer claims "no function reports a simultaneous band"
  (the feature it was released alongside), and its stale tallies (`49`
  files / `13` unlisted, the RZ replication's test count, the
  "16 of the 40" validation split) are corrected or replaced with
  references to surfaces that cannot go stale.
### Fixed — the confirmed-open audit backlog (rounds 1–4)

- **`lp(se="hac", band=…)` now returns the `cov_se_max_rel_diff` the model
  card always promised** (the largest relative gap between the band's
  cross-horizon covariance diagonal and the per-horizon SEs — measured up to
  9.7% on a routine design, so not a promise about a number that is always
  ~0). `smooth_lp(band="sup-t")` returns it too; closed-form band routes
  report `None`.
- **`growth_at_risk` returns `bse_powell`** — computed, golden-pinned, and
  walked through the card's "How to read the output" since the feature
  shipped, but dropped at the binding. At `horizon=4` (where `hac_lags=3`)
  it differs from `bse` elementwise and was unrecoverable.
- **`markov_switching_ar` returns the full `(n, k)` smoothed and filtered
  probability matrices** the docs promised, not just the last regime's
  column (unrecoverable at `k >= 3`). `smoothed_prob_last_regime` is kept,
  bit-identical to the last column.
- **`gmm_nonlinear` now blames the moment function, not `initial`, when the
  callback's return has the wrong shape** — the old message told the user to
  reshape `initial`, and following that advice made things worse. The
  callback's return is validated before the Rust boundary: a bad return names
  `moments_fn`, states the `(n_obs, n_moments)` 2-D contract, and shows the
  `return g.reshape(-1, 1)` remedy (which now, verified, converges).
- **The interval-coverage page is generated-checked against the runner.** The
  published tables had silently dropped the `ols se_type="hc3"` row (the one
  estimator 0.2.0 added because of the previous audit — restored with its
  honest 0.863 UNDER verdict) and carried a second stale row (`iv_gmm` HAC
  stress at the pre-0.2.0 0.632 where the shipped default measures 0.842).
  Tables are regenerated from a full fresh run; a new
  `docs/examples/coverage/check_page.py` + binding-suite test now diff every
  page row against the runner's probe registry, so a dropped row fails CI
  instead of a promise.
- **The functional-shock family now carries the generated-regressor warning
  the library already prints for FAVAR and dynamic Nelson-Siegel**: `flp`'s
  per-element `se` conditions on `functional_pca` scores as if they were data
  (measured se/sd ≈ 0.66, and one-fifth of the truth on one column of the
  card's own worked example); externally supplied scores are unaffected and
  `flp_scenario`'s `w'beta` contrasts are algebraically immune — the card,
  guide, and docstring now say exactly that, and the coverage page's
  "not measured" list names it.
- **`ivx_test`'s joint-Wald size caveat is documented where the advice was.**
  Measured: size 0.056/0.105/0.164/0.269 at k=1/3/5/8 (n=250, ρ=1, δ=−0.9),
  non-convergent in n (0.22 at n=256000). The card's "keep the defaults" is
  scoped to the single-predictor test, the horse-race advice now names
  `alpha=0.50` beyond a few predictors, and `which-model-when.md`'s
  self-contradiction is resolved. Behaviour is unchanged (bit-identical
  probe pre/post) — this round documents; a size-restoring joint test is
  future work.
- **`panel_lp(jackknife=True)`'s cookbook advice matched the regime where it
  costs most.** The Dhaene-Jochmans correction fixes the bias but inflates the
  estimator's variance (+36–53% measured) while the reported `se` comes from
  the uncorrected fit, costing 6–8pp of coverage at short T — precisely where
  the cookbook recommended it. The cookbook, card, guide, and docstring now
  recommend it at moderate-to-long T and say what the `se` ignores at short T.
- **`iv_gmm`'s docstring leads with its `(x, z, y)` positional order** — three
  same-shaped float arrays, so a swap coerces cleanly and returns
  plausible-looking garbage; keywords recommended.
- **Two stale runtime docstrings fixed, with a tripwire.** `long_memory_d.__doc__`
  called `se` "asymptotic" — the exact label of the quantity round 1 proved
  ~25% too narrow; `predictive_regression.__doc__` named a `rho` key that does
  not exist. Both now name every returned key with the cards' semantics, and a
  new `test_docstring_keys.py` diffs `sorted(fn(...).keys())` against the
  docstring for these functions so this drift class fails a test.
- **`results.plot_estimates` (predictive regressions) renders the "naive
  normal intervals" caveat on the figure** that `summary()` already printed,
  so the forest plot can no longer circulate without the warning.
- **`docs/quickstart.md` no longer ships literal harness markup**, and the
  stale test-count claims in `README.md`/`testing.md` were re-measured and
  corrected (README now uses resilient phrasing).

### Fixed — audit round 6 (run against this release's own merged tree)

- **`seasonal_strength` refuses a constant series** instead of returning a
  float-noise variance ratio that measured ≈ 0.61–0.67 on flat lines —
  coincidentally straddling the 0.64 `nsdiffs` threshold, so a zero-variance
  series read as "seasonality dominates". The refusal matches the
  constant-series behaviour of every sibling diagnostic; `nsdiffs` and
  `check_series` were already guarded.
- **`bvar_ssvs` no longer blames missing values for its own internal
  overflow.** On all-finite data of extreme magnitude (observed from
  max|y| ≈ 6e11 on explosive series) the Gibbs sampler's overflow escaped
  through the shared linear-algebra guard, whose message says "drop or impute"
  values that do not exist. The error now names the real cause and the real
  remedy (rescale/standardize; `bvar_fit`'s closed form may tolerate
  magnitudes the sampler cannot).
- **`har_rv`'s docstring halved its own breaking change** — it said the
  correction moves `bse` by +0.17% at the fixture where the true factor
  `sqrt(577/573)` is +0.35% (the CHANGELOG entry below was already correct).
- **`zivot_andrews`'s documented `trim` range is now `(0, 1/3]`** — `trim=0`
  was structurally unreachable (the candidate window must hold `lags + 1`
  observations), so the documented lower endpoint could never run.
- **`proxy_ar_sets`' `kind` enumeration is aligned across surfaces** (the
  docstring and stub omitted `"ray_below"`/`"ray_above"`), and its
  **long-horizon coverage is now disclosed**: the propagated sets' measured
  coverage keeps declining past the card's h≤8 table — 0.876–0.894 at the
  default `horizon=12` on the card's own DGP, 0.80–0.85 on a routine VAR(1)
  at T=250, one-sided (the truth sits above the set), fading in T. The card,
  docstring, and failure modes now carry the numbers, mirroring the
  disclosure `proxy_svar_bands` already made.
- **The seasonal-strength saturation hazard is documented**: below ~4 full
  cycles the STL cycle-subseries interpolates noise into the seasonal and
  `nsdiffs` flags D=1 on 100% of white-noise series at n ≤ 28 (period 12),
  38% at four cycles — identical in R's `forecast::nsdiffs`, so a disclosure,
  not a divergence.
- Full round record — including the Bayesian-calibration sweep that found the
  conjugate `bvar_fit` core machine-exact against an independent oracle and
  the checker pass that verified no test was weakened across the release's
  ten merged branches — in `docs/roadmap/20-audit-round-6-findings.md`.

### Changed — BREAKING

- **`bvar_hierarchical` now defaults `hyperprior="glp"`.** Audit round 6 drew
  data from the model's own prior and measured the old pure-ML-II default
  collapsing `lambda1_opt` to the search-box floor on 15–24% of datasets —
  and 90% credible IRF bands refit at a collapsed selection covered **5.7%**.
  The marginal-likelihood profile genuinely peaks at λ→0 there (classic
  empirical-Bayes variance-component collapse), so the card's old "check the
  profile" advice would reassure exactly when it should alarm. The GLP Gamma
  hyperprior (mode 0.2, sd 0.4) — the guard GLP (2015) themselves recommend —
  eliminated the floor collapse entirely in the same experiment (verified
  post-fix on the finder's own seed stream: 0.000 vs 0.220 below 1e-3).
  `hyperprior="none"` remains the pure ML-II escape hatch; selected lambdas
  and posteriors move wherever the two objectives disagree. The card now also
  states the measured plug-in calibration of the GLP route itself (90% bands
  cover 0.82–0.85; selection uncertainty is not propagated) and the mild
  long-horizon conservatism of the AR(4) scale rule.
- **`lp(cumulative="both")` and `lp_state(cumulative="both")` no longer pair
  the cumulative-shock regressor with lag-augmented standard errors.** The
  regressor is `Σ_{j=0..h} shock_{t+j}`, so nearby base times share *future*
  shocks that past-lag augmentation cannot project out; the audit measured
  nominal-95% coverage of **0.472–0.550 at h=12, flat in T**, with the
  shortfall matching the omitted overlapping-score autocovariance in closed
  form — and every `band=` route reused that `se`, making a sup-t band
  narrower than an honest pointwise interval. The default `se` for this mode
  is now **HAC** with `maxlags >= h` (measured coverage 0.882 at T=400 rising
  to 0.920 at T=1600), the result stamps `se_method`, and an explicit
  `se="lag_augmented"` with `cumulative="both"` **raises** — in Rust and in
  Python — with the closed-form reason and the escape hatch named. Defaults
  everywhere else are unchanged.
- **`smooth_lp`'s default CV λ grid is now scale-relative** — the old fixed
  `1e-2…1e6` ladder was compared against `X'X`, which carries data units
  squared, so rescaling the shock ×100 pinned `lambda_used` at the grid
  maximum and changed the unit-normalized IRF by up to 7.6×. The default grid
  is the same 17-point half-decade ladder anchored to the penalized spline
  block's mean diagonal (`tr(S'S)/k`) — which transforms exactly as the CV
  optimum does — making λ selection invariant across eight decades of shock
  or outcome scale (verified: λ/c² constant to 1e-9, unit IRF drift ≤ 2e-11).
  Explicitly supplied grids remain absolute and are documented as such.
- **`bvar_ssvs`'s scale-carrying hyperpriors are now semi-automatic.**
  `gamma_b` was an absolute Gamma rate on an error *precision* (units 1/y²)
  and `kappa0`/`kappa1` absolute spike/slab sds on covariance off-diagonals
  (units 1/y) — so changing the units of `y` (percent → decimal) moved
  posterior inclusion probabilities by up to 0.543 and flipped 8 selection
  decisions on the audit's probe, while a Rust doc claimed scale-invariance.
  The defaults now resolve per equation against the unrestricted-OLS residual
  variance (`rate = 0.01·s2_j`; sds `0.1/√s2_i`, `10/√s2_i`), exactly
  reproducing the old semantics at unit-variance data and making the posterior
  equivariant across eight decades (verified to MC-noise zero: max
  |Δinclusion| 0.0000, 0 flips). Explicit floats remain the absolute escape
  hatch and reproduce the old build bit-for-bit.
- **`har_rv` now defaults `use_correction=True`**, so the whole library takes
  the finite-sample `n/(n−k)` HAC scaling by default. It was the one HAC
  surface defaulting `False`, and the docs told the story backwards —
  `expectations.md` said tsecon's `True` default *matched* statsmodels when
  statsmodels `cov_type="HAC"` defaults `False` (its `cov_hac_simple` helper,
  inconsistently, defaults `True`). `bse` and `tvalues` move by exactly
  `sqrt(n/(n−k))` — +0.35% at the golden fixture's n=577, k=4;
  `use_correction=False` restores the old numbers and is the bit-for-bit
  match for a *default* statsmodels `cov_type="HAC"` call. Every docstring,
  card, cookbook, and migration-guide claim now states which way each library
  defaults.
- **`cv_splits(purge=…, embargo=…)` can no longer be silently ignored.**
  On `expanding`/`rolling`, `purge` now acts — the last `purge` rows are
  dropped from the end of each training window (the López de Prado purge
  adapted to walk-forward; test geometry unchanged) — and nonzero `embargo`
  **raises**: a walk-forward training window ends before its own test block by
  construction, so an embargo has nothing to remove, and implementing it would
  be a permanent no-op wearing an argument's name. The guide's claim that
  these schemes "handle leakage automatically" was false for h-step labels
  and is rewritten.

- **`long_memory_d`'s `se` changed meaning, and the number it reports moves.**
  It used to be the large-*m* asymptotic closed form — `pi/sqrt(24m)` for GPH,
  `1/(2*sqrt(m))` for local Whittle — a **constant that ignored the data**
  entirely: two different series with the same bandwidth got the same standard
  error. The data-dependent one was computed and then discarded. `se` is now
  the standard error at the bandwidth actually used, and the closed form is
  reported alongside it as `se_asymptotic` (plus `se_regression` for GPH).

  This matters because the old value was too **narrow**. Measured at `n=512`,
  `m=22`: `se` = 0.17037 against `se_asymptotic` = 0.13672, so intervals built
  from the old number were about **25% too tight at the library's own default
  bandwidth** — and the model card claimed the opposite, telling readers the
  bands were conservatively wide. Verified against exact ARFIMA(0,d,0) draws
  (Davies-Harte circulant embedding, 3000 replications): the new `se` tracks
  the realised sampling dispersion at a ratio of 0.99–1.04 with 94.5–95.7%
  coverage, where `se_asymptotic` ran 0.76–0.88 and covered 87–92%.

  If you were reading `se`, your intervals were too narrow and will now widen.
  If you need the old value for continuity, it is `se_asymptotic`.

- **`var_irf_bands(bias_correct=True)` now raises on `method="asymptotic"`**
  instead of silently discarding the flag. Kilian's correction re-centres
  bootstrap draws and the delta-method arm draws none, so the flag never had
  anything to act on — and this library's own coverage audit *instructed*
  callers to set it, without saying `method="bootstrap"` was required. Both
  instructions now name the method, and the flag is echoed in the result.

- **`historical_decomposition` rejects the five sampling arguments the
  `identification="cholesky"` path never reads** (`n_draws`, `max_tries`,
  `seed`, `lambda1`, `n_weight_draws`), naming the ones that were set. An
  accepted-and-inert `seed` makes a single point decomposition look like a
  seeded draw from a set-identified posterior.

- **`zero_sign_svar(weighted=True)` raises on a zero at horizon >= 1.** The ARW
  importance weight was hardcoded to 1.0 on every path, so the flag could never
  change the output, while the docstring implied non-impact patterns received a
  non-unit weight. The volume element for those patterns is not implemented;
  the refusal is now loud rather than a silent downgrade. New `arw_weighted`
  key says whether the weighting actually applied.

- **`panel_unit_root(test="llc", lrv_kernel="truncated")` raises** instead of
  returning `statistic=nan, p_value=nan`. The truncated kernel is not positive
  semi-definite, so the long-run variance can come out negative and `sqrt`
  mints a NaN — it did on 57 of 60 test panels. The error names the kernel and
  the PSD alternatives.

---

`proxy_svar` identified an impulse response and told you nothing about how
precisely. This release attaches uncertainty to it — twice, because one answer
is not enough. When the instrument is strong you want a band; when it is weak a
band is the wrong object no matter how you build it, and you want a set.

### Added — proxy-SVAR inference

- **`proxy_svar_bands`** — Jentsch-Lunsford moving-block bootstrap confidence
  bands for the external-instrument SVAR impulse response. The joint pair
  `(u_t, m_t)` is resampled under **one** set of block starts, and inside every
  draw the VAR is reconstructed, re-estimated, re-identified, and the unit-effect
  normalization is re-imposed. Returns the Hall/basic band (`lower`/`upper`,
  recommended) alongside the Efron percentile band (`lower_efron`/`upper_efron`)
  that Mertens-Ravn and Gertler-Karadi report; the two separate when the
  bootstrap distribution is skewed.

  **The `h = 0` cell of `norm_var` is degenerate at `unit` by construction** —
  verified `[1.000000, 1.000000]`. That is not a bug, it is the free proof that
  the normalization is re-imposed inside the loop rather than hoisted out of it.

  **`bands="wild"` reproduces the published Mertens-Ravn / Gertler-Karadi bands
  and is not asymptotically valid.** With the common Rademacher draw those papers
  use, `e_t` hits the residuals and the proxy alike, so `m*_t u*_t' = e_t^2 m_t
  u_t' = m_t u_t'` and the identifying moment is **bit-identical in every draw** —
  verified 200/200, maximum deviation exactly `0.000e+00`. The step that does the
  identifying contributes no variance at all. Measured on a known-truth DGP at
  nominal 0.90, the wild arm covers **0.113** at impact against the moving
  block's 0.860, with a mean width of 0.018 against 0.173. The arm is shipped
  because reproducing a published figure is a legitimate thing to want; it sets
  `asymptotically_valid=False` and carries a `validity_note`, and it is not
  inference.

  Failed draws are **counted by reason, never dropped**: six counters
  (`too_few_proxy_obs`, `zero_proxy_variance`, `near_zero_gamma_norm`,
  `refit_failed`, `identification_failed`, `non_finite`), a total in `n_failed`,
  and a `failure_warning`. Dropping them would trim exactly the
  near-zero-denominator tail and shrink the interval precisely when the
  instrument is weakest. A nonzero `n_failed` means reach for `proxy_ar_sets`
  instead.

  These are strong-instrument asymptotics. The moving-block arm's own shortfall
  at longer horizons (0.78-0.81 for a nominal 0.90) is inherited from the
  reduced-form VAR bootstrap, not introduced by the proxy layer — the Cholesky
  reference lands within 0.07 at every horizon on the same replications — and
  there is no Kilian bias correction on this path.

- **`proxy_ar_sets`** — weak-instrument-robust Anderson-Rubin confidence **sets**
  for the same impulse response, inverted in closed form rather than searched
  over a grid. Under weak identification no bounded set can be honest (Dufour
  1997), so a cell may come back as a bounded `"interval"`, an `"exterior"` set
  (the **complement** of a rejected middle region — two rays), the `"whole"`
  line, `"empty"`, a degenerate `"point"`, or a one-sided ray. The shape is the
  answer. Branch on `kind`: an `"exterior"` set reports `lower`/`upper` as
  ±infinity precisely so it cannot be mistaken for an interval, and
  **`excludes_zero` on an unbounded set does not establish a sign** — both signs
  can be members while zero sits in the rejected middle.

  **Reduced-form (VAR coefficient) uncertainty is propagated by default**,
  because omitting it is not a drift but a collapse. Measured at nominal 0.95 on
  an estimated VAR, `T = 300`, VAR(2), excluding the degenerate `(norm_var, 0)`
  cell, coverage at `h = 0..8` runs `.952 .529 .458 .315 .247 .195 .163 .135
  .119` omitted against `.952 .953 .954 .947 .941 .936 .930 .922 .913`
  propagated. **The propagation is conservative under a weak instrument**: the
  weak arm goes from `.9413` omitted to **`.9908`** propagated, because the extra
  variance turns exterior sets into the whole line. That is the correct direction
  to err, and it is disclosed rather than tuned. The price is width — the paired
  median set-width ratio at `h = 8` is **13.5x**. With
  `reduced_form_uncertainty=False` the returned `level` is `None`: a set
  conditional on the reduced form has no honest 1-alpha label.

### Validation, and what kind it is

Neither function is pinned to an independent package, and neither is described
as if it were. No external package implements Jentsch-Lunsford moving-block
proxy-SVAR bands, so the band golden is a **documented-formula NumPy
transcription** with the block starts pinned in the fixture (the RNG becomes a
shared input, and everything downstream is compared cell for cell); it pins the
arithmetic, and the theory is carried by property tests and seeded Monte-Carlo
coverage. The Anderson-Rubin "golden" is likewise a **co-derived NumPy
transcription by the same author from the same specification** — not a
third-party reference. Its load-bearing validation is instead a **brute-force
grid inversion** that re-tests `AR(lambda) <= c` directly at thousands of
candidate values per cell for every shape the set can take, plus a numerical
Jacobian check on the reduced-form correction. See the
[validation matrix](docs/reference/validation-matrix.md) for the row-by-row
grading.

### Not in this release

Named so they are not read into the above: **simultaneous (joint) bands** — both
new surfaces are **pointwise**, covering each `(horizon, variable)` cell at the
nominal rate and saying nothing about the path as a whole, and every other band
in the library is pointwise too. Also still absent: SARIMA seasonal orders
`(P, D, Q, s)`; Angrist-Pischke, Cragg-Donald, and Kleibergen-Paap statistics;
and bootstrap bands for the other point-identification schemes (`long_run_svar`,
`max_share_svar`, `hetero_svar`, `nongaussian_svar` remain point-only).
### Added — simultaneous (sup-t) bands

- **A band you can read as a statement about a whole path.** Every band this
  library produced was **pointwise**: it covered one horizon, or one
  `(horizon, series)` cell, at the level you asked for and promised nothing
  about the path a reader actually traces with their eye. The
  [interval-coverage audit](docs/examples/interval-coverage.md) measured what
  that costs — a nominal 90% pointwise IRF band contained the whole `h = 0..12` path
  in **72.2%** of samples at T=500, and nominal 95% forecast bands contained
  every horizon of every series at once in **40.9%** at T=100 and still only
  **48.1%** at T=800. The gap is **multiplicity, not sample size**: it does not
  shrink as the data grows, which is what makes it a design gap rather than a
  small-sample caveat. `var_irf_bands`, `var_forecast`, `lp` and `smooth_lp` now
  take a band selector — `band="sup-t"`, `"sidak"` or `"bonferroni"` — that
  changes the multiplier and **nothing else**: same point estimate, same
  standard errors, a larger `c` in `point ± c·se` chosen so the whole declared
  family is covered at once. The sup-t construction is Montiel Olea and
  Plagborg-Møller's. **Nothing that already reads these functions changes
  meaning**: `var_irf_bands` and `var_forecast` keep returning pointwise
  `lower`/`upper` whatever you pass and hand the simultaneous edges back as
  extra `sim_lower`/`sim_upper` keys (with `critical_value`,
  `pointwise_critical_value`, `band_scope`, `n_cells` and `n_cells_used`), and
  the LP family returns no band at all unless `band` is set. `lp` and
  `smooth_lp` take their own `band_alpha`; the sup-t simulation is a pure
  function of `band_seed` and `band_n_sim`.
- **Measured, with both arms scored on the same replications** — joint coverage
  of the whole family, pointwise band against sup-t band:

```text
  object                        nominal  design                                pointwise   sup-t
  var_forecast                      95%  T=100, 12 horizons x 2 series, 6000    41.2+-0.6   90.5+-0.4
  var_irf_bands (asymptotic)        90%  T=500, h=0..12, 3000 reps              70.4+-0.8   84.8+-0.7
  lp (lag-augmented)                90%  T=240, 13 horizons, 400 reps           36.5        81.8
  lp (lag-augmented)                90%  T=720, 13 horizons, 400 reps           42.7        89.5
```

Tripling T moved LP's *pointwise* joint rate from 36.5% to 42.7%. It is not
converging.

- **Neither VAR simultaneous rate reaches nominal — 84.8% against 90%, 90.5%
  against 95% — and the tests say so rather than tuning.** A sup-t band fixes
  multiplicity and **inherits every other defect of the band it widens**,
  because it reuses the same standard errors. The IRF band's own *marginal*
  coverage on that design is 88.7% at h=0 falling to 85.2% at h=12, so what that
  cell needs is a better standard error, not a bigger multiplier; `var_forecast`
  is a plug-in band that ignores coefficient sampling error, so its marginal
  rate is 93.3% rather than 95%. LP is the clean case and it demonstrates the
  mechanism: at T=720, where the marginals sit on nominal, sup-t lands on
  nominal.
- **The cell family is a user-visible choice, not a detail.** "Simultaneous over
  what?" has several defensible answers that give different critical values, and
  a band whose scope is ambiguous is worse than no band. `var_irf_bands` takes
  `band_scope="horizon"` (the default: `K = horizon+1`, one family per
  response-shock pair — the object the audit measured), `"shock"`
  (`K = k(horizon+1)`), or `"all"` (`K = k²(horizon+1)`); `var_forecast` takes
  `"all"` (the default, `K = steps·k`) or `"horizon"` (`K = steps`); `lp` and
  `smooth_lp` band the horizons of the one response. **Every result reports its
  scope and its `K`.**
- **`lp_iv`, `lp_multiplier` and `lp_state` get Šidák and Bonferroni only.**
  `band="sup-t"` is **refused with an error naming the reason** — no
  cross-horizon covariance exists for them, and none was invented. Their bands
  must not be described as sup-t.
- **What the four routes cost.** At `K = 13` and `alpha = 0.10`: pointwise
  1.6449, sup-t 2.20–2.65 depending on the persistence of the path, Šidák
  2.6490, Bonferroni 2.6653. Only sup-t uses the dependence across cells, which
  is why it is the one that moves; the closed forms pay for a worst case that a
  smooth response path does not present.
- **Two shapes that must not be blurred.** On `var_irf_bands(method="bootstrap")`
  the simultaneous band is symmetric `point ± c·se`, while the reported
  percentile band is asymmetric Efron. The simultaneous band is therefore **not**
  guaranteed to lie outside `lower`/`upper` cell by cell — only outside the
  symmetric `point ± z·se`.
- **Nothing you already plotted changed.** The pointwise output is
  bit-identical, verified rather than assumed:
  FNV-1a fingerprints of the raw f64 bit patterns were captured against the
  unmodified tree and re-checked after every edit, and the statsmodels goldens
  still pass.
- **Measured again, from Python, by the audit that asked for this.** The
  interval-coverage harness now carries a simultaneous arm in
  `docs/examples/coverage/irf_bands.py` and
  `docs/examples/coverage/forecast_intervals.py`. Both read the pointwise and
  the simultaneous arm off the **same call** — the pointwise band is
  bit-identical whether or not a band is requested, and the harness asserts
  that — so the comparison is paired and every replication that gains the whole
  path gains it because the multiplier grew, never because the point estimate
  moved. It reproduces the crate numbers: joint coverage 42.0% → 90.5% for
  `var_forecast` (K=24, 2000 reps) and 71.7% → 85.2% for `var_irf_bands` (K=13,
  1000 reps), with the shortfall-to-nominal asserted rather than smoothed. The
  closed forms are worth a look there too: on the IRF cell Šidák and Bonferroni
  land at 91.7% and 92.0% against a nominal 90%, over the line because
  over-widening cancelled a marginal shortfall of a different origin. `lp` and
  `smooth_lp` have no harness arm yet.
- **Implementation.** One owner for the critical value, `tsecon_stats::simultaneous`,
  with four routes (sup-t from bootstrap draws, sup-t from a covariance by
  Gaussian simulation, Šidák, Bonferroni). LP's cross-horizon covariance was
  **built**, not faked — the Frisch-Waugh-Lovell influence representation, with
  `sqrt(diag)` matching the reported standard errors to 1.3e-15 and ragged
  samples handled so the matrix is PSD by construction. `smooth_lp` already
  computed a joint covariance and discarded all but the diagonal, so its band is
  exact and nearly free.

## [0.2.0] - 2026-08-05

An interval-coverage audit took the library's interval-producing surfaces,
pointed each at its own nominal level, and measured what fraction of intervals
actually covered. It found four real defects. This release fixes them. **One
fix is breaking**: it changes numbers callers were already getting, and
changing those numbers is the point of the release.

### Changed — BREAKING
- **`iv_gmm(weight="hac")` was a silent no-op, and its standard errors will
  move.** `bandwidth` defaulted to `0.0`, and a Bartlett kernel truncated at
  zero lags *is* the White estimator — so `weight="hac"` returned results
  bit-identical to `weight="robust"` (max |Δ se| = 0.000e+00 over 3000
  replications) while the caller believed they had bought serial-correlation
  robustness. `bandwidth` now defaults to `None`, which selects the
  Newey-West rule of thumb `floor(4 * (n/100)^(2/9))`; an **explicit**
  `bandwidth=0.0` now raises instead of silently degrading to White; and the
  truncation actually used comes back as `hac_bandwidth`. On an AR(1)-error
  design at n = 250 the slope standard error moves from 0.10522 (`robust`, and
  the old `hac`) to 0.09393 (`hac`, automatic bandwidth 4) or 0.09228 (`hac`,
  `bandwidth=10`). If you passed `weight="hac"` before, you were reporting
  White standard errors under a HAC label and your numbers will change.
  **This does not restore coverage.** The audit measured 0.868 ± 0.006 against
  a nominal 0.95 at `bandwidth=10`, and the automatic rule picks *fewer* lags
  (4 at T = 250). A defensible default is not a remedy: those intervals are
  still too narrow, and the fix is that you can now see and set the bandwidth
  rather than that the bandwidth is now right.
- **`iv_gmm(method="2sls", weight="hac")` now raises.** 2SLS fixes its weight
  matrix at `(Z'Z/n)^-1` by construction, so accepting a weight argument there
  was the same silent no-op in a second place.

### Added — inference
- **`ols(se_type="hc2")` and `ols(se_type="hc3")`** — the leverage-corrected
  heteroskedasticity-robust standard errors, matched to statsmodels HC2/HC3 to
  2.96e-15 on the audit's own T = 25 chi2(1)-regressor high-leverage design. On
  that design tsecon's own slope standard error runs 0.1749746 (nonrobust),
  0.1751173 (hc0), 0.1825724 (hc1), 0.2095910 (hc2), 0.2629148 (hc3): HC3 is
  44% larger than HC1, which is exactly the leverage correction that hc1's
  `n/(n-k)` factor does not buy. An observation whose leverage is numerically
  equal to 1 is refused, not returned as a near-infinite standard error.
- **`iv_gmm` returns `first_stage`** — a list of dicts with keys `regressor`,
  `fstat`, `dof_num`, `dof_den`, `pval`: a heteroskedasticity-robust
  per-regressor first-stage F. Entries are **omitted** where the statistic is
  undefined (an exogenous regressor, no excluded instruments, a regressor the
  instruments reproduce exactly, rank-deficient `Z`, a non-finite statistic),
  so the list can be shorter than the regressor count and must be indexed by
  `regressor`, never by position; a missing entry is not a failed fit. **With
  two or more endogenous regressors this is not a weak-identification test** —
  every regressor can clear 10 while the system is under-identified, because
  the instruments may predict only one common combination of them. The right
  objects are Angrist-Pischke (per regressor) and Cragg-Donald /
  Kleibergen-Paap against Stock-Yogo (joint), and none of those are
  implemented. Even with a single endogenous regressor, F > 10 is not a safety
  threshold: the audit measured 0.915 coverage at a median first-stage F of
  10.5.
- **`arima_fit(drift_uncertainty=True)`** — with `d >= 1` and `constant=True`
  the h-step forecast contains an estimated drift whose uncertainty grows like
  h², and the default omits it entirely (the reported se is exactly
  `sigma*sqrt(h)`). Opt in and the se matches the closed form
  `sigma*sqrt(h + h²/(T-1))` to 5.22e-09. It is **opt-in** so the default path
  stays bit-identical and keeps matching the statsmodels `get_forecast` golden
  at 1e-6: the two are different estimands, not a right one and a wrong one.
- **`arima_fit` returns `bse`, `param_cov`, and `cov_ok`** — parameter standard
  errors from the observed information (statsmodels `cov_type="approx"`).
  ARIMA previously reported no parameter standard errors at all. `bse` and
  `param_cov` are `None` with `cov_ok=False` when that matrix is too
  ill-conditioned to invert honestly, rather than reporting a number the
  numerics do not support.

### Added — ergonomics
- **Forgiving input**: every estimator now accepts a pandas `DataFrame`/`Series`
  (or any `.to_numpy` array-like), off-dtype/non-contiguous float arrays,
  **integer and boolean arrays** (data read as `int`, a 0/1 dummy, a `y > 0`
  mask), and a **plain list of numbers**. All are converted to `float64` at the
  boundary instead of raising. Coercion is *parameter-aware*, so it never
  touches an argument that is not data: the four audited integer label/index
  parameters (`hetero_svar.regime_labels`, `var_granger.caused`/`causing`,
  `favar.slow_indices`), restriction-tuple specs, tuple-valued options, and
  callables pass through untouched, and ragged panel lists keep their container
  while each per-unit array is converted. A *nested* Python list is
  deliberately left alone — `[(0, 1), (0, 2)]` is a restriction spec and
  `[[1.0, 2.0], [3.0, 4.0]]` is a matrix, and the values are indistinguishable.
- **Errors that teach**: the messages a first run is most likely to hit now say
  what happened, why, and what to try, with the offending numbers included. For
  example, too few observations for the requested lags now reports the required
  row count and the degrees-of-freedom arithmetic behind it, and suggests a
  concrete smaller `lags`. A wrong-rank array argument reports the shapes it
  received instead of the low-level `'ndarray' object is not an instance of
  'ndarray'`.
- **`tsecon.summarize(result)`**: a uniform, opt-in renderer for *any* function's
  output. Plain dicts get a generic aligned `.summary()`; the six bespoke
  `tsecon.results` objects pass through unchanged. Still a `dict` subclass, so
  the plain-data contract is preserved.
- **Cookbook**: short single-task recipes under `docs/cookbook/`, each a
  self-contained page with executed output.

### Added — estimators
- `ndiffs` — how many differences a series needs, with the per-order test
  evidence rather than just the integer.
- `box_cox_lambda` — variance-stabilising Box-Cox lambda (MLE, matched to
  `scipy.stats.boxcox_normmax`; Guerrero as a documented-formula alternative).
- `engle_granger` — the two-step cointegration test now returns p-values and
  critical values from the MacKinnon response surfaces, not just the statistic.

### Fixed

The four defects the interval-coverage audit found, with measured coverage
before and after (nominal 0.95 throughout):

- **`ols` robust intervals under-covered badly under high leverage.** On the
  T = 25 chi2(1) design: **hc1 0.682 → hc3 0.863**. Fixed by adding hc2/hc3.
  hc1 is still available and is still the wrong choice on that design; hc3 does
  not reach nominal there either, and the honest reading is that a T = 25
  tail-heavy design is hard, not that hc3 solves it.
- **`arima_fit` forecast intervals ignored drift uncertainty.** With `d >= 1`
  and `constant=True`, at h = 24 and T = 60: **0.902 → 0.945** with
  `drift_uncertainty=True`.
- **`iv_gmm(weight="hac")` never applied a HAC correction** (bit-identical to
  `weight="robust"`, max |Δ se| = 0.000e+00 over 3000 replications). See the
  breaking note above. **Not fixed by this release:** coverage on the audited
  AR(1)-error design is 0.868 at `bandwidth=10`, and the new automatic default
  picks fewer lags than that.
- **`iv_gmm` reported no first-stage evidence at all.** Now returns
  `first_stage`, with the caveats above — this is a diagnostic, not a
  weak-identification test.

### Not in this release

Named so they are not read into the above: SARIMA seasonal orders `(P, D, Q, s)`;
Anderson-Rubin and other weak-IV-robust confidence sets for `iv_gmm`;
Angrist-Pischke, Cragg-Donald, and Kleibergen-Paap statistics; and simultaneous
(joint) bands anywhere — every band in the library is pointwise.

## [0.1.0] - 2026-07-23

First tagged release, published to PyPI as `tsecon`. Pre-1.0 and under active
development: minor versions may make breaking changes, patch versions are
fixes, until 1.0.

### Added — foundations and first model classes
- **Foundations**: Philox RNG (bit-identical to NumPy), special functions and
  the distribution zoo, structured linear algebra (Levinson-Durbin, Toeplitz,
  discrete Lyapunov), the resampling/bootstrap engine, the exact-diffuse
  linear-Gaussian state-space (Kalman) engine, the numerical optimizer suite
  with the Monahan stationarity transform, and the HAC/robust-inference module.
- **Diagnostics**: ACF/PACF, Ljung-Box, Jarque-Bera, ARCH-LM; the full
  unit-root workflow (ADF with MacKinnon p-values, KPSS, `check_stationarity`);
  the semiparametric Phillips family — `phillips_perron` (Z-tau/Z-alpha
  unit-root test) and `phillips_ouliaris` (single-equation residual
  cointegration test), matched to `arch` to < 1e-10 with MacKinnon
  response-surface p-values; spectral analysis (periodogram, Welch, coherence).
- **One-call battery**: `check_series` — the Module 01 flagship — runs the
  diagnostic families in order (outlier screen, ADF+KPSS quadrant with
  analysis-scale routing, Ljung-Box/ACF/PACF, ARCH-LM, Jarque-Bera, a
  sup-F/Bai-Perron mean-shift scan, GPH long memory, seasonality evidence;
  for a 2D panel: per-series integration, Johansen, and VAR lag selection
  with a stability check) and ends in recommendations that route to concrete
  tsecon calls — every hypothesis test on the record in `tests_run` with the
  multiple-testing arithmetic shown, never silently corrected.
  `tsecon.results.check_series` adds `.summary()` and `.plot_diagnostics()`.
- **Univariate models**: exact-MLE ARIMA; GARCH/GJR/EGARCH with normal and
  Student-t QMLE, Bollerslev-Wooldridge robust standard errors, and a fused
  allocation-free likelihood with analytic gradients; GAS/DCS score-driven
  volatility (Gaussian and Student-t); Markov-switching AR; trend-cycle
  filters (HP, one-sided HP, Baxter-King, Christiano-Fitzgerald, Hamilton);
  long memory (fractional differencing/integration, GPH, local Whittle).
- **Multivariate and structural**: reduced-form VAR with IRF/FEVD/Granger/
  forecasting, frequentist IRF confidence bands (`var_irf_bands` — Lütkepohl
  (1990) delta-method SEs validated against statsmodels to machine precision,
  and a Kilian (1998) residual bootstrap with optional bias correction), and an
  honest stability block (`is_stable`/`min_root`); sign-restricted Bayesian
  SVARs; `zero_sign_svar` — the corrected Rubio-Ramírez-Waggoner-Zha (2010) /
  Arias-Rubio-Ramírez-Waggoner (2018) **zero + sign** restricted SVAR (a
  superset of the sign-only sampler that reproduces the recursive Cholesky IRF
  as its degenerate impact-only-zero corner, with the weight-invariant
  identified-set envelope as the prior-robust deliverable); and four closed-form
  point-identification schemes —
  `long_run_svar` (Blanchard-Quah long-run restrictions), `max_share_svar`
  (Uhlig/Francis maximum-FEV-share and Barsky-Sims news shocks), `proxy_svar`
  (external-instrument SVAR-IV with a first-stage-F report and NaN-window
  handling), and `hetero_svar` (Rigobon two-regime identification through
  heteroskedasticity, with a Box's-M covariance-equality gate); FAVAR;
  Diebold-Yilmaz connectedness; the PCA factor model with Bai-Ng selection;
  Johansen cointegration and VECM; multivariate GARCH (CCC/DCC).
- **Statistical (non-Gaussian) SVAR identification**: `nongaussian_svar` —
  independent-component / ICA identification (Lanne-Meitz-Saikkonen 2017;
  Gouriéroux-Monfort-Renne 2017) that point-identifies the *whole* structural
  impact matrix from the reduced-form residuals alone — no sign, zero, long-run,
  proxy, or variance-regime restriction — by exploiting the mutual independence
  and non-Gaussianity of the shocks. A deterministic symmetric FastICA fixed
  point (Hyvärinen log-cosh contrast, identity-initialized — bit-reproducible, no
  RNG) rotates the whitened residuals to maximal non-Gaussianity and returns
  `B = Σ_u^{1/2} Q`; column sign and order are conventions, and the scheme
  **fails under Gaussianity** — a boundary the returned `shock_kurtosis`
  diagnostic flags (near zero ⇒ weakly identified). Validated by an independent
  NumPy FastICA golden (itself cross-checked to `sklearn.decomposition.FastICA`)
  for the core, plus seeded Monte-Carlo recovery of the true `B` up to sign and
  permutation.
- **Post-identification and prior-robust SVAR tools**: a layer that *takes* an
  identification (any impact matrix `A0`, or a sign-restricted set) and answers
  what comes after — `structural_fevd` (forecast-error variance decomposition
  for an arbitrary structural `A0`, the gap the recursive-only `var_fevd`
  leaves; reproduces `var_fevd`/statsmodels exactly for the Cholesky case,
  rows sum to 1 by the rotation-invariant-denominator identity);
  `historical_decomposition` (per-`(time, variable, shock)` contributions with
  the exact `y = baseline + Σ_j hd` adding-up identity, in a Cholesky point mode
  and an importance-weighted sign-identified set mode); `fry_pagan_svar`
  (Fry-Pagan 2011 median-target — the single accepted, coherent draw closest to
  the pointwise-median band, the answer to "medians mix models");
  `robust_svar_bounds` (Giacomini-Kitagawa 2021 prior-robust identified-set
  bounds via the Gafarov-Meier-Montiel-Olea 2018 active-set closed form, exact
  for a single restricted shock and a conservative marginal outer bound for
  jointly-restricted shocks — removing the Haar-prior artifact that pointwise
  sign-restricted bands carry); and `narrative_svar` (Antolín-Díaz-Rubio-Ramírez
  2018 narrative sign restrictions — shock-sign and "most/least important
  contributor" episode statements imposed by importance-reweighting with weight
  `1/P̂(N|S)`, reporting `ess`/`min_ptilde`; a strict superset of
  `sign_restricted_svar` that reproduces it bit-for-bit with no narrative
  restrictions).
- **Local projections**: `lp` (lag-augmented inference by default), `lp_iv`
  with a per-horizon first-stage F, state-dependent `lp_state`, a three-valued
  `cumulative` mode, and `lp_multiplier` — the one-step Ramey-Zubairy integral
  multiplier as a first-class entry point (because outcome-only cumulation is
  a cumulative IRF, not a multiplier).
- **Bayesian**: a Minnesota-NIW Bayesian VAR with closed-form posterior,
  posterior impulse-response draws, and ArviZ-exact convergence diagnostics;
  `bvar_hierarchical` — empirical-Bayes (ML-II / GLP MAP-II) selection of
  the prior tightness by maximizing the Giannone-Lenza-Primiceri (2015)
  marginal likelihood, then refitting the posterior at the optimum; and
  `bvar_ssvs` — the George-Sun-Ni (2008) spike-and-slab **stochastic-search
  variable selection** BVAR (a four-block Gibbs sampler returning per-coefficient
  posterior inclusion probabilities, optional error-precision selection, and
  Cholesky-orthogonalized IRF draws), MC-recovery-validated on a sparse VAR.
- **Forecasting and evaluation**: Diebold-Mariano (HLN), Clark-West,
  Giacomini-White, Theta, accuracy measures, and the rolling/expanding
  backtest engine.
- **GMM**: linear IV-GMM (2SLS/two-step/iterated, Hansen J) and nonlinear GMM
  with Python-callback moment conditions.
- **Predictive regressions**: OLS/Stambaugh/IVX in one call plus the joint
  IVX test — Monte-Carlo-validated to hold size through an exact unit root.
- **Panels**: fixed effects with clustered/Driscoll-Kraay SEs, panel LP,
  mean-group VAR, the heterogeneous-panel trio (mean group, CCE-MG, PMG), and
  `panel_unit_root` — the three first-generation panel unit-root tests
  (Levin-Lin-Chu, Im-Pesaran-Shin, Fisher/Maddala-Wu-Choi) of the joint
  unit-root null, validated to R `plm::purtest` (and, for Fisher, statsmodels).
- **Nowcasting and mixed frequencies**: MIDAS (weights/U-MIDAS/weighted),
  DFM nowcasting (two-step and exact one-step MLE) with a ragged edge and the
  Bańbura-Modugno news decomposition.
- **Term structure**: Nelson-Siegel, Svensson, dynamic Nelson-Siegel, and the
  arbitrage-free (AFNS) yield adjustment of Christensen-Diebold-Rudebusch.
- **Applied-macro extensions**: recession-probability models (static and
  Kauppi-Saikkonen dynamic probit/logit); survey-expectations tools
  (Coibion-Gorodnichenko, Mincer-Zarnowitz, disagreement); the specification
  and stability battery (White, Koenker-Breusch-Pagan, RESET, Chow, CUSUM);
  and a linear rational-expectations (DSGE-lite) solver via Blanchard-Kahn.
- **Python layer**: maturin mixed layout (`tsecon._core` + a pure-Python
  package); the opt-in `tsecon.results` rendering layer — `dict` subclasses
  with `.summary()`/`.plot_*()` that preserve the plain-dict contract;
  complete type stubs with `py.typed`.
- **Configurable inference**: a uniform `se_type=` on regression estimators;
  configurable interval coverage; cumulative IRF views.
- **Evidence beyond fixtures**: a seeded Monte Carlo validation suite (size /
  coverage / consistency) and frontier experiments (LP vs VAR; weak-IV LP-IV);
  a 25-case cross-library parity benchmark harness (statsmodels, SciPy,
  scikit-learn, `arch`) that gates CI; two replications of published results —
  Ramey-Zubairy (2018) government-spending multipliers and Estrella-Mishkin
  (1998) yield-curve recession prediction — running offline from committed
  public data; Rust and Python coverage tooling.
- **Docs**: a 15-chapter teaching guide, model cards for every estimator
  family, a generated API reference with a drift guard, a validation matrix,
  a testing-and-validation map, migration guides (statsmodels/R/Stata) with a
  Rosetta glossary, a worked figure gallery, and an interactive demo.
- **Packaging**: complete `pyproject.toml` metadata, abi3-py39 wheels tested
  on Python 3.9 and 3.13 in CI, GitHub Actions CI (Rust gates, a three-OS
  wheel matrix, a mypy stub check, and an evidence job running the Monte Carlo
  suites and the parity gate), and a tag-triggered release pipeline with PyPI
  trusted publishing.

### Removed
- **Data-fetching loaders** (`tsecon.datasets`): built, then deliberately
  removed before release. A library that hardcodes external data URLs owns
  their breakage (the widely-cited FRED-MD URL had already moved), so tsecon
  ships no network code — the only runtime dependency is NumPy, and the
  replications run on small public datasets committed to the repository.

Every estimator is validated against a reference implementation (statsmodels,
SciPy, NumPy, `arch`, `linearmodels`, scikit-learn, ArviZ) or a documented
closed form in the test suite.
