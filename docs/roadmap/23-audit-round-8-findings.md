# Adversarial audit, round 8 — findings

> **Working document.** Continuation of
> [round 7](22-audit-round-7-findings.md), run under
> [the brief](16-adversarial-audit-brief.md). Excluded from the published
> site.

Round 8 ran against the post-0.4.0 tree with three jobs, in priority order:
adversarial first contact with the **0.4.0 additions** (`proxy_first_stage`,
`acm_term_premium`, the copula family `copula_fit`/`copula_select`/
`pseudo_obs`, `lp_did`, `scale_ar`, and the round-7 garch boundary/`se_valid`
machinery); an edge attack on the **two post-freeze fixes**
(`tsecon_stats::special::ln_gamma_half_ratio`'s seam, and the
`NU_GAUSSIAN_RIDGE` deterministic `converged` flag in both `tsecon-gas`
models); and **fresh eyes on three older estimators no round had touched**
(`theta_forecast`, `hamilton_filter`, `bns_jump_test` — none of them appears
in any round-2–7 findings doc). Finder/refuter discipline throughout: every
candidate was reproduced by a minimal probe committed under
`scratch/round8/` and then attacked with an alternative explanation /
reference re-derivation before being reported; the refuted ones are listed
with the evidence that killed them.

**378/378 probe comparisons attempted were reached** (per-probe counts
below), plus the seam-accuracy grid, three Monte-Carlo designs, and four
refutation re-runs. **Raised: 8 candidates. Survived refutation: 3 — all
documentation defects, all fixed this round with regression tests; 5
refuted. Zero code defects found.** The rebuilt module is bit-identical to the released
0.4.0 build on every touched surface (`scratch/round8/p12_no_behavior_drift.py`,
old-vs-new JSON diff empty).

---

# Confirmed, and fixed this round

## C1 — `theta_forecast`'s "Matches statsmodels ThetaModel" only holds at non-default statsmodels settings

**`overclaim`** (lens 5; the brief's three-surface-disagreement rule paid
off exactly as written). The runtime `__doc__` said, in full: *"Matches
statsmodels ThetaModel."* The forecasting card said *"`statsmodels`
`ThetaModel` (deseasonalize=True), matched numerically"*. Only the Rust
module doc — invisible from Python — carried the real qualifier:
`use_test=False`. statsmodels' **default** (`use_test=True`) runs a
seasonality pre-test and skips deseasonalization when it fails; tsecon
always deseasonalizes when `period > 1`.

**Observed** (`p9_theta.py`, `p9b_theta_usetest.py`): on iid data declared
with `period=12`, tsecon diverges from a *default* statsmodels
`ThetaModel` fit in **29/30 seeded draws, worst 2.6% relative**; against
`use_test=False` the same draws agree (worst 1.9e-5, and see refuted R2).
On genuinely seasonal data and on the realgdp golden configuration the
match is 1.9e-8 / 5.5e-7 — the pre-test then chooses deseasonalization
too, which is why no golden ever saw this.

**Fix:** the docstring, `.pyi`, and card now state the
`ThetaModel(deseasonalize=True, use_test=False)` equivalence and the
default-vs-default divergence mechanism, with the measured numbers on the
card. Regression test:
`test_docstring_keys.py::test_theta_forecast_docstring_qualifies_the_statsmodels_match`
(the bare unqualified sentence is asserted absent). Behavior unchanged.

## C2 — The copula workflow promised invariance under "any strictly monotone transform"; a decreasing transform flips the fit

**`overclaim`** (lens 5, caught by lens-2-style probing). Three
Python-facing surfaces (`pseudo_obs.__doc__`, `copula_fit.__doc__`, the
copulas card) plus the crate docs claimed the pseudo-observations — and
any copula fitted to them — are invariant to *"any strictly monotone
transform"* of a margin, "bit-identical (property-tested)". The property
test (`copula_properties.rs`) uses only **increasing** transforms
(`exp`, cube), and the mathematics is increasing-only: **observed**
(`p2_copula.py`), negating one margin reverses its ranks
(`u -> 1 - u` exactly, absent ties) and the fitted Gaussian `rho` flips
from +0.5697 to −0.5697. The textbook invariance (McNeil-Frey-Embrechts
Prop. 7.7) is stated for strictly increasing maps; the doc's examples
(logs, standardization, exp) were all increasing, but the claim as
written was false.

**Fix:** all doc surfaces now say strictly *increasing* and state what a
decreasing transform does instead. Regression tests assert the true
behavior and the corrected wording: Rust
`decreasing_margin_reverses_ranks_and_flips_dependence` (new; the old
property test renamed `increasing_margin_invariance_is_exact`, asserts
unchanged) and Python
`test_copula.py::test_pseudo_obs_decreasing_transform_flips_dependence`.

## C3 — `acm_term_premium` returns three keys no doc surface names

**`cosmetic`** (the round-6 `bvar_ssvs` echo-key residue class, applied
consistently). The result dict carries echoed inputs `maturities`,
`n_factors`, `periods_per_year`; neither the docstring's returns
enumeration nor the term-structure card's key list mentioned them.
**Fix:** both now do (and the `.pyi`), with a returned-keys ⊆ doc-words
tripwire test (`test_acm.py::test_acm_docstring_names_every_returned_key`)
so the next added key fails a test instead of shipping undocumented.

# Refuted (kept out, with the evidence)

- **R1 — `copula_fit(family="frank")` "misses the MLE"** (`p2_copula.py`
  showed my scipy reference at θ=−7.08 beating tsecon's θ=4.26).
  Probe artifact: statsmodels' Frank log-density returns NaN over most of
  θ<0 and my unguarded `minimize_scalar` chased the NaNs. With a
  NaN-guarded objective (`p2b_frank_refute.py`) the reference optimum is
  θ=4.2627158, log-lik 113.86571140221267 — tsecon matches to 6.8e-8 in θ
  and **7.1e-14 in log-likelihood**, and the θ<0 side is genuinely worse
  everywhere it is defined.
- **R2 — residual `theta_forecast` mismatches against
  `use_test=False`** (6/30 iid draws >1e-6, worst 1.9e-5). Reference-side
  optimizer slack on a flat SES objective at the α→0 boundary:
  profiling the exact concentrated SSE (`p9c_theta_alpha_refute.py`),
  statsmodels stops at α=1.15e-4 with SSE 1154.682 while the optimum
  (and tsecon's landing, below α=1e-6) has SSE 1154.549 — **tsecon's
  point is the better optimum of the shared criterion**, so the "matched
  numerically" promise is kept where an interior optimum exists (realgdp
  golden: 5.5e-7).
- **R3 — `bns_jump_test` "low power"** (0.40 against what my probe called
  a large jump). Probe miscalibration, settled by the brief's
  predict-in-closed-form technique: the injected jump was 8.5% of the
  day's quadratic variation, predicted z ≈ 2.14, and 0.40 is the correct
  power at that alternative; at relative jumps 0.27/0.51 (predicted
  z ≈ 6.8/12.8) measured power is **1.000/1.000**
  (`p11b_bns_power_refute.py`). Null size is 0.0557 at nominal 0.05
  (3000 reps, n=390).
- **R4 — `dcs_local_level(density="t")` certifying `converged=True` on
  clean Gaussian data** (7/20 seeded series, ν̂ between 12.9 and 108,
  `p6_gas_ridge.py`). Not a flag defect: those are genuine finite-sample
  interior optima (a Gaussian sample whose empirical tails come out
  slightly heavy has an interior ν MLE), the optimizer's certificate is
  about the sample's likelihood, and the docstring's boundary sentence
  describes the ν→∞ *tendency*, not an always-False promise. The
  deterministic ridge flag is aimed at the boundary rides, and every one
  of those (ν̂ from 4e12 up to 1e307) reported `converged=False`.
- **R5 — `acm_term_premium` runs silently on percent-unit input.**
  Documented trap, not a finding: the docstring's "UNITS ARE
  LOAD-BEARING" paragraph states exactly this hazard and no surface
  promises a runtime unit check. Recorded so the next round does not
  re-derive it.

# The post-freeze fixes, attacked at their edges — held

- **`ln_gamma_half_ratio`'s seam** (`p8_lgh_seam.py`, Rust arithmetic
  transcribed to Python doubles against a 60-digit Decimal/Stirling
  reference): literal-branch error just below the seam ≤ **4.6e-13**
  absolute (the doc's "~1e-10" is conservative), series branch above at
  the double rounding floor (≤ 3.2e-16), seam jump ≈ **1.9e-13** (a
  phantom loglik step of ~1e-10 at T=500 — and the only unbounded-ν
  caller crosses the seam at ν=2e3, deep inside flag-forced-False
  territory; GARCH boxes ν at 500 → x=250, the t-copula at 1000 → x=500,
  both safely on the literal branch their goldens pinned), monotone
  across the seam at 0.25 granularity, and at x=5e15 the helper is exact
  to <1e-15 relative where the literal difference is garbage (Python's
  cancels to 0.0 outright). The committed Rust tests assert the doc's
  claims at the stated tolerances.
- **`NU_GAUSSIAN_RIDGE`** (`p6_gas_ridge.py`, 79/79): across 20 DCS + 20
  GAS clean-Gaussian fits and 12 near-threshold (true ν=800) fits,
  **every** ν̂ > 1e3 reported `converged=False` (rides land at 1e9–1e307,
  never in (1e3, 1e7)); every certified `converged=True` landing sits at
  ν̂ ≤ 196 — an empirical no-man's-land around the 1e3 threshold exactly
  as its doc claims ("far above any genuine interior optimum … far below
  any tolerance stop"); genuine heavy-tail optima certify 10/10 (DCS,
  true ν=5) and 10/10 (GAS, true ν=6); Gaussian and Laplace densities
  unaffected (`converged=True` on clean data; `needs_dof()` gates the
  rule and Laplace's `nu` is NaN).

# Swept and found sound

- **`proxy_first_stage`** (35/35, `p1_proxy_first_stage.py`): HC1/
  classical/HAC effective F, β, and SE match a statsmodels OLS
  reconstruction to 1e-9 (HAC with the weakivtest T/(T−2) correction,
  reproduced exactly); all four MOP critical values match
  `scipy.stats.ncx2.ppf(0.95, 1, 1/tau)` to 1e-9 relative and `tau_bound`
  inverts through scipy's CDF to 1e-6; default `hac_lags` is the
  Newey-West rule on the residual sample; `hac_lags` under
  `variance="hc1"` **raises** (the round-3 `hac_lags` lesson, honored);
  every axis alive; F scale-invariant over 16 decades; the length-T proxy
  alias is bit-identical; a NaN-prefix availability window reproduces the
  statsmodels fit on the overlap; the `proxy_svar`-stamped `first_stage`
  dict is bit-identical to the standalone call; 7 degenerate inputs all
  refuse with teaching errors; returned keys == documented set.
- **The copula family** (68/68 + refutations): `pseudo_obs` equals scipy
  `rankdata(method="average")/(n+1)` exactly, ties included; MLE matches
  NaN-guarded scipy optima of the statsmodels log-density for all five
  families (log-lik to ≤1e-10; the t's (ρ,ν) jointly); tau inversion
  matches `fit_corr_param` and the closed maps to 1e-12 (t: ρ pinned by
  τ, ν profiled; NaN SEs with `se_valid=False` as documented); AIC/BIC
  arithmetic exact with the right k per family; all tail-dependence
  closed forms verified including the Demarta-McNeil value 0.25317 at
  (ρ=.5, ν=4) — the number statsmodels' own `dependence_tail` gets wrong,
  as the card records; `copula_select` rankings are consistent with the
  per-fit AIC/BIC, the t wins on t-copula data, and Clayton/Gumbel are
  skipped-with-reason on negative-τ data while `copula_fit` raises the
  same teaching error; 8 degenerate probes (raw data, boundary u, n<20,
  NaN, comonotone pair, constant column, unknown family/method, 3-column
  u) all refuse with teaching messages; the whole workflow is
  bit-identical under increasing margin transforms.
- **`acm_term_premium`** (19/19): `fitted = risk_neutral + term_premium`
  holds **bit-exactly** (max |Δ| = 0.0) and the maturity-1 premium is
  exactly zero; the short rate, decimal-units level, and per-maturity
  `yield_rsquared` behave on a smooth panel; `n_factors` and
  `periods_per_year` axes alive; `rx_maturities` follows the documented
  n−1-in-grid rule on contiguous and sparse grids; 7 degenerate probes
  refuse with teaching errors, including the no-adjacent-pair grid and
  the k-maturities-cannot-identify-k-factors cross-section. The fixture
  generator was read: genuinely non-circular (pure NumPy, no tsecon
  import; the repo-wide `grep -l "import tsecon" fixtures/*.py` tripwire
  returns nothing).
- **`lp_did`** (51/51): coefficients **and** the fixest-convention
  entity-clustered SEs match a from-scratch independent NumPy
  implementation of the clean-control long-difference regression at
  **1e-10** on five horizons (post h=0/3/6 with the D_{i,t+h}=0 control
  rule, pre h=−2/−4 with the D_{it}=0 rule), with matching `nobs` and
  `n_switchers`; h=−1 is exactly zero; a homogeneous-effect staggered DGP
  is recovered at every horizon with clean pre-trends; `reweight`/
  `never_treated_only`/`pooled`/`nonabsorbing_lag` all alive (pooled adds
  exactly the documented keys and leaves the event study untouched);
  coef/se exactly scale-equivariant; 7 degenerate probes (reversal under
  absorbing, non-0/1 treatment, no switchers, all-treated, NaN, overlong
  window, shape mismatch) all refuse with teaching errors.
- **`scale_ar`** (8/8): axis alive on all three surfaces (`bvar_fit`,
  `bvar_irf_draws`, `bvar_hierarchical`), pairwise distinct at 1/2/4,
  default ≡ explicit 4 bitwise; `scale_ar=0`, negative, and ≥T all refuse
  with teaching errors (the negative case through the round-7 central
  coercion message, as designed).
- **The garch boundary machinery** (82/82 edge-invariants on 20 fresh
  fits across 4 DGP families × 5 seeds): `se_valid[i]=False ⇔ NaN` in
  both SE vectors, per parameter, every fit; `boundary` set ⇒ non-empty
  teaching `boundary_note`, interior fits ⇒ `note=None` and all-valid
  SEs; flag vectors aligned with `param_names`; both classes exercised
  (8 flagged / 12 interior); `GARCHResults.summary()` mentions the
  boundary iff flagged.
- **`hamilton_filter`** (13/13): β, trend, and cycle equal a statsmodels
  OLS of y_t on [1, y_{t−h..t−h−p+1}] to 1.2e-12; `first_index = h+p−1`;
  trend+cycle ≡ y; h/p axes alive; scale-equivariant; degenerate probes
  refuse (including the constant series, refused as rank-deficient with
  a teaching message).
- **`bns_jump_test` + the realized trio** (13/13 + MC): the ratio equals
  an independent transcription of the Huang-Tauchen studentized ratio
  statistic to 1e-12, and RV/BV/RQ/TQ each equal their documented closed
  forms exactly (the (π/2) and n·μ_{4/3}^{-3} scalings verified —
  the TQ/BV² studentization is dimensionally consistent); the ratio is
  exactly scale-invariant; null size 0.0557/0.0140 at nominal 0.05/0.01
  (3000 reps); power 1.000 against detectable jumps (see R3); degenerate
  probes refuse.

# Residue (recorded, no action)

- The Rust `ThetaForecast` struct computes `alpha`, `b0`, `one_step`,
  `seasonal`, `multiplicative`; the binding returns the bare forecast
  array. The card documents "a bare array of length `steps`", so this is
  the documented contract, not a dropped-quantity defect (lens 4's source
  read, run and cleared) — but exposing `alpha`/`b0` would be a
  reasonable future enhancement, noted here so the next lens-4 pass does
  not re-flag it.
- `copula_fit` on near-Gaussian data reports the t family's ν̂ at ~424
  with `se_valid=True, converged=True` and a log-likelihood 0.0008 above
  the Gaussian's — consistent with the card's "ν at the barrier" failure
  mode (the fit "matches the Gaussian log-likelihood from above"); the
  copula's ν box is 1000 and no deterministic flag applies there. Same
  no-man's-land argument as R4; nothing dishonest observed.

# Comparisons attempted but not made (stated, not dropped)

- **`scale_ar=4` default vs the 0.3.0 wheel bit-identity** — the
  CHANGELOG's claim rests on an out-of-band wheel-hash comparison made at
  release time; no 0.3.0 wheel exists in this environment, so the claim
  was not re-verified (the default-vs-explicit bitwise check above is the
  reachable half).
- **`ln_gamma_half_ratio` through the actual Rust binary** — the function
  is not exposed to Python; the probe mirrors its arithmetic in Python
  doubles (identical IEEE operations, libm-ulp caveat stated) and the
  committed Rust unit tests carry the in-binary assertions.
- **`lp_did` vs a live R/fixest run** — no R in this environment; the
  claim rests on the committed `fixtures/lpdid.json` (generated by the
  authors' code per its generator) plus this round's independent NumPy
  reference at 1e-10.
- **A full independent re-implementation of the ACM three-step pipeline**
  — not rebuilt in-probe; the committed generator *is* that
  re-implementation (read and verified non-circular), and this round
  checked its integrity plus the internal identities instead.
- **GAS/DCS-t against an external reference** — none exists (the card
  says so); the ridge attack used internal invariants and known-truth
  DGPs only.

# Fix hygiene (the checker's questions, answered)

No test was weakened or deleted: the copula property-test rename kept its
assertions verbatim and gained a sibling test; every other test change is
purely additive (2 new Rust assertions groups, 3 new Python tests). No
tolerance changed anywhere. The rebuilt module's numerics are
**bit-identical** to the released 0.4.0 build on every touched surface
(`p12_no_behavior_drift.py` old-vs-new JSON diff: empty) — the round's
fixes are documentation, `.pyi`, cards, and tests only. Affected suites
after rebuild: `cargo test -p tsecon-copula` 27/27; pytest over
`test_copula.py` + `test_acm.py` + `test_docstring_keys.py`: 82 passed.

# Bottom line

The 0.4.0 estimator surface is in the best first-contact shape of any
release this audit has touched: **zero code defects** across 378 reached
comparisons, with the independent-reference checks (statsmodels OLS/ncx2
for the first stage, NaN-guarded scipy MLEs for all five copula families,
a from-scratch LP-DiD with fixest-convention clustered SEs) agreeing to
1e-9–1e-14. The two post-freeze repairs survive the attacks they were
built against, with measured margins two to three orders better than
their docs claim. What the round did find is the familiar class-5 residue
— a reference-match claim that silently assumed a non-default reference
setting, an invariance stated one adjective too broadly, and three
undocumented echo keys — all fixed here, each with a tripwire so it
cannot silently regress.
