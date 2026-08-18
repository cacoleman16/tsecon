# Adversarial audit, round 6 — findings

> **Working document.** Continuation of [rounds 3–4](18-audit-rounds-3-4-findings.md),
> run under [the brief](16-adversarial-audit-brief.md). Excluded from the
> published site.

Run against the freshly built **0.3.0** tree — the same session that merged the
five new estimator families (DF-GLS, Zivot-Andrews, STL, SETAR, EVT) and the
rounds-1–4 fix backlog — so this round had two jobs the earlier rounds could
not: adversarial first contact with the new surface, and the checker role over
the fixes themselves. Four independent finders (new-surface lenses 1–5;
lens 7 as Bayesian calibration of the `bvar_*` family; lens 7 on break-date CIs
and proxy sets; a fix-checker over the merge range), then one refuter per
finder, defaulting to refuted, reproducing everything with independent seeds
and code. Probe scripts were scratchpad-only; the audit itself was read-only.

**Raised: 13 candidates. Survived refutation: 9 confirmed + 2 rescoped-narrower.
Fixed in the same release: 7. Recorded open: 2 (both documented at every
surface).**

---

# Confirmed, and fixed in 0.3.0

## 1 — `bvar_hierarchical`'s ML-II default collapsed to the search floor, and bands at the selection covered 6%

**`trap`** — the round's headline, and lens 7's cleanest yield to date. Drawing
θ from the model's own prior (λ1 = 0.2, s0 = 1; n=2, p=1, T=120) and fitting
with the shipped default `hyperprior="none"`: `lambda1_opt` collapsed below the
1e-3 neighbourhood of the box floor in **15–24%** of replications (finder 24.4%
at seeds 10k+, refuter 15.3% at seeds 900k+, 19.2% at the all-defaults
`lambda0=100` arm), and 90% credible IRF bands refit at a collapsed selection
covered **0.057** (finder and refuter numerically identical). Overall coverage
at the selected λ was 0.64–0.68 against nominal 0.90. The card's only relevant
failure mode ("the optimum is then noise — check the `grid_log_ml` profile")
would have *reassured* the user: in collapsed replications the floor peak beat
λ=0.2 by a median **2.26 log points** — the marginal likelihood genuinely peaks
at λ→0 (classic empirical-Bayes variance-component collapse; worst in small
systems, fading by ~16 slopes). The optimizer was exonerated (polish never
below the grid max, 0/499 across both agents).

**Fix (this release, BREAKING):** the default `hyperprior` is now `"glp"` —
the Gamma(mode 0.2, sd 0.4) hyperprior GLP (2015) themselves recommend, which
eliminated the floor collapse entirely in the same experiment (0/150 at the
finder's own seed stream, verified post-fix like-for-like: default 0.000 vs
explicit `"none"` 0.220 below 1e-3). `hyperprior="none"` remains the pure
ML-II escape hatch, with the red-flag advice on all three doc surfaces. The
card's failure modes now state the collapse, the misleading-profile fact, and
the plug-in caveat below.

## 2 — The GLP plug-in's own calibration, measured: 90% bands cover 0.82–0.85

**`trap` → documentation.** Even on the well-behaved `"glp"` route, bands refit
at `lambda1_opt` are a plug-in that ignores selection uncertainty: full SBC
with λ1 drawn from the model's asserted Gamma hyperprior measured c90 =
**0.819–0.848** (finder and refuter, independent streams) against an
exact-λ oracle at ~0.90. Additionally the AR(4) empirical-Bayes scale rule
produces mild long-horizon *conservatism* (c68 ≈ 0.75–0.78, c90 ≈ 0.94–0.95 at
h ≥ 4) — present identically in the independent oracle, i.e. a property of the
documented prior rule, not a code defect. **Fix:** both facts are now in the
card's calibration paragraph and the docstring.

## 3 — `seasonal_strength` returned a float-noise ratio ≈ 0.64 on constant series

**`silent-wrong-answer`** (narrow — degenerate input). A flat line returned
`seasonal_strength` ≈ 0.61–0.67 depending on the constant (0.6406 at c=3.7,
0.6465 at c=1e6; exactly 0.0 only at c=0.0) — a ratio of ~1e-32 decomposition
float-noise variances, unpinnable against the reference (statsmodels-composed
gives 0.627 vs tsecon's 0.641 on identical input) and sitting at the `nsdiffs`
0.64 threshold. Every sibling diagnostic raises on constants and `nsdiffs` /
`check_series` guard it themselves; the standalone function was the only
unguarded door. **Fix:** `seasonal_strength` now raises the teaching
`ConstantSeries` error (`FiltersError::ConstantSeries`), exact-equality guard
so near-constant data still runs; regression tests at both surfaces.

## 4 — `bvar_ssvs` blamed missing values for its own internal overflow

**`cosmetic`** (misattributed diagnostic — the rounds-3/4 shape). On all-finite
explosive-magnitude input (observed from max|y| ≈ 6e11 upward) the sampler's
internal overflow escaped through the shared linear-algebra hygiene guard,
whose message says "drop or impute them before estimating" — unfollowable on
data with nothing missing (`bvar_fit` succeeds on the identical matrix). The
Rust doc even claimed the path was "not observed for valid inputs".
**Fix:** after input validation, internal `Linalg` errors are re-labelled with
magnitude/rescaling advice that never mentions missing values; regression test
pins the message both ways.

## 5 — `har_rv`'s docstring halved its own breaking change: "+0.17%" for a true +0.35%

**`cosmetic`** (found by the fix-checker). `sqrt(577/573) = 1.00348`; the
CHANGELOG said +0.35% correctly, the runtime `__doc__` and the Rust doc said
+0.17% (the sqrt-halving applied twice). **Fix:** both surfaces corrected.

## 6 — `zivot_andrews` documented `trim ∈ [0, 1/3]` but `trim=0` can never run

**`cosmetic`.** The candidate window must hold ≥ `lags + 1` observations
(`int(n·trim) ≥ lags + 1`), so the documented lower endpoint is structurally
unreachable at every `lags` including 0. **Fix:** all three surfaces now say
`(0, 1/3]` and state the coupling.

## 7 — `proxy_ar_sets`' `kind` enumeration disagreed across surfaces

**`cosmetic`.** `__doc__` and the `.pyi` listed five kinds; the card and the
code have seven (`"ray_below"`/`"ray_above"` missing). **Fix:** enumerations
aligned.

# Confirmed, recorded open (documented, not yet re-engineered)

## 8 — `proxy_ar_sets`' propagated coverage keeps declining past the published table, through the function's own default horizon

**`trap`.** The card's coverage table stopped at h=8 (worst published 0.913)
while the default is `horizon=12` and the card's own worked example uses it.
Measured on the card's own DGP: **0.876–0.894 at h=12** (worst cell ≈ 0.85);
on a routine VAR(1), T=250: **0.80–0.85** (refuter's own DGP, so not
cherry-picked). Misses are one-sided — the truth sits *above* the set, because
the propagated variance shrinks together with Ψ̂_h at long horizons — and fade
in T (0.907 by T=1000). The sibling `proxy_svar_bands` card already disclosed
its own long-horizon shortfall; this one did not. **This release:** the
docstring, card table commentary, and failure modes now carry the h=12 numbers
and the one-sided mechanism. **Open:** the underlying estimate-correlated
propagated variance at long horizons is a real inference problem; a
re-engineered long-horizon correction (or a joint band) is future work.

## 9 — The seasonal-strength rule saturates below ~4 cycles, flagging pure noise

**`trap`** (doc-gap; the reference behaves identically, so this cannot be a
code defect). With 2 cycles (n=24, period 12) `seasonal_strength` is 1.000 on
*every* white-noise draw and `nsdiffs` returns D=1 on 100% of them; 38% at
n=48; 0% by n=120. R's `forecast::nsdiffs` has no guard either (verified in
`unitRoot.R` by both agents), and the promise is "the rule
`forecast::nsdiffs(test="seas")` implements" — kept. The `stop="TooShort"`
marker warns in the *other* direction. **This release:** the card's failure
modes now carry the measured saturation table and the "not enough data to
tell" reading. **Open:** whether a minimum-cycles advisory (not a refusal —
that would diverge from the reference) belongs in the `nsdiffs` output itself.

# Rescoped / residue

- **`bvar_ssvs["diagnostics"]` extra keys** — rescoped by the refuter: the
  runtime `__doc__` never enumerates keys (finder misread), only the card
  does; `burn`/`thin` are benign echoes. The card's key list now includes
  them. Residue only.
- **Negative integer arguments raise raw `OverflowError`** (`lags=-1`,
  `outer_iter=-1`, …) instead of the library's typed teaching `ValueError` —
  a pre-existing, library-wide PyO3-conversion convention (old functions
  behave the same), so recorded as a known cosmetic convention rather than a
  new-code finding. A uniform negative-argument message is a candidate
  small-fix for a future round.

# Swept and found sound

- **The five new 0.3.0 estimator families**: 56/56 lens-1 axis comparisons
  reached (zero silent no-ops — every axis proven alive on some dataset);
  10/10 scale sweeps across sixteen decades clean (invariants ≤ 4.5e-14,
  equivariants ≤ 5.2e-12 closed-form / ≤ 5e-6 for the two EVT MLEs within
  their documented scipy slack); 112/112 degenerate-input probes reached —
  raises are typed and teaching, boundary EVT fits honestly report
  `se_valid=False` / `converged=False`; all four card snippets with expected
  output reproduce byte-for-byte; 9/9 key-set diffs clean across the three
  doc surfaces; constant-diagnostic scan clean after classification.
- **`bvar_fit`'s conjugate family is machine-exact**: prior/posterior/logML
  ≤ 1.7e-15 against an independent NumPy/SciPy NIW oracle (15/15); the
  posterior sampler's Σ marginal passes exact KS against the closed form
  (10 seeds × 20k draws, pooled p=0.69); full 250-rep SBC finds tsecon's
  ranks equal to the oracle's to MC noise — **no code defect in the
  Bayesian core**.
- **`bvar_hierarchical`'s internals**: grid profile bit-identical to
  `bvar_fit`'s logML, refit-at-optimum bit-identical, `log_posterior` =
  logML + exact Gamma log-kernel (≤ 7e-15), dominance certificate 0/500.
- **`bai_perron` break-date CIs and `sup_f_test` size**: swept by finder 3
  within its disclosed scoping and found consistent (no confirmed finding
  survived; the round's proxy yield came from `proxy_ar_sets`).
- **The 0.3.0 fixes and merges themselves** (the checker): across the full
  merge range (157 files, +47,984/−432) — 679 asserts added, 4 removed and
  each replaced by a strictly stronger version, zero deleted test files,
  zero widened tolerances, zero new unconditional skips; all four claimed
  test-change justifications verified against the diff; the F1/F5 and F3/F5
  interaction hazards checked and clean (`check_page.py` exits 0 on the
  merged tree; no coverage-page probe touches a code path F1 changed); the
  public callable count (137) verified against the live module.

---

## Reproducing

Probe scripts were scratchpad-only. The generative designs worth rebuilding:
the NIW prior-draw SBC harness (finder 2 / refuter — validate the harness on
the fixed-λ arm first: it must sit at nominal before you believe anything it
says about selection), the `proxy_ar_sets` coverage harness with its
truth-validation step (estimand checked against a T=200k fit before any
coverage is counted), and the post-fix like-for-like collapse probe
(`postfix_hier.py`, preserved in the session scratchpad) whose numbers are
quoted in finding 1.
