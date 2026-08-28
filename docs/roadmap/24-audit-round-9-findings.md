# Adversarial audit, round 9 — findings

> **Working document.** Continuation of
> [round 8](23-audit-round-8-findings.md), run under
> [the brief](16-adversarial-audit-brief.md). Excluded from the published
> site.

Round 9 is the first audit round seeded from **outside**: a downstream
statistical-arbitrage project ran a probe battery against the live library
(the published v0.2.0 wheel — itself a finding, see the tag note in the
0.5.0 PR) and reported sixteen items. The round ran in three stages, each
under finder/refuter discipline with every verdict independently
re-derived before it counted.

**Stage 1 — triage of the sixteen field items** (10 probers + 10 refuters,
all read-only against the live 0.5.0 build; probe scripts under the session
scratchpad `triage/`). Verdicts, all refuter-upheld: **6 already fixed** on
main (items 1, 4, 6, 7, 10, 14 — resolved across 0.3.0–0.5.0; item 6,
`engle_granger`, was probed live rather than trusted from the changelog
because it is one of the four functions the Python suite does not
exercise); **4 still real** (items 2, 11, 12, 13); **1 partial** (item 3 —
the 0.5.0 DCC build-out closed most of it); **3 absent features** (items 5,
8, 9); **2 by design with documented contracts** (items 15, 16 — item 16's
`params_named` shipped anyway as an additive ergonomic key).

**Stage 2 — repo-wide sweeps of the four defect classes the field items
instantiate** (4 finders + 16 refuters, read-only): Rust-computed-but-
never-bound; inert/absorbed/mis-indexed arguments; missing convergence and
boundary signals; docs citing a convention the code deviates from.
**Raised: 28 candidates across the four sweeps. Verified: 14 confirmed, 0
refuted** (the four finders' top candidates all survived independent
re-derivation — a materially higher hit rate than any internal round,
which is the expected signature of sweeping *classes with a proven
instance* rather than open hunting).

**Stage 3 — fixes.** All four still-real field items and all fourteen
sweep findings were fixed, each in a worktree with its own verification
battery, merged sequentially with per-crate + clippy + fmt gates.

## Confirmed and fixed (field items)

1. **(item 2, footgun)** `markov_switching_ar` never returned the AR
   coefficients — stored privately in `tsecon-regime` with no accessor.
   Fixed: public `ar()` + an `ar` key; the Hamilton-replication guard was
   flipped, per its own docstring, into a three-way comparison — tsecon
   AR(4) = (0.0147, −0.0532, −0.2459, −0.2120), max |diff| **0.0048 vs
   Hamilton's published** φ's and 0.0043 vs statsmodels' exact MLE.
2. **(item 11, footgun)** `cv_splits(scheme="purged_kfold")` **absorbed
   the embargo into the purge** (`purge.max(embargo)`): measured, (21,10)
   was bit-identical to (21,0). The convention was verified against AFML's
   own Snippet 7.3, mlfinlab, and — decisively — the repo's own guide 12,
   which taught the additive rule the core didn't implement. Fixed to
   additive with measured-gap tests (21,10)→31, (21,30)→51. BREAKING.
3. **(item 12, correctness-trap)** `vecm` fit the no-deterministic case
   while `johansen` documents an unrestricted constant; docstrings
   byte-identical since v0.2.0 and silent about it (measured beta cosine
   0.6318 between the two on drifting log levels). Fixed both halves:
   `deterministic="n"|"co"` shipped at a statsmodels golden (1e-6 on
   α/β/Γ/`det_coef`/Σᵤ/llf, both cases), `johansen` gained `evec`, and
   the reconciliation is pinned (co↔johansen cosine 1±1e-10).
4. **(item 13)** `ivx` indexed its localizing sequence by the raw length
   while every sample-size object used N = n−1 (KMS 2015 wants N). Fixed
   at both call sites — and **the independent NumPy fixture generator
   shared the same misreading** (fixed there too; a validation-circularity
   lesson recorded below). Measured β shift 6.2e-6–8.8e-6 relative;
   goldens regenerated, no tolerance widened.

## Confirmed and fixed (sweep findings)

**Class: Rust-computed-but-never-bound** — `dfm_nowcast` returned factors
with no loadings (the fitted model was unreproducible; now returns
loadings/factor_ar/factor_cov/idiosyncratic/center/scale with the exact
factor-to-series mapping pinned at 1e-10); `bvar_fit` returned a posterior
with no uncertainty (now omega_bar/s_bar/v_bar with the posterior-sd
recipe validated by a 40,000-draw NIW MC within 5%); `var_fit` hid its
residuals (now resid/fitted/nobs/df_resid, reconstruction bitwise);
`dcc_garch` stage-1 (per-series univariate dicts bitwise-equal to direct
`garch_fit`, std_residuals, ADCC nbar).

**Class: inert/absorbed/mis-indexed arguments** — `var_fevd` emitted a
**transposed** array relative to its own docs (variable-major reality vs
documented horizon-first; misreads cleanly because both axes' slices
row-sum to 1; fixed to horizon-first, BREAKING, with a k≠horizon aliasing
guard); the spectral trio defaulted `detrend="none"` while claiming scipy
parity (scipy: 'constant'; measured default-vs-default welch gap **1678.1
at f=0**, now 2.7e-15; BREAKING); `garch_fit` silently discarded `o` under
`vol="garch"` — the exact `arch_model(y, p=1, o=1, q=1)` porting trap its
own docstring warns about (now a teaching refusal, sentinel default);
`panel_fe`/`panel_lp` silently absorbed `bandwidth` under the default
cluster SEs (now a teaching refusal, cv_splits-style).

**Class: missing convergence/boundary signals** — `panel_pmg` hard-failed
**13–16/20** textbook I(1) panels: the finder blamed the absolute 1e-12
tolerance, but the fixer's re-investigation found the iteration genuinely
**diverges** from the pinned θ=0 start (θ walking at ~0.68/iteration to
−179 by iteration 260), so the shipped repair is a relative rule (constant
3e-13 chosen from the measured plateau window [1.9e-13, 5.9e-13) that
preserves the golden's stopping iterate bit-for-bit) **plus** a
deterministic restart from the PSS unrestricted-ARDL start — post-fix
0/20 at every scale; `arima_fit` dropped the crate's converged flag and
reported finite SEs with cov_ok=True at an MA-invertibility pileup
(θ = −1.000000 on 8/14 over-differenced seeds; now converged + per-param
boundary/se_valid/boundary_note, tier-2 reduced-Hessian SEs recorded as a
stated follow-up); `quantile_lp`/`growth_at_risk` dropped the IRLS
converged flag (a deterministic real exhaustion case was found at
(τ=0.5, h=1), T=200, and pinned); `dfm_nowcast(method="mle")` ignored both
optimizer stages' certificates (now converged/iterations — and the crate's
own statsmodels-gap fixture honestly reports converged=False at 1355
iterations).

**Class: docs-vs-cited-paper** — `cg_series` built a fixed-horizon
revision where Coibion-Gorodnichenko's estimand requires **fixed-event**
(measured on an exact sticky-information DGP: fixed-event slope 1.013 at
β=1 recovering λ=0.503, fixed-horizon slope 0.765 — pinned to its derived
closed-form plim β(1+ρ)/2 = 0.75; new `cg_series_fixed_event` + honest
re-documentation of the old path); `bns_jump_test` omitted Huang-Tauchen's
M/(M−1) and M/(M−2) factors while citing HT's size claims (measured: the
5% one-sided decision **flips** on a standard M=78 day, z 1.689 vs 1.564;
fixed to HT with null size measured 0.053 at 4000 reps; the exported
BNS-2004 measures unchanged, documented).

## Refuted / not defects

Stage-1 items 15 and 16 (garch conditional `variance_forecast` key and the
missing name→value mapping) were upheld as documented contracts — the
conditional key is documented twice and the facade is the named-access
route — with `params_named` added anyway as an additive key. All four
stage-2 sweeps' verified candidates survived; the finders' unverified tails
(4 additional candidates per sweep beyond the verification cap) are listed
in the sweep transcripts and remain open for a future round.

## Lessons recorded

1. **A field probe battery against the live wheel found things eight
   internal rounds had not** — short-sample robustness batteries and
   cited-convention diffs are now standing checks in the brief.
2. **Independent reimplementation is not independent reading**: the ivx
   fixture generator reproduced the same n-vs-N misreading as the crate.
   Where a golden's generator transcribes the same paper as the code, the
   transcription itself needs a second reader.
3. **Class sweeps out-hit open hunting**: 14/14 verified candidates
   confirmed, against roughly one-in-three for open-ended rounds. When a
   defect class has one proven instance, sweep the class.
