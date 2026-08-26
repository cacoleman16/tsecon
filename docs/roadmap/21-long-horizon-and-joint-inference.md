# Long-horizon and joint inference — re-engineering the two open audit problems

> **Working document.** Follow-up engineering investigation of the two
> problems the audits recorded open: [round 6](20-audit-round-6-findings.md)
> finding 8 (`proxy_ar_sets`' one-sided long-horizon under-coverage) and
> [rounds 3–4](18-audit-rounds-3-4-findings.md) finding 4 (`ivx_test`'s joint
> Wald losing its size in the number of predictors). Excluded from the
> published site.

Both problems were taken from "documented honestly" to "measured against
candidate corrections", on seeded Monte Carlo with the round-6 harness rules:
the estimand validated against a huge-sample fit before any coverage is
counted, every candidate arm run on the same draws as the baseline, and the
NumPy transcriptions cross-checked against the shipped crate before being
believed. The two experiment harnesses are committed:

* `docs/examples/coverage/experiments/proxy_ar_long_horizon.py`
* `docs/examples/coverage/experiments/ivx_joint_size.py`

**Outcome: both problems shipped a measured, opt-in repair** —
`proxy_ar_sets(..., rf_method="second_order")` and
`ivx_test(..., joint="bonferroni")` — with the defaults unchanged (a default
flip needs its own audit round). Four candidate directions were measured and
discarded with evidence; they are recorded below so nobody re-runs them on
faith.

---

## Problem A — `proxy_ar_sets` at long horizons

**The finding.** Propagated coverage keeps declining past the card's
published table: 0.876–0.894 at the default `horizon=12` on the card's own
VAR(2) DGP, 0.80–0.85 on a routine VAR(1) at T=250; every miss one-sided (the
truth sits *above* the set); fading in T.

**The mechanism, measured directly.** The delta-method reduced-form variance
is evaluated at the estimated coefficients. At h=12 on the card DGP (500
reps): corr(propagated sd, |point error|) = **+0.72**, while the mean
propagated sd is **0.94** of the empirical sd of the point — the first-order
variance is essentially right *on average* and too small in exactly the
under-persistent draws that miss. This kills any fix that merely rescales the
variance; what is needed is a variance that decouples from (or convexifies
against) the estimate.

### The candidates, on the same 500 seeded draws

Nominal 0.95; `h ∈ {8, 12}` shown; `above/below` = miss directions at h=12;
`wr` = paired median interval-width ratio to the delta baseline. Full
by-horizon rows are printed by the harness.

**Card VAR(2), T=300, strong instrument** (baseline reproduces the audit:
0.889 at h=12):

| arm | h=8 | h=12 | above/below | wr h=8 | wr h=12 | s/rep |
|---|---|---|---|---|---|---|
| delta (shipped) | .913 | .889 | 166 / 0 | 1.00 | 1.00 | 0.002 |
| **delta2 = second_order** (exact simulation of the coefficient uncertainty through the α→Ψ map) | **.944** | **.964** | 54 / 0 | 1.14 | 1.44 | 0.003 |
| bcvar (delta at Pope-bias-corrected coefficients) | .939 | .929 | 107 / 0 | 1.14 | 1.26 | 0.005 |
| floor (monotone relative-variance floor) | .974 | .969 | 47 / 0 | 1.00 | 1.00 | 0.002 |
| boot-v (parametric-bootstrap ψ-variance, B=256) | .909 | .925 | 112 / 0 | 0.96 | 1.07 | 0.64 |
| boot-c (parametric-bootstrap critical value, per cell) | .889 | **.842** | 237 / 0 | 1.12 | 1.34 | 0.64 |

**Routine VAR(1), T=250** (baseline reproduces the audit: 0.830 at h=12):

| arm | h=8 | h=12 | above/below | wr h=8 | wr h=12 |
|---|---|---|---|---|---|
| delta | .888 | .830 | 255 / 0 | 1.00 | 1.00 |
| **delta2 = second_order** | **.928** | **.932** | 102 / 0 | 1.16 | 1.47 |
| bcvar | .922 | .877 | 185 / 0 | 1.16 | 1.28 |
| floor | .934 | .914 | 129 / 0 | 1.00 | 1.00 |
| boot-v | .879 | .859 | 212 / 0 | 0.95 | 1.03 |
| boot-c | .883 | .829 | 256 / 0 | 1.32 | 1.94 |

**Weak-instrument guard** (φ=0.06, 250 reps): every candidate keeps the weak
arm conservative (coverage ≥ 0.996 for all but boot-c) and — except boot-c —
leaves the bounded-cell share bit-identical at 0.108. **boot-c raises the
bounded share to 0.219**: its per-cell critical values can fall below the
chi-square value, which loosens the boundedness threshold under weak
identification. That is a weak-IV-robustness red flag on top of its coverage
failure.

### Verdicts

* **Shipped: `rf_method="second_order"`** (with `rf_draws`, `rf_seed`),
  implemented as `psi_reduced_form_cov_mc` in `tsecon-ident` — antithetic
  Gaussian coefficient draws pushed through the exact MA recursion, equal to
  the delta method to first order, plus the convexity that grows with h. It
  is the only arm that moves both DGPs to within ~2pp of nominal at both
  h=8 and h=12, it cuts the one-sided misses ~3x, its price is bounded
  (1.14–1.16x median width at h=8, 1.44–1.47x at h=12 — against the 13.5x the
  propagation itself already costs), and the weak arm is untouched (the
  correction is quadratic in γ, so it still vanishes with relevance;
  boundedness is decided by the same statistic bit-for-bit). Honest residual:
  on the harder VAR(1)-at-T=250 DGP it reaches 0.932 at h=12, not 0.95 —
  closer, not exact. The remaining gap is the part of the coupling that
  convexity alone does not undo (the bias channel bcvar targets); a
  combination arm (second_order at bias-corrected coefficients) is the
  natural next candidate if a future round wants the last 2pp.
* **Discarded: boot-c** (bootstrap-calibrated critical values). It makes the
  long horizon *worse* (0.842/0.829) for a structural reason worth recording:
  the bootstrap world's truth is the *fitted* VAR — less persistent than the
  DGP in exactly the draws that matter — so the bootstrap distribution of the
  AR statistic inherits the very coupling it was meant to calibrate away.
  Self-referential calibration cannot see a bias in its own reference point.
* **Not shipped, viable fallback: floor.** Free at the median and decent
  coverage, but it over-covers mid-horizons (0.97 rows where delta was at
  nominal), it is a heuristic with no order argument, and second_order beats
  it where it matters.
* **Partial: bcvar, boot-v.** Real but insufficient movement; bcvar's
  evaluation-point channel is complementary to second_order's convexity
  channel (see above).

---

## Problem B — the IVX joint Wald in k

**The finding.** Size ~0.05/0.10/0.17/0.26–0.28 at k=1/3/5/8 (ρ=1, δ=−0.9,
n=250) at the shipped `alpha=0.95`; excess decays like `n^{-0.025}`; `k=1` is
the only tested and the only calibrated k.

**Mechanism diagnostics** (committed harness prints these):

* The scalar (k=1) statistic is calibrated *deep into its tail* at the hard
  corner: P(W > χ²₁ quantile) = 0.048 / 0.0089 / 0.0011 at the 95 / 99 /
  99.9% points. The k=1 "cancellation of two errors" is not a 5%-point
  coincidence — the scalar test is simply a good test. This is the fact the
  Bonferroni repair leans on.
* At δ=0 the **demeaned** normaliser `s2u·(Z'Z − N z̄z̄')` is at nominal for
  every k (0.052 at k=1, 0.069 at k=8) while the shipped raw normaliser is
  conservative — i.e. the raw Z'Z's rank-one overstatement is the only δ=0
  distortion, and the whole k-problem lives in the endogeneity × demeaning
  interaction.
* Under endogeneity the demeaning term is a *conditionally deterministic
  shift* along the realized z̄ direction, not extra Gaussian noise: a
  design-phase probe (scratchpad-only, like the audits' probes) measured the
  standardized mean of the endogenous coordinate of c at ≈ 1.3 sd, and even
  an oracle E[cc′] normaliser left the joint test at 0.10–0.12. No variance
  matrix fixes a shift — which is why the two "fix the matrix" candidates
  below had to fail.

### Size (2000 reps/cell, nominal 0.05, MC se ≤ 0.011)

Worst-case design rows (`first` = endogeneity on predictor 0; the `factor`
design with cross-correlated predictors is strictly worse for chi2 and the
same for Bonferroni — full 64-cell table in the harness output):

ρ=1.0, δ=−0.9, n=250:

| k | chi2 a=0.95 (default) | a=0.70 | a=0.50 | a=0.30 | demeaned | FM | **bonferroni** |
|---|---|---|---|---|---|---|---|
| 1 | 0.053 | 0.054 | 0.056 | 0.046 | 0.195 | 0.058 | 0.053 |
| 3 | 0.097 | 0.073 | 0.057 | 0.049 | 0.237 | 0.100 | 0.022 |
| 5 | 0.168 | 0.120 | 0.093 | 0.081 | 0.314 | 0.170 | 0.017 |
| 8 | 0.277 | 0.204 | 0.126 | 0.080 | 0.427 | 0.280 | 0.018 |

Same corner, `factor` design: default reaches 0.342 at k=8; bonferroni 0.041.
Across the whole grid (`k ∈ {1,3,5,8} × ρ ∈ {0.95,1} × δ ∈ {0,−0.9} ×
n ∈ {250,1000}` × two designs), bonferroni spans **0.011–0.059** — never
materially above nominal; conservative at the unit-root corner and at δ=0
(inheriting the scalar test's own δ=0 conservatism plus dependence slack).

Wild bootstrap (restricted system wild: null-imposed return residuals, AR(1)
predictor residuals, one shared Rademacher weight per date, x* rebuilt
recursively; B=199, 500 reps): **0.190 at k=1** and 0.082 at k=8 (δ=−0.9,
ρ=1, n=250); fine at δ=0 (0.052/0.046). Runtime 4–29 ms per test. It
under-corrects at k=8 and *breaks* the k=1 case the shipped test already
gets right — the bootstrap plugs in ρ̂, and no estimator is consistent for
the local-to-unity parameter the null distribution depends on. Discarded.

The demeaned control is verified to reject **more** (as rounds 3–4 found),
and the KMS-style FM normaliser `s2u·Z'Z − N z̄z̄'·Ω_FM` moves nothing
(0.280 vs 0.277 at k=8) — consistent with the shift mechanism above.

### Power (n=250, δ=−0.9, size-adjusted where size is broken)

"adj" = rejection against that statistic's own empirical null 95% quantile
(infeasible in practice — the honest benchmark); bonferroni is raw (its size
is at or below nominal):

| ρ | k | alternative | slope | default (adj) | a=0.5 (adj) | **bonferroni** |
|---|---|---|---|---|---|---|
| 1.0 | 3 | sparse | 0.02 | 0.378 | 0.203 | 0.341 |
| 1.0 | 3 | sparse | 0.04 | 0.865 | 0.582 | 0.846 |
| 1.0 | 3 | diffuse | 0.04 | 0.764 | 0.521 | 0.606 |
| 1.0 | 8 | sparse | 0.04 | 0.766 | 0.536 | **0.801** |
| 1.0 | 8 | diffuse | 0.04 | 0.605 | 0.469 | 0.530 |
| 0.95 | 3 | sparse | 0.04 | 0.312 | 0.185 | **0.342** |
| 0.95 | 8 | diffuse | 0.04 | 0.170 | 0.139 | 0.124 |

Bonferroni matches (sometimes beats) the size-corrected chi-square benchmark
against sparse alternatives — the horse-race question — and gives up a fifth
to a quarter of the power against diffuse alternatives. It **dominates**
size-adjusted `alpha=0.5` everywhere measured.

### Verdicts

* **Shipped: `joint="bonferroni"`** — `ivx_bonferroni` in `tsecon-predreg`
  (union-intersection: the certified scalar test per predictor, joint
  rejection at level/k), surfaced as `ivx_test(..., joint="bonferroni")`
  with `wald_scalar`/`pvalue_scalar` per predictor (which the card had noted
  as missing from the surface). Its validity argument invokes no new
  asymptotics: each statistic is exactly the scalar object whose
  uniform-over-persistence size the crate already certifies and whose deep
  tail is measured, and Bonferroni is valid under arbitrary dependence.
  Honest residual: conservative (0.016–0.025) at the unit-root/strong-
  endogeneity corner, and diffuse alternatives pay; the chi-square default
  stays for small k or ρ safely below 1.
* **Discarded with evidence: demeaned variance** (rejects more — it is the
  right variance only at δ=0), **FM-corrected normaliser** (no movement — the
  distortion is a conditional shift, not a variance term), **wild bootstrap**
  (under-corrects at k=8, breaks k=1, and is the only candidate that would
  have *degraded* a currently-correct case).
* **Documented, not a fix: the alpha ladder.** `alpha=0.5` halves the k=8
  distortion (0.126) and restores convergence in n; `alpha=0.3` gets to
  ~0.08 at the price of further power (and at δ=0 walks the scalar test's
  conservatism toward exactness). No fixed alpha holds 0.05 at n=250, k=8.
* The suite now pins the defect itself: a k=5 chi-square size regression
  test (must stay ≥ 0.10) runs beside the Bonferroni size/power tests — the
  missing k>1 coverage rounds 3–4 flagged.

---

## What shipped (both defaults unchanged)

| surface | Problem A | Problem B |
|---|---|---|
| Rust | `tsecon_ident::proxy_ar::psi_reduced_form_cov_mc` | `tsecon_predreg::ivx_bonferroni` + `IvxBonferroniResult` |
| binding | `proxy_ar_sets(..., rf_method=, rf_draws=, rf_seed=)`, no-op-proof guards | `ivx_test(..., joint=)`, extra keys only in the new mode |
| stub / api.md | updated + regenerated | updated + regenerated |
| tests | 6 crate tests (determinism, delta-limit, convexity direction, end-to-end widening + boundedness invariance, draw-count contract) + 3 Python tests | 6 crate tests (incl. the k>1 size regression) + 4 Python tests |
| card | structural-identification.md: measured table, mechanism, discards | predictive-regressions.md: three-row size table, power, discards, horse-race example |
| default | `rf_method="delta"` unchanged | `joint="chi2"` unchanged |

Default flips (making `second_order` and/or `bonferroni`-at-large-k the
defaults) are deliberate future decisions for an audit round that can weigh
the width/power prices against the size gains on fresh seeds.

## Reproducing

Both harnesses are seeded end to end, validate their transcriptions against
the installed `tsecon` before measuring, and print every table above:

```
.venv/bin/python docs/examples/coverage/experiments/proxy_ar_long_horizon.py   # ~15 min
.venv/bin/python docs/examples/coverage/experiments/ivx_joint_size.py          # ~5 min
```

`--quick` smoke modes run in about a minute each.

---

## 2026-08-23 follow-up — the `second_order` residual ~2pp, measured and shipped

The Problem-A verdict above recorded an honest residual: `second_order`
reaches 0.932 at `h=12` on the routine VAR(1) at T=250, not 0.95, and named
"a combination arm (second_order at bias-corrected coefficients)" as the
natural next candidate. That arm — `delta2bc` in the harness: the same
antithetic Gaussian coefficient draws, **centred at Pope (1990)
bias-corrected coefficients** (Kilian stationarity shrinkage) instead of the
raw least-squares fit — is now measured on the same 500 seeded draws
(`--arms delta,delta2,bcvar,delta2bc`; the boot arms were not re-run):

**Card VAR(2), T=300** (nominal 0.95; `mean` is over h ≥ 1):

| arm | mean | h=8 | h=12 | above/below h=12 | wr h=8 | wr h=12 |
|---|---|---|---|---|---|---|
| delta | .924 | .913 | .889 | 166 / 0 | 1.00 | 1.00 |
| delta2 = second_order | .951 | .944 | .964 | 54 / 0 | 1.14 | 1.44 |
| bcvar | .944 | .939 | .929 | 107 / 0 | 1.14 | 1.26 |
| **delta2bc = second_order_bc** | .965 | .970 | **.982** | 27 / 0 | 1.30 | 1.78 |

**Routine VAR(1), T=250** (the residual-gap DGP):

| arm | mean | h=8 | h=12 | above/below h=12 | wr h=8 | wr h=12 |
|---|---|---|---|---|---|---|
| delta | .900 | .888 | .830 | 255 / 0 | 1.00 | 1.00 |
| delta2 = second_order | .936 | .928 | .932 | 102 / 0 | 1.16 | 1.47 |
| bcvar | .927 | .922 | .877 | 185 / 0 | 1.16 | 1.28 |
| **delta2bc = second_order_bc** | .958 | .957 | **.966** | 51 / 0 | 1.33 | 1.82 |

**Weak-instrument guard** (φ=0.06, 250 reps): coverage stays conservative
(0.999 at h=4, 1.000 from h=5 on), and the bounded-cell share is bit-identical at 0.108
for every arm — the centring enters `v0` only, so the boundedness statistic
never moves.

**Verdict — shipped, with its honest shape stated.** The residual gap *is*
the bias channel, as the original verdict conjectured: adding the
evaluation-point correction on top of the convexity closes it (0.932 →
0.966 at the residual cell). But it does not close it *to* nominal — it
crosses to the conservative side, and on the card VAR(2), where
`second_order` had already reached nominal, it overshoots further (0.964 →
0.982) at ~1.25x `second_order`'s width. `delta2bc` is the only arm measured
at or above nominal at **every** horizon on both DGPs, which makes it a
**conservative floor rather than a calibration** — the two channels
overlap, and stacking them buys guaranteed-side coverage, not exactness.

Shipped as `proxy_ar_sets(..., rf_method="second_order_bc")` (same
`rf_draws`/`rf_seed` knobs), implemented as
`tsecon_ident::proxy_ar::pope_bias_corrected_coefs` (Pope's closed form with
the eigenvalue sum evaluated as the real trace power series
`sum_j tr(A^{j+1}) (A')^j`, Kilian 0.05-step shrinkage, and the harness's
conservative no-ops: an unstable fit, a non-convergent series, or a
non-shrinkable correction all return the coefficients unchanged) feeding the
existing `psi_reduced_form_cov_mc`. Crate tests pin the AR(1) closed form
`E[a_hat] - a = -(1+3a)/T` exactly, the harness's NumPy transcription at
1e-10 on a VAR(2) and on a shrunk near-unit-root case, the unstable no-op
bit-for-bit, the h-growing variance excess over `second_order`, and the
boundedness/point invariance. Both earlier defaults remain unchanged;
`second_order` remains the best point-calibration choice, and the registry
(`docs/examples/coverage/run_all.py`, via its `proxy_garch_tail` family
module `docs/examples/coverage/proxy_garch_tail.py`) now measures all three
arms every run.
