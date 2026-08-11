# Adversarial audit, rounds 3–4 — findings

> **Working document.** Continuation of [round 2](17-audit-round-2-findings.md),
> run under [the brief](16-adversarial-audit-brief.md). Excluded from the
> published site.

Read-only throughout; HEAD unchanged, `git status` clean at both ends. No cargo:
every probe drives the already-built Python surface.

Same method — independent finder per lens, every finding then sent to an agent
whose job was to **refute** it, defaulting to refuted.

**Rounds 3–4: 8 candidates raised, 3 survived.** Refutation killed 4, re-scoped
2 of the 3 survivors, and merged 1 into a round-2 finding.

---

## 0 — The audit's own tooling had the defect it hunts

Round 2's report claimed *"164 axis-comparisons over all 45 switch-carrying
callables."* A checker that counts comparisons **made** rather than **attempted**
found otherwise:

| | axes / cases | fully swept | partial | **never compared** |
|---|---|---|---|---|
| lens 1 switch axes | 59 | 38 | 5 | **16** |
| lens 1 seed cases | 12 | 6 | — | **6** |

A probe template that raises produces no comparison, and the sweep output is
indistinguishable from a clean pass — *an argument that looks like it does
something and does not*, in the tooling built to find exactly that. Most were bad
templates (`favar(data=)` where the argument is `panel=`;
`markov_switching_ar(n_regimes=)` where it is `k_regimes=`; kernel
`"quadratic_spectral"` where the menu is `"qs"`).

**The scale sweep did not have this defect** — it printed `reference scale
FAILED` for every template that raised — but it still left **13 functions**
unreached.

**Both holes are closed and nothing was hiding in either.** Lens 1 pass 2:
3 apparent hits, all refuted. Lens 2 pass 2: 13 functions swept across nine
decades, 13 clean, and its one hit independently re-derived the confirmed
`panel_pmg` finding — evidence the sweep bites. This is now a hard constraint in
the brief: **report comparisons made, never attempted.** The round-3
degenerate-input sweep, told this up front, reported 192 attempted / 192 reached
/ 0 template-raised; round 4's lens-4 sweep reports 64/64.

---

# Confirmed

## 1 — `panel_fe` reports a t-statistic for a regressor the fixed effects annihilate

**`silent-wrong-answer`. Verdict: this is round-2 finding 3's new headline
evidence,** not a separate finding — same guard, same line
(`crates/tsecon-panel/src/fe.rs:262`), different trigger, one fix. But round 2's
trigger returns `±inf` and this one returns publishable numbers.

**Observed.** A regressor constant *within* every entity — a firm's founding
year, a country's land area — is zeroed exactly by the within transformation
(`max|Qx| = 4.441e-16` against a raw scale of 1.4139). `panel_fe` fits it to that
residue and returns a coefficient, an SE and a t.

**The invariance test settles it.** Adding a constant to an absorbed regressor
must change nothing, since the fixed effects absorb it:

```
+0      t_dead = -0.0180    beta_live = -0.889607134419
+1      t_dead = +0.1675    beta_live = -0.889607134419
+10     t_dead = +0.3647    beta_live = -0.889607134419
+100    t_dead = +2.1258    beta_live = -0.889607134419
```

`beta_live` is bit-identical to 12 digits — the estimator is otherwise correct —
while `t_dead` reaches nominal significance at 5% purely from adding 100. **A
statistic that moves when the data does not is not a statistic about the data.**

**The guard is adversarially selective**, reproduced on two independent draws:

| entity-level covariate | result |
|---|---|
| founding year (int), dummy, categorical, all-3.0, halves | **raised** (5/5) |
| log land area, latitude, GDP p.c., share ∈ [0,1], N(0,1) | **RETURNED** (5/5) |

`share ∈ [0,1]` came back at **t = −2.267**. It protects you from the toy cases
and fails on exactly the covariates applied panel work uses.

**Mechanism.** `fe.rs:260-262` is `xtx.llt(Side::Lower)` — a **positive-definiteness
test, not a rank test**. numpy's Cholesky succeeds on this trigger and on round
2's at rank 1 of 2. It fires only when the residue is bit-exactly zero, which is
why exactly-representable entity constants raise and ordinary doubles do not.

**Only the default `se_type` launders it**, and the arithmetic is exact: bread
`(X̃'X̃)⁻¹[dead,dead] = 4.58e27` × cluster meat `3.54e-59` = `7.40e-04`. The two
blow-ups cancel, so the cluster t is scale-**equivariant** in the dead column and
1e-16 residue yields an O(1) t. `nonrobust` and `driscoll_kraay` do not cancel
and honestly report `bse ≈ 7e13`, `t = 0.000`. Default is `se_type="cluster"`
(`bindings/python/src/lib.rs:3559`).

**Knife-edge, not graceful degradation.** At `eps = 0` the cluster t is +0.466;
at every `eps` from 1e-14 to 1.0 it is **+2.244, stable across 14 decades**.

**Expected.** `panel_fe.__doc__`: *"Matches linearmodels PanelOLS conventions."*
linearmodels 7.0 raises `AbsorbingEffectError` — a dedicated exception class that
names the offending variable and offers `drop_absorbed=True` — at every
`cov_type`. The library's own doc comment at `fe.rs:129` names this trigger as
*the* example of what the guard exists to catch: *"`SingularDesign` if the
within-transformed design is collinear (e.g. a regressor constant within every
entity)."* That is **not** documentation of the silent path — a user sees that
error only when the guard fires, i.e. only when nothing is wrong. It is proof of
intent, which makes the silent path worse.

**The guard has no test.** `grep -rn "SingularDesign\|singular"` across
`crates/tsecon-panel/tests/` and `bindings/python/tests/test_panel_fceval.py`
returns nothing.

**Rates.** 600/600 returns across six (N,T) shapes and 780/780 across a T sweep;
**19.2% nominally significant at 5%**, 11.0% at 1%, max |t| = 25.79.

**Boundary, honestly.** At k=1 linearmodels also returns garbage — its rank check
is relative and has no scale reference with one column — so the finding holds
only at **k ≥ 2**: an entity characteristic included *alongside* time-varying
controls. Mitigation: the live coefficients are uncontaminated (max shift
5.55e-16). That does not help the user who included the entity characteristic
*because that was the coefficient they wanted*.

---

## 2 — `flp`'s standard errors condition on the estimated eigenfunctions

**`trap`** (top of the band). Generated-regressor problem: `flp` regresses on
FPCA scores estimated with `O_p(T^{-1/2})` error, and the HAC sandwich treats
them as fixed.

**Observed** — internal control, same function, same draws, same code path, only
the scores differ:

| arm | T=200 | T=800 | T=3200 |
|---|---|---|---|
| true scores handed in — `se/sd` | 0.989 | 1.005 | 1.009 |
| true scores — cov95 | 0.943 | 0.951 | 0.948 |
| **FPCA scores — `se/sd`** | **0.675** | **0.664** | **0.674** |
| **FPCA scores — cov95** | **0.811** | **0.801** | **0.810** |

`mean_se` agrees to three digits between arms; only the true sd differs.
`|bias|/sd ≤ 0.096`. Flat over a 16× range of T ⇒ inconsistency, not a small
sample.

**On the model card's own worked example** (`functional-shocks.md:248-321`,
n=400) against *population* eigenfunctions: β₁ `se/sd` **0.854**, β₂ `se/sd`
**0.212** — one-fifth of the truth, at the card's own sample size, on the column
the card prints (`se[:5, 0]` at `:284`).

**Closed form, which makes it a proof rather than a measurement.** PCA
perturbation gives
`se/sd = sqrt(V_OLS / (V_OLS + Σ_{j≠k} λ_kλ_j/(λ_k−λ_j)²·β_j²/T))`, predicting
0.675 / 0.679 / 0.677 at T = 200 / 800 / 3200 against measured 0.671 / 0.664 /
0.667.

**Mechanism swept two ways.** With all responses equal there is nothing for the
eigenvector rotation to mix and coverage sits at nominal (0.952); as
signal-to-noise falls, `se/sd` degrades to 0.247 — **worst exactly where the
finding is most significant.**

**Expected.** `functional-shocks.md:117-119` — *"`se` its per-element standard
errors"* — unqualified, and `:271-284` runs precisely
`functional_pca → scores → flp` and prints `betas` and `se` side by side. **No
page anywhere warns that FPCA scores are generated regressors.** The library's
own house style discloses this exact hazard in warning boxes for two
structurally identical two-step estimators — `docs/guide/07-multivariate.md:513`
(FAVAR: *"The factors are generated regressors … bands that condition on F̂ as if
it were data are too narrow"*) and `docs/guide/15-term-structure.md:146`
(dynamic Nelson-Siegel). The functional-shock family has neither.

**Why the kept statsmodels promise does not exculpate.** `flp.__doc__` promises
*"Matches statsmodels OLS(...).fit(cov_type='HAC') per horizon at 1e-8"* and that
**is** kept (max |Δse| 3.5e-16). But the `ols(hac, maxlags=0)` refutation worked
because statsmodels was asked the *same question*; here statsmodels never saw a
first stage, so it cannot be the authority on whether the first stage should
enter.

**Scope correction, which caps severity.** The finder claimed the guide's
headline scenario `delta = np.ones(M)` does not recover. **It does** —
`flp_scenario` gives `se/sd` 0.986–1.006 and cov95 0.944–0.952 at every horizon,
identical to the true-factor control. The finder had mislabelled their own arm.
The immunity is algebraic: under an arbitrary basis rotation with the weights
rotated to match, raw `beta_1` moves **194.5%** and `se_1` **102.3%**, while
`w'beta` agrees to 1.3e-15 and `sqrt(w'Cov w)` to 1.1e-16 — because
`Φ̂ = RΦ ⇒ ŵ'β̂ = w'β`. The card states the underlying fact at `:70-71`: *"their
span is fine, their labels are not."*

So the surviving claim: **`flp`'s per-element `se` (and the diagonal of `covs`)
is inconsistent for `functional_pca` scores, the card's own printed example
understates it, and the docs disclose the identical hazard for FAVAR and DNS but
not here.** `flp` is also documented for externally supplied scores, where the
`se` is correct.

---

## 3 — Diagnostics that misattribute their own cause

**`cosmetic`**, grouped because they are one shape — the same shape as round 2's
*"`panel_pmg` blames the panel for a floating-point failure"*.

**`gmm_nonlinear` blames `initial` for a fault in the moment function.** With a
valid `initial=array([0.5])` and a `moments_fn` returning a 1-D array, the error
names `arg1` (= `initial`, correct as passed) and tells the user to reshape it,
never mentioning `moments_fn`. Cause:
`bindings/python/python/tsecon/_coerce.py:197-218` — `_rank_error` enumerates
only `args`/`kwargs`, and the `_RANK_HINT = "is not an instance of"` trigger at
`:194` also matches a callback-return coercion error. **Following the advice
makes it worse**: reshaping `initial` as instructed yields a second, more opaque
`TypeError`. The real fix is `.reshape(-1, 1)` on the moment function's *return*.
Blast radius is one function — `gmm_nonlinear` is the only public
callback-taking callable — and the native Rust message on the adjacent route is
excellent.

**`long_memory_d`'s runtime docstring is stale relative to its own round-1 fix**,
and this is the surface the brief calls authoritative:

| surface | says |
|---|---|
| runtime `__doc__` | *"Returns the estimate `d` and its **asymptotic** `se`."* |
| actually returns | `d`, `m`, `se`, `se_asymptotic`, `se_regression` (GPH) |
| `long-memory.md:124-126` | `se` = "at the bandwidth actually used"; `se_asymptotic` = "textbook large-`m`", **materially too narrow** |

The docstring omits three of five returned keys and calls `se` "asymptotic" —
the exact label the card attaches to the quantity round 1 found was ~25% too
narrow. The fix landed in the code and the card and missed `__doc__`.

---

# Refuted

- **`cg_regression`'s intercept / `forecast_efficiency`'s Wald size.** Numbers
  reproduce (intercept cov95 0.860 vs slope 0.891 at n=100; MZ Wald size 0.812).
  Four things kill it. (a) **The finder's own defense fails on their own DGP**:
  they computed the score autocovariance as `0.3^k` — the *slope's* score — while
  the intercept's is `u_t` itself with `γ(k) = 0.6^k`, geometric and never zero,
  so the bandwidth does **not** cover and this is literally the disclosed failure
  mode at `expectations.md:63`. (b) The whole gap is the **Bartlett kernel in
  closed form**. (c) It **vanishes in n**, monotone, because
  `L = ⌊4(n/100)^{2/9}⌋` grows — the opposite of the `lp(cumulative="both")`
  signature. (d) An **oracle arm proves the code is right**: the same `params`
  through the true asymptotic covariance give Wald size 0.946/0.948/0.955. And it
  is disclosed in four places, including a table indexed by score autocorrelation
  at `docs/examples/interval-coverage.md:632-645` whose band contains both
  measured numbers.
- **`zero_sign_svar(weighted=)` bit-identical.** Refuted on the library's own
  documented theory (`crates/tsecon-ident/src/zero.rs:293-305`): for impact-only
  zeros the restriction functions are linear in `Q`, the volume element is
  `Q`-independent, and the ARW weight is **exactly one**. All probed zeros were at
  horizon 0. The round-1 fix is live where it matters — a zero at horizon ≥ 1
  **raises** rather than substituting 1.0.
- **`growth_at_risk(rearrange=)` bit-identical.** A no-op precisely when the
  fitted quantile curve does not cross, which is what rearrangement means.
- **`var_forecast(band_scope=)` bit-identical.** Inert only at
  `band="pointwise"`, where a simultaneity scope has no meaning; it moves the
  critical value on every simultaneous band (sup-t 2.694654 → 3.044305).
- **`factor_model`'s Bai-Ng criteria returning the ceiling.** `icp`/`pcp` return
  `kmax` on small-N panels where the eigenvalue-ratio `er` gets it right — but
  the criteria do select correctly at larger N (r=5 → 5), `kmax` is an exposed
  parameter defaulting to 8, and PCp's `kmax`-dependence is Bai-Ng (2002) by
  construction, since the penalty uses `σ̂²(kmax)`. Overselection at small N is a
  textbook property, not a defect.

---

# Swept and found sound

- **Lens 4 (discarded computation) is essentially clean.** Its mechanical half —
  call each function on 8 independent datasets and flag every returned quantity
  that comes back bit-identical — reached **64 of 64** functions with **0**
  template failures and surfaced 124 constant leaves. On triage all but one are
  legitimately constant: tabulated critical values (`phillips_perron`,
  `phillips_ouliaris`, `engle_granger`, `johansen`), deterministic bandwidths
  (`long_memory_d`'s `m = ⌊√n⌋`), CUSUM bounds (a function of T and k), echoed
  input flags, sample sizes, `converged` flags that were all true, zero failure
  counters, and IVX's `rz = 1 − n^{-cz}`. The one real item is
  `long_memory_d`'s stale docstring, above.
- **`long_memory_d` is sound — the round-1 discarded-SE fix works.** The harness
  was validated by reproducing the model card's published numbers exactly
  (0.954/0.951/0.948 against the card's *"94–96%"*), then measured the
  previously unmeasured `d ≥ 0.5` region: GPH sits at nominal across
  `d ∈ [0, 0.9]` (cov95 0.938–0.954). Local Whittle's 5–9% narrowness is real and
  exactly as the card discloses. New observation: LW's documented box pins
  `d̂ = 0.999999985` in up to 30% of draws at `d = 0.9`, and since the reported
  `se` is a deterministic function of `m`, coverage turns non-monotone in α
  (cov68 0.641, cov95 0.967 at d=0.8, n=1024).
- **SVAR restriction validation** — 36/52 malformed inputs raise, each naming the
  offending index. Unsatisfiable sets return `accepted: 0` with **all-NaN**
  quantiles rather than laundering them.
- **`robust_svar_bounds` diverging from its siblings is correct** — it uses an
  exact active-set optimizer, not rotation sampling, so it can represent a
  measure-zero non-empty set a sampler can never hit.
- **`lp`'s collinearity guard is monotone** — the direct sibling of finding 1,
  tested for the same pathology and clean.
- **Argument-axis scale equivariance holds** — `lp` irf/se equivariant to 6e-15
  over `c ∈ [1e-150, 1e8]` when scaling the *shock* alone; `lp_iv` invariant to
  instrument scale to 1e-14.
- **Thirteen previously-unreached functions are scale-clean** across nine
  decades: `afns_adjustment`, `dsge_solve`, `functional_pca`, `flp`,
  `flp_scenario`, `favar`, `max_share_svar`, `historical_decomposition`,
  `panel_mean_group` (mg and cce), `ivx_test`, `mcmc_diagnostics`,
  `phillips_ouliaris`.
- **Six previously-unswept seed cases all move with the seed** — `bvar_ssvs`,
  `sign_restricted_svar`, `zero_sign_svar`, `narrative_svar`, `fry_pagan_svar`,
  `robust_svar_bounds`.

---

# Incidental

- `docs/examples/interval-coverage.md:1063`'s "what is not measured" list omits
  `flp` / `flp_scenario`.
- These surfaces ship **no interval of any kind**, so there is nothing for a
  coverage round to measure: `nelson_siegel`, `svensson`, `dynamic_ns`,
  `weighted_midas`, `favar`, `bvar_fit`, `structural_fevd`.
- `iv_gmm`'s positional argument order is `(x, z, y)`, not `(y, x, z)` — an easy
  silent misuse, since all three are float arrays of compatible shape. Worth a
  keyword-only signature or a docstring warning.

---

## Reproducing

Probe scripts were scratchpad-only (an audit is read-only), so every finding
above carries a self-contained reproducer inline. The four harnesses worth
rebuilding are described in
[round 2's report](17-audit-round-2-findings.md#reproducing-this-audit); rounds
3–4 add two more:

- **The coverage checker** — for each (function, axis), count how many
  comparisons were actually *made*, not attempted. Nine lines of logic; it found
  a 27% hole in work already written up as complete.
- **The constant-diagnostic detector** — call each function on K independent
  datasets, flatten every returned leaf, and flag those bit-identical across all
  K. Classify rather than judge: echoed inputs, tabulated critical values,
  dimensions and deterministic functions of *n* are all legitimately constant, so
  the probe should partition into benign-by-name and candidates rather than
  reporting a raw count.
