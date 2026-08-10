# Adversarial audit, round 2 — findings

> **Working document, not a description of a completed thing.** It records what
> the second adversarial audit found, what it *failed* to find and why, and the
> method corrections that came out of it. It lives in `docs/roadmap/` and is
> excluded from the published site. Its companion is
> [the brief](16-adversarial-audit-brief.md), which tells you how to run the
> next one.

Run against `feat/simultaneous-bands` at `f4e622c`, `tsecon 0.2.0`, extension
verified fresh (zero `.rs` newer than `_core.abi3.so`). **Read-only throughout —
`git status` clean at start and end.** No cargo was run at any point: every probe
drives the already-built Python surface, which sidesteps the workspace-cargo
stall hazard rather than merely obeying it.

Method per [the brief](16-adversarial-audit-brief.md): seven lenses, each swept
by an independent finder, every finding then sent to an independent agent whose
job was to **refute** it, defaulting to refuted unless it reproduced the numbers
itself.

**Scoreboard: 21 candidate findings raised, 8 survived refutation.** Refutation
killed 13 — including 3 of the orchestrator's own 4 — narrowed 2 to a smaller
true claim, and *strengthened* 2 beyond what their finders claimed.

---

## Read this first: the method correction

**Three of my four lens-1 findings died from one systematic mistake — I
established "what was promised" from `bindings/python/python/tsecon/__init__.pyi`
when the authoritative text is the Rust doc comment that `help()` shows.** The
`.pyi` is a thin type stub. In all three cases the runtime docstring and the
`docs/reference/model-cards/` page said the opposite of what I claimed:

| I read in the `.pyi` | What `__doc__` and the model card actually say |
|---|---|
| `link` is "probit" or "logit" | *"probit only"* + `recession.md:58` *"`link` is ignored"* |
| `lrv_kernel` listed, unqualified | *"configure the **LLC** long-run variance"* + `panel-unit-root.md:99` *"ignored by IPS/Fisher"* |
| `seed` accepted on asymptotic path | `n_boot` returns **`None`** to announce it was ignored |

**Rule for the next round: establish every promise from
`print(tsecon.<fn>.__doc__)` and `docs/reference/model-cards/`. Treat the `.pyi`
and the Rust crate docs as non-promises — a Python user sees neither.** This one
check would have saved three false positives.

The rule cuts both ways, and the round hit the *other* side of it three times:
`bvar_ssvs`'s scale-invariance claim lives only in a Rust module doc and never on
the Python surface; the `lp(cumulative="both")` finding was first cited to a Rust
crate doc whose own text scoped it to the level case, and had to be re-cited to
the model card and guide, which say it unqualified; and `dfm_nowcast`'s
interior-NaN truncation *looked* silent until the guide turned out to document it
in a dedicated bullet. **Both a false positive and a false negative live in the
gap between these four doc surfaces.**

A second premise died with them: **this library does not have a convention of
refusing inert arguments.** A 64-trial survey across 24 functions found **7 raise
vs ~42 silently ignore** — silent-ignore is the norm by ~6:1, and many are
explicitly documented as such. The four raise sites are recent, targeted fixes
from the *previous* audit, applied where a caller would otherwise receive a
**different estimator**, not where output is merely unaffected.

---

# Confirmed findings, ranked by what a user would experience

## 1 — `lp(cumulative="both")` reports an inconsistent standard error

**`silent-wrong-answer`.** The most serious finding of the round, and exactly the
class the brief predicts lens 7 produces.

**Observed.** I reproduced this independently of the finder, different seed
(`[424242, 7, rep]`), 400 reps, on a DGP with no propagation at all
(`y_t = s_t + η_t`, so the truth is exactly 1 at every horizon):

| h | true sd | mean reported se | se/sd | cov @ nominal 95% |
|---:|---:|---:|---:|---:|
| 0 | 0.0240 | 0.0224 | 0.932 | 0.930 |
| 4 | 0.0416 | 0.0223 | 0.535 | 0.743 |
| 8 | 0.0555 | 0.0224 | 0.403 | 0.590 |
| 12 | 0.0656 | 0.0224 | 0.341 | **0.507** |

The reported SE is **flat across horizons** — 0.02225 to 0.02238, a **0.55%
spread** — while the true sampling sd grows **2.73×**. A nominal 95% interval
covers **0.507** at h=12. On the finder's richer DGP: 0.683 ± 0.007 at T=200 and
0.717 ± 0.007 at T=800, so **quadrupling T barely moves it — this is
inconsistency, not a small-sample effect.** Bias is negligible throughout
(|bias|/sd ≤ 0.063); the point estimate is fine and the standard error is wrong.
`se="hac"` on the identical draws recovers to 0.950 (finder) / 0.900 (mine).

**Mechanism.** `crates/tsecon-lp/src/level.rs:84` augments the horizon-h
regression with `h` lags of the impulse — *past* shocks. Under
`Cumulation::Both` the regressor is `Σ_{j=0}^{h} shock_{t+j}`, so two base times
`k ≤ h` apart share `shock_t … shock_{t+h−k}`: a **future**-shock overlap that no
past-lag augmentation can project out. The score is therefore not a martingale
difference sequence and HC1 is inconsistent for every h ≥ 1. The defect is
exactly zero at h=0 (se/sd = 0.932) and grows with h, as the mechanism predicts.

**The shortfall matches a closed form**, which turns this from a measurement
into a proof. With score `S_h(t)·U_h(t)`, `Cov(score_t, score_{t−k}) = (h+1−k)²`,
so `sd/se` should equal `sqrt([(h+1)² + 2Σ_{k=1}^{h}(h+1−k)²] / (h+1)²)`.
Measured against predicted at T=1600: h=1 **1.222/1.225**, h=4 **1.865/1.844**,
h=8 **2.501/2.457**, h=12 **3.003/2.948**. The gap is exactly the omitted
autocovariance terms.

**Decisive control, same draws:** `cumulative=None` covers 0.938–0.966 and
`cumulative="outcome"` covers 0.918–0.962 at the same horizons under the same
default SE. **Only `"both"` collapses.** The library already ships the fix —
`se="hac"` restores 0.902–0.956 at T=1600 and improves in T — it is just not the
default for this mode.

**Expected — citation corrected.** My original citation
(`crates/tsecon-lp/src/lib.rs:68-91`) was wrong in the same way my lens-1
findings were: it is a Rust *crate* doc a Python user never sees, and its own
section defines the score as the **level** score `s_t = shock_t · u_{t,h}`, so it
is defensible as scoped to the level case. The user-facing promise is
`docs/reference/model-cards/local-projections.md:36-42` — `se="lag_augmented"` is
*"the **default and the recommendation**… inference-robust even under
persistence"* — with *"`cumulative` as above"*, covering `"both"`, one sentence
later; and `docs/guide/09-local-projections.md:150`. Both unqualified.
**No doc anywhere pairs `cumulative="both"` with a standard-error warning**
(verified by grep across `docs/`, `crates/tsecon-lp/src/`, `bindings/python/`).
The nearest thing is `docs/examples/interval-coverage.md:1058` — *"`lp(cumulative=...)`
intervals are unmeasured"* — a disclosure of ignorance on a different page, which
does not stop the model card recommending the default.

**Blast radius: the simultaneous band is narrower than a correct pointwise one.**
All four `band=` routes reuse the bit-identical `se`, so at h=12 a **sup-t band
has 0.0757 half-width against a true sd of 0.0758** — a *joint* statement
narrower than an honest *pointwise* interval.

**Why the suite is green.** The crate's only coverage test,
`crates/tsecon-lp/tests/properties.rs:89`, runs `LpSpec::new(hmax, 4)` =
`Cumulation::None`. The one Python test that touches this path
(`bindings/python/tests/test_lp_multiplier.py:149-161`) asserts that point
estimates differ and never reads `se`.

**Scope — I checked the siblings.** `lp_iv(cumulative="both")` does **not**
inherit it: its reported SE grows with horizon (232.8% spread over h=0..10) and
`se/sd` holds at 0.928–0.949. So the flagship
`docs/examples/replication-ramey-zubairy.md:116` route, which recommends
`lp_iv(..., cumulative="both")`, is sound. The defect is specific to `lp`.
`lp_state(cumulative="both")` **does** share it, now measured rather than
guessed: se/sd 0.466, cov95 **0.640 ± 0.028** at h=12, T=800.

**Reproducer.** `.../audit2/repro_both_trivial.py`; the check above is
self-contained in this report's transcript.

---

## 2 — `bvar_ssvs` is not scale-invariant, and posterior inclusion flips

**`silent-wrong-answer`**, plus an **`overclaim`** on one Rust doc line.

**Where.** `crates/tsecon-bayes/src/ssvs.rs:136-137` (`gamma_a: 0.01,
gamma_b: 0.01`), used at `:356`/`:379` as `rate = gamma_b + 0.5 * s_jj`.
`gamma_b` is an **absolute** Gamma rate on an error *precision* in units of
`1/y²`, while `s_jj` carries `y²`.

**Observed.** Same data, same seed, only the units of `y` change. Percent →
decimal (`c = 1e−2`) is the single most common unit convention change in macro:

| c | `sigma_mean[0]/c²` (must be constant) | mean inclusion prob |
|---|---|---|
| 1e−4 | **8455.6** (17,000× too large) | 0.6443 |
| 1e−2 | **1.3441** (176% too large) | 0.6738 |
| 1 | 0.48711 | 0.8242 |
| 1e4 | 0.48702 | 0.8242 |

Invariance holds bit-stably *upward* over four decades and breaks *downward*.
Per-coefficient posterior inclusion probability moves by up to **0.517**, and
**two coefficients cross 0.5** — the variable-selection decision itself flips.

**The MC-noise explanation was tested and killed.** The refuter ran a
control-vs-treatment design at `n_draws=20000`: varying the *seed* at fixed `c`
moves inclusion by at most **0.054**; varying `c` at fixed seed moves it
**0.390–0.445 on all six seeds, with 2 selection flips every time**.
`sigma_mean[0]/c²` is 0.4863 ± 0.0003 at c=1 against 1.3416 ± 0.0004 at c=1e−2 —
a 2.76× shift at roughly 2000 seed-sd.

**Mechanism confirmed arithmetically.** `shape = 0.01 + T/2 = 119.51`; at
c=1e−2 the prior supplies **64.2%** of the rate (predicting 1.3037 vs observed
1.3416), and at c=1e−4 it supplies **100.0%** (predicting 8368 vs observed 8456).
Setting `gamma_b = 0.01·c²` restores invariance and drops max |Δ inclusion| from
0.517 to 0.041.

**The bias propagates to the outputs users read:** at c=1e−2, median `irf_draws`
is **+65% at h=1 and +40% at h=4**; at c=1e−3, `coef_mean[1,0]` is **−82%**.
`mean_model_size` slides 6.85 → 5.24 → 4.17.

**Expected — with a correction to the finder.** `ssvs.rs:36` says *"so the
estimator is scale-invariant and needs no data transformation."* That is a **Rust
module doc only**; the runtime docstring, the `.pyi:408`, and
`docs/reference/model-cards/bayesian.md:165-167` never claim it — the model card
makes the narrower and *true* claim that the spike/slab dials are scale-free
*across variables in different units*. So the finding stands on the numbers, and
`ssvs.rs:36` is a separate `overclaim`. `gamma_b` is a public kwarg, so the
remedy exists but is undocumented.

---

## 3 — `panel_fe` accepts a rank-deficient design its own guard exists to reject

**`silent-wrong-answer`.**

**Observed.** With two exactly duplicated regressors, `panel_fe` returns
`params=[1.15e14, -1.15e14]`, `bse=[0.2503, 0.0]`, `tvalues=[4.60e14, -inf]`.

**The guard is a coin flip, and it is anti-monotone in the severity of the
collinearity.** `crates/tsecon-panel/src/fe.rs:262` does
`xtx.llt(Side::Lower).map_err(|_| PanelError::SingularDesign{...})` — the intent
to reject is explicit. Over 200 seeds the refuter measured it catching **105 and
passing 95**. I reproduced the anti-monotonicity independently, sweeping
`x1 = x0 + eps·noise`:

| eps | 1e−7 | 1e−9 | 1e−11 | 1e−13 | 1e−15 | 0 |
|---|---|---|---|---|---|---|
| result | accepted | **REJECTED** | **REJECTED** | accepted | accepted | accepted |

It rejects the mild case and accepts exact duplication.

**Expected.** `panel_fe.__doc__`: *"Matches linearmodels PanelOLS conventions."*
The refuter verified that match is otherwise excellent — across four (N,T,k)
shapes and both `nonrobust` and `cluster`, rel |Δparams| ≤ **5.7e−16** and
rel |Δbse| ≤ **5.1e−16**. And linearmodels' refusal here is a *deliberate* rank
check — `ValueError: exog does not have full column rank`, on three seeds — not
an incidental crash. This is the inverse of the refuted `ols(hac, maxlags=0)`
case: there the reference did the same thing; here the reference refuses and
tsecon answers.

**One sub-claim I could not confirm.** The refuter additionally reports that
`fe.rs:265`'s `xd.qr().solve_lstsq()` — a QR *without* column pivoting — returns
a point that is off the least-squares manifold entirely (RSS excess +0.80%,
identified `b₀+b₁` wrong by 4.8%). My independent reconstruction gave mixed
signs (+2.05%, +0.28%, **−0.19%**, +2.06%), and a negative excess means my
within-transform is not bit-identical to tsecon's. **Treat "off the LS manifold"
as plausible and unverified**; the accept-exact-duplication behaviour and the
anti-monotone guard are firmly confirmed by both of us.

---

## 4 — `ols(se_type="hac")` misses its own documented statsmodels match at the default

**`trap`**, with a separate **`overclaim`** underneath it that is the real cause.

**Observed.** With `maxlags` matched, each library at its own default:

| T | max rel \|Δbse\| | at matched settings | √(n/(n−k))−1 |
|---:|---:|---:|---:|
| 25 | 6.600358e−02 | 2.0e−15 | 6.600358e−02 |
| 240 | 6.309211e−03 | 2.9e−15 | 6.309211e−03 |
| 5000 | 3.001351e−04 | 1.3e−14 | 3.001351e−04 |

The gap is *exactly* the small-sample factor and nothing else. At matched
settings agreement is ~2e−15, five orders better than promised — the estimator is
right, only the default differs.

**Expected.** `tsecon.ols.__doc__`: *"HAC results match statsmodels
`cov_type="HAC"` at 1e-10."* Repeated unqualified at
`docs/guide/03-inference-toolkit.md:587` and `docs/examples/README.md:89`. And
`docs/migration/from-statsmodels.md:84` maps
`.fit(cov_type="HAC", cov_kwds={"maxlags": L})` → `ols(..., se_type="hac", maxlags=L)`
as *the* migration equivalent — precisely the call pair that disagrees by 6.6%.

**The sharper defect underneath.** tsecon defaults `use_correction=True`
(`lib.rs:705`); statsmodels `cov_type="HAC"` defaults `False` (confirmed in
`statsmodels/base/covtype.py`, 0.14.6). The library states the reference's
default **backwards**:
- `crates/tsecon-hac/src/ols.rs:118` — *"statsmodels `use_correction`, default `true` there"*
- `docs/reference/model-cards/expectations.md:47` — *"leave it on to match statsmodels' default"*

So the `True` default was chosen *in order to* match statsmodels and matches its
opposite. **Fairness note:** `statsmodels.stats.sandwich_covariance.cov_hac_simple`
really does default `True` — statsmodels is inconsistent between its two APIs —
so `ols.rs:118` is defensible if "there" means the helper. It is not defensible
for `cov_type="HAC"`, which is what both tsecon docstrings name.

**Blast radius.** `cg_regression` at its defaults differs from statsmodels HAC by
**8.44e−03** and matches at `use_correction=True` to 1.6e−15. Meanwhile `har_rv`
defaults `False` (`lib.rs:4150`) and *does* match at its default to 4.9e−14. One
library, one reference, opposite defaults, both claiming a match.

**Why the green suite cannot see it.** `test_smoke.py:168` passes
`use_correction=case["use_correction"]`; `fixtures/generate_fixtures.py:339-353`
feeds the same value to statsmodels. Matched settings, legitimately 1e-10.
Nothing tests default-to-default. `validation-matrix.md:138-139` is likewise
about matched settings and is **not** an overclaim. Partial mitigation exists at
`docs/cookbook/hac-standard-errors.md` (*"packages differ on this default"*) but
it never says which way either defaults and does not qualify "1e-10".

---

## 5 — The published interval-coverage tables silently drop a harvested row

**`trap`.** `docs/examples/interval-coverage.md:191-235`, `:237-282`, `:308-338`.

**Expected**, `docs/reference/testing.md:343-346` verbatim: *"`run_all.py`
harvests the consolidated tables from the structured results the modules return —
**nothing is transcribed, so a schema change makes the runner exit non-zero
rather than silently dropping a row.**"*

**Observed.** `run_all.py` (447.2 s, reps=3000, exit 0) emits **40 / 40 / 32 /
24** rows; the page publishes **39 / 39 / 31 / 23**. I verified the drop by
counting: Table 1 has **four** `tsecon.ols` rows where the runner emits five. The
missing one is `se_type="hc3"; small T, leverage` → `0.863 ± 0.006, UNDER`.
`0.863` survives only as a parenthetical *inside the `hc1` row's* prose
(`:320`), with a different cause attributed (`ESTIMATOR`, not the runner's
`APPROXIMATION`).

**Why it matters more than a missing row usually would:** the dropped row is the
**`hc3` estimator `0.2.0` added because of this very audit** — the one row a
reader following the "Fixed in 0.2.0" table would go looking for. The arithmetic
corroborates: `:10` and `:183` claim "**40** interval-valued outputs" while
`testing.md:355-357` reports 31 + 7 + 1 = **39**.

---

## 6 — `panel_lp(jackknife=True)` costs 8pp of coverage, and the cookbook recommends it exactly where it costs most

**`trap`.** This is what survived of a larger Driscoll-Kraay claim that was
otherwise refuted (see below). N=20, T=60, h=8:

| | bias | \|bias\|/sd | sd | reported se | se/sd | cov 95% |
|---|---|---|---|---|---|---|
| default | −0.0701 | 0.299 | 0.2346 | 0.2121 | 0.904 | 0.880 |
| `jackknife=True` | **+0.0048** | **0.015** | **0.3191** | 0.2121 | 0.665 | **0.804** |

The Dhaene-Jochmans correction works exactly as advertised on the point estimate
— bias essentially eliminated — but it inflates the estimator's variance **36%**
while the reported `se` is **bit-identical**, because
`crates/tsecon-panel/src/lp.rs:53-57` deliberately keeps the uncorrected SE
citing DJ Thm 3.1's asymptotic equivalence. At T=60 that equivalence has not
arrived; it does by T=240 (0.944 → 0.927).

**Why it is a trap rather than a design note.** The Rust doc discloses the
choice, but the Python user sees a one-line docstring plus
`docs/cookbook/panel-lp-standard-errors.md:90-92` recommending the jackknife
*"when `T` is short"* — precisely the regime where it costs 8 points — with no
standard-error caveat.

---

## 7 — `smooth_lp`'s default CV λ grid is absolute

**`trap`.** `crates/tsecon-lp/src/smooth.rs:690` `default_grid()` is a fixed
`1e-2 … 1e6` ladder compared against `A0 + λP` where `A0 = X'X` carries data
units squared.

**Observed, with the finder's headline corrected.** Rescaling only the shock by
×100 (percentage points → basis points) changes the unit-normalized IRF by
**33.92%**, not the 122.54% originally claimed — that figure appears nowhere in
the finder's own artifacts and does not reproduce. `lambda_used` pins at the
grid maximum 1e6.

**Correction that narrows it usefully:** over `c ∈ [1e−2, 1e1]` the IRF is
**bit-for-bit invariant** (max |Δ| = 5.8e−16) because CV correctly tracks
λ\* ∝ c² across the grid. **The defect exists only at the grid endpoints**, so
percent↔decimal on both series is safe. Passing `lambda_grid = default·c²`
restores invariance exactly — the estimator is fine, only the fixed grid is not.

**Partially disclosed**, which caps severity at `trap`:
`docs/reference/model-cards/local-projections.md` documents the grid and says *"a
`lambda_used` at the top of the grid means 'as smooth as allowed' — extend
`lambda_grid` if that worries you"*, and `lambda_used`/`cv_grid`/`cv_scores` are
all returned. The residue: nothing says λ\* moves with the *variance* of the
data, though the library's house style does exactly that elsewhere
(`machine-learning.md:46` on `lasso`: *"center `y` and standardize the columns of
`X` yourself … the most important line on this card"*). And that same card's
worked example lands at **grid index 15 of 16** while its prose says "CV lands
mid-grid".

---

## 8 — Documentation drift

**`overclaim`/`cosmetic`.** All measured.

- **Two pages still say a simultaneous band does not exist.**
  `docs/reference/testing.md:374` — *"No function in the library reports a
  simultaneous (sup-t) band."* `docs/reference/model-cards/structural-identification.md:393`
  — *"No simultaneous band exists anywhere in this library"*, thirteen lines
  after the same card recommends `proxy_ar_sets`. `var_irf_bands`,
  `var_forecast`, `lp`, `smooth_lp` all take `band="sup-t"` today. Same class:
  `docs/examples/coverage/lp_family.py:1051-1057` prints on every run that *"none
  of these functions reports one"* while `lp` and `smooth_lp`, measured in that
  very module, do.
- **Test counts wrong in three of four places.** Measured: **Python 652**
  (`pytest --collect-only`, confirmed), Rust ~1249. `README.md:25` says 562
  Python; `ROADMAP.md:13` says 538; `testing.md:638` says "the full **332-test**
  Python suite"; `testing.md:28` says 1235 Rust. Only `testing.md:32` (652) is
  right — on a page that opens *"Every number in this section was produced by a
  command run on this working tree."*
- **`docs/quickstart.md:412-413` ships stray harness markup** — literally
  `</content>` and `</invoke>`, leftover tool-call XML committed into the
  published on-ramp page. Verified by direct read.
- **`panel_pmg` blames the panel for a floating-point failure.**
  `crates/tsecon-panelts/src/error.rs:159-164` reports *"the panel may be too
  short or too weakly cointegrated"* when the real cause is that
  `TOL = 1e-12` (`pmg.rs:111`) is absolute on `theta`, so once `eps·‖θ‖∞ > 1e-12`
  the convergence test is unsatisfiable in double precision. Non-monotone in
  scale (9 raises / 9 successes over c ∈ 1e3…1e12); every *successful* run
  matches the c=1 answer to 1.8e−13, so there is no wrong number — the defect is
  the false diagnosis. Scaling y and x together never fails.
- **Missing shape preconditions panic instead of raising.**
  `crates/tsecon-coint/src/engle_granger.rs:202` and
  `bindings/python/src/lib.rs:2926` raise `pyo3_runtime.PanicException`, which
  subclasses `BaseException`, so a caller's `except Exception` will not catch it.
  Scope is narrow and was corrected downward: a sweep of every public callable
  taking 1–2 array arguments across five degenerate shapes found **531 clean
  exceptions, 11 clean returns, exactly 1 panic**. Not a systemic pattern — two
  bounds checks.
- **Smaller drift**, all in `docs/reference/testing.md` unless noted: `:515-517`
  says 36 of **49** files with **13** unlisted (actual **50**/**14**); `:525`
  claims 4 tests in `test_replication_ramey_zubairy.py` (actual **3**);
  `:41`/`:166` claim **392** property tests (actual **398**); `:711` claims
  benchmarks cover **25** of 128 functions (actual **23**); `:37-39`
  mis-describes the 7 ignored tests. `docs/reference/results.md:117-128` omits
  `CheckSeriesResults` and `CoefficientFrame`, both in `tsecon.results.__all__`,
  while `quickstart.md:368` calls that page the catalogue of "every wrapper".
  `dfm_nowcast`'s docstring promises `smoothed_factors` is `T x r` where it is
  the balanced block (195 of 200 rows in the *documented* ragged case), though
  `docs/guide/11-nowcasting.md:344` states the truth correctly.
  `__init__.pyi:1409` says `link` is "probit" or "logit" with no dynamic caveat,
  contradicting the runtime docstring — an IDE hover shows the stub.

---

# A methodological result worth keeping

**For a pure SE-scale miss, the percentage-point shortfall peaks between nominal
0.70 and 0.82 and is *smallest* at 95%.** Fitting each measured miss to
`coverage(a) = Φ(c·z_a − b) + Φ(c·z_a + b) − 1` with `c = mean_se/sd`,
`b = bias/sd` reproduces the measurements at 68% to within 0.01–0.03 in every
case. `lp(cumulative="both")` at T=800 h=12 loses 26.3 pp at 68%, 26.2 pp at 90%,
23.3 pp at 95%.

This is the **opposite shape** from the IRF sweep quoted in
`interval-coverage.md` (4.9 / 13.7 / 15.0 pp). So a 95%-only measurement
*understates* the worst shortfall for scale-type defects, and badly understates
the relative damage — at 68% nominal, `cumulative="both"` at T=800 delivers
0.417, losing **39%** of its promise versus 25% at 95%. Where the observed 90/95%
shortfall *exceeds* the scale-model prediction, the residual is a shape problem
(the reported SE's own randomness and non-normality), and those are the cases
where sweeping α is genuinely informative.

---

# Refuted — recorded so the next round does not re-derive them

- **`recession_probit(link=)` ignored on the dynamic path.** Documented: runtime
  docstring *"probit only"*, `recession.md:58` *"`link` is ignored"*, both
  predating this audit. Residue: `link="banana"` silently accepted there
  (validation lives only in the `else` branch, `lib.rs:5736`) — `cosmetic`.
- **`panel_unit_root(lrv_kernel=, lrv_bandwidth=)` inert on the default test.**
  Documented in `__doc__` and `panel-unit-root.md:99`. My "truncated raises on
  one path, no-op on another" framing was also wrong: the raise is a
  data-dependent numeric guard on the *estimated* LRV, not name validation — on a
  random-walk panel it returns cleanly (statistic −0.894976).
- **`var_irf_bands` accepts `seed`/`n_boot` on the asymptotic path.** The
  function reports in its return value that it ignored them (`n_boot` → `None`).
- **`gmm_nonlinear` returns the caller's start with `converged=True`.** The
  cleanest kill of the round. `neldermead.rs:26-27` promises *"Termination
  semantics match `scipy.optimize.minimize(method="Nelder-Mead")`"*; scipy at
  identical tolerances from identical starts returns **bit-identical** values
  (`3.6569824e-12`, `1.3931362e-11`) with `success=True`. Matching the reference
  is the promise and it is kept. scipy's own defaults are 10,000× looser.
- **`markov_switching_ar` floor-pinned variance with `converged=True`.**
  Textbook unbounded Gaussian-mixture likelihood; the floor is deliberate and
  documented at `model.rs:28-35`. statsmodels on the identical series returns
  `sigma2 = 2.55e-31` — **19 orders of magnitude** further into the spike. Decays
  from 4.33% of seeds at T=40 to **0.00%** at T=400.
- **`ccc_garch`/`dcc_garch` accept a singular correlation matrix.** The finder's
  ε-sweep was confounded (it moved the marginal variance along with ρ). Sweeping
  ρ properly gives a **strictly monotone** ladder, and ρ = 1−1e−12 — a
  legitimately positive-definite input — already returns +4580.5. A large
  positive loglik is the correct value of an unbounded Gaussian likelihood as
  |R| → 0, not a symptom. The "negative eigenvalue" is −7.95e−16 = 3.6 machine eps.
- **`panel_fe` at N=1 giving `bse ≈ 4.6e−16`.** linearmodels' `ZeroDivisionError`
  is its **poolability F-test** (`df_num = 0`), not input validation. Asked for
  the same estimand — 2 entities clustered into one group — linearmodels returns
  `bse = 6.84e−16` and statsmodels `1.50e−16`. Universal CRVE property at G=1.
- **`dfm_nowcast`'s interior-NaN truncation.** Documented twice: the runtime
  docstring and a dedicated bullet at `docs/guide/11-nowcasting.md:297`. Leading
  NaN is cleanly rejected. Only the `T x r` docstring parenthetical survives.
- **`ffbs.rs:72 DIFFUSE_COLLAPSE_TOL`** — my strongest pre-loaded lead, dead on
  two grounds I verified myself. `FfbsSampler` is **unreachable from Python** (no
  binding calls it), and the comment does not contradict the code: what the
  previous round made relative was the *element-level* `F_inf` test, while
  `filter.rs:113-122` argues at length that the period-level test is deliberately
  absolute because *"`P_inf` is not a variance"* — it is a dimensionless rank
  indicator.
- **`qreg.rs P_TOL`/`RESID_FLOOR = 1e-6`** — refuted on the `ols(hac, maxlags=0)`
  precedent, as predicted. Promise is *"statsmodels QuantReg, all defaults"*;
  over nine decades max |Δ| is **1.8e−15** at c=1e−8. statsmodels is equally
  scale-fragile and tsecon tracks it.
- **`max_share_svar(sign=)`** — bit-identical on 6 of 6 stable VARs, but the
  rules genuinely diverge on a sign-reversing VAR.
- **`panel_lp(jackknife=)` leaving `se` identical** — deliberate and cited,
  `lp.rs:56,117`, Dhaene & Jochmans (2015) Thm 3.1.
- **`proxy_svar_bands(robust_f=)`** — moves only the first-stage F diagnostic.
- **`ols(use_correction=)` inert off the HAC path** — correct by construction.
- **`panel_lp` Driscoll-Kraay under-coverage at N=100/T=60.** Every number
  reproduces (0.884 at N=100 vs 0.890 at N=5, so 20× the data buys nothing), but
  **the docs are ahead of the finding**:
  `docs/cookbook/panel-lp-standard-errors.md:53-55` is a whole page on exactly
  this — *"When the regressor is a single common shock … the effective sample
  size is closer to `T` than to `N × T`. Driscoll-Kraay is the estimator that
  knows this"* — and lines 42-47 document the `cluster` disaster, *"Nothing warns
  you"*, explicitly. The interval keeps its asymptotic promise (0.948 at T=400).
  Two sub-claims were also wrong: `bandwidth=0.0` is **not** an `iv_gmm` sibling
  (default is 4.0, negatives raise a typed error, and bandwidth=0 actually
  *covers better* here — 0.921 vs 0.880), and most of the T=60 miss is **Nickell
  bias in the point estimate** (|bias|/sd = 0.299 at se/sd = 0.904), documented
  at `crates/tsecon-panel/src/lp.rs:32-42`, not an SE defect. Only the
  `jackknife` sub-claim survived, as finding 6.
- **`quantile_lp`'s tail under-coverage.** Killed by the finder's own data: the
  τ=0.05 shortfall is **already present at h=0** (cov95 0.865, se/sd 0.808 —
  identical to the h=8 headline), and at h=0 `quantile_lp` *is* a quantile
  regression. So it is not an LP finding at all, and it is already published at
  the same magnitude — `docs/examples/interval-coverage.md:263` records
  `quantile_regression` τ=0.05 T=200 at **0.866 ± 0.006**, se/sd 0.818. It is
  also disclosed in prose at `docs/reference/model-cards/quantile.md:50-51` (the
  kernel density estimate "biases every `bse` here *downward*") and `:173-176`.
  **Worth noting for the next round:** the real `quantile_lp` interval gap sits
  one paragraph above what was measured — `quantile.md:43-48` self-declares that
  the Powell sandwich "is **not** a HAC one" under overlapping multi-step
  outcomes. The finder measured the τ axis when the documented open gap is the
  h axis.

---

# Swept and found sound

- **The validation matrix passes its most serious check.** All 66 rows (52
  estimator + 14 foundational) resolve to a real test file and a real fixture,
  and every row labelled *independent package* has a generator that genuinely
  imports that package. **No documented-formula golden wearing an
  independent-package label.** Provenance table exact; the 37-of-69
  Python-reload claim exact, name for name.
- **Fixture-generator circularity check passes.** `grep -l "import tsecon"
  fixtures/*.py` returns nothing across all 50 generators, including
  function-local imports.
- **The once-fabricated proof output in `testing.md` is now real** — re-running
  the pasted snippet reproduces it character for character.
- **Every runnable snippet in `README.md`, `docs/quickstart.md`,
  `docs/index.md` reproduces byte-for-byte** on whole-block diffs, including the
  18-line `VARResults.summary()` and the 128-row function table.
- **No inert seed on any stochastic path**, across 12 seed-taking functions ×
  sibling switches. Given that class's history here, the strongest single
  negative result of the round.
- **Degenerate-input handling is genuinely strong**: 1387 of 1694 probe units
  raised, almost all naming the offending index or reason. NaN/inf rejection is
  near-universal and explicit.
- **`lp(cumulative=True)` — the spelling the docs recommend — is at nominal**
  (0.890 ± 0.005 at T=200 h=12, 0.937 at T=800). The lag-augmentation argument
  does close for cumulated-outcome-on-contemporaneous-impulse. Worth a docs line
  that it is materially worse-calibrated than the level path at T=200, but not a
  defect.
- **Scale sweep clean across nine decades** for the whole VAR family, all six LP
  entry points, `ols`/`iv_gmm` at every `se_type`, the full SVAR identification
  set, MIDAS, realized measures, filters, and every forecast-evaluation test.

---

## Reproducing this audit

**The probe scripts were scratchpad-only and are gone** — an audit is read-only,
so nothing was committed. Every finding above therefore carries a self-contained
reproducer inline, and the harnesses are described below well enough to rebuild.
If a future round wants them durable, the place to put them is
`docs/examples/coverage/`, next to the five families that already live there.

Seeds: lenses 1–3 at **20260810**, lens 7 at **20260729** (matching the existing
coverage harness, so its numbers compose with the published ones).

The four pieces worth rebuilding:

- **`fingerprint()`** — a structure-aware, *bit-exact* digest of whatever a
  tsecon function returns: floats fed through `to_bits()` so NaN payloads and
  signed zero are not collapsed, dicts fed in sorted-key order, arrays with shape
  and dtype. Plus a `per_key_fingerprint()` so a *partial* no-op is visible when
  a switch moves `loglik` but leaves every band identical.
- **The switch sweep** — for each of the 45 switch-carrying callables, call at
  every value of every switch **on identical data** and compare exactly, sweeping
  the *cross-product* where a function has more than one switch. Both no-ops
  found in the previous round were inert only on the default value of a sibling
  argument, so one-axis-at-a-time would have missed them.
- **The scale sweep** — run each call at `c ∈ {1e-8 … 1e8}`, fit a per-leaf
  exponent from the mild decades, and flag any leaf whose exponent *breaks* at an
  extreme, plus any bool/int/NaN-pattern change. Fitting the exponent rather than
  asserting one is what let it cover the whole surface without per-function
  knowledge of which quantities are equivariant.
- **The convention survey** — 64 trials across 24 functions, asking for each
  whether an argument that is inapplicable *given another argument's value*
  raises or is silently ignored. This is the artefact that killed two findings,
  and it is worth keeping current: it is the only evidence for what this
  library's actual convention is.
