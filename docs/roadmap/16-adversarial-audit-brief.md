# Adversarial audit brief

> **For an agent or contributor with no prior context.** This is a working
> brief, not a description of a completed thing. It tells you what to hunt for
> in `tsecon`, what has already been found, and the methods that worked. It
> lives in `docs/roadmap/` and is excluded from the published site.

---

## What this audit is for

`tsecon` has, at the time of writing, ~1250 Rust tests, ~650 Python tests, and
golden fixtures pinned to statsmodels, `arch`, `linearmodels`, scikit-learn and
SciPy. It is unusually well tested.

An adversarial audit run against that suite produced **27 confirmed defects.**

That is the whole premise. **Every real defect this project has shipped had the
same shape: the arithmetic was right and the answer was still wrong.** A green
test suite is not evidence of correctness, and your job is to find what it
cannot see.

The canonical example: `iv_gmm(weight="hac")` returned White standard errors.
Not approximately — bit-for-bit identical to `weight="robust"`, max |Δse| =
`0.000e+00` over 3000 replications. The bandwidth defaulted to `0.0`, and a
Bartlett kernel truncated at zero lags *is* the White estimator. Every golden
test passed throughout, because the arithmetic was correct. It was answering a
different question than the one being asked.

---

## The failure classes, in priority order

These are priors, not a checklist. They are ranked by how often they have
actually bitten this repository.

### 1. Silent no-ops

An argument that looks like it changes the estimator and does not, at some or
all of its values.

Found so far: `iv_gmm(bandwidth=0.0)`; `var_irf_bands(bias_correct=True)` on the
default `method="asymptotic"`; `zero_sign_svar(weighted=...)` with the ARW
weight hardcoded to 1.0; `historical_decomposition`'s five sampling arguments
including `seed` on the cholesky path; `cv_splits(purge=…, embargo=…)` on the
default `scheme="expanding"`.

**Method that works.** For every function with a mode / kernel / method / weight
/ scheme string argument, or a numeric knob whose default could be degenerate
(`0`, `1`, `None`), call it at each value **on identical data and compare the
output arrays exactly**. Bit-identical output from two supposedly different
settings *is* the finding. Enumerate the surface from
`bindings/python/python/tsecon/__init__.pyi`.

An accepted-and-ignored `seed` is the worst member of this class: it makes an
unreproducible result look reproducible.

### 2. An absolute tolerance on a quantity that carries scale

The single most productive lens. Rescale the data and the estimator silently
changes character.

Found so far, all the same mistake in different crates: the state-space filter's
`TOLERANCE_DIFFUSE = 1e-10` on a variance in units of *y*²; `garch_fit`'s
finite-difference step floored at `0.1` while `omega` for decimal returns is
~`1e-6`; `markov_switching_ar`'s `1e-10` M-step ridge; `lasso`'s absolute
convergence test; Nelder-Mead's absolute `f_tol`.

**Method that works.** Grep the crates for hardcoded `1e-8` / `1e-10` / `1e-12`
constants and ask, for each: *in what units is the thing on the other side of
this comparison?* Then feed badly-scaled input through the Python surface —
multiply a series by `1e-8` or `1e8` and see whether anything changes character.

**A single-scale test structurally cannot see this class.** Any fix must come
with a scale sweep across several decades.

### 3. Silent wrong answers on degenerate input

A function that returns a plausible number where it should have raised.

Found so far: the Kalman filter returning `loglik = 0.0` *successfully* for
small-variance series — and worse, at an intermediate scale, silently dropping
part of the sample and returning a 51%-wrong likelihood that was finite and
plausible; `panel_unit_root(lrv_kernel="truncated")` returning
`statistic=nan, p_value=nan` on 57 of 60 panels; `gas_volatility` certifying
`converged=True` on an all-zero series with `omega=5e-324`.

**Method that works.** Feed empty, constant, all-NaN, single-observation, and
perfectly-collinear input through the public API. Ask whether the result is
honest. Check iteration caps that report success on non-convergence.

### 4. Discarded computation

The right quantity is computed and then thrown away.

Found so far: `long_memory_d` computed a data-dependent `se_regression` and
returned an asymptotic constant instead — a standard error identical for every
dataset with the same bandwidth, and ~25% too narrow at the library's own
default.

**Method that works.** When a reported diagnostic looks suspiciously stable
across datasets, read the source for a richer quantity that was computed and
dropped.

### 5. Documentation that contradicts the code

The base rate here is high; roughly twenty were found and fixed in two days, and
that sweep was not exhaustive.

**Method that works.** Do not grep for the claim — **re-run the snippet and diff
the whole block.** That is how unrelated drift gets caught (a pasted `vecm`
suggestion said `k_ar_diff=1` where the run printed `0`). Check every numeric
claim in `README.md`, `docs/quickstart.md`, `docs/index.md`,
`docs/reference/testing.md` by counting it yourself. Check every
"roadmap"/"not yet"/"v2"/"does not expose" claim against the current callable
surface — several shipped features were still described as unbuilt.

Docstrings in the `.pyi` versus the actual returned dict keys: call the function
and diff. A documented key that is never set, and a returned key that is
undocumented, are both real.

### 6. Validation claims that do not hold

`docs/reference/validation-matrix.md` exists so a reader can tell a genuine
cross-implementation match from a weaker claim. When it is wrong it is worse
than absent.

**The most serious thing you can find** is a row claiming an *independent
package* golden that is really a documented-formula golden wearing a stronger
label. Verify by reading the generator and confirming which package it actually
imports and whether it computes the same estimand.

Also check: does the named test file exist and load the named fixture? Is the
asserted tolerance in the test source the one the table quotes? An integrity
failure was found here once already — `testing.md` claimed all public functions
were exercised and **pasted a fabricated proof output**.

### 7. Intervals nobody has measured

`docs/examples/interval-coverage.md` measures 40 interval-valued surfaces. The
library has 128 callables. Its own "What is not measured" section is honest
about the gap.

**Method that works.** Enumerate every function returning something
interval-like (`lower`/`upper`/`se`/`bse`/`conf`/`band`/`ci`/`bounds`), cross
against what the audit measures, then *actually measure* the two or three most
at risk with a seeded Monte Carlo from a known-truth DGP. Prefer small effective
samples, plug-in variances, estimated nuisance parameters, or an asymptotic
approximation at a boundary.

A measured miss here is the most valuable thing an audit can produce. This is
how `growth_at_risk` was found covering **0.61 at h=12** against a nominal 0.95.

---

## How to run it

### Every finding needs a reproducer

A finding with no reproducer is discarded. For each, give: the exact call or
`file:line`, what you **observed** (numbers), and what you **expected** (quote
the promise from the docstring, the signature, or the docs).

> "This looks suspicious" is not a finding.
> "I ran X and got Y where the docs promise Z" is.

### Do not pad

Three real findings beat twenty speculative ones. **"My lens found nothing" is a
useful result** and will not be held against you. Say why you believe the area
is sound.

### Verify adversarially

Every finding should go to a second agent whose job is to **refute** it,
defaulting to refuted unless it can reproduce the finding itself. That verifier
must check:

- Is the "expected" behaviour actually promised, or did the finder misread it?
- Is the observation explained by something legitimate — a documented
  convention, a deliberate design choice, a property of the estimator?
- Is it reachable through the public Python API, or only by constructing
  something pathological the library would reject anyway?
- Do the verifier's own numbers match the finder's?

This works. A finding that `ols(se_type="hac", maxlags=0)` was an unfixed
sibling of the `iv_gmm` bug was **refuted**: the arithmetic reproduces, but the
docstring promises "matches statsmodels `cov_type='HAC'`", and statsmodels does
the same thing at `maxlags=0`. Matching the reference is the promise, and it is
kept.

### When you fix something, a second agent checks the fix

The checker's most important job is **not** confirming the defect is gone. It is
answering: *did the fixer weaken or delete a test to get green?* Look for
removed asserts and widened tolerances in `git diff`. A fix that ships by
lowering the bar is worse than no fix.

Also: did they fix the cause or the symptom? An absolute tolerance replaced by a
*different* absolute tolerance is not a fix.

This works too. A checker found a defect **inside the fixer's own scope** —
`which-model-when.md` advertised `gw_test` as the *conditional* test while the
shipped function is unconditional, and the same fixer had just written "the
conditional GW test is still a roadmap item" into another chapter.

---

## Hard constraints

These are not style preferences. Each was learned by losing hours.

| Rule | Why |
|---|---|
| **Never run workspace-wide cargo** (`cargo test/build/check/clippy --workspace`) | It compiles silently for far longer than the 180-second no-progress threshold, so the harness kills the agent, retries six times, and reports the whole workflow failed. Use `cargo test -p <crate>`. Run the full suite yourself, outside the workflow. |
| **Never ask an agent for verbatim reproduction of long code** | Same stall detector, different trigger: a huge accumulated context followed by one enormous generation with no tool call in it. Ask for `file:line` plus short quotes. |
| **An audit is read-only** | No edits, no `git checkout/restore/stash/reset/commit/push/switch`. Probe scripts go in the scratchpad. |
| **Fixture generators must never `import tsecon`** | A reference that calls the code it validates is circular and worthless. Mechanically checkable: `grep -l "import tsecon" fixtures/*.py` must return nothing. |
| **Never `cmd \| tail` and read the exit code** | A pipeline returns the *last* command's status. This masked a failed `maturin` build and a failed `cargo check`, both reported as green. Use `set -o pipefail` or capture `$?` directly. A trailing `grep` that finds nothing exits 1 and fakes a *failure* in the other direction — read the log, not the status. |
| **`--exclude tsecon-python` applies to `cargo test` only** | The PyO3 crate SIGABRTs on macOS hunting for libpython, so excluding it from *tests* is correct. Carrying that exclusion into `clippy`/`fmt` is not — they never run the binary, and doing so meant the most-edited file was the least-checked. That put a red commit on `main`. |
| **Rebuild before judging Python behaviour** | `maturin develop --release -m bindings/python/Cargo.toml`. A stale `.venv` extension will show you the old behaviour and you will conclude the wrong thing. |
| **Establish every promise from `__doc__` and the model card — never from the `.pyi` or a Rust crate doc** | `print(tsecon.<fn>.__doc__)` and `docs/reference/model-cards/` are what a Python user actually reads. The `.pyi` is a thin type stub and the Rust crate docs are invisible from Python; **neither is a promise**. Round 2 lost three findings to this in one sitting — `recession_probit`'s stub omits "probit only", `panel_unit_root`'s omits "LLC" — and hit the *inverse* three times, where a real defect looked worse than it was because the claim it violated lived only in a Rust module doc. Both a false positive and a false negative live in the gap between these four surfaces. |
| **But read all three surfaces, and treat a disagreement among them as the finding** | The rule above is about *which surface binds*, not about reading only one. Round 4 found `long_memory_d`'s runtime `__doc__` — the surface this table calls authoritative — is the **stalest** of the three: it promises *"the estimate `d` and its asymptotic `se`"* while the function returns five keys and the model card correctly distinguishes `se` from a `se_asymptotic` it warns is materially too narrow. A fix can land in the code and the card and miss `__doc__`. Diff `sorted(fn(...).keys())` against all three. |
| **Report comparisons *made*, never comparisons *attempted*** | A probe template that raises produces no comparison, and the sweep output is indistinguishable from a clean pass — the very failure class this brief hunts, in the tooling built to hunt it. Round 2's lens-1 sweep reported "164 axis-comparisons over all 45 callables" while **16 of 59 axes and 6 of 12 seed cases were never compared at all.** Print the reached/attempted ratio every run. The scale sweep already does this right (`reference scale FAILED`); copy it. |

---

## Techniques worth reusing

**A floor-free oracle.** Where two implementations of the same quantity exist
and only one has the suspect guard, the other is an uncontaminated reference.
`tsecon-ssm`'s `filter_matrix` (Joseph form) has no rank floor at all, which
made it the ideal check on the filter fix — bit-identical across nine decades.

**Prove a refactor moved nothing, on one machine.** Capture `to_bits()`
fingerprints of every emitted double against the parent commit and re-check
after each edit. **Never as a cross-platform CI gate** — `faer` dispatches
different SIMD kernels per CPU, and library policy is bit-reproducibility *per
platform, not across*. A stored float snapshot compared across machines must use
a tolerance. This has broken CI twice.

**Prefer a same-run invariant.** "Does requesting a simultaneous band change the
pointwise output?" is a same-run comparison — one binary, one CPU, one set of
kernels — so it can be bit-exact *and* portable. That is almost always the
better test than a stored cross-machine hash.

**Predict the shortfall in closed form, then measure it.** A coverage miss you
can only measure is a finding; one whose size you can *derive* is a proof, and it
cannot be argued away as a finite-sample artefact. For `lp(cumulative="both")`
the overlapping-score autocovariance gives
`sd/se = sqrt([(h+1)² + 2Σ_{k=1}^{h}(h+1−k)²] / (h+1)²)`, which matched the
measured ratio to three digits at every horizon. That turned a contested finding
into a settled one in a single step.

**Separate the treatment from the seed before believing a Monte Carlo gap.** Run
the *same* perturbation you care about at several seeds, and the seed alone at
fixed perturbation, and compare the two spreads. For `bvar_ssvs` this was
decisive: varying the seed moved posterior inclusion by at most 0.054, varying
the data scale moved it 0.390–0.445 on every one of six seeds. Without that
control the finding is indistinguishable from a badly-mixed chain, and a refuter
will say so.

**Mutation-test the tests.** Break the implementation deliberately and confirm
the test fails. Several assertions turned out to be inert: a CSS-covariance
guard passed under the exact mutation it existed to catch, because a
same-vs-same tolerance cannot detect a change that makes both sides agree
*better*.

**Check what a fix's tests structurally cannot see.** All eight tests for one
fix used a single-state model, where `‖z‖₁ = 1` and `max_diag(P) == z'Pz` are
the same number — so an entire class of bug was invisible by construction.

---

## Already found and fixed — do not re-report these

> **Rounds 2–6 have been run.** Results in
> [17-audit-round-2-findings.md](17-audit-round-2-findings.md) (21 candidates,
> 8 survivors),
> [18-audit-rounds-3-4-findings.md](18-audit-rounds-3-4-findings.md) (14
> candidates raised, 7 survived), and
> [20-audit-round-6-findings.md](20-audit-round-6-findings.md) (13 raised,
> 9 confirmed + 2 rescoped, 7 fixed in the same release — round 6 also ran
> the checker role over the 0.3.0 merge range and the first Bayesian-
> calibration/SBC pass, which is where its headline came from). **Read the
> "Refuted" sections before you start** — together they record every dead end
> with the evidence that killed it (no count quoted here: the one that used
> to sit in this sentence went stale within two commits), including several
> that look compelling on first contact (`gmm_nonlinear`
> returning its own starting value; `ccc_garch` on a singular correlation;
> `panel_fe` at N=1; `zero_sign_svar`'s "dead" `weighted` flag; `cg_regression`'s
> intercept). Re-deriving those is the single easiest way to waste a round.
>
> Lenses 1, 2, 3 and 4 are now swept to **measured, reported completion** —
> 59/59 switch axes, 12/12 seed cases, 47 functions scale-swept, 128/128
> callables degenerated on the data axis plus 192 argument-axis triples, 64/64
> functions checked for constant diagnostics, and all 41 crates cross-referenced
> for public Rust names the binding layer never mentions (172 of them).
>
> **Lens 4 needs both halves.** Its mechanical probe flags *returned* quantities
> that come back constant; its source read finds quantities **never returned at
> all**. The two do not overlap — the mechanical half declared the returned
> surface clean while the source read found three documented-then-dropped
> quantities. Run both.
>
> Where the next round has most room: **lens 7** — the `bvar_*` family framed as
> Bayesian calibration (draw from the model's own prior and check the credible
> set) rather than frequentist coverage, `bai_perron`'s Bai-1997 break-date CIs,
> `proxy_ar_sets`, `adl_midas`.

Fixed in `0.2.0`: `ols` gained `hc2`/`hc3`; `iv_gmm`'s HAC bandwidth no-op;
`iv_gmm` reports `first_stage_f`; `arima_fit`'s missing drift-uncertainty term.

Fixed since: the state-space diffuse-tolerance floor; `panel_unit_root`'s NaN;
`zero_sign_svar`'s dead `weighted` flag; `garch_fit`'s finite-difference step;
`markov_switching_ar`'s M-step ridge; `lasso`/`elastic_net` convergence;
`gas_volatility`'s constant `converged` flag and its all-zero certification;
`long_memory_d`'s discarded standard error; `growth_at_risk`'s missing HAC
correction; Nelder-Mead's absolute `f_tol` certificate;
`var_irf_bands(bias_correct=True)`; `historical_decomposition`'s inert sampling
arguments; and roughly twenty documentation claims.

## Known open — pick these up

**The rounds-2–4 backlog was cleared in 0.3.0** — the `lp(cumulative="both")`
standard-error defect (default now HAC for that mode, invalid pairing raises),
`bvar_ssvs`'s scale-carrying hyperpriors (now semi-automatic), the HAC
`use_correction` default story (`har_rv` flipped, every claim corrected),
the three dropped-at-the-binding quantities, `cv_splits`' inert purge/embargo
(act-or-raise), `smooth_lp`'s absolute λ grid, `flp`'s generated-regressor
disclosure, the IVX joint-k caveats, the coverage-page hc3 row (plus a
sync-check that makes a dropped row fail CI), `panel_lp`'s jackknife
cookbook advice, `gmm_nonlinear`'s misattributed callback error, the panic
pair, and the documentation drift. Round 6 then fixed its own headline
(`bvar_hierarchical`'s ML-II collapse — default now `"glp"`) plus
`seasonal_strength`'s constant-series noise, `bvar_ssvs`'s misattributed
overflow message, and four doc-surface disagreements. See the CHANGELOG's
0.3.0 section and [20-audit-round-6-findings.md](20-audit-round-6-findings.md).

**Still open — confirmed, documented where user-facing, not yet re-engineered:**

- **`garch_fit` returns silent all-NaN standard errors** when a
  *dimensionless* coefficient (`alpha`/`gamma`/`beta`) sits at its boundary
  (round 1; 10 of 120 probe units). Nothing tests it.
- **`garch_fit`'s fitted parameters are not scale-robust** — 52 of 330
  cross-scale comparisons converged to a different point (round 1). The
  standard-error machinery was fixed; the fit was not.
- **Nelder-Mead's x-side floor is a mixed-scale test** — theoretical; not yet
  realized in a probe.
- **The diffuse period terminates on a norm test over `P_inf`** — the fix is
  rank-counting termination.
- **`lp_iv`, `lp_multiplier`, `lp_state` have no cross-horizon covariance**,
  so they get Šidák/Bonferroni bands only and refuse sup-t.
- **`proxy_ar_sets`' propagated variance is estimate-correlated at long
  horizons** (round 6): coverage 0.80–0.89 at the default `horizon=12`
  depending on the DGP, one-sided misses, fading in T. Documented on every
  surface; a re-engineered long-horizon correction is the open item.
- **`ivx_test`'s joint Wald size in k** (rounds 3–4): documented everywhere
  in 0.3.0; a size-restoring joint test is the open item.
- **`flp`'s per-element `se` for estimated scores** (rounds 3–4): documented
  in 0.3.0; a generated-regressor correction (or bootstrap route) is the
  open item.
- **A minimum-cycles advisory for `nsdiffs`/`seasonal_strength`** (round 6):
  below ~4 cycles the rule flags pure noise (documented with measurements);
  whether the advisory belongs in the output itself is open.
- **Uniform typed errors for negative integer arguments** (round 6):
  `lags=-1` and friends raise raw `OverflowError` library-wide.
- **The unmeasured-seven list is closed** — measured in
  `docs/examples/coverage/quantile_panel_lp.py` and `factor_midas.py`
  (the registry now holds 50 outputs; `check_page.py` guards the page).
  `weighted_midas`, `dfm_nowcast`, `nelson_siegel` ship no interval
  (per-run key-set tripwire). Headline numbers: post-fix
  `lp(cumulative="both")` 0.920 at T=400; FAVAR bands 0.673 at N=20, T=800
  (the generated-regressor cost, priced against a true-factor oracle);
  the panel card's SPJ coverage gain re-measured at 8× the replications is
  +2.5pp, not +8pp (bias removal corroborated exactly). Still outside the
  registry: `growth_at_risk` (card table), `proxy_svar`/`proxy_ar_sets`
  (card), `nongaussian_svar`, GARCH forecast intervals, `flp`. Only two
  nominal levels are swept anywhere.

---

## Reporting

Rank by what a user would actually experience, not by how clever the finding is.
Use these severities:

- `silent-wrong-answer` — a plausible number where the answer is meaningless or
  materially wrong
- `silent-noop` — an argument that cannot change the output
- `trap` — reachable, documented behaviour that will mislead a careful reader
- `overclaim` — a claim in the docs the code does not support
- `cosmetic` — real but harmless

For each: title, `file:line`, observed (with numbers), expected (quote the
promise), reproducer, severity. Then hand it to someone whose job is to prove
you wrong.
