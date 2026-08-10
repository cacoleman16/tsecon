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

- **`garch_fit` still returns silent all-NaN standard errors** when a
  *dimensionless* coefficient (`alpha`/`gamma`/`beta`) sits at its boundary.
  Hit in 10 of 120 probe units, present identically before the scale fix. Same
  user-visible symptom, different cause, and nothing tests it.
- **`garch_fit`'s fitted parameters are still not scale-robust** — in 52 of 330
  cross-scale comparisons the optimizer converged to a different point. The
  standard-error machinery was fixed; the fit was not.
- **Nelder-Mead's x-side floor is a mixed-scale test** — `x_stop` is set by the
  largest component of `x_best`, so a huge coordinate can relax the test for an
  O(1) one. Theoretical; not yet realized in a probe.
- **The diffuse period terminates on a norm test over `P_inf`**, and `T`
  propagates `P_inf`, so a reparametrized model can end it one step early
  (0.47 nats). Absolute and relative floors fail this identically. The fix is
  rank-counting termination.
- **`cv_splits(purge=…, embargo=…)` are inert** on `scheme="expanding"` and
  `"rolling"` while the guide claims those schemes handle leakage
  automatically. Confirmed, not yet fixed.
- **`lp_iv`, `lp_multiplier`, `lp_state` have no cross-horizon covariance**, so
  they get Šidák/Bonferroni bands only and refuse sup-t.
- **Coverage is unmeasured** for `quantile_lp`, panel LP (Driscoll-Kraay),
  `favar`, `dfm_nowcast`, `nelson_siegel`, the `bvar_*` family, MIDAS, and
  `lp(cumulative=…)`. Only two nominal levels are swept anywhere.
- **SARIMA seasonal orders** `(P,D,Q,s)` are not implemented; the docs are
  honest about it.

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
