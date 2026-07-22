# Testing & validation

This page is the single map of how `tsecon` is tested. If you are deciding
whether to trust a number this library produced, start here: it names every
tier of the test suite, says what each tier can and cannot prove, gives the
command to run it, and — at the end — lists the things that are honestly
**not** covered.

The [validation matrix](validation-matrix.md) is the per-method companion to
this page: it names, family by family, the reference each golden is measured
against and the tolerance it is held to. This page explains the machinery
around it.

---

## The state of the suite

Verified on this working tree (macOS, Apple silicon, Rust 1.97.1, CPython
3.12.7, release build of the extension):

| Tier | Count | Command |
|---|---|---|
| Rust tests (total) | **1010 passed, 0 failed, 0 ignored** | `cargo test --workspace --exclude tsecon-python` |
| — integration tests in `crates/*/tests/` | 836 | |
| — unit tests in `src/` (`#[cfg(test)]`) | 130 | |
| — documentation tests | 44 | |
| Python binding tests | **496 passed** in 5.3 s | `.venv/bin/python -m pytest bindings/python/tests -q` |
| Crates | 41, **every one** with a `tests/` directory | |
| Golden fixtures | 61 JSON files, produced by 42 generator scripts | `fixtures/` |
| Public Python functions | 122 — **all 122** are called at least once in the binding suite | |

Of the 836 Rust integration tests, **174 are golden tests** (`golden.rs` in 38
crates, plus `unitroot_golden.rs`, `smooth_golden.rs`, `pmg_golden.rs`, and the
new identification/unit-root goldens) and **434 are property tests**
(`properties.rs` in 38 crates, plus `unitroot_properties.rs`,
`smooth_properties.rs`, and `pmg_properties.rs`). The remainder are validation, cross-check, and
reproducibility suites described below.

---

## 1 · The philosophy: validation-gated

One rule governs the project, and it is stated the same way in
[CONTRIBUTING](https://github.com/cacoleman16/tsecon/blob/main/CONTRIBUTING.md):
**nothing lands without a named golden target it has to hit.** A "named target"
is one of exactly three things:

1. an **independent reference implementation** (statsmodels, SciPy, `arch`,
   `linearmodels`, scikit-learn, ArviZ) computing the same estimand through a
   completely separate code path;
2. a **documented closed form** — the published formula transcribed into NumPy,
   with the algebra written out in the generator's docstring; or
3. a **statistical property** established by seeded Monte Carlo (size,
   coverage, consistency, parameter recovery).

The [validation matrix](validation-matrix.md) says which of the three each
method family gets. Across its 40 estimator-family rows, **20 name an
independent package**, 16 are documented-formula goldens, and 4 are mixed or
explicitly property-validated. That distribution is published rather than
smoothed over, because a documented-formula golden is a weaker claim than a
cross-implementation match and you should know which one you are relying on.

### Reference libraries are an offline build tool, not a dependency

This is the part that most often surprises people. statsmodels, `arch`,
`linearmodels`, scikit-learn, ArviZ, and SciPy appear **only** inside
`fixtures/generate_*.py`. Those scripts are run once, by a developer, to
*produce* the JSON goldens that get committed. The shipped wheel never imports
them. The runtime dependency list in
[`bindings/python/pyproject.toml`](../../bindings/python/pyproject.toml) is
exactly one line:

```toml
dependencies = ["numpy>=1.22"]
```

So the validation is as strong as a statsmodels cross-check, and the install is
as light as NumPy. Those two facts are usually in tension; the fixture
mechanism is how they are reconciled.

### Generators must not import `tsecon`

A reference that called the code it is supposed to validate would be circular
and worthless. The rule is mechanically checkable, and the tree honors it:

```sh
$ grep -l "import tsecon" fixtures/*.py
$          # no output — no generator imports the library under test
```

Every generator computes its numbers from an independent library or from a
formula written out in its own docstring. Two representative examples:

- [`generate_panel_fixtures.py`](../../fixtures/generate_panel_fixtures.py) —
  *independent package*. Simulates a balanced `N=30, T=120` panel with entity
  fixed effects and a known dynamic response, then pins the within estimator
  with `linearmodels.panel.PanelOLS` under three covariance estimators
  (clustered by entity, Driscoll-Kraay, nonrobust).
- [`generate_tsecon-gas_fixtures.py`](../../fixtures/generate_tsecon-gas_fixtures.py)
  — *documented formula*, and it says so in its own first paragraph: "This is a
  DOCUMENTED-FORMULA golden… Nothing here calls the Rust code, so the golden is
  non-circular." The Creal-Koopman-Lucas score-driven recursion, the Gaussian
  and Student-*t* densities, the score `∇_t`, the information `I_t` and the
  scaling `S_t` are all written out in the docstring and then applied in plain
  NumPy. The crate must reproduce the filtered variance path and the
  log-likelihood to ~1e-10.

### Fixtures store derived numbers, never a dataset

Each JSON holds only simulated draws from a seeded NumPy `default_rng` through
a known DGP, or *transformations* of the two public-domain reference series
bundled with statsmodels (the Nile river-flow series, 1871–1970; and US
macrodata from BEA/FRED). No licensed dataset is redistributed. Each file
carries a `_meta` block pinning the exact reference-library versions used, so
the values are reproducible. See
[`fixtures/README.md`](../../fixtures/README.md).

---

## 2 · The tiers

### Tier 1 — Rust golden tests

**What it proves:** the arithmetic agrees with a named reference on a specific
dataset, to a stated tolerance.

144 tests across 33 crates load `fixtures/*.json` and assert the crate
reproduces the stored reference values. Tolerances are the *asserted* bounds in
the test source and are frequently far tighter than the spec floor — 1e-12
relative for diagnostics, 1e-8 for VAR parameters and IRFs, bit-exact for the
Philox RNG stream against NumPy's.

```sh
cargo test --workspace --exclude tsecon-python golden
```

**What it cannot prove:** anything about repeated sampling. A golden match says
the code computes the documented quantity; it says nothing about whether that
quantity is a valid test statistic. That is Tier 5.

### Tier 2 — Rust property tests

**What it proves:** invariants that must hold for *every* input, not just the
fixture's. 316 tests across 32 crates. These are hand-written with seeded
generators, and they fall into recognizable families:

| Invariant family | Real example |
|---|---|
| **Stability** | `tsecon-linalg::levinson_ar_is_stable` — the Levinson-Durbin AR fit must always land inside the unit circle. `tsecon-dsge::solved_p_is_stable`. |
| **Adding-up** | `tsecon-var::fevd_rows_sum_to_one`; `tsecon-connect::gfevd_rows_sum_to_one`. |
| **Coverage** | `tsecon-lp::lag_augmented_monte_carlo_coverage_is_nominal` — simulates an AR(1), and asserts the 95% lag-augmented LP interval covers `ρ^h` between 0.85 and 0.99 at every horizon. |
| **Size** | `tsecon-predreg::ivx_wald_holds_size_uniformly_over_persistence` — 3000 reps, `T=250`, endogeneity −0.9, for ρ ∈ {0.90, 0.95, 0.99, 1.00}; IVX size must stay inside 0.05 ± 0.02 at *every* ρ, including the exact unit root. Its sibling `naive_ols_over_rejects_at_the_unit_root` pins the failure it exists to fix. |
| **Symmetry** | `tsecon-stats::symmetry`; `tsecon-linalg::symmetrize_properties`. |
| **Specialization / nesting** | `tsecon-termstructure::svensson_nests_nelson_siegel_and_fits_at_least_as_well`; `tsecon-midas::umidas_is_free_lag_limit_of_weighted`; `tsecon-hac::hc1_is_hc0_scaled_and_hac_bw0_matches_hc0`. |
| **Annihilation / reconstruction** | `tsecon-filters::hp_cycle_plus_trend_reconstructs_input_exactly`; `bk_cycle_annihilates_constant_and_linear_trend`; `tsecon-spectral::periodogram_satisfies_parseval`. |
| **No leakage** | `tsecon-ml::purged_kfold_excludes_all_leaky_indices` — no split may put a test index at or before a training index. |
| **Sampler correctness** | `tsecon-bayes::ffbs_geweke_getting_it_right` — a Geweke joint-distribution ("getting it right") test comparing marginal-conditional iid draws from the joint against the successive-conditional simulator across five test functions. This is the strongest available check on a Gibbs kernel. |
| **RNG reproducibility** | `tsecon-rng::advance_k_equals_k_draws_from_fresh_stream`, `substreams_are_pairwise_independent_smoke`, `clone_replays_identical_sequence`. |

**What it proves that a golden cannot:** that the estimator is *internally
coherent* — that its own identities hold, that its interval means what it says,
and that nesting relationships between estimators are real rather than
coincidental at one dataset.

### Tier 3 — Rust validation tests ("errors that teach")

**What it proves:** every guard returns a *typed, informative* error rather
than panicking, producing `NaN`, or silently returning garbage.

Eight dedicated tests carry names like `error_paths`,
`errors_display_teaching_messages`, `error_paths_teach`, and
`guardrails_teach_on_degenerate_inputs`, alongside
[`tsecon-spectest/tests/validation.rs`](../../crates/tsecon-spectest/tests/validation.rs)
(9 tests, "every guard in the crate returns a typed `SpecTestError` rather than
panicking") and
[`tsecon-ident/tests/dgp_validation.rs`](../../crates/tsecon-ident/tests/dgp_validation.rs).

The standard is not just "it errors" but "the message names the problem". From
[`tsecon-diag`](../../crates/tsecon-diag/tests/properties.rs):

```rust
let err = acf(&[1.0, 1.0, 1.0], 1, false).unwrap_err();
assert!(err.to_string().contains("constant"), "message should teach: {msg}");
```

and from [`tsecon-hac`](../../crates/tsecon-hac/tests/properties.rs), where the
error variant carries the offending index:

```rust
assert!(matches!(
    lrv(&[1.0, f64::NAN, 0.5], Kernel::Bartlett, 4.0),
    Err(HacError::NonFinite { index: 1, .. })
));
```

There are also targeted cross-check and reproducibility suites —
[`tsecon-ssm/tests/crosscheck.rs`](../../crates/tsecon-ssm/tests/crosscheck.rs)
(univariate filter against an independent Joseph-form matrix filter) and
[`tsecon-bootstrap/tests/reproducibility.rs`](../../crates/tsecon-bootstrap/tests/reproducibility.rs).

### Tier 4 — Python binding tests

**What it proves:** the *shipped* module reproduces the same goldens the Rust
core hits, and that nothing is lost or corrupted crossing the PyO3 boundary.

471 tests in 42 files. 37 of the 53 fixture JSONs are reloaded here and checked
a second time through the Python API, so the guarantee is end-to-end rather
than core-only. But the suite adds four things the Rust tests structurally
cannot cover:

- **Marshalling.** NumPy arrays have to arrive in Rust as the caller meant
  them. Several tests deliberately pass **non-contiguous** input — a transposed
  array (`np.array(...).T`, whose `C_CONTIGUOUS` flag is `False`) is the normal
  way a series-major fixture becomes a `T × k` panel, and
  `test_coint_regime.py`, `test_favar.py`, `test_midas_mgarch.py` and
  `test_mean_group_var.py` all take that path.
- **Dict keys and shapes.** Estimators return Python dicts; the `Results`
  facades (`test_results_*.py`, 148 tests) assert key by key that the object
  *is* the dict the raw function has always returned, with rendering added on
  top and nothing removed.
- **Error propagation.** 41 `pytest.raises` assertions check that a Rust
  `Err(...)` surfaces as a Python `ValueError`/`RuntimeError` with a message
  you can act on, rather than an abort. `test_gmm_nonlinear.py` goes the other
  direction too: a Python moment function that raises must propagate its
  message back out through the Rust Nelder-Mead driver
  (`match="boom from the Python moment function"`).
- **Surface completeness.** All 93 exported functions are invoked as
  `tsecon.f(...)` somewhere in the suite — verified, not asserted:

  ```sh
  .venv/bin/python -c "
  import tsecon, re, pathlib
  fns = {n for n in dir(tsecon) if not n.startswith('_') and callable(getattr(tsecon, n))}
  txt = ''.join(p.read_text() for p in pathlib.Path('bindings/python/tests').glob('*.py'))
  print(sorted(f for f in fns if not re.search(rf'tsecon\.{f}\s*\(', txt)))
  "
  # []
  ```

### Tier 5 — Monte Carlo validation

**What it proves:** the statistical properties a fixture match *cannot* prove —
that a test holds its size, that an interval covers at its nominal rate, that
an estimator is consistent. These are claims about repeated sampling, and
simulation is the only honest check.

Full write-up: **[Monte Carlo validation](../examples/monte-carlo.md)**.

```sh
.venv/bin/python docs/examples/monte_carlo.py     # 3.0 s on a release build here
```

The headline result is the IVX size table. With a persistent, endogenous
predictor and a true slope of zero (`reps=2000, T=250, corr(u,e) = −0.95`,
nominal 0.05):

| ρ | OLS *t*-test | IVX Wald |
|---|---|---|
| 0.90 | 0.065 | **0.062** |
| 0.95 | 0.076 | **0.057** |
| 0.99 | 0.140 | **0.051** |
| 1.00 | **0.278** | **0.053** |

The naive OLS *t*-test rejects a true null **27.8% of the time at an exact unit
root** — five and a half times its nominal rate — while IVX sits on 0.05 across
the entire range. No golden fixture could have caught that difference: both
estimators compute their documented formula correctly. The page also publishes
a limitation it would be easy to hide (HAC recovers coverage under serial
correlation but only to 0.451 at φ = 0.95, not to 0.95) and confirms the
textbook Kendall AR(1) bias `−(1+3φ)/T` at four sample sizes.

### Tier 6 — Frontier Monte Carlo

**What it proves:** the *comparative* questions, where the answer is a
trade-off rather than a verdict.

Full write-up:
**[Frontier Monte Carlo](../examples/monte-carlo-frontier.md)**.

```sh
.venv/bin/python docs/examples/monte_carlo_frontier.py   # 0.74 s measured here
```

Two experiments:

- **LP vs VAR (bias/variance).** With the correct lag order the VAR is
  dramatically more efficient (RMSE 0.0014 vs LP's 0.1189 at h = 12). Truncate
  a lag and the VAR's average absolute bias quadruples (0.0056 → 0.0241) while
  LP's does not move (0.0090 → 0.0089) — yet the VAR's *average RMSE still
  stays lower* (0.0451 vs 0.1112). The conditional answer is the finding; the
  page explicitly declines to conclude "use LP".
- **LP-IV with a weak instrument.** The surprising result: nominal 95% coverage
  barely moves across instrument strengths (0.92–0.96), while the **point
  estimate** breaks — at a first-stage F of 1.68 the median estimate is 1.29
  against a truth of 1.0, a 29% bias. Hence `tsecon.lp_iv` returns
  `first_stage_f` per horizon.

### Tier 7 — Published-result replication

**What it proves:** that the library recovers a number a *journal* published,
from the original authors' data — the only tier where neither the data nor the
answer is ours.

Every other tier checks tsecon against a reference implementation, a closed
form, or a DGP we wrote. Those can all be simultaneously wrong in the same
direction if a specification is misunderstood. A replication cannot: the data,
the identification, and the target number all come from outside.

**1. [Ramey & Zubairy (2018)](../examples/replication-ramey-zubairy.md)**, *JPE*
126(2) — government-spending multipliers from US historical data, using their
military-news shock and Gordon-Krenn normalisation, via `tsecon.lp_multiplier`:

| h (quarters) | 4 | 8 | 12 | 16 | 20 |
|---|---|---|---|---|---|
| integral multiplier | 0.635 | 0.657 | 0.700 | 0.706 | 0.743 |

0.64–0.74 against RZ's published **0.6–0.8**, and below one — their central
claim. The dataset is RZ's public replication file, committed at
`fixtures/ramey_zubairy.csv`, so the replication runs fully offline;
`test_replication_ramey_zubairy.py` re-runs the estimation and pins the result.

**2. [Estrella & Mishkin (1998)](../examples/replication-yield-curve-recession.md)**,
*REStat* 80(1) — the Treasury yield curve predicts recessions. A probit of the
NBER recession indicator twelve months ahead on the term spread
(`GS10 − TB3MS`), monthly FRED data 1953–2026, recovers the signature result: a
spread coefficient of **−0.58** (`z = −9.6`), a −1pp inversion implying a **48%**
recession probability within the year against **0.8%** for a +3pp steepness.
Guarded offline by `test_replication_yield_curve.py` against a committed FRED
snapshot.

Both pages state scope explicitly: they reproduce the economic result — the
sign, significance and magnitude of the published finding — not a line-by-line
port of the authors' code or their exact inference conventions.

### Tier 8 — Benchmarks (parity first)

**What it proves:** that tsecon and a mature reference compute *the same
number* — before anything is timed.

Full write-up:
[`benchmarks/README.md`](https://github.com/cacoleman16/tsecon/blob/main/benchmarks/README.md).

```sh
.venv/bin/python benchmarks/bench.py            # full run
.venv/bin/python benchmarks/bench.py --quick    # fewer repeats
```

The script **exits non-zero if any parity check fails**, so it doubles as a
cross-library correctness gate. Four operations are covered
(`adf`/`adfuller`, `var_fit`/statsmodels `VAR`, `ols`+HAC/statsmodels HAC,
`garch_fit`/`arch.arch_model`), and the parity matrix is the deliverable — it
is machine-independent, unlike every timing number.

The harness also auto-detects debug builds and refuses to let their timings be
read as speed claims. The published example run is a case study in why: on a
debug build the estimates are *identical to machine precision* and tsecon is
**slower on 4 of 4 operations**; on a release wheel it is faster on 3 of 4, and
the fourth (GARCH QMLE, ~4× slower than `arch`) is published as a loss rather
than dropped.

### Tier 9 — The structural guards

Two tests in
[`bindings/python/tests/test_stub_sync.py`](../../bindings/python/tests/test_stub_sync.py)
catch the most common "forgot a step" mistakes — the ones that ship a working
library with lying documentation.

- **Stub sync** (`test_stub_matches_runtime`). The type stub
  `python/tsecon/__init__.pyi` must describe *exactly* the runtime function
  surface. It compares the set of public callables on the imported module
  against the `def` lines in the stub and fails on either a **missing** or an
  **extra** name. Add a binding without updating the stub and CI fails; leave a
  removed function documented and CI fails too. `test_py_typed_marker_present`
  additionally asserts the PEP 561 marker ships.
- **API drift guard** (`test_api_reference_not_stale`). `docs/reference/api.md`
  is generated from the stub by `docs/gen_api_reference.py`. The test *runs the
  generator* in a subprocess and asserts the committed file is byte-identical
  afterwards. A forgotten regeneration fails CI instead of silently shipping a
  stale reference. The fix it prints is the fix you run:

  ```sh
  .venv/bin/python docs/gen_api_reference.py
  ```

---

## 3 · The Python test files

All 34 files in
[`bindings/python/tests/`](../../bindings/python/tests), with collected test
counts:

| File | Tests | What it covers |
|---|---:|---|
| `test_backtest.py` | 4 | Pseudo-out-of-sample backtest engine; no external golden — the naive forecaster makes every quantity a closed form checked against NumPy. |
| `test_coint_regime.py` | 3 | Johansen / Engle-Granger cointegration and Markov-switching AR against `coint.json` and `regime.json`. |
| `test_cv_splits.py` | 4 | Leakage-safe CV split geometry: no test index at or before a train index; purge/embargo gaps honored. |
| `test_replication_ramey_zubairy.py` | 4 | The RZ government-spending replication, offline against the committed panel: multiplier below one across horizons, strong first stage, and a guard that it is not the outcome-only cumulative trap. |
| `test_replication_yield_curve.py` | 2 | The Estrella-Mishkin yield-curve recession probit, offline against the committed FRED snapshot: the spread coefficient stays significantly negative. |
| `test_depth.py` | 4 | Realized volatility / HAR-RV, Diebold-Yilmaz connectedness, PCA factor model vs `{realized,connect,favar}.json`. |
| `test_dynamic_ns.py` | 4 | Dynamic Nelson-Siegel (Diebold-Li 2006) two-step fit; row-100 cross-sectional golden anchors the per-date fit exactly. |
| `test_favar.py` | 4 | Two-step FAVAR (Bernanke-Boivin-Eliasz 2005): step-1 factors must match the NumPy PCA golden up to a joint sign flip; assembly and IRFs checked structurally. |
| `test_gmm.py` | 2 | IV-GMM two-step robust fit against a `linearmodels` `IVGMM` golden. |
| `test_gmm_nonlinear.py` | 3 | Nonlinear GMM with the moment function written **in Python** and called back into from Rust; exactly-identified mean/variance system has a closed form. Also pins exception propagation Python → Rust → Python. |
| `test_intervals.py` | 12 | Interval API audit: every band must equal mean ± z·se at the *requested* coverage, with the multipliers pinned to `scipy.stats.norm.ppf` values (1.9600, 1.6449, 0.9945). |
| `test_lp_ml.py` | 7 | Local projections and penalized regression against the same statsmodels / `linearmodels` / sklearn fixtures the crates use. |
| `test_lp_state.py` | 4 | State-dependent (interacted) LP, Ramey-Zubairy 2018. No golden exists, so it mirrors the crate property test: a 2× state-1 impact must be recovered and separate significantly from state 0. |
| `test_mean_group_var.py` | 5 | Pesaran-Smith mean-group panel VAR pinned against the already-bound per-entity `var_fit`/`var_irf` primitives averaged by hand — must agree to machine precision. |
| `test_midas_mgarch.py` | 4 | MIDAS weighting/design and CCC/DCC multivariate GARCH against `midas.json`, `mgarch.json`. |
| `test_ml_paths.py` | 3 | Adaptive LASSO oracle behavior and elastic-net path monotonicity with AIC/BIC selection on a sparse design. |
| `test_new_crates.py` | 9 | GAS score-driven volatility, mean-group / CCE-MG panel, DFM nowcasting; `panel_mean_group` tight against its statsmodels golden, the other two structural. |
| `test_panel_fceval.py` | 3 | Panel estimators and the Clark-West / Giacomini-White forecast comparison tests. |
| `test_pmg_news.py` | 3 | PMG panel estimator against its documented-formula golden; `dfm_news` against its exact adding-up identity. |
| `test_predreg.py` | 2 | IVX / Stambaugh predictive-regression point estimates and Wald statistics (the *size* claim lives in the crate's MC property tests). |
| `test_realized_extras.py` | 7 | Realized/tripower quarticity, BNS jump test, Parkinson & Garman-Klass range variances against documented closed forms. |
| `test_results_arima.py` | 20 | `ARIMAResults` is *additive*: key-by-key dict equality against a raw `arima_fit` call, then the rendering. |
| `test_results_dsge.py` | 26 | `DSGEResults` against the Cagan money-demand model, which has a closed-form saddle-path solution (`G = 1/(1−aρ)`, `P = ρ`, `Q = 1`). |
| `test_results_garch.py` | 22 | `GARCHResults` backward compatibility — every original `garch_fit` key must survive untouched. |
| `test_results_lp.py` | 29 | LP results facade: dict/list contracts, summary, IRF grid, round-trip through `to_dict()`. |
| `test_results_predreg.py` | 35 | Predictive-regression facade on the Stambaugh DGP it exists for (ρ = 0.99, corr = −0.9, **true β = 0**) — the case whose reporting the summary must get right. |
| `test_results_var.py` | 16 | VAR facade: dict/list contracts, summary, IRF grid. |
| `test_roadmap_gaps.py` | 6 | Recession probability, survey expectations, and long-memory GPH / local-Whittle bindings. |
| `test_smoke.py` | 33 | End-to-end: the Rust core called from Python across the core surface, plus Philox bit-compatibility against the live NumPy. |
| `test_spectest_afns_dsge.py` | 18 | Specification tests (White/Breusch-Pagan, RESET, Chow, CUSUM), the AFNS yield adjustment, and `dsge_solve`. |
| `test_spectral.py` | 3 | Periodogram / Welch / coherence against `scipy.signal` fixtures. |
| `test_stub_sync.py` | 3 | The structural guards: stub ↔ runtime surface, `py.typed` present, `api.md` not stale. |
| `test_survey_longmemory_bindings.py` | 9 | `forecast_disagreement` on a ragged panel and `frac_integrate` as the exact inverse of `frac_diff`; every expected number hand-computed or built from a tiny in-test NumPy reference. |
| `test_termstructure.py` | 3 | Nelson-Siegel (Diebold-Li) and Svensson curve fits against `termstructure.json`. |
| `test_weighted_midas.py` | 4 | Weighted MIDAS NLS on a simulated exp-Almon DGP plus closed-form self-consistency against U-MIDAS (no golden fit exists in `midas.json`). |

---

## 4 · How to run everything

```sh
# 1. Rust core — 1010 tests
cargo test --workspace --exclude tsecon-python

# 2. Python bindings — 496 tests
.venv/bin/python -m pytest bindings/python/tests -q

# 3. Monte Carlo evidence (seeded, reproducible)
.venv/bin/python docs/examples/monte_carlo.py
.venv/bin/python docs/examples/monte_carlo_frontier.py

# 4. Cross-library parity gate (exits non-zero on any parity failure)
.venv/bin/python benchmarks/bench.py

# 5. Lints and formatting, exactly as CI runs them
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings

# 6. Docs — fails on any broken link or missing nav entry
.venv/bin/python docs/gen_api_reference.py
.venv/bin/python -m mkdocs build --strict
```

### The `--exclude tsecon-python` caveat (macOS)

CI runs plain `cargo test --workspace` on Ubuntu and it passes. **On macOS it
does not**, and the reason is a dynamic-linking detail, not a test failure.
`tsecon-python` is the PyO3 `cdylib` binding crate. `cargo test` builds a test
*binary* for it, and that binary links against libpython, which the dynamic
loader cannot find at runtime:

```
     Running unittests src/lib.rs (target/debug/deps/_core-a58bbb63cbdd04d5)
dyld[20882]: Library not loaded: @rpath/libpython3.12.dylib
error: test failed, to rerun pass `-p tsecon-python --lib`
Caused by:
  process didn't exit successfully: ... (signal: 6, SIGABRT: process abort signal)
```

The abort matters for more than tidiness: it kills the run, so the reported
test count is **truncated** — you lose every crate that had not finished. Since
`tsecon-python` holds no Rust tests of its own (its behavior is covered
entirely by the Python suite), excluding it costs nothing:

```sh
cargo test --workspace --exclude tsecon-python
```

Two follow-on tips when counting: redirect to a file rather than piping through
`grep`, since piping loses buffered lines, and sum the `test result:` lines
across all binaries — cargo prints one per test target, not one total.

```sh
cargo test --workspace --exclude tsecon-python > /tmp/rust.txt 2>&1
grep "test result" /tmp/rust.txt | awk '{p+=$4; f+=$6} END {print p, "passed,", f, "failed"}'
# 1010 passed, 0 failed
```

### Build a release extension before timing anything

`maturin develop` installs a **debug** build of the Rust core: unoptimised,
full symbols, and on this machine 43.4 MB against a release build's 6.0 MB. It
is correct but slow, and it makes the Python suite and the Monte Carlo scripts
feel much heavier than they are. For a sense of scale, the same GARCH(1,1)
QMLE fit in the benchmark harness takes **544.9 ms debug** and **33.8 ms
release** — a ~16× gap on exactly the optimiser-heavy path the fitting tests
spend their time in.

```sh
maturin develop --release -m bindings/python/Cargo.toml
```

With a release extension installed, the full 332-test Python suite runs in
**4.4 s** and `docs/examples/monte_carlo.py` in **3.0 s** on this machine. Do
not quote any timing taken against a debug build.

---

## 5 · What is *not* tested

Honest limitations. These are gaps we know about, not ones you should have to
discover.

**Validation strength.**

- **16 of the 40 estimator families are documented-formula goldens, not
  cross-implementation checks.** No independent package computes the quantity,
  so the generator transcribes the published closed form into NumPy and pins
  the crate to it. This proves the Rust reproduces the documented algebra; it
  does *not* independently confirm the algebra is the statistically right
  choice. Where that gap matters, a seeded Monte Carlo property test carries
  the claim instead — but you should read the
  [validation matrix](validation-matrix.md) row before relying on one of these.
- **GARCH parity is at optimiser tolerance, not machine precision.** Fixed
  parameter log-likelihoods match `arch` to 1e-8, but the *fitted* QMLE
  parameters are asserted only to `atol 1e-3` (log-likelihood `rtol 1e-5`),
  because two different optimisers will not land on bit-identical parameters.
  That is a real and stated difference, not a hidden one.
- **Multivariate GARCH has no external DCC reference at all.** The univariate
  stage is `arch`-pinned through `tsecon-garch`, but the DCC dynamics are
  validated only by properties (every `R_t`/`H_t` positive definite,
  correlation targeting) plus loose single-realization parameter recovery. The
  dynamic Kauppi-Saikkonen recession probit is in the same position.
- **Monte Carlo results carry their own simulation error.** The suites use
  2000–3000 reps (400–500 for the frontier experiments), so a size estimate has
  a standard error around 0.005. The property-test bands are set deliberately
  wide for this reason (IVX size is accepted in 0.05 ± 0.02, LP coverage in
  [0.85, 0.99]); they catch a broken estimator, not a third-decimal drift.

**Coverage gaps.**

- **Coverage is measured, but not gated.** See the section below for the real
  numbers. There is deliberately no coverage threshold in CI: a percentage gate
  manufactures pressure to write make-work tests, which is the opposite of the
  point. Coverage is used as a finder, and the findings are acted on by hand.
- **No randomized property framework.** Property tests are hand-written with
  seeded generators — there is no `proptest`/`quickcheck` in the workspace, so
  there is no automatic input shrinking or search for adversarial inputs, and
  no fuzzing.
- **The `tsecon-python` binding crate has no Rust tests.** Everything about the
  PyO3 layer is covered from the Python side only.
- **dtype and layout coercion is under-tested.** Non-contiguous (transposed)
  input is exercised, and Python lists are exercised via the GMM callback, but
  nothing in the suite passes `float32`, Fortran-ordered, or masked arrays to
  assert the coercion behavior. If you feed a non-`float64` array, you are
  outside what the tests pin.
- **No network is exercised, by design.** The library ships no data loaders and
  makes no external requests, so there is nothing to test on that front. The two
  published-result replications run against small public datasets committed to
  the repo (`fixtures/ramey_zubairy.csv`, `fixtures/yield_curve_recession.csv`),
  so they are reproduced offline and cannot break on a provider's URL change.
- **Benchmarks compare 25 of 122 functions.** The parity gate covers the unit-root
  tests, the diagnostics, VAR and its IRF/FEVD/Granger, Johansen, the filters,
  the spectra, ridge/elastic-net, and the GARCH family — a broad spot check, not a
  library-wide cross-library audit — that job belongs to the fixtures.

**CI scope.**

- **CI jobs.** Four run on every push: the Rust workspace
  (`fmt` + `clippy -D warnings` + `cargo test --workspace`); the Python wheel
  built and installed on Ubuntu/macOS/Windows; a `mypy --strict` stub check; and
  an **evidence** job that runs both Monte Carlo suites and the cross-library
  parity gate against a release wheel. `docs.yml` adds `mkdocs build --strict`.
  Because the Monte Carlo scripts are seeded and assert their own expectations,
  a *statistical* regression — a test losing its size, an interval losing
  coverage — fails the build rather than quietly rotting in the docs. The
  benchmark harness gates on cross-library **agreement** only; timings are
  reported but never fail CI, since a shared runner cannot measure them
  reliably.
- **Only CPython 3.12 is exercised in CI**, though the wheel is `abi3-py39` and
  the package declares `requires-python = ">=3.9"`. Older interpreters are
  supported by the ABI contract, not by a test run.

---

## 6 · Coverage

Coverage here is a **finder, not a target**. A percentage bought with make-work
tests launders untested risk into a green badge, so there is no threshold in CI
and no badge; the numbers below are reported as measured, and what they *found*
matters more than what they are.

```sh
./scripts/coverage.sh          # Rust, cargo-llvm-cov — ~3m15s
coverage run -m pytest bindings/python/tests && coverage report   # Python, see .coveragerc
```

**Rust** (`cargo llvm-cov --workspace --exclude tsecon-python`):

| | region | line |
|---|---|---|
| workspace | 89.94% | 85.22% |
| excluding `error.rs` | — | **89.56%** |

**33% of all remaining missed lines are `Display::fmt` match arms in `error.rs`
files.** Those are deliberately left: exercising every error's prose would be
make-work, and the error *types* are already asserted by the validation tier.

**Python** (`coverage.py`, branch coverage, over the pure-Python package only —
the compiled `_core` is Rust and is not measurable by coverage.py, so it is
excluded rather than counted as a phantom gap): the `results/*` modules measured
89–100%; `datasets.py` was the weakest at 71%, since only the `local_path=`
parse path had tests and the whole download/cache/digest round-trip did not.

### What it actually found

The useful output was not a percentage — it was **four publicly exported
estimators with zero Rust-side coverage**, whose input guards existed but had
nothing asserting on them:

| Where | Why an untested guard mattered |
|---|---|
| `realized::parkinson` / `garman_klass` | An inverted bar (`high < low`) would sail through `(ln(H/L))²` — the square destroys the sign — returning a finite, positive, wrong variance. The guard rejects it; nothing proved that. |
| `midas::adl_midas` | Parameter ordering `[c, ρ₁..ρ_P, b₁..b_K]` unpinned: a transposed AR/high-frequency block returns a full set of plausible numbers. |
| `panel_lp` jackknife | Silently *replaces* the reported estimates with the Dhaene-Jochmans correction; a sign error yields a complete, wrong impulse response and no error. |
| `VarResults::ma_rep` (`lags == 0`) | Returning zeros instead of `Ψ₀ = I` would silently zero the impact response of every IRF and FEVD built on it. |

`fit_svensson` was a fifth: with exactly four maturities the design is exactly
determined, giving zero residuals and **R² = 1** — a curve that looks flawless
and carries no information.

To be precise about what this was: these guards **already worked**. Coverage did
not find broken code, it found *untested safety nets* — code whose correctness
nothing would notice regressing. That is a real class of risk and a weaker claim
than "we found bugs", and it is worth stating as the former.

42 Rust and 60 Python tests were added against exactly these paths. The
per-crate movement is where the work landed, not the workspace total:
`tsecon-realized` 42.81% → 82.27% line, `tsecon-midas` 69.21% → 75.16%,
`tsecon-termstructure` 77.11% → 82.89%, `tsecon-panel` 77.19% → 81.29%.

Least-covered crates after this pass, i.e. where a future look should start:
`tsecon-connect` 74.67%, `tsecon-midas` 75.16%, `tsecon-longmemory` 76.05%,
`tsecon-ident` 76.58%.

---

## See also

- **[Validation matrix](validation-matrix.md)** — the per-family table: which
  reference, which fixture, which test, which tolerance.
- **[Monte Carlo validation](../examples/monte-carlo.md)** — size, coverage,
  and consistency, with real output.
- **[Frontier Monte Carlo](../examples/monte-carlo-frontier.md)** — LP vs VAR,
  and weak-instrument LP-IV.
- **[CONTRIBUTING](https://github.com/cacoleman16/tsecon/blob/main/CONTRIBUTING.md)**
  — the golden-fixture discipline as a contributor workflow, plus how to add an
  estimator end to end.
- **[`fixtures/README.md`](../../fixtures/README.md)** — what each fixture
  contains and how to regenerate it.
- **[Benchmark harness](https://github.com/cacoleman16/tsecon/blob/main/benchmarks/README.md)**
  — the parity-first rules and the published example runs.
