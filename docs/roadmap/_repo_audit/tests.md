# Repository audit — test-suite health

> **Working document.** One of seven sweeps of the whole-repository audit
> run against `main` at `19d308e` (tsecon 0.8.0). This sweep asks not
> whether the estimators are right — eleven audit rounds did that — but
> whether the suite itself is sound, fast, reproducible and honest.
> Excluded from the published site.

## Scope and method

The unit of examination is the test suite: 1527 collected Python tests in
99 files under `bindings/python/tests/`, 1724 `#[test]` functions under
`crates/*/{src,tests}`, 96 golden fixtures with 77 generators under
`fixtures/`, and the CI workflow. Seven questions, each answered by a probe
script committed under [`lab/audit/repo/tests/`](../../../lab/audit/repo/tests/)
with its output under `out/`:

| # | question | probe | evidence |
|---|---|---|---|
| 1 | Where does the runtime go, and could a Monte-Carlo replication count be trimmed without weakening an assertion? | `run_suite.sh a`, `diff_runs.py durations`, `mc_trim_probe.py` | `out/run_a.log`, `out/durations_a.txt`, `out/mc_trim_probe.txt` |
| 2 | Is the suite reproducible run to run, under `PYTHONHASHSEED=1`, and for the Monte-Carlo-heaviest Rust crates? | `run_suite.sh b`, `run_suite.sh c`, `diff_runs.py diff`, `cargo test --release -p tsecon-ml -p tsecon-var` ×2 | `out/diff_abc.txt`, `out/rust_*.txt` |
| 3 | Can every test fail? Are its assertions numeric, structural, or absent? Is any tolerance looser than the matrix documents? Is any body duplicated? | `assertion_scan.py`, `tolerance_vs_matrix.py`, `tolerance_headroom.py`, `rust_no_assert.py` | `out/assertion_scan.txt`, `out/tolerance_vs_matrix.txt`, `out/tolerance_headroom.txt`, `out/rust_no_assert.txt` |
| 4 | Does every skip / `importorskip` / `#[ignore]` still have a live reason? | `skips_scan.py` + the run logs | `out/skips.txt` |
| 5 | Do the fixtures' recorded reference-library versions match the venv, and does every generator parse, import and say how to run? | `fixture_meta_drift.py` | `out/fixture_drift.txt` |
| 6 | How deep is the coverage of each of the 173 public callables? | `coverage_depth.py` | `out/coverage_depth.txt` |
| 7 | What does CI run that a contributor does not, and vice versa? | `ci_sim/sitecustomize.py`, reading `ci.yml` against `testing.md` / `CONTRIBUTING.md` | `out/run_ci_sim.log` |

**Environment.** A fresh worktree venv (`.venv-wt`: numpy 2.4.6, scipy
1.17.1, statsmodels 0.15.0, scikit-learn 1.9.0, arch 8.0.0, linearmodels
7.0, MAPIE 1.5.0, pandas 3.0.5, polars 1.44.1, matplotlib 3.11.1, pytest
9.1.1, CPython 3.11) with the extension built `maturin develop --release`.
`pytest-randomly` is not installed anywhere in the project, so the
`-p no:randomly` variant the brief names is moot: collection order is
pytest's default file order on every run, and the third run uses
`PYTHONHASHSEED=1` instead. The host is a 4-core container shared with six
sibling audit sessions building and testing at the same time (load average
6–20 throughout), so every second quoted here is inflated by contention;
**the ordering and the shares are the evidence, the absolute seconds are
not.** The Rust runs use `--release` because the maturin build had already
produced the release dependency graph; `cargo test --workspace` in CI is the
dev profile, and the `#[ignore]` timings below are release-profile numbers.

**Discipline.** Every candidate was refuted before it was reported: a test
without `assert` that calls something which raises on failure is a valid
smoke test and is listed as clean; a `< 0.35` on a Monte-Carlo bound is not
a golden tolerance and is not compared to the matrix; a skip that never
fires on seeded data is a dead guard, not a hidden gap. Nothing in the
suite was weakened. The one repair this sweep was allowed to make — removing
a skip whose reason no longer holds — was not needed: no such skip exists.

## Totals

| measure | value |
|---|---|
| Python tests collected / passed / skipped (run a) | 1527 collected, **1526 passed, 1 skipped, 0 failed** in 402.4 s wall; sum of per-test times 395.9 s |
| Python test functions (static, before parametrisation) | 1118 in 99 files: 864 numeric, 143 raises-only, 110 structural-only, **1 with no assertion** (a valid smoke test; clean bills) |
| duplicated test bodies | **0** identical-AST bodies across names or files (15 names are reused across files, every one with a different body) |
| Rust `#[test]` functions scanned | 1724; **0** lack an assertion path, apart from the compile-time `all_types_are_send_sync_clone` and the two `#[ignore]`d utilities (a timing probe, a snapshot emitter) |
| skip-family sites in Python | 77 (68 `importorskip`, 8 `pytest.skip`, 1 `skipif`, 0 `xfail`); **exactly 1 fires** in the full-extras venv, on every run, by design |
| `#[ignore]` sites in Rust | 10 real (the grep also sees 4 doc-comment mentions); every one carries a reason string |
| fixtures with a provenance block | 91 of 96; **80 record at least one reference-library version, 16 record none** |
| generators | 77 of 77 parse; 4 import a package outside the documented reference set (`arviz`, `cvxpy`+`clarabel`, `reservoirpy`, `skglm`); the 2 R-dependent ones say so and give the command |
| public callables exercised through `tsecon.<name>(` | 173 of 173 (the `test_exercise_gap.py` claim holds); strongest Python-tier assertion is a reference match for 157, a property/sign/bound for 13, structural only for 3 |
| run-to-run reproducibility (Python a vs b vs c, Rust ml ×2, var ×2) | **identical**: 0 outcome differences and 0 printed-number differences across the three Python runs; 0 differing outcome lines across the Rust reruns |

## Findings

Severity: **severe** = a test that cannot fail on a golden-claimed surface,
or a nondeterministic pass; **moderate** = a tolerance looser than
documented, a stale skip hiding a gap, a documented claim the suite does not
back; **low** = slow, duplicated, drift that changes nothing.

**No severe finding.** Nothing is nondeterministic (finding 9 records the
measurement), and the one assertion-free test is a valid smoke test.

### 1 · `test_garch_gjr_asymmetry_detected` cannot fail on its claim, and four of the five `garch.json` reference cases have no Python re-check — moderate

[`test_smoke.py:367`](../../../bindings/python/tests/test_smoke.py):

```python
def test_garch_gjr_asymmetry_detected():
    ret = np.array(GARCHFX["returns"])
    r = tsecon.garch_fit(ret, vol="gjr", mean="zero", dist="normal")
    assert "gamma[1]" in " ".join(r["param_names"]).lower() or len(r["params"]) == 4
    assert np.isfinite(r["loglik"])
```

The name claims asymmetry is *detected*; the assertions check that a
parameter is named `gamma[1]` (or that there are four of them) and that the
log-likelihood is finite. Neither can fail on the sign, size or significance
of `gamma`. And the claim is not true of the data: `garch.json`'s
`gjr111_zero_normal` case — loaded into the same `GARCHFX` this test reads —
records `arch`'s fitted `gamma[1] = −0.0169` on these symmetric-GARCH
returns, i.e. no asymmetry to detect. The same fixture case carries
`fit_params`, `fit_loglike`, `fit_bse_robust`, `loglike_fixed` and the
conditional-volatility head/tail, none of which any Python test compares
against. `grep gjr111\|egarch111\|garch11_const\|garch11_zero_t
bindings/python/tests/*.py` returns nothing: the Python tier pins only
`garch11_zero_normal` (`test_smoke.py:356`), so the const-mean, GJR, EGARCH
and Student-*t* legs of the validation matrix's GARCH row are Rust-only.
That is consistent with the tiering `testing.md` describes (the tight pins
live in the crates), but this test's name says otherwise.

*Proposal (not applied — assertions are out of this sweep's remit):* assert
the GJR case's `fit_params` at the matrix's `1e-3` rel, `fit_bse_robust` at
`5e-3` rel and `fit_loglike ≥ arch − 1e-6`, and rename the test to what it
then proves (`test_garch_gjr_matches_arch_fixture`). Parametrising
`test_garch_fit_matches_arch_fixture` over the four normal-innovation cases
does the same for EGARCH and the const-mean case at once.

### 2 · Fourteen Python golden re-checks are looser than the tolerance the validation matrix documents — moderate by definition, with 2–11 orders of headroom measured

`tolerance_vs_matrix.py` joins each test file to the matrix rows whose
fixture it loads and compares the loosest literal in a fixture-referencing
assertion against the loosest number the row quotes. Of 53 raw candidates,
39 were refuted by reading (a bound on a Monte-Carlo rate, a property check,
a `not allclose`, a fixture shared by two rows with different tolerances).
The 14 that stand are all Python re-checks of a value the Rust golden pins
at the documented tolerance. `tolerance_headroom.py` reproduces each one and
measures the achieved error:

| test (line) | asserted | matrix documents | achieved |
|---|---|---|---|
| `test_favar.py:44–45` \|PC1\|, \|PC2\| | `atol=1e-4` | favar row: 1e-6 rel | 1.8e-14, 3.0e-14 |
| `test_depth.py:89–90` \|PC1\|, \|PC2\| | `atol=1e-5` | favar row: 1e-6 rel | 1.8e-14, 3.0e-14 |
| `test_gmm.py:39–40` Hansen `j_stat`, `j_pval` | `< 1e-4` | gmm row: 1e-6 | 4.4e-16, 2.2e-16 |
| `test_predreg.py:40–41` `beta_ivx`, `wald` | `< 1e-6`, `< 1e-5` | predreg row: 1e-9 | 3.5e-17, 3.6e-15 |
| `test_roadmap_gaps.py:31–35` probit params, bse, loglik, probabilities | `atol=1e-5`, `< 1e-4` | recession row: 1e-6 | 2.3e-8, 8.2e-10, 1.3e-13, 1.5e-8 |
| `test_spectest_afns_dsge.py:133` AFNS adjustment | `rtol=1e-9` | afns row: 1e-10 | 1.5e-11 |
| `test_proxy_first_stage.py:90` `tau_bound` | `rel=1e-5` | first-stage row: 1e-6 | 7.1e-15 |

The golden claim itself is intact in every case — the Rust pin is at the
documented value and passes — and the PyO3 boundary is exact, which is why
the Python side lands at 1e-14. What is looser is the "checked a second time
through the Python API" leg that `testing.md` §Tier 4 describes. *Proposal:*
tighten each literal to the matrix value; the measured headroom says every
one passes with at least two orders to spare.

### 3 · `fixtures/README.md` says every fixture records its reference-library versions; 16 of 96 do not, four of them third-party goldens — moderate

> Each fixture records the exact reference-library versions used, so the
> values are reproducible. — `fixtures/README.md`

`fixture_meta_drift.py` reads the provenance block under whichever of the
five spellings the tree uses (`_meta`, `meta`, `_doc`, `_source`, `_note`)
and finds a `library x.y.z` pair anywhere in it, including free text.
Sixteen fixtures record no version at all:

- **no provenance block of any kind (5):** `fry_pagan_svar.json`,
  `historical_decomposition_chol.json`, `long_run_svar.json`,
  `tsecon-gas.json`, `zero_sign_svar.json` — all documented-formula NumPy
  goldens, so the missing record is the NumPy/SciPy version.
- **a block, but no version (11):** `backtest_string_snapshot.json` (a
  self-snapshot; by design), `hetero_svar.json`, `nowcast_mle.json`,
  `nowcast_news.json`, `tsecon-dsge.json`, `tsecon-nowcast.json`,
  `tsecon-panelroot.json`, and four whose reference **is** a third-party
  package: `phillips.json` (its `_meta.note` says "PP stats from
  arch.PhillipsPerron; … statsmodels MacKinnon-N surfaces" — no `arch` or
  `statsmodels` version), `tsecon-survey.json` ("statsmodels OLS
  cov_type=HAC"), `var_irf_bands.json` ("statsmodels IRAnalysis.stderr"),
  `tsecon-recession.json` (a statsmodels probit/logit, per
  `test_recession_probit_matches_statsmodels`).

For the four third-party ones the README's reproducibility promise cannot be
kept: the statsmodels 0.14 → 0.15 change already altered `adfuller`'s
return contract (the ten `FutureWarning`s in `out/run_a.log` are from
exactly that), and a regenerated `phillips.json` on a future `arch` has no
recorded baseline to be compared against. *Proposal:* add
`"arch": arch.__version__` / `"statsmodels": …` to those four generators'
`_meta` and a `numpy`/`scipy` line to the eleven formula goldens. (A
`_meta` **typo** this sweep could have fixed in place was looked for and not
found; the gap is absence, not misspelling, so nothing was edited.)

### 4 · One Monte-Carlo test is 61 % of the suite's runtime, and its file is 77 % — low (slow), with a trim that keeps the assertion

Run a, sum of per-test times 395.9 s:

| share | test | what it does |
|---|---|---|
| **241.4 s (61 %)** | `test_auto_arima.py::test_recovery_small_mc_nonseasonal` | 3 DGPs × 12 replications of `auto_arima(y)` at `T=300` with default caps = 36 full stepwise searches (~21 candidate fits each) |
| 43.8 s (11 %) | `test_auto_arima.py::test_seasonal_co2_monthly_selects_a_seasonal_model` | one seasonal search (`s=12`, `T=240`, `max_p=max_q=2`, `max_P=max_Q=1`) |
| 23.4 s (6 %) | `test_conformal.py::test_conformal_kwargs_still_live_where_documented` | 21 `conformal_forecast`/`conformal_backtest` calls incl. ARIMA bases and EnbPI bootstraps |
| 304.5 s (77 %) | `test_auto_arima.py` (8 tests) | |

The MC test's assertion is `within >= 0.5 * reps` per DGP, with the docstring
saying the threshold sits "well below the measured MC rates"
(`scripts/mc_auto_arima_recovery.py`). The test prints nothing per
replication, so `mc_trim_probe.py` replays the identical loop (same seeds,
same simulator, same call) and records every selection
(`out/mc_trim_probe.txt`):

| DGP | within-one at `reps=12` (bar 6) | at `reps=6`, same first six seeds (bar 3) |
|---|---|---|
| AR(1) φ=0.6 | 10/12 | 5/6 |
| MA(1) θ=0.6 | 11/12 | 6/6 |
| ARMA(1,1) | 12/12 | 6/6 |

Because the bar is a *fraction* of `reps`, halving `reps` to 6 keeps the
50 % bar and the seeds, and at the measured 33/36 = 92 % within-one rate the
false-failure probability at six draws is ~1e-4. What it costs is power
against a *partial* regression: a selector fallen to a 30 % within-one rate
passes a 6-draw bar with probability 0.26 against 0.12 at twelve; a
selector at chance fails both. That trade — ~120 s of every run against a
weaker gate on a slow drift — is the maintainers' to make; the numbers are
here so it can be made on evidence. Caps (`max_p=max_q=2`) cut a single
call from 10.8 s to 6.3 s here but change what is searched, so they are not
proposed. The seasonal CO2 test is a single call and has nothing to trim.
The docstring's phrase "CI-sane recovery gate" is not yet earned at four
minutes; a `slow` marker with `-m "not slow"` in the developer loop would be
honest about it.

Underneath is a number for the performance sweep rather than this one: one
`arima_fit(p=1, d=0, q=0)` at `T=300` costs **318 ms** on the release wheel
(measured under load; `auto_arima` then costs ~0.5 s per candidate). That
is the reason 36 searches take four minutes.

### 5 · The suite's only skip fires on every run: a parametrised case that can never execute — low

`out/run_a.log`: `SKIPPED [1] test_trees.py:86: the forest fixes
min_samples_split at 2`. `test_forest_single_tree_bridge_reproduces_the_tree`
is parametrised over all eight `trees.json` cases; one of them
(`depth3_leaf1_split30`) has `min_samples_split=30`, and `random_forest` has
no `min_samples_split` parameter (`regression_tree` does — stub line 3166
vs 3202), so the case skips itself unconditionally. That is `testing.md`'s
"1 skipped". It hides nothing about the estimator (the case is pinned
through `regression_tree`), but it documents an API asymmetry between the
two tree entry points, and a skip that fires 100 % of the time is a dead
parametrisation. *Proposal:* filter the case out of the `parametrize` list
(or add the knob to `random_forest`).

### 6 · Stale hand counts on the testing pages — low

All verified against this tree:

| page | says | measured |
|---|---|---|
| `testing.md` §state, line 36 | "Of the **9** ignored tests, 7 are in `tsecon-var` … 2 in `tsecon-panel`" | **10** (`tsecon-ml/tests/structured_properties.rs:739 pds_coverage_full_measurement`, added in 0.8.0, is not in the list; the table row above it already says 10) |
| `testing.md` §Tier 4 | "**337** `pytest.raises` assertions" | **475** sites |
| `testing.md` §Tier 4 | "**53** of the 96 fixture JSONs are named by file in the tests" | **63** distinct `*.json` literals |
| `CONTRIBUTING.md` §Running the tests | "across all **41** crates" | **43** directories under `crates/` (the same page's gate table and `testing.md` say 43) |
| `testing.md` §state | "1526 passed, 0 failed, 1 skipped in 474 s" | 1526 / 0 / 1 in 402 s here — **correct** |

The page already warns that hand counts go stale and shows a reproducible
command for the golden/property split; these four are the ones without one.

### 7 · Coverage depth: `box_cox_lambda` is reached by one test, and that test checks a tuple is not coerced — low

`coverage_depth.py` counts, per public callable, the distinct test functions
that call it and the strongest assertion among them. Three callables are
never checked numerically through Python:

| callable | tests | what they assert | Rust pin |
|---|---|---|---|
| `box_cox_lambda` | 1 (`test_coerce.py::test_tuple_options_are_not_treated_as_data`) | that a tuple argument survives coercion | `advisors.json`: llf 1e-13, λ 1e-6 |
| `fvar_scenario` | 2 (`test_exercise_gap.py`) | keys, shapes, an error path — by that file's stated design | `tsecon-funcshock/tests/golden.rs` |
| `summarize` | 2 | that the rendering wraps a result | not a numeric surface |

Twelve more are checked numerically only by a property, sign or bound (no
fixture, no reference library, no closed form in the test):
`bootstrap_indices` (determinism/range), `dfm_news` (adding-up),
`engle_granger` and `ndiffs` (by `test_exercise_gap.py`'s design),
`fry_pagan_svar` (`mt_statistic >= 0`, shapes), `gas_volatility`
("is sane"), `optimal_block_length` (`1 ≤ · ≤ 30`), `phillips_ouliaris`
(`p < 0.10`), `quantile_lp` (convergence flags), `robust_svar_bounds`
(containment), `svensson` (nests Nelson-Siegel), `cv_splits` (leakage
geometry — the right kind of test; no reference exists). Every one has a
Rust golden or property test behind it, so this is depth, not absence; the
one to act on is `box_cox_lambda`, whose only Python test is about argument
marshalling. Forty-three callables are reached by exactly one test function.

### 8 · Four generators need packages the documented reference environment does not have — low

`fixture_meta_drift.py` imports every generator's dependencies statically:
`generate_bayes_fixtures.py` needs `arviz`, `generate_convex_fixtures.py`
needs `cvxpy` and `clarabel`, `generate_neural_fixtures.py` needs
`reservoirpy`, `generate_structured_fixtures.py` needs `skglm`. None is in
the extras, the CI evidence job, or the venv line this audit was given;
`cvxpy`, `reservoirpy` and `skglm` are named in the validation matrix and
the model cards, `clarabel` nowhere in `docs/`. The fixtures themselves are
committed, so nothing fails today; regenerating those four does. The two
R-dependent generators (`bn_filters`, `lpdid`) state the requirement and
the exact command in their docstrings, and `fixtures/README.md` repeats it.
Nine formula-golden generators carry no run line of their own; the README's
generic `.venv/bin/python fixtures/generate_<x>.py` covers them.

### 9 · Reproducibility — clean

Three full Python runs — a (402 s), b (431 s), c with `PYTHONHASHSEED=1`
(388 s) — and two release runs each of the two Monte-Carlo-heaviest Rust
crates (`out/diff_abc.txt`, `out/rust_*_outcomes.txt`):

| comparison | outcomes | printed numbers |
|---|---|---|
| Python a vs b vs c (1527 test ids) | identical: 1526 passed, 1 skipped in each; **0 differences** | the captured stdout of every passing test, timings masked, is identical in all three — including the two Monte-Carlo prints (`panel_lp joint coverage … {'pointwise': 0.305, 'sidak': 0.765, 'bonferroni': 0.765}`, `ensemble beats mean member 10/10, median member 10/10`) |
| `cargo test --release -p tsecon-ml` ×2 | 127 passed, 1 ignored, both runs; **0 differing lines** | — (no `--nocapture`) |
| `cargo test --release -p tsecon-var` ×2 | 81 passed, 7 ignored, both runs; **0 differing lines** | — |

`PYTHONHASHSEED` cannot move a result because no test iterates a set of
floats into an assertion, and every Monte-Carlo test seeds
`default_rng` explicitly (the round-11 seed-contract sweep established
that for the library; this run establishes it for the suite). One
artefact was refuted on the way: the first version of `diff_runs.py`
attributed the `--durations` block to the last captured-stdout section and
reported one "difference" — the slowest-tests ordering, not a test's
output.

### 10 · CI runs a smaller Python suite than a contributor with the extras — recorded as a known gap, with the number

`ci.yml`'s `python` job installs `pytest numpy scipy` and nothing else
before `pytest bindings/python/tests` on Linux, macOS and Windows;
`release.yml:54–58` runs the same reduced step. Every `importorskip` for
statsmodels, arch, scikit-learn, linearmodels, MAPIE, pandas and matplotlib
therefore skips there. `ci_sim/sitecustomize.py` blocks exactly those
imports in the full venv, and a run over the 44 test files that mention any
of them (the other 55 cannot differ) gives:

| | tests |
|---|---:|
| run in the 44 files under the CI environment | 976 passed, **67 skipped**, 0 failed, 0 collection errors |
| of the 67: the design skip of finding 5 | 1 |
| gated on `pandas` | 22 |
| gated on `matplotlib` | 22 |
| gated on `statsmodels` | 17 |
| gated on `arch` | 3 |
| gated on `scikit-learn` | 1 |
| gated on `mapie` | 1 |

So CI's three operating systems run **1460 of the 1526** tests a full-extras
developer runs; the 66 they never see are exactly the cross-library
re-checks (`test_phillips_perron_matches_arch`, `test_var_irf_bands` vs
statsmodels, `test_conformal` vs MAPIE, the pandas/DataFrame marshalling
paths, every `plot_*` test) plus the seasonal CO2 search of finding 4.
Nothing errors at collection without the extras — the `importorskip`
discipline is complete.

`testing.md` says so in its own words ("extras-gated files skip collection
or at runtime without them"), the CI comment claims only that the *wheel*
is under test, and the evidence job separately installs statsmodels, arch
and scikit-learn for `bench.py`'s 65-metric parity gate. So this is a
documented gap, not a finding. What the docs do not say is the size of it,
which is the number above: the cross-library re-checks that CI's three
operating systems never see. `CONTRIBUTING.md`'s local gate also lists
`mkdocs build --strict`, which CI runs in `docs.yml` only when `docs/` or
`mkdocs.yml` change — parity, not a gap.

### Findings at a glance

| # | severity | finding | applied? |
|---|---|---|---|
| 1 | moderate | `test_garch_gjr_asymmetry_detected` cannot fail on its claim; 4 of 5 `garch.json` arch cases have no Python re-check | no — proposal |
| 2 | moderate | 14 Python golden re-checks looser than the matrix (7 files); measured headroom 1e-8…1e-17 | no — proposal |
| 3 | moderate | `fixtures/README.md` promises a recorded reference version in every fixture; 16 of 96 have none, 4 of them third-party goldens | no — nothing to fix in place |
| 4 | low | one MC test is 61 % of the suite (241 s); `reps` 12→6 keeps the assertion | no — proposal |
| 5 | low | the suite's only skip fires on every run (a parametrised case `random_forest` cannot take) | no — proposal |
| 6 | low | four stale hand counts in `testing.md` / `CONTRIBUTING.md` | no — handed to the docs sweep |
| 7 | low | `box_cox_lambda` reached by one test, a coerce test; 12 callables property-only in the Python tier | no — proposal |
| 8 | low | 4 generators need packages outside the documented reference environment | no — proposal |
| 9 | clean | Python suite a = b = c (outcomes and printed numbers); Rust ml ×2 and var ×2 identical | — |
| 10 | known gap | CI's numpy+scipy job runs 1460 of 1526 tests; the 66 it never sees are the cross-library re-checks | — |


## Slowest tests

Run a (`out/run_a.log`, `--durations=40`; `out/durations_a.txt` for the
per-file sums). "Trim" says whether a Monte-Carlo replication count could
be reduced without weakening the assertion; nothing was trimmed.

| s | test | what it does | trim |
|---:|---|---|---|
| 241.4 | `test_auto_arima::test_recovery_small_mc_nonseasonal` | 3 DGPs × 12 `auto_arima` searches, `T=300`; asserts within-one ≥ 50 % per DGP | **yes, with a stated cost** — `reps` 12→6 keeps the 50 % bar (5/6, 6/6, 6/6 measured) and loses power against a partial regression (finding 4) |
| 43.8 | `test_auto_arima::test_seasonal_co2_monthly_selects_a_seasonal_model` | one seasonal search on the CO2 series; asserts `D=1`, a seasonal part, trace-argmin consistency | no — single call |
| 23.4 | `test_conformal::test_conformal_kwargs_still_live_where_documented` | 21 conformal calls, each pair asserted bit-different | no — no replication count |
| 8.8 | `test_auto_arima::test_deterministic_and_trace_refit_consistent` | two `auto_arima` calls + refit; bit-equality | no — the second call *is* the determinism check |
| 7.7 | `test_neural::test_mlp_ensemble_beats_mean_member_always_and_median_member_mostly` | 10 seeds × 9-member MLP ensembles; asserts 10/10 (Jensen) and ≥ 7/10 | no — the 10/10 assertion needs all ten |
| 5.2 | `test_auto_arima::test_ic_variants_and_bic_prefers_smaller` | two searches (`aicc`, `bic`) | no |
| 2.3 | `test_convergence_flags::test_dfm_nowcast_mle_reports_converged_and_iterations` | one DFM MLE fit; type checks | no |
| 2.2 | `test_auto_arima::test_result_keys_are_a_superset_of_arima_fit` | one search + one fit; key set | no |
| 2.0 | `test_simultaneous_bands::test_panel_lp_joint_coverage_pointwise_fails_and_closed_forms_repair_it` | 200-rep panel-LP coverage MC | no — already 2 s; the 0.30 gap and ≥ 0.70 bars need the reps |
| 1.9 | `test_conformal::test_theta_and_arima_bases_run` | base-forecaster plumbing | no |
| 1.8 | `test_auto_arima::test_d_selection_on_a_random_walk_with_evidence` | one search | no |
| 1.8 | `test_mgarch_covariance_surface::test_univariate_and_second_stage_dist_are_independent_knobs` | four DCC fits | no |
| 1.7 | `test_replication_hamilton_markov::test_statsmodels_reaches_the_eviews_benchmark` (setup) | statsmodels MarkovAutoregression fit | no — reference side |
| 1.6 | `test_audit_round11::test_egarch_multistep_forecast_refusal_is_documented_and_clean` | EGARCH fits | no |
| 1.6 | `test_dcc_buildout::test_student_t_second_stage` | DCC-t fit | no |
| 1.6 | `test_convergence_flags::test_auto_arima_shares_the_new_keys` | one search | no |
| 1.5 | `test_copula::test_copula_fit_matches_reference[t_rho05_nu4-t-mle]` | t-copula MLE vs fixture | no |
| 1.5 | `test_copula::test_copula_select_crowns_the_generator_winner[t_rho05_nu4]` | four copula fits | no |
| 1.5 | `test_copula::test_t_tail_dependence_uses_correct_form_not_statsmodels_bug` | t-copula MLE | no |
| 1.4 | `test_dcc_buildout::test_dcc_garch_docstring_names_every_returned_key` | one DCC fit + docstring parse | no |

Per file: `test_auto_arima.py` 304.5 s (8 tests), `test_conformal.py`
26.3 s (34), `test_neural.py` 10.3 s (22), `test_dcc_buildout.py` 9.4 s
(18), `test_copula.py` 7.4 s (71), `test_convergence_flags.py` 4.9 s (13),
`test_mgarch_covariance_surface.py` 3.8 s (20), `test_simultaneous_bands.py`
3.0 s (58); the remaining 91 files total 26.3 s. The suite without
`test_auto_arima.py` runs in ~90 s here.

## Provenance drift

Recorded reference-library version in each fixture's provenance block vs
the audit venv (`out/fixture_drift.txt` has every fixture; this is the
summary). The drift is between the environment the fixtures were
*generated* in and the venv this audit was given, not a defect: the JSONs
are frozen goldens and every Rust and Python golden test passes against
them here. It is worth reading for what it says about *which* environment
would reproduce them.

| library | recorded | installed here | fixtures |
|---|---|---|---:|
| statsmodels | 0.14.6 | 0.15.0 | 41 |
| statsmodels | 0.15.0 | 0.15.0 | 1 (`kernel.json`) |
| numpy | 2.5.1 | 2.4.6 | 37 |
| numpy | 2.4.6 | 2.4.6 | 25 |
| numpy | 1.26.4 | 2.4.6 | 12 (the original set: `arima`, `diagnostics`, `distributions`, `filters`, `forecast`, `hac`, `linalg`, `philox`, `simultaneous`, `ssm`, `unitroot`, `var`) |
| scipy | 1.18.0 | 1.17.1 | 16 |
| scipy | 1.17.1 | 1.17.1 | 20 |
| arch | 8.0.0 | 8.0.0 | 4 (`advisors`, `dfgls`, `garch`, `zivot_andrews`) |
| linearmodels | 7.0 | 7.0 | 4 |
| scikit-learn | 1.9.0 | 1.9.0 | 5 |
| arviz | 1.2.0 | — | 1 (`convergence.json`) |
| cvxpy / clarabel | 1.9.2 / 0.11.1 | — | 1 (`convex.json`) |
| skglm | 0.5 | — | 1 (`structured.json`) |
| *(none recorded)* | | | **16** (finding 3) |

Two things the table shows. First, the generation environment is *newer*
than the setup line this audit pins (37 fixtures record numpy 2.5.1 and 16
record scipy 1.18.0, against 2.4.6 / 1.17.1 installed), so "regenerate and diff" from
the documented venv would produce noise before it produced signal. Second,
statsmodels 0.14.6 is the reference for 41 fixtures and 0.15.0 for one; the
0.15 `adfuller` contract change is the kind of thing that turns a
regeneration into a debugging session, which is why finding 3 asks for the
version on the four fixtures that lack it.

Generators: 77 of 77 parse; 4 need packages outside the venv (finding 8);
2 are R-dependent and say so (`generate_bn_filters_fixtures.py` —
`Rscript` + `$BNFILTER_R_DIR`; `generate_lpdid_fixtures.py` — `fixest`);
1 imports `tsecon` by declared design (`generate_backtest_string_snapshot.py`).

## Coverage depth

`coverage_depth.py`, over the 173 public callables (`out/coverage_depth.txt`):

| strongest Python-tier assertion | callables |
|---|---:|
| reference (fixture value, reference library, or a closed form computed in the test) | 157 |
| numeric property / sign / bound only | 13 |
| structural only (keys, shapes, finiteness) | 3 |
| raises only / none | 0 |
| unexercised | 0 |

| distinct test functions calling it | callables |
|---|---:|
| 1 | 43 |
| 2 | 24 |
| 3–5 | 46 |
| 6–20 | 53 |
| > 20 | 7 (`check_series` 57, `lp` 27, `garch_fit` 27, `dcc_garch` 27, `arima_fit` 24, `var_fit` 23, `backtest` 21) |

The classification is a static heuristic (a hand-typed literal from a
paper reads as "numeric"); every row in the two short lists in finding 7
was read before being reported. `tripower_quarticity` was initially flagged
numeric-only and refuted on reading: the test transcribes the
`μ_{4/3}^{-3}` closed form with `math.gamma` and is a reference check.

## Clean bills

- **Nothing in the suite is assertion-free in a way that matters.** The
  one Python test with no `assert`, `test_check_series.py:420
  test_reports_are_json_serializable`, calls `json.dumps(rep)` on every
  report; `json.dumps` raises on anything unserialisable, so the test fails
  exactly when its name says. On the Rust side the only `#[test]` without
  an assertion path is `tsecon-rng`'s `all_types_are_send_sync_clone`, a
  compile-time trait check that fails at build. The 34 Rust goldens the
  first pass flagged as "unwrap only" all assert through the shared
  `tests/common/mod.rs` helpers (`assert_rel_close`, `assert_mat_close`);
  the probe now resolves those.
- **No duplicated test body** across 1118 functions. The 15 names reused
  across files (`test_is_a_dict`, `test_teaching_errors`, …) are the
  `Results`-facade and per-family conventions, each with a different body.
- **No stale skip.** All 68 `importorskip` sites import in the full venv;
  the eight `pytest.skip` sites are either environment guards that do not
  fire in a checkout (`docs tree not present`, `Rust source not
  available`) or data-dependent guards on seeded draws that never fire
  (`test_garch_boundary.py:85/110`, `test_proxy_svar_bands.py:361` — all
  three tests PASSED in every run); the one `skipif`
  (`test_zivot_andrews.py:28`) guards a fixture that is present. The
  brief's example — an `importorskip("mapie")` where MAPIE is now an
  extra — does not arise: MAPIE is not an extra (`pyproject.toml` declares
  `plots`, `polars`, `all`), and `test_conformal.py:361` gates a genuine
  cross-library check. There was therefore no skip to remove.
- **Every `#[ignore]` states a reason**, and each still holds: three
  bit-pattern fingerprints are platform-specific by construction; the
  snapshot emitter and the timing probe are utilities; the five
  Monte-Carlo measurements are "run in release with --ignored" — measured
  here in release at 36 s (`tsecon-ml::pds_coverage_full_measurement`,
  300 reps × 2 cells, prints PDS coverage 0.950 / 0.903 against oracle
  0.953 / 0.930), 221 s (`tsecon-var::mc_irf_simultaneous_coverage`,
  pointwise joint 70.4 %, sup-t joint 84.8 %) and 205 s
  (`tsecon-var::mc_forecast_simultaneous_coverage`); the `tsecon-panel`
  pair was not timed (outside the two crates the brief named). At
  dev-profile speed (16× on the optimiser paths per `testing.md`) the
  ignore is still the right call for all three. The timing probe
  (`cost_of_the_sup_t_route_at_the_production_default`) runs in 2 s and
  prints; it is a utility, not a test.
  The one that could join the always-on set is the `tsecon-var` sup-t
  timing probe, which is a print, not a test.
- **All 173 public callables are exercised** through `tsecon.<name>(` —
  `test_exercise_gap.py`'s closing of the last three names holds on this
  tree.
- **The 1526/0/1 count `testing.md` publishes is what this tree produces**,
  and the 474 s it quotes is within contention noise of the 402 s here.

## Open

- **Not done — dev-profile timing of the `#[ignore]`d Monte-Carlo tests.**
  Only release-profile numbers were taken (the dev-profile dependency
  graph would have been a second full compile on a saturated host). The
  brief's test — "a run-in-release ignore whose test now takes 3 s in
  debug" — is therefore answered for release only.
- **Not done — the CI simulation over all 99 files.** It ran over the 44
  files that mention an extra (the other 55 cannot change outcome without
  one) and with the 241-s MC test deselected, so its count is exact for
  skips and errors but its wall time is not comparable to run a.
- **Handed to the performance sweep:** `arima_fit` at `T=300` costs ~300 ms
  and a default `auto_arima` ~10 s on the release wheel under load. That is
  the root of finding 4 and is a runtime, not a suite, question.
- **Handed to the docs sweep:** the four stale counts in finding 6.
- **Not attempted:** macOS / Windows / Python 3.9 and 3.13 legs of CI;
  `pytest-randomly` (not installed, not adopted); coverage measurement of
  the Rust crates by line (`cargo llvm-cov`) — the brief asked for
  assertion depth, not line coverage.
