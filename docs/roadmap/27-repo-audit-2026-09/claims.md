# Repository audit — claims sweep (round 12)

> **Working document.** One of seven parallel sweeps of the whole repository
> "to date", run under [the brief](../16-adversarial-audit-brief.md) against
> the installed 0.8.0 wheel built from `origin/main` at 19d308e. Excluded from
> the published site (the `roadmap/` rule in `mkdocs.yml`).

Round 10 found 34 stale claims before 0.7.0 ("the documentation was lying").
Two releases later this sweep repeats that pass end to end and widens it:
every identifier the docs name, every count and version they state, every
"not implemented / roadmap" denial, every fenced `python` block with a pasted
output, the three most recent CHANGELOG sections, and the JOSS draft.

**Totals: 25 confirmed findings (2 severe, 9 moderate, 14 low), all fixed
in-branch except the four recorded under *Open*; 5 clean bills.** The severe
pair share one cause: pages that were correct when written and were never
re-read after a later release moved the ground under them (the 0.8.0
machine-learning wave under guide chapter 12; the 0.5.0 HAR-window
correction under guide chapter 6).

---

## Scope and method

Everything below was probed on the wheel built in this worktree
(`maturin develop --release`, tsecon 0.8.0, 173 public callables) with the
scripts under [`lab/audit/repo/claims/`](../../../lab/audit/repo/claims/);
their logs are committed under `out/`.

| Sweep | Script | What it does |
|---|---|---|
| Names | `sweep_names.py` (+ `collect_keys.py`, `registry_ext.py`, `probe_phantoms.py`) | Every backticked identifier, `tsecon.<name>` mention and `<name>(kw=...)` call form on 82 pages, classified as a public callable (A), a kwarg of the function the paragraph is about (B, via `inspect.signature`), a returned key of that function (C, by calling it on the round-11 registry input, extended to the eleven 0.8.0 callables), a results-layer name (R), a Rust-side symbol (X), a name that exists elsewhere in the library (W) or a phantom candidate (P). Migration tables are read column-aware. Every P and every W with an explicit function attribution was reviewed by hand; the non-default-mode key claims (bands, `method=`, pooled variants, 2-D inputs) were re-probed in those modes. |
| Counts and versions | `sweep_counts.sh` | Recomputes every number the repo states about itself with the repo's own commands (the rules `testing.md` documents) and every version string. |
| Denials | `sweep_denials.py` | Every "not implemented / roadmap / does not ship / R has, tsecon does not" sentence that can be checked against the 173-callable surface, plus the runtime advisory strings in `_inspect.py`. |
| Executable claims | `sweep_exec.py` + `exec_runner.py` | Every fenced `python` block on the README, `docs/index.md`, the quickstart, the 12 cookbook pages, the 15 guide chapters, the 35 model cards and `results.md`, run sequentially per page in one namespace from the repository root; where a pasted-output block follows, printed output compared token by token, numbers at the printed precision. `# -> value` trailing comments were also evaluated (advisory: the comparator is strict and was normalised by hand). |
| CHANGELOG | `sweep_changelog.py` | 54 checks over every 0.8.0 / 0.7.0 / 0.6.0 bullet that names a function, a kwarg, a refusal or a returned key. |
| Paper | `sweep_paper.py` | Every quantitative claim in `paper/paper.md`, every function it names, every `@key` against `paper.bib`. |

Discipline: finder/refuter. A count dated to a release is not stale; a kwarg
that belongs to statsmodels in a migration table is not a tsecon phantom; a
pasted number that differs in floating-point noise is cosmetic. Eleven
candidate failures in the CHANGELOG sweep and about a dozen in the names sweep
were refuted on re-reading (a probe bug, a hypothesis name read as a key, a
math symbol read as a kwarg) before anything below was recorded.

## Totals per class and severity

| Class | Candidates | Confirmed | Severe | Moderate | Low |
|---|---|---|---|---|---|
| 1 · Names (phantom callables / kwargs / keys) | 12 960 records; 1 240 P + 725 W reviewed | 2 | 0 | 2 | 0 |
| 2 · Counts and versions | 41 numbers, 9 version strings | 9 | 0 | 3 | 6 |
| 3 · Denials of shipped capability | 42 sentences | 2 | 1 | 1 | 0 |
| 4 · Executable claims | 319 blocks, 89 with pasted output | 9 | 1 | 1 | 7 |
| 5 · CHANGELOG 0.8.0 / 0.7.0 / 0.6.0 | 54 checks | 0 | — | — | — |
| 6 · JOSS draft | 7 numbers, 132 names, 18 citations | 2 | 0 | 0 | 2 |
| **Total** | | **25** (+1 open) | **2** | **9** | **14** |

Severity follows the brief: severe = denies a shipped capability or a pasted
output wrong in a way that changes the conclusion; moderate = a phantom
function/kwarg/key or a wrong count on a page a newcomer reads; low =
cosmetic drift.

---

## Findings

### F1 · SEVERE — guide chapter 12 denied the entire 0.8.0 machine-learning wave (fixed)

`docs/guide/12-machine-learning.md` was last revised when Tier 1 of Module 10
shipped and never re-read after 0.8.0 added eleven Tier-2 callables. It said,
in six places, that shipped functions did not exist:

| line (at 19d308e) | text | reality on 0.8.0 |
|---|---|---|
| 223–228 | "`group_lasso` … is on the roadmap; the call below shows the intended API, not a shipped function" | `tsecon.group_lasso` ships (group and sparse-group LASSO with a KKT certificate) |
| 301 | "No tree learner ships in tsecon today" | `regression_tree`, `random_forest` (iid / block / stationary resampling, quantile forests) |
| 344 | "no neural estimator is callable in tsecon today" | `mlp_regression`, `echo_state_network` |
| 360 | "no importance, Shapley, PD or ALE function is in the Python API today" | `random_forest(importance="block_permutation", importance_groups=, permutation_block=)` returns grouped block-permutation importance |
| 372 | "The remaining previews (`group_lasso`, `factor_forecast`) are still on the roadmap" | only `factor_forecast` is |
| 595–597 | "what is not listed above is not yet written in Rust either … Still absent: group and sparse-group LASSO, native random forests with block resampling, componentwise boosting, GBT adapters, PDS-LASSO, dependence-aware interpretation (block permutation …)" | all five shipped in 0.8.0 (`boosting`, `pds_lasso` included) |

The preview code block also *ran* against the wheel and failed only because
its imagined signature (`groups="lag-block"`) is not the shipped one — the
denial sweep's `DENIED-BUT-SHIPPED` row and the exec sweep's
`ERROR [preview]` row are the same finding from two sides. **Fix:** the six
passages rewritten to name the shipped calls and their honest grades, the
preview replaced by a running `group_lasso` block, and the "What tsecon
implements today" section given the Tier-2 list; the "Still absent" list now
names only what is absent (GBT adapters, Shapley/ALE, DML, the GLP sparsity
diagnostic, the desparsified LASSO, the macro random forest, the contamination
harness). Verified by the denial sweep (6 rows) and by re-running the chapter
(all 8 pasted-output blocks match, the new block runs).

### F2 · SEVERE — the HAR-RV worked example in guide chapter 6 stated a conclusion the current build contradicts (fixed)

`docs/guide/06-volatility.md` lines 391–398 quoted
`fit["params"] # [0.635, 0.169, 0.179, 0.398]` and
`fit["tvalues"] # [3.17, 4.52, 2.12, 3.63] — all three horizons load`, then
concluded "All three horizon coefficients are individually significant — the
signature HAR result". On the installed wheel the same block gives
`[0.638, 0.114, 0.174, 0.457]` and `[3.17, 3.08, 1.86, 4.33]`: the weekly
coefficient's *t* is 1.86, not significant at 5%. The numbers predate the
0.5.0 correction of `har_rv`'s windows to Corsi's inclusive definition (the
0.6.0 CHANGELOG re-derived the BNS numbers on this page but not these).
Deterministic (fixture data). **Fix:** numbers refreshed and the sentence
rewritten to the honest reading (daily and monthly significant, weekly
borderline), with a one-line note on why the earlier revision differed. The
inline-comment checker caught it; the R² line (0.144, 577) and the log-variant
R² (0.29) were right.

### F3 · MODERATE — a model-card worked example still used the pre-0.6.0 `var_fevd` layout and now crashes (fixed)

`docs/reference/model-cards/structural-identification.md` line 1294: the
`structural_fevd` example checks its `impact=None` case against `var_fevd`
with `np.transpose(fevd, (1, 0, 2))[:, :12, :]`, i.e. the variable-major
layout 0.6.0 replaced, and raises `ValueError: operands could not be broadcast
together with shapes (3,12,3) (12,3,3)` while the pasted output says
`matches var_fevd: True`. The 0.6.0 CHANGELOG says guide and migration
examples were re-run under the new indexing; this card was missed. **Fix:**
`np.allclose(fevd[:12], vf)` (both arrays are `[h][variable][shock]`;
verified `True` on the wheel, and the alternative alignment `fevd[1:]` is
`False`), comments corrected.

### F4 · MODERATE — quickstart counts and API table stopped at 162 (fixed)

`docs/quickstart.md` printed `# 162` for the callable count and headed the
API table "The 162 functions"; the README, `docs/index.md`, `testing.md` and
the wheel say 173. The "Regression, machine learning, and GMM" table lacked
all eleven 0.8.0 functions. **Fix:** 173 in both places; eleven rows added.

### F5 · MODERATE — the model-card index was missing six cards and most of one family (fixed)

`docs/reference/README.md` promises "one card per method family" and listed
29 of the 35 cards: `copulas.md` and the five ML cards (`ml-convex`,
`ml-kernel`, `ml-neural`, `ml-structured`, `ml-trees`) were absent (all six
are in the mkdocs nav). Its "Cointegration & regimes" row named three of the
fourteen functions the card covers. The page also called `testing.md` "nine
tiers" (it has ten). **Fix:** six rows added, the regimes row completed,
"ten tiers".

### F6 · MODERATE — `CITATION.cff` cited 0.7.0 (fixed)

`version: 0.7.0`, `date-released: 2026-08-28` on a tree whose workspace,
`pyproject.toml`, wheel and CHANGELOG all say 0.8.0 (2026-09-03). The README
routes "please cite" through this file. **Fix:** 0.8.0 / 2026-09-03.

### F7 · MODERATE — `ng_perron(y, regression=…)` is a phantom kwarg in two guide chapters (fixed)

`docs/guide/01-foundations.md` line 342 and
`docs/guide/02-exploration-and-diagnostics.md` line 514 document
`ng_perron(y, regression="c"|"ct")`; the parameter is `trend` (the migration
tables have it right). A reader copying the guide gets `TypeError`. The only
phantom kwarg the names sweep found with an explicit function attribution.
**Fix:** `trend=` in both.

### F8 · MODERATE — `n_unstable` is a phantom key on the DSGE card (fixed)

`docs/reference/model-cards/dsge.md` line 115: "the count above the line is
`n_unstable`, and it should equal your jump count". `dsge_solve` returns
`g`, `p`, `q`, `eigenvalue_moduli`, `verdict` — no such key (the count lives
inside the `verdict` string). The only phantom returned-key claim the sweep
confirmed after re-probing every non-default mode (bands, `identification=
"sign"`, `pooled=`, per-test `panel_unit_root` extras, 2-D `check_series`
reports all carry the keys the cards name). **Fix:** wording.

### F9 · MODERATE — `testing.md`'s measured table drifted on seven counts (fixed)

Recomputed with the page's own rules: golden fixtures 91 → **96** JSON files,
72 → **77** generator scripts (the same page's Tier-4 paragraph already said
96); "Of the 9 ignored tests … 7 in `tsecon-var`, 2 in `tsecon-panel`" while
the table above it says 10 — the tenth is `tsecon-ml`'s 600-replication PDS
coverage measurement; 67 → **72** `*golden*.rs` files and 57 → **62**
`*propert*.rs` files (the *test* counts in the same sentences, 316 / 647 /
111, were right); 337 → **475** `pytest.raises` calls; "53 of the 96 fixture
JSONs" → **63** (rule now stated: `*.json` literals in the test files that
name an existing fixture, deduplicated); "38 of the 99 files … the 55 not
listed" → **61**; "77 estimator-family rows" in the validation matrix → **88**.
The headline counts the page is quoted for — 1775 Rust / 1526 Python tests,
43 crates, 173 functions — reproduce (static `#[test]` 1489 + 242 = 1731 with
10 `#[ignore]`; pytest collects 1527 = 1526 + 1 skipped).

### F10 · MODERATE — `check_series`' NaN refusal told users Kalman imputation was "on the Module 01 roadmap" (fixed, with a regression test)

`bindings/python/python/tsecon/_inspect.py` line 126: "Impute or trim the
gaps first (state-space/Kalman imputation is on the Module 01 roadmap)".
`local_level_smooth` has accepted NaN as missing and returned the smoothed
level through the gaps since it shipped (its docstring says so; verified on a
three-gap series, every returned array finite). The battery itself
legitimately refuses to impute; the message denied the route that exists.
Same class as round 10's seasonal-ARIMA denial, smaller blast radius.
**Fix:** the message now routes to `local_level_smooth`;
`test_nan_refusal_routes_to_the_shipped_imputer_not_a_roadmap` pins the route,
pins "roadmap" out of the text, and runs the imputer on the refused input.

### F11 · LOW — hierarchical-BVAR pasted outputs drifted at the third decimal (fixed)

`docs/guide/10-bayesian.md` and `model-cards/bayesian.md`: pasted
`lambda1 0.1942 / 0.3058 / gain 3.564 / evaluations 81`, wheel gives
`0.1944 / 0.3032 / 3.563 / 82` (deterministic across runs and across the two
pages' separate processes; the log-ML values agree to four decimals, the
objective is flat at the optimum). The prose "0.31" became "0.30"; the
conclusion is unchanged. Refreshed.

### F12 · LOW — validation-matrix reload count (fixed)

"the Python binding tests reload for 58 of the 91 files" (twice) → 63 of
96; the page's list of the 33 fixtures never opened from Python is still
exactly the set difference on this tree.

### F13 · LOW — the JOSS draft undercounts HAC consumers and omits the wave it describes (fixed)

`paper/paper.md`: "eighteen of the crates consume" the HAC engine — nineteen
declare `tsecon-hac` (the 0.8.0 CHANGELOG records `tsecon-ml` joining). The
Functionality section, which says it describes 0.8.0, listed only the Tier-1
penalized solvers under machine learning; the eleven 0.8.0 functions are now
named. Every other number (173, 43, 99 files, 1479 / 242 / 54 / 1775 / 10)
reproduces; all 18 citation keys resolve in `paper.bib`, none is uncited.
The `date:` field predates the artifact, as the YAML comment already says.

### F14 · LOW — README benchmark ratios were from an earlier run (fixed)

"ADF is ~13× and VAR(2) ~24× faster than statsmodels — and GARCH QMLE is ~4×
slower than `arch`". `benchmarks/README.md`'s published release run:
11.09×, 21.46× and 0.41× (about 2.4× slower). Refreshed to the published
numbers.

### F15 · LOW — two ROADMAP §0 sentences contradicted the rest of the tree (fixed)

"two replications of published results" (the gallery holds eight, and §0 says
so three paragraphs later), and "the built-but-unbound backlog is otherwise
drained" while guide chapters 2, 4, 7 and 11 correctly list `select_order`,
`forecast_cov`, `companion`/`roots_moduli`, `ewc_lrv`/`ewc_default_b`,
`fit_css` and `adl_midas` as Rust-only (all present in the crates, none
bound). Both sentences corrected.

### The rest, one line each (all fixed)

| # | Sev. | Where | Claim | Reality |
|---|---|---|---|---|
| 16 | low | `CONTRIBUTING.md:103` | "across all 41 crates" | 43 |
| 17 | low | `docs/reference/results.md:13` | "Six families … have hand-written summaries" | seven (`CheckSeriesResults.summary()` too) |
| 18 | low | `model-cards/var-svar.md:193` pasted keys | `var_irf_bands` key list without `bias_correct` | key present |
| 19 | low | `cookbook/results-table-export.md:10` pasted output | "(3 fields)" and no `se_method` line | `summarize(lp)` prints 4 fields incl. `se_method lag_augmented` |
| 20 | low | `cookbook/sign-restricted-svar.md:11` pasted output | code prints the acceptance rate, output omits it | `acceptance rate: 0.4237` |
| 21 | low | `guide/08:327` pasted output | adding-up residual `3.1086e-15` | `3.5527e-15` (rounding noise; refreshed) |
| 22 | low | `guide/08:349` pasted output | omits the block's last print | `no narrative == sign_restricted_svar: True` |
| 23 | low | `model-cards/structural-identification.md:1575` pasted output | omits the block's last print | `unrestricted shock 0 is NaN: True` |
| 24 | low | `model-cards/structural-identification.md:1714` | "(`rate` 0.326)" | the key is `diagnostics["narrative_acceptance_rate"]` |
| 25 | low | `model-cards/bayesian.md:319` | `inclusion_prob_cov` "appears only with `ssvs_cov=True`" | read charitably it is right (kwarg exists; key absent by default) — recorded as refuted, kept here so the next sweep does not re-find it |

---

## Executable-claims table

319 fenced `python` blocks on 55 pages were executed; 89 carry a pasted
output block. Tally on the tree as merged: **80 matched, 8 mismatched, 1
skipped, 9 errors**; the 221 blocks without a pasted output ran clean apart
from the errors listed. After the fixes every pasted output on the nine
edited pages matches (43/43 on re-run; the one block whose line moved is
`structural-identification.md:1576`). Blocks that ran clean without a pasted
output are summarised per page below the table.

| page | block line | status | note |
|---|---|---|---|
| `README.md` | 61, 81 | NO-OUTPUT | inline `# -> "UnitRoot"` / `"Stationary"` and `# 0.8.0` / `# 173` comments verified |
| `docs/quickstart.md` | 28 | NO-OUTPUT | `# 173` inline comment verified (after F4) |
| `docs/quickstart.md` | 54 | MATCH | |
| `docs/quickstart.md` | 369 | MATCH | |
| `docs/quickstart.md` | 428 | SKIP | needs `my_macro_panel.csv`, the reader's own file |
| `docs/cookbook/fiscal-multiplier.md` | 13, 60 | MATCH | |
| `docs/cookbook/garch-fat-tails.md` | 12, 53, 80 | MATCH | |
| `docs/cookbook/growth-at-risk.md` | 11, 53, 78 | MATCH | |
| `docs/cookbook/hac-standard-errors.md` | 16, 66, 88 | MATCH | |
| `docs/cookbook/panel-lp-standard-errors.md` | 10, 61, 91 | MATCH | |
| `docs/cookbook/results-table-export.md` | 10 | MISMATCH → MATCH | #19 |
| `docs/cookbook/results-table-export.md` | 44, 67, 101, 125 | MATCH | |
| `docs/cookbook/screen-a-series.md` | 12, 61 | MATCH | |
| `docs/cookbook/sign-restricted-svar.md` | 11 | MISMATCH → MATCH | #20 |
| `docs/cookbook/sign-restricted-svar.md` | 66 | MATCH | |
| `docs/cookbook/structural-breaks.md` | 11, 39, 66 | MATCH | |
| `docs/cookbook/unit-root-test.md` | 11, 53, 70 | MATCH | |
| `docs/cookbook/var-forecast-intervals.md` | 10, 61 | MATCH | |
| `docs/cookbook/var-irf-bands.md` | 11, 59 | MATCH | |
| `docs/guide/02-exploration-and-diagnostics.md` | 194 | MATCH | |
| `docs/guide/03-inference-toolkit.md` | 268 | ERROR | preview-labelled (`ewc_default_b`), expected |
| `docs/guide/04-univariate-models.md` | 163, 368, 430 | MATCH | |
| `docs/guide/05-forecasting.md` | 252 | MATCH | |
| `docs/guide/05-forecasting.md` | 349 | ERROR | preview-labelled (typed `bt` object), expected |
| `docs/guide/06-volatility.md` | 384 | NO-OUTPUT | inline comments wrong — F2 |
| `docs/guide/07-multivariate.md` | 209 | MATCH | |
| `docs/guide/08-causal-identification.md` | 174, 284, 303, 394, 459, 516, 557, 633 | MATCH | |
| `docs/guide/08-causal-identification.md` | 327 | MISMATCH → MATCH | #21 |
| `docs/guide/08-causal-identification.md` | 349 | MISMATCH → MATCH | #22 |
| `docs/guide/10-bayesian.md` | 268 | MISMATCH → MATCH | F11 |
| `docs/guide/10-bayesian.md` | 311 | MATCH | |
| `docs/guide/11-nowcasting.md` | 201, 302, 371 | MATCH | |
| `docs/guide/12-machine-learning.md` | 209, 255, 384, 417, 456, 497, 525 | MATCH | |
| `docs/guide/12-machine-learning.md` | 227 | ERROR → runs | the `group_lasso` "preview" — F1 |
| `docs/reference/model-cards/bayesian.md` | 91 | ERROR | fragment (`post = tsecon.bvar_fit(Y, …)` with no `Y`), not standalone |
| `docs/reference/model-cards/bayesian.md` | 213 | MISMATCH → MATCH | F11 |
| `docs/reference/model-cards/bayesian.md` | 357 | MATCH | |
| `docs/reference/model-cards/check-series.md` | 293, 315 | MATCH | |
| `docs/reference/model-cards/local-projections.md` | 288 | ERROR | fragment (`lp_multiplier(y, g, news, …)`), not standalone |
| `docs/reference/model-cards/panel.md` | 380 | MATCH | |
| `docs/reference/model-cards/structural-identification.md` | 120, 210, 312, 443, 592, 845, 977, 1115, 1390, 1479, 1679 | MATCH | |
| `docs/reference/model-cards/structural-identification.md` | 1294 | ERROR → MATCH | F3 |
| `docs/reference/model-cards/structural-identification.md` | 1575 | MISMATCH → MATCH | #23 |
| `docs/reference/model-cards/structural-identification.md` | 1646 | ERROR | a restriction-schema listing fenced as `python` (`"+"\|"-"`), not code |
| `docs/reference/model-cards/unit-root-cointegration-tests.md` | 101, 211, 296, 371, 471 | MATCH | |
| `docs/reference/model-cards/var-svar.md` | 193 | MISMATCH → MATCH | #18 |
| `docs/reference/model-cards/var-svar.md` | 465 | MATCH | |
| `docs/reference/results.md` | 20 | MATCH | |
| `docs/reference/results.md` | 87, 138 | ERROR | fragments (`data`, `r` undefined), illustrative |

Blocks without a pasted output that ran clean, by page group: README 2,
`docs/index.md` 0 (no code), quickstart 2, cookbook 0 (every recipe block has
a pasted output), guide chapters 1–15 137 (including all of chapter 12's after
the fix), model cards and `results.md` 80. No block timed out (per-block
budget 300 s; the slowest block, guide chapter 8's narrative SVAR at 2000
draws, took 31 s).

The inline `# -> value` comments (206 checked) produced 66 strict mismatches;
all were read by hand: one is F2, two are artifacts of the checker (the
comment refers to a variable the block later rebinds — guide 7's two Granger
tests, guide 6's jump-day `day[40] += 2.0`), the rest are prose after the
number, quote style, or `np.float64(...)` reprs. Blocks needing optional
extras: none (statsmodels and matplotlib were present in the venv; nothing
imported anything else).

## Clean bills

- **CHANGELOG 0.8.0 / 0.7.0 / 0.6.0** — 54 of 54 checks pass: every named
  function exists, every named kwarg is in the signature, every documented
  refusal fires with the documented wording (the round-10 inert-kwarg
  sentinels on `ccc_garch`/`dcc_garch`/`dcc_test`, the seven conformal
  guards, `hamilton_filter`, `bn_filter`, `backtest(period=)`,
  `spread_zscore(dt=)`, `threshold_vecm(n_grid_beta=)`,
  `vecm(first_season=)`, the proxy-family inf refusal, `star_test`'s panic
  fix), every documented key is returned (the 0.6.0 binding-gap keys on
  `dfm_nowcast`/`bvar_fit`/`var_fit`/`dcc_garch`, `params_named`,
  `converged` flags, `det_coef_coint`, `evec`, `level`, `iterations`, `ar`),
  and the behavioural identities hold (`random_forest(bootstrap="none",
  max_features="all", n_trees=1)` reproduces `regression_tree` bit-for-bit,
  `l1_trend_filter(penalty="l2")` reproduces `hp_filter` at 1e-8,
  `group_lasso(l1_ratio=1)` reproduces `lasso`, `johansen` ↔ `vecm("co")` β
  at cosine 1, the purged-k-fold gap is `purge + embargo`). Eleven candidate
  failures were probe bugs, refuted before recording.
- **Names** — all 5 454 callable mentions resolve; no phantom callable
  anywhere in the doc set; every migration-table "Roadmap" marker (A/B SVAR,
  ETS, VARMAX/ARDL, RUR/Leybourne-McCabe, `seasonal_decompose`, SV priors,
  mixed-frequency DFM) is confirmed absent; every "R has / statsmodels has"
  gap list is accurate.
- **Denials** — 35 of 42 checked denials are true today (EWC in Python,
  `arima_fit(exog=)`, CSS, `select_order`, Johansen p-values,
  Toda-Yamamoto, ADL-MIDAS, conditional GW, Bai-Perron partial model and
  heterogeneity-robust CIs, `bn_filter` dynamic demeaning, compact kernels,
  rotated copulas, LP-DiD covariates/IV, PMG Hausman, CD/CIPS/PANIC, Tsay,
  GIRF, Bayesian joint bands, multitaper/phase, Chen-Liu outliers); the
  `check_series` seasonality advice routes to `arima_fit(seasonal=…)`,
  `mstl` and `stl` as round 10 required.
- **Counts that reproduce** — 173 callables (registry 173/173, all callable
  on their canonical input), 43 crates every one with `tests/`, 108 k Rust
  source lines ("~100,000"), 1527 collected Python tests in 99 files
  (1526 + 1 skipped), 213 results-facade tests in 9 files, 51 replication
  tests in 9 files, 8 coverage modules / 63 surfaces / 35 functions, 25
  benchmark operations, 15 guide chapters, 12 recipes, 8 replication pages
  and 8 scripts, 35 model cards all in the nav, `generate_*.py` importing
  `tsecon` exactly as `testing.md` describes, toolchain 1.97.1 / MSRV 1.85 /
  `maturin>=1.14,<2.0` / `numpy>=1.22` / `>=3.9` as documented.
- **JOSS draft** — 18/18 citation keys resolve, 0 uncited entries, 120 of
  132 backticked names are public callables and the other 12 are packages.

## Open

- `testing.md` "290 tests across 39 crates load `fixtures/*.json`" — no
  reproducible counting rule is stated (the static `*golden*.rs` count is
  316); left as written, flagged for the next revision to state the rule.
- Two model-card blocks and two `results.md` blocks are fragments that
  cannot run standalone (`bayesian.md:91`, `local-projections.md:288`,
  `results.md:87/138`), and `structural-identification.md:1648` is a schema
  listing fenced as `python`; none claims to run, none was changed.
- `guide/09-local-projections.md:388` prints a round-off residual
  (`5.6e-17` pasted, `1.7e-16` measured) in a `# ->` comment; cosmetic,
  left.
- The strict inline-comment comparator in `exec_runner.py` needs a
  "last-binding" rule before it can be trusted unattended; its 66 flags were
  triaged by hand this round.
- Not reached in the time budget: the notebooks under `notebooks/` (three
  `.ipynb` files with pasted outputs), and the examples' `showcase*.py`
  scripts' printed figures were not re-rendered.
