# Repository audit — hygiene and dead weight

**Baseline:** `origin/main` at 19d308e (tsecon 0.8.0, 2026-09-03). Branch `audit12/hygiene`.
**Sweep:** the repository *as an artifact* — what is tracked, what is dead, what is
duplicated — not estimator behaviour (rounds 1–11 covered that).

## Scope and method

Every number below was produced by a command run in the worktree; the command is
quoted next to the number so it can be re-run.

| Area | Method |
|---|---|
| Top-level inventory | `git ls-tree HEAD`; per entry `git ls-files -z -- <p> \| xargs -0 du -ch`, `git log -1 --format=%cs -- <p>`, and `git grep` of the path across README, ROADMAP, CONTRIBUTING, CHANGELOG, `mkdocs.yml`, `.github/`, `docs/`, `pyproject.toml`. |
| Tracked-but-shouldn't-be | `git ls-files \| grep -E '__pycache__\|\.pyc$\|^site/\|target[^/]*/\|\.ipynb_checkpoints\|\.log$\|\.html$\|…'`; 30 largest via `git ls-files -z \| xargs -0 ls -l \| sort -k5 -n \| tail -30`; `.gitignore` probed with `git check-ignore --no-index -v` on 16 representative paths. |
| Fixtures | For each of 96 JSON + 11 CSV: `_meta` presence via `json.load`; literal-filename references counted with `git grep -l -E '["'\''`/ ]<name>' -- ':(glob)crates/**/tests/**'`, `':(glob)crates/**/src/**'`, `':(glob)bindings/python/tests/**'`, `':(glob)fixtures/generate_*'`, `':(glob)docs/**'`. (A plain `'crates/*/tests'` pathspec silently matches nothing under `git grep`; the `:(glob)` form is required, and `var.json` must be delimiter-anchored so it does not match `favar.json`.) Generator→JSON mapping by `grep -oE '[A-Za-z0-9_-]+\.json'` over every `generate_*`. |
| Dead code | Script over `git ls-files crates bindings/python/src` (`*.rs`): every `pub fn/struct/enum/trait/type` in `crates/*/src` (1,411 items) counted by `\bname\b` occurrences across the whole workspace, split into own-`src` / own-`tests` / other crates / bindings, minus the definition line. Candidates were then refuted by hand: doc comments, `#[cfg(test)]` modules, `#[doc(hidden)]`, `.pyi`, docs. The three `#[allow(dead_code)]` in library code were verified stale by deleting them and running `cargo clippy -p tsecon-ident --all-targets -- -D warnings` (exit 0, 3m24s, `CARGO_TARGET_DIR=$PWD/target-wt CARGO_PROFILE_DEV_DEBUG=0`); the file was restored afterwards. |
| TODO inventory | `git grep -n -I -E '\b(TODO\|FIXME\|XXX\|HACK)\b\|unimplemented!\(\|todo!\('` over `*.rs *.py *.md *.yml *.yaml *.toml *.pyi`; each hit dated with `git blame -L n,n --porcelain`; staleness checked against the shipped `.pyi` and `crates/*/src`. |
| Docs reachability | `mkdocs.yml` nav entries vs `git ls-files 'docs/**/*.md'` minus `exclude_docs` (`roadmap/`, `demo/`, `_hooks/`); internal links in the 50 non-mkdocs Markdown files (top-level, `notebooks/`, `fixtures/`, `benchmarks/`, `lab/`, `paper/`, `docs/roadmap/`) resolved against the working tree after stripping fenced code. |
| Duplication | `md5sum` and per-function name counts over the 13 `tests/common/mod.rs`; `const`/`static` names defined in >1 crate; numeric-table lines identical across crates; shared normalized lines (>40 chars) between README, `docs/index.md`, `docs/guide/README.md`, `docs/reference/README.md`, ROADMAP. |

Not done (see *Open*): the 50 `pub(crate)` candidates were not compile-verified one
by one; `mkdocs build --strict` was not re-run locally (mkdocs is not installed in
this container — CI runs it); the 176 crate-private `pub struct/enum` were counted,
not classified.

## Totals

| | |
|---|---|
| Tracked files / bytes | 1,219 files, 35 MB (`git ls-files \| wc -l`; `… du -ch \| tail -1`) |
| By extension (bytes) | `.json` 13.9 MB (112), `.rs` 7.1 MB (566), `.png` 4.7 MB (37), `.md` 4.1 MB (138), `.py` 3.3 MB (272) |
| Top-level entries | 29 (14 files, 15 directories) |
| Fixtures | 96 JSON (79 `_meta`, 5 `meta`, 1 `_source`, 11 none), 11 CSV, 77 `.py` + 2 `.R` generators, 1 README — 15 MB |
| Public Rust items | 1,411 (946 `fn`, 319 `struct`, 141 `enum`, 3 `trait`, 2 `type`) |
| `pub fn` with zero references anywhere | 9 |
| `pub fn` used only inside their own crate's `src/` | 50 (206 never referenced outside their crate) |
| `pub struct/enum` never referenced outside their crate | 258 (176 of them not even by the crate's own tests) |
| `#[allow(dead_code)]` / `allow(unused*)` | 16 in 14 files: 13 crate-level in `tests/common/mod.rs` (justified idiom), 3 in library code — all 3 stale |
| TODO-family markers | 57 `TODO`, 0 `FIXME`/`XXX`/`HACK`/`unimplemented!`/`todo!`; 52 are `TODO(phase0)` in Rust sources, 50 of those dated 2026-07-18 |
| Docs pages vs nav | 88 buildable pages, 88 nav entries, 0 unreachable, 0 nav entries without a file |
| Broken internal links (non-mkdocs Markdown) | 0 of 171 in 50 files |
| Findings | **2 severe, 8 moderate, 12 low** |
| Applied fixes | 2 (commit 2541cb0): `.gitignore` gaps closed; six `*.log` files untracked |

## Findings

### H1 — `scratch/round8/` is a committed debugging directory at the repository root — severe

21 files, 78,413 bytes, all last touched 2026-08-25 (`git log -1 --format=%cs -- scratch`).
Contents: 17 one-off probe scripts (`p1_proxy_first_stage.py` … `p12_no_behavior_drift.py`,
`p2b_frank_refute.py`, `p9c_theta_alpha_refute.py`), two drift JSONs, and three shell
wrappers. `scratch/round8/build.sh:12` hard-codes a machine path:

```
/home/user/tsecon/.venv/bin/maturin develop --release -m bindings/python/Cargo.toml \
```

and `cargo_test.sh` writes its output to `scratch/round8/cargo_copula.log` (an
ignored pattern). The only references in the repository are two sentences in
`docs/roadmap/23-audit-round-8-findings.md:19,28` that cite the scripts as the evidence
trail. Round 11 later established a proper home for exactly this kind of material —
`lab/audit/round11/` (`docs/roadmap/26-audit-round-11-findings.md:12-15`) — so the
repository now has the same artifact class in two places under two conventions.

**Recommendation:** move `scratch/round8/` to `lab/audit/round8/` (update the two
references in `23-audit-round-8-findings.md`), strip the `/home/user` path from
`build.sh`, and add `/scratch/` to `.gitignore` so the name cannot come back. Deleting
outright is also defensible: the findings doc quotes every number the scripts produced.

### H2 — `irf_table.tex` at the repository root is a cookbook example's output — severe

364 bytes, committed 2026-07-29 in f288150 ("Retention round …"), never touched since.
It is byte-for-byte the file written by the cookbook page's last line:

```
docs/cookbook/results-table-export.md:128:  print("bytes written:", Path("irf_table.tex").write_text(tex, encoding="utf-8"))
```

Nothing references it (`git grep -n irf_table` finds only that cookbook line). Anyone
who runs the cookbook page from the repo root regenerates it and gets a dirty tree.

**Recommendation:** `git rm irf_table.tex`; add `/irf_table.tex` to `.gitignore` (or
change the cookbook to write under a temp dir). Not applied here because deletions are
the owner's call.

### H3 — `docs/demo/` is a 247 KB orphaned, stale, hand-rendered artifact — moderate

`docs/demo/demo_template.html` (36 KB) and `docs/demo/index.html` (116 KB) are 675 lines
each with 565 lines in common; `index.html` is the template with `__PHILOX__` and
`__DATA__` substituted (`diff` shows exactly lines 100 and 305 differ). The embedded
payload is stamped `"generated_by":"tsecon 0.0.1 (Rust core)","date":"2026-07-17"` —
eight releases and 150+ functions ago. No generator is committed (`git grep -l
demo_template.html` outside `docs/demo/` is empty), nothing links to it (`git grep -n
'docs/demo\|demo/index'` outside the directory is empty, and `git log -S'docs/demo' --
README.md docs/index.md mkdocs.yml` is empty — it was never linked), and
`mkdocs.yml:13` excludes it from the site. It is therefore reachable only by browsing
the GitHub tree, where its "0.0.1" numbers are the first thing a reader sees.

**Recommendation:** delete, or move to `prototypes/` with the generator that fills the
template and a regeneration step; do not leave a 0.0.1-era "live demo" in `docs/`.

### H4 — `lab/audit/round11/out/` tracks 430 KB of regenerable sweep output, six files of it in violation of `.gitignore` — moderate (partly fixed)

`git ls-files -z lab/audit/round11/out | xargs -0 stat -c %s` sums to 429,850 bytes: seven
JSON files (sweep_e/f/g 116/116/88 KB, …), one Markdown table, and six `.log` files
(61,539 bytes). Every sweep script documents both as its own output
(`sweep_e_contract.py:11  Out: lab/audit/round11/out/sweep_e.log, sweep_e.json`, and
likewise for e_docdiff, ef_cards, f, g, h). The `.log` files are matched by the
repository's own rule — `git check-ignore --no-index -v` → `.gitignore:25:*.log` — which
has existed since 0.3.0 (13a44ff, 2026-08-18); they were force-added in the 0.8.0 merge.

**Applied (2541cb0):** the six `.log` files are untracked (`git rm --cached`); each is a
stdout copy of the sibling JSON, and nothing outside `lab/` references any of them.
**Proposed:** decide whether the JSON outputs are evidence (then cite the specific files
from `26-audit-round-11-findings.md`, which today cites only the directory and
`registry.py`) or artifacts (then untrack `out/` entirely and add `lab/audit/*/out/` to
`.gitignore`). Also add `audit/` to the directory map in `lab/README.md`, which does not
mention it (`grep -n -i audit lab/README.md` is empty).

### H5 — three `#[allow(dead_code)]` in library code are stale; the code compiles clean without them — moderate

`crates/tsecon-ident/src/summary.rs:68`, `:169`, `:277` each carry a comment of the form
"first live caller is the forthcoming … ; wired ahead of it". The callers arrived:

| item | callers today (`git grep -n '\bname('` in `crates/tsecon-ident/src`) |
|---|---|
| `structural_ma` (l.69) | `nongaussian.rs` ×1, `robust_bounds.rs` ×2, `structural_fevd.rs` ×1, `summary.rs` ×5 |
| `weighted_quantile_sorted` (l.170) | `summary.rs` ×6 (via `summarize_weighted`) |
| `summarize_weighted` (l.278) | `narrative.rs` ×2, `summary.rs` ×2 |

Verification: with the three attribute lines deleted, `cargo clippy -p tsecon-ident
--all-targets -- -D warnings` exits 0. The comments now misdescribe the module ("forthcoming
narrative sign-restriction sampler" — `narrative.rs` exists). The other 13 `allow`s are
crate-level `#![allow(dead_code)]` in `tests/common/mod.rs` with a doc-comment
justification each (every integration binary uses a subset) — fine.

**Recommendation:** delete the three attributes and the three "forthcoming" comments.

### H6 — 52 `TODO(phase0)` markers have no owner and several are stale — moderate

All 57 TODO-family hits are plain `TODO`; 52 are `TODO(phase0)` in `crates/*/src`, and 50
of those blame to the 2026-07-18 initial import (the other two: `lpdid.rs` 2026-08-20,
`irf_bootstrap.rs` 2026-07-22). No roadmap page, ROADMAP.md, or CONTRIBUTING.md mentions
`phase0` (`git grep -c phase0 -- ROADMAP.md CONTRIBUTING.md ':(glob)docs/**/*.md'` finds one
incidental line in `26-audit-round-11-findings.md`). Stale examples, verified against the
shipped surface:

| marker | says | reality |
|---|---|---|
| `crates/tsecon-var/src/irf.rs:9`, `src/lib.rs:33` | "bootstrap IRF confidence bands are TODO(phase0)" | `irf_bootstrap.rs` (`pub fn bootstrap_irf_bands`, `…_simultaneous`) shipped 2026-07-22; Python `var_irf_bands` (`__init__.pyi:418`) |
| `crates/tsecon-forecast/src/benchmarks.rs:32`, `theta.rs:45` | "arrive with the backtesting slice" | `backtest`, `var_backtest.json`, `backtest_string_snapshot.json` all shipped |
| `crates/tsecon-garch/src/results.rs:150` | EGARCH multi-step by simulation pending | round 11 scrubbed "TODO(phase0)" from the *user-facing* message (`test_audit_round11.py:317`) but left the source marker |
| `crates/tsecon-panelts/src/lib.rs:44`, `pmg.rs:97` | "a documented `TODO`" | points at no tracking entry |

Live examples with no home: unbalanced panels (`tsecon-panel` ×4), LDL' pre-whitening
in `tsecon-ssm`, Clark-West / Giacomini-White in `tsecon-forecast/dm.rs:38`, Temme
expansion in `tsecon-stats/special.rs:326`.

**Recommendation:** one sweep — delete the stale ones, and give the live ones a row in
the relevant `docs/roadmap/NN-*.md` page (or a single "deferred in source" list) so the
marker is `TODO(roadmap-NN)` rather than a phase that ended in July.

### H7 — nine public functions are referenced by nothing, two with a false justification — moderate

Scan of 946 `pub fn` (name-boundary matches across every `.rs` in `crates/` and
`bindings/python/src`, definition line excluded). The crates are not published
(`grep publish crates/*/Cargo.toml` is empty, no crates.io mention anywhere), so
unreferenced `pub` is dead weight, not API.

| function | location | note |
|---|---|---|
| `_norm_cdf` | `crates/tsecon-copula/src/family.rs:601` | doc says "re-exported for the property tests' independence checks"; `git grep _norm_cdf crates/tsecon-copula/tests/` is empty |
| `_debye_1` | `crates/tsecon-copula/src/family.rs:607` | same false "re-exported for the property tests" claim |
| `Scaler::fit_transform` | `crates/tsecon-ml/src/standardize.rs:135` | |
| `Tree::predict_row` | `crates/tsecon-ml/src/tree.rs:176` | |
| `within_residuals` | `crates/tsecon-panel/src/fe.rs:189` | `#[must_use]` accessor never read |
| `SeedSequence::from_entropy_words` | `crates/tsecon-rng/src/seedseq.rs:120` | NumPy-parity API, no test exercises it |
| `SeedSequence::entropy_words` | `crates/tsecon-rng/src/seedseq.rs:253` | |
| `business_cycle_monthly` | `crates/tsecon-spectral/src/business_cycle.rs:68` | monthly preset never bound (`__init__.pyi` has no match) |
| `disturbance_dim` | `crates/tsecon-ssm/src/model.rs:264` | |

None is in a `#[cfg(test)]` module, a doctest, a macro, docs, or the stub (all checked).
Beyond these nine: 50 `pub fn` are used only inside their own crate's `src/` (43 with a
workspace-unique name — the safe `pub(crate)` list is in the dead-code table), and 176
`pub struct/enum` are referenced by nothing outside their crate, not even its tests.

**Recommendation:** delete the nine (or bind + test `business_cycle_monthly` and the two
`SeedSequence` methods if they are wanted); downgrade the 43 to `pub(crate)`; fix the two
copula doc comments regardless.

### H8 — `fixtures/README.md` overstates fixture provenance: 17 of 96 JSON fixtures have no `_meta` — moderate

`fixtures/README.md` says "Each fixture records the exact reference-library versions
used, so the values are reproducible." Checked with `json.load` on every file:

- 79 have a top-level `_meta`;
- 5 use `meta` instead (`auto_arima`, `ou`, `simultaneous`, `tsecon-dcs`, `tsecon-panelroot`) — same content, different key, so any tooling keyed on `_meta` misses them;
- 1 uses `_source` (`engle_granger`);
- 11 record **no** library, version, seed, or generator string anywhere in the file (`re.findall` for statsmodels/numpy/scipy/version/seed is empty): `backtest_string_snapshot` (documented self-snapshot — fine), `fry_pagan_svar` (407 KB), `historical_decomposition_chol` (292 KB), `long_run_svar`, `zero_sign_svar`, `tsecon-gas` (173 KB), `tsecon-dsge`, `nowcast_mle`, `nowcast_news`, `tsecon-nowcast`, `tsecon-recession`.

Several of the eleven are closed-form / self-consistency fixtures where "no third-party
reference" is the honest answer — but then the file should say so (a `_meta.reference:
"closed form, see generator docstring"`), and the README should not claim otherwise.
All 96 are still loaded by a test and every generator maps to an existing JSON (see
*Clean bills*), so this is a provenance-labelling gap, not dead data.

**Recommendation:** normalise to `_meta` everywhere (rename the 5 `meta` and 1
`_source`), add a minimal `_meta` (generator, seed, reference or "closed form") to the
eleven, and soften the README sentence until then.

### H9 — release metadata drift: `CITATION.cff` is one release behind; the fourth notebook is unlisted — moderate

`CITATION.cff:17-18` reads `version: 0.7.0` / `date-released: 2026-08-28`, while
`Cargo.toml:51`, `bindings/python/pyproject.toml:9` and `CHANGELOG.md:10` are 0.8.0
(2026-09-03). The combined "0.7.0 + 0.8.0" merge (19d308e) bumped the file once
(`git diff 19d308e~1 19d308e -- CITATION.cff`: `0.6.0 → 0.7.0`). `paper/paper.md:40`
says 0.8.0. A citing reader gets the wrong version from the file README.md:184 points
them to.

Related: `notebooks/04_gertler_karadi.ipynb` (72 KB, the largest notebook) is absent
from `notebooks/README.md`'s table (three rows) and from README.md:38's "Two more
runnable notebooks" — `git grep 04_gertler -- README.md docs/index.md
notebooks/README.md` is empty. CI executes it (`ci.yml:157`) but no reader is told it
exists.

**Recommendation:** bump `CITATION.cff` to 0.8.0 / 2026-09-03 and add a release-checklist
line for it; add notebook 4 to both READMEs.

### H10 — quantified duplication: 13 test-helper copies, one special-function copy, and repeated constants — moderate

*Test helpers.* 13 `crates/*/tests/common/mod.rs`, 1,242 lines total, only one pair
byte-identical (`tsecon-connect` ≡ `tsecon-var`, md5 `13711427`). Per-function:
`load_fixture` defined 12× (10 bodies byte-identical), `uniform` 10×, `assert_rel_close`
10×, a private xorshift/LCG `next_u64` 9× (each file's header explains "does not depend on
tsecon-rng"), `as_mat` 8×, `assert_mat_close` 5×. Each crate compiles its own copy.

*Special functions.* `crates/tsecon-ml/src/pds.rs:253-363` (≈110 lines) re-implements
`ERF_A/ERF_B/ERF_THRESH`, `ERFC_C/D/P/Q/XBIG`, `SQRPI`, `erf_small`, `erfc_abs`, `erfc` from
`crates/tsecon-stats/src/special.rs:157-317` (constant tables whitespace-identical;
`tsecon-ml` does not depend on `tsecon-stats`, so this is a dependency-avoidance copy).

*Constants in more than one crate* (`const`/`static` names, `crates/*/src`):
`Z_975` (`tsecon-bootstrap/blocklength.rs:14`, `tsecon-ml/pds.rs:107`), `LN_2PI`
(`tsecon-gas/level.rs:169`, and twice inside `tsecon-stats`: `dist/normal.rs:12`,
`special.rs:53`), `UNIFORM_RETRIES = 128` (`tsecon-bayes/dense.rs:255`,
`tsecon-ident/haar.rs:43`, `tsecon-ident/zero.rs:62`). `NU_BOUNDS` and `MIN_OBS` share a
name across crates with *different* values — intentional, but worth a comment.

Critical-value tables are **not** duplicated: MacKinnon p-value and critical-value
surfaces live only in `tsecon-diag` (`mackinnon.rs`, `mackinnon_ext.rs`) and
`tsecon-coint` (`engle_granger.rs`) calls into them; no other crate defines its own.

**Recommendation:** a `tsecon-testkit` dev-dependency crate (or a workspace-level
`tests/common.rs` pulled in by `#[path]`) for the helpers; make `tsecon-ml` depend on
`tsecon-stats` and delete the copy; hoist `Z_975`/`LN_2PI` into `tsecon-stats` and reuse.

### Low-severity findings (one line each)

- **L1** `scripts/cross_check_auto_arima_pmdarima.py` (4,882 B, 2026-08-25) is referenced by nothing in the repository (`git grep -l cross_check_auto_arima` returns only itself); its sibling `mc_auto_arima_recovery.py` is cited from the ARIMA model card — cite it there too or delete it.
- **L2** `prototypes/viz/tsecon_style.py` is a *live* dependency of all five gallery generators (`docs/examples/showcase*.py:16  sys.path.insert(0, str(REPO / "prototypes" / "viz"))`) and of the shipped palette (`results/_plotting.py:18` "kept in sync with prototypes/viz/tsecon_style.py" — verified in sync, 12/12 colours); "prototypes" is the wrong home for build-critical code — move to `docs/examples/_style/`.
- **L3** `prototypes/viz/make_preview.py` renders `docs/assets/viz-preview/*.png`; two of the four (`diagnostics-dashboard.png` 159,222 B, `irf-panel-grid.png` 166,725 B) are referenced only from the site-excluded `docs/roadmap/13-visualization.md`, so 326 KB ships in `docs/assets/` for no published page.
- **L4** 24 lines in 10 tracked files embed the developer machine path `/home/user/tsecon/…` (4 fixture-generator docstrings: `generate_tsecon-copula/-dcs/-evt/var_backtest_fixtures.py`; `lab/README.md` ×5, `lab/REPORT.md` ×4, `lab/laplace/README.md` ×5, `lab/laplace/tests.py` ×2, `lab/prophet_lite/README.md` ×3, `lab/prophet_lite/tests.py` ×1) — all in prose/docstrings, none in code paths; replace with `.venv/bin/python`.
- **L5** `.gitignore` gaps — **applied** (2541cb0): `.venv-wt/` (created by the round-8 scripts), `.mypy_cache/`, `.ruff_cache/`, `.hypothesis/`, `.ipynb_checkpoints/`, `.idea/`, `.vscode/`, `*.swp`/`*.swo` were all "NOT ignored" under `git check-ignore --no-index`; still open: `/scratch/` and `/irf_table.tex` once H1/H2 are actioned.
- **L6** No `SECURITY.md`, `CODEOWNERS`, `.github/dependabot.yml`, or issue/PR templates (`git ls-files | grep -iE 'SECURITY|dependabot|CODEOWNERS|TEMPLATE'` is empty), and `grep -i 'security\|vulnerab' CONTRIBUTING.md README.md GOVERNANCE.md` finds no reporting route.
- **L7** `bindings/python/src/lib.rs` is a single 12,706-line, 552,756-byte file holding 160 `#[pyfunction]`s; only the 0.8.0 ML wave was split into `ml_*.rs` (5 files, 309–410 lines each) — the precedent exists, the rest of the file did not follow it.
- **L8** The "173 functions" count is hand-stated in five files (`README.md:64,109,120`, `ROADMAP.md:15`, `docs/index.md:55`, `paper/paper.md:40,117`) and no test asserts the literal (`test_stub_sync.py` checks stub↔runtime parity, not the documented number).
- **L9** `docs/roadmap/15-proxy-svar-bands.md` has zero inbound references from ROADMAP.md, README, CHANGELOG, CONTRIBUTING or any `docs/**/*.md`; every other roadmap page has ≥1.
- **L10** `docs/reference/api.md` is both generated (`docs/gen_api_reference.py`, run by `docs.yml:36` before every build) and tracked; in sync today (regenerating produced no diff) but nothing fails CI if it drifts — add `git diff --exit-code` after the regenerate step, as `ci.yml:173-175` already does for notebooks.
- **L11** `lab/README.md`'s directory map omits `lab/audit/` (added 2026-09-03), so the lab's own README no longer describes the largest thing in it (430 KB of 808 KB).
- **L12** `LN_2PI` is defined twice *inside* `tsecon-stats` (`dist/normal.rs:12`, `special.rs:53`) with the same 35-digit literal.

## Proposed removals

Recommendation only; nothing below was deleted. Sizes from `git ls-files -z -- <p> | xargs -0 stat -c %s`.

| Path | Bytes | Files | Last touch | Referenced by | Recommendation | Why |
|---|---:|---:|---|---|---|---|
| `scratch/round8/` | 78,413 | 21 | 2026-08-25 | `docs/roadmap/23-audit-round-8-findings.md:19,28` | **move** to `lab/audit/round8/` (or delete) + ignore `/scratch/` | H1: debugging junk at root, hard-coded `/home/user` path, duplicates the `lab/audit/` convention |
| `irf_table.tex` | 364 | 1 | 2026-07-29 | nothing | **delete** + ignore | H2: output of `docs/cookbook/results-table-export.md:128` |
| `docs/demo/` | 246,807 | 2 | 2026-08-27 | nothing | **delete** (or move to `prototypes/` with a generator) | H3: 0.0.1-era rendered demo, no generator, never linked |
| `lab/audit/round11/out/*.log` | 61,539 | 6 | 2026-09-03 | nothing outside `lab/` | **untracked — applied** (2541cb0) | H4: matched by `.gitignore:25 *.log` |
| `lab/audit/round11/out/*.json`, `sweep_g_table.md` | 368,311 | 8 | 2026-09-03 | nothing (directory cited by `26-audit-round-11-findings.md:15`) | **owner's call**: keep and cite by filename, or untrack + ignore `lab/audit/*/out/` | H4: regenerable by the sibling scripts |
| `scripts/cross_check_auto_arima_pmdarima.py` | 4,882 | 1 | 2026-08-25 | nothing | **delete** or cite from `reference/model-cards/arima.md` | L1 |
| `docs/assets/viz-preview/diagnostics-dashboard.png`, `irf-panel-grid.png` | 325,947 | 2 | 2026-07-18 | `docs/roadmap/13-visualization.md` only (site-excluded) | **move** next to the roadmap page or delete | L3 |
| `prototypes/viz/` | 16,817 | 2 | 2026-07-18 | 5 gallery generators, `_plotting.py`, `roadmap/13` | **move** to `docs/examples/_style/` (do not delete) | L2: live build dependency mis-labelled as a prototype |
| `crates/tsecon-ml/src/pds.rs:253-363` (erf/erfc copy) | ≈4 KB | — | 2026-09-03 | — | **delete** after adding `tsecon-stats` dependency | H10 |
| 9 dead `pub fn` (H7 table) | — | — | — | nothing | **delete** or bind + test | H7 |

Not proposed for removal, checked and kept: `notebooks/` (CI-executed, README-linked,
outputs stripped by `build.py`), `benchmarks/` (CI gate `ci.yml:139`), `paper/`
(README:186), `lab/{prophet_lite,laplace,experiments}` (CHANGELOG:1871 and two model
cards cite `exp06`), `.coveragerc` (`testing.md:858`), `Cargo.lock` (correct for a
workspace that ships a binary extension), `docs/reference/api.md` (generated but the
nav needs it), `__init__.pyi` (hand-maintained, guarded by `test_stub_sync.py`).

## Dead-code table

Public Rust items by crate. `fn_no_ext` = never referenced outside the crate;
`fn_only_src` = referenced only by the crate's own `src/` (not even its tests);
`se_no_ext` = `struct`/`enum` never referenced outside the crate. Sorted by `fn_only_src`.

| crate | pub fn | fn_no_ext | fn_only_src | struct+enum | se_no_ext |
|---|---:|---:|---:|---:|---:|
| tsecon-ident | 117 | 21 | 9 | 47 | 24 |
| tsecon-ml | 70 | 21 | 6 | 45 | 20 |
| tsecon-gas | 35 | 5 | 4 | 12 | 8 |
| tsecon-dsge | 18 | 6 | 3 | 5 | 4 |
| tsecon-garch | 22 | 4 | 3 | 8 | 1 |
| tsecon-recession | 6 | 3 | 3 | 4 | 3 |
| tsecon-rng | 27 | 13 | 3 | 4 | 0 |
| tsecon-copula | 19 | 10 | 2 | 6 | 3 |
| tsecon-hac | 15 | 7 | 2 | 6 | 1 |
| tsecon-lp | 31 | 3 | 2 | 18 | 12 |
| tsecon-panel | 18 | 2 | 2 | 12 | 7 |
| tsecon-regime | 26 | 2 | 2 | 16 | 10 |
| tsecon-ssm | 36 | 9 | 2 | 9 | 5 |
| tsecon-termstructure | 25 | 11 | 2 | 8 | 8 |
| tsecon-diag | 33 | 7 | 0 | 41 | 22 |
| tsecon-forecast | 40 | 5 | 0 | 23 | 15 |
| tsecon-var | 36 | 7 | 0 | 21 | 11 |
| (26 other crates, ≤1 each) | 411 | 76 | 7 | 175 | 84 |
| **total** | **946** | **206** | **50** | **460** | **258** |

Caveat: 334 `pub fn` names are shared by more than one crate (`new`, `fit`, `forecast`…),
so name-boundary counting over-counts references for those; the `fn_only_src` and
zero-reference lists below are restricted to workspace-unique names where it matters.

**Zero references anywhere (9):** see H7.

**Referenced only from their own `src/`, workspace-unique name (43) — `pub(crate)` candidates:**
`tsecon-arima::seasonal_ma` (results.rs:146); `tsecon-dsge::n_variables` (model.rs:124);
`tsecon-favar::singular_values` (pca.rs:190); `tsecon-garch::{backcast_value (model.rs:101),
loglike_obs (model.rs:194), n_mean_params (spec.rs:155)}`; `tsecon-gas::{log_density
(kernel.rs:73), scaled_score (kernel.rs:166), steady_state_gain (level.rs:943),
forecast_from (model.rs:359)}`; `tsecon-hac::{andrews_constant (kernel.rs:130), andrews_q
(kernel.rs:144)}`; `tsecon-ident::{identified_set_bounds (robust_bounds.rs:389),
is_satisfied (sign.rs:249), zeros_per_shock (zero.rs:249), all_impact_only (zero.rs:256)}`;
`tsecon-longmemory::low_frequency_periodogram` (spectral.rs:60);
`tsecon-lp::{accumulates_outcome (spec.rs:92), accumulates_impulse (spec.rs:98)}`;
`tsecon-ml::{uses_gamma (kernel_ridge.rs:120), kernel_matrix (kernel_ridge.rs:200),
leaf_for (tree.rs:153), predict_with (tree.rs:170)}`; `tsecon-nowcast::state_space`
(statespace.rs:163); `tsecon-panel::broadcast_common` (data.rs:89);
`tsecon-recession::{loglik_term (link.rs:44), score_factor (link.rs:72), info_weight
(link.rs:91)}`; `tsecon-regime::{matches_spec (params.rs:255), expanded_states
(spec.rs:53)}`; `tsecon-rng::from_seed_sequence` (stream.rs:43); `tsecon-ssm::obs_dim`
(model.rs:252); plus the 9 zero-reference items above (full list with counts in the
scan output; regenerate with the method in *Scope*).

**Stale `#[allow(dead_code)]` (3):** `crates/tsecon-ident/src/summary.rs:68,169,277` — H5.

**Test-helper duplication (13 files, 1,242 lines):** H10.

## Clean bills

Checked and found in order; listed so the next audit does not repeat them.

- **No tracked build or editor artifacts.** `git ls-files` contains no `__pycache__`, `.pyc`, `site/`, `target*/`, `.ipynb_checkpoints`, `.DS_Store`, `.swp`, `.orig`, `.idea/`, `.vscode/`, `.egg-info`, `.coverage`, or cache directories (the only hits of the artifact grep were the `docs/demo` HTML and the six logs, both handled above).
- **Notebooks carry no outputs.** All four `.ipynb` have `output_type` count 0 and every `execution_count` is `null`; `notebooks/build.py` strips them by design and `ci.yml:171-175` fails the build if an `.ipynb` drifts from its `_src.py`.
- **Largest tracked files are all legitimate.** The 30 largest (1.58 MB `nongaussian_svar.json` down to 167 KB) are 21 golden fixtures, 7 gallery PNGs, `CHANGELOG.md` (177 KB) and the binding `lib.rs`; no binaries, no stray data dumps. Total tracked size 35 MB, 13.9 MB of it fixtures.
- **Every fixture is loaded by a test.** 93/96 JSON fixtures are named in `crates/**/tests` or `bindings/python/tests`; the three exceptions were refuted — `robust_svar_bounds.json` is loaded by an in-source `#[cfg(test)]` test (`crates/tsecon-ident/src/robust_bounds.rs:842`), and `acm_published_10y.csv` / `gsw_nss_params.csv` are generator *inputs* (both read by `generate_acm_fixtures.py`). The 8 vendored CSVs with no generator are the data files `fixtures/README.md` documents as such, each with a source header.
- **Every generator maps to an existing JSON.** 79 generators → 96 JSON files; the only "missing" name (`out.json`) is a temp file inside the two R bridges. `tsecon-nowcast.json` is written by exactly one generator (`generate_tsecon-nowcast_fixtures.py:61`); the other mention is a docstring.
- **Docs nav is complete.** 88 buildable pages ↔ 88 nav entries; 0 pages unreachable, 0 nav entries pointing at a missing file; `exclude_docs` covers exactly `roadmap/`, `demo/`, `_hooks/`, the generator, `requirements.txt`, `__pycache__/`.
- **No broken internal links** in the 50 Markdown files mkdocs never sees (README, ROADMAP, CONTRIBUTING, CHANGELOG, GOVERNANCE, CODE_OF_CONDUCT, THIRD-PARTY-LICENSES, `notebooks/`, `fixtures/`, `benchmarks/`, `lab/`, `paper/`, `docs/roadmap/`): 171 links checked, 0 broken. Nothing to fix under the "broken links" mandate.
- **No `FIXME` / `XXX` / `HACK` / `unimplemented!` / `todo!`** anywhere (57 hits, all `TODO`).
- **Critical values are single-sourced** in `tsecon-diag` (H10).
- **`docs/reference/api.md` is in sync** with `gen_api_reference.py` (regeneration produced no diff, 173 functions / 59 sections); `__init__.pyi` is guarded by `test_stub_sync.py`.
- **Palette is in sync** between `results/_plotting.py` and `prototypes/viz/tsecon_style.py` (12/12 colours present).
- **Markdown near-duplication is minimal.** README ↔ `docs/index.md` share only the six-line pitch paragraph; no other pair among README / index / guide overview / reference overview / quickstart / ROADMAP shares a single line >40 chars.
- **Toolchain pinned** (`rust-toolchain.toml`: 1.97.1 with rustfmt + clippy) and `Cargo.lock` tracked — correct for a workspace that ships a binary extension.
- **`.github/`** holds only the three workflows; CI runs fmt, clippy `-D warnings`, `cargo test --workspace`, wheel install + pytest on a matrix, mypy, both Monte Carlo suites, the benchmark parity gate, notebook execution and sync, and `mkdocs build --strict`.

## Open

- **`pub(crate)` candidates not compile-verified.** The 43 names above were counted textually; I did not change each to `pub(crate)` and rebuild. Expect a few to be reachable through a re-export or a trait impl the grep did not distinguish.
- **176 crate-private `pub struct/enum` not classified.** Many are results/spec types that are genuinely internal; the count is reported, the list is not.
- **`mkdocs build --strict` not re-run here** (mkdocs is not installed in the container); the nav/page comparison was done from `mkdocs.yml` and `git ls-files` directly.
- **R generator provenance** (`generate_bn_filter_fixtures.R`, `generate_lpdid_fixtures.R`) not examined beyond confirming their outputs exist and are tested.
- **`lab/` experiments were not re-run**; the 24 `/home/user` mentions were confirmed to be prose only, not executed paths.
- **Not applied, by mandate:** every deletion/move in *Proposed removals*, the three `allow` deletions (H5), the TODO sweep (H6), the fixture `_meta` normalisation (H8), the `CITATION.cff` bump (H9). Each is a one-line change once the owner decides.
