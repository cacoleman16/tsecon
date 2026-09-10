# Audit round 13 — probe scripts

The post-wave adversarial sweep over the six callables added in 0.9.0
(`setar_threshold_ci`, `var_girf`, `threshold_var_girf`, `jsz_fit`,
`jsz_loadings`, `panel_distributed_lag`) and the changed count pre-flight in
`_coerce`. Findings: `docs/roadmap/28-audit-round-13-findings.md`. All scripts
run from the repository root with the release extension built into `.venv`
and write to `out/` (summaries as `.txt`, because `*.log` is ignored
repo-wide; per-cell dumps as `.json`).

| script | what it does |
|---|---|
| `registry.py` | canonical valid inputs for all 179 public callables: the security sweep's registry (round 11 + the 0.8.0 wave) plus the six 0.9.0 entries (`NEW`) |
| `common.py` | round 11's walkers (bit-exact compare, JSON/pickle round-trip, non-finite paths, shapes) plus the card-section and bracket-tolerant token helpers |
| `probe_stub_parse.py` | does the shipped stub parse (`ast.parse`), which docstrings are unterminated, and where the regex-based api.md generator leaked stub source into prose — run before and after the fix (`out/stub_parse_{pre,post}fix.txt`, `out/stub_mypy_{pre,post}fix.txt`) |
| `sweep_e_contract.py` | sweep E: summarize / JSON / pickle round-trips, non-finite floats vs the docstring, returned keys vs `__doc__`, stub and card in both directions, array shapes vs the documented shapes |
| `sweep_f_drift.py` | sweep F: `inspect.signature` vs the stub (all 179), prose defaults on four surfaces vs the runtime default, kwargs in call snippets across docstrings / cards / guides 13-16 / api.md, listed string values passed, inert-keyword refusals, documented no-ops measured |
| `sweep_f_positional.py` | sweep F(f): a count of 2^48 passed positionally in every positional slot of every callable, cold and warm cache, must be named by the seal (`out/sweep_f_positional.txt`) |
| `cell_runner.py`, `parent.py` | the one-cell-per-line child protocol shared by G and S: every cell in a child under a 4 GB `RLIMIT_AS` cap and a deadline; panics, aborts and hangs attributed to the exact cell |
| `sweep_s_malformed.py` | sweep S: every parameter of the six corrupted one at a time (NaN / inf / empty / wrong rank / short rows and columns / nested list / int and bool arrays / transposed / duplicated row / string / None / scalar; ints 0, 1, 2, −1, 2^31, 2^47, 2^63, 1.5, True, "3", None, [1]; floats nan, ±inf, −1, 0, 1e300, 1e-300, int, True, "abc", None, [0.5]; bools, strings, integer lists, the bands tuple, the Optional slots) — 673 cells |
| `sweep_g_timing.py` | sweep G: three sizes in fresh subprocesses with log-log slope and a re-time of flagged cells, a defaults-only pass, then the count arguments at 2^47 / 2^31 / 10^6 and at the documented cap ± 1 under the memory cap |
| `sweep_h_seed.py` | sweep H: same seed twice in-process and across a restart, different seed, `seed=None`, determinism of all six, `RAYON_NUM_THREADS` 1 vs 4 |
| `sweep_c_claims.py` | sweep C: the binding tests that pin printed numbers, the cards' runnable examples against their expected output, `bench.py --json` vs the committed dashboard JSON, the LP-vs-VAR script vs chapter 16's verbatim block, the Rust property-test output (`out/rust_props.txt`) vs the quoted Monte-Carlo numbers |
