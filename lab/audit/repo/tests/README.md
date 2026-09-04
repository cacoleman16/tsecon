# Repo audit — test-suite health probes

Probe scripts for the test-suite sweep of the whole-repository audit
(`docs/roadmap/_repo_audit/tests.md`). All run from the worktree root with
the release extension built into `.venv-wt`.

| script | what it measures |
|---|---|
| `run_suite.sh <label>` | one full Python suite run: `-q -p no:cacheprovider --durations=40 -rA --junitxml` into `out/run_<label>.{log,xml}` |
| `diff_runs.py durations <run>` | 20 slowest tests and per-file wall time from a run's junit XML |
| `diff_runs.py diff <a> <b> …` | outcome set and captured-stdout (printed numbers) diff between runs |
| `assertion_scan.py` | AST classification of every test function: none / weak-only / raises-only / numeric; tolerance literals; duplicated bodies |
| `tolerance_vs_matrix.py` | loosest fixture-referencing tolerance per test file vs the matrix row's documented tolerance |
| `rust_no_assert.py` | `#[test]` fns with no assert / `?` / `unwrap_err` / helper assert |
| `skips_scan.py` | every `pytest.skip` / `importorskip` / `skipif` / `xfail` and Rust `#[ignore]` with its reason |
| `fixture_meta_drift.py` | recorded reference-library versions in every fixture JSON vs the installed venv; generator parse / import / run-doc check |
| `coverage_depth.py` | per public callable: distinct tests calling it and the strongest assertion kind |
| `tolerance_headroom.py` | achieved error behind the 14 Python re-checks that are looser than the matrix |
| `mc_trim_probe.py` | per-replication replay of the 241-s auto_arima Monte-Carlo test (evidence for the trim proposal) |
| `ci_sim/sitecustomize.py` | import blocker reproducing CI's numpy+scipy-only python job (`PYTHONPATH=lab/audit/repo/tests/ci_sim`) |

Outputs land in `out/` (committed: the text summaries and the run logs).
