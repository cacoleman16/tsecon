# Audit round 16 — repro scripts and the house-wide abort sweep

The minimal reproduction behind each finding in
[`docs/roadmap/31-audit-round-16-findings.md`](../../../docs/roadmap/31-audit-round-16-findings.md),
one directory per finder lens, plus the measurement labs the fix groups
left behind. Each lens script is the version that showed the defect on the
wheel built from `fdd90d3`, which was the PR #30 head when the round ran.
After the fixes, a script prints the corrected behaviour (or the named
refusal) instead of the defect. The refuted findings' scripts still
reproduce what was observed; the findings document explains why each of
those is not a defect.

Run them from the repository root with an installed `tsecon` wheel and the
reference environment the round used (numpy 2.5.3, scipy 1.18.1,
statsmodels 0.15.0, arch 8.0.0, linearmodels 7.0, scikit-learn 1.9.1,
pandas 2.3.3, statsforecast 2.1.1). Anything that may allocate wildly runs
in a child process under a memory cap:

```sh
bash -c 'ulimit -v 4000000; timeout 120 python lab/audit/round16/fuzz_svar/repro_minroot_flip.py'
```

| Directory | Lens | Findings |
|---|---|---|
| `fuzz_svar/` | Differential fuzzing of the structural-VAR additions | SVF-1 … SVF-5 |
| `fuzz_stats/` | Differential fuzzing of the statistical and state-space additions (`dhref.py`/`mpref.py` are the Doornik-Hansen references) | DF16-1 … DF16-5 |
| `hostile/` | Malformed-input sweep of the batch-2 surface | H16-01 … H16-14 |
| `abort_sweep/` | House-wide abort/hang sweep of every public callable | A1 … A10, B1 … B7, C1 |
| `docs_truth/` | Documentation truth | DOC16-1 … DOC16-6 |
| `api/` | API consistency | API16-1 … API16-9 |
| `mc_claims/` | Statistical-claim re-measurement | STAT16-1, STAT16-2 (both refuted) |
| `var_numerics/`, `var_budgets/`, `ident_bayes/` | Labs of the fix groups: before/after bit-identity of valid calls, budget footprints and boundaries, cost curves | — |

## The abort sweep

`abort_sweep/` is reusable. `cases.py` holds a validated baseline call for
every public callable (278 variants) and the list of integer- or size-like
parameters to sweep (521, including the components of list, tuple and
dict arguments); `prelude.py` holds the small seeded fixtures. The harness
runs each case in its own child process under `ulimit -v 4000000` and a
60 s deadline, at the values 2^31, 2^40, 2^62, −1 and 0, and classifies the
outcome: a result, a refusal that names the parameter, an unnamed refusal,
an abort, a timeout, a `MemoryError` or a panic.

```sh
cd lab/audit/round16/abort_sweep
W=2 python harness.py results.jsonl        # W child processes at a time
python analyze.py results.jsonl            # counts, then every non-named case
```

Keep `W` small: each child may use up to 4 GB. A new callable needs a
baseline in `cases.py`; `python check_baselines.py` lists the callables
without one and re-runs every baseline, which must succeed before a sweep
means anything.

Round 16's own runs, at the head it audited and on the merged fixes (the
latter's records are in `abort_sweep/results_final.jsonl`, trimmed to the
case, its class and the exception's type and message):

| Outcome | `fdd90d3` (round 16 ran here) | merged fixes |
|---|---|---|
| Refusal naming the parameter | 1,830 | 2,177 |
| Result returned | 538 | 528 |
| Unnamed refusal | 194 | **0** |
| Abort (SIGABRT, interpreter killed) | 136 | **0** |
| Timeout (60 s) | 12 | **0** |
| `MemoryError` or panic | 0 | 0 |
| Calls | 2,710 | 2,705 |

The re-run makes five fewer calls: the corrected `narrative_svar` baseline
no longer sweeps `n_weight_draws`, which the `narrative_svar#narr` variant
sweeps with narrative restrictions in place. A "result returned" at 2^62
is mostly a seed, for which any 64-bit value is legitimate; the rest are
counts the callable accepts by design, such as `max_epochs` under early
stopping.
