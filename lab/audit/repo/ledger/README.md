# Ledger probes (repository audit, September 2026)

Closure probes for `docs/roadmap/_repo_audit/ledger.md` — one probe per
ledger claim that is cheap to prove on the wheel (a refusal now fires, a
kwarg or key now exists, a default was flipped, or a recorded gap is still
there).

- `probes.py` — the battery (58 probes, each wrapped so one failure cannot
  hide another; prints attempted vs reached).
- `probes_run.txt` — the run this ledger cites: tsecon 0.8.0 built from
  `19d308e` in this worktree (`maturin develop --release`, numpy 2.4.6,
  scipy 1.17.1, statsmodels 0.15.0, scikit-learn 1.9.0, mapie 1.5.0),
  58/58 reached, 0 errors; verdicts CLOSED 31 / OPEN 18 / INFO 9.

Re-run from the repository root:

    .venv-wt/bin/python lab/audit/repo/ledger/probes.py > lab/audit/repo/ledger/probes_run.txt

Read-only against the tree; the only artefact is the log.
