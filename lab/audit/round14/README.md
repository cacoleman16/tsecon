# Audit round 14 — the refusal-naming re-run

A copy of round 13's malformed-input sweep (`sweep_s_malformed.py`, the
one-cell-per-line child protocol in `cell_runner.py`/`parent.py`, the
registry and the shared helpers) re-run against the 0.10.0 tree after the
refusal-naming pass of the `hygiene` slice: the 83 refusals of
`lab/audit/round13/out/sweep_s.txt` that did not name the mutated
argument (Rust sufficiency and dimension messages, PyO3's `Can't extract
`str` to `Vec``, array slots handed a scalar / string / `None`, the
`delays` entry blamed on `delay`) were fixed in the Rust messages, in the
bindings and in the `_coerce` rebuilds, and this sweep counts what is
left. Same 673 cells, same 4 GB cap per child, same verdict rules. Output
in `out/sweep_s.txt` (summary) and `out/sweep_s_cells.json` (per cell).

Run from the repository root with the release extension built into the
venv:

    .venv/bin/python lab/audit/round14/sweep_s_malformed.py

The in-process replay of the 83 cells is the CI tripwire
`bindings/python/tests/test_security_audit.py::test_every_sweep_s_refusal_names_the_offending_parameter`.
