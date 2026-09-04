# Repo audit — API-consistency probes

Probe scripts for the API sweep of the whole-repository audit
(`docs/roadmap/_repo_audit/api.md`). All run from the repository root with
the release extension built into the sweep's venv.

| script | what it produces |
|---|---|
| `registry.py` | canonical inputs for all 173 public callables (round-11 registry plus the 0.8.0 wave) |
| `probe_surface.py` | the master surface table `surface.json`: runtime signature, stub signature, returned keys on the canonical call |
| `render_surface_table.py` | `out/surface_table.md`, the compact markdown rendering of `surface.json` |
| `probe_clusters.py` | parameter-name and return-key clusters (`out/clusters.md`; `out/clusters.json` carries the default per spelling) |
| `probe_blast.py` | blast radius of each rename proposal: mentions per spelling across the repo (`out/blast.json`) |
| `probe_conventions.py` | five malformed calls per callable (NaN, empty, wrong ndim, unknown string, negative count); `out/conventions.json` |
| `summarize_conventions.py` | tabulates the conventions probe into `out/compliance.md` |
| `probe_docstrings.py` | docstring structure per callable on both surfaces (`out/docstrings.json`) |
| `probe_shapes.py` | multivariate input-shape convention per callable (`out/shapes.json`) |
| `apply_doc_fixes.py` | the reproducible docstring and stub fixes the sweep applied; reads `surface.json` and the pre-fix `docstrings.json` |

`surface.json` (the master table the report's appendix summarises) is
committed. `out/conventions.json` and `out/clusters.json` are regenerated
by their probes and are not.
