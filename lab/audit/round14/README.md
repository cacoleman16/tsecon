# Audit round 14 — the post-wave sweep over the 0.10.0 surface

Six finder/refuter sweeps over the thirteen callables the 0.10.0 wave added
(`unobserved_components`, `tvp_regression`, `ets_fit`, `auto_ets`,
`var_conditional_forecast`, `var_diagnostics`, `var_select_order`,
`spa_test`, `model_confidence_set`, `stepm_test`, `fmols`, `dols`, `ccr`),
the `mask=` parameter added to `panel_fe`, `panel_distributed_lag`,
`panel_lp` and `lp_did`, and the Blanchard-Quah replication page. The report
is [`docs/roadmap/29-audit-round-14-findings.md`](../../../docs/roadmap/29-audit-round-14-findings.md).

Run every script from the repository root with the release extension built
into the venv:

| script | sweep | output |
|---|---|---|
| `sweep_e_contract.py` | E — result-object contract, keys, shapes, the library-wide container census | `out/sweep_e.txt`, `sweep_e.json`, `sweep_e_conventions.txt` |
| `sweep_f_drift.py` | F — signature / stub / docstring drift, the hand-transcribed value table, the inert-keyword contract, the mask identity | `out/sweep_f.txt`, `sweep_f.json` |
| `sweep_g_timing.py [--caps-only\|--timing-only]` | G — three-size timing with the log-log slope; every allocation-sizing count at 2^47 / 2^31 / 10^6 / the documented cap ± 1 | `out/sweep_g.txt`, `sweep_g_caps.txt`, `sweep_g*.json` |
| `sweep_h_seed.py` | H — the seed contract, determinism, 1 vs 4 rayon threads | `out/sweep_h.txt`, `sweep_h.json` |
| `sweep_s14_malformed.py [--only name]` | S — 1563 malformed-input cells over the new surface, the masks and the nested `conditions` | `out/sweep_s14.txt`, `sweep_s14_cells.json` |
| `sweep_c_claims.py` | C — every quoted number against a corpus of committed artifacts; every runnable card example executed and diffed | `out/sweep_c.txt`, `sweep_c.json` |

`out/rust_props.txt` is the `--nocapture` output of the fourteen Rust
property and golden binaries the new cards cite, and `out/sweep_c_pytest.txt`
the `-s` output of the wave's nine pytest files; both feed sweep C's corpus
and are produced by hand:

    CARGO_TARGET_DIR=$PWD/target-int cargo test -p <crate> --release \
        --test <binary> -- --nocapture   >> lab/audit/round14/out/rust_props.txt
    .venv/bin/python -m pytest -q -s -p no:cacheprovider \
        bindings/python/tests/test_{ets,mcs,uc,var_cf,fmols,panel_unbalanced,replication_blanchard_quah,api_audit,security_audit}.py \
        > lab/audit/round14/out/sweep_c_pytest.txt

`registry.py` extends round 13's registry (179 entries) with the thirteen new
callables, four `name@mask` pseudo-entries that call the same function with an
unbalanced observation mask, and `auto_ets@mul` (a series whose ETS winner is
multiplicative, the only configuration on which `auto_ets`'s `seed` is live).
`cell_runner.py`/`parent.py` are round 13's one-cell-per-child protocol with
two fixes: a `name@suffix` is resolved to its real callable, and the `also`
key — which round 13 recorded on a cell and never applied — now sets the
extra keywords that make the mutated slot live.

## The earlier re-run in this directory

`sweep_s_malformed.py` and `out/sweep_s.txt` / `out/sweep_s_cells.json` are
**not** part of round 14's six sweeps: they are the 0.10.0 hygiene slice's
re-run of round 13's 673-cell malformed-input sweep over the *0.9.0* six,
counting what its refusal-naming pass left unnamed. They are kept as the
before/after record for that pass; round 14's own malformed-input sweep is
`sweep_s14_malformed.py`.

The in-process replay of round 13's 83 cells is the CI tripwire
`bindings/python/tests/test_security_audit.py::test_every_sweep_s_refusal_names_the_offending_parameter`;
round 14's confirmed findings are pinned in
`bindings/python/tests/test_audit_round14.py`.
