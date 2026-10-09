# lab/bq_bands — evidence behind the Blanchard-Quah bands

These are the artifacts that produce every coverage and cross-check number
quoted for `long_run_svar_bands`, `long_run_svar(impact_signs=...)` and
`historical_decomposition(identification="impact")` on three pages:
`docs/reference/model-cards/structural-identification.md`,
`docs/examples/replication-blanchard-quah.md` and the CHANGELOG. Each is seeded
and writes its output to `out/`.

| Source | What it measures | Output |
|---|---|---|
| `mc_long_run_band_coverage` in `crates/tsecon-var/tests/long_run_bands_props.rs` (`#[ignore]`) | Per-cell coverage of the nominal-90% bands on three DGPs (a moderate VAR(2), a persistent VAR(2), and the page's VAR(8) as truth), for four arms (Efron, Hall from the same draws, Efron + Kilian, delta) and every object (plain, cumulative, level, long-run). 1,000 replications × 299 draws per design. | `out/mc_coverage.txt` |
| `guard_numbers.py` | The achieved values and the cross-seed drift quoted in `bindings/python/tests/test_replication_blanchard_quah.py`, with the native-vs-hand-built agreement | `out/guard_numbers.txt` |
| `card_examples.py` | The model card's two runnable examples, verbatim, with their printed output | `out/card_examples.txt` |
| `page_table_move.py` | How far the page's band table moved when it switched to the native stream | `out/page_table_move.txt` |

The Monte Carlo runs in about four minutes in release on the build machine:

```sh
cargo test --release -p tsecon-var --test long_run_bands_props \
    mc_long_run_band_coverage -- --ignored --nocapture > lab/bq_bands/out/mc_coverage.txt
```

The Python scripts run with the built wheel on the path. The page's own numbers
(the band tables, the Kilian and delta columns, the cross-check agreements) are
printed by `docs/examples/replication_blanchard_quah.py`.

The golden fixtures are `fixtures/long_run_svar_bands.json` and
`fixtures/historical_decomposition_impact.json`. Their generators,
`fixtures/generate_long_run_svar_bands_fixtures.py` and
`fixtures/generate_historical_decomposition_impact_fixtures.py`, never import
tsecon.
