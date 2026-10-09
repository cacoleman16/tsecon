# Audit round 16 — structural identification and Bayesian fixes

Scripts behind the numbers quoted by the round-16 fixes to the
structural-identification and Bayesian callables (size budgets, keyword-named
refusals, extreme-scale impact matrices). Each prints the output stored
under `out/`.

| Script | What it measures | Quoted in |
|---|---|---|
| `cost_narrative.py` | ns per work unit of the narrative weight loop, `n_draws * n_weight_draws * (T_eff n^2 + (h + 3) n^3)` | `NARRATIVE_WEIGHT_WORK` (2^36, "about a minute and a half") in `crates/tsecon-ident/src/narrative.rs` |
| `cost_proxy_ar_delta.py` | ns per work unit of the `proxy_ar_sets` delta-method Jacobians, quadratic in `horizon` | `PROXY_AR_DELTA_WORK` (2^32, "under half a minute") in `crates/tsecon-ident/src/proxy_ar.rs` |
| `hd_scale_range.py` | the impact scales at which `historical_decomposition(identification="impact")` works (exact to 1e-15) and where it is refused | the structural-identification card's `historical_decomposition` passage |
| `bit_identity.py` | a battery of 28 valid seeded calls, pickled under the baseline wheel and the fixed wheel and compared bit for bit | "valid calls are bit-identical" in the CHANGELOG entry |

`bit_identity.py dump OUT.pkl` runs the battery against the installed
`tsecon`; `bit_identity.py compare A.pkl B.pkl` compares two dumps. The
stored comparison is baseline `fdd90d3` against the fixed wheel.

The timings share a 4-core machine with other work, so they are indicative:
the two runs made while writing the fixes measured 1.1–1.35 and 1.2–1.3 ns
per narrative unit, and 2.0–6.0 and 1.6–5.6 ns per delta-method unit.
