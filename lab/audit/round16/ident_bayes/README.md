# Audit round 16 — structural identification and Bayesian fixes

Scripts behind the numbers quoted by the round-16 fixes to the
structural-identification and Bayesian callables (size budgets, keyword-named
refusals, extreme-scale impact matrices). Each prints the output stored
under `out/`.

| Script | What it measures | Quoted in |
|---|---|---|
| `cost_narrative.py` | ns per work unit of the narrative weight loop, `n_draws * n_weight_draws * (T_eff n^2 + (h + 3) n^3)` | `NARRATIVE_WEIGHT_WORK` (2^35, "at most about a minute": 0.6 to 2.0 ns per unit) in `crates/tsecon-ident/src/narrative.rs` |
| `cost_proxy_ar_delta.py` | ns per work unit of the `proxy_ar_sets` delta-method Jacobians, quadratic in `horizon` | `PROXY_AR_DELTA_WORK` (2^32, "under half a minute") in `crates/tsecon-ident/src/proxy_ar.rs` |
| `hd_scale_range.py` | the impact scales at which `historical_decomposition(identification="impact")` works (exact to 1e-15) and where it is refused | the structural-identification card's `historical_decomposition` passage |
| `boundaries.py` | both sides of every size budget: the quoted largest value runs (time, peak RSS) and the next is refused by name | the "largest value that fits" of every refusal; the per-replication MA path of `proxy_svar_bands`; the combined checks of `narrative_svar`, `historical_decomposition(identification="sign")` and `proxy_ar_sets` |
| `bit_identity.py` | a battery of 28 valid seeded calls, pickled under the baseline wheel and the fixed wheel and compared bit for bit | "valid calls are bit-identical" in the CHANGELOG entry |

`bit_identity.py dump OUT.pkl` runs the battery against the installed
`tsecon`; `bit_identity.py compare A.pkl B.pkl` compares two dumps. The
stored comparison is baseline `fdd90d3` against the fixed wheel.

The timings share a 4-core machine with other work, so they are indicative:
the runs made while writing the fixes measured 1.1–1.35 and 1.2–1.3 ns per
narrative unit on the 4-variable case, and 2.0–6.0 and 1.6–5.6 ns per
delta-method unit. The review run stored in `out/cost_narrative.txt` adds a
3-variable case where most draws are accepted (1.96 ns per unit; the
4-variable case then ran at 0.63–0.64), which is why the narrative cap was
lowered from 2^36 to 2^35: 22 to 67 s at the cap, not up to two minutes and
a half.

`boundaries.py` probes both sides of every size budget of these callables
in capped subprocesses (`ulimit -v 4000000`): the largest value a refusal
quotes must run, the next must be refused by name. `out/boundaries.txt`
holds its output (time and peak RSS of each boundary run). All but one
quoted value ran to completion within the 4 GB cap. `robust_svar_bounds` at
its largest `n_draws` (6.9 million draws at horizon 4) was still running at
the probe's 200 s deadline: its loop length is the caller's own draw count,
bounded in memory but not in work. The `proxy_ar_sets` second-order
simulation at its largest horizon, likewise unbounded in work, took 125 s.
