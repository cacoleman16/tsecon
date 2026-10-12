# lab/stl_cost — the STL/MSTL iteration budget

The evidence behind `STL_WORK_BUDGET` (`crates/tsecon-filters/src/stl.rs`),
added in audit round 16: `inner_iter`, `outer_iter` and MSTL's `iterate` are
fixed loop counts, and a request of more than 32 inner passes per fit (the
robust default's count) whose LOESS work exceeds 2^32 evaluation units is
refused by name before any smoothing. Run with the built wheel on the path;
each part writes `out/stl_cost_<part>.txt`.

| Part | What it measures |
|---|---|
| `rates` | Seconds per inner pass and ns per LOESS unit across series lengths, periods, windows and jumps |
| `cap` | The largest admissible `inner_iter` on the 96-observation monthly series of the audit, run once |
| `refusals` | `inner_iter`, `outer_iter` and `iterate` at 2**31, refused, with the quoted maximum |
