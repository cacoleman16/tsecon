# lab/unitroot_cost — the lag budget of the unit-root and ARCH-LM tests

The evidence behind the shared lag-design budget of `tsecon-diag`
(`crates/tsecon-diag/src/budget.rs`), added in audit round 16: `arch_lm`,
`adf`, `dfgls`, `ng_perron` and the `zivot_andrews` lag search refuse, by
name and before any lag column is built, an order whose lag design exceeds
2^26 entries or whose regression or lag search exceeds 2^35 multiply-adds
(orders up to the default rule's are exempt from the work budget). Run with
the built wheel on the path; each part writes `out/lag_cost_<part>.txt`.

| Part | What it measures |
|---|---|
| `adf` | The default `adf` search at n = 1e3 to 1e5: seconds and ns per multiply-add |
| `adf_cap` | The largest admissible explicit `adf` search at n = 20,000 (the work budget binds), run once |
| `arch_cap` | The largest admissible `arch_lm` order at n = 30,000 (the work budget binds), run once |
| `refusals` | The orders that used to abort the interpreter at n = 1e6, refused, with the quoted maximum |
| `thresholds` | The longest series whose default order fits the memory budget (formula only) |
