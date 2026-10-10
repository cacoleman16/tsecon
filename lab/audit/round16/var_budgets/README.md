# Audit round 16: size budgets of the VAR family

Scripts behind the `var_budgets` fixes (VAR-family horizon, step,
replication and simulation budgets). Each `.out` is the script's output; the
`_before` files ran against the wheel built at the parent commit (`fdd90d3`),
the others against this branch, on the same Linux x86-64 machine.

| script | what it shows |
|---|---|
| `fingerprint_valid_calls.py` | FNV-1a over the IEEE bits of every number 33 valid calls return (`var_irf`, `var_fevd`, `connectedness`, `fvar_scenario`, `mean_group_var`, `long_run_svar`, both arms of `var_irf_bands` and `long_run_svar_bands` with and without simultaneous bands, bias correction, anchors and kept draws, `lp`, `smooth_lp`, `var_forecast`). `fingerprint_before.out` and `fingerprint_after.out` are identical: no valid output moved by a bit. The hashes are per platform. |
| `footprint.py` | Peak-RSS growth per horizon (or per replication), each case in a fresh process after a warm-up call. Against the counts in `crates/tsecon-var/src/budget.rs`: a `k x k` cube entry is counted at `8 ceil(k/8) k + 16` doubles (faer) plus `4 k^2 + (k + 1)(k + 10)` (Python list), times 8 copies. `long_run_svar` is the heaviest callable at 3.2 copies. In `footprint_after.out`, the cases above the new horizon budget are refused by name. The bootstrap replications fell from 3730 to 608 bytes (`var_irf_bands`) and from 4087 to 647 bytes (`long_run_svar_bands`) at horizon 12 on two variables, against 808 and 848 bytes counted. The `_bias` cases (`bias_correct=True`) were added in review; `footprint_bias_b292427.out` is the same cases at commit `b292427`, before the inner Kilian estimates moved into one store: 5644 bytes per replication on three variables against 2032 counted, 1190 after. |
| `vib_asym_cost.py` | Time of the delta-method arm against its Jacobian work `horizon (horizon + 1) / 2 * k^2 * lags k^2`. The cap is 2^28: 3.4 s at horizon 2047 on a bivariate VAR(8), and 5.4 s at horizon 28 on a 20-variable VAR(4). |
| `joint_cov_cost.py` | Time of `var_forecast`'s simultaneous-band joint covariance, which is cubic in `steps`. The 2^31 cap is about ten seconds (685 steps on two variables). |
| `refusals.py` | The message of each new size refusal, each naming the keyword and the largest value that fits. |
| `h16_03_lrsb_n_boot_4e6_after.out` | The auditor's `lrsb_budget_abort.py --abort` (H16-03) on this branch. `n_boot=4_000_000` at horizon 0 used to abort at a 4 GB cap; it now completes in 129 s at 1.12 GB. The script's "budget counts 64 values" text is the pre-fix constant; the count is now 58. |
| `bias_boundary.py` | The largest `n_boot` the bootstrap budget admits with `bias_correct=True` on three variables, run under a 4 GB address-space cap. `bias_boundary_b292427.out` (commit `b292427`, backtrace trimmed to the failing frames): `var_irf_bands` at the admitted 1056780 aborted in the allocator. `bias_boundary_after.out`: it completes in 70 s at a 1.16 GiB peak, and `long_run_svar_bands` at its admitted 1016750 in 72 s at 1.19 GiB. |
