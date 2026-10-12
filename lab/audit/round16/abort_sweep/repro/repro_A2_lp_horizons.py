"""A2_lp_horizons

Generated from the house-wide abort sweep (results.jsonl). Each call runs in
its own subprocess under `ulimit -v 4000000; timeout 60` with the sweep's
small seeded fixtures (../prelude.py). Expected: a refusal naming the
parameter; actual: see ACTUAL lines (rc=-6 is SIGABRT from the Rust
allocator's `memory allocation of N bytes failed`).
Run: python lab/audit/round16/abort_sweep/repro/repro_A2_lp_horizons.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from repro_common import run

CASES = [
    ('horizons', 'tsecon.flp(y=CY_OUT, scores=CURVES[:, :2], horizons=2**31, hac_maxlags=3)'),  # flp:horizons=2**31 -> abort tsecon_funcshock::flp::flp
    ('horizons', 'tsecon.flp_scenario(y=CY_OUT, curves=CURVES, delta=DELTA, n_factors=2, horizons=2**31, hac_maxlags=3)'),  # flp_scenario:horizons=2**31 -> abort tsecon_funcshock::flp::flp
    ('horizons', "tsecon.lp(y=Y, shock=Y2, horizons=2**31, n_lag_controls=2, se='hac', maxlags=3)"),  # lp:horizons=2**31 -> abort tsecon_lp::level::lp
    ('horizons', 'tsecon.lp_iv(y=Y, impulse=Y2 + 0.5 * Y3, instrument=Y3, horizons=2**31, n_lag_controls=2)'),  # lp_iv:horizons=2**31 -> abort tsecon_lp::iv::lp_iv
    ('horizons', 'tsecon.lp_multiplier(y=Y, impulse=Y2 + 0.5 * Y3, instrument=Y3, horizons=2**31, n_lag_controls=2, maxlags=3)'),  # lp_multiplier:horizons=2**31 -> abort tsecon_lp::iv::lp_multiplier
    ('horizons', "tsecon.lp_state(y=Y, shock=Y2, state_indicator=REG.astype(float), horizons=2**31, n_lag_controls=2, se='hac', maxlags=3)"),  # lp_state:horizons=2**31 -> abort tsecon_lp::state::lp_state
    ('horizons', 'tsecon.quantile_lp(y=Y, shock=Y2, horizons=2**31, n_lag_controls=2, taus=[0.25, 0.5, 0.75])'),  # quantile_lp:horizons=2**31 -> abort tsecon_quantile::lp::quantile_lp
]

if __name__ == "__main__":
    run(CASES, timeout=60)
