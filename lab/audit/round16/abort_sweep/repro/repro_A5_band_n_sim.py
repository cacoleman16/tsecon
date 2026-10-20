"""A5_band_n_sim

Generated from the house-wide abort sweep (results.jsonl). Each call runs in
its own subprocess under `ulimit -v 4000000; timeout 60` with the sweep's
small seeded fixtures (../prelude.py). Expected: a refusal naming the
parameter; actual: see ACTUAL lines (rc=-6 is SIGABRT from the Rust
allocator's `memory allocation of N bytes failed`).
Run: python lab/audit/round16/abort_sweep/repro/repro_A5_band_n_sim.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from repro_common import run

CASES = [
    ('band_n_sim', "tsecon.lp(y=Y, shock=Y2, horizons=6, n_lag_controls=2, band='sup-t', band_n_sim=2**31)"),  # lp#supt:band_n_sim=2**31 -> abort tsecon_lp::band::build_band
    ('band_n_sim', "tsecon.smooth_lp(y=Y, shock=Y2, horizons=6, n_lag_controls=2, lam=1.0, band='sup-t', band_n_sim=2**31)"),  # smooth_lp#supt:band_n_sim=2**31 -> abort tsecon_lp::band::build_band
    ('band_n_sim', "tsecon.var_forecast(data=X2, lags=2, steps=4, band='sup-t', band_n_sim=2**31)"),  # var_forecast#supt:band_n_sim=2**31 -> abort 
    ('band_n_sim', "tsecon.var_irf_bands(data=X2, lags=1, horizon=6, band='sup-t', band_n_sim=2**31)"),  # var_irf_bands#supt:band_n_sim=2**31 -> abort tsecon_var::irf_asymptotic::irf_asymptotic_critical_values
]

if __name__ == "__main__":
    run(CASES, timeout=60)
