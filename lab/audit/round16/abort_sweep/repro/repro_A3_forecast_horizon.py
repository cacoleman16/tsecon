"""A3_forecast_horizon

Generated from the house-wide abort sweep (results.jsonl). Each call runs in
its own subprocess under `ulimit -v 4000000; timeout 60` with the sweep's
small seeded fixtures (../prelude.py). Expected: a refusal naming the
parameter; actual: see ACTUAL lines (rc=-6 is SIGABRT from the Rust
allocator's `memory allocation of N bytes failed`).
Run: python lab/audit/round16/abort_sweep/repro/repro_A3_forecast_horizon.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from repro_common import run

CASES = [
    ('forecast_horizon', 'tsecon.ccc_garch(returns=G2, forecast_horizon=2**31)'),  # ccc_garch:forecast_horizon=2**31 -> abort 
    ('horizon', "tsecon.conformal_forecast(y=POS, method='enbpi', base='ar', lags=2, n_boot=10, seed=1, horizon=2**31)"),  # conformal_forecast#enbpi:horizon=2**31 -> abort tsecon_forecast::conformal::enbpi
    ('forecast_horizon', 'tsecon.dcc_garch(returns=G2, forecast_horizon=2**31)'),  # dcc_garch:forecast_horizon=2**31 -> abort 
    ('horizon', 'tsecon.gas_volatility(y=G, horizon=2**31)'),  # gas_volatility:horizon=2**31 -> abort tsecon_gas::model::forecast_from
    ('steps', 'tsecon.theta_forecast(y=SEAS, steps=2**31, period=12)'),  # theta_forecast:steps=2**31 -> abort tsecon_forecast::theta::theta_forecast_with
]

if __name__ == "__main__":
    run(CASES, timeout=60)
