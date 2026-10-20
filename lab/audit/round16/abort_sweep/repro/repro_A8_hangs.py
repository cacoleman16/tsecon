"""A8_hangs

Generated from the house-wide abort sweep (results.jsonl). Each call runs in
its own subprocess under `ulimit -v 4000000; timeout 60` with the sweep's
small seeded fixtures (../prelude.py). Expected: a refusal naming the
parameter; actual: see ACTUAL lines (rc=-6 is SIGABRT from the Rust
allocator's `memory allocation of N bytes failed`).
Run: python lab/audit/round16/abort_sweep/repro/repro_A8_hangs.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from repro_common import run

CASES = [
    ('n_weight_draws', "tsecon.historical_decomposition(data=X3, restrictions=[(0, 0, 0, '+'), (1, 0, 0, '+')], identification='sign', narrative_restrictions=[{'type': 'shock_sign', 'shock': 0, 'period': 10, 'sign': '+'}], lags=1, n_draws=20, max_tries=200, n_weight_draws=2**31)"),  # historical_decomposition#narr:n_weight_draws=2**31 -> timeout 
    ('inner_iter', 'tsecon.mstl(y=SEAS, periods=[12], windows=[7], iterate=2, trend=23, low_pass=13, inner_iter=2**31, outer_iter=0)'),  # mstl:inner_iter=2**31 -> timeout 
    ('outer_iter', 'tsecon.mstl(y=SEAS, periods=[12], windows=[7], iterate=2, trend=23, low_pass=13, inner_iter=2, outer_iter=2**31)'),  # mstl:outer_iter=2**31 -> timeout 
    ('n_weight_draws', "tsecon.narrative_svar(data=X3, sign_restrictions=[(0, 0, 0, '+'), (1, 0, 0, '+')], narrative_restrictions=[{'type': 'shock_sign', 'shock': 0, 'period': 10, 'sign': '+'}], lags=1, horizon=4, n_draws=20, max_tries=200, n_weight_draws=2**31)"),  # narrative_svar#narr:n_weight_draws=2**31 -> timeout 
    ('inner_iter', 'tsecon.stl(y=SEAS, period=12, seasonal=7, trend=23, low_pass=13, inner_iter=2**31, outer_iter=0)'),  # stl:inner_iter=2**31 -> timeout 
    ('outer_iter', 'tsecon.stl(y=SEAS, period=12, seasonal=7, trend=23, low_pass=13, inner_iter=2, outer_iter=2**31)'),  # stl:outer_iter=2**31 -> timeout 
]

if __name__ == "__main__":
    run(CASES, timeout=60)
