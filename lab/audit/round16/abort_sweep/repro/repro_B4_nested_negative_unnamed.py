"""B4_nested_negative_unnamed

Generated from the house-wide abort sweep (results.jsonl). Each call runs in
its own subprocess under `ulimit -v 4000000; timeout 60` with the sweep's
small seeded fixtures (../prelude.py). Expected: a refusal naming the
parameter; actual: see ACTUAL lines (rc=-6 is SIGABRT from the Rust
allocator's `memory allocation of N bytes failed`).
Run: python lab/audit/round16/abort_sweep/repro/repro_B4_nested_negative_unnamed.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from repro_common import run

CASES = [
    ('restrictions', "tsecon.sign_restricted_svar(data=X3, restrictions=[(0, 0, -1, '+'), (1, 0, 0, '+')], lags=1, horizon=4, n_draws=30, max_tries=100)"),  # sign_restricted_svar#comp:restrictions[0][2]=-1 -> unnamed: 'sign_restricted_svar: an integer argument is negative, but the parameter counts something '
    ('sign_restrictions', "tsecon.narrative_svar(data=X3, sign_restrictions=[(0, 0, -1, '+'), (1, 0, 0, '+')], lags=1, horizon=4, n_draws=20, max_tries=100, n_weight_draws=20)"),  # narrative_svar#comp:sign_restrictions[0][2]=-1 -> unnamed: 'narrative_svar: an integer argument is negative, but the parameter counts something — a la'
    ('zero_restrictions', "tsecon.zero_sign_svar(data=X3, sign_restrictions=[(0, 0, 0, '+'), (1, 0, 0, '+')], zero_restrictions=[(2, 0, -1)], lags=1, horizon=4, n_draws=20, max_tries=100)"),  # zero_sign_svar#comp:zero_restrictions[0][2]=-1 -> unnamed: 'zero_sign_svar: an integer argument is negative, but the parameter counts something — a la'
    ('restrictions', "tsecon.fry_pagan_svar(data=X3, restrictions=[(0, 0, -1, '+'), (1, 0, 0, '+')], lags=1, horizon=4, n_draws=30, max_tries=100)"),  # fry_pagan_svar#comp:restrictions[0][2]=-1 -> unnamed: 'fry_pagan_svar: an integer argument is negative, but the parameter counts something — a la'
    ('restrictions', "tsecon.robust_svar_bounds(data=X3, restrictions=[(0, 0, -1, '+'), (1, 0, 0, '+')], lags=1, horizon=4, n_draws=30)"),  # robust_svar_bounds#comp:restrictions[0][2]=-1 -> unnamed: 'robust_svar_bounds: an integer argument is negative, but the parameter counts something — '
    ('restrictions', "tsecon.historical_decomposition(data=X3, restrictions=[(0, 0, -1, '+'), (1, 0, 0, '+')], identification='sign', lags=1, n_draws=20, max_tries=100)"),  # historical_decomposition#signcomp:restrictions[0][2]=-1 -> unnamed: 'historical_decomposition: an integer argument is negative, but the parameter counts someth'
    ('narrative_restrictions', "tsecon.narrative_svar(data=X3, sign_restrictions=[(0, 0, 0, '+'), (1, 0, 0, '+')], narrative_restrictions=[{'type': 'shock_sign', 'shock': 0, 'period': -1, 'sign': '+'}], lags=1, horizon=4, n_draws=20, max_tries=200, n_weight_draws=20)"),  # narrative_svar#narr:narrative_restrictions[0]['period']=-1 -> unnamed: 'narrative_svar: an integer argument is negative, but the parameter counts something — a la'
    ('narrative_restrictions', "tsecon.historical_decomposition(data=X3, restrictions=[(0, 0, 0, '+'), (1, 0, 0, '+')], identification='sign', narrative_restrictions=[{'type': 'shock_sign', 'shock': 0, 'period': -1, 'sign': '+'}], lags=1, n_draws=20, max_tries=200, n_weight_draws=20)"),  # historical_decomposition#narr:narrative_restrictions[0]['period']=-1 -> unnamed: 'historical_decomposition: an integer argument is negative, but the parameter counts someth'
]

if __name__ == "__main__":
    run(CASES, timeout=60)
