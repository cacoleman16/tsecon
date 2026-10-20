"""A7_output_sizes

Generated from the house-wide abort sweep (results.jsonl). Each call runs in
its own subprocess under `ulimit -v 4000000; timeout 60` with the sweep's
small seeded fixtures (../prelude.py). Expected: a refusal naming the
parameter; actual: see ACTUAL lines (rc=-6 is SIGABRT from the Rust
allocator's `memory allocation of N bytes failed`).
Run: python lab/audit/round16/abort_sweep/repro/repro_A7_output_sizes.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from repro_common import run

CASES = [
    ('maturities', 'tsecon.acm_term_premium(yields=YLD, maturities=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 2147483648], n_factors=3)'),  # acm_term_premium:maturities[-1]=2**31 -> abort tsecon_termstructure::acm::price_recursion
    ('n', 'tsecon.bootstrap_indices(n=2**31, seed=1, p=0.1)'),  # bootstrap_indices:n=2**31 -> abort tsecon_bootstrap::schemes::indices
    ('n', 'tsecon.cv_splits(n=2**31, train=50, horizon=2, step=5)'),  # cv_splits:n=2**31 -> abort 
    ('k', "tsecon.midas_weights(scheme='exp_almon', theta1=0.1, theta2=-0.01, k=2**31)"),  # midas_weights:k=2**31 -> abort tsecon_midas::weights::exp_almon_weights
    ('n', 'tsecon.philox_uniforms(seed=1, n=2**31)'),  # philox_uniforms:n=2**31 -> abort 
]

if __name__ == "__main__":
    run(CASES, timeout=60)
