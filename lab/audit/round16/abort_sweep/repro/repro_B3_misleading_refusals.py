"""B3_misleading_refusals

Generated from the house-wide abort sweep (results.jsonl). Each call runs in
its own subprocess under `ulimit -v 4000000; timeout 60` with the sweep's
small seeded fixtures (../prelude.py). Expected: a refusal naming the
parameter; actual: see ACTUAL lines (rc=-6 is SIGABRT from the Rust
allocator's `memory allocation of N bytes failed`).
Run: python lab/audit/round16/abort_sweep/repro/repro_B3_misleading_refusals.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from repro_common import run

CASES = [
    ('scale_ar', 'tsecon.bvar_hierarchical(data=X3, lags=2, scale_ar=2**31)'),  # bvar_hierarchical:scale_ar=2**31 -> unnamed: 'hierarchical lambda1 optimization failed to converge; the posterior is very poorly conditi'
    ('scale_ar', 'tsecon.bvar_fit(data=X3, lags=2, scale_ar=2**31)'),  # bvar_fit:scale_ar=2**31 -> unnamed: 'this model needs at least 4294967298 rows for the AR(p) scale regression that calibrates t'
    ('scale_ar', 'tsecon.bvar_irf_draws(data=X3, lags=2, horizon=8, n_draws=50, scale_ar=2**31)'),  # bvar_irf_draws:scale_ar=2**31 -> unnamed: 'this model needs at least 4294967298 rows for the AR(p) scale regression that calibrates t'
    ('degree', "tsecon.kernel_ridge(x=XNC, y=YNC, kernel='polynomial', degree=2**31)"),  # kernel_ridge#poly:degree=2**31 -> unnamed: 'not positive definite: the kernel ridge system K + alpha*I (n = 200) failed its Cholesky f'
    ('max_iter', 'tsecon.lasso_path(x=HIGHDIM_X, y=HIGHDIM_Y, n_lambdas=20, max_iter=0)'),  # lasso_path:max_iter=0 -> unnamed: 'coordinate descent did not converge after 1 sweeps (last max coefficient change 2.80289682'
    ('max_iter', 'tsecon.pds_lasso(y=HIGHDIM_Y, d=HD_D, x=HIGHDIM_X, hac_lags=3, max_iter=0)'),  # pds_lasso:max_iter=0 -> unnamed: 'coordinate descent did not converge after 1 sweeps (last max coefficient change 6.19755251'
    ('n_lag_controls', 'tsecon.quantile_lp(y=Y, shock=Y2, horizons=3, n_lag_controls=2**31, taus=[0.25, 0.5, 0.75])'),  # quantile_lp:n_lag_controls=2**31 -> unnamed: 'horizon 0 leaves nobs = 0 usable observations from a series of length 200, but the design '
]

if __name__ == "__main__":
    run(CASES, timeout=60)
