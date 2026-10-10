"""A1_irf_horizon

Generated from the house-wide abort sweep (results.jsonl). Each call runs in
its own subprocess under `ulimit -v 4000000; timeout 60` with the sweep's
small seeded fixtures (../prelude.py). Expected: a refusal naming the
parameter; actual: see ACTUAL lines (rc=-6 is SIGABRT from the Rust
allocator's `memory allocation of N bytes failed`).
Run: python lab/audit/round16/abort_sweep/repro/repro_A1_irf_horizon.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from repro_common import run

CASES = [
    ('horizon', 'tsecon.bvar_irf_draws(data=X3, lags=2, horizon=2**31, n_draws=50)'),  # bvar_irf_draws:horizon=2**31 -> abort tsecon_bayes::niw::cholesky_irf
    ('horizon', 'tsecon.bvar_ssvs(data=X2, lags=1, n_draws=200, burn=50, horizon=2**40)'),  # bvar_ssvs:horizon=2**40 -> abort tsecon_bayes::niw::cholesky_irf
    ('horizon', 'tsecon.connectedness(data=X3, lags=2, horizon=2**31)'),  # connectedness:horizon=2**31 -> abort tsecon_var::irf::ma_rep
    ('horizon', 'tsecon.favar(panel=DFMP, policy=_ar1(80, 3), n_factors=1, lags=1, horizon=2**31, slow_indices=[0, 1])'),  # favar:horizon=2**31 -> abort 
    ('horizon', 'tsecon.fry_pagan_svar(data=X3, restrictions=SR3, lags=1, horizon=2**31, n_draws=30, max_tries=100)'),  # fry_pagan_svar:horizon=2**31 -> abort tsecon_bayes::niw::cholesky_irf
    ('horizon', 'tsecon.fvar_scenario(y=CY_OUT, curves=CURVES, delta=DELTA, n_factors=2, lags=1, horizon=2**31)'),  # fvar_scenario:horizon=2**31 -> abort tsecon_var::irf::ma_rep
    ('horizon', 'tsecon.hetero_svar(data=X2 * np.where(REG[:, None] == 1, 2.0, 1.0) * np.array([1.0, 0.5]), regime_labels=REG, lags=1, horizon=2**31)'),  # hetero_svar:horizon=2**31 -> abort tsecon_var::irf::ma_rep
    ('horizon', 'tsecon.long_run_svar(data=X2, lags=2, horizon=2**31)'),  # long_run_svar:horizon=2**31 -> abort tsecon_ident::long_run::long_run_svar
    ('horizon', 'tsecon.max_share_svar(data=X3, lags=1, target=0, h0=0, h1=8, horizon=2**31)'),  # max_share_svar:horizon=2**31 -> abort tsecon_var::irf::ma_rep
    ('horizon', 'tsecon.mean_group_var(entities=ENT, lags=1, horizon=2**31)'),  # mean_group_var:horizon=2**31 -> abort tsecon_var::irf::ma_rep
    ('horizon', 'tsecon.narrative_svar(data=X3, sign_restrictions=SR3, lags=1, horizon=2**40, n_draws=20, max_tries=100, n_weight_draws=20)'),  # narrative_svar:horizon=2**40 -> abort tsecon_bayes::niw::cholesky_irf
    ('horizon', 'tsecon.nongaussian_svar(data=np.column_stack([G[:200], G2[:200, 1]]) + 0.3 * np.column_stack([G2[:200,1], G[:200]]), lags=1, horizon=2**31, max_iter=100)'),  # nongaussian_svar:horizon=2**31 -> abort tsecon_ident::summary::structural_ma
    ('horizon', 'tsecon.proxy_ar_sets(data=X3, proxy=PROXY, lags=2, horizon=2**31)'),  # proxy_ar_sets:horizon=2**31 -> abort tsecon_var::irf::ma_rep
    ('horizon', 'tsecon.proxy_svar(data=X3, proxy=PROXY, lags=2, horizon=2**31)'),  # proxy_svar:horizon=2**31 -> abort tsecon_var::irf::ma_rep
    ('horizon', 'tsecon.proxy_svar_bands(data=X3, proxy=PROXY, lags=2, horizon=2**31, n_boot=30, block_length=5)'),  # proxy_svar_bands:horizon=2**31 -> abort tsecon_var::irf::ma_rep
    ('horizon', 'tsecon.robust_svar_bounds(data=X3, restrictions=SR3, lags=1, horizon=2**31, n_draws=30)'),  # robust_svar_bounds:horizon=2**31 -> abort tsecon_ident::robust_bounds::robust_svar_bounds
    ('horizon', 'tsecon.sign_restricted_svar(data=X3, restrictions=SR3, lags=1, horizon=2**31, n_draws=30, max_tries=100)'),  # sign_restricted_svar:horizon=2**31 -> abort tsecon_bayes::niw::cholesky_irf
    ('horizon', 'tsecon.structural_fevd(data=X2, lags=2, horizon=2**31)'),  # structural_fevd:horizon=2**31 -> abort tsecon_bayes::niw::cholesky_irf
    ('horizon', 'tsecon.var_fevd(data=X2, lags=2, horizon=2**31)'),  # var_fevd:horizon=2**31 -> abort tsecon_var::irf::ma_rep
    ('horizon', 'tsecon.var_irf(data=X2, lags=2, horizon=2**31)'),  # var_irf:horizon=2**31 -> abort tsecon_var::irf::ma_rep
    ('horizon', 'tsecon.var_irf_bands(data=X2, lags=2, horizon=2**31)'),  # var_irf_bands:horizon=2**31 -> abort tsecon_var::irf::ma_rep
    ('horizon', 'tsecon.zero_sign_svar(data=X3, sign_restrictions=SR3, zero_restrictions=[(2, 0, 0)], lags=1, horizon=2**31, n_draws=20, max_tries=100)'),  # zero_sign_svar:horizon=2**31 -> abort tsecon_bayes::niw::cholesky_irf
]

if __name__ == "__main__":
    run(CASES, timeout=60)
