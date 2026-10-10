"""A4_draw_counts

Generated from the house-wide abort sweep (results.jsonl). Each call runs in
its own subprocess under `ulimit -v 4000000; timeout 60` with the sweep's
small seeded fixtures (../prelude.py). Expected: a refusal naming the
parameter; actual: see ACTUAL lines (rc=-6 is SIGABRT from the Rust
allocator's `memory allocation of N bytes failed`).
Run: python lab/audit/round16/abort_sweep/repro/repro_A4_draw_counts.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from repro_common import run

CASES = [
    ('n_draws', 'tsecon.bvar_irf_draws(data=X3, lags=2, horizon=8, n_draws=2**40)'),  # bvar_irf_draws:n_draws=2**40 -> abort 
    ('n_draws', 'tsecon.bvar_ssvs(data=X2, lags=1, n_draws=2**40, burn=50, horizon=4)'),  # bvar_ssvs:n_draws=2**40 -> abort tsecon_bayes::ssvs::bvar_ssvs
    ('n_chains', 'tsecon.bvar_ssvs(data=X2, lags=1, n_draws=200, burn=50, horizon=4, n_chains=2**31)'),  # bvar_ssvs:n_chains=2**31 -> abort tsecon_bayes::ssvs::bvar_ssvs
    ('n_boot', "tsecon.conformal_backtest(y=POS, method='enbpi', base='ar', lags=2, n_boot=2**31, seed=1, n_eval=20)"),  # conformal_backtest#enbpi:n_boot=2**31 -> abort tsecon_forecast::conformal::fit_enbpi_ensemble
    ('n_boot', "tsecon.conformal_forecast(y=POS, method='enbpi', base='ar', lags=2, n_boot=2**31, seed=1)"),  # conformal_forecast#enbpi:n_boot=2**31 -> abort tsecon_forecast::conformal::fit_enbpi_ensemble
    ('n_draws', 'tsecon.fry_pagan_svar(data=X3, restrictions=SR3, lags=1, horizon=4, n_draws=2**31, max_tries=100)'),  # fry_pagan_svar:n_draws=2**31 -> abort 
    ('n_boot', 'tsecon.hansen_seo_test(data=C2, n_grid=30, n_boot=2**31)'),  # hansen_seo_test:n_boot=2**31 -> abort tsecon_bootstrap::replicate::par_replicate::
    ('n_draws', "tsecon.historical_decomposition(data=X3, restrictions=SR3, identification='sign', lags=1, n_draws=2**31, max_tries=100)"),  # historical_decomposition#sign:n_draws=2**31 -> abort 
    ('n_seeds', 'tsecon.mlp_regression(x=XNC, y=YNC, hidden=[4], batch_size=32, max_epochs=20, patience=5, n_seeds=2**31)'),  # mlp_regression:n_seeds=2**31 -> abort tsecon_ml::mlp::mlp_regression
    ('n_draws', 'tsecon.narrative_svar(data=X3, sign_restrictions=SR3, lags=1, horizon=4, n_draws=2**31, max_tries=100, n_weight_draws=20)'),  # narrative_svar:n_draws=2**31 -> abort 
    ('n_boot', 'tsecon.proxy_svar_bands(data=X3, proxy=PROXY, lags=2, horizon=6, n_boot=2**31, block_length=5)'),  # proxy_svar_bands:n_boot=2**31 -> abort tsecon_var::proxy_bands::proxy_svar_bands
    ('n_trees', 'tsecon.random_forest(x=XNC, y=YNC, n_trees=2**31, max_features=2, max_depth=4)'),  # random_forest:n_trees=2**31 -> abort tsecon_ml::forest::random_forest
    ('n_permutations', "tsecon.random_forest(x=XNC, y=YNC, n_trees=20, importance='block_permutation', n_permutations=2**31, permutation_block=5)"),  # random_forest#perm:n_permutations=2**31 -> abort tsecon_ml::importance::block_permutation_importance
    ('n_draws', 'tsecon.robust_svar_bounds(data=X3, restrictions=SR3, lags=1, horizon=4, n_draws=2**31)'),  # robust_svar_bounds:n_draws=2**31 -> abort tsecon_ident::robust_bounds::robust_svar_bounds
    ('n_boot', 'tsecon.setar_test(y=SETY, p=1, n_boot=2**31)'),  # setar_test:n_boot=2**31 -> abort tsecon_bootstrap::replicate::par_replicate::
    ('n_draws', 'tsecon.sign_restricted_svar(data=X3, restrictions=SR3, lags=1, horizon=4, n_draws=2**31, max_tries=100)'),  # sign_restricted_svar:n_draws=2**31 -> abort 
    ('n_boot', 'tsecon.threshold_var_test(data=TV, p=1, n_grid=20, n_boot=2**31)'),  # threshold_var_test:n_boot=2**31 -> abort tsecon_bootstrap::replicate::par_replicate::
    ('n_boot', "tsecon.var_irf_bands(data=X2, lags=1, horizon=6, method='bootstrap', n_boot=2**31)"),  # var_irf_bands#boot:n_boot=2**31 -> abort tsecon_var::irf_bootstrap::run_bootstrap
    ('n_draws', 'tsecon.zero_sign_svar(data=X3, sign_restrictions=SR3, zero_restrictions=[(2, 0, 0)], lags=1, horizon=4, n_draws=2**31, max_tries=100)'),  # zero_sign_svar:n_draws=2**31 -> abort 
]

if __name__ == "__main__":
    run(CASES, timeout=60)
