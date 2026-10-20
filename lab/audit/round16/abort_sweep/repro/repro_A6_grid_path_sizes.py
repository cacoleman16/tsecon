"""A6_grid_path_sizes

Generated from the house-wide abort sweep (results.jsonl). Each call runs in
its own subprocess under `ulimit -v 4000000; timeout 60` with the sweep's
small seeded fixtures (../prelude.py). Expected: a refusal naming the
parameter; actual: see ACTUAL lines (rc=-6 is SIGABRT from the Rust
allocator's `memory allocation of N bytes failed`).
Run: python lab/audit/round16/abort_sweep/repro/repro_A6_grid_path_sizes.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from repro_common import run

CASES = [
    ('n_steps', 'tsecon.boosting(x=HIGHDIM_X, y=HIGHDIM_Y, n_steps=2**31)'),  # boosting:n_steps=2**31 -> abort tsecon_ml::boosting::boosting
    ('n_grid', 'tsecon.bvar_hierarchical(data=X3, lags=2, n_grid=2**31)'),  # bvar_hierarchical:n_grid=2**31 -> abort tsecon_bayes::hierarchical::bvar_hierarchical
    ('rff_features', 'tsecon.kernel_ridge(x=XNC, y=YNC, rff_features=2**31)'),  # kernel_ridge:rff_features=2**31 -> abort tsecon_ml::kernel_ridge::kernel_ridge
    ('n_lambdas', 'tsecon.lasso_path(x=HIGHDIM_X, y=HIGHDIM_Y, n_lambdas=2**31)'),  # lasso_path:n_lambdas=2**31 -> abort tsecon_ml::path::regularization_path
    ('hidden', 'tsecon.mlp_regression(x=XNC, y=YNC, hidden=[2147483648], batch_size=32, max_epochs=20, patience=5, n_seeds=1)'),  # mlp_regression:hidden[0]=2**31 -> abort tsecon_ml::mlp::train_member
    ('max_epochs', 'tsecon.mlp_regression(x=XNC, y=YNC, hidden=[4], batch_size=32, max_epochs=2**31, patience=5, n_seeds=1)'),  # mlp_regression:max_epochs=2**31 -> abort tsecon_ml::mlp::train_member
    ('n_gamma', 'tsecon.star(y=SETY, p=1, n_gamma=2**31, n_c=5)'),  # star:n_gamma=2**31 -> abort tsecon_regime::star::star
    ('n_c', 'tsecon.star(y=SETY, p=1, n_gamma=5, n_c=2**31)'),  # star:n_c=2**31 -> abort tsecon_regime::star::star
    ('n_grid_beta', 'tsecon.threshold_vecm(data=C2, n_grid_gamma=20, n_grid_beta=2**31)'),  # threshold_vecm#beta:n_grid_beta=2**31 -> abort tsecon_coint::tvecm::threshold_vecm
]

if __name__ == "__main__":
    run(CASES, timeout=60)
