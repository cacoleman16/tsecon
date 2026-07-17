//! Golden-value tests against `fixtures/favar.json`: PCA on a
//! standardized `n = 300` by `N = 24` two-factor panel via SVD.
//! Eigenvalues (`s^2 / n`), the absolute principal components `|PC1|`,
//! `|PC2|`, and the absolute first-factor loadings match NumPy to `1e-6`.
//! Factors are identified only up to sign, so the fixture and this crate
//! store / compare absolute values.

mod common;

use common::{as_mat, as_vec, assert_rel_close, load_fixture};
use tsecon_favar::FactorModel;
use tsecon_linalg::faer::Mat;

// The fixture stores X_standardized as N x n (series in rows); the model
// wants n x N (observations in rows), so transpose.
fn panel() -> Mat<f64> {
    let fx = load_fixture("favar.json");
    let xt = as_mat(&fx["X_standardized"]); // 24 x 300
    Mat::from_fn(xt.ncols(), xt.nrows(), |i, j| xt[(j, i)])
}

#[test]
fn golden_dimensions() {
    let fx = load_fixture("favar.json");
    let model = FactorModel::fit(panel().as_ref()).unwrap();
    assert_eq!(model.n_obs(), fx["n"].as_u64().unwrap() as usize);
    assert_eq!(model.n_series(), fx["big_n"].as_u64().unwrap() as usize);
    assert_eq!(model.max_factors(), 24);
}

#[test]
fn golden_eigenvalues() {
    let fx = load_fixture("favar.json");
    let model = FactorModel::fit(panel().as_ref()).unwrap();
    let expected = as_vec(&fx["eigenvalues"]);
    assert_eq!(model.eigenvalues().len(), expected.len());
    for (k, &e) in expected.iter().enumerate() {
        assert_rel_close(model.eigenvalues()[k], e, 1e-6, &format!("eigenvalue[{k}]"));
    }
    // Eigenvalues of a standardized panel sum to N.
    let total: f64 = model.eigenvalues().iter().sum();
    assert_rel_close(total, 24.0, 1e-9, "sum of eigenvalues");
}

#[test]
fn golden_absolute_principal_components() {
    let fx = load_fixture("favar.json");
    let model = FactorModel::fit(panel().as_ref()).unwrap();
    let factors = model.factors(2).unwrap();
    let pc1 = as_vec(&fx["pc1_abs"]);
    let pc2 = as_vec(&fx["pc2_abs"]);
    for t in 0..model.n_obs() {
        assert_rel_close(factors[(t, 0)].abs(), pc1[t], 1e-6, &format!("|PC1|[{t}]"));
        assert_rel_close(factors[(t, 1)].abs(), pc2[t], 1e-6, &format!("|PC2|[{t}]"));
    }
}

#[test]
fn golden_absolute_loadings() {
    let fx = load_fixture("favar.json");
    let model = FactorModel::fit(panel().as_ref()).unwrap();
    let loadings = model.loadings(1).unwrap();
    let l1 = as_vec(&fx["loadings_pc1_abs"]);
    for i in 0..model.n_series() {
        assert_rel_close(loadings[(i, 0)].abs(), l1[i], 1e-6, &format!("|L1|[{i}]"));
    }
}

#[test]
fn golden_sign_convention_largest_loading_positive() {
    // The documented convention: the largest-magnitude loading in each
    // column is positive. Verify on the first two factors.
    let model = FactorModel::fit(panel().as_ref()).unwrap();
    let loadings = model.loadings(2).unwrap();
    for k in 0..2 {
        let mut best = 0usize;
        let mut best_abs = 0.0f64;
        for i in 0..model.n_series() {
            if loadings[(i, k)].abs() > best_abs {
                best_abs = loadings[(i, k)].abs();
                best = i;
            }
        }
        assert!(
            loadings[(best, k)] > 0.0,
            "factor {k}: largest-magnitude loading (row {best}) should be positive"
        );
    }
}

#[test]
fn golden_eigenvalue_ratio_picks_two() {
    // Ahn-Horenstein ER on the fixture: eigenvalues about
    // [11, 9, 0.8, 0.6, ...], so the ratio spikes at k = 2.
    let model = FactorModel::fit(panel().as_ref()).unwrap();
    let (r_hat, ratios) = tsecon_favar::eigenvalue_ratio(model.eigenvalues(), 8).unwrap();
    assert_eq!(
        r_hat, 2,
        "eigenvalue-ratio estimator should pick 2, ratios = {ratios:?}"
    );
}

#[test]
fn golden_full_rank_reconstruction_is_exact() {
    // Z_hat with all N components reproduces the standardized panel.
    let model = FactorModel::fit(panel().as_ref()).unwrap();
    let recon = model.reconstruct_standardized(model.max_factors()).unwrap();
    let z = panel(); // already standardized (idempotent re-standardization)
    for i in 0..model.n_obs() {
        for j in 0..model.n_series() {
            assert!(
                (recon[(i, j)] - z[(i, j)]).abs() < 1e-9,
                "reconstruction[{i},{j}]: {} vs {}",
                recon[(i, j)],
                z[(i, j)]
            );
        }
    }
}
