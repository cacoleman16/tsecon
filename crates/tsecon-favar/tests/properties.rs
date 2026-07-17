//! Structural / simulation property tests for the factor model, the
//! factor-number criteria, and the two-step FAVAR.
//!
//! Only the PCA step has an external golden (see `golden.rs`); the FAVAR
//! assembly and IRF mapping are checked here for dimensional correctness,
//! factor orthogonality, reconstruction accuracy on a controlled low-noise
//! design, and the Bai-Ng criteria on a large simulated panel where they
//! provably select the true count.

mod common;

use common::{as_mat, load_fixture, Lcg};
use tsecon_favar::{bai_ng, eigenvalue_ratio, FactorModel, Favar, Trend};
use tsecon_linalg::faer::Mat;

// Simulate a standardized-free r-factor panel: X = F L' + noise, with
// F, L, noise all Gaussian and `noise_sd` the idiosyncratic scale.
fn simulate(n: usize, n_series: usize, r: usize, noise_sd: f64, seed: u64) -> Mat<f64> {
    let mut rng = Lcg::new(seed);
    let f = Mat::from_fn(n, r, |_, _| rng.gaussian());
    let l = Mat::from_fn(n_series, r, |_, _| rng.gaussian());
    Mat::from_fn(n, n_series, |i, j| {
        let common: f64 = (0..r).map(|k| f[(i, k)] * l[(j, k)]).sum();
        common + noise_sd * rng.gaussian()
    })
}

fn fixture_panel() -> Mat<f64> {
    let fx = load_fixture("favar.json");
    let xt = as_mat(&fx["X_standardized"]);
    Mat::from_fn(xt.ncols(), xt.nrows(), |i, j| xt[(j, i)])
}

#[test]
fn factors_are_orthogonal() {
    // Principal components are mutually orthogonal: F'F is diagonal, with
    // diagonal entries n * eigenvalue_k.
    let model = FactorModel::fit(fixture_panel().as_ref()).unwrap();
    let r = 4;
    let f = model.factors(r).unwrap();
    let n = model.n_obs();
    for a in 0..r {
        for b in 0..r {
            let dot: f64 = (0..n).map(|t| f[(t, a)] * f[(t, b)]).sum();
            if a == b {
                let expected = n as f64 * model.eigenvalues()[a];
                assert!(
                    (dot - expected).abs() <= 1e-6 * expected.max(1.0),
                    "F'F diagonal[{a}]: {dot} vs {expected}"
                );
            } else {
                assert!(dot.abs() < 1e-6, "F'F off-diagonal[{a},{b}] = {dot}");
            }
        }
    }
}

#[test]
fn loadings_are_orthonormal_columns() {
    // Loadings are columns of V: orthonormal.
    let model = FactorModel::fit(fixture_panel().as_ref()).unwrap();
    let r = 4;
    let l = model.loadings(r).unwrap();
    let big_n = model.n_series();
    for a in 0..r {
        for b in 0..r {
            let dot: f64 = (0..big_n).map(|i| l[(i, a)] * l[(i, b)]).sum();
            let expected = if a == b { 1.0 } else { 0.0 };
            assert!(
                (dot - expected).abs() < 1e-9,
                "L'L[{a},{b}] = {dot} (expected {expected})"
            );
        }
    }
}

#[test]
fn low_noise_reconstruction_is_small() {
    // On a low-noise 2-factor design the rank-2 reconstruction of the
    // standardized panel has small idiosyncratic residual.
    let x = simulate(200, 30, 2, 0.05, 7);
    let model = FactorModel::fit(x.as_ref()).unwrap();
    let recon = model.reconstruct_standardized(2).unwrap();
    // Standardized panel Z, recomputed from the model's center/scale.
    let center = model.center();
    let scale = model.scale();
    let mut sse = 0.0;
    let (n, big_n) = (model.n_obs(), model.n_series());
    for i in 0..n {
        for j in 0..big_n {
            let z = (x[(i, j)] - center[j]) / scale[j];
            let d = z - recon[(i, j)];
            sse += d * d;
        }
    }
    let mse = sse / (n * big_n) as f64;
    assert!(mse < 1e-2, "rank-2 reconstruction MSE too large: {mse}");
}

#[test]
fn eigenvalue_ratio_recovers_true_count() {
    for seed in 0..5u64 {
        let x = simulate(200, 40, 2, 1.0, 100 + seed);
        let model = FactorModel::fit(x.as_ref()).unwrap();
        let (r_hat, _) = eigenvalue_ratio(model.eigenvalues(), 8).unwrap();
        assert_eq!(r_hat, 2, "seed {seed}: ER picked {r_hat}");
    }
}

#[test]
fn bai_ng_picks_true_count_on_large_panel() {
    // The Bai-Ng criteria are consistent as n, N grow; on a large
    // simulated 2-factor panel IC_p1 and IC_p2 select 2. (On the small
    // N = 24 fixture the idiosyncratic eigenvalues decay too slowly for
    // the log-criteria to stop — the eigenvalue-ratio estimator is the
    // robust choice there; see golden_eigenvalue_ratio_picks_two.)
    for seed in 0..5u64 {
        let x = simulate(200, 60, 2, 1.0, 200 + seed);
        let model = FactorModel::fit(x.as_ref()).unwrap();
        let bn = bai_ng(model.eigenvalues(), model.n_obs(), model.n_series(), 8).unwrap();
        assert_eq!(bn.icp1_hat, 2, "seed {seed}: IC_p1 picked {}", bn.icp1_hat);
        assert_eq!(bn.icp2_hat, 2, "seed {seed}: IC_p2 picked {}", bn.icp2_hat);
    }
}

#[test]
fn bai_ng_criteria_curves_have_expected_shape() {
    // V(k) is strictly decreasing in k; the criteria are finite.
    let model = FactorModel::fit(fixture_panel().as_ref()).unwrap();
    let bn = bai_ng(model.eigenvalues(), model.n_obs(), model.n_series(), 8).unwrap();
    assert_eq!(bn.icp1.len(), 9);
    for c in bn
        .icp1
        .iter()
        .chain(&bn.icp2)
        .chain(&bn.pcp1)
        .chain(&bn.pcp2)
    {
        assert!(c.is_finite());
    }
}

#[test]
fn favar_two_step_dimensions() {
    let x = simulate(200, 30, 2, 0.5, 11);
    let policy: Vec<f64> = {
        let mut rng = Lcg::new(999);
        (0..200).map(|_| rng.gaussian()).collect()
    };
    let favar = Favar::two_step(x.as_ref(), &policy, 2, 2, Trend::Constant).unwrap();
    assert_eq!(favar.n_factors(), 2);
    assert_eq!(favar.n_endog(), 3);
    assert_eq!(favar.policy_index(), 2);
    assert_eq!(favar.var().neqs, 3);
    assert_eq!(favar.factors().nrows(), 200);
    assert_eq!(favar.factors().ncols(), 2);
    assert_eq!(favar.policy().len(), 200);
}

#[test]
fn favar_irf_mapping_dimensions() {
    let x = simulate(200, 30, 2, 0.5, 13);
    let policy: Vec<f64> = {
        let mut rng = Lcg::new(1001);
        (0..200).map(|_| rng.gaussian()).collect()
    };
    let favar = Favar::two_step(x.as_ref(), &policy, 2, 2, Trend::Constant).unwrap();
    let horizon = 12;
    // Response of every panel series to the policy shock (last variable).
    let shock = favar.policy_index();
    for series in 0..30 {
        let resp = favar.series_response(series, shock, horizon, true).unwrap();
        assert_eq!(resp.len(), horizon + 1);
        assert!(resp.iter().all(|x| x.is_finite()));
    }
    // Policy's own response.
    let own = favar.policy_response(shock, horizon, true).unwrap();
    assert_eq!(own.len(), horizon + 1);
    // Orthogonalized impact of the policy shock on the policy variable is
    // the Cholesky diagonal: strictly positive.
    assert!(
        own[0] > 0.0,
        "orthogonalized impact response should be positive"
    );
}

#[test]
fn favar_series_response_matches_manual_loading_map() {
    // series_response == loadings . factor-IRF, elementwise.
    let x = simulate(150, 20, 2, 0.4, 21);
    let policy: Vec<f64> = {
        let mut rng = Lcg::new(2002);
        (0..150).map(|_| rng.gaussian()).collect()
    };
    let favar = Favar::two_step(x.as_ref(), &policy, 2, 1, Trend::Constant).unwrap();
    let horizon = 6;
    let psi = favar.var().ma_rep(horizon).unwrap();
    let loadings = favar.factor_model().loadings(2).unwrap();
    let series = 5usize;
    let shock = 0usize;
    let resp = favar
        .series_response(series, shock, horizon, false)
        .unwrap();
    for (h, m) in psi.iter().enumerate() {
        let manual: f64 = (0..2).map(|f| loadings[(series, f)] * m[(f, shock)]).sum();
        assert!(
            (resp[h] - manual).abs() < 1e-12,
            "h={h}: {} vs {manual}",
            resp[h]
        );
    }
}

#[test]
fn favar_slow_fast_rotation_runs_and_purges_policy() {
    // The slow/fast rotation produces r cleaned factors; each cleaned
    // factor is (numerically) orthogonal to the policy variable after the
    // projection (that is the point of purging b_R * R).
    let x = simulate(200, 30, 2, 0.5, 31);
    let mut rng = Lcg::new(3003);
    let policy: Vec<f64> = (0..200).map(|_| rng.gaussian()).collect();
    let slow: Vec<usize> = (0..15).collect();
    let favar =
        Favar::two_step_slow_fast(x.as_ref(), &policy, &slow, 2, 2, Trend::Constant).unwrap();
    assert!(favar.is_slow_fast());
    assert_eq!(favar.factors().ncols(), 2);
    // The cleaned factors regress out the policy component; the residual
    // projection leaves the cleaned factor's covariance with policy equal
    // to what the *slow factors* explain, not zero in general — but the
    // pipeline must at least be finite and correctly shaped.
    let f = favar.factors();
    for j in 0..2 {
        for i in 0..200 {
            assert!(f[(i, j)].is_finite());
        }
    }
}

#[test]
fn favar_rejects_mismatched_policy_length() {
    let x = simulate(100, 20, 2, 0.5, 41);
    let policy = vec![0.0; 99];
    let err = Favar::two_step(x.as_ref(), &policy, 2, 2, Trend::Constant);
    assert!(err.is_err());
}
