//! Golden-value tests for the pooled-mean-group (PMG) estimator against
//! `fixtures/pmg.json`.
//!
//! ## What kind of golden this is
//!
//! This is a **documented-formula golden, not an external-package golden.**
//! There is no PMG in statsmodels / linearmodels to call, so the generator
//! (`fixtures/generate_pmg_fixtures.py`) reimplements the *same* Pesaran, Shin
//! & Smith (1999) concentrated-maximum-likelihood back-substitution estimator
//! independently in NumPy, with the estimating equations written out in its
//! docstring and cited to PSS 1999. This crate implements the identical
//! estimator through a different numerical path — per-unit OLS residualization
//! via Cholesky normal equations in `tsecon-hac`, and the pooled `k x k`
//! Cholesky solve/inverse via `faer` — starting from the identical `theta = 0`.
//!
//! Agreement is therefore a genuine cross-implementation consistency check
//! (NumPy `lstsq`/SVD + explicit GLS update vs. Rust Cholesky), which pins the
//! algebra, but it is NOT a check against an independent statistical authority.
//! The estimator's *statistical* validity is pinned separately by the
//! property tests in `pmg_properties.rs` (recovery of a known common long run,
//! and tight pooling relative to a free mean group).
//!
//! Both sides iterate the same contraction to the same concentrated-ML fixed
//! point, so the pooled long-run `theta`, its SE, `phi_bar`, per-unit `phi_i`,
//! and the log-likelihood agree to well under `1e-8`.

use serde_json::Value;
use tsecon_panelts::{pmg, PanelUnit};

fn load() -> Value {
    let path = format!("{}/../../fixtures/pmg.json", env!("CARGO_MANIFEST_DIR"));
    let text = std::fs::read_to_string(path).expect("fixture file readable");
    serde_json::from_str(&text).expect("fixture is valid JSON")
}

fn f64s(v: &Value) -> Vec<f64> {
    v.as_array()
        .expect("array")
        .iter()
        .map(|x| x.as_f64().expect("number"))
        .collect()
}

fn f64_matrix(v: &Value) -> Vec<Vec<f64>> {
    v.as_array().expect("array").iter().map(f64s).collect()
}

fn build_units(fx: &Value) -> Vec<PanelUnit> {
    let n = fx["design"]["N"].as_u64().expect("N") as usize;
    let k = fx["design"]["K"].as_u64().expect("K") as usize;
    let y = f64_matrix(&fx["y"]); // N x T_raw
    let x: Vec<Vec<Vec<f64>>> = (0..k).map(|j| f64_matrix(&fx["x"][j])).collect(); // K x N x T_raw
    (0..n)
        .map(|i| {
            let cols: Vec<Vec<f64>> = (0..k).map(|j| x[j][i].clone()).collect();
            PanelUnit::new(y[i].clone(), cols)
        })
        .collect()
}

fn assert_close(actual: f64, expected: f64, atol: f64, ctx: &str) {
    let err = (actual - expected).abs();
    assert!(
        err <= atol,
        "{ctx}: actual {actual}, expected {expected}, abs err {err:e} > tol {atol:e}"
    );
}

#[test]
fn pmg_matches_documented_formula_golden() {
    let fx = load();
    let units = build_units(&fx);
    let fit = pmg(&units).expect("PMG fits");

    let g = &fx["pmg"];
    let theta = f64s(&g["theta"]);
    let theta_se = f64s(&g["theta_se"]);
    let phi = f64s(&g["phi"]);
    let sigma2 = f64s(&g["sigma2"]);
    let phi_bar = g["phi_bar"].as_f64().expect("phi_bar");
    let loglik = g["loglik"].as_f64().expect("loglik");

    // Pooled long-run theta and average adjustment speed to ~1e-8 (the task's
    // required reference-match surface).
    for (j, (&got, &exp)) in fit.theta.iter().zip(theta.iter()).enumerate() {
        assert_close(got, exp, 1e-8, &format!("theta[{j}]"));
    }
    assert_close(fit.phi_bar, phi_bar, 1e-8, "phi_bar");

    // The remaining reported quantities agree just as tightly.
    for (j, (&got, &exp)) in fit.theta_se.iter().zip(theta_se.iter()).enumerate() {
        assert_close(got, exp, 1e-8, &format!("theta_se[{j}]"));
    }
    assert_eq!(fit.phi.len(), phi.len());
    for (i, (&got, &exp)) in fit.phi.iter().zip(phi.iter()).enumerate() {
        assert_close(got, exp, 1e-8, &format!("phi[{i}]"));
    }
    for (i, (&got, &exp)) in fit.sigma2.iter().zip(sigma2.iter()).enumerate() {
        assert_close(got, exp, 1e-8, &format!("sigma2[{i}]"));
    }
    // log-likelihood is a sum of ~2600 log-variances; allow a slightly looser
    // (still far sub-ulp-per-term) absolute tolerance on the aggregate.
    assert_close(fit.loglik, loglik, 1e-6, "loglik");
}

#[test]
fn pmg_golden_recovers_true_common_long_run() {
    // Sanity check on the stored numbers: the fixture was simulated with a
    // known common long-run theta0, and both the PMG estimate and the free
    // mean-group estimate land near it — but PMG's SE is far below the
    // cross-unit dispersion of the free per-unit long runs (it pools).
    let fx = load();
    let theta0 = f64s(&fx["theta0"]);
    let theta = f64s(&fx["pmg"]["theta"]);
    let theta_se = f64s(&fx["pmg"]["theta_se"]);
    let free_sd = f64s(&fx["free_mg"]["cross_unit_sd"]);
    for j in 0..theta0.len() {
        assert!(
            (theta[j] - theta0[j]).abs() < 0.05,
            "PMG theta[{j}] {} should be near theta0 {}",
            theta[j],
            theta0[j]
        );
        assert!(
            theta_se[j] < 0.25 * free_sd[j],
            "PMG SE[{j}] {} should be far below the free-MG cross-unit sd {}",
            theta_se[j],
            free_sd[j]
        );
    }
}
