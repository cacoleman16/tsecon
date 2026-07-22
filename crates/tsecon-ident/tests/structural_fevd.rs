//! Golden and invariant tests for structural FEVD (`structural_fevd`).
//!
//! `fixtures/structural_fevd.json` is produced by
//! `fixtures/generate_structural_fevd_fixtures.py`. The Cholesky-case expected
//! shares come from statsmodels `VARResults.fevd` — a fully INDEPENDENT
//! implementation — so reproducing them in this crate is a genuine
//! cross-implementation check, not a circular one. The general-`A0 = P Q` case
//! (which `tsecon_var::fevd` cannot produce and for which no external reference
//! exists) is an independent NumPy running-sum, additionally pinned by the exact
//! algebraic invariants FEVD must satisfy for any admissible `A0`.
//!
//! * **`core_matches_statsmodels_cholesky`** feeds the stored orthogonalized MA
//!   straight to [`structural_fevd_from_theta`] and bit-matches the statsmodels
//!   FEVD to ~1e-10 — the strong, non-circular validation of the Cholesky case.
//! * **`pipeline_cholesky_from_reduced_form`** rebuilds the MA from the stored
//!   reduced form via `cholesky_irf` (the exact call the `impact=None` binding
//!   makes) and matches at 1e-8.
//! * **`general_impact_matches_numpy`** validates the novel general-`A0` path
//!   both from the stored `Theta` and rebuilt through the shared general-impact
//!   MA helper (`structural_fevd(b, A0, ..)`).
//! * invariants — rows sum to one, rotation-invariant denominator, column
//!   sign-flip invariance — plus the bad-input guards round it out.

use serde_json::Value;
use tsecon_bayes::cholesky_irf;
use tsecon_ident::structural_fevd::{structural_fevd, structural_fevd_from_theta};
use tsecon_linalg::faer::Mat;

const TOL_CORE: f64 = 1e-10; // observed agreement is < 1e-13; margin for platforms
const TOL_PIPE: f64 = 1e-8; // reduced-form-fed: adds one Cholesky round-trip

fn load() -> Value {
    let path = format!(
        "{}/../../fixtures/structural_fevd.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let text = std::fs::read_to_string(path).expect("fixture readable");
    serde_json::from_str(&text).expect("valid JSON")
}

fn f64s(v: &Value) -> Vec<f64> {
    v.as_array()
        .expect("array")
        .iter()
        .map(|x| x.as_f64().expect("number"))
        .collect()
}

fn rows(v: &Value) -> Vec<Vec<f64>> {
    v.as_array().expect("array").iter().map(f64s).collect()
}

fn u(v: &Value) -> usize {
    v.as_u64().expect("uint") as usize
}

fn mat(v: &Value) -> Mat<f64> {
    let r = rows(v);
    let nr = r.len();
    let nc = r[0].len();
    Mat::from_fn(nr, nc, |i, j| r[i][j])
}

fn mats(v: &Value) -> Vec<Mat<f64>> {
    v.as_array().expect("array").iter().map(mat).collect()
}

fn close(actual: f64, expected: f64, tol: f64, what: &str) {
    let err = (actual - expected).abs();
    let rel = err / (1.0 + expected.abs());
    assert!(
        err < tol || rel < tol,
        "{what}: actual={actual:.15e} expected={expected:.15e} abs_err={err:.3e} rel={rel:.3e}"
    );
}

/// Cell-by-cell comparison of a `[h][i][j]` FEVD against the fixture block.
fn check_fevd(actual: &[Mat<f64>], expected: &Value, n: usize, tol: f64, what: &str) {
    let exp = expected.as_array().expect("fevd array");
    assert_eq!(actual.len(), exp.len(), "{what}: horizon count");
    for (h, (got, want)) in actual.iter().zip(exp.iter()).enumerate() {
        let want_rows = rows(want);
        for i in 0..n {
            for j in 0..n {
                close(
                    got[(i, j)],
                    want_rows[i][j],
                    tol,
                    &format!("{what} [h={h}][{i}][{j}]"),
                );
            }
        }
    }
}

#[test]
fn core_matches_statsmodels_cholesky() {
    // theta from NumPy, expected shares from statsmodels VARResults.fevd:
    // both independent of tsecon. The strong Cholesky-case golden.
    let fx = load();
    let n = u(&fx["n"]);
    let theta = mats(&fx["theta_chol"]);
    let fevd = structural_fevd_from_theta(&theta).expect("core FEVD");
    check_fevd(
        &fevd,
        &fx["fevd_statsmodels"],
        n,
        TOL_CORE,
        "chol vs statsmodels",
    );
    // and against the stored NumPy running-sum (identical to statsmodels).
    check_fevd(&fevd, &fx["fevd_chol"], n, TOL_CORE, "chol vs numpy");
}

#[test]
fn pipeline_cholesky_from_reduced_form() {
    // Rebuild the orthogonalized MA from the stored reduced form exactly as the
    // impact=None binding will: cholesky_irf(reg_coefs, sigma, lags, horizon).
    let fx = load();
    let n = u(&fx["n"]);
    let lags = u(&fx["lags"]);
    let horizon = u(&fx["horizon"]);
    let b = mat(&fx["reg_coefs"]);
    let sigma = mat(&fx["sigma"]);
    let theta = cholesky_irf(b.as_ref(), sigma.as_ref(), lags, horizon).expect("cholesky_irf");
    let fevd = structural_fevd_from_theta(&theta).expect("core FEVD");
    check_fevd(&fevd, &fx["fevd_chol"], n, TOL_PIPE, "pipeline chol");
}

#[test]
fn general_impact_matches_numpy() {
    // The novel path: FEVD for a non-triangular A0 = P Q.
    let fx = load();
    let n = u(&fx["n"]);
    let lags = u(&fx["lags"]);
    let horizon = u(&fx["horizon"]);

    // (a) directly from the stored general-impact MA.
    let theta = mats(&fx["theta_general"]);
    let fevd = structural_fevd_from_theta(&theta).expect("core FEVD");
    check_fevd(&fevd, &fx["fevd_general"], n, TOL_CORE, "general core");

    // (b) rebuilt through the shared general-impact MA helper (structural_ma),
    // the exact call the impact=<A0> binding makes.
    let b = mat(&fx["reg_coefs"]);
    let a0 = mat(&fx["impact_general"]);
    let fevd_pipe = structural_fevd(b.as_ref(), a0.as_ref(), lags, horizon).expect("pipeline FEVD");
    check_fevd(
        &fevd_pipe,
        &fx["fevd_general"],
        n,
        TOL_PIPE,
        "general pipeline",
    );
}

#[test]
fn rows_sum_to_one_both_cases() {
    let fx = load();
    let n = u(&fx["n"]);
    for key in ["theta_chol", "theta_general"] {
        let theta = mats(&fx[key]);
        let fevd = structural_fevd_from_theta(&theta).expect("FEVD");
        for (h, m) in fevd.iter().enumerate() {
            for i in 0..n {
                let row: f64 = (0..n).map(|j| m[(i, j)]).sum();
                assert!(
                    (row - 1.0).abs() < 1e-12,
                    "{key} row ({i}) at horizon {h} sums to {row}"
                );
            }
        }
    }
}

#[test]
fn denominator_is_rotation_invariant() {
    // The (h+1)-step forecast MSE diagonal (the FEVD denominator) is invariant
    // to the rotation Q in A0 = P Q, because A0 A0' = Sigma regardless of Q.
    let fx = load();
    let n = u(&fx["n"]);
    let horizon = u(&fx["horizon"]);
    let theta_c = mats(&fx["theta_chol"]);
    let theta_g = mats(&fx["theta_general"]);
    let stored = rows(&fx["mse_diag_chol"]); // [h][i]

    let mse = |theta: &[Mat<f64>]| -> Vec<Vec<f64>> {
        let mut cum = vec![vec![0.0f64; n]; n];
        let mut out = Vec::new();
        for th in theta {
            for i in 0..n {
                for j in 0..n {
                    cum[i][j] += th[(i, j)] * th[(i, j)];
                }
            }
            out.push((0..n).map(|i| cum[i].iter().sum()).collect::<Vec<f64>>());
        }
        out
    };
    let dc = mse(&theta_c);
    let dg = mse(&theta_g);
    for h in 0..=horizon {
        for i in 0..n {
            // chol vs general (the invariance), and both vs the stored NumPy.
            close(dc[h][i], dg[h][i], 1e-9, &format!("rot-inv [h={h}][{i}]"));
            close(
                dc[h][i],
                stored[h][i],
                TOL_PIPE,
                &format!("stored mse [h={h}][{i}]"),
            );
        }
    }
}

#[test]
fn column_sign_flip_is_invariant() {
    // Negating columns of A0 (a sign-normalization choice) leaves the shares
    // unchanged, since responses enter the decomposition squared.
    let fx = load();
    let n = u(&fx["n"]);
    let lags = u(&fx["lags"]);
    let horizon = u(&fx["horizon"]);
    let b = mat(&fx["reg_coefs"]);
    let a0 = mat(&fx["impact_general"]);
    let base = structural_fevd(b.as_ref(), a0.as_ref(), lags, horizon).expect("base");
    // Flip a subset of columns.
    let flip = [-1.0, 1.0, -1.0];
    let a0f = Mat::from_fn(n, n, |i, j| a0[(i, j)] * flip[j % flip.len()]);
    let flipped = structural_fevd(b.as_ref(), a0f.as_ref(), lags, horizon).expect("flipped");
    for (x, y) in base.iter().zip(flipped.iter()) {
        for i in 0..n {
            for j in 0..n {
                assert!((x[(i, j)] - y[(i, j)]).abs() < 1e-12);
            }
        }
    }
}

#[test]
fn rejects_bad_input() {
    let empty: Vec<Mat<f64>> = Vec::new();
    assert!(structural_fevd_from_theta(&empty).is_err());
    let nonsquare = vec![Mat::<f64>::zeros(3, 2)];
    assert!(structural_fevd_from_theta(&nonsquare).is_err());
    // p = 0 is rejected by the general-impact MA construction.
    let b = Mat::<f64>::zeros(4, 3);
    let a0 = Mat::<f64>::identity(3, 3);
    assert!(structural_fevd(b.as_ref(), a0.as_ref(), 0, 4).is_err());
}
