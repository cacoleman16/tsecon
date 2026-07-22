//! Golden validation of the heteroskedasticity SVAR core
//! ([`hetero_decompose`], [`box_m_test`]) against an independent numpy/scipy
//! reference.
//!
//! `fixtures/hetero_svar.json` is produced by
//! `fixtures/generate_hetero_svar_fixtures.py`, which never imports tsecon.
//! It simulates a two-regime VAR(1) with a known impact matrix and distinct
//! structural-shock variance ratios, fits the reduced form by OLS, and stores
//! the two within-regime residual covariances together with the reference
//! decomposition (`scipy.linalg.eigh` on the generalized pencil), the
//! structural IRF, and the Bartlett-corrected Box's M test. The checks here
//! are:
//!
//! * the Cholesky-whitening decomposition reproduces the scipy generalized-eig
//!   reference for `B`, the variance ratios, and the identification margins;
//! * `Psi_h B` reproduces the stored structural IRF;
//! * Box's M reproduces the reference statistic/dof/p-value;
//! * (secondary, weaker) the recovered `B`/ratios are consistent with the true
//!   DGP at this seed and sample size.

use serde_json::Value;
use tsecon_ident::{box_m_test, hetero_decompose, SignConvention};
use tsecon_linalg::faer::Mat;

/// Golden tolerance for the deterministic linear algebra (OLS residual
/// covariances -> Cholesky-whitening EVD -> Box's M). scipy.linalg.eigh and
/// the Rust route both call LAPACK, so this is a tight cross-check.
const TOL: f64 = 1e-8;
/// Slightly looser bound for the companion-powered IRF at the longest horizon,
/// where the reference `B` and the recovered `B` differ by up to `TOL`.
const TOL_IRF: f64 = 1e-7;
/// Finite-sample Monte-Carlo consistency bound (fixed seed, T = 8000): the
/// estimator recovers the true DGP, but this is a statistical property, not a
/// tight golden.
const TOL_MC: f64 = 5e-2;

fn load() -> Value {
    let path = format!(
        "{}/../../fixtures/hetero_svar.json",
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
    Mat::from_fn(r.len(), r[0].len(), |i, j| r[i][j])
}

fn close_at(actual: f64, expected: f64, tol: f64, what: &str) {
    let err = (actual - expected).abs();
    let rel = err / (1.0 + expected.abs());
    assert!(
        err < tol || rel < tol,
        "{what}: actual={actual:.15e} expected={expected:.15e} abs_err={err:.3e} rel={rel:.3e}"
    );
}

fn close_slice(actual: &[f64], expected: &[f64], tol: f64, what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what} length");
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        close_at(*a, *e, tol, &format!("{what}[{i}]"));
    }
}

fn close_mat(actual: &Mat<f64>, expected: &[Vec<f64>], tol: f64, what: &str) {
    assert_eq!(actual.nrows(), expected.len(), "{what} rows");
    for i in 0..actual.nrows() {
        for j in 0..actual.ncols() {
            close_at(
                actual[(i, j)],
                expected[i][j],
                tol,
                &format!("{what}[{i}][{j}]"),
            );
        }
    }
}

#[test]
fn decomposition_matches_scipy_generalized_eig() {
    let fx = load();
    let sigma1 = mat(&fx["sigma1"]);
    let sigma2 = mat(&fx["sigma2"]);

    let d = hetero_decompose(sigma1.as_ref(), sigma2.as_ref(), SignConvention::MaxAbs)
        .expect("decompose");

    // Impact matrix B (canonicalized) and the variance ratios (ascending).
    close_mat(&d.b, &rows(&fx["B"]), TOL, "B");
    close_slice(
        &d.lambda,
        &f64s(&fx["variance_ratios"]),
        TOL,
        "variance_ratios",
    );

    // Identification margins.
    close_at(
        d.min_ratio_gap,
        fx["min_ratio_gap"].as_f64().expect("f64"),
        TOL,
        "min_ratio_gap",
    );
    close_slice(
        &d.ratio_dist_from_unity,
        &f64s(&fx["ratio_dist_from_unity"]),
        TOL,
        "ratio_dist_from_unity",
    );
}

#[test]
fn structural_irf_matches_reference() {
    let fx = load();
    let sigma1 = mat(&fx["sigma1"]);
    let sigma2 = mat(&fx["sigma2"]);
    let d = hetero_decompose(sigma1.as_ref(), sigma2.as_ref(), SignConvention::MaxAbs)
        .expect("decompose");

    // Theta_h = Psi_h @ B, with Psi_h the stored reduced-form MA weights.
    let psi = fx["psi"].as_array().expect("array");
    let theta_ref = fx["structural_irf"].as_array().expect("array");
    assert_eq!(psi.len(), theta_ref.len());
    for (h, (psi_h, theta_h)) in psi.iter().zip(theta_ref.iter()).enumerate() {
        let p = mat(psi_h);
        let theta = &p * &d.b;
        close_mat(
            &theta,
            &rows(theta_h),
            TOL_IRF,
            &format!("structural_irf[{h}]"),
        );
    }
}

#[test]
fn box_m_matches_reference() {
    let fx = load();
    let s1 = mat(&fx["s1_boxm"]);
    let s2 = mat(&fx["s2_boxm"]);
    let n1 = u(&fx["n1"]);
    let n2 = u(&fx["n2"]);

    let r = box_m_test(&[(s1.as_ref(), n1), (s2.as_ref(), n2)]).expect("box m");

    let bm = &fx["box_m"];
    close_at(
        r.statistic,
        bm["statistic"].as_f64().expect("f64"),
        TOL,
        "box_m.statistic",
    );
    assert_eq!(r.dof, u(&bm["dof"]), "box_m.dof");
    close_at(
        r.pvalue,
        bm["pvalue"].as_f64().expect("f64"),
        TOL,
        "box_m.pvalue",
    );
    // Distinct-regimes verdict (p < 0.05) is the necessary-condition flag.
    assert_eq!(
        r.pvalue < 0.05,
        bm["distinct_regimes"].as_bool().expect("bool"),
        "box_m.distinct_regimes"
    );
}

#[test]
fn recovers_the_true_dgp_at_this_seed() {
    // Secondary consistency check: the estimator is not just internally
    // self-consistent but recovers the true B/Lambda in large samples.
    let fx = load();
    let sigma1 = mat(&fx["sigma1"]);
    let sigma2 = mat(&fx["sigma2"]);
    let d = hetero_decompose(sigma1.as_ref(), sigma2.as_ref(), SignConvention::MaxAbs)
        .expect("decompose");

    close_slice(
        &d.lambda,
        &f64s(&fx["mc_variance_ratios_true"]),
        TOL_MC,
        "mc variance ratios",
    );
    close_mat(&d.b, &rows(&fx["mc_b_true_canonical"]), TOL_MC, "mc B");
}
