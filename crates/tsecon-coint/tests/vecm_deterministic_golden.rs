//! Golden and reconciliation tests for the VECM deterministic-terms cases
//! (field item 12) against `fixtures/vecm_deterministic.json`: seeded
//! *drifting* cointegrated data (rank 1) where statsmodels
//! `VECM(deterministic = "n")` and `VECM(deterministic = "co")` pin both
//! supported cases, and `coint_johansen(det_order = 0)` arbitrates that
//! the unrestricted-constant fit spans exactly the cointegrating space the
//! Johansen rank test works in — while the no-deterministic fit visibly
//! does not (the reporter's beta-cosine divergence, pinned as documented
//! behavior).

mod common;

use common::{as_endog, as_mat, as_vec, assert_mat_close, assert_rel_close, load_fixture, num};
use tsecon_coint::tsecon_linalg::faer::Mat;
use tsecon_coint::{fit_vecm, fit_vecm_det, johansen, VecmDeterministic};

fn cosine(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    dot / (na * nb)
}

fn beta_col0(beta: &Mat<f64>) -> Vec<f64> {
    (0..beta.nrows()).map(|i| beta[(i, 0)]).collect()
}

/// `deterministic = "n"`: alpha, beta, gamma, sigma_u, llf match
/// statsmodels `VECM(..., deterministic = "n")` to 1e-6 relative, and
/// `det_coef` is `k x 0`. Also pins that `fit_vecm` (the historical
/// three-argument entry point) is exactly this case.
#[test]
fn golden_vecm_deterministic_none() {
    let fx = load_fixture("vecm_deterministic.json");
    let endog = as_endog(&fx["data"]);
    let vb = &fx["vecm_n"];

    let res = fit_vecm_det(endog.as_ref(), 1, 1, VecmDeterministic::None).unwrap();
    assert_eq!(res.neqs, 3);
    assert_eq!(res.nobs, 398);
    assert_eq!(res.deterministic, VecmDeterministic::None);
    assert_eq!(res.det_coef.ncols(), 0);
    assert_mat_close(&res.alpha, &vb["alpha"], 1e-6, "alpha (n)");
    assert_mat_close(&res.beta, &vb["beta"], 1e-6, "beta (n)");
    assert_mat_close(&res.gamma, &vb["gamma"], 1e-6, "gamma (n)");
    assert_mat_close(&res.sigma_u, &vb["sigma_u"], 1e-6, "sigma_u (n)");
    assert_rel_close(res.llf, num(&vb["llf"]), 1e-6, "llf (n)");

    // The default entry point is bit-identical to the explicit "n" case.
    let default = fit_vecm(endog.as_ref(), 1, 1).unwrap();
    for i in 0..3 {
        assert_eq!(default.beta[(i, 0)], res.beta[(i, 0)], "fit_vecm != n");
    }
    assert_eq!(default.llf, res.llf, "fit_vecm llf != n llf");
}

/// `deterministic = "co"` (unrestricted constant): alpha, beta, gamma,
/// det_coef, sigma_u, llf match statsmodels `VECM(..., deterministic =
/// "co")` to 1e-6 relative.
#[test]
fn golden_vecm_deterministic_constant() {
    let fx = load_fixture("vecm_deterministic.json");
    let endog = as_endog(&fx["data"]);
    let vb = &fx["vecm_co"];

    let res = fit_vecm_det(endog.as_ref(), 1, 1, VecmDeterministic::Constant).unwrap();
    assert_eq!(res.deterministic, VecmDeterministic::Constant);
    assert_mat_close(&res.alpha, &vb["alpha"], 1e-6, "alpha (co)");
    assert_mat_close(&res.beta, &vb["beta"], 1e-6, "beta (co)");
    assert_mat_close(&res.gamma, &vb["gamma"], 1e-6, "gamma (co)");
    assert_mat_close(&res.det_coef, &vb["det_coef"], 1e-6, "det_coef (co)");
    assert_mat_close(&res.sigma_u, &vb["sigma_u"], 1e-6, "sigma_u (co)");
    assert_rel_close(res.llf, num(&vb["llf"]), 1e-6, "llf (co)");
}

/// The reconciliation the field reporter asked for: the
/// unrestricted-constant VECM solves the same reduced-rank eigenproblem
/// as the Johansen rank test (`det_order = 0`), so its eigenvalues equal
/// the test's to 1e-8 and its beta spans the same direction as the
/// normalized first Johansen eigenvector (cosine 1 to 1e-10) — while the
/// no-deterministic default's beta diverges on this drifting draw exactly
/// as the fixture pins (cosine ~0.63; the reporter measured ~0.57 on
/// their data).
#[test]
fn vecm_constant_reconciles_with_johansen() {
    let fx = load_fixture("vecm_deterministic.json");
    let endog = as_endog(&fx["data"]);

    let joh = johansen(endog.as_ref(), 1).unwrap();
    let co = fit_vecm_det(endog.as_ref(), 1, 1, VecmDeterministic::Constant).unwrap();
    let none = fit_vecm(endog.as_ref(), 1, 1).unwrap();

    // Same canonical-correlation eigenvalues.
    for i in 0..3 {
        assert_rel_close(
            co.eig[i],
            joh.eig[i],
            1e-8,
            &format!("eig[{i}] co vs johansen"),
        );
    }
    // Same cointegrating direction: normalize the first Johansen
    // eigenvector as the VECM does (leading entry 1).
    let joh_beta: Vec<f64> = (0..3)
        .map(|i| joh.evec[(i, 0)] / joh.evec[(0, 0)])
        .collect();
    let cos_co = cosine(&beta_col0(&co.beta), &joh_beta);
    assert!(
        cos_co.abs() > 1.0 - 1e-10,
        "cosine(beta_co, beta_johansen) = {cos_co}, expected ~1"
    );

    // And the documented divergence of the "n" default (the defect the
    // field report demonstrated): pinned by the fixture.
    let cos_n = cosine(&beta_col0(&none.beta), &beta_col0(&co.beta));
    let pinned = num(&fx["beta_cosine_n_co"]);
    assert_rel_close(cos_n, pinned, 1e-6, "cosine(beta_n, beta_co)");
    assert!(
        cos_n < 0.8,
        "cosine(beta_n, beta_co) = {cos_n}: the two deterministic cases \
         should visibly diverge on this drifting draw"
    );

    // The Johansen eigenvectors the fixture pins are reproduced too —
    // per column up to sign (both are S_11-orthonormal, but eigensolvers
    // pick column signs arbitrarily).
    let evec_fx = as_mat(&fx["johansen"]["evec"]);
    for j in 0..3 {
        let dot: f64 = (0..3).map(|i| joh.evec[(i, j)] * evec_fx[(i, j)]).sum();
        let sign = if dot < 0.0 { -1.0 } else { 1.0 };
        for i in 0..3 {
            assert_rel_close(
                sign * joh.evec[(i, j)],
                evec_fx[(i, j)],
                1e-6,
                &format!("evec[({i},{j})] (sign-aligned)"),
            );
        }
    }
    // Johansen eigenvalues against the fixture.
    let eig_fx = as_vec(&fx["johansen"]["eig"]);
    for (i, &e) in eig_fx.iter().enumerate() {
        assert_rel_close(joh.eig[i], e, 1e-8, &format!("johansen eig[{i}]"));
    }
}

/// The degenerate short-run case `k_ar_diff = 0` under the unrestricted
/// constant: the regression block is the constant alone; gamma is `k x 0`,
/// det_coef is `k x 1` and finite, and the fit succeeds.
#[test]
fn vecm_constant_k_ar_diff_zero() {
    let fx = load_fixture("vecm_deterministic.json");
    let endog = as_endog(&fx["data"]);
    let res = fit_vecm_det(endog.as_ref(), 0, 1, VecmDeterministic::Constant).unwrap();
    assert_eq!(res.gamma.ncols(), 0);
    assert_eq!(res.det_coef.ncols(), 1);
    for i in 0..3 {
        assert!(res.det_coef[(i, 0)].is_finite());
    }
    assert!(res.llf.is_finite());
}
