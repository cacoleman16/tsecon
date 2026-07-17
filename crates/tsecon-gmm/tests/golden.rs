//! Golden-value tests against `fixtures/gmm.json` (linearmodels 7.0).
//!
//! The fixture DGP is a linear IV regression of `y` on `[const, w, x]` where
//! `x` is endogenous and `[const, w]` are exogenous; the instrument set is
//! `[z1, z2]` plus the included exogenous regressors, so the full instrument
//! matrix is `Z = [const, w, z1, z2]` (4 columns) against a `k = 3` regressor
//! matrix `X = [const, w, x]` — one over-identifying restriction.
//!
//! The golden numbers come from `linearmodels.iv.IVGMM(...).fit()` with the
//! default `weight_type="robust"` and `cov_type="robust"` (2-step efficient
//! GMM, heteroskedasticity-robust weighting and robust sandwich covariance).
//! We reproduce, via [`tsecon_gmm::two_step_gmm`] with
//! [`tsecon_gmm::GmmWeight::Robust`]:
//!
//! * `params` (const, w, x) to `1e-9` (fixture printed at full precision;
//!   the point estimate matches to machine precision);
//! * `bse` (const, w, x) to `1e-6` — the full GMM sandwich covariance with
//!   the step-2 estimation weight `W = S(u1)^{-1}` and the moment covariance
//!   `S` recomputed at the step-2 residuals (the collapsed efficient form
//!   only reaches ~5e-5);
//! * `j_stat` to `1e-6` and `j_pval` to `1e-6` — the Hansen J uses the
//!   step-2 estimation weight evaluated at the step-2 residuals.

use serde_json::Value;
use tsecon_gmm::{two_step_gmm, GmmWeight};

fn load() -> Value {
    let path = format!("{}/../../fixtures/gmm.json", env!("CARGO_MANIFEST_DIR"));
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

fn assert_close(actual: f64, expected: f64, atol: f64, ctx: &str) {
    let err = (actual - expected).abs();
    assert!(
        err <= atol,
        "{ctx}: actual {actual}, expected {expected}, abs err {err:e} > tol {atol:e}"
    );
}

#[test]
fn ivgmm_two_step_robust_matches_linearmodels() {
    let fx = load();
    let y = f64s(&fx["y"]);
    let x = f64s(&fx["x"]);
    let w = f64s(&fx["w"]);
    let z1 = f64s(&fx["z1"]);
    let z2 = f64s(&fx["z2"]);
    let n = y.len();
    let cst = vec![1.0_f64; n];

    // X = [const, w, x] (x endogenous); Z = [const, w, z1, z2].
    let x_cols = vec![cst.clone(), w.clone(), x];
    let z_cols = vec![cst, w, z1, z2];

    let fit = two_step_gmm(&x_cols, &z_cols, &y, GmmWeight::Robust).expect("two-step GMM fits");

    // Parameter order in the fixture is const, w, x.
    let gp = &fx["ivgmm"]["params"];
    let gb = &fx["ivgmm"]["bse"];
    let names = ["const", "w", "x"];
    for (i, name) in names.iter().enumerate() {
        assert_close(
            fit.params[i],
            gp[*name].as_f64().unwrap(),
            1e-9,
            &format!("param[{name}]"),
        );
        assert_close(
            fit.bse[i],
            gb[*name].as_f64().unwrap(),
            1e-6,
            &format!("bse[{name}]"),
        );
    }

    // Hansen J-test: over-identified with dof = 4 - 3 = 1.
    let jtest = fit
        .jtest
        .expect("over-identified model has a Hansen J-test");
    assert_eq!(jtest.dof, 1);
    assert_close(
        jtest.stat,
        fx["ivgmm"]["j_stat"].as_f64().unwrap(),
        1e-6,
        "j_stat",
    );
    assert_close(
        jtest.pval,
        fx["ivgmm"]["j_pval"].as_f64().unwrap(),
        1e-6,
        "j_pval",
    );

    assert_eq!(fit.steps, 2);
    assert_eq!(fit.nobs, n);
    assert_eq!(fit.nmoments, 4);
    assert_eq!(fit.nparams, 3);
}
