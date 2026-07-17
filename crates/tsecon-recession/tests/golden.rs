//! Independent-reference golden tests for the STATIC probit and logit
//! (validation target (a)).
//!
//! `fixtures/tsecon-recession.json` is produced by
//! `fixtures/generate_tsecon-recession_fixtures.py`, which fits statsmodels'
//! `sm.Probit` / `sm.Logit` on a fixed simulated recession dataset. statsmodels
//! reaches the exact-likelihood MLE, its analytic-Hessian standard errors, the
//! log-likelihood, McFadden's pseudo-R^2, and the fitted probability path by an
//! entirely separate code path, so reproducing its numbers to ~1e-6 is a
//! genuine cross-implementation check — not circular.
//!
//! The DYNAMIC probit has no statsmodels reference and is validated
//! property-only in `properties.rs`.

use serde_json::Value;
use tsecon_recession::{fit_static, Link};

fn load() -> Value {
    let path = format!(
        "{}/../../fixtures/tsecon-recession.json",
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

fn g(v: &Value) -> f64 {
    v.as_f64().expect("number")
}

/// Reference match tolerance: statsmodels vs the Rust MLE, ~1e-6.
const TOL: f64 = 1e-6;

fn close(actual: f64, expected: f64, tol: f64, what: &str) {
    let err = (actual - expected).abs();
    let rel = err / (1.0 + expected.abs());
    assert!(
        err < tol || rel < tol,
        "{what}: actual={actual:.15e} expected={expected:.15e} abs_err={err:.3e}"
    );
}

fn design(fx: &Value) -> (Vec<f64>, Vec<Vec<f64>>) {
    let y = f64s(&fx["y"]);
    let c = f64s(&fx["const"]);
    let spread = f64s(&fx["spread"]);
    let lead = f64s(&fx["lead"]);
    (y, vec![c, spread, lead])
}

fn check_block(fit: &tsecon_recession::RecessionFit, block: &Value) {
    let params = f64s(&block["params"]);
    let bse = f64s(&block["bse"]);
    let tvalues = f64s(&block["tvalues"]);
    let fitted = f64s(&block["fitted"]);

    for j in 0..params.len() {
        close(fit.params[j], params[j], TOL, "param");
        close(fit.bse[j], bse[j], TOL, "bse");
        close(fit.zstats[j], tvalues[j], TOL, "zstat");
    }
    close(fit.loglik, g(&block["llf"]), TOL, "llf");
    close(fit.loglik_null, g(&block["llnull"]), TOL, "llnull");
    close(fit.pseudo_r2, g(&block["prsquared"]), TOL, "pseudo_r2");
    assert_eq!(fit.fitted.len(), fitted.len());
    for (&got, &want) in fit.fitted.iter().zip(fitted.iter()) {
        close(got, want, TOL, "fitted_prob");
    }
}

#[test]
fn static_probit_matches_statsmodels() {
    let fx = load();
    let (y, x) = design(&fx);
    let fit = fit_static(&y, &x, Link::Probit).expect("probit fit");
    assert!(fit.converged, "probit optimizer did not converge");
    check_block(&fit, &fx["probit"]);
}

#[test]
fn static_logit_matches_statsmodels() {
    let fx = load();
    let (y, x) = design(&fx);
    let fit = fit_static(&y, &x, Link::Logit).expect("logit fit");
    assert!(fit.converged, "logit optimizer did not converge");
    check_block(&fit, &fx["logit"]);
}

#[test]
fn pseudo_r2_definition_is_consistent() {
    // McFadden's pseudo-R^2 = 1 - llf/llnull, recomputed from the reported
    // parts, must agree with the reported value.
    let fx = load();
    let (y, x) = design(&fx);
    let fit = fit_static(&y, &x, Link::Probit).expect("probit fit");
    let recomputed = 1.0 - fit.loglik / fit.loglik_null;
    close(
        fit.pseudo_r2,
        recomputed,
        1e-12,
        "pseudo_r2 self-consistency",
    );
}
