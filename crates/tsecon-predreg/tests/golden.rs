//! Documented-formula golden tests (validation target (a)).
//!
//! `fixtures/predreg.json` is produced by
//! `fixtures/generate_predreg_fixtures.py`, which computes every published
//! quantity by literally writing the closed-form formula in NumPy (no call to
//! this crate). Matching it to ~1e-9 proves the Rust reproduces the published
//! algebra of Stambaugh (1999) and Kostakis-Magdalinos-Stamatogiannis (2015).
//! It does not by itself prove the formulas are statistically correct — that
//! is what `properties.rs` establishes by Monte-Carlo.

use serde_json::Value;
use tsecon_predreg::{ivx, ivx_multi, ols_predictive, stambaugh, IvxConfig};

fn load() -> Value {
    let path = format!("{}/../../fixtures/predreg.json", env!("CARGO_MANIFEST_DIR"));
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

const TOL: f64 = 1e-9;

fn close(actual: f64, expected: f64, tol: f64, what: &str) {
    let err = (actual - expected).abs();
    let rel = err / (1.0 + expected.abs());
    assert!(
        err < tol || rel < tol,
        "{what}: actual={actual:.15e} expected={expected:.15e} abs_err={err:.3e}"
    );
}

#[test]
fn scalar_ols_matches_documented_formula() {
    let fx = load();
    let s = &fx["scalar"];
    let r = f64s(&s["r"]);
    let x = f64s(&s["x"]);
    let fit = ols_predictive(&r, &x).expect("ols");
    let o = &s["ols"];
    close(fit.beta, g(&o["beta_ols"]), TOL, "beta_ols");
    close(fit.alpha, g(&o["alpha_ols"]), TOL, "alpha_ols");
    close(fit.se, g(&o["se"]), TOL, "se(beta_ols)");
    close(fit.tstat, g(&o["tstat"]), TOL, "t(beta_ols)");
}

#[test]
fn scalar_stambaugh_matches_documented_formula() {
    let fx = load();
    let s = &fx["scalar"];
    let r = f64s(&s["r"]);
    let x = f64s(&s["x"]);
    let c = stambaugh(&r, &x).expect("stambaugh");
    let e = &s["stambaugh"];
    close(c.beta_ols, g(&s["ols"]["beta_ols"]), TOL, "beta_ols");
    close(c.rho_ols, g(&e["rho_ols"]), TOL, "rho_ols");
    close(c.sigma_ee, g(&e["sigma_ee"]), TOL, "sigma_ee");
    close(c.sigma_ue, g(&e["sigma_ue"]), TOL, "sigma_ue");
    close(c.kendall_bias, g(&e["bias_rho"]), TOL, "kendall_bias");
    close(c.bias_term, g(&e["bias_term"]), TOL, "bias_term");
    close(
        c.beta_corrected,
        g(&e["beta_corrected"]),
        TOL,
        "beta_corrected",
    );
    close(c.se, g(&e["se"]), TOL, "se(beta_corrected)");
}

#[test]
fn scalar_ivx_matches_documented_formula() {
    let fx = load();
    let s = &fx["scalar"];
    let r = f64s(&s["r"]);
    let x = f64s(&s["x"]);
    let e = &s["ivx"];
    let cfg = IvxConfig {
        cz: g(&e["cz"]),
        alpha: g(&e["alpha"]),
    };
    let fit = ivx(&r, &x, cfg).expect("ivx");
    close(fit.rz, g(&e["Rz"]), TOL, "Rz");
    close(fit.beta_ivx, g(&e["beta_ivx"]), TOL, "beta_ivx");
    close(fit.wald, g(&e["wald"]), TOL, "wald");
    close(fit.pvalue, g(&e["pvalue"]), 1e-8, "pvalue");
    // The full instrument path.
    let z_expected = f64s(&e["z"]);
    assert_eq!(fit.instrument.len(), z_expected.len(), "instrument length");
    for (t, (a, b)) in fit.instrument.iter().zip(&z_expected).enumerate() {
        close(*a, *b, TOL, &format!("z_{t}"));
    }
}

#[test]
fn multi_ivx_matches_documented_formula() {
    let fx = load();
    let m = &fx["multi"];
    let r = f64s(&m["r"]);
    let x1 = f64s(&m["x1"]);
    let x2 = f64s(&m["x2"]);
    let e = &m["ivx"];
    let cfg = IvxConfig {
        cz: g(&e["cz"]),
        alpha: g(&e["alpha"]),
    };
    let fit = ivx_multi(&r, &[x1, x2], cfg).expect("ivx_multi");
    let beta = f64s(&e["beta_ivx"]);
    close(fit.beta_ivx[0], beta[0], TOL, "beta_ivx[0]");
    close(fit.beta_ivx[1], beta[1], TOL, "beta_ivx[1]");
    close(fit.sigma2_u, g(&e["s2u"]), TOL, "s2u");
    close(fit.wald, g(&e["wald"]), TOL, "wald");
    close(fit.pvalue, g(&e["pvalue"]), 1e-8, "pvalue");
}
