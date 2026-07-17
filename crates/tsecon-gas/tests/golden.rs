//! Golden-value tests against `fixtures/tsecon-gas.json`, a
//! documented-formula golden: the filtered variance path `f_t`, the total
//! log-likelihood, the one-step-ahead variance, and the multi-step
//! forecast are computed in NumPy by literally applying the score-driven
//! recursion and observation density (see the fixture generator's
//! docstring). Rust must reproduce them to ~1e-10.

use serde_json::Value;
use tsecon_gas::{Density, GasModel, GasParams};

fn load_fixture() -> Value {
    let path = format!(
        "{}/../../fixtures/tsecon-gas.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let text = std::fs::read_to_string(&path).expect("read fixture");
    serde_json::from_str(&text).expect("parse fixture")
}

fn as_f64_vec(v: &Value) -> Vec<f64> {
    v.as_array()
        .expect("array")
        .iter()
        .map(|x| x.as_f64().expect("number"))
        .collect()
}

fn assert_close(actual: f64, expected: f64, tol: f64, what: &str) {
    let scale = expected.abs().max(1.0);
    assert!(
        (actual - expected).abs() <= tol * scale,
        "{what}: {actual} vs {expected} (diff {:e}, tol {tol:e})",
        (actual - expected).abs()
    );
}

/// Gaussian GAS(1,1): filtered variance path, total log-likelihood, the
/// one-step-ahead variance, and the 10-step forecast all reproduce the
/// documented NumPy recursion to 1e-10.
#[test]
fn golden_gaussian() {
    let fx = load_fixture();
    let case = &fx["gaussian_golden"];
    let y = as_f64_vec(&case["y"]);
    let p = &case["params"];
    let params = GasParams::gaussian(
        p["omega"].as_f64().unwrap(),
        p["a"].as_f64().unwrap(),
        p["b"].as_f64().unwrap(),
    );
    let model = GasModel::new(&y, Density::Gaussian).unwrap();
    let out = model.filter(&params).unwrap();

    let exp_var = as_f64_vec(&case["variance"]);
    assert_eq!(out.variance.len(), exp_var.len());
    for (t, (&a, &e)) in out.variance.iter().zip(&exp_var).enumerate() {
        assert_close(a, e, 1e-10, &format!("gaussian f[{t}]"));
    }
    assert_close(
        out.loglik,
        case["loglik"].as_f64().unwrap(),
        1e-10,
        "gaussian loglik",
    );
    assert_close(
        out.next_variance,
        case["next_variance"].as_f64().unwrap(),
        1e-10,
        "gaussian next_variance",
    );

    let exp_fc = as_f64_vec(&case["forecast"]);
    let fc = model.forecast(&params, exp_fc.len()).unwrap();
    for (h, (&a, &e)) in fc.iter().zip(&exp_fc).enumerate() {
        assert_close(a, e, 1e-10, &format!("gaussian forecast[{h}]"));
    }
}

/// Student-t GAS(1,1): the outlier-robust scaled score, the filtered path,
/// the log-likelihood (whose density piece is additionally cross-checked
/// against scipy.stats.t in the generator), the one-step variance, and the
/// forecast all reproduce the documented recursion to 1e-10.
#[test]
fn golden_student_t() {
    let fx = load_fixture();
    let case = &fx["student_t_golden"];
    let y = as_f64_vec(&case["y"]);
    let p = &case["params"];
    let params = GasParams::student_t(
        p["omega"].as_f64().unwrap(),
        p["a"].as_f64().unwrap(),
        p["b"].as_f64().unwrap(),
        p["nu"].as_f64().unwrap(),
    );
    let model = GasModel::new(&y, Density::StudentT).unwrap();
    let out = model.filter(&params).unwrap();

    let exp_var = as_f64_vec(&case["variance"]);
    assert_eq!(out.variance.len(), exp_var.len());
    for (t, (&a, &e)) in out.variance.iter().zip(&exp_var).enumerate() {
        assert_close(a, e, 1e-10, &format!("student-t f[{t}]"));
    }
    assert_close(
        out.loglik,
        case["loglik"].as_f64().unwrap(),
        1e-10,
        "student-t loglik",
    );
    assert_close(
        out.next_variance,
        case["next_variance"].as_f64().unwrap(),
        1e-10,
        "student-t next_variance",
    );

    let exp_fc = as_f64_vec(&case["forecast"]);
    let fc = model.forecast(&params, exp_fc.len()).unwrap();
    for (h, (&a, &e)) in fc.iter().zip(&exp_fc).enumerate() {
        assert_close(a, e, 1e-10, &format!("student-t forecast[{h}]"));
    }
}

/// The Student-t scaling constant `E[g^2] = 2 nu/(nu+3)` recorded in the
/// fixture (validated there by numerical integration) matches the closed
/// form the Rust scaled score is built on.
#[test]
fn golden_scaling_constant() {
    let fx = load_fixture();
    for entry in fx["scaling_check"].as_array().unwrap() {
        let nu = entry["nu"].as_f64().unwrap();
        let numeric = entry["e_g2_numeric"].as_f64().unwrap();
        let analytic = 2.0 * nu / (nu + 3.0);
        assert_close(
            entry["e_g2_analytic"].as_f64().unwrap(),
            analytic,
            1e-12,
            "analytic E[g^2]",
        );
        assert_close(numeric, analytic, 1e-8, "numeric E[g^2]");
    }
}
