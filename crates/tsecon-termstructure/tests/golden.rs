//! Golden-value tests against `fixtures/termstructure.json`.
//!
//! The fixture (NumPy 2.5.1) documents the Nelson-Siegel loadings at
//! `lambda = 0.0609` (Diebold-Li 2006, monthly), and the cross-sectional OLS
//! of `yields_date100` on those loadings — the `[level, slope, curvature]`
//! factors and the centered R^2.

use serde_json::Value;
use tsecon_termstructure::{fit_nelson_siegel, nelson_siegel_loadings};

fn load() -> Value {
    let path = format!(
        "{}/../../fixtures/termstructure.json",
        env!("CARGO_MANIFEST_DIR")
    );
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

fn assert_close(actual: f64, expected: f64, atol: f64, ctx: &str) {
    let err = (actual - expected).abs();
    assert!(
        err <= atol,
        "{ctx}: actual {actual}, expected {expected}, abs err {err:e} > atol {atol:e}"
    );
}

#[test]
fn ns_loadings_match_numpy() {
    let fx = load();
    let maturities = f64s(&fx["maturities"]);
    let lambda = fx["lambda"].as_f64().expect("lambda");
    let expected = f64_matrix(&fx["ns_loadings"]); // 3 columns x 8 maturities

    let [level, slope, curvature] = nelson_siegel_loadings(&maturities, lambda).expect("loadings");
    let got = [level, slope, curvature];

    for (col, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g.len(), e.len(), "column {col} length");
        for (i, (&gi, &ei)) in g.iter().zip(e.iter()).enumerate() {
            assert_close(gi, ei, 1e-10, &format!("ns_loadings[{col}][{i}]"));
        }
    }
}

#[test]
fn ns_fit_factors_and_rsquared_match_numpy() {
    let fx = load();
    let maturities = f64s(&fx["maturities"]);
    let lambda = fx["lambda"].as_f64().expect("lambda");
    let yields = f64s(&fx["yields_date100"]);
    let expected_factors = f64s(&fx["ns_fit_factors"]);
    let expected_r2 = fx["ns_fit_rsquared"].as_f64().expect("rsquared");

    let fit = fit_nelson_siegel(&maturities, &yields, lambda).expect("fit");

    for (i, (&g, &e)) in fit.factors.iter().zip(expected_factors.iter()).enumerate() {
        assert_close(g, e, 1e-8, &format!("ns_fit_factors[{i}]"));
    }
    assert_close(fit.rsquared, expected_r2, 1e-8, "ns_fit_rsquared");
}
