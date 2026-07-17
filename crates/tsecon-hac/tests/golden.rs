//! Golden-value tests against `fixtures/hac.json` (generated with
//! statsmodels 0.14.6; see `fixtures/generate_fixtures.py::gen_hac`).
//!
//! The regression block pins OLS params, nonrobust bse, and HAC bse /
//! tvalues over maxlags {4, 8, 12} x use_correction {true, false}; the
//! `lrv_nile_demeaned` block pins Bartlett and EWC long-run variances on
//! the demeaned Nile (loaded from `fixtures/diagnostics.json`). Spec
//! tolerance is 1e-10 relative; everything is asserted at that bound.

use serde_json::Value;
use tsecon_hac::{ewc_lrv, lrv, newey_west_maxlags, ols, Kernel, SeType};

fn load(name: &str) -> Value {
    let path = format!("{}/../../fixtures/{name}", env!("CARGO_MANIFEST_DIR"));
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

/// Relative comparison; falls back to absolute when the reference is 0.
fn assert_close(actual: f64, expected: f64, rtol: f64, ctx: &str) {
    if expected == 0.0 {
        assert!(
            actual.abs() <= rtol,
            "{ctx}: actual {actual}, expected 0 (atol {rtol})"
        );
    } else {
        let rel = ((actual - expected) / expected).abs();
        assert!(
            rel <= rtol,
            "{ctx}: actual {actual}, expected {expected}, rel err {rel:e} > {rtol:e}"
        );
    }
}

fn assert_all_close(actual: &[f64], expected: &[f64], rtol: f64, ctx: &str) {
    assert_eq!(actual.len(), expected.len(), "{ctx}: length mismatch");
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_close(a, e, rtol, &format!("{ctx}[{i}]"));
    }
}

const TOL: f64 = 1e-10;

/// Design [const, x1, x2] exactly as the fixture assembles it.
fn regression_design(fx: &Value) -> (Vec<f64>, Vec<Vec<f64>>) {
    let reg = &fx["regression"];
    let y = f64s(&reg["y"]);
    let x1 = f64s(&reg["x1"]);
    let x2 = f64s(&reg["x2"]);
    let constant = vec![1.0; y.len()];
    (y, vec![constant, x1, x2])
}

fn demeaned_nile() -> Vec<f64> {
    let y = f64s(&load("diagnostics.json")["nile"]);
    let mean = y.iter().sum::<f64>() / y.len() as f64;
    y.iter().map(|v| v - mean).collect()
}

#[test]
fn ols_params_and_nonrobust_bse_match_statsmodels() {
    let fx = load("hac.json");
    let (y, x) = regression_design(&fx);
    let fit = ols(&y, &x).unwrap();

    assert_all_close(
        &fit.params,
        &f64s(&fx["regression"]["ols_params"]),
        TOL,
        "ols params",
    );
    let inf = fit.inference(SeType::NonRobust).unwrap();
    assert_all_close(
        &inf.bse,
        &f64s(&fx["regression"]["ols_bse_nonrobust"]),
        TOL,
        "nonrobust bse",
    );
}

#[test]
fn hac_bse_and_tvalues_match_statsmodels_all_cases() {
    let fx = load("hac.json");
    let (y, x) = regression_design(&fx);
    let fit = ols(&y, &x).unwrap();

    let cases = fx["regression"]["hac_cases"].as_array().expect("cases");
    assert_eq!(cases.len(), 6, "expected 3 maxlags x 2 corrections");
    for case in cases {
        let maxlags = case["maxlags"].as_u64().expect("maxlags") as f64;
        let use_correction = case["use_correction"].as_bool().expect("flag");
        let ctx = format!("HAC maxlags={maxlags} correction={use_correction}");

        let inf = fit
            .inference(SeType::Hac {
                kernel: Kernel::Bartlett,
                bandwidth: maxlags,
                use_correction,
            })
            .unwrap();
        assert_all_close(&inf.bse, &f64s(&case["bse"]), TOL, &format!("{ctx} bse"));
        assert_all_close(
            &inf.tvalues,
            &f64s(&case["tvalues"]),
            TOL,
            &format!("{ctx} tvalues"),
        );
    }
}

#[test]
fn bartlett_lrv_on_demeaned_nile_matches_fixture() {
    let fx = load("hac.json");
    let z = demeaned_nile();
    for bw in [5_usize, 10, 20] {
        let expected = fx["lrv_nile_demeaned"]["bartlett"][bw.to_string()]
            .as_f64()
            .expect("bartlett value");
        let actual = lrv(&z, Kernel::Bartlett, bw as f64).unwrap();
        assert_close(actual, expected, TOL, &format!("bartlett lrv bw={bw}"));
    }
}

#[test]
fn ewc_lrv_on_demeaned_nile_matches_fixture() {
    let fx = load("hac.json");
    let z = demeaned_nile();
    for b in [4_usize, 8, 16] {
        let expected = fx["lrv_nile_demeaned"]["ewc"][b.to_string()]
            .as_f64()
            .expect("ewc value");
        let actual = ewc_lrv(&z, b).unwrap();
        assert_close(actual, expected, TOL, &format!("ewc lrv B={b}"));
    }
}

#[test]
fn newey_west_maxlags_rule_matches_fixture_integer() {
    let fx = load("hac.json");
    let z = demeaned_nile();
    let expected = fx["lrv_nile_demeaned"]["newey_west_auto_maxlags_floor_4_n100_2_9"]
        .as_u64()
        .expect("integer") as usize;
    assert_eq!(newey_west_maxlags(z.len()), expected);
}
