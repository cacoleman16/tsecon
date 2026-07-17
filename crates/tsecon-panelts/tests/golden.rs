//! Golden-value tests against `fixtures/tsecon-panelts.json`.
//!
//! The fixture DGP is a heterogeneous panel `y_it = a_i + b_i' x_it +
//! gamma_i f_t + e_it` (N = 24 units, T = 70 periods, K = 2 regressors) with a
//! single common factor `f_t` loaded heterogeneously into `y` and into every
//! `x`. The golden MG and CCE-MG numbers come from per-unit `statsmodels.OLS`
//! fits (Moore-Penrose / SVD least squares) averaged with the closed-form
//! mean-group formulas; see `fixtures/generate_tsecon-panelts_fixtures.py`.
//!
//! This crate solves the identical least-squares problems through a *different*
//! numerical path — Cholesky normal equations in `tsecon-hac::ols` — and
//! applies the same deterministic averaging. Because the panel is stored at
//! full float precision and parsed with serde_json's `float_roundtrip`, both
//! sides evaluate the same map on bit-identical inputs, so:
//!
//! * MG / CCE-MG `coef`, per-unit slopes match to `1e-10`;
//! * `se` and `tstat` match to `1e-10` (deterministic functions of the slopes).

use serde_json::Value;
use tsecon_panelts::{cce_mean_group, mean_group, PanelUnit};

fn load() -> Value {
    let path = format!(
        "{}/../../fixtures/tsecon-panelts.json",
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
        "{ctx}: actual {actual}, expected {expected}, abs err {err:e} > tol {atol:e}"
    );
}

/// Rebuild the panel of [`PanelUnit`]s from the fixture (`y` is N x T, `x` is
/// K x N x T).
fn build_units(fx: &Value) -> Vec<PanelUnit> {
    let n = fx["design"]["N"].as_u64().expect("N") as usize;
    let k = fx["design"]["K"].as_u64().expect("K") as usize;
    let y = f64_matrix(&fx["y"]); // N x T
    let x: Vec<Vec<Vec<f64>>> = (0..k).map(|j| f64_matrix(&fx["x"][j])).collect(); // K x N x T

    (0..n)
        .map(|i| {
            let cols: Vec<Vec<f64>> = (0..k).map(|j| x[j][i].clone()).collect();
            PanelUnit::new(y[i].clone(), cols)
        })
        .collect()
}

fn check_against(fit_coef: &[f64], fit_se: &[f64], fit_t: &[f64], block: &Value, label: &str) {
    let coef = f64s(&block["coef"]);
    let se = f64s(&block["se"]);
    let tstat = f64s(&block["tstat"]);
    for j in 0..coef.len() {
        assert_close(fit_coef[j], coef[j], 1e-10, &format!("{label} coef[{j}]"));
        assert_close(fit_se[j], se[j], 1e-10, &format!("{label} se[{j}]"));
        assert_close(fit_t[j], tstat[j], 1e-10, &format!("{label} tstat[{j}]"));
    }
}

#[test]
fn mean_group_matches_statsmodels_golden() {
    let fx = load();
    let units = build_units(&fx);
    let fit = mean_group(&units).expect("MG fits");

    check_against(&fit.coef, &fit.se, &fit.tstat, &fx["mg"], "MG");

    // Per-unit slope vectors match to machine precision, too.
    let per_unit = f64_matrix(&fx["mg"]["coef_per_unit"]);
    assert_eq!(fit.coef_per_unit.len(), per_unit.len());
    for (i, (got, exp)) in fit.coef_per_unit.iter().zip(per_unit.iter()).enumerate() {
        for j in 0..got.len() {
            assert_close(got[j], exp[j], 1e-10, &format!("MG per-unit[{i}][{j}]"));
        }
    }
}

#[test]
fn cce_mean_group_matches_statsmodels_golden() {
    let fx = load();
    let units = build_units(&fx);
    let fit = cce_mean_group(&units).expect("CCE-MG fits");

    check_against(&fit.coef, &fit.se, &fit.tstat, &fx["cce"], "CCE-MG");

    let per_unit = f64_matrix(&fx["cce"]["coef_per_unit"]);
    assert_eq!(fit.coef_per_unit.len(), per_unit.len());
    for (i, (got, exp)) in fit.coef_per_unit.iter().zip(per_unit.iter()).enumerate() {
        for j in 0..got.len() {
            assert_close(got[j], exp[j], 1e-10, &format!("CCE per-unit[{i}][{j}]"));
        }
    }
}

/// The fixture also demonstrates the raison d'être of CCE: plain MG is
/// noticeably biased for the true mean slope under the common factor, while
/// CCE-MG is close. This is a sanity check on the stored numbers themselves.
#[test]
fn fixture_shows_mg_bias_cce_close() {
    let fx = load();
    let truth = f64s(&fx["true_mean_slopes"]);
    let mg = f64s(&fx["mg"]["coef"]);
    let cce = f64s(&fx["cce"]["coef"]);
    for j in 0..truth.len() {
        let mg_bias = (mg[j] - truth[j]).abs();
        let cce_bias = (cce[j] - truth[j]).abs();
        assert!(
            cce_bias < mg_bias,
            "coord {j}: CCE bias {cce_bias:e} should beat MG bias {mg_bias:e}"
        );
    }
}
