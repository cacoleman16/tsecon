//! Independent-reference golden tests for the quantile estimators.
//!
//! `fixtures/tsecon-quantile.json` is produced by
//! `fixtures/generate_tsecon-quantile_fixtures.py`:
//!
//! * quantile regression is pinned to statsmodels
//!   `QuantReg(endog, exog).fit(q=tau)` (all defaults: Schnabel-Koenker
//!   IRLS, robust Powell sandwich, Epanechnikov kernel, Hall-Sheather
//!   bandwidth) — `params`, `bse`, `bandwidth`, `sparsity`, `iterations`;
//! * quantile LPs are pinned to the same statsmodels reference run per
//!   horizon on a numpy-assembled design in tsecon-lp's column order;
//! * growth-at-risk is pinned to per-tau statsmodels fits plus a numpy
//!   `sort` rearrangement, including one case where the raw quantile
//!   curves genuinely cross.
//!
//! TOLERANCES. Coefficients: 1e-6, which is the IRLS stopping tolerance
//! `p_tol` both implementations share — beyond it the two IRLS paths may
//! legitimately stop one iteration apart, and one extra iteration moves the
//! coefficients by at most ~p_tol. Everything downstream of the residuals
//! (bse, bandwidth, sparsity, fitted paths) inherits that wobble; those are
//! pinned at 1e-6 relative. Iteration counts must match within ±2 (a float
//! straddle of p_tol can shift the count by one on either side).

use serde_json::Value;
use tsecon_quantile::{growth_at_risk, quantile_lp, quantile_regression};

const TOL: f64 = 1e-6;

fn load() -> Value {
    let path = format!(
        "{}/../../fixtures/tsecon-quantile.json",
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

fn columns(v: &Value) -> Vec<Vec<f64>> {
    v.as_array().expect("array").iter().map(f64s).collect()
}

fn g(v: &Value) -> f64 {
    v.as_f64().expect("number")
}

fn close(actual: f64, expected: f64, what: &str) {
    let err = (actual - expected).abs();
    let rel = err / (1.0 + expected.abs());
    assert!(
        err < TOL || rel < TOL,
        "{what}: actual={actual:.15e} expected={expected:.15e} abs_err={err:.3e} rel={rel:.3e}"
    );
}

fn close_slice(actual: &[f64], expected: &[f64], what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what} length");
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        close(a, e, &format!("{what}[{i}]"));
    }
}

#[test]
fn quantile_regression_matches_statsmodels_quantreg() {
    let fx = load();
    let cases = fx["qreg"].as_array().expect("array");
    assert!(!cases.is_empty());
    for case in cases {
        let name = case["name"].as_str().unwrap_or("?");
        let y = f64s(&case["y"]);
        let cols = columns(&case["columns"]);
        let fits_fx = case["fits"].as_array().expect("array");
        let taus: Vec<f64> = fits_fx.iter().map(|f| g(&f["tau"])).collect();

        let fits = quantile_regression(&y, &cols, &taus).expect("fit ok");
        assert_eq!(fits.len(), taus.len());
        for (fit, fxf) in fits.iter().zip(fits_fx.iter()) {
            let label = format!("qreg[{name}] tau={}", fit.tau);
            close_slice(
                &fit.params,
                &f64s(&fxf["params"]),
                &format!("{label} params"),
            );
            close_slice(&fit.bse, &f64s(&fxf["bse"]), &format!("{label} bse"));
            close(
                fit.bandwidth,
                g(&fxf["bandwidth"]),
                &format!("{label} bandwidth"),
            );
            close(
                fit.sparsity,
                g(&fxf["sparsity"]),
                &format!("{label} sparsity"),
            );
            let iters_fx = fxf["iterations"].as_u64().expect("uint") as i64;
            assert!(
                (fit.iterations as i64 - iters_fx).abs() <= 2,
                "{label} iterations: got {} expected {iters_fx} (±2)",
                fit.iterations
            );
            assert!(fit.converged, "{label} must converge");
        }
    }
}

#[test]
fn quantile_lp_matches_statsmodels_per_horizon() {
    let fx = load();
    let case = &fx["qlp"];
    let y = f64s(&case["y"]);
    let shock = f64s(&case["shock"]);
    let taus = f64s(&case["taus"]);
    let horizons = case["horizons"].as_u64().expect("uint") as usize;
    let p = case["n_lag_controls"].as_u64().expect("uint") as usize;

    let r = quantile_lp(&y, &shock, &taus, horizons, p).expect("qlp ok");
    assert_eq!(r.horizons, (0..=horizons).collect::<Vec<_>>());
    let irf_fx = columns(&case["irf"]);
    let se_fx = columns(&case["se"]);
    for (i, &tau) in taus.iter().enumerate() {
        close_slice(&r.irf[i], &irf_fx[i], &format!("qlp irf tau={tau}"));
        close_slice(&r.se[i], &se_fx[i], &format!("qlp se tau={tau}"));
    }
}

#[test]
fn growth_at_risk_matches_statsmodels_plus_rearrangement() {
    let fx = load();
    let cases = fx["gar"].as_array().expect("array");
    assert!(!cases.is_empty());
    // The fixture set must include a genuine crossing so the rearrangement
    // is pinned doing real work, and a non-crossing case for the null path.
    let n_crossing = cases
        .iter()
        .filter(|c| c["crossing"].as_bool().expect("bool"))
        .count();
    assert!(n_crossing >= 1, "fixture must exercise a real crossing");
    assert!(
        n_crossing < cases.len(),
        "fixture must include a clean case"
    );

    for case in cases {
        let name = case["name"].as_str().unwrap_or("?");
        let y = f64s(&case["y"]);
        let conditions = columns(&case["conditions"]);
        let horizon = case["horizon"].as_u64().expect("uint") as usize;
        let taus = f64s(&case["taus"]);

        let r = growth_at_risk(&y, &conditions, horizon, &taus, true).expect("gar ok");
        let params_fx = columns(&case["params"]);
        let bse_fx = columns(&case["bse"]);
        let raw_fx = columns(&case["fitted_raw"]);
        let rearr_fx = columns(&case["fitted_rearranged"]);
        for (i, &tau) in taus.iter().enumerate() {
            let label = format!("gar[{name}] tau={tau}");
            close_slice(&r.params[i], &params_fx[i], &format!("{label} params"));
            close_slice(&r.bse[i], &bse_fx[i], &format!("{label} bse"));
            close_slice(&r.fitted_raw[i], &raw_fx[i], &format!("{label} fitted_raw"));
            close_slice(
                &r.fitted[i],
                &rearr_fx[i],
                &format!("{label} fitted (rearranged)"),
            );
        }
        assert_eq!(
            r.crossing,
            case["crossing"].as_bool().expect("bool"),
            "gar[{name}] crossing flag"
        );
        close_slice(
            &r.current,
            &f64s(&case["current"]),
            &format!("gar[{name}] current"),
        );

        // rearrange = false must return the raw fits unchanged.
        let raw = growth_at_risk(&y, &conditions, horizon, &taus, false).expect("gar raw ok");
        assert_eq!(
            raw.fitted, raw.fitted_raw,
            "gar[{name}] rearrange=false identity"
        );
        assert_eq!(
            raw.crossing, r.crossing,
            "gar[{name}] crossing is treatment-free"
        );
    }
}
