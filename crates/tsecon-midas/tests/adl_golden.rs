//! Golden tests for ADL-MIDAS against `fixtures/adl_midas.json`.
//!
//! The fixture (statsmodels 0.15.0) fits OLS of `y_t` on
//! `[1, y_{t-1..t-P}, hf_lags[t, :]]` over the rows `t = P..n-1`, with the
//! y-lag columns built by `pandas.Series.shift`, under nonrobust and HAC
//! (Bartlett, `use_correction=True`) covariances — at the effective-rows
//! Newey-West default bandwidth and at `maxlags = 8`. Here the same design is
//! assembled the way the Python binding does (y-lag column `p` is
//! `y[P - p .. n - p]`) and passed to [`tsecon_midas::adl_midas`]; that pins
//! the alignment, the parameter order `[const, rho_1..rho_P, b_1..b_K]`, and
//! the standard errors at 1e-10.

use serde_json::Value;
use tsecon_hac::{newey_west_maxlags, Kernel};
use tsecon_midas::{adl_midas, SeType};

fn load() -> Value {
    let path = format!(
        "{}/../../fixtures/adl_midas.json",
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

/// `(y_eff, y_lag_columns, hf_columns)` for ADL-MIDAS with `p` AR lags: the
/// binding's alignment.
fn design(y: &[f64], hf_rows: &[Vec<f64>], p: usize) -> (Vec<f64>, Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let n = y.len();
    let k = hf_rows[0].len();
    let y_eff = y[p..].to_vec();
    let y_lags = (1..=p).map(|lag| y[p - lag..n - lag].to_vec()).collect();
    let hf = (0..k)
        .map(|j| hf_rows[p..].iter().map(|row| row[j]).collect())
        .collect();
    (y_eff, y_lags, hf)
}

fn assert_rel(actual: &[f64], expected: &[f64], tol: f64, what: &str) -> f64 {
    assert_eq!(actual.len(), expected.len(), "{what}: length");
    let mut worst = 0.0f64;
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        let d = (a - e).abs() / e.abs().max(1.0);
        worst = worst.max(d);
        assert!(d <= tol, "{what}[{i}]: {a} vs {e} (rel {d:e})");
    }
    worst
}

#[test]
fn adl_midas_matches_statsmodels_ols_and_hac() {
    let fx = load();
    let mut worst = 0.0f64;
    for case in fx["cases"].as_array().expect("cases") {
        let name = case["name"].as_str().unwrap();
        let p = case["ar_lags"].as_u64().unwrap() as usize;
        let y = f64s(&case["y"]);
        let hf_rows: Vec<Vec<f64>> = case["hf_lags"]
            .as_array()
            .unwrap()
            .iter()
            .map(f64s)
            .collect();
        let (y_eff, y_lags, hf) = design(&y, &hf_rows, p);
        let nobs = case["nobs"].as_u64().unwrap() as usize;
        assert_eq!(y_eff.len(), nobs);
        // The binding's default bandwidth is the Newey-West rule on the
        // EFFECTIVE rows n - P, which the fixture records.
        assert_eq!(
            newey_west_maxlags(nobs),
            case["default_maxlags"].as_u64().unwrap() as usize,
            "{name}: default maxlags"
        );
        for (label, fit) in case["fits"].as_object().unwrap() {
            let se = match fit["se_type"].as_str().unwrap() {
                "nonrobust" => SeType::NonRobust,
                _ => SeType::Hac {
                    kernel: Kernel::Bartlett,
                    bandwidth: fit["maxlags"].as_u64().unwrap() as f64,
                    use_correction: true,
                },
            };
            let r = adl_midas(&y_eff, &y_lags, &hf, se).expect("adl_midas fits");
            let what = format!("{name}/{label}");
            assert_eq!(r.nobs, nobs);
            worst = worst.max(assert_rel(
                &r.params,
                &f64s(&case["params"]),
                1e-10,
                &format!("{what} params"),
            ));
            worst = worst.max(assert_rel(
                &r.bse,
                &f64s(&fit["bse"]),
                1e-10,
                &format!("{what} bse"),
            ));
            worst = worst.max(assert_rel(
                &r.tvalues,
                &f64s(&fit["tvalues"]),
                1e-10,
                &format!("{what} tvalues"),
            ));
            worst = worst.max(assert_rel(
                &r.residuals,
                &f64s(&case["residuals"]),
                1e-10,
                &format!("{what} residuals"),
            ));
            worst = worst.max(assert_rel(
                &[r.rsquared],
                &[case["rsquared"].as_f64().unwrap()],
                1e-10,
                &format!("{what} rsquared"),
            ));
        }
    }
    eprintln!("adl_midas worst relative deviation {worst:e}");
    // Measured: 3.2e-14 worst relative deviation on the committed fixture.
    assert!(worst < 1e-11, "worst deviation {worst:e}");
}

#[test]
fn effective_rows_bandwidth_differs_from_the_raw_length_rule() {
    // The n = 275, P = 4 case is where a copy of umidas' rule (on the raw
    // length) would silently pick 5 lags instead of 4.
    let fx = load();
    let case = &fx["cases"][1];
    assert_eq!(case["name"], "n275_p4");
    let n = case["n"].as_u64().unwrap() as usize;
    let p = case["ar_lags"].as_u64().unwrap() as usize;
    assert_eq!(newey_west_maxlags(n - p), 4);
    assert_eq!(newey_west_maxlags(n), 5);
    assert_eq!(case["maxlags_rule_on_raw_n"].as_u64().unwrap(), 5);
}

#[test]
fn hf_column_one_is_the_last_high_frequency_observation() {
    // Within one case, column c + ratio equals column c lagged one
    // low-frequency period: the most-recent-first stacking the names hf1..hfK
    // describe (hf1 = high-frequency lag 0).
    let fx = load();
    for case in fx["cases"].as_array().unwrap() {
        let ratio = case["ratio"].as_u64().unwrap() as usize;
        let rows: Vec<Vec<f64>> = case["hf_lags"]
            .as_array()
            .unwrap()
            .iter()
            .map(f64s)
            .collect();
        let k = rows[0].len();
        for t in 1..rows.len() {
            for c in 0..k - ratio {
                assert_eq!(rows[t][c + ratio].to_bits(), rows[t - 1][c].to_bits());
            }
        }
    }
}
