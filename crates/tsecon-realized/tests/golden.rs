//! Golden-value tests against `fixtures/realized.json` (generated with
//! statsmodels 0.14.6 / NumPy 2.5.1; see the fixture `_meta`).
//!
//! `measures_small` pins realized variance and bipower variation on a fixed
//! 7-element return vector to 1e-12; `har` pins the Corsi (2009) HAR-RV
//! params / HAC bse / centered R^2 to 1e-8. The HAR block is exactly OLS on
//! `[const, RV_{t-1}, mean(RV[t-6..t-1]), mean(RV[t-23..t-1])]` over
//! `t = start+1 .. n-1`, with Bartlett HAC(maxlags=5, use_correction=false).

use serde_json::Value;
use tsecon_realized::{bipower_variation, har_rv, realized_variance, HarConfig, HarVariant};

fn load() -> Value {
    let path = format!(
        "{}/../../fixtures/realized.json",
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

fn assert_all_close(actual: &[f64], expected: &[f64], atol: f64, ctx: &str) {
    assert_eq!(actual.len(), expected.len(), "{ctx}: length mismatch");
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() <= atol,
            "{ctx}[{i}]: actual {a}, expected {e}, |diff| {:e} > {atol:e}",
            (a - e).abs()
        );
    }
}

#[test]
fn measures_small_rv_and_bipower_match_fixture() {
    let fx = load();
    let m = &fx["measures_small"];
    let returns = f64s(&m["returns"]);

    let rv = realized_variance(&returns).unwrap();
    let bv = bipower_variation(&returns).unwrap();

    let rv_exp = m["rv"].as_f64().expect("rv");
    let bv_exp = m["bipower"].as_f64().expect("bipower");
    assert!(
        (rv - rv_exp).abs() <= 1e-12,
        "realized variance: {rv} vs {rv_exp}"
    );
    assert!(
        (bv - bv_exp).abs() <= 1e-12,
        "bipower variation: {bv} vs {bv_exp}"
    );
}

#[test]
fn har_params_bse_rsquared_match_fixture() {
    let fx = load();
    let rv = f64s(&fx["rv_series"]);
    let har = &fx["har"];
    let start = har["start"].as_u64().expect("start") as usize;

    let config = HarConfig {
        start,
        variant: HarVariant::Level,
        hac_maxlags: 5,
        use_correction: false,
    };
    let fit = har_rv(&rv, &config).unwrap();

    assert_all_close(&fit.params, &f64s(&har["params"]), 1e-8, "har params");
    assert_all_close(&fit.bse, &f64s(&har["bse"]), 1e-8, "har bse");

    let r2_exp = har["rsquared"].as_f64().expect("rsquared");
    assert!(
        (fit.rsquared - r2_exp).abs() <= 1e-8,
        "har rsquared: {} vs {r2_exp}",
        fit.rsquared
    );
}

#[test]
fn har_default_config_uses_fixture_settings() {
    // The library default (start=22, Level, maxlags=5, no correction) must
    // reproduce the fixture without any explicit configuration.
    let fx = load();
    let rv = f64s(&fx["rv_series"]);
    let fit = har_rv(&rv, &HarConfig::default()).unwrap();
    assert_all_close(
        &fit.params,
        &f64s(&fx["har"]["params"]),
        1e-8,
        "default params",
    );
}
