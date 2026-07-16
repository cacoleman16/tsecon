//! Golden-value tests for the unit-root layer against
//! `fixtures/unitroot.json` (generated with statsmodels 0.14.6 on the Nile
//! series and a seeded random walk; see `fixtures/generate_fixtures.py`).
//!
//! Tolerances per the spec: ADF statistics to 1e-8 and p-values to 1e-7,
//! `usedlag`/`nobs` exact; KPSS statistics to 1e-8, lags exact, bounded
//! p-values to 1e-6; the MacKinnon p-value grid to 1e-8. Comparisons are
//! relative (absolute against 0), as in the crate's other golden tests.

use serde_json::Value;
use tsecon_diag::{
    adf, kpss, mackinnon_crit, mackinnon_p, AdfLagSelection, AdfRegression, KpssLags,
    KpssRegression,
};

fn fixture() -> Value {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../fixtures/unitroot.json");
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

fn series(fx: &Value, name: &str) -> Vec<f64> {
    f64s(&fx[name])
}

fn adf_regression(code: &str) -> AdfRegression {
    match code {
        "n" => AdfRegression::NoConstant,
        "c" => AdfRegression::Constant,
        "ct" => AdfRegression::ConstantTrend,
        other => panic!("unknown adf regression {other:?}"),
    }
}

#[test]
fn adf_cases_match_statsmodels() {
    let fx = fixture();
    for case in fx["adf"].as_array().expect("adf cases") {
        let name = case["series"].as_str().expect("series name");
        let reg_code = case["regression"].as_str().expect("regression");
        let regression = adf_regression(reg_code);
        let maxlag = case["maxlag"].as_u64().map(|v| v as usize);
        let (selection, autolag_label) = match case["autolag"].as_str() {
            Some("AIC") => (AdfLagSelection::Aic(maxlag), "AIC"),
            Some("BIC") => (AdfLagSelection::Bic(maxlag), "BIC"),
            Some("t-stat") => (AdfLagSelection::TStat(maxlag), "t-stat"),
            Some(other) => panic!("unknown autolag {other:?}"),
            None => (
                AdfLagSelection::Fixed(maxlag.expect("fixed case carries maxlag")),
                "fixed",
            ),
        };
        let ctx = format!("adf[{name}/{reg_code}/{autolag_label}]");
        let y = series(&fx, name);
        let res = adf(&y, regression, selection).expect("adf runs");

        assert_close(
            res.statistic,
            case["stat"].as_f64().expect("stat"),
            1e-8,
            &format!("{ctx} stat"),
        );
        assert_close(
            res.p_value,
            case["pvalue"].as_f64().expect("pvalue"),
            1e-7,
            &format!("{ctx} pvalue"),
        );
        assert_eq!(
            res.used_lag,
            case["usedlag"].as_u64().expect("usedlag") as usize,
            "{ctx} usedlag"
        );
        assert_eq!(
            res.nobs,
            case["nobs"].as_u64().expect("nobs") as usize,
            "{ctx} nobs"
        );
        let crit = &case["crit"];
        assert_close(
            res.crit.pct1,
            crit["1%"].as_f64().expect("1%"),
            1e-8,
            &format!("{ctx} crit 1%"),
        );
        assert_close(
            res.crit.pct5,
            crit["5%"].as_f64().expect("5%"),
            1e-8,
            &format!("{ctx} crit 5%"),
        );
        assert_close(
            res.crit.pct10,
            crit["10%"].as_f64().expect("10%"),
            1e-8,
            &format!("{ctx} crit 10%"),
        );
        assert_eq!(res.regression, regression, "{ctx} regression echoed");
    }
}

#[test]
fn kpss_cases_match_statsmodels() {
    let fx = fixture();
    for case in fx["kpss"].as_array().expect("kpss cases") {
        let name = case["series"].as_str().expect("series name");
        let reg_code = case["regression"].as_str().expect("regression");
        let regression = match reg_code {
            "c" => KpssRegression::Constant,
            "ct" => KpssRegression::ConstantTrend,
            other => panic!("unknown kpss regression {other:?}"),
        };
        let nlags_label = case["nlags"].as_str().expect("nlags");
        let lags = match nlags_label {
            "auto" => KpssLags::Auto,
            "legacy" => KpssLags::Legacy,
            other => panic!("unknown kpss nlags {other:?}"),
        };
        let ctx = format!("kpss[{name}/{reg_code}/{nlags_label}]");
        let y = series(&fx, name);
        let res = kpss(&y, regression, lags).expect("kpss runs");

        assert_close(
            res.statistic,
            case["stat"].as_f64().expect("stat"),
            1e-8,
            &format!("{ctx} stat"),
        );
        assert_close(
            res.p_value,
            case["pvalue_interpolated_bounded"]
                .as_f64()
                .expect("pvalue"),
            1e-6,
            &format!("{ctx} pvalue"),
        );
        assert_eq!(
            res.lags,
            case["lags"].as_u64().expect("lags") as usize,
            "{ctx} lags"
        );
        assert_eq!(res.nobs, y.len(), "{ctx} nobs");
        let crit = &case["crit"];
        for (actual, key) in [
            (res.crit.pct10, "10%"),
            (res.crit.pct5, "5%"),
            (res.crit.pct2_5, "2.5%"),
            (res.crit.pct1, "1%"),
        ] {
            assert_close(
                actual,
                crit[key].as_f64().expect("crit value"),
                1e-12,
                &format!("{ctx} crit {key}"),
            );
        }
    }
}

#[test]
fn mackinnon_pvalue_grid_matches_statsmodels() {
    let fx = fixture();
    let block = &fx["mackinnon_p_N1"];
    for code in ["n", "c", "ct"] {
        let regression = adf_regression(code);
        let stats = f64s(&block[code]["stat_grid"]);
        let expected = f64s(&block[code]["pvalues"]);
        assert_eq!(stats.len(), expected.len(), "grid lengths");
        for (&s, &p) in stats.iter().zip(expected.iter()) {
            assert_close(
                mackinnon_p(s, regression),
                p,
                1e-8,
                &format!("mackinnon_p[{code}](stat = {s})"),
            );
        }
    }
}

#[test]
fn mackinnon_asymptotic_critical_values_match_statsmodels() {
    let fx = fixture();
    let block = &fx["mackinnon_p_N1"];
    for code in ["n", "c", "ct"] {
        let expected = f64s(&block[code]["crit_1_5_10"]);
        let crit = mackinnon_crit(adf_regression(code), None);
        assert_close(crit.pct1, expected[0], 1e-12, &format!("{code} asy 1%"));
        assert_close(crit.pct5, expected[1], 1e-12, &format!("{code} asy 5%"));
        assert_close(crit.pct10, expected[2], 1e-12, &format!("{code} asy 10%"));
    }
}
