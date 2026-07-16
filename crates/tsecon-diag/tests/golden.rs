//! Golden-value tests against `fixtures/diagnostics.json` (generated with
//! statsmodels 0.14.6 on the Nile series; see
//! `fixtures/generate_fixtures.py`).
//!
//! Tolerances: the spec asks for 1e-8 on acf/pacf values, test statistics,
//! and p-values; everything actually matches at 1e-12 relative (absolute
//! when the reference is 0), so the tests pin the tighter bound. The
//! `levinson_durbin_10` block is validated in tsecon-linalg, not here.

use serde_json::Value;
use tsecon_diag::{acf, arch_lm, jarque_bera, ljung_box, pacf_ols, pacf_yw};

fn fixture() -> Value {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/diagnostics.json"
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

fn nile(fx: &Value) -> Vec<f64> {
    f64s(&fx["nile"])
}

fn demeaned(y: &[f64]) -> Vec<f64> {
    let m = y.iter().sum::<f64>() / y.len() as f64;
    y.iter().map(|&v| v - m).collect()
}

const TOL: f64 = 1e-12;

#[test]
fn acf_unadjusted_matches_statsmodels() {
    let fx = fixture();
    let y = nile(&fx);
    let expected = f64s(&fx["acf_20_unadjusted"]);
    let res = acf(&y, 20, false).unwrap();
    assert_all_close(&res.acf, &expected, TOL, "acf unadjusted");
    assert_eq!(res.bartlett_se.len(), 21);
}

#[test]
fn acf_adjusted_matches_statsmodels() {
    let fx = fixture();
    let y = nile(&fx);
    let expected = f64s(&fx["acf_20_adjusted"]);
    let res = acf(&y, 20, true).unwrap();
    assert_all_close(&res.acf, &expected, TOL, "acf adjusted");
}

#[test]
fn bartlett_se_matches_brockwell_davis_formula() {
    // No fixture block for the bands; pin them against a direct evaluation
    // of the Brockwell-Davis formula from the golden acf values instead.
    let fx = fixture();
    let y = nile(&fx);
    let r = f64s(&fx["acf_20_unadjusted"]);
    let n = y.len() as f64;
    let res = acf(&y, 20, false).unwrap();
    assert_eq!(res.bartlett_se[0], 0.0);
    assert_close(res.bartlett_se[1], (1.0 / n).sqrt(), TOL, "se[1]");
    for k in 2..=20 {
        let cum: f64 = r[1..k].iter().map(|&v| v * v).sum();
        let expected = ((1.0 + 2.0 * cum) / n).sqrt();
        assert_close(res.bartlett_se[k], expected, TOL, &format!("se[{k}]"));
    }
}

#[test]
fn pacf_ywm_matches_statsmodels() {
    let fx = fixture();
    let y = nile(&fx);
    let expected = f64s(&fx["pacf_20_ywm"]);
    let res = pacf_yw(&y, 20).unwrap();
    assert_all_close(&res, &expected, TOL, "pacf ywm");
}

#[test]
fn pacf_ols_matches_statsmodels() {
    let fx = fixture();
    let y = nile(&fx);
    let expected = f64s(&fx["pacf_20_ols"]);
    let res = pacf_ols(&y, 20).unwrap();
    assert_all_close(&res, &expected, TOL, "pacf ols");
}

#[test]
fn ljung_box_and_box_pierce_match_statsmodels() {
    let fx = fixture();
    let y = demeaned(&nile(&fx)); // fixture ran acorr_ljungbox on y - mean
    let block = &fx["ljung_box_lags_1_10"];
    let res = ljung_box(&y, 10).unwrap();
    assert_eq!(res.lags, (1..=10).collect::<Vec<_>>());
    assert_all_close(&res.lb_stat, &f64s(&block["lb_stat"]), TOL, "lb_stat");
    assert_all_close(&res.lb_pvalue, &f64s(&block["lb_pvalue"]), TOL, "lb_pvalue");
    assert_all_close(&res.bp_stat, &f64s(&block["bp_stat"]), TOL, "bp_stat");
    assert_all_close(&res.bp_pvalue, &f64s(&block["bp_pvalue"]), TOL, "bp_pvalue");
}

#[test]
fn ljung_box_is_demeaning_invariant() {
    // The ACF demeans internally, so raw and demeaned input must agree.
    let fx = fixture();
    let y = nile(&fx);
    let raw = ljung_box(&y, 10).unwrap();
    let dm = ljung_box(&demeaned(&y), 10).unwrap();
    assert_all_close(&raw.lb_stat, &dm.lb_stat, 1e-10, "lb demeaning");
    assert_all_close(&raw.bp_stat, &dm.bp_stat, 1e-10, "bp demeaning");
}

#[test]
fn arch_lm_matches_statsmodels() {
    let fx = fixture();
    let y = demeaned(&nile(&fx)); // fixture ran het_arch on y - mean
    let block = &fx["arch_lm_4"];
    let res = arch_lm(&y, 4).unwrap();
    assert_close(
        res.statistic,
        block["lm_stat"].as_f64().unwrap(),
        TOL,
        "arch lm stat",
    );
    assert_close(
        res.p_value,
        block["lm_pvalue"].as_f64().unwrap(),
        TOL,
        "arch lm pvalue",
    );
    assert_eq!(res.df, 4);
    assert_eq!(res.nobs, y.len() - 4);
}

#[test]
fn jarque_bera_matches_statsmodels() {
    let fx = fixture();
    let y = nile(&fx);
    let diff: Vec<f64> = y.windows(2).map(|w| w[1] - w[0]).collect();
    let block = &fx["jarque_bera_on_diff"];
    let res = jarque_bera(&diff).unwrap();
    assert_close(res.statistic, block["stat"].as_f64().unwrap(), TOL, "jb stat");
    assert_close(
        res.p_value,
        block["pvalue"].as_f64().unwrap(),
        TOL,
        "jb pvalue",
    );
    assert_close(res.skewness, block["skew"].as_f64().unwrap(), TOL, "jb skew");
    assert_close(
        res.kurtosis,
        block["kurtosis"].as_f64().unwrap(),
        TOL,
        "jb kurtosis",
    );
    assert_eq!(res.n, diff.len());
}
