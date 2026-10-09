//! Independent-reference golden tests for the serial-correlation tests.
//!
//! `fixtures/serial_corr.json` is produced by
//! `fixtures/generate_serial_corr_fixtures.py` (it never imports tsecon):
//!
//! * Breusch-Godfrey — statsmodels `acorr_breusch_godfrey` on an OLS fit
//!   (zero-padded lagged residuals, default `nlags = min(10, T // 5)`, a
//!   constant added to the auxiliary design when the model has none): the LM
//!   statistic and p-value, the F-form and its p-value, the degrees of
//!   freedom and the added-constant flag;
//! * the Durbin-Watson statistic — statsmodels `durbin_watson`;
//! * the exact Durbin-Watson p-values — the documented Imhof (1961) formula
//!   on the eigenvalues of `Z'AZ` from numpy's complete QR and LAPACK
//!   `eigvalsh`, integrated by QUADPACK (a different eigen route and a
//!   different quadrature from the crate's).
//!
//! Cases: seeded regressions with AR(1) and iid errors, constant-only and
//! multi-regressor designs, a design without a constant, an implicit
//! constant, a short (T = 15) and a long (T = 500) sample, and three US
//! macro regressions built from statsmodels' macrodata by transformation.

use serde_json::Value;
use tsecon_spectest::{
    breusch_godfrey_default_nlags, breusch_godfrey_test, durbin_watson_statistic,
    durbin_watson_test, DwAlternative,
};

const TOL: f64 = 1e-10;

fn load() -> Value {
    let path = format!(
        "{}/../../fixtures/serial_corr.json",
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

fn u(v: &Value) -> usize {
    v.as_u64().expect("uint") as usize
}

/// Absolute-or-relative agreement at `TOL`; returns the deviation used.
fn close(actual: f64, expected: f64, what: &str) -> f64 {
    let err = (actual - expected).abs();
    let rel = err / expected.abs().max(1.0);
    assert!(
        err < TOL || rel < TOL,
        "{what}: actual={actual:.15e} expected={expected:.15e} abs_err={err:.3e} rel={rel:.3e}"
    );
    rel.min(err)
}

#[test]
fn breusch_godfrey_matches_statsmodels() {
    let fx = load();
    let cases = fx["cases"].as_array().expect("cases");
    assert!(cases.len() >= 10);
    let mut worst: f64 = 0.0;
    let mut n_checked = 0;
    for case in cases {
        let name = case["name"].as_str().unwrap_or("?");
        let y = f64s(&case["y"]);
        let x = columns(&case["columns"]);
        for block in case["breusch_godfrey"].as_array().expect("bg") {
            let arg = block["nlags_arg"].as_u64().map(|v| v as usize);
            let got = breusch_godfrey_test(&y, &x, arg).expect("bg ok");
            let label = format!("bg[{name}, nlags={arg:?}]");
            assert_eq!(got.nlags, u(&block["nlags"]), "{label} nlags");
            assert_eq!(got.df, u(&block["df"]), "{label} df");
            assert_eq!(got.f_df_num, u(&block["f_df_num"]), "{label} f_df_num");
            assert_eq!(got.f_df_den, u(&block["f_df_den"]), "{label} f_df_den");
            assert_eq!(got.nobs, y.len(), "{label} nobs");
            assert_eq!(
                got.constant_added,
                block["constant_added"].as_bool().expect("bool"),
                "{label} constant_added"
            );
            for (a, key) in [
                (got.statistic, "statistic"),
                (got.pvalue, "pvalue"),
                (got.fstat, "fstat"),
                (got.f_pvalue, "f_pvalue"),
            ] {
                worst = worst.max(close(a, g(&block[key]), &format!("{label} {key}")));
            }
            n_checked += 1;
        }
    }
    assert!(n_checked >= 25, "only {n_checked} BG cells checked");
    eprintln!("breusch_godfrey: {n_checked} cells, worst deviation {worst:.2e}");
}

#[test]
fn durbin_watson_statistic_and_exact_pvalues_match_reference() {
    let fx = load();
    let cases = fx["cases"].as_array().expect("cases");
    let mut worst_stat: f64 = 0.0;
    let mut worst_p: f64 = 0.0;
    for case in cases {
        let name = case["name"].as_str().unwrap_or("?");
        let y = f64s(&case["y"]);
        let x = columns(&case["columns"]);
        let block = &case["durbin_watson"];
        for (alt, key) in [
            (DwAlternative::Positive, "pvalue_positive"),
            (DwAlternative::Negative, "pvalue_negative"),
            (DwAlternative::TwoSided, "pvalue_two_sided"),
        ] {
            let got = durbin_watson_test(&y, &x, alt).expect("dw ok");
            let label = format!("dw[{name}, {}]", alt.name());
            worst_stat = worst_stat.max(close(
                got.statistic,
                g(&block["statistic"]),
                &format!("{label} statistic"),
            ));
            worst_p = worst_p.max(close(got.pvalue, g(&block[key]), &format!("{label} {key}")));
            assert_eq!(got.alternative, alt);
            assert_eq!(got.nobs, u(&block["nobs"]), "{label} nobs");
            assert_eq!(got.df_resid, u(&block["df_resid"]), "{label} df_resid");
            // The two one-sided p-values are complementary.
            assert!((got.pvalue_positive + got.pvalue_negative - 1.0).abs() < 1e-12);
        }
    }
    eprintln!("durbin_watson: worst statistic deviation {worst_stat:.2e}, worst p-value deviation {worst_p:.2e}");
}

#[test]
fn durbin_watson_statistic_on_residuals_matches_the_regression_test() {
    // The residual-level helper is the same statistic the test reports.
    let fx = load();
    let case = &fx["cases"][0];
    let y = f64s(&case["y"]);
    let x = columns(&case["columns"]);
    let fit = tsecon_hac::ols(&y, &x).expect("ols");
    let d = durbin_watson_statistic(&fit.residuals).expect("dw");
    close(d, g(&case["durbin_watson"]["statistic"]), "dw helper");
}

#[test]
fn default_order_is_the_statsmodels_rule() {
    for (n, p) in [(4, 0), (5, 1), (15, 3), (49, 9), (50, 10), (500, 10)] {
        assert_eq!(breusch_godfrey_default_nlags(n), p, "n = {n}");
    }
}
