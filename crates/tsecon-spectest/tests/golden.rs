//! Independent-reference golden tests for the specification / diagnostic tests.
//!
//! `fixtures/tsecon-spectest.json` is produced by
//! `fixtures/generate_tsecon-spectest_fixtures.py`:
//!
//! * White, Breusch-Pagan (Koenker studentized), and RESET are pinned to
//!   statsmodels' `het_white`, `het_breuschpagan(robust=True)`, and
//!   `linear_reset(power=3, use_f=True)`;
//! * the Chow statistic is assembled from statsmodels OLS residual sums of
//!   squares with a `scipy.stats.f` p-value;
//! * the CUSUM path, bounds, and `sigma` are a documented Brown-Durbin-Evans
//!   recursion evaluated with plain numpy by refitting each expanding window —
//!   a different code path from this crate's incremental recursion.
//!
//! Each reaches its numbers independently of the tsecon Rust crate, so
//! reproducing them to `~1e-8` is a genuine cross-implementation check.

use serde_json::Value;
use tsecon_spectest::{breusch_pagan_test, chow_test, cusum_test, reset_test, white_test, HetTest};

const TOL: f64 = 1e-8;

fn load() -> Value {
    let path = format!(
        "{}/../../fixtures/tsecon-spectest.json",
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

fn close(actual: f64, expected: f64, what: &str) {
    let err = (actual - expected).abs();
    let rel = err / (1.0 + expected.abs());
    assert!(
        err < TOL || rel < TOL,
        "{what}: actual={actual:.15e} expected={expected:.15e} abs_err={err:.3e} rel={rel:.3e}"
    );
}

fn check_het(got: &HetTest, block: &Value, label: &str) {
    close(
        got.statistic,
        g(&block["statistic"]),
        &format!("{label} statistic"),
    );
    assert_eq!(got.df, u(&block["df"]), "{label} df");
    close(got.pvalue, g(&block["pvalue"]), &format!("{label} pvalue"));
    close(got.fstat, g(&block["fstat"]), &format!("{label} fstat"));
    assert_eq!(got.f_df_num, u(&block["f_df_num"]), "{label} f_df_num");
    assert_eq!(got.f_df_den, u(&block["f_df_den"]), "{label} f_df_den");
    close(
        got.f_pvalue,
        g(&block["f_pvalue"]),
        &format!("{label} f_pvalue"),
    );
}

#[test]
fn white_and_breusch_pagan_match_statsmodels() {
    let fx = load();
    let cases = fx["white_breusch_pagan"].as_array().expect("array");
    assert!(!cases.is_empty());
    for case in cases {
        let name = case["name"].as_str().unwrap_or("?");
        let y = f64s(&case["y"]);
        let x = columns(&case["columns"]);

        let white = white_test(&y, &x).expect("white ok");
        check_het(&white, &case["white"], &format!("white[{name}]"));

        let bp = breusch_pagan_test(&y, &x).expect("bp ok");
        check_het(&bp, &case["breusch_pagan"], &format!("bp[{name}]"));
    }
}

#[test]
fn reset_matches_statsmodels() {
    let fx = load();
    let cases = fx["reset"].as_array().expect("array");
    assert!(!cases.is_empty());
    for case in cases {
        let name = case["name"].as_str().unwrap_or("?");
        let y = f64s(&case["y"]);
        let x = columns(&case["columns"]);
        let got = reset_test(&y, &x, 3).expect("reset ok");
        let block = &case["reset"];
        close(
            got.fstat,
            g(&block["fstat"]),
            &format!("reset[{name}] fstat"),
        );
        assert_eq!(got.df_num, u(&block["df_num"]), "reset[{name}] df_num");
        assert_eq!(got.df_den, u(&block["df_den"]), "reset[{name}] df_den");
        close(
            got.pvalue,
            g(&block["pvalue"]),
            &format!("reset[{name}] pvalue"),
        );
    }
}

#[test]
fn chow_matches_ssr_reference() {
    let fx = load();
    let cases = fx["chow"].as_array().expect("array");
    assert!(!cases.is_empty());
    for case in cases {
        let name = case["name"].as_str().unwrap_or("?");
        let y = f64s(&case["y"]);
        let x = columns(&case["columns"]);
        let split = u(&case["split"]);
        let got = chow_test(&y, &x, split).expect("chow ok");
        let block = &case["chow"];
        close(
            got.fstat,
            g(&block["fstat"]),
            &format!("chow[{name}] fstat"),
        );
        assert_eq!(got.df_num, u(&block["df_num"]), "chow[{name}] df_num");
        assert_eq!(got.df_den, u(&block["df_den"]), "chow[{name}] df_den");
        close(
            got.pvalue,
            g(&block["pvalue"]),
            &format!("chow[{name}] pvalue"),
        );
        close(
            got.ssr_pooled,
            g(&block["ssr_pooled"]),
            &format!("chow[{name}] ssr_pooled"),
        );
        close(got.ssr1, g(&block["ssr1"]), &format!("chow[{name}] ssr1"));
        close(got.ssr2, g(&block["ssr2"]), &format!("chow[{name}] ssr2"));
    }
}

#[test]
fn cusum_matches_documented_formula() {
    let fx = load();
    let cases = fx["cusum"].as_array().expect("array");
    assert!(!cases.is_empty());
    for case in cases {
        let name = case["name"].as_str().unwrap_or("?");
        let y = f64s(&case["y"]);
        let x = columns(&case["columns"]);
        let got = cusum_test(&y, &x).expect("cusum ok");
        let block = &case["cusum"];

        close(
            got.sigma,
            g(&block["sigma"]),
            &format!("cusum[{name}] sigma"),
        );
        close(got.a, g(&block["a"]), &format!("cusum[{name}] a"));

        let w = f64s(&block["recursive_residuals"]);
        let path = f64s(&block["path"]);
        let bu = f64s(&block["bound_upper"]);
        let bl = f64s(&block["bound_lower"]);
        assert_eq!(
            got.recursive_residuals.len(),
            w.len(),
            "cusum[{name}] w len"
        );
        assert_eq!(got.path.len(), path.len(), "cusum[{name}] path len");
        for i in 0..w.len() {
            close(
                got.recursive_residuals[i],
                w[i],
                &format!("cusum[{name}] w[{i}]"),
            );
            close(got.path[i], path[i], &format!("cusum[{name}] path[{i}]"));
            close(
                got.bound_upper[i],
                bu[i],
                &format!("cusum[{name}] bound_upper[{i}]"),
            );
            close(
                got.bound_lower[i],
                bl[i],
                &format!("cusum[{name}] bound_lower[{i}]"),
            );
        }
    }
}
