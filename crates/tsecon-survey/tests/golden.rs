//! Independent-reference golden tests for tsecon-survey.
//!
//! `fixtures/tsecon-survey.json` is produced offline by
//! `fixtures/generate_survey_fixtures.py`. The CG regression and the
//! Mincer-Zarnowitz efficiency test are checked against *statsmodels*
//! `OLS(...).fit(cov_type="HAC", cov_kwds={"maxlags": L, "use_correction":
//! ...}, use_t=False)` — an independent implementation of the same estimand.
//! The disagreement measures are checked against *numpy* `np.std` /
//! `np.percentile`. The two derived scalars statsmodels does not report — the
//! implied rigidity `beta/(1+beta)` and the IQR `P75 - P25` — are documented
//! closed forms. Matching to ~1e-8 establishes the crate reproduces the
//! reference arithmetic; it does not depend on this crate's own output.

use serde_json::Value;
use tsecon_survey::{cg_regression, cg_series, disagreement, efficiency_test, HacBandwidth};

fn load() -> Value {
    let path = format!(
        "{}/../../fixtures/tsecon-survey.json",
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

fn cols(v: &Value) -> Vec<Vec<f64>> {
    v.as_array().expect("array").iter().map(f64s).collect()
}

fn g(v: &Value) -> f64 {
    v.as_f64().expect("number")
}

const TOL: f64 = 1e-8;

fn close(actual: f64, expected: f64, tol: f64, what: &str) {
    let err = (actual - expected).abs();
    let rel = err / (1.0 + expected.abs());
    assert!(
        err < tol || rel < tol,
        "{what}: actual={actual:.15e} expected={expected:.15e} abs_err={err:.3e}"
    );
}

fn check_cg(node: &Value, label: &str) {
    let errors = f64s(&node["errors"]);
    let revisions = f64s(&node["revisions"]);
    let maxlags = node["maxlags"].as_u64().expect("maxlags") as usize;
    let use_correction = node["use_correction"].as_bool().expect("use_correction");
    let fit = cg_regression(
        &errors,
        &revisions,
        HacBandwidth::Lags(maxlags),
        use_correction,
    )
    .expect("cg_regression");
    close(
        fit.intercept,
        g(&node["intercept"]),
        TOL,
        &format!("{label} intercept"),
    );
    close(fit.slope, g(&node["slope"]), TOL, &format!("{label} slope"));
    close(
        fit.se_intercept,
        g(&node["se_intercept"]),
        TOL,
        &format!("{label} se_intercept"),
    );
    close(
        fit.se_slope,
        g(&node["se_slope"]),
        TOL,
        &format!("{label} se_slope"),
    );
    close(
        fit.t_intercept,
        g(&node["t_intercept"]),
        TOL,
        &format!("{label} t_intercept"),
    );
    close(
        fit.t_slope,
        g(&node["t_slope"]),
        TOL,
        &format!("{label} t_slope"),
    );
    close(
        fit.p_intercept,
        g(&node["p_intercept"]),
        TOL,
        &format!("{label} p_intercept"),
    );
    close(
        fit.p_slope,
        g(&node["p_slope"]),
        TOL,
        &format!("{label} p_slope"),
    );
    close(
        fit.r_squared,
        g(&node["r_squared"]),
        TOL,
        &format!("{label} r_squared"),
    );
    close(
        fit.implied_rigidity,
        g(&node["implied_rigidity"]),
        TOL,
        &format!("{label} implied_rigidity"),
    );
    assert_eq!(fit.nobs, node["nobs"].as_u64().expect("nobs") as usize);
    assert_eq!(fit.maxlags, maxlags);
}

#[test]
fn cg_regression_matches_statsmodels_hac() {
    let fx = load();
    check_cg(&fx["cg"], "cg");
    check_cg(&fx["cg_alt"], "cg_alt");
}

#[test]
fn cg_series_matches_documented_alignment() {
    let fx = load();
    let node = &fx["cg_build"];
    let mean_forecast = f64s(&node["mean_forecast"]);
    let actual = f64s(&node["actual"]);
    let h = node["h"].as_u64().expect("h") as usize;
    let (errors, revisions) = cg_series(&mean_forecast, &actual, h).expect("cg_series");
    let exp_err = f64s(&node["errors"]);
    let exp_rev = f64s(&node["revisions"]);
    assert_eq!(errors.len(), exp_err.len(), "errors length");
    assert_eq!(revisions.len(), exp_rev.len(), "revisions length");
    for (t, (a, b)) in errors.iter().zip(&exp_err).enumerate() {
        close(*a, *b, TOL, &format!("error_{t}"));
    }
    for (t, (a, b)) in revisions.iter().zip(&exp_rev).enumerate() {
        close(*a, *b, TOL, &format!("revision_{t}"));
    }
}

fn check_efficiency(node: &Value, label: &str) {
    let errors = f64s(&node["errors"]);
    let regressors = cols(&node["regressors"]);
    let maxlags = node["maxlags"].as_u64().expect("maxlags") as usize;
    let use_correction = node["use_correction"].as_bool().expect("use_correction");
    let fit = efficiency_test(
        &errors,
        &regressors,
        HacBandwidth::Lags(maxlags),
        use_correction,
    )
    .expect("efficiency_test");

    let params = f64s(&node["params"]);
    let bse = f64s(&node["bse"]);
    let tvalues = f64s(&node["tvalues"]);
    let pvalues = f64s(&node["pvalues"]);
    assert_eq!(fit.params.len(), params.len(), "{label} nparams");
    for i in 0..params.len() {
        close(
            fit.params[i],
            params[i],
            TOL,
            &format!("{label} params[{i}]"),
        );
        close(fit.bse[i], bse[i], TOL, &format!("{label} bse[{i}]"));
        close(
            fit.tvalues[i],
            tvalues[i],
            TOL,
            &format!("{label} tvalues[{i}]"),
        );
        close(
            fit.pvalues[i],
            pvalues[i],
            TOL,
            &format!("{label} pvalues[{i}]"),
        );
    }
    close(
        fit.r_squared,
        g(&node["r_squared"]),
        TOL,
        &format!("{label} r_squared"),
    );
    close(fit.wald, g(&node["wald"]), TOL, &format!("{label} wald"));
    assert_eq!(
        fit.wald_df,
        node["wald_df"].as_u64().expect("wald_df") as usize,
        "{label} wald_df"
    );
    close(
        fit.wald_pvalue,
        g(&node["wald_pvalue"]),
        TOL,
        &format!("{label} wald_pvalue"),
    );
}

#[test]
fn efficiency_test_matches_statsmodels_hac_wald() {
    let fx = load();
    check_efficiency(&fx["efficiency"], "efficiency");
    check_efficiency(&fx["efficiency_multi"], "efficiency_multi");
}

fn check_disagreement(node: &Value, label: &str) {
    let panel = cols(&node["panel"]);
    let ddof = node["ddof"].as_u64().expect("ddof") as usize;
    let d = disagreement(&panel, ddof).expect("disagreement");
    let std = f64s(&node["std"]);
    let p25 = f64s(&node["p25"]);
    let p50 = f64s(&node["p50"]);
    let p75 = f64s(&node["p75"]);
    let iqr = f64s(&node["iqr"]);
    let counts: Vec<usize> = node["counts"]
        .as_array()
        .expect("array")
        .iter()
        .map(|v| v.as_u64().expect("count") as usize)
        .collect();
    assert_eq!(d.std.len(), std.len(), "{label} periods");
    for t in 0..std.len() {
        close(d.std[t], std[t], TOL, &format!("{label} std[{t}]"));
        close(d.p25[t], p25[t], TOL, &format!("{label} p25[{t}]"));
        close(d.p50[t], p50[t], TOL, &format!("{label} p50[{t}]"));
        close(d.p75[t], p75[t], TOL, &format!("{label} p75[{t}]"));
        close(d.iqr[t], iqr[t], TOL, &format!("{label} iqr[{t}]"));
        assert_eq!(d.counts[t], counts[t], "{label} counts[{t}]");
    }
}

#[test]
fn disagreement_matches_numpy() {
    let fx = load();
    check_disagreement(&fx["disagreement_pop"], "pop");
    check_disagreement(&fx["disagreement_sample"], "sample");
    check_disagreement(&fx["disagreement_ragged"], "ragged");
}
