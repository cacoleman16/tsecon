//! Golden-value tests for the Engle-Granger two-step cointegration test
//! against `fixtures/engle_granger.json`, generated from
//! `statsmodels.tsa.stattools.coint` (statsmodels 0.14.6) by
//! `fixtures/generate_engle_granger_fixtures.py`.
//!
//! `coint` returns `(stat, pvalue, crit)`, and all three are pinned here
//! across seven systems (six simulated, one real: log consumption on log
//! GDP and log investment from statsmodels' `macrodata`), three
//! deterministic specifications, and every lag rule (`aic` / `bic` /
//! `t-stat` / fixed). The step-1 cointegrating coefficients are pinned
//! too.
//!
//! Two behaviours deliberately go *beyond* statsmodels and are pinned as
//! such: `N > 6`, where `coint` raises `IndexError` and this crate returns
//! a NaN p-value next to the (published) 2010 critical values, and the
//! near-perfect step-1 fit, where `coint` warns and returns `-inf` with
//! p = 0 and this crate returns a named error.

mod common;

use serde_json::Value;
use tsecon_coint::{engle_granger, CointError, EngleGrangerTrend};
use tsecon_diag::{mackinnon_coint_crit, AdfLagSelection, PoTrend};
use tsecon_linalg::faer::Mat;

use common::{as_endog, load_fixture, num};

// ------------------------------------------------------------- helpers

fn trend_of(code: &str) -> EngleGrangerTrend {
    match code {
        "n" => EngleGrangerTrend::None,
        "c" => EngleGrangerTrend::Constant,
        "ct" => EngleGrangerTrend::ConstantTrend,
        other => panic!("unknown trend {other:?}"),
    }
}

fn po_trend_of(code: &str) -> PoTrend {
    match code {
        "n" => PoTrend::None,
        "c" => PoTrend::Constant,
        "ct" => PoTrend::ConstantTrend,
        other => panic!("unknown trend {other:?}"),
    }
}

/// statsmodels `autolag` / `maxlag` pair as this crate's lag rule:
/// `autolag = "aic"` with `maxlag = None` is the `coint` default.
fn lag_rule(case: &Value) -> AdfLagSelection {
    let maxlag = case["maxlag"].as_u64().map(|m| m as usize);
    match case["autolag"].as_str() {
        Some("aic") => AdfLagSelection::Aic(maxlag),
        Some("bic") => AdfLagSelection::Bic(maxlag),
        Some("t-stat") => AdfLagSelection::TStat(maxlag),
        Some(other) => panic!("unknown autolag {other:?}"),
        None => AdfLagSelection::Fixed(maxlag.expect("autolag = null needs a maxlag")),
    }
}

/// Combined absolute+relative comparison (numpy `allclose` shape). The
/// p-value surface reaches ~1e-27 in the deep tail on these cases and
/// `tsecon-stats`' normal CDF tracks scipy's to 3.3e-12 relative even
/// there, so the relative arm carries the test and the absolute floor only
/// guards an exactly-saturated 0 or 1.
fn assert_pclose(actual: f64, expected: f64, ctx: &str) {
    let tol = 1e-14 + 1e-9 * expected.abs();
    assert!(
        (actual - expected).abs() <= tol,
        "{ctx}: actual {actual}, expected {expected}, |diff| {:e} > {tol:e}",
        (actual - expected).abs()
    );
}

/// Relative comparison with an absolute floor at `tol` (values here are
/// O(1) statistics and critical values, never near zero).
fn assert_close(actual: f64, expected: f64, tol: f64, ctx: &str) {
    let scale = expected.abs().max(1.0);
    let rel = (actual - expected).abs() / scale;
    assert!(
        rel <= tol,
        "{ctx}: actual {actual}, expected {expected}, rel {rel:e} > {tol:e}"
    );
}

fn system(fx: &Value, name: &str) -> Mat<f64> {
    as_endog(&fx["systems"][name]["data"])
}

// --------------------------------------------------------------- golden

/// Every `coint` case: statistic, MacKinnon cointegration p-value, and the
/// MacKinnon (2010) critical values, plus the residual-ADF bookkeeping
/// (`used_lag`, `nobs`) that shows the lag rule matched.
#[test]
fn golden_engle_granger_matches_statsmodels_coint() {
    let fx = load_fixture("engle_granger.json");
    let cases = fx["cases"].as_array().expect("cases array");
    assert!(cases.len() >= 29, "fixture lost cases: {}", cases.len());

    for case in cases {
        let name = case["system"].as_str().expect("system");
        let trend_code = case["trend"].as_str().expect("trend");
        let endog = system(&fx, name);
        let ctx = format!(
            "{name}/{trend_code}/autolag={}/maxlag={}",
            case["autolag"], case["maxlag"]
        );

        let res = engle_granger(endog.as_ref(), trend_of(trend_code), lag_rule(case))
            .unwrap_or_else(|e| panic!("{ctx}: engle_granger failed: {e}"));

        // Shape of the problem.
        assert_eq!(
            res.n_vars,
            case["n_vars"].as_u64().expect("n_vars") as usize,
            "{ctx}: n_vars"
        );
        assert_eq!(
            res.nobs,
            case["nobs"].as_u64().expect("nobs") as usize,
            "{ctx}: nobs"
        );

        // Step 2 bookkeeping: the lag rule must pick statsmodels' lag.
        assert_eq!(
            res.resid_adf.used_lag,
            case["used_lag"].as_u64().expect("used_lag") as usize,
            "{ctx}: used_lag"
        );
        assert_eq!(
            res.resid_adf.nobs,
            case["adf_nobs"].as_u64().expect("adf_nobs") as usize,
            "{ctx}: residual ADF nobs"
        );

        // The three numbers `coint` returns.
        assert_close(res.stat, num(&case["stat"]), 1e-10, &format!("{ctx}: stat"));
        assert_eq!(res.stat, res.statistic(), "{ctx}: statistic() alias");
        match case["pvalue"].as_f64() {
            Some(p) => assert_pclose(res.p_value, p, &format!("{ctx}: pvalue")),
            // N > 6: statsmodels raises IndexError; this crate returns NaN.
            None => assert!(
                res.p_value.is_nan(),
                "{ctx}: p-value should be NaN for N = {}, got {}",
                res.n_vars,
                res.p_value
            ),
        }
        match (&res.crit, case["crit"].as_object()) {
            (Some(c), Some(e)) => {
                assert_close(c.pct1, num(&e["1%"]), 1e-11, &format!("{ctx}: crit 1%"));
                assert_close(c.pct5, num(&e["5%"]), 1e-11, &format!("{ctx}: crit 5%"));
                assert_close(c.pct10, num(&e["10%"]), 1e-11, &format!("{ctx}: crit 10%"));
            }
            // trend = "n": no published 2010 no-constant surface.
            (None, None) => assert_eq!(trend_code, "n", "{ctx}: unexpected missing crit"),
            (a, e) => panic!("{ctx}: crit availability mismatch ({a:?} vs {e:?})"),
        }

        // Step 1 coefficients, re-ordered by the generator into this
        // crate's [deterministics..., series...] design order.
        let expected: Vec<f64> = case["coint_coefs"]
            .as_array()
            .expect("coint_coefs")
            .iter()
            .map(num)
            .collect();
        assert_eq!(
            res.coint_coefs.len(),
            expected.len(),
            "{ctx}: coefficient count"
        );
        for (i, (&a, &e)) in res.coint_coefs.iter().zip(&expected).enumerate() {
            assert_close(a, e, 1e-10, &format!("{ctx}: coint_coefs[{i}]"));
        }
        assert_eq!(res.resid.len(), res.nobs, "{ctx}: residual length");
        assert_eq!(res.trend, trend_of(trend_code), "{ctx}: trend echoed back");

        // The exact quantity statsmodels' collinearity guard reads --
        // `OLS.rsquared`, centered when the design carries an intercept and
        // uncentered when it does not -- recomputed from the residuals, so
        // the guard is pinned and not just the statistic. Getting the
        // centering backwards would make the guard fire on ordinary
        // large-mean level data (`trend = "n"` on logs).
        let ssr: f64 = res.resid.iter().map(|e| e * e).sum();
        let y0: Vec<f64> = (0..res.nobs).map(|i| endog[(i, 0)]).collect();
        let tss: f64 = if trend_code == "n" {
            y0.iter().map(|v| v * v).sum()
        } else {
            let mean = y0.iter().sum::<f64>() / y0.len() as f64;
            y0.iter().map(|v| (v - mean).powi(2)).sum()
        };
        assert_close(
            1.0 - ssr / tss,
            num(&case["rsquared"]),
            1e-9,
            &format!("{ctx}: step-1 rsquared (the collinearity guard)"),
        );
    }
}

/// The MacKinnon (2010) surfaces themselves, at the exact `nobs = T - 1`
/// the golden cases use, so a critical-value mismatch localizes to the
/// surface rather than to the test statistic.
#[test]
fn golden_coint_critical_value_surfaces() {
    let fx = load_fixture("engle_granger.json");
    for code in ["c", "ct"] {
        let trend = po_trend_of(code);
        let block = fx["crit_map"][code].as_object().expect("crit_map block");
        for (key, entry) in block {
            let n_vars = entry["n_vars"].as_u64().expect("n_vars") as usize;
            let nobs = entry["nobs"].as_u64().expect("nobs") as usize;
            let expected: Vec<f64> = entry["crit"]
                .as_array()
                .expect("crit")
                .iter()
                .map(num)
                .collect();
            let cv = mackinnon_coint_crit(trend, n_vars, nobs).expect("surface published");
            assert_close(cv.pct1, expected[0], 1e-11, &format!("{code}/{key} 1%"));
            assert_close(cv.pct5, expected[1], 1e-11, &format!("{code}/{key} 5%"));
            assert_close(cv.pct10, expected[2], 1e-11, &format!("{code}/{key} 10%"));
        }
    }
}

// ------------------------------------------------------------ behaviour

/// A cointegrated system rejects the no-cointegration null and a
/// non-cointegrated one does not — the decision the p-value and the 5%
/// critical value exist to support, and they must agree with each other.
#[test]
fn decision_agrees_between_pvalue_and_critical_value() {
    let fx = load_fixture("engle_granger.json");
    for (name, cointegrated) in [("co_k2", true), ("no_k2", false)] {
        let endog = system(&fx, name);
        let res = engle_granger(
            endog.as_ref(),
            EngleGrangerTrend::ConstantTrend,
            AdfLagSelection::Aic(None),
        )
        .unwrap();
        let crit = res.crit.expect("ct surface is published");
        let reject_by_p = res.p_value < 0.05;
        let reject_by_crit = res.stat < crit.pct5;
        assert_eq!(
            reject_by_p, reject_by_crit,
            "{name}: p = {} and stat {} vs 5% {} disagree",
            res.p_value, res.stat, crit.pct5
        );
        assert_eq!(
            reject_by_p, cointegrated,
            "{name}: expected reject = {cointegrated}, got p = {}",
            res.p_value
        );
    }
}

/// `trend = "n"` has a p-value (the 1994 surfaces cover it) but no 2010
/// critical values — exactly what `statsmodels.coint` reports.
#[test]
fn no_constant_case_has_pvalue_but_no_critical_values() {
    let fx = load_fixture("engle_granger.json");
    let endog = system(&fx, "co_k3");
    let res = engle_granger(
        endog.as_ref(),
        EngleGrangerTrend::None,
        AdfLagSelection::Aic(None),
    )
    .unwrap();
    assert!(res.p_value.is_finite() && (0.0..=1.0).contains(&res.p_value));
    assert!(res.crit.is_none(), "no 2010 no-constant surface exists");
}

/// Above the 1994 tables (`N > 6`) the p-value is NaN but the 2010
/// critical values are still published up to `N = 12`, so the test remains
/// usable — where `statsmodels.coint` simply raises `IndexError`.
#[test]
fn seven_series_keeps_critical_values_without_a_pvalue() {
    let fx = load_fixture("engle_granger.json");
    let endog = system(&fx, "co_k7");
    let res = engle_granger(
        endog.as_ref(),
        EngleGrangerTrend::Constant,
        AdfLagSelection::Aic(None),
    )
    .unwrap();
    assert_eq!(res.n_vars, 7);
    assert!(res.p_value.is_nan(), "N = 7 is past the 1994 tables");
    let crit = res.crit.expect("2010 surfaces reach N = 12");
    assert!(crit.pct1 < crit.pct5 && crit.pct5 < crit.pct10);
}

/// An (almost) perfectly collinear step-1 regression is named rather than
/// reported as overwhelming evidence: statsmodels returns `-inf` with
/// p = 0 here, which reads exactly like a decisive rejection.
#[test]
fn perfect_fit_is_reported_as_an_error_not_as_evidence() {
    let fx = load_fixture("engle_granger.json");
    let base = system(&fx, "co_k2");
    let n = base.nrows();
    // Second series an exact affine image of the first: R^2 = 1.
    let endog = Mat::from_fn(n, 2, |i, j| {
        if j == 0 {
            base[(i, 0)]
        } else {
            2.0 * base[(i, 0)] + 3.0
        }
    });
    let err = engle_granger(
        endog.as_ref(),
        EngleGrangerTrend::Constant,
        AdfLagSelection::Aic(None),
    )
    .unwrap_err();
    assert!(
        matches!(err, CointError::Singular { .. }),
        "expected a collinearity error, got {err:?}"
    );
    assert!(
        err.to_string().contains("collinear"),
        "the message should name the problem: {err}"
    );
}

/// Fewer than two series is a dimension error, not a panic.
#[test]
fn single_series_is_rejected() {
    let m = Mat::<f64>::from_fn(20, 1, |i, _| i as f64);
    let err = engle_granger(
        m.as_ref(),
        EngleGrangerTrend::Constant,
        AdfLagSelection::Aic(None),
    )
    .unwrap_err();
    assert!(matches!(err, CointError::Dimension { .. }), "{err:?}");
}
