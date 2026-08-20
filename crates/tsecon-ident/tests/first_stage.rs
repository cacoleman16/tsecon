//! Independent-reference golden tests for the proxy-SVAR first-stage
//! strength diagnostics.
//!
//! `fixtures/proxy_first_stage.json` is produced by
//! `fixtures/generate_proxy_first_stage_fixtures.py`, whose every number
//! comes from statsmodels OLS (classical / HC1 / HAC-Bartlett covariance)
//! and scipy.stats.ncx2 — never from this crate. Reproducing the numbers is
//! a genuine cross-implementation check of
//!
//! * the first-stage regression algebra (beta, se, the three F variants,
//!   reliability) at `rtol = 1e-9`, and
//! * the Montiel Olea-Pflueger critical values and tau bounds — scipy
//!   inverts the noncentral chi-square by its own path, this crate bisects
//!   the exact df-1 closed-form CDF — at `atol = 1e-6`.

use serde_json::Value;
use tsecon_ident::{mop_critical_value, mop_tau_bound, proxy_first_stage, FirstStageVariance};
use tsecon_linalg::faer::Mat;

const RTOL: f64 = 1e-9;
const CV_ATOL: f64 = 1e-6;

fn load() -> Value {
    let path = format!(
        "{}/../../fixtures/proxy_first_stage.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let text = std::fs::read_to_string(path).expect("fixture readable");
    serde_json::from_str(&text).expect("valid JSON")
}

fn proxy_f64s(v: &Value) -> Vec<f64> {
    v.as_array()
        .expect("array")
        .iter()
        .map(|x| {
            if x.is_null() {
                f64::NAN
            } else {
                x.as_f64().expect("number")
            }
        })
        .collect()
}

fn mat(v: &Value) -> Mat<f64> {
    let rows: Vec<Vec<f64>> = v
        .as_array()
        .expect("array")
        .iter()
        .map(|r| {
            r.as_array()
                .expect("row")
                .iter()
                .map(|x| x.as_f64().expect("number"))
                .collect()
        })
        .collect();
    Mat::from_fn(rows.len(), rows[0].len(), |i, j| rows[i][j])
}

fn close(actual: f64, expected: f64, what: &str) {
    let err = (actual - expected).abs();
    assert!(
        err <= RTOL * (1.0 + expected.abs()),
        "{what}: got {actual}, expected {expected} (err {err:.3e})"
    );
}

#[test]
fn golden_first_stage_regression_algebra() {
    let fx = load();
    for case in fx["cases"].as_array().expect("cases") {
        let name = case["name"].as_str().expect("name");
        let u = mat(&case["u"]);
        let proxy = proxy_f64s(&case["proxy"]);
        let norm_var = case["norm_var"].as_u64().expect("norm_var") as usize;
        let variance = match case["hac_lags"].as_u64() {
            Some(l) => FirstStageVariance::HacBartlett { lags: l as usize },
            None => FirstStageVariance::Hc1,
        };
        let d = proxy_first_stage(u.as_ref(), &proxy, norm_var, variance)
            .unwrap_or_else(|e| panic!("[{name}] proxy_first_stage failed: {e}"));
        let e = &case["expected"];
        close(d.beta, e["beta"].as_f64().unwrap(), &format!("{name}.beta"));
        close(d.se, e["se"].as_f64().unwrap(), &format!("{name}.se"));
        close(
            d.effective_f,
            e["effective_f"].as_f64().unwrap(),
            &format!("{name}.effective_f"),
        );
        close(
            d.f_classical,
            e["f_classical"].as_f64().unwrap(),
            &format!("{name}.f_classical"),
        );
        close(
            d.f_hc1,
            e["f_hc1"].as_f64().unwrap(),
            &format!("{name}.f_hc1"),
        );
        close(
            d.reliability,
            e["reliability"].as_f64().unwrap(),
            &format!("{name}.reliability"),
        );
        assert_eq!(
            d.n_proxy,
            e["n_proxy"].as_u64().unwrap() as usize,
            "{name}.n_proxy"
        );
        // The fixture's tau bound (scipy brentq) against the crate's bisection.
        match e["tau_bound"].as_f64() {
            Some(tb) => {
                let err = (d.tau_bound - tb).abs();
                assert!(
                    err <= 1e-6 * (1.0 + tb.abs()),
                    "{name}.tau_bound: got {}, expected {tb}",
                    d.tau_bound
                );
            }
            None => assert!(
                d.tau_bound.is_infinite(),
                "{name}.tau_bound should be +inf, got {}",
                d.tau_bound
            ),
        }
        // Stamped flags are consistent with the stamped thresholds.
        assert_eq!(d.weak_mop_tau10, d.effective_f <= d.mop_cv_tau10, "{name}");
        assert_eq!(d.weak_folklore, d.effective_f < 10.0, "{name}");
    }
}

#[test]
fn golden_mop_critical_values_match_scipy() {
    let fx = load();
    for row in fx["critical_values"].as_array().expect("cvs") {
        let tau = row["tau"].as_f64().unwrap();
        let alpha = row["alpha"].as_f64().unwrap();
        let cv = row["cv"].as_f64().unwrap();
        let got = mop_critical_value(tau, alpha).expect("cv computes");
        assert!(
            (got - cv).abs() <= CV_ATOL,
            "cv(tau={tau}, alpha={alpha}): got {got}, scipy {cv}"
        );
    }
}

#[test]
fn golden_mop_tau_bounds_match_scipy() {
    let fx = load();
    for row in fx["tau_bounds"].as_array().expect("tau_bounds") {
        let f = row["f"].as_f64().unwrap();
        let alpha = row["alpha"].as_f64().unwrap();
        let got = mop_tau_bound(f, alpha).expect("tau bound computes");
        match row["tau"].as_f64() {
            Some(tau) => assert!(
                (got - tau).abs() <= 1e-6 * (1.0 + tau),
                "tau_bound(F={f}): got {got}, scipy {tau}"
            ),
            None => assert!(
                got.is_infinite(),
                "tau_bound(F={f}) should be +inf, got {got}"
            ),
        }
    }
}
