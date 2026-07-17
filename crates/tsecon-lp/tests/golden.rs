//! Golden-value tests against `fixtures/lp.json` (statsmodels 0.14.6 for the
//! HAC-OLS block, linearmodels 7.0 for the IV block).
//!
//! The fixture DGP is `y_t = sum_h 0.9^h e_{t-h} + noise` with `e` the
//! observed shock, `x` an endogenous impulse, and `z` an instrument for `x`.
//! Each horizon regresses the outcome on `[shock, const, y_{t-1..t-4}]` (the
//! `n_lag_controls = 4` fixture design). We pin:
//!
//! * `ols_lp`: OLS `beta` (1e-10) and Newey-West HAC `se` (1e-8),
//!   `maxlags = h + 4`, `use_correction = True` — reproduced through
//!   [`tsecon_lp::lp`] with [`tsecon_lp::SeSpec::Hac`].
//! * `iv_lp`: just-identified 2SLS `beta` (1e-8) and linearmodels kernel-HAC
//!   `se` (1e-6), Bartlett `bandwidth = h + 4` — through
//!   [`tsecon_lp::lp_iv`].

use serde_json::Value;
use tsecon_lp::{lp, lp_iv, LpSpec};

fn load() -> Value {
    let path = format!("{}/../../fixtures/lp.json", env!("CARGO_MANIFEST_DIR"));
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

fn assert_close(actual: f64, expected: f64, atol: f64, rtol: f64, ctx: &str) {
    let tol = atol + rtol * expected.abs();
    let err = (actual - expected).abs();
    assert!(
        err <= tol,
        "{ctx}: actual {actual}, expected {expected}, abs err {err:e} > tol {tol:e}"
    );
}

#[test]
fn ols_lp_matches_statsmodels_hac() {
    let fx = load();
    let y = f64s(&fx["y"]);
    let e = f64s(&fx["e"]);
    let n_lag = fx["n_lag_controls"].as_u64().expect("n_lag") as usize;
    let entries = fx["ols_lp"].as_array().expect("ols_lp array");
    let hmax = entries.len() - 1;

    // Default HAC maxlags = h + n_lag_controls reproduces the fixture's
    // horizon-growing window, so a single run covers every horizon.
    let spec = LpSpec::new(hmax, n_lag).with_hac(None);
    let res = lp(&y, &e, spec).expect("lp HAC");

    for (h, entry) in entries.iter().enumerate() {
        let beta = entry["beta"].as_f64().expect("beta");
        let se_hac = entry["se_hac"].as_f64().expect("se_hac");
        let maxlags = entry["maxlags"].as_u64().expect("maxlags") as usize;
        let nobs = entry["nobs"].as_u64().expect("nobs") as usize;

        assert_eq!(maxlags, h + n_lag, "h={h}: fixture maxlags convention");
        assert_eq!(res.nobs_per_h[h], nobs, "h={h}: nobs");
        assert_close(res.irf[h], beta, 0.0, 1e-10, &format!("ols beta h={h}"));
        assert_close(res.se[h], se_hac, 1e-8, 0.0, &format!("ols se_hac h={h}"));
    }
}

#[test]
fn ols_lp_fixed_maxlags_override() {
    // A fixed maxlags override must reproduce statsmodels at that same lag;
    // check h=0 (where h + n_lag = 4 = the fixed value, so it must agree
    // with the default-path golden exactly).
    let fx = load();
    let y = f64s(&fx["y"]);
    let e = f64s(&fx["e"]);
    let spec = LpSpec::new(0, 4).with_hac(Some(4));
    let res = lp(&y, &e, spec).expect("lp HAC fixed");
    let entry = &fx["ols_lp"][0];
    assert_close(
        res.se[0],
        entry["se_hac"].as_f64().unwrap(),
        1e-8,
        0.0,
        "fixed-maxlags se h=0",
    );
}

#[test]
fn iv_lp_matches_linearmodels_kernel() {
    let fx = load();
    let y = f64s(&fx["y"]);
    let x = f64s(&fx["x"]);
    let z = f64s(&fx["z"]);
    let n_lag = fx["n_lag_controls"].as_u64().expect("n_lag") as usize;
    let entries = fx["iv_lp"].as_array().expect("iv_lp array");
    let hmax = entries.len() - 1;

    let spec = LpSpec::new(hmax, n_lag);
    let res = lp_iv(&y, &x, &z, spec).expect("lp_iv");

    for (h, entry) in entries.iter().enumerate() {
        let beta = entry["beta"].as_f64().expect("beta");
        let se_kernel = entry["se_kernel"].as_f64().expect("se_kernel");
        let bandwidth = entry["bandwidth"].as_u64().expect("bandwidth") as usize;
        let nobs = entry["nobs"].as_u64().expect("nobs") as usize;

        assert_eq!(bandwidth, h + n_lag, "h={h}: fixture bandwidth convention");
        assert_eq!(res.nobs_per_h[h], nobs, "h={h}: nobs");
        assert_close(res.irf[h], beta, 1e-8, 0.0, &format!("iv beta h={h}"));
        assert_close(
            res.se[h],
            se_kernel,
            1e-6,
            0.0,
            &format!("iv se_kernel h={h}"),
        );
        // The instrument is strong here (z essentially equals x plus a
        // little noise), so the effective F should be comfortably large.
        assert!(
            res.first_stage_f[h] > 10.0,
            "h={h}: first-stage effective F = {} should exceed the weak-IV \
             rule of thumb",
            res.first_stage_f[h]
        );
    }
}
