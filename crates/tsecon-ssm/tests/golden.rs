//! Golden-value tests against statsmodels fixtures (ssm.json): the Nile
//! local level model with exact-diffuse initialization (Durbin & Koopman
//! 2012 fixed parameters), the same model with a missing stretch, and an
//! AR(2)-with-constant SARIMAX log-likelihood under stationary
//! initialization.

mod common;

use common::{as_f64_vec, assert_rel_close, col, load_fixture};
use tsecon_ssm::LinearGaussianSSM;

/// Local level on the Nile data at the DK (2012) parameter values with
/// exact diffuse initialization: log-likelihood, filtered state `a_{t|t}`,
/// filtered covariance `P_{t|t}`, smoothed state, and smoothed covariance
/// all match statsmodels `use_exact_diffuse=True` to 1e-6 relative —
/// including t = 0, which lies inside the diffuse period.
///
/// (Measured agreement on macOS arm64 is <= 1e-11 relative throughout;
/// the asserted 1e-6 is the golden-harness tier for logliks at fixed
/// parameters, kept loose for cross-platform stability.)
#[test]
fn golden_local_level_exact_diffuse() {
    let fx = load_fixture("ssm.json");
    let nile = as_f64_vec(&fx["nile"]);
    let s2_eps = fx["local_level_params"]["sigma2_eps"].as_f64().unwrap();
    let s2_eta = fx["local_level_params"]["sigma2_eta"].as_f64().unwrap();
    let block = &fx["local_level_exact_diffuse"];

    let model = LinearGaussianSSM::local_level(s2_eps, s2_eta).unwrap();
    let y = col(&nile);
    let so = model.smooth(y.as_ref()).unwrap();

    // One diffuse period: P_inf = 1 collapses after the first observation.
    assert_eq!(so.filter.d_diffuse, 1);

    assert_rel_close(
        so.filter.loglik,
        block["loglike"].as_f64().unwrap(),
        1e-6,
        "loglike",
    );

    let filtered = as_f64_vec(&block["filtered_state"]);
    let filtered_cov = as_f64_vec(&block["filtered_state_cov"]);
    let smoothed = as_f64_vec(&block["smoothed_state"]);
    let smoothed_cov = as_f64_vec(&block["smoothed_state_cov"]);
    for t in 0..nile.len() {
        assert_rel_close(
            so.filter.filtered_state[t][0],
            filtered[t],
            1e-6,
            &format!("filtered_state[{t}]"),
        );
        assert_rel_close(
            so.filter.filtered_state_cov[t][(0, 0)],
            filtered_cov[t],
            1e-6,
            &format!("filtered_state_cov[{t}]"),
        );
        assert_rel_close(
            so.smoothed_state[t][0],
            smoothed[t],
            1e-6,
            &format!("smoothed_state[{t}]"),
        );
        assert_rel_close(
            so.smoothed_state_cov[t][(0, 0)],
            smoothed_cov[t],
            1e-6,
            &format!("smoothed_state_cov[{t}]"),
        );
    }
}

/// Same model with observations 20..40 missing (NaN): exact-diffuse
/// log-likelihood and smoothed state match statsmodels to 1e-6 relative.
#[test]
fn golden_local_level_missing_exact_diffuse() {
    let fx = load_fixture("ssm.json");
    let s2_eps = fx["local_level_params"]["sigma2_eps"].as_f64().unwrap();
    let s2_eta = fx["local_level_params"]["sigma2_eta"].as_f64().unwrap();
    let block = &fx["local_level_missing_20_40_exact_diffuse"];
    let y_missing = as_f64_vec(&block["y"]);
    assert!(y_missing[20].is_nan() && y_missing[39].is_nan());
    assert!(!y_missing[19].is_nan() && !y_missing[40].is_nan());

    let model = LinearGaussianSSM::local_level(s2_eps, s2_eta).unwrap();
    let y = col(&y_missing);
    let so = model.smooth(y.as_ref()).unwrap();

    assert_rel_close(
        so.filter.loglik,
        block["loglike"].as_f64().unwrap(),
        1e-6,
        "loglike (missing 20..40)",
    );
    let smoothed = as_f64_vec(&block["smoothed_state"]);
    assert_eq!(smoothed.len(), y_missing.len());
    for (t, expected) in smoothed.iter().enumerate() {
        assert_rel_close(
            so.smoothed_state[t][0],
            *expected,
            1e-6,
            &format!("smoothed_state[{t}] (missing 20..40)"),
        );
    }
}

/// AR(2) with constant at fixed parameters [const, ar1, ar2, sigma2],
/// stationary initialization: log-likelihood matches statsmodels
/// `SARIMAX(order=(2,0,0), trend='c').loglike(params)` to 1e-6 relative.
#[test]
fn golden_ar2_sarimax_loglike() {
    let fx = load_fixture("ssm.json");
    let block = &fx["ar2_sarimax"];
    let y = as_f64_vec(&block["y"]);
    let params = as_f64_vec(&block["params_const_ar1_ar2_sigma2"]);
    let (c, ar1, ar2, sigma2) = (params[0], params[1], params[2], params[3]);

    let model = LinearGaussianSSM::ar(&[ar1, ar2], sigma2, c).unwrap();
    let ym = col(&y);
    let loglik = model.loglike(ym.as_ref()).unwrap();

    assert_rel_close(
        loglik,
        block["loglike"].as_f64().unwrap(),
        1e-6,
        "ar2 loglike",
    );

    // Stationary initialization: no diffuse period at all.
    assert_eq!(model.filter(ym.as_ref()).unwrap().d_diffuse, 0);
}
