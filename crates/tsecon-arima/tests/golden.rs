//! Golden-value tests against the statsmodels fixture
//! (`fixtures/arima.json`): fixed-parameter exact log-likelihoods for
//! ARMA(1,1) and ARIMA(1,1,1) with simple differencing, and the full
//! exact-MLE fit of ARMA(1,1)+constant on the Nile data, including
//! information criteria and 12-step forecasts at the fixture's fitted
//! parameters.

mod common;

use common::{as_vec, assert_rel_close, load_fixture};
use tsecon_arima::ArimaSpec;

fn nile() -> Vec<f64> {
    as_vec(&load_fixture("diagnostics.json")["nile"])
}

/// ARMA(1,1) exact log-likelihood at fixed `[phi, theta, sigma2]` on the
/// demeaned simulated series matches statsmodels
/// `SARIMAX(y - 2, order=(1, 0, 1)).loglike` to 1e-8 relative.
#[test]
fn golden_arma11_loglike_fixed() {
    let fx = load_fixture("arima.json");
    let block = &fx["arma11"];
    let y: Vec<f64> = as_vec(&block["y"]).iter().map(|v| v - 2.0).collect();
    let params = as_vec(&block["fixed_params_phi_theta_sigma2"]);

    let spec = ArimaSpec::new(1, 0, 1).unwrap();
    let ll = spec.loglike(&y, &params).unwrap();
    assert_rel_close(
        ll,
        block["loglike_fixed_demeaned"].as_f64().unwrap(),
        1e-8,
        "arma11 loglike_fixed_demeaned",
    );
}

/// ARIMA(1,1,1) with simple differencing on the Nile: the exact
/// log-likelihood at fixed parameters matches statsmodels
/// `SARIMAX(order=(1, 1, 1), simple_differencing=True).loglike` to 1e-8
/// relative (one observation is lost to differencing).
#[test]
fn golden_nile_arima111_simple_diff_loglike_fixed() {
    let fx = load_fixture("arima.json");
    let block = &fx["nile_arima111_simple_diff"];
    let y = nile();
    let params = as_vec(&block["fixed_params_phi_theta_sigma2"]);

    let spec = ArimaSpec::new(1, 1, 1).unwrap();
    let ll = spec.loglike(&y, &params).unwrap();
    assert_rel_close(
        ll,
        block["loglike_fixed"].as_f64().unwrap(),
        1e-8,
        "nile arima(1,1,1) loglike_fixed",
    );
}

/// Exact MLE of ARMA(1,1)+constant on the Nile: the fit must **match or
/// beat** the fixture's log-likelihood, and its parameters must agree
/// with the independently cross-verified maximizer.
///
/// The fixture's `params`/`loglike` (ll = -638.1168 at `[314.84,
/// 0.6588, -0.2480, 20480.5]`) pin statsmodels' *default-fit stopping
/// point*, which is not a stationary point of the likelihood: the
/// numerical gradient of statsmodels' own `loglike` there is O(1e-2) in
/// the AR coordinate, and `scipy.optimize.minimize(method='Nelder-Mead',
/// xatol=fatol=1e-10)` started *at* those parameters walks away to
///
/// ```text
/// [127.946195, 0.861032973, -0.517678814, 19891.6944],
/// loglike = -637.0387845333154
/// ```
///
/// (scipy 1.17.1 / statsmodels 0.14.6 — the fixture's own versions).
/// This crate's optimizer reaches the same maximizer, so the parameter
/// parity gate is applied against that cross-verified optimum, at 1e-4
/// relative — far tighter than the nominal 1e-2 — while the fixture's
/// stall point remains the match-or-beat floor for the log-likelihood.
#[test]
fn golden_nile_arma11c_fit() {
    let fx = load_fixture("arima.json");
    let block = &fx["nile_arma11c_fit"];
    let y = nile();
    let sm_stall = as_vec(&block["params_const_phi_theta_sigma2"]);
    let ll_fix = block["loglike"].as_f64().unwrap();
    // Cross-verified maximizer (see the doc comment for provenance).
    let optimum = [127.946195, 0.861032973, -0.517678814, 19891.6944];
    let ll_optimum = -637.0387845333154;

    let spec = ArimaSpec::new(1, 0, 1).unwrap().with_constant(true);

    // --- Default fit: match or beat the statsmodels loglik. ---
    let res = spec.fit(&y).unwrap();
    assert!(res.converged, "MLE did not converge");
    assert_eq!(res.nobs, 100);
    assert_eq!(res.k_params, 4);
    assert_eq!(
        res.param_names(),
        &["const", "ar.L1", "ma.L1", "sigma2"],
        "param names"
    );
    assert!(
        res.loglik >= ll_fix - 1e-5 * ll_fix.abs(),
        "loglik {} worse than the fixture floor {ll_fix}",
        res.loglik
    );

    // --- Parameters agree with the cross-verified maximizer. ---
    assert_rel_close(res.loglik, ll_optimum, 1e-8, "loglik (true optimum)");
    for (i, name) in ["const", "ar.L1", "ma.L1", "sigma2"].iter().enumerate() {
        assert_rel_close(res.params()[i], optimum[i], 1e-4, name);
    }
    assert_rel_close(res.constant().unwrap(), optimum[0], 1e-4, "constant()");
    assert_rel_close(res.ar()[0], optimum[1], 1e-4, "ar()");
    assert_rel_close(res.ma()[0], optimum[2], 1e-4, "ma()");
    assert_rel_close(res.sigma2(), optimum[3], 1e-4, "sigma2()");

    // --- AIC/BIC exactly consistent with the achieved loglik, k = 4,
    //     nobs = 100 (statsmodels conventions: sigma2 counted in k). ---
    assert_rel_close(res.aic, -2.0 * res.loglik + 8.0, 1e-12, "aic identity");
    assert_rel_close(
        res.bic,
        -2.0 * res.loglik + 4.0 * 100f64.ln(),
        1e-12,
        "bic identity",
    );

    // --- Started at the fixture's stall point, the optimizer continues
    //     to the same maximizer (it is not a local optimum to stay at). ---
    let res_sm = spec.fit_with_start(&y, &sm_stall).unwrap();
    assert!(res_sm.converged, "refit from fixture params did not converge");
    assert_rel_close(
        res_sm.loglik,
        ll_optimum,
        1e-8,
        "loglik (refit from fixture params)",
    );
}

/// Twelve-step forecast mean and standard errors evaluated at the
/// fixture's fitted parameters match statsmodels
/// `get_forecast(12)` to 1e-6 relative.
#[test]
fn golden_nile_arma11c_forecast_at_fixture_params() {
    let fx = load_fixture("arima.json");
    let block = &fx["nile_arma11c_fit"];
    let y = nile();
    let params = as_vec(&block["params_const_phi_theta_sigma2"]);
    let mean_exp = as_vec(&block["forecast_mean_12"]);
    let se_exp = as_vec(&block["forecast_se_12"]);

    let spec = ArimaSpec::new(1, 0, 1).unwrap().with_constant(true);
    let res = spec.at_params(&y, &params).unwrap();
    let fc = res.forecast(12).unwrap();

    assert_eq!(fc.mean.len(), 12);
    assert_eq!(fc.se.len(), 12);
    for h in 0..12 {
        assert_rel_close(fc.mean[h], mean_exp[h], 1e-6, &format!("forecast mean[{h}]"));
        assert_rel_close(fc.se[h], se_exp[h], 1e-6, &format!("forecast se[{h}]"));
    }

    // The fixed-parameter results object reports the fixture loglik too.
    assert_rel_close(
        res.loglik,
        block["loglike"].as_f64().unwrap(),
        1e-8,
        "loglik at fixture params",
    );
}
