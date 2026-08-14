//! Golden-value tests against the statsmodels SARIMAX seasonal fixture
//! (`fixtures/sarima.json`): fixed-parameter exact log-likelihoods for
//! the airline model ARIMA(0,1,1)(0,1,1)_12, a quarterly SAR with
//! constant, and the full mixed SARIMA(1,1,1)(1,1,1)_4 — all with
//! simple differencing — plus full-fit match-or-beat gates, approx
//! standard errors at the recorded parameters, and levels forecasts
//! against the statsmodels levels state-space form.

mod common;

use common::{as_vec, assert_rel_close, load_fixture};
use tsecon_arima::ArimaSpec;

fn log_airline() -> Vec<f64> {
    // The fixture flags the airline case as log passengers; the raw
    // series is Box-Jenkins Series G, embedded in the generator. Rebuild
    // the log series from the quarterly fixture's sibling — the fixture
    // stores only the flag, so the canonical numbers live here too, and
    // the generator's copy arbitrates.
    const AIRLINE: [f64; 144] = [
        112., 118., 132., 129., 121., 135., 148., 148., 136., 119., 104., 118., 115., 126., 141.,
        135., 125., 149., 170., 170., 158., 133., 114., 140., 145., 150., 178., 163., 172., 178.,
        199., 199., 184., 162., 146., 166., 171., 180., 193., 181., 183., 218., 230., 242., 209.,
        191., 172., 194., 196., 196., 236., 235., 229., 243., 264., 272., 237., 211., 180., 201.,
        204., 188., 235., 227., 234., 264., 302., 293., 259., 229., 203., 229., 242., 233., 267.,
        269., 270., 315., 364., 347., 312., 274., 237., 278., 284., 277., 317., 313., 318., 374.,
        413., 405., 355., 306., 271., 306., 315., 301., 356., 348., 355., 422., 465., 467., 404.,
        347., 305., 336., 340., 318., 362., 348., 363., 435., 491., 505., 404., 359., 310., 337.,
        360., 342., 406., 396., 420., 472., 548., 559., 463., 407., 362., 405., 417., 391., 419.,
        461., 472., 535., 622., 606., 508., 461., 390., 432.,
    ];
    AIRLINE.iter().map(|v| v.ln()).collect()
}

fn airline_spec() -> ArimaSpec {
    ArimaSpec::new(0, 1, 1)
        .unwrap()
        .seasonal(0, 1, 1, 12)
        .unwrap()
}

/// The airline model's exact log-likelihood at fixed parameters matches
/// statsmodels `SARIMAX(order=(0,1,1), seasonal_order=(0,1,1,12),
/// simple_differencing=True).loglike` to 1e-8 relative (13 observations
/// are lost to differencing).
#[test]
fn golden_airline_loglike_fixed() {
    let fx = load_fixture("sarima.json");
    let block = &fx["airline_011_011_12"];
    let y = log_airline();
    let params = as_vec(&block["fixed_params_theta_Theta_sigma2"]);
    let ll = airline_spec().loglike(&y, &params).unwrap();
    assert_rel_close(
        ll,
        block["loglike_fixed_simple_diff"].as_f64().unwrap(),
        1e-8,
        "airline loglike_fixed",
    );
}

/// Exact MLE on the airline model must match or beat the statsmodels
/// fit's log-likelihood, and land on its parameters (the airline
/// likelihood is well-behaved; 5e-3 relative absorbs the two
/// optimizers' stopping rules).
#[test]
fn golden_airline_fit() {
    let fx = load_fixture("sarima.json");
    let block = &fx["airline_011_011_12"];
    let y = log_airline();
    let res = airline_spec().fit(&y).unwrap();
    let ll_ref = block["fit_loglike"].as_f64().unwrap();
    assert!(
        res.loglik >= ll_ref - 1e-6 * ll_ref.abs(),
        "airline fit must match or beat statsmodels: {} vs {ll_ref}",
        res.loglik
    );
    let ref_params = as_vec(&block["fit_params"]);
    for (i, (got, want)) in res.params().iter().zip(&ref_params).enumerate() {
        assert_rel_close(*got, *want, 5e-3, &format!("airline fit param {i}"));
    }
    assert_eq!(res.nobs, block["nobs_effective"].as_u64().unwrap() as usize);
}

/// Approx (observed-information) standard errors at the statsmodels
/// fitted parameters match `cov_type='approx'` — same estimator, same
/// evaluation point.
#[test]
fn golden_airline_bse_at_recorded_params() {
    let fx = load_fixture("sarima.json");
    let block = &fx["airline_011_011_12"];
    let y = log_airline();
    let params = as_vec(&block["fit_params"]);
    let res = airline_spec().at_params(&y, &params).unwrap();
    let bse = res.bse().unwrap();
    let want = as_vec(&block["fit_bse_approx"]);
    for (i, (got, want)) in bse.iter().zip(&want).enumerate() {
        assert_rel_close(*got, *want, 1e-4, &format!("airline bse {i}"));
    }
}

/// Levels forecasts at the recorded parameters against the statsmodels
/// levels state-space form (`simple_differencing=False`, exact diffuse
/// initialization). The two conventions differ by initialization
/// effects decaying like the MA roots to the n-th power; at n = 144 the
/// point forecasts agree far tighter than the 1e-6 relative gate, and
/// the standard errors to 5e-3 (the diffuse form carries slightly
/// different filtered uncertainty into the forecast origin).
#[test]
fn golden_airline_levels_forecast() {
    let fx = load_fixture("sarima.json");
    let block = &fx["airline_011_011_12"];
    let y = log_airline();
    let params = as_vec(&block["fit_params"]);
    let res = airline_spec().at_params(&y, &params).unwrap();
    let fc = res.forecast(24).unwrap();
    let want_mean = as_vec(&block["forecast_mean_24_levels_ssm"]);
    let want_se = as_vec(&block["forecast_se_24_levels_ssm"]);
    for h in 0..24 {
        assert_rel_close(
            fc.mean[h],
            want_mean[h],
            1e-6,
            &format!("airline forecast mean h={}", h + 1),
        );
        assert_rel_close(
            fc.se[h],
            want_se[h],
            5e-3,
            &format!("airline forecast se h={}", h + 1),
        );
    }
}

/// Quarterly SAR(1)x(1)_4 with constant: fixed-parameter loglike to
/// 1e-8, the fit matches or beats statsmodels, and the approx standard
/// errors agree at the recorded parameters.
#[test]
fn golden_quarterly_sar_constant() {
    let fx = load_fixture("sarima.json");
    let block = &fx["quarterly_sar_c"];
    let y = as_vec(&block["y"]);
    let spec = ArimaSpec::new(1, 0, 0)
        .unwrap()
        .seasonal(1, 0, 0, 4)
        .unwrap()
        .with_constant(true);

    let fixed = as_vec(&block["fixed_params_const_phi_Phi_sigma2"]);
    let ll = spec.loglike(&y, &fixed).unwrap();
    assert_rel_close(
        ll,
        block["loglike_fixed"].as_f64().unwrap(),
        1e-8,
        "quarterly SAR loglike_fixed",
    );

    let res = spec.fit(&y).unwrap();
    let ll_ref = block["fit_loglike"].as_f64().unwrap();
    assert!(
        res.loglik >= ll_ref - 1e-6 * ll_ref.abs(),
        "quarterly SAR fit must match or beat statsmodels: {} vs {ll_ref}",
        res.loglik
    );
    let ref_params = as_vec(&block["fit_params"]);
    for (i, (got, want)) in res.params().iter().zip(&ref_params).enumerate() {
        assert_rel_close(*got, *want, 5e-3, &format!("quarterly SAR fit param {i}"));
    }

    // Gate against statsmodels' real four-point Hessian at the same
    // point — the same kind of estimator this crate computes. The
    // complex-step `fit_bse_approx` differs from it by up to ~5e-3
    // relative when the evaluation point is not exactly stationary
    // (measured on this very case), so it gets the looser sanity gate.
    let at = spec.at_params(&y, &ref_params).unwrap();
    let bse = at.bse().unwrap();
    let want_hess3 = as_vec(&block["fit_bse_hess3"]);
    for (i, (got, want)) in bse.iter().zip(&want_hess3).enumerate() {
        assert_rel_close(*got, *want, 1e-4, &format!("quarterly SAR bse (hess3) {i}"));
    }
    let want_cs = as_vec(&block["fit_bse_approx"]);
    for (i, (got, want)) in bse.iter().zip(&want_cs).enumerate() {
        assert_rel_close(
            *got,
            *want,
            1e-2,
            &format!("quarterly SAR bse (approx) {i}"),
        );
    }

    // Forecasts: d = D = 0, so simple and levels forms coincide and the
    // recorded statsmodels forecast is directly comparable.
    let fc = at.forecast(12).unwrap();
    let want_mean = as_vec(&block["forecast_mean_12"]);
    let want_se = as_vec(&block["forecast_se_12"]);
    for h in 0..12 {
        assert_rel_close(
            fc.mean[h],
            want_mean[h],
            1e-6,
            &format!("quarterly SAR forecast mean h={}", h + 1),
        );
        assert_rel_close(
            fc.se[h],
            want_se[h],
            1e-6,
            &format!("quarterly SAR forecast se h={}", h + 1),
        );
    }
}

/// The full mixed SARIMA(1,1,1)(1,1,1)_4 at fixed parameters: one gate
/// covering seasonal + regular differencing and both polynomial
/// expansions at once.
#[test]
fn golden_mixed_sarima_loglike_fixed() {
    let fx = load_fixture("sarima.json");
    let block = &fx["mixed_111_111_4"];
    let y = as_vec(&block["y"]);
    let spec = ArimaSpec::new(1, 1, 1)
        .unwrap()
        .seasonal(1, 1, 1, 4)
        .unwrap();
    let params = as_vec(&block["fixed_params_phi_theta_Phi_Theta_sigma2"]);
    let ll = spec.loglike(&y, &params).unwrap();
    assert_rel_close(
        ll,
        block["loglike_fixed"].as_f64().unwrap(),
        1e-8,
        "mixed SARIMA loglike_fixed",
    );
}
