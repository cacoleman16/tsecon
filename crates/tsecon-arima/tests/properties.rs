//! Property and consistency tests: parameter recovery on simulated data,
//! CSS/MLE asymptotic agreement, forecast-variance monotonicity and the
//! exact random-walk law, standardized residuals, the AR cross-check
//! against the engine's own constructor, and error-path coverage.

mod common;

use common::{as_vec, assert_rel_close, integrate, load_fixture, simulate_arma, Lcg};
use tsecon_arima::{arma_ssm, ArimaError, ArimaSpec, EstimationMethod};
use tsecon_ssm::tsecon_linalg::faer::Mat;
use tsecon_ssm::LinearGaussianSSM;

fn nile() -> Vec<f64> {
    as_vec(&load_fixture("diagnostics.json")["nile"])
}

/// Exact MLE on one seeded simulated ARMA(1,1)+constant sample recovers
/// the generating parameters within loose Monte Carlo bounds.
#[test]
fn mle_recovers_simulated_arma11c() {
    let mut rng = Lcg::new(20260716);
    let (c0, phi0, theta0, s20) = (1.0, 0.6, 0.3, 1.5);
    let y = simulate_arma(&mut rng, 600, c0, &[phi0], &[theta0], s20);

    let spec = ArimaSpec::new(1, 0, 1).unwrap().with_constant(true);
    let res = spec.fit(&y).unwrap();

    assert!(res.converged, "MLE did not converge");
    assert!(
        (res.ar()[0] - phi0).abs() < 0.12,
        "phi {} vs {phi0}",
        res.ar()[0]
    );
    assert!(
        (res.ma()[0] - theta0).abs() < 0.15,
        "theta {} vs {theta0}",
        res.ma()[0]
    );
    assert!(
        (res.sigma2() / s20 - 1.0).abs() < 0.15,
        "sigma2 {} vs {s20}",
        res.sigma2()
    );
    // The implied unconditional mean c / (1 - phi) is the well-identified
    // function of the constant; compare on that scale.
    let mean_hat = res.constant().unwrap() / (1.0 - res.ar()[0]);
    let mean0 = c0 / (1.0 - phi0);
    assert!(
        (mean_hat - mean0).abs() < 0.5,
        "mean {mean_hat} vs {mean0}"
    );
}

/// Exact MLE on one seeded integrated sample: ARIMA(1,1,1) fit on the
/// cumulated series recovers the ARMA parameters of the differences.
#[test]
fn mle_recovers_simulated_arima111() {
    let mut rng = Lcg::new(7);
    let x = simulate_arma(&mut rng, 500, 0.0, &[0.5], &[-0.3], 2.0);
    let y = integrate(&x, 1);

    let spec = ArimaSpec::new(1, 1, 1).unwrap();
    let res = spec.fit(&y).unwrap();

    assert!(res.converged);
    assert_eq!(res.nobs, 499);
    assert!((res.ar()[0] - 0.5).abs() < 0.15, "phi {}", res.ar()[0]);
    assert!((res.ma()[0] + 0.3).abs() < 0.15, "theta {}", res.ma()[0]);
    assert!((res.sigma2() / 2.0 - 1.0).abs() < 0.15, "sigma2 {}", res.sigma2());
}

/// CSS and exact MLE agree to ~1e-2 on a long simulated series (they are
/// asymptotically equivalent; the conditioning effect is O(1/n)).
#[test]
fn css_and_mle_agree_on_long_series() {
    let mut rng = Lcg::new(42);
    let y = simulate_arma(&mut rng, 800, 0.5, &[0.7], &[0.4], 1.2);

    let spec = ArimaSpec::new(1, 0, 1).unwrap().with_constant(true);
    let mle = spec.fit(&y).unwrap();
    let css = spec.fit_css(&y).unwrap();

    assert_eq!(css.method, EstimationMethod::Css);
    assert_eq!(mle.method, EstimationMethod::ExactMle);
    assert_eq!(css.nobs, mle.nobs - 1, "CSS conditions on the first p obs");

    assert!(
        (mle.ar()[0] - css.ar()[0]).abs() < 0.02,
        "phi: mle {} vs css {}",
        mle.ar()[0],
        css.ar()[0]
    );
    assert!(
        (mle.ma()[0] - css.ma()[0]).abs() < 0.03,
        "theta: mle {} vs css {}",
        mle.ma()[0],
        css.ma()[0]
    );
    assert!(
        (mle.constant().unwrap() - css.constant().unwrap()).abs() < 0.05,
        "const: mle {:?} vs css {:?}",
        mle.constant(),
        css.constant()
    );
    assert!(
        (mle.sigma2() / css.sigma2() - 1.0).abs() < 0.02,
        "sigma2: mle {} vs css {}",
        mle.sigma2(),
        css.sigma2()
    );

    // The reported CSS loglik obeys its documented concentrated form.
    let n_c = css.nobs as f64;
    let expected_ll =
        -0.5 * n_c * ((2.0 * std::f64::consts::PI).ln() + 1.0 + css.sigma2().ln());
    assert_rel_close(css.loglik, expected_ll, 1e-10, "css loglik identity");
}

/// Forecast standard errors are monotone nondecreasing in the horizon,
/// both without differencing (stationary ARMA: they converge upward to
/// the unconditional standard deviation) and with d = 1 (they grow
/// without bound).
#[test]
fn forecast_se_monotone_nondecreasing() {
    let y = nile();

    let arma = ArimaSpec::new(1, 0, 1).unwrap().with_constant(true);
    let fc = arma.fit(&y).unwrap().forecast(24).unwrap();
    for h in 1..fc.se.len() {
        assert!(
            fc.se[h] >= fc.se[h - 1] - 1e-9,
            "d=0 se not monotone at h={h}: {} < {}",
            fc.se[h],
            fc.se[h - 1]
        );
    }

    let arima = ArimaSpec::new(1, 1, 1).unwrap();
    let fc = arima.fit(&y).unwrap().forecast(24).unwrap();
    for h in 1..fc.se.len() {
        assert!(
            fc.se[h] >= fc.se[h - 1] - 1e-9,
            "d=1 se not monotone at h={h}: {} < {}",
            fc.se[h],
            fc.se[h - 1]
        );
    }
}

/// ARIMA(0,1,0) — the pure random walk — has the closed-form forecast
/// law `mean_h = y_n`, `se_h = sigma sqrt(h)`; the cumulator-augmented
/// prediction recursion must reproduce it to near machine precision.
#[test]
fn random_walk_forecast_law_exact() {
    let mut rng = Lcg::new(99);
    let x = simulate_arma(&mut rng, 200, 0.0, &[], &[], 4.0);
    let y = integrate(&x, 1);
    let sigma2 = 3.7; // any fixed value; the law is exact in it

    let spec = ArimaSpec::new(0, 1, 0).unwrap();
    let res = spec.at_params(&y, &[sigma2]).unwrap();
    assert_eq!(res.method, EstimationMethod::Fixed);
    let fc = res.forecast(10).unwrap();
    let last = y[y.len() - 1];
    for h in 1..=10 {
        assert_rel_close(fc.mean[h - 1], last, 1e-12, &format!("rw mean[{h}]"));
        assert_rel_close(
            fc.se[h - 1],
            (h as f64 * sigma2).sqrt(),
            1e-10,
            &format!("rw se[{h}]"),
        );
    }
}

/// Standardized one-step prediction errors from a well-specified fit
/// have (approximately) zero mean and unit variance, and there are
/// exactly `n - d` of them.
#[test]
fn residuals_standardized() {
    let y = nile();
    let spec = ArimaSpec::new(1, 0, 1).unwrap().with_constant(true);
    let res = spec.fit(&y).unwrap();
    let e = res.residuals().unwrap();
    assert_eq!(e.len(), 100);
    let m = e.iter().sum::<f64>() / e.len() as f64;
    let v = e.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / e.len() as f64;
    assert!(m.abs() < 0.3, "residual mean {m}");
    assert!((0.6..=1.4).contains(&v), "residual variance {v}");
}

/// For a pure AR(p) with intercept, [`arma_ssm`] reproduces the engine's
/// own [`LinearGaussianSSM::ar`] constructor: identical log-likelihoods
/// on the same data (the MA generalization must not disturb the AR
/// conventions).
#[test]
fn arma_ssm_matches_ar_constructor() {
    let mut rng = Lcg::new(3);
    let y = simulate_arma(&mut rng, 300, 0.8, &[0.5, -0.2], &[], 1.0);
    let y_mat = Mat::from_fn(y.len(), 1, |i, _| y[i]);

    let a = arma_ssm(&[0.5, -0.2], &[], 1.3, 0.8).unwrap();
    let b = LinearGaussianSSM::ar(&[0.5, -0.2], 1.3, 0.8).unwrap();
    let ll_a = a.loglike(y_mat.as_ref()).unwrap();
    let ll_b = b.loglike(y_mat.as_ref()).unwrap();
    assert_rel_close(ll_a, ll_b, 1e-12, "AR cross-check loglik");
}

/// `at_params` reports exactly the same log-likelihood as `loglike`, and
/// its AIC/BIC follow the statsmodels conventions at those parameters.
#[test]
fn at_params_consistent_with_loglike() {
    let y = nile();
    let spec = ArimaSpec::new(1, 1, 1).unwrap();
    let params = [0.3, -0.6, 20000.0];
    let ll = spec.loglike(&y, &params).unwrap();
    let res = spec.at_params(&y, &params).unwrap();
    assert_rel_close(res.loglik, ll, 0.0, "at_params loglik");
    assert_eq!(res.nobs, 99);
    assert_rel_close(res.aic, -2.0 * ll + 6.0, 1e-12, "aic");
    assert_rel_close(res.bic, -2.0 * ll + 3.0 * 99f64.ln(), 1e-12, "bic");
}

/// Gaussian forecast intervals bracket the mean symmetrically with the
/// 1.96-sigma half-width at alpha = 0.05.
#[test]
fn forecast_conf_int() {
    let y = nile();
    let spec = ArimaSpec::new(1, 0, 1).unwrap().with_constant(true);
    let res = spec
        .at_params(&y, &[314.84, 0.6588, -0.248, 20480.5])
        .unwrap();
    let fc = res.forecast(5).unwrap();
    let ci = fc.conf_int(0.05).unwrap();
    for (h, (lo, hi)) in ci.iter().enumerate() {
        let half = hi - fc.mean[h];
        assert_rel_close(fc.mean[h] - lo, half, 1e-12, "symmetry");
        assert_rel_close(half / fc.se[h], 1.959963984540054, 1e-9, "z quantile");
    }
    assert!(fc.conf_int(0.0).is_err());
    assert!(fc.conf_int(1.0).is_err());
}

/// Error paths: order caps, sample-size floors, malformed parameter
/// vectors, NaN data, boundary starting values, and zero-step forecasts
/// all fail loudly with typed errors — never a panic.
#[test]
fn error_paths() {
    // Order sanity cap.
    assert!(matches!(
        ArimaSpec::new(1001, 0, 0),
        Err(ArimaError::InvalidArgument { .. })
    ));

    let spec = ArimaSpec::new(1, 0, 1).unwrap().with_constant(true);

    // Too few observations (k + 1 = 5 needed).
    assert!(matches!(
        spec.fit(&[1.0, 2.0, 3.0, 4.0]),
        Err(ArimaError::InsufficientObservations { .. })
    ));

    // NaN data are rejected on the simple-differencing path.
    let mut y = nile();
    y[3] = f64::NAN;
    assert!(matches!(
        spec.fit(&y),
        Err(ArimaError::NonFinite { .. })
    ));
    let y = nile();

    // Wrong parameter-vector length.
    assert!(matches!(
        spec.loglike(&y, &[0.5, 0.1, 1.0]),
        Err(ArimaError::Dimension { .. })
    ));
    // Non-positive sigma2.
    assert!(matches!(
        spec.loglike(&y, &[300.0, 0.5, 0.1, 0.0]),
        Err(ArimaError::InvalidArgument { .. })
    ));
    // Non-stationary AR coefficients admit no stationary initialization.
    assert!(spec.loglike(&y, &[300.0, 1.2, 0.1, 100.0]).is_err());
    // Boundary starting values are rejected by the reparameterization.
    assert!(matches!(
        spec.fit_with_start(&y, &[300.0, 1.0, 0.1, 100.0]),
        Err(ArimaError::Optim(_))
    ));

    // Zero-step forecasts.
    let res = spec.at_params(&y, &[314.84, 0.6588, -0.248, 20480.5]).unwrap();
    assert!(matches!(
        res.forecast(0),
        Err(ArimaError::InvalidArgument { .. })
    ));

    // Differencing longer than the sample.
    let short = ArimaSpec::new(0, 3, 0).unwrap();
    assert!(matches!(
        short.fit(&[1.0, 2.0, 3.0]),
        Err(ArimaError::InsufficientObservations { .. })
    ));
}
