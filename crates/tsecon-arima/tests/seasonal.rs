//! Seasonal (SARIMA) behavior: the multiplied-out polynomials against
//! dense ARMA equivalents (bit-exact), seasonal differencing against a
//! manual difference, closed-form forecast laws for the seasonal random
//! walk, the levels-forecast difference-equation invariant, parameter
//! naming/layout, validation errors, and parameter recovery on seeded
//! simulated seasonal samples.

mod common;

use common::{assert_rel_close, simulate_arma, Lcg};
use tsecon_arima::ArimaSpec;

/// Simulates a multiplicative SARMA sample by expanding the seasonal
/// polynomials into their dense ARMA form and reusing the shared
/// simulator (the expansion here is written out longhand, independent of
/// the crate's own expansion).
#[allow(clippy::too_many_arguments)]
fn simulate_sarma(
    rng: &mut Lcg,
    n: usize,
    ar: &[f64],
    ma: &[f64],
    sar: &[f64],
    sma: &[f64],
    s: usize,
    sigma2: f64,
) -> Vec<f64> {
    let mut ar_full = vec![0.0; ar.len() + s * sar.len()];
    ar_full[..ar.len()].copy_from_slice(ar);
    for (j, &b) in sar.iter().enumerate() {
        let js = s * (j + 1);
        ar_full[js - 1] += b;
        for (i, &a) in ar.iter().enumerate() {
            ar_full[js + i] -= a * b;
        }
    }
    let mut ma_full = vec![0.0; ma.len() + s * sma.len()];
    ma_full[..ma.len()].copy_from_slice(ma);
    for (j, &b) in sma.iter().enumerate() {
        let js = s * (j + 1);
        ma_full[js - 1] += b;
        for (i, &a) in ma.iter().enumerate() {
            ma_full[js + i] += a * b;
        }
    }
    simulate_arma(rng, n, 0.0, &ar_full, &ma_full, sigma2)
}

/// The seasonal AR factor is exactly the dense expanded polynomial: a
/// SARIMA(1,0,0)(1,0,0)_4 log-likelihood is bit-identical to the dense
/// ARMA(5,0) at `[phi, 0, 0, Phi, -phi*Phi]`, and the MA side to the
/// dense ARMA(0,5) at `[theta, 0, 0, Theta, +theta*Theta]`.
#[test]
fn seasonal_expansion_equals_dense_arma() {
    let mut rng = Lcg::new(11);
    let y = simulate_arma(&mut rng, 200, 0.0, &[0.5], &[0.3], 1.0);

    let (phi, cap_phi, sigma2) = (0.5, 0.3, 1.3);
    let seasonal = ArimaSpec::new(1, 0, 0)
        .unwrap()
        .seasonal(1, 0, 0, 4)
        .unwrap();
    let dense = ArimaSpec::new(5, 0, 0).unwrap();
    let ll_seasonal = seasonal.loglike(&y, &[phi, cap_phi, sigma2]).unwrap();
    let ll_dense = dense
        .loglike(&y, &[phi, 0.0, 0.0, cap_phi, -phi * cap_phi, sigma2])
        .unwrap();
    assert_eq!(ll_seasonal, ll_dense, "AR expansion must be bit-exact");

    let (theta, cap_theta) = (0.4, -0.25);
    let seasonal = ArimaSpec::new(0, 0, 1)
        .unwrap()
        .seasonal(0, 0, 1, 4)
        .unwrap();
    let dense = ArimaSpec::new(0, 0, 5).unwrap();
    let ll_seasonal = seasonal.loglike(&y, &[theta, cap_theta, sigma2]).unwrap();
    let ll_dense = dense
        .loglike(&y, &[theta, 0.0, 0.0, cap_theta, theta * cap_theta, sigma2])
        .unwrap();
    assert_eq!(ll_seasonal, ll_dense, "MA expansion must be bit-exact");
}

/// Seasonal differencing matches the manual seasonal-then-regular
/// difference: SARIMA(0,1,0)(0,1,0)_4 on `y` has the same log-likelihood
/// as ARIMA(0,0,0) on the hand-differenced series.
#[test]
fn seasonal_differencing_matches_manual() {
    let mut rng = Lcg::new(23);
    let y = simulate_arma(&mut rng, 120, 0.3, &[0.4], &[], 1.0);

    let s = 4;
    let w: Vec<f64> = (s..y.len()).map(|t| y[t] - y[t - s]).collect();
    let x: Vec<f64> = (1..w.len()).map(|t| w[t] - w[t - 1]).collect();

    let spec = ArimaSpec::new(0, 1, 0)
        .unwrap()
        .seasonal(0, 1, 0, s)
        .unwrap();
    let flat = ArimaSpec::new(0, 0, 0).unwrap();
    let sigma2 = 1.7;
    let ll_spec = spec.loglike(&y, &[sigma2]).unwrap();
    let ll_manual = flat.loglike(&x, &[sigma2]).unwrap();
    assert_eq!(
        ll_spec, ll_manual,
        "seasonal differencing must be bit-exact"
    );
}

/// The pure seasonal random walk (0,0,0)(0,1,0)_s obeys its closed-form
/// forecast law: the point forecast repeats the last observed season and
/// the standard error is `sigma * sqrt(ceil(h/s))` exactly.
#[test]
fn seasonal_random_walk_forecast_law_exact() {
    let s = 4;
    let mut rng = Lcg::new(31);
    // A seasonal random walk: y_t = y_{t-s} + e_t.
    let n = 60;
    let mut y = vec![0.0; n];
    for t in 0..n {
        let prev = if t >= s { y[t - s] } else { 0.0 };
        y[t] = prev + rng.gaussian();
    }

    let sigma2 = 2.25;
    let spec = ArimaSpec::new(0, 0, 0)
        .unwrap()
        .seasonal(0, 1, 0, s)
        .unwrap();
    let res = spec.at_params(&y, &[sigma2]).unwrap();
    let fc = res.forecast(3 * s).unwrap();
    for h in 1..=3 * s {
        let expected_mean = y[n - s + (h - 1) % s];
        let expected_se = sigma2.sqrt() * (h.div_ceil(s) as f64).sqrt();
        assert_rel_close(
            fc.mean[h - 1],
            expected_mean,
            1e-12,
            &format!("seasonal RW mean at h={h}"),
        );
        assert_rel_close(
            fc.se[h - 1],
            expected_se,
            1e-12,
            &format!("seasonal RW se at h={h}"),
        );
    }
}

/// The airline-shaped specification's levels forecast satisfies the
/// difference equation: applying `(1-L)(1-L^s)` to the forecast path
/// (seeded with the observed tail) recovers the ARMA forecast of the
/// fully differenced series from the identical dense model — the
/// undifferencing augmentation cannot drift from the ARMA engine.
#[test]
fn levels_forecast_satisfies_difference_equation() {
    let s = 4;
    let mut rng = Lcg::new(47);
    // Integrated seasonal data: cumulate an SMA(1)xMA(1) sample twice
    // (once regularly, once seasonally).
    let x = simulate_sarma(&mut rng, 160, &[], &[0.35], &[], &[-0.4], s, 1.0);
    let mut w = x.clone();
    let mut acc = 0.0;
    for v in &mut w {
        acc += *v;
        *v = acc;
    }
    let mut y = w.clone();
    for t in s..y.len() {
        y[t] += y[t - s];
    }

    let spec = ArimaSpec::new(0, 1, 1)
        .unwrap()
        .seasonal(0, 1, 1, s)
        .unwrap();
    let params = [0.35, -0.4, 1.0];
    let res = spec.at_params(&y, &params).unwrap();
    let steps = 10;
    let fc = res.forecast(steps).unwrap();

    // The identical dense ARMA on the hand-differenced series.
    let n = y.len();
    let wd: Vec<f64> = (s..n).map(|t| y[t] - y[t - s]).collect();
    let xd: Vec<f64> = (1..wd.len()).map(|t| wd[t] - wd[t - 1]).collect();
    let dense = ArimaSpec::new(0, 0, 1 + s).unwrap();
    let theta = params[0];
    let cap_theta = params[1];
    let mut ma_full = vec![0.0; 1 + s];
    ma_full[0] = theta;
    ma_full[s - 1] += cap_theta;
    ma_full[s] += theta * cap_theta;
    let mut dense_params = ma_full;
    dense_params.push(params[2]);
    let fc_x = dense
        .at_params(&xd, &dense_params)
        .unwrap()
        .forecast(steps)
        .unwrap();

    // Difference the forecast path: values before the horizon come from
    // the observed series.
    let yv = |h: isize| -> f64 {
        if h >= 1 {
            fc.mean[(h - 1) as usize]
        } else {
            y[(n as isize - 1 + h) as usize]
        }
    };
    for h in 1..=steps as isize {
        let implied_x = yv(h) - yv(h - 1) - yv(h - s as isize) + yv(h - 1 - s as isize);
        assert_rel_close(
            implied_x,
            fc_x.mean[(h - 1) as usize],
            1e-9,
            &format!("difference-equation invariant at h={h}"),
        );
    }
}

/// Parameter names, layout, and count follow the statsmodels SARIMAX
/// conventions.
#[test]
fn param_names_and_layout() {
    let spec = ArimaSpec::new(2, 1, 1)
        .unwrap()
        .seasonal(1, 1, 1, 12)
        .unwrap()
        .with_constant(true);
    assert_eq!(spec.k_params(), 1 + 2 + 1 + 1 + 1 + 1);
    assert_eq!(
        spec.param_names(),
        vec![
            "const".to_owned(),
            "ar.L1".to_owned(),
            "ar.L2".to_owned(),
            "ma.L1".to_owned(),
            "ar.S.L12".to_owned(),
            "ma.S.L12".to_owned(),
            "sigma2".to_owned(),
        ]
    );
    assert_eq!(spec.seasonal_p(), 1);
    assert_eq!(spec.seasonal_d(), 1);
    assert_eq!(spec.seasonal_q(), 1);
    assert_eq!(spec.period(), 12);

    // Two-seasonal-lag names step by the period.
    let spec = ArimaSpec::new(0, 0, 0)
        .unwrap()
        .seasonal(2, 0, 0, 4)
        .unwrap();
    assert_eq!(
        spec.param_names(),
        vec![
            "ar.S.L4".to_owned(),
            "ar.S.L8".to_owned(),
            "sigma2".to_owned()
        ]
    );
}

/// Validation: a period below 2 is rejected when any seasonal order is
/// nonzero, all-zero seasonal orders are the non-seasonal model at any
/// period, and the expanded-order cap binds.
#[test]
fn seasonal_validation_errors() {
    let base = ArimaSpec::new(1, 0, 0).unwrap();
    assert!(base.seasonal(1, 0, 0, 1).is_err(), "s = 1 must be rejected");
    assert!(base.seasonal(0, 1, 0, 0).is_err(), "s = 0 must be rejected");
    assert!(
        base.seasonal(1, 0, 0, 1000).is_err(),
        "expanded AR order 1 + 1000 must exceed the cap"
    );

    // (0,0,0) seasonal orders: the non-seasonal model, whatever the
    // period says.
    let mut rng = Lcg::new(5);
    let y = simulate_arma(&mut rng, 80, 0.0, &[0.5], &[], 1.0);
    let none = base.seasonal(0, 0, 0, 12).unwrap();
    assert_eq!(none, base);
    assert_eq!(
        none.loglike(&y, &[0.5, 1.0]).unwrap(),
        base.loglike(&y, &[0.5, 1.0]).unwrap()
    );

    // Too few observations for the seasonal difference.
    let spec = ArimaSpec::new(0, 0, 0)
        .unwrap()
        .seasonal(0, 1, 0, 12)
        .unwrap();
    assert!(
        spec.fit(&y[..12]).is_err(),
        "n = s must be too short for D = 1"
    );
}

/// Exact MLE recovers the parameters of a seeded multiplicative
/// SAR(1)xAR(1) sample (loose tolerances — one seeded draw, finite n).
#[test]
fn mle_recovers_simulated_seasonal_ar() {
    let s = 4;
    let (phi0, cap_phi0, s20) = (0.5, 0.4, 1.0);
    let mut rng = Lcg::new(20260813);
    let y = simulate_sarma(&mut rng, 800, &[phi0], &[], &[cap_phi0], &[], s, s20);

    let spec = ArimaSpec::new(1, 0, 0)
        .unwrap()
        .seasonal(1, 0, 0, s)
        .unwrap();
    let res = spec.fit(&y).unwrap();
    assert!(res.converged, "seasonal MLE must converge on this sample");
    let phi = res.ar()[0];
    let cap_phi = res.seasonal_ar()[0];
    assert!(
        (phi - phi0).abs() < 0.12,
        "phi: {phi} vs {phi0} (seeded draw, tol 0.12)"
    );
    assert!(
        (cap_phi - cap_phi0).abs() < 0.12,
        "Phi: {cap_phi} vs {cap_phi0} (seeded draw, tol 0.12)"
    );
    assert!(
        (res.sigma2() - s20).abs() < 0.2,
        "sigma2: {} vs {s20}",
        res.sigma2()
    );

    // The full results plumbing stays coherent for seasonal specs.
    assert_eq!(res.params().len(), res.param_names().len());
    let resid = res.residuals().unwrap();
    assert_eq!(resid.len(), y.len());
    let bse = res.bse().unwrap();
    assert_eq!(bse.len(), res.params().len());
    assert!(bse.iter().all(|v| v.is_finite() && *v > 0.0));
}

/// CSS and exact MLE agree on a long seasonal sample (they differ only
/// in the treatment of the first p + s*P observations).
#[test]
fn css_and_mle_agree_on_long_seasonal_series() {
    let s = 4;
    let mut rng = Lcg::new(99);
    let y = simulate_sarma(&mut rng, 2500, &[0.4], &[], &[0.35], &[], s, 1.5);

    let spec = ArimaSpec::new(1, 0, 0)
        .unwrap()
        .seasonal(1, 0, 0, s)
        .unwrap();
    let mle = spec.fit(&y).unwrap();
    let css = spec.fit_css(&y).unwrap();
    for (name, a, b) in [
        ("phi", mle.ar()[0], css.ar()[0]),
        ("Phi", mle.seasonal_ar()[0], css.seasonal_ar()[0]),
        ("sigma2", mle.sigma2(), css.sigma2()),
    ] {
        assert_rel_close(b, a, 0.05, &format!("CSS vs MLE {name}"));
    }
}
