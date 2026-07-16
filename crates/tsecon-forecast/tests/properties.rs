//! Property and guardrail tests: invariants that hold by construction,
//! plus the teaching errors on degenerate inputs.

use tsecon_forecast::{
    dm_test, drift, historical_mean, mape, mase, mdae, mse, naive, seasonal_naive, smape,
    theta_forecast, theta_forecast_with, DmLoss, ForecastError,
};

mod realgdp;
use realgdp::REALGDP;

/// Deterministic pseudo-random series from a splitmix-style generator —
/// enough irregularity to make the invariants non-trivial.
fn pseudo_series(n: usize, seed: u64) -> Vec<f64> {
    let mut state = seed;
    (0..n)
        .map(|t| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = (state >> 11) as f64 / (1u64 << 53) as f64;
            10.0 + 0.05 * t as f64 + 2.0 * (u - 0.5)
        })
        .collect()
}

#[test]
fn mase_of_insample_seasonal_naive_is_one() {
    for &period in &[1usize, 4, 12] {
        let y = pseudo_series(120, 7 + period as u64);
        // The in-sample seasonal-naive forecast of y[t] is y[t - period];
        // its MAE over t = period..n equals the MASE scaling denominator
        // by construction, so MASE == 1 exactly.
        let actual = &y[period..];
        let forecast = &y[..y.len() - period];
        let m = mase(actual, forecast, &y, period).unwrap();
        assert!(
            (m - 1.0).abs() < 1e-12,
            "MASE of in-sample seasonal naive (period {period}) should be 1, got {m}"
        );
    }
}

#[test]
fn dm_of_identical_errors_is_a_clear_error() {
    let e = pseudo_series(50, 3);
    let err = dm_test(&e, &e, 1, DmLoss::Squared).unwrap_err();
    assert_eq!(err, ForecastError::DegenerateLossDifferential);
    let msg = format!("{err}");
    assert!(
        msg.contains("identical losses") && msg.contains("compared with itself"),
        "degenerate DM error should teach, got: {msg}"
    );
    // Same under absolute and custom loss.
    assert_eq!(
        dm_test(&e, &e, 3, DmLoss::Absolute).unwrap_err(),
        ForecastError::DegenerateLossDifferential
    );
    // Sign-flipped errors have identical squared losses too.
    let neg: Vec<f64> = e.iter().map(|v| -v).collect();
    assert_eq!(
        dm_test(&e, &neg, 1, DmLoss::Squared).unwrap_err(),
        ForecastError::DegenerateLossDifferential
    );
}

#[test]
fn drift_forecast_lies_on_the_endpoint_slope_line() {
    let y = pseudo_series(83, 11);
    let n = y.len();
    let steps = 12;
    let res = drift(&y, steps, 0.95).unwrap();
    let slope = (y[n - 1] - y[0]) / (n as f64 - 1.0);
    for (j, &f) in res.mean.iter().enumerate() {
        let h = (j + 1) as f64;
        let line = y[n - 1] + h * slope;
        assert!(
            ((f - line) / line).abs() < 1e-12,
            "drift forecast at h={h} should sit on the line through the \
             first and last observations, got {f} vs {line}"
        );
    }
    // The classic drift widening: sigma_h strictly increasing, and faster
    // than the naive sqrt(h) rate (relative to sigma_1).
    for j in 1..steps {
        assert!(res.sigma_h[j] > res.sigma_h[j - 1]);
        let h = (j + 1) as f64;
        assert!(res.sigma_h[j] / res.sigma_h[0] > h.sqrt() * 0.999999);
    }
    // Intervals bracket the point forecast.
    for j in 0..steps {
        assert!(res.lower[j] < res.mean[j] && res.mean[j] < res.upper[j]);
    }
}

#[test]
fn seasonal_naive_repeats_the_last_season() {
    let m = 7usize;
    let y = pseudo_series(60, 21);
    let n = y.len();
    let steps = 2 * m + 3;
    let res = seasonal_naive(&y, m, steps, 0.9).unwrap();
    for (j, &f) in res.mean.iter().enumerate() {
        let expected = y[n - m + (j % m)];
        assert_eq!(
            f,
            expected,
            "seasonal-naive h={} should repeat the last season exactly",
            j + 1
        );
    }
    // sigma_h steps up once per completed cycle: sqrt(k+1) pattern.
    let s1 = res.sigma_h[0];
    for (j, &s) in res.sigma_h.iter().enumerate() {
        let k = (j / m) as f64;
        assert!(((s / s1) - (k + 1.0).sqrt()).abs() < 1e-12);
    }
}

#[test]
fn naive_repeats_last_value_with_sqrt_h_widening() {
    let y = pseudo_series(40, 5);
    let res = naive(&y, 6, 0.95).unwrap();
    for &f in &res.mean {
        assert_eq!(f, *y.last().unwrap());
    }
    let s1 = res.sigma_h[0];
    for (j, &s) in res.sigma_h.iter().enumerate() {
        assert!(((s / s1) - ((j + 1) as f64).sqrt()).abs() < 1e-12);
    }
}

#[test]
fn historical_mean_is_flat_with_constant_sigma() {
    let y = pseudo_series(40, 9);
    let ybar = y.iter().sum::<f64>() / y.len() as f64;
    let res = historical_mean(&y, 5, 0.8).unwrap();
    for &f in &res.mean {
        assert!(((f - ybar) / ybar).abs() < 1e-15);
    }
    for &s in &res.sigma_h {
        assert_eq!(s, res.sigma_h[0], "mean-model sigma_h is horizon-free");
    }
}

#[test]
fn theta_one_collapses_to_flat_ses() {
    // theta = 1 puts zero weight on the trend line, so the deseasonalized
    // forecast is the constant SES level.
    let y = pseudo_series(50, 13);
    let res = theta_forecast_with(&y, 1, 6, 1.0).unwrap();
    for &f in &res.forecast {
        assert_eq!(f, res.one_step);
    }
}

#[test]
fn theta_seasonal_factors_average_to_one() {
    let res = theta_forecast(REALGDP, 4, 4).unwrap();
    assert!(res.multiplicative);
    let mean = res.seasonal.iter().sum::<f64>() / res.seasonal.len() as f64;
    assert!(
        (mean - 1.0).abs() < 1e-12,
        "multiplicative factors are normalized to mean one, got {mean}"
    );
    // Reseasonalization phase: horizon j uses factor (n + j) % period.
    let n = REALGDP.len();
    let deseasonalized_level: Vec<f64> = res
        .forecast
        .iter()
        .enumerate()
        .map(|(j, f)| f / res.seasonal[(n + j) % 4])
        .collect();
    // Deseasonalized theta forecasts grow linearly with slope b0/2.
    let d0 = deseasonalized_level[1] - deseasonalized_level[0];
    assert!(((d0 - 0.5 * res.b0) / res.b0).abs() < 1e-10);
}

#[test]
fn guardrails_teach_on_degenerate_inputs() {
    // MAPE with a zero actual.
    let err = mape(&[1.0, 0.0, 2.0], &[1.0, 1.0, 2.0]).unwrap_err();
    assert_eq!(err, ForecastError::ZeroActualInMape { index: 1 });
    assert!(format!("{err}").contains("MASE"), "should point to MASE");

    // sMAPE with a zero denominator.
    let err = smape(&[1.0, 0.0], &[1.0, 0.0]).unwrap_err();
    assert_eq!(err, ForecastError::ZeroDenominatorInSmape { index: 1 });

    // MASE with a constant training series.
    let err = mase(&[1.0, 2.0], &[1.0, 1.0], &[3.0; 10], 1).unwrap_err();
    assert!(matches!(
        err,
        ForecastError::ZeroScaleDenominator { period: 1, .. }
    ));

    // NaN propagation discipline: loud error, never a silent skip.
    let nan_err = mse(&[1.0, f64::NAN], &[1.0, 1.0]).unwrap_err();
    assert!(matches!(nan_err, ForecastError::NonFinite { index: 1, .. }));
    assert!(format!("{nan_err}").contains("does not skip missing values"));
    assert!(matches!(
        naive(&[1.0, f64::INFINITY, 2.0], 3, 0.95).unwrap_err(),
        ForecastError::NonFinite { index: 1, .. }
    ));
    assert!(matches!(
        dm_test(&[1.0, f64::NAN, 0.5], &[1.0, 0.2, 0.4], 1, DmLoss::Squared).unwrap_err(),
        ForecastError::NonFinite { index: 1, .. }
    ));
    assert!(matches!(
        theta_forecast(&[1.0, 2.0, f64::NAN, 3.0, 4.0], 1, 2).unwrap_err(),
        ForecastError::NonFinite { index: 2, .. }
    ));

    // Length mismatch.
    assert!(matches!(
        mdae(&[1.0, 2.0], &[1.0]).unwrap_err(),
        ForecastError::LengthMismatch { .. }
    ));

    // Invalid DM horizon.
    let e = pseudo_series(10, 1);
    let f = pseudo_series(10, 2);
    assert!(matches!(
        dm_test(&e, &f, 0, DmLoss::Squared).unwrap_err(),
        ForecastError::InvalidHorizon { h: 0, .. }
    ));
    assert!(matches!(
        dm_test(&e, &f, 10, DmLoss::Squared).unwrap_err(),
        ForecastError::InvalidHorizon { h: 10, .. }
    ));

    // Invalid interval level and step counts.
    let y = pseudo_series(20, 4);
    assert!(matches!(
        naive(&y, 5, 1.0).unwrap_err(),
        ForecastError::InvalidLevel { .. }
    ));
    assert!(matches!(
        drift(&y, 0, 0.95).unwrap_err(),
        ForecastError::InvalidSteps { steps: 0 }
    ));
    assert!(matches!(
        theta_forecast_with(&y, 1, 4, 0.5).unwrap_err(),
        ForecastError::InvalidTheta { .. }
    ));
    assert!(matches!(
        seasonal_naive(&y, 0, 3, 0.95).unwrap_err(),
        ForecastError::InvalidPeriod { period: 0, .. }
    ));
}
