//! Property and guardrail tests for the nested- and conditional-model
//! predictive-ability tests (Clark-West, Giacomini-White).

use tsecon_forecast::{cw_test, gw_test, gw_test_conditional, ForecastError};

/// Deterministic pseudo-random series in roughly `[-1, 1]`.
fn prand(n: usize, seed: u64) -> Vec<f64> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = (state >> 11) as f64 / (1u64 << 53) as f64;
            2.0 * (u - 0.5)
        })
        .collect()
}

#[test]
fn gw_conditional_with_constant_equals_unconditional() {
    // For several random loss pairs, the conditional test with the single
    // constant instrument h_t = 1 must reproduce the unconditional statistic.
    for seed in [1u64, 42, 7, 2024] {
        let loss1 = prand(80, seed).iter().map(|v| v * v).collect::<Vec<_>>();
        let loss2 = prand(80, seed + 100)
            .iter()
            .map(|v| v * v)
            .collect::<Vec<_>>();
        for lags in [0usize, 1, 4] {
            let uncond = gw_test(&loss1, &loss2, lags).unwrap();
            let ones = vec![vec![1.0]; loss1.len()];
            let cond = gw_test_conditional(&loss1, &loss2, &ones, lags).unwrap();
            assert_eq!(cond.df, 1);
            assert_eq!(
                cond.gw_stat, uncond.gw_stat,
                "conditional (h=1) must equal unconditional (seed {seed}, lags {lags})"
            );
            assert_eq!(cond.p_value, uncond.p_value);
        }
    }
}

#[test]
fn gw_conditional_multidimensional_is_chi2_q() {
    let loss1 = prand(90, 5).iter().map(|v| v * v).collect::<Vec<_>>();
    let loss2 = prand(90, 55).iter().map(|v| v * v).collect::<Vec<_>>();
    let dl: Vec<f64> = loss1.iter().zip(loss2.iter()).map(|(a, b)| a - b).collect();
    // Instruments: constant plus one lag of the loss differential.
    let h: Vec<Vec<f64>> = (0..dl.len())
        .map(|t| vec![1.0, if t == 0 { 0.0 } else { dl[t - 1] }])
        .collect();
    let res = gw_test_conditional(&loss1, &loss2, &h, 2).unwrap();
    assert_eq!(res.df, 2);
    assert!(res.gw_stat.is_finite() && res.gw_stat >= 0.0);
    assert!(res.p_value > 0.0 && res.p_value <= 1.0);
}

#[test]
fn clark_west_favors_the_larger_model_when_it_helps() {
    // Construct a signal the small (restricted) model ignores but the large
    // one captures: yhat_small = 0, yhat_large = 0.9 * y. Then
    // f_t = y^2 - (0.1 y)^2 + (0.9 y)^2 = 1.8 y^2 > 0, so mean(f) > 0 and the
    // one-sided CW test rejects in favour of the larger model.
    let y = prand(120, 314);
    let yhat_small = vec![0.0; y.len()];
    let yhat_large: Vec<f64> = y.iter().map(|v| 0.9 * v).collect();
    let e_small: Vec<f64> = y
        .iter()
        .zip(yhat_small.iter())
        .map(|(a, b)| a - b)
        .collect();
    let e_large: Vec<f64> = y
        .iter()
        .zip(yhat_large.iter())
        .map(|(a, b)| a - b)
        .collect();

    let res = cw_test(&e_small, &e_large, &yhat_small, &yhat_large, 0).unwrap();
    assert!(
        res.mean_adj_diff > 0.0,
        "adjusted differential favours large model"
    );
    assert!(
        res.cw_stat > 1.64,
        "should reject at 5% one-sided, got {}",
        res.cw_stat
    );
    assert!(res.p_value < 0.05);
    // p-value is the upper-tail normal survival: monotone-decreasing in stat.
    assert!(res.p_value > 0.0 && res.p_value < 0.5);
}

#[test]
fn predability_guardrails_teach() {
    let a = prand(40, 1);
    let b = prand(40, 2);

    // Clark-West: length mismatch across the four series.
    assert!(matches!(
        cw_test(&a, &b[..39], &a, &b, 0).unwrap_err(),
        ForecastError::LengthMismatch { .. }
    ));
    // Non-finite input.
    let mut nan = a.clone();
    nan[4] = f64::NAN;
    assert!(matches!(
        cw_test(&nan, &b, &a, &b, 0).unwrap_err(),
        ForecastError::NonFinite { index: 4, .. }
    ));
    // Too short.
    assert!(matches!(
        cw_test(&a[..1], &b[..1], &a[..1], &b[..1], 0).unwrap_err(),
        ForecastError::SeriesTooShort { .. }
    ));
    // lrv_lags >= n.
    let err = cw_test(&a, &b, &a, &b, 40).unwrap_err();
    assert!(matches!(
        err,
        ForecastError::InvalidLrvLags {
            lags: 40,
            n: 40,
            ..
        }
    ));
    assert!(format!("{err}").contains("lag truncation"));

    // Clark-West with identical forecasts => constant differential, zero LRV.
    assert!(matches!(
        cw_test(&a, &a, &b, &b, 1).unwrap_err(),
        ForecastError::SingularWaldCovariance { q: 1 }
    ));

    // Giacomini-White unconditional: loss length mismatch.
    assert!(matches!(
        gw_test(&a, &b[..39], 0).unwrap_err(),
        ForecastError::LengthMismatch { .. }
    ));

    // Conditional: empty test functions, and a row-count mismatch.
    assert_eq!(
        gw_test_conditional(&a, &b, &[], 0).unwrap_err(),
        ForecastError::EmptyTestFunctions
    );
    let short_h = vec![vec![1.0]; 39];
    assert!(matches!(
        gw_test_conditional(&a, &b, &short_h, 0).unwrap_err(),
        ForecastError::LengthMismatch { .. }
    ));

    // Conditional: a degenerate (constant-zero) instrument column gives Shat
    // a zero row/column, so it is singular and the Wald form is undefined.
    let degenerate = vec![vec![1.0, 0.0]; a.len()];
    assert!(matches!(
        gw_test_conditional(&a, &b, &degenerate, 1).unwrap_err(),
        ForecastError::SingularWaldCovariance { q: 2 }
    ));
}
