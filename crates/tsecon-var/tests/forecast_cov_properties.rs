//! Properties of the parameter-uncertainty forecast covariance
//! (`ForecastOptions::with_param_uncertainty`, Lütkepohl 2005 section 3.5)
//! beyond the statsmodels golden in `forecast_cov_golden.rs`:
//!
//! * the default options are bit-identical to the pre-existing
//!   `forecast_cov` / `forecast_interval` (same-run comparison, portable);
//! * `cov(h) - MSE(h) = Omega(h)/T` is symmetric positive semi-definite at
//!   every horizon, so the bands only widen;
//! * the term is `O(1/T)`: the `cov/MSE` ratio falls toward 1 as `T` grows;
//! * an intercept-only VAR(0) has the closed form `cov(h) = (1 + 1/T) Sigma_u`;
//! * shorter requests reproduce the prefix of longer ones bit for bit;
//! * the cubic work is budgeted and the refusal names `steps`;
//! * on a seeded Monte Carlo the widened band covers more often than the
//!   plug-in band on the same draws.

mod common;

use common::Lcg;
use tsecon_linalg::faer::{Mat, Side};
use tsecon_var::{ForecastOptions, Trend, VarError, VarResults, VarSpec};

fn simulate_var1(a: [[f64; 2]; 2], t: usize, burn: usize, lcg: &mut Lcg) -> Mat<f64> {
    let mut y = Mat::<f64>::zeros(t, 2);
    let mut state = [0.0_f64; 2];
    for s in 0..t + burn {
        let e0 = lcg.gaussian();
        let e1 = 0.4 * e0 + (1.0_f64 - 0.16).sqrt() * lcg.gaussian();
        let next = [
            a[0][0] * state[0] + a[0][1] * state[1] + e0,
            a[1][0] * state[0] + a[1][1] * state[1] + e1,
        ];
        state = next;
        if s >= burn {
            y[(s - burn, 0)] = state[0];
            y[(s - burn, 1)] = state[1];
        }
    }
    y
}

const A: [[f64; 2]; 2] = [[0.70, 0.15], [0.10, 0.60]];

fn fit(y: &Mat<f64>, lags: usize, trend: Trend) -> VarResults {
    VarSpec::new(lags, trend).unwrap().fit(y.as_ref()).unwrap()
}

fn with() -> ForecastOptions {
    ForecastOptions::new().with_param_uncertainty(true)
}

#[test]
fn default_options_are_bitwise_the_existing_covariance_and_interval() {
    let mut lcg = Lcg::new(11);
    let y = simulate_var1(A, 150, 50, &mut lcg);
    for (lags, trend) in [(1, Trend::Constant), (3, Trend::None), (2, Trend::Constant)] {
        let res = fit(&y, lags, trend);
        let old = res.forecast_cov(10).unwrap();
        let new = res
            .forecast_cov_with(10, ForecastOptions::default())
            .unwrap();
        for (a, b) in old.iter().zip(&new) {
            for j in 0..2 {
                for i in 0..2 {
                    assert_eq!(a[(i, j)].to_bits(), b[(i, j)].to_bits());
                }
            }
        }
        let fi = res.forecast_interval(10, 0.1).unwrap();
        let fw = res
            .forecast_interval_with(10, 0.1, ForecastOptions::new())
            .unwrap();
        for h in 0..10 {
            for j in 0..2 {
                for (a, b) in [
                    (fi.point[(h, j)], fw.point[(h, j)]),
                    (fi.lower[(h, j)], fw.lower[(h, j)]),
                    (fi.upper[(h, j)], fw.upper[(h, j)]),
                    (fi.se[(h, j)], fw.se[(h, j)]),
                ] {
                    assert_eq!(a.to_bits(), b.to_bits());
                }
            }
        }
    }
}

#[test]
fn the_added_term_is_symmetric_psd_and_only_widens_the_band() {
    let mut lcg = Lcg::new(23);
    let y = simulate_var1(A, 120, 50, &mut lcg);
    for (lags, trend) in [(1, Trend::Constant), (2, Trend::None), (4, Trend::Constant)] {
        let res = fit(&y, lags, trend);
        let mse = res.forecast_cov(16).unwrap();
        let cov = res.forecast_cov_with(16, with()).unwrap();
        let plain = res.forecast_interval(16, 0.05).unwrap();
        let wide = res.forecast_interval_with(16, 0.05, with()).unwrap();
        for h in 0..16 {
            let d = &cov[h] - &mse[h];
            assert_eq!(d[(0, 1)].to_bits(), d[(1, 0)].to_bits(), "h={h} symmetric");
            // PSD: a tiny ridge must make it Cholesky-factorable.
            let ridge = 1e-12 * (d[(0, 0)] + d[(1, 1)]).abs().max(1e-300);
            let shifted = Mat::from_fn(2, 2, |i, j| d[(i, j)] + if i == j { ridge } else { 0.0 });
            assert!(
                shifted.llt(Side::Lower).is_ok(),
                "VAR({lags}) h={h}: Omega/T not PSD: {d:?}"
            );
            for j in 0..2 {
                assert!(d[(j, j)] > 0.0, "h={h}: diagonal term must be positive");
                assert!(wide.lower[(h, j)] < plain.lower[(h, j)]);
                assert!(wide.upper[(h, j)] > plain.upper[(h, j)]);
                assert_eq!(wide.point[(h, j)].to_bits(), plain.point[(h, j)].to_bits());
            }
        }
    }
}

/// `Omega(h)/T` is a first-order `O(1/T)` term: with sixteen times the data
/// the relative inflation of the h = 8 variance must shrink by roughly that
/// factor (asserted: by more than four).
#[test]
fn the_term_vanishes_at_rate_one_over_t() {
    let mut lcg = Lcg::new(5);
    let y_long = simulate_var1(A, 3200, 100, &mut lcg);
    let y_short = Mat::from_fn(200, 2, |i, j| y_long[(i, j)]);
    let inflation = |y: &Mat<f64>| {
        let res = fit(y, 1, Trend::Constant);
        let mse = res.forecast_cov(8).unwrap();
        let cov = res.forecast_cov_with(8, with()).unwrap();
        cov[7][(0, 0)] / mse[7][(0, 0)] - 1.0
    };
    let short = inflation(&y_short);
    let long = inflation(&y_long);
    assert!(short > 0.0 && long > 0.0);
    assert!(
        long < short / 4.0,
        "inflation T=200: {short:.5}, T=3200: {long:.5}"
    );
}

/// An intercept-only VAR(0) forecasts the sample mean: MSE(h) = Sigma_u at
/// every horizon and the estimation error of the mean adds Sigma_u / T.
#[test]
fn intercept_only_var_has_the_sample_mean_closed_form() {
    let mut lcg = Lcg::new(77);
    let y = simulate_var1(A, 90, 50, &mut lcg);
    let res = fit(&y, 0, Trend::Constant);
    let t = res.nobs as f64;
    let cov = res.forecast_cov_with(6, with()).unwrap();
    for c in &cov {
        for j in 0..2 {
            for i in 0..2 {
                let want = (1.0 + 1.0 / t) * res.sigma_u[(i, j)];
                assert!((c[(i, j)] - want).abs() <= 1e-14 * res.sigma_u[(0, 0)].abs());
            }
        }
    }
}

#[test]
fn shorter_requests_reproduce_the_prefix_bitwise() {
    let mut lcg = Lcg::new(91);
    let y = simulate_var1(A, 140, 50, &mut lcg);
    let res = fit(&y, 2, Trend::Constant);
    let long = res.forecast_cov_with(15, with()).unwrap();
    for steps in [1, 4, 9] {
        let short = res.forecast_cov_with(steps, with()).unwrap();
        for (a, b) in short.iter().zip(&long) {
            for j in 0..2 {
                for i in 0..2 {
                    assert_eq!(a[(i, j)].to_bits(), b[(i, j)].to_bits(), "steps={steps}");
                }
            }
        }
    }
}

#[test]
fn the_cubic_work_is_budgeted_and_the_refusal_names_steps() {
    let mut lcg = Lcg::new(3);
    let y = simulate_var1(A, 100, 50, &mut lcg);
    let res = fit(&y, 1, Trend::Constant);
    assert!(res.forecast_cov_with(400, with()).is_ok());
    for steps in [1_000_000, usize::MAX / 2] {
        match res.forecast_cov_with(steps, with()) {
            Err(e @ VarError::InvalidParameter { name: "steps", .. }) => {
                let msg = e.to_string();
                assert!(msg.contains("steps") && msg.contains("CUBIC"), "{msg}");
            }
            other => panic!("steps={steps}: expected a budget refusal, got {other:?}"),
        }
    }
    // The direct entry point refuses the same way.
    assert!(matches!(
        res.forecast_param_uncertainty(1_000_000),
        Err(VarError::InvalidParameter { name: "steps", .. })
    ));
    // Without the term nothing is budgeted beyond what forecast_cov does.
    assert!(res.forecast_cov_with(5000, ForecastOptions::new()).is_ok());
    assert!(matches!(
        res.forecast_cov_with(0, with()),
        Err(VarError::InvalidArgument { .. })
    ));
}

/// Same draws, two bands: the widened band covers at least as often in every
/// cell and strictly more often pooled. 600 replications of the
/// interval-coverage audit's VAR(1) at T = 60 (smaller than the audit's 100
/// so the plug-in shortfall is large enough to see in a debug-build test).
#[test]
fn the_widened_band_covers_more_often_on_the_same_draws() {
    let reps = 600;
    let (t, h) = (60, 8);
    let mut lcg = Lcg::new(20261008);
    let mut hits_plain = 0usize;
    let mut hits_param = 0usize;
    for _ in 0..reps {
        let y = simulate_var1(A, t + h, 100, &mut lcg);
        let train = Mat::from_fn(t, 2, |i, j| y[(i, j)]);
        let res = fit(&train, 1, Trend::Constant);
        let plain = res.forecast_interval(h, 0.05).unwrap();
        let wide = res.forecast_interval_with(h, 0.05, with()).unwrap();
        for s in 0..h {
            for j in 0..2 {
                let v = y[(t + s, j)];
                let a = plain.lower[(s, j)] <= v && v <= plain.upper[(s, j)];
                let b = wide.lower[(s, j)] <= v && v <= wide.upper[(s, j)];
                assert!(
                    !a || b,
                    "a plug-in hit must stay a hit under the wider band"
                );
                hits_plain += usize::from(a);
                hits_param += usize::from(b);
            }
        }
    }
    let cells = (reps * h * 2) as f64;
    let (cp, cw) = (hits_plain as f64 / cells, hits_param as f64 / cells);
    eprintln!("pooled coverage T={t}: plug-in {cp:.4}, with Omega/T {cw:.4}");
    assert!(cw > cp + 0.004, "plug-in {cp:.4} vs widened {cw:.4}");
    assert!(
        cw < 0.96,
        "the widened band is not meant to over-cover: {cw:.4}"
    );
}
