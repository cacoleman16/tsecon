//! Seeded Monte-Carlo and closed-form property tests for
//! `unobserved_components` and `tvp_regression` — the statistical claims a
//! golden match cannot prove: parameter recovery at large T, forecast-
//! interval coverage, the pile-up frequency of the TVP variance estimator
//! on constant versus moving coefficients, and the exact identities
//! (scale invariance, the recursive-least-squares limit, the zero-sum
//! deterministic seasonal, missing-data variance widening). Randomness
//! comes from the tests' own seeded LCG; every number quoted in the
//! model card comes from these tests at these seeds.

mod common;

use common::Lcg;
use tsecon_ssm::{tvp_regression, unobserved_components, SsmError, TrendSpec, TvpOptions, UcOptions, UcSpec};

fn normal(rng: &mut Lcg) -> f64 {
    let u1 = rng.uniform().max(1e-300);
    let u2 = rng.uniform();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

fn simulate_local_level(rng: &mut Lcg, n: usize, s2_eps: f64, s2_eta: f64) -> (Vec<f64>, Vec<f64>) {
    let mut level = 10.0;
    let mut y = Vec::with_capacity(n);
    let mut mu = Vec::with_capacity(n);
    for _ in 0..n {
        mu.push(level);
        y.push(level + s2_eps.sqrt() * normal(rng));
        level += s2_eta.sqrt() * normal(rng);
    }
    (y, mu)
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

/// Parameter recovery: local level with sigma2_eps = 1, sigma2_eta = 0.3
/// at T = 1200 over 12 seeds — the median relative error of each variance
/// is small and every estimate is within the sampling band.
#[test]
fn local_level_mle_recovers_the_variances_at_large_t() {
    let (s2_eps, s2_eta) = (1.0, 0.3);
    let mut err_eps = Vec::new();
    let mut err_eta = Vec::new();
    for seed in 0..12u64 {
        let mut rng = Lcg::new(1000 + seed);
        let (y, _) = simulate_local_level(&mut rng, 1200, s2_eps, s2_eta);
        let fit = unobserved_components(&y, &UcSpec::default(), &UcOptions::default()).unwrap();
        assert!(fit.converged, "seed {seed}: not converged");
        assert!(fit.at_boundary.iter().all(|b| !b), "seed {seed}: spurious boundary");
        err_eps.push(fit.params[0] / s2_eps - 1.0);
        err_eta.push(fit.params[1] / s2_eta - 1.0);
        assert!(
            (fit.params[0] / s2_eps - 1.0).abs() < 0.25 && (fit.params[1] / s2_eta - 1.0).abs() < 0.4,
            "seed {seed}: {:?}",
            fit.params
        );
        assert!(fit.se.iter().all(|s| s.is_finite() && *s > 0.0));
    }
    let mut abs_eps: Vec<f64> = err_eps.iter().map(|e| e.abs()).collect();
    let mut abs_eta: Vec<f64> = err_eta.iter().map(|e| e.abs()).collect();
    let (m_eps, m_eta) = (median(&mut abs_eps), median(&mut abs_eta));
    let bias_eps = err_eps.iter().sum::<f64>() / err_eps.len() as f64;
    let bias_eta = err_eta.iter().sum::<f64>() / err_eta.len() as f64;
    eprintln!(
        "local level recovery T=1200, 12 seeds: median |rel err| eps {m_eps:.4} eta {m_eta:.4}; mean rel bias eps {bias_eps:.4} eta {bias_eta:.4}"
    );
    assert!(m_eps < 0.08 && m_eta < 0.15);
}

/// Forecast-interval coverage: local linear trend DGP, T = 150 in-sample,
/// 12 periods ahead, 80 seeds; the nominal 95% interval `forecast +/-
/// 1.96 sqrt(forecast_var)` covers the realized future at horizons 1, 6
/// and 12 within a Monte-Carlo band of the nominal rate.
#[test]
fn forecast_intervals_cover_at_the_nominal_rate() {
    let (s2_eps, s2_eta, s2_zeta) = (1.0, 0.1, 0.01);
    let (n, h, reps) = (150usize, 12usize, 80u64);
    let mut hits = vec![0usize; h];
    for seed in 0..reps {
        let mut rng = Lcg::new(5000 + seed);
        let (mut level, mut slope) = (5.0, 0.05);
        let mut y = Vec::with_capacity(n + h);
        for _ in 0..(n + h) {
            y.push(level + s2_eps.sqrt() * normal(&mut rng));
            level += slope + s2_eta.sqrt() * normal(&mut rng);
            slope += s2_zeta.sqrt() * normal(&mut rng);
        }
        let spec = UcSpec {
            trend: TrendSpec::LocalLinearTrend,
            ..UcSpec::default()
        };
        let fit = unobserved_components(
            &y[..n],
            &spec,
            &UcOptions {
                forecast_steps: h,
                ..UcOptions::default()
            },
        )
        .unwrap();
        for k in 0..h {
            let half = 1.959964 * fit.forecast_var[k].sqrt();
            assert!(fit.forecast_var[k] > 0.0);
            if (y[n + k] - fit.forecast[k]).abs() <= half {
                hits[k] += 1;
            }
        }
        if seed > 0 {
            // Variances grow with the horizon.
            assert!(fit.forecast_var.windows(2).all(|w| w[1] > w[0]));
        }
    }
    let cov: Vec<f64> = hits.iter().map(|&c| c as f64 / reps as f64).collect();
    eprintln!("forecast coverage (nominal 0.95), 80 seeds, h=1..12: {cov:?}");
    for &k in &[0usize, 5, 11] {
        assert!(cov[k] > 0.86 && cov[k] < 0.995, "h={} coverage {}", k + 1, cov[k]);
    }
    let pooled = hits.iter().sum::<usize>() as f64 / (reps as usize * h) as f64;
    assert!(pooled > 0.90 && pooled < 0.98, "pooled coverage {pooled}");
}

fn simulate_tvp(rng: &mut Lcg, n: usize, q: &[f64], s2_eps: f64) -> (Vec<f64>, Vec<Vec<f64>>) {
    let k = q.len();
    let mut beta: Vec<f64> = (0..k).map(|j| 1.0 - j as f64).collect();
    let mut x: Vec<Vec<f64>> = vec![Vec::with_capacity(n); k - 1];
    let mut y = Vec::with_capacity(n);
    for _ in 0..n {
        let mut yt = beta[0];
        for j in 1..k {
            let xj = normal(rng);
            x[j - 1].push(xj);
            yt += beta[j] * xj;
        }
        y.push(yt + s2_eps.sqrt() * normal(rng));
        for j in 0..k {
            beta[j] += q[j].sqrt() * normal(rng);
        }
    }
    (y, x)
}

/// Pile-up: with constant true coefficients the TVP variance MLE lands on
/// zero in a large share of samples (Stock & Watson 1998) and is flagged;
/// with clearly moving coefficients it (almost) never is.
#[test]
fn tvp_pile_up_flags_constant_coefficients_and_not_moving_ones() {
    let (n, seeds) = (200usize, 24u64);
    let mut flagged_const = 0usize;
    let mut flagged_moving = 0usize;
    let mut total = 0usize;
    for seed in 0..seeds {
        let mut rng = Lcg::new(9000 + seed);
        let (y, x) = simulate_tvp(&mut rng, n, &[0.0, 0.0], 1.0);
        let fit = tvp_regression(&y, &x, &TvpOptions::default()).unwrap();
        flagged_const += fit.pile_up.iter().filter(|b| **b).count();
        let mut rng2 = Lcg::new(9500 + seed);
        let (y2, x2) = simulate_tvp(&mut rng2, n, &[0.05, 0.05], 1.0);
        let fit2 = tvp_regression(&y2, &x2, &TvpOptions::default()).unwrap();
        flagged_moving += fit2.pile_up.iter().filter(|b| **b).count();
        total += 2;
        // The estimated variances are in the right order of magnitude
        // when the coefficients move.
        assert!(fit2.sigma2_beta.iter().all(|&v| v > 0.005 && v < 0.3), "seed {seed}: {:?}", fit2.sigma2_beta);
    }
    let (fc, fm) = (flagged_const as f64 / total as f64, flagged_moving as f64 / total as f64);
    eprintln!("tvp pile-up share: constant coefficients {fc:.3}, moving (q=0.05) {fm:.3} over {total} coefficients");
    assert!(fc >= 0.5, "constant coefficients should pile up at zero often: {fc}");
    assert!(fm <= 0.1, "moving coefficients should rarely be flagged: {fm}");
}

/// Closed form: with every state variance zero the filtered coefficient
/// at `t` is the OLS estimate on observations `0..=t` (recursive least
/// squares), once `t >= k`.
#[test]
fn tvp_zero_state_variance_is_expanding_window_ols() {
    let mut rng = Lcg::new(31);
    let (y, x) = simulate_tvp(&mut rng, 60, &[0.0, 0.0, 0.0], 0.5);
    let fit = tvp_regression(
        &y,
        &x,
        &TvpOptions {
            fixed_params: Some(vec![0.5, 0.0, 0.0, 0.0]),
            ..TvpOptions::default()
        },
    )
    .unwrap();
    assert_eq!(fit.nobs_diffuse, 3);
    for &t in &[3usize, 10, 30, 59] {
        // OLS on rows 0..=t by normal equations.
        let k = 3;
        let mut xtx = vec![vec![0.0; k]; k];
        let mut xty = vec![0.0; k];
        for r in 0..=t {
            let row = [1.0, x[0][r], x[1][r]];
            for i in 0..k {
                xty[i] += row[i] * y[r];
                for j in 0..k {
                    xtx[i][j] += row[i] * row[j];
                }
            }
        }
        // 3x3 solve by Gauss-Jordan.
        let mut a: Vec<Vec<f64>> = xtx.iter().zip(&xty).map(|(row, b)| {
            let mut r = row.clone();
            r.push(*b);
            r
        }).collect();
        for c in 0..k {
            let p = (c..k).max_by(|&i, &j| a[i][c].abs().total_cmp(&a[j][c].abs())).unwrap();
            a.swap(c, p);
            let d = a[c][c];
            for v in a[c].iter_mut() {
                *v /= d;
            }
            let pr = a[c].clone();
            for r in 0..k {
                if r != c {
                    let f = a[r][c];
                    for (v, pv) in a[r].iter_mut().zip(&pr) {
                        *v -= f * pv;
                    }
                }
            }
        }
        for i in 0..k {
            let ols = a[i][k];
            assert!(
                (fit.beta_filtered[t][i] - ols).abs() <= 1e-8 * ols.abs().max(1.0),
                "t={t} coef {i}: {} vs OLS {ols}",
                fit.beta_filtered[t][i]
            );
        }
    }
}

/// Exact scale invariance of the fit: `c * y` gives variances times
/// `c^2`, the level path times `c`, the log-likelihood minus `n ln c`,
/// and identical standardized residuals — bit-for-bit up to the exact
/// rescaling, because the search runs on the standardized series.
#[test]
fn fit_is_invariant_to_the_scale_of_y() {
    let mut rng = Lcg::new(77);
    let (y, _) = simulate_local_level(&mut rng, 150, 2.0, 0.5);
    let c = 1000.0;
    let yc: Vec<f64> = y.iter().map(|v| v * c).collect();
    let spec = UcSpec {
        trend: TrendSpec::LocalLinearTrend,
        ..UcSpec::default()
    };
    let opts = UcOptions {
        forecast_steps: 4,
        ..UcOptions::default()
    };
    let a = unobserved_components(&y, &spec, &opts).unwrap();
    let b = unobserved_components(&yc, &spec, &opts).unwrap();
    let close = |u: f64, v: f64| (u - v).abs() <= 1e-9 * v.abs().max(1e-300);
    for i in 0..a.params.len() {
        assert!(close(b.params[i] * 1.0, a.params[i] * c * c), "param {i}");
        if a.se[i].is_finite() {
            assert!(close(b.se[i], a.se[i] * c * c), "se {i}");
        }
    }
    assert!((b.loglik - (a.loglik - 150.0 * c.ln())).abs() < 1e-7 * a.loglik.abs());
    for t in 0..150 {
        assert!(close(b.level.as_ref().unwrap().smoothed[t], a.level.as_ref().unwrap().smoothed[t] * c));
        if a.std_resid[t].is_finite() {
            assert!((a.std_resid[t] - b.std_resid[t]).abs() < 1e-9);
        }
    }
    for k in 0..4 {
        assert!(close(b.forecast[k], a.forecast[k] * c));
        assert!(close(b.forecast_var[k], a.forecast_var[k] * c * c));
    }
    assert_eq!(a.at_boundary, b.at_boundary);
}

/// A deterministic dummy seasonal (zero seasonal variance) sums to zero
/// over any full period, in both the filtered (after the diffuse period)
/// and smoothed paths.
#[test]
fn deterministic_dummy_seasonal_sums_to_zero_over_a_period() {
    let mut rng = Lcg::new(5);
    let n = 96;
    let pattern = [1.5, -0.5, -1.5, 0.5];
    let mut y = Vec::with_capacity(n);
    let mut level = 3.0;
    for t in 0..n {
        y.push(level + pattern[t % 4] + normal(&mut rng));
        level += 0.3 * normal(&mut rng);
    }
    let spec = UcSpec {
        trend: TrendSpec::LocalLevel,
        seasonal: Some(4),
        stochastic_seasonal: false,
        ..UcSpec::default()
    };
    let fit = unobserved_components(&y, &spec, &UcOptions::default()).unwrap();
    let s = fit.seasonal.as_ref().unwrap();
    let amp: f64 = s.smoothed.iter().map(|v| v.abs()).fold(0.0, f64::max);
    for t in fit.nobs_diffuse..(n - 4) {
        let sm: f64 = s.smoothed[t..t + 4].iter().sum();
        assert!(sm.abs() < 1e-8 * amp, "smoothed seasonal sum at {t}: {sm}");
        let fl: f64 = s.filtered[t..t + 4].iter().sum();
        assert!(fl.abs() < 1e-6 * amp.max(1.0), "filtered seasonal sum at {t}: {fl}");
    }
    // The estimated pattern recovers the truth (zero-mean version).
    let mean: f64 = pattern.iter().sum::<f64>() / 4.0;
    for j in 0..4 {
        let est = s.smoothed[n - 4 + j];
        assert!((est - (pattern[j] - mean)).abs() < 0.5, "pattern {j}: {est}");
    }
}

/// Inside a gap of missing observations the smoothed variance is larger
/// than outside, and the smoothed path bridges the gap.
#[test]
fn missing_gap_widens_the_smoothed_variance() {
    let mut rng = Lcg::new(11);
    let (mut y, _) = simulate_local_level(&mut rng, 120, 1.0, 0.3);
    for v in y.iter_mut().take(65).skip(45) {
        *v = f64::NAN;
    }
    let fit = unobserved_components(&y, &UcSpec::default(), &UcOptions::default()).unwrap();
    assert_eq!(fit.nobs_observed, 100);
    let lv = fit.level.as_ref().unwrap();
    assert!(lv.smoothed_var[55] > 2.0 * lv.smoothed_var[30]);
    assert!(lv.smoothed_var[55] > 2.0 * lv.smoothed_var[90]);
    assert!(lv.smoothed[55].is_finite() && fit.resid[55].is_nan() && fit.std_resid[55].is_nan());
    assert!(fit.fitted[55].is_finite());
}

/// The cycle frequency lands inside the requested period bounds.
#[test]
fn cycle_period_bounds_confine_the_estimated_frequency() {
    let mut rng = Lcg::new(2024);
    let n = 240;
    let (lam, rho) = (2.0 * std::f64::consts::PI / 8.0, 0.92);
    let (mut c, mut cs) = (1.0, 0.0);
    let mut level = 0.0;
    let mut y = Vec::with_capacity(n);
    for _ in 0..n {
        y.push(level + c + 0.5 * normal(&mut rng));
        let (nc, ncs) = (
            rho * (lam.cos() * c + lam.sin() * cs) + 0.3 * normal(&mut rng),
            rho * (-lam.sin() * c + lam.cos() * cs) + 0.3 * normal(&mut rng),
        );
        c = nc;
        cs = ncs;
        level += 0.1 * normal(&mut rng);
    }
    let spec = UcSpec {
        cycle: true,
        damped_cycle: true,
        stochastic_cycle: true,
        cycle_period_bounds: (4.0, 20.0),
        ..UcSpec::default()
    };
    let fit = unobserved_components(&y, &spec, &UcOptions::default()).unwrap();
    let i = fit.param_names.iter().position(|p| p == "frequency.cycle").unwrap();
    let f = fit.params[i];
    let (lo, hi) = (2.0 * std::f64::consts::PI / 20.0, 2.0 * std::f64::consts::PI / 4.0);
    assert!(f > lo && f < hi, "frequency {f} outside ({lo}, {hi})");
    assert!((2.0 * std::f64::consts::PI / f - 8.0).abs() < 1.5, "period {}", 2.0 * std::f64::consts::PI / f);
    let d = fit.params[fit.param_names.iter().position(|p| p == "damping.cycle").unwrap()];
    assert!(d > 0.7 && d < 1.0, "damping {d}");
    assert!(fit.cycle.is_some());
}

/// Every refusal names the offending argument.
#[test]
fn refusals_name_the_offending_argument() {
    let mut rng = Lcg::new(1);
    let (y, _) = simulate_local_level(&mut rng, 40, 1.0, 0.3);
    let msg = |r: Result<tsecon_ssm::UcFit, SsmError>| -> String {
        match r {
            Ok(_) => panic!("expected a refusal"),
            Err(e) => e.to_string(),
        }
    };
    let spec = |f: &dyn Fn(&mut UcSpec)| {
        let mut s = UcSpec::default();
        f(&mut s);
        s
    };
    let cases: Vec<(String, &str)> = vec![
        (msg(unobserved_components(&[], &UcSpec::default(), &UcOptions::default())), "y is empty"),
        (msg(unobserved_components(&[1.0, f64::INFINITY, 2.0], &UcSpec::default(), &UcOptions::default())), "y contains an infinity"),
        (msg(unobserved_components(&y, &spec(&|s| s.seasonal = Some(1)), &UcOptions::default())), "seasonal = 1"),
        (
            msg(unobserved_components(
                &y,
                &spec(&|s| s.freq_seasonal = vec![tsecon_ssm::FreqSeasonalSpec { period: 12.0, harmonics: 7, stochastic: true }]),
                &UcOptions::default(),
            )),
            "freq_seasonal[0].harmonics = 7",
        ),
        (
            msg(unobserved_components(&y, &spec(&|s| s.freq_seasonal = vec![tsecon_ssm::FreqSeasonalSpec::new(1.5)]), &UcOptions::default())),
            "freq_seasonal[0].period = 1.5",
        ),
        (
            msg(unobserved_components(&y, &spec(&|s| { s.cycle = true; s.cycle_period_bounds = (1.0, 10.0); }), &UcOptions::default())),
            "cycle_period_bounds = (1, 10)",
        ),
        (msg(unobserved_components(&y, &spec(&|s| s.exog = vec![vec![1.0; 39]]), &UcOptions::default())), "exog column 0 has length 39"),
        (msg(unobserved_components(&y, &spec(&|s| s.exog = vec![vec![0.0; 40]]), &UcOptions::default())), "exog column 0 is identically zero"),
        (msg(unobserved_components(&y, &spec(&|s| s.trend = TrendSpec::FixedIntercept), &UcOptions::default())), "level = \"fixed intercept\""),
        (
            msg(unobserved_components(&y, &UcSpec::default(), &UcOptions { forecast_exog: vec![vec![1.0; 3]], ..UcOptions::default() })),
            "forecast_exog was given but forecast_steps = 0",
        ),
        (
            msg(unobserved_components(&y, &UcSpec::default(), &UcOptions { forecast_steps: 3, forecast_exog: vec![vec![1.0; 3]], ..UcOptions::default() })),
            "forecast_exog was given but the model has no regressors",
        ),
        (
            msg(unobserved_components(&y, &spec(&|s| s.exog = vec![(0..40).map(|t| t as f64).collect()]), &UcOptions { forecast_steps: 3, ..UcOptions::default() })),
            "forecast_steps = 3 with 1 regressors requires forecast_exog",
        ),
        (
            msg(unobserved_components(&y, &UcSpec::default(), &UcOptions { fixed_params: Some(vec![1.0]), ..UcOptions::default() })),
            "fixed_params has length 1 but the specification has 2 parameters",
        ),
        (
            msg(unobserved_components(&y, &UcSpec::default(), &UcOptions { fixed_params: Some(vec![1.0, -1.0]), ..UcOptions::default() })),
            "fixed_params[1] (sigma2.level) = -1",
        ),
        (msg(unobserved_components(&y, &UcSpec::default(), &UcOptions { n_starts: 0, ..UcOptions::default() })), "n_starts = 0"),
        (msg(unobserved_components(&[1.0, 1.0, 1.0, 1.0, 1.0], &UcSpec::default(), &UcOptions::default())), "y is constant"),
        (msg(unobserved_components(&[1.0, f64::NAN, 2.0], &UcSpec::default(), &UcOptions::default())), "observed (non-NaN) values"),
    ];
    for (m, needle) in &cases {
        assert!(m.contains(needle), "message {m:?} does not name {needle:?}");
    }
    assert!(TrendSpec::parse("bogus").unwrap_err().to_string().contains("level = \"bogus\""));

    let tmsg = |r: Result<tsecon_ssm::TvpFit, SsmError>| -> String {
        match r {
            Ok(_) => panic!("expected a refusal"),
            Err(e) => e.to_string(),
        }
    };
    let x = vec![(0..40).map(|t| (t as f64).sin()).collect::<Vec<f64>>()];
    let tcases: Vec<(String, &str)> = vec![
        (tmsg(tvp_regression(&y, &[vec![1.0; 39]], &TvpOptions::default())), "x column 0 has length 39"),
        (tmsg(tvp_regression(&y, &[vec![f64::NAN; 40]], &TvpOptions::default())), "x column 0 contains a NaN"),
        (tmsg(tvp_regression(&y, &[], &TvpOptions { constant: false, ..TvpOptions::default() })), "x has no columns and constant = false"),
        (tmsg(tvp_regression(&y, &x, &TvpOptions { fixed_params: Some(vec![1.0, 0.0]), ..TvpOptions::default() })), "fixed_params has length 2"),
        (tmsg(tvp_regression(&y, &x, &TvpOptions { fixed_params: Some(vec![0.0, 0.0, 0.0]), ..TvpOptions::default() })), "fixed_params[0] = 0 (sigma2_eps)"),
        (tmsg(tvp_regression(&y, &x, &TvpOptions { fixed_params: Some(vec![1.0, -0.1, 0.0]), ..TvpOptions::default() })), "fixed_params[1] = -0.1"),
        (tmsg(tvp_regression(&y[..3], &[x[0][..3].to_vec()], &TvpOptions::default())), "at least k + 2"),
        (tmsg(tvp_regression(&y, &x, &TvpOptions { n_starts: 0, ..TvpOptions::default() })), "n_starts = 0"),
    ];
    for (m, needle) in &tcases {
        assert!(m.contains(needle), "message {m:?} does not name {needle:?}");
    }
}
