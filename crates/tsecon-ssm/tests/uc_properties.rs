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
use tsecon_linalg::faer::Mat;
use tsecon_ssm::{
    tvp_regression, unobserved_components, FreqSeasonalSpec, Initialization, LinearGaussianSSM,
    SsmError, TrendSpec, TvpOptions, UcOptions, UcSpec,
};

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
        assert!(
            fit.at_boundary.iter().all(|b| !b),
            "seed {seed}: spurious boundary"
        );
        err_eps.push(fit.params[0] / s2_eps - 1.0);
        err_eta.push(fit.params[1] / s2_eta - 1.0);
        assert!(
            (fit.params[0] / s2_eps - 1.0).abs() < 0.25
                && (fit.params[1] / s2_eta - 1.0).abs() < 0.4,
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
    let (s2_eps, s2_eta, s2_zeta): (f64, f64, f64) = (1.0, 0.1, 0.01);
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
        assert!(
            cov[k] > 0.86 && cov[k] < 0.995,
            "h={} coverage {}",
            k + 1,
            cov[k]
        );
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
/// with clearly moving coefficients it (almost) never is. The recovery
/// claim is about the MEDIAN estimate across seeds: the MLE of a
/// random-walk variance is severely attenuated in individual samples of
/// this length — at `T = 200` with a true `0.05` the draws here run from
/// well under a hundredth of the truth to about twice it — which is the
/// estimator's documented weakness, not a defect.
#[test]
fn tvp_pile_up_flags_constant_coefficients_and_not_moving_ones() {
    let (n, seeds) = (200usize, 24u64);
    let mut flagged_const = 0usize;
    let mut flagged_moving = 0usize;
    let mut total = 0usize;
    let mut moving_est: Vec<f64> = Vec::new();
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
        // Every estimate is a positive, finite number; the MLE of a
        // random-walk variance is badly attenuated in samples of this
        // length (Stock & Watson 1998), so the recovery claim below is
        // about the MEDIAN across seeds, not about each draw.
        assert!(
            fit2.sigma2_beta.iter().all(|&v| v > 0.0 && v.is_finite()),
            "seed {seed}: {:?}",
            fit2.sigma2_beta
        );
        moving_est.extend(fit2.sigma2_beta.iter().copied());
    }
    let (fc, fm) = (
        flagged_const as f64 / total as f64,
        flagged_moving as f64 / total as f64,
    );
    let med = median(&mut moving_est.clone());
    let lo = moving_est.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = moving_est.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    eprintln!(
        "tvp pile-up share: constant coefficients {fc:.3}, moving (q=0.05) {fm:.3} over {total} coefficients; \
         moving estimates min {lo:.4} median {med:.4} max {hi:.4} (truth 0.05)"
    );
    assert!(
        med > 0.02 && med < 0.12,
        "median estimate of a true 0.05 random-walk variance: {med}"
    );
    assert!(
        fc >= 0.5,
        "constant coefficients should pile up at zero often: {fc}"
    );
    assert!(
        fm <= 0.1,
        "moving coefficients should rarely be flagged: {fm}"
    );
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
        let mut a: Vec<Vec<f64>> = xtx
            .iter()
            .zip(&xty)
            .map(|(row, b)| {
                let mut r = row.clone();
                r.push(*b);
                r
            })
            .collect();
        for c in 0..k {
            let p = (c..k)
                .max_by(|&i, &j| a[i][c].abs().total_cmp(&a[j][c].abs()))
                .unwrap();
            a.swap(c, p);
            let d = a[c][c];
            for v in a[c].iter_mut() {
                *v /= d;
            }
            let pr = a[c].clone();
            for (r, row) in a.iter_mut().enumerate() {
                if r != c {
                    let f = row[c];
                    for (v, pv) in row.iter_mut().zip(&pr) {
                        *v -= f * pv;
                    }
                }
            }
        }
        for (i, row) in a.iter().enumerate() {
            let ols = row[k];
            assert!(
                (fit.beta_filtered[t][i] - ols).abs() <= 1e-8 * ols.abs().max(1.0),
                "t={t} coef {i}: {} vs OLS {ols}",
                fit.beta_filtered[t][i]
            );
        }
    }
}

/// Scale equivariance of the whole fit: `c * y` gives variances times
/// `c^2`, the level path times `c`, the forecasts times `c`, the
/// log-likelihood minus `n ln c`, and identical standardized residuals.
/// The search itself runs on the standardized series, so this is close to
/// exact — but not bit-for-bit: `var(c y)` is not exactly `c^2 var(y)` in
/// floating point, so the two runs standardize by scales that differ in
/// the last ulp. The measured deviations at this seed are 2.9e-10 on the
/// estimates, 2.3e-8 on the observed-information standard errors (a
/// numerical Hessian, which amplifies that last ulp), 2.2e-15 on the
/// log-likelihood, 3.1e-11 on the level and forecast paths and 3.3e-10 on
/// the standardized residuals. A variance the likelihood cannot tell from
/// zero (flagged in
/// `at_boundary`, here `sigma2.trend` at ~1e-22) is compared on the scale
/// of the parameter vector, not to itself: the relative difference of two
/// numerical zeros is not a number about the estimator.
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
    let rel = |u: f64, v: f64| (u - v).abs() / v.abs().max(1e-300);
    // Floor for a parameter that is numerically zero: 1e-10 of the largest
    // estimate, in the rescaled units.
    let pscale = a.params.iter().map(|v| v.abs()).fold(0.0, f64::max) * c * c;
    let mut worst_param: f64 = 0.0;
    let mut worst_se: f64 = 0.0;
    for i in 0..a.params.len() {
        let want = a.params[i] * c * c;
        let denom = want.abs().max(1e-10 * pscale);
        worst_param = worst_param.max((b.params[i] - want).abs() / denom);
        if a.se[i].is_finite() {
            worst_se = worst_se.max(rel(b.se[i], a.se[i] * c * c));
        }
    }
    eprintln!(
        "scale equivariance (c = {c}): worst relative deviation — params {worst_param:e}, se {worst_se:e}"
    );
    // Under EXACT-DIFFUSE initialization the shift is `-(n - d) ln c`, not
    // `-n ln c`: each diffuse element contributes a scale-free
    // `-(ln 2 pi + ln F_inf) / 2` (`P_inf` never touches the data), so the
    // `d` diffuse periods do not move with the units of `y`. This is the
    // same identity `tests/scale.rs` pins on the local level.
    assert_eq!(a.nobs_diffuse, b.nobs_diffuse);
    assert_eq!(
        a.nobs_diffuse, 2,
        "two diffuse states, one resolved per step"
    );
    let shift = (150.0 - a.nobs_diffuse as f64) * c.ln();
    let ll_rel = (b.loglik - (a.loglik - shift)).abs() / a.loglik.abs();
    let mut worst_path: f64 = 0.0;
    let mut worst_sr: f64 = 0.0;
    // Paths are compared relative to their own amplitude, so a level that
    // happens to cross zero does not manufacture a relative error.
    let lv = a.level.as_ref().unwrap();
    let amp = lv.smoothed.iter().map(|v| v.abs()).fold(0.0, f64::max) * c;
    for t in 0..150 {
        let want = lv.smoothed[t] * c;
        worst_path = worst_path.max((b.level.as_ref().unwrap().smoothed[t] - want).abs() / amp);
        if a.std_resid[t].is_finite() {
            worst_sr = worst_sr.max((a.std_resid[t] - b.std_resid[t]).abs());
        }
    }
    for k in 0..4 {
        worst_path = worst_path.max((b.forecast[k] - a.forecast[k] * c).abs() / amp);
        worst_path =
            worst_path.max((b.forecast_var[k] - a.forecast_var[k] * c * c).abs() / (amp * amp));
    }
    eprintln!(
        "scale equivariance (c = {c}): loglik {ll_rel:e}, level/forecast paths {worst_path:e}, standardized residuals {worst_sr:e}"
    );
    assert!(worst_param <= 1e-9, "params: {worst_param:e}");
    assert!(worst_se <= 1e-6, "se: {worst_se:e}");
    assert!(ll_rel < 1e-12, "loglik: {ll_rel:e}");
    assert!(worst_path < 1e-9, "level / forecast paths: {worst_path:e}");
    assert!(worst_sr < 1e-9, "standardized residuals: {worst_sr:e}");
    assert_eq!(a.at_boundary, b.at_boundary);
    let _ = close;
}

/// A deterministic dummy seasonal (zero seasonal variance) is exactly
/// periodic and sums to zero over any full period — both are identities of
/// the *smoothed* path, which conditions every date on the same
/// information set `y_1..y_n`: `gamma_t + ... + gamma_{t-3} = 0` holds for
/// the states themselves, so it holds for their common conditional
/// expectation, and `gamma_{t+4} = gamma_t` follows. The *filtered* path
/// satisfies neither and is deliberately not asserted: `filtered[t]` and
/// `filtered[t+1]` condition on different samples, so four consecutive
/// filtered values are four estimates of a zero sum taken at four
/// different times, not an estimate of zero.
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
    let mut worst_sum: f64 = 0.0;
    let mut worst_period: f64 = 0.0;
    for t in 0..(n - 4) {
        let sm: f64 = s.smoothed[t..t + 4].iter().sum();
        worst_sum = worst_sum.max(sm.abs());
        assert!(sm.abs() < 1e-8 * amp, "smoothed seasonal sum at {t}: {sm}");
        let per = (s.smoothed[t + 4] - s.smoothed[t]).abs();
        worst_period = worst_period.max(per);
        assert!(
            per < 1e-8 * amp,
            "smoothed seasonal is not 4-periodic at {t}: {} vs {}",
            s.smoothed[t],
            s.smoothed[t + 4]
        );
    }
    eprintln!(
        "deterministic seasonal: worst |4-period sum| {worst_sum:e}, worst |gamma_t - gamma_{{t+4}}| {worst_period:e} (amplitude {amp:.3})"
    );
    // The estimated pattern recovers the truth (zero-mean version).
    let mean: f64 = pattern.iter().sum::<f64>() / 4.0;
    for (j, p) in pattern.iter().enumerate() {
        let est = s.smoothed[n - 4 + j];
        assert!((est - (p - mean)).abs() < 0.5, "pattern {j}: {est}");
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
    let i = fit
        .param_names
        .iter()
        .position(|p| p == "frequency.cycle")
        .unwrap();
    let f = fit.params[i];
    let (lo, hi) = (
        2.0 * std::f64::consts::PI / 20.0,
        2.0 * std::f64::consts::PI / 4.0,
    );
    assert!(f > lo && f < hi, "frequency {f} outside ({lo}, {hi})");
    assert!(
        (2.0 * std::f64::consts::PI / f - 8.0).abs() < 1.5,
        "period {}",
        2.0 * std::f64::consts::PI / f
    );
    let d = fit.params[fit
        .param_names
        .iter()
        .position(|p| p == "damping.cycle")
        .unwrap()];
    assert!(d > 0.7 && d < 1.0, "damping {d}");
    assert!(fit.cycle.is_some());
}

/// A trigonometric seasonal at the **Nyquist harmonic** — `period` even and
/// `harmonics = period / 2`, which is the default for an even period, and
/// therefore what `freq_seasonal=[12]` builds — carries a sine state at
/// frequency `pi` that the observation never loads on and the transition
/// never feeds into an observed state. It stays diffuse for the entire
/// sample and contributes nothing to the likelihood, so the log-likelihood
/// must equal that of the same model with the state struck out by hand.
///
/// This is an identity about the model, not a comparison with any package,
/// and it is the one an exact-diffuse filter breaks when it lets the
/// annihilated diffuse mass reappear as roundoff: with the always-live
/// `P_inf` of the superfluous state holding the diffuse period open, a
/// purely relative `F_inf` test votes "still diffuse" on `1e-32` of dust
/// and adds `-(ln 2 pi + ln F_inf) / 2 ~ +18` per period. Before the
/// cancellation guard in `filter.rs` this model came out 55 log-likelihood
/// points too high (and disagreed with statsmodels by the same amount).
#[test]
fn a_nyquist_harmonic_state_cannot_change_the_likelihood() {
    let mut rng = Lcg::new(4242);
    let n = 80;
    let mut y = Vec::with_capacity(n);
    let mut level = 2.0;
    for t in 0..n {
        let seasonal = [0.9, -0.4, -0.9, 0.4][t % 4];
        y.push(level + seasonal + normal(&mut rng));
        level += 0.4 * normal(&mut rng);
    }
    let (s2_eps, s2_level, s2_freq) = (0.8, 0.3, 0.05);
    let spec = UcSpec {
        trend: TrendSpec::LocalLevel,
        // period 4, default harmonics = 2 = period / 2: the Nyquist case.
        freq_seasonal: vec![FreqSeasonalSpec::new(4.0)],
        ..UcSpec::default()
    };
    assert_eq!(spec.freq_seasonal[0].harmonics, 2);
    let fit = unobserved_components(
        &y,
        &spec,
        &UcOptions {
            fixed_params: Some(vec![s2_eps, s2_level, s2_freq]),
            ..UcOptions::default()
        },
    )
    .unwrap();
    assert_eq!(fit.k_states, 5);
    assert_eq!(fit.nobs_diffuse, n, "the superfluous state never resolves");

    // The same model with the frequency-pi sine state struck out:
    // [level, h1 cos, h1 sin, h2 cos].
    let lam = 2.0 * std::f64::consts::PI / 4.0;
    let (s1, c1) = lam.sin_cos();
    let c2 = (2.0 * lam).cos();
    let reduced = LinearGaussianSSM::builder(1, 4, 4)
        .z(Mat::from_fn(1, 4, |_, j| if j == 2 { 0.0 } else { 1.0 }))
        .h(Mat::from_fn(1, 1, |_, _| s2_eps))
        .t(Mat::from_fn(4, 4, |i, j| match (i, j) {
            (0, 0) => 1.0,
            (1, 1) | (2, 2) => c1,
            (1, 2) => s1,
            (2, 1) => -s1,
            (3, 3) => c2,
            _ => 0.0,
        }))
        .r(Mat::from_fn(4, 4, |i, j| if i == j { 1.0 } else { 0.0 }))
        .q(Mat::from_fn(4, 4, |i, j| match (i, j) {
            (0, 0) => s2_level,
            (a, b) if a == b => s2_freq,
            _ => 0.0,
        }))
        .initialization(Initialization::Diffuse)
        .build()
        .unwrap();
    let ll_reduced = reduced
        .loglike(Mat::from_fn(n, 1, |i, _| y[i]).as_ref())
        .unwrap();
    eprintln!(
        "Nyquist harmonic: 5-state {} vs 4-state {} (diff {:e})",
        fit.loglik,
        ll_reduced,
        (fit.loglik - ll_reduced).abs()
    );
    assert!(
        (fit.loglik - ll_reduced).abs() < 1e-8,
        "the superfluous frequency-pi sine state moved the log-likelihood: \
         {} vs {ll_reduced}",
        fit.loglik
    );
}

/// **Why the default upper period bound is the sample length.** Under
/// exact-diffuse initialization the log-likelihood of a stochastic cycle
/// is unbounded above as the frequency goes to zero, and this test
/// measures the divergence instead of asserting it in prose.
///
/// At `lambda = 0` the cycle's second state is unobservable: it stays
/// diffuse for the whole sample and costs nothing (the Nyquist situation).
/// At a small `lambda > 0` it is *weakly* observable through
/// `rho sin lambda`, so its diffuse direction does resolve — and resolves
/// with `F_inf ~ (rho sin lambda)^2`, contributing
/// `-(ln 2 pi + ln F_inf)/2 ~ -ln lambda`. Halving `lambda` therefore buys
/// `ln 2 ~ 0.69` of log-likelihood for nothing, forever. A free optimizer
/// on `(0, pi)` reports a "cycle" of period `10^6` and a log-likelihood a
/// dozen points above any genuine optimum; the default period bound
/// `(2, nobs)` keeps it out of that region, and an explicit
/// `cycle_period_bounds` puts the user back in charge.
#[test]
fn the_cycle_log_likelihood_diverges_as_the_frequency_goes_to_zero() {
    let mut rng = Lcg::new(808);
    let n = 120;
    let mut y = Vec::with_capacity(n);
    let mut level = 0.0;
    for _ in 0..n {
        y.push(level + normal(&mut rng));
        level += 0.3 * normal(&mut rng);
    }
    // Explicitly opened up, which is the only way to reach the region.
    let spec = UcSpec {
        trend: TrendSpec::LocalLevel,
        cycle: true,
        damped_cycle: true,
        stochastic_cycle: true,
        cycle_period_bounds: (2.0, 1e9),
        ..UcSpec::default()
    };
    let mut prev = f64::NEG_INFINITY;
    let mut lls = Vec::new();
    for &freq in &[1e-2f64, 1e-4, 1e-6] {
        let fit = unobserved_components(
            &y,
            &spec,
            &UcOptions {
                fixed_params: Some(vec![0.8, 0.3, 0.2, freq, 0.9]),
                ..UcOptions::default()
            },
        )
        .unwrap();
        lls.push((freq, fit.loglik));
        assert!(
            fit.loglik > prev,
            "the log-likelihood must keep rising as the frequency falls: {:?}",
            lls
        );
        prev = fit.loglik;
    }
    eprintln!("cycle frequency -> 0 divergence: {lls:?}");
    // Measured at this seed: -188.15 at 1e-2, -183.55 at 1e-4, -162.53 at
    // 1e-6 — 25 log-likelihood points of pure singularity over four
    // decades. Assert at least 3 per two decades so the test fails on a
    // regression, not on noise.
    for w in lls.windows(2) {
        assert!(
            w[1].1 - w[0].1 > 3.0,
            "two decades of frequency bought only {}",
            w[1].1 - w[0].1
        );
    }
    // Below roughly 1e-7 the filter's cancellation guard declares the
    // direction resolved (`F_inf ~ (rho sin lambda)^2` falls under
    // `TOLERANCE_CANCEL` times the magnitude `P_inf` would have had
    // without cancellation), the second cycle state simply stays diffuse
    // as it does at `lambda = 0` exactly, and the divergence stops. That
    // is a floor on the damage, not a fix: 1e-6 is already 25 points of
    // nonsense, which is why the period bound is the answer.
    let deep = unobserved_components(
        &y,
        &spec,
        &UcOptions {
            fixed_params: Some(vec![0.8, 0.3, 0.2, 1e-8, 0.9]),
            ..UcOptions::default()
        },
    )
    .unwrap();
    eprintln!(
        "cycle frequency 1e-8 (past the cancellation guard): {}",
        deep.loglik
    );
    assert!(deep.loglik < lls[0].1, "the guard must cap the divergence");

    // The default bounds keep the estimator out of it: an infinite upper
    // period bound means the sample length, so the fitted period cannot
    // exceed `nobs`.
    let fit = unobserved_components(
        &y,
        &UcSpec {
            cycle_period_bounds: (2.0, f64::INFINITY),
            ..spec.clone()
        },
        &UcOptions::default(),
    )
    .unwrap();
    let i = fit
        .param_names
        .iter()
        .position(|p| p == "frequency.cycle")
        .unwrap();
    let period = 2.0 * std::f64::consts::PI / fit.params[i];
    eprintln!("default bounds: fitted cycle period {period:.3} (nobs {n})");
    assert!(
        period <= n as f64 + 1e-9,
        "the default bounds let the cycle period run past the sample: {period}"
    );

    // An empty admissible band is refused, naming the argument.
    let err = unobserved_components(
        &y[..10],
        &UcSpec {
            cycle_period_bounds: (20.0, f64::INFINITY),
            ..spec.clone()
        },
        &UcOptions::default(),
    );
    match err {
        Ok(_) => panic!("expected a refusal"),
        Err(e) => {
            let m = e.to_string();
            assert!(
                m.contains("cycle_period_bounds = (20, inf)") && m.contains("10 observations"),
                "message {m:?}"
            );
        }
    }
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
        (
            msg(unobserved_components(
                &[],
                &UcSpec::default(),
                &UcOptions::default(),
            )),
            "y is empty",
        ),
        (
            msg(unobserved_components(
                &[1.0, f64::INFINITY, 2.0],
                &UcSpec::default(),
                &UcOptions::default(),
            )),
            "y contains an infinity",
        ),
        (
            msg(unobserved_components(
                &y,
                &spec(&|s| s.seasonal = Some(1)),
                &UcOptions::default(),
            )),
            "seasonal = 1",
        ),
        (
            msg(unobserved_components(
                &y,
                &spec(&|s| {
                    s.freq_seasonal = vec![tsecon_ssm::FreqSeasonalSpec {
                        period: 12.0,
                        harmonics: 7,
                        stochastic: true,
                    }]
                }),
                &UcOptions::default(),
            )),
            "freq_seasonal[0].harmonics = 7",
        ),
        (
            msg(unobserved_components(
                &y,
                &spec(&|s| s.freq_seasonal = vec![tsecon_ssm::FreqSeasonalSpec::new(1.5)]),
                &UcOptions::default(),
            )),
            "freq_seasonal[0].period = 1.5",
        ),
        (
            msg(unobserved_components(
                &y,
                &spec(&|s| {
                    s.cycle = true;
                    s.cycle_period_bounds = (1.0, 10.0);
                }),
                &UcOptions::default(),
            )),
            "cycle_period_bounds = (1, 10)",
        ),
        (
            msg(unobserved_components(
                &y,
                &spec(&|s| s.exog = vec![vec![1.0; 39]]),
                &UcOptions::default(),
            )),
            "exog column 0 has length 39",
        ),
        (
            msg(unobserved_components(
                &y,
                &spec(&|s| s.exog = vec![vec![0.0; 40]]),
                &UcOptions::default(),
            )),
            "exog column 0 is identically zero",
        ),
        (
            msg(unobserved_components(
                &y,
                &spec(&|s| s.trend = TrendSpec::FixedIntercept),
                &UcOptions::default(),
            )),
            "level = \"fixed intercept\"",
        ),
        (
            msg(unobserved_components(
                &y,
                &UcSpec::default(),
                &UcOptions {
                    forecast_exog: vec![vec![1.0; 3]],
                    ..UcOptions::default()
                },
            )),
            "forecast_exog was given but forecast_steps = 0",
        ),
        (
            msg(unobserved_components(
                &y,
                &UcSpec::default(),
                &UcOptions {
                    forecast_steps: 3,
                    forecast_exog: vec![vec![1.0; 3]],
                    ..UcOptions::default()
                },
            )),
            "forecast_exog was given but the model has no regressors",
        ),
        (
            msg(unobserved_components(
                &y,
                &spec(&|s| s.exog = vec![(0..40).map(|t| t as f64).collect()]),
                &UcOptions {
                    forecast_steps: 3,
                    ..UcOptions::default()
                },
            )),
            "forecast_steps = 3 with 1 regressors requires forecast_exog",
        ),
        (
            msg(unobserved_components(
                &y,
                &UcSpec::default(),
                &UcOptions {
                    fixed_params: Some(vec![1.0]),
                    ..UcOptions::default()
                },
            )),
            "fixed_params has length 1 but the specification has 2 parameters",
        ),
        (
            msg(unobserved_components(
                &y,
                &UcSpec::default(),
                &UcOptions {
                    fixed_params: Some(vec![1.0, -1.0]),
                    ..UcOptions::default()
                },
            )),
            "fixed_params[1] (sigma2.level) = -1",
        ),
        (
            msg(unobserved_components(
                &y,
                &UcSpec::default(),
                &UcOptions {
                    n_starts: 0,
                    ..UcOptions::default()
                },
            )),
            "n_starts = 0",
        ),
        (
            msg(unobserved_components(
                &[1.0, 1.0, 1.0, 1.0, 1.0],
                &UcSpec::default(),
                &UcOptions::default(),
            )),
            "y is constant",
        ),
        (
            msg(unobserved_components(
                &[1.0, f64::NAN, 2.0],
                &UcSpec::default(),
                &UcOptions::default(),
            )),
            "observed (non-NaN) values",
        ),
        // Counts that are products of user integers: these must refuse
        // BEFORE anything is allocated, not abort the allocator. The
        // Python wrapper only rejects counts at or above 2^48.
        (
            msg(unobserved_components(
                &y,
                &spec(&|s| s.seasonal = Some(1_000_000)),
                &UcOptions::default(),
            )),
            "seasonal = 1000000 with 40 observations",
        ),
        (
            msg(unobserved_components(
                &y,
                &spec(&|s| s.freq_seasonal = vec![FreqSeasonalSpec::new(1e7)]),
                &UcOptions::default(),
            )),
            "freq_seasonal[0].period = 10000000 with 40 observations",
        ),
        (
            msg(unobserved_components(
                &y,
                &UcSpec::default(),
                &UcOptions {
                    forecast_steps: 1_000_000_000_000,
                    ..UcOptions::default()
                },
            )),
            "forecast_steps = 1000000000000: at most 100000",
        ),
        (
            msg(unobserved_components(
                &y,
                &UcSpec::default(),
                &UcOptions {
                    n_starts: 1 << 40,
                    ..UcOptions::default()
                },
            )),
            "at most 64 starting values",
        ),
    ];
    for (m, needle) in &cases {
        assert!(m.contains(needle), "message {m:?} does not name {needle:?}");
    }
    assert!(TrendSpec::parse("bogus")
        .unwrap_err()
        .to_string()
        .contains("level = \"bogus\""));

    let tmsg = |r: Result<tsecon_ssm::TvpFit, SsmError>| -> String {
        match r {
            Ok(_) => panic!("expected a refusal"),
            Err(e) => e.to_string(),
        }
    };
    let x = vec![(0..40).map(|t| (t as f64).sin()).collect::<Vec<f64>>()];
    let tcases: Vec<(String, &str)> = vec![
        (
            tmsg(tvp_regression(&y, &[vec![1.0; 39]], &TvpOptions::default())),
            "x column 0 has length 39",
        ),
        (
            tmsg(tvp_regression(
                &y,
                &[vec![f64::NAN; 40]],
                &TvpOptions::default(),
            )),
            "x column 0 contains a NaN",
        ),
        (
            tmsg(tvp_regression(
                &y,
                &[],
                &TvpOptions {
                    constant: false,
                    ..TvpOptions::default()
                },
            )),
            "x has no columns and constant = false",
        ),
        (
            tmsg(tvp_regression(
                &y,
                &x,
                &TvpOptions {
                    fixed_params: Some(vec![1.0, 0.0]),
                    ..TvpOptions::default()
                },
            )),
            "fixed_params has length 2",
        ),
        (
            tmsg(tvp_regression(
                &y,
                &x,
                &TvpOptions {
                    fixed_params: Some(vec![0.0, 0.0, 0.0]),
                    ..TvpOptions::default()
                },
            )),
            "fixed_params[0] = 0 (sigma2_eps)",
        ),
        (
            tmsg(tvp_regression(
                &y,
                &x,
                &TvpOptions {
                    fixed_params: Some(vec![1.0, -0.1, 0.0]),
                    ..TvpOptions::default()
                },
            )),
            "fixed_params[1] = -0.1",
        ),
        (
            tmsg(tvp_regression(
                &y[..3],
                &[x[0][..3].to_vec()],
                &TvpOptions::default(),
            )),
            "at least k + 2",
        ),
        (
            tmsg(tvp_regression(
                &y,
                &x,
                &TvpOptions {
                    n_starts: 0,
                    ..TvpOptions::default()
                },
            )),
            "n_starts = 0",
        ),
        (
            tmsg(tvp_regression(
                &y,
                &x,
                &TvpOptions {
                    n_starts: 1 << 40,
                    ..TvpOptions::default()
                },
            )),
            "at most 64 starting values",
        ),
    ];
    for (m, needle) in &tcases {
        assert!(m.contains(needle), "message {m:?} does not name {needle:?}");
    }
}
