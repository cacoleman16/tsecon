//! Property / behavioural tests for the pooled-mean-group (PMG) estimator —
//! the statistical content that no single golden value pins:
//!
//! * **consistency:** on data simulated from an ARDL(1,1) with a *known common*
//!   long-run `theta0`, PMG recovers `theta0` within Monte-Carlo bands;
//! * **the point of pooling:** PMG's long-run SE is far below the cross-unit
//!   dispersion of a *free* mean group of the per-unit long-run estimates —
//!   pooling buys precision that free heterogeneity cannot;
//! * stable adjustment (`phi_bar < 0`) and a self-consistent fit;
//! * the validation layer rejects malformed / too-short panels.

use tsecon_panelts::{pmg, tsecon_hac::ols, PanelTsError, PanelUnit};
use tsecon_rng::Stream;
use tsecon_stats::{ContinuousDist, StdNormal};

fn gaussian(stream: &mut Stream) -> f64 {
    let u = stream.uniform_f64().clamp(1e-12, 1.0 - 1e-12);
    StdNormal.ppf(u).expect("ppf on interior point")
}

/// Simulate a heterogeneous ARDL(1,1) panel with a *common* long-run
/// coefficient `theta0` (scalar `x`):
///
/// ```text
/// y_it = mu_i + lambda_i y_{i,t-1} + d0_i x_it + d1_i x_{i,t-1} + e_it,
/// ```
///
/// with `lambda_i ∈ (0.2, 0.7)`, free short-run `d0_i`, and
/// `d1_i = theta0 (1 - lambda_i) - d0_i` so that every unit's long run
/// `(d0_i + d1_i)/(1 - lambda_i) = theta0`. `x` is a stationary AR(1).
fn simulate(seed: u64, n: usize, t_raw: usize, theta0: f64) -> Vec<PanelUnit> {
    let mut s = Stream::new(seed);
    let burn = 50usize;
    let tt = t_raw + burn;
    (0..n)
        .map(|_| {
            let lam = 0.2 + 0.5 * s.uniform_f64(); // (0.2, 0.7)
            let mu = 0.5 + gaussian(&mut s);
            let d0 = 0.6 + 0.25 * gaussian(&mut s);
            let d1 = theta0 * (1.0 - lam) - d0;
            let rho = 0.3 + 0.3 * s.uniform_f64();
            let xmean = gaussian(&mut s);

            let mut x = vec![0.0_f64; tt];
            x[0] = xmean;
            for t in 1..tt {
                x[t] = xmean * (1.0 - rho) + rho * x[t - 1] + gaussian(&mut s);
            }
            let mut y = vec![0.0_f64; tt];
            y[0] = mu / (1.0 - lam);
            for t in 1..tt {
                y[t] = mu + lam * y[t - 1] + d0 * x[t] + d1 * x[t - 1] + 0.5 * gaussian(&mut s);
            }
            PanelUnit::new(y[burn..].to_vec(), vec![x[burn..].to_vec()])
        })
        .collect()
}

/// Free per-unit long run from an unrestricted EC OLS: regress `Δy` on
/// `[const, y_{-1}, x_{-1}, Δx]` and read `theta_i = -coef(x_{-1})/coef(y_{-1})`.
fn free_long_run(unit: &PanelUnit) -> f64 {
    let y = &unit.y;
    let x = &unit.x[0];
    let t = y.len() - 1;
    let cons = vec![1.0_f64; t];
    let ylag: Vec<f64> = (0..t).map(|r| y[r]).collect();
    let xlag: Vec<f64> = (0..t).map(|r| x[r]).collect();
    let dx: Vec<f64> = (0..t).map(|r| x[r + 1] - x[r]).collect();
    let dy: Vec<f64> = (0..t).map(|r| y[r + 1] - y[r]).collect();
    let fit = ols(&dy, &[cons, ylag, xlag, dx]).expect("per-unit EC OLS");
    let phi_i = fit.params[1];
    -fit.params[2] / phi_i
}

#[test]
fn pmg_recovers_known_common_long_run() {
    let theta0 = 1.30;
    let units = simulate(2026_0717, 45, 70, theta0);
    let fit = pmg(&units).expect("PMG fits");

    // Consistency: PMG lands on the common long run within a tight band.
    assert!(
        (fit.theta[0] - theta0).abs() < 0.08,
        "PMG theta {} should recover theta0 {theta0}",
        fit.theta[0]
    );
    // Stable error correction on average.
    assert!(
        fit.phi_bar < 0.0,
        "average adjustment speed should be negative, got {}",
        fit.phi_bar
    );
    // Internal consistency: k, N, and a finite log-likelihood.
    assert_eq!(fit.k, 1);
    assert_eq!(fit.n_units, 45);
    assert_eq!(fit.phi.len(), 45);
    assert!(fit.loglik.is_finite());
    assert!(fit.theta_se[0] > 0.0);
}

#[test]
fn pmg_pools_far_tighter_than_free_mean_group() {
    let theta0 = 1.30;
    let units = simulate(7, 50, 70, theta0);
    let fit = pmg(&units).expect("PMG fits");

    // Cross-unit dispersion of the free per-unit long-run estimates.
    let free: Vec<f64> = units.iter().map(free_long_run).collect();
    let n = free.len() as f64;
    let mean = free.iter().sum::<f64>() / n;
    let var = free.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let free_sd = var.sqrt();

    // Both are near the truth, but PMG's pooled SE is a small fraction of the
    // free cross-unit spread: that precision gain is the whole point of PMG.
    assert!((mean - theta0).abs() < 0.15, "free-MG mean {mean}");
    assert!(
        fit.theta_se[0] < 0.30 * free_sd,
        "PMG SE {} should be far below free-MG cross-unit sd {free_sd}",
        fit.theta_se[0]
    );
}

#[test]
fn pmg_rejects_too_few_units() {
    let u = simulate(1, 1, 40, 1.0);
    assert!(matches!(pmg(&u), Err(PanelTsError::TooFewUnits { n: 1 })));
    assert!(matches!(pmg(&[]), Err(PanelTsError::TooFewUnits { n: 0 })));
}

#[test]
fn pmg_rejects_too_few_periods() {
    // k = 1 long-run regressor needs T_raw >= k + 3 = 4 periods.
    let short = vec![
        PanelUnit::new(vec![1.0, 2.0, 3.0], vec![vec![0.1, 0.2, 0.3]]),
        PanelUnit::new(vec![1.0, 2.0, 3.0], vec![vec![0.1, 0.2, 0.3]]),
    ];
    assert!(matches!(
        pmg(&short),
        Err(PanelTsError::PmgTooFewPeriods {
            unit: 0,
            got: 3,
            needed: 4
        })
    ));
}

#[test]
fn pmg_rejects_inconsistent_and_ragged() {
    let inconsistent = vec![
        PanelUnit::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![vec![0.1, 0.2, 0.3, 0.4, 0.5]],
        ),
        PanelUnit::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![vec![0.1, 0.2, 0.3, 0.4, 0.5], vec![0.5, 0.6, 0.7, 0.8, 0.9]],
        ),
    ];
    assert!(matches!(
        pmg(&inconsistent),
        Err(PanelTsError::InconsistentRegressors { unit: 1, .. })
    ));

    let ragged = vec![
        PanelUnit::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![vec![0.1, 0.2, 0.3, 0.4, 0.5]],
        ),
        PanelUnit::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![vec![0.1, 0.2, 0.3]]),
    ];
    assert!(matches!(
        pmg(&ragged),
        Err(PanelTsError::RaggedUnit { unit: 1, .. })
    ));
}
