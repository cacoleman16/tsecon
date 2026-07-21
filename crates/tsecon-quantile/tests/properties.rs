//! Monte-Carlo property tests — the statistical validation of the crate.
//! Golden fixtures pin the *algebra* to statsmodels; these seeded
//! simulations establish that the algebra is the *statistically correct*
//! one:
//!
//! * (a) OPTIMALITY: the fitted coefficients minimize the in-sample check
//!   loss — every perturbation raises it (up to the IRLS smoothing floor).
//! * (b) tau = 0.5 is the LAD/median fit: with an intercept-only design the
//!   fitted value is the sample quantile (the fraction of observations
//!   below it is tau), and the median-LP tracks the least-squares LP under
//!   symmetric errors on average across seeded replications.
//! * (c) REARRANGEMENT: rearranged growth-at-risk quantile paths are
//!   monotone across tau at every evaluation point, the crossing flag is
//!   exactly "raw paths violate monotonicity somewhere", and rearrangement
//!   is a no-op when nothing crosses.
//! * (d) ABG: in a location-scale DGP where the condition shifts the
//!   volatility of future outcomes, the lower conditional quantiles respond
//!   more strongly to the condition than the median — the
//!   Adrian-Boyarchenko-Giannone stylized fact, here true by construction.
//!
//! All randomness is the library's seeded Philox stream (`tsecon_rng`);
//! every number below is reproducible run to run.

use tsecon_quantile::{growth_at_risk, quantile_lp, quantile_regression};
use tsecon_rng::Stream;
use tsecon_stats::{ContinuousDist, StdNormal};

/// Standard normal via the inverse-CDF of a Philox uniform.
fn gaussian(s: &mut Stream) -> f64 {
    let u = s.uniform_f64().clamp(1e-12, 1.0 - 1e-12);
    StdNormal.ppf(u).expect("ppf on interior point")
}

/// In-sample check loss `sum_t rho_tau(y_t - x_t' b)`.
fn check_loss(y: &[f64], cols: &[Vec<f64>], beta: &[f64], tau: f64) -> f64 {
    (0..y.len())
        .map(|t| {
            let fit: f64 = cols.iter().zip(beta.iter()).map(|(c, b)| c[t] * b).sum();
            let u = y[t] - fit;
            u * (tau - if u < 0.0 { 1.0 } else { 0.0 })
        })
        .sum()
}

#[test]
fn fitted_coefficients_minimize_the_check_loss() {
    // Target (a). Perturbations well above the IRLS tolerance (1e-6) must
    // never lower the in-sample check loss beyond smoothing-floor slack.
    let mut s = Stream::new(42);
    let n = 150;
    let x1: Vec<f64> = (0..n).map(|_| gaussian(&mut s)).collect();
    let y: Vec<f64> = x1
        .iter()
        .map(|&v| 0.5 + 1.2 * v + (0.6 + 0.3 * v.abs()) * gaussian(&mut s))
        .collect();
    let cols = vec![vec![1.0; n], x1];

    for &tau in &[0.1, 0.25, 0.5, 0.75, 0.9] {
        let fit = &quantile_regression(&y, &cols, &[tau]).expect("fit ok")[0];
        let base = check_loss(&y, &cols, &fit.params, tau);
        for &scale in &[1e-3, 1e-2, 1e-1] {
            // Coordinate steps and seeded random directions.
            let mut dirs: Vec<Vec<f64>> = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
            for _ in 0..10 {
                dirs.push(vec![gaussian(&mut s), gaussian(&mut s)]);
            }
            for d in &dirs {
                for sign in [-1.0, 1.0] {
                    let b: Vec<f64> = fit
                        .params
                        .iter()
                        .zip(d.iter())
                        .map(|(p, di)| p + sign * scale * di)
                        .collect();
                    let perturbed = check_loss(&y, &cols, &b, tau);
                    assert!(
                        perturbed >= base - 1e-6 * (1.0 + base.abs()),
                        "tau={tau} scale={scale}: perturbation lowered the check \
                         loss ({perturbed} < {base})"
                    );
                }
            }
        }
    }
}

#[test]
fn intercept_only_fit_is_the_sample_quantile() {
    // Target (b): with only a constant, the check-loss minimizer is the
    // sample tau-quantile, so the fraction of observations below the fit
    // must be tau up to order-statistic granularity.
    let mut s = Stream::new(7);
    let n = 400;
    let y: Vec<f64> = (0..n).map(|_| 2.0 * gaussian(&mut s) - 0.3).collect();
    let cols = vec![vec![1.0; n]];
    for &tau in &[0.05, 0.25, 0.5, 0.75, 0.95] {
        let fit = &quantile_regression(&y, &cols, &[tau]).expect("fit ok")[0];
        let below = y.iter().filter(|&&v| v < fit.params[0]).count() as f64;
        let target = tau * n as f64;
        assert!(
            (below - target).abs() <= 3.0,
            "tau={tau}: {below} of {n} observations below the fit, expected ~{target}"
        );
    }
}

#[test]
fn one_call_with_many_taus_equals_many_calls_with_one() {
    let mut s = Stream::new(11);
    let n = 120;
    let x1: Vec<f64> = (0..n).map(|_| gaussian(&mut s)).collect();
    let y: Vec<f64> = x1.iter().map(|&v| v + gaussian(&mut s)).collect();
    let cols = vec![vec![1.0; n], x1];
    let taus = [0.2, 0.5, 0.8];
    let joint = quantile_regression(&y, &cols, &taus).expect("joint ok");
    for (i, &tau) in taus.iter().enumerate() {
        let single = &quantile_regression(&y, &cols, &[tau]).expect("single ok")[0];
        assert_eq!(
            &joint[i], single,
            "tau={tau}: joint call must equal single call"
        );
    }
}

/// The LS-LP impulse coefficient on the identical design quantile_lp uses.
fn ls_lp_irf(y: &[f64], shock: &[f64], h: usize, p: usize) -> f64 {
    let n = y.len();
    let start = p;
    let nobs = n - h - start;
    let outcome: Vec<f64> = (start..start + nobs).map(|t| y[t + h]).collect();
    let mut cols: Vec<Vec<f64>> = Vec::new();
    cols.push(shock[start..start + nobs].to_vec());
    cols.push(vec![1.0; nobs]);
    for lag in 1..=p {
        cols.push((start..start + nobs).map(|t| y[t - lag]).collect());
    }
    for lag in 1..=p {
        cols.push((start..start + nobs).map(|t| shock[t - lag]).collect());
    }
    tsecon_hac::ols(&outcome, &cols).expect("ols ok").params[0]
}

#[test]
fn median_lp_tracks_least_squares_lp_under_symmetric_errors() {
    // Target (b), LP form: with symmetric iid errors the conditional median
    // and mean coincide, so the tau = 0.5 LP and the LS-LP estimate the
    // same population IRF. Averaged over seeded replications the two must
    // agree closely at every horizon.
    let reps = 40;
    let n = 200;
    let p = 2;
    let max_h = 3;
    let mut s = Stream::new(20260721);
    let mut mean_gap = vec![0.0_f64; max_h + 1];
    for _ in 0..reps {
        let shock: Vec<f64> = (0..n).map(|_| gaussian(&mut s)).collect();
        let mut y = vec![0.0_f64; n];
        for t in 0..n {
            let prev = if t > 0 { y[t - 1] } else { 0.0 };
            let sprev = if t > 0 { shock[t - 1] } else { 0.0 };
            y[t] = 0.5 * prev + shock[t] + 0.3 * sprev + 0.8 * gaussian(&mut s);
        }
        let q = quantile_lp(&y, &shock, &[0.5], max_h, p).expect("qlp ok");
        for (h, gap) in mean_gap.iter_mut().enumerate() {
            *gap += (q.irf[0][h] - ls_lp_irf(&y, &shock, h, p)) / reps as f64;
        }
    }
    for (h, gap) in mean_gap.iter().enumerate() {
        assert!(
            gap.abs() < 0.05,
            "h={h}: mean gap between median-LP and LS-LP is {gap}, expected ~0 \
             under symmetric errors"
        );
    }
}

/// Location-scale growth-at-risk DGP: the condition `x` raises the
/// volatility of next-period `y`, so lower quantiles react more (ABG).
fn gar_dgp(s: &mut Stream, n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut x = vec![0.0_f64; n];
    let mut y = vec![0.0_f64; n];
    for t in 1..n {
        x[t] = 0.8 * x[t - 1] + 0.5 * gaussian(s);
        let scale = 0.4 * (0.4 * x[t - 1]).exp();
        y[t] = 0.2 + 0.3 * y[t - 1] - 0.4 * x[t - 1] + scale * gaussian(s);
    }
    (y, x)
}

#[test]
fn lower_quantiles_respond_more_to_a_variance_shifting_condition() {
    // Target (d): in the location-scale DGP the tau-quantile slope on the
    // condition is b1 + z_tau * (d sigma / dx); with z_0.05 < 0 < z_0.95
    // the slopes must order slope(0.05) < slope(0.5) < slope(0.95), and the
    // tails must sit clearly away from the median.
    let mut s = Stream::new(1913);
    let (y, x) = gar_dgp(&mut s, 3000);
    let r = growth_at_risk(&y, &[x], 1, &[0.05, 0.5, 0.95], true).expect("gar ok");
    let slope = |i: usize| r.params[i][1]; // [const, x, y_t]
    assert!(
        slope(0) < slope(1) - 0.1 && slope(1) < slope(2) - 0.1,
        "quantile slopes on the volatility-shifting condition must fan out: \
         got {} (tau=0.05), {} (tau=0.5), {} (tau=0.95)",
        slope(0),
        slope(1),
        slope(2)
    );
}

#[test]
fn rearranged_quantile_paths_are_monotone_and_crossing_is_reported_exactly() {
    // Target (c), over many short samples where crossings actually happen.
    let taus = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
    let mut s = Stream::new(271828);
    let mut saw_crossing = false;
    for _ in 0..25 {
        let (y, x) = gar_dgp(&mut s, 70);
        let n = y.len();
        let r = growth_at_risk(&y, &[x], 4, &taus, true).expect("gar ok");
        // Monotone after rearrangement, at every evaluation point.
        for j in 1..taus.len() {
            for t in 0..n {
                assert!(
                    r.fitted[j][t] >= r.fitted[j - 1][t],
                    "rearranged quantiles must be monotone in tau (t={t}, j={j})"
                );
            }
        }
        // The crossing flag is exactly "raw violates monotonicity".
        let raw_violation =
            (1..taus.len()).any(|j| (0..n).any(|t| r.fitted_raw[j][t] < r.fitted_raw[j - 1][t]));
        assert_eq!(
            r.crossing, raw_violation,
            "crossing flag must mirror the raw paths"
        );
        // No crossing => rearrangement is a no-op.
        if !r.crossing {
            assert_eq!(
                r.fitted, r.fitted_raw,
                "no crossing: rearrangement must be a no-op"
            );
        }
        saw_crossing |= r.crossing;
        // The current risk read is the last column of the fitted paths.
        for (j, &c) in r.current.iter().enumerate() {
            assert_eq!(
                c,
                r.fitted[j][n - 1],
                "current read must be the last fitted column"
            );
        }
    }
    assert!(
        saw_crossing,
        "the replication set must include at least one genuine crossing"
    );
}
