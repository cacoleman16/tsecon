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

/// Exact-truth overlapping-horizon design for the growth-at-risk standard
/// errors. `x` is AR(1) with `phi = 0.8` (macro conditioners are
/// persistent); `y_{t+h} = 2 - x_t + v_{t+h}` with `v` an MA(h-1) of unit
/// variance, so `growth_at_risk`'s own design `[const, x_t, y_t]` is
/// CORRECTLY SPECIFIED at every tau, with true coefficients
/// `[2 + Phi^{-1}(tau), -1, 0]` — `y_t` carries no information about
/// `v_{t+h}` because the two MA windows do not overlap. Returns `(y, x)`.
fn overlapping_dgp(s: &mut Stream, n: usize, h: usize) -> (Vec<f64>, Vec<f64>) {
    let burn = 200;
    let tot = burn + n;
    let mut x = vec![0.0_f64; tot];
    for t in 1..tot {
        x[t] = 0.8 * x[t - 1] + gaussian(s);
    }
    let eps: Vec<f64> = (0..tot).map(|_| gaussian(s)).collect();
    let scale = 1.0 / (h as f64).sqrt();
    let mut y = vec![0.0_f64; tot];
    for t in h..tot {
        let v: f64 = (0..h).map(|j| eps[t - j]).sum::<f64>() * scale;
        y[t] = 2.0 - x[t - h] + v;
    }
    (y[burn..].to_vec(), x[burn..].to_vec())
}

/// One (horizon, seed) cell: coverage of the TRUE conditions slope (-1) by
/// the corrected and uncorrected 95% intervals, plus the mean se ratio.
fn coverage_cell(h: usize, n: usize, reps: usize, seed: u64) -> (f64, f64, f64) {
    const Z: f64 = 1.959_963_984_540_054;
    let mut s = Stream::new(seed);
    let (mut hit_hac, mut hit_powell) = (0usize, 0usize);
    let (mut sum_hac, mut sum_powell) = (0.0_f64, 0.0_f64);
    for _ in 0..reps {
        let (y, x) = overlapping_dgp(&mut s, n, h);
        let r = growth_at_risk(&y, &[x], h, &[0.05], false).expect("gar ok");
        let (b, se_hac, se_powell) = (r.params[0][1], r.bse[0][1], r.bse_powell[0][1]);
        if (b + 1.0).abs() <= Z * se_hac {
            hit_hac += 1;
        }
        if (b + 1.0).abs() <= Z * se_powell {
            hit_powell += 1;
        }
        sum_hac += se_hac;
        sum_powell += se_powell;
    }
    let d = reps as f64;
    (
        hit_hac as f64 / d,
        hit_powell as f64 / d,
        sum_hac / sum_powell,
    )
}

#[test]
fn overlapping_horizons_widen_the_growth_at_risk_standard_errors() {
    // A HORIZON SWEEP, for the same reason a scale-dependent bug needs a
    // scale sweep: a single-horizon test cannot see this class of defect.
    // The uncorrected Powell sandwich treats the check-loss score as a
    // martingale difference, which the overlapping h-step windows make
    // false for h > 1, and the resulting intervals are too narrow — worst
    // at exactly the multi-quarter horizons growth-at-risk exists for.
    //
    // Measured here (n = 200, tau = 0.05, 250 replications), coverage of a
    // nominal 95% interval around the true slope:
    //     h  =  1      4      8
    //   Powell  .888   .776   .672
    //   HAC     .888   .812   .756
    // The h = 1 column must be IDENTICAL (no overlap, no correction).
    let n = 200;
    let reps = 250;
    let mut previous_ratio = 0.0;
    for (i, &h) in [1usize, 4, 8].iter().enumerate() {
        let (cov_hac, cov_powell, ratio) = coverage_cell(h, n, reps, 8_675_309 + h as u64);
        println!("MEAS h={h} powell={cov_powell:.3} hac={cov_hac:.3} ratio={ratio:.4}");
        if h == 1 {
            assert!(
                (ratio - 1.0).abs() < 1e-15,
                "h=1 has no overlapping windows: the correction must be an \
                 exact no-op, got a mean se ratio of {ratio}"
            );
            assert!(
                (cov_hac - cov_powell).abs() < 1e-15,
                "h=1 coverage must be identical, got {cov_hac} vs {cov_powell}"
            );
        } else {
            // The correction must WIDEN, monotonically in the horizon --
            // more overlap, more neglected autocovariance.
            assert!(
                ratio > 1.10 && ratio > previous_ratio + 0.05,
                "h={h}: the Newey-West correction must widen the sandwich \
                 materially and more than at the shorter horizon (ratio \
                 {ratio}, previous {previous_ratio})"
            );
            // ... and it must buy real coverage back.
            assert!(
                cov_hac > cov_powell + 0.02,
                "h={h}: corrected coverage {cov_hac} must beat uncorrected \
                 {cov_powell} by a clear margin"
            );
        }
        // The defect is not fully cured: the density estimate f_hat(0) that
        // BOTH sandwiches divide by is itself biased upward in these
        // overlapping samples. This assertion documents the residual and
        // will fire (usefully) if anyone ever fixes it -- read the model
        // card's coverage table before widening it.
        if h == 8 {
            assert!(
                cov_hac < 0.90,
                "corrected coverage at h=8 is {cov_hac}: if this now reaches \
                 nominal, the model card's coverage table is stale"
            );
        }
        previous_ratio = if i == 0 { 0.0 } else { ratio };
    }
}
