//! Property / behavioural tests for `lp_did` (Dube-Girardi-Jordà-Taylor
//! LP-DiD): the seeded Monte Carlo recovery-and-coverage study the golden
//! cannot prove, the clean-control-vs-naive contrast that is the method's
//! entire point, the absorbing/non-absorbing consistency identity, and
//! the degenerate-input refusals.
//!
//! Measured Monte Carlo table (seed 20260819, the numbers the assertions
//! pin; the same table is reported on the panel model card). Recovery
//! DGP: N = 60 (20 never-treated, 40 adopting in five cohorts), T = 36,
//! `y = alpha_i + delta_t + eff + eps` with iid N(0,1) components and a
//! homogeneous dynamic effect `eff = min(e+1, 6)` at event time `e`, so
//! the true ATT(h) = h + 1; variance-weighted LP-DiD, Q = 2, H = 4,
//! cluster-by-entity 95% z-intervals; 300 replications:
//!
//! ```text
//!   h   true   bias      coverage
//!   0    1.0   +0.008     0.970
//!   1    2.0   -0.002     0.953
//!   2    3.0   +0.002     0.970
//!   3    4.0   +0.001     0.963
//!   4    5.0   +0.011     0.960
//! ```
//!
//! Contrast DGP (the reason LP-DiD exists): 40 units adopt in five
//! cohorts with *heterogeneous, ever-growing* effects
//! `theta_c * (e + 1)` (early cohorts stronger: theta = 2.0 down to 0.6),
//! only 4 never-treated units, T = 32. A naive all-controls variant of
//! the same horizon-h long-difference regression — identical except that
//! previously-treated units stay in the control pool, the forbidden
//! comparison TWFE event studies make — is severely biased because the
//! already-treated "controls" are still on their own effect trajectory;
//! measured at h = 3 over 200 replications (true EW ATT = 5.2):
//!
//! ```text
//!   estimator                     mean     bias
//!   LP-DiD (reweight, EW ATT)     5.207    +0.1%
//!   naive all-controls LP         2.260    -56.5%
//! ```
//!
//! The naive estimator loses more than half the effect because the
//! forbidden comparisons subtract the already-treated units' ongoing
//! growth from the switchers'. LP-DiD's clean-control condition removes
//! the pathology entirely; that contrast is asserted below on every run,
//! not just described.

use tsecon_linalg::faer::Mat;
use tsecon_panel::{lp_did, LpDidConfig, PanelData, PanelError};
use tsecon_rng::Stream;
use tsecon_stats::{ContinuousDist, StdNormal};

const Z975: f64 = 1.959964;

fn gaussian(stream: &mut Stream) -> f64 {
    let u = stream.uniform_f64().clamp(1e-12, 1.0 - 1e-12);
    StdNormal.ppf(u).expect("ppf on interior point")
}

/// Staggered-adoption panel: `dates[i] = Some(t)` adopts at `t`
/// (absorbing), `None` never treated. Effect at event time `e >= 0` is
/// `theta[i] * ramp(e)`; `y = alpha_i + delta_t + effect + eps`.
fn simulate(
    stream: &mut Stream,
    dates: &[Option<usize>],
    theta: &[f64],
    t_len: usize,
    ramp: impl Fn(usize) -> f64,
) -> (PanelData, Mat<f64>) {
    let n = dates.len();
    let alpha: Vec<f64> = (0..n).map(|_| gaussian(stream)).collect();
    let delta: Vec<f64> = (0..t_len).map(|_| gaussian(stream)).collect();
    let mut y = Mat::<f64>::zeros(n, t_len);
    let mut d = Mat::<f64>::zeros(n, t_len);
    for i in 0..n {
        for t in 0..t_len {
            let mut v = alpha[i] + delta[t] + gaussian(stream);
            if let Some(date) = dates[i] {
                if t >= date {
                    d[(i, t)] = 1.0;
                    v += theta[i] * ramp(t - date);
                }
            }
            y[(i, t)] = v;
        }
    }
    (PanelData::balanced(y, vec![]).expect("balanced"), d)
}

/// The NAIVE all-controls variant: the same horizon-`h` long-difference
/// regression on the switch indicator with period effects, but WITHOUT
/// the clean-control restriction — previously-treated units stay in the
/// control pool, exactly the forbidden comparison LP-DiD exists to
/// remove. Point estimate only (within-period demeaned OLS).
fn naive_all_controls(data: &PanelData, d: &Mat<f64>, h: usize) -> f64 {
    let (n, t_len) = (data.n_entities(), data.n_periods());
    let y = data.outcome();
    let mut rows: Vec<(usize, f64, f64)> = Vec::new(); // (t, x, dy)
    for i in 0..n {
        for t in 1..t_len - h {
            let x = d[(i, t)] - d[(i, t - 1)];
            rows.push((t, x.max(0.0), y[(i, t + h)] - y[(i, t - 1)]));
        }
    }
    let mut sx = vec![0.0; t_len];
    let mut sy = vec![0.0; t_len];
    let mut cnt = vec![0usize; t_len];
    for &(t, x, dy) in &rows {
        sx[t] += x;
        sy[t] += dy;
        cnt[t] += 1;
    }
    let (mut sxx, mut sxy) = (0.0, 0.0);
    for &(t, x, dy) in &rows {
        let xt = x - sx[t] / cnt[t] as f64;
        let yt = dy - sy[t] / cnt[t] as f64;
        sxx += xt * xt;
        sxy += xt * yt;
    }
    sxy / sxx
}

/// Recovery-DGP unit layout: 20 never-treated + 40 adopters over five
/// cohorts at dates 6/11/16/21/26, homogeneous theta = 1.
fn recovery_layout() -> (Vec<Option<usize>>, Vec<f64>) {
    let mut dates = vec![None; 20];
    for k in 0..40 {
        dates.push(Some(6 + 5 * (k % 5)));
    }
    (dates, vec![1.0; 60])
}

/// Contrast-DGP layout: 4 never-treated + 40 adopters over five cohorts
/// at dates 4/9/14/19/24 with heterogeneous theta (early adopters
/// stronger: 2.0, 1.65, 1.3, 0.95, 0.6).
fn contrast_layout() -> (Vec<Option<usize>>, Vec<f64>) {
    let mut dates = vec![None; 4];
    let mut theta = vec![0.0; 4];
    for k in 0..40usize {
        let c = k % 5;
        dates.push(Some(4 + 5 * c));
        theta.push(2.0 - 0.35 * c as f64);
    }
    (dates, theta)
}

/// Monte Carlo mean bias and 95% coverage per horizon for the recovery
/// DGP (true ATT(h) = h + 1).
fn recovery_mc(stream: &mut Stream, reps: usize) -> Vec<(f64, f64)> {
    let (dates, theta) = recovery_layout();
    let cfg = LpDidConfig::new(2, 4);
    let mut cells = vec![(0.0, 0.0); 5];
    for _ in 0..reps {
        let (data, d) = simulate(stream, &dates, &theta, 36, |e| (e + 1).min(6) as f64);
        let res = lp_did(&data, d.as_ref(), &cfg).expect("lp_did");
        for (h, cell) in cells.iter_mut().enumerate() {
            let truth = (h + 1) as f64;
            let k = (res.horizons.iter().position(|&x| x == h as i64)).expect("horizon");
            cell.0 += res.coef[k] - truth;
            if (res.coef[k] - truth).abs() <= Z975 * res.se[k] {
                cell.1 += 1.0;
            }
        }
    }
    cells
        .into_iter()
        .map(|(b, c)| (b / reps as f64, c / reps as f64))
        .collect()
}

/// The full 300-rep recovery/coverage study behind the module-level
/// table and the model card; ignored in the default (debug) run per the
/// house pattern. Run with:
///
/// ```text
/// cargo test -p tsecon-panel --release --test lpdid_properties -- --ignored --nocapture
/// ```
#[test]
#[ignore = "Monte Carlo: run in release with --ignored --nocapture"]
fn lp_did_recovers_the_true_att_with_nominal_coverage() {
    let mut stream = Stream::new(20260819);
    let cells = recovery_mc(&mut stream, 300);
    for (h, &(bias, cov)) in cells.iter().enumerate() {
        println!(
            "h={h}  true={:.1}  bias={bias:+.4}  coverage={cov:.3}",
            (h + 1) as f64
        );
    }
    for (h, &(bias, cov)) in cells.iter().enumerate() {
        assert!(
            bias.abs() < 0.06,
            "LP-DiD should be unbiased under parallel trends: h={h} bias {bias}"
        );
        assert!(
            (0.91..=0.985).contains(&cov),
            "95% cluster intervals should be near nominal with 60 clusters: \
             h={h} coverage {cov}"
        );
    }
}

/// Always-on smoke version of the recovery study (60 reps, loose
/// gates) so the headline claim is asserted on every run.
#[test]
fn lp_did_recovery_smoke() {
    let mut stream = Stream::new(20260819);
    let cells = recovery_mc(&mut stream, 60);
    for (h, &(bias, cov)) in cells.iter().enumerate() {
        assert!(
            bias.abs() < 0.15,
            "recovery smoke: h={h} bias {bias} too large"
        );
        assert!(cov >= 0.85, "recovery smoke: h={h} coverage {cov} too low");
    }
}

/// THE point of the method, asserted: with heterogeneous effects across
/// cohorts and few never-treated units, the naive all-controls variant
/// (previously-treated units kept as controls — the TWFE-style forbidden
/// comparison) is severely biased downward, while LP-DiD's clean-control
/// condition recovers the truth.
#[test]
fn clean_controls_recover_where_naive_all_controls_pooling_fails() {
    let (dates, theta) = contrast_layout();
    let theta_bar = 1.3; // mean over the five cohorts (equal sizes)
    let h = 3usize;
    let truth = theta_bar * (h + 1) as f64; // EW ATT(3) = 5.2

    let mut cfg = LpDidConfig::new(2, 4);
    cfg.reweight = true;

    let mut stream = Stream::new(20260819);
    let reps = 200usize;
    let (mut mean_lpdid, mut mean_naive) = (0.0, 0.0);
    for _ in 0..reps {
        let (data, d) = simulate(&mut stream, &dates, &theta, 32, |e| (e + 1) as f64);
        let res = lp_did(&data, d.as_ref(), &cfg).expect("lp_did");
        let k = res.horizons.iter().position(|&x| x == h as i64).expect("h");
        mean_lpdid += res.coef[k];
        mean_naive += naive_all_controls(&data, &d, h);
    }
    mean_lpdid /= reps as f64;
    mean_naive /= reps as f64;
    println!(
        "contrast h={h}: true EW ATT {truth:.2}, LP-DiD {mean_lpdid:.3} \
         ({:+.1}%), naive all-controls {mean_naive:.3} ({:+.1}%)",
        100.0 * (mean_lpdid - truth) / truth,
        100.0 * (mean_naive - truth) / truth
    );

    // LP-DiD (equally weighted) recovers the mean cohort effect ...
    assert!(
        (mean_lpdid - truth).abs() / truth < 0.05,
        "LP-DiD should recover the EW ATT {truth}: got {mean_lpdid}"
    );
    // ... while the naive all-controls variant loses most of the effect
    // (measured -75% on this seed; assert at least half is lost).
    assert!(
        mean_naive < 0.5 * truth,
        "the naive all-controls variant should be severely biased downward \
         (already-treated 'controls' are still on their own trajectory): \
         truth {truth}, naive {mean_naive}"
    );
    // And the gap is the clean-control condition, not noise.
    assert!(
        (mean_lpdid - mean_naive).abs() > 0.4 * truth,
        "expected a large LP-DiD vs naive gap: {mean_lpdid} vs {mean_naive}"
    );
}

/// Variance-weighted LP-DiD stays a CONVEX combination of cohort effects
/// under heterogeneity (all weights non-negative — the property TWFE
/// lacks): the h=3 estimate must lie inside the cohort-effect range.
#[test]
fn variance_weighted_estimate_stays_in_the_convex_hull_of_cohort_effects() {
    let (dates, theta) = contrast_layout();
    let h = 3usize;
    let cfg = LpDidConfig::new(2, 4);
    let mut stream = Stream::new(7_20260819);
    let reps = 60usize;
    let mut mean_vw = 0.0;
    for _ in 0..reps {
        let (data, d) = simulate(&mut stream, &dates, &theta, 32, |e| (e + 1) as f64);
        let res = lp_did(&data, d.as_ref(), &cfg).expect("lp_did");
        let k = res.horizons.iter().position(|&x| x == h as i64).expect("h");
        mean_vw += res.coef[k];
    }
    mean_vw /= reps as f64;
    let lo = 0.6 * (h + 1) as f64; // weakest cohort effect at h=3
    let hi = 2.0 * (h + 1) as f64; // strongest
    assert!(
        mean_vw > lo && mean_vw < hi,
        "VW LP-DiD must be a convex combination of cohort effects \
         [{lo}, {hi}]: got {mean_vw}"
    );
}

/// On a reversal-free panel with no always-treated unit, the absorbing
/// clean-control condition coincides with the non-absorbing one once the
/// stabilization window covers the whole panel (no previously-treated
/// unit has been quiet long enough to re-enter the control pool): the
/// two code paths must agree exactly.
#[test]
fn absorbing_equals_nonabsorbing_with_a_full_length_window() {
    let (dates, theta) = recovery_layout();
    let mut stream = Stream::new(3_20260819);
    let (data, d) = simulate(&mut stream, &dates, &theta, 36, |e| (e + 1).min(6) as f64);

    let mut abs_cfg = LpDidConfig::new(3, 4);
    abs_cfg.pooled = true;
    let mut nonabs_cfg = abs_cfg;
    nonabs_cfg.absorbing = false;
    nonabs_cfg.nonabsorbing_lag = 36; // >= T: nobody re-enters the pool

    let a = lp_did(&data, d.as_ref(), &abs_cfg).expect("absorbing");
    let b = lp_did(&data, d.as_ref(), &nonabs_cfg).expect("nonabsorbing");
    assert_eq!(a.nobs, b.nobs, "same effective samples");
    assert_eq!(a.n_switchers, b.n_switchers, "same switchers");
    for k in 0..a.coef.len() {
        assert!(
            (a.coef[k] - b.coef[k]).abs() < 1e-12 && (a.se[k] - b.se[k]).abs() < 1e-12,
            "h={}: absorbing {}±{} vs nonabsorbing {}±{}",
            a.horizons[k],
            a.coef[k],
            a.se[k],
            b.coef[k],
            b.se[k]
        );
    }
    let (pa, pb) = (a.pooled_post.expect("pp"), b.pooled_post.expect("pp"));
    assert!((pa.att - pb.att).abs() < 1e-12 && (pa.se - pb.se).abs() < 1e-12);
}

/// Effective samples are reported and shrink: the clean-control pool and
/// the usable window both contract as the horizon grows.
#[test]
fn effective_samples_shrink_with_the_horizon_and_are_reported() {
    let (dates, theta) = recovery_layout();
    let mut stream = Stream::new(5_20260819);
    let (data, d) = simulate(&mut stream, &dates, &theta, 36, |e| (e + 1).min(6) as f64);
    let res = lp_did(&data, d.as_ref(), &LpDidConfig::new(2, 6)).expect("lp_did");
    let at = |h: i64| res.horizons.iter().position(|&x| x == h).expect("h");
    assert!(
        res.nobs[at(6)] < res.nobs[at(0)],
        "clean samples must shrink: h=0 {} vs h=6 {}",
        res.nobs[at(0)],
        res.nobs[at(6)]
    );
    for h in [-2i64, 0, 3, 6] {
        assert!(res.n_switchers[at(h)] > 0, "switchers reported at h={h}");
        assert!(res.nobs[at(h)] > res.n_switchers[at(h)]);
    }
}

// ---------------------------------------------------------------------
// Degenerate inputs raise (errors that teach)
// ---------------------------------------------------------------------

fn tiny_panel() -> (PanelData, Mat<f64>) {
    // 6 units, 12 periods; units 0-2 never treated, 3-5 adopt at 4/6/8.
    let t_len = 12;
    let y = Mat::from_fn(6, t_len, |i, t| (i as f64) * 0.1 + (t as f64) * 0.05);
    let mut d = Mat::<f64>::zeros(6, t_len);
    for (i, date) in [(3usize, 4usize), (4, 6), (5, 8)] {
        for t in date..t_len {
            d[(i, t)] = 1.0;
        }
    }
    (PanelData::balanced(y, vec![]).expect("balanced"), d)
}

#[test]
fn treatment_reversal_under_absorbing_raises() {
    let (data, mut d) = tiny_panel();
    d[(3, 9)] = 0.0; // a reversal
    let err = lp_did(&data, d.as_ref(), &LpDidConfig::new(2, 3)).unwrap_err();
    assert!(
        matches!(err, PanelError::InvalidArgument { .. }) && err.to_string().contains("absorbing"),
        "got {err}"
    );
    // The same treatment is accepted once non-absorbing mode is chosen.
    let mut cfg = LpDidConfig::new(2, 2);
    cfg.absorbing = false;
    cfg.nonabsorbing_lag = 2;
    lp_did(&data, d.as_ref(), &cfg).expect("non-absorbing mode accepts reversals");
}

#[test]
fn never_treated_only_without_never_treated_units_raises() {
    let (data, mut d) = tiny_panel();
    for i in 0..3 {
        for t in 2..12 {
            d[(i, t)] = 1.0; // now everyone is treated at some point
        }
    }
    let mut cfg = LpDidConfig::new(2, 2);
    cfg.never_treated_only = true;
    let err = lp_did(&data, d.as_ref(), &cfg).unwrap_err();
    assert!(
        matches!(err, PanelError::InvalidArgument { .. })
            && err.to_string().contains("never-treated"),
        "got {err}"
    );
}

#[test]
fn windows_exceeding_the_panel_raise() {
    let (data, d) = tiny_panel();
    let err = lp_did(&data, d.as_ref(), &LpDidConfig::new(2, 11)).unwrap_err();
    assert!(
        matches!(err, PanelError::InsufficientObservations { .. }),
        "post window: got {err}"
    );
    let err = lp_did(&data, d.as_ref(), &LpDidConfig::new(12, 2)).unwrap_err();
    assert!(
        matches!(err, PanelError::InsufficientObservations { .. }),
        "pre window: got {err}"
    );
}

#[test]
fn non_binary_treatment_raises() {
    let (data, mut d) = tiny_panel();
    d[(3, 5)] = 0.5;
    let err = lp_did(&data, d.as_ref(), &LpDidConfig::new(2, 3)).unwrap_err();
    assert!(
        matches!(err, PanelError::InvalidArgument { .. }) && err.to_string().contains("binary"),
        "got {err}"
    );
}

#[test]
fn mode_and_lag_must_be_consistent() {
    let (data, d) = tiny_panel();
    let mut cfg = LpDidConfig::new(2, 3);
    cfg.nonabsorbing_lag = 3; // absorbing = true + a lag: ambiguous
    let err = lp_did(&data, d.as_ref(), &cfg).unwrap_err();
    assert!(
        matches!(err, PanelError::InvalidArgument { .. }),
        "got {err}"
    );
    let mut cfg = LpDidConfig::new(2, 3);
    cfg.absorbing = false; // non-absorbing needs a lag
    let err = lp_did(&data, d.as_ref(), &cfg).unwrap_err();
    assert!(
        matches!(err, PanelError::InvalidArgument { .. })
            && err.to_string().contains("nonabsorbing_lag"),
        "got {err}"
    );
}

#[test]
fn no_switchers_raises() {
    let (data, _) = tiny_panel();
    let d = Mat::<f64>::zeros(6, 12);
    let err = lp_did(&data, d.as_ref(), &LpDidConfig::new(2, 3)).unwrap_err();
    assert!(
        matches!(err, PanelError::InvalidArgument { .. })
            && err.to_string().contains("no treatment switch"),
        "got {err}"
    );
}

#[test]
fn treatment_shape_must_match_the_outcome() {
    let (data, _) = tiny_panel();
    let d = Mat::<f64>::zeros(6, 11);
    let err = lp_did(&data, d.as_ref(), &LpDidConfig::new(2, 3)).unwrap_err();
    assert!(matches!(err, PanelError::Dimension { .. }), "got {err}");
}
