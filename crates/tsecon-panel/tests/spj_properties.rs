//! Property / behavioural tests for `panel_lp`'s split-panel jackknife
//! (`bias_correction = Spj`, Mei-Sheng-Shi 2026): the seeded Monte Carlo
//! bias-and-coverage study the golden transcription cannot prove, the
//! documented relation to the existing Dhaene-Jochmans `jackknife` flag,
//! and the degenerate-input refusals.
//!
//! Measured Monte Carlo table (the numbers the assertions below pin; the
//! same table is reported on the panel model card): seed 20260818,
//! 300 replications per T, N = 50 entities, DGP
//! `y_{i,t} = alpha_i + 0.8 y_{i,t-1} + 0.8 s_t + e_{i,t}` with a common
//! iid N(0,1) shock and iid N(0,1) idiosyncratic errors; spec
//! `max_horizon = 2`, one shock lag + one outcome lag, Driscoll-Kraay
//! SEs with bandwidth 2 (= pLP's auto rule floor((T-h)^(1/4)) for every
//! T here); true IRF `0.8 * 0.8^h`. "cov" is 95%-nominal coverage of the
//! true `irf[h]`.
//!
//! ```text
//!   T   h   bias FE    bias SPJ   |FE|/|SPJ|   cov FE   cov SPJ
//!  20   1   -0.094     +0.023        4.1x       0.770    0.817
//!  20   2   -0.137     +0.009       15.2x       0.743    0.823
//!  40   1   -0.035     +0.015        2.3x       0.900    0.873
//!  40   2   -0.054     +0.019        2.9x       0.870    0.863
//!  80   1   -0.009     +0.012        0.7x       0.920    0.907
//!  80   2   -0.013     +0.019        0.7x       0.900    0.877
//! ```
//!
//! A 2000-replication rerun on an independent seed sharpens the noisy
//! cells: the SPJ bias at h = 2 is +0.007 (T = 40), +0.002 (T = 80,
//! MC se 0.003), +0.003 (T = 160), while FE is -0.060 / -0.025 / -0.010
//! — i.e. FE shrinks like O(1/T) and SPJ removes ~85-95% of the bias;
//! the 300-rep T = 80 SPJ entries above are Monte-Carlo noise (the FE
//! and SPJ columns share draws, so the small-bias cells wobble
//! together), not residual bias.
//!
//! Reading the coverage honestly: at T = 20 the FE t-interval is
//! clearly invalid (0.74 at h = 2) and SPJ recovers about half the gap
//! (0.82) — but neither reaches the nominal 95%, because with a COMMON
//! shock the horizon-h residual contains common future shocks, DK is the
//! right covariance family, and DK itself is a short-T approximation
//! (exactly the caveat the module docs and the cookbook page carry). At
//! T >= 40 both sit in the high-0.80s/low-0.90s, within Monte-Carlo
//! noise of each other (~2pp per cell at 300 reps), with the bias gone
//! from SPJ. The model card states these measured numbers rather than
//! promising nominal coverage at T = 20.

use tsecon_linalg::faer::Mat;
use tsecon_panel::{panel_lp, LpBiasCorrection, PanelData, PanelError, PanelLpConfig, PanelSeType};
use tsecon_rng::Stream;
use tsecon_stats::{ContinuousDist, StdNormal};

const Z975: f64 = 1.959964;
const RHO: f64 = 0.8;
const BETA: f64 = 0.8;
const N_ENT: usize = 50;

/// One standard-normal draw by inverse transform from a Philox uniform.
fn gaussian(stream: &mut Stream) -> f64 {
    let u = stream.uniform_f64().clamp(1e-12, 1.0 - 1e-12);
    StdNormal.ppf(u).expect("ppf on interior point")
}

/// Simulate the dynamic-panel DGP
/// `y_{i,t} = alpha_i + rho y_{i,t-1} + beta s_t + e_{i,t}` with a
/// common iid shock, at stationarity via burn-in.
fn simulate_panel(stream: &mut Stream, n_ent: usize, t_len: usize) -> (PanelData, Vec<f64>) {
    let burn = 25;
    let total = t_len + burn;
    let shock_all: Vec<f64> = (0..total).map(|_| gaussian(stream)).collect();
    let alpha: Vec<f64> = (0..n_ent).map(|_| gaussian(stream)).collect();
    let mut y = vec![vec![0.0_f64; total]; n_ent];
    for (i, row) in y.iter_mut().enumerate() {
        row[0] = alpha[i] + BETA * shock_all[0] + gaussian(stream);
        for t in 1..total {
            row[t] = alpha[i] + RHO * row[t - 1] + BETA * shock_all[t] + gaussian(stream);
        }
    }
    let outcome = Mat::from_fn(n_ent, t_len, |i, t| y[i][burn + t]);
    (
        PanelData::balanced(outcome, vec![]).expect("balanced panel"),
        shock_all[burn..].to_vec(),
    )
}

fn spec(bias_correction: LpBiasCorrection) -> PanelLpConfig {
    let mut cfg = PanelLpConfig::new(2, 1, PanelSeType::DriscollKraay { bandwidth: 2.0 });
    cfg.bias_correction = bias_correction;
    cfg
}

/// Per-horizon Monte Carlo summary for one estimator.
#[derive(Debug, Clone, Copy, Default)]
struct McCell {
    bias: f64,
    coverage: f64,
}

/// Runs the Monte Carlo for one T; returns `(fe, spj)` cells for
/// horizons `0..=2`.
fn run_mc(stream: &mut Stream, t_len: usize, reps: usize) -> ([McCell; 3], [McCell; 3]) {
    let mut fe_cells = [McCell::default(); 3];
    let mut spj_cells = [McCell::default(); 3];
    for _ in 0..reps {
        let (data, shock) = simulate_panel(stream, N_ENT, t_len);
        let fe = panel_lp(&data, &shock, &spec(LpBiasCorrection::None)).expect("FE lp");
        let spj = panel_lp(&data, &shock, &spec(LpBiasCorrection::Spj)).expect("SPJ lp");
        for h in 0..=2usize {
            let truth = BETA * RHO.powi(h as i32);
            fe_cells[h].bias += fe.irf[h] - truth;
            spj_cells[h].bias += spj.irf[h] - truth;
            if (fe.irf[h] - truth).abs() <= Z975 * fe.se[h] {
                fe_cells[h].coverage += 1.0;
            }
            if (spj.irf[h] - truth).abs() <= Z975 * spj.se[h] {
                spj_cells[h].coverage += 1.0;
            }
        }
    }
    let r = reps as f64;
    for h in 0..3 {
        fe_cells[h].bias /= r;
        fe_cells[h].coverage /= r;
        spj_cells[h].bias /= r;
        spj_cells[h].coverage /= r;
    }
    (fe_cells, spj_cells)
}

/// The full three-T bias/coverage study behind the module-level table
/// and the model card. Ignored in the default (debug) run because 10,800
/// panel fits take minutes unoptimized — the house pattern for coverage
/// Monte Carlos (see tsecon-var's simultaneous_bands.rs). Run with:
///
/// ```text
/// cargo test -p tsecon-panel --release --test spj_properties -- --ignored --nocapture
/// ```
///
/// The always-on `spj_bias_cut_smoke_at_t20` below keeps the headline
/// claim asserted on every run.
#[test]
#[ignore = "Monte Carlo: run in release with --ignored --nocapture"]
fn spj_cuts_nickell_bias_and_restores_coverage_where_fe_undercovers() {
    let reps = 300usize;
    let mut stream = Stream::new(20260818);

    let mut rows = Vec::new();
    for &t_len in &[20usize, 40, 80] {
        rows.push((t_len, run_mc(&mut stream, t_len, reps)));
    }
    for &(t_len, (fe, spj)) in &rows {
        for h in 1..=2usize {
            println!(
                "T={t_len:3} h={h}  bias_fe={:+.4}  bias_spj={:+.4}  ratio={:4.1}x  \
                 cov_fe={:.3}  cov_spj={:.3}",
                fe[h].bias,
                spj[h].bias,
                fe[h].bias.abs() / spj[h].bias.abs().max(1e-12),
                fe[h].coverage,
                spj[h].coverage
            );
        }
    }

    let (_, (fe20, spj20)) = rows[0];
    let (_, (fe40, spj40)) = rows[1];
    let (_, (fe80, spj80)) = rows[2];

    // (1) The FE Nickell bias at h=2 is real and O(1/T): visible at
    // T=20 and shrinking by at least half from T=20 to T=80 (the O(1/T)
    // rate predicts a quarter; assert conservatively for MC noise).
    assert!(
        fe20[2].bias.abs() > 0.08,
        "expected a visible FE Nickell bias at T=20 h=2, got {}",
        fe20[2].bias
    );
    assert!(
        fe80[2].bias.abs() < fe20[2].bias.abs() / 2.0,
        "FE bias should shrink like O(1/T): T=20 {} vs T=80 {}",
        fe20[2].bias,
        fe80[2].bias
    );
    assert!(
        fe40[2].bias.abs() < fe20[2].bias.abs(),
        "FE bias should shrink with T: T=20 {} vs T=40 {}",
        fe20[2].bias,
        fe40[2].bias
    );

    // (2) SPJ removes most of the bias where the bias is visible
    // (measured 15.2x at T=20 h=2 and 2.9x at T=40 h=2 on this seed;
    // ~9x at T=40 in a 2000-rep rerun).
    assert!(
        spj20[2].bias.abs() < fe20[2].bias.abs() / 3.0,
        "SPJ should cut the T=20 h=2 bias by >3x: FE {} vs SPJ {}",
        fe20[2].bias,
        spj20[2].bias
    );
    assert!(
        spj40[2].bias.abs() < fe40[2].bias.abs() / 1.8,
        "SPJ should cut the T=40 h=2 bias: FE {} vs SPJ {}",
        fe40[2].bias,
        spj40[2].bias
    );
    // At T=80 both biases are within Monte-Carlo noise of zero; assert
    // only that SPJ stays small rather than manufacturing a ratio of
    // two noise terms (the 2000-rep rerun puts it at +0.002).
    assert!(
        spj80[2].bias.abs() < 0.03,
        "SPJ T=80 h=2 bias should be near zero, got {}",
        spj80[2].bias
    );

    // (3) Coverage, honestly banded (MC se ~ 2pp per cell at 300 reps).
    // See the module-level table for the measured numbers and the
    // reading. Where FE is already near-valid (T >= 40) the two coincide
    // within noise; SPJ must never fall materially below FE.
    for (t_label, fe, spj) in [(20, fe20, spj20), (40, fe40, spj40), (80, fe80, spj80)] {
        for h in 1..=2usize {
            assert!(
                spj[h].coverage >= fe[h].coverage - 0.05,
                "T={t_label} h={h}: SPJ coverage {} materially below FE {}",
                spj[h].coverage,
                fe[h].coverage
            );
        }
    }
    assert!(
        fe20[2].coverage < 0.80,
        "FE + DK should undercover clearly at T=20 h=2, got {}",
        fe20[2].coverage
    );
    assert!(
        spj20[2].coverage > fe20[2].coverage + 0.04,
        "SPJ should out-cover FE at T=20 h=2: FE {} vs SPJ {}",
        fe20[2].coverage,
        spj20[2].coverage
    );
    assert!(
        (0.80..=0.99).contains(&spj20[2].coverage),
        "SPJ T=20 h=2 coverage {} outside the honest [0.80, 0.99] band \
         (nominal 95% is NOT reached at T=20; the card says so)",
        spj20[2].coverage
    );
    assert!(
        (0.84..=0.99).contains(&spj40[2].coverage),
        "SPJ T=40 h=2 coverage {} outside [0.84, 0.99]",
        spj40[2].coverage
    );
    assert!(
        (0.84..=0.99).contains(&spj80[2].coverage),
        "SPJ T=80 h=2 coverage {} outside [0.84, 0.99]",
        spj80[2].coverage
    );
    assert!(
        fe80[2].coverage > fe20[2].coverage,
        "FE coverage should recover as T grows: T=20 {} vs T=80 {}",
        fe20[2].coverage,
        fe80[2].coverage
    );
}

/// Always-on Monte Carlo smoke for the headline claim (the full study is
/// the `#[ignore]`d test above): at T = 20 the FE Nickell bias on
/// `irf[2]` is visible and the SPJ correction removes most of it, and
/// the SPJ+adjusted-score-DK interval covers no worse than the FE one.
/// 60 replications keep the debug run to a few seconds; measured on this
/// seed: bias_fe = -0.1264, bias_spj = +0.0040 (31.6x), cov_fe = 0.733,
/// cov_spj = 0.783.
#[test]
fn spj_bias_cut_smoke_at_t20() {
    let reps = 60usize;
    let mut stream = Stream::new(20260818);
    let (fe, spj) = run_mc(&mut stream, 20, reps);
    println!(
        "smoke T=20 h=2: bias_fe={:+.4} bias_spj={:+.4} cov_fe={:.3} cov_spj={:.3}",
        fe[2].bias, spj[2].bias, fe[2].coverage, spj[2].coverage
    );
    assert!(
        fe[2].bias.abs() > 0.08,
        "expected a visible FE Nickell bias at T=20 h=2, got {}",
        fe[2].bias
    );
    assert!(
        spj[2].bias.abs() < fe[2].bias.abs() / 2.5,
        "SPJ should cut the T=20 h=2 bias by >2.5x: FE {} vs SPJ {}",
        fe[2].bias,
        spj[2].bias
    );
    assert!(
        spj[2].coverage >= fe[2].coverage,
        "SPJ coverage {} should not fall below FE {} at T=20 h=2",
        spj[2].coverage,
        fe[2].coverage
    );
}

/// Documented relation between the two corrections: when the two
/// half-samples are the same set of regression rows — horizon 0, no lag
/// controls, even T — the DJ and SPJ POINT estimates coincide exactly
/// (same fits, same `2F - (A+B)/2` combination). The SEs still differ by
/// construction (full-sample plug-in vs adjusted-score sandwich), which
/// is also asserted so the routes cannot silently collapse into one.
#[test]
fn spj_and_dj_points_coincide_where_the_halves_are_identical() {
    let mut stream = Stream::new(7);
    let (data, shock) = simulate_panel(&mut stream, 10, 24);

    let mut dj = PanelLpConfig::new(0, 0, PanelSeType::ClusterEntity);
    dj.jackknife = true;
    let mut spj = PanelLpConfig::new(0, 0, PanelSeType::ClusterEntity);
    spj.bias_correction = LpBiasCorrection::Spj;

    let a = panel_lp(&data, &shock, &dj).expect("dj lp");
    let b = panel_lp(&data, &shock, &spj).expect("spj lp");
    assert!(
        (a.irf[0] - b.irf[0]).abs() < 1e-12,
        "h=0, no lags, even T: DJ {} and SPJ {} must coincide",
        a.irf[0],
        b.irf[0]
    );
    assert!(
        (a.se[0] - b.se[0]).abs() > 1e-12,
        "the SPJ adjusted-score SE must differ from the DJ plug-in SE"
    );
    assert_eq!(a.bias_correction, LpBiasCorrection::DhaeneJochmans);
    assert_eq!(b.bias_correction, LpBiasCorrection::Spj);
    assert!(a.jackknife && !b.jackknife);
}

/// ... and where the halves are NOT identical (lags/leads in play), the
/// two corrections genuinely differ: the DJ windowed halves re-burn lags
/// and truncate leads at the split, the MSS halves keep them.
#[test]
fn spj_and_dj_points_differ_once_lags_and_leads_cross_the_split() {
    let mut stream = Stream::new(11);
    let (data, shock) = simulate_panel(&mut stream, 10, 24);

    let mut dj = PanelLpConfig::new(2, 1, PanelSeType::ClusterEntity);
    dj.jackknife = true;
    let mut spj = PanelLpConfig::new(2, 1, PanelSeType::ClusterEntity);
    spj.bias_correction = LpBiasCorrection::Spj;

    let a = panel_lp(&data, &shock, &dj).expect("dj lp");
    let b = panel_lp(&data, &shock, &spj).expect("spj lp");
    assert!(
        (a.irf[2] - b.irf[2]).abs() > 1e-10,
        "with lags and a positive horizon the two corrections use \
         different half-samples and must not coincide"
    );
}

#[test]
fn spj_with_nonrobust_se_is_refused() {
    let mut stream = Stream::new(3);
    let (data, shock) = simulate_panel(&mut stream, 8, 30);
    let mut cfg = PanelLpConfig::new(2, 1, PanelSeType::NonRobust);
    cfg.bias_correction = LpBiasCorrection::Spj;
    assert!(matches!(
        panel_lp(&data, &shock, &cfg).unwrap_err(),
        PanelError::InvalidArgument { .. }
    ));
}

#[test]
fn spj_combined_with_the_dj_flag_is_refused_as_ambiguous() {
    let mut stream = Stream::new(4);
    let (data, shock) = simulate_panel(&mut stream, 8, 30);
    let mut cfg = PanelLpConfig::new(2, 1, PanelSeType::ClusterEntity);
    cfg.jackknife = true;
    cfg.bias_correction = LpBiasCorrection::Spj;
    assert!(matches!(
        panel_lp(&data, &shock, &cfg).unwrap_err(),
        PanelError::InvalidArgument { .. }
    ));
    // jackknife = true WITH the matching enum value is not a conflict.
    cfg.bias_correction = LpBiasCorrection::DhaeneJochmans;
    let res = panel_lp(&data, &shock, &cfg).expect("dj via either knob");
    assert_eq!(res.bias_correction, LpBiasCorrection::DhaeneJochmans);
}

/// A panel long enough for the full-sample fit but too short for the
/// median split must fail as a *sample-size* error naming the half, not
/// as a downstream singular design. Here T = 8 with hmax = 3 and one lag
/// leaves 4 usable rows at h = 3 -> a 2/2 split, but the common shock
/// plus its lag need at least 3 distinct demeaned periods per half, so
/// the tightest feasible configuration is refused one step earlier by
/// the degrees-of-freedom-per-half guard.
#[test]
fn t_too_short_for_the_split_raises_a_sample_size_error() {
    let mut stream = Stream::new(5);
    let (data, shock) = simulate_panel(&mut stream, 6, 8);
    let mut cfg = PanelLpConfig::new(3, 1, PanelSeType::ClusterEntity);
    cfg.bias_correction = LpBiasCorrection::Spj;
    let err = panel_lp(&data, &shock, &cfg).unwrap_err();
    match err {
        PanelError::InsufficientObservations { what, .. } => {
            assert!(
                what.contains("split-panel jackknife"),
                "error should name the split, got: {what}"
            );
        }
        other => panic!("expected InsufficientObservations, got {other:?}"),
    }
}

/// Odd usable-row counts give the extra row to the FIRST half (the pLP
/// floor-median convention): with T = 9, no lags, h = 0 there are 9
/// usable rows and the split is 5/4. Pinned indirectly: the fit succeeds
/// and matches a hand-built 5/4 recomputation of the combination.
#[test]
fn odd_row_counts_split_five_four_with_the_extra_row_first() {
    let (n, t) = (6usize, 9usize);
    let shock: Vec<f64> = (0..t).map(|tt| (0.7 * tt as f64).sin()).collect();
    let mut stream = Stream::new(6);
    let outcome = Mat::from_fn(n, t, |i, tt| {
        0.3 * i as f64 + 0.8 * shock[tt] + 0.1 * gaussian(&mut stream)
    });
    let data = PanelData::balanced(outcome.clone(), vec![]).expect("balanced");

    let mut cfg = PanelLpConfig::new(0, 0, PanelSeType::ClusterEntity);
    cfg.bias_correction = LpBiasCorrection::Spj;
    let res = panel_lp(&data, &shock, &cfg).expect("spj lp");

    // Hand recomputation: within OLS on rows [0..9), [0..5), [5..9).
    let within = |rows: std::ops::Range<usize>| -> f64 {
        let m = rows.len() as f64;
        let smean: f64 = rows.clone().map(|tt| shock[tt]).sum::<f64>() / m;
        let mut num = 0.0;
        let mut den = 0.0;
        for i in 0..n {
            let ymean: f64 = rows.clone().map(|tt| outcome[(i, tt)]).sum::<f64>() / m;
            for tt in rows.clone() {
                let xs = shock[tt] - smean;
                num += xs * (outcome[(i, tt)] - ymean);
                den += xs * xs;
            }
        }
        num / den
    };
    let expect = 2.0 * within(0..9) - 0.5 * (within(0..5) + within(5..9));
    assert!(
        (res.irf[0] - expect).abs() < 1e-10,
        "5/4 split: got {}, hand recomputation {}",
        res.irf[0],
        expect
    );
}
