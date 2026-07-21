//! Property / behavioural tests for smooth local projections: the limits in
//! `lambda` (raw LP at 0, a straight line at infinity), the
//! Barnichon-Brownlees MSE claim under a smooth true IRF, and the
//! cross-validation bookkeeping.

use tsecon_lp::{lp, smooth_lp, LpSpec, SeKind, SmoothLpSpec};
use tsecon_rng::Stream;
use tsecon_stats::{ContinuousDist, StdNormal};

/// One standard-normal draw by inverse transform from a Philox uniform.
fn gaussian(stream: &mut Stream) -> f64 {
    let u = stream.uniform_f64().clamp(1e-12, 1.0 - 1e-12);
    StdNormal.ppf(u).expect("ppf on interior point")
}

/// The smooth hump `psi_j = (1 + 0.8 j) exp(-0.35 j)` used as the true IRF.
fn psi(j: usize) -> f64 {
    (1.0 + 0.8 * j as f64) * (-0.35 * j as f64).exp()
}

/// Simulate `y_t = sum_{j=0}^{30} psi_j e_{t-j} + noise_sd * w_t` and return
/// `(y, e)` of length `n` after a burn-in.
fn simulate_smooth_irf(stream: &mut Stream, n: usize, noise_sd: f64) -> (Vec<f64>, Vec<f64>) {
    let jmax = 30usize;
    let burn = jmax;
    let total = n + burn;
    let e: Vec<f64> = (0..total).map(|_| gaussian(stream)).collect();
    let mut y = Vec::with_capacity(total);
    for t in 0..total {
        let mut v = noise_sd * gaussian(stream);
        for j in 0..=jmax.min(t) {
            v += psi(j) * e[t - j];
        }
        y.push(v);
    }
    (y[burn..].to_vec(), e[burn..].to_vec())
}

#[test]
fn lambda_zero_reproduces_per_horizon_lp() {
    // THE internal-consistency anchor: with the default interpolating basis
    // (K = H + 1), lambda = 0 makes the stacked problem decouple horizon by
    // horizon, so the smoothed IRF must equal the per-horizon HAC-path lp()
    // point estimates to numerical precision — on a fresh simulated series,
    // not the fixture.
    let mut stream = Stream::new(20260721);
    let (y, e) = simulate_smooth_irf(&mut stream, 250, 1.0);
    let hmax = 10usize;
    let p = 3usize;

    let smooth = smooth_lp(&y, &e, &SmoothLpSpec::new(hmax, p).with_lambda(0.0))
        .expect("smooth_lp lambda=0");
    let raw = lp(&y, &e, LpSpec::new(hmax, p).with_hac(None)).expect("lp HAC");

    for h in 0..=hmax {
        let diff = (smooth.irf[h] - raw.irf[h]).abs();
        assert!(
            diff < 1e-8,
            "h={h}: lambda=0 smooth irf {} vs per-horizon lp {} (diff {diff:e})",
            smooth.irf[h],
            raw.irf[h]
        );
    }
    // The result's own raw path is that same estimator.
    assert_eq!(smooth.irf_raw, raw.irf);
    assert_eq!(smooth.se_raw, raw.se);
    assert_eq!(smooth.se_kind, SeKind::SmoothStackedHac);
}

#[test]
fn huge_lambda_drives_rth_differences_to_zero() {
    // lambda -> inf shrinks D_r theta to zero; with the uniform basis that
    // is exactly "IRF is a polynomial of degree r - 1 in h": a straight
    // line for the default r = 2, a constant for r = 1.
    let mut stream = Stream::new(31415);
    let (y, e) = simulate_smooth_irf(&mut stream, 250, 1.0);
    let hmax = 10usize;

    let line = smooth_lp(&y, &e, &SmoothLpSpec::new(hmax, 3).with_lambda(1e10))
        .expect("smooth_lp r=2 lambda huge");
    let scale = line
        .irf
        .iter()
        .fold(0.0_f64, |acc, v| acc.max(v.abs()))
        .max(1e-12);
    for h in 0..=hmax - 2 {
        let d2 = line.irf[h + 2] - 2.0 * line.irf[h + 1] + line.irf[h];
        assert!(
            d2.abs() < 1e-5 * scale,
            "h={h}: second difference {d2:e} not ~0 at lambda=1e10 (scale {scale})"
        );
    }

    let flat = smooth_lp(
        &y,
        &e,
        &SmoothLpSpec::new(hmax, 3)
            .with_penalty_order(1)
            .with_lambda(1e10),
    )
    .expect("smooth_lp r=1 lambda huge");
    for h in 0..hmax {
        let d1 = flat.irf[h + 1] - flat.irf[h];
        assert!(
            d1.abs() < 1e-5 * scale,
            "h={h}: first difference {d1:e} not ~0 at lambda=1e10 with r=1"
        );
    }
}

#[test]
fn smooth_lp_beats_raw_lp_in_mse_under_a_smooth_irf() {
    // The Barnichon-Brownlees claim: when the true IRF is smooth, the
    // penalized estimator (with cross-validated lambda) has lower MSE than
    // the raw per-horizon LP. Asserted on the AVERAGE across seeded Monte
    // Carlo replications and horizons — individual replications may go
    // either way.
    let reps = 40usize;
    let n = 120usize;
    let hmax = 10usize;
    let p = 2usize;
    let noise_sd = 3.0;
    let grid = vec![10.0, 100.0, 1e3, 1e4];
    let mut stream = Stream::new(20190301); // REStat 2019 vintage.

    let mut mse_raw = 0.0;
    let mut mse_smooth = 0.0;
    let mut smooth_wins = 0usize;
    for _ in 0..reps {
        let (y, e) = simulate_smooth_irf(&mut stream, n, noise_sd);
        let spec = SmoothLpSpec::new(hmax, p).with_cv(Some(grid.clone()), 4);
        let res = smooth_lp(&y, &e, &spec).expect("smooth_lp CV in MC");

        let (mut sse_s, mut sse_r) = (0.0, 0.0);
        for h in 0..=hmax {
            let truth = psi(h);
            sse_s += (res.irf[h] - truth).powi(2);
            sse_r += (res.irf_raw[h] - truth).powi(2);
        }
        mse_smooth += sse_s;
        mse_raw += sse_r;
        if sse_s < sse_r {
            smooth_wins += 1;
        }
    }

    assert!(
        mse_smooth < 0.85 * mse_raw,
        "smooth LP should beat raw LP on average under a smooth true IRF: \
         total smooth MSE {mse_smooth} vs raw {mse_raw} over {reps} reps"
    );
    // The average win should be broad-based, not carried by one outlier.
    assert!(
        smooth_wins * 3 > reps * 2,
        "smooth LP won only {smooth_wins}/{reps} replications"
    );
}

#[test]
fn cv_bookkeeping_is_consistent() {
    let mut stream = Stream::new(777);
    let (y, e) = simulate_smooth_irf(&mut stream, 200, 1.5);
    let grid = vec![0.1, 10.0, 1000.0];

    let cv = smooth_lp(
        &y,
        &e,
        &SmoothLpSpec::new(8, 2).with_cv(Some(grid.clone()), 4),
    )
    .expect("smooth_lp CV");
    assert_eq!(cv.cv_grid, grid);
    assert_eq!(cv.cv_scores.len(), grid.len());
    assert!(cv.cv_scores.iter().all(|s| s.is_finite() && *s > 0.0));
    assert!(
        grid.contains(&cv.lambda),
        "chosen lambda comes from the grid"
    );
    let best = cv.cv_scores.iter().cloned().fold(f64::INFINITY, f64::min);
    let chosen_idx = grid.iter().position(|&l| l == cv.lambda).expect("in grid");
    assert_eq!(
        cv.cv_scores[chosen_idx], best,
        "chosen lambda minimizes the CV score"
    );

    // Refitting at the chosen lambda as a Fixed value reproduces the CV fit.
    let fixed = smooth_lp(&y, &e, &SmoothLpSpec::new(8, 2).with_lambda(cv.lambda))
        .expect("smooth_lp fixed");
    assert_eq!(fixed.irf, cv.irf);
    assert_eq!(fixed.se, cv.se);
    assert!(fixed.cv_grid.is_empty() && fixed.cv_scores.is_empty());

    // The default grid path runs too and reports 17 log-spaced candidates.
    let default_cv = smooth_lp(&y, &e, &SmoothLpSpec::new(8, 2)).expect("default CV");
    assert_eq!(default_cv.cv_grid.len(), 17);
    assert!(grid_is_increasing(&default_cv.cv_grid));
}

fn grid_is_increasing(g: &[f64]) -> bool {
    g.windows(2).all(|w| w[0] < w[1])
}

#[test]
fn basis_is_a_partition_of_unity_and_ses_are_positive() {
    let mut stream = Stream::new(2024);
    let (y, e) = simulate_smooth_irf(&mut stream, 220, 1.0);
    for &(degree, n_basis, r) in &[(3usize, 9usize, 2usize), (2, 5, 1), (1, 4, 1)] {
        let spec = SmoothLpSpec::new(8, 2)
            .with_degree(degree)
            .with_n_basis(n_basis)
            .with_penalty_order(r)
            .with_lambda(25.0);
        let res = smooth_lp(&y, &e, &spec).expect("smooth_lp");
        for (h, row) in res.basis.iter().enumerate() {
            let sum: f64 = row.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-10,
                "degree={degree} K={n_basis}: basis row {h} sums to {sum}, not 1"
            );
            assert!(row.iter().all(|&b| b >= -1e-12), "negative basis value");
        }
        assert!(
            res.se.iter().all(|s| s.is_finite() && *s > 0.0),
            "SE path must be positive and finite"
        );
        assert_eq!(res.theta.len(), n_basis);
    }
}
