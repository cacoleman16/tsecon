//! Property tests for `l1_trend_filter` and `boosting` beyond the goldens:
//! the `lam -> 0` and `lam_max` limits, exact piecewise structure (zero
//! differences off the knots), scale equivariance, the `converged` flag
//! proven to fire, the L2 form against an independent dense solve, the
//! teaching errors, wall time at `n = 10000`; boosting's monotone RSS,
//! its small-step limit (the OLS fit on the selected support), support
//! recovery under AIC stopping, determinism, teaching errors, and wall
//! time at `n = 500, p = 50, n_steps = 500`.

// The independent dense Gaussian-elimination helpers below are written
// with explicit row/column indices on purpose: that is the form a reader
// can check against the textbook, and it is test code.
#![allow(clippy::needless_range_loop)]

mod common;

use std::time::Instant;

use common::{as_f64_vec, load_fixture, mat_from_cols, Lcg};
use tsecon_ml::{
    boosting, l1_trend_filter, BoostStop, BoostingOptions, MlError, Penalty, TrendFilterOptions,
};

fn diff_k(x: &[f64], k: usize) -> Vec<f64> {
    let mut v = x.to_vec();
    for _ in 0..k {
        v = v.windows(2).map(|w| w[1] - w[0]).collect();
    }
    v
}

fn l1(order: usize) -> TrendFilterOptions {
    TrendFilterOptions {
        order,
        penalty: Penalty::L1,
        ..TrendFilterOptions::default()
    }
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .fold(0.0f64, |m, (x, y)| m.max((x - y).abs()))
}

/// `lam = 0` returns the data bit for bit; a vanishing `lam` returns it
/// at `1e-10`. The trend is within `2^order * lam` of `y` for any `lam`
/// (`x = y - D' z` with `|z| <= lam`), so `lam = 1e-14 * lam_max` — of
/// order 1e-11 on these series — must land inside 1e-10 and does.
#[test]
fn trend_filter_lam_to_zero_returns_the_data() {
    let fx = load_fixture("convex.json");
    let y = as_f64_vec(&fx["series"]["pwl"]);
    for k in [1usize, 2] {
        let fit = l1_trend_filter(&y, 0.0, l1(k)).unwrap();
        assert_eq!(fit.trend, y, "k={k}: lam=0 must return y exactly");
        assert!(fit.converged && fit.n_iter == 0 && fit.objective == 0.0);
        let lam = 1e-14 * fit.lam_max;
        let tiny = l1_trend_filter(&y, lam, l1(k)).unwrap();
        let d = max_abs_diff(&tiny.trend, &y);
        assert!(d <= 1e-10, "k={k}: lam={lam:e} gives |trend - y| = {d:e}");
        // The bound is attained (every constraint active at +-lam), so
        // allow the rounding of y - (y - D'z).
        let ymax = y.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        assert!(
            d <= (1u32 << k) as f64 * lam + 64.0 * f64::EPSILON * ymax,
            "k={k}: the 2^k lam bound is violated: {d:e}"
        );
        println!("k={k}: lam={lam:e} gives |trend - y| = {d:e}");
    }
}

/// Just below `lam_max` the trend has at least one knot; at `lam_max` and
/// above it has none and is a polynomial (differences of order `k` zero).
#[test]
fn trend_filter_lam_max_is_the_knot_threshold() {
    let fx = load_fixture("convex.json");
    for (name, k) in [("pwl", 2usize), ("steps", 1), ("rw", 2), ("rw", 1)] {
        let y = as_f64_vec(&fx["series"][name]);
        let probe = l1_trend_filter(&y, 1.0, l1(k)).unwrap();
        let lm = probe.lam_max;
        let below = l1_trend_filter(&y, 0.999 * lm, l1(k)).unwrap();
        assert!(
            !below.knots.is_empty(),
            "{name} k={k}: no knot just below lam_max"
        );
        for f in [1.0, 1.001, 10.0] {
            let at = l1_trend_filter(&y, f * lm, l1(k)).unwrap();
            assert!(at.knots.is_empty(), "{name} k={k}: knots at {f} lam_max");
            let dx = diff_k(&at.trend, k);
            let scale = y.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            for (i, d) in dx.iter().enumerate() {
                assert!(
                    d.abs() <= 1e-9 * scale,
                    "{name} k={k}: D^k trend[{i}] = {d:e}"
                );
            }
        }
    }
}

/// The L1 trend is genuinely piecewise polynomial: off the reported knots
/// the `order`-th differences are zero to rounding (the active-set polish
/// solves the equality-constrained problem exactly), the knot count is
/// small relative to `n`, and on the piecewise-linear and step designs the
/// knots sit near the true breakpoints.
#[test]
fn trend_filter_is_exactly_piecewise_off_the_knots() {
    let fx = load_fixture("convex.json");
    // (series, order, lam fraction, true breakpoints, max knots)
    let cases: [(&str, usize, f64, &[usize], usize); 2] = [
        ("pwl", 2, 0.2, &[35, 80, 115], 12),
        ("steps", 1, 0.5, &[30, 60, 90], 12),
    ];
    for (name, k, frac, breaks, max_knots) in cases {
        let y = as_f64_vec(&fx["series"][name]);
        let probe = l1_trend_filter(&y, 1.0, l1(k)).unwrap();
        let fit = l1_trend_filter(&y, frac * probe.lam_max, l1(k)).unwrap();
        assert!(fit.converged);
        assert!(
            !fit.knots.is_empty() && fit.knots.len() <= max_knots,
            "{name}: {} knots",
            fit.knots.len()
        );
        let dx = diff_k(&fit.trend, k);
        let scale = y.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let mut worst_off = 0.0f64;
        for (i, d) in dx.iter().enumerate() {
            if !fit.knots.contains(&i) {
                worst_off = worst_off.max(d.abs());
            }
        }
        // The residue is the banded solve's rounding on the equality
        // constraints, of order eps * lam * ||D D'||; 1e-10 * max(|y|, lam)
        // is four decades above it and six below the knot threshold.
        let lam = frac * probe.lam_max;
        assert!(
            worst_off <= 1e-10 * scale.max(lam),
            "{name}: off-knot difference {worst_off:e} is not zero to rounding"
        );
        // Each true breakpoint has a knot within a few observations. The
        // knot index i refers to the difference D^k x at positions i..i+k,
        // so a break at b appears at i ~ b - 1.
        for &b in breaks {
            let near = fit
                .knots
                .iter()
                .any(|&i| (i as i64 - (b as i64 - 1)).abs() <= 4);
            assert!(
                near,
                "{name}: no knot near breakpoint {b}; knots {:?}",
                fit.knots
            );
        }
        println!(
            "{name} k={k}: {} knots at {:?}, off-knot max {worst_off:e}",
            fit.knots.len(),
            fit.knots
        );
    }
}

/// Scale equivariance: `y -> c y`, `lam -> c lam` scales the trend by `c`.
/// The relative-gap stopping rule (and the closed-form polish) make this
/// exact to rounding across twelve decades.
#[test]
fn trend_filter_is_scale_equivariant() {
    let fx = load_fixture("convex.json");
    let y = as_f64_vec(&fx["series"]["ar_trend"]);
    for k in [1usize, 2] {
        let base = l1_trend_filter(&y, 20.0, l1(k)).unwrap();
        let mag = base.trend.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        for &c in &[1e-6, 1e-3, 1e3, 1e6] {
            let yc: Vec<f64> = y.iter().map(|v| v * c).collect();
            let fit = l1_trend_filter(&yc, 20.0 * c, l1(k)).unwrap();
            assert_eq!(fit.knots, base.knots, "k={k} c={c:e}: knot set moved");
            let scaled: Vec<f64> = fit.trend.iter().map(|v| v / c).collect();
            let d = max_abs_diff(&scaled, &base.trend);
            assert!(
                d <= 1e-10 * mag,
                "k={k} c={c:e}: |trend/c - trend_1| = {d:e}"
            );
        }
    }
}

/// The honesty flag fires two ways. A starved iteration budget returns
/// `converged = false` with a certified gap above `tol * objective` and a
/// finite trend, while the default budget converges at the same `tol`. A
/// `tol` below the certificate's floating-point floor (1e-14 here) also
/// returns `converged = false` — after the stall detector ends the loop in
/// far fewer iterations than the budget — with the honest gap, and the
/// trend it returns is still the converged one to 1e-8.
#[test]
fn trend_filter_converged_flag_fires_on_a_starved_budget() {
    let fx = load_fixture("convex.json");
    let y = as_f64_vec(&fx["series"]["rw"]);
    let opts = TrendFilterOptions {
        order: 2,
        penalty: Penalty::L1,
        tol: 1e-9,
        max_iter: 1,
    };
    let starved = l1_trend_filter(&y, 100.0, opts).unwrap();
    assert!(
        !starved.converged,
        "one Newton step cannot certify a 1e-9 gap"
    );
    assert_eq!(starved.n_iter, 1);
    assert!(starved.duality_gap > 1e-9 * starved.objective);
    assert!(starved.trend.iter().all(|v| v.is_finite()));
    let full = l1_trend_filter(
        &y,
        100.0,
        TrendFilterOptions {
            max_iter: 10_000,
            ..opts
        },
    )
    .unwrap();
    assert!(full.converged);
    assert!(full.duality_gap <= 1e-9 * full.objective);
    // A tighter tol is live: it cannot loosen the certificate.
    assert!(full.duality_gap <= starved.duality_gap);

    let below_floor = l1_trend_filter(
        &y,
        100.0,
        TrendFilterOptions {
            tol: 1e-14,
            max_iter: 10_000,
            ..opts
        },
    )
    .unwrap();
    assert!(
        !below_floor.converged,
        "1e-14 is below the certificate's floating-point floor"
    );
    assert!(
        below_floor.n_iter < 200,
        "stall detector did not fire: {} iterations",
        below_floor.n_iter
    );
    assert!(below_floor.duality_gap > 1e-14 * below_floor.objective);
    assert!(
        below_floor.duality_gap <= 1e-9 * below_floor.objective,
        "the returned gap is still tiny"
    );
    assert!(max_abs_diff(&below_floor.trend, &full.trend) <= 1e-8);
    println!(
        "below-floor tol: {} iterations, relative gap {:e}",
        below_floor.n_iter,
        below_floor.duality_gap / below_floor.objective
    );
}

/// The L2 form solves `(I + lam D'D) x = y`: checked against a dense
/// Gaussian elimination written here, for both orders, at 1e-10.
#[test]
fn trend_filter_l2_matches_a_dense_solve() {
    let mut rng = Lcg::new(31);
    let n = 60usize;
    let y: Vec<f64> = (0..n).map(|i| 0.05 * (i as f64) + rng.normal()).collect();
    for (k, lam) in [(1usize, 3.0), (2, 1600.0), (2, 0.7)] {
        let m = n - k;
        // Dense D.
        let stencil: Vec<f64> = if k == 1 {
            vec![-1.0, 1.0]
        } else {
            vec![1.0, -2.0, 1.0]
        };
        let mut a = vec![vec![0.0; n]; n];
        for (i, row) in a.iter_mut().enumerate() {
            row[i] = 1.0;
        }
        for r in 0..m {
            for ai in 0..=k {
                for bi in 0..=k {
                    a[r + ai][r + bi] += lam * stencil[ai] * stencil[bi];
                }
            }
        }
        // Gaussian elimination with partial pivoting.
        let mut b = y.clone();
        for c in 0..n {
            let piv = (c..n)
                .max_by(|&i, &j| a[i][c].abs().partial_cmp(&a[j][c].abs()).unwrap())
                .unwrap();
            a.swap(c, piv);
            b.swap(c, piv);
            for r in c + 1..n {
                let f = a[r][c] / a[c][c];
                if f != 0.0 {
                    for col in c..n {
                        a[r][col] -= f * a[c][col];
                    }
                    b[r] -= f * b[c];
                }
            }
        }
        let mut x = vec![0.0; n];
        for r in (0..n).rev() {
            let s: f64 = (r + 1..n).map(|c| a[r][c] * x[c]).sum();
            x[r] = (b[r] - s) / a[r][r];
        }
        let opts = TrendFilterOptions {
            order: k,
            penalty: Penalty::L2,
            ..TrendFilterOptions::default()
        };
        let fit = l1_trend_filter(&y, lam, opts).unwrap();
        let d = max_abs_diff(&fit.trend, &x);
        assert!(d <= 1e-10, "k={k} lam={lam}: |banded - dense| = {d:e}");
        // cycle + trend reconstructs y (to an ulp: cycle is y - trend).
        for i in 0..n {
            let back = fit.trend[i] + fit.cycle[i];
            assert!((back - y[i]).abs() <= 4.0 * f64::EPSILON * y[i].abs().max(1.0));
        }
    }
}

/// Every teaching error names the argument and states the fix; nothing
/// panics.
#[test]
fn trend_filter_teaching_errors() {
    let y: Vec<f64> = (0..20).map(|i| (i as f64).sin()).collect();
    let short = l1_trend_filter(&y[..2], 1.0, l1(2)).unwrap_err();
    assert_eq!(short, MlError::InsufficientData { needed: 3, got: 2 });
    assert_eq!(
        short.to_string(),
        "insufficient data: 2 observations, at least 3 required"
    );
    assert_eq!(
        l1_trend_filter(&[], 1.0, l1(1)).unwrap_err(),
        MlError::InsufficientData { needed: 2, got: 0 }
    );
    let mut bad = y.clone();
    bad[5] = f64::NAN;
    assert_eq!(
        l1_trend_filter(&bad, 1.0, l1(2)).unwrap_err(),
        MlError::NonFinite { what: "y" }
    );
    bad[5] = f64::INFINITY;
    assert_eq!(
        l1_trend_filter(&bad, 1.0, l1(2)).unwrap_err(),
        MlError::NonFinite { what: "y" }
    );
    for lam in [-1.0, f64::NAN, f64::INFINITY] {
        let e = l1_trend_filter(&y, lam, l1(2)).unwrap_err().to_string();
        assert!(e.contains("lam must be finite and non-negative"), "{e}");
    }
    for order in [0usize, 3] {
        let e = l1_trend_filter(&y, 1.0, l1(order)).unwrap_err().to_string();
        assert!(e.contains("order must be 1") && e.contains("or 2"), "{e}");
    }
    let e = l1_trend_filter(&y, 1.0, TrendFilterOptions { tol: 0.0, ..l1(2) })
        .unwrap_err()
        .to_string();
    assert!(e.contains("tol must be finite and positive"), "{e}");
    let e = l1_trend_filter(
        &y,
        1.0,
        TrendFilterOptions {
            max_iter: 0,
            ..l1(2)
        },
    )
    .unwrap_err()
    .to_string();
    assert!(e.contains("max_iter must be at least 1"), "{e}");
    // The lam = 0 and L2 paths still validate.
    assert!(l1_trend_filter(&bad, 0.0, l1(2)).is_err());
    let l2 = TrendFilterOptions {
        penalty: Penalty::L2,
        ..l1(2)
    };
    assert!(l1_trend_filter(&bad, 1600.0, l2).is_err());
    assert!(l1_trend_filter(&y[..1], 1600.0, l2).is_err());
}

/// Wall time at `n = 10000`: every interior-point step is a banded `O(n)`
/// solve, so the whole fit is a fraction of a second even unoptimized. The
/// figure is printed for the report; the assertion only guards against an
/// `O(n^2)` regression.
#[test]
fn trend_filter_n_10000_is_fast() {
    let mut rng = Lcg::new(2026);
    let n = 10_000usize;
    let mut y = Vec::with_capacity(n);
    let (mut level, mut slope) = (0.0f64, 0.01f64);
    for i in 0..n {
        if i % 1000 == 0 {
            slope = 0.02 * rng.normal();
        }
        level += slope;
        y.push(level + 0.3 * rng.normal());
    }
    let t0 = Instant::now();
    let fit = l1_trend_filter(&y, 50.0, l1(2)).unwrap();
    let secs = t0.elapsed().as_secs_f64();
    assert!(
        fit.converged,
        "n=10000 did not converge in {} iterations",
        fit.n_iter
    );
    println!(
        "l1_trend_filter n=10000: {secs:.3} s wall, {} iterations, {} knots, gap {:e}",
        fit.n_iter,
        fit.knots.len(),
        fit.duality_gap / fit.objective
    );
    assert!(
        secs < 30.0,
        "n=10000 took {secs} s — the O(n) structure is lost"
    );
    let t1 = Instant::now();
    let l2 = l1_trend_filter(
        &y,
        1600.0,
        TrendFilterOptions {
            penalty: Penalty::L2,
            ..l1(2)
        },
    )
    .unwrap();
    println!(
        "l2 (HP form) n=10000: {:.4} s wall",
        t1.elapsed().as_secs_f64()
    );
    assert!(l2.converged);
}

/// A seeded sparse design: `n` rows, `p` standardized columns, the first
/// three carrying signal.
fn sparse_design(seed: u64, n: usize, p: usize, noise: f64) -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut rng = Lcg::new(seed);
    let mut cols: Vec<Vec<f64>> = (0..p)
        .map(|_| (0..n).map(|_| rng.normal()).collect())
        .collect();
    for c in cols.iter_mut() {
        let m = c.iter().sum::<f64>() / n as f64;
        let sd = (c.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / n as f64).sqrt();
        for v in c.iter_mut() {
            *v = (*v - m) / sd;
        }
    }
    let beta = [3.0, -2.0, 1.5];
    let mut y: Vec<f64> = (0..n)
        .map(|i| (0..3).map(|j| cols[j][i] * beta[j]).sum::<f64>() + noise * rng.normal())
        .collect();
    let m = y.iter().sum::<f64>() / n as f64;
    for v in y.iter_mut() {
        *v -= m;
    }
    (cols, y)
}

/// Ordinary least squares on the given columns by normal equations with
/// Gaussian elimination (independent of the crate).
fn ols_on(cols: &[Vec<f64>], y: &[f64], support: &[usize]) -> Vec<f64> {
    let q = support.len();
    let mut a = vec![vec![0.0; q + 1]; q];
    for (r, &jr) in support.iter().enumerate() {
        for (c, &jc) in support.iter().enumerate() {
            a[r][c] = cols[jr].iter().zip(&cols[jc]).map(|(u, v)| u * v).sum();
        }
        a[r][q] = cols[jr].iter().zip(y).map(|(u, v)| u * v).sum();
    }
    for c in 0..q {
        let piv = (c..q)
            .max_by(|&i, &j| a[i][c].abs().partial_cmp(&a[j][c].abs()).unwrap())
            .unwrap();
        a.swap(c, piv);
        for r in c + 1..q {
            let f = a[r][c] / a[c][c];
            for col in c..=q {
                a[r][col] -= f * a[c][col];
            }
        }
    }
    let mut b = vec![0.0; q];
    for r in (0..q).rev() {
        let s: f64 = (r + 1..q).map(|c| a[r][c] * b[c]).sum();
        b[r] = (a[r][q] - s) / a[r][r];
    }
    b
}

/// The residual sum of squares never rises along the path (each step is a
/// shrunken least-squares improvement), the first step's trace is exactly
/// `nu` (`tr(nu H_j) = nu`), the trace stays inside `(0, n)`, and the
/// AIC-chosen step is a genuine minimizer of the path. (The trace is *not*
/// asserted monotone: the boosting operator is not symmetric, so
/// `x_j' B x_j` can exceed `x_j' x_j` near the saturated `nu = 1` limit
/// and the trace dips by ~1e-5 there — the dense transcription agrees.)
#[test]
fn boosting_rss_path_is_nonincreasing() {
    let (cols, y) = sparse_design(5, 120, 10, 1.0);
    let x = mat_from_cols(&cols);
    for nu in [0.05, 0.1, 0.5, 1.0] {
        let opts = BoostingOptions {
            learning_rate: nu,
            n_steps: 300,
            stop: BoostStop::Aic,
        };
        let fit = boosting(x.as_ref(), &y, opts, None).unwrap();
        for w in fit.rss_path.windows(2) {
            assert!(
                w[1] <= w[0] * (1.0 + 1e-12),
                "nu={nu}: RSS rose {} -> {}",
                w[0],
                w[1]
            );
        }
        assert!(
            (fit.df_path[0] - nu).abs() <= 1e-12,
            "nu={nu}: first df {}",
            fit.df_path[0]
        );
        for (m, &df) in fit.df_path.iter().enumerate() {
            assert!(df > 0.0 && df < 120.0, "nu={nu}: df[{m}] = {df}");
        }
        let best = fit.best_step;
        assert!(fit.aic_path.iter().all(|&a| a >= fit.aic_path[best]));
        assert_eq!(fit.coef, fit.coef_path[best]);
        assert!(fit.df_path[best] > 0.0 && fit.df_path[best] < 120.0);
    }
}

/// With a tiny learning rate and many steps, `L2` boosting converges to the
/// ordinary-least-squares fit on the set of columns it ever selects
/// (Bühlmann & Yu 2003, section 3): the boosting operator's residual
/// projection shrinks geometrically toward the OLS projection.
#[test]
fn boosting_small_rate_many_steps_approaches_ols_on_selected_columns() {
    let (cols, y) = sparse_design(9, 150, 6, 0.5);
    let x = mat_from_cols(&cols);
    let opts = BoostingOptions {
        learning_rate: 0.05,
        n_steps: 4_000,
        stop: BoostStop::None,
    };
    let fit = boosting(x.as_ref(), &y, opts, None).unwrap();
    let mut support: Vec<usize> = fit.selected.clone();
    support.sort_unstable();
    support.dedup();
    let ols = ols_on(&cols, &y, &support);
    let mut worst = 0.0f64;
    for (q, &j) in support.iter().enumerate() {
        worst = worst.max((fit.coef[j] - ols[q]).abs());
    }
    for j in 0..6 {
        if !support.contains(&j) {
            assert_eq!(fit.coef[j], 0.0);
        }
    }
    println!(
        "boosting nu=0.05, 4000 steps: support {:?}, |coef - OLS| = {worst:e}",
        support
    );
    assert!(
        worst <= 1e-6,
        "coefficients {:?} vs OLS {:?}",
        fit.coef,
        ols
    );
    // The rss path ends at the OLS residual on that support.
    let ols_fit: Vec<f64> = (0..150)
        .map(|i| {
            support
                .iter()
                .enumerate()
                .map(|(q, &j)| cols[j][i] * ols[q])
                .sum()
        })
        .collect();
    let ols_rss: f64 = y.iter().zip(&ols_fit).map(|(a, b)| (a - b) * (a - b)).sum();
    let last = *fit.rss_path.last().unwrap();
    assert!((last - ols_rss).abs() <= 1e-8 * ols_rss);
}

/// On a sparse truth the AIC-stopped model recovers the support: the true
/// signals are nonzero with the right sign and the noise columns stay at
/// exactly zero or negligible.
#[test]
fn boosting_aic_recovers_the_sparse_support() {
    let (cols, y) = sparse_design(11, 200, 12, 0.5);
    let x = mat_from_cols(&cols);
    let fit = boosting(x.as_ref(), &y, BoostingOptions::default(), None).unwrap();
    assert!(fit.best_step + 1 < 500, "AIC ran to the end of the path");
    let signs = [1.0, -1.0, 1.0];
    for j in 0..3 {
        assert!(fit.coef[j] * signs[j] > 0.5, "signal {j}: {}", fit.coef[j]);
    }
    let noise_max = (3..12).map(|j| fit.coef[j].abs()).fold(0.0f64, f64::max);
    assert!(noise_max <= 0.1, "noise coefficients up to {noise_max}");
    let n_noise_selected = (3..12).filter(|&j| fit.coef[j] != 0.0).count();
    println!(
        "boosting AIC: best_step {}, df {:.3}, {} of 9 noise columns touched (max |coef| {noise_max:.3e})",
        fit.best_step,
        fit.df_path[fit.best_step],
        n_noise_selected
    );
    // The fixture's own sparse design, through the crate.
    let fx = load_fixture("convex.json");
    let design = &fx["boost_designs"]["sparse"];
    let xs = common::as_mat(&design["X"]);
    let ys = as_f64_vec(&design["y"]);
    let f2 = boosting(xs.as_ref(), &ys, BoostingOptions::default(), None).unwrap();
    let truth = as_f64_vec(&design["true_beta"]);
    for j in 0..truth.len() {
        if truth[j] != 0.0 {
            assert!(f2.coef[j] * truth[j] > 0.0 && (f2.coef[j] - truth[j]).abs() < 0.5);
        } else {
            assert!(f2.coef[j].abs() < 0.15, "noise column {j}: {}", f2.coef[j]);
        }
    }
}

/// Seedless: the selection sequence and every path are bit-identical
/// across runs, and identical whether or not `x_test` is passed.
#[test]
fn boosting_is_deterministic() {
    let (cols, y) = sparse_design(3, 80, 7, 0.8);
    let x = mat_from_cols(&cols);
    let opts = BoostingOptions {
        learning_rate: 0.2,
        n_steps: 150,
        stop: BoostStop::Aic,
    };
    let a = boosting(x.as_ref(), &y, opts, None).unwrap();
    let b = boosting(x.as_ref(), &y, opts, None).unwrap();
    assert_eq!(a, b);
    let xt = mat_from_cols(&cols.iter().map(|c| c[..5].to_vec()).collect::<Vec<_>>());
    let c = boosting(x.as_ref(), &y, opts, Some(xt.as_ref())).unwrap();
    assert_eq!(a.selected, c.selected);
    assert_eq!(a.coef_path, c.coef_path);
    assert_eq!(a.fitted[..5].to_vec(), c.predicted.clone().unwrap());
    // stop = None reports the last step of the same path.
    let d = boosting(
        x.as_ref(),
        &y,
        BoostingOptions {
            stop: BoostStop::None,
            ..opts
        },
        None,
    )
    .unwrap();
    assert_eq!(d.best_step, 149);
    assert_eq!(d.coef_path, a.coef_path);
    assert_eq!(d.aic_path, a.aic_path);
}

/// Teaching errors name the argument; nothing panics.
#[test]
fn boosting_teaching_errors() {
    let (cols, y) = sparse_design(1, 30, 4, 0.5);
    let x = mat_from_cols(&cols);
    let opts = BoostingOptions::default();
    let e = boosting(
        mat_from_cols(&cols.iter().map(|c| c[..2].to_vec()).collect::<Vec<_>>()).as_ref(),
        &y[..2],
        opts,
        None,
    )
    .unwrap_err();
    assert_eq!(e, MlError::InsufficientData { needed: 3, got: 2 });
    assert_eq!(
        e.to_string(),
        "insufficient data: 2 observations, at least 3 required"
    );
    let mut bad = cols.clone();
    bad[1][3] = f64::NAN;
    assert_eq!(
        boosting(mat_from_cols(&bad).as_ref(), &y, opts, None).unwrap_err(),
        MlError::NonFinite { what: "x" }
    );
    let mut ybad = y.clone();
    ybad[0] = f64::INFINITY;
    assert_eq!(
        boosting(x.as_ref(), &ybad, opts, None).unwrap_err(),
        MlError::NonFinite { what: "y" }
    );
    for nu in [0.0, -0.1, 1.5, f64::NAN] {
        let e = boosting(
            x.as_ref(),
            &y,
            BoostingOptions {
                learning_rate: nu,
                ..opts
            },
            None,
        )
        .unwrap_err()
        .to_string();
        assert!(e.contains("learning_rate must lie in (0, 1]"), "{e}");
    }
    let e = boosting(x.as_ref(), &y, BoostingOptions { n_steps: 0, ..opts }, None)
        .unwrap_err()
        .to_string();
    assert!(e.contains("n_steps must be at least 1"), "{e}");
    assert_eq!(
        boosting(x.as_ref(), &y[..10], opts, None).unwrap_err(),
        MlError::DimensionMismatch {
            what: "y length must equal the number of rows of x",
            expected: 30,
            got: 10
        }
    );
    let xt = mat_from_cols(
        &cols[..3]
            .iter()
            .map(|c| c[..5].to_vec())
            .collect::<Vec<_>>(),
    );
    let e = boosting(x.as_ref(), &y, opts, Some(xt.as_ref())).unwrap_err();
    assert_eq!(
        e,
        MlError::DimensionMismatch {
            what: "x_test must have the same number of columns as x",
            expected: 4,
            got: 3
        }
    );
    let mut xt_bad = cols.iter().map(|c| c[..5].to_vec()).collect::<Vec<_>>();
    xt_bad[0][0] = f64::NAN;
    assert_eq!(
        boosting(x.as_ref(), &y, opts, Some(mat_from_cols(&xt_bad).as_ref())).unwrap_err(),
        MlError::NonFinite { what: "x_test" }
    );
    let zeros = vec![vec![0.0; 30]; 3];
    let e = boosting(mat_from_cols(&zeros).as_ref(), &y, opts, None)
        .unwrap_err()
        .to_string();
    assert!(e.contains("every column of x has zero norm"), "{e}");
    // A single zero column among live ones is simply never selected.
    let mut one_zero = cols.clone();
    one_zero[2] = vec![0.0; 30];
    let fit = boosting(mat_from_cols(&one_zero).as_ref(), &y, opts, None).unwrap();
    assert!(fit.selected.iter().all(|&j| j != 2));
    assert_eq!(fit.coef[2], 0.0);
}

/// Wall time at `n = 500, p = 50, n_steps = 500` (the report figure): the
/// factored operator keeps each step at `O(n (p + m))`.
#[test]
fn boosting_n500_p50_is_fast() {
    let (cols, y) = sparse_design(77, 500, 50, 1.0);
    let x = mat_from_cols(&cols);
    let t0 = Instant::now();
    let fit = boosting(x.as_ref(), &y, BoostingOptions::default(), None).unwrap();
    let secs = t0.elapsed().as_secs_f64();
    println!(
        "boosting n=500 p=50 n_steps=500: {secs:.3} s wall, best_step {}, df {:.2}",
        fit.best_step, fit.df_path[fit.best_step]
    );
    assert_eq!(fit.coef_path.len(), 500);
    assert!(secs < 60.0, "boosting took {secs} s");
}
