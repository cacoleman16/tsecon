//! Property and seeded Monte Carlo tests for `ardl_bounds_test` — the
//! claims the statsmodels golden cannot make on its own:
//!
//! * **Algebra**: the UECM is a reparameterisation of the levels ARDL (same
//!   SSR); the Wald F equals the restricted-vs-unrestricted SSR F; F, t and
//!   the p-values are invariant to rescaling `y` and `x`; every case's F is
//!   invariant to the origin of the trend, which only moves the constant
//!   and the long-run intercept (and in exactly the predicted way — the
//!   check that the trend runs `t + 1` over the full sample).
//! * **The response surface** (documents the verdict sliver): the surface
//!   p at a tabulated critical value is within 0.0013 / 0.0011 / 0.0003 /
//!   0.0001 of nominal at the 90 / 95 / 99 / 99.9% levels; the jump at
//!   `stat_star` is at most 0.017; `F = 0` gives `p = 1`. Neither
//!   continuity nor monotonicity is asserted (the surface has neither).
//! * **Size**: under the null of no level relationship (independent random
//!   walks, `T = 200`, one regressor, case 3, ARDL(1, 1)), the upper-bound
//!   p-value rejects at close to 5%; with a white-noise regressor, so does
//!   the lower-bound one.
//! * **Determinism**, and **refusals** that name the parameter.
//!
//! Measured numbers are printed (`--nocapture`) and quoted in the model
//! card; the assertions leave binomial Monte Carlo slack.

use tsecon_coint::tsecon_linalg::faer::linalg::solvers::SolveLstsq;
use tsecon_coint::tsecon_linalg::faer::Mat;
use tsecon_coint::{
    ardl_bounds_test, pss, ArdlOptions, ArdlResult, ArdlVerdict, CointError, DolsIc, PssCase,
};
use tsecon_rng::Stream;

fn draw_normal(stream: &mut Stream) -> f64 {
    tsecon_bootstrap::WildWeights::Normal.draw(stream)
}

/// A cointegrated system with `k` random-walk regressors.
fn coint_system(seed: u64, t: usize, k: usize) -> (Vec<f64>, Mat<f64>) {
    let mut s = Stream::new(seed);
    let mut x = Mat::<f64>::zeros(t, k);
    for j in 0..k {
        let mut acc = 0.0;
        for i in 0..t {
            acc += draw_normal(&mut s);
            x[(i, j)] = acc;
        }
    }
    let mut y = vec![0.0; t];
    for i in 1..t {
        let eq: f64 = (0..k).map(|j| (0.6 - 0.4 * j as f64) * x[(i - 1, j)]).sum();
        y[i] = y[i - 1] - 0.35 * (y[i - 1] - 1.5 - eq)
            + 0.3 * (x[(i, 0)] - x[(i - 1, 0)])
            + 0.002 * i as f64
            + draw_normal(&mut s);
    }
    (y, x)
}

fn fixed(p: usize, q: Vec<usize>, case: usize) -> ArdlOptions {
    ArdlOptions {
        lags: Some(p),
        order: Some(q),
        case: PssCase::from_index(case).expect("case"),
        ..ArdlOptions::default()
    }
}

/// SSR of the OLS of `y` on `z` (an independent route: plain faer QR).
fn ssr(z: &Mat<f64>, y: &[f64]) -> f64 {
    let ym = Mat::from_fn(y.len(), 1, |i, _| y[i]);
    let b = z.qr().solve_lstsq(&ym);
    let fit = z * &b;
    (0..y.len()).map(|i| (y[i] - fit[(i, 0)]).powi(2)).sum()
}

/// The UECM design built independently in the test, with the trend
/// shifted by `origin` (`trend = t + 1 + origin`).
fn uecm_design(
    y: &[f64],
    x: &Mat<f64>,
    n_det: usize,
    p: usize,
    q: &[usize],
    origin: f64,
) -> (Mat<f64>, Vec<f64>) {
    let t_all = y.len();
    let k = x.ncols();
    let h = q.iter().copied().fold(p, usize::max);
    let mut rows: Vec<Vec<f64>> = Vec::new();
    for t in h..t_all {
        let mut r = Vec::new();
        if n_det >= 1 {
            r.push(1.0);
        }
        if n_det >= 2 {
            r.push((t + 1) as f64 + origin);
        }
        r.push(y[t - 1]);
        for j in 0..k {
            r.push(x[(t - 1, j)]);
        }
        for i in 1..p {
            r.push(y[t - i] - y[t - i - 1]);
        }
        for (j, &qj) in q.iter().enumerate() {
            for l in 0..qj {
                r.push(x[(t - l, j)] - x[(t - l - 1, j)]);
            }
        }
        rows.push(r);
    }
    let m = rows[0].len();
    let z = Mat::from_fn(rows.len(), m, |i, j| rows[i][j]);
    let dy: Vec<f64> = (h..t_all).map(|t| y[t] - y[t - 1]).collect();
    (z, dy)
}

fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs().max(1e-300)
}

#[test]
fn uecm_is_a_reparameterisation_of_the_levels_ardl() {
    let (y, x) = coint_system(11, 180, 2);
    for case in 1..=5 {
        let q = vec![2, 3];
        let r = ardl_bounds_test(&y, x.as_ref(), &fixed(3, q.clone(), case)).expect("fit");
        // levels ARDL: y_t on [det, y_{t-1..t-3}, x_{j,t..t-q_j}]
        let n_det = r.n_det;
        let h = 3;
        let rows: Vec<Vec<f64>> = (h..y.len())
            .map(|t| {
                let mut v = Vec::new();
                if n_det >= 1 {
                    v.push(1.0);
                }
                if n_det >= 2 {
                    v.push((t + 1) as f64);
                }
                for i in 1..=3 {
                    v.push(y[t - i]);
                }
                for (j, &qj) in q.iter().enumerate() {
                    for l in 0..=qj {
                        v.push(x[(t - l, j)]);
                    }
                }
                v
            })
            .collect();
        let z = Mat::from_fn(rows.len(), rows[0].len(), |i, j| rows[i][j]);
        let yl: Vec<f64> = (h..y.len()).map(|t| y[t]).collect();
        assert_eq!(z.ncols(), r.n_params, "same column count");
        let s = ssr(&z, &yl);
        assert!(
            rel(r.ssr, s) < 1e-9,
            "case {case}: UECM ssr {} vs levels {s}",
            r.ssr
        );
    }
}

#[test]
fn wald_f_equals_the_restricted_vs_unrestricted_ssr_f() {
    let (y, x) = coint_system(12, 160, 3);
    for case in 1..=5 {
        let q = vec![1, 2, 1];
        let r = ardl_bounds_test(&y, x.as_ref(), &fixed(2, q.clone(), case)).expect("fit");
        let (z, dy) = uecm_design(&y, &x, r.n_det, 2, &q, 0.0);
        let mut rest: Vec<usize> = (r.n_det..r.n_det + 1 + 3).collect();
        match case {
            2 => rest.push(0),
            4 => rest.push(1),
            _ => {}
        }
        let keep: Vec<usize> = (0..z.ncols()).filter(|c| !rest.contains(c)).collect();
        let zr = Mat::from_fn(z.nrows(), keep.len(), |i, j| z[(i, keep[j])]);
        let su = ssr(&z, &dy);
        let sr = if keep.is_empty() {
            dy.iter().map(|v| v * v).sum()
        } else {
            ssr(&zr, &dy)
        };
        let f = ((sr - su) / rest.len() as f64) / (su / r.df_resid as f64);
        assert_eq!(r.n_restrictions, rest.len());
        assert!(
            rel(r.statistic, f) < 1e-9,
            "case {case}: Wald {} vs SSR {f}",
            r.statistic
        );
    }
}

#[test]
fn statistics_are_invariant_to_rescaling_the_data() {
    let (y, x) = coint_system(13, 150, 2);
    for case in [1, 3, 5] {
        let base = ardl_bounds_test(&y, x.as_ref(), &fixed(2, vec![2, 1], case)).expect("fit");
        let ys: Vec<f64> = y.iter().map(|v| 37.3 * v).collect();
        let xs = Mat::from_fn(x.nrows(), 2, |i, j| x[(i, j)] * [0.013, -250.0][j]);
        let r = ardl_bounds_test(&ys, xs.as_ref(), &fixed(2, vec![2, 1], case)).expect("fit");
        assert!(rel(r.statistic, base.statistic) < 1e-9, "case {case} F");
        assert!(rel(r.t_statistic, base.t_statistic) < 1e-9, "case {case} t");
        assert!(
            rel(r.p_value_upper, base.p_value_upper) < 1e-8,
            "case {case} p"
        );
        assert!(rel(r.adjustment, base.adjustment) < 1e-9, "case {case} pi");
        // the long-run slopes scale by c_y / c_x
        let k0 = base.n_det;
        assert!(rel(r.long_run[k0], base.long_run[k0] * 37.3 / 0.013) < 1e-8);
        assert!(rel(r.long_run[k0 + 1], base.long_run[k0 + 1] * 37.3 / -250.0) < 1e-8);
    }
}

#[test]
fn every_case_is_invariant_to_the_trend_origin_and_the_trend_runs_t_plus_one() {
    let (y, x) = coint_system(14, 140, 2);
    let q = vec![1, 2];
    for case in 1..=5 {
        let r = ardl_bounds_test(&y, x.as_ref(), &fixed(2, q.clone(), case)).expect("fit");
        for origin in [0.0, -60.5, 1000.0] {
            let (z, dy) = uecm_design(&y, &x, r.n_det, 2, &q, origin);
            let ym = Mat::from_fn(dy.len(), 1, |i, _| dy[i]);
            let b = z.qr().solve_lstsq(&ym);
            // same SSR, same level block, so the same F in every case
            let su = ssr(&z, &dy);
            assert!(rel(su, r.ssr) < 1e-9, "case {case} origin {origin}");
            for (j, c) in (r.n_det..r.n_det + 3).enumerate() {
                assert!(
                    rel(b[(c, 0)], r.params[c]) < 1e-7,
                    "case {case} level coef {j}"
                );
            }
            if r.n_det == 2 {
                // the trend coefficient is unchanged; the constant moves by
                // -origin * trend, the long-run intercept by origin * trend / pi
                assert!(rel(b[(1, 0)], r.params[1]) < 1e-7);
                let moved = r.params[0] - origin * r.params[1];
                assert!(
                    (b[(0, 0)] - moved).abs() < 1e-7 * moved.abs().max(1.0),
                    "case {case}"
                );
                let lr0 = -b[(0, 0)] / b[(2, 0)];
                let expect = r.long_run[0] + origin * r.params[1] / r.adjustment;
                assert!((lr0 - expect).abs() < 1e-6 * expect.abs().max(1.0));
            } else if origin == 0.0 && r.n_det == 1 {
                assert!(rel(b[(0, 0)], r.params[0]) < 1e-7);
            }
        }
    }
}

#[test]
fn surface_p_at_the_tabulated_critical_values_is_near_nominal() {
    let levels = [0.10, 0.05, 0.01, 0.001];
    let bound = [0.0013, 0.0011, 0.0003, 0.0001];
    let mut worst = [0.0_f64; 4];
    for e in pss::PSS_TABLE.iter() {
        for (i, &c) in e.crit.iter().enumerate() {
            let p = pss::pvalue(c, e.k as usize, e.case as usize, e.i1).expect("key");
            worst[i] = worst[i].max((p - levels[i]).abs());
        }
    }
    println!("max |surface p - nominal| at the tabulated crit (90/95/99/99.9%): {worst:?}");
    for i in 0..4 {
        assert!(worst[i] < bound[i], "level {i}: {}", worst[i]);
    }
    // the verdict therefore follows the p-values: a statistic just above
    // the tabulated 95% upper bound can still read p_upper slightly > 0.05
    // (or below it) — documented, not smoothed.
}

#[test]
fn surface_jump_at_stat_star_is_bounded_and_zero_f_gives_p_one() {
    let mut jump = 0.0_f64;
    for e in pss::PSS_TABLE.iter() {
        let (k, c) = (e.k as usize, e.case as usize);
        let lo = pss::pvalue(e.stat_star * (1.0 - 1e-12), k, c, e.i1).expect("key");
        let at = pss::pvalue(e.stat_star, k, c, e.i1).expect("key");
        let hi = pss::pvalue(e.stat_star * (1.0 + 1e-12), k, c, e.i1).expect("key");
        // `stat_star` itself is on the cubic branch (the quadratic takes
        // over strictly above it)
        assert!(
            (at - lo).abs() < 1e-9,
            "stat_star belongs to the cubic branch"
        );
        jump = jump.max((hi - lo).abs());
        assert_eq!(pss::pvalue(0.0, k, c, e.i1), Some(1.0));
        assert!(pss::pvalue(f64::NAN, k, c, e.i1).expect("key").is_nan());
        let tiny = pss::pvalue(1e6, k, c, e.i1).expect("key");
        assert!((0.0..1e-6).contains(&tiny));
    }
    println!("largest p jump at stat_star: {jump}");
    assert!(jump <= 0.017, "jump {jump}");
    assert!(
        jump > 0.01,
        "the documented 0.0165 jump disappeared: {jump}"
    );
    assert!(pss::pvalue(1.0, 11, 3, true).is_none());
}

/// One null draw: `(y, x)` independent; `x` a random walk (`i1`) or
/// white noise.
fn null_draw(stream: &mut Stream, t: usize, i1: bool) -> (Vec<f64>, Mat<f64>) {
    let mut y = vec![0.0; t];
    let mut x = Mat::<f64>::zeros(t, 1);
    let mut acc_y = 0.0;
    let mut acc_x = 0.0;
    for i in 0..t {
        acc_y += draw_normal(stream);
        y[i] = acc_y;
        let e = draw_normal(stream);
        acc_x += e;
        x[(i, 0)] = if i1 { acc_x } else { e };
    }
    (y, x)
}

#[test]
fn bounds_test_size_under_the_null_is_near_nominal() {
    let reps = 2000;
    let t = 200;
    let opts = fixed(1, vec![1], 3);
    let mut streams = Stream::substreams(20261008, reps).expect("substreams");
    let mut rej_upper = 0usize;
    let mut rej_lower = 0usize;
    for (m, s) in streams.iter_mut().enumerate() {
        // the first half: I(1) regressor (the upper bound's null); the
        // second: an I(0) regressor (the lower bound's)
        let i1 = m < reps / 2;
        let (y, x) = null_draw(s, t, i1);
        let r = ardl_bounds_test(&y, x.as_ref(), &opts).expect("fit");
        if i1 && r.p_value_upper < 0.05 {
            rej_upper += 1;
        }
        if !i1 && r.p_value_lower < 0.05 {
            rej_lower += 1;
        }
    }
    let half = (reps / 2) as f64;
    let (su, sl) = (rej_upper as f64 / half, rej_lower as f64 / half);
    println!("null size @5%, T = {t}, k_x = 1, case 3, ARDL(1,1), {} draws each: upper bound (I(1) x) {su:.3}, lower bound (I(0) x) {sl:.3}", reps / 2);
    // binomial MC se at 1000 draws: 0.0069
    assert!((0.035..=0.075).contains(&su), "upper-bound size {su}");
    assert!((0.035..=0.075).contains(&sl), "lower-bound size {sl}");
}

#[test]
fn power_against_a_level_relationship() {
    let mut streams = Stream::substreams(20261013, 200).expect("substreams");
    let mut rej = 0usize;
    for s in streams.iter_mut() {
        let t = 200;
        let mut x = vec![0.0; t];
        let mut y = vec![0.0; t];
        for i in 1..t {
            x[i] = x[i - 1] + draw_normal(s);
            y[i] = y[i - 1] - 0.1 * (y[i - 1] - x[i - 1]) + draw_normal(s);
        }
        let xm = Mat::from_fn(t, 1, |i, _| x[i]);
        let r = ardl_bounds_test(&y, xm.as_ref(), &fixed(1, vec![1], 3)).expect("fit");
        if r.verdict == ArdlVerdict::LevelRelationship {
            rej += 1;
        }
    }
    let power = rej as f64 / 200.0;
    println!("power at pi = -0.1, T = 200, case 3, ARDL(1,1), 200 draws: {power:.3}");
    assert!(power > 0.85, "power {power}");
}

#[test]
fn deterministic_bit_for_bit() {
    let (y, x) = coint_system(15, 120, 2);
    let a = ardl_bounds_test(&y, x.as_ref(), &ArdlOptions::default()).expect("fit");
    let b = ardl_bounds_test(&y, x.as_ref(), &ArdlOptions::default()).expect("fit");
    assert_eq!(a, b);
    assert!(a.selected && a.n_models == 64);
}

fn err(y: &[f64], x: &Mat<f64>, opts: &ArdlOptions) -> String {
    match ardl_bounds_test(y, x.as_ref(), opts) {
        Ok(r) => panic!(
            "expected a refusal, got F = {} for {opts:?} T = {}",
            r.statistic,
            y.len()
        ),
        Err(e) => e.to_string(),
    }
}

#[test]
fn refusals_name_the_parameter() {
    let (y, x) = coint_system(16, 120, 2);
    let base = ArdlOptions::default;
    let m = err(
        &y,
        &x,
        &ArdlOptions {
            lags: Some(0),
            ..base()
        },
    );
    assert!(
        m.contains("lags = 0") && m.contains("silently accepts"),
        "{m}"
    );
    let m = err(
        &y,
        &x,
        &ArdlOptions {
            order: Some(vec![1]),
            ..base()
        },
    );
    assert!(m.contains("order has 1 entry but x has 2"), "{m}");
    let m = err(
        &y,
        &x,
        &ArdlOptions {
            order: Some(vec![1, 0]),
            ..base()
        },
    );
    assert!(m.contains("order[1] = 0") && m.contains("follow-up"), "{m}");
    let m = err(
        &y,
        &x,
        &ArdlOptions {
            lags: Some(2),
            max_lag: Some(3),
            ..base()
        },
    );
    assert!(
        m.contains("max_lag = 3") && m.contains("lags = 2 is fixed"),
        "{m}"
    );
    let m = err(
        &y,
        &x,
        &ArdlOptions {
            order: Some(vec![1, 1]),
            max_order: Some(3),
            ..base()
        },
    );
    assert!(
        m.contains("max_order = 3") && m.contains("order = [1, 1] is fixed"),
        "{m}"
    );
    let m = err(
        &y,
        &x,
        &ArdlOptions {
            max_lag: Some(0),
            ..base()
        },
    );
    assert!(m.contains("max_lag = 0"), "{m}");
    let m = err(
        &y,
        &x,
        &ArdlOptions {
            max_order: Some(0),
            ..base()
        },
    );
    assert!(m.contains("max_order = 0"), "{m}");
    for a in [0.0, 1.0, -0.1, f64::NAN, f64::INFINITY] {
        let m = err(&y, &x, &ArdlOptions { alpha: a, ..base() });
        assert!(m.contains("alpha = ") && m.contains("(0, 1)"), "{m}");
    }
    // the grid guard: 4 x 4^8 = 262 144 models at k_x = 8 with the defaults
    let (y8, x8) = coint_system(17, 400, 8);
    let m = err(&y8, &x8, &base());
    assert!(
        m.contains("262144 candidate models") && m.contains("max_order = 4 (the default)"),
        "{m}"
    );
    assert!(m.contains("Fix order"), "{m}");
    // fixing the order lifts it (the lags-only search fits 4 models)
    let r = ardl_bounds_test(
        &y8,
        x8.as_ref(),
        &ArdlOptions {
            order: Some(vec![1; 8]),
            ..base()
        },
    )
    .expect("lags-only search");
    assert_eq!(r.n_models, 4);
    // absurd caps saturate instead of overflowing
    let m = err(
        &y,
        &x,
        &ArdlOptions {
            max_order: Some(1 << 40),
            ..base()
        },
    );
    assert!(
        m.contains("more than 1.8e19") || m.contains("candidate models"),
        "{m}"
    );
    let m = err(
        &y,
        &x,
        &ArdlOptions {
            lags: Some(1 << 40),
            order: Some(vec![1, 1]),
            ..base()
        },
    );
    assert!(
        m.contains("too large for T = 120") && m.contains("lags = 1099511627776"),
        "{m}"
    );
    // a design past the memory budget is refused before it is built
    // (T = 2^20 rows, a fixed order of 100 on one regressor: 2^20 x 103)
    let big_t = 1 << 20;
    let yb: Vec<f64> = (0..big_t).map(|i| (i as f64 * 0.37).sin()).collect();
    let xb = Mat::from_fn(big_t, 1, |i, _| (i as f64 * 0.11).cos());
    let m = err(&yb, &xb, &fixed(1, vec![100], 3));
    assert!(m.contains("512 MiB") && m.contains("max_order"), "{m}");
    // too few rows for the largest candidate names the caps
    // (T = 20 still fits: 16 rows for 15 regressors at the default caps)
    let m = err(&y[..15], &Mat::from_fn(15, 2, |i, j| x[(i, j)]), &base());
    assert!(
        m.contains("too large for T = 15") && m.contains("max_lag = 4 (the default)"),
        "{m}"
    );
    assert!(
        m.contains("11 row(s) for") && m.contains("= 15 regressors"),
        "{m}"
    );
    // inputs
    let m = err(&y[..119], &x, &base());
    assert!(m.contains("expected 119, got 120"), "{m}");
    let mut yn = y.clone();
    yn[7] = f64::NAN;
    let m = err(&yn, &x, &base());
    assert!(m.contains("index 7 of y"), "{m}");
    let mut xn = x.clone();
    xn[(3, 1)] = f64::INFINITY;
    let m = err(&y, &xn, &base());
    assert!(m.contains("x at row 3, column 1"), "{m}");
    let m = err(&y, &Mat::<f64>::zeros(120, 0), &base());
    assert!(m.contains("at least one regressor"), "{m}");
    let m = err(&[], &Mat::<f64>::zeros(0, 1), &base());
    assert!(m.contains("too large for T = 0"), "{m}");
    let m = err(&[1.0], &Mat::<f64>::zeros(1, 1), &fixed(1, vec![1], 3));
    assert!(m.contains("too large for T = 1"), "{m}");
    // a duplicated regressor and a constant regressor are collinear
    let dup = Mat::from_fn(120, 2, |i, _| x[(i, 0)]);
    assert!(matches!(
        ardl_bounds_test(&y, dup.as_ref(), &fixed(1, vec![1, 1], 3)),
        Err(CointError::Singular { .. })
    ));
    let cst = Mat::from_fn(120, 1, |_, _| 2.5);
    let m = err(&y, &cst, &fixed(1, vec![1], 3));
    assert!(
        m.contains("numerically singular") && m.contains("constant column of x"),
        "{m}"
    );
    let m = err(&y, &cst, &base());
    assert!(m.contains("numerically singular"), "{m}");
}

#[test]
fn eleven_regressors_have_no_bounds_and_say_so() {
    let (y, x) = coint_system(18, 150, 11);
    let r: ArdlResult = ardl_bounds_test(&y, x.as_ref(), &fixed(1, vec![1; 11], 3)).expect("fit");
    assert!(r.statistic.is_finite() && r.crit_upper.iter().all(|v| v.is_nan()));
    assert!(r.p_value_lower.is_nan() && r.p_value_upper.is_nan());
    assert_eq!(r.verdict, ArdlVerdict::Unavailable);
    assert!(r
        .crit_note
        .as_deref()
        .is_some_and(|n| n.contains("stops at 10")));
    let x10 = Mat::from_fn(150, 10, |i, j| x[(i, j)]);
    let r10 = ardl_bounds_test(&y, x10.as_ref(), &fixed(1, vec![1; 10], 3));
    assert!(r10.expect("fit").crit_note.is_none());
}

#[test]
fn ic_value_is_on_the_statsmodels_scale_not_the_dols_scale() {
    let (y, x) = coint_system(19, 150, 1);
    for ic in [DolsIc::Aic, DolsIc::Bic, DolsIc::Hqic] {
        let r = ardl_bounds_test(
            &y,
            x.as_ref(),
            &ArdlOptions {
                ic,
                ..ArdlOptions::default()
            },
        )
        .expect("fit");
        // -2 llf + c (m + 1) is O(nobs); ln(RSS/n) + m c / n would be O(1)
        assert!(r.ic_value.abs() > 50.0, "{ic:?}: {}", r.ic_value);
        assert_eq!(r.max_lag, 4);
        assert_eq!(r.max_order, 4);
        assert_eq!(r.n_models, 16);
    }
}
