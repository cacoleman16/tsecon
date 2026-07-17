//! Verifies the strong-Wolfe conditions hold on every accepted step, on a
//! spread of points and descent directions.

mod common;

use common::{dot, rosenbrock, rosenbrock_grad, Lcg};
use tsecon_optim::{strong_wolfe, FnObjectiveGrad, ObjectiveFn, OptimError, WolfeOptions};

/// Checks both conditions at the accepted step for objective `obj` from
/// point `x` along direction `d`.
fn check_wolfe<F: ObjectiveFn>(mut obj: F, x: &[f64], d: &[f64], opts: &WolfeOptions) {
    let f0 = obj.value(x);
    let g0 = obj.gradient(x).unwrap();
    let dphi0 = dot(&g0, d);
    assert!(dphi0 < 0.0, "test setup must supply a descent direction");

    let res = strong_wolfe(&mut obj, x, d, f0, &g0, 1.0, opts).unwrap();
    assert!(res.success, "line search failed at x = {x:?}, d = {d:?}");
    assert!(res.step > 0.0);

    let slack = 1e-12 * (1.0 + f0.abs());
    // Sufficient decrease (Armijo).
    assert!(
        res.f <= f0 + opts.c1 * res.step * dphi0 + slack,
        "Armijo violated: f = {}, bound = {}",
        res.f,
        f0 + opts.c1 * res.step * dphi0
    );
    // Strong curvature.
    let dphi = dot(&res.g, d);
    assert!(
        dphi.abs() <= opts.c2 * dphi0.abs() * (1.0 + 1e-12),
        "curvature violated: |phi'(a)| = {}, bound = {}",
        dphi.abs(),
        opts.c2 * dphi0.abs()
    );
    // The returned point is consistent: x + step * d.
    for i in 0..x.len() {
        assert!((res.x[i] - (x[i] + res.step * d[i])).abs() <= 1e-14 * (1.0 + res.x[i].abs()));
    }
}

fn rosen_obj() -> impl ObjectiveFn {
    FnObjectiveGrad::new(rosenbrock, |x: &[f64]| rosenbrock_grad(x))
}

/// Steepest-descent directions from random points on Rosenbrock 2d and 5d.
#[test]
fn wolfe_conditions_on_rosenbrock() {
    let opts = WolfeOptions::default();
    let mut rng = Lcg::new(20260716);
    for n in [2usize, 5] {
        for _ in 0..25 {
            let x: Vec<f64> = (0..n).map(|_| rng.uniform(-2.0, 2.0)).collect();
            let g = rosenbrock_grad(&x);
            if g.iter().all(|&v| v.abs() < 1e-8) {
                continue; // stationary start, no descent direction
            }
            let d: Vec<f64> = g.iter().map(|&v| -v).collect();
            check_wolfe(rosen_obj(), &x, &d, &opts);
        }
    }
}

/// Random (still descending) non-gradient directions, and a tighter c2.
#[test]
fn wolfe_conditions_random_directions() {
    let tight = WolfeOptions {
        c2: 0.4,
        ..WolfeOptions::default()
    };
    let mut rng = Lcg::new(42);
    let mut checked = 0;
    while checked < 25 {
        let x: Vec<f64> = (0..3).map(|_| rng.uniform(-2.0, 2.0)).collect();
        let g = rosenbrock_grad(&x);
        let d: Vec<f64> = (0..3).map(|_| rng.uniform(-1.0, 1.0)).collect();
        if dot(&g, &d) >= -1e-8 {
            continue; // not a descent direction; redraw
        }
        check_wolfe(rosen_obj(), &x, &d, &tight);
        checked += 1;
    }
}

/// On the cond-1e6 quadratic the exact minimizing step along -g is tiny;
/// the search must still deliver strong-Wolfe points.
#[test]
fn wolfe_conditions_ill_conditioned_quadratic() {
    let lambdas: Vec<f64> = (0..7).map(|i| 10f64.powi(i)).collect();
    let quad = |lams: Vec<f64>| {
        let l2 = lams.clone();
        FnObjectiveGrad::new(
            move |x: &[f64]| {
                0.5 * x
                    .iter()
                    .zip(&lams)
                    .map(|(&xi, &li)| li * xi * xi)
                    .sum::<f64>()
            },
            move |x: &[f64]| x.iter().zip(&l2).map(|(&xi, &li)| li * xi).collect(),
        )
    };
    let opts = WolfeOptions::default();
    let mut rng = Lcg::new(7);
    for _ in 0..10 {
        let x: Vec<f64> = (0..7).map(|_| rng.uniform(0.5, 1.5)).collect();
        let g: Vec<f64> = x.iter().zip(&lambdas).map(|(&xi, &li)| li * xi).collect();
        let d: Vec<f64> = g.iter().map(|&v| -v).collect();
        check_wolfe(quad(lambdas.clone()), &x, &d, &opts);
    }
}

/// Misuse errors: non-descent direction, bad constants, bad initial step.
#[test]
fn wolfe_input_errors() {
    let mut obj = rosen_obj();
    let x = [-1.2, 1.0];
    let f0 = obj.value(&x);
    let g0 = obj.gradient(&x).unwrap();

    // Ascent direction.
    let d: Vec<f64> = g0.clone();
    assert!(matches!(
        strong_wolfe(&mut obj, &x, &d, f0, &g0, 1.0, &WolfeOptions::default()),
        Err(OptimError::Domain { .. })
    ));

    // c2 <= c1.
    let bad = WolfeOptions {
        c1: 0.5,
        c2: 0.4,
        ..WolfeOptions::default()
    };
    let d: Vec<f64> = g0.iter().map(|&v| -v).collect();
    assert!(matches!(
        strong_wolfe(&mut obj, &x, &d, f0, &g0, 1.0, &bad),
        Err(OptimError::InvalidOption { .. })
    ));

    // Nonpositive initial step.
    assert!(matches!(
        strong_wolfe(&mut obj, &x, &d, f0, &g0, 0.0, &WolfeOptions::default()),
        Err(OptimError::InvalidOption { .. })
    ));

    // Length mismatch.
    assert!(matches!(
        strong_wolfe(
            &mut obj,
            &x,
            &d[..1],
            f0,
            &g0,
            1.0,
            &WolfeOptions::default()
        ),
        Err(OptimError::DimensionMismatch { .. })
    ));
}

/// Unbounded-below direction: no strong-Wolfe point exists; the search
/// reports failure but still returns the best (improved) trial.
#[test]
fn wolfe_unbounded_reports_failure_with_progress() {
    let mut obj = FnObjectiveGrad::new(|x: &[f64]| -x[0], |_x: &[f64]| vec![-1.0]);
    let x = [0.0];
    let f0 = obj.value(&x);
    let g0 = obj.gradient(&x).unwrap();
    let d = [1.0];
    let res = strong_wolfe(&mut obj, &x, &d, f0, &g0, 1.0, &WolfeOptions::default()).unwrap();
    assert!(!res.success);
    assert!(res.f < f0, "best trial should still improve");
}
