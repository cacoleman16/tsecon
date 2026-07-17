//! Accuracy of the central-difference numerical gradient against an
//! analytic gradient on a polynomial (documented error order eps^(2/3)).

mod common;

use common::Lcg;
use tsecon_optim::{central_difference_gradient, eval_gradient, FnObjective, FnObjectiveGrad};

/// Test polynomial with interactions:
/// `f = 2 x0^3 - 3 x0^2 x1 + 4 x1^2 + x2^4 - 5 x2 + x0 x1 x2`.
fn poly(x: &[f64]) -> f64 {
    2.0 * x[0].powi(3) - 3.0 * x[0] * x[0] * x[1] + 4.0 * x[1] * x[1] + x[2].powi(4) - 5.0 * x[2]
        + x[0] * x[1] * x[2]
}

fn poly_grad(x: &[f64]) -> Vec<f64> {
    vec![
        6.0 * x[0] * x[0] - 6.0 * x[0] * x[1] + x[1] * x[2],
        -3.0 * x[0] * x[0] + 8.0 * x[1] + x[0] * x[2],
        4.0 * x[2].powi(3) - 5.0 + x[0] * x[1],
    ]
}

/// Central differences match the analytic gradient to ~1e-8 absolute at
/// order-one points — consistent with the documented eps^(2/3) error.
#[test]
fn central_difference_matches_analytic_on_polynomial() {
    let mut rng = Lcg::new(123);
    let mut obj = FnObjective::new(poly);
    for _ in 0..50 {
        let x: Vec<f64> = (0..3).map(|_| rng.uniform(-2.0, 2.0)).collect();
        let num = central_difference_gradient(&mut obj, &x);
        let exact = poly_grad(&x);
        for (i, (&ni, &ei)) in num.iter().zip(&exact).enumerate() {
            assert!(
                (ni - ei).abs() <= 1e-8,
                "component {i} at {x:?}: numeric {ni}, exact {ei}"
            );
        }
    }
}

/// The step scales with |x| so relative accuracy survives at large
/// arguments.
#[test]
fn central_difference_scales_with_x() {
    let mut obj = FnObjective::new(poly);
    let x = [150.0, -80.0, 40.0];
    let num = central_difference_gradient(&mut obj, &x);
    let exact = poly_grad(&x);
    for (&ni, &ei) in num.iter().zip(&exact) {
        assert!(
            (ni - ei).abs() <= 1e-7 * ei.abs().max(1.0),
            "numeric {ni}, exact {ei}"
        );
    }
}

/// `eval_gradient` returns the analytic gradient exactly when one is
/// supplied, and the central difference otherwise.
#[test]
fn eval_gradient_prefers_analytic() {
    let x = [0.3, -1.1, 0.7];
    let mut with_grad = FnObjectiveGrad::new(poly, |x: &[f64]| poly_grad(x));
    let g = eval_gradient(&mut with_grad, &x).unwrap();
    assert_eq!(g, poly_grad(&x), "must be bitwise the analytic gradient");

    let mut without = FnObjective::new(poly);
    let g = eval_gradient(&mut without, &x).unwrap();
    let exact = poly_grad(&x);
    for (&ni, &ei) in g.iter().zip(&exact) {
        assert!((ni - ei).abs() <= 1e-8);
        assert!(ni != ei, "numeric path should differ in the last bits");
    }
}

/// On a quadratic the central difference is exact up to rounding (the
/// truncation term vanishes: third derivative is zero).
#[test]
fn central_difference_exact_on_quadratic() {
    let mut obj = FnObjective::new(|x: &[f64]| 3.0 * x[0] * x[0] + 2.0 * x[0] + 1.0);
    let g = central_difference_gradient(&mut obj, &[0.5]);
    assert!((g[0] - 5.0).abs() <= 1e-9);
}
