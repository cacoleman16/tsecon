//! BFGS / L-BFGS accuracy tests on the analytic test functions of the
//! spec: Rosenbrock (2d, 10d), the sphere, and an ill-conditioned
//! quadratic with condition number 1e6.

mod common;

use common::{rosenbrock, rosenbrock_grad};
use tsecon_optim::{
    bfgs, lbfgs, BfgsOptions, FnObjective, FnObjectiveGrad, LbfgsOptions, ObjectiveFn,
    OptimError, Termination,
};

fn rosen_obj() -> impl ObjectiveFn {
    FnObjectiveGrad::new(rosenbrock, |x: &[f64]| rosenbrock_grad(x))
}

/// BFGS reaches the 2d Rosenbrock minimum (1, 1) to 1e-8 from the standard
/// start (-1.2, 1) (More-Garbow-Hillstrom 1981, problem 1).
#[test]
fn bfgs_rosenbrock_2d() {
    let opts = BfgsOptions {
        grad_tol: 1e-10,
        ..BfgsOptions::default()
    };
    let res = bfgs(&mut rosen_obj(), &[-1.2, 1.0], &opts).unwrap();
    assert!(res.converged, "termination: {}", res.termination);
    assert_eq!(res.termination, Termination::GradientTolerance);
    for &xi in &res.x {
        assert!((xi - 1.0).abs() <= 1e-8, "x = {:?}", res.x);
    }
    assert!(res.f <= 1e-16);
    assert!(res.gevals > 0, "analytic gradient must be used");
}

/// L-BFGS (m = 10) matches on 2d Rosenbrock.
#[test]
fn lbfgs_rosenbrock_2d() {
    let opts = LbfgsOptions {
        grad_tol: 1e-10,
        ..LbfgsOptions::default()
    };
    let res = lbfgs(&mut rosen_obj(), &[-1.2, 1.0], &opts).unwrap();
    assert!(res.converged, "termination: {}", res.termination);
    for &xi in &res.x {
        assert!((xi - 1.0).abs() <= 1e-8, "x = {:?}", res.x);
    }
}

/// The standard 10d start: alternating (-1.2, 1) (More-Garbow-Hillstrom
/// 1981 extended Rosenbrock convention).
fn start_10d() -> Vec<f64> {
    (0..10)
        .map(|i| if i % 2 == 0 { -1.2 } else { 1.0 })
        .collect()
}

/// BFGS on the chained 10d Rosenbrock to 1e-8.
#[test]
fn bfgs_rosenbrock_10d() {
    let opts = BfgsOptions {
        grad_tol: 1e-10,
        ..BfgsOptions::default()
    };
    let res = bfgs(&mut rosen_obj(), &start_10d(), &opts).unwrap();
    assert!(res.converged, "termination: {}", res.termination);
    for &xi in &res.x {
        assert!((xi - 1.0).abs() <= 1e-8, "x = {:?}", res.x);
    }
}

/// L-BFGS on the chained 10d Rosenbrock to 1e-8.
#[test]
fn lbfgs_rosenbrock_10d() {
    let opts = LbfgsOptions {
        grad_tol: 1e-10,
        ..LbfgsOptions::default()
    };
    let res = lbfgs(&mut rosen_obj(), &start_10d(), &opts).unwrap();
    assert!(res.converged, "termination: {}", res.termination);
    for &xi in &res.x {
        assert!((xi - 1.0).abs() <= 1e-8, "x = {:?}", res.x);
    }
}

fn sphere_obj() -> impl ObjectiveFn {
    FnObjectiveGrad::new(
        |x: &[f64]| x.iter().map(|v| v * v).sum(),
        |x: &[f64]| x.iter().map(|v| 2.0 * v).collect(),
    )
}

/// The sphere is solved essentially exactly and quickly by both methods.
#[test]
fn sphere_10d() {
    let x0 = vec![3.0; 10];
    let opts = BfgsOptions {
        grad_tol: 1e-10,
        ..BfgsOptions::default()
    };
    let res = bfgs(&mut sphere_obj(), &x0, &opts).unwrap();
    assert!(res.converged);
    assert!(res.iterations <= 10, "iterations = {}", res.iterations);
    for &xi in &res.x {
        assert!(xi.abs() <= 1e-10);
    }

    let opts = LbfgsOptions {
        grad_tol: 1e-10,
        ..LbfgsOptions::default()
    };
    let res = lbfgs(&mut sphere_obj(), &x0, &opts).unwrap();
    assert!(res.converged);
    for &xi in &res.x {
        assert!(xi.abs() <= 1e-10);
    }
}

/// Diagonal quadratic with eigenvalues 10^0 .. 10^6 (condition number
/// 1e6): `f = 0.5 sum lambda_i x_i^2`.
fn ill_conditioned() -> (impl ObjectiveFn, Vec<f64>) {
    let lambdas: Vec<f64> = (0..7).map(|i| 10f64.powi(i)).collect();
    let l2 = lambdas.clone();
    let obj = FnObjectiveGrad::new(
        move |x: &[f64]| {
            0.5 * x
                .iter()
                .zip(&lambdas)
                .map(|(&xi, &li)| li * xi * xi)
                .sum::<f64>()
        },
        move |x: &[f64]| x.iter().zip(&l2).map(|(&xi, &li)| li * xi).collect(),
    );
    (obj, vec![1.0; 7])
}

/// BFGS solves the cond-1e6 quadratic to ||x||_inf <= 1e-8.
#[test]
fn bfgs_ill_conditioned_quadratic() {
    let (mut obj, x0) = ill_conditioned();
    let opts = BfgsOptions {
        grad_tol: 1e-9,
        ..BfgsOptions::default()
    };
    let res = bfgs(&mut obj, &x0, &opts).unwrap();
    assert!(res.converged, "termination: {}", res.termination);
    for &xi in &res.x {
        assert!(xi.abs() <= 1e-8, "x = {:?}", res.x);
    }
    assert!(res.f <= 1e-14);
}

/// L-BFGS solves the cond-1e6 quadratic to ||x||_inf <= 1e-8.
#[test]
fn lbfgs_ill_conditioned_quadratic() {
    let (mut obj, x0) = ill_conditioned();
    let opts = LbfgsOptions {
        grad_tol: 1e-9,
        ..LbfgsOptions::default()
    };
    let res = lbfgs(&mut obj, &x0, &opts).unwrap();
    assert!(res.converged, "termination: {}", res.termination);
    for &xi in &res.x {
        assert!(xi.abs() <= 1e-8, "x = {:?}", res.x);
    }
}

/// With no analytic gradient, BFGS runs on the central-difference helper:
/// still converges (to the numerical-gradient floor), probes counted as
/// fevals, no analytic-gradient evaluations.
#[test]
fn bfgs_numerical_gradient_rosenbrock() {
    let mut obj = FnObjective::new(rosenbrock);
    let opts = BfgsOptions {
        grad_tol: 1e-6,
        ..BfgsOptions::default()
    };
    let res = bfgs(&mut obj, &[-1.2, 1.0], &opts).unwrap();
    assert!(res.converged, "termination: {}", res.termination);
    for &xi in &res.x {
        assert!((xi - 1.0).abs() <= 1e-5, "x = {:?}", res.x);
    }
    assert_eq!(res.gevals, 0);
    // Each gradient costs 2n = 4 probes, so fevals must dominate iterations.
    assert!(res.fevals > 4 * res.iterations);
}

/// Budget controls report honest non-convergence.
#[test]
fn bfgs_max_iter_termination() {
    let opts = BfgsOptions {
        grad_tol: 1e-12,
        max_iter: Some(2),
        ..BfgsOptions::default()
    };
    let res = bfgs(&mut rosen_obj(), &[-1.2, 1.0], &opts).unwrap();
    assert!(!res.converged);
    assert_eq!(res.termination, Termination::MaxIterations);
    assert_eq!(res.iterations, 2);
    assert!(res.f < rosenbrock(&[-1.2, 1.0]), "still made progress");
}

/// Feval budget termination.
#[test]
fn lbfgs_max_fevals_termination() {
    let opts = LbfgsOptions {
        grad_tol: 1e-12,
        max_fevals: Some(3),
        ..LbfgsOptions::default()
    };
    let res = lbfgs(&mut rosen_obj(), &[-1.2, 1.0], &opts).unwrap();
    assert!(!res.converged);
    assert_eq!(res.termination, Termination::MaxFevals);
}

/// Input validation errors.
#[test]
fn quasi_newton_input_errors() {
    let opts = BfgsOptions::default();
    assert!(matches!(
        bfgs(&mut rosen_obj(), &[], &opts),
        Err(OptimError::EmptyInput { .. })
    ));
    assert!(matches!(
        bfgs(&mut rosen_obj(), &[f64::NAN, 1.0], &opts),
        Err(OptimError::NonFinite { .. })
    ));
    let mut bad = FnObjective::new(|_x: &[f64]| f64::NAN);
    assert!(matches!(
        bfgs(&mut bad, &[1.0], &opts),
        Err(OptimError::NonFinite { .. })
    ));
    let bad_opts = BfgsOptions {
        grad_tol: -1.0,
        ..BfgsOptions::default()
    };
    assert!(matches!(
        bfgs(&mut rosen_obj(), &[1.0], &bad_opts),
        Err(OptimError::InvalidOption { .. })
    ));
    let bad_mem = LbfgsOptions {
        memory: 0,
        ..LbfgsOptions::default()
    };
    assert!(matches!(
        lbfgs(&mut rosen_obj(), &[1.0], &bad_mem),
        Err(OptimError::InvalidOption { .. })
    ));
}

/// An analytic gradient of the wrong length is a caller bug and errors.
#[test]
fn wrong_length_gradient_errors() {
    let mut obj = FnObjectiveGrad::new(
        |x: &[f64]| x.iter().map(|v| v * v).sum(),
        |_x: &[f64]| vec![0.0; 3],
    );
    assert!(matches!(
        bfgs(&mut obj, &[1.0, 2.0], &BfgsOptions::default()),
        Err(OptimError::DimensionMismatch { .. })
    ));
}
