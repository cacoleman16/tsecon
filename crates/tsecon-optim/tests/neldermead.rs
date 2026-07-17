//! Adaptive Nelder-Mead behavior: Rosenbrock 2d, adaptive coefficients in
//! higher dimension, restart support, termination semantics.

mod common;

use common::rosenbrock;
use tsecon_optim::{nelder_mead, FnObjective, NelderMeadOptions, OptimError, Termination};

/// Nelder-Mead finds the 2d Rosenbrock minimum from the standard start.
#[test]
fn nm_rosenbrock_2d() {
    let mut obj = FnObjective::new(rosenbrock);
    let opts = NelderMeadOptions {
        x_tol: 1e-10,
        f_tol: 1e-10,
        max_iter: Some(2000),
        max_fevals: Some(4000),
        ..NelderMeadOptions::default()
    };
    let res = nelder_mead(&mut obj, &[-1.2, 1.0], &opts).unwrap();
    assert!(res.converged, "termination: {}", res.termination);
    assert_eq!(res.termination, Termination::SimplexTolerance);
    for &xi in &res.x {
        assert!((xi - 1.0).abs() <= 1e-5, "x = {:?}", res.x);
    }
    assert!(res.f <= 1e-10);
    assert_eq!(res.gevals, 0);
}

/// The Gao-Han adaptive coefficients handle a 6d problem where the
/// standard simplex is prone to stagnation: both should solve the sphere,
/// and the adaptive run must converge.
#[test]
fn nm_adaptive_6d_sphere() {
    let x0 = vec![2.0; 6];
    let opts = NelderMeadOptions {
        x_tol: 1e-9,
        f_tol: 1e-9,
        max_iter: Some(5000),
        max_fevals: Some(10000),
        ..NelderMeadOptions::default()
    };
    let mut obj = FnObjective::new(|x: &[f64]| x.iter().map(|v| v * v).sum::<f64>());
    let res = nelder_mead(&mut obj, &x0, &opts).unwrap();
    assert!(res.converged, "termination: {}", res.termination);
    assert!(res.f <= 1e-12, "f = {:e}", res.f);
}

/// Restarts guard against premature simplex collapse: the restarted run is
/// never worse, and from a deliberately coarse first convergence it
/// improves.
#[test]
fn nm_restart_improves() {
    // Loose f_tol converges early on the Rosenbrock valley floor.
    let base = NelderMeadOptions {
        x_tol: 1e-6,
        f_tol: 1e-6,
        max_iter: Some(4000),
        max_fevals: Some(8000),
        ..NelderMeadOptions::default()
    };
    let mut obj = FnObjective::new(rosenbrock);
    let plain = nelder_mead(&mut obj, &[-1.2, 1.0], &base).unwrap();

    let restarted_opts = NelderMeadOptions {
        restarts: 3,
        ..base
    };
    let mut obj = FnObjective::new(rosenbrock);
    let restarted = nelder_mead(&mut obj, &[-1.2, 1.0], &restarted_opts).unwrap();

    assert!(restarted.f <= plain.f);
    assert!(restarted.fevals > plain.fevals, "restarts actually ran");
    assert!(restarted.converged);
}

/// Budget terminations are honest.
#[test]
fn nm_budget_terminations() {
    let mut obj = FnObjective::new(rosenbrock);
    let opts = NelderMeadOptions {
        x_tol: 0.0,
        f_tol: 0.0,
        max_iter: Some(5),
        ..NelderMeadOptions::default()
    };
    let res = nelder_mead(&mut obj, &[-1.2, 1.0], &opts).unwrap();
    assert!(!res.converged);
    assert_eq!(res.termination, Termination::MaxIterations);
    assert_eq!(res.iterations, 5);

    let mut obj = FnObjective::new(rosenbrock);
    let opts = NelderMeadOptions {
        x_tol: 0.0,
        f_tol: 0.0,
        max_iter: Some(10_000),
        max_fevals: Some(20),
        ..NelderMeadOptions::default()
    };
    let res = nelder_mead(&mut obj, &[-1.2, 1.0], &opts).unwrap();
    assert!(!res.converged);
    assert_eq!(res.termination, Termination::MaxFevals);
    // Overshoot is at most one shrink step (n + 2 evaluations).
    assert!(res.fevals <= 20 + 4);
}

/// Non-finite regions are treated as infeasible, not fatal: minimize
/// f(x) = x^2 with f = NaN for x < -0.5, starting right of the hole.
#[test]
fn nm_infeasible_region() {
    let mut obj = FnObjective::new(|x: &[f64]| if x[0] < -0.5 { f64::NAN } else { x[0] * x[0] });
    let res = nelder_mead(&mut obj, &[1.0], &NelderMeadOptions::default()).unwrap();
    assert!(res.converged);
    assert!(res.x[0].abs() <= 1e-6);
}

/// Input validation errors.
#[test]
fn nm_input_errors() {
    let opts = NelderMeadOptions::default();
    let mut obj = FnObjective::new(rosenbrock);
    assert!(matches!(
        nelder_mead(&mut obj, &[], &opts),
        Err(OptimError::EmptyInput { .. })
    ));
    let mut obj = FnObjective::new(rosenbrock);
    assert!(matches!(
        nelder_mead(&mut obj, &[f64::INFINITY, 0.0], &opts),
        Err(OptimError::NonFinite { .. })
    ));
    let mut all_nan = FnObjective::new(|_: &[f64]| f64::NAN);
    assert!(matches!(
        nelder_mead(&mut all_nan, &[1.0], &opts),
        Err(OptimError::NonFinite { .. })
    ));
    let bad = NelderMeadOptions {
        x_tol: -1.0,
        ..NelderMeadOptions::default()
    };
    let mut obj = FnObjective::new(rosenbrock);
    assert!(matches!(
        nelder_mead(&mut obj, &[1.0, 1.0], &bad),
        Err(OptimError::InvalidOption { .. })
    ));
}
