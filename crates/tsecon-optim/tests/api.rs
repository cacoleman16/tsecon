//! Top-level API: `minimize` dispatch and the multistart helper.

mod common;

use common::{rosenbrock, rosenbrock_grad, Lcg};
use tsecon_optim::{
    minimize, multistart, FnObjective, FnObjectiveGrad, Method, NelderMeadOptions, OptimError,
};

/// Every method dispatches through `minimize` and solves the sphere.
#[test]
fn minimize_dispatch_all_methods() {
    for method in [Method::nelder_mead(), Method::bfgs(), Method::lbfgs()] {
        let mut obj = FnObjectiveGrad::new(
            |x: &[f64]| x.iter().map(|v| v * v).sum(),
            |x: &[f64]| x.iter().map(|v| 2.0 * v).collect(),
        );
        let res = minimize(&mut obj, &[1.5, -2.0, 0.5], &method).unwrap();
        assert!(res.converged, "{method:?}: {}", res.termination);
        assert!(res.f <= 1e-10, "{method:?}: f = {:e}", res.f);
    }
}

/// `minimize` also drives Rosenbrock through the enum-carried options.
#[test]
fn minimize_with_custom_options() {
    let mut obj = FnObjective::new(rosenbrock);
    let opts = NelderMeadOptions {
        x_tol: 1e-10,
        f_tol: 1e-10,
        max_iter: Some(2000),
        max_fevals: Some(4000),
        restarts: 1,
        ..NelderMeadOptions::default()
    };
    let res = minimize(&mut obj, &[-1.2, 1.0], &Method::NelderMead(opts)).unwrap();
    assert!(res.converged);
    assert!((res.x[0] - 1.0).abs() <= 1e-5);
}

/// Asymmetric double well `f(x) = (x^2 - 1)^2 + 0.2 x`: a local minimum
/// near +0.97 and the global one near -1.02. A single run from 0.9 lands
/// in the local basin; multistart with spread-out perturbations finds the
/// global one.
#[test]
fn multistart_escapes_local_minimum() {
    let dwell = |x: &[f64]| {
        let t = x[0] * x[0] - 1.0;
        t * t + 0.2 * x[0]
    };

    // Single start: trapped in the local basin.
    let mut obj = FnObjective::new(dwell);
    let single = minimize(&mut obj, &[0.9], &Method::nelder_mead()).unwrap();
    assert!(single.x[0] > 0.0, "sanity: single run stays local");

    // Multistart: deterministic perturbations sweep both basins.
    let mut rng = Lcg::new(2026);
    let mut obj = FnObjective::new(dwell);
    let ms = multistart(&mut obj, &[0.9], &Method::nelder_mead(), 6, |_, x| {
        x[0] = rng.uniform(-2.0, 2.0);
    })
    .unwrap();
    assert!(ms.best.x[0] < -1.0, "best = {:?}", ms.best.x);
    assert!(ms.best.f < single.f);
    assert!(ms.best_start > 0, "a perturbed start must have won");
    assert_eq!(ms.n_converged, 6);
    assert!(ms.total_fevals >= ms.best.fevals);
    assert!(ms.total_iterations >= ms.best.iterations);
}

/// Multistart with BFGS and an analytic gradient on Rosenbrock: start 0 is
/// the unperturbed x0 and already succeeds.
#[test]
fn multistart_unperturbed_first_start() {
    let mut obj = FnObjectiveGrad::new(rosenbrock, |x: &[f64]| rosenbrock_grad(x));
    let ms = multistart(&mut obj, &[-1.2, 1.0], &Method::bfgs(), 3, |k, x| {
        // Small deterministic nudges for starts 1, 2.
        x[0] += 0.1 * k as f64;
        x[1] -= 0.1 * k as f64;
    })
    .unwrap();
    assert!(ms.n_converged >= 1);
    assert!((ms.best.x[0] - 1.0).abs() <= 1e-6);
}

/// Multistart argument validation.
#[test]
fn multistart_errors() {
    let mut obj = FnObjective::new(rosenbrock);
    assert!(matches!(
        multistart(&mut obj, &[0.0, 0.0], &Method::nelder_mead(), 0, |_, _| {}),
        Err(OptimError::InvalidOption { .. })
    ));
    let mut obj = FnObjective::new(rosenbrock);
    assert!(matches!(
        multistart(&mut obj, &[0.0, 0.0], &Method::nelder_mead(), 2, |_, x| {
            x[0] = f64::NAN;
        }),
        Err(OptimError::NonFinite { .. })
    ));
}
