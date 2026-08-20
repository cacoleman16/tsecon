//! Adaptive Nelder-Mead behavior: Rosenbrock 2d, adaptive coefficients in
//! higher dimension, restart support, termination semantics, and the three
//! ways `converged` used to be uninformative — a default budget that a
//! restarted search could not finish inside, an absolute `x_tol` below the
//! floating-point resolution of the incumbent (both: `converged = false` on
//! a good answer), and an absolute `f_tol` below the floating-point
//! resolution of the objective (`converged = true` on a bad one).

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

/// The **default** budget is sized per run, so asking for restarts does not
/// silently make convergence unreachable.
///
/// Regression test. The default used to be a flat `200 * n` shared across
/// all `1 + restarts` runs. A 3-parameter likelihood needs ~280 evaluations
/// to satisfy the default tolerances once, so with `restarts: 2` the 600
/// evaluations ran out during the second run and *every* fit — however
/// well-posed — came back `MaxFevals` / `converged = false`. Both
/// directions are asserted here: the same options minus the restarts must
/// converge too, and the restarted run must genuinely do more work.
#[test]
fn nm_default_budget_is_per_run() {
    let x0 = [-1.2, 1.0, -0.5];

    let mut obj = FnObjective::new(rosenbrock);
    let plain = nelder_mead(&mut obj, &x0, &NelderMeadOptions::default()).unwrap();
    assert!(
        plain.converged,
        "single run: {} after {} fevals",
        plain.termination, plain.fevals
    );

    let restarted_opts = NelderMeadOptions {
        restarts: 2,
        ..NelderMeadOptions::default()
    };
    let mut obj = FnObjective::new(rosenbrock);
    let restarted = nelder_mead(&mut obj, &x0, &restarted_opts).unwrap();
    assert!(
        restarted.converged,
        "restarted run: {} after {} fevals",
        restarted.termination, restarted.fevals
    );
    assert_eq!(restarted.termination, Termination::SimplexTolerance);
    // The restarts really ran, and the extra work bought the default
    // budget: more evaluations than one run, but still inside 200*n*3.
    assert!(restarted.fevals > plain.fevals);
    assert!(restarted.fevals <= 200 * x0.len() * 3 + x0.len() + 2);
    assert!(restarted.f <= plain.f);

    // An *explicitly* supplied budget keeps the shared-across-restarts
    // meaning: the same problem starved at one run's worth of evaluations
    // reports failure rather than pretending.
    let starved = NelderMeadOptions {
        restarts: 2,
        max_fevals: Some(plain.fevals + 10),
        ..NelderMeadOptions::default()
    };
    let mut obj = FnObjective::new(rosenbrock);
    let res = nelder_mead(&mut obj, &x0, &starved).unwrap();
    assert!(!res.converged, "starved run claimed convergence");
    assert_eq!(res.termination, Termination::MaxFevals);
}

/// The simplex-size test tracks the scale of the point it is measuring.
///
/// Regression test for an absolute `x_tol`. The same quadratic is written
/// in coordinates stretched by `c`; the minimizer sits at `x = c` and the
/// optimizer lands on it exactly. But distinct doubles near `c` are
/// `ulp(c) = eps * c` apart, so for `c` beyond about `x_tol / eps` the
/// simplex physically cannot shrink below the absolute `1e-8` — the search
/// used to burn its entire budget sitting on the exact answer and then
/// report `converged = false`. The resolution floor makes the test
/// satisfiable at every scale, and the recovered optimum stays exact.
#[test]
fn nm_x_scale_sweep_converges() {
    for &c in &[1.0_f64, 1e2, 1e4, 1e6, 1e8, 1e10, 1e12] {
        let mut obj = FnObjective::new(move |x: &[f64]| {
            x.iter().map(|v| (v / c - 1.0) * (v / c - 1.0)).sum::<f64>()
        });
        let opts = NelderMeadOptions {
            max_iter: Some(20_000),
            max_fevals: Some(20_000),
            ..NelderMeadOptions::default()
        };
        let x0 = vec![0.3 * c, -0.2 * c, 0.7 * c];
        let res = nelder_mead(&mut obj, &x0, &opts).unwrap();
        assert!(
            res.converged,
            "c = {c:e}: {} after {} fevals",
            res.termination, res.fevals
        );
        assert_eq!(res.termination, Termination::SimplexTolerance);
        // The objective is scale-free (it is written in x/c), so the same
        // accuracy is demanded of every scale.
        assert!(res.f <= 1e-14, "c = {c:e}: f = {:e}", res.f);
        // Equivalently in x: within the simplex the run stopped on, which
        // is `max(x_tol, 4 eps c)` wide.
        let reached = 4.0 * (1e-8 + 4.0 * f64::EPSILON * c);
        for &xi in &res.x {
            assert!((xi - c).abs() <= reached, "c = {c:e}: x = {:?}", res.x);
        }
        // And it costs a normal amount of work — not the whole budget.
        assert!(res.fevals < 2000, "c = {c:e}: {} fevals", res.fevals);
    }
}

/// The resolution floor is *inert* at the scales the model crates work at.
///
/// The floor is `4 eps ||x_best||_inf`; at the O(1) reparameterized working
/// spaces the model crates optimize over that is ~9e-16, six orders under
/// the default `x_tol`, so `x_tol` is what stops the search and tightening
/// it must still buy accuracy. If the floor were binding, both runs below
/// would stop at the same place.
#[test]
fn nm_resolution_floor_is_inert_at_unit_scale() {
    let budgeted = |x_tol: f64, f_tol: f64| NelderMeadOptions {
        x_tol,
        f_tol,
        max_iter: Some(20_000),
        max_fevals: Some(20_000),
        ..NelderMeadOptions::default()
    };
    let mut obj = FnObjective::new(rosenbrock);
    let loose = nelder_mead(&mut obj, &[-1.2, 1.0], &budgeted(1e-8, 1e-8)).unwrap();
    let mut obj = FnObjective::new(rosenbrock);
    let tight = nelder_mead(&mut obj, &[-1.2, 1.0], &budgeted(1e-11, 1e-11)).unwrap();

    assert!(loose.converged && tight.converged);
    assert!(
        tight.fevals > loose.fevals,
        "tightening x_tol from 1e-8 to 1e-11 changed nothing ({} vs {} \
         fevals) — the resolution floor is binding at unit scale",
        tight.fevals,
        loose.fevals
    );
    assert!(tight.f <= loose.f);
}

/// The f-spread test tracks the scale of the objective, and says so when
/// the objective runs out of resolution before the search runs out of
/// progress.
///
/// Regression test for an absolute `f_tol`. The same Rosenbrock is written
/// with an additive constant `k` — the level of an objective is arbitrary
/// (an unnormalized log-likelihood, an offset to keep a criterion
/// positive), and the minimizer is at all-ones for every `k`. But two
/// vertex values near `k` land on the same double as soon as the
/// *variation* of the objective drops under `ulp(k)`, so past `k ~ 1e7` the
/// spread hits exactly zero, every simplex move ties, and the simplex
/// collapses wherever it happens to be. With an absolute `f_tol` that came
/// back `converged = true` — at `k = 1e15`, after 104 evaluations, with `x`
/// off by 2.05. Certifying that is worse than failing on it: the caller has
/// no way to tell it from the `k = 0` run that lands within 1.6e-9.
///
/// Both ends are asserted here, because a flag that is merely pessimistic
/// is no better than one that is merely optimistic.
#[test]
fn nm_f_offset_sweep_reports_resolution_loss() {
    // Well-conditioned: the level leaves room to resolve `f_tol = 1e-8`, so
    // the certificate is granted and the answer is good.
    for &k in &[0.0_f64, 1e3, 1e6] {
        let mut obj = FnObjective::new(move |x: &[f64]| rosenbrock(x) + k);
        let res = nelder_mead(&mut obj, &[-1.2, 1.0], &NelderMeadOptions::default()).unwrap();
        assert!(res.converged, "k = {k:e}: {}", res.termination);
        assert_eq!(res.termination, Termination::SimplexTolerance);
        assert!(res.f - k <= 1e-8, "k = {k:e}: f - k = {:e}", res.f - k);
    }

    // Past `f_tol / (4 eps) ~ 1e7` the spread test is decided by rounding.
    // The optimizer still returns the best point it found — it is simply
    // honest that no tolerance was verified.
    for &k in &[1e10_f64, 1e15] {
        let mut obj = FnObjective::new(move |x: &[f64]| rosenbrock(x) + k);
        let res = nelder_mead(&mut obj, &[-1.2, 1.0], &NelderMeadOptions::default()).unwrap();
        assert!(
            !res.converged,
            "k = {k:e}: certified convergence with x = {:?} after {} fevals",
            res.x, res.fevals
        );
        assert_eq!(res.termination, Termination::ObjectiveResolution);
        assert!(res.f.is_finite() && res.f >= k);
    }

    // The boundary is where the doc comment says it is: the same `k = 1e6`
    // objective, asked for a tolerance finer than its resolution
    // (`4 eps * 1e6 = 8.9e-10`), reports the resolution limit instead.
    let mut obj = FnObjective::new(|x: &[f64]| rosenbrock(x) + 1e6);
    let opts = NelderMeadOptions {
        f_tol: 1e-12,
        max_iter: Some(20_000),
        max_fevals: Some(20_000),
        ..NelderMeadOptions::default()
    };
    let res = nelder_mead(&mut obj, &[-1.2, 1.0], &opts).unwrap();
    assert_eq!(res.termination, Termination::ObjectiveResolution);
    // ...and it stops there rather than grinding out the whole budget, the
    // way the unfloored absolute test would have.
    assert!(res.fevals < 2000, "{} fevals", res.fevals);
}

/// The f-side floor is *relative*, so multiplying the objective — which
/// leaves every level ratio intact — changes nothing.
///
/// This is the control for the additive sweep above: if the new floor were
/// really a loosening knob rather than a resolution measure it would fire
/// here too, and it must not, because a multiplied objective loses no
/// resolution at all.
#[test]
fn nm_f_multiplicative_rescaling_is_inert() {
    for &k in &[1.0_f64, 1e6, 1e12, 1e18] {
        let mut obj = FnObjective::new(move |x: &[f64]| rosenbrock(x) * k);
        let res = nelder_mead(&mut obj, &[-1.2, 1.0], &NelderMeadOptions::default()).unwrap();
        assert!(res.converged, "k = {k:e}: {}", res.termination);
        assert_eq!(res.termination, Termination::SimplexTolerance);
        for &xi in &res.x {
            assert!((xi - 1.0).abs() <= 1e-5, "k = {k:e}: x = {:?}", res.x);
        }
    }
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

/// **A near-zero starting coordinate cannot begin pre-converged** (audit
/// round 7). scipy's initial-simplex rule (`0.05 * x0_i`, `0.00025` only at
/// exactly zero) gave a coordinate starting at `1e-9` a `5e-11` simplex
/// edge — below the default `x_tol = 1e-8`, so the simplex-size test held
/// in that direction before the first iteration and the run could certify
/// convergence at the starting value. Realized in the wild by the DCS
/// local-level fits, whose standardized log-scale coordinate starts at
/// `ln(1) ≈ 0`. The displacement is now floored at the same `0.00025` an
/// exactly-zero coordinate gets.
#[test]
fn nm_near_zero_start_coordinate_still_moves() {
    // Smooth quadratic; optimum at (1.0, 0.25), start at (1.0, 1e-9): the
    // only way to the optimum is along the coordinate whose simplex edge
    // used to be degenerate.
    let mut obj = FnObjective::new(|x: &[f64]| (x[0] - 1.0).powi(2) + (x[1] - 0.25).powi(2));
    let res = nelder_mead(&mut obj, &[1.0, 1e-9], &NelderMeadOptions::default()).unwrap();
    assert!(
        (res.x[1] - 0.25).abs() < 1e-6,
        "x1 = {} stalled at its near-zero start (f = {}, {:?})",
        res.x[1],
        res.f,
        res.termination
    );
    // Sweep the whole near-zero band, both signs, at every magnitude the
    // old rule handled worse than an exact zero.
    for mag in [0.0, 1e-12, 1e-9, 1e-6, 1e-4, 4e-3] {
        for sign in [1.0, -1.0] {
            let start = sign * mag;
            let mut obj =
                FnObjective::new(|x: &[f64]| (x[0] - 1.0).powi(2) + (x[1] - 0.25).powi(2));
            let res = nelder_mead(&mut obj, &[1.0, start], &NelderMeadOptions::default()).unwrap();
            assert!(
                (res.x[1] - 0.25).abs() < 1e-6,
                "start {start:e}: x1 = {} did not reach the optimum",
                res.x[1]
            );
        }
    }
}
