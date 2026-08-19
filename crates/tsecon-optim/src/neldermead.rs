//! Nelder-Mead simplex minimization with Gao-Han (2012) adaptive
//! parameters.
//!
//! The derivative-free workhorse for rough or kinked likelihoods
//! (Nelder-Mead 1965). The adaptive variant scales the
//! expansion/contraction/shrink coefficients with the dimension `n`
//! (Gao-Han 2012, "Implementing the Nelder-Mead simplex algorithm with
//! adaptive parameters", Comput. Optim. Appl. 51):
//!
//! ```text
//! reflection   alpha = 1
//! expansion    beta  = 1 + 2/n
//! contraction  gamma = 0.75 - 1/(2n)
//! shrink       delta = 1 - 1/n
//! ```
//!
//! which markedly improves behavior for `n` above ~5 where the standard
//! `(1, 2, 1/2, 1/2)` simplex stagnates. For `n = 1` (where the adaptive
//! shrink degenerates to 0) and for `n = 2` (where the formulas coincide
//! with the standard values) the standard coefficients are used.

use crate::error::OptimError;
use crate::objective::{Counted, ObjectiveFn};
use crate::result::{OptimizeResult, Termination};

/// Options for [`nelder_mead`]. Termination semantics match
/// `scipy.optimize.minimize(method="Nelder-Mead")` for easy cross-checking:
/// convergence requires **both** the simplex size and the function spread
/// below their tolerances.
#[derive(Debug, Clone, Copy)]
pub struct NelderMeadOptions {
    /// Simplex-size tolerance: converged when
    /// `max_j ||x_j - x_best||_inf <= max(x_tol, RESOLUTION_ULPS * eps *
    /// ||x_best||_inf)` over the non-best vertices; default `1e-8`.
    ///
    /// The second term is a floating-point *resolution floor*, not a
    /// loosening knob: distinct simplex vertices near a point of magnitude
    /// `m` cannot be closer than `ulp(m) = eps * m`, so an absolute
    /// `x_tol` below that is unsatisfiable and the search would grind out
    /// its whole budget on an already-exact answer. The floor only binds
    /// once `||x_best||_inf` exceeds roughly `x_tol / (4 * eps)` — about
    /// `1e7` at the default — and is inert for the O(1) reparameterized
    /// working spaces the model crates optimize over.
    pub x_tol: f64,
    /// Function-spread tolerance: the run stops when
    /// `max_j |f_j - f_best| <= max(f_tol, RESOLUTION_ULPS * eps *
    /// |f_best|)`; default `1e-8`.
    ///
    /// The second term is the same floating-point *resolution floor* as on
    /// [`x_tol`](NelderMeadOptions::x_tol), and it is needed for the same
    /// reason: vertex values of magnitude `|f|` cannot differ by less than
    /// `ulp(|f|) = eps * |f|` without being the same double, so an absolute
    /// `f_tol` below that is not a tolerance the search can be held to.
    ///
    /// Unlike the x side, though, the floor is not a fix — it is a
    /// *detector*. Reaching it means the objective has stopped
    /// discriminating between the vertices, so the stopping decision was
    /// made by rounding rather than by the search, and the point returned
    /// is only as good as the objective's conditioning allowed. Whenever
    /// `f_tol` is finer than that floor the run therefore terminates
    /// [`Termination::ObjectiveResolution`] and reports `converged =
    /// false`, rather than certifying a tolerance it never verified. The
    /// floor binds once `|f_best|` exceeds roughly `f_tol / (4 * eps)` —
    /// about `1e7` at the default — which no correctly centered objective
    /// in this workspace reaches; the usual cause is a large additive
    /// constant, and subtracting it restores the certificate.
    ///
    /// Note that `f_tol = 0` (exact equality of the vertex values) is
    /// finer than the floor at every nonzero `|f_best|`, so it always
    /// terminates `ObjectiveResolution`.
    pub f_tol: f64,
    /// Iteration budget; `None` (default) means `200 * n` **per run**, that
    /// is `200 * n * (1 + restarts)` in total. An explicit budget is a
    /// total, shared across restarts.
    pub max_iter: Option<usize>,
    /// Objective-evaluation budget; `None` (default) means `200 * n` **per
    /// run**, that is `200 * n * (1 + restarts)` in total; an explicit
    /// budget is a total, shared across restarts. Checked between
    /// iterations, so it can overshoot by at most `n + 2` evaluations (one
    /// shrink step).
    pub max_fevals: Option<usize>,
    /// Number of restarts after convergence: the simplex is rebuilt around
    /// the current best point and the search re-run, guarding against the
    /// false convergence Nelder-Mead is prone to (simplex collapse along a
    /// valley). An explicitly supplied budget is shared across restarts;
    /// the *default* budget is sized per run, so raising `restarts` raises
    /// the default budget with it. Default 0.
    pub restarts: usize,
    /// Use the Gao-Han (2012) dimension-adaptive coefficients (default
    /// `true`); `false` selects the standard `(1, 2, 1/2, 1/2)`.
    pub adaptive: bool,
    /// Relative displacement used to build the initial simplex: vertex `i`
    /// displaces coordinate `i` away from zero by
    /// `max(initial_step * |x0_i|, 0.00025)`. Default `0.05`.
    ///
    /// The absolute floor closes a mixed-scale hole in scipy's rule
    /// (`nonzdelt * x0_i`, with `zdelt = 0.00025` reserved for *exactly*
    /// zero coordinates): a coordinate starting at `1e-9` used to get a
    /// `5e-11` simplex edge — *smaller* than an exactly-zero coordinate's,
    /// and already below the default `x_tol`, so the simplex-size test was
    /// satisfied in that direction before the search began and the run
    /// could certify convergence at the starting value (audit round 7:
    /// realized by the DCS local-level fits, whose standardized log-scale
    /// coordinate starts at `ln(1) ≈ 0`). For `|x0_i| >= 0.005` the floor
    /// is inert and the vertex is bit-identical to scipy's.
    pub initial_step: f64,
}

impl Default for NelderMeadOptions {
    fn default() -> Self {
        Self {
            x_tol: 1e-8,
            f_tol: 1e-8,
            max_iter: None,
            max_fevals: None,
            restarts: 0,
            adaptive: true,
            initial_step: 0.05,
        }
    }
}

impl NelderMeadOptions {
    fn validate(&self) -> Result<(), OptimError> {
        if !(self.x_tol >= 0.0 && self.x_tol.is_finite()) {
            return Err(OptimError::InvalidOption {
                name: "x_tol",
                value: self.x_tol,
                requirement: "0 <= x_tol < infinity",
            });
        }
        if !(self.f_tol >= 0.0 && self.f_tol.is_finite()) {
            return Err(OptimError::InvalidOption {
                name: "f_tol",
                value: self.f_tol,
                requirement: "0 <= f_tol < infinity",
            });
        }
        if !(self.initial_step > 0.0 && self.initial_step.is_finite()) {
            return Err(OptimError::InvalidOption {
                name: "initial_step",
                value: self.initial_step,
                requirement: "0 < initial_step < infinity",
            });
        }
        Ok(())
    }
}

/// Absolute floor on the initial-simplex displacement (scipy's `zdelt`,
/// applied here as a floor for *near*-zero coordinates too — see
/// [`NelderMeadOptions::initial_step`]).
const ZERO_STEP: f64 = 0.00025;

/// Width, in units of the last place, of the floating-point resolution
/// floor under [`NelderMeadOptions::x_tol`] and
/// [`NelderMeadOptions::f_tol`]. A simplex of `n + 1` distinct vertices
/// needs a few ulps of room around the incumbent, so neither the size test
/// nor the spread test can be driven below this no matter the budget.
const RESOLUTION_ULPS: f64 = 4.0;

/// Minimizes `f` by the Nelder-Mead simplex method with Gao-Han (2012)
/// adaptive parameters. See [`NelderMeadOptions`] for the convergence
/// tests, budgets, and restart support; the module docs give the
/// coefficient formulas and references.
///
/// Non-finite objective values are treated as `+infinity` (infeasible), so
/// the simplex simply moves away from them.
///
/// # Errors
///
/// * [`OptimError::EmptyInput`] — `x0` is empty;
/// * [`OptimError::NonFinite`] — `x0` contains NaN/infinity, or the
///   objective is non-finite at every vertex of the initial simplex;
/// * [`OptimError::InvalidOption`] — malformed options.
pub fn nelder_mead<F: ObjectiveFn + ?Sized>(
    f: &mut F,
    x0: &[f64],
    opts: &NelderMeadOptions,
) -> Result<OptimizeResult, OptimError> {
    opts.validate()?;
    let n = x0.len();
    if n == 0 {
        return Err(OptimError::EmptyInput { what: "x0" });
    }
    if x0.iter().any(|v| !v.is_finite()) {
        return Err(OptimError::NonFinite { what: "x0" });
    }
    // Default budgets are per run: a restarted search re-solves the problem
    // from a fresh simplex each time, so a budget sized for one run and
    // then shared across `1 + restarts` of them can only ever terminate on
    // exhaustion, leaving `converged` false on every fit.
    let runs = 1usize.saturating_add(opts.restarts);
    let per_run = 200usize.saturating_mul(n);
    let max_iter = opts
        .max_iter
        .unwrap_or_else(|| per_run.saturating_mul(runs));
    let max_fevals = opts
        .max_fevals
        .unwrap_or_else(|| per_run.saturating_mul(runs));

    // Gao-Han (2012) adaptive coefficients; standard for n <= 2 (identical
    // at n = 2, degenerate shrink at n = 1).
    let nf = n as f64;
    let (alpha, beta, gamma, delta) = if opts.adaptive && n > 2 {
        (1.0, 1.0 + 2.0 / nf, 0.75 - 1.0 / (2.0 * nf), 1.0 - 1.0 / nf)
    } else {
        (1.0, 2.0, 0.5, 0.5)
    };

    let mut c = Counted::new(f);
    let mut iterations = 0usize;
    let mut termination;

    // Simplex state, re-seeded around the incumbent best on each restart.
    let mut seed = x0.to_vec();
    let mut best_x = x0.to_vec();
    let mut best_f = f64::INFINITY;
    let mut runs_done = 0usize;

    'restart: loop {
        // Build the initial simplex around `seed`.
        let mut simplex: Vec<Vec<f64>> = Vec::with_capacity(n + 1);
        simplex.push(seed.clone());
        for i in 0..n {
            let mut v = seed.clone();
            // Relative displacement with an absolute floor, away from
            // zero. Bit-identical to scipy's `nonzdelt * x0_i` whenever
            // `|x0_i| >= ZERO_STEP / initial_step` (0.005 at the default),
            // and to its `zdelt` at exactly zero; in between, the floor
            // keeps the simplex edge above the default `x_tol` so a
            // near-zero coordinate cannot start out pre-converged (see
            // `NelderMeadOptions::initial_step`).
            let step = (opts.initial_step * v[i].abs()).max(ZERO_STEP);
            v[i] += if v[i] < 0.0 { -step } else { step };
            simplex.push(v);
        }
        let mut fx: Vec<f64> = simplex.iter().map(|v| c.value(v)).collect();
        if fx.iter().all(|v| !v.is_finite()) {
            return Err(OptimError::NonFinite {
                what: "objective on the initial simplex",
            });
        }

        loop {
            // Order the simplex: index 0 = best, index n = worst.
            let mut order: Vec<usize> = (0..=n).collect();
            order.sort_by(|&a, &b| fx[a].total_cmp(&fx[b]));
            let permuted: Vec<Vec<f64>> = order.iter().map(|&i| simplex[i].clone()).collect();
            let permuted_f: Vec<f64> = order.iter().map(|&i| fx[i]).collect();
            simplex = permuted;
            fx = permuted_f;

            if fx[0] < best_f {
                best_f = fx[0];
                best_x.copy_from_slice(&simplex[0]);
            }

            // Convergence: simplex size AND f-spread below tolerance. Both
            // tests are floored at the floating-point resolution of the
            // quantity they measure — two distinct doubles near a magnitude
            // `m` cannot be closer than `ulp(m)`, so an absolute tolerance
            // below that can never be met and the search would burn its
            // whole budget sitting on an answer it already has.
            //
            // The two floors mean opposite things, though. On the x side
            // the floor is the fix: the simplex has genuinely shrunk to the
            // resolution of `x_best`, and the answer is as good as doubles
            // get. On the f side reaching the floor says the objective has
            // stopped *discriminating* — the vertex values agree only
            // because they rounded onto the same double, which happens as
            // soon as the variation of `f` drops under `ulp(|f_best|)`
            // however far the incumbent still is from the minimum. So a run
            // that stops there is reported as `ObjectiveResolution` rather
            // than certified against a tolerance it never verified.
            let size = simplex[1..]
                .iter()
                .map(|v| {
                    v.iter()
                        .zip(&simplex[0])
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0, f64::max)
                })
                .fold(0.0, f64::max);
            let x_scale = simplex[0].iter().fold(0.0_f64, |m, v| m.max(v.abs()));
            let x_stop = opts.x_tol.max(RESOLUTION_ULPS * f64::EPSILON * x_scale);
            let f_resolution = RESOLUTION_ULPS * f64::EPSILON * fx[0].abs();
            let f_stop = opts.f_tol.max(f_resolution);
            let spread = fx[n] - fx[0];
            if size <= x_stop && spread.abs() <= f_stop {
                termination = if opts.f_tol < f_resolution {
                    Termination::ObjectiveResolution
                } else {
                    Termination::SimplexTolerance
                };
                break;
            }
            if iterations >= max_iter {
                termination = Termination::MaxIterations;
                break 'restart;
            }
            if c.fevals >= max_fevals {
                termination = Termination::MaxFevals;
                break 'restart;
            }
            iterations += 1;

            // Centroid of all vertices but the worst.
            let mut cen = vec![0.0; n];
            for v in &simplex[..n] {
                for (ci, vi) in cen.iter_mut().zip(v) {
                    *ci += vi;
                }
            }
            for ci in cen.iter_mut() {
                *ci /= nf;
            }

            let towards = |coef: f64| -> Vec<f64> {
                cen.iter()
                    .zip(&simplex[n])
                    .map(|(&ci, &wi)| ci + coef * (ci - wi))
                    .collect()
            };

            // Reflection.
            let xr = towards(alpha);
            let fr = c.value(&xr);
            if fr < fx[0] {
                // Expansion (greedy value form, as in scipy).
                let xe = towards(alpha * beta);
                let fe = c.value(&xe);
                if fe < fr {
                    simplex[n] = xe;
                    fx[n] = fe;
                } else {
                    simplex[n] = xr;
                    fx[n] = fr;
                }
            } else if fr < fx[n - 1] {
                simplex[n] = xr;
                fx[n] = fr;
            } else {
                let mut shrink = false;
                if fr < fx[n] {
                    // Outside contraction.
                    let xc = towards(alpha * gamma);
                    let fc = c.value(&xc);
                    if fc <= fr {
                        simplex[n] = xc;
                        fx[n] = fc;
                    } else {
                        shrink = true;
                    }
                } else {
                    // Inside contraction.
                    let xcc = towards(-gamma);
                    let fcc = c.value(&xcc);
                    if fcc < fx[n] {
                        simplex[n] = xcc;
                        fx[n] = fcc;
                    } else {
                        shrink = true;
                    }
                }
                if shrink {
                    for j in 1..=n {
                        let (head, tail) = simplex.split_at_mut(j);
                        let bestv = &head[0];
                        for (vi, bi) in tail[0].iter_mut().zip(bestv) {
                            *vi = bi + delta * (*vi - bi);
                        }
                        fx[j] = c.value(&simplex[j]);
                    }
                }
            }
        }

        // A run converged (SimplexTolerance).
        runs_done += 1;
        if runs_done > opts.restarts {
            break;
        }
        seed.copy_from_slice(&best_x);
    }

    let converged = termination.converged();
    Ok(OptimizeResult {
        x: best_x,
        f: best_f,
        iterations,
        fevals: c.fevals,
        gevals: 0,
        converged,
        termination,
    })
}
