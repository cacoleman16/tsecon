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
/// below their (absolute) tolerances.
#[derive(Debug, Clone, Copy)]
pub struct NelderMeadOptions {
    /// Absolute simplex-size tolerance: converged when
    /// `max_j ||x_j - x_best||_inf <= x_tol` over the non-best vertices;
    /// default `1e-8`.
    pub x_tol: f64,
    /// Absolute function-spread tolerance: converged when
    /// `max_j |f_j - f_best| <= f_tol`; default `1e-8`.
    pub f_tol: f64,
    /// Iteration budget; `None` (default) means `200 * n`.
    pub max_iter: Option<usize>,
    /// Objective-evaluation budget; `None` (default) means `200 * n`.
    /// Checked between iterations, so it can overshoot by at most `n + 2`
    /// evaluations (one shrink step).
    pub max_fevals: Option<usize>,
    /// Number of restarts after convergence: the simplex is rebuilt around
    /// the current best point and the search re-run, guarding against the
    /// false convergence Nelder-Mead is prone to (simplex collapse along a
    /// valley). Budgets are shared across restarts. Default 0.
    pub restarts: usize,
    /// Use the Gao-Han (2012) dimension-adaptive coefficients (default
    /// `true`); `false` selects the standard `(1, 2, 1/2, 1/2)`.
    pub adaptive: bool,
    /// Relative displacement used to build the initial simplex: vertex `i`
    /// displaces coordinate `i` by `initial_step * x0_i` (or by `0.00025`
    /// when `x0_i == 0`, matching scipy). Default `0.05`.
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

/// Absolute displacement used for coordinates that are exactly zero when
/// building the initial simplex (scipy's `zdelt`).
const ZERO_STEP: f64 = 0.00025;

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
    let max_iter = opts.max_iter.unwrap_or(200 * n);
    let max_fevals = opts.max_fevals.unwrap_or(200 * n);

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
            if v[i] != 0.0 {
                v[i] += opts.initial_step * v[i];
            } else {
                v[i] = ZERO_STEP;
            }
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

            // Convergence: simplex size AND f-spread below tolerance.
            let size = simplex[1..]
                .iter()
                .map(|v| {
                    v.iter()
                        .zip(&simplex[0])
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0, f64::max)
                })
                .fold(0.0, f64::max);
            let spread = fx[n] - fx[0];
            if size <= opts.x_tol && spread.abs() <= opts.f_tol {
                termination = Termination::SimplexTolerance;
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
