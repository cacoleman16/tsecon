//! A general nonlinear GMM driver: minimize the GMM criterion
//! `Q(theta) = gbar(theta)' W gbar(theta)` over parameters `theta` for a
//! user-supplied moment function, using the derivative-free Nelder-Mead
//! simplex from `tsecon-optim`.
//!
//! The moment function returns, at each parameter value, the `n x m` matrix
//! of per-observation moment contributions `g_i(theta)` (row `i`, moment
//! `j`); the sample moment vector is the column mean
//! `gbar(theta) = (1/n) sum_i g_i(theta)`. This is the general Hansen (1982)
//! estimator — the linear IV-GMM of [`crate::linear`] is the special case
//! `g_i(theta) = z_i (y_i - x_i' theta)`, which has a closed form and so does
//! not need this driver.

use std::cell::Cell;

use tsecon_optim::{minimize, FnObjective, Method};

use crate::error::GmmError;

/// A fitted nonlinear GMM problem.
#[derive(Debug, Clone, PartialEq)]
pub struct NonlinearGmmFit {
    /// The parameter vector minimizing `gbar' W gbar`.
    pub params: Vec<f64>,
    /// The GMM criterion `gbar' W gbar` at the optimum.
    pub objective: f64,
    /// The sample moment vector `gbar` at the optimum (`m` entries); close to
    /// zero when the model is exactly identified (`m == params.len()`).
    pub gbar: Vec<f64>,
    /// Whether the Nelder-Mead search satisfied a convergence test.
    pub converged: bool,
    /// Number of simplex iterations performed.
    pub iterations: usize,
    /// Number of moment-function evaluations by the optimizer.
    pub fevals: usize,
    /// Number of moment conditions `m`.
    pub nmoments: usize,
    /// Number of parameters.
    pub nparams: usize,
}

fn quad_form(gbar: &[f64], w: &[f64], m: usize) -> f64 {
    // gbar' W gbar with W stored row-major m x m.
    let mut acc = 0.0;
    for i in 0..m {
        let gi = gbar[i];
        for j in 0..m {
            acc += gi * w[i * m + j] * gbar[j];
        }
    }
    acc
}

/// Column means `gbar` of the `n x m` moment matrix, returning `None` on a
/// shape mismatch (wrong row count or ragged rows).
fn column_means(moments: &[Vec<f64>], n: usize, m: usize) -> Option<Vec<f64>> {
    if moments.len() != n {
        return None;
    }
    let mut gbar = vec![0.0_f64; m];
    for row in moments {
        if row.len() != m {
            return None;
        }
        for (j, &v) in row.iter().enumerate() {
            gbar[j] += v;
        }
    }
    let nf = n as f64;
    for g in &mut gbar {
        *g /= nf;
    }
    Some(gbar)
}

/// Minimize the GMM criterion `gbar(theta)' W gbar(theta)` by Nelder-Mead.
///
/// * `moments_fn(theta)` returns the `n x m` matrix of per-observation moment
///   contributions (outer index = observation, inner = moment). Its shape
///   must be the same at every `theta`.
/// * `initial` is the starting parameter vector.
/// * `weight` is the `m x m` GMM weighting matrix, row-major; pass `None` for
///   the identity (the natural choice for an exactly identified problem, and
///   the first step of an efficient two-step scheme).
///
/// The number of moments `m` and observations `n` are inferred from a first
/// evaluation of `moments_fn` at `initial`.
///
/// # Errors
///
/// [`GmmError::EmptyInput`] for an empty `initial` or an empty moment matrix;
/// [`GmmError::UnderIdentified`] if `m < params`;
/// [`GmmError::DimensionMismatch`] if `weight` is not `m x m`;
/// [`GmmError::NonFinite`] if `weight` has non-finite entries;
/// [`GmmError::InconsistentMoments`] if the moment matrix changes shape during
/// the search; and propagation of [`GmmError::Optim`] from the optimizer.
pub fn gmm_nonlinear<F>(
    mut moments_fn: F,
    initial: &[f64],
    weight: Option<&[f64]>,
) -> Result<NonlinearGmmFit, GmmError>
where
    F: FnMut(&[f64]) -> Vec<Vec<f64>>,
{
    if initial.is_empty() {
        return Err(GmmError::EmptyInput {
            what: "initial parameter vector",
        });
    }
    // Infer (n, m) from the moment matrix at the starting point.
    let first = moments_fn(initial);
    let n = first.len();
    if n == 0 {
        return Err(GmmError::EmptyInput {
            what: "moment matrix (no observations)",
        });
    }
    let m = first[0].len();
    if m == 0 {
        return Err(GmmError::EmptyInput {
            what: "moment matrix (no moment conditions)",
        });
    }
    for row in &first {
        if row.len() != m {
            return Err(GmmError::InconsistentMoments {
                what: "the initial moment matrix has ragged rows",
            });
        }
    }
    if m < initial.len() {
        return Err(GmmError::UnderIdentified {
            moments: m,
            params: initial.len(),
        });
    }

    // Weight matrix: identity or the validated user matrix.
    let wmat: Vec<f64> = match weight {
        None => {
            let mut w = vec![0.0_f64; m * m];
            for i in 0..m {
                w[i * m + i] = 1.0;
            }
            w
        }
        Some(w) => {
            if w.len() != m * m {
                return Err(GmmError::DimensionMismatch {
                    what: "nonlinear GMM weight matrix (must be m x m)",
                    expected: m * m,
                    got: w.len(),
                });
            }
            if w.iter().any(|v| !v.is_finite()) {
                return Err(GmmError::NonFinite {
                    what: "nonlinear GMM weight matrix",
                });
            }
            w.to_vec()
        }
    };

    // Objective: Q(theta) = gbar(theta)' W gbar(theta). A shape mismatch
    // mid-search is recorded and surfaced as an error after optimization;
    // the optimizer meanwhile sees +infinity (an infeasible point). The
    // closure borrows `moments_fn` mutably; it is released (via `drop`)
    // before we re-evaluate the moments at the optimum below.
    let shape_err = Cell::new(false);
    let (best_x, objective_value, converged, iterations, fevals) = {
        let mut objective = FnObjective::new(|theta: &[f64]| {
            let moments = moments_fn(theta);
            match column_means(&moments, n, m) {
                Some(gbar) => quad_form(&gbar, &wmat, m),
                None => {
                    shape_err.set(true);
                    f64::INFINITY
                }
            }
        });
        let res = minimize(&mut objective, initial, &Method::nelder_mead())?;
        (res.x, res.f, res.converged, res.iterations, res.fevals)
    };
    if shape_err.get() {
        return Err(GmmError::InconsistentMoments {
            what: "the moment matrix changed shape during the search",
        });
    }

    // Re-evaluate the moment means at the optimum for reporting.
    let final_moments = moments_fn(&best_x);
    let gbar = column_means(&final_moments, n, m).ok_or(GmmError::InconsistentMoments {
        what: "the moment matrix changed shape at the optimum",
    })?;

    Ok(NonlinearGmmFit {
        params: best_x,
        objective: objective_value,
        gbar,
        converged,
        iterations,
        fevals,
        nmoments: m,
        nparams: initial.len(),
    })
}
