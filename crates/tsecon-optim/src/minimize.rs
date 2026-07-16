//! The top-level [`minimize`] entry point and the [`multistart`] helper.

use crate::error::OptimError;
use crate::neldermead::{nelder_mead, NelderMeadOptions};
use crate::objective::ObjectiveFn;
use crate::quasinewton::{bfgs, lbfgs, BfgsOptions, LbfgsOptions};
use crate::result::OptimizeResult;

/// The optimization method, carrying its per-algorithm options.
///
/// ```
/// use tsecon_optim::{minimize, FnObjective, Method, NelderMeadOptions};
///
/// let mut sphere = FnObjective::new(|x: &[f64]| x.iter().map(|v| v * v).sum::<f64>());
/// let method = Method::NelderMead(NelderMeadOptions::default());
/// let res = minimize(&mut sphere, &[1.0, -2.0], &method).unwrap();
/// assert!(res.converged && res.f < 1e-12);
/// ```
#[derive(Debug, Clone, Copy)]
pub enum Method {
    /// Adaptive Nelder-Mead simplex (Gao-Han 2012); derivative-free.
    NelderMead(NelderMeadOptions),
    /// Dense BFGS with strong-Wolfe line search (Nocedal-Wright 2006,
    /// algorithm 6.1).
    Bfgs(BfgsOptions),
    /// Limited-memory BFGS, two-loop recursion (Nocedal-Wright 2006,
    /// algorithm 7.5).
    Lbfgs(LbfgsOptions),
}

impl Method {
    /// Adaptive Nelder-Mead with default options.
    pub fn nelder_mead() -> Self {
        Method::NelderMead(NelderMeadOptions::default())
    }

    /// BFGS with default options.
    pub fn bfgs() -> Self {
        Method::Bfgs(BfgsOptions::default())
    }

    /// L-BFGS with default options (`memory = 10`).
    pub fn lbfgs() -> Self {
        Method::Lbfgs(LbfgsOptions::default())
    }
}

/// Minimizes `objective` from `x0` with the given method — the single
/// entry point every model crate calls.
///
/// Dispatches to [`nelder_mead`], [`bfgs`], or [`lbfgs`]; see those for
/// algorithmic detail. The result always contains the best point found;
/// check [`OptimizeResult::converged`].
///
/// # Errors
///
/// Whatever the dispatched algorithm returns — malformed inputs and
/// options; see [`bfgs`], [`lbfgs`], [`nelder_mead`].
pub fn minimize<F: ObjectiveFn + ?Sized>(
    objective: &mut F,
    x0: &[f64],
    method: &Method,
) -> Result<OptimizeResult, OptimError> {
    match method {
        Method::NelderMead(opts) => nelder_mead(objective, x0, opts),
        Method::Bfgs(opts) => bfgs(objective, x0, opts),
        Method::Lbfgs(opts) => lbfgs(objective, x0, opts),
    }
}

/// The outcome of a [`multistart`] run.
#[derive(Debug, Clone, PartialEq)]
pub struct MultistartResult {
    /// The best run's result (lowest objective value; on ties, the earliest
    /// start wins).
    pub best: OptimizeResult,
    /// Index of the start that produced [`best`](MultistartResult::best)
    /// (0 = the unperturbed `x0`).
    pub best_start: usize,
    /// Number of starts whose run converged.
    pub n_converged: usize,
    /// Total iterations across all starts.
    pub total_iterations: usize,
    /// Total objective evaluations across all starts.
    pub total_fevals: usize,
}

/// Runs [`minimize`] from `n_starts` starting points — `x0` itself plus
/// `n_starts - 1` perturbations of it — and returns the best result.
///
/// Threshold, Markov-switching, and GARCH-in-mean likelihoods routinely
/// have local optima; multistart is the recommended default for them
/// (ROADMAP 00, derivative-free optimizers). The caller supplies the
/// perturbation: `perturb(k, x)` receives the start index `k >= 1` and a
/// buffer preloaded with `x0` to mutate in place (draw randomness from a
/// `tsecon-rng` substream for reproducible parallel studies).
///
/// Starts whose objective is non-finite at the perturbed point simply
/// count as non-converged failures; an error is returned only if *every*
/// start fails.
///
/// # Errors
///
/// * [`OptimError::InvalidOption`] — `n_starts == 0`;
/// * [`OptimError::NonFinite`] — a perturbed start contains NaN/infinity;
/// * the last underlying error if no start produced a result.
pub fn multistart<F, P>(
    objective: &mut F,
    x0: &[f64],
    method: &Method,
    n_starts: usize,
    mut perturb: P,
) -> Result<MultistartResult, OptimError>
where
    F: ObjectiveFn + ?Sized,
    P: FnMut(usize, &mut [f64]),
{
    if n_starts == 0 {
        return Err(OptimError::InvalidOption {
            name: "n_starts",
            value: 0.0,
            requirement: "n_starts >= 1",
        });
    }
    let mut best: Option<(usize, OptimizeResult)> = None;
    let mut n_converged = 0usize;
    let mut total_iterations = 0usize;
    let mut total_fevals = 0usize;
    let mut last_err: Option<OptimError> = None;

    for k in 0..n_starts {
        let mut start = x0.to_vec();
        if k > 0 {
            perturb(k, &mut start);
            if start.iter().any(|v| !v.is_finite()) {
                return Err(OptimError::NonFinite {
                    what: "perturbed start",
                });
            }
        }
        match minimize(objective, &start, method) {
            Ok(res) => {
                total_iterations += res.iterations;
                total_fevals += res.fevals;
                if res.converged {
                    n_converged += 1;
                }
                let better = match &best {
                    None => true,
                    Some((_, b)) => res.f < b.f,
                };
                if better {
                    best = Some((k, res));
                }
            }
            Err(e) => last_err = Some(e),
        }
    }

    match best {
        Some((best_start, best)) => Ok(MultistartResult {
            best,
            best_start,
            n_converged,
            total_iterations,
            total_fevals,
        }),
        None => Err(last_err.unwrap_or(OptimError::InvalidOption {
            name: "n_starts",
            value: n_starts as f64,
            requirement: "at least one start must succeed",
        })),
    }
}
