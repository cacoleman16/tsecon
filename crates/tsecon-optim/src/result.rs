//! The optimization result type and termination reasons.

use core::fmt;

/// Why an optimizer stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Termination {
    /// The gradient infinity-norm fell below `grad_tol` (BFGS / L-BFGS).
    GradientTolerance,
    /// Both the simplex size and the function spread across vertices fell
    /// below their tolerances (Nelder-Mead).
    SimplexTolerance,
    /// The step between successive iterates fell below `x_tol`.
    StepTolerance,
    /// The decrease between successive objective values fell below `f_tol`.
    FunctionTolerance,
    /// The iteration budget `max_iter` was exhausted.
    MaxIterations,
    /// The objective-evaluation budget `max_fevals` was exhausted.
    MaxFevals,
    /// The strong-Wolfe line search could not find an acceptable step
    /// (typically: the iterate is at numerical precision already, the
    /// objective is non-smooth, or a parameter is diverging to a boundary
    /// of a reparameterized domain).
    LineSearchFailed,
    /// The gradient evaluated non-finite away from the starting point; the
    /// best point found so far is returned.
    GradientFailed,
}

impl Termination {
    /// Whether this reason counts as successful convergence.
    pub fn converged(self) -> bool {
        matches!(
            self,
            Termination::GradientTolerance
                | Termination::SimplexTolerance
                | Termination::StepTolerance
                | Termination::FunctionTolerance
        )
    }
}

impl fmt::Display for Termination {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let msg = match self {
            Termination::GradientTolerance => "gradient norm below tolerance",
            Termination::SimplexTolerance => "simplex size and f-spread below tolerance",
            Termination::StepTolerance => "step size below tolerance",
            Termination::FunctionTolerance => "function decrease below tolerance",
            Termination::MaxIterations => "maximum iterations reached",
            Termination::MaxFevals => "maximum function evaluations reached",
            Termination::LineSearchFailed => "line search failed to find an acceptable step",
            Termination::GradientFailed => "gradient evaluated non-finite",
        };
        f.write_str(msg)
    }
}

/// The outcome of a minimization run.
///
/// Optimizers always return the best point found, even when `converged` is
/// `false` — inspect [`termination`](OptimizeResult::termination) to decide
/// whether to trust it, restart, or report a boundary problem.
#[derive(Debug, Clone, PartialEq)]
pub struct OptimizeResult {
    /// The best point found.
    pub x: Vec<f64>,
    /// The objective value at [`x`](OptimizeResult::x).
    pub f: f64,
    /// Number of iterations performed.
    pub iterations: usize,
    /// Number of objective evaluations, including central-difference
    /// gradient probes.
    pub fevals: usize,
    /// Number of analytic-gradient evaluations (0 when the objective
    /// supplies no gradient or for Nelder-Mead).
    pub gevals: usize,
    /// Whether a convergence test was satisfied
    /// (`termination.converged()`).
    pub converged: bool,
    /// Why the optimizer stopped.
    pub termination: Termination,
}
