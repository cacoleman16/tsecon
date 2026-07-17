//! Output types for filtering and estimation.

use crate::error::GasError;
use crate::kernel::Density;
use crate::model::{forecast_from, GasParams};

/// The output of the score-driven filter at fixed parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct GasFiltered {
    /// The filtered conditional-variance path `f_1, ..., f_N`.
    pub variance: Vec<f64>,
    /// The scaled scores `s_1, ..., s_N` used by the recursion.
    pub scores: Vec<f64>,
    /// The one-step-ahead variance `f_{N+1}` (deterministic given the
    /// filtered path and the last observation).
    pub next_variance: f64,
    /// The total log-likelihood `sum_t log p(y_t | f_t)`.
    pub loglik: f64,
}

/// The result of maximum-likelihood estimation.
#[derive(Debug, Clone, PartialEq)]
pub struct GasResults {
    /// The estimated parameters.
    pub params: GasParams,
    /// The observation density that was estimated.
    pub density: Density,
    /// The maximized log-likelihood.
    pub loglik: f64,
    /// The filtered conditional-variance path `f_1, ..., f_N` at the MLE.
    pub variance: Vec<f64>,
    /// The standardized residuals `y_t / sqrt(f_t)` at the MLE.
    pub std_resid: Vec<f64>,
    /// The one-step-ahead variance `f_{N+1}` at the MLE.
    pub next_variance: f64,
    /// The number of observations.
    pub n_obs: usize,
    /// Whether the optimizer reported convergence.
    pub converged: bool,
    /// Optimizer iterations performed.
    pub iterations: usize,
    /// Objective (log-likelihood) evaluations performed.
    pub fevals: usize,
}

impl GasResults {
    /// The number of estimated parameters (`3` for the Gaussian, `4` for
    /// the Student-t).
    #[must_use]
    pub fn n_params(&self) -> usize {
        if self.density.needs_dof() {
            4
        } else {
            3
        }
    }

    /// The Akaike information criterion `2k - 2 loglik`.
    #[must_use]
    pub fn aic(&self) -> f64 {
        2.0 * self.n_params() as f64 - 2.0 * self.loglik
    }

    /// The Bayesian information criterion `k ln(N) - 2 loglik`.
    #[must_use]
    pub fn bic(&self) -> f64 {
        self.n_params() as f64 * (self.n_obs as f64).ln() - 2.0 * self.loglik
    }

    /// The `h`-step-ahead variance forecast `[f_{N+1}, ..., f_{N+h}]` from
    /// the estimated parameters.
    ///
    /// # Errors
    ///
    /// * [`GasError::InvalidHorizon`] — `h == 0`.
    pub fn forecast(&self, h: usize) -> Result<Vec<f64>, GasError> {
        forecast_from(self.next_variance, self.params.omega, self.params.b, h)
    }
}
