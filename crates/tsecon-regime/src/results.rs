//! Result types and regime-classification helpers.

use crate::params::MsarParams;

/// Output of [`MarkovSwitchingAr::filter`](crate::MarkovSwitchingAr::filter):
/// the Hamilton (1989) filter log-likelihood and filtered regime
/// probabilities.
#[derive(Debug, Clone, PartialEq)]
pub struct FilterResult {
    /// The prediction-error-decomposition log-likelihood.
    pub loglik: f64,
    /// `n`-by-`k` filtered marginal regime probabilities `P(S_t | Y_t)`,
    /// `n = T - order`. `filtered_prob[t][i]` is the probability of regime
    /// `i` at usable time `t`.
    pub filtered_prob: Vec<Vec<f64>>,
}

/// Output of [`MarkovSwitchingAr::smooth`](crate::MarkovSwitchingAr::smooth):
/// filtered and Kim (1994) smoothed regime probabilities.
#[derive(Debug, Clone, PartialEq)]
pub struct SmoothResult {
    /// The prediction-error-decomposition log-likelihood.
    pub loglik: f64,
    /// `n`-by-`k` filtered marginal regime probabilities `P(S_t | Y_t)`.
    pub filtered_prob: Vec<Vec<f64>>,
    /// `n`-by-`k` smoothed marginal regime probabilities `P(S_t | Y_T)`.
    pub smoothed_prob: Vec<Vec<f64>>,
}

impl SmoothResult {
    /// The most probable regime at each period under the smoothed
    /// probabilities (see [`classify`]).
    pub fn classified(&self) -> Vec<usize> {
        classify(&self.smoothed_prob)
    }
}

/// Output of [`MarkovSwitchingAr::fit`](crate::MarkovSwitchingAr::fit): the
/// estimated parameters, the maximized log-likelihood, and the smoothed
/// probabilities at the optimum.
#[derive(Debug, Clone, PartialEq)]
pub struct FitResult {
    /// The estimated parameters.
    pub params: MsarParams,
    /// The maximized (observed-data) log-likelihood.
    pub loglik: f64,
    /// `n`-by-`k` smoothed regime probabilities at the estimate.
    pub smoothed_prob: Vec<Vec<f64>>,
    /// Number of EM iterations performed.
    pub iterations: usize,
    /// Whether the log-likelihood increment fell below the tolerance
    /// (`false` if `max_iter` was reached first).
    pub converged: bool,
}

impl FitResult {
    /// The most probable regime at each period under the smoothed
    /// probabilities (see [`classify`]).
    pub fn classified(&self) -> Vec<usize> {
        classify(&self.smoothed_prob)
    }
}

/// The most probable regime per period: `argmax_i prob[t][i]`.
///
/// This is the pointwise (marginal) maximum-a-posteriori regime path, the
/// usual regime-classification summary of a Markov-switching fit (Hamilton
/// 1989). Ties resolve to the lowest regime index. An empty row maps to
/// regime `0`.
pub fn classify(prob: &[Vec<f64>]) -> Vec<usize> {
    prob.iter()
        .map(|row| {
            let mut best = 0usize;
            let mut best_val = f64::NEG_INFINITY;
            for (i, &v) in row.iter().enumerate() {
                if v > best_val {
                    best_val = v;
                    best = i;
                }
            }
            best
        })
        .collect()
}
