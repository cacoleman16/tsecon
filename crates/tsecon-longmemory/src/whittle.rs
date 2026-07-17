//! The Robinson (1995) Gaussian semiparametric ("local Whittle") estimator of
//! the memory parameter `d`.
//!
//! Robinson (1995, *Annals of Statistics* 23:1630-1661) concentrates the
//! local Gaussian likelihood over the unknown short-run scale, leaving the
//! univariate objective in `d` alone
//!
//! ```text
//!   R(d) = log( (1/m) sum_{j=1}^m lambda_j^{2d} I(lambda_j) )
//!          - (2d/m) sum_{j=1}^m log lambda_j,
//! ```
//!
//! evaluated at the lowest `m` Fourier frequencies `lambda_j = 2*pi*j/n`. The
//! estimator is `d_hat = argmin_{d in (-1/2, 1)} R(d)`, minimized here through
//! [`tsecon_optim`] (this crate never reimplements a minimizer). Under the
//! standard conditions `sqrt(m) (d_hat - d) -> N(0, 1/4)`, giving the known
//! asymptotic standard error
//!
//! ```text
//!   se(d_hat) = 1 / (2 sqrt(m)).
//! ```
//!
//! The concentrated objective is invariant, in its *minimizer*, to the
//! periodogram's overall normalization: rescaling every `I(lambda_j)` by a
//! constant merely adds a `d`-independent constant to `R(d)`.

use tsecon_optim::{minimize, FnObjective, Method, NelderMeadOptions};

use crate::error::LongMemoryError;
use crate::spectral::{check_bandwidth, low_frequency_periodogram};

/// The lower/upper bounds of the admissible memory interval `(-1/2, 1)`.
const D_LOWER: f64 = -0.5;
const D_UPPER: f64 = 1.0;

/// The result of a local-Whittle estimation.
#[derive(Debug, Clone, PartialEq)]
pub struct WhittleResult {
    /// The estimated memory parameter `d = argmin R(d)`.
    pub d: f64,
    /// The Robinson (1995) asymptotic standard error `1 / (2 sqrt(m))`.
    pub se: f64,
    /// The minimized value of the concentrated objective `R(d_hat)` (its level
    /// depends on the periodogram's normalization and is reported only as a
    /// diagnostic; the minimizer `d` does not).
    pub objective: f64,
    /// The number of low Fourier frequencies used.
    pub m: usize,
}

/// Evaluate the concentrated local-Whittle objective `R(d)` given the
/// precomputed `log lambda_j`, `sum log lambda_j`, and periodogram ordinates.
///
/// Returns `+inf` for `d` outside the open interval `(-1/2, 1)` so the
/// derivative-free optimizer stays inside the admissible domain (every
/// optimizer in `tsecon-optim` treats a non-finite value as an infeasible
/// point).
fn objective(d: f64, log_lambda: &[f64], sum_log_lambda: f64, i_j: &[f64]) -> f64 {
    if !(d > D_LOWER && d < D_UPPER) {
        return f64::INFINITY;
    }
    let m = i_j.len() as f64;
    // (1/m) sum_j lambda_j^{2d} I_j = (1/m) sum_j exp(2 d log lambda_j) I_j.
    let mut weighted = 0.0_f64;
    for (&ll, &i) in log_lambda.iter().zip(i_j.iter()) {
        weighted += (2.0 * d * ll).exp() * i;
    }
    weighted /= m;
    weighted.ln() - (2.0 * d / m) * sum_log_lambda
}

/// Estimate the memory parameter `d` by Robinson's (1995) local-Whittle
/// estimator on the lowest `m` Fourier frequencies.
///
/// Use [`crate::default_bandwidth`] for the textbook `m = floor(sqrt(n))`. The
/// concentrated objective is minimized over `(-1/2, 1)` by adaptive
/// Nelder-Mead from a neutral start (`d = 0`).
///
/// # Errors
///
/// [`LongMemoryError::EmptyInput`] if `x` is empty;
/// [`LongMemoryError::InvalidBandwidth`] unless `2 <= m <= (n-1)/2`;
/// [`LongMemoryError::Spectral`] / [`LongMemoryError::NonPositivePeriodogram`]
/// from the periodogram layer; [`LongMemoryError::Optim`] if the minimizer
/// rejects its inputs; and [`LongMemoryError::OptimizationFailed`] if it does
/// not reach a finite interior optimum.
///
/// # Example
/// ```
/// use tsecon_longmemory::{local_whittle, default_bandwidth};
/// let x: Vec<f64> = (0..512).map(|t| ((t as f64) * 0.05).sin()).collect();
/// let m = default_bandwidth(x.len());
/// let fit = local_whittle(&x, m).unwrap();
/// assert!(fit.d > -0.5 && fit.d < 1.0 && fit.se > 0.0);
/// ```
pub fn local_whittle(x: &[f64], m: usize) -> Result<WhittleResult, LongMemoryError> {
    if x.is_empty() {
        return Err(LongMemoryError::EmptyInput { what: "x" });
    }
    let n = x.len();
    check_bandwidth(m, n, 2)?;

    let (lambdas, i_j) = low_frequency_periodogram(x, m)?;
    let log_lambda: Vec<f64> = lambdas.iter().map(|&l| l.ln()).collect();
    let sum_log_lambda: f64 = log_lambda.iter().sum();

    let mut obj = FnObjective::new(|p: &[f64]| objective(p[0], &log_lambda, sum_log_lambda, &i_j));
    // A slightly larger initial simplex step than the default helps the 1-D
    // search bracket the minimum away from the neutral start.
    let opts = NelderMeadOptions {
        initial_step: 0.1,
        ..NelderMeadOptions::default()
    };
    let res = minimize(&mut obj, &[0.0], &Method::NelderMead(opts))?;

    let d = res.x[0];
    if !res.converged || !res.f.is_finite() || !(d > D_LOWER && d < D_UPPER) {
        return Err(LongMemoryError::OptimizationFailed {
            reason: "the concentrated objective did not converge to an interior minimum",
        });
    }

    let se = 1.0 / (2.0 * (m as f64).sqrt());
    Ok(WhittleResult {
        d,
        se,
        objective: res.f,
        m,
    })
}
