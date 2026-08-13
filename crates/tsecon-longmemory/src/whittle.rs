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
//! [`tsecon_optim`] (this crate never reimplements a minimizer).
//!
//! ## Standard errors
//!
//! Expanding the score and Hessian of `R` at the truth gives
//! `Var(d_hat) = 1 / (4 sum_j nu_j^2)` with
//! `nu_j = log lambda_j - (1/m) sum_k log lambda_k`, so
//!
//! ```text
//!   se(d_hat) = 1 / ( 2 sqrt( sum_j nu_j^2 ) ).
//! ```
//!
//! [`WhittleResult::se`] reports this. Robinson's stated
//! `sqrt(m)(d_hat - d) -> N(0, 1/4)` follows by substituting the limit
//! `(1/m) sum_j nu_j^2 -> 1`, giving `1 / (2 sqrt(m))`; that closed form is
//! reported separately as [`WhittleResult::se_asymptotic`]. Note `nu_j` does
//! not depend on `n` — `log lambda_j = log(2 pi / n) + log j` and the centring
//! removes the `n` term — so the correction is a pure function of `m`.
//!
//! The limit is approached slowly: `(1/m) sum_j nu_j^2` is `0.65` at `m = 22`,
//! `0.71` at `m = 32` and `0.76` at `m = 45`, so at the library's own default
//! bandwidth `m = floor(sqrt(n))` the asymptotic constant is roughly a quarter
//! too narrow. On exact ARFIMA(0, d, 0) draws (Davies-Harte circulant
//! embedding, 1500 replications, `d in {0, 0.2, 0.4}`) at `n = 512`, `m = 22`,
//! the realised sampling standard deviation of `d_hat` is `0.139-0.141` while
//! `1 / (2 sqrt(m)) = 0.107` (nominal-95% intervals cover 86-87%) and the `se`
//! above is `0.133` (covering 93%).
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
    /// The local-Whittle standard error at the bandwidth actually used,
    /// `1 / (2 sqrt(sum_j (log lambda_j - mean log lambda)^2))`. This is the
    /// value to build confidence intervals from.
    pub se: f64,
    /// Robinson's large-`m` closed form `1 / (2 sqrt(m))`, i.e. [`Self::se`]
    /// with `(1/m) sum_j nu_j^2` replaced by its limit `1`. Reported for
    /// reference only: it is materially too narrow at small `m` (see the
    /// module documentation).
    pub se_asymptotic: f64,
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

    // se = 1 / (2 sqrt(sum_j (log lambda_j - mean)^2)). check_bandwidth has
    // already guaranteed m >= 2 distinct frequencies, so the sum is positive.
    let ll_bar = sum_log_lambda / m as f64;
    let s_nu: f64 = log_lambda
        .iter()
        .map(|&l| (l - ll_bar) * (l - ll_bar))
        .sum();
    let se = 1.0 / (2.0 * s_nu.sqrt());
    // The large-m limit of the same expression ((1/m) sum nu_j^2 -> 1).
    let se_asymptotic = 1.0 / (2.0 * (m as f64).sqrt());
    Ok(WhittleResult {
        d,
        se,
        se_asymptotic,
        objective: res.f,
        m,
    })
}
