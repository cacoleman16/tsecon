//! The Geweke-Porter-Hudak (1983) log-periodogram estimator of the memory
//! parameter `d`.
//!
//! Under fractional integration the spectral density near the origin behaves
//! like `f(lambda) ~ C * (4 sin^2(lambda/2))^{-d}` as `lambda -> 0`. Taking
//! logs of the periodogram `I(lambda_j)` at the lowest `m` Fourier frequencies
//! `lambda_j = 2*pi*j/n` gives the linear regression (Geweke & Porter-Hudak
//! 1983, *J. Time Series Analysis* 4:221-238)
//!
//! ```text
//!   log I(lambda_j) = c + d * R_j + error_j,
//!   R_j = -2 * log( 2 * sin(lambda_j / 2) )  =  -log( 4 sin^2(lambda_j/2) ),
//! ```
//!
//! so the OLS slope on the regressor `R_j` estimates `d` directly. The
//! regression is run through [`tsecon_hac::ols`] — this crate never
//! reimplements least squares.
//!
//! ## Standard errors
//!
//! Because `log(I/f)` is asymptotically `log` of a mean-one exponential, the
//! regression error has the known variance `pi^2 / 6`. The classic GPH
//! asymptotic standard error uses the large-`m` limit
//! `sum_j (R_j - Rbar)^2 -> 4m`, giving the closed form
//!
//! ```text
//!   se(d_hat) = pi / sqrt(24 m).
//! ```
//!
//! [`GphResult::se`] reports this documented asymptotic SE; for completeness
//! [`GphResult::se_regression`] additionally reports the finite-sample OLS
//! nonrobust standard error of the slope (which uses the realised regressor
//! sum of squares and residual variance).

use tsecon_hac::{ols, SeType};

use crate::error::LongMemoryError;
use crate::spectral::{check_bandwidth, low_frequency_periodogram};

/// The result of a GPH log-periodogram regression.
#[derive(Debug, Clone, PartialEq)]
pub struct GphResult {
    /// The estimated memory parameter `d` (the OLS slope on `R_j`).
    pub d: f64,
    /// The documented GPH asymptotic standard error `pi / sqrt(24 m)`.
    pub se: f64,
    /// The finite-sample OLS nonrobust standard error of the slope (uses the
    /// realised `sum (R_j - Rbar)^2` and residual variance `RSS/(m-2)`).
    pub se_regression: f64,
    /// The regression intercept `c` (an estimate of `log C` up to the
    /// periodogram's normalization; reported for completeness).
    pub intercept: f64,
    /// The number of low Fourier frequencies used.
    pub m: usize,
}

/// Estimate the memory parameter `d` by the GPH log-periodogram regression on
/// the lowest `m` Fourier frequencies.
///
/// Use [`crate::default_bandwidth`] for the textbook `m = floor(sqrt(n))`.
///
/// # Errors
///
/// [`LongMemoryError::EmptyInput`] if `x` is empty;
/// [`LongMemoryError::InvalidBandwidth`] unless `3 <= m <= (n-1)/2` (at least
/// three ordinates are needed for a slope plus a residual degree of freedom);
/// [`LongMemoryError::Spectral`] / [`LongMemoryError::NonPositivePeriodogram`]
/// from the periodogram layer; and [`LongMemoryError::Hac`] if the (well
/// conditioned) log-periodogram regression is rejected.
///
/// # Example
/// ```
/// use tsecon_longmemory::{gph, default_bandwidth};
/// // A short deterministic series just to exercise the API.
/// let x: Vec<f64> = (0..256).map(|t| ((t as f64) * 0.1).sin()).collect();
/// let m = default_bandwidth(x.len());
/// let fit = gph(&x, m).unwrap();
/// assert!(fit.d.is_finite() && fit.se > 0.0);
/// ```
pub fn gph(x: &[f64], m: usize) -> Result<GphResult, LongMemoryError> {
    if x.is_empty() {
        return Err(LongMemoryError::EmptyInput { what: "x" });
    }
    let n = x.len();
    // Need at least 3 ordinates: a slope, an intercept, and one residual
    // degree of freedom for the nonrobust SE.
    check_bandwidth(m, n, 3)?;

    let (lambdas, i_j) = low_frequency_periodogram(x, m)?;

    // Regressor R_j = -2 log(2 sin(lambda_j/2)); response y_j = log I(lambda_j).
    let ones = vec![1.0_f64; m];
    let regressor: Vec<f64> = lambdas
        .iter()
        .map(|&lam| -2.0 * (2.0 * (lam / 2.0).sin()).ln())
        .collect();
    let y: Vec<f64> = i_j.iter().map(|&i| i.ln()).collect();

    let fit = ols(&y, &[ones, regressor])?;
    let d = fit.params[1];
    let intercept = fit.params[0];
    let se_regression = fit.inference(SeType::NonRobust)?.bse[1];

    // Documented asymptotic SE: pi / sqrt(24 m).
    let se = std::f64::consts::PI / (24.0 * m as f64).sqrt();

    Ok(GphResult {
        d,
        se,
        se_regression,
        intercept,
        m,
    })
}
