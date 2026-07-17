//! The Stambaugh (1999) finite-sample bias correction.
//!
//! When the predictor `x` is persistent and its AR(1) innovation `e` is
//! correlated with the regression error `u`, the OLS predictive slope is
//! biased in finite samples:
//!
//! ```text
//! E[beta_ols - beta] = (sigma_ue / sigma_ee) * E[rho_hat - rho].
//! ```
//!
//! Stambaugh (1999, *J. Financial Economics* 54:375-421, eqs. 4-6) removes it
//! with the least-squares AR(1) root `rho_ols` and the Kendall (1954) bias of
//! that root, `E[rho_hat - rho] ~ -(1 + 3 rho) / n`.

use tsecon_hac::ols;

use crate::error::PredRegError;
use crate::ols::ols_predictive;

/// The Stambaugh bias-corrected predictive slope and its ingredients.
#[derive(Debug, Clone, PartialEq)]
pub struct StambaughCorrection {
    /// The uncorrected OLS predictive slope `beta_ols`.
    pub beta_ols: f64,
    /// Least-squares AR(1) root of the predictor, `rho_ols`.
    pub rho_ols: f64,
    /// Kendall bias of the AR(1) root, `-(1 + 3 rho_ols) / n`
    /// (an estimate of `E[rho_hat - rho]`, which is negative).
    pub kendall_bias: f64,
    /// Innovation covariance `sigma_ue = mean(u_hat * e_hat)`.
    pub sigma_ue: f64,
    /// Predictor innovation variance `sigma_ee = mean(e_hat^2)`.
    pub sigma_ee: f64,
    /// The bias removed from `beta_ols`,
    /// `(sigma_ue / sigma_ee) * kendall_bias`.
    pub bias_term: f64,
    /// The corrected slope `beta_corrected = beta_ols - bias_term`.
    pub beta_corrected: f64,
    /// Standard error of the corrected slope. The correction shifts the point
    /// estimate by a data-dependent constant; to first order its sampling
    /// variance is the OLS variance, so this equals the OLS `se(beta_ols)`.
    pub se: f64,
    /// Number of aligned observations `N = n - 1`.
    pub nobs: usize,
}

/// Compute the Stambaugh (1999) bias-corrected predictive slope.
///
/// # Errors
///
/// Propagates the input validation of [`ols_predictive`], plus
/// [`PredRegError::Hac`] if the AR(1) fit of the predictor is singular
/// (a constant predictor).
pub fn stambaugh(r: &[f64], x: &[f64]) -> Result<StambaughCorrection, PredRegError> {
    // `ols_predictive` validates inputs (finiteness, lengths, N > 2).
    let ols_fit = ols_predictive(r, x)?;
    let n_full = x.len();
    let big_n = ols_fit.nobs; // N = n - 1

    // AR(1) least squares: x_t = c + rho x_{t-1} + e_t.
    let xlag = &x[..n_full - 1];
    let xcur = &x[1..];
    let cst = vec![1.0_f64; xlag.len()];
    let ar_design = [cst, xlag.to_vec()];
    let ar_fit = ols(xcur, &ar_design)?;
    let rho_ols = ar_fit.params[1];
    let e_hat = ar_fit.residuals; // length n - 1 = N, aligned to target time

    // Innovation second moments (divide by N, the innovation convention).
    let nf = big_n as f64;
    let sigma_ee = e_hat.iter().map(|e| e * e).sum::<f64>() / nf;
    let sigma_ue = ols_fit
        .residuals
        .iter()
        .zip(e_hat.iter())
        .map(|(u, e)| u * e)
        .sum::<f64>()
        / nf;

    let kendall_bias = -(1.0 + 3.0 * rho_ols) / n_full as f64;
    let bias_term = (sigma_ue / sigma_ee) * kendall_bias;
    let beta_corrected = ols_fit.beta - bias_term;

    Ok(StambaughCorrection {
        beta_ols: ols_fit.beta,
        rho_ols,
        kendall_bias,
        sigma_ue,
        sigma_ee,
        bias_term,
        beta_corrected,
        se: ols_fit.se,
        nobs: big_n,
    })
}
