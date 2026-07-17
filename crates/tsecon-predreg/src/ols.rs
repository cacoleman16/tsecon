//! The plain OLS predictive regression `r_{t+1} = alpha + beta x_t + u_{t+1}`.
//!
//! Least squares is delegated to [`tsecon_hac::ols`] — the library's single
//! OLS owner — so the coefficient, its nonrobust standard error, and the
//! t-statistic match every other regression module bit for bit.

use tsecon_hac::{ols, SeType};

use crate::align::align_pair;
use crate::error::PredRegError;

/// A fitted OLS predictive regression.
#[derive(Debug, Clone, PartialEq)]
pub struct OlsPredReg {
    /// Intercept `alpha`.
    pub alpha: f64,
    /// Predictive slope `beta_ols` on the lagged predictor.
    pub beta: f64,
    /// Nonrobust standard error of `beta` (statsmodels `cov_type="nonrobust"`).
    pub se: f64,
    /// t-statistic `beta / se` for `H0: beta = 0`.
    pub tstat: f64,
    /// Unbiased residual variance `sum u_hat^2 / (N - 2)`.
    pub sigma2_u: f64,
    /// Residuals `u_hat_t = b_t - alpha - beta a_t`, length `N = n - 1`.
    pub residuals: Vec<f64>,
    /// Number of aligned observations `N = n - 1`.
    pub nobs: usize,
}

/// Fit the OLS predictive regression of `r_{t+1}` on `x_t`.
///
/// Uses the crate's one-period alignment: `a_t = x[t]`, `b_t = r[t+1]`,
/// `t = 0 .. n-2`. The design is `[const, a_t]`.
///
/// # Errors
///
/// [`PredRegError::EmptyInput`] / [`PredRegError::DimensionMismatch`] /
/// [`PredRegError::NonFinite`] on malformed input, and
/// [`PredRegError::DegreesOfFreedom`] when fewer than three observations are
/// supplied (`N = n - 1 <= 2`). Propagates [`PredRegError::Hac`] from the OLS
/// solve (e.g. a constant predictor with a singular design).
pub fn ols_predictive(r: &[f64], x: &[f64]) -> Result<OlsPredReg, PredRegError> {
    let (a, b) = align_pair(r, x)?;
    let n = a.len();
    let cst = vec![1.0_f64; n];
    let design = [cst, a.to_vec()];
    let fit = ols(b, &design)?;
    let inf = fit.inference(SeType::NonRobust)?;
    let alpha = fit.params[0];
    let beta = fit.params[1];
    let se = inf.bse[1];
    let tstat = inf.tvalues[1];
    // Unbiased residual variance sigma2_u = RSS / (N - k), k = 2.
    let rss: f64 = fit.residuals.iter().map(|e| e * e).sum();
    let sigma2_u = rss / (n as f64 - 2.0);
    Ok(OlsPredReg {
        alpha,
        beta,
        se,
        tstat,
        sigma2_u,
        residuals: fit.residuals,
        nobs: n,
    })
}
