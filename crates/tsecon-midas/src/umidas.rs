//! Unrestricted (U-MIDAS) and ADL-MIDAS regressions.
//!
//! Both estimators are *exactly ordinary least squares* on a stacked design,
//! so they delegate every numeric — the solve and the standard errors
//! (nonrobust / HC / HAC) — to the library's single HAC/OLS owner,
//! [`tsecon_hac`]. This module only assembles the design and reports the fit.
//!
//! ## U-MIDAS (Foroni, Marcellino & Schumacher 2015)
//!
//! When the frequency mismatch is small, restricting the high-frequency lag
//! coefficients to a weight function buys little; the *unrestricted* MIDAS
//! simply keeps all `K` lag coefficients free and estimates them by OLS:
//!
//! ```text
//! y_t = c + sum_{k=1}^{K} b_k x_{t,k} + u_t,
//! ```
//!
//! with `x_{t,k}` the most-recent-first stacked high-frequency lags (see
//! [`crate::stack_high_freq_lags`]). Parameter order is `[c, b_1, ..., b_K]`.
//!
//! ## ADL-MIDAS
//!
//! Adding an autoregressive term — lagged low-frequency `y` — to the U-MIDAS
//! design gives the autoregressive-distributed-lag MIDAS specification
//! (Andreou, Ghysels & Kourtellos 2013; Ghysels 2016):
//!
//! ```text
//! y_t = c + sum_{p=1}^{P} rho_p y_{t-p} + sum_{k=1}^{K} b_k x_{t,k} + u_t.
//! ```
//!
//! Parameter order is `[c, rho_1, ..., rho_P, b_1, ..., b_K]`. The caller
//! passes the lagged-`y` columns and the stacked high-frequency columns
//! already aligned to the (trimmed) target `y`; index bookkeeping lives in the
//! design builder, not here.

use tsecon_hac::{ols, SeType};

use crate::error::MidasError;

/// A fitted (U-)MIDAS regression: OLS point estimates plus standard errors,
/// t-statistics, covariance, and the centered R-squared.
#[derive(Debug, Clone, PartialEq)]
pub struct MidasFit {
    /// Coefficient estimates in design order (`[c, ...]`; see the module and
    /// function docs for the exact layout of each specification).
    pub params: Vec<f64>,
    /// Standard errors under the requested [`SeType`], one per parameter.
    pub bse: Vec<f64>,
    /// t-statistics `params / bse`, one per parameter.
    pub tvalues: Vec<f64>,
    /// Parameter covariance matrix, `nparams x nparams` row-major.
    pub cov: Vec<f64>,
    /// Residuals `u_t = y_t - x_t' b`, length [`nobs`](MidasFit::nobs).
    pub residuals: Vec<f64>,
    /// Number of low-frequency observations used in the fit.
    pub nobs: usize,
    /// Number of estimated parameters (including the intercept).
    pub nparams: usize,
    /// Centered coefficient of determination
    /// `1 - RSS / sum_t (y_t - ybar)^2` (the design always includes an
    /// intercept, so this is the statsmodels `rsquared`). `NaN` if the target
    /// is numerically constant.
    pub rsquared: f64,
}

/// Fit an unrestricted MIDAS regression `y = c + sum_k b_k x_{t,k} + u` by
/// OLS (Foroni, Marcellino & Schumacher 2015).
///
/// `hf_lags` are the `K` most-recent-first stacked high-frequency lag columns,
/// each aligned to `y` (`hf_lags[j].len() == y.len()`); build them with
/// [`crate::stack_high_freq_lags`]. `se_type` selects the standard-error
/// flavor from the shared HAC engine (nonrobust reproduces the golden
/// fixture; [`SeType::Hac`] gives serial-correlation-robust errors). Parameter
/// order is `[c, b_1, ..., b_K]`.
///
/// # Errors
///
/// [`MidasError::InvalidLagCount`] if `hf_lags` is empty; otherwise any
/// [`MidasError::Ols`] the shared engine raises (length mismatch between a
/// column and `y`, no residual degrees of freedom, collinear/singular design,
/// non-finite input, or an invalid HAC bandwidth).
pub fn umidas(y: &[f64], hf_lags: &[Vec<f64>], se_type: SeType) -> Result<MidasFit, MidasError> {
    if hf_lags.is_empty() {
        return Err(MidasError::InvalidLagCount {
            what: "U-MIDAS regression",
            k: 0,
            needed: 1,
        });
    }
    let mut columns = Vec::with_capacity(hf_lags.len() + 1);
    columns.push(vec![1.0; y.len()]);
    columns.extend(hf_lags.iter().cloned());
    fit_design(y, columns, se_type)
}

/// Fit an ADL-MIDAS regression
/// `y = c + sum_p rho_p y_{t-p} + sum_k b_k x_{t,k} + u` by OLS (Andreou,
/// Ghysels & Kourtellos 2013).
///
/// `y_lags` are the `P` lagged low-frequency target columns
/// (`y_{t-1}, ..., y_{t-P}`) and `hf_lags` the `K` stacked high-frequency lag
/// columns, all aligned to the (trimmed) target `y`. Parameter order is
/// `[c, rho_1, ..., rho_P, b_1, ..., b_K]`.
///
/// # Errors
///
/// [`MidasError::InvalidLagCount`] if both `y_lags` and `hf_lags` are empty;
/// otherwise any [`MidasError::Ols`] the shared engine raises.
pub fn adl_midas(
    y: &[f64],
    y_lags: &[Vec<f64>],
    hf_lags: &[Vec<f64>],
    se_type: SeType,
) -> Result<MidasFit, MidasError> {
    if y_lags.is_empty() && hf_lags.is_empty() {
        return Err(MidasError::InvalidLagCount {
            what: "ADL-MIDAS regression",
            k: 0,
            needed: 1,
        });
    }
    let mut columns = Vec::with_capacity(y_lags.len() + hf_lags.len() + 1);
    columns.push(vec![1.0; y.len()]);
    columns.extend(y_lags.iter().cloned());
    columns.extend(hf_lags.iter().cloned());
    fit_design(y, columns, se_type)
}

/// Shared OLS fit + inference + centered R-squared for a fully assembled
/// design (the intercept column is already `columns[0]`).
fn fit_design(y: &[f64], columns: Vec<Vec<f64>>, se_type: SeType) -> Result<MidasFit, MidasError> {
    let fit = ols(y, &columns)?;
    let inference = fit.inference(se_type)?;

    let n = fit.nobs;
    let rss: f64 = fit.residuals.iter().map(|u| u * u).sum();
    let rsquared = if n == 0 {
        f64::NAN
    } else {
        let ybar = y.iter().sum::<f64>() / n as f64;
        let tss: f64 = y.iter().map(|v| (v - ybar).powi(2)).sum();
        if tss > 0.0 {
            1.0 - rss / tss
        } else {
            f64::NAN
        }
    };

    Ok(MidasFit {
        params: fit.params,
        bse: inference.bse,
        tvalues: inference.tvalues,
        cov: inference.cov,
        residuals: fit.residuals,
        nobs: n,
        nparams: fit.nparams,
        rsquared,
    })
}
