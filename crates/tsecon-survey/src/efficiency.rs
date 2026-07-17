//! Forecast efficiency / rationality: the Mincer-Zarnowitz test in
//! error-on-forecast form, with a HAC Wald statistic.
//!
//! Under rational expectations the forecast error must be orthogonal to any
//! information available when the forecast was made — in particular to the
//! forecast itself. Regressing the error on a constant and the forecast (or on
//! any set of predetermined regressors),
//!
//! ```text
//! (actual_t - forecast_t)  =  a  +  b * forecast_t  +  e_t,
//! ```
//!
//! forecast rationality is the joint hypothesis `H0: a = 0 and b = 0`
//! (equivalently, in the level form `actual = a + (1+b) forecast`, the classic
//! Mincer-Zarnowitz (1969) `intercept = 0, slope = 1`). With more regressors
//! `x_{1,t}, .., x_{q,t}` the same logic gives a general orthogonality /
//! efficiency test: every coefficient — intercept included — should be zero.
//!
//! The joint test is a Wald statistic built from the Bartlett/Newey-West HAC
//! covariance `V` of the OLS coefficients `b`:
//!
//! ```text
//! W  =  b' V^{-1} b   ~   chi-square(k),      k = 1 + q,
//! ```
//!
//! matching statsmodels `res.wald_test(np.eye(k), use_f=False)` (chi-square,
//! not F, because HAC inference defaults to `use_t=False`). A large `W` (small
//! p-value) rejects forecast efficiency.

use tsecon_linalg::faer::linalg::solvers::DenseSolveCore;
use tsecon_linalg::faer::{Mat, Side};
use tsecon_stats::chi2_sf;

use crate::common::{hac_regression, HacBandwidth};
use crate::error::SurveyError;

/// Result of the Mincer-Zarnowitz forecast-efficiency regression and its HAC
/// Wald test.
#[derive(Debug, Clone, PartialEq)]
pub struct EfficiencyTest {
    /// Coefficients: `params[0]` is the intercept, then one slope per
    /// regressor (the forecast, and any further predetermined signals).
    pub params: Vec<f64>,
    /// HAC standard errors, one per parameter.
    pub bse: Vec<f64>,
    /// t-statistics `params / bse`, one per parameter.
    pub tvalues: Vec<f64>,
    /// Two-sided normal p-values, one per parameter.
    pub pvalues: Vec<f64>,
    /// Centered R-squared of the regression.
    pub r_squared: f64,
    /// The Wald statistic `W = b' V^{-1} b` for `H0: all coefficients = 0`.
    pub wald: f64,
    /// Degrees of freedom of the Wald test, `k = 1 + (number of regressors)`.
    pub wald_df: usize,
    /// The chi-square(`wald_df`) p-value `chi2_sf(wald, wald_df)`.
    pub wald_pvalue: f64,
    /// The Newey-West lag truncation `L` actually used.
    pub maxlags: usize,
    /// Number of observations in the regression.
    pub nobs: usize,
}

/// Run the Mincer-Zarnowitz efficiency test: regress the forecast `errors` on
/// a constant plus `regressors` (typically the forecast itself, optionally with
/// further predetermined signals) with a Bartlett/Newey-West HAC covariance,
/// then jointly test that all coefficients are zero.
///
/// `use_correction` toggles the statsmodels small-sample factor `n/(n - k)`
/// (default `True` there).
///
/// # Errors
///
/// [`SurveyError::EmptyInput`] if `errors` or `regressors` is empty;
/// [`SurveyError::DimensionMismatch`] on a length mismatch;
/// [`SurveyError::NonFinite`] on NaN/inf; [`SurveyError::ConstantResponse`] if
/// the errors do not vary; [`SurveyError::Singular`] if the HAC covariance is
/// not positive definite (collinear regressors); and [`SurveyError::Hac`] /
/// [`SurveyError::Stats`] from the OLS/HAC and chi-square layers.
pub fn efficiency_test(
    errors: &[f64],
    regressors: &[Vec<f64>],
    bandwidth: HacBandwidth,
    use_correction: bool,
) -> Result<EfficiencyTest, SurveyError> {
    if errors.is_empty() {
        return Err(SurveyError::EmptyInput {
            what: "forecast errors",
        });
    }
    if regressors.is_empty() {
        return Err(SurveyError::EmptyInput {
            what: "efficiency-test regressors (supply at least the forecast)",
        });
    }

    let reg = hac_regression(errors, regressors, bandwidth, use_correction)?;
    let k = reg.nparams;

    // Wald quadratic form W = b' V^{-1} b, V the k x k HAC covariance (SPD).
    // Invert V by its Cholesky factor (faer), then form the quadratic product.
    let v = Mat::from_fn(k, k, |i, j| reg.cov[i * k + j]);
    let v_inv = v
        .llt(Side::Lower)
        .map_err(|_| SurveyError::Singular {
            what: "HAC parameter covariance V in the Wald form b' V^{-1} b",
        })?
        .inverse();
    let mut wald = 0.0_f64;
    for i in 0..k {
        for j in 0..k {
            wald += reg.params[i] * v_inv[(i, j)] * reg.params[j];
        }
    }
    if !wald.is_finite() || wald < 0.0 {
        return Err(SurveyError::Singular {
            what: "Wald quadratic form b' V^{-1} b (non-finite or negative)",
        });
    }
    let wald_pvalue = chi2_sf(wald, k as f64)?;

    Ok(EfficiencyTest {
        params: reg.params,
        bse: reg.bse,
        tvalues: reg.tvalues,
        pvalues: reg.pvalues,
        r_squared: reg.r_squared,
        wald,
        wald_df: k,
        wald_pvalue,
        maxlags: reg.maxlags,
        nobs: reg.nobs,
    })
}
