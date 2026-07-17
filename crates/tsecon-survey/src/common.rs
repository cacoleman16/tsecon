//! Shared plumbing for the two OLS-HAC estimators in this crate (the
//! Coibion-Gorodnichenko regression and the Mincer-Zarnowitz efficiency
//! test).
//!
//! Both fit `y = X b + u` by ordinary least squares with a Bartlett /
//! Newey-West (1987) HAC covariance and report normal-based two-sided
//! p-values — exactly statsmodels `OLS(...).fit(cov_type="HAC",
//! cov_kwds={"maxlags": L, "use_correction": ...}, use_t=False)`. The single
//! owner of the OLS + HAC arithmetic is [`tsecon_hac`]; this module only
//! assembles the design (prepending the intercept column), forwards the
//! bandwidth, and packages the reported quantities.

use tsecon_hac::{newey_west_maxlags, ols, Kernel, SeType};
use tsecon_stats::{ContinuousDist, StdNormal};

use crate::error::SurveyError;

/// The Newey-West lag-truncation bandwidth `L` (statsmodels `maxlags`) used
/// by the Bartlett HAC covariance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HacBandwidth {
    /// An explicit lag truncation `L` (statsmodels `cov_kwds={"maxlags": L}`).
    /// Bartlett weights are `w_j = 1 - j/(L + 1)`, zero for `j > L`.
    Lags(usize),
    /// The ubiquitous rule of thumb `L = floor(4 (n/100)^(2/9))` (via
    /// [`tsecon_hac::newey_west_maxlags`]), evaluated at the regression's
    /// sample size.
    Auto,
}

impl HacBandwidth {
    /// Resolve the bandwidth to a concrete lag count for a sample of size `n`.
    #[must_use]
    pub fn resolve(self, n: usize) -> usize {
        match self {
            HacBandwidth::Lags(l) => l,
            HacBandwidth::Auto => newey_west_maxlags(n),
        }
    }
}

/// A fitted OLS regression with Bartlett-HAC inference — the common payload
/// behind both public estimators.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct HacRegression {
    /// Coefficients: `params[0]` is the intercept, then one per regressor.
    pub params: Vec<f64>,
    /// HAC standard errors, one per parameter.
    pub bse: Vec<f64>,
    /// t-statistics `params / bse`, one per parameter.
    pub tvalues: Vec<f64>,
    /// Two-sided normal p-values `2 * Phi(-|t|)` (statsmodels `use_t=False`).
    pub pvalues: Vec<f64>,
    /// Full HAC parameter covariance, `k x k` row-major.
    pub cov: Vec<f64>,
    /// Centered R-squared `1 - RSS/TSS` (a constant is always included).
    pub r_squared: f64,
    /// The resolved Newey-West lag truncation `L` actually used.
    pub maxlags: usize,
    /// Number of observations `n`.
    pub nobs: usize,
    /// Number of parameters `k` (intercept plus regressors).
    pub nparams: usize,
}

/// Fit `y` on a constant plus `regressors` (given WITHOUT the constant column)
/// by OLS with a Bartlett-HAC covariance.
///
/// Mirrors statsmodels `add_constant` (the intercept is prepended, so it is
/// `params[0]`). All validation of finiteness and degrees of freedom is
/// delegated to [`tsecon_hac::ols`]; this function additionally checks that
/// the response varies (needed for the centered R-squared).
pub(crate) fn hac_regression(
    y: &[f64],
    regressors: &[Vec<f64>],
    bandwidth: HacBandwidth,
    use_correction: bool,
) -> Result<HacRegression, SurveyError> {
    let n = y.len();
    if n == 0 {
        return Err(SurveyError::EmptyInput {
            what: "regression response",
        });
    }
    for col in regressors {
        if col.len() != n {
            return Err(SurveyError::DimensionMismatch {
                what: "regressor column vs response",
                expected: n,
                got: col.len(),
            });
        }
    }
    // tsecon_hac::ols checks finiteness of every column too, but we surface a
    // survey-specific message for the response here for a clearer error.
    check_finite(y, "regression response")?;

    // Design: intercept first (statsmodels exog convention), then regressors.
    let mut design: Vec<Vec<f64>> = Vec::with_capacity(regressors.len() + 1);
    design.push(vec![1.0_f64; n]);
    design.extend(regressors.iter().cloned());

    let fit = ols(y, &design)?;
    let maxlags = bandwidth.resolve(n);
    let inference = fit.inference(SeType::Hac {
        kernel: Kernel::Bartlett,
        bandwidth: maxlags as f64,
        use_correction,
    })?;

    // Two-sided normal p-values (statsmodels reports these when use_t=False,
    // its default for robust covariance types). erfc keeps relative accuracy
    // deep in the tail, so even |t| ~ 11 reproduces to relative 1e-8.
    let normal = StdNormal;
    let pvalues: Vec<f64> = inference
        .tvalues
        .iter()
        .map(|t| 2.0 * normal.sf(t.abs()))
        .collect();

    let r_squared = centered_r_squared(y, &fit.residuals)?;

    Ok(HacRegression {
        params: fit.params,
        bse: inference.bse,
        tvalues: inference.tvalues,
        pvalues,
        cov: inference.cov,
        r_squared,
        maxlags,
        nobs: fit.nobs,
        nparams: fit.nparams,
    })
}

/// Centered coefficient of determination `1 - RSS/TSS`, with
/// `TSS = sum_t (y_t - ybar)^2` — statsmodels `rsquared` for a model that
/// contains a constant.
fn centered_r_squared(y: &[f64], residuals: &[f64]) -> Result<f64, SurveyError> {
    let n = y.len() as f64;
    let mean = y.iter().sum::<f64>() / n;
    let tss: f64 = y.iter().map(|v| (v - mean) * (v - mean)).sum();
    if tss <= 0.0 {
        return Err(SurveyError::ConstantResponse {
            what: "regression response",
        });
    }
    let rss: f64 = residuals.iter().map(|u| u * u).sum();
    Ok(1.0 - rss / tss)
}

/// Reject any NaN/infinite entry in `x`.
pub(crate) fn check_finite(x: &[f64], what: &'static str) -> Result<(), SurveyError> {
    if x.iter().all(|v| v.is_finite()) {
        Ok(())
    } else {
        Err(SurveyError::NonFinite { what })
    }
}
