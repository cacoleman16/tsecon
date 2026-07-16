//! Engle's ARCH-LM test for conditional heteroskedasticity.

use tsecon_stats::chi2_sf;

use crate::error::DiagError;
use crate::report::{check_alpha, DiagnosticReport};
use crate::validate::check_series;

/// Result of Engle's ARCH-LM test.
#[derive(Debug, Clone, PartialEq)]
pub struct ArchLmResult {
    /// The LM statistic `nobs * R^2` from the auxiliary regression.
    pub statistic: f64,
    /// Chi-squared p-value with `df = nlags` degrees of freedom.
    pub p_value: f64,
    /// Degrees of freedom (= the number of lags in the auxiliary
    /// regression).
    pub df: usize,
    /// Number of observations in the auxiliary regression (`n - nlags`).
    pub nobs: usize,
}

/// Engle's LM test for autoregressive conditional heteroskedasticity
/// (Engle 1982), the gateway diagnostic before GARCH modeling.
///
/// Regress the squared residuals on an intercept and their own `nlags`
/// lags over the `nobs = n - nlags` usable rows:
///
/// ```text
/// e_t^2 = a_0 + a_1 e_{t-1}^2 + ... + a_m e_{t-m}^2 + u_t
/// LM    = nobs * R^2  ~  chi^2(m)  under the no-ARCH null
/// ```
///
/// This matches statsmodels `het_arch(resid, nlags=m)` (LM variant,
/// `ddof=0`).
///
/// The test is designed for raw model residuals; applied to GARCH-
/// *standardized* residuals its asymptotic distribution is wrong — use a
/// Li-Mak test there (later phase).
///
/// # Errors
///
/// * [`DiagError::NonFinite`] if the residuals contain NaN or infinities.
/// * [`DiagError::InvalidLags`] if `nlags == 0`.
/// * [`DiagError::SeriesTooShort`] unless `n >= 2 * nlags + 2` (the
///   auxiliary regression needs more rows than coefficients).
/// * [`DiagError::ConstantSeries`] /
///   [`DiagError::SingularDesign`] for degenerate squared residuals.
pub fn arch_lm(resid: &[f64], nlags: usize) -> Result<ArchLmResult, DiagError> {
    let n = check_series(resid, 2, "arch_lm")?;
    if nlags == 0 {
        return Err(DiagError::InvalidLags {
            what: "arch_lm",
            nlags,
            n,
            requirement: "nlags >= 1",
        });
    }
    // The auxiliary regression has n - nlags rows and nlags + 1
    // coefficients; require at least one residual degree of freedom.
    if n < 2 * nlags + 2 {
        return Err(DiagError::SeriesTooShort {
            what: "arch_lm",
            n,
            needed: 2 * nlags + 2,
        });
    }

    let e2: Vec<f64> = resid.iter().map(|&e| e * e).collect();
    // Rows t = nlags..n-1; column j holds e2_{t-j} for j = 1..=nlags.
    let cols: Vec<Vec<f64>> = (1..=nlags)
        .map(|j| (nlags..n).map(|t| e2[t - j]).collect())
        .collect();
    let response = &e2[nlags..];
    let fit = crate::ols::ols_with_intercept(&cols, response, "arch_lm")?;

    let nobs = n - nlags;
    let statistic = nobs as f64 * fit.r_squared;
    let p_value = chi2_sf(statistic, nlags as f64)?;
    Ok(ArchLmResult {
        statistic,
        p_value,
        df: nlags,
        nobs,
    })
}

impl ArchLmResult {
    /// Summarize the test as a [`DiagnosticReport`], rejecting at
    /// significance level `alpha`.
    ///
    /// # Errors
    ///
    /// [`DiagError::InvalidAlpha`] unless `0 < alpha < 1`.
    pub fn report(&self, alpha: f64) -> Result<DiagnosticReport, DiagError> {
        check_alpha(alpha)?;
        let reject = self.p_value < alpha;
        let interpretation = if reject {
            format!(
                "Reject homoskedasticity at the {:.0}% level: the squared \
                 residuals are autocorrelated through lag {} — volatility \
                 clusters. Standard errors that assume constant variance are \
                 unreliable; model the conditional variance with a \
                 GARCH-family specification (or at minimum use \
                 heteroskedasticity-robust standard errors). This test says \
                 nothing about the mean equation — keep the Ljung-Box result \
                 in view too.",
                alpha * 100.0,
                self.df
            )
        } else {
            format!(
                "No evidence of ARCH effects through lag {} at the {:.0}% \
                 level: squared residuals look serially uncorrelated, so a \
                 constant-variance model is defensible. If the data are \
                 low-frequency this is common; at daily or higher frequency \
                 a non-rejection is worth re-checking with more lags.",
                self.df,
                alpha * 100.0
            )
        };
        Ok(DiagnosticReport {
            test: format!("ARCH-LM({})", self.df),
            statistic: self.statistic,
            p_value: self.p_value,
            df: self.df,
            alpha,
            reject,
            interpretation,
        })
    }
}
