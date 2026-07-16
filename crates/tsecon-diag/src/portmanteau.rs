//! Ljung-Box and Box-Pierce portmanteau tests for autocorrelation.

use tsecon_stats::chi2_sf;

use crate::acf::autocorrelations;
use crate::error::DiagError;
use crate::report::{check_alpha, DiagnosticReport};
use crate::validate::check_series;

/// Ljung-Box and Box-Pierce statistics for every lag `1..=nlags`.
///
/// All vectors have length `nlags`; entry `i` corresponds to
/// `lags[i] = i + 1`.
#[derive(Debug, Clone, PartialEq)]
pub struct PortmanteauResult {
    /// The lags `1, 2, ..., nlags` the cumulative statistics are reported
    /// at.
    pub lags: Vec<usize>,
    /// Ljung-Box statistics `Q_LB(h)` for `h = 1..=nlags`.
    pub lb_stat: Vec<f64>,
    /// Chi-squared p-values of the Ljung-Box statistics, `df = h`.
    pub lb_pvalue: Vec<f64>,
    /// Box-Pierce statistics `Q_BP(h)` for `h = 1..=nlags`.
    pub bp_stat: Vec<f64>,
    /// Chi-squared p-values of the Box-Pierce statistics, `df = h`.
    pub bp_pvalue: Vec<f64>,
    /// Number of observations the statistics were computed from.
    pub n: usize,
}

/// Ljung-Box (and Box-Pierce) portmanteau tests of the null that the
/// series is white noise, evaluated at every lag `h = 1..=nlags`.
///
/// With `r_k` the unadjusted sample autocorrelations (see [`crate::acf`]),
///
/// ```text
/// Q_LB(h) = n (n + 2) * sum_{k=1}^{h} r_k^2 / (n - k)     (Ljung-Box 1978)
/// Q_BP(h) = n * sum_{k=1}^{h} r_k^2                       (Box-Pierce 1970)
/// ```
///
/// each compared against a chi-squared distribution with `h` degrees of
/// freedom. This matches statsmodels `acorr_ljungbox(y, lags=range(1,
/// nlags+1), boxpierce=True)` with default settings.
///
/// The Ljung-Box finite-sample correction `(n+2)/(n-k) > 1` implies
/// `Q_LB(h) >= Q_BP(h)`, and both statistics are nondecreasing in `h`.
///
/// Applied to the residuals of a fitted ARMA(p, q) model the correct
/// degrees of freedom are `h - p - q`; this first slice always uses
/// `df = h` (raw-series mode — the statsmodels default).
// TODO(phase0): add the model-degrees-of-freedom adjustment (`model_df`)
// once the univariate estimators exist.
///
/// # Errors
///
/// As [`crate::acf`] (the statistics require `1 <= nlags <= n - 1`).
pub fn ljung_box(y: &[f64], nlags: usize) -> Result<PortmanteauResult, DiagError> {
    let n = check_series(y, 2, "ljung_box")?;
    if nlags == 0 || nlags > n - 1 {
        return Err(DiagError::InvalidLags {
            what: "ljung_box",
            nlags,
            n,
            requirement: "1 <= nlags <= n - 1",
        });
    }
    let r = autocorrelations(y, nlags, false, "ljung_box")?;
    let nf = n as f64;

    let mut lags = Vec::with_capacity(nlags);
    let mut lb_stat = Vec::with_capacity(nlags);
    let mut lb_pvalue = Vec::with_capacity(nlags);
    let mut bp_stat = Vec::with_capacity(nlags);
    let mut bp_pvalue = Vec::with_capacity(nlags);

    let mut lb_acc = 0.0_f64;
    let mut bp_acc = 0.0_f64;
    for (h, &rk) in r.iter().enumerate().skip(1) {
        let rk2 = rk * rk;
        lb_acc += rk2 / (nf - h as f64);
        bp_acc += rk2;
        let lb = nf * (nf + 2.0) * lb_acc;
        let bp = nf * bp_acc;
        lags.push(h);
        lb_stat.push(lb);
        lb_pvalue.push(chi2_sf(lb, h as f64)?);
        bp_stat.push(bp);
        bp_pvalue.push(chi2_sf(bp, h as f64)?);
    }

    Ok(PortmanteauResult {
        lags,
        lb_stat,
        lb_pvalue,
        bp_stat,
        bp_pvalue,
        n,
    })
}

impl PortmanteauResult {
    /// Summarize the Ljung-Box test at the largest computed lag as a
    /// [`DiagnosticReport`], rejecting at significance level `alpha`.
    ///
    /// # Errors
    ///
    /// [`DiagError::InvalidAlpha`] unless `0 < alpha < 1`.
    pub fn report(&self, alpha: f64) -> Result<DiagnosticReport, DiagError> {
        check_alpha(alpha)?;
        // ljung_box always produces at least one lag, so the vectors are
        // nonempty by construction.
        let idx = self.lags.len() - 1;
        let h = self.lags[idx];
        let statistic = self.lb_stat[idx];
        let p_value = self.lb_pvalue[idx];
        let reject = p_value < alpha;
        let interpretation = if reject {
            format!(
                "Reject whiteness at the {:.0}% level: significant \
                 autocorrelation remains through lag {h}. If these are model \
                 residuals, the model has not captured all linear dependence \
                 — inspect the residual ACF/PACF (`acf`, `pacf_yw`) to see \
                 which lags misbehave and consider a richer AR/MA \
                 specification. Note: on ARMA(p, q) residuals the correct \
                 degrees of freedom are h - p - q, which this test does not \
                 yet subtract, making it mildly conservative there.",
                alpha * 100.0
            )
        } else {
            format!(
                "No evidence against whiteness through lag {h} at the {:.0}% \
                 level: the autocorrelations are jointly consistent with \
                 white noise. Linear dependence looks exhausted — next check \
                 higher moments: `jarque_bera` for normality and `arch_lm` \
                 for conditional heteroskedasticity, which this test cannot \
                 see.",
                alpha * 100.0
            )
        };
        Ok(DiagnosticReport {
            test: format!("Ljung-Box Q({h})"),
            statistic,
            p_value,
            df: h,
            alpha,
            reject,
            interpretation,
        })
    }
}
