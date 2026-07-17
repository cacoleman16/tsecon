//! Jarque-Bera test for normality.

use tsecon_stats::chi2_sf;

use crate::error::DiagError;
use crate::report::{check_alpha, DiagnosticReport};
use crate::validate::check_series;

/// Result of the Jarque-Bera normality test.
#[derive(Debug, Clone, PartialEq)]
pub struct JarqueBeraResult {
    /// The Jarque-Bera statistic `JB = n/6 * (S^2 + (K - 3)^2 / 4)`.
    pub statistic: f64,
    /// Chi-squared p-value with 2 degrees of freedom.
    pub p_value: f64,
    /// Sample skewness `S = m_3 / m_2^{3/2}` (0 under normality).
    pub skewness: f64,
    /// Sample kurtosis `K = m_4 / m_2^2`, *not* excess (3 under normality).
    pub kurtosis: f64,
    /// Number of observations.
    pub n: usize,
}

/// Jarque-Bera test of the null that the data are normally distributed
/// (Jarque & Bera 1980).
///
/// With `m_j = (1/n) sum (x_t - x_bar)^j` the central sample moments
/// (biased, `n` denominators — the statsmodels
/// `sm.stats.stattools.jarque_bera` convention, no small-sample
/// adjustment),
///
/// ```text
/// S  = m_3 / m_2^{3/2}
/// K  = m_4 / m_2^2                (raw kurtosis; normal value 3)
/// JB = n/6 * (S^2 + (K - 3)^2/4)  ~  chi^2(2)  under the null
/// ```
///
/// The asymptotic chi-squared approximation is famously poor in small
/// samples (over-rejects); treat borderline p-values with caution below a
/// few hundred observations.
// TODO(phase0): simulated small-sample p-values and the Urzua (1996) ALM
// correction.
///
/// # Errors
///
/// * [`DiagError::NonFinite`] if the data contain NaN or infinities.
/// * [`DiagError::SeriesTooShort`] if `n < 2`.
/// * [`DiagError::ConstantSeries`] if the sample variance is zero.
pub fn jarque_bera(x: &[f64]) -> Result<JarqueBeraResult, DiagError> {
    let n = check_series(x, 2, "jarque_bera")?;
    let nf = n as f64;
    let mean = x.iter().sum::<f64>() / nf;
    let mut m2 = 0.0_f64;
    let mut m3 = 0.0_f64;
    let mut m4 = 0.0_f64;
    for &v in x {
        let d = v - mean;
        let d2 = d * d;
        m2 += d2;
        m3 += d2 * d;
        m4 += d2 * d2;
    }
    m2 /= nf;
    m3 /= nf;
    m4 /= nf;
    if m2 <= 0.0 {
        return Err(DiagError::ConstantSeries {
            what: "jarque_bera",
        });
    }
    let skewness = m3 / (m2 * m2.sqrt());
    let kurtosis = m4 / (m2 * m2);
    let excess = kurtosis - 3.0;
    let statistic = nf / 6.0 * (skewness * skewness + excess * excess / 4.0);
    let p_value = chi2_sf(statistic, 2.0)?;
    Ok(JarqueBeraResult {
        statistic,
        p_value,
        skewness,
        kurtosis,
        n,
    })
}

impl JarqueBeraResult {
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
                "Reject normality at the {:.0}% level (skewness {:.3} vs 0, \
                 kurtosis {:.3} vs 3 under the null). Point estimates remain \
                 consistent, but Gaussian-likelihood standard errors and \
                 forecast intervals are unreliable — consider Student-t \
                 innovations or QMLE-robust standard errors. Heavy tails \
                 often signal volatility clustering: run `arch_lm` next.",
                alpha * 100.0,
                self.skewness,
                self.kurtosis
            )
        } else {
            format!(
                "No evidence against normality at the {:.0}% level (skewness \
                 {:.3}, kurtosis {:.3}). Gaussian-based inference and \
                 interval forecasts are defensible; remember the chi-squared \
                 approximation over-rejects in small samples, so a \
                 non-rejection at n = {} is reassuring but not proof.",
                alpha * 100.0,
                self.skewness,
                self.kurtosis,
                self.n
            )
        };
        Ok(DiagnosticReport {
            test: "Jarque-Bera".to_string(),
            statistic: self.statistic,
            p_value: self.p_value,
            df: 2,
            alpha,
            reject,
            interpretation,
        })
    }
}
