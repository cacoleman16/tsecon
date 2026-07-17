//! The Heterogeneous Autoregressive model of realized volatility (HAR-RV,
//! Corsi 2009): a parsimonious long-memory-mimicking regression of realized
//! variance on its own daily, weekly, and monthly averages.
//!
//! With `RV_t` the daily realized variance, define the lagged aggregates
//! known at the end of day `t-1`
//!
//! ```text
//!   RV^d_{t-1} = RV_{t-1}
//!   RV^w_{t-1} = (1/5)  (RV_{t-2} + ... + RV_{t-6})
//!   RV^m_{t-1} = (1/22) (RV_{t-2} + ... + RV_{t-23})
//! ```
//!
//! and run the ordinary-least-squares regression
//!
//! ```text
//!   RV_t = c + beta_d RV^d_{t-1} + beta_w RV^w_{t-1} + beta_m RV^m_{t-1} + e_t.
//! ```
//!
//! Equivalently, writing the weekly/monthly averages as the trailing means
//! `mean(RV[t-6..t-1])` and `mean(RV[t-23..t-1])` (Python-style half-open
//! windows ending just before the daily lag). Because the regressors are
//! overlapping moving averages the errors are serially correlated, so the
//! standard errors are the HAC (Newey-West / Bartlett) sandwich errors
//! delegated to [`tsecon_hac`]; the library never reimplements HAC
//! (ROADMAP: one owner per capability).
//!
//! The log and square-root variants (Corsi 2009 discusses both, since `RV`
//! is strongly right-skewed) apply the transform to the RV series *before*
//! forming the regressors, i.e. they estimate the HAR on `ln RV` or
//! `sqrt(RV)`; select them with [`HarVariant`].

use crate::error::RealizedError;
use tsecon_hac::{ols, Kernel, SeType};

/// Which transform of realized variance the HAR is estimated on. `RV` is
/// strongly right-skewed, so log- and sqrt-HAR are common (Corsi 2009); the
/// regressor construction is identical after transforming the series.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HarVariant {
    /// Regress `RV_t` on its lagged daily/weekly/monthly averages.
    Level,
    /// Regress `ln RV_t` on the lagged averages of `ln RV` (requires
    /// strictly positive `RV`).
    Log,
    /// Regress `sqrt(RV_t)` on the lagged averages of `sqrt(RV)` (requires
    /// non-negative `RV`).
    Sqrt,
}

impl HarVariant {
    fn apply(self, rv: &[f64]) -> Result<Vec<f64>, RealizedError> {
        let mut out = Vec::with_capacity(rv.len());
        for (index, &v) in rv.iter().enumerate() {
            if !v.is_finite() {
                return Err(RealizedError::NonFinite {
                    what: "HAR realized-variance series",
                    index,
                    value: v,
                });
            }
            let t = match self {
                HarVariant::Level => v,
                HarVariant::Log => {
                    if v <= 0.0 {
                        return Err(RealizedError::InvalidOhlc {
                            what: "log-HAR",
                            index,
                            detail: "realized variance must be strictly positive to take a log",
                        });
                    }
                    v.ln()
                }
                HarVariant::Sqrt => {
                    if v < 0.0 {
                        return Err(RealizedError::InvalidOhlc {
                            what: "sqrt-HAR",
                            index,
                            detail: "realized variance must be non-negative to take a square root",
                        });
                    }
                    v.sqrt()
                }
            };
            out.push(t);
        }
        Ok(out)
    }
}

/// Estimation settings for [`har_rv`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HarConfig {
    /// Burn-in index: the first target row is `t = start + 1`. Must be
    /// `>= 22` so the monthly regressor `mean(RV[t-23..t-1])` is defined.
    /// The library-standard (and fixture) value is `22`.
    pub start: usize,
    /// Which RV transform to regress (level / log / sqrt).
    pub variant: HarVariant,
    /// Bartlett-kernel HAC lag truncation (statsmodels `maxlags`). The
    /// fixture uses `5`, matching the weekly aggregation horizon.
    pub hac_maxlags: usize,
    /// Apply the statsmodels `n/(n-k)` small-sample correction to the HAC
    /// covariance. The fixture was generated with this **off**
    /// (`use_correction=False`).
    pub use_correction: bool,
}

impl Default for HarConfig {
    fn default() -> Self {
        HarConfig {
            start: 22,
            variant: HarVariant::Level,
            hac_maxlags: 5,
            use_correction: false,
        }
    }
}

/// A fitted HAR-RV regression.
///
/// Coefficients are ordered `[const, beta_d, beta_w, beta_m]`. On a
/// persistent series `beta_d + beta_w + beta_m` is close to the RV
/// autoregressive persistence (it is the model's implied one-step
/// sensitivity of `RV_t` to a uniform shift in all lagged averages).
#[derive(Debug, Clone, PartialEq)]
pub struct HarFit {
    /// `[const, beta_d, beta_w, beta_m]`.
    pub params: Vec<f64>,
    /// HAC standard errors, aligned with [`HarFit::params`].
    pub bse: Vec<f64>,
    /// HAC t-statistics, aligned with [`HarFit::params`].
    pub tvalues: Vec<f64>,
    /// Centered coefficient of determination
    /// `R^2 = 1 - RSS/TSS` (the design includes a constant).
    pub rsquared: f64,
    /// Number of target rows in the estimation sample.
    pub nobs: usize,
}

/// Fit the Corsi (2009) HAR-RV model on a realized-variance series.
///
/// Builds the `[const, RV_{t-1}, RV^w_{t-1}, RV^m_{t-1}]` design over the
/// sample `t = start+1 .. n-1` (see the module docs for the exact window
/// definitions), fits it by OLS via [`tsecon_hac::ols`], and reports
/// Bartlett HAC standard errors matching statsmodels
/// `cov_type="HAC", cov_kwds={"maxlags": hac_maxlags, "use_correction": ...}`.
///
/// # Errors
///
/// [`RealizedError::InsufficientHarSample`] if `start < 22` or the series
/// is too short to leave one target row; [`RealizedError::NonFinite`] /
/// [`RealizedError::InvalidOhlc`] from the (log/sqrt) transform; and
/// [`RealizedError::Hac`] wrapping any collinear-design or degrees-of-freedom
/// failure from the delegated OLS/HAC solve.
pub fn har_rv(rv: &[f64], config: &HarConfig) -> Result<HarFit, RealizedError> {
    let n = rv.len();
    // Need start >= 22 (monthly window) and at least one target row
    // t = start+1 <= n-1, i.e. n >= start + 2.
    if config.start < 22 || n < config.start + 2 {
        return Err(RealizedError::InsufficientHarSample {
            start: config.start,
            n,
        });
    }

    let s = config.variant.apply(rv)?;

    let first = config.start + 1;
    let rows = n - first; // targets t = first ..= n-1
    let mut y = Vec::with_capacity(rows);
    let mut daily = Vec::with_capacity(rows);
    let mut weekly = Vec::with_capacity(rows);
    let mut monthly = Vec::with_capacity(rows);
    for t in first..n {
        y.push(s[t]);
        daily.push(s[t - 1]);
        // Trailing 5-day mean ending at t-2: indices t-6 ..= t-2.
        let w: f64 = s[t - 6..t - 1].iter().sum::<f64>() / 5.0;
        weekly.push(w);
        // Trailing 22-day mean ending at t-2: indices t-23 ..= t-2.
        let m: f64 = s[t - 23..t - 1].iter().sum::<f64>() / 22.0;
        monthly.push(m);
    }

    let constant = vec![1.0; rows];
    let design = vec![constant, daily, weekly, monthly];
    let fit = ols(&y, &design)?;
    let inference = fit.inference(SeType::Hac {
        kernel: Kernel::Bartlett,
        bandwidth: config.hac_maxlags as f64,
        use_correction: config.use_correction,
    })?;

    let rsquared = centered_r_squared(&y, &fit.residuals);

    Ok(HarFit {
        params: fit.params,
        bse: inference.bse,
        tvalues: inference.tvalues,
        rsquared,
        nobs: rows,
    })
}

/// Centered `R^2 = 1 - RSS/TSS` with `TSS = sum (y - ybar)^2`, matching
/// statsmodels' reported `rsquared` for a model that includes a constant.
fn centered_r_squared(y: &[f64], residuals: &[f64]) -> f64 {
    let n = y.len() as f64;
    let ybar = y.iter().sum::<f64>() / n;
    let tss: f64 = y.iter().map(|&v| (v - ybar) * (v - ybar)).sum();
    let rss: f64 = residuals.iter().map(|&e| e * e).sum();
    1.0 - rss / tss
}
