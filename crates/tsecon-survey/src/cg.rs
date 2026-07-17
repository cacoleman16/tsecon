//! The Coibion-Gorodnichenko (2015) information-rigidity regression.
//!
//! Coibion & Gorodnichenko (2015, *American Economic Review* 105:2644-2678,
//! "Information Rigidity and the Expectations Formation Process") show that in
//! a wide class of imperfect-information models the ex-post mean forecast
//! ERROR is predictable from the ex-ante mean forecast REVISION. Regressing
//!
//! ```text
//! (actual_{t+h} - forecast_t)  =  c  +  beta * (forecast_t - forecast_{t-1})  +  u_t
//!   \___________________/              \_____________________________/
//!      mean forecast error                 mean forecast revision
//! ```
//!
//! by OLS with HAC (Newey-West) inference identifies the degree of information
//! rigidity through the slope `beta`:
//!
//! * under **full-information rational expectations** the error is
//!   unpredictable, so `beta = 0`;
//! * under **sticky information** (Mankiw-Reis) a fraction `lambda` of agents
//!   do not update each period, and `beta = lambda / (1 - lambda) > 0`;
//! * under **noisy information** (Woodford / Sims) with Kalman gain `G`,
//!   `beta = (1 - G) / G > 0`.
//!
//! Both structural maps invert to the same reduced form, so this crate reports
//! the **implied degree of rigidity**
//!
//! ```text
//! implied_rigidity  =  beta / (1 + beta)
//! ```
//!
//! ( `= lambda` under sticky information, `= 1 - G` under noisy information ).
//! It is `0` at `beta = 0` and increases toward `1` as `beta` grows.

use crate::common::{check_finite, hac_regression, HacBandwidth};
use crate::error::SurveyError;

/// Result of the Coibion-Gorodnichenko information-rigidity regression.
#[derive(Debug, Clone, PartialEq)]
pub struct CgRegression {
    /// Intercept `c`.
    pub intercept: f64,
    /// Slope `beta` on the mean forecast revision — the information-rigidity
    /// coefficient (0 under full-information rational expectations, positive
    /// under sticky/noisy information).
    pub slope: f64,
    /// HAC standard error of the intercept.
    pub se_intercept: f64,
    /// HAC standard error of the slope.
    pub se_slope: f64,
    /// t-statistic of the intercept, `intercept / se_intercept`.
    pub t_intercept: f64,
    /// t-statistic of the slope, `slope / se_slope`.
    pub t_slope: f64,
    /// Two-sided normal p-value of the intercept.
    pub p_intercept: f64,
    /// Two-sided normal p-value of the slope (the test of `H0: beta = 0`,
    /// i.e. full-information rational expectations).
    pub p_slope: f64,
    /// Centered R-squared of the regression.
    pub r_squared: f64,
    /// The implied degree of information rigidity `beta / (1 + beta)`.
    pub implied_rigidity: f64,
    /// The Newey-West lag truncation `L` actually used.
    pub maxlags: usize,
    /// Number of observations in the regression.
    pub nobs: usize,
}

/// Run the Coibion-Gorodnichenko (2015) regression of the mean forecast
/// `errors` on the mean forecast `revisions`, with a Bartlett/Newey-West HAC
/// covariance.
///
/// The two series must be pre-aligned and equal length: `errors[t]` is the
/// realized error of the forecast whose one-period revision is `revisions[t]`.
/// See [`cg_series`] for a convenience that builds them from a fixed-horizon
/// mean-forecast series and the realized actual.
///
/// `use_correction` toggles the statsmodels small-sample factor `n/(n - k)` on
/// the covariance (statsmodels `cov_kwds={"use_correction": ...}`; its default
/// there is `True`).
///
/// # Errors
///
/// [`SurveyError::EmptyInput`] on empty input;
/// [`SurveyError::DimensionMismatch`] if the lengths differ;
/// [`SurveyError::NonFinite`] on NaN/inf; [`SurveyError::ConstantResponse`] if
/// the errors do not vary; and [`SurveyError::Hac`] from the OLS/HAC layer
/// (e.g. too few observations, or a degenerate revision series).
pub fn cg_regression(
    errors: &[f64],
    revisions: &[f64],
    bandwidth: HacBandwidth,
    use_correction: bool,
) -> Result<CgRegression, SurveyError> {
    if errors.is_empty() {
        return Err(SurveyError::EmptyInput {
            what: "mean forecast errors",
        });
    }
    if revisions.len() != errors.len() {
        return Err(SurveyError::DimensionMismatch {
            what: "revisions vs errors",
            expected: errors.len(),
            got: revisions.len(),
        });
    }

    let reg = hac_regression(errors, &[revisions.to_vec()], bandwidth, use_correction)?;
    let slope = reg.params[1];
    let implied_rigidity = slope / (1.0 + slope);

    Ok(CgRegression {
        intercept: reg.params[0],
        slope,
        se_intercept: reg.bse[0],
        se_slope: reg.bse[1],
        t_intercept: reg.tvalues[0],
        t_slope: reg.tvalues[1],
        p_intercept: reg.pvalues[0],
        p_slope: reg.pvalues[1],
        r_squared: reg.r_squared,
        implied_rigidity,
        maxlags: reg.maxlags,
        nobs: reg.nobs,
    })
}

/// Build the aligned CG error and revision series from a fixed-horizon
/// mean-forecast series and the realized actual.
///
/// Convention (fixed forecast horizon `h`): `mean_forecast[t]` is the
/// `h`-step-ahead mean forecast made at time `t` of the outcome
/// `actual[t + h]`. For every usable `t` (with `t - 1 >= 0` and
/// `t + h <= n - 1`) the aligned pair is
///
/// ```text
/// error_t    = actual[t + h] - mean_forecast[t]
/// revision_t = mean_forecast[t] - mean_forecast[t - 1]
/// ```
///
/// so the returned vectors run over `t = 1 ..= n - 1 - h` and have length
/// `n - 1 - h`. Feed them to [`cg_regression`].
///
/// # Errors
///
/// [`SurveyError::EmptyInput`] on empty input;
/// [`SurveyError::DimensionMismatch`] if the two series differ in length;
/// [`SurveyError::NonFinite`] on NaN/inf; and [`SurveyError::SeriesTooShort`]
/// if `n < h + 2` (no usable aligned pair exists).
pub fn cg_series(
    mean_forecast: &[f64],
    actual: &[f64],
    h: usize,
) -> Result<(Vec<f64>, Vec<f64>), SurveyError> {
    if mean_forecast.is_empty() {
        return Err(SurveyError::EmptyInput {
            what: "mean_forecast",
        });
    }
    if actual.len() != mean_forecast.len() {
        return Err(SurveyError::DimensionMismatch {
            what: "actual vs mean_forecast",
            expected: mean_forecast.len(),
            got: actual.len(),
        });
    }
    check_finite(mean_forecast, "mean_forecast")?;
    check_finite(actual, "actual")?;

    let n = mean_forecast.len();
    // Need t in [1, n-1-h]: smallest usable n is h + 2 (t = 1 with t+h = h+1).
    if n < h + 2 {
        return Err(SurveyError::SeriesTooShort {
            what: "mean_forecast/actual for the requested horizon h",
            got: n,
            need: h + 2,
        });
    }

    let count = n - 1 - h;
    let mut errors = Vec::with_capacity(count);
    let mut revisions = Vec::with_capacity(count);
    for t in 1..=(n - 1 - h) {
        errors.push(actual[t + h] - mean_forecast[t]);
        revisions.push(mean_forecast[t] - mean_forecast[t - 1]);
    }
    Ok((errors, revisions))
}
