//! Nelson-Siegel (1987) and Svensson (1994) factor loadings.
//!
//! The Nelson-Siegel yield curve writes the zero-coupon yield at maturity `t`
//! as a linear combination of three maturity-dependent loadings driven by a
//! single decay parameter `lambda`:
//!
//! ```text
//! y(t) = beta0 * 1
//!      + beta1 * (1 - e^{-lambda t}) / (lambda t)
//!      + beta2 * [ (1 - e^{-lambda t}) / (lambda t) - e^{-lambda t} ].
//! ```
//!
//! The three loadings are, in order, a **level** factor (constant `1`, loads
//! equally on every maturity — the long rate), a **slope** factor (starts at
//! `1` at the short end and decays monotonically to `0`, so it is essentially
//! minus the yield-curve spread), and a **curvature** factor (a hump that is
//! `0` at both ends and peaks at intermediate maturities). See
//! [`crate`] docs and Diebold & Li (2006).
//!
//! Svensson (1994) adds a *second* curvature term with its own decay rate
//! `lambda2`, giving a four-factor curve that can fit a second hump:
//!
//! ```text
//! y(t) = beta0
//!      + beta1 * (1 - e^{-l1 t}) / (l1 t)
//!      + beta2 * [ (1 - e^{-l1 t}) / (l1 t) - e^{-l1 t} ]
//!      + beta3 * [ (1 - e^{-l2 t}) / (l2 t) - e^{-l2 t} ].
//! ```
//!
//! ## References
//!
//! - Nelson, C. R., & Siegel, A. F. (1987). "Parsimonious Modeling of Yield
//!   Curves." *Journal of Business*, 60(4), 473-489.
//! - Svensson, L. E. O. (1994). "Estimating and Interpreting Forward Interest
//!   Rates: Sweden 1992-1994." NBER Working Paper 4871.
//! - Diebold, F. X., & Li, C. (2006). "Forecasting the Term Structure of
//!   Government Bond Yields." *Journal of Econometrics*, 130(2), 337-364.

use crate::error::TermStructureError;

/// Below this value of `x = lambda * t` the closed-form loadings lose
/// precision to catastrophic cancellation in `1 - e^{-x}`, so the loadings use
/// the analytic `x -> 0` limits instead. At `x = 1e-6` the two agree to well
/// under 1e-12, so the switch is invisible to the golden tolerance.
const SMALL_X: f64 = 1e-6;

/// The Nelson-Siegel slope loading `g(x) = (1 - e^{-x}) / x` with a safe
/// `x -> 0` limit.
///
/// As `x -> 0`, `g(x) -> 1 - x/2 + x^2/6 - ...`; the second-order Taylor
/// expansion is used below [`SMALL_X`] to avoid `0/0`. The loading therefore
/// tends to `1` at the short end (maturity `t -> 0`), which is the defining
/// property that makes `beta1` the yield-curve *slope*.
#[inline]
fn slope_loading(x: f64) -> f64 {
    if x.abs() < SMALL_X {
        // 1 - x/2 + x^2/6 (Taylor of (1 - e^{-x})/x about 0).
        1.0 - 0.5 * x + x * x / 6.0
    } else {
        (1.0 - (-x).exp()) / x
    }
}

/// The Nelson-Siegel curvature loading `h(x) = (1 - e^{-x})/x - e^{-x}`.
///
/// As `x -> 0`, `h(x) -> x/2 - x^2/3 + ...`, so the loading tends to `0` at
/// both the short end (`x -> 0`) and the long end (`x -> inf`), peaking at an
/// intermediate maturity — the hump that makes `beta2` the *curvature*.
#[inline]
fn curvature_loading(x: f64) -> f64 {
    slope_loading(x) - (-x).exp()
}

/// Validate a maturity grid: non-empty, finite, strictly positive.
pub(crate) fn check_maturities(maturities: &[f64]) -> Result<(), TermStructureError> {
    if maturities.is_empty() {
        return Err(TermStructureError::EmptyMaturities);
    }
    for (index, &t) in maturities.iter().enumerate() {
        if !t.is_finite() || t <= 0.0 {
            return Err(TermStructureError::InvalidMaturity { index, value: t });
        }
    }
    Ok(())
}

/// Validate a decay parameter: finite and strictly positive.
pub(crate) fn check_lambda(lambda: f64, what: &'static str) -> Result<(), TermStructureError> {
    if !lambda.is_finite() || lambda <= 0.0 {
        return Err(TermStructureError::InvalidLambda {
            what,
            value: lambda,
        });
    }
    Ok(())
}

/// The three Nelson-Siegel loading columns at decay `lambda`, one entry per
/// maturity.
///
/// Returns `[level, slope, curvature]` where, for each maturity `t` and
/// `x = lambda * t`:
///
/// - `level[i] = 1`,
/// - `slope[i] = (1 - e^{-x}) / x`,
/// - `curvature[i] = (1 - e^{-x}) / x - e^{-x}`.
///
/// A cross-sectional OLS of an observed yield curve on these three columns
/// recovers the `[beta0, beta1, beta2] = [level, slope, curvature]` factors
/// (see [`crate::fit_nelson_siegel`]).
///
/// The `x -> 0` limits are handled analytically (`slope -> 1`,
/// `curvature -> 0`), so the loadings are well defined and continuous as the
/// short maturity shrinks toward zero.
///
/// # Errors
///
/// [`TermStructureError::EmptyMaturities`] for an empty grid,
/// [`TermStructureError::InvalidMaturity`] for a non-positive/non-finite
/// maturity, and [`TermStructureError::InvalidLambda`] for a
/// non-positive/non-finite `lambda`.
///
/// # Example
///
/// ```
/// use tsecon_termstructure::nelson_siegel_loadings;
/// let [level, slope, curv] = nelson_siegel_loadings(&[3.0, 60.0, 120.0], 0.0609).unwrap();
/// assert_eq!(level, vec![1.0, 1.0, 1.0]);
/// assert!(slope[0] > slope[2]); // slope decays with maturity
/// assert!(curv.iter().all(|&c| c >= 0.0));
/// ```
pub fn nelson_siegel_loadings(
    maturities: &[f64],
    lambda: f64,
) -> Result<[Vec<f64>; 3], TermStructureError> {
    check_maturities(maturities)?;
    check_lambda(lambda, "lambda")?;

    let n = maturities.len();
    let mut level = vec![1.0_f64; n];
    let mut slope = vec![0.0_f64; n];
    let mut curvature = vec![0.0_f64; n];
    for (i, &t) in maturities.iter().enumerate() {
        let x = lambda * t;
        level[i] = 1.0;
        slope[i] = slope_loading(x);
        curvature[i] = curvature_loading(x);
    }
    Ok([level, slope, curvature])
}

/// The four Svensson (1994) loading columns, one entry per maturity.
///
/// Returns `[level, slope, curvature1, curvature2]` where the first three
/// columns are the Nelson-Siegel loadings at `lambda1` and the fourth is a
/// second curvature loading `(1 - e^{-l2 t})/(l2 t) - e^{-l2 t}` at an
/// independent decay `lambda2`. The extra term lets the curve fit a second
/// hump that the three-factor model cannot.
///
/// # Errors
///
/// As [`nelson_siegel_loadings`], plus
/// [`TermStructureError::InvalidLambda`] (`"lambda2"`) for a
/// non-positive/non-finite second decay.
pub fn svensson_loadings(
    maturities: &[f64],
    lambda1: f64,
    lambda2: f64,
) -> Result<[Vec<f64>; 4], TermStructureError> {
    check_maturities(maturities)?;
    check_lambda(lambda1, "lambda1")?;
    check_lambda(lambda2, "lambda2")?;

    let [level, slope, curv1] = nelson_siegel_loadings(maturities, lambda1)?;
    let curv2: Vec<f64> = maturities
        .iter()
        .map(|&t| curvature_loading(lambda2 * t))
        .collect();
    Ok([level, slope, curv1, curv2])
}

/// The Nelson-Siegel *instantaneous forward-rate* loadings at decay `lambda`.
///
/// The Nelson-Siegel model is most naturally stated for the instantaneous
/// forward curve, of which the yield curve is the maturity-average:
///
/// ```text
/// f(t) = beta0 + beta1 * e^{-lambda t} + beta2 * (lambda t) * e^{-lambda t}.
/// ```
///
/// Returns `[level, slope, curvature]` forward loadings
/// `[1, e^{-x}, x e^{-x}]` with `x = lambda t`. Used by
/// [`crate::NsFit::forward_at`]; exposed for callers that want the forward
/// decomposition directly.
///
/// # Errors
///
/// As [`nelson_siegel_loadings`].
pub fn nelson_siegel_forward_loadings(
    maturities: &[f64],
    lambda: f64,
) -> Result<[Vec<f64>; 3], TermStructureError> {
    check_maturities(maturities)?;
    check_lambda(lambda, "lambda")?;

    let n = maturities.len();
    let mut level = vec![1.0_f64; n];
    let mut slope = vec![0.0_f64; n];
    let mut curvature = vec![0.0_f64; n];
    for (i, &t) in maturities.iter().enumerate() {
        let x = lambda * t;
        let e = (-x).exp();
        level[i] = 1.0;
        slope[i] = e;
        curvature[i] = x * e;
    }
    Ok([level, slope, curvature])
}
