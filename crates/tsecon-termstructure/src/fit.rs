//! Cross-sectional Nelson-Siegel fit at a fixed decay `lambda`, plus the
//! fitted-yield and implied-forward curve utilities.

use crate::error::TermStructureError;
use crate::loadings::{
    check_lambda, check_maturities, nelson_siegel_forward_loadings, nelson_siegel_loadings,
};
use tsecon_hac::{ols, HacError};

/// Check a slice of observations for NaN/inf, reporting the first offender.
pub(crate) fn check_finite(xs: &[f64], what: &'static str) -> Result<(), TermStructureError> {
    for (index, &v) in xs.iter().enumerate() {
        if !v.is_finite() {
            return Err(TermStructureError::NonFinite {
                what,
                index,
                value: v,
            });
        }
    }
    Ok(())
}

/// The centered coefficient of determination,
/// `R^2 = 1 - sum(resid^2) / sum((y - mean(y))^2)`.
///
/// This is the R^2 statsmodels reports for a regression that includes a
/// constant (the Nelson-Siegel level loading is a constant column), so it
/// matches the golden `ns_fit_rsquared`. Returns `NaN` when the yields are
/// exactly constant (`sum((y - mean)^2) = 0`), for which R^2 is undefined.
pub(crate) fn centered_rsquared(y: &[f64], residuals: &[f64]) -> f64 {
    let n = y.len() as f64;
    let mean = y.iter().sum::<f64>() / n;
    let sst: f64 = y.iter().map(|&yi| (yi - mean).powi(2)).sum();
    let ssr: f64 = residuals.iter().map(|&r| r * r).sum();
    if sst > 0.0 {
        1.0 - ssr / sst
    } else {
        f64::NAN
    }
}

/// Map a cross-sectional OLS failure to a term-structure error. The
/// dimension/finiteness cases are pre-validated by the callers, so in practice
/// only the singular-design case is reachable.
pub(crate) fn map_ols_err(err: HacError, what: &'static str) -> TermStructureError {
    match err {
        HacError::SingularDesign { .. } | HacError::DegreesOfFreedom { .. } => {
            TermStructureError::SingularDesign { what }
        }
        HacError::NonFinite { what, index, value } => {
            TermStructureError::NonFinite { what, index, value }
        }
        _ => TermStructureError::SingularDesign { what },
    }
}

/// Solve the cross-sectional least-squares problem `yields ~ loadings` and
/// return `(factors, residuals, rsquared)`. Shared by the Nelson-Siegel and
/// Svensson fits.
pub(crate) fn fit_columns(
    yields: &[f64],
    columns: &[Vec<f64>],
    what: &'static str,
) -> Result<(Vec<f64>, Vec<f64>, f64), TermStructureError> {
    let fit = ols(yields, columns).map_err(|e| map_ols_err(e, what))?;
    let r2 = centered_rsquared(yields, &fit.residuals);
    Ok((fit.params, fit.residuals, r2))
}

/// A fitted three-factor Nelson-Siegel yield curve at a fixed decay `lambda`.
///
/// Produced by [`fit_nelson_siegel`] and
/// [`crate::fit_nelson_siegel_optimal_lambda`]. Holds the recovered
/// `[level, slope, curvature]` factors, the decay `lambda` at which they were
/// estimated, the cross-sectional fit residuals, and the centered R^2.
#[derive(Debug, Clone, PartialEq)]
pub struct NsFit {
    /// The recovered factors `[beta0, beta1, beta2] = [level, slope,
    /// curvature]`.
    ///
    /// - `level` (`beta0`) is the long-rate / overall height of the curve;
    ///   every maturity loads on it equally.
    /// - `slope` (`beta1`) governs the short-to-long spread; its loading is
    ///   `1` at the short end and decays to `0`, so a positive `beta1` lifts
    ///   the short end (an inverting curve). Minus `beta1` is the usual
    ///   long-minus-short slope.
    /// - `curvature` (`beta2`) is the medium-term hump; its loading is `0` at
    ///   both ends and peaks at intermediate maturities.
    pub factors: [f64; 3],
    /// The decay parameter the loadings were evaluated at.
    pub lambda: f64,
    /// Cross-sectional residuals `y_i - yhat_i`, one per input maturity.
    pub residuals: Vec<f64>,
    /// Centered R^2 of the cross-sectional fit (matches statsmodels'
    /// constant-included R^2).
    pub rsquared: f64,
}

impl NsFit {
    /// The fitted yield at an arbitrary maturity `t`,
    /// `yhat(t) = level + slope * g(lt) + curvature * h(lt)` with
    /// `g(x) = (1 - e^{-x})/x` and `h(x) = g(x) - e^{-x}`.
    ///
    /// # Errors
    ///
    /// [`TermStructureError::InvalidMaturity`] for a non-positive/non-finite
    /// maturity.
    pub fn yield_at(&self, maturity: f64) -> Result<f64, TermStructureError> {
        let [level, slope, curv] = nelson_siegel_loadings(&[maturity], self.lambda)?;
        Ok(self.factors[0] * level[0] + self.factors[1] * slope[0] + self.factors[2] * curv[0])
    }

    /// The fitted instantaneous forward rate at maturity `t`,
    /// `f(t) = level + slope * e^{-lt} + curvature * (lt) e^{-lt}`.
    ///
    /// The Nelson-Siegel yield is the maturity-average of this forward curve;
    /// at the short end `f(0) = level + slope` equals the fitted short yield,
    /// and as `t -> inf` both the yield and the forward tend to `level`. See
    /// [`crate::nelson_siegel_forward_loadings`].
    ///
    /// # Errors
    ///
    /// [`TermStructureError::InvalidMaturity`] for a non-positive/non-finite
    /// maturity.
    pub fn forward_at(&self, maturity: f64) -> Result<f64, TermStructureError> {
        let [level, slope, curv] = nelson_siegel_forward_loadings(&[maturity], self.lambda)?;
        Ok(self.factors[0] * level[0] + self.factors[1] * slope[0] + self.factors[2] * curv[0])
    }

    /// The fitted yields on a maturity grid (one `yield_at` per maturity).
    ///
    /// # Errors
    ///
    /// As [`NsFit::yield_at`].
    pub fn fitted(&self, maturities: &[f64]) -> Result<Vec<f64>, TermStructureError> {
        let [level, slope, curv] = nelson_siegel_loadings(maturities, self.lambda)?;
        Ok((0..maturities.len())
            .map(|i| {
                self.factors[0] * level[i] + self.factors[1] * slope[i] + self.factors[2] * curv[i]
            })
            .collect())
    }
}

/// Fit a single yield curve by cross-sectional OLS on the Nelson-Siegel
/// loadings at a **fixed** decay `lambda` (Diebold & Li 2006).
///
/// With `lambda` held fixed the model is linear in the three factors, so the
/// fit is an ordinary least-squares regression of the observed yields on the
/// `[level, slope, curvature]` loading columns. This is exactly the Diebold-Li
/// two-step convention: fix `lambda` (they use `0.0609` for monthly
/// maturities), then read the factors off an OLS. Other authors instead
/// *estimate* `lambda` per curve by nonlinear least squares — see
/// [`crate::fit_nelson_siegel_optimal_lambda`].
///
/// # Errors
///
/// Validation errors from [`nelson_siegel_loadings`]
/// ([`TermStructureError::EmptyMaturities`],
/// [`TermStructureError::InvalidMaturity`],
/// [`TermStructureError::InvalidLambda`]);
/// [`TermStructureError::DimensionMismatch`] if `yields.len()` differs from
/// `maturities.len()`; [`TermStructureError::Underdetermined`] with fewer than
/// four maturities (three factors need strictly more than three pricing
/// equations); [`TermStructureError::NonFinite`] on NaN/inf yields; and
/// [`TermStructureError::SingularDesign`] if the loadings are collinear.
///
/// # Example
///
/// ```
/// use tsecon_termstructure::fit_nelson_siegel;
/// let maturities = [3.0, 12.0, 36.0, 60.0, 120.0];
/// let yields = [4.1, 4.0, 4.1, 4.25, 4.43];
/// let fit = fit_nelson_siegel(&maturities, &yields, 0.0609).unwrap();
/// assert_eq!(fit.factors.len(), 3);
/// assert!(fit.rsquared > 0.5);
/// ```
pub fn fit_nelson_siegel(
    maturities: &[f64],
    yields: &[f64],
    lambda: f64,
) -> Result<NsFit, TermStructureError> {
    check_maturities(maturities)?;
    check_lambda(lambda, "lambda")?;
    if yields.len() != maturities.len() {
        return Err(TermStructureError::DimensionMismatch {
            what: "Nelson-Siegel yields vs maturities",
            expected: maturities.len(),
            got: yields.len(),
        });
    }
    if maturities.len() <= 3 {
        return Err(TermStructureError::Underdetermined {
            what: "Nelson-Siegel fit",
            maturities: maturities.len(),
            factors: 3,
        });
    }
    check_finite(yields, "Nelson-Siegel yields")?;

    let [level, slope, curv] = nelson_siegel_loadings(maturities, lambda)?;
    let columns = [level, slope, curv];
    let (params, residuals, rsquared) =
        fit_columns(yields, &columns, "Nelson-Siegel cross-sectional fit")?;

    Ok(NsFit {
        factors: [params[0], params[1], params[2]],
        lambda,
        residuals,
        rsquared,
    })
}
