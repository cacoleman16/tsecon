//! The Svensson (1994) four-factor extension of Nelson-Siegel.

use crate::error::TermStructureError;
use crate::fit::{check_finite, fit_columns};
use crate::loadings::{check_lambda, check_maturities, svensson_loadings};

/// A fitted four-factor Svensson (1994) yield curve at fixed decays
/// `lambda1`, `lambda2`.
///
/// Extends [`crate::NsFit`] with a second curvature factor `beta3` loaded on
/// its own decay `lambda2`, letting the curve fit a second hump. Produced by
/// [`fit_svensson`].
#[derive(Debug, Clone, PartialEq)]
pub struct SvenssonFit {
    /// The recovered factors `[beta0, beta1, beta2, beta3] = [level, slope,
    /// curvature1, curvature2]`. `beta0..beta2` carry the Nelson-Siegel
    /// interpretation (level, slope, first hump); `beta3` is a second hump
    /// governed by `lambda2`.
    pub factors: [f64; 4],
    /// The first decay parameter (governs slope and the first curvature).
    pub lambda1: f64,
    /// The second decay parameter (governs the second curvature).
    pub lambda2: f64,
    /// Cross-sectional residuals `y_i - yhat_i`, one per input maturity.
    pub residuals: Vec<f64>,
    /// Centered R^2 of the cross-sectional fit.
    pub rsquared: f64,
}

impl SvenssonFit {
    /// The fitted yield at an arbitrary maturity `t`,
    /// `yhat(t) = level + slope * g(l1 t) + curv1 * h(l1 t) + curv2 * h(l2 t)`
    /// with `g(x) = (1 - e^{-x})/x` and `h(x) = g(x) - e^{-x}`.
    ///
    /// # Errors
    ///
    /// [`TermStructureError::InvalidMaturity`] for a non-positive/non-finite
    /// maturity.
    pub fn yield_at(&self, maturity: f64) -> Result<f64, TermStructureError> {
        let [level, slope, c1, c2] = svensson_loadings(&[maturity], self.lambda1, self.lambda2)?;
        Ok(self.factors[0] * level[0]
            + self.factors[1] * slope[0]
            + self.factors[2] * c1[0]
            + self.factors[3] * c2[0])
    }

    /// The fitted yields on a maturity grid.
    ///
    /// # Errors
    ///
    /// As [`SvenssonFit::yield_at`].
    pub fn fitted(&self, maturities: &[f64]) -> Result<Vec<f64>, TermStructureError> {
        let [level, slope, c1, c2] = svensson_loadings(maturities, self.lambda1, self.lambda2)?;
        Ok((0..maturities.len())
            .map(|i| {
                self.factors[0] * level[i]
                    + self.factors[1] * slope[i]
                    + self.factors[2] * c1[i]
                    + self.factors[3] * c2[i]
            })
            .collect())
    }
}

/// Fit a single yield curve by cross-sectional OLS on the four Svensson (1994)
/// loadings at **fixed** decays `lambda1`, `lambda2`.
///
/// As in the Nelson-Siegel case, holding both decays fixed makes the model
/// linear in the four factors, so the fit is an OLS of the yields on the
/// `[level, slope, curvature1, curvature2]` columns. The extra curvature term
/// requires at least five maturities to be identified.
///
/// # Errors
///
/// Validation errors from [`svensson_loadings`];
/// [`TermStructureError::DimensionMismatch`] if `yields.len()` differs from
/// `maturities.len()`; [`TermStructureError::Underdetermined`] with fewer than
/// five maturities; [`TermStructureError::NonFinite`] on NaN/inf yields; and
/// [`TermStructureError::SingularDesign`] if the loadings are collinear (which
/// happens when `lambda1` and `lambda2` are too close).
pub fn fit_svensson(
    maturities: &[f64],
    yields: &[f64],
    lambda1: f64,
    lambda2: f64,
) -> Result<SvenssonFit, TermStructureError> {
    check_maturities(maturities)?;
    check_lambda(lambda1, "lambda1")?;
    check_lambda(lambda2, "lambda2")?;
    if yields.len() != maturities.len() {
        return Err(TermStructureError::DimensionMismatch {
            what: "Svensson yields vs maturities",
            expected: maturities.len(),
            got: yields.len(),
        });
    }
    if maturities.len() <= 4 {
        return Err(TermStructureError::Underdetermined {
            what: "Svensson fit",
            maturities: maturities.len(),
            factors: 4,
        });
    }
    check_finite(yields, "Svensson yields")?;

    let [level, slope, c1, c2] = svensson_loadings(maturities, lambda1, lambda2)?;
    let columns = [level, slope, c1, c2];
    let (params, residuals, rsquared) =
        fit_columns(yields, &columns, "Svensson cross-sectional fit")?;

    Ok(SvenssonFit {
        factors: [params[0], params[1], params[2], params[3]],
        lambda1,
        lambda2,
        residuals,
        rsquared,
    })
}
