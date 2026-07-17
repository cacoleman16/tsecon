//! Dynamic Nelson-Siegel (Diebold & Li 2006): cross-sectional factors for
//! every date in a panel, AR(1) factor dynamics, and one-step curve forecasts.

use crate::error::TermStructureError;
use crate::fit::{check_finite, fit_nelson_siegel, map_ols_err};
use crate::loadings::{check_lambda, check_maturities, nelson_siegel_loadings};
use tsecon_hac::ols;

/// A fitted AR(1) process `x_t = c + phi * x_{t-1} + e_t`, estimated by OLS.
///
/// Used per factor in the two-step dynamic Nelson-Siegel: each factor series
/// (level, slope, curvature) follows its own AR(1) (Diebold & Li 2006, who
/// find the level to be highly persistent).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ar1 {
    /// The intercept `c`.
    pub intercept: f64,
    /// The autoregressive coefficient `phi`.
    pub phi: f64,
}

impl Ar1 {
    /// The one-step-ahead conditional forecast `E[x_{t+1} | x_t] = c + phi x_t`.
    pub fn forecast(&self, last: f64) -> f64 {
        self.intercept + self.phi * last
    }

    /// The unconditional (long-run) mean `c / (1 - phi)`, or `NaN` when
    /// `phi = 1` (a unit root, where no finite mean exists).
    pub fn unconditional_mean(&self) -> f64 {
        let denom = 1.0 - self.phi;
        if denom.abs() < f64::EPSILON {
            f64::NAN
        } else {
            self.intercept / denom
        }
    }
}

/// Fit an AR(1) to a series by OLS of `x_t` on `[1, x_{t-1}]`.
///
/// # Errors
///
/// [`TermStructureError::PanelTooShort`] if fewer than four observations are
/// supplied (`x_t` on a constant and one lag needs `T - 1 > 2` regression
/// rows to have residual degrees of freedom), [`TermStructureError::NonFinite`]
/// on NaN/inf input, and [`TermStructureError::SingularDesign`] if the lagged
/// regressor is constant (a flat series).
pub fn ar1_fit(series: &[f64]) -> Result<Ar1, TermStructureError> {
    let t = series.len();
    if t < 4 {
        return Err(TermStructureError::PanelTooShort {
            what: "AR(1) factor dynamics",
            dates: t,
            needed: 4,
        });
    }
    check_finite(series, "AR(1) series")?;

    let y: Vec<f64> = series[1..].to_vec();
    let lag: Vec<f64> = series[..t - 1].to_vec();
    let ones = vec![1.0_f64; y.len()];
    let fit = ols(&y, &[ones, lag]).map_err(|e| map_ols_err(e, "AR(1) factor dynamics"))?;
    Ok(Ar1 {
        intercept: fit.params[0],
        phi: fit.params[1],
    })
}

/// A dynamic Nelson-Siegel fit over a panel of yield curves at a fixed decay
/// `lambda`.
///
/// Step one of Diebold & Li's (2006) two-step estimator: fit the three
/// Nelson-Siegel factors cross-sectionally, date by date, holding `lambda`
/// fixed. The resulting `[level, slope, curvature]` time series are the state
/// whose dynamics step two models (here, independent AR(1)s — see
/// [`DynamicNsFit::forecast`]).
///
/// Produced by [`fit_dynamic_ns`].
#[derive(Debug, Clone, PartialEq)]
pub struct DynamicNsFit {
    /// The maturity grid shared by every date, in the same units as `lambda`.
    pub maturities: Vec<f64>,
    /// The fixed decay parameter.
    pub lambda: f64,
    /// The `[level, slope, curvature]` factors, one triple per date, in panel
    /// order.
    pub factors: Vec<[f64; 3]>,
    /// The per-date centered R^2 of the cross-sectional fit.
    pub rsquared: Vec<f64>,
}

/// A one-step-ahead dynamic Nelson-Siegel forecast: the forecast factor vector
/// mapped back through the loadings to a forecast yield curve.
#[derive(Debug, Clone, PartialEq)]
pub struct DynamicNsForecast {
    /// The one-step-ahead factor forecast `[level, slope, curvature]`.
    pub factors: [f64; 3],
    /// The fitted AR(1) for each factor, in `[level, slope, curvature]` order.
    pub factor_ar1: [Ar1; 3],
    /// The forecast yields on [`DynamicNsFit::maturities`].
    pub yields: Vec<f64>,
}

impl DynamicNsFit {
    /// The `i`-th factor's time series (`0 = level`, `1 = slope`,
    /// `2 = curvature`) across the panel.
    fn factor_series(&self, i: usize) -> Vec<f64> {
        self.factors.iter().map(|f| f[i]).collect()
    }

    /// The estimated **level** factor time series (the long-rate proxy).
    pub fn level(&self) -> Vec<f64> {
        self.factor_series(0)
    }

    /// The estimated **slope** factor time series.
    pub fn slope(&self) -> Vec<f64> {
        self.factor_series(1)
    }

    /// The estimated **curvature** factor time series.
    pub fn curvature(&self) -> Vec<f64> {
        self.factor_series(2)
    }

    /// The fitted yield curve for date `date_index`, reconstructed from that
    /// date's factors on [`DynamicNsFit::maturities`].
    ///
    /// # Errors
    ///
    /// [`TermStructureError::DimensionMismatch`] if `date_index` is out of
    /// range.
    pub fn fitted_curve(&self, date_index: usize) -> Result<Vec<f64>, TermStructureError> {
        let f = self
            .factors
            .get(date_index)
            .ok_or(TermStructureError::DimensionMismatch {
                what: "dynamic Nelson-Siegel date index",
                expected: self.factors.len(),
                got: date_index,
            })?;
        let [level, slope, curv] = nelson_siegel_loadings(&self.maturities, self.lambda)?;
        Ok((0..self.maturities.len())
            .map(|i| f[0] * level[i] + f[1] * slope[i] + f[2] * curv[i])
            .collect())
    }

    /// One-step-ahead forecast: fit an independent AR(1) to each factor,
    /// forecast the factor vector one period ahead, then map it back through
    /// the loadings to a forecast yield curve.
    ///
    /// ## Two-step vs one-step (state-space) estimation
    ///
    /// This is Diebold & Li's (2006) **two-step** procedure: extract the
    /// factors cross-sectionally (step one, [`fit_dynamic_ns`]) and then treat
    /// them as observed data for the factor VAR/AR (step two, here). It is
    /// simple, robust, and the original Diebold-Li recipe, but it ignores the
    /// cross-sectional measurement error when estimating the dynamics.
    ///
    /// The **one-step** alternative casts the whole model as a linear Gaussian
    /// state space — factors as latent states, the loadings as the measurement
    /// matrix — and estimates the measurement and transition parameters jointly
    /// by maximum likelihood via the Kalman filter (Diebold, Rudebusch &
    /// Aruoba 2006). That is more efficient and yields proper standard errors,
    /// but requires numerical MLE of the full system. Full state-space DNS
    /// estimation is deferred:
    // TODO(phase0): one-step state-space DNS via tsecon-ssm — build the
    // Kalman filter over latent [level, slope, curvature] states with the
    // Nelson-Siegel loadings as the fixed measurement matrix and a VAR(1)
    // transition, and estimate the transition/measurement covariances by MLE
    // (Diebold-Rudebusch-Aruoba 2006). Also generalise step two from diagonal
    // AR(1)s to a full VAR(1) with tsecon-var.
    ///
    /// # Errors
    ///
    /// [`TermStructureError::PanelTooShort`] if the panel has fewer than four
    /// dates (an AR(1) cannot be fit), and any AR(1)-fit error from
    /// [`ar1_fit`].
    pub fn forecast(&self) -> Result<DynamicNsForecast, TermStructureError> {
        let last = self
            .factors
            .last()
            .ok_or(TermStructureError::PanelTooShort {
                what: "dynamic Nelson-Siegel forecast",
                dates: 0,
                needed: 4,
            })?;

        let ar_level = ar1_fit(&self.level())?;
        let ar_slope = ar1_fit(&self.slope())?;
        let ar_curv = ar1_fit(&self.curvature())?;

        let factors = [
            ar_level.forecast(last[0]),
            ar_slope.forecast(last[1]),
            ar_curv.forecast(last[2]),
        ];

        let [level, slope, curv] = nelson_siegel_loadings(&self.maturities, self.lambda)?;
        let yields = (0..self.maturities.len())
            .map(|i| factors[0] * level[i] + factors[1] * slope[i] + factors[2] * curv[i])
            .collect();

        Ok(DynamicNsForecast {
            factors,
            factor_ar1: [ar_level, ar_slope, ar_curv],
            yields,
        })
    }
}

/// Fit the dynamic Nelson-Siegel factors for every date in a yield panel at a
/// fixed decay `lambda` (Diebold & Li 2006, step one).
///
/// `panel` is a `n_dates x n_maturities` matrix: row `d` is the yield curve
/// observed on date `d` at `maturities`. Each row is fit by the cross-sectional
/// [`fit_nelson_siegel`], and the factor triples are stacked into the returned
/// [`DynamicNsFit`].
///
/// # Errors
///
/// [`TermStructureError::PanelTooShort`] if the panel is empty; the maturity
/// and `lambda` validation of [`nelson_siegel_loadings`];
/// [`TermStructureError::DimensionMismatch`] if a panel row's length differs
/// from `maturities.len()`; and any per-date fit error from
/// [`fit_nelson_siegel`].
pub fn fit_dynamic_ns(
    panel: &[Vec<f64>],
    maturities: &[f64],
    lambda: f64,
) -> Result<DynamicNsFit, TermStructureError> {
    check_maturities(maturities)?;
    check_lambda(lambda, "lambda")?;
    if panel.is_empty() {
        return Err(TermStructureError::PanelTooShort {
            what: "dynamic Nelson-Siegel panel",
            dates: 0,
            needed: 1,
        });
    }

    let m = maturities.len();
    let mut factors = Vec::with_capacity(panel.len());
    let mut rsquared = Vec::with_capacity(panel.len());
    for row in panel {
        if row.len() != m {
            return Err(TermStructureError::DimensionMismatch {
                what: "dynamic Nelson-Siegel panel row vs maturities",
                expected: m,
                got: row.len(),
            });
        }
        let fit = fit_nelson_siegel(maturities, row, lambda)?;
        factors.push(fit.factors);
        rsquared.push(fit.rsquared);
    }

    Ok(DynamicNsFit {
        maturities: maturities.to_vec(),
        lambda,
        factors,
        rsquared,
    })
}
