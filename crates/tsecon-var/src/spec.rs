//! Model specification for a reduced-form VAR(p).

use tsecon_linalg::faer::MatRef;

use crate::error::VarError;
use crate::estimate::estimate;
use crate::results::VarResults;

/// Deterministic trend specification.
///
/// Mirrors the statsmodels `VAR.fit(trend=...)` options implemented so
/// far: `"n"` (no deterministic terms) and `"c"` (a constant in every
/// equation). Linear/quadratic trends are `// TODO(phase0)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Trend {
    /// No deterministic terms (statsmodels `trend="n"`).
    None,
    /// A constant (intercept) in every equation (statsmodels
    /// `trend="c"`).
    Constant,
}

impl Trend {
    /// Number of deterministic regressors this trend contributes per
    /// equation.
    pub(crate) fn n_terms(self) -> usize {
        match self {
            Self::None => 0,
            Self::Constant => 1,
        }
    }
}

/// Specification of a reduced-form VAR(p) (Lütkepohl 2005, eq. 2.1.1):
///
/// ```text
/// y_t = c + A_1 y_{t-1} + ... + A_p y_{t-p} + u_t,   u_t ~ (0, Sigma_u)
/// ```
///
/// with `y_t` a `k`-vector, `A_i` `k x k` coefficient matrices, and `c`
/// present iff `trend == Trend::Constant`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VarSpec {
    /// Lag order `p`. `lags = 0` is permitted (intercept-only model)
    /// when `trend == Trend::Constant`; it is what statsmodels'
    /// `select_order` fits for the `p = 0` candidate.
    pub lags: usize,
    /// Deterministic trend terms.
    pub trend: Trend,
}

impl VarSpec {
    /// Creates a specification, validating that it has at least one
    /// regressor per equation.
    ///
    /// # Errors
    ///
    /// [`VarError::InvalidArgument`] if `lags == 0` and
    /// `trend == Trend::None` (the design matrix would be empty).
    pub fn new(lags: usize, trend: Trend) -> Result<Self, VarError> {
        if lags == 0 && trend == Trend::None {
            return Err(VarError::InvalidArgument {
                what: "lags = 0 with Trend::None leaves no regressors; \
                       include a constant or at least one lag",
            });
        }
        Ok(Self { lags, trend })
    }

    /// Fits the VAR by equation-by-equation OLS (multivariate least
    /// squares; Lütkepohl 2005, section 3.2) on `endog`, an
    /// `n x k` matrix with observations in rows, oldest first.
    ///
    /// See [`VarResults`] for the estimator conventions (all of which
    /// follow statsmodels `VAR.fit`).
    ///
    /// # Errors
    ///
    /// * [`VarError::NonFinite`] if `endog` contains NaN/infinity;
    /// * [`VarError::Dimension`] if `endog` has zero columns;
    /// * [`VarError::InvalidArgument`] for `lags = 0` with no trend;
    /// * [`VarError::InsufficientObservations`] if fewer than
    ///   `lags + (k * lags + n_trend) + 1` rows are available (OLS needs
    ///   `T > k p + n_trend` usable observations);
    /// * [`VarError::NotPositiveDefinite`] if the regressor cross-product
    ///   or the residual covariance is numerically singular.
    pub fn fit(&self, endog: MatRef<'_, f64>) -> Result<VarResults, VarError> {
        estimate(endog, self.lags, self.trend, 0)
    }
}
