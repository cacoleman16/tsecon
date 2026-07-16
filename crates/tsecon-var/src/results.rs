//! Estimation results object and companion-form stability analysis.

use tsecon_linalg::faer::{Mat, MatRef, Side};
use tsecon_linalg::{companion_from_var, spectral_radius, LinalgError};

use crate::error::VarError;
use crate::spec::VarSpec;

/// Results of a reduced-form VAR(p) OLS fit.
///
/// Estimator conventions match statsmodels `VARResults` exactly (the
/// golden fixture arbitrates):
///
/// * `params` is the `(n_trend + k p) x k` stacked coefficient matrix
///   with regressor rows ordered `[const, lag-1 var 1..k, lag-2 var
///   1..k, ...]` and one column per equation;
/// * `sigma_u` uses the degrees-of-freedom-adjusted divisor `T - m`
///   (`m = n_trend + k p` regressors per equation), `sigma_u_mle` the
///   ML divisor `T`;
/// * `llf` is the Gaussian log-likelihood at the ML covariance
///   (Lütkepohl 2005, eq. 3.4.5);
/// * the information criteria are `ln det(sigma_u_mle)` plus the
///   statsmodels penalty with free-parameter count `p k^2 + k n_trend`
///   (deterministic terms are counted, the `k(k+1)/2` covariance
///   parameters are not; Lütkepohl 2005, section 4.3).
///
/// All fields are public so the structural identification / SVAR layer
/// can consume the reduced form (`sigma_u`, `coefs`, `resid`) directly.
#[derive(Debug, Clone)]
pub struct VarResults {
    /// The specification that was fitted.
    pub spec: VarSpec,
    /// Number of endogenous variables `k`.
    pub neqs: usize,
    /// Effective number of observations `T` (rows of `endog` minus
    /// `lags` presample rows).
    pub nobs: usize,
    /// Number of regressors per equation, `m = n_trend + k p`
    /// (statsmodels `df_model`).
    pub df_model: usize,
    /// Residual degrees of freedom per equation, `T - m`.
    pub df_resid: usize,
    /// The estimation sample (`n x k`, oldest row first); the last
    /// `lags` rows seed [`VarResults::forecast`].
    pub endog: Mat<f64>,
    /// Stacked OLS coefficients, `(n_trend + k p) x k` — statsmodels
    /// `params` layout (rows: deterministic terms then lags; columns:
    /// equations).
    pub params: Mat<f64>,
    /// Intercept vector `c` (length `k`; all zeros when the trend is
    /// [`crate::Trend::None`]).
    pub intercept: Vec<f64>,
    /// Lag coefficient matrices `[A_1, ..., A_p]`, each `k x k`, with
    /// `A_i[(r, c)]` the effect of variable `c` at lag `i` on variable
    /// `r`.
    pub coefs: Vec<Mat<f64>>,
    /// OLS residuals `U = Y - Z B` (`T x k`).
    pub resid: Mat<f64>,
    /// Degrees-of-freedom-adjusted residual covariance,
    /// `U'U / (T - m)` (Lütkepohl 2005, eq. 3.2.19; statsmodels
    /// `sigma_u`).
    pub sigma_u: Mat<f64>,
    /// Maximum-likelihood residual covariance, `U'U / T` (statsmodels
    /// `sigma_u_mle`).
    pub sigma_u_mle: Mat<f64>,
    /// Inverse regressor cross-product `(Z'Z)^{-1}` (`m x m`); together
    /// with `sigma_u` it gives the coefficient covariance
    /// `kron((Z'Z)^{-1}, sigma_u)` (Lütkepohl 2005, eq. 3.2.21).
    pub zz_inv: Mat<f64>,
    /// Gaussian log-likelihood at the ML covariance:
    /// `-(T k / 2)(1 + ln 2 pi) - (T / 2) ln det(sigma_u_mle)`.
    pub llf: f64,
    /// Akaike information criterion,
    /// `ln det(sigma_u_mle) + 2 f / T` with `f = p k^2 + k n_trend`.
    pub aic: f64,
    /// Schwarz (Bayesian) information criterion,
    /// `ln det(sigma_u_mle) + f ln(T) / T`.
    pub bic: f64,
    /// Hannan-Quinn information criterion,
    /// `ln det(sigma_u_mle) + 2 f ln(ln T) / T`.
    pub hqic: f64,
    /// Final prediction error,
    /// `((T + m) / (T - m))^k det(sigma_u_mle)` (Lütkepohl 2005,
    /// eq. 4.3.1).
    pub fpe: f64,
}

impl VarResults {
    /// The `kp x kp` companion matrix of the estimated lag polynomial
    /// (Lütkepohl 2005, eq. 2.1.8).
    ///
    /// # Errors
    ///
    /// [`VarError::InvalidArgument`] if `lags == 0` (no lag polynomial),
    /// plus anything [`companion_from_var`] can return.
    pub fn companion(&self) -> Result<Mat<f64>, VarError> {
        if self.spec.lags == 0 {
            return Err(VarError::InvalidArgument {
                what: "a VAR(0) has no companion matrix",
            });
        }
        let refs: Vec<MatRef<'_, f64>> = self.coefs.iter().map(Mat::as_ref).collect();
        Ok(companion_from_var(&refs)?)
    }

    /// Moduli of the roots of the reverse characteristic polynomial
    /// `det(I - A_1 z - ... - A_p z^p) = 0`, i.e. the reciprocals of the
    /// companion eigenvalue moduli, sorted in decreasing order —
    /// statsmodels `VARResults.roots` moduli.
    ///
    /// The process is stable iff every root lies outside the unit
    /// circle, i.e. the *smallest* returned modulus exceeds 1
    /// (Lütkepohl 2005, eq. 2.1.12). A zero companion eigenvalue maps
    /// to `f64::INFINITY` (a root at infinity).
    ///
    /// Returns an empty vector for `lags == 0`.
    ///
    /// # Errors
    ///
    /// [`VarError::Linalg`] if the eigenvalue iteration fails.
    pub fn roots_moduli(&self) -> Result<Vec<f64>, VarError> {
        if self.spec.lags == 0 {
            return Ok(Vec::new());
        }
        let comp = self.companion()?;
        let eigs = comp
            .eigenvalues()
            .map_err(|_| LinalgError::EigenFailed { what: "VAR roots" })?;
        let mut moduli: Vec<f64> = eigs
            .iter()
            .map(|c| {
                let m = c.re.hypot(c.im);
                if m == 0.0 {
                    f64::INFINITY
                } else {
                    m.recip()
                }
            })
            .collect();
        moduli.sort_by(|a, b| b.partial_cmp(a).unwrap_or(core::cmp::Ordering::Equal));
        Ok(moduli)
    }

    /// Stability check: `true` iff the spectral radius of the companion
    /// matrix is strictly below 1 (equivalently, all characteristic
    /// roots lie outside the unit circle; Lütkepohl 2005, prop. 2.1).
    ///
    /// A VAR(0) is trivially stable.
    ///
    /// # Errors
    ///
    /// [`VarError::Linalg`] if the eigenvalue iteration fails.
    pub fn is_stable(&self) -> Result<bool, VarError> {
        if self.spec.lags == 0 {
            return Ok(true);
        }
        let comp = self.companion()?;
        Ok(spectral_radius(comp.as_ref())? < 1.0)
    }
}

/// Lower Cholesky factor of a symmetric positive definite matrix, with a
/// crate-specific error naming the offending matrix.
pub(crate) fn chol_lower(m: MatRef<'_, f64>, what: &'static str) -> Result<Mat<f64>, VarError> {
    m.llt(Side::Lower)
        .map(|f| f.L().to_owned())
        .map_err(|_| VarError::NotPositiveDefinite { what })
}

/// `ln det(M)` of a symmetric positive definite matrix via its Cholesky
/// factor: `2 sum_i ln L_ii`.
pub(crate) fn ln_det_spd(m: MatRef<'_, f64>, what: &'static str) -> Result<f64, VarError> {
    let l = chol_lower(m, what)?;
    let mut ld = 0.0;
    for i in 0..l.nrows() {
        ld += l[(i, i)].ln();
    }
    Ok(2.0 * ld)
}
