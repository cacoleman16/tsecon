//! Impulse response functions: MA(infinity) coefficients of the
//! estimated VAR, non-orthogonalized and orthogonalized.

use tsecon_linalg::faer::Mat;

use crate::error::VarError;
use crate::results::{chol_lower, VarResults};

// TODO(phase0): bootstrap IRF confidence bands. The hook will consume
// tsecon-bootstrap's recursive-design residual resampler: re-estimate the
// VAR on each bootstrap path, recompute `Irf`, and report percentile /
// Hall bands per (response, impulse, horizon) cell (Lütkepohl 2005,
// appendix D; Kilian 1998 bias-adjusted variant later).

/// Impulse responses of an estimated VAR to `horizon` periods.
///
/// Produced by [`VarResults::irf`]; both arrays have `horizon + 1`
/// entries, indexed by horizon `h = 0..=horizon`, each a `k x k` matrix
/// whose `(i, j)` entry is the response of variable `i` to a shock in
/// variable `j` `h` periods earlier.
#[derive(Debug, Clone)]
pub struct Irf {
    /// Non-orthogonalized MA coefficients `Psi_h` (unit reduced-form
    /// innovation in each variable); `Psi_0 = I`.
    pub irfs: Vec<Mat<f64>>,
    /// Orthogonalized responses `Psi_h P` with `P` the lower Cholesky
    /// factor of `sigma_u` (one-standard-deviation structural shocks
    /// under the recursive ordering; Lütkepohl 2005, section 3.7).
    pub orth_irfs: Vec<Mat<f64>>,
}

/// MA(infinity) coefficient matrices `Psi_0, ..., Psi_horizon` of a VAR
/// lag polynomial with coefficient matrices `coefs = [A_1, ..., A_p]`
/// (Lütkepohl 2005, eq. 2.1.22):
///
/// ```text
/// Psi_0 = I,    Psi_h = sum_{i=1}^{min(h, p)} Psi_{h-i} A_i
/// ```
///
/// These are the responses of `y_{t+h}` to a unit reduced-form
/// innovation at time `t`. Exposed as a free function so callers (and
/// property tests) can expand an arbitrary lag polynomial without an
/// estimation step; the returned vector has `horizon + 1` entries.
///
/// # Errors
///
/// * [`VarError::InvalidArgument`] if `coefs` is empty or a matrix is
///   `0 x 0`;
/// * [`VarError::Dimension`] if the matrices are not square and of equal
///   size;
/// * [`VarError::NonFinite`] on NaN/infinite entries.
pub fn ma_rep(coefs: &[Mat<f64>], horizon: usize) -> Result<Vec<Mat<f64>>, VarError> {
    if coefs.is_empty() {
        return Err(VarError::InvalidArgument {
            what: "coefs must contain at least one lag matrix",
        });
    }
    let k = coefs[0].nrows();
    if k == 0 {
        return Err(VarError::InvalidArgument {
            what: "coefs matrices must be non-empty",
        });
    }
    for a in coefs {
        if a.nrows() != k || a.ncols() != k {
            return Err(VarError::Dimension {
                what: "all VAR coefficient matrices must be square of one size",
                expected: k,
                got: if a.nrows() != k { a.nrows() } else { a.ncols() },
            });
        }
        for j in 0..k {
            for i in 0..k {
                if !a[(i, j)].is_finite() {
                    return Err(VarError::NonFinite { what: "coefs" });
                }
            }
        }
    }
    let p = coefs.len();
    let mut psi: Vec<Mat<f64>> = Vec::with_capacity(horizon + 1);
    psi.push(Mat::from_fn(k, k, |i, j| f64::from(u8::from(i == j))));
    for h in 1..=horizon {
        let mut acc = Mat::<f64>::zeros(k, k);
        for i in 1..=h.min(p) {
            acc += &psi[h - i] * &coefs[i - 1];
        }
        psi.push(acc);
    }
    Ok(psi)
}

impl VarResults {
    /// Non-orthogonalized MA coefficients `Psi_0, ..., Psi_horizon` of
    /// the estimated lag polynomial (see [`ma_rep`]); for a VAR(0) all
    /// responses beyond `Psi_0 = I` are zero.
    ///
    /// # Errors
    ///
    /// Propagates [`ma_rep`] failures (impossible for coefficients
    /// produced by a successful fit).
    pub fn ma_rep(&self, horizon: usize) -> Result<Vec<Mat<f64>>, VarError> {
        let k = self.neqs;
        if self.spec.lags == 0 {
            let mut psi = vec![Mat::<f64>::zeros(k, k); horizon + 1];
            for i in 0..k {
                psi[0][(i, i)] = 1.0;
            }
            return Ok(psi);
        }
        ma_rep(&self.coefs, horizon)
    }

    /// Orthogonalized MA coefficients `Psi_h P`, with `P` the lower
    /// Cholesky factor of the df-adjusted `sigma_u` (statsmodels
    /// `orth_ma_rep`; Lütkepohl 2005, section 3.7.1). Column `j` of the
    /// horizon-`h` matrix is the response to a one-standard-deviation
    /// shock in the `j`-th orthogonalized innovation under the
    /// recursive (Cholesky) ordering of the variables.
    ///
    /// # Errors
    ///
    /// [`VarError::NotPositiveDefinite`] if `sigma_u` has no Cholesky
    /// factor, plus [`ma_rep`] failures.
    pub fn orth_ma_rep(&self, horizon: usize) -> Result<Vec<Mat<f64>>, VarError> {
        let p_chol = chol_lower(self.sigma_u.as_ref(), "sigma_u")?;
        let psi = self.ma_rep(horizon)?;
        Ok(psi.iter().map(|m| m * &p_chol).collect())
    }

    /// Impulse responses to `horizon` periods: both the
    /// non-orthogonalized `Psi_h` and the Cholesky-orthogonalized
    /// `Psi_h P` arrays (statsmodels `VARResults.irf`).
    ///
    /// # Errors
    ///
    /// See [`VarResults::orth_ma_rep`].
    pub fn irf(&self, horizon: usize) -> Result<Irf, VarError> {
        let irfs = self.ma_rep(horizon)?;
        let p_chol = chol_lower(self.sigma_u.as_ref(), "sigma_u")?;
        let orth_irfs = irfs.iter().map(|m| m * &p_chol).collect();
        Ok(Irf { irfs, orth_irfs })
    }
}
