//! Forecast-error variance decomposition.

use tsecon_linalg::faer::Mat;

use crate::error::VarError;
use crate::results::VarResults;

/// Forecast-error variance decomposition of an estimated VAR
/// (Lütkepohl 2005, section 2.3.3; statsmodels `VARResults.fevd`).
#[derive(Debug, Clone)]
pub struct Fevd {
    /// One `periods x k` matrix per variable: `decomp[i][(h, j)]` is
    /// the share of the `(h + 1)`-step forecast-error variance of
    /// variable `i` attributable to the `j`-th orthogonalized
    /// (Cholesky) shock. Every row sums to 1.
    pub decomp: Vec<Mat<f64>>,
}

impl VarResults {
    /// Forecast-error variance decomposition to `periods` horizons.
    ///
    /// With `Theta_h = Psi_h P` the orthogonalized responses
    /// ([`VarResults::orth_ma_rep`]), the share of shock `j` in the
    /// `(h + 1)`-step forecast-error variance of variable `i` is
    ///
    /// ```text
    /// omega_{ij}(h + 1) = sum_{s=0}^{h} Theta_s[i, j]^2
    ///                     / sum_{m} sum_{s=0}^{h} Theta_s[i, m]^2
    /// ```
    ///
    /// (Lütkepohl 2005, eq. 2.3.37) — the denominator equals the `i`-th
    /// diagonal entry of the `(h + 1)`-step forecast MSE matrix since
    /// `P P' = sigma_u`. Matches statsmodels `fevd(periods)` (which
    /// uses horizons `0..periods - 1`).
    ///
    /// # Errors
    ///
    /// * [`VarError::InvalidArgument`] if `periods == 0`;
    /// * [`VarError::NotPositiveDefinite`] if `sigma_u` has no Cholesky
    ///   factor.
    pub fn fevd(&self, periods: usize) -> Result<Fevd, VarError> {
        if periods == 0 {
            return Err(VarError::InvalidArgument {
                what: "fevd needs at least one period",
            });
        }
        let k = self.neqs;
        let orth = self.orth_ma_rep(periods - 1)?;
        let mut decomp = vec![Mat::<f64>::zeros(periods, k); k];
        // Running sums of squared orthogonalized responses per (i, j).
        let mut cum = Mat::<f64>::zeros(k, k);
        for (h, theta) in orth.iter().enumerate() {
            for i in 0..k {
                for j in 0..k {
                    cum[(i, j)] += theta[(i, j)] * theta[(i, j)];
                }
            }
            for i in 0..k {
                let mse_i: f64 = (0..k).map(|j| cum[(i, j)]).sum();
                if mse_i <= 0.0 {
                    return Err(VarError::NotPositiveDefinite {
                        what: "forecast MSE diagonal in fevd",
                    });
                }
                for j in 0..k {
                    decomp[i][(h, j)] = cum[(i, j)] / mse_i;
                }
            }
        }
        Ok(Fevd { decomp })
    }
}
