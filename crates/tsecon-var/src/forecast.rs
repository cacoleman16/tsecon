//! Iterated point forecasts and asymptotic forecast-error intervals.

use tsecon_linalg::faer::Mat;
use tsecon_stats::special::inv_norm_cdf;

use crate::error::VarError;
use crate::results::VarResults;

/// Point forecasts with symmetric asymptotic intervals, produced by
/// [`VarResults::forecast_interval`]. All matrices are `steps x k`,
/// row `h` being the `(h + 1)`-step-ahead values.
#[derive(Debug, Clone)]
pub struct ForecastInterval {
    /// Iterated point forecasts.
    pub point: Mat<f64>,
    /// Lower interval bounds, `point - z_{1 - alpha/2} * se`.
    pub lower: Mat<f64>,
    /// Upper interval bounds, `point + z_{1 - alpha/2} * se`.
    pub upper: Mat<f64>,
}

impl VarResults {
    /// Iterated point forecasts `steps` periods past the estimation
    /// sample (Lütkepohl 2005, eq. 3.5.5):
    ///
    /// ```text
    /// y_{T+h|T} = c + sum_{i=1}^{p} A_i y_{T+h-i|T}
    /// ```
    ///
    /// with `y_{s|T} = y_s` for `s <= T`. The last `p` rows of the
    /// stored estimation sample seed the recursion, matching
    /// statsmodels `forecast(y[-p:], steps)`. Returns a `steps x k`
    /// matrix.
    ///
    /// # Errors
    ///
    /// [`VarError::InvalidArgument`] if `steps == 0`.
    pub fn forecast(&self, steps: usize) -> Result<Mat<f64>, VarError> {
        if steps == 0 {
            return Err(VarError::InvalidArgument {
                what: "forecast needs at least one step",
            });
        }
        let k = self.neqs;
        let n = self.endog.nrows();
        let mut out = Mat::<f64>::zeros(steps, k);
        for h in 0..steps {
            for r in 0..k {
                let mut v = self.intercept[r];
                for (i, a) in self.coefs.iter().enumerate() {
                    // y_{T + h - i} (1-based lag i = index i + 1).
                    for c in 0..k {
                        let lagged = if h > i {
                            out[(h - i - 1, c)]
                        } else {
                            self.endog[(n - (i + 1 - h), c)]
                        };
                        v += a[(r, c)] * lagged;
                    }
                }
                out[(h, r)] = v;
            }
        }
        Ok(out)
    }

    /// Asymptotic forecast-error covariance matrices for horizons
    /// `1..=steps` (Lütkepohl 2005, eq. 2.2.11):
    ///
    /// ```text
    /// MSE(h) = sum_{i=0}^{h-1} Psi_i sigma_u Psi_i'
    /// ```
    ///
    /// using the df-adjusted `sigma_u` and treating the coefficients as
    /// known — innovation uncertainty only, no parameter-estimation
    /// term, exactly like statsmodels `forecast_interval` /
    /// `VARProcess.mse`.
    ///
    /// # Errors
    ///
    /// [`VarError::InvalidArgument`] if `steps == 0`.
    pub fn forecast_cov(&self, steps: usize) -> Result<Vec<Mat<f64>>, VarError> {
        if steps == 0 {
            return Err(VarError::InvalidArgument {
                what: "forecast_cov needs at least one step",
            });
        }
        let psi = self.ma_rep(steps - 1)?;
        let k = self.neqs;
        let mut covs = Vec::with_capacity(steps);
        let mut acc = Mat::<f64>::zeros(k, k);
        for phi in &psi {
            acc += phi * &self.sigma_u * phi.transpose();
            covs.push(acc.clone());
        }
        Ok(covs)
    }

    /// Point forecasts with symmetric `1 - alpha` asymptotic intervals
    /// (statsmodels `forecast_interval`):
    ///
    /// ```text
    /// y_{T+h|T} +/- z_{1 - alpha/2} sqrt(diag MSE(h))
    /// ```
    ///
    /// The intervals reflect innovation uncertainty only (see
    /// [`VarResults::forecast_cov`]); parameter uncertainty and
    /// bootstrap intervals are `// TODO(phase0)` alongside the
    /// tsecon-bootstrap IRF bands.
    ///
    /// # Errors
    ///
    /// * [`VarError::InvalidArgument`] if `steps == 0` or `alpha` is
    ///   not strictly inside `(0, 1)`;
    /// * [`VarError::Stats`] if the normal quantile fails (impossible
    ///   for valid `alpha`).
    pub fn forecast_interval(
        &self,
        steps: usize,
        alpha: f64,
    ) -> Result<ForecastInterval, VarError> {
        if !(alpha > 0.0 && alpha < 1.0) {
            return Err(VarError::InvalidArgument {
                what: "alpha must lie strictly between 0 and 1",
            });
        }
        let point = self.forecast(steps)?;
        let covs = self.forecast_cov(steps)?;
        let z = inv_norm_cdf(1.0 - alpha / 2.0)?;
        let k = self.neqs;
        let mut lower = Mat::<f64>::zeros(steps, k);
        let mut upper = Mat::<f64>::zeros(steps, k);
        for h in 0..steps {
            for j in 0..k {
                let se = covs[h][(j, j)].max(0.0).sqrt();
                lower[(h, j)] = point[(h, j)] - z * se;
                upper[(h, j)] = point[(h, j)] + z * se;
            }
        }
        Ok(ForecastInterval {
            point,
            lower,
            upper,
        })
    }
}
