//! Step 1 of two-step (Engle) estimation: the `k` independent univariate
//! GARCH fits, delegated wholesale to `tsecon-garch`.
//!
//! Both CCC (Bollerslev 1990) and DCC (Engle 2002) share this stage: fit a
//! univariate volatility model to each series, read off its conditional
//! variances `sigma2_{i,t}` and standardized residuals
//! `z_{i,t} = eps_{i,t} / sigma_{i,t}`, and hand those to the
//! correlation stage.

use tsecon_garch::{GarchModel, GarchResults, GarchSpec};

use crate::error::MgarchError;

/// The fitted univariate stage: one [`GarchResults`] per series plus the
/// stacked conditional variances and standardized residuals.
#[derive(Debug, Clone)]
pub struct UnivariateStage {
    /// Number of series `k`.
    pub k: usize,
    /// Number of observations `T`.
    pub nobs: usize,
    /// The per-series univariate GARCH fits, in input order.
    pub univariate: Vec<GarchResults>,
    /// Standardized residuals, time-major: `z[t][i] = eps_{i,t} / sigma_{i,t}`.
    pub z: Vec<Vec<f64>>,
    /// Conditional variances, time-major: `sigma2[t][i]`.
    pub sigma2: Vec<Vec<f64>>,
    /// The additive constant part of the Gaussian log-likelihood that comes
    /// from the univariate volatilities alone, i.e.
    /// `-0.5 sum_{t,i} (ln(2 pi) + ln sigma2_{i,t} + z_{i,t}^2)`.
    ///
    /// The full multivariate log-likelihood is this plus the
    /// correlation-copula correction `-0.5 sum_t (ln|R_t| + z_t' R_t^{-1} z_t
    /// - z_t' z_t)` (see [`crate::ccc`] / [`crate::dcc`]).
    pub volatility_loglik: f64,
}

impl UnivariateStage {
    /// Fits `spec` to every column of `series` (each inner vector is one
    /// series of length `T`) and assembles the stacked residuals.
    ///
    /// # Errors
    ///
    /// * [`MgarchError::TooFewSeries`] — fewer than two series;
    /// * [`MgarchError::RaggedInput`] — series of differing lengths;
    /// * [`MgarchError::NonFinite`] — a standardized residual left the
    ///   representable range;
    /// * [`MgarchError::Univariate`] — the univariate fit itself failed on
    ///   some series (the index and underlying [`tsecon_garch::GarchError`]
    ///   are carried).
    pub fn fit(series: &[Vec<f64>], spec: GarchSpec) -> Result<Self, MgarchError> {
        let k = series.len();
        if k < 2 {
            return Err(MgarchError::TooFewSeries { got: k });
        }
        let nobs = series[0].len();
        for (i, s) in series.iter().enumerate() {
            if s.len() != nobs {
                return Err(MgarchError::RaggedInput {
                    expected: nobs,
                    series: i,
                    actual: s.len(),
                });
            }
        }

        let mut univariate = Vec::with_capacity(k);
        for (i, s) in series.iter().enumerate() {
            let model = GarchModel::new(s, spec).map_err(|e| MgarchError::Univariate {
                series: i,
                source: e,
            })?;
            let res = model.fit().map_err(|e| MgarchError::Univariate {
                series: i,
                source: e,
            })?;
            univariate.push(res);
        }

        // Stack time-major and accumulate the univariate log-likelihood.
        let mut z = vec![vec![0.0_f64; k]; nobs];
        let mut sigma2 = vec![vec![0.0_f64; k]; nobs];
        let ln_2pi = (2.0 * core::f64::consts::PI).ln();
        let mut volatility_loglik = 0.0;
        for (i, res) in univariate.iter().enumerate() {
            let s2 = res.conditional_variance();
            let zi = &res.std_residuals;
            for t in 0..nobs {
                let s2t = s2[t];
                let zt = zi[t];
                if !s2t.is_finite() || s2t <= 0.0 || !zt.is_finite() {
                    return Err(MgarchError::NonFinite {
                        what: "univariate standardized residual / conditional variance",
                    });
                }
                z[t][i] = zt;
                sigma2[t][i] = s2t;
                volatility_loglik += -0.5 * (ln_2pi + s2t.ln() + zt * zt);
            }
        }

        Ok(Self {
            k,
            nobs,
            univariate,
            z,
            sigma2,
            volatility_loglik,
        })
    }
}
