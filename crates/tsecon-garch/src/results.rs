//! The fitted-model results object and analytic variance forecasts.

use crate::error::GarchError;
use crate::inference::StdErrors;
use crate::model::GarchModel;
use crate::spec::{GarchSpec, VolSpec};

/// Estimation output of [`GarchModel::fit`].
///
/// Vectors indexed by parameter are aligned with
/// [`GarchResults::param_names`] (the `arch` ordering: mean, `omega`,
/// `alpha`s, `gamma`s, `beta`s, distribution).
#[derive(Debug, Clone)]
pub struct GarchResults {
    /// The model specification.
    pub spec: GarchSpec,
    /// Estimated parameters.
    pub params: Vec<f64>,
    /// Parameter names, aligned with [`GarchResults::params`].
    pub param_names: Vec<String>,
    /// Maximized log-likelihood.
    pub loglik: f64,
    /// Akaike information criterion, `2 k - 2 loglik` (Akaike 1974),
    /// matching `arch`'s definition.
    pub aic: f64,
    /// Schwarz Bayesian information criterion, `k ln(T) - 2 loglik`
    /// (Schwarz 1978), matching `arch`'s definition.
    pub bic: f64,
    /// Number of observations `T`.
    pub nobs: usize,
    /// Conditional volatility `sigma_t = sqrt(sigma2_t)`.
    pub conditional_volatility: Vec<f64>,
    /// Standardized residuals `z_t = eps_t / sigma_t`.
    pub std_residuals: Vec<f64>,
    /// Classical MLE standard errors (inverse numerical Hessian); NaN in
    /// flat directions.
    pub se_mle: Vec<f64>,
    /// Bollerslev-Wooldridge (1992) robust standard errors (`arch`'s
    /// default covariance); NaN in flat directions.
    pub se_robust: Vec<f64>,
    /// Whether at least one optimizer stage terminated by its convergence
    /// criterion; the final point is at least as good as that stage's
    /// (the best point found is returned either way).
    pub converged: bool,
    /// Residuals `eps_t` at the fitted mean parameters (needed by the
    /// forecast recursions).
    pub(crate) resids: Vec<f64>,
    /// Conditional variances `sigma2_t` at the fitted parameters.
    pub(crate) sigma2: Vec<f64>,
}

impl GarchResults {
    /// Assembles the results object at `params` (used by
    /// [`GarchModel::fit`]).
    pub(crate) fn build(
        model: &GarchModel,
        params: Vec<f64>,
        loglik: f64,
        se: StdErrors,
        converged: bool,
    ) -> Result<Self, GarchError> {
        let spec = *model.spec();
        let sigma2 = model.conditional_variance(&params)?;
        let (mean, ..) = spec.split_params(&params)?;
        let mu = mean.first().copied().unwrap_or(0.0);
        let resids: Vec<f64> = model.y().iter().map(|v| v - mu).collect();
        let conditional_volatility: Vec<f64> = sigma2.iter().map(|s| s.sqrt()).collect();
        let std_residuals: Vec<f64> = resids
            .iter()
            .zip(&conditional_volatility)
            .map(|(e, s)| e / s)
            .collect();
        let k = params.len() as f64;
        let nobs = model.y().len();
        Ok(Self {
            spec,
            param_names: spec.param_names(),
            loglik,
            aic: 2.0 * k - 2.0 * loglik,
            bic: k * (nobs as f64).ln() - 2.0 * loglik,
            nobs,
            conditional_volatility,
            std_residuals,
            se_mle: se.mle,
            se_robust: se.robust,
            converged,
            params,
            resids,
            sigma2,
        })
    }

    /// Residuals `eps_t` at the fitted mean parameters.
    pub fn residuals(&self) -> &[f64] {
        &self.resids
    }

    /// Conditional variances `sigma2_t` at the fitted parameters.
    pub fn conditional_variance(&self) -> &[f64] {
        &self.sigma2
    }

    /// Analytic `h`-step-ahead conditional-variance forecasts
    /// `E[sigma2_{T+m} | F_T]`, `m = 1..=horizon`.
    ///
    /// For GARCH and GJR the recursion replaces unobserved future terms by
    /// their conditional expectations (Bollerslev 1986;
    /// Glosten-Jagannathan-Runkle 1993):
    ///
    /// ```text
    /// E[eps_{T+m}^2]              = E[sigma2_{T+m}]        (m >= 1)
    /// E[eps_{T+m}^2 1[eps < 0]]   = 0.5 E[sigma2_{T+m}]    (m >= 1)
    /// ```
    ///
    /// with `0.5 = P(z < 0)` exact for the symmetric normal and
    /// standardized-t innovations this crate ships; observed values are
    /// used for `m <= 0`. As `h` grows the forecasts converge to the
    /// unconditional variance `omega / (1 - persistence)`.
    ///
    /// For EGARCH only the one-step forecast is analytic
    /// (`E[exp(x)] != exp(E[x])`, Nelson 1991):
    ///
    /// ```text
    /// ln sigma2_{T+1} = omega + sum_i alpha_i (|z_{T+1-i}| - sqrt(2/pi))
    ///                         + sum_i gamma_i z_{T+1-i}
    ///                         + sum_j beta_j ln sigma2_{T+1-j}
    /// ```
    ///
    /// // TODO(phase0): EGARCH multi-step forecasts by simulation (as in
    /// // `arch`), sharing the parallel path engine of ROADMAP 03.
    ///
    /// # Errors
    ///
    /// [`GarchError::InvalidSpec`] if `horizon == 0`;
    /// [`GarchError::UnsupportedForecast`] for EGARCH with `horizon > 1`.
    pub fn forecast_variance(&self, horizon: usize) -> Result<Vec<f64>, GarchError> {
        if horizon == 0 {
            return Err(GarchError::InvalidSpec {
                what: "forecast horizon must be at least 1",
            });
        }
        let (_, omega, alphas, gammas, betas, _) = self.spec.split_params(&self.params)?;
        let n = self.resids.len();
        match self.spec.vol {
            VolSpec::Garch { .. } | VolSpec::Gjr { .. } => {
                let mut fcast = vec![0.0_f64; horizon];
                for m in 1..=horizon {
                    let mut v = omega;
                    // Index helper: lag i of time T+m is T+m-i; for
                    // m - i >= 1 that is forecast m-i, otherwise observed
                    // index n + (m - i) - 1.
                    for (i, &a) in alphas.iter().enumerate() {
                        let lag = m as i64 - (i + 1) as i64;
                        v += a * if lag >= 1 {
                            fcast[lag as usize - 1]
                        } else {
                            let e = self.resids[(n as i64 + lag - 1) as usize];
                            e * e
                        };
                    }
                    for (i, &g) in gammas.iter().enumerate() {
                        let lag = m as i64 - (i + 1) as i64;
                        v += g * if lag >= 1 {
                            0.5 * fcast[lag as usize - 1]
                        } else {
                            let e = self.resids[(n as i64 + lag - 1) as usize];
                            if e < 0.0 {
                                e * e
                            } else {
                                0.0
                            }
                        };
                    }
                    for (j, &b) in betas.iter().enumerate() {
                        let lag = m as i64 - (j + 1) as i64;
                        v += b * if lag >= 1 {
                            fcast[lag as usize - 1]
                        } else {
                            self.sigma2[(n as i64 + lag - 1) as usize]
                        };
                    }
                    fcast[m - 1] = v;
                }
                Ok(fcast)
            }
            VolSpec::Egarch { .. } => {
                if horizon > 1 {
                    return Err(GarchError::UnsupportedForecast {
                        what: "EGARCH multi-step forecasts require simulation \
                               (TODO(phase0)); only horizon = 1 is analytic",
                    });
                }
                let norm_const = (2.0 / core::f64::consts::PI).sqrt();
                let mut v = omega;
                for (i, &a) in alphas.iter().enumerate() {
                    let z = self.std_residuals[n - 1 - i];
                    v += a * (z.abs() - norm_const);
                }
                for (i, &g) in gammas.iter().enumerate() {
                    v += g * self.std_residuals[n - 1 - i];
                }
                for (j, &b) in betas.iter().enumerate() {
                    v += b * self.sigma2[n - 1 - j].ln();
                }
                Ok(vec![v.exp()])
            }
        }
    }
}
