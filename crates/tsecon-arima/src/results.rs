//! Fitted-model results: parameters, information criteria, forecasting
//! with undifferencing, and standardized residuals.

use tsecon_linalg::faer::Mat;
use tsecon_ssm::LinearGaussianSSM;
use tsecon_stats::dist::ContinuousDist;
use tsecon_stats::StdNormal;

use crate::error::ArimaError;
use crate::spec::ArimaSpec;
use crate::ssm::arma_ssm;

/// How a set of [`ArimaResults`] was estimated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EstimationMethod {
    /// Exact Gaussian maximum likelihood on the state-space form
    /// ([`ArimaSpec::fit`]).
    ExactMle,
    /// Conditional sum of squares ([`ArimaSpec::fit_css`]); the reported
    /// log-likelihood is conditional, not the exact one.
    Css,
    /// Evaluated at user-supplied parameters without optimization
    /// ([`ArimaSpec::at_params`]).
    Fixed,
}

/// A fitted ARIMA model.
///
/// Produced by [`ArimaSpec::fit`] (exact MLE) or [`ArimaSpec::fit_css`]
/// (conditional sum of squares); all invariants (parameter layout,
/// stationarity of the fitted AR block) are established at construction,
/// which is why the parameter storage is private behind accessors.
#[derive(Debug, Clone)]
pub struct ArimaResults {
    /// The specification that was fit.
    pub spec: ArimaSpec,
    /// The estimation method used.
    pub method: EstimationMethod,
    /// Maximized log-likelihood: exact (prediction-error decomposition)
    /// for [`EstimationMethod::ExactMle`], conditional for
    /// [`EstimationMethod::Css`].
    pub loglik: f64,
    /// Akaike information criterion `-2 loglik + 2 k` with `k` counting
    /// the constant, AR, MA, *and* `sigma2` parameters — the statsmodels
    /// convention.
    pub aic: f64,
    /// Bayesian information criterion `-2 loglik + k ln(nobs)`
    /// (statsmodels convention; same `k` as AIC).
    pub bic: f64,
    /// Effective number of observations behind `loglik`: `n - d` for
    /// exact MLE (simple differencing), `n - d - p` for CSS (which also
    /// conditions on the first `p` observations).
    pub nobs: usize,
    /// Number of estimated parameters `k` (constant + p + q + 1).
    pub k_params: usize,
    /// Whether the optimizer satisfied a convergence test; when `false`
    /// the reported parameters are the best point found and should be
    /// treated with care.
    pub converged: bool,
    params: Vec<f64>,
    param_names: Vec<String>,
    /// The ARMA estimation sample (the `d`-times-differenced data).
    x: Vec<f64>,
    /// Undifferencing anchors (see [`crate::diff`]).
    anchors: Vec<f64>,
}

impl ArimaResults {
    /// Assembles results from a completed fit (crate-internal: the
    /// estimation code guarantees the parameter-vector invariants).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_fit(
        spec: ArimaSpec,
        method: EstimationMethod,
        params: Vec<f64>,
        loglik: f64,
        nobs: usize,
        converged: bool,
        x: Vec<f64>,
        anchors: Vec<f64>,
    ) -> Self {
        let k = spec.k_params();
        debug_assert_eq!(params.len(), k);
        let aic = -2.0 * loglik + 2.0 * k as f64;
        let bic = -2.0 * loglik + k as f64 * (nobs as f64).ln();
        Self {
            spec,
            method,
            loglik,
            aic,
            bic,
            nobs,
            k_params: k,
            converged,
            params,
            param_names: spec.param_names(),
            x,
            anchors,
        }
    }

    /// The packed parameter vector `[const?, ar_1..ar_p, ma_1..ma_q,
    /// sigma2]`, aligned with [`ArimaResults::param_names`].
    pub fn params(&self) -> &[f64] {
        &self.params
    }

    /// Parameter names (statsmodels style: `const`, `ar.L1`, ...,
    /// `ma.L1`, ..., `sigma2`), aligned with [`ArimaResults::params`].
    pub fn param_names(&self) -> &[String] {
        &self.param_names
    }

    /// The fitted constant, if the specification includes one.
    pub fn constant(&self) -> Option<f64> {
        self.spec.include_constant().then(|| self.params[0])
    }

    /// The fitted AR coefficients `phi_1..phi_p`.
    pub fn ar(&self) -> &[f64] {
        let c = usize::from(self.spec.include_constant());
        &self.params[c..c + self.spec.p()]
    }

    /// The fitted MA coefficients `theta_1..theta_q`.
    pub fn ma(&self) -> &[f64] {
        let c = usize::from(self.spec.include_constant());
        &self.params[c + self.spec.p()..c + self.spec.p() + self.spec.q()]
    }

    /// The fitted innovation variance `sigma2`.
    pub fn sigma2(&self) -> f64 {
        self.params[self.params.len() - 1]
    }

    /// The state-space form at the fitted parameters.
    fn model(&self) -> Result<LinearGaussianSSM, ArimaError> {
        arma_ssm(
            self.ar(),
            self.ma(),
            self.sigma2(),
            self.constant().unwrap_or(0.0),
        )
    }

    /// Standardized one-step prediction errors from the Kalman filter,
    ///
    /// ```text
    /// e_t = v_t / sqrt(F_t),   v_t = x_t - Z a_{t|t-1},
    /// F_t = Z P_{t|t-1} Z'
    /// ```
    ///
    /// on the (differenced) estimation sample (Durbin & Koopman 2012,
    /// section 2.12; statsmodels `standardized_forecasts_error`). Under a
    /// correct model these are iid N(0, 1), which is what residual
    /// diagnostics should be run on. Length `n - d`.
    ///
    /// # Errors
    ///
    /// [`ArimaError::Ssm`] if filtering at the stored parameters fails
    /// (cannot happen for parameters produced by this crate's fits).
    pub fn residuals(&self) -> Result<Vec<f64>, ArimaError> {
        let model = self.model()?;
        let n = self.x.len();
        let y_mat = Mat::from_fn(n, 1, |i, _| self.x[i]);
        let out = model.filter(y_mat.as_ref())?;
        let mut resid = Vec::with_capacity(n);
        for t in 0..n {
            let v = self.x[t] - out.predicted_state[t][0];
            let f = out.predicted_state_cov[t][(0, 0)];
            if !f.is_finite() || f <= 0.0 {
                return Err(ArimaError::NonFinite {
                    what: "one-step prediction variance F_t",
                });
            }
            resid.push(v / f.sqrt());
        }
        Ok(resid)
    }

    /// Out-of-sample forecasts of the *levels* `y_{n+1..n+steps}` with
    /// standard errors, via the state-space prediction recursion.
    ///
    /// Starting from the filtered moments `(a_{T|T}, P_{T|T})` of the
    /// ARMA state, the recursion iterates `a <- c + T a`, `P <- T P T' +
    /// R Q R'` (Durbin & Koopman 2012, section 4.11). For `d > 0` the
    /// state is augmented with `d` exact cumulator states carrying the
    /// partial sums back to levels, so the reported variance is the
    /// correct *cumulative* forecast-error variance
    /// (`Var[sum_j (Delta^d y - forecast)]`, including all
    /// cross-horizon covariances), not a naive sum of the differenced
    /// series' variances. For an ARIMA(0,1,0) this reproduces the
    /// random-walk `se_h = sigma sqrt(h)` exactly.
    ///
    /// Standard errors reflect innovation and filtering uncertainty only
    /// (parameters treated as known — the statsmodels `get_forecast`
    /// convention).
    ///
    /// # Errors
    ///
    /// * [`ArimaError::InvalidArgument`] for `steps == 0`;
    /// * [`ArimaError::Ssm`] if filtering at the stored parameters fails.
    pub fn forecast(&self, steps: usize) -> Result<ArimaForecast, ArimaError> {
        if steps == 0 {
            return Err(ArimaError::InvalidArgument {
                what: "forecast requires steps >= 1",
            });
        }
        let model = self.model()?;
        let n = self.x.len();
        let y_mat = Mat::from_fn(n, 1, |i, _| self.x[i]);
        let out = model.filter(y_mat.as_ref())?;

        let m = model.state_dim();
        let d = self.spec.d();
        let mm = m + d;
        let sigma2 = self.sigma2();
        let intercept = self.constant().unwrap_or(0.0);

        // Augmented transition: the ARMA block, plus one cumulator row
        // per difference order. With Z = e_1', row (m + i) carries
        // Z T = (first row of T) on the ARMA columns and ones on
        // cumulator columns m..=m+i; the disturbance loading of every
        // cumulator is Z R = 1 and its intercept Z c = c.
        let t_mat = model.t().at(0);
        let r_mat = model.r().at(0);
        let t_star = Mat::from_fn(mm, mm, |i, j| {
            if i < m && j < m {
                t_mat[(i, j)]
            } else if i >= m && j < m {
                t_mat[(0, j)]
            } else if i >= m && j >= m && j <= i {
                1.0
            } else {
                0.0
            }
        });
        let r_star: Vec<f64> = (0..mm)
            .map(|i| if i < m { r_mat[(i, 0)] } else { 1.0 })
            .collect();
        let c_star: Vec<f64> = (0..mm)
            .map(|i| if i == 0 || i >= m { intercept } else { 0.0 })
            .collect();

        // Initial augmented moments at the forecast origin T: the
        // filtered ARMA state, and the (exactly known) undifferencing
        // anchors — cumulator m + i tracks the (d-1-i)-times-differenced
        // series, so the last cumulator is the level.
        // `filtered_state` is non-empty because `difference` guarantees
        // at least one observation.
        let a_last = &out.filtered_state[n - 1];
        let p_last = &out.filtered_state_cov[n - 1];
        let mut a: Vec<f64> = (0..mm)
            .map(|i| {
                if i < m {
                    a_last[i]
                } else {
                    self.anchors[d - 1 - (i - m)]
                }
            })
            .collect();
        let mut p = Mat::from_fn(
            mm,
            mm,
            |i, j| {
                if i < m && j < m {
                    p_last[(i, j)]
                } else {
                    0.0
                }
            },
        );

        let obs_idx = if d == 0 { 0 } else { mm - 1 };
        let mut mean = Vec::with_capacity(steps);
        let mut se = Vec::with_capacity(steps);
        let mut a_next = vec![0.0; mm];
        for _ in 0..steps {
            for i in 0..mm {
                let mut s = c_star[i];
                for j in 0..mm {
                    s += t_star[(i, j)] * a[j];
                }
                a_next[i] = s;
            }
            a.copy_from_slice(&a_next);

            let mut p_next = t_star.as_ref() * p.as_ref() * t_star.as_ref().transpose();
            for i in 0..mm {
                for j in 0..mm {
                    p_next[(i, j)] += sigma2 * r_star[i] * r_star[j];
                }
            }
            // Restore exact symmetry lost to roundoff.
            for i in 0..mm {
                for j in 0..i {
                    let v = 0.5 * (p_next[(i, j)] + p_next[(j, i)]);
                    p_next[(i, j)] = v;
                    p_next[(j, i)] = v;
                }
            }
            p = p_next;

            mean.push(a[obs_idx]);
            se.push(p[(obs_idx, obs_idx)].max(0.0).sqrt());
        }
        Ok(ArimaForecast { mean, se })
    }
}

/// Point forecasts and standard errors from
/// [`ArimaResults::forecast`], in level units.
#[derive(Debug, Clone, PartialEq)]
pub struct ArimaForecast {
    /// Forecast means for horizons `1..=steps`.
    pub mean: Vec<f64>,
    /// Forecast standard errors (innovation + filtering uncertainty;
    /// parameters treated as known).
    pub se: Vec<f64>,
}

impl ArimaForecast {
    /// Symmetric Gaussian `(1 - alpha)` forecast intervals
    /// `mean_h -+ z_{1 - alpha/2} se_h` (statsmodels
    /// `get_forecast(...).conf_int(alpha)` convention).
    ///
    /// # Errors
    ///
    /// [`ArimaError::InvalidArgument`] unless `0 < alpha < 1`;
    /// [`ArimaError::Stats`] if the normal quantile fails.
    pub fn conf_int(&self, alpha: f64) -> Result<Vec<(f64, f64)>, ArimaError> {
        if !(alpha > 0.0 && alpha < 1.0) {
            return Err(ArimaError::InvalidArgument {
                what: "conf_int requires 0 < alpha < 1",
            });
        }
        let z = StdNormal.ppf(1.0 - 0.5 * alpha)?;
        Ok(self
            .mean
            .iter()
            .zip(&self.se)
            .map(|(&m, &s)| (m - z * s, m + z * s))
            .collect())
    }
}
