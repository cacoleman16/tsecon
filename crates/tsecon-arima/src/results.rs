//! Fitted-model results: parameters, information criteria, forecasting
//! with undifferencing, and standardized residuals.

use tsecon_linalg::faer::Mat;
use tsecon_ssm::LinearGaussianSSM;
use tsecon_stats::dist::ContinuousDist;
use tsecon_stats::StdNormal;

use crate::cov::{observed_information, ParamCov};
use crate::error::ArimaError;
use crate::estimate::css_ssr;
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
    /// The ARMA estimation sample (the fully differenced data).
    x: Vec<f64>,
    /// Regular undifferencing anchors (see [`crate::diff`]).
    anchors: Vec<f64>,
    /// Seasonal undifferencing anchors (see [`crate::diff`]).
    seasonal_anchors: Vec<Vec<f64>>,
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
        seasonal_anchors: Vec<Vec<f64>>,
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
            seasonal_anchors,
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

    /// The fitted seasonal AR coefficients `Phi_1..Phi_P` (empty for a
    /// non-seasonal specification).
    pub fn seasonal_ar(&self) -> &[f64] {
        let at = usize::from(self.spec.include_constant()) + self.spec.p() + self.spec.q();
        &self.params[at..at + self.spec.seasonal_p()]
    }

    /// The fitted seasonal MA coefficients `Theta_1..Theta_Q` (empty for
    /// a non-seasonal specification).
    pub fn seasonal_ma(&self) -> &[f64] {
        let at = usize::from(self.spec.include_constant())
            + self.spec.p()
            + self.spec.q()
            + self.spec.seasonal_p();
        &self.params[at..at + self.spec.seasonal_q()]
    }

    /// The fitted innovation variance `sigma2`.
    pub fn sigma2(&self) -> f64 {
        self.params[self.params.len() - 1]
    }

    /// The multiplied-out dense AR and MA coefficient vectors at the
    /// fitted parameters — `phi(L)Phi(L^s)` and `theta(L)Theta(L^s)`.
    fn expanded_polys(&self) -> (Vec<f64>, Vec<f64>) {
        (
            crate::spec::multiply_lag_polys(
                self.ar(),
                self.seasonal_ar(),
                self.spec.period(),
                -1.0,
            ),
            crate::spec::multiply_lag_polys(self.ma(), self.seasonal_ma(), self.spec.period(), 1.0),
        )
    }

    /// The state-space form at the fitted parameters, with the intercept
    /// overridden — the hook the drift-uncertainty derivative needs
    /// (everything else about the model is held fixed).
    fn model_at(&self, intercept: f64) -> Result<LinearGaussianSSM, ArimaError> {
        let (ar_full, ma_full) = self.expanded_polys();
        arma_ssm(&ar_full, &ma_full, self.sigma2(), intercept)
    }

    /// The state-space form at the fitted parameters.
    fn model(&self) -> Result<LinearGaussianSSM, ArimaError> {
        self.model_at(self.constant().unwrap_or(0.0))
    }

    /// Negative total log-likelihood at an arbitrary packed parameter
    /// vector, using the objective that this fit's
    /// [`EstimationMethod`] maximized: the exact Kalman likelihood for
    /// [`EstimationMethod::ExactMle`] and [`EstimationMethod::Fixed`],
    /// the conditional (CSS) Gaussian likelihood
    ///
    /// ```text
    /// -l = n_c/2 (ln 2*pi + ln sigma2) + SSR / (2 sigma2)
    /// ```
    ///
    /// for [`EstimationMethod::Css`], with `sigma2` free rather than
    /// concentrated out (the CSS optimum is a stationary point of this
    /// in `sigma2` too, since `sigma2_hat = SSR / n_c`).
    ///
    /// Differentiating the wrong objective is the classic way to get a
    /// covariance that is not the covariance of the estimator that was
    /// actually computed, which is why this dispatches on the method.
    fn neg_loglik_at(&self, params: &[f64]) -> Result<f64, ArimaError> {
        let blocks = self.spec.unpack(params)?;
        let (ar_full, ma_full) = blocks.expanded(self.spec.period());
        match self.method {
            EstimationMethod::ExactMle | EstimationMethod::Fixed => {
                let model = arma_ssm(&ar_full, &ma_full, blocks.sigma2, blocks.constant)?;
                let n = self.x.len();
                let y_mat = Mat::from_fn(n, 1, |i, _| self.x[i]);
                let ll = model.loglike(y_mat.as_ref())?;
                if ll.is_finite() {
                    Ok(-ll)
                } else {
                    Err(ArimaError::NonFinite {
                        what: "the exact log-likelihood at a probe parameter vector",
                        at: None,
                    })
                }
            }
            EstimationMethod::Css => {
                let (ssr, n_c) = css_ssr(&self.x, blocks.constant, &ar_full, &ma_full).ok_or(
                    ArimaError::NonFinite {
                        what: "the conditional-sum-of-squares recursion at a probe \
                               parameter vector",
                        at: None,
                    },
                )?;
                let n_c = n_c as f64;
                let s2 = blocks.sigma2;
                let neg =
                    0.5 * n_c * ((2.0 * std::f64::consts::PI).ln() + s2.ln()) + ssr / (2.0 * s2);
                if neg.is_finite() {
                    Ok(neg)
                } else {
                    Err(ArimaError::NonFinite {
                        what: "the conditional log-likelihood at a probe parameter vector",
                        at: None,
                    })
                }
            }
        }
    }

    /// Parameter covariance from the observed information — the inverse
    /// of the negative numerical Hessian of the log-likelihood at the
    /// reported parameters, in the natural (untransformed) parameter
    /// space `[const?, ar.., ma.., sigma2]`.
    ///
    /// This is the statsmodels `SARIMAX(...).fit(cov_type='approx')`
    /// estimator; see [`crate::cov`] for the step rules and the measured
    /// agreement. It is recomputed on each call (`k(k+1)/2` groups of
    /// four likelihood evaluations), so hold onto the result if you need
    /// it more than once.
    ///
    /// The `sigma2` slot is differentiated multiplicatively, so the
    /// standard errors are invariant to the units of the series: on
    /// ARIMA(0,1,0)+c the closed form `se(sigma2) = sqrt(2 sigma2^2 / n)`
    /// is reproduced to 1e-7 for `sigma2` anywhere from 1e-8 to 1e2. The
    /// remaining `k - 1` coordinates use the statsmodels step rule.
    ///
    /// For [`EstimationMethod::Fixed`] the supplied parameters are not
    /// generally a maximizer, so the "covariance" is the observed
    /// information at whatever point you named; it is a valid curvature
    /// summary there but only a sampling covariance if that point is the
    /// MLE.
    ///
    /// # Errors
    ///
    /// [`ArimaError::CovarianceFailed`] when a finite-difference probe
    /// leaves the admissible region (a fit that stopped on the
    /// stationarity/invertibility boundary), or when the observed
    /// information is non-finite, singular, or numerically rank-deficient
    /// (near-cancelling AR and MA roots — the parameters are not
    /// identified by this sample; lower `p` or `q`).
    pub fn param_cov(&self) -> Result<ParamCov, ArimaError> {
        // Only `sigma2` — the last slot — is a positive scale parameter.
        // The constant and the AR/MA coefficients live on the whole line.
        let mut log_scale = vec![false; self.params.len()];
        if let Some(last) = log_scale.last_mut() {
            *last = true;
        }
        observed_information(|p| self.neg_loglik_at(p), &self.params, &log_scale)
    }

    /// Parameter standard errors `sqrt(diag(cov))` in packed parameter
    /// order — the statsmodels `.bse` vector, aligned with
    /// [`ArimaResults::param_names`].
    ///
    /// An entry is NaN when its variance came out negative (a numerical
    /// Hessian that is not negative definite at the reported
    /// parameters); see [`ParamCov::se`].
    ///
    /// # Errors
    ///
    /// As for [`ArimaResults::param_cov`].
    pub fn bse(&self) -> Result<Vec<f64>, ArimaError> {
        Ok(self.param_cov()?.se().to_vec())
    }

    /// The lag-polynomial blocks whose fitted roots sit within the
    /// documented epsilon of the unit circle: `(block name, min root
    /// modulus, packed-parameter index range)`.
    ///
    /// Seasonal moduli are power-mapped by `1/s` exactly as in
    /// [`crate::auto`]'s admissibility guard, so the check on the factor
    /// polynomials is equivalent to checking the multiplied-out one.
    fn boundary_blocks(&self) -> Vec<(&'static str, f64, core::ops::Range<usize>)> {
        use crate::auto::{min_root_modulus, ROOT_ADMISSIBILITY_THRESHOLD};
        let c = usize::from(self.spec.include_constant());
        let p = self.spec.p();
        let q = self.spec.q();
        let sp = self.spec.seasonal_p();
        let sq = self.spec.seasonal_q();
        let s = self.spec.period().max(1) as f64;
        let neg = |coefs: &[f64]| -> Vec<f64> { coefs.iter().map(|v| -v).collect() };

        let mut out: Vec<(&'static str, f64, core::ops::Range<usize>)> = Vec::new();
        let mut push =
            |name: &'static str, modulus: Option<f64>, range: core::ops::Range<usize>, pw: f64| {
                if let Some(m) = modulus {
                    let m = m.powf(pw);
                    if m < ROOT_ADMISSIBILITY_THRESHOLD {
                        out.push((name, m, range));
                    }
                }
            };
        push(
            "AR (stationarity)",
            min_root_modulus(&neg(self.ar())),
            c..c + p,
            1.0,
        );
        push(
            "MA (invertibility)",
            min_root_modulus(self.ma()),
            c + p..c + p + q,
            1.0,
        );
        push(
            "seasonal AR (stationarity)",
            min_root_modulus(&neg(self.seasonal_ar())),
            c + p + q..c + p + q + sp,
            1.0 / s,
        );
        push(
            "seasonal MA (invertibility)",
            min_root_modulus(self.seasonal_ma()),
            c + p + q + sp..c + p + q + sp + sq,
            1.0 / s,
        );
        out
    }

    /// Per-parameter flags (packed order, aligned with
    /// [`ArimaResults::param_names`]): `true` when the parameter belongs
    /// to an AR or MA block whose fitted polynomial has a root within the
    /// documented epsilon of the unit circle —
    /// [`crate::auto::ROOT_ADMISSIBILITY_THRESHOLD`] (`1.001`, i.e.
    /// within 0.1% of the boundary; the same rule `auto_arima` uses to
    /// exclude candidates from selection). The constraint is on the
    /// polynomial — a *direction* in parameter space, not a coordinate —
    /// so every coefficient of a flagged block is flagged. The constant
    /// and `sigma2` never trigger.
    ///
    /// At such a boundary the observed information is singular in the
    /// constrained direction by construction, so no classical standard
    /// error exists for the flagged parameters, and their sampling
    /// distribution is non-standard (an MA root on the unit circle is
    /// the classic over-differencing pile-up). This mirrors
    /// `tsecon-garch`'s round-7 `boundary`/`se_valid` contract.
    pub fn boundary(&self) -> Vec<bool> {
        let mut flags = vec![false; self.params.len()];
        for (_, _, range) in self.boundary_blocks() {
            for f in &mut flags[range] {
                *f = true;
            }
        }
        flags
    }

    /// A teaching note describing the active stationarity/invertibility
    /// boundaries behind [`ArimaResults::boundary`]; `None` when no block
    /// is flagged.
    pub fn boundary_note(&self) -> Option<String> {
        let blocks = self.boundary_blocks();
        if blocks.is_empty() {
            return None;
        }
        let names = self.param_names();
        let mut causes: Vec<String> = Vec::new();
        let mut flagged: Vec<&str> = Vec::new();
        for (name, m, range) in &blocks {
            causes.push(format!(
                "the {name} polynomial has a root of modulus {m:.6}, within 0.1% of \
                 the unit circle"
            ));
            flagged.extend(names[range.clone()].iter().map(|n| n.as_str()));
        }
        Some(format!(
            "Boundary fit: {causes}. Parameters in a boundary block ({flagged}) have \
             no classical standard errors — the observed information is singular in \
             the constrained direction by construction, and the estimator's sampling \
             distribution there is non-standard (an MA root at the unit circle is the \
             classic over-differencing pile-up). Their bse entries are NaN with \
             se_valid false. Interior parameters' standard errors still come from the \
             full-vector observed information, which is itself degraded by the \
             boundary direction — treat them as approximate; reduced-Hessian standard \
             errors over the free directions only (the tsecon-garch treatment) are a \
             documented follow-up. If the flagged root is an MA root, the series is \
             likely over-differenced: refit with d lowered by one.",
            causes = causes.join("; "),
            flagged = flagged.join(", ")
        ))
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
                    what: "the one-step prediction variance F_t",
                    at: None,
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
    /// convention). That convention makes intervals too narrow whenever
    /// an estimated constant drives a trending forecast: on a random walk
    /// with drift (`T = 60`, nominal 95%) the measured coverage at
    /// `h = 24` is 90%. [`ArimaResults::forecast_with`] adds the missing
    /// term; this method's numbers are deliberately left unchanged so the
    /// statsmodels parity gate keeps meaning what it says.
    ///
    /// # Errors
    ///
    /// * [`ArimaError::InvalidArgument`] for `steps == 0`;
    /// * [`ArimaError::Ssm`] if filtering at the stored parameters fails.
    pub fn forecast(&self, steps: usize) -> Result<ArimaForecast, ArimaError> {
        self.forecast_at_intercept(steps, self.constant().unwrap_or(0.0))
    }

    /// The forecast recursion with the intercept overridden.
    ///
    /// Called with the fitted constant this *is* [`ArimaResults::forecast`]
    /// — same operations in the same order, so the default path is
    /// unchanged to the bit. Called at `c +- delta` it supplies the
    /// derivative the drift-uncertainty correction needs.
    fn forecast_at_intercept(
        &self,
        steps: usize,
        intercept: f64,
    ) -> Result<ArimaForecast, ArimaError> {
        if steps == 0 {
            return Err(ArimaError::InvalidArgument {
                what: "steps = 0: a forecast needs at least one step ahead; pass steps >= 1",
            });
        }
        let model = self.model_at(intercept)?;
        let n = self.x.len();
        let y_mat = Mat::from_fn(n, 1, |i, _| self.x[i]);
        let out = model.filter(y_mat.as_ref())?;

        let m = model.state_dim();
        let d = self.spec.d();
        let sd = self.spec.seasonal_d();
        let s = self.spec.period();
        let mm = m + d + sd * s;
        let sigma2 = self.sigma2();

        // Augmented transition: the ARMA block; one cumulator row per
        // regular difference order; and one s-long delay line per
        // seasonal difference order. With Z = e_1', cumulator row (m + i)
        // carries Z T = (first row of T) on the ARMA columns and ones on
        // cumulator columns m..=m+i, so cumulator m + i tracks the
        // (d-1-i)-times-regularly-differenced series and the last
        // cumulator tracks the seasonally-differenced-only series.
        //
        // Undoing a seasonal difference v_t = w_t + v_{t-s} needs the
        // last s values of v, so each seasonal stage k (from the
        // most-differenced down to the level) is a delay line
        // [v_t, v_{t-1}, .., v_{t-s+1}]: its head row is the row
        // producing the new value one stage above plus a 1 on its own
        // oldest element (v_{t+1-s}), and the remaining rows shift. The
        // innovation loading and the intercept telescope through every
        // "new current value" row exactly once (Z R = 1, Z c = c).
        let t_mat = model.t().at(0);
        let r_mat = model.r().at(0);
        let mut rows = vec![vec![0.0; mm]; mm];
        for (i, row) in rows.iter_mut().enumerate().take(m) {
            for (j, v) in row.iter_mut().enumerate().take(m) {
                *v = t_mat[(i, j)];
            }
        }
        for i in 0..d {
            for j in 0..m {
                rows[m + i][j] = t_mat[(0, j)];
            }
            for v in &mut rows[m + i][m..=m + i] {
                *v = 1.0;
            }
        }
        // The row producing the new value of the series one stage above
        // the delay line being built: the last regular cumulator, or the
        // new ARMA observation when d = 0.
        let mut chain = if d > 0 {
            rows[m + d - 1].clone()
        } else {
            let mut r = vec![0.0; mm];
            r[..m].copy_from_slice(&(0..m).map(|j| t_mat[(0, j)]).collect::<Vec<f64>>());
            r
        };
        for b in 0..sd {
            let base = m + d + b * s;
            chain[base + s - 1] += 1.0;
            rows[base] = chain.clone();
            for j in 1..s {
                rows[base + j][base + j - 1] = 1.0;
            }
        }
        let t_star = Mat::from_fn(mm, mm, |i, j| rows[i][j]);

        // "New current value" rows: the ARMA head state, every regular
        // cumulator, and the head of every seasonal delay line.
        let is_current = |i: usize| -> bool {
            i == 0 || (i >= m && i < m + d) || (i >= m + d && (i - m - d) % s == 0)
        };
        let r_star: Vec<f64> = (0..mm)
            .map(|i| {
                if i < m {
                    r_mat[(i, 0)]
                } else if is_current(i) {
                    1.0
                } else {
                    0.0
                }
            })
            .collect();
        let c_star: Vec<f64> = (0..mm)
            .map(|i| if is_current(i) { intercept } else { 0.0 })
            .collect();

        // Initial augmented moments at the forecast origin T: the
        // filtered ARMA state, and the (exactly known) undifferencing
        // anchors. Cumulator m + i takes the terminal value of the
        // (d-1-i)-times-regularly-differenced series; seasonal delay
        // line b (stage k = D-1-b, so the last line is the level) takes
        // the last s in-sample values of that stage, most recent first.
        // `filtered_state` is non-empty because `difference` guarantees
        // at least one observation.
        let a_last = &out.filtered_state[n - 1];
        let p_last = &out.filtered_state_cov[n - 1];
        let mut a: Vec<f64> = (0..mm)
            .map(|i| {
                if i < m {
                    a_last[i]
                } else if i < m + d {
                    self.anchors[d - 1 - (i - m)]
                } else {
                    let b = (i - m - d) / s;
                    let j = (i - m - d) % s;
                    let stage = &self.seasonal_anchors[sd - 1 - b];
                    // Chronological storage (most recent last) read
                    // backwards: element j holds the value j steps back.
                    stage[s - 1 - j]
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

        let obs_idx = if sd > 0 {
            m + d + (sd - 1) * s
        } else if d > 0 {
            mm - 1
        } else {
            0
        };
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

    /// Forecasts with a choice of which uncertainty sources enter the
    /// standard errors — see [`ForecastOptions`].
    ///
    /// With the default options this is exactly
    /// [`ArimaResults::forecast`]. With
    /// [`ForecastOptions::with_drift_uncertainty(true)`](ForecastOptions::with_drift_uncertainty)
    /// the standard errors additionally carry the delta-method
    /// contribution of the *estimated constant*,
    ///
    /// ```text
    /// se_h = sqrt( se_known_h^2 + (d yhat_{T+h} / d c)^2 Var(c_hat) )
    /// ```
    ///
    /// with `Var(c_hat)` the leading diagonal entry of
    /// [`ArimaResults::param_cov`]. The point forecasts are untouched;
    /// only the bands widen.
    ///
    /// **Why a finite difference is exact here.** The Kalman gains
    /// depend on `T`, `R`, `Q` and the initial `P` — never on the
    /// intercept — so the filtered state, and hence every forecast mean,
    /// is an *affine* function of `c` for any `(p, d, q)`. The central
    /// difference at `c +- delta` therefore returns the derivative to
    /// roundoff rather than to `O(delta^2)`, which is why `delta` is
    /// chosen large (`1e-3 (1 + |c|)`) — the only error to trade against
    /// is cancellation, and there is no truncation term to balance.
    ///
    /// For ARIMA(0, 1, 0) with a constant this reduces to the textbook
    /// random-walk-with-drift result
    ///
    /// ```text
    /// d yhat_{T+h} / d c = h,   Var(c_hat) = sigma2 / n,
    /// se_h = sigma sqrt(h + h^2 / n),
    /// ```
    ///
    /// `n` being the number of differenced observations. That is the
    /// term the parameters-known convention drops: at `T = 60`, `h = 24`
    /// and a nominal 95% level it is the difference between 90% and 94.5%
    /// measured coverage.
    ///
    /// Only the constant's uncertainty is added. AR/MA/`sigma2`
    /// uncertainty is a genuinely smaller, second-order effect on a
    /// trending forecast, and it is not included — the option is named
    /// for what it does.
    ///
    /// # Errors
    ///
    /// * everything [`ArimaResults::forecast`] can return;
    /// * [`ArimaError::InvalidArgument`] when `drift_uncertainty` is set
    ///   on a specification with no constant (there is no drift to be
    ///   uncertain about — the option would silently do nothing);
    /// * [`ArimaError::CovarianceFailed`] when the parameter covariance
    ///   cannot be formed, or when `Var(c_hat)` comes out negative or
    ///   non-finite.
    pub fn forecast_with(
        &self,
        steps: usize,
        options: ForecastOptions,
    ) -> Result<ArimaForecast, ArimaError> {
        let base = self.forecast(steps)?;
        if !options.drift_uncertainty {
            return Ok(base);
        }
        let c = self.constant().ok_or(ArimaError::InvalidArgument {
            what: "drift_uncertainty needs an estimated constant, but this specification \
                   has none. Refit with ArimaSpec::with_constant(true) — with no constant \
                   the forecast does not depend on an estimated drift and the correction \
                   would be identically zero",
        })?;
        let var_c = self
            .param_cov()?
            .get(0, 0)
            .ok_or(ArimaError::CovarianceFailed {
                what: "the parameter covariance is empty, so Var(c_hat) is unavailable",
            })?;
        if !var_c.is_finite() || var_c < 0.0 {
            return Err(ArimaError::CovarianceFailed {
                what: "Var(c_hat) is negative or non-finite, so the numerical Hessian is \
                       not negative definite at the reported parameters; the drift \
                       correction would be imaginary. Check that the fit converged",
            });
        }

        // The forecast mean is affine in c (see the method docs), so a
        // wide central step costs nothing in truncation and buys
        // conditioning.
        let delta = 1e-3 * (1.0 + c.abs());
        let up = self.forecast_at_intercept(steps, c + delta)?;
        let down = self.forecast_at_intercept(steps, c - delta)?;
        let se = base
            .se
            .iter()
            .zip(up.mean.iter().zip(&down.mean))
            .map(|(&s, (&mu, &md))| {
                let dydc = (mu - md) / (2.0 * delta);
                (s * s + dydc * dydc * var_c).max(0.0).sqrt()
            })
            .collect();
        Ok(ArimaForecast {
            mean: base.mean,
            se,
        })
    }
}

/// Which uncertainty sources enter the forecast standard errors of
/// [`ArimaResults::forecast_with`].
///
/// The default is the statsmodels `get_forecast` convention —
/// innovation and filtering uncertainty with the parameters treated as
/// known — so `ForecastOptions::default()` reproduces
/// [`ArimaResults::forecast`] exactly.
///
/// Marked `#[non_exhaustive]`: build it with
/// [`ForecastOptions::new`] and the `with_*` setters so that future
/// uncertainty sources can be added without breaking callers.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct ForecastOptions {
    /// Add the delta-method contribution of the estimated constant to
    /// the forecast variance (default `false`).
    pub drift_uncertainty: bool,
}

impl ForecastOptions {
    /// The statsmodels-parity defaults: no parameter uncertainty.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Toggles the estimated constant's delta-method contribution.
    #[must_use]
    pub fn with_drift_uncertainty(mut self, drift_uncertainty: bool) -> Self {
        self.drift_uncertainty = drift_uncertainty;
        self
    }
}

/// Point forecasts and standard errors from
/// [`ArimaResults::forecast`] or [`ArimaResults::forecast_with`], in
/// level units.
#[derive(Debug, Clone, PartialEq)]
pub struct ArimaForecast {
    /// Forecast means for horizons `1..=steps`. Identical whichever
    /// [`ForecastOptions`] produced them — the options only widen bands.
    pub mean: Vec<f64>,
    /// Forecast standard errors: innovation + filtering uncertainty with
    /// the parameters treated as known, plus whatever parameter
    /// uncertainty the [`ForecastOptions`] asked for (nothing, by
    /// default).
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
                what: "conf_int needs an alpha strictly inside (0, 1) — alpha = 0.05 \
                       gives a 95% interval",
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
