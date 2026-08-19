//! The DCS robust local level: a score-driven time-varying **level** with
//! Gaussian, Student-t, or Laplace observation density (Harvey 2013;
//! Creal-Koopman-Lucas 2013; the Student-t case is the DCS-t local level
//! of Harvey & Luati 2014, *JASA* 109, "Filtering with heavy tails").
//!
//! # Model
//!
//! ```text
//! y_t      = mu_t + eps_t,      eps_t iid ~ p(.; scale[, nu]),
//! e_t      = y_t - mu_t                       (one-step prediction error)
//! u_t      = scale^2 * d log p(y_t | mu_t) / d mu_t
//! mu_{t+1} = mu_t + kappa * u_t,              kappa >= 0.
//! ```
//!
//! `mu_t` is the *predictive* level — the one-step-ahead prediction of
//! `y_t` given `y_1..y_{t-1}` — and the driver `u_t` is the conditional
//! score of the chosen observation density, scaled by `scale^2`. That
//! scaling is a fixed multiple of the inverse-Fisher-information scaling
//! (the Fisher information for `mu` is `1/scale^2` for the Gaussian,
//! `(nu+1)/((nu+3) scale^2)` for the Student-t, and `1/scale^2` for the
//! Laplace, so `scale^2 * score` differs from `I^{-1} * score` only by a
//! constant that is absorbed into `kappa`).
//!
//! # The three drivers (analytic scores)
//!
//! * **Gaussian** — `log p = -0.5 ln(2 pi s^2) - e^2/(2 s^2)`, so
//!   `d log p/d mu = e/s^2` and
//!
//!   ```text
//!   u_t = e_t :   mu_{t+1} = mu_t + kappa e_t,
//!   ```
//!
//!   which is *exactly* the steady-state (constant-gain, innovations-form)
//!   Kalman filter of the Gaussian local level — the nested control. See
//!   the mapping below.
//!
//! * **Student-t** (`nu > 2`) — `log p = c(nu, s) - (nu+1)/2 *
//!   ln(1 + e^2/(nu s^2))` with the constant
//!   `c = ln G((nu+1)/2) - ln G(nu/2) - 0.5 ln(nu pi s^2)`, so
//!   `d log p/d mu = (nu+1) e / (nu s^2 + e^2)` and
//!
//!   ```text
//!   u_t = (nu+1) e_t / (nu + e_t^2/s^2).
//!   ```
//!
//!   The driver is **bounded and redescending**: a huge outlier moves the
//!   level by ~ `kappa (nu+1) s^2 / e -> 0`, so one bad tick is *ignored*
//!   rather than absorbed — the robustness that distinguishes this filter
//!   from the Kalman local level, whose Gaussian driver grows linearly in
//!   the outlier. As `nu -> inf`, `u_t -> e_t`, recovering the Gaussian.
//!
//! * **Laplace** (GED with shape 1) — `log p = -ln(2 b) - |e|/b` with
//!   `scale = b`, so `d log p/d mu = sign(e)/b` and
//!
//!   ```text
//!   u_t = scale * sign(e_t):
//!   ```
//!
//!   a constant-magnitude "sign filter" step — the level tracks a local
//!   *median* (Harvey 2013, ch. 3). Bounded, but not redescending.
//!
//! # Gaussian limit: the exact `kappa <-> q` mapping
//!
//! The Gaussian local level `y_t = mu_t + eps_t`,
//! `mu_{t+1} = mu_t + eta_t`, with `Var(eps) = sigma2_eps` and
//! `Var(eta) = sigma2_eta`, has signal-to-noise ratio
//! `q = sigma2_eta/sigma2_eps`. Its steady-state Riccati fixed point
//! `P = P - P^2/(P + sigma2_eps) + sigma2_eta` gives
//!
//! ```text
//! p     = P/sigma2_eps = (q + sqrt(q^2 + 4q)) / 2,
//! kappa = p/(1 + p)                (constant predictive Kalman gain),
//! q     = kappa^2 / (1 - kappa)    (inverse),
//! scale^2 = sigma2_eps (1 + p) = sigma2_eps / (1 - kappa)
//!           (one-step prediction-error variance F).
//! ```
//!
//! At those mapped parameters the DCS-Gaussian filter reproduces the
//! steady-state Kalman filter's predicted-state path, gain, and (full)
//! log-likelihood *exactly*; the golden fixture
//! `fixtures/tsecon-dcs.json` pins this against statsmodels
//! `UnobservedComponents(y, 'llevel')` run at the mapped parameters with
//! known steady-state initialization. On a finite sample the *fitted*
//! DCS-Gaussian `kappa` matches the steady-state gain implied by the
//! exact-Kalman MLE only up to the transient (the exact filter's gain
//! varies before it converges); the observed gap is ~1e-3 at `T = 500`,
//! ~1e-2 on the `T = 100` Nile series.
//!
//! # Estimation
//!
//! [`DcsModel::fit`] maximizes the exact conditional likelihood given
//! `mu_1` (initialized robustly at the median of the first ten
//! observations) by prediction-error decomposition:
//! `sum_t log p(y_t | mu_t)`. Optimization is `tsecon-optim` Nelder-Mead
//! over the unconstrained working vector `z` mapped through the house
//! [`Positive`] transform, `(kappa, scale, nu - 2) = exp(z)` — enforcing
//! `kappa > 0`, `scale > 0`, `nu > 2` — with a deterministic three-point
//! multistart over `kappa` (0.05, 0.3, 0.8), guarding the multimodality a
//! near-flat signal-to-noise surface can have.
//!
//! Standard errors are **observed information**: the numerical Hessian of
//! the negative log-likelihood at the MLE in *natural* coordinates
//! (four-point central differences with per-parameter step scales, the
//! same statsmodels-`approx_hess3` formulas the GARCH crate uses),
//! inverted; `sqrt(diag)`. Entries are NaN when the Hessian is singular
//! or a probe leaves the parameter domain (a flat or boundary optimum) —
//! reported honestly rather than clipped.
//!
//! # Honest notes
//!
//! * `converged` is the optimizer's certificate, not a quality grade. On
//!   clean *Gaussian* data the Student-t likelihood has no interior
//!   maximum in `nu` (the Gaussian is its `nu -> inf` boundary), so the
//!   fit reports `converged = false` while the level path and
//!   `kappa`/`scale` are fine — same behavior, same reason, as the
//!   volatility model in this crate ([`crate::GasModel`]).
//! * Under heavy contamination `nu_hat` is **not** an estimate of the
//!   clean noise's tail index: the fat tail is doing outlier duty, and
//!   `nu_hat` is pushed toward its lower boundary `2`. Read it as a
//!   robustness dial the data chose, not as a tail measurement.
//! * The Laplace likelihood is only piecewise smooth (the `|e|` kink and
//!   the sign driver propagate through `kappa`: every sign flip is a
//!   kink). Derivative-free Nelder-Mead handles the kinks and a denser
//!   deterministic multistart reduces trapping in shallow kink basins,
//!   but for this density `converged` certifies the best basin found,
//!   not global optimality, and the reported observed-information
//!   standard errors are a smooth-quadratic approximation — read the
//!   Laplace SEs as indicative.
//! * There is no smoother: the DCS literature filters. `level[t]` is the
//!   prediction of `y_t` given data through `t-1`, and the model's
//!   `h`-step-ahead prediction from the end of the sample is flat at
//!   [`DcsResults::next_level`] (a local level has no slope).
//!
//! Provenance: implemented from the published literature; graduated from
//! the lab prototype `lab/laplace/robust_filter.py` (measured there:
//! -22%/-31% level RMSE vs the Kalman pipeline at 5%/10% additive
//! contamination, zero clean-data tax). Deliberate divergences from the
//! lab spec: (i) the Laplace filter uses the exact hard sign both in
//! fitting and filtering (the lab smoothed `sign(e)` as `tanh(e/h)` for
//! L-BFGS-B stability; Nelder-Mead needs no smoothing, so here the
//! estimated model and the reported filter coincide exactly); (ii) the
//! Student-t degrees of freedom are constrained to `nu > 2` (the house
//! convention, finite observation variance) where the lab allowed
//! `nu in [0.8, 200]`; (iii) `kappa > 0` strictly (exp-reparameterized)
//! where the lab box-bounded `kappa in [0, 5]`.
//!
//! # References
//!
//! * Creal, Koopman & Lucas (2013), *J. Appl. Econometrics* 28(5).
//! * Harvey (2013), *Dynamic Models for Volatility and Heavy Tails*, CUP.
//! * Harvey & Luati (2014), "Filtering with Heavy Tails", *JASA*
//!   109(507), 1112-1122.
//! * Durbin & Koopman (2012), *Time Series Analysis by State Space
//!   Methods*, 2nd ed. (steady-state Kalman filter).

use tsecon_optim::{
    multistart, FnObjective, Method, NelderMeadOptions, Transform, TransformedObjective,
};
use tsecon_stats::special::ln_gamma;

use crate::error::GasError;

/// `ln(2 pi)`.
const LN_2PI: f64 = 1.837_877_066_409_345_3;

/// Minimum observations for [`DcsModel::new`]: the robust ten-point median
/// initialization plus any meaningful likelihood shape need a real sample
/// (the lab prototype was validated from `T = 30` up).
const MIN_OBS: usize = 30;

/// The observation density driving the DCS level recursion.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DcsDensity {
    /// Gaussian errors: the nested control — exactly the steady-state
    /// Kalman local level (see the module docs for the mapping).
    Gaussian,
    /// Student-t errors with `nu > 2` degrees of freedom: the
    /// Harvey-Luati (2014) DCS-t local level with a bounded, redescending
    /// driver — the robust default.
    StudentT,
    /// Laplace errors: the sign filter — the level tracks a local median.
    Laplace,
}

impl DcsDensity {
    /// Human-readable name, for diagnostics.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            DcsDensity::Gaussian => "gaussian",
            DcsDensity::StudentT => "student-t",
            DcsDensity::Laplace => "laplace",
        }
    }

    /// Whether this density carries a degrees-of-freedom parameter.
    #[must_use]
    pub fn needs_dof(self) -> bool {
        matches!(self, DcsDensity::StudentT)
    }

    /// Number of estimated parameters (`kappa`, `scale`, and `nu` for the
    /// Student-t).
    #[must_use]
    pub fn n_params(self) -> usize {
        if self.needs_dof() {
            3
        } else {
            2
        }
    }
}

/// Parameters of the DCS local level.
///
/// The recursion is `mu_{t+1} = mu_t + kappa u_t` with `u_t` the
/// `scale^2`-scaled conditional score of the observation density (module
/// docs). `nu` is the Student-t degrees of freedom (`nu > 2`), ignored
/// (and conventionally NaN) for the Gaussian and Laplace densities.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DcsParams {
    /// Score loading `kappa >= 0` — the constant gain. In the Gaussian
    /// case this *is* the steady-state Kalman gain.
    pub kappa: f64,
    /// Scale of the observation density, `scale > 0`: the standard
    /// deviation for the Gaussian (equal to the one-step prediction-error
    /// standard deviation), the `t` scale parameter (not the standard
    /// deviation: `Var(eps) = scale^2 nu/(nu-2)`) for the Student-t, and
    /// the Laplace scale `b` (`Var(eps) = 2 b^2`).
    pub scale: f64,
    /// Degrees of freedom `nu > 2` for the Student-t density; NaN
    /// otherwise.
    pub nu: f64,
}

impl DcsParams {
    /// Gaussian-density parameters (`nu` set to NaN, unused).
    #[must_use]
    pub fn gaussian(kappa: f64, scale: f64) -> Self {
        Self {
            kappa,
            scale,
            nu: f64::NAN,
        }
    }

    /// Student-t-density parameters.
    #[must_use]
    pub fn student_t(kappa: f64, scale: f64, nu: f64) -> Self {
        Self { kappa, scale, nu }
    }

    /// Laplace-density parameters (`nu` set to NaN, unused).
    #[must_use]
    pub fn laplace(kappa: f64, scale: f64) -> Self {
        Self {
            kappa,
            scale,
            nu: f64::NAN,
        }
    }
}

/// Validate the DCS parameters for a density, returning a clear error
/// rather than propagating a NaN through the recursion.
fn validate_dcs_params(density: DcsDensity, params: &DcsParams) -> Result<(), GasError> {
    let DcsParams { kappa, scale, nu } = *params;
    if !(kappa.is_finite() && kappa >= 0.0) {
        return Err(GasError::InvalidParameter {
            name: "kappa",
            value: kappa,
            requirement: "kappa >= 0 (finite)",
        });
    }
    if !(scale.is_finite() && scale > 0.0) {
        return Err(GasError::InvalidParameter {
            name: "scale",
            value: scale,
            requirement: "scale > 0 (finite)",
        });
    }
    if density.needs_dof() && !(nu.is_finite() && nu > 2.0) {
        return Err(GasError::InvalidParameter {
            name: "nu",
            value: nu,
            requirement: "nu > 2 (finite)",
        });
    }
    Ok(())
}

/// The output of the DCS level filter at fixed parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct DcsFiltered {
    /// The predictive level path `mu_1, ..., mu_N`: `level[t]` is the
    /// one-step-ahead prediction of `y_t` given `y_1..y_{t-1}`.
    pub level: Vec<f64>,
    /// The score drivers `u_1, ..., u_N` (module docs).
    pub scores: Vec<f64>,
    /// The out-of-sample one-step prediction `mu_{N+1}` (deterministic
    /// given the path and the last observation; the `h`-step prediction
    /// of a local level is flat at this value).
    pub next_level: f64,
    /// The total log-likelihood `sum_t log p(y_t | mu_t)` (exact
    /// conditional on `mu_1`).
    pub loglik: f64,
}

/// Observed-information standard errors of the DCS parameters.
///
/// Each entry is `sqrt` of the corresponding diagonal of the inverse
/// numerical Hessian of the negative log-likelihood at the MLE, in
/// natural coordinates. Entries are NaN when unavailable (singular
/// Hessian, a probe leaving the domain, or a negative diagonal at a flat
/// or boundary optimum) and for parameters the density does not have
/// (`nu` outside the Student-t).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DcsStdErrors {
    /// Standard error of `kappa`.
    pub kappa: f64,
    /// Standard error of `scale`.
    pub scale: f64,
    /// Standard error of `nu` (NaN unless Student-t).
    pub nu: f64,
}

/// The result of maximum-likelihood estimation of the DCS local level.
#[derive(Debug, Clone, PartialEq)]
pub struct DcsResults {
    /// The estimated parameters.
    pub params: DcsParams,
    /// Observed-information standard errors (see [`DcsStdErrors`]).
    pub se: DcsStdErrors,
    /// The observation density that was estimated.
    pub density: DcsDensity,
    /// The maximized log-likelihood.
    pub loglik: f64,
    /// The predictive level path at the MLE (`level[t]` predicts `y_t`).
    pub level: Vec<f64>,
    /// The one-step prediction errors `e_t = y_t - mu_t` at the MLE.
    pub resid: Vec<f64>,
    /// The out-of-sample one-step prediction `mu_{N+1}` at the MLE.
    pub next_level: f64,
    /// The number of observations.
    pub n_obs: usize,
    /// Whether the best multistart run's optimizer reported convergence.
    /// This is the optimizer's certificate, not a fit grade: a Student-t
    /// fit on effectively Gaussian data has its optimum at the `nu -> inf`
    /// boundary and honestly reports `false` (module docs).
    pub converged: bool,
    /// Total optimizer iterations across the multistart runs.
    pub iterations: usize,
    /// Total objective evaluations across the multistart runs.
    pub fevals: usize,
}

impl DcsResults {
    /// The number of estimated parameters (`2`, or `3` for Student-t).
    #[must_use]
    pub fn n_params(&self) -> usize {
        self.density.n_params()
    }

    /// The Akaike information criterion `2k - 2 loglik`.
    #[must_use]
    pub fn aic(&self) -> f64 {
        2.0 * self.n_params() as f64 - 2.0 * self.loglik
    }

    /// The Bayesian information criterion `k ln(N) - 2 loglik`.
    #[must_use]
    pub fn bic(&self) -> f64 {
        self.n_params() as f64 * (self.n_obs as f64).ln() - 2.0 * self.loglik
    }
}

/// A DCS local-level model bound to a series.
///
/// Construct with [`DcsModel::new`], then [`filter`](DcsModel::filter) at
/// fixed parameters, [`loglike`](DcsModel::loglike) for the likelihood
/// alone, or [`fit`](DcsModel::fit) for maximum-likelihood estimation.
///
/// ```
/// use tsecon_gas::{DcsDensity, DcsModel, DcsParams};
///
/// let y: Vec<f64> = (0..60).map(|t| (t as f64 * 0.7).sin()).collect();
/// let model = DcsModel::new(&y, DcsDensity::StudentT).unwrap();
/// let out = model.filter(&DcsParams::student_t(0.3, 0.5, 6.0)).unwrap();
/// assert_eq!(out.level.len(), y.len());
/// assert!(out.loglik.is_finite());
/// ```
#[derive(Debug, Clone)]
pub struct DcsModel<'a> {
    y: &'a [f64],
    density: DcsDensity,
    /// Robust initial level: median of the first ten observations.
    mu0: f64,
    /// Robust innovation-scale guess (MAD of first differences), used only
    /// for starting values.
    s_rob: f64,
}

impl<'a> DcsModel<'a> {
    /// Bind the model to the series `y` under observation density
    /// `density`.
    ///
    /// # Errors
    ///
    /// * [`GasError::InsufficientData`] — fewer than 30 observations;
    /// * [`GasError::NonFinite`] — `y` contains a NaN or infinity;
    /// * [`GasError::DegenerateLevel`] — `y` is constant, so the
    ///   innovation scale has no maximum-likelihood estimate (the
    ///   likelihood is unbounded above as `scale -> 0`).
    pub fn new(y: &'a [f64], density: DcsDensity) -> Result<Self, GasError> {
        if y.len() < MIN_OBS {
            return Err(GasError::InsufficientData {
                needed: MIN_OBS,
                got: y.len(),
            });
        }
        if y.iter().any(|v| !v.is_finite()) {
            return Err(GasError::NonFinite { what: "y" });
        }
        // A constant series leaves the level model with nothing to
        // measure: mu_1 equals every observation, every prediction error
        // is exactly zero, and the likelihood grows without bound as the
        // scale is driven to zero — no MLE exists. Diagnose it rather
        // than letting the optimizer certify wherever its floor stopped.
        let first = y[0];
        if y.iter().all(|&v| v == first) {
            return Err(GasError::DegenerateLevel {
                what: "every observation is identical",
            });
        }
        let mu0 = median_of_prefix(y, 10);
        let s_rob = robust_scale_diff(y);
        Ok(Self {
            y,
            density,
            mu0,
            s_rob,
        })
    }

    /// The observation density.
    #[must_use]
    pub fn density(&self) -> DcsDensity {
        self.density
    }

    /// The number of observations.
    #[must_use]
    pub fn n_obs(&self) -> usize {
        self.y.len()
    }

    /// The robust initial level `mu_1` (median of the first ten
    /// observations) every filter pass starts from.
    #[must_use]
    pub fn mu_init(&self) -> f64 {
        self.mu0
    }

    /// Run the DCS level filter at fixed parameters.
    ///
    /// Returns the predictive level path, the score drivers, the
    /// out-of-sample one-step prediction `mu_{N+1}`, and the total
    /// log-likelihood (exact conditional on `mu_1`).
    ///
    /// # Errors
    ///
    /// * [`GasError::InvalidParameter`] — a parameter is out of domain;
    /// * [`GasError::NonFinite`] — the filter produced a non-finite value
    ///   (an explosive `kappa` at this scale).
    pub fn filter(&self, params: &DcsParams) -> Result<DcsFiltered, GasError> {
        validate_dcs_params(self.density, params)?;
        let DcsParams { kappa, scale, nu } = *params;
        let n = self.y.len();
        let s2 = scale * scale;

        let mut level = vec![0.0_f64; n];
        let mut scores = vec![0.0_f64; n];
        let mut loglik = 0.0_f64;
        let mut m = self.mu0;

        match self.density {
            DcsDensity::Gaussian => {
                // log p = -0.5 ln(2 pi s^2) - e^2/(2 s^2); u = e.
                let c = -0.5 * (LN_2PI + s2.ln());
                for (t, &yt) in self.y.iter().enumerate() {
                    level[t] = m;
                    let e = yt - m;
                    loglik += c - 0.5 * e * e / s2;
                    scores[t] = e;
                    m += kappa * e;
                }
            }
            DcsDensity::StudentT => {
                // log p = c(nu, s) - (nu+1)/2 ln(1 + e^2/(nu s^2));
                // u = (nu+1) e / (nu + e^2/s^2)  (redescending).
                let c = ln_gamma(0.5 * (nu + 1.0))
                    - ln_gamma(0.5 * nu)
                    - 0.5 * (nu * core::f64::consts::PI * s2).ln();
                for (t, &yt) in self.y.iter().enumerate() {
                    level[t] = m;
                    let e = yt - m;
                    let z2 = e * e / s2;
                    loglik += c - 0.5 * (nu + 1.0) * (z2 / nu).ln_1p();
                    let u = (nu + 1.0) * e / (nu + z2);
                    scores[t] = u;
                    m += kappa * u;
                }
            }
            DcsDensity::Laplace => {
                // log p = -ln(2 b) - |e|/b; u = b sign(e), the exact hard
                // sign (sign(0) = 0, matching the score's subgradient
                // convention and NumPy's `sign`).
                let c = -(2.0 * scale).ln();
                for (t, &yt) in self.y.iter().enumerate() {
                    level[t] = m;
                    let e = yt - m;
                    loglik += c - e.abs() / scale;
                    let sgn = if e == 0.0 { 0.0 } else { e.signum() };
                    let u = scale * sgn;
                    scores[t] = u;
                    m += kappa * u;
                }
            }
        }
        let next_level = m;

        if !loglik.is_finite() || !next_level.is_finite() || level.iter().any(|v| !v.is_finite()) {
            return Err(GasError::NonFinite {
                what: "filtered level path or log-likelihood",
            });
        }

        Ok(DcsFiltered {
            level,
            scores,
            next_level,
            loglik,
        })
    }

    /// The total log-likelihood at fixed parameters (a thin wrapper over
    /// [`filter`](DcsModel::filter)).
    ///
    /// # Errors
    ///
    /// Same as [`filter`](DcsModel::filter).
    pub fn loglike(&self, params: &DcsParams) -> Result<f64, GasError> {
        Ok(self.filter(params)?.loglik)
    }

    /// Maximum-likelihood estimation.
    ///
    /// **Scale-adaptive** (audit round 7): the optimizer runs on the
    /// internally standardized series `y / s_rob` (the robust MAD scale
    /// of the first differences — the same quantity the starting values
    /// already use) and the optimum is mapped back through the exact
    /// reparameterization `y -> c y` (`scale -> c scale`; `kappa`, `nu`,
    /// and the whole score recursion are unit-free). Rescaling the data
    /// is a pure relabeling of the model, so the estimator must commute
    /// with it; without this the Laplace sign filter — whose likelihood
    /// is piecewise in `kappa` — landed in *different kink basins
    /// depending on the units of `y`* (measured: 11/20 seeded series
    /// moved `kappa` by up to 57% across eight decades of rescaling, with
    /// mapped log-likelihood gaps up to 4.6; the smooth t/Gaussian
    /// likelihoods moved 0/20). For power-of-two rescalings the
    /// standardization is an exact exponent shift, so the fit commutes
    /// bit-exactly. The reported log-likelihood, level path, residuals,
    /// and standard errors are all evaluated at the mapped parameters on
    /// the original data.
    ///
    /// Optimizes the mean negative log-likelihood by Nelder-Mead over the
    /// unconstrained working vector `z` mapped through the house
    /// [`Positive`](tsecon_optim::Positive) transform,
    /// `(kappa, scale, nu - 2) = exp(z)`, from a deterministic
    /// multistart over `kappa` (`scale` starts at the (standardized) MAD
    /// of the first differences, `nu` at 8). Standard errors are
    /// observed-information (module docs).
    ///
    /// # Errors
    ///
    /// * [`GasError::Optim`] — every start failed to produce a finite
    ///   objective;
    /// * errors from [`filter`](DcsModel::filter) at the optimum.
    pub fn fit(&self) -> Result<DcsResults, GasError> {
        let s = if self.s_rob.is_finite() && self.s_rob > 0.0 {
            self.s_rob
        } else {
            1.0
        };
        let (params, converged, iterations, fevals) = if s == 1.0 {
            self.optimize()?
        } else {
            let scaled: Vec<f64> = self.y.iter().map(|v| v / s).collect();
            let inner = DcsModel::new(&scaled, self.density)?;
            let (p, converged, iterations, fevals) = inner.optimize()?;
            (
                DcsParams {
                    kappa: p.kappa,
                    scale: p.scale * s,
                    nu: p.nu,
                },
                converged,
                iterations,
                fevals,
            )
        };

        // Everything reported is evaluated at the mapped parameters on the
        // ORIGINAL data, so every output carries the caller's units.
        let filtered = self.filter(&params)?;
        let resid: Vec<f64> = self
            .y
            .iter()
            .zip(&filtered.level)
            .map(|(&yt, &mt)| yt - mt)
            .collect();
        let se = self.observed_information_se(&params);

        Ok(DcsResults {
            params,
            se,
            density: self.density,
            loglik: filtered.loglik,
            level: filtered.level,
            resid,
            next_level: filtered.next_level,
            n_obs: self.y.len(),
            converged,
            iterations,
            fevals,
        })
    }

    /// The multistart Nelder-Mead search on this model's own data;
    /// returns the best parameters (in this model's units), the
    /// convergence certificate, and the iteration/evaluation totals.
    /// [`DcsModel::fit`] calls it on the internally standardized model.
    fn optimize(&self) -> Result<(DcsParams, bool, usize, usize), GasError> {
        let density = self.density;

        // Working space: theta = (kappa, scale, nu - 2) = exp(z), via the
        // house Positive transform; z0 at kappa = 0.05, scale = s_rob,
        // nu = 8.
        let transform = tsecon_optim::Positive;
        let mut theta0 = vec![0.05, self.s_rob];
        if density.needs_dof() {
            theta0.push(8.0 - 2.0);
        }
        let z0 = transform.inverse_vec(&theta0)?;

        let n = self.y.len() as f64;
        let model = self.clone();
        let inner = FnObjective::new(move |theta: &[f64]| {
            let params = params_from_natural(density, theta);
            match model.filter(&params) {
                Ok(out) if out.loglik.is_finite() => -out.loglik / n,
                _ => f64::INFINITY,
            }
        });
        let mut objective = TransformedObjective::new(inner, transform);

        let nm = NelderMeadOptions {
            restarts: 2,
            ..NelderMeadOptions::default()
        };
        let method = Method::NelderMead(nm);
        // Deterministic kappa multistart (z[0] overwritten per start; the
        // first start is z0 itself at kappa = 0.05). The Laplace sign
        // filter's likelihood is piecewise in kappa — every sign flip is a
        // kink — so it gets a denser grid to keep Nelder-Mead out of
        // shallow kink basins; `converged` certifies the best basin found,
        // not global optimality over the kinks.
        let kappa_starts: &[f64] = if matches!(density, DcsDensity::Laplace) {
            &[0.02, 0.1, 0.3, 0.8]
        } else {
            &[0.3, 0.8]
        };
        let ms = multistart(
            &mut objective,
            &z0,
            &method,
            1 + kappa_starts.len(),
            |k, z| {
                z[0] = kappa_starts[k - 1].ln();
            },
        )?;
        let res = ms.best;
        if !res.f.is_finite() {
            return Err(GasError::Optim(tsecon_optim::OptimError::NonFinite {
                what: "objective at the optimum",
            }));
        }

        let theta = objective.constrained(&res.x)?;
        let params = params_from_natural(density, &theta);
        Ok((params, res.converged, ms.total_iterations, ms.total_fevals))
    }

    /// Observed-information standard errors at `params`, in natural
    /// coordinates. Failures (singular Hessian, probes leaving the
    /// domain, negative diagonals) are reported as NaN entries, never as
    /// errors: the point estimate stands on its own.
    fn observed_information_se(&self, params: &DcsParams) -> DcsStdErrors {
        let density = self.density;
        let mut theta = vec![params.kappa, params.scale];
        if density.needs_dof() {
            theta.push(params.nu);
        }
        // Per-parameter step scales in each parameter's own units:
        // kappa is dimensionless with a small-gain floor, scale carries
        // the units of y (and is strictly positive at any MLE), nu is
        // dimensionless and > 2.
        let mut steps = vec![params.kappa.max(1e-4), params.scale];
        if density.needs_dof() {
            steps.push(params.nu);
        }
        let neg_ll = |th: &[f64]| -> Result<f64, GasError> {
            let p = params_from_theta(density, th);
            Ok(-self.filter(&p)?.loglik)
        };
        let nan = DcsStdErrors {
            kappa: f64::NAN,
            scale: f64::NAN,
            nu: f64::NAN,
        };
        let hess = match numerical_hessian(neg_ll, &theta, &steps) {
            Ok(h) => h,
            Err(_) => return nan,
        };
        let cov = match invert(&hess) {
            Ok(c) => c,
            Err(_) => return nan,
        };
        let se_at = |i: usize| -> f64 {
            // A negative diagonal (non-PD Hessian at a flat or boundary
            // optimum) surfaces as NaN via sqrt, kept deliberately.
            cov[i][i].sqrt()
        };
        DcsStdErrors {
            kappa: se_at(0),
            scale: se_at(1),
            nu: if density.needs_dof() {
                se_at(2)
            } else {
                f64::NAN
            },
        }
    }
}

/// Map the positive natural vector `(kappa, scale[, nu - 2])` (the image
/// of the working space under [`tsecon_optim::Positive`]) to parameters.
fn params_from_natural(density: DcsDensity, theta: &[f64]) -> DcsParams {
    DcsParams {
        kappa: theta[0],
        scale: theta[1],
        nu: if density.needs_dof() {
            2.0 + theta[2]
        } else {
            f64::NAN
        },
    }
}

/// Map `(kappa, scale[, nu])` — nu itself, not shifted — to parameters
/// (the coordinates the standard errors are computed in).
fn params_from_theta(density: DcsDensity, theta: &[f64]) -> DcsParams {
    DcsParams {
        kappa: theta[0],
        scale: theta[1],
        nu: if density.needs_dof() {
            theta[2]
        } else {
            f64::NAN
        },
    }
}

/// Median of the first `k` (or all, if fewer) observations — the robust
/// level initialization.
fn median_of_prefix(y: &[f64], k: usize) -> f64 {
    let mut head: Vec<f64> = y[..k.min(y.len())].to_vec();
    head.sort_unstable_by(f64::total_cmp);
    let n = head.len();
    if n % 2 == 1 {
        head[n / 2]
    } else {
        0.5 * (head[n / 2 - 1] + head[n / 2])
    }
}

/// MAD-based scale of the first differences (consistent for the Gaussian
/// innovation scale up to the usual 0.6745 constant, with the sqrt(2)
/// correction because differencing doubles the variance); falls back to
/// the sample standard deviation when the MAD is zero. Used only for
/// starting values.
fn robust_scale_diff(y: &[f64]) -> f64 {
    let d: Vec<f64> = y.windows(2).map(|w| w[1] - w[0]).collect();
    let med = median_all(&d);
    let abs_dev: Vec<f64> = d.iter().map(|&v| (v - med).abs()).collect();
    let mad = median_all(&abs_dev);
    let s = mad / 0.6745 / core::f64::consts::SQRT_2;
    if s > 0.0 {
        return s;
    }
    let n = y.len() as f64;
    let mean = y.iter().sum::<f64>() / n;
    (y.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>() / n).sqrt()
}

/// Median of a full slice.
fn median_all(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_unstable_by(f64::total_cmp);
    let n = s.len();
    if n % 2 == 1 {
        s[n / 2]
    } else {
        0.5 * (s[n / 2 - 1] + s[n / 2])
    }
}

/// Four-point central-difference Hessian of `f` (the negative total
/// log-likelihood) with per-coordinate step scales `h_i = eps^(1/4) *
/// step_scale[i]` — the statsmodels `approx_hess3` formulas, following
/// the GARCH crate's inference module (see its module docs for why the
/// step scales must be per-parameter, in the parameter's own units).
fn numerical_hessian<F>(mut f: F, x: &[f64], step_scale: &[f64]) -> Result<Vec<Vec<f64>>, GasError>
where
    F: FnMut(&[f64]) -> Result<f64, GasError>,
{
    let n = x.len();
    let h: Vec<f64> = step_scale
        .iter()
        .map(|&s| f64::EPSILON.powf(0.25) * s)
        .collect();
    let mut hess = vec![vec![0.0; n]; n];
    let mut probe = x.to_vec();
    let mut eval = |probe: &mut Vec<f64>, di: (usize, f64), dj: (usize, f64)| {
        probe.copy_from_slice(x);
        probe[di.0] += di.1;
        probe[dj.0] += dj.1;
        f(probe)
    };
    for i in 0..n {
        for j in i..n {
            let fpp = eval(&mut probe, (i, h[i]), (j, h[j]))?;
            let fpm = eval(&mut probe, (i, h[i]), (j, -h[j]))?;
            let fmp = eval(&mut probe, (i, -h[i]), (j, h[j]))?;
            let fmm = eval(&mut probe, (i, -h[i]), (j, -h[j]))?;
            let v = ((fpp - fpm) - (fmp - fmm)) / (4.0 * h[i] * h[j]);
            hess[i][j] = v;
            hess[j][i] = v;
        }
    }
    Ok(hess)
}

/// Invert a small symmetric matrix by Gauss-Jordan elimination with
/// partial pivoting (2x2 or 3x3 here).
fn invert(a: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, GasError> {
    let n = a.len();
    let mut m: Vec<Vec<f64>> = a
        .iter()
        .enumerate()
        .map(|(i, row)| {
            let mut r = row.clone();
            r.extend((0..n).map(|j| if i == j { 1.0 } else { 0.0 }));
            r
        })
        .collect();
    for col in 0..n {
        let pivot_row = (col..n)
            .max_by(|&r1, &r2| {
                m[r1][col]
                    .abs()
                    .partial_cmp(&m[r2][col].abs())
                    .unwrap_or(core::cmp::Ordering::Equal)
            })
            .ok_or(GasError::NonFinite {
                what: "observed-information Hessian (empty)",
            })?;
        let pivot = m[pivot_row][col];
        if !pivot.is_finite() || pivot.abs() < 1e-300 {
            return Err(GasError::NonFinite {
                what: "observed-information Hessian (singular)",
            });
        }
        m.swap(col, pivot_row);
        for v in m[col].iter_mut() {
            *v /= pivot;
        }
        let pivot_vals = m[col].clone();
        for (r, row) in m.iter_mut().enumerate() {
            if r == col {
                continue;
            }
            let factor = row[col];
            if factor != 0.0 {
                for (v, &pv) in row.iter_mut().zip(&pivot_vals) {
                    *v -= factor * pv;
                }
            }
        }
    }
    Ok(m.into_iter().map(|row| row[n..].to_vec()).collect())
}

/// The steady-state Kalman gain of the Gaussian local level with
/// signal-to-noise ratio `q = sigma2_eta / sigma2_eps` — the exact
/// `q -> kappa` half of the mapping in the module docs:
/// `p = (q + sqrt(q^2 + 4q))/2`, `kappa = p/(1+p)`.
///
/// # Errors
///
/// [`GasError::InvalidParameter`] unless `q >= 0` (finite).
pub fn steady_state_gain(q: f64) -> Result<f64, GasError> {
    if !(q.is_finite() && q >= 0.0) {
        return Err(GasError::InvalidParameter {
            name: "q",
            value: q,
            requirement: "q >= 0 (finite)",
        });
    }
    let p = 0.5 * (q + (q * q + 4.0 * q).sqrt());
    Ok(p / (1.0 + p))
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use tsecon_stats::ContinuousDist;

    #[test]
    fn student_t_log_density_matches_stats_crate() {
        // The Student-t observation density is StudentT.ln_pdf(e/s) - ln s.
        let (nu, e, s) = (5.0_f64, 0.7_f64, 1.3_f64);
        let expected = tsecon_stats::StudentT::new(nu).unwrap().ln_pdf(e / s) - s.ln();
        let c = ln_gamma(0.5 * (nu + 1.0))
            - ln_gamma(0.5 * nu)
            - 0.5 * (nu * core::f64::consts::PI * s * s).ln();
        let ours = c - 0.5 * (nu + 1.0) * (e * e / (nu * s * s)).ln_1p();
        assert!((ours - expected).abs() < 1e-13, "{ours} vs {expected}");
    }

    #[test]
    fn t_driver_is_redescending_and_nests_gaussian() {
        let (s, nu) = (1.0, 5.0);
        let u = |e: f64, nu: f64| (nu + 1.0) * e / (nu + e * e / (s * s));
        // Redescending: beyond the mode of the psi-function, larger errors
        // move the level *less*.
        assert!(u(50.0, nu) < u(3.0, nu));
        assert!(u(50.0, nu) < 0.2 * 50.0);
        // Gaussian nesting: u -> e as nu -> inf.
        assert!((u(1.7, 1e9) - 1.7).abs() < 1e-6);
    }

    #[test]
    fn steady_state_gain_round_trips() {
        for q in [1e-4, 0.01, 0.1, 1.0, 10.0] {
            let k = steady_state_gain(q).unwrap();
            assert!((0.0..1.0).contains(&k));
            // inverse: q = kappa^2/(1-kappa)
            assert!((q - k * k / (1.0 - k)).abs() < 1e-12 * q.max(1.0));
        }
        assert!((steady_state_gain(0.0).unwrap()).abs() < 1e-15);
        assert!(steady_state_gain(-1.0).is_err());
    }

    #[test]
    fn median_and_mad_helpers() {
        assert_eq!(median_of_prefix(&[3.0, 1.0, 2.0, 100.0], 3), 2.0);
        let y: Vec<f64> = (0..40).map(|t| t as f64).collect();
        // first differences all 1.0 -> MAD 0 -> std fallback, positive.
        assert!(robust_scale_diff(&y) > 0.0);
    }

    #[test]
    fn constant_series_is_refused() {
        let y = vec![2.5; 100];
        let err = DcsModel::new(&y, DcsDensity::StudentT).unwrap_err();
        assert!(matches!(err, GasError::DegenerateLevel { .. }));
        let msg = err.to_string();
        assert!(msg.contains("identical"), "{msg}");
        assert!(msg.contains("unbounded"), "{msg}");
    }

    #[test]
    fn parameter_domains_are_enforced() {
        let y: Vec<f64> = (0..50).map(|t| (t as f64 * 0.31).sin()).collect();
        let m = DcsModel::new(&y, DcsDensity::StudentT).unwrap();
        assert!(m.filter(&DcsParams::student_t(-0.1, 1.0, 5.0)).is_err());
        assert!(m.filter(&DcsParams::student_t(0.1, 0.0, 5.0)).is_err());
        assert!(m.filter(&DcsParams::student_t(0.1, 1.0, 2.0)).is_err());
        let g = DcsModel::new(&y, DcsDensity::Gaussian).unwrap();
        assert!(g.filter(&DcsParams::gaussian(0.1, 1.0)).is_ok());
        let short = vec![1.0, 2.0, 3.0];
        assert!(matches!(
            DcsModel::new(&short, DcsDensity::Gaussian).unwrap_err(),
            GasError::InsufficientData { needed: 30, .. }
        ));
    }
}
