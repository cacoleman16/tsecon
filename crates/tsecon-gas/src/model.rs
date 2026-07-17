//! The score-driven model: filtering, maximum-likelihood estimation, and
//! multi-step variance forecasting.

use tsecon_optim::{minimize, Method, NelderMeadOptions};

use crate::error::GasError;
use crate::kernel::{log_density, scaled_score, validate_params, Density};
use crate::results::{GasFiltered, GasResults};

/// Lower floor on the filtered variance, guarding positivity when a large
/// `a` and a small realized `y^2` would otherwise drive `f_{t+1}` to zero
/// or below. Chosen far below any economically meaningful variance so it
/// never perturbs a well-specified fit.
const VAR_FLOOR: f64 = 1e-12;

/// Parameters of a GAS(1,1) time-varying-variance model.
///
/// The recursion is `f_{t+1} = omega + a * s_t + b * f_t` with `s_t` the
/// inverse-information scaled score of the observation density
/// ([`crate::kernel`]). For the [`Density::Gaussian`] model `nu` is unused;
/// for [`Density::StudentT`] it is the degrees of freedom (`nu > 2`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GasParams {
    /// Intercept `omega > 0`. The stationary mean of the filtered variance
    /// is `omega / (1 - b)` (the score has conditional mean zero, so `a`
    /// does not enter the mean).
    pub omega: f64,
    /// Score loading `a >= 0`.
    pub a: f64,
    /// Persistence `0 <= b < 1`.
    pub b: f64,
    /// Degrees of freedom `nu > 2` for the Student-t density; ignored (and
    /// conventionally `f64::NAN`) for the Gaussian.
    pub nu: f64,
}

impl GasParams {
    /// A Gaussian-model parameter triple (`nu` set to NaN, unused).
    #[must_use]
    pub fn gaussian(omega: f64, a: f64, b: f64) -> Self {
        Self {
            omega,
            a,
            b,
            nu: f64::NAN,
        }
    }

    /// A Student-t-model parameter tuple.
    #[must_use]
    pub fn student_t(omega: f64, a: f64, b: f64, nu: f64) -> Self {
        Self { omega, a, b, nu }
    }
}

/// A score-driven variance model bound to a return series.
///
/// Construct with [`GasModel::new`], then [`filter`](GasModel::filter) at
/// fixed parameters, [`loglike`](GasModel::loglike) for the log-likelihood
/// only, or [`fit`](GasModel::fit) for maximum-likelihood estimation.
#[derive(Debug, Clone)]
pub struct GasModel<'a> {
    y: &'a [f64],
    density: Density,
}

impl<'a> GasModel<'a> {
    /// Bind the model to the return series `y` under observation density
    /// `density`.
    ///
    /// # Errors
    ///
    /// * [`GasError::InsufficientData`] — fewer than two observations;
    /// * [`GasError::NonFinite`] — `y` contains a NaN or infinity.
    pub fn new(y: &'a [f64], density: Density) -> Result<Self, GasError> {
        if y.len() < 2 {
            return Err(GasError::InsufficientData {
                needed: 2,
                got: y.len(),
            });
        }
        if y.iter().any(|v| !v.is_finite()) {
            return Err(GasError::NonFinite { what: "y" });
        }
        Ok(Self { y, density })
    }

    /// The observation density.
    #[must_use]
    pub fn density(&self) -> Density {
        self.density
    }

    /// The number of observations.
    #[must_use]
    pub fn n_obs(&self) -> usize {
        self.y.len()
    }

    /// Run the score-driven filter at fixed parameters.
    ///
    /// Returns the filtered variance path `f_1, ..., f_N`, the one-step-
    /// ahead variance `f_{N+1}`, the per-step scaled scores, and the total
    /// log-likelihood `sum_t log p(y_t | f_t)`.
    ///
    /// The recursion is initialized at the stationary mean
    /// `f_1 = omega / (1 - b)` and advanced by
    /// `f_{t+1} = omega + a s_t + b f_t`, with each variance floored at
    /// [`VAR_FLOOR`] to preserve positivity.
    ///
    /// # Errors
    ///
    /// * [`GasError::InvalidParameter`] — a parameter is out of domain;
    /// * [`GasError::DofMismatch`] — Student-t without a valid `nu`;
    /// * [`GasError::NonFinite`] — the filter produced a non-finite value.
    pub fn filter(&self, params: &GasParams) -> Result<GasFiltered, GasError> {
        validate_params(self.density, params.omega, params.a, params.b, params.nu)?;
        let (omega, a, b, nu) = (params.omega, params.a, params.b, params.nu);
        let n = self.y.len();

        let mut f = vec![0.0_f64; n];
        let mut scores = vec![0.0_f64; n];
        let mut loglik = 0.0_f64;

        // Stationary-mean initialization.
        let mut ft = omega / (1.0 - b);
        for (t, &yt) in self.y.iter().enumerate() {
            f[t] = ft;
            loglik += log_density(self.density, nu, yt, ft);
            let st = scaled_score(self.density, nu, yt, ft);
            scores[t] = st;
            let mut next = omega + a * st + b * ft;
            if next < VAR_FLOOR {
                next = VAR_FLOOR;
            }
            ft = next;
        }
        let next_variance = ft;

        if !loglik.is_finite() || f.iter().any(|v| !v.is_finite()) || !next_variance.is_finite() {
            return Err(GasError::NonFinite {
                what: "filtered variance path or log-likelihood",
            });
        }

        Ok(GasFiltered {
            variance: f,
            scores,
            next_variance,
            loglik,
        })
    }

    /// The total log-likelihood at fixed parameters (a thin wrapper over
    /// [`filter`](GasModel::filter)).
    ///
    /// # Errors
    ///
    /// Same as [`filter`](GasModel::filter).
    pub fn loglike(&self, params: &GasParams) -> Result<f64, GasError> {
        Ok(self.filter(params)?.loglik)
    }

    /// Maximum-likelihood estimation by Nelder-Mead in a reparameterized
    /// working space.
    ///
    /// The constraints `omega > 0`, `a >= 0`, `0 <= b < 1`, and (for the
    /// Student-t) `nu > 2` are enforced by the smooth reparameterization
    ///
    /// ```text
    /// omega = exp(z0),  a = exp(z1),
    /// b     = 1 / (1 + exp(-z2)),         (logistic, in (0, 1))
    /// nu    = 2 + exp(z3).                 (Student-t only)
    /// ```
    ///
    /// so the optimizer works over an unconstrained `z`. Note the
    /// persistence constraint is `b < 1` alone — unlike GARCH there is no
    /// `a + b < 1` requirement, because the mean-zero score enters the
    /// recursion with loading `a` without shifting the stationary mean.
    ///
    /// Starting values target the sample second moment:
    /// `b0 = 0.9`, `a0 = 0.05`, `omega0 = (1 - b0) * mean(y^2)`, `nu0 = 8`.
    ///
    /// # Errors
    ///
    /// * [`GasError::Optim`] — the optimizer failed to produce a finite
    ///   objective;
    /// * errors from [`filter`](GasModel::filter) at the optimum.
    pub fn fit(&self) -> Result<GasResults, GasError> {
        let density = self.density;
        let dim = if density.needs_dof() { 4 } else { 3 };

        // Sample second moment for the intercept start.
        let sig2 = {
            let s: f64 = self.y.iter().map(|v| v * v).sum();
            (s / self.y.len() as f64).max(VAR_FLOOR)
        };
        let b0 = 0.9_f64;
        let a0 = 0.05_f64;
        let omega0 = (1.0 - b0) * sig2;
        let nu0 = 8.0_f64;

        let mut z0 = vec![
            omega0.ln(),
            a0.ln(),
            (b0 / (1.0 - b0)).ln(), // logit(b0)
        ];
        if density.needs_dof() {
            z0.push((nu0 - 2.0).ln());
        }
        z0.truncate(dim);

        // Objective: negative log-likelihood over the working parameters.
        // A non-finite likelihood maps to +inf so Nelder-Mead retreats.
        let y = self.y;
        let mut objective = tsecon_optim::FnObjective::new(move |z: &[f64]| {
            let params = params_from_working(density, z);
            let model = GasModel { y, density };
            match model.filter(&params) {
                Ok(out) if out.loglik.is_finite() => -out.loglik,
                _ => f64::INFINITY,
            }
        });

        let nm = NelderMeadOptions {
            restarts: 2,
            ..NelderMeadOptions::default()
        };
        let res = minimize(&mut objective, &z0, &Method::NelderMead(nm))?;
        if !res.f.is_finite() {
            return Err(GasError::Optim(tsecon_optim::OptimError::NonFinite {
                what: "objective at the optimum",
            }));
        }

        let params = params_from_working(density, &res.x);
        let filtered = self.filter(&params)?;
        let std_resid: Vec<f64> = self
            .y
            .iter()
            .zip(&filtered.variance)
            .map(|(&yt, &ft)| yt / ft.sqrt())
            .collect();

        Ok(GasResults {
            params,
            density,
            loglik: filtered.loglik,
            variance: filtered.variance,
            std_resid,
            next_variance: filtered.next_variance,
            n_obs: self.y.len(),
            converged: res.converged,
            iterations: res.iterations,
            fevals: res.fevals,
        })
    }

    /// Filter at `params` and return the `h`-step-ahead variance forecast
    /// `[f_{N+1}, ..., f_{N+h}]`.
    ///
    /// See [`forecast_from`] for the projection formula.
    ///
    /// # Errors
    ///
    /// * [`GasError::InvalidHorizon`] — `h == 0`;
    /// * errors from [`filter`](GasModel::filter).
    pub fn forecast(&self, params: &GasParams, h: usize) -> Result<Vec<f64>, GasError> {
        let filtered = self.filter(params)?;
        forecast_from(filtered.next_variance, params.omega, params.b, h)
    }
}

/// Map unconstrained working parameters `z` to model parameters. The layout
/// is `[ln omega, ln a, logit b]` for the Gaussian and additionally
/// `[..., ln(nu - 2)]` for the Student-t.
fn params_from_working(density: Density, z: &[f64]) -> GasParams {
    let omega = z[0].exp();
    let a = z[1].exp();
    let b = 1.0 / (1.0 + (-z[2]).exp());
    let nu = if density.needs_dof() {
        2.0 + z[3].exp()
    } else {
        f64::NAN
    };
    GasParams { omega, a, b, nu }
}

/// Project the `h`-step-ahead variance forecast from the one-step variance
/// `f_next = f_{N+1}` (already known at time `N`, since the score is
/// observed).
///
/// For `k >= 2` the conditional expectation of the score is zero, so
/// `E[f_{N+k} | F_N] = omega + b * E[f_{N+k-1} | F_N]`; the recursion is
/// seeded at `f_{N+1}` and converges to the stationary mean
/// `omega / (1 - b)`.
///
/// # Errors
///
/// * [`GasError::InvalidHorizon`] — `h == 0`.
pub fn forecast_from(f_next: f64, omega: f64, b: f64, h: usize) -> Result<Vec<f64>, GasError> {
    if h == 0 {
        return Err(GasError::InvalidHorizon);
    }
    let mut out = Vec::with_capacity(h);
    out.push(f_next);
    for k in 1..h {
        let prev = out[k - 1];
        out.push(omega + b * prev);
    }
    Ok(out)
}
