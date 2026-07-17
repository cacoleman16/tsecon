//! The [`GarchModel`] entry point: likelihood evaluation at fixed
//! parameters, quasi-maximum-likelihood estimation, and standard errors.

use tsecon_optim::{
    minimize, Bounded, FnObjective, Method, NelderMeadOptions, Positive, Transform,
};
use tsecon_stats::special::ln_gamma;

use crate::error::GarchError;
use crate::inference::{self, StdErrors};
use crate::recursion::{backcast, egarch_recursion, garch_recursion};
use crate::results::GarchResults;
use crate::spec::{DistSpec, GarchSpec, MeanSpec, VolSpec};

/// `arch`-compatible bounds on the Student-t degrees of freedom during
/// estimation (`arch.univariate.StudentsT.bounds`). Fixed-parameter
/// evaluation only requires `nu > 2`.
const NU_BOUNDS: (f64, f64) = (2.05, 500.0);

/// A univariate conditional-variance model bound to a return series.
///
/// ```
/// use tsecon_garch::{DistSpec, GarchModel, GarchSpec, MeanSpec, VolSpec};
///
/// let y = [0.4, -1.2, 0.3, 0.8, -0.5, 1.4, -0.9, 0.2, -0.1, 0.6,
///          -0.7, 1.1, 0.05, -0.3, 0.9, -1.5, 0.45, -0.2, 0.75, -0.6];
/// let spec = GarchSpec {
///     mean: MeanSpec::Zero,
///     vol: VolSpec::Garch { p: 1, q: 1 },
///     dist: DistSpec::Normal,
/// };
/// let model = GarchModel::new(&y, spec).unwrap();
/// let ll = model.loglike(&[0.05, 0.1, 0.8]).unwrap();
/// assert!(ll.is_finite());
/// ```
#[derive(Debug, Clone)]
pub struct GarchModel {
    spec: GarchSpec,
    y: Vec<f64>,
    /// Backcast variance from the residuals at the mean model's starting
    /// values (see [`crate::recursion`]); held fixed through estimation,
    /// as in `arch`.
    backcast: f64,
}

impl GarchModel {
    /// Binds `spec` to the observed series `y` and precomputes the
    /// backcast.
    ///
    /// # Errors
    ///
    /// * [`GarchError::InvalidSpec`] — malformed lag structure;
    /// * [`GarchError::NonFinite`] — NaN/infinity in `y`, or a
    ///   zero-variance series (backcast would be zero, and the
    ///   log-likelihood undefined);
    /// * [`GarchError::InsufficientData`] — fewer observations than
    ///   `max_lag + n_params + 1`.
    pub fn new(y: &[f64], spec: GarchSpec) -> Result<Self, GarchError> {
        spec.validate()?;
        if y.iter().any(|v| !v.is_finite()) {
            return Err(GarchError::NonFinite { what: "y" });
        }
        let needed = spec.vol.max_lag() + spec.n_params() + 1;
        if y.len() < needed {
            return Err(GarchError::InsufficientData {
                needed,
                got: y.len(),
            });
        }
        let bc = backcast(&Self::starting_resids(y, spec.mean));
        if !(bc > 0.0 && bc.is_finite()) {
            return Err(GarchError::NonFinite {
                what: "backcast (series has zero presample variance)",
            });
        }
        Ok(Self {
            spec,
            y: y.to_vec(),
            backcast: bc,
        })
    }

    /// The model specification.
    pub fn spec(&self) -> &GarchSpec {
        &self.spec
    }

    /// The observed series.
    pub fn y(&self) -> &[f64] {
        &self.y
    }

    /// The fixed backcast variance used to initialize the recursion.
    pub fn backcast_value(&self) -> f64 {
        self.backcast
    }

    /// Residuals at the mean model's starting values: `y` for a zero mean,
    /// `y - mean(y)` for a constant mean.
    fn starting_resids(y: &[f64], mean: MeanSpec) -> Vec<f64> {
        match mean {
            MeanSpec::Zero => y.to_vec(),
            MeanSpec::Constant => {
                let mu = y.iter().sum::<f64>() / y.len() as f64;
                y.iter().map(|v| v - mu).collect()
            }
        }
    }

    /// Residuals `eps_t = y_t - mu` at the given parameters.
    fn resids(&self, mean_params: &[f64]) -> Vec<f64> {
        let mu = mean_params.first().copied().unwrap_or(0.0);
        self.y.iter().map(|v| v - mu).collect()
    }

    /// The conditional-variance path `sigma2_t` at `params` (fixed
    /// backcast; see [`crate::recursion`] for the exact conventions).
    ///
    /// # Errors
    ///
    /// Parameter validation errors ([`GarchSpec::validate_params`]);
    /// [`GarchError::NonFinite`] if the recursion leaves `(0, inf)`
    /// (possible only under extreme admissible parameters).
    pub fn conditional_variance(&self, params: &[f64]) -> Result<Vec<f64>, GarchError> {
        self.spec.validate_params(params)?;
        let (mean, omega, alphas, gammas, betas, _) = self.spec.split_params(params)?;
        let resids = self.resids(mean);
        let mut sigma2 = vec![0.0; resids.len()];
        match self.spec.vol {
            VolSpec::Garch { .. } | VolSpec::Gjr { .. } => {
                garch_recursion(omega, alphas, gammas, betas, &resids, self.backcast, &mut sigma2);
            }
            VolSpec::Egarch { .. } => {
                egarch_recursion(
                    omega,
                    alphas,
                    gammas,
                    betas,
                    &resids,
                    self.backcast.ln(),
                    &mut sigma2,
                );
            }
        }
        if sigma2.iter().any(|&s| !(s > 0.0 && s.is_finite())) {
            return Err(GarchError::NonFinite {
                what: "conditional variance",
            });
        }
        Ok(sigma2)
    }

    /// Per-observation log-likelihood contributions at `params`.
    ///
    /// Normal innovations (Bollerslev-Wooldridge 1992 QMLE objective):
    ///
    /// ```text
    /// l_t = -1/2 [ ln(2 pi) + ln sigma2_t + eps_t^2 / sigma2_t ]
    /// ```
    ///
    /// Standardized Student-t innovations (Bollerslev 1987), algebraically
    /// identical to `ln f_Z(eps_t / sigma_t) - ln sigma_t` with `f_Z` the
    /// unit-variance t density of
    /// [`tsecon_stats::Standardized::student_t`]:
    ///
    /// ```text
    /// l_t = ln Gamma((nu+1)/2) - ln Gamma(nu/2) - 1/2 ln(pi (nu-2))
    ///       - 1/2 ln sigma2_t
    ///       - (nu+1)/2 * ln(1 + eps_t^2 / (sigma2_t (nu - 2)))
    /// ```
    ///
    /// Both include all constants, matching `arch`'s
    /// `Normal.loglikelihood` / `StudentsT.loglikelihood` exactly.
    ///
    /// # Errors
    ///
    /// As for [`GarchModel::conditional_variance`].
    pub fn loglike_obs(&self, params: &[f64]) -> Result<Vec<f64>, GarchError> {
        let sigma2 = self.conditional_variance(params)?;
        let (mean, _, _, _, _, dist) = self.spec.split_params(params)?;
        let resids = self.resids(mean);
        let lls = match self.spec.dist {
            DistSpec::Normal => {
                let ln2pi = (2.0 * core::f64::consts::PI).ln();
                resids
                    .iter()
                    .zip(&sigma2)
                    .map(|(&e, &s2)| -0.5 * (ln2pi + s2.ln() + e * e / s2))
                    .collect()
            }
            DistSpec::StudentT => {
                let nu = dist[0];
                let c = ln_gamma(0.5 * (nu + 1.0))
                    - ln_gamma(0.5 * nu)
                    - 0.5 * (core::f64::consts::PI * (nu - 2.0)).ln();
                resids
                    .iter()
                    .zip(&sigma2)
                    .map(|(&e, &s2)| {
                        c - 0.5 * s2.ln()
                            - 0.5 * (nu + 1.0) * (e * e / (s2 * (nu - 2.0))).ln_1p()
                    })
                    .collect()
            }
        };
        Ok(lls)
    }

    /// The total log-likelihood at `params` (sum of
    /// [`GarchModel::loglike_obs`]).
    ///
    /// # Errors
    ///
    /// As for [`GarchModel::loglike_obs`].
    pub fn loglike(&self, params: &[f64]) -> Result<f64, GarchError> {
        Ok(self.loglike_obs(params)?.iter().sum())
    }

    /// MLE and Bollerslev-Wooldridge robust standard errors at `params`
    /// (usually the fitted values) — see [`crate::inference`] for the
    /// estimators and the `arch`-compatible numerical-derivative
    /// conventions.
    ///
    /// # Errors
    ///
    /// [`GarchError::SingularHessian`] at a flat/boundary point; any
    /// likelihood error raised at a finite-difference probe.
    pub fn standard_errors(&self, params: &[f64]) -> Result<StdErrors, GarchError> {
        self.spec.validate_params(params)?;
        inference::std_errors(
            |p| self.loglike(p).map(|ll| -ll),
            |p| {
                self.loglike_obs(p)
                    .map(|lls| lls.into_iter().map(|l| -l).collect())
            },
            params,
            self.y.len(),
        )
    }

    /// Starting values by an `arch`-style grid search: candidate
    /// persistence/shock splits scaled to the sample variance of the
    /// starting residuals (log variance for EGARCH), the best candidate by
    /// log-likelihood winning.
    fn starting_values(&self) -> Result<Vec<f64>, GarchError> {
        let start_resids = Self::starting_resids(&self.y, self.spec.mean);
        let v = start_resids.iter().map(|e| e * e).sum::<f64>() / start_resids.len() as f64;
        let mu0 = match self.spec.mean {
            MeanSpec::Zero => None,
            MeanSpec::Constant => Some(self.y.iter().sum::<f64>() / self.y.len() as f64),
        };
        let (p, o, q) = self.spec.vol.lags();

        let mut vol_candidates: Vec<Vec<f64>> = Vec::new();
        match self.spec.vol {
            VolSpec::Garch { .. } | VolSpec::Gjr { .. } => {
                let alpha_totals = [0.01, 0.05, 0.1, 0.2];
                let gamma_totals: &[f64] = if o == 0 { &[0.0] } else { &[-0.04, 0.0, 0.1] };
                let persistences = [0.5, 0.7, 0.9, 0.98];
                for &a in &alpha_totals {
                    for &g in gamma_totals {
                        for &pers in &persistences {
                            let b = pers - a - 0.5 * g;
                            if b < 0.0 {
                                continue;
                            }
                            let mut cand = Vec::with_capacity(1 + p + o + q);
                            cand.push(v * (1.0 - pers));
                            cand.extend(std::iter::repeat_n(a / p as f64, p));
                            cand.extend(std::iter::repeat_n(g / o.max(1) as f64, o));
                            if q > 0 {
                                cand.extend(std::iter::repeat_n(b / q as f64, q));
                            }
                            vol_candidates.push(cand);
                        }
                    }
                }
            }
            VolSpec::Egarch { .. } => {
                let alphas = [0.05, 0.1, 0.2];
                let betas = [0.9, 0.95, 0.98, 0.99];
                for &a in &alphas {
                    for &b in &betas {
                        let mut cand = Vec::with_capacity(1 + p + o + q);
                        cand.push((1.0 - b) * v.ln());
                        cand.extend(std::iter::repeat_n(a / p as f64, p));
                        cand.extend(std::iter::repeat_n(0.0, o));
                        if q > 0 {
                            cand.extend(std::iter::repeat_n(b / q as f64, q));
                        }
                        vol_candidates.push(cand);
                    }
                }
            }
        }
        let nu_candidates: &[f64] = match self.spec.dist {
            DistSpec::Normal => &[],
            DistSpec::StudentT => &[8.0, 30.0],
        };

        let mut best: Option<(f64, Vec<f64>)> = None;
        for vol in &vol_candidates {
            let dist_options: Vec<Vec<f64>> = if nu_candidates.is_empty() {
                vec![Vec::new()]
            } else {
                nu_candidates.iter().map(|&nu| vec![nu]).collect()
            };
            for dist in dist_options {
                let mut cand = Vec::with_capacity(self.spec.n_params());
                if let Some(mu) = mu0 {
                    cand.push(mu);
                }
                cand.extend_from_slice(vol);
                cand.extend_from_slice(&dist);
                if self.spec.validate_params(&cand).is_err() {
                    continue;
                }
                if let Ok(ll) = self.loglike(&cand) {
                    if ll.is_finite() && best.as_ref().is_none_or(|(b, _)| ll > *b) {
                        best = Some((ll, cand));
                    }
                }
            }
        }
        best.map(|(_, cand)| cand)
            .ok_or(GarchError::InvalidSpec {
                what: "no admissible starting value found (degenerate series?)",
            })
    }

    /// Fits the model by quasi-maximum likelihood and returns the results
    /// object.
    ///
    /// **Constraint handling** (documented choice): the search runs in an
    /// unconstrained working space via the `tsecon-optim`
    /// reparameterization toolkit — `omega = exp(z)` ([`Positive`]) for
    /// GARCH/GJR (the EGARCH log-variance intercept is unrestricted), and
    /// `nu` through the `arch`-compatible box (2.05, 500) ([`Bounded`]);
    /// all other coordinates are untransformed, and the joint constraints
    /// (coefficient signs, persistence `< 1`; see
    /// [`GarchSpec::validate_params`]) are enforced by returning
    /// `+infinity` for inadmissible points, which every optimizer in
    /// `tsecon-optim` treats as an infeasible trial. The interior optimum
    /// of a stationary model is untouched by the barrier.
    ///
    /// **Search strategy**: `arch`-style grid starting values, L-BFGS with
    /// central-difference gradients, a Nelder-Mead polish (2 restarts),
    /// and a final L-BFGS pass, keeping the best point found. The fixture
    /// tests pin the optimum to within 1e-6 absolute log-likelihood of the
    /// `arch` package (or better).
    ///
    /// # Errors
    ///
    /// Starting-value/optimizer failures; likelihood errors at the
    /// optimum. If standard errors cannot be computed at the optimum
    /// (singular Hessian at a flat or boundary point), the fit still
    /// succeeds with NaN standard errors — flatness is reported, not
    /// hidden.
    pub fn fit(&self) -> Result<GarchResults, GarchError> {
        let sv = self.starting_values()?;
        let k = self.spec.n_params();
        let omega_idx = self.spec.n_mean_params();
        let omega_log = !matches!(self.spec.vol, VolSpec::Egarch { .. });
        let nu_idx = matches!(self.spec.dist, DistSpec::StudentT).then_some(k - 1);
        let positive = Positive;
        let nu_box = Bounded::new(NU_BOUNDS.0, NU_BOUNDS.1)?;

        let to_natural = |z: &[f64]| -> Result<Vec<f64>, GarchError> {
            let mut theta = z.to_vec();
            if omega_log {
                positive.forward(&z[omega_idx..=omega_idx], &mut theta[omega_idx..=omega_idx])?;
            }
            if let Some(i) = nu_idx {
                nu_box.forward(&z[i..=i], &mut theta[i..=i])?;
            }
            Ok(theta)
        };
        let to_working = |theta: &[f64]| -> Result<Vec<f64>, GarchError> {
            let mut z = theta.to_vec();
            if omega_log {
                positive.inverse(&theta[omega_idx..=omega_idx], &mut z[omega_idx..=omega_idx])?;
            }
            if let Some(i) = nu_idx {
                nu_box.inverse(&theta[i..=i], &mut z[i..=i])?;
            }
            Ok(z)
        };

        let mut objective = FnObjective::new(|z: &[f64]| -> f64 {
            let Ok(theta) = to_natural(z) else {
                return f64::INFINITY;
            };
            if self.spec.validate_params(&theta).is_err() {
                return f64::INFINITY;
            }
            match self.loglike(&theta) {
                Ok(ll) if ll.is_finite() => -ll,
                _ => f64::INFINITY,
            }
        });

        let z0 = to_working(&sv)?;
        let nm_opts = NelderMeadOptions {
            restarts: 2,
            max_iter: Some(20_000),
            max_fevals: Some(40_000),
            ..NelderMeadOptions::default()
        };
        let stage1 = minimize(&mut objective, &z0, &Method::lbfgs())?;
        let stage2 = minimize(&mut objective, &stage1.x, &Method::NelderMead(nm_opts))?;
        let stage3 = minimize(&mut objective, &stage2.x, &Method::lbfgs())?;
        // Each stage starts from the previous best and every optimizer
        // returns the best point it saw, so the objective is non-increasing
        // across stages; `converged` is true when at least one stage
        // terminated by a convergence criterion (the final point is at
        // least as good as that stage's).
        let converged = stage1.converged || stage2.converged || stage3.converged;
        let best = [stage1, stage2, stage3]
            .into_iter()
            .min_by(|a, b| a.f.partial_cmp(&b.f).unwrap_or(core::cmp::Ordering::Equal))
            .ok_or(GarchError::InvalidSpec {
                what: "optimization produced no result",
            })?;
        if !best.f.is_finite() {
            return Err(GarchError::NonFinite {
                what: "optimized log-likelihood",
            });
        }

        let params = to_natural(&best.x)?;
        let loglik = -best.f;
        // Flat or boundary optima can defeat the numerical Hessian; report
        // NaN standard errors rather than failing the fit.
        let se = self.standard_errors(&params).unwrap_or(StdErrors {
            mle: vec![f64::NAN; k],
            robust: vec![f64::NAN; k],
        });
        GarchResults::build(self, params, loglik, se, converged)
    }
}
