//! The [`GarchModel`] entry point: likelihood evaluation at fixed
//! parameters, quasi-maximum-likelihood estimation, and standard errors.

use tsecon_optim::{minimize, Bounded, Method, NelderMeadOptions, Positive, Transform};
use tsecon_stats::special::ln_gamma;

use crate::error::GarchError;
use crate::inference::{self, StdErrors};
use crate::objective::FitObjective;
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
        if let Some(index) = y.iter().position(|v| !v.is_finite()) {
            return Err(GarchError::NonFinite {
                what: "the series y",
                at: Some(index),
            });
        }
        let needed = spec.vol.max_lag() + spec.n_params() + 1;
        if y.len() < needed {
            return Err(GarchError::InsufficientData {
                needed,
                got: y.len(),
                max_lag: spec.vol.max_lag(),
                n_params: spec.n_params(),
            });
        }
        let bc = backcast(&Self::starting_resids(y, spec.mean));
        if !(bc > 0.0 && bc.is_finite()) {
            return Err(GarchError::NonFinite {
                what: "the variance backcast: the series has zero presample variance, so \
                       it is constant over the first observations and there is no \
                       volatility to model",
                at: None,
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
                garch_recursion(
                    omega,
                    alphas,
                    gammas,
                    betas,
                    &resids,
                    self.backcast,
                    &mut sigma2,
                );
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
                what: "the conditional variance recursion",
                at: None,
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
                        c - 0.5 * s2.ln() - 0.5 * (nu + 1.0) * (e * e / (s2 * (nu - 2.0))).ln_1p()
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

    /// Finite-difference step scales for [`crate::inference::std_errors`]:
    /// for each parameter, the length over which the log-likelihood varies
    /// appreciably in that direction, in that parameter's own units.
    ///
    /// statsmodels (and therefore `arch`) uses `max(|theta_i|, 0.1)` for
    /// every coordinate alike. That absolute floor is only defensible for
    /// a dimensionless parameter; applied to one carrying the units of the
    /// data it destroys the estimate — see [`crate::inference`]. The rules
    /// here differ from statsmodels' only where the units demand it, and
    /// each is equivariant under `y -> c * y`:
    ///
    /// * `mu` is in the units of `y`, so the floor is a tenth of the
    ///   residual root-mean-square rather than a flat `0.1` (the same
    ///   thing, at the percent-return scale `arch` is fitted on);
    /// * GARCH/GJR `omega` is a *variance*: strictly positive (enforced by
    ///   [`GarchSpec::validate_params`]), in units of `y^2`, and routinely
    ///   many orders of magnitude below any absolute floor. Its own value
    ///   is the only scale it has, so the step is purely relative — to
    ///   first order this is the step you would take differentiating with
    ///   respect to `ln(omega)`;
    /// * the EGARCH intercept is a *log*-variance: rescaling `y` shifts it
    ///   by `(1 - sum(beta)) * 2 ln c` rather than stretching it, so it is
    ///   an O(1) quantity and keeps the `max(|omega|, 0.1)` rule;
    /// * `alpha`, `gamma`, `beta` and `nu` are dimensionless and keep
    ///   `max(|theta_i|, 0.1)` unchanged.
    fn step_scales(&self, params: &[f64]) -> Result<Vec<f64>, GarchError> {
        let (mean, omega, _, _, _, _) = self.spec.split_params(params)?;
        let mut scales: Vec<f64> = params.iter().map(|v| v.abs().max(0.1)).collect();
        if let Some(&mu) = mean.first() {
            let resids = self.resids(mean);
            let rms = (resids.iter().map(|e| e * e).sum::<f64>() / resids.len() as f64).sqrt();
            scales[0] = mu.abs().max(0.1 * rms);
        }
        if !matches!(self.spec.vol, VolSpec::Egarch { .. }) {
            scales[self.spec.n_mean_params()] = omega;
        }
        Ok(scales)
    }

    /// Which parameters sit at an active constraint boundary at `params`,
    /// at the resolution of the standard-error finite-difference probes.
    ///
    /// A constraint counts as *active* when a Hessian probe from `params`
    /// would cross it — i.e. exactly when the classical full-vector
    /// covariance is not computable. The probes move coordinate `i` by
    /// `h_i = eps^(1/4) * step_scale[i]`, up to two coordinates at once,
    /// so with a small safety factor:
    ///
    /// * GARCH/GJR `alpha_i` (resp. `beta_j`) is at its sign bound when
    ///   `alpha_i <= safety * h_i`;
    /// * a GJR pair constraint `alpha_i + gamma_i >= 0` is active when the
    ///   sum is within the *joint* probe reach — both members are flagged,
    ///   since probing either can cross it;
    /// * the persistence bound (`< 1`) is active when the persistence plus
    ///   twice the largest single-coordinate reach over the *remaining
    ///   free* coefficients touches 1 — every free `alpha`/`gamma`/`beta`
    ///   is then flagged (the constraint is a direction, not a
    ///   coordinate; an IGARCH fit has no interior coefficient standard
    ///   errors);
    /// * EGARCH: same rule for `|sum(beta)| < 1` over the `beta`s.
    ///
    /// `omega` never triggers (GARCH/GJR probe it relatively, EGARCH's is
    /// unrestricted); `mu` is unrestricted; `nu`'s admissibility bound
    /// (`> 2`) is below the reach of its probe from the optimizer box
    /// `(2.05, 500)`.
    fn boundary_mask(&self, params: &[f64], step_scale: &[f64]) -> Result<Vec<bool>, GarchError> {
        // The probes of `inference::numerical_hessian`, with slack for
        // float slop in the comparisons below.
        const SAFETY: f64 = 1.25;
        let h: Vec<f64> = step_scale
            .iter()
            .map(|&s| f64::EPSILON.powf(0.25) * s)
            .collect();
        let (_, _, alphas, gammas, betas, _) = self.spec.split_params(params)?;
        let k = params.len();
        let mut mask = vec![false; k];
        let (p, o, q) = self.spec.vol.lags();
        let a0 = self.spec.n_mean_params() + 1; // first alpha
        let g0 = a0 + p; // first gamma
        let b0 = g0 + o; // first beta

        match self.spec.vol {
            VolSpec::Garch { .. } | VolSpec::Gjr { .. } => {
                for (i, &a) in alphas.iter().enumerate() {
                    if a <= SAFETY * h[a0 + i] {
                        mask[a0 + i] = true;
                    }
                }
                for (j, &b) in betas.iter().enumerate() {
                    if b <= SAFETY * h[b0 + j] {
                        mask[b0 + j] = true;
                    }
                }
                for (i, &g) in gammas.iter().enumerate() {
                    let (a, ha) = if i < p {
                        (alphas[i], h[a0 + i])
                    } else {
                        (0.0, 0.0)
                    };
                    if a + g <= SAFETY * (ha + h[g0 + i]) {
                        mask[g0 + i] = true;
                        if i < p {
                            mask[a0 + i] = true;
                        }
                    }
                }
                // Persistence reach over the still-free coefficients: a
                // Hessian probe raises it by at most w_i h_i + w_j h_j
                // <= 2 max(w h), with w = 1 for alpha/beta, 0.5 for gamma.
                let mut reach: f64 = 0.0;
                for i in 0..p {
                    if !mask[a0 + i] {
                        reach = reach.max(h[a0 + i]);
                    }
                }
                for i in 0..o {
                    if !mask[g0 + i] {
                        reach = reach.max(0.5 * h[g0 + i]);
                    }
                }
                for j in 0..q {
                    if !mask[b0 + j] {
                        reach = reach.max(h[b0 + j]);
                    }
                }
                let pers = self.spec.persistence(params)?;
                if pers + SAFETY * 2.0 * reach >= 1.0 {
                    for i in 0..p {
                        mask[a0 + i] = true;
                    }
                    for i in 0..o {
                        mask[g0 + i] = true;
                    }
                    for j in 0..q {
                        mask[b0 + j] = true;
                    }
                }
            }
            VolSpec::Egarch { .. } => {
                let sum_beta: f64 = betas.iter().sum();
                let mut reach: f64 = 0.0;
                for j in 0..q {
                    reach = reach.max(h[b0 + j]);
                }
                if sum_beta.abs() + SAFETY * 2.0 * reach >= 1.0 {
                    for j in 0..q {
                        mask[b0 + j] = true;
                    }
                }
            }
        }
        Ok(mask)
    }

    /// A teaching note describing the active boundaries behind a `mask`
    /// from [`GarchModel::boundary_mask`]; `None` when nothing is flagged.
    pub(crate) fn boundary_note(&self, params: &[f64], mask: &[bool]) -> Option<String> {
        if mask.len() != params.len() || !mask.iter().any(|&b| b) {
            return None;
        }
        let names = self.spec.param_names();
        let flagged: Vec<&str> = names
            .iter()
            .zip(mask)
            .filter(|(_, &b)| b)
            .map(|(n, _)| n.as_str())
            .collect();
        let pers = self.spec.persistence(params).ok()?;
        // Classify the cause from the fitted values themselves: a masked
        // coefficient that is not near a coordinate zero can only have
        // been flagged by the joint persistence / |sum(beta)| constraint.
        let step_scale = self.step_scales(params).ok()?;
        let near_zero = |i: usize, v: f64| v.abs() <= 4.0 * f64::EPSILON.powf(0.25) * step_scale[i];
        let (_, _, alphas, gammas, _, _) = self.spec.split_params(params).ok()?;
        let a0 = self.spec.n_mean_params() + 1;
        let g0 = a0 + alphas.len();
        let mut causes: Vec<String> = Vec::new();
        let joint_active = names
            .iter()
            .zip(mask)
            .enumerate()
            .any(|(i, (name, &b))| b && !name.starts_with("gamma") && !near_zero(i, params[i]));
        if joint_active {
            causes.push(match self.spec.vol {
                VolSpec::Egarch { .. } => format!(
                    "|sum(beta)| = {:.6} sits at its stationarity bound (1)",
                    pers.abs()
                ),
                _ => format!(
                    "the persistence sits at its upper bound (1) — an integrated (IGARCH) \
                     fit, persistence = {pers:.6}"
                ),
            });
        }
        let coord_zeros: Vec<&str> = names
            .iter()
            .zip(mask)
            .enumerate()
            .filter(|(i, (name, &b))| {
                b && (name.starts_with("alpha") || name.starts_with("beta"))
                    && near_zero(*i, params[*i])
            })
            .map(|(_, (n, _))| n.as_str())
            .collect();
        if !coord_zeros.is_empty() {
            causes.push(format!(
                "{} at the sign constraint (0)",
                coord_zeros.join(", ")
            ));
        }
        for (i, &g) in gammas.iter().enumerate() {
            if mask[g0 + i] && !near_zero(g0 + i, g) {
                let a = alphas.get(i).copied().unwrap_or(0.0);
                if (a + g).abs() <= 4.0 * f64::EPSILON.powf(0.25) * (step_scale[g0 + i] + 1.0) {
                    causes.push(format!(
                        "alpha[{n}] + gamma[{n}] at its lower bound (0)",
                        n = i + 1
                    ));
                }
            }
        }
        if causes.is_empty() {
            causes.push("one or more coefficients sit at an active constraint".to_owned());
        }
        Some(format!(
            "Boundary fit: {causes}. Parameters at an active boundary ({flagged}) have no \
             classical standard errors — the observed information is singular there by \
             construction, and their sampling distribution is a boundary mixture, not normal. \
             Their entries in se_mle/se_robust are NaN with se_valid false; interior parameters \
             keep finite standard errors from the reduced Hessian over the free directions. \
             With alpha at 0 the variance recursion carries no shock feedback and beta is only \
             weakly identified (a likelihood ridge): treat the whole fit with care.",
            causes = causes.join("; "),
            flagged = flagged.join(", ")
        ))
    }

    /// MLE and Bollerslev-Wooldridge robust standard errors at `params`
    /// (usually the fitted values) — see [`crate::inference`] for the
    /// estimators and `GarchModel::step_scales` for the
    /// numerical-derivative steps.
    ///
    /// **Boundary-aware** (audit round 7): parameters at an active
    /// constraint boundary ([`GarchModel::boundary_mask`]) are excluded
    /// from the numerical Hessian — a probe across the constraint is not
    /// evaluable and the information is singular there by construction —
    /// and come back as `se = NaN`, `se_valid = false`,
    /// `boundary = true`, while interior parameters keep finite standard
    /// errors from the reduced Hessian. If even the reduced problem is
    /// degenerate (singular reduced Hessian, no free direction, or a
    /// failed probe), the report is all-NaN with every `se_valid` false —
    /// flatness is reported, never a fabricated number and never a silent
    /// error.
    ///
    /// # Errors
    ///
    /// Parameter validation only ([`GarchSpec::validate_params`],
    /// step-scale validation). Numerical failure is not an error: it is
    /// reported through the flags as described above.
    pub fn standard_errors(&self, params: &[f64]) -> Result<StdErrors, GarchError> {
        self.spec.validate_params(params)?;
        let step_scale = self.step_scales(params)?;
        let mask = self.boundary_mask(params, &step_scale)?;
        let free: Vec<bool> = mask.iter().map(|&b| !b).collect();
        let result = inference::std_errors(
            |p| self.loglike(p).map(|ll| -ll),
            |p| {
                self.loglike_obs(p)
                    .map(|lls| lls.into_iter().map(|l| -l).collect())
            },
            params,
            self.y.len(),
            &step_scale,
            &free,
        );
        Ok(result.unwrap_or_else(|_| StdErrors::all_invalid(params.len(), mask)))
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
        best.map(|(_, cand)| cand).ok_or(GarchError::InvalidSpec {
            what: "no admissible starting value gives a finite log-likelihood; the \
                   series is degenerate (constant, or nearly so) or on a scale that \
                   overflows — GARCH is normally fitted to returns in percent",
        })
    }

    /// The internal standardization scale: the root-mean-square of the
    /// starting residuals (the same quantity the backcast and the grid
    /// starting values are built from). Estimation runs on `y / s` and the
    /// optimum is mapped back — see [`GarchModel::fit`]. Falls back to 1
    /// (no rescaling) if the RMS is degenerate, leaving whatever honest
    /// error the unscaled path raises.
    fn standardization_scale(&self) -> f64 {
        let resids = Self::starting_resids(&self.y, self.spec.mean);
        let rms = (resids.iter().map(|e| e * e).sum::<f64>() / resids.len() as f64).sqrt();
        if rms.is_finite() && rms > 0.0 {
            rms
        } else {
            1.0
        }
    }

    /// Maps a parameter vector fitted on the standardized series `y / s`
    /// back to the units of `y`: `mu -> s * mu`, GARCH/GJR
    /// `omega -> s^2 * omega`, EGARCH `omega -> omega + (1 - sum(beta)) *
    /// ln(s^2)` (its intercept is a log-variance, which *shifts* under
    /// rescaling), every dimensionless coefficient (`alpha`, `gamma`,
    /// `beta`, `nu`) unchanged. This is the exact reparameterization
    /// `y -> c y` of the model, applied with `c = s`.
    fn params_from_standardized(&self, params: &[f64], s: f64) -> Result<Vec<f64>, GarchError> {
        let mut out = params.to_vec();
        let nm = self.spec.n_mean_params();
        if nm > 0 {
            out[0] *= s;
        }
        match self.spec.vol {
            VolSpec::Egarch { .. } => {
                let (_, _, _, _, betas, _) = self.spec.split_params(params)?;
                let sum_beta: f64 = betas.iter().sum();
                out[nm] += (1.0 - sum_beta) * 2.0 * s.ln();
            }
            _ => out[nm] *= s * s,
        }
        Ok(out)
    }

    /// Fits the model by quasi-maximum likelihood and returns the results
    /// object.
    ///
    /// **Scale-adaptive estimation** (audit round 7). The optimizer runs
    /// on the internally standardized series `y / s`, `s` the RMS of the
    /// starting residuals, and the optimum is mapped back through the
    /// exact reparameterization `y -> c y` (see
    /// `GarchModel::params_from_standardized`) — the same trick `arch`'s
    /// `rescale=True` applies, done unconditionally so it cannot be
    /// forgotten. Rescaling the data is a pure relabeling of the model,
    /// so the *estimator* should commute with it; without this, the
    /// round-1 audit measured 52/330 cross-scale comparisons converging
    /// to a different point (the optimizer's paths, not its optima,
    /// depend on the units through termination arithmetic). The
    /// log-likelihood, conditional variances, and standard errors in the
    /// results are all evaluated at the mapped parameters *on the
    /// original data*, so their units are the caller's.
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
    /// **Search strategy**: `arch`-style grid starting values, L-BFGS, a
    /// Nelder-Mead polish (2 restarts), and a final L-BFGS pass, keeping
    /// the best point found. The fixture tests pin the optimum to within
    /// 1e-6 absolute log-likelihood of the `arch` package (or better).
    ///
    /// The likelihood is evaluated through [`crate::objective`], whose
    /// value is bit-identical to [`GarchModel::loglike`] but allocation-
    /// free, and which supplies an analytic gradient for every volatility
    /// specification under normal innovations (Student-t keeps the
    /// optimizer's central differences).
    ///
    /// # Errors
    ///
    /// Starting-value/optimizer failures; likelihood errors at the
    /// optimum. If standard errors cannot be computed at the optimum
    /// (singular Hessian at a flat or boundary point), the fit still
    /// succeeds with NaN standard errors and per-parameter
    /// `se_valid`/`boundary` flags — flatness is reported, not hidden.
    pub fn fit(&self) -> Result<GarchResults, GarchError> {
        let s = self.standardization_scale();
        let (params, converged) = if s == 1.0 {
            self.optimize()?
        } else {
            let scaled: Vec<f64> = self.y.iter().map(|v| v / s).collect();
            let inner = Self::new(&scaled, self.spec)?;
            let (inner_params, converged) = inner.optimize()?;
            (self.params_from_standardized(&inner_params, s)?, converged)
        };
        // Everything reported is evaluated at the mapped parameters on the
        // ORIGINAL data (`self.loglike` is bit-identical to the optimizer's
        // objective, re-based to the caller's units).
        let loglik = self.loglike(&params)?;
        // Boundary-aware standard errors: finite for interior parameters,
        // NaN + `se_valid = false` at an active boundary, all-invalid when
        // even the reduced Hessian is degenerate. Never a silent NaN row —
        // the flags and the note say why.
        let se = self.standard_errors(&params)?;
        let note = self.boundary_note(&params, &se.boundary);
        GarchResults::build(self, params, loglik, se, converged, note)
    }

    /// The three-stage QMLE search on this model's own data; returns the
    /// best parameters (in this model's units) and the convergence flag.
    /// [`GarchModel::fit`] calls it on the internally standardized model.
    fn optimize(&self) -> Result<(Vec<f64>, bool), GarchError> {
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

        // The estimation-time objective: a fused, allocation-free
        // evaluation of the same likelihood, with an analytic gradient
        // where one exists (see [`crate::objective`]). Its value agrees
        // with `self.loglike` bit for bit.
        let mut objective = FitObjective::new(self, NU_BOUNDS);

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
                what: "the QMLE optimizer produced no result at all (this should be \
                       unreachable; please report it)",
            })?;
        if !best.f.is_finite() {
            return Err(GarchError::NonFinite {
                what: "the optimized log-likelihood",
                at: None,
            });
        }

        Ok((to_natural(&best.x)?, converged))
    }
}
