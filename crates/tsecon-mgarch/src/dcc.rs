//! Dynamic Conditional Correlation GARCH (Engle 2002).
//!
//! The conditional covariance is `H_t = D_t R_t D_t` exactly as in CCC, but
//! the correlation now *evolves* through a scalar GARCH-like recursion on an
//! auxiliary matrix `Q_t`:
//!
//! ```text
//! Q_t = (1 - a - b) Qbar + a z_{t-1} z_{t-1}' + b Q_{t-1},
//! R_t = diag(Q_t)^{-1/2} Q_t diag(Q_t)^{-1/2},
//! ```
//!
//! with `z_t` the standardized residuals from the univariate stage, `a, b >=
//! 0`, `a + b < 1`, and `Qbar` **correlation-targeted** to the sample second
//! moment `(1/T) sum_t z_t z_t'` of those residuals (Engle 2002). Estimation
//! is two-step (Engle): step 1 is the `k` univariate GARCH fits
//! ([`crate::stage`]); step 2 maximizes the DCC quasi-log-likelihood over
//! `(a, b)` with the univariate parameters held fixed.
//!
//! # Validation status — read this
//!
//! There is **no external third-party DCC reference** available in this
//! project, so — unlike the univariate GARCH crate, which is pinned to
//! Kevin Sheppard's `arch` — the DCC path here is **not** validated against a
//! golden implementation. It is validated instead by four internal checks,
//! each exercised in the property tests:
//!
//! 1. **CCC special case (exact).** At `a = b = 0` the recursion gives
//!    `Q_t = Qbar` for all `t`, so `R_t = corr(Qbar)`, and the DCC
//!    log-likelihood equals the CCC log-likelihood to `1e-10`.
//! 2. **Positive-definiteness (exact).** Every `R_t` on the fixture data
//!    factorizes cleanly (a successful Cholesky *is* the PD certificate).
//! 3. **Correlation targeting.** The sample mean of the driving term
//!    `z_t z_t'` equals `Qbar` by construction, so the recursion's
//!    unconditional level `E[Q_t] = Qbar` (fixed-point check).
//! 4. **Simulation recovery (Monte-Carlo, loose).** On the fixture's
//!    simulated data (truth `a = 0.03`, `b = 0.95`), the estimated
//!    persistence `a + b` lands within `0.05` of the true `0.98`. This is a
//!    deliberately loose single-realization bar, not a precision claim.

use tsecon_garch::GarchSpec;
use tsecon_linalg::faer::{Mat, MatRef};
use tsecon_optim::{minimize, FnObjective, Method, NelderMeadOptions};

use crate::ccc::scale_correlation;
use crate::dynamics::{
    adcc_delta, asymmetric_moment, eval_dynamics, gaussian_loglik, student_t_loglik, CorrDist,
    DccVariant, DynParams,
};
use crate::error::MgarchError;
use crate::stage::UnivariateStage;
use crate::util::{cholesky, corr_from_cov, moment_matrix, quad_form};

/// The maximum admissible persistence `a + b` during estimation; a hard
/// margin below one keeps the correlation recursion strictly stationary.
const MAX_PERSISTENCE: f64 = 1.0 - 1e-6;

/// Nelder-Mead starting points `(a, b)` for the step-2 search. Several
/// starts guard against the flat ridge along `a + b ~ const`; the best
/// (lowest negative log-likelihood) wins. None sits at the fixture truth, so
/// recovery is not begged.
const STARTS: [[f64; 2]; 3] = [[0.05, 0.90], [0.03, 0.94], [0.01, 0.97]];

/// Nelder-Mead starting points `(a, b, g)` for the ADCC step-2 search. One
/// start sits at `g = 0` (the DCC special case) so symmetric data can
/// collapse the asymmetry cleanly.
const ADCC_STARTS: [[f64; 3]; 4] = [
    [0.03, 0.94, 0.02],
    [0.05, 0.90, 0.05],
    [0.01, 0.97, 0.00],
    [0.03, 0.90, 0.10],
];

/// Admissible degrees-of-freedom window for the Student-t second stage.
/// Below `NU_MIN` the standardized t's variance rescaling `nu - 2`
/// degenerates; above `NU_MAX` the likelihood is Gaussian to machine
/// precision and the surface is flat.
const NU_MIN: f64 = 2.05;
const NU_MAX: f64 = 1.0e4;

/// Student-t starting values for `nu` appended to each dynamic start, plus
/// one deliberately near-Gaussian start.
const NU_STARTS: [f64; 2] = [8.0, 25.0];

/// A DCC-GARCH model: a univariate [`GarchSpec`] applied to every series,
/// with a scalar dynamic correlation on top.
///
/// The correlation recursion defaults to Engle (2002) DCC with a Gaussian
/// second stage — the historical behavior, bit-identical to earlier
/// releases. Opt into the corrected recursion of Aielli (2013) or the
/// asymmetric recursion of Cappiello-Engle-Sheppard (2006) with
/// [`DccGarch::with_variant`], and into a Student-t second-stage likelihood
/// with [`DccGarch::with_dist`] (see [`crate::dynamics`] for the formulas
/// and the consistency argument).
#[derive(Debug, Clone, Copy)]
pub struct DccGarch {
    spec: GarchSpec,
    variant: DccVariant,
    dist: CorrDist,
}

impl DccGarch {
    /// A DCC model whose per-series volatilities follow `spec`
    /// (variant [`DccVariant::Dcc`], distribution [`CorrDist::Normal`]).
    pub fn new(spec: GarchSpec) -> Self {
        Self {
            spec,
            variant: DccVariant::Dcc,
            dist: CorrDist::Normal,
        }
    }

    /// Selects the correlation recursion (DCC, cDCC, or ADCC).
    pub fn with_variant(mut self, variant: DccVariant) -> Self {
        self.variant = variant;
        self
    }

    /// Selects the second-stage innovation distribution.
    pub fn with_dist(mut self, dist: CorrDist) -> Self {
        self.dist = dist;
        self
    }

    /// The univariate specification applied to each series.
    pub fn spec(&self) -> &GarchSpec {
        &self.spec
    }

    /// The correlation recursion variant.
    pub fn variant(&self) -> DccVariant {
        self.variant
    }

    /// The second-stage innovation distribution.
    pub fn dist(&self) -> CorrDist {
        self.dist
    }

    /// Fits the model to `series` by two-step (Engle) estimation.
    ///
    /// # Errors
    ///
    /// * every [`MgarchError`] from the univariate stage;
    /// * [`MgarchError::Optim`] if the step-2 search fails outright;
    /// * [`MgarchError::Linalg`] if a correlation matrix along the fitted
    ///   path cannot be factorized.
    pub fn fit(&self, series: &[Vec<f64>]) -> Result<DccFit, MgarchError> {
        let stage = UnivariateStage::fit(series, self.spec)?;
        if self.variant == DccVariant::Dcc && self.dist == CorrDist::Normal {
            // The historical default path, kept verbatim (same starts, same
            // objective arithmetic, same optimizer trajectory) so default
            // results stay bit-identical release over release.
            return fit_classic(stage);
        }
        fit_general(stage, self.variant, self.dist)
    }
}

/// The pre-variant Engle (2002) DCC + Gaussian estimation path, unchanged.
fn fit_classic(stage: UnivariateStage) -> Result<DccFit, MgarchError> {
    let qbar = moment_matrix(&stage.z, stage.k);

        // Step 2: maximize the DCC quasi-log-likelihood over (a, b). The
        // objective is the *negative* full Gaussian log-likelihood, with an
        // infinite wall on the infeasible region (a, b >= 0, a + b < 1) — the
        // optimizer treats non-finite values as infeasible points.
        let mut best_x = [STARTS[0][0], STARTS[0][1]];
        let mut best_f = f64::INFINITY;
        let mut converged = false;
        let opts = NelderMeadOptions::default();
        {
            let stage_ref = &stage;
            let qbar_ref = &qbar;
            let mut objective = FnObjective::new(|x: &[f64]| {
                let (a, b) = (x[0], x[1]);
                if !a.is_finite() || !b.is_finite() || a < 0.0 || b < 0.0 || a + b > MAX_PERSISTENCE
                {
                    return f64::INFINITY;
                }
                match dcc_full_loglik(stage_ref, qbar_ref.as_ref(), a, b) {
                    Ok(ll) if ll.is_finite() => -ll,
                    _ => f64::INFINITY,
                }
            });
            for start in STARTS {
                let method = Method::NelderMead(opts);
                let res = minimize(&mut objective, &start, &method)?;
                if res.f < best_f {
                    best_f = res.f;
                    best_x = [res.x[0], res.x[1]];
                    converged = res.converged;
                }
            }
        }

        if !best_f.is_finite() {
            return Err(MgarchError::Optim(tsecon_optim::OptimError::NonFinite {
                what: "DCC step-2 objective (no feasible start converged)",
            }));
        }

        // Clamp tiny negative excursions the simplex may leave behind, then
        // rebuild the fitted path at the optimum (propagating real errors).
        let a = best_x[0].max(0.0);
        let b = best_x[1].max(0.0);
        let (correlation_path, q_forecast) = dcc_path(&stage, qbar.as_ref(), a, b)?;
        let loglik = dcc_full_loglik(&stage, qbar.as_ref(), a, b)?;

        Ok(DccFit {
            stage,
            qbar,
            a,
            b,
            g: 0.0,
            nu: None,
            variant: DccVariant::Dcc,
            dist: CorrDist::Normal,
            nbar: None,
            loglik,
            correlation_path,
            q_forecast,
            converged,
        })
}

/// Step-2 estimation for every opt-in configuration (cDCC / ADCC and/or the
/// Student-t second stage). Same multi-start Nelder-Mead architecture as the
/// classic path, generalized to the parameter vector
/// `[a, b (, g for ADCC) (, nu for Student-t)]`.
fn fit_general(
    stage: UnivariateStage,
    variant: DccVariant,
    dist: CorrDist,
) -> Result<DccFit, MgarchError> {
    let qbar = moment_matrix(&stage.z, stage.k);
    let (nbar, delta) = match variant {
        DccVariant::Adcc => {
            let nbar = asymmetric_moment(&stage);
            let delta = adcc_delta(qbar.as_ref(), nbar.as_ref())?;
            (Some(nbar), delta)
        }
        _ => (None, 0.0),
    };

    // Precompute the per-t univariate log-determinant sum_i ln sigma2_{i,t}
    // once (constant across step-2 evaluations).
    let ln_det_d2: Vec<f64> = stage
        .sigma2
        .iter()
        .map(|row| row.iter().map(|s2| s2.ln()).sum())
        .collect();

    let n_dyn = if variant == DccVariant::Adcc { 3 } else { 2 };
    let with_nu = dist == CorrDist::StudentT;

    // Assemble the start list.
    let mut starts: Vec<Vec<f64>> = Vec::new();
    let base: Vec<Vec<f64>> = if variant == DccVariant::Adcc {
        ADCC_STARTS.iter().map(|s| s.to_vec()).collect()
    } else {
        STARTS.iter().map(|s| s.to_vec()).collect()
    };
    if with_nu {
        for s in &base {
            let mut x = s.clone();
            x.push(NU_STARTS[0]);
            starts.push(x);
        }
        // One deliberately near-Gaussian start from the middle base point.
        let mut x = base[base.len() / 2].clone();
        x.push(NU_STARTS[1]);
        starts.push(x);
    } else {
        starts = base;
    }

    let mut best_x: Vec<f64> = starts[0].clone();
    let mut best_f = f64::INFINITY;
    let mut converged = false;
    let opts = NelderMeadOptions::default();
    {
        let stage_ref = &stage;
        let qbar_ref = &qbar;
        let nbar_ref = nbar.as_ref();
        let ln_det_d2_ref = &ln_det_d2;
        let mut objective = FnObjective::new(|x: &[f64]| {
            let (a, b) = (x[0], x[1]);
            let g = if variant == DccVariant::Adcc { x[2] } else { 0.0 };
            let nu = if with_nu { x[n_dyn] } else { f64::NAN };
            if !a.is_finite() || !b.is_finite() || a < 0.0 || b < 0.0 {
                return f64::INFINITY;
            }
            // Stationarity/positivity wall: a + b (+ delta g) < 1 keeps the
            // recursion stationary and (for ADCC) the intercept PSD
            // (Cappiello-Engle-Sheppard 2006 sufficient condition).
            if !g.is_finite() || g < 0.0 || a + b + delta * g > MAX_PERSISTENCE {
                return f64::INFINITY;
            }
            if variant == DccVariant::Adcc && g > MAX_PERSISTENCE {
                return f64::INFINITY;
            }
            if with_nu && !(NU_MIN..=NU_MAX).contains(&nu) {
                return f64::INFINITY;
            }
            let eval = match eval_dynamics(
                stage_ref,
                variant,
                qbar_ref.as_ref(),
                nbar_ref.map(|m| m.as_ref()),
                DynParams { a, b, g },
                false,
            ) {
                Ok(e) => e,
                Err(_) => return f64::INFINITY,
            };
            let ll = match dist {
                CorrDist::Normal => gaussian_loglik(stage_ref.k, ln_det_d2_ref, &eval),
                CorrDist::StudentT => student_t_loglik(stage_ref.k, ln_det_d2_ref, &eval, nu),
            };
            if ll.is_finite() {
                -ll
            } else {
                f64::INFINITY
            }
        });
        for start in &starts {
            let method = Method::NelderMead(opts);
            let res = minimize(&mut objective, start, &method)?;
            if res.f < best_f {
                best_f = res.f;
                best_x = res.x.clone();
                converged = res.converged;
            }
        }
    }

    if !best_f.is_finite() {
        return Err(MgarchError::Optim(tsecon_optim::OptimError::NonFinite {
            what: "DCC step-2 objective (no feasible start converged)",
        }));
    }

    // Clamp tiny negative excursions, then rebuild the path at the optimum.
    let a = best_x[0].max(0.0);
    let b = best_x[1].max(0.0);
    let g = if variant == DccVariant::Adcc {
        best_x[2].max(0.0)
    } else {
        0.0
    };
    let nu = with_nu.then(|| best_x[n_dyn].clamp(NU_MIN, NU_MAX));

    let eval = eval_dynamics(
        &stage,
        variant,
        qbar.as_ref(),
        nbar.as_ref().map(|m| m.as_ref()),
        DynParams { a, b, g },
        true,
    )?;
    let loglik = match dist {
        CorrDist::Normal => gaussian_loglik(stage.k, &ln_det_d2, &eval),
        CorrDist::StudentT => {
            student_t_loglik(stage.k, &ln_det_d2, &eval, nu.unwrap_or(f64::NAN))
        }
    };
    let correlation_path = eval.r_path.unwrap_or_default();

    Ok(DccFit {
        stage,
        qbar: eval.target,
        a,
        b,
        g,
        nu,
        variant,
        dist,
        nbar,
        loglik,
        correlation_path,
        q_forecast: eval.q_next,
        converged,
    })
}

/// A fitted DCC-GARCH model (any [`DccVariant`], any [`CorrDist`]).
#[derive(Debug, Clone)]
pub struct DccFit {
    /// The fitted univariate stage.
    pub stage: UnivariateStage,
    /// The correlation-targeting matrix: `Qbar = (1/T) sum_t z_t z_t'` for
    /// DCC/ADCC; for cDCC this is Aielli's `S` — the correlation-normalized
    /// sample second moment of the rescaled residuals
    /// `z*_t = diag(Q_t)^{1/2} z_t` at the fitted `(a, b)`.
    pub qbar: Mat<f64>,
    /// The estimated news coefficient `a`.
    pub a: f64,
    /// The estimated persistence coefficient `b`.
    pub b: f64,
    /// The estimated asymmetric news coefficient `g` on
    /// `n_{t-1} n_{t-1}'`, `n_t = min(z_t, 0)` (Cappiello-Engle-Sheppard
    /// 2006). Exactly `0.0` for non-ADCC variants.
    pub g: f64,
    /// The estimated Student-t degrees of freedom of the second stage;
    /// `None` under the Gaussian second stage.
    pub nu: Option<f64>,
    /// Which correlation recursion was estimated.
    pub variant: DccVariant,
    /// The second-stage innovation distribution.
    pub dist: CorrDist,
    /// The asymmetric targeting matrix `Nbar = (1/T) sum_t n_t n_t'`
    /// (ADCC only; `None` otherwise).
    pub nbar: Option<Mat<f64>>,
    /// The full log-likelihood at the two-step estimates: Gaussian under
    /// [`CorrDist::Normal`], standardized multivariate Student-t under
    /// [`CorrDist::StudentT`] (step 1 remains whatever the univariate spec
    /// used — the documented two-step convention).
    pub loglik: f64,
    /// The dynamic correlation path `R_t`, `t = 0..T` (length `T`).
    pub correlation_path: Vec<Mat<f64>>,
    /// `Q_{T+1}` — the auxiliary matrix one step past the sample, used for
    /// the one-step covariance forecast.
    q_forecast: Mat<f64>,
    /// Whether at least one step-2 start converged by the Nelder-Mead
    /// criterion (the best point found is returned either way).
    pub converged: bool,
}

impl DccFit {
    /// Number of series `k`.
    pub fn k(&self) -> usize {
        self.stage.k
    }

    /// Number of observations `T`.
    pub fn nobs(&self) -> usize {
        self.stage.nobs
    }

    /// The estimated persistence `a + b` of the correlation recursion.
    pub fn persistence(&self) -> f64 {
        self.a + self.b
    }

    /// The exact one-step-ahead auxiliary matrix `Q_{T+1}` (crate-internal:
    /// the forecast recursion in [`crate::forecast`] starts here).
    pub(crate) fn q_next(&self) -> MatRef<'_, f64> {
        self.q_forecast.as_ref()
    }

    /// The conditional covariance `H_t = D_t R_t D_t` at time index `t`
    /// (`0 <= t < T`).
    ///
    /// # Errors
    ///
    /// [`MgarchError::InvalidParameter`] if `t` is out of range.
    pub fn conditional_covariance(&self, t: usize) -> Result<Mat<f64>, MgarchError> {
        if t >= self.stage.nobs {
            return Err(MgarchError::InvalidParameter {
                name: "t",
                value: t as f64,
                requirement: "0 <= t < T",
            });
        }
        let d: Vec<f64> = self.stage.sigma2[t].iter().map(|s| s.sqrt()).collect();
        Ok(scale_correlation(self.correlation_path[t].as_ref(), &d))
    }

    /// The one-step-ahead conditional covariance forecast `H_{T+1}`.
    ///
    /// `R_{T+1} = corr(Q_{T+1})` with
    /// `Q_{T+1} = (1 - a - b) Qbar + a z_T z_T' + b Q_T`, and `D_{T+1}` from
    /// the per-series analytic one-step variance forecasts. **Multi-step DCC
    /// forecasts are not analytic** — `E[R_{T+m}]` has no closed form because
    /// the `diag(Q)^{-1/2}` normalization is nonlinear — and require
    /// simulation.
    ///
    // TODO(phase0): multi-step DCC covariance forecasts by Monte-Carlo
    // simulation of the (z_t, Q_t) recursion, sharing the parallel path
    // engine of ROADMAP 03; only the one-step forecast is analytic here.
    ///
    /// # Errors
    ///
    /// [`MgarchError::Univariate`] if a univariate one-step forecast fails.
    pub fn forecast_covariance_one_step(&self) -> Result<Mat<f64>, MgarchError> {
        let k = self.stage.k;
        let r_next = corr_from_cov(self.q_forecast.as_ref());
        let mut d = vec![0.0_f64; k];
        for (i, res) in self.stage.univariate.iter().enumerate() {
            let path = res
                .forecast_variance(1)
                .map_err(|e| MgarchError::Univariate {
                    series: i,
                    source: e,
                })?;
            d[i] = path[0].sqrt();
        }
        Ok(scale_correlation(r_next.as_ref(), &d))
    }
}

/// One DCC recursion step: `Q_next = (1 - a - b) Qbar + a z z' + b Q`.
fn advance_q(qbar: MatRef<'_, f64>, q: MatRef<'_, f64>, z: &[f64], a: f64, b: f64) -> Mat<f64> {
    let k = qbar.nrows();
    let omega = 1.0 - a - b;
    Mat::from_fn(k, k, |i, j| {
        omega * qbar[(i, j)] + a * z[i] * z[j] + b * q[(i, j)]
    })
}

/// The full DCC Gaussian log-likelihood at `(a, b)` — no path storage, for
/// use inside the step-2 optimizer.
///
/// ```text
/// L = -0.5 sum_t [ k ln(2 pi) + sum_i ln sigma2_{i,t} + ln|R_t| + z_t' R_t^{-1} z_t ].
/// ```
///
/// # Errors
///
/// [`MgarchError::Linalg`] if some `R_t` cannot be factorized.
pub(crate) fn dcc_full_loglik(
    stage: &UnivariateStage,
    qbar: MatRef<'_, f64>,
    a: f64,
    b: f64,
) -> Result<f64, MgarchError> {
    let ln_2pi = (2.0 * core::f64::consts::PI).ln();
    let k = stage.k as f64;
    let mut q: Mat<f64> = qbar.to_owned();
    let mut ll = 0.0;
    for t in 0..stage.nobs {
        let r = corr_from_cov(q.as_ref());
        let chol = cholesky(r.as_ref())?;
        let quad = quad_form(chol.factor.as_ref(), &stage.z[t]);
        let mut ln_det_h = 0.0;
        for &s2 in &stage.sigma2[t] {
            ln_det_h += s2.ln();
        }
        ll += -0.5 * (k * ln_2pi + ln_det_h + chol.log_det() + quad);
        q = advance_q(qbar, q.as_ref(), &stage.z[t], a, b);
    }
    Ok(ll)
}

/// The fitted correlation path plus the one-step-ahead `Q_{T+1}`.
///
/// Returns `(r_path, q_forecast)` where `r_path[t] = R_t` for `t = 0..T` and
/// `q_forecast = Q_{T+1}`.
///
/// # Errors
///
/// [`MgarchError::Linalg`] if some `R_t` is not positive-definite.
pub(crate) fn dcc_path(
    stage: &UnivariateStage,
    qbar: MatRef<'_, f64>,
    a: f64,
    b: f64,
) -> Result<(Vec<Mat<f64>>, Mat<f64>), MgarchError> {
    let mut q: Mat<f64> = qbar.to_owned();
    let mut r_path = Vec::with_capacity(stage.nobs);
    for t in 0..stage.nobs {
        let r = corr_from_cov(q.as_ref());
        // Certify positive-definiteness (the factorization is the check).
        let _ = cholesky(r.as_ref())?;
        r_path.push(r);
        q = advance_q(qbar, q.as_ref(), &stage.z[t], a, b);
    }
    Ok((r_path, q))
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::ccc::ccc_loglik;
    use crate::util::corr_from_cov;
    use tsecon_garch::{DistSpec, GarchSpec, MeanSpec, VolSpec};

    struct Rng(u64);
    impl Rng {
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        fn uniform(&mut self) -> f64 {
            ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        }
        fn normal(&mut self) -> f64 {
            let u1 = self.uniform();
            let u2 = self.uniform();
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        }
    }

    fn spec() -> GarchSpec {
        GarchSpec {
            mean: MeanSpec::Zero,
            vol: VolSpec::Garch { p: 1, q: 1 },
            dist: DistSpec::Normal,
        }
    }

    fn synthetic() -> Vec<Vec<f64>> {
        let mut rng = Rng(0xABCD_1234);
        let n = 400;
        let (mut s0, mut s1) = (Vec::with_capacity(n), Vec::with_capacity(n));
        let (mut v0, mut v1) = (1.0_f64, 1.0_f64);
        for _ in 0..n {
            let e0 = rng.normal();
            let common = rng.normal();
            let e1 = 0.6 * e0 + 0.8 * rng.normal();
            let x0 = v0.sqrt() * e0 + 0.2 * common;
            let x1 = v1.sqrt() * e1 + 0.2 * common;
            v0 = 0.05 + 0.1 * x0 * x0 + 0.85 * v0;
            v1 = 0.04 + 0.08 * x1 * x1 + 0.88 * v1;
            s0.push(x0);
            s1.push(x1);
        }
        vec![s0, s1]
    }

    /// Validation (a): at `a = b = 0` the DCC quasi-log-likelihood equals the
    /// CCC log-likelihood to 1e-10. `Q_t` collapses to `Qbar` for all `t`, so
    /// `R_t = corr(Qbar)`, which is exactly the CCC correlation.
    #[test]
    fn ccc_special_case() {
        let stage = UnivariateStage::fit(&synthetic(), spec()).unwrap();
        let qbar = moment_matrix(&stage.z, stage.k);
        let r_ccc = corr_from_cov(qbar.as_ref());
        let ll_ccc = ccc_loglik(&stage, r_ccc.as_ref()).unwrap();
        let ll_dcc0 = dcc_full_loglik(&stage, qbar.as_ref(), 0.0, 0.0).unwrap();
        assert!(
            (ll_ccc - ll_dcc0).abs() <= 1e-10 * ll_ccc.abs().max(1.0),
            "CCC {ll_ccc} vs DCC(0,0) {ll_dcc0}"
        );
    }

    /// Validation (c): correlation targeting. The sample mean of the driving
    /// term `z_t z_t'` equals `Qbar` by construction (to machine precision),
    /// which is the fixed point of the recursion: substituting `E[z z'] =
    /// E[Q_{t-1}] = Qbar` into `Q_t = (1-a-b)Qbar + a z z' + b Q_{t-1}`
    /// returns `Qbar` for any `(a, b)`.
    #[test]
    fn targeting_fixed_point() {
        let stage = UnivariateStage::fit(&synthetic(), spec()).unwrap();
        let qbar = moment_matrix(&stage.z, stage.k);
        // Mean of z z' over the sample.
        let k = stage.k;
        let mut mean = Mat::<f64>::zeros(k, k);
        for row in &stage.z {
            for i in 0..k {
                for j in 0..k {
                    mean[(i, j)] += row[i] * row[j];
                }
            }
        }
        let inv_t = 1.0 / stage.nobs as f64;
        for i in 0..k {
            for j in 0..k {
                assert!((mean[(i, j)] * inv_t - qbar[(i, j)]).abs() <= 1e-12);
            }
        }
        // Fixed-point identity for a representative (a, b).
        let (a, b) = (0.03, 0.95);
        let fixed = advance_q(qbar.as_ref(), qbar.as_ref(), &[0.0; 2], a, b);
        // With z z' replaced by its mean Qbar the news term is (a)Qbar; add it
        // back to check the identity (1-a-b)Qbar + a Qbar + b Qbar = Qbar.
        for i in 0..k {
            for j in 0..k {
                let full = fixed[(i, j)] + a * qbar[(i, j)];
                assert!((full - qbar[(i, j)]).abs() <= 1e-12);
            }
        }
    }
}
