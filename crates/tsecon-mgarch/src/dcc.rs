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

/// A DCC-GARCH model: a univariate [`GarchSpec`] applied to every series,
/// with a scalar dynamic correlation on top.
#[derive(Debug, Clone, Copy)]
pub struct DccGarch {
    spec: GarchSpec,
}

impl DccGarch {
    /// A DCC model whose per-series volatilities follow `spec`.
    pub fn new(spec: GarchSpec) -> Self {
        Self { spec }
    }

    /// The univariate specification applied to each series.
    pub fn spec(&self) -> &GarchSpec {
        &self.spec
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
            loglik,
            correlation_path,
            q_forecast,
            converged,
        })
    }
}

/// A fitted DCC-GARCH model.
#[derive(Debug, Clone)]
pub struct DccFit {
    /// The fitted univariate stage.
    pub stage: UnivariateStage,
    /// The correlation-targeting matrix `Qbar = (1/T) sum_t z_t z_t'`.
    pub qbar: Mat<f64>,
    /// The estimated news coefficient `a`.
    pub a: f64,
    /// The estimated persistence coefficient `b`.
    pub b: f64,
    /// The full Gaussian log-likelihood at the two-step estimates.
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
