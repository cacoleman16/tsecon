//! Constant Conditional Correlation GARCH (Bollerslev 1990).
//!
//! The conditional covariance factorizes as
//!
//! ```text
//! H_t = D_t R D_t,    D_t = diag(sigma_{1,t}, ..., sigma_{k,t}),
//! ```
//!
//! with `sigma_{i,t}` the univariate GARCH conditional standard deviation of
//! series `i` and `R` a *constant* correlation matrix. Estimation is the
//! two-step estimator: fit each univariate GARCH (step 1, [`crate::stage`]),
//! then set `R` to the sample correlation matrix of the standardized
//! residuals `z_{i,t} = eps_{i,t} / sigma_{i,t}` (step 2).
//!
//! Because `H_t = D_t R D_t` with `R` positive-definite and `D_t` positive
//! diagonal, every `H_t` is positive-definite and symmetric (asserted in the
//! property tests). Multi-step covariance forecasts are analytic: the
//! univariate variance forecasts drive `D_{T+m}` while `R` stays fixed.

use tsecon_garch::GarchSpec;
use tsecon_linalg::faer::{Mat, MatRef};

use crate::error::MgarchError;
use crate::stage::UnivariateStage;
use crate::util::{cholesky, corr_from_cov, moment_matrix, quad_form};

/// A CCC-GARCH model: a univariate [`GarchSpec`] applied to every series,
/// plus a constant conditional correlation.
#[derive(Debug, Clone, Copy)]
pub struct CccGarch {
    spec: GarchSpec,
}

impl CccGarch {
    /// A CCC model whose per-series volatilities follow `spec`.
    pub fn new(spec: GarchSpec) -> Self {
        Self { spec }
    }

    /// The univariate specification applied to each series.
    pub fn spec(&self) -> &GarchSpec {
        &self.spec
    }

    /// Fits the model to `series` (each inner vector is one series of the
    /// common length `T`).
    ///
    /// # Errors
    ///
    /// Propagates every [`MgarchError`] from the univariate stage
    /// ([`UnivariateStage::fit`]) and from factorizing the correlation
    /// matrix.
    pub fn fit(&self, series: &[Vec<f64>]) -> Result<CccFit, MgarchError> {
        let stage = UnivariateStage::fit(series, self.spec)?;
        CccFit::from_stage(stage)
    }
}

/// A fitted CCC-GARCH model.
#[derive(Debug, Clone)]
pub struct CccFit {
    /// The fitted univariate stage (per-series GARCH results and stacked
    /// residuals).
    pub stage: UnivariateStage,
    /// The constant conditional correlation `R`, `k x k`.
    pub correlation: Mat<f64>,
    /// The full Gaussian log-likelihood at the two-step estimates.
    pub loglik: f64,
}

impl CccFit {
    /// Assembles the fit from a completed univariate stage: correlation
    /// targeting and the log-likelihood.
    pub(crate) fn from_stage(stage: UnivariateStage) -> Result<Self, MgarchError> {
        let qbar = moment_matrix(&stage.z, stage.k);
        let correlation = corr_from_cov(qbar.as_ref());
        let loglik = ccc_loglik(&stage, correlation.as_ref())?;
        Ok(Self {
            stage,
            correlation,
            loglik,
        })
    }

    /// Number of series `k`.
    pub fn k(&self) -> usize {
        self.stage.k
    }

    /// Number of observations `T`.
    pub fn nobs(&self) -> usize {
        self.stage.nobs
    }

    /// The conditional covariance `H_t = D_t R D_t` at time index `t`
    /// (`0 <= t < T`).
    ///
    /// # Errors
    ///
    /// [`MgarchError::InvalidHorizon`] is not used here; the only failure is
    /// an out-of-range `t`, reported as [`MgarchError::InvalidParameter`].
    pub fn conditional_covariance(&self, t: usize) -> Result<Mat<f64>, MgarchError> {
        if t >= self.stage.nobs {
            return Err(MgarchError::InvalidParameter {
                name: "t",
                value: t as f64,
                requirement: "0 <= t < T",
            });
        }
        let d: Vec<f64> = self.stage.sigma2[t].iter().map(|s| s.sqrt()).collect();
        Ok(scale_correlation(self.correlation.as_ref(), &d))
    }

    /// Analytic multi-step covariance forecasts `H_{T+m}` for `m = 1..=horizon`.
    ///
    /// Each `H_{T+m} = D_{T+m} R D_{T+m}` where `D_{T+m}` is built from the
    /// per-series analytic univariate variance forecasts
    /// ([`tsecon_garch::GarchResults::forecast_variance`]) and `R` is the
    /// constant correlation. This is exact for CCC precisely because the
    /// correlation does not evolve (Bollerslev 1990); the DCC analogue needs
    /// simulation (see [`crate::dcc`]).
    ///
    /// # Errors
    ///
    /// * [`MgarchError::InvalidHorizon`] if `horizon == 0`;
    /// * [`MgarchError::Univariate`] if a univariate forecast fails.
    pub fn forecast_covariance(&self, horizon: usize) -> Result<Vec<Mat<f64>>, MgarchError> {
        if horizon == 0 {
            return Err(MgarchError::InvalidHorizon);
        }
        // Per-series variance forecast paths (each length `horizon`).
        let mut var_paths = Vec::with_capacity(self.stage.k);
        for (i, res) in self.stage.univariate.iter().enumerate() {
            let path = res
                .forecast_variance(horizon)
                .map_err(|e| MgarchError::Univariate {
                    series: i,
                    source: e,
                })?;
            var_paths.push(path);
        }
        let mut out = Vec::with_capacity(horizon);
        for m in 0..horizon {
            let d: Vec<f64> = var_paths.iter().map(|p| p[m].sqrt()).collect();
            out.push(scale_correlation(self.correlation.as_ref(), &d));
        }
        Ok(out)
    }
}

/// Builds `D R D` for a diagonal `D = diag(d)`.
pub(crate) fn scale_correlation(r: MatRef<'_, f64>, d: &[f64]) -> Mat<f64> {
    let k = r.nrows();
    Mat::from_fn(k, k, |i, j| d[i] * r[(i, j)] * d[j])
}

/// The full Gaussian log-likelihood of a CCC model with correlation `R`:
///
/// ```text
/// L = -0.5 sum_t [ k ln(2 pi) + sum_i ln sigma2_{i,t} + ln|R| + z_t' R^{-1} z_t ].
/// ```
///
/// Factorizing `R` once (constant across `t`) and reusing the factor for the
/// per-`t` quadratic forms.
///
/// # Errors
///
/// [`MgarchError::Linalg`] if `R` cannot be factorized.
pub(crate) fn ccc_loglik(stage: &UnivariateStage, r: MatRef<'_, f64>) -> Result<f64, MgarchError> {
    let chol = cholesky(r)?;
    let ln_det_r = chol.log_det();
    let ln_2pi = (2.0 * core::f64::consts::PI).ln();
    let k = stage.k as f64;
    let mut ll = 0.0;
    for t in 0..stage.nobs {
        let quad = quad_form(chol.factor.as_ref(), &stage.z[t]);
        let mut ln_det_h = 0.0;
        for &s2 in &stage.sigma2[t] {
            ln_det_h += s2.ln();
        }
        ll += -0.5 * (k * ln_2pi + ln_det_h + ln_det_r + quad);
    }
    Ok(ll)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use tsecon_garch::{DistSpec, GarchSpec, MeanSpec, VolSpec};

    /// A tiny SplitMix64 -> standard-normal generator (tests must not depend
    /// on tsecon-rng).
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

    /// Two mildly heteroskedastic, correlated series long enough for the
    /// univariate fits.
    fn synthetic() -> Vec<Vec<f64>> {
        let mut rng = Rng(0x1234_5678);
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

    /// The CCC log-likelihood equals a from-scratch bivariate Gaussian
    /// recomputation with an explicit 2x2 inverse of `H_t = D_t R D_t`.
    #[test]
    fn ccc_loglik_matches_bruteforce_2x2() {
        let fit = CccGarch::new(spec()).fit(&synthetic()).unwrap();
        let r = &fit.correlation;
        let (r00, r01, r11) = (r[(0, 0)], r[(0, 1)], r[(1, 1)]);
        let ln_2pi = (2.0 * std::f64::consts::PI).ln();

        // eps_t = returns (zero mean); recompute directly from residuals.
        let e0 = fit.stage.univariate[0].residuals().to_vec();
        let e1 = fit.stage.univariate[1].residuals().to_vec();
        let mut brute = 0.0;
        for t in 0..fit.nobs() {
            let (s0, s1) = (fit.stage.sigma2[t][0].sqrt(), fit.stage.sigma2[t][1].sqrt());
            // H = D R D.
            let h00 = s0 * r00 * s0;
            let h11 = s1 * r11 * s1;
            let h01 = s0 * r01 * s1;
            let det = h00 * h11 - h01 * h01;
            let (x0, x1) = (e0[t], e1[t]);
            // x' H^{-1} x with the explicit 2x2 inverse.
            let quad = (h11 * x0 * x0 - 2.0 * h01 * x0 * x1 + h00 * x1 * x1) / det;
            brute += -0.5 * (2.0 * ln_2pi + det.ln() + quad);
        }
        assert!(
            (brute - fit.loglik).abs() <= 1e-9 * brute.abs().max(1.0),
            "brute {brute} vs fit {}",
            fit.loglik
        );
    }

    /// The CCC log-likelihood also equals the "univariate terms + Gaussian
    /// copula correction" decomposition (Bollerslev 1990).
    #[test]
    fn ccc_loglik_copula_decomposition() {
        let fit = CccGarch::new(spec()).fit(&synthetic()).unwrap();
        let stage = &fit.stage;
        let chol = cholesky(fit.correlation.as_ref()).unwrap();
        let ln_det_r = chol.log_det();
        // Copula correction: -0.5 sum_t (ln|R| + z' R^{-1} z - z' z).
        let mut copula = 0.0;
        for t in 0..fit.nobs() {
            let quad = quad_form(chol.factor.as_ref(), &stage.z[t]);
            let zz: f64 = stage.z[t].iter().map(|v| v * v).sum();
            copula += -0.5 * (ln_det_r + quad - zz);
        }
        let recomposed = stage.volatility_loglik + copula;
        assert!((recomposed - fit.loglik).abs() <= 1e-10 * recomposed.abs().max(1.0));
    }
}
