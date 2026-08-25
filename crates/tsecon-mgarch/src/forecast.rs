//! h-step correlation and covariance forecasts for the fitted DCC family.
//!
//! # Convention — read this before comparing to other packages
//!
//! Multi-step DCC correlation forecasts have **no closed form**: the map
//! `Q -> diag(Q)^{-1/2} Q diag(Q)^{-1/2}` is nonlinear, so
//! `E[R_{T+h}] != corr(E[Q_{T+h}])` exactly. This module implements the
//! standard **Engle-Sheppard (2001) forward recursion on `Q`** (their first
//! approximation, `E[z_t z_t'] ~= E[Q_t]`):
//!
//! ```text
//! h = 1:      Q_{T+1}    exact (in the information set),
//! h >= 2:     E[Q_{T+h}] = (1 - a - b) Qbar + (a + b) E[Q_{T+h-1}],
//! R_{T+h}    := corr(E[Q_{T+h}]).
//! ```
//!
//! Because each `E[Q_{T+h}]` is a convex combination of positive
//! semi-definite matrices, every forecast correlation matrix is a *proper*
//! correlation matrix by construction, and the path converges geometrically
//! (rate `a + b`) to the unconditional `corr(Qbar)`.
//!
//! Per variant:
//! * **cDCC** — same recursion with `S` in place of `Qbar`. Here
//!   `E_{t-1}[z*_t z*_t'] = Q_t` holds *exactly* (Aielli 2013), so the only
//!   approximation left is the final nonlinear normalization.
//! * **ADCC** — under the targeting approximation `E[n_t n_t'] ~= Nbar`, the
//!   asymmetric news term cancels against the `- g Nbar` intercept and the
//!   mean recursion is again `(1-a-b) Qbar + (a+b) E[Q_{T+h-1}]`.
//!
//! Covariances are assembled as `H_{T+h} = D_{T+h} R_{T+h} D_{T+h}` with
//! `D_{T+h}` from the per-series analytic univariate variance forecasts —
//! the further standard approximation `E[D R D] ~= E[D] E[R] E[D]`.

use tsecon_linalg::faer::Mat;

use crate::ccc::scale_correlation;
use crate::dcc::DccFit;
use crate::error::MgarchError;
use crate::util::{cholesky, corr_from_cov};

/// h-step forecasts from a fitted DCC-family model.
#[derive(Debug, Clone)]
pub struct DccForecast {
    /// Forecast correlation matrices `R_{T+1}, ..., R_{T+h}` (each `k x k`),
    /// by the Engle-Sheppard `Q`-recursion convention (see the module docs).
    pub correlation: Vec<Mat<f64>>,
    /// Forecast covariance matrices `H_{T+m} = D_{T+m} R_{T+m} D_{T+m}`.
    pub covariance: Vec<Mat<f64>>,
    /// Per-series analytic variance forecasts, time-major:
    /// `variance[m][i] = sigma2_{i,T+1+m}`.
    pub variance: Vec<Vec<f64>>,
}

impl DccFit {
    /// h-step correlation/covariance forecasts for `m = 1..=horizon`.
    ///
    /// Convention: the Engle-Sheppard (2001) forward recursion on `Q` —
    /// exact `Q_{T+1}` at `h = 1`, then
    /// `E[Q_{T+h}] = (1-a-b) Qbar + (a+b) E[Q_{T+h-1}]` normalized to a
    /// correlation each step (see [`crate::forecast`] for why this is an
    /// approximation for `h >= 2` and what it converges to). Covariances
    /// scale the forecast correlations by the per-series analytic
    /// univariate variance forecasts.
    ///
    /// # Errors
    ///
    /// * [`MgarchError::InvalidHorizon`] if `horizon == 0`;
    /// * [`MgarchError::Univariate`] if a univariate variance forecast
    ///   fails (e.g. EGARCH beyond one step);
    /// * [`MgarchError::Linalg`] if a forecast correlation cannot be
    ///   factorized (the PD certificate).
    pub fn forecast(&self, horizon: usize) -> Result<DccForecast, MgarchError> {
        if horizon == 0 {
            return Err(MgarchError::InvalidHorizon);
        }
        let k = self.stage.k;

        // Per-series analytic variance forecast paths (each length horizon).
        let mut var_paths = Vec::with_capacity(k);
        for (i, res) in self.stage.univariate.iter().enumerate() {
            let path = res
                .forecast_variance(horizon)
                .map_err(|e| MgarchError::Univariate {
                    series: i,
                    source: e,
                })?;
            var_paths.push(path);
        }

        let persistence = self.a + self.b;
        let mut correlation = Vec::with_capacity(horizon);
        let mut covariance = Vec::with_capacity(horizon);
        let mut variance = Vec::with_capacity(horizon);
        let mut eq: Mat<f64> = self.q_next().to_owned();
        for m in 0..horizon {
            if m > 0 {
                // E[Q_{T+1+m}] = (1-a-b) Qbar + (a+b) E[Q_{T+m}].
                let prev = eq;
                eq = Mat::from_fn(k, k, |i, j| {
                    (1.0 - persistence) * self.qbar[(i, j)] + persistence * prev[(i, j)]
                });
            }
            let r = corr_from_cov(eq.as_ref());
            // Certify positive-definiteness (the factorization is the check).
            let _ = cholesky(r.as_ref())?;
            let d: Vec<f64> = var_paths.iter().map(|p| p[m].sqrt()).collect();
            covariance.push(scale_correlation(r.as_ref(), &d));
            correlation.push(r);
            variance.push(var_paths.iter().map(|p| p[m]).collect());
        }
        Ok(DccForecast {
            correlation,
            covariance,
            variance,
        })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use crate::dcc::DccGarch;
    use crate::error::MgarchError;
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
        let mut rng = Rng(0xF0CA_57ED);
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

    /// `forecast(1)` and the legacy `forecast_covariance_one_step` are the
    /// same computation — bitwise-equal covariance.
    #[test]
    fn one_step_matches_legacy_forecast() {
        let fit = DccGarch::new(spec()).fit(&synthetic()).unwrap();
        let legacy = fit.forecast_covariance_one_step().unwrap();
        let fc = fit.forecast(1).unwrap();
        assert_eq!(fc.correlation.len(), 1);
        assert_eq!(fc.covariance.len(), 1);
        for i in 0..fit.k() {
            for j in 0..fit.k() {
                assert_eq!(
                    fc.covariance[0][(i, j)],
                    legacy[(i, j)],
                    "H_1[{i}][{j}] differs from the legacy one-step forecast"
                );
            }
        }
    }

    /// The forecast correlation path decays geometrically (rate `a + b`)
    /// toward the unconditional `corr(Qbar)` — the Engle-Sheppard recursion
    /// in expectation.
    #[test]
    fn forecast_converges_to_unconditional_correlation() {
        let fit = DccGarch::new(spec()).fit(&synthetic()).unwrap();
        let horizon = 200;
        let fc = fit.forecast(horizon).unwrap();
        let r_bar = corr_from_cov(fit.qbar.as_ref());
        let k = fit.k();
        let dist = |m: &tsecon_linalg::faer::Mat<f64>| -> f64 {
            let mut d = 0.0_f64;
            for i in 0..k {
                for j in 0..k {
                    d = d.max((m[(i, j)] - r_bar[(i, j)]).abs());
                }
            }
            d
        };
        let d1 = dist(&fc.correlation[0]);
        let d_last = dist(&fc.correlation[horizon - 1]);
        // Long-horizon forecast is at the unconditional level, and closer
        // than the short-horizon forecast was.
        assert!(d_last <= 1e-6, "R_200 still {d_last} from corr(Qbar)");
        assert!(d_last <= d1 + 1e-12);
        // Every forecast matrix is a correlation matrix: unit diagonal.
        for r in &fc.correlation {
            for i in 0..k {
                assert!((r[(i, i)] - 1.0).abs() <= 1e-14);
            }
        }
    }

    /// `horizon == 0` is rejected, matching the CCC forecast contract.
    #[test]
    fn zero_horizon_rejected() {
        let fit = DccGarch::new(spec()).fit(&synthetic()).unwrap();
        let err = fit.forecast(0).unwrap_err();
        assert!(matches!(err, MgarchError::InvalidHorizon));
    }
}
