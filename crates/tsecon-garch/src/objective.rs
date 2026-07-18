//! The estimation-time objective: a fused, allocation-free evaluation of
//! the negative log-likelihood in the optimizer's unconstrained working
//! space, with an analytic gradient where one is available.
//!
//! # Why this exists
//!
//! [`GarchModel::loglike`](crate::GarchModel::loglike) is the *reference*
//! implementation — clear, allocating, and written to read like the
//! algebra. Estimation calls it a few thousand times per fit, so `fit`
//! drives this module instead:
//!
//! * the residual, variance, and per-observation-likelihood vectors are
//!   fused into a single pass over reusable buffers (four heap allocations
//!   of length `T` per evaluation become zero);
//! * for the GARCH/GJR recursion under normal innovations the gradient is
//!   analytic, so a gradient costs one extra sweep instead of the `2k`
//!   likelihood evaluations a central difference needs.
//!
//! The fused value is **bit-identical** to
//! [`GarchModel::loglike`](crate::GarchModel::loglike): the same terms are
//! formed in the same order and summed in the same order. The unit tests at
//! the bottom of this file assert exact equality on a grid of parameters,
//! not approximate equality.
//!
//! Specifications outside the analytic-gradient set (EGARCH, Student-t)
//! fall back to [`GarchModel::loglike`] for the value and to the
//! optimizer's central differences for the gradient, exactly as before.
//!
//! # The analytic gradient
//!
//! With `eps_t = y_t - mu` and
//!
//! ```text
//! v_t = omega + sum_i alpha_i x_{t-i} + sum_i gamma_i x^-_{t-i}
//!             + sum_j beta_j v_{t-j},
//! x_s = eps_s^2,   x^-_s = eps_s^2 1[eps_s < 0]
//! ```
//!
//! (presample `x -> backcast`, `x^- -> backcast/2`, `v -> backcast`, all
//! held fixed as in `arch`), the variance derivatives satisfy the same
//! recursion driven by different forcing terms:
//!
//! ```text
//! dv_t/domega   = 1          + sum_j beta_j dv_{t-j}/domega
//! dv_t/dalpha_i = x_{t-i}    + sum_j beta_j dv_{t-j}/dalpha_i
//! dv_t/dgamma_i = x^-_{t-i}  + sum_j beta_j dv_{t-j}/dgamma_i
//! dv_t/dbeta_j  = v_{t-j}    + sum_k beta_k dv_{t-k}/dbeta_j
//! dv_t/dmu      = -2 sum_i alpha_i eps_{t-i}
//!                 -2 sum_i gamma_i eps_{t-i} 1[eps_{t-i} < 0]
//!                 + sum_j beta_j dv_{t-j}/dmu
//! ```
//!
//! with all presample derivatives zero (the backcast is a constant of the
//! estimation, not a function of the parameters). Under normal innovations
//! `l_t = -1/2 (ln 2pi + ln v_t + eps_t^2 / v_t)`, so
//!
//! ```text
//! dl/dtheta = sum_t w_t dv_t/dtheta,     w_t = -1/2 (1/v_t - eps_t^2/v_t^2)
//! dl/dmu   += sum_t eps_t / v_t                (the direct residual term)
//! ```
//!
//! Finally the chain rule through the working-space reparameterization is
//! diagonal: only `omega = exp(z)` is transformed, contributing the factor
//! `domega/dz = omega`.

use tsecon_optim::ObjectiveFn;

use crate::model::GarchModel;
use crate::spec::{DistSpec, VolSpec};

/// Fused negative-log-likelihood objective over the working vector `z`.
pub(crate) struct FitObjective<'a> {
    model: &'a GarchModel,
    /// Number of mean parameters (0 or 1).
    nm: usize,
    /// Lag orders of the variance recursion.
    p: usize,
    o: usize,
    q: usize,
    /// Total parameter count.
    k: usize,
    /// Index of `omega` in the parameter vector.
    omega_idx: usize,
    /// Whether `omega` is estimated as `exp(z)`.
    omega_log: bool,
    /// Index of the Student-t `nu`, when present.
    nu_idx: Option<usize>,
    /// Bounds of the `nu` box transform.
    nu_bounds: (f64, f64),
    /// Whether the fused fast path applies (GARCH/GJR recursion).
    fast_value: bool,
    /// Whether the analytic gradient applies (GARCH/GJR + normal).
    analytic_grad: bool,
    /// Natural-parameter scratch (length `k`).
    theta: Vec<f64>,
    /// Conditional-variance path (length `T`).
    sigma2: Vec<f64>,
    /// Rolling window of `dv_{t-j}/dtheta`, `j = 1..=q`, row-major
    /// `q x k` with row `j-1` holding lag `j`.
    dhist: Vec<f64>,
    /// Accumulator for `dl/dtheta` (length `k`).
    dtheta: Vec<f64>,
    /// `dv_t/dtheta` for the current `t` (length `k`).
    dcur: Vec<f64>,
    /// Scratch copy of the `alpha`/`gamma`/`beta` blocks, in that order.
    coef: Vec<f64>,
}

impl<'a> FitObjective<'a> {
    /// Builds the objective for `model`, mirroring the working-space layout
    /// that [`GarchModel::fit`](crate::GarchModel::fit) uses.
    pub(crate) fn new(model: &'a GarchModel, nu_bounds: (f64, f64)) -> Self {
        let spec = model.spec();
        let (p, o, q) = spec.vol.lags();
        let k = spec.n_params();
        let nm = spec.n_mean_params();
        let omega_log = !matches!(spec.vol, VolSpec::Egarch { .. });
        let nu_idx = matches!(spec.dist, DistSpec::StudentT).then_some(k - 1);
        let fast_value = !matches!(spec.vol, VolSpec::Egarch { .. });
        let analytic_grad = fast_value && matches!(spec.dist, DistSpec::Normal);
        let t = model.y().len();
        Self {
            model,
            nm,
            p,
            o,
            q,
            k,
            omega_idx: nm,
            omega_log,
            nu_idx,
            nu_bounds,
            fast_value,
            analytic_grad,
            theta: vec![0.0; k],
            sigma2: vec![0.0; t],
            dhist: vec![0.0; q * k],
            dtheta: vec![0.0; k],
            dcur: vec![0.0; k],
            coef: Vec::with_capacity(p + o + q),
        }
    }

    /// Maps the working vector `z` into natural parameters, writing into
    /// [`Self::theta`]. Returns `false` if the map or the admissibility
    /// check fails (an infeasible trial point).
    fn set_natural(&mut self, z: &[f64]) -> bool {
        if z.len() != self.k || z.iter().any(|v| !v.is_finite()) {
            return false;
        }
        self.theta.copy_from_slice(z);
        if self.omega_log {
            self.theta[self.omega_idx] = z[self.omega_idx].exp();
        }
        if let Some(i) = self.nu_idx {
            // `Bounded::forward`: theta = lo + (hi - lo) / (1 + exp(-z)).
            let (lo, hi) = self.nu_bounds;
            self.theta[i] = lo + (hi - lo) / (1.0 + (-z[i]).exp());
        }
        self.theta.iter().all(|v| v.is_finite())
            && self.model.spec().validate_params(&self.theta).is_ok()
    }

    /// Multiplies a natural-space gradient in place by the diagonal
    /// Jacobian `dtheta/dz` of the working-space map.
    fn chain_to_working(&self, grad: &mut [f64]) {
        if self.omega_log {
            grad[self.omega_idx] *= self.theta[self.omega_idx];
        }
        if let Some(i) = self.nu_idx {
            let (lo, hi) = self.nu_bounds;
            // d/dz [lo + (hi-lo) s(z)] = (theta - lo) (hi - theta) / (hi - lo).
            let t = self.theta[i];
            grad[i] *= (t - lo) * (hi - t) / (hi - lo);
        }
    }

    /// The `alpha`/`gamma`/`beta` blocks of [`Self::theta`].
    fn blocks(&self) -> (f64, &[f64], &[f64], &[f64]) {
        let base = self.nm;
        let omega = self.theta[base];
        let a = &self.theta[base + 1..base + 1 + self.p];
        let g = &self.theta[base + 1 + self.p..base + 1 + self.p + self.o];
        let b = &self.theta[base + 1 + self.p + self.o..base + 1 + self.p + self.o + self.q];
        (omega, a, g, b)
    }

    /// Fused GARCH/GJR variance recursion plus log-likelihood, over
    /// [`Self::sigma2`]. Returns `None` if the variance path leaves
    /// `(0, inf)` — the same condition
    /// [`GarchModel::conditional_variance`](crate::GarchModel::conditional_variance)
    /// reports as an error.
    ///
    /// Term formation and summation order match
    /// [`GarchModel::loglike`](crate::GarchModel::loglike) exactly, so the
    /// two agree bit for bit.
    fn fused_loglike(&mut self) -> Option<f64> {
        let y = self.model.y();
        let bc = self.model.backcast_value();
        let nobs = y.len();
        let mu = if self.nm > 0 { self.theta[0] } else { 0.0 };
        // Copy the (tiny) coefficient blocks into scratch so the borrow of
        // `self.theta` ends before `self.sigma2` is written.
        let (omega, na, ng) = {
            let base = self.nm;
            let (na, ng, nb) = (self.p, self.o, self.q);
            let omega = self.theta[base];
            self.coef.clear();
            self.coef
                .extend_from_slice(&self.theta[base + 1..base + 1 + na + ng + nb]);
            (omega, na, ng)
        };
        let (alphas, rest) = self.coef.split_at(na);
        let (gammas, betas) = rest.split_at(ng);
        let sigma2 = &mut self.sigma2;

        // Pass 1: the variance path. The presample conventions only bite
        // for `t < max_lag`, so that prefix is peeled off and the bulk of
        // the series runs through branch-free inner loops. Terms are formed
        // in the same order in both halves, so the sum is unchanged.
        let head = self.p.max(self.o).max(self.q).min(nobs);
        let mut ok = true;
        for t in 0..head {
            let mut v = omega;
            for (i, &a) in alphas.iter().enumerate() {
                v += a * match t.checked_sub(i + 1) {
                    Some(s) => {
                        let e = y[s] - mu;
                        e * e
                    }
                    None => bc,
                };
            }
            for (i, &g) in gammas.iter().enumerate() {
                v += g * match t.checked_sub(i + 1) {
                    Some(s) => {
                        let e = y[s] - mu;
                        if e < 0.0 {
                            e * e
                        } else {
                            0.0
                        }
                    }
                    None => 0.5 * bc,
                };
            }
            for (j, &b) in betas.iter().enumerate() {
                v += b * match t.checked_sub(j + 1) {
                    Some(s) => sigma2[s],
                    None => bc,
                };
            }
            sigma2[t] = v;
            ok &= v > 0.0 && v.is_finite();
        }
        for t in head..nobs {
            let mut v = omega;
            for (i, &a) in alphas.iter().enumerate() {
                let e = y[t - i - 1] - mu;
                v += a * (e * e);
            }
            for (i, &g) in gammas.iter().enumerate() {
                let e = y[t - i - 1] - mu;
                v += g * if e < 0.0 { e * e } else { 0.0 };
            }
            for (j, &b) in betas.iter().enumerate() {
                v += b * sigma2[t - j - 1];
            }
            sigma2[t] = v;
            ok &= v > 0.0 && v.is_finite();
        }
        if !ok {
            return None;
        }

        // Pass 2: the log-likelihood, in the reference term order.
        let sum = match self.model.spec().dist {
            DistSpec::Normal => {
                let ln2pi = (2.0 * core::f64::consts::PI).ln();
                let mut acc = 0.0;
                for (t, &s2) in sigma2.iter().enumerate() {
                    let e = y[t] - mu;
                    acc += -0.5 * (ln2pi + s2.ln() + e * e / s2);
                }
                acc
            }
            DistSpec::StudentT => {
                let nu = self.theta[self.k - 1];
                let c = tsecon_stats::special::ln_gamma(0.5 * (nu + 1.0))
                    - tsecon_stats::special::ln_gamma(0.5 * nu)
                    - 0.5 * (core::f64::consts::PI * (nu - 2.0)).ln();
                let mut acc = 0.0;
                for (t, &s2) in sigma2.iter().enumerate() {
                    let e = y[t] - mu;
                    acc +=
                        c - 0.5 * s2.ln() - 0.5 * (nu + 1.0) * (e * e / (s2 * (nu - 2.0))).ln_1p();
                }
                acc
            }
        };
        sum.is_finite().then_some(sum)
    }

    /// Analytic gradient of the *total* log-likelihood with respect to the
    /// natural parameters, for GARCH/GJR under normal innovations.
    /// Assumes [`Self::fused_loglike`] has just populated
    /// [`Self::sigma2`] at the same `theta`.
    fn analytic_dloglike(&mut self) -> Option<Vec<f64>> {
        let y = self.model.y();
        let bc = self.model.backcast_value();
        let nobs = y.len();
        let k = self.k;
        let (p, o, q) = (self.p, self.o, self.q);
        let mu = if self.nm > 0 { self.theta[0] } else { 0.0 };
        let (_, alphas, gammas, betas) = {
            let (w, a, g, b) = self.blocks();
            (w, a.to_vec(), g.to_vec(), b.to_vec())
        };
        // Parameter-vector offsets of each coefficient block.
        let w_idx = self.nm;
        let a0 = w_idx + 1;
        let g0 = a0 + p;
        let b0 = g0 + o;

        self.dtheta.iter_mut().for_each(|v| *v = 0.0);
        self.dhist.iter_mut().for_each(|v| *v = 0.0);

        for t in 0..nobs {
            let dcur = &mut self.dcur;
            dcur.iter_mut().for_each(|v| *v = 0.0);

            // Forcing terms.
            dcur[w_idx] = 1.0;
            for i in 0..p {
                dcur[a0 + i] = match t.checked_sub(i + 1) {
                    Some(s) => {
                        let e = y[s] - mu;
                        e * e
                    }
                    None => bc,
                };
            }
            for i in 0..o {
                dcur[g0 + i] = match t.checked_sub(i + 1) {
                    Some(s) => {
                        let e = y[s] - mu;
                        if e < 0.0 {
                            e * e
                        } else {
                            0.0
                        }
                    }
                    None => 0.5 * bc,
                };
            }
            for j in 0..q {
                dcur[b0 + j] = match t.checked_sub(j + 1) {
                    Some(s) => self.sigma2[s],
                    None => bc,
                };
            }
            if self.nm > 0 {
                let mut dmu = 0.0;
                for (i, &a) in alphas.iter().enumerate() {
                    if let Some(s) = t.checked_sub(i + 1) {
                        dmu += a * (-2.0) * (y[s] - mu);
                    }
                }
                for (i, &g) in gammas.iter().enumerate() {
                    if let Some(s) = t.checked_sub(i + 1) {
                        let e = y[s] - mu;
                        if e < 0.0 {
                            dmu += g * (-2.0) * e;
                        }
                    }
                }
                dcur[0] = dmu;
            }

            // Autoregressive part: + sum_j beta_j dv_{t-j}/dtheta. Presample
            // derivatives are zero, which `dhist` already encodes.
            for (j, &b) in betas.iter().enumerate() {
                if t > j {
                    let row = &self.dhist[j * k..(j + 1) * k];
                    for (d, &h) in dcur.iter_mut().zip(row) {
                        *d += b * h;
                    }
                }
            }

            // Accumulate dl/dtheta.
            let v = self.sigma2[t];
            let e = y[t] - mu;
            let wgt = -0.5 * (1.0 / v - (e * e) / (v * v));
            for (acc, &d) in self.dtheta.iter_mut().zip(dcur.iter()) {
                *acc += wgt * d;
            }
            if self.nm > 0 {
                self.dtheta[0] += e / v;
            }

            // Roll the history window.
            if q > 0 {
                if q > 1 {
                    self.dhist.copy_within(0..(q - 1) * k, k);
                }
                self.dhist[..k].copy_from_slice(&self.dcur);
            }
        }

        self.dtheta
            .iter()
            .all(|v| v.is_finite())
            .then(|| self.dtheta.clone())
    }
}

impl ObjectiveFn for FitObjective<'_> {
    fn value(&mut self, z: &[f64]) -> f64 {
        if !self.set_natural(z) {
            return f64::INFINITY;
        }
        let ll = if self.fast_value {
            self.fused_loglike()
        } else {
            self.model.loglike(&self.theta).ok()
        };
        match ll {
            Some(v) if v.is_finite() => -v,
            _ => f64::INFINITY,
        }
    }

    fn gradient(&mut self, z: &[f64]) -> Option<Vec<f64>> {
        if !self.analytic_grad || !self.set_natural(z) {
            return None;
        }
        // Populates `sigma2` at this theta, and rejects infeasible paths.
        self.fused_loglike()?;
        let mut g = self.analytic_dloglike()?;
        // Objective is the *negative* log-likelihood.
        g.iter_mut().for_each(|v| *v = -*v);
        self.chain_to_working(&mut g);
        g.iter().all(|v| v.is_finite()).then_some(g)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::spec::{GarchSpec, MeanSpec};

    fn series(n: usize) -> Vec<f64> {
        // Deterministic, mildly heteroskedastic, sign-varying.
        let mut s = 0.7_f64;
        let mut out = Vec::with_capacity(n);
        for t in 0..n {
            s = (s * 1.7 + 0.31).fract();
            let e = (s - 0.5) * (1.0 + 0.5 * ((t as f64) * 0.03).sin());
            out.push(e * 2.0);
        }
        out
    }

    fn model(vol: VolSpec, mean: MeanSpec, dist: DistSpec) -> GarchModel {
        GarchModel::new(&series(400), GarchSpec { mean, vol, dist }).unwrap()
    }

    /// The fused value must equal the reference `loglike` bit for bit.
    #[test]
    fn fused_value_is_bit_identical_to_reference() {
        let cases: Vec<(VolSpec, MeanSpec, DistSpec, Vec<f64>)> = vec![
            (
                VolSpec::Garch { p: 1, q: 1 },
                MeanSpec::Constant,
                DistSpec::Normal,
                vec![0.01, 0.05, 0.08, 0.90],
            ),
            (
                VolSpec::Garch { p: 1, q: 1 },
                MeanSpec::Zero,
                DistSpec::Normal,
                vec![0.05, 0.10, 0.85],
            ),
            (
                VolSpec::Garch { p: 2, q: 1 },
                MeanSpec::Constant,
                DistSpec::StudentT,
                vec![-0.02, 0.04, 0.06, 0.03, 0.85, 7.5],
            ),
            (
                VolSpec::Gjr { p: 1, o: 1, q: 1 },
                MeanSpec::Constant,
                DistSpec::Normal,
                vec![0.03, 0.05, 0.04, 0.06, 0.88],
            ),
            (
                VolSpec::Gjr { p: 1, o: 1, q: 2 },
                MeanSpec::Zero,
                DistSpec::StudentT,
                vec![0.05, 0.04, 0.06, 0.50, 0.35, 12.0],
            ),
        ];
        for (vol, mean, dist, theta) in cases {
            let m = model(vol, mean, dist);
            let mut obj = FitObjective::new(&m, (2.05, 500.0));
            obj.theta.copy_from_slice(&theta);
            let fused = obj.fused_loglike().expect("feasible");
            let reference = m.loglike(&theta).expect("feasible");
            assert_eq!(
                fused.to_bits(),
                reference.to_bits(),
                "fused != reference for {vol:?}: {fused} vs {reference}"
            );
        }
    }

    /// The analytic gradient must match a high-quality central difference of
    /// the same objective, in working space.
    #[test]
    fn analytic_gradient_matches_central_differences() {
        let cases: Vec<(VolSpec, MeanSpec, Vec<f64>)> = vec![
            (
                VolSpec::Garch { p: 1, q: 1 },
                MeanSpec::Constant,
                vec![0.01, 0.05, 0.08, 0.90],
            ),
            (
                VolSpec::Garch { p: 1, q: 1 },
                MeanSpec::Zero,
                vec![0.05, 0.10, 0.85],
            ),
            (
                VolSpec::Garch { p: 2, q: 2 },
                MeanSpec::Constant,
                vec![0.02, 0.05, 0.05, 0.03, 0.45, 0.40],
            ),
            (
                VolSpec::Gjr { p: 1, o: 1, q: 1 },
                MeanSpec::Constant,
                vec![0.03, 0.05, 0.04, 0.06, 0.88],
            ),
            (
                VolSpec::Gjr { p: 1, o: 1, q: 2 },
                MeanSpec::Zero,
                vec![0.05, 0.04, 0.06, 0.50, 0.35],
            ),
        ];
        for (vol, mean, theta) in cases {
            let m = model(vol, mean, DistSpec::Normal);
            let mut obj = FitObjective::new(&m, (2.05, 500.0));
            // Working-space point: omega enters as log(omega).
            let mut z = theta.clone();
            let wi = m.spec().n_mean_params();
            z[wi] = theta[wi].ln();

            let analytic = obj.gradient(&z).expect("analytic gradient");
            let numeric = tsecon_optim::central_difference_gradient(&mut obj, &z);
            let scale = numeric.iter().fold(1.0_f64, |m, v| m.max(v.abs()));
            for (i, (&a, &n)) in analytic.iter().zip(&numeric).enumerate() {
                assert!(
                    (a - n).abs() <= 1e-5 * scale,
                    "{vol:?} coord {i}: analytic {a} vs numeric {n}"
                );
            }
        }
    }

    /// Infeasible working points are rejected the same way the reference
    /// objective rejects them.
    #[test]
    fn infeasible_points_return_infinity_and_no_gradient() {
        let m = model(
            VolSpec::Garch { p: 1, q: 1 },
            MeanSpec::Constant,
            DistSpec::Normal,
        );
        let mut obj = FitObjective::new(&m, (2.05, 500.0));
        // alpha + beta > 1: non-stationary, rejected by validate_params.
        let z = vec![0.0, (0.05_f64).ln(), 0.30, 0.85];
        assert_eq!(obj.value(&z), f64::INFINITY);
        assert!(obj.gradient(&z).is_none());
        // Negative alpha.
        let z = vec![0.0, (0.05_f64).ln(), -0.10, 0.85];
        assert_eq!(obj.value(&z), f64::INFINITY);
        assert!(obj.gradient(&z).is_none());
    }
}
