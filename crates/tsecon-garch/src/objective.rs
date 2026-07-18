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
//! Both the GARCH/GJR recursion and the EGARCH log-variance recursion are
//! fused here, and both carry an analytic gradient under normal
//! innovations. Student-t keeps the optimizer's central differences (the
//! `nu` derivative needs a digamma that `tsecon-stats` does not ship).
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
//!
//! # The EGARCH gradient
//!
//! EGARCH models the *log* variance, with `h_t = ln sigma2_t`,
//! `z_t = eps_t exp(-h_t/2)` and `c = sqrt(2/pi)`:
//!
//! ```text
//! h_t = omega + sum_i alpha_i (|z_{t-i}| - c) + sum_i gamma_i z_{t-i}
//!             + sum_j beta_j h_{t-j}
//! ```
//!
//! (presample: the `alpha`/`gamma` terms are dropped, `h_{t-j} ->
//! ln backcast`, as in `arch`). Differentiating `z_s` gives
//!
//! ```text
//! dz_s/dtheta = -1[theta = mu] exp(-h_s/2) - (z_s/2) dh_s/dtheta
//! d|z_s|/dtheta = sign(z_s) dz_s/dtheta
//! ```
//!
//! and because `sign(z) z = |z|` the two feedback terms collapse into a
//! single time-varying coefficient. The log-variance derivative therefore
//! satisfies a linear recursion of exactly the same shape as the GARCH one,
//! only with *lag-dependent, time-varying* coefficients:
//!
//! ```text
//! dh_t/dtheta = f_t(theta)
//!             + sum_i [ -(alpha_i |z_{t-i}| + gamma_i z_{t-i}) / 2 ] dh_{t-i}/dtheta
//!             + sum_j beta_j dh_{t-j}/dtheta
//! ```
//!
//! with forcing terms
//!
//! ```text
//! f_t(omega)   = 1
//! f_t(alpha_i) = |z_{t-i}| - c
//! f_t(gamma_i) = z_{t-i}
//! f_t(beta_j)  = h_{t-j}
//! f_t(mu)      = -sum_i (alpha_i sign(z_{t-i}) + gamma_i) exp(-h_{t-i}/2)
//! ```
//!
//! (every term whose lag falls in the presample contributing zero, except
//! `f_t(beta_j) = ln backcast`, which is a constant of the estimation).
//! Under normal innovations `l_t = -1/2 (ln 2pi + h_t + eps_t^2 e^{-h_t})`,
//! so
//!
//! ```text
//! dl/dtheta = sum_t w_t dh_t/dtheta,   w_t = -1/2 (1 - eps_t^2 / sigma2_t)
//! dl/dmu   += sum_t eps_t / sigma2_t
//! ```
//!
//! The EGARCH intercept is unrestricted, so for this branch the
//! working-space Jacobian is the identity.
//!
//! # A measured, rejected shortcut
//!
//! The EGARCH value pass computes `h_t`, then `sigma2_t = exp(h_t)`, then
//! `ln sigma2_t` again inside the likelihood — an apparently redundant
//! logarithm. It is *not* redundant in floating point: `ln(exp(h))` differs
//! from `h` in the last bit for 1052 of 1500 observations on the benchmark
//! series (max absolute difference 1.1e-16), so reusing `h_t` would break
//! the bit-identity contract with
//! [`GarchModel::loglike`](crate::GarchModel::loglike). Measured, it buys
//! 34.1 -> 31.3 us per evaluation at `T = 1500` — 8%, in exchange for the
//! guarantee that the estimated and the reported likelihood are the same
//! function. The logarithm stays.
//!
//! What remains is intrinsic: the recursion is serially dependent (`z_t`
//! needs `sigma2_t` needs `exp`), so it cannot be vectorized, and an
//! `exp`-free version of the same loop still costs 12.9 us per evaluation
//! against the shipped 34.1 us. EGARCH evaluations are simply ~6x a
//! GARCH(1,1) evaluation on the same series (35.6 us vs 5.5 us).

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
    /// Whether the recursion is the EGARCH log-variance one.
    egarch: bool,
    /// Whether the analytic gradient applies (normal innovations).
    analytic_grad: bool,
    /// Largest lag of the recursion, `max(p, o, q)`.
    nlag: usize,
    /// Natural-parameter scratch (length `k`).
    theta: Vec<f64>,
    /// Conditional-variance path (length `T`).
    sigma2: Vec<f64>,
    /// EGARCH log-variance path `h_t` (length `T`; unused otherwise).
    lns2: Vec<f64>,
    /// EGARCH standardized residuals `z_t` (length `T`; unused otherwise).
    zstd: Vec<f64>,
    /// Whether the last EGARCH value pass hit the `ln(f64::MAX)` clamp, at
    /// which point the log-variance stops depending on the parameters and
    /// the analytic gradient is no longer valid.
    clamped: bool,
    /// Rolling window of `dv_{t-j}/dtheta`, `j = 1..=nlag`, row-major
    /// `nlag x k` with row `j-1` holding lag `j`.
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
        let egarch = matches!(spec.vol, VolSpec::Egarch { .. });
        let omega_log = !egarch;
        let nu_idx = matches!(spec.dist, DistSpec::StudentT).then_some(k - 1);
        let analytic_grad = matches!(spec.dist, DistSpec::Normal);
        let nlag = p.max(o).max(q);
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
            egarch,
            analytic_grad,
            nlag,
            theta: vec![0.0; k],
            sigma2: vec![0.0; t],
            lns2: vec![0.0; if egarch { t } else { 0 }],
            zstd: vec![0.0; if egarch { t } else { 0 }],
            clamped: false,
            dhist: vec![0.0; nlag * k],
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

    /// Fused EGARCH log-variance recursion plus log-likelihood, over
    /// [`Self::lns2`], [`Self::sigma2`] and [`Self::zstd`]. Returns `None`
    /// if the variance path leaves `(0, inf)`.
    ///
    /// Term formation and summation order match
    /// [`crate::recursion::egarch_recursion`] followed by
    /// [`GarchModel::loglike`](crate::GarchModel::loglike) exactly — in
    /// particular the likelihood uses `sigma2_t.ln()`, *not* the recursion's
    /// own `h_t`, because `ln(exp(h))` is not bit-identical to `h`.
    fn fused_egarch_loglike(&mut self) -> Option<f64> {
        let y = self.model.y();
        let ln_bc = self.model.backcast_value().ln();
        let mu = if self.nm > 0 { self.theta[0] } else { 0.0 };
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
        let norm_const = (2.0 / core::f64::consts::PI).sqrt();
        let ln_max = f64::MAX.ln();
        let (lns2, zstd, sigma2) = (&mut self.lns2, &mut self.zstd, &mut self.sigma2);

        let mut clamped = false;
        let mut ok = true;
        for (t, &yt) in y.iter().enumerate() {
            let mut v = omega;
            for (i, &a) in alphas.iter().enumerate() {
                if let Some(s) = t.checked_sub(i + 1) {
                    v += a * (zstd[s].abs() - norm_const);
                }
            }
            for (i, &g) in gammas.iter().enumerate() {
                if let Some(s) = t.checked_sub(i + 1) {
                    v += g * zstd[s];
                }
            }
            for (j, &b) in betas.iter().enumerate() {
                v += b * match t.checked_sub(j + 1) {
                    Some(s) => lns2[s],
                    None => ln_bc,
                };
            }
            if v > ln_max {
                v = ln_max;
                clamped = true;
            }
            lns2[t] = v;
            let s2 = v.exp();
            sigma2[t] = s2;
            zstd[t] = (yt - mu) / s2.sqrt();
            ok &= s2 > 0.0 && s2.is_finite();
        }
        self.clamped = clamped;
        if !ok {
            return None;
        }

        // The likelihood pass, in the reference term order.
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

    /// Analytic gradient of the *total* log-likelihood for the EGARCH
    /// recursion under normal innovations. Assumes
    /// [`Self::fused_egarch_loglike`] has just populated [`Self::lns2`],
    /// [`Self::sigma2`] and [`Self::zstd`] at the same `theta`.
    ///
    /// Returns `None` if the value pass hit the log-variance clamp, where
    /// `h_t` stops depending on the parameters and the recursion below no
    /// longer describes the derivative.
    fn analytic_egarch_dloglike(&mut self) -> Option<Vec<f64>> {
        if self.clamped {
            return None;
        }
        let y = self.model.y();
        let ln_bc = self.model.backcast_value().ln();
        let k = self.k;
        let (p, o, q) = (self.p, self.o, self.q);
        let nlag = self.nlag;
        let mu = if self.nm > 0 { self.theta[0] } else { 0.0 };
        let (alphas, gammas, betas) = {
            let (_, a, g, b) = self.blocks();
            (a.to_vec(), g.to_vec(), b.to_vec())
        };
        let w_idx = self.nm;
        let a0 = w_idx + 1;
        let g0 = a0 + p;
        let b0 = g0 + o;
        let norm_const = (2.0 / core::f64::consts::PI).sqrt();

        self.dtheta.iter_mut().for_each(|v| *v = 0.0);
        self.dhist.iter_mut().for_each(|v| *v = 0.0);

        for (t, &yt) in y.iter().enumerate() {
            let dcur = &mut self.dcur;
            dcur.iter_mut().for_each(|v| *v = 0.0);

            // Forcing terms. Presample `alpha`/`gamma` lags are dropped by
            // the recursion, so they force nothing.
            dcur[w_idx] = 1.0;
            for i in 0..p {
                if let Some(s) = t.checked_sub(i + 1) {
                    dcur[a0 + i] = self.zstd[s].abs() - norm_const;
                }
            }
            for i in 0..o {
                if let Some(s) = t.checked_sub(i + 1) {
                    dcur[g0 + i] = self.zstd[s];
                }
            }
            for j in 0..q {
                dcur[b0 + j] = match t.checked_sub(j + 1) {
                    Some(s) => self.lns2[s],
                    None => ln_bc,
                };
            }
            if self.nm > 0 {
                // f_t(mu) = -sum_i (alpha_i sign(z) + gamma_i) exp(-h/2),
                // and exp(-h_s/2) = 1 / sqrt(sigma2_s).
                let mut dmu = 0.0;
                for (i, &a) in alphas.iter().enumerate() {
                    if let Some(s) = t.checked_sub(i + 1) {
                        let sgn = if self.zstd[s] < 0.0 { -1.0 } else { 1.0 };
                        dmu -= a * sgn / self.sigma2[s].sqrt();
                    }
                }
                for (i, &g) in gammas.iter().enumerate() {
                    if let Some(s) = t.checked_sub(i + 1) {
                        dmu -= g / self.sigma2[s].sqrt();
                    }
                }
                dcur[0] = dmu;
            }

            // Autoregressive part. The `alpha`/`gamma` feedback through
            // `z_{t-i}` collapses to a single time-varying coefficient
            // -(alpha_i |z| + gamma_i z) / 2 on `dh_{t-i}`; the `beta`
            // feedback is the usual constant one. Presample derivatives are
            // zero, which `dhist` already encodes.
            for lag in 1..=nlag {
                if t < lag {
                    continue;
                }
                let s = t - lag;
                let mut c = 0.0;
                if let Some(&a) = alphas.get(lag - 1) {
                    c -= 0.5 * a * self.zstd[s].abs();
                }
                if let Some(&g) = gammas.get(lag - 1) {
                    c -= 0.5 * g * self.zstd[s];
                }
                if let Some(&b) = betas.get(lag - 1) {
                    c += b;
                }
                if c == 0.0 {
                    continue;
                }
                let row = &self.dhist[(lag - 1) * k..lag * k];
                for (d, &h) in dcur.iter_mut().zip(row) {
                    *d += c * h;
                }
            }

            // Accumulate dl/dtheta.
            let s2 = self.sigma2[t];
            let e = yt - mu;
            let wgt = -0.5 * (1.0 - (e * e) / s2);
            for (acc, &d) in self.dtheta.iter_mut().zip(dcur.iter()) {
                *acc += wgt * d;
            }
            if self.nm > 0 {
                self.dtheta[0] += e / s2;
            }

            // Roll the history window.
            if nlag > 1 {
                self.dhist.copy_within(0..(nlag - 1) * k, k);
            }
            self.dhist[..k].copy_from_slice(&self.dcur);
        }

        self.dtheta
            .iter()
            .all(|v| v.is_finite())
            .then(|| self.dtheta.clone())
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
        let ll = if self.egarch {
            self.fused_egarch_loglike()
        } else {
            self.fused_loglike()
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
        // Populates the variance path at this theta, and rejects
        // infeasible paths.
        let mut g = if self.egarch {
            self.fused_egarch_loglike()?;
            self.analytic_egarch_dloglike()?
        } else {
            self.fused_loglike()?;
            self.analytic_dloglike()?
        };
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
            (
                VolSpec::Egarch { p: 1, o: 1, q: 1 },
                MeanSpec::Constant,
                DistSpec::Normal,
                vec![0.02, -0.05, 0.12, -0.06, 0.95],
            ),
            (
                VolSpec::Egarch { p: 1, o: 1, q: 1 },
                MeanSpec::Zero,
                DistSpec::Normal,
                vec![-0.30, 0.20, 0.08, 0.80],
            ),
            (
                VolSpec::Egarch { p: 1, o: 0, q: 1 },
                MeanSpec::Constant,
                DistSpec::Normal,
                vec![0.01, -0.10, 0.15, 0.90],
            ),
            (
                VolSpec::Egarch { p: 2, o: 1, q: 2 },
                MeanSpec::Constant,
                DistSpec::StudentT,
                vec![-0.02, -0.08, 0.10, 0.05, -0.04, 0.50, 0.40, 9.0],
            ),
        ];
        for (vol, mean, dist, theta) in cases {
            let m = model(vol, mean, dist);
            let mut obj = FitObjective::new(&m, (2.05, 500.0));
            obj.theta.copy_from_slice(&theta);
            let fused = if obj.egarch {
                obj.fused_egarch_loglike()
            } else {
                obj.fused_loglike()
            }
            .expect("feasible");
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
            (
                VolSpec::Egarch { p: 1, o: 1, q: 1 },
                MeanSpec::Constant,
                vec![0.02, -0.05, 0.12, -0.06, 0.95],
            ),
            (
                VolSpec::Egarch { p: 1, o: 1, q: 1 },
                MeanSpec::Zero,
                vec![-0.30, 0.20, 0.08, 0.80],
            ),
            (
                VolSpec::Egarch { p: 1, o: 0, q: 1 },
                MeanSpec::Constant,
                vec![0.01, -0.10, 0.15, 0.90],
            ),
            (
                VolSpec::Egarch { p: 1, o: 1, q: 0 },
                MeanSpec::Constant,
                vec![0.01, -0.20, 0.15, -0.05],
            ),
            (
                VolSpec::Egarch { p: 2, o: 1, q: 2 },
                MeanSpec::Constant,
                vec![-0.02, -0.08, 0.10, 0.05, -0.04, 0.50, 0.40],
            ),
        ];
        for (vol, mean, theta) in cases {
            let m = model(vol, mean, DistSpec::Normal);
            let mut obj = FitObjective::new(&m, (2.05, 500.0));
            // Working-space point: for GARCH/GJR omega enters as
            // log(omega); the EGARCH log-variance intercept is untransformed.
            let mut z = theta.clone();
            let wi = m.spec().n_mean_params();
            if !matches!(vol, VolSpec::Egarch { .. }) {
                z[wi] = theta[wi].ln();
            }

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
