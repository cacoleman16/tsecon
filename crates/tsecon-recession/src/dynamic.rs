//! Dynamic probit (Kauppi & Saikkonen 2008): the linear index carries its own
//! autoregressive term,
//!
//! ```text
//! index_t = w + x_t' b + rho * index_{t-1},   P(y_t = 1) = Phi(index_t),
//! ```
//!
//! fit by maximum likelihood over `(w, b, rho)`.
//!
//! There is no statsmodels reference for this estimator, so it is validated
//! PROPERTY-ONLY (see `tests/properties.rs`): on data from a known
//! dynamic-probit DGP it recovers `rho` and `b` within Monte-Carlo bands, and
//! its log-likelihood exceeds the static probit's on persistent data.

use tsecon_optim::{minimize, Method, NelderMeadOptions, ObjectiveFn};
use tsecon_stats::{ContinuousDist, StdNormal};

use crate::design::{loglik_null, se_and_z};
use crate::error::RecessionError;
use crate::link::Link;

/// A fitted dynamic probit.
///
/// `params` is laid out as `[w, b_0, .., b_{m-1}, rho]`; `bse` and `zstats`
/// follow the same order.
#[derive(Debug, Clone, PartialEq)]
pub struct DynamicProbitFit {
    /// Intercept `w`.
    pub w: f64,
    /// Covariate slopes `b`, one per column of `x`.
    pub beta: Vec<f64>,
    /// Autoregressive index coefficient `rho` (in `(-1, 1)`).
    pub rho: f64,
    /// The full parameter vector `[w, b.., rho]`.
    pub params: Vec<f64>,
    /// Standard errors from the inverse observed information (a numerical
    /// Hessian of the log-likelihood at the MLE).
    pub bse: Vec<f64>,
    /// z-statistics `params / bse`.
    pub zstats: Vec<f64>,
    /// Maximized log-likelihood.
    pub loglik: f64,
    /// Intercept-only log-likelihood (McFadden's denominator).
    pub loglik_null: f64,
    /// McFadden's pseudo-R^2, `1 - loglik / loglik_null`.
    pub pseudo_r2: f64,
    /// Fitted recession probabilities `Phi(index_t)`.
    pub fitted: Vec<f64>,
    /// The fitted latent index path `index_t`.
    pub index: Vec<f64>,
    /// Sample size.
    pub n: usize,
    /// Whether the optimizer reported convergence.
    pub converged: bool,
}

/// Build the recursive index path from natural parameters `[w, b.., rho]`.
///
/// The pre-sample index is initialized at the stationary mean `w / (1 - rho)`
/// (well defined because `|rho| < 1`), the standard Kauppi-Saikkonen start.
fn index_path(x: &[Vec<f64>], w: f64, b: &[f64], rho: f64, n: usize) -> Vec<f64> {
    let mut idx = vec![0.0_f64; n];
    let mut prev = w / (1.0 - rho); // stationary mean initialization
    for t in 0..n {
        let mut lin = w;
        for (j, col) in x.iter().enumerate() {
            lin += b[j] * col[t];
        }
        let it = lin + rho * prev;
        idx[t] = it;
        prev = it;
    }
    idx
}

/// Total probit log-likelihood of the dynamic index at natural parameters
/// `theta = [w, b_0.., rho]`. Returns `-inf` for `|rho| >= 1` (a non-stationary
/// index has no stationary initialization), which the optimizer treats as an
/// infeasible point.
fn loglik_natural(y: &[f64], x: &[Vec<f64>], theta: &[f64], n: usize, m: usize) -> f64 {
    let w = theta[0];
    let b = &theta[1..1 + m];
    let rho = theta[1 + m];
    if rho.is_nan() || rho.abs() >= 1.0 {
        return f64::NEG_INFINITY;
    }
    let idx = index_path(x, w, b, rho, n);
    let mut ll = 0.0;
    for t in 0..n {
        ll += Link::Probit.loglik_term(y[t], idx[t]);
    }
    ll
}

/// Negative log-likelihood objective in a WORKING parameterization that keeps
/// `rho` in `(-1, 1)` smoothly via `rho = tanh(z_rho)`, so the unconstrained
/// optimizer never leaves the stationary region.
struct DynNegLogLik<'a> {
    y: &'a [f64],
    x: &'a [Vec<f64>],
    n: usize,
    m: usize,
}

impl DynNegLogLik<'_> {
    /// Map working `z = [w, b.., z_rho]` to natural `[w, b.., rho]`.
    fn to_natural(&self, z: &[f64]) -> Vec<f64> {
        let mut theta = z.to_vec();
        let last = self.m + 1;
        theta[last] = z[last].tanh();
        theta
    }
}

impl ObjectiveFn for DynNegLogLik<'_> {
    fn value(&mut self, z: &[f64]) -> f64 {
        let theta = self.to_natural(z);
        -loglik_natural(self.y, self.x, &theta, self.n, self.m)
    }
}

/// Fit the dynamic probit of `y` on covariate columns `x` (given WITHOUT a
/// constant — the intercept `w` is estimated separately) by maximum
/// likelihood over `(w, b, rho)`.
///
/// The index is initialized at its stationary mean and `rho` is held in
/// `(-1, 1)` throughout the search. Standard errors come from a numerical
/// Hessian of the log-likelihood at the optimum.
///
/// # Errors
///
/// Returns a [`RecessionError`] for malformed input (empty `y`, no covariate
/// columns, mismatched lengths, non-finite or non-binary values, a degenerate
/// response, or fewer than `m + 3` observations for `m + 2` parameters), or a
/// [singular information matrix](RecessionError::SingularInformation).
pub fn fit_dynamic_probit(y: &[f64], x: &[Vec<f64>]) -> Result<DynamicProbitFit, RecessionError> {
    let n = y.len();
    if n == 0 {
        return Err(RecessionError::EmptyInput { what: "y" });
    }
    if x.is_empty() {
        return Err(RecessionError::NoRegressors);
    }
    let m = x.len();
    for (j, col) in x.iter().enumerate() {
        if col.len() != n {
            return Err(RecessionError::DimensionMismatch {
                what: "an x covariate column",
                expected: n,
                got: col.len(),
            });
        }
        if col.iter().any(|v| !v.is_finite()) {
            let _ = j;
            return Err(RecessionError::NonFinite { what: "x" });
        }
    }
    let mut ones = 0usize;
    for (t, &yt) in y.iter().enumerate() {
        if !yt.is_finite() {
            return Err(RecessionError::NonFinite { what: "y" });
        }
        if yt != 0.0 && yt != 1.0 {
            return Err(RecessionError::NonBinaryResponse {
                index: t,
                value: yt,
            });
        }
        if yt == 1.0 {
            ones += 1;
        }
    }
    if ones == 0 || ones == n {
        return Err(RecessionError::Degenerate { ones, n });
    }
    let k = m + 2; // w, b (m of them), rho
    if n <= k {
        return Err(RecessionError::DegreesOfFreedom { n, k });
    }

    // Warm start: a static probit on [const, x] gives good (w, b); rho starts
    // at a mildly persistent 0.2 (working z_rho = atanh(0.2)).
    let (w0, b0) = static_warm_start(y, x, n, m);
    let mut z0 = Vec::with_capacity(k);
    z0.push(w0);
    z0.extend_from_slice(&b0);
    z0.push(0.2_f64.atanh());

    let mut obj = DynNegLogLik { y, x, n, m };
    // Derivative-free simplex: the recursive index makes an analytic gradient
    // awkward, and Nelder-Mead is robust on the smooth-but-recursive surface.
    let opts = NelderMeadOptions {
        max_iter: Some(20_000),
        max_fevals: Some(40_000),
        restarts: 2,
        ..NelderMeadOptions::default()
    };
    let res = minimize(&mut obj, &z0, &Method::NelderMead(opts))?;
    let theta = obj.to_natural(&res.x);

    let w = theta[0];
    let beta = theta[1..1 + m].to_vec();
    let rho = theta[1 + m];

    let idx = index_path(x, w, &beta, rho, n);
    let fitted: Vec<f64> = idx.iter().map(|&z| StdNormal.cdf(z)).collect();

    // Observed information = negative numerical Hessian of the log-likelihood
    // in NATURAL parameters, then covariance = its inverse.
    let info = neg_hessian(y, x, &theta, n, m);
    let cov = crate::design::inv_spd_rowmajor(&info, k)?;
    let (bse, zstats) = se_and_z(&theta, &cov, k);

    let loglik = -res.f;
    let ll_null = loglik_null(ones, n);
    let pseudo_r2 = 1.0 - loglik / ll_null;

    Ok(DynamicProbitFit {
        w,
        beta,
        rho,
        params: theta,
        bse,
        zstats,
        loglik,
        loglik_null: ll_null,
        pseudo_r2,
        fitted,
        index: idx,
        n,
        converged: res.converged,
    })
}

/// Static-probit warm start for `(w, b)` using ordinary least squares of the
/// centered response on `[1, x]` (a cheap linear-probability approximation;
/// only a starting point, never reported). Falls back to zeros if the normal
/// equations are ill-conditioned.
fn static_warm_start(y: &[f64], x: &[Vec<f64>], n: usize, m: usize) -> (f64, Vec<f64>) {
    // One-pass means.
    let ybar = y.iter().sum::<f64>() / n as f64;
    let xbar: Vec<f64> = x.iter().map(|c| c.iter().sum::<f64>() / n as f64).collect();
    // Univariate slopes b_j = cov(x_j, y) / var(x_j); intercept from the means.
    // (A diagonal approximation — enough to seed the ML search.)
    let mut b = vec![0.0_f64; m];
    for j in 0..m {
        let mut cov = 0.0;
        let mut var = 0.0;
        for t in 0..n {
            let dx = x[j][t] - xbar[j];
            cov += dx * (y[t] - ybar);
            var += dx * dx;
        }
        b[j] = if var > 1e-12 { cov / var } else { 0.0 };
    }
    // Scale the linear-probability slopes to a probit index scale (~ / 0.4,
    // the standard-normal density at the median) and set w so the mean index
    // matches Phi^{-1}(ybar) roughly.
    let scale = 1.0 / 0.4;
    for bj in b.iter_mut() {
        *bj *= scale;
    }
    let mut mean_lin = 0.0;
    for j in 0..m {
        mean_lin += b[j] * xbar[j];
    }
    let target = StdNormal.ppf(ybar.clamp(1e-3, 1.0 - 1e-3)).unwrap_or(0.0);
    let w = target - mean_lin;
    (w, b)
}

/// The negative Hessian of the total log-likelihood at natural parameters
/// `theta`, by central finite differences (observed information). Row-major
/// `k x k`, symmetric.
fn neg_hessian(y: &[f64], x: &[Vec<f64>], theta: &[f64], n: usize, m: usize) -> Vec<f64> {
    let k = theta.len();
    let ll = |p: &[f64]| loglik_natural(y, x, p, n, m);
    // Per-coordinate steps; the fourth-root of machine epsilon balances the
    // O(h^2) truncation and O(eps/h^2) rounding errors of a second difference.
    let base = f64::EPSILON.powf(0.25);
    let h: Vec<f64> = theta.iter().map(|&v| base * v.abs().max(1.0)).collect();

    let mut hess = vec![0.0_f64; k * k];
    let f0 = ll(theta);
    let mut work = theta.to_vec();
    // Diagonal: (f(x+h) + f(x-h) - 2 f0) / h^2.
    for i in 0..k {
        let hi = h[i];
        work[i] = theta[i] + hi;
        let fp = ll(&work);
        work[i] = theta[i] - hi;
        let fm = ll(&work);
        work[i] = theta[i];
        hess[i * k + i] = (fp + fm - 2.0 * f0) / (hi * hi);
    }
    // Off-diagonal: the four-point mixed second difference.
    for i in 0..k {
        for j in (i + 1)..k {
            let hi = h[i];
            let hj = h[j];
            work[i] = theta[i] + hi;
            work[j] = theta[j] + hj;
            let fpp = ll(&work);
            work[j] = theta[j] - hj;
            let fpm = ll(&work);
            work[i] = theta[i] - hi;
            work[j] = theta[j] + hj;
            let fmp = ll(&work);
            work[j] = theta[j] - hj;
            let fmm = ll(&work);
            work[i] = theta[i];
            work[j] = theta[j];
            let d = (fpp - fpm - fmp + fmm) / (4.0 * hi * hj);
            hess[i * k + j] = d;
            hess[j * k + i] = d;
        }
    }
    // Observed information is the negative Hessian of the log-likelihood.
    for v in hess.iter_mut() {
        *v = -*v;
    }
    hess
}
