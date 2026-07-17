//! Static probit / logit of a binary recession indicator on leading
//! predictors, by exact-likelihood maximum likelihood.

use tsecon_optim::{minimize, BfgsOptions, Method, ObjectiveFn};

use crate::design::{inv_spd_rowmajor, linear_index, loglik_null, se_and_z, validate};
use crate::error::RecessionError;
use crate::link::Link;

/// A fitted static binary-choice recession model (probit or logit).
///
/// Coefficients, standard errors and z-statistics are ordered exactly as the
/// design columns passed to [`fit_static`] (typically constant first, then the
/// term spread and any further leading predictors).
#[derive(Debug, Clone, PartialEq)]
pub struct RecessionFit {
    /// The link used (probit or logit).
    pub link: Link,
    /// Maximum-likelihood coefficients, one per design column.
    pub params: Vec<f64>,
    /// Standard errors from the inverse observed information (the negative
    /// analytic Hessian at the MLE) — statsmodels' default `nonrobust` cov.
    pub bse: Vec<f64>,
    /// z-statistics `params / bse` (an MLE coefficient over its standard
    /// error is asymptotically standard normal).
    pub zstats: Vec<f64>,
    /// The full coefficient covariance, row-major `k x k`.
    pub cov: Vec<f64>,
    /// Maximized log-likelihood.
    pub loglik: f64,
    /// Intercept-only log-likelihood (McFadden's denominator).
    pub loglik_null: f64,
    /// McFadden's pseudo-R^2, `1 - loglik / loglik_null`.
    pub pseudo_r2: f64,
    /// Fitted recession probabilities `F(x_t' beta)`, one per observation.
    pub fitted: Vec<f64>,
    /// Sample size.
    pub n: usize,
    /// Number of parameters (design columns).
    pub k: usize,
    /// Whether the optimizer reported convergence.
    pub converged: bool,
}

/// The negative log-likelihood objective with its analytic gradient (the
/// negative score), evaluated over the full sample.
struct NegLogLik<'a> {
    y: &'a [f64],
    x: &'a [Vec<f64>],
    link: Link,
    n: usize,
    k: usize,
}

impl ObjectiveFn for NegLogLik<'_> {
    fn value(&mut self, beta: &[f64]) -> f64 {
        let idx = linear_index(self.x, beta, self.n);
        let mut ll = 0.0;
        for (&yt, &it) in self.y.iter().zip(idx.iter()) {
            ll += self.link.loglik_term(yt, it);
        }
        -ll
    }

    fn gradient(&mut self, beta: &[f64]) -> Option<Vec<f64>> {
        let idx = linear_index(self.x, beta, self.n);
        let mut grad = vec![0.0_f64; self.k];
        for (t, (&yt, &it)) in self.y.iter().zip(idx.iter()).enumerate() {
            let g = self.link.score_factor(yt, it);
            for (gj, col) in grad.iter_mut().zip(self.x.iter()) {
                *gj += g * col[t];
            }
        }
        // Objective is the NEGATIVE log-likelihood, so the gradient is the
        // negative score.
        for gj in grad.iter_mut() {
            *gj = -*gj;
        }
        Some(grad)
    }
}

/// Fit a static probit or logit of the binary recession indicator `y` on the
/// design `x` (columns given explicitly, e.g. a constant column plus the term
/// spread), by exact maximum likelihood.
///
/// The estimator returns coefficients, their standard errors and z-statistics
/// (from the inverse observed information), the maximized log-likelihood,
/// McFadden's pseudo-R^2, and the in-sample fitted probability path.
///
/// # Errors
///
/// Returns a [`RecessionError`] for malformed input (empty `y`, no regressors,
/// mismatched column lengths, non-finite or non-binary values, a degenerate
/// all-zero/all-one response, or too few degrees of freedom), for
/// (quasi-)complete [separation](RecessionError::Separation), and for a
/// [singular information matrix](RecessionError::SingularInformation).
pub fn fit_static(y: &[f64], x: &[Vec<f64>], link: Link) -> Result<RecessionFit, RecessionError> {
    let (n, k) = validate(y, x)?;
    let ones = y.iter().filter(|&&v| v == 1.0).count();

    // Maximize the exact log-likelihood == minimize its negative, from the
    // origin, with the analytic gradient. A tight gradient tolerance so the
    // MLE reproduces an independent reference (statsmodels) to ~1e-6.
    let mut obj = NegLogLik { y, x, link, n, k };
    let opts = BfgsOptions {
        grad_tol: 1e-8,
        // Secondary stop: once the log-likelihood stalls to relative 1e-12 the
        // iterate is at the MLE even if the strong-Wolfe search can no longer
        // shrink the gradient below `grad_tol` (a flat logit likelihood).
        f_tol: 1e-12,
        ..BfgsOptions::default()
    };
    let res = minimize(&mut obj, &vec![0.0; k], &Method::Bfgs(opts))?;
    let params = res.x;

    let idx = linear_index(x, &params, n);
    let fitted: Vec<f64> = idx.iter().map(|&z| link.prob(z)).collect();

    // (Quasi-)complete separation: the fitted probabilities have saturated to
    // the observed response (every recession fitted at ~1, every expansion at
    // ~0). The likelihood is then maximized only in the limit, so no finite
    // MLE exists and the reported coefficients are an artefact of the stopping
    // rule.
    let separated = (0..n).all(|t| (fitted[t] - y[t]).abs() < 1e-6);
    if separated {
        return Err(RecessionError::Separation);
    }

    // Observed information I = -H = sum_t w_t x_t x_t', then cov = I^{-1}.
    let info = observed_information(y, x, &idx, link, n, k);
    let cov = inv_spd_rowmajor(&info, k)?;
    let (bse, zstats) = se_and_z(&params, &cov, k);

    let loglik = -res.f;
    let ll_null = loglik_null(ones, n);
    let pseudo_r2 = 1.0 - loglik / ll_null;

    Ok(RecessionFit {
        link,
        params,
        bse,
        zstats,
        cov,
        loglik,
        loglik_null: ll_null,
        pseudo_r2,
        fitted,
        n,
        k,
        converged: res.converged,
    })
}

/// The observed-information matrix `I = -H = sum_t w_t x_t x_t'` (row-major),
/// where `w_t` is the link's analytic-Hessian weight at the fitted index.
fn observed_information(
    y: &[f64],
    x: &[Vec<f64>],
    idx: &[f64],
    link: Link,
    n: usize,
    k: usize,
) -> Vec<f64> {
    let mut info = vec![0.0_f64; k * k];
    for t in 0..n {
        let w = link.info_weight(y[t], idx[t]);
        for i in 0..k {
            let xi = x[i][t];
            let wxi = w * xi;
            for j in i..k {
                info[i * k + j] += wxi * x[j][t];
            }
        }
    }
    // Mirror the upper triangle into the lower.
    for i in 0..k {
        for j in 0..i {
            info[i * k + j] = info[j * k + i];
        }
    }
    info
}
