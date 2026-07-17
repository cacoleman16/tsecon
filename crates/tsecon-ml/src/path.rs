//! Regularization path for the elastic net, with AIC/BIC model selection.
//!
//! # `lambda` grid
//!
//! The path runs from the smallest penalty that zeros every coefficient,
//!
//! ```text
//! lambda_max = max_j |x_j' y| / (n * l1_ratio),
//! ```
//!
//! (the value at which the first coordinate's soft-threshold argument
//! `|x_j' y|` equals the threshold `n * lambda * l1_ratio`, so all
//! coefficients start at zero — the *glmnet* convention), geometrically
//! down to `lambda_min = eps * lambda_max`. With the default
//! `eps = 1e-3` that spans **three decades**, `n_lambdas` points
//! log-spaced. Coefficients are computed with **warm starts**: each
//! `lambda` is initialized from the previous (larger) `lambda`'s solution,
//! the standard path-continuation speed-up.
//!
//! # Information criteria
//!
//! For each grid point the effective degrees of freedom is the number of
//! nonzero coefficients — an unbiased estimate of the LASSO's df (Zou,
//! Hastie & Tibshirani 2007). With residual sum of squares `RSS` and
//! sample size `n`,
//!
//! ```text
//! AIC = n * ln(RSS / n) + 2 * df,
//! BIC = n * ln(RSS / n) + ln(n) * df.
//! ```
//!
//! [`RegPath::aic_best`] / [`RegPath::bic_best`] return the grid index
//! minimizing each criterion.

use tsecon_linalg::faer::MatRef;

use crate::coordinate_descent::{cd_engine, CoordDescentOptions};
use crate::error::MlError;
use crate::util::{check_xy, columns, dot};

/// Controls the shape of the `lambda` grid and the inner solver.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PathOptions {
    /// Number of `lambda` grid points (>= 1).
    pub n_lambdas: usize,
    /// Ratio `lambda_min / lambda_max`. The default `1e-3` spans three
    /// decades; must lie in `(0, 1]`.
    pub eps: f64,
    /// Coordinate-descent stopping controls used at every grid point.
    pub cd: CoordDescentOptions,
}

impl Default for PathOptions {
    fn default() -> Self {
        Self {
            n_lambdas: 100,
            eps: 1e-3,
            cd: CoordDescentOptions::default(),
        }
    }
}

/// A full elastic-net regularization path plus its information criteria.
#[derive(Debug, Clone, PartialEq)]
pub struct RegPath {
    /// The `l1_ratio` the path was computed at.
    pub l1_ratio: f64,
    /// Penalty grid, descending from `lambda_max` to `eps * lambda_max`.
    pub lambdas: Vec<f64>,
    /// `coefs[i]` is the coefficient vector at `lambdas[i]` (length `p`).
    pub coefs: Vec<Vec<f64>>,
    /// Residual sum of squares `||y - X b||^2` at each grid point.
    pub rss: Vec<f64>,
    /// Degrees of freedom (number of nonzero coefficients) at each point.
    pub df: Vec<usize>,
    /// Akaike information criterion at each grid point.
    pub aic: Vec<f64>,
    /// Bayesian (Schwarz) information criterion at each grid point.
    pub bic: Vec<f64>,
}

impl RegPath {
    /// Grid index minimizing the AIC (first minimizer on ties).
    #[must_use]
    pub fn aic_best(&self) -> usize {
        argmin(&self.aic)
    }

    /// Grid index minimizing the BIC (first minimizer on ties).
    #[must_use]
    pub fn bic_best(&self) -> usize {
        argmin(&self.bic)
    }
}

fn argmin(v: &[f64]) -> usize {
    let mut best = 0usize;
    let mut best_val = f64::INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x < best_val {
            best_val = x;
            best = i;
        }
    }
    best
}

/// The smallest penalty that leaves every coordinate at zero:
/// `lambda_max = max_j |x_j' y| / (n * l1_ratio)`.
///
/// For `l1_ratio = 0` (pure ridge) there is no finite such value; callers
/// must supply `l1_ratio > 0` for a path.
fn lambda_max(cols: &[Vec<f64>], y: &[f64], n: usize, l1_ratio: f64) -> f64 {
    let max_corr = cols.iter().map(|c| dot(c, y).abs()).fold(0.0f64, f64::max);
    max_corr / (n as f64 * l1_ratio)
}

/// Computes the elastic-net regularization path over a log-spaced `lambda`
/// grid and its AIC/BIC scores.
///
/// `x` is the `n x p` design (no intercept), `y` the centered target, and
/// `l1_ratio in (0, 1]` the elastic-net mixing (a path requires a nonzero
/// `L1` share to define `lambda_max`). Grid points descend from
/// `lambda_max` to `eps * lambda_max`; coefficients use warm starts.
///
/// # Errors
///
/// * [`MlError::EmptyInput`] / [`MlError::DimensionMismatch`] /
///   [`MlError::NonFinite`] on malformed inputs;
/// * [`MlError::InvalidArgument`] if `l1_ratio` is not in `(0, 1]`,
///   `n_lambdas == 0`, `eps` is not in `(0, 1]`, or the CD options are
///   invalid;
/// * [`MlError::NoConvergence`] propagated from the inner solver.
pub fn regularization_path(
    x: MatRef<'_, f64>,
    y: &[f64],
    l1_ratio: f64,
    opts: PathOptions,
) -> Result<RegPath, MlError> {
    let (n, p) = check_xy(x, y)?;
    if !l1_ratio.is_finite() || !(0.0..=1.0).contains(&l1_ratio) || l1_ratio == 0.0 {
        return Err(MlError::InvalidArgument {
            what: "l1_ratio must lie in (0, 1] for a regularization path",
        });
    }
    if opts.n_lambdas == 0 {
        return Err(MlError::InvalidArgument {
            what: "n_lambdas must be at least 1",
        });
    }
    if !opts.eps.is_finite() || !(0.0..=1.0).contains(&opts.eps) || opts.eps == 0.0 {
        return Err(MlError::InvalidArgument {
            what: "eps must lie in (0, 1]",
        });
    }

    let cols = columns(x);
    let lam_max = lambda_max(&cols, y, n, l1_ratio);

    // Log-spaced descending grid. With a single point we return lambda_max.
    let m = opts.n_lambdas;
    let log_max = lam_max.ln();
    let log_min = (lam_max * opts.eps).ln();
    let lambdas: Vec<f64> = (0..m)
        .map(|i| {
            if m == 1 {
                lam_max
            } else {
                let t = i as f64 / (m as f64 - 1.0);
                (log_max + t * (log_min - log_max)).exp()
            }
        })
        .collect();

    let mut coefs = Vec::with_capacity(m);
    let mut rss = Vec::with_capacity(m);
    let mut df = Vec::with_capacity(m);
    let mut aic = Vec::with_capacity(m);
    let mut bic = Vec::with_capacity(m);

    let mut warm = vec![0.0f64; p];
    let ln_n = (n as f64).ln();
    for &lam in &lambdas {
        let fit = cd_engine(&cols, y, lam, l1_ratio, &warm, opts.cd)?;
        // Residual and RSS.
        let mut resid = y.to_vec();
        for (j, bj) in fit.coef.iter().enumerate() {
            if *bj != 0.0 {
                for (ri, &xij) in resid.iter_mut().zip(&cols[j]) {
                    *ri -= xij * bj;
                }
            }
        }
        let ss = dot(&resid, &resid);
        let k = fit.coef.iter().filter(|b| **b != 0.0).count();
        // Guard the log against a perfect (zero-RSS) fit.
        let ss_safe = ss.max(f64::MIN_POSITIVE);
        let base = n as f64 * (ss_safe / n as f64).ln();
        aic.push(base + 2.0 * k as f64);
        bic.push(base + ln_n * k as f64);
        rss.push(ss);
        df.push(k);
        warm = fit.coef.clone();
        coefs.push(fit.coef);
    }

    Ok(RegPath {
        l1_ratio,
        lambdas,
        coefs,
        rss,
        df,
        aic,
        bic,
    })
}
