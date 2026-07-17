//! Cyclical coordinate descent for the LASSO, elastic net, and adaptive
//! LASSO, matching scikit-learn's `ElasticNet` / `Lasso` objective exactly.
//!
//! # Objective (scikit-learn convention)
//!
//! For the `n x p` design `X`, target `y`, penalty `alpha >= 0`, and mixing
//! parameter `l1_ratio in [0, 1]`, the solver minimizes
//!
//! ```text
//! (1 / (2n)) ||y - X b||_2^2
//!   + alpha * l1_ratio * ||b||_1
//!   + 0.5 * alpha * (1 - l1_ratio) * ||b||_2^2 .
//! ```
//!
//! `l1_ratio = 1` is the LASSO; `l1_ratio = 0` is a coordinate-descent
//! ridge (the [`crate::ridge`] closed form is preferred there — note the
//! **different** `alpha` scale: elastic net carries the `1/(2n)` data-fit
//! factor, ridge does not; see `fixtures/ml.json`'s `_meta.objective_note`).
//! No intercept is fitted — pass a centered `y` and centered/standardized
//! columns.
//!
//! # Coordinate update (Friedman, Hastie & Tibshirani 2010)
//!
//! Cycling one coordinate `j` at a time with the others fixed, let
//! `R = y - X b` be the current residual and `R_{(-j)} = R + x_j b_j` the
//! residual with feature `j` removed. The scalar sub-problem
//!
//! ```text
//! min_{b_j}  (1/(2n)) ||R_{(-j)} - x_j b_j||^2
//!            + alpha * l1_ratio * |b_j|
//!            + 0.5 * alpha * (1 - l1_ratio) * b_j^2
//! ```
//!
//! has the closed-form soft-thresholding solution (Friedman–Hastie–
//! Tibshirani 2010, eq. 5; *glmnet*)
//!
//! ```text
//! b_j <- S( x_j' R_{(-j)} , n * alpha * l1_ratio )
//!        / ( ||x_j||^2 + n * alpha * (1 - l1_ratio) ),
//! ```
//!
//! where `S(z, t) = sign(z) * max(|z| - t, 0)` is the soft-threshold
//! operator. Multiplying numerator and denominator through by `n` puts the
//! update in scikit-learn's un-normalized `cd_fast` form
//! (`alpha_cd = alpha * l1_ratio * n`, `beta_cd = alpha * (1 - l1_ratio) *
//! n`, `norm_cols_X[j] = ||x_j||^2`), so the fixed point is identical to
//! scikit-learn's to floating-point precision.
//!
//! # Active-set strategy
//!
//! After each full sweep over all `p` coordinates the solver polishes the
//! *active set* (currently nonzero coordinates) with cheaper sweeps until
//! that set converges, then takes another full sweep to test whether any
//! zeroed coordinate should re-enter. It stops when a **full** sweep moves
//! no coefficient by more than `tol`. This is the classic *glmnet*
//! two-loop scheme; it reaches the same global optimum as naive full
//! cycling (the objective is convex and separable in the penalty) while
//! spending most iterations on the handful of active features.
//!
//! # Convergence
//!
//! Convergence is declared on the **maximum absolute coefficient change**
//! in a full sweep falling below `tol` (the roadmap's stated criterion).
//! Against the golden fixture — where scikit-learn ran to `tol = 1e-12` —
//! the default `tol = 1e-11` reproduces every coefficient to better than
//! `1e-9` absolute, comfortably inside the `1e-6` fixture tolerance (the
//! golden test asserts the achieved figure).

use tsecon_linalg::faer::MatRef;

use crate::error::MlError;
use crate::ridge::ols_svd;
use crate::util::{check_xy, columns, dot};

/// Stopping controls for the coordinate-descent solvers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CoordDescentOptions {
    /// Convergence tolerance on the maximum absolute coefficient change in
    /// a full sweep. Smaller tightens the match to scikit-learn's optimum.
    pub tol: f64,
    /// Maximum number of coordinate sweeps (full and active-set combined)
    /// before [`MlError::NoConvergence`] is returned.
    pub max_iter: usize,
}

impl Default for CoordDescentOptions {
    fn default() -> Self {
        Self {
            tol: 1e-11,
            max_iter: 100_000,
        }
    }
}

/// Result of a penalized-regression fit.
#[derive(Debug, Clone, PartialEq)]
pub struct PenalizedFit {
    /// Estimated coefficient vector, length `p`.
    pub coef: Vec<f64>,
    /// Number of coordinate sweeps performed.
    pub n_iter: usize,
    /// Largest absolute coefficient change in the final full sweep.
    pub max_change: f64,
}

/// Soft-threshold operator `S(z, t) = sign(z) * max(|z| - t, 0)`.
#[inline]
fn soft_threshold(z: f64, t: f64) -> f64 {
    if z > t {
        z - t
    } else if z < -t {
        z + t
    } else {
        0.0
    }
}

/// Validates the shared penalty configuration.
fn check_penalty(alpha: f64, l1_ratio: f64, opts: CoordDescentOptions) -> Result<(), MlError> {
    if !alpha.is_finite() || alpha < 0.0 {
        return Err(MlError::InvalidArgument {
            what: "alpha must be finite and non-negative",
        });
    }
    if !l1_ratio.is_finite() || !(0.0..=1.0).contains(&l1_ratio) {
        return Err(MlError::InvalidArgument {
            what: "l1_ratio must lie in [0, 1]",
        });
    }
    if !opts.tol.is_finite() || opts.tol <= 0.0 {
        return Err(MlError::InvalidArgument {
            what: "tol must be finite and positive",
        });
    }
    if opts.max_iter == 0 {
        return Err(MlError::InvalidArgument {
            what: "max_iter must be at least 1",
        });
    }
    Ok(())
}

/// One coordinate sweep over `order`. Updates `beta` and the residual `r`
/// in place and returns the largest absolute coefficient change seen.
///
/// `cols[j]` is column `j` of `X`, `norm2[j] = ||x_j||^2`, `l1_pen =
/// n*alpha*l1_ratio`, `l2_pen = n*alpha*(1-l1_ratio)`.
#[allow(clippy::too_many_arguments)]
fn sweep(
    order: &[usize],
    beta: &mut [f64],
    r: &mut [f64],
    cols: &[Vec<f64>],
    norm2: &[f64],
    l1_pen: f64,
    l2_pen: f64,
) -> f64 {
    let mut max_change = 0.0f64;
    for &j in order {
        if norm2[j] == 0.0 {
            // A constant (zero-variance) column contributes nothing and is
            // pinned at zero; skipping avoids a 0/0 update.
            continue;
        }
        let bj = beta[j];
        let xj = &cols[j];
        // Add feature j back into the residual: R_{(-j)} = R + x_j b_j.
        if bj != 0.0 {
            for (ri, &xij) in r.iter_mut().zip(xj) {
                *ri += xij * bj;
            }
        }
        // z = x_j' R_{(-j)}
        let z = dot(xj, r);
        let new_bj = soft_threshold(z, l1_pen) / (norm2[j] + l2_pen);
        // Subtract the updated contribution back out of the residual.
        if new_bj != 0.0 {
            for (ri, &xij) in r.iter_mut().zip(xj) {
                *ri -= xij * new_bj;
            }
        }
        let change = (new_bj - bj).abs();
        if change > max_change {
            max_change = change;
        }
        beta[j] = new_bj;
    }
    max_change
}

/// Core coordinate-descent engine operating on pre-extracted columns and a
/// caller-provided warm start (used by the regularization path for warm
/// starts along the `lambda` grid).
pub(crate) fn cd_engine(
    cols: &[Vec<f64>],
    y: &[f64],
    alpha: f64,
    l1_ratio: f64,
    warm_start: &[f64],
    opts: CoordDescentOptions,
) -> Result<PenalizedFit, MlError> {
    let n = y.len();
    let p = cols.len();
    let norm2: Vec<f64> = cols.iter().map(|c| dot(c, c)).collect();
    let l1_pen = (n as f64) * alpha * l1_ratio;
    let l2_pen = (n as f64) * alpha * (1.0 - l1_ratio);

    let mut beta = warm_start.to_vec();
    // Residual for the warm start: R = y - X beta.
    let mut r = y.to_vec();
    for (j, bj) in beta.iter().enumerate() {
        if *bj != 0.0 {
            for (ri, &xij) in r.iter_mut().zip(&cols[j]) {
                *ri -= xij * bj;
            }
        }
    }

    let all: Vec<usize> = (0..p).collect();
    let mut n_iter = 0usize;
    let mut last_full_change;

    loop {
        // Full sweep over every coordinate.
        n_iter += 1;
        last_full_change = sweep(&all, &mut beta, &mut r, cols, &norm2, l1_pen, l2_pen);
        if last_full_change < opts.tol {
            break;
        }
        if n_iter >= opts.max_iter {
            return Err(MlError::NoConvergence {
                iterations: n_iter,
                max_change: last_full_change,
            });
        }

        // Polish the active set until it stabilizes.
        let active: Vec<usize> = (0..p).filter(|&j| beta[j] != 0.0).collect();
        if !active.is_empty() {
            loop {
                n_iter += 1;
                let ch = sweep(&active, &mut beta, &mut r, cols, &norm2, l1_pen, l2_pen);
                if ch < opts.tol {
                    break;
                }
                if n_iter >= opts.max_iter {
                    return Err(MlError::NoConvergence {
                        iterations: n_iter,
                        max_change: ch,
                    });
                }
            }
        }
    }

    Ok(PenalizedFit {
        coef: beta,
        n_iter,
        max_change: last_full_change,
    })
}

/// Fits the elastic net `min_b (1/(2n))||y - Xb||^2 + alpha*l1_ratio*||b||_1
/// + 0.5*alpha*(1-l1_ratio)*||b||^2` by cyclical coordinate descent.
///
/// `x` is the `n x p` design (no intercept column), `y` the centered
/// length-`n` target. See the [module docs](self) for the objective, the
/// soft-thresholding update, and the active-set strategy.
///
/// # Errors
///
/// * [`MlError::EmptyInput`] / [`MlError::DimensionMismatch`] /
///   [`MlError::NonFinite`] on malformed inputs;
/// * [`MlError::InvalidArgument`] if `alpha < 0`, `l1_ratio` is outside
///   `[0, 1]`, `tol <= 0`, or `max_iter == 0`;
/// * [`MlError::NoConvergence`] if the sweep budget is exhausted.
pub fn elastic_net(
    x: MatRef<'_, f64>,
    y: &[f64],
    alpha: f64,
    l1_ratio: f64,
    opts: CoordDescentOptions,
) -> Result<PenalizedFit, MlError> {
    let (_n, p) = check_xy(x, y)?;
    check_penalty(alpha, l1_ratio, opts)?;
    let cols = columns(x);
    let warm = vec![0.0; p];
    cd_engine(&cols, y, alpha, l1_ratio, &warm, opts)
}

/// Fits the LASSO — elastic net with `l1_ratio = 1`, i.e.
/// `min_b (1/(2n))||y - Xb||^2 + alpha*||b||_1`.
///
/// # Errors
///
/// As [`elastic_net`].
pub fn lasso(
    x: MatRef<'_, f64>,
    y: &[f64],
    alpha: f64,
    opts: CoordDescentOptions,
) -> Result<PenalizedFit, MlError> {
    elastic_net(x, y, alpha, 1.0, opts)
}

/// Fits the adaptive LASSO of Zou (2006): a weighted-`L1` penalty
/// `alpha * l1_ratio * sum_j w_j |b_j|` with data-driven weights
/// `w_j = 1 / |b_j^{ols}|^gamma`.
///
/// The weighted problem is solved by **feature rescaling**: with the
/// substitution `b_j = tilde b_j / w_j`, the penalty becomes an ordinary
/// (unweighted) `L1` penalty on `tilde b` applied to the rescaled design
/// `tilde x_j = x_j / w_j = x_j * |b_j^{ols}|^gamma`. We run the plain
/// elastic-net coordinate descent on `tilde X` and undo the scaling,
/// `b_j = tilde b_j * |b_j^{ols}|^gamma`. A feature whose OLS coefficient
/// is essentially zero gets weight `+inf` (rescaled column `0`), so its
/// coefficient is pinned at exactly `0` — the mechanism by which adaptive
/// weighting drives true zeros out more aggressively than the plain LASSO
/// (Zou 2006, oracle property).
///
/// The OLS pilot estimate is the minimum-norm least-squares fit (thin SVD);
/// `gamma > 0` controls how sharply small pilot coefficients are penalized
/// (`gamma = 1` is the common default). With `l1_ratio < 1` an unweighted
/// ridge term `0.5*alpha*(1-l1_ratio)*||tilde b||^2` is retained on the
/// rescaled coordinates.
///
/// # Errors
///
/// * As [`elastic_net`], plus [`MlError::InvalidArgument`] if `gamma` is
///   not finite and positive;
/// * [`MlError::DecompositionFailed`] if the OLS pilot SVD fails.
pub fn adaptive_lasso(
    x: MatRef<'_, f64>,
    y: &[f64],
    alpha: f64,
    l1_ratio: f64,
    gamma: f64,
    opts: CoordDescentOptions,
) -> Result<PenalizedFit, MlError> {
    let (n, p) = check_xy(x, y)?;
    check_penalty(alpha, l1_ratio, opts)?;
    if !gamma.is_finite() || gamma <= 0.0 {
        return Err(MlError::InvalidArgument {
            what: "gamma must be finite and positive",
        });
    }

    // OLS pilot -> adaptive scale s_j = |b_j^{ols}|^gamma = 1 / w_j.
    let b_ols = ols_svd(x, y)?;
    // A pilot coefficient indistinguishable from zero forces the feature
    // out (scale 0). The threshold is relative to the pilot's magnitude.
    let pilot_max = b_ols.iter().fold(0.0f64, |m, &b| m.max(b.abs()));
    let zero_tol = pilot_max * (n.max(p) as f64) * f64::EPSILON;
    let scale: Vec<f64> = b_ols
        .iter()
        .map(|&b| {
            if b.abs() <= zero_tol {
                0.0
            } else {
                b.abs().powf(gamma)
            }
        })
        .collect();

    // Rescaled columns tilde x_j = x_j * scale_j.
    let cols = columns(x);
    let scaled_cols: Vec<Vec<f64>> = cols
        .iter()
        .zip(&scale)
        .map(|(c, &s)| c.iter().map(|v| v * s).collect())
        .collect();

    let warm = vec![0.0; p];
    let fit = cd_engine(&scaled_cols, y, alpha, l1_ratio, &warm, opts)?;

    // Undo the rescaling: b_j = tilde b_j * scale_j.
    let coef: Vec<f64> = fit.coef.iter().zip(&scale).map(|(tb, &s)| tb * s).collect();
    Ok(PenalizedFit {
        coef,
        n_iter: fit.n_iter,
        max_change: fit.max_change,
    })
}
