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
//! no coefficient by more than the tolerance below. This is the classic
//! *glmnet* two-loop scheme; it reaches the same global optimum as naive full
//! cycling (the objective is convex and separable in the penalty) while
//! spending most iterations on the handful of active features.
//!
//! # Convergence
//!
//! Convergence is declared on the maximum coefficient change in a full
//! sweep, measured **relative to the scale of the problem**: a full sweep
//! stops the solver when
//!
//! ```text
//! max_j |b_j^new - b_j^old| * ||x_j||  <=  tol * ||y|| .
//! ```
//!
//! `|Δb_j| * ||x_j||` is the Euclidean size of the change feature `j`
//! makes to the fitted values, so both sides carry the units of `y` and
//! `tol` is **dimensionless**. That matters because the objective has an
//! exact equivariance: sending `X -> s*X` and `alpha -> s*alpha` leaves
//! `(1/(2n))||y - Xb||^2 + alpha*l1_ratio*||b||_1 + ...` bit-for-bit
//! identical with the solution at `b/s`, and sending `y -> c*y`,
//! `alpha -> c*alpha` scales the solution by `c`. A tolerance compared
//! against a *bare* coefficient change is blind to both: coefficients of a
//! large-scale design move by `O(1/s)` per sweep, so an absolute `tol`
//! is met after a single sweep and the solver returns a silently wrong,
//! badly under-converged answer (it stops one soft-threshold step away
//! from the zero warm start). Normalizing by `||x_j||` and `||y||` makes
//! the stopping rule — and therefore the returned coefficients — exactly
//! invariant under both reparametrizations. `properties.rs`'s
//! `coordinate_descent_is_scale_equivariant`,
//! `coordinate_descent_is_equivariant_in_the_target_scale` and
//! `regularization_path_is_scale_equivariant` sweep `s` and `c` over
//! twenty-odd decades each; with an absolute rule they miss the scale-1
//! answer by as much as 9% while reporting success.
//!
//! Normalizing *per column* rather than by one global constant also keeps
//! the rule sane on designs whose columns carry wildly different scales:
//! each coordinate is judged by how much it moved the fit, not by the
//! magnitude of a coefficient whose units are the column's own.
//!
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
    /// Convergence tolerance on the maximum coefficient change in a full
    /// sweep, expressed **relative** to the problem's scale: the solver
    /// stops when `max_j |Δb_j| * ||x_j|| <= tol * ||y||`. Being
    /// dimensionless, `tol` means the same thing however `X` and `y` are
    /// scaled (see the [module docs](self#convergence)). Smaller tightens
    /// the match to scikit-learn's optimum; values below `~1e-15` are
    /// below what double precision can resolve and will exhaust
    /// `max_iter`.
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
    /// Largest absolute coefficient change in the final full sweep, in the
    /// coefficients' own units. Diagnostic only — the stopping rule
    /// compares the *scale-free* [`Self::max_rel_change`] against `tol`.
    pub max_change: f64,
    /// Largest coefficient change in the final full sweep measured
    /// relative to the problem's scale, `max_j |Δb_j| * ||x_j|| / ||y||`.
    /// This is the quantity tested against
    /// [`CoordDescentOptions::tol`], so `max_rel_change <= tol` on a
    /// successful return regardless of how `X` and `y` are scaled.
    pub max_rel_change: f64,
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

/// The size of one sweep's largest coefficient move, reported two ways.
#[derive(Debug, Clone, Copy)]
struct SweepChange {
    /// `max_j |Δb_j|`, in the coefficients' own units.
    abs: f64,
    /// `max_j |Δb_j| * ||x_j||` — the same moves measured as the change
    /// each feature makes to the fitted values, i.e. in the units of `y`.
    /// The stopping rule divides this by `||y||`, which is what makes the
    /// tolerance scale-free.
    fit: f64,
}

/// One coordinate sweep over `order`. Updates `beta` and the residual `r`
/// in place and returns the largest coefficient change seen, both in raw
/// coefficient units and in fitted-value units.
///
/// `cols[j]` is column `j` of `X`, `norm2[j] = ||x_j||^2`, `norm[j] =
/// ||x_j||`, `l1_pen = n*alpha*l1_ratio`, `l2_pen = n*alpha*(1-l1_ratio)`.
#[allow(clippy::too_many_arguments)]
fn sweep(
    order: &[usize],
    beta: &mut [f64],
    r: &mut [f64],
    cols: &[Vec<f64>],
    norm2: &[f64],
    norm: &[f64],
    l1_pen: f64,
    l2_pen: f64,
) -> SweepChange {
    let mut max_change = 0.0f64;
    let mut max_fit_change = 0.0f64;
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
        // ||x_j (b_j^new - b_j^old)||_2 = |Δb_j| * ||x_j||: how far this
        // coordinate moved the fit, in the units of y.
        let fit_change = change * norm[j];
        if fit_change > max_fit_change {
            max_fit_change = fit_change;
        }
        beta[j] = new_bj;
    }
    SweepChange {
        abs: max_change,
        fit: max_fit_change,
    }
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
    let norm: Vec<f64> = norm2.iter().map(|v| v.sqrt()).collect();
    let l1_pen = (n as f64) * alpha * l1_ratio;
    let l2_pen = (n as f64) * alpha * (1.0 - l1_ratio);

    // Convergence budget in the units of y. A sweep converges when no
    // coordinate moved the fit by more than `tol` times the size of the
    // target; both sides scale identically under X -> s*X (with
    // alpha -> s*alpha) and under y -> c*y (with alpha -> c*alpha), so the
    // stopping point — and hence the answer — is scale-invariant. A
    // degenerate y = 0 leaves the budget at zero, which is right: the
    // solution is exactly zero and the sweep change is exactly zero, so
    // the `<=` test below still terminates on the first sweep.
    let y_norm = dot(y, y).sqrt();
    let budget = opts.tol * y_norm;

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
        last_full_change = sweep(&all, &mut beta, &mut r, cols, &norm2, &norm, l1_pen, l2_pen);
        if last_full_change.fit <= budget {
            break;
        }
        if n_iter >= opts.max_iter {
            return Err(MlError::NoConvergence {
                iterations: n_iter,
                max_change: last_full_change.abs,
            });
        }

        // Polish the active set until it stabilizes.
        let active: Vec<usize> = (0..p).filter(|&j| beta[j] != 0.0).collect();
        if !active.is_empty() {
            loop {
                n_iter += 1;
                let ch = sweep(
                    &active, &mut beta, &mut r, cols, &norm2, &norm, l1_pen, l2_pen,
                );
                if ch.fit <= budget {
                    break;
                }
                if n_iter >= opts.max_iter {
                    return Err(MlError::NoConvergence {
                        iterations: n_iter,
                        max_change: ch.abs,
                    });
                }
            }
        }
    }

    Ok(PenalizedFit {
        coef: beta,
        n_iter,
        max_change: last_full_change.abs,
        // y_norm == 0 only when y is identically zero, in which case the
        // sweep change is exactly zero too; report 0 rather than 0/0.
        max_rel_change: if y_norm > 0.0 {
            last_full_change.fit / y_norm
        } else {
            0.0
        },
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
        max_rel_change: fit.max_rel_change,
    })
}
