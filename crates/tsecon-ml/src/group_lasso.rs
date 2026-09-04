//! Group LASSO (Yuan & Lin 2006) and sparse-group LASSO (Simon, Friedman,
//! Hastie & Tibshirani 2013) by block coordinate descent.
//!
//! # Objective
//!
//! For the `n x p` design `X`, target `y`, a partition of the columns into
//! groups `g` with per-group weights `w_g > 0`, penalty `alpha >= 0` and
//! mixing parameter `l1_ratio in [0, 1]`, the solver minimizes
//!
//! ```text
//! (1 / (2n)) ||y - X b||_2^2
//!   + alpha * [ (1 - l1_ratio) * sum_g w_g ||b_g||_2  +  l1_ratio * ||b||_1 ] .
//! ```
//!
//! `l1_ratio = 0` is the group LASSO of Yuan & Lin (2006); `0 < l1_ratio <
//! 1` is the sparse-group LASSO of Simon et al. (2013), which selects
//! groups *and* thins them from the inside; `l1_ratio = 1` drops the group
//! term and is exactly the crate's [`crate::lasso`] (same `1/(2n)` data-fit
//! scaling, same `alpha`), which transitively pins this module to
//! scikit-learn's `Lasso`. The default weights `w_g = sqrt(|g|)` are the
//! Yuan-Lin convention that puts groups of different sizes on the same
//! footing. No intercept is fitted: pass a centered `y` and
//! centered/standardized columns.
//!
//! # Block update (the prox math the roadmap warns about)
//!
//! The objective is separable across groups given the others, so the
//! solver cycles over groups. For group `g` with columns `X_g`, let
//! `R_{(-g)} = y - X b + X_g b_g` be the residual with the group removed.
//! The block subproblem
//!
//! ```text
//! min_{b_g} (1/(2n)) ||R_{(-g)} - X_g b_g||^2 + lam2 w_g ||b_g||_2 + lam1 ||b_g||_1 ,
//! lam1 = alpha * l1_ratio,  lam2 = alpha * (1 - l1_ratio),
//! ```
//!
//! has a closed-form answer only when `X_g' X_g` is a multiple of the
//! identity. Yuan & Lin's original algorithm assumed *groupwise
//! orthonormal* designs for that reason; applying its closed form to a raw
//! `X_g` converges smoothly to the minimizer of a **different** objective
//! (the penalty ends up on the orthonormalized coordinates). This module
//! keeps the penalty on the coefficients as stated and instead solves each
//! block by proximal gradient (majorization-minimization) with the exact
//! per-block Lipschitz constant
//!
//! ```text
//! L_g = lambda_max(X_g' X_g) / n
//! ```
//!
//! (largest eigenvalue of the block Gram matrix, from a symmetric
//! eigendecomposition). Each inner step is `u = b_g + (1/L_g) X_g' R / n`
//! followed by the two-level prox of Simon et al. (2013, eq. 6):
//! soft-threshold every coordinate at `lam1 / L_g`, then shrink the whole
//! block by `(1 - lam2 w_g / (L_g ||S||))_+`. Before iterating, the exact
//! group-zero test of Simon et al. (2013, eq. 5) is applied:
//! `b_g = 0` is the block minimizer iff `||S(X_g' R_{(-g)} / n, lam1)||_2 <=
//! lam2 w_g`. A singleton group needs a single prox step (the block Gram is
//! the scalar `||x_j||^2`), and that step is algebraically identical to the
//! coordinate-descent soft-threshold update of [`crate::coordinate_descent`]
//! with `alpha` replaced by `alpha * (l1_ratio + (1 - l1_ratio) w_g)`.
//!
//! # Stopping rule and the KKT certificate
//!
//! Because a wrong Lipschitz constant or a wrong prox order converges
//! *smoothly* to a wrong answer, convergence is not declared on coefficient
//! movement alone. A full sweep that moves no coordinate by more than `tol
//! * ||y|| / ||x_j||` (the same dimensionless rule as the coordinate-descent
//! engine) triggers an evaluation of the subgradient Karush-Kuhn-Tucker
//! conditions at the current point, with the gradient `grad = -X'(y - Xb)/n`:
//!
//! * inactive group (`b_g = 0`): residual `max(0, ||S(-grad_g, lam1)||_2 -
//!   lam2 w_g)`;
//! * active group, nonzero coordinate: `|grad_j + lam2 w_g b_j/||b_g||_2 +
//!   lam1 sign(b_j)|`;
//! * active group, zero coordinate: `max(0, |grad_j| - lam1)`.
//!
//! The largest of these is [`GroupLassoFit::kkt_violation`]; the solver
//! returns `converged = true` only once it is at or below `tol *
//! max_j |x_j' y| / n` (the gradient at `b = 0`, so the bound is scale-free
//! like `tol` itself). The problem is convex, so a small KKT residual
//! *certifies* proximity to the global optimum regardless of how the
//! iterate was produced — the integration test `structured_golden.rs`
//! re-evaluates the same conditions independently.
//!
//! When the sweep budget runs out the last iterate is returned with
//! `converged = false` and its measured `kkt_violation`, so the caller can
//! judge it rather than receive nothing.

use tsecon_linalg::faer::{Mat, MatRef, Side};

use crate::coordinate_descent::CoordDescentOptions;
use crate::error::MlError;
use crate::util::{check_xy, columns, dot};

/// Per-group penalty weights `w_g` in the group term
/// `alpha * (1 - l1_ratio) * sum_g w_g ||b_g||_2`.
#[derive(Debug, Clone, PartialEq)]
pub enum GroupWeights {
    /// `w_g = sqrt(|g|)`, the Yuan & Lin (2006) convention (the default).
    SqrtSize,
    /// `w_g = 1` for every group.
    Uniform,
    /// One positive finite weight per distinct group label, in ascending
    /// label order.
    Custom(Vec<f64>),
}

/// Result of a (sparse-)group-LASSO fit.
#[derive(Debug, Clone, PartialEq)]
pub struct GroupLassoFit {
    /// Estimated coefficient vector, length `p`.
    pub coef: Vec<f64>,
    /// Number of block-coordinate sweeps performed (full and active-set).
    pub n_iter: usize,
    /// `true` iff the sweep-change rule *and* the KKT certificate were met
    /// within `max_iter`. When `false` the returned `coef` is the last
    /// iterate; read `kkt_violation` before using it.
    pub converged: bool,
    /// Labels (as passed in `groups`) of the groups with a nonzero block,
    /// ascending.
    pub active_groups: Vec<i64>,
    /// Column indices with a nonzero coefficient, ascending.
    pub active_set: Vec<usize>,
    /// Objective value at `coef`.
    pub objective: f64,
    /// Largest subgradient KKT residual at `coef` (see the module docs).
    /// On a converged return this is at most `tol * max_j |x_j' y| / n`.
    pub kkt_violation: f64,
    /// Largest coefficient change in the final full sweep, relative to the
    /// problem's scale: `max_j |Δb_j| * ||x_j|| / ||y||`.
    pub max_rel_change: f64,
    /// Smallest `alpha` at which the all-zero vector is the solution for
    /// this design, `l1_ratio` and weights — see
    /// [`group_lasso_alpha_max`].
    pub alpha_max: f64,
}

/// Column-index members of each distinct group label, labels ascending.
struct GroupStructure {
    labels: Vec<i64>,
    members: Vec<Vec<usize>>,
}

fn group_structure(groups: &[i64], p: usize) -> Result<GroupStructure, MlError> {
    if groups.len() != p {
        return Err(MlError::DimensionMismatch {
            what: "groups must carry one integer label per column of x",
            expected: p,
            got: groups.len(),
        });
    }
    let mut labels: Vec<i64> = groups.to_vec();
    labels.sort_unstable();
    labels.dedup();
    let members: Vec<Vec<usize>> = labels
        .iter()
        .map(|&lab| (0..p).filter(|&j| groups[j] == lab).collect())
        .collect();
    Ok(GroupStructure { labels, members })
}

fn resolve_weights(weights: &GroupWeights, st: &GroupStructure) -> Result<Vec<f64>, MlError> {
    let n_groups = st.labels.len();
    match weights {
        GroupWeights::SqrtSize => Ok(st.members.iter().map(|m| (m.len() as f64).sqrt()).collect()),
        GroupWeights::Uniform => Ok(vec![1.0; n_groups]),
        GroupWeights::Custom(w) => {
            if w.len() != n_groups {
                return Err(MlError::DimensionMismatch {
                    what: "group_weights must have one entry per distinct group label \
                           (ascending label order)",
                    expected: n_groups,
                    got: w.len(),
                });
            }
            if w.iter().any(|v| !v.is_finite() || *v <= 0.0) {
                return Err(MlError::InvalidArgument {
                    what: "group_weights must be finite and strictly positive",
                });
            }
            Ok(w.clone())
        }
    }
}

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

/// Soft-threshold operator `S(z, t) = sign(z) * max(|z| - t, 0)`.
#[inline]
fn soft(z: f64, t: f64) -> f64 {
    if z > t {
        z - t
    } else if z < -t {
        z + t
    } else {
        0.0
    }
}

/// Per-block Lipschitz constants `L_g = lambda_max(X_g' X_g) / n`.
fn block_lipschitz(
    cols: &[Vec<f64>],
    members: &[Vec<usize>],
    n: usize,
) -> Result<Vec<f64>, MlError> {
    let nf = n as f64;
    let mut out = Vec::with_capacity(members.len());
    for m in members {
        let k = m.len();
        if k == 1 {
            out.push(dot(&cols[m[0]], &cols[m[0]]) / nf);
            continue;
        }
        let gram = Mat::from_fn(k, k, |a, b| dot(&cols[m[a]], &cols[m[b]]));
        let eig =
            gram.self_adjoint_eigen(Side::Lower)
                .map_err(|_| MlError::DecompositionFailed {
                    what: "group-LASSO block Lipschitz eigenproblem",
                })?;
        // faer returns the eigenvalues in nondecreasing order.
        let lam_max = eig
            .S()
            .column_vector()
            .iter()
            .copied()
            .fold(0.0f64, f64::max);
        out.push(lam_max / nf);
    }
    Ok(out)
}

/// Fixed problem data shared by the sweep and the KKT evaluation.
struct Problem<'a> {
    cols: &'a [Vec<f64>],
    members: &'a [Vec<usize>],
    weights: &'a [f64],
    lips: &'a [f64],
    norm: &'a [f64],
    n: usize,
    lam1: f64,
    lam2: f64,
}

/// Largest subgradient KKT residual at `beta` given the residual `r = y -
/// X beta` (module docs, "Stopping rule and the KKT certificate").
fn kkt_violation(pb: &Problem<'_>, beta: &[f64], r: &[f64]) -> f64 {
    let nf = pb.n as f64;
    let mut worst = 0.0f64;
    for (g, m) in pb.members.iter().enumerate() {
        let w = pb.weights[g];
        // Negative gradient block: -grad_j = x_j' r / n.
        let neg_grad: Vec<f64> = m.iter().map(|&j| dot(&pb.cols[j], r) / nf).collect();
        let nb = m.iter().map(|&j| beta[j] * beta[j]).sum::<f64>().sqrt();
        if nb == 0.0 {
            let s2: f64 = neg_grad.iter().map(|&z| soft(z, pb.lam1).powi(2)).sum();
            worst = worst.max(s2.sqrt() - pb.lam2 * w);
        } else {
            for (k, &j) in m.iter().enumerate() {
                let bj = beta[j];
                let v = if bj != 0.0 {
                    (-neg_grad[k] + pb.lam2 * w * bj / nb + pb.lam1 * bj.signum()).abs()
                } else {
                    neg_grad[k].abs() - pb.lam1
                };
                worst = worst.max(v);
            }
        }
    }
    worst.max(0.0)
}

/// Inner proximal-gradient steps per block visit. The block subproblem
/// converges linearly at rate `1 - mu_g / L_g`; the outer loop revisits any
/// block that has not settled, so this only bounds the work per visit.
const INNER_MAX: usize = 200;

/// One block-coordinate sweep over the groups in `order`. Updates `beta` and
/// `r` in place; returns the largest coefficient move in fitted-value units
/// (`max_j |Δb_j| * ||x_j||`).
fn sweep(
    pb: &Problem<'_>,
    order: &[usize],
    beta: &mut [f64],
    r: &mut [f64],
    inner_budget: f64,
) -> f64 {
    let nf = pb.n as f64;
    let mut max_fit_change = 0.0f64;
    for &g in order {
        let m = &pb.members[g];
        let l = pb.lips[g];
        if l == 0.0 {
            // Every column of the group is identically zero: it contributes
            // nothing to the fit and stays at zero (0/0 guard).
            continue;
        }
        let w = pb.weights[g];
        let k = m.len();
        let old: Vec<f64> = m.iter().map(|&j| beta[j]).collect();

        // Partial residual with the group removed: R_{(-g)} = R + X_g b_g.
        for (idx, &j) in m.iter().enumerate() {
            let bj = old[idx];
            if bj != 0.0 {
                for (ri, &xij) in r.iter_mut().zip(&pb.cols[j]) {
                    *ri += xij * bj;
                }
            }
        }
        // Group-zero test (Simon et al. 2013, eq. 5): z = X_g' R_{(-g)} / n.
        let z: Vec<f64> = m.iter().map(|&j| dot(&pb.cols[j], r) / nf).collect();
        let s_norm = z
            .iter()
            .map(|&v| soft(v, pb.lam1).powi(2))
            .sum::<f64>()
            .sqrt();
        let mut new = vec![0.0f64; k];
        if s_norm > pb.lam2 * w {
            // Proximal-gradient (MM) iterations with step 1 / L_g, starting
            // from the previous block value. Keep `r` as the full residual
            // R = R_{(-g)} - X_g b_g throughout.
            new.copy_from_slice(&old);
            for (idx, &j) in m.iter().enumerate() {
                let bj = new[idx];
                if bj != 0.0 {
                    for (ri, &xij) in r.iter_mut().zip(&pb.cols[j]) {
                        *ri -= xij * bj;
                    }
                }
            }
            let t = 1.0 / l;
            let t_lam1 = t * pb.lam1;
            let t_lam2_w = t * pb.lam2 * w;
            for _ in 0..INNER_MAX {
                // u = b_g + t * X_g' R / n, then the two-level prox.
                let mut s = vec![0.0f64; k];
                let mut s_norm2 = 0.0f64;
                for (idx, &j) in m.iter().enumerate() {
                    let u = new[idx] + t * dot(&pb.cols[j], r) / nf;
                    let v = soft(u, t_lam1);
                    s[idx] = v;
                    s_norm2 += v * v;
                }
                let s_norm = s_norm2.sqrt();
                let shrink = if s_norm > t_lam2_w {
                    1.0 - t_lam2_w / s_norm
                } else {
                    0.0
                };
                let mut inner_change = 0.0f64;
                for (idx, &j) in m.iter().enumerate() {
                    let nb = shrink * s[idx];
                    let d = nb - new[idx];
                    if d != 0.0 {
                        for (ri, &xij) in r.iter_mut().zip(&pb.cols[j]) {
                            *ri -= xij * d;
                        }
                        inner_change = inner_change.max(d.abs() * pb.norm[j]);
                    }
                    new[idx] = nb;
                }
                if inner_change <= inner_budget {
                    break;
                }
            }
        }
        // `r` already holds R_{(-g)} - X_g b_g^new in both branches (the
        // zero branch left it at R_{(-g)}).
        for (idx, &j) in m.iter().enumerate() {
            let change = (new[idx] - old[idx]).abs() * pb.norm[j];
            max_fit_change = max_fit_change.max(change);
            beta[j] = new[idx];
        }
    }
    max_fit_change
}

/// Objective value at `beta` with residual `r`.
fn objective(pb: &Problem<'_>, beta: &[f64], r: &[f64]) -> f64 {
    let nf = pb.n as f64;
    let fit = dot(r, r) / (2.0 * nf);
    let group: f64 = pb
        .members
        .iter()
        .zip(pb.weights)
        .map(|(m, &w)| w * m.iter().map(|&j| beta[j] * beta[j]).sum::<f64>().sqrt())
        .sum();
    let l1: f64 = beta.iter().map(|b| b.abs()).sum();
    fit + pb.lam2 * group + pb.lam1 * l1
}

/// Smallest `alpha` for which `b = 0` minimizes the group / sparse-group
/// LASSO objective (the top of a regularization path).
///
/// With `z_g = X_g' y / n`, zero is optimal iff every group satisfies
/// `||S(z_g, alpha l1_ratio)||_2 <= alpha (1 - l1_ratio) w_g`. For
/// `l1_ratio = 0` that is `alpha >= max_g ||z_g||_2 / w_g`; for `l1_ratio =
/// 1` it is `alpha >= max_j |z_j|` (the coordinate-descent `lambda_max`); in
/// between the left side is decreasing and the right side increasing in
/// `alpha`, so each group's threshold is the unique root of their
/// difference, located by bisection, and `alpha_max` is the largest.
///
/// # Errors
///
/// As [`group_lasso`] minus the `alpha` / `tol` / `max_iter` checks.
pub fn group_lasso_alpha_max(
    x: MatRef<'_, f64>,
    y: &[f64],
    groups: &[i64],
    l1_ratio: f64,
    weights: &GroupWeights,
) -> Result<f64, MlError> {
    let (n, p) = check_xy(x, y)?;
    if !l1_ratio.is_finite() || !(0.0..=1.0).contains(&l1_ratio) {
        return Err(MlError::InvalidArgument {
            what: "l1_ratio must lie in [0, 1]",
        });
    }
    let st = group_structure(groups, p)?;
    let w = resolve_weights(weights, &st)?;
    let cols = columns(x);
    Ok(alpha_max_inner(&cols, y, &st.members, &w, n, l1_ratio))
}

fn alpha_max_inner(
    cols: &[Vec<f64>],
    y: &[f64],
    members: &[Vec<usize>],
    weights: &[f64],
    n: usize,
    l1_ratio: f64,
) -> f64 {
    let nf = n as f64;
    let mut worst = 0.0f64;
    for (g, m) in members.iter().enumerate() {
        let z: Vec<f64> = m.iter().map(|&j| dot(&cols[j], y) / nf).collect();
        let z_inf = z.iter().fold(0.0f64, |a, &v| a.max(v.abs()));
        let z_norm = z.iter().map(|v| v * v).sum::<f64>().sqrt();
        let w = weights[g];
        let a_g = if z_norm == 0.0 {
            0.0
        } else if l1_ratio >= 1.0 {
            z_inf
        } else if l1_ratio <= 0.0 {
            z_norm / w
        } else {
            // phi(a) = ||S(z, a l1)|| - a (1 - l1) w: phi(0) > 0, and at
            // a = z_inf / l1 the soft-threshold kills every coordinate so
            // phi < 0. Bisect the sign change.
            let phi = |a: f64| {
                let s: f64 = z.iter().map(|&v| soft(v, a * l1_ratio).powi(2)).sum();
                s.sqrt() - a * (1.0 - l1_ratio) * w
            };
            let mut lo = 0.0f64;
            let mut hi = z_inf / l1_ratio;
            for _ in 0..200 {
                let mid = 0.5 * (lo + hi);
                if mid <= lo || mid >= hi {
                    break;
                }
                if phi(mid) > 0.0 {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            hi
        };
        worst = worst.max(a_g);
    }
    worst
}

/// Fits the group LASSO / sparse-group LASSO
///
/// ```text
/// min_b (1/(2n))||y - Xb||^2
///       + alpha * [ (1 - l1_ratio) * sum_g w_g ||b_g||_2 + l1_ratio * ||b||_1 ]
/// ```
///
/// by block coordinate descent with exact per-block Lipschitz constants
/// and the two-level prox (see the [module docs](self)).
///
/// `x` is the `n x p` design (no intercept column; center/standardize
/// first), `y` the centered length-`n` target, `groups` one integer label
/// per column (any integers, contiguous or not — columns sharing a label
/// form a group), `alpha >= 0` the penalty, `l1_ratio in [0, 1]` the
/// within-group `L1` share (`0` = group LASSO, `1` = LASSO), `weights` the
/// per-group weights, and `opts` the stopping controls (`tol` is the same
/// dimensionless coefficient-change rule as [`crate::elastic_net`] and also
/// bounds the returned KKT residual relative to `max_j |x_j' y| / n`).
///
/// # Errors
///
/// * [`MlError::EmptyInput`] / [`MlError::DimensionMismatch`] /
///   [`MlError::NonFinite`] on malformed `x`, `y`, `groups` or custom
///   weights;
/// * [`MlError::InvalidArgument`] if `alpha < 0`, `l1_ratio` is outside
///   `[0, 1]`, a custom weight is not finite and positive, `tol <= 0`, or
///   `max_iter == 0`;
/// * [`MlError::DecompositionFailed`] if a block eigenproblem fails.
///
/// Running out of sweeps is **not** an error: the last iterate is returned
/// with `converged = false` and its `kkt_violation`.
pub fn group_lasso(
    x: MatRef<'_, f64>,
    y: &[f64],
    groups: &[i64],
    alpha: f64,
    l1_ratio: f64,
    weights: &GroupWeights,
    opts: CoordDescentOptions,
) -> Result<GroupLassoFit, MlError> {
    let (n, p) = check_xy(x, y)?;
    check_penalty(alpha, l1_ratio, opts)?;
    let st = group_structure(groups, p)?;
    let w = resolve_weights(weights, &st)?;
    let cols = columns(x);
    let norm: Vec<f64> = cols.iter().map(|c| dot(c, c).sqrt()).collect();
    let lips = block_lipschitz(&cols, &st.members, n)?;
    let nf = n as f64;
    let pb = Problem {
        cols: &cols,
        members: &st.members,
        weights: &w,
        lips: &lips,
        norm: &norm,
        n,
        lam1: alpha * l1_ratio,
        lam2: alpha * (1.0 - l1_ratio),
    };

    // Scale-free budgets: coefficient moves in the units of y, KKT residuals
    // in the units of the gradient at zero.
    let y_norm = dot(y, y).sqrt();
    let budget = opts.tol * y_norm;
    let inner_budget = 0.01 * budget;
    let grad0 = cols
        .iter()
        .map(|c| (dot(c, y) / nf).abs())
        .fold(0.0f64, f64::max);
    let kkt_budget = opts.tol * grad0;

    let mut beta = vec![0.0f64; p];
    let mut r = y.to_vec();
    let all: Vec<usize> = (0..st.members.len()).collect();
    let mut n_iter = 0usize;
    let mut converged = false;
    let mut last_full;
    'outer: loop {
        n_iter += 1;
        last_full = sweep(&pb, &all, &mut beta, &mut r, inner_budget);
        if last_full <= budget && kkt_violation(&pb, &beta, &r) <= kkt_budget {
            converged = true;
            break;
        }
        if n_iter >= opts.max_iter {
            break;
        }
        // Polish the active groups until they settle, then re-test all.
        let active: Vec<usize> = (0..st.members.len())
            .filter(|&g| st.members[g].iter().any(|&j| beta[j] != 0.0))
            .collect();
        if !active.is_empty() {
            loop {
                n_iter += 1;
                let ch = sweep(&pb, &active, &mut beta, &mut r, inner_budget);
                if ch <= budget {
                    break;
                }
                if n_iter >= opts.max_iter {
                    break 'outer;
                }
            }
        }
    }

    let kkt = kkt_violation(&pb, &beta, &r);
    let obj = objective(&pb, &beta, &r);
    let active_set: Vec<usize> = (0..p).filter(|&j| beta[j] != 0.0).collect();
    let active_groups: Vec<i64> = st
        .members
        .iter()
        .zip(&st.labels)
        .filter(|(m, _)| m.iter().any(|&j| beta[j] != 0.0))
        .map(|(_, &lab)| lab)
        .collect();
    let alpha_max = alpha_max_inner(&cols, y, &st.members, &w, n, l1_ratio);
    Ok(GroupLassoFit {
        coef: beta,
        n_iter,
        converged,
        active_groups,
        active_set,
        objective: obj,
        kkt_violation: kkt,
        max_rel_change: if y_norm > 0.0 {
            last_full / y_norm
        } else {
            0.0
        },
        alpha_max,
    })
}
