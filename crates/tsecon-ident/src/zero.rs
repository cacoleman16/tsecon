//! Exact zero restrictions on structural impulse responses and the
//! Rubio-Ramirez-Waggoner-Zha (2010) column-recursion that draws rotations
//! satisfying every zero *by construction*.
//!
//! A structural IRF set is `Theta_0, ..., Theta_H`, each `n x n` with
//! `Theta_h[(i, j)]` the response of variable `i` at horizon `h` to
//! structural shock `j`. Under the rotation parameterization
//! `Theta_h = Theta^chol_h Q`, with `Theta^chol_h = Psi_h P` the
//! Cholesky-orthogonalized base ([`tsecon_bayes::cholesky_irf`],
//! `P = chol(Sigma)` lower) and `Q` orthogonal. A short-run zero
//! "`Theta_h[(i, j)] = 0`" is therefore the *linear-in-`Q`* constraint
//!
//! ```text
//! e_i' Theta^chol_h q_j = r . q_j = 0,   r = row i of Theta^chol_h,
//! ```
//!
//! i.e. shock column `q_j` must lie in the null space of the stacked
//! restriction rows `Z_j` for that shock. The constraint rows come directly
//! from the base `cholesky_irf` matrices — no extra reduced-form quantity is
//! needed (Arias, Rubio-Ramirez & Waggoner 2018).
//!
//! # The RWZ column recursion (exact zeros)
//!
//! Order the shocks by *descending* zero count. At step `t` (0-indexed, `t`
//! columns already fixed, `z` restriction rows on this shock) the null space
//! has dimension `n - z - t`; feasibility requires it be at least one, i.e.
//! `z <= n - 1 - t`, validated once at [`ZeroRestrictionSet::new`]. For each
//! shock in order:
//!
//! 1. stack `M = [ Z_j ; q_{prev}' ... ]` — the zero rows plus the transposes
//!    of the already-built columns;
//! 2. build an orthonormal basis `U` of `row-space(M)` by modified
//!    Gram-Schmidt (previous columns are already orthonormal; the `Z_j` rows
//!    are orthonormalized against them and each other, numerically dependent
//!    rows dropped);
//! 3. draw a standard-normal `x` from the [`Stream`] and project out the row
//!    space: `w = x - U (U' x)`, `q_j = w / ||w||`.
//!
//! `q_j` is a uniform draw on the unit sphere of `null(M)`: it satisfies
//! `Z_j q_j = 0` exactly (the zeros are exact — column-sign flips preserve
//! them since `0 = -0`) and is orthonormal to every previously-built column,
//! so the assembled `Q = [q_1 ... q_n]` is orthogonal to machine precision.
//!
//! # The recursive/Cholesky special case
//!
//! Imposing `Theta_0[(i, j)] = 0` for all `i < j` (the strict upper triangle
//! of the impact matrix) makes every null space exactly one-dimensional, so
//! `Q` is determined up to column signs; the Gaussian draw only picks those
//! signs, which the positive-diagonal normalization then fixes. The result
//! is `Q = I` and `Theta_h = Psi_h chol(Sigma)` — the unique Cholesky
//! (recursive) identification — *deterministically*, independent of the seed.
//! This is the crate's strong closed-form golden.

use tsecon_linalg::faer::{Mat, MatRef};
use tsecon_rng::Stream;
use tsecon_stats::special::inv_norm_cdf;

use crate::error::IdentError;

/// Retry budget for rejecting the (probability `2^-53`) exact-zero uniform
/// the inverse normal CDF cannot map to a finite quantile.
const UNIFORM_RETRIES: usize = 128;

/// Relative norm below which a Gram-Schmidt residual is treated as
/// numerically dependent (row dropped from the row-space basis).
const MGS_DROP_TOL: f64 = 1e-10;

/// One standard-normal draw by inverse-CDF transform of a stream uniform
/// (Wichura AS241 `inv_norm_cdf`), rejecting the exact 0 that
/// [`Stream::uniform_f64`] can (with probability `2^-53`) return.
///
/// This duplicates `haar::std_normal` verbatim so the zero-restriction module
/// stays in its own files (no edit to `haar.rs`), keeping parallel builders
/// collision-free; the two share the same Gaussian source semantics.
fn std_normal(stream: &mut Stream) -> Result<f64, IdentError> {
    for _ in 0..UNIFORM_RETRIES {
        let u = stream.uniform_f64();
        if u > 0.0 {
            return Ok(inv_norm_cdf(u)?);
        }
    }
    Err(IdentError::NoConvergence {
        what:
            "positive uniform draw for a Gaussian null-space vector (stream returned 0 repeatedly)",
    })
}

/// A single exact zero restriction: the response of `variable` to `shock` at
/// `horizon` is constrained to be exactly zero (`Theta_horizon[(variable,
/// shock)] = 0`). Indices are zero-based; `horizon = 0` is the impact matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ZeroRestriction {
    /// Zero-based index of the response variable `i`.
    pub variable: usize,
    /// Zero-based index of the structural shock `j`.
    pub shock: usize,
    /// Horizon the zero is imposed at (`0` = impact).
    pub horizon: usize,
}

impl ZeroRestriction {
    /// A zero restriction on the response of `variable` to `shock` at
    /// `horizon`.
    pub fn at(variable: usize, shock: usize, horizon: usize) -> Self {
        Self {
            variable,
            shock,
            horizon,
        }
    }
}

/// A validated collection of exact zero restrictions against a fixed model
/// dimension `n_vars` and impulse-response horizon `horizon` (the maximum
/// horizon index, so `Theta_0..=Theta_horizon` exist).
///
/// Construction deduplicates the restrictions, validates every index,
/// computes the per-shock zero counts, the descending-count shock ordering
/// the RWZ recursion builds columns in, and the RWZ feasibility condition
/// `z_{order[t]} <= n - 1 - t`. An empty restriction set is valid: it makes
/// the sampler degenerate to the pure sign-restricted (or, with no signs
/// either, the recursive) case, and every column is drawn from the full
/// sphere.
#[derive(Debug, Clone)]
pub struct ZeroRestrictionSet {
    restrictions: Vec<ZeroRestriction>,
    n_vars: usize,
    horizon: usize,
    /// Shock indices sorted by descending zero count (ties by ascending
    /// index) — the order the RWZ recursion fixes columns in.
    order: Vec<usize>,
    /// `zeros_per_shock[j]` = number of zero restrictions on shock `j`.
    zeros_per_shock: Vec<usize>,
    /// Whether every zero restriction lies on the impact matrix (`horizon =
    /// 0`), the linear-in-`Q` case in which the ARW importance weight is
    /// exactly one.
    all_impact_only: bool,
}

impl ZeroRestrictionSet {
    /// Builds and validates the set: every variable and shock index must be
    /// below `n_vars`, and every horizon at most `horizon`. Duplicate
    /// restrictions are collapsed before counting.
    ///
    /// # Errors
    ///
    /// * [`IdentError::InvalidArgument`] if `n_vars == 0`, or the zero
    ///   pattern is infeasible / over-restricted (some shock carries more
    ///   than `n - 1 - t` zeros at its recursion step `t`, so its null space
    ///   would be empty);
    /// * [`IdentError::RestrictionOutOfRange`] if any variable, shock, or
    ///   horizon index is out of range.
    pub fn new(
        restrictions: Vec<ZeroRestriction>,
        n_vars: usize,
        horizon: usize,
    ) -> Result<Self, IdentError> {
        if n_vars == 0 {
            return Err(IdentError::InvalidArgument {
                what: "n_vars must be at least 1",
            });
        }

        // Validate, then deduplicate (a repeated (variable, shock, horizon)
        // is one constraint row, not two, and must not inflate the feasibility
        // count).
        let mut unique: Vec<ZeroRestriction> = Vec::with_capacity(restrictions.len());
        for r in restrictions {
            if r.variable >= n_vars {
                return Err(IdentError::RestrictionOutOfRange {
                    what: "response variable",
                    index: r.variable,
                    bound: n_vars,
                });
            }
            if r.shock >= n_vars {
                return Err(IdentError::RestrictionOutOfRange {
                    what: "structural shock",
                    index: r.shock,
                    bound: n_vars,
                });
            }
            if r.horizon > horizon {
                return Err(IdentError::RestrictionOutOfRange {
                    what: "restriction horizon",
                    index: r.horizon,
                    bound: horizon + 1,
                });
            }
            if !unique.contains(&r) {
                unique.push(r);
            }
        }

        let mut zeros_per_shock = vec![0usize; n_vars];
        for r in &unique {
            zeros_per_shock[r.shock] += 1;
        }
        let all_impact_only = unique.iter().all(|r| r.horizon == 0);

        // Descending zero count, ties broken by ascending shock index (stable
        // and reproducible).
        let mut order: Vec<usize> = (0..n_vars).collect();
        order.sort_by(|&a, &b| zeros_per_shock[b].cmp(&zeros_per_shock[a]).then(a.cmp(&b)));

        // RWZ feasibility: at step t the null space has dimension
        // n - z - t >= 1.
        for (t, &shock) in order.iter().enumerate() {
            let z = zeros_per_shock[shock];
            if z + t + 1 > n_vars {
                return Err(IdentError::InvalidArgument {
                    what: "zero pattern is infeasible / over-restricted (a shock's null space is empty)",
                });
            }
        }

        Ok(Self {
            restrictions: unique,
            n_vars,
            horizon,
            order,
            zeros_per_shock,
            all_impact_only,
        })
    }

    /// The validated, deduplicated restrictions.
    pub fn restrictions(&self) -> &[ZeroRestriction] {
        &self.restrictions
    }

    /// Number of variables (and shocks) the set was validated against.
    pub fn n_vars(&self) -> usize {
        self.n_vars
    }

    /// Maximum horizon index the set was validated against.
    pub fn horizon(&self) -> usize {
        self.horizon
    }

    /// The shock ordering (descending zero count) the recursion fixes columns
    /// in. A permutation of `0..n_vars`.
    pub fn order(&self) -> &[usize] {
        &self.order
    }

    /// Per-shock zero counts, indexed by shock.
    pub fn zeros_per_shock(&self) -> &[usize] {
        &self.zeros_per_shock
    }

    /// Whether every zero restriction is on the impact matrix (`horizon =
    /// 0`). In that (linear-in-`Q`) case the ARW-2018 importance weight is
    /// exactly one for every draw.
    pub fn all_impact_only(&self) -> bool {
        self.all_impact_only
    }

    /// The constraint rows for `shock`: row `variable` of `base[horizon]` for
    /// each restriction on that shock. Each returned vector has length
    /// `n_vars`.
    fn rows_for_shock(&self, shock: usize, base: &[Mat<f64>]) -> Vec<Vec<f64>> {
        let n = self.n_vars;
        self.restrictions
            .iter()
            .filter(|r| r.shock == shock)
            .map(|r| {
                let m = &base[r.horizon];
                (0..n).map(|c| m[(r.variable, c)]).collect::<Vec<f64>>()
            })
            .collect()
    }
}

/// ARW-2018 importance log-volume for a drawn rotation.
///
/// Returns `0.0` (unit weight) when every zero restriction lies on the impact
/// matrix `Theta_0 = P Q`: those restriction functions are linear in `Q`, the
/// volume element of the structural-vs-orthogonal-reduced-form
/// parameterization is `Q`-independent, and the weight is *exactly* one
/// (Arias, Rubio-Ramirez & Waggoner 2018) — this is why the recursive/
/// Cholesky golden and every impact-only applied SVAR are unweighted.
///
/// For zeros at horizon `>= 1` (or on a long-run matrix) the restriction
/// functions are nonlinear through the reduced-form coefficients and the ARW
/// volume element is nonconstant; the exact correction requires the ARW
/// Appendix volume-element ratio (differentiating the IRF map with respect to
/// the structural coefficients). That correction is **not yet applied**: this
/// build returns the conditionally-uniform (unit) weight for those patterns
/// too — the honest RWZ (2010) draw — and the prior-robust deliverable is the
/// weight-invariant identified-set envelope (min/max), not the weighted
/// bands. This function is the single, isolated swap point for a future exact
/// ARW weight.
fn arw_log_volume(zeros: &ZeroRestrictionSet) -> f64 {
    // Impact-only is exact at 0.0; the horizon >= 1 branch is intentionally
    // also 0.0 until the ARW volume-element ratio is ported (see docs).
    let _ = zeros.all_impact_only();
    0.0
}

/// Orthonormalizes `v` against the current basis `basis` (modified
/// Gram-Schmidt) and appends it if the residual is not numerically dependent.
/// `basis` is assumed orthonormal on entry and stays orthonormal on exit.
fn orthonormalize_push(basis: &mut Vec<Vec<f64>>, mut v: Vec<f64>) {
    let orig_norm = norm(&v);
    if orig_norm == 0.0 {
        return;
    }
    for u in basis.iter() {
        let d = dot(u, &v);
        for (vi, ui) in v.iter_mut().zip(u.iter()) {
            *vi -= d * ui;
        }
    }
    let resid = norm(&v);
    if resid > MGS_DROP_TOL * orig_norm {
        let inv = 1.0 / resid;
        for vi in v.iter_mut() {
            *vi *= inv;
        }
        basis.push(v);
    }
}

/// Euclidean norm.
fn norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Dot product of equal-length slices.
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Draws an orthogonal rotation `Q` satisfying every zero restriction exactly,
/// via the RWZ (2010) column recursion, returning `(Q, importance_weight)`.
///
/// `base` is the Cholesky-orthogonalized IRF set `Theta^chol_h = Psi_h P`
/// (from [`tsecon_bayes::cholesky_irf`]); its length must exceed every
/// restriction horizon and each matrix must be `n x n` with `n =
/// zeros.n_vars()`. Columns are built in `zeros.order()` and placed at their
/// true shock indices, so `Q` is a genuine `n x n` orthogonal matrix. The
/// returned weight is the ARW-2018 importance weight (exactly `1.0` for
/// impact-only patterns; see [`arw_log_volume`]).
///
/// # Errors
///
/// * [`IdentError::Dimension`] if `base` is empty, its matrices are not
///   `n x n`, or it is too short for a restriction horizon;
/// * [`IdentError::NoConvergence`] if a null space collapses numerically
///   (should not happen under the construction-time feasibility check) or the
///   uniform stream degenerates;
/// * [`IdentError::Stats`] if the inverse normal CDF fails on a stream
///   uniform.
pub fn zero_constrained_rotation(
    base: &[Mat<f64>],
    zeros: &ZeroRestrictionSet,
    stream: &mut Stream,
) -> Result<(Mat<f64>, f64), IdentError> {
    let n = zeros.n_vars();
    if base.is_empty() {
        return Err(IdentError::Dimension {
            what: "base IRF set must be non-empty",
            expected: 1,
            got: 0,
        });
    }
    for m in base {
        if m.nrows() != n || m.ncols() != n {
            return Err(IdentError::Dimension {
                what: "each base IRF matrix must be n x n",
                expected: n,
                got: m.nrows(),
            });
        }
    }
    // Every restriction horizon must index into base.
    if zeros.horizon() + 1 > base.len() {
        return Err(IdentError::Dimension {
            what: "base IRF set is shorter than the restriction horizon + 1",
            expected: zeros.horizon() + 1,
            got: base.len(),
        });
    }

    let mut q = Mat::<f64>::zeros(n, n);

    for (t, &shock) in zeros.order().iter().enumerate() {
        // Basis of row-space(M): previously-built columns (already
        // orthonormal), then the zero-restriction rows for this shock.
        let mut basis: Vec<Vec<f64>> = Vec::with_capacity(n);
        for &prev in zeros.order().iter().take(t) {
            let col: Vec<f64> = (0..n).map(|i| q[(i, prev)]).collect();
            basis.push(col);
        }
        for row in zeros.rows_for_shock(shock, base) {
            orthonormalize_push(&mut basis, row);
        }

        // Draw a Gaussian vector and project out the row space.
        let mut x = vec![0.0f64; n];
        for xi in x.iter_mut() {
            *xi = std_normal(stream)?;
        }
        for u in &basis {
            let d = dot(u, &x);
            for (xi, ui) in x.iter_mut().zip(u.iter()) {
                *xi -= d * ui;
            }
        }
        let w_norm = norm(&x);
        if w_norm <= MGS_DROP_TOL {
            return Err(IdentError::NoConvergence {
                what: "zero-restricted null space collapsed (over-restricted rotation column)",
            });
        }
        let inv = 1.0 / w_norm;
        for i in 0..n {
            q[(i, shock)] = x[i] * inv;
        }
    }

    let weight = arw_log_volume(zeros).exp();
    Ok((q, weight))
}

/// Candidate structural IRF `Theta_h = Theta^chol_h Q` (post-multiplying each
/// horizon matrix by the rotation). Shared shape with the sign sampler.
pub(crate) fn structural_irf(base: &[Mat<f64>], q: MatRef<'_, f64>) -> Vec<Mat<f64>> {
    base.iter().map(|m| m.as_ref() * q).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tsecon_bayes::cholesky_irf;

    /// A small stable VAR(1) in the crate regressor layout and a PD covariance
    /// (`Sigma = A A'`, `A` lower-triangular positive-diagonal).
    fn toy_var() -> (Mat<f64>, Mat<f64>) {
        let n = 3;
        let phi = [[0.5, 0.1, 0.0], [0.0, 0.4, 0.1], [0.1, 0.0, 0.3]];
        let mut b = Mat::<f64>::zeros(1 + n, n);
        for i in 0..n {
            for v in 0..n {
                b[(1 + v, i)] = phi[i][v];
            }
        }
        let a = [[1.0, 0.0, 0.0], [0.4, 0.9, 0.0], [0.2, 0.3, 0.7]];
        let sigma = Mat::from_fn(n, n, |i, j| {
            a[i].iter().zip(a[j].iter()).map(|(x, y)| x * y).sum()
        });
        (b, sigma)
    }

    fn orthogonal_to(q: MatRef<'_, f64>, tol: f64) -> bool {
        let n = q.nrows();
        for a in 0..n {
            for b in 0..n {
                let mut s = 0.0;
                for i in 0..n {
                    s += q[(i, a)] * q[(i, b)];
                }
                let target = if a == b { 1.0 } else { 0.0 };
                if (s - target).abs() > tol {
                    return false;
                }
            }
        }
        true
    }

    #[test]
    fn descending_order_and_feasibility() -> Result<(), IdentError> {
        // Recursive pattern: shock j gets zeros for all i < j.
        let mut rs = Vec::new();
        for j in 0..3 {
            for i in 0..j {
                rs.push(ZeroRestriction::at(i, j, 0));
            }
        }
        let set = ZeroRestrictionSet::new(rs, 3, 6)?;
        assert_eq!(set.order(), &[2, 1, 0]);
        assert_eq!(set.zeros_per_shock(), &[0, 1, 2]);
        assert!(set.all_impact_only());
        Ok(())
    }

    #[test]
    fn over_restricted_is_rejected() {
        // Shock 0 with 3 zeros in a 3-var model: at step t=0 needs z <= 2.
        let rs = vec![
            ZeroRestriction::at(0, 0, 0),
            ZeroRestriction::at(1, 0, 0),
            ZeroRestriction::at(2, 0, 0),
        ];
        assert!(matches!(
            ZeroRestrictionSet::new(rs, 3, 6),
            Err(IdentError::InvalidArgument { .. })
        ));
    }

    #[test]
    fn out_of_range_is_rejected() {
        let rs = vec![ZeroRestriction::at(5, 0, 0)];
        assert!(matches!(
            ZeroRestrictionSet::new(rs, 3, 6),
            Err(IdentError::RestrictionOutOfRange { .. })
        ));
        let rs = vec![ZeroRestriction::at(0, 0, 9)];
        assert!(matches!(
            ZeroRestrictionSet::new(rs, 3, 6),
            Err(IdentError::RestrictionOutOfRange { .. })
        ));
    }

    #[test]
    fn duplicates_are_collapsed() -> Result<(), IdentError> {
        let rs = vec![ZeroRestriction::at(0, 1, 0), ZeroRestriction::at(0, 1, 0)];
        let set = ZeroRestrictionSet::new(rs, 3, 6)?;
        assert_eq!(set.restrictions().len(), 1);
        assert_eq!(set.zeros_per_shock()[1], 1);
        Ok(())
    }

    #[test]
    fn recursive_rotation_is_identity_after_normalization() -> Result<(), IdentError> {
        // Strict-upper-triangle impact zeros => Q = I up to column signs; the
        // positive-diagonal normalization fixes signs, so Theta_0 = P.
        let (b, sigma) = toy_var();
        let horizon = 8;
        let base = cholesky_irf(b.as_ref(), sigma.as_ref(), 1, horizon)?;
        let mut rs = Vec::new();
        for j in 0..3 {
            for i in 0..j {
                rs.push(ZeroRestriction::at(i, j, 0));
            }
        }
        let set = ZeroRestrictionSet::new(rs, 3, horizon)?;
        let mut streams = Stream::substreams(123, 1)?;
        let (q, w) = zero_constrained_rotation(&base, &set, &mut streams[0])?;
        assert_eq!(w, 1.0);
        assert!(orthogonal_to(q.as_ref(), 1e-12));
        // Each column is +/- a unit basis vector.
        for j in 0..3 {
            let mut nonzero = 0;
            for i in 0..3 {
                if q[(i, j)].abs() > 1e-9 {
                    nonzero += 1;
                    assert!((q[(i, j)].abs() - 1.0).abs() < 1e-9);
                }
            }
            assert_eq!(nonzero, 1);
        }
        Ok(())
    }

    #[test]
    fn general_rotation_satisfies_zeros() -> Result<(), IdentError> {
        // A non-recursive, horizon>=1 pattern: shock 0 has a zero on variable
        // 2 at impact and a zero on variable 1 at horizon 2.
        let (b, sigma) = toy_var();
        let horizon = 6;
        let base = cholesky_irf(b.as_ref(), sigma.as_ref(), 1, horizon)?;
        let rs = vec![ZeroRestriction::at(2, 0, 0), ZeroRestriction::at(1, 0, 2)];
        let set = ZeroRestrictionSet::new(rs, 3, horizon)?;
        assert!(!set.all_impact_only());
        let mut streams = Stream::substreams(7, 1)?;
        let (q, _w) = zero_constrained_rotation(&base, &set, &mut streams[0])?;
        assert!(orthogonal_to(q.as_ref(), 1e-12));
        let theta = structural_irf(&base, q.as_ref());
        assert!(theta[0][(2, 0)].abs() < 1e-12, "impact zero not satisfied");
        assert!(
            theta[2][(1, 0)].abs() < 1e-12,
            "horizon-2 zero not satisfied"
        );
        Ok(())
    }

    #[test]
    fn empty_zero_set_gives_full_sphere_rotation() -> Result<(), IdentError> {
        let (b, sigma) = toy_var();
        let base = cholesky_irf(b.as_ref(), sigma.as_ref(), 1, 4)?;
        let set = ZeroRestrictionSet::new(Vec::new(), 3, 4)?;
        assert!(set.all_impact_only()); // vacuously
        let mut streams = Stream::substreams(9, 1)?;
        let (q, w) = zero_constrained_rotation(&base, &set, &mut streams[0])?;
        assert_eq!(w, 1.0);
        assert!(orthogonal_to(q.as_ref(), 1e-12));
        Ok(())
    }
}
