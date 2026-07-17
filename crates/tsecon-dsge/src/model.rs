//! The linear rational-expectations model and its reduced form.

use tsecon_linalg::faer::linalg::solvers::{FullPivLu, Solve};
use tsecon_linalg::faer::Mat;

use crate::error::DsgeError;

/// A linearized rational-expectations model in first-order expectational form
///
/// ```text
/// A . E_t[y_{t+1}] = B . y_t + C . z_{t+1}
/// ```
///
/// where the endogenous vector `y_t = [k_t ; x_t]` stacks the `n_predetermined`
/// PREDETERMINED (backward-looking) variables `k_t` on top of the remaining
/// NON-PREDETERMINED (jump / forward-looking) variables `x_t`, and `z_{t+1}` is
/// a mean-zero exogenous innovation with `E_t[z_{t+1}] = 0`.
///
/// `A` and `B` are `n x n`; `C` is `n x n_z` where `n_z` is the number of
/// shocks. The predetermined block occupies the first `n_predetermined` rows
/// and columns of `y` (and hence of `A`, `B`, and the rows of `C`).
///
/// # Convention
///
/// Because `E_t[z_{t+1}] = 0`, the innovation drops out of the forward-looking
/// solve, which works on the homogeneous system `E_t[y_{t+1}] = M y_t` with
/// `M = A^{-1} B`. The innovation re-enters only the realized law of motion of
/// the predetermined block, `k_{t+1} = P k_t + Q z_{t+1}` with `Q` the
/// predetermined rows of `N = A^{-1} C`. Shocks must therefore load on
/// predetermined (exogenous-state) equations, not on jump equations; a shock on
/// a jump row is rejected with [`DsgeError::ShockOnJump`].
#[derive(Debug, Clone)]
pub struct LinearReModel {
    a: Mat<f64>,
    b: Mat<f64>,
    c: Mat<f64>,
    n: usize,
    n_pre: usize,
    n_shocks: usize,
}

/// The reduced form `E_t[y_{t+1}] = M y_t + N z`, with `M = A^{-1} B` and
/// `N = A^{-1} C`.
pub(crate) struct ReducedForm {
    pub(crate) m: Mat<f64>,
    pub(crate) n_mat: Mat<f64>,
}

impl LinearReModel {
    /// Builds a model from its matrices `A` (`n x n`), `B` (`n x n`), and `C`
    /// (`n x n_z`), each given in row-major order as a slice of rows, together
    /// with the number of predetermined variables.
    ///
    /// # Errors
    ///
    /// * [`DsgeError::EmptyInput`] if `A` has no rows;
    /// * [`DsgeError::NotSquare`] / [`DsgeError::DimensionMismatch`] on shape
    ///   violations between `A`, `B`, and `C`;
    /// * [`DsgeError::NonFinite`] on any NaN/infinite entry;
    /// * [`DsgeError::InvalidPartition`] if `n_predetermined > n`.
    pub fn new(
        a: &[Vec<f64>],
        b: &[Vec<f64>],
        c: &[Vec<f64>],
        n_predetermined: usize,
    ) -> Result<Self, DsgeError> {
        let n = a.len();
        if n == 0 {
            return Err(DsgeError::EmptyInput { what: "A" });
        }
        if b.len() != n {
            return Err(DsgeError::DimensionMismatch {
                what: "B must have the same number of rows as A",
                expected: n,
                got: b.len(),
            });
        }
        let a = square("A", a, n)?;
        let b = square("B", b, n)?;
        if c.len() != n {
            return Err(DsgeError::DimensionMismatch {
                what: "C must have one row per endogenous variable (rows of A)",
                expected: n,
                got: c.len(),
            });
        }
        let n_shocks = c[0].len();
        if n_shocks == 0 {
            return Err(DsgeError::EmptyInput {
                what: "C (needs >= 1 shock column)",
            });
        }
        let mut cmat = Mat::<f64>::zeros(n, n_shocks);
        for (i, row) in c.iter().enumerate() {
            if row.len() != n_shocks {
                return Err(DsgeError::DimensionMismatch {
                    what: "every row of C must have the same number of columns",
                    expected: n_shocks,
                    got: row.len(),
                });
            }
            for (j, &v) in row.iter().enumerate() {
                if !v.is_finite() {
                    return Err(DsgeError::NonFinite { what: "C" });
                }
                cmat[(i, j)] = v;
            }
        }
        if n_predetermined > n {
            return Err(DsgeError::InvalidPartition { n_predetermined, n });
        }
        Ok(Self {
            a,
            b,
            c: cmat,
            n,
            n_pre: n_predetermined,
            n_shocks,
        })
    }

    /// The number of endogenous variables `n`.
    #[must_use]
    pub fn n_variables(&self) -> usize {
        self.n
    }

    /// The number of predetermined (backward-looking) variables.
    #[must_use]
    pub fn n_predetermined(&self) -> usize {
        self.n_pre
    }

    /// The number of non-predetermined (jump) variables.
    #[must_use]
    pub fn n_jump(&self) -> usize {
        self.n - self.n_pre
    }

    /// The number of exogenous shocks `n_z`.
    #[must_use]
    pub fn n_shocks(&self) -> usize {
        self.n_shocks
    }

    /// Forms the reduced form `M = A^{-1} B`, `N = A^{-1} C` via a full-pivot
    /// LU of `A`. Returns [`DsgeError::SingularA`] if `A` is numerically
    /// singular.
    pub(crate) fn reduced_form(&self) -> Result<ReducedForm, DsgeError> {
        let lu = FullPivLu::new(self.a.as_ref());
        // Full pivoting orders the pivots in decreasing magnitude, so the ratio
        // of the smallest to the largest diagonal of U is a cheap reciprocal-
        // condition proxy: a tiny ratio means A is (numerically) rank-deficient.
        let u = lu.U();
        let mut max_piv = 0.0f64;
        let mut min_piv = f64::INFINITY;
        for i in 0..self.n {
            let v = u[(i, i)].abs();
            max_piv = max_piv.max(v);
            min_piv = min_piv.min(v);
        }
        if max_piv == 0.0 || min_piv / max_piv < 1e-12 {
            return Err(DsgeError::SingularA);
        }
        let m = lu.solve(self.b.as_ref());
        let n_mat = lu.solve(self.c.as_ref());
        if !is_finite(&m) || !is_finite(&n_mat) {
            return Err(DsgeError::SingularA);
        }
        Ok(ReducedForm { m, n_mat })
    }
}

/// Validates a slice-of-rows as a square `n x n` matrix and copies it into a
/// faer [`Mat`].
fn square(what: &'static str, rows: &[Vec<f64>], n: usize) -> Result<Mat<f64>, DsgeError> {
    let mut m = Mat::<f64>::zeros(n, n);
    for (i, row) in rows.iter().enumerate() {
        if row.len() != n {
            return Err(DsgeError::NotSquare {
                what,
                rows: n,
                cols: row.len(),
            });
        }
        for (j, &v) in row.iter().enumerate() {
            if !v.is_finite() {
                return Err(DsgeError::NonFinite { what });
            }
            m[(i, j)] = v;
        }
    }
    Ok(m)
}

/// True when every entry of `m` is finite.
fn is_finite(m: &Mat<f64>) -> bool {
    for j in 0..m.ncols() {
        for i in 0..m.nrows() {
            if !m[(i, j)].is_finite() {
                return false;
            }
        }
    }
    true
}
