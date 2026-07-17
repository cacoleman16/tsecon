//! Private dense helpers: small vector/matrix kernels, triangular solves,
//! a cyclic-Jacobi symmetric eigendecomposition (the rank-aware primitive
//! behind pseudo-inverses and PSD square roots), and inverse-CDF draws
//! from a [`tsecon_rng::Stream`].
//!
//! State and covariance dimensions in this crate are small (a handful of
//! states, a handful of variables), so simple `O(n^3)` kernels with exact
//! symmetry hygiene beat any clever blocking.

use tsecon_linalg::faer::{Mat, MatRef};
use tsecon_rng::Stream;
use tsecon_stats::special::inv_norm_cdf;

use crate::error::BayesError;

/// `m v` for a matrix and a slice.
pub(crate) fn mat_vec(m: MatRef<'_, f64>, v: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; m.nrows()];
    for (j, &vj) in v.iter().enumerate() {
        if vj != 0.0 {
            for (i, o) in out.iter_mut().enumerate() {
                *o += m[(i, j)] * vj;
            }
        }
    }
    out
}

/// Exact symmetrization in place: `m <- 0.5 (m + m')`.
pub(crate) fn symmetrize_in_place(m: &mut Mat<f64>) {
    let n = m.nrows();
    for j in 0..n {
        for i in 0..j {
            let v = 0.5 * (m[(i, j)] + m[(j, i)]);
            m[(i, j)] = v;
            m[(j, i)] = v;
        }
    }
}

/// Squared Frobenius norm.
pub(crate) fn frob_sq(m: MatRef<'_, f64>) -> f64 {
    let mut s = 0.0;
    for j in 0..m.ncols() {
        for i in 0..m.nrows() {
            s += m[(i, j)] * m[(i, j)];
        }
    }
    s
}

/// Solves `L x = b` in place for lower-triangular `L` (forward
/// substitution). The caller guarantees a nonzero diagonal.
pub(crate) fn forward_solve_in_place(l: MatRef<'_, f64>, b: &mut [f64]) {
    let n = b.len();
    for i in 0..n {
        let mut s = b[i];
        for j in 0..i {
            s -= l[(i, j)] * b[j];
        }
        b[i] = s / l[(i, i)];
    }
}

/// Solves `L' x = b` in place for lower-triangular `L` (backward
/// substitution on the transpose).
pub(crate) fn backward_solve_in_place(l: MatRef<'_, f64>, b: &mut [f64]) {
    let n = b.len();
    for i in (0..n).rev() {
        let mut s = b[i];
        for j in (i + 1)..n {
            s -= l[(j, i)] * b[j];
        }
        b[i] = s / l[(i, i)];
    }
}

/// Solves `(L L') X = B` column by column given the lower Cholesky factor
/// `L`; returns `X`.
pub(crate) fn chol_solve_mat(l: MatRef<'_, f64>, b: MatRef<'_, f64>) -> Mat<f64> {
    let n = l.nrows();
    let k = b.ncols();
    let mut out = Mat::<f64>::zeros(n, k);
    let mut col = vec![0.0; n];
    for c in 0..k {
        for (r, slot) in col.iter_mut().enumerate() {
            *slot = b[(r, c)];
        }
        forward_solve_in_place(l, &mut col);
        backward_solve_in_place(l, &mut col);
        for (r, &v) in col.iter().enumerate() {
            out[(r, c)] = v;
        }
    }
    out
}

/// `(L L')^{-1}` given the lower Cholesky factor `L`, exactly symmetrized.
pub(crate) fn chol_inverse(l: MatRef<'_, f64>) -> Mat<f64> {
    let n = l.nrows();
    let eye = Mat::<f64>::identity(n, n);
    let mut inv = chol_solve_mat(l, eye.as_ref());
    symmetrize_in_place(&mut inv);
    inv
}

/// Eigendecomposition `A = V diag(w) V'` of a symmetric matrix.
pub(crate) struct SymEigen {
    /// Eigenvalues (unsorted, as left by the Jacobi sweeps).
    pub(crate) values: Vec<f64>,
    /// Orthonormal eigenvectors, one per column.
    pub(crate) vectors: Mat<f64>,
}

impl SymEigen {
    /// A square root `S` with `S S' = A_+`, where `A_+` clips negative
    /// eigenvalues (roundoff on a PSD input) to zero:
    /// `S = V diag(sqrt(max(w, 0)))`.
    pub(crate) fn psd_sqrt(&self) -> Mat<f64> {
        let n = self.values.len();
        Mat::from_fn(n, n, |i, j| {
            self.vectors[(i, j)] * self.values[j].max(0.0).sqrt()
        })
    }

    /// The Moore-Penrose pseudo-inverse `A^+ = V diag(1/w or 0) V'`,
    /// treating eigenvalues at or below `rel_tol * max|w|` as exact zeros
    /// (the rank decision).
    pub(crate) fn pinv(&self, rel_tol: f64) -> Mat<f64> {
        let n = self.values.len();
        let scale = self.values.iter().fold(0.0f64, |a, &w| a.max(w.abs()));
        let cutoff = rel_tol * scale;
        let inv_w: Vec<f64> = self
            .values
            .iter()
            .map(|&w| if w > cutoff { 1.0 / w } else { 0.0 })
            .collect();
        let mut out = Mat::<f64>::zeros(n, n);
        for j in 0..n {
            for i in 0..=j {
                let mut s = 0.0;
                for (k, &iw) in inv_w.iter().enumerate() {
                    if iw != 0.0 {
                        s += self.vectors[(i, k)] * iw * self.vectors[(j, k)];
                    }
                }
                out[(i, j)] = s;
                out[(j, i)] = s;
            }
        }
        out
    }
}

/// Iteration budget for the cyclic Jacobi sweeps (each sweep rotates every
/// off-diagonal pair once; quadratic convergence sets in after a few).
const JACOBI_MAX_SWEEPS: usize = 100;

/// Symmetric eigendecomposition by cyclic Jacobi rotations (Golub & Van
/// Loan 2013, §8.5): numerically robust for the small covariance matrices
/// this crate handles, with orthonormal eigenvectors to machine precision.
///
/// The input is read as its exactly symmetrized part.
pub(crate) fn sym_eigen(a: MatRef<'_, f64>) -> Result<SymEigen, BayesError> {
    let n = a.nrows();
    if a.ncols() != n {
        return Err(BayesError::Dimension {
            what: "sym_eigen requires a square matrix",
            expected: n,
            got: a.ncols(),
        });
    }
    let mut m = Mat::from_fn(n, n, |i, j| 0.5 * (a[(i, j)] + a[(j, i)]));
    for j in 0..n {
        for i in 0..n {
            if !m[(i, j)].is_finite() {
                return Err(BayesError::NonFinite {
                    what: "sym_eigen input",
                });
            }
        }
    }
    let mut v = Mat::<f64>::identity(n, n);
    if n == 1 {
        return Ok(SymEigen {
            values: vec![m[(0, 0)]],
            vectors: v,
        });
    }

    let scale = frob_sq(m.as_ref()).sqrt().max(f64::MIN_POSITIVE);
    let tol = 1e-30 * scale * scale; // squared off-diagonal target

    for _sweep in 0..JACOBI_MAX_SWEEPS {
        // Sum of squared off-diagonal entries (upper triangle, doubled).
        let mut off = 0.0;
        for j in 0..n {
            for i in 0..j {
                off += 2.0 * m[(i, j)] * m[(i, j)];
            }
        }
        if off <= tol {
            return Ok(SymEigen {
                values: (0..n).map(|i| m[(i, i)]).collect(),
                vectors: v,
            });
        }
        for p in 0..(n - 1) {
            for q in (p + 1)..n {
                let apq = m[(p, q)];
                if apq.abs() <= 1e-300 {
                    continue;
                }
                // Classical Jacobi rotation angle (Golub & Van Loan 8.5.2).
                let theta = (m[(q, q)] - m[(p, p)]) / (2.0 * apq);
                let t = if theta.abs() > 1e150 {
                    // Avoid overflow in theta^2: t ~ 1/(2 theta).
                    0.5 / theta
                } else {
                    theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt())
                };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                // Update m = J' m J on rows/columns p and q.
                for k in 0..n {
                    let mkp = m[(k, p)];
                    let mkq = m[(k, q)];
                    m[(k, p)] = c * mkp - s * mkq;
                    m[(k, q)] = s * mkp + c * mkq;
                }
                for k in 0..n {
                    let mpk = m[(p, k)];
                    let mqk = m[(q, k)];
                    m[(p, k)] = c * mpk - s * mqk;
                    m[(q, k)] = s * mpk + c * mqk;
                }
                // Accumulate the eigenvector rotation.
                for k in 0..n {
                    let vkp = v[(k, p)];
                    let vkq = v[(k, q)];
                    v[(k, p)] = c * vkp - s * vkq;
                    v[(k, q)] = s * vkp + c * vkq;
                }
            }
        }
    }
    Err(BayesError::NoConvergence {
        what: "cyclic Jacobi symmetric eigendecomposition",
    })
}

/// Retry budget for rejecting the (probability `2^-53`) exact-zero uniform
/// that the inverse CDF cannot accept.
const UNIFORM_RETRIES: usize = 128;

/// A uniform draw strictly inside `(0, 1)`: rejects the exact 0 that
/// [`Stream::uniform_f64`] can produce (1 is unreachable by construction).
pub(crate) fn positive_uniform(stream: &mut Stream) -> Result<f64, BayesError> {
    for _ in 0..UNIFORM_RETRIES {
        let u = stream.uniform_f64();
        if u > 0.0 {
            return Ok(u);
        }
    }
    Err(BayesError::NoConvergence {
        what: "positive uniform draw (stream returned 0 repeatedly)",
    })
}

/// One standard normal draw by inverse-CDF transform of a stream uniform
/// (Wichura AS241 `inv_norm_cdf`, ~1e-16 relative accuracy).
///
/// TODO(phase0): replace with a ziggurat sampler once the shared RNG layer
/// grows one; the inverse CDF is exact but roughly 5x slower per draw.
pub(crate) fn std_normal(stream: &mut Stream) -> Result<f64, BayesError> {
    Ok(inv_norm_cdf(positive_uniform(stream)?)?)
}
