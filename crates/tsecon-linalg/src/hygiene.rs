//! Positive-definiteness hygiene helpers.
//!
//! Per the architecture doc, every covariance matrix the library emits
//! passes through one central symmetrize-and-factorize path: `0.5 (M + M')`
//! followed by an `L L'` Cholesky with a bounded, logged jitter ladder.
//! These are the shared utilities implementing that path.

use faer::{Mat, MatRef, Side};

use crate::error::LinalgError;

/// Relative size of the first jitter rung: `1e-12 * scale`, where `scale`
/// is the mean absolute diagonal of the matrix.
const JITTER_INITIAL_REL: f64 = 1e-12;
/// Relative size of the last jitter rung: `1e-8 * scale` (architecture
/// doc ladder `1e-12 -> 1e-8`).
const JITTER_MAX_REL: f64 = 1e-8;
/// Multiplicative step between rungs.
const JITTER_STEP: f64 = 10.0;

/// Returns the symmetric part `0.5 (M + M')` of a square matrix.
///
/// Floating-point covariance updates (e.g. `P - K F K'` in a Kalman
/// filter) drift off exact symmetry; downstream Cholesky and eigenvalue
/// routines that read only one triangle then silently disagree with ones
/// that read the other. Symmetrizing restores the invariant exactly:
/// the result satisfies `out[(i, j)] == out[(j, i)]` bitwise.
///
/// # Errors
///
/// * [`LinalgError::NotSquare`] if `m` is not square;
/// * [`LinalgError::NonFinite`] if `m` contains NaN/infinity.
pub fn symmetrize(m: MatRef<'_, f64>) -> Result<Mat<f64>, LinalgError> {
    let n = m.nrows();
    if m.ncols() != n {
        return Err(LinalgError::NotSquare {
            what: "m",
            rows: m.nrows(),
            cols: m.ncols(),
        });
    }
    let mut out = Mat::<f64>::zeros(n, n);
    for j in 0..n {
        for i in 0..=j {
            let v = 0.5 * (m[(i, j)] + m[(j, i)]);
            if !v.is_finite() {
                return Err(LinalgError::NonFinite { what: "m" });
            }
            out[(i, j)] = v;
            out[(j, i)] = v;
        }
    }
    Ok(out)
}

/// Result of [`jittered_cholesky`]: the factor plus a report of how much
/// diagonal regularization was needed.
#[derive(Debug, Clone)]
pub struct JitteredCholesky {
    /// Lower-triangular `L` with `sym(M) + jitter * I = L L'`
    /// (strict upper triangle is zero).
    pub factor: Mat<f64>,
    /// The jitter that was actually added to the diagonal
    /// (`0.0` when the clean factorization succeeded).
    pub jitter: f64,
    /// Number of factorization attempts, including the clean first one
    /// (`1` means no jitter was needed).
    pub attempts: usize,
}

impl JitteredCholesky {
    /// Log-determinant of the factorized matrix, `2 sum_i ln L_{ii}`.
    ///
    /// This is the numerically correct way to obtain a covariance
    /// log-determinant — never via an explicit determinant.
    pub fn log_det(&self) -> f64 {
        let n = self.factor.nrows();
        let mut s = 0.0;
        for i in 0..n {
            s += self.factor[(i, i)].ln();
        }
        2.0 * s
    }
}

/// Cholesky factorization with a bounded jitter ladder.
///
/// The input is first symmetrized (`0.5 (M + M')`), then factorized with
/// `faer`'s `L L'` decomposition. If the clean factorization fails (a
/// nonpositive pivot: the matrix is numerically indefinite), a scaled
/// identity `jitter * I` is added to the diagonal and the factorization is
/// retried, with `jitter` walking the ladder
///
/// ```text
/// jitter = scale * {1e-12, 1e-11, ..., 1e-8},   scale = mean |diag(M)|
/// ```
///
/// (five jittered attempts after the clean one; `scale` falls back to 1
/// when the diagonal is entirely zero). The jitter actually used is
/// reported in [`JitteredCholesky::jitter`] so callers can log it — a
/// triggered ladder is a diagnostic signal, not a silent repair.
///
/// # Errors
///
/// * [`LinalgError::NotSquare`] / [`LinalgError::EmptyInput`] on shape
///   violations;
/// * [`LinalgError::NonFinite`] if `m` contains NaN/infinity;
/// * [`LinalgError::JitterExhausted`] if the matrix is still not positive
///   definite at the top of the ladder — the matrix is genuinely
///   indefinite (e.g. a negative eigenvalue far below roundoff), and the
///   caller should fall back to an eigenvalue-based repair rather than
///   larger jitter.
pub fn jittered_cholesky(m: MatRef<'_, f64>) -> Result<JitteredCholesky, LinalgError> {
    let n = m.nrows();
    if m.ncols() != n {
        return Err(LinalgError::NotSquare {
            what: "m",
            rows: m.nrows(),
            cols: m.ncols(),
        });
    }
    if n == 0 {
        return Err(LinalgError::EmptyInput { what: "m" });
    }
    let sym = symmetrize(m)?;

    // Scale for the jitter: mean absolute diagonal, guarded against an
    // all-zero diagonal.
    let mut scale = 0.0;
    for i in 0..n {
        scale += sym[(i, i)].abs();
    }
    scale /= n as f64;
    if scale == 0.0 {
        scale = 1.0;
    }

    let mut attempts = 0usize;
    let mut jitter = 0.0f64;
    loop {
        attempts += 1;
        let candidate = if jitter == 0.0 {
            sym.clone()
        } else {
            let mut c = sym.clone();
            for i in 0..n {
                c[(i, i)] += jitter;
            }
            c
        };
        match candidate.llt(Side::Lower) {
            Ok(llt) => {
                return Ok(JitteredCholesky {
                    factor: llt.L().to_owned(),
                    jitter,
                    attempts,
                });
            }
            Err(_) => {
                let next = if jitter == 0.0 {
                    scale * JITTER_INITIAL_REL
                } else {
                    jitter * JITTER_STEP
                };
                // Allow one final attempt exactly at the top rung, with a
                // tolerance for the accumulated multiplication roundoff.
                if next > scale * JITTER_MAX_REL * (1.0 + 1e-9) {
                    return Err(LinalgError::JitterExhausted {
                        attempts,
                        max_jitter: jitter,
                    });
                }
                jitter = next;
            }
        }
    }
}
