//! Private ordinary-least-squares helper for the regression-based
//! diagnostics ([`crate::pacf_ols`], [`crate::arch_lm`]).
//!
//! Solves the intercept-augmented regression `y = a + X b + e` by centering
//! the columns (which partials out the intercept exactly and drastically
//! improves the conditioning of the normal equations) and solving the
//! centered normal equations `(Xc'Xc) b = Xc'y_c` with a Cholesky
//! factorization plus one step of iterative refinement. At the problem
//! sizes of this crate (a handful of lag columns) this matches a QR/SVD
//! solve to near machine precision.
//!
//! // TODO(phase0): delegate to shared linalg (tsecon-linalg QR) later.

use crate::error::DiagError;

/// Result of an intercept-augmented OLS fit.
pub(crate) struct OlsFit {
    /// Slope coefficients, in the order the columns were supplied.
    pub(crate) slopes: Vec<f64>,
    /// Centered coefficient of determination `1 - RSS/TSS`.
    pub(crate) r_squared: f64,
}

/// Fit `y = a + X b + e` by OLS, where `cols` are the columns of `X`.
///
/// All columns must have the same length as `y`. Requires
/// `n >= p + 2` observations so at least one residual degree of freedom
/// remains.
pub(crate) fn ols_with_intercept(
    cols: &[Vec<f64>],
    y: &[f64],
    what: &'static str,
) -> Result<OlsFit, DiagError> {
    let n = y.len();
    let p = cols.len();
    if n < p + 2 {
        return Err(DiagError::SeriesTooShort {
            what,
            n,
            needed: p + 2,
        });
    }
    // Compile-time-impossible for the callers in this crate, which always
    // build equal-length columns; keep a debug check only.
    debug_assert!(cols.iter().all(|c| c.len() == n));

    let nf = n as f64;
    let y_mean = y.iter().sum::<f64>() / nf;
    let x_means: Vec<f64> = cols.iter().map(|c| c.iter().sum::<f64>() / nf).collect();

    // Centered cross products: s = Xc'Xc (dense symmetric, row-major p x p),
    // rhs = Xc'y_c.
    let mut s = vec![0.0_f64; p * p];
    let mut rhs = vec![0.0_f64; p];
    for i in 0..p {
        for j in 0..=i {
            let mut acc = 0.0;
            for (&xi, &xj) in cols[i].iter().zip(cols[j].iter()) {
                acc += (xi - x_means[i]) * (xj - x_means[j]);
            }
            s[i * p + j] = acc;
            s[j * p + i] = acc;
        }
        let mut acc = 0.0;
        for (&xi, &yt) in cols[i].iter().zip(y.iter()) {
            acc += (xi - x_means[i]) * (yt - y_mean);
        }
        rhs[i] = acc;
    }

    let chol = cholesky(&s, p).ok_or(DiagError::SingularDesign { what })?;
    let mut beta = chol_solve(&chol, p, &rhs);

    // One step of iterative refinement in working precision: cheap and
    // recovers a couple of digits when the lag columns are correlated.
    let mut resid = rhs.clone();
    for i in 0..p {
        let mut acc = 0.0;
        for j in 0..p {
            acc += s[i * p + j] * beta[j];
        }
        resid[i] -= acc;
    }
    let correction = chol_solve(&chol, p, &resid);
    for (b, d) in beta.iter_mut().zip(correction.iter()) {
        *b += d;
    }

    // Centered R^2 = 1 - RSS/TSS with TSS = sum (y - ybar)^2.
    let mut rss = 0.0;
    let mut tss = 0.0;
    for t in 0..n {
        let yc = y[t] - y_mean;
        let mut fit = 0.0;
        for j in 0..p {
            fit += beta[j] * (cols[j][t] - x_means[j]);
        }
        rss += (yc - fit) * (yc - fit);
        tss += yc * yc;
    }
    if tss <= 0.0 {
        return Err(DiagError::ConstantSeries { what });
    }
    // OLS with an intercept guarantees 0 <= RSS <= TSS in exact arithmetic;
    // clamp away last-ulp rounding excursions only.
    let r_squared = (1.0 - rss / tss).clamp(0.0, 1.0);

    Ok(OlsFit {
        slopes: beta,
        r_squared,
    })
}

/// Dense Cholesky factorization `S = L L'` of a symmetric positive-definite
/// `p x p` matrix in row-major storage. Returns the lower factor, or `None`
/// if a pivot is non-positive (or negligibly small relative to the original
/// diagonal), i.e. the matrix is not numerically positive definite.
fn cholesky(s: &[f64], p: usize) -> Option<Vec<f64>> {
    let mut l = vec![0.0_f64; p * p];
    for i in 0..p {
        for j in 0..=i {
            let mut acc = s[i * p + j];
            for k in 0..j {
                acc -= l[i * p + k] * l[j * p + k];
            }
            if i == j {
                // Relative pivot tolerance: guards against collinear lags.
                let tol = s[i * p + i].abs() * 1e-13;
                if acc <= tol.max(f64::MIN_POSITIVE) {
                    return None;
                }
                l[i * p + i] = acc.sqrt();
            } else {
                l[i * p + j] = acc / l[j * p + j];
            }
        }
    }
    Some(l)
}

/// Solve `L L' x = b` given the lower Cholesky factor `L` (row-major).
fn chol_solve(l: &[f64], p: usize, b: &[f64]) -> Vec<f64> {
    // Forward substitution: L z = b.
    let mut z = vec![0.0_f64; p];
    for i in 0..p {
        let mut acc = b[i];
        for k in 0..i {
            acc -= l[i * p + k] * z[k];
        }
        z[i] = acc / l[i * p + i];
    }
    // Back substitution: L' x = z.
    let mut x = vec![0.0_f64; p];
    for i in (0..p).rev() {
        let mut acc = z[i];
        for k in (i + 1)..p {
            acc -= l[k * p + i] * x[k];
        }
        x[i] = acc / l[i * p + i];
    }
    x
}
