//! Equal-weighted cosine (EWC) long-run variance estimation — the
//! Lazarus-Lewis-Stock-Watson (2018) recommendation and this library's
//! default HAR inference policy (ROADMAP §5).
//!
//! Instead of kernel-weighting autocovariances, the EWC estimator projects
//! the (mean-zero) series onto the first `B` type-II discrete cosine basis
//! vectors and averages the squared coefficients:
//!
//! ```text
//! Lambda_j  = sqrt(2/n) * sum_{t=1}^{n} cos(pi * j * (t - 1/2)/n) * x_t
//! omega_hat = (1/B) * sum_{j=1}^{B} Lambda_j^2
//! ```
//!
//! Because the `Lambda_j` are asymptotically i.i.d. `N(0, omega)`,
//! `omega_hat` is (asymptotically) `omega * chi^2_B / B`, which delivers
//! **exact-t fixed-b inference**: a test of `theta = theta_0` uses
//! `t = (theta_hat - theta_0) / sqrt(omega_hat / n)` compared against
//! Student-t critical values with `B` degrees of freedom — *not* the
//! normal. That t_B rule, with `B = round(0.4 * n^(2/3))`
//! ([`ewc_default_b`]), is the library-wide default inference policy per
//! Lazarus, Lewis, Stock & Watson (2018, Journal of Business & Economic
//! Statistics, "HAR Inference: Recommendations for Practice").

use core::f64::consts::PI;

use crate::error::HacError;
use crate::validate::{check_finite, check_min_len};

/// Equal-weighted cosine (EWC) long-run variance with `b` degrees of
/// freedom (Lazarus-Lewis-Stock-Watson 2018).
///
/// Computes `(1/B) * sum_{j=1}^{B} Lambda_j^2` with
/// `Lambda_j = sqrt(2/n) * sum_t cos(pi * j * (t - 1/2)/n) * x_t` on a
/// mean-zero series (demean first, or pass residuals/scores). Inference
/// based on this estimator uses Student-t critical values with `B`
/// degrees of freedom — the library-wide default policy; see the module
/// docs. Use [`ewc_default_b`] for the recommended `B`.
///
/// Requires `1 <= b <= n - 1`: together with the excluded `j = 0` (mean)
/// vector, the `n - 1` cosine vectors form an orthonormal basis, so at
/// `b = n - 1` the estimator collapses to the naive (short-run) sample
/// variance `sum_t x_t^2 / (n - 1)` of a demeaned series — no long-run
/// smoothing at all.
///
/// # Errors
///
/// [`HacError::SeriesTooShort`] if `n < 2`, [`HacError::NonFinite`] on
/// NaN/inf input, [`HacError::InvalidDof`] if `b` is 0 or exceeds `n - 1`.
pub fn ewc_lrv(x: &[f64], b: usize) -> Result<f64, HacError> {
    const WHAT: &str = "EWC long-run variance";
    check_min_len(x, 2, WHAT)?;
    check_finite(x, WHAT)?;
    let n = x.len();
    if b == 0 || b > n - 1 {
        return Err(HacError::InvalidDof { what: WHAT, b, n });
    }

    let nf = n as f64;
    let scale = (2.0 / nf).sqrt();
    let mut sum_sq = 0.0;
    for j in 1..=b {
        let pj = PI * j as f64;
        let mut lambda = 0.0;
        for (t, &z) in x.iter().enumerate() {
            lambda += (pj * ((t as f64 + 1.0 - 0.5) / nf)).cos() * z;
        }
        lambda *= scale;
        sum_sq += lambda * lambda;
    }
    Ok(sum_sq / b as f64)
}

/// Matrix-valued EWC long-run covariance of a `k`-variate mean-zero series
/// (Lazarus-Lewis-Stock-Watson 2018):
///
/// ```text
/// Lambda_j  = sqrt(2/n) * sum_{t=1}^{n} cos(pi * j * (t - 1/2)/n) * z_t   (k-vector)
/// omega_hat = (1/B) * sum_{j=1}^{B} Lambda_j Lambda_j'                      (k x k)
/// ```
///
/// `z` holds the `k` series as columns, each of length `n` (the layout
/// [`crate::ols`] takes its design in); the result is the `k x k` matrix in
/// row-major order. This is the meat of the EWC sandwich covariance
/// ([`crate::SeType::Ewc`]): pass the OLS scores `x_t u_t`.
///
/// The cosine weights are evaluated once per `(j, t)` and shared across the
/// columns, and every column accumulates in exactly the order [`ewc_lrv`]
/// uses, so at `k = 1` the single entry is **bit-identical** to
/// `ewc_lrv(&z[0], b)`. Cost `O(n B k + B k^2)`.
///
/// The `j = 0` (mean) vector is excluded, so adding a constant to any
/// column leaves the estimate unchanged in exact arithmetic; demeaning
/// first only helps conditioning.
///
/// # Errors
///
/// [`HacError::EmptyDesign`] with no columns, [`HacError::DimensionMismatch`]
/// if the columns differ in length, [`HacError::SeriesTooShort`] if
/// `n < 2`, [`HacError::NonFinite`] on NaN/inf input, and
/// [`HacError::InvalidDof`] unless `1 <= b <= n - 1`.
pub fn ewc_lrv_matrix(z: &[Vec<f64>], b: usize) -> Result<Vec<f64>, HacError> {
    ewc_matrix_impl(z, b, "EWC long-run covariance")
}

/// [`ewc_lrv_matrix`] with the caller's name for error messages.
pub(crate) fn ewc_matrix_impl(
    z: &[Vec<f64>],
    b: usize,
    what: &'static str,
) -> Result<Vec<f64>, HacError> {
    let k = z.len();
    let Some(first) = z.first() else {
        return Err(HacError::EmptyDesign);
    };
    let n = first.len();
    for (column, col) in z.iter().enumerate() {
        if col.len() != n {
            return Err(HacError::DimensionMismatch {
                what,
                column,
                expected: n,
                got: col.len(),
            });
        }
    }
    for col in z {
        check_min_len(col, 2, what)?;
        check_finite(col, what)?;
    }
    if b == 0 || b > n - 1 {
        return Err(HacError::InvalidDof { what, b, n });
    }

    let nf = n as f64;
    let scale = (2.0 / nf).sqrt();
    let mut cosines = vec![0.0_f64; n];
    let mut lambda = vec![0.0_f64; k];
    let mut omega = vec![0.0_f64; k * k];
    for j in 1..=b {
        let pj = PI * j as f64;
        for (t, c) in cosines.iter_mut().enumerate() {
            *c = (pj * ((t as f64 + 1.0 - 0.5) / nf)).cos();
        }
        for (l, col) in lambda.iter_mut().zip(z.iter()) {
            let mut acc = 0.0;
            for (&c, &v) in cosines.iter().zip(col.iter()) {
                acc += c * v;
            }
            *l = acc * scale;
        }
        for r in 0..k {
            for c in 0..=r {
                omega[r * k + c] += lambda[r] * lambda[c];
            }
        }
    }
    let bf = b as f64;
    for r in 0..k {
        for c in 0..=r {
            let v = omega[r * k + c] / bf;
            omega[r * k + c] = v;
            omega[c * k + r] = v;
        }
    }
    Ok(omega)
}

/// The Lazarus-Lewis-Stock-Watson (2018) recommended EWC degrees of
/// freedom, `B = round(0.4 * n^(2/3))`, clamped into the valid range
/// `[1, n - 1]` (for `n >= 2`; degenerate `n` clamps to 1 and [`ewc_lrv`]
/// will reject it).
///
/// Their size/power analysis selects this rate as the practical
/// recommendation for HAR inference; pair the resulting estimate with
/// Student-t critical values with `B` degrees of freedom.
#[must_use]
pub fn ewc_default_b(n: usize) -> usize {
    let b = (0.4 * (n as f64).powf(2.0 / 3.0)).round() as usize;
    b.clamp(1, n.saturating_sub(1).max(1))
}
