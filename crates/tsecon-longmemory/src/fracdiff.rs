//! Fractional differencing `(1 - L)^d` and its exact inverse, fractional
//! integration `(1 - L)^{-d}`.
//!
//! The fractional-difference operator is defined by the binomial series
//!
//! ```text
//!   (1 - L)^d = sum_{k=0}^inf pi_k(d) L^k,
//! ```
//!
//! whose coefficients obey the exact recursion (Hosking 1981; the binomial
//! theorem for a real exponent)
//!
//! ```text
//!   pi_0(d) = 1,   pi_k(d) = pi_{k-1}(d) * (k - 1 - d) / k,   k = 1, 2, ...
//! ```
//!
//! (equivalently `pi_k(d) = (-1)^k * binom(d, k) = Gamma(k - d) /
//! (Gamma(-d) * Gamma(k + 1))`). Applied to a finite sample `x_0, ..., x_{n-1}`
//! and truncated at the start of the sample, the filtered series is the finite
//! convolution
//!
//! ```text
//!   y_t = sum_{k=0}^{t} pi_k(d) * x_{t-k},   t = 0, ..., n-1.
//! ```
//!
//! ## Exact invertibility
//!
//! Because the truncated filter is a lower-triangular Toeplitz operator with
//! unit diagonal (`pi_0 = 1`), and `(1 - L)^d (1 - L)^{-d} = 1` as a power
//! series, applying `frac_diff(., d)` and then `frac_integrate(., d)` recovers
//! the original series *exactly* (up to floating-point round-off). That is the
//! sense in which [`frac_integrate`] is the inverse of [`frac_diff`].

use crate::error::LongMemoryError;

/// The first `n_weights` binomial coefficients `pi_0(d), ..., pi_{n_weights-1}(d)`
/// of the fractional-difference operator `(1 - L)^d`.
///
/// Computed by the exact recursion `pi_0 = 1`, `pi_k = pi_{k-1} (k-1-d)/k`.
/// Passing a negative `d` yields the coefficients of the fractional-integration
/// operator `(1 - L)^{-|d|}`.
///
/// # Errors
///
/// [`LongMemoryError::EmptyInput`] if `n_weights == 0`;
/// [`LongMemoryError::InvalidArgument`] if `d` is not finite.
///
/// # Example
/// ```
/// use tsecon_longmemory::frac_diff_weights;
/// // (1 - L)^1 = 1 - L: weights are exactly [1, -1, 0, 0, ...].
/// let w = frac_diff_weights(1.0, 4).unwrap();
/// assert_eq!(w, vec![1.0, -1.0, 0.0, 0.0]);
/// ```
pub fn frac_diff_weights(d: f64, n_weights: usize) -> Result<Vec<f64>, LongMemoryError> {
    if n_weights == 0 {
        return Err(LongMemoryError::EmptyInput { what: "n_weights" });
    }
    if !d.is_finite() {
        return Err(LongMemoryError::InvalidArgument {
            what: "the memory parameter d must be finite",
        });
    }
    let mut w = vec![0.0_f64; n_weights];
    w[0] = 1.0;
    for k in 1..n_weights {
        // pi_k = pi_{k-1} * (k - 1 - d) / k.
        w[k] = w[k - 1] * ((k as f64 - 1.0 - d) / k as f64);
    }
    Ok(w)
}

/// Fractionally difference `x` by order `d`: apply `(1 - L)^d` as the
/// start-of-sample-truncated finite convolution
/// `y_t = sum_{k=0}^{t} pi_k(d) x_{t-k}`.
///
/// For integer `d` this reduces to the ordinary difference (`d = 1` gives
/// `y_t = x_t - x_{t-1}` with `y_0 = x_0`). For fractional `d in (0, 0.5)` it
/// produces the stationary long-memory transform; for `d < 0` it fractionally
/// integrates (equivalently, call [`frac_integrate`] with `-d`).
///
/// # Errors
///
/// [`LongMemoryError::EmptyInput`] if `x` is empty;
/// [`LongMemoryError::NonFinite`] if `x` holds a NaN/infinity;
/// [`LongMemoryError::InvalidArgument`] if `d` is not finite.
///
/// # Example
/// ```
/// use tsecon_longmemory::frac_diff;
/// let x = vec![1.0, 2.0, 3.0, 4.0];
/// // d = 1 is the plain first difference (with y_0 = x_0).
/// let y = frac_diff(&x, 1.0).unwrap();
/// assert_eq!(y, vec![1.0, 1.0, 1.0, 1.0]);
/// ```
pub fn frac_diff(x: &[f64], d: f64) -> Result<Vec<f64>, LongMemoryError> {
    let n = x.len();
    if n == 0 {
        return Err(LongMemoryError::EmptyInput { what: "x" });
    }
    check_finite(x, "x")?;
    // frac_diff_weights validates `d`; it cannot hit the empty-weights branch
    // because `n >= 1` here.
    let w = frac_diff_weights(d, n)?;
    let mut y = vec![0.0_f64; n];
    for (t, yt) in y.iter_mut().enumerate() {
        let mut acc = 0.0;
        for (k, &wk) in w.iter().enumerate().take(t + 1) {
            acc += wk * x[t - k];
        }
        *yt = acc;
    }
    Ok(y)
}

/// Fractionally integrate `x` by order `d`: apply `(1 - L)^{-d}`.
///
/// This is the exact inverse of [`frac_diff`] by the same `d`: for any finite
/// series, `frac_integrate(frac_diff(x, d), d) == x` up to floating-point
/// round-off, because the truncated filter is lower-triangular Toeplitz with
/// unit diagonal. Implemented as `frac_diff(x, -d)`.
///
/// # Errors
///
/// Same as [`frac_diff`].
///
/// # Example
/// ```
/// use tsecon_longmemory::{frac_diff, frac_integrate};
/// let x = vec![0.5, -1.2, 3.3, 2.0, -0.7];
/// let d = 0.3;
/// let recovered = frac_integrate(&frac_diff(&x, d).unwrap(), d).unwrap();
/// for (a, b) in recovered.iter().zip(x.iter()) {
///     assert!((a - b).abs() < 1e-12);
/// }
/// ```
pub fn frac_integrate(x: &[f64], d: f64) -> Result<Vec<f64>, LongMemoryError> {
    frac_diff(x, -d)
}

/// Reject a series containing a NaN or an infinite entry.
fn check_finite(x: &[f64], what: &'static str) -> Result<(), LongMemoryError> {
    if x.iter().all(|v| v.is_finite()) {
        Ok(())
    } else {
        Err(LongMemoryError::NonFinite { what })
    }
}
