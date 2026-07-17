//! Levinson-Durbin recursion and Toeplitz system solving.
//!
//! Both algorithms exploit the Toeplitz structure of stationary
//! autocovariance matrices to solve in `O(n^2)` operations instead of the
//! `O(n^3)` of a dense factorization. References: Brockwell & Davis (1991),
//! *Time Series: Theory and Methods*, section 5.2; Golub & Van Loan (2013),
//! *Matrix Computations*, section 4.7.

use crate::error::LinalgError;

/// Output of the Levinson-Durbin recursion up to order `p`.
///
/// Conventions follow Brockwell & Davis (1991, section 5.2): for each order
/// `m` the fitted AR(m) model is
///
/// ```text
/// x_t = phi_{m,1} x_{t-1} + ... + phi_{m,m} x_{t-m} + e_t,   Var(e_t) = v_m
/// ```
///
/// and the partial autocorrelation at lag `m` is `phi_{m,m}`.
#[derive(Debug, Clone, PartialEq)]
pub struct LevinsonDurbin {
    /// AR coefficients at each order: `ar_coefs[m - 1]` holds
    /// `[phi_{m,1}, ..., phi_{m,m}]` for `m = 1, ..., p`.
    pub ar_coefs: Vec<Vec<f64>>,
    /// Partial autocorrelations `[1.0, phi_{1,1}, ..., phi_{p,p}]`
    /// (length `p + 1`, with the conventional `pacf[0] = 1`).
    pub pacf: Vec<f64>,
    /// Innovation (one-step prediction error) variances per order,
    /// `[v_0, v_1, ..., v_p]` with `v_0 = gamma(0)` (length `p + 1`).
    pub innovation_variance: Vec<f64>,
}

impl LevinsonDurbin {
    /// The AR coefficients of the highest fitted order `p`
    /// (empty slice when `p = 0`).
    pub fn ar_coefs_final(&self) -> &[f64] {
        self.ar_coefs.last().map(Vec::as_slice).unwrap_or(&[])
    }

    /// The innovation variance of the highest fitted order, `v_p`.
    pub fn innovation_variance_final(&self) -> f64 {
        // `innovation_variance` always has length p + 1 >= 1 by construction.
        self.innovation_variance[self.innovation_variance.len() - 1]
    }
}

/// Runs the Levinson-Durbin recursion on an autocovariance sequence.
///
/// `acov` must contain the autocovariances `[gamma(0), gamma(1), ...,
/// gamma(order)]` (extra trailing lags are ignored). The recursion
/// (Durbin 1960; Brockwell & Davis 1991, proposition 5.2.1) is
///
/// ```text
/// phi_{1,1} = gamma(1) / gamma(0),          v_1 = v_0 (1 - phi_{1,1}^2)
/// phi_{m,m} = [gamma(m) - sum_{j=1}^{m-1} phi_{m-1,j} gamma(m-j)] / v_{m-1}
/// phi_{m,j} = phi_{m-1,j} - phi_{m,m} phi_{m-1,m-j},   j = 1, ..., m-1
/// v_m       = v_{m-1} (1 - phi_{m,m}^2)
/// ```
///
/// with `v_0 = gamma(0)`. This matches the convention used by
/// `statsmodels.tsa.stattools.levinson_durbin` with `isacov=True`.
///
/// # Errors
///
/// * [`LinalgError::EmptyInput`] if `acov` is empty;
/// * [`LinalgError::DimensionMismatch`] if `acov.len() < order + 1`;
/// * [`LinalgError::NonFinite`] if `acov` contains NaN/infinity;
/// * [`LinalgError::NotPositiveDefinite`] if `gamma(0) <= 0` or an
///   innovation variance becomes nonpositive (the sequence is not a valid
///   positive definite autocovariance up to the requested order; this is
///   also the numerically-degenerate `|pacf| >= 1` case).
pub fn levinson_durbin(acov: &[f64], order: usize) -> Result<LevinsonDurbin, LinalgError> {
    if acov.is_empty() {
        return Err(LinalgError::EmptyInput { what: "acov" });
    }
    if acov.len() < order + 1 {
        return Err(LinalgError::DimensionMismatch {
            what: "acov must have at least order + 1 entries",
            expected: order + 1,
            got: acov.len(),
        });
    }
    if acov[..=order].iter().any(|g| !g.is_finite()) {
        return Err(LinalgError::NonFinite { what: "acov" });
    }
    let gamma0 = acov[0];
    if gamma0 <= 0.0 {
        return Err(LinalgError::NotPositiveDefinite {
            what: "gamma(0) must be strictly positive",
        });
    }

    let mut ar_coefs: Vec<Vec<f64>> = Vec::with_capacity(order);
    let mut pacf = Vec::with_capacity(order + 1);
    let mut innov = Vec::with_capacity(order + 1);
    pacf.push(1.0);
    innov.push(gamma0);

    // phi holds the coefficients of the current order.
    let mut phi: Vec<f64> = Vec::with_capacity(order);
    for m in 1..=order {
        let v_prev = innov[m - 1];
        if v_prev <= 0.0 {
            return Err(LinalgError::NotPositiveDefinite {
                what: "innovation variance hit zero during the Levinson-Durbin \
                       recursion (autocovariance sequence is singular)",
            });
        }
        // Reflection coefficient phi_{m,m}.
        let mut num = acov[m];
        for j in 1..m {
            num -= phi[j - 1] * acov[m - j];
        }
        let k = num / v_prev;
        if !k.is_finite() || k.abs() >= 1.0 {
            // |phi_{m,m}| >= 1 implies v_m <= 0: the sequence is not a
            // positive definite autocovariance up to this order, and the
            // recursion is numerically meaningless past this point
            // (architecture doc: detect pacf magnitudes reaching 1).
            return Err(LinalgError::NotPositiveDefinite {
                what: "partial autocorrelation reached magnitude 1 during the \
                       Levinson-Durbin recursion (autocovariance sequence is \
                       not positive definite at this order)",
            });
        }
        // In-place order update: phi_{m,j} = phi_{m-1,j} - k phi_{m-1,m-j}.
        let prev = phi.clone();
        for j in 1..m {
            phi[j - 1] = prev[j - 1] - k * prev[m - 1 - j];
        }
        phi.push(k);
        pacf.push(k);
        innov.push(v_prev * (1.0 - k * k));
        ar_coefs.push(phi.clone());
    }

    Ok(LevinsonDurbin {
        ar_coefs,
        pacf,
        innovation_variance: innov,
    })
}

/// Biased sample autocovariances of the demeaned series, lags `0..=nlags`.
///
/// ```text
/// gamma_hat(h) = (1/n) sum_{t=h}^{n-1} (y_t - ybar)(y_{t-h} - ybar)
/// ```
///
/// The `1/n` (biased) normalization guarantees the sequence is positive
/// semidefinite (Brockwell & Davis 1991, proposition 5.1.1 discussion),
/// which the `1/(n-h)` (unbiased) variant does not. This matches
/// `statsmodels.tsa.stattools.acovf(..., adjusted=False, demean=True)`.
///
/// # Errors
///
/// * [`LinalgError::EmptyInput`] if `y` is empty;
/// * [`LinalgError::DimensionMismatch`] if `y.len() < nlags + 1`;
/// * [`LinalgError::NonFinite`] if `y` contains NaN/infinity.
pub fn autocovariances_biased(y: &[f64], nlags: usize) -> Result<Vec<f64>, LinalgError> {
    if y.is_empty() {
        return Err(LinalgError::EmptyInput { what: "y" });
    }
    if y.len() < nlags + 1 {
        return Err(LinalgError::DimensionMismatch {
            what: "series must have at least nlags + 1 observations",
            expected: nlags + 1,
            got: y.len(),
        });
    }
    if y.iter().any(|v| !v.is_finite()) {
        return Err(LinalgError::NonFinite { what: "y" });
    }
    let n = y.len();
    let mean = y.iter().sum::<f64>() / n as f64;
    let x: Vec<f64> = y.iter().map(|v| v - mean).collect();
    let mut acov = Vec::with_capacity(nlags + 1);
    for h in 0..=nlags {
        let mut s = 0.0;
        for t in h..n {
            s += x[t] * x[t - h];
        }
        acov.push(s / n as f64);
    }
    Ok(acov)
}

/// Levinson-Durbin fit directly from a series.
///
/// Computes biased autocovariances of the demeaned series
/// ([`autocovariances_biased`]) and runs [`levinson_durbin`] on them. This
/// replicates `statsmodels.tsa.stattools.levinson_durbin(y, nlags=order,
/// isacov=False)`.
///
/// # Errors
///
/// Propagates the errors of [`autocovariances_biased`] and
/// [`levinson_durbin`]; a constant series fails with
/// [`LinalgError::NotPositiveDefinite`] since `gamma(0) = 0`.
pub fn levinson_durbin_from_series(y: &[f64], order: usize) -> Result<LevinsonDurbin, LinalgError> {
    let acov = autocovariances_biased(y, order)?;
    levinson_durbin(&acov, order)
}

/// Solves `T x = b` for a symmetric positive definite Toeplitz matrix `T`
/// given its first column, in `O(n^2)` via the Levinson recursion.
///
/// `first_col = [r_0, r_1, ..., r_{n-1}]` defines `T[i, j] = r_{|i-j|}`.
/// The algorithm is Golub & Van Loan (2013), Algorithm 4.7.2 (Levinson):
/// it maintains the Yule-Walker solution `y_k` of the order-`k` subsystem
/// alongside the solution `x_k` of `T_k x = b_{1..k}`, updating both with
/// the reflection coefficients. Validated against
/// `scipy.linalg.solve_toeplitz`.
///
/// # Errors
///
/// * [`LinalgError::EmptyInput`] if `first_col` is empty;
/// * [`LinalgError::DimensionMismatch`] if `rhs.len() != first_col.len()`;
/// * [`LinalgError::NonFinite`] on NaN/infinite inputs;
/// * [`LinalgError::NotPositiveDefinite`] if `r_0 <= 0` or a leading
///   principal minor of `T` is found to be nonpositive during the
///   recursion (the Levinson recursion is only stable for positive
///   definite `T`; for indefinite Toeplitz systems use a dense solver).
pub fn toeplitz_solve(first_col: &[f64], rhs: &[f64]) -> Result<Vec<f64>, LinalgError> {
    let n = first_col.len();
    if n == 0 {
        return Err(LinalgError::EmptyInput { what: "first_col" });
    }
    if rhs.len() != n {
        return Err(LinalgError::DimensionMismatch {
            what: "rhs must have the same length as first_col",
            expected: n,
            got: rhs.len(),
        });
    }
    if first_col.iter().chain(rhs.iter()).any(|v| !v.is_finite()) {
        return Err(LinalgError::NonFinite {
            what: "first_col / rhs",
        });
    }
    let r0 = first_col[0];
    if r0 <= 0.0 {
        return Err(LinalgError::NotPositiveDefinite {
            what: "Toeplitz diagonal r_0 must be strictly positive",
        });
    }

    // Normalize to unit diagonal: solve T_hat x = b with T_hat = T / r_0,
    // then rescale x <- x / r_0 at the end.
    let r: Vec<f64> = first_col[1..].iter().map(|v| v / r0).collect();

    let mut x = vec![0.0; n];
    x[0] = rhs[0];
    if n == 1 {
        x[0] /= r0;
        return Ok(x);
    }

    // y solves the order-k Yule-Walker system T_hat_k y = -(r_1, ..., r_k).
    let mut y = vec![0.0; n - 1];
    y[0] = -r[0];
    let mut beta = 1.0;
    let mut alpha = -r[0];

    for k in 1..n {
        beta *= 1.0 - alpha * alpha;
        if beta <= 0.0 || !beta.is_finite() {
            return Err(LinalgError::NotPositiveDefinite {
                what: "leading principal minor of the Toeplitz matrix is nonpositive",
            });
        }
        // mu = (b_{k+1} - r(1:k)' x(k:-1:1)) / beta
        let mut mu = rhs[k];
        for i in 0..k {
            mu -= r[i] * x[k - 1 - i];
        }
        mu /= beta;
        for i in 0..k {
            x[i] += mu * y[k - 1 - i];
        }
        x[k] = mu;

        if k < n - 1 {
            // alpha = -(r_{k+1} + r(1:k)' y(k:-1:1)) / beta
            let mut a = r[k];
            for i in 0..k {
                a += r[i] * y[k - 1 - i];
            }
            alpha = -a / beta;
            let prev: Vec<f64> = y[..k].to_vec();
            for i in 0..k {
                y[i] = prev[i] + alpha * prev[k - 1 - i];
            }
            y[k] = alpha;
        }
    }

    for xi in &mut x {
        *xi /= r0;
    }
    Ok(x)
}
