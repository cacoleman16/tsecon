//! MIDAS lag-weight functions.
//!
//! A MIDAS regression compresses `K` high-frequency lags of a predictor into
//! a single low-frequency regressor `sum_{k=1}^{K} w_k x_{t,k}` whose weights
//! `w_k` follow a low-dimensional, tightly parameterized shape. This module
//! implements the three classical weight parameterizations, each returned
//! **normalized to sum to one** (the identification convention used
//! throughout: the aggregate slope then multiplies a proper weighted average,
//! so it is comparable across weight schemes and across `K`).
//!
//! Lags are indexed `k = 1, 2, ..., K`, most-recent lag first (`k = 1` is the
//! most recent high-frequency observation entering period `t`), matching the
//! `fixtures/midas.json` stacking convention and R `midasr`'s `mls`
//! embedding.
//!
//! ## Exponential Almon
//!
//! `w_k proportional to exp(theta1 k + theta2 k^2)`, normalized so
//! `sum_k w_k = 1` (Ghysels, Sinko & Valkanov 2007, "MIDAS regressions").
//! With `theta2 < 0` the profile decays; `theta1` tilts the peak. Two
//! parameters generate hump-shaped and monotone decay patterns alike.
//!
//! ## Beta (two-parameter)
//!
//! `w_k proportional to x_k^{t1 - 1} (1 - x_k)^{t2 - 1}` with the lag grid
//! `x_k = (k - 1)/(K - 1)` mapped onto `[0, 1]` (Ghysels, Santa-Clara &
//! Valkanov 2004; Ghysels, Sinko & Valkanov 2007). The endpoints are clamped
//! to `[EPS, 1 - EPS]` with `EPS = 1e-8` so the Beta density is finite at
//! `x = 0, 1` even when a shape parameter is below one — the exact convention
//! the reference implementations (and the golden fixture) use. Requires
//! `t1, t2 > 0`.
//!
//! ## Almon polynomial distributed lag (PDL)
//!
//! Unlike the two above, the Almon/PDL scheme is *linear* in its parameters:
//! the `K` lag coefficients are restricted to lie on a degree-`q` polynomial,
//! `b_k = sum_{j=0}^{q} a_j k^j` (Almon 1965). It is therefore expressed as a
//! `K x (q + 1)` **basis** matrix `Q` with `Q[k, j] = k^j`; regressing the
//! low-frequency target on `X_stacked * Q` and reading back `Q a` recovers the
//! `K` smooth lag coefficients from only `q + 1` free parameters. See
//! [`almon_pdl_basis`]. A convenience [`almon_weights`] evaluates and
//! normalizes the polynomial profile for a given coefficient vector.

use crate::error::MidasError;

/// Endpoint clamp for the Beta lag grid, `x_k in [BETA_EPS, 1 - BETA_EPS]`.
/// Keeps `x^{t1-1}` and `(1-x)^{t2-1}` finite at the boundary lags when a
/// shape parameter is below one, matching the reference MIDAS packages.
pub const BETA_EPS: f64 = 1e-8;

/// Exponential-Almon lag weights,
/// `w_k proportional to exp(theta1 k + theta2 k^2)`, `k = 1..=K`, normalized
/// to sum to one (Ghysels, Sinko & Valkanov 2007).
///
/// The exponents are shifted by their maximum before exponentiating
/// (`exp(g_k - max_j g_j)`), which is algebraically identical after
/// normalization but never overflows, so the function is safe to call at the
/// large `theta` values a nonlinear optimizer may probe.
///
/// # Errors
///
/// [`MidasError::InvalidLagCount`] if `k == 0`;
/// [`MidasError::InvalidWeightParam`] if `theta1` or `theta2` is non-finite;
/// [`MidasError::DegenerateWeights`] if the shifted mass is not finite and
/// positive (only reachable at pathological hyperparameters).
pub fn exp_almon_weights(theta1: f64, theta2: f64, k: usize) -> Result<Vec<f64>, MidasError> {
    if k == 0 {
        return Err(MidasError::InvalidLagCount {
            what: "exponential-Almon weights",
            k,
            needed: 1,
        });
    }
    check_param("exponential-Almon weights", "theta1", theta1, true)?;
    check_param("exponential-Almon weights", "theta2", theta2, true)?;

    let g: Vec<f64> = (1..=k)
        .map(|j| {
            let jf = j as f64;
            theta1 * jf + theta2 * jf * jf
        })
        .collect();
    normalize_log_weights(&g, "exponential-Almon weights")
}

/// Beta (two-parameter) lag weights,
/// `w_k proportional to x_k^{t1 - 1} (1 - x_k)^{t2 - 1}` with
/// `x_k = (k - 1)/(K - 1)` clamped to `[BETA_EPS, 1 - BETA_EPS]`, `k = 1..=K`,
/// normalized to sum to one (Ghysels, Sinko & Valkanov 2007).
///
/// Evaluated in log space (`(t1 - 1) ln x_k + (t2 - 1) ln(1 - x_k)`) with a
/// max-shift before exponentiating, so extreme shape parameters neither
/// overflow nor underflow before normalization.
///
/// # Errors
///
/// [`MidasError::InvalidLagCount`] if `k < 2` (the `x_k` grid is degenerate
/// for a single lag); [`MidasError::InvalidWeightParam`] if `t1` or `t2` is
/// non-finite or not strictly positive; [`MidasError::DegenerateWeights`] if
/// the mass is not finite and positive.
pub fn beta_weights(t1: f64, t2: f64, k: usize) -> Result<Vec<f64>, MidasError> {
    if k < 2 {
        return Err(MidasError::InvalidLagCount {
            what: "Beta weights",
            k,
            needed: 2,
        });
    }
    check_param("Beta weights", "t1", t1, false)?;
    check_param("Beta weights", "t2", t2, false)?;

    let denom = (k - 1) as f64;
    let logs: Vec<f64> = (1..=k)
        .map(|j| {
            let raw = (j - 1) as f64 / denom;
            let x = raw.clamp(BETA_EPS, 1.0 - BETA_EPS);
            (t1 - 1.0) * x.ln() + (t2 - 1.0) * (1.0 - x).ln()
        })
        .collect();
    normalize_log_weights(&logs, "Beta weights")
}

/// The Almon polynomial-distributed-lag basis: a `K x (degree + 1)` matrix
/// `Q` in column-major storage (outer `Vec` = the `degree + 1` columns, each
/// an inner `Vec` of length `K`), with `Q[k - 1][j] = k^j` for lag
/// `k = 1..=K` and power `j = 0..=degree` (Almon 1965).
///
/// Restricting the `K` high-frequency lag coefficients to a degree-`q`
/// polynomial, `b = Q a`, turns U-MIDAS's `K` free lag parameters into just
/// `q + 1`: regress the target on `X_stacked * Q` (columns returned here) to
/// estimate `a`, then form `Q a` for the smooth lag profile. Unlike the
/// exponential-Almon and Beta schemes this is linear in parameters, so it is
/// estimable by ordinary least squares.
///
/// # Errors
///
/// [`MidasError::InvalidLagCount`] if `k == 0`;
/// [`MidasError::InvalidPolynomialDegree`] if `degree + 1 > k` (the basis
/// would have more columns than lags and the restricted design would be
/// rank-deficient).
pub fn almon_pdl_basis(k: usize, degree: usize) -> Result<Vec<Vec<f64>>, MidasError> {
    if k == 0 {
        return Err(MidasError::InvalidLagCount {
            what: "Almon PDL basis",
            k,
            needed: 1,
        });
    }
    if degree + 1 > k {
        return Err(MidasError::InvalidPolynomialDegree { degree, k });
    }
    let mut cols = Vec::with_capacity(degree + 1);
    for j in 0..=degree {
        let col: Vec<f64> = (1..=k).map(|lag| (lag as f64).powi(j as i32)).collect();
        cols.push(col);
    }
    Ok(cols)
}

/// Evaluate and normalize an Almon polynomial lag profile for the coefficient
/// vector `a = [a_0, a_1, ..., a_q]`: `w_k proportional to sum_j a_j k^j`,
/// `k = 1..=K`, normalized so `sum_k w_k = 1`.
///
/// This is the [`almon_pdl_basis`] profile expressed as a weight vector,
/// provided for parity with [`exp_almon_weights`] / [`beta_weights`] when the
/// polynomial coefficients are already known. Because a polynomial can change
/// sign, the raw profile is normalized by the sum of the values (not a
/// softmax); the sum must be finite and non-zero.
///
/// # Errors
///
/// [`MidasError::InvalidLagCount`] if `k == 0`;
/// [`MidasError::InvalidWeightParam`] if `coeffs` is empty or non-finite, or
/// if the polynomial degree exceeds `K - 1`;
/// [`MidasError::DegenerateWeights`] if the profile sums to a non-finite or
/// zero total.
pub fn almon_weights(coeffs: &[f64], k: usize) -> Result<Vec<f64>, MidasError> {
    if k == 0 {
        return Err(MidasError::InvalidLagCount {
            what: "Almon weights",
            k,
            needed: 1,
        });
    }
    if coeffs.is_empty() {
        return Err(MidasError::InvalidWeightParam {
            what: "Almon weights",
            name: "coeffs",
            value: f64::NAN,
            requirement: "at least one polynomial coefficient",
        });
    }
    if coeffs.len() > k {
        return Err(MidasError::InvalidPolynomialDegree {
            degree: coeffs.len() - 1,
            k,
        });
    }
    for (j, &c) in coeffs.iter().enumerate() {
        if !c.is_finite() {
            return Err(MidasError::NonFinite {
                what: "Almon weights coefficients",
                index: j,
                value: c,
            });
        }
    }
    let raw: Vec<f64> = (1..=k)
        .map(|lag| {
            let x = lag as f64;
            // Horner evaluation of sum_j coeffs[j] x^j.
            coeffs.iter().rev().fold(0.0, |acc, &c| acc * x + c)
        })
        .collect();
    let sum: f64 = raw.iter().sum();
    if !sum.is_finite() || sum == 0.0 {
        return Err(MidasError::DegenerateWeights {
            what: "Almon weights",
        });
    }
    Ok(raw.into_iter().map(|v| v / sum).collect())
}

/// Reject a non-finite hyperparameter, and (when `allow_nonpositive` is
/// `false`) a non-positive one.
fn check_param(
    what: &'static str,
    name: &'static str,
    value: f64,
    allow_nonpositive: bool,
) -> Result<(), MidasError> {
    if !value.is_finite() {
        return Err(MidasError::InvalidWeightParam {
            what,
            name,
            value,
            requirement: "a finite value",
        });
    }
    if !allow_nonpositive && value <= 0.0 {
        return Err(MidasError::InvalidWeightParam {
            what,
            name,
            value,
            requirement: "a strictly positive value",
        });
    }
    Ok(())
}

/// Softmax-style normalization of log-weights: shift by the maximum,
/// exponentiate, and divide by the total. Returns
/// [`MidasError::DegenerateWeights`] if the total is not finite and positive.
fn normalize_log_weights(logs: &[f64], what: &'static str) -> Result<Vec<f64>, MidasError> {
    let mut max = f64::NEG_INFINITY;
    for &g in logs {
        if g > max {
            max = g;
        }
    }
    if !max.is_finite() {
        return Err(MidasError::DegenerateWeights { what });
    }
    let exps: Vec<f64> = logs.iter().map(|&g| (g - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    if !sum.is_finite() || sum <= 0.0 {
        return Err(MidasError::DegenerateWeights { what });
    }
    Ok(exps.into_iter().map(|v| v / sum).collect())
}
