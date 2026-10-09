//! Durbin-Watson (1950, 1951) test for first-order serial correlation in the
//! errors of `y = X b + u`, with the EXACT p-value under normal errors.
//!
//! The statistic is computed from the OLS residuals `e`:
//!
//! ```text
//! d = sum_{t=2}^T (e_t - e_{t-1})^2 / sum_{t=1}^T e_t^2  =  e'A e / e'e,
//! ```
//!
//! with `A = D'D` (`D` the `(T-1) x T` first-difference matrix); `d` is near
//! `2 (1 - r_1)`, so `d < 2` signals positive and `d > 2` negative
//! first-order autocorrelation. This is statsmodels' `durbin_watson`.
//!
//! **Exact null distribution.** Under `H0: u ~ N(0, s^2 I)` with `X` fixed,
//! `e = M u` (`M = I - X (X'X)^{-1} X'`), and with `nu_1..nu_{T-k}` the
//! eigenvalues of `M A M` on the orthogonal complement of `col(X)`,
//!
//! ```text
//! P(d <= c) = P( sum_i (nu_i - c) z_i^2 <= 0 ),   z_i iid N(0, 1),
//! ```
//!
//! evaluated by Imhof's (1961) inversion formula (see [`crate::imhof`]). The
//! eigenvalues come from one dense symmetric eigendecomposition: with `Q` an
//! orthonormal basis of `col(X)` (Householder QR, `faer`) the matrix
//! `B = M A M + 8 Q Q'` has the `nu_i` as its `T - k` smallest eigenvalues
//! and `8` (above every `nu_i`, which are at most `4`) `k` times. The cost
//! is CUBIC in `T`, which is why [`DW_EXACT_MAX_NOBS`] caps the sample for
//! the p-value; [`durbin_watson_regression_statistic`] computes `d` alone in
//! time linear in `T`, with no cap.
//!
//! The exact p-value is exact for normal errors and strictly exogenous
//! regressors; it is invalid with a lagged dependent variable in `X` (use
//! the Breusch-Godfrey test) and approximate for non-normal errors.

use tsecon_hac::ols;
use tsecon_linalg::faer::linalg::matmul::matmul;
use tsecon_linalg::faer::{Accum, Mat, Par, Side};

use crate::common::{pow2_normalized, validate};
use crate::error::SpecTestError;
use crate::imhof::prob_le_zero;

/// Largest sample for which the exact p-value is computed (the statistic
/// itself has no cap; see [`durbin_watson_regression_statistic`]): the dense
/// `T x T` eigendecomposition is cubic in `T` (and holds a `T x T` matrix
/// in memory, 128 MB at this cap).
pub const DW_EXACT_MAX_NOBS: usize = 4000;

/// Shift placed on `col(X)` so its `k` eigenvalues sit above every `nu_i`.
const SHIFT: f64 = 8.0;

/// The alternative hypothesis of the Durbin-Watson test.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DwAlternative {
    /// Positive first-order autocorrelation (`rho > 0`): the classical
    /// one-sided test, `p = P(d <= d_obs)`.
    Positive,
    /// Negative first-order autocorrelation (`rho < 0`): `p = P(d >= d_obs)`.
    Negative,
    /// Either sign: `p = min(1, 2 min(P(d <= d_obs), P(d >= d_obs)))`.
    TwoSided,
}

impl DwAlternative {
    /// The option string the bindings accept for this alternative.
    pub fn name(self) -> &'static str {
        match self {
            Self::Positive => "positive",
            Self::Negative => "negative",
            Self::TwoSided => "two_sided",
        }
    }
}

/// Outcome of the Durbin-Watson test.
#[derive(Debug, Clone, PartialEq)]
pub struct DurbinWatsonTest {
    /// The Durbin-Watson statistic `d`.
    pub statistic: f64,
    /// The exact p-value for [`Self::alternative`].
    pub pvalue: f64,
    /// The alternative the p-value refers to.
    pub alternative: DwAlternative,
    /// `P(d <= d_obs)` under the null (the `Positive` p-value).
    pub pvalue_positive: f64,
    /// `P(d >= d_obs)` under the null (the `Negative` p-value).
    pub pvalue_negative: f64,
    /// Number of observations `T`.
    pub nobs: usize,
    /// Residual degrees of freedom `T - k` (the number of `nu_i`).
    pub df_resid: usize,
}

/// The Durbin-Watson statistic of a residual series,
/// `sum (e_t - e_{t-1})^2 / sum e_t^2` (statsmodels `durbin_watson`).
///
/// # Errors
///
/// [`SpecTestError::EmptyInput`] for fewer than two residuals,
/// [`SpecTestError::NonFinite`] for a NaN/inf entry, and
/// [`SpecTestError::DegenerateResponse`] when every residual is zero.
pub fn durbin_watson_statistic(residuals: &[f64]) -> Result<f64, SpecTestError> {
    if residuals.len() < 2 {
        return Err(SpecTestError::EmptyInput {
            what: "residuals (the Durbin-Watson statistic needs at least two)",
        });
    }
    if residuals.iter().any(|v| !v.is_finite()) {
        return Err(SpecTestError::NonFinite { what: "residuals" });
    }
    let den: f64 = residuals.iter().map(|v| v * v).sum();
    if den == 0.0 {
        return Err(SpecTestError::DegenerateResponse {
            what: "Durbin-Watson statistic (every residual is zero)",
        });
    }
    let num: f64 = residuals
        .windows(2)
        .map(|w| (w[1] - w[0]) * (w[1] - w[0]))
        .sum();
    Ok(num / den)
}

/// The `T - k` eigenvalues `nu_1 <= ... <= nu_{T-k}` of `M A M` on the
/// orthogonal complement of `col(X)` — the coefficients of the Durbin-Watson
/// null distribution. `x_cols` must have full column rank (checked by the
/// caller's OLS fit).
pub(crate) fn dw_null_eigenvalues(
    x_cols: &[Vec<f64>],
    n: usize,
) -> Result<Vec<f64>, SpecTestError> {
    let k = x_cols.len();
    let x = Mat::from_fn(n, k, |i, j| x_cols[j][i]);
    let q = x.qr().compute_thin_Q();
    // W = A Q, with A = D'D tridiagonal: diagonal (1, 2, ..., 2, 1), off -1.
    let w = Mat::from_fn(n, k, |t, j| {
        let diag = if t == 0 || t + 1 == n { 1.0 } else { 2.0 };
        let mut v = diag * q[(t, j)];
        if t > 0 {
            v -= q[(t - 1, j)];
        }
        if t + 1 < n {
            v -= q[(t + 1, j)];
        }
        v
    });
    // G = Q'AQ (k x k), C = Q (G + 8 I) - W, and
    // B = M A M + 8 Q Q' = A + C Q' - Q W'.
    let mut g = q.transpose() * &w;
    for i in 0..k {
        g[(i, i)] += SHIFT;
    }
    let c = &q * &g - &w;
    let mut b = Mat::<f64>::zeros(n, n);
    for t in 0..n {
        b[(t, t)] = if t == 0 || t + 1 == n { 1.0 } else { 2.0 };
        if t + 1 < n {
            b[(t + 1, t)] = -1.0;
            b[(t, t + 1)] = -1.0;
        }
    }
    matmul(
        b.as_mut(),
        Accum::Add,
        c.as_ref(),
        q.transpose(),
        1.0,
        Par::Seq,
    );
    matmul(
        b.as_mut(),
        Accum::Add,
        q.as_ref(),
        w.transpose(),
        -1.0,
        Par::Seq,
    );

    let ev =
        b.self_adjoint_eigenvalues(Side::Lower)
            .map_err(|_| SpecTestError::NumericalBreakdown {
                what: "eigenvalues of the Durbin-Watson null matrix M A M",
            })?;
    if ev.len() != n {
        return Err(SpecTestError::NumericalBreakdown {
            what: "eigenvalues of the Durbin-Watson null matrix M A M",
        });
    }
    let m = n - k;
    // Sanity: the k shifted eigenvalues sit at 8, every nu_i in [0, 4].
    let shifted_ok = ev[m..].iter().all(|v| (v - SHIFT).abs() <= 1e-8);
    let nu_ok = ev[..m].iter().all(|v| (-1e-8..=4.0 + 1e-8).contains(v));
    if !shifted_ok || !nu_ok {
        return Err(SpecTestError::NumericalBreakdown {
            what: "eigenvalues of the Durbin-Watson null matrix M A M (the design is \
                   numerically rank deficient)",
        });
    }
    Ok(ev[..m].iter().map(|v| v.clamp(0.0, 4.0)).collect())
}

/// Validate the regression inputs and the Durbin-Watson degrees of freedom,
/// returning `(T, k)`.
fn checked_dims(y: &[f64], x_cols: &[Vec<f64>]) -> Result<(usize, usize), SpecTestError> {
    let (n, k) = validate(y, x_cols)?;
    if n < k.saturating_add(2) {
        return Err(SpecTestError::DegreesOfFreedom {
            what: "Durbin-Watson test, which needs one observation more than the \
                   regression (n >= k + 2: with n = k + 1 the statistic is a constant; the \
                   count below is the number of design columns plus that one)",
            n,
            k: k + 1,
        });
    }
    Ok((n, k))
}

/// The statistic on the OLS fit, and the normalized design the null
/// eigenvalues are computed from.
fn statistic_on_fit(y: &[f64], x_cols: &[Vec<f64>]) -> Result<(f64, Vec<Vec<f64>>), SpecTestError> {
    // d and its null distribution are invariant to rescaling y or a design
    // column; the exact power-of-two normalization keeps the sums of squares
    // of extreme-scale data from overflowing or underflowing.
    let y = pow2_normalized(y);
    let x_cols: Vec<Vec<f64>> = x_cols.iter().map(|c| pow2_normalized(c)).collect();
    let fit = ols(&y, &x_cols)?;
    let u = fit.residuals;
    let u_sq: f64 = u.iter().map(|v| v * v).sum();
    let y_sq: f64 = y.iter().map(|v| v * v).sum();
    if u_sq == 0.0 || u_sq <= 1e-26 * y_sq {
        return Err(SpecTestError::DegenerateResponse {
            what: "Durbin-Watson test (the OLS residuals are zero: y is an exact linear \
                   combination of the design columns)",
        });
    }
    Ok((durbin_watson_statistic(&u)?, x_cols))
}

/// The Durbin-Watson statistic of the OLS regression of `y` on `x_cols`
/// (statsmodels exog convention), without its null distribution: linear in
/// `T`, so unlike [`durbin_watson_test`] it has no sample cap. Returns
/// `(d, T, T - k)`; `d` is bit-identical to [`DurbinWatsonTest::statistic`].
///
/// # Errors
///
/// As [`durbin_watson_test`], except that there is no
/// [`SpecTestError::SampleTooLong`] and no eigendecomposition to fail.
pub fn durbin_watson_regression_statistic(
    y: &[f64],
    x_cols: &[Vec<f64>],
) -> Result<(f64, usize, usize), SpecTestError> {
    let (n, k) = checked_dims(y, x_cols)?;
    let (statistic, _) = statistic_on_fit(y, x_cols)?;
    Ok((statistic, n, n - k))
}

/// Durbin-Watson test on the OLS regression of `y` on `x_cols` (statsmodels
/// exog convention: include the constant column yourself), with the exact
/// p-value for `alternative` under normal errors and fixed regressors.
///
/// # Errors
///
/// [`SpecTestError::DegreesOfFreedom`] unless `T >= k + 2` (with `T = k + 1`
/// the residual lives on a line and `d` is a constant);
/// [`SpecTestError::SampleTooLong`] if `T` exceeds [`DW_EXACT_MAX_NOBS`];
/// [`SpecTestError::DegenerateResponse`] if the residuals are (numerically)
/// all zero; the usual empty/dimension/finite validation errors;
/// [`SpecTestError::SingularDesign`] for a collinear design; and
/// [`SpecTestError::NumericalBreakdown`] if the eigendecomposition or the
/// Imhof quadrature fails.
pub fn durbin_watson_test(
    y: &[f64],
    x_cols: &[Vec<f64>],
    alternative: DwAlternative,
) -> Result<DurbinWatsonTest, SpecTestError> {
    let (n, k) = checked_dims(y, x_cols)?;
    if n > DW_EXACT_MAX_NOBS {
        return Err(SpecTestError::SampleTooLong {
            what: "Durbin-Watson exact p-value",
            n,
            max: DW_EXACT_MAX_NOBS,
        });
    }
    let (statistic, x_cols) = statistic_on_fit(y, x_cols)?;
    let nu = dw_null_eigenvalues(&x_cols, n)?;

    let spread = nu[nu.len() - 1] - nu[0];
    let (p_le, p_ge) = if spread <= 1e-12 {
        // The null distribution is a point mass at d_obs: no evidence either way.
        (1.0, 1.0)
    } else {
        let lam: Vec<f64> = nu.iter().map(|v| v - statistic).collect();
        let p_le = prob_le_zero(&lam)?;
        (p_le, 1.0 - p_le)
    };
    let pvalue = match alternative {
        DwAlternative::Positive => p_le,
        DwAlternative::Negative => p_ge,
        DwAlternative::TwoSided => (2.0 * p_le.min(p_ge)).min(1.0),
    };
    Ok(DurbinWatsonTest {
        statistic,
        pvalue,
        alternative,
        pvalue_positive: p_le,
        pvalue_negative: p_ge,
        nobs: n,
        df_resid: n - k,
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    /// Eigenvalues of the first-difference matrix `A = D'D`, ascending:
    /// `2 - 2 cos(pi j / n)`, `j = 0..n-1`.
    fn a_eigenvalues(n: usize) -> Vec<f64> {
        (0..n)
            .map(|j| 2.0 - 2.0 * (std::f64::consts::PI * j as f64 / n as f64).cos())
            .collect()
    }

    fn design(n: usize, k: usize) -> Vec<Vec<f64>> {
        let mut cols = vec![vec![1.0; n]];
        for j in 1..k {
            cols.push(
                (0..n)
                    .map(|t| ((0.31 * j as f64 + 0.07) * (t * t + 3 * j) as f64).sin())
                    .collect(),
            );
        }
        cols
    }

    #[test]
    fn constant_only_design_has_closed_form_eigenvalues() {
        // M projects out exactly the eigenvector of A's zero eigenvalue, so
        // the nu_i are A's remaining eigenvalues.
        for n in [3usize, 10, 57] {
            let nu = dw_null_eigenvalues(&[vec![1.0; n]], n).expect("eigs");
            let lam = a_eigenvalues(n);
            assert_eq!(nu.len(), n - 1);
            for (a, b) in nu.iter().zip(lam[1..].iter()) {
                assert!((a - b).abs() < 1e-12, "n={n}: {a} vs {b}");
            }
        }
    }

    #[test]
    fn eigenvalues_obey_the_durbin_watson_interlacing_bounds() {
        // With the constant in X (k columns), lam_{i+1} <= nu_i <= lam_{i+k}
        // (Cauchy interlacing after projecting out the constant) — the
        // theorem behind the d_L / d_U bounds tables.
        for (n, k) in [(20usize, 2usize), (35, 3), (60, 5)] {
            let x = design(n, k);
            let nu = dw_null_eigenvalues(&x, n).expect("eigs");
            let lam = a_eigenvalues(n);
            assert_eq!(nu.len(), n - k);
            for (i, v) in nu.iter().enumerate() {
                assert!(
                    *v >= lam[i + 1] - 1e-10,
                    "n={n} k={k} i={i}: {v} < {}",
                    lam[i + 1]
                );
                assert!(
                    *v <= lam[i + k] + 1e-10,
                    "n={n} k={k} i={i}: {v} > {}",
                    lam[i + k]
                );
            }
        }
    }

    #[test]
    fn exact_cdf_lies_between_the_bounds_distributions() {
        // P(d_U <= c) <= P(d <= c) <= P(d_L <= c) at every c, with d_L, d_U
        // built from A's closed-form eigenvalues.
        let (n, k) = (30usize, 3usize);
        let x = design(n, k);
        let nu = dw_null_eigenvalues(&x, n).expect("eigs");
        let lam = a_eigenvalues(n);
        let lower: Vec<f64> = (0..n - k).map(|i| lam[i + 1]).collect();
        let upper: Vec<f64> = (0..n - k).map(|i| lam[i + k]).collect();
        for c in [1.0, 1.4, 1.8, 2.0, 2.4, 3.0] {
            let shift = |v: &[f64]| v.iter().map(|a| a - c).collect::<Vec<f64>>();
            let p = prob_le_zero(&shift(&nu)).expect("p");
            let p_l = prob_le_zero(&shift(&lower)).expect("pl");
            let p_u = prob_le_zero(&shift(&upper)).expect("pu");
            assert!(
                p_u <= p + 1e-12 && p <= p_l + 1e-12,
                "c={c}: {p_u} <= {p} <= {p_l}"
            );
        }
    }

    #[test]
    fn three_observation_mean_model_has_a_closed_form_pvalue() {
        // n = 3, X = ones: nu = (1, 3), so P(d <= c) = (2/pi) atan(sqrt((c-1)/(3-c))).
        for y in [[0.3, -1.1, 0.9], [1.0, 2.0, 2.5], [-0.2, 0.4, 0.1]] {
            let r = durbin_watson_test(&y, &[vec![1.0; 3]], DwAlternative::Positive).expect("dw");
            let c = r.statistic;
            let exact = 2.0 / std::f64::consts::PI * ((c - 1.0) / (3.0 - c)).sqrt().atan();
            assert!(
                (r.pvalue - exact).abs() < 1e-12,
                "y={y:?}: {} vs {exact}",
                r.pvalue
            );
        }
    }
}
