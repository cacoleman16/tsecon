//! Shared plumbing: input validation, the OLS-backed auxiliary regression,
//! and the F-distribution survival function.
//!
//! The single owner of the OLS arithmetic is [`tsecon_hac::ols`]; this module
//! only assembles designs, extracts residual sums of squares, and forms
//! centered `R^2` values on top of that fit. Distribution tails come from
//! [`tsecon_stats`] (chi-square) and the regularized incomplete beta function
//! (F). Nothing here reimplements a solver.

use tsecon_hac::{ols, OlsFit};
use tsecon_stats::special::beta_inc;

use crate::error::SpecTestError;

/// Validate a response `y` and design columns `x_cols` (statsmodels exog
/// convention — include the constant column yourself). Returns `(n, k)`.
///
/// Checks, in order: `y` non-empty; at least one design column; every column
/// the same length as `y`; and every value finite. The per-test residual
/// degree-of-freedom requirement (`n > k` etc.) is enforced by each caller.
pub(crate) fn validate(y: &[f64], x_cols: &[Vec<f64>]) -> Result<(usize, usize), SpecTestError> {
    let n = y.len();
    if n == 0 {
        return Err(SpecTestError::EmptyInput { what: "y" });
    }
    if x_cols.is_empty() {
        return Err(SpecTestError::NoRegressors);
    }
    let k = x_cols.len();
    for (j, col) in x_cols.iter().enumerate() {
        if col.len() != n {
            return Err(SpecTestError::DimensionMismatch {
                what: col_name(j),
                expected: n,
                got: col.len(),
            });
        }
    }
    for &v in y.iter() {
        if !v.is_finite() {
            return Err(SpecTestError::NonFinite { what: "y" });
        }
    }
    for col in x_cols.iter() {
        for &v in col.iter() {
            if !v.is_finite() {
                return Err(SpecTestError::NonFinite {
                    what: "a design column",
                });
            }
        }
    }
    Ok((n, k))
}

/// A stable static name for the `j`-th design column (error messages only).
fn col_name(j: usize) -> &'static str {
    match j {
        0 => "design column 0",
        1 => "design column 1",
        2 => "design column 2",
        3 => "design column 3",
        _ => "a design column",
    }
}

/// `true` if some design column is (numerically) constant — the intercept the
/// White / Breusch-Pagan auxiliary regressions need for a centered `R^2`.
pub(crate) fn has_constant(x_cols: &[Vec<f64>]) -> bool {
    x_cols.iter().any(|col| {
        let first = match col.first() {
            Some(&v) => v,
            None => return false,
        };
        col.iter()
            .all(|&v| (v - first).abs() <= first.abs() * 1e-12 + 1e-12)
    })
}

/// Fitted values `yhat_t = y_t - u_t` of an OLS fit.
pub(crate) fn fitted(y: &[f64], fit: &OlsFit) -> Vec<f64> {
    y.iter()
        .zip(fit.residuals.iter())
        .map(|(&yt, &ut)| yt - ut)
        .collect()
}

/// Residual sum of squares of an OLS fit.
pub(crate) fn ssr(fit: &OlsFit) -> f64 {
    fit.residuals.iter().map(|u| u * u).sum()
}

/// The outcome of an auxiliary OLS regression with an intercept: the centered
/// `R^2`, its residual and total sums of squares, and the sample size.
pub(crate) struct AuxFit {
    /// Centered coefficient of determination `1 - SSR/TSS`.
    pub r2: f64,
    /// Residual sum of squares of the auxiliary regression.
    pub ssr: f64,
    /// Total (centered) sum of squares of the auxiliary regressand.
    pub tss: f64,
}

/// Regress `z` on the columns `aux` (which must contain a constant) by OLS and
/// return the centered `R^2` decomposition. `what` names the regression for
/// error messages.
pub(crate) fn aux_regression(
    z: &[f64],
    aux: &[Vec<f64>],
    what: &'static str,
) -> Result<AuxFit, SpecTestError> {
    let n = z.len();
    let m = aux.len();
    if n <= m {
        return Err(SpecTestError::DegreesOfFreedom { what, n, k: m });
    }
    let mean = z.iter().sum::<f64>() / n as f64;
    let tss: f64 = z.iter().map(|&v| (v - mean) * (v - mean)).sum();
    if tss <= 0.0 {
        return Err(SpecTestError::DegenerateResponse { what });
    }
    let fit = ols(z, aux)?;
    let rss = ssr(&fit);
    Ok(AuxFit {
        r2: 1.0 - rss / tss,
        ssr: rss,
        tss,
    })
}

/// Survival function of the F distribution with `d1` numerator and `d2`
/// denominator degrees of freedom,
///
/// ```text
/// SF(x) = I_{d2 / (d2 + d1 x)}(d2 / 2, d1 / 2),
/// ```
///
/// via the regularized incomplete beta function (Abramowitz & Stegun 1964,
/// eq. 26.6.2 with the symmetry `I_x(a, b) = 1 - I_{1-x}(b, a)`). Evaluating
/// the upper tail directly through `beta_inc` avoids the `1 - cdf`
/// cancellation, so p-values of order `1e-8` keep full relative accuracy. This
/// mirrors the private helper in `tsecon-var::causality`.
pub(crate) fn f_sf(x: f64, d1: f64, d2: f64) -> Result<f64, SpecTestError> {
    if x.is_nan() {
        return Err(SpecTestError::NonFinite {
            what: "F statistic",
        });
    }
    if x <= 0.0 {
        return Ok(1.0);
    }
    if x == f64::INFINITY {
        return Ok(0.0);
    }
    Ok(beta_inc(d2 / 2.0, d1 / 2.0, d2 / (d2 + d1 * x))?)
}
