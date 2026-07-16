//! Sample autocorrelation and partial autocorrelation functions.
//!
//! Conventions follow statsmodels (`statsmodels.tsa.stattools.acf` /
//! `pacf`), which this crate's golden fixtures were generated with:
//!
//! * the series is demeaned once with the full-sample mean;
//! * the "unadjusted" (biased) autocovariance divides every lag by `n`,
//!   which guarantees a positive-semidefinite autocovariance sequence;
//! * the "adjusted" (unbiased-denominator) variant divides lag `k` by
//!   `n - k` and can produce autocorrelations exceeding one in modulus;
//! * `pacf_yw` matches statsmodels `method="ywm"` (Yule-Walker with the
//!   `n`-denominator "mle" autocovariance), and `pacf_ols` matches
//!   `method="ols"` (successive lag regressions with an intercept).

use crate::error::DiagError;
use crate::validate::check_series;

/// Sample autocorrelation function together with Bartlett standard errors.
///
/// Both vectors have length `nlags + 1` and are indexed by lag, so
/// `acf[0] == 1` and `bartlett_se[0] == 0`.
#[derive(Debug, Clone, PartialEq)]
pub struct AcfResult {
    /// Autocorrelations `r_0, r_1, ..., r_nlags` with `r_0 = 1`.
    pub acf: Vec<f64>,
    /// Bartlett standard errors of the autocorrelations (Bartlett 1946;
    /// Brockwell & Davis 1991, §7.2):
    /// `se(r_k) = sqrt((1 + 2 * sum_{j=1}^{k-1} r_j^2) / n)` for `k >= 2`,
    /// `se(r_1) = 1/sqrt(n)`, `se(r_0) = 0`.
    ///
    /// This is the variance of `r_k` under an MA(k-1) null (not under white
    /// noise); `±z_{alpha/2} * se` gives the usual widening confidence
    /// bands drawn on ACF plots (statsmodels `acf(..., bartlett_confint=
    /// True)`).
    pub bartlett_se: Vec<f64>,
}

/// Sample autocorrelation function with Bartlett standard-error bands.
///
/// The series is demeaned once with the full-sample mean `m`, and
///
/// ```text
/// c_k = (1/d_k) * sum_{t=0}^{n-k-1} (y_t - m)(y_{t+k} - m),
/// r_k = c_k / c_0,
/// ```
///
/// where `d_k = n` when `adjusted == false` (the default in statsmodels
/// and R; guarantees a positive-semidefinite sequence) and `d_k = n - k`
/// when `adjusted == true` (unbiased denominator; `c_0` still uses `n`, so
/// `r_k^adj = n/(n-k) * r_k`). Reference: Box & Jenkins (1976);
/// Brockwell & Davis (1991), §7.2.
///
/// # Errors
///
/// * [`DiagError::NonFinite`] if the series contains NaN or infinities.
/// * [`DiagError::SeriesTooShort`] if `n < 2`.
/// * [`DiagError::InvalidLags`] unless `1 <= nlags <= n - 1`.
/// * [`DiagError::ConstantSeries`] if the sample variance is zero.
pub fn acf(y: &[f64], nlags: usize, adjusted: bool) -> Result<AcfResult, DiagError> {
    let n = check_series(y, 2, "acf")?;
    if nlags == 0 || nlags > n - 1 {
        return Err(DiagError::InvalidLags {
            what: "acf",
            nlags,
            n,
            requirement: "1 <= nlags <= n - 1",
        });
    }
    let r = autocorrelations(y, nlags, adjusted, "acf")?;

    // Bartlett standard errors (see `AcfResult::bartlett_se`).
    let nf = n as f64;
    let mut se = vec![0.0_f64; nlags + 1];
    se[1] = (1.0 / nf).sqrt();
    let mut cum = 0.0;
    for k in 2..=nlags {
        cum += r[k - 1] * r[k - 1];
        se[k] = ((1.0 + 2.0 * cum) / nf).sqrt();
    }

    Ok(AcfResult {
        acf: r,
        bartlett_se: se,
    })
}

/// Partial autocorrelation function via Yule-Walker with the biased
/// (`n`-denominator) autocovariance — the statsmodels `method="ywm"`
/// ("Yule-Walker mle") convention.
///
/// `pacf[k]` is the lag-`k` coefficient `phi_{kk}` of the AR(`k`) model
/// fitted by solving the Yule-Walker equations, computed for all orders at
/// once with the Durbin-Levinson recursion (Durbin 1960; Brockwell & Davis
/// 1991, §8.2). The biased autocovariance keeps the implied Toeplitz matrix
/// positive semidefinite, which guarantees `|pacf[k]| <= 1` (the adjusted
/// variant does not — the reason statsmodels changed its default).
///
/// Returns a vector of length `nlags + 1` with `pacf[0] = 1`.
///
/// # Errors
///
/// As [`acf`], except the lag constraint is `1 <= nlags < n/2` (mirroring
/// statsmodels: partial correlations beyond half the sample length are not
/// meaningfully estimable), plus [`DiagError::NumericalBreakdown`] if the
/// prediction-error variance in the recursion collapses to zero
/// (numerically degenerate series).
pub fn pacf_yw(y: &[f64], nlags: usize) -> Result<Vec<f64>, DiagError> {
    let n = check_series(y, 2, "pacf_yw")?;
    check_pacf_lags(nlags, n, "pacf_yw")?;
    let r = autocorrelations(y, nlags, false, "pacf_yw")?;
    durbin_levinson_pacf(&r, nlags)
}

/// Partial autocorrelation function via successive OLS lag regressions —
/// the statsmodels `method="ols"` convention.
///
/// For each `k = 1..=nlags`, regress `y_t` on an intercept and
/// `y_{t-1}, ..., y_{t-k}` over `t = k..n-1` (each regression uses all
/// `n - k` available rows — statsmodels' `efficient=True`); `pacf[k]` is
/// the estimated coefficient on `y_{t-k}` (Box & Jenkins 1976, §3.2.5).
/// No bias adjustment is applied (statsmodels `adjusted=False`).
///
/// Unlike [`pacf_yw`], the OLS estimate is not constrained to `[-1, 1]` in
/// small samples.
///
/// Returns a vector of length `nlags + 1` with `pacf[0] = 1`.
///
/// # Errors
///
/// As [`pacf_yw`], plus [`DiagError::SingularDesign`] if a lag regression
/// is collinear.
pub fn pacf_ols(y: &[f64], nlags: usize) -> Result<Vec<f64>, DiagError> {
    let n = check_series(y, 2, "pacf_ols")?;
    check_pacf_lags(nlags, n, "pacf_ols")?;

    let mut pacf = Vec::with_capacity(nlags + 1);
    pacf.push(1.0_f64);
    for k in 1..=nlags {
        // Rows t = k..n-1; column j holds y_{t-j} for j = 1..=k.
        let cols: Vec<Vec<f64>> = (1..=k)
            .map(|j| (k..n).map(|t| y[t - j]).collect())
            .collect();
        let response = &y[k..];
        let fit = crate::ols::ols_with_intercept(&cols, response, "pacf_ols")?;
        pacf.push(fit.slopes[k - 1]);
    }
    Ok(pacf)
}

/// Shared lag validation for the PACF estimators: statsmodels requires
/// `nlags < n // 2` for every `pacf` method.
fn check_pacf_lags(nlags: usize, n: usize, what: &'static str) -> Result<(), DiagError> {
    if nlags == 0 || 2 * nlags >= n {
        return Err(DiagError::InvalidLags {
            what,
            nlags,
            n,
            requirement:
                "1 <= nlags < n/2 (partial correlations beyond half the sample \
                 are not meaningfully estimable)",
        });
    }
    Ok(())
}

/// Sample autocorrelations `r_0..r_nlags` (see [`acf`] for the formula).
/// Assumes `y` and `nlags` were already validated; still checks for the
/// degenerate zero-variance case.
pub(crate) fn autocorrelations(
    y: &[f64],
    nlags: usize,
    adjusted: bool,
    what: &'static str,
) -> Result<Vec<f64>, DiagError> {
    let n = y.len();
    let nf = n as f64;
    let mean = y.iter().sum::<f64>() / nf;
    let c0: f64 = y.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>() / nf;
    if c0 <= 0.0 {
        return Err(DiagError::ConstantSeries { what });
    }
    let mut r = Vec::with_capacity(nlags + 1);
    r.push(1.0);
    for k in 1..=nlags {
        let mut acc = 0.0;
        for t in 0..(n - k) {
            acc += (y[t] - mean) * (y[t + k] - mean);
        }
        let denom = if adjusted { (n - k) as f64 } else { nf };
        r.push((acc / denom) / c0);
    }
    Ok(r)
}

/// Durbin-Levinson recursion on an autocorrelation sequence
/// `r_0 = 1, r_1, ..., r_nlags`, returning the partial autocorrelations
/// `phi_{kk}` for `k = 0..=nlags` (with `phi_00 = 1`).
///
/// Recursion (Brockwell & Davis 1991, Proposition 5.2.1):
///
/// ```text
/// phi_kk   = (r_k - sum_{j=1}^{k-1} phi_{k-1,j} r_{k-j}) / v_{k-1}
/// phi_kj   = phi_{k-1,j} - phi_kk * phi_{k-1,k-j}      (j = 1..k-1)
/// v_k      = v_{k-1} * (1 - phi_kk^2),   v_0 = 1
/// ```
fn durbin_levinson_pacf(r: &[f64], nlags: usize) -> Result<Vec<f64>, DiagError> {
    let mut pacf = vec![1.0_f64; nlags + 1];
    // phi[j] = phi_{k,j} for the current order k (1-indexed by lag j).
    let mut phi = vec![0.0_f64; nlags + 1];
    let mut prev = vec![0.0_f64; nlags + 1];
    let mut v = 1.0_f64;
    for k in 1..=nlags {
        if v <= 0.0 {
            // Positive in exact arithmetic for a PSD autocorrelation
            // sequence; can only fail for numerically degenerate input.
            return Err(DiagError::NumericalBreakdown {
                what: "Durbin-Levinson recursion (pacf_yw)",
            });
        }
        let mut acc = r[k];
        for j in 1..k {
            acc -= prev[j] * r[k - j];
        }
        let alpha = acc / v;
        for j in 1..k {
            phi[j] = prev[j] - alpha * prev[k - j];
        }
        phi[k] = alpha;
        prev[..=k].copy_from_slice(&phi[..=k]);
        v *= 1.0 - alpha * alpha;
        pacf[k] = alpha;
    }
    Ok(pacf)
}
