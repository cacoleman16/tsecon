//! The Johansen (1991) cointegration-rank test.
//!
//! Given `k` series that are each integrated of order one, the Johansen
//! procedure tests the rank `r` of the long-run impact matrix
//! `Pi = alpha beta'` in the vector error-correction representation
//!
//! ```text
//! Delta y_t = Pi y_{t-1} + sum_{i=1}^{p-1} Gamma_i Delta y_{t-i} + u_t.
//! ```
//!
//! The test is the reduced-rank regression of Johansen (1988, 1991): after
//! partialling the `k_ar_diff = p - 1` lagged differences out of both
//! `Delta y_t` and `y_{t-1}` (the two auxiliary regressions), the squared
//! sample canonical correlations `lambda_1 > ... > lambda_k` between the
//! two residual sets solve the eigenproblem
//! `S_10 S_00^{-1} S_01 v = lambda S_11 v`. The likelihood-ratio statistics
//!
//! ```text
//! trace(r)   = -T sum_{i=r+1}^{k} ln(1 - lambda_i)
//! max_eig(r) = -T ln(1 - lambda_{r+1})
//! ```
//!
//! test `H0: rank <= r` (trace) and `H0: rank = r` vs `rank = r + 1`
//! (maximum eigenvalue) respectively (Lütkepohl 2005, section 7.2;
//! Johansen 1991).
//!
//! Conventions follow statsmodels 0.14.6 `coint_johansen(det_order = 0,
//! k_ar_diff)` exactly — the constant-in-the-data (`det_order = 0`) case,
//! which demeans every block over the effective sample. The golden fixture
//! `fixtures/coint.json` (`johansen` block) arbitrates the eigenvalues and
//! both statistics.

use tsecon_linalg::faer::{Mat, MatRef};

use crate::critvals::{critical_values, DetOrder};
use crate::error::CointError;
use crate::linalg::{check_finite, demean_columns, partial_out, reduced_rank_eig};

/// Result of the Johansen cointegration-rank test.
///
/// Every vector is indexed by the null rank `r = 0, 1, ..., k - 1`. The
/// statistic at index `r` tests `H0: rank <= r` (trace) or `H0: rank = r`
/// (maximum eigenvalue); reject when the statistic exceeds the critical
/// value and move on to `r + 1`. The cointegration rank is the first `r`
/// whose null is *not* rejected.
#[derive(Debug, Clone)]
pub struct JohansenResult {
    /// Number of series `k`.
    pub neqs: usize,
    /// Effective sample size `T` used in the statistics (rows remaining
    /// after one difference and `k_ar_diff` further presample rows).
    pub nobs: usize,
    /// Number of lagged differences partialled out, `k_ar_diff = p - 1`.
    pub k_ar_diff: usize,
    /// The eigenvalues `lambda_1 > ... > lambda_k` (squared canonical
    /// correlations), decreasing.
    pub eig: Vec<f64>,
    /// `S_11`-orthonormalized eigenvectors, one column per eigenvalue in
    /// the same decreasing order; the first `r` columns span the estimated
    /// cointegrating space under rank `r`.
    pub evec: Mat<f64>,
    /// Trace statistics `trace(r)` for `r = 0, ..., k - 1`.
    pub trace_stat: Vec<f64>,
    /// Maximum-eigenvalue statistics `max_eig(r)` for `r = 0, ..., k - 1`.
    pub max_eig_stat: Vec<f64>,
    /// Trace critical values at the 90%, 95%, and 99% levels, one row per
    /// null rank (MacKinnon-Haug-Michelis 1999, `det_order = 0`).
    pub trace_crit: Vec<[f64; 3]>,
    /// Maximum-eigenvalue critical values at the 90%, 95%, and 99% levels,
    /// one row per null rank.
    pub max_eig_crit: Vec<[f64; 3]>,
}

/// A significance level for reading off the tabulated critical values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignificanceLevel {
    /// 90% (10% size) — column 0 of the critical-value tables.
    Ten,
    /// 95% (5% size) — column 1, the conventional choice.
    Five,
    /// 99% (1% size) — column 2.
    One,
}

impl SignificanceLevel {
    fn column(self) -> usize {
        match self {
            SignificanceLevel::Ten => 0,
            SignificanceLevel::Five => 1,
            SignificanceLevel::One => 2,
        }
    }
}

impl JohansenResult {
    /// The cointegration rank selected by the sequential trace test at
    /// significance level `alpha`: the smallest `r` for which `trace(r)`
    /// does not exceed its critical value. Returns `k` if every null is
    /// rejected (the levels look stationary).
    ///
    /// Returns `None` if any needed critical value is untabulated (`k` or
    /// `k - r` outside `1 ..= 12`).
    pub fn rank_trace(&self, alpha: SignificanceLevel) -> Option<usize> {
        let col = alpha.column();
        for r in 0..self.neqs {
            let cv = self.trace_crit[r][col];
            if !cv.is_finite() {
                return None;
            }
            if self.trace_stat[r] <= cv {
                return Some(r);
            }
        }
        Some(self.neqs)
    }

    /// The cointegration rank selected by the sequential maximum-eigenvalue
    /// test at significance level `alpha`: the smallest `r` for which
    /// `max_eig(r)` does not exceed its critical value.
    ///
    /// Returns `None` if any needed critical value is untabulated.
    pub fn rank_max_eig(&self, alpha: SignificanceLevel) -> Option<usize> {
        let col = alpha.column();
        for r in 0..self.neqs {
            let cv = self.max_eig_crit[r][col];
            if !cv.is_finite() {
                return None;
            }
            if self.max_eig_stat[r] <= cv {
                return Some(r);
            }
        }
        Some(self.neqs)
    }
}

/// Runs the Johansen cointegration-rank test on `endog` (a `T x k` matrix,
/// oldest row first, one column per series) with `k_ar_diff` lagged
/// differences and a constant in the data (statsmodels `det_order = 0`).
///
/// # Errors
///
/// * [`CointError::Dimension`] if `endog` has no columns;
/// * [`CointError::NonFinite`] if `endog` contains a NaN or infinity;
/// * [`CointError::InsufficientObservations`] if the effective sample is
///   too small for the auxiliary regressions;
/// * [`CointError::NotPositiveDefinite`] / [`CointError::Linalg`] if a
///   residual second-moment matrix is singular or the eigensolver fails.
pub fn johansen(endog: MatRef<'_, f64>, k_ar_diff: usize) -> Result<JohansenResult, CointError> {
    let k = endog.ncols();
    if k == 0 {
        return Err(CointError::Dimension {
            what: "endog must have at least one column",
            expected: 1,
            got: 0,
        });
    }
    check_finite(endog, "endog")?;
    let n = endog.nrows();
    // Effective sample: one difference, then k_ar_diff presample rows.
    if n <= k_ar_diff + 1 {
        return Err(CointError::InsufficientObservations {
            needed: k_ar_diff + 2,
            got: n,
        });
    }
    let t = n - 1 - k_ar_diff;
    // Need more usable rows than short-run regressors plus the levels.
    let n_short = k * k_ar_diff;
    if t <= n_short + k {
        return Err(CointError::InsufficientObservations {
            needed: n_short + k + 1,
            got: t,
        });
    }

    // statsmodels detrends the level data by a constant up front
    // (det_order = 0). Differences then annihilate that constant, so the
    // per-block demeaning below is what actually pins the moments, but we
    // replicate the sequence exactly.
    let mut e0 = endog.to_owned();
    demean_columns(&mut e0);

    // First differences dx[i] = e0[i + 1] - e0[i], i = 0 .. n - 2.
    let dx = Mat::from_fn(n - 1, k, |i, j| e0[(i + 1, j)] - e0[(i, j)]);

    // Effective differences dx[k_ar_diff ..], the response of the
    // short-run regression; lagged-difference regressors z; and the lagged
    // level lx = e0[1 .. n - k_ar_diff].
    let mut dx_eff = Mat::from_fn(t, k, |i, j| dx[(k_ar_diff + i, j)]);
    let mut z = Mat::<f64>::zeros(t, n_short);
    for i in 0..t {
        for lag in 1..=k_ar_diff {
            for j in 0..k {
                z[(i, (lag - 1) * k + j)] = dx[(k_ar_diff + i - lag, j)];
            }
        }
    }
    let mut lx = Mat::from_fn(t, k, |i, j| e0[(1 + i, j)]);

    demean_columns(&mut dx_eff);
    demean_columns(&mut z);
    demean_columns(&mut lx);

    // Auxiliary-regression residuals (partial the lagged differences out).
    let r0 = partial_out(dx_eff.as_ref(), z.as_ref()); // differences
    let r1 = partial_out(lx.as_ref(), z.as_ref()); // lagged levels

    let tf = t as f64;
    let s00 = Mat::from_fn(k, k, |i, j| dot_cols(r0.as_ref(), r0.as_ref(), i, j) / tf);
    let s01 = Mat::from_fn(k, k, |i, j| dot_cols(r0.as_ref(), r1.as_ref(), i, j) / tf);
    let s11 = Mat::from_fn(k, k, |i, j| dot_cols(r1.as_ref(), r1.as_ref(), i, j) / tf);

    let (eig, evec) = reduced_rank_eig(s00.as_ref(), s01.as_ref(), s11.as_ref())?;

    // Likelihood-ratio statistics (Johansen 1991; Lütkepohl 2005 eq. 7.2.13).
    let mut trace_stat = vec![0.0; k];
    let mut max_eig_stat = vec![0.0; k];
    for r in 0..k {
        max_eig_stat[r] = -tf * (1.0 - eig[r]).ln();
        let acc: f64 = eig[r..k].iter().map(|&lam| (1.0 - lam).ln()).sum();
        trace_stat[r] = -tf * acc;
    }

    // Critical values: MacKinnon-Haug-Michelis, det_order = 0. Row for the
    // null rank r uses the "k - r common trends" table row.
    let mut trace_crit = Vec::with_capacity(k);
    let mut max_eig_crit = Vec::with_capacity(k);
    for r in 0..k {
        let (tr, mx) = critical_values(DetOrder::Constant, k - r);
        trace_crit.push(tr);
        max_eig_crit.push(mx);
    }

    Ok(JohansenResult {
        neqs: k,
        nobs: t,
        k_ar_diff,
        eig,
        evec,
        trace_stat,
        max_eig_stat,
        trace_crit,
        max_eig_crit,
    })
}

/// Inner product of column `a` of `x` with column `b` of `y` (both have
/// the same number of rows).
fn dot_cols(x: MatRef<'_, f64>, y: MatRef<'_, f64>, a: usize, b: usize) -> f64 {
    let mut s = 0.0;
    for i in 0..x.nrows() {
        s += x[(i, a)] * y[(i, b)];
    }
    s
}
