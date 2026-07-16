//! Hamilton (2018) regression filter — the proposed replacement for the
//! Hodrick-Prescott filter — and its random-walk special case.

use crate::decomposition::{Alignment, Decomposition};
use crate::error::{check_finite, FiltersError};
use crate::hp::Frequency;

/// Result of the Hamilton (2018) regression filter: the OLS coefficients
/// together with the trend (fitted values) / cycle (residuals)
/// decomposition.
#[derive(Debug, Clone, PartialEq)]
pub struct HamiltonResult {
    /// OLS coefficients `[intercept, b_1, ..., b_p]` on
    /// `[1, y_{t-h}, y_{t-h-1}, ..., y_{t-h-p+1}]`.
    pub beta: Vec<f64>,
    /// Fitted values (`trend`) and residuals (`cycle`), aligned to input
    /// observations `h + p - 1, ..., n - 1`
    /// (`alignment.lost_start = h + p - 1`).
    pub decomposition: Decomposition,
}

/// Hamilton's recommended `(h, p)` defaults by sampling frequency: the
/// horizon `h` spans two years and `p` one year of lags — `(2, 1)`
/// annual, `(8, 4)` quarterly, `(24, 12)` monthly (Hamilton 2018,
/// section 4).
pub fn hamilton_defaults(freq: Frequency) -> (usize, usize) {
    match freq {
        Frequency::Annual => (2, 1),
        Frequency::Quarterly => (8, 4),
        Frequency::Monthly => (24, 12),
    }
}

/// Hamilton (2018) regression filter.
///
/// Regresses `y_{t}` on a constant and `p` lags of the series dated `h`
/// periods earlier,
///
/// ```text
/// y_t = beta_0 + beta_1 y_{t-h} + beta_2 y_{t-h-1} + ...
///              + beta_p y_{t-h-p+1} + v_t,
/// ```
///
/// estimated by OLS over `t = h + p - 1, ..., n - 1` (0-indexed). The
/// cycle is the residual `v_t` and the trend the fitted value, so
/// `trend + cycle == y` on the aligned range. The first `h + p - 1`
/// observations are lost (`alignment.lost_start = h + p - 1`,
/// `lost_end = 0`).
///
/// Quarterly defaults are `h = 8`, `p = 4` (see [`hamilton_defaults`]).
/// The regression is solved by Householder QR (numerically stable for
/// the highly collinear lag columns; never normal equations). A constant
/// series makes the lag columns collinear with the intercept and returns
/// [`FiltersError::RankDeficient`].
///
/// Reference: Hamilton (2018), "Why You Should Never Use the
/// Hodrick-Prescott Filter", *Review of Economics and Statistics*
/// 100(5).
pub fn hamilton_filter(y: &[f64], h: usize, p: usize) -> Result<HamiltonResult, FiltersError> {
    if h == 0 {
        return Err(FiltersError::InvalidParameter {
            name: "h",
            value: 0.0,
            requirement: "a horizon >= 1",
        });
    }
    if p == 0 {
        return Err(FiltersError::InvalidParameter {
            name: "p",
            value: 0.0,
            requirement: "a lag count >= 1",
        });
    }
    let n = y.len();
    let lost = h + p - 1;
    // Need at least p + 1 regression rows for the p + 1 coefficients.
    let needed = lost + p + 1;
    if n < needed {
        return Err(FiltersError::SeriesTooShort {
            filter: "hamilton_filter",
            needed,
            got: n,
        });
    }
    check_finite(y)?;

    let m = n - lost; // regression rows, one per t = lost..n-1
    let k = p + 1; // intercept + p lags

    // Design matrix in column-major storage: [1, y_{t-h}, ..., y_{t-h-p+1}].
    let mut cols: Vec<Vec<f64>> = Vec::with_capacity(k);
    cols.push(vec![1.0; m]);
    for j in 0..p {
        cols.push((lost..n).map(|t| y[t - h - j]).collect());
    }
    let rhs: Vec<f64> = y[lost..].to_vec();

    let beta = householder_lstsq(cols, rhs, "hamilton_filter OLS")?;

    let mut trend = Vec::with_capacity(m);
    let mut cycle = Vec::with_capacity(m);
    for t in lost..n {
        let mut fit = beta[0];
        for j in 0..p {
            fit += beta[j + 1] * y[t - h - j];
        }
        trend.push(fit);
        cycle.push(y[t] - fit);
    }

    Ok(HamiltonResult {
        beta,
        decomposition: Decomposition {
            trend: Some(trend),
            cycle,
            alignment: Alignment {
                lost_start: lost,
                lost_end: 0,
                input_len: n,
            },
        },
    })
}

/// Random-walk special case of the Hamilton (2018) filter.
///
/// When the series is a random walk (with drift), the population
/// coefficients of the [`hamilton_filter`] regression are
/// `beta_1 = 1` and `beta_0 = beta_2 = ... = beta_p = 0`, so the filter
/// reduces to the `h`-period difference
///
/// ```text
/// cycle_t = y_t - y_{t-h},    trend_t = y_{t-h},
/// ```
///
/// for `t = h, ..., n - 1` (`alignment.lost_start = h`, `lost_end = 0`;
/// `trend + cycle == y` exactly on the aligned range). Hamilton (2018,
/// section 6) recommends this variant when the regression sample is
/// short.
pub fn hamilton_filter_random_walk(y: &[f64], h: usize) -> Result<Decomposition, FiltersError> {
    if h == 0 {
        return Err(FiltersError::InvalidParameter {
            name: "h",
            value: 0.0,
            requirement: "a horizon >= 1",
        });
    }
    let n = y.len();
    if n < h + 1 {
        return Err(FiltersError::SeriesTooShort {
            filter: "hamilton_filter_random_walk",
            needed: h + 1,
            got: n,
        });
    }
    check_finite(y)?;

    let trend: Vec<f64> = y[..n - h].to_vec();
    let cycle: Vec<f64> = (h..n).map(|t| y[t] - y[t - h]).collect();
    Ok(Decomposition {
        trend: Some(trend),
        cycle,
        alignment: Alignment {
            lost_start: h,
            lost_end: 0,
            input_len: n,
        },
    })
}

/// Least squares `min_beta ||A beta - b||_2` by Householder QR without
/// pivoting (Golub & Van Loan 2013, algorithm 5.2.1).
///
/// `cols` holds the columns of `A` (each of length `m = b.len()`); the
/// factorization overwrites them. Rank deficiency is detected by
/// comparing each diagonal of `R` against a scaled tolerance
/// `m * eps * max_j ||a_j||`.
fn householder_lstsq(
    mut cols: Vec<Vec<f64>>,
    mut b: Vec<f64>,
    what: &'static str,
) -> Result<Vec<f64>, FiltersError> {
    let k = cols.len();
    let m = b.len();
    debug_assert!(m >= k);

    let max_colnorm = cols
        .iter()
        .map(|c| c.iter().map(|v| v * v).sum::<f64>().sqrt())
        .fold(0.0_f64, f64::max);
    let tol = m as f64 * f64::EPSILON * max_colnorm;

    let mut v = vec![0.0_f64; m];
    for j in 0..k {
        // Householder vector annihilating rows j+1.. of column j.
        let norm = cols[j][j..].iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm <= tol {
            return Err(FiltersError::RankDeficient { what });
        }
        let alpha = if cols[j][j] >= 0.0 { -norm } else { norm };
        v[j] = cols[j][j] - alpha;
        v[(j + 1)..m].copy_from_slice(&cols[j][(j + 1)..m]);
        let vtv: f64 = v[j..].iter().map(|x| x * x).sum();
        cols[j][j] = alpha; // R[j][j]
        for x in &mut cols[j][(j + 1)..] {
            *x = 0.0;
        }
        // Reflect the remaining columns and the right-hand side:
        // c <- c - 2 v (v'c) / (v'v).
        for col in cols.iter_mut().skip(j + 1) {
            let dot: f64 = v[j..].iter().zip(&col[j..]).map(|(a, c)| a * c).sum();
            let fac = 2.0 * dot / vtv;
            for (ci, vi) in col[j..].iter_mut().zip(&v[j..]) {
                *ci -= fac * vi;
            }
        }
        let dot: f64 = v[j..].iter().zip(&b[j..]).map(|(a, c)| a * c).sum();
        let fac = 2.0 * dot / vtv;
        for (bi, vi) in b[j..].iter_mut().zip(&v[j..]) {
            *bi -= fac * vi;
        }
    }

    // Back substitution R beta = (Q'b)[..k]; R[i][j] = cols[j][i], j >= i.
    let mut beta = vec![0.0_f64; k];
    for i in (0..k).rev() {
        let mut s = b[i];
        for j in (i + 1)..k {
            s -= cols[j][i] * beta[j];
        }
        let rii = cols[i][i];
        if rii.abs() <= tol {
            return Err(FiltersError::RankDeficient { what });
        }
        beta[i] = s / rii;
    }
    Ok(beta)
}
