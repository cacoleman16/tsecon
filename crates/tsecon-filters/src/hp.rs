//! Hodrick-Prescott filter: sparse pentadiagonal solve, Ravn-Uhlig
//! frequency-rule lambda defaults, and the real-time one-sided variant.

use crate::decomposition::{Alignment, Decomposition};
use crate::error::{check_finite, FiltersError};

/// Sampling frequency of a series, used to pick the Ravn-Uhlig
/// smoothing parameter for the Hodrick-Prescott filter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Frequency {
    /// Annual observations (1 per year).
    Annual,
    /// Quarterly observations (4 per year).
    Quarterly,
    /// Monthly observations (12 per year).
    Monthly,
}

impl Frequency {
    /// Observations per year.
    pub fn periods_per_year(self) -> f64 {
        match self {
            Frequency::Annual => 1.0,
            Frequency::Quarterly => 4.0,
            Frequency::Monthly => 12.0,
        }
    }
}

/// Ravn-Uhlig (2002) frequency rule for the Hodrick-Prescott smoothing
/// parameter.
///
/// The rule scales the canonical quarterly value by the fourth power of
/// the observation-frequency ratio,
///
/// ```text
/// lambda(s) = 1600 * (s / 4)^4,    s = observations per year,
/// ```
///
/// giving `6.25` for annual, `1600` for quarterly, and `129600` for
/// monthly data.
///
/// Reference: Ravn & Uhlig (2002), "On Adjusting the Hodrick-Prescott
/// Filter for the Frequency of Observations", *Review of Economics and
/// Statistics* 84(2).
pub fn ravn_uhlig_lambda(freq: Frequency) -> f64 {
    let s = freq.periods_per_year();
    1600.0 * (s / 4.0).powi(4)
}

/// Hodrick-Prescott trend-cycle decomposition.
///
/// The trend `tau` minimizes
///
/// ```text
/// sum_t (y_t - tau_t)^2
///   + lambda * sum_t ((tau_{t+1} - tau_t) - (tau_t - tau_{t-1}))^2,
/// ```
///
/// whose first-order condition is the linear system
/// `(I + lambda * K'K) tau = y` with `K` the `(n-2) x n` second-difference
/// matrix. `I + lambda * K'K` is symmetric positive definite and
/// pentadiagonal, so the system is solved with a banded (bandwidth-2)
/// `L D L'` factorization in `O(n)` time and memory — never a dense
/// inversion. The cycle is `y - tau`, so `cycle + trend` reconstructs the
/// input exactly and the alignment is full sample (nothing lost).
///
/// `lambda` must be finite and non-negative; use
/// [`ravn_uhlig_lambda`] for the frequency-rule defaults (1600 for
/// quarterly data). `lambda = 0` returns `trend = y`. Series of length 1
/// or 2 have no second differences, so the penalty is empty and the
/// trend equals the series.
///
/// References: Hodrick & Prescott (1997), "Postwar U.S. Business Cycles:
/// An Empirical Investigation", *Journal of Money, Credit and Banking*
/// 29(1); Ravn & Uhlig (2002). Users should be aware of the Hamilton
/// (2018) critique, "Why You Should Never Use the Hodrick-Prescott
/// Filter" — see [`hamilton_filter`](crate::hamilton_filter) for the
/// recommended alternative.
pub fn hp_filter(y: &[f64], lambda: f64) -> Result<Decomposition, FiltersError> {
    if !lambda.is_finite() || lambda < 0.0 {
        return Err(FiltersError::InvalidParameter {
            name: "lambda",
            value: lambda,
            requirement: "a finite value >= 0",
        });
    }
    if y.is_empty() {
        return Err(FiltersError::SeriesTooShort {
            filter: "hp_filter",
            needed: 1,
            got: 0,
        });
    }
    check_finite(y)?;

    let trend = solve_hp_trend(y, lambda)?;
    let cycle: Vec<f64> = y.iter().zip(&trend).map(|(yi, ti)| yi - ti).collect();
    Ok(Decomposition {
        trend: Some(trend),
        cycle,
        alignment: Alignment::full(y.len()),
    })
}

/// One-sided (real-time) Hodrick-Prescott filter.
///
/// For each `t`, runs the full two-sided filter [`hp_filter`] on the
/// expanding sample `y[0..=t]` and keeps only the final trend point, so
/// `trend[t]` uses no data after `t`. This is the standard recursive
/// construction of the one-sided HP filter (equivalent to the concurrent
/// Kalman-filter estimate of the local-linear-trend formulation, Stock &
/// Watson 1999, "Forecasting Inflation", *Journal of Monetary Economics*
/// 44(2); see also Meyer-Gohde 2010).
///
/// Cost: each expanding-window solve is `O(t)`, so the whole filter is
/// `O(n^2)` — still trivial for macro sample sizes (a 1000-point series
/// is about half a million flops). A single-pass `O(n)` Kalman
/// implementation is a possible future optimization.
/// The early points are estimated from very short samples and are
/// essentially unfiltered (`trend[0] = y[0]`); practitioners commonly
/// discard a burn-in of the first several observations, which is left to
/// the caller since no data is actually lost. Alignment is full sample.
pub fn hp_filter_one_sided(y: &[f64], lambda: f64) -> Result<Decomposition, FiltersError> {
    if !lambda.is_finite() || lambda < 0.0 {
        return Err(FiltersError::InvalidParameter {
            name: "lambda",
            value: lambda,
            requirement: "a finite value >= 0",
        });
    }
    if y.is_empty() {
        return Err(FiltersError::SeriesTooShort {
            filter: "hp_filter_one_sided",
            needed: 1,
            got: 0,
        });
    }
    check_finite(y)?;

    let n = y.len();
    let mut trend = Vec::with_capacity(n);
    for t in 0..n {
        let tau = solve_hp_trend(&y[..=t], lambda)?;
        // tau has length t + 1 >= 1, so last() is always present; the
        // unwrap-free form keeps the no-panic guarantee explicit.
        let last = tau.last().copied().unwrap_or(y[t]);
        trend.push(last);
    }
    let cycle: Vec<f64> = y.iter().zip(&trend).map(|(yi, ti)| yi - ti).collect();
    Ok(Decomposition {
        trend: Some(trend),
        cycle,
        alignment: Alignment::full(n),
    })
}

/// Solve `(I + lambda * K'K) tau = y` with a pentadiagonal `L D L'`
/// factorization (`K` = second-difference matrix), in `O(n)`.
///
/// The matrix is assembled directly in banded storage by accumulating
/// `lambda * k_r k_r'` over the second-difference rows
/// `k_r = (..., 1, -2, 1, ...)`, which handles the `n < 3` (no penalty
/// rows) and boundary cases uniformly:
///
/// ```text
/// diag:  1+lambda, 1+5lambda, 1+6lambda, ..., 1+6lambda, 1+5lambda, 1+lambda
/// off1:  -2lambda, -4lambda, ..., -4lambda, -2lambda
/// off2:  lambda, ..., lambda
/// ```
///
/// `I + lambda * K'K` is symmetric positive definite for `lambda >= 0`
/// (identity plus a positive semidefinite term), so the factorization
/// cannot break down in exact arithmetic; the pivot check guards against
/// pathological rounding only.
fn solve_hp_trend(y: &[f64], lambda: f64) -> Result<Vec<f64>, FiltersError> {
    let n = y.len();
    // Banded storage: d = main diagonal, e = first subdiagonal
    // (e[i] = A[i+1][i]), f = second subdiagonal (f[i] = A[i+2][i]).
    let mut d = vec![1.0_f64; n];
    let mut e = vec![0.0_f64; n.saturating_sub(1)];
    let mut f = vec![0.0_f64; n.saturating_sub(2)];
    for r in 0..n.saturating_sub(2) {
        // Row r of K has (1, -2, 1) in columns (r, r+1, r+2).
        d[r] += lambda;
        d[r + 1] += 4.0 * lambda;
        d[r + 2] += lambda;
        e[r] += -2.0 * lambda;
        e[r + 1] += -2.0 * lambda;
        f[r] += lambda;
    }

    // A = L D L' with unit-lower-triangular L of bandwidth 2:
    //   D[i]  = d[i] - l1[i-1]^2 D[i-1] - l2[i-2]^2 D[i-2]
    //   l1[i] = (e[i] - l2[i-1] l1[i-1] D[i-1]) / D[i]
    //   l2[i] = f[i] / D[i]
    let mut dd = vec![0.0_f64; n];
    let mut l1 = vec![0.0_f64; n.saturating_sub(1)];
    let mut l2 = vec![0.0_f64; n.saturating_sub(2)];
    for i in 0..n {
        let mut di = d[i];
        if i >= 1 {
            di -= l1[i - 1] * l1[i - 1] * dd[i - 1];
        }
        if i >= 2 {
            di -= l2[i - 2] * l2[i - 2] * dd[i - 2];
        }
        if !(di.is_finite() && di > 0.0) {
            // Unreachable for finite input: A is SPD by construction.
            return Err(FiltersError::InvalidParameter {
                name: "lambda",
                value: lambda,
                requirement: "a value for which I + lambda*K'K stays numerically positive definite",
            });
        }
        dd[i] = di;
        if i + 1 < n {
            let mut t = e[i];
            if i >= 1 {
                t -= l2[i - 1] * l1[i - 1] * dd[i - 1];
            }
            l1[i] = t / di;
        }
        if i + 2 < n {
            l2[i] = f[i] / di;
        }
    }

    // Forward substitution L z = y, diagonal scale, back substitution
    // L' tau = D^{-1} z.
    let mut tau = vec![0.0_f64; n];
    for i in 0..n {
        let mut zi = y[i];
        if i >= 1 {
            zi -= l1[i - 1] * tau[i - 1];
        }
        if i >= 2 {
            zi -= l2[i - 2] * tau[i - 2];
        }
        tau[i] = zi;
    }
    for i in 0..n {
        tau[i] /= dd[i];
    }
    for i in (0..n).rev() {
        if i + 1 < n {
            tau[i] -= l1[i] * tau[i + 1];
        }
        if i + 2 < n {
            tau[i] -= l2[i] * tau[i + 2];
        }
    }
    Ok(tau)
}
