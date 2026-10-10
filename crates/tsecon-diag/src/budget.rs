//! Size budgets of the lag regressions shared by [`crate::arch_lm`],
//! [`crate::adf`], [`crate::dfgls`] and [`crate::ng_perron`] (and, through
//! the base-ADF lag search, [`crate::zivot_andrews`]).
//!
//! A lag order the sample admits can still ask for a design no machine
//! holds, or a lag search that runs for hours: the statsmodels bound lets
//! `adf` search up to `n/2` lags, a design of about `n^2 / 4` entries, and
//! `arch_lm` admits `nlags` up to `(n - 2) / 2`. Every such order is sized
//! before any lag column is built and refused, naming the order and the
//! largest one that fits, when it exceeds either budget:
//!
//! * [`LAG_DESIGN_MAX_CELLS`] — the lag design, `rows x columns` doubles,
//!   at most `2^26` entries (512 MiB per copy; the QR keeps a second one),
//!   the design budget of `breusch_godfrey_test` and `ardl_bounds_test`;
//! * [`LAG_WORK_BUDGET`] — the multiply-adds of the regression or of the
//!   whole lag search, at most `2^35`.
//!
//! The work counts are the leading terms: a Householder QR of a
//! `rows x k` design costs `rows k^2` multiply-adds, so a search over
//! `0..=L` lags from `s` fixed columns costs `rows (s^2 + ... + (s+L)^2)`
//! plus the final refit; the centered normal equations of `arch_lm` cost
//! `rows p^2 / 2 + p^3`. Measured at 1.1 to 1.5 ns per multiply-add for
//! the `adf` QR search and 1.7 ns for the `arch_lm` normal equations
//! (`lab/unitroot_cost/lag_cost.py`), so the budget is under a minute: the
//! largest admissible `adf` search at n = 20,000 ran in 36 s, the largest
//! `arch_lm` order at n = 30,000 in 57 s.
//!
//! The budgets police the lag order, not the series length: an order no
//! longer than the default rule's (Schwert's `ceil(12 (n/100)^(1/4))`,
//! capped, for the unit-root tests) is exempt from the work budget, so a
//! default call — and the internal callers that use the default
//! (`check_stationarity`, `ndiffs`, the panel tests) — is refused only when
//! its design would not fit the memory budget (a default `adf` search above
//! about 610,000 observations).

use crate::error::{DiagError, LagLimit};

/// Largest lag design one call builds, in `f64` entries (`2^26`, 512 MiB).
pub const LAG_DESIGN_MAX_CELLS: u128 = 1 << 26;

/// Work budget of one lag regression or lag search, in multiply-adds
/// (`2^35`, under a minute at the measured rates).
pub const LAG_WORK_BUDGET: u128 = 1 << 35;

/// The phrase a refusal quotes for the budget(s) a design exceeds, or
/// `None` when it fits both.
pub fn lag_budget_exceeded(cells: u128, work: u128) -> Option<&'static str> {
    match (cells > LAG_DESIGN_MAX_CELLS, work > LAG_WORK_BUDGET) {
        (true, true) => Some(
            "both the memory budget of 2^26 = 67108864 entries (512 MiB) and the work \
             budget of 2^35 multiply-adds (under a minute)",
        ),
        (true, false) => Some("the memory budget of 2^26 = 67108864 entries (512 MiB)"),
        (false, true) => Some("the work budget of 2^35 multiply-adds (under a minute)"),
        (false, false) => None,
    }
}

/// `sum_{j=lo}^{hi} j^2`, saturating (`0` for an empty range).
fn sum_squares(lo: u128, hi: u128) -> u128 {
    if hi < lo {
        return 0;
    }
    let s = |m: u128| {
        m.saturating_mul(m.saturating_add(1))
            .saturating_mul(m.saturating_mul(2).saturating_add(1))
            / 6
    };
    s(hi).saturating_sub(if lo == 0 { 0 } else { s(lo - 1) })
}

/// Size of an ADF-type lag regression on `n` observations with `fixed`
/// columns besides the lagged differences (the deterministics and the
/// lagged level) and `lags` lagged differences — `(cells, work)` of the
/// design and its QR. With `search`, the work is that of fitting every
/// order `0..=lags` on the common sample trimmed at `lags` and refitting the
/// chosen one on the longest sample (bounded by the longest fit).
pub(crate) fn adf_cost(n: usize, fixed: usize, lags: usize, search: bool) -> (u128, u128) {
    let (n, s, l) = (n as u128, fixed as u128, lags as u128);
    let rows = n.saturating_sub(1).saturating_sub(l);
    let k = s.saturating_add(l);
    let cells = rows.saturating_mul(k);
    let one = rows.saturating_mul(k.saturating_mul(k));
    let work = if search {
        let longest = n.saturating_sub(1).saturating_mul(k.saturating_mul(k));
        rows.saturating_mul(sum_squares(s, k))
            .saturating_add(longest)
    } else {
        one
    };
    (cells, work)
}

/// Size of the ARCH-LM auxiliary regression of `n` squared residuals on
/// `p` of their own lags: `(cells, work)` with `cells = (n - p) p` and
/// `work = (n - p) p^2 / 2 + p^3` (the centered normal equations and their
/// Cholesky factor).
pub(crate) fn arch_cost(n: usize, p: usize) -> (u128, u128) {
    let (n, p) = (n as u128, p as u128);
    let rows = n.saturating_sub(p);
    let p2 = p.saturating_mul(p);
    let cells = rows.saturating_mul(p);
    let work = (rows.saturating_mul(p2) / 2).saturating_add(p2.saturating_mul(p));
    (cells, work)
}

/// The largest order in `lo..=hi` whose `cost` fits both budgets, or
/// `None` when not even `lo` does (`cost` grows with the order on the
/// admissible range, so the admissible orders are a prefix: bisect).
pub(crate) fn largest_within_budget(
    lo: usize,
    hi: usize,
    cost: impl Fn(usize) -> (u128, u128),
) -> Option<usize> {
    let fits = |p: usize| {
        let (c, w) = cost(p);
        lag_budget_exceeded(c, w).is_none()
    };
    if hi < lo || !fits(lo) {
        return None;
    }
    let (mut lo, mut hi) = (lo, hi);
    while lo < hi {
        let mid = lo + (hi - lo).div_ceil(2);
        if fits(mid) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    Some(lo)
}

/// Refuse a lag order past the sample bound `max_sample` (with `rule` the
/// bound's statement; `None` when the sample admits no order), or one whose
/// design `cost` exceeds the budgets — before anything is allocated. `min`
/// is the smallest admissible order; orders up to `free` (the default
/// rule's order) are exempt from the work budget, so a default call is
/// refused only when its design would not fit in memory. The refusal quotes
/// the largest order within the sample bound and the budgets.
#[allow(clippy::too_many_arguments)]
pub(crate) fn check_lag_order(
    what: &'static str,
    param: &'static str,
    value: usize,
    defaulted: bool,
    n: usize,
    min: usize,
    max_sample: Option<usize>,
    free: usize,
    rule: &'static str,
    cost: impl Fn(usize) -> (u128, u128),
) -> Result<(), DiagError> {
    // The work an order is charged against the budget: none up to `free`.
    let charged = |p: usize| {
        let (c, w) = cost(p);
        (c, if p <= free { 0 } else { w })
    };
    let sample_max = max_sample.filter(|&m| m >= min);
    let max = || sample_max.and_then(|m| largest_within_budget(min, m, charged));
    if sample_max.is_none_or(|m| value > m) {
        return Err(DiagError::LagOrderTooLong {
            what,
            param,
            value,
            defaulted,
            n,
            max: max(),
            limit: LagLimit::Sample { rule, sample_max },
        });
    }
    let (cells, charged_work) = charged(value);
    if lag_budget_exceeded(cells, charged_work).is_some() {
        return Err(DiagError::LagOrderTooLong {
            what,
            param,
            value,
            defaulted,
            n,
            max: max(),
            limit: LagLimit::Budget {
                cells: u64::try_from(cells).unwrap_or(u64::MAX),
                work: u64::try_from(cost(value).1).unwrap_or(u64::MAX),
                work_exempt: value <= free,
            },
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sum_squares_matches_the_loop() {
        for lo in 0..6u128 {
            for hi in 0..12u128 {
                let direct: u128 = (lo..=hi).map(|j| j * j).sum();
                assert_eq!(sum_squares(lo, hi), direct, "lo {lo} hi {hi}");
            }
        }
    }

    #[test]
    fn costs_saturate_instead_of_wrapping() {
        let (c, w) = adf_cost(usize::MAX, 3, usize::MAX / 2, true);
        assert!(lag_budget_exceeded(c, w).is_some());
        let (c, w) = arch_cost(usize::MAX, usize::MAX / 3);
        assert!(lag_budget_exceeded(c, w).is_some());
    }

    #[test]
    fn largest_within_budget_is_the_last_fitting_order() {
        let n = 200_000;
        let cost = |p: usize| adf_cost(n, 2, p, true);
        let best = largest_within_budget(0, n / 2 - 2, cost).unwrap();
        let fits = |p: usize| {
            let (c, w) = cost(p);
            lag_budget_exceeded(c, w).is_none()
        };
        assert!(fits(best) && !fits(best + 1), "best {best}");
        assert_eq!(largest_within_budget(5, 4, cost), None);
    }
}
