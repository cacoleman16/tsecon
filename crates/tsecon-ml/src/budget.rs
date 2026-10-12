//! The memory budget every estimator in this crate checks a user count
//! against before allocating.
//!
//! A grid, path, ensemble, network width, epoch count or fold count is a
//! user-supplied integer that sizes an allocation. Each one is turned into
//! the number of `f64`-sized values (an `f64`, a `usize` index or a `Vec`
//! header word each count once) its working set holds, with checked
//! arithmetic so an absurd count cannot wrap, and refused beyond
//! [`ML_MAX_VALUES`] — `2^28` values, 2 GiB, the house budget of
//! `spa_test` and `long_run_svar_bands` — before anything is allocated. The refusal names the Python keyword and
//! the largest value that fits for the data at hand. Without it a count of
//! `2^31` reached the allocator, which aborts the process instead of
//! raising.
//!
//! The counts are of storage the call is certain to hold at once (the
//! returned paths, the kept ensemble members, the materialized folds); a
//! data-dependent size such as a tree's node count enters at its minimum,
//! so the budget never refuses a call that fits.

use crate::error::MlError;

/// The working-set budget of one call, in `f64`-sized values (`2^28`, 2 GiB).
pub const ML_MAX_VALUES: usize = 1 << 28;

/// Whether a working set of `values` (`None` when counting it overflowed)
/// fits the budget.
pub(crate) fn fits(values: Option<usize>) -> bool {
    values.is_some_and(|v| v <= ML_MAX_VALUES)
}

/// The largest count in `lo..=hi` whose working set fits, or `None` when not
/// even `lo` does. `values` must be nondecreasing in the count (every
/// working set here is), which makes the bisection exact.
pub(crate) fn largest_fitting(
    lo: usize,
    hi: usize,
    values: impl Fn(usize) -> Option<usize>,
) -> Option<usize> {
    if lo > hi || !fits(values(lo)) {
        return None;
    }
    let (mut lo, mut hi) = (lo, hi);
    while lo < hi {
        let mid = lo + (hi - lo).div_ceil(2);
        if fits(values(mid)) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    Some(lo)
}

/// Checks `keyword = value` against the budget: `values(count)` is the
/// working set at `count`, nondecreasing in it, and `holds` says what it
/// holds. Beyond the budget the refusal names `keyword`, the values asked
/// for and the largest count from `min` up that fits.
pub(crate) fn check_count(
    keyword: &str,
    value: usize,
    min: usize,
    holds: &str,
    values: impl Fn(usize) -> Option<usize>,
) -> Result<(), MlError> {
    let need = values(value);
    if fits(need) {
        return Ok(());
    }
    let need = need.map_or_else(|| "more than 2^64".to_string(), |v| format!("{v}"));
    let largest = match largest_fitting(min, value, &values) {
        Some(m) => format!("{keyword} can be at most {m} here"),
        None => format!(
            "not even {keyword} = {min} fits with this data and the other arguments as given"
        ),
    };
    Err(MlError::InvalidValue {
        what: format!(
            "{keyword} = {value} is too large: {holds} would hold {need} values, beyond the \
             2^28-value (2 GiB) memory budget, which is checked before anything is \
             allocated; {largest}"
        ),
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn largest_fitting_is_exact_and_overflow_safe() {
        // 1000 values per unit plus 24 fixed: floor((2^28 - 24) / 1000).
        let v = |c: usize| c.checked_mul(1000)?.checked_add(24);
        let m = largest_fitting(1, usize::MAX, v).unwrap();
        assert_eq!(m, (ML_MAX_VALUES - 24) / 1000);
        assert!(fits(v(m)) && !fits(v(m + 1)));
        assert_eq!(largest_fitting(1, usize::MAX, |_| None), None);
        assert!(check_count("n_things", usize::MAX, 1, "the things", v).is_err());
        assert!(check_count("n_things", m, 1, "the things", v).is_ok());
        let e = check_count("n_things", m + 1, 1, "the things", v)
            .unwrap_err()
            .to_string();
        assert!(
            e.contains(&format!("n_things can be at most {m} here")),
            "{e}"
        );
    }
}
