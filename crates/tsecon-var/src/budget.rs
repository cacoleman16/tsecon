//! Size budgets shared by the VAR-family engines.
//!
//! Every allocation that scales with a user count (a horizon, a step count,
//! a replication or simulation count) is sized before it is made and refused
//! beyond [`BUDGET_VALUES`] (`2^28` doubles, 2 GiB), with checked arithmetic
//! so that an absurd count cannot overflow the very count meant to refuse it.
//! A refusal is a [`VarError::SizeBudget`] naming the keyword and the largest
//! value that fits at the other arguments, never an allocator abort.
//!
//! # Counting real storage
//!
//! The counts are of memory as it is actually held, not of cells:
//!
//! * [`mat_values`]: a `k x k` faer matrix pads each column to 64 bytes and
//!   is a separate, 64-byte-aligned heap allocation behind a header, so it
//!   costs `8 ceil(k / 8) k + 16` doubles (a 2 x 2 matrix: 32 doubles, not 4);
//! * [`list_values`]: the same matrix as a Python nested list holds `k^2`
//!   floats of 32 bytes each (24 rounded up by the allocator) and `k + 1`
//!   list objects of about `80 + 8 k` bytes each, `4 k^2 + (k + 1)(k + 10)`
//!   doubles in all.
//!
//! A horizon-indexed cube (an impulse response, a forecast-error
//! covariance path) is counted [`CUBE_COPIES`] times in both forms. The
//! heaviest caller holds about 3.2 such copies at its peak (`long_run_svar`:
//! the moving-average weights, the responses, and the responses, cumulated
//! responses and decomposition as nested lists), measured by
//! `lab/audit/round16/var_budgets/footprint.py`, so the count is an upper
//! bound with room to spare, not an estimate.

use crate::error::VarError;

/// The working-set budget of one call, in `f64` values (`2^28`, 2 GiB).
pub const BUDGET_VALUES: usize = 1 << 28;

/// Copies of a horizon-indexed `k x k` cube, each counted as faer matrices
/// plus Python nested lists, that one call is budgeted for.
pub const CUBE_COPIES: usize = 8;

/// Storage of one `k x k` faer matrix in `f64` units: columns padded to 64
/// bytes, plus 128 bytes of allocator alignment slack and matrix header.
pub fn mat_values(k: usize) -> Option<usize> {
    dense_values(k, k)
}

/// Storage of one dense `rows x cols` faer matrix in `f64` units: each
/// column padded to 64 bytes, plus 128 bytes of alignment slack and header.
pub fn dense_values(rows: usize, cols: usize) -> Option<usize> {
    rows.div_ceil(8)
        .checked_mul(8)?
        .checked_mul(cols)?
        .checked_add(16)
}

/// Storage of one `k x k` matrix as a Python nested list, in `f64` units:
/// `k^2` floats at 32 bytes each and `k + 1` lists of about `80 + 8 k` bytes.
pub fn list_values(k: usize) -> Option<usize> {
    let floats = k.checked_mul(k)?.checked_mul(4)?;
    let lists = k.checked_add(1)?.checked_mul(k.checked_add(10)?)?;
    floats.checked_add(lists)
}

/// Storage of one `k x k` entry of a reported cube in both forms (the faer
/// matrix and its Python nested list).
pub fn cube_entry_values(k: usize) -> Option<usize> {
    mat_values(k)?.checked_add(list_values(k)?)
}

/// Storage of `copies` copies of a cube of `entries` matrices of `k x k`,
/// or `None` when the count overflows.
pub fn cube_values(k: usize, entries: usize, copies: usize) -> Option<usize> {
    entries
        .checked_mul(copies)?
        .checked_mul(cube_entry_values(k)?)
}

/// The largest `v` in `lower..=upper` with `fits(v)`, for a `fits` that is
/// true up to some point and false beyond it (every budget here is monotone
/// in each count); `None` when not even `lower` fits.
pub fn largest_fitting(lower: usize, upper: usize, fits: impl Fn(usize) -> bool) -> Option<usize> {
    if lower > upper || !fits(lower) {
        return None;
    }
    let (mut ok, mut bad) = (lower, upper);
    if fits(bad) {
        return Some(bad);
    }
    // Invariant: fits(ok) && !fits(bad).
    while bad - ok > 1 {
        let mid = ok + (bad - ok) / 2;
        if fits(mid) {
            ok = mid;
        } else {
            bad = mid;
        }
    }
    Some(ok)
}

/// The subject of an impulse-response horizon refusal.
pub const HORIZON_WHAT: &str = "horizon: every horizon adds one k x k matrix to each of the \
    copies the call holds (the moving-average weights, the responses and their Python nested \
    lists, counted as real storage: padded matrices and list objects), and at this horizon that \
    working set exceeds the 2^28-value (2 GiB) budget, checked before anything is allocated; \
    reduce horizon";

/// The subject of a forecast step-count refusal.
pub const STEPS_WHAT: &str = "steps: a forecast to steps periods ahead holds steps matrices of \
    k x k in several copies (the moving-average weights, the forecast-error covariances and \
    their Python nested lists, counted as real storage: padded matrices and list objects), and \
    that working set exceeds the 2^28-value (2 GiB) budget, checked before anything is \
    allocated; reduce steps";

/// Refuses a cube of `value + extra` matrices of `k x k`, held `copies`
/// times, whose storage exceeds [`BUDGET_VALUES`]: the refusal names `name`
/// and the largest value that fits (`lower` is the smallest admissible one).
///
/// # Errors
///
/// [`VarError::SizeBudget`] naming `name`.
pub fn check_cube(
    what: &'static str,
    name: &'static str,
    value: usize,
    lower: usize,
    extra: usize,
    k: usize,
    copies: usize,
) -> Result<(), VarError> {
    let fits = |v: usize| {
        v.checked_add(extra)
            .and_then(|entries| cube_values(k, entries, copies))
            .is_some_and(|c| c <= BUDGET_VALUES)
    };
    if fits(value) {
        return Ok(());
    }
    Err(VarError::SizeBudget {
        what,
        name,
        value,
        max: largest_fitting(lower, value, fits),
    })
}

/// Refuses an impulse-response `horizon` (`horizon + 1` matrices of `k x k`)
/// beyond the budget, naming `horizon`. Every moving-average expansion in
/// the crate ([`crate::ma_rep`], [`crate::VarResults::ma_rep`]) checks this
/// before allocating, so every caller inherits it.
///
/// # Errors
///
/// [`VarError::SizeBudget`] naming `horizon`.
pub fn check_horizon(k: usize, horizon: usize) -> Result<(), VarError> {
    check_cube(HORIZON_WHAT, "horizon", horizon, 0, 1, k, CUBE_COPIES)
}

/// Refuses a forecast of `steps` periods (`steps` matrices of `k x k`) beyond
/// the budget, naming `steps`. The forecast paths check this before their
/// moving-average expansion, so the refusal names their own keyword.
///
/// # Errors
///
/// [`VarError::SizeBudget`] naming `steps`.
pub fn check_steps(k: usize, steps: usize) -> Result<(), VarError> {
    check_cube(STEPS_WHAT, "steps", steps, 1, 0, k, CUBE_COPIES)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

    #[test]
    fn largest_fitting_finds_the_boundary() {
        assert_eq!(largest_fitting(0, 1000, |v| v <= 377), Some(377));
        assert_eq!(largest_fitting(2, 1000, |v| v <= 1), None);
        assert_eq!(largest_fitting(0, 10, |_| true), Some(10));
        assert_eq!(largest_fitting(0, usize::MAX, |v| v <= 5), Some(5));
        assert_eq!(largest_fitting(5, 4, |_| true), None);
    }

    #[test]
    fn storage_counts_match_their_formulas() {
        // A 2 x 2 matrix: two 64-byte columns plus 128 bytes.
        assert_eq!(mat_values(2), Some(32));
        assert_eq!(mat_values(9), Some(16 * 9 + 16));
        assert_eq!(list_values(2), Some(16 + 3 * 12));
        assert_eq!(cube_entry_values(2), Some(84));
        assert_eq!(mat_values(usize::MAX), None);
        assert_eq!(cube_values(2, usize::MAX, 8), None);
    }

    #[test]
    fn horizon_budget_is_exact_at_its_boundary_and_names_horizon() {
        let per = 8 * 84;
        let max = BUDGET_VALUES / per - 1;
        assert!(check_horizon(2, max).is_ok());
        match check_horizon(2, max + 1) {
            Err(VarError::SizeBudget {
                name: "horizon",
                value,
                max: Some(m),
                ..
            }) => {
                assert_eq!(value, max + 1);
                assert_eq!(m, max);
            }
            other => panic!("{other:?}"),
        }
        for h in [1usize << 31, 1 << 40, usize::MAX] {
            let msg = check_horizon(2, h).unwrap_err().to_string();
            assert!(msg.starts_with("horizon:"), "{msg}");
            assert!(msg.contains(&format!("is {max})")), "{msg}");
        }
        // Steps count entries, not entries - 1.
        assert!(check_steps(2, max + 1).is_ok());
        let msg = check_steps(2, max + 2).unwrap_err().to_string();
        assert!(
            msg.starts_with("steps:") && msg.contains(&format!("is {})", max + 1)),
            "{msg}"
        );
        // A system too wide for even one matrix: nothing fits.
        assert!(matches!(
            check_horizon(1 << 20, 0),
            Err(VarError::SizeBudget { max: None, .. })
        ));
    }
}
