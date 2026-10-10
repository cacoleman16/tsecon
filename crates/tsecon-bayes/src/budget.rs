//! Size budgets shared by the Bayesian and structural-identification engines.
//!
//! Every allocation that scales with a user count — an impulse-response
//! horizon, a posterior draw count, a chain count, a grid size — is sized
//! before it is made and refused beyond [`MEMORY_BUDGET_VALUES`] (`2^28`
//! doubles, 2 GiB, the budget of `long_run_svar_bands`), naming the Python
//! keyword and the largest value that fits with the other arguments as
//! passed. The count is of real storage in `f64` units, not of cells, and it
//! is computed in saturating `u128` arithmetic so an absurd count cannot
//! overflow the check meant to refuse it:
//!
//! * one `n x n` faer matrix occupies [`mat_values`]`(n) = 8 ceil(n / 8) n + 8`
//!   doubles (faer pads each column to 64 bytes; the header takes about a
//!   cache line);
//! * one matrix handed back to Python as a nested list costs a further
//!   [`py_matrix_values`]`(n) = 5 n^2 + 11 n + 3` doubles (the Rust nested
//!   `Vec` staged for the conversion, plus about four doubles per Python
//!   float and list slot);
//! * one RNG substream is [`STREAM_VALUES`] doubles (72 bytes), one per
//!   posterior draw, spawned before the first draw is taken;
//! * a summarized `(horizon, variable, shock)` cell — its quantile vector,
//!   envelope and the Python lists they become — is at most
//!   [`SUMMARY_CELL_VALUES`] doubles.
//!
//! Each engine states its own count next to its check. The engines whose
//! sizes these budgets bound: `cholesky_irf` and the NIW / SSVS posterior
//! draws in this crate; the sign, zero-sign, narrative, Fry-Pagan and
//! robust-bound samplers, the non-Gaussian SVAR and the structural FEVD in
//! `tsecon-ident`; and the structural bindings that build their IRFs on the
//! VAR moving-average representation.

use crate::error::BayesError;

/// The working-set budget of one call, in `f64` values (`2^28`, i.e. 2 GiB).
pub const MEMORY_BUDGET_VALUES: u128 = 1 << 28;

/// How [`MEMORY_BUDGET_VALUES`] is quoted in a refusal.
pub const MEMORY_BUDGET: &str = "the 2^28-value (2 GiB) working-set budget";

/// One `tsecon_rng::Stream` substream, in `f64` units (72 bytes).
pub const STREAM_VALUES: u128 = 9;

/// One summarized `(horizon, variable, shock)` cell, in `f64` units: the
/// quantile vector and its header, the envelope, and the Python lists the
/// binding builds from them (generous, so the count stays an upper bound).
pub const SUMMARY_CELL_VALUES: u128 = 64;

/// Storage of one `n x n` faer matrix in `f64` units:
/// `8 ceil(n / 8) n + 8` (column padding to 64 bytes plus the header).
pub fn mat_values(n: usize) -> u128 {
    rect_values(n, n)
}

/// Storage of one `rows x cols` faer matrix in `f64` units.
pub fn rect_values(rows: usize, cols: usize) -> u128 {
    let padded = (rows as u128).div_ceil(8).saturating_mul(8);
    padded.saturating_mul(cols as u128).saturating_add(8)
}

/// The extra cost of handing one `n x n` matrix back to Python as a nested
/// list, in `f64` units: the staged Rust `Vec<Vec<f64>>` (`n^2 + 3 n + 3`)
/// plus the Python list of lists (about `4 n^2 + 8 n`).
pub fn py_matrix_values(n: usize) -> u128 {
    let n = n as u128;
    n.saturating_mul(n)
        .saturating_mul(5)
        .saturating_add(n.saturating_mul(11))
        .saturating_add(3)
}

/// One size argument of a budget check: its Python keyword, the value
/// passed, and the smallest value the callable accepts.
#[derive(Debug, Clone, Copy)]
pub struct SizeArg {
    /// The Python keyword a refusal names.
    pub name: &'static str,
    /// The value passed.
    pub value: usize,
    /// The smallest admissible value (the search floor of the largest
    /// admissible value a refusal quotes).
    pub min: usize,
}

impl SizeArg {
    /// A size argument `name = value` whose smallest admissible value is
    /// `min`.
    pub const fn new(name: &'static str, value: usize, min: usize) -> Self {
        Self { name, value, min }
    }
}

/// The largest `x` in `lo..=hi` with `fits(x)`, for a `fits` that holds on
/// an initial segment; `None` when even `fits(lo)` fails (or `lo > hi`).
pub fn largest_fitting(lo: usize, hi: usize, fits: impl Fn(usize) -> bool) -> Option<usize> {
    if lo > hi || !fits(lo) {
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

/// One budget a call is sized against: what it counts, its limit, how the
/// limit is quoted, and the count — a non-decreasing function of every size
/// argument of the [`check_budgets`] call, in the order they are listed.
pub struct Budget<'a> {
    /// What the budget counts, as a refusal states it.
    pub what: &'static str,
    /// The largest admissible count.
    pub limit: u128,
    /// How `limit` is quoted (e.g. [`MEMORY_BUDGET`]).
    pub phrase: &'static str,
    /// The count at the given argument values.
    pub cost: &'a dyn Fn(&[usize]) -> u128,
}

/// Refuses a call whose size arguments push any of `budgets` past its
/// limit, before anything is allocated. Several budgets are checked
/// together (a memory and a work budget, or the working sets of two stages
/// that are live at once) so that the largest value a refusal quotes passes
/// every one of them, not only the budget that tripped first.
///
/// The refusal names the FIRST argument of `args` (in the order given) that
/// does not fit even with every other argument at its minimum. When each
/// fits alone and only their product does not, it names the LAST argument
/// that can be lowered to fit (one passed above its minimum), never one
/// already at its minimum. It quotes the largest value of the named
/// argument that fits every budget with every other argument as passed
/// (`None` when none does), and the budget that value runs into. Callers
/// list the per-draw size (the horizon) before the draw count, so an absurd
/// draw count is blamed on the draw count and an absurd horizon on the
/// horizon.
///
/// # Errors
///
/// [`BayesError::TooLarge`] as described.
pub fn check_budgets(args: &[SizeArg], budgets: &[Budget<'_>]) -> Result<(), BayesError> {
    let fits = |v: &[usize]| budgets.iter().all(|b| (b.cost)(v) <= b.limit);
    let passed: Vec<usize> = args.iter().map(|a| a.value).collect();
    if fits(&passed) {
        return Ok(());
    }
    let Some(last) = args.len().checked_sub(1) else {
        // No size argument to blame (a fixed cost past its budget).
        let b = budgets.iter().find(|b| (b.cost)(&passed) > b.limit);
        return Err(BayesError::TooLarge {
            arg: "the size arguments",
            value: 0,
            max: None,
            what: b.map_or("the working set", |b| b.what),
            budget: b.map_or(MEMORY_BUDGET, |b| b.phrase),
        });
    };
    let floors: Vec<usize> = args.iter().map(|a| a.min.min(a.value)).collect();
    let with = |base: &[usize], i: usize, x: usize| {
        let mut v = base.to_vec();
        v[i] = x;
        v
    };
    let largest = |i: usize| {
        let a = &args[i];
        largest_fitting(a.min, a.value.saturating_sub(1), |x| {
            fits(&with(&passed, i, x))
        })
    };
    let alone = (0..args.len()).find(|&i| !fits(&with(&floors, i, passed[i])));
    let (blame, max) = match alone {
        Some(i) => (i, largest(i)),
        None => {
            // A product of arguments that each fit alone: blame the last one
            // that can be lowered to fit, else the last one above its floor.
            let reducible: Vec<usize> = (0..args.len())
                .rev()
                .filter(|&i| passed[i] > floors[i])
                .collect();
            reducible
                .iter()
                .find_map(|&i| largest(i).map(|m| (i, Some(m))))
                .or_else(|| reducible.first().map(|&i| (i, None)))
                .unwrap_or((last, None))
        }
    };
    // The budget the quoted value runs into: the first one that fails one
    // step above it (at the value passed when no value fits).
    let at = with(
        &passed,
        blame,
        max.map_or(passed[blame], |m| m.saturating_add(1)),
    );
    let tripped = budgets
        .iter()
        .find(|b| (b.cost)(&at) > b.limit)
        .or_else(|| budgets.iter().find(|b| (b.cost)(&passed) > b.limit));
    let (what, budget) = tripped.map_or(("the working set", MEMORY_BUDGET), |b| (b.what, b.phrase));
    let arg = &args[blame];
    Err(BayesError::TooLarge {
        arg: arg.name,
        value: arg.value,
        max,
        what,
        budget,
    })
}

/// [`check_budgets`] against the single budget `cost <= limit`.
///
/// # Errors
///
/// [`BayesError::TooLarge`] as for [`check_budgets`].
pub fn check_sizes(
    what: &'static str,
    limit: u128,
    budget: &'static str,
    args: &[SizeArg],
    cost: impl Fn(&[usize]) -> u128,
) -> Result<(), BayesError> {
    check_budgets(
        args,
        &[Budget {
            what,
            limit,
            phrase: budget,
            cost: &cost,
        }],
    )
}

/// [`check_sizes`] against [`MEMORY_BUDGET_VALUES`].
///
/// # Errors
///
/// [`BayesError::TooLarge`] when the working set `values` exceeds the
/// memory budget.
pub fn check_memory(
    what: &'static str,
    args: &[SizeArg],
    values: impl Fn(&[usize]) -> u128,
) -> Result<(), BayesError> {
    check_sizes(what, MEMORY_BUDGET_VALUES, MEMORY_BUDGET, args, values)
}

/// The working set of an impulse-response path to `horizon` for `n`
/// variables: `(horizon + 1)` matrices of [`mat_values`]`(n)`, each
/// costing `per_matrix_extra` more (for example [`py_matrix_values`] when it
/// is returned to Python).
pub fn irf_path_values(n: usize, horizon: usize, per_matrix_extra: u128) -> u128 {
    (horizon as u128)
        .saturating_add(1)
        .saturating_mul(mat_values(n).saturating_add(per_matrix_extra))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    #[test]
    fn largest_fitting_bisects_the_boundary() {
        assert_eq!(largest_fitting(0, 100, |x| x <= 37), Some(37));
        assert_eq!(largest_fitting(1, 100, |x| x <= 100), Some(100));
        assert_eq!(largest_fitting(5, 100, |x| x <= 4), None);
        assert_eq!(largest_fitting(5, 4, |_| true), None);
        assert_eq!(
            largest_fitting(0, usize::MAX - 1, |x| x < (1 << 40)),
            Some((1 << 40) - 1)
        );
    }

    #[test]
    fn mat_values_counts_padded_columns() {
        assert_eq!(mat_values(1), 16);
        assert_eq!(mat_values(3), 8 * 3 + 8);
        assert_eq!(mat_values(8), 64 + 8);
        assert_eq!(mat_values(9), 16 * 9 + 8);
        assert_eq!(rect_values(9, 2), 16 * 2 + 8);
        assert_eq!(py_matrix_values(3), 45 + 33 + 3);
    }

    #[test]
    fn check_sizes_blames_the_absurd_argument_and_quotes_the_largest_fit() {
        // cost = draws * (horizon + 1) * 10
        let cost = |v: &[usize]| (v[1] as u128) * (v[0] as u128 + 1) * 10;
        let args = |h, d| [SizeArg::new("horizon", h, 0), SizeArg::new("n_draws", d, 1)];
        assert!(check_sizes("w", 1000, "b", &args(9, 10), cost).is_ok());
        // An absurd draw count is blamed on the draw count.
        match check_sizes("w", 1000, "b", &args(9, 1 << 31), cost) {
            Err(BayesError::TooLarge {
                arg, value, max, ..
            }) => {
                assert_eq!((arg, value, max), ("n_draws", 1 << 31, Some(10)));
            }
            other => panic!("{other:?}"),
        }
        // An absurd horizon is blamed on the horizon, with the largest
        // horizon that fits at the draw count passed.
        match check_sizes("w", 1000, "b", &args(1 << 31, 10), cost) {
            Err(BayesError::TooLarge { arg, max, .. }) => {
                assert_eq!((arg, max), ("horizon", Some(9)));
            }
            other => panic!("{other:?}"),
        }
        // Each fits alone (the other at its floor) and only the product
        // does not: the last argument, the draw count, is named.
        match check_sizes("w", 1000, "b", &args(49, 10), cost) {
            Err(BayesError::TooLarge { arg, max, .. }) => {
                assert_eq!((arg, max), ("n_draws", Some(2)));
            }
            other => panic!("{other:?}"),
        }
        // No value of the named argument fits with the other as passed.
        match check_sizes("w", 1000, "b", &args(1 << 31, 1 << 31), cost) {
            Err(BayesError::TooLarge { arg, max, .. }) => {
                assert_eq!((arg, max), ("horizon", None));
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn a_product_is_never_blamed_on_an_argument_at_its_minimum() {
        // cost = chains * draws * (horizon + 1); chains = 1 is its minimum,
        // so the product overflow is blamed on the draw count, the last
        // argument that can be lowered, never on "chains = 1".
        let cost = |v: &[usize]| (v[0] as u128 + 1) * (v[1] as u128) * (v[2] as u128);
        let args = [
            SizeArg::new("horizon", 99, 0),
            SizeArg::new("n_draws", 20, 1),
            SizeArg::new("n_chains", 1, 1),
        ];
        match check_sizes("w", 1000, "b", &args, cost) {
            Err(BayesError::TooLarge { arg, max, .. }) => {
                assert_eq!((arg, max), ("n_draws", Some(10)));
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn several_budgets_quote_a_value_that_passes_all_of_them() {
        // Memory: draws * 10 <= 1000 (draws <= 100). Work: draws * reps <=
        // 600. At reps = 20 the memory budget alone would quote 100, which
        // the work budget refuses; the quote is 30, and the work budget is
        // named as the one it runs into.
        let memory = |v: &[usize]| (v[1] as u128) * 10;
        let work = |v: &[usize]| (v[0] as u128) * (v[1] as u128);
        let budgets = [
            Budget {
                what: "memory",
                limit: 1000,
                phrase: "m",
                cost: &memory,
            },
            Budget {
                what: "work",
                limit: 600,
                phrase: "w",
                cost: &work,
            },
        ];
        let args = |reps, draws| {
            [
                SizeArg::new("reps", reps, 1),
                SizeArg::new("n_draws", draws, 1),
            ]
        };
        assert!(check_budgets(&args(20, 30), &budgets).is_ok());
        match check_budgets(&args(20, 1 << 31), &budgets) {
            Err(BayesError::TooLarge { arg, max, what, .. }) => {
                assert_eq!((arg, max, what), ("n_draws", Some(30), "work"));
            }
            other => panic!("{other:?}"),
        }
        // With few reps the memory budget is the binding one.
        match check_budgets(&args(2, 1 << 31), &budgets) {
            Err(BayesError::TooLarge { arg, max, what, .. }) => {
                assert_eq!((arg, max, what), ("n_draws", Some(100), "memory"));
            }
            other => panic!("{other:?}"),
        }
        assert!(check_budgets(&args(2, 100), &budgets).is_ok());
    }

    #[test]
    fn a_fixed_cost_past_its_budget_is_refused_without_panicking() {
        match check_sizes("w", 10, "b", &[], |_| 11) {
            Err(BayesError::TooLarge { arg, max, .. }) => {
                assert_eq!((arg, max), ("the size arguments", None));
            }
            other => panic!("{other:?}"),
        }
        assert!(check_sizes("w", 10, "b", &[], |_| 10).is_ok());
    }

    #[test]
    fn saturating_counts_refuse_instead_of_wrapping() {
        let cost = |v: &[usize]| irf_path_values(v[0], v[1], py_matrix_values(v[0]));
        let args = [
            SizeArg::new("n", usize::MAX, 1),
            SizeArg::new("horizon", usize::MAX, 0),
        ];
        assert!(check_memory("w", &args, cost).is_err());
    }
}
