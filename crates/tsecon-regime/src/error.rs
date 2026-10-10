//! Error type for the Markov-switching crate.

use core::fmt;

/// Errors returned by the Markov-switching models.
///
/// Every fallible public function in this crate returns
/// `Result<_, RegimeError>`; no library code path panics on user input.
#[derive(Debug, Clone, PartialEq)]
pub enum RegimeError {
    /// The model specification itself is malformed (e.g. fewer than two
    /// regimes, or an expanded state space too large to materialize).
    InvalidSpec {
        /// Description of the violated requirement.
        what: &'static str,
    },
    /// A parameter value is outside its admissible domain (a non-positive
    /// variance, a transition probability outside `[0, 1]`, ...).
    InvalidParameter {
        /// Name of the offending parameter (or parameter group).
        name: &'static str,
        /// The invalid value that was supplied.
        value: f64,
        /// Human-readable statement of the violated constraint.
        requirement: &'static str,
    },
    /// A transition matrix column does not sum to one (the columns of the
    /// column-stochastic transition matrix `P[i][j] = P(S_t = i | S_{t-1} =
    /// j)` must each be a probability distribution over the destination
    /// regime).
    NotStochastic {
        /// Index of the offending column (the conditioning regime `j`).
        column: usize,
        /// The column sum that should have been one.
        sum: f64,
    },
    /// A parameter or data container has the wrong length for the
    /// specification.
    DimensionMismatch {
        /// Description of the offending input.
        what: &'static str,
        /// The expected length.
        expected: usize,
        /// The actual length.
        actual: usize,
    },
    /// An input or an intermediate quantity contains NaN or infinity where
    /// finite values are required (data, parameters, or a mixture
    /// likelihood that underflowed to zero).
    NonFinite {
        /// Name of the offending quantity.
        what: &'static str,
    },
    /// Too few observations for the requested model. `what` names the
    /// data argument and the parameters that set the requirement (a lag
    /// order, a delay, a trim fraction), so the message says which knob
    /// to turn.
    InsufficientData {
        /// Which input, under which parameters, ran out of observations.
        what: &'static str,
        /// Minimum number of observations required.
        needed: usize,
        /// Number of observations supplied.
        got: usize,
    },
    /// The working set a simulation would allocate exceeds the engine's
    /// memory budget, or the allocator refused it (mapped from the GIRF
    /// engine's [`tsecon_var::VarError::MemoryBudget`]).
    MemoryBudget {
        /// Which engine and which parameters size the working set.
        what: &'static str,
        /// Bytes the request would need.
        bytes: u128,
        /// The engine's budget in bytes.
        budget: usize,
    },
    /// A linear system in the estimation step (the stationary distribution
    /// solve, or an M-step normal-equation system) was singular.
    Singular {
        /// Description of the system that could not be solved.
        what: &'static str,
    },
    /// The Markov-switching filter's expanded state space
    /// `k_regimes^(order + 1)` exceeds its cap (the Hamilton filter tracks
    /// the last `order + 1` regimes jointly), refused before anything is
    /// allocated. The message states the largest admissible `k_regimes` at
    /// this `order` and the largest `order` at this `k_regimes`.
    StateSpaceTooLarge {
        /// The requested number of regimes.
        k_regimes: usize,
        /// The requested autoregressive order.
        order: usize,
        /// The cap on `k_regimes^(order + 1)`.
        max_states: usize,
    },
    /// The STAR `(gamma, c)` grid search would evaluate more than its cell
    /// budget, `n_gamma * n_c` cells per candidate delay, refused before
    /// the grid is allocated. The message states the largest admissible
    /// `n_gamma` at this `n_c` and `n_c` at this `n_gamma`.
    GridTooLarge {
        /// The requested gamma-grid size.
        n_gamma: usize,
        /// The requested c-grid size.
        n_c: usize,
        /// The cell budget.
        max_cells: usize,
    },
}

/// The largest `k >= 1` with `k^(order + 1) <= max` (`max >= 1`).
fn max_regimes_at_order(order: usize, max: usize) -> usize {
    let fits = |k: usize| {
        let mut m: usize = 1;
        for _ in 0..=order {
            match m.checked_mul(k) {
                Some(v) if v <= max => m = v,
                _ => return false,
            }
        }
        true
    };
    let (mut lo, mut hi) = (1usize, max);
    while lo < hi {
        let mid = lo + (hi - lo).div_ceil(2);
        if fits(mid) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    lo
}

/// The largest `order` with `k^(order + 1) <= max`, or `None` when even
/// `order = 0` (`k` states) does not fit (`k >= 2`).
fn max_order_at_regimes(k: usize, max: usize) -> Option<usize> {
    if k > max {
        return None;
    }
    let mut m = k;
    let mut order = 0usize;
    while let Some(v) = m.checked_mul(k).filter(|&v| v <= max) {
        m = v;
        order += 1;
    }
    Some(order)
}

impl fmt::Display for RegimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSpec { what } => write!(f, "invalid model specification: {what}"),
            Self::InvalidParameter {
                name,
                value,
                requirement,
            } => write!(
                f,
                "invalid parameter {name} = {value}: requires {requirement}"
            ),
            Self::NotStochastic { column, sum } => write!(
                f,
                "transition matrix column {column} sums to {sum}, not 1 \
                 (columns must be probability distributions)"
            ),
            Self::DimensionMismatch {
                what,
                expected,
                actual,
            } => write!(
                f,
                "dimension mismatch: {what} (expected {expected}, got {actual})"
            ),
            Self::NonFinite { what } => {
                write!(f, "non-finite value (NaN or infinity) in {what}")
            }
            Self::InsufficientData { what, needed, got } => write!(
                f,
                "insufficient data for {what}: {got} observations, at least {needed} \
                 required"
            ),
            Self::MemoryBudget {
                what,
                bytes,
                budget,
            } => write!(
                f,
                "{what}: the working set would need {bytes} bytes ({:.2} GiB), beyond the \
                 engine's memory budget of {budget} bytes ({:.2} GiB) — or the allocator \
                 refused it; reduce n_draws, the horizon, or the number of histories \
                 (the budget is a fixed constant of the engine, not a machine probe)",
                *bytes as f64 / f64::from(1u32 << 30),
                *budget as f64 / f64::from(1u32 << 30)
            ),
            Self::Singular { what } => {
                write!(f, "singular linear system: {what}")
            }
            Self::StateSpaceTooLarge {
                k_regimes,
                order,
                max_states,
            } => {
                write!(
                    f,
                    "k_regimes = {k_regimes} and order = {order}: the Hamilton filter tracks \
                     the last order + 1 regimes jointly, k_regimes^(order + 1) states, which \
                     exceeds the cap of {max_states} states (checked before anything is \
                     allocated); "
                )?;
                let k_max = max_regimes_at_order(*order, *max_states);
                if k_max >= 2 {
                    write!(
                        f,
                        "at order = {order} the largest admissible k_regimes is {k_max}"
                    )?;
                } else {
                    write!(f, "no k_regimes >= 2 is admissible at order = {order}")?;
                }
                // Only an order >= 1 is offered: `fit` (and so
                // markov_switching_ar) refuses order = 0.
                match max_order_at_regimes(*k_regimes, *max_states) {
                    Some(p) if p >= 1 => write!(
                        f,
                        ", and at k_regimes = {k_regimes} the largest admissible order is {p}"
                    ),
                    _ => {
                        write!(
                            f,
                            ", and no order >= 1 is admissible at k_regimes = {k_regimes}"
                        )?;
                        // The bound at order = 1 is already stated when
                        // order = 1 is the order requested.
                        if *order != 1 {
                            write!(
                                f,
                                " (order = 1 admits at most {} regimes)",
                                max_regimes_at_order(1, *max_states)
                            )?;
                        }
                        Ok(())
                    }
                }
            }
            Self::GridTooLarge {
                n_gamma,
                n_c,
                max_cells,
            } => {
                write!(
                    f,
                    "n_gamma = {n_gamma} and n_c = {n_c}: the (gamma, c) grid search would \
                     evaluate n_gamma * n_c = {} cells per candidate delay, beyond its budget \
                     of {max_cells} (2^20) cells (checked before the grid is allocated); ",
                    (*n_gamma as u128) * (*n_c as u128)
                )?;
                let (g_max, c_max) = (max_cells / n_c.max(&1), max_cells / n_gamma.max(&1));
                match (g_max >= 2, c_max >= 2) {
                    (true, true) => write!(
                        f,
                        "at n_c = {n_c} the largest admissible n_gamma is {g_max}, and at \
                         n_gamma = {n_gamma} the largest admissible n_c is {c_max}"
                    ),
                    (true, false) => write!(
                        f,
                        "at n_c = {n_c} the largest admissible n_gamma is {g_max} (n_gamma = \
                         {n_gamma} leaves no room for n_c >= 2)"
                    ),
                    (false, true) => write!(
                        f,
                        "at n_gamma = {n_gamma} the largest admissible n_c is {c_max} (n_c = \
                         {n_c} leaves no room for n_gamma >= 2)"
                    ),
                    (false, false) => write!(
                        f,
                        "reduce both: each may be at most {} when the other is 2 (the \
                         default is 25 x 25)",
                        max_cells / 2
                    ),
                }
            }
        }
    }
}

impl std::error::Error for RegimeError {}
