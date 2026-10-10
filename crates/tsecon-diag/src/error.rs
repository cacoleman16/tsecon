//! Error types for `tsecon-diag`.
//!
//! Every fallible entry point in this crate returns `Result<_, DiagError>`;
//! nothing in the non-test code path panics. Error messages follow the
//! library's "errors that teach" pillar: they state what went wrong, why it
//! matters statistically, and what the caller can do about it.

use core::fmt;

use tsecon_stats::StatsError;

/// Errors produced by the diagnostic statistics in this crate.
#[derive(Debug, Clone, PartialEq)]
pub enum DiagError {
    /// The series has too few observations for the requested computation.
    SeriesTooShort {
        /// Which diagnostic needed more data.
        what: &'static str,
        /// The number of observations supplied.
        n: usize,
        /// The minimum number of observations required.
        needed: usize,
    },
    /// The requested number of lags is outside the valid range for a series
    /// of this length.
    InvalidLags {
        /// Which diagnostic rejected the lag count.
        what: &'static str,
        /// The lag count that was supplied.
        nlags: usize,
        /// The number of observations supplied.
        n: usize,
        /// Human-readable statement of the violated constraint.
        requirement: &'static str,
    },
    /// The input contains a NaN or infinite value. Diagnostics never skip
    /// missing values silently; clean or impute the series first.
    NonFinite {
        /// Index of the first offending observation.
        index: usize,
        /// The offending value.
        value: f64,
    },
    /// The series is (numerically) constant, so its sample variance is zero
    /// and correlation-based diagnostics are undefined.
    ConstantSeries {
        /// Which diagnostic found the degenerate series.
        what: &'static str,
    },
    /// The regressor cross-product matrix of an internal OLS step is not
    /// positive definite (collinear or degenerate lag matrix).
    SingularDesign {
        /// Which diagnostic hit the singular design.
        what: &'static str,
    },
    /// A numerical invariant that holds in exact arithmetic (e.g. positive
    /// innovation variance in the Durbin-Levinson recursion) broke down.
    NumericalBreakdown {
        /// Which algorithm broke down.
        what: &'static str,
    },
    /// The significance level `alpha` passed to a report is outside (0, 1).
    InvalidAlpha {
        /// The offending value.
        value: f64,
    },
    /// The trimming fraction passed to a break-search test is outside
    /// `[0, 1/3]`.
    InvalidTrim {
        /// The offending value.
        value: f64,
    },
    /// An argument combination or input that the statistic cannot use, with
    /// a self-contained teaching message (e.g. a repeated horizon, an option
    /// that has no effect under the chosen estimator, or increments with
    /// zero variance about the drift).
    InvalidArgument {
        /// The full message: what was wrong and what to do instead.
        what: &'static str,
    },
    /// A lag order — fixed, or the cap of an automatic search, supplied or
    /// defaulted — that the series cannot carry, or whose lag design would
    /// exceed the size budgets ([`crate::LAG_DESIGN_MAX_CELLS`],
    /// [`crate::LAG_WORK_BUDGET`]). Refused before any lag column is built.
    LagOrderTooLong {
        /// Which diagnostic refused the order.
        what: &'static str,
        /// The name of the order (`"nlags"`, `"maxlag"`, `"lags"`); the
        /// Python bindings rename it to the keyword the caller passed.
        param: &'static str,
        /// The order asked for.
        value: usize,
        /// `true` when the order came from the default rule, not the caller.
        defaulted: bool,
        /// The number of observations supplied.
        n: usize,
        /// The largest admissible order, or `None` when no order fits.
        max: Option<usize>,
        /// Which limit binds.
        limit: LagLimit,
    },
    /// An error propagated from the `tsecon-stats` special functions (e.g.
    /// the chi-squared survival function used for p-values).
    Stats(StatsError),
}

/// The limit a [`DiagError::LagOrderTooLong`] order runs into.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LagLimit {
    /// The sample-length bound.
    Sample {
        /// The bound's statement, in terms of the order `p`.
        rule: &'static str,
        /// The longest order the sample admits (before the size budgets),
        /// or `None` when none does.
        sample_max: Option<usize>,
    },
    /// The size budgets: the lag design's entries and the multiply-adds of
    /// its regression (or lag search) that the order asks for.
    Budget {
        /// Entries of the lag design.
        cells: u128,
        /// Multiply-adds of the regression or lag search.
        work: u128,
    },
}

impl fmt::Display for DiagError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DiagError::SeriesTooShort { what, n, needed } => write!(
                f,
                "{what}: series has {n} observations but needs at least {needed}; \
                 supply more data or reduce the requested lag order"
            ),
            DiagError::InvalidLags {
                what,
                nlags,
                n,
                requirement,
            } => write!(
                f,
                "{what}: nlags = {nlags} is invalid for a series of length {n}: \
                 requires {requirement}"
            ),
            DiagError::NonFinite { index, value } => write!(
                f,
                "input contains a non-finite value ({value}) at index {index}; \
                 diagnostics do not skip missing values silently — drop or impute \
                 NaN/inf observations before testing"
            ),
            DiagError::ConstantSeries { what } => write!(
                f,
                "{what}: the series is constant (zero sample variance), so \
                 correlation-based diagnostics are undefined; check that the \
                 right column was passed and that differencing did not remove \
                 all variation"
            ),
            DiagError::SingularDesign { what } => write!(
                f,
                "{what}: the lag regressor matrix is numerically singular \
                 (collinear lags); this usually means the series is (near-)\
                 deterministic or far too short for the requested lag order"
            ),
            DiagError::NumericalBreakdown { what } => write!(
                f,
                "{what}: numerical breakdown — an invariant that holds in exact \
                 arithmetic failed; this indicates a (near-)degenerate series"
            ),
            DiagError::InvalidAlpha { value } => write!(
                f,
                "significance level alpha = {value} is invalid: requires \
                 0 < alpha < 1 (conventional choices are 0.01, 0.05, 0.10)"
            ),
            DiagError::InvalidTrim { value } => write!(
                f,
                "trim = {value} is invalid: requires 0 <= trim <= 1/3. The \
                 trim fraction excludes the first and last int(n*trim) \
                 observations from the break search — a break too close to \
                 either end cannot be told apart from the sample boundary \
                 (the conventional choice is 0.15)"
            ),
            DiagError::InvalidArgument { what } => write!(f, "{what}"),
            DiagError::LagOrderTooLong {
                what,
                param,
                value,
                defaulted,
                n,
                max,
                limit,
            } => {
                let order = if *defaulted {
                    format!("the default {param} = {value}")
                } else {
                    format!("{param} = {value}")
                };
                match limit {
                    LagLimit::Sample { rule, .. } => write!(
                        f,
                        "{what}: {order} is too long for a series of length {n}: {rule}"
                    )?,
                    LagLimit::Budget { cells, work } => write!(
                        f,
                        "{what}: {order} needs a lag design of {cells} entries and about \
                         {:.2e} multiply-adds for a series of length {n}, past {} \
                         (checked before any lag column is built)",
                        *work as f64,
                        crate::budget::lag_budget_exceeded(*cells, *work).unwrap_or("the budget"),
                    )?,
                }
                match (max, limit) {
                    (Some(m), LagLimit::Sample { sample_max, .. }) => {
                        write!(f, "; pass {param} <= {m}")?;
                        match sample_max {
                            Some(s) if s > m => write!(
                                f,
                                " (orders {} to {s} fit the sample but exceed the size \
                                 budgets of 2^26 design entries and 2^35 multiply-adds)",
                                m + 1
                            ),
                            _ => Ok(()),
                        }
                    }
                    (Some(m), LagLimit::Budget { .. }) => {
                        write!(f, "; pass {param} <= {m}, or a shorter series")
                    }
                    (
                        None,
                        LagLimit::Sample {
                            sample_max: Some(_),
                            ..
                        },
                    ) => write!(
                        f,
                        "; not even the shortest {param} fits the size budgets of 2^26 \
                         design entries and 2^35 multiply-adds at this sample length, so \
                         supply a shorter series"
                    ),
                    (
                        None,
                        LagLimit::Sample {
                            sample_max: None, ..
                        },
                    ) => write!(f, "; no {param} fits this series, so supply a longer one"),
                    (None, LagLimit::Budget { .. }) => write!(
                        f,
                        "; not even the shortest {param} fits at this sample length, so \
                         supply a shorter series"
                    ),
                }
            }
            DiagError::Stats(e) => write!(f, "special-function error: {e}"),
        }
    }
}

impl std::error::Error for DiagError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            DiagError::Stats(e) => Some(e),
            _ => None,
        }
    }
}

impl From<StatsError> for DiagError {
    fn from(e: StatsError) -> Self {
        DiagError::Stats(e)
    }
}
