//! Error types for `tsecon-filters`.

use core::fmt;

/// Errors produced by the trend-cycle filters in this crate.
///
/// All fallible library entry points return `Result<_, FiltersError>`;
/// nothing in the non-test code path panics.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FiltersError {
    /// A filter parameter is outside its valid domain (e.g. `lambda < 0`
    /// for the Hodrick-Prescott filter, or `low < 2` for a band-pass
    /// filter, which would place the upper band edge above the Nyquist
    /// frequency).
    InvalidParameter {
        /// Name of the offending parameter.
        name: &'static str,
        /// The invalid value that was supplied.
        value: f64,
        /// Human-readable statement of the violated constraint.
        requirement: &'static str,
    },
    /// The input series is too short for the requested filter (e.g. a
    /// Baxter-King filter with truncation `K` needs at least `2K + 1`
    /// observations to produce a single output point).
    SeriesTooShort {
        /// Name of the filter that rejected the series.
        filter: &'static str,
        /// Minimum number of observations required.
        needed: usize,
        /// Number of observations supplied.
        got: usize,
    },
    /// The input series contains a NaN or infinite value. Filters propagate
    /// non-finite values in surprising, filter-specific ways (a single NaN
    /// contaminates the entire Hodrick-Prescott trend), so they are
    /// rejected up front.
    NonFiniteInput {
        /// Index of the first non-finite observation.
        index: usize,
    },
    /// The regressor matrix of the Hamilton (2018) regression filter is
    /// numerically rank deficient (e.g. the series is constant, so every
    /// lag column is collinear with the intercept).
    RankDeficient {
        /// Name of the computation that encountered the deficiency.
        what: &'static str,
    },
}

impl fmt::Display for FiltersError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FiltersError::InvalidParameter {
                name,
                value,
                requirement,
            } => write!(
                f,
                "invalid parameter `{name}` = {value}: requires {requirement}"
            ),
            FiltersError::SeriesTooShort {
                filter,
                needed,
                got,
            } => write!(
                f,
                "{filter} requires at least {needed} observations, got {got}"
            ),
            FiltersError::NonFiniteInput { index } => {
                write!(f, "input series has a non-finite value at index {index}")
            }
            FiltersError::RankDeficient { what } => {
                write!(f, "{what}: regressor matrix is numerically rank deficient")
            }
        }
    }
}

impl std::error::Error for FiltersError {}

/// Reject series containing NaN or infinities, returning the index of the
/// first offender.
pub(crate) fn check_finite(y: &[f64]) -> Result<(), FiltersError> {
    match y.iter().position(|v| !v.is_finite()) {
        Some(index) => Err(FiltersError::NonFiniteInput { index }),
        None => Ok(()),
    }
}
