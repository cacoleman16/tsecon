//! Error type shared by the functional-shock estimators.
//!
//! Every fallible public function returns `Result<_, FuncShockError>`;
//! nothing outside `#[cfg(test)]` panics on user input. Messages follow the
//! library's "errors that teach" pillar: they say what went wrong, why it
//! matters, and what the caller can do about it.

use core::fmt;

use tsecon_hac::HacError;
use tsecon_var::VarError;

/// Errors produced by the functional PCA, the functional local projection,
/// the scenario reconstruction, and the FVAR scenario.
#[derive(Debug, Clone, PartialEq)]
pub enum FuncShockError {
    /// A required input was empty.
    EmptyInput {
        /// Name of the offending argument.
        what: &'static str,
    },
    /// A row of the curve panel (or score matrix) has a different length
    /// from the first row — the observations do not share one grid.
    RaggedRow {
        /// Which matrix argument was ragged.
        what: &'static str,
        /// 0-indexed offending row.
        row: usize,
        /// Length of the first row (the grid every row must share).
        expected: usize,
        /// Length of the offending row.
        got: usize,
    },
    /// An input contained a NaN or infinite entry.
    NonFinite {
        /// Name of the offending argument.
        what: &'static str,
    },
    /// The requested number of functional principal components is zero or
    /// exceeds the number of grid points `M`.
    InvalidFactorCount {
        /// The requested `n_factors`.
        requested: usize,
        /// The maximum allowed (`M`, the number of grid points).
        max: usize,
    },
    /// Two inputs that must share a dimension do not (e.g. `y` vs the score
    /// rows, a scenario curve vs the eigenfunction grid, weights vs `K`).
    DimensionMismatch {
        /// Description of the constraint that was violated.
        what: &'static str,
        /// The length that was expected.
        expected: usize,
        /// The length that was received.
        got: usize,
    },
    /// The curve panel has zero total variance (every day's curve is
    /// identical), so the covariance has no principal components and the
    /// explained-variance shares are `0/0`.
    ZeroVariance,
    /// Too few time periods for the request: fewer than 2 curves for a
    /// covariance, or a local-projection horizon whose usable sample cannot
    /// identify the regression.
    HorizonTooLong {
        /// The offending horizon.
        horizon: usize,
        /// Usable observations at that horizon (`T - horizon - n_lag_controls`).
        nobs: usize,
        /// Parameters the horizon regression must estimate
        /// (`1 + K + n_lag_controls`).
        nparams: usize,
    },
    /// The outcome series is too short for the requested lag controls.
    SeriesTooShort {
        /// The sample size `T`.
        n: usize,
        /// The requested number of lagged-outcome controls.
        n_lag_controls: usize,
    },
    /// The symmetric eigendecomposition of the curve covariance failed to
    /// converge (pathological input; should not occur on finite data).
    EigenFailed,
    /// A scenario variance `w' Cov w` came out materially negative — the
    /// supplied covariance is not positive semi-definite (e.g. a truncated
    /// kernel or a hand-edited matrix).
    NegativeVariance {
        /// The horizon at which the quadratic form went negative.
        horizon: usize,
        /// The offending value of `w' Cov w`.
        value: f64,
    },
    /// An error propagated from the OLS/HAC engine ([`tsecon_hac`]).
    Hac(HacError),
    /// An error propagated from the VAR engine ([`tsecon_var`]).
    Var(VarError),
}

impl fmt::Display for FuncShockError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyInput { what } => write!(
                f,
                "empty input: {what}; supply at least one observation \
                 (curves are rows of a T x M panel on a shared grid)"
            ),
            Self::RaggedRow {
                what,
                row,
                expected,
                got,
            } => write!(
                f,
                "ragged input in {what}: row {row} has length {got} but row 0 has \
                 length {expected}; every observation must live on the same grid — \
                 interpolate the curves onto a common set of maturities first"
            ),
            Self::NonFinite { what } => write!(
                f,
                "non-finite value (NaN or infinity) in {what}; the estimators do \
                 not skip missing values silently — clean or interpolate the data \
                 first"
            ),
            Self::InvalidFactorCount { requested, max } => write!(
                f,
                "invalid n_factors {requested}: the number of functional principal \
                 components must satisfy 1 <= n_factors <= {max} (the number of \
                 grid points M); a curve observed on M points has at most M \
                 eigenfunctions"
            ),
            Self::DimensionMismatch {
                what,
                expected,
                got,
            } => write!(
                f,
                "dimension mismatch: {what} (expected length {expected}, got {got})"
            ),
            Self::ZeroVariance => write!(
                f,
                "zero variance: every curve in the panel is identical, so the \
                 covariance has no principal components and explained-variance \
                 shares are undefined; functional PCA needs curves that move"
            ),
            Self::HorizonTooLong {
                horizon,
                nobs,
                nparams,
            } => write!(
                f,
                "horizon {horizon} leaves only {nobs} usable observations for a \
                 regression with {nparams} parameters (constant + K scores + lag \
                 controls); shorten the horizon, reduce n_lag_controls, or supply \
                 a longer sample"
            ),
            Self::SeriesTooShort { n, n_lag_controls } => write!(
                f,
                "series too short: T = {n} observations with n_lag_controls = \
                 {n_lag_controls} leaves no usable sample; supply a longer series \
                 or fewer lag controls"
            ),
            Self::EigenFailed => write!(
                f,
                "the symmetric eigendecomposition of the curve covariance did not \
                 converge; this indicates pathological input — check the panel for \
                 extreme values"
            ),
            Self::NegativeVariance { horizon, value } => write!(
                f,
                "negative scenario variance {value:.3e} at horizon {horizon}: the \
                 supplied coefficient covariance is not positive semi-definite; \
                 use the covariance matrices exactly as returned by flp (Bartlett \
                 HAC is PSD by construction)"
            ),
            Self::Hac(e) => write!(f, "OLS/HAC-engine error: {e}"),
            Self::Var(e) => write!(f, "VAR-engine error: {e}"),
        }
    }
}

impl std::error::Error for FuncShockError {}

impl From<HacError> for FuncShockError {
    fn from(e: HacError) -> Self {
        Self::Hac(e)
    }
}

impl From<VarError> for FuncShockError {
    fn from(e: VarError) -> Self {
        Self::Var(e)
    }
}
