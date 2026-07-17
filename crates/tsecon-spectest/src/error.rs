//! Error type shared by every specification / diagnostic test in the crate.
//!
//! Every fallible public function returns `Result<_, SpecTestError>`; nothing
//! outside `#[cfg(test)]` panics on user input. Messages follow the library's
//! "errors that teach" pillar: they say what went wrong, why it matters, and
//! what the caller can do about it.

use core::fmt;

use tsecon_hac::HacError;
use tsecon_stats::StatsError;

/// Errors produced by the White, Breusch-Pagan, RESET, Chow, and CUSUM tests.
#[derive(Debug, Clone, PartialEq)]
pub enum SpecTestError {
    /// A required series was empty.
    EmptyInput {
        /// Name of the offending argument.
        what: &'static str,
    },
    /// The design has no columns (no regressors, not even a constant).
    NoRegressors,
    /// The response `y` and a design column, or two design columns, have
    /// incompatible lengths.
    DimensionMismatch {
        /// Description of the constraint that was violated.
        what: &'static str,
        /// The length that was expected.
        expected: usize,
        /// The length that was received.
        got: usize,
    },
    /// An input contained a NaN or infinite entry.
    NonFinite {
        /// Name of the offending argument.
        what: &'static str,
    },
    /// Fewer usable observations than regressors (`n <= k`): the regression has
    /// no residual degrees of freedom, so the test statistic is undefined.
    DegreesOfFreedom {
        /// The description of the regression that ran short.
        what: &'static str,
        /// The sample size `n`.
        n: usize,
        /// The number of regressors `k` the fit needs `n` to exceed.
        k: usize,
    },
    /// The design has no constant (intercept) column. The White and
    /// Breusch-Pagan auxiliary regressions form a *centered* `R^2`, which is
    /// only the LM statistic when the auxiliary design contains a constant;
    /// this crate keeps the statsmodels convention of an explicit intercept
    /// column rather than silently adding one.
    MissingConstant {
        /// The test that requires the intercept.
        what: &'static str,
    },
    /// The auxiliary/regression response had zero total sum of squares (every
    /// value identical), so a centered `R^2` is `0/0` and the LM statistic is
    /// undefined — a degenerate input, e.g. residuals that are all equal.
    DegenerateResponse {
        /// Which regression produced the degenerate response.
        what: &'static str,
    },
    /// The Chow split point does not leave each sub-sample with more
    /// observations than regressors. A valid known split needs
    /// `k < split < n - k` so both regimes are estimable and
    /// `n - 2k` denominator degrees of freedom remain.
    InvalidSplit {
        /// The requested 0-indexed split (first regime is `0..split`).
        split: usize,
        /// The sample size `n`.
        n: usize,
        /// The number of regressors `k`.
        k: usize,
    },
    /// The RESET maximum fitted-value power was below 2; RESET adds the powers
    /// `yhat^2 .. yhat^max_power`, so it needs `max_power >= 2` to add any term.
    InvalidPower {
        /// The offending `max_power`.
        max_power: usize,
    },
    /// A design (original, auxiliary, or a recursive-residual window) was rank
    /// deficient — collinear columns — so the normal equations have no unique
    /// solution.
    SingularDesign {
        /// Which regression's design was singular.
        what: &'static str,
    },
    /// An error propagated from the OLS engine ([`tsecon_hac`]).
    Hac(HacError),
    /// An error propagated from the statistics layer ([`tsecon_stats`]:
    /// chi-square / F tails).
    Stats(StatsError),
}

impl fmt::Display for SpecTestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyInput { what } => write!(
                f,
                "empty input: {what}; supply a response and at least one design column"
            ),
            Self::NoRegressors => write!(
                f,
                "no regressors: the design has zero columns; include at least a \
                 constant column (a column of ones) so the model has an intercept"
            ),
            Self::DimensionMismatch {
                what,
                expected,
                got,
            } => write!(
                f,
                "dimension mismatch: {what} (expected length {expected}, got {got})"
            ),
            Self::NonFinite { what } => write!(
                f,
                "non-finite value (NaN or infinity) in {what}; the tests do not skip \
                 missing values silently — clean the data first"
            ),
            Self::DegreesOfFreedom { what, n, k } => write!(
                f,
                "insufficient degrees of freedom for the {what}: n = {n} observations \
                 with k = {k} regressors requires n > k; supply a longer series or a \
                 smaller design"
            ),
            Self::MissingConstant { what } => write!(
                f,
                "the {what} needs an explicit constant (intercept) column in the \
                 design so its auxiliary regression forms a centered R^2; add a \
                 column of ones (statsmodels exog convention)"
            ),
            Self::DegenerateResponse { what } => write!(
                f,
                "degenerate response in the {what}: the regressand has zero total \
                 sum of squares (all values identical), so the centered R^2 and the \
                 LM statistic are undefined"
            ),
            Self::InvalidSplit { split, n, k } => write!(
                f,
                "invalid Chow split {split} for n = {n}, k = {k}: a known split needs \
                 k < split < n - k so both sub-samples are estimable and n - 2k \
                 denominator degrees of freedom remain"
            ),
            Self::InvalidPower { max_power } => write!(
                f,
                "invalid RESET max_power {max_power}: RESET adds fitted-value powers \
                 yhat^2 .. yhat^max_power, so it requires max_power >= 2"
            ),
            Self::SingularDesign { what } => write!(
                f,
                "singular design in the {what}: the columns are collinear (rank \
                 deficient), so the normal equations have no unique solution — drop a \
                 redundant regressor"
            ),
            Self::Hac(e) => write!(f, "OLS-engine error: {e}"),
            Self::Stats(e) => write!(f, "statistics-layer error: {e}"),
        }
    }
}

impl std::error::Error for SpecTestError {}

impl From<HacError> for SpecTestError {
    fn from(e: HacError) -> Self {
        match e {
            HacError::SingularDesign { what } => Self::SingularDesign { what },
            other => Self::Hac(other),
        }
    }
}

impl From<StatsError> for SpecTestError {
    fn from(e: StatsError) -> Self {
        Self::Stats(e)
    }
}
