//! Error type shared by every specification / diagnostic test in the crate.
//!
//! Every fallible public function returns `Result<_, SpecTestError>`; nothing
//! outside `#[cfg(test)]` panics on user input. Messages follow the library's
//! "errors that teach" pillar: they say what went wrong, why it matters, and
//! what the caller can do about it.

use core::fmt;

use tsecon_hac::HacError;
use tsecon_stats::StatsError;

/// Errors produced by the White, Breusch-Pagan, RESET, Chow, CUSUM,
/// Breusch-Godfrey, and Durbin-Watson tests.
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
    /// The Breusch-Godfrey order was zero — explicitly, or as the default
    /// `min(10, n / 5)` on a sample shorter than five observations — so the
    /// auxiliary regression would add no lagged residual to test.
    InvalidLags {
        /// The offending order.
        nlags: usize,
        /// The sample size `n`.
        n: usize,
        /// `true` when the zero came from the default rule, not the caller.
        defaulted: bool,
    },
    /// The Breusch-Godfrey auxiliary design (`n x k_aux`: the design, the
    /// `nlags` lagged residuals and any added constant) would exceed the
    /// test's memory budget (`BG_MAX_AUX_CELLS` entries) or the work budget
    /// of its regression (`BG_WORK_BUDGET` multiply-adds); refused before
    /// any lag column is allocated.
    AuxiliaryTooLarge {
        /// The sample size `n`.
        n: usize,
        /// The number of auxiliary regressors the order asked for.
        k_aux: usize,
        /// The order that was asked for (or defaulted).
        nlags: usize,
        /// The largest order both budgets and `n > k_aux` admit (`0` when
        /// even one lag does not fit).
        max_nlags: usize,
        /// `true` when the order came from the default rule.
        defaulted: bool,
    },
    /// The sample is longer than an exact computation supports: the exact
    /// Durbin-Watson p-value eigen-decomposes an `n x n` matrix, which is
    /// cubic in `n`.
    SampleTooLong {
        /// The computation that is capped.
        what: &'static str,
        /// The sample size `n`.
        n: usize,
        /// The largest supported `n`.
        max: usize,
    },
    /// A numerical routine (an eigendecomposition or a quadrature) failed to
    /// reach its accuracy target.
    NumericalBreakdown {
        /// The routine that failed.
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
            Self::InvalidLags {
                nlags,
                n,
                defaulted,
            } => {
                if *defaulted {
                    write!(
                        f,
                        "invalid Breusch-Godfrey order: with n = {n} observations of y the \
                         default nlags = min(10, n // 5) is {nlags}; the test needs at least \
                         one lagged residual, so pass nlags >= 1 explicitly (keeping \
                         n > k + nlags) or supply a longer y"
                    )
                } else {
                    write!(
                        f,
                        "invalid Breusch-Godfrey order nlags = {nlags}: the test regresses the \
                         residuals on their own lags 1..nlags, so it needs nlags >= 1"
                    )
                }
            }
            Self::AuxiliaryTooLarge {
                n,
                k_aux,
                nlags,
                max_nlags,
                defaulted,
            } => {
                let order = if *defaulted {
                    format!("the default order nlags = min(10, n // 5) = {nlags}")
                } else {
                    format!("nlags = {nlags}")
                };
                write!(
                    f,
                    "Breusch-Godfrey auxiliary regression too large: with {order} it has \
                     n = {n} observations by k_aux = {k_aux} regressors ({} entries, about \
                     {:.2e} multiply-adds), past {}; ",
                    n.saturating_mul(*k_aux),
                    crate::breusch_godfrey::breusch_godfrey_work(*n, *k_aux) as f64,
                    crate::breusch_godfrey::breusch_godfrey_exceeded_budget(*n, *k_aux)
                        .unwrap_or("the budget"),
                )?;
                let one_lag = k_aux.saturating_sub(*nlags).saturating_add(1);
                if *max_nlags == 0
                    && crate::breusch_godfrey::breusch_godfrey_work(0, one_lag)
                        > crate::breusch_godfrey::BG_WORK_BUDGET
                {
                    write!(
                        f,
                        "even one lag does not fit with this design at any sample length \
                         (the 6 k_aux^3 term alone is past the work budget), so use fewer \
                         design columns"
                    )
                } else if *max_nlags == 0 {
                    write!(
                        f,
                        "even one lag does not fit with this design, so use a shorter sample \
                         or fewer design columns"
                    )
                } else {
                    write!(f, "pass nlags <= {max_nlags} or use a shorter sample")
                }
            }
            Self::SampleTooLong { what, n, max } => write!(
                f,
                "sample too long for the {what}: n = {n} exceeds the supported maximum of \
                 {max} (the exact null distribution eigen-decomposes an n x n matrix, which \
                 is cubic in n); for a long sample use the Breusch-Godfrey test, whose cost \
                 is linear in n at a fixed order"
            ),
            Self::NumericalBreakdown { what } => write!(
                f,
                "numerical breakdown in the {what}: the computation did not reach its \
                 accuracy target"
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
