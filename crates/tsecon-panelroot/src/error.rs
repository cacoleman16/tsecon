//! Error type shared by the panel unit-root tests.
//!
//! Every fallible public function in this crate returns
//! `Result<_, PanelRootError>`; nothing outside `#[cfg(test)]` panics on user
//! input. Messages follow the library's "errors that teach" pillar: they
//! state what went wrong, why it matters statistically, and what the caller
//! can do about it.

use core::fmt;

use tsecon_diag::DiagError;
use tsecon_hac::HacError;
use tsecon_stats::StatsError;

/// Errors produced by the panel unit-root tests in this crate.
#[derive(Debug, Clone, PartialEq)]
pub enum PanelRootError {
    /// A panel needs at least two cross-section units: the combination
    /// statistics (t-bar, `-2 sum ln p`, the pooled ADF) are averages or
    /// sums over units and are undefined for `n < 2`.
    TooFewUnits {
        /// The number of units supplied.
        n: usize,
    },
    /// The Levin-Lin-Chu test requires a balanced panel (a common length
    /// `T`) so its `T`-bar bias adjustment and pooled regression are well
    /// defined; unit `unit` broke that.
    UnbalancedForLlc {
        /// The offending unit (0-indexed).
        unit: usize,
        /// The length shared by the preceding units.
        expected: usize,
        /// The length of the offending unit.
        got: usize,
    },
    /// A single unit's series was too short for even the differencing and
    /// trimming the requested specification needs.
    UnitTooShort {
        /// The offending unit (0-indexed).
        unit: usize,
        /// The length of that unit's series.
        len: usize,
    },
    /// The Im-Pesaran-Shin test is undefined without a deterministic term:
    /// its standardization moments are tabulated only for the intercept and
    /// the intercept-plus-trend cases (Im-Pesaran-Shin 2003, Table 3). Use
    /// `"c"` or `"ct"`, or switch to `"llc"`/`"fisher"` for the no-constant
    /// case.
    IpsNoConstant,
    /// An input series contained a NaN or infinite value in the named unit.
    /// Panel tests never skip missing values silently.
    NonFinite {
        /// The offending unit (0-indexed).
        unit: usize,
    },
    /// The per-unit augmented Dickey-Fuller test failed for one unit; the
    /// underlying [`DiagError`] carries the specifics.
    Adf {
        /// The offending unit (0-indexed).
        unit: usize,
        /// The error propagated from `tsecon-diag`.
        source: DiagError,
    },
    /// An auxiliary regression or the long-run-variance step in the
    /// Levin-Lin-Chu pipeline failed for one unit.
    Hac {
        /// The offending unit (0-indexed).
        unit: usize,
        /// The error propagated from `tsecon-hac`.
        source: HacError,
    },
    /// The pooled Levin-Lin-Chu regression degenerated: the projected
    /// lagged level has zero variation across the whole panel, so its
    /// coefficient and t-ratio are undefined.
    DegeneratePool,
    /// An error propagated from the `tsecon-stats` distribution layer (the
    /// normal CDF / chi-squared survival function used for p-values).
    Stats(StatsError),
}

impl fmt::Display for PanelRootError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooFewUnits { n } => write!(
                f,
                "panel unit-root tests need at least 2 cross-section units, \
                 got {n}; the combined statistic is an average/sum over units \
                 (supply the panel as an N x T array of rows, or a list of \
                 per-unit series)"
            ),
            Self::UnbalancedForLlc {
                unit,
                expected,
                got,
            } => write!(
                f,
                "test = \"llc\" requires a balanced panel: unit {unit} has \
                 length {got} but the earlier units have length {expected}; \
                 the Levin-Lin-Chu bias adjustment is tabulated by a common \
                 T. Use test = \"ips\" or \"fisher\", which accept unbalanced \
                 panels, or trim every unit to a common length"
            ),
            Self::UnitTooShort { unit, len } => write!(
                f,
                "unit {unit} has only {len} observations — too few to \
                 difference, lag, and fit the test regression; drop the unit \
                 or supply a longer series"
            ),
            Self::IpsNoConstant => write!(
                f,
                "regression = \"n\" (no deterministics) is not valid for the \
                 Im-Pesaran-Shin test: its t-bar standardization moments are \
                 tabulated only for the intercept (\"c\") and intercept+trend \
                 (\"ct\") cases (Im-Pesaran-Shin 2003, Table 3). Use \"c\" or \
                 \"ct\", or switch to \"llc\"/\"fisher\" for the mean-zero case"
            ),
            Self::NonFinite { unit } => write!(
                f,
                "unit {unit} contains a non-finite value (NaN or infinity); \
                 panel unit-root tests do not skip missing values silently — \
                 clean or impute the series first"
            ),
            Self::Adf { unit, source } => {
                write!(f, "per-unit ADF failed for unit {unit}: {source}")
            }
            Self::Hac { unit, source } => write!(
                f,
                "the Levin-Lin-Chu auxiliary regression / long-run variance \
                 failed for unit {unit}: {source}"
            ),
            Self::DegeneratePool => write!(
                f,
                "the pooled Levin-Lin-Chu regression is degenerate: the \
                 projected lagged levels have zero variation across the \
                 panel, so the common-root t-ratio is undefined; this needs \
                 genuinely stochastic, non-collinear series"
            ),
            Self::Stats(e) => write!(f, "distribution-layer error: {e}"),
        }
    }
}

impl std::error::Error for PanelRootError {}

impl From<StatsError> for PanelRootError {
    fn from(e: StatsError) -> Self {
        Self::Stats(e)
    }
}
