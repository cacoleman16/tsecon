//! Error type shared by the cointegration and VECM layer.

use core::fmt;

use tsecon_diag::DiagError;
use tsecon_linalg::LinalgError;
use tsecon_stats::StatsError;

/// Errors returned by the cointegration / VECM layer.
///
/// Every fallible public function in this crate returns
/// `Result<_, CointError>`; no library code path panics on user input.
#[derive(Debug, Clone, PartialEq)]
pub enum CointError {
    /// An error bubbled up from the structured linear-algebra layer
    /// (Cholesky, eigenvalues, ...).
    Linalg(LinalgError),
    /// An error bubbled up from the special-function layer.
    Stats(StatsError),
    /// An error bubbled up from the diagnostics layer (the augmented
    /// Dickey-Fuller step of Engle-Granger).
    Diag(DiagError),
    /// Two inputs (or an input and a model dimension) have incompatible
    /// sizes.
    Dimension {
        /// Description of the constraint that was violated.
        what: &'static str,
        /// The size that was expected.
        expected: usize,
        /// The size that was received.
        got: usize,
    },
    /// A scalar or structural argument was outside its valid domain.
    InvalidArgument {
        /// Description of the domain violation.
        what: &'static str,
    },
    /// An input contained a NaN or infinity.
    NonFinite {
        /// Name of the offending argument.
        what: &'static str,
        /// Zero-based `(row, column)` of the first offending entry.
        at: Option<(usize, usize)>,
    },
    /// A matrix that must be symmetric positive definite (a residual
    /// second-moment matrix `S_00`, `S_11`, ...) failed its Cholesky
    /// factorization: the auxiliary regressors are collinear or the
    /// sample is degenerate.
    NotPositiveDefinite {
        /// Name of the offending matrix.
        what: &'static str,
    },
    /// A square matrix that had to be inverted (a normalizing
    /// `beta[:r, :r]` block, an `S_00`, ...) was numerically singular.
    Singular {
        /// Name of the offending matrix.
        what: &'static str,
    },
    /// The requested cointegration rank is outside `0 ..= k` (`k` the
    /// number of series). Rank `0` is no cointegration (a VAR in
    /// differences); rank `k` is a stationary level VAR.
    InvalidRank {
        /// The rank that was requested.
        rank: usize,
        /// The number of series `k` (the maximum admissible rank).
        neqs: usize,
    },
    /// The sample is too short for the requested specification: the
    /// Johansen auxiliary regressions need strictly more usable rows than
    /// short-run regressors.
    InsufficientObservations {
        /// Minimum number of usable rows required after differencing.
        needed: usize,
        /// Number of usable rows available after differencing.
        got: usize,
        /// Number of rows in the data as it was supplied.
        nobs: usize,
        /// Number of series `k` in the system.
        neqs: usize,
        /// Lagged-difference order `k_ar_diff` that was requested.
        k_ar_diff: usize,
    },
}

impl fmt::Display for CointError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Linalg(e) => write!(f, "linear algebra failure: {e}"),
            Self::Stats(e) => write!(f, "special-function failure: {e}"),
            Self::Diag(e) => write!(f, "diagnostics failure: {e}"),
            Self::Dimension {
                what,
                expected,
                got,
            } => write!(f, "{what} (expected {expected}, got {got})"),
            Self::InvalidArgument { what } => write!(f, "{what}"),
            Self::NonFinite {
                what,
                at: Some((row, col)),
            } => write!(
                f,
                "non-finite value (NaN or inf) in {what} at row {row}, column {col}: \
                 the cointegration estimators have no missing-value handling, so one gap \
                 would corrupt every eigenvalue and test statistic — drop or impute those \
                 rows first (pandas: df.dropna())"
            ),
            Self::NonFinite { what, at: None } => write!(
                f,
                "non-finite value (NaN or inf) in {what}: check the inputs for missing \
                 values or magnitudes large enough to overflow"
            ),
            Self::NotPositiveDefinite { what } => write!(
                f,
                "matrix is not positive definite: {what}; two of the series are exact \
                 linear combinations — a duplicated column, a scaled copy of another \
                 series, or a column that is constant over the sample. Drop the redundant \
                 series and refit."
            ),
            Self::Singular { what } => write!(f, "numerically singular: {what}"),
            Self::InvalidRank { rank, neqs } => write!(
                f,
                "invalid cointegration rank {rank}: must satisfy 0 <= rank <= {neqs}, the \
                 number of series (rank 0 is a VAR in differences, rank {neqs} a \
                 stationary level VAR); run johansen() first and use its selected rank"
            ),
            Self::InsufficientObservations {
                needed,
                got,
                nobs,
                neqs,
                k_ar_diff,
            } => {
                let per_eq = neqs * k_ar_diff + neqs;
                write!(
                    f,
                    "the Johansen/VECM auxiliary regressions need at least {needed} usable \
                     rows but only {got} of the {nobs} rows supplied survive one difference \
                     plus {k_ar_diff} presample lag(s): with k={neqs} series each regression \
                     has k*k_ar_diff + k = {per_eq} regressors. {}",
                    k_ar_diff_hint(*nobs, *neqs)
                )
            }
        }
    }
}

/// Concrete "what to try" clause for
/// [`CointError::InsufficientObservations`].
///
/// Both entry points build the same effective sample `t = n - 1 -
/// k_ar_diff` and need `t > k * k_ar_diff + k`, i.e.
/// `n >= k_ar_diff * (k + 1) + k + 2`. Inverting that bound gives the
/// largest lagged-difference order this sample can support.
fn k_ar_diff_hint(nobs: usize, neqs: usize) -> String {
    let d_max = nobs.saturating_sub(neqs + 2) / (neqs + 1);
    if d_max >= 1 {
        format!("Try k_ar_diff <= {d_max}, drop a series, or supply a longer sample.")
    } else if nobs >= neqs + 2 {
        format!(
            "Only k_ar_diff = 0 fits {nobs} input rows on k={neqs} series; each extra \
             lagged difference costs {} more input rows.",
            neqs + 1
        )
    } else {
        format!(
            "Even k_ar_diff = 0 would need {} input rows on k={neqs} series, so supply a \
             longer sample or fit fewer series.",
            neqs + 2
        )
    }
}

impl std::error::Error for CointError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Linalg(e) => Some(e),
            Self::Stats(e) => Some(e),
            Self::Diag(e) => Some(e),
            _ => None,
        }
    }
}

impl From<LinalgError> for CointError {
    fn from(e: LinalgError) -> Self {
        Self::Linalg(e)
    }
}

impl From<StatsError> for CointError {
    fn from(e: StatsError) -> Self {
        Self::Stats(e)
    }
}

impl From<DiagError> for CointError {
    fn from(e: DiagError) -> Self {
        Self::Diag(e)
    }
}
