//! Error type shared by the VAR estimation and analysis layer.

use core::fmt;

use tsecon_linalg::LinalgError;
use tsecon_stats::StatsError;

/// Errors returned by the VAR layer.
///
/// Every fallible public function in this crate returns
/// `Result<_, VarError>`; no library code path panics on user input.
///
/// The `Display` impls are the text a Python caller sees, so each one
/// states what happened, why it most likely happened, and what to change.
#[derive(Debug, Clone, PartialEq)]
pub enum VarError {
    /// An error bubbled up from the structured linear-algebra layer
    /// (companion eigenvalues, Cholesky, ...).
    Linalg(LinalgError),
    /// An error bubbled up from the special-function layer (incomplete
    /// beta, inverse normal CDF, ...).
    Stats(StatsError),
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
    /// A structural argument was outside its valid domain.
    InvalidArgument {
        /// Description of the domain violation.
        what: &'static str,
    },
    /// A numeric argument was outside its admissible domain. Carries the
    /// offending value so the message can name it.
    InvalidParameter {
        /// Name of the offending parameter.
        name: &'static str,
        /// The value that was supplied.
        value: f64,
        /// Human-readable statement of the violated constraint.
        requirement: &'static str,
    },
    /// An input contained a NaN or infinity.
    NonFinite {
        /// Name of the offending argument.
        what: &'static str,
        /// Zero-based `(row, column)` of the first offending entry, when
        /// the check scanned a specific array.
        at: Option<(usize, usize)>,
    },
    /// A matrix that must be symmetric positive definite (residual
    /// covariance, regressor cross-product, Wald middle matrix) failed
    /// its Cholesky factorization.
    NotPositiveDefinite {
        /// Name of the offending matrix.
        what: &'static str,
    },
    /// The sample is too short for the requested specification: OLS
    /// needs strictly more usable observations than regressors per
    /// equation.
    InsufficientObservations {
        /// Minimum number of rows required.
        needed: usize,
        /// Number of rows available.
        got: usize,
        /// Lag order `p` that was requested.
        lags: usize,
        /// Number of series `k` in the system.
        neqs: usize,
        /// Number of deterministic terms per equation.
        n_trend: usize,
    },
}

impl fmt::Display for VarError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Linalg(e) => write!(f, "linear algebra failure: {e}"),
            Self::Stats(e) => write!(f, "special-function failure: {e}"),
            Self::Dimension {
                what,
                expected,
                got,
            } => write!(f, "{what} (expected {expected}, got {got})"),
            Self::InvalidArgument { what } => write!(f, "{what}"),
            Self::InvalidParameter {
                name,
                value,
                requirement,
            } => write!(f, "invalid {name} = {value}: requires {requirement}"),
            Self::NonFinite {
                what,
                at: Some((row, col)),
            } => write!(
                f,
                "non-finite value (NaN or inf) in {what} at row {row}, column {col}: \
                 the VAR estimator has no missing-value handling, so a single gap would \
                 corrupt every coefficient — drop or impute those rows first \
                 (pandas: df.dropna())"
            ),
            Self::NonFinite { what, at: None } => write!(
                f,
                "non-finite value (NaN or inf) in {what}: check the inputs for missing \
                 values or magnitudes large enough to overflow"
            ),
            Self::NotPositiveDefinite { what } => write!(
                f,
                "matrix is not positive definite: {what}; two columns are exact linear \
                 combinations — a duplicated series, a scaled copy of another series, or \
                 a column that is constant over the estimation sample. Drop the redundant \
                 column and refit."
            ),
            Self::InsufficientObservations {
                needed,
                got,
                lags,
                neqs,
                n_trend,
            } => {
                let per_eq = n_trend + neqs * lags;
                write!(
                    f,
                    "VAR({lags}) on k={neqs} series needs at least {needed} rows but got \
                     {got}: lagging consumes {lags} rows and each equation then has \
                     n_trend + k*lags = {per_eq} regressors, which leaves no residual \
                     degrees of freedom. {}",
                    lag_hint(*needed, *got, *lags, *neqs, *n_trend)
                )
            }
        }
    }
}

/// Concrete "what to try" clause for [`VarError::InsufficientObservations`].
///
/// The direct fit path uses `offset = 0`, in which case the requirement is
/// exactly `lags + (n_trend + k lags) + 1` and the largest estimable lag
/// order can be reported exactly. `select_order` fits candidates on a
/// common subsample (`offset > 0`), so the requirement no longer matches
/// that identity and only the generic advice is emitted.
fn lag_hint(needed: usize, got: usize, lags: usize, neqs: usize, n_trend: usize) -> String {
    if needed != lags + n_trend + neqs * lags + 1 {
        return "Reduce maxlags, drop a series, or supply a longer sample.".to_string();
    }
    let p_max = got.saturating_sub(n_trend + 1) / (neqs + 1);
    if p_max >= 1 {
        format!("Try lags <= {p_max}, drop a series, or supply a longer sample.")
    } else {
        let min_rows = n_trend + neqs + 2;
        format!(
            "Even lags=1 would need {min_rows} rows on k={neqs} series, so supply a longer \
             sample or fit fewer series."
        )
    }
}

impl std::error::Error for VarError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Linalg(e) => Some(e),
            Self::Stats(e) => Some(e),
            _ => None,
        }
    }
}

impl From<LinalgError> for VarError {
    fn from(e: LinalgError) -> Self {
        Self::Linalg(e)
    }
}

impl From<StatsError> for VarError {
    fn from(e: StatsError) -> Self {
        Self::Stats(e)
    }
}
