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
    /// The working set a simulation would allocate exceeds the engine's
    /// memory budget, or the allocator refused it (a refusal is an error,
    /// never a process abort). Carries the counts that multiplied into
    /// the request so the message can name them.
    MemoryBudget {
        /// Which engine and which parameters size the working set.
        what: &'static str,
        /// Bytes the request would need.
        bytes: u128,
        /// The engine's budget in bytes.
        budget: usize,
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
        /// Where the first offending entry sits, and why that one entry is
        /// fatal for this caller, when the check scanned a specific array.
        at: Option<NonFiniteCell>,
    },
    /// A matrix that must be symmetric positive definite (residual
    /// covariance, regressor cross-product, Wald middle matrix) failed
    /// its Cholesky factorization.
    NotPositiveDefinite {
        /// Name of the offending matrix.
        what: &'static str,
    },
    /// One bootstrap replication failed in a way the full sample did not,
    /// and the band cannot be formed without it. Carries the 0-based index
    /// of the FIRST failing replication (in replication order, so the same
    /// seed always names the same one) and what went wrong. Failed draws are
    /// never silently dropped: dropping exactly the draws that hit a boundary
    /// would narrow the band.
    Replication {
        /// 0-based index of the first failing replication.
        index: usize,
        /// What failed in that replication.
        what: &'static str,
    },
    /// The sample leaves fewer residual degrees of freedom than series:
    /// with `df_resid = T - m < k` the `T x k` residual matrix has rank at
    /// most `df_resid`, so `Sigma_u`, the `k x k` residual covariance, is
    /// singular whatever the data (the log-likelihood, the information
    /// criteria and every orthogonalisation need its determinant or its
    /// Cholesky factor). Refused before the fit, so the message names the
    /// sample size and not a collinear column.
    SingularResidualCovariance {
        /// Minimum number of rows required, `lags + m + k`.
        needed: usize,
        /// Number of rows available.
        got: usize,
        /// Lag order `p` that was requested.
        lags: usize,
        /// Number of series `k` in the system.
        neqs: usize,
        /// Number of deterministic terms per equation.
        n_trend: usize,
        /// The residual degrees of freedom `T - m` the sample leaves.
        df_resid: usize,
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

/// The position of the first non-finite entry a scan found, and why that
/// one entry is fatal for the caller (printed after the position, so each
/// caller explains its own consequence instead of borrowing another's).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NonFiniteCell {
    /// Zero-based row of the offending entry.
    pub row: usize,
    /// Zero-based column of the offending entry.
    pub col: usize,
    /// Why one non-finite entry is fatal here, and what to do about it.
    pub why: &'static str,
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
            Self::InvalidParameter {
                name,
                value,
                requirement,
            } => write!(f, "invalid {name} = {value}: requires {requirement}"),
            Self::NonFinite {
                what,
                at: Some(NonFiniteCell { row, col, why }),
            } => write!(
                f,
                "non-finite value (NaN or inf) in {what} at row {row}, column {col}: {why}"
            ),
            Self::NonFinite { what, at: None } => write!(
                f,
                "non-finite value (NaN or inf) in {what}: check the inputs for missing \
                 values or magnitudes large enough to overflow"
            ),
            Self::Replication { index, what } => write!(
                f,
                "bootstrap replication {index} (0-based, the first to fail; the same seed \
                 always fails at the same replication) could not be completed: {what}. \
                 Failed replications are not dropped, because dropping exactly the draws \
                 that hit a boundary would narrow the band; change seed or n_boot to see \
                 whether it is isolated, or report the bands with method=\"asymptotic\""
            ),
            Self::NotPositiveDefinite { what } => write!(
                f,
                "matrix is not positive definite: {what}; two columns are exact linear \
                 combinations — a duplicated series, a scaled copy of another series, or \
                 a column that is constant over the estimation sample. Drop the redundant \
                 column and refit."
            ),
            Self::SingularResidualCovariance {
                needed,
                got,
                lags,
                neqs,
                n_trend,
                df_resid,
            } => {
                let per_eq = n_trend + neqs * lags;
                write!(
                    f,
                    "Sigma_u, the residual covariance, is singular: data has {got} rows, and a \
                     VAR with lags = p = {lags} on its k={neqs} series has n_trend + k*lags = \
                     {per_eq} regressors per equation, which leaves df_resid = {df_resid} \
                     residual degrees of freedom — fewer than the k={neqs} series, so the \
                     k x k residual covariance has rank at most {df_resid} whatever the data \
                     (no column needs dropping). It needs at least {needed} rows. {}",
                    resid_df_hint(*needed, *got, *lags, *neqs, *n_trend)
                )
            }
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
                    "data has {got} rows but a VAR with lags = p = {lags} on its k={neqs} \
                     series needs at least {needed}: lagging consumes {lags} rows and each \
                     equation then has n_trend + k*lags = {per_eq} regressors, which leaves \
                     no residual degrees of freedom. {}",
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

/// Concrete "what to try" clause for
/// [`VarError::SingularResidualCovariance`]: the largest lag order whose
/// residual degrees of freedom reach `k` on the direct fit path (`needed ==
/// lags + n_trend + k lags + k`); `select_order`'s common subsample only
/// gets the generic advice.
fn resid_df_hint(needed: usize, got: usize, lags: usize, neqs: usize, n_trend: usize) -> String {
    if needed != lags + n_trend + neqs * lags + neqs {
        return "Reduce maxlags, drop a series, or supply a longer sample.".to_string();
    }
    let p_max = got.saturating_sub(n_trend + neqs) / (neqs + 1);
    if p_max >= 1 {
        format!("Try lags <= {p_max}, drop a series, or supply a longer sample.")
    } else {
        let min_rows = n_trend + 2 * neqs + 1;
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
