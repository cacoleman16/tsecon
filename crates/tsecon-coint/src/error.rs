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
    /// A univariate series input contained a NaN or infinity. Unlike
    /// [`CointError::NonFinite`], whose located form carries the
    /// multivariate cointegration teaching text, the consequence and
    /// remedy here live in `what`, so each univariate surface (e.g. the
    /// OU utilities) states its own.
    NonFiniteSeries {
        /// Name of the offending series plus the surface-specific
        /// consequence and remedy.
        what: &'static str,
        /// Zero-based index of the first offending entry.
        index: usize,
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
        /// Deterministic columns per equation beyond the lags and levels
        /// (unrestricted constant/trend plus terms restricted to the
        /// cointegration relation); they consume the same degrees of
        /// freedom as the stochastic regressors.
        n_det: usize,
        /// Centered seasonal-dummy columns (`seasons - 1`); like `n_det`
        /// they count against the usable rows.
        n_seasonal: usize,
    },
    /// The sample is too short for the Hansen-Seo threshold VECM: after
    /// one difference and `k_ar_diff` presample lags, the usable rows
    /// cannot hold two regimes of `max(m + 1, ceil(trim * n))`
    /// observations each (`m = 2 + k * k_ar_diff` regressors per regime).
    /// `needed` and `got` are both in usable-row units, and `needed` is
    /// exact: the smallest usable sample this `(k, k_ar_diff, trim)`
    /// accepts.
    ThresholdInsufficientObservations {
        /// Exact minimum number of usable rows for this specification.
        needed: usize,
        /// Number of usable rows available (`nobs - k_ar_diff - 1`).
        got: usize,
        /// Number of rows in the data as it was supplied.
        nobs: usize,
        /// Number of series `k` in the system.
        neqs: usize,
        /// Lagged-difference order `k_ar_diff` that was requested.
        k_ar_diff: usize,
        /// Regressors per regime, `m = 2 + k * k_ar_diff`.
        n_regressors: usize,
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
            Self::NonFiniteSeries { what, index } => write!(
                f,
                "non-finite value (NaN or inf) at index {index} of {what}"
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
                n_det,
                n_seasonal,
            } => {
                let base = neqs * k_ar_diff + neqs;
                let per_eq = base + n_det + n_seasonal;
                write!(
                    f,
                    "the Johansen/VECM auxiliary regressions need at least {needed} usable \
                     rows but only {got} of the {nobs} rows supplied survive one difference \
                     plus {k_ar_diff} presample lag(s): with k={neqs} series each regression \
                     has k*k_ar_diff + k = {base} regressors",
                )?;
                if n_det + n_seasonal > 0 {
                    write!(f, " plus")?;
                    if *n_det > 0 {
                        write!(f, " {n_det} deterministic column(s)")?;
                    }
                    if *n_det > 0 && *n_seasonal > 0 {
                        write!(f, " and")?;
                    }
                    if *n_seasonal > 0 {
                        write!(f, " {n_seasonal} seasonal-dummy column(s)")?;
                    }
                    write!(f, " = {per_eq} in total")?;
                }
                write!(f, ". {}", k_ar_diff_hint(*nobs, *neqs, *n_det, *n_seasonal))
            }
            Self::ThresholdInsufficientObservations {
                needed,
                got,
                nobs,
                neqs,
                k_ar_diff,
                n_regressors,
            } => write!(
                f,
                "the Hansen-Seo threshold VECM needs at least {needed} usable rows: each \
                 of its two regimes must keep at least max(m + 1, ceil(trim * n)) rows, \
                 with m = 2 + k*k_ar_diff = {n_regressors} regressors per regime for \
                 k={neqs} series and k_ar_diff={k_ar_diff}. Only {got} of the {nobs} rows \
                 supplied survive one difference plus {k_ar_diff} presample lag(s); supply \
                 at least {} input rows, or lower k_ar_diff or trim.",
                needed + k_ar_diff + 1
            ),
        }
    }
}

/// Concrete "what to try" clause for
/// [`CointError::InsufficientObservations`].
///
/// Both entry points build the same effective sample `t = n - 1 -
/// k_ar_diff` and need `t > k * k_ar_diff + k + n_det + n_seasonal`
/// (every deterministic and seasonal-dummy column consumes a degree of
/// freedom alongside the stochastic regressors), i.e.
/// `n >= k_ar_diff * (k + 1) + k + n_det + n_seasonal + 2`. Inverting
/// that bound gives the largest lagged-difference order this sample can
/// support at the requested deterministic/seasonal specification.
fn k_ar_diff_hint(nobs: usize, neqs: usize, n_det: usize, n_seasonal: usize) -> String {
    let extra = n_det + n_seasonal;
    let d_max = nobs.saturating_sub(neqs + 2 + extra) / (neqs + 1);
    let seasonal_note = if n_seasonal > 0 {
        format!(
            " (the {n_seasonal} seasonal-dummy column(s) already consume {n_seasonal} \
             of the degrees of freedom — reducing seasons also helps)"
        )
    } else {
        String::new()
    };
    if d_max >= 1 {
        format!(
            "Try k_ar_diff <= {d_max}{seasonal_note}, drop a series, or supply a \
             longer sample."
        )
    } else if nobs >= neqs + 2 + extra {
        let with_cols = if extra > 0 {
            format!(" with {extra} deterministic/seasonal column(s)")
        } else {
            String::new()
        };
        format!(
            "Only k_ar_diff = 0 fits {nobs} input rows on k={neqs} series{with_cols}; \
             each extra lagged difference costs {} more input rows.{seasonal_note}",
            neqs + 1
        )
    } else {
        let with_cols = if extra > 0 {
            format!(" with {extra} deterministic/seasonal column(s)")
        } else {
            String::new()
        };
        format!(
            "Even k_ar_diff = 0 would need {} input rows on k={neqs} series{with_cols}, \
             so supply a longer sample, fit fewer series{}.",
            neqs + 2 + extra,
            if n_seasonal > 0 {
                ", or reduce seasons (each seasonal dummy costs one usable row)"
            } else {
                ""
            }
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
