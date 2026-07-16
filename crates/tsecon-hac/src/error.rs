//! Error types for `tsecon-hac`.
//!
//! Every fallible entry point in this crate returns `Result<_, HacError>`;
//! nothing in the non-test code path panics. Error messages follow the
//! library's "errors that teach" pillar: they state what went wrong, why it
//! matters statistically, and what the caller can do about it.

use core::fmt;

/// Errors produced by the HAC / long-run variance machinery in this crate.
#[derive(Debug, Clone, PartialEq)]
pub enum HacError {
    /// The series has too few observations for the requested computation.
    SeriesTooShort {
        /// Which estimator needed more data.
        what: &'static str,
        /// The number of observations supplied.
        n: usize,
        /// The minimum number of observations required.
        needed: usize,
    },
    /// The input contains a NaN or infinite value. HAC estimators never skip
    /// missing values silently; clean or impute the series first.
    NonFinite {
        /// Which input the offending value was found in.
        what: &'static str,
        /// Index of the first offending observation.
        index: usize,
        /// The offending value.
        value: f64,
    },
    /// The kernel bandwidth is invalid (negative, NaN, or infinite).
    InvalidBandwidth {
        /// The offending bandwidth.
        value: f64,
    },
    /// The EWC degrees-of-freedom parameter `B` is outside the valid range
    /// `1 <= B <= n - 1` for a series of length `n`.
    InvalidDof {
        /// Which estimator rejected the degrees of freedom.
        what: &'static str,
        /// The degrees of freedom supplied.
        b: usize,
        /// The number of observations supplied.
        n: usize,
    },
    /// The design matrix has no columns.
    EmptyDesign,
    /// A design column's length does not match the response vector's.
    DimensionMismatch {
        /// Which input had the wrong length.
        what: &'static str,
        /// Zero-based index of the offending column.
        column: usize,
        /// The expected length (the length of `y`).
        expected: usize,
        /// The length actually supplied.
        got: usize,
    },
    /// Fewer observations than parameters (or exactly as many): no residual
    /// degrees of freedom remain, so standard errors are undefined.
    DegreesOfFreedom {
        /// The number of observations supplied.
        n: usize,
        /// The number of regressors.
        k: usize,
    },
    /// The regressor cross-product matrix `X'X` is not (numerically)
    /// positive definite: the design is collinear or degenerate.
    SingularDesign {
        /// Which computation hit the singular design.
        what: &'static str,
    },
    /// The series is (numerically) constant at zero, so autocovariance-based
    /// bandwidth selection and long-run variances are undefined.
    ConstantSeries {
        /// Which estimator found the degenerate series.
        what: &'static str,
    },
    /// The requested kernel is not supported by this procedure (e.g. the
    /// truncated kernel has no Newey-West (1994) plug-in rule and is not
    /// positive semi-definite).
    UnsupportedKernel {
        /// Which procedure rejected the kernel.
        what: &'static str,
        /// The name of the rejected kernel.
        kernel: &'static str,
    },
    /// A numerical invariant that holds in exact arithmetic broke down
    /// (e.g. a negative sandwich-variance diagonal from a non-PSD kernel,
    /// or a unit AR(1) root in a plug-in bandwidth).
    NumericalBreakdown {
        /// Which algorithm broke down.
        what: &'static str,
    },
}

impl fmt::Display for HacError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HacError::SeriesTooShort { what, n, needed } => write!(
                f,
                "{what}: series has {n} observations but needs at least {needed}; \
                 supply more data or reduce the requested lag order/bandwidth"
            ),
            HacError::NonFinite { what, index, value } => write!(
                f,
                "{what}: contains a non-finite value ({value}) at index {index}; \
                 HAC estimators do not skip missing values silently — drop or \
                 impute NaN/inf observations before estimating"
            ),
            HacError::InvalidBandwidth { value } => write!(
                f,
                "bandwidth = {value} is invalid: requires a finite value >= 0 \
                 (for Bartlett/Parzen/truncated this is the lag-truncation \
                 parameter, statsmodels' `maxlags`; for quadratic spectral it \
                 is Andrews' real-valued S_T)"
            ),
            HacError::InvalidDof { what, b, n } => write!(
                f,
                "{what}: B = {b} degrees of freedom is invalid for a series of \
                 length {n}: requires 1 <= B <= n - 1; the LLSW (2018) default \
                 is B = round(0.4 * n^(2/3))"
            ),
            HacError::EmptyDesign => write!(
                f,
                "the design matrix has no columns; pass at least one regressor \
                 (include the constant column explicitly, statsmodels-style)"
            ),
            HacError::DimensionMismatch {
                what,
                column,
                expected,
                got,
            } => write!(
                f,
                "{what}: column {column} has {got} observations but the \
                 response has {expected}; every design column must be \
                 index-aligned with y"
            ),
            HacError::DegreesOfFreedom { n, k } => write!(
                f,
                "n = {n} observations with k = {k} regressors leaves no \
                 residual degrees of freedom (requires n > k); standard errors \
                 and the n/(n-k) small-sample correction are undefined"
            ),
            HacError::SingularDesign { what } => write!(
                f,
                "{what}: the regressor cross-product matrix X'X is numerically \
                 singular (collinear columns); drop redundant regressors — a \
                 common cause is passing the constant column twice"
            ),
            HacError::ConstantSeries { what } => write!(
                f,
                "{what}: the series is (numerically) zero/constant, so \
                 autocovariances carry no information and the long-run \
                 variance is undefined; check that the right column was \
                 passed and that it was not zeroed by prior transformations"
            ),
            HacError::UnsupportedKernel { what, kernel } => write!(
                f,
                "{what}: the {kernel} kernel is not supported here; the \
                 truncated kernel is not positive semi-definite and has no \
                 published plug-in bandwidth rule — use Bartlett, Parzen, or \
                 quadratic spectral"
            ),
            HacError::NumericalBreakdown { what } => write!(
                f,
                "{what}: numerical breakdown — an invariant that holds in \
                 exact arithmetic failed; for HAC covariance this usually \
                 means a non-positive-semi-definite kernel (truncated) or a \
                 (near-)degenerate series"
            ),
        }
    }
}

impl std::error::Error for HacError {}
