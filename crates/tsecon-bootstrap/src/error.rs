//! Error type for the `tsecon-bootstrap` crate.

use core::fmt;

use tsecon_rng::RngError;

/// Errors produced by the `tsecon-bootstrap` crate.
///
/// The resampling loops themselves are infallible once their inputs are
/// validated; errors arise only from invalid scheme parameters, samples too
/// small (or too degenerate) for a procedure, or from the seeding hierarchy
/// in [`tsecon_rng`].
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum BootstrapError {
    /// The sample has zero observations: no resampling scheme or block-length
    /// selector is defined on an empty sample.
    EmptySample,
    /// A block bootstrap was requested with a block length outside
    /// `1..=n`.
    InvalidBlockLength {
        /// The requested block length.
        block_length: usize,
        /// The sample size the block must fit in.
        n: usize,
    },
    /// The stationary bootstrap was requested with a restart probability
    /// outside `(0, 1]` (or non-finite). The expected block length is `1/p`,
    /// so `p = 0` would never terminate a block and `p > 1` is not a
    /// probability.
    InvalidProbability {
        /// The offending restart probability.
        p: f64,
    },
    /// The sample is too short for the Politis-White block-length selection
    /// rule, which needs autocovariances out to lag
    /// `ceil(sqrt(n)) + K_n` (see [`crate::optimal_block_length`]).
    SampleTooShort {
        /// The sample size provided.
        n: usize,
        /// The minimum sample size the procedure requires.
        required: usize,
    },
    /// The series is numerically degenerate for block-length selection:
    /// zero sample variance (a constant series) or a zero long-run-variance
    /// estimate, either of which makes the optimal-block-length formula
    /// divide by zero.
    DegenerateSeries,
    /// The series contains a NaN or infinity.
    NonFiniteData,
    /// An error bubbled up from the RNG seeding hierarchy (e.g. the
    /// SeedSequence spawn limit when requesting too many replications).
    Rng(RngError),
}

impl fmt::Display for BootstrapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BootstrapError::EmptySample => {
                write!(f, "cannot resample an empty sample (n = 0)")
            }
            BootstrapError::InvalidBlockLength { block_length, n } => write!(
                f,
                "block length {block_length} is outside the valid range 1..={n}"
            ),
            BootstrapError::InvalidProbability { p } => write!(
                f,
                "stationary-bootstrap restart probability {p} is outside (0, 1]"
            ),
            BootstrapError::SampleTooShort { n, required } => write!(
                f,
                "sample of size {n} is too short for Politis-White block-length \
                 selection (requires at least {required} observations)"
            ),
            BootstrapError::DegenerateSeries => write!(
                f,
                "series is degenerate (zero variance or zero long-run variance); \
                 optimal block length is undefined"
            ),
            BootstrapError::NonFiniteData => {
                write!(f, "series contains NaN or infinite values")
            }
            BootstrapError::Rng(e) => write!(f, "rng error: {e}"),
        }
    }
}

impl std::error::Error for BootstrapError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            BootstrapError::Rng(e) => Some(e),
            _ => None,
        }
    }
}

impl From<RngError> for BootstrapError {
    fn from(e: RngError) -> Self {
        BootstrapError::Rng(e)
    }
}
