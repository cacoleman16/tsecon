//! Error types for `tsecon-spectral`.

use core::fmt;

/// Errors produced by the spectral estimators in this crate.
///
/// All fallible library entry points return `Result<_, SpectralError>`;
/// nothing in the non-test code path panics.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SpectralError {
    /// The input series contains a NaN or infinite value. The FFT would
    /// propagate a single non-finite sample across every frequency bin, so
    /// inputs are screened up front and the first offender is reported.
    NonFiniteInput {
        /// Index of the first non-finite observation.
        index: usize,
    },
    /// A numeric parameter is outside its valid domain (e.g. a sampling
    /// frequency `fs <= 0`, a zero segment length, or `noverlap >= nperseg`,
    /// which would advance the segment window by zero or fewer samples).
    InvalidParameter {
        /// Name of the offending parameter.
        name: &'static str,
        /// The invalid value that was supplied.
        value: f64,
        /// Human-readable statement of the violated constraint.
        requirement: &'static str,
    },
    /// The requested Welch segment length exceeds the sample size, so not
    /// even one segment fits.
    SegmentTooLong {
        /// Requested segment length.
        nperseg: usize,
        /// Number of observations supplied.
        n: usize,
    },
    /// The two series passed to a cross-spectral routine
    /// ([`crate::coherence`]) have different lengths.
    LengthMismatch {
        /// Length of the first series.
        x_len: usize,
        /// Length of the second series.
        y_len: usize,
    },
}

impl fmt::Display for SpectralError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SpectralError::NonFiniteInput { index } => {
                write!(f, "input series has a non-finite value at index {index}")
            }
            SpectralError::InvalidParameter {
                name,
                value,
                requirement,
            } => write!(
                f,
                "invalid parameter `{name}` = {value}: requires {requirement}"
            ),
            SpectralError::SegmentTooLong { nperseg, n } => write!(
                f,
                "segment length nperseg = {nperseg} exceeds the sample size {n}"
            ),
            SpectralError::LengthMismatch { x_len, y_len } => write!(
                f,
                "cross-spectral inputs differ in length: x has {x_len}, y has {y_len}"
            ),
        }
    }
}

impl std::error::Error for SpectralError {}

/// Reject a series containing NaN or infinities, returning the index of the
/// first offender.
pub(crate) fn check_finite(x: &[f64]) -> Result<(), SpectralError> {
    match x.iter().position(|v| !v.is_finite()) {
        Some(index) => Err(SpectralError::NonFiniteInput { index }),
        None => Ok(()),
    }
}
