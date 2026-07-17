//! Error types for `tsecon-termstructure`.
//!
//! Every fallible entry point in this crate returns
//! `Result<_, TermStructureError>`; nothing in the non-test code path panics.
//! Messages follow the library's "errors that teach" pillar: they state what
//! went wrong, why it matters, and what the caller can do about it.

use core::fmt;

/// Errors produced by the yield-curve / term-structure estimators.
#[derive(Debug, Clone, PartialEq)]
pub enum TermStructureError {
    /// The maturity grid was empty. A curve needs at least one maturity, and a
    /// cross-sectional factor fit needs strictly more maturities than factors.
    EmptyMaturities,
    /// A maturity was non-finite or non-positive. Nelson-Siegel loadings are
    /// defined for `t > 0` (the `t -> 0` limit is handled internally only for
    /// the loadings themselves, never as a data maturity).
    InvalidMaturity {
        /// Zero-based index of the offending maturity.
        index: usize,
        /// The offending value.
        value: f64,
    },
    /// The decay parameter `lambda` was non-finite or non-positive. The
    /// loadings `(1 - e^{-lt}) / (lt)` require a strictly positive real
    /// `lambda`; Diebold-Li (2006) fix `lambda = 0.0609` for monthly data.
    InvalidLambda {
        /// Which decay parameter was rejected (`"lambda"`, `"lambda1"`, or
        /// `"lambda2"`).
        what: &'static str,
        /// The offending value.
        value: f64,
    },
    /// A yield or factor input contained a NaN or infinite value. The
    /// estimators never skip missing values silently.
    NonFinite {
        /// Which input the offending value was found in.
        what: &'static str,
        /// Index of the first offending observation.
        index: usize,
        /// The offending value.
        value: f64,
    },
    /// Two index-aligned inputs had mismatched lengths (e.g. the yield vector
    /// and the maturity grid, or two panel rows).
    DimensionMismatch {
        /// A human-readable description of the two things that disagreed.
        what: &'static str,
        /// The expected length.
        expected: usize,
        /// The length actually supplied.
        got: usize,
    },
    /// Fewer distinct pricing observations than factors, so the cross-sectional
    /// least-squares fit is not identified (e.g. 3 maturities for the 3-factor
    /// Nelson-Siegel curve, or 4 for Svensson).
    Underdetermined {
        /// Which fit was under-identified.
        what: &'static str,
        /// The number of maturities (pricing equations) supplied.
        maturities: usize,
        /// The number of factors (parameters) to estimate.
        factors: usize,
    },
    /// The panel had too few dates for the requested dynamic operation (a
    /// factor AR(1) needs at least two dates; a forecast needs the panel to be
    /// non-empty).
    PanelTooShort {
        /// Which operation needed more dates.
        what: &'static str,
        /// The number of dates supplied.
        dates: usize,
        /// The minimum number of dates required.
        needed: usize,
    },
    /// The cross-sectional design (the loading columns) is numerically
    /// collinear, so the normal equations have no stable solution. With a
    /// well-separated maturity grid this only happens for a degenerate
    /// `lambda`.
    SingularDesign {
        /// Which fit hit the singular design.
        what: &'static str,
    },
    /// The optimizer used to estimate `lambda` failed to make progress from
    /// the supplied starting value. Try a different `lambda0` (a good default
    /// is the Diebold-Li 0.0609 for monthly data).
    OptimizationFailed {
        /// A short description of the failure reason.
        reason: &'static str,
    },
    /// An AFNS diagonal factor volatility was negative or non-finite. The
    /// arbitrage-free yield-adjustment term (Christensen-Diebold-Rudebusch
    /// 2011) is a sum of squared volatilities, so each `sigma_ii` must be a
    /// finite, non-negative real (`0` is allowed and nests plain
    /// Nelson-Siegel).
    InvalidSigma {
        /// Zero-based index into `[sigma_11, sigma_22, sigma_33]`.
        index: usize,
        /// The offending value.
        value: f64,
    },
}

impl fmt::Display for TermStructureError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TermStructureError::EmptyMaturities => write!(
                f,
                "the maturity grid is empty; supply the maturities (in the same \
                 time units as lambda, e.g. months for the Diebold-Li 0.0609) \
                 at which the yields are observed"
            ),
            TermStructureError::InvalidMaturity { index, value } => write!(
                f,
                "maturity[{index}] = {value} is invalid: Nelson-Siegel loadings \
                 require strictly positive, finite maturities"
            ),
            TermStructureError::InvalidLambda { what, value } => write!(
                f,
                "{what} = {value} is invalid: the loadings (1 - e^-lt)/(lt) \
                 require a strictly positive, finite decay rate (Diebold-Li \
                 (2006) use lambda = 0.0609 for monthly maturities)"
            ),
            TermStructureError::NonFinite { what, index, value } => write!(
                f,
                "{what}: contains a non-finite value ({value}) at index {index}; \
                 clean or impute NaN/inf observations before fitting"
            ),
            TermStructureError::DimensionMismatch {
                what,
                expected,
                got,
            } => write!(
                f,
                "{what}: expected length {expected} but got {got}; the yields \
                 must be index-aligned with the maturity grid"
            ),
            TermStructureError::Underdetermined {
                what,
                maturities,
                factors,
            } => write!(
                f,
                "{what}: {maturities} maturities cannot identify {factors} \
                 factors; supply strictly more maturities than factors"
            ),
            TermStructureError::PanelTooShort {
                what,
                dates,
                needed,
            } => write!(
                f,
                "{what}: the panel has {dates} dates but needs at least \
                 {needed}; supply a longer factor history"
            ),
            TermStructureError::SingularDesign { what } => write!(
                f,
                "{what}: the loading columns are numerically collinear, so the \
                 cross-sectional least-squares problem has no stable solution; \
                 check the maturity grid and lambda"
            ),
            TermStructureError::OptimizationFailed { reason } => write!(
                f,
                "lambda estimation failed ({reason}); try a different starting \
                 value (the Diebold-Li 0.0609 is a robust default for monthly \
                 data)"
            ),
            TermStructureError::InvalidSigma { index, value } => write!(
                f,
                "sigma_diag[{index}] = {value} is invalid: the AFNS \
                 yield-adjustment term (Christensen-Diebold-Rudebusch 2011) sums \
                 squared factor volatilities, so each must be finite and \
                 non-negative (use 0 to recover plain Nelson-Siegel)"
            ),
        }
    }
}

impl std::error::Error for TermStructureError {}
