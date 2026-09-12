//! Error types for `tsecon-ets`.
//!
//! Every fallible entry point in this crate returns `Result<_, EtsError>`;
//! nothing in the non-test code path panics. Messages follow the library's
//! "errors that teach" pillar — they name the offending argument, say why
//! it is a problem for this model family, and say what to pass instead.

use core::fmt;

use tsecon_optim::OptimError;

/// Errors produced by the ETS estimators.
#[derive(Debug, Clone, PartialEq)]
pub enum EtsError {
    /// The model specification is internally inconsistent (a damped trend
    /// without a trend, seasonal periods without a seasonal component, a
    /// seasonal component without periods, ...). The message names the
    /// argument and the fix.
    InvalidSpec {
        /// What was wrong and what to pass instead.
        what: &'static str,
    },
    /// An input contained a NaN or infinite value. The innovations form
    /// conditions every state update on the observed error, so it has no
    /// missing-value mechanism (R's `ets` refuses NaN too); interpolate or
    /// use a Kalman-filter model for series with gaps.
    NonFinite {
        /// Which input the offending value was found in.
        what: &'static str,
        /// Index of the first offending element.
        index: usize,
        /// The offending value.
        value: f64,
    },
    /// The data contain a non-positive value but the specification has a
    /// multiplicative error, trend or seasonal component, none of which is
    /// defined for `y <= 0`.
    NonPositiveData {
        /// Index of the first non-positive observation.
        index: usize,
        /// The offending value.
        value: f64,
        /// The offending specification, e.g. `ETS(M,A,M)`.
        spec: String,
    },
    /// Too few observations for the requested operation.
    TooFewObservations {
        /// Observations required.
        needed: usize,
        /// Observations supplied.
        got: usize,
        /// Which operation needed them.
        what: String,
    },
    /// A smoothing parameter, initial state, or option is outside the
    /// domain the model needs.
    InvalidParameter {
        /// The argument's name, e.g. `alpha`, `initial_seasonal[3]`, `level`.
        name: String,
        /// The offending value.
        value: f64,
        /// The requirement it violated.
        requirement: &'static str,
    },
    /// Two index-aligned inputs had mismatched lengths.
    DimensionMismatch {
        /// What was mismatched.
        what: &'static str,
        /// Expected length.
        expected: usize,
        /// Supplied length.
        got: usize,
    },
    /// The recursion hit a state where a multiplicative component divides by
    /// (numerically) zero, or produced a non-finite fitted value, so the
    /// model is not defined at these parameters for these data.
    Degenerate {
        /// Which quantity degenerated.
        what: &'static str,
        /// The observation index at which it happened.
        index: usize,
    },
    /// The requested model is not in class 1 (additive error, additive or
    /// no trend, additive or no seasonality), so no closed-form forecast
    /// variance exists; the simulation-based interval is used instead.
    NotClass1 {
        /// The specification, e.g. `ETS(M,A,M)`.
        spec: String,
    },
    /// An option was outside its domain.
    InvalidOption {
        /// The option's name.
        name: &'static str,
        /// The offending value.
        value: f64,
        /// The requirement it violated.
        requirement: &'static str,
    },
    /// The optimizer rejected its inputs (propagated from `tsecon-optim`).
    Optim(OptimError),
    /// The `auto_ets` restrictions left no candidate model, or every
    /// candidate failed to fit.
    NoCandidates {
        /// Why.
        what: String,
    },
}

impl fmt::Display for EtsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EtsError::InvalidSpec { what } => write!(f, "{what}"),
            EtsError::NonFinite { what, index, value } => write!(
                f,
                "{what}: contains a non-finite value ({value}) at index {index}; the \
                 innovations state-space form updates every state with the observed \
                 one-step error, so it has no missing-value mechanism (R's ets refuses \
                 NaN as well) — interpolate the gap first, or fit a Kalman-filter model \
                 (local_level_smooth) which handles NaN exactly"
            ),
            EtsError::NonPositiveData { index, value, spec } => write!(
                f,
                "y[{index}] = {value} is not strictly positive, but {spec} has a \
                 multiplicative component and multiplicative errors, trends and \
                 seasonals are defined only for y > 0; pass error=\"add\", trend=None or \
                 \"add\", seasonal=None or \"add\" for data with zeros or negative values"
            ),
            EtsError::TooFewObservations { needed, got, what } => write!(
                f,
                "y has {got} observations but {what} needs at least {needed}; supply a \
                 longer series or a smaller model"
            ),
            EtsError::InvalidParameter {
                name,
                value,
                requirement,
            } => write!(f, "{name} = {value} is invalid: {requirement}"),
            EtsError::DimensionMismatch {
                what,
                expected,
                got,
            } => write!(f, "{what}: expected length {expected} but got {got}"),
            EtsError::Degenerate { what, index } => write!(
                f,
                "the ETS recursion degenerated at observation {index}: {what}; the model \
                 is not defined at these parameters and initial states for these data \
                 (a multiplicative component divided by a zero state, or the fitted \
                 value overflowed) — check the initial states, or use additive components"
            ),
            EtsError::NotClass1 { spec } => write!(
                f,
                "{spec} is not a class-1 (linear, additive-error) model, so no \
                 closed-form forecast variance exists for it; its prediction intervals \
                 are simulation-based"
            ),
            EtsError::InvalidOption {
                name,
                value,
                requirement,
            } => write!(f, "{name} = {value} is invalid: {requirement}"),
            EtsError::Optim(e) => write!(f, "optimizer error: {e}"),
            EtsError::NoCandidates { what } => write!(f, "{what}"),
        }
    }
}

impl std::error::Error for EtsError {}

impl From<OptimError> for EtsError {
    fn from(e: OptimError) -> Self {
        EtsError::Optim(e)
    }
}
