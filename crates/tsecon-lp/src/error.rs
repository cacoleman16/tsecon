//! Error type for `tsecon-lp`.
//!
//! Every fallible entry point returns `Result<_, LpError>`; nothing in the
//! non-test code path panics. Messages follow the library's "errors that
//! teach" pillar: what went wrong, why it matters for the projection, and
//! what to do about it. Errors bubbling up from the shared HAC/OLS engine
//! are wrapped in [`LpError::Hac`] so the caller sees a single error type.

use core::fmt;

use tsecon_hac::HacError;

/// Errors produced by the local-projection estimators in this crate.
#[derive(Debug, Clone, PartialEq)]
pub enum LpError {
    /// An input series contains a NaN or infinite value. LP never skips
    /// missing observations silently; clean or impute the series first.
    NonFinite {
        /// Which input the offending value was found in.
        what: &'static str,
        /// Index of the first offending observation.
        index: usize,
        /// The offending value.
        value: f64,
    },
    /// Two index-aligned inputs have different lengths.
    LengthMismatch {
        /// Human-readable description of the mismatch.
        what: &'static str,
        /// Length of the reference series (usually `y`).
        expected: usize,
        /// Length actually supplied.
        got: usize,
    },
    /// After building lag controls and shifting the outcome forward by `h`,
    /// too few observations remain to fit the horizon-`h` regression.
    HorizonTooLong {
        /// The horizon that ran out of data.
        horizon: usize,
        /// Effective observations available at that horizon.
        nobs: usize,
        /// Regressors in the horizon design (including the constant).
        nparams: usize,
    },
    /// The requested maximum horizon leaves no usable sample even at `h = 0`
    /// given the series length and the number of lag controls.
    SeriesTooShort {
        /// Number of observations supplied.
        n: usize,
        /// Number of lagged-`y` controls requested.
        n_lag_controls: usize,
    },
    /// The state indicator passed to [`crate::lp_state`] is not binary
    /// (values other than 0 and 1 were found).
    NonBinaryState {
        /// Index of the first offending observation.
        index: usize,
        /// The offending value.
        value: f64,
    },
    /// The state indicator selects (almost) no observations for one regime,
    /// so that regime's interacted block is collinear/unidentified.
    DegenerateState {
        /// The horizon at which a regime emptied out.
        horizon: usize,
        /// Observations falling in the sparse regime.
        regime_nobs: usize,
    },
    /// The B-spline configuration passed to [`crate::smooth_lp`] is
    /// infeasible (degree/basis/penalty constraints violated).
    SplineConfig {
        /// Maximum horizon requested.
        horizons: usize,
        /// B-spline degree requested.
        degree: usize,
        /// Number of basis functions (after defaulting).
        n_basis: usize,
        /// Difference-penalty order requested.
        penalty_order: usize,
        /// The violated constraint, with the fix spelled out.
        constraint: &'static str,
    },
    /// A smoothing parameter (fixed `lambda` or a grid entry) is negative or
    /// non-finite.
    InvalidLambda {
        /// The offending value.
        value: f64,
    },
    /// An explicit cross-validation grid with no entries was supplied.
    EmptyLambdaGrid,
    /// The cross-validation fold structure is infeasible: too few folds,
    /// more folds than usable base periods, or a fold whose held-out block
    /// plus dependence buffer leaves too little training data.
    CvConfig {
        /// Number of folds requested.
        n_folds: usize,
        /// Usable base periods (`n - n_lag_controls`).
        n_base: usize,
    },
    /// An error propagated from the shared HAC / OLS engine (singular
    /// design, degrees-of-freedom exhaustion, invalid bandwidth, ...).
    Hac(HacError),
}

impl fmt::Display for LpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LpError::NonFinite { what, index, value } => write!(
                f,
                "{what}: contains a non-finite value ({value}) at index {index}; \
                 local projections do not skip missing values silently — drop or \
                 impute NaN/inf observations before estimating"
            ),
            LpError::LengthMismatch {
                what,
                expected,
                got,
            } => write!(
                f,
                "{what}: expected {expected} observations but got {got}; the \
                 outcome, impulse, instrument, and state inputs must be \
                 index-aligned and equal length"
            ),
            LpError::HorizonTooLong {
                horizon,
                nobs,
                nparams,
            } => write!(
                f,
                "horizon h = {horizon}: only {nobs} usable observations remain \
                 after building lag controls and shifting the outcome forward, \
                 but the design has {nparams} regressors; shorten the maximum \
                 horizon, reduce n_lag_controls, or supply a longer series"
            ),
            LpError::SeriesTooShort { n, n_lag_controls } => write!(
                f,
                "a series of length {n} with {n_lag_controls} lagged-y controls \
                 leaves no usable local-projection sample; supply more data or \
                 reduce n_lag_controls"
            ),
            LpError::NonBinaryState { index, value } => write!(
                f,
                "state indicator: value {value} at index {index} is not 0 or 1; \
                 lp_state expects a binary regime indicator (interactions are \
                 formed with the lagged indicator and its complement)"
            ),
            LpError::DegenerateState {
                horizon,
                regime_nobs,
            } => write!(
                f,
                "horizon h = {horizon}: one regime contains only {regime_nobs} \
                 observations, so its interacted block is (near-)collinear and \
                 the per-state response is unidentified; use a more balanced \
                 state indicator or a shorter horizon"
            ),
            LpError::SplineConfig {
                horizons,
                degree,
                n_basis,
                penalty_order,
                constraint,
            } => write!(
                f,
                "smooth-LP spline configuration (horizons = {horizons}, degree = \
                 {degree}, n_basis = {n_basis}, penalty_order = {penalty_order}) \
                 is infeasible: the constraint is {constraint}"
            ),
            LpError::InvalidLambda { value } => write!(
                f,
                "smoothing parameter lambda = {value} is not a finite \
                 non-negative number; use lambda = 0 for the unpenalized \
                 stacked estimator, a positive value for smoothing, or \
                 cross-validation to pick one from a grid"
            ),
            LpError::EmptyLambdaGrid => write!(
                f,
                "the cross-validation lambda grid is empty; supply at least \
                 one candidate value, or pass None to use the default \
                 log-spaced grid"
            ),
            LpError::CvConfig { n_folds, n_base } => write!(
                f,
                "cross-validation with {n_folds} folds over {n_base} usable \
                 base periods is infeasible: each fold holds out a contiguous \
                 block plus a buffer of horizons + n_lag_controls periods on \
                 each side, and what remains must still support the stacked \
                 fit; use 2 <= n_folds << n_base, reduce horizons, or supply \
                 a longer series"
            ),
            LpError::Hac(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for LpError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            LpError::Hac(e) => Some(e),
            _ => None,
        }
    }
}

impl From<HacError> for LpError {
    fn from(e: HacError) -> Self {
        LpError::Hac(e)
    }
}
