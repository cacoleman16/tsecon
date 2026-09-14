//! Error types for `tsecon-panel`.
//!
//! Every fallible entry point in this crate returns `Result<_, PanelError>`;
//! nothing in the non-test code path panics. Error messages follow the
//! library's "errors that teach" pillar: they state what went wrong, why it
//! matters statistically, and what the caller can do about it.

use core::fmt;

use tsecon_var::VarError;

/// Errors produced by the panel estimation machinery in this crate.
#[derive(Debug, Clone, PartialEq)]
pub enum PanelError {
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
    /// An observed input cell contained a NaN or infinity. Panel
    /// estimators never skip missing values silently: drop or impute
    /// them, or declare the missing cells through an observation mask
    /// ([`crate::PanelData::unbalanced`]).
    NonFinite {
        /// Name of the offending input.
        what: &'static str,
    },
    /// The Driscoll-Kraay kernel bandwidth is invalid (negative, NaN, or
    /// infinite).
    InvalidBandwidth {
        /// The offending bandwidth.
        value: f64,
    },
    /// The within (fixed-effects) estimator has no residual degrees of
    /// freedom: it needs `nobs > k + N` because the entity demeaning
    /// absorbs one mean per entity in addition to the `k` slope
    /// parameters.
    DegreesOfFreedom {
        /// Stacked observations `nobs` (`N * T` on a balanced panel, the
        /// observed cells otherwise).
        n: usize,
        /// Number of slope regressors `k`.
        k: usize,
        /// Number of entities `N` with an observation (absorbed fixed
        /// effects).
        n_entities: usize,
    },
    /// The within estimator with a general [`crate::FixedEffects`] menu
    /// (time effects, entity trends) has no residual degrees of freedom:
    /// it needs `nobs > k + n_absorbed`, where `n_absorbed` counts every
    /// absorbed effect and trend slope.
    DegreesOfFreedomAbsorbed {
        /// Stacked observations `nobs` (`N * T` on a balanced panel, the
        /// observed cells otherwise).
        n: usize,
        /// Number of slope regressors `k`.
        k: usize,
        /// Number of absorbed parameters (entity effects, time effects,
        /// entity-trend slopes; on an unbalanced panel only entities and
        /// periods with an observation count).
        n_absorbed: usize,
    },
    /// The sample (or a sub-sample such as a jackknife half-panel) is too
    /// short for the requested horizons/lags.
    InsufficientObservations {
        /// Which computation ran out of data.
        what: &'static str,
        /// Minimum number of usable observations (or periods) required.
        needed: usize,
        /// Number available.
        got: usize,
    },
    /// The within-transformed regressor cross-product `X'X` is not
    /// (numerically) positive definite: the design is collinear. A common
    /// cause is including a regressor that is constant within every
    /// entity — the within transformation zeroes it out exactly.
    SingularDesign {
        /// Which computation hit the singular design.
        what: &'static str,
    },
    /// An estimator (or one of its modes) is defined for balanced panels
    /// only and was handed an unbalanced one (a mask with unobserved
    /// cells). The message names the parameter and says what to pass.
    Unbalanced {
        /// Which estimator or mode refused, and why.
        what: &'static str,
    },
    /// A per-entity matrix of the mean-group panel VAR holds a NaN or
    /// infinity: a VAR needs contiguous observations within each entity,
    /// so an internal gap cannot be masked.
    NonFiniteEntity {
        /// Zero-based index of the offending entity.
        entity: usize,
    },
    /// A per-entity VAR fit inside the mean-group estimator failed.
    EntityVar {
        /// Zero-based index of the offending entity.
        entity: usize,
        /// The underlying VAR-layer error.
        source: VarError,
    },
}

impl fmt::Display for PanelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PanelError::Dimension {
                what,
                expected,
                got,
            } => write!(
                f,
                "dimension mismatch: {what} (expected {expected}, got {got})"
            ),
            PanelError::InvalidArgument { what } => write!(f, "invalid argument: {what}"),
            PanelError::NonFinite { what } => write!(
                f,
                "{what}: contains a non-finite value (NaN or infinity) in an observed \
                 cell; panel estimators do not skip missing values silently — drop or \
                 impute them first, or declare the missing cells through the \
                 observation mask (mask=..., an N x T array of 0/1 flags with 1 where \
                 the entity is observed in that period; masked-out cells may hold NaN)"
            ),
            PanelError::Unbalanced { what } => {
                write!(f, "unbalanced panel (mask with unobserved cells): {what}")
            }
            PanelError::NonFiniteEntity { entity } => write!(
                f,
                "entities[{entity}]: contains a non-finite value (NaN or infinity); a \
                 VAR needs contiguous observations within each entity, so an internal \
                 gap cannot be masked out — pass each entity's contiguous span as its \
                 own T_i x k matrix (the time dimensions may differ across entities; \
                 trim leading and trailing missing rows before the call)"
            ),
            PanelError::InvalidBandwidth { value } => write!(
                f,
                "Driscoll-Kraay bandwidth = {value} is invalid: requires a \
                 finite value >= 0 (the Bartlett lag-truncation parameter, \
                 linearmodels' `bandwidth` / statsmodels' `maxlags`)"
            ),
            PanelError::DegreesOfFreedom { n, k, n_entities } => write!(
                f,
                "n = {n} stacked observations with k = {k} regressors and \
                 N = {n_entities} absorbed entity means leaves no residual \
                 degrees of freedom (requires n > k + N); supply more \
                 periods or drop regressors"
            ),
            PanelError::DegreesOfFreedomAbsorbed { n, k, n_absorbed } => write!(
                f,
                "n = {n} stacked observations with k = {k} regressors and \
                 {n_absorbed} absorbed fixed-effect parameters (entity effects, \
                 time effects and entity-trend slopes) leaves no residual \
                 degrees of freedom (requires n > k + n_absorbed); supply more \
                 periods or entities, or drop effects/regressors"
            ),
            PanelError::InsufficientObservations { what, needed, got } => write!(
                f,
                "{what}: needs at least {needed} usable observations, got \
                 {got}; reduce the horizon/lag order or supply more periods"
            ),
            PanelError::SingularDesign { what } => write!(
                f,
                "{what}: the within-transformed cross-product X'X is \
                 numerically singular (collinear design); a regressor that \
                 is constant within every entity is zeroed exactly by the \
                 within transformation — drop it or use a random-effects / \
                 between estimator instead"
            ),
            PanelError::EntityVar { entity, source } => {
                write!(f, "per-entity VAR fit failed for entity {entity}: {source}")
            }
        }
    }
}

impl std::error::Error for PanelError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            PanelError::EntityVar { source, .. } => Some(source),
            _ => None,
        }
    }
}
