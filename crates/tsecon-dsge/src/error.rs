//! Error type for the linear rational-expectations solver.
//!
//! Every fallible public function in this crate returns
//! `Result<_, DsgeError>`; nothing outside `#[cfg(test)]` panics on user input.
//! Messages follow the library's "errors that teach" pillar: they say what went
//! wrong, why it matters, and what the caller can do about it.

use core::fmt;

use crate::solve::BlanchardKahnVerdict;

/// Errors produced by the Blanchard-Kahn solver, model construction, and
/// simulation.
#[derive(Debug, Clone, PartialEq)]
pub enum DsgeError {
    /// A required matrix or vector was empty.
    EmptyInput {
        /// Name of the offending argument.
        what: &'static str,
    },
    /// A matrix that must be square was not.
    NotSquare {
        /// Name of the offending argument.
        what: &'static str,
        /// The number of rows received.
        rows: usize,
        /// The number of columns received.
        cols: usize,
    },
    /// Two dimensions that had to agree did not.
    DimensionMismatch {
        /// Description of the constraint that was violated.
        what: &'static str,
        /// The dimension that was expected.
        expected: usize,
        /// The dimension that was received.
        got: usize,
    },
    /// An input contained a NaN or infinite entry.
    NonFinite {
        /// Name of the offending argument.
        what: &'static str,
    },
    /// The predetermined/jump split `n_predetermined` was out of range: it must
    /// satisfy `0 <= n_predetermined <= n` where `n` is the number of
    /// endogenous variables.
    InvalidPartition {
        /// The `n_predetermined` that was supplied.
        n_predetermined: usize,
        /// The number of endogenous variables `n`.
        n: usize,
    },
    /// The lead matrix `A` in `A E_t[y_{t+1}] = B y_t + C z` was numerically
    /// singular, so the reduced form `M = A^{-1} B` does not exist. This crate
    /// solves only models with an invertible `A` (the "regular" case); a
    /// singular `A` signals a model with static/definitional equations that
    /// must be substituted out first, or a genuine singularity pencil that
    /// needs the QZ (generalized Schur) generalization not implemented here.
    SingularA,
    /// The predetermined block of the stable eigenvector matrix was singular,
    /// so the policy rule `G = V_xs V_ks^{-1}` could not be formed. This
    /// happens only in degenerate models (e.g. a repeated eigenvalue with a
    /// defective eigenspace); the solver requires `M` to be diagonalizable on
    /// its stable subspace.
    SingularStableBlock,
    /// An eigenvalue lay on (within tolerance of) the unit circle. The
    /// Blanchard-Kahn classification into stable/unstable is then undefined —
    /// the model has a unit root and the solution is knife-edge. Re-specify the
    /// model away from the boundary.
    UnitRoot {
        /// The offending eigenvalue modulus (close to 1).
        modulus: f64,
    },
    /// The computed policy rule carried a non-negligible imaginary part. For a
    /// real model the complex eigenvalues occur in conjugate pairs and the
    /// policy matrices are real; a residual imaginary part signals a numerical
    /// breakdown (e.g. a near-defective eigenspace).
    ComplexSolution {
        /// The largest absolute imaginary part encountered.
        imag: f64,
    },
    /// A shock loaded directly onto a non-predetermined (jump) equation: the
    /// jump rows of `N = A^{-1} C` are not (numerically) zero. Under this
    /// crate's convention the exogenous innovation `z_{t+1}` has
    /// `E_t[z_{t+1}] = 0` and enters only the predetermined block's law of
    /// motion; an innovation written onto a forward-looking equation is not
    /// representable as `jump_t = G predetermined_t`. Move the shock onto an
    /// exogenous (predetermined) AR state, as in the standard formulation.
    ShockOnJump {
        /// The largest absolute jump-row entry of `N`.
        magnitude: f64,
    },
    /// The model does not satisfy the Blanchard-Kahn condition, so no unique
    /// non-explosive solution exists. Carries the verdict (indeterminate or no
    /// stable solution) with the eigenvalue/jump counts.
    BlanchardKahn(BlanchardKahnVerdict),
    /// The dense eigendecomposition failed to converge.
    EigenFailed,
    /// A simulation/impulse-response argument was invalid.
    Simulation {
        /// What was wrong with the request.
        what: &'static str,
    },
}

impl fmt::Display for DsgeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyInput { what } => write!(
                f,
                "empty input: {what}; supply the model matrices A, B, C with at \
                 least one endogenous variable"
            ),
            Self::NotSquare { what, rows, cols } => write!(
                f,
                "matrix {what} must be square (got {rows} x {cols}); A and B are \
                 n x n and describe the n endogenous equations"
            ),
            Self::DimensionMismatch {
                what,
                expected,
                got,
            } => write!(
                f,
                "dimension mismatch: {what} (expected {expected}, got {got})"
            ),
            Self::NonFinite { what } => write!(
                f,
                "non-finite value (NaN or infinity) in {what}; clean the model \
                 matrices before solving"
            ),
            Self::InvalidPartition { n_predetermined, n } => write!(
                f,
                "invalid predetermined/jump split: n_predetermined = \
                 {n_predetermined} but there are only n = {n} endogenous \
                 variables (need 0 <= n_predetermined <= n)"
            ),
            Self::SingularA => write!(
                f,
                "the lead matrix A is singular, so M = A^{{-1}} B does not exist; \
                 this solver handles only an invertible A — substitute out static \
                 equations, or use a QZ-based solver for a singular pencil"
            ),
            Self::SingularStableBlock => write!(
                f,
                "the predetermined block of the stable eigenvectors is singular, \
                 so the policy rule G = V_xs V_ks^{{-1}} is undefined; the model \
                 is defective (a repeated eigenvalue with too few eigenvectors) \
                 on its stable subspace"
            ),
            Self::UnitRoot { modulus } => write!(
                f,
                "an eigenvalue lies on the unit circle (|lambda| = {modulus:.6}); \
                 the Blanchard-Kahn stable/unstable split is undefined at the \
                 boundary — re-specify the model away from the unit root"
            ),
            Self::ComplexSolution { imag } => write!(
                f,
                "the policy rule retained a non-negligible imaginary part \
                 (max |Im| = {imag:.3e}); for a real model this should cancel, so \
                 this signals a near-defective eigenspace or numerical breakdown"
            ),
            Self::ShockOnJump { magnitude } => write!(
                f,
                "a shock loads directly on a jump equation (max |N_jump| = \
                 {magnitude:.3e}); under this crate's convention E_t[z_{{t+1}}] = 0 \
                 and the innovation enters only the predetermined law of motion — \
                 route the shock through an exogenous AR state instead"
            ),
            Self::BlanchardKahn(verdict) => write!(f, "Blanchard-Kahn: {verdict}"),
            Self::EigenFailed => write!(
                f,
                "the dense eigendecomposition of M = A^{{-1}} B failed to \
                 converge; the transition matrix may be pathologically \
                 conditioned"
            ),
            Self::Simulation { what } => write!(f, "invalid simulation request: {what}"),
        }
    }
}

impl std::error::Error for DsgeError {}
