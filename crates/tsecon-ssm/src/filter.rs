//! Kalman filtering: the univariate (sequential) filter with exact diffuse
//! initialization — the primary path — and a standard matrix filter with a
//! Joseph-form covariance update as the cross-check path.
//!
//! # Univariate (sequential) filtering
//!
//! Following Koopman & Durbin (2000) and Durbin & Koopman (2012, §6.4), the
//! elements of the observation vector `y_t` are brought in one at a time.
//! With `Z_i` the `i`-th row of `Z`, `sigma2_i = H_ii`, and `(a_{t,i},
//! P_{t,i})` the state moments after absorbing elements `1..i-1` of `y_t`:
//!
//! ```text
//! v_{t,i} = y_{t,i} - d_i - Z_i a_{t,i}
//! F_{t,i} = Z_i P_{t,i} Z_i' + sigma2_i,      M_{t,i} = P_{t,i} Z_i'
//! a_{t,i+1} = a_{t,i} + M_{t,i} v_{t,i} / F_{t,i}
//! P_{t,i+1} = P_{t,i} - M_{t,i} M_{t,i}' / F_{t,i}
//! ```
//!
//! followed by the time transition `a_{t+1,1} = c + T a_{t,p+1}`,
//! `P_{t+1,1} = T P_{t,p+1} T' + R Q R'`. No `p x p` inversion ever occurs,
//! missing elements (NaN in `y`) are simply skipped, and the log-likelihood
//! accumulates one scalar prediction-error term per observed element:
//! `-(ln 2*pi + ln F_{t,i} + v_{t,i}^2 / F_{t,i}) / 2`.
//!
//! This decomposition of the joint Gaussian density is exact only when `H`
//! is diagonal; a non-diagonal `H` is rejected with
//! [`SsmError::NonDiagonalH`].
//! // TODO(phase0): LDL' pre-whitening transform to lift the diagonal-H
//! // restriction (Durbin & Koopman 2012, §6.4.1).
//!
//! # Exact diffuse initialization
//!
//! Diffuse states are handled exactly (Koopman 1997; Koopman & Durbin 2003;
//! Durbin & Koopman 2012, ch. 5) via the two-matrix decomposition
//! `P_t = P_{*,t} + kappa P_{inf,t}` with `kappa -> infinity` — never a
//! large-kappa approximation. Per scalar element, with
//! `F_{inf} = Z_i P_inf Z_i'`, `F_* = Z_i P_* Z_i' + sigma2_i`,
//! `M_inf = P_inf Z_i'`, `M_* = P_* Z_i'`, and `F_inf > 0`:
//!
//! ```text
//! K0 = M_inf / F_inf,     K1 = M_* / F_inf - M_inf F_* / F_inf^2
//! a      <- a + K0 v
//! P_*    <- P_* - M_* K0' - M_inf K1'
//! P_inf  <- P_inf - M_inf K0'
//! loglik <- loglik - (ln 2*pi + ln F_inf) / 2
//! ```
//!
//! while an element with `F_inf = 0` gets the ordinary update above. The
//! diffuse period ends at the first `t` whose incoming `P_inf` is
//! numerically zero; the `-(ln 2*pi + ln F_inf)/2` diffuse likelihood
//! contribution matches statsmodels' `use_exact_diffuse=True` filter, so
//! log-likelihoods are directly comparable. (Cross-package caution: some
//! implementations, e.g. following Francke, Koopman & de Vos 2010, omit
//! constants from the diffuse contribution.)
//!
//! # Numerical-rank decisions
//!
//! Three decisions in the recursions above ask "is this quantity zero?":
//! whether `P_inf` has washed out, whether `F_inf` is positive (diffuse
//! branch), and whether `F_*` is positive (informative element). The two
//! `F` tests are asked *relative to the scale of the quantity being
//! tested*; the `P_inf` washout test is absolute, and deliberately so —
//! see [`TOLERANCE_RANK`], which sets out why the three are not treated
//! alike. Each `F` test also has a roundoff companion, read against a
//! reference the updates cannot collapse ([`TOLERANCE_CANCEL`]).
//!
//! An absolute floor on `F_*` is a units bug: `F_*` is a variance in the
//! units of `y` squared, so a floor of `1e-10` silently declares every
//! observation uninformative for any series whose variance happens to sit
//! below it, and the filter then returns `loglik = 0` as though it had
//! succeeded. The relative test is invariant to rescaling `y`, which is the
//! property the closed-form `loglik(c y) = loglik(y) - n ln c` identity in
//! `tests/scale.rs` pins.
//!
//! Getting *relative to what* right is the other half. The reference scale
//! must be direction-aware: `F_{*,i}` is a variance along the single
//! direction `Z_i`, so measuring it against a scale that sees states `Z_i`
//! does not load on lets an unobserved nuisance state — a regression
//! coefficient in different units, an approximate-diffuse prior of `1e6` —
//! veto every observation in the sample. The reference used here is the
//! cancellation-free quadratic form `|Z_i|' |P| |Z_i|`
//! ([`abs_quad_form`](crate::dense::abs_quad_form)), which is both
//! direction-aware and invariant to rescaling an individual state
//! coordinate. `tests/scale.rs` pins both invariances on multi-state
//! models.

use tsecon_linalg::faer::{Mat, MatRef};
use tsecon_linalg::jittered_cholesky;

use crate::dense::{
    abs_quad_form, axpy, chol_solve, dot, frob_sq, mat_t_vec, mat_vec, outer_sub, row_to_vec,
    sandwich, symmetrize_in_place,
};
use crate::error::SsmError;
use crate::model::LinearGaussianSSM;

/// Tolerance for the filter's three numerical-rank decisions.
///
/// The two `F` tests are *relative*: each compares a computed nonnegative
/// variance against this fraction of a reference scale carrying the same
/// units, so both are invariant to rescaling the data and to rescaling any
/// individual state coordinate. The washout test is *absolute*, because the
/// quantity it tests is dimensionless. In detail:
///
/// * **informative element.** An element carries information while
///   `F_* > TOLERANCE_RANK * (|Z_i|' |P_*| |Z_i| + H_ii)`; otherwise it is
///   skipped exactly like a missing value. The reference is the magnitude
///   `Z_i P_* Z_i' + H_ii` would have had with no cancellation, so the test
///   reads "did at least `1e-10` of the available magnitude survive the
///   sum?" — and while it passes the second, roundoff test of
///   [`TOLERANCE_CANCEL`], read against the period's predicted `P_*`.
/// * **diffuse element.** An element takes the diffuse update while
///   `F_inf > TOLERANCE_RANK * |Z_i|' |P_inf| |Z_i|`, the same test applied
///   to the diffuse part — and while it passes the second, roundoff test
///   of [`TOLERANCE_CANCEL`].
/// * **diffuse period.** A step is diffuse while `||P_inf,t||_F^2 >
///   TOLERANCE_RANK`, an **absolute** threshold.
///
/// ## Why the washout test is absolute, and the other two are not
///
/// The "three uses, one treatment" symmetry is the wrong instinct here, and
/// it is worth recording why. `F_*` and `F_inf` are *variances*: `F_*` is
/// in units of `y` squared and `F_inf` inherits the units of the diffuse
/// prior, so an absolute floor on either is a units error — that is the
/// defect this tolerance was rewritten to fix. `P_inf` is not a variance.
/// It is a **rank indicator** for the limit `P_1 = P_* + kappa P_inf`,
/// `kappa -> infinity`: it never touches `y`, `H` or `Q`, it is built as a
/// 0/1 diagonal (`I`, or the flagged states of a `Mixed` initialization),
/// and it is only ever asked how many diffuse directions are left. It is
/// `O(1)` by construction, so there was no units bug at that site.
///
/// Normalizing it by `||P_inf,1||_F^2` — the diffuse prior's *total*
/// initial size — would only add a failure mode: for a non-uniform prior
/// like `diag(1e10, 1)` a still-live diffuse direction of size `1` is
/// `1e-20` of the total and would be declared washed out, ending the
/// diffuse period early and silently. Absolute is correct here.
///
/// The value `1e-10` is inherited from statsmodels' `tolerance_diffuse`,
/// which applies it as an absolute floor everywhere. On unit-scale data the
/// two agree to the last bit (`tests/golden.rs` is the proof); they part
/// company only where the absolute floor is wrong, namely when a variance's
/// own scale is below `1e-10` — small-variance data — or far above it,
/// where the absolute floor is too permissive about cancellation.
pub(crate) const TOLERANCE_RANK: f64 = 1e-10;

/// Floor for the two roundoff tests: the fraction of the variance `Z_i`
/// could see before the updates that `F_inf` — and `F_*` — must keep (see
/// below; the `F_*` use is the last section).
///
/// The relative `F_inf` test above asks "did at least `TOLERANCE_RANK` of
/// the available magnitude survive the sum `Z_i P_inf Z_i'`?", with the
/// available magnitude read off the *current* `P_inf`. That catches
/// cancellation inside the quadratic form, but not cancellation that has
/// already happened inside `P_inf` itself: once every diffuse direction
/// `Z_i` can see has been annihilated, `Z_i P_inf Z_i'` and its
/// cancellation-free reference are *both* roundoff, their ratio is `O(1)`,
/// and the test votes "still diffuse" on pure dust. That is not a corner
/// case: a state the observation never loads on and the transition never
/// feeds into an observed state stays diffuse for the whole sample — the
/// Nyquist sine partner of a trigonometric seasonal with `harmonics =
/// period / 2`, which is statsmodels' default for an even period, is
/// exactly that — so `||P_inf||_F^2` never washes out and the element test
/// stays live long after it has anything real to decide. Taking the diffuse
/// branch there adds a spurious `-(ln 2*pi + ln F_inf)/2` with
/// `F_inf ~ 1e-32`, i.e. tens of log-likelihood points of pure noise.
///
/// The second test answers two questions the covariance recursion cannot
/// answer about itself, with two companions carried through the diffuse
/// period. Neither feeds the arithmetic of the update, only the branch.
///
/// * **What is `F_inf`, without the dust?** A square-root factor `B` of
///   `P_inf` (`P_inf = B B'`, one column per diffuse state of `P_inf,1`),
///   moved as `B <- T B`; each diffuse update removes the resolved
///   direction by the orthogonal projection `B <- B (I - u u' / u'u)`,
///   `u = B' Z_i'`, and `F_root = u'u`. The covariance downdate
///   `P_inf - M_inf M_inf' / F_inf` leaves dust of order `eps` relative to
///   the cancelled terms — and up to `eps / r` after a direction resolved
///   with `F_inf` only a fraction `r` of them, because it divides by
///   `F_inf`. The projection divides by nothing, and `F_root` is a sum of
///   squares of entries that are themselves roundoff once the direction
///   is gone, so its dust is of order `eps^2` where the covariance form's
///   is of order `eps / r`.
/// * **Against what?** `U = Phi P_inf,1 Phi'`, the diffuse covariance with
///   nothing resolved, carried by the *actual* transition (`U <- T U T'`)
///   and read as `|Z_i|' |U| |Z_i|`. In exact arithmetic `P_inf = U -
///   (positive semidefinite downdates)`, so `0 <= F_inf <= Z_i U Z_i'`:
///   `U` is the most diffuse variance `Z_i` could ever see, a resolved
///   direction keeps it at full size (which is what refuses the Nyquist
///   dust), and it grows exactly as fast as `T^t` does — not faster.
///
/// An element takes the diffuse branch only when, in addition to the
/// relative test above, `F_root > TOLERANCE_CANCEL * |Z_i|' |U| |Z_i|`, so
/// this test can only ever make the diffuse branch harder to enter.
///
/// **Why not an elementwise bound.** Until 0.11.0 the reference was an
/// upper bound on `|P_inf|` with every sign made favourable: `|P_inf,1|`
/// plus `|M_inf| |K0|'` at each downdate, carried as `|T| P_mag |T|'`. An
/// elementwise-absolute transport grows like `rho(|T|)^(2t)` even when `T`
/// preserves norms: the dummy seasonal of period 12 has `T^12 = I` but
/// `rho(|T|) = 1.9995`, so the bound grew about fourfold per period, and a
/// rotation (a trigonometric seasonal, a cycle) by up to
/// `(|cos| + |sin|)^2 = 2`. With the same calendar month missing in two
/// consecutive years, the genuine `F_inf = 1.091` that resolves it at
/// `t = 26` met a threshold of `203.8` and was refused, where the
/// reference above is `|Z_i|' |U| |Z_i| = 2`; `P_inf` never washed out,
/// `nobs_diffuse` came back as the sample length and the log-likelihood
/// 0.64 to 2.87 too low. A year or more missing at the start did the same,
/// the bound compounding through the whole gap
/// (`tests/uc_extensions_golden.rs`, `long_diffuse`; the replay is
/// `scripts/uc_long_diffuse_evidence.py`).
///
/// The value `1e-14` is a few dozen multiples of the double-precision
/// epsilon, and it is a floor on what the *update* can resolve rather than
/// on what `F_root` can see: the update itself still runs on the covariance
/// form, whose own roundoff is `eps` of the same magnitude, so a direction
/// `Z_i` sees with `F_inf` below `1e-14 Z_i U Z_i'` cannot be applied
/// faithfully even when `F_root` says it is there. It keeps genuine diffuse
/// mass that is small only because `Z_i` sees it weakly (the `1e-10` live
/// `F_inf` at `s = 1e-5` in `tests/scale.rs`), and it caps the
/// `lambda -> 0` divergence of a diffuse cycle, whose weakly observed state
/// shows itself at `F_root` of order `(rho sin lambda)^2` of the reference.
/// On the series of `tests/uc_properties.rs`
/// (`scripts/uc_exact_diffuse_reference.py`): at frequency `1e-4` it is
/// taken at once (`4.9e-11`) and the log-likelihood is the exact one; at
/// `1e-6` the first sighting (`4.9e-15`) is refused and the next
/// (`4.1e-14`) taken, `0.010` below the exact log-likelihood; at `1e-8`
/// every sighting (at most `7.7e-15`) is refused and the state stays
/// diffuse, as it does at `lambda = 0`.
///
/// ## The same dust on `F_*`
///
/// The informative-element test has the same blind spot within a period.
/// When an element with `H_ii = 0` pins the direction a later element of
/// the same `y_t` sees — two series with zero idiosyncratic variance
/// loading on one factor, observed together — `P_*` along that direction
/// is roundoff of the pinning update, so `F_*` and its cancellation-free
/// reference `|Z_i|' |P_*| |Z_i| + H_ii` are both dust and their ratio is
/// `O(1)`. Taking the update then adds `-(ln 2*pi + ln F_* + v^2 / F_*)/2`
/// with `F_*` of order `1e-19`, about `+20` log-likelihood points whose
/// sign and size follow the roundoff, not the data (a redundant element's
/// innovation is zero in exact arithmetic: the joint density of `y_t` is
/// singular, and the value the filter reports is the density on its
/// support, with the redundant element skipped like a missing value).
///
/// The reference that does not collapse is the period's *predicted* `P_*`
/// (the one stored in `predicted_state_cov`, before any of the period's
/// sequential updates): in exact arithmetic the updates only remove
/// variance, so `0 <= F_* <= |Z_i|' |P_{t|t-1}| |Z_i| + H_ii`, and the
/// covariance downdates leave roundoff of order `eps` of that magnitude.
/// A standard update is therefore taken only when, in addition to the
/// relative test of [`TOLERANCE_RANK`], `F_* > TOLERANCE_CANCEL *
/// (|Z_i|' |P_{t|t-1}| |Z_i| + H_ii)`. For the first element updated in a
/// period the two references coincide and this test is implied by the
/// first, so a single observed series per period — every univariate model
/// — is untouched; it can only ever make the update harder to enter. The
/// element it refuses is skipped exactly like a missing value, which is
/// the density on the support: setting the redundant cell to NaN gives the
/// same log-likelihood and smoothed states (`tests/scale.rs`,
/// `redundant_zero_variance_element_is_skipped_like_a_missing_value`; the
/// `dfm_smooth` goldens of `tsecon-nowcast`). An idiosyncratic variance
/// below `1e-14` of the variance the element could see is, at this
/// precision, a zero one.
///
/// **When the pinning was itself nearly redundant.** "Roundoff of order
/// `eps`" holds for a downdate that divides by an `F_*` of the size of its
/// reference. An element that pins a direction an earlier one of the same
/// period almost pinned — two zero-variance series whose loadings are at
/// an angle `theta` — has `F_*` of order `theta^2` of its reference, and
/// its downdate leaves dust of order `eps / theta`, which can pass the
/// `1e-14` test: with two factors and pinning loadings `0.02` rad apart,
/// the redundant third zero-variance series was taken in every period, and
/// each of the 40 column orders that pin with that pair came out 115 to 839
/// log-likelihood points high over 60 periods (the covariance form cannot
/// tell that dust from a variance; `tests/scale.rs`). So an `F_*`
/// that passes the test but sits within [`SUSPECT_DUST`] of the predicted
/// reference, after an update of the same (non-diffuse) period, is read a
/// second time off a square-root companion ([`StarRoot`]): a factor `S` of
/// the predicted `P_*` on the states `Z_t` loads on, downdated by the
/// period's updates in Potter's form `S <- S (I - g u u')`, `u = S' Z_k'`
/// — for `H_kk = 0` the orthogonal projection that removes the pinned
/// direction, dividing by nothing small — and `F = u'u + H_ii`, a sum of
/// squares whose dust is of order `(eps / theta)^2`. The update is taken
/// only when that `F` also clears `TOLERANCE_CANCEL` of the reference. The
/// companion decides only; the update itself runs on the covariance form,
/// as before, and the companion is built only in a period that has such an
/// element, so the filter's other paths are bit-identical
/// (`tests/scale.rs`,
/// `nearly_collinear_pinning_does_not_let_the_redundant_element_in`).
pub(crate) const TOLERANCE_CANCEL: f64 = 1e-14;

/// Ceiling of the band, as a fraction of the predicted reference
/// `|Z_i|' |P_{t|t-1}| |Z_i| + H_ii`, in which an `F_*` that passed both
/// tests after an earlier update of the period is confirmed on the
/// square-root companion (see [`TOLERANCE_CANCEL`], "When the pinning was
/// itself nearly redundant").
///
/// A pin whose `F_*` is a fraction `r` of its reference leaves dust of
/// order `eps / sqrt(r)` of the reference, so dust reaches this ceiling
/// only behind a pin with `r` below about `1e-19`, which the tests refuse:
/// the first update of a period must clear `TOLERANCE_RANK` (`1e-10`) of a
/// reference that is still the predicted one, a later one the companion's
/// `1e-14`. (Two consecutive pins each near that floor, whose own `F_*`
/// carry about two correct digits, could compound past it.) Above the
/// ceiling an `F_*` is taken as computed, so a model whose elements all
/// carry real idiosyncratic variance never builds the companion: an
/// element lands in the band only when its `H_ii` is below a millionth of
/// the variance it could see.
const SUSPECT_DUST: f64 = 1e-6;

/// Verdict of the predicted-reference test on an `F_*` (see
/// [`TOLERANCE_CANCEL`]).
enum Dust {
    /// Far above roundoff of the period's updates (or no update yet).
    Clear,
    /// At most `TOLERANCE_CANCEL` of the reference: roundoff, skipped.
    Below,
    /// In the [`SUSPECT_DUST`] band; the reference, for the companion.
    Suspect(f64),
}

/// `ln(2 pi)`, the per-element likelihood constant.
#[inline]
fn ln_2pi() -> f64 {
    (2.0 * std::f64::consts::PI).ln()
}

/// Per-element filtering record kept for the backward smoothing pass
/// (the univariate smoother re-derives its gains from these).
#[derive(Debug, Clone)]
pub(crate) struct ObsStep {
    /// True when the element was observed *and* informative (`F > 0`);
    /// missing (NaN) and numerically singular elements are both skipped
    /// by filter and smoother alike.
    pub(crate) observed: bool,
    /// True when the filter took the *exact-diffuse* branch for this
    /// element. Recorded rather than re-derived, so the smoother replays
    /// the branch the filter actually took instead of re-running a
    /// threshold comparison that could disagree with it.
    pub(crate) diffuse: bool,
    /// Prediction error `v_{t,i}`.
    pub(crate) v: f64,
    /// Finite innovation variance `F_{*,t,i}` (includes `H_ii`).
    pub(crate) f_star: f64,
    /// Diffuse innovation variance `F_{inf,t,i}` (zero outside the
    /// diffuse update branch).
    pub(crate) f_inf: f64,
    /// `M_{*,t,i} = P_{*,t,i} Z_i'`.
    pub(crate) m_star: Vec<f64>,
    /// `M_{inf,t,i} = P_{inf,t,i} Z_i'`.
    pub(crate) m_inf: Vec<f64>,
}

impl ObsStep {
    /// Record for a skipped element (missing or numerically singular).
    fn skipped() -> Self {
        ObsStep {
            observed: false,
            diffuse: false,
            v: 0.0,
            f_star: 0.0,
            f_inf: 0.0,
            m_star: Vec::new(),
            m_inf: Vec::new(),
        }
    }
}

/// Output of the univariate (sequential) Kalman filter.
///
/// Quantities follow the statsmodels reporting conventions: `filtered_*`
/// hold the posterior moments `a_{t|t}`, `P_{t|t}` and `predicted_*` hold
/// the one-step-ahead moments `a_{t+1|t}`, `P_{t+1|t}` (one extra trailing
/// entry for the post-sample prediction). During the diffuse period the
/// covariance entries are the *finite* parts `P_*`; the diffuse parts live
/// in [`FilterOutput::predicted_diffuse_state_cov`].
#[derive(Debug, Clone)]
pub struct FilterOutput {
    /// Joint log-likelihood by prediction-error decomposition, with the
    /// exact-diffuse contribution `-(ln 2*pi + ln F_inf)/2` during diffuse
    /// elements (matches statsmodels `use_exact_diffuse=True`).
    pub loglik: f64,
    /// Number of time periods in the (contiguous, initial) diffuse period:
    /// `t < d_diffuse` was filtered with the exact-diffuse recursions.
    pub d_diffuse: usize,
    /// Predicted state means `a_{t+1|t}`, indexed `0..=n` (entry `t` is
    /// the mean before absorbing `y_t`; entry `n` is the post-sample
    /// one-step prediction).
    pub predicted_state: Vec<Vec<f64>>,
    /// Predicted state covariances `P_{t+1|t}` (finite part `P_*` during
    /// the diffuse period), indexed `0..=n`.
    pub predicted_state_cov: Vec<Mat<f64>>,
    /// Predicted diffuse covariances `P_{inf,t}`, indexed `0..=n`; exactly
    /// zero from the end of the diffuse period onwards.
    pub predicted_diffuse_state_cov: Vec<Mat<f64>>,
    /// Filtered state means `a_{t|t}`, indexed `0..n`.
    pub filtered_state: Vec<Vec<f64>>,
    /// Filtered state covariances `P_{t|t}` (finite part during the
    /// diffuse period), indexed `0..n`.
    pub filtered_state_cov: Vec<Mat<f64>>,
    /// Per-element records, row-major (`t * p + i`), for the smoother.
    pub(crate) steps: Vec<ObsStep>,
}

/// Output of the standard matrix Kalman filter (the cross-check path).
///
/// Same reporting conventions as [`FilterOutput`]; no diffuse quantities
/// because this path requires a proper (finite-covariance) initialization.
#[derive(Debug, Clone)]
pub struct MatrixFilterOutput {
    /// Joint log-likelihood by prediction-error decomposition.
    pub loglik: f64,
    /// Predicted state means `a_{t+1|t}`, indexed `0..=n`.
    pub predicted_state: Vec<Vec<f64>>,
    /// Predicted state covariances `P_{t+1|t}`, indexed `0..=n`.
    pub predicted_state_cov: Vec<Mat<f64>>,
    /// Filtered state means `a_{t|t}`, indexed `0..n`.
    pub filtered_state: Vec<Vec<f64>>,
    /// Filtered state covariances `P_{t|t}`, indexed `0..n`.
    pub filtered_state_cov: Vec<Mat<f64>>,
}

/// Validates the observation matrix: `n x p` shape with `n >= 1`, entries
/// finite or NaN (NaN encodes "missing"; infinities are rejected).
fn validate_observations(model: &LinearGaussianSSM, y: MatRef<'_, f64>) -> Result<(), SsmError> {
    if y.ncols() != model.obs_dim() {
        return Err(SsmError::Dimension {
            what: "observations y must be n x p (one column per observation element)",
            expected: model.obs_dim(),
            got: y.ncols(),
        });
    }
    if y.nrows() == 0 {
        return Err(SsmError::InvalidArgument {
            what: "observations y must contain at least one time period",
        });
    }
    for j in 0..y.ncols() {
        for i in 0..y.nrows() {
            if y[(i, j)].is_infinite() {
                return Err(SsmError::NonFinite {
                    what: "y (entries must be finite or NaN-for-missing)",
                });
            }
        }
    }
    Ok(())
}

/// Square-root factor `B` (`m x r`) of the initial diffuse covariance,
/// `P_inf,1 = B B'`, one column per diffuse state.
///
/// Every initialization builds `P_inf,1` as a nonnegative diagonal (`I`
/// under [`Initialization::Diffuse`](crate::Initialization::Diffuse), the
/// flagged unit entries under `Mixed`, zero otherwise), so `B` is the
/// matching columns of `diag(P_inf,1)^(1/2)`; off-diagonal entries never
/// occur and are not read.
fn diffuse_root(p_inf: MatRef<'_, f64>) -> Mat<f64> {
    let idx: Vec<usize> = (0..p_inf.nrows())
        .filter(|&i| p_inf[(i, i)] > 0.0)
        .collect();
    Mat::from_fn(p_inf.nrows(), idx.len(), |i, j| {
        if i == idx[j] {
            p_inf[(i, i)].sqrt()
        } else {
            0.0
        }
    })
}

/// Square-root companion of a period's `P_*` for the informative-element
/// decision only (see [`TOLERANCE_CANCEL`], "When the pinning was itself
/// nearly redundant"): a factor `S` with `S S' = P_{t|t-1}` restricted to
/// the states `Z_t` loads on — the only block a quadratic form `Z_i P Z_i'`
/// reads — downdated in Potter's form by the period's standard updates.
struct StarRoot {
    /// The state indices with a nonzero entry in some row of `Z_t`.
    support: Vec<usize>,
    /// `|support| x rank` factor of `P_*` on `support`.
    s: Mat<f64>,
}

impl StarRoot {
    /// The factor of the period's predicted `p_pre` on the support of `z`,
    /// replayed through the standard updates `done` (element indices of
    /// `z` and `h`) the period has already taken.
    fn replay(
        p_pre: MatRef<'_, f64>,
        z: MatRef<'_, f64>,
        h: MatRef<'_, f64>,
        done: &[usize],
    ) -> Self {
        let support: Vec<usize> = (0..z.ncols())
            .filter(|&j| (0..z.nrows()).any(|i| z[(i, j)] != 0.0))
            .collect();
        let sub = Mat::from_fn(support.len(), support.len(), |a, b| {
            p_pre[(support[a], support[b])]
        });
        let mut root = StarRoot {
            s: pivoted_root(sub),
            support,
        };
        for &k in done {
            root.downdate(&row_to_vec(z, k), h[(k, k)]);
        }
        root
    }

    /// `u = S' Z_i'` and `F = u'u + H_ii`: the element's prediction variance
    /// read as a sum of squares.
    fn project(&self, zi: &[f64], h_ii: f64) -> (Vec<f64>, f64) {
        let u: Vec<f64> = (0..self.s.ncols())
            .map(|l| {
                self.support
                    .iter()
                    .enumerate()
                    .map(|(a, &j)| self.s[(a, l)] * zi[j])
                    .sum()
            })
            .collect();
        let f = dot(&u, &u) + h_ii;
        (u, f)
    }

    /// Potter's update for the element `(zi, h_ii)`: `S <- S (I - g u u')`
    /// with `g = 1 / (F + sqrt(H_ii F))`, so `S S'` becomes `P - M M' / F`
    /// (`M = P Z_i' = S u`); for `H_ii = 0` it is the orthogonal projection
    /// that removes the pinned direction.
    fn downdate(&mut self, zi: &[f64], h_ii: f64) {
        let (u, f) = self.project(zi, h_ii);
        if f.is_nan() || f <= 0.0 {
            return;
        }
        let g = 1.0 / (f + (h_ii * f).sqrt());
        let w: Vec<f64> = (0..self.s.nrows())
            .map(|a| {
                u.iter()
                    .enumerate()
                    .map(|(l, ul)| self.s[(a, l)] * ul)
                    .sum()
            })
            .collect();
        for (l, ul) in u.iter().enumerate() {
            for (a, wa) in w.iter().enumerate() {
                self.s[(a, l)] -= g * wa * ul;
            }
        }
    }
}

/// A factor `S` (`n x rank`) of the symmetric positive semidefinite `a`,
/// `a = S S'` to roundoff, by Cholesky with diagonal pivoting: each step
/// takes the largest remaining diagonal and the factorization stops when
/// none is positive, so a singular (or roundoff-indefinite) matrix gets a
/// factor of its numerical rank instead of a failure.
fn pivoted_root(mut a: Mat<f64>) -> Mat<f64> {
    let n = a.nrows();
    let mut used = vec![false; n];
    let mut cols: Vec<Vec<f64>> = Vec::new();
    loop {
        let mut pivot = None;
        let mut best = 0.0;
        for (j, &u) in used.iter().enumerate() {
            if !u && a[(j, j)] > best {
                best = a[(j, j)];
                pivot = Some(j);
            }
        }
        let Some(j) = pivot else { break };
        used[j] = true;
        let sd = best.sqrt();
        // The rows pivoted before are exactly zero in exact arithmetic.
        let col: Vec<f64> = (0..n)
            .map(|i| {
                if used[i] && i != j {
                    0.0
                } else {
                    a[(i, j)] / sd
                }
            })
            .collect();
        for c in 0..n {
            for r in 0..n {
                a[(r, c)] -= col[r] * col[c];
            }
        }
        cols.push(col);
    }
    Mat::from_fn(n, cols.len(), |i, k| cols[k][i])
}

/// Runs the univariate (sequential) Kalman filter with exact diffuse
/// initialization — the primary filtering path (see the module docs for
/// the recursions and references).
///
/// `y` is `n x p` with NaN marking a missing element; individual elements
/// of a partially missing `y_t` are used.
///
/// # Errors
///
/// * [`SsmError::NonDiagonalH`] when `H` is not diagonal;
/// * [`SsmError::Dimension`] / [`SsmError::InvalidArgument`] /
///   [`SsmError::NonFinite`] on malformed `y`;
/// * [`SsmError::Linalg`] when the requested initialization cannot be
///   resolved (e.g. stationary initialization of an explosive `T`).
pub fn filter_univariate(
    model: &LinearGaussianSSM,
    y: MatRef<'_, f64>,
) -> Result<FilterOutput, SsmError> {
    if !model.h_is_diagonal() {
        return Err(SsmError::NonDiagonalH);
    }
    validate_observations(model, y)?;
    let n = y.nrows();
    let p = model.obs_dim();
    let ln2pi = ln_2pi();

    let init = model.initial_state()?;
    let mut a = init.a1;
    let mut p_star = init.p_star;
    let mut p_inf = init.p_inf;
    // Two companions of `P_inf`, carried through the diffuse period for the
    // diffuse-element decision only (see TOLERANCE_CANCEL): a square-root
    // factor `B` (`P_inf = B B'`), which reads `F_inf` without the
    // cancellation dust of the covariance form, and `U = Phi P_inf,1 Phi'`,
    // the diffuse covariance with nothing resolved, which is the
    // reference it is read against.
    let mut p_inf_root = diffuse_root(p_inf.as_ref());
    let mut p_inf_unres = p_inf.clone();

    let mut out = FilterOutput {
        loglik: 0.0,
        d_diffuse: 0,
        predicted_state: Vec::with_capacity(n + 1),
        predicted_state_cov: Vec::with_capacity(n + 1),
        predicted_diffuse_state_cov: Vec::with_capacity(n + 1),
        filtered_state: Vec::with_capacity(n),
        filtered_state_cov: Vec::with_capacity(n),
        steps: Vec::with_capacity(n * p),
    };
    // Once P_inf collapses to (numerical) zero it stays zero, so the
    // diffuse period is a contiguous prefix; the flag makes that explicit.
    // A proper (non-diffuse) initialization has P_inf = 0 exactly, so the
    // diffuse machinery is inert for the whole sample.
    let mut in_diffuse = frob_sq(p_inf.as_ref()) > 0.0;
    // Observed (non-NaN) elements, and how many of them carried
    // information. An empty effective sample is an error, not a zero.
    let mut n_observed = 0usize;
    let mut n_informative = 0usize;

    for t in 0..n {
        let z = model.z().at(t);
        let h = model.h().at(t);
        let tr = model.t().at(t);

        // Absolute, on purpose: P_inf is a dimensionless rank indicator,
        // not a variance (see TOLERANCE_RANK).
        let diffuse = in_diffuse && frob_sq(p_inf.as_ref()) > TOLERANCE_RANK;
        if diffuse {
            out.d_diffuse += 1;
        } else if in_diffuse {
            in_diffuse = false;
            // The diffuse period is over. What is left in `P_inf` is
            // roundoff of a quantity the recursion has annihilated, and
            // `T P_inf T'` would keep amplifying it for the rest of the
            // sample (a `[[1, 1], [0, 1]]` trend block grows it
            // polynomially in `t`), turning `1e-6` of dust at the end of
            // the diffuse period into an `O(1)` "diffuse" forecast
            // variance a hundred periods later. Zero it: that is the
            // documented contract of `predicted_diffuse_state_cov`, and it
            // is what the algebra says.
            p_inf = Mat::zeros(p_inf.nrows(), p_inf.ncols());
        }

        out.predicted_state.push(a.clone());
        out.predicted_state_cov.push(p_star.clone());
        out.predicted_diffuse_state_cov.push(p_inf.clone());
        // The predicted P_* just stored: the reference the informative-
        // element test reads once an update has moved `p_star` off it.
        let p_star_pre = out.predicted_state_cov[t].as_ref();
        // Its standard deviations, for the O(m) Cauchy-Schwarz bound
        // `|Z_i|' |P| |Z_i| <= (sum_j |Z_ij| sd_j)^2` that settles the test
        // without the O(m^2) form whenever F_* is not dust.
        let p_star_pre_sd: Vec<f64> = (0..p_star_pre.nrows())
            .map(|j| p_star_pre[(j, j)].max(0.0).sqrt())
            .collect();
        let mut updated = false;
        // The square-root companion, built only if an element of this
        // period lands in the SUSPECT_DUST band (see TOLERANCE_CANCEL), and
        // where this period's steps start in `out.steps`.
        let mut star_root: Option<StarRoot> = None;
        let step0 = out.steps.len();

        // Sequential scalar updates: (a, p_star, p_inf) morph in place
        // from predicted to filtered moments.
        for i in 0..p {
            let yti = y[(t, i)];
            if yti.is_nan() {
                out.steps.push(ObsStep::skipped());
                continue;
            }
            n_observed += 1;
            let zi = row_to_vec(z, i);
            let v = yti - model.obs_intercept()[i] - dot(&zi, &a);
            let m_star = mat_vec(p_star.as_ref(), &zi);
            // Clamp roundoff-negative variances to zero (statsmodels does
            // the same) before branching.
            let f_star = (dot(&zi, &m_star) + h[(i, i)]).max(0.0);
            // Reference scale for F_*: the magnitude Z_i P_* Z_i' + H_ii
            // would have had if nothing in the sum cancelled. F_* below
            // TOLERANCE_RANK of this is cancellation noise; F_* above it is
            // a real (if tiny) variance. Direction-aware — a state Z_i does
            // not load on contributes nothing, however large its variance.
            let f_star_scale = abs_quad_form(&zi, p_star.as_ref()) + h[(i, i)];
            // ... and the same magnitude read off the period's PREDICTED
            // P_* (before any of this period's sequential updates), which
            // the updates cannot collapse: once an earlier element of y_t
            // has pinned the direction Z_i sees, both F_* and the scale
            // above are roundoff of that update (see TOLERANCE_CANCEL).
            // Until the period's first update the two coincide and the
            // first test implies this one; after it, the O(m) bound
            // settles every F_* far above dust and the exact form is read
            // only for the rest. An F_* that clears the test but stays
            // within SUSPECT_DUST of the reference is confirmed on the
            // square-root companion below.
            let against_predicted = |f: f64| {
                if !updated {
                    return Dust::Clear;
                }
                let bound: f64 = zi
                    .iter()
                    .zip(&p_star_pre_sd)
                    .map(|(zj, sd)| zj.abs() * sd)
                    .sum();
                if f > SUSPECT_DUST * (bound * bound + h[(i, i)]) {
                    return Dust::Clear;
                }
                let reference = abs_quad_form(&zi, p_star_pre) + h[(i, i)];
                if f <= TOLERANCE_CANCEL * reference {
                    Dust::Below
                } else if f > SUSPECT_DUST * reference {
                    Dust::Clear
                } else {
                    Dust::Suspect(reference)
                }
            };
            let (f_inf, m_inf, f_inf_scale, u_root, f_root, f_inf_unres) = if diffuse {
                let mi = mat_vec(p_inf.as_ref(), &zi);
                let scale = abs_quad_form(&zi, p_inf.as_ref());
                let u = mat_t_vec(p_inf_root.as_ref(), &zi);
                let f_root = dot(&u, &u);
                let unres = abs_quad_form(&zi, p_inf_unres.as_ref());
                (dot(&zi, &mi).max(0.0), mi, scale, u, f_root, unres)
            } else {
                (0.0, Vec::new(), 0.0, Vec::new(), 0.0, 0.0)
            };

            // Each reference scale is a sum of nonnegative terms that
            // dominate the corresponding terms of the quantity it is
            // compared against (`|z' P z| <= |z|' |P| |z|`; `H` is PSD), so
            // both are nonnegative and passing `f > TOLERANCE_RANK * scale`
            // implies `f > 0` — including when the scale is itself zero,
            // where the test degenerates to exactly `f > 0`. The branches
            // below therefore never divide by a zero variance.
            // The same holds for `f_root`, a sum of squares, against
            // `f_inf_unres`, so the root downdate below divides by a
            // positive `f_root` too.
            // The predicted-P_* reference dominates `F_*` the same way in
            // exact arithmetic (the period's updates only remove variance
            // along `Z_i`).
            debug_assert!(f_star_scale >= 0.0 && f_inf_scale >= 0.0 && f_inf_unres >= 0.0);

            if f_inf > TOLERANCE_RANK * f_inf_scale && f_root > TOLERANCE_CANCEL * f_inf_unres {
                // Exact-diffuse element update (Koopman & Durbin 2003).
                let k0: Vec<f64> = m_inf.iter().map(|x| x / f_inf).collect();
                let f12 = -f_star / f_inf;
                let k1: Vec<f64> = m_star
                    .iter()
                    .zip(&k0)
                    .map(|(ms, k0j)| ms / f_inf + k0j * f12)
                    .collect();
                axpy(&mut a, v, &k0);
                // P_* <- P_* L0' + P_inf L1' = P_* - M_* K0' - M_inf K1'.
                outer_sub(&mut p_star, &m_star, &k0);
                outer_sub(&mut p_star, &m_inf, &k1);
                // P_inf <- P_inf L0' = P_inf - M_inf K0'.
                outer_sub(&mut p_inf, &m_inf, &k0);
                // The same downdate on the root: B <- B (I - u u' / u'u),
                // an orthogonal projection of B's columns that removes the
                // resolved direction (B' z = 0) without dividing by a
                // small number, so what is left of it is roundoff of B
                // itself, not roundoff amplified by 1 / F_inf.
                let bu = mat_vec(p_inf_root.as_ref(), &u_root);
                let u_k: Vec<f64> = u_root.iter().map(|x| x / f_root).collect();
                outer_sub(&mut p_inf_root, &bu, &u_k);
                out.loglik -= 0.5 * (ln2pi + f_inf.ln());
                n_informative += 1;
                updated = true;
                out.steps.push(ObsStep {
                    observed: true,
                    diffuse: true,
                    v,
                    f_star,
                    f_inf,
                    m_star,
                    m_inf,
                });
            } else if f_star > TOLERANCE_RANK * f_star_scale
                && match against_predicted(f_star) {
                    Dust::Clear => true,
                    Dust::Below => false,
                    // A diffuse period's updates are not Potter downdates
                    // of P_*; the companion runs on standard periods only.
                    Dust::Suspect(_) if diffuse => true,
                    Dust::Suspect(reference) => {
                        let root = star_root.get_or_insert_with(|| {
                            // The standard updates this period has taken.
                            let done: Vec<usize> = (0..i)
                                .filter(|&k| {
                                    let s = &out.steps[step0 + k];
                                    s.observed && !s.diffuse
                                })
                                .collect();
                            StarRoot::replay(p_star_pre, z, h, &done)
                        });
                        root.project(&zi, h[(i, i)]).1 > TOLERANCE_CANCEL * reference
                    }
                }
            {
                // Standard scalar update (Koopman & Durbin 2000).
                let k0: Vec<f64> = m_star.iter().map(|x| x / f_star).collect();
                axpy(&mut a, v, &k0);
                outer_sub(&mut p_star, &m_star, &k0);
                if let Some(root) = star_root.as_mut() {
                    root.downdate(&zi, h[(i, i)]);
                }
                out.loglik -= 0.5 * (ln2pi + f_star.ln() + v * v / f_star);
                n_informative += 1;
                updated = true;
                out.steps.push(ObsStep {
                    observed: true,
                    diffuse: false,
                    v,
                    f_star,
                    f_inf: 0.0,
                    m_star,
                    m_inf: Vec::new(),
                });
            } else {
                // Numerically singular element: carries no information —
                // a zero (or roundoff) variance, or one an earlier element
                // of y_t has already pinned (a series with zero
                // idiosyncratic variance that is redundant given another).
                out.steps.push(ObsStep::skipped());
            }
        }

        // Rank-one updates drift off exact symmetry at roundoff level;
        // restore the invariant before the covariances are stored/reused.
        symmetrize_in_place(&mut p_star);
        symmetrize_in_place(&mut p_inf);

        out.filtered_state.push(a.clone());
        out.filtered_state_cov.push(p_star.clone());

        // Time transition: a <- c + T a, P_* <- T P_* T' + R Q R',
        // P_inf <- T P_inf T'.
        let mut a_next = mat_vec(tr, &a);
        axpy(&mut a_next, 1.0, model.state_intercept());
        a = a_next;
        let rqr = model.rqr(t)?;
        let mut ps_next = sandwich(tr, p_star.as_ref());
        ps_next += &rqr;
        p_star = ps_next;
        p_inf = sandwich(tr, p_inf.as_ref());
        if in_diffuse {
            // Both companions move with the *actual* transition: B <- T B,
            // U <- T U T'. (Never |T|: an elementwise-absolute transport
            // compounds like rho(|T|)^(2t) even when T preserves norms.)
            p_inf_root = tr * p_inf_root.as_ref();
            p_inf_unres = sandwich(tr, p_inf_unres.as_ref());
        }
        symmetrize_in_place(&mut p_star);
        symmetrize_in_place(&mut p_inf);
    }

    // Every observed element was skipped as uninformative: the returned
    // log-likelihood would be exactly 0.0 — not a small number, an *empty
    // sum* — and reporting that as a success is a silent wrong answer.
    // (An all-missing `y` is not this case: it is an explicit, documented
    // request to run the recursions with nothing to condition on.)
    if n_observed > 0 && n_informative == 0 {
        return Err(SsmError::NoInformation);
    }

    out.predicted_state.push(a);
    out.predicted_state_cov.push(p_star);
    out.predicted_diffuse_state_cov.push(p_inf);
    Ok(out)
}

/// Runs the standard matrix Kalman filter with a Joseph-form covariance
/// update — the cross-check path (Durbin & Koopman 2012, §4.3).
///
/// Per period, with the observed subvector `y_t^o` (missing elements
/// dropped) and the correspondingly subset `Z^o`, `d^o`, `H^oo`:
///
/// ```text
/// v_t = y_t^o - d^o - Z^o a_t,      F_t = Z^o P_t Z^o' + H^oo
/// K_t = P_t Z^o' F_t^{-1},          a_{t|t} = a_t + K_t v_t
/// P_{t|t} = (I - K_t Z^o) P_t (I - K_t Z^o)' + K_t H^oo K_t'   (Joseph)
/// a_{t+1} = c + T a_{t|t},          P_{t+1} = T P_{t|t} T' + R Q R'
/// loglik += -(p_t^o ln 2*pi + ln det F_t + v_t' F_t^{-1} v_t) / 2
/// ```
///
/// The Joseph form is stable for any gain (Bucy & Joseph 1968); `F_t` is
/// factorized through the shared jitter-ladder Cholesky, so `ln det F_t`
/// comes from the factor diagonal, never an explicit determinant.
///
/// Unlike the univariate path this filter accepts a non-diagonal `H`, but
/// it requires a proper initialization: exact-diffuse models are rejected
/// with [`SsmError::DiffuseNotSupported`] (use [`filter_univariate`]).
///
/// # Errors
///
/// * [`SsmError::DiffuseNotSupported`] when the initialization has a
///   diffuse part;
/// * [`SsmError::Dimension`] / [`SsmError::InvalidArgument`] /
///   [`SsmError::NonFinite`] on malformed `y`;
/// * [`SsmError::Linalg`] when `F_t` is not positive definite within the
///   jitter ladder, or the initialization cannot be resolved.
pub fn filter_matrix(
    model: &LinearGaussianSSM,
    y: MatRef<'_, f64>,
) -> Result<MatrixFilterOutput, SsmError> {
    validate_observations(model, y)?;
    let init = model.initial_state()?;
    if init.has_diffuse() {
        return Err(SsmError::DiffuseNotSupported {
            what: "the standard matrix Kalman filter",
        });
    }
    let n = y.nrows();
    let p = model.obs_dim();
    let m = model.state_dim();
    let ln2pi = ln_2pi();

    let mut a = init.a1;
    let mut p_mat = init.p_star;

    let mut out = MatrixFilterOutput {
        loglik: 0.0,
        predicted_state: Vec::with_capacity(n + 1),
        predicted_state_cov: Vec::with_capacity(n + 1),
        filtered_state: Vec::with_capacity(n),
        filtered_state_cov: Vec::with_capacity(n),
    };

    for t in 0..n {
        let z = model.z().at(t);
        let h = model.h().at(t);
        let tr = model.t().at(t);

        out.predicted_state.push(a.clone());
        out.predicted_state_cov.push(p_mat.clone());

        let obs: Vec<usize> = (0..p).filter(|&i| !y[(t, i)].is_nan()).collect();
        let k = obs.len();
        if k > 0 {
            // Subset system for the observed elements.
            let z_o = Mat::from_fn(k, m, |r, c| z[(obs[r], c)]);
            let h_oo = Mat::from_fn(k, k, |r, c| h[(obs[r], obs[c])]);
            let v: Vec<f64> = obs
                .iter()
                .map(|&i| y[(t, i)] - model.obs_intercept()[i] - dot(&row_to_vec(z, i), &a))
                .collect();

            // M = P Z_o' (m x k), F = Z_o M + H_oo (k x k).
            let m_mat = p_mat.as_ref() * z_o.as_ref().transpose();
            let mut f = z_o.as_ref() * m_mat.as_ref();
            f += &h_oo;
            symmetrize_in_place(&mut f);
            let chol = jittered_cholesky(f.as_ref())?;

            // K = M F^{-1} (m x k): row r of K solves F x = (row r of M)'.
            let mut kal = Mat::<f64>::zeros(m, k);
            for r in 0..m {
                let x = chol_solve(&chol.factor, &row_to_vec(m_mat.as_ref(), r));
                for (c, xc) in x.iter().enumerate() {
                    kal[(r, c)] = *xc;
                }
            }

            // a_{t|t} = a_t + K v (K already contains F^{-1}).
            axpy(&mut a, 1.0, &mat_vec(kal.as_ref(), &v));
            let f_inv_v = chol_solve(&chol.factor, &v);

            // Joseph update: P <- (I - K Z_o) P (I - K Z_o)' + K H_oo K'.
            let kz = kal.as_ref() * z_o.as_ref();
            let i_kz = Mat::from_fn(m, m, |r, c| {
                let eye = if r == c { 1.0 } else { 0.0 };
                eye - kz[(r, c)]
            });
            let mut p_new = sandwich(i_kz.as_ref(), p_mat.as_ref());
            let kh = kal.as_ref() * h_oo.as_ref();
            let khk = kh.as_ref() * kal.as_ref().transpose();
            p_new += &khk;
            symmetrize_in_place(&mut p_new);
            p_mat = p_new;

            out.loglik -= 0.5 * (k as f64 * ln2pi + chol.log_det() + dot(&v, &f_inv_v));
        }

        out.filtered_state.push(a.clone());
        out.filtered_state_cov.push(p_mat.clone());

        // Time transition.
        let mut a_next = mat_vec(tr, &a);
        axpy(&mut a_next, 1.0, model.state_intercept());
        a = a_next;
        let rqr = model.rqr(t)?;
        let mut p_next = sandwich(tr, p_mat.as_ref());
        p_next += &rqr;
        symmetrize_in_place(&mut p_next);
        p_mat = p_next;
    }

    out.predicted_state.push(a);
    out.predicted_state_cov.push(p_mat);
    Ok(out)
}

impl LinearGaussianSSM {
    /// Univariate (sequential) Kalman filter — the primary path; see
    /// [`filter_univariate`].
    pub fn filter(&self, y: MatRef<'_, f64>) -> Result<FilterOutput, SsmError> {
        filter_univariate(self, y)
    }

    /// Standard matrix Kalman filter with Joseph-form update — the
    /// cross-check path; see [`filter_matrix`].
    pub fn filter_matrix(&self, y: MatRef<'_, f64>) -> Result<MatrixFilterOutput, SsmError> {
        filter_matrix(self, y)
    }

    /// Log-likelihood of `y` by prediction-error decomposition (with the
    /// exact-diffuse correction during diffuse steps), via the univariate
    /// filter.
    pub fn loglike(&self, y: MatRef<'_, f64>) -> Result<f64, SsmError> {
        Ok(filter_univariate(self, y)?.loglik)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A rank-2 PSD 4 x 4 matrix, `B B'` of a 4 x 2 `B`.
    fn rank_two() -> Mat<f64> {
        let b = Mat::from_fn(4, 2, |i, j| {
            [[1.0, 0.5], [-0.3, 2.0], [0.7, 0.0], [1.2, -0.4]][i][j]
        });
        Mat::from_fn(4, 4, |i, j| (0..2).map(|k| b[(i, k)] * b[(j, k)]).sum())
    }

    #[test]
    fn the_pivoted_root_factors_a_singular_psd_matrix_to_its_rank() {
        let a = rank_two();
        let s = pivoted_root(a.clone());
        assert!(s.ncols() <= 4);
        for i in 0..4 {
            for j in 0..4 {
                let ss: f64 = (0..s.ncols()).map(|k| s[(i, k)] * s[(j, k)]).sum();
                assert!(
                    (ss - a[(i, j)]).abs() <= 1e-14 * 4.0,
                    "({i}, {j}): {ss} vs {}",
                    a[(i, j)]
                );
            }
        }
        // A zero matrix has an empty factor, not a failure.
        assert_eq!(pivoted_root(Mat::zeros(3, 3)).ncols(), 0);
    }

    #[test]
    fn the_potter_downdate_is_the_covariance_downdate() {
        // S S' after the update by (z, h) is P - M M' / F, M = P z', F = z P z' + h,
        // for h > 0 and for the projection h = 0.
        let p = Mat::from_fn(3, 3, |i, j| {
            [[2.0, 0.3, -0.4], [0.3, 1.0, 0.2], [-0.4, 0.2, 1.5]][i][j]
        });
        let z = Mat::from_fn(1, 3, |_, j| [0.5, -1.0, 0.25][j]);
        for h_ii in [0.7, 0.0] {
            let h = Mat::from_fn(1, 1, |_, _| h_ii);
            let root = StarRoot::replay(p.as_ref(), z.as_ref(), h.as_ref(), &[0]);
            let zi = row_to_vec(z.as_ref(), 0);
            let m = mat_vec(p.as_ref(), &zi);
            let f = dot(&zi, &m) + h_ii;
            for i in 0..3 {
                for j in 0..3 {
                    let ss: f64 = (0..root.s.ncols())
                        .map(|k| root.s[(i, k)] * root.s[(j, k)])
                        .sum();
                    let want = p[(i, j)] - m[i] * m[j] / f;
                    assert!(
                        (ss - want).abs() <= 1e-14,
                        "h = {h_ii}, ({i}, {j}): {ss} vs {want}"
                    );
                }
            }
            // With h = 0 the pinned direction is gone: z S S' z' is roundoff.
            if h_ii == 0.0 {
                assert!(root.project(&zi, 0.0).1 <= 1e-28);
            }
        }
    }
}
