//! Weak-instrument-robust (Anderson-Rubin / Fieller) confidence sets for
//! proxy-SVAR impulse responses.
//!
//! # Why a *set* and not an interval
//!
//! [`crate::proxy_svar`] point-identifies the unit-effect-normalized impulse
//! response as a **ratio** of two estimated moments,
//! `lambda_{i,h} = unit * (e_i' Psi_h gamma) / (e_k' gamma)`, where
//! `gamma = E[m_t u_t]` is the residual-instrument covariance and `k` is the
//! normalizing variable. When the instrument is weak the denominator
//! `gamma_k` is close to zero, the ratio has no usable Gaussian limit, and a
//! Wald interval `lambda_hat +/- z*se` is not merely imprecise — it is the
//! wrong shape. Dufour (1997) is the sharp statement: if the parameter is
//! not identified over part of the parameter space, **no bounded confidence
//! set can have correct coverage**. So any honest procedure must be allowed
//! to return an unbounded set.
//!
//! This library has measured the cost of ignoring that. The interval-coverage
//! audit (`docs/examples/interval-coverage.md`) found `iv_gmm` covering
//! `0.839` at a nominal `0.95` under weak instruments, with the *median*
//! reported standard error only `0.456` of the true sampling standard
//! deviation — and it found the familiar `F > 10` rule of thumb is not a safe
//! gate, since coverage was already down to `0.915` at a median first-stage
//! `F` of `10.5`. Reporting a tidy symmetric interval there is the failure,
//! not the remedy.
//!
//! # The construction
//!
//! Fix a cell `(h, i)` and a candidate value `lam`. The null
//! `H0: lambda_{i,h} = lam` is
//!
//! ```text
//! unit * e_i' Psi_h gamma - lam * e_k' gamma = 0,   i.e.   a(lam)' gamma = 0
//! with a(lam) = unit * Psi_h' e_i - lam * e_k,
//! ```
//!
//! which is **affine in `lam` and linear in `gamma`**. That makes it a
//! single Anderson-Rubin moment condition — "the instrument is uncorrelated
//! with this constructed linear combination of reduced-form innovations" —
//! and there is nothing nonlinear left to invert numerically.
//!
//! With `g_t = m~_t u~_t` over the overlap `O` (tilde = demeaned over `O`),
//! `gamma_hat = mean_t g_t` and `sqrt(T_O)(gamma_hat - gamma) -> N(0, Omega)`,
//! the null-imposed statistic is
//!
//! ```text
//! AR_{i,h}(lam) = T_O * g_hat(lam)^2 / V_hat(lam),
//!   g_hat(lam) = q1 - lam*q0                     (affine)
//!   V_hat(lam) = v0 - 2*lam*v1 + lam^2*v2        (quadratic, >= 0 by PSD)
//!   q1 = unit * (Psi_h gamma_hat)_i,  q0 = gamma_hat_k,
//!   v0 = unit^2 * (Psi_h Omega Psi_h')[i,i],
//!   v1 = unit * (Psi_h Omega)[i,k],  v2 = Omega[k,k].
//! ```
//!
//! **Imposing the null in the variance is the whole point.** It is what makes
//! `V_hat` depend on `lam`, and that dependence is the only reason the set can
//! be anything other than a symmetric interval. A variance frozen at
//! `lam_hat` silently reproduces the Wald interval this module exists to
//! escape.
//!
//! Inverting `AR(lam) <= c` is then a **quadratic inequality**,
//! `A*lam^2 + B*lam + C <= 0`, with
//!
//! ```text
//! A = T_O*q0^2 - c*v2,   B = 2*(c*v1 - T_O*q1*q0),   C = T_O*q1^2 - c*v0,
//! ```
//!
//! solved in closed form. No grid, no search.
//!
//! # Four shapes, and why they are all in the return type
//!
//! The solution set of a quadratic inequality is one of: a bounded interval,
//! a single point, the union of two rays (the complement of an open
//! interval), a half-line, the whole real line, or the empty set.
//! [`ArSet`] carries the shape as data, so an unbounded set **cannot** be
//! flattened into a wide-but-finite `(lower, upper)` pair. That flattening is
//! precisely the dishonesty the method exists to avoid: a caller handed
//! `(r_1, r_2)` for an exterior set would shade exactly the region the data
//! *reject*. [`ArSet::endpoints`] therefore returns `(-inf, +inf)` for an
//! exterior set — its true outer bounds — and the rejected middle is
//! available separately from [`ArSet::excluded_middle`].
//!
//! # Three facts derived here, checkable against a grid
//!
//! All three were derived from the moment structure above (not taken from a
//! paper) and are pinned by the crate tests, including a brute-force grid
//! inversion that re-tests `AR(lam) <= c` directly.
//!
//! 1. **Boundedness is one scalar, shared by every cell.**
//!    `A = T_O*gamma_k^2 - c*Omega[k,k]` depends on neither `i` nor `h`. So
//!    [`ProxyArResult::ar_bound_stat`] `= T_O*gamma_k^2/Omega[k,k]` — a robust
//!    Wald statistic for `H0: gamma_k = 0`, i.e. a first-stage relevance test
//!    — decides boundedness for the entire grid at once. You never get
//!    "`h = 0` bounded, `h = 8` unbounded". Note the threshold is `c` (about
//!    `3.84` at 95%), **not** `10`: a bounded set certifies relevance at the
//!    test's own level, not instrument strength. See
//!    [`ProxyArResult::first_stage_f`].
//!
//! 2. **The set is never empty in this just-identified 1x1 case.** At
//!    `lam_hat = q1/q0` the moment is exactly zero, so
//!    `A*lam_hat^2 + B*lam_hat + C = -c*V_hat(lam_hat) <= 0` and the point
//!    estimate always belongs to its own set. Emptiness needs
//!    over-identification (where `min_lam AR(lam)` is a J-statistic), a
//!    degenerate `V_hat(lam_hat) = 0`, or a caller intersecting cells.
//!    [`ArSet::Empty`] stays in the contract for those futures but is not a
//!    normal outcome here.
//!
//! 3. **The `(norm_var, h = 0)` cell is a single point at `unit`.** There
//!    `Psi_0 = I` gives `a(lam) = (unit - lam) e_k`, so the quadratic is
//!    `A*(lam - unit)^2` and the discriminant is exactly zero. The correct
//!    answer is `{unit}` when `A > 0` and all of `R` otherwise — never
//!    `Empty`. A naive `A > 0 && D <= 0 => Empty` branch reports an empty
//!    confidence set for the impact response of the variable the user
//!    normalized on.
//!
//! A note on fact 2 versus the design spec: the spec claims `A > 0` forces
//! `D > 0` (strict). That is off by the boundary case. The correct statement
//! is `A > 0 => D >= 0`, with `D = 0` exactly when `V_hat(lam_hat) = 0` —
//! which is fact 3, a case that genuinely occurs. `D >= 4*A*c*V_hat(lam_hat)`
//! is the identity behind both.
//!
//! # Reduced-form uncertainty is not optional in practice
//!
//! `Psi_h` is an *estimate* in every real application — it comes from a fitted
//! VAR — and its sampling error enters the numerator moment
//! `unit * e_i' Psi_hat_h gamma_hat`. [`ArVariance::Hc0`] and
//! [`ArVariance::HacBartlett`] estimate the variance of the **identification
//! step only** (`gamma_hat`), so on their own they leave that error out.
//!
//! Omitting it is defensible under weak-instrument (local-to-zero relevance)
//! asymptotics, where the term enters multiplied by `gamma` and vanishes with
//! it. With a **strong** instrument it is a first-order omission, and the
//! cost is not a drift — it is a collapse. Measured in
//! `fixtures/generate_proxy_ar_fixtures.py` at nominal `0.95`, `T = 300`, a
//! VAR(2) with spectral radius `0.68`, with the VAR **estimated**, i.e. the
//! only configuration a real caller is ever in:
//!
//! | arm | reduced form | mean | coverage by horizon `h = 0..8` |
//! |---|---|---|---|
//! | strong | omitted | `0.323` | `.952 .529 .458 .315 .247 .195 .163 .135 .119` |
//! | strong | propagated | `0.938` | `.952 .953 .954 .947 .941 .936 .930 .922 .913` |
//! | weak | omitted | `0.941` | `.949 .950 .944 .945 .942 .941 .939 .935 .929` |
//! | weak | propagated | `0.991` | `.949 .981 .988 .995 .997 .999 .999 .998 .998` |
//!
//! Every entry **excludes the `(norm_var, 0)` cell**, which is the point
//! `{unit}` and covers with probability exactly `1` by construction.
//! Averaging it in turns the `h = 0` column of this three-variable system
//! from `.952` into `.968` — enough to make the impact row look better than
//! every other row and hide the fact that the collapse starts at `h = 1`.
//!
//! The two omitted-reduced-form rows are the theory: flat and at nominal in
//! the weak arm, a collapse in the strong one. `.119` at `h = 8` is not a
//! confidence set. Paired on the same draws, the propagated variance makes
//! the median set at `h = 8` **`13.5` times wider** than the moment-only one.
//! The width ratio a Gaussian would need in order to cover `.119` at a
//! nominal `.95` is `13.1`, so the measured width gap and the measured
//! coverage gap — computed from different quantities — agree to three
//! percent. The price is in the last row: propagating is conservative under
//! weak identification (`.991`). The mechanism is measurable — across the weak
//! arm the [`ArSet::Exterior`] share falls from `.085` to `.027` and the
//! [`ArSet::Whole`] share rises from `.830` to `.889`, while the *bounded*
//! share does not move by a single cell. The correction buys nothing there and
//! costs some rejected middles; it is what stands between `.119` and `.95`
//! everywhere else.
//!
//! **So: pass [`ArReducedForm`] whenever `psi` came from an estimated VAR.**
//! [`psi_reduced_form_cov`] builds it from the VAR's coefficient covariance,
//! and [`ArMoments`] documents the algebra — the correction is a constant
//! added to `v0` and a constant added to `v1`, so the closed form is
//! unchanged. When it is absent, [`ProxyArResult::reduced_form_uncertainty`]
//! is `false` and [`ProxyArResult::level`] is `None`: the sets are then a
//! `1 - alpha` set only *conditional on the reduced form*, and the type says
//! so rather than printing a level the coverage does not support.
//!
//! [`ArVariance::Supplied`] does **not** substitute for this. It replaces
//! `Omega`, the covariance of `gamma_hat`; the `Psi_h` term is a separate
//! additive piece and both are needed.
//!
//! # What this set does *not* cover
//!
//! * **These sets are pointwise, not joint.** One set per `(h, i)` cell. The
//!   simultaneous coverage of a whole impulse-response path is far below the
//!   pointwise level; this repository's own analysis measured 90% pointwise
//!   as 72% joint for `var_irf_bands`. Do not present the collection as a
//!   band.
//! * **Cumulative responses come free** by passing the cumulated MA sequence
//!   `sum_{s<=h} Psi_s` as `psi`: the parameter stays linear in `gamma`, so
//!   the identical quadratic applies. Same for any ratio of two linear
//!   functionals of `gamma`.
//!
//! # Attribution, honestly
//!
//! The method is the Anderson-Rubin / Fieller construction — confidence sets
//! by test inversion for a ratio of moments — applied to the proxy-SVAR
//! estimand, and is associated with Montiel Olea, Stock and Watson's work on
//! inference in SVARs identified with an external instrument. Foundational:
//! Anderson and Rubin (1949); Fieller (1954) for the ratio; Dufour (1997) for
//! the impossibility of bounded valid sets; Staiger and Stock (1997) for weak-
//! instrument asymptotics; Jentsch and Lunsford (2019) for why the wild
//! bootstrap is invalid for proxy SVARs.
//!
//! **The closed-form algebra above was derived in this crate from the moment
//! structure in [`crate::proxy`], and has not been checked line-by-line
//! against Montiel Olea-Stock-Watson.** Whether they present the inversion in
//! this form (versus a grid), which variance estimator they use, and whether
//! they recommend the chi-square or a finite-sample critical value are not
//! confirmed here. Precise bibliographic details for that paper are
//! deliberately omitted rather than asserted unverified. Treat the
//! mathematics as self-verifying (the grid test proves it) and the
//! attribution as approximate.

use tsecon_linalg::faer::{Mat, MatRef};
use tsecon_linalg::jittered_cholesky;
use tsecon_rng::Stream;
use tsecon_stats::special::{inv_beta_inc, inv_norm_cdf};

use crate::error::IdentError;

/// Relative floor below which the quadratic's leading coefficient `A` counts
/// as zero (the knife edge where the set degenerates to a half-line).
const TAU_A_REL: f64 = 1e-12;
/// Relative floor below which the linear coefficient `B` counts as zero.
const TAU_B_REL: f64 = 1e-12;
/// Relative floor below which the constant coefficient `C` counts as zero.
const TAU_C_REL: f64 = 1e-12;
/// Relative floor below which the discriminant counts as zero. Chosen a
/// thousand-fold above the few-ulp cancellation error that the exactly
/// degenerate `(norm_var, h = 0)` cell produces in `B^2 - 4AC`; a set
/// mistakenly reported as a point instead of an interval by this tolerance
/// has relative width below `2e-6`.
const TAU_D_REL: f64 = 1e-12;
/// Relative slack in the per-cell positive-semidefiniteness check
/// `v1^2 <= v0*v2`.
const PSD_REL: f64 = 1e-9;

/// The shape of a weak-instrument-robust confidence set for one
/// `(horizon, variable)` cell.
///
/// The shape is carried as **data**, not as a flag beside a `(lower, upper)`
/// pair, so an unbounded set cannot be silently rendered as a finite
/// interval. Under weak identification that is not a stylistic preference:
/// a bounded set there would have no valid coverage (Dufour 1997).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ArSet {
    /// A closed bounded interval `[lo, hi]` with `lo <= hi`. The ordinary
    /// strong-instrument outcome.
    Interval {
        /// Lower endpoint.
        lo: f64,
        /// Upper endpoint.
        hi: f64,
    },
    /// A single point. Produced by the `(norm_var, h = 0)` cell, where the
    /// unit-effect normalization pins the response by construction, and by
    /// any other cell whose null-imposed variance vanishes at the point
    /// estimate.
    Point(f64),
    /// The half-line `(-inf, hi]`. Only at the exact knife edge where the
    /// quadratic degenerates to a linear inequality.
    RayBelow {
        /// Upper endpoint; the set extends to `-inf`.
        hi: f64,
    },
    /// The half-line `[lo, +inf)`. Only at the exact knife edge where the
    /// quadratic degenerates to a linear inequality.
    RayAbove {
        /// Lower endpoint; the set extends to `+inf`.
        lo: f64,
    },
    /// The union of two rays `(-inf, lo] u [hi, +inf)`: the data reject the
    /// **open middle** `(lo, hi)` and accept everything outside it.
    ///
    /// `lo` and `hi` bound the *rejected* region, not the set. Plotting them
    /// as an interval shades exactly the values the data rule out. Use
    /// [`ArSet::endpoints`] (which returns `(-inf, +inf)` here) for the set's
    /// own bounds and [`ArSet::excluded_middle`] for the rejected region.
    Exterior {
        /// Lower edge of the rejected open middle.
        lo: f64,
        /// Upper edge of the rejected open middle.
        hi: f64,
    },
    /// The whole real line: the data restrict this response not at all.
    Whole,
    /// The empty set. Unreachable in the just-identified single-instrument
    /// case (the point estimate always belongs to its own set); retained for
    /// over-identified extensions and for callers intersecting cells.
    Empty,
}

/// The shape of an [`ArSet`] without its payload — a discriminant for
/// bindings and tabular output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArSetKind {
    /// [`ArSet::Interval`].
    Interval,
    /// [`ArSet::Point`].
    Point,
    /// [`ArSet::RayBelow`].
    RayBelow,
    /// [`ArSet::RayAbove`].
    RayAbove,
    /// [`ArSet::Exterior`].
    Exterior,
    /// [`ArSet::Whole`].
    Whole,
    /// [`ArSet::Empty`].
    Empty,
}

impl ArSetKind {
    /// A stable lowercase name, for tabular output and language bindings.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Interval => "interval",
            Self::Point => "point",
            Self::RayBelow => "ray_below",
            Self::RayAbove => "ray_above",
            Self::Exterior => "exterior",
            Self::Whole => "whole",
            Self::Empty => "empty",
        }
    }
}

impl ArSet {
    /// The shape discriminant.
    pub fn kind(&self) -> ArSetKind {
        match self {
            Self::Interval { .. } => ArSetKind::Interval,
            Self::Point(_) => ArSetKind::Point,
            Self::RayBelow { .. } => ArSetKind::RayBelow,
            Self::RayAbove { .. } => ArSetKind::RayAbove,
            Self::Exterior { .. } => ArSetKind::Exterior,
            Self::Whole => ArSetKind::Whole,
            Self::Empty => ArSetKind::Empty,
        }
    }

    /// Whether `lam` belongs to the set.
    ///
    /// Comparisons are exact. For [`ArSet::Point`] that means only the stored
    /// value returns `true`; compare with a tolerance if you are testing a
    /// separately computed estimate against a point set.
    pub fn contains(&self, lam: f64) -> bool {
        match *self {
            Self::Interval { lo, hi } => lam >= lo && lam <= hi,
            Self::Point(p) => lam == p,
            Self::RayBelow { hi } => lam <= hi,
            Self::RayAbove { lo } => lam >= lo,
            Self::Exterior { lo, hi } => lam <= lo || lam >= hi,
            Self::Whole => true,
            Self::Empty => false,
        }
    }

    /// Whether the set is bounded (a finite interval, a point, or empty).
    ///
    /// `false` means no honest finite interval exists for this cell — report
    /// that, do not substitute a large finite number.
    pub fn is_bounded(&self) -> bool {
        matches!(self, Self::Interval { .. } | Self::Point(_) | Self::Empty)
    }

    /// The set's own outer bounds, with infinities where the set is
    /// unbounded: `(-inf, +inf)` for [`ArSet::Whole`] **and for
    /// [`ArSet::Exterior`]**, and `(NaN, NaN)` for [`ArSet::Empty`].
    ///
    /// This is the safe accessor for a caller that wants a `(lower, upper)`
    /// pair: an exterior set renders as "unbounded in both directions", which
    /// is true, rather than as the interval the data reject.
    pub fn endpoints(&self) -> (f64, f64) {
        match *self {
            Self::Interval { lo, hi } => (lo, hi),
            Self::Point(p) => (p, p),
            Self::RayBelow { hi } => (f64::NEG_INFINITY, hi),
            Self::RayAbove { lo } => (lo, f64::INFINITY),
            Self::Exterior { .. } | Self::Whole => (f64::NEG_INFINITY, f64::INFINITY),
            Self::Empty => (f64::NAN, f64::NAN),
        }
    }

    /// The rejected open middle `(lo, hi)` of an [`ArSet::Exterior`], or
    /// `None` for every other shape.
    ///
    /// For an unbounded set this length is the informative finite summary;
    /// the set's own "width" is `+inf` and says nothing.
    pub fn excluded_middle(&self) -> Option<(f64, f64)> {
        match *self {
            Self::Exterior { lo, hi } => Some((lo, hi)),
            _ => None,
        }
    }

    /// Lebesgue measure of the set: `hi - lo` for an interval, `0` for a
    /// point or the empty set, `+inf` for every unbounded shape.
    ///
    /// Averaging this across cells is meaningless once any cell is unbounded
    /// — the mean is `+inf`. Report the fraction of bounded cells and the
    /// excluded-middle lengths instead.
    pub fn width(&self) -> f64 {
        match *self {
            Self::Interval { lo, hi } => hi - lo,
            Self::Point(_) | Self::Empty => 0.0,
            _ => f64::INFINITY,
        }
    }
}

/// Which estimator supplies the moment covariance `Omega` — the asymptotic
/// variance of `sqrt(T_O) * (gamma_hat - gamma)`.
///
/// `Omega` must be positive semidefinite: the branch logic below relies on
/// `V_hat(lam) >= 0` for all `lam`. Both built-in estimators are PSD by
/// construction on a gap-free overlap; a supplied matrix is checked per cell.
#[derive(Debug, Clone, Copy)]
pub enum ArVariance<'a> {
    /// White/HC0: `Omega = (1/T_O) * sum_t (g_t - gamma_hat)(g_t - gamma_hat)'`.
    ///
    /// The default. Correct when `m_t u_t` is serially uncorrelated, which
    /// theory gives for a valid surprise-based proxy.
    Hc0,
    /// Bartlett-kernel HAC with `lags` lags:
    /// `Omega = Gamma_0 + sum_{j=1..L} (1 - j/(L+1)) (Gamma_j + Gamma_j')`.
    ///
    /// Use when `m_t u_t` may be serially correlated (a time-aggregated or
    /// smoothed proxy). The Bartlett kernel is required rather than a
    /// rectangular one because the branch logic assumes a PSD `Omega`.
    ///
    /// Autocovariances are formed over **calendar time**: `g_t` pairs with
    /// `g_{t-j}` only when both dates are inside the proxy's availability
    /// window. Pairing over compacted positions instead would splice across
    /// a NaN gap and invent dependence that is not in the data. The price is
    /// that with interior gaps the usual Bartlett PSD guarantee no longer
    /// applies, so the per-cell check still runs.
    HacBartlett {
        /// Bartlett bandwidth `L`; must be smaller than the overlap count.
        lags: usize,
    },
    /// A caller-supplied `n x n` covariance, e.g. `T_O` times the covariance
    /// of `gamma_hat*` across Jentsch-Lunsford moving-block replications that
    /// jointly resample `(u_t, m_t)` and re-fit the VAR.
    ///
    /// This replaces `Omega` — the covariance of `gamma_hat`, the
    /// *identification* step — and nothing else. It does **not** pick up
    /// `Psi_h` estimation error: that term is a separate additive piece of
    /// `V_hat(lam)`, and it arrives through [`ArReducedForm`]. A bootstrap
    /// `Omega` supplied here without an [`ArReducedForm`] under-covers exactly
    /// as badly as [`ArVariance::Hc0`] does; see the module docs for the
    /// measured numbers.
    ///
    /// Note the extension is this crate's, not a published Jentsch-Lunsford
    /// result: their work establishes the bootstrap's validity for the IRF
    /// estimator, not its use as a variance input to an AR statistic.
    Supplied(MatRef<'a, f64>),
}

/// The reduced-form (`Psi_h`) estimation error — the piece
/// [`ArVariance`] cannot see and the module docs measure the cost of.
///
/// # What the caller must supply
///
/// Write `w_{h,i} = e_i' (Psi_hat_h - Psi_h) gamma`, the reduced-form part of
/// the numerator moment's sampling error (the `unit` factor is applied by this
/// crate, so nothing here depends on it and `unit`-equivariance survives).
/// Then:
///
/// * `psi_var[h]` is `T_O * Cov(Psi_hat_h gamma)`, the `n x n` matrix whose
///   `(i, i)` entry is `T_O * Var(w_{h,i})`. Only the diagonal is read; the
///   full matrix is accepted because that is what the natural formula
///   produces. `psi_var[0]` is zero — `Psi_0 = I` is not estimated.
/// * `psi_gamma_cov[h][(i, j)]` is `T_O * Cov(w_{h,i}, gamma_hat_j)`, and
///   `None` asserts it is zero.
///
/// [`psi_reduced_form_cov`] computes `psi_var` from a VAR's coefficient
/// covariance by the delta method.
///
/// # Why `None` is the right default for the cross-covariance
///
/// With OLS on a design that includes an intercept, the lag-block coefficient
/// error is `(Z'Z)^{-1} sum_t z_t u_t'`, and its covariance with
/// `gamma_hat = mean_t m_t u_t` reduces to
/// `(Z'Z)^{-1} (sum_t z_t) * E[m_t u_t u_t'] / T_O`. But `(Z'Z)^{-1} Z' 1` is
/// the OLS fit of a constant vector on `Z`, which is the unit vector on the
/// intercept and **zero on every lag**. So the term vanishes exactly in the
/// lag block, under i.i.d. innovations and a proxy uncorrelated with the
/// regressors. Simulated at `T = 300` it comes to two to five percent of its
/// Cauchy-Schwarz bound in both a strong and a weak arm. Supply it anyway if
/// you have a joint bootstrap that estimates it, or if conditional
/// heteroskedasticity makes the argument above uncomfortable.
///
/// # This must vanish with `gamma`
///
/// Both terms are quadratic and linear in `gamma` respectively, so they die
/// with the instrument's relevance — which is what keeps the set weak-IV
/// robust. A `Psi_h` variance scaled by the *normalized* impact `b` instead
/// would keep the set bounded under weak instruments and destroy exactly the
/// property this construction exists for.
#[derive(Debug, Clone, Copy)]
pub struct ArReducedForm<'a> {
    /// `T_O * Cov(Psi_hat_h gamma)`, one `n x n` matrix per horizon, the same
    /// length as `psi`.
    pub psi_var: &'a [Mat<f64>],
    /// `T_O * Cov(Psi_hat_h gamma, gamma_hat)`, one `n x n` matrix per
    /// horizon; `None` asserts it is zero.
    pub psi_gamma_cov: Option<&'a [Mat<f64>]>,
}

/// The full variance specification: which estimator supplies `Omega`, and
/// whether reduced-form estimation error is propagated.
///
/// An [`ArVariance`] converts into this with `reduced_form: None`, so
/// `proxy_ar_sets(.., ArVariance::Hc0, ..)` still compiles and still means
/// "moment covariance only" — but that path now reports
/// [`ProxyArResult::reduced_form_uncertainty`] `= false` and a `None`
/// [`ProxyArResult::level`], because its coverage holds only conditional on
/// the reduced form.
#[derive(Debug, Clone, Copy)]
pub struct ArVarianceSpec<'a> {
    /// The estimator for the moment covariance `Omega`.
    pub moment: ArVariance<'a>,
    /// The reduced-form correction, or `None` to omit it.
    pub reduced_form: Option<ArReducedForm<'a>>,
}

impl<'a> ArVarianceSpec<'a> {
    /// `Omega` only — the identification step's variance, with reduced-form
    /// estimation error omitted. Equivalent to passing the [`ArVariance`]
    /// directly.
    pub fn moment_only(moment: ArVariance<'a>) -> Self {
        Self {
            moment,
            reduced_form: None,
        }
    }

    /// `Omega` plus the reduced-form correction: the full joint variance.
    pub fn with_reduced_form(moment: ArVariance<'a>, reduced_form: ArReducedForm<'a>) -> Self {
        Self {
            moment,
            reduced_form: Some(reduced_form),
        }
    }
}

impl<'a> From<ArVariance<'a>> for ArVarianceSpec<'a> {
    fn from(moment: ArVariance<'a>) -> Self {
        Self::moment_only(moment)
    }
}

/// Which critical value the test inverts.
#[derive(Debug, Clone, Copy)]
pub enum ArCritical {
    /// `chi2_{1, level}` — one instrument, one restriction. About `3.8415` at
    /// `level = 0.95`. The general Anderson-Rubin default.
    Chi2 {
        /// Confidence level `1 - alpha`, strictly inside `(0, 1)`.
        level: f64,
    },
    /// `F_{1, T_O - 2, level}`, a finite-sample flavour. Slightly larger than
    /// the chi-square value, so slightly wider sets; the two agree as `T_O`
    /// grows. Which of the two the original method prescribes is not
    /// confirmed here.
    F {
        /// Confidence level `1 - alpha`, strictly inside `(0, 1)`.
        level: f64,
    },
    /// A critical value supplied directly — for a bootstrapped critical value,
    /// or to place the test exactly at the boundedness knife edge.
    Value(f64),
}

/// One `(horizon, variable)` cell: its confidence set, its point estimate,
/// and every coefficient needed to re-derive or re-test the set.
#[derive(Debug, Clone, Copy)]
pub struct ArCell {
    /// The confidence set for `lambda_{i,h}`.
    pub set: ArSet,
    /// The point estimate `unit * (Psi_h gamma)_i / gamma_k`, computed the
    /// same way [`crate::proxy_svar`] computes `irf[h][i]` (as
    /// `(Psi_h b)_i` with `b = unit * gamma / gamma_k`), so the two agree
    /// bit-for-bit.
    pub point: f64,
    /// Quadratic coefficient `A = T_O*q0^2 - c*v2`. Cell-independent.
    pub a: f64,
    /// Quadratic coefficient `B = 2*(c*v1 - T_O*q1*q0)`.
    pub b: f64,
    /// Quadratic coefficient `C = T_O*q1^2 - c*v0`.
    pub c: f64,
    /// Numerator moment `q1 = unit * (Psi_h gamma)_i`.
    pub q1: f64,
    /// Denominator moment `q0 = gamma_k`. Cell-independent.
    pub q0: f64,
    /// Variance coefficient `v0 = unit^2 * (Psi_h Omega Psi_h')[i,i]`.
    pub v0: f64,
    /// Variance coefficient `v1 = unit * (Psi_h Omega)[i,k]`.
    pub v1: f64,
    /// Variance coefficient `v2 = Omega[k,k]`. Cell-independent.
    pub v2: f64,
    /// Whether zero is outside the set — literally `!set.contains(0.0)`,
    /// which away from the knife edge is `C > 0`, i.e. `T_O*q1^2 > c*v0`.
    ///
    /// It is defined through the set rather than through `C` so that the two
    /// answers cannot disagree: [`ArSet::Whole`] at the knife edge is produced
    /// by `C <= tau_c`, and reading `C > 0.0` exactly would report
    /// "excludes zero" for a set that is the entire real line.
    ///
    /// This is a robust test of `H0: unit * e_i' Psi_h gamma = 0` and it
    /// survives weak identification: when the set is unbounded the *magnitude*
    /// question dies but "is this response nonzero" can still be answered.
    ///
    /// **It does not answer the sign question for an unbounded set.** An
    /// [`ArSet::Exterior`] such as `(-inf, -2.27] u [0.0368, +inf)` excludes
    /// zero while containing values of *both* signs — the data reject a small
    /// positive response and nothing else. Read a sign off this flag only when
    /// [`ArSet::is_bounded`] is `true`; otherwise inspect the set.
    pub excludes_zero: bool,
}

impl ArCell {
    /// The null-imposed moment `g_hat(lam) = q1 - lam*q0`.
    pub fn moment(&self, lam: f64) -> f64 {
        self.q1 - lam * self.q0
    }

    /// The null-imposed variance `V_hat(lam) = v0 - 2*lam*v1 + lam^2*v2`.
    pub fn variance(&self, lam: f64) -> f64 {
        self.v0 - 2.0 * lam * self.v1 + lam * lam * self.v2
    }

    /// The inverted quadratic `A*lam^2 + B*lam + C`; the set is where this is
    /// `<= 0`.
    pub fn quadratic(&self, lam: f64) -> f64 {
        (self.a * lam + self.b) * lam + self.c
    }

    /// The Anderson-Rubin statistic `T_O * g_hat(lam)^2 / V_hat(lam)`.
    ///
    /// Returns `NaN` at a `lam` where the variance vanishes (`0/0`); that
    /// happens exactly at `lam = unit` in the `(norm_var, h = 0)` cell, so a
    /// grid-based cross-check must skip it rather than evaluate the ratio.
    pub fn ar_stat(&self, lam: f64, n_proxy: usize) -> f64 {
        let g = self.moment(lam);
        n_proxy as f64 * g * g / self.variance(lam)
    }
}

/// Weak-instrument-robust confidence sets for every `(horizon, variable)`
/// cell, plus the diagnostics that say how much to trust them.
#[derive(Debug, Clone)]
pub struct ProxyArResult {
    /// `cells[h][i]` for `h = 0..=H` and `i = 0..n`. Pointwise sets — see the
    /// module docs on why this is not a joint band.
    pub cells: Vec<Vec<ArCell>>,
    /// The moment covariance `Omega` actually used (`n x n`).
    pub omega: Mat<f64>,
    /// The identifying moment `gamma = E[m_t u_t]`, identical to
    /// [`crate::ProxySvarResult::cov_um`].
    pub cov_um: Vec<f64>,
    /// The unit-effect impact vector `b = unit * gamma / gamma[norm_var]`,
    /// identical to [`crate::ProxySvarResult::impact`].
    pub impact: Vec<f64>,
    /// Overlap count `T_O` — the number of finite proxy observations. The AR
    /// scaling uses this, never the full residual length `T`.
    pub n_proxy: usize,
    /// The confidence level these sets actually carry, or `None` when they
    /// carry none.
    ///
    /// `Some(1 - alpha)` requires **both** that a level was given (an
    /// [`ArCritical::Value`] leaves it `None`) **and** that
    /// [`Self::reduced_form_uncertainty`] is `true`. Without the reduced-form
    /// term the sets are a `1 - alpha` set only *conditional on the reduced
    /// form*, and the measured unconditional coverage with a strong
    /// instrument falls to `.119` by `h = 8` — so this field reports `None`
    /// rather than a number the sets do not earn. The level that was
    /// requested is still recoverable from [`Self::critical_value`].
    pub level: Option<f64>,
    /// Whether `V_hat(lam)` includes reduced-form (`Psi_h`) estimation error,
    /// i.e. whether an [`ArReducedForm`] was supplied.
    ///
    /// `false` means the sets condition on `Psi_h` as if it were known. That
    /// is valid under weak-instrument asymptotics and badly invalid with a
    /// strong instrument at any horizon past impact; see the module docs for
    /// the measured collapse. This flag is the machine-readable form of that
    /// warning — branch on it, do not rely on a reader noticing the docs.
    pub reduced_form_uncertainty: bool,
    /// The critical value `c` that was inverted.
    pub critical_value: f64,
    /// `T_O * gamma_k^2 / Omega[k,k]` — the robust relevance statistic that
    /// alone decides boundedness for every cell.
    ///
    /// The rule is `ar_bound_stat > c`, and **only** this statistic obeys it.
    /// [`Self::first_stage_f`] is asymptotically equivalent but numerically
    /// different, so a user whose `first_stage_f` sits just the other side of
    /// `c` will otherwise file a bug against the documentation.
    pub ar_bound_stat: f64,
    /// Whether every cell's set is bounded, `= ar_bound_stat > c` up to the
    /// leading-coefficient tolerance. Weak identification is a property of
    /// the denominator alone, so this is all-or-nothing across the grid.
    pub ar_bounded_all: bool,
    /// The HC1-robust first-stage `F` that [`crate::proxy_svar`] reports,
    /// carried here so it can be printed beside every set.
    ///
    /// A bounded set certifies only `ar_bound_stat > c` (about `3.84` at 95%)
    /// — it does **not** certify a strong instrument. An `F` of `4.5` with a
    /// tidy bounded set is still a weak instrument.
    pub first_stage_f: f64,
}

impl ProxyArResult {
    /// The Anderson-Rubin statistic at `lam` for cell `(h, i)`, or `None` if
    /// the indices are out of range.
    pub fn ar_stat(&self, h: usize, i: usize, lam: f64) -> Option<f64> {
        self.cells
            .get(h)
            .and_then(|row| row.get(i))
            .map(|cell| cell.ar_stat(lam, self.n_proxy))
    }

    /// The number of cells whose set is bounded, and the total number of
    /// cells. Under this construction the first is either `0` or the second.
    pub fn bounded_count(&self) -> (usize, usize) {
        let mut bounded = 0;
        let mut total = 0;
        for row in &self.cells {
            for cell in row {
                total += 1;
                if cell.set.is_bounded() {
                    bounded += 1;
                }
            }
        }
        (bounded, total)
    }
}

/// Weak-instrument-robust (Anderson-Rubin) confidence sets for the
/// unit-effect-normalized proxy-SVAR impulse responses.
///
/// Inputs mirror [`crate::proxy_svar`] exactly, minus `sigma_u` (no shock
/// series is produced here): `u` is the `T x n` residual matrix, `proxy` the
/// length-`T` instrument with non-finite entries marking unavailability,
/// `psi` the reduced-form MA sequence `Psi_0..Psi_H` with `Psi_0 = I`,
/// `norm_var` the variable whose impact is normalized to `unit`.
///
/// Pass the **cumulated** MA sequence to get confidence sets for cumulative
/// responses; the algebra is unchanged.
///
/// The overlap, the means, `gamma`, `b` and the point responses are computed
/// in exactly the operation order [`crate::proxy_svar`] uses, so
/// `cells[h][i].point` equals that function's `irf[h][i]` bit-for-bit on the
/// same inputs. Only the finite proxy rows enter, and the AR statistic is
/// scaled by that overlap count `T_O`, never by `T`: prepending NaNs to the
/// proxy leaves the sets unchanged.
///
/// `variance` accepts either an [`ArVariance`] — the moment covariance alone,
/// which leaves `Psi_h` estimation error out and therefore returns
/// `level: None` — or an [`ArVarianceSpec`] carrying an [`ArReducedForm`].
/// **Pass the latter whenever `psi` came from an estimated VAR**; see the
/// module docs for what the omission costs.
///
/// # Errors
///
/// * [`IdentError::Dimension`] if `proxy`, `psi`, or a supplied `Omega` do
///   not match the `T x n` residual shape;
/// * [`IdentError::RestrictionOutOfRange`] if `norm_var >= n`;
/// * [`IdentError::NonFinite`] if `u`, `psi`, or a supplied `Omega` contain a
///   NaN or infinity (the proxy may carry NaNs — they mark unavailability);
/// * [`IdentError::InvalidArgument`] if `unit` is zero or non-finite, the
///   confidence level is outside `(0, 1)`, a supplied critical value is not
///   positive and finite, the HAC bandwidth is not below the overlap count,
///   the proxy overlap has fewer than three finite observations, the proxy
///   has no variance over the overlap, `gamma[norm_var]` is zero,
///   `Omega[norm_var, norm_var]` is not positive, a supplied `Omega` is not
///   symmetric, a supplied [`ArReducedForm`] has a negative variance on its
///   diagonal, or the moment covariance fails the per-cell
///   positive-semidefiniteness check;
/// * [`IdentError::Stats`] if the quantile function behind the critical value
///   rejects the level.
pub fn proxy_ar_sets<'a>(
    u: MatRef<'_, f64>,
    proxy: &[f64],
    psi: &[Mat<f64>],
    norm_var: usize,
    unit: f64,
    variance: impl Into<ArVarianceSpec<'a>>,
    critical: ArCritical,
) -> Result<ProxyArResult, IdentError> {
    let ArVarianceSpec {
        moment: variance,
        reduced_form,
    } = variance.into();
    let t = u.nrows();
    let n = u.ncols();

    if n == 0 || t == 0 {
        return Err(IdentError::InvalidArgument {
            what: "residual matrix u must have at least one row and one column",
        });
    }
    if proxy.len() != t {
        return Err(IdentError::Dimension {
            what: "proxy length must equal the number of residual rows T",
            expected: t,
            got: proxy.len(),
        });
    }
    if psi.is_empty() {
        return Err(IdentError::InvalidArgument {
            what: "psi must contain at least Psi_0 (the identity)",
        });
    }
    for ph in psi {
        if ph.nrows() != n || ph.ncols() != n {
            return Err(IdentError::Dimension {
                what: "every MA matrix Psi_h must be n x n",
                expected: n,
                got: if ph.nrows() != n {
                    ph.nrows()
                } else {
                    ph.ncols()
                },
            });
        }
    }
    if norm_var >= n {
        return Err(IdentError::RestrictionOutOfRange {
            what: "norm_var",
            index: norm_var,
            bound: n,
        });
    }
    if !unit.is_finite() || unit == 0.0 {
        return Err(IdentError::InvalidArgument {
            what: "unit (the impact size on norm_var) must be nonzero and finite",
        });
    }
    for j in 0..n {
        for i in 0..t {
            if !u[(i, j)].is_finite() {
                return Err(IdentError::NonFinite { what: "u" });
            }
        }
    }
    for ph in psi {
        for j in 0..n {
            for i in 0..n {
                if !ph[(i, j)].is_finite() {
                    return Err(IdentError::NonFinite { what: "psi" });
                }
            }
        }
    }
    if let Some(rf) = reduced_form {
        check_reduced_form(&rf, psi.len(), n)?;
    }

    // Overlap and moments, in the same order proxy_svar computes them so the
    // two agree bit-for-bit.
    let overlap: Vec<usize> = (0..t).filter(|&r| proxy[r].is_finite()).collect();
    let n_proxy = overlap.len();
    if n_proxy < 3 {
        return Err(IdentError::InvalidArgument {
            what: "proxy overlap has fewer than 3 finite observations; cannot run the first stage",
        });
    }
    let no = n_proxy as f64;

    let mut mbar = 0.0;
    for &r in &overlap {
        mbar += proxy[r];
    }
    mbar /= no;
    let mut ubar = vec![0.0f64; n];
    for (j, slot) in ubar.iter_mut().enumerate() {
        let mut s = 0.0;
        for &r in &overlap {
            s += u[(r, j)];
        }
        *slot = s / no;
    }

    let mut gamma = vec![0.0f64; n];
    for (j, slot) in gamma.iter_mut().enumerate() {
        let mut s = 0.0;
        for &r in &overlap {
            s += (proxy[r] - mbar) * (u[(r, j)] - ubar[j]);
        }
        *slot = s / no;
    }

    let g_norm = gamma[norm_var];
    if g_norm == 0.0 {
        return Err(IdentError::InvalidArgument {
            what:
                "gamma[norm_var] is zero: the instrument has no first-stage relevance for norm_var",
        });
    }
    let b: Vec<f64> = gamma.iter().map(|&g| unit * (g / g_norm)).collect();

    // The centered moment contributions g~_t = m~_t u~_t - gamma, stored in
    // overlap order; every variance estimator is a function of these.
    let mut gtil = Mat::<f64>::zeros(n_proxy, n);
    for (p, &r) in overlap.iter().enumerate() {
        let md = proxy[r] - mbar;
        for j in 0..n {
            gtil[(p, j)] = md * (u[(r, j)] - ubar[j]) - gamma[j];
        }
    }

    let omega = match variance {
        ArVariance::Hc0 => omega_hc0(gtil.as_ref(), no),
        ArVariance::HacBartlett { lags } => {
            if lags >= n_proxy {
                return Err(IdentError::InvalidArgument {
                    what: "HAC bandwidth must be smaller than the proxy overlap count",
                });
            }
            omega_hac_bartlett(gtil.as_ref(), &overlap, t, lags, no)
        }
        ArVariance::Supplied(om) => {
            if om.nrows() != n || om.ncols() != n {
                return Err(IdentError::Dimension {
                    what: "the supplied moment covariance Omega must be n x n",
                    expected: n,
                    got: if om.nrows() != n {
                        om.nrows()
                    } else {
                        om.ncols()
                    },
                });
            }
            let mut scale = 0.0f64;
            for j in 0..n {
                for i in 0..n {
                    if !om[(i, j)].is_finite() {
                        return Err(IdentError::NonFinite { what: "omega" });
                    }
                    scale = scale.max(om[(i, j)].abs());
                }
            }
            for j in 0..n {
                for i in 0..j {
                    if (om[(i, j)] - om[(j, i)]).abs() > 1e-10 * (1.0 + scale) {
                        return Err(IdentError::InvalidArgument {
                            what: "the supplied moment covariance Omega is not symmetric",
                        });
                    }
                }
            }
            Mat::from_fn(n, n, |i, j| om[(i, j)])
        }
    };

    let v2 = omega[(norm_var, norm_var)];
    if !v2.is_finite() || v2 <= 0.0 {
        return Err(IdentError::InvalidArgument {
            what: "Omega[norm_var, norm_var] is not positive: the identifying moment has no \
                   estimated variance, so no confidence set can be formed",
        });
    }

    // Critical value.
    let (crit, level) = match critical {
        ArCritical::Chi2 { level } => {
            check_level(level)?;
            let z = inv_norm_cdf(0.5 + 0.5 * level)?;
            (z * z, Some(level))
        }
        ArCritical::F { level } => {
            check_level(level)?;
            // F_{1,d} = t_d^2 and P(t_d^2 <= x) = I_{x/(x+d)}(1/2, d/2), so
            // inverting the incomplete beta at `level` and undoing the change
            // of variable gives the quantile. d = T_O - 2 >= 1 here.
            let d = (n_proxy - 2) as f64;
            let z = inv_beta_inc(0.5, 0.5 * d, level)?;
            if !z.is_finite() || z >= 1.0 {
                return Err(IdentError::InvalidArgument {
                    what: "confidence level is too close to 1 for the F critical value to be \
                           representable; use a smaller level or ArCritical::Chi2",
                });
            }
            (d * z / (1.0 - z), Some(level))
        }
        ArCritical::Value(c) => {
            if !c.is_finite() || c <= 0.0 {
                return Err(IdentError::InvalidArgument {
                    what: "a supplied critical value must be positive and finite",
                });
            }
            (c, None)
        }
    };

    // The boundedness switch: one scalar for the entire grid.
    let a_coef = no * g_norm * g_norm - crit * v2;
    let tau_a = TAU_A_REL * (no * g_norm * g_norm).max(crit * v2);
    let ar_bound_stat = no * g_norm * g_norm / v2;
    let ar_bounded_all = a_coef > tau_a;

    // Per-horizon: P_h = Psi_h Omega (O(n^3)); the cell coefficients need
    // only its k-th column and the diagonal of P_h Psi_h'.
    let mut cells: Vec<Vec<ArCell>> = Vec::with_capacity(psi.len());
    for (h, ph) in psi.iter().enumerate() {
        let mut p_h = Mat::<f64>::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                let mut s = 0.0;
                for l in 0..n {
                    s += ph[(i, l)] * omega[(l, j)];
                }
                p_h[(i, j)] = s;
            }
        }
        let mut row: Vec<ArCell> = Vec::with_capacity(n);
        for i in 0..n {
            // Point estimate, computed exactly as proxy_svar does.
            let mut point = 0.0;
            for (j, &bj) in b.iter().enumerate() {
                point += ph[(i, j)] * bj;
            }
            let mut psi_gamma = 0.0;
            for (j, &gj) in gamma.iter().enumerate() {
                psi_gamma += ph[(i, j)] * gj;
            }
            let mut qdiag = 0.0;
            for j in 0..n {
                qdiag += p_h[(i, j)] * ph[(i, j)];
            }

            // Reduced-form correction. The linearized moment error is
            // a(lam)'(gamma_hat - gamma) + unit*w_{h,i}, so
            // V(lam) = a'Omega a + 2 a'c + s2 with
            // c_j = T_O*Cov(gamma_hat_j, unit*w) and s2 = T_O*Var(unit*w).
            // Expanding a(lam) = unit*Psi_h' e_i - lam*e_k splits that into a
            // constant on v0 and a constant on v1; nothing touches v2 or q0,
            // which is why boundedness (and weak-IV robustness) is untouched.
            let (rf_v0, rf_v1) = match reduced_form {
                None => (0.0, 0.0),
                Some(rf) => {
                    let s2 = rf.psi_var[h][(i, i)];
                    let (cross_num, cross_den) = match rf.psi_gamma_cov {
                        None => (0.0, 0.0),
                        Some(xs) => {
                            let x = &xs[h];
                            let mut s = 0.0;
                            for j in 0..n {
                                s += ph[(i, j)] * x[(i, j)];
                            }
                            (s, x[(i, norm_var)])
                        }
                    };
                    (unit * unit * (s2 + 2.0 * cross_num), unit * cross_den)
                }
            };

            row.push(ar_cell(
                n_proxy,
                crit,
                ArMoments {
                    q1: unit * psi_gamma,
                    q0: g_norm,
                    v0: unit * unit * qdiag + rf_v0,
                    v1: unit * p_h[(i, norm_var)] + rf_v1,
                    v2,
                    point,
                },
            )?);
        }
        cells.push(row);
    }

    let first_stage_f = first_stage_hc1(u, proxy, &overlap, norm_var, mbar, ubar[norm_var], no)?;

    let reduced_form_uncertainty = reduced_form.is_some();
    Ok(ProxyArResult {
        cells,
        omega,
        cov_um: gamma,
        impact: b,
        n_proxy,
        // A level is only claimed when the sets earn it: the moment-only
        // variance conditions on the reduced form, and its unconditional
        // coverage with a strong instrument is measured in the module docs.
        level: level.filter(|_| reduced_form_uncertainty),
        reduced_form_uncertainty,
        critical_value: crit,
        ar_bound_stat,
        ar_bounded_all,
        first_stage_f,
    })
}

/// Shape, finiteness and sign checks on a supplied [`ArReducedForm`].
fn check_reduced_form(rf: &ArReducedForm<'_>, horizons: usize, n: usize) -> Result<(), IdentError> {
    let check = |mats: &[Mat<f64>], what: &'static str, diag_nonneg: bool| {
        if mats.len() != horizons {
            return Err(IdentError::Dimension {
                what,
                expected: horizons,
                got: mats.len(),
            });
        }
        for m in mats {
            if m.nrows() != n || m.ncols() != n {
                return Err(IdentError::Dimension {
                    what: "every reduced-form correction matrix must be n x n",
                    expected: n,
                    got: if m.nrows() != n { m.nrows() } else { m.ncols() },
                });
            }
            for j in 0..n {
                for i in 0..n {
                    if !m[(i, j)].is_finite() {
                        return Err(IdentError::NonFinite {
                            what: "reduced-form correction",
                        });
                    }
                }
                if diag_nonneg && m[(j, j)] < 0.0 {
                    return Err(IdentError::InvalidArgument {
                        what: "ArReducedForm::psi_var has a negative diagonal entry; it is a \
                               covariance matrix of Psi_hat_h gamma, so its diagonal is a \
                               variance and cannot be negative",
                    });
                }
            }
        }
        Ok(())
    };
    check(
        rf.psi_var,
        "ArReducedForm::psi_var must have one n x n matrix per MA matrix in psi",
        true,
    )?;
    if let Some(x) = rf.psi_gamma_cov {
        check(
            x,
            "ArReducedForm::psi_gamma_cov must have one n x n matrix per MA matrix in psi",
            false,
        )?;
    }
    Ok(())
}

/// The delta-method reduced-form correction `T_O * Cov(Psi_hat_h gamma)` for
/// every horizon — the [`ArReducedForm::psi_var`] input, built from a fitted
/// VAR's coefficient covariance.
///
/// # The algebra
///
/// `vec(Psi_hat_h) - vec(Psi_h) = G_h (alpha_hat - alpha) + o_p(T^{-1/2})`
/// with the Lütkepohl (1990) Jacobian
///
/// ```text
/// G_h = sum_{m=0}^{h-1} J (A')^{h-1-m} (x) Psi_m,   J = [I_n  0],
/// ```
///
/// `A` the companion matrix of `coefs` and `(x)` the Kronecker product. Then
/// `Psi_hat_h gamma - Psi_h gamma = Gamma_h (alpha_hat - alpha)` with
/// `Gamma_h = (gamma' (x) I_n) G_h`, so the correction is the sandwich
/// `T_O * Gamma_h Cov(alpha_hat) Gamma_h'`. This is the same Jacobian the
/// VAR crate's asymptotic IRF bands use; it is reproduced here because
/// `tsecon-ident` cannot depend on `tsecon-var` (that crate depends on this
/// one), so the reduced form arrives as an input, exactly as `psi` does.
///
/// `psi_var[0]` is zero: `Psi_0 = I` is not estimated.
///
/// # The `cov_alpha` layout
///
/// `cov_alpha` is `Cov(vec(alpha_hat))` for the **lag block only** —
/// deterministic terms dropped — indexed `r = a*n + e`, where `a` runs over
/// the `n*p` stacked lag regressors `(y_{t-1}', .., y_{t-p}')` and `e` over
/// the `n` equations. For OLS that is
/// `((Z'Z)^{-1} restricted to the lag rows and columns) (x) Sigma_u` in the
/// NumPy `kron` convention. It is `Cov(alpha_hat)` itself, **not** `T` times
/// it: the `T_O` scaling the AR statistic needs is applied here.
///
/// From `tsecon-var`'s `VarResults` (which this crate cannot name, since the
/// dependency runs the other way) that is, with `k = res.neqs`,
/// `p = res.spec.lags` and `n_trend = res.df_model - k*p` the number of
/// deterministic regressors:
///
/// ```text
/// cov_alpha[(r, c)] = res.zz_inv[(r/k + n_trend, c/k + n_trend)]
///                   * res.sigma_u[(r%k, c%k)],   r, c in 0..p*k*k
/// coefs             = res.coefs                  // A_1..A_p, A_i[(r, c)]
///                                                // = effect of variable c
///                                                // at lag i on variable r
/// ```
///
/// The `n_trend` offset is what drops the intercept and trend rows, and it
/// must not be skipped: the deterministic block is exactly where the
/// cross-covariance with `gamma_hat` lives (see [`ArReducedForm`]).
///
/// # Errors
///
/// [`IdentError::Dimension`] if the shapes disagree, [`IdentError::NonFinite`]
/// on a NaN or infinity, [`IdentError::InvalidArgument`] if `coefs` is empty
/// or `n_proxy` is zero.
pub fn psi_reduced_form_cov(
    psi: &[Mat<f64>],
    coefs: &[Mat<f64>],
    cov_alpha: MatRef<'_, f64>,
    gamma: &[f64],
    n_proxy: usize,
) -> Result<Vec<Mat<f64>>, IdentError> {
    let n = gamma.len();
    let p = coefs.len();
    if n == 0 || p == 0 {
        return Err(IdentError::InvalidArgument {
            what: "psi_reduced_form_cov needs a nonempty gamma and at least one lag matrix",
        });
    }
    if n_proxy == 0 {
        return Err(IdentError::InvalidArgument {
            what: "n_proxy must be positive",
        });
    }
    if psi.is_empty() {
        return Err(IdentError::InvalidArgument {
            what: "psi must contain at least Psi_0 (the identity)",
        });
    }
    for m in psi.iter().chain(coefs.iter()) {
        if m.nrows() != n || m.ncols() != n {
            return Err(IdentError::Dimension {
                what: "psi and coefs matrices must all be n x n with n = gamma.len()",
                expected: n,
                got: if m.nrows() != n { m.nrows() } else { m.ncols() },
            });
        }
    }
    let np = n * p;
    let dim = np * n;
    if cov_alpha.nrows() != dim || cov_alpha.ncols() != dim {
        return Err(IdentError::Dimension {
            what: "cov_alpha must be (n*n*p) x (n*n*p): the lag-block coefficient covariance",
            expected: dim,
            got: cov_alpha.nrows(),
        });
    }
    for j in 0..dim {
        for i in 0..dim {
            if !cov_alpha[(i, j)].is_finite() {
                return Err(IdentError::NonFinite { what: "cov_alpha" });
            }
        }
    }
    for &g in gamma {
        if !g.is_finite() {
            return Err(IdentError::NonFinite { what: "gamma" });
        }
    }

    // Companion transpose A', and the first n rows of its powers.
    let mut at = Mat::<f64>::zeros(np, np);
    for (l, a_l) in coefs.iter().enumerate() {
        for i in 0..n {
            for j in 0..n {
                // companion[(i, l*n + j)] = A_{l+1}[(i, j)]; transposed.
                at[(l * n + j, i)] = a_l[(i, j)];
            }
        }
    }
    for i in 0..n * (p - 1) {
        // companion[(n + i, i)] = 1; transposed.
        at[(i, n + i)] = 1.0;
    }
    let horizon = psi.len() - 1;
    // atpow[j] = first n rows of (A')^j, an n x np block.
    let mut atpow: Vec<Mat<f64>> = Vec::with_capacity(horizon.max(1));
    let mut cur = Mat::<f64>::from_fn(np, np, |i, j| f64::from(u8::from(i == j)));
    for _ in 0..horizon {
        atpow.push(cur.submatrix(0, 0, n, np).to_owned());
        cur = &cur * &at;
    }

    let mut out: Vec<Mat<f64>> = Vec::with_capacity(psi.len());
    out.push(Mat::<f64>::zeros(n, n));
    for h in 1..=horizon {
        // Gamma_h (n x dim): row i is sum_j gamma_j * G_h[j*n + i, :], and
        // G_h[(jj*n + ii), (a*n + e)] = sum_m atpow[h-1-m][(jj, a)] *
        // psi[m][(ii, e)]. Accumulating directly avoids ever forming G_h.
        let mut gm = Mat::<f64>::zeros(n, dim);
        for m in 0..h {
            let ap = &atpow[h - 1 - m];
            let pm = &psi[m];
            for a in 0..np {
                // wa = sum_jj gamma_jj * atpow[h-1-m][(jj, a)]
                let mut wa = 0.0;
                for (jj, &gj) in gamma.iter().enumerate() {
                    wa += gj * ap[(jj, a)];
                }
                if wa == 0.0 {
                    continue;
                }
                for ii in 0..n {
                    for e in 0..n {
                        gm[(ii, a * n + e)] += wa * pm[(ii, e)];
                    }
                }
            }
        }
        // T_O * Gamma_h cov_alpha Gamma_h'.
        let mut left = Mat::<f64>::zeros(n, dim);
        for i in 0..n {
            for c in 0..dim {
                let mut s = 0.0;
                for r in 0..dim {
                    s += gm[(i, r)] * cov_alpha[(r, c)];
                }
                left[(i, c)] = s;
            }
        }
        let scale = n_proxy as f64;
        let mut cov = Mat::<f64>::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                let mut s = 0.0;
                for c in 0..dim {
                    s += left[(i, c)] * gm[(j, c)];
                }
                cov[(i, j)] = scale * s;
            }
        }
        out.push(cov);
    }
    Ok(out)
}

/// Second-order (simulation-based) reduced-form correction
/// `T_O * Cov(Psi_hat_h gamma)` — the [`ArReducedForm::psi_var`] input again,
/// but with the coefficient uncertainty pushed through the **exact** nonlinear
/// map `alpha -> Psi_h` instead of its first-order linearization.
///
/// # Why a second order exists (audit round 6, finding 8)
///
/// [`psi_reduced_form_cov`] evaluates the delta-method Jacobian at the
/// estimated coefficients. At long horizons that couples the propagated
/// variance to the estimate itself: a fitted VAR that draws less persistent
/// than the truth makes `Psi_hat_h` too small *and* makes the variance
/// propagated for it too small, in the same replications. The measured cost
/// on the module docs' own DGP is a one-sided coverage decline past the
/// published table — 0.889 at `h = 12` against a nominal 0.95 (0.830 on a
/// routine VAR(1) at `T = 250`), with every miss on the side the truth sits
/// on. The measured correlation between the delta standard deviation and the
/// point error at `h = 12` is about `+0.7`, while their *average* ratio is
/// ~0.94 — the first-order variance is right on average and wrong in exactly
/// the draws that matter.
///
/// This function replaces the linearization with the exact propagation of the
/// same Gaussian coefficient uncertainty: draw
/// `alpha* ~ N(alpha_hat, Cov(alpha_hat))` in antithetic pairs from the
/// seeded Philox stream, map each draw through the full MA recursion
/// `Psi_0 = I, Psi_h = sum_i Psi_{h-i} A_i`, and return the sample covariance
/// of `Psi_h(alpha*) gamma` scaled by `T_O`. To first order this equals the
/// delta method; the difference is the convexity of `alpha -> Psi_h`, which
/// grows with the horizon and is exactly what re-inflates the variance in the
/// under-persistent draws. Measured like-for-like on the same 500 seeded
/// replications: coverage at `h = 12` moves from 0.889 to 0.964 (VAR(2),
/// `T = 300`) and 0.830 to 0.932 (VAR(1), `T = 250`), at a median width cost
/// of ~1.15x at `h = 8` and ~1.45x at `h = 12`; weak-instrument behaviour is
/// unchanged (the correction is still quadratic in `gamma`, so it still
/// vanishes with relevance).
///
/// # Contract
///
/// * `horizon` is the largest horizon to produce; the result has
///   `horizon + 1` entries and entry `0` is zero (`Psi_0 = I` is not
///   estimated).
/// * `coefs` / `cov_alpha` / `gamma` / `n_proxy` are exactly
///   [`psi_reduced_form_cov`]'s inputs (same layout for `cov_alpha`,
///   documented there). The MA sequence is rebuilt internally from `coefs`,
///   so this function corresponds to the **raw** (non-cumulated) `psi`.
/// * `draws` must be even (the sampler is antithetic) and at least 32;
///   256 is a good default — the estimate is a `draws`-sample covariance,
///   and its own Monte-Carlo error shrinks like `1/sqrt(draws)`.
/// * `seed` makes the result bit-reproducible; the same inputs and seed give
///   the same matrices on every platform.
///
/// The returned matrices are sample covariances, hence positive semidefinite
/// by construction, which is what the per-cell PSD check downstream assumes.
///
/// # Errors
///
/// [`IdentError::Dimension`] / [`IdentError::NonFinite`] as for
/// [`psi_reduced_form_cov`]; [`IdentError::InvalidArgument`] if `draws` is
/// odd or below 32, or `coefs` is empty, or `n_proxy` is zero;
/// [`IdentError::Linalg`] if `cov_alpha` is too indefinite to factor even
/// with the jitter ladder (a genuinely indefinite coefficient covariance).
pub fn psi_reduced_form_cov_mc(
    horizon: usize,
    coefs: &[Mat<f64>],
    cov_alpha: MatRef<'_, f64>,
    gamma: &[f64],
    n_proxy: usize,
    draws: usize,
    seed: u64,
) -> Result<Vec<Mat<f64>>, IdentError> {
    let n = gamma.len();
    let p = coefs.len();
    if n == 0 || p == 0 {
        return Err(IdentError::InvalidArgument {
            what: "psi_reduced_form_cov_mc needs a nonempty gamma and at least one lag matrix",
        });
    }
    if n_proxy == 0 {
        return Err(IdentError::InvalidArgument {
            what: "n_proxy must be positive",
        });
    }
    if draws < 32 || draws % 2 != 0 {
        return Err(IdentError::InvalidArgument {
            what: "draws must be an even number of at least 32 (the sampler works in \
                   antithetic pairs; 256 is a reasonable default)",
        });
    }
    for m in coefs {
        if m.nrows() != n || m.ncols() != n {
            return Err(IdentError::Dimension {
                what: "coefs matrices must all be n x n with n = gamma.len()",
                expected: n,
                got: if m.nrows() != n { m.nrows() } else { m.ncols() },
            });
        }
        for j in 0..n {
            for i in 0..n {
                if !m[(i, j)].is_finite() {
                    return Err(IdentError::NonFinite { what: "coefs" });
                }
            }
        }
    }
    let dim = n * n * p;
    if cov_alpha.nrows() != dim || cov_alpha.ncols() != dim {
        return Err(IdentError::Dimension {
            what: "cov_alpha must be (n*n*p) x (n*n*p): the lag-block coefficient covariance",
            expected: dim,
            got: cov_alpha.nrows(),
        });
    }
    for j in 0..dim {
        for i in 0..dim {
            if !cov_alpha[(i, j)].is_finite() {
                return Err(IdentError::NonFinite { what: "cov_alpha" });
            }
        }
    }
    for &g in gamma {
        if !g.is_finite() {
            return Err(IdentError::NonFinite { what: "gamma" });
        }
    }

    // alpha layout r = (l*n + j)*n + e  <->  A_l[(e, j)], matching cov_alpha.
    let mut alpha_hat = vec![0.0f64; dim];
    for (l, a_l) in coefs.iter().enumerate() {
        for j in 0..n {
            for e in 0..n {
                alpha_hat[(l * n + j) * n + e] = a_l[(e, j)];
            }
        }
    }
    let chol = jittered_cholesky(cov_alpha)?.factor;

    let mut stream = Stream::new(seed);
    let mut z = vec![0.0f64; dim];
    let mut dev = vec![0.0f64; dim];
    let mut alpha = vec![0.0f64; dim];
    let mut coefs_draw: Vec<Mat<f64>> = (0..p).map(|_| Mat::<f64>::zeros(n, n)).collect();
    // w[(d, h*n + i)] = (Psi_h(alpha_d) gamma)_i for every draw d.
    let mut w = Mat::<f64>::zeros(draws, (horizon + 1) * n);
    let mut psi_draw: Vec<Mat<f64>> = Vec::with_capacity(horizon + 1);

    for pair in 0..draws / 2 {
        for zi in z.iter_mut() {
            *zi = std_normal(&mut stream)?;
        }
        // dev = L z (L lower-triangular).
        for r in 0..dim {
            let mut s = 0.0;
            for c in 0..=r {
                s += chol[(r, c)] * z[c];
            }
            dev[r] = s;
        }
        for (side, sgn) in [(0usize, 1.0f64), (1, -1.0)] {
            let d = 2 * pair + side;
            for r in 0..dim {
                alpha[r] = alpha_hat[r] + sgn * dev[r];
            }
            for (l, a_l) in coefs_draw.iter_mut().enumerate() {
                for j in 0..n {
                    for e in 0..n {
                        a_l[(e, j)] = alpha[(l * n + j) * n + e];
                    }
                }
            }
            // Psi recursion at the drawn coefficients.
            psi_draw.clear();
            psi_draw.push(Mat::from_fn(n, n, |i, j| f64::from(u8::from(i == j))));
            for h in 1..=horizon {
                let mut acc = Mat::<f64>::zeros(n, n);
                for i in 1..=h.min(p) {
                    let prev = &psi_draw[h - i];
                    let a_i = &coefs_draw[i - 1];
                    for rr in 0..n {
                        for cc in 0..n {
                            let mut s = 0.0;
                            for kk in 0..n {
                                s += prev[(rr, kk)] * a_i[(kk, cc)];
                            }
                            acc[(rr, cc)] += s;
                        }
                    }
                }
                psi_draw.push(acc);
            }
            for (h, ph) in psi_draw.iter().enumerate() {
                for i in 0..n {
                    let mut s = 0.0;
                    for (j, &gj) in gamma.iter().enumerate() {
                        s += ph[(i, j)] * gj;
                    }
                    w[(d, h * n + i)] = s;
                }
            }
        }
    }

    // Centered sample covariance per horizon, scaled by T_O.
    let scale = n_proxy as f64 / (draws - 1) as f64;
    let mut out: Vec<Mat<f64>> = Vec::with_capacity(horizon + 1);
    out.push(Mat::<f64>::zeros(n, n));
    for h in 1..=horizon {
        let mut mean = vec![0.0f64; n];
        for (i, slot) in mean.iter_mut().enumerate() {
            let mut s = 0.0;
            for d in 0..draws {
                s += w[(d, h * n + i)];
            }
            *slot = s / draws as f64;
        }
        let mut cov = Mat::<f64>::zeros(n, n);
        for d in 0..draws {
            for i in 0..n {
                let wi = w[(d, h * n + i)] - mean[i];
                for j in 0..n {
                    cov[(i, j)] += wi * (w[(d, h * n + j)] - mean[j]);
                }
            }
        }
        for j in 0..n {
            for i in 0..n {
                cov[(i, j)] *= scale;
            }
        }
        out.push(cov);
    }
    Ok(out)
}

/// One standard-normal draw by inverse-CDF transform of a stream uniform,
/// rejecting the exact 0 that [`Stream::uniform_f64`] can (with probability
/// `2^-53`) return — the same construction `haar.rs` and `zero.rs` use.
fn std_normal(stream: &mut Stream) -> Result<f64, IdentError> {
    for _ in 0..128 {
        let u = stream.uniform_f64();
        if u > 0.0 {
            return Ok(inv_norm_cdf(u)?);
        }
    }
    Err(IdentError::NoConvergence {
        what: "positive uniform draw for a Gaussian coefficient perturbation \
               (stream returned 0 repeatedly)",
    })
}

/// The five scalars that define one cell's Anderson-Rubin problem, plus the
/// point estimate.
///
/// Splitting this out is what makes a *corrected* variance possible without
/// duplicating the branch logic. The linearized statistic is
/// `a(lam)'(gamma_hat - gamma) + unit * e_i'(Psi_hat_h - Psi_h) gamma`; the
/// second term does not involve `lam`, so including reduced-form estimation
/// error adds a constant to `v0` and a constant to `v1` and leaves the
/// quadratic structure — and therefore [`ar_cell`] — untouched.
///
/// [`proxy_ar_sets`] applies exactly those two constants when it is handed an
/// [`ArReducedForm`], so most callers want that route. Reach for [`ar_cell`]
/// directly only for a variance this crate has no input for — an
/// over-identified extension, or a joint bootstrap that estimates the whole
/// `V_hat(lam)` polynomial at once.
///
/// Any such correction **must vanish with `gamma`**, as this one does. A
/// `Psi_h` variance term scaled by the normalized impact `b` instead of by
/// the raw `gamma` would keep the set bounded under weak instruments and
/// destroy exactly the robustness this construction is for.
#[derive(Debug, Clone, Copy)]
pub struct ArMoments {
    /// Numerator moment `unit * (Psi_h gamma)_i`.
    pub q1: f64,
    /// Denominator moment `gamma_k`.
    pub q0: f64,
    /// `unit^2 * (Psi_h Omega Psi_h')[i,i]`, plus any reduced-form correction.
    pub v0: f64,
    /// `unit * (Psi_h Omega)[i,k]`, plus any reduced-form correction.
    pub v1: f64,
    /// `Omega[k,k]`.
    pub v2: f64,
    /// The point estimate, used as the representative of a degenerate
    /// (double-root) set. Normally `q1/q0`, or the `(Psi_h b)_i` form that
    /// matches [`crate::proxy_svar`] bit-for-bit.
    pub point: f64,
}

/// Build one cell's confidence set from its moments — the whole taxonomy,
/// tolerances and root-finding in one reusable place.
///
/// [`proxy_ar_sets`] is this function in a loop. Call it directly when you
/// have variance terms this crate cannot compute (see [`ArMoments`]), or to
/// re-invert at a different critical value without recomputing `Omega`.
///
/// # Errors
///
/// [`IdentError::InvalidArgument`] if `n_proxy` is zero, the critical value
/// is not positive and finite, `v2` is not positive, the moments fail the
/// positive-semidefiniteness condition `v1^2 <= v0*v2`, or the discriminant
/// is genuinely negative with a positive leading coefficient (which only an
/// indefinite variance can produce).
pub fn ar_cell(n_proxy: usize, critical_value: f64, m: ArMoments) -> Result<ArCell, IdentError> {
    if n_proxy == 0 {
        return Err(IdentError::InvalidArgument {
            what: "n_proxy must be positive",
        });
    }
    if !critical_value.is_finite() || critical_value <= 0.0 {
        return Err(IdentError::InvalidArgument {
            what: "the critical value must be positive and finite",
        });
    }
    let ArMoments {
        q1,
        q0,
        v0,
        v1,
        v2,
        point,
    } = m;
    if !v2.is_finite() || v2 <= 0.0 {
        return Err(IdentError::InvalidArgument {
            what: "v2 (the variance of the denominator moment) must be positive and finite",
        });
    }
    // V_hat(lam) = v0 - 2 lam v1 + lam^2 v2 is nonnegative for every lam iff
    // v0 >= 0, v2 >= 0 and v1^2 <= v0*v2 — the 2x2 PSD condition for Omega
    // projected onto span{Psi_h' e_i, e_k}. Everything downstream (the
    // taxonomy, the never-empty result, the nesting of levels) assumes it.
    let psd_scale = v0.abs().max(v1.abs()).max(v2.abs());
    if v0 < -PSD_REL * psd_scale || v1 * v1 > v0 * v2 + PSD_REL * psd_scale * psd_scale {
        return Err(IdentError::InvalidArgument {
            what: "the null-imposed variance V(lam) = v0 - 2*lam*v1 + lam^2*v2 goes negative \
                   for some lam (v1^2 > v0*v2), so the AR statistic is not a quadratic-set \
                   problem. ArVariance::Hc0 is PSD by construction on any overlap; \
                   ArVariance::HacBartlett is PSD only when the overlap is contiguous, so a \
                   proxy with interior gaps can reach this — shorten the HAC bandwidth, or \
                   fall back to Hc0. A supplied Omega or ArReducedForm must be PSD",
        });
    }

    let no = n_proxy as f64;
    let c = critical_value;
    let a = no * q0 * q0 - c * v2;
    let b = 2.0 * (c * v1 - no * q1 * q0);
    let cc = no * q1 * q1 - c * v0;

    let tau_a = TAU_A_REL * (no * q0 * q0).max(c * v2);
    let tau_b = TAU_B_REL * (2.0 * (c * v1).abs()).max(2.0 * (no * q1 * q0).abs());
    let tau_c = TAU_C_REL * (no * q1 * q1).max(c * v0);
    let set = solve_set(a, b, cc, tau_a, tau_b, tau_c, point)?;

    Ok(ArCell {
        set,
        point,
        a,
        b,
        c: cc,
        q1,
        q0,
        v0,
        v1,
        v2,
        // Read off the set, not off `cc`, so a single tolerance decides both.
        // `cc > 0.0` is the same answer everywhere except the knife edge,
        // where `solve_set` calls `cc <= tau_c` the whole line: reading `cc`
        // there would claim a set that contains every real number excludes 0.
        excludes_zero: !set.contains(0.0),
    })
}

/// `0 < level < 1`, the domain the critical-value quantiles need.
fn check_level(level: f64) -> Result<(), IdentError> {
    if !(level > 0.0 && level < 1.0) {
        return Err(IdentError::InvalidArgument {
            what: "confidence level must lie strictly inside (0, 1)",
        });
    }
    Ok(())
}

/// HC0/White moment covariance `(1/T_O) sum_t g~_t g~_t'`, PSD by
/// construction (a sum of outer products).
fn omega_hc0(gtil: MatRef<'_, f64>, no: f64) -> Mat<f64> {
    let n = gtil.ncols();
    let n_o = gtil.nrows();
    let mut om = Mat::<f64>::zeros(n, n);
    for p in 0..n_o {
        for i in 0..n {
            let gi = gtil[(p, i)];
            for j in 0..n {
                om[(i, j)] += gi * gtil[(p, j)];
            }
        }
    }
    for j in 0..n {
        for i in 0..n {
            om[(i, j)] /= no;
        }
    }
    om
}

/// Bartlett-kernel HAC moment covariance.
///
/// `Gamma_j = (1/T_O) * sum g~_t g~_{t-j}'` over **calendar-time** pairs: `t`
/// and `t - j` must both carry a finite proxy value. With a contiguous
/// overlap this is the textbook Newey-West estimator; with interior gaps it
/// is the honest analogue (the alternative — lagging over compacted positions
/// — would splice across the gap and manufacture dependence).
fn omega_hac_bartlett(
    gtil: MatRef<'_, f64>,
    overlap: &[usize],
    t: usize,
    lags: usize,
    no: f64,
) -> Mat<f64> {
    let n = gtil.ncols();
    let mut om = omega_hc0(gtil, no);
    if lags == 0 {
        return om;
    }
    // Calendar date -> row of `gtil`, or `usize::MAX` where the proxy is
    // unavailable.
    let mut pos = vec![usize::MAX; t];
    for (p, &r) in overlap.iter().enumerate() {
        pos[r] = p;
    }
    for j in 1..=lags {
        let w = 1.0 - (j as f64) / ((lags + 1) as f64);
        let mut gam = Mat::<f64>::zeros(n, n);
        for (p, &r) in overlap.iter().enumerate() {
            if r < j {
                continue;
            }
            let q = pos[r - j];
            if q == usize::MAX {
                continue;
            }
            for a in 0..n {
                let ga = gtil[(p, a)];
                for bb in 0..n {
                    gam[(a, bb)] += ga * gtil[(q, bb)];
                }
            }
        }
        for a in 0..n {
            for bb in 0..n {
                om[(a, bb)] += w * (gam[(a, bb)] + gam[(bb, a)]) / no;
            }
        }
    }
    om
}

/// HC1-robust first-stage `F` for the regression of the `norm_var` residual
/// on the proxy over the overlap — the identical construction
/// [`crate::proxy_svar`] reports, carried here so both numbers can be printed
/// side by side.
fn first_stage_hc1(
    u: MatRef<'_, f64>,
    proxy: &[f64],
    overlap: &[usize],
    norm_var: usize,
    mbar: f64,
    ybar: f64,
    no: f64,
) -> Result<f64, IdentError> {
    let mut smm = 0.0;
    let mut smy = 0.0;
    for &r in overlap {
        let md = proxy[r] - mbar;
        smm += md * md;
        smy += md * (u[(r, norm_var)] - ybar);
    }
    if smm == 0.0 {
        return Err(IdentError::InvalidArgument {
            what: "instrument has zero variance over the overlap; no first stage",
        });
    }
    let beta = smy / smm;
    let dof = no - 2.0;
    let mut meat = 0.0;
    for &r in overlap {
        let md = proxy[r] - mbar;
        let e = (u[(r, norm_var)] - ybar) - beta * md;
        meat += md * md * e * e;
    }
    let var_hc1 = (no / dof) * meat / (smm * smm);
    Ok(beta * beta / var_hc1)
}

/// Solve `A*lam^2 + B*lam + C <= 0` in closed form and classify the result.
///
/// `point` is the point estimate, used as the representative of a degenerate
/// (double-root) set: when the discriminant vanishes the double root
/// `-B/(2A)` and the point estimate `q1/q0` are the *same number*
/// mathematically — `D = 0` happens exactly when `V_hat(lam_hat) = 0`, which
/// forces `lam_hat` to be that root — and `q1/q0` is computed without the
/// cancellation `-B/(2A)` suffers. Using it makes the `(norm_var, h = 0)`
/// cell return exactly `unit`.
fn solve_set(
    a: f64,
    b: f64,
    c: f64,
    tau_a: f64,
    tau_b: f64,
    tau_c: f64,
    point: f64,
) -> Result<ArSet, IdentError> {
    if a.abs() <= tau_a {
        // Knife edge: the quadratic has degenerated to B*lam + C <= 0. This
        // is where the relevance statistic sits exactly at the critical
        // value, so the arithmetic itself is marginal; a locally degenerate
        // answer is the right response, not a hard error.
        if b.abs() <= tau_b {
            // C is also compared against a tolerance, and it must be: the
            // never-empty result forces C <= 0 whenever A = B = 0, so a
            // positive C here is float noise. Reading it literally makes the
            // (norm_var, h = 0) cell come back EMPTY at the knife edge, where
            // A, B and C are all cancellation residue of order 1e-16.
            return Ok(if c <= tau_c {
                ArSet::Whole
            } else {
                ArSet::Empty
            });
        }
        let root = -c / b;
        return Ok(if b > 0.0 {
            ArSet::RayBelow { hi: root }
        } else {
            ArSet::RayAbove { lo: root }
        });
    }

    let d = b * b - 4.0 * a * c;
    let tau_d = TAU_D_REL * (b * b).max((4.0 * a * c).abs());

    if a > 0.0 {
        if d > tau_d {
            let (lo, hi) = stable_roots(a, b, c, d);
            Ok(ArSet::Interval { lo, hi })
        } else if d >= -tau_d {
            Ok(ArSet::Point(point))
        } else {
            // Unreachable with a PSD Omega: the identity
            // D >= 4*A*c*V_hat(lam_hat) with A > 0 and V_hat >= 0 forces
            // D >= 0. A genuinely negative discriminant therefore means the
            // supplied moment covariance is indefinite, which is a global
            // property of the input worth failing loudly on rather than
            // reporting an empty set for one cell.
            Err(IdentError::InvalidArgument {
                what: "negative discriminant with a positive leading coefficient: the moment \
                       covariance Omega must be indefinite, since a PSD Omega makes the point \
                       estimate a member of its own confidence set; supply a PSD Omega",
            })
        }
    } else if d > tau_d {
        let (lo, hi) = stable_roots(a, b, c, d);
        Ok(ArSet::Exterior { lo, hi })
    } else {
        // A concave quadratic whose maximum -D/(4A) is <= 0 is nonpositive
        // everywhere.
        Ok(ArSet::Whole)
    }
}

/// The two real roots of `A*lam^2 + B*lam + C`, sorted ascending, computed by
/// the cancellation-free pair `s = -B - sign(B)*sqrt(D)`, `r = s/(2A)` and
/// `r' = 2C/s` (Vieta).
///
/// The textbook `(-B +/- sqrt(D))/(2A)` loses all significance in the branch
/// where `-B` and `sqrt(D)` nearly cancel — which is exactly the regime this
/// module lives in, since the endpoints diverge like `1/A` as the relevance
/// statistic approaches the critical value from above. Sorting (rather than
/// assigning `+sqrt(D)` to the upper endpoint) is what keeps `lo <= hi` when
/// `A < 0`.
///
/// `s` cannot be zero here: `|s| = |B| + sqrt(D)` and the caller only enters
/// with `D > 0`.
fn stable_roots(a: f64, b: f64, c: f64, d: f64) -> (f64, f64) {
    let sq = d.max(0.0).sqrt();
    let sgn = if b < 0.0 { -1.0 } else { 1.0 };
    let s = -b - sgn * sq;
    let r1 = s / (2.0 * a);
    let r2 = 2.0 * c / s;
    if r1 <= r2 {
        (r1, r2)
    } else {
        (r2, r1)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    /// The knife-edge branch: a linear inequality with a positive slope is a
    /// half-line below the root, and with a negative slope a half-line above.
    #[test]
    fn knife_edge_gives_half_lines() -> Result<(), IdentError> {
        // 2*lam - 4 <= 0  <=>  lam <= 2.
        let s = solve_set(0.0, 2.0, -4.0, 1e-12, 1e-12, 1e-12, 2.0)?;
        assert_eq!(s, ArSet::RayBelow { hi: 2.0 });
        // -2*lam - 4 <= 0  <=>  lam >= -2.
        let s = solve_set(0.0, -2.0, -4.0, 1e-12, 1e-12, 1e-12, -2.0)?;
        assert_eq!(s, ArSet::RayAbove { lo: -2.0 });
        // 0*lam + C with C <= 0 accepts everything; with C > 0 nothing.
        assert_eq!(
            solve_set(0.0, 0.0, -1.0, 1e-12, 1e-12, 1e-12, 0.0)?,
            ArSet::Whole
        );
        assert_eq!(
            solve_set(0.0, 0.0, 1.0, 1e-12, 1e-12, 1e-12, 0.0)?,
            ArSet::Empty
        );
        Ok(())
    }

    /// The convex and concave branches, and the ordering trap: with `A < 0`
    /// the `+sqrt(D)` root is the *smaller* one.
    #[test]
    fn convex_and_concave_branches() -> Result<(), IdentError> {
        // (lam - 1)(lam - 3) <= 0 on [1, 3].
        let s = solve_set(1.0, -4.0, 3.0, 1e-12, 1e-12, 1e-12, 2.0)?;
        match s {
            ArSet::Interval { lo, hi } => {
                assert!((lo - 1.0).abs() < 1e-13);
                assert!((hi - 3.0).abs() < 1e-13);
            }
            other => panic!("expected an interval, got {other:?}"),
        }
        // -(lam - 1)(lam - 3) <= 0 outside (1, 3).
        let s = solve_set(-1.0, 4.0, -3.0, 1e-12, 1e-12, 1e-12, 2.0)?;
        match s {
            ArSet::Exterior { lo, hi } => {
                assert!((lo - 1.0).abs() < 1e-13);
                assert!((hi - 3.0).abs() < 1e-13);
                assert!(lo <= hi, "roots must come back sorted");
            }
            other => panic!("expected an exterior set, got {other:?}"),
        }
        // A concave quadratic with no real roots is nonpositive everywhere.
        assert_eq!(
            solve_set(-1.0, 0.0, -1.0, 1e-12, 1e-12, 1e-12, 0.0)?,
            ArSet::Whole
        );
        // A convex quadratic with a double root is that single point, and the
        // representative is the point estimate.
        assert_eq!(
            solve_set(2.0, -4.0, 2.0, 1e-12, 1e-12, 1e-12, 1.0)?,
            ArSet::Point(1.0)
        );
        Ok(())
    }

    /// A convex quadratic with a genuinely negative discriminant can only
    /// come from an indefinite `Omega`, and is refused rather than reported
    /// as an empty set.
    #[test]
    fn negative_discriminant_is_refused() {
        assert!(matches!(
            solve_set(1.0, 0.0, 1.0, 1e-12, 1e-12, 1e-12, 0.0),
            Err(IdentError::InvalidArgument { .. })
        ));
    }

    /// `endpoints()` never hands back the rejected middle of an exterior set
    /// as if it were the set.
    #[test]
    fn exterior_endpoints_are_infinite() {
        let s = ArSet::Exterior { lo: -1.0, hi: 2.0 };
        assert_eq!(s.endpoints(), (f64::NEG_INFINITY, f64::INFINITY));
        assert_eq!(s.excluded_middle(), Some((-1.0, 2.0)));
        assert!(!s.is_bounded());
        assert!(s.width().is_infinite());
        assert!(s.contains(-5.0) && s.contains(10.0));
        assert!(!s.contains(0.0));
        assert_eq!(s.kind().as_str(), "exterior");
    }

    /// The quadratic root finder stays accurate where the textbook formula
    /// cancels: roots separated by 16 orders of magnitude.
    #[test]
    fn roots_survive_cancellation() {
        // (lam - 1e-8)(lam - 1e8) = lam^2 - (1e8 + 1e-8) lam + 1.
        let (lo, hi) = stable_roots(1.0, -(1e8 + 1e-8), 1.0, {
            let b: f64 = -(1e8 + 1e-8);
            b * b - 4.0
        });
        assert!((lo / 1e-8 - 1.0).abs() < 1e-9, "lo = {lo}");
        assert!((hi / 1e8 - 1.0).abs() < 1e-13, "hi = {hi}");
    }

    /// The F critical value approaches the chi-square one as the degrees of
    /// freedom grow, and exceeds it at every finite sample size.
    #[test]
    fn f_critical_value_nests_chi2() -> Result<(), IdentError> {
        let z = inv_norm_cdf(0.975)?;
        let chi2 = z * z;
        assert!(
            (chi2 - 3.841_458_820_694_124).abs() < 1e-12,
            "chi2 = {chi2}"
        );
        for &d in &[10.0f64, 100.0, 10_000.0] {
            let zb = inv_beta_inc(0.5, 0.5 * d, 0.95)?;
            let f = d * zb / (1.0 - zb);
            assert!(f > chi2, "F_{{1,{d}}} = {f} must exceed chi2 = {chi2}");
            assert!(f < chi2 * (1.0 + 20.0 / d), "F_{{1,{d}}} = {f} too large");
        }
        Ok(())
    }
}
