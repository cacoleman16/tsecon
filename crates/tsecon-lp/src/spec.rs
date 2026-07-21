//! Specification and result types shared by the local-projection estimators.

/// How the standard errors on the impulse-response coefficients are formed.
///
/// The default, [`SeSpec::LagAugmented`], is the recommendation of Montiel
/// Olea & Plagborg-Møller (2021): augment each horizon-`h` regression with
/// the impulse's own lags `1..=h` and then use ordinary
/// heteroskedasticity-robust (HC1) standard errors. See the crate docs for
/// why this dominates HAC for LP inference.
///
/// [`SeSpec::Hac`] reproduces the classic Jordà (2005) / statsmodels
/// practice: no impulse-lag augmentation, and a Newey-West (1987) HAC
/// covariance whose lag truncation grows with the horizon.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SeSpec {
    /// Lag-augmented LP with HC1 (Eicker-Huber-White) standard errors
    /// (Montiel Olea & Plagborg-Møller 2021). This is the library default.
    #[default]
    LagAugmented,
    /// Newey-West HAC standard errors (statsmodels `cov_type="HAC"`,
    /// Bartlett kernel, `use_correction=True`).
    Hac {
        /// Fixed lag truncation (statsmodels `maxlags`) for every horizon.
        /// `None` uses the horizon-growing default `maxlags = h +
        /// n_lag_controls`, the convention behind the golden fixture and the
        /// usual advice that the HAC window track the horizon-`h` MA order.
        maxlags: Option<usize>,
    },
}

/// Which standard-error construction actually produced a result's `se` path.
///
/// Returned in [`LpResult::se_kind`] so downstream code and tables can label
/// the inference without re-deriving it from the spec.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeKind {
    /// Lag-augmented LP with HC1 standard errors (the default; Montiel Olea
    /// & Plagborg-Møller 2021).
    LagAugmentedHc1,
    /// Newey-West Bartlett HAC standard errors with the recorded (possibly
    /// horizon-varying) lag truncation.
    HacBartlett,
    /// Kernel HAC standard errors matching the linearmodels IV2SLS
    /// convention (used by [`crate::lp_iv`]).
    IvKernelHac,
    /// Bartlett-HAC sandwich over the base-time-aggregated scores of the
    /// stacked smooth-LP estimator, pushed through the B-spline basis by the
    /// delta method (used by [`crate::smooth_lp`]; conditional on the
    /// smoothing parameter — see [`crate::SmoothLpResult::se`]).
    SmoothStackedHac,
}

/// Which side(s) of the projection are accumulated over the horizon.
///
/// This is the difference between an impulse response and a *multiplier*, and
/// getting it wrong is the classic LP-IV trap:
///
/// * [`Cumulation::None`] — level response `y_{t+h}` on `x_t`.
/// * [`Cumulation::Outcome`] — `sum_{j=0}^{h} y_{t+j}` on `x_t`. This is the
///   Ramey-Zubairy *cumulative impulse response*: cumulative output per unit
///   of **contemporaneous** impulse. It is a perfectly good IRF and it is
///   what `cumulative = true` has always meant, but it is **not** a
///   multiplier: because the denominator never grows, the ratio grows without
///   bound in `h`.
/// * [`Cumulation::Both`] — `sum_{j=0}^{h} y_{t+j}` on `sum_{j=0}^{h} x_{t+j}`.
///   Now numerator and denominator are the same accumulated object, and the
///   coefficient is the *integral multiplier*: extra cumulated `y` per extra
///   cumulated `x` through horizon `h`. In the just-identified IV case this
///   is Ramey & Zubairy's (2018) one-step integral multiplier — see
///   [`lp_multiplier`](crate::lp_multiplier), which is the recommended front
///   door.
///
/// Under [`Cumulation::Both`] an external instrument stays **contemporaneous**
/// (`z_t`): the accumulation belongs to the endogenous variables, not to the
/// identifying variation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Cumulation {
    /// Level response: `y_{t+h}` on `x_t`.
    #[default]
    None,
    /// Cumulative impulse response: `sum_j y_{t+j}` on `x_t`.
    Outcome,
    /// Integral multiplier: `sum_j y_{t+j}` on `sum_j x_{t+j}`.
    Both,
}

impl Cumulation {
    /// Whether the outcome column is accumulated.
    #[must_use]
    pub fn accumulates_outcome(self) -> bool {
        !matches!(self, Cumulation::None)
    }

    /// Whether the impulse column is accumulated.
    #[must_use]
    pub fn accumulates_impulse(self) -> bool {
        matches!(self, Cumulation::Both)
    }
}

/// Configuration for a local-projection run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LpSpec {
    /// Maximum horizon `H`; responses are estimated for every `h` in
    /// `0..=horizons` (so the result paths have length `horizons + 1`).
    pub horizons: usize,
    /// Number of lagged-outcome controls `y_{t-1}, ..., y_{t-p}` included in
    /// every horizon regression (Jordà 2005 uses these to soak up the
    /// pre-impulse dynamics).
    pub n_lag_controls: usize,
    /// Standard-error construction; see [`SeSpec`]. Defaults to
    /// lag-augmented HC1.
    pub se: SeSpec,
    /// Which side(s) of the projection are accumulated over the horizon; see
    /// [`Cumulation`]. Defaults to [`Cumulation::None`].
    ///
    /// [`Cumulation::Outcome`] regresses the cumulated outcome
    /// `sum_{j=0}^{h} y_{t+j}` on the contemporaneous impulse (Ramey-Zubairy
    /// cumulative IRF), so the reported standard errors are the cumulative
    /// ones by construction (no need to cumulate level SEs, which would be
    /// wrong). [`Cumulation::Both`] also accumulates the impulse, turning the
    /// coefficient into an integral multiplier.
    pub cumulation: Cumulation,
}

impl LpSpec {
    /// A spec with the library defaults: lag-augmented HC1 inference, level
    /// (non-cumulative) responses.
    ///
    /// `horizons` is the maximum horizon and `n_lag_controls` the number of
    /// lagged-outcome controls.
    #[must_use]
    pub fn new(horizons: usize, n_lag_controls: usize) -> Self {
        LpSpec {
            horizons,
            n_lag_controls,
            se: SeSpec::LagAugmented,
            cumulation: Cumulation::None,
        }
    }

    /// Builder: switch to Newey-West HAC inference with the given (optional)
    /// fixed lag truncation.
    #[must_use]
    pub fn with_hac(mut self, maxlags: Option<usize>) -> Self {
        self.se = SeSpec::Hac { maxlags };
        self
    }

    /// Builder: request cumulative-**outcome** (Ramey-Zubairy cumulative IRF)
    /// responses. `true` maps to [`Cumulation::Outcome`], `false` to
    /// [`Cumulation::None`]; this is the historical spelling and its meaning
    /// is unchanged.
    ///
    /// For an integral *multiplier* you want both sides accumulated: use
    /// [`LpSpec::with_cumulation`] with [`Cumulation::Both`], or better
    /// [`lp_multiplier`](crate::lp_multiplier).
    #[must_use]
    pub fn cumulative(mut self, cumulative: bool) -> Self {
        self.cumulation = if cumulative {
            Cumulation::Outcome
        } else {
            Cumulation::None
        };
        self
    }

    /// Builder: set the accumulation mode explicitly.
    #[must_use]
    pub fn with_cumulation(mut self, cumulation: Cumulation) -> Self {
        self.cumulation = cumulation;
        self
    }
}

/// The estimated impulse-response function of a single-impulse local
/// projection, one entry per horizon.
#[derive(Debug, Clone, PartialEq)]
pub struct LpResult {
    /// Horizons estimated, `[0, 1, ..., H]`.
    pub horizons: Vec<usize>,
    /// Point impulse response at each horizon (the impulse coefficient, or
    /// the cumulative response when `spec.cumulative`).
    pub irf: Vec<f64>,
    /// Standard error of the response at each horizon.
    pub se: Vec<f64>,
    /// Effective number of observations in each horizon regression.
    pub nobs_per_h: Vec<usize>,
    /// Which standard-error construction produced [`LpResult::se`].
    pub se_kind: SeKind,
}

/// The result of a just-identified LP-IV run.
#[derive(Debug, Clone, PartialEq)]
pub struct LpIvResult {
    /// Horizons estimated, `[0, 1, ..., H]`.
    pub horizons: Vec<usize>,
    /// 2SLS impulse response (coefficient on the instrumented impulse) at
    /// each horizon.
    pub irf: Vec<f64>,
    /// Kernel-HAC standard error of the response at each horizon
    /// (linearmodels IV2SLS convention).
    pub se: Vec<f64>,
    /// First-stage effective-F diagnostic at each horizon: the HAC-robust
    /// first-stage F statistic for the excluded instrument (for a single
    /// instrument this is the Montiel Olea & Pflueger 2013 effective F).
    /// Weak-instrument concern below the usual rule-of-thumb of 10.
    pub first_stage_f: Vec<f64>,
    /// Effective number of observations in each horizon regression.
    pub nobs_per_h: Vec<usize>,
    /// Always [`SeKind::IvKernelHac`]; recorded for symmetry with
    /// [`LpResult`].
    pub se_kind: SeKind,
}

/// The result of a one-step integral-multiplier run
/// ([`lp_multiplier`](crate::lp_multiplier)).
#[derive(Debug, Clone, PartialEq)]
pub struct LpMultiplierResult {
    /// Horizons estimated, `[0, 1, ..., H]`.
    pub horizons: Vec<usize>,
    /// The integral multiplier at each horizon: extra cumulated outcome per
    /// extra cumulated impulse through horizon `h`.
    pub multiplier: Vec<f64>,
    /// Kernel-HAC standard error **of the multiplier itself**. Because the
    /// multiplier is estimated as a single 2SLS coefficient rather than as a
    /// ratio of two separately-estimated responses, this is the standard
    /// error of the reported parameter — not a leg's SE and not a
    /// delta-method approximation.
    pub se: Vec<f64>,
    /// First-stage effective-F diagnostic for the excluded instrument in the
    /// *cumulated* first stage (`sum_j x_{t+j}` on `z_t` and controls).
    /// Weak-instrument concern below the usual rule-of-thumb of 10.
    pub first_stage_f: Vec<f64>,
    /// Cumulative response of the outcome at each horizon (the numerator's
    /// reduced form), reported for transparency.
    pub cumulative_outcome: Vec<f64>,
    /// Cumulative response of the impulse at each horizon (the denominator's
    /// reduced form), reported for transparency.
    pub cumulative_impulse: Vec<f64>,
    /// Effective number of observations in each horizon regression.
    pub nobs_per_h: Vec<usize>,
    /// Always [`SeKind::IvKernelHac`].
    pub se_kind: SeKind,
}

/// The result of a state-dependent (interacted) local projection.
///
/// Per Ramey & Zubairy (2018), the impulse and controls are interacted with
/// the *lagged* state indicator so the regime is predetermined. Two response
/// paths are reported: `irf_state1` when the lagged indicator is 1, and
/// `irf_state0` when it is 0.
#[derive(Debug, Clone, PartialEq)]
pub struct LpStateResult {
    /// Horizons estimated, `[0, 1, ..., H]`.
    pub horizons: Vec<usize>,
    /// Impulse response in state 1 (lagged indicator `= 1`) at each horizon.
    pub irf_state1: Vec<f64>,
    /// Standard error of the state-1 response at each horizon.
    pub se_state1: Vec<f64>,
    /// Impulse response in state 0 (lagged indicator `= 0`) at each horizon.
    pub irf_state0: Vec<f64>,
    /// Standard error of the state-0 response at each horizon.
    pub se_state0: Vec<f64>,
    /// Effective number of observations in each horizon regression.
    pub nobs_per_h: Vec<usize>,
    /// Which standard-error construction produced the state SE paths.
    pub se_kind: SeKind,
}
