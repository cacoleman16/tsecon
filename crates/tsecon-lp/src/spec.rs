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
    /// When `true`, regress the *cumulated* outcome `sum_{j=0}^{h} y_{t+j}`
    /// on the impulse (Ramey-Zubairy convention), so the reported response
    /// is a cumulative impulse response and its standard errors are correct
    /// by construction (no need to cumulate level SEs, which would be
    /// wrong).
    pub cumulative: bool,
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
            cumulative: false,
        }
    }

    /// Builder: switch to Newey-West HAC inference with the given (optional)
    /// fixed lag truncation.
    #[must_use]
    pub fn with_hac(mut self, maxlags: Option<usize>) -> Self {
        self.se = SeSpec::Hac { maxlags };
        self
    }

    /// Builder: request cumulative (Ramey-Zubairy) responses.
    #[must_use]
    pub fn cumulative(mut self, cumulative: bool) -> Self {
        self.cumulative = cumulative;
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
