//! Harvey's structural time-series models ("unobserved components") by
//! exact-diffuse maximum likelihood: [`unobserved_components`].
//!
//! # Model
//!
//! ```text
//! y_t = mu_t + gamma_t + c_t + beta' x_t + eps_t,   eps_t ~ N(0, sigma2_irregular)
//! ```
//!
//! with the components assembled exactly as statsmodels'
//! `UnobservedComponents` (and Harvey 1989, ch. 2) enumerate them:
//!
//! * **level / trend** (`mu_t`), chosen by [`TrendSpec`]:
//!   `mu_{t+1} = mu_t + nu_t + xi_t`, `nu_{t+1} = nu_t + zeta_t`, with the
//!   irregular, level (`xi`) and slope (`zeta`) disturbances switched on or
//!   off per specification (local level, local linear trend, smooth trend,
//!   random walk with drift, deterministic trend, ...);
//! * **dummy seasonal** (`gamma_t`, period `s`): `sum_{j=0}^{s-1}
//!   gamma_{t-j} = omega_t`, `omega_t ~ N(0, sigma2_seasonal)` (zero
//!   variance = fixed seasonal dummies), `s - 1` states;
//! * **trigonometric (frequency-domain) seasonal**: for each period `p`
//!   and harmonics `h`, the `h` rotation pairs `(gamma_j, gamma_j*)` at
//!   frequencies `2 pi j / p`, each driven by two independent
//!   disturbances of common variance (Harvey 1989, §2.3.4), `2h` states;
//! * **stochastic cycle** (`c_t`): `[c_{t+1}, c*_{t+1}]' = rho R(lambda)
//!   [c_t, c*_t]' + kappa_t` with `R` the rotation by the cycle frequency
//!   `lambda` and `rho` the damping (1 when undamped), the two disturbances
//!   sharing one variance (zero variance = deterministic cycle). The
//!   frequency is confined to `(2 pi / max, 2 pi / min)` by
//!   `cycle_period_bounds`, whose default upper period is the **sample
//!   length**: see [`UcSpec::cycle_period_bounds`] for why an unbounded
//!   period is not a safe default under exact-diffuse initialization;
//! * **regressors** with time-invariant coefficients `beta`, estimated
//!   jointly by MLE (statsmodels `mle_regression=True`).
//!
//! Every state is initialized exactly diffuse (Koopman 1997), which is
//! what statsmodels does under `use_exact_diffuse=True` — including the
//! cycle states, so the log-likelihoods are directly comparable. The
//! observation `y` may contain NaN for a missing period; regressors may
//! not.
//!
//! # Estimation
//!
//! The parameters (in statsmodels' order: `sigma2.irregular`, the state
//! variances in component order, `frequency.cycle`, `damping.cycle`, the
//! `beta`s) are estimated by BFGS + Nelder-Mead on the exact-diffuse
//! prediction-error-decomposition log-likelihood, in the square-root /
//! logistic working space of [`crate::mle`], from a small deterministic
//! ladder of starts. `y` is standardized by its standard deviation and
//! each regressor by its largest absolute value for the search and the
//! estimates mapped back exactly (variances by `s^2`, coefficients by
//! `s / c_j`); the reported log-likelihood, components and forecasts are
//! then re-evaluated on the original data at the mapped-back estimates,
//! so they are exactly the fixed-parameter quantities at the MLE.
//!
//! A variance whose estimate cannot be told from zero (pile-up) is
//! flagged in `at_boundary` and gets a NaN standard error; see
//! [`crate::mle`] for the criterion. Standard errors are observed-
//! information (numerical Hessian in the constrained space).
//!
//! # Reporting conventions
//!
//! `aic = 2 (k_params + k_diffuse) - 2 loglik` and `bic = -2 loglik +
//! (k_params + k_diffuse) ln(nobs)` with `k_diffuse` the number of
//! diffuse states — statsmodels' `df_model` convention, so the numbers
//! agree with its `UnobservedComponentsResults`. Filtered covariances
//! inside the diffuse period are the finite part `P_*`; smoothed
//! covariances are exact through it. The standardized residual is
//! `v_t / sqrt(F_t)` after the diffuse period and NaN inside it (where no
//! finite prediction variance exists) and at missing periods.

use tsecon_linalg::faer::Mat;

use crate::dense::{dot, mat_vec, sandwich};
use crate::error::SsmError;
use crate::filter::TOLERANCE_RANK;
use crate::mle::{maximize, ols, variance, ParamKind, ParamSpec};
use crate::model::{Initialization, LinearGaussianSSM};
use crate::smoother::smooth_univariate;

/// The level / trend specification, in statsmodels' vocabulary (either
/// the long or the short name parses).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrendSpec {
    /// `"irregular"` / `"ntrend"`: white noise only.
    Irregular,
    /// `"fixed intercept"`: a constant level, no irregular.
    FixedIntercept,
    /// `"deterministic constant"` / `"dconstant"`: constant level plus
    /// irregular.
    DeterministicConstant,
    /// `"local level"` / `"llevel"`: random-walk level plus irregular.
    LocalLevel,
    /// `"random walk"` / `"rwalk"`: random-walk level, no irregular.
    RandomWalk,
    /// `"fixed slope"`: deterministic linear trend, no irregular.
    FixedSlope,
    /// `"deterministic trend"` / `"dtrend"`: deterministic linear trend
    /// plus irregular.
    DeterministicTrend,
    /// `"local linear deterministic trend"` / `"lldtrend"`: random-walk
    /// level with a fixed slope, plus irregular.
    LocalLinearDeterministicTrend,
    /// `"random walk with drift"` / `"rwdrift"`: random-walk level with a
    /// fixed slope, no irregular.
    RandomWalkWithDrift,
    /// `"local linear trend"` / `"lltrend"`: random-walk level and
    /// random-walk slope, plus irregular.
    LocalLinearTrend,
    /// `"smooth trend"` / `"strend"`: integrated random-walk slope
    /// (level disturbance off), plus irregular.
    SmoothTrend,
    /// `"random trend"` / `"rtrend"`: smooth trend without irregular.
    RandomTrend,
}

impl TrendSpec {
    /// Parses a statsmodels level/trend name (long or short form).
    ///
    /// # Errors
    ///
    /// [`SsmError::InvalidSpec`] naming the accepted spellings.
    pub fn parse(name: &str) -> Result<Self, SsmError> {
        let key = name.trim().to_ascii_lowercase();
        Ok(match key.as_str() {
            "irregular" | "ntrend" => Self::Irregular,
            "fixed intercept" => Self::FixedIntercept,
            "deterministic constant" | "dconstant" => Self::DeterministicConstant,
            "local level" | "llevel" => Self::LocalLevel,
            "random walk" | "rwalk" => Self::RandomWalk,
            "fixed slope" => Self::FixedSlope,
            "deterministic trend" | "dtrend" => Self::DeterministicTrend,
            "local linear deterministic trend" | "lldtrend" => Self::LocalLinearDeterministicTrend,
            "random walk with drift" | "rwdrift" => Self::RandomWalkWithDrift,
            "local linear trend" | "lltrend" => Self::LocalLinearTrend,
            "smooth trend" | "strend" => Self::SmoothTrend,
            "random trend" | "rtrend" => Self::RandomTrend,
            _ => {
                return Err(SsmError::InvalidSpec {
                    message: format!(
                        "level = {name:?} is not a level/trend specification; pass one of \
                         \"irregular\"/\"ntrend\", \"fixed intercept\", \"deterministic \
                         constant\"/\"dconstant\", \"local level\"/\"llevel\", \"random \
                         walk\"/\"rwalk\", \"fixed slope\", \"deterministic trend\"/\"dtrend\", \
                         \"local linear deterministic trend\"/\"lldtrend\", \"random walk \
                         with drift\"/\"rwdrift\", \"local linear trend\"/\"lltrend\", \
                         \"smooth trend\"/\"strend\", \"random trend\"/\"rtrend\""
                    ),
                })
            }
        })
    }

    /// The long statsmodels name.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Irregular => "irregular",
            Self::FixedIntercept => "fixed intercept",
            Self::DeterministicConstant => "deterministic constant",
            Self::LocalLevel => "local level",
            Self::RandomWalk => "random walk",
            Self::FixedSlope => "fixed slope",
            Self::DeterministicTrend => "deterministic trend",
            Self::LocalLinearDeterministicTrend => "local linear deterministic trend",
            Self::RandomWalkWithDrift => "random walk with drift",
            Self::LocalLinearTrend => "local linear trend",
            Self::SmoothTrend => "smooth trend",
            Self::RandomTrend => "random trend",
        }
    }

    /// Component switches `(irregular, level, stochastic_level, trend,
    /// stochastic_trend)`.
    pub fn flags(&self) -> (bool, bool, bool, bool, bool) {
        match self {
            Self::Irregular => (true, false, false, false, false),
            Self::FixedIntercept => (false, true, false, false, false),
            Self::DeterministicConstant => (true, true, false, false, false),
            Self::LocalLevel => (true, true, true, false, false),
            Self::RandomWalk => (false, true, true, false, false),
            Self::FixedSlope => (false, true, false, true, false),
            Self::DeterministicTrend => (true, true, false, true, false),
            Self::LocalLinearDeterministicTrend => (true, true, true, true, false),
            Self::RandomWalkWithDrift => (false, true, true, true, false),
            Self::LocalLinearTrend => (true, true, true, true, true),
            Self::SmoothTrend => (true, true, false, true, true),
            Self::RandomTrend => (false, true, false, true, true),
        }
    }
}

/// One trigonometric seasonal block.
#[derive(Debug, Clone, PartialEq)]
pub struct FreqSeasonalSpec {
    /// Seasonal period (at least 2; need not be an integer).
    pub period: f64,
    /// Number of harmonics (`1..=floor(period / 2)`).
    pub harmonics: usize,
    /// Whether the block has a (common) disturbance variance.
    pub stochastic: bool,
}

impl FreqSeasonalSpec {
    /// The statsmodels default: `floor(period / 2)` harmonics, stochastic.
    pub fn new(period: f64) -> Self {
        Self {
            period,
            harmonics: (period / 2.0).floor().max(1.0) as usize,
            stochastic: true,
        }
    }
}

/// The model specification.
#[derive(Debug, Clone, PartialEq)]
pub struct UcSpec {
    /// Level / trend component.
    pub trend: TrendSpec,
    /// Dummy-seasonal period (`None` = no dummy seasonal).
    pub seasonal: Option<usize>,
    /// Whether the dummy seasonal has a disturbance variance.
    pub stochastic_seasonal: bool,
    /// Trigonometric seasonal blocks (possibly empty).
    pub freq_seasonal: Vec<FreqSeasonalSpec>,
    /// Include a stochastic cycle.
    pub cycle: bool,
    /// Estimate a damping factor `rho in (0, 1)` for the cycle.
    pub damped_cycle: bool,
    /// Whether the cycle has a disturbance variance.
    pub stochastic_cycle: bool,
    /// `(min, max)` cycle period bounds; the frequency is confined to
    /// `(2 pi / max, 2 pi / min)`.
    ///
    /// An infinite `max` means **the sample length**, not an unbounded
    /// period. Under exact-diffuse initialization the log-likelihood of a
    /// stochastic cycle diverges as `lambda -> 0+`: the cycle's second
    /// state becomes weakly observable, its diffuse direction resolves
    /// with `F_inf ~ (rho sin lambda)^2`, and the diffuse contribution
    /// `-(ln 2 pi + ln F_inf) / 2` grows without bound. An optimizer given
    /// `(0, pi)` walks into that singularity; a cycle longer than the
    /// sample is not identified anyway.
    pub cycle_period_bounds: (f64, f64),
    /// Regressor columns, each of length `n`.
    pub exog: Vec<Vec<f64>>,
}

impl Default for UcSpec {
    fn default() -> Self {
        Self {
            trend: TrendSpec::LocalLevel,
            seasonal: None,
            stochastic_seasonal: true,
            freq_seasonal: Vec::new(),
            cycle: false,
            damped_cycle: false,
            stochastic_cycle: false,
            cycle_period_bounds: (2.0, f64::INFINITY),
            exog: Vec::new(),
        }
    }
}

/// Estimation and reporting options.
#[derive(Debug, Clone, PartialEq)]
pub struct UcOptions {
    /// Number of out-of-sample forecasts.
    pub forecast_steps: usize,
    /// Regressor columns for the forecast periods (one column per
    /// regressor, each of length `forecast_steps`); required when the
    /// model has regressors and `forecast_steps > 0`.
    pub forecast_exog: Vec<Vec<f64>>,
    /// Evaluate at these parameters (statsmodels order) instead of
    /// estimating.
    pub fixed_params: Option<Vec<f64>>,
    /// Number of deterministic starts for the search (at least 1).
    pub n_starts: usize,
}

impl Default for UcOptions {
    fn default() -> Self {
        Self {
            forecast_steps: 0,
            forecast_exog: Vec::new(),
            fixed_params: None,
            n_starts: 3,
        }
    }
}

/// Filtered and smoothed paths of one component with their variances.
#[derive(Debug, Clone, PartialEq)]
pub struct UcComponent {
    /// `E[component_t | y_1..y_t]`.
    pub filtered: Vec<f64>,
    /// Its variance (finite part inside the diffuse period).
    pub filtered_var: Vec<f64>,
    /// `E[component_t | y_1..y_n]`.
    pub smoothed: Vec<f64>,
    /// Its variance (exact through the diffuse period).
    pub smoothed_var: Vec<f64>,
}

/// The fitted structural model.
#[derive(Debug, Clone, PartialEq)]
pub struct UcFit {
    /// Long name of the level/trend specification.
    pub trend_specification: String,
    /// Parameter names in statsmodels' order.
    pub param_names: Vec<String>,
    /// Estimates (or the fixed parameters).
    pub params: Vec<f64>,
    /// Observed-information standard errors (NaN at a boundary, when the
    /// Hessian is singular, or under `fixed_params`).
    pub se: Vec<f64>,
    /// Boundary (pile-up) flag per parameter.
    pub at_boundary: Vec<bool>,
    /// Exact-diffuse log-likelihood at `params`.
    pub loglik: f64,
    /// `2 (k_params + k_diffuse) - 2 loglik`.
    pub aic: f64,
    /// `-2 loglik + (k_params + k_diffuse) ln(nobs)`.
    pub bic: f64,
    /// Number of time periods (missing included).
    pub nobs: usize,
    /// Number of observed (non-NaN) periods.
    pub nobs_observed: usize,
    /// Length of the diffuse period.
    pub nobs_diffuse: usize,
    /// State dimension.
    pub k_states: usize,
    /// Number of diffuse states.
    pub k_diffuse: usize,
    /// Number of parameters.
    pub k_params: usize,
    /// False under `fixed_params`.
    pub estimated: bool,
    /// The optimizer's convergence certificate (true under `fixed_params`).
    pub converged: bool,
    /// Optimizer iterations (0 under `fixed_params`).
    pub n_iter: usize,
    /// Log-likelihood evaluations during the search (0 under
    /// `fixed_params`).
    pub n_fevals: usize,
    /// State names.
    pub state_names: Vec<String>,
    /// Filtered state means `a_{t|t}`, `nobs x k_states`.
    pub filtered_state: Vec<Vec<f64>>,
    /// Filtered state variances (diagonal of `P_{t|t}`).
    pub filtered_state_var: Vec<Vec<f64>>,
    /// Smoothed state means, `nobs x k_states`.
    pub smoothed_state: Vec<Vec<f64>>,
    /// Smoothed state variances (diagonal).
    pub smoothed_state_var: Vec<Vec<f64>>,
    /// Level component (None without a level).
    pub level: Option<UcComponent>,
    /// Slope component (None without a trend).
    pub slope: Option<UcComponent>,
    /// Dummy seasonal component.
    pub seasonal: Option<UcComponent>,
    /// Trigonometric seasonal components, one per block (the sum of the
    /// block's harmonics; variance = sum of their variances, as statsmodels
    /// reports it).
    pub freq_seasonal: Vec<UcComponent>,
    /// Cycle component.
    pub cycle: Option<UcComponent>,
    /// One-step-ahead predictions `E[y_t | y_1..y_{t-1}]`.
    pub fitted: Vec<f64>,
    /// One-step prediction errors `v_t` (NaN at missing periods).
    pub resid: Vec<f64>,
    /// `v_t / sqrt(F_t)` after the diffuse period; NaN inside it and at
    /// missing periods.
    pub std_resid: Vec<f64>,
    /// Point forecasts for `1..=forecast_steps`.
    pub forecast: Vec<f64>,
    /// Forecast variances (state uncertainty plus the irregular).
    pub forecast_var: Vec<f64>,
}

/// A state-variance parameter and the state rows it drives.
#[derive(Debug, Clone)]
struct StateVar {
    param: usize,
    col: usize,
    rows: Vec<usize>,
}

/// Resolved state layout and parameter bookkeeping.
#[derive(Debug, Clone)]
struct Layout {
    trend_spec: TrendSpec,
    level: bool,
    seasonal: Option<usize>,
    freq: Vec<FreqSeasonalSpec>,
    freq_bounds: (f64, f64),
    exog: Vec<Vec<f64>>,
    unused_state: bool,
    k_states: usize,
    k_posdef: usize,
    level_idx: Option<usize>,
    trend_idx: Option<usize>,
    seasonal_off: Option<usize>,
    freq_offs: Vec<usize>,
    cycle_off: Option<usize>,
    specs: Vec<ParamSpec>,
    idx_irregular: Option<usize>,
    state_vars: Vec<StateVar>,
    idx_cycle_freq: Option<usize>,
    idx_damping: Option<usize>,
    idx_beta: usize,
    state_names: Vec<String>,
}

/// Largest accepted `forecast_steps`.
///
/// The forecast loop is `O(forecast_steps * k_states^2)` and allocates two
/// vectors of that length, so an unchecked horizon coming in from Python
/// as a plain integer is an allocator abort waiting to happen (the wrapper
/// only refuses counts at or above `2^48`). A hundred thousand periods is
/// several centuries of monthly data and far past the point where the
/// forecast variance carries information; anything larger is a mistake,
/// and the refusal says so.
pub const MAX_FORECAST_STEPS: usize = 100_000;

/// Largest accepted `n_starts`.
///
/// The deterministic start ladder has five distinct rungs and repeats, so
/// more than a handful of starts buys nothing; the cap is what keeps a
/// mistyped count from allocating a start vector per unit of `usize`.
pub const MAX_STARTS: usize = 64;

fn invalid(message: String) -> SsmError {
    SsmError::InvalidSpec { message }
}

/// Frequency of the largest periodogram ordinate of `x` over the Fourier
/// frequencies `2 pi j / n` strictly inside `(lo, hi)`; `None` when no
/// Fourier frequency lies inside or `x` is too short.
fn periodogram_peak(x: &[f64], lo: f64, hi: f64) -> Option<f64> {
    let n = x.len();
    if n < 4 {
        return None;
    }
    let mean = x.iter().sum::<f64>() / n as f64;
    let mut best: Option<(f64, f64)> = None;
    for j in 1..=(n / 2) {
        let w = 2.0 * std::f64::consts::PI * j as f64 / n as f64;
        if w <= lo || w >= hi {
            continue;
        }
        let (mut c, mut s) = (0.0, 0.0);
        for (t, v) in x.iter().enumerate() {
            let (sn, cs) = (w * t as f64).sin_cos();
            c += (v - mean) * cs;
            s += (v - mean) * sn;
        }
        let power = c * c + s * s;
        if best.is_none_or(|(_, b)| power > b) {
            best = Some((w, power));
        }
    }
    best.map(|(w, _)| w)
}

impl Layout {
    fn new(spec: &UcSpec, n: usize) -> Result<Self, SsmError> {
        let (irregular, level, stochastic_level, trend, stochastic_trend) = spec.trend.flags();
        if let Some(s) = spec.seasonal {
            if s < 2 {
                return Err(invalid(format!(
                    "seasonal = {s}: the dummy-seasonal period must be at least 2 (pass \
                     None for no seasonal component)"
                )));
            }
            // Also the memory bound: the dummy seasonal costs `s - 1`
            // states, so an unchecked period is an allocation the size of
            // a user-supplied integer.
            if s > n {
                return Err(invalid(format!(
                    "seasonal = {s} with {n} observations: the dummy-seasonal period must \
                     not exceed the sample length (it costs s - 1 states, and a period \
                     the sample never completes is not identified)"
                )));
            }
        }
        for (i, f) in spec.freq_seasonal.iter().enumerate() {
            if !(f.period.is_finite() && f.period >= 2.0) {
                return Err(invalid(format!(
                    "freq_seasonal[{i}].period = {}: a trigonometric seasonal period must be \
                     a finite number >= 2",
                    f.period
                )));
            }
            if f.period > n as f64 {
                return Err(invalid(format!(
                    "freq_seasonal[{i}].period = {} with {n} observations: a trigonometric \
                     seasonal period must not exceed the sample length (it costs up to \
                     2 floor(period / 2) states, and a period the sample never completes \
                     is not identified)",
                    f.period
                )));
            }
            let max_h = (f.period / 2.0).floor() as usize;
            if f.harmonics == 0 || f.harmonics > max_h {
                return Err(invalid(format!(
                    "freq_seasonal[{i}].harmonics = {} (the Python keyword is \
                     freq_seasonal_harmonics[{i}]): for period {} the harmonics \
                     must be between 1 and floor(period / 2) = {max_h}",
                    f.harmonics, f.period
                )));
            }
        }
        let (pmin, pmax) = spec.cycle_period_bounds;
        if spec.cycle && !(pmin.is_finite() && pmin >= 2.0 && pmax > pmin) {
            return Err(invalid(format!(
                "cycle_period_bounds = ({pmin}, {pmax}): need 2 <= min < max (max may be \
                 infinite, meaning the sample length); the cycle frequency is confined \
                 to (2 pi / max, 2 pi / min)"
            )));
        }
        // An infinite upper period bound means the sample length, NOT an
        // unbounded period. The reason is not taste: with exact-diffuse
        // initialization the log-likelihood of a stochastic cycle is
        // unbounded above as the frequency goes to zero. At `lambda = 0`
        // the cycle's second state is unobservable and simply stays
        // diffuse (it costs nothing, exactly like the Nyquist harmonic);
        // at a small `lambda > 0` it is *weakly* observable, so its
        // diffuse direction does resolve, and it resolves with
        // `F_inf ~ (rho sin lambda)^2 -> 0`, contributing
        // `-(ln 2 pi + ln F_inf) / 2 -> +infinity`. An optimizer let loose
        // on `(0, pi)` walks into that singularity and reports a "cycle"
        // of period 10^6 with a log-likelihood a dozen points above any
        // genuine optimum. A cycle longer than the sample is not
        // identified by the sample in any case. statsmodels leaves the
        // bound at infinity when the series carries no frequency
        // information (`structural.py`, `cycle_period_bounds=(2, inf)`)
        // and does not meet the singularity because its diffuse tolerance
        // is an absolute `1e-10` on `F_inf`, which quietly clips it;
        // `uc_properties.rs` pins the divergence with numbers.
        let pmax = if pmax.is_finite() { pmax } else { n as f64 };
        if spec.cycle && pmax <= pmin {
            return Err(invalid(format!(
                "cycle_period_bounds = ({pmin}, inf) with {n} observations: an infinite \
                 upper period bound means the sample length, so the admissible band is \
                 empty (a cycle longer than the sample is not identified); pass a finite \
                 cycle_period_bounds with min < {n} or a longer series"
            )));
        }
        let freq_bounds = (
            2.0 * std::f64::consts::PI / pmax,
            2.0 * std::f64::consts::PI / pmin,
        );
        for (j, col) in spec.exog.iter().enumerate() {
            if col.len() != n {
                return Err(invalid(format!(
                    "exog column {j} has length {} but y has {n} periods; every regressor \
                     must be aligned with y",
                    col.len()
                )));
            }
            if col.iter().any(|v| !v.is_finite()) {
                return Err(invalid(format!(
                    "exog column {j} contains a NaN or infinity; regressors must be finite \
                     (NaN marks a missing value in y only)"
                )));
            }
            if col.iter().all(|&v| v == 0.0) {
                return Err(invalid(format!(
                    "exog column {j} is identically zero, so its coefficient is not \
                     identified; drop the column"
                )));
            }
        }
        let has_stochastic = irregular
            || (level && stochastic_level)
            || (trend && stochastic_trend)
            || (spec.seasonal.is_some() && spec.stochastic_seasonal)
            || spec.freq_seasonal.iter().any(|f| f.stochastic)
            || (spec.cycle && spec.stochastic_cycle);
        if !has_stochastic {
            return Err(invalid(format!(
                "level = {:?} with every other component deterministic leaves the model \
                 without a stochastic element (a likelihood needs one); pick a level \
                 specification with an irregular term (e.g. \"dconstant\" instead of \
                 \"fixed intercept\", \"dtrend\" instead of \"fixed slope\") or make a \
                 component stochastic",
                spec.trend.name()
            )));
        }
        let unused_state =
            !level && spec.seasonal.is_none() && spec.freq_seasonal.is_empty() && !spec.cycle;

        // State offsets in statsmodels' order.
        let mut i = 0usize;
        let mut state_names = Vec::new();
        let level_idx = if level {
            i += 1;
            state_names.push("level".to_string());
            Some(i - 1)
        } else {
            None
        };
        let trend_idx = if trend {
            i += 1;
            state_names.push("trend".to_string());
            Some(i - 1)
        } else {
            None
        };
        let seasonal_off = spec.seasonal.map(|s| {
            let off = i;
            i += s - 1;
            for j in 0..(s - 1) {
                state_names.push(if j == 0 {
                    "seasonal".to_string()
                } else {
                    format!("seasonal.L{j}")
                });
            }
            off
        });
        let mut freq_offs = Vec::new();
        for f in &spec.freq_seasonal {
            freq_offs.push(i);
            for h in 1..=f.harmonics {
                state_names.push(format!("freq_seasonal_{}({}).h{h}", f.period, f.harmonics));
                state_names.push(format!("freq_seasonal_{}({}).h{h}*", f.period, f.harmonics));
            }
            i += 2 * f.harmonics;
        }
        let cycle_off = if spec.cycle {
            state_names.push("cycle".to_string());
            state_names.push("cycle.auxiliary".to_string());
            i += 2;
            Some(i - 2)
        } else {
            None
        };
        let k_states = if unused_state {
            state_names.push("unused".to_string());
            1
        } else {
            i
        };

        // Parameters in statsmodels' order.
        let mut specs = Vec::new();
        let idx_irregular = if irregular {
            specs.push(ParamSpec {
                name: "sigma2.irregular".to_string(),
                kind: ParamKind::Variance,
            });
            Some(0)
        } else {
            None
        };
        let mut state_vars = Vec::new();
        let mut col = 0usize;
        let mut push_var = |name: String, rows: Vec<usize>| {
            let param = specs.len();
            specs.push(ParamSpec {
                name,
                kind: ParamKind::Variance,
            });
            let width = rows.len();
            state_vars.push(StateVar { param, col, rows });
            col += width;
        };
        if let (Some(l), true) = (level_idx, stochastic_level) {
            push_var("sigma2.level".to_string(), vec![l]);
        }
        if let (Some(t), true) = (trend_idx, stochastic_trend) {
            push_var("sigma2.trend".to_string(), vec![t]);
        }
        if let (Some(off), true) = (seasonal_off, spec.stochastic_seasonal) {
            push_var("sigma2.seasonal".to_string(), vec![off]);
        }
        for (b, f) in spec.freq_seasonal.iter().enumerate() {
            if f.stochastic {
                let off = freq_offs[b];
                push_var(
                    format!("sigma2.freq_seasonal_{}({})", f.period, f.harmonics),
                    (off..off + 2 * f.harmonics).collect(),
                );
            }
        }
        if let (Some(off), true) = (cycle_off, spec.stochastic_cycle) {
            push_var("sigma2.cycle".to_string(), vec![off, off + 1]);
        }
        let k_posdef = col.max(1);
        let idx_cycle_freq = if spec.cycle {
            specs.push(ParamSpec {
                name: "frequency.cycle".to_string(),
                kind: ParamKind::Bounded {
                    low: freq_bounds.0,
                    high: freq_bounds.1,
                },
            });
            Some(specs.len() - 1)
        } else {
            None
        };
        let idx_damping = if spec.cycle && spec.damped_cycle {
            specs.push(ParamSpec {
                name: "damping.cycle".to_string(),
                kind: ParamKind::Bounded {
                    low: 0.0,
                    high: 1.0,
                },
            });
            Some(specs.len() - 1)
        } else {
            None
        };
        let idx_beta = specs.len();
        for j in 0..spec.exog.len() {
            specs.push(ParamSpec {
                name: format!("beta.x{}", j + 1),
                kind: ParamKind::Free,
            });
        }

        Ok(Layout {
            trend_spec: spec.trend,
            level,
            seasonal: spec.seasonal,
            freq: spec.freq_seasonal.clone(),
            freq_bounds,
            exog: spec.exog.clone(),
            unused_state,
            k_states,
            k_posdef,
            level_idx,
            trend_idx,
            seasonal_off,
            freq_offs,
            cycle_off,
            specs,
            idx_irregular,
            state_vars,
            idx_cycle_freq,
            idx_damping,
            idx_beta,
            state_names,
        })
    }

    fn k_params(&self) -> usize {
        self.specs.len()
    }

    fn k_exog(&self) -> usize {
        self.exog.len()
    }

    fn k_diffuse(&self) -> usize {
        if self.unused_state {
            0
        } else {
            self.k_states
        }
    }

    /// The state-space system at `params`.
    fn build_ssm(&self, params: &[f64]) -> Result<LinearGaussianSSM, SsmError> {
        let m = self.k_states;
        let r = self.k_posdef;
        let mut z = Mat::<f64>::zeros(1, m);
        let mut t = Mat::<f64>::zeros(m, m);
        let mut sel = Mat::<f64>::zeros(m, r);
        let mut q = Mat::<f64>::zeros(r, r);
        let h = Mat::from_fn(1, 1, |_, _| self.idx_irregular.map_or(0.0, |i| params[i]));
        if let Some(l) = self.level_idx {
            z[(0, l)] = 1.0;
            t[(l, l)] = 1.0;
            if let Some(tr) = self.trend_idx {
                t[(l, tr)] = 1.0;
            }
        }
        if let Some(tr) = self.trend_idx {
            t[(tr, tr)] = 1.0;
        }
        if let (Some(off), Some(s)) = (self.seasonal_off, self.seasonal) {
            let n = s - 1;
            z[(0, off)] = 1.0;
            for j in 0..n {
                t[(off, off + j)] = -1.0;
            }
            for j in 1..n {
                t[(off + j, off + j - 1)] = 1.0;
            }
        }
        for (b, f) in self.freq.iter().enumerate() {
            let off = self.freq_offs[b];
            let lam = 2.0 * std::f64::consts::PI / f.period;
            for k in 1..=f.harmonics {
                let o = off + 2 * (k - 1);
                z[(0, o)] = 1.0;
                let (s, c) = (lam * k as f64).sin_cos();
                t[(o, o)] = c;
                t[(o, o + 1)] = s;
                t[(o + 1, o)] = -s;
                t[(o + 1, o + 1)] = c;
            }
        }
        if let Some(off) = self.cycle_off {
            z[(0, off)] = 1.0;
            let freq = self.idx_cycle_freq.map_or(0.0, |i| params[i]);
            let rho = self.idx_damping.map_or(1.0, |i| params[i]);
            let (s, c) = freq.sin_cos();
            t[(off, off)] = rho * c;
            t[(off, off + 1)] = rho * s;
            t[(off + 1, off)] = -rho * s;
            t[(off + 1, off + 1)] = rho * c;
        }
        for sv in &self.state_vars {
            let v = params[sv.param];
            for (k, &row) in sv.rows.iter().enumerate() {
                sel[(row, sv.col + k)] = 1.0;
                q[(sv.col + k, sv.col + k)] = v;
            }
        }
        let init = if self.unused_state {
            Initialization::Known {
                a1: vec![0.0],
                p1: Mat::zeros(1, 1),
            }
        } else {
            Initialization::Diffuse
        };
        LinearGaussianSSM::builder(1, m, r)
            .z(z)
            .h(h)
            .t(t)
            .r(sel)
            .q(q)
            .initialization(init)
            .build()
    }

    /// `beta' x_t` for every period (zero without regressors).
    fn regression_effect(&self, params: &[f64], n: usize) -> Vec<f64> {
        let mut eff = vec![0.0; n];
        for (j, col) in self.exog.iter().enumerate() {
            let b = params[self.idx_beta + j];
            for (e, x) in eff.iter_mut().zip(col) {
                *e += b * x;
            }
        }
        eff
    }

    /// Log-likelihood at `params` (NaN where undefined).
    fn loglik_value(&self, y: &[f64], params: &[f64]) -> f64 {
        let model = match self.build_ssm(params) {
            Ok(m) => m,
            Err(_) => return f64::NAN,
        };
        let eff = self.regression_effect(params, y.len());
        let ym = Mat::from_fn(y.len(), 1, |i, _| y[i] - eff[i]);
        match model.filter(ym.as_ref()) {
            Ok(fo) => fo.loglik,
            Err(_) => f64::NAN,
        }
    }

    /// The deterministic ladder of starting values (standardized units).
    fn starting_values(&self, y: &[f64], n_starts: usize) -> Vec<Vec<f64>> {
        let rows: Vec<usize> = (0..y.len()).filter(|&t| y[t].is_finite()).collect();
        let beta = ols(&self.exog, y, &rows).unwrap_or_else(|| vec![0.0; self.k_exog()]);
        let mut resid: Vec<f64> = rows.iter().map(|&t| y[t]).collect();
        for (j, col) in self.exog.iter().enumerate() {
            for (r, &t) in rows.iter().enumerate() {
                resid[r] -= beta[j] * col[t];
            }
        }
        let detrended: Vec<f64> = if self.level {
            resid.windows(2).map(|w| w[1] - w[0]).collect()
        } else {
            resid.clone()
        };
        let mut base = variance(&detrended);
        if !(base.is_finite() && base > 0.0) {
            base = 1.0;
        }
        // Cycle frequency start: the periodogram peak of the (differenced)
        // residual inside the frequency bounds — the cycle likelihood is
        // multimodal in the frequency, so the bound midpoint alone can
        // land on a local optimum. Falls back to the midpoint when no
        // Fourier frequency lies inside the bounds.
        let (lo, hi) = self.freq_bounds;
        let pgram_freq = periodogram_peak(&detrended, lo, hi).unwrap_or(0.5 * (lo + hi));
        let ladder = [1.0, 0.1, 10.0, 0.01, 100.0];
        // Cycle starts: the periodogram peak first (the frequency
        // likelihood is multimodal, and the midpoint of the admissible
        // band alone lands on a local optimum), then the band's middle
        // and quarter points.
        let freq_pos = [f64::NAN, 0.5, 0.25, 0.75, f64::NAN];
        // Damping starts spread over the interval rather than clustered
        // near one: a cycle whose damping wants to be near zero (the
        // cycle degenerating towards white noise, which happens whenever
        // the series has no cycle in the admissible band) is a corner the
        // search has to be able to walk to.
        let damp = [0.9, 0.5, 0.1, 0.95, 0.7];
        let mut starts = Vec::with_capacity(n_starts);
        for k in 0..n_starts {
            let idx = k % ladder.len();
            let scale = ladder[idx];
            let mut p = vec![0.0; self.k_params()];
            if let Some(i) = self.idx_irregular {
                p[i] = if self.level { 0.25 * base } else { base };
            }
            for sv in &self.state_vars {
                let name = self.specs[sv.param].name.as_str();
                let frac = if name == "sigma2.level" {
                    0.5
                } else if name == "sigma2.trend" {
                    0.01
                } else {
                    0.1
                };
                p[sv.param] = frac * base * scale;
            }
            if let Some(i) = self.idx_cycle_freq {
                p[i] = if freq_pos[idx].is_nan() {
                    pgram_freq
                } else {
                    lo + freq_pos[idx] * (hi - lo)
                };
            }
            if let Some(i) = self.idx_damping {
                p[i] = damp[idx];
            }
            for (j, b) in beta.iter().enumerate() {
                p[self.idx_beta + j] = *b;
            }
            starts.push(p);
        }
        starts
    }

    fn validate_fixed(&self, fixed: &[f64]) -> Result<(), SsmError> {
        if fixed.len() != self.k_params() {
            return Err(invalid(format!(
                "fixed_params has length {} but the specification has {} parameters, in \
                 this order: {}",
                fixed.len(),
                self.k_params(),
                self.specs
                    .iter()
                    .map(|s| s.name.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            )));
        }
        for (i, (spec, &v)) in self.specs.iter().zip(fixed).enumerate() {
            if !v.is_finite() {
                return Err(invalid(format!(
                    "fixed_params[{i}] ({}) = {v} is not finite",
                    spec.name
                )));
            }
            match spec.kind {
                ParamKind::Variance if v < 0.0 => {
                    return Err(invalid(format!(
                        "fixed_params[{i}] ({}) = {v}: a variance must be >= 0",
                        spec.name
                    )));
                }
                ParamKind::Bounded { low, high } if v < low || v > high => {
                    return Err(invalid(format!(
                        "fixed_params[{i}] ({}) = {v} lies outside [{low}, {high}]",
                        spec.name
                    )));
                }
                _ => {}
            }
        }
        Ok(())
    }
}

/// Fits (or evaluates at `fixed_params`) a structural time-series model;
/// see the module docs for the model, the estimator, and the reporting
/// conventions.
///
/// `y` may contain NaN for missing periods.
///
/// # Errors
///
/// [`SsmError::InvalidSpec`] for every malformed input, naming the
/// argument; the filter's own errors when the likelihood cannot be
/// evaluated at `fixed_params`.
pub fn unobserved_components(
    y: &[f64],
    spec: &UcSpec,
    opts: &UcOptions,
) -> Result<UcFit, SsmError> {
    let n = y.len();
    if n == 0 {
        return Err(invalid(
            "y is empty; pass at least one observation".to_string(),
        ));
    }
    if y.iter().any(|v| v.is_infinite()) {
        return Err(invalid(
            "y contains an infinity; entries must be finite or NaN (missing)".to_string(),
        ));
    }
    let layout = Layout::new(spec, n)?;
    let nobs_observed = y.iter().filter(|v| v.is_finite()).count();
    let need = layout.k_states + 2;
    if nobs_observed < need {
        return Err(invalid(format!(
            "y has {nobs_observed} observed (non-NaN) values but the specification has \
             {} states; at least k_states + 2 = {need} observations are needed",
            layout.k_states
        )));
    }
    let h = opts.forecast_steps;
    if h > MAX_FORECAST_STEPS {
        return Err(invalid(format!(
            "forecast_steps = {h}: at most {MAX_FORECAST_STEPS} forecast periods are \
             accepted (the horizon costs O(forecast_steps * k_states^2) work and two \
             buffers of that length, and a forecast this far out carries no information \
             about y)"
        )));
    }
    if !opts.forecast_exog.is_empty() {
        if h == 0 {
            return Err(invalid(
                "forecast_exog was given but forecast_steps = 0; it is inert without a \
                 forecast horizon — pass forecast_steps or drop forecast_exog"
                    .to_string(),
            ));
        }
        if layout.k_exog() == 0 {
            return Err(invalid(
                "forecast_exog was given but the model has no regressors (exog is empty); \
                 pass exog as well or drop forecast_exog"
                    .to_string(),
            ));
        }
    }
    if h > 0 && layout.k_exog() > 0 {
        if opts.forecast_exog.len() != layout.k_exog() {
            return Err(invalid(format!(
                "forecast_steps = {h} with {} regressors requires forecast_exog with {} \
                 columns of length {h} (got {} columns)",
                layout.k_exog(),
                layout.k_exog(),
                opts.forecast_exog.len()
            )));
        }
        for (j, col) in opts.forecast_exog.iter().enumerate() {
            if col.len() != h {
                return Err(invalid(format!(
                    "forecast_exog column {j} has length {} but forecast_steps = {h}",
                    col.len()
                )));
            }
            if col.iter().any(|v| !v.is_finite()) {
                return Err(invalid(format!(
                    "forecast_exog column {j} contains a NaN or infinity"
                )));
            }
        }
    }

    if let Some(fixed) = &opts.fixed_params {
        layout.validate_fixed(fixed)?;
        let at_boundary: Vec<bool> = layout
            .specs
            .iter()
            .zip(fixed)
            .map(|(s, &v)| s.kind == ParamKind::Variance && v == 0.0)
            .collect();
        let se = vec![f64::NAN; layout.k_params()];
        return evaluate(
            &layout,
            y,
            fixed,
            opts,
            false,
            true,
            (0, 0),
            se,
            at_boundary,
        );
    }

    if opts.n_starts == 0 {
        return Err(invalid(
            "n_starts = 0: the search needs at least one starting value".to_string(),
        ));
    }
    if opts.n_starts > MAX_STARTS {
        return Err(invalid(format!(
            "n_starts = {}: at most {MAX_STARTS} starting values are accepted (the \
             deterministic ladder has five distinct rungs and then repeats, so more \
             starts cost time without covering new ground)",
            opts.n_starts
        )));
    }
    // Scale-adaptive search: standardize y and the regressors, estimate,
    // map back exactly, then evaluate on the original data.
    let observed: Vec<f64> = y.iter().copied().filter(|v| v.is_finite()).collect();
    let s = variance(&observed).sqrt();
    if !(s.is_finite() && s > 0.0) {
        return Err(invalid(
            "y is constant (standard deviation 0): every variance would be estimated at \
             zero and the likelihood is unbounded; a constant series has no structural \
             model to fit"
                .to_string(),
        ));
    }
    let y_s: Vec<f64> = y.iter().map(|v| v / s).collect();
    let col_scale: Vec<f64> = spec
        .exog
        .iter()
        .map(|col| col.iter().fold(0.0f64, |m, v| m.max(v.abs())))
        .collect();
    let mut spec_s = spec.clone();
    for (col, &c) in spec_s.exog.iter_mut().zip(&col_scale) {
        for v in col.iter_mut() {
            *v /= c;
        }
    }
    let layout_s = Layout::new(&spec_s, n)?;
    let starts = layout_s.starting_values(&y_s, opts.n_starts);
    let outcome = maximize(&layout_s.specs, &starts, |p| layout_s.loglik_value(&y_s, p))?;
    let back = |i: usize, v: f64| -> f64 {
        match layout_s.specs[i].kind {
            ParamKind::Variance => v * s * s,
            ParamKind::Bounded { .. } => v,
            ParamKind::Free => v * s / col_scale[i - layout_s.idx_beta],
        }
    };
    let params: Vec<f64> = outcome
        .params
        .iter()
        .enumerate()
        .map(|(i, &v)| back(i, v))
        .collect();
    let se: Vec<f64> = outcome
        .se
        .iter()
        .enumerate()
        .map(|(i, &v)| back(i, v))
        .collect();
    evaluate(
        &layout,
        y,
        &params,
        opts,
        true,
        outcome.converged,
        (outcome.n_iter, outcome.n_fevals),
        se,
        outcome.at_boundary,
    )
}

#[allow(clippy::too_many_arguments)]
fn evaluate(
    layout: &Layout,
    y: &[f64],
    params: &[f64],
    opts: &UcOptions,
    estimated: bool,
    converged: bool,
    (n_iter, n_fevals): (usize, usize),
    se: Vec<f64>,
    at_boundary: Vec<bool>,
) -> Result<UcFit, SsmError> {
    let n = y.len();
    let model = layout.build_ssm(params)?;
    let eff = layout.regression_effect(params, n);
    let y_adj: Vec<f64> = y.iter().zip(&eff).map(|(v, e)| v - e).collect();
    let ym = Mat::from_fn(n, 1, |i, _| y_adj[i]);
    let so = smooth_univariate(&model, ym.as_ref())?;
    let fo = &so.filter;
    let m = layout.k_states;

    let diag = |mat: &Mat<f64>| -> Vec<f64> { (0..m).map(|i| mat[(i, i)]).collect() };
    let filtered_state = fo.filtered_state.clone();
    let filtered_state_var: Vec<Vec<f64>> = fo.filtered_state_cov.iter().map(diag).collect();
    let smoothed_state = so.smoothed_state.clone();
    let smoothed_state_var: Vec<Vec<f64>> = so.smoothed_state_cov.iter().map(diag).collect();

    let component = |idx: &[usize]| -> UcComponent {
        let sum_over = |states: &[Vec<f64>]| -> Vec<f64> {
            states
                .iter()
                .map(|st| idx.iter().map(|&i| st[i]).sum())
                .collect()
        };
        UcComponent {
            filtered: sum_over(&filtered_state),
            filtered_var: sum_over(&filtered_state_var),
            smoothed: sum_over(&smoothed_state),
            smoothed_var: sum_over(&smoothed_state_var),
        }
    };
    let level = layout.level_idx.map(|i| component(&[i]));
    let slope = layout.trend_idx.map(|i| component(&[i]));
    let seasonal = layout.seasonal_off.map(|i| component(&[i]));
    let freq_seasonal: Vec<UcComponent> = layout
        .freq
        .iter()
        .zip(&layout.freq_offs)
        .map(|(f, &off)| {
            let idx: Vec<usize> = (0..f.harmonics).map(|k| off + 2 * k).collect();
            component(&idx)
        })
        .collect();
    let cycle = layout.cycle_off.map(|i| component(&[i]));

    let mut fitted = vec![f64::NAN; n];
    let mut resid = vec![f64::NAN; n];
    let mut std_resid = vec![f64::NAN; n];
    for t in 0..n {
        let z = model.z().at(t);
        let zrow: Vec<f64> = (0..m).map(|j| z[(0, j)]).collect();
        let pred = eff[t] + dot(&zrow, &fo.predicted_state[t]);
        fitted[t] = pred;
        if y[t].is_finite() {
            resid[t] = y[t] - pred;
            let step = &fo.steps[t];
            if t >= fo.d_diffuse && step.observed && step.f_star > 0.0 {
                std_resid[t] = step.v / step.f_star.sqrt();
            }
        }
    }

    let h = opts.forecast_steps;
    let mut forecast = Vec::with_capacity(h);
    let mut forecast_var = Vec::with_capacity(h);
    if h > 0 {
        let tr = model.t().at(n);
        let z = model.z().at(n);
        let zrow: Vec<f64> = (0..m).map(|j| z[(0, j)]).collect();
        let hvar = model.h().at(n)[(0, 0)];
        let rqr = model.rqr(n)?;
        let mut a = fo.predicted_state[n].clone();
        let mut p = fo.predicted_state_cov[n].clone();
        // The diffuse part is carried through the horizon too: a state the
        // observation never loads on (the sine state of a harmonic at
        // frequency pi, say) keeps a diffuse prior forever without making
        // the forecast infinite, so the test is on Z P_inf Z' along the
        // horizon — the same washout tolerance the filter uses, P_inf
        // being a rank indicator that ends at roundoff, not at exact zero.
        let mut p_inf = fo.predicted_diffuse_state_cov[n].clone();
        for step in 0..h {
            if dot(&zrow, &mat_vec(p_inf.as_ref(), &zrow)) > TOLERANCE_RANK {
                return Err(invalid(format!(
                    "forecast_steps = {h}: at horizon {} the forecast still carries a \
                     diffuse (infinite) variance — the diffuse initialization of the \
                     states the observation loads on has not resolved by the end of the \
                     sample (nobs_diffuse = {} of {n} periods); supply more observations \
                     or drop the forecast",
                    step + 1,
                    fo.d_diffuse
                )));
            }
            let mut xb = 0.0;
            for (j, col) in opts.forecast_exog.iter().enumerate() {
                xb += params[layout.idx_beta + j] * col[step];
            }
            forecast.push(xb + dot(&zrow, &a));
            forecast_var.push(dot(&zrow, &mat_vec(p.as_ref(), &zrow)) + hvar);
            a = mat_vec(tr, &a);
            let mut pn = sandwich(tr, p.as_ref());
            pn += &rqr;
            p = pn;
            p_inf = sandwich(tr, p_inf.as_ref());
        }
    }

    let k_params = layout.k_params();
    let k_diffuse = layout.k_diffuse();
    let df = (k_params + k_diffuse) as f64;
    let loglik = fo.loglik;
    Ok(UcFit {
        trend_specification: layout.trend_spec.name().to_string(),
        param_names: layout.specs.iter().map(|s| s.name.clone()).collect(),
        params: params.to_vec(),
        se,
        at_boundary,
        loglik,
        aic: 2.0 * df - 2.0 * loglik,
        bic: -2.0 * loglik + df * (n as f64).ln(),
        nobs: n,
        nobs_observed: y.iter().filter(|v| v.is_finite()).count(),
        nobs_diffuse: fo.d_diffuse,
        k_states: m,
        k_diffuse,
        k_params,
        estimated,
        converged,
        n_iter,
        n_fevals,
        state_names: layout.state_names.clone(),
        filtered_state,
        filtered_state_var,
        smoothed_state,
        smoothed_state_var,
        level,
        slope,
        seasonal,
        freq_seasonal,
        cycle,
        fitted,
        resid,
        std_resid,
        forecast,
        forecast_var,
    })
}
