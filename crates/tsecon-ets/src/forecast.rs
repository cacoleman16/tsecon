//! Point forecasts, forecast variances and prediction intervals.
//!
//! The point forecast is the recursion run forward with zero innovations
//! from the final state (R `forecast.ets`, statsmodels `ETSResults.forecast`
//! — "simulation without errors"). For the **class-1** models — additive
//! error, additive or no trend, additive or no seasonal (Hyndman et al.
//! 2008, chapter 6) — the `h`-step forecast-error variance has the closed
//! form
//!
//! ```text
//! v_h = sigma^2 [ 1 + sum_{j=1}^{h-1} c_j^2 ],   c_j = w' F^{j-1} g,
//! ```
//!
//! where `(w, F, g)` are the linear state-space matrices; `c_j` is the
//! `j`-step-ahead forecast from the state `g = (alpha, beta, 0, ..., 0,
//! gamma)` — the response of `y_{t+j}` to a unit innovation at `t` — so
//! the crate evaluates `c_j` with the same recursion that produces the
//! point forecast and never types the six per-model formulas of Table 6.1
//! (the fixture generator transcribes those and pins this implementation
//! to them). Prediction intervals are then Gaussian,
//! `mean_h ± z_{(1+level)/2} sqrt(v_h)`.
//!
//! For every other model (multiplicative error, trend or seasonal) no
//! closed form ships: `n_sim` innovation paths `e ~ N(0, sigma^2)` are
//! drawn from a seeded Philox stream (Box-Muller), pushed through the
//! innovations recursion, and the interval bounds are the empirical
//! `(1 - level)/2` and `(1 + level)/2` quantiles of the simulated
//! `y_{t+h}` (linear interpolation between order statistics), with the
//! variance the across-path sample variance.

use tsecon_rng::Stream;
use tsecon_stats::special::inv_norm_cdf;

use crate::error::EtsError;
use crate::filter::{extend, one_step, update, STATE_TOL};
use crate::fit::EtsFit;
use crate::spec::{Component, ErrorType, EtsParams, EtsSpec, EtsStates};

/// How the interval was computed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntervalMethod {
    /// Class-1 closed-form variance, Gaussian interval.
    Exact,
    /// Seeded simulation, empirical quantiles.
    Simulated,
}

impl IntervalMethod {
    /// The name reported in results.
    pub fn name(self) -> &'static str {
        match self {
            IntervalMethod::Exact => "exact",
            IntervalMethod::Simulated => "simulated",
        }
    }
}

/// Forecasts with intervals.
#[derive(Debug, Clone, PartialEq)]
pub struct EtsForecast {
    /// Point forecasts for steps `1..=h`.
    pub mean: Vec<f64>,
    /// Forecast-error variance per step (closed form or simulated).
    pub variance: Vec<f64>,
    /// Lower interval bound per step.
    pub lower: Vec<f64>,
    /// Upper interval bound per step.
    pub upper: Vec<f64>,
    /// The interval's nominal coverage.
    pub level: f64,
    /// Exact or simulated.
    pub method: IntervalMethod,
    /// Simulated paths used (`0` under the exact method).
    pub n_sim: usize,
    /// Seed of the simulation (`0` under the exact method).
    pub seed: u64,
}

/// One simulation step from `(l, b, ring)` at slot `idx` with innovation
/// `e`: returns the realised `y` and updates the state in place.
#[inline]
fn sim_step(
    spec: &EtsSpec,
    eff: (f64, f64, f64, f64),
    l: &mut f64,
    b: &mut f64,
    ring: &mut [f64],
    idx: usize,
    e: f64,
) -> Result<f64, &'static str> {
    let (alpha, beta, gamma, phi) = eff;
    let (q, phib) = extend(spec, phi, *l, *b);
    let s = if spec.has_seasonal() { ring[idx] } else { 0.0 };
    if spec.seasonal == Component::Multiplicative && s.abs() < STATE_TOL {
        return Err("a multiplicative seasonal state is zero");
    }
    let f = one_step(spec, q, s);
    if !f.is_finite() {
        return Err("the forecast is not finite");
    }
    let y = match spec.error {
        ErrorType::Additive => f + e,
        ErrorType::Multiplicative => f * (1.0 + e),
    };
    let (l_new, b_new, s_new) = update(spec, alpha, beta, gamma, *l, *b, q, phib, s, y)?;
    *l = l_new;
    *b = b_new;
    if spec.has_seasonal() {
        ring[idx] = s_new;
    }
    Ok(y)
}

/// Runs the recursion `h` steps forward from `state` with the given
/// innovations (`errors[j]` is the innovation of step `j`; the empty slice
/// or zeros give the point forecast). `state.seasonal[j]` is the seasonal
/// in force for step `j` (time order).
fn path(
    spec: &EtsSpec,
    params: &EtsParams,
    state: &EtsStates,
    errors: &[f64],
    h: usize,
) -> Result<Vec<f64>, EtsError> {
    let eff = params.effective();
    let mut l = state.level;
    let mut b = state.trend.unwrap_or(0.0);
    let mut ring: Vec<f64> = state.seasonal.clone().unwrap_or_default();
    let m = spec.m();
    let mut out = Vec::with_capacity(h);
    for j in 0..h {
        let e = errors.get(j).copied().unwrap_or(0.0);
        let y = sim_step(spec, eff, &mut l, &mut b, &mut ring, j % m, e)
            .map_err(|what| EtsError::Degenerate { what, index: j })?;
        out.push(y);
    }
    Ok(out)
}

fn check_h(h: usize) -> Result<(), EtsError> {
    if h == 0 {
        return Err(EtsError::InvalidOption {
            name: "horizon",
            value: 0.0,
            requirement: "at least one forecast step is needed",
        });
    }
    Ok(())
}

/// The zero-innovation path: point forecasts for steps `1..=h` from
/// `state` (statsmodels `forecast` / `simulate(random_errors=0)`).
///
/// # Errors
///
/// [`EtsError::InvalidOption`] for `h = 0`, [`EtsError::InvalidParameter`]
/// / [`EtsError::DimensionMismatch`] for parameters or states outside the
/// model's domain, [`EtsError::Degenerate`] if the recursion divides by a
/// zero state.
pub fn forecast_from_state(
    spec: &EtsSpec,
    params: &EtsParams,
    state: &EtsStates,
    h: usize,
) -> Result<Vec<f64>, EtsError> {
    spec.validate()?;
    check_h(h)?;
    params.check_domain()?;
    state.check_domain(spec)?;
    path(spec, params, state, &[], h)
}

/// Simulates `y_{t+1..t+h}` from `state` along the given innovation
/// paths (`errors[i][j]`: innovation of path `i` at step `j`, in the
/// error's own units — additive or relative), the arithmetic of
/// statsmodels `ETSResults.simulate(random_errors=...)`.
///
/// # Errors
///
/// As for [`forecast_from_state`]; [`EtsError::DimensionMismatch`] if the
/// paths have unequal lengths.
pub fn simulate_paths(
    spec: &EtsSpec,
    params: &EtsParams,
    state: &EtsStates,
    errors: &[Vec<f64>],
) -> Result<Vec<Vec<f64>>, EtsError> {
    spec.validate()?;
    params.check_domain()?;
    state.check_domain(spec)?;
    let h = errors.first().map_or(0, Vec::len);
    check_h(h)?;
    errors
        .iter()
        .map(|e| {
            if e.len() != h {
                return Err(EtsError::DimensionMismatch {
                    what: "innovation paths (all must have the same length)",
                    expected: h,
                    got: e.len(),
                });
            }
            if let Some((i, &v)) = e.iter().enumerate().find(|(_, v)| !v.is_finite()) {
                return Err(EtsError::NonFinite {
                    what: "errors",
                    index: i,
                    value: v,
                });
            }
            path(spec, params, state, e, h)
        })
        .collect()
}

/// The class-1 closed-form forecast-error variances `v_1..v_h` (see the
/// module docs).
///
/// # Errors
///
/// [`EtsError::NotClass1`] unless `spec` is class 1;
/// [`EtsError::InvalidOption`] for `h = 0` or a non-positive `sigma2`.
pub fn class1_forecast_variance(
    spec: &EtsSpec,
    params: &EtsParams,
    sigma2: f64,
    h: usize,
) -> Result<Vec<f64>, EtsError> {
    spec.validate()?;
    if !spec.is_class1() {
        return Err(EtsError::NotClass1 { spec: spec.name() });
    }
    check_h(h)?;
    if !(sigma2.is_finite() && sigma2 > 0.0) {
        return Err(EtsError::InvalidOption {
            name: "sigma2",
            value: sigma2,
            requirement: "the innovation variance must be a positive finite number",
        });
    }
    params.check_domain()?;
    let (alpha, beta, gamma, _) = params.effective();
    let m = spec.m();
    // g in time order for the forecast steps: the freshly updated
    // seasonal (loading gamma) is next used m steps ahead.
    let g = EtsStates {
        level: alpha,
        trend: spec.has_trend().then_some(beta),
        seasonal: spec.has_seasonal().then(|| {
            let mut s = vec![0.0; m];
            s[m - 1] = gamma;
            s
        }),
    };
    let c = if h > 1 {
        path(spec, params, &g, &[], h - 1)?
    } else {
        Vec::new()
    };
    let mut v = Vec::with_capacity(h);
    let mut acc = 1.0;
    v.push(sigma2 * acc);
    for cj in c {
        acc += cj * cj;
        v.push(sigma2 * acc);
    }
    Ok(v)
}

/// One standard normal draw (Box-Muller; two uniforms per draw, the sine
/// partner discarded — the convention of `tsecon-bootstrap`).
#[inline]
fn standard_normal(stream: &mut Stream) -> f64 {
    let u1 = 1.0 - stream.uniform_f64();
    let u2 = stream.uniform_f64();
    (-2.0 * u1.ln()).sqrt() * (core::f64::consts::TAU * u2).cos()
}

/// Type-7 (linear interpolation) quantile of a sorted slice.
fn quantile_sorted(s: &[f64], p: f64) -> f64 {
    let n = s.len();
    if n == 1 {
        return s[0];
    }
    let pos = p * (n - 1) as f64;
    let lo = pos.floor() as usize;
    let frac = pos - lo as f64;
    if lo + 1 >= n {
        s[n - 1]
    } else {
        s[lo] + frac * (s[lo + 1] - s[lo])
    }
}

/// Forecasts `h` steps from a fit with `level` prediction intervals —
/// exact for class-1 models, simulation-based (`n_sim` seeded paths)
/// otherwise; see the module docs.
///
/// # Errors
///
/// [`EtsError::InvalidOption`] for `h = 0`, `level` outside `(0, 1)`, or
/// `n_sim < 2` on a simulated model; [`EtsError::Degenerate`] if a
/// simulated path degenerates.
pub fn forecast(
    fit: &EtsFit,
    h: usize,
    level: f64,
    n_sim: usize,
    seed: u64,
) -> Result<EtsForecast, EtsError> {
    check_h(h)?;
    if !(level.is_finite() && level > 0.0 && level < 1.0) {
        return Err(EtsError::InvalidOption {
            name: "level",
            value: level,
            requirement: "the interval's nominal coverage must lie strictly between 0 \
                          and 1 (e.g. 0.95)",
        });
    }
    let spec = &fit.spec;
    let mean = forecast_from_state(spec, &fit.params, &fit.final_state, h)?;
    if spec.is_class1() {
        let variance = class1_forecast_variance(spec, &fit.params, fit.sigma2, h)?;
        let z = inv_norm_cdf((1.0 + level) / 2.0).map_err(|_| EtsError::InvalidOption {
            name: "level",
            value: level,
            requirement: "the normal quantile of (1 + level) / 2 must exist",
        })?;
        let lower = mean
            .iter()
            .zip(&variance)
            .map(|(m, v)| m - z * v.sqrt())
            .collect();
        let upper = mean
            .iter()
            .zip(&variance)
            .map(|(m, v)| m + z * v.sqrt())
            .collect();
        return Ok(EtsForecast {
            mean,
            variance,
            lower,
            upper,
            level,
            method: IntervalMethod::Exact,
            n_sim: 0,
            seed: 0,
        });
    }
    if n_sim < 2 {
        return Err(EtsError::InvalidOption {
            name: "n_sim",
            value: n_sim as f64,
            requirement: "the simulation-based interval needs at least two paths \
                          (5000 is the default)",
        });
    }
    let sigma = fit.sigma2.sqrt();
    let mut stream = Stream::new(seed);
    let mut errors = vec![vec![0.0; h]; n_sim];
    for row in errors.iter_mut() {
        for e in row.iter_mut() {
            *e = sigma * standard_normal(&mut stream);
        }
    }
    let paths = simulate_paths(spec, &fit.params, &fit.final_state, &errors)?;
    let mut variance = Vec::with_capacity(h);
    let mut lower = Vec::with_capacity(h);
    let mut upper = Vec::with_capacity(h);
    let mut col = vec![0.0; n_sim];
    for j in 0..h {
        for (i, p) in paths.iter().enumerate() {
            col[i] = p[j];
        }
        let mu = col.iter().sum::<f64>() / n_sim as f64;
        let var = col.iter().map(|v| (v - mu) * (v - mu)).sum::<f64>() / (n_sim - 1) as f64;
        variance.push(var);
        col.sort_by(|a, b| a.total_cmp(b));
        lower.push(quantile_sorted(&col, (1.0 - level) / 2.0));
        upper.push(quantile_sorted(&col, (1.0 + level) / 2.0));
    }
    Ok(EtsForecast {
        mean,
        variance,
        lower,
        upper,
        level,
        method: IntervalMethod::Simulated,
        n_sim,
        seed,
    })
}
