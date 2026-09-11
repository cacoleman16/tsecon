//! Initial-state heuristics.
//!
//! [`heuristic_initial_states`] is the Hyndman et al. (2008, section 2.6.1)
//! heuristic exactly as statsmodels implements it
//! (`statsmodels.tsa.exponential_smoothing.initialization._initialization_heuristic`,
//! the `initialization_method="heuristic"` of both `ETSModel` and
//! `holtwinters.ExponentialSmoothing`):
//!
//! 1. **Seasonal states** (seasonal models): take the first
//!    `k = max(min(5, floor(n / m)), ceil((10 + 2 floor(m / 2)) / m))`
//!    cycles of the data, compute the centred moving average of order `m`
//!    (the classical `2 x m` average with half weights at the ends when
//!    `m` is even), detrend the cycles by subtraction (additive) or
//!    division (multiplicative), average the detrended values by season
//!    over the cycles (ignoring the ends where the average is undefined),
//!    and normalise the `m` indices to mean zero (additive) or mean one
//!    (multiplicative).
//! 2. **Level and trend**: regress the first ten values of the moving
//!    average (or of the data itself for a non-seasonal model) on an
//!    intercept and a linear time index `1..10`; the level is the
//!    intercept, the additive trend the slope, the multiplicative trend
//!    `1 + slope / intercept`.
//!
//! [`simple_initial_states`] is statsmodels' `_initialization_simple`
//! (Hyndman & Athanasopoulos, FPP3 section 8.6 conventions): level `y_0`
//! and trend `y_1 - y_0` (or `y_1 / y_0`) without seasonality; with
//! seasonality, the level is the mean of the first cycle, the trend the
//! mean seasonal difference over the second cycle divided by `m` (as
//! statsmodels does for both trend types), and the seasonal states the
//! first cycle minus / divided by the level. It is the starting point of
//! the estimated initialisation when the sample is too short for the
//! heuristic.

use crate::error::EtsError;
use crate::filter::check_data;
use crate::spec::{Component, EtsSpec, EtsStates};

/// Minimum sample for the heuristic: ten seasonally adjusted values after
/// losing `floor(m / 2)` at each end of the moving average.
pub(crate) fn heuristic_min_obs(spec: &EtsSpec) -> usize {
    10 + 2 * (spec.m() / 2)
}

/// Least squares of `v` on `[1, t]`, `t = 1..=v.len()`: `(intercept, slope)`.
fn line_fit(v: &[f64]) -> (f64, f64) {
    let n = v.len() as f64;
    let tbar = (n + 1.0) / 2.0;
    let ybar = v.iter().sum::<f64>() / n;
    let mut sxy = 0.0;
    let mut sxx = 0.0;
    for (i, &yi) in v.iter().enumerate() {
        let dt = (i as f64 + 1.0) - tbar;
        sxy += dt * (yi - ybar);
        sxx += dt * dt;
    }
    let slope = if sxx > 0.0 { sxy / sxx } else { 0.0 };
    (ybar - slope * tbar, slope)
}

/// Centred moving average of order `m` (the `2 x m` average for even `m`),
/// `None` where undefined.
fn centred_ma(v: &[f64], m: usize) -> Vec<Option<f64>> {
    let n = v.len();
    let mut out = vec![None; n];
    if m % 2 == 1 {
        let half = (m - 1) / 2;
        for i in half..n.saturating_sub(half) {
            let w = &v[i - half..=i + half];
            out[i] = Some(w.iter().sum::<f64>() / m as f64);
        }
    } else {
        let half = m / 2;
        // pandas: rolling(m, center=True).mean() at i covers [i-half, i+half-1];
        // then .shift(-1).rolling(2).mean() averages the windows at i and i+1.
        for i in half..n.saturating_sub(half) {
            let a = v[i - half..i + half].iter().sum::<f64>() / m as f64;
            let b = v[i - half + 1..=i + half].iter().sum::<f64>() / m as f64;
            out[i] = Some((a + b) / 2.0);
        }
    }
    out
}

/// The Hyndman et al. (2008, section 2.6.1) heuristic initial states, as
/// statsmodels' `initialization_method="heuristic"` computes them (see
/// the module docs). Needs at least ten observations, and for a seasonal
/// model at least two full cycles and `10 + 2 floor(m / 2)` observations.
///
/// # Errors
///
/// [`EtsError::TooFewObservations`] when the sample is too short;
/// [`EtsError::NonFinite`] / [`EtsError::NonPositiveData`] for bad data;
/// [`EtsError::Degenerate`] if the heuristic produces a non-positive
/// multiplicative state or a non-finite value.
pub fn heuristic_initial_states(spec: &EtsSpec, y: &[f64]) -> Result<EtsStates, EtsError> {
    spec.validate()?;
    check_data(spec, y)?;
    let n = y.len();
    if n < 10 {
        return Err(EtsError::TooFewObservations {
            needed: 10,
            got: n,
            what: "the heuristic initialisation (it regresses the first ten values on a \
                   linear trend)"
                .into(),
        });
    }
    let m = spec.m();
    let mut seasonal = None;
    let mut level_source: Vec<f64> = y.to_vec();
    if spec.has_seasonal() {
        if n < 2 * m {
            return Err(EtsError::TooFewObservations {
                needed: 2 * m,
                got: n,
                what: format!(
                    "the heuristic initialisation of a seasonal model with \
                     seasonal_periods = {m} (two full cycles)"
                ),
            });
        }
        let min_obs = heuristic_min_obs(spec);
        if n < min_obs {
            return Err(EtsError::TooFewObservations {
                needed: min_obs,
                got: n,
                what: format!(
                    "the heuristic initialisation of a seasonal model with \
                     seasonal_periods = {m} (10 + 2 * floor(m / 2) values after the \
                     centred moving average)"
                ),
            });
        }
        let mut k_cycles = (n / m).min(5);
        k_cycles = k_cycles.max(min_obs.div_ceil(m));
        let series = &y[..(m * k_cycles).min(n)];
        let ma = centred_ma(series, m);
        // Average detrended values by season over the cycles.
        let mut sums = vec![0.0; m];
        let mut counts = vec![0usize; m];
        for (i, (&v, t)) in series.iter().zip(&ma).enumerate() {
            if let Some(t) = t {
                let d = match spec.seasonal {
                    Component::Multiplicative => v / t,
                    _ => v - t,
                };
                sums[i % m] += d;
                counts[i % m] += 1;
            }
        }
        let mut s: Vec<f64> = (0..m)
            .map(|j| {
                if counts[j] > 0 {
                    sums[j] / counts[j] as f64
                } else {
                    f64::NAN
                }
            })
            .collect();
        let mean = s.iter().sum::<f64>() / m as f64;
        match spec.seasonal {
            Component::Multiplicative => s.iter_mut().for_each(|v| *v /= mean),
            _ => s.iter_mut().for_each(|v| *v -= mean),
        }
        for (j, &v) in s.iter().enumerate() {
            if !v.is_finite() {
                return Err(EtsError::Degenerate {
                    what: "the heuristic seasonal index is not finite",
                    index: j,
                });
            }
            if spec.seasonal == Component::Multiplicative && v <= 0.0 {
                return Err(EtsError::Degenerate {
                    what: "the heuristic multiplicative seasonal index is not positive \
                           (the centred moving average changes sign within the first \
                           cycles)",
                    index: j,
                });
            }
        }
        seasonal = Some(s);
        level_source = ma.into_iter().flatten().collect();
    }
    let head = &level_source[..level_source.len().min(10)];
    if head.len() < 10 {
        return Err(EtsError::TooFewObservations {
            needed: 10,
            got: head.len(),
            what: "the heuristic level/trend regression".into(),
        });
    }
    let (level, slope) = line_fit(head);
    let trend = match spec.trend {
        Component::None => None,
        Component::Additive => Some(slope),
        Component::Multiplicative => Some(1.0 + slope / level),
    };
    let st = EtsStates {
        level,
        trend,
        seasonal,
    };
    if !st.level.is_finite() || st.trend.is_some_and(|b| !b.is_finite()) {
        return Err(EtsError::Degenerate {
            what: "the heuristic level/trend regression is not finite",
            index: 0,
        });
    }
    Ok(st)
}

/// statsmodels' `_initialization_simple`: the first observations as the
/// initial states (see the module docs). Needs two observations with a
/// trend and two full cycles with a seasonal.
///
/// # Errors
///
/// [`EtsError::TooFewObservations`]; [`EtsError::NonFinite`] /
/// [`EtsError::NonPositiveData`] for bad data.
pub fn simple_initial_states(spec: &EtsSpec, y: &[f64]) -> Result<EtsStates, EtsError> {
    spec.validate()?;
    check_data(spec, y)?;
    let n = y.len();
    let m = spec.m();
    if spec.has_seasonal() {
        if n < 2 * m {
            return Err(EtsError::TooFewObservations {
                needed: 2 * m,
                got: n,
                what: format!(
                    "the simple initialisation of a seasonal model with seasonal_periods \
                     = {m} (two full cycles)"
                ),
            });
        }
        let level = y[..m].iter().sum::<f64>() / m as f64;
        let trend = if spec.has_trend() {
            // statsmodels: mean over the second cycle of (y_i - y_{i-m}) / m,
            // for both trend types.
            Some(
                (m..2 * m)
                    .map(|i| (y[i] - y[i - m]) / m as f64)
                    .sum::<f64>()
                    / m as f64,
            )
        } else {
            None
        };
        let seasonal: Vec<f64> = match spec.seasonal {
            Component::Multiplicative => y[..m].iter().map(|v| v / level).collect(),
            _ => y[..m].iter().map(|v| v - level).collect(),
        };
        return Ok(EtsStates {
            level,
            trend,
            seasonal: Some(seasonal),
        });
    }
    let needed = if spec.has_trend() { 2 } else { 1 };
    if n < needed {
        return Err(EtsError::TooFewObservations {
            needed,
            got: n,
            what: "the simple initialisation".into(),
        });
    }
    let trend = match spec.trend {
        Component::None => None,
        Component::Additive => Some(y[1] - y[0]),
        Component::Multiplicative => Some(y[1] / y[0]),
    };
    Ok(EtsStates {
        level: y[0],
        trend,
        seasonal: None,
    })
}

/// The starting states for the estimated initialisation: the heuristic
/// when the sample allows it, the simple rule otherwise (statsmodels'
/// choice for `initialization_method="estimated"`), then normalised to
/// the crate's seasonal convention (additive indices sum to zero,
/// multiplicative indices average one, the level and an additive trend
/// absorbing the shift).
pub(crate) fn starting_states(spec: &EtsSpec, y: &[f64]) -> Result<EtsStates, EtsError> {
    let mut st = if y.len() >= heuristic_min_obs(spec) {
        heuristic_initial_states(spec, y)?
    } else {
        simple_initial_states(spec, y)?
    };
    normalise_seasonal(spec, &mut st);
    // A multiplicative trend must start strictly positive.
    if spec.trend == Component::Multiplicative {
        if let Some(b) = st.trend.as_mut() {
            if *b <= 0.0 || !b.is_finite() {
                *b = 1.0;
            }
        }
    }
    Ok(st)
}

/// Normalises the seasonal states in place (sum to zero / average one),
/// moving the shift into the level (and an additive trend under a
/// multiplicative seasonal), which leaves every fitted value unchanged.
pub(crate) fn normalise_seasonal(spec: &EtsSpec, st: &mut EtsStates) {
    if let Some(s) = st.seasonal.as_mut() {
        let m = s.len() as f64;
        let mean = s.iter().sum::<f64>() / m;
        match spec.seasonal {
            Component::Multiplicative => {
                if mean > 0.0 && mean.is_finite() {
                    s.iter_mut().for_each(|v| *v /= mean);
                    st.level *= mean;
                    if spec.trend == Component::Additive {
                        if let Some(b) = st.trend.as_mut() {
                            *b *= mean;
                        }
                    }
                }
            }
            _ => {
                s.iter_mut().for_each(|v| *v -= mean);
                st.level += mean;
            }
        }
    }
}
