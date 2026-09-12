//! The innovations state-space recursion: one pass through the data at
//! fixed smoothing parameters and initial states, giving fitted values,
//! residuals, the state paths and the concentrated Gaussian
//! log-likelihood — Hyndman et al. (2008, Tables 2.2 and 2.3), in the
//! arithmetic of R's `forecast::ets` (`etscalc.c`, `update`).
//!
//! With `q = l_{t-1} op_b (b_{t-1} op_d phi)` the trend-extended level
//! (`op_b` is `+` for an additive trend, `*` for a multiplicative one;
//! `op_d` is `*` / `^`), `s = s_{t-m}` the seasonal in force and `f` the
//! one-step forecast (`q + s`, `q * s`, or `q`):
//!
//! ```text
//! p   = y_t - s            (additive seasonal) | y_t / s   (multiplicative) | y_t
//! l_t = q + alpha (p - q)
//! b_t = phi b_{t-1} + beta (p - q)                       (additive trend)
//!     = b_{t-1}^phi + beta (p - q) / l_{t-1}             (multiplicative trend)
//! s_t = s + gamma ((y_t - q) - s)                         (additive seasonal)
//!     = s + gamma (y_t / q - s)                           (multiplicative seasonal)
//! e_t = y_t - f            (additive error) | (y_t - f) / f  (multiplicative)
//! ```
//!
//! The state update is the same for both error types — only the residual
//! and the likelihood differ (Hyndman et al. 2008, section 2.5). The
//! log-likelihood is the Gaussian one with `sigma^2` concentrated out,
//! `-(n/2) (ln(2 pi sigma^2) + 1) - sum_t ln|f_t| [multiplicative error]`,
//! `sigma^2 = (1/n) sum e_t^2` — statsmodels' `ETSModel.loglike`; R's
//! `ets` reports the same quantity up to the additive constant
//! `-(n/2)(ln(2 pi / n) + 1)` it omits.

use crate::error::EtsError;
use crate::spec::{Component, ErrorType, EtsParams, EtsSpec, EtsStates};

/// Below this magnitude a state a multiplicative component divides by is
/// treated as zero (R's `etscalc.c` `TOL`).
pub(crate) const STATE_TOL: f64 = 1.0e-10;

/// The output of one recursion pass at fixed parameters and initial
/// states.
#[derive(Debug, Clone, PartialEq)]
pub struct Smoothed {
    /// One-step-ahead fitted values `f_t`.
    pub fitted: Vec<f64>,
    /// Residuals: `y_t - f_t` (additive error) or `(y_t - f_t) / f_t`
    /// (multiplicative error).
    pub resid: Vec<f64>,
    /// Level path `l_t` after the update at `t`.
    pub level: Vec<f64>,
    /// Trend path `b_t` (`None` without a trend).
    pub trend: Option<Vec<f64>>,
    /// Seasonal path `s_t` — the state updated at `t` (`None` without a
    /// seasonal).
    pub seasonal: Option<Vec<f64>>,
    /// The state after the last observation, seasonal states in time
    /// order for the forecast steps (see [`EtsStates`]).
    pub final_state: EtsStates,
    /// The concentrated Gaussian log-likelihood.
    pub loglik: f64,
    /// `sigma^2 = mean(e_t^2)`.
    pub sigma2: f64,
}

/// Sums needed for the likelihood alone.
#[derive(Debug, Clone, Copy)]
pub(crate) struct LikSums {
    pub sse: f64,
    pub sum_log_f: f64,
}

/// The trend-extended level `q` and the (damped) trend term `phib` from
/// the previous state — the deterministic part of the one-step forecast.
#[inline]
pub(crate) fn extend(spec: &EtsSpec, phi: f64, l: f64, b: f64) -> (f64, f64) {
    match spec.trend {
        Component::None => (l, 0.0),
        Component::Additive => {
            let phib = phi * b;
            (l + phib, phib)
        }
        Component::Multiplicative => {
            let phib = if spec.damped { b.powf(phi) } else { b };
            (l * phib, phib)
        }
    }
}

/// The one-step forecast `f` from `q` and the seasonal in force.
#[inline]
pub(crate) fn one_step(spec: &EtsSpec, q: f64, s: f64) -> f64 {
    match spec.seasonal {
        Component::None => q,
        Component::Additive => q + s,
        Component::Multiplicative => q * s,
    }
}

/// The state update given the observation `yt` (Hyndman et al. 2008,
/// Tables 2.2/2.3; R `etscalc.c` `update`): returns `(l_t, b_t, s_t)`.
/// `Err(what)` marks a division by a ~zero state.
#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) fn update(
    spec: &EtsSpec,
    alpha: f64,
    beta: f64,
    gamma: f64,
    l: f64,
    b: f64,
    q: f64,
    phib: f64,
    s: f64,
    yt: f64,
) -> Result<(f64, f64, f64), &'static str> {
    let p = match spec.seasonal {
        Component::None => yt,
        Component::Additive => yt - s,
        Component::Multiplicative => {
            if s.abs() < STATE_TOL {
                return Err("a multiplicative seasonal state is zero");
            }
            yt / s
        }
    };
    let l_new = q + alpha * (p - q);
    let b_new = match spec.trend {
        Component::None => b,
        Component::Additive => phib + beta * (p - q),
        Component::Multiplicative => {
            if l.abs() < STATE_TOL {
                return Err("the level a multiplicative trend divides by is zero");
            }
            phib + beta * (p - q) / l
        }
    };
    let s_new = match spec.seasonal {
        Component::None => 0.0,
        Component::Additive => s + gamma * ((yt - q) - s),
        Component::Multiplicative => {
            if q.abs() < STATE_TOL {
                return Err(
                    "the trend-extended level a multiplicative seasonal divides by is zero",
                );
            }
            s + gamma * (yt / q - s)
        }
    };
    if !(l_new.is_finite() && b_new.is_finite() && s_new.is_finite()) {
        return Err("a state update overflowed");
    }
    Ok((l_new, b_new, s_new))
}

/// One recursion pass. Writes the per-period paths into `paths` when
/// given (each vector must have length `n`); returns the likelihood sums.
/// `Err((what, index))` marks a degenerate step (division by a ~zero state
/// or a non-finite fitted value).
#[allow(clippy::type_complexity)]
pub(crate) fn recurse(
    spec: &EtsSpec,
    y: &[f64],
    params: &EtsParams,
    init: &EtsStates,
    ring: &mut Vec<f64>,
    mut paths: Option<(&mut [f64], &mut [f64], &mut [f64], &mut [f64], &mut [f64])>,
) -> Result<(LikSums, EtsStates), (&'static str, usize)> {
    let (alpha, beta, gamma, phi) = params.effective();
    let m = spec.m();
    let mult_error = spec.error == ErrorType::Multiplicative;
    let mut l = init.level;
    let mut b = init.trend.unwrap_or(0.0);
    ring.clear();
    if let Some(s) = &init.seasonal {
        ring.extend_from_slice(s);
    }
    let has_s = spec.has_seasonal();
    let mut sse = 0.0;
    let mut sum_log_f = 0.0;
    for (t, &yt) in y.iter().enumerate() {
        let (q, phib) = extend(spec, phi, l, b);
        let idx = t % m;
        let s = if has_s { ring[idx] } else { 0.0 };
        if spec.seasonal == Component::Multiplicative && s.abs() < STATE_TOL {
            return Err(("a multiplicative seasonal state is zero", t));
        }
        let f = one_step(spec, q, s);
        if !f.is_finite() {
            return Err(("the fitted value is not finite", t));
        }
        let e = if mult_error {
            if f == 0.0 {
                return Err(("the fitted value is zero under multiplicative errors", t));
            }
            (yt - f) / f
        } else {
            yt - f
        };
        let (l_new, b_new, s_new) =
            update(spec, alpha, beta, gamma, l, b, q, phib, s, yt).map_err(|w| (w, t))?;
        if let Some((fit, res, lv, tr, se)) = paths.as_mut() {
            fit[t] = f;
            res[t] = e;
            lv[t] = l_new;
            tr[t] = b_new;
            se[t] = s_new;
        }
        l = l_new;
        b = b_new;
        if has_s {
            ring[idx] = s_new;
        }
        sse += e * e;
        if mult_error {
            sum_log_f += f.abs().ln();
        }
    }
    let n = y.len();
    let final_state = EtsStates {
        level: l,
        trend: if spec.has_trend() { Some(b) } else { None },
        seasonal: if has_s {
            // Time order for the forecast steps: step j uses ring[(n + j) % m].
            Some((0..m).map(|j| ring[(n + j) % m]).collect())
        } else {
            None
        },
    };
    Ok((LikSums { sse, sum_log_f }, final_state))
}

/// The concentrated Gaussian log-likelihood from the recursion sums;
/// `None` when the residual variance is not a positive finite number.
pub(crate) fn loglik_from_sums(n: usize, sums: LikSums, mult_error: bool) -> Option<f64> {
    let nf = n as f64;
    let sigma2 = sums.sse / nf;
    if !(sigma2.is_finite() && sigma2 > 0.0) {
        return None;
    }
    let mut ll = -0.5 * nf * ((2.0 * std::f64::consts::PI * sigma2).ln() + 1.0);
    if mult_error {
        ll -= sums.sum_log_f;
    }
    ll.is_finite().then_some(ll)
}

/// Validates the data for `spec`: non-empty, finite, and strictly
/// positive when any component is multiplicative.
pub(crate) fn check_data(spec: &EtsSpec, y: &[f64]) -> Result<(), EtsError> {
    if let Some((i, &v)) = y.iter().enumerate().find(|(_, v)| !v.is_finite()) {
        return Err(EtsError::NonFinite {
            what: "y",
            index: i,
            value: v,
        });
    }
    // A seasonal period the sample never completes is not identified, and it
    // costs m - 1 free initial states: refuse it BY NAME rather than let the
    // parameter count blow up and report the spec instead (audit round 14).
    if spec.seasonal.is_present() && spec.seasonal_periods > y.len() {
        return Err(EtsError::TooFewObservations {
            needed: spec.seasonal_periods,
            got: y.len(),
            what: format!(
                "seasonal_periods = {} (a seasonal period the sample never completes is \
                 not identified, and it costs m - 1 free initial states)",
                spec.seasonal_periods
            ),
        });
    }
    if spec.needs_positive_data() {
        if let Some((i, &v)) = y.iter().enumerate().find(|(_, v)| **v <= 0.0) {
            return Err(EtsError::NonPositiveData {
                index: i,
                value: v,
                spec: spec.name(),
            });
        }
    }
    Ok(())
}

/// Runs the ETS recursion on `y` at fixed smoothing parameters and initial
/// states — the analogue of statsmodels `ETSModel.smooth(params)` with a
/// known initialisation — returning the fitted values, residuals, state
/// paths, final state and the concentrated Gaussian log-likelihood.
///
/// # Errors
///
/// * [`EtsError::InvalidSpec`] for an inconsistent `spec`;
/// * [`EtsError::NonFinite`] / [`EtsError::NonPositiveData`] for bad data;
/// * [`EtsError::TooFewObservations`] when `y` is empty;
/// * [`EtsError::InvalidParameter`] / [`EtsError::DimensionMismatch`] for
///   parameters or states outside the model's domain;
/// * [`EtsError::Degenerate`] when the recursion divides by a zero state
///   or the residual variance is zero.
pub fn smooth(
    spec: &EtsSpec,
    y: &[f64],
    params: &EtsParams,
    init: &EtsStates,
) -> Result<Smoothed, EtsError> {
    spec.validate()?;
    check_data(spec, y)?;
    let n = y.len();
    if n == 0 {
        return Err(EtsError::TooFewObservations {
            needed: 1,
            got: 0,
            what: "running the ETS recursion".into(),
        });
    }
    params.check_domain()?;
    init.check_domain(spec)?;
    let mut fitted = vec![0.0; n];
    let mut resid = vec![0.0; n];
    let mut level = vec![0.0; n];
    let mut trend = vec![0.0; n];
    let mut seasonal = vec![0.0; n];
    let mut ring = Vec::with_capacity(spec.m());
    let (sums, final_state) = recurse(
        spec,
        y,
        params,
        init,
        &mut ring,
        Some((
            &mut fitted,
            &mut resid,
            &mut level,
            &mut trend,
            &mut seasonal,
        )),
    )
    .map_err(|(what, index)| EtsError::Degenerate { what, index })?;
    let mult_error = spec.error == ErrorType::Multiplicative;
    let loglik = loglik_from_sums(n, sums, mult_error).ok_or(EtsError::Degenerate {
        what: "the residual variance is zero or not finite, so the concentrated \
               likelihood is undefined",
        index: n,
    })?;
    Ok(Smoothed {
        fitted,
        resid,
        level,
        trend: spec.has_trend().then_some(trend),
        seasonal: spec.has_seasonal().then_some(seasonal),
        final_state,
        loglik,
        sigma2: sums.sse / n as f64,
    })
}

/// The concentrated Gaussian log-likelihood at fixed parameters and
/// initial states (statsmodels `ETSModel.loglike`).
///
/// # Errors
///
/// As for [`smooth`].
pub fn loglik(
    spec: &EtsSpec,
    y: &[f64],
    params: &EtsParams,
    init: &EtsStates,
) -> Result<f64, EtsError> {
    spec.validate()?;
    check_data(spec, y)?;
    if y.is_empty() {
        return Err(EtsError::TooFewObservations {
            needed: 1,
            got: 0,
            what: "evaluating the ETS likelihood".into(),
        });
    }
    params.check_domain()?;
    init.check_domain(spec)?;
    let mut ring = Vec::with_capacity(spec.m());
    let (sums, _) = recurse(spec, y, params, init, &mut ring, None)
        .map_err(|(what, index)| EtsError::Degenerate { what, index })?;
    loglik_from_sums(y.len(), sums, spec.error == ErrorType::Multiplicative).ok_or(
        EtsError::Degenerate {
            what: "the residual variance is zero or not finite, so the concentrated \
                   likelihood is undefined",
            index: y.len(),
        },
    )
}
