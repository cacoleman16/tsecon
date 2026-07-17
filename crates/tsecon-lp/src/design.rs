//! Shared helpers for assembling the per-horizon local-projection design.
//!
//! Every LP estimator in this crate builds, for a given horizon `h`, a
//! regression sample over the observations `t` for which both the lag
//! controls (`y_{t-1}, ..., y_{t-p}`) and the shifted outcome (`y_{t+h}`, or
//! the cumulated outcome) exist. These helpers centralise that index
//! bookkeeping so the level, IV, cumulative, and state-dependent paths agree
//! on exactly which rows enter each regression.

/// Non-finite guard shared by all entry points.
pub(crate) fn check_finite(x: &[f64], what: &'static str) -> Result<(), crate::LpError> {
    for (i, &v) in x.iter().enumerate() {
        if !v.is_finite() {
            return Err(crate::LpError::NonFinite {
                what,
                index: i,
                value: v,
            });
        }
    }
    Ok(())
}

/// The first usable observation index and the sample length for horizon `h`.
///
/// `start = max(n_lag_controls, n_shock_lags)` (need both blocks of lags),
/// and the sample runs `t = start ..= n - 1 - h` so that `y_{t+h}` exists.
/// Returns `(start, nobs)`; `nobs` is zero when the horizon exhausts the
/// series.
pub(crate) fn horizon_sample(
    n: usize,
    h: usize,
    n_lag_controls: usize,
    n_shock_lags: usize,
) -> (usize, usize) {
    let start = n_lag_controls.max(n_shock_lags);
    if n <= start + h {
        return (start, 0);
    }
    (start, n - h - start)
}

/// The outcome column for horizon `h` over `t in [start, start + nobs)`.
///
/// `cumulative = false` yields `y_{t+h}` (level response); `true` yields the
/// running sum `sum_{j=0}^{h} y_{t+j}` (Ramey-Zubairy cumulative response).
pub(crate) fn outcome_column(
    y: &[f64],
    h: usize,
    start: usize,
    nobs: usize,
    cumulative: bool,
) -> Vec<f64> {
    (start..start + nobs)
        .map(|t| {
            if cumulative {
                (0..=h).map(|j| y[t + j]).sum()
            } else {
                y[t + h]
            }
        })
        .collect()
}

/// A lag column `x_{t-lag}` over `t in [start, start + nobs)`.
pub(crate) fn lag_column(x: &[f64], lag: usize, start: usize, nobs: usize) -> Vec<f64> {
    (start..start + nobs).map(|t| x[t - lag]).collect()
}

/// The contemporaneous column `x_t` over `t in [start, start + nobs)`.
pub(crate) fn contemporaneous_column(x: &[f64], start: usize, nobs: usize) -> Vec<f64> {
    x[start..start + nobs].to_vec()
}

/// A constant (intercept) column of length `nobs`.
pub(crate) fn const_column(nobs: usize) -> Vec<f64> {
    vec![1.0; nobs]
}

/// Assemble the standard single-impulse design columns in the fixture order
/// `[shock_t, const, y_{t-1..t-p}, shock_{t-1..t-q}]`.
///
/// `n_shock_lags = q` is `0` for the plain HAC path and `h` for the
/// lag-augmented path. The impulse coefficient is always column 0.
pub(crate) fn single_impulse_design(
    y: &[f64],
    shock: &[f64],
    h: usize,
    start: usize,
    nobs: usize,
    n_lag_controls: usize,
    n_shock_lags: usize,
) -> Vec<Vec<f64>> {
    let mut cols = Vec::with_capacity(2 + n_lag_controls + n_shock_lags);
    cols.push(contemporaneous_column(shock, start, nobs));
    cols.push(const_column(nobs));
    for lag in 1..=n_lag_controls {
        cols.push(lag_column(y, lag, start, nobs));
    }
    for lag in 1..=n_shock_lags {
        cols.push(lag_column(shock, lag, start, nobs));
    }
    let _ = h;
    cols
}
