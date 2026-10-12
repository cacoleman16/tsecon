//! Quantile local projections: the per-horizon check-loss analogue of the
//! Jordà LP, mirroring `tsecon-lp`'s design conventions exactly.
//!
//! For each horizon `h` the design over `t = p, ..., n - 1 - h` is, in
//! `tsecon-lp` column order,
//!
//! ```text
//! y_{t+h}  on  [shock_t, const, y_{t-1..t-p}, shock_{t-1..t-p}]
//! ```
//!
//! and the quantile-`tau` impulse response at horizon `h` is the check-loss
//! coefficient on `shock_t` (column 0), with the Powell kernel-sandwich
//! standard error from the shared core in [`crate::qreg`].

use crate::error::QuantileError;
use crate::qreg::{check_finite, fit_one, validate_taus};

/// Quantile local-projection impulse responses; produced by [`quantile_lp`].
#[derive(Debug, Clone, PartialEq)]
pub struct QuantileLp {
    /// The quantile levels, in the order they were passed.
    pub taus: Vec<f64>,
    /// The horizons `0..=max_horizon`.
    pub horizons: Vec<usize>,
    /// Impulse responses, indexed `irf[tau_index][horizon]`.
    pub irf: Vec<Vec<f64>>,
    /// Powell-sandwich standard errors, indexed `se[tau_index][horizon]`.
    pub se: Vec<Vec<f64>>,
    /// Per-fit IRLS convergence flags, indexed
    /// `converged[tau_index][horizon]` (same shape as `irf`/`se`). `false`
    /// means that (tau, horizon) fit hit the 1000-iteration IRLS cap
    /// before the max-abs coefficient change dropped to the shared `1e-6`
    /// tolerance: the reported coefficient is the last iterate, not a
    /// verified check-loss minimum (statsmodels `QuantReg` warns in the
    /// same situation; here it is reported per fit instead).
    pub converged: Vec<Vec<bool>>,
}

/// Quantile local projections of `y` on `shock` at each tau, for horizons
/// `0..=horizons`, controlling for a constant and `n_lag_controls` lags of
/// both `y` and `shock`.
///
/// Per (tau, horizon) this matches statsmodels `QuantReg` on the identical
/// numpy-assembled design (see `fixtures/generate_tsecon-quantile_fixtures.py`).
///
/// # Errors
///
/// The shared shape/finiteness/tau errors of
/// [`crate::quantile_regression`], plus
/// [`QuantileError::LagControlsExhaustSample`] when `n_lag_controls`
/// leaves even horizon 0 with no more usable observations than design
/// parameters, and [`QuantileError::HorizonsExhaustSample`] when
/// `horizons` does (both checked before anything is sized by
/// `horizons`).
pub fn quantile_lp(
    y: &[f64],
    shock: &[f64],
    taus: &[f64],
    horizons: usize,
    n_lag_controls: usize,
) -> Result<QuantileLp, QuantileError> {
    validate_taus(taus)?;
    if y.is_empty() {
        return Err(QuantileError::EmptyInput { what: "y" });
    }
    if shock.len() != y.len() {
        return Err(QuantileError::DimensionMismatch {
            what: "shock vs y",
            expected: y.len(),
            got: shock.len(),
        });
    }
    check_finite(y, "y")?;
    check_finite(shock, "shock")?;

    let n = y.len();
    let p = n_lag_controls;
    // Horizon h keeps nobs(h) = n - p - h rows for k = 2 + 2p parameters
    // and needs nobs > k. Both sides are checked here, in checked arithmetic
    // and before anything is sized by `horizons`: an `n_lag_controls` that
    // leaves even horizon 0 short is refused by that name, and the largest
    // usable horizon is n - p - k - 1.
    let k = match p.checked_mul(2).and_then(|v| v.checked_add(2)) {
        Some(k) if n > p && n - p > k => k,
        _ => {
            return Err(QuantileError::LagControlsExhaustSample {
                n_lag_controls: p,
                n,
            })
        }
    };
    let max_horizons = n - p - k - 1;
    if horizons > max_horizons {
        return Err(QuantileError::HorizonsExhaustSample {
            horizons,
            n,
            n_lag_controls: p,
            k,
            max_horizons,
        });
    }
    let mut irf = vec![Vec::with_capacity(horizons + 1); taus.len()];
    let mut se = vec![Vec::with_capacity(horizons + 1); taus.len()];
    let mut converged = vec![Vec::with_capacity(horizons + 1); taus.len()];
    for h in 0..=horizons {
        // Sample bookkeeping mirrors tsecon-lp::design::horizon_sample with
        // n_shock_lags = n_lag_controls: start = p, t runs to n - 1 - h.
        let start = p;
        let nobs = if n > start + h { n - h - start } else { 0 };
        if nobs <= k {
            return Err(QuantileError::HorizonExhaustsSample {
                horizon: h,
                n,
                nobs,
                k,
            });
        }
        let outcome: Vec<f64> = (start..start + nobs).map(|t| y[t + h]).collect();
        let mut cols: Vec<Vec<f64>> = Vec::with_capacity(k);
        cols.push(shock[start..start + nobs].to_vec());
        cols.push(vec![1.0; nobs]);
        for lag in 1..=p {
            cols.push((start..start + nobs).map(|t| y[t - lag]).collect());
        }
        for lag in 1..=p {
            cols.push((start..start + nobs).map(|t| shock[t - lag]).collect());
        }
        for (i, &tau) in taus.iter().enumerate() {
            let fit = fit_one(&outcome, &cols, tau)?;
            irf[i].push(fit.params[0]);
            se[i].push(fit.bse[0]);
            converged[i].push(fit.converged);
        }
    }
    Ok(QuantileLp {
        taus: taus.to_vec(),
        horizons: (0..=horizons).collect(),
        irf,
        se,
        converged,
    })
}
