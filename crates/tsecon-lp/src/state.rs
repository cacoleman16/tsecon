//! State-dependent (interacted) local projections: [`lp_state`].

use tsecon_hac::{ols, Kernel, SeType};

use crate::design::{check_finite, horizon_sample, outcome_column};
use crate::error::LpError;
use crate::spec::{LpSpec, LpStateResult, SeKind, SeSpec};

/// Estimate a two-regime, state-dependent local-projection response.
///
/// Following Ramey & Zubairy (2018), the impulse and every control are
/// interacted with the **lagged** state indicator `I_{t-1}` and its
/// complement `1 - I_{t-1}` (lagged so the regime is predetermined and not
/// itself moved by the shock). The horizon-`h` regression is
///
/// ```text
///   y_{t+h} =   I_{t-1} [ b1_h shock_t + c1 + sum_l a1_l y_{t-l} ]
///           + (1-I_{t-1})[ b0_h shock_t + c0 + sum_l a0_l y_{t-l} ] + u
/// ```
///
/// and reports `b1_h` (state 1) and `b0_h` (state 0) with their standard
/// errors. Inference follows `spec.se`: lag-augmented HC1 (with the impulse
/// lags interacted per regime) or Newey-West HAC.
///
/// # Errors
///
/// [`LpError::LengthMismatch`] on unequal input lengths,
/// [`LpError::NonFinite`] on NaN/inf input, [`LpError::NonBinaryState`] if
/// the indicator is not 0/1, [`LpError::DegenerateState`] when a regime is
/// (nearly) empty, [`LpError::HorizonTooLong`] / [`LpError::SeriesTooShort`]
/// on an exhausted sample, and [`LpError::Hac`] from the OLS/HAC engine.
pub fn lp_state(
    y: &[f64],
    shock: &[f64],
    state_indicator: &[f64],
    spec: LpSpec,
) -> Result<LpStateResult, LpError> {
    let n = y.len();
    if shock.len() != n {
        return Err(LpError::LengthMismatch {
            what: "impulse (shock) vs outcome (y)",
            expected: n,
            got: shock.len(),
        });
    }
    if state_indicator.len() != n {
        return Err(LpError::LengthMismatch {
            what: "state indicator vs outcome (y)",
            expected: n,
            got: state_indicator.len(),
        });
    }
    check_finite(y, "outcome (y)")?;
    check_finite(shock, "impulse (shock)")?;
    for (i, &v) in state_indicator.iter().enumerate() {
        if v != 0.0 && v != 1.0 {
            return Err(LpError::NonBinaryState { index: i, value: v });
        }
    }

    let p = spec.n_lag_controls;
    if n <= p {
        return Err(LpError::SeriesTooShort {
            n,
            n_lag_controls: p,
        });
    }

    let se_kind = match spec.se {
        SeSpec::LagAugmented => SeKind::LagAugmentedHc1,
        SeSpec::Hac { .. } => SeKind::HacBartlett,
    };

    let mut horizons = Vec::with_capacity(spec.horizons + 1);
    let mut irf_state1 = Vec::with_capacity(spec.horizons + 1);
    let mut se_state1 = Vec::with_capacity(spec.horizons + 1);
    let mut irf_state0 = Vec::with_capacity(spec.horizons + 1);
    let mut se_state0 = Vec::with_capacity(spec.horizons + 1);
    let mut nobs_per_h = Vec::with_capacity(spec.horizons + 1);

    for h in 0..=spec.horizons {
        let n_shock_lags = match spec.se {
            SeSpec::LagAugmented => h,
            SeSpec::Hac { .. } => 0,
        };
        // Need y_{t-p}, shock_{t-q}, and the lagged indicator I_{t-1}.
        let min_lag = p.max(n_shock_lags).max(1);
        let (start, nobs) = horizon_sample(n, h, min_lag, 0);
        // Two interacted blocks: [shock, const, p y-lags, q shock-lags] x 2.
        let block = 2 + p + n_shock_lags;
        let nparams = 2 * block;
        if nobs <= nparams {
            return Err(LpError::HorizonTooLong {
                horizon: h,
                nobs,
                nparams,
            });
        }

        // Lagged indicator over the sample and a degeneracy guard.
        let d: Vec<f64> = (start..start + nobs)
            .map(|t| state_indicator[t - 1])
            .collect();
        let n1 = d.iter().filter(|&&v| v == 1.0).count();
        let n0 = nobs - n1;
        if n1 <= block || n0 <= block {
            return Err(LpError::DegenerateState {
                horizon: h,
                regime_nobs: n1.min(n0),
            });
        }

        let response = outcome_column(y, h, start, nobs, spec.cumulative);
        let cols = interacted_design(y, shock, &d, h, start, nobs, p, n_shock_lags);

        let fit = ols(&response, &cols)?;
        let se_type = match spec.se {
            SeSpec::LagAugmented => SeType::Hc1,
            SeSpec::Hac { maxlags } => SeType::Hac {
                kernel: Kernel::Bartlett,
                bandwidth: maxlags.unwrap_or(h + p) as f64,
                use_correction: true,
            },
        };
        let inf = fit.inference(se_type)?;

        // Column order: 0 = state1 impulse, 1 = state0 impulse.
        horizons.push(h);
        irf_state1.push(fit.params[0]);
        se_state1.push(inf.bse[0]);
        irf_state0.push(fit.params[1]);
        se_state0.push(inf.bse[1]);
        nobs_per_h.push(nobs);
    }

    Ok(LpStateResult {
        horizons,
        irf_state1,
        se_state1,
        irf_state0,
        se_state0,
        nobs_per_h,
        se_kind,
    })
}

/// Build the fully-interacted design. Column order:
/// `[d*shock, (1-d)*shock, d, (1-d), d*y_lag.., (1-d)*y_lag.., d*shock_lag..,
/// (1-d)*shock_lag..]`, so column 0 is the state-1 impulse and column 1 the
/// state-0 impulse.
#[allow(clippy::too_many_arguments)]
fn interacted_design(
    y: &[f64],
    shock: &[f64],
    d: &[f64],
    h: usize,
    start: usize,
    nobs: usize,
    p: usize,
    n_shock_lags: usize,
) -> Vec<Vec<f64>> {
    let idx = || start..start + nobs;
    let mut cols: Vec<Vec<f64>> = Vec::with_capacity(2 * (2 + p + n_shock_lags));

    // Interacted impulse (state 1 then state 0).
    cols.push(idx().enumerate().map(|(k, t)| d[k] * shock[t]).collect());
    cols.push(
        idx()
            .enumerate()
            .map(|(k, t)| (1.0 - d[k]) * shock[t])
            .collect(),
    );
    // Regime intercepts.
    cols.push(d.to_vec());
    cols.push(d.iter().map(|&v| 1.0 - v).collect());
    // Interacted y-lag controls.
    for lag in 1..=p {
        cols.push(idx().enumerate().map(|(k, t)| d[k] * y[t - lag]).collect());
        cols.push(
            idx()
                .enumerate()
                .map(|(k, t)| (1.0 - d[k]) * y[t - lag])
                .collect(),
        );
    }
    // Interacted impulse-lag augmentation.
    for lag in 1..=n_shock_lags {
        cols.push(
            idx()
                .enumerate()
                .map(|(k, t)| d[k] * shock[t - lag])
                .collect(),
        );
        cols.push(
            idx()
                .enumerate()
                .map(|(k, t)| (1.0 - d[k]) * shock[t - lag])
                .collect(),
        );
    }
    let _ = h;
    cols
}
