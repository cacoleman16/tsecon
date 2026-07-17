//! Panel local projections (Jordà 2005) with fixed effects.
//!
//! For each horizon `h`, a separate within (fixed-effects) regression of
//! the horizon-`h` outcome on a common observed shock plus lagged
//! controls:
//!
//! ```text
//! y_{i,t+h} = alpha_i^(h) + beta_h shock_t
//!           + sum_{l=1..Ls} gamma_l^(h) shock_{t-l}
//!           + sum_{l=1..Ly} delta_l^(h) y_{i,t-l} + e_{i,t+h}
//! ```
//!
//! `beta_h` is the impulse response at horizon `h` (Jordà 2005; the panel
//! variant is the workhorse of the Jordà-Schularick-Taylor macrohistory
//! literature). Standard errors come from the full [`PanelSeType`] menu;
//! Driscoll-Kraay is the natural default here because a common shock
//! induces cross-sectional dependence in the horizon-`h` errors by
//! construction.
//!
//! ## Cumulative multipliers (Ramey & Zubairy 2018)
//!
//! With [`PanelLpConfig::cumulative`] the regressand at horizon `h` is
//! the **cumulated outcome** `sum_{j=0..h} y_{i,t+j}`, estimated directly
//! per horizon. This is why the cumulative IRF is *not* produced by
//! cumulating the level point estimates: because the cumulated sum is
//! itself the regressand, `se[h]` is the correct standard error of the
//! cumulative response, whereas summing per-horizon point estimates would
//! require the full cross-horizon covariance of the level IRF (and
//! cumulating per-horizon standard errors is simply wrong — it ignores
//! the strong positive correlation of overlapping LP samples).
//!
//! ## Nickell bias — read this before trusting short-T dynamic panels
//!
//! The within transformation correlates the demeaned lagged outcome with
//! the demeaned error, so **fixed effects + lagged outcomes + short T
//! biases dynamic coefficients**: for an AR(1) panel the incidental-
//! parameter bias is approximately `-(1 + rho)/(T - 1)` (Nickell 1981) —
//! at `T = 20` and `rho = 0.5` that is a bias of about `-0.08`, and in
//! local projections the effect is horizon-amplified (roughly `O(h/T)`;
//! see Module 07 of the roadmap). It shrinks with the number of periods,
//! not the number of entities.
//!
//! [`PanelLpConfig::jackknife`] enables the half-panel jackknife of
//! Dhaene & Jochmans (2015): estimate on the full sample and on the two
//! time halves of the panel (each with its own fixed effects), then
//! combine
//!
//! ```text
//! theta_jack = 2 theta_full - (theta_half1 + theta_half2) / 2
//! ```
//!
//! which removes the O(1/T) bias term because each half-panel estimate
//! carries roughly twice the bias of the full-panel one. When `T` is odd
//! the two halves overlap by one period (each of length `(T+1)/2`), per
//! Dhaene-Jochmans. Standard errors are kept from the full-sample fit:
//! the jackknifed estimator has the same asymptotic variance as the
//! uncorrected one (Dhaene & Jochmans 2015, Theorem 3.1).
//!
//! // TODO(phase0): entity-varying shocks (an `N x T` impulse panel),
//! // user-supplied extra controls from `PanelData::regressor`, panel
//! // LP-IV, and the analytical Nickell corrections of Mei-Sheng-Shi.

use tsecon_linalg::faer::MatRef;

use crate::data::PanelData;
use crate::error::PanelError;
use crate::fe::{fit_within, PanelSeType, WithinFit};

/// Configuration for [`panel_lp`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PanelLpConfig {
    /// Maximum horizon `H`; responses are estimated for `h = 0..=H`.
    pub max_horizon: usize,
    /// Number of lagged-shock controls `shock_{t-1} .. shock_{t-Ls}`.
    pub shock_lags: usize,
    /// Number of lagged-outcome controls `y_{i,t-1} .. y_{i,t-Ly}`.
    /// Lagged outcomes soak up persistent noise but expose short-T
    /// panels to Nickell bias (see the module docs).
    pub outcome_lags: usize,
    /// Estimate the cumulated outcome `sum_{j<=h} y_{i,t+j}` per horizon
    /// (Ramey-Zubairy convention) instead of the level `y_{i,t+h}`.
    pub cumulative: bool,
    /// Apply the Dhaene-Jochmans (2015) half-panel jackknife bias
    /// correction to the point estimates (see the module docs).
    pub jackknife: bool,
    /// Covariance estimator for the per-horizon standard errors.
    pub cov: PanelSeType,
}

impl PanelLpConfig {
    /// A configuration with `n_lag_controls` lags of both the shock and
    /// the outcome as controls, level (non-cumulative) responses, and no
    /// jackknife. Adjust individual fields to taste.
    #[must_use]
    pub fn new(max_horizon: usize, n_lag_controls: usize, cov: PanelSeType) -> Self {
        Self {
            max_horizon,
            shock_lags: n_lag_controls,
            outcome_lags: n_lag_controls,
            cumulative: false,
            jackknife: false,
            cov,
        }
    }
}

/// Impulse responses from a panel local projection; produced by
/// [`panel_lp`]. All per-horizon vectors have `max_horizon + 1` entries,
/// indexed by horizon.
#[derive(Debug, Clone)]
pub struct PanelLpResult {
    /// Impulse response `beta_h` per horizon — the coefficient on
    /// `shock_t` (cumulated-outcome coefficient when `cumulative`; the
    /// jackknifed point estimate when `jackknife`).
    pub irf: Vec<f64>,
    /// Standard error of `irf[h]` under `se_type`, from the full-sample
    /// fit (unchanged by the jackknife; Dhaene & Jochmans 2015, Thm 3.1).
    pub se: Vec<f64>,
    /// Full per-horizon coefficient vectors, ordered
    /// `[shock_t, shock_{t-1..Ls}, y_{t-1..Ly}]` (jackknifed when
    /// `jackknife`).
    pub params: Vec<Vec<f64>>,
    /// Stacked observations used at each horizon (the sample shrinks as
    /// `h` grows).
    pub nobs: Vec<usize>,
    /// Covariance estimator used for `se`.
    pub se_type: PanelSeType,
    /// Whether the regressand was the cumulated outcome.
    pub cumulative: bool,
    /// Whether the half-panel jackknife was applied.
    pub jackknife: bool,
}

/// Estimates a panel local projection of the outcome in `data` on the
/// common shock series, horizon by horizon (see the module docs for the
/// regression, conventions, and the Nickell-bias warning).
///
/// `shock` must have `data.n_periods()` observations, aligned with the
/// outcome's periods. Regressors stored in `data` are currently ignored
/// (controls are generated internally from shock and outcome lags;
/// user-supplied controls are `// TODO(phase0)`).
///
/// # Errors
///
/// * [`PanelError::Dimension`] if `shock` is not `n_periods` long;
/// * [`PanelError::NonFinite`] if `shock` contains NaN/infinity;
/// * [`PanelError::InsufficientObservations`] /
///   [`PanelError::DegreesOfFreedom`] when a horizon (or a jackknife
///   half-panel) leaves too small a sample;
/// * [`PanelError::SingularDesign`] for collinear controls.
pub fn panel_lp(
    data: &PanelData,
    shock: &[f64],
    config: &PanelLpConfig,
) -> Result<PanelLpResult, PanelError> {
    let t_len = data.n_periods();
    if shock.len() != t_len {
        return Err(PanelError::Dimension {
            what: "shock series must be aligned with the panel's periods",
            expected: t_len,
            got: shock.len(),
        });
    }
    if shock.iter().any(|v| !v.is_finite()) {
        return Err(PanelError::NonFinite { what: "shock" });
    }

    let hmax = config.max_horizon;
    let mut irf = Vec::with_capacity(hmax + 1);
    let mut se = Vec::with_capacity(hmax + 1);
    let mut params = Vec::with_capacity(hmax + 1);
    let mut nobs = Vec::with_capacity(hmax + 1);

    // Half-panel windows for the jackknife: overlapping halves of length
    // ceil(T/2) when T is odd (Dhaene-Jochmans 2015).
    let half = t_len.div_ceil(2);

    for h in 0..=hmax {
        let full = lp_fit_window(data, shock, config, h, 0, t_len)?;
        let inference = full.inference(config.cov)?;
        let coefs = if config.jackknife {
            let first = lp_fit_window(data, shock, config, h, 0, half)?;
            let second = lp_fit_window(data, shock, config, h, t_len - half, t_len)?;
            full.params
                .iter()
                .zip(first.params.iter().zip(second.params.iter()))
                .map(|(&f, (&a, &b))| 2.0 * f - 0.5 * (a + b))
                .collect()
        } else {
            full.params.clone()
        };
        irf.push(coefs[0]);
        se.push(inference.bse[0]);
        nobs.push(full.nobs);
        params.push(coefs);
    }

    Ok(PanelLpResult {
        irf,
        se,
        params,
        nobs,
        se_type: config.cov,
        cumulative: config.cumulative,
        jackknife: config.jackknife,
    })
}

/// One per-horizon within regression restricted to the period window
/// `[w0, w1)`: regression index `t` runs over
/// `[w0 + max(Ls, Ly), w1 - h)` so every lag and lead stays inside the
/// window (the jackknife half-panels therefore never leak information
/// across the split).
fn lp_fit_window(
    data: &PanelData,
    shock: &[f64],
    config: &PanelLpConfig,
    h: usize,
    w0: usize,
    w1: usize,
) -> Result<WithinFit, PanelError> {
    let lag_max = config.shock_lags.max(config.outcome_lags);
    let t_start = w0 + lag_max;
    let t_end = w1.saturating_sub(h);
    let n_ent = data.n_entities();
    let k = 1 + config.shock_lags + config.outcome_lags;
    if t_end <= t_start {
        return Err(PanelError::InsufficientObservations {
            what: "panel local projection horizon window",
            needed: t_start + 1,
            got: t_end,
        });
    }
    let n_per = t_end - t_start;
    let n = n_ent * n_per;
    if n <= k + n_ent {
        return Err(PanelError::DegreesOfFreedom {
            n,
            k,
            n_entities: n_ent,
        });
    }

    let outcome: MatRef<'_, f64> = data.outcome();
    let mut y = vec![0.0_f64; n];
    let mut x_cols = vec![vec![0.0_f64; n]; k];
    for i in 0..n_ent {
        for (s, t) in (t_start..t_end).enumerate() {
            let r = i * n_per + s;
            y[r] = if config.cumulative {
                (0..=h).map(|j| outcome[(i, t + j)]).sum()
            } else {
                outcome[(i, t + h)]
            };
            x_cols[0][r] = shock[t];
            for l in 1..=config.shock_lags {
                x_cols[l][r] = shock[t - l];
            }
            for l in 1..=config.outcome_lags {
                x_cols[config.shock_lags + l][r] = outcome[(i, t - l)];
            }
        }
    }
    fit_within(&y, &x_cols, n_ent, n_per)
}
