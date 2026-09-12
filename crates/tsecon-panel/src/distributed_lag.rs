//! Distributed-lag panel regressions — the climate-impact specification.
//!
//! The workhorse of the climate-economy literature (Dell, Jones & Olken
//! 2012, AEJ:Macro; Burke, Hsiang & Miguel 2015, Nature; Hsiang 2016,
//! Annual Review of Resource Economics) regresses an outcome on `L` lags
//! of a weather regressor with entity and time effects:
//!
//! ```text
//! y_it = sum_{l=0..L} beta_l x_{i,t-l}  [+ sum_{l=0..L} gamma_l x^2_{i,t-l}]
//!        + alpha_i + delta_t [+ g_i * t] + e_it
//! ```
//!
//! and reads two objects off the fit:
//!
//! * the **cumulative effect** `B = sum_l beta_l` — the long-run impact of
//!   a permanent one-unit change in `x` once `L` periods have elapsed —
//!   with the delta-method standard error `sqrt(1' V_beta 1)`, `V_beta`
//!   the `(L+1) x (L+1)` covariance block of the lag coefficients;
//! * for the quadratic response (`powers = 2`), the **marginal effect of
//!   the cumulative response** at a point `x`, `B_1 + 2 B_2 x` with
//!   `B_p = sum_l` of the power-`p` lag coefficients, and its **turning
//!   point** `x* = -B_1 / (2 B_2)` (the temperature at which the
//!   cumulative response peaks — the BHM "optimum"), both with
//!   delta-method standard errors from the joint `2(L+1) x 2(L+1)`
//!   covariance block: the gradient of `x*` is `-1/(2 B_2)` in every
//!   power-1 lag and `B_1 / (2 B_2^2)` in every power-2 lag.
//!
//! The lag design drops the first `L` periods of every entity
//! (`n_periods_used = T - L`), then delegates to [`panel_ols_fe_with`] —
//! the within transformation, OLS, and the nonrobust / entity-clustered /
//! Driscoll-Kraay covariances are the ones `fixtures/panel.json` and
//! `fixtures/panel_dl.json` pin against linearmodels `PanelOLS`; nothing
//! here re-implements them. On an unbalanced panel
//! ([`PanelData::unbalanced`]) a lagged row `(i, t)` is used only when
//! entity `i` is observed in every period `t - L ..= t`, exactly as a
//! within-entity `shift` followed by dropping incomplete rows does; the
//! observed lagged cells become the mask of the lagged panel
//! (`fixtures/panel_unbalanced.json` pins this against `PanelOLS` on the
//! Arellano-Bond `EmplUK` panel and on a seeded ragged panel).
//!
//! ## What the specification assumes (read before use)
//!
//! * **Strict exogeneity of the regressor.** Weather is the canonical
//!   case; the design has no lagged dependent variable, and adding one
//!   through `regressors` would put Nickell (1981) bias back into a
//!   short-`T` within estimator — use `panel_lp` with a bias correction
//!   for dynamic panels.
//! * **Missing cells are declared, never guessed.** A NaN in an observed
//!   cell is refused; an unbalanced panel says which cells are missing
//!   through the observation mask, and every lag of a row must be
//!   observed for the row to enter. Entities entering late or leaving
//!   early cost nothing but rows; internal gaps cost the `L` rows after
//!   each gap.
//! * **Inference.** Entity clustering (the DJO default) needs many
//!   entities and ignores cross-sectional dependence; Driscoll-Kraay
//!   (the BHM robustness choice) is robust to it but needs a long `T`.
//!   The 95% intervals use the normal critical value 1.959964; the
//!   measured coverage of the cumulative-effect interval is reported on
//!   the panel model card.

use tsecon_linalg::faer::Mat;

use crate::data::PanelData;
use crate::error::PanelError;
use crate::fe::{panel_ols_fe_with, FixedEffects, PanelSeType};

/// `Phi^{-1}(0.975)`: the normal critical value of the 95% intervals.
const Z_975: f64 = 1.959_963_984_540_054;

/// Configuration for [`panel_distributed_lag`].
#[derive(Debug, Clone, PartialEq)]
pub struct DistributedLagConfig {
    /// Number of lags `L`; lags `0..=L` of every regressor enter the
    /// design and the first `L` periods of each entity are dropped.
    pub lags: usize,
    /// `1` for a linear response, `2` to add the lags of `x^2` (the
    /// BHM quadratic response). Nothing else is accepted.
    pub powers: usize,
    /// Which fixed effects to sweep out (default: entity and time).
    pub effects: FixedEffects,
    /// Covariance estimator for the standard errors.
    pub se_type: PanelSeType,
    /// Points at which the marginal effect of the cumulative quadratic
    /// response is evaluated (every regressor is evaluated at every
    /// point). `None` under `powers = 2` evaluates at each regressor's
    /// pooled sample mean; `Some` under `powers = 1` is refused — the
    /// linear cumulative response has one constant marginal effect,
    /// which is `cumulative_effect` itself.
    pub eval_points: Option<Vec<f64>>,
}

impl DistributedLagConfig {
    /// `lags` lags of a linear response with entity and time effects.
    #[must_use]
    pub fn new(lags: usize, se_type: PanelSeType) -> Self {
        Self {
            lags,
            powers: 1,
            effects: FixedEffects::TWO_WAY,
            se_type,
            eval_points: None,
        }
    }
}

/// A fitted distributed-lag panel regression; produced by
/// [`panel_distributed_lag`]. Design columns are ordered regressor-major,
/// then power, then lag: column `(j * powers + (p - 1)) * (L + 1) + l`
/// holds `x_j^p` at lag `l`. The per-regressor containers below are
/// indexed `[regressor][power - 1][lag]` (lag-level) or
/// `[regressor][power - 1]` (cumulative).
#[derive(Debug, Clone)]
pub struct DistributedLagResult {
    /// Slope estimates in design-column order.
    pub params: Vec<f64>,
    /// Column names `"{name}_L{lag}"` / `"{name}^2_L{lag}"`, aligned with
    /// `params`.
    pub names: Vec<String>,
    /// Standard errors under `se_type`.
    pub bse: Vec<f64>,
    /// t-statistics `params / bse`.
    pub tvalues: Vec<f64>,
    /// Full parameter covariance, `K x K`.
    pub cov: Mat<f64>,
    /// `lag_effects[j][p-1][l]`: the coefficient on `x_j^p` at lag `l`.
    pub lag_effects: Vec<Vec<Vec<f64>>>,
    /// Standard errors aligned with `lag_effects`.
    pub lag_se: Vec<Vec<Vec<f64>>>,
    /// `cumulative_effect[j][p-1] = sum_l lag_effects[j][p-1][l]`.
    pub cumulative_effect: Vec<Vec<f64>>,
    /// Delta-method standard error `sqrt(1' V 1)` of the cumulative
    /// effect.
    pub cumulative_se: Vec<Vec<f64>>,
    /// `cumulative_effect - 1.959964 * cumulative_se`.
    pub cumulative_ci_low: Vec<Vec<f64>>,
    /// `cumulative_effect + 1.959964 * cumulative_se`.
    pub cumulative_ci_high: Vec<Vec<f64>>,
    /// `powers = 2` only: `eval_points[j]` — the points at which
    /// regressor `j`'s marginal effect was evaluated.
    pub eval_points: Option<Vec<Vec<f64>>>,
    /// `powers = 2` only: `marginal_effect[j][q] = B_1 + 2 B_2 x` at
    /// `x = eval_points[j][q]`.
    pub marginal_effect: Option<Vec<Vec<f64>>>,
    /// Delta-method standard errors aligned with `marginal_effect`.
    pub marginal_se: Option<Vec<Vec<f64>>>,
    /// `powers = 2` only: `turning_point[j] = -B_1 / (2 B_2)` (NaN when
    /// `B_2` is exactly zero).
    pub turning_point: Option<Vec<f64>>,
    /// Delta-method standard errors of `turning_point`.
    pub turning_point_se: Option<Vec<f64>>,
    /// Stacked observations after dropping the lag window: `N * (T - L)`
    /// on a balanced panel, the lagged rows whose every lag is observed on
    /// an unbalanced one.
    pub nobs: usize,
    /// Number of entities `N`.
    pub n_entities: usize,
    /// Periods per entity after dropping the lag window, `T - L`.
    pub n_periods_used: usize,
    /// The lag order `L`.
    pub lags: usize,
    /// The response order (1 or 2).
    pub powers: usize,
    /// Residual degrees of freedom `nobs - K - n_absorbed`.
    pub df_resid: usize,
    /// The covariance estimator used.
    pub se_type: PanelSeType,
    /// The fixed effects swept out.
    pub effects: FixedEffects,
}

/// Fits the distributed-lag panel regression of the module docs.
///
/// # Errors
///
/// * [`PanelError::InvalidArgument`] if the panel has no regressors,
///   `powers` is not 1 or 2, the effects menu is not a panel model
///   ([`FixedEffects::validate`]), `eval_points` is passed under
///   `powers = 1`, or an empty `eval_points` is passed;
/// * [`PanelError::NonFinite`] for a non-finite evaluation point;
/// * [`PanelError::InsufficientObservations`] if fewer than two periods
///   remain after dropping the `L` lag periods, or (unbalanced panel) no
///   entity has `L + 1` consecutive observed periods;
/// * every error of [`panel_ols_fe_with`] on the lagged design
///   (degrees of freedom, absorbed or collinear columns, bandwidth).
pub fn panel_distributed_lag(
    data: &PanelData,
    cfg: &DistributedLagConfig,
) -> Result<DistributedLagResult, PanelError> {
    let k = data.n_regressors();
    if k == 0 {
        return Err(PanelError::InvalidArgument {
            what: "the panel has no regressors; a distributed-lag regression needs \
                   at least one regressor to lag",
        });
    }
    if cfg.powers != 1 && cfg.powers != 2 {
        return Err(PanelError::InvalidArgument {
            what: "powers must be 1 (linear response: lags of x) or 2 (quadratic \
                   response: lags of x and of x^2, the Burke-Hsiang-Miguel form); \
                   higher polynomial orders are not offered",
        });
    }
    cfg.effects.validate()?;
    if cfg.powers == 1 && cfg.eval_points.is_some() {
        return Err(PanelError::InvalidArgument {
            what: "eval_points was given but powers=1 ignores it: the marginal \
                   effect of a linear cumulative response is the constant \
                   cumulative_effect itself, so there is nothing to evaluate — \
                   pass powers=2 for the quadratic response or drop eval_points",
        });
    }
    if let Some(pts) = &cfg.eval_points {
        if pts.is_empty() {
            return Err(PanelError::InvalidArgument {
                what: "eval_points is empty; pass at least one point at which to \
                       evaluate the marginal effect, or omit it to use each \
                       regressor's sample mean",
            });
        }
        if pts.iter().any(|v| !v.is_finite()) {
            return Err(PanelError::NonFinite {
                what: "eval_points",
            });
        }
    }
    let (n_ent, t_len) = (data.n_entities(), data.n_periods());
    let lags = cfg.lags;
    // Saturating: `lags` near `usize::MAX` must refuse, not overflow.
    if lags > t_len.saturating_sub(2) {
        return Err(PanelError::InsufficientObservations {
            what: "distributed-lag design (lags): dropping the first `lags` periods of \
                   every entity must leave at least two periods",
            needed: lags.saturating_add(2),
            got: t_len,
        });
    }
    let t_used = t_len - lags;
    let powers = cfg.powers;
    let n_lag_cols = lags + 1;
    let idx = |j: usize, p: usize, l: usize| (j * powers + (p - 1)) * n_lag_cols + l;

    // Lagged, balanced design: y[:, L..] on x_j^p[:, L-l .. T-l].
    let outcome = data.outcome();
    let y = Mat::from_fn(n_ent, t_used, |i, t| outcome[(i, t + lags)]);
    let mut regs: Vec<(String, Mat<f64>)> = Vec::with_capacity(k * powers * n_lag_cols);
    for j in 0..k {
        let x = data.regressor(j).ok_or(PanelError::Dimension {
            what: "regressor index",
            expected: k,
            got: j,
        })?;
        let name = &data.names()[j];
        for p in 1..=powers {
            for l in 0..=lags {
                let m = Mat::from_fn(n_ent, t_used, |i, t| {
                    let v = x[(i, t + lags - l)];
                    if p == 1 {
                        v
                    } else {
                        v * v
                    }
                });
                let col_name = if p == 1 {
                    format!("{name}_L{l}")
                } else {
                    format!("{name}^2_L{l}")
                };
                regs.push((col_name, m));
            }
        }
    }
    let lagged = if data.is_balanced() {
        PanelData::balanced(y, regs)?
    } else {
        // A lagged row needs every cell t - L ..= t of its entity; the
        // observed lagged cells are the mask of the lagged panel.
        let lag_mask: Vec<Vec<bool>> = (0..n_ent)
            .map(|i| {
                (0..t_used)
                    .map(|t| (0..=lags).all(|l| data.observed(i, t + lags - l)))
                    .collect()
            })
            .collect();
        if !lag_mask.iter().flatten().any(|&m| m) {
            let longest_run = (0..n_ent)
                .map(|i| {
                    let mut best = 0;
                    let mut run = 0;
                    for t in 0..t_len {
                        run = if data.observed(i, t) { run + 1 } else { 0 };
                        best = best.max(run);
                    }
                    best
                })
                .max()
                .unwrap_or(0);
            return Err(PanelError::InsufficientObservations {
                what: "distributed-lag design on the unbalanced panel (mask): no entity \
                       is observed in lags + 1 consecutive periods, so no lagged row \
                       can be formed — reduce lags or fill the gaps (counts are \
                       consecutive observed periods, the longest run in the mask)",
                needed: lags + 1,
                got: longest_run,
            });
        }
        PanelData::unbalanced(y, regs, &lag_mask)?
    };
    let fit = panel_ols_fe_with(&lagged, cfg.effects)?;
    let inf = fit.inference(cfg.se_type)?;
    let cov = inf.cov;
    let params = fit.params;

    // Per-lag and cumulative effects with the delta method.
    let mut lag_effects = vec![vec![vec![0.0_f64; n_lag_cols]; powers]; k];
    let mut lag_se = lag_effects.clone();
    let mut cumulative_effect = vec![vec![0.0_f64; powers]; k];
    let mut cumulative_se = cumulative_effect.clone();
    for j in 0..k {
        for p in 1..=powers {
            let mut sum = 0.0;
            let mut var = 0.0;
            for l in 0..=lags {
                let a = idx(j, p, l);
                lag_effects[j][p - 1][l] = params[a];
                lag_se[j][p - 1][l] = inf.bse[a];
                sum += params[a];
                for m in 0..=lags {
                    var += cov[(a, idx(j, p, m))];
                }
            }
            cumulative_effect[j][p - 1] = sum;
            cumulative_se[j][p - 1] = var.max(0.0).sqrt();
        }
    }
    let cumulative_ci_low: Vec<Vec<f64>> = cumulative_effect
        .iter()
        .zip(&cumulative_se)
        .map(|(c, s)| c.iter().zip(s).map(|(c, s)| c - Z_975 * s).collect())
        .collect();
    let cumulative_ci_high: Vec<Vec<f64>> = cumulative_effect
        .iter()
        .zip(&cumulative_se)
        .map(|(c, s)| c.iter().zip(s).map(|(c, s)| c + Z_975 * s).collect())
        .collect();

    // Quadratic response: marginal effects and the turning point.
    let (mut eval_points, mut marginal_effect, mut marginal_se) = (None, None, None);
    let (mut turning_point, mut turning_point_se) = (None, None);
    if powers == 2 {
        let points: Vec<Vec<f64>> = match &cfg.eval_points {
            Some(pts) => vec![pts.clone(); k],
            None => (0..k)
                .map(|j| {
                    // Pooled sample mean of the raw (unlagged) regressor
                    // over the observed cells (every cell when balanced —
                    // the same loop, in the same order).
                    let x = data.regressor(j).map_or(0.0, |m| {
                        let mut s = 0.0;
                        let mut count = 0usize;
                        for t in 0..t_len {
                            for i in 0..n_ent {
                                if data.observed(i, t) {
                                    s += m[(i, t)];
                                    count += 1;
                                }
                            }
                        }
                        s / count as f64
                    });
                    vec![x]
                })
                .collect(),
        };
        let mut me = vec![Vec::new(); k];
        let mut mse = vec![Vec::new(); k];
        let mut tp = vec![0.0_f64; k];
        let mut tpse = vec![0.0_f64; k];
        let block = 2 * n_lag_cols;
        let mut grad = vec![0.0_f64; block];
        for j in 0..k {
            let b1 = cumulative_effect[j][0];
            let b2 = cumulative_effect[j][1];
            let base = idx(j, 1, 0);
            let quad_form = |g: &[f64]| {
                let mut v = 0.0;
                for a in 0..block {
                    for b in 0..block {
                        v += g[a] * cov[(base + a, base + b)] * g[b];
                    }
                }
                v.max(0.0).sqrt()
            };
            for &x in &points[j] {
                for l in 0..n_lag_cols {
                    grad[l] = 1.0;
                    grad[n_lag_cols + l] = 2.0 * x;
                }
                me[j].push(b1 + 2.0 * b2 * x);
                mse[j].push(quad_form(&grad));
            }
            if b2 == 0.0 {
                tp[j] = f64::NAN;
                tpse[j] = f64::NAN;
            } else {
                tp[j] = -b1 / (2.0 * b2);
                for l in 0..n_lag_cols {
                    grad[l] = -1.0 / (2.0 * b2);
                    grad[n_lag_cols + l] = b1 / (2.0 * b2 * b2);
                }
                tpse[j] = quad_form(&grad);
            }
        }
        eval_points = Some(points);
        marginal_effect = Some(me);
        marginal_se = Some(mse);
        turning_point = Some(tp);
        turning_point_se = Some(tpse);
    }

    Ok(DistributedLagResult {
        params,
        names: fit.names,
        bse: inf.bse,
        tvalues: inf.tvalues,
        cov,
        lag_effects,
        lag_se,
        cumulative_effect,
        cumulative_se,
        cumulative_ci_low,
        cumulative_ci_high,
        eval_points,
        marginal_effect,
        marginal_se,
        turning_point,
        turning_point_se,
        nobs: fit.nobs,
        n_entities: n_ent,
        n_periods_used: t_used,
        lags,
        powers,
        df_resid: fit.df_resid,
        se_type: cfg.se_type,
        effects: cfg.effects,
    })
}
