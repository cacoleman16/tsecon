//! LP-DiD — local-projections difference-in-differences (Dube, Girardi,
//! Jordà & Taylor 2025, *Journal of Applied Econometrics* 40(7),
//! doi:10.1002/jae.70000; NBER WP 31184).
//!
//! For each post-treatment horizon `h = 0..=H` a separate cross-section
//! regression of the long difference on the treatment switch, with time
//! (period) fixed effects:
//!
//! ```text
//! y_{i,t+h} - y_{i,t-1} = beta_h ΔD_{it} + delta_t^h + e_{it}^h
//! ```
//!
//! estimated **only** on observations that are either newly treated
//! (`ΔD_{it} = 1`) or *clean controls*. The clean-control condition is
//! what makes this a DiD rather than a two-way-fixed-effects event study:
//! it removes the "forbidden comparisons" against previously-treated
//! units whose own dynamic effects contaminate TWFE event studies with
//! negative weights (Goodman-Bacon 2021; de Chaisemartin & D'Haultfœuille
//! 2020). Pre-treatment horizons `h = -Q..=-2` regress
//! `y_{i,t+h} - y_{i,t-1}` on the same switch indicator over the same
//! kind of clean sample and display pre-trends; `h = -1` is the omitted
//! baseline (identically zero by construction).
//!
//! ## Clean-control conditions (transcribed from the authors' code)
//!
//! * **Absorbing treatment** (`absorbing = true`; treatment never
//!   reverses): controls at post horizon `h` must satisfy
//!   `D_{i,t+h} = 0` — not yet treated through `t + h`; controls at pre
//!   horizons must satisfy `D_{it} = 0` — not yet treated at `t`.
//!   (`LP_DiD_R_example_VW.R` lines 136-150 of the authors' repository,
//!   github.com/danielegirardi/lpdid.)
//! * **Non-absorbing treatment** (`absorbing = false`; units may enter
//!   and exit): with a stabilization window `L =`
//!   [`LpDidConfig::nonabsorbing_lag`], an observation is clean at post
//!   horizon `h` if its treatment status did not change in
//!   `[t - L, t - 1]` nor in `[t + 1, t + h]`; at pre horizon `-j` the
//!   no-change window is `[t - L - (j - 1), t - 1]`. Newly treated
//!   observations must satisfy the same windows (so an entry is only an
//!   event after `L` quiet periods, and must persist through `t + h`);
//!   previously-treated units re-enter the control pool `L` periods
//!   after their last status change — the DGJT §3.2 effect-stabilization
//!   assumption. Changes outside the observed panel are treated as
//!   unobserved-and-clean, exactly like the reference's Stata missing
//!   semantics (`LPDiD_nonabsorbing_example.do` lines 191-212, variant
//!   5a/5b). Rows with `ΔD_{it} = -1` (an exit at `t`) are excluded
//!   from both groups.
//! * **`never_treated_only = true`** replaces the control-side condition
//!   with "the unit is never treated in the observed sample" (the Stata
//!   `lpdid` package's `nevertreated` option; the do-file's variant 4
//!   uses the related not-yet-treated pool).
//!
//! ## Weighting: variance-weighted vs equally-weighted ATT
//!
//! OLS on the clean sample yields a **variance-weighted** ATT: each
//! period-`t` clean 2×2 comparison enters with weight proportional to
//! `n_t p_t (1 - p_t)` where `p_t` is the switcher share in that
//! period's clean cell (DGJT §2.5 — all weights are non-negative, unlike
//! TWFE, but more-precisely-estimated cohorts count more). With
//! [`LpDidConfig::reweight`] each observation is weighted by the inverse
//! of its cell's switcher-residual share — the transcription of the
//! authors' `get_reweights` (`LP_DiD_R_example_EW.R` lines 138-166; also
//! `R/func.R` `get_weights` in the alexCardazzi/lpdid R port) — which
//! collapses, because the weight is constant within a period cell, to
//! weighting each cell's 2×2 by its **number of switchers**: the
//! **equally-weighted ATT** across treated observations. Period cells
//! with no switcher drop from the reweighted sample (their reference
//! weight is undefined/NA — fixest drops them); a cell with switchers
//! but no clean controls is refused (its equally-weighted contribution
//! is undefined — the reference produces an infinite weight there).
//!
//! ## Pooled ATT
//!
//! With [`LpDidConfig::pooled`] two additional single-number estimates
//! are reported (the event study is still estimated — unlike the R port,
//! where `pooled = TRUE` replaces it):
//!
//! * **post**: regressand `mean(y_{i,t..t+H}) - y_{i,t-1}` on the clean
//!   sample of horizon `H` (the most restrictive window) — the average
//!   effect over the post window;
//! * **pre** (needs `pre_window >= 2`): regressand
//!   `mean(y_{i,t-Q..t-2}) - y_{i,t-1}` on the pre-horizon clean sample
//!   — a single pooled pre-trend test.
//!
//! Under `reweight` the post pooled regression uses the horizon-`H`
//! weights and the pre pooled regression the horizon-0 weights,
//! following the authors' `LP_DiD_R_example_EW.R` (lines 267-284).
//!
//! ## Standard errors
//!
//! Cluster-robust by entity (the reference implementations' default and
//! only the entity dimension is exposed here), with the
//! fixest/reghdfe small-sample convention the authors' code fixes via
//! `setFixest_ssc(ssc(adj = TRUE, cluster.adj = TRUE))`:
//! `(n-1)/(n-K) * G/(G-1)` times the cluster sandwich, where `K` counts
//! the slope **plus every absorbed period effect** (time effects are not
//! nested in entity clusters, so they are counted — deliberately
//! different from the nested-cluster `n/(n-k)` convention of the
//! within-entity estimators in `fe.rs`, and pinned against a run of the
//! authors' fixest code in `fixtures/lpdid.json`). `G` is the number of
//! entities present in each horizon's clean sample.
//!
//! Weighted regressions use the WLS sandwich (scores `ω x̃ e`); the
//! estimates and standard errors are invariant to the overall scale of
//! the weights, so the reference's normalization constant (their `den`)
//! is irrelevant and not reproduced.
//!
//! ## Effective samples shrink — read `nobs` and `n_switchers`
//!
//! The clean-control condition and the horizon leads shrink the sample
//! as `|h|` grows, and the switching cohorts themselves thin out.
//! `nobs[h]` and `n_switchers[h]` report the actually-used rows and the
//! number of treatment switchers per horizon; a pre-trend "test" at a
//! horizon with a handful of switchers is noise, and the model card says
//! so.
//!
//! // TODO(phase0): covariates / regression-adjustment LP-DiD (DGJT
//! // §4.1.1), the composition-effects correction (DGJT §2.10, the
//! // do-file's variants 4a/5a), pre-mean-differenced baselines
//! // (`pmd`), outcome-difference lag controls (`dylags`), an IV
//! // variant, and unbalanced panels. Doubly-robust (AIPW) LP-DiD has
//! // no reference implementation anywhere and is explicitly out of
//! // scope until one exists to validate against.

use tsecon_linalg::faer::MatRef;

use crate::data::PanelData;
use crate::error::PanelError;

/// Configuration for [`lp_did`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LpDidConfig {
    /// Number of pre-treatment horizons `Q`: pre-trend coefficients are
    /// estimated at `h = -Q..=-2` (`h = -1` is the omitted baseline, so
    /// `Q <= 1` estimates no pre-trends).
    pub pre_window: usize,
    /// Number of post-treatment horizons `H`: effects are estimated at
    /// `h = 0..=H`.
    pub post_window: usize,
    /// Treatment is absorbing (never reverses). When `true`, a reversal
    /// in the treatment matrix is refused; when `false`,
    /// [`LpDidConfig::nonabsorbing_lag`] must be set.
    pub absorbing: bool,
    /// Stabilization window `L` for non-absorbing treatment: dynamic
    /// effects are assumed settled `L` periods after a status change, so
    /// units re-enter the control pool after `L` quiet periods (DGJT
    /// §3.2). Must be `>= 1` when `absorbing = false` and `0` when
    /// `absorbing = true`.
    pub nonabsorbing_lag: usize,
    /// Reweight to the equally-weighted ATT (DGJT §2.5) instead of the
    /// OLS variance-weighted ATT (see the module docs for the exact
    /// transcription).
    pub reweight: bool,
    /// Also estimate the pooled post ATT (average effect over
    /// `0..=post_window`) and, when `pre_window >= 2`, the pooled
    /// pre-trend.
    pub pooled: bool,
    /// Restrict the control pool to never-treated units (units with no
    /// treatment in the observed sample) instead of not-yet-treated /
    /// stabilized units.
    pub never_treated_only: bool,
}

impl LpDidConfig {
    /// Absorbing-treatment configuration with the given event window,
    /// variance-weighted, no pooled estimates, not-yet-treated controls.
    #[must_use]
    pub fn new(pre_window: usize, post_window: usize) -> Self {
        Self {
            pre_window,
            post_window,
            absorbing: true,
            nonabsorbing_lag: 0,
            reweight: false,
            pooled: false,
            never_treated_only: false,
        }
    }
}

/// One pooled ATT estimate from [`lp_did`] (see the module docs).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LpDidPooled {
    /// Pooled average treatment effect on the treated.
    pub att: f64,
    /// Cluster-by-entity standard error of `att`.
    pub se: f64,
    /// Rows used in the pooled regression.
    pub nobs: usize,
    /// Treatment switchers among them.
    pub n_switchers: usize,
}

/// Event-study estimates from [`lp_did`]. All per-horizon vectors are
/// aligned with `horizons`.
#[derive(Debug, Clone)]
pub struct LpDidResult {
    /// Event-time horizons, `-pre_window ..= post_window`. The `-1`
    /// entry is the omitted baseline: its coefficient, standard error,
    /// and sample sizes are stored as exact zeros (it is not estimated —
    /// the long differences are relative to `t - 1` by construction).
    pub horizons: Vec<i64>,
    /// Event-study coefficient per horizon (`beta_h`): the variance-
    /// weighted ATT, or the equally-weighted ATT under
    /// [`LpDidConfig::reweight`].
    pub coef: Vec<f64>,
    /// Cluster-by-entity standard error per horizon (fixest/reghdfe
    /// small-sample convention; see the module docs).
    pub se: Vec<f64>,
    /// Rows actually used per horizon — the clean-control sample
    /// shrinks as `|h|` grows.
    pub nobs: Vec<usize>,
    /// Treatment switchers (`ΔD = 1` rows) per horizon.
    pub n_switchers: Vec<usize>,
    /// Pooled post-window ATT, when [`LpDidConfig::pooled`].
    pub pooled_post: Option<LpDidPooled>,
    /// Pooled pre-trend estimate, when [`LpDidConfig::pooled`] and
    /// `pre_window >= 2`.
    pub pooled_pre: Option<LpDidPooled>,
    /// Stamped: absorbing-treatment mode.
    pub absorbing: bool,
    /// Stamped: stabilization window (0 in absorbing mode).
    pub nonabsorbing_lag: usize,
    /// Stamped: equally-weighted (`true`) vs variance-weighted ATT.
    pub reweight: bool,
    /// Stamped: never-treated-only control pool.
    pub never_treated_only: bool,
}

/// Which long difference a regression uses, and over which clean sample.
enum Spec {
    /// Post horizon `h`: `y_{t+h} - y_{t-1}`.
    Post(usize),
    /// Pre horizon `-j` (`j >= 2`): `y_{t-j} - y_{t-1}`.
    Pre(usize),
    /// Pooled post: `mean(y_{t..t+H}) - y_{t-1}` on the horizon-`H`
    /// clean sample.
    PooledPost,
    /// Pooled pre: `mean(y_{t-Q..t-2}) - y_{t-1}` on the pre clean
    /// sample.
    PooledPre,
}

/// Estimates LP-DiD event-study coefficients (and optionally pooled
/// ATTs) from a balanced panel outcome and a binary treatment matrix
/// (see the module docs for the estimator, the clean-control conditions,
/// and every transcribed convention).
///
/// `treatment` must be `n_entities x n_periods`, entries exactly 0 or 1.
/// Regressors stored in `data` are currently ignored (covariates are
/// `// TODO(phase0)`).
///
/// # Errors
///
/// * [`PanelError::Dimension`] if `treatment` is not `N x T`;
/// * [`PanelError::InvalidArgument`] if `treatment` has entries other
///   than 0/1, if a treatment reversal occurs under `absorbing = true`,
///   if `absorbing = false` without `nonabsorbing_lag >= 1` (or
///   `absorbing = true` with a nonzero lag), if
///   `never_treated_only = true` but every unit is treated at some
///   point, if fewer than two entities are supplied (entity-clustered
///   standard errors need clusters), or if no treatment switch exists;
/// * [`PanelError::InsufficientObservations`] if the window exceeds the
///   panel (`post_window + 2 > T`, or `pre_window + 1 > T` when
///   `pre_window >= 2`), or a horizon's regression has too few rows;
/// * [`PanelError::SingularDesign`] if at some horizon no period cell
///   mixes switchers with clean controls (the switch indicator is then
///   absorbed by the period effects — there is no clean comparison at
///   that horizon), or, under `reweight`, if a switcher cell has no
///   clean control (its equally-weighted contribution is undefined).
pub fn lp_did(
    data: &PanelData,
    treatment: MatRef<'_, f64>,
    config: &LpDidConfig,
) -> Result<LpDidResult, PanelError> {
    let n_ent = data.n_entities();
    let t_len = data.n_periods();
    if treatment.nrows() != n_ent {
        return Err(PanelError::Dimension {
            what: "treatment entity dimension must match the outcome's",
            expected: n_ent,
            got: treatment.nrows(),
        });
    }
    if treatment.ncols() != t_len {
        return Err(PanelError::Dimension {
            what: "treatment period dimension must match the outcome's",
            expected: t_len,
            got: treatment.ncols(),
        });
    }
    for i in 0..n_ent {
        for t in 0..t_len {
            let v = treatment[(i, t)];
            if v != 0.0 && v != 1.0 {
                return Err(PanelError::InvalidArgument {
                    what: "treatment must be a binary 0/1 indicator (found an entry \
                           that is neither 0 nor 1); LP-DiD is defined for binary \
                           treatments — code intensity designs as events first",
                });
            }
        }
    }
    if n_ent < 2 {
        return Err(PanelError::InvalidArgument {
            what: "LP-DiD clusters standard errors by entity and needs at least \
                   two entities",
        });
    }
    match (config.absorbing, config.nonabsorbing_lag) {
        (true, 0) | (false, 1..) => {}
        (true, _) => {
            return Err(PanelError::InvalidArgument {
                what: "absorbing = true with a nonzero nonabsorbing_lag is \
                       ambiguous: the stabilization window only defines the \
                       clean-control condition for non-absorbing treatments — \
                       set absorbing = false to use it",
            });
        }
        (false, 0) => {
            return Err(PanelError::InvalidArgument {
                what: "absorbing = false requires nonabsorbing_lag >= 1: the \
                       non-absorbing clean-control condition needs a \
                       stabilization window (the number of periods after a \
                       treatment change when dynamic effects are assumed \
                       settled; DGJT 2025, section 3.2)",
            });
        }
    }

    let hmax = config.post_window;
    let q = config.pre_window;
    // Feasibility of the largest windows: post horizon H needs a row with
    // t >= 1 and t + H <= T - 1; pre horizon -Q needs t >= Q <= T - 1.
    if t_len < hmax + 2 {
        return Err(PanelError::InsufficientObservations {
            what: "LP-DiD post window: horizon H needs some period t with both \
                   a lagged baseline (t >= 1) and the horizon-H lead inside the \
                   panel (t + H <= T - 1)",
            needed: hmax + 2,
            got: t_len,
        });
    }
    if q >= 2 && t_len < q + 1 {
        return Err(PanelError::InsufficientObservations {
            what: "LP-DiD pre window: horizon -Q needs some period t >= Q with \
                   the lag y_{t-Q} inside the panel",
            needed: q + 1,
            got: t_len,
        });
    }

    // Per-unit treatment path bookkeeping.
    let d = |i: usize, t: usize| treatment[(i, t)] != 0.0;
    let mut ever = vec![false; n_ent];
    for (i, e) in ever.iter_mut().enumerate() {
        *e = (0..t_len).any(|t| d(i, t));
    }
    if config.never_treated_only && ever.iter().all(|&e| e) {
        return Err(PanelError::InvalidArgument {
            what: "never_treated_only = true but every unit is treated at some \
                   point in the sample: there are no never-treated units to \
                   serve as controls — use the not-yet-treated pool \
                   (never_treated_only = false) or supply more units",
        });
    }
    if config.absorbing {
        for i in 0..n_ent {
            for t in 1..t_len {
                if !d(i, t) && d(i, t - 1) {
                    return Err(PanelError::InvalidArgument {
                        what: "treatment reverses (a unit switches from 1 back to \
                               0) but absorbing = true: the absorbing clean-control \
                               condition (not yet treated through t + h) is not \
                               defined for reversible treatments — set \
                               absorbing = false and choose a nonabsorbing_lag \
                               (DGJT 2025, section 3.2)",
                    });
                }
            }
        }
    }
    // ΔD_{it} for t >= 1 (entry +1, exit -1); prefix change counts for the
    // non-absorbing no-change windows: chg_prefix[i][t] = number of status
    // changes at periods 1..=t.
    let ddelta = |i: usize, t: usize| -> i8 {
        debug_assert!(t >= 1);
        i8::from(d(i, t)) - i8::from(d(i, t - 1))
    };
    let mut chg_prefix = vec![vec![0usize; t_len]; n_ent];
    for (i, row) in chg_prefix.iter_mut().enumerate() {
        for t in 1..t_len {
            row[t] = row[t - 1] + usize::from(ddelta(i, t) != 0);
        }
    }
    // No observed status change at periods [a, b] (0-based ΔD indices,
    // clamped to [1, T-1]; changes outside the panel are unobserved and
    // count as clean — the reference's Stata missing semantics).
    let no_change = |i: usize, a: i64, b: i64| -> bool {
        let lo = a.max(1);
        let hi = b.min(t_len as i64 - 1);
        if lo > hi {
            return true;
        }
        let (lo, hi) = (lo as usize, hi as usize);
        chg_prefix[i][hi] - chg_prefix[i][lo - 1] == 0
    };

    let outcome = data.outcome();
    let l = config.nonabsorbing_lag as i64;

    // Collect one regression's rows: (entity, period, x = ΔD, long-diff y).
    let collect = |spec: &Spec| -> Vec<(usize, usize, f64, f64)> {
        let mut rows = Vec::new();
        for i in 0..n_ent {
            for t in 1..t_len {
                let dd = ddelta(i, t);
                // The pre/post clean-control condition (module docs).
                let (ok, yv) = match *spec {
                    Spec::Post(h) => {
                        if t + h > t_len - 1 {
                            continue;
                        }
                        let clean = if config.absorbing {
                            match dd {
                                1 => true,
                                0 if config.never_treated_only => !ever[i],
                                0 => !d(i, t + h),
                                _ => unreachable!("reversal refused under absorbing"),
                            }
                        } else {
                            let ti = t as i64;
                            dd >= 0
                                && no_change(i, ti - l, ti - 1)
                                && no_change(i, ti + 1, ti + h as i64)
                                && (dd == 1 || !config.never_treated_only || !ever[i])
                        };
                        (clean, outcome[(i, t + h)] - outcome[(i, t - 1)])
                    }
                    Spec::Pre(j) => {
                        if t < j {
                            continue;
                        }
                        let clean = if config.absorbing {
                            match dd {
                                1 => true,
                                0 if config.never_treated_only => !ever[i],
                                0 => !d(i, t),
                                _ => unreachable!("reversal refused under absorbing"),
                            }
                        } else {
                            let ti = t as i64;
                            dd >= 0
                                && no_change(i, ti - l - (j as i64 - 1), ti - 1)
                                && (dd == 1 || !config.never_treated_only || !ever[i])
                        };
                        (clean, outcome[(i, t - j)] - outcome[(i, t - 1)])
                    }
                    Spec::PooledPost => {
                        if t + hmax > t_len - 1 {
                            continue;
                        }
                        let clean = if config.absorbing {
                            match dd {
                                1 => true,
                                0 if config.never_treated_only => !ever[i],
                                0 => !d(i, t + hmax),
                                _ => unreachable!("reversal refused under absorbing"),
                            }
                        } else {
                            let ti = t as i64;
                            dd >= 0
                                && no_change(i, ti - l, ti - 1)
                                && no_change(i, ti + 1, ti + hmax as i64)
                                && (dd == 1 || !config.never_treated_only || !ever[i])
                        };
                        let mean_post = (0..=hmax).map(|k| outcome[(i, t + k)]).sum::<f64>()
                            / (hmax + 1) as f64;
                        (clean, mean_post - outcome[(i, t - 1)])
                    }
                    Spec::PooledPre => {
                        if t < q {
                            continue;
                        }
                        let clean = if config.absorbing {
                            match dd {
                                1 => true,
                                0 if config.never_treated_only => !ever[i],
                                0 => !d(i, t),
                                _ => unreachable!("reversal refused under absorbing"),
                            }
                        } else {
                            let ti = t as i64;
                            dd >= 0
                                && no_change(i, ti - l - (q as i64 - 1), ti - 1)
                                && (dd == 1 || !config.never_treated_only || !ever[i])
                        };
                        let mean_pre =
                            (2..=q).map(|k| outcome[(i, t - k)]).sum::<f64>() / (q - 1) as f64;
                        (clean, mean_pre - outcome[(i, t - 1)])
                    }
                };
                if ok {
                    rows.push((i, t, f64::from(ddelta(i, t).max(0)), yv));
                }
            }
        }
        rows
    };

    // Reweighting cell tables. Post-side regressions weight by their own
    // clean sample's cell composition (equal to the reference's
    // horizon-h weights map on every cell that survives the outcome
    // requirement); pre-side regressions use the horizon-0 cells — the
    // reference joins the h = 0 weights map onto the pre samples
    // (`LP_DiD_R_example_EW.R` lines 188-198), which differs from the
    // pre samples' own composition in non-absorbing mode.
    let w0_cells: Option<Vec<Option<f64>>> = if config.reweight && (q >= 2 || config.pooled) {
        Some(cell_weights(&collect(&Spec::Post(0)), t_len)?)
    } else {
        None
    };
    let run = |spec: &Spec, what: &'static str| -> Result<CellRegression, PanelError> {
        let rows = collect(spec);
        let weights: Option<Vec<Option<f64>>> = if !config.reweight {
            None
        } else {
            match spec {
                Spec::Post(_) | Spec::PooledPost => Some(cell_weights(&rows, t_len)?),
                Spec::Pre(_) | Spec::PooledPre => w0_cells.clone(),
            }
        };
        cell_regression(&rows, weights.as_deref(), n_ent, t_len, what)
    };

    let lo = -(q as i64);
    let grid: Vec<i64> = (lo..=hmax as i64).collect();
    let mut coef = vec![0.0; grid.len()];
    let mut se = vec![0.0; grid.len()];
    let mut nobs = vec![0usize; grid.len()];
    let mut n_switchers = vec![0usize; grid.len()];
    let idx = |h: i64| (h - lo) as usize;

    for h in 0..=hmax {
        let r = run(&Spec::Post(h), "LP-DiD post-horizon regression")?;
        let k = idx(h as i64);
        coef[k] = r.beta;
        se[k] = r.se;
        nobs[k] = r.nobs;
        n_switchers[k] = r.n_switchers;
    }
    for j in 2..=q {
        let r = run(&Spec::Pre(j), "LP-DiD pre-horizon regression")?;
        let k = idx(-(j as i64));
        coef[k] = r.beta;
        se[k] = r.se;
        nobs[k] = r.nobs;
        n_switchers[k] = r.n_switchers;
    }
    // The h = -1 entry stays identically zero: it is the omitted baseline.

    let (pooled_post, pooled_pre) = if config.pooled {
        let post = run(&Spec::PooledPost, "LP-DiD pooled post regression")?;
        let post = LpDidPooled {
            att: post.beta,
            se: post.se,
            nobs: post.nobs,
            n_switchers: post.n_switchers,
        };
        let pre = if q >= 2 {
            let r = run(&Spec::PooledPre, "LP-DiD pooled pre regression")?;
            Some(LpDidPooled {
                att: r.beta,
                se: r.se,
                nobs: r.nobs,
                n_switchers: r.n_switchers,
            })
        } else {
            None
        };
        (Some(post), pre)
    } else {
        (None, None)
    };

    Ok(LpDidResult {
        horizons: grid,
        coef,
        se,
        nobs,
        n_switchers,
        pooled_post,
        pooled_pre,
        absorbing: config.absorbing,
        nonabsorbing_lag: config.nonabsorbing_lag,
        reweight: config.reweight,
        never_treated_only: config.never_treated_only,
    })
}

/// One estimated LP-DiD regression.
struct CellRegression {
    beta: f64,
    se: f64,
    nobs: usize,
    n_switchers: usize,
}

/// Per-period-cell equally-weighted-ATT weights for one clean sample
/// (DGJT §2.5; the transcription of the reference's weights map — see
/// the module docs): cell `t` with `n1` switchers and `n0` clean
/// controls gets row weight `(n1 + n0) / n0`; a cell without a switcher
/// gets `None` (its reference weight is NA and the row drops); a
/// switcher cell without controls is refused (the reference produces an
/// infinite weight there).
fn cell_weights(
    rows: &[(usize, usize, f64, f64)],
    t_len: usize,
) -> Result<Vec<Option<f64>>, PanelError> {
    let mut n1 = vec![0usize; t_len];
    let mut n0 = vec![0usize; t_len];
    for &(_, t, x, _) in rows {
        if x == 1.0 {
            n1[t] += 1;
        } else {
            n0[t] += 1;
        }
    }
    let mut w = vec![None; t_len];
    for t in 0..t_len {
        if n1[t] == 0 {
            continue;
        }
        if n0[t] == 0 {
            return Err(PanelError::SingularDesign {
                what: "LP-DiD reweighting: a period cell contains treatment \
                       switchers but no clean control, so its equally-weighted \
                       contribution is undefined (the reference implementation \
                       produces an infinite weight); enlarge the control pool \
                       (e.g. never_treated_only = false) or use the \
                       variance-weighted estimator (reweight = false)",
            });
        }
        w[t] = Some((n1[t] + n0[t]) as f64 / n0[t] as f64);
    }
    Ok(w)
}

/// OLS/WLS of the long difference on the switch indicator with period
/// fixed effects, over explicit rows `(entity, period, x, y)`, clustered
/// by entity with the fixest/reghdfe small-sample convention
/// `(n-1)/(n-K) * G/(G-1)`, `K = 1 + #period cells` (see the module
/// docs; pinned against a fixest run in `fixtures/lpdid.json`).
///
/// `weights` is `None` for OLS (the variance-weighted ATT) or a
/// per-period cell-weight table from [`cell_weights`] (rows in cells
/// without a weight are dropped, matching the reference's NA-weight
/// behaviour).
fn cell_regression(
    rows: &[(usize, usize, f64, f64)],
    weights: Option<&[Option<f64>]>,
    n_ent: usize,
    t_len: usize,
    what: &'static str,
) -> Result<CellRegression, PanelError> {
    if !rows.iter().any(|&(_, _, x, _)| x == 1.0) {
        return Err(PanelError::InvalidArgument {
            what: "LP-DiD found no treatment switch (no unit has ΔD = 1 in the \
                   usable window): nothing identifies a treatment effect — check \
                   the treatment indicator's coding (0 before treatment, 1 from \
                   entry) and the window sizes",
        });
    }
    let mut used: Vec<(usize, usize, f64, f64, f64)> = Vec::with_capacity(rows.len());
    for &(i, t, x, y) in rows {
        let w = match weights {
            None => 1.0,
            Some(table) => match table[t] {
                Some(w) => w,
                None => continue, // no switcher in this cell: weight undefined
            },
        };
        used.push((i, t, x, y, w));
    }
    let n = used.len();
    // Weighted within-period demeaning of x and y.
    let mut sw = vec![0.0f64; t_len];
    let mut swx = vec![0.0f64; t_len];
    let mut swy = vec![0.0f64; t_len];
    for &(_, t, x, y, w) in &used {
        sw[t] += w;
        swx[t] += w * x;
        swy[t] += w * y;
    }
    let n_cells = sw.iter().filter(|&&s| s > 0.0).count();
    let (mut sxx, mut sxy, mut sx2) = (0.0f64, 0.0f64, 0.0f64);
    for &(_, t, x, y, w) in &used {
        let xt = x - swx[t] / sw[t];
        let yt = y - swy[t] / sw[t];
        sxx += w * xt * xt;
        sxy += w * xt * yt;
        sx2 += w * x * x;
    }
    if sxx <= 1e-12 * sx2.max(1.0) {
        return Err(PanelError::SingularDesign {
            what: "LP-DiD: no period cell mixes treatment switchers with clean \
                   controls, so the switch indicator is collinear with the period \
                   effects and no clean comparison identifies the effect at this \
                   horizon — enlarge the window, the control pool, or the panel",
        });
    }
    let k = 1 + n_cells;
    if n <= k {
        return Err(PanelError::InsufficientObservations {
            what,
            needed: k + 1,
            got: n,
        });
    }
    let beta = sxy / sxx;

    // Cluster-by-entity sandwich on the WLS scores, fixest/reghdfe
    // small-sample factors (module docs).
    let mut g = vec![0.0f64; n_ent];
    let mut in_sample = vec![false; n_ent];
    let mut n_switchers = 0usize;
    for &(i, t, x, y, w) in &used {
        let xt = x - swx[t] / sw[t];
        let yt = y - swy[t] / sw[t];
        let e = yt - beta * xt;
        g[i] += w * xt * e;
        in_sample[i] = true;
        if x == 1.0 {
            n_switchers += 1;
        }
    }
    let n_clusters = in_sample.iter().filter(|&&b| b).count();
    debug_assert!(n_clusters >= 1);
    if n_clusters < 2 {
        return Err(PanelError::InvalidArgument {
            what: "LP-DiD: a horizon's clean sample contains a single entity — \
                   entity-clustered standard errors need at least two clusters",
        });
    }
    let meat: f64 = g.iter().map(|v| v * v).sum();
    let adj = (n as f64 - 1.0) / (n - k) as f64;
    let cadj = n_clusters as f64 / (n_clusters as f64 - 1.0);
    let var = adj * cadj * meat / (sxx * sxx);
    Ok(CellRegression {
        beta,
        se: var.max(0.0).sqrt(),
        nobs: n,
        n_switchers,
    })
}
