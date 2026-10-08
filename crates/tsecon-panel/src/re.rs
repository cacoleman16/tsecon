//! Random-effects panel regression: feasible GLS with Swamy-Arora-type
//! variance components and a per-entity quasi-demeaning weight `theta_i`,
//! following `linearmodels.panel.RandomEffects` (version 7.0, unweighted;
//! the golden fixture `fixtures/panel_static.json` arbitrates).
//!
//! The model is `y_it = x_it' beta + u_i + e_it` with the entity effect
//! `u_i` uncorrelated with every regressor (`E[u_i | X] = 0`; the
//! assumption fixed effects do not need). With observed cells stacked
//! entity-major, `K` parameters (the constant counted), `N` entities with
//! at least one cell, `T_i` cells of entity `i` and `n = sum_i T_i`:
//!
//! 1. **Within step** — the residual `e_w` of the minimum-norm least
//!    squares of `y - ybar_i` on `X - xbar_i` (rank-revealing: the demeaned
//!    constant and any time-invariant regressor are annihilated, which is
//!    legal here — estimating time-invariant regressors is the main reason
//!    to use random effects), and
//!    `sigma2_eps = e_w'e_w / (n - K - N + 1)` with `K` counting the
//!    constant and any absorbed column (linearmodels' denominator, also
//!    without a constant).
//! 2. **Between step** — the residual `u_b` of the unweighted minimum-norm
//!    least squares of `ybar_i` on `xbar_i` over the `N` entities (this
//!    design may be rank-deficient: time dummies on an unbalanced panel).
//! 3. **Effect variance** — when `small_sample` is false or the `T_i` of
//!    the entities with an observation are all equal (linearmodels'
//!    `np.ptp(count) == 0` — deliberately **not**
//!    [`PanelData::is_balanced`], which is false for a panel whose mask
//!    drops a whole period while every entity keeps the same count):
//!    `sigma2_effects = max(0, u_b'u_b/(N - K) - sigma2_eps / Tbar)` with
//!    the **harmonic** mean `Tbar = N / sum_i (1/T_i)`. With
//!    `small_sample` on an unbalanced panel, linearmodels' formula:
//!    `M = sum_i T_i xbar_i xbar_i'`, `S = sum_i (T_i xbar_i)(T_i xbar_i)'`,
//!    `sigma2_effects = max(0, (sum_i T_i u_b,i^2 - (N - K) sigma2_eps) /
//!    (n - tr(M^{-1} S)))`. This is neither Stata's Baltagi-Chang `sa` nor
//!    plm's unbalanced `swar`, so no parity with either is claimed on
//!    unbalanced data. A singular `M` (a rank-deficient between design,
//!    e.g. time dummies) is refused by naming `small_sample`; linearmodels
//!    raises `LinAlgError` there.
//! 4. **GLS** — `theta_i = 1 - sqrt(sigma2_eps / (T_i sigma2_effects +
//!    sigma2_eps))`, `rho = sigma2_effects / (sigma2_effects + sigma2_eps)`
//!    (the intraclass correlation), and full-rank OLS of
//!    `y_it - theta_i ybar_i` on `x_it - theta_i xbar_i` with
//!    `df_resid = n - K`. `sigma2_effects = 0` gives `theta_i = 0`: pooled
//!    OLS.
//!
//! The covariance menu ([`StaticSeType`]) runs on the quasi-demeaned design
//! and its residuals. The full-rank guard on the final design uses the
//! numpy singular-value criterion `panel_fe` uses; linearmodels solves with
//! LAPACK `gelsy` and does not refuse, so a collinear final design is a
//! documented refusal here, not a parity case.
//!
//! **The Mundlak / correlated-random-effects identity.** Passing the
//! entity means of the time-varying regressors as extra regressors makes
//! the random-effects slopes on the time-varying regressors equal the
//! fixed-effects slopes exactly (Mundlak 1978), on unbalanced panels too
//! when the means run over the observed cells; the cluster-robust Wald test
//! on the mean coefficients is the robust Hausman test (Wooldridge 2010,
//! section 10.7.3). `tests/static_properties.rs` pins the identity against
//! [`crate::panel_ols_fe`].

use tsecon_linalg::faer::linalg::solvers::DenseSolveCore;
use tsecon_linalg::faer::{Mat, Side};

use crate::data::PanelData;
use crate::error::PanelError;
use crate::fe::WithinFit;
use crate::pooled::{
    design_names, n_params, numerical_rank, pooled_inference, pooled_ols,
    refuse_duplicate_constant, residual_projection, StackedRows, StaticInference, StaticSeType,
};

/// Relative size `|e_w| / |y - ybar_i|` of the within residuals at or below
/// which the within fit counts as exact (sigma2_eps is rounding noise).
const WITHIN_EXACT_FIT: f64 = 1e-12;

/// Options of [`panel_random_effects`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RandomEffectsConfig {
    /// Include an intercept (`const`, the first parameter). Default `true`.
    pub constant: bool,
    /// linearmodels' small-sample variance-component formula on an
    /// unbalanced panel (see the module docs); a no-op when every entity
    /// with an observation has the same number of cells. Default `false`.
    pub small_sample: bool,
}

impl Default for RandomEffectsConfig {
    fn default() -> Self {
        Self {
            constant: true,
            small_sample: false,
        }
    }
}

/// A fitted random-effects panel regression; produced by
/// [`panel_random_effects`].
#[derive(Debug, Clone)]
pub struct RandomEffectsFit {
    /// GLS coefficients, `const` first when the constant is included.
    pub params: Vec<f64>,
    /// Parameter names aligned with `params`.
    pub names: Vec<String>,
    /// Observed cells `n`.
    pub nobs: usize,
    /// Entities with at least one observed cell `N`.
    pub n_entities: usize,
    /// Residual degrees of freedom `n - K`.
    pub df_resid: usize,
    /// Idiosyncratic variance `sigma2_eps` (within step).
    pub sigma2_eps: f64,
    /// Entity-effect variance `sigma2_effects`, clamped at 0.
    pub sigma2_effects: f64,
    /// `sigma2_effects / (sigma2_effects + sigma2_eps)`.
    pub rho: f64,
    /// Per-entity quasi-demeaning weight `theta_i`, length `N` in layout
    /// order (`NaN` for an entity with no observed cell).
    pub theta: Vec<f64>,
    /// Whether the constant was included.
    pub constant: bool,
    /// Whether the small-sample formula was requested.
    pub small_sample: bool,
    fit: WithinFit,
}

/// Fits the random-effects estimator (see the module docs for the exact
/// linearmodels conventions).
///
/// # Errors
///
/// * [`PanelError::InsufficientData`] for an empty design (`k = 0` without
///   the constant, naming `regressors`), `N <= K` (naming `regressors`),
///   `n - K - N + 1 <= 0` or `sigma2_eps = 0` (naming `outcome`);
/// * [`PanelError::RankDeficient`] for a collinear final design or a
///   regressor duplicating the constant (naming `regressors`), and for a
///   singular small-sample moment matrix (naming `small_sample`).
pub fn panel_random_effects(
    data: &PanelData,
    config: RandomEffectsConfig,
) -> Result<RandomEffectsFit, PanelError> {
    let RandomEffectsConfig {
        constant,
        small_sample,
    } = config;
    let k = data.n_regressors();
    let kk = n_params(k, constant, "panel_re")?;
    let cells = data.observed_cells();
    refuse_duplicate_constant(data, &cells, constant)?;
    let n = cells.len();
    let means = EntityMeans::new(data, &cells, constant, kk);
    let n_ent = means.present.len();
    if n_ent <= kk {
        return Err(PanelError::InsufficientData {
            what: format!(
                "regressors: random effects needs more entities with an observation \
                 than parameters (N = {n_ent} entities, K = {kk} parameters counting the \
                 constant): the between regression of the entity means has N rows; drop \
                 regressors or supply more entities"
            ),
        });
    }
    if n < kk + n_ent {
        return Err(PanelError::InsufficientData {
            what: format!(
                "outcome: the within step leaves no degrees of freedom for sigma2_eps \
                 (n - K - N + 1 = {n} - {kk} - {n_ent} + 1 <= 0 with n observed cells, K \
                 parameters and N entities); random effects needs entities observed in \
                 more than one period"
            ),
        });
    }

    // Within step: residual of the min-norm LS of the entity-demeaned data.
    let mut yd = vec![0.0_f64; n];
    let mut xd = Mat::<f64>::zeros(n, kk);
    for r in 0..n {
        let p = means.row_pos[r];
        yd[r] = means.y[r] - means.ybar[p];
        for j in 0..kk {
            xd[(r, j)] = means.x[(r, j)] - means.xbar[(p, j)];
        }
    }
    let ew = residual_projection(&yd, &xd, "regressors")?;
    let ssr_w = ew.iter().map(|e| e * e).sum::<f64>();
    let sigma2_eps = ssr_w / (n + 1 - kk - n_ent) as f64;
    // A within fit that is exact up to rounding (the outcome an exact
    // linear function of the regressors within every entity, or constant
    // within every entity) leaves sigma2_eps at rounding noise and
    // theta_i = 1 - O(1e-8): the GLS weights are then meaningless.
    let tss_w = yd.iter().map(|v| v * v).sum::<f64>();
    if sigma2_eps.is_nan()
        || sigma2_eps <= 0.0
        || ssr_w <= WITHIN_EXACT_FIT * WITHIN_EXACT_FIT * tss_w
    {
        return Err(PanelError::InsufficientData {
            what: format!(
                "outcome: the within residuals vanish (|e_w| / |y - ybar_i| = {:e}): the \
                 outcome is an exact linear function of the regressors within every \
                 entity (or constant within every entity), so sigma2_eps = 0 and the \
                 random-effects weights theta_i are undefined",
                if tss_w > 0.0 {
                    (ssr_w / tss_w).sqrt()
                } else {
                    0.0
                }
            ),
        });
    }

    // Between step: residual of the unweighted min-norm LS of the means.
    let ub = residual_projection(&means.ybar, &means.xbar, "regressors")?;
    let counts: Vec<f64> = means
        .present
        .iter()
        .map(|&i| means.count[i] as f64)
        .collect();
    let tmin = counts.iter().copied().fold(f64::INFINITY, f64::min);
    let tmax = counts.iter().copied().fold(0.0_f64, f64::max);
    let balanced = tmin == tmax;
    let nf = n_ent as f64;
    let sigma2_effects = if small_sample && !balanced {
        let w = Mat::from_fn(n_ent, kk, |p, j| counts[p].sqrt() * means.xbar[(p, j)]);
        let singular = || PanelError::RankDeficient {
            what: "small_sample: linearmodels' small-sample variance-component formula \
                   needs M = sum_i T_i xbar_i xbar_i' to be invertible, but the between \
                   design (the entity means of the regressors) is rank-deficient — time \
                   dummies on an unbalanced panel do this; pass small_sample=False (the \
                   default Swamy-Arora components work on this design) or drop the \
                   collinear regressors (linearmodels raises LinAlgError here)"
                .to_string(),
        };
        if numerical_rank(&w) != Some(kk) {
            return Err(singular());
        }
        let mut m = Mat::<f64>::zeros(kk, kk);
        let mut s = Mat::<f64>::zeros(kk, kk);
        for (p, &t) in counts.iter().enumerate() {
            for a in 0..kk {
                for b in 0..kk {
                    let xab = means.xbar[(p, a)] * means.xbar[(p, b)];
                    m[(a, b)] += t * xab;
                    s[(a, b)] += t * t * xab;
                }
            }
        }
        let m_inv = m.llt(Side::Lower).map_err(|_| singular())?.inverse();
        let mut tr = 0.0;
        for a in 0..kk {
            for b in 0..kk {
                tr += m_inv[(a, b)] * s[(b, a)];
            }
        }
        let ssr: f64 = counts.iter().zip(&ub).map(|(t, u)| t * u * u).sum();
        let denom = n as f64 - tr;
        if denom.is_nan() || denom <= 0.0 {
            return Err(PanelError::InsufficientData {
                what: format!(
                    "small_sample: the small-sample denominator n - tr(M^-1 S) = {denom} \
                     is not positive on this panel; pass small_sample=False"
                ),
            });
        }
        ((ssr - (nf - kk as f64) * sigma2_eps) / denom).max(0.0)
    } else {
        let tbar = nf / counts.iter().map(|t| 1.0 / t).sum::<f64>();
        let ssr: f64 = ub.iter().map(|u| u * u).sum();
        (ssr / (nf - kk as f64) - sigma2_eps / tbar).max(0.0)
    };
    let rho = sigma2_effects / (sigma2_effects + sigma2_eps);
    let theta_p: Vec<f64> = counts
        .iter()
        .map(|t| 1.0 - (sigma2_eps / (t * sigma2_effects + sigma2_eps)).sqrt())
        .collect();
    let mut theta = vec![f64::NAN; data.n_entities()];
    for (p, &i) in means.present.iter().enumerate() {
        theta[i] = theta_p[p];
    }

    // GLS: full-rank OLS on the quasi-demeaned data.
    let mut ys = vec![0.0_f64; n];
    let mut xs = Mat::<f64>::zeros(n, kk);
    for r in 0..n {
        let p = means.row_pos[r];
        let th = theta_p[p];
        ys[r] = means.y[r] - th * means.ybar[p];
        for j in 0..kk {
            xs[(r, j)] = means.x[(r, j)] - th * means.xbar[(p, j)];
        }
    }
    let fit = pooled_ols(
        StackedRows {
            y: ys,
            x: xs,
            row_entity: cells.iter().map(|c| c.0).collect(),
            row_period: cells.iter().map(|c| c.1).collect(),
            n_periods: data.n_periods(),
        },
        "regressors: the random-effects (quasi-demeaned) design is rank-deficient — \
         duplicated or linearly dependent regressors (numpy singular-value criterion); \
         drop the redundant columns",
    )?;
    Ok(RandomEffectsFit {
        params: fit.params.clone(),
        names: design_names(data.names(), constant),
        nobs: n,
        n_entities: n_ent,
        df_resid: n - kk,
        sigma2_eps,
        sigma2_effects,
        rho,
        theta,
        constant,
        small_sample,
        fit,
    })
}

impl RandomEffectsFit {
    /// Standard errors, t-statistics and the covariance under `se_type`,
    /// computed on the quasi-demeaned design (see `pooled.rs`).
    ///
    /// # Errors
    ///
    /// [`PanelError::InvalidBandwidth`] for a negative or non-finite
    /// Driscoll-Kraay bandwidth.
    pub fn inference(&self, se_type: StaticSeType) -> Result<StaticInference, PanelError> {
        pooled_inference(&self.fit, se_type)
    }
}

/// The stacked observed cells with the design (constant first) and the
/// per-entity means over the observed cells — shared by random effects
/// and the between estimator.
pub(crate) struct EntityMeans {
    /// Outcome over the observed cells, entity-major.
    pub(crate) y: Vec<f64>,
    /// Design over the observed cells (`n x K`).
    pub(crate) x: Mat<f64>,
    /// Observed cells per layout entity.
    pub(crate) count: Vec<usize>,
    /// Layout indices of the entities with an observation.
    pub(crate) present: Vec<usize>,
    /// Position (in `present`) of each stacked row's entity.
    pub(crate) row_pos: Vec<usize>,
    /// Entity means of the outcome, one per present entity.
    pub(crate) ybar: Vec<f64>,
    /// Entity means of the design (`N x K`).
    pub(crate) xbar: Mat<f64>,
}

impl EntityMeans {
    pub(crate) fn new(
        data: &PanelData,
        cells: &[(usize, usize)],
        constant: bool,
        kk: usize,
    ) -> Self {
        let n = cells.len();
        let off = usize::from(constant);
        let outcome = data.outcome();
        let y: Vec<f64> = cells.iter().map(|&c| outcome[c]).collect();
        let mut x = Mat::<f64>::zeros(n, kk);
        if constant {
            for r in 0..n {
                x[(r, 0)] = 1.0;
            }
        }
        for j in 0..data.n_regressors() {
            if let Some(m) = data.regressor(j) {
                for (r, &c) in cells.iter().enumerate() {
                    x[(r, j + off)] = m[c];
                }
            }
        }
        let n_layout = data.n_entities();
        let mut count = vec![0usize; n_layout];
        for &(i, _) in cells {
            count[i] += 1;
        }
        let present: Vec<usize> = (0..n_layout).filter(|&i| count[i] > 0).collect();
        let mut pos = vec![0usize; n_layout];
        for (p, &i) in present.iter().enumerate() {
            pos[i] = p;
        }
        let row_pos: Vec<usize> = cells.iter().map(|c| pos[c.0]).collect();
        let n_ent = present.len();
        let mut ybar = vec![0.0_f64; n_ent];
        let mut xbar = Mat::<f64>::zeros(n_ent, kk);
        for r in 0..n {
            let p = row_pos[r];
            ybar[p] += y[r];
            for j in 0..kk {
                xbar[(p, j)] += x[(r, j)];
            }
        }
        for (p, &i) in present.iter().enumerate() {
            let t = count[i] as f64;
            ybar[p] /= t;
            for j in 0..kk {
                xbar[(p, j)] /= t;
            }
        }
        Self {
            y,
            x,
            count,
            present,
            row_pos,
            ybar,
            xbar,
        }
    }
}
