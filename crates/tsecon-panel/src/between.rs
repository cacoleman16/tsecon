//! The between estimator: OLS of the entity means `ybar_i` on the entity
//! means `xbar_i` over the `N` entities with an observation, following
//! `linearmodels.panel.BetweenOLS` (version 7.0; the golden fixture
//! `fixtures/panel_static.json` arbitrates).
//!
//! The means run over each entity's observed cells. With `reweight` every
//! row is multiplied by `sqrt(T_i / mean_i T_i)` — weighted least squares
//! with weights proportional to the entity's cell count (Stata
//! `xtreg, be wls`); with equal `T_i` it is a no-op. There are `N` rows, so
//! `nobs = N` and `df_resid = N - K`. The covariance menu
//! ([`StaticSeType`]) runs on the (weighted) means: `NonRobust` uses
//! `s2 = e'e / (N - K)`, `Hc1` and `ClusterEntity` are `N/(N - K)` times
//! the White sandwich — clustering by entity with one row per entity *is*
//! the White sandwich, so the two coincide numerically (linearmodels
//! agrees). Driscoll-Kraay is refused (by naming `se_type`): the between
//! regression has no time dimension.
//!
//! The full-rank guard uses the numpy singular-value criterion `panel_fe`
//! uses; linearmodels solves with LAPACK `gelsy` and does not refuse, so a
//! collinear between design (time dummies on an unbalanced panel, say) is
//! a documented refusal here.

use tsecon_linalg::faer::Mat;

use crate::data::PanelData;
use crate::error::PanelError;
use crate::fe::WithinFit;
use crate::pooled::{
    design_names, n_params, pooled_inference, pooled_ols, refuse_duplicate_constant, StackedRows,
    StaticInference, StaticSeType,
};
use crate::re::EntityMeans;

/// Options of [`panel_between`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BetweenConfig {
    /// Include an intercept (`const`, the first parameter). Default `true`.
    pub constant: bool,
    /// Weight each entity's row by `sqrt(T_i / mean T_i)` (linearmodels
    /// `reweight=True`, Stata `xtreg, be wls`). Default `false`.
    pub reweight: bool,
}

impl Default for BetweenConfig {
    fn default() -> Self {
        Self {
            constant: true,
            reweight: false,
        }
    }
}

/// A fitted between regression; produced by [`panel_between`].
#[derive(Debug, Clone)]
pub struct BetweenFit {
    /// Coefficients, `const` first when the constant is included.
    pub params: Vec<f64>,
    /// Parameter names aligned with `params`.
    pub names: Vec<String>,
    /// Rows of the between regression: the entities with an observation.
    pub nobs: usize,
    /// Entities with at least one observed cell (equal to `nobs`).
    pub n_entities: usize,
    /// Residual degrees of freedom `N - K`.
    pub df_resid: usize,
    /// Whether the constant was included.
    pub constant: bool,
    /// Whether the rows were reweighted by `sqrt(T_i / mean T_i)`.
    pub reweight: bool,
    fit: WithinFit,
}

/// Fits the between estimator (see the module docs).
///
/// # Errors
///
/// * [`PanelError::InsufficientData`] for an empty design or `N <= K`
///   (naming `regressors`);
/// * [`PanelError::RankDeficient`] for a collinear between design or a
///   regressor duplicating the constant (naming `regressors`).
pub fn panel_between(data: &PanelData, config: BetweenConfig) -> Result<BetweenFit, PanelError> {
    let BetweenConfig { constant, reweight } = config;
    let k = data.n_regressors();
    let kk = n_params(k, constant, "panel_between")?;
    let cells = data.observed_cells();
    refuse_duplicate_constant(data, &cells, constant)?;
    let means = EntityMeans::new(data, &cells, constant, kk);
    let n_ent = means.present.len();
    if n_ent <= kk {
        return Err(PanelError::InsufficientData {
            what: format!(
                "regressors: the between estimator regresses the N = {n_ent} entity means \
                 on K = {kk} columns (the constant counted) and needs N > K; drop \
                 regressors or supply more entities"
            ),
        });
    }
    let mean_t = cells.len() as f64 / n_ent as f64;
    let root_w: Vec<f64> = means
        .present
        .iter()
        .map(|&i| {
            if reweight {
                (means.count[i] as f64 / mean_t).sqrt()
            } else {
                1.0
            }
        })
        .collect();
    let y: Vec<f64> = (0..n_ent).map(|p| root_w[p] * means.ybar[p]).collect();
    let x = Mat::from_fn(n_ent, kk, |p, j| root_w[p] * means.xbar[(p, j)]);
    let fit = pooled_ols(
        StackedRows {
            y,
            x,
            row_entity: (0..n_ent).collect(),
            row_period: vec![0; n_ent],
            n_periods: 1,
        },
        "regressors: the between design (the entity means of the regressors) is \
         rank-deficient — duplicated or linearly dependent means, e.g. time dummies, \
         whose entity means depend only on the observation pattern (numpy \
         singular-value criterion); drop the redundant columns",
    )?;
    Ok(BetweenFit {
        params: fit.params.clone(),
        names: design_names(data.names(), constant),
        nobs: n_ent,
        n_entities: n_ent,
        df_resid: n_ent - kk,
        constant,
        reweight,
        fit,
    })
}

impl BetweenFit {
    /// Standard errors, t-statistics and the covariance under `se_type`
    /// (nonrobust, HC1, or entity-clustered — numerically HC1).
    ///
    /// # Errors
    ///
    /// [`PanelError::InvalidArgument`] for
    /// [`StaticSeType::DriscollKraay`]: the between regression has one row
    /// per entity and no time dimension.
    pub fn inference(&self, se_type: StaticSeType) -> Result<StaticInference, PanelError> {
        if let StaticSeType::DriscollKraay { .. } = se_type {
            return Err(PanelError::InvalidArgument {
                what: "se_type: Driscoll-Kraay standard errors are not available for \
                       the between estimator — it has one row per entity (the entity \
                       means) and no time dimension to aggregate scores over; use \
                       \"nonrobust\", \"hc1\" or \"cluster\"",
            });
        }
        pooled_inference(&self.fit, se_type)
    }
}
