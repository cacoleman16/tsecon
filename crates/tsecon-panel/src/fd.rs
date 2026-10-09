//! The first-difference estimator: OLS of `y_it - y_i,t-1` on
//! `x_it - x_i,t-1` (plus an intercept on request), following
//! `linearmodels.panel.FirstDifferenceOLS` (version 7.0; the golden fixture
//! `fixtures/panel_static.json` arbitrates).
//!
//! **Rows.** A difference row `(i, t)` exists when entity `i` is observed
//! in calendar period `t >= 1` **and** in `t - 1` — calendar adjacency in
//! the `N x T` layout. An internal gap therefore drops the differences that
//! would bridge it, and a period nobody observes is never bridged: the
//! difference across it is not a one-period change. linearmodels differs
//! here: it differences consecutive entries of its time axis, so with an
//! interior period missing for everyone it differences **across** the hole
//! (on `EmplUK` without 1980: 751 rows against 611 calendar rows, slopes
//! `[-0.4427, 0.5248]` against `[-0.4375, 0.4127]`;
//! `tests/static_properties.rs` documents the case). Its time axis is also
//! built in first-appearance order, so a frame whose first entity lacks the
//! first period differences non-adjacent years (808 rows on plm's natural
//! `EmplUK` order instead of 891) — the fixture generator builds its frame
//! time-major and asserts the sorted axis. Rows are entity-major.
//!
//! **Intercept.** `constant = false` matches linearmodels, which refuses a
//! constant in first differences. `constant = true` adds the intercept that
//! plm `model = "fd"` and Stata `regress D.y D.x` include by default — a
//! linear trend in levels; it is pinned against linearmodels `PooledOLS` on
//! the hand-differenced frame.
//!
//! With `m` rows and `K` parameters, `df_resid = m - K`, and the covariance
//! menu ([`StaticSeType`]) is the pooled one on the differenced design
//! (Driscoll-Kraay indexed by the later period `t`). Differencing a
//! strictly exogenous model with i.i.d. errors induces an MA(1) error, so
//! entity clustering is the default in the Python binding.

use tsecon_linalg::faer::Mat;

use crate::data::PanelData;
use crate::error::PanelError;
use crate::fe::WithinFit;
use crate::pooled::{
    check_data_scale, design_names, pooled_inference, pooled_ols, scale_error, StackedRows,
    StaticInference, StaticSeType,
};

/// Relative tolerance below which a differenced regressor counts as
/// annihilated (time-invariant): the same absorption tolerance as the
/// within estimator's guard.
const DIFFERENCE_TOL: f64 = 1e-10;

/// A fitted first-difference regression; produced by
/// [`panel_first_difference`].
#[derive(Debug, Clone)]
pub struct FirstDifferenceFit {
    /// Coefficients, `const` first when the intercept is included.
    pub params: Vec<f64>,
    /// Parameter names aligned with `params`.
    pub names: Vec<String>,
    /// Difference rows `m`: cells observed in both `t` and `t - 1`.
    pub nobs: usize,
    /// Entities contributing at least one difference.
    pub n_entities: usize,
    /// Residual degrees of freedom `m - K`.
    pub df_resid: usize,
    /// Whether the intercept was included.
    pub constant: bool,
    fit: WithinFit,
}

/// Fits the first-difference estimator (see the module docs).
///
/// # Errors
///
/// * [`PanelError::InsufficientData`] for no regressors (naming
///   `regressors`), `T < 2` or `m <= K` (naming `outcome`), and for data
///   whose sums of squares leave the range of double precision (naming
///   `outcome` or `regressors`);
/// * [`PanelError::RankDeficient`] for a regressor that never changes
///   between consecutive observed periods (time-invariant: first
///   differencing removes it) or a collinear differenced design (naming
///   `regressors`).
pub fn panel_first_difference(
    data: &PanelData,
    constant: bool,
) -> Result<FirstDifferenceFit, PanelError> {
    let k = data.n_regressors();
    if k == 0 {
        return Err(PanelError::InsufficientData {
            what: "regressors: panel_fd needs at least one regressor (k = 0); the \
                   intercept-only first-difference regression is just the mean change \
                   of the outcome"
                .to_string(),
        });
    }
    let t_len = data.n_periods();
    if t_len < 2 {
        return Err(PanelError::InsufficientData {
            what: format!(
                "outcome: first differences need at least 2 periods; the panel has \
                 T = {t_len}"
            ),
        });
    }
    check_data_scale(data, &data.observed_cells())?;
    let kk = k + usize::from(constant);
    let off = usize::from(constant);
    let mut rows: Vec<(usize, usize)> = Vec::new();
    for i in 0..data.n_entities() {
        for t in 1..t_len {
            if data.observed(i, t) && data.observed(i, t - 1) {
                rows.push((i, t));
            }
        }
    }
    let m = rows.len();
    if m <= kk {
        return Err(PanelError::InsufficientData {
            what: format!(
                "outcome: only {m} first differences exist (cells observed in both t \
                 and t - 1, by calendar adjacency) for K = {kk} parameters; the \
                 first-difference regression needs more differences than parameters"
            ),
        });
    }
    let outcome = data.outcome();
    let dy: Vec<f64> = rows
        .iter()
        .map(|&(i, t)| outcome[(i, t)] - outcome[(i, t - 1)])
        .collect();
    let mut dx = Mat::<f64>::zeros(m, kk);
    if constant {
        for r in 0..m {
            dx[(r, 0)] = 1.0;
        }
    }
    for j in 0..k {
        if let Some(xm) = data.regressor(j) {
            let mut raw = 0.0_f64;
            let mut diff = 0.0_f64;
            for (r, &(i, t)) in rows.iter().enumerate() {
                let d = xm[(i, t)] - xm[(i, t - 1)];
                dx[(r, j + off)] = d;
                raw += xm[(i, t)] * xm[(i, t)];
                diff += d * d;
            }
            if !diff.is_finite() {
                return Err(scale_error());
            }
            if diff.sqrt() <= DIFFERENCE_TOL * raw.sqrt() {
                return Err(PanelError::RankDeficient {
                    what: format!(
                        "regressors: column {j} ({}) does not change between consecutive \
                         observed periods of any entity (it is time-invariant), so first \
                         differencing removes it and its coefficient is not identified; \
                         drop it, or estimate it with panel_re (random effects) or \
                         panel_between",
                        data.names()[j]
                    ),
                });
            }
        }
    }
    let mut n_ent = 0usize;
    let mut last = usize::MAX;
    for &(i, _) in &rows {
        if i != last {
            n_ent += 1;
            last = i;
        }
    }
    let fit = pooled_ols(
        StackedRows {
            y: dy,
            x: dx,
            row_entity: rows.iter().map(|r| r.0).collect(),
            row_period: rows.iter().map(|r| r.1).collect(),
            n_periods: t_len,
        },
        "regressors: the first-differenced design is rank-deficient — duplicated or \
         linearly dependent differenced regressors (a regressor that is a linear trend \
         differences to a constant, which collides with constant=True) under the numpy \
         singular-value criterion; drop the redundant columns",
    )?;
    Ok(FirstDifferenceFit {
        params: fit.params.clone(),
        names: design_names(data.names(), constant),
        nobs: m,
        n_entities: n_ent,
        df_resid: m - kk,
        constant,
        fit,
    })
}

impl FirstDifferenceFit {
    /// Standard errors, t-statistics and the covariance under `se_type`
    /// on the differenced design (Driscoll-Kraay over the later period of
    /// each difference, calendar lags).
    ///
    /// # Errors
    ///
    /// [`PanelError::InvalidBandwidth`] for a negative or non-finite
    /// Driscoll-Kraay bandwidth; [`PanelError::InsufficientData`] (naming
    /// `se_type`) for [`StaticSeType::ClusterEntity`] when every difference
    /// belongs to one entity, or [`StaticSeType::DriscollKraay`] when every
    /// difference falls in one period (a two-period panel): that group's
    /// score sum is `X'e = 0` by the normal equations, so the covariance
    /// would be identically zero; and (naming `outcome`) for a non-finite
    /// covariance.
    pub fn inference(&self, se_type: StaticSeType) -> Result<StaticInference, PanelError> {
        pooled_inference(&self.fit, se_type)
    }
}
