//! Panel data container: the `N x T` layout with an optional observation
//! mask for unbalanced (ragged) panels.
//!
//! Every panel is stored as `N x T` matrices (entities in rows, periods in
//! columns) so the within transformation, clustering, and per-period score
//! aggregation can index `(entity, time)` cells directly without id
//! lookups. A **balanced** panel observes every entity in every period and
//! carries no mask ([`PanelData::balanced`]); an **unbalanced** panel
//! carries an `N x T` observation mask ([`PanelData::unbalanced`]) — `true`
//! where the cell is observed — and the estimators skip the other cells:
//! the within projections average over the observed cells of each entity
//! or period, the lag designs use only rows whose every lag is observed,
//! the entity-cluster and Driscoll-Kraay score sums run over the observed
//! cells, and the degrees-of-freedom accounting counts the entities and
//! periods that carry at least one observation. Cells outside the mask may
//! hold anything, `NaN` included, and are never read.
//!
//! The balanced constructor stays the fast path: a mask that is `true`
//! everywhere is stored as no mask at all, so a fully observed panel takes
//! exactly the balanced code path (operation for operation — the results
//! are bit-identical, which `tests/unbalanced_golden.rs` asserts).

use tsecon_linalg::faer::{Mat, MatRef};

use crate::error::PanelError;

/// A panel: one outcome and `k` regressors, each observed as an
/// `n_entities x n_periods` matrix (entity `i` in row `i`, period `t` in
/// column `t`), plus an optional observation mask (see the module docs).
#[derive(Debug, Clone)]
pub struct PanelData {
    outcome: Mat<f64>,
    regressors: Vec<Mat<f64>>,
    names: Vec<String>,
    /// `None` for a balanced panel; else `N * T` flags, entity-major
    /// (`mask[i * T + t]`), `true` where the cell is observed.
    mask: Option<Vec<bool>>,
    /// Number of observed cells (`N * T` when balanced).
    n_obs: usize,
}

impl PanelData {
    /// Creates a balanced panel from an `N x T` outcome matrix and named
    /// `N x T` regressor matrices.
    ///
    /// Regressors may be entity-varying or common across entities (use
    /// [`PanelData::broadcast_common`] to lift a common time series into
    /// the `N x T` layout).
    ///
    /// # Errors
    ///
    /// * [`PanelError::InvalidArgument`] if the outcome is empty
    ///   (`N == 0` or `T == 0`);
    /// * [`PanelError::Dimension`] if a regressor's shape differs from
    ///   the outcome's;
    /// * [`PanelError::NonFinite`] if any cell is NaN/infinite (an
    ///   unbalanced panel must say which cells are missing through
    ///   [`PanelData::unbalanced`]; nothing is skipped silently).
    pub fn balanced(
        outcome: Mat<f64>,
        regressors: Vec<(String, Mat<f64>)>,
    ) -> Result<Self, PanelError> {
        Self::build(outcome, regressors, None)
    }

    /// Creates an unbalanced panel: `N x T` matrices as in
    /// [`PanelData::balanced`] plus an `N x T` observation `mask` (`N` rows
    /// of `T` flags, `true` where entity `i` is observed in period `t`).
    /// Cells with a `false` flag are ignored by every estimator and may
    /// hold `NaN`. A mask that is `true` everywhere yields a balanced
    /// panel (the fast path; results are bit-identical to
    /// [`PanelData::balanced`]).
    ///
    /// # Errors
    ///
    /// * [`PanelError::InvalidArgument`] if the outcome is empty, if the
    ///   mask has no observed cell;
    /// * [`PanelError::Dimension`] if a regressor's shape, or the mask's,
    ///   differs from the outcome's;
    /// * [`PanelError::NonFinite`] if an **observed** cell is NaN/infinite.
    pub fn unbalanced(
        outcome: Mat<f64>,
        regressors: Vec<(String, Mat<f64>)>,
        mask: &[Vec<bool>],
    ) -> Result<Self, PanelError> {
        let (n, t) = (outcome.nrows(), outcome.ncols());
        if mask.len() != n {
            return Err(PanelError::Dimension {
                what: "mask: the observation mask's entity dimension must match the \
                       outcome's (one row of period flags per entity)",
                expected: n,
                got: mask.len(),
            });
        }
        if let Some(row) = mask.iter().find(|r| r.len() != t) {
            return Err(PanelError::Dimension {
                what: "mask: every row of the observation mask must hold one flag per \
                       period (the outcome's period dimension)",
                expected: t,
                got: row.len(),
            });
        }
        let flat: Vec<bool> = mask.iter().flat_map(|r| r.iter().copied()).collect();
        if flat.iter().all(|&m| m) {
            return Self::build(outcome, regressors, None);
        }
        if !flat.iter().any(|&m| m) {
            return Err(PanelError::InvalidArgument {
                what: "mask: the observation mask marks no cell as observed (every flag \
                       is false); mark the observed (entity, period) cells true",
            });
        }
        Self::build(outcome, regressors, Some(flat))
    }

    fn build(
        outcome: Mat<f64>,
        regressors: Vec<(String, Mat<f64>)>,
        mask: Option<Vec<bool>>,
    ) -> Result<Self, PanelError> {
        let (n, t) = (outcome.nrows(), outcome.ncols());
        if n == 0 || t == 0 {
            return Err(PanelError::InvalidArgument {
                what: "outcome: the outcome panel must have at least one entity and one \
                       period",
            });
        }
        check_finite(outcome.as_ref(), mask.as_deref(), "outcome")?;
        let mut names = Vec::with_capacity(regressors.len());
        let mut mats = Vec::with_capacity(regressors.len());
        for (name, m) in regressors {
            if m.nrows() != n {
                return Err(PanelError::Dimension {
                    what: "regressors: a regressor's entity dimension must match the \
                           outcome's",
                    expected: n,
                    got: m.nrows(),
                });
            }
            if m.ncols() != t {
                return Err(PanelError::Dimension {
                    what: "regressors: a regressor's period dimension must match the \
                           outcome's",
                    expected: t,
                    got: m.ncols(),
                });
            }
            check_finite(m.as_ref(), mask.as_deref(), "regressors")?;
            names.push(name);
            mats.push(m);
        }
        let n_obs = mask
            .as_ref()
            .map_or(n * t, |m| m.iter().filter(|&&v| v).count());
        Ok(Self {
            outcome,
            regressors: mats,
            names,
            mask,
            n_obs,
        })
    }

    /// Lifts a common (entity-invariant) time series of length `T` into
    /// the `n_entities x T` panel layout by repeating it in every row —
    /// the natural representation for an aggregate shock or policy
    /// variable observed once per period.
    #[must_use]
    pub fn broadcast_common(series: &[f64], n_entities: usize) -> Mat<f64> {
        Mat::from_fn(n_entities, series.len(), |_, t| series[t])
    }

    /// Number of entities `N` (rows of the layout, observed or not).
    #[must_use]
    pub fn n_entities(&self) -> usize {
        self.outcome.nrows()
    }

    /// Number of periods `T` (columns of the layout, observed or not).
    #[must_use]
    pub fn n_periods(&self) -> usize {
        self.outcome.ncols()
    }

    /// Number of observed cells: `N * T` for a balanced panel, the number
    /// of `true` mask flags otherwise.
    #[must_use]
    pub fn nobs(&self) -> usize {
        self.n_obs
    }

    /// Number of regressors `k`.
    #[must_use]
    pub fn n_regressors(&self) -> usize {
        self.regressors.len()
    }

    /// The `N x T` outcome matrix.
    #[must_use]
    pub fn outcome(&self) -> MatRef<'_, f64> {
        self.outcome.as_ref()
    }

    /// The `j`-th `N x T` regressor matrix, or `None` past the end.
    #[must_use]
    pub fn regressor(&self, j: usize) -> Option<MatRef<'_, f64>> {
        self.regressors.get(j).map(Mat::as_ref)
    }

    /// Regressor names, in design-column order.
    #[must_use]
    pub fn names(&self) -> &[String] {
        &self.names
    }

    /// `true` when every cell is observed (no mask, or a mask that is
    /// `true` everywhere — the two are stored identically).
    #[must_use]
    pub fn is_balanced(&self) -> bool {
        self.mask.is_none()
    }

    /// The observation mask, entity-major (`mask[i * T + t]`), or `None`
    /// for a balanced panel.
    #[must_use]
    pub fn mask(&self) -> Option<&[bool]> {
        self.mask.as_deref()
    }

    /// Whether entity `i` is observed in period `t` (always `true` on a
    /// balanced panel; `false` outside the layout).
    #[must_use]
    pub fn observed(&self, i: usize, t: usize) -> bool {
        if i >= self.n_entities() || t >= self.n_periods() {
            return false;
        }
        self.mask
            .as_ref()
            .is_none_or(|m| m[i * self.n_periods() + t])
    }

    /// The observed `(entity, period)` cells in entity-major order — the
    /// row order of every stacked design in this crate.
    #[must_use]
    pub fn observed_cells(&self) -> Vec<(usize, usize)> {
        let (n, t) = (self.n_entities(), self.n_periods());
        let mut cells = Vec::with_capacity(self.n_obs);
        for i in 0..n {
            for tt in 0..t {
                if self.observed(i, tt) {
                    cells.push((i, tt));
                }
            }
        }
        cells
    }

    /// Observed cells per entity (`T` everywhere on a balanced panel).
    #[must_use]
    pub fn entity_counts(&self) -> Vec<usize> {
        let (n, t) = (self.n_entities(), self.n_periods());
        (0..n)
            .map(|i| (0..t).filter(|&tt| self.observed(i, tt)).count())
            .collect()
    }

    /// Observed cells per period (`N` everywhere on a balanced panel).
    #[must_use]
    pub fn period_counts(&self) -> Vec<usize> {
        let (n, t) = (self.n_entities(), self.n_periods());
        (0..t)
            .map(|tt| (0..n).filter(|&i| self.observed(i, tt)).count())
            .collect()
    }
}

/// Rejects NaN/infinite observed cells with [`PanelError::NonFinite`]
/// (masked-out cells are not read).
fn check_finite(
    m: MatRef<'_, f64>,
    mask: Option<&[bool]>,
    what: &'static str,
) -> Result<(), PanelError> {
    let t = m.ncols();
    for j in 0..t {
        for i in 0..m.nrows() {
            if mask.is_some_and(|mk| !mk[i * t + j]) {
                continue;
            }
            if !m[(i, j)].is_finite() {
                return Err(PanelError::NonFinite { what });
            }
        }
    }
    Ok(())
}
