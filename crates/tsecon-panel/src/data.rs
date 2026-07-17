//! Balanced-panel data container.
//!
//! A balanced panel observes every entity `i = 0..N` at every period
//! `t = 0..T`. [`PanelData`] stores the outcome and each regressor as an
//! `N x T` matrix (entities in rows, periods in columns) so the within
//! transformation, clustering, and per-period score aggregation can index
//! `(entity, time)` cells directly without id lookups.
//!
//! // TODO(phase0): unbalanced/ragged panels. The planned design adds an
//! // optional `N x T` observation mask to this container; the within
//! // demeaning, entity clustering, and Driscoll-Kraay period aggregation
//! // in `fe.rs`/`lp.rs` all iterate cells and will simply skip masked
//! // ones, so the balanced constructor below stays the fast path and the
//! // estimator code needs no structural change.

use tsecon_linalg::faer::{Mat, MatRef};

use crate::error::PanelError;

/// A balanced panel: one outcome and `k` regressors, each observed as an
/// `n_entities x n_periods` matrix (entity `i` in row `i`, period `t` in
/// column `t`).
#[derive(Debug, Clone)]
pub struct PanelData {
    outcome: Mat<f64>,
    regressors: Vec<Mat<f64>>,
    names: Vec<String>,
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
    /// * [`PanelError::NonFinite`] if any cell is NaN/infinite.
    pub fn balanced(
        outcome: Mat<f64>,
        regressors: Vec<(String, Mat<f64>)>,
    ) -> Result<Self, PanelError> {
        let (n, t) = (outcome.nrows(), outcome.ncols());
        if n == 0 || t == 0 {
            return Err(PanelError::InvalidArgument {
                what: "the outcome panel must have at least one entity and one period",
            });
        }
        check_finite(outcome.as_ref(), "outcome panel")?;
        let mut names = Vec::with_capacity(regressors.len());
        let mut mats = Vec::with_capacity(regressors.len());
        for (name, m) in regressors {
            if m.nrows() != n {
                return Err(PanelError::Dimension {
                    what: "regressor entity dimension must match the outcome's",
                    expected: n,
                    got: m.nrows(),
                });
            }
            if m.ncols() != t {
                return Err(PanelError::Dimension {
                    what: "regressor period dimension must match the outcome's",
                    expected: t,
                    got: m.ncols(),
                });
            }
            check_finite(m.as_ref(), "regressor panel")?;
            names.push(name);
            mats.push(m);
        }
        Ok(Self {
            outcome,
            regressors: mats,
            names,
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

    /// Number of entities `N`.
    #[must_use]
    pub fn n_entities(&self) -> usize {
        self.outcome.nrows()
    }

    /// Number of periods `T`.
    #[must_use]
    pub fn n_periods(&self) -> usize {
        self.outcome.ncols()
    }

    /// Total stacked observations `N * T` (the panel is balanced).
    #[must_use]
    pub fn nobs(&self) -> usize {
        self.n_entities() * self.n_periods()
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
}

/// Rejects NaN/infinite cells with [`PanelError::NonFinite`].
fn check_finite(m: MatRef<'_, f64>, what: &'static str) -> Result<(), PanelError> {
    for j in 0..m.ncols() {
        for i in 0..m.nrows() {
            if !m[(i, j)].is_finite() {
                return Err(PanelError::NonFinite { what });
            }
        }
    }
    Ok(())
}
