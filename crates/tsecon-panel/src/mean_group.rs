//! Mean-group panel VAR (Pesaran & Smith 1995).
//!
//! The mean-group (MG) estimator fits a separate reduced-form VAR to every
//! entity and averages: with per-entity estimates `theta_i`,
//!
//! ```text
//! theta_MG = (1/N) sum_i theta_i,
//! se(theta_MG) = sd(theta_i) / sqrt(N)
//!              = sqrt( sum_i (theta_i - theta_MG)^2 / (N (N - 1)) )
//! ```
//!
//! (Pesaran & Smith 1995, J. Econometrics; the dispersion-based standard
//! error is the Pesaran-Shin-Smith 1999 convention). The averaging is
//! applied to the intercepts, the lag coefficient matrices `A_1..A_p`,
//! and the Cholesky-orthogonalized impulse responses — IRFs are averaged
//! **after** orthogonalization entity by entity, so each entity's IRF is
//! scaled by its own residual covariance.
//!
//! ## What this estimator is (and is not)
//!
//! This is the **heterogeneous-panel first slice**: it is consistent as
//! `N, T -> infinity` even when coefficients differ across entities
//! (its target is the cross-entity average effect), and — unlike pooled
//! fixed-effects dynamics — it does not suffer Nickell bias from pooling,
//! though each per-entity OLS VAR still carries its own O(1/T) small-T
//! bias. It requires a time dimension long enough to fit every entity's
//! VAR separately.
//!
//! // TODO(phase0): the pooled short-T panel VAR — Holtz-Eakin-Newey-
//! // Rosen (1988) / Arellano-Bond GMM with forward-orthogonal-deviation
//! // (Helmert) transformation, instrument collapsing, Hansen-J and AR(2)
//! // diagnostics — per the Module 04 roadmap row ("Panel VAR (GMM)").
//! // The MG estimator here is the complementary large-T tool, not a
//! // substitute for small-T micro panels.

use tsecon_linalg::faer::Mat;
use tsecon_var::{Trend, VarResults, VarSpec};

use crate::error::PanelError;

/// Mean-group panel VAR estimates; produced by [`mean_group_var`].
#[derive(Debug, Clone)]
pub struct MeanGroupVar {
    /// Number of entities `N`.
    pub n_entities: usize,
    /// Number of endogenous variables `k` (common across entities).
    pub neqs: usize,
    /// Lag order `p` of every per-entity VAR.
    pub lags: usize,
    /// Mean-group intercept (length `k`; zeros under [`Trend::None`]).
    pub intercept: Vec<f64>,
    /// Standard error of the mean-group intercept,
    /// `sd across entities / sqrt(N)`.
    pub intercept_se: Vec<f64>,
    /// Mean-group lag coefficient matrices `[A_1, ..., A_p]`, each
    /// `k x k`, averaged elementwise across entities.
    pub coefs: Vec<Mat<f64>>,
    /// Elementwise standard errors of `coefs`
    /// (`sd across entities / sqrt(N)`).
    pub coefs_se: Vec<Mat<f64>>,
    /// Mean-group Cholesky-orthogonalized impulse responses, horizons
    /// `0..=horizon`: entry `h` is the `k x k` matrix whose `(i, j)`
    /// element is the cross-entity average response of variable `i` to a
    /// one-standard-deviation orthogonalized shock to variable `j`,
    /// `h` periods earlier (each entity orthogonalized by its own
    /// `sigma_u`; Lütkepohl 2005, section 3.7).
    pub orth_irfs: Vec<Mat<f64>>,
    /// Elementwise standard errors of `orth_irfs`
    /// (`sd across entities / sqrt(N)`) — the cross-entity dispersion of
    /// the response, not a within-entity sampling band.
    pub orth_irfs_se: Vec<Mat<f64>>,
    /// The per-entity fits, for inspection of heterogeneity.
    pub entity_results: Vec<VarResults>,
}

/// Fits the Pesaran-Smith (1995) mean-group panel VAR: one
/// [`VarSpec`] fit per entity (equation-by-equation OLS via
/// `tsecon-var`), then cross-entity averages and dispersion-based
/// standard errors for the intercept, the lag matrices, and the
/// Cholesky-orthogonalized IRFs up to `horizon` (see the module docs for
/// formulas, references, and scope caveats).
///
/// `entities` holds one `T_i x k` data matrix per entity (observations
/// in rows, oldest first); the time dimensions may differ but the `k`
/// variables must match.
///
/// # Errors
///
/// * [`PanelError::InsufficientObservations`] with fewer than 2 entities
///   (the dispersion standard error needs `N >= 2`);
/// * [`PanelError::Dimension`] if entities disagree on `k`;
/// * [`PanelError::EntityVar`] if any per-entity fit or IRF fails
///   (too few rows, collinearity, non-PD residual covariance), naming
///   the entity.
pub fn mean_group_var(
    entities: &[Mat<f64>],
    lags: usize,
    trend: Trend,
    horizon: usize,
) -> Result<MeanGroupVar, PanelError> {
    let n_ent = entities.len();
    if n_ent < 2 {
        return Err(PanelError::InsufficientObservations {
            what: "mean-group panel VAR (cross-entity dispersion needs N >= 2)",
            needed: 2,
            got: n_ent,
        });
    }
    let neqs = entities[0].ncols();
    for e in entities {
        if e.ncols() != neqs {
            return Err(PanelError::Dimension {
                what: "every entity must observe the same variables",
                expected: neqs,
                got: e.ncols(),
            });
        }
    }

    let spec =
        VarSpec::new(lags, trend).map_err(|source| PanelError::EntityVar { entity: 0, source })?;
    let mut results = Vec::with_capacity(n_ent);
    let mut irfs = Vec::with_capacity(n_ent);
    for (entity, e) in entities.iter().enumerate() {
        let fit = spec
            .fit(e.as_ref())
            .map_err(|source| PanelError::EntityVar { entity, source })?;
        let irf = fit
            .irf(horizon)
            .map_err(|source| PanelError::EntityVar { entity, source })?;
        irfs.push(irf.orth_irfs);
        results.push(fit);
    }

    // Cross-entity mean and dispersion SE, elementwise.
    let (intercept, intercept_se) = mg_vec(n_ent, neqs, |i, j| results[i].intercept[j]);
    let mut coefs = Vec::with_capacity(lags);
    let mut coefs_se = Vec::with_capacity(lags);
    for lag in 0..lags {
        let (m, s) = mg_mat(n_ent, neqs, |i, r, c| results[i].coefs[lag][(r, c)]);
        coefs.push(m);
        coefs_se.push(s);
    }
    let mut orth_irfs = Vec::with_capacity(horizon + 1);
    let mut orth_irfs_se = Vec::with_capacity(horizon + 1);
    // `h` indexes each entity's IRF sequence inside the averaging closure
    // (`irfs[i][h]`), so it is a genuine cross-entity index rather than a
    // range-loop that could be an iterator.
    #[allow(clippy::needless_range_loop)]
    for h in 0..=horizon {
        let (m, s) = mg_mat(n_ent, neqs, |i, r, c| irfs[i][h][(r, c)]);
        orth_irfs.push(m);
        orth_irfs_se.push(s);
    }

    Ok(MeanGroupVar {
        n_entities: n_ent,
        neqs,
        lags,
        intercept,
        intercept_se,
        coefs,
        coefs_se,
        orth_irfs,
        orth_irfs_se,
        entity_results: results,
    })
}

/// Mean and dispersion SE (`sd / sqrt(N)`, `N - 1` divisor) of a
/// per-entity scalar family laid out as a `k`-vector.
fn mg_vec(n_ent: usize, k: usize, get: impl Fn(usize, usize) -> f64) -> (Vec<f64>, Vec<f64>) {
    let nf = n_ent as f64;
    let mut mean = vec![0.0_f64; k];
    let mut se = vec![0.0_f64; k];
    for j in 0..k {
        let m = (0..n_ent).map(|i| get(i, j)).sum::<f64>() / nf;
        let ss = (0..n_ent).map(|i| (get(i, j) - m).powi(2)).sum::<f64>();
        mean[j] = m;
        se[j] = (ss / (nf - 1.0) / nf).sqrt();
    }
    (mean, se)
}

/// Mean and dispersion SE of a per-entity `k x k` matrix family.
fn mg_mat(
    n_ent: usize,
    k: usize,
    get: impl Fn(usize, usize, usize) -> f64,
) -> (Mat<f64>, Mat<f64>) {
    let nf = n_ent as f64;
    let mean = Mat::from_fn(k, k, |r, c| {
        (0..n_ent).map(|i| get(i, r, c)).sum::<f64>() / nf
    });
    let se = Mat::from_fn(k, k, |r, c| {
        let m = mean[(r, c)];
        let ss = (0..n_ent).map(|i| (get(i, r, c) - m).powi(2)).sum::<f64>();
        (ss / (nf - 1.0) / nf).sqrt()
    });
    (mean, se)
}

/// Convenience: the response path of one `(response, impulse)` pair from
/// a mean-group IRF array, as `(point, se)` vectors over horizons.
///
/// Returns `None` if the indices are out of range.
#[must_use]
pub fn mg_irf_path(
    mg: &MeanGroupVar,
    response: usize,
    impulse: usize,
) -> Option<(Vec<f64>, Vec<f64>)> {
    if response >= mg.neqs || impulse >= mg.neqs {
        return None;
    }
    let point = mg
        .orth_irfs
        .iter()
        .map(|m: &Mat<f64>| m[(response, impulse)])
        .collect();
    let se = mg
        .orth_irfs_se
        .iter()
        .map(|m: &Mat<f64>| m[(response, impulse)])
        .collect();
    Some((point, se))
}
