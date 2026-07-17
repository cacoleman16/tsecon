//! The Pesaran & Smith (1995) mean-group (MG) estimator.
//!
//! For a heterogeneous panel in which every unit `i` obeys its own linear
//! time-series regression
//!
//! ```text
//! y_it = a_i + b_i' x_it + e_it,   t = 1..T_i,
//! ```
//!
//! with a *unit-specific* slope vector `b_i`, the mean-group estimator runs a
//! separate OLS per unit, drops the intercept, and reports the simple
//! cross-unit average of the slope vectors,
//!
//! ```text
//! b_MG = (1/N) sum_i b_i .
//! ```
//!
//! Because the `b_i` are treated as an i.i.d. sample from the slope
//! distribution, the sampling variance of `b_MG` is the cross-unit *sample
//! covariance* of the `b_i` divided by `N`,
//!
//! ```text
//! Var(b_MG) = (1 / (N (N-1))) sum_i (b_i - b_MG)(b_i - b_MG)' ,
//! ```
//!
//! so the standard error of coordinate `k` is `SE_k = sd_i(b_ik) / sqrt(N)`
//! with `sd` the `ddof = 1` sample standard deviation, and the mean-group
//! t-statistic is `b_MG,k / SE_k`, referred to a `t_{N-1}` distribution.
//!
//! The per-unit OLS is delegated to [`tsecon_hac::ols`]; this crate never
//! reimplements least squares.

use tsecon_hac::ols;
use tsecon_stats::{ContinuousDist, StudentT};

use crate::error::PanelTsError;

/// One unit of a heterogeneous panel: its own response and regressor columns.
///
/// `x` holds the `k` regressor columns (the constant is **not** included — it
/// is added internally and is never part of the averaged slope vector). Every
/// column must be index-aligned with `y`.
#[derive(Debug, Clone, PartialEq)]
pub struct PanelUnit {
    /// Response series for this unit, length `T_i`.
    pub y: Vec<f64>,
    /// Regressor columns for this unit, `k` columns each of length `T_i`.
    pub x: Vec<Vec<f64>>,
}

impl PanelUnit {
    /// Construct a unit from its response and regressor columns.
    pub fn new(y: Vec<f64>, x: Vec<Vec<f64>>) -> Self {
        Self { y, x }
    }

    /// Number of regressors `k` carried by this unit (excluding the intercept).
    pub fn k(&self) -> usize {
        self.x.len()
    }

    /// Number of time periods `T_i` observed for this unit.
    pub fn t(&self) -> usize {
        self.y.len()
    }
}

/// A fitted mean-group (or CCE mean-group) estimate.
///
/// All vectors are indexed by regressor `k` in the order the columns were
/// supplied in [`PanelUnit::x`]; for the CCE estimator only the own-`x` slopes
/// survive, so the length is still `k`.
#[derive(Debug, Clone, PartialEq)]
pub struct MeanGroup {
    /// Mean-group coefficient vector `b_MG`, length `k`.
    pub coef: Vec<f64>,
    /// Standard errors `SE_k = sd_i(b_ik) / sqrt(N)`, length `k`.
    pub se: Vec<f64>,
    /// t-statistics `coef / se`, length `k`.
    pub tstat: Vec<f64>,
    /// The per-unit slope vectors `b_i` that were averaged, `N` rows of `k`.
    pub coef_per_unit: Vec<Vec<f64>>,
    /// Number of units `N`.
    pub n_units: usize,
    /// Number of averaged regressors `k`.
    pub k: usize,
}

impl MeanGroup {
    /// Two-sided p-values for `H0: b_MG,k = 0`, using a `t_{N-1}` reference
    /// (Pesaran & Smith 1995 treat the `N` unit slopes as the sample).
    ///
    /// # Errors
    ///
    /// [`PanelTsError::Stats`] if the Student-t distribution cannot be formed
    /// (only for a degenerate `N`, which the constructors already reject).
    pub fn pvalues(&self) -> Result<Vec<f64>, PanelTsError> {
        let dist = StudentT::new((self.n_units - 1) as f64)?;
        Ok(self.tstat.iter().map(|&t| 2.0 * dist.sf(t.abs())).collect())
    }

    /// Equal-tailed confidence intervals at the given `level` (e.g. `0.95`),
    /// `coef ± t_{N-1, 1-alpha/2} · se`, one `(lower, upper)` pair per
    /// regressor.
    ///
    /// # Errors
    ///
    /// [`PanelTsError::Stats`] if `level` is not in `(0, 1)` (the quantile
    /// then falls outside the distribution's domain).
    pub fn conf_int(&self, level: f64) -> Result<Vec<(f64, f64)>, PanelTsError> {
        let dist = StudentT::new((self.n_units - 1) as f64)?;
        let q = dist.ppf(1.0 - 0.5 * (1.0 - level))?;
        Ok(self
            .coef
            .iter()
            .zip(self.se.iter())
            .map(|(&c, &s)| (c - q * s, c + q * s))
            .collect())
    }
}

/// Validate the common shape requirements shared by every mean-group variant
/// and return the common regressor count `k`.
///
/// Checks: at least two units; each unit has at least one regressor; every
/// unit shares the same `k`; every regressor column is index-aligned with its
/// own unit's response.
pub(crate) fn validate_units(units: &[PanelUnit]) -> Result<usize, PanelTsError> {
    let n = units.len();
    if n < 2 {
        return Err(PanelTsError::TooFewUnits { n });
    }
    let k = units[0].k();
    if k == 0 {
        return Err(PanelTsError::NoRegressors { unit: 0 });
    }
    for (i, unit) in units.iter().enumerate() {
        if unit.k() == 0 {
            return Err(PanelTsError::NoRegressors { unit: i });
        }
        if unit.k() != k {
            return Err(PanelTsError::InconsistentRegressors {
                unit: i,
                expected: k,
                got: unit.k(),
            });
        }
        let t = unit.t();
        for (column, col) in unit.x.iter().enumerate() {
            if col.len() != t {
                return Err(PanelTsError::RaggedUnit {
                    unit: i,
                    column,
                    expected: t,
                    got: col.len(),
                });
            }
        }
    }
    Ok(k)
}

/// Build a design matrix `[const, x1, ..., xm]` for one unit.
pub(crate) fn design_with_const(x: &[Vec<f64>], t: usize) -> Vec<Vec<f64>> {
    let mut cols = Vec::with_capacity(x.len() + 1);
    cols.push(vec![1.0_f64; t]);
    cols.extend(x.iter().cloned());
    cols
}

/// Assemble a [`MeanGroup`] from the collected per-unit slope vectors.
///
/// Each `slopes[i]` must have length `k` and there must be `n >= 2` of them;
/// both invariants are guaranteed by [`validate_units`] and the callers.
pub(crate) fn assemble_mean_group(slopes: Vec<Vec<f64>>, k: usize) -> MeanGroup {
    let n = slopes.len();
    let nf = n as f64;

    // Cross-unit mean of the slope vectors.
    let mut coef = vec![0.0_f64; k];
    for row in &slopes {
        for (c, &r) in coef.iter_mut().zip(row.iter()) {
            *c += r;
        }
    }
    for c in coef.iter_mut() {
        *c /= nf;
    }

    // Cross-unit sum of squared deviations, per coordinate.
    let mut ss = vec![0.0_f64; k];
    for row in &slopes {
        for ((s, &r), &c) in ss.iter_mut().zip(row.iter()).zip(coef.iter()) {
            let d = r - c;
            *s += d * d;
        }
    }

    // SE_k = sd_i(b_ik) / sqrt(N) with sd the ddof=1 sample sd; t = coef / se.
    let denom = nf - 1.0;
    let se: Vec<f64> = ss.iter().map(|&s| (s / denom).sqrt() / nf.sqrt()).collect();
    let tstat: Vec<f64> = coef.iter().zip(se.iter()).map(|(&c, &s)| c / s).collect();

    MeanGroup {
        coef,
        se,
        tstat,
        coef_per_unit: slopes,
        n_units: n,
        k,
    }
}

/// The Pesaran & Smith (1995) mean-group estimator.
///
/// Runs a per-unit OLS of `y_i` on `[const, x_i]`, drops the intercept, and
/// MG-averages the slope vectors (see the module docs for the exact SE
/// formula). Units may be *unbalanced* (different `T_i`) since each regression
/// stands alone.
///
/// # Errors
///
/// [`PanelTsError::TooFewUnits`] for `N < 2`; [`PanelTsError::NoRegressors`],
/// [`PanelTsError::InconsistentRegressors`], or [`PanelTsError::RaggedUnit`]
/// for malformed units; [`PanelTsError::Ols`] wrapping any per-unit OLS
/// failure (too few periods, collinear or non-finite design).
pub fn mean_group(units: &[PanelUnit]) -> Result<MeanGroup, PanelTsError> {
    let k = validate_units(units)?;
    let mut slopes = Vec::with_capacity(units.len());
    for (i, unit) in units.iter().enumerate() {
        let design = design_with_const(&unit.x, unit.t());
        let fit = ols(&unit.y, &design).map_err(|source| PanelTsError::Ols { unit: i, source })?;
        // params = [const, b_i1, ..., b_ik]; keep the slopes only.
        slopes.push(fit.params.iter().skip(1).copied().collect::<Vec<f64>>());
    }
    Ok(assemble_mean_group(slopes, k))
}
