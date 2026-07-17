//! The Banbura-Modugno (2014) **news / update decomposition** for the
//! dynamic-factor nowcast.
//!
//! # The problem
//!
//! When a newer data vintage arrives, the nowcast of a target series revises.
//! For a *fixed* parameter model the Kalman smoother is a purely **linear**
//! operator on the observed data (zero intercept: `d = 0`, `c = 0`, and the
//! stationary initialization has zero mean), so the target nowcast is an
//! *affine* function of the observations. This makes the revision decompose
//! **exactly** as a weighted sum of *news* — the surprises in the
//! newly-revealed observations.
//!
//! # The exact identity
//!
//! Let `g(vintage)` be the target nowcast: the common-component projection
//! `center_s + scale_s * (loadings_s . f_hat_{target_period})`, where
//! `f_hat` is the smoothed factor from running [`smooth_fixed`] on the
//! standardized vintage. Let `O` be the set of cells observed in the **new**
//! vintage, `J ⊆ O` the *newly-revealed* subset (missing in the old vintage,
//! present in the new), and `B = O \ J` the cells observed in both (with
//! identical values). Because `g` is affine under the fixed new-vintage
//! missing pattern,
//!
//! ```text
//! g(new) = c_O + Σ_{cell∈O} w_cell · y_cell ,
//! ```
//!
//! with constant weights `w_cell = ∂g/∂y_cell`. Replacing each `J`-cell by its
//! **old-vintage Kalman forecast** `ŷ_j = E_old[y_j | old data]` and
//! subtracting,
//!
//! ```text
//! g(new) − g(new|J←forecast) = Σ_{j∈J} w_j · (y_j − ŷ_j) = Σ_{j∈J} w_j · news_j .
//! ```
//!
//! The **projection property** of the Kalman smoother — filling a missing cell
//! with its own smoothed conditional mean adds no information and leaves the
//! state estimate unchanged — gives `g(new|J←forecast) = g(old)` exactly.
//! Hence
//!
//! ```text
//! new_nowcast − old_nowcast = Σ_{j∈J} weight_j · news_j          (EXACT)
//! ```
//!
//! where `news_j = actual_j − forecast_j` and
//! `contribution_j = weight_j · news_j`. This adding-up identity is
//! **self-validating** and is asserted to machine precision (~`1e-10`) in the
//! tests; the only numerical slack is the projection-property residual between
//! two distinct smoother runs (old pattern vs. new pattern with `J` filled at
//! the forecast), which is `~1e-11` in practice.
//!
//! # The Kalman weights
//!
//! Because the smoother is exactly linear with zero intercept, the sensitivity
//! `∂f_hat_{target_period}/∂z_j` (`z` = standardized observation) equals the
//! smoothed factor obtained by running [`smooth_fixed`] on a **unit-impulse**
//! standardized panel: the new-vintage missing pattern, all observed cells set
//! to `0` except cell `j` set to `1`. Writing that impulse response as
//! `imp_j` (length `r`), the level weight is
//!
//! ```text
//! weight_j = (scale_s / scale_i) · (loadings_s . imp_j) = ∂nowcast / ∂actual_j ,
//! ```
//!
//! reusing the crate's own Kalman machinery ([`smooth_fixed`]) rather than any
//! new filter. The generator `fixtures/generate_nowcast_news_fixtures.py`
//! cross-checks this analytic weight against a **finite-difference** sensitivity
//! `[g(new; y_j+ε) − g(new; y_j−ε)] / 2ε` computed with an independent NumPy
//! Kalman smoother, to `~1e-6` (`fixtures/nowcast_news.json`).

use tsecon_linalg::faer::{Mat, MatRef};

use crate::error::NowcastError;
use crate::statespace::{smooth_fixed, DfmParams};
use crate::twostep::{standardize_panel, Nowcaster};

/// The contribution of a single newly-observed data point to the nowcast
/// revision, in the Banbura-Modugno news decomposition.
///
/// All quantities are in the **level** (de-standardized) units of the relevant
/// series, so that `contribution == weight * news` exactly.
#[derive(Debug, Clone, PartialEq)]
pub struct NewsContribution {
    /// Series index `i` of the newly-observed cell.
    pub series: usize,
    /// Period (row) index `t` of the newly-observed cell.
    pub period: usize,
    /// The actual value revealed in the new vintage, `y_j` (level).
    pub actual: f64,
    /// The old-vintage Kalman forecast of the cell,
    /// `ŷ_j = E_old[y_j | old data]` (level).
    pub forecast: f64,
    /// The news / surprise, `news_j = actual_j − forecast_j` (level).
    pub news: f64,
    /// The Kalman weight `∂nowcast / ∂actual_j`: the sensitivity of the target
    /// nowcast to this observation under the new-vintage missing pattern.
    pub weight: f64,
    /// The contribution to the revision, `contribution_j = weight_j · news_j`.
    pub contribution: f64,
}

/// The Banbura-Modugno news decomposition of a nowcast revision between two
/// data vintages.
///
/// The adding-up identity `total_revision == Σ contribution_j` holds to
/// machine precision.
#[derive(Debug, Clone, PartialEq)]
pub struct NewsDecomposition {
    /// The target series index whose nowcast was decomposed.
    pub target_series: usize,
    /// The target period (row) whose nowcast was decomposed.
    pub target_period: usize,
    /// The old-vintage nowcast of the target (level).
    pub old_nowcast: f64,
    /// The new-vintage nowcast of the target (level).
    pub new_nowcast: f64,
    /// The total revision, `new_nowcast − old_nowcast` (level).
    pub total_revision: f64,
    /// One entry per newly-revealed data point (missing in the old vintage,
    /// present in the new), in row-major `(period, series)` order.
    pub contributions: Vec<NewsContribution>,
}

impl Nowcaster {
    /// Decomposes the revision of this model's nowcast of `target_series` at
    /// `target_period` between an `old_vintage` and a `new_vintage` of the same
    /// `T x N` shape, where the new vintage reveals additional observations at
    /// the ragged edge.
    ///
    /// Returns the old and new nowcasts, the total revision, and a per-
    /// newly-observed-datapoint breakdown `{series, period, news, weight,
    /// contribution}` satisfying the exact adding-up identity
    /// `total_revision == Σ contribution_j` (see the [module docs](crate::news)).
    ///
    /// The nowcast is the common-component projection
    /// `center_s + scale_s · (loadings_s . f_hat_{target_period})`, consistent
    /// with the crate's other nowcast routines.
    ///
    /// # Errors
    ///
    /// * [`NowcastError::SeriesOutOfRange`] if `target_series >= N`;
    /// * [`NowcastError::InvalidArgument`] if `target_period` is out of range,
    ///   the two vintages differ in shape, or a cell observed in the old
    ///   vintage is missing or changed in the new (the decomposition requires
    ///   the new vintage to only *add* observations);
    /// * [`NowcastError::DimensionMismatch`] on a column-count mismatch with
    ///   the fitted model;
    /// * whatever [`smooth_fixed`] returns.
    pub fn news_decomposition(
        &self,
        old_vintage: MatRef<'_, f64>,
        new_vintage: MatRef<'_, f64>,
        target_series: usize,
        target_period: usize,
    ) -> Result<NewsDecomposition, NowcastError> {
        news_decomposition_at(
            self.params(),
            self.center(),
            self.scale(),
            old_vintage,
            new_vintage,
            target_series,
            target_period,
        )
    }
}

/// The parameter-explicit form of the Banbura-Modugno news decomposition.
///
/// Identical to [`Nowcaster::news_decomposition`] but taking the fixed model
/// [`DfmParams`] (on the standardized scale) together with the standardization
/// moments `center` / `scale` directly, rather than reading them off a fitted
/// [`Nowcaster`]. This is the reference-validated entry point: the golden
/// fixture supplies known parameters and moments so the Kalman weights can be
/// checked against an independent finite-difference reference.
///
/// # Errors
///
/// See [`Nowcaster::news_decomposition`]; additionally
/// [`NowcastError::DimensionMismatch`] if `center` / `scale` do not have length
/// `N`, and [`NowcastError::InvalidArgument`] if any `scale` entry is not
/// strictly positive.
pub fn news_decomposition_at(
    params: &DfmParams,
    center: &[f64],
    scale: &[f64],
    old_vintage: MatRef<'_, f64>,
    new_vintage: MatRef<'_, f64>,
    target_series: usize,
    target_period: usize,
) -> Result<NewsDecomposition, NowcastError> {
    let n = params.n_series();
    let r = params.n_factors();

    // ---- Shape / argument validation. ------------------------------------
    if center.len() != n {
        return Err(NowcastError::DimensionMismatch {
            what: "center must have length N",
            expected: n,
            got: center.len(),
        });
    }
    if scale.len() != n {
        return Err(NowcastError::DimensionMismatch {
            what: "scale must have length N",
            expected: n,
            got: scale.len(),
        });
    }
    for &s in scale {
        if !s.is_finite() || s <= 0.0 {
            return Err(NowcastError::InvalidArgument {
                what: "scale entries must be finite and strictly positive",
            });
        }
    }
    if target_series >= n {
        return Err(NowcastError::SeriesOutOfRange {
            requested: target_series,
            n_series: n,
        });
    }
    if old_vintage.nrows() != new_vintage.nrows() || old_vintage.ncols() != new_vintage.ncols() {
        return Err(NowcastError::InvalidArgument {
            what: "old and new vintages must have the same shape",
        });
    }
    let t = new_vintage.nrows();
    if t == 0 {
        return Err(NowcastError::EmptyInput {
            what: "vintage has no observations",
        });
    }
    if new_vintage.ncols() != n {
        return Err(NowcastError::DimensionMismatch {
            what: "vintage must have N columns",
            expected: n,
            got: new_vintage.ncols(),
        });
    }
    if target_period >= t {
        return Err(NowcastError::InvalidArgument {
            what: "target_period is out of range",
        });
    }

    // ---- Classify cells and collect the newly-revealed set J. ------------
    // A cell is newly revealed iff it is missing (NaN) in the old vintage and
    // finite in the new. Cells observed in the old vintage must be unchanged
    // in the new (the new vintage only *adds* observations); infinities are
    // rejected exactly as `smooth_fixed` does.
    let mut newly: Vec<(usize, usize)> = Vec::new();
    for tt in 0..t {
        for i in 0..n {
            let o = old_vintage[(tt, i)];
            let ny = new_vintage[(tt, i)];
            if o.is_infinite() || ny.is_infinite() {
                return Err(NowcastError::NonFinite {
                    what: "vintage (entries must be finite or NaN-for-missing)",
                });
            }
            let o_obs = o.is_finite();
            let n_obs = ny.is_finite();
            if o_obs {
                // Observed in the old vintage: must still be observed and equal.
                if !n_obs || o != ny {
                    return Err(NowcastError::InvalidArgument {
                        what: "a cell observed in the old vintage is missing or \
                               changed in the new vintage",
                    });
                }
            } else if n_obs {
                newly.push((tt, i));
            }
        }
    }

    // ---- Smooth both vintages once. --------------------------------------
    let z_old = standardize_panel(old_vintage, center, scale);
    let z_new = standardize_panel(new_vintage, center, scale);
    let sm_old = smooth_fixed(params, z_old.as_ref())?;
    let sm_new = smooth_fixed(params, z_new.as_ref())?;

    let old_nowcast = destandardize(
        params,
        center,
        scale,
        target_series,
        &sm_old.smoothed_factors[target_period],
    );
    let new_nowcast = destandardize(
        params,
        center,
        scale,
        target_series,
        &sm_new.smoothed_factors[target_period],
    );
    let total_revision = new_nowcast - old_nowcast;

    // ---- Per-cell forecast, news, weight, contribution. ------------------
    let loadings = &params.loadings;
    let mut contributions = Vec::with_capacity(newly.len());
    for &(tt, i) in &newly {
        // Old-vintage Kalman forecast of this cell (level): the common-
        // component projection onto the old-vintage smoothed factor.
        let forecast = destandardize(params, center, scale, i, &sm_old.smoothed_factors[tt]);
        let actual = new_vintage[(tt, i)];
        let news = actual - forecast;

        // Analytic Kalman weight via a unit-impulse smoother pass over the
        // NEW-vintage missing pattern (see the module docs).
        let imp = impulse_response(params, new_vintage, tt, i, target_period)?;
        let mut load_dot = 0.0;
        for k in 0..r {
            load_dot += loadings[(target_series, k)] * imp[k];
        }
        let weight = (scale[target_series] / scale[i]) * load_dot;
        let contribution = weight * news;

        contributions.push(NewsContribution {
            series: i,
            period: tt,
            actual,
            forecast,
            news,
            weight,
            contribution,
        });
    }

    Ok(NewsDecomposition {
        target_series,
        target_period,
        old_nowcast,
        new_nowcast,
        total_revision,
        contributions,
    })
}

/// De-standardizes a smoothed factor into series `series`'s level:
/// `center + scale * (loadings_series . factor)`.
fn destandardize(
    params: &DfmParams,
    center: &[f64],
    scale: &[f64],
    series: usize,
    factor: &[f64],
) -> f64 {
    let loadings = &params.loadings;
    let mut acc = 0.0;
    for (k, &f) in factor.iter().enumerate() {
        acc += loadings[(series, k)] * f;
    }
    center[series] + scale[series] * acc
}

/// The impulse response of the smoothed factor at `target_period` to a unit
/// standardized observation at cell `(imp_period, imp_series)`, under the
/// missing pattern of `new_vintage`.
///
/// Runs [`smooth_fixed`] on a standardized panel that is `0` at every observed
/// cell except the impulse cell (set to `1`), with NaN wherever `new_vintage`
/// is missing. Because the smoother is exactly linear with zero intercept, this
/// response equals `∂f_hat_{target_period}/∂z_{imp}` — the analytic Kalman
/// sensitivity, no finite differencing.
fn impulse_response(
    params: &DfmParams,
    new_vintage: MatRef<'_, f64>,
    imp_period: usize,
    imp_series: usize,
    target_period: usize,
) -> Result<Vec<f64>, NowcastError> {
    let t = new_vintage.nrows();
    let n = new_vintage.ncols();
    let impulse = Mat::from_fn(t, n, |tt, i| {
        if new_vintage[(tt, i)].is_finite() {
            if tt == imp_period && i == imp_series {
                1.0
            } else {
                0.0
            }
        } else {
            f64::NAN
        }
    });
    let sm = smooth_fixed(params, impulse.as_ref())?;
    Ok(sm.smoothed_factors[target_period].clone())
}
