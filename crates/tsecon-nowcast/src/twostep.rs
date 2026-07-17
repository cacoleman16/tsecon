//! The Doz-Giannone-Reichlin (2011) two-step estimator and the
//! Banbura-Modugno (2014) ragged-edge nowcast.
//!
//! # The estimator
//!
//! Step 1 standardizes the `T x N` panel and extracts `r` principal-component
//! factors (via the validated [`tsecon_favar::FactorModel`]). Step 2 fits a
//! factor VAR(`p`) to those PC factors (via [`tsecon_var`]) for the transition
//! dynamics, reads the idiosyncratic variances off the standardized
//! reconstruction residuals, and assembles the state space. A single Kalman
//! smoother pass ([`crate::statespace::smooth_fixed`]) then re-estimates the
//! common factor optimally, and the nowcast of a target series is its loadings
//! dotted with the smoothed factor at the sample edge, de-standardized.
//!
//! # What is and is not validated
//!
//! This is the **DGR estimator**, *not* the one-step Gaussian MLE that
//! statsmodels' `DynamicFactor.fit` computes; the two produce different
//! parameter estimates and smoothed factors, and the crate does **not**
//! tolerance-match them. The Kalman/state-space *step* is reference-exact
//! against statsmodels at fixed parameters (see [`crate::statespace`] and
//! `tests/golden.rs`); the two-step estimates themselves are validated
//! structurally — the smoothed factor tracks the simulated truth and the
//! ragged-edge nowcast moves in the economically expected direction.

use tsecon_favar::FactorModel;
use tsecon_linalg::faer::{Mat, MatRef};
use tsecon_var::{Trend, VarSpec};

use crate::error::NowcastError;
use crate::statespace::{smooth_fixed, DfmParams, DfmSmoothing};

/// A fitted two-step dynamic-factor nowcasting model.
///
/// Holds the estimated parameters on the standardized scale together with the
/// standardization moments needed to map factor movements back onto the level
/// of each observed series.
#[derive(Debug, Clone)]
pub struct Nowcaster {
    n_series: usize,
    n_factors: usize,
    factor_order: usize,
    /// Column means of the training panel (length `N`).
    center: Vec<f64>,
    /// Column standard deviations (`ddof = 0`) of the training panel (`N`).
    scale: Vec<f64>,
    /// Estimated parameters on the *standardized* scale.
    params: DfmParams,
    /// Smoothed states/factors/loglik from the balanced training pass.
    smoothing: DfmSmoothing,
}

impl Nowcaster {
    /// Fits the two-step DGR model to a **balanced** panel `data`
    /// (`T x N`, observations in rows, oldest first) with `n_factors`
    /// common factors and a factor VAR of order `factor_order`.
    ///
    /// # Errors
    ///
    /// * [`NowcastError::EmptyInput`] / [`NowcastError::InvalidArgument`] for
    ///   a degenerate panel or factor/lag count;
    /// * [`NowcastError::NonFinite`] if the training panel contains NaN or
    ///   infinity (training requires a balanced, fully-observed panel — the
    ///   ragged edge is only introduced at nowcast time);
    /// * errors propagated from the factor model, the factor VAR, or the
    ///   state-space smoother.
    pub fn fit_two_step(
        data: MatRef<'_, f64>,
        n_factors: usize,
        factor_order: usize,
    ) -> Result<Self, NowcastError> {
        let t = data.nrows();
        let n = data.ncols();
        if t == 0 || n == 0 {
            return Err(NowcastError::EmptyInput {
                what: "training panel is empty",
            });
        }
        if n_factors == 0 || factor_order == 0 {
            return Err(NowcastError::InvalidArgument {
                what: "n_factors and factor_order must both be at least 1",
            });
        }
        if n_factors > n {
            return Err(NowcastError::InvalidArgument {
                what: "n_factors cannot exceed the number of series",
            });
        }
        for j in 0..n {
            for i in 0..t {
                if !data[(i, j)].is_finite() {
                    return Err(NowcastError::NonFinite {
                        what: "training panel (must be balanced and finite)",
                    });
                }
            }
        }

        // --- Step 1: standardized PCA factors and loadings. ---
        let fm = FactorModel::fit(data)?;
        let center = fm.center().to_vec();
        let scale = fm.scale().to_vec();
        let factors = fm.factors(n_factors)?; // T x r
        let loadings = fm.loadings(n_factors)?; // N x r

        // Idiosyncratic variances = per-series variance of the standardized
        // reconstruction residual Z - F L' (population / ddof = 0). The mean
        // residual is ~0 by the least-squares projection, so the mean of
        // squares is the variance.
        let recon = fm.reconstruct_standardized(n_factors)?; // T x N
        let z_std = standardize_panel(data, &center, &scale);
        let mut idiosyncratic = vec![0.0; n];
        for j in 0..n {
            let mut ss = 0.0;
            for i in 0..t {
                let e = z_std[(i, j)] - recon[(i, j)];
                ss += e * e;
            }
            idiosyncratic[j] = ss / t as f64;
        }

        // --- Step 2: factor VAR(p) transition (no deterministic term: the
        // PC factors are mean-zero by construction). ---
        let spec = VarSpec::new(factor_order, Trend::None)?;
        let var = spec.fit(factors.as_ref())?;

        // Stacked AR block [A_1 | ... | A_p] as the top r rows of the
        // companion transition (r x r p).
        let r = n_factors;
        let factor_ar = Mat::from_fn(r, r * factor_order, |i, col| {
            let lag = col / r;
            let j = col % r;
            var.coefs[lag][(i, j)]
        });
        // The factor-innovation covariance (ML residual covariance).
        let factor_cov = var.sigma_u_mle.clone();

        let params = DfmParams {
            loadings,
            factor_ar,
            factor_cov,
            idiosyncratic,
        };

        // --- Kalman smoother pass over the balanced standardized panel. ---
        let smoothing = smooth_fixed(&params, z_std.as_ref())?;

        Ok(Self {
            n_series: n,
            n_factors,
            factor_order,
            center,
            scale,
            params,
            smoothing,
        })
    }

    /// Number of series `N`.
    #[inline]
    pub fn n_series(&self) -> usize {
        self.n_series
    }

    /// Number of factors `r`.
    #[inline]
    pub fn n_factors(&self) -> usize {
        self.n_factors
    }

    /// Factor-VAR order `p`.
    #[inline]
    pub fn factor_order(&self) -> usize {
        self.factor_order
    }

    /// The estimated parameters (standardized scale).
    #[inline]
    pub fn params(&self) -> &DfmParams {
        &self.params
    }

    /// The smoothed factors from the balanced training pass (`T x r`).
    #[inline]
    pub fn smoothed_factors(&self) -> &[Vec<f64>] {
        &self.smoothing.smoothed_factors
    }

    /// The Gaussian log-likelihood of the balanced training pass at the
    /// two-step parameters.
    #[inline]
    pub fn loglik(&self) -> f64 {
        self.smoothing.loglik
    }

    /// Column means used to standardize the training panel.
    #[inline]
    pub fn center(&self) -> &[f64] {
        &self.center
    }

    /// Column standard deviations (`ddof = 0`) used to standardize.
    #[inline]
    pub fn scale(&self) -> &[f64] {
        &self.scale
    }

    /// De-standardizes a factor movement into series `series`'s level:
    /// `center + scale * (loadings_i . f)`.
    fn destandardized_fit(&self, series: usize, factor: &[f64]) -> f64 {
        let loadings = &self.params.loadings;
        let mut acc = 0.0;
        for k in 0..self.n_factors {
            acc += loadings[(series, k)] * factor[k];
        }
        self.center[series] + self.scale[series] * acc
    }

    /// Nowcasts every series at the sample edge from the given panel `data`
    /// (`T x N`, oldest first), which may be **ragged**: the last rows of
    /// some series may be NaN (missing). The Kalman filter uses only the
    /// available observations, then the smoothed factor at the final period
    /// is mapped onto each series and de-standardized to its level.
    ///
    /// Returns the length-`N` vector of edge nowcasts alongside the smoothing
    /// output (whose `smoothed_factors` last entry is the edge factor).
    ///
    /// The panel is standardized with the *training* moments before filtering
    /// (out-of-sample scaling), matching how the parameters were estimated.
    ///
    /// # Errors
    ///
    /// [`NowcastError::DimensionMismatch`] on a column-count mismatch and
    /// whatever the state-space smoother returns.
    pub fn nowcast_panel(&self, data: MatRef<'_, f64>) -> Result<NowcastResult, NowcastError> {
        if data.ncols() != self.n_series {
            return Err(NowcastError::DimensionMismatch {
                what: "nowcast panel must have N columns",
                expected: self.n_series,
                got: data.ncols(),
            });
        }
        if data.nrows() == 0 {
            return Err(NowcastError::EmptyInput {
                what: "nowcast panel has no observations",
            });
        }
        // Standardize with the training moments; NaN passes through as NaN.
        let z = standardize_panel(data, &self.center, &self.scale);
        let smoothing = smooth_fixed(&self.params, z.as_ref())?;
        let edge_factor = smoothing
            .smoothed_factors
            .last()
            .ok_or(NowcastError::EmptyInput {
                what: "smoother produced no periods",
            })?
            .clone();
        let mut values = Vec::with_capacity(self.n_series);
        for series in 0..self.n_series {
            values.push(self.destandardized_fit(series, &edge_factor));
        }
        Ok(NowcastResult {
            values,
            edge_factor,
            smoothing,
        })
    }

    /// Nowcasts a single target `series` at the edge of `data`. Convenience
    /// wrapper over [`Self::nowcast_panel`].
    ///
    /// # Errors
    ///
    /// [`NowcastError::SeriesOutOfRange`] if `series >= N`, plus anything
    /// [`Self::nowcast_panel`] returns.
    pub fn nowcast_series(
        &self,
        data: MatRef<'_, f64>,
        series: usize,
    ) -> Result<f64, NowcastError> {
        if series >= self.n_series {
            return Err(NowcastError::SeriesOutOfRange {
                requested: series,
                n_series: self.n_series,
            });
        }
        let res = self.nowcast_panel(data)?;
        Ok(res.values[series])
    }
}

/// The result of an edge nowcast over a (possibly ragged) panel.
#[derive(Debug, Clone)]
pub struct NowcastResult {
    /// Edge nowcast (level) of every series (length `N`).
    pub values: Vec<f64>,
    /// The smoothed factor vector at the final period (length `r`).
    pub edge_factor: Vec<f64>,
    /// The full smoothing output over the nowcast panel.
    pub smoothing: DfmSmoothing,
}

/// Standardizes a panel column-by-column with the supplied moments,
/// `Z[(i,j)] = (X[(i,j)] - center[j]) / scale[j]`. NaN entries stay NaN
/// (missing observations propagate through the standardization untouched).
pub(crate) fn standardize_panel(data: MatRef<'_, f64>, center: &[f64], scale: &[f64]) -> Mat<f64> {
    Mat::from_fn(data.nrows(), data.ncols(), |i, j| {
        (data[(i, j)] - center[j]) / scale[j]
    })
}
