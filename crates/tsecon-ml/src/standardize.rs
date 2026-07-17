//! Train-set-aware standardization for leakage-safe pipelines.
//!
//! # Fit on train, apply to test — never the reverse
//!
//! **Warning.** Standardizing the *whole* series before splitting it into
//! train and test folds leaks the future into the past: every training
//! point is then centered and scaled using means and standard deviations
//! that were computed partly from the held-out test observations. The
//! leak is silent — nothing errors — but it inflates apparent
//! out-of-sample accuracy and invalidates any honest evaluation.
//!
//! The only correct order is: **split first, fit the scaler on the
//! training rows, then apply those frozen train-set scales to the test
//! rows.** [`Scaler::fit`] remembers the per-column training mean and
//! standard deviation; [`Scaler::transform`] replays them on any later
//! matrix. Do the same for the target with [`TargetCenterer`].
//!
//! ```
//! use tsecon_ml::faer::Mat;
//! use tsecon_ml::{Scaler, TargetCenterer};
//!
//! # fn demo() -> Result<(), tsecon_ml::MlError> {
//! let x_train = Mat::from_fn(4, 2, |i, j| (i * 2 + j) as f64);
//! let x_test = Mat::from_fn(2, 2, |i, j| (i + j) as f64);
//! let y_train = [1.0, 2.0, 3.0, 4.0];
//!
//! // Fit ONLY on the training rows...
//! let scaler = Scaler::fit(x_train.as_ref())?;
//! let centerer = TargetCenterer::fit(&y_train)?;
//! // ...then apply the frozen train scales to train and test alike.
//! let x_train_std = scaler.transform(x_train.as_ref())?;
//! let x_test_std = scaler.transform(x_test.as_ref())?; // test never informs the scales
//! let y_train_c = centerer.transform(&y_train);
//! # let _ = (x_train_std, x_test_std, y_train_c); Ok(())
//! # }
//! # demo().unwrap();
//! ```

use tsecon_linalg::faer::{Mat, MatRef};

use crate::error::MlError;

/// Per-column standardizer that remembers the training-set mean and scale.
///
/// [`Scaler::fit`] computes, for each column, the mean and the population
/// standard deviation (`ddof = 0`, i.e. dividing by `n`). [`transform`]
/// maps each entry to `(x - mean) / scale`. A column whose training
/// standard deviation is (numerically) zero is a constant: it is centered
/// to zero and its scale is stored as `1.0`, so `transform` returns `0.0`
/// for it rather than dividing by zero.
///
/// [`transform`]: Scaler::transform
#[derive(Debug, Clone, PartialEq)]
pub struct Scaler {
    mean: Vec<f64>,
    scale: Vec<f64>,
}

impl Scaler {
    /// Fits per-column means and population standard deviations on `x`
    /// (`n x p`, the **training** rows only).
    ///
    /// # Errors
    ///
    /// * [`MlError::EmptyInput`] if `x` has no rows or columns;
    /// * [`MlError::NonFinite`] on any NaN/infinite entry.
    pub fn fit(x: MatRef<'_, f64>) -> Result<Self, MlError> {
        let n = x.nrows();
        let p = x.ncols();
        if n == 0 || p == 0 {
            return Err(MlError::EmptyInput { what: "x" });
        }
        let mut mean = vec![0.0; p];
        let mut scale = vec![0.0; p];
        for j in 0..p {
            let mut s = 0.0;
            for i in 0..n {
                let v = x[(i, j)];
                if !v.is_finite() {
                    return Err(MlError::NonFinite { what: "x" });
                }
                s += v;
            }
            let m = s / n as f64;
            let mut var = 0.0;
            for i in 0..n {
                let d = x[(i, j)] - m;
                var += d * d;
            }
            var /= n as f64;
            let sd = var.sqrt();
            mean[j] = m;
            // Constant column: scale of 1 so transform yields 0, not NaN.
            scale[j] = if sd > 0.0 { sd } else { 1.0 };
        }
        Ok(Self { mean, scale })
    }

    /// Applies the frozen train-set scales to `x` (`n x p`), returning a new
    /// `n x p` matrix of standardized values `(x - mean) / scale`.
    ///
    /// # Errors
    ///
    /// * [`MlError::DimensionMismatch`] if `x` has a different column count
    ///   than the fitted scaler;
    /// * [`MlError::NonFinite`] on any NaN/infinite entry.
    pub fn transform(&self, x: MatRef<'_, f64>) -> Result<Mat<f64>, MlError> {
        let n = x.nrows();
        let p = x.ncols();
        if p != self.mean.len() {
            return Err(MlError::DimensionMismatch {
                what: "column count must match the fitted scaler",
                expected: self.mean.len(),
                got: p,
            });
        }
        for j in 0..p {
            for i in 0..n {
                if !x[(i, j)].is_finite() {
                    return Err(MlError::NonFinite { what: "x" });
                }
            }
        }
        Ok(Mat::from_fn(n, p, |i, j| {
            (x[(i, j)] - self.mean[j]) / self.scale[j]
        }))
    }

    /// Convenience: [`fit`](Scaler::fit) on `x` then [`transform`](Scaler::transform)
    /// the same `x`. **Only** appropriate when `x` is the training matrix.
    ///
    /// # Errors
    ///
    /// As [`fit`](Scaler::fit).
    pub fn fit_transform(x: MatRef<'_, f64>) -> Result<(Self, Mat<f64>), MlError> {
        let scaler = Self::fit(x)?;
        let out = scaler.transform(x)?;
        Ok((scaler, out))
    }

    /// The stored per-column training means.
    #[must_use]
    pub fn means(&self) -> &[f64] {
        &self.mean
    }

    /// The stored per-column training scales (standard deviations, with
    /// constant columns recorded as `1.0`).
    #[must_use]
    pub fn scales(&self) -> &[f64] {
        &self.scale
    }
}

/// Remembers the training-set mean of the target so it can be subtracted
/// from later (test) targets, matching the intercept-free convention of the
/// penalized solvers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TargetCenterer {
    mean: f64,
}

impl TargetCenterer {
    /// Fits the mean of the training target `y`.
    ///
    /// # Errors
    ///
    /// * [`MlError::EmptyInput`] if `y` is empty;
    /// * [`MlError::NonFinite`] on any NaN/infinite entry.
    pub fn fit(y: &[f64]) -> Result<Self, MlError> {
        if y.is_empty() {
            return Err(MlError::EmptyInput { what: "y" });
        }
        if y.iter().any(|v| !v.is_finite()) {
            return Err(MlError::NonFinite { what: "y" });
        }
        let mean = y.iter().sum::<f64>() / y.len() as f64;
        Ok(Self { mean })
    }

    /// Subtracts the stored training mean from `y`.
    #[must_use]
    pub fn transform(&self, y: &[f64]) -> Vec<f64> {
        y.iter().map(|v| v - self.mean).collect()
    }

    /// Adds the stored training mean back onto centered predictions,
    /// mapping model output back to the original target scale.
    #[must_use]
    pub fn inverse_transform(&self, y_centered: &[f64]) -> Vec<f64> {
        y_centered.iter().map(|v| v + self.mean).collect()
    }

    /// The stored training-set target mean.
    #[must_use]
    pub fn mean(&self) -> f64 {
        self.mean
    }
}
