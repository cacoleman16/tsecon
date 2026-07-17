//! Leakage-safe cross-validation for serially dependent data.
//!
//! # Why naive K-fold leaks
//!
//! Standard K-fold cross-validation shuffles observations into folds and
//! assumes each is exchangeable. Time series violate this twice over:
//!
//! * **Serial correlation.** Because `y_t` and `y_{t+1}` are dependent, a
//!   training point placed immediately next to a test point leaks
//!   information across the split — the model effectively sees a smoothed
//!   version of the held-out target. Test error is optimistically biased.
//! * **Overlapping targets.** When the label is built over a window (a
//!   multi-step return, a cumulative multiplier, an `h`-step-ahead
//!   forecast), a single training label's window can *physically overlap*
//!   the test window. The training and test sets then share raw data even
//!   though their index sets are disjoint — textbook leakage.
//!
//! This module provides splitters that respect the arrow of time:
//!
//! * [`expanding_origin_splits`] / [`rolling_origin_splits`] — the
//!   forecaster's evaluation, training only on the past of each test block;
//! * [`purged_kfold_splits`] — blocked K-fold with **purging** and an
//!   **embargo** (Lopez de Prado 2018, *Advances in Financial Machine
//!   Learning*, ch. 7): drop training observations whose label window can
//!   overlap the test block (purge) and a further band immediately after it
//!   (embargo, for forward serial correlation);
//!
//! and a [`cv_select`] driver that scores an elastic-net `lambda` grid
//! under a pluggable loss.

use tsecon_linalg::faer::MatRef;

use crate::coordinate_descent::{cd_engine, CoordDescentOptions};
use crate::error::MlError;
use crate::util::{check_xy, columns};

/// A single train/test partition of the row indices `0..n`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Split {
    /// Row indices used for training (ascending).
    pub train: Vec<usize>,
    /// Row indices used for testing (ascending, contiguous).
    pub test: Vec<usize>,
}

/// Expanding-origin (growing-window) splits: for each origin the training
/// set is the entire history `0..origin` and the test set is the next
/// `horizon` observations `origin..origin+horizon`. The first origin is
/// `initial_train`; successive origins advance by `step`.
///
/// Training sets are strictly nested (each is a prefix superset of the
/// previous), and no test index ever precedes its training set — the
/// honest out-of-sample forecasting protocol (Tashman 2000; Hyndman &
/// Athanasopoulos, *Forecasting: Principles and Practice*, "time series
/// cross-validation").
///
/// # Errors
///
/// [`MlError::InvalidArgument`] if `initial_train == 0`, `horizon == 0`,
/// `step == 0`, or `initial_train + horizon > n` (no split fits).
pub fn expanding_origin_splits(
    n: usize,
    initial_train: usize,
    horizon: usize,
    step: usize,
) -> Result<Vec<Split>, MlError> {
    validate_origin(n, initial_train, horizon, step)?;
    let mut splits = Vec::new();
    let mut origin = initial_train;
    while origin + horizon <= n {
        let train: Vec<usize> = (0..origin).collect();
        let test: Vec<usize> = (origin..origin + horizon).collect();
        splits.push(Split { train, test });
        origin += step;
    }
    Ok(splits)
}

/// Rolling-origin (sliding fixed-window) splits: like
/// [`expanding_origin_splits`] but the training set is the most recent
/// `window` observations `origin-window..origin` rather than all history.
/// Useful when the data-generating process drifts and stale history hurts.
///
/// # Errors
///
/// [`MlError::InvalidArgument`] if `window == 0`, `horizon == 0`,
/// `step == 0`, or `window + horizon > n`.
pub fn rolling_origin_splits(
    n: usize,
    window: usize,
    horizon: usize,
    step: usize,
) -> Result<Vec<Split>, MlError> {
    validate_origin(n, window, horizon, step)?;
    let mut splits = Vec::new();
    let mut origin = window;
    while origin + horizon <= n {
        let train: Vec<usize> = (origin - window..origin).collect();
        let test: Vec<usize> = (origin..origin + horizon).collect();
        splits.push(Split { train, test });
        origin += step;
    }
    Ok(splits)
}

fn validate_origin(
    n: usize,
    train_size: usize,
    horizon: usize,
    step: usize,
) -> Result<(), MlError> {
    if train_size == 0 {
        return Err(MlError::InvalidArgument {
            what: "the initial training window must be non-empty",
        });
    }
    if horizon == 0 {
        return Err(MlError::InvalidArgument {
            what: "horizon must be at least 1",
        });
    }
    if step == 0 {
        return Err(MlError::InvalidArgument {
            what: "step must be at least 1",
        });
    }
    if train_size + horizon > n {
        return Err(MlError::InvalidArgument {
            what: "initial training window plus horizon exceeds the sample size",
        });
    }
    Ok(())
}

/// Blocked K-fold with purging and embargo (Lopez de Prado 2018, ch. 7).
///
/// The indices `0..n` are cut into `k` contiguous blocks; each block serves
/// once as the test set. Training is every *other* index, minus two guard
/// bands measured from the test block `[test_start, test_end)`:
///
/// * **Purge** (`purge` observations, both sides): drop training indices in
///   `[test_start - purge, test_start)` and `[test_end, test_end + purge)`.
///   These are the observations whose label window — of length up to
///   `purge` — can overlap the test block, so keeping them would leak the
///   held-out data into training.
/// * **Embargo** (`embargo` observations, right side only): additionally
///   drop training indices in `[test_end, test_end + embargo)`. This kills
///   the forward serial-correlation leak from test features into the
///   immediately following training labels; there is no left embargo
///   because information flows forward in time.
///
/// The right-hand exclusion therefore extends to
/// `test_end + max(purge, embargo)`. Blocks are made as even as possible;
/// the first `n % k` blocks get one extra index.
///
/// # Errors
///
/// [`MlError::InvalidArgument`] if `k < 2` or `k > n`.
pub fn purged_kfold_splits(
    n: usize,
    k: usize,
    purge: usize,
    embargo: usize,
) -> Result<Vec<Split>, MlError> {
    if k < 2 {
        return Err(MlError::InvalidArgument {
            what: "k must be at least 2",
        });
    }
    if k > n {
        return Err(MlError::InvalidArgument {
            what: "k must not exceed the sample size",
        });
    }

    // Contiguous block boundaries; the first `n % k` blocks are one longer.
    let base = n / k;
    let rem = n % k;
    let mut bounds = Vec::with_capacity(k + 1);
    bounds.push(0usize);
    let mut acc = 0usize;
    for b in 0..k {
        acc += base + usize::from(b < rem);
        bounds.push(acc);
    }

    let mut splits = Vec::with_capacity(k);
    for b in 0..k {
        let test_start = bounds[b];
        let test_end = bounds[b + 1];
        let test: Vec<usize> = (test_start..test_end).collect();

        // Left guard: [test_start - purge, test_start).
        let left_excl = test_start.saturating_sub(purge);
        // Right guard: [test_end, test_end + max(purge, embargo)).
        let right_excl_end = test_end + purge.max(embargo);

        let train: Vec<usize> = (0..n)
            .filter(|&i| {
                // Not in the test block.
                if i >= test_start && i < test_end {
                    return false;
                }
                // Not in the left purge band.
                if i >= left_excl && i < test_start {
                    return false;
                }
                // Not in the right purge/embargo band.
                if i >= test_end && i < right_excl_end {
                    return false;
                }
                true
            })
            .collect();

        splits.push(Split { train, test });
    }
    Ok(splits)
}

/// Loss aggregating (`y_true`, `y_pred`) into a scalar; smaller is better.
/// See [`mse`] and [`mae`].
pub type Loss = fn(&[f64], &[f64]) -> f64;

/// Mean squared error `(1/m) sum (y_true - y_pred)^2`.
///
/// Returns `0.0` for empty input.
#[must_use]
pub fn mse(y_true: &[f64], y_pred: &[f64]) -> f64 {
    if y_true.is_empty() {
        return 0.0;
    }
    let s: f64 = y_true
        .iter()
        .zip(y_pred)
        .map(|(a, b)| (a - b) * (a - b))
        .sum();
    s / y_true.len() as f64
}

/// Mean absolute error `(1/m) sum |y_true - y_pred|`.
///
/// Returns `0.0` for empty input.
#[must_use]
pub fn mae(y_true: &[f64], y_pred: &[f64]) -> f64 {
    if y_true.is_empty() {
        return 0.0;
    }
    let s: f64 = y_true.iter().zip(y_pred).map(|(a, b)| (a - b).abs()).sum();
    s / y_true.len() as f64
}

/// Result of a cross-validated `lambda` search.
#[derive(Debug, Clone, PartialEq)]
pub struct CvResult {
    /// The `lambda` grid that was scored.
    pub lambdas: Vec<f64>,
    /// Fold-averaged loss at each `lambda` (`mean_loss[i]` for
    /// `lambdas[i]`).
    pub mean_loss: Vec<f64>,
    /// Grid index of the minimum mean loss (first minimizer on ties).
    pub best_index: usize,
    /// The selected `lambda` (`lambdas[best_index]`).
    pub best_lambda: f64,
}

/// Cross-validates an elastic-net `lambda` grid under a pluggable loss.
///
/// For every split the elastic net is fit on the training rows and used to
/// predict the test rows; the `loss` closure scores those predictions. The
/// per-`lambda` losses are averaged **across folds with equal fold weight**
/// (each fold contributes one loss value regardless of its test size), and
/// the minimizing `lambda` is returned.
///
/// `loss` is any `Fn(y_true, y_pred) -> f64`; the crate ships [`mse`] and
/// [`mae`]. Supplying the leakage-safe splitters from this module
/// ([`expanding_origin_splits`], [`purged_kfold_splits`], ...) is what
/// makes the selection honest — `cv_select` itself is agnostic to how the
/// folds were built.
///
/// # Errors
///
/// * [`MlError::EmptyInput`] if `lambdas` or `splits` is empty;
/// * [`MlError::InvalidArgument`] if any `lambda < 0`, `l1_ratio` is
///   outside `(0, 1]`, or a split references an out-of-range row;
/// * plus every error [`elastic_net`](crate::elastic_net) can raise from
///   the per-fold fits.
pub fn cv_select<L>(
    x: MatRef<'_, f64>,
    y: &[f64],
    splits: &[Split],
    lambdas: &[f64],
    l1_ratio: f64,
    loss: L,
    cd: CoordDescentOptions,
) -> Result<CvResult, MlError>
where
    L: Fn(&[f64], &[f64]) -> f64,
{
    let (n, _p) = check_xy(x, y)?;
    if lambdas.is_empty() {
        return Err(MlError::EmptyInput { what: "lambdas" });
    }
    if splits.is_empty() {
        return Err(MlError::EmptyInput { what: "splits" });
    }
    if !l1_ratio.is_finite() || !(0.0..=1.0).contains(&l1_ratio) || l1_ratio == 0.0 {
        return Err(MlError::InvalidArgument {
            what: "l1_ratio must lie in (0, 1]",
        });
    }
    for &lam in lambdas {
        if !lam.is_finite() || lam < 0.0 {
            return Err(MlError::InvalidArgument {
                what: "every lambda must be finite and non-negative",
            });
        }
    }
    for s in splits {
        if s.train.iter().chain(&s.test).any(|&i| i >= n) {
            return Err(MlError::InvalidArgument {
                what: "a split references a row index outside 0..n",
            });
        }
        if s.train.is_empty() || s.test.is_empty() {
            return Err(MlError::InvalidArgument {
                what: "every split must have a non-empty train and test set",
            });
        }
    }

    let cols = columns(x);
    let mut sum_loss = vec![0.0f64; lambdas.len()];

    for s in splits {
        // Build the training sub-design (columns restricted to train rows)
        // and target once per fold; reused across the whole lambda grid.
        let train_cols: Vec<Vec<f64>> = cols
            .iter()
            .map(|c| s.train.iter().map(|&i| c[i]).collect())
            .collect();
        let y_train: Vec<f64> = s.train.iter().map(|&i| y[i]).collect();
        let y_test: Vec<f64> = s.test.iter().map(|&i| y[i]).collect();

        // Warm-start down the (descending-in-penalty) grid as given.
        let p = cols.len();
        let mut warm = vec![0.0f64; p];
        for (li, &lam) in lambdas.iter().enumerate() {
            let fit = cd_engine(&train_cols, &y_train, lam, l1_ratio, &warm, cd)?;
            // Predict on the test rows.
            let preds: Vec<f64> = s
                .test
                .iter()
                .map(|&i| {
                    fit.coef
                        .iter()
                        .enumerate()
                        .map(|(j, b)| cols[j][i] * b)
                        .sum()
                })
                .collect();
            sum_loss[li] += loss(&y_test, &preds);
            warm = fit.coef;
        }
    }

    let k = splits.len() as f64;
    let mean_loss: Vec<f64> = sum_loss.iter().map(|s| s / k).collect();

    // Argmin.
    let mut best_index = 0usize;
    let mut best_val = f64::INFINITY;
    for (i, &m) in mean_loss.iter().enumerate() {
        if m < best_val {
            best_val = m;
            best_index = i;
        }
    }

    Ok(CvResult {
        lambdas: lambdas.to_vec(),
        mean_loss,
        best_index,
        best_lambda: lambdas[best_index],
    })
}
