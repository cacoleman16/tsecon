//! # tsecon-ml — penalized regression and leakage-safe cross-validation
//!
//! The machine-learning slice of the `tsecon` library (roadmap Module 10,
//! Tier 1). It owns the penalized-regression solver stack and the
//! dependence-aware model-selection machinery that keeps time-series
//! evaluation honest, matching the scikit-learn objectives exactly so its
//! numbers are drop-in comparable:
//!
//! * [`ridge`] — ridge regression in closed form via the thin SVD,
//!   minimizing scikit-learn's `||y - Xb||^2 + alpha*||b||^2` (no `1/n`
//!   factor — read the objective note below);
//! * [`elastic_net`] / [`lasso`] — cyclical coordinate descent with
//!   covariance-style residual updates and a *glmnet* active-set strategy,
//!   minimizing `(1/(2n))||y - Xb||^2 + alpha*l1_ratio*||b||_1 +
//!   0.5*alpha*(1-l1_ratio)*||b||^2` (Friedman, Hastie & Tibshirani 2010);
//! * [`adaptive_lasso`] — Zou's (2006) weighted-`L1` penalty folded in by
//!   feature rescaling, so true zeros are driven out more reliably;
//! * [`regularization_path`] — the full elastic-net path from `lambda_max`
//!   down three decades, with AIC/BIC tuning on the number of nonzeros
//!   (Zou, Hastie & Tibshirani 2007);
//! * [`expanding_origin_splits`] / [`rolling_origin_splits`] /
//!   [`purged_kfold_splits`] and [`cv_select`] — time-series
//!   cross-validation with purging and embargo (Lopez de Prado 2018),
//!   plus a pluggable-loss selection driver;
//! * [`Scaler`] / [`TargetCenterer`] — standardization that is fit on the
//!   training rows and replayed on test rows, with a loud warning against
//!   standardizing before splitting.
//!
//! ## Two objective conventions (read before choosing `alpha`)
//!
//! scikit-learn scales ridge and the `L1` family **differently**, and this
//! crate matches each namesake exactly, so the two `alpha` scales are not
//! interchangeable:
//!
//! * **Ridge** minimizes `||y - Xb||^2 + alpha*||b||^2` — *no* `1/n`.
//! * **LASSO / elastic net** minimize
//!   `(1/(2n))||y - Xb||^2 + alpha*l1_ratio*||b||_1 +
//!   0.5*alpha*(1-l1_ratio)*||b||^2`.
//!
//! The golden fixture `fixtures/ml.json` records both in its
//! `_meta.objective_note`. No routine fits an intercept: callers pass a
//! centered `y` and centered (typically standardized) design columns, which
//! is exactly what [`Scaler`] and [`TargetCenterer`] produce.
//!
//! All fallible routines return [`MlError`]; nothing in this crate's
//! non-test code path panics on user input.

#![warn(missing_docs)]
#![warn(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

pub mod coordinate_descent;
pub mod cv;
pub mod error;
pub mod forest;
pub mod group_lasso;
pub mod importance;
pub mod kernel_regression;
pub mod kernel_ridge;
pub mod path;
pub mod pds;
pub mod post_lasso;
pub mod ridge;
pub mod standardize;
pub mod tree;
mod util;

pub use coordinate_descent::{
    adaptive_lasso, elastic_net, lasso, CoordDescentOptions, PenalizedFit,
};
pub use cv::{
    cv_select, expanding_origin_splits, mae, mse, purged_kfold_splits, rolling_origin_splits,
    CvResult, Loss, Split,
};
pub use error::MlError;
pub use forest::{
    random_forest, resample_indices, ForestFit, ForestOptions, Importance, MaxFeatures, Resampling,
};
pub use group_lasso::{group_lasso, group_lasso_alpha_max, GroupLassoFit, GroupWeights};
pub use kernel_regression::{
    cv_criterion, kernel_regression, BandwidthSpec, KernelRegressionFit, KernelRegressionOptions,
    RegressionKernel, RegressionKind,
};
pub use kernel_ridge::{
    kernel_matrix, kernel_ridge, KernelRidgeFit, KernelRidgeOptions, KernelType,
};
pub use path::{regularization_path, PathOptions, RegPath};
pub use pds::{pds_lasso, PdsAlpha, PdsFit};
pub use post_lasso::{post_lasso, PostLassoFit};
pub use ridge::ridge;
pub use standardize::{Scaler, TargetCenterer};
pub use tree::{regression_tree, RegressionTree, TreeFit, TreeOptions};

// Re-export the dense backend (through tsecon-linalg) so callers, tests,
// and doctests construct `Mat`/`MatRef` inputs against one faer version.
pub use tsecon_linalg::faer;
