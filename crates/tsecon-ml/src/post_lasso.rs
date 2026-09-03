//! Post-LASSO ordinary least squares (Belloni & Chernozhukov 2013).
//!
//! The LASSO buys its sparsity with shrinkage: every retained coefficient is
//! pulled toward zero by the penalty, so the point estimates are biased
//! toward zero even on the true support. Belloni & Chernozhukov (2013,
//! *Bernoulli* 19(2)) show that refitting OLS on the LASSO-selected columns
//! — the *post-LASSO* estimator — removes that shrinkage bias while
//! retaining the LASSO's rate of convergence, which is why the refit is the
//! standard object to read economically.
//!
//! # No standard errors, deliberately
//!
//! This routine returns **no** standard errors. OLS standard errors
//! computed on a data-selected support are not valid: the selection event
//! is a function of the same sample, so the refit's sampling distribution
//! is not the textbook Gaussian centered at the truth (Leeb & Pötscher
//! 2005), and `n - |S|` residual degrees of freedom overstate what is left
//! after searching over `p` columns. Reporting them would invite exactly
//! the error the roadmap's implementation warnings forbid. For inference
//! on a low-dimensional target coefficient use [`crate::pds_lasso`]
//! (post-double-selection), whose Neyman-orthogonal construction is robust
//! to selection mistakes.

use tsecon_linalg::faer::{Mat, MatRef};

use crate::coordinate_descent::{elastic_net, CoordDescentOptions};
use crate::error::MlError;
use crate::ridge::ols_svd;
use crate::util::{check_xy, columns, dot};

/// Result of a post-LASSO OLS refit.
#[derive(Debug, Clone, PartialEq)]
pub struct PostLassoFit {
    /// Column indices selected by the LASSO (nonzero coefficients),
    /// ascending.
    pub support: Vec<usize>,
    /// The first-stage LASSO / elastic-net coefficients, length `p`.
    pub coef_lasso: Vec<f64>,
    /// The OLS refit on `support`, length `p`, exactly zero off-support.
    pub coef_ols: Vec<f64>,
    /// `support.len()`.
    pub n_selected: usize,
    /// Residual sum of squares `||y - X coef_ols||^2` of the refit.
    pub rss: f64,
}

/// Post-LASSO OLS: fit the LASSO (or elastic net, for `l1_ratio < 1`) with
/// the crate's scikit-learn objective, take its nonzero support, and refit
/// ordinary least squares on those columns alone.
///
/// `x` is the `n x p` design (no intercept; center/standardize first), `y`
/// the centered target, `alpha` and `l1_ratio` the elastic-net penalty
/// exactly as in [`crate::elastic_net`], and `opts` the coordinate-descent
/// stopping controls. The refit is the minimum-norm least-squares solution
/// on the selected columns (thin SVD), so a collinear support does not
/// fail — it returns the minimum-norm refit.
///
/// Returns no standard errors: see the [module docs](self) for why, and
/// [`crate::pds_lasso`] for valid post-selection inference.
///
/// # Errors
///
/// * As [`crate::elastic_net`] for the first stage;
/// * [`MlError::InsufficientData`] if the LASSO selects at least as many
///   columns as there are observations (the refit would interpolate).
pub fn post_lasso(
    x: MatRef<'_, f64>,
    y: &[f64],
    alpha: f64,
    l1_ratio: f64,
    opts: CoordDescentOptions,
) -> Result<PostLassoFit, MlError> {
    let (n, p) = check_xy(x, y)?;
    let first = elastic_net(x, y, alpha, l1_ratio, opts)?;
    let support: Vec<usize> = (0..p).filter(|&j| first.coef[j] != 0.0).collect();
    let n_selected = support.len();
    if n_selected >= n {
        return Err(MlError::InsufficientData {
            got: n,
            needed: n_selected + 1,
            what: "the post-LASSO OLS refit needs more observations than selected \
                   columns; raise alpha to select fewer",
        });
    }

    let cols = columns(x);
    let mut coef_ols = vec![0.0f64; p];
    let mut resid = y.to_vec();
    if n_selected > 0 {
        let xs = Mat::from_fn(n, n_selected, |i, k| cols[support[k]][i]);
        let b = ols_svd(xs.as_ref(), y)?;
        for (k, &j) in support.iter().enumerate() {
            coef_ols[j] = b[k];
            for (ri, &xij) in resid.iter_mut().zip(&cols[j]) {
                *ri -= xij * b[k];
            }
        }
    }
    let rss = dot(&resid, &resid);
    Ok(PostLassoFit {
        support,
        coef_lasso: first.coef,
        coef_ols,
        n_selected,
        rss,
    })
}
