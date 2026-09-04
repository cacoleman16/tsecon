//! Componentwise `L2` boosting with single-column least-squares base
//! learners (Bühlmann & Yu 2003; Bühlmann 2006) — the `glmboost` engine —
//! with the corrected-AIC stopping rule and an exact, matrix-free trace of
//! the boosting operator.
//!
//! # Algorithm
//!
//! With the `n x p` design `X` (columns `x_j`, no intercept — pass a
//! centered `y` and centered/standardized columns, as everywhere in this
//! crate), learning rate `nu ∈ (0, 1]`, and `F_0 = 0`, step `m` does
//!
//! ```text
//! U      = y - F_{m-1}                               (current residual)
//! b_j    = x_j' U / x_j' x_j           for every j   (LS fit of U on x_j)
//! j*     = argmin_j ||U - x_j b_j||^2 = argmax_j (x_j' U)^2 / x_j' x_j
//! F_m    = F_{m-1} + nu * x_{j*} b_{j*},   coef_{j*} += nu * b_{j*}.
//! ```
//!
//! Ties in the selection are broken toward the smallest column index, and
//! nothing is random: the same inputs give the same `selected` sequence
//! bit for bit (the procedure is *seedless*). Zero-norm (constant, after
//! centering) columns are never selectable.
//!
//! # The boosting operator and its trace
//!
//! `F_m = B_m y` for the linear *boosting operator*
//!
//! ```text
//! B_m = B_{m-1} + nu * H_{j*} (I - B_{m-1}),     H_j = x_j x_j' / x_j' x_j,
//! ```
//!
//! whose trace is the effective degrees of freedom Bühlmann (2006) plugs
//! into the corrected AIC
//!
//! ```text
//! AIC_c(m) = log( ||y - F_m||^2 / n ) + (1 + tr(B_m)/n) / (1 - (tr(B_m) + 2)/n).
//! ```
//!
//! Rather than form the `n x n` matrix, this implementation keeps `B_m`
//! in the exact rank-`m` factored form
//!
//! ```text
//! B_m = sum_{i=1}^{m} nu * x_{j_i} w_i',    w_i = (I - B_{i-1})' x_{j_i} / x_{j_i}' x_{j_i},
//! ```
//!
//! where `B_{i-1}' x_{j_i} = sum_{l<i} nu * w_l (x_{j_l}' x_{j_i})` needs
//! only the `p x p` Gram matrix and the stored `w`s. `tr(B_m) =
//! sum_i nu * w_i' x_{j_i}` is then **exact in exact arithmetic** — it is
//! the same quantity a dense `n x n` update would produce, evaluated in a
//! different order, so the two agree to rounding (the golden test pins it
//! at `1e-12` against a dense NumPy transcription). Cost is `O(n m)` per
//! step and `O(n M)` memory for `M` steps, in place of `O(n^2)` each.
//! Nothing here is approximate; what is *not* done is materializing
//! `B_m` itself, so the fit at the selected step is recomputed as
//! `X coef` rather than read off a stored operator.
//!
//! # Stopping
//!
//! [`BoostStop::Aic`] returns the step minimizing `AIC_c` over
//! `m = 1..=n_steps` (the first minimum on ties); [`BoostStop::None`]
//! returns the final step. The paths are computed either way. Where
//! `tr(B_m) + 2 >= n` the corrected-AIC denominator is not positive and
//! that entry is `+inf` (never selected).
//!
//! References: Bühlmann, P. & Yu, B. (2003), "Boosting with the L2
//! loss: regression and classification", *JASA* 98(462); Bühlmann, P.
//! (2006), "Boosting for high-dimensional linear models", *Annals of
//! Statistics* 34(2); Hofner, Mayr, Robinzonov & Schmid (2014), "Model-based
//! boosting in R: a hands-on tutorial using the R package mboost",
//! *Computational Statistics* 29; Ng, S. (2014), "Viewpoint: Boosting
//! recessions", *Canadian Journal of Economics* 47(1).

use tsecon_linalg::faer::MatRef;

use crate::error::MlError;
use crate::util::{check_xy, columns, dot};

/// Early-stopping rule for [`boosting`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoostStop {
    /// Return the step minimizing Bühlmann's (2006) corrected AIC.
    Aic,
    /// Run all `n_steps` and return the last one.
    None,
}

/// Configuration of [`boosting`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoostingOptions {
    /// Step-length shrinkage `nu ∈ (0, 1]`; `0.1` is the conventional
    /// slow-learning choice, `1.0` the unshrunk greedy fit.
    pub learning_rate: f64,
    /// Number of boosting iterations to run (`>= 1`).
    pub n_steps: usize,
    /// Which step is reported as `coef` / `fitted` / `predicted`.
    pub stop: BoostStop,
}

impl Default for BoostingOptions {
    fn default() -> Self {
        Self {
            learning_rate: 0.1,
            n_steps: 500,
            stop: BoostStop::Aic,
        }
    }
}

/// Result of [`boosting`].
#[derive(Debug, Clone, PartialEq)]
pub struct BoostingFit {
    /// Coefficients at `best_step`, length `p`.
    pub coef: Vec<f64>,
    /// Coefficient vector after each step: `n_steps` rows of length `p`
    /// (row `m` is the model after `m + 1` boosting iterations).
    pub coef_path: Vec<Vec<f64>>,
    /// Column selected at each step, length `n_steps`.
    pub selected: Vec<usize>,
    /// `||y - F_m||^2` after each step (nonincreasing).
    pub rss_path: Vec<f64>,
    /// `tr(B_m)` after each step — the effective degrees of freedom.
    pub df_path: Vec<f64>,
    /// Corrected AIC after each step.
    pub aic_path: Vec<f64>,
    /// 0-based index into the path arrays of the reported model.
    pub best_step: usize,
    /// `X coef`, length `n`.
    pub fitted: Vec<f64>,
    /// `X_test coef` when `x_test` was supplied.
    pub predicted: Option<Vec<f64>>,
}

/// Componentwise `L2` boosting of `y` on the columns of `x`; see the
/// [module docs](self).
///
/// `x_test` (optional, `n_test x p`) is scored with the `best_step`
/// coefficients into `predicted`.
///
/// # Errors
///
/// * [`MlError::InsufficientData`] if `x` has fewer than three rows;
/// * [`MlError::EmptyInput`] if `x` has no columns;
/// * [`MlError::DimensionMismatch`] if `y.len() != x.nrows()` or
///   `x_test.ncols() != x.ncols()`;
/// * [`MlError::NonFinite`] on a NaN/infinite entry of `x`, `y`, or
///   `x_test`;
/// * [`MlError::InvalidArgument`] if `learning_rate` is outside `(0, 1]`,
///   `n_steps == 0`, or every column of `x` has zero norm.
pub fn boosting(
    x: MatRef<'_, f64>,
    y: &[f64],
    opts: BoostingOptions,
    x_test: Option<MatRef<'_, f64>>,
) -> Result<BoostingFit, MlError> {
    let n = x.nrows();
    if n < 3 {
        return Err(MlError::InsufficientData {
            needed: 3,
            got: n,
            what: "boosting",
        });
    }
    let (n, p) = check_xy(x, y)?;
    if let Some(xt) = x_test {
        if xt.ncols() != p {
            return Err(MlError::DimensionMismatch {
                what: "x_test must have the same number of columns as x",
                expected: p,
                got: xt.ncols(),
            });
        }
        for j in 0..p {
            for i in 0..xt.nrows() {
                if !xt[(i, j)].is_finite() {
                    return Err(MlError::NonFinite { what: "x_test" });
                }
            }
        }
    }
    let nu = opts.learning_rate;
    if !nu.is_finite() || nu <= 0.0 || nu > 1.0 {
        return Err(MlError::InvalidArgument {
            what: "learning_rate must lie in (0, 1] (0.1 is the conventional slow-learning \
                   choice; 1.0 is the unshrunk greedy fit)",
        });
    }
    let steps = opts.n_steps;
    if steps == 0 {
        return Err(MlError::InvalidArgument {
            what: "n_steps must be at least 1",
        });
    }

    let cols = columns(x);
    let norm2: Vec<f64> = cols.iter().map(|c| dot(c, c)).collect();
    if norm2.iter().all(|&v| v == 0.0) {
        return Err(MlError::InvalidArgument {
            what: "every column of x has zero norm, so no base learner can be fit \
                   (center/standardize x and drop constant columns)",
        });
    }
    // p x p Gram matrix for the operator bookkeeping.
    let gram: Vec<Vec<f64>> = (0..p)
        .map(|a| (0..p).map(|b| dot(&cols[a], &cols[b])).collect())
        .collect();

    let nf = n as f64;
    let mut coef = vec![0.0; p];
    let mut fit = vec![0.0; n];
    let mut resid = y.to_vec();
    // Factored operator: B_m = sum_i nu * x_{j_i} w_i'.
    let mut ws: Vec<Vec<f64>> = Vec::with_capacity(steps);
    let mut trace = 0.0f64;

    let mut coef_path: Vec<Vec<f64>> = Vec::with_capacity(steps);
    let mut selected: Vec<usize> = Vec::with_capacity(steps);
    let mut rss_path: Vec<f64> = Vec::with_capacity(steps);
    let mut df_path: Vec<f64> = Vec::with_capacity(steps);
    let mut aic_path: Vec<f64> = Vec::with_capacity(steps);

    for _ in 0..steps {
        // Componentwise selection: best single-column LS fit of the residual.
        let mut best_j = usize::MAX;
        let mut best_score = f64::NEG_INFINITY;
        let mut best_xr = 0.0;
        for j in 0..p {
            if norm2[j] == 0.0 {
                continue;
            }
            let xr = dot(&cols[j], &resid);
            let score = xr * xr / norm2[j];
            if score > best_score {
                best_score = score;
                best_j = j;
                best_xr = xr;
            }
        }
        let j = best_j;
        let b = best_xr / norm2[j];
        let step_coef = nu * b;
        coef[j] += step_coef;
        for i in 0..n {
            fit[i] += step_coef * cols[j][i];
            resid[i] = y[i] - fit[i];
        }

        // w_m = (x_j - B_{m-1}' x_j) / ||x_j||^2, with
        // B_{m-1}' x_j = sum_l nu * w_l * gram[j_l][j].
        let mut w = cols[j].clone();
        for (l, wl) in ws.iter().enumerate() {
            let s = nu * gram[selected[l]][j];
            if s != 0.0 {
                for i in 0..n {
                    w[i] -= s * wl[i];
                }
            }
        }
        for v in w.iter_mut() {
            *v /= norm2[j];
        }
        trace += nu * dot(&w, &cols[j]);
        ws.push(w);

        let rss = dot(&resid, &resid);
        let denom = 1.0 - (trace + 2.0) / nf;
        let aic = if denom > 0.0 {
            (rss / nf).ln() + (1.0 + trace / nf) / denom
        } else {
            f64::INFINITY
        };
        coef_path.push(coef.clone());
        selected.push(j);
        rss_path.push(rss);
        df_path.push(trace);
        aic_path.push(aic);
    }

    let best_step = match opts.stop {
        BoostStop::None => steps - 1,
        BoostStop::Aic => {
            let mut best = 0usize;
            for (m, &a) in aic_path.iter().enumerate() {
                if a < aic_path[best] {
                    best = m;
                }
            }
            best
        }
    };
    let coef_best = coef_path[best_step].clone();
    let fitted: Vec<f64> = (0..n)
        .map(|i| (0..p).map(|j| cols[j][i] * coef_best[j]).sum())
        .collect();
    let predicted = x_test.map(|xt| {
        (0..xt.nrows())
            .map(|i| (0..p).map(|j| xt[(i, j)] * coef_best[j]).sum())
            .collect()
    });

    Ok(BoostingFit {
        coef: coef_best,
        coef_path,
        selected,
        rss_path,
        df_path,
        aic_path,
        best_step,
        fitted,
        predicted,
    })
}
