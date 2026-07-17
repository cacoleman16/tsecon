//! Static (approximate) factor extraction by principal components.

use tsecon_linalg::faer::{Mat, MatRef};

use crate::error::FavarError;

/// Principal-component factor model of a standardized panel
/// (Stock-Watson approximate factor model; Stock & Watson 2002, JASA).
///
/// Given an `n x N` panel `X` (observations in rows, `N` series in
/// columns) the panel is first standardized column by column to mean 0
/// and standard deviation 1 (population / `ddof = 0` scaling, matching
/// the fixture generator and `numpy.std(..., ddof=0)`):
///
/// ```text
/// Z[:, j] = (X[:, j] - mean_j) / sd_j
/// ```
///
/// A thin singular value decomposition `Z = U S V'` then yields
///
/// ```text
/// principal components (factors)  F = U S      (n x m),
/// loadings                        L = V        (N x m),
/// eigenvalues                     lambda_k = s_k^2 / n,
/// ```
///
/// with `m = min(n, N)`. The `r`-factor model keeps the first `r`
/// columns of `F` and `L`; the reconstruction `F_r L_r'` is the best
/// rank-`r` approximation of `Z` in Frobenius norm (Eckart-Young). The
/// eigenvalues are the sample variances explained by each component and
/// sum to `N` for a standardized panel.
///
/// **Sign convention.** Principal components and loadings are identified
/// only up to a joint sign flip of each column (`F_k -> -F_k`,
/// `L_k -> -L_k` leaves `Z` unchanged). This model fixes the sign so
/// that, in every loading column, the entry of largest absolute value is
/// positive; the same flip is applied to the matching factor column so
/// `F_k L_k'` is preserved. Downstream comparisons that must be
/// sign-free should compare absolute values.
#[derive(Debug, Clone)]
pub struct FactorModel {
    // Full-width principal components `U S` (n x m), sign-fixed.
    scores: Mat<f64>,
    // Full-width loadings `V` (N x m), sign-fixed (same flip as scores).
    loadings: Mat<f64>,
    // Eigenvalues `s_k^2 / n`, descending, length m.
    eigenvalues: Vec<f64>,
    // Singular values `s_k`, descending, length m.
    singular_values: Vec<f64>,
    // Column means used to center the panel (length N).
    center: Vec<f64>,
    // Column standard deviations (ddof = 0) used to scale (length N).
    scale: Vec<f64>,
    // Number of observations n (rows of the panel).
    n: usize,
}

impl FactorModel {
    /// Fits the principal-component factor model to a panel `x`
    /// (`n x N`, observations in rows, oldest first).
    ///
    /// The panel is standardized (population scaling) and decomposed by
    /// the shared `faer` thin SVD. All `m = min(n, N)` components are
    /// retained internally; select an `r`-factor view with
    /// [`FactorModel::factors`] / [`FactorModel::loadings`].
    ///
    /// # Errors
    ///
    /// * [`FavarError::EmptyInput`] if the panel has zero rows or
    ///   columns;
    /// * [`FavarError::InvalidArgument`] if fewer than two observations
    ///   are supplied (variance is undefined for a single row);
    /// * [`FavarError::NonFinite`] on any NaN/infinite entry;
    /// * [`FavarError::ZeroVariance`] if a column is constant;
    /// * [`FavarError::SvdFailed`] if the SVD iteration does not
    ///   converge.
    pub fn fit(x: MatRef<'_, f64>) -> Result<Self, FavarError> {
        let n = x.nrows();
        let n_series = x.ncols();
        if n == 0 || n_series == 0 {
            return Err(FavarError::EmptyInput { what: "panel x" });
        }
        if n < 2 {
            return Err(FavarError::InvalidArgument {
                what: "at least two observations are required to standardize the panel",
            });
        }
        for j in 0..n_series {
            for i in 0..n {
                if !x[(i, j)].is_finite() {
                    return Err(FavarError::NonFinite { what: "panel x" });
                }
            }
        }

        // Column means and population standard deviations.
        let mut center = vec![0.0f64; n_series];
        let mut scale = vec![0.0f64; n_series];
        for j in 0..n_series {
            let mut mean = 0.0;
            for i in 0..n {
                mean += x[(i, j)];
            }
            mean /= n as f64;
            let mut ss = 0.0;
            for i in 0..n {
                let d = x[(i, j)] - mean;
                ss += d * d;
            }
            let var = ss / n as f64;
            let sd = var.sqrt();
            if sd <= 0.0 {
                return Err(FavarError::ZeroVariance { column: j });
            }
            center[j] = mean;
            scale[j] = sd;
        }

        // Standardized panel Z.
        let z = Mat::from_fn(n, n_series, |i, j| (x[(i, j)] - center[j]) / scale[j]);

        let svd = z.thin_svd().map_err(|_| FavarError::SvdFailed)?;
        let u = svd.U();
        let v = svd.V();
        let singular_values: Vec<f64> = svd.S().column_vector().iter().copied().collect();
        let m = singular_values.len();

        // Scores F = U S, loadings L = V.
        let mut scores = Mat::from_fn(n, m, |i, k| u[(i, k)] * singular_values[k]);
        let mut loadings = Mat::from_fn(n_series, m, |i, k| v[(i, k)]);

        // Sign convention: largest-magnitude loading positive per column.
        for k in 0..m {
            let mut best = 0usize;
            let mut best_abs = 0.0f64;
            for i in 0..n_series {
                let a = loadings[(i, k)].abs();
                if a > best_abs {
                    best_abs = a;
                    best = i;
                }
            }
            if loadings[(best, k)] < 0.0 {
                for i in 0..n_series {
                    loadings[(i, k)] = -loadings[(i, k)];
                }
                for i in 0..n {
                    scores[(i, k)] = -scores[(i, k)];
                }
            }
        }

        let eigenvalues: Vec<f64> = singular_values.iter().map(|s| s * s / n as f64).collect();

        Ok(Self {
            scores,
            loadings,
            eigenvalues,
            singular_values,
            center,
            scale,
            n,
        })
    }

    /// Number of observations `n`.
    pub fn n_obs(&self) -> usize {
        self.n
    }

    /// Number of observed series `N` in the panel.
    pub fn n_series(&self) -> usize {
        self.center.len()
    }

    /// Maximum number of components `m = min(n, N)`.
    pub fn max_factors(&self) -> usize {
        self.eigenvalues.len()
    }

    /// Eigenvalues `lambda_k = s_k^2 / n` in descending order (length
    /// `m`); the variances explained by each principal component. These
    /// feed the [`crate::criteria`] factor-number rules.
    pub fn eigenvalues(&self) -> &[f64] {
        &self.eigenvalues
    }

    /// Singular values `s_k` of the standardized panel, descending
    /// (length `m`).
    pub fn singular_values(&self) -> &[f64] {
        &self.singular_values
    }

    /// Column means used to center the panel (length `N`).
    pub fn center(&self) -> &[f64] {
        &self.center
    }

    /// Column standard deviations (`ddof = 0`) used to scale the panel
    /// (length `N`).
    pub fn scale(&self) -> &[f64] {
        &self.scale
    }

    /// The first `r` principal components `F_r = U_r S_r` (`n x r`); the
    /// estimated common factors, sign-fixed by the model's convention.
    ///
    /// # Errors
    ///
    /// [`FavarError::InvalidFactorCount`] if `r == 0` or `r` exceeds
    /// [`FactorModel::max_factors`].
    pub fn factors(&self, r: usize) -> Result<Mat<f64>, FavarError> {
        self.check_r(r)?;
        Ok(Mat::from_fn(self.n, r, |i, k| self.scores[(i, k)]))
    }

    /// The first `r` loading columns `L_r = V_r` (`N x r`); entry
    /// `(i, k)` is the loading of series `i` on factor `k`.
    ///
    /// # Errors
    ///
    /// [`FavarError::InvalidFactorCount`] if `r == 0` or `r` exceeds
    /// [`FactorModel::max_factors`].
    pub fn loadings(&self, r: usize) -> Result<Mat<f64>, FavarError> {
        self.check_r(r)?;
        Ok(Mat::from_fn(self.n_series(), r, |i, k| {
            self.loadings[(i, k)]
        }))
    }

    /// Rank-`r` reconstruction of the *standardized* panel,
    /// `Z_hat = F_r L_r'` (`n x N`) — the best rank-`r` Frobenius
    /// approximation of the standardized data (Eckart-Young). The
    /// idiosyncratic residual `Z - Z_hat` is small precisely when the
    /// panel is well described by `r` common factors.
    ///
    /// # Errors
    ///
    /// [`FavarError::InvalidFactorCount`] as [`FactorModel::factors`].
    pub fn reconstruct_standardized(&self, r: usize) -> Result<Mat<f64>, FavarError> {
        self.check_r(r)?;
        let n_series = self.n_series();
        Ok(Mat::from_fn(self.n, n_series, |i, j| {
            (0..r)
                .map(|k| self.scores[(i, k)] * self.loadings[(j, k)])
                .sum()
        }))
    }

    fn check_r(&self, r: usize) -> Result<(), FavarError> {
        if r == 0 || r > self.max_factors() {
            return Err(FavarError::InvalidFactorCount {
                what: "number of factors must satisfy 1 <= r <= min(n, N)",
                requested: r,
                max: self.max_factors(),
            });
        }
        Ok(())
    }
}
