//! Functional principal components of a panel of curve observations.

use tsecon_linalg::faer::{Mat, Side};

use crate::error::FuncShockError;

/// Functional principal-component decomposition of a `T x M` curve panel;
/// produced by [`functional_pca`].
///
/// The decomposition writes each demeaned curve as
/// `X_t(m) - mean(m) ~= sum_k s_{t,k} phi_k(m)`: `eigenfunctions[k]` is
/// `phi_k` on the grid, `scores[t][k]` is `s_{t,k}`, and the eigenvalues are
/// the sample variances of the scores (population divisor `T`).
#[derive(Debug, Clone)]
pub struct Fpca {
    /// The mean curve on the grid (length `M`).
    pub mean_curve: Vec<f64>,
    /// The leading `K` eigenfunctions; `eigenfunctions[k]` has length `M`.
    /// Orthonormal in the discrete (Euclidean) inner product on the grid,
    /// each sign-fixed so its entry of largest absolute value is positive
    /// (first index on ties — the numpy `argmax` convention the golden
    /// fixture uses).
    pub eigenfunctions: Vec<Vec<f64>>,
    /// Scores `s_{t,k} = <X_t - mean, phi_k>`; `scores[t]` has length `K`.
    pub scores: Vec<Vec<f64>>,
    /// The leading `K` eigenvalues of the covariance, descending.
    pub eigenvalues: Vec<f64>,
    /// Explained-variance shares `lambda_k / trace(cov)` (length `K`).
    pub explained: Vec<f64>,
    /// Total variance `trace(cov)` — the sum of ALL `M` eigenvalues, so the
    /// explained shares are shares of everything, not of the kept `K`.
    pub total_variance: f64,
}

/// Functional PCA of a `T x M` panel of curve observations (e.g. daily
/// yield-curve changes on a maturity grid): demean, eigendecompose the
/// `M x M` covariance `cov = Xc' Xc / T` (population divisor `T`; `faer`
/// self-adjoint eigensolver through `tsecon-linalg`), and keep the leading
/// `n_factors` eigenpairs.
///
/// `curves[t]` is the curve observed at time `t` on a shared grid of `M`
/// points. The inner product is the discrete (Euclidean) one on the grid —
/// no quadrature weights — matching the discretized implementation of
/// Inoue & Rossi (2021) and the numpy reference in the golden fixture.
///
/// **Sign convention.** Eigenvectors are identified only up to sign. Each
/// eigenfunction's sign is fixed so that its entry of largest absolute value
/// is positive (ties broken by the first such index), and the matching score
/// column inherits the same flip, so the reconstruction
/// `sum_k s_{t,k} phi_k` is unchanged. The golden fixture applies the
/// identical rule, which is what makes the pin well-defined.
///
/// **Rank caveat.** The covariance of `T` curves has rank at most
/// `min(T - 1, M)`; eigenfunctions attached to (numerically) zero
/// eigenvalues are arbitrary orthonormal completions and should not be
/// interpreted. Keep `n_factors` within the rank in short samples.
///
/// # Errors
///
/// * [`FuncShockError::EmptyInput`] for an empty panel or empty rows;
/// * [`FuncShockError::DimensionMismatch`] if fewer than 2 curves are
///   supplied (a covariance needs variation over time);
/// * [`FuncShockError::RaggedRow`] if the rows have unequal lengths;
/// * [`FuncShockError::NonFinite`] on NaN/infinite entries;
/// * [`FuncShockError::InvalidFactorCount`] unless
///   `1 <= n_factors <= M`;
/// * [`FuncShockError::ZeroVariance`] if every curve is identical;
/// * [`FuncShockError::EigenFailed`] if the eigensolver does not converge.
pub fn functional_pca(curves: &[Vec<f64>], n_factors: usize) -> Result<Fpca, FuncShockError> {
    let t = curves.len();
    if t == 0 {
        return Err(FuncShockError::EmptyInput {
            what: "curves (T x M panel)",
        });
    }
    let m = curves[0].len();
    if m == 0 {
        return Err(FuncShockError::EmptyInput {
            what: "curves (each row must have at least one grid point)",
        });
    }
    for (row, c) in curves.iter().enumerate() {
        if c.len() != m {
            return Err(FuncShockError::RaggedRow {
                what: "curves",
                row,
                expected: m,
                got: c.len(),
            });
        }
        if c.iter().any(|v| !v.is_finite()) {
            return Err(FuncShockError::NonFinite { what: "curves" });
        }
    }
    if t < 2 {
        return Err(FuncShockError::DimensionMismatch {
            what: "curves: a covariance needs at least 2 observations (rows)",
            expected: 2,
            got: t,
        });
    }
    if n_factors == 0 || n_factors > m {
        return Err(FuncShockError::InvalidFactorCount {
            requested: n_factors,
            max: m,
        });
    }

    // Mean curve and demeaned panel.
    let tf = t as f64;
    let mut mean_curve = vec![0.0_f64; m];
    for c in curves {
        for (acc, v) in mean_curve.iter_mut().zip(c.iter()) {
            *acc += v;
        }
    }
    for v in &mut mean_curve {
        *v /= tf;
    }
    let xc: Vec<Vec<f64>> = curves
        .iter()
        .map(|c| {
            c.iter()
                .zip(mean_curve.iter())
                .map(|(v, mu)| v - mu)
                .collect()
        })
        .collect();

    // Covariance cov = Xc' Xc / T (M x M, population divisor).
    let mut cov = Mat::<f64>::zeros(m, m);
    for row in &xc {
        for i in 0..m {
            let ri = row[i];
            for j in 0..=i {
                cov[(i, j)] += ri * row[j];
            }
        }
    }
    for i in 0..m {
        for j in 0..=i {
            let v = cov[(i, j)] / tf;
            cov[(i, j)] = v;
            cov[(j, i)] = v;
        }
    }

    let total_variance: f64 = (0..m).map(|i| cov[(i, i)]).sum();
    if total_variance <= 0.0 {
        return Err(FuncShockError::ZeroVariance);
    }

    // Symmetric eigendecomposition; faer returns eigenvalues nondecreasing.
    let eig = cov
        .self_adjoint_eigen(Side::Lower)
        .map_err(|_| FuncShockError::EigenFailed)?;
    let u = eig.U();
    let lambda: Vec<f64> = eig.S().column_vector().iter().copied().collect();

    // Leading K pairs in DESCENDING eigenvalue order, sign-fixed.
    let mut eigenvalues = Vec::with_capacity(n_factors);
    let mut eigenfunctions = Vec::with_capacity(n_factors);
    for k in 0..n_factors {
        let col = m - 1 - k;
        eigenvalues.push(lambda[col]);
        let mut phi: Vec<f64> = (0..m).map(|i| u[(i, col)]).collect();
        // Sign convention: largest-|.| entry positive, first index on ties.
        let mut best = 0usize;
        let mut best_abs = 0.0_f64;
        for (i, v) in phi.iter().enumerate() {
            let a = v.abs();
            if a > best_abs {
                best_abs = a;
                best = i;
            }
        }
        if phi[best] < 0.0 {
            for v in &mut phi {
                *v = -*v;
            }
        }
        eigenfunctions.push(phi);
    }

    // Scores s_{t,k} = <xc_t, phi_k>.
    let scores: Vec<Vec<f64>> = xc
        .iter()
        .map(|row| {
            eigenfunctions
                .iter()
                .map(|phi| row.iter().zip(phi.iter()).map(|(x, p)| x * p).sum())
                .collect()
        })
        .collect();

    let explained: Vec<f64> = eigenvalues.iter().map(|l| l / total_variance).collect();

    Ok(Fpca {
        mean_curve,
        eigenfunctions,
        scores,
        eigenvalues,
        explained,
        total_variance,
    })
}
