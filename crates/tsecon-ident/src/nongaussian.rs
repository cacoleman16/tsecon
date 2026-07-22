//! Non-Gaussian / independent-component structural VAR identification.
//!
//! Point-identify the structural impact matrix `B` of an SVAR from the
//! reduced-form residuals alone, without sign, zero, long-run, or external-
//! instrument restrictions, by exploiting the **statistical independence and
//! non-Gaussianity** of the structural shocks (Lanne, Meitz & Saikkonen 2017;
//! Gourieroux, Monfort & Renne 2017; the econometric reading of Hyvarinen's
//! FastICA, Hyvarinen 1999 / Hyvarinen & Oja 2000).
//!
//! # The model and why it identifies
//!
//! The reduced-form innovations factor as `u_t = B eps_t`, where the
//! structural shocks `eps_t` are **mutually independent** and **non-Gaussian**
//! (at most one component may be Gaussian). Second moments alone leave `B`
//! identified only up to an orthogonal rotation: any `B Q` with `Q Q' = I`
//! reproduces `Sigma_u = B B'`. That is exactly the rotational indeterminacy
//! the sign / zero / proxy schemes elsewhere in this crate resolve with
//! *economic* restrictions. Here the **higher-order** moments break it: for
//! independent non-Gaussian sources the rotation that makes the recovered
//! components maximally independent (equivalently, maximally non-Gaussian by
//! the central-limit heuristic) is unique up to column **sign** and
//! **permutation** — pure conventions, fixed below by a deterministic rule.
//! This is *statistical* identification: it needs no cross-equation
//! restriction, but it **fails when the shocks are Gaussian**, and any column
//! whose shock is close to Gaussian is only weakly identified.
//!
//! # Algorithm (deterministic, reproducible)
//!
//! 1. **Whiten.** With the symmetric inverse square root `W = Sigma_u^{-1/2}`
//!    (from the eigendecomposition of `Sigma_u`), form `z_t = W u_t` so
//!    `Cov(z) = I`. Working in whitened coordinates turns the search for `B`
//!    into a search over *orthogonal* rotations.
//! 2. **Rotate for independence.** Find the orthogonal `W_ica` maximizing a
//!    non-Gaussianity contrast on the rows of `W_ica z`, via the **symmetric
//!    (parallel) FastICA fixed point** with the standard log-cosh contrast
//!    `g(u) = tanh(u)` (Hyvarinen). The iteration is seedless and starts from
//!    the identity, so it is bit-reproducible; the symmetric-decorrelation
//!    step `W <- (W W')^{-1/2} W` keeps the rows orthonormal.
//! 3. **Unwhiten.** The whitened rotation is `Q = W_ica'`, and the structural
//!    impact matrix is `B = W^{-1} Q = Sigma_u^{1/2} Q`, which satisfies
//!    `B B' = Sigma_u` exactly (`Q` orthogonal).
//! 4. **Fix the conventions.** Order the columns of `B` by a stable rule
//!    ([`OrderBy::Kurtosis`], descending absolute excess kurtosis — most
//!    non-Gaussian, hence most strongly identified, first; or
//!    [`OrderBy::ColumnNorm`]) and sign each column so its largest-magnitude
//!    entry is positive.
//!
//! Structural IRFs are `Theta_h = Psi_h B` through the shared general-impact
//! MA helper ([`crate::summary::structural_ma`]); `Theta_0 = B`.
//!
//! # Diagnostics and honesty
//!
//! [`NonGaussianSvar::shock_kurtosis`] reports the recovered shocks' excess
//! kurtosis in the identified order. A value near zero flags a **near-Gaussian,
//! weakly identified** column: the rotation is poorly pinned in that direction
//! and neither its sign nor its loadings should be trusted. Column order and
//! sign are **conventions**, not economics — map a recovered shock to a named
//! structural shock only by inspecting its impact loadings. This estimator is
//! only as good as the independence and non-Gaussianity of the true shocks.

use core::cmp::Ordering;

use tsecon_linalg::faer::{Mat, MatRef, Side};
use tsecon_linalg::LinalgError;

use crate::error::IdentError;
use crate::summary::structural_ma;

/// Non-Gaussianity contrast driving the FastICA fixed point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Contrast {
    /// The log-cosh contrast `G(u) = ln cosh(u)`, `g(u) = tanh(u)`
    /// (Hyvarinen's general-purpose robust default).
    LogCosh,
}

/// Stable rule for ordering the identified structural shocks (columns of `B`),
/// whose order is otherwise only a convention.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderBy {
    /// Descending **absolute excess kurtosis** of the recovered shocks: the
    /// most non-Gaussian (most strongly identified) shock first, near-Gaussian
    /// (weakly identified) columns last. Ties break by ascending raw index.
    Kurtosis,
    /// Descending Euclidean norm of the impact column of `B`. Ties break by
    /// ascending raw index.
    ColumnNorm,
}

/// A non-Gaussian / independent-component SVAR identification.
///
/// Layout convention (shared with [`crate::structural_fevd`] and
/// `tsecon.var_irf`): `impact[(i, j)]` and `irf[h][(i, j)]` are the response of
/// variable `i` to structural shock `j` (at impact / horizon `h`).
#[derive(Debug, Clone, PartialEq)]
pub struct NonGaussianSvar {
    /// Structural impact matrix `B` (`n x n`, `B B' = Sigma_u`); column `j` is
    /// the one-standard-deviation impact vector of shock `j`. Columns are
    /// ordered and signed by the chosen conventions.
    pub impact: Mat<f64>,
    /// Structural impulse responses `Theta_h = Psi_h B`, length `horizon + 1`,
    /// with `Theta_0 = impact`.
    pub irf: Vec<Mat<f64>>,
    /// Whitened rotation `Q = W_ica'` with `B = Sigma_u^{1/2} Q`, columns
    /// ordered and signed to match `impact` (`Q Q' = I`).
    pub rotation: Mat<f64>,
    /// Excess kurtosis of each recovered structural shock, in the identified
    /// column order. A value near zero flags a near-Gaussian, weakly
    /// identified shock.
    pub shock_kurtosis: Vec<f64>,
    /// Whether the FastICA fixed point met `tol` within `max_iter`.
    pub converged: bool,
    /// Number of FastICA iterations performed.
    pub n_iter: usize,
    /// The permutation applied to fix the column order: `order[j]` is the raw
    /// FastICA component index placed at identified position `j`.
    pub order: Vec<usize>,
}

/// Identify an SVAR by non-Gaussian independent components from the reduced
/// form.
///
/// `resid` is the `T x n` reduced-form residual matrix `U` (`u_t` in row `t`),
/// `sigma` its `n x n` innovation covariance `Sigma_u` (used for whitening, so
/// that `B B' = Sigma_u` holds exactly), and `b_coefs` the packed OLS
/// coefficient matrix (`(1 + n p) x n`, intercept row then lag blocks, the
/// `VarResults::params` layout) feeding the structural MA for the IRFs. The
/// maximal IRF horizon is `horizon`.
///
/// The FastICA fixed point uses the log-cosh contrast, starts from the
/// identity, and runs up to `max_iter` iterations to tolerance `tol`; the
/// columns are then ordered by `order_by` and signed max-abs-positive. The
/// whole routine is deterministic: identical inputs give bit-identical output.
///
/// # Errors
///
/// * [`IdentError::InvalidArgument`] if `n == 0`, `T < 2`, `max_iter == 0`,
///   `tol` is not a positive finite number, or `sigma` is not positive
///   definite;
/// * [`IdentError::Dimension`] if `sigma` is not `n x n` square, `resid` does
///   not have `n` columns, or `b_coefs` is not `(1 + n p) x n`;
/// * [`IdentError::NonFinite`] on any NaN/infinite entry in `resid`, `sigma`,
///   or `b_coefs`;
/// * [`IdentError::Linalg`] if the whitening or decorrelation eigensolve
///   fails.
#[allow(clippy::too_many_arguments)]
pub fn nongaussian_svar(
    resid: MatRef<'_, f64>,
    sigma: MatRef<'_, f64>,
    b_coefs: MatRef<'_, f64>,
    p: usize,
    horizon: usize,
    contrast: Contrast,
    max_iter: usize,
    tol: f64,
    order_by: OrderBy,
) -> Result<NonGaussianSvar, IdentError> {
    let Contrast::LogCosh = contrast; // only contrast supported today
    let n = sigma.nrows();
    if n == 0 {
        return Err(IdentError::InvalidArgument {
            what: "sigma must be at least 1 x 1",
        });
    }
    if sigma.ncols() != n {
        return Err(IdentError::Dimension {
            what: "sigma must be square",
            expected: n,
            got: sigma.ncols(),
        });
    }
    if resid.ncols() != n {
        return Err(IdentError::Dimension {
            what: "resid must have one column per variable (matching sigma)",
            expected: n,
            got: resid.ncols(),
        });
    }
    let t = resid.nrows();
    if t < 2 {
        return Err(IdentError::InvalidArgument {
            what: "resid must contain at least two observations",
        });
    }
    if max_iter == 0 {
        return Err(IdentError::InvalidArgument {
            what: "max_iter must be at least 1",
        });
    }
    if !(tol.is_finite() && tol > 0.0) {
        return Err(IdentError::InvalidArgument {
            what: "tol must be a positive finite number",
        });
    }
    for j in 0..n {
        for i in 0..n {
            if !sigma[(i, j)].is_finite() {
                return Err(IdentError::NonFinite { what: "sigma" });
            }
        }
    }
    for j in 0..n {
        for i in 0..t {
            if !resid[(i, j)].is_finite() {
                return Err(IdentError::NonFinite { what: "resid" });
            }
        }
    }

    // (1) Whiten. W = Sigma_u^{-1/2}, and keep Sigma_u^{1/2} to unwhiten.
    let w_half = sym_matrix_power(sigma, 0.5)?;
    let w_half_inv = sym_matrix_power(sigma, -0.5)?;

    // z_t = W u_t; in row-observation layout, Z = U W' = U W (W symmetric).
    let mut z = resid * w_half_inv.as_ref();
    // Center the whitened columns (standard FastICA preconditioning; a no-op
    // to rounding when the residuals carry a fitted intercept).
    for j in 0..n {
        let mut mean = 0.0;
        for i in 0..t {
            mean += z[(i, j)];
        }
        mean /= t as f64;
        for i in 0..t {
            z[(i, j)] -= mean;
        }
    }

    // (2) Symmetric FastICA fixed point -> orthogonal unmixing W_ica.
    let (w_ica, converged, n_iter) = fastica_symmetric(z.as_ref(), max_iter, tol)?;

    // (3) Unwhiten. Q = W_ica'; B = Sigma_u^{1/2} Q.
    let q_raw = w_ica.transpose().to_owned();
    let b_raw = w_half.as_ref() * q_raw.as_ref();

    // Recovered sources S = z W_ica' (T x n); per-column excess kurtosis.
    let sources = z.as_ref() * w_ica.transpose();
    let kurt_raw: Vec<f64> = (0..n)
        .map(|j| excess_kurtosis(sources.as_ref(), j, t))
        .collect();

    // (4a) Order columns by the chosen stable rule.
    let key: Vec<f64> = match order_by {
        OrderBy::Kurtosis => kurt_raw.iter().map(|k| k.abs()).collect(),
        OrderBy::ColumnNorm => (0..n)
            .map(|j| {
                (0..n)
                    .map(|i| b_raw[(i, j)] * b_raw[(i, j)])
                    .sum::<f64>()
                    .sqrt()
            })
            .collect(),
    };
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        key[b]
            .partial_cmp(&key[a])
            .unwrap_or(Ordering::Equal)
            .then(a.cmp(&b))
    });

    // Reorder B, Q and kurtosis; (4b) sign each column max-abs-positive.
    let mut impact = Mat::<f64>::zeros(n, n);
    let mut rotation = Mat::<f64>::zeros(n, n);
    let mut shock_kurtosis = vec![0.0f64; n];
    for (pos, &src) in order.iter().enumerate() {
        // largest-|entry| row of this B column (ties -> smallest row index).
        let mut istar = 0usize;
        let mut best = -1.0f64;
        for i in 0..n {
            let a = b_raw[(i, src)].abs();
            if a > best {
                best = a;
                istar = i;
            }
        }
        let flip = if b_raw[(istar, src)] < 0.0 { -1.0 } else { 1.0 };
        for i in 0..n {
            impact[(i, pos)] = flip * b_raw[(i, src)];
            rotation[(i, pos)] = flip * q_raw[(i, src)];
        }
        shock_kurtosis[pos] = kurt_raw[src];
    }

    // (5) Structural IRFs Theta_h = Psi_h B via the shared general-impact MA.
    let irf = structural_ma(b_coefs, impact.as_ref(), p, horizon)?;

    Ok(NonGaussianSvar {
        impact,
        irf,
        rotation,
        shock_kurtosis,
        converged,
        n_iter,
        order,
    })
}

/// Symmetric parallel FastICA with the log-cosh (tanh) contrast on whitened
/// data `z` (`T x n`), starting from the identity unmixing. Returns the
/// orthogonal unmixing matrix `W_ica` (rows are components), whether it
/// converged, and the iteration count.
///
/// The update mirrors Hyvarinen's parallel `_ica_par` exactly:
/// `W1 = (g(W z') z) / T - diag(mean g'(W z')) W`, followed by symmetric
/// decorrelation `W1 <- (W1 W1')^{-1/2} W1`, with the convergence measure
/// `max_j | |w1_j . w_j| - 1 |`.
fn fastica_symmetric(
    z: MatRef<'_, f64>,
    max_iter: usize,
    tol: f64,
) -> Result<(Mat<f64>, bool, usize), IdentError> {
    let t = z.nrows();
    let n = z.ncols();
    let tf = t as f64;

    let mut w = Mat::<f64>::identity(n, n);
    let mut converged = false;
    let mut n_iter = 0usize;

    for it in 0..max_iter {
        n_iter = it + 1;

        // Sources S = z W' (T x n): S[t, j] = (W z_t)_j.
        let s = z * w.transpose();

        // g = tanh(S); column means of g'(S) = 1 - tanh^2(S).
        let g = Mat::from_fn(t, n, |i, j| s[(i, j)].tanh());
        let mut beta = vec![0.0f64; n];
        for j in 0..n {
            let mut acc = 0.0;
            for i in 0..t {
                let th = g[(i, j)];
                acc += 1.0 - th * th;
            }
            beta[j] = acc / tf;
        }

        // W1 = (g' z) / T - diag(beta) W, with (g' z)[j, k] = sum_t g[t,j] z[t,k].
        let gz = g.transpose() * z;
        let w1_pre = Mat::from_fn(n, n, |j, k| gz[(j, k)] / tf - beta[j] * w[(j, k)]);

        // Symmetric decorrelation: W1 = (W1 W1')^{-1/2} W1.
        let wwt = w1_pre.as_ref() * w1_pre.transpose();
        let inv_sqrt = sym_matrix_power(wwt.as_ref(), -0.5)?;
        let w1 = inv_sqrt.as_ref() * w1_pre.as_ref();

        // Convergence: rows are near-parallel to the previous iterate.
        let mut lim = 0.0f64;
        for j in 0..n {
            let mut dot = 0.0;
            for k in 0..n {
                dot += w1[(j, k)] * w[(j, k)];
            }
            let d = (dot.abs() - 1.0).abs();
            if d > lim {
                lim = d;
            }
        }

        w = w1;
        if lim < tol {
            converged = true;
            break;
        }
    }

    Ok((w, converged, n_iter))
}

/// Symmetric matrix raised to a real power via its eigendecomposition:
/// `M = V diag(lambda) V'`, `M^power = V diag(lambda^power) V'`. Requires `M`
/// symmetric positive definite for negative powers (every eigenvalue > 0).
fn sym_matrix_power(m: MatRef<'_, f64>, power: f64) -> Result<Mat<f64>, IdentError> {
    let n = m.nrows();
    let eig = m.self_adjoint_eigen(Side::Lower).map_err(|_| {
        IdentError::Linalg(LinalgError::EigenFailed {
            what: "nongaussian whitening / decorrelation eigenproblem",
        })
    })?;
    let lambda: Vec<f64> = eig.S().column_vector().iter().copied().collect();
    let v = eig.U();

    let mut scaled = vec![0.0f64; n];
    for (i, &l) in lambda.iter().enumerate() {
        if l <= 0.0 || l.is_nan() {
            return Err(IdentError::InvalidArgument {
                what: "matrix is not positive definite (non-positive eigenvalue)",
            });
        }
        scaled[i] = l.powf(power);
    }

    // V diag(scaled) V'.
    Ok(Mat::from_fn(n, n, |i, j| {
        (0..n).map(|k| v[(i, k)] * scaled[k] * v[(j, k)]).sum()
    }))
}

/// Excess kurtosis `E[(x - mu)^4] / (E[(x - mu)^2])^2 - 3` of column `j` of the
/// `T x n` source matrix (population moments, divisor `T`).
fn excess_kurtosis(sources: MatRef<'_, f64>, j: usize, t: usize) -> f64 {
    let tf = t as f64;
    let mut mean = 0.0;
    for i in 0..t {
        mean += sources[(i, j)];
    }
    mean /= tf;
    let mut m2 = 0.0;
    let mut m4 = 0.0;
    for i in 0..t {
        let d = sources[(i, j)] - mean;
        let d2 = d * d;
        m2 += d2;
        m4 += d2 * d2;
    }
    m2 /= tf;
    m4 /= tf;
    if m2 <= 0.0 {
        return f64::NAN;
    }
    m4 / (m2 * m2) - 3.0
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 3x3 symmetric positive-definite covariance from `A A'`.
    fn toy_sigma() -> Mat<f64> {
        let a = [[1.0, 0.0, 0.0], [0.4, 0.9, 0.0], [0.2, 0.3, 0.7]];
        Mat::from_fn(3, 3, |i, j| {
            a[i].iter().zip(a[j].iter()).map(|(x, y)| x * y).sum()
        })
    }

    #[test]
    fn sym_matrix_power_roundtrips() -> Result<(), IdentError> {
        let sigma = toy_sigma();
        let half = sym_matrix_power(sigma.as_ref(), 0.5)?;
        let inv_half = sym_matrix_power(sigma.as_ref(), -0.5)?;
        // half * half == sigma.
        let prod = half.as_ref() * half.as_ref();
        for i in 0..3 {
            for j in 0..3 {
                assert!((prod[(i, j)] - sigma[(i, j)]).abs() < 1e-12);
            }
        }
        // half * inv_half == I.
        let id = half.as_ref() * inv_half.as_ref();
        for i in 0..3 {
            for j in 0..3 {
                let e = if i == j { 1.0 } else { 0.0 };
                assert!((id[(i, j)] - e).abs() < 1e-12);
            }
        }
        Ok(())
    }

    #[test]
    fn sym_matrix_power_rejects_non_pd() {
        // A matrix with a negative eigenvalue: inverse sqrt must fail.
        let m = Mat::from_fn(2, 2, |i, j| if i == j { -1.0 } else { 0.0 });
        assert!(matches!(
            sym_matrix_power(m.as_ref(), -0.5),
            Err(IdentError::InvalidArgument { .. })
        ));
    }

    #[test]
    fn rejects_bad_shapes() {
        let sigma = toy_sigma();
        let resid = Mat::<f64>::zeros(50, 3);
        let b = Mat::<f64>::zeros(1 + 3, 3); // p = 1
                                             // resid with wrong column count.
        let bad = Mat::<f64>::zeros(50, 2);
        assert!(matches!(
            nongaussian_svar(
                bad.as_ref(),
                sigma.as_ref(),
                b.as_ref(),
                1,
                4,
                Contrast::LogCosh,
                100,
                1e-8,
                OrderBy::Kurtosis
            ),
            Err(IdentError::Dimension { .. })
        ));
        // tol <= 0.
        assert!(matches!(
            nongaussian_svar(
                resid.as_ref(),
                sigma.as_ref(),
                b.as_ref(),
                1,
                4,
                Contrast::LogCosh,
                100,
                0.0,
                OrderBy::Kurtosis
            ),
            Err(IdentError::InvalidArgument { .. })
        ));
        // max_iter == 0.
        assert!(matches!(
            nongaussian_svar(
                resid.as_ref(),
                sigma.as_ref(),
                b.as_ref(),
                1,
                4,
                Contrast::LogCosh,
                0,
                1e-8,
                OrderBy::Kurtosis
            ),
            Err(IdentError::InvalidArgument { .. })
        ));
    }

    #[test]
    fn excess_kurtosis_of_gaussian_is_near_zero() {
        // Deterministic pseudo-Gaussian via Box-Muller on a fixed LCG (test
        // only; the library never uses a system RNG). Excess kurtosis ~ 0.
        let t = 20000usize;
        let mut state: u64 = 0x2026_0722_dead_beef;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 11) as f64) / ((1u64 << 53) as f64)
        };
        let mut col = Mat::<f64>::zeros(t, 1);
        let mut i = 0;
        while i < t {
            let u1 = next().max(1e-15);
            let u2 = next();
            let r = (-2.0 * u1.ln()).sqrt();
            col[(i, 0)] = r * (2.0 * std::f64::consts::PI * u2).cos();
            if i + 1 < t {
                col[(i + 1, 0)] = r * (2.0 * std::f64::consts::PI * u2).sin();
            }
            i += 2;
        }
        let k = excess_kurtosis(col.as_ref(), 0, t);
        assert!(
            k.abs() < 0.15,
            "gaussian excess kurtosis should be ~0, got {k}"
        );
    }
}
