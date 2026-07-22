//! Structural forecast-error variance decomposition (FEVD) for an *arbitrary*
//! structural impact matrix.
//!
//! [`tsecon_var::VarResults::fevd`](../../tsecon_var/index.html) is
//! recursive-Cholesky **only** — its impact matrix is hardcoded to
//! `P = chol_lower(Sigma)`, so it cannot decompose the forecast-error variance
//! attributable to shocks identified by any *non-recursive* scheme. This module
//! closes that gap: given the reduced-form MA dynamics and any admissible
//! structural impact matrix `A0` (columns = one-standard-deviation structural
//! impulse vectors, `A0 A0' = Sigma`; produced by sign / zero / proxy /
//! max-share / long-run identification, all of which yield a non-triangular
//! `A0 = P Q`), it returns the per-`(variable, shock, horizon)` FEV shares.
//!
//! # The decomposition
//!
//! With structural MA weights `Theta_s = Psi_s A0` (`Psi_s` the reduced-form MA
//! weights, `Psi_0 = I`, so `Theta_0 = A0`), the share of structural shock `j`
//! in the `(h + 1)`-step forecast-error variance of variable `i` is
//!
//! ```text
//! omega_{ij}(h) = sum_{s=0}^{h} Theta_s[i, j]^2
//!                 / sum_{m} sum_{s=0}^{h} Theta_s[i, m]^2
//! ```
//!
//! (Lütkepohl 2005, eq. 2.3.37, generalized from `A0 = P` to any `A0`). Three
//! facts make this well posed for *any* admissible `A0`:
//!
//! * **Rows sum to one.** `sum_j omega_{ij}(h) = 1` exactly, for every
//!   `(i, h)`.
//! * **The denominator is rotation-invariant.** It is the `i`-th diagonal entry
//!   of the `(h + 1)`-step forecast MSE matrix, `sum_{s<=h} Psi_s Sigma Psi_s'`,
//!   which depends on `A0` only through `A0 A0' = Sigma`. Writing `A0 = P Q`
//!   with `Q` orthogonal, `sum_m Theta_s[i, m]^2` is invariant to `Q`; the
//!   rotation only re-splits the fixed total across shocks `j`.
//! * **Column sign flips are invariant.** Negating any column of `A0` (a
//!   sign-normalization choice) leaves every `omega_{ij}` unchanged — the
//!   responses enter squared.
//!
//! When `A0 = P` (i.e. `Q = I`) this reduces **exactly** to
//! `tsecon_var::VarResults::fevd` and statsmodels `VARResults.fevd`.
//!
//! # Entry points
//!
//! * [`structural_fevd_from_theta`] — the core accumulation, taking the
//!   structural MA set `Theta_s` directly (so an identification scheme that has
//!   already built its structural IRFs — a sign-restriction draw, a max-share
//!   shock — can decompose them without rebuilding the dynamics).
//! * [`structural_fevd`] — the convenience path taking the packed reduced-form
//!   coefficients `b`, an impact matrix `A0`, the lag length `p` and a horizon;
//!   it builds `Theta_s = Psi_s A0` via the shared general-impact MA helper and
//!   then calls [`structural_fevd_from_theta`]. Passing `A0 = chol_lower(Sigma)`
//!   reproduces the recursive `tsecon_var` FEVD.

use tsecon_linalg::faer::{Mat, MatRef};

use crate::error::IdentError;
use crate::summary::structural_ma;

/// Structural FEVD from a supplied structural MA set `theta`.
///
/// `theta[s]` is the `n x n` structural moving-average matrix `Theta_s =
/// Psi_s A0` at horizon `s` (with `theta[0] = A0`); the slice has length
/// `horizon + 1` for horizons `0..=horizon`. Returns one `n x n` matrix per
/// horizon, `out[h][(i, j)]` being the share of structural shock `j` in the
/// `(h + 1)`-step forecast-error variance of variable `i`. Every row sums to
/// one: `sum_j out[h][(i, j)] = 1`.
///
/// # Errors
///
/// * [`IdentError::InvalidArgument`] if `theta` is empty, or if some
///   variable's `(h + 1)`-step forecast MSE diagonal is not strictly positive
///   (a degenerate impact matrix, so the share is undefined);
/// * [`IdentError::Dimension`] if any `theta[s]` is not square, or the matrices
///   differ in size across horizons;
/// * [`IdentError::NonFinite`] on any NaN/infinite entry in `theta`.
pub fn structural_fevd_from_theta(theta: &[Mat<f64>]) -> Result<Vec<Mat<f64>>, IdentError> {
    if theta.is_empty() {
        return Err(IdentError::InvalidArgument {
            what: "structural_fevd needs at least one horizon (theta is empty)",
        });
    }
    let n = theta[0].nrows();
    if theta[0].ncols() != n {
        return Err(IdentError::Dimension {
            what: "each structural MA matrix must be square",
            expected: n,
            got: theta[0].ncols(),
        });
    }
    for th in theta {
        if th.nrows() != n {
            return Err(IdentError::Dimension {
                what: "structural MA matrices must share their row dimension across horizons",
                expected: n,
                got: th.nrows(),
            });
        }
        if th.ncols() != n {
            return Err(IdentError::Dimension {
                what: "structural MA matrices must share their column dimension across horizons",
                expected: n,
                got: th.ncols(),
            });
        }
        for j in 0..n {
            for i in 0..n {
                if !th[(i, j)].is_finite() {
                    return Err(IdentError::NonFinite { what: "theta" });
                }
            }
        }
    }

    let mut out = Vec::with_capacity(theta.len());
    // Running sums of squared structural responses per (i, j) across horizons:
    // cum[(i, j)] = sum_{s<=h} Theta_s[i, j]^2 after processing horizon h.
    let mut cum = Mat::<f64>::zeros(n, n);
    for th in theta {
        for i in 0..n {
            for j in 0..n {
                cum[(i, j)] += th[(i, j)] * th[(i, j)];
            }
        }
        let mut share = Mat::<f64>::zeros(n, n);
        for i in 0..n {
            // The i-th (h+1)-step forecast MSE diagonal (the denominator). It is
            // a finite sum of squares (theta finiteness is validated above), so
            // `<= 0.0` is a complete degeneracy check (no NaN can arise).
            let mse_i: f64 = (0..n).map(|j| cum[(i, j)]).sum();
            if mse_i <= 0.0 {
                return Err(IdentError::InvalidArgument {
                    what: "forecast MSE diagonal in structural_fevd is not positive \
                           (degenerate impact matrix)",
                });
            }
            for j in 0..n {
                share[(i, j)] = cum[(i, j)] / mse_i;
            }
        }
        out.push(share);
    }
    Ok(out)
}

/// Structural FEVD from the reduced form and an arbitrary structural impact
/// matrix `A0`.
///
/// Builds the structural MA set `Theta_s = Psi_s A0` from the packed
/// reduced-form regressor coefficients `b` (shape `(1 + n p) x n`: intercept row
/// then the `p` lag blocks, `b[(1 + (l - 1) n + j, i)]` the coefficient of
/// `y_{t-l, j}` in equation `i`) and the impact matrix `impact = A0` (`n x n`,
/// columns = one-standard-deviation structural impulse vectors), then
/// decomposes it via [`structural_fevd_from_theta`]. Returns horizons
/// `0..=horizon`. Passing `impact = chol_lower(Sigma)` reproduces
/// `tsecon_var::VarResults::fevd` exactly.
///
/// # Errors
///
/// Propagates the shape / finiteness / `p >= 1` checks of the general-impact MA
/// construction (see [`crate::summary::structural_ma`]) as [`IdentError`], plus
/// the [`structural_fevd_from_theta`] error conditions.
pub fn structural_fevd(
    b: MatRef<'_, f64>,
    impact: MatRef<'_, f64>,
    p: usize,
    horizon: usize,
) -> Result<Vec<Mat<f64>>, IdentError> {
    let theta = structural_ma(b, impact, p, horizon)?;
    structural_fevd_from_theta(&theta)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tsecon_bayes::cholesky_irf;

    /// A small stable VAR(1) in the crate regressor layout plus a
    /// positive-definite `Sigma = A A'` (lower-triangular `A`).
    fn toy_var() -> (Mat<f64>, Mat<f64>) {
        let n = 3;
        let phi = [[0.5, 0.1, 0.0], [0.0, 0.4, 0.1], [0.1, 0.0, 0.3]];
        let mut b = Mat::<f64>::zeros(1 + n, n);
        for i in 0..n {
            for v in 0..n {
                b[(1 + v, i)] = phi[i][v];
            }
        }
        let a = [[1.0, 0.0, 0.0], [0.4, 0.9, 0.0], [0.2, 0.3, 0.7]];
        let sigma = Mat::from_fn(n, n, |i, j| {
            a[i].iter().zip(a[j].iter()).map(|(x, y)| x * y).sum()
        });
        (b, sigma)
    }

    fn toy_chol() -> Mat<f64> {
        let a = [[1.0, 0.0, 0.0], [0.4, 0.9, 0.0], [0.2, 0.3, 0.7]];
        Mat::from_fn(3, 3, |i, j| a[i][j])
    }

    #[test]
    fn rows_sum_to_one() -> Result<(), IdentError> {
        let (b, _) = toy_var();
        let a0 = toy_chol();
        let fevd = structural_fevd(b.as_ref(), a0.as_ref(), 1, 8)?;
        for (h, m) in fevd.iter().enumerate() {
            for i in 0..3 {
                let row: f64 = (0..3).map(|j| m[(i, j)]).sum();
                assert!(
                    (row - 1.0).abs() < 1e-12,
                    "fevd row ({i}) at horizon {h} sums to {row}, not 1"
                );
                for j in 0..3 {
                    assert!(
                        (-1e-12..=1.0 + 1e-12).contains(&m[(i, j)]),
                        "share ({i},{j}) at horizon {h} = {} out of [0,1]",
                        m[(i, j)]
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    fn from_theta_matches_convenience_path() -> Result<(), IdentError> {
        // structural_fevd(b, A0, ..) == structural_fevd_from_theta(theta) where
        // theta is the general-impact MA. Here A0 = chol reproduces cholesky_irf.
        let (b, sigma) = toy_var();
        let a0 = toy_chol();
        let theta = cholesky_irf(b.as_ref(), sigma.as_ref(), 1, 6)?;
        let a = structural_fevd_from_theta(&theta)?;
        let bfevd = structural_fevd(b.as_ref(), a0.as_ref(), 1, 6)?;
        for (x, y) in a.iter().zip(bfevd.iter()) {
            for i in 0..3 {
                for j in 0..3 {
                    assert!((x[(i, j)] - y[(i, j)]).abs() < 1e-12);
                }
            }
        }
        Ok(())
    }

    #[test]
    fn column_sign_flip_is_invariant() -> Result<(), IdentError> {
        // Negating columns of A0 leaves every share unchanged (squares).
        let (b, _) = toy_var();
        let a0 = toy_chol();
        let base = structural_fevd(b.as_ref(), a0.as_ref(), 1, 7)?;
        let flip = [-1.0, 1.0, -1.0];
        let a0f = Mat::from_fn(3, 3, |i, j| a0[(i, j)] * flip[j]);
        let flipped = structural_fevd(b.as_ref(), a0f.as_ref(), 1, 7)?;
        for (x, y) in base.iter().zip(flipped.iter()) {
            for i in 0..3 {
                for j in 0..3 {
                    assert!((x[(i, j)] - y[(i, j)]).abs() < 1e-13);
                }
            }
        }
        Ok(())
    }

    #[test]
    fn rejects_bad_input() {
        let empty: Vec<Mat<f64>> = Vec::new();
        assert!(matches!(
            structural_fevd_from_theta(&empty),
            Err(IdentError::InvalidArgument { .. })
        ));
        let nonsquare = vec![Mat::<f64>::zeros(3, 2)];
        assert!(matches!(
            structural_fevd_from_theta(&nonsquare),
            Err(IdentError::Dimension { .. })
        ));
        // A degenerate (zero) impact makes the MSE diagonal vanish.
        let zero = vec![Mat::<f64>::zeros(2, 2)];
        assert!(matches!(
            structural_fevd_from_theta(&zero),
            Err(IdentError::InvalidArgument { .. })
        ));
    }
}
