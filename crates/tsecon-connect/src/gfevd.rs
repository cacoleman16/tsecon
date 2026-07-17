//! Generalized forecast-error variance decomposition (Pesaran-Shin 1998),
//! row-normalized to the Diebold-Yilmaz (2012) connectedness convention.

use tsecon_linalg::faer::{Mat, MatRef};

use crate::error::ConnectError;

/// Generalized forecast-error variance decomposition of Pesaran and Shin
/// (1998, *Economics Letters* 58, eq. 8), row-normalized so each row
/// sums to one (the Diebold-Yilmaz 2012 connectedness convention).
///
/// Given the reduced-form MA(inf) coefficient matrices
/// `psi = [Psi_0, ..., Psi_H]` (`Psi_0 = I`, so a horizon-`H` table
/// requires `H + 1` matrices; see [`tsecon_var::ma_rep`]) and the
/// residual covariance `Sigma`, the *un-normalized* generalized share of
/// shock `j` in the `H`-step forecast-error variance of variable `i` is
///
/// ```text
///                 sigma_jj^{-1} sum_{h=0}^{H} (e_i' Psi_h Sigma e_j)^2
/// theta_ij(H) = -------------------------------------------------------
///                       sum_{h=0}^{H} e_i' Psi_h Sigma Psi_h' e_i
/// ```
///
/// where `e_i` is the `i`-th unit vector. Unlike the orthogonalized
/// (Cholesky) FEVD, the generalized shares do not sum to one across `j`
/// because the generalized shocks are correlated; Diebold and Yilmaz
/// (2012, *Int. J. Forecasting* 28, sec. 2.2) therefore row-normalize
///
/// ```text
/// theta~_ij(H) = theta_ij(H) / sum_{m=1}^{k} theta_im(H),
/// ```
/// which is what this function returns (`k x k`, every row summing to
/// one). Note the shares are invariant to a positive rescaling of
/// `Sigma`: the `sigma_jj^{-1}`, the squared numerator, and the
/// denominator together cancel any common factor.
///
/// # Errors
///
/// * [`ConnectError::InvalidArgument`] if `psi` is empty or its matrices
///   are `0 x 0`;
/// * [`ConnectError::Dimension`] if the `Psi_h` are not all square of
///   `Sigma`'s order;
/// * [`ConnectError::NonFinite`] on NaN/infinite entries in `psi` or
///   `sigma`;
/// * [`ConnectError::NotPositiveDefinite`] if a `Sigma` diagonal entry,
///   an own-variable forecast-error variance, or a normalization row sum
///   is non-positive.
pub fn generalized_fevd(
    psi: &[Mat<f64>],
    sigma: MatRef<'_, f64>,
) -> Result<Mat<f64>, ConnectError> {
    if psi.is_empty() {
        return Err(ConnectError::InvalidArgument {
            what: "psi must contain at least Psi_0",
        });
    }
    let k = sigma.nrows();
    if k == 0 {
        return Err(ConnectError::InvalidArgument {
            what: "sigma must be non-empty",
        });
    }
    if sigma.ncols() != k {
        return Err(ConnectError::Dimension {
            what: "sigma must be square",
            expected: k,
            got: sigma.ncols(),
        });
    }
    for a in psi {
        if a.nrows() != k || a.ncols() != k {
            return Err(ConnectError::Dimension {
                what: "every Psi_h must be square of sigma's order",
                expected: k,
                got: if a.nrows() != k { a.nrows() } else { a.ncols() },
            });
        }
    }
    for i in 0..k {
        for j in 0..k {
            if !sigma[(i, j)].is_finite() {
                return Err(ConnectError::NonFinite { what: "sigma" });
            }
        }
    }
    for a in psi {
        for j in 0..k {
            for i in 0..k {
                if !a[(i, j)].is_finite() {
                    return Err(ConnectError::NonFinite { what: "psi" });
                }
            }
        }
    }
    for j in 0..k {
        if sigma[(j, j)] <= 0.0 {
            return Err(ConnectError::NotPositiveDefinite {
                what: "sigma diagonal entry",
            });
        }
    }

    // Accumulate the numerator (per i, j) and the denominator (per i)
    // across horizons. With m_h = Psi_h Sigma:
    //   numerator_ij  += m_h[i, j]^2
    //   denominator_i += (Psi_h Sigma Psi_h')[i, i]
    //                  = sum_l m_h[i, l] * Psi_h[i, l].
    let mut numer = Mat::<f64>::zeros(k, k);
    let mut denom = vec![0.0_f64; k];
    for a in psi {
        let m = a * sigma;
        for i in 0..k {
            let mut d = 0.0;
            for l in 0..k {
                d += m[(i, l)] * a[(i, l)];
            }
            denom[i] += d;
            for j in 0..k {
                numer[(i, j)] += m[(i, j)] * m[(i, j)];
            }
        }
    }

    let mut theta = Mat::<f64>::zeros(k, k);
    for i in 0..k {
        if denom[i] <= 0.0 {
            return Err(ConnectError::NotPositiveDefinite {
                what: "own-variable forecast-error variance",
            });
        }
        // Un-normalized generalized shares, then the row sum.
        let mut row_sum = 0.0;
        for j in 0..k {
            let t = numer[(i, j)] / (sigma[(j, j)] * denom[i]);
            theta[(i, j)] = t;
            row_sum += t;
        }
        if row_sum <= 0.0 {
            return Err(ConnectError::NotPositiveDefinite {
                what: "generalized-FEVD row sum",
            });
        }
        for j in 0..k {
            theta[(i, j)] /= row_sum;
        }
    }
    Ok(theta)
}
