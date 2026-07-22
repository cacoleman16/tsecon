//! Identification through heteroskedasticity, known-regime case
//! (Rigobon 2003; Lanne & Lütkepohl 2008).
//!
//! Statistical point-identification of the SVAR impact matrix `B` from the
//! fact that the *structural-shock variances* change across two exogenously
//! known variance regimes while the propagation matrix `B` stays constant.
//! No exclusion, sign, or narrative restriction is needed — the second
//! moments of the reduced-form residuals in the two regimes carry enough
//! information to pin down `B`, up to column sign and column order, whenever
//! the shocks' between-regime variance ratios are pairwise distinct.
//!
//! # Model
//!
//! Write the reduced-form residuals as `u_t = B e_t` with `B` an `n x n`
//! invertible matrix that is **constant** across regimes and `e_t` a vector
//! of mutually uncorrelated structural shocks whose diagonal covariance
//! *changes* across two exogenously known regimes `s in {1, 2}`. Normalizing
//! regime 1 to `Cov(e_t | s = 1) = I` and writing
//! `Cov(e_t | s = 2) = Lambda = diag(lambda_1, ..., lambda_n)` (`lambda_j > 0`
//! the relative shock variances) gives the two reduced-form residual
//! covariances
//!
//! ```text
//!   Sigma_1 = B B'            (regime 1)
//!   Sigma_2 = B Lambda B'     (regime 2)
//! ```
//!
//! # Closed form
//!
//! `Sigma_1` is SPD, so factor `Sigma_1 = P P'` (lower Cholesky `P`). Form the
//! symmetric matrix `M = P^{-1} Sigma_2 P^{-T}` and take its symmetric
//! eigendecomposition `M = W diag(lambda_1, ..., lambda_n) W'`
//! (`W` orthogonal). Then
//!
//! ```text
//!   B = P W,     Lambda = diag(eigenvalues of M).
//! ```
//!
//! This reconstructs the model exactly: `B B' = P W W' P' = P P' = Sigma_1`
//! and `B Lambda B' = P W Lambda W' P' = P M P' = Sigma_2`. The
//! Cholesky-whitening route is the same algorithm LAPACK `sygv` /
//! `scipy.linalg.eigh(a, b)` use for the generalized symmetric-definite
//! pencil `(Sigma_2, Sigma_1)`, so the eigenvalues `lambda_j` are exactly the
//! generalized eigenvalues — the regime-2 variances of the structural shocks
//! (regime 1 = 1), i.e. the **variance ratios** — and column `j` of `B` is the
//! impact vector of structural shock `j`.
//!
//! # Identification
//!
//! `(B, Lambda)` is unique up to (a) a joint permutation of the shocks
//! (columns of `B` and diagonal of `Lambda`) and (b) a sign flip of each
//! column of `B`. It is otherwise unique — the SVAR is point-identified — **if
//! and only if the eigenvalues are pairwise distinct**: `lambda_i != lambda_j`
//! for all `i != j`. If two ratios coincide, `W` is only determined up to an
//! orthogonal rotation within that eigenspace and the two shocks are not
//! separately identified. A necessary condition for *any* identification is
//! `Lambda != I` (the variances truly change, `Sigma_1 != Sigma_2`). The
//! shocks come out ordered by variance ratio; attaching economic meaning to
//! them is the caller's job — the statistics do not label shocks.
//!
//! # Scope and deferred variants
//!
//! This module builds the **exactly-two-known-regimes** case only. The
//! `> 2`-regime generalization and the Markov-switching / GARCH
//! variance-process variants (Lanne, Lütkepohl & Maciejowska 2010) have no
//! closed form — they need an ML/scoring or Markov-switching-filter engine
//! that does not exist in this workspace — and are deferred as future work,
//! not implemented here.

use tsecon_linalg::faer::{Mat, MatRef, Side};
use tsecon_linalg::{jittered_cholesky, symmetrize, LinalgError};
use tsecon_stats::chi2_sf;

use crate::error::IdentError;

/// Column sign convention applied to the recovered impact matrix `B`.
///
/// A column of `B` and its negation are observationally identical (both
/// reconstruct `Sigma_1`/`Sigma_2`), so each column's sign is a free
/// convention that must be fixed for the output to be deterministic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignConvention {
    /// For each column, make the largest-magnitude entry positive (ties
    /// broken toward the smallest row index). This is robust even when the
    /// diagonal entry is near zero, and is the default.
    MaxAbs,
    /// For each column `j`, make the diagonal entry `B[j, j]` non-negative.
    /// Natural when `B` is nearly triangular, but fragile when `B[j, j]` is
    /// close to zero.
    Diagonal,
}

/// The heteroskedasticity decomposition of two within-regime residual
/// covariances into a constant impact matrix and the regime-2 variance
/// ratios.
#[derive(Debug, Clone)]
pub struct HeteroDecomp {
    /// Structural impact matrix `B` (`= Theta_0`), `n x n`. Row `i` is a
    /// variable, column `j` a shock; columns are ordered by ascending
    /// variance ratio and sign-canonicalized.
    pub b: Mat<f64>,
    /// The `n` variance ratios `lambda_j` (regime-2 variance of shock `j`,
    /// regime 1 = 1), ascending. These are the generalized eigenvalues of the
    /// pencil `(Sigma_2, Sigma_1)`.
    pub lambda: Vec<f64>,
    /// `min_{i < j} |lambda_i - lambda_j|` — the point-identification margin.
    /// A value near zero flags a near-unidentified pair of shocks. Defined as
    /// `+inf` for `n < 2` (a single shock is trivially identified).
    pub min_ratio_gap: f64,
    /// `|lambda_j - 1|` per shock, ascending with `lambda`. A value near zero
    /// means shock `j`'s variance barely changes between regimes, so that
    /// shock is only weakly identified by the heteroskedasticity.
    pub ratio_dist_from_unity: Vec<f64>,
}

/// Result of the Bartlett-corrected Box's M test of covariance-matrix
/// equality across regimes.
#[derive(Debug, Clone)]
pub struct BoxMResult {
    /// The Bartlett-corrected statistic `(1 - c1) M`, asymptotically
    /// chi-square under equal covariances.
    pub statistic: f64,
    /// Degrees of freedom `(g - 1) d (d + 1) / 2`.
    pub dof: usize,
    /// Upper-tail p-value `P(chi^2_dof > statistic)`. A small p-value means
    /// the regimes' covariances differ, so the heteroskedasticity
    /// identification has content.
    pub pvalue: f64,
}

/// Solves the lower-triangular system `L X = B` for `X` by forward
/// substitution, column by column. `L` must be lower triangular with nonzero
/// diagonal (a Cholesky factor); `B` is `n x m`.
fn solve_lower(l: MatRef<'_, f64>, b: MatRef<'_, f64>) -> Mat<f64> {
    let n = l.nrows();
    let m = b.ncols();
    let mut x = Mat::<f64>::zeros(n, m);
    for col in 0..m {
        for i in 0..n {
            let mut acc = b[(i, col)];
            for k in 0..i {
                acc -= l[(i, k)] * x[(k, col)];
            }
            x[(i, col)] = acc / l[(i, i)];
        }
    }
    x
}

/// Validates that a matrix is square `n x n` and finite.
fn check_square_finite(m: MatRef<'_, f64>, n: usize, what: &'static str) -> Result<(), IdentError> {
    if m.nrows() != n || m.ncols() != n {
        return Err(IdentError::Dimension {
            what,
            expected: n,
            got: if m.nrows() != n { m.nrows() } else { m.ncols() },
        });
    }
    for j in 0..n {
        for i in 0..n {
            if !m[(i, j)].is_finite() {
                return Err(IdentError::NonFinite { what });
            }
        }
    }
    Ok(())
}

/// Decomposes two within-regime residual covariances `Sigma_1`, `Sigma_2`
/// into the constant impact matrix `B` and the regime-2 variance ratios.
///
/// Both inputs must be square, of the same size `n >= 1`, finite, and
/// `Sigma_1` positive definite (it is Cholesky-factorized). The returned `B`
/// satisfies `B B' = Sigma_1` and `B Lambda B' = Sigma_2` up to the
/// canonicalized column sign/order; see the [module docs](self) for the
/// identification condition.
///
/// # Errors
///
/// * [`IdentError::InvalidArgument`] if `n == 0`;
/// * [`IdentError::Dimension`] if the inputs are not square of one size;
/// * [`IdentError::NonFinite`] if either input has a NaN/infinite entry;
/// * [`IdentError::Linalg`] if `Sigma_1` is not positive definite (Cholesky
///   fails after the jitter ladder) or the symmetric eigensolver fails.
pub fn hetero_decompose(
    sigma1: MatRef<'_, f64>,
    sigma2: MatRef<'_, f64>,
    sign: SignConvention,
) -> Result<HeteroDecomp, IdentError> {
    let n = sigma1.nrows();
    if n == 0 {
        return Err(IdentError::InvalidArgument {
            what: "sigma1 must be at least 1 x 1",
        });
    }
    check_square_finite(sigma1, n, "sigma1")?;
    check_square_finite(sigma2, n, "sigma2")?;

    // P = lower Cholesky factor of Sigma_1 (SPD).
    let p = jittered_cholesky(sigma1)?.factor;

    // M = P^{-1} Sigma_2 P^{-T}, formed with two triangular solves:
    //   Z = P^{-1} Sigma_2                (solve P Z = Sigma_2)
    //   M = Z P^{-T} = (P^{-1} Z')'       (solve P Y = Z', then M = Y').
    let z = solve_lower(p.as_ref(), sigma2);
    let zt = z.transpose().to_owned();
    let y = solve_lower(p.as_ref(), zt.as_ref());
    let m = symmetrize(y.transpose())?;

    // Symmetric EVD; faer returns eigenvalues in nondecreasing order, so the
    // shocks come out ordered by ascending variance ratio (no permutation is
    // needed to match scipy.linalg.eigh, which sorts the same way).
    let eig = m.self_adjoint_eigen(Side::Lower).map_err(|_| {
        IdentError::Linalg(LinalgError::EigenFailed {
            what: "heteroskedasticity symmetric eigenproblem",
        })
    })?;
    let lambda: Vec<f64> = eig.S().column_vector().iter().copied().collect();
    let w = eig.U();

    // B = P W, then canonicalize each column's sign.
    let mut b = &p * w;
    for j in 0..n {
        let flip = match sign {
            SignConvention::MaxAbs => {
                let mut best = 0usize;
                let mut best_abs = -1.0_f64;
                for i in 0..n {
                    let a = b[(i, j)].abs();
                    if a > best_abs {
                        best_abs = a;
                        best = i;
                    }
                }
                b[(best, j)] < 0.0
            }
            SignConvention::Diagonal => b[(j, j)] < 0.0,
        };
        if flip {
            for i in 0..n {
                b[(i, j)] = -b[(i, j)];
            }
        }
    }

    // Since lambda is sorted ascending, the closest pair is adjacent.
    let min_ratio_gap = if n < 2 {
        f64::INFINITY
    } else {
        (1..n)
            .map(|k| (lambda[k] - lambda[k - 1]).abs())
            .fold(f64::INFINITY, f64::min)
    };
    let ratio_dist_from_unity: Vec<f64> = lambda.iter().map(|l| (l - 1.0).abs()).collect();

    Ok(HeteroDecomp {
        b,
        lambda,
        min_ratio_gap,
        ratio_dist_from_unity,
    })
}

/// Bartlett-corrected Box's M test of the null that all `g >= 2` group
/// covariance matrices are equal.
///
/// Each entry of `groups` is `(S_s, n_s)`: `S_s` is the group-`s` covariance
/// with the *unbiased* divisor `nu_s = n_s - 1` (mean subtracted), and `n_s`
/// is the group sample size. With pooled covariance
/// `S_p = (sum_s nu_s S_s) / (sum_s nu_s)` and `M = (sum_s nu_s) ln|S_p| -
/// sum_s nu_s ln|S_s|`, the Bartlett factor is
///
/// ```text
///   c1 = [ (2 d^2 + 3 d - 1) / (6 (d + 1)(g - 1)) ]
///        * [ (sum_s 1/nu_s) - 1/(sum_s nu_s) ]
/// ```
///
/// and the reported statistic is `(1 - c1) M`, asymptotically chi-square with
/// `(g - 1) d (d + 1) / 2` degrees of freedom (`d` the dimension).
///
/// # Errors
///
/// * [`IdentError::InvalidArgument`] if fewer than two groups, `d == 0`, or
///   any `n_s < 2`;
/// * [`IdentError::Dimension`] if the covariances are not square of one size;
/// * [`IdentError::Linalg`] if a covariance is not positive definite;
/// * [`IdentError::Stats`] if the chi-square survival function fails.
pub fn box_m_test(groups: &[(MatRef<'_, f64>, usize)]) -> Result<BoxMResult, IdentError> {
    let g = groups.len();
    if g < 2 {
        return Err(IdentError::InvalidArgument {
            what: "Box's M test needs at least two groups",
        });
    }
    let d = groups[0].0.nrows();
    if d == 0 {
        return Err(IdentError::InvalidArgument {
            what: "group covariances must be at least 1 x 1",
        });
    }

    let mut sum_nu = 0.0_f64;
    let mut inv_nu_sum = 0.0_f64;
    let mut pooled = Mat::<f64>::zeros(d, d);
    let mut nus = Vec::with_capacity(g);
    for (s, n_s) in groups {
        check_square_finite(*s, d, "group covariance")?;
        if *n_s < 2 {
            return Err(IdentError::InvalidArgument {
                what: "each group needs at least two observations for Box's M",
            });
        }
        let nu = (*n_s - 1) as f64;
        nus.push(nu);
        sum_nu += nu;
        inv_nu_sum += 1.0 / nu;
        for j in 0..d {
            for i in 0..d {
                pooled[(i, j)] += nu * s[(i, j)];
            }
        }
    }
    for j in 0..d {
        for i in 0..d {
            pooled[(i, j)] /= sum_nu;
        }
    }

    // ln|S| via the crate's hygiene Cholesky (ln|S| = 2 sum ln L_ii).
    let ln_det_pooled = jittered_cholesky(pooled.as_ref())?.log_det();
    let mut m_stat = sum_nu * ln_det_pooled;
    for ((s, _), nu) in groups.iter().zip(nus.iter()) {
        m_stat -= nu * jittered_cholesky(*s)?.log_det();
    }

    let dd = d as f64;
    let gg = g as f64;
    let c1 = ((2.0 * dd * dd + 3.0 * dd - 1.0) / (6.0 * (dd + 1.0) * (gg - 1.0)))
        * (inv_nu_sum - 1.0 / sum_nu);
    let statistic = (1.0 - c1) * m_stat;
    // d(d+1) is always even, so this integer division is exact.
    let dof = (g - 1) * d * (d + 1) / 2;
    let pvalue = chi2_sf(statistic, dof as f64)?;

    Ok(BoxMResult {
        statistic,
        dof,
        pvalue,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds an `n x n` faer matrix from a row-major nested slice.
    fn mat(rows: &[&[f64]]) -> Mat<f64> {
        let n = rows.len();
        Mat::from_fn(n, rows[0].len(), |i, j| rows[i][j])
    }

    /// `A B'` for square `A`, `B` (used to form covariances from factors).
    fn a_bt(a: MatRef<'_, f64>, b: MatRef<'_, f64>) -> Mat<f64> {
        let n = a.nrows();
        Mat::from_fn(n, n, |i, j| (0..n).map(|k| a[(i, k)] * b[(j, k)]).sum())
    }

    /// Applies the max-abs column-sign canonicalization to a copy of `b`.
    fn canon_maxabs(b: &Mat<f64>) -> Mat<f64> {
        let n = b.nrows();
        let mut out = b.clone();
        for j in 0..n {
            let mut best = 0usize;
            let mut best_abs = -1.0_f64;
            for i in 0..n {
                let a = out[(i, j)].abs();
                if a > best_abs {
                    best_abs = a;
                    best = i;
                }
            }
            if out[(best, j)] < 0.0 {
                for i in 0..n {
                    out[(i, j)] = -out[(i, j)];
                }
            }
        }
        out
    }

    #[test]
    fn reconstructs_the_two_covariances() -> Result<(), IdentError> {
        // A known non-triangular B and distinct Lambda, then the exact
        // covariances they induce; the decomposition must invert them.
        let b_true = mat(&[&[1.0, 0.5, -0.3], &[0.4, 1.0, 0.2], &[-0.2, 0.3, 1.0]]);
        let lambda_true = [0.5_f64, 2.0, 5.0];
        let n = 3;
        let sigma1 = a_bt(b_true.as_ref(), b_true.as_ref());
        // B Lambda B' = (B sqrt(Lambda)) (B sqrt(Lambda))'.
        let bl = Mat::from_fn(n, n, |i, j| b_true[(i, j)] * lambda_true[j].sqrt());
        let sigma2 = a_bt(bl.as_ref(), bl.as_ref());

        let d = hetero_decompose(sigma1.as_ref(), sigma2.as_ref(), SignConvention::MaxAbs)?;

        // Reconstruction identities (the oracle).
        let recon1 = a_bt(d.b.as_ref(), d.b.as_ref());
        let bl2 = Mat::from_fn(n, n, |i, j| d.b[(i, j)] * d.lambda[j].sqrt());
        let recon2 = a_bt(bl2.as_ref(), bl2.as_ref());
        for i in 0..n {
            for j in 0..n {
                assert!((recon1[(i, j)] - sigma1[(i, j)]).abs() < 1e-10);
                assert!((recon2[(i, j)] - sigma2[(i, j)]).abs() < 1e-10);
            }
        }

        // Recovered ratios equal the true ones (ascending, distinct).
        for (got, want) in d.lambda.iter().zip(lambda_true.iter()) {
            assert!((got - want).abs() < 1e-12, "lambda {got} vs {want}");
        }
        // Recovered B equals the canonicalized true B (columns already in
        // ascending-lambda order because lambda_true is ascending).
        let want_b = canon_maxabs(&b_true);
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (d.b[(i, j)] - want_b[(i, j)]).abs() < 1e-10,
                    "B[{i},{j}] {} vs {}",
                    d.b[(i, j)],
                    want_b[(i, j)]
                );
            }
        }

        // Point-identified: positive gap, ratios away from unity.
        assert!(d.min_ratio_gap > 0.5);
        assert_eq!(d.ratio_dist_from_unity.len(), n);
        Ok(())
    }

    #[test]
    fn equal_covariances_give_unit_ratios_and_zero_gap() -> Result<(), IdentError> {
        let sigma = mat(&[&[2.0, 0.3, 0.1], &[0.3, 1.5, 0.2], &[0.1, 0.2, 1.0]]);
        let d = hetero_decompose(sigma.as_ref(), sigma.as_ref(), SignConvention::MaxAbs)?;
        for l in &d.lambda {
            assert!((l - 1.0).abs() < 1e-10, "ratio {l} should be ~1");
        }
        assert!(
            d.min_ratio_gap < 1e-9,
            "gap {} should be ~0",
            d.min_ratio_gap
        );
        Ok(())
    }

    #[test]
    fn box_m_flags_equal_and_unequal_regimes() -> Result<(), IdentError> {
        // Equal covariances -> statistic ~ 0, p-value ~ 1.
        let s = mat(&[&[2.0, 0.3, 0.1], &[0.3, 1.5, 0.2], &[0.1, 0.2, 1.0]]);
        let eq = box_m_test(&[(s.as_ref(), 500), (s.as_ref(), 500)])?;
        assert_eq!(eq.dof, 3 * 4 / 2); // d(d+1)/2 = 6
        assert!(eq.statistic.abs() < 1e-8, "stat {}", eq.statistic);
        assert!((eq.pvalue - 1.0).abs() < 1e-6, "pvalue {}", eq.pvalue);

        // A clearly scaled second covariance -> large statistic, tiny p.
        let s2 = mat(&[&[6.0, 0.9, 0.3], &[0.9, 4.5, 0.6], &[0.3, 0.6, 3.0]]);
        let neq = box_m_test(&[(s.as_ref(), 500), (s2.as_ref(), 500)])?;
        assert!(neq.statistic > 100.0, "stat {}", neq.statistic);
        assert!(neq.pvalue < 1e-6, "pvalue {}", neq.pvalue);
        Ok(())
    }

    #[test]
    fn rejects_bad_shapes() {
        let sq = mat(&[&[1.0, 0.0], &[0.0, 1.0]]);
        let rect = Mat::<f64>::from_fn(2, 3, |_, _| 1.0);
        assert!(matches!(
            hetero_decompose(rect.as_ref(), sq.as_ref(), SignConvention::MaxAbs),
            Err(IdentError::Dimension { .. })
        ));
        // Mismatched sizes between the two covariances.
        let sq3 = Mat::<f64>::identity(3, 3);
        assert!(matches!(
            hetero_decompose(sq.as_ref(), sq3.as_ref(), SignConvention::MaxAbs),
            Err(IdentError::Dimension { .. })
        ));
        // Non-finite entry.
        let bad = mat(&[&[1.0, 0.0], &[0.0, f64::NAN]]);
        assert!(matches!(
            hetero_decompose(bad.as_ref(), sq.as_ref(), SignConvention::MaxAbs),
            Err(IdentError::NonFinite { .. })
        ));
        // Box's M with a single group / too-small group.
        assert!(matches!(
            box_m_test(&[(sq.as_ref(), 100)]),
            Err(IdentError::InvalidArgument { .. })
        ));
        assert!(matches!(
            box_m_test(&[(sq.as_ref(), 1), (sq.as_ref(), 100)]),
            Err(IdentError::InvalidArgument { .. })
        ));
    }
}
