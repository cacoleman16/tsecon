//! Blanchard-Quah (1989) long-run structural identification.
//!
//! The recursive **long-run** (frequency-zero) scheme point-identifies the
//! structural shocks of a reduced-form VAR by a closed form — no rotation
//! sampling, no RNG, no Bayesian draws. It is the exact analog of R's
//! `vars::BQ`.
//!
//! # The reduced form
//!
//! For a stable VAR(p)
//!
//! ```text
//! y_t = c + A_1 y_{t-1} + ... + A_p y_{t-p} + u_t,   E[u_t u_t'] = Sigma_u,
//! ```
//!
//! the MA(inf) representation is `Psi_0 = I`,
//! `Psi_h = sum_{i=1..min(h,p)} Psi_{h-i} A_i` (the same recursion as
//! `tsecon_var::ma_rep`), and the total frequency-zero multiplier is
//!
//! ```text
//! C(1) = sum_{h>=0} Psi_h = (I - A_1 - ... - A_p)^{-1}.
//! ```
//!
//! [`long_run_multiplier`] exposes `C(1)` on its own: it is the shared
//! prerequisite that every Gali-style long-run identification reuses.
//!
//! # The structural mapping
//!
//! With structural shocks `eps_t` (`E[eps eps'] = I`) and `u_t = B eps_t`,
//! the impact matrix `B` satisfies `B B' = Sigma_u`, the structural IRF is
//! `Theta_h = Psi_h B`, and the cumulative long-run impact is
//! `LR = C(1) B`.
//!
//! The **identifying restriction** makes `LR` lower-triangular (positive
//! diagonal): shock `j` has zero permanent effect on every variable ordered
//! before it. Because `LR LR' = C(1) Sigma_u C(1)'` is known from the
//! reduced form, `LR` is exactly the lower Cholesky factor of that SPD
//! matrix, and
//!
//! ```text
//! LR = chol_lower( C(1) Sigma_u C(1)' ),   B = C(1)^{-1} LR = D LR,
//! ```
//!
//! where `D := I - sum A_i = C(1)^{-1}`. This is bit-for-bit `vars::BQ`
//! (using the degrees-of-freedom-adjusted `Sigma_u = U'U / (T - m)`).
//!
//! # Sign and normalization
//!
//! Each structural column is identified only up to sign. The default
//! (`normalize_impact = false`) pins the sign by the positive-diagonal
//! Cholesky on `LR`, matching `vars::BQ` with no column re-signing. Passing
//! `normalize_impact = true` instead forces a positive diagonal on the
//! impact matrix `B` (flip column `j` when `B[j,j] < 0`).
//!
//! # A caveat about the data (not the map)
//!
//! Long-run/BQ estimates are known to be sensitive in finite samples to the
//! imprecisely estimated long-run multiplier `C(1)` (Faust & Leeper 1997).
//! That is an inference caveat about the *data*; the reduced-form ->
//! structural map implemented here is an exact, deterministic closed form.

use tsecon_linalg::faer::linalg::solvers::DenseSolveCore;
use tsecon_linalg::faer::{Mat, MatRef, Side};
use tsecon_linalg::symmetrize;

use crate::error::IdentError;

/// Message emitted when the long-run matrix `I - sum A_i` is singular.
const SINGULAR_LR: &str =
    "long-run matrix (I - A_1 - ... - A_p) is singular; the VAR has a unit root at frequency zero";
/// Message emitted when `C(1) Sigma_u C(1)'` is not positive definite.
const NOT_PD: &str =
    "long-run covariance C(1) Sigma_u C(1)' is not positive definite (Sigma_u must be PD)";
/// Message emitted when the supplied long-run zeros are not a column
/// permutation of the recursive lower-triangular pattern.
const NOT_TRIANGULARIZABLE: &str = "long-run zero pattern is not a column permutation of the recursive (lower-triangular) scheme; a general zero restriction is not solvable by a single Cholesky — use the planned zero_svar (Rubio-Ramirez-Waggoner-Zha QR) method";

/// The point-identified Blanchard-Quah structural SVAR.
///
/// All matrices are `k x k` (with `k` the number of variables), indexed
/// `[variable][shock]`; `irf` has length `horizon + 1`.
#[derive(Debug, Clone)]
pub struct LongRunSvar {
    /// Contemporaneous structural impact matrix `B = Theta_0`;
    /// `impact[i][j]` is the on-impact response of variable `i` to a unit
    /// structural shock `j`.
    pub impact: Mat<f64>,
    /// Long-run cumulative structural impact `LR = C(1) B`,
    /// lower-triangular with positive diagonal under the default recursive
    /// scheme; `long_run[i][j]` is the permanent response of variable `i`
    /// to shock `j`, and entries with `j > i` are exactly zero.
    pub long_run: Mat<f64>,
    /// Reduced-form frequency-zero multiplier `C(1) = (I - sum A_i)^{-1}`,
    /// exposed for downstream long-run methods.
    pub long_run_multiplier: Mat<f64>,
    /// Structural impulse responses `Theta_h = Psi_h B` for
    /// `h = 0..=horizon`; `irf[0] == impact`.
    pub irf: Vec<Mat<f64>>,
}

/// Validates a slice of VAR lag-coefficient matrices and returns their
/// common dimension `k`.
fn validate_coefs(coefs: &[MatRef<'_, f64>]) -> Result<usize, IdentError> {
    if coefs.is_empty() {
        return Err(IdentError::InvalidArgument {
            what: "coefs must contain at least one VAR lag matrix",
        });
    }
    let k = coefs[0].nrows();
    if k == 0 {
        return Err(IdentError::InvalidArgument {
            what: "VAR coefficient matrices must be non-empty",
        });
    }
    for a in coefs {
        if a.nrows() != k || a.ncols() != k {
            return Err(IdentError::Dimension {
                what: "all VAR coefficient matrices must be square of one size",
                expected: k,
                got: if a.nrows() != k { a.nrows() } else { a.ncols() },
            });
        }
        for j in 0..k {
            for i in 0..k {
                if !a[(i, j)].is_finite() {
                    return Err(IdentError::NonFinite { what: "coefs" });
                }
            }
        }
    }
    Ok(k)
}

/// Computes `D = I - sum_i A_i` and its inverse `C(1) = D^{-1}`.
fn long_run_matrices(coefs: &[MatRef<'_, f64>]) -> Result<(Mat<f64>, Mat<f64>), IdentError> {
    let k = validate_coefs(coefs)?;
    let mut d = Mat::<f64>::identity(k, k);
    for a in coefs {
        for j in 0..k {
            for i in 0..k {
                d[(i, j)] -= a[(i, j)];
            }
        }
    }
    let c1 = d.partial_piv_lu().inverse();
    for j in 0..k {
        for i in 0..k {
            if !c1[(i, j)].is_finite() {
                return Err(IdentError::InvalidArgument { what: SINGULAR_LR });
            }
        }
    }
    Ok((d, c1))
}

/// Reduced-form frequency-zero multiplier `C(1) = (I - A_1 - ... - A_p)^{-1}`.
///
/// This is the total cumulative reduced-form impulse response (`sum_h Psi_h`)
/// and the shared prerequisite of every long-run identification scheme.
///
/// # Errors
///
/// * [`IdentError::InvalidArgument`] if `coefs` is empty, contains a `0 x 0`
///   matrix, or the matrix `I - sum A_i` is singular (a unit root at
///   frequency zero);
/// * [`IdentError::Dimension`] if the lag matrices are not square and of one
///   size;
/// * [`IdentError::NonFinite`] on NaN/infinite coefficients.
pub fn long_run_multiplier(coefs: &[MatRef<'_, f64>]) -> Result<Mat<f64>, IdentError> {
    Ok(long_run_matrices(coefs)?.1)
}

/// Resolves the shock ordering `col_of_pos`, where `col_of_pos[c]` is the
/// shock placed at triangular position `c` (the position with exactly `c`
/// long-run zeros). `None` yields the identity (classic BQ); `Some(pairs)`
/// must be a column permutation of the strict-upper-triangular pattern.
fn resolve_ordering(
    long_run_zeros: Option<&[(usize, usize)]>,
    k: usize,
) -> Result<Vec<usize>, IdentError> {
    let pairs = match long_run_zeros {
        None => return Ok((0..k).collect()),
        Some(p) => p,
    };
    // Per-shock set of long-run-zero variables.
    let mut zero_rows: Vec<Vec<usize>> = vec![Vec::new(); k];
    for &(var, shock) in pairs {
        if var >= k {
            return Err(IdentError::RestrictionOutOfRange {
                what: "long-run-zero variable index",
                index: var,
                bound: k,
            });
        }
        if shock >= k {
            return Err(IdentError::RestrictionOutOfRange {
                what: "long-run-zero shock index",
                index: shock,
                bound: k,
            });
        }
        zero_rows[shock].push(var);
    }
    // Each shock's zeros must be a prefix {0, ..., m-1}, and the counts
    // {m} must be a permutation of {0, ..., k-1}.
    let mut count_to_shock: Vec<Option<usize>> = vec![None; k];
    for (s, rows) in zero_rows.iter().enumerate() {
        let mut z = rows.clone();
        z.sort_unstable();
        z.dedup();
        for (idx, &v) in z.iter().enumerate() {
            if v != idx {
                return Err(IdentError::InvalidArgument {
                    what: NOT_TRIANGULARIZABLE,
                });
            }
        }
        let m = z.len();
        if m >= k || count_to_shock[m].is_some() {
            return Err(IdentError::InvalidArgument {
                what: NOT_TRIANGULARIZABLE,
            });
        }
        count_to_shock[m] = Some(s);
    }
    let mut col_of_pos = vec![0usize; k];
    for (c, slot) in count_to_shock.iter().enumerate() {
        match slot {
            Some(s) => col_of_pos[c] = *s,
            None => {
                return Err(IdentError::InvalidArgument {
                    what: NOT_TRIANGULARIZABLE,
                })
            }
        }
    }
    Ok(col_of_pos)
}

/// Blanchard-Quah long-run SVAR: the closed-form structural impact matrix,
/// long-run matrix, reduced-form long-run multiplier, and structural IRF.
///
/// `coefs` are the reduced-form lag matrices `[A_1, ..., A_p]` (each
/// `k x k`), `sigma_u` the (df-adjusted) reduced-form residual covariance.
/// `long_run_zeros` selects the identifying pattern: `None` is the classic
/// recursive lower-triangular scheme; `Some(pairs)` a `(variable, shock)`
/// zero pattern that must be a column permutation of it. Set
/// `normalize_impact` to force a positive diagonal on `B` instead of on the
/// long-run matrix.
///
/// # Errors
///
/// * [`IdentError::InvalidArgument`] for `k < 2`, a singular long-run matrix
///   (unit root), a non-positive-definite long-run covariance, or a
///   non-triangularizable zero pattern;
/// * [`IdentError::Dimension`] if `sigma_u` is not `k x k` for the `k` implied
///   by `coefs`;
/// * [`IdentError::RestrictionOutOfRange`] if a restriction indexes a
///   variable or shock outside `0..k`;
/// * [`IdentError::NonFinite`] on NaN/infinite inputs.
pub fn long_run_svar(
    coefs: &[MatRef<'_, f64>],
    sigma_u: MatRef<'_, f64>,
    horizon: usize,
    long_run_zeros: Option<&[(usize, usize)]>,
    normalize_impact: bool,
) -> Result<LongRunSvar, IdentError> {
    let (d, c1) = long_run_matrices(coefs)?;
    let k = d.nrows();
    if k < 2 {
        return Err(IdentError::InvalidArgument {
            what: "long-run identification needs at least 2 variables",
        });
    }
    if sigma_u.nrows() != k || sigma_u.ncols() != k {
        return Err(IdentError::Dimension {
            what: "sigma_u must be k x k for the k implied by coefs",
            expected: k,
            got: if sigma_u.nrows() != k {
                sigma_u.nrows()
            } else {
                sigma_u.ncols()
            },
        });
    }
    for j in 0..k {
        for i in 0..k {
            if !sigma_u[(i, j)].is_finite() {
                return Err(IdentError::NonFinite { what: "sigma_u" });
            }
        }
    }

    // Target M = C(1) Sigma_u C(1)', symmetrized to kill roundoff asymmetry
    // before the Cholesky.
    let m0 = c1.as_ref() * sigma_u;
    let m = m0.as_ref() * c1.transpose();
    let m_sym = symmetrize(m.as_ref())?;
    let lr_lower = m_sym
        .llt(Side::Lower)
        .map_err(|_| IdentError::InvalidArgument { what: NOT_PD })?
        .L()
        .to_owned();

    // Place the Cholesky columns at their identified shock positions.
    let col_of_pos = resolve_ordering(long_run_zeros, k)?;
    let mut lr = Mat::<f64>::zeros(k, k);
    for (c, &s) in col_of_pos.iter().enumerate() {
        for i in 0..k {
            lr[(i, s)] = lr_lower[(i, c)];
        }
    }

    // Structural impact matrix B = C(1)^{-1} LR = D LR.
    let mut b = d.as_ref() * lr.as_ref();

    if normalize_impact {
        for s in 0..k {
            if b[(s, s)] < 0.0 {
                for i in 0..k {
                    b[(i, s)] = -b[(i, s)];
                    lr[(i, s)] = -lr[(i, s)];
                }
            }
        }
    }

    // MA(inf) weights Psi_h and structural IRF Theta_h = Psi_h B.
    let p = coefs.len();
    let mut psi: Vec<Mat<f64>> = Vec::with_capacity(horizon + 1);
    psi.push(Mat::<f64>::identity(k, k));
    for h in 1..=horizon {
        let mut acc = Mat::<f64>::zeros(k, k);
        for i in 1..=h.min(p) {
            acc += psi[h - i].as_ref() * coefs[i - 1];
        }
        psi.push(acc);
    }
    let irf: Vec<Mat<f64>> = psi.iter().map(|ps| ps.as_ref() * b.as_ref()).collect();

    Ok(LongRunSvar {
        impact: b,
        long_run: lr,
        long_run_multiplier: c1,
        irf,
    })
}
