//! Max-share / maximum-FEV structural identification.
//!
//! Identify the single structural shock whose contribution to the
//! forecast-error variance (FEV) of a *target* variable, accumulated over a
//! horizon window `[h0, h1]`, is maximal (Uhlig 2004 penalty-free eigenvalue
//! variant; Francis, Owyang, Roush & DiCecio 2014 finite-horizon
//! main-business-cycle shock; Barsky & Sims 2011 news shock).
//!
//! # The closed form
//!
//! Given the orthogonalized moving-average coefficients `Theta_s = Psi_s P`
//! (where `Psi_s` are the reduced-form MA weights and `P` is the lower
//! Cholesky factor of the innovation covariance; `Theta_0 = P`), write the
//! target's row as the `k`-vector `r_s = Theta_s[target, :]'`. A structural
//! impact vector is `b = P q` for a unit vector `q` (`q'q = 1`); the target's
//! response to that single unit-variance shock at horizon `s` is `r_s' q`, and
//! its contribution to the target's FEV accumulated over `[h0, h1]` is
//!
//! ```text
//! N(q) = sum_{s=h0..h1} (r_s' q)^2 = q' A q,   A = sum_{s=h0..h1} r_s r_s'.
//! ```
//!
//! Because `P P' = Sigma`, `tr(A)` equals the target's *total* accumulated FEV
//! over the window, so `max_{q'q=1} q'Aq` is solved by the leading eigenvector
//! of the symmetric PSD matrix `A`, and `lambda_max / tr(A)` is the identified
//! shock's FEV share (in `[0, 1]`). No RNG, no rejection sampling, no
//! iteration — a deterministic point estimate given the reduced form.
//!
//! # Variants
//!
//! * **`weighting`** — [`MaxShareWeighting::Window`] (Uhlig / Francis, the
//!   incremental window sum above, whose share is an exact accumulated-FEV
//!   fraction) or [`MaxShareWeighting::Cumulative`] (Barsky-Sims: maximize the
//!   mean over target horizons of the *cumulative* FEV share, `A = sum_h C_h /
//!   tr(C_h)` with `C_h = sum_{s=0..h} r_s r_s'`; here `tr(A) = h1 - h0 + 1`,
//!   so the share is the window-mean cumulative share).
//! * **`exclude_impact`** — the Barsky-Sims news restriction: force the shock
//!   to have zero impact on the target, `r_0' q = 0`. Solved by projecting `q`
//!   onto `null(c')` with `c = r_0`: `q = N z` for an orthonormal basis `N`
//!   (`k x (k-1)`) of that null space, then `z*` is the leading eigenvector of
//!   `N' A N`. (Any orthonormal `N` spanning `null(c')` yields the same `q*`
//!   up to sign and the same reduced eigenvalues.)
//! * **`sign`** — the eigenvector is defined only up to sign, so it is pinned
//!   deterministically (default [`MaxShareSign::Cumsum`]: the target's
//!   cumulative windowed response is made non-negative). Both this crate and
//!   the golden reference apply the identical rule, which is what makes the
//!   identified shock reproducible across two independent eigensolvers.
//!
//! # Statistical, not economic, identification
//!
//! Max-share recovers "the linear combination of reduced-form innovations that
//! maximizes the target's windowed FEV share." That equals the economic shock
//! of interest *only* when a single shock dominates the band; when several
//! shocks contribute comparably the recovered shock is a mongrel (the
//! "max-share shock is not the news shock" critique, cf. Kurmann-Sims 2021).
//! The returned [`MaxShareResult::eigenvalues`] (ascending) are a degeneracy
//! diagnostic: a small gap between the last two means the maximizing direction
//! is ill-conditioned and neither implementation is trustworthy.

use tsecon_linalg::faer::{Mat, Side};

use crate::error::IdentError;

/// Windowing scheme for the accumulated forecast-error variance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaxShareWeighting {
    /// Uhlig (2004) / Francis et al. (2014): the incremental window sum
    /// `A = sum_{s=h0..h1} r_s r_s'`. Its share is an exact accumulated-FEV
    /// fraction.
    Window,
    /// Barsky-Sims (2011): the sum over target horizons of the cumulative FEV
    /// share, `A = sum_{h=h0..h1} C_h / tr(C_h)`.
    Cumulative,
}

/// Sign normalization for the identified rotation column `q*` (defined only up
/// to sign by the eigenproblem).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaxShareSign {
    /// Flip so the target's cumulative windowed response
    /// `sum_{s=h0..h1} (Theta_s q*)[target]` is non-negative.
    Cumsum,
    /// Flip so the target's impact response `(P q*)[target]` is non-negative.
    /// Invalid together with `exclude_impact` (that response is forced to
    /// zero).
    Impact,
    /// Leave the raw eigenvector sign untouched.
    None,
}

/// The identified max-share structural shock.
#[derive(Debug, Clone, PartialEq)]
pub struct MaxShareResult {
    /// Structural impulse responses `irf[h][i] = (Theta_h q*)[i]`, the response
    /// of variable `i` to the identified one-standard-deviation max-share shock
    /// at horizon `h`. Shape `[horizon + 1][k]`.
    pub irf: Vec<Vec<f64>>,
    /// Impact response `b* = Theta_0 q* = P q*` (equals `irf[0]`). Length `k`.
    pub impact: Vec<f64>,
    /// Identified rotation column `q*` (unit vector in the orthogonalized /
    /// whitened coordinate; the structural shock is `eps*_t = q*' P^{-1} u_t`).
    /// Length `k`.
    pub q: Vec<f64>,
    /// FEV share of the target explained by the identified shock over
    /// `[h0, h1]`: `lambda_max / tr(A)` (an exact accumulated-FEV fraction for
    /// `Window`; the window-mean cumulative share for `Cumulative`).
    pub share_window: f64,
    /// Cumulative FEV share of the identified shock for the target at each
    /// horizon `0..=horizon` (the Barsky-Sims-style profile; the `fevd`
    /// formula for a single shock). Length `horizon + 1`.
    pub fev_share: Vec<f64>,
    /// Eigenvalues of the solved eigenproblem, ascending: of `A` (length `k`),
    /// or of the projected `N' A N` when `exclude_impact` (length `k - 1`). A
    /// degeneracy diagnostic — compare the last two.
    pub eigenvalues: Vec<f64>,
}

/// Identify the max-share / maximum-FEV structural shock from the
/// orthogonalized MA coefficients `theta` (`theta[s] = Theta_s = Psi_s P`, one
/// `k x k` matrix per horizon `s = 0..=horizon`, exactly
/// `VarResults::orth_ma_rep`).
///
/// The maximal horizon is inferred as `horizon = theta.len() - 1`.
///
/// # Errors
///
/// * [`IdentError::InvalidArgument`] if `theta` is empty, `h0 > h1`, the
///   matrices are not square/consistent, `exclude_impact` is paired with
///   `MaxShareSign::Impact`, or `exclude_impact` is requested with `k < 2` or a
///   degenerate target impact row;
/// * [`IdentError::RestrictionOutOfRange`] if `target >= k` or `h1 > horizon`;
/// * [`IdentError::NonFinite`] if any `theta` entry is NaN or infinite;
/// * [`IdentError::Linalg`] if the symmetric eigensolver fails.
pub fn max_share_shock(
    theta: &[Mat<f64>],
    target: usize,
    h0: usize,
    h1: usize,
    exclude_impact: bool,
    weighting: MaxShareWeighting,
    sign: MaxShareSign,
) -> Result<MaxShareResult, IdentError> {
    if theta.is_empty() {
        return Err(IdentError::InvalidArgument {
            what: "theta must contain at least the impact matrix (horizon >= 0)",
        });
    }
    let horizon = theta.len() - 1;
    let k = theta[0].nrows();
    if k == 0 {
        return Err(IdentError::InvalidArgument {
            what: "theta matrices must be non-empty (k >= 1)",
        });
    }
    // Shape + finiteness of every horizon matrix.
    for m in theta {
        if m.nrows() != k || m.ncols() != k {
            return Err(IdentError::InvalidArgument {
                what: "every theta[s] must be the same square k x k matrix",
            });
        }
        for j in 0..k {
            for i in 0..k {
                if !m[(i, j)].is_finite() {
                    return Err(IdentError::NonFinite { what: "theta" });
                }
            }
        }
    }
    if target >= k {
        return Err(IdentError::RestrictionOutOfRange {
            what: "target",
            index: target,
            bound: k,
        });
    }
    if h0 > h1 {
        return Err(IdentError::InvalidArgument {
            what: "window bounds must satisfy h0 <= h1",
        });
    }
    if h1 > horizon {
        return Err(IdentError::RestrictionOutOfRange {
            what: "h1 (window end must not exceed the IRF horizon)",
            index: h1,
            bound: horizon + 1,
        });
    }
    if exclude_impact && sign == MaxShareSign::Impact {
        return Err(IdentError::InvalidArgument {
            what: "sign=\"impact\" is invalid with exclude_impact=true (impact response is zero)",
        });
    }
    if exclude_impact && k < 2 {
        return Err(IdentError::InvalidArgument {
            what: "exclude_impact requires k >= 2 (the null space of the impact row is empty)",
        });
    }

    // Target row r_s = Theta_s[target, :]' as a k-vector.
    let row = |s: usize| -> Vec<f64> { (0..k).map(|j| theta[s][(target, j)]).collect() };

    // Symmetric PSD objective matrix A (k x k).
    let a = objective_matrix(theta, target, h0, h1, weighting, k)?;
    let trace_a: f64 = (0..k).map(|i| a[(i, i)]).sum();

    // Solve the (possibly impact-constrained) eigenproblem.
    let (q, eigenvalues, lambda_max) = if exclude_impact {
        let c = row(0); // c = r_0 = target row of P
                        // theta is finite-checked above, so cc is finite and non-negative here.
        let cc: f64 = c.iter().map(|v| v * v).sum();
        if cc <= 0.0 {
            return Err(IdentError::InvalidArgument {
                what: "exclude_impact: the target impact row is zero (no constraint to impose)",
            });
        }
        // Orthonormal basis N (k x (k-1)) of null(c') = the eigenvalue-1
        // subspace of the projector Pperp = I - c c' / (c'c).
        let mut pperp = Mat::<f64>::zeros(k, k);
        for i in 0..k {
            for j in 0..k {
                let ident = if i == j { 1.0 } else { 0.0 };
                pperp[(i, j)] = ident - c[i] * c[j] / cc;
            }
        }
        let eig_p = pperp.self_adjoint_eigen(Side::Lower).map_err(|_| {
            IdentError::Linalg(tsecon_linalg::LinalgError::EigenFailed {
                what: "max-share impact-exclusion projector",
            })
        })?;
        let up = eig_p.U();
        // Eigenvalues ascending [0, 1, ..., 1]; columns 1..k are the eval-1
        // subspace. N is k x (k-1).
        let n_basis = Mat::from_fn(k, k - 1, |i, col| up[(i, col + 1)]);

        // Reduced objective N' A N ((k-1) x (k-1)).
        let a_red = Mat::from_fn(k - 1, k - 1, |i, j| {
            let mut s = 0.0;
            for u in 0..k {
                for v in 0..k {
                    s += n_basis[(u, i)] * a[(u, v)] * n_basis[(v, j)];
                }
            }
            s
        });
        let eig = a_red.self_adjoint_eigen(Side::Lower).map_err(|_| {
            IdentError::Linalg(tsecon_linalg::LinalgError::EigenFailed {
                what: "max-share reduced eigenproblem",
            })
        })?;
        let evals: Vec<f64> = eig.S().column_vector().iter().copied().collect();
        let uu = eig.U();
        let top = k - 2; // leading column of the (k-1)-dim problem
                         // q* = N z*, z* = leading eigenvector of the reduced problem.
        let q: Vec<f64> = (0..k)
            .map(|i| (0..k - 1).map(|c2| n_basis[(i, c2)] * uu[(c2, top)]).sum())
            .collect();
        let lambda_max = evals[top];
        (q, evals, lambda_max)
    } else {
        let eig = a.self_adjoint_eigen(Side::Lower).map_err(|_| {
            IdentError::Linalg(tsecon_linalg::LinalgError::EigenFailed {
                what: "max-share eigenproblem",
            })
        })?;
        let evals: Vec<f64> = eig.S().column_vector().iter().copied().collect();
        let u = eig.U();
        let top = k - 1;
        let q: Vec<f64> = (0..k).map(|i| u[(i, top)]).collect();
        let lambda_max = evals[top];
        (q, evals, lambda_max)
    };

    // Structural IRF irf[s] = Theta_s q* (matrix-vector product per horizon).
    let mut irf: Vec<Vec<f64>> = (0..=horizon)
        .map(|s| {
            (0..k)
                .map(|i| (0..k).map(|j| theta[s][(i, j)] * q[j]).sum())
                .collect()
        })
        .collect();
    let mut q = q;

    // Sign normalization.
    let flip = match sign {
        MaxShareSign::Cumsum => {
            let s_target: f64 = (h0..=h1).map(|s| irf[s][target]).sum();
            s_target < 0.0
        }
        MaxShareSign::Impact => irf[0][target] < 0.0,
        MaxShareSign::None => false,
    };
    if flip {
        for v in &mut q {
            *v = -*v;
        }
        for h in &mut irf {
            for v in h {
                *v = -*v;
            }
        }
    }

    let impact = irf[0].clone();

    // FEV share of the target explained by the identified shock over the window.
    let share_window = lambda_max / trace_a;

    // Cumulative FEV-share profile (single-shock fevd formula for the target).
    let mut fev_share = Vec::with_capacity(horizon + 1);
    let mut num = 0.0;
    let mut den = 0.0;
    for (irf_h, theta_h) in irf.iter().zip(theta.iter()) {
        num += irf_h[target] * irf_h[target];
        den += (0..k).map(|j| theta_h[(target, j)].powi(2)).sum::<f64>();
        fev_share.push(num / den);
    }

    Ok(MaxShareResult {
        irf,
        impact,
        q,
        share_window,
        fev_share,
        eigenvalues,
    })
}

/// Builds the symmetric PSD objective matrix `A` (`k x k`).
fn objective_matrix(
    theta: &[Mat<f64>],
    target: usize,
    h0: usize,
    h1: usize,
    weighting: MaxShareWeighting,
    k: usize,
) -> Result<Mat<f64>, IdentError> {
    let row = |s: usize| -> Vec<f64> { (0..k).map(|j| theta[s][(target, j)]).collect() };
    match weighting {
        MaxShareWeighting::Window => {
            let mut a = Mat::<f64>::zeros(k, k);
            for s in h0..=h1 {
                let r = row(s);
                for i in 0..k {
                    for j in 0..k {
                        a[(i, j)] += r[i] * r[j];
                    }
                }
            }
            Ok(a)
        }
        MaxShareWeighting::Cumulative => {
            // A = sum_{h=h0..h1} C_h / tr(C_h), C_h = sum_{s=0..h} r_s r_s',
            // accumulated from s = 0.
            let mut a = Mat::<f64>::zeros(k, k);
            let mut c_h = Mat::<f64>::zeros(k, k);
            let mut d_h = 0.0;
            for h in 0..=h1 {
                let r = row(h);
                for i in 0..k {
                    for j in 0..k {
                        c_h[(i, j)] += r[i] * r[j];
                    }
                }
                d_h += r.iter().map(|v| v * v).sum::<f64>();
                if h >= h0 {
                    // d_h is a finite non-negative sum of squares by construction.
                    if d_h <= 0.0 {
                        return Err(IdentError::InvalidArgument {
                            what: "cumulative weighting: the target FEV denominator is zero",
                        });
                    }
                    for i in 0..k {
                        for j in 0..k {
                            a[(i, j)] += c_h[(i, j)] / d_h;
                        }
                    }
                }
            }
            Ok(a)
        }
    }
}
