//! Asymptotic (Lütkepohl 1990 delta-method) standard errors of the VAR
//! impulse responses — the analytic, closed-form companion to the point
//! IRFs in [`crate::irf`].
//!
//! These are the standard errors underlying the frequentist
//! `method="asymptotic"` branch of the Python `var_irf_bands`: for each
//! horizon `h` and cell `(i, j)` (response of variable `i` to a shock in
//! variable `j`) they give the delta-method standard error of the
//! impulse-response coefficient, from which symmetric Wald bands
//! `point ± z_{1-α/2}·se` are formed by the caller.
//!
//! Every formula and every intermediate object matches statsmodels
//! `IRAnalysis.stderr` / `cum_effect_stderr` (Lütkepohl 2005, sections
//! 3.7.1–3.7.2); the golden fixture `fixtures/var_irf_bands.json`
//! arbitrates to `rtol ≤ 1e-6`.
//!
//! ## The algebra (Lütkepohl 2005, ch. 3.7)
//!
//! The reduced-form responses `Φ_h = J A^h J'` (companion form) have
//! asymptotic covariance
//!
//! ```text
//! Cov(vec Φ_h) = G_h Σ_α G_h',
//!   G_h = ∂ vec(Φ_h) / ∂ vec(A)' = Σ_{m=0}^{h-1} (A')^{h-1-m}[:k] ⊗ Φ_m,
//! ```
//!
//! with `Σ_α = (Z'Z)^{-1} ⊗ Σ_u` (restricted to the lag coefficients,
//! deterministic terms dropped). The orthogonalized responses
//! `Θ_h = Φ_h P` (`P = chol Σ_u`) add a term in `vech(Σ_u)`:
//!
//! ```text
//! Cov(vec Θ_h) = C_h Σ_α C_h' + (1/T) C̄_h Σ_σ C̄_h',
//!   C_h = (P' ⊗ I_k) G_h,     C̄_h = (I_k ⊗ Φ_h) H,
//!   Σ_σ = 2 D_k^+ (Σ_u ⊗ Σ_u) D_k^{+'},
//!   H   = L_k' B^{-1},  B = L_k [ (I_k ⊗ P) K_{kk} + (P ⊗ I_k) ] L_k',
//! ```
//!
//! where `D_k`, `L_k`, `K_{kk}` are the duplication, elimination, and
//! commutation matrices. The cumulative variants replace `G_h` by
//! `F_h = Σ_{i≤h} G_i` and `Φ_h` by the cumulated response `Ξ_h`.
//!
//! The standard error of cell `(i, j)` at horizon `h` is
//! `sqrt` of the `(j k + i)`-th diagonal entry of `Cov(vec Φ_h)` — the
//! column-stacking (`vec`) index of `Φ_h[i, j]`.

use tsecon_linalg::faer::linalg::solvers::DenseSolveCore;
use tsecon_linalg::faer::{Mat, MatRef};

use crate::error::VarError;
use crate::results::{chol_lower, VarResults};

/// Kronecker product `a ⊗ b` (numpy `np.kron` convention):
/// `(a ⊗ b)[i·b_r + p, j·b_c + q] = a[i, j] · b[p, q]`.
fn kron(a: MatRef<'_, f64>, b: MatRef<'_, f64>) -> Mat<f64> {
    let (br, bc) = (b.nrows(), b.ncols());
    Mat::from_fn(a.nrows() * br, a.ncols() * bc, |r, c| {
        a[(r / br, c / bc)] * b[(r % br, c % bc)]
    })
}

/// `g · mid · g'`, the delta-method sandwich.
fn sandwich(g: &Mat<f64>, mid: &Mat<f64>) -> Mat<f64> {
    let gt = g.transpose().to_owned();
    let left = g * mid;
    &left * &gt
}

/// Elimination matrix `L_k` (`k(k+1)/2 × k²`) with `vech(M) = L_k vec(M)`
/// — one unit row per lower-triangular position in `vec` (column-major)
/// order, matching statsmodels `tsatools.elimination_matrix`.
fn elimination_matrix(k: usize) -> Mat<f64> {
    let half = k * (k + 1) / 2;
    let mut l = Mat::<f64>::zeros(half, k * k);
    let mut r = 0;
    for a in 0..k * k {
        let (row, col) = (a % k, a / k); // vec index a = col·k + row
        if row >= col {
            l[(r, a)] = 1.0;
            r += 1;
        }
    }
    l
}

/// Commutation matrix `K_{k,k}` (`k² × k²`) with `vec(A') = K vec(A)`;
/// `K[i·k + j, j·k + i] = 1` (statsmodels `tsatools.commutation_matrix`).
fn commutation_matrix(k: usize) -> Mat<f64> {
    let mut km = Mat::<f64>::zeros(k * k, k * k);
    for i in 0..k {
        for j in 0..k {
            km[(i * k + j, j * k + i)] = 1.0;
        }
    }
    km
}

/// Duplication matrix `D_k` (`k² × k(k+1)/2`) with `vec(S) = D_k vech(S)`
/// for symmetric `S`; the `vech` order is the upper-triangle row-major
/// enumeration used by statsmodels `tsatools.duplication_matrix`.
fn duplication_matrix(k: usize) -> Mat<f64> {
    let half = k * (k + 1) / 2;
    let mut d = Mat::<f64>::zeros(k * k, half);
    let mut c = 0;
    for i in 0..k {
        for j in i..k {
            // vec index of (row=i, col=j) is j·k + i; symmetric partner
            // (row=j, col=i) is i·k + j.
            d[(j * k + i, c)] = 1.0;
            if i != j {
                d[(i * k + j, c)] = 1.0;
            }
            c += 1;
        }
    }
    d
}

/// Inverse of a general square matrix via LU with partial pivoting.
fn inv_general(m: &Mat<f64>, what: &'static str) -> Result<Mat<f64>, VarError> {
    let inv = m.partial_piv_lu().inverse();
    for j in 0..inv.ncols() {
        for i in 0..inv.nrows() {
            if !inv[(i, j)].is_finite() {
                return Err(VarError::NotPositiveDefinite { what });
            }
        }
    }
    Ok(inv)
}

/// Moore–Penrose pseudoinverse of a full-column-rank matrix,
/// `A^+ = (A'A)^{-1} A'` (exact for the duplication matrix, which has
/// full column rank).
fn pinv_full_col_rank(a: &Mat<f64>, what: &'static str) -> Result<Mat<f64>, VarError> {
    let at = a.transpose().to_owned();
    let ata = &at * a; // (A'A), square full rank
    let ata_inv = inv_general(&ata, what)?;
    Ok(&ata_inv * &at)
}

/// Standard-error matrices `se[h][(i, j)] = sqrt` of the `(j k + i)`-th
/// diagonal of `covs[h]` (the `vec` index of `Φ_h[i, j]`), reshaped `k×k`
/// — the elementwise analogue of statsmodels `unvec(sqrt(diag(cov)))`.
fn se_from_cov(covs: &[Mat<f64>], k: usize) -> Vec<Mat<f64>> {
    covs.iter()
        .map(|c| {
            Mat::from_fn(k, k, |i, j| {
                let idx = j * k + i;
                c[(idx, idx)].max(0.0).sqrt()
            })
        })
        .collect()
}

/// The three horizon-independent building blocks of the orthogonalized
/// delta-method covariance: `(P' ⊗ I_k)`, `H = L_k' B^{-1}`, and the
/// `vech(Σ_u)` covariance `Σ_σ = 2 D_k^+ (Σ_u ⊗ Σ_u) D_k^{+'}`.
type OrthDeltaPieces = (Mat<f64>, Mat<f64>, Mat<f64>);

fn orth_delta_pieces(
    sigma_u: MatRef<'_, f64>,
    p_chol: &Mat<f64>,
    ik: &Mat<f64>,
    k: usize,
) -> Result<OrthDeltaPieces, VarError> {
    let pik = kron(p_chol.transpose(), ik.as_ref()); // (P' ⊗ I_k)

    // H = L_k' B^{-1},  B = L_k [ (I⊗P) K + (P⊗I) ] L_k'.
    let lk = elimination_matrix(k);
    let kkk = commutation_matrix(k);
    let ikp = kron(ik.as_ref(), p_chol.as_ref()); // (I_k ⊗ P)
    let pik_full = kron(p_chol.as_ref(), ik.as_ref()); // (P ⊗ I_k)
    let inner = &(&ikp * &kkk) + &pik_full;
    let lkt = lk.transpose().to_owned();
    let b = &(&lk * &inner) * &lkt;
    let b_inv = inv_general(&b, "orthogonalized IRF band matrix B")?;
    let h_mat = &lkt * &b_inv; // (k² × k(k+1)/2)

    // Sigma_sigma = 2 D_k^+ (Sigma_u ⊗ Sigma_u) D_k^{+'}.
    let dk = duplication_matrix(k);
    let dk_pinv = pinv_full_col_rank(&dk, "duplication matrix")?;
    let sigxsig = kron(sigma_u, sigma_u);
    let dpt = dk_pinv.transpose().to_owned();
    let cov_sig_half = &(&dk_pinv * &sigxsig) * &dpt;
    let cov_sig = Mat::from_fn(cov_sig_half.nrows(), cov_sig_half.ncols(), |i, j| {
        2.0 * cov_sig_half[(i, j)]
    });

    Ok((pik, h_mat, cov_sig))
}

/// Delta-method (Lütkepohl 1990) asymptotic standard errors of the VAR
/// impulse responses to `horizon` periods.
///
/// Returns a vector of `horizon + 1` matrices, each `k × k`, whose
/// `(i, j)` entry is the standard error of the response of variable `i`
/// to a shock in variable `j` at that horizon:
///
/// * `orth = false` — reduced-form (forecast-error) responses `Φ_h`;
///   `orth = true` — Cholesky-orthogonalized responses `Θ_h = Φ_h P`
///   (one-standard-deviation structural shocks, recursive ordering).
/// * `cumulative = false` — the per-horizon responses; `cumulative =
///   true` — the cumulated responses `Ξ_h = Σ_{i≤h} Φ_i`.
///
/// Matches statsmodels `IRAnalysis.stderr(orth=…)` (non-cumulative) and
/// `cum_effect_stderr(orth=…)` (cumulative) to `rtol ≤ 1e-6`.
///
/// # Errors
///
/// * [`VarError::InvalidArgument`] if the fit has no lags (a VAR(0) has
///   no coefficient covariance to propagate);
/// * [`VarError::NotPositiveDefinite`] if `sigma_u` has no Cholesky
///   factor or an intermediate matrix is singular;
/// * propagates [`crate::irf::ma_rep`] / companion-form failures.
pub fn irf_asymptotic_se(
    res: &VarResults,
    horizon: usize,
    orth: bool,
    cumulative: bool,
) -> Result<Vec<Mat<f64>>, VarError> {
    let k = res.neqs;
    let p = res.spec.lags;
    if p == 0 {
        return Err(VarError::InvalidArgument {
            what: "asymptotic IRF standard errors require at least one lag",
        });
    }
    let k2 = k * k;
    let dim = p * k2; // dimension of vec(alpha) = vec([A_1, ..., A_p])
    let t = res.nobs as f64;
    let n_trend = res.df_model - k * p;

    // Non-orthogonalized MA coefficients Phi_0, ..., Phi_horizon.
    let phi = res.ma_rep(horizon)?;

    // Coefficient covariance Sigma_alpha = (Z'Z)^{-1} ⊗ Sigma_u,
    // restricted to the lag block (deterministic terms dropped). Index
    // R = a·k + e picks regressor a (in 0..pk) and equation e (in 0..k).
    let cov_a = Mat::from_fn(dim, dim, |r, c| {
        let (a1, e1) = (r / k + n_trend, r % k);
        let (a2, e2) = (c / k + n_trend, c % k);
        res.zz_inv[(a1, a2)] * res.sigma_u[(e1, e2)]
    });

    // First k rows of (A')^idx for idx = 0..horizon-1 (companion form).
    let comp = res.companion()?;
    let at = comp.transpose().to_owned();
    let kp = at.nrows();
    let mut atpow: Vec<Mat<f64>> = Vec::with_capacity(horizon.max(1));
    let mut cur = Mat::<f64>::from_fn(kp, kp, |i, j| f64::from(u8::from(i == j)));
    for _ in 0..horizon {
        atpow.push(cur.submatrix(0, 0, k, kp).to_owned());
        cur = &cur * &at;
    }

    // Jacobians G_i = sum_{m=0}^{i-1} (A')^{i-1-m}[:k] ⊗ Phi_m, i = 1..H.
    let mut g: Vec<Mat<f64>> = Vec::with_capacity(horizon);
    for i in 1..=horizon {
        let mut gi = Mat::<f64>::zeros(k2, dim);
        for m in 0..i {
            let piece = kron(atpow[i - 1 - m].as_ref(), phi[m].as_ref());
            gi = &gi + &piece;
        }
        g.push(gi);
    }

    // Cholesky factor P and I_k reused by the orthogonalized branches.
    let p_chol = chol_lower(res.sigma_u.as_ref(), "sigma_u")?;
    let ik = Mat::<f64>::from_fn(k, k, |i, j| f64::from(u8::from(i == j)));

    let mut covs: Vec<Mat<f64>> = Vec::with_capacity(horizon + 1);

    if !orth && !cumulative {
        covs.push(Mat::<f64>::zeros(k2, k2));
        for gi in g.iter() {
            covs.push(sandwich(gi, &cov_a));
        }
    } else if orth && !cumulative {
        let (pik, h_mat, cov_sig) = orth_delta_pieces(res.sigma_u.as_ref(), &p_chol, &ik, k)?;
        for i in 0..=horizon {
            let apiece = if i == 0 {
                Mat::<f64>::zeros(k2, k2)
            } else {
                let ci = &pik * &g[i - 1];
                sandwich(&ci, &cov_a)
            };
            let cibar = &kron(ik.as_ref(), phi[i].as_ref()) * &h_mat;
            let bpiece_raw = sandwich(&cibar, &cov_sig);
            let cov = Mat::from_fn(k2, k2, |r, c| apiece[(r, c)] + bpiece_raw[(r, c)] / t);
            covs.push(cov);
        }
    } else if !orth && cumulative {
        let mut f = Mat::<f64>::zeros(k2, dim);
        for i in 0..=horizon {
            if i > 0 {
                f = &f + &g[i - 1];
            }
            if i == 0 {
                covs.push(Mat::<f64>::zeros(k2, k2));
            } else {
                covs.push(sandwich(&f, &cov_a));
            }
        }
    } else {
        // orth && cumulative
        let (pik, h_mat, cov_sig) = orth_delta_pieces(res.sigma_u.as_ref(), &p_chol, &ik, k)?;
        // Cumulated non-orth responses Xi_h.
        let mut xi = Vec::with_capacity(horizon + 1);
        let mut acc = Mat::<f64>::zeros(k, k);
        for phi_m in phi.iter() {
            acc = &acc + phi_m;
            xi.push(acc.clone());
        }
        let mut f = Mat::<f64>::zeros(k2, dim);
        for i in 0..=horizon {
            if i > 0 {
                f = &f + &g[i - 1];
            }
            let apiece = if i == 0 {
                Mat::<f64>::zeros(k2, k2)
            } else {
                let bn = &pik * &f;
                sandwich(&bn, &cov_a)
            };
            let bnbar = &kron(ik.as_ref(), xi[i].as_ref()) * &h_mat;
            let bpiece_raw = sandwich(&bnbar, &cov_sig);
            let cov = Mat::from_fn(k2, k2, |r, c| apiece[(r, c)] + bpiece_raw[(r, c)] / t);
            covs.push(cov);
        }
    }

    Ok(se_from_cov(&covs, k))
}
