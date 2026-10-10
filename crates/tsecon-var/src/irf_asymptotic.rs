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
pub(crate) fn kron(a: MatRef<'_, f64>, b: MatRef<'_, f64>) -> Mat<f64> {
    let (br, bc) = (b.nrows(), b.ncols());
    Mat::from_fn(a.nrows() * br, a.ncols() * bc, |r, c| {
        a[(r / br, c / bc)] * b[(r % br, c % bc)]
    })
}

/// `g · mid · g'`, the delta-method sandwich.
pub(crate) fn sandwich(g: &Mat<f64>, mid: &Mat<f64>) -> Mat<f64> {
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
pub(crate) fn duplication_matrix(k: usize) -> Mat<f64> {
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
pub(crate) fn pinv_full_col_rank(a: &Mat<f64>, what: &'static str) -> Result<Mat<f64>, VarError> {
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
/// * [`VarError::SizeBudget`] when the delta-method working set or the
///   Jacobian work exceeds its budget ([`check_irf_asymptotic_budget`]),
///   checked before anything is allocated;
/// * [`VarError::NotPositiveDefinite`] if `sigma_u` has no Cholesky
///   factor or an intermediate matrix is singular;
/// * propagates [`crate::irf::ma_rep`] / companion-form failures.
pub fn irf_asymptotic_se(
    res: &VarResults,
    horizon: usize,
    orth: bool,
    cumulative: bool,
) -> Result<Vec<Mat<f64>>, VarError> {
    if res.spec.lags > 0 {
        check_irf_asymptotic_budget(res.neqs, res.spec.lags, horizon, None, 0)?;
    }
    let w = delta_workspace(res, horizon)?;
    let (k, k2, dim, t) = (w.k, w.k2, w.dim, w.t);
    let (phi, cov_a, g, p_chol, ik) = (&w.phi, &w.cov_a, &w.g, &w.p_chol, &w.ik);

    let mut covs: Vec<Mat<f64>> = Vec::with_capacity(horizon + 1);

    if !orth && !cumulative {
        covs.push(Mat::<f64>::zeros(k2, k2));
        for gi in g.iter() {
            covs.push(sandwich(gi, cov_a));
        }
    } else if orth && !cumulative {
        let (pik, h_mat, cov_sig) = orth_delta_pieces(res.sigma_u.as_ref(), p_chol, ik, k)?;
        for i in 0..=horizon {
            let apiece = if i == 0 {
                Mat::<f64>::zeros(k2, k2)
            } else {
                let ci = &pik * &g[i - 1];
                sandwich(&ci, cov_a)
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
                covs.push(sandwich(&f, cov_a));
            }
        }
    } else {
        // orth && cumulative
        let (pik, h_mat, cov_sig) = orth_delta_pieces(res.sigma_u.as_ref(), p_chol, ik, k)?;
        // Cumulated non-orth responses Xi_h.
        let xi = cumulate(phi);
        let mut f = Mat::<f64>::zeros(k2, dim);
        for i in 0..=horizon {
            if i > 0 {
                f = &f + &g[i - 1];
            }
            let apiece = if i == 0 {
                Mat::<f64>::zeros(k2, k2)
            } else {
                let bn = &pik * &f;
                sandwich(&bn, cov_a)
            };
            let bnbar = &kron(ik.as_ref(), xi[i].as_ref()) * &h_mat;
            let bpiece_raw = sandwich(&bnbar, &cov_sig);
            let cov = Mat::from_fn(k2, k2, |r, c| apiece[(r, c)] + bpiece_raw[(r, c)] / t);
            covs.push(cov);
        }
    }

    Ok(se_from_cov(&covs, k))
}

/// Running total `Xi_h = sum_{i <= h} M_i` of a horizon-indexed cube.
fn cumulate(m: &[Mat<f64>]) -> Vec<Mat<f64>> {
    let mut out = Vec::with_capacity(m.len());
    let mut acc = Mat::<f64>::zeros(m[0].nrows(), m[0].ncols());
    for mh in m.iter() {
        acc = &acc + mh;
        out.push(acc.clone());
    }
    out
}

/// The horizon-independent delta-method building blocks, shared verbatim by
/// [`irf_asymptotic_se`] (which turns them into per-horizon variances) and by
/// [`irf_asymptotic_critical_values`] (which turns them into a *cross-horizon*
/// covariance). Extracting them moves no arithmetic: every expression below is
/// the one `irf_asymptotic_se` used before the simultaneous-band work, which is
/// what `tests/pointwise_bitwise_baseline.rs` pins.
pub(crate) struct DeltaWorkspace {
    /// Number of series.
    pub(crate) k: usize,
    /// `k * k`, the length of `vec(Phi_h)`.
    pub(crate) k2: usize,
    /// `p * k^2`, the length of `vec(alpha)`.
    pub(crate) dim: usize,
    /// Effective sample size `T`, the divisor on the `vech(Sigma_u)` term.
    pub(crate) t: f64,
    /// Non-orthogonalized MA coefficients `Phi_0, ..., Phi_horizon`.
    pub(crate) phi: Vec<Mat<f64>>,
    /// `Sigma_alpha = (Z'Z)^{-1} ⊗ Sigma_u`, lag block only.
    pub(crate) cov_a: Mat<f64>,
    /// Jacobians `G_1, ..., G_horizon` (`g[i - 1]` is `G_i`; `G_0 = 0`).
    pub(crate) g: Vec<Mat<f64>>,
    /// Lower Cholesky factor `P` of `Sigma_u`.
    pub(crate) p_chol: Mat<f64>,
    /// `I_k`.
    pub(crate) ik: Mat<f64>,
}

pub(crate) fn delta_workspace(
    res: &VarResults,
    horizon: usize,
) -> Result<DeltaWorkspace, VarError> {
    let k = res.neqs;
    let p = res.spec.lags;
    if p == 0 {
        return Err(VarError::InvalidArgument {
            what: "asymptotic IRF standard errors need lags >= 1: a VAR(0) has no \
                   dynamics, so there is no coefficient covariance to propagate",
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
    let p_chol = chol_lower(res.sigma_u.as_ref(), "Sigma_u, the residual covariance")?;
    let ik = Mat::<f64>::from_fn(k, k, |i, j| f64::from(u8::from(i == j)));

    Ok(DeltaWorkspace {
        k,
        k2,
        dim,
        t,
        phi,
        cov_a,
        g,
        p_chol,
        ik,
    })
}

// ===========================================================================
// The delta-method size budget (shared with the Blanchard-Quah delta arm)
// ===========================================================================

/// The work cap of the delta-method Jacobians `G_h = sum_{m<h} (A')^{h-1-m}
/// ⊗ Phi_m`, `h = 1..=horizon`: [`delta_jacobian_work`] multiply-adds
/// (`2^28`). The sums are QUADRATIC in `horizon`, so the storage budget
/// alone admits horizons that run for hours. Shared by `var_irf_bands` and
/// `long_run_svar_bands`; measured (`lab/audit/round16/var_budgets/
/// vib_asym_cost.py`) at 0.77 s for `k = 2`, `lags = 8`, `horizon = 1000`
/// (`2^25.9`), 3.4 s at `horizon = 2047`, the largest that fits (`2^28.0`),
/// and 5.4 s for `k = 20`, `lags = 4`, `horizon = 28` (`2^28.0`).
pub const DELTA_JACOBIAN_WORK: u128 = 1 << 28;

/// The Jacobian work `horizon (horizon + 1) / 2 * k^2 * lags k^2` checked
/// against [`DELTA_JACOBIAN_WORK`]. Saturating, so an absurd `horizon`
/// cannot overflow the count meant to refuse it.
pub fn delta_jacobian_work(k: usize, lags: usize, horizon: usize) -> u128 {
    let h = horizon as u128;
    let k2 = (k as u128).saturating_mul(k as u128);
    let dim = (lags as u128).saturating_mul(k2);
    h.saturating_mul(h.saturating_add(1))
        .saturating_div(2)
        .saturating_mul(k2)
        .saturating_mul(dim)
}

/// The size terms of one delta-method band, in `f64` values (`None` when a
/// count overflows), so that a refusal can name the term that is over.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct DeltaTerms {
    /// Everything a pointwise band holds: the per-horizon Jacobians and
    /// loadings, the coefficient covariance, the reported cubes.
    pub(crate) base: Option<usize>,
    /// The joint covariance of the largest band family and the family cell
    /// lists (`Some(0)` without a simultaneous band).
    pub(crate) family: Option<usize>,
    /// The sup-t uniforms and maxima (`Some(0)` unless sup-t).
    pub(crate) sim: Option<usize>,
    /// The sup-t simulation work `n_families * band_n_sim * K^2`.
    pub(crate) sim_work: Option<u128>,
    /// [`delta_jacobian_work`].
    pub(crate) jacobian_work: u128,
}

impl DeltaTerms {
    fn values(&self) -> Option<usize> {
        self.base?.checked_add(self.family?)?.checked_add(self.sim?)
    }

    pub(crate) fn fits(&self) -> bool {
        matches!(self.values(), Some(v) if v <= crate::budget::BUDGET_VALUES)
            && matches!(self.sim_work, Some(w) if w <= simultaneous::SUP_T_MAX_WORK)
            && self.jacobian_work <= DELTA_JACOBIAN_WORK
    }
}

/// The band family of `scope`: `(n_families, K)` for `horizon + 1` horizons
/// of a `k`-variable system, or `None` on overflow.
pub(crate) fn family_dims(k: usize, horizon: usize, scope: IrfBandScope) -> Option<(usize, usize)> {
    let hp1 = horizon.checked_add(1)?;
    match scope {
        IrfBandScope::Horizon => Some((k.checked_mul(k)?, hp1)),
        IrfBandScope::Shock => Some((k, hp1.checked_mul(k)?)),
        IrfBandScope::All => Some((1, hp1.checked_mul(k)?.checked_mul(k)?)),
    }
}

/// The sup-t terms shared by both delta arms: `K band_n_sim` uniforms plus
/// `band_n_sim` maxima, and `n_families band_n_sim K^2` work.
pub(crate) fn sup_t_terms(n_fam: usize, kf: usize, n_sim: usize) -> (Option<usize>, Option<u128>) {
    let values = kf.checked_add(1).and_then(|v| v.checked_mul(n_sim));
    let work = (n_fam as u128)
        .checked_mul(n_sim as u128)
        .and_then(|w| w.checked_mul(kf as u128))
        .and_then(|w| w.checked_mul(kf as u128));
    (values, work)
}

const DELTA_STORAGE_WHAT: &str = "horizon and lags: the delta-method working set (the \
    (horizon + 1) k^2 response cells with their Jacobians in lags * k^2 coefficients, the \
    (lags * k^2)^2 coefficient covariance and the reported bands, counted as real storage) \
    exceeds the 2^28-value (2 GiB) budget, checked before anything is allocated; reduce \
    horizon or lags, or use method=\"bootstrap\"";

const DELTA_FAMILY_WHAT: &str = "horizon and band_scope: a simultaneous band over a family of K \
    cells needs the K x K joint covariance of the family (K = horizon + 1 under \
    band_scope=\"horizon\", k (horizon + 1) under \"shock\", k^2 (horizon + 1) under \"all\"), \
    and with the rest of the delta-method working set that exceeds the 2^28-value (2 GiB) \
    budget, checked before anything is allocated; reduce horizon or narrow band_scope";

pub(crate) const SIM_WHAT: &str = "band_n_sim: the sup-t simulation draws band_n_sim Gaussian \
    vectors per family of K cells (K * band_n_sim uniforms, and n_families * band_n_sim * K^2 \
    multiply-adds to colour them), which exceeds the 2^28-value (2 GiB) memory budget or the \
    2^36 work budget, checked before anything is allocated; lower band_n_sim (100000 is the \
    default) or narrow band_scope";

const SIM_HORIZON_WHAT: &str = "horizon and band_scope: at this horizon a sup-t family of K \
    cells is too large to simulate even band_n_sim = 2 Gaussian vectors (K * band_n_sim \
    uniforms within the 2^28-value (2 GiB) memory budget, n_families * band_n_sim * K^2 \
    multiply-adds within the 2^36 work budget), checked before anything is allocated; reduce \
    horizon or narrow band_scope";

const DELTA_JACOBIAN_WHAT: &str = "horizon: the delta-method Jacobians of the moving-average \
    weights cost horizon * (horizon + 1) / 2 * k^2 * lags * k^2 multiply-adds, QUADRATIC in \
    horizon, which exceeds the 2^28 work cap (a few seconds), checked before any of them is \
    built; reduce horizon or lags, or use method=\"bootstrap\", whose cost is linear in \
    horizon";

/// The refusal, if any, of a delta-method band whose terms at `(horizon,
/// n_sim)` are `terms(horizon, n_sim)`. It names the term that is over, in
/// this order: the pointwise working set (`horizon and lags`), a
/// simultaneous band's family covariance (`horizon and band_scope`), the
/// Jacobian work (`horizon`), and the sup-t simulation (`band_n_sim`, or
/// `horizon and band_scope` when not even 2 draws fit); each with the
/// largest value of the named keyword that passes every check at the other
/// arguments. A pointwise band never hears about `band_scope` or
/// `band_n_sim`, which it does not read.
pub(crate) fn delta_refusal(
    terms: impl Fn(usize, usize) -> DeltaTerms,
    horizon: usize,
    n_sim: usize,
    supt: bool,
) -> Option<VarError> {
    use crate::budget::{largest_fitting, BUDGET_VALUES};
    let t = terms(horizon, n_sim);
    if t.fits() {
        return None;
    }
    let max_horizon = || largest_fitting(0, horizon, |h| terms(h, n_sim).fits());
    let over = |v: Option<usize>| !matches!(v, Some(v) if v <= BUDGET_VALUES);
    let refuse = |what: &'static str| VarError::SizeBudget {
        what,
        name: "horizon",
        value: horizon,
        max: max_horizon(),
    };
    if over(t.base) {
        return Some(refuse(DELTA_STORAGE_WHAT));
    }
    let with_family = t.base.zip(t.family).and_then(|(b, f)| b.checked_add(f));
    if over(with_family) {
        return Some(refuse(DELTA_FAMILY_WHAT));
    }
    // Before the simulation: no band_n_sim can rescue a horizon whose
    // Jacobians are over their cap, so that horizon is what to reduce.
    if t.jacobian_work > DELTA_JACOBIAN_WORK {
        return Some(refuse(DELTA_JACOBIAN_WHAT));
    }
    // Only the sup-t simulation is left over (a pointwise or closed-form band
    // simulates nothing, so it never gets here).
    if supt {
        if let Some(max) = largest_fitting(2, n_sim, |n| terms(horizon, n).fits()) {
            return Some(VarError::SizeBudget {
                what: SIM_WHAT,
                name: "band_n_sim",
                value: n_sim,
                max: Some(max),
            });
        }
    }
    Some(refuse(SIM_HORIZON_WHAT))
}

/// The delta-method terms of [`irf_asymptotic_se`] (`band = None`) and
/// [`irf_asymptotic_critical_values`] (`band = Some`), counted as real
/// storage: the `G_h` Jacobians (`k^2 x lags k^2` each) and the companion
/// powers, the per-horizon covariances, the coefficient covariance, and the
/// reported cubes ([`crate::budget::CUBE_COPIES`] copies); with a band, the
/// second workspace and the per-horizon loadings it builds, and per family
/// the `K x K` joint covariance, its factor and the `K x lags k^2` loading
/// block.
pub(crate) fn irf_delta_terms(
    k: usize,
    lags: usize,
    horizon: usize,
    band: Option<(BandMethod, IrfBandScope)>,
    n_sim: usize,
) -> DeltaTerms {
    let (mut family, mut sim, mut sim_work) = (Some(0usize), Some(0usize), Some(0u128));
    if let Some((method, scope)) = band {
        family = irf_family_values(k, lags, horizon, scope);
        if method == BandMethod::SupT {
            (sim, sim_work) = match family_dims(k, horizon, scope) {
                Some((n_fam, kf)) => sup_t_terms(n_fam, kf, n_sim),
                None => (None, None),
            };
        }
    }
    DeltaTerms {
        base: irf_base_values(k, lags, horizon),
        family,
        sim,
        sim_work,
        jacobian_work: delta_jacobian_work(k, lags, horizon),
    }
}

/// The pointwise term of [`irf_delta_terms`].
fn irf_base_values(k: usize, lags: usize, horizon: usize) -> Option<usize> {
    use crate::budget::{cube_values, dense_values, CUBE_COPIES};
    let k2 = k.checked_mul(k)?;
    let dim = k2.checked_mul(lags)?;
    let kp = k.checked_mul(lags)?;
    let hp1 = horizon.checked_add(1)?;
    // Per horizon: G_h, the companion power (A')^h[:k], the covariance.
    let per_horizon = dense_values(k2, dim)?
        .checked_add(dense_values(k, kp)?)?
        .checked_add(dense_values(k2, k2)?)?;
    hp1.checked_mul(per_horizon)?
        .checked_add(dense_values(dim, dim)?)?
        .checked_add(dense_values(kp, kp)?.checked_mul(2)?)?
        .checked_add(dense_values(k2, dim)?.checked_mul(2)?)?
        .checked_add(dense_values(k2, k2)?.checked_mul(8)?)?
        .checked_add(cube_values(k, hp1, CUBE_COPIES)?)
}

/// The simultaneous-band term of [`irf_delta_terms`]: the second workspace's
/// per-horizon loadings, the largest family's joint covariance with its
/// factor and loading blocks, and the family cell lists.
fn irf_family_values(k: usize, lags: usize, horizon: usize, scope: IrfBandScope) -> Option<usize> {
    use crate::budget::dense_values;
    let k2 = k.checked_mul(k)?;
    let dim = k2.checked_mul(lags)?;
    let half = k.checked_mul(k.checked_add(1)?)? / 2;
    let hp1 = horizon.checked_add(1)?;
    let (_, kf) = family_dims(k, horizon, scope)?;
    let loadings = hp1
        .checked_mul(
            dense_values(k2, dim)?
                .checked_mul(3)?
                .checked_add(dense_values(k2, half)?)?,
        )?
        .checked_add(dense_values(dim, dim)?.checked_mul(2)?)?;
    let joint = kf
        .checked_mul(kf)?
        .checked_mul(6)?
        .checked_add(kf.checked_mul(dim.checked_add(half)?)?.checked_mul(4)?)?;
    let cells = hp1.checked_mul(k2)?.checked_mul(3)?;
    loadings.checked_add(joint)?.checked_add(cells)
}

/// Refuses a `var_irf_bands` delta-method request on a `k`-variable VAR with
/// `lags` lags whose working set, Jacobian work or sup-t simulation exceeds
/// its budget ([`crate::budget::BUDGET_VALUES`], [`DELTA_JACOBIAN_WORK`],
/// [`simultaneous::SUP_T_MAX_WORK`]), naming the keyword whose term is over
/// and the largest value that fits. `band` is `None` for the pointwise band
/// alone; `n_sim` is read only under [`BandMethod::SupT`].
///
/// # Errors
///
/// [`VarError::SizeBudget`] naming `horizon` or `band_n_sim`.
pub fn check_irf_asymptotic_budget(
    k: usize,
    lags: usize,
    horizon: usize,
    band: Option<(BandMethod, IrfBandScope)>,
    n_sim: usize,
) -> Result<(), VarError> {
    let supt = matches!(band, Some((BandMethod::SupT, _)));
    match delta_refusal(
        |h, n| irf_delta_terms(k, lags, h, band, n),
        horizon,
        n_sim,
        supt,
    ) {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

// ===========================================================================
// Simultaneous (sup-t) bands
// ===========================================================================
//
// Everything below is additive. It never touches `irf_asymptotic_se`, and the
// pointwise band it feeds is unchanged to the last bit
// (`tests/pointwise_bitwise_baseline.rs`).
//
// ## Why
//
// A pointwise band is a statement about one cell. Read as a statement about a
// whole impulse-response path it is badly anti-conservative, and the shortfall
// does not shrink with the sample: it is a multiplicity problem, not a
// consistency problem. tsecon's own interval-coverage audit measured a nominal
// 90% pointwise band containing the entire h = 0..12 path in 72.2% of samples
// at T = 500 (and 65.0% at T = 200, 56.7% at T = 100).
//
// A simultaneous band keeps the same point estimate and the same pointwise
// standard errors and replaces only the multiplier: `point ± c·se` with `c`
// chosen so that *every* cell of a declared family is covered at once with
// probability `1 - alpha`. The sup-t construction used here — the `1 - alpha`
// quantile of the maximum absolute t-statistic over the family — is the method
// of Montiel Olea and Plagborg-Møller, "Simultaneous confidence bands: Theory,
// implementation, and an application to SVARs".

use tsecon_rng::Stream;
use tsecon_stats::simultaneous;

/// Which multiplier a band applies to its pointwise standard errors.
///
/// [`BandMethod::Pointwise`] is the pre-existing behaviour and the default
/// everywhere; the other three widen the multiplier without moving the point
/// estimate or the standard errors.
///
/// This enum lives in `irf_asymptotic` because that is where the band algebra
/// is, but it is the shared vocabulary for every banded surface in the crate —
/// [`crate::irf_bootstrap`] and [`crate::forecast`] re-export it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BandMethod {
    /// `z_{1 - alpha/2}`: the ordinary marginal multiplier. Makes **no** joint
    /// promise across cells.
    Pointwise,
    /// Sup-t: the `1 - alpha` quantile of `max_cell |t|`, from bootstrap draws
    /// where they exist and from the delta-method covariance otherwise. The
    /// tightest of the three simultaneous routes, because it uses the actual
    /// dependence across cells.
    SupT,
    /// Šidák: `z` at per-cell level `1 - (1 - alpha)^(1/K)`. Exact under
    /// independence across cells — a condition no impulse-response path meets,
    /// so in practice a mild improvement on Bonferroni, not a correct answer.
    Sidak,
    /// Bonferroni: `z` at per-cell level `alpha / K`. Valid under any
    /// dependence, and correspondingly the loosest.
    Bonferroni,
}

impl BandMethod {
    /// Parse the Python-facing spelling: `"pointwise"`, `"sup-t"`, `"sidak"`,
    /// `"bonferroni"`. `"supt"` and `"sup_t"` are accepted spellings of
    /// `"sup-t"`.
    ///
    /// # Errors
    ///
    /// [`VarError::InvalidArgument`] naming the four accepted values.
    pub fn parse(s: &str) -> Result<Self, VarError> {
        match s {
            "pointwise" => Ok(BandMethod::Pointwise),
            "sup-t" | "supt" | "sup_t" => Ok(BandMethod::SupT),
            "sidak" => Ok(BandMethod::Sidak),
            "bonferroni" => Ok(BandMethod::Bonferroni),
            _ => Err(VarError::InvalidArgument {
                what: "unknown band; expected \"pointwise\" (the default, a marginal \
                       band), \"sup-t\" (simultaneous, tightest), \"sidak\", or \
                       \"bonferroni\"",
            }),
        }
    }

    /// The canonical Python-facing spelling, for echoing back in a result dict.
    pub fn label(self) -> &'static str {
        match self {
            BandMethod::Pointwise => "pointwise",
            BandMethod::SupT => "sup-t",
            BandMethod::Sidak => "sidak",
            BandMethod::Bonferroni => "bonferroni",
        }
    }

    /// Whether this method makes a joint promise over the declared cell family.
    pub fn is_simultaneous(self) -> bool {
        !matches!(self, BandMethod::Pointwise)
    }
}

/// Which cells an IRF band is simultaneous **over**.
///
/// This is the load-bearing choice, not a detail: every cell added to a family
/// widens the band for every other cell in it, so a band whose scope is
/// ambiguous is worse than no band. The scope is therefore reported alongside
/// the band, and the critical value is returned per `(response, shock)` cell so
/// the caller can see exactly which multiplier hit which cell.
///
/// Measured on the audit's own design (a stationary bivariate VAR(1), h = 0..12,
/// alpha = 0.10, orthogonalized), the three scopes cost roughly `c = 2.3`,
/// `2.6`, and `2.8` against a pointwise `z = 1.645`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IrfBandScope {
    /// One family per `(response, shock)` pair: the `horizon + 1` cells of that
    /// single impulse-response path. `K = horizon + 1`.
    ///
    /// **The default.** It is the narrowest defensible family and it is exactly
    /// the object the audit measured — "does the band contain the whole path of
    /// *this* response to *this* shock?" — which is what a reader of a single
    /// IRF panel is implicitly asking.
    Horizon,
    /// One family per shock: every response of every variable to that shock,
    /// over every horizon. `K = k * (horizon + 1)`.
    ///
    /// The right scope when a figure shows one shock's whole column of panels
    /// and the reader draws a conclusion from the column as a whole.
    Shock,
    /// A single family: every horizon, every response, every shock.
    /// `K = k * k * (horizon + 1)`.
    ///
    /// The right scope when a conclusion is read off the entire IRF grid at
    /// once. It is also the most conservative, by a wide margin at large `k`.
    All,
}

impl IrfBandScope {
    /// Parse the Python-facing spelling: `"horizon"`, `"shock"`, `"all"`.
    ///
    /// # Errors
    ///
    /// [`VarError::InvalidArgument`] naming the three accepted values.
    pub fn parse(s: &str) -> Result<Self, VarError> {
        match s {
            "horizon" => Ok(IrfBandScope::Horizon),
            "shock" => Ok(IrfBandScope::Shock),
            "all" => Ok(IrfBandScope::All),
            _ => Err(VarError::InvalidArgument {
                what: "unknown band_scope; expected \"horizon\" (the default: joint \
                       over horizons, separately for each response-shock pair), \
                       \"shock\" (joint over horizons and responses, per shock), or \
                       \"all\" (joint over the whole IRF grid)",
            }),
        }
    }

    /// The canonical Python-facing spelling, for echoing back.
    pub fn label(self) -> &'static str {
        match self {
            IrfBandScope::Horizon => "horizon",
            IrfBandScope::Shock => "shock",
            IrfBandScope::All => "all",
        }
    }
}

/// The multiplier to apply to each impulse-response standard error, plus
/// everything needed to say honestly what the resulting band promises.
///
/// Deliberately *not* a band: the caller already holds `point` and `se` and
/// applies the multiplier itself (or calls [`apply_critical_values`]). That
/// keeps the simultaneous band anchored on bit-identically the same point
/// estimate and standard errors as the pointwise band, so
/// `lower_simultaneous <= lower_pointwise <= upper_pointwise <= upper_simultaneous`
/// holds by construction rather than by coincidence.
#[derive(Debug, Clone, PartialEq)]
pub struct IrfCriticalValues {
    /// `values[i][j]` is the multiplier for every cell of the family that
    /// contains (response `i`, shock `j`). Under [`IrfBandScope::Shock`] the
    /// entries are constant down each column; under [`IrfBandScope::All`] every
    /// entry is the same number.
    pub values: Vec<Vec<f64>>,
    /// Number of cells in each family — the `K` the multiplier answers for.
    pub n_cells: usize,
    /// `n_cells_used[i][j]`: cells of that family with a strictly positive
    /// standard error. Cells pinned by construction (the Cholesky zeros at
    /// `h = 0`, and the whole `orth = false` impact matrix) carry no information
    /// about simultaneous coverage, take no part in choosing the multiplier, and
    /// keep their zero-width band. When this is below [`Self::n_cells`] the band
    /// is simultaneous over fewer cells than it looks.
    pub n_cells_used: Vec<Vec<usize>>,
    /// The pointwise multiplier `z_{1 - alpha/2}`, for reference. Every entry of
    /// [`Self::values`] is `>=` this.
    pub pointwise: f64,
    /// The method that produced [`Self::values`].
    pub method: BandMethod,
    /// The cell family [`Self::values`] is simultaneous over.
    pub scope: IrfBandScope,
    /// The two-sided level, echoed back.
    pub alpha: f64,
}

/// The `(horizon, response, shock)` cells of each family, paired with the
/// `(response, shock)` grid entries that share that family's multiplier.
///
/// Cell order inside a family is horizon-major, then response, then shock — and
/// it is the order the covariance rows and the flattened draws must use.
pub(crate) type Family = (Vec<(usize, usize, usize)>, Vec<(usize, usize)>);

pub(crate) fn irf_families(k: usize, horizon: usize, scope: IrfBandScope) -> Vec<Family> {
    match scope {
        IrfBandScope::Horizon => {
            let mut out = Vec::with_capacity(k * k);
            for i in 0..k {
                for j in 0..k {
                    let cells = (0..=horizon).map(|h| (h, i, j)).collect();
                    out.push((cells, vec![(i, j)]));
                }
            }
            out
        }
        IrfBandScope::Shock => {
            let mut out = Vec::with_capacity(k);
            for j in 0..k {
                let mut cells = Vec::with_capacity(k * (horizon + 1));
                for h in 0..=horizon {
                    for i in 0..k {
                        cells.push((h, i, j));
                    }
                }
                out.push((cells, (0..k).map(|i| (i, j)).collect()));
            }
            out
        }
        IrfBandScope::All => {
            let mut cells = Vec::with_capacity(k * k * (horizon + 1));
            for h in 0..=horizon {
                for i in 0..k {
                    for j in 0..k {
                        cells.push((h, i, j));
                    }
                }
            }
            let grid = (0..k).flat_map(|i| (0..k).map(move |j| (i, j))).collect();
            vec![(cells, grid)]
        }
    }
}

/// Per-horizon delta-method loadings: the rows of `vec(Theta_h)` as linear
/// functions of `vec(alpha)` and of `vech(Sigma_u)`.
///
/// The pointwise standard errors are the diagonal of `L_h Sigma_alpha L_h'`
/// (plus the `vech` term) at a single `h`. A simultaneous band needs the
/// *cross-horizon* blocks `L_h Sigma_alpha L_g'`, which are built from exactly
/// the same loadings — the only new algebra in this module.
struct Loadings {
    /// `alpha[h]`: `k^2 x dim` loading of `vec(Theta_h)` on `vec(alpha)`.
    alpha: Vec<Mat<f64>>,
    /// `sigma[h]`: `k^2 x k(k+1)/2` loading on `vech(Sigma_u)`; `None` for
    /// reduced-form (non-orthogonalized) responses, which do not involve
    /// `Sigma_u` at all.
    sigma: Option<Vec<Mat<f64>>>,
    /// `Sigma_alpha`.
    cov_a: Mat<f64>,
    /// `Sigma_sigma`, the covariance of `vech(Sigma_u)`.
    cov_sig: Option<Mat<f64>>,
    /// `T`.
    t: f64,
    /// Number of series.
    k: usize,
}

fn loadings(
    res: &VarResults,
    w: &DeltaWorkspace,
    horizon: usize,
    orth: bool,
    cumulative: bool,
) -> Result<Loadings, VarError> {
    let (k, k2, dim) = (w.k, w.k2, w.dim);

    // G_h (non-cumulative) or F_h = sum_{i <= h} G_i (cumulative), with the
    // h = 0 entry identically zero: Phi_0 = I does not depend on alpha.
    let mut jac: Vec<Mat<f64>> = Vec::with_capacity(horizon + 1);
    let mut f = Mat::<f64>::zeros(k2, dim);
    for h in 0..=horizon {
        if h == 0 {
            jac.push(Mat::<f64>::zeros(k2, dim));
        } else if cumulative {
            f = &f + &w.g[h - 1];
            jac.push(f.clone());
        } else {
            jac.push(w.g[h - 1].clone());
        }
    }

    if !orth {
        return Ok(Loadings {
            alpha: jac,
            sigma: None,
            cov_a: w.cov_a.clone(),
            cov_sig: None,
            t: w.t,
            k,
        });
    }

    let (pik, h_mat, cov_sig) = orth_delta_pieces(res.sigma_u.as_ref(), &w.p_chol, &w.ik, k)?;
    let alpha: Vec<Mat<f64>> = jac.iter().map(|j| &pik * j).collect();
    let resp = if cumulative {
        cumulate(&w.phi)
    } else {
        w.phi.clone()
    };
    let sigma: Vec<Mat<f64>> = resp
        .iter()
        .map(|m| &kron(w.ik.as_ref(), m.as_ref()) * &h_mat)
        .collect();

    Ok(Loadings {
        alpha,
        sigma: Some(sigma),
        cov_a: w.cov_a.clone(),
        cov_sig: Some(cov_sig),
        t: w.t,
        k,
    })
}

/// Joint delta-method covariance of the listed cells, row-major `K x K`.
///
/// `Sigma = M_alpha Sigma_alpha M_alpha' + (1/T) M_sigma Sigma_sigma M_sigma'`,
/// where row `r` of `M` is row `j*k + i` (the `vec` index of `Theta_h[i, j]`) of
/// the horizon-`h` loading. At `K = 1` this reduces to exactly the variance
/// `irf_asymptotic_se` squares — see
/// `simultaneous_diagonal_matches_pointwise_se` in the tests, which measures the
/// agreement rather than asserting it by construction.
///
/// The result is explicitly symmetrized (the sandwich is symmetric in exact
/// arithmetic but not in floating point, and `sup_t_from_cov` checks symmetry to
/// 1e-8 relative) and its diagonal is clamped at zero, matching
/// `se_from_cov`'s own `.max(0.0)`.
fn joint_cov(l: &Loadings, cells: &[(usize, usize, usize)]) -> Vec<f64> {
    let kk = cells.len();
    let dim = l.cov_a.ncols();
    let m_alpha = Mat::from_fn(kk, dim, |r, c| {
        let (h, i, j) = cells[r];
        l.alpha[h][(j * l.k + i, c)]
    });
    let mut raw = sandwich(&m_alpha, &l.cov_a);
    if let (Some(s), Some(cs)) = (&l.sigma, &l.cov_sig) {
        let half = cs.ncols();
        let m_sig = Mat::from_fn(kk, half, |r, c| {
            let (h, i, j) = cells[r];
            s[h][(j * l.k + i, c)]
        });
        let b = sandwich(&m_sig, cs);
        raw = Mat::from_fn(kk, kk, |r, c| raw[(r, c)] + b[(r, c)] / l.t);
    }
    let mut out = vec![0.0f64; kk * kk];
    for a in 0..kk {
        for b in 0..kk {
            out[a * kk + b] = if a == b {
                raw[(a, a)].max(0.0)
            } else {
                0.5 * (raw[(a, b)] + raw[(b, a)])
            };
        }
    }
    out
}

/// Recommended `n_sim` for the Gaussian simulation behind
/// [`BandMethod::SupT`]'s asymptotic route.
///
/// This is a quantile deep in the tail of a maximum, so it wants a lot of
/// draws and they are cheap. Cost is `O(n_sim * K^2)` **per family**, so the
/// total work scales with the scope: `k^2` families at `K = horizon + 1` under
/// [`IrfBandScope::Horizon`], one family at `K = k^2 (horizon + 1)` under
/// [`IrfBandScope::All`] — the latter being the expensive one.
pub const DEFAULT_N_SIM: usize = 100_000;

/// Critical values for a **simultaneous** band on the asymptotic
/// (delta-method) impulse responses.
///
/// The multiplier is applied to the standard errors
/// [`irf_asymptotic_se`] already returns; nothing else about the band changes.
/// The four methods are:
///
/// * [`BandMethod::Pointwise`] — `z_{1 - alpha/2}` everywhere. Included so that
///   a caller can route all four through one code path; the result is exactly
///   the existing marginal band.
/// * [`BandMethod::SupT`] — the `1 - alpha` quantile of `max |t|` under
///   `N(0, Sigma)`, where `Sigma` is the delta-method covariance of the whole
///   cell family (this function's only genuinely new algebra: the cross-horizon
///   blocks `G_h Sigma_alpha G_g'`, which the pointwise path never forms).
///   Simulated from `n_sim` seeded draws, so the band is a pure function of
///   `seed`; **expose that seed to the user**.
/// * [`BandMethod::Sidak`] / [`BandMethod::Bonferroni`] — closed forms in the
///   number of *non-degenerate* cells. They need no covariance and no seed.
///
/// # Degenerate cells
///
/// Cells pinned by construction have `se = 0`: the above-diagonal impact
/// responses under a Cholesky ordering, and the entire `orth = false` impact
/// matrix. They are excluded from the maximum and from the Šidák/Bonferroni
/// cell count, and keep their zero-width band. [`IrfCriticalValues::n_cells_used`]
/// reports how many cells actually carried information. If a family is
/// *entirely* degenerate (only possible at `horizon = 0, orth = false`) the
/// pointwise multiplier is returned rather than an error, since every band in
/// that family has zero width either way.
///
/// # What the band does and does not fix
///
/// It fixes multiplicity. It inherits everything else from the pointwise band:
/// if the delta-method standard error is too small in finite samples, or the
/// point estimate is biased, the simultaneous band under-covers jointly by
/// about as much as the pointwise band under-covers marginally.
///
/// # Errors
///
/// * [`VarError::InvalidParameter`] if `alpha` is outside `(0, 1)`;
/// * [`VarError::InvalidArgument`] if `n_sim < 2` under [`BandMethod::SupT`];
/// * [`VarError::SizeBudget`] naming `horizon` or `band_n_sim` when the
///   working set, the Jacobian work or the sup-t simulation exceeds its
///   budget ([`check_irf_asymptotic_budget`]), checked before anything is
///   allocated;
/// * anything [`irf_asymptotic_se`] can return (a VAR(0) fit, a non-PSD
///   `Sigma_u`, a singular intermediate matrix);
/// * [`VarError::Stats`] from the simultaneous-band layer.
#[allow(clippy::too_many_arguments)]
pub fn irf_asymptotic_critical_values(
    res: &VarResults,
    horizon: usize,
    orth: bool,
    cumulative: bool,
    alpha: f64,
    method: BandMethod,
    scope: IrfBandScope,
    seed: u64,
    n_sim: usize,
) -> Result<IrfCriticalValues, VarError> {
    if !(alpha > 0.0 && alpha < 1.0) {
        return Err(VarError::InvalidParameter {
            name: "alpha",
            value: alpha,
            requirement: "a value strictly inside (0, 1) — alpha = 0.1 gives 90% bands",
        });
    }
    if method == BandMethod::SupT && n_sim < 2 {
        return Err(VarError::InvalidArgument {
            what: "band_n_sim must be at least 2 to simulate a sup-t critical value; \
                   100000 is the recommended default and 50000 the practical floor \
                   (this is a quantile in the tail of a maximum)",
        });
    }
    if res.spec.lags > 0 {
        check_irf_asymptotic_budget(
            res.neqs,
            res.spec.lags,
            horizon,
            Some((method, scope)),
            n_sim,
        )?;
    }
    // Every route needs the per-cell standard errors (to count degenerate
    // cells); only sup-t needs the full covariance and the RNG.
    let w = delta_workspace(res, horizon)?;
    let l = loadings(res, &w, horizon, orth, cumulative)?;
    critical_values_from_family_covs(
        res.neqs,
        horizon,
        alpha,
        method,
        scope,
        seed,
        n_sim,
        |cells| Ok(joint_cov(&l, cells)),
    )
}

/// The per-family critical-value loop shared by every **asymptotic** IRF band:
/// for each family of [`irf_families`]`(k, horizon, scope)`, `family_cov`
/// returns the row-major `K x K` joint covariance of that family's cells (in
/// the family's cell order, symmetrized, diagonal clamped at zero), and this
/// turns it into one multiplier.
///
/// Sup-t draws its uniforms from substream `f` of
/// `SeedSequence(seed).spawn(n_families)` — one stream per family, filled
/// afresh for each — so the band is a pure function of `seed`. Šidák and
/// Bonferroni use only the count of cells with a strictly positive standard
/// error. Cells whose standard error is zero are pinned by construction and
/// take no part. Extracted verbatim from [`irf_asymptotic_critical_values`]
/// (whose output is bit-identical to before the extraction) so the
/// Blanchard-Quah delta-method arm in [`crate::long_run_bands`] can supply its
/// own `J V J'` blocks.
#[allow(clippy::too_many_arguments)]
pub(crate) fn critical_values_from_family_covs<F>(
    k: usize,
    horizon: usize,
    alpha: f64,
    method: BandMethod,
    scope: IrfBandScope,
    seed: u64,
    n_sim: usize,
    family_cov: F,
) -> Result<IrfCriticalValues, VarError>
where
    F: Fn(&[(usize, usize, usize)]) -> Result<Vec<f64>, VarError>,
{
    let z = simultaneous::pointwise_critical_value(alpha).map_err(VarError::Stats)?;
    let families = irf_families(k, horizon, scope);

    let mut values = vec![vec![z; k]; k];
    let mut used = vec![vec![0usize; k]; k];
    let n_cells = families[0].0.len();

    // Every caller budgets its whole working set first; this backstop keeps
    // the uniforms below from ever reaching the allocator unbudgeted.
    if method == BandMethod::SupT && !simultaneous::sup_t_n_sim_fits(families.len(), n_cells, n_sim)
    {
        let max = simultaneous::sup_t_max_n_sim(families.len(), n_cells);
        return Err(VarError::SizeBudget {
            what: SIM_WHAT,
            name: "band_n_sim",
            value: n_sim,
            max: (max >= 2).then_some(max),
        });
    }

    let mut streams = if method == BandMethod::SupT {
        Stream::substreams(seed, families.len()).map_err(|_| VarError::InvalidArgument {
            what: "cannot spawn one reproducible RNG substream per band family; \
                   reduce the number of series or the horizon",
        })?
    } else {
        Vec::new()
    };
    let mut uniforms = if method == BandMethod::SupT {
        vec![0.0f64; simultaneous::required_uniforms(n_cells, n_sim)]
    } else {
        Vec::new()
    };

    for (f, (cells, grid)) in families.iter().enumerate() {
        let sigma = family_cov(cells)?;
        let se = simultaneous::std_errors_from_cov(&sigma, n_cells).map_err(VarError::Stats)?;
        let n_used = se.iter().filter(|s| **s > 0.0).count();

        let c = if n_used == 0 {
            // Every cell pinned by construction: the band has zero width
            // whatever multiplier we pick, so do not manufacture an error.
            z
        } else {
            match method {
                BandMethod::Pointwise => z,
                BandMethod::SupT => {
                    if let Some(stream) = streams.get_mut(f) {
                        stream.fill_uniform_f64(&mut uniforms);
                    }
                    simultaneous::sup_t_from_cov(&sigma, n_cells, alpha, &uniforms)
                        .map_err(VarError::Stats)?
                }
                BandMethod::Sidak => {
                    simultaneous::sidak_critical_value(alpha, n_used).map_err(VarError::Stats)?
                }
                BandMethod::Bonferroni => simultaneous::bonferroni_critical_value(alpha, n_used)
                    .map_err(VarError::Stats)?,
            }
        };
        for &(i, j) in grid {
            values[i][j] = c;
            used[i][j] = n_used;
        }
    }

    Ok(IrfCriticalValues {
        values,
        n_cells,
        n_cells_used: used,
        pointwise: z,
        method,
        scope,
        alpha,
    })
}

/// The lower and upper bounds of a banded impulse-response cube, each
/// `horizon + 1` matrices of shape `k x k`.
pub type IrfBandBounds = (Vec<Mat<f64>>, Vec<Mat<f64>>);

/// Apply per-`(response, shock)` critical values to an impulse-response cube:
/// `(point - c·se, point + c·se)`.
///
/// The band is anchored on the caller's own `point` and `se`, so it shares the
/// pointwise band's centre exactly and — because every entry of
/// [`IrfCriticalValues::values`] is at least [`IrfCriticalValues::pointwise`] —
/// contains the symmetric pointwise band cell by cell.
///
/// # Errors
///
/// [`VarError::Dimension`] if `point` and `se` disagree in length, or if either
/// is not `k x k` per horizon for the `k` the critical values were built for.
pub fn apply_critical_values(
    point: &[Mat<f64>],
    se: &[Mat<f64>],
    cv: &IrfCriticalValues,
) -> Result<IrfBandBounds, VarError> {
    if point.len() != se.len() {
        return Err(VarError::Dimension {
            what: "the impulse-response point cube and its standard errors must have \
                   the same number of horizons",
            expected: point.len(),
            got: se.len(),
        });
    }
    let k = cv.values.len();
    for m in point.iter().chain(se.iter()) {
        if m.nrows() != k || m.ncols() != k {
            return Err(VarError::Dimension {
                what: "every impulse-response matrix must be k x k for the k the \
                       critical values were built for",
                expected: k,
                got: m.nrows(),
            });
        }
    }
    let lower = point
        .iter()
        .zip(se.iter())
        .map(|(p, s)| Mat::from_fn(k, k, |i, j| p[(i, j)] - cv.values[i][j] * s[(i, j)]))
        .collect();
    let upper = point
        .iter()
        .zip(se.iter())
        .map(|(p, s)| Mat::from_fn(k, k, |i, j| p[(i, j)] + cv.values[i][j] * s[(i, j)]))
        .collect();
    Ok((lower, upper))
}
