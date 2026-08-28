//! Hansen-Seo (2002) two-regime threshold vector error-correction model
//! (threshold cointegration), and the sup-LM test for threshold
//! cointegration with a fixed-regressor bootstrap p-value.
//!
//! This module lives in `tsecon-coint` (not in the univariate threshold
//! home `tsecon-regime`) because the object being thresholded *is* the
//! cointegration machinery of this crate: the regime variable is the
//! error-correction term `w_{t-1}(beta) = beta' y_{t-1}`, the null model of
//! the test is the linear rank-1 VECM this crate already estimates
//! ([`crate::fit_vecm_det`] with [`VecmDeterministic::Constant`]), and the
//! grid for `beta` is centered on that linear Johansen ML estimate.
//!
//! The model (Hansen & Seo 2002, eq. 2-4; rank-1 cointegration):
//!
//! ```text
//! Delta y_t = A_1' X_{t-1}(beta) 1{w_{t-1}(beta) <= gamma}
//!           + A_2' X_{t-1}(beta) 1{w_{t-1}(beta) >  gamma} + u_t,
//! X_{t-1}(beta) = (1, w_{t-1}(beta), Delta y_{t-1}', ..., Delta y_{t-l}')',
//! w_t(beta) = beta' y_t,   beta = (1, beta_2, ..., beta_k)'.
//! ```
//!
//! * [`threshold_vecm`] is their estimator: the concentrated Gaussian MLE
//!   under an unrestricted common error covariance, i.e. **grid search over
//!   `(beta, gamma)` with per-cell OLS** minimizing `ln det SigmaHat(beta,
//!   gamma)`. The `gamma` grid is the trimmed order statistics of
//!   `w_{t-1}(beta)` (each regime keeps at least a `trim` fraction of the
//!   sample — Hansen-Seo's `pi_0`, they suggest 0.05 — and at least `m + 1`
//!   observations so both regime regressions are estimable); the `beta`
//!   grid (bivariate systems only) spans the linear estimate plus/minus
//!   `beta_span` first-order standard errors. For `k > 2` the cointegrating
//!   vector must be supplied (`beta = Some(..)`): a `(k-1)`-dimensional
//!   grid is not searched.
//! * [`hansen_seo_test`] is their sup-LM test of linear cointegration
//!   (`A_1 = A_2`) against the two-regime alternative, with `beta` fixed at
//!   the null (linear VECM) estimate as the paper prescribes. Under the
//!   null `gamma` is an unidentified nuisance parameter (the Davies
//!   problem), so no chi-squared p-value exists; the p-value comes from
//!   their Section-4 **fixed-regressor bootstrap** (Hansen 1996): hold
//!   `X_{t-1}`, `w_{t-1}`, and the null residuals `u~_t` fixed, draw
//!   `y*_t = u~_t eta_t` with scalar `eta_t` iid N(0,1), re-residualize
//!   `y*` on the same regressors, and recompute the same sup-LM over the
//!   same grid. Each replication runs on its own SeedSequence-spawned
//!   Philox substream ([`tsecon_bootstrap::par_replicate`] — the same
//!   contract as `tsecon-regime`'s Hansen-1996 `setar_test`), so the
//!   p-value is bit-identical at any rayon thread count.
//!
//! The pointwise LM statistic (Hansen & Seo 2002, eq. 10-12) is the
//! coefficient-difference quadratic form with **Eicker-White** covariance:
//! with `x_{1t} = X_{t-1} d_{1t}`, `x_{2t} = X_{t-1} d_{2t}`, `M_i = sum_t
//! x_{it} x_{it}'`, `A^_i = M_i^{-1} sum_t x_{it} u~_t'`,
//!
//! ```text
//! Omega_i = sum_t (u~_t u~_t') (x) (x_{it} x_{it}'),
//! V_i     = (I (x) M_i^{-1}) Omega_i (I (x) M_i^{-1}),
//! LM(gamma) = vec(A^_1 - A^_2)' (V_1 + V_2)^{-1} vec(A^_1 - A^_2),
//! ```
//!
//! (`(x)` the Kronecker product, equations the outer index of `vec`), and
//! `sup-LM = max` over the trimmed `gamma` grid. Because the null residuals
//! are orthogonal to the full-sample regressors, `A^_1 - A^_2` equals the
//! coefficient difference of the unrestricted two-regime fit — the LM and
//! Wald numerators coincide; the score form keeps every bootstrap quantity
//! fixed-regressor.
//!
//! **Validation grade (honest):** no third-party Hansen-Seo implementation
//! runs in this container (R and `tsDyn` are unavailable through the egress
//! proxy), so the golden fixture `fixtures/tvecm.json` is an independent
//! NumPy transcription of the published algorithm
//! (`fixtures/generate_tvecm_fixtures.py` — never imports tsecon), pinned
//! at 1e-10 (fixed `beta`) / 1e-8 (estimated `beta`, where the eigensolver
//! enters). Statistical correctness — sup-LM size near nominal under the
//! linear-cointegration null, `(beta, gamma)` recovery under a threshold
//! DGP — is carried by the seeded Monte Carlo property tests in
//! `tests/tvecm_properties.rs`, whose measured numbers are quoted in the
//! model card.
//!
//! Reference: Hansen, B. E. & Seo, B. (2002), "Testing for two-regime
//! threshold cointegration in vector error-correction models",
//! *Journal of Econometrics* 110(2), 293-318. Also Hansen (1996),
//! *Econometrica* 64(2) (the fixed-regressor bootstrap).

use tsecon_bootstrap::{par_replicate, WildWeights};
use tsecon_linalg::faer::MatRef;

use crate::error::CointError;
use crate::linalg::check_finite;
use crate::vecm::{fit_vecm_det, VecmDeterministic};

// ------------------------------------------------------------ small linalg
//
// Row-major Cholesky helpers, mirroring the self-contained solver of
// `tsecon-regime::setar` (the grid loops factor thousands of tiny Gram
// matrices; pulling faer's factorization objects into the sweep would cost
// more in allocation than the arithmetic itself).

/// Lower-triangular Cholesky factor of the symmetric positive-definite
/// row-major `k x k` matrix `a`; `None` if a pivot is not strictly positive.
fn cholesky(a: &[f64], k: usize) -> Option<Vec<f64>> {
    let mut l = vec![0.0_f64; k * k];
    for i in 0..k {
        for j in 0..=i {
            let mut sum = a[i * k + j];
            for m in 0..j {
                sum -= l[i * k + m] * l[j * k + m];
            }
            if i == j {
                if !(sum > 0.0 && sum.is_finite()) {
                    return None;
                }
                l[i * k + i] = sum.sqrt();
            } else {
                l[i * k + j] = sum / l[j * k + j];
            }
        }
    }
    Some(l)
}

/// Solve `L L' x = b` given the lower Cholesky factor `l`.
fn chol_solve(l: &[f64], k: usize, b: &[f64]) -> Vec<f64> {
    let mut x = b.to_vec();
    for i in 0..k {
        for j in 0..i {
            x[i] -= l[i * k + j] * x[j];
        }
        x[i] /= l[i * k + i];
    }
    for i in (0..k).rev() {
        for j in (i + 1)..k {
            x[i] -= l[j * k + i] * x[j];
        }
        x[i] /= l[i * k + i];
    }
    x
}

/// `ln det(M)` from the lower Cholesky factor: `2 sum ln L_ii`.
fn ln_det_from_chol(l: &[f64], k: usize) -> f64 {
    let mut ld = 0.0;
    for i in 0..k {
        ld += l[i * k + i].ln();
    }
    2.0 * ld
}

// ------------------------------------------------------------- the design

/// The usable-sample design at a fixed cointegrating vector `beta`.
struct Design {
    /// Regressor rows `X_{t-1}(beta) = [1, w_{t-1}, Delta y lags]` (n x m).
    x: Vec<Vec<f64>>,
    /// Response rows `Delta y_t` (n x k).
    y: Vec<Vec<f64>>,
    /// Threshold variable `w_{t-1}(beta)` (n).
    w: Vec<f64>,
    n: usize,
    k: usize,
    m: usize,
}

/// Build the design over the usable sample `t = l+1 .. T-1` (0-indexed
/// levels): `n = T - l - 1` rows, `m = 2 + k l` regressors.
fn build_design(endog: MatRef<'_, f64>, l: usize, beta: &[f64]) -> Design {
    let t_total = endog.nrows();
    let k = endog.ncols();
    let p = l + 1;
    let n = t_total - p;
    let m = 2 + k * l;
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut w = Vec::with_capacity(n);
    for i in 0..n {
        let t = p + i;
        let mut wt = 0.0;
        for (j, &bj) in beta.iter().enumerate() {
            wt += bj * endog[(t - 1, j)];
        }
        w.push(wt);
        let mut row = Vec::with_capacity(m);
        row.push(1.0);
        row.push(wt);
        for lag in 1..=l {
            for j in 0..k {
                row.push(endog[(t - lag, j)] - endog[(t - lag - 1, j)]);
            }
        }
        x.push(row);
        y.push((0..k).map(|j| endog[(t, j)] - endog[(t - 1, j)]).collect());
    }
    Design { x, y, w, n, k, m }
}

// -------------------------------------------------------------- the scan

/// The precomputed threshold scan at one `beta`: sorted order, the trimmed
/// (and evenly subsampled) candidate grid, and per-candidate Cholesky
/// factors of the two regime Gram matrices — which depend only on the
/// regressors and the split, so the bootstrap reuses them across
/// replications.
struct Scan {
    /// Row indices sorted by ascending `w`.
    order: Vec<usize>,
    /// Low-regime size `#{w <= gamma}` per candidate (ascending).
    cand_nlow: Vec<usize>,
    /// Candidate thresholds (a subset of the order statistics of `w`).
    cand_gamma: Vec<f64>,
    /// Cholesky factor of the low-regime `X'X` per candidate.
    chol_low: Vec<Vec<f64>>,
    /// Cholesky factor of the high-regime `X'X` per candidate.
    chol_high: Vec<Vec<f64>>,
    /// Cholesky factor of the full-sample `X'X` (the null-fit normal
    /// equations).
    chol_total: Vec<f64>,
    /// Minimum observations per regime actually enforced.
    min_regime: usize,
}

/// Evenly-spaced subsample of `0..count`: `min(count, n_grid)` indices
/// `round(j (count-1) / (n_grid-1))` in exact integer arithmetic
/// (half-up), deduplicated. Requires `n_grid >= 2`.
fn even_indices(count: usize, n_grid: usize) -> Vec<usize> {
    if count <= n_grid {
        return (0..count).collect();
    }
    let mut out = Vec::with_capacity(n_grid);
    for j in 0..n_grid {
        let idx = (2 * j * (count - 1) + (n_grid - 1)) / (2 * (n_grid - 1));
        if out.last() != Some(&idx) {
            out.push(idx);
        }
    }
    out
}

impl Scan {
    fn build(design: &Design, trim: f64, n_grid: usize) -> Result<Scan, CointError> {
        let n = design.n;
        let m = design.m;

        let mut m_total = vec![0.0_f64; m * m];
        for row in &design.x {
            for a in 0..m {
                for b in 0..=a {
                    m_total[a * m + b] += row[a] * row[b];
                }
            }
        }
        for a in 0..m {
            for b in (a + 1)..m {
                m_total[a * m + b] = m_total[b * m + a];
            }
        }
        let chol_total = cholesky(&m_total, m).ok_or(CointError::Singular {
            what: "the full-sample regressor cross-product X'X of the threshold \
                   VECM design (collinear lagged differences, or a constant \
                   error-correction term — check the series and beta)",
        })?;

        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| {
            design.w[a]
                .partial_cmp(&design.w[b])
                .unwrap_or(core::cmp::Ordering::Equal)
        });

        // Each regime must hold at least max(m + 1, ceil(trim * n))
        // observations: m + 1 so both regime regressions are estimable with
        // a residual degree of freedom, ceil(trim * n) is Hansen-Seo's pi_0
        // trimming keeping each regime a nondegenerate sample fraction.
        let min_regime = (m + 1).max((trim * n as f64).ceil() as usize);

        // Feasible candidates: tie-group ends (every observation with
        // w <= gamma joins the low regime together) inside the trimming
        // bounds — then an evenly-spaced subsample of at most n_grid.
        let mut feas_nlow = Vec::new();
        let mut feas_gamma = Vec::new();
        for i in 0..n {
            let row = order[i];
            let group_end = i + 1 == n || design.w[order[i + 1]] > design.w[row];
            let nlow = i + 1;
            if group_end && nlow >= min_regime && nlow <= n - min_regime {
                feas_nlow.push(nlow);
                feas_gamma.push(design.w[row]);
            }
        }
        if feas_gamma.is_empty() {
            // The pre-check guarantees n >= 2 * min_regime, so an empty
            // candidate set means tie groups: observations sharing a w
            // value join the low regime together, and every tie-group
            // boundary falls outside the trimmed regime-size window.
            return Err(CointError::InvalidArgument {
                what: "no feasible threshold candidate survives the trimming: the \
                       error-correction term w_{t-1} takes too few distinct values, \
                       so every tie-group boundary leaves one regime below its \
                       minimum size max(m + 1, ceil(trim * n)) (observations with \
                       equal w always split to the same regime) — lower trim, or \
                       check the series for heavy rounding/discreteness",
            });
        }
        let keep = even_indices(feas_nlow.len(), n_grid);
        let cand_nlow: Vec<usize> = keep.iter().map(|&i| feas_nlow[i]).collect();
        let cand_gamma: Vec<f64> = keep.iter().map(|&i| feas_gamma[i]).collect();

        let mut chol_low = Vec::with_capacity(cand_nlow.len());
        let mut chol_high = Vec::with_capacity(cand_nlow.len());
        let mut xtx_low = vec![0.0_f64; m * m];
        let mut ci = 0usize;
        for (i, &row) in order.iter().enumerate() {
            let xr = &design.x[row];
            for a in 0..m {
                for b in 0..=a {
                    xtx_low[a * m + b] += xr[a] * xr[b];
                }
            }
            while ci < cand_nlow.len() && cand_nlow[ci] == i + 1 {
                let mut lo = xtx_low.clone();
                for a in 0..m {
                    for b in (a + 1)..m {
                        lo[a * m + b] = lo[b * m + a];
                    }
                }
                let hi: Vec<f64> = m_total.iter().zip(&lo).map(|(&t, &l)| t - l).collect();
                let cl = cholesky(&lo, m).ok_or(CointError::Singular {
                    what: "a low-regime X'X at a threshold candidate (a regime \
                           segment with collinear regressors — raise trim)",
                })?;
                let ch = cholesky(&hi, m).ok_or(CointError::Singular {
                    what: "a high-regime X'X at a threshold candidate (a regime \
                           segment with collinear regressors — raise trim)",
                })?;
                chol_low.push(cl);
                chol_high.push(ch);
                ci += 1;
            }
        }

        Ok(Scan {
            order,
            cand_nlow,
            cand_gamma,
            chol_low,
            chol_high,
            chol_total,
            min_regime,
        })
    }

    /// `ln det SigmaHat(gamma)` over the candidate grid (SigmaHat the pooled
    /// ML residual covariance of the two-regime OLS), plus the index of the
    /// first candidate attaining the minimum.
    fn logdet_profile(&self, design: &Design) -> Result<(Vec<f64>, usize), CointError> {
        let m = design.m;
        let k = design.k;
        let n = design.n as f64;

        // Full-sample cross products S = X'Y (per equation) and Y'Y.
        let mut s_total = vec![vec![0.0_f64; m]; k];
        let mut yy_total = vec![0.0_f64; k * k];
        for (xr, yr) in design.x.iter().zip(&design.y) {
            for j in 0..k {
                let yj = yr[j];
                for i in 0..m {
                    s_total[j][i] += xr[i] * yj;
                }
                for j2 in 0..k {
                    yy_total[j * k + j2] += yj * yr[j2];
                }
            }
        }

        let mut path = Vec::with_capacity(self.cand_gamma.len());
        let mut best = 0usize;
        let mut best_val = f64::INFINITY;
        let mut s_low = vec![vec![0.0_f64; m]; k];
        let mut yy_low = vec![0.0_f64; k * k];
        let mut ci = 0usize;
        for (i, &row) in self.order.iter().enumerate() {
            let xr = &design.x[row];
            let yr = &design.y[row];
            for j in 0..k {
                let yj = yr[j];
                for i2 in 0..m {
                    s_low[j][i2] += xr[i2] * yj;
                }
                for j2 in 0..k {
                    yy_low[j * k + j2] += yj * yr[j2];
                }
            }
            while ci < self.cand_nlow.len() && self.cand_nlow[ci] == i + 1 {
                // Pooled residual cross product E = E_low + E_high with
                // E_r = Y_r'Y_r - S_r' M_r^{-1} S_r.
                let b_low: Vec<Vec<f64>> = (0..k)
                    .map(|j| chol_solve(&self.chol_low[ci], m, &s_low[j]))
                    .collect();
                let s_high: Vec<Vec<f64>> = (0..k)
                    .map(|j| {
                        s_total[j]
                            .iter()
                            .zip(&s_low[j])
                            .map(|(&t, &l)| t - l)
                            .collect()
                    })
                    .collect();
                let b_high: Vec<Vec<f64>> = (0..k)
                    .map(|j| chol_solve(&self.chol_high[ci], m, &s_high[j]))
                    .collect();
                let mut sig = vec![0.0_f64; k * k];
                for j in 0..k {
                    for j2 in 0..k {
                        let mut fit_l = 0.0;
                        let mut fit_h = 0.0;
                        for i2 in 0..m {
                            fit_l += s_low[j][i2] * b_low[j2][i2];
                            fit_h += s_high[j][i2] * b_high[j2][i2];
                        }
                        let yy_high = yy_total[j * k + j2] - yy_low[j * k + j2];
                        sig[j * k + j2] = ((yy_low[j * k + j2] - fit_l) + (yy_high - fit_h)) / n;
                    }
                }
                // Symmetrize the floating-point remainder before factoring.
                for j in 0..k {
                    for j2 in (j + 1)..k {
                        let avg = 0.5 * (sig[j * k + j2] + sig[j2 * k + j]);
                        sig[j * k + j2] = avg;
                        sig[j2 * k + j] = avg;
                    }
                }
                let lchol = cholesky(&sig, k).ok_or(CointError::NotPositiveDefinite {
                    what: "the pooled two-regime residual covariance at a \
                           threshold candidate (two response series are \
                           collinear, or a regime is degenerate)",
                })?;
                let val = ln_det_from_chol(&lchol, k);
                if val < best_val {
                    best_val = val;
                    best = ci;
                }
                path.push(val);
                ci += 1;
            }
        }
        Ok((path, best))
    }
}

// ------------------------------------------------------- LM path machinery

/// The per-candidate Hansen-Seo LM statistic of one residual matrix `u`
/// (n x k rows): the coefficient-difference quadratic form with
/// Eicker-White covariance (module docs, eq. 10-12). Returns the path
/// aligned with `scan.cand_gamma` and the index of the first maximum;
/// `Err` names the factorization that failed.
fn lm_path(
    scan: &Scan,
    design: &Design,
    u: &[Vec<f64>],
) -> Result<(Vec<f64>, usize), &'static str> {
    let m = design.m;
    let k = design.k;
    let mk = m * k;

    // Full-sample S = X'U (per equation) and Omega = sum (uu') (x) (xx').
    let mut s_total = vec![vec![0.0_f64; m]; k];
    let mut omega_total = vec![0.0_f64; mk * mk];
    let mut xouter = vec![0.0_f64; m * m];
    for (xr, ur) in design.x.iter().zip(u) {
        for a in 0..m {
            for b in 0..=a {
                let v = xr[a] * xr[b];
                xouter[a * m + b] = v;
                xouter[b * m + a] = v;
            }
        }
        for j in 0..k {
            let uj = ur[j];
            for i in 0..m {
                s_total[j][i] += xr[i] * uj;
            }
            for (j2, &uj2) in ur.iter().enumerate().take(k) {
                let c = uj * uj2;
                let base_r = j * m;
                let base_c = j2 * m;
                for a in 0..m {
                    let rowoff = (base_r + a) * mk + base_c;
                    let xoff = a * m;
                    for b in 0..m {
                        omega_total[rowoff + b] += c * xouter[xoff + b];
                    }
                }
            }
        }
    }

    let mut path = Vec::with_capacity(scan.cand_gamma.len());
    let mut best = 0usize;
    let mut best_val = f64::NEG_INFINITY;
    let mut s_low = vec![vec![0.0_f64; m]; k];
    let mut omega_low = vec![0.0_f64; mk * mk];
    let mut ci = 0usize;
    let mut scratch = vec![0.0_f64; m];
    for (i, &row) in scan.order.iter().enumerate() {
        let xr = &design.x[row];
        let ur = &u[row];
        for a in 0..m {
            for b in 0..=a {
                let v = xr[a] * xr[b];
                xouter[a * m + b] = v;
                xouter[b * m + a] = v;
            }
        }
        for j in 0..k {
            let uj = ur[j];
            for i2 in 0..m {
                s_low[j][i2] += xr[i2] * uj;
            }
            for (j2, &uj2) in ur.iter().enumerate().take(k) {
                let c = uj * uj2;
                let base_r = j * m;
                let base_c = j2 * m;
                for a in 0..m {
                    let rowoff = (base_r + a) * mk + base_c;
                    let xoff = a * m;
                    for b in 0..m {
                        omega_low[rowoff + b] += c * xouter[xoff + b];
                    }
                }
            }
        }
        while ci < scan.cand_nlow.len() && scan.cand_nlow[ci] == i + 1 {
            let cl = &scan.chol_low[ci];
            let ch = &scan.chol_high[ci];
            // vec(A^_1 - A^_2), equations the outer index.
            let mut d = vec![0.0_f64; mk];
            for j in 0..k {
                let b1 = chol_solve(cl, m, &s_low[j]);
                for i2 in 0..m {
                    scratch[i2] = s_total[j][i2] - s_low[j][i2];
                }
                let b2 = chol_solve(ch, m, &scratch);
                for i2 in 0..m {
                    d[j * m + i2] = b1[i2] - b2[i2];
                }
            }
            // V = V_1 + V_2, block (j, j2) = M_1^{-1} O1 M_1^{-1}
            //                              + M_2^{-1} O2 M_2^{-1}.
            let mut v = vec![0.0_f64; mk * mk];
            let mut block = vec![0.0_f64; m * m];
            for j in 0..k {
                for j2 in 0..k {
                    for (lchol, high_regime) in [(cl, false), (ch, true)] {
                        // High-regime block = total - low.
                        for a in 0..m {
                            for b in 0..m {
                                let idx = (j * m + a) * mk + j2 * m + b;
                                block[a * m + b] = if high_regime {
                                    omega_total[idx] - omega_low[idx]
                                } else {
                                    omega_low[idx]
                                };
                            }
                        }
                        // W = M^{-1} O (solve per column), then
                        // B = W M^{-1} = (M^{-1} W')' (solve per row of W).
                        let mut wmat = vec![0.0_f64; m * m];
                        for col in 0..m {
                            for a in 0..m {
                                scratch[a] = block[a * m + col];
                            }
                            let sol = chol_solve(lchol, m, &scratch);
                            for a in 0..m {
                                wmat[a * m + col] = sol[a];
                            }
                        }
                        for a in 0..m {
                            let sol = chol_solve(lchol, m, &wmat[a * m..(a + 1) * m]);
                            for b in 0..m {
                                v[(j * m + a) * mk + j2 * m + b] += sol[b];
                            }
                        }
                    }
                }
            }
            // Symmetrize the floating-point remainder before factoring.
            for a in 0..mk {
                for b in (a + 1)..mk {
                    let avg = 0.5 * (v[a * mk + b] + v[b * mk + a]);
                    v[a * mk + b] = avg;
                    v[b * mk + a] = avg;
                }
            }
            let vchol = cholesky(&v, mk).ok_or(
                "the Eicker-White covariance V_1 + V_2 at a threshold candidate \
                 is singular (a regime holds too few observations relative to \
                 the k*m coefficients tested — raise trim or lower k_ar_diff)",
            )?;
            let sol = chol_solve(&vchol, mk, &d);
            let lm: f64 = d.iter().zip(&sol).map(|(&a, &b)| a * b).sum();
            if lm > best_val {
                best_val = lm;
                best = ci;
            }
            path.push(lm);
            ci += 1;
        }
    }
    Ok((path, best))
}

// ------------------------------------------------------------- validation

/// The smallest usable-row count above `n_failed` that satisfies the
/// Hansen-Seo trimming requirement `n >= 2 * max(m + 1, ceil(trim * n))`
/// — the exact minimum the insufficiency message may claim (`ceil(trim *
/// n)` grows with `n`, so feasibility is not monotone and the bound is
/// found by scanning upward from the failure point). Terminates because
/// `2 * ceil(trim * n) < n + 2` once `n > 2 / (1 - 2 trim)`.
fn min_usable_rows(m: usize, trim: f64, n_failed: usize) -> usize {
    let mut n = (n_failed + 1).max(2 * (m + 1));
    loop {
        let min_regime = (m + 1).max((trim * n as f64).ceil() as usize);
        if n >= 2 * min_regime {
            return n;
        }
        n += 1;
    }
}

fn validate_common(
    endog: MatRef<'_, f64>,
    k_ar_diff: usize,
    trim: f64,
    n_grid: usize,
    n_grid_requirement: &'static str,
) -> Result<(), CointError> {
    let k = endog.ncols();
    if k < 2 {
        return Err(CointError::Dimension {
            what: "threshold cointegration needs at least two series; pass a 2-D \
                   array shaped (n_obs, n_series), observations in rows, oldest \
                   first (a single series is a threshold AR — use setar)",
            expected: 2,
            got: k,
        });
    }
    check_finite(endog, "the data matrix")?;
    if !(trim > 0.0 && trim < 0.5) || !trim.is_finite() {
        return Err(CointError::InvalidArgument {
            what: "trim must satisfy 0 < trim < 0.5: it is Hansen-Seo's pi_0, \
                   the minimum fraction of the sample each regime must keep \
                   (they suggest 0.05)",
        });
    }
    if n_grid < 2 {
        return Err(CointError::InvalidArgument {
            what: n_grid_requirement,
        });
    }
    let t = endog.nrows();
    let p = k_ar_diff + 1;
    let m = 2 + k * k_ar_diff;
    let n = t.saturating_sub(p);
    let min_regime = (m + 1).max((trim * n as f64).ceil() as usize);
    if n < 2 * min_regime {
        return Err(CointError::ThresholdInsufficientObservations {
            needed: min_usable_rows(m, trim, n),
            got: n,
            nobs: t,
            neqs: k,
            k_ar_diff,
            n_regressors: m,
        });
    }
    Ok(())
}

/// Validate and normalize a user-supplied cointegrating vector so its first
/// element is exactly 1 (the Hansen-Seo normalization).
fn normalize_beta(beta: &[f64], k: usize) -> Result<Vec<f64>, CointError> {
    if beta.len() != k {
        return Err(CointError::Dimension {
            what: "beta must have one entry per series (the cointegrating \
                   vector w_t = beta' y_t)",
            expected: k,
            got: beta.len(),
        });
    }
    for &b in beta {
        if !b.is_finite() {
            return Err(CointError::NonFinite {
                what: "the supplied cointegrating vector beta",
                at: None,
            });
        }
    }
    if beta[0] == 0.0 {
        return Err(CointError::InvalidArgument {
            what: "beta[0] must be nonzero: the Hansen-Seo normalization fixes \
                   the first coefficient of the cointegrating vector at 1, so \
                   order the series with the normalized one first",
        });
    }
    Ok(beta.iter().map(|&b| b / beta[0]).collect())
}

/// The linear rank-1 Johansen fit (unrestricted constant), its normalized
/// `beta`, its log-likelihood, and the first-order standard error of each
/// free `beta` coefficient used to center the grid.
fn linear_anchor(
    endog: MatRef<'_, f64>,
    k_ar_diff: usize,
) -> Result<(Vec<f64>, f64, Vec<f64>), CointError> {
    let k = endog.ncols();
    let fit = fit_vecm_det(endog, k_ar_diff, 1, VecmDeterministic::Constant)?;
    let beta: Vec<f64> = (0..k).map(|j| fit.beta[(j, 0)]).collect();

    // First-order (conditional-information) standard errors of the free
    // beta coefficients: Cov(beta_2..k) = [q * H'(R1'R1)H]^{-1} with
    // q = alpha' Sigma_u^{-1} alpha and R1 the lagged levels partialled of
    // the constant and lagged differences — a search-region scale, not an
    // inference formula (superconsistent rate; see Hansen-Seo section 5).
    let t_total = endog.nrows();
    let p = k_ar_diff + 1;
    let n = t_total - p;
    let nb = 1 + k * k_ar_diff; // constant + lagged differences
    let mut xb = Vec::with_capacity(n);
    let mut ylag = Vec::with_capacity(n);
    for i in 0..n {
        let t = p + i;
        let mut row = Vec::with_capacity(nb);
        row.push(1.0);
        for lag in 1..=k_ar_diff {
            for j in 0..k {
                row.push(endog[(t - lag, j)] - endog[(t - lag - 1, j)]);
            }
        }
        xb.push(row);
        ylag.push((0..k).map(|j| endog[(t - 1, j)]).collect::<Vec<f64>>());
    }
    let mut gram = vec![0.0_f64; nb * nb];
    let mut cross = vec![vec![0.0_f64; nb]; k];
    for (xr, yr) in xb.iter().zip(&ylag) {
        for a in 0..nb {
            for b in 0..=a {
                gram[a * nb + b] += xr[a] * xr[b];
            }
            for (cj, &yj) in cross.iter_mut().zip(yr.iter()) {
                cj[a] += xr[a] * yj;
            }
        }
    }
    for a in 0..nb {
        for b in (a + 1)..nb {
            gram[a * nb + b] = gram[b * nb + a];
        }
    }
    let gchol = cholesky(&gram, nb).ok_or(CointError::Singular {
        what: "the short-run regressor cross-product in the beta standard-error \
               step (collinear lagged differences)",
    })?;
    let coefs: Vec<Vec<f64>> = cross.iter().map(|c| chol_solve(&gchol, nb, c)).collect();
    // M = sum_t r1_t r1_t' with r1 the partialled lagged levels.
    let mut mr = vec![0.0_f64; k * k];
    for (xr, yr) in xb.iter().zip(&ylag) {
        let mut r1 = vec![0.0_f64; k];
        for j in 0..k {
            let mut fitv = 0.0;
            for a in 0..nb {
                fitv += coefs[j][a] * xr[a];
            }
            r1[j] = yr[j] - fitv;
        }
        for j in 0..k {
            for j2 in 0..k {
                mr[j * k + j2] += r1[j] * r1[j2];
            }
        }
    }
    // q = alpha' Sigma_u^{-1} alpha.
    let mut sig = vec![0.0_f64; k * k];
    for j in 0..k {
        for j2 in 0..k {
            sig[j * k + j2] = fit.sigma_u[(j, j2)];
        }
    }
    let schol = cholesky(&sig, k).ok_or(CointError::NotPositiveDefinite {
        what: "the linear VECM residual covariance Sigma_u",
    })?;
    let alpha: Vec<f64> = (0..k).map(|j| fit.alpha[(j, 0)]).collect();
    let sinv_a = chol_solve(&schol, k, &alpha);
    let q: f64 = alpha.iter().zip(&sinv_a).map(|(&a, &b)| a * b).sum();
    let mut se = vec![0.0_f64; k];
    for j in 1..k {
        let info = q * mr[j * k + j];
        se[j] = if info > 0.0 && info.is_finite() {
            (1.0 / info).sqrt()
        } else {
            f64::NAN
        };
    }
    Ok((beta, fit.llf, se))
}

// --------------------------------------------------------------- results

/// Output of [`threshold_vecm`]: the Hansen-Seo (2002) two-regime
/// threshold VECM at the concentrated-MLE grid optimum.
///
/// Coefficient rows are equations (`k` of them); within a row the columns
/// are `[constant, ect, Delta y_{t-1} (k entries), ..., Delta y_{t-l}]` —
/// the `ect` column is the loading on the error-correction term
/// `w_{t-1}(beta)`. The "low" regime is `w_{t-1} <= threshold`.
#[derive(Debug, Clone, PartialEq)]
pub struct TvecmResult {
    /// The cointegrating vector at the optimum, normalized `beta[0] = 1`.
    pub beta: Vec<f64>,
    /// The estimated threshold `gamma` (an order statistic of
    /// `w_{t-1}(beta)`).
    pub threshold: f64,
    /// Low-regime coefficients (`k x m`, rows = equations).
    pub coefs_low: Vec<Vec<f64>>,
    /// High-regime coefficients (`k x m`).
    pub coefs_high: Vec<Vec<f64>>,
    /// Eicker-White standard errors of `coefs_low` (`k x m`; the
    /// heteroskedasticity-robust form Hansen-Seo report, no small-sample
    /// degrees-of-freedom correction).
    pub se_low: Vec<Vec<f64>>,
    /// Eicker-White standard errors of `coefs_high` (`k x m`).
    pub se_high: Vec<Vec<f64>>,
    /// Low-regime sample size.
    pub n_low: usize,
    /// High-regime sample size.
    pub n_high: usize,
    /// Usable observations `n = T - k_ar_diff - 1`.
    pub nobs: usize,
    /// Low-regime sample fraction `n_low / n`.
    pub frac_low: f64,
    /// Pooled ML residual covariance `(E_1 + E_2) / n` (`k x k`).
    pub sigma: Vec<Vec<f64>>,
    /// `ln det sigma` — the concentrated-MLE grid criterion at the optimum.
    pub log_det_sigma: f64,
    /// Gaussian log-likelihood `-(n k / 2)(ln 2 pi + 1) - (n/2) ln det
    /// sigma`.
    pub llf: f64,
    /// Log-likelihood of the *linear* VECM at `beta_linear` (the Johansen
    /// ML fit when `beta` was estimated; the OLS fit at the supplied
    /// vector when `beta` was fixed).
    pub llf_linear: f64,
    /// The linear cointegrating vector the grid was centered on (the
    /// supplied vector itself when `beta` was fixed).
    pub beta_linear: Vec<f64>,
    /// The searched grid of second-coordinate values `beta[1]` (empty when
    /// `beta` was fixed; bivariate grid search only).
    pub beta_grid: Vec<f64>,
    /// The error-correction term `w_{t-1}(beta)` per usable row (time
    /// order) — threshold it at `threshold` to recover the regimes.
    pub ect: Vec<f64>,
    /// Minimum per-regime size enforced: `max(m + 1, ceil(trim n))`.
    pub min_regime: usize,
    /// Number of series `k`.
    pub neqs: usize,
    /// Regressors per regime `m = 2 + k * k_ar_diff`.
    pub n_regressors: usize,
    /// Lagged differences `l`.
    pub k_ar_diff: usize,
}

/// Output of [`hansen_seo_test`]: the Hansen-Seo (2002) sup-LM test of
/// linear cointegration against two-regime threshold cointegration, with
/// the fixed-regressor bootstrap p-value.
#[derive(Debug, Clone, PartialEq)]
pub struct HansenSeoTest {
    /// The sup-LM statistic.
    pub stat: f64,
    /// Fixed-regressor bootstrap p-value `(1 + #{LM* >= LM}) / (n_boot +
    /// 1)`. **Not** a chi-squared tail — `gamma` is unidentified under the
    /// null (the Davies problem).
    pub p_value: f64,
    /// The threshold at which the sup is attained.
    pub threshold: f64,
    /// The cointegrating vector used (the null Johansen ML estimate, or
    /// the supplied vector), normalized `beta[0] = 1`.
    pub beta: Vec<f64>,
    /// Number of bootstrap replications.
    pub n_boot: usize,
    /// Usable observations `n = T - k_ar_diff - 1`.
    pub nobs: usize,
    /// The candidate threshold grid (ascending).
    pub thresholds: Vec<f64>,
    /// `LM(gamma)` per candidate, aligned with `thresholds`.
    pub lm_path: Vec<f64>,
    /// The bootstrap sup-LM draws, in replication order (deterministic in
    /// `seed` at any thread count).
    pub boot_stats: Vec<f64>,
    /// Minimum per-regime size enforced.
    pub min_regime: usize,
    /// Number of series `k`.
    pub neqs: usize,
    /// Regressors per regime `m = 2 + k * k_ar_diff`.
    pub n_regressors: usize,
    /// Lagged differences `l`.
    pub k_ar_diff: usize,
}

// ------------------------------------------------------------------- fit

/// Fit the Hansen-Seo (2002) two-regime threshold VECM by concentrated
/// Gaussian MLE: grid search over `(beta, gamma)` with per-cell two-regime
/// OLS, minimizing `ln det SigmaHat`.
///
/// `endog` is `T x k` (observations in rows, oldest first); `k_ar_diff`
/// the number of lagged differences `l`. `trim` is Hansen-Seo's `pi_0`
/// (each regime keeps at least `max(m + 1, ceil(trim n))` observations;
/// they suggest `0.05`). The `gamma` grid is at most `n_grid_gamma`
/// evenly-spaced feasible order statistics of `w_{t-1}(beta)`.
///
/// `beta = None` estimates the cointegrating vector: **bivariate systems
/// only**, over a grid of `n_grid_beta` points spanning the linear
/// Johansen ML estimate plus/minus `beta_span` first-order standard
/// errors. `beta = Some(b)` fixes it (any `k >= 2`; normalized so
/// `b[0] = 1`) and searches `gamma` alone.
///
/// # Errors
///
/// * [`CointError::Dimension`] for `k < 2`, or a `beta` of the wrong
///   length;
/// * [`CointError::InvalidArgument`] for `trim` outside `(0, 0.5)`, a
///   degenerate grid request, `beta[0] = 0`, or `beta = None` with
///   `k > 2` (pass the cointegrating vector — the `(k-1)`-dimensional
///   grid is not searched);
/// * [`CointError::NonFinite`] for NaN/infinite data or `beta`;
/// * [`CointError::ThresholdInsufficientObservations`] when the usable
///   sample cannot hold two trimmed regimes;
/// * [`CointError::Singular`] / [`CointError::NotPositiveDefinite`] on
///   degenerate designs (collinear series or regime segments).
#[allow(clippy::too_many_arguments)]
pub fn threshold_vecm(
    endog: MatRef<'_, f64>,
    k_ar_diff: usize,
    trim: f64,
    n_grid_gamma: usize,
    n_grid_beta: usize,
    beta_span: f64,
    beta: Option<&[f64]>,
) -> Result<TvecmResult, CointError> {
    validate_common(
        endog,
        k_ar_diff,
        trim,
        n_grid_gamma,
        "the gamma grid needs at least 2 points (n_grid_gamma >= 2; \
         Hansen-Seo used 300)",
    )?;
    let k = endog.ncols();

    // The beta candidates and the linear anchor.
    let (beta_cands, beta_linear, llf_linear, beta_grid): (Vec<Vec<f64>>, Vec<f64>, f64, Vec<f64>) =
        match beta {
            Some(b) => {
                let bn = normalize_beta(b, k)?;
                let design = build_design(endog, k_ar_diff, &bn);
                let llf0 = linear_llf(&design)?;
                (vec![bn.clone()], bn, llf0, Vec::new())
            }
            None => {
                if k > 2 {
                    return Err(CointError::InvalidArgument {
                        what: "estimating beta by grid search is supported for \
                               bivariate systems only (the Hansen-Seo grid is \
                               one-dimensional); for k > 2 pass beta = Some(..) — \
                               e.g. the linear estimate from a rank-1 \
                               fit_vecm_det(.., Constant)",
                    });
                }
                if n_grid_beta == 0 {
                    return Err(CointError::InvalidArgument {
                        what: "n_grid_beta must be >= 1 (1 fixes beta at the \
                               linear Johansen estimate)",
                    });
                }
                if !(beta_span >= 0.0 && beta_span.is_finite()) {
                    return Err(CointError::InvalidArgument {
                        what: "beta_span must be finite and >= 0 (the half-width \
                               of the beta grid in linear-estimate standard \
                               errors)",
                    });
                }
                let (bl, llf0, se) = linear_anchor(endog, k_ar_diff)?;
                let se1 = se[1];
                if !se1.is_finite() {
                    return Err(CointError::Singular {
                        what: "the first-order standard error of the linear beta \
                               estimate (needed to center the beta grid) is \
                               degenerate — pass beta = Some(..) instead",
                    });
                }
                let center = bl[1];
                let grid: Vec<f64> = if n_grid_beta == 1 || beta_span == 0.0 {
                    vec![center]
                } else {
                    let lo = center - beta_span * se1;
                    let hi = center + beta_span * se1;
                    (0..n_grid_beta)
                        .map(|i| lo + (hi - lo) * i as f64 / (n_grid_beta - 1) as f64)
                        .collect()
                };
                let cands: Vec<Vec<f64>> = grid.iter().map(|&b2| vec![1.0, b2]).collect();
                (cands, bl, llf0, grid)
            }
        };

    // Concentrated grid search: first strictly smaller ln det wins,
    // iterating beta candidates in grid order, gamma ascending.
    let mut best: Option<(usize, usize, f64)> = None; // (beta idx, cand idx, value)
    for (bi, bcand) in beta_cands.iter().enumerate() {
        let design = build_design(endog, k_ar_diff, bcand);
        let scan = Scan::build(&design, trim, n_grid_gamma)?;
        let (path, arg) = scan.logdet_profile(&design)?;
        let val = path[arg];
        let better = match best {
            None => true,
            Some((_, _, bv)) => val < bv,
        };
        if better {
            best = Some((bi, arg, val));
        }
    }
    let (bi, arg, _) = best.ok_or(CointError::InvalidArgument {
        what: "the beta grid is empty — n_grid_beta must be >= 1",
    })?;
    let beta_hat = beta_cands[bi].clone();
    let design = build_design(endog, k_ar_diff, &beta_hat);
    let scan = Scan::build(&design, trim, n_grid_gamma)?;
    let gamma = scan.cand_gamma[arg];

    // Final refit at (beta^, gamma^) with residuals and Eicker-White SEs.
    let n = design.n;
    let m = design.m;
    let low_rows: Vec<usize> = (0..n).filter(|&t| design.w[t] <= gamma).collect();
    let high_rows: Vec<usize> = (0..n).filter(|&t| design.w[t] > gamma).collect();
    let (coefs_low, se_low, e_low) = regime_ols(&design, &low_rows)?;
    let (coefs_high, se_high, e_high) = regime_ols(&design, &high_rows)?;
    let nf = n as f64;
    let mut sigma = vec![vec![0.0_f64; k]; k];
    for j in 0..k {
        for j2 in 0..k {
            sigma[j][j2] = (e_low[j * k + j2] + e_high[j * k + j2]) / nf;
        }
    }
    let mut sig_flat = vec![0.0_f64; k * k];
    for j in 0..k {
        for j2 in 0..k {
            sig_flat[j * k + j2] = 0.5 * (sigma[j][j2] + sigma[j2][j]);
        }
    }
    let schol = cholesky(&sig_flat, k).ok_or(CointError::NotPositiveDefinite {
        what: "the pooled threshold-VECM residual covariance at the optimum",
    })?;
    let log_det_sigma = ln_det_from_chol(&schol, k);
    let kf = k as f64;
    let llf = -nf * kf / 2.0 * (core::f64::consts::TAU.ln() + 1.0) - nf / 2.0 * log_det_sigma;

    Ok(TvecmResult {
        beta: beta_hat,
        threshold: gamma,
        coefs_low,
        coefs_high,
        se_low,
        se_high,
        n_low: low_rows.len(),
        n_high: high_rows.len(),
        nobs: n,
        frac_low: low_rows.len() as f64 / nf,
        sigma,
        log_det_sigma,
        llf,
        llf_linear,
        beta_linear,
        beta_grid,
        ect: design.w,
        min_regime: scan.min_regime,
        neqs: k,
        n_regressors: m,
        k_ar_diff,
    })
}

/// Gaussian log-likelihood of the *linear* (one-regime) OLS fit of the
/// design — the fixed-`beta` analogue of the Johansen `llf`.
fn linear_llf(design: &Design) -> Result<f64, CointError> {
    let all: Vec<usize> = (0..design.n).collect();
    let (_, _, e) = regime_ols(design, &all)?;
    let k = design.k;
    let nf = design.n as f64;
    let mut sig = vec![0.0_f64; k * k];
    for j in 0..k {
        for j2 in 0..k {
            sig[j * k + j2] = 0.5 * (e[j * k + j2] + e[j2 * k + j]) / nf;
        }
    }
    let schol = cholesky(&sig, k).ok_or(CointError::NotPositiveDefinite {
        what: "the linear-fit residual covariance at the supplied beta",
    })?;
    let kf = k as f64;
    Ok(-nf * kf / 2.0 * (core::f64::consts::TAU.ln() + 1.0)
        - nf / 2.0 * ln_det_from_chol(&schol, k))
}

/// The pieces one regime's OLS hands back: coefficients (`k x m`, rows =
/// equations), standard errors (`k x m`), and the residual cross product
/// `E = U'U` (row-major `k x k`).
type RegimeOlsParts = (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<f64>);

/// OLS of the design's responses on its regressors over `rows`:
/// coefficients (`k x m`, rows = equations), Eicker-White standard errors
/// (`k x m`), and the residual cross product `E = U'U` (row-major
/// `k x k`).
fn regime_ols(design: &Design, rows: &[usize]) -> Result<RegimeOlsParts, CointError> {
    let m = design.m;
    let k = design.k;
    let mut gram = vec![0.0_f64; m * m];
    let mut cross = vec![vec![0.0_f64; m]; k];
    for &t in rows {
        let xr = &design.x[t];
        let yr = &design.y[t];
        for a in 0..m {
            for b in 0..=a {
                gram[a * m + b] += xr[a] * xr[b];
            }
        }
        for j in 0..k {
            for a in 0..m {
                cross[j][a] += xr[a] * yr[j];
            }
        }
    }
    for a in 0..m {
        for b in (a + 1)..m {
            gram[a * m + b] = gram[b * m + a];
        }
    }
    let gchol = cholesky(&gram, m).ok_or(CointError::Singular {
        what: "a regime X'X in the final threshold-VECM refit (a collinear \
               regime segment — raise trim)",
    })?;
    let coefs: Vec<Vec<f64>> = cross.iter().map(|c| chol_solve(&gchol, m, c)).collect();

    // Residuals, their cross product, and the White meat per equation.
    let mut e = vec![0.0_f64; k * k];
    let mut meat = vec![vec![0.0_f64; m * m]; k];
    let mut u = vec![0.0_f64; k];
    for &t in rows {
        let xr = &design.x[t];
        let yr = &design.y[t];
        for j in 0..k {
            let mut fitv = 0.0;
            for a in 0..m {
                fitv += coefs[j][a] * xr[a];
            }
            u[j] = yr[j] - fitv;
        }
        for j in 0..k {
            for j2 in 0..k {
                e[j * k + j2] += u[j] * u[j2];
            }
            let u2 = u[j] * u[j];
            for a in 0..m {
                for b in 0..=a {
                    meat[j][a * m + b] += u2 * xr[a] * xr[b];
                }
            }
        }
    }
    let mut se = vec![vec![0.0_f64; m]; k];
    let mut scratch = vec![0.0_f64; m];
    for j in 0..k {
        for a in 0..m {
            for b in (a + 1)..m {
                meat[j][a * m + b] = meat[j][b * m + a];
            }
        }
        // diag of G^{-1} Meat G^{-1}: solve per column, then once more.
        let mut wmat = vec![0.0_f64; m * m];
        for col in 0..m {
            for a in 0..m {
                scratch[a] = meat[j][a * m + col];
            }
            let sol = chol_solve(&gchol, m, &scratch);
            for a in 0..m {
                wmat[a * m + col] = sol[a];
            }
        }
        for a in 0..m {
            let sol = chol_solve(&gchol, m, &wmat[a * m..(a + 1) * m]);
            let v = sol[a];
            se[j][a] = if v >= 0.0 { v.sqrt() } else { f64::NAN };
        }
    }
    Ok((coefs, se, e))
}

// ------------------------------------------------------------------ test

/// The Hansen-Seo (2002) sup-LM test of linear cointegration against
/// two-regime threshold cointegration, p-valued by their fixed-regressor
/// bootstrap (module docs).
///
/// `beta = None` fixes the cointegrating vector at the null (linear
/// rank-1 Johansen ML, unrestricted constant) estimate, as the paper
/// prescribes; `beta = Some(b)` uses the supplied vector (normalized
/// `b[0] = 1`) — their "fixed-beta" variant for a known cointegrating
/// relation. The `gamma` grid is at most `n_grid` evenly-spaced feasible
/// order statistics of `w_{t-1}` under `trim` (their `pi_0`).
///
/// # Errors
///
/// The input errors of [`threshold_vecm`], plus
/// [`CointError::InvalidArgument`] for `n_boot = 0` and
/// [`CointError::Singular`] if the Eicker-White covariance is singular at
/// some candidate on the observed data.
pub fn hansen_seo_test(
    endog: MatRef<'_, f64>,
    k_ar_diff: usize,
    trim: f64,
    n_grid: usize,
    n_boot: usize,
    seed: u64,
    beta: Option<&[f64]>,
) -> Result<HansenSeoTest, CointError> {
    validate_common(
        endog,
        k_ar_diff,
        trim,
        n_grid,
        "the gamma grid needs at least 2 points (n_grid >= 2; Hansen-Seo \
         used 300)",
    )?;
    if n_boot == 0 {
        return Err(CointError::InvalidArgument {
            what: "n_boot must be >= 1: the sup-LM null distribution is \
                   available only by the fixed-regressor bootstrap (499+ \
                   recommended; Hansen-Seo used 1000+)",
        });
    }
    let k = endog.ncols();
    let beta_used = match beta {
        Some(b) => normalize_beta(b, k)?,
        None => linear_anchor(endog, k_ar_diff)?.0,
    };

    let design = build_design(endog, k_ar_diff, &beta_used);
    let scan = Scan::build(&design, trim, n_grid)?;

    // Null (linear) OLS residuals U~ = Y - X (X'X)^{-1} X'Y.
    let m = design.m;
    let mut cross = vec![vec![0.0_f64; m]; k];
    for (xr, yr) in design.x.iter().zip(&design.y) {
        for j in 0..k {
            for a in 0..m {
                cross[j][a] += xr[a] * yr[j];
            }
        }
    }
    let coefs: Vec<Vec<f64>> = cross
        .iter()
        .map(|c| chol_solve(&scan.chol_total, m, c))
        .collect();
    let resid: Vec<Vec<f64>> = design
        .x
        .iter()
        .zip(&design.y)
        .map(|(xr, yr)| {
            (0..k)
                .map(|j| {
                    let mut fitv = 0.0;
                    for a in 0..m {
                        fitv += coefs[j][a] * xr[a];
                    }
                    yr[j] - fitv
                })
                .collect()
        })
        .collect();

    let (path, arg) =
        lm_path(&scan, &design, &resid).map_err(|what| CointError::Singular { what })?;
    let stat = path[arg];

    // Fixed-regressor bootstrap: y*_t = u~_t eta_t, eta iid N(0,1);
    // re-residualize on the same regressors; same sup over the same grid.
    let boot_stats: Vec<f64> = par_replicate(seed, n_boot, |_rep, stream| {
        let ystar: Vec<Vec<f64>> = resid
            .iter()
            .map(|ur| {
                let eta = WildWeights::Normal.draw(stream);
                ur.iter().map(|&u| u * eta).collect()
            })
            .collect();
        let mut bcross = vec![vec![0.0_f64; m]; k];
        for (xr, yr) in design.x.iter().zip(&ystar) {
            for j in 0..k {
                for a in 0..m {
                    bcross[j][a] += xr[a] * yr[j];
                }
            }
        }
        let bcoefs: Vec<Vec<f64>> = bcross
            .iter()
            .map(|c| chol_solve(&scan.chol_total, m, c))
            .collect();
        let ustar: Vec<Vec<f64>> = design
            .x
            .iter()
            .zip(&ystar)
            .map(|(xr, yr)| {
                (0..k)
                    .map(|j| {
                        let mut fitv = 0.0;
                        for a in 0..m {
                            fitv += bcoefs[j][a] * xr[a];
                        }
                        yr[j] - fitv
                    })
                    .collect()
            })
            .collect();
        match lm_path(&scan, &design, &ustar) {
            // A singular candidate in a replication counts as an extreme
            // draw (conservative direction for the p-value).
            Ok((bpath, barg)) => bpath[barg],
            Err(_) => f64::INFINITY,
        }
    })
    .map_err(|_| CointError::InvalidArgument {
        what: "n_boot exceeds the RNG substream spawn limit (< 2^32)",
    })?;

    let exceed = boot_stats.iter().filter(|&&s| s >= stat).count();
    let p_value = (1 + exceed) as f64 / (n_boot + 1) as f64;

    Ok(HansenSeoTest {
        stat,
        p_value,
        threshold: scan.cand_gamma[arg],
        beta: beta_used,
        n_boot,
        nobs: design.n,
        thresholds: scan.cand_gamma,
        lm_path: path,
        boot_stats,
        min_regime: scan.min_regime,
        neqs: k,
        n_regressors: m,
        k_ar_diff,
    })
}
