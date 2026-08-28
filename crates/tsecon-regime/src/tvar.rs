//! Two-regime threshold vector autoregression (TVAR) and its
//! linearity test with a fixed-regressor bootstrap p-value.
//!
//! This module lives in `tsecon-regime` — next to [`crate::setar`], not in
//! `tsecon-var` — because the TVAR *is* the multivariate SETAR: the same
//! observed-regime split on a lagged value of the series, the same
//! concentrated-least-squares threshold scan over trimmed order
//! statistics, and the same Hansen (1996) fixed-regressor bootstrap
//! contract for the linearity test (one SeedSequence-spawned Philox
//! substream per replication, bit-identical at any thread count).
//! `tsecon-var`'s analysis surface (IRFs, FEVDs, forecasts) assumes a
//! single linear regime and would silently mislead if pointed at a
//! per-regime fit — regime-dependent *generalized* impulse responses
//! (Koop-Pesaran-Potter 1996) are the honest tool there and are
//! deliberately **not** shipped with this estimator (deferred; see the
//! model card).
//!
//! The model (Tong 1983; Tsay 1998; Lo & Zivot 2001):
//!
//! ```text
//! y_t = A_1' X_t 1{z_t <= gamma} + A_2' X_t 1{z_t > gamma} + u_t,
//! X_t = (1?, y_{t-1}', ..., y_{t-p}')',
//! z_t = y_{threshold_index, t-d}    (the delay-d lag of one series).
//! ```
//!
//! * [`threshold_var`] estimates `(gamma, d, A_1, A_2)` by concentrated
//!   least squares / Gaussian MLE: for each candidate threshold (the
//!   unique order statistics of `z` with a `trim` fraction excluded at
//!   each end, and at least `m + 1` observations per regime) run OLS in
//!   each regime and minimize `ln det SigmaHat(gamma)` — the concentrated
//!   Gaussian criterion, the multivariate analogue of SETAR's pooled SSR.
//!   Supplying several `delays` searches the delay jointly (all
//!   candidates share the common usable sample so criteria are
//!   comparable, exactly as [`crate::setar`] does).
//! * [`threshold_var_test`] tests linearity (`A_1 = A_2`, a single-regime
//!   VAR) against the two-regime TVAR. The statistic is the sup over the
//!   trimmed grid of the coefficient-difference quadratic form with
//!   **Eicker-White** covariance evaluated at the null residuals — the
//!   heteroskedasticity-robust sup-Wald in its score (LM) form, the exact
//!   multivariate analogue of the Hansen-Seo (2002) sup-LM (see
//!   `tsecon-coint::tvecm` for the formulas; because the null residuals
//!   are orthogonal to the full-sample regressors, the Wald and LM
//!   numerators coincide, and the score form keeps every bootstrap
//!   quantity fixed-regressor). Under the null `gamma` is an unidentified
//!   nuisance parameter (the Davies problem), so no chi-squared p-value
//!   exists: the p-value comes from the Hansen (1996) fixed-regressor
//!   wild bootstrap — `y*_t = u~_t eta_t`, scalar `eta_t` iid N(0,1),
//!   re-residualized on the same regressors, same sup over the same grid.
//!   (R `tsDyn`'s `TVAR.LRtest` is a *different* convention — a sup-LR
//!   `T (ln det Sigma_0 - ln det Sigma_1)` with a residual bootstrap;
//!   the two tests answer the same question but their statistics are not
//!   comparable numbers.)
//!
//! **Validation grade (honest):** no third-party TVAR implementation runs
//! in this container (R and `tsDyn` are unavailable through the egress
//! proxy), so the golden fixture `fixtures/tvar.json` is an independent
//! NumPy transcription of the documented algorithm
//! (`fixtures/generate_tvar_fixtures.py` — never imports tsecon), pinned
//! at 1e-10. Statistical correctness — test size near nominal under a
//! linear VAR null, `(gamma, A_1, A_2)` recovery under a threshold DGP —
//! is carried by the seeded Monte Carlo property tests in
//! `tests/tvar_properties.rs`, whose measured numbers are quoted in the
//! model card.

use crate::error::RegimeError;
use crate::linsolve::{chol_solve, cholesky};
use tsecon_bootstrap::{par_replicate, WildWeights};

// ------------------------------------------------------------- the design

/// The usable-sample design for one delay: regressor rows, response rows,
/// and threshold variable, all in time order.
struct Design {
    /// Regressor rows `[1?, y_{t-1}', ..., y_{t-p}']` (n x m).
    x: Vec<Vec<f64>>,
    /// Response rows `y_t` (n x k).
    y: Vec<Vec<f64>>,
    /// Threshold variable `z_t = y_{threshold_index, t-d}` (n).
    z: Vec<f64>,
    n: usize,
    k: usize,
    m: usize,
}

/// Build the design over the common usable sample `t = start .. T-1`
/// (0-indexed; `start >= max(p, d)`).
fn build_design(
    endog: &[Vec<f64>],
    p: usize,
    threshold_index: usize,
    delay: usize,
    start: usize,
    constant: bool,
) -> Design {
    let t_total = endog.len();
    let k = endog[0].len();
    let n = t_total - start;
    let m = k * p + usize::from(constant);
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut z = Vec::with_capacity(n);
    for t in start..t_total {
        let mut row = Vec::with_capacity(m);
        if constant {
            row.push(1.0);
        }
        for lag in 1..=p {
            row.extend_from_slice(&endog[t - lag]);
        }
        x.push(row);
        y.push(endog[t].clone());
        z.push(endog[t - delay][threshold_index]);
    }
    Design { x, y, z, n, k, m }
}

// -------------------------------------------------------------- the scan

/// The precomputed threshold scan: sorted order, the trimmed candidate
/// grid (optionally subsampled to at most `n_grid` evenly-spaced points),
/// and per-candidate Cholesky factors of the two regime Gram matrices —
/// response-independent, so the bootstrap reuses them across replications.
struct Scan {
    order: Vec<usize>,
    cand_nlow: Vec<usize>,
    cand_gamma: Vec<f64>,
    chol_low: Vec<Vec<f64>>,
    chol_high: Vec<Vec<f64>>,
    chol_total: Vec<f64>,
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
    fn build(design: &Design, trim: f64, n_grid: Option<usize>) -> Result<Scan, RegimeError> {
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
        let chol_total = cholesky(&m_total, m).ok_or(RegimeError::Singular {
            what: "the full-sample VAR design X'X (a constant or collinear \
                   series, or p too large for the sample?)",
        })?;

        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| {
            design.z[a]
                .partial_cmp(&design.z[b])
                .unwrap_or(core::cmp::Ordering::Equal)
        });

        // Each regime must hold at least max(m + 1, ceil(trim * n))
        // observations: m + 1 so both regime regressions are estimable
        // with a residual degree of freedom, ceil(trim * n) is the
        // trimming keeping each regime a nondegenerate sample fraction.
        let min_regime = (m + 1).max((trim * n as f64).ceil() as usize);

        let mut feas_nlow = Vec::new();
        let mut feas_gamma = Vec::new();
        for i in 0..n {
            let row = order[i];
            let group_end = i + 1 == n || design.z[order[i + 1]] > design.z[row];
            let nlow = i + 1;
            if group_end && nlow >= min_regime && nlow <= n - min_regime {
                feas_nlow.push(nlow);
                feas_gamma.push(design.z[row]);
            }
        }
        if feas_gamma.is_empty() {
            return Err(RegimeError::InsufficientData {
                needed: 2 * min_regime,
                got: n,
            });
        }
        let (cand_nlow, cand_gamma): (Vec<usize>, Vec<f64>) = match n_grid {
            None => (feas_nlow, feas_gamma),
            Some(g) => {
                let keep = even_indices(feas_nlow.len(), g);
                (
                    keep.iter().map(|&i| feas_nlow[i]).collect(),
                    keep.iter().map(|&i| feas_gamma[i]).collect(),
                )
            }
        };

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
                let cl = cholesky(&lo, m).ok_or(RegimeError::Singular {
                    what: "a low-regime VAR design X'X at a threshold candidate \
                           (a collinear regime segment — raise trim)",
                })?;
                let ch = cholesky(&hi, m).ok_or(RegimeError::Singular {
                    what: "a high-regime VAR design X'X at a threshold candidate \
                           (a collinear regime segment — raise trim)",
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

    /// `ln det SigmaHat(gamma)` over the candidate grid (SigmaHat the
    /// pooled ML residual covariance of the two-regime OLS), plus the
    /// index of the first candidate attaining the minimum.
    fn logdet_profile(&self, design: &Design) -> Result<(Vec<f64>, usize), RegimeError> {
        let m = design.m;
        let k = design.k;
        let n = design.n as f64;

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
                for j in 0..k {
                    for j2 in (j + 1)..k {
                        let avg = 0.5 * (sig[j * k + j2] + sig[j2 * k + j]);
                        sig[j * k + j2] = avg;
                        sig[j2 * k + j] = avg;
                    }
                }
                let lchol = cholesky(&sig, k).ok_or(RegimeError::Singular {
                    what: "the pooled two-regime residual covariance at a \
                           threshold candidate (two response series are \
                           collinear, or a regime is degenerate)",
                })?;
                let mut ld = 0.0;
                for j in 0..k {
                    ld += lchol[j * k + j].ln();
                }
                let val = 2.0 * ld;
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

/// The per-candidate robust linearity statistic of one residual matrix
/// `u` (n x k rows): the coefficient-difference quadratic form with
/// Eicker-White covariance (module docs; the multivariate analogue of
/// Hansen-Seo 2002 eq. 10-12, transcribed from `tsecon-coint::tvecm`).
/// Returns the path aligned with `scan.cand_gamma` and the index of the
/// first maximum; `Err` names the factorization that failed.
fn lm_path(
    scan: &Scan,
    design: &Design,
    u: &[Vec<f64>],
) -> Result<(Vec<f64>, usize), &'static str> {
    let m = design.m;
    let k = design.k;
    let mk = m * k;

    let mut s_total = vec![vec![0.0_f64; m]; k];
    let mut omega_total = vec![0.0_f64; mk * mk];
    let mut xouter = vec![0.0_f64; m * m];
    for (xr, ur) in design.x.iter().zip(u) {
        accumulate(xr, ur, m, k, &mut xouter, &mut s_total, &mut omega_total);
    }

    let mut path = Vec::with_capacity(scan.cand_gamma.len());
    let mut best = 0usize;
    let mut best_val = f64::NEG_INFINITY;
    let mut s_low = vec![vec![0.0_f64; m]; k];
    let mut omega_low = vec![0.0_f64; mk * mk];
    let mut ci = 0usize;
    let mut scratch = vec![0.0_f64; m];
    for (i, &row) in scan.order.iter().enumerate() {
        accumulate(
            &design.x[row],
            &u[row],
            m,
            k,
            &mut xouter,
            &mut s_low,
            &mut omega_low,
        );
        while ci < scan.cand_nlow.len() && scan.cand_nlow[ci] == i + 1 {
            let cl = &scan.chol_low[ci];
            let ch = &scan.chol_high[ci];
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
            let mut v = vec![0.0_f64; mk * mk];
            let mut block = vec![0.0_f64; m * m];
            for j in 0..k {
                for j2 in 0..k {
                    for (lchol, high_regime) in [(cl, false), (ch, true)] {
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
                 the k*m coefficients tested — raise trim or lower p)",
            )?;
            let sol = chol_solve(&vchol, mk, &d);
            let stat: f64 = d.iter().zip(&sol).map(|(&a, &b)| a * b).sum();
            if stat > best_val {
                best_val = stat;
                best = ci;
            }
            path.push(stat);
            ci += 1;
        }
    }
    Ok((path, best))
}

/// Add one row's contribution to `S = X'U` and `Omega = sum (uu') (x)
/// (xx')` (row-major `mk x mk`, equations the outer index).
fn accumulate(
    xr: &[f64],
    ur: &[f64],
    m: usize,
    k: usize,
    xouter: &mut [f64],
    s: &mut [Vec<f64>],
    omega: &mut [f64],
) {
    let mk = m * k;
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
            s[j][i] += xr[i] * uj;
        }
        for (j2, &uj2) in ur.iter().enumerate().take(k) {
            let c = uj * uj2;
            let base_r = j * m;
            let base_c = j2 * m;
            for a in 0..m {
                let rowoff = (base_r + a) * mk + base_c;
                let xoff = a * m;
                for b in 0..m {
                    omega[rowoff + b] += c * xouter[xoff + b];
                }
            }
        }
    }
}

// ------------------------------------------------------------- validation

fn validate_common(
    endog: &[Vec<f64>],
    p: usize,
    threshold_index: usize,
    trim: f64,
) -> Result<usize, RegimeError> {
    if endog.is_empty() || endog[0].is_empty() {
        return Err(RegimeError::InsufficientData { needed: 1, got: 0 });
    }
    let k = endog[0].len();
    for row in endog {
        if row.len() != k {
            return Err(RegimeError::DimensionMismatch {
                what: "every row of the data matrix must have one entry per \
                       series",
                expected: k,
                actual: row.len(),
            });
        }
        for &v in row {
            if !v.is_finite() {
                return Err(RegimeError::NonFinite {
                    what: "the data matrix (the TVAR requires finite \
                           observations)",
                });
            }
        }
    }
    if k < 2 {
        return Err(RegimeError::InvalidSpec {
            what: "a threshold VAR needs at least two series (pass a 2-D array \
                   shaped (n_obs, n_series), observations in rows, oldest \
                   first; a single series is a threshold AR — use setar)",
        });
    }
    if p == 0 {
        return Err(RegimeError::InvalidSpec {
            what: "the TVAR requires p >= 1 (the model is a vector \
                   autoregression; with p = 0 there is no lag to regress on)",
        });
    }
    if threshold_index >= k {
        return Err(RegimeError::InvalidParameter {
            name: "threshold_index",
            value: threshold_index as f64,
            requirement: "threshold_index < n_series (it selects which series' \
                          lag drives the regime split)",
        });
    }
    if !(trim > 0.0 && trim < 0.5) || !trim.is_finite() {
        return Err(RegimeError::InvalidParameter {
            name: "trim",
            value: trim,
            requirement: "0 < trim < 0.5 (the minimum sample fraction each \
                          regime must keep)",
        });
    }
    let col = endog.iter().map(|r| r[threshold_index]);
    let first = endog[0][threshold_index];
    if col.clone().all(|v| v == first) {
        return Err(RegimeError::InvalidSpec {
            what: "the threshold series is constant: a threshold VAR needs \
                   variation in the threshold variable z_t",
        });
    }
    Ok(k)
}

fn validate_delay(delay: usize) -> Result<(), RegimeError> {
    if delay == 0 {
        return Err(RegimeError::InvalidParameter {
            name: "delay",
            value: 0.0,
            requirement: "delay >= 1 (the threshold variable is the lagged \
                          value y_{threshold_index, t-delay})",
        });
    }
    Ok(())
}

fn check_length(t: usize, start: usize, m: usize, trim: f64) -> Result<(), RegimeError> {
    let n = t.saturating_sub(start);
    let min_regime = (m + 1).max((trim * n as f64).ceil() as usize);
    if n < 2 * min_regime {
        return Err(RegimeError::InsufficientData {
            needed: start + 2 * min_regime,
            got: t,
        });
    }
    Ok(())
}

// --------------------------------------------------------------- results

/// Output of [`threshold_var`]: the concentrated-least-squares two-regime
/// TVAR fit.
///
/// Coefficient rows are equations (`k` of them); within a row the columns
/// are `[constant?, y_{t-1} (k entries), ..., y_{t-p} (k entries)]` (the
/// constant first when `constant = true`). The "low" regime is
/// `z_t <= threshold`.
#[derive(Debug, Clone, PartialEq)]
pub struct TvarFit {
    /// The estimated threshold `gamma` (an order statistic of `z`).
    pub threshold: f64,
    /// The delay `d` used (estimated when several candidates were
    /// supplied).
    pub delay: usize,
    /// The series whose lag drives the regime split.
    pub threshold_index: usize,
    /// Low-regime coefficients (`k x m`, rows = equations).
    pub coefs_low: Vec<Vec<f64>>,
    /// High-regime coefficients (`k x m`).
    pub coefs_high: Vec<Vec<f64>>,
    /// Classical nonrobust standard errors of `coefs_low` (`k x m`;
    /// per-regime per-equation `sqrt(s_jr^2 diag[(X_r'X_r)^{-1}])` with
    /// `s_jr^2 = SSR_jr / (n_r - m)`).
    pub se_low: Vec<Vec<f64>>,
    /// Classical nonrobust standard errors of `coefs_high` (`k x m`).
    pub se_high: Vec<Vec<f64>>,
    /// Low-regime sample size.
    pub n_low: usize,
    /// High-regime sample size.
    pub n_high: usize,
    /// Usable observations `n = T - max(p, max delay)`.
    pub nobs: usize,
    /// Pooled ML residual covariance `(E_1 + E_2) / n` (`k x k`).
    pub sigma: Vec<Vec<f64>>,
    /// Low-regime ML residual covariance `E_1 / n_1` (`k x k`).
    pub sigma_low: Vec<Vec<f64>>,
    /// High-regime ML residual covariance `E_2 / n_2` (`k x k`).
    pub sigma_high: Vec<Vec<f64>>,
    /// `ln det sigma` — the grid criterion at the optimum.
    pub log_det_sigma: f64,
    /// Gaussian log-likelihood `-(n k / 2)(ln 2 pi + 1) - (n/2) ln det
    /// sigma` (common-covariance concentrated form).
    pub llf: f64,
    /// Akaike criterion `n ln det sigma + 2 q`, `q = 2 k m + 1` (both
    /// coefficient blocks plus the threshold — the multivariate analogue
    /// of the SETAR convention; covariance parameters excluded).
    pub aic: f64,
    /// Schwarz criterion `n ln det sigma + q ln n`.
    pub bic: f64,
    /// The feasible candidate grid for the chosen delay (trimmed unique
    /// order statistics of `z`), ascending — the fit scans **all** of it.
    pub thresholds: Vec<f64>,
    /// `ln det SigmaHat(gamma)` per candidate, aligned with `thresholds`.
    pub logdet_path: Vec<f64>,
    /// Minimum per-regime size enforced: `max(m + 1, ceil(trim n))`.
    pub min_regime: usize,
    /// Number of series `k`.
    pub neqs: usize,
    /// Regressors per regime `m = k p + constant`.
    pub n_regressors: usize,
}

/// Output of [`threshold_var_test`]: the robust sup-Wald (score-form)
/// linearity test with a fixed-regressor wild-bootstrap p-value.
#[derive(Debug, Clone, PartialEq)]
pub struct TvarTest {
    /// The sup statistic (module docs).
    pub stat: f64,
    /// Fixed-regressor wild-bootstrap p-value
    /// `(1 + #{W* >= W}) / (n_boot + 1)`. **Not** a chi-squared tail —
    /// the threshold is unidentified under the null (Davies problem).
    pub p_value: f64,
    /// The threshold at which the sup is attained.
    pub threshold: f64,
    /// The delay tested.
    pub delay: usize,
    /// The series whose lag drives the regime split.
    pub threshold_index: usize,
    /// Number of bootstrap replications.
    pub n_boot: usize,
    /// Usable observations `n = T - max(p, delay)`.
    pub nobs: usize,
    /// The candidate threshold grid (ascending; at most `n_grid`
    /// evenly-spaced feasible order statistics).
    pub thresholds: Vec<f64>,
    /// The statistic per candidate, aligned with `thresholds`.
    pub wald_path: Vec<f64>,
    /// The bootstrap sup draws, in replication order (deterministic in
    /// `seed` at any thread count).
    pub boot_stats: Vec<f64>,
    /// Minimum per-regime size enforced.
    pub min_regime: usize,
    /// Number of series `k`.
    pub neqs: usize,
    /// Regressors per regime `m = k p + constant`.
    pub n_regressors: usize,
}

// ------------------------------------------------------------------- fit

/// Fit a two-regime threshold VAR(`p`) by concentrated least squares /
/// Gaussian MLE (module docs).
///
/// `endog` rows are observations (oldest first), each with one entry per
/// series. `threshold_index` selects the series whose `delay`-lag drives
/// the regime split; `delays` lists the candidate delays (one value fixes
/// the delay, several estimate it jointly with the threshold on the
/// common usable sample `t >= max(p, max delays)`). `trim` is the minimum
/// sample fraction per regime; each regime additionally keeps at least
/// `m + 1` observations (`m = k p + constant`). Ties in the grid are
/// grouped, and the first candidate attaining the minimal
/// `ln det SigmaHat` wins.
///
/// # Errors
///
/// * [`RegimeError::NonFinite`] for NaN/infinite observations;
/// * [`RegimeError::InvalidSpec`] for `k < 2`, `p = 0`, an empty `delays`
///   list, or a constant threshold series;
/// * [`RegimeError::InvalidParameter`] for `trim` outside `(0, 0.5)`, a
///   zero delay, or `threshold_index >= k`;
/// * [`RegimeError::InsufficientData`] when the usable sample cannot hold
///   two regimes of `max(m + 1, ceil(trim n))` observations;
/// * [`RegimeError::Singular`] for collinear designs or a degenerate
///   residual covariance.
pub fn threshold_var(
    endog: &[Vec<f64>],
    p: usize,
    threshold_index: usize,
    delays: &[usize],
    trim: f64,
    constant: bool,
) -> Result<TvarFit, RegimeError> {
    let k = validate_common(endog, p, threshold_index, trim)?;
    if delays.is_empty() {
        return Err(RegimeError::InvalidSpec {
            what: "delays must contain at least one candidate delay",
        });
    }
    for &d in delays {
        validate_delay(d)?;
    }
    let max_delay = delays.iter().copied().max().unwrap_or(1);
    let start = p.max(max_delay);
    let m = k * p + usize::from(constant);
    check_length(endog.len(), start, m, trim)?;

    // Concentrated scan over (delay, gamma): first strictly smaller
    // ln det wins, iterating delays in the given order.
    let mut best: Option<(usize, f64)> = None; // (delay, value)
    for &d in delays {
        let design = build_design(endog, p, threshold_index, d, start, constant);
        let scan = Scan::build(&design, trim, None)?;
        let (path, arg) = scan.logdet_profile(&design)?;
        let val = path[arg];
        let better = match best {
            None => true,
            Some((_, bv)) => val < bv,
        };
        if better {
            best = Some((d, val));
        }
    }
    let (delay, _) = best.ok_or(RegimeError::InvalidSpec {
        what: "delays must contain at least one candidate delay",
    })?;
    let design = build_design(endog, p, threshold_index, delay, start, constant);
    let scan = Scan::build(&design, trim, None)?;
    let (path, arg) = scan.logdet_profile(&design)?;
    let gamma = scan.cand_gamma[arg];

    // Final refit at (delay^, gamma^).
    let n = design.n;
    let low_rows: Vec<usize> = (0..n).filter(|&t| design.z[t] <= gamma).collect();
    let high_rows: Vec<usize> = (0..n).filter(|&t| design.z[t] > gamma).collect();
    let (coefs_low, se_low, e_low) = regime_ols(&design, &low_rows)?;
    let (coefs_high, se_high, e_high) = regime_ols(&design, &high_rows)?;
    let nf = n as f64;
    let n1 = low_rows.len() as f64;
    let n2 = high_rows.len() as f64;
    let to_mat = |flat: &[f64], scale: f64| -> Vec<Vec<f64>> {
        (0..k)
            .map(|j| (0..k).map(|j2| flat[j * k + j2] / scale).collect())
            .collect()
    };
    let pooled: Vec<f64> = e_low.iter().zip(&e_high).map(|(&a, &b)| a + b).collect();
    let sigma = to_mat(&pooled, nf);
    let sigma_low = to_mat(&e_low, n1);
    let sigma_high = to_mat(&e_high, n2);
    let mut sig_flat = vec![0.0_f64; k * k];
    for j in 0..k {
        for j2 in 0..k {
            sig_flat[j * k + j2] = 0.5 * (sigma[j][j2] + sigma[j2][j]);
        }
    }
    let schol = cholesky(&sig_flat, k).ok_or(RegimeError::Singular {
        what: "the pooled TVAR residual covariance at the optimum",
    })?;
    let mut log_det_sigma = 0.0;
    for j in 0..k {
        log_det_sigma += schol[j * k + j].ln();
    }
    log_det_sigma *= 2.0;
    let kf = k as f64;
    let llf = -nf * kf / 2.0 * (core::f64::consts::TAU.ln() + 1.0) - nf / 2.0 * log_det_sigma;
    let q = (2 * k * m + 1) as f64;
    let aic = nf * log_det_sigma + 2.0 * q;
    let bic = nf * log_det_sigma + q * nf.ln();

    Ok(TvarFit {
        threshold: gamma,
        delay,
        threshold_index,
        coefs_low,
        coefs_high,
        se_low,
        se_high,
        n_low: low_rows.len(),
        n_high: high_rows.len(),
        nobs: n,
        sigma,
        sigma_low,
        sigma_high,
        log_det_sigma,
        llf,
        aic,
        bic,
        thresholds: scan.cand_gamma,
        logdet_path: path,
        min_regime: scan.min_regime,
        neqs: k,
        n_regressors: m,
    })
}

/// The pieces one regime's OLS hands back: coefficients (`k x m`, rows =
/// equations), standard errors (`k x m`), and the residual cross product
/// `E = U'U` (row-major `k x k`).
type RegimeOlsParts = (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<f64>);

/// OLS of the design's responses on its regressors over `rows`:
/// coefficients (`k x m`, rows = equations), classical nonrobust standard
/// errors (`k x m`), and the residual cross product `E = U'U` (row-major
/// `k x k`).
fn regime_ols(design: &Design, rows: &[usize]) -> Result<RegimeOlsParts, RegimeError> {
    let m = design.m;
    let k = design.k;
    if rows.len() < m + 1 {
        return Err(RegimeError::InsufficientData {
            needed: m + 1,
            got: rows.len(),
        });
    }
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
    let gchol = cholesky(&gram, m).ok_or(RegimeError::Singular {
        what: "a regime X'X in the final TVAR refit (a collinear regime \
               segment — raise trim)",
    })?;
    let coefs: Vec<Vec<f64>> = cross.iter().map(|c| chol_solve(&gchol, m, c)).collect();

    // diag[(X'X)^{-1}] via the Cholesky factor: solve for each unit vector.
    let mut xtx_inv_diag = vec![0.0_f64; m];
    let mut unit = vec![0.0_f64; m];
    for a in 0..m {
        unit[a] = 1.0;
        let sol = chol_solve(&gchol, m, &unit);
        xtx_inv_diag[a] = sol[a];
        unit[a] = 0.0;
    }

    let mut e = vec![0.0_f64; k * k];
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
        }
    }
    let dof = (rows.len() - m) as f64;
    let se: Vec<Vec<f64>> = (0..k)
        .map(|j| {
            let s2 = e[j * k + j] / dof;
            xtx_inv_diag.iter().map(|&d| (s2 * d).sqrt()).collect()
        })
        .collect();
    Ok((coefs, se, e))
}

// ------------------------------------------------------------------ test

/// Robust sup-Wald (score-form) test of a linear VAR(`p`) against the
/// two-regime TVAR, p-valued by the Hansen (1996) fixed-regressor wild
/// bootstrap (module docs).
///
/// The candidate grid is at most `n_grid` evenly-spaced feasible order
/// statistics of `z` under `trim` (Hansen-Seo used 300).
///
/// # Errors
///
/// The input errors of [`threshold_var`], plus
/// [`RegimeError::InvalidParameter`] for `n_boot = 0` or `n_grid < 2`,
/// and [`RegimeError::Singular`] if the Eicker-White covariance is
/// singular at some candidate on the observed data.
#[allow(clippy::too_many_arguments)]
pub fn threshold_var_test(
    endog: &[Vec<f64>],
    p: usize,
    threshold_index: usize,
    delay: usize,
    trim: f64,
    constant: bool,
    n_grid: usize,
    n_boot: usize,
    seed: u64,
) -> Result<TvarTest, RegimeError> {
    let k = validate_common(endog, p, threshold_index, trim)?;
    validate_delay(delay)?;
    if n_boot == 0 {
        return Err(RegimeError::InvalidParameter {
            name: "n_boot",
            value: 0.0,
            requirement: "n_boot >= 1 (the null distribution is available \
                          only by bootstrap; 499+ recommended)",
        });
    }
    if n_grid < 2 {
        return Err(RegimeError::InvalidParameter {
            name: "n_grid",
            value: n_grid as f64,
            requirement: "n_grid >= 2 (the number of threshold candidates the \
                          sup is taken over; Hansen-Seo used 300)",
        });
    }
    let start = p.max(delay);
    let m = k * p + usize::from(constant);
    check_length(endog.len(), start, m, trim)?;

    let design = build_design(endog, p, threshold_index, delay, start, constant);
    let scan = Scan::build(&design, trim, Some(n_grid))?;

    // Null (linear VAR) OLS residuals.
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
        lm_path(&scan, &design, &resid).map_err(|what| RegimeError::Singular { what })?;
    let stat = path[arg];

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
    .map_err(|_| RegimeError::InvalidParameter {
        name: "n_boot",
        value: n_boot as f64,
        requirement: "n_boot within the RNG substream spawn limit (< 2^32)",
    })?;

    let exceed = boot_stats.iter().filter(|&&s| s >= stat).count();
    let p_value = (1 + exceed) as f64 / (n_boot + 1) as f64;

    Ok(TvarTest {
        stat,
        p_value,
        threshold: scan.cand_gamma[arg],
        delay,
        threshold_index,
        n_boot,
        nobs: design.n,
        thresholds: scan.cand_gamma,
        wald_path: path,
        boot_stats,
        min_regime: scan.min_regime,
        neqs: k,
        n_regressors: m,
    })
}
