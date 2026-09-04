//! Self-exciting threshold autoregression (SETAR) and the Hansen (1996)
//! bootstrap linearity test.
//!
//! The two-regime SETAR(`p`) model of Tong & Lim (1980), in the notation of
//! Hansen (1997):
//!
//! ```text
//! y_t = x_t' beta_1 * 1{ y_{t-d} <= gamma } +
//!       x_t' beta_2 * 1{ y_{t-d} >  gamma } + e_t,
//! x_t = (1, y_{t-1}, ..., y_{t-p})',
//! ```
//!
//! with threshold `gamma`, delay `d`, and iid errors. Because the model is
//! linear in `(beta_1, beta_2)` for fixed `(gamma, d)`, estimation is by
//! **concentrated least squares** (Tong & Lim 1980; Hansen 1997, 2000): grid
//! over threshold candidates — the order statistics of the threshold
//! variable `y_{t-d}` with a `trim` fraction excluded at each end — run OLS
//! in each regime per candidate, and pick the `(gamma, d)` minimizing the
//! pooled SSR. [`setar`] implements this with one incremental
//! normal-equation sweep over the sorted threshold variable (each candidate
//! adds one observation to the low regime), then refits the chosen split by
//! Householder QR for the reported coefficients and classical standard
//! errors.
//!
//! [`setar_test`] is the Hansen (1996) test of linearity against the
//! threshold alternative: the statistic is
//! `sup-F = n (S0 - S1) / S1` where `S0` is the linear-AR SSR and `S1` the
//! SETAR SSR at the concentrated optimum (Hansen 1997, eq. F12). Its null
//! distribution is **nonstandard** — under linearity the threshold `gamma`
//! is an unidentified nuisance parameter (the Davies problem) — so the
//! p-value comes from Hansen's **fixed-regressor wild bootstrap**: regress
//! `y*_t = ehat_t eta_t` (`eta_t` iid standard normal, `ehat` the null
//! residuals) on the *same* fixed regressors and take the same sup over the
//! same candidate grid, repeated `n_boot` times. Never compare `sup-F` to a
//! chi-squared table.
//!
//! Every bootstrap draw is reproducible through the library RNG
//! ([`tsecon_bootstrap::par_replicate`]: one SeedSequence-spawned Philox
//! substream per replication), so the p-value is bit-identical at any rayon
//! thread count.
//!
//! References: Tong & Lim (1980), JRSS-B 42(3); Hansen (1996),
//! Econometrica 64(2); Hansen (1997), Studies in Nonlinear Dynamics &
//! Econometrics 2(1); Hansen (2000), Econometrica 68(3).

use crate::error::RegimeError;
use crate::linsolve::{chol_solve, cholesky};
use tsecon_bootstrap::{par_replicate, WildWeights};

// ------------------------------------------------------------ small linalg
//
// The row-major Cholesky pair the scan factors its Gram matrices with
// lives in `crate::linsolve` (shared with the threshold VAR).

/// Plain OLS `y = X b + e` by Householder QR (columns in `cols`), returning
/// coefficients, classical nonrobust standard errors
/// `sqrt(s^2 diag[(X'X)^{-1}])` with `s^2 = SSR / (n - k)`, the residual
/// vector, and the SSR. Mirrors the QR helper of `tsecon-diag::phillips`
/// (error growth proportional to `cond(X)`, not `cond(X)^2`).
pub(crate) struct Ols {
    pub(crate) params: Vec<f64>,
    pub(crate) bse: Vec<f64>,
    pub(crate) resid: Vec<f64>,
    pub(crate) ssr: f64,
}

pub(crate) fn ols_qr(cols: &[Vec<f64>], y: &[f64], what: &'static str) -> Result<Ols, RegimeError> {
    let n = y.len();
    let k = cols.len();
    debug_assert!(cols.iter().all(|c| c.len() == n));
    if k == 0 || n < k + 1 {
        return Err(RegimeError::InsufficientData {
            needed: k + 1,
            got: n,
        });
    }

    let mut a: Vec<Vec<f64>> = cols.to_vec();
    let mut qty: Vec<f64> = y.to_vec();
    let mut rdiag = vec![0.0_f64; k];

    for j in 0..k {
        let sub: f64 = a[j][j..].iter().map(|&v| v * v).sum();
        let head: f64 = a[j][..j].iter().map(|&v| v * v).sum();
        let norm = sub.sqrt();
        let tol = ((head + sub).sqrt() * 1e-13).max(f64::MIN_POSITIVE);
        if norm.is_nan() || norm <= tol {
            return Err(RegimeError::Singular { what });
        }
        let alpha = if a[j][j] >= 0.0 { -norm } else { norm };
        a[j][j] -= alpha;
        rdiag[j] = alpha;
        let vtv: f64 = a[j][j..].iter().map(|&v| v * v).sum();

        let (left, right) = a.split_at_mut(j + 1);
        let v = &left[j][j..];
        for col in right.iter_mut() {
            let dot: f64 = v.iter().zip(&col[j..]).map(|(&vi, &ci)| vi * ci).sum();
            let f = 2.0 * dot / vtv;
            for (vi, ci) in v.iter().zip(col[j..].iter_mut()) {
                *ci -= f * vi;
            }
        }
        let dot: f64 = v.iter().zip(&qty[j..]).map(|(&vi, &qi)| vi * qi).sum();
        let f = 2.0 * dot / vtv;
        for (vi, qi) in v.iter().zip(qty[j..].iter_mut()) {
            *qi -= f * vi;
        }
    }

    let mut beta = vec![0.0_f64; k];
    for j in (0..k).rev() {
        let mut acc = qty[j];
        for (m, bm) in beta.iter().enumerate().skip(j + 1) {
            acc -= a[m][j] * bm;
        }
        beta[j] = acc / rdiag[j];
    }

    let mut resid = vec![0.0_f64; n];
    let mut ssr = 0.0;
    for (i, ri) in resid.iter_mut().enumerate() {
        let mut fit = 0.0;
        for (bj, col) in beta.iter().zip(cols.iter()) {
            fit += bj * col[i];
        }
        let e = y[i] - fit;
        *ri = e;
        ssr += e * e;
    }

    let sigma2 = ssr / (n - k) as f64;
    if !(sigma2 >= 0.0 && sigma2.is_finite()) {
        return Err(RegimeError::NonFinite { what });
    }

    // diag[(X'X)^{-1}] = squared row norms of R^{-1}.
    let mut xtx_inv_diag = vec![0.0_f64; k];
    let mut x = vec![0.0_f64; k];
    for c in 0..k {
        x[c] = 1.0 / rdiag[c];
        for j in (0..c).rev() {
            let mut acc = 0.0;
            for (l, xl) in x.iter().enumerate().take(c + 1).skip(j + 1) {
                acc += a[l][j] * xl;
            }
            x[j] = -acc / rdiag[j];
        }
        for (dj, &xj) in xtx_inv_diag.iter_mut().zip(x.iter()).take(c + 1) {
            *dj += xj * xj;
        }
    }
    let bse = xtx_inv_diag.iter().map(|&d| (sigma2 * d).sqrt()).collect();

    Ok(Ols {
        params: beta,
        bse,
        resid,
        ssr,
    })
}

// ------------------------------------------------------------- the design

/// The usable-sample design for one delay: response, regressor columns, and
/// threshold variable, all in time order. Shared with the STAR module.
pub(crate) struct Design {
    /// Regressor columns (`k` columns of length `n`): `[1?, y_{t-1}, ...,
    /// y_{t-p}]`.
    pub(crate) cols: Vec<Vec<f64>>,
    /// Response `y_t`.
    pub(crate) y: Vec<f64>,
    /// Threshold variable `z_t = y_{t-d}`.
    pub(crate) z: Vec<f64>,
    pub(crate) n: usize,
    pub(crate) k: usize,
}

/// Build the design over the common usable sample `t = start .. T-1`
/// (0-indexed; `start >= max(p, d)`).
pub(crate) fn build_design(
    y: &[f64],
    p: usize,
    delay: usize,
    start: usize,
    constant: bool,
) -> Design {
    let t_total = y.len();
    // Every caller must have already refused start >= t_total (their
    // insufficient-data checks); this guards the usize subtraction against
    // a future caller reintroducing the wrap (audit round 10, finding 1).
    debug_assert!(
        start < t_total,
        "build_design: start ({start}) must be below the series length ({t_total}); \
         callers must run their sufficiency check first"
    );
    let n = t_total.saturating_sub(start);
    let k = p + usize::from(constant);
    let mut cols: Vec<Vec<f64>> = Vec::with_capacity(k);
    if constant {
        cols.push(vec![1.0; n]);
    }
    for lag in 1..=p {
        cols.push((start..t_total).map(|t| y[t - lag]).collect());
    }
    let resp: Vec<f64> = y.get(start..).unwrap_or(&[]).to_vec();
    let z: Vec<f64> = (start..t_total).map(|t| y[t - delay]).collect();
    Design {
        cols,
        y: resp,
        z,
        n,
        k,
    }
}

// -------------------------------------------------------------- the scan

/// The precomputed threshold scan for one design: sorted order, feasible
/// candidate grid, and per-candidate Cholesky factors of the two regime
/// Gram matrices (which depend only on `X` and the split — not on the
/// response — so the bootstrap reuses them across replications).
struct Scan {
    n: usize,
    k: usize,
    /// Row indices sorted by ascending threshold-variable value.
    order: Vec<usize>,
    /// Low-regime size `#{z <= gamma}` per candidate (ascending).
    cand_nlow: Vec<usize>,
    /// Candidate threshold values (the trimmed unique order statistics).
    cand_gamma: Vec<f64>,
    /// Cholesky factor of the low-regime `X'X` per candidate.
    chol_low: Vec<Vec<f64>>,
    /// Cholesky factor of the high-regime `X'X` per candidate.
    chol_high: Vec<Vec<f64>>,
    /// Cholesky factor of the full-sample `X'X` (the linear null fit).
    chol_total: Vec<f64>,
    /// Minimum observations per regime actually enforced.
    min_regime: usize,
}

/// The SSR profile of one response vector over a scan's candidate grid.
struct Profile {
    /// SSR of the linear (single-regime) fit.
    ssr_linear: f64,
    /// Pooled two-regime SSR per candidate, aligned with `cand_gamma`.
    ssr_path: Vec<f64>,
    /// Index of the first candidate attaining the minimum pooled SSR.
    best: usize,
}

impl Scan {
    fn build(design: &Design, trim: f64) -> Result<Scan, RegimeError> {
        let n = design.n;
        let k = design.k;

        // Full-sample Gram matrix and its factor (the linear null design).
        let mut xtx_total = vec![0.0_f64; k * k];
        for t in 0..n {
            for a in 0..k {
                let xa = design.cols[a][t];
                for b in 0..=a {
                    xtx_total[a * k + b] += xa * design.cols[b][t];
                }
            }
        }
        for a in 0..k {
            for b in (a + 1)..k {
                xtx_total[a * k + b] = xtx_total[b * k + a];
            }
        }
        let chol_total = cholesky(&xtx_total, k).ok_or(RegimeError::Singular {
            what: "the linear AR design X'X (is the series constant, or p too \
                   large for the sample?)",
        })?;

        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| {
            design.z[a]
                .partial_cmp(&design.z[b])
                .unwrap_or(core::cmp::Ordering::Equal)
        });

        // Each regime must hold at least max(k + 1, ceil(trim * n))
        // observations: k + 1 so both regime regressions are estimable with
        // a residual degree of freedom, ceil(trim * n) is the Hansen (1997)
        // trimming that keeps the threshold away from the sample edges.
        let min_regime = (k + 1).max((trim * n as f64).ceil() as usize);

        let mut cand_nlow = Vec::new();
        let mut cand_gamma = Vec::new();
        let mut chol_low = Vec::new();
        let mut chol_high = Vec::new();
        let mut xtx_low = vec![0.0_f64; k * k];
        for i in 0..n {
            let row = order[i];
            for a in 0..k {
                let xa = design.cols[a][row];
                for b in 0..=a {
                    xtx_low[a * k + b] += xa * design.cols[b][row];
                }
            }
            let nlow = i + 1;
            // A candidate sits at the end of each tie group (all obs with
            // z <= gamma must fall in the low regime together).
            let group_end = i + 1 == n || design.z[order[i + 1]] > design.z[row];
            if group_end && nlow >= min_regime && nlow <= n - min_regime {
                let mut lo = xtx_low.clone();
                let mut hi = vec![0.0_f64; k * k];
                for a in 0..k {
                    for b in (a + 1)..k {
                        lo[a * k + b] = lo[b * k + a];
                    }
                }
                for (h, (t, l)) in hi.iter_mut().zip(xtx_total.iter().zip(lo.iter())) {
                    *h = t - l;
                }
                let cl = cholesky(&lo, k).ok_or(RegimeError::Singular {
                    what: "a low-regime design X'X at a threshold candidate \
                           (a near-constant segment of the series makes the \
                           regime regression collinear)",
                })?;
                let ch = cholesky(&hi, k).ok_or(RegimeError::Singular {
                    what: "a high-regime design X'X at a threshold candidate \
                           (a near-constant segment of the series makes the \
                           regime regression collinear)",
                })?;
                cand_nlow.push(nlow);
                cand_gamma.push(design.z[row]);
                chol_low.push(cl);
                chol_high.push(ch);
            }
        }

        if cand_gamma.is_empty() {
            return Err(RegimeError::InsufficientData {
                needed: 2 * min_regime,
                got: n,
            });
        }

        Ok(Scan {
            n,
            k,
            order,
            cand_nlow,
            cand_gamma,
            chol_low,
            chol_high,
            chol_total,
            min_regime,
        })
    }

    /// SSR profile of the response `yb` (length `n`, time order) over the
    /// candidate grid, plus the linear-fit SSR — one `O(n k + G k^2)` pass
    /// using the prefactored per-candidate Gram matrices.
    fn profile(&self, design: &Design, yb: &[f64]) -> Profile {
        let k = self.k;
        let mut xty_total = vec![0.0_f64; k];
        let mut yty_total = 0.0_f64;
        for (t, &yt) in yb.iter().enumerate().take(self.n) {
            yty_total += yt * yt;
            for (a, acc) in xty_total.iter_mut().enumerate() {
                *acc += design.cols[a][t] * yt;
            }
        }
        let b0 = chol_solve(&self.chol_total, k, &xty_total);
        let fitted0: f64 = b0.iter().zip(&xty_total).map(|(&b, &c)| b * c).sum();
        let ssr_linear = (yty_total - fitted0).max(0.0);

        let mut ssr_path = Vec::with_capacity(self.cand_gamma.len());
        let mut best = 0usize;
        let mut best_ssr = f64::INFINITY;
        let mut xty_low = vec![0.0_f64; k];
        let mut yty_low = 0.0_f64;
        let mut ci = 0usize;
        for (i, &row) in self.order.iter().enumerate() {
            let yt = yb[row];
            yty_low += yt * yt;
            for (a, acc) in xty_low.iter_mut().enumerate() {
                *acc += design.cols[a][row] * yt;
            }
            while ci < self.cand_nlow.len() && self.cand_nlow[ci] == i + 1 {
                let bl = chol_solve(&self.chol_low[ci], k, &xty_low);
                let fit_l: f64 = bl.iter().zip(&xty_low).map(|(&b, &c)| b * c).sum();
                let xty_high: Vec<f64> = xty_total
                    .iter()
                    .zip(&xty_low)
                    .map(|(&t, &l)| t - l)
                    .collect();
                let bh = chol_solve(&self.chol_high[ci], k, &xty_high);
                let fit_h: f64 = bh.iter().zip(&xty_high).map(|(&b, &c)| b * c).sum();
                let ssr = ((yty_low - fit_l).max(0.0) + ((yty_total - yty_low) - fit_h).max(0.0))
                    .max(0.0);
                if ssr < best_ssr {
                    best_ssr = ssr;
                    best = ci;
                }
                ssr_path.push(ssr);
                ci += 1;
            }
        }
        Profile {
            ssr_linear,
            ssr_path,
            best,
        }
    }

    /// The Hansen (1997) sup-F of one response: `n (S0 - S1) / S1` at the
    /// SSR-minimizing candidate (`+inf` for a degenerate perfect threshold
    /// fit — it counts as an extreme draw in the bootstrap tail).
    fn supf(&self, design: &Design, yb: &[f64]) -> f64 {
        let prof = self.profile(design, yb);
        let s1 = prof.ssr_path[prof.best];
        if s1 <= 0.0 {
            return f64::INFINITY;
        }
        self.n as f64 * (prof.ssr_linear - s1).max(0.0) / s1
    }
}

// ------------------------------------------------------------- validation

fn validate_common(y: &[f64], p: usize, trim: f64) -> Result<(), RegimeError> {
    for &v in y {
        if !v.is_finite() {
            return Err(RegimeError::NonFinite {
                what: "the input series y (SETAR requires finite observations)",
            });
        }
    }
    if p == 0 {
        return Err(RegimeError::InvalidSpec {
            what: "SETAR requires p >= 1 (the model is an autoregression; \
                   with p = 0 there is no lag to regress on)",
        });
    }
    if !(trim > 0.0 && trim < 0.5) || !trim.is_finite() {
        return Err(RegimeError::InvalidParameter {
            name: "trim",
            value: trim,
            requirement: "0 < trim < 0.5 (the fraction of threshold-variable \
                          order statistics excluded at each end)",
        });
    }
    if !y.is_empty() && y.iter().all(|&v| v == y[0]) {
        return Err(RegimeError::InvalidSpec {
            what: "the series is constant: a threshold autoregression needs \
                   variation in the threshold variable y_{t-d}",
        });
    }
    Ok(())
}

fn validate_delay(delay: usize) -> Result<(), RegimeError> {
    if delay == 0 {
        return Err(RegimeError::InvalidParameter {
            name: "delay",
            value: 0.0,
            requirement: "delay >= 1 (the threshold variable is the lagged \
                          value y_{t-delay})",
        });
    }
    Ok(())
}

fn check_length(t: usize, start: usize, k: usize, trim: f64) -> Result<(), RegimeError> {
    // Need a usable sample large enough for two regimes of at least
    // max(k + 1, ceil(trim * n)) observations each.
    let n = t.saturating_sub(start);
    let min_regime = (k + 1).max((trim * n as f64).ceil() as usize);
    if n < 2 * min_regime {
        return Err(RegimeError::InsufficientData {
            needed: start + 2 * min_regime,
            got: t,
        });
    }
    Ok(())
}

// --------------------------------------------------------------- results

/// Output of [`setar`]: the concentrated-least-squares two-regime SETAR fit.
///
/// Coefficient vectors are ordered `[constant?, lag 1, ..., lag p]` (the
/// constant first when `constant = true`). The "low" regime is
/// `y_{t-d} <= threshold`, the "high" regime `y_{t-d} > threshold`.
#[derive(Debug, Clone, PartialEq)]
pub struct SetarFit {
    /// The estimated threshold `gamma` (an order statistic of `y_{t-d}`).
    pub threshold: f64,
    /// The delay `d` used (estimated when several candidates were supplied).
    pub delay: usize,
    /// Low-regime coefficients.
    pub coefs_low: Vec<f64>,
    /// High-regime coefficients.
    pub coefs_high: Vec<f64>,
    /// Classical nonrobust standard errors of `coefs_low`
    /// (`sqrt(s_1^2 diag[(X_1'X_1)^{-1}])`, `s_1^2 = SSR_1 / (n_1 - k)`).
    pub se_low: Vec<f64>,
    /// Classical nonrobust standard errors of `coefs_high`.
    pub se_high: Vec<f64>,
    /// Low-regime sample size `n_1`.
    pub n_low: usize,
    /// High-regime sample size `n_2`.
    pub n_high: usize,
    /// Usable observations `n = T - max(p, max delay)`.
    pub nobs: usize,
    /// Pooled residual sum of squares `SSR_1 + SSR_2` at the optimum.
    pub ssr: f64,
    /// Pooled error variance `SSR / (n - 2k)`.
    pub sigma2: f64,
    /// Low-regime error variance `SSR_1 / (n_1 - k)`.
    pub sigma2_low: f64,
    /// High-regime error variance `SSR_2 / (n_2 - k)`.
    pub sigma2_high: f64,
    /// Akaike criterion `n ln(SSR/n) + 2 m`, `m = 2k + 1` (both coefficient
    /// blocks plus the threshold).
    pub aic: f64,
    /// Schwarz criterion `n ln(SSR/n) + m ln(n)`.
    pub bic: f64,
    /// The feasible candidate grid for the chosen delay (trimmed unique
    /// order statistics of `y_{t-d}`), ascending.
    pub thresholds: Vec<f64>,
    /// Pooled SSR per candidate, aligned with `thresholds`.
    pub ssr_path: Vec<f64>,
    /// Minimum per-regime size enforced: `max(k + 1, ceil(trim n))`.
    pub min_regime: usize,
    /// Number of regressors per regime `k = p + constant`.
    pub k: usize,
}

/// Output of [`setar_test`]: the Hansen (1996) sup-F linearity test with a
/// fixed-regressor wild-bootstrap p-value.
#[derive(Debug, Clone, PartialEq)]
pub struct SetarTest {
    /// The sup-F statistic `n (S0 - S1) / S1`.
    pub stat: f64,
    /// Fixed-regressor wild-bootstrap p-value
    /// `(1 + #{F* >= F}) / (n_boot + 1)`. **Not** a chi-squared tail — the
    /// threshold is unidentified under the null (Davies problem).
    pub p_value: f64,
    /// The threshold at which the sup is attained.
    pub threshold: f64,
    /// The delay tested.
    pub delay: usize,
    /// Number of bootstrap replications.
    pub n_boot: usize,
    /// Usable observations `n = T - max(p, delay)`.
    pub nobs: usize,
    /// SSR of the null linear AR fit (`S0`).
    pub ssr_linear: f64,
    /// Pooled SETAR SSR at the concentrated optimum (`S1`).
    pub ssr_setar: f64,
    /// The candidate threshold grid, ascending.
    pub thresholds: Vec<f64>,
    /// `F(gamma) = n (S0 - S(gamma)) / S(gamma)` per candidate.
    pub f_path: Vec<f64>,
    /// The bootstrap sup-F draws, in replication order (deterministic in
    /// `seed` at any thread count).
    pub boot_stats: Vec<f64>,
}

// ------------------------------------------------------------------- fit

/// Fit a two-regime SETAR(`p`) by concentrated least squares (Tong-Lim
/// 1980; Hansen 1997).
///
/// `delays` lists the candidate delays `d` for the threshold variable
/// `y_{t-d}`; supply one value to fix the delay, several to estimate it
/// jointly with the threshold (all fits then share the common usable sample
/// `t = max(p, max delays) .. T-1` so pooled SSRs are comparable). `trim`
/// is the fraction of the threshold variable's order statistics excluded at
/// each end of the candidate grid; each regime must additionally hold at
/// least `k + 1` observations (`k = p + constant`). Ties in the grid are
/// grouped (every observation with `y_{t-d} <= gamma` is in the low
/// regime), and the first candidate attaining the minimal pooled SSR wins.
///
/// # Errors
///
/// * [`RegimeError::NonFinite`] for NaN/infinite observations.
/// * [`RegimeError::InvalidSpec`] for `p = 0`, an empty `delays` list, or a
///   constant series.
/// * [`RegimeError::InvalidParameter`] for `trim` outside `(0, 0.5)` or a
///   zero delay.
/// * [`RegimeError::InsufficientData`] when the usable sample cannot hold
///   two regimes of `max(k + 1, ceil(trim n))` observations.
/// * [`RegimeError::Singular`] for collinear regime designs (near-constant
///   series segments).
pub fn setar(
    y: &[f64],
    p: usize,
    delays: &[usize],
    trim: f64,
    constant: bool,
) -> Result<SetarFit, RegimeError> {
    validate_common(y, p, trim)?;
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
    let k = p + usize::from(constant);
    check_length(y.len(), start, k, trim)?;

    // Concentrated LS over (delay, gamma): first strictly smaller pooled
    // SSR wins, iterating delays in the given order.
    let mut best: Option<(usize, f64, Scan, Profile, Design)> = None;
    for &d in delays {
        let design = build_design(y, p, d, start, constant);
        let scan = Scan::build(&design, trim)?;
        let prof = scan.profile(&design, &design.y);
        let ssr = prof.ssr_path[prof.best];
        let better = match &best {
            None => true,
            Some((_, best_ssr, _, _, _)) => ssr < *best_ssr,
        };
        if better {
            best = Some((d, ssr, scan, prof, design));
        }
    }
    let (delay, _, scan, prof, design) = match best {
        Some(b) => b,
        None => {
            return Err(RegimeError::InvalidSpec {
                what: "delays must contain at least one candidate delay",
            })
        }
    };
    let threshold = scan.cand_gamma[prof.best];

    // Final refit of the chosen split by QR for coefficients and SEs.
    let n = design.n;
    let low_rows: Vec<usize> = (0..n).filter(|&t| design.z[t] <= threshold).collect();
    let high_rows: Vec<usize> = (0..n).filter(|&t| design.z[t] > threshold).collect();
    let take = |rows: &[usize]| -> (Vec<Vec<f64>>, Vec<f64>) {
        let cols: Vec<Vec<f64>> = design
            .cols
            .iter()
            .map(|c| rows.iter().map(|&t| c[t]).collect())
            .collect();
        let yy: Vec<f64> = rows.iter().map(|&t| design.y[t]).collect();
        (cols, yy)
    };
    let (cols_lo, y_lo) = take(&low_rows);
    let (cols_hi, y_hi) = take(&high_rows);
    let fit_lo = ols_qr(&cols_lo, &y_lo, "the low-regime OLS refit")?;
    let fit_hi = ols_qr(&cols_hi, &y_hi, "the high-regime OLS refit")?;
    let n_low = low_rows.len();
    let n_high = high_rows.len();
    let ssr = fit_lo.ssr + fit_hi.ssr;
    if !(ssr > 0.0 && ssr.is_finite()) {
        return Err(RegimeError::NonFinite {
            what: "the pooled SETAR residual sum of squares (degenerate \
                   perfect fit)",
        });
    }
    let nf = n as f64;
    let sigma2 = ssr / (nf - 2.0 * k as f64);
    let sigma2_low = fit_lo.ssr / (n_low - k) as f64;
    let sigma2_high = fit_hi.ssr / (n_high - k) as f64;
    let m_params = (2 * k + 1) as f64;
    let aic = nf * (ssr / nf).ln() + 2.0 * m_params;
    let bic = nf * (ssr / nf).ln() + m_params * nf.ln();

    Ok(SetarFit {
        threshold,
        delay,
        coefs_low: fit_lo.params,
        coefs_high: fit_hi.params,
        se_low: fit_lo.bse,
        se_high: fit_hi.bse,
        n_low,
        n_high,
        nobs: n,
        ssr,
        sigma2,
        sigma2_low,
        sigma2_high,
        aic,
        bic,
        thresholds: scan.cand_gamma,
        ssr_path: prof.ssr_path,
        min_regime: scan.min_regime,
        k,
    })
}

// ------------------------------------------------------------------ test

/// Hansen (1996) sup-F test of linearity against a two-regime SETAR(`p`)
/// alternative with delay `delay`, p-valued by the fixed-regressor wild
/// bootstrap.
///
/// The statistic is `sup-F = n (S0 - S1) / S1` (Hansen 1997): `S0` the SSR
/// of the linear AR(`p`), `S1` the SETAR SSR minimized over the trimmed
/// candidate grid. Under the null the threshold is an unidentified nuisance
/// parameter (the Davies problem), so no chi-squared p-value exists; each
/// bootstrap replication regresses `y*_t = ehat_t eta_t` (`eta_t` iid
/// standard normal, `ehat` the linear-fit residuals) on the *same* fixed
/// regressors and recomputes the sup over the *same* grid. The p-value is
/// `(1 + #{F* >= F}) / (n_boot + 1)`.
///
/// Replications run in parallel over rayon with one SeedSequence-spawned
/// Philox substream each ([`tsecon_bootstrap::par_replicate`]), so the
/// result is bit-identical for a given `seed` at any thread count.
///
/// # Errors
///
/// The input errors of [`setar`], plus
/// [`RegimeError::InvalidParameter`] for `n_boot = 0` and
/// [`RegimeError::NonFinite`] if the observed threshold fit is a degenerate
/// perfect fit.
pub fn setar_test(
    y: &[f64],
    p: usize,
    delay: usize,
    trim: f64,
    constant: bool,
    n_boot: usize,
    seed: u64,
) -> Result<SetarTest, RegimeError> {
    validate_common(y, p, trim)?;
    validate_delay(delay)?;
    if n_boot == 0 {
        return Err(RegimeError::InvalidParameter {
            name: "n_boot",
            value: 0.0,
            requirement: "n_boot >= 1 (the null distribution is available \
                          only by bootstrap; 499+ recommended)",
        });
    }
    let start = p.max(delay);
    let k = p + usize::from(constant);
    check_length(y.len(), start, k, trim)?;

    let design = build_design(y, p, delay, start, constant);
    let scan = Scan::build(&design, trim)?;
    let prof = scan.profile(&design, &design.y);
    let s1 = prof.ssr_path[prof.best];
    if s1 <= 0.0 {
        return Err(RegimeError::NonFinite {
            what: "the SETAR residual sum of squares (degenerate perfect \
                   fit; the sup-F statistic is unbounded)",
        });
    }

    // The observed statistic and per-candidate F path use the QR linear fit
    // for S0 (also the source of the bootstrap residuals).
    let lin = ols_qr(&design.cols, &design.y, "the null linear AR fit")?;
    let s0 = lin.ssr;
    let nf = design.n as f64;
    let stat = nf * (s0 - s1).max(0.0) / s1;
    let f_path: Vec<f64> = prof
        .ssr_path
        .iter()
        .map(|&s| {
            if s > 0.0 {
                nf * (s0 - s).max(0.0) / s
            } else {
                f64::INFINITY
            }
        })
        .collect();

    let resid = lin.resid;
    let boot_stats: Vec<f64> = par_replicate(seed, n_boot, |_rep, stream| {
        let ystar: Vec<f64> = resid
            .iter()
            .map(|&e| e * WildWeights::Normal.draw(stream))
            .collect();
        scan.supf(&design, &ystar)
    })
    .map_err(|_| RegimeError::InvalidParameter {
        name: "n_boot",
        value: n_boot as f64,
        requirement: "n_boot within the RNG substream spawn limit (< 2^32)",
    })?;

    let exceed = boot_stats.iter().filter(|&&f| f >= stat).count();
    let p_value = (1 + exceed) as f64 / (n_boot + 1) as f64;

    Ok(SetarTest {
        stat,
        p_value,
        threshold: scan.cand_gamma[prof.best],
        delay,
        n_boot,
        nobs: design.n,
        ssr_linear: s0,
        ssr_setar: s1,
        thresholds: scan.cand_gamma,
        f_path,
        boot_stats,
    })
}
