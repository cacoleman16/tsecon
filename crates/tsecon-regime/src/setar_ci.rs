//! Hansen (1997, 2000) likelihood-ratio confidence sets for the threshold
//! of a two-regime SETAR — and of the general sample-splitting regression
//! the SETAR is a special case of — with the heteroskedasticity-robust
//! scaling and the conservative slope intervals built on them.
//!
//! # The problem
//!
//! The concentrated-least-squares threshold estimate `gamma_hat` of
//! [`setar`] is superconsistent (rate `n`, Chan 1993), but its sampling
//! distribution is nonstandard and, with a *fixed* threshold effect,
//! depends on nuisance parameters that cannot be estimated. Hansen (2000,
//! "Sample splitting and threshold estimation", *Econometrica* 68(3))
//! solves this with a "small threshold effect" asymptotic frame
//! (`delta_n = c n^{-a}`, `0 < a < 1/2`) in which the **likelihood-ratio
//! statistic for `gamma`** has a free-of-nuisance-parameters limit;
//! Hansen (1997, "Inference in TAR models", *Studies in Nonlinear Dynamics
//! & Econometrics* 2(1)) applies the same construction to the threshold
//! autoregression. The recipe:
//!
//! ```text
//! LR_n(gamma) = n (S_n(gamma) - S_n(gamma_hat)) / S_n(gamma_hat),
//! ```
//!
//! with `S_n(gamma)` the pooled two-regime SSR at the split `{q_t <=
//! gamma}` — the profile the concentrated scan already produces. Under
//! homoskedastic errors `LR_n(gamma_0) -> xi` with the closed-form law
//!
//! ```text
//! P(xi <= x) = (1 - exp(-x/2))^2,
//! ```
//!
//! so the `(1 - alpha)` critical value is `c(alpha) = -2 ln(1 - sqrt(1 -
//! alpha))` (Hansen 2000, Table 1: 4.50 / 5.10 / 5.94 / 6.53 / 7.35 / 8.75
//! / 10.59 at 80 / 85 / 90 / 92.5 / 95 / 97.5 / 99%) and the p-value
//! function is `p(x) = 1 - (1 - exp(-x/2))^2`. The **confidence set** is
//! the no-rejection region of the LR test inverted over the candidate grid,
//!
//! ```text
//! Gamma_hat(alpha) = { gamma : LR_n(gamma) <= c(alpha) },
//! ```
//!
//! which always contains `gamma_hat` (where `LR_n = 0`), is typically
//! **asymmetric** around it, and can be **disjoint** when the SSR profile
//! has several near-minimal valleys — [`ThresholdCi`] therefore returns the
//! set as a list of closed intervals plus its convex hull, and never
//! pretends the set is an interval.
//!
//! Because `S_n(gamma)` only changes when `gamma` crosses an observed value
//! of the threshold variable, `LR_n` is a **step function**: it is constant
//! on `[gamma_i, gamma_{i+1})` between adjacent candidates. The intervals
//! are reported by their grid endpoints, as in Hansen's own programs (the
//! minimum and maximum candidate in each run), and
//! [`ThresholdCiOptions::null_threshold`] evaluates `LR_n` at an arbitrary
//! `gamma_0` exactly, through the candidate whose split it shares.
//!
//! # Heteroskedasticity (Hansen 2000, §3.4)
//!
//! Under conditional heteroskedasticity the limit is `xi` scaled by
//!
//! ```text
//! eta^2 = E[e^2 (x'delta)^2 | q = gamma_0] / (sigma^2 E[(x'delta)^2 | q = gamma_0]),
//! ```
//!
//! `delta = beta_1 - beta_2` the threshold effect, and the set becomes
//! `{gamma : LR_n(gamma) <= eta_hat^2 c(alpha)}`. `eta_hat^2` is estimated
//! the way Hansen's own programs (`thresh.prc` / `thresh.m`) do: with the
//! regime fits at `gamma_hat`, residuals `e_hat_t`, and `x_t' delta_hat`,
//! regress `r1_t = (x_t' delta_hat)^2` and `r2_t = e_hat_t^2 (x_t'
//! delta_hat)^2` each on a quadratic polynomial in the threshold variable
//! `q_t` (with intercept), evaluate both fitted values at `q = gamma_hat`
//! (`g1`, `g2`), and set `eta_hat^2 = (g2 / g1) / sigma_hat^2` with
//! `sigma_hat^2 = S_n(gamma_hat) / n`. The implementation fits the quadratic
//! in the centered basis `[1, q - gamma_hat, (q - gamma_hat)^2]`, which spans
//! the same column space, so the fitted value at `gamma_hat` is the
//! intercept — algebraically identical to Hansen's `[1, q, q^2]` form and
//! better conditioned. With `het_robust = false`, `eta^2 = 1` and the
//! homoskedastic set is returned. (Hansen 1997 notes that the LR
//! construction is not scale-free under heteroskedasticity; the
//! correction is what makes it usable there.)
//!
//! # Conservative slope intervals (Hansen 2000, §3.3)
//!
//! Conventional confidence intervals for the regime coefficients treat
//! `gamma_hat` as known; Hansen shows that is asymptotically valid but
//! recommends, for a finite-sample guard, the **union** over every `gamma`
//! in a threshold confidence region of the conventional
//! `(1 - alpha_slope)` intervals `beta_hat_j(gamma) +/- z se_j(gamma)`. His
//! applied work takes an 80% threshold region for this (the region level is
//! [`ThresholdCiOptions::slope_region_level`], default `0.80`); the
//! per-`gamma` standard errors are the classical per-regime ones (so at
//! `gamma_hat` the conventional interval is exactly `setar`'s `se_low` /
//! `se_high` times the normal quantile) or, with `het_robust`, White's HC0.
//! The result is conservative by construction — each union is at least as
//! wide as the interval at `gamma_hat`.
//!
//! # Validation and honesty
//!
//! No third-party threshold-CI implementation runs in the fixture
//! container (no R `tsDyn`, no Hansen GAUSS/MATLAB programs), so the golden
//! (`fixtures/setar_ci.json`) is a cross-implementation transcription:
//! an independent NumPy implementation of the LR profile, the `eta^2`
//! regressions, the set construction, and the slope unions on seeded SETAR
//! series, pinned at 1e-10, plus the closed-form critical values and
//! p-value function pinned as documented formulas. Coverage of the
//! `gamma_0` set is **measured** by seeded Monte Carlo in the crate's
//! property tests, on a design after Hansen's (2000, §5) threshold-
//! regression experiment and on a SETAR(2); the numbers are quoted in the
//! model card, and the asymptotic theory (conservative when the threshold
//! effect is fixed rather than shrinking) is what they are read against.
//!
//! References: Hansen (1997), SNDE 2(1); Hansen (2000), Econometrica 68(3);
//! Chan (1993), Annals of Statistics 21(1).

use crate::error::RegimeError;
use crate::linsolve::{chol_solve, cholesky};
use crate::setar::{build_design, ols_qr, setar, Design, Ols, Scan};
use tsecon_stats::special::inv_norm_cdf;

// --------------------------------------------------------------- options

/// Options of [`setar_threshold_ci`] / [`threshold_regression_ci`].
#[derive(Debug, Clone, PartialEq)]
pub struct ThresholdCiOptions {
    /// Confidence level of the threshold set (`1 - alpha`), in `(0, 1)`.
    /// Default `0.95`.
    pub level: f64,
    /// Scale the critical value by Hansen's (2000, §3.4) `eta_hat^2`
    /// heteroskedasticity correction. Default `false`.
    pub het_robust: bool,
    /// Level of the conventional per-regime slope intervals whose union
    /// over the threshold region is reported (`None`: no slope intervals).
    /// Default `None`.
    pub slope_level: Option<f64>,
    /// Level of the threshold region the slope union runs over (Hansen's
    /// applied convention is 80%). Default `0.80`; acts only with
    /// `slope_level`.
    pub slope_region_level: f64,
    /// A null threshold `gamma_0` at which to report `LR_n(gamma_0)` and
    /// its p-value (test inversion at one point). Must lie inside the
    /// candidate grid `[thresholds[0], thresholds[last]]`. Default `None`.
    pub null_threshold: Option<f64>,
}

impl Default for ThresholdCiOptions {
    fn default() -> Self {
        Self {
            level: 0.95,
            het_robust: false,
            slope_level: None,
            slope_region_level: 0.80,
            null_threshold: None,
        }
    }
}

// --------------------------------------------------------------- results

/// The conservative slope intervals of Hansen (2000, §3.3): per regime and
/// coefficient, the union over every candidate threshold in the
/// `region_level` confidence region of the conventional `level` interval.
#[derive(Debug, Clone, PartialEq)]
pub struct SlopeCi {
    /// Level of the conventional per-`gamma` intervals.
    pub level: f64,
    /// Level of the threshold region unioned over.
    pub region_level: f64,
    /// Smallest candidate threshold in the region.
    pub region_low: f64,
    /// Largest candidate threshold in the region.
    pub region_high: f64,
    /// Number of candidate thresholds in the region.
    pub n_region: usize,
    /// Lower bounds for the low-regime coefficients (`[constant?, lag 1,
    /// ..., lag p]` for a SETAR; the supplied column order otherwise).
    pub low_lower: Vec<f64>,
    /// Upper bounds for the low-regime coefficients.
    pub low_upper: Vec<f64>,
    /// Lower bounds for the high-regime coefficients.
    pub high_lower: Vec<f64>,
    /// Upper bounds for the high-regime coefficients.
    pub high_upper: Vec<f64>,
}

/// Output of [`setar_threshold_ci`] / [`threshold_regression_ci`]: the
/// Hansen likelihood-ratio confidence set for the threshold.
#[derive(Debug, Clone, PartialEq)]
pub struct ThresholdCi {
    /// The threshold estimate `gamma_hat` (bit-identical to [`setar`]'s).
    pub threshold: f64,
    /// The SETAR delay used (`None` for the general threshold regression).
    pub delay: Option<usize>,
    /// Usable observations `n`.
    pub nobs: usize,
    /// Regressors per regime `k`.
    pub k: usize,
    /// The confidence level requested.
    pub level: f64,
    /// The closed-form critical value `c(alpha) = -2 ln(1 - sqrt(level))`.
    pub lr_crit: f64,
    /// The critical value actually applied, `eta2 * lr_crit`.
    pub lr_crit_scaled: f64,
    /// Hansen's heteroskedasticity scale `eta_hat^2` (exactly `1` unless
    /// `het_robust`).
    pub eta2: f64,
    /// Whether the heteroskedasticity correction was applied.
    pub het_robust: bool,
    /// The candidate threshold grid (bit-identical to [`setar`]'s
    /// `thresholds`), ascending.
    pub thresholds: Vec<f64>,
    /// Pooled SSR per candidate (bit-identical to [`setar`]'s `ssr_path`).
    pub ssr_path: Vec<f64>,
    /// `LR_n(gamma) = n (S_n(gamma) - S_min) / S_min` per candidate; exactly
    /// `0` at `gamma_hat`.
    pub lr_stat: Vec<f64>,
    /// `lr_stat[i] <= lr_crit_scaled` per candidate.
    pub in_set: Vec<bool>,
    /// The confidence set as maximal runs of in-set candidates, each
    /// `(low, high)` in threshold units, ascending; can hold several
    /// intervals.
    pub intervals: Vec<(f64, f64)>,
    /// `intervals.len() == 1`.
    pub is_connected: bool,
    /// Convex hull of the set: smallest in-set candidate.
    pub ci_low: f64,
    /// Convex hull of the set: largest in-set candidate.
    pub ci_high: f64,
    /// Number of candidates in the set.
    pub n_in_set: usize,
    /// With a `null_threshold`: the candidate whose split `gamma_0` shares
    /// (the largest candidate `<= gamma_0`).
    pub null_threshold_used: Option<f64>,
    /// With a `null_threshold`: `LR_n(gamma_0)` (unscaled).
    pub lr_at_null: Option<f64>,
    /// With a `null_threshold`: the p-value `p(LR_n(gamma_0) / eta2)`,
    /// `p(x) = 1 - (1 - exp(-x/2))^2`.
    pub pvalue_at_threshold: Option<f64>,
    /// With a `slope_level`: the conservative slope intervals.
    pub slope: Option<SlopeCi>,
}

// ----------------------------------------------------------- closed forms

/// Hansen's (2000, Table 1) closed-form critical value of the threshold LR
/// statistic, `c = -2 ln(1 - sqrt(level))`, i.e. the `level` quantile of
/// `P(xi <= x) = (1 - exp(-x/2))^2`: 4.50 / 5.94 / 7.35 / 10.59 at 80 /
/// 90 / 95 / 99%.
///
/// # Errors
///
/// [`RegimeError::InvalidParameter`] unless `0 < level < 1`.
pub fn hansen_lr_critical_value(level: f64) -> Result<f64, RegimeError> {
    validate_level(level, "level")?;
    Ok(-2.0 * (1.0 - level.sqrt()).ln())
}

/// The p-value function of the threshold LR statistic, `p(x) = 1 - (1 -
/// exp(-x/2))^2`, evaluated in the cancellation-free form `exp(-x/2) (2 -
/// exp(-x/2))` (the naive form loses relative accuracy in the far tail).
///
/// # Errors
///
/// [`RegimeError::NonFinite`] for NaN, [`RegimeError::InvalidParameter`]
/// for a negative statistic.
pub fn hansen_lr_pvalue(lr: f64) -> Result<f64, RegimeError> {
    if lr.is_nan() {
        return Err(RegimeError::NonFinite {
            what: "the likelihood-ratio statistic passed to the Hansen p-value function",
        });
    }
    if lr < 0.0 {
        return Err(RegimeError::InvalidParameter {
            name: "lr",
            value: lr,
            requirement: "lr >= 0 (the threshold LR statistic n (S(gamma) - S_min) / \
                          S_min is nonnegative by construction)",
        });
    }
    let e = (-lr / 2.0).exp();
    Ok(e * (2.0 - e))
}

fn validate_level(level: f64, name: &'static str) -> Result<(), RegimeError> {
    if !(level > 0.0 && level < 1.0) {
        return Err(RegimeError::InvalidParameter {
            name,
            value: level,
            requirement: "0 < level < 1 (a confidence level such as 0.95; NaN and \
                          infinities are refused)",
        });
    }
    Ok(())
}

fn validate_options(opts: &ThresholdCiOptions) -> Result<(), RegimeError> {
    validate_level(opts.level, "level")?;
    if let Some(sl) = opts.slope_level {
        validate_level(sl, "slope_level")?;
        validate_level(opts.slope_region_level, "slope_region_level")?;
    }
    if let Some(g0) = opts.null_threshold {
        if !g0.is_finite() {
            return Err(RegimeError::NonFinite {
                what: "null_threshold (the null value gamma_0 must be a finite number \
                       inside the candidate grid)",
            });
        }
    }
    Ok(())
}

// --------------------------------------------------------------- helpers

/// OLS in each regime of the split `{z <= gamma}` / `{z > gamma}` by QR,
/// with the row indices of each regime.
fn split_fit(
    design: &Design,
    gamma: f64,
) -> Result<(Ols, Ols, Vec<usize>, Vec<usize>), RegimeError> {
    let n = design.n;
    let low_rows: Vec<usize> = (0..n).filter(|&t| design.z[t] <= gamma).collect();
    let high_rows: Vec<usize> = (0..n).filter(|&t| design.z[t] > gamma).collect();
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
    let fit_lo = ols_qr(
        &cols_lo,
        &y_lo,
        "the low-regime OLS refit at a candidate threshold",
    )?;
    let fit_hi = ols_qr(
        &cols_hi,
        &y_hi,
        "the high-regime OLS refit at a candidate threshold",
    )?;
    Ok((fit_lo, fit_hi, low_rows, high_rows))
}

/// White's HC0 standard errors `sqrt(diag[(X'X)^{-1} X' diag(e^2) X
/// (X'X)^{-1}])` of one regime regression.
fn hc0_se(cols: &[Vec<f64>], resid: &[f64], what: &'static str) -> Result<Vec<f64>, RegimeError> {
    let k = cols.len();
    let n = resid.len();
    let mut xtx = vec![0.0_f64; k * k];
    let mut meat = vec![0.0_f64; k * k];
    for t in 0..n {
        let w = resid[t] * resid[t];
        for a in 0..k {
            let xa = cols[a][t];
            for b in 0..=a {
                let prod = xa * cols[b][t];
                xtx[a * k + b] += prod;
                meat[a * k + b] += w * prod;
            }
        }
    }
    for a in 0..k {
        for b in (a + 1)..k {
            xtx[a * k + b] = xtx[b * k + a];
            meat[a * k + b] = meat[b * k + a];
        }
    }
    let l = cholesky(&xtx, k).ok_or(RegimeError::Singular { what })?;
    // A = (X'X)^{-1}, column by column.
    let mut ainv = vec![0.0_f64; k * k];
    for j in 0..k {
        let mut e = vec![0.0_f64; k];
        e[j] = 1.0;
        let col = chol_solve(&l, k, &e);
        for (i, &v) in col.iter().enumerate() {
            ainv[i * k + j] = v;
        }
    }
    let mut se = vec![0.0_f64; k];
    for (i, se_i) in se.iter_mut().enumerate() {
        let mut v = 0.0;
        for a in 0..k {
            let mut inner = 0.0;
            for b in 0..k {
                inner += meat[a * k + b] * ainv[b * k + i];
            }
            v += ainv[i * k + a] * inner;
        }
        if !v.is_finite() {
            return Err(RegimeError::NonFinite { what });
        }
        *se_i = v.max(0.0).sqrt();
    }
    Ok(se)
}

/// Hansen's `eta_hat^2` (see the module docs) from the regime fits at
/// `gamma_hat`.
fn eta2_hansen(design: &Design, gamma_hat: f64, s_min: f64) -> Result<f64, RegimeError> {
    let n = design.n;
    let (fit_lo, fit_hi, rows_lo, rows_hi) = split_fit(design, gamma_hat)?;
    let delta: Vec<f64> = fit_lo
        .params
        .iter()
        .zip(&fit_hi.params)
        .map(|(&a, &b)| a - b)
        .collect();
    let mut resid = vec![0.0_f64; n];
    for (j, &t) in rows_lo.iter().enumerate() {
        resid[t] = fit_lo.resid[j];
    }
    for (j, &t) in rows_hi.iter().enumerate() {
        resid[t] = fit_hi.resid[j];
    }
    let mut r1 = vec![0.0_f64; n];
    let mut r2 = vec![0.0_f64; n];
    for t in 0..n {
        let xd: f64 = delta
            .iter()
            .zip(&design.cols)
            .map(|(&d, col)| d * col[t])
            .sum();
        r1[t] = xd * xd;
        r2[t] = resid[t] * resid[t] * xd * xd;
    }
    let qc: Vec<f64> = design.z.iter().map(|&q| q - gamma_hat).collect();
    let qcols = vec![
        vec![1.0_f64; n],
        qc.clone(),
        qc.iter().map(|&v| v * v).collect(),
    ];
    let m1 = ols_qr(
        &qcols,
        &r1,
        "the quadratic regression of (x'delta)^2 on the threshold variable \
         (the eta^2 heteroskedasticity correction needs at least three distinct \
         threshold-variable values)",
    )?;
    let m2 = ols_qr(
        &qcols,
        &r2,
        "the quadratic regression of e^2 (x'delta)^2 on the threshold variable \
         (the eta^2 heteroskedasticity correction needs at least three distinct \
         threshold-variable values)",
    )?;
    let g1 = m1.params[0];
    let g2 = m2.params[0];
    let sigma2 = s_min / n as f64;
    let eta2 = (g2 / g1) / sigma2;
    if !(eta2 > 0.0 && eta2.is_finite()) {
        return Err(RegimeError::NonFinite {
            what: "the heteroskedasticity correction eta^2 (Hansen 2000, section 3.4): \
                   the quadratic fits of (x'delta)^2 and e^2 (x'delta)^2 on the \
                   threshold variable gave a non-positive or non-finite ratio at \
                   the threshold estimate, so the correction is not identified \
                   here; pass het_robust = false, or use a sample with a larger \
                   threshold effect",
        });
    }
    Ok(eta2)
}

// ------------------------------------------------------------------ core

/// The LR profile, set, and extras from a design plus its concentrated
/// scan (`thresholds`, `ssr_path`, `best` — the first index attaining the
/// minimal SSR, whose candidate is `gamma_hat`).
fn ci_core(
    design: &Design,
    thresholds: Vec<f64>,
    ssr_path: Vec<f64>,
    best: usize,
    delay: Option<usize>,
    opts: &ThresholdCiOptions,
) -> Result<ThresholdCi, RegimeError> {
    let n = design.n;
    let nf = n as f64;
    let g = thresholds.len();
    let gamma_hat = thresholds[best];
    let s_min = ssr_path[best];
    if !(s_min > 0.0 && s_min.is_finite()) {
        return Err(RegimeError::NonFinite {
            what: "the SETAR residual sum of squares at the threshold estimate \
                   (degenerate perfect fit; the LR profile is unbounded)",
        });
    }
    let lr_stat: Vec<f64> = ssr_path
        .iter()
        .map(|&s| (nf * (s - s_min) / s_min).max(0.0))
        .collect();

    let lr_crit = hansen_lr_critical_value(opts.level)?;
    let eta2 = if opts.het_robust {
        eta2_hansen(design, gamma_hat, s_min)?
    } else {
        1.0
    };
    let lr_crit_scaled = eta2 * lr_crit;

    let in_set: Vec<bool> = lr_stat.iter().map(|&v| v <= lr_crit_scaled).collect();
    let intervals = runs(&thresholds, &in_set);
    let n_in_set = in_set.iter().filter(|&&b| b).count();
    let ci_low = intervals.first().map_or(gamma_hat, |iv| iv.0);
    let ci_high = intervals.last().map_or(gamma_hat, |iv| iv.1);

    // Test inversion at one null value: LR_n is a step function of gamma,
    // constant on [gamma_i, gamma_{i+1}), so gamma_0 shares the split of the
    // largest candidate <= gamma_0.
    let (null_threshold_used, lr_at_null, pvalue_at_threshold) = match opts.null_threshold {
        None => (None, None, None),
        Some(g0) => {
            if g0 < thresholds[0] || g0 > thresholds[g - 1] {
                return Err(RegimeError::InvalidParameter {
                    name: "null_threshold",
                    value: g0,
                    requirement: "a value inside the trimmed candidate grid \
                                  [thresholds[0], thresholds[last]] (outside it the \
                                  implied sample split violates the trimming, so the \
                                  LR statistic is not defined)",
                });
            }
            let idx = thresholds.partition_point(|&t| t <= g0) - 1;
            let lr0 = lr_stat[idx];
            (
                Some(thresholds[idx]),
                Some(lr0),
                Some(hansen_lr_pvalue(lr0 / eta2)?),
            )
        }
    };

    let slope = match opts.slope_level {
        None => None,
        Some(sl) => Some(slope_union(
            design,
            &thresholds,
            &lr_stat,
            eta2,
            sl,
            opts.slope_region_level,
            opts.het_robust,
        )?),
    };

    Ok(ThresholdCi {
        threshold: gamma_hat,
        delay,
        nobs: n,
        k: design.k,
        level: opts.level,
        lr_crit,
        lr_crit_scaled,
        eta2,
        het_robust: opts.het_robust,
        thresholds,
        ssr_path,
        lr_stat,
        is_connected: intervals.len() == 1,
        in_set,
        intervals,
        ci_low,
        ci_high,
        n_in_set,
        null_threshold_used,
        lr_at_null,
        pvalue_at_threshold,
        slope,
    })
}

/// Maximal runs of `true` in `flag`, as `(thresholds[start],
/// thresholds[end])` pairs.
fn runs(thresholds: &[f64], flag: &[bool]) -> Vec<(f64, f64)> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < flag.len() {
        if flag[i] {
            let start = i;
            while i + 1 < flag.len() && flag[i + 1] {
                i += 1;
            }
            out.push((thresholds[start], thresholds[i]));
        }
        i += 1;
    }
    out
}

/// The Hansen (2000, §3.3) union of conventional slope intervals over the
/// `region_level` threshold region.
fn slope_union(
    design: &Design,
    thresholds: &[f64],
    lr_stat: &[f64],
    eta2: f64,
    slope_level: f64,
    region_level: f64,
    het_robust: bool,
) -> Result<SlopeCi, RegimeError> {
    let k = design.k;
    let crit_region = eta2 * hansen_lr_critical_value(region_level)?;
    let z = inv_norm_cdf(1.0 - (1.0 - slope_level) / 2.0).map_err(|_| {
        RegimeError::InvalidParameter {
            name: "slope_level",
            value: slope_level,
            requirement: "0 < slope_level < 1",
        }
    })?;
    let mut low_lower = vec![f64::INFINITY; k];
    let mut low_upper = vec![f64::NEG_INFINITY; k];
    let mut high_lower = vec![f64::INFINITY; k];
    let mut high_upper = vec![f64::NEG_INFINITY; k];
    let mut region_low = f64::INFINITY;
    let mut region_high = f64::NEG_INFINITY;
    let mut n_region = 0usize;
    for (i, &gamma) in thresholds.iter().enumerate() {
        if lr_stat[i] > crit_region {
            continue;
        }
        n_region += 1;
        region_low = region_low.min(gamma);
        region_high = region_high.max(gamma);
        let (fit_lo, fit_hi, rows_lo, rows_hi) = split_fit(design, gamma)?;
        let (se_lo, se_hi) = if het_robust {
            let take = |rows: &[usize]| -> Vec<Vec<f64>> {
                design
                    .cols
                    .iter()
                    .map(|c| rows.iter().map(|&t| c[t]).collect())
                    .collect()
            };
            (
                hc0_se(
                    &take(&rows_lo),
                    &fit_lo.resid,
                    "the low-regime HC0 covariance at a candidate threshold",
                )?,
                hc0_se(
                    &take(&rows_hi),
                    &fit_hi.resid,
                    "the high-regime HC0 covariance at a candidate threshold",
                )?,
            )
        } else {
            (fit_lo.bse.clone(), fit_hi.bse.clone())
        };
        for j in 0..k {
            low_lower[j] = low_lower[j].min(fit_lo.params[j] - z * se_lo[j]);
            low_upper[j] = low_upper[j].max(fit_lo.params[j] + z * se_lo[j]);
            high_lower[j] = high_lower[j].min(fit_hi.params[j] - z * se_hi[j]);
            high_upper[j] = high_upper[j].max(fit_hi.params[j] + z * se_hi[j]);
        }
    }
    Ok(SlopeCi {
        level: slope_level,
        region_level,
        region_low,
        region_high,
        n_region,
        low_lower,
        low_upper,
        high_lower,
        high_upper,
    })
}

// -------------------------------------------------------- public entries

/// Hansen (1997, 2000) likelihood-ratio confidence set for the threshold of
/// a two-regime SETAR(`p`), on top of exactly the [`setar`] fit.
///
/// The fit is [`setar`]`(y, p, delays, trim, constant)` itself — the
/// returned `threshold`, `delay`, `thresholds`, and `ssr_path` are
/// bit-identical to its — and the profile `LR_n(gamma) = n (S_n(gamma) -
/// S_min) / S_min` is inverted against the closed-form critical value
/// `c = -2 ln(1 - sqrt(level))` (times `eta_hat^2` when `het_robust`). See
/// the module docs for the construction, the reporting convention for the
/// (possibly disjoint) set, the heteroskedasticity correction, and the
/// conservative slope intervals.
///
/// # Errors
///
/// The input errors of [`setar`]; [`RegimeError::InvalidParameter`] for a
/// `level` / `slope_level` / `slope_region_level` outside `(0, 1)` or a
/// `null_threshold` outside the candidate grid; [`RegimeError::NonFinite`]
/// for a NaN `null_threshold`, a degenerate perfect fit, or an
/// unidentified `eta^2`; [`RegimeError::Singular`] if a regime refit or
/// the `eta^2` quadratic regression is collinear.
pub fn setar_threshold_ci(
    y: &[f64],
    p: usize,
    delays: &[usize],
    trim: f64,
    constant: bool,
    opts: &ThresholdCiOptions,
) -> Result<ThresholdCi, RegimeError> {
    validate_options(opts)?;
    let fit = setar(y, p, delays, trim, constant)?;
    let max_delay = delays.iter().copied().max().unwrap_or(1);
    let start = p.max(max_delay);
    let design = build_design(y, p, fit.delay, start, constant);
    let best = fit
        .thresholds
        .iter()
        .position(|&g| g == fit.threshold)
        .ok_or(RegimeError::NonFinite {
            what: "the SETAR threshold estimate (not found on its own candidate grid)",
        })?;
    ci_core(
        &design,
        fit.thresholds,
        fit.ssr_path,
        best,
        Some(fit.delay),
        opts,
    )
}

/// Hansen (2000) likelihood-ratio confidence set for the threshold of the
/// general sample-splitting regression `y_t = x_t' beta_1 1{q_t <= gamma}
/// + x_t' beta_2 1{q_t > gamma} + e_t` on user-supplied regressor columns
/// `x` (include the constant yourself) and threshold variable `q`.
///
/// The threshold is estimated by the same concentrated scan [`setar`] uses
/// (trimmed unique order statistics of `q`, each regime holding at least
/// `max(k + 1, ceil(trim n))` observations, first minimal-SSR candidate
/// wins), then the set is built exactly as in [`setar_threshold_ci`]. This
/// is the estimator of Hansen's paper itself; the crate's coverage tests
/// run it on a design after his Monte Carlo experiment.
///
/// # Errors
///
/// [`RegimeError::NonFinite`] for non-finite inputs;
/// [`RegimeError::DimensionMismatch`] for columns or `q` of a different
/// length than `y`; [`RegimeError::InvalidSpec`] for no regressors or a
/// constant `q`; [`RegimeError::InvalidParameter`] for `trim` outside
/// `(0, 0.5)` or the option errors of [`setar_threshold_ci`];
/// [`RegimeError::InsufficientData`] / [`RegimeError::Singular`] as the
/// scan reports them.
pub fn threshold_regression_ci(
    y: &[f64],
    x: &[Vec<f64>],
    q: &[f64],
    trim: f64,
    opts: &ThresholdCiOptions,
) -> Result<ThresholdCi, RegimeError> {
    validate_options(opts)?;
    let n = y.len();
    let k = x.len();
    if k == 0 {
        return Err(RegimeError::InvalidSpec {
            what: "the threshold regression needs at least one regressor column in x \
                   (pass the constant as a column of ones if the model has one)",
        });
    }
    if q.len() != n {
        return Err(RegimeError::DimensionMismatch {
            what: "the threshold variable q must have one value per observation of y",
            expected: n,
            actual: q.len(),
        });
    }
    for col in x {
        if col.len() != n {
            return Err(RegimeError::DimensionMismatch {
                what: "every regressor column of x must have one value per observation of y",
                expected: n,
                actual: col.len(),
            });
        }
    }
    if y.iter().any(|v| !v.is_finite()) {
        return Err(RegimeError::NonFinite {
            what: "the response y (the threshold regression requires finite observations)",
        });
    }
    if q.iter().any(|v| !v.is_finite()) {
        return Err(RegimeError::NonFinite {
            what: "the threshold variable q (finite values required)",
        });
    }
    if x.iter().any(|c| c.iter().any(|v| !v.is_finite())) {
        return Err(RegimeError::NonFinite {
            what: "the regressor columns x (finite values required)",
        });
    }
    if !(trim > 0.0 && trim < 0.5) {
        return Err(RegimeError::InvalidParameter {
            name: "trim",
            value: trim,
            requirement: "0 < trim < 0.5 (the fraction of threshold-variable order \
                          statistics excluded at each end)",
        });
    }
    if n > 0 && q.iter().all(|&v| v == q[0]) {
        return Err(RegimeError::InvalidSpec {
            what: "the threshold variable q is constant: a threshold regression needs \
                   variation in q to split the sample",
        });
    }
    let min_regime = (k + 1).max((trim * n as f64).ceil() as usize);
    if n < 2 * min_regime {
        return Err(RegimeError::InsufficientData {
            what: "y and q under the requested trim (the sample must hold two regimes \
                   of max(k + 1, ceil(trim n)) observations each)",
            needed: 2 * min_regime,
            got: n,
        });
    }
    let design = Design {
        cols: x.to_vec(),
        y: y.to_vec(),
        z: q.to_vec(),
        n,
        k,
    };
    let scan = Scan::build(&design, trim)?;
    let prof = scan.profile(&design, &design.y);
    ci_core(
        &design,
        scan.cand_gamma,
        prof.ssr_path,
        prof.best,
        None,
        opts,
    )
}
