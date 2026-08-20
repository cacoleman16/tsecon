//! First-stage instrument-strength diagnostics for the proxy SVAR: the
//! Montiel Olea-Pflueger **effective F** under classical, HC1, or
//! HAC-Bartlett variance, with the MOP tau-based critical values stamped
//! beside it.
//!
//! # What the effective F is, and why the folklore threshold is not enough
//!
//! [`crate::proxy_svar`] identifies from a *single* instrument, so its first
//! stage is the just-identified scalar regression of the `norm_var` residual
//! on the proxy over the overlap. In that just-identified case the
//! Montiel Olea-Pflueger (2013, JBES) effective first-stage F-statistic
//! **coincides with the robust F** — the squared robust t-statistic of the
//! first-stage slope,
//!
//! ```text
//! F_eff = beta_hat^2 / Var_robust(beta_hat),
//! ```
//!
//! which is exactly what [`crate::proxy_svar`] already reports as
//! `first_stage_f` when `robust_f = true` (HC1). This module names that
//! equivalence, adds the HAC-Bartlett variance for serially correlated
//! proxies, and — the part the folklore omits — attaches the **critical
//! values the statistic is supposed to be compared against**.
//!
//! The "F > 10" rule of thumb descends from Staiger-Stock / Stock-Yogo
//! homoskedastic TSLS-bias calculations; it is not a valid threshold for a
//! robust F. Montiel Olea and Pflueger instead test the null "the worst-case
//! (Nagar-benchmark) relative bias exceeds `tau`" at level `alpha`; with one
//! instrument the effective degrees of freedom equal 1 and their critical
//! value reduces to a **noncentral chi-square quantile**
//!
//! ```text
//! cv(tau, alpha) = Q_{chi2'(df = 1, ncp = 1/tau)}(1 - alpha),
//! ```
//!
//! the construction implemented by the Stata `weakivtest` companion
//! (Pflueger & Wang 2015, Stata Journal 15(1)): critical value
//! `invnchi2(K_eff, K_eff * x, 1 - alpha) / K_eff` with `x = 1/tau` and
//! `K_eff = 1` when there is one instrument. At `alpha = 0.05` this gives
//!
//! | tau (worst-case relative bias) | critical value |
//! |---|---|
//! | 5%  | 37.42 |
//! | 10% | 23.11 |
//! | 20% | 15.06 |
//! | 30% | 12.05 |
//!
//! so the folklore "10" corresponds to *no* tabulated MOP row — an
//! effective F of 12 clears only the 30%-bias bar, and 23.1 is the number
//! that certifies the conventional 10% bias bound. (`weakivtest` prints
//! 12.039 for the last row because it rounds `x` to 3.33; this module uses
//! the exact `1/tau`.)
//!
//! [`FirstStageDiagnostics::tau_bound`] inverts the table: the smallest
//! `tau` whose weak-instrument null the observed effective F rejects at
//! `alpha = 0.05`. It is `+inf` when the effective F cannot even reject
//! zero relevance (F below the central chi-square quantile, 3.84 at 5%).
//!
//! # Honest scope
//!
//! * The MOP critical values are derived for the Nagar bias of TSLS/LIML in
//!   the linear IV model; carrying them to the proxy-SVAR first stage is
//!   the field's standard practice (they are the thresholds reported in
//!   applications of Montiel Olea-Stock-Watson-style SVAR-IV inference),
//!   not a theorem about the IRF estimand itself. The diagnostic gates
//!   *trust in Wald-type bands*; it does not repair them. When it flags
//!   weakness, the honest object is [`crate::proxy_ar::proxy_ar_sets`], whose
//!   validity does not depend on instrument strength at all.
//! * The Lewis-Mertens (FRBNY Staff Report 1020) generalized first-stage
//!   statistic extends this machinery to **multiple** endogenous
//!   regressors/instruments; with one instrument and one target shock it
//!   collapses to the effective F computed here, so this module implements
//!   the single-instrument case only and does not claim the general one.
//! * The classical (homoskedastic) F is reported for comparison with
//!   published tables (Gertler-Karadi 2015 report both), but the effective
//!   F under HC1/HAC is the number the MOP thresholds are for.
//!
//! Formulas were verified against the `weakivtest` implementation
//! (Pflueger-Wang) and the Windmeijer (2025, J. Econometrics) statement
//! that the just-identified effective F equals the robust F; the golden
//! fixture pins the regression algebra against statsmodels OLS with HC1 and
//! HAC(Bartlett) covariance and the critical values against
//! `scipy.stats.ncx2.ppf`.

use tsecon_linalg::faer::MatRef;
use tsecon_stats::special::{erfc, inv_norm_cdf};

use crate::error::IdentError;

/// Which variance estimator the effective F divides by.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FirstStageVariance {
    /// Homoskedastic OLS variance `s^2 / Smm`, `s^2 = SSE / (T_O - 2)`.
    /// The resulting statistic is the classical first-stage F; the MOP
    /// thresholds still apply (under homoskedasticity the effective F and
    /// the classical F estimate the same object).
    Classical,
    /// HC1 (heteroskedasticity-robust with the `T_O / (T_O - 2)` degrees-of-
    /// freedom correction). The default, and identical to the
    /// `robust_f = true` statistic of [`crate::proxy_svar`].
    Hc1,
    /// Bartlett-kernel HAC (Newey-West) with `lags` lags, sharing the HC1
    /// degrees-of-freedom correction — the `weakivtest` convention. Use for
    /// a time-aggregated or smoothed proxy whose score `m~_t e_t` may be
    /// serially correlated; a genuine surprise series should not need it.
    ///
    /// Autocovariances pair over **calendar time**: `t` and `t - j` must
    /// both lie inside the proxy's availability window, exactly as
    /// [`crate::proxy_ar::ArVariance::HacBartlett`] does, so interior NaN
    /// gaps are never spliced across.
    HacBartlett {
        /// Bartlett bandwidth `L`; must be smaller than the overlap count.
        lags: usize,
    },
}

/// First-stage strength diagnostics for a single-instrument proxy SVAR.
///
/// All statistics are computed from the regression of the `norm_var`
/// residual on `[1, m]` over the overlap (the finite-proxy rows), in the
/// identical operation order as [`crate::proxy_svar`], so `f_hc1` here and
/// that function's `first_stage_f` (with `robust_f = true`) agree
/// bit-for-bit on the same inputs.
#[derive(Debug, Clone, Copy)]
pub struct FirstStageDiagnostics {
    /// First-stage slope `beta_hat = Smy / Smm`.
    pub beta: f64,
    /// Standard error of `beta_hat` under the requested variance estimator.
    pub se: f64,
    /// The Montiel Olea-Pflueger **effective F** under the requested
    /// variance estimator: `beta_hat^2 / Var(beta_hat)`. This is the number
    /// the `mop_cv_*` thresholds are for.
    pub effective_f: f64,
    /// The classical (homoskedastic) first-stage F, for comparison with
    /// published tables.
    pub f_classical: f64,
    /// The HC1 effective F — always reported, whatever `variance` was
    /// requested, because it is the library-wide default
    /// ([`crate::proxy_svar`]'s `first_stage_f`).
    pub f_hc1: f64,
    /// First-stage `R^2 = Corr(m, u_norm)^2` — the Stock-Watson reliability
    /// reported by [`crate::proxy_svar`].
    pub reliability: f64,
    /// Overlap count `T_O` (finite proxy observations).
    pub n_proxy: usize,
    /// The HAC bandwidth actually used, or `None` for Classical / HC1.
    pub hac_lags: Option<usize>,
    /// MOP critical value at `tau = 5%`, `alpha = 0.05` (37.42).
    pub mop_cv_tau5: f64,
    /// MOP critical value at `tau = 10%`, `alpha = 0.05` (23.11) — the
    /// conventional "not weak" bar.
    pub mop_cv_tau10: f64,
    /// MOP critical value at `tau = 20%`, `alpha = 0.05` (15.06).
    pub mop_cv_tau20: f64,
    /// MOP critical value at `tau = 30%`, `alpha = 0.05` (12.05).
    pub mop_cv_tau30: f64,
    /// The smallest worst-case relative bias `tau` that the observed
    /// `effective_f` rejects at `alpha = 0.05` — the diagnostic's one-number
    /// summary. `+inf` when even zero relevance cannot be rejected
    /// (`effective_f` below the central `chi2_1` quantile, 3.84).
    pub tau_bound: f64,
    /// `effective_f <= mop_cv_tau10`: the instrument fails the conventional
    /// 10%-bias MOP bar. Route to [`crate::proxy_ar::proxy_ar_sets`].
    pub weak_mop_tau10: bool,
    /// `effective_f < 10`: fails even the (homoskedastic-folklore) rule of
    /// thumb. Kept because the literature reports it, not because it is a
    /// valid robust threshold — `mop_cv_tau10` is the honest bar.
    pub weak_folklore: bool,
}

/// The MOP tau grid stamped into every [`FirstStageDiagnostics`].
const MOP_TAUS: [f64; 4] = [0.05, 0.10, 0.20, 0.30];
/// The significance level of the stamped critical values (the `weakivtest`
/// default).
const MOP_ALPHA: f64 = 0.05;

/// First-stage instrument-strength diagnostics: the effective F under the
/// requested variance estimator, the MOP tau-based critical values, and the
/// implied worst-case-bias bound.
///
/// `u` is the `T x n` matrix of reduced-form VAR residuals, `proxy` the
/// length-`T` instrument aligned to the residual rows (non-finite entries
/// mark unavailability and are dropped), `norm_var` the target variable of
/// the first stage — the same conventions as [`crate::proxy_svar`].
///
/// # Errors
///
/// * [`IdentError::Dimension`] if `proxy` does not match the residual rows;
/// * [`IdentError::RestrictionOutOfRange`] if `norm_var >= n`;
/// * [`IdentError::NonFinite`] if `u` contains a NaN/infinity;
/// * [`IdentError::InvalidArgument`] if the overlap has fewer than three
///   observations, the proxy has no variance over the overlap, the HAC
///   bandwidth is not below the overlap count, or the requested variance
///   estimate is degenerate (zero) so no F can be formed.
pub fn proxy_first_stage(
    u: MatRef<'_, f64>,
    proxy: &[f64],
    norm_var: usize,
    variance: FirstStageVariance,
) -> Result<FirstStageDiagnostics, IdentError> {
    let t = u.nrows();
    let n = u.ncols();
    if n == 0 || t == 0 {
        return Err(IdentError::InvalidArgument {
            what: "residual matrix u must have at least one row and one column",
        });
    }
    if proxy.len() != t {
        return Err(IdentError::Dimension {
            what: "proxy length must equal the number of residual rows T",
            expected: t,
            got: proxy.len(),
        });
    }
    if norm_var >= n {
        return Err(IdentError::RestrictionOutOfRange {
            what: "norm_var",
            index: norm_var,
            bound: n,
        });
    }
    for i in 0..t {
        if !u[(i, norm_var)].is_finite() {
            return Err(IdentError::NonFinite { what: "u" });
        }
    }

    // Overlap and means, in proxy_svar's operation order so the HC1 numbers
    // agree bit-for-bit.
    let overlap: Vec<usize> = (0..t).filter(|&r| proxy[r].is_finite()).collect();
    let n_proxy = overlap.len();
    if n_proxy < 3 {
        return Err(IdentError::InvalidArgument {
            what: "proxy overlap has fewer than 3 finite observations; cannot run the first stage",
        });
    }
    let no = n_proxy as f64;
    if let FirstStageVariance::HacBartlett { lags } = variance {
        if lags >= n_proxy {
            return Err(IdentError::InvalidArgument {
                what: "HAC bandwidth must be smaller than the proxy overlap count",
            });
        }
    }

    let mut mbar = 0.0;
    for &r in &overlap {
        mbar += proxy[r];
    }
    mbar /= no;
    let mut ybar = 0.0;
    for &r in &overlap {
        ybar += u[(r, norm_var)];
    }
    ybar /= no;

    let mut smm = 0.0;
    let mut smy = 0.0;
    let mut syy = 0.0;
    for &r in &overlap {
        let md = proxy[r] - mbar;
        let yd = u[(r, norm_var)] - ybar;
        smm += md * md;
        smy += md * yd;
        syy += yd * yd;
    }
    if smm == 0.0 {
        return Err(IdentError::InvalidArgument {
            what: "instrument has zero variance over the overlap; no first stage",
        });
    }
    let beta = smy / smm;
    let reliability = if syy > 0.0 {
        smy * smy / (smm * syy)
    } else {
        0.0
    };

    let dof = (n_proxy - 2) as f64;
    let mut sse = 0.0;
    let mut meat = 0.0;
    for &r in &overlap {
        let md = proxy[r] - mbar;
        let e = (u[(r, norm_var)] - ybar) - beta * md;
        sse += e * e;
        meat += md * md * e * e;
    }
    let var_classical = (sse / dof) / smm;
    let var_hc1 = (no / dof) * meat / (smm * smm);

    let (var_used, hac_lags) = match variance {
        FirstStageVariance::Classical => (var_classical, None),
        FirstStageVariance::Hc1 => (var_hc1, None),
        FirstStageVariance::HacBartlett { lags } => {
            // Bartlett HAC on the score s_t = m~_t e_t, paired over calendar
            // time (both dates must be in the overlap). The [2,2] element of
            // the full sandwich reduces exactly to this scalar form by the
            // Frisch-Waugh partialling of the intercept. Same HC1-style
            // degrees-of-freedom correction as `weakivtest` applies.
            let mut present = vec![false; t];
            for &r in &overlap {
                present[r] = true;
            }
            let mut s_hac = meat; // Gamma_0
            for j in 1..=lags {
                let w = 1.0 - (j as f64) / ((lags + 1) as f64);
                let mut gam = 0.0;
                for &r in &overlap {
                    if r < j || !present[r - j] {
                        continue;
                    }
                    let md = proxy[r] - mbar;
                    let e = (u[(r, norm_var)] - ybar) - beta * md;
                    let md_l = proxy[r - j] - mbar;
                    let e_l = (u[(r - j, norm_var)] - ybar) - beta * md_l;
                    gam += (md * e) * (md_l * e_l);
                }
                s_hac += w * 2.0 * gam;
            }
            ((no / dof) * s_hac / (smm * smm), Some(lags))
        }
    };
    if var_used <= 0.0 || !var_used.is_finite() {
        return Err(IdentError::InvalidArgument {
            what: "the first-stage variance estimate is not positive and finite; no effective F \
                   can be formed (a HAC estimate can be driven to zero or below by a bandwidth \
                   close to the sample size)",
        });
    }

    let effective_f = beta * beta / var_used;
    let f_classical = beta * beta / var_classical;
    let f_hc1 = beta * beta / var_hc1;

    let mut cvs = [0.0f64; 4];
    for (slot, &tau) in cvs.iter_mut().zip(MOP_TAUS.iter()) {
        *slot = mop_critical_value(tau, MOP_ALPHA)?;
    }
    let tau_bound = mop_tau_bound(effective_f, MOP_ALPHA)?;

    Ok(FirstStageDiagnostics {
        beta,
        se: var_used.sqrt(),
        effective_f,
        f_classical,
        f_hc1,
        reliability,
        n_proxy,
        hac_lags,
        mop_cv_tau5: cvs[0],
        mop_cv_tau10: cvs[1],
        mop_cv_tau20: cvs[2],
        mop_cv_tau30: cvs[3],
        tau_bound,
        weak_mop_tau10: effective_f <= cvs[1],
        weak_folklore: effective_f < 10.0,
    })
}

/// CDF of the noncentral chi-square with **one** degree of freedom and
/// noncentrality `lambda`: `P(X <= x) = Phi(sqrt(x) - sqrt(lambda)) -
/// Phi(-sqrt(x) - sqrt(lambda))`, the exact closed form from
/// `X = (Z + sqrt(lambda))^2`.
fn ncx2_1_cdf(x: f64, lambda: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    let s = x.sqrt();
    let d = lambda.sqrt();
    // Phi(z) = erfc(-z / sqrt(2)) / 2.
    let phi = |z: f64| 0.5 * erfc(-z / std::f64::consts::SQRT_2);
    phi(s - d) - phi(-s - d)
}

/// The Montiel Olea-Pflueger critical value for the single-instrument
/// effective F: the `1 - alpha` quantile of the noncentral chi-square with
/// one degree of freedom and noncentrality `1 / tau`, where `tau` is the
/// worst-case (Nagar-benchmark) relative bias being tested.
///
/// `mop_critical_value(0.10, 0.05)` is the conventional bar, 23.11. Solved
/// by bisection on the exact df-1 closed-form CDF; accurate to ~1e-12
/// relative, pinned against `scipy.stats.ncx2.ppf` in the golden fixture.
///
/// # Errors
///
/// [`IdentError::InvalidArgument`] if `tau` is not positive and finite or
/// `alpha` is outside `(0, 1)`; [`IdentError::Stats`] only if the internal
/// normal quantile rejects `alpha` (unreachable once the range check
/// passes).
pub fn mop_critical_value(tau: f64, alpha: f64) -> Result<f64, IdentError> {
    if tau <= 0.0 || !tau.is_finite() {
        return Err(IdentError::InvalidArgument {
            what: "tau (the worst-case relative-bias benchmark) must be positive and finite",
        });
    }
    if !(alpha > 0.0 && alpha < 1.0) {
        return Err(IdentError::InvalidArgument {
            what: "alpha must lie strictly inside (0, 1)",
        });
    }
    let lambda = 1.0 / tau;
    let p = 1.0 - alpha;
    // Bracket the quantile: a Gaussian start, then expand.
    let z = inv_norm_cdf(p)?;
    let mut hi = (lambda.sqrt() + z.abs() + 1.0).powi(2) + 1.0;
    while ncx2_1_cdf(hi, lambda) < p {
        hi *= 2.0;
        if !hi.is_finite() {
            return Err(IdentError::NoConvergence {
                what: "noncentral chi-square quantile bracketing",
            });
        }
    }
    let mut lo = 0.0f64;
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if ncx2_1_cdf(mid, lambda) < p {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Ok(0.5 * (lo + hi))
}

/// The smallest worst-case relative bias `tau` that an observed effective F
/// rejects at level `alpha` — the inverse of [`mop_critical_value`] in
/// `tau`.
///
/// Returns `+inf` when `f_eff` is at or below the **central** chi-square
/// quantile (3.84 at `alpha = 0.05`): such an F cannot reject *any* bias
/// bound, not even the hypothesis of zero relevance. Smaller is stronger:
/// Gertler-Karadi's baseline effective F of ~20 gives a bound of ~0.12
/// (12% worst-case bias certified), while an F of 23.11 gives exactly 0.10.
///
/// # Errors
///
/// [`IdentError::InvalidArgument`] if `f_eff` is not finite and nonnegative
/// or `alpha` is outside `(0, 1)`.
pub fn mop_tau_bound(f_eff: f64, alpha: f64) -> Result<f64, IdentError> {
    if !f_eff.is_finite() || f_eff < 0.0 {
        return Err(IdentError::InvalidArgument {
            what: "the effective F must be finite and nonnegative",
        });
    }
    if !(alpha > 0.0 && alpha < 1.0) {
        return Err(IdentError::InvalidArgument {
            what: "alpha must lie strictly inside (0, 1)",
        });
    }
    let p = 1.0 - alpha;
    // cv(lambda) is strictly increasing in lambda; cv(0) is the central
    // chi2_1 quantile. Below that, no lambda > 0 (no tau < inf) is rejected.
    if ncx2_1_cdf(f_eff, 0.0) <= p {
        return Ok(f64::INFINITY);
    }
    // Solve ncx2_cdf(f_eff; 1, lambda) = p for lambda: the CDF is strictly
    // decreasing in lambda at fixed x, so bisection applies directly.
    let mut lo = 0.0f64;
    let mut hi = 1.0f64;
    while ncx2_1_cdf(f_eff, hi) > p {
        hi *= 2.0;
        if !hi.is_finite() {
            return Err(IdentError::NoConvergence {
                what: "MOP tau-bound bracketing",
            });
        }
    }
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if ncx2_1_cdf(f_eff, mid) > p {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Ok(1.0 / (0.5 * (lo + hi)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tsecon_linalg::faer::Mat;

    /// The stamped critical values match the published weakivtest table for
    /// one instrument (37.418, 23.109, 15.062; the tau=30% row differs in
    /// the third decimal because weakivtest rounds x = 1/tau to 3.33).
    #[test]
    fn mop_critical_values_match_published_table() -> Result<(), IdentError> {
        assert!((mop_critical_value(0.05, 0.05)? - 37.418).abs() < 5e-3);
        assert!((mop_critical_value(0.10, 0.05)? - 23.109).abs() < 5e-3);
        assert!((mop_critical_value(0.20, 0.05)? - 15.062).abs() < 5e-3);
        // exact 1/0.3 vs weakivtest's rounded 3.33: 12.046 vs 12.039
        assert!((mop_critical_value(0.30, 0.05)? - 12.046).abs() < 5e-3);
        Ok(())
    }

    /// tau_bound inverts mop_critical_value, and the no-relevance region
    /// returns +inf.
    #[test]
    fn tau_bound_inverts_critical_value() -> Result<(), IdentError> {
        for &tau in &[0.05, 0.10, 0.20, 0.30, 0.5] {
            let cv = mop_critical_value(tau, 0.05)?;
            let back = mop_tau_bound(cv, 0.05)?;
            assert!((back - tau).abs() < 1e-8, "tau {tau} round-trips to {back}");
        }
        assert!(mop_tau_bound(3.0, 0.05)?.is_infinite());
        assert!(mop_tau_bound(0.0, 0.05)?.is_infinite());
        Ok(())
    }

    /// The df-1 noncentral chi-square CDF closed form at lambda = 0 is the
    /// central chi2_1 CDF (checked at its 95% quantile).
    #[test]
    fn ncx2_reduces_to_central_chi2() {
        // chi2_1 95% quantile = 3.8414588...
        let p = ncx2_1_cdf(3.841458820694124, 0.0);
        assert!((p - 0.95).abs() < 1e-12);
    }

    fn toy_uv(strong: bool) -> (Mat<f64>, Vec<f64>) {
        // Deterministic pseudo-noise; the point is only relative strength.
        let t = 400;
        let mut u = Mat::<f64>::zeros(t, 2);
        let mut proxy = vec![0.0f64; t];
        let mut s: u64 = 0x9E3779B97F4A7C15;
        let mut unif = || {
            // xorshift64*; deterministic and dependency-free.
            s ^= s >> 12;
            s ^= s << 25;
            s ^= s >> 27;
            let v = s.wrapping_mul(0x2545F4914F6CDD1D);
            (v >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        for r in 0..t {
            let eps = unif();
            let noise = unif();
            u[(r, 0)] = eps;
            u[(r, 1)] = 0.5 * eps + unif();
            proxy[r] = if strong {
                eps + 0.3 * noise
            } else {
                0.02 * eps + noise
            };
        }
        (u, proxy)
    }

    /// Strong DGP -> large effective F clearing the tau=10% bar; weak DGP
    /// -> small effective F failing it (and usually the folklore bar too).
    #[test]
    fn strength_orders_the_effective_f() -> Result<(), IdentError> {
        let (u_s, m_s) = toy_uv(true);
        let d_s = proxy_first_stage(u_s.as_ref(), &m_s, 0, FirstStageVariance::Hc1)?;
        let (u_w, m_w) = toy_uv(false);
        let d_w = proxy_first_stage(u_w.as_ref(), &m_w, 0, FirstStageVariance::Hc1)?;
        assert!(
            d_s.effective_f > d_s.mop_cv_tau10,
            "strong F {}",
            d_s.effective_f
        );
        assert!(!d_s.weak_mop_tau10);
        assert!(d_w.effective_f < d_s.effective_f);
        assert!(d_w.weak_mop_tau10, "weak F {}", d_w.effective_f);
        assert!(d_s.tau_bound < 0.10);
        assert!(d_w.tau_bound > 0.30);
        Ok(())
    }

    /// The effective F is invariant to rescaling the proxy and the
    /// residuals (separately and jointly).
    #[test]
    fn effective_f_is_scale_invariant() -> Result<(), IdentError> {
        let (u, m) = toy_uv(true);
        let base = proxy_first_stage(u.as_ref(), &m, 0, FirstStageVariance::Hc1)?;
        let m2: Vec<f64> = m.iter().map(|&x| -3.7 * x).collect();
        let d2 = proxy_first_stage(u.as_ref(), &m2, 0, FirstStageVariance::Hc1)?;
        let u3 = Mat::from_fn(u.nrows(), u.ncols(), |i, j| 250.0 * u[(i, j)]);
        let d3 = proxy_first_stage(u3.as_ref(), &m, 0, FirstStageVariance::Hc1)?;
        assert!((base.effective_f - d2.effective_f).abs() < 1e-9 * base.effective_f);
        assert!((base.effective_f - d3.effective_f).abs() < 1e-9 * base.effective_f);
        // HAC path too.
        let hb = proxy_first_stage(
            u.as_ref(),
            &m,
            0,
            FirstStageVariance::HacBartlett { lags: 4 },
        )?;
        let h3 = proxy_first_stage(
            u3.as_ref(),
            &m2,
            0,
            FirstStageVariance::HacBartlett { lags: 4 },
        )?;
        assert!((hb.effective_f - h3.effective_f).abs() < 1e-9 * hb.effective_f);
        Ok(())
    }

    /// HAC with zero lags is HC0-scaled-by-HC1-dof, i.e. exactly the HC1
    /// effective F; and the classical/HC1 fields agree with the dedicated
    /// variances.
    #[test]
    fn hac_zero_lags_equals_hc1() -> Result<(), IdentError> {
        let (u, m) = toy_uv(true);
        let hc1 = proxy_first_stage(u.as_ref(), &m, 0, FirstStageVariance::Hc1)?;
        let hac0 = proxy_first_stage(
            u.as_ref(),
            &m,
            0,
            FirstStageVariance::HacBartlett { lags: 0 },
        )?;
        assert_eq!(hc1.effective_f, hac0.effective_f);
        assert_eq!(hc1.f_hc1, hac0.f_hc1);
        assert_eq!(hc1.effective_f, hc1.f_hc1);
        let cl = proxy_first_stage(u.as_ref(), &m, 0, FirstStageVariance::Classical)?;
        assert_eq!(cl.effective_f, cl.f_classical);
        Ok(())
    }

    /// Bit-for-bit agreement with proxy_svar's first_stage_f on the same
    /// inputs (HC1 and classical), including under a NaN prefix.
    #[test]
    fn agrees_with_proxy_svar_bit_for_bit() -> Result<(), IdentError> {
        let (u, mut m) = toy_uv(true);
        for slot in m.iter_mut().take(37) {
            *slot = f64::NAN;
        }
        let psi = vec![Mat::<f64>::identity(2, 2)];
        let sigma = Mat::<f64>::identity(2, 2);
        let robust = crate::proxy_svar(u.as_ref(), &m, &psi, sigma.as_ref(), 0, 1.0, true)?;
        let classical = crate::proxy_svar(u.as_ref(), &m, &psi, sigma.as_ref(), 0, 1.0, false)?;
        let d = proxy_first_stage(u.as_ref(), &m, 0, FirstStageVariance::Hc1)?;
        assert_eq!(d.f_hc1, robust.first_stage_f);
        assert_eq!(d.effective_f, robust.first_stage_f);
        assert_eq!(d.f_classical, classical.first_stage_f);
        assert_eq!(d.reliability, robust.reliability);
        assert_eq!(d.n_proxy, robust.n_proxy);
        Ok(())
    }

    /// Guards: bandwidth >= overlap, short overlap, zero-variance proxy,
    /// bad tau/alpha.
    #[test]
    fn guards_fire() {
        let (u, m) = toy_uv(true);
        assert!(matches!(
            proxy_first_stage(
                u.as_ref(),
                &m,
                0,
                FirstStageVariance::HacBartlett { lags: 400 }
            ),
            Err(IdentError::InvalidArgument { .. })
        ));
        assert!(matches!(
            proxy_first_stage(
                u.as_ref(),
                &vec![0.0; u.nrows()],
                0,
                FirstStageVariance::Hc1
            ),
            Err(IdentError::InvalidArgument { .. })
        ));
        assert!(matches!(
            proxy_first_stage(u.as_ref(), &m[..10], 0, FirstStageVariance::Hc1),
            Err(IdentError::Dimension { .. })
        ));
        assert!(matches!(
            proxy_first_stage(u.as_ref(), &m, 7, FirstStageVariance::Hc1),
            Err(IdentError::RestrictionOutOfRange { .. })
        ));
        assert!(mop_critical_value(0.0, 0.05).is_err());
        assert!(mop_critical_value(0.1, 1.0).is_err());
        assert!(mop_tau_bound(f64::NAN, 0.05).is_err());
    }
}
