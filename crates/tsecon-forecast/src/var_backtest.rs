//! The VaR backtest battery: Kupiec (1995) unconditional coverage,
//! Christoffersen (1998) independence and conditional coverage, and the
//! Engle-Manganelli (2004) dynamic quantile (DQ) regression test.
//!
//! A VaR forecast at level `alpha` claims `P(return_t < VaR_t | past) =
//! alpha` every period. Backtesting turns the realized *hit sequence*
//! `hit_t = 1{return_t < VaR_t}` into evidence about that claim along two
//! separate axes:
//!
//! * **Unconditional coverage** — is the violation *rate* right? Kupiec's
//!   proportion-of-failures likelihood ratio compares the Bernoulli
//!   likelihood at the nominal `alpha` with the likelihood at the observed
//!   rate `pi_hat = n1/n`:
//!
//!   ```text
//!   LR_uc = -2 ln[ (1-alpha)^n0 alpha^n1 / ((1-pi_hat)^n0 pi_hat^n1) ]
//!         ~ chi^2(1)
//!   ```
//!
//! * **Independence** — are the violations *clustered*? A correct VaR
//!   model's hits are unforecastable, so yesterday's hit must not predict
//!   today's. Christoffersen's LR_ind tests the first-order Markov
//!   alternative with transition counts `n_ij` (state `i` at `t-1` to state
//!   `j` at `t`, over the `n-1` transition pairs) and transition
//!   probabilities `pi01 = n01/(n00+n01)`, `pi11 = n11/(n10+n11)` against
//!   the iid null with the pooled rate `pi2 = (n01+n11)/(n-1)`:
//!
//!   ```text
//!   LR_ind = -2 [ ll(pi2) - ll(pi01, pi11) ] ~ chi^2(1),
//!   LR_cc  = LR_uc + LR_ind               ~ chi^2(2)
//!   ```
//!
//!   with the conventional `0 * ln(0) = 0` continuity treatment for empty
//!   cells — in particular `n11 = 0` (violations never consecutive), the
//!   *common* case at small `alpha`, is handled by continuity, not an
//!   error. `LR_cc` is reported as the sum, the standard practice
//!   (Christoffersen 1998, ignoring the first-observation conditioning).
//!
//! * **Dynamic quantile** — is *anything in the information set*
//!   forecasting the hits? Engle-Manganelli regress the demeaned hits
//!   `Hit_t = hit_t - alpha` on a constant, `dq_lags` lagged demeaned
//!   hits, and the contemporaneous VaR forecast, over `t = dq_lags..n`.
//!   Under the null every coefficient is zero and, because `Hit_t` is a
//!   centered Bernoulli with variance `alpha (1-alpha)`,
//!
//!   ```text
//!   DQ = Hit' X (X'X)^{-1} X' Hit / (alpha (1-alpha)) ~ chi^2(k)
//!   ```
//!
//!   with `k` the **rank** of the design (`dq_lags + 2` when the VaR
//!   forecast is available, `dq_lags + 1` from a bare hit sequence, and
//!   one less when a supplied VaR path is constant over the window —
//!   an unconditional VaR — and therefore collinear with the intercept:
//!   the projection is unchanged, the degrees of freedom honestly
//!   shrink, and the verdict says so). This nests coverage *and*
//!   dependence *and* whether risk is priced into the forecast itself —
//!   the strictest of the three.
//!
//! # The sign convention (fixed once, per ROADMAP Module 03)
//!
//! `returns` and `var_forecasts` live on the **same scale — the return
//! (P&L) scale** — and `var_forecasts[t]` is the model's `alpha`-quantile
//! of the conditional return distribution, so for small `alpha` it is
//! typically a **negative** number. A violation is `return_t <
//! var_forecasts[t]` (strict; a return exactly on the VaR boundary is not
//! a violation). Working in positive-loss space instead? Negate both
//! series before calling. This matches the hit definition in
//! Engle & Manganelli (2004) and the `rugarch`/GAS `VaRTest` convention.
//!
//! `alpha` is the coverage level of the VaR being tested — the *expected
//! violation probability* (0.05 for a 95% VaR, 0.01 for a 99% VaR) — not
//! a test significance level.
//!
//! # Small samples
//!
//! With `T = 250` observations at `alpha = 0.01` only ~2.5 violations are
//! expected, and the chi-squared asymptotics of all three tests are
//! unreliable (Module 03 flags this; Dufour 2006 Monte Carlo p-values are
//! the planned fix). The [`VarBacktestResult::verdict`] warns when fewer
//! than five violations were expected.
//!
//! # References
//!
//! Kupiec (1995), "Techniques for Verifying the Accuracy of Risk
//! Measurement Models", *J. Derivatives* 3(2). Christoffersen (1998),
//! "Evaluating Interval Forecasts", *IER* 39(4). Engle & Manganelli
//! (2004), "CAViaR", *JBES* 22(4). Worked example pinned in the goldens:
//! J.P. Morgan's 1998 disclosure of 20 95%-VaR breaches in 252 trading
//! days (Jorion, *Value at Risk*, ch. 6), for which `LR_uc = 3.913` —
//! a borderline rejection against the 3.84 critical value.

use crate::error::ForecastError;
use crate::validate::check_finite;
use tsecon_stats::chi2_sf;

/// Result of the VaR backtest battery: Kupiec unconditional coverage,
/// Christoffersen independence / conditional coverage, and the
/// Engle-Manganelli dynamic quantile test, plus a teaching verdict.
#[derive(Debug, Clone, PartialEq)]
pub struct VarBacktestResult {
    /// Number of observations in the hit sequence.
    pub n: usize,
    /// The VaR coverage level being tested (expected violation rate).
    pub alpha: f64,
    /// Observed number of violations `n1`.
    pub n_violations: usize,
    /// Expected number of violations `alpha * n`.
    pub expected_violations: f64,
    /// Observed violation rate `pi_hat = n1 / n`.
    pub hit_rate: f64,
    /// Kupiec (1995) proportion-of-failures LR statistic, `chi^2(1)`.
    pub lr_uc: f64,
    /// Upper-tail `chi^2(1)` p-value of `lr_uc`.
    pub p_uc: f64,
    /// Transition count: no violation at `t-1`, no violation at `t`.
    pub n00: usize,
    /// Transition count: no violation at `t-1`, violation at `t`.
    pub n01: usize,
    /// Transition count: violation at `t-1`, no violation at `t`.
    pub n10: usize,
    /// Transition count: violation at `t-1`, violation at `t`.
    pub n11: usize,
    /// Estimated `P(violation_t | no violation_{t-1}) = n01/(n00+n01)`
    /// (0 when there are no transitions from the no-violation state).
    pub pi01: f64,
    /// Estimated `P(violation_t | violation_{t-1}) = n11/(n10+n11)`
    /// (0 when there are no transitions from the violation state).
    pub pi11: f64,
    /// Christoffersen (1998) independence LR statistic, `chi^2(1)`.
    pub lr_ind: f64,
    /// Upper-tail `chi^2(1)` p-value of `lr_ind`.
    pub p_ind: f64,
    /// Conditional-coverage LR statistic `lr_uc + lr_ind`, `chi^2(2)`.
    pub lr_cc: f64,
    /// Upper-tail `chi^2(2)` p-value of `lr_cc`.
    pub p_cc: f64,
    /// Engle-Manganelli (2004) dynamic quantile statistic, `chi^2(dq_df)`.
    pub dq_stat: f64,
    /// Upper-tail `chi^2(dq_df)` p-value of `dq_stat`.
    pub p_dq: f64,
    /// Number of lagged hits in the DQ regression.
    pub dq_lags: usize,
    /// DQ degrees of freedom — the *rank* of the DQ design: constant +
    /// `dq_lags` lagged hits (+ the VaR forecast when it entered).
    pub dq_df: usize,
    /// Whether the contemporaneous VaR forecast entered the DQ regression
    /// (`false` when backtesting a bare hit sequence, or when the supplied
    /// VaR path was dropped as collinear — see
    /// [`VarBacktestResult::dq_var_dropped`]).
    pub dq_includes_var: bool,
    /// `true` when VaR forecasts were supplied but dropped from the DQ
    /// regression because they are (numerically) in the span of the
    /// constant and lagged hits — a constant VaR path (an unconditional
    /// VaR model) is the common case. The projection is unchanged by the
    /// drop; only the chi-squared degrees of freedom honestly shrink.
    pub dq_var_dropped: bool,
    /// Plain-language verdict over all three tests, in the library's
    /// errors-that-teach style.
    pub verdict: String,
}

/// Backtest a VaR forecast path against realized returns.
///
/// * `returns` — realized returns (or P&L) per period.
/// * `var_forecasts` — the one-step-ahead VaR forecasts, index-aligned
///   with `returns`, on the **same (return) scale**: `var_forecasts[t]` is
///   the model's `alpha`-quantile of the conditional return distribution
///   (typically negative for small `alpha`). See the
///   [module docs](self) for the sign convention.
/// * `alpha` — the VaR coverage level (expected violation rate; 0.05 for
///   a 95% VaR).
/// * `dq_lags` — lagged hits in the DQ regression (Engle-Manganelli use 4).
///
/// Computes `hit_t = 1{returns[t] < var_forecasts[t]}` and runs the full
/// battery; the DQ regression includes the contemporaneous VaR forecast.
///
/// # Errors
///
/// [`ForecastError::LengthMismatch`], [`ForecastError::NonFinite`],
/// [`ForecastError::InvalidAlpha`] (`alpha` outside (0, 1)),
/// [`ForecastError::InvalidDqLags`], [`ForecastError::NoViolations`] /
/// [`ForecastError::AllViolations`] (degenerate hit sequences),
/// [`ForecastError::SingularDqDesign`], and wrapped
/// [`ForecastError::Stats`] from the chi-squared p-values.
pub fn var_backtest(
    returns: &[f64],
    var_forecasts: &[f64],
    alpha: f64,
    dq_lags: usize,
) -> Result<VarBacktestResult, ForecastError> {
    const WHAT: &str = "VaR backtest";
    if returns.len() != var_forecasts.len() {
        return Err(ForecastError::LengthMismatch {
            what: WHAT,
            expected: returns.len(),
            actual: var_forecasts.len(),
        });
    }
    check_finite(returns, WHAT)?;
    check_finite(var_forecasts, WHAT)?;
    let hits: Vec<f64> = returns
        .iter()
        .zip(var_forecasts.iter())
        .map(|(&r, &q)| if r < q { 1.0 } else { 0.0 })
        .collect();
    backtest_battery(&hits, Some(var_forecasts), alpha, dq_lags)
}

/// Backtest a pre-computed 0/1 violation ("hit") sequence.
///
/// * `hits` — the violation indicators: exactly `1.0` where the return
///   fell below its VaR forecast, `0.0` elsewhere (any other value is a
///   teaching error).
/// * `var_forecasts` — optionally the aligned VaR forecasts, which lets
///   the DQ regression include the contemporaneous VaR regressor exactly
///   as Engle & Manganelli specify; without them the DQ test runs on the
///   constant and lagged hits only (`dq_df = dq_lags + 1`), which is
///   still a valid — just less strict — DQ variant.
/// * `alpha`, `dq_lags` — as in [`var_backtest`].
///
/// # Errors
///
/// As [`var_backtest`], plus [`ForecastError::InvalidHitValue`] when the
/// hit series contains anything other than 0 and 1.
pub fn var_backtest_hits(
    hits: &[f64],
    var_forecasts: Option<&[f64]>,
    alpha: f64,
    dq_lags: usize,
) -> Result<VarBacktestResult, ForecastError> {
    const WHAT: &str = "VaR backtest";
    for (index, &value) in hits.iter().enumerate() {
        if !value.is_finite() {
            return Err(ForecastError::NonFinite {
                what: WHAT,
                index,
                value,
            });
        }
        if value != 0.0 && value != 1.0 {
            return Err(ForecastError::InvalidHitValue { index, value });
        }
    }
    if let Some(var) = var_forecasts {
        if var.len() != hits.len() {
            return Err(ForecastError::LengthMismatch {
                what: WHAT,
                expected: hits.len(),
                actual: var.len(),
            });
        }
        check_finite(var, WHAT)?;
    }
    backtest_battery(hits, var_forecasts, alpha, dq_lags)
}

/// `count * ln(p)` with the continuity convention `0 * ln(0) = 0`.
fn xlogy(count: usize, p: f64) -> f64 {
    if count == 0 {
        0.0
    } else {
        count as f64 * p.ln()
    }
}

/// The shared battery on a validated hit sequence.
fn backtest_battery(
    hits: &[f64],
    var_forecasts: Option<&[f64]>,
    alpha: f64,
    dq_lags: usize,
) -> Result<VarBacktestResult, ForecastError> {
    if !(alpha > 0.0 && alpha < 1.0) {
        return Err(ForecastError::InvalidAlpha { value: alpha });
    }
    let n = hits.len();
    // Enough observations for the DQ regression: dq_lags presample rows
    // plus at least k + 1 usable rows, k = constant + lags (+ VaR).
    let k = 1 + dq_lags + usize::from(var_forecasts.is_some());
    let needed = dq_lags + k + 1;
    if dq_lags == 0 || n < needed {
        return Err(ForecastError::InvalidDqLags {
            lags: dq_lags,
            n,
            needed,
        });
    }

    let n1 = hits.iter().filter(|&&h| h == 1.0).count();
    let n0 = n - n1;
    if n1 == 0 {
        return Err(ForecastError::NoViolations { n, alpha });
    }
    if n0 == 0 {
        return Err(ForecastError::AllViolations { n, alpha });
    }

    // ---- Kupiec (1995) proportion-of-failures LR ---------------------
    let pi_hat = n1 as f64 / n as f64;
    let ll_null = xlogy(n0, 1.0 - alpha) + xlogy(n1, alpha);
    let ll_alt = xlogy(n0, 1.0 - pi_hat) + xlogy(n1, pi_hat);
    let lr_uc = (-2.0 * (ll_null - ll_alt)).max(0.0);
    let p_uc = chi2_sf(lr_uc, 1.0)?;

    // ---- Christoffersen (1998) first-order Markov independence -------
    let (mut n00, mut n01, mut n10, mut n11) = (0usize, 0usize, 0usize, 0usize);
    for t in 1..n {
        match (hits[t - 1] == 1.0, hits[t] == 1.0) {
            (false, false) => n00 += 1,
            (false, true) => n01 += 1,
            (true, false) => n10 += 1,
            (true, true) => n11 += 1,
        }
    }
    let from0 = n00 + n01;
    let from1 = n10 + n11;
    let pi01 = if from0 > 0 {
        n01 as f64 / from0 as f64
    } else {
        0.0
    };
    let pi11 = if from1 > 0 {
        n11 as f64 / from1 as f64
    } else {
        0.0
    };
    let pi2 = (n01 + n11) as f64 / (n - 1) as f64;
    let ll0 = xlogy(n00 + n10, 1.0 - pi2) + xlogy(n01 + n11, pi2);
    let ll1 = xlogy(n00, 1.0 - pi01) + xlogy(n01, pi01) + xlogy(n10, 1.0 - pi11) + xlogy(n11, pi11);
    let lr_ind = (-2.0 * (ll0 - ll1)).max(0.0);
    let p_ind = chi2_sf(lr_ind, 1.0)?;
    let lr_cc = lr_uc + lr_ind;
    let p_cc = chi2_sf(lr_cc, 2.0)?;

    // ---- Engle-Manganelli (2004) dynamic quantile --------------------
    // Regress Hit_t = hit_t - alpha on x_t = [1, Hit_{t-1..t-L}, VaR_t]
    // over t = L..n-1 and form DQ = Hit'X (X'X)^- X'Hit / (alpha(1-alpha)),
    // computed as the squared norm of the projection of the Hit vector
    // onto the column space of X (modified Gram-Schmidt). Rank matters:
    // the chi-squared degrees of freedom must equal the rank of the
    // design, not its nominal column count. A VaR column inside the span
    // of the constant and lagged hits — a *constant* VaR path, i.e. an
    // unconditional VaR model, is the common case — is dropped with the
    // df reduced and the verdict saying so. A *lagged-hit* column inside
    // the span of its predecessors instead means the sample has too few
    // violations to identify the lag coefficients at all, which is a
    // teaching error, not a silent df reduction.
    let h: Vec<f64> = hits.iter().map(|&v| v - alpha).collect();
    let n_dq = n - dq_lags;
    let mut cols: Vec<Vec<f64>> = Vec::with_capacity(k);
    cols.push(vec![1.0; n_dq]);
    for j in 1..=dq_lags {
        cols.push((dq_lags..n).map(|t| h[t - j]).collect());
    }
    if let Some(var) = var_forecasts {
        cols.push(var[dq_lags..].to_vec());
    }
    let y = &h[dq_lags..];

    let dot = |a: &[f64], b: &[f64]| a.iter().zip(b.iter()).map(|(x, z)| x * z).sum::<f64>();
    // Modified Gram-Schmidt with rank-revealing column dropping: a column
    // whose out-of-span component is below DROP_TOL of its own norm is
    // numerically inside the span of the columns before it.
    const DROP_TOL: f64 = 1e-8;
    let mut basis: Vec<Vec<f64>> = Vec::with_capacity(k);
    let mut var_dropped = false;
    for (idx, col) in cols.iter().enumerate() {
        let norm0 = dot(col, col).sqrt();
        let mut v = col.clone();
        for q in &basis {
            let proj = dot(q, &v);
            for (vi, qi) in v.iter_mut().zip(q.iter()) {
                *vi -= proj * qi;
            }
        }
        let norm = dot(&v, &v).sqrt();
        if norm <= DROP_TOL * norm0.max(f64::MIN_POSITIVE) {
            if var_forecasts.is_some() && idx == k - 1 {
                var_dropped = true;
            } else {
                return Err(ForecastError::SingularDqDesign {
                    k,
                    n_violations: n1,
                });
            }
        } else {
            for vi in v.iter_mut() {
                *vi /= norm;
            }
            basis.push(v);
        }
    }
    let dq_df = basis.len();
    let quad: f64 = basis
        .iter()
        .map(|q| {
            let c = dot(q, y);
            c * c
        })
        .sum();
    let dq_stat = quad / (alpha * (1.0 - alpha));
    let p_dq = chi2_sf(dq_stat, dq_df as f64)?;
    let dq_includes_var = var_forecasts.is_some() && !var_dropped;

    let expected_violations = alpha * n as f64;
    let verdict = build_verdict(
        n,
        alpha,
        n1,
        expected_violations,
        pi_hat,
        lr_uc,
        p_uc,
        pi11,
        lr_ind,
        p_ind,
        lr_cc,
        p_cc,
        dq_stat,
        p_dq,
        dq_df,
        dq_includes_var,
        var_dropped,
    );

    Ok(VarBacktestResult {
        n,
        alpha,
        n_violations: n1,
        expected_violations,
        hit_rate: pi_hat,
        lr_uc,
        p_uc,
        n00,
        n01,
        n10,
        n11,
        pi01,
        pi11,
        lr_ind,
        p_ind,
        lr_cc,
        p_cc,
        dq_stat,
        p_dq,
        dq_lags,
        dq_df,
        dq_includes_var,
        dq_var_dropped: var_dropped,
        verdict,
    })
}

/// The teaching verdict, judged at the conventional 5% test size.
#[allow(clippy::too_many_arguments)]
fn build_verdict(
    n: usize,
    alpha: f64,
    n1: usize,
    expected: f64,
    pi_hat: f64,
    lr_uc: f64,
    p_uc: f64,
    pi11: f64,
    lr_ind: f64,
    p_ind: f64,
    lr_cc: f64,
    p_cc: f64,
    dq_stat: f64,
    p_dq: f64,
    dq_df: usize,
    with_var: bool,
    var_dropped: bool,
) -> String {
    const SIZE: f64 = 0.05;
    let mut s = format!(
        "{n1} violations in {n} observations where {expected:.1} were \
         expected at the {:.0}% VaR (rate {:.3} vs nominal {alpha}).",
        100.0 * (1.0 - alpha),
        pi_hat,
    );
    if p_uc < SIZE {
        let direction = if pi_hat > alpha {
            "the VaR is too aggressive (understates risk)"
        } else {
            "the VaR is too conservative (overstates risk)"
        };
        s.push_str(&format!(
            " Reject unconditional coverage at 5% (Kupiec LR_uc = {lr_uc:.3}, \
             p = {p_uc:.3}): {direction}."
        ));
    } else {
        s.push_str(&format!(
            " No rejection of unconditional coverage (Kupiec LR_uc = \
             {lr_uc:.3}, p = {p_uc:.3}): the violation rate is consistent \
             with alpha."
        ));
    }
    if p_ind < SIZE {
        s.push_str(&format!(
            " Reject independence at 5% (Christoffersen LR_ind = {lr_ind:.3}, \
             p = {p_ind:.3}): violations cluster — P(violation | violation \
             yesterday) = {pi11:.3} vs {alpha} under independence — so the \
             model is too slow to update after a breach even if the overall \
             rate is right."
        ));
    } else {
        s.push_str(&format!(
            " No rejection of independence (Christoffersen LR_ind = \
             {lr_ind:.3}, p = {p_ind:.3})."
        ));
    }
    s.push_str(&format!(
        " Conditional coverage (LR_cc = {lr_cc:.3}, chi-squared(2), p = \
         {p_cc:.3}) {}.",
        if p_cc < SIZE {
            "rejects both properties jointly at 5%"
        } else {
            "does not reject jointly"
        }
    ));
    let dq_note = if with_var {
        "lagged hits + the VaR forecast"
    } else if var_dropped {
        "lagged hits only — the supplied VaR path is constant (collinear \
         with the intercept) over the evaluation window, so it adds \
         nothing to the regression and the degrees of freedom shrink \
         accordingly"
    } else {
        "lagged hits only (no VaR forecasts supplied)"
    };
    if p_dq < SIZE {
        s.push_str(&format!(
            " Reject the dynamic quantile test at 5% (DQ = {dq_stat:.3}, \
             chi-squared({dq_df}) on {dq_note}, p = {p_dq:.3}): something in \
             the information set still forecasts the violations."
        ));
    } else {
        s.push_str(&format!(
            " No rejection from the dynamic quantile test (DQ = {dq_stat:.3}, \
             chi-squared({dq_df}) on {dq_note}, p = {p_dq:.3})."
        ));
    }
    if expected < 5.0 {
        s.push_str(&format!(
            " Caution: only {expected:.1} violations were expected, and with \
             so few the chi-squared asymptotics of all three tests are \
             unreliable (Kupiec 1995); prefer a longer window or a larger \
             alpha."
        ));
    }
    if alpha < 0.5 && pi_hat > 0.5 {
        s.push_str(
            " WARNING: more than half the observations are violations — \
             almost certainly a sign-convention slip. The convention is \
             returns and VaR quantiles on the return scale (violation = \
             return < VaR); negate both series if you work in positive-loss \
             space.",
        );
    }
    s
}
