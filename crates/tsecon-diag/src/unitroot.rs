//! Unit-root and stationarity testing: the ADF test (Said-Dickey), the
//! KPSS test (Kwiatkowski-Phillips-Schmidt-Shin), and the joint
//! ADF + KPSS confirmatory decision workflow.
//!
//! Conventions follow statsmodels 0.14.6 (`adfuller`, `kpss`) exactly —
//! lag selection, trimming, information criteria, bandwidth rules, and
//! p-value interpolation — so results are pinned against its fixtures.

use core::fmt;

use crate::error::DiagError;
use crate::mackinnon::{mackinnon_crit, mackinnon_p, AdfCriticalValues};
use crate::ols::ols_detailed;
use crate::report::check_alpha;
use crate::validate::check_series;

/// Deterministic terms included in the ADF test regression.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdfRegression {
    /// No deterministic terms (statsmodels `"n"`). Only appropriate when
    /// the series has zero mean under the null.
    NoConstant,
    /// Constant only (statsmodels `"c"`, the conventional default).
    Constant,
    /// Constant and linear trend (statsmodels `"ct"`), for series that may
    /// be stationary around a deterministic trend.
    ConstantTrend,
}

impl AdfRegression {
    /// Number of deterministic columns.
    fn ntrend(self) -> usize {
        match self {
            AdfRegression::NoConstant => 0,
            AdfRegression::Constant => 1,
            AdfRegression::ConstantTrend => 2,
        }
    }

    /// The statsmodels code for this specification.
    fn code(self) -> &'static str {
        match self {
            AdfRegression::NoConstant => "n",
            AdfRegression::Constant => "c",
            AdfRegression::ConstantTrend => "ct",
        }
    }
}

/// How the number of lagged differences in the ADF regression is chosen.
///
/// The automatic variants search `0..=maxlag`; `None` uses the Schwert
/// (1989) rule `maxlag = ceil(12 (n/100)^{1/4})`, capped at
/// `n/2 - ntrend - 1` (the statsmodels default).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdfLagSelection {
    /// Use exactly this many lagged differences (statsmodels
    /// `autolag=None` with `maxlag` set).
    Fixed(usize),
    /// Minimize the Akaike information criterion (statsmodels `"AIC"`,
    /// the default).
    Aic(Option<usize>),
    /// Minimize the Bayesian information criterion (statsmodels `"BIC"`).
    Bic(Option<usize>),
    /// statsmodels `"t-stat"`: starting from `maxlag`, pick the largest
    /// lag whose highest-lag coefficient is significant at the 5% level of
    /// a two-sided normal test (`|t| >= 1.6448536269514722`, the one-sided
    /// 95% normal quantile as hard-coded by statsmodels); 0 if none is.
    TStat(Option<usize>),
}

/// Result of the augmented Dickey-Fuller unit-root test.
///
/// The null hypothesis is a unit root (the coefficient on the lagged
/// level equals zero in the differenced regression); the alternative is
/// stationarity. Small p-values therefore speak *for* stationarity.
#[derive(Debug, Clone, PartialEq)]
pub struct AdfResult {
    /// The tau statistic: the OLS t-ratio on the lagged level.
    pub statistic: f64,
    /// MacKinnon (1994) approximate asymptotic p-value.
    pub p_value: f64,
    /// The number of lagged differences used in the final regression.
    pub used_lag: usize,
    /// Effective observations in the final regression
    /// (`n - 1 - used_lag`).
    pub nobs: usize,
    /// MacKinnon (2010) finite-sample critical values at `nobs`.
    pub crit: AdfCriticalValues,
    /// The deterministic specification that was tested.
    pub regression: AdfRegression,
}

/// statsmodels' hard-coded `norm.ppf(0.95)` threshold for the `"t-stat"`
/// lag selection rule.
const ADF_TSTAT_STOP: f64 = 1.6448536269514722;

/// Build the ADF design for `nlags` lagged differences on the trimmed
/// sample: response `dy_t` for `t = nlags+1 .. n-1` (0-indexed levels) and
/// columns `[level y_{t-1}, dy_{t-1}, .., dy_{t-nlags}]` with the
/// deterministics (constant, then trend `1..rows`) prepended or appended.
fn adf_design(
    y: &[f64],
    nlags: usize,
    ntrend: usize,
    prepend: bool,
) -> (Vec<Vec<f64>>, Vec<f64>) {
    let n = y.len();
    let rows = n - 1 - nlags;
    let t0 = n - rows;
    let mut cols: Vec<Vec<f64>> = Vec::with_capacity(ntrend + 1 + nlags);
    let push_deterministics = |cols: &mut Vec<Vec<f64>>| {
        if ntrend >= 1 {
            cols.push(vec![1.0; rows]);
        }
        if ntrend >= 2 {
            cols.push((1..=rows).map(|i| i as f64).collect());
        }
    };
    if prepend {
        push_deterministics(&mut cols);
    }
    cols.push((t0..n).map(|t| y[t - 1]).collect());
    for j in 1..=nlags {
        cols.push((t0..n).map(|t| y[t - j] - y[t - j - 1]).collect());
    }
    if !prepend {
        push_deterministics(&mut cols);
    }
    let dy = (t0..n).map(|t| y[t] - y[t - 1]).collect();
    (cols, dy)
}

/// Augmented Dickey-Fuller unit-root test (Dickey & Fuller 1979; Said &
/// Dickey 1984), matching statsmodels `adfuller`.
///
/// The test regression on the `nobs = n - 1 - p` usable rows is
///
/// ```text
/// dy_t = (deterministics) + gamma y_{t-1} + sum_{j=1..p} b_j dy_{t-j} + e_t
/// tau  = gamma_hat / se(gamma_hat)      (OLS, nonrobust SEs)
/// ```
///
/// with `H0: gamma = 0` (unit root) against `H1: gamma < 0` (stationary).
/// Automatic lag selection fits every candidate on the common sample
/// trimmed at `maxlag` (so information criteria are comparable), then
/// refits at the chosen lag on the longest available sample. P-values are
/// MacKinnon (1994) response surfaces and critical values are MacKinnon
/// (2010) finite-sample surfaces — never the raw Dickey-Fuller tables.
///
/// # Errors
///
/// * [`DiagError::NonFinite`] if the series contains NaN or infinities.
/// * [`DiagError::ConstantSeries`] if the series is constant.
/// * [`DiagError::SeriesTooShort`] if too few observations remain after
///   differencing and trimming for the requested specification.
/// * [`DiagError::InvalidLags`] if a requested lag order exceeds the
///   statsmodels bound `n/2 - ntrend - 1`.
/// * [`DiagError::SingularDesign`] /
///   [`DiagError::NumericalBreakdown`] for (near-)deterministic series
///   whose lag design is collinear or fits exactly.
pub fn adf(
    y: &[f64],
    regression: AdfRegression,
    lags: AdfLagSelection,
) -> Result<AdfResult, DiagError> {
    let ntrend = regression.ntrend();
    let n = check_series(y, 2 * (ntrend + 1), "adf")?;
    if y.iter().all(|&v| v == y[0]) {
        return Err(DiagError::ConstantSeries { what: "adf" });
    }

    // statsmodels bound: maxlag <= n//2 - ntrend - 1.
    let max_allowed = (n / 2 - ntrend - 1) as u64;
    let check_maxlag = |m: usize| -> Result<(), DiagError> {
        if m as u64 > max_allowed {
            return Err(DiagError::InvalidLags {
                what: "adf",
                nlags: m,
                n,
                requirement: "maxlag <= n/2 - ntrend - 1, with ntrend the \
                              number of deterministic terms (statsmodels bound)",
            });
        }
        Ok(())
    };
    let resolve_maxlag = |user: Option<usize>| -> Result<usize, DiagError> {
        match user {
            Some(m) => {
                check_maxlag(m)?;
                Ok(m)
            }
            None => {
                // Schwert (1989) rule, as in statsmodels: ceil, then cap.
                let schwert = (12.0 * (n as f64 / 100.0).powf(0.25)).ceil() as u64;
                Ok(schwert.min(max_allowed) as usize)
            }
        }
    };

    let used_lag = match lags {
        AdfLagSelection::Fixed(l) => {
            check_maxlag(l)?;
            l
        }
        AdfLagSelection::Aic(user) | AdfLagSelection::Bic(user) => {
            let maxlag = resolve_maxlag(user)?;
            let rows = n - 1 - maxlag;
            let k_max = ntrend + 1 + maxlag;
            if rows < k_max + 1 {
                return Err(DiagError::SeriesTooShort {
                    what: "adf",
                    n,
                    needed: 2 * maxlag + ntrend + 3,
                });
            }
            let (cols, dy) = adf_design(y, maxlag, ntrend, true);
            let startlag = ntrend + 1;
            let mut best_lag = 0usize;
            let mut best_ic = f64::INFINITY;
            for lag in 0..=maxlag {
                let fit = ols_detailed(&cols[..startlag + lag], &dy, "adf")?;
                let ic = match lags {
                    AdfLagSelection::Bic(_) => fit.bic(),
                    _ => fit.aic(),
                };
                if ic < best_ic {
                    best_ic = ic;
                    best_lag = lag;
                }
            }
            best_lag
        }
        AdfLagSelection::TStat(user) => {
            let maxlag = resolve_maxlag(user)?;
            let rows = n - 1 - maxlag;
            let k_max = ntrend + 1 + maxlag;
            if rows < k_max + 1 {
                return Err(DiagError::SeriesTooShort {
                    what: "adf",
                    n,
                    needed: 2 * maxlag + ntrend + 3,
                });
            }
            let (cols, dy) = adf_design(y, maxlag, ntrend, true);
            let startlag = ntrend + 1;
            let mut best_lag = 0usize;
            for lag in (1..=maxlag).rev() {
                let fit = ols_detailed(&cols[..startlag + lag], &dy, "adf")?;
                let t_last = fit.t_values[startlag + lag - 1];
                if t_last.abs() >= ADF_TSTAT_STOP {
                    best_lag = lag;
                    break;
                }
            }
            best_lag
        }
    };

    // Final regression on the longest sample for the chosen lag, with the
    // deterministics appended (statsmodels column order: level first).
    let rows = n - 1 - used_lag;
    let k = ntrend + 1 + used_lag;
    if rows < k + 1 {
        return Err(DiagError::SeriesTooShort {
            what: "adf",
            n,
            needed: 2 * used_lag + ntrend + 3,
        });
    }
    let (cols, dy) = adf_design(y, used_lag, ntrend, false);
    let fit = ols_detailed(&cols, &dy, "adf")?;
    let statistic = fit.t_values[0];
    Ok(AdfResult {
        statistic,
        p_value: mackinnon_p(statistic, regression),
        used_lag,
        nobs: rows,
        crit: mackinnon_crit(regression, Some(rows)),
        regression,
    })
}

// ------------------------------------------------------------------ KPSS

/// Deterministic component removed before the KPSS test.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KpssRegression {
    /// Level stationarity: the series is demeaned (statsmodels `"c"`).
    Constant,
    /// Trend stationarity: a constant plus linear trend is removed by OLS
    /// (statsmodels `"ct"`).
    ConstantTrend,
}

impl KpssRegression {
    /// The statsmodels code for this specification.
    fn code(self) -> &'static str {
        match self {
            KpssRegression::Constant => "c",
            KpssRegression::ConstantTrend => "ct",
        }
    }
}

/// How the Bartlett long-run-variance bandwidth of the KPSS test is
/// chosen.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KpssLags {
    /// The statsmodels legacy Schwert-style rule
    /// `ceil(12 (n/100)^{1/4})`, capped at `n - 1`.
    Legacy,
    /// The Hobijn-Franses-Ooms (1998) automatic bandwidth (statsmodels
    /// `"auto"`, its default), capped at `n - 1`.
    Auto,
    /// A fixed bandwidth; must be `< n`.
    Fixed(usize),
}

/// Interpolation table for the KPSS null distribution (Kwiatkowski et al.
/// 1992, table 1): upper-tail critical values at 10% / 5% / 2.5% / 1%.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KpssCriticalValues {
    /// The 10% critical value.
    pub pct10: f64,
    /// The 5% critical value.
    pub pct5: f64,
    /// The 2.5% critical value.
    pub pct2_5: f64,
    /// The 1% critical value.
    pub pct1: f64,
}

/// Result of the KPSS stationarity test.
///
/// The null hypothesis is (level or trend) stationarity; the alternative
/// is a unit root — the reverse of the ADF orientation. Small p-values
/// speak *against* stationarity.
#[derive(Debug, Clone, PartialEq)]
pub struct KpssResult {
    /// The LM statistic `sum_t S_t^2 / (n^2 s^2(l))` (eq. 11 of KPSS 1992).
    pub statistic: f64,
    /// P-value by linear interpolation in the KPSS critical-value table,
    /// bounded to `[0.01, 0.10]` (the statsmodels convention: a value at a
    /// bound means the true p-value is at least/at most the bound).
    pub p_value: f64,
    /// The Bartlett bandwidth actually used.
    pub lags: usize,
    /// Number of observations.
    pub nobs: usize,
    /// The critical values the p-value was interpolated in.
    pub crit: KpssCriticalValues,
    /// The deterministic specification that was tested.
    pub regression: KpssRegression,
}

/// Bartlett-kernel long-run variance `s^2(l)` of eq. 10 in Kwiatkowski et
/// al. (1992):
///
/// ```text
/// s^2(l) = (1/n) sum e_t^2 + (2/n) sum_{i=1..l} (1 - i/(l+1)) sum_t e_t e_{t+i}
/// ```
///
/// // TODO(phase0): delegate to the tsecon-hac Bartlett LRV once wired up.
fn bartlett_lrv(e: &[f64], lags: usize) -> f64 {
    let n = e.len();
    let mut s: f64 = e.iter().map(|&v| v * v).sum();
    for i in 1..=lags.min(n - 1) {
        let dot: f64 = e[i..].iter().zip(&e[..n - i]).map(|(&a, &b)| a * b).sum();
        s += 2.0 * dot * (1.0 - i as f64 / (lags as f64 + 1.0));
    }
    s / n as f64
}

/// Hobijn-Franses-Ooms (1998) automatic Bartlett bandwidth as implemented
/// by statsmodels `_kpss_autolag`: with `m = floor(n^{2/9})` and
/// `g_i = (1/n) sum_t e_t e_{t+i}` the sample autocovariances,
///
/// ```text
/// s0 = g_0 + 2 (g_1 + .. + g_m)
/// s1 = 2 (1 g_1 + 2 g_2 + .. + m g_m)
/// l  = floor( 1.1447 ((s1/s0)^2)^{1/3} n^{1/3} )
/// ```
fn kpss_autolag(resids: &[f64], n: usize) -> Result<usize, DiagError> {
    let nf = n as f64;
    let covlags = nf.powf(2.0 / 9.0) as usize;
    let mut s0: f64 = resids.iter().map(|&v| v * v).sum::<f64>() / nf;
    let mut s1 = 0.0;
    for i in 1..=covlags.min(n - 1) {
        let dot: f64 = resids[i..]
            .iter()
            .zip(&resids[..n - i])
            .map(|(&a, &b)| a * b)
            .sum();
        let rp = dot / (nf / 2.0);
        s0 += rp;
        s1 += i as f64 * rp;
    }
    if s0 == 0.0 || !s0.is_finite() {
        return Err(DiagError::NumericalBreakdown { what: "kpss" });
    }
    let s_hat = s1 / s0;
    let gamma_hat = 1.1447 * (s_hat * s_hat).powf(1.0 / 3.0);
    Ok((gamma_hat * nf.powf(1.0 / 3.0)) as usize)
}

/// Linear interpolation of the KPSS statistic in the critical-value table
/// (`numpy.interp` semantics: clamped to the table's p-value range).
fn kpss_pvalue(stat: f64, crit: &[f64; 4]) -> f64 {
    const PVALS: [f64; 4] = [0.10, 0.05, 0.025, 0.01];
    if stat <= crit[0] {
        return PVALS[0];
    }
    if stat >= crit[3] {
        return PVALS[3];
    }
    for i in 0..3 {
        if stat <= crit[i + 1] {
            let slope = (PVALS[i + 1] - PVALS[i]) / (crit[i + 1] - crit[i]);
            return slope * (stat - crit[i]) + PVALS[i];
        }
    }
    PVALS[3]
}

/// KPSS stationarity test (Kwiatkowski, Phillips, Schmidt & Shin 1992),
/// matching statsmodels `kpss`.
///
/// The series is demeaned ([`KpssRegression::Constant`]) or OLS-detrended
/// ([`KpssRegression::ConstantTrend`]); with partial sums
/// `S_t = e_1 + .. + e_t` of the residuals the LM statistic is
///
/// ```text
/// KPSS = sum_t S_t^2 / (n^2 s^2(l))
/// ```
///
/// where `s^2(l)` is the Bartlett long-run variance ([`bartlett_lrv`],
/// Newey & West 1987 weights). `H0`: the series is (level/trend)
/// stationary; large statistics reject. The p-value interpolates the KPSS
/// table and is bounded to `[0.01, 0.10]` — the test is extremely
/// bandwidth-sensitive, so the bandwidth used is always reported.
///
/// # Errors
///
/// * [`DiagError::NonFinite`] if the series contains NaN or infinities.
/// * [`DiagError::SeriesTooShort`] for fewer than 4 observations.
/// * [`DiagError::ConstantSeries`] if the series is constant.
/// * [`DiagError::InvalidLags`] if a fixed bandwidth is `>= n`.
/// * [`DiagError::NumericalBreakdown`] if the long-run variance is not
///   strictly positive (an exactly deterministic series).
pub fn kpss(
    y: &[f64],
    regression: KpssRegression,
    lags: KpssLags,
) -> Result<KpssResult, DiagError> {
    let n = check_series(y, 4, "kpss")?;
    if y.iter().all(|&v| v == y[0]) {
        return Err(DiagError::ConstantSeries { what: "kpss" });
    }
    let nf = n as f64;

    let (resids, crit_row) = match regression {
        KpssRegression::Constant => {
            let mean = y.iter().sum::<f64>() / nf;
            let r: Vec<f64> = y.iter().map(|&v| v - mean).collect();
            (r, [0.347, 0.463, 0.574, 0.739])
        }
        KpssRegression::ConstantTrend => {
            // OLS of y on [1, t], t = 1..n, via the centered closed form.
            let t_mean = (nf + 1.0) / 2.0;
            let y_mean = y.iter().sum::<f64>() / nf;
            let mut sxy = 0.0;
            let mut sxx = 0.0;
            for (i, &v) in y.iter().enumerate() {
                let tc = (i + 1) as f64 - t_mean;
                sxy += tc * (v - y_mean);
                sxx += tc * tc;
            }
            let slope = sxy / sxx;
            let r: Vec<f64> = y
                .iter()
                .enumerate()
                .map(|(i, &v)| v - y_mean - slope * ((i + 1) as f64 - t_mean))
                .collect();
            (r, [0.119, 0.146, 0.176, 0.216])
        }
    };

    let nlags = match lags {
        KpssLags::Legacy => {
            let l = (12.0 * (nf / 100.0).powf(0.25)).ceil() as usize;
            l.min(n - 1)
        }
        KpssLags::Auto => kpss_autolag(&resids, n)?.min(n - 1),
        KpssLags::Fixed(l) => {
            if l >= n {
                return Err(DiagError::InvalidLags {
                    what: "kpss",
                    nlags: l,
                    n,
                    requirement: "nlags < n (the Bartlett window cannot \
                                  exceed the sample)",
                });
            }
            l
        }
    };

    // eq. 11, p. 165 of KPSS (1992): eta = sum of squared partial sums.
    let mut cum = 0.0;
    let mut eta = 0.0;
    for &e in &resids {
        cum += e;
        eta += cum * cum;
    }
    eta /= nf * nf;

    let s2 = bartlett_lrv(&resids, nlags);
    if !(s2 > 0.0 && s2.is_finite()) {
        return Err(DiagError::NumericalBreakdown { what: "kpss" });
    }
    let statistic = eta / s2;

    Ok(KpssResult {
        statistic,
        p_value: kpss_pvalue(statistic, &crit_row),
        lags: nlags,
        nobs: n,
        crit: KpssCriticalValues {
            pct10: crit_row[0],
            pct5: crit_row[1],
            pct2_5: crit_row[2],
            pct1: crit_row[3],
        },
        regression,
    })
}

// ------------------------------------------- confirmatory decision logic

/// The four cells of the ADF + KPSS confirmatory matrix.
///
/// The two tests have opposite nulls (ADF: unit root; KPSS:
/// stationarity), so running both yields a 2x2 evidence table (Elder &
/// Kennedy 2001) rather than one bare verdict.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StationarityQuadrant {
    /// ADF rejects the unit root and KPSS does not reject stationarity:
    /// both tests point at a stationary series.
    Stationary,
    /// ADF fails to reject and KPSS rejects: both tests point at a unit
    /// root.
    UnitRoot,
    /// Both tests reject their nulls — mutually contradictory readings,
    /// typically a deterministic trend, structural breaks, or long memory.
    Conflict,
    /// Neither test rejects — the sample is too uninformative to separate
    /// the hypotheses (both tests have low power here).
    Inconclusive,
}

/// The workflow recommendation attached to a [`StationarityQuadrant`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Recommendation {
    /// Model the series in levels.
    Proceed,
    /// First-difference the series and re-test.
    Difference,
    /// Remove (or explicitly model) a deterministic trend, then re-test.
    Detrend,
}

/// Joint ADF + KPSS stationarity report — the Module 01 decision-workflow
/// pattern: the evidence from both tests, the confirmatory quadrant they
/// fall in, and a teaching interpretation with a concrete next step.
#[derive(Debug, Clone, PartialEq)]
pub struct StationarityReport {
    /// The underlying ADF test (constant regression, AIC lag selection).
    pub adf: AdfResult,
    /// The underlying KPSS test (constant regression, automatic
    /// bandwidth).
    pub kpss: KpssResult,
    /// The significance level both decisions were taken at.
    pub alpha: f64,
    /// Whether ADF rejects its unit-root null (`p < alpha`).
    pub adf_rejects: bool,
    /// Whether KPSS rejects its stationarity null (`p < alpha`). Note the
    /// KPSS p-value is table-bounded to `[0.01, 0.10]`, so decisions at
    /// `alpha` outside that range are not meaningful.
    pub kpss_rejects: bool,
    /// The confirmatory quadrant the two decisions fall in.
    pub quadrant: StationarityQuadrant,
    /// The recommended next step.
    pub recommendation: Recommendation,
    /// Plain-language interpretation of the joint evidence.
    pub interpretation: String,
}

impl fmt::Display for StationarityReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ADF({}, lag {}): stat = {:.4}, p = {:.4} [{}]; \
             KPSS({}, bw {}): stat = {:.4}, p = {:.4} [{}] — {}",
            self.adf.regression.code(),
            self.adf.used_lag,
            self.adf.statistic,
            self.adf.p_value,
            if self.adf_rejects {
                "reject unit root"
            } else {
                "fail to reject"
            },
            self.kpss.regression.code(),
            self.kpss.lags,
            self.kpss.statistic,
            self.kpss.p_value,
            if self.kpss_rejects {
                "reject stationarity"
            } else {
                "fail to reject"
            },
            self.interpretation
        )
    }
}

/// Classify a pair of test decisions into the confirmatory quadrant.
fn classify(adf_rejects: bool, kpss_rejects: bool) -> (StationarityQuadrant, Recommendation) {
    match (adf_rejects, kpss_rejects) {
        (true, false) => (StationarityQuadrant::Stationary, Recommendation::Proceed),
        (false, true) => (StationarityQuadrant::UnitRoot, Recommendation::Difference),
        (true, true) => (StationarityQuadrant::Conflict, Recommendation::Detrend),
        (false, false) => (
            StationarityQuadrant::Inconclusive,
            Recommendation::Difference,
        ),
    }
}

fn interpretation(quadrant: StationarityQuadrant, alpha: f64) -> String {
    let pct = alpha * 100.0;
    match quadrant {
        StationarityQuadrant::Stationary => format!(
            "At the {pct:.0}% level ADF rejects a unit root and KPSS finds \
             no evidence against stationarity — the two tests agree the \
             series looks I(0). Proceed to model it in levels; check the \
             ACF/PACF for short-run dynamics next."
        ),
        StationarityQuadrant::UnitRoot => format!(
            "At the {pct:.0}% level ADF cannot reject a unit root and KPSS \
             rejects stationarity — the tests agree the series looks I(1). \
             Difference it once and re-run this battery on the differences \
             before modeling; regressing I(1) levels on each other risks \
             spurious regression unless you are explicitly testing for \
             cointegration."
        ),
        StationarityQuadrant::Conflict => format!(
            "At the {pct:.0}% level both tests reject: ADF says no unit \
             root while KPSS says not level-stationary. This pattern \
             usually means stationarity around a deterministic trend, \
             structural breaks, or long memory rather than a clean I(0)/\
             I(1) dichotomy. Detrend (or re-run both tests with a trend \
             specification) and consider break-robust tests before \
             resorting to differencing."
        ),
        StationarityQuadrant::Inconclusive => format!(
            "At the {pct:.0}% level neither test rejects its null: the \
             sample carries too little information to separate a unit root \
             from stationarity — both tests are low-powered here. The \
             conservative default is to difference (over-differencing is \
             usually less costly than spurious levels regression), but a \
             longer sample is the only real fix."
        ),
    }
}

/// One-call ADF + KPSS confirmatory stationarity check at the 5% level.
///
/// Runs [`adf`] with a constant and AIC lag selection and [`kpss`] with a
/// constant and the automatic bandwidth — the two tests' conventional
/// defaults — then classifies the joint outcome into the confirmatory
/// quadrant and attaches a teaching recommendation (difference, detrend,
/// or proceed). See [`check_stationarity_at`] to choose the level.
///
/// # Errors
///
/// Propagates any [`DiagError`] from the underlying tests.
pub fn check_stationarity(y: &[f64]) -> Result<StationarityReport, DiagError> {
    check_stationarity_at(y, 0.05)
}

/// [`check_stationarity`] at a caller-chosen significance level.
///
/// Note the KPSS p-value is bounded to `[0.01, 0.10]`, so `alpha` outside
/// that range cannot flip the KPSS decision; conventional choices are
/// 0.05 and 0.10.
///
/// # Errors
///
/// [`DiagError::InvalidAlpha`] unless `0 < alpha < 1`, plus anything the
/// underlying tests return.
pub fn check_stationarity_at(y: &[f64], alpha: f64) -> Result<StationarityReport, DiagError> {
    check_alpha(alpha)?;
    let adf_res = adf(y, AdfRegression::Constant, AdfLagSelection::Aic(None))?;
    let kpss_res = kpss(y, KpssRegression::Constant, KpssLags::Auto)?;
    let adf_rejects = adf_res.p_value < alpha;
    let kpss_rejects = kpss_res.p_value < alpha;
    let (quadrant, recommendation) = classify(adf_rejects, kpss_rejects);
    Ok(StationarityReport {
        adf: adf_res,
        kpss: kpss_res,
        alpha,
        adf_rejects,
        kpss_rejects,
        quadrant,
        recommendation,
        interpretation: interpretation(quadrant, alpha),
    })
}
