//! Zivot-Andrews (1992) unit-root test with one endogenous structural
//! break.
//!
//! The null hypothesis is a unit root (with drift, and *no* break); the
//! alternative is stationarity around a broken deterministic component —
//! an intercept shift ([`ZaRegression::Constant`], ZA "model A"), a
//! trend-slope shift ([`ZaRegression::Trend`], "model B"), or both
//! ([`ZaRegression::ConstantTrend`], "model C"). For every candidate break
//! date inside the trimmed range an ADF-style regression with the
//! appropriate break dummies is fit, and the test statistic is the
//! **minimum** t-statistic on the lagged level over all candidates — the
//! break date is chosen where the unit-root null looks *least* favorable,
//! which is what makes the distribution nonstandard (Zivot & Andrews 1992).
//!
//! # Reference conventions (statsmodels 0.14.6, matched exactly)
//!
//! The implementation matches `statsmodels.tsa.stattools.zivot_andrews`
//! bit-for-bit in structure (`arch.unitroot.ZivotAndrews` 8.0.0 implements
//! the identical algorithm and agrees to machine precision — the two share
//! the same Baum code lineage, so they are *not* independent references).
//! The conventions that had to be reverse-engineered from the source, in
//! full, because none of them is documented:
//!
//! * **Lag selection is the Baum (2004/2015) approximation**: a *single*
//!   up-front `adfuller(y, regression="ct", autolag=...)` on the base model
//!   (constant + trend, **no** break dummies) picks the augmentation lag,
//!   which is then reused for every candidate break regression — the
//!   original paper re-selects per candidate. This is faster and "slightly
//!   more pessimistic" (statsmodels' own words). With `autolag = None` and
//!   a fixed lag, that lag is used directly; with neither, the lag is the
//!   *truncated* (not ceiled) Schwert rule `int(12 (n/100)^{1/4})`.
//! * **Trimming**: `trimcnt = int(n * trim)` (truncation); candidate break
//!   periods are `bp = trimcnt + 1 ..= n - trimcnt` (1-based period), so
//!   the reported 0-based `break_index = bp - 1` ranges over
//!   `[trimcnt, n - trimcnt - 1]`. `trim` must lie in `[0, 1/3]`.
//! * **Regression** (every model includes constant *and* trend; the
//!   `regression` choice selects which *break dummies* enter): on rows
//!   `t = lags+1 ..= n-1` (0-based observation index of `dy_t`),
//!
//!   ```text
//!   dy_t = b0 + [theta DU_t] + b1 tr_t + [gamma DT_t] + alpha y_{t-1}
//!          + sum_{j=1..lags} c_j dy_{t-j} + e_t
//!   ```
//!
//!   with the statistic the OLS t-ratio on `alpha` (classical SEs,
//!   `s^2 = SSR/(rows - k)`).
//! * **Break-dummy timing**: with `cutoff = bp - lags - 1` (the regression
//!   row of observation `bp`), the intercept dummy is
//!   `DU_t = 1{t >= bp}` — the shift *begins at* `bp = break_index + 1`,
//!   i.e. `break_index` is the last pre-break observation. The trend-shift
//!   dummy is `DT_t = (t - bp + 2) 1{t >= bp}` (scaled) for `"ct"` but
//!   `DT_t = (t - bp + 2) 1{t >= bp - 1}` for `"t"` — the `"t"` model's
//!   ramp starts one observation *earlier*. That asymmetry is inherited
//!   from the reference (and from Baum's Stata `zandrews`); it changes the
//!   `"t"` statistic (no DU dummy absorbs the offset there), so it is
//!   replicated exactly rather than "fixed".
//! * **Numerical scaling** (replicated for bit-parity, irrelevant in exact
//!   arithmetic): `dy` and `y` are normalized to unit Euclidean norm, the
//!   constant column is `1/sqrt(n)`, and the trend column is
//!   `(t + 2) sqrt(3)/n^{3/2}`.
//! * **Ties**: the *first* (earliest) candidate attaining the minimum wins
//!   (`np.argmin` semantics).
//!
//! # P-values and critical values
//!
//! The p-value linearly interpolates the statistic in a simulated null
//! distribution table (100,000 Monte Carlo replications of 2,000
//! observations), transcribed from `statsmodels.tsa.stattools.
//! ZivotAndrewsUnitRoot` (statsmodels 0.14.6, BSD-3 — same transcription
//! precedent as the MacKinnon surfaces in [`crate::mackinnon`]); values are
//! clamped to the table range `[1e-5, 0.999]`. Critical values interpolate
//! the same table at 1/5/10% (which are exact table knots). The
//! interpolation is honest to roughly the table's Monte Carlo resolution —
//! do not read more than two decimals into a ZA p-value.
//!
//! # Caveat (teachable)
//!
//! ZA allows the break only under the *alternative*: a break under the
//! null (a unit root with a broken drift) is mis-specified and produces
//! spurious rejections (the classic criticism — see Lee & Strazicich 2003,
//! whose minimum-LM test allows the break under both hypotheses). Treat a
//! ZA rejection with a wildly implausible estimated break date as a red
//! flag, not a discovery.
//!
//! References: Zivot & Andrews (1992), JBES 10(3); Baum (2004, rev. 2015),
//! Stata module ZANDREWS; Schwert (1989), JBES 7(2); Perron (1989),
//! Econometrica 57(6); Lee & Strazicich (2003), REStat 85(4).

use crate::error::DiagError;
use crate::ols::ols_detailed;
use crate::unitroot::{adf, AdfLagSelection, AdfRegression};
use crate::validate::check_series;

const WHAT: &str = "zivot_andrews";

/// Which deterministic component is allowed to break under the
/// alternative. Every model's regression includes a constant *and* a
/// linear trend; this selects the break dummies only.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZaRegression {
    /// Intercept (level) shift only — ZA "model A" (statsmodels `"c"`,
    /// the conventional default).
    Constant,
    /// Trend-slope shift only — ZA "model B" (statsmodels `"t"`).
    Trend,
    /// Both intercept and trend-slope shift — ZA "model C" (statsmodels
    /// `"ct"`).
    ConstantTrend,
}

impl ZaRegression {
    /// The statsmodels code for this specification.
    pub fn code(self) -> &'static str {
        match self {
            ZaRegression::Constant => "c",
            ZaRegression::Trend => "t",
            ZaRegression::ConstantTrend => "ct",
        }
    }

    /// Number of non-lag columns in the candidate regression
    /// (statsmodels `basecols`): constant, dummies, trend, lagged level.
    fn basecols(self) -> usize {
        match self {
            ZaRegression::ConstantTrend => 5,
            _ => 4,
        }
    }
}

/// How the augmentation lag shared by all candidate regressions is chosen
/// (the Baum single up-front selection; see the module docs).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZaLagSelection {
    /// Use exactly this many lagged differences (statsmodels
    /// `autolag=None` with `maxlag` set).
    Fixed(usize),
    /// statsmodels `autolag=None`, `maxlag=None`: the *truncated* Schwert
    /// rule `int(12 (n/100)^{1/4})` — note the ADF default ceils instead.
    SchwertTrunc,
    /// Minimize AIC in the base `"ct"` ADF regression (statsmodels
    /// `autolag="aic"`, the default), searching `0..=maxlag`; `None` uses
    /// the ADF default maxlag (ceiled Schwert rule, capped).
    Aic(Option<usize>),
    /// Minimize BIC in the base `"ct"` ADF regression.
    Bic(Option<usize>),
    /// statsmodels `"t-stat"` rule in the base `"ct"` ADF regression.
    TStat(Option<usize>),
}

/// Critical values of the Zivot-Andrews minimum-t distribution
/// (interpolated in the transcribed statsmodels simulation table; they do
/// not depend on the sample size).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ZaCriticalValues {
    /// The 1% critical value.
    pub pct1: f64,
    /// The 5% critical value.
    pub pct5: f64,
    /// The 10% critical value.
    pub pct10: f64,
}

/// Result of the Zivot-Andrews structural-break unit-root test.
///
/// The null hypothesis is a unit root with no break; the alternative is
/// stationarity around one broken deterministic component, so small
/// p-values speak *for* break-stationarity.
#[derive(Debug, Clone, PartialEq)]
pub struct ZaResult {
    /// The minimum-t statistic over all candidate break dates.
    pub statistic: f64,
    /// P-value interpolated in the simulated null table, clamped to
    /// `[1e-5, 0.999]`.
    pub p_value: f64,
    /// 0-based index into `y` of the *last pre-break* observation at the
    /// minimizing candidate: the estimated shift begins at
    /// `break_index + 1` (the statsmodels `bpidx` convention). Always in
    /// `[trimcnt, n - trimcnt - 1]` with `trimcnt = int(n * trim)`.
    pub break_index: usize,
    /// The augmentation lag shared by all candidate regressions.
    pub used_lag: usize,
    /// Length of `y`.
    pub nobs: usize,
    /// Critical values at 1/5/10%.
    pub crit: ZaCriticalValues,
    /// The break specification that was tested.
    pub regression: ZaRegression,
    /// The trimming fraction used.
    pub trim: f64,
}

// ------------------------------------------------- simulated null tables
//
// Transcribed verbatim from `statsmodels.tsa.stattools.
// ZivotAndrewsUnitRoot.__init__` (statsmodels 0.14.6, BSD-3-Clause;
// simulated there with 100,000 Monte Carlo replications of 2,000
// observations). Rows are `(percentile-of-null * 100, statistic)`, both
// ascending. The same transcription-with-attribution precedent as the
// MacKinnon response surfaces ([`crate::mackinnon`]) and the arch adf-z
// tables ([`crate::mackinnon_ext`]).

/// Model A ("c"): intercept-break null quantiles.
const ZA_TABLE_C: [(f64, f64); 48] = [
    (0.001, -6.78442),
    (0.1, -5.83192),
    (0.2, -5.68139),
    (0.3, -5.58461),
    (0.4, -5.51308),
    (0.5, -5.45043),
    (0.6, -5.39924),
    (0.7, -5.36023),
    (0.8, -5.33219),
    (0.9, -5.30294),
    (1.0, -5.27644),
    (2.5, -5.0334),
    (5.0, -4.81067),
    (7.5, -4.67636),
    (10.0, -4.56618),
    (12.5, -4.4813),
    (15.0, -4.40507),
    (17.5, -4.33947),
    (20.0, -4.28155),
    (22.5, -4.22683),
    (25.0, -4.1783),
    (27.5, -4.13101),
    (30.0, -4.08586),
    (32.5, -4.04455),
    (35.0, -4.0038),
    (37.5, -3.96144),
    (40.0, -3.92078),
    (42.5, -3.88178),
    (45.0, -3.84503),
    (47.5, -3.80549),
    (50.0, -3.77031),
    (52.5, -3.73209),
    (55.0, -3.696),
    (57.5, -3.65985),
    (60.0, -3.62126),
    (65.0, -3.5458),
    (70.0, -3.46848),
    (75.0, -3.38533),
    (80.0, -3.29112),
    (85.0, -3.17832),
    (90.0, -3.04165),
    (92.5, -2.95146),
    (95.0, -2.83179),
    (96.0, -2.76465),
    (97.0, -2.68624),
    (98.0, -2.57884),
    (99.0, -2.40044),
    (99.9, -1.88932),
];

/// Model B ("t"): trend-break null quantiles.
const ZA_TABLE_T: [(f64, f64); 48] = [
    (0.001, -83.9094),
    (0.1, -13.8837),
    (0.2, -9.13205),
    (0.3, -6.32564),
    (0.4, -5.60803),
    (0.5, -5.38794),
    (0.6, -5.26585),
    (0.7, -5.18734),
    (0.8, -5.12756),
    (0.9, -5.07984),
    (1.0, -5.03421),
    (2.5, -4.65634),
    (5.0, -4.4058),
    (7.5, -4.25214),
    (10.0, -4.13678),
    (12.5, -4.03765),
    (15.0, -3.95185),
    (17.5, -3.87945),
    (20.0, -3.81295),
    (22.5, -3.75273),
    (25.0, -3.69836),
    (27.5, -3.64785),
    (30.0, -3.59819),
    (32.5, -3.55146),
    (35.0, -3.50522),
    (37.5, -3.45987),
    (40.0, -3.41672),
    (42.5, -3.37465),
    (45.0, -3.33394),
    (47.5, -3.29393),
    (50.0, -3.25316),
    (52.5, -3.21244),
    (55.0, -3.17124),
    (57.5, -3.13211),
    (60.0, -3.09204),
    (65.0, -3.01135),
    (70.0, -2.92897),
    (75.0, -2.83614),
    (80.0, -2.73893),
    (85.0, -2.6284),
    (90.0, -2.49611),
    (92.5, -2.41337),
    (95.0, -2.3082),
    (96.0, -2.25797),
    (97.0, -2.19648),
    (98.0, -2.1132),
    (99.0, -1.99138),
    (99.9, -1.67466),
];

/// Model C ("ct"): intercept-and-trend-break null quantiles.
const ZA_TABLE_CT: [(f64, f64); 48] = [
    (0.001, -38.178),
    (0.1, -6.43107),
    (0.2, -6.07279),
    (0.3, -5.95496),
    (0.4, -5.86254),
    (0.5, -5.77081),
    (0.6, -5.72541),
    (0.7, -5.68406),
    (0.8, -5.65163),
    (0.9, -5.60419),
    (1.0, -5.57556),
    (2.5, -5.29704),
    (5.0, -5.07332),
    (7.5, -4.93003),
    (10.0, -4.82668),
    (12.5, -4.73711),
    (15.0, -4.6602),
    (17.5, -4.5897),
    (20.0, -4.52855),
    (22.5, -4.471),
    (25.0, -4.42011),
    (27.5, -4.37387),
    (30.0, -4.32705),
    (32.5, -4.28126),
    (35.0, -4.23793),
    (37.5, -4.19822),
    (40.0, -4.158),
    (42.5, -4.11946),
    (45.0, -4.08064),
    (47.5, -4.04286),
    (50.0, -4.00489),
    (52.5, -3.96837),
    (55.0, -3.932),
    (57.5, -3.89496),
    (60.0, -3.85577),
    (65.0, -3.77795),
    (70.0, -3.69794),
    (75.0, -3.61852),
    (80.0, -3.52485),
    (85.0, -3.41665),
    (90.0, -3.28527),
    (92.5, -3.19724),
    (95.0, -3.08769),
    (96.0, -3.03088),
    (97.0, -2.96091),
    (98.0, -2.85581),
    (99.0, -2.71015),
    (99.9, -2.28767),
];

fn za_table(regression: ZaRegression) -> &'static [(f64, f64); 48] {
    match regression {
        ZaRegression::Constant => &ZA_TABLE_C,
        ZaRegression::Trend => &ZA_TABLE_T,
        ZaRegression::ConstantTrend => &ZA_TABLE_CT,
    }
}

/// `numpy.interp` semantics on an ascending knot vector: clamp outside the
/// range, otherwise piecewise-linear with the same
/// `y0 + slope * (x - x0)` arithmetic (bit-parity with the reference).
fn interp(x: f64, xs: impl Fn(usize) -> f64, ys: impl Fn(usize) -> f64, len: usize) -> f64 {
    if x <= xs(0) {
        return ys(0);
    }
    if x >= xs(len - 1) {
        return ys(len - 1);
    }
    // Find the segment [i, i+1] with xs(i) <= x < xs(i+1).
    let mut i = 0;
    while i + 2 < len && x >= xs(i + 1) {
        i += 1;
    }
    let (x0, x1) = (xs(i), xs(i + 1));
    let (y0, y1) = (ys(i), ys(i + 1));
    let slope = (y1 - y0) / (x1 - x0);
    slope * (x - x0) + y0
}

/// P-value of a ZA statistic: interpolate the statistic in the simulated
/// null table (statistic -> percentile / 100), clamped to
/// `[1e-5, 0.999]` (the table range).
pub fn za_p(stat: f64, regression: ZaRegression) -> f64 {
    let table = za_table(regression);
    interp(stat, |i| table[i].1, |i| table[i].0, table.len()) / 100.0
}

/// Critical values at 1/5/10% — exact knots of the simulated table, read
/// through the same interpolation the reference uses.
pub fn za_crit(regression: ZaRegression) -> ZaCriticalValues {
    let table = za_table(regression);
    let at = |pct: f64| interp(pct, |i| table[i].0, |i| table[i].1, table.len());
    ZaCriticalValues {
        pct1: at(1.0),
        pct5: at(5.0),
        pct10: at(10.0),
    }
}

/// Re-attribute an error surfaced by the internal base-ADF autolag step to
/// this test, so the caller sees the function they actually called.
fn reattribute(err: DiagError) -> DiagError {
    match err {
        DiagError::SeriesTooShort { n, needed, .. } => DiagError::SeriesTooShort {
            what: WHAT,
            n,
            needed,
        },
        DiagError::InvalidLags {
            nlags,
            n,
            requirement,
            ..
        } => DiagError::InvalidLags {
            what: WHAT,
            nlags,
            n,
            requirement,
        },
        DiagError::ConstantSeries { .. } => DiagError::ConstantSeries { what: WHAT },
        DiagError::SingularDesign { .. } => DiagError::SingularDesign { what: WHAT },
        DiagError::NumericalBreakdown { .. } => DiagError::NumericalBreakdown { what: WHAT },
        other => other,
    }
}

/// Zivot-Andrews (1992) unit-root test with one endogenous break,
/// matching `statsmodels.tsa.stattools.zivot_andrews` (see the module docs
/// for the exact conventions, including the Baum single up-front lag
/// selection and the break-dummy timing).
///
/// `H0`: unit root (no break); `H1`: stationarity around one broken
/// deterministic component selected by `regression`. The statistic is the
/// minimum over candidate break dates `bp = trimcnt+1 ..= n-trimcnt`
/// (`trimcnt = int(n * trim)`) of the t-ratio on the lagged level in
///
/// ```text
/// dy_t = b0 + [theta DU_t] + b1 tr_t + [gamma DT_t] + alpha y_{t-1}
///        + sum_{j=1..lags} c_j dy_{t-j} + e_t
/// ```
///
/// and `break_index = bp_min - 1` is the last pre-break observation.
/// P-values/critical values interpolate the simulated null table
/// ([`za_p`], [`za_crit`]).
///
/// # Errors
///
/// * [`DiagError::InvalidTrim`] unless `0 <= trim <= 1/3`.
/// * [`DiagError::NonFinite`] if the series contains NaN or infinities.
/// * [`DiagError::ConstantSeries`] if the series is constant.
/// * [`DiagError::SeriesTooShort`] if fewer observations remain than
///   coefficients (plus one residual degree of freedom) in the candidate
///   regressions.
/// * [`DiagError::InvalidLags`] if `lags > trimcnt - 1`: the first
///   candidate's break dummy would have no pre-break regime, making the
///   design collinear (the reference raises an opaque rank error here, or
///   silently degrades for `lags > trimcnt`; this library refuses with an
///   explanation instead — increase `trim` or reduce the lag).
/// * [`DiagError::SingularDesign`] / [`DiagError::NumericalBreakdown`]
///   if any candidate design is collinear or fits exactly (the reference
///   rank-checks only the first candidate; this library checks all).
pub fn zivot_andrews(
    y: &[f64],
    regression: ZaRegression,
    trim: f64,
    lag_selection: ZaLagSelection,
) -> Result<ZaResult, DiagError> {
    if !(0.0..=1.0 / 3.0).contains(&trim) {
        return Err(DiagError::InvalidTrim { value: trim });
    }
    let basecols = regression.basecols();
    let n = check_series(y, basecols + 3, WHAT)?;
    if y.iter().all(|&v| v == y[0]) {
        return Err(DiagError::ConstantSeries { what: WHAT });
    }
    let nf = n as f64;

    // ---- Baum single up-front lag selection (see module docs).
    let lags = match lag_selection {
        ZaLagSelection::Fixed(l) => l,
        ZaLagSelection::SchwertTrunc => (12.0 * (nf / 100.0).powf(0.25)) as usize,
        ZaLagSelection::Aic(maxlag) => {
            adf(
                y,
                AdfRegression::ConstantTrend,
                AdfLagSelection::Aic(maxlag),
            )
            .map_err(reattribute)?
            .used_lag
        }
        ZaLagSelection::Bic(maxlag) => {
            adf(
                y,
                AdfRegression::ConstantTrend,
                AdfLagSelection::Bic(maxlag),
            )
            .map_err(reattribute)?
            .used_lag
        }
        ZaLagSelection::TStat(maxlag) => {
            adf(
                y,
                AdfRegression::ConstantTrend,
                AdfLagSelection::TStat(maxlag),
            )
            .map_err(reattribute)?
            .used_lag
        }
    };

    // ---- Trimmed candidate range and feasibility.
    let trimcnt = (nf * trim) as usize; // int(n * trim), truncated
    if lags + 1 > trimcnt {
        return Err(DiagError::InvalidLags {
            what: WHAT,
            nlags: lags,
            n,
            requirement: "lags <= int(n * trim) - 1: every candidate break inside the \
                          trimmed window must leave at least one pre-break regression \
                          row, or the break dummy is collinear with the constant; \
                          increase trim or reduce the lag order",
        });
    }
    let rows = n - 1 - lags;
    let k = basecols + lags;
    if rows < k + 1 {
        return Err(DiagError::SeriesTooShort {
            what: WHAT,
            n,
            needed: 2 * lags + basecols + 2,
        });
    }

    // ---- statsmodels' exact numerical scaling (bit-parity; the t-ratio
    // is scale-invariant in exact arithmetic).
    let dy_norm = {
        let mut s = 0.0;
        for w in y.windows(2) {
            let d = w[1] - w[0];
            s += d * d;
        }
        s.sqrt()
    };
    let y_norm = y.iter().map(|&v| v * v).sum::<f64>().sqrt();
    if !(dy_norm > 0.0 && dy_norm.is_finite() && y_norm > 0.0 && y_norm.is_finite()) {
        return Err(DiagError::NumericalBreakdown { what: WHAT });
    }
    // Standardized differences and levels.
    let dys: Vec<f64> = y.windows(2).map(|w| (w[1] - w[0]) / dy_norm).collect();
    let ys: Vec<f64> = y.iter().map(|&v| v / y_norm).collect();
    let c_const = 1.0 / nf.sqrt();
    let t_scale = 3.0_f64.sqrt() / (nf * nf.sqrt());

    // ---- Static columns (constant, trend, lagged level, lagged diffs);
    // the break dummies are rewritten per candidate.
    // Column order matches statsmodels: for "c"/"ct" the intercept dummy
    // is column 1 and the trend column 2; for "t" the trend is column 1.
    let mut cols: Vec<Vec<f64>> = vec![vec![0.0; rows]; k];
    for v in cols[0].iter_mut() {
        *v = c_const;
    }
    let trend_col = match regression {
        ZaRegression::Trend => 1,
        _ => 2,
    };
    for (i, v) in cols[trend_col].iter_mut().enumerate() {
        *v = (lags as f64 + 3.0 + i as f64) * t_scale;
    }
    for (i, v) in cols[basecols - 1].iter_mut().enumerate() {
        *v = ys[lags + i];
    }
    for j in 1..=lags {
        for i in 0..rows {
            cols[basecols - 1 + j][i] = dys[lags + i - j];
        }
    }
    let dep: Vec<f64> = dys[lags..].to_vec();

    // ---- Minimum-t search over the trimmed candidate range.
    let mut best_stat = f64::INFINITY;
    let mut best_bp = 0usize;
    for bp in (trimcnt + 1)..=(n - trimcnt) {
        let cutoff = bp - lags - 1; // >= 1 by the lags guard above
        match regression {
            ZaRegression::Constant | ZaRegression::ConstantTrend => {
                // DU_t = 1{t >= bp}, scaled by the constant's normalization.
                for (i, v) in cols[1].iter_mut().enumerate() {
                    *v = if i < cutoff { 0.0 } else { c_const };
                }
                if regression == ZaRegression::ConstantTrend {
                    // DT_t = (t - bp + 2) 1{t >= bp}, scaled.
                    for (i, v) in cols[3].iter_mut().enumerate() {
                        *v = if i < cutoff {
                            0.0
                        } else {
                            (i - cutoff + 2) as f64 * t_scale
                        };
                    }
                }
            }
            ZaRegression::Trend => {
                // DT_t = (t - bp + 2) 1{t >= bp - 1}, scaled: the "t"
                // model's ramp starts one row earlier (reference quirk,
                // replicated — see module docs).
                for (i, v) in cols[2].iter_mut().enumerate() {
                    *v = if i + 1 < cutoff {
                        0.0
                    } else {
                        (i + 2 - cutoff) as f64 * t_scale
                    };
                }
            }
        }
        let fit = ols_detailed(&cols, &dep, WHAT)?;
        let t_alpha = fit.t_values[basecols - 1];
        if t_alpha < best_stat {
            best_stat = t_alpha;
            best_bp = bp;
        }
    }
    if !best_stat.is_finite() || best_bp == 0 {
        return Err(DiagError::NumericalBreakdown { what: WHAT });
    }

    Ok(ZaResult {
        statistic: best_stat,
        p_value: za_p(best_stat, regression),
        break_index: best_bp - 1,
        used_lag: lags,
        nobs: n,
        crit: za_crit(regression),
        regression,
        trim,
    })
}
