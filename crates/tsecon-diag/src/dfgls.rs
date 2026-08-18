//! DF-GLS unit-root test (Elliott, Rothenberg & Stock 1996): the ADF test
//! run on a *GLS-detrended* series, giving near-optimal local asymptotic
//! power — the recommended default over the plain ADF when a constant or
//! trend must be estimated.
//!
//! The test has three moving parts, each matching `arch.unitroot.DFGLS`
//! (arch 8.0.0) exactly:
//!
//! 1. **GLS detrending** ([`gls_detrend`]): the deterministics `z_t`
//!    (constant, or constant + trend) are estimated under the *local
//!    alternative* `rho = 1 + cbar/T` by quasi-differencing both `y` and
//!    `z` at `cbar = -7.0` (constant case) or `cbar = -13.5` (trend case)
//!    — the ERS choices at which the local asymptotic power envelope is
//!    tangent at 50% power — and regressing the quasi-differences on each
//!    other. The detrended series is `y_t - z_t' beta_gls`. This engine is
//!    a reusable crate-internal function: the Ng-Perron (2001) M-tests use
//!    the identical detrending step and will call it when they land.
//!
//! 2. **Lag selection** (Perron & Qu 2007, as implemented by arch): when
//!    no fixed lag is given, the ADF lag length is chosen by AIC/BIC/t-stat
//!    on the **OLS**-detrended series (not the GLS-detrended one — OLS
//!    detrending at this step improves finite-sample power) in a regression
//!    with **no deterministics**, all candidates fitted on the common
//!    sample trimmed at `maxlag`. The default `maxlag` is the Schwert rule
//!    `ceil(12 (T/100)^{1/4})` capped at arch's feasibility bound
//!    `(T-1)/2 - 1`.
//!
//! 3. **The ADF regression without deterministics** on the GLS-detrended
//!    series: `dy_t = gamma y_{t-1} + sum b_j dy_{t-j} + e_t`; the
//!    statistic is the OLS t-ratio on `gamma` (nonrobust SEs).
//!
//! P-values and critical values use the response surfaces shipped with
//! arch (`arch.unitroot.critical_values.dfgls`), computed by Kevin
//! Sheppard with the MacKinnon (1994, 2010) response-surface methodology
//! from novel simulations; the constants are transcribed verbatim below
//! with attribution. They are *not* an independently published table — the
//! golden fixtures grade them as "bit-for-bit against arch's surfaces"
//! (see `fixtures/generate_dfgls_fixtures.py`).
//!
//! References: Elliott, Rothenberg & Stock (1996), Econometrica 64(4),
//! 813-836; Ng & Perron (2001), Econometrica 69(6); Perron & Qu (2007),
//! Economics Letters 94(1); MacKinnon (1994, JBES 12(2); 2010, Queen's
//! University working paper 1227).

use tsecon_stats::{ContinuousDist, StdNormal};

use crate::error::DiagError;
use crate::mackinnon::AdfCriticalValues;
use crate::ols::ols_detailed;
use crate::unitroot::{adf_design, AdfLagSelection};
use crate::validate::check_series;

/// Deterministic component removed by GLS detrending before the DF-GLS
/// regression. The no-deterministics case does not exist for DF-GLS: with
/// nothing to estimate, GLS detrending is a no-op and the plain ADF `"n"`
/// case already attains the power envelope.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DfglsTrend {
    /// Constant only (arch `"c"`, the conventional default);
    /// quasi-differencing at `cbar = -7.0`.
    Constant,
    /// Constant and linear trend (arch `"ct"`), for series that may be
    /// stationary around a deterministic trend; `cbar = -13.5`.
    ConstantTrend,
}

impl DfglsTrend {
    /// The ERS (1996) local-alternative constant for this specification.
    fn cbar(self) -> f64 {
        match self {
            DfglsTrend::Constant => -7.0,
            DfglsTrend::ConstantTrend => -13.5,
        }
    }

    /// Number of deterministic columns in the detrending regression.
    fn ntrend(self) -> usize {
        match self {
            DfglsTrend::Constant => 1,
            DfglsTrend::ConstantTrend => 2,
        }
    }
}

/// Result of the DF-GLS unit-root test.
///
/// The null hypothesis is a unit root; the alternative is (level or trend)
/// stationarity, so — as for the ADF test — small p-values speak *for*
/// stationarity.
#[derive(Debug, Clone, PartialEq)]
pub struct DfglsResult {
    /// The tau statistic: the OLS t-ratio on the lagged level in the
    /// trendless ADF regression on the GLS-detrended series.
    pub statistic: f64,
    /// Approximate p-value from arch's DF-GLS response surface
    /// (MacKinnon-1994-style; saturates at 0/1 outside the surface range).
    pub p_value: f64,
    /// The number of lagged differences used in the final regression.
    pub used_lag: usize,
    /// Effective observations in the final regression
    /// (`n - 1 - used_lag`).
    pub nobs: usize,
    /// Finite-sample critical values at `nobs` from arch's DF-GLS `1/n`
    /// response surface.
    pub crit: AdfCriticalValues,
    /// The deterministic specification that was GLS-detrended.
    pub trend: DfglsTrend,
}

// ------------------------------------------------------ detrending engine

/// The deterministic design `z` of the detrending regressions: a column of
/// ones, plus (for the trend case) `t = 1..T` — matching arch
/// `add_trend(nobs, trend)`.
fn trend_columns(n: usize, ntrend: usize) -> Vec<Vec<f64>> {
    let mut z: Vec<Vec<f64>> = Vec::with_capacity(ntrend);
    z.push(vec![1.0; n]);
    if ntrend >= 2 {
        z.push((1..=n).map(|t| t as f64).collect());
    }
    z
}

/// GLS-detrend `y` at the local alternative `rho = 1 + cbar/T` (Elliott,
/// Rothenberg & Stock 1996): quasi-difference both `y` and the
/// deterministics `z` (first observation kept with weight 1, thereafter
/// `x_t - (1 + cbar/T) x_{t-1}`), estimate `beta` by OLS of the
/// quasi-differenced `y` on the quasi-differenced `z`, and return
/// `y_t - z_t' beta`.
///
/// Crate-internal on purpose: this is the shared GLS-detrending engine —
/// the Ng-Perron (2001) M-tests use the identical step (same `cbar`
/// choices) and will reuse it when they land.
///
/// # Errors
///
/// [`DiagError::SingularDesign`] / [`DiagError::NumericalBreakdown`] when
/// the detrending regression is degenerate (an exactly deterministic
/// series).
pub(crate) fn gls_detrend(
    y: &[f64],
    ntrend: usize,
    cbar: f64,
    what: &'static str,
) -> Result<Vec<f64>, DiagError> {
    let n = y.len();
    let ct = cbar / n as f64;
    let z = trend_columns(n, ntrend);

    // Quasi-difference: row 0 unchanged, then x_t - (1 + ct) x_{t-1}.
    let quasi = |col: &[f64]| -> Vec<f64> {
        let mut out = Vec::with_capacity(n);
        out.push(col[0]);
        for t in 1..n {
            out.push(col[t] - (1.0 + ct) * col[t - 1]);
        }
        out
    };
    let zq: Vec<Vec<f64>> = z.iter().map(|c| quasi(c)).collect();
    let yq = quasi(y);

    let fit = ols_detailed(&zq, &yq, what)?;
    let beta = &fit.params;

    Ok((0..n)
        .map(|t| {
            let mut det = 0.0;
            for (b, col) in beta.iter().zip(z.iter()) {
                det += b * col[t];
            }
            y[t] - det
        })
        .collect())
}

/// OLS-detrend `y` on the same deterministics (used only for the
/// Perron-Qu lag-selection step).
fn ols_detrend(y: &[f64], ntrend: usize, what: &'static str) -> Result<Vec<f64>, DiagError> {
    let n = y.len();
    let z = trend_columns(n, ntrend);
    let fit = ols_detailed(&z, y, what)?;
    let beta = &fit.params;
    Ok((0..n)
        .map(|t| {
            let mut det = 0.0;
            for (b, col) in beta.iter().zip(z.iter()) {
                det += b * col[t];
            }
            y[t] - det
        })
        .collect())
}

// -------------------------------------------------------- lag selection

/// arch's t-stat stopping threshold (`norm.ppf(0.95)` hard-coded).
const DFGLS_TSTAT_STOP: f64 = 1.6448536269514722;

/// arch `_df_select_lags(y, trend="n", ...)`: choose the ADF lag length on
/// `y` (already detrended; no deterministics in the candidate regressions),
/// all candidates fitted on the common sample trimmed at `maxlag`.
///
/// Matches arch's information criteria exactly: with
/// `sigma2 = SSR / rows` (the MLE variance on the `rows`-observation
/// common sample) and `llf = -rows/2 [ln(2 pi) + ln(sigma2) + 1]`,
/// `AIC = -2 llf + 2 lag` and `BIC = -2 llf + ln(rows) lag` — the penalty
/// counts only the lag terms, which shifts every candidate by the same
/// constant relative to the statsmodels convention and therefore selects
/// the same lag. `t-stat` walks down from `maxlag` and keeps the first lag
/// whose last-lag t-ratio (computed with the MLE variance, as arch does)
/// clears `1.645`; 0 if none does.
fn select_lags(
    y: &[f64],
    maxlag: usize,
    method: AdfLagSelection,
    what: &'static str,
) -> Result<usize, DiagError> {
    let n = y.len();
    let rows = n - 1 - maxlag; // guarded by the caller
    let rows_f = rows as f64;
    let (cols, dy) = adf_design(y, maxlag, 0, false);

    if let AdfLagSelection::TStat(_) = method {
        for lag in (1..=maxlag).rev() {
            let fit = ols_detailed(&cols[..1 + lag], &dy, what)?;
            // ols_detailed t-ratios use s^2 = SSR/(rows - k); arch's
            // selection t-ratio uses the MLE SSR/rows — rescale.
            let k = 1 + lag;
            let t_arch = fit.t_values[lag] * (rows_f / (rows_f - k as f64)).sqrt();
            if t_arch.abs() >= DFGLS_TSTAT_STOP {
                return Ok(lag);
            }
        }
        return Ok(0);
    }

    let penalty = match method {
        AdfLagSelection::Bic(_) => rows_f.ln(),
        _ => 2.0,
    };
    let mut best_lag = 0usize;
    let mut best_ic = f64::INFINITY;
    for lag in 0..=maxlag {
        let fit = ols_detailed(&cols[..1 + lag], &dy, what)?;
        let sigma2 = fit.ssr / rows_f;
        let ic = rows_f * ((2.0 * core::f64::consts::PI).ln() + sigma2.ln() + 1.0)
            + penalty * lag as f64;
        if ic < best_ic {
            best_ic = ic;
            best_lag = lag;
        }
    }
    Ok(best_lag)
}

// ------------------------------------------------------ response surfaces
//
// Transcribed verbatim (full-precision doubles) from arch 8.0.0,
// `arch/unitroot/critical_values/dfgls.py` (Kevin Sheppard, NCSA license):
// response surfaces for the DF-GLS statistic computed "using the
// methodology of MacKinnon (1994) and (2010) simulation". The p-value map
// is Phi(poly(stat)) with a quadratic small-p branch below `tau_star` and
// a cubic large-p branch above it, saturating at 0/1 outside
// [tau_min, tau_max]; critical values are 1/n polynomials in nobs.
// `fixtures/dfgls.json` re-exports the same constants (provenance block)
// and pins this transcription bit-for-bit in the tests below.

/// DF-GLS response surface for one detrending specification.
struct DfglsSurface {
    /// Boundary between the small-p and large-p polynomial regions.
    tau_star: f64,
    /// Below this the p-value saturates at 0.
    tau_min: f64,
    /// Above this the p-value saturates at 1.
    tau_max: f64,
    /// Small-p (left tail) quadratic `[c0, c1, c2]`.
    small_p: [f64; 3],
    /// Large-p cubic `[d0, d1, d2, d3]`.
    large_p: [f64; 4],
    /// `1/n` critical-value polynomials, rows = 1% / 5% / 10%, each
    /// `[b0, b1, b2, b3]` for `b0 + b1/n + b2/n^2 + b3/n^3`.
    cv: [[f64; 4]; 3],
}

/// arch `dfgls.py` constants, `"c"` (constant) case.
// Literals are transcribed byte-for-byte from arch's source (and re-checked
// against the fixture provenance block below), so clippy's shorter
// round-trip spellings are deliberately not applied.
#[allow(clippy::excessive_precision)]
const DFGLS_C: DfglsSurface = DfglsSurface {
    tau_star: -0.4795076091714674,
    tau_min: -17.561302895074206,
    tau_max: 13.365361509140614,
    small_p: [0.67422739, 1.25475826, 0.03572509],
    large_p: [0.50612497, 0.98305664, -0.05648525, 0.00140875],
    cv: [
        [-2.56781793e0, -2.05575392e1, 1.82727674e2, -1.77866664e3],
        [-1.94363325e0, -2.17272746e1, 2.60815068e2, -2.26914916e3],
        [-1.61998241e0, -2.32734708e1, 3.06474378e2, -2.57483557e3],
    ],
};

/// arch `dfgls.py` constants, `"ct"` (constant + trend) case.
#[allow(clippy::excessive_precision)]
const DFGLS_CT: DfglsSurface = DfglsSurface {
    tau_star: -2.1960404365401298,
    tau_min: -13.681153542634465,
    tau_max: 8.73743383728356,
    small_p: [2.38767685, 1.57454737, 0.05754439],
    large_p: [2.60561421, 1.67850224, 0.0373599, -0.01017936],
    cv: [
        [-3.40689134, -21.69971242, 27.26295939, -816.84404772],
        [-2.84677178, -19.69109364, 84.7664136, -799.40722401],
        [-2.55890707, -19.42621991, 116.53759752, -840.31342847],
    ],
};

fn dfgls_surface(trend: DfglsTrend) -> &'static DfglsSurface {
    match trend {
        DfglsTrend::Constant => &DFGLS_C,
        DfglsTrend::ConstantTrend => &DFGLS_CT,
    }
}

/// Horner evaluation from the highest-degree coefficient down, matching
/// `numpy.polyval` on the reversed coefficient vector (as in
/// [`crate::mackinnon`]).
fn polyval_ascending(coeffs: &[f64], x: f64) -> f64 {
    let mut acc = 0.0;
    for &c in coeffs.iter().rev() {
        acc = acc * x + c;
    }
    acc
}

/// Approximate p-value for a DF-GLS statistic, matching
/// `arch.unitroot.unitroot.mackinnonp(stat, regression, dist_type="dfgls")`.
pub(crate) fn dfgls_p(stat: f64, trend: DfglsTrend) -> f64 {
    let s = dfgls_surface(trend);
    if stat > s.tau_max {
        return 1.0;
    }
    if stat < s.tau_min {
        return 0.0;
    }
    let g = if stat <= s.tau_star {
        polyval_ascending(&s.small_p, stat)
    } else {
        polyval_ascending(&s.large_p, stat)
    };
    StdNormal.cdf(g)
}

/// Finite-sample DF-GLS critical values at the 1% / 5% / 10% levels,
/// matching `mackinnoncrit(regression, nobs, dist_type="dfgls")`.
pub(crate) fn dfgls_crit(trend: DfglsTrend, nobs: usize) -> AdfCriticalValues {
    let s = dfgls_surface(trend);
    let x = 1.0 / nobs as f64;
    AdfCriticalValues {
        pct1: polyval_ascending(&s.cv[0], x),
        pct5: polyval_ascending(&s.cv[1], x),
        pct10: polyval_ascending(&s.cv[2], x),
    }
}

// ---------------------------------------------------------------- dfgls

/// DF-GLS unit-root test (Elliott, Rothenberg & Stock 1996), matching
/// `arch.unitroot.DFGLS`.
///
/// GLS-detrends `y` at the ERS local alternative ([`gls_detrend`];
/// `cbar = -7.0` for [`DfglsTrend::Constant`], `-13.5` for
/// [`DfglsTrend::ConstantTrend`]), then runs the ADF regression *without
/// deterministics* on the detrended series:
///
/// ```text
/// dyd_t = gamma yd_{t-1} + sum_{j=1..p} b_j dyd_{t-j} + e_t
/// tau   = gamma_hat / se(gamma_hat)      (OLS, nonrobust SEs)
/// ```
///
/// with `H0: gamma = 0` (unit root) against `H1: gamma < 0`. Automatic lag
/// selection follows Perron & Qu (2007) as arch implements it: the
/// AIC/BIC/t-stat search runs on the **OLS**-detrended series with no
/// deterministics, candidates fitted on the common sample trimmed at
/// `maxlag` (default: Schwert's `ceil(12 (T/100)^{1/4})` capped at
/// `(T-1)/2 - 1`); the final regression is refit at the chosen lag on the
/// longest available sample. P-values and critical values come from the
/// transcribed arch DF-GLS response surfaces ([`module docs`](self)).
///
/// # Errors
///
/// * [`DiagError::NonFinite`] if the series contains NaN or infinities.
/// * [`DiagError::ConstantSeries`] if the series is constant.
/// * [`DiagError::SeriesTooShort`] if too few observations remain after
///   differencing and trimming for the requested specification (the arch
///   minimum `3 + ntrend + lags`, plus one residual degree of freedom in
///   the final regression).
/// * [`DiagError::SingularDesign`] / [`DiagError::NumericalBreakdown`] for
///   (near-)deterministic series — e.g. an exact linear trend — whose
///   detrending or lag design is collinear or fits exactly.
pub fn dfgls(
    y: &[f64],
    trend: DfglsTrend,
    lags: AdfLagSelection,
) -> Result<DfglsResult, DiagError> {
    const WHAT: &str = "dfgls";
    let ntrend = trend.ntrend();
    // arch's minimum: 3 + ntrend + (fixed lags, if any); the regression
    // additionally needs a residual degree of freedom (n >= 2 lags + 3).
    let fixed = match lags {
        AdfLagSelection::Fixed(l) => l,
        _ => 0,
    };
    let min_n = (3 + ntrend + fixed).max(2 * fixed + 3);
    let n = check_series(y, min_n, WHAT)?;
    if y.iter().all(|&v| v == y[0]) {
        return Err(DiagError::ConstantSeries { what: WHAT });
    }

    // 1. GLS detrend at the ERS local alternative.
    let y_gls = gls_detrend(y, ntrend, trend.cbar(), WHAT)?;
    // A detrended series that is zero to machine precision relative to the
    // original scale means the deterministics explain y exactly (e.g. an
    // exact linear trend): there is no stochastic variation to test.
    let scale = y.iter().fold(0.0_f64, |a, &v| a.max(v.abs())).max(1.0);
    if y_gls.iter().all(|&v| v.abs() <= 1e-12 * scale) {
        return Err(DiagError::NumericalBreakdown { what: WHAT });
    }

    // 2. Lag length: fixed, or Perron-Qu selection on the OLS-detrended
    //    series with no deterministics (arch `_df_select_lags(., "n", .)`).
    let used_lag = match lags {
        AdfLagSelection::Fixed(l) => l,
        AdfLagSelection::Aic(user) | AdfLagSelection::Bic(user) | AdfLagSelection::TStat(user) => {
            let maxlag = match user {
                Some(m) => m,
                None => {
                    // Schwert rule capped at arch's bound (T-1)/2 - 1.
                    let max_max = ((n - 1) / 2).saturating_sub(1);
                    let schwert = (12.0 * (n as f64 / 100.0).powf(0.25)).ceil() as usize;
                    schwert.min(max_max)
                }
            };
            // Common trimmed sample must leave a residual dof at maxlag.
            let rows = n.saturating_sub(1 + maxlag);
            if rows < maxlag + 2 {
                return Err(DiagError::SeriesTooShort {
                    what: WHAT,
                    n,
                    needed: 2 * maxlag + 3,
                });
            }
            let y_ols = ols_detrend(y, ntrend, WHAT)?;
            select_lags(&y_ols, maxlag, lags, WHAT)?
        }
    };

    // 3. Trendless ADF regression on the GLS-detrended series, longest
    //    available sample for the chosen lag.
    let rows = n - 1 - used_lag;
    if rows < used_lag + 2 {
        return Err(DiagError::SeriesTooShort {
            what: WHAT,
            n,
            needed: 2 * used_lag + 3,
        });
    }
    let (cols, dy) = adf_design(&y_gls, used_lag, 0, false);
    let fit = ols_detailed(&cols, &dy, WHAT)?;
    let statistic = fit.t_values[0];

    Ok(DfglsResult {
        statistic,
        p_value: dfgls_p(statistic, trend),
        used_lag,
        nobs: rows,
        crit: dfgls_crit(trend, rows),
        trend,
    })
}

#[cfg(test)]
mod tests {
    //! Directly pin the transcribed DF-GLS response surfaces against
    //! `fixtures/dfgls.json` (arch 8.0.0). The surface maps are
    //! `pub(crate)`, unreachable from an integration test, so the
    //! transcription is pinned here in-crate (the [`crate::mackinnon_ext`]
    //! precedent).

    use super::*;
    use serde_json::Value;

    fn fixture() -> Value {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../fixtures/dfgls.json");
        let text = std::fs::read_to_string(path).expect("fixture file readable");
        serde_json::from_str(&text).expect("fixture is valid JSON")
    }

    fn f64s(v: &Value) -> Vec<f64> {
        v.as_array()
            .expect("array")
            .iter()
            .map(|x| x.as_f64().expect("number"))
            .collect()
    }

    fn trend(code: &str) -> DfglsTrend {
        match code {
            "c" => DfglsTrend::Constant,
            "ct" => DfglsTrend::ConstantTrend,
            other => panic!("unknown trend {other:?}"),
        }
    }

    /// Combined abs+rel p-value comparison (numpy `allclose`): moderate
    /// p-values pin relatively at 1e-8; the absolute floor covers the deep
    /// tail where tsecon-stats and scipy normal CDFs differ.
    fn assert_pclose(actual: f64, expected: f64, ctx: &str) {
        let tol = 1e-12 + 1e-8 * expected.abs();
        assert!(
            (actual - expected).abs() <= tol,
            "{ctx}: actual {actual}, expected {expected}, |diff| {:e} > {tol:e}",
            (actual - expected).abs()
        );
    }

    fn assert_rel(actual: f64, expected: f64, rtol: f64, ctx: &str) {
        let rel = if expected == 0.0 {
            actual.abs()
        } else {
            ((actual - expected) / expected).abs()
        };
        assert!(
            rel <= rtol,
            "{ctx}: actual {actual}, expected {expected}, rel {rel:e}"
        );
    }

    #[test]
    fn dfgls_p_map_matches_arch() {
        let fx = fixture();
        for code in ["c", "ct"] {
            let block = &fx["dfgls_map"][code];
            let stats = f64s(&block["stat_grid"]);
            let pvals = f64s(&block["pvalues"]);
            assert_eq!(stats.len(), pvals.len());
            for (&s, &p) in stats.iter().zip(&pvals) {
                assert_pclose(dfgls_p(s, trend(code)), p, &format!("dfgls p[{code}]({s})"));
            }
        }
    }

    #[test]
    fn dfgls_crit_map_matches_arch() {
        let fx = fixture();
        for code in ["c", "ct"] {
            let block = &fx["dfgls_map"][code]["crit"];
            for (nobs_str, cv_expected) in block.as_object().expect("crit map") {
                let nobs: usize = nobs_str.parse().expect("nobs key");
                let expected = f64s(cv_expected);
                let cv = dfgls_crit(trend(code), nobs);
                assert_rel(
                    cv.pct1,
                    expected[0],
                    1e-12,
                    &format!("crit[{code}/{nobs}] 1%"),
                );
                assert_rel(
                    cv.pct5,
                    expected[1],
                    1e-12,
                    &format!("crit[{code}/{nobs}] 5%"),
                );
                assert_rel(
                    cv.pct10,
                    expected[2],
                    1e-12,
                    &format!("crit[{code}/{nobs}] 10%"),
                );
            }
        }
    }

    #[test]
    fn transcribed_constants_match_fixture_provenance() {
        // The generator re-exports arch's raw constants; the transcription
        // above must be the same doubles, bit for bit.
        let fx = fixture();
        let meta = &fx["_meta"];
        for (code, s) in [("c", &DFGLS_C), ("ct", &DFGLS_CT)] {
            assert_eq!(meta["dfgls_tau_star"][code].as_f64().unwrap(), s.tau_star);
            assert_eq!(meta["dfgls_tau_min"][code].as_f64().unwrap(), s.tau_min);
            assert_eq!(meta["dfgls_tau_max"][code].as_f64().unwrap(), s.tau_max);
            assert_eq!(f64s(&meta["dfgls_small_p"][code]), s.small_p.to_vec());
            assert_eq!(f64s(&meta["dfgls_large_p"][code]), s.large_p.to_vec());
            let cv: Vec<Vec<f64>> = meta["dfgls_cv_approx"][code]
                .as_array()
                .unwrap()
                .iter()
                .map(f64s)
                .collect();
            for (row, expected) in s.cv.iter().zip(&cv) {
                assert_eq!(&row.to_vec(), expected);
            }
        }
    }
}
