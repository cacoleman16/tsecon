//! Ng-Perron (2001) M unit-root tests: MZa, MZt, MSB, and MPT with GLS
//! detrending, the autoregressive spectral density estimator at frequency
//! zero, and MAIC lag selection — the modified tests with good size *and*
//! good power that Ng & Perron built on Perron & Ng (1996) and Elliott,
//! Rothenberg & Stock (1996).
//!
//! The pipeline, per Ng & Perron (2001, Econometrica 69(6), 1519-1554):
//!
//! 1. **GLS detrending** at the ERS local alternative — the *identical*
//!    engine the DF-GLS test uses ([`crate::dfgls`]'s `gls_detrend`, called
//!    with the same `cbar = -7.0` (constant) / `-13.5` (constant + trend)),
//!    so the detrended series is bit-for-bit the one `dfgls` regresses on.
//! 2. **MAIC lag selection** (section 4 of the paper): the ADF regression
//!    on the GLS-detrended series with *no deterministics*,
//!    `dy_t = b0 y_{t-1} + sum_{j=1..k} b_j dy_{t-j} + e_t`, is fitted for
//!    every `k` in `0..=kmax` on the common sample trimmed at `kmax`, and
//!    `k` minimizes `MAIC(k) = ln(sigma2_k) + 2 (tau(k) + k) / (T - kmax)`
//!    with `tau(k) = b0_hat^2 sum y_{t-1}^2 / sigma2_k` and
//!    `sigma2_k = SSR_k / (T - kmax)`. The data-dependent `tau(k)` penalty
//!    is the paper's remedy for the severe size distortion that AIC/BIC lag
//!    choices produce under a large negative MA root.
//! 3. **The autoregressive spectral density estimator at frequency zero**
//!    (equation (4) of the paper): refit the same regression at the chosen
//!    `k` on the longest available sample and set
//!    `s2_AR = sigma2_e / (1 - b(1))^2` with `sigma2_e = SSR / (T - k)` and
//!    `b(1) = sum_{j=1..k} b_j_hat`.
//! 4. **The M statistics** on the full detrended series `yd_1..yd_T`
//!    (`T = n`, the series length), with
//!    `kappa = T^{-2} sum_{t=1}^{T-1} yd_t^2` (the sum of squared *lagged*
//!    levels) and `w = T^{-1} yd_T^2`:
//!
//!    ```text
//!    MZa = (w - s2_AR) / (2 kappa)
//!    MSB = sqrt(kappa / s2_AR)
//!    MZt = (w - s2_AR) / (2 sqrt(kappa s2_AR))   ( = MZa * MSB exactly )
//!    MPT = (cbar^2 kappa - cbar w) / s2_AR             (constant)
//!        = (cbar^2 kappa + (1 - cbar) w) / s2_AR       (constant + trend)
//!    ```
//!
//!    All four reject the unit-root null for *small* values (MZa and MZt
//!    are negative under the alternative; MSB and MPT are positive but
//!    small). `MZt = MZa * MSB` is an exact algebraic identity and is
//!    enforced as an internal invariant test.
//!
//! **Critical values** are transcribed from Ng & Perron (2001), Table 1
//! (asymptotic, GLS-detrended case) — the same table EViews ships — and
//! cross-checked against the independent transcription in Nazlioglu's
//! GAUSS `tspdlib` (`gls.src`, `MGLS`). No p-value response surface for
//! the M tests exists in any reference this library allows itself, and
//! none is fabricated: the result reports the four statistics and the
//! 1/5/10% critical values, statistic-only (the Phillips-Ouliaris Za
//! precedent).
//!
//! **Validation** (no runnable independent implementation exists anywhere:
//! statsmodels 0.14.6 and arch 8.0.0 do not implement the M tests, and no
//! CRAN package does — `urca::ur.ers` stops at DF-GLS and the ERS P-test):
//! the transcribed Table 1, seeded Monte-Carlo size at the asymptotic
//! critical values on null data, power-ordering checks, an independent
//! NumPy re-implementation re-pinned through the Python binding, and the
//! bitwise cross-pin to the shared `dfgls` detrending engine. See
//! `tests/ng_perron_properties.rs` and the validation matrix.

use crate::dfgls::{gls_detrend, DfglsTrend};
use crate::error::DiagError;
use crate::mackinnon::AdfCriticalValues;
use crate::ols::ols_detailed;
use crate::unitroot::adf_design;
use crate::validate::check_series;

/// How the ADF lag length `k` entering the autoregressive spectral density
/// estimator is chosen.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NgPerronLagSelection {
    /// Use exactly this many lagged differences.
    Fixed(usize),
    /// Ng-Perron (2001) MAIC on the GLS-detrended series (the paper's
    /// recommendation and the default), searching `0..=max_lag`. `None`
    /// uses Schwert's `ceil(12 (T/100)^{1/4})` capped at `(T-1)/2 - 1`
    /// (the same default cap as [`crate::dfgls`]; the paper's own
    /// simulations truncate rather than round up — pass an explicit
    /// maximum to reproduce a specific study).
    Maic(Option<usize>),
}

/// Asymptotic 1/5/10% critical values for the four M statistics
/// (Ng & Perron 2001, Table 1, GLS-detrended case). Every test rejects
/// for *small* values of its statistic.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NgPerronCriticalValues {
    /// Critical values for MZa (negative; reject below).
    pub mza: AdfCriticalValues,
    /// Critical values for MZt (negative; reject below).
    pub mzt: AdfCriticalValues,
    /// Critical values for MSB (positive; reject below).
    pub msb: AdfCriticalValues,
    /// Critical values for MPT (positive; reject below).
    pub mpt: AdfCriticalValues,
}

/// Result of the Ng-Perron (2001) M unit-root tests.
///
/// The null hypothesis is a unit root; the alternative is (level or
/// trend) stationarity. All four statistics reject the null when they are
/// *small* (below the critical value): MZa and MZt are one-sided to the
/// left like the ADF tau, and MSB and MPT are positive statistics that
/// shrink under the alternative. There are no p-values: no published
/// response surface exists for the M tests, so compare each statistic
/// against its own critical values in [`NgPerronResult::crit`].
#[derive(Debug, Clone, PartialEq)]
pub struct NgPerronResult {
    /// The MZa statistic — the modified, GLS-detrended Phillips Z_alpha.
    pub mza: f64,
    /// The MZt statistic — the modified Z_tau; equals `mza * msb` exactly.
    pub mzt: f64,
    /// The MSB statistic (Sargan-Bhargava, modified): `sqrt(kappa/s2_ar)`.
    pub msb: f64,
    /// The MPT statistic — the modified ERS point-optimal statistic.
    pub mpt: f64,
    /// The lag length used in the spectral-density regression (MAIC-chosen
    /// or fixed).
    pub used_lag: usize,
    /// Effective observations in that regression (`n - 1 - used_lag`).
    pub nobs: usize,
    /// The autoregressive spectral density estimate at frequency zero,
    /// `s2_AR = sigma2_e / (1 - b(1))^2`.
    pub s2_ar: f64,
    /// Ng-Perron (2001) Table 1 asymptotic critical values for this trend
    /// specification.
    pub crit: NgPerronCriticalValues,
    /// The deterministic specification that was GLS-detrended (shared with
    /// [`crate::dfgls`]).
    pub trend: DfglsTrend,
}

// ---------------------------------------------------- Table 1 transcription
//
// Transcribed from Ng & Perron (2001), Econometrica 69(6), Table 1
// ("asymptotic critical values", GLS-detrended case): rows MZa/MZt/MSB/MPT,
// columns 1% / 5% / 10%. The same values are shipped by EViews and by the
// GAUSS tspdlib (`gls.src` proc `MGLS`, Nazlioglu), which was used as an
// independent cross-check of this transcription. The table is asymptotic
// only — the paper publishes no finite-sample surface.

/// Table 1, `zt = {1}` (constant; `cbar = -7`).
const NP_CRIT_C: NgPerronCriticalValues = NgPerronCriticalValues {
    mza: AdfCriticalValues {
        pct1: -13.8,
        pct5: -8.1,
        pct10: -5.7,
    },
    mzt: AdfCriticalValues {
        pct1: -2.58,
        pct5: -1.98,
        pct10: -1.62,
    },
    msb: AdfCriticalValues {
        pct1: 0.174,
        pct5: 0.233,
        pct10: 0.275,
    },
    mpt: AdfCriticalValues {
        pct1: 1.78,
        pct5: 3.17,
        pct10: 4.45,
    },
};

/// Table 1, `zt = {1, t}` (constant + trend; `cbar = -13.5`).
const NP_CRIT_CT: NgPerronCriticalValues = NgPerronCriticalValues {
    mza: AdfCriticalValues {
        pct1: -23.8,
        pct5: -17.3,
        pct10: -14.2,
    },
    mzt: AdfCriticalValues {
        pct1: -3.42,
        pct5: -2.91,
        pct10: -2.62,
    },
    msb: AdfCriticalValues {
        pct1: 0.143,
        pct5: 0.168,
        pct10: 0.185,
    },
    mpt: AdfCriticalValues {
        pct1: 4.03,
        pct5: 5.48,
        pct10: 6.67,
    },
};

/// Ng-Perron (2001, Table 1) asymptotic critical values for the M tests
/// under GLS detrending. Every statistic rejects the unit-root null when
/// it falls *below* its critical value.
pub fn ng_perron_crit(trend: DfglsTrend) -> NgPerronCriticalValues {
    match trend {
        DfglsTrend::Constant => NP_CRIT_C,
        DfglsTrend::ConstantTrend => NP_CRIT_CT,
    }
}

// ------------------------------------------------------------ MAIC search

/// Ng-Perron (2001, section 4) MAIC lag selection on the (GLS-detrended)
/// series `y`: every candidate `k` in `0..=maxlag` is fitted on the common
/// sample trimmed at `maxlag`, and the criterion is
/// `ln(sigma2_k) + 2 (tau(k) + k) / rows` with
/// `sigma2_k = SSR_k / rows`, `tau(k) = b0^2 sum y_{t-1}^2 / sigma2_k`,
/// and `rows = n - 1 - maxlag` (the common-sample size). The `tau(k)` term
/// penalizes the small-`k` fits whose `b0` is far from zero — exactly the
/// configurations that arise under a negative MA root and wreck the size
/// of AIC/BIC-selected tests.
fn select_maic(y: &[f64], maxlag: usize, what: &'static str) -> Result<usize, DiagError> {
    let n = y.len();
    let rows = n - 1 - maxlag; // guarded by the caller
    let rows_f = rows as f64;
    let (cols, dy) = adf_design(y, maxlag, 0, false);
    // Sum of squared lagged levels over the common sample: the level
    // column of the trimmed design.
    let sum_lev2: f64 = cols[0].iter().map(|&v| v * v).sum();

    let mut best_lag = 0usize;
    let mut best_ic = f64::INFINITY;
    for k in 0..=maxlag {
        let fit = ols_detailed(&cols[..1 + k], &dy, what)?;
        let sigma2 = fit.ssr / rows_f;
        let b0 = fit.params[0];
        let tau = b0 * b0 * sum_lev2 / sigma2;
        let ic = sigma2.ln() + 2.0 * (tau + k as f64) / rows_f;
        if ic < best_ic {
            best_ic = ic;
            best_lag = k;
        }
    }
    Ok(best_lag)
}

// ------------------------------------------------------------- ng_perron

/// Ng-Perron (2001) M unit-root tests (MZa, MZt, MSB, MPT) with GLS
/// detrending, MAIC lag selection, and the autoregressive spectral density
/// estimator at frequency zero.
///
/// GLS-detrends `y` through the *same engine and constants* as
/// [`crate::dfgls`] (`cbar = -7.0` for [`DfglsTrend::Constant`], `-13.5`
/// for [`DfglsTrend::ConstantTrend`]), selects the ADF lag length by MAIC
/// on the detrended series (or uses a fixed lag), estimates the spectral
/// density at frequency zero from the autoregression, and forms the four M
/// statistics ([module docs](self)). All four reject the unit-root null
/// for *small* values; compare against the Table 1 critical values in the
/// result — there are no p-values, because no published response surface
/// exists for these tests.
///
/// This is the test battery to reach for when a large negative MA
/// component is suspected (e.g. an over-differenced series): the MAIC lag
/// rule is the standard remedy for the size distortion that wrecks
/// ADF/DF-GLS with AIC/BIC selection there, while GLS detrending keeps the
/// near-optimal local power of DF-GLS.
///
/// # Errors
///
/// * [`DiagError::NonFinite`] if the series contains NaN or infinities.
/// * [`DiagError::ConstantSeries`] if the series is constant.
/// * [`DiagError::SeriesTooShort`] if too few observations remain after
///   differencing and trimming for the requested specification.
/// * [`DiagError::SingularDesign`] / [`DiagError::NumericalBreakdown`] for
///   (near-)deterministic series — e.g. an exact linear trend — whose
///   detrending or lag design is collinear or fits exactly, and for the
///   degenerate spectral cases (`b(1)` at 1, or a detrended series with no
///   variation before its last observation).
pub fn ng_perron(
    y: &[f64],
    trend: DfglsTrend,
    lags: NgPerronLagSelection,
) -> Result<NgPerronResult, DiagError> {
    const WHAT: &str = "ng_perron";
    let ntrend = trend.ntrend();
    let fixed = match lags {
        NgPerronLagSelection::Fixed(l) => l,
        NgPerronLagSelection::Maic(_) => 0,
    };
    // Same floor as dfgls (whose engine this reuses), plus one residual
    // degree of freedom in the k-lag regression.
    let min_n = (3 + ntrend + fixed).max(2 * fixed + 3);
    let n = check_series(y, min_n, WHAT)?;
    if y.iter().all(|&v| v == y[0]) {
        return Err(DiagError::ConstantSeries { what: WHAT });
    }

    // 1. GLS detrend at the ERS local alternative — the shared dfgls
    //    engine, same cbar, same deterministics: bit-identical output.
    let y_gls = gls_detrend(y, ntrend, trend.cbar(), WHAT)?;
    let scale = y.iter().fold(0.0_f64, |a, &v| a.max(v.abs())).max(1.0);
    if y_gls.iter().all(|&v| v.abs() <= 1e-12 * scale) {
        return Err(DiagError::NumericalBreakdown { what: WHAT });
    }

    // 2. Lag length: fixed, or MAIC on the GLS-detrended series (the
    //    paper's own rule — not the Perron-Qu OLS-detrended variant that
    //    dfgls uses for its AIC/BIC selection).
    let used_lag = match lags {
        NgPerronLagSelection::Fixed(l) => l,
        NgPerronLagSelection::Maic(user) => {
            let maxlag = match user {
                Some(m) => m,
                None => {
                    let max_max = ((n - 1) / 2).saturating_sub(1);
                    let schwert = (12.0 * (n as f64 / 100.0).powf(0.25)).ceil() as usize;
                    schwert.min(max_max)
                }
            };
            // The common trimmed sample must keep a residual degree of
            // freedom at k = maxlag.
            let rows = n.saturating_sub(1 + maxlag);
            if rows < maxlag + 2 {
                return Err(DiagError::SeriesTooShort {
                    what: WHAT,
                    n,
                    needed: 2 * maxlag + 3,
                });
            }
            select_maic(&y_gls, maxlag, WHAT)?
        }
    };

    // 3. Autoregressive spectral density at frequency zero: refit at the
    //    chosen lag on the longest available sample.
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
    let sigma2_e = fit.ssr / rows as f64;
    let b1: f64 = fit.params[1..].iter().sum();
    let denom = 1.0 - b1;
    let s2_ar = sigma2_e / (denom * denom);
    if !(s2_ar.is_finite() && s2_ar > 0.0) {
        // b(1) numerically at 1: the implied AR long-run variance blows
        // up; the autoregression itself says "unit root in the errors".
        return Err(DiagError::NumericalBreakdown { what: WHAT });
    }

    // 4. The M statistics on the full detrended series.
    let nf = n as f64;
    let kappa = y_gls[..n - 1].iter().map(|&v| v * v).sum::<f64>() / (nf * nf);
    if !(kappa.is_finite() && kappa > 0.0) {
        return Err(DiagError::NumericalBreakdown { what: WHAT });
    }
    let w = y_gls[n - 1] * y_gls[n - 1] / nf;
    let mza = (w - s2_ar) / (2.0 * kappa);
    let msb = (kappa / s2_ar).sqrt();
    let mzt = (w - s2_ar) / (2.0 * (kappa * s2_ar).sqrt());
    let cbar = trend.cbar();
    let mpt = match trend {
        DfglsTrend::Constant => (cbar * cbar * kappa - cbar * w) / s2_ar,
        DfglsTrend::ConstantTrend => (cbar * cbar * kappa + (1.0 - cbar) * w) / s2_ar,
    };

    Ok(NgPerronResult {
        mza,
        mzt,
        msb,
        mpt,
        used_lag,
        nobs: rows,
        s2_ar,
        crit: ng_perron_crit(trend),
        trend,
    })
}

#[cfg(test)]
mod tests {
    //! In-crate pins that need `pub(crate)` internals: the bitwise
    //! cross-pin to the shared dfgls detrending engine, and the Table 1
    //! transcription's internal consistency. The statistical validation
    //! (Monte-Carlo size/power, invariances, the error surface) lives in
    //! `tests/ng_perron_properties.rs` on the public API.

    use super::*;

    /// The dfgls_properties LCG (Knuth MMIX), kept in sync so the two test
    /// suites exercise the same kind of series.
    struct Lcg(u64);

    impl Lcg {
        fn uniform(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }

        fn normalish(&mut self) -> f64 {
            (0..12).map(|_| self.uniform()).sum::<f64>() - 6.0
        }
    }

    fn walk(n: usize, seed: u64) -> Vec<f64> {
        let mut rng = Lcg(seed);
        let mut acc = 0.0;
        (0..n)
            .map(|_| {
                acc += rng.normalish();
                acc
            })
            .collect()
    }

    /// The four M statistics recomputed *from scratch* off the shared
    /// dfgls detrending engine — `gls_detrend` called with `DfglsTrend`'s
    /// own `cbar()`/`ntrend()`, i.e. exactly what `dfgls` runs internally
    /// on the same inputs — must equal `ng_perron`'s output bit for bit.
    /// This pins the wiring: the detrended series entering the M tests is
    /// the dfgls one, not a fork.
    #[test]
    fn m_stats_recomputed_bitwise_from_the_shared_dfgls_detrending() {
        for (seed, trend) in [
            (11_u64, DfglsTrend::Constant),
            (11, DfglsTrend::ConstantTrend),
            (47, DfglsTrend::Constant),
            (47, DfglsTrend::ConstantTrend),
        ] {
            let y = walk(180, seed);
            for k in [0usize, 3] {
                let r = ng_perron(&y, trend, NgPerronLagSelection::Fixed(k)).unwrap();

                // Reference path: the dfgls engine, then the documented
                // formulas, using the same crate primitives.
                let yd = gls_detrend(&y, trend.ntrend(), trend.cbar(), "test").unwrap();
                let n = yd.len();
                let (cols, dy) = adf_design(&yd, k, 0, false);
                let fit = ols_detailed(&cols, &dy, "test").unwrap();
                let sigma2_e = fit.ssr / (n - 1 - k) as f64;
                let b1: f64 = fit.params[1..].iter().sum();
                let s2_ar = sigma2_e / ((1.0 - b1) * (1.0 - b1));
                let nf = n as f64;
                let kappa = yd[..n - 1].iter().map(|&v| v * v).sum::<f64>() / (nf * nf);
                let w = yd[n - 1] * yd[n - 1] / nf;
                let cbar = trend.cbar();

                assert_eq!(r.s2_ar, s2_ar, "s2_ar seed {seed} {trend:?} k {k}");
                assert_eq!(
                    r.mza,
                    (w - s2_ar) / (2.0 * kappa),
                    "mza seed {seed} {trend:?} k {k}"
                );
                assert_eq!(
                    r.msb,
                    (kappa / s2_ar).sqrt(),
                    "msb seed {seed} {trend:?} k {k}"
                );
                assert_eq!(
                    r.mzt,
                    (w - s2_ar) / (2.0 * (kappa * s2_ar).sqrt()),
                    "mzt seed {seed} {trend:?} k {k}"
                );
                let mpt = match trend {
                    DfglsTrend::Constant => (cbar * cbar * kappa - cbar * w) / s2_ar,
                    DfglsTrend::ConstantTrend => (cbar * cbar * kappa + (1.0 - cbar) * w) / s2_ar,
                };
                assert_eq!(r.mpt, mpt, "mpt seed {seed} {trend:?} k {k}");
            }
        }
    }

    /// Table 1 internal consistency: the rejection region tightens as the
    /// level drops (every statistic rejects *below* its critical value, so
    /// the 1% value must be the smallest), and the trend case is harder to
    /// reject than the constant case at every level for MZa/MZt (and its
    /// MSB values are smaller — same ordering, different scale).
    #[test]
    fn table1_transcription_is_internally_consistent() {
        for trend in [DfglsTrend::Constant, DfglsTrend::ConstantTrend] {
            let c = ng_perron_crit(trend);
            for cv in [c.mza, c.mzt, c.msb, c.mpt] {
                assert!(cv.pct1 < cv.pct5 && cv.pct5 < cv.pct10, "{trend:?}: {cv:?}");
            }
        }
        let (c, ct) = (ng_perron_crit(DfglsTrend::Constant), ng_perron_crit(DfglsTrend::ConstantTrend));
        assert!(ct.mza.pct5 < c.mza.pct5);
        assert!(ct.mzt.pct5 < c.mzt.pct5);
        assert!(ct.msb.pct5 < c.msb.pct5);
        // MPT runs the other way: the trend-case point-optimal statistic
        // has a larger null median, hence larger critical values.
        assert!(ct.mpt.pct5 > c.mpt.pct5);
    }
}
