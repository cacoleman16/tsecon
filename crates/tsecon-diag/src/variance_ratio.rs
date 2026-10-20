//! Lo-MacKinlay (1988) variance-ratio test and the Chow-Denning (1993)
//! multiple-horizon joint test.
//!
//! The variance-ratio test asks whether the *increments* of a series are
//! serially uncorrelated — the martingale-difference ("random walk")
//! property — by comparing the variance of `q`-period changes with `q`
//! times the variance of one-period changes. Under the null the ratio is
//! one; positive autocorrelation of the increments pushes it above one,
//! mean reversion below. It is **not** a unit-root or stationarity test:
//! both a random walk and its (non-random-walk) cousins with autocorrelated
//! increments have a unit root, and the test discriminates between them.
//!
//! * [`variance_ratio`] — the single-horizon test, matching
//!   `arch.unitroot.VarianceRatio` (arch 8.0.0) in its conventions: `y` is
//!   the LEVEL series (log prices, not returns); the overlapping estimator
//!   divides the `q`-period sum of squares by `nq · q`; the optional
//!   debiasing is Lo-MacKinlay's unbiased-variance correction; the robust
//!   variance is the heteroskedasticity-consistent `theta` of Campbell, Lo &
//!   MacKinlay (1997, eq. 2.4.43); the non-overlapping estimator drops the
//!   trailing `(n - 1) mod q` observations.
//! * [`chow_denning`] — the Chow-Denning (1993) joint test over several
//!   horizons: the maximum absolute standardized statistic, referred to the
//!   studentized-maximum-modulus bound `SMM(alpha; m, inf)`, which is the
//!   Šidák critical value [`sidak_critical_value`]; the p-value inverts the
//!   same bound, `1 - (1 - 2 Phi(-CD))^m`.
//!
//! Deliberate divergences from arch (each pinned by a test): the p-value is
//! the accurate `erfc` form `2 Phi(-|z|)`, not arch's `2 - 2 Phi(|z|)`
//! (which loses ~1e-10 relative at |z| = 5 and underflows to 0 beyond
//! |z| = 8.3); a sample with fewer than two `q`-period sums is refused under
//! both trends (arch returns a ratio identically 0 under trend `"c"` and a
//! single-squared-sum ratio under trend `"n"`, neither of which carries
//! inferential content); constant (or, under drift, exactly linear) input is
//! refused rather than producing `0/0`; and `debiased`/`robust` requested
//! together with the non-overlapping estimator — which has neither — are
//! refused rather than silently ignored.
//!
//! References: Lo & MacKinlay (1988), Review of Financial Studies 1(1);
//! Chow & Denning (1993), Journal of Econometrics 58(3); Campbell, Lo &
//! MacKinlay (1997), *The Econometrics of Financial Markets*, §2.4;
//! Hochberg & Tamhane (1987), *Multiple Comparison Procedures*.

use tsecon_stats::simultaneous::sidak_critical_value;
use tsecon_stats::{ContinuousDist, StdNormal};

use crate::error::DiagError;
use crate::validate::check_series;

const WHAT: &str = "variance_ratio";

/// The drift specification of the random-walk null.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VrTrend {
    /// A random walk without drift (arch `"n"`): increments are centred at
    /// zero.
    None,
    /// A random walk with drift (arch `"c"`, the default): increments are
    /// centred at the sample drift `(y[n-1] - y[0]) / (n - 1)`.
    Drift,
}

/// Which variance of `sqrt(nq) (vr - 1)` standardizes the statistic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VrSeType {
    /// Heteroskedasticity-robust `theta` (Lo-MacKinlay's `z*`; CLM 2.4.43),
    /// valid under conditional heteroskedasticity such as ARCH.
    Robust,
    /// The iid-increment variance `2 (2q - 1)(q - 1) / (3q)` (CLM 2.4.39);
    /// over-rejects under ARCH.
    Homoskedastic,
    /// The non-overlapping estimator's iid variance `2 (q - 1)`.
    NonOverlapping,
}

impl VrSeType {
    /// The Python-facing label: `"robust"`, `"homoskedastic"`, or
    /// `"nonoverlapping"`.
    pub fn as_str(self) -> &'static str {
        match self {
            VrSeType::Robust => "robust",
            VrSeType::Homoskedastic => "homoskedastic",
            VrSeType::NonOverlapping => "nonoverlapping",
        }
    }
}

/// Options of the variance-ratio test. [`Default`] is arch's default call:
/// drift, overlapping, debiased, robust.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VarianceRatioSpec {
    /// Drift specification of the null.
    pub trend: VrTrend,
    /// Overlapping `q`-period sums (the default, more efficient) or
    /// non-overlapping blocks.
    pub overlap: bool,
    /// Lo-MacKinlay unbiased-variance correction. Only meaningful with
    /// `overlap`; requesting it without is refused.
    pub debiased: bool,
    /// Heteroskedasticity-robust standardization. Only meaningful with
    /// `overlap`; requesting it without is refused.
    pub robust: bool,
}

impl Default for VarianceRatioSpec {
    fn default() -> Self {
        VarianceRatioSpec {
            trend: VrTrend::Drift,
            overlap: true,
            debiased: true,
            robust: true,
        }
    }
}

impl VarianceRatioSpec {
    /// The non-overlapping estimator for the given trend (`debiased` and
    /// `robust` off, the only valid combination).
    pub fn non_overlapping(trend: VrTrend) -> Self {
        VarianceRatioSpec {
            trend,
            overlap: false,
            debiased: false,
            robust: false,
        }
    }
}

/// Result of the single-horizon variance-ratio test.
///
/// The null is that the increments are a martingale difference sequence
/// (`vr = 1`); `stat` is asymptotically standard normal under it, so the
/// two-sided `p_value` rejects for large `|stat|`. `stat > 0` (`vr > 1`)
/// signals positive autocorrelation of the increments (momentum),
/// `stat < 0` mean reversion.
#[derive(Debug, Clone, PartialEq)]
pub struct VarianceRatioResult {
    /// The horizon `q`.
    pub lags: usize,
    /// The variance ratio `sigma2_q / sigma2_1`.
    pub vr: f64,
    /// The standardized statistic `sqrt(nobs) (vr - 1) / sqrt(stat_variance)`.
    pub stat: f64,
    /// Two-sided p-value `2 Phi(-|stat|)`, computed through `erfc` so it
    /// keeps full relative accuracy in the tail.
    pub p_value: f64,
    /// The asymptotic variance of `sqrt(nobs) (vr - 1)` that was used.
    pub stat_variance: f64,
    /// Increments used: `n - 1 - n_dropped`.
    pub nobs: usize,
    /// Trailing observations dropped by the non-overlapping estimator so
    /// that `nobs` is a multiple of `q` (`0` with overlap).
    pub n_dropped: usize,
    /// Which standardization was applied.
    pub se_type: VrSeType,
    /// Whether the debiasing correction was applied.
    pub debiased: bool,
}

/// One-sided standard-normal quantiles of the variance-ratio statistic, at
/// the six levels arch reports.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VrCriticalValues {
    /// The 1% quantile.
    pub pct1: f64,
    /// The 5% quantile.
    pub pct5: f64,
    /// The 10% quantile.
    pub pct10: f64,
    /// The 90% quantile.
    pub pct90: f64,
    /// The 95% quantile.
    pub pct95: f64,
    /// The 99% quantile.
    pub pct99: f64,
}

/// Chow-Denning critical values `SMM(alpha; m, inf)` at the 1%, 5% and 10%
/// levels: reject the joint null at level `alpha` when the maximum absolute
/// statistic exceeds the value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChowDenningCrit {
    /// Critical value at `alpha = 0.01`.
    pub pct1: f64,
    /// Critical value at `alpha = 0.05`.
    pub pct5: f64,
    /// Critical value at `alpha = 0.10`.
    pub pct10: f64,
}

/// Result of the Chow-Denning multiple-horizon joint test.
#[derive(Debug, Clone, PartialEq)]
pub struct ChowDenningResult {
    /// The per-horizon variance-ratio tests, in the order the horizons were
    /// given.
    pub individual: Vec<VarianceRatioResult>,
    /// `max_i |stat_i|`.
    pub stat: f64,
    /// The horizon attaining the maximum (the first, in the given order, on
    /// an exact tie).
    pub lag: usize,
    /// `1 - (1 - 2 Phi(-stat))^m`, the Šidák / SMM(inf) p-value; equal to
    /// the single test's p-value when `m = 1`.
    pub p_value: f64,
    /// The SMM(inf) critical values for `m` horizons.
    pub crit: ChowDenningCrit,
}

/// The one-sided `N(0, 1)` quantiles at 1/5/10/90/95/99% that arch reports
/// as the variance-ratio critical values.
///
/// # Errors
///
/// [`DiagError::Stats`] only if the inverse normal CDF rejects a level in
/// `(0, 1)`, which it does not.
pub fn variance_ratio_crit() -> Result<VrCriticalValues, DiagError> {
    let p = |u: f64| StdNormal.ppf(u).map_err(DiagError::from);
    Ok(VrCriticalValues {
        pct1: p(0.01)?,
        pct5: p(0.05)?,
        pct10: p(0.1)?,
        pct90: p(0.9)?,
        pct95: p(0.95)?,
        pct99: p(0.99)?,
    })
}

/// Lo-MacKinlay (1988) variance-ratio test of the random-walk
/// (martingale-difference increments) hypothesis at horizon `lags = q`,
/// matching `arch.unitroot.VarianceRatio` (arch 8.0.0).
///
/// `y` is the **level** series (e.g. log prices) of length `n`; the test is
/// about its increments `Δy_t`. With `nq = n - 1` increments, drift
/// `mu = (y[n-1] - y[0]) / nq` (zero under [`VrTrend::None`]) and
/// `d_t = Δy_t - mu`:
///
/// ```text
/// sigma2_1 = sum d_t^2 / nq
/// overlap:  sigma2_q = sum_{t=q}^{n-1} (y_t - y_{t-q} - q mu)^2 / (nq q)
/// blocks:   sigma2_q = sum_{j=1}^{nq/q} (y_{jq} - y_{(j-1)q} - q mu)^2 / nq
/// debiased (overlap only):
///           sigma2_1 *= nq / (nq - 1)
///           sigma2_q *= nq q / (q (nq - q + 1)(1 - q/nq))
/// vr   = sigma2_q / sigma2_1
/// stat = sqrt(nq) (vr - 1) / sqrt(V)
/// ```
///
/// with `V = 2(q - 1)` for blocks, `2(2q - 1)(q - 1)/(3q)` overlapping and
/// homoskedastic, and the robust
/// `theta = sum_{k=1}^{q-1} 4 (1 - k/q)^2 delta_k`,
/// `delta_k = nq sum_t d_t^2 d_{t-k}^2 / (sum_t d_t^2)^2` (non-debiased,
/// drift-centred). The non-overlapping estimator first drops the trailing
/// `(n - 1) mod q` observations (arch does the same, with a warning) and
/// recomputes the drift on the shortened sample.
///
/// The computation runs on `y` rescaled by an exact power of two (chosen
/// from `max |y|`), so the result is bit-identical to the unscaled
/// arithmetic wherever that arithmetic neither overflows nor underflows,
/// while extreme magnitudes cannot overflow the squared sums.
///
/// Cost: `O(n)` plus `O(n q)` for the robust variance.
///
/// # Errors
///
/// * [`DiagError::NonFinite`] if `y` contains NaN or infinities.
/// * [`DiagError::SeriesTooShort`] if `n < 2`.
/// * [`DiagError::InvalidLags`] if `lags < 2`, or if the sample holds fewer
///   than two `q`-period sums (`n < q + 2` with overlap, `n < 2q + 1`
///   without) — checked before any `O(n q)` work.
/// * [`DiagError::InvalidArgument`] if `debiased` or `robust` is requested
///   without `overlap`, or if the increments have (numerically) zero
///   variance about the drift (a constant series, or an exact linear trend
///   under [`VrTrend::Drift`]).
/// * [`DiagError::NumericalBreakdown`] if the robust variance is not
///   positive (increments whose squares never overlap within `q - 1` lags).
pub fn variance_ratio(
    y: &[f64],
    lags: usize,
    spec: VarianceRatioSpec,
) -> Result<VarianceRatioResult, DiagError> {
    let n = check_series(y, 2, WHAT)?;
    check_horizon(n, lags, spec)?;
    let n_dropped = if spec.overlap { 0 } else { (n - 1) % lags };
    let inc = Increments::new(&y[..n - n_dropped], spec.trend)?;
    let deltas = if spec.overlap && spec.robust {
        inc.robust_deltas(lags - 1)
    } else {
        Vec::new()
    };
    inc.test(lags, spec, n_dropped, &deltas)
}

/// The `O(1)` argument checks of one horizon, run before any data work.
fn check_horizon(n: usize, q: usize, spec: VarianceRatioSpec) -> Result<(), DiagError> {
    if q < 2 {
        return Err(DiagError::InvalidLags {
            what: WHAT,
            nlags: q,
            n,
            requirement: "lags >= 2 (at q = 1 the q-period variance is the \
                          one-period variance itself, so the ratio is identically 1)",
        });
    }
    if !spec.overlap && spec.debiased {
        return Err(DiagError::InvalidArgument {
            what: "variance_ratio: debiased=True has no effect with overlap=False: the \
                   non-overlapping variance ratio uses the plain block variances (arch \
                   silently ignores the flag). Use overlap=True for the Lo-MacKinlay \
                   debiased estimator, or drop debiased",
        });
    }
    if !spec.overlap && spec.robust {
        return Err(DiagError::InvalidArgument {
            what: "variance_ratio: robust=True has no effect with overlap=False: the \
                   non-overlapping variance ratio has only the iid variance 2(q - 1) \
                   (arch silently ignores the flag). Use overlap=True for the \
                   heteroskedasticity-robust statistic, or drop robust",
        });
    }
    // At least two q-period sums, uniformly for both trends.
    if spec.overlap {
        if q > n - 2 {
            return Err(DiagError::InvalidLags {
                what: WHAT,
                nlags: q,
                n,
                requirement: "lags <= n - 2 with overlap=True (at least two \
                              overlapping q-period sums; one sum carries no \
                              information about the ratio)",
            });
        }
    } else if (n - 1) / q < 2 {
        return Err(DiagError::InvalidLags {
            what: WHAT,
            nlags: q,
            n,
            requirement: "lags <= (n - 1) / 2 with overlap=False (at least two \
                          non-overlapping q-period blocks; one block carries no \
                          information about the ratio)",
        });
    }
    Ok(())
}

/// The drift-centred increments of the sample a test uses, on the levels
/// rescaled by an exact power of two. Every horizon of an overlapping joint
/// test shares one of these (same sample, same drift).
struct Increments {
    /// The rescaled levels.
    ys: Vec<f64>,
    /// Number of increments, `ys.len() - 1`.
    nq: usize,
    /// The (rescaled) drift removed from each increment.
    mu: f64,
    /// Squared drift-centred increments: `sigma2_1`'s numerator and the
    /// robust variance's building block (non-debiased, as in CLM 2.4.43).
    z2: Vec<f64>,
    /// `sum z2`.
    ss1: f64,
}

impl Increments {
    fn new(raw: &[f64], trend: VrTrend) -> Result<Self, DiagError> {
        let nq = raw.len() - 1;
        // Exact power-of-two rescaling: every operation below commutes with
        // it bit-for-bit, and the squared sums can no longer overflow.
        let ymax = raw.iter().fold(0.0_f64, |m, &v| m.max(v.abs()));
        if ymax == 0.0 {
            return Err(zero_variance());
        }
        let exponent = (ymax.log2().ceil() as i32).clamp(-1000, 1000);
        let scale = 2.0_f64.powi(-exponent);
        let ys: Vec<f64> = raw.iter().map(|&v| v * scale).collect();
        let ymax = ymax * scale;
        let mu = match trend {
            VrTrend::None => 0.0,
            VrTrend::Drift => (ys[nq] - ys[0]) / nq as f64,
        };
        let z2: Vec<f64> = ys
            .windows(2)
            .map(|w| {
                let v = (w[1] - w[0]) - mu;
                v * v
            })
            .collect();
        let ss1: f64 = z2.iter().sum();
        // Increments that are zero up to the rounding of the levels
        // themselves (|Δy| below a few ulps of max|y|) make the ratio 0/0 or
        // rounding noise.
        let sigma2_1 = ss1 / nq as f64;
        if sigma2_1.is_nan() || sigma2_1.sqrt() <= 16.0 * f64::EPSILON * ymax {
            return Err(zero_variance());
        }
        Ok(Increments {
            ys,
            nq,
            mu,
            z2,
            ss1,
        })
    }

    /// `delta_k = nq sum_t z2_t z2_{t-k} / (sum z2)^2` for `k = 1..=kmax`
    /// (element `k - 1`): `O(nq kmax)`, the robust variance's whole cost.
    fn robust_deltas(&self, kmax: usize) -> Vec<f64> {
        let nqf = self.nq as f64;
        let scale_d = self.ss1 * self.ss1;
        (1..=kmax)
            .map(|k| {
                let dot: f64 = self.z2[k..]
                    .iter()
                    .zip(&self.z2[..self.nq - k])
                    .map(|(&a, &b)| a * b)
                    .sum();
                nqf * dot / scale_d
            })
            .collect()
    }

    /// The test at horizon `q` (already checked by [`check_horizon`]);
    /// `deltas` holds at least `q - 1` robust `delta_k` when the spec is
    /// overlapping and robust, and is unused otherwise.
    fn test(
        &self,
        q: usize,
        spec: VarianceRatioSpec,
        n_dropped: usize,
        deltas: &[f64],
    ) -> Result<VarianceRatioResult, DiagError> {
        let (ys, nq) = (&self.ys, self.nq);
        let nqf = nq as f64;
        let qf = q as f64;
        let mut sigma2_1 = self.ss1 / nqf;
        let qmu = qf * self.mu;
        let mut sigma2_q = if spec.overlap {
            let sq: f64 = (q..=nq)
                .map(|t| {
                    let v = (ys[t] - ys[t - q]) - qmu;
                    v * v
                })
                .sum();
            sq / (nqf * qf)
        } else {
            let sq: f64 = (1..=nq / q)
                .map(|j| {
                    let v = (ys[j * q] - ys[(j - 1) * q]) - qmu;
                    v * v
                })
                .sum();
            sq / nqf
        };
        let debiased = spec.overlap && spec.debiased;
        if debiased {
            sigma2_1 *= nqf / (nqf - 1.0);
            let m = qf * (nqf - qf + 1.0) * (1.0 - qf / nqf);
            sigma2_q *= (nqf * qf) / m;
        }

        let (stat_variance, se_type) = if !spec.overlap {
            (2.0 * (qf - 1.0), VrSeType::NonOverlapping)
        } else if !spec.robust {
            // Integer numerator/denominator as in arch, exact in u128.
            let q128 = q as u128;
            let num = 2 * (2 * q128 - 1) * (q128 - 1);
            (num as f64 / (3 * q128) as f64, VrSeType::Homoskedastic)
        } else {
            debug_assert!(deltas.len() >= q - 1, "robust deltas computed up to q - 1");
            let mut theta = 0.0;
            for (k, &delta) in (1..q).zip(deltas) {
                let w = 1.0 - k as f64 / qf;
                theta += 4.0 * (w * w) * delta;
            }
            if !(theta > 0.0 && theta.is_finite()) {
                return Err(DiagError::NumericalBreakdown {
                    what: "variance_ratio (robust variance theta is not positive: no two \
                           nonzero squared increments lie within q - 1 periods of each \
                           other; use robust=False or a different horizon)",
                });
            }
            (theta, VrSeType::Robust)
        };

        let vr = sigma2_q / sigma2_1;
        let stat = nqf.sqrt() * (vr - 1.0) / stat_variance.sqrt();
        if !(vr.is_finite() && stat.is_finite()) {
            return Err(DiagError::NumericalBreakdown { what: WHAT });
        }
        Ok(VarianceRatioResult {
            lags: q,
            vr,
            stat,
            p_value: two_sided_p(stat),
            stat_variance,
            nobs: nq,
            n_dropped,
            se_type,
            debiased,
        })
    }
}

/// Chow-Denning (1993) joint variance-ratio test over the horizons `lags`
/// (distinct, each `>= 2`, in any order), with every horizon tested under
/// the same `spec`.
///
/// `stat = max_i |z(q_i)|`, referred to the studentized maximum modulus
/// with infinite degrees of freedom: the level-`alpha` critical value is
/// the Šidák value `Phi^{-1}(1 - (1 - (1 - alpha)^(1/m)) / 2)` and the
/// p-value is `1 - (1 - 2 Phi(-stat))^m` (evaluated as
/// `-expm1(m ln1p(-p_min))`). The bound treats the `m` statistics as
/// independent; the horizons share data and are positively correlated, so
/// the test is conservative. With `m = 1` it is exactly the single
/// [`variance_ratio`] test.
///
/// CRAN `vrtest::Chow.Denning` (vrtest 1.2) applies the same bound to the
/// overlapping, non-debiased, drift-centred statistics (both the
/// homoskedastic `CD1` and robust `CD2`) and reports no p-value; here the
/// per-horizon statistics follow `spec`.
///
/// Cost: with `overlap`, the horizons share one set of increments and the
/// robust variance's lag products, so the call is `O(n max q)` however many
/// horizons are given; without it, each block length truncates the sample
/// on its own, `O(n)` per horizon. Each per-horizon result is bit-identical
/// to the corresponding [`variance_ratio`] call.
///
/// # Errors
///
/// [`DiagError::InvalidArgument`] for an empty or duplicated horizon set;
/// otherwise every error of [`variance_ratio`]. The argument checks of all
/// horizons (`lags < 2`, the two-sums rule, inert flags) run before any
/// data work, so they take precedence over a data error such as zero
/// variance; among them, the first failing horizon in the given order is
/// reported.
pub fn chow_denning(
    y: &[f64],
    lags: &[usize],
    spec: VarianceRatioSpec,
) -> Result<ChowDenningResult, DiagError> {
    if lags.is_empty() {
        return Err(DiagError::InvalidArgument {
            what: "variance_ratio: lags is empty; pass at least one horizon q >= 2",
        });
    }
    let mut sorted = lags.to_vec();
    sorted.sort_unstable();
    if sorted.windows(2).any(|w| w[0] == w[1]) {
        return Err(DiagError::InvalidArgument {
            what: "variance_ratio: lags contains a repeated horizon; each horizon may \
                   appear once (a repeat would count the same statistic twice in the \
                   joint test's multiplicity correction)",
        });
    }
    // Every horizon's argument checks run before any data work, so a bad
    // horizon late in the list is refused in O(m), not after the O(n q) work
    // of the ones before it.
    let n = check_series(y, 2, WHAT)?;
    for &q in lags {
        check_horizon(n, q, spec)?;
    }
    let individual = if spec.overlap {
        // One sample and one drift for every horizon: the increments, and
        // the robust variance's delta_k up to the longest horizon, are
        // computed once and shared — O(n max q) rather than O(n sum q), with
        // results bit-identical to the single-horizon calls.
        let inc = Increments::new(y, spec.trend)?;
        let deltas = if spec.robust {
            inc.robust_deltas(sorted[sorted.len() - 1] - 1)
        } else {
            Vec::new()
        };
        lags.iter()
            .map(|&q| inc.test(q, spec, 0, &deltas))
            .collect::<Result<Vec<_>, _>>()?
    } else {
        // Each block length truncates the sample (and re-estimates the
        // drift) on its own.
        lags.iter()
            .map(|&q| variance_ratio(y, q, spec))
            .collect::<Result<Vec<_>, _>>()?
    };

    let mut stat = individual[0].stat.abs();
    let mut lag = individual[0].lags;
    for r in &individual[1..] {
        if r.stat.abs() > stat {
            stat = r.stat.abs();
            lag = r.lags;
        }
    }
    let m = lags.len();
    let p_min = two_sided_p(stat);
    let p_value = if m == 1 {
        p_min
    } else {
        -((m as f64) * (-p_min).ln_1p()).exp_m1()
    };
    let crit = ChowDenningCrit {
        pct1: sidak_critical_value(0.01, m)?,
        pct5: sidak_critical_value(0.05, m)?,
        pct10: sidak_critical_value(0.10, m)?,
    };
    Ok(ChowDenningResult {
        individual,
        stat,
        lag,
        p_value,
        crit,
    })
}

/// `2 Phi(-|z|)` through `erfc` (full relative accuracy in the tail).
fn two_sided_p(z: f64) -> f64 {
    2.0 * StdNormal.sf(z.abs())
}

fn zero_variance() -> DiagError {
    DiagError::InvalidArgument {
        what: "variance_ratio: the increments of y have (numerically) zero variance \
               about the drift — a constant series, or an exact linear trend under \
               trend=\"c\" — so the variance ratio is 0/0. Pass the LEVEL series \
               (e.g. log prices) of a variable that actually moves",
    }
}
