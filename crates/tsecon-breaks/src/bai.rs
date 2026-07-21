//! The Bai–Perron multiple-break estimator: global partitions, sequential
//! `supF(l+1|l)` selection at 5%, per-regime OLS, and Bai (1997)
//! break-date confidence intervals.

use tsecon_hac::{ols, SeType};

use crate::cdf::bai_argmax_two_sided_crit;
use crate::check::validate;
use crate::dp::global_partitions;
use crate::error::BreaksError;
use crate::segments::SegTable;
use crate::tables::{BP_SEQ_CV_5PCT, BP_SEQ_TRIMS};

/// Configuration of [`bai_perron`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BaiPerronConfig {
    /// Maximum number of breaks considered (`1..=10`; the sequential
    /// procedure may select fewer, down to zero).
    pub max_breaks: usize,
    /// Trimming fraction: minimal regime length `h = ceil(trim * T)`.
    /// Must be one of the published-critical-value trimmings
    /// `{0.05, 0.10, 0.15, 0.20, 0.25}`.
    pub trim: f64,
}

impl Default for BaiPerronConfig {
    fn default() -> Self {
        Self {
            max_breaks: 5,
            trim: 0.15,
        }
    }
}

/// One regime of the selected partition, fitted by OLS.
#[derive(Debug, Clone, PartialEq)]
pub struct RegimeFit {
    /// First observation of the regime (0-indexed, inclusive).
    pub start: usize,
    /// Last observation of the regime (0-indexed, inclusive).
    pub end: usize,
    /// OLS coefficients on the `q` columns of `x` within the regime.
    pub params: Vec<f64>,
    /// Nonrobust standard errors of `params`.
    pub se: Vec<f64>,
    /// Residual sum of squares of the regime regression.
    pub ssr: f64,
}

/// Bai (1997) confidence interval for one estimated break date.
#[derive(Debug, Clone, PartialEq)]
pub struct BreakCi {
    /// The estimated break date (last observation of the earlier regime).
    pub date: usize,
    /// The scale `L = delta' Q delta / sigma^2` (larger break => tighter CI).
    pub scale: f64,
    /// Lower end of the 90% interval (clipped to the sample).
    pub lower90: usize,
    /// Upper end of the 90% interval (clipped to the sample).
    pub upper90: usize,
    /// Lower end of the 95% interval (clipped to the sample).
    pub lower95: usize,
    /// Upper end of the 95% interval (clipped to the sample).
    pub upper95: usize,
}

/// Full output of the Bai–Perron procedure.
#[derive(Debug, Clone, PartialEq)]
pub struct BaiPerron {
    /// Number of breaks selected by the sequential `supF(l+1|l)` tests at 5%.
    pub n_breaks: usize,
    /// Selected break dates (0-indexed last observation of each regime;
    /// empty when `n_breaks == 0`).
    pub break_dates: Vec<usize>,
    /// `sup_f_seq[l] = supF(l+1|l)` for `l = 0..max_breaks-1`, each using
    /// the `l`-break global minimizers as the null partition.
    pub sup_f_seq: Vec<f64>,
    /// The published 5% critical values `c(q, l+1)` the statistics were
    /// compared against.
    pub sup_f_crit: Vec<f64>,
    /// `ssr_path[m]` = global minimal SSR with `m` breaks, `m = 0..max_breaks`.
    pub ssr_path: Vec<f64>,
    /// The global minimizers for every `m = 1..=max_breaks`
    /// (`break_dates_by_m[m-1]` has `m` dates).
    pub break_dates_by_m: Vec<Vec<usize>>,
    /// Per-regime OLS fits for the selected partition (`n_breaks + 1`
    /// entries; a single full-sample fit when `n_breaks == 0`).
    pub regimes: Vec<RegimeFit>,
    /// Bai (1997) 90%/95% confidence intervals for each selected break
    /// (empty when `n_breaks == 0`).
    pub ci: Vec<BreakCi>,
    /// Minimal regime length `h = ceil(trim * T)` actually used.
    pub h: usize,
}

/// `supF(l+1|l)`: the best additional break inside any segment of the null
/// partition, in the Wald form matching the published critical values.
fn seq_stat(tbl: &SegTable, null_dates: &[usize], q: usize) -> Result<f64, BreaksError> {
    let t = tbl.t;
    let mut best = 0.0_f64;
    let mut start = 0_usize;
    let mut bounds: Vec<(usize, usize)> = Vec::with_capacity(null_dates.len() + 1);
    for &d in null_dates {
        bounds.push((start, d));
        start = d + 1;
    }
    bounds.push((start, t - 1));
    for (s, e) in bounds {
        if let Some((_, split_ssr)) = tbl.best_split(s, e) {
            if split_ssr <= 0.0 {
                return Err(BreaksError::DegenerateFit {
                    what: "supF(l+1|l) denominator (segment split SSR)",
                });
            }
            let n_i = e - s + 1;
            let f = (n_i - 2 * q) as f64 * (tbl.ssr(s, e) - split_ssr).max(0.0) / split_ssr;
            best = best.max(f);
        }
    }
    Ok(best)
}

/// Per-regime OLS (delegated to [`tsecon_hac::ols`], the library's single
/// OLS owner) with nonrobust standard errors.
fn fit_regimes(y: &[f64], x: &[Vec<f64>], dates: &[usize]) -> Result<Vec<RegimeFit>, BreaksError> {
    let t = y.len();
    let mut out = Vec::with_capacity(dates.len() + 1);
    let mut start = 0_usize;
    let mut cuts: Vec<usize> = dates.to_vec();
    cuts.push(t - 1);
    for end in cuts {
        let ys = &y[start..=end];
        let cols: Vec<Vec<f64>> = x.iter().map(|c| c[start..=end].to_vec()).collect();
        let fit = ols(ys, &cols)?;
        let inf = fit.inference(SeType::NonRobust)?;
        let ssr: f64 = fit.residuals.iter().map(|r| r * r).sum();
        out.push(RegimeFit {
            start,
            end,
            params: fit.params,
            se: inf.bse,
            ssr,
        });
        start = end + 1;
    }
    Ok(out)
}

/// Bai (1997) homogeneous-case confidence intervals for the selected breaks.
fn break_cis(
    x: &[Vec<f64>],
    dates: &[usize],
    regimes: &[RegimeFit],
    ssr_m: f64,
    t: usize,
    q: usize,
) -> Result<Vec<BreakCi>, BreaksError> {
    // Full-sample second-moment matrix Q = X'X / T (homogeneity assumption).
    let mut qm = vec![0.0_f64; q * q];
    for a in 0..q {
        for b in 0..=a {
            let s: f64 = (0..t).map(|i| x[a][i] * x[b][i]).sum();
            qm[a * q + b] = s / t as f64;
            qm[b * q + a] = qm[a * q + b];
        }
    }
    let df = t - (dates.len() + 1) * q;
    let sigma2 = ssr_m / df as f64;
    let c90 = bai_argmax_two_sided_crit(0.90)?;
    let c95 = bai_argmax_two_sided_crit(0.95)?;
    let mut out = Vec::with_capacity(dates.len());
    for (i, &d) in dates.iter().enumerate() {
        let delta: Vec<f64> = (0..q)
            .map(|a| regimes[i + 1].params[a] - regimes[i].params[a])
            .collect();
        let mut quad = 0.0_f64;
        for a in 0..q {
            for b in 0..q {
                quad += delta[a] * qm[a * q + b] * delta[b];
            }
        }
        let scale = quad / sigma2;
        if !(scale > 0.0 && scale.is_finite()) {
            return Err(BreaksError::DegenerateFit {
                what: "break-date CI scale delta' Q delta / sigma^2 (the two \
                       regimes' coefficients are identical)",
            });
        }
        let hw90 = (c90 / scale).ceil() as i64;
        let hw95 = (c95 / scale).ceil() as i64;
        let di = d as i64;
        let last = (t - 1) as i64;
        out.push(BreakCi {
            date: d,
            scale,
            lower90: (di - hw90 - 1).max(0) as usize,
            upper90: (di + hw90 + 1).min(last) as usize,
            lower95: (di - hw95 - 1).max(0) as usize,
            upper95: (di + hw95 + 1).min(last) as usize,
        });
    }
    Ok(out)
}

/// Estimate multiple structural breaks by the Bai–Perron procedure.
///
/// Fits the pure-structural-change regression `y_t = x_t' beta_j + u_t`
/// (regime `j`), finds the global SSR-minimizing partitions for
/// `m = 1..=max_breaks` by dynamic programming, selects the number of
/// breaks by sequential `supF(l+1|l)` tests at the 5% level against the
/// published Bai–Perron critical values, and reports per-regime OLS fits
/// plus Bai (1997) break-date confidence intervals (homogeneous-moments
/// case — see the crate docs for the exact assumption).
///
/// `x` is a slice of `q` columns of length `T`; include the constant
/// explicitly. Break dates are 0-indexed as the last observation of each
/// regime.
///
/// # Errors
///
/// Input validation ([`BreaksError::EmptyInput`],
/// [`BreaksError::DimensionMismatch`], [`BreaksError::NonFinite`],
/// [`BreaksError::InvalidArgument`], [`BreaksError::TrimTooSmall`]),
/// [`BreaksError::UnsupportedQ`] (`q > 10`),
/// [`BreaksError::UnsupportedTrim`] (trim outside the published grid),
/// [`BreaksError::InfeasibleBreaks`] (`(max_breaks+1) * h > T`),
/// [`BreaksError::Singular`] (collinear columns on some admissible
/// segment), and [`BreaksError::DegenerateFit`] (an exact fit making an F
/// denominator zero).
pub fn bai_perron(
    y: &[f64],
    x: &[Vec<f64>],
    config: BaiPerronConfig,
) -> Result<BaiPerron, BreaksError> {
    let (t, q, h) = validate(y, x, config.trim)?;
    if q > 10 {
        return Err(BreaksError::UnsupportedQ { q });
    }
    if config.max_breaks == 0 {
        return Err(BreaksError::InvalidArgument {
            what: "max_breaks must be at least 1 (for a known-date single break \
                   use chow_test; for a plain fit use ols)",
        });
    }
    if config.max_breaks > 10 {
        return Err(BreaksError::InvalidArgument {
            what: "max_breaks must be at most 10: the published Bai-Perron \
                   sequential critical values stop at 10 breaks",
        });
    }
    let trim_idx = BP_SEQ_TRIMS
        .iter()
        .position(|v| (v - config.trim).abs() < 1e-9)
        .ok_or(BreaksError::UnsupportedTrim {
            trim_pct: (config.trim * 100.0).round() as usize,
        })?;
    if (config.max_breaks + 1) * h > t {
        return Err(BreaksError::InfeasibleBreaks {
            max_breaks: config.max_breaks,
            h,
            t,
        });
    }
    let tbl = SegTable::build(y, x, h)?;
    let gp = global_partitions(&tbl, config.max_breaks);

    let mut sup_f_seq = Vec::with_capacity(config.max_breaks);
    for l in 0..config.max_breaks {
        let null_dates: &[usize] = if l == 0 { &[] } else { &gp.dates_by_m[l - 1] };
        sup_f_seq.push(seq_stat(&tbl, null_dates, q)?);
    }
    let sup_f_crit: Vec<f64> = BP_SEQ_CV_5PCT[trim_idx][q - 1][..config.max_breaks].to_vec();

    let mut n_breaks = 0_usize;
    for l in 0..config.max_breaks {
        if sup_f_seq[l] > sup_f_crit[l] {
            n_breaks = l + 1;
        } else {
            break;
        }
    }
    let break_dates: Vec<usize> = if n_breaks == 0 {
        Vec::new()
    } else {
        gp.dates_by_m[n_breaks - 1].clone()
    };
    let regimes = fit_regimes(y, x, &break_dates)?;
    let ci = if n_breaks == 0 {
        Vec::new()
    } else {
        break_cis(x, &break_dates, &regimes, gp.ssr_path[n_breaks], t, q)?
    };
    Ok(BaiPerron {
        n_breaks,
        break_dates,
        sup_f_seq,
        sup_f_crit,
        ssr_path: gp.ssr_path,
        break_dates_by_m: gp.dates_by_m,
        regimes,
        ci,
        h,
    })
}
