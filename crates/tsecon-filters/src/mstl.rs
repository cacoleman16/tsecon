//! MSTL — Multiple Seasonal-Trend decomposition using LOESS (Bandara,
//! Hyndman & Bergmeir 2021, arXiv:2107.13462): STL iterated over several
//! seasonal periods for series with more than one seasonal cycle (the
//! canonical example is hourly data with daily *and* weekly seasonality,
//! periods 24 and 168).
//!
//! The algorithm, exactly as `statsmodels.tsa.seasonal.MSTL` implements it:
//!
//! * periods are sorted ascending (paired windows travel with their
//!   period), and any period `>= n / 2` is dropped — statsmodels warns and
//!   removes it; this port removes it and reports it in
//!   [`MstlResult::dropped_periods`];
//! * each period gets a *seasonal window* — the `seasonal` LOESS window of
//!   its [`stl`] pass. `windows = None` uses the paper's rule
//!   `7 + 4 * k` for the k-th (1-based) sorted period: 11, 15, 19, ...;
//! * starting from the raw series, `iterate` rounds (default 2; forced to
//!   1 when only one period survives) cycle over the sorted periods: add
//!   the current estimate of this period's seasonal back, run STL at this
//!   period with its window, store the refreshed seasonal, subtract it —
//!   so each pass re-extracts one seasonal from a series deseasonalized of
//!   all the *others*;
//! * `trend` and the robustness `weights` come from the final STL fit (the
//!   largest period, last round), and `resid` is the fully deseasonalized
//!   series minus that trend, so `y = sum(seasonal) + trend + resid`;
//! * every other STL knob (`trend`/`low_pass` windows, degrees, jumps,
//!   `robust`, `inner_iter`/`outer_iter`) is forwarded unchanged to every
//!   STL pass, exactly like statsmodels' `stl_kwargs` (the per-pass
//!   `seasonal` window comes from `windows`, so [`MstlParams::stl`]'s
//!   `seasonal` field is ignored, as statsmodels ignores a `seasonal` key
//!   in `stl_kwargs`).
//!
//! Deliberate deviations, all on the safe side and all *refusals* where
//! statsmodels crashes or degrades silently: an empty `periods`, duplicate
//! periods, `iterate = 0`, and "every period was dropped" are teaching
//! errors here (statsmodels raises `NameError`/produces duplicate columns
//! in the first three cases and `NameError` in the last). The Box-Cox
//! `lmbda` pre-transform of statsmodels' MSTL is **not implemented** —
//! transform the series yourself before decomposing if you need it.
//!
//! Accuracy: pinned elementwise against statsmodels 0.14.6 MSTL (trend,
//! every per-period seasonal, resid, robustness weights) on a seeded
//! two-seasonal hourly-like series (periods 24/168), a seeded
//! three-seasonal awkward-period series (5/12/31), the degenerate
//! single-period case, and a dropped-period case, at 1e-8
//! (`fixtures/mstl.json`; observed agreement ≤ ~5e-11 on components and
//! ~3e-10 on the robust case's weights, where 15 outer iterations
//! amplify ulp noise). The single-period
//! case is additionally required to reproduce this crate's own [`stl`]
//! bitwise — internal consistency, graded separately from the third-party
//! golden.

use crate::error::{check_finite, FiltersError};
use crate::stl::{sample_var, stl, StlParams};

// ------------------------------------------------------------- parameters

/// Tuning parameters for [`mstl`], mirroring `statsmodels.tsa.seasonal.MSTL`
/// (same defaults, same resolution rules).
///
/// `..Default::default()` gives the statsmodels defaults:
///
/// ```
/// use tsecon_filters::MstlParams;
/// let params = MstlParams { iterate: 3, ..Default::default() };
/// # let _ = params;
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MstlParams {
    /// Seasonal LOESS window for each period (odd, `>= 3`), paired with the
    /// period of the same index *before* sorting — the pairs are sorted
    /// together by period. `None` uses the Bandara et al. rule
    /// `7 + 4 * k` for the k-th sorted period (11, 15, 19, ...).
    pub windows: Option<Vec<usize>>,
    /// Rounds of the seasonal-refinement loop (`>= 1`; the statsmodels
    /// default is 2). With a single period only one round is ever run
    /// (statsmodels does the same).
    pub iterate: usize,
    /// STL parameters forwarded to every per-period STL pass —
    /// statsmodels' `stl_kwargs`. The `seasonal` field is ignored (the
    /// per-period window comes from `windows`), exactly as statsmodels
    /// discards a `seasonal` entry in `stl_kwargs`.
    pub stl: StlParams,
}

impl Default for MstlParams {
    fn default() -> Self {
        MstlParams {
            windows: None,
            iterate: 2,
            stl: StlParams::default(),
        }
    }
}

/// Result of an [`mstl`] decomposition:
/// `y = seasonal[0] + ... + seasonal[k-1] + trend + resid` elementwise
/// (`resid` is computed as exactly that difference, in statsmodels'
/// operation order).
#[derive(Debug, Clone, PartialEq)]
pub struct MstlResult {
    /// The seasonal periods actually decomposed: sorted ascending, with
    /// any period `>= n / 2` removed (see [`MstlResult::dropped_periods`]).
    pub periods: Vec<usize>,
    /// The seasonal LOESS window used for each entry of `periods`.
    pub windows: Vec<usize>,
    /// One seasonal component per entry of `periods`, each of length `n`.
    pub seasonal: Vec<Vec<f64>>,
    /// The trend component (length `n`), from the final STL pass.
    pub trend: Vec<f64>,
    /// The remainder `y - sum(seasonal) - trend` (length `n`).
    pub resid: Vec<f64>,
    /// Bisquare robustness weights of the final STL pass (length `n`); all
    /// 1 unless that pass ran outer iterations (`robust = true` or an
    /// explicit `outer_iter > 0`).
    pub weights: Vec<f64>,
    /// Refinement rounds actually run (1 when a single period survived,
    /// else the requested `iterate`).
    pub iterate: usize,
    /// Periods removed because they are `>= n / 2` — too long to
    /// distinguish from trend. statsmodels emits a `UserWarning` and drops
    /// them silently from the result; this port reports them here.
    pub dropped_periods: Vec<usize>,
}

// ------------------------------------------------------------- resolution

/// Sort (period, window) pairs ascending by period, apply the default
/// window rule when none are given, drop periods `>= n / 2`, and validate —
/// the statsmodels `_process_periods_and_windows` semantics with teaching
/// errors in place of its crashes.
#[allow(clippy::type_complexity)]
fn resolve_periods_windows(
    n: usize,
    periods: &[usize],
    windows: &Option<Vec<usize>>,
) -> Result<(Vec<usize>, Vec<usize>, Vec<usize>), FiltersError> {
    if periods.is_empty() {
        return Err(FiltersError::InvalidParameter {
            name: "periods",
            value: 0.0,
            requirement: "at least one seasonal period (observations per cycle) — e.g. \
                 [24, 168] for hourly data with daily and weekly cycles, or [12] \
                 for one monthly cycle. With no seasonal cycle there is nothing \
                 for MSTL to decompose; use a trend filter instead",
        });
    }
    // Pair each period with its window, then sort by period (statsmodels
    // sorts the (period, window) pairs together; with defaults the sorted
    // periods take windows 11, 15, 19, ... positionally).
    let mut pairs: Vec<(usize, usize)> = match windows {
        Some(w) => {
            if w.len() != periods.len() {
                return Err(FiltersError::InvalidParameter {
                    name: "windows",
                    value: w.len() as f64,
                    requirement: "one seasonal LOESS window per period (same length as \
                         `periods`); or None to use the MSTL default rule \
                         7 + 4*k for the k-th sorted period (11, 15, 19, ...)",
                });
            }
            periods.iter().copied().zip(w.iter().copied()).collect()
        }
        None => {
            let mut sorted: Vec<usize> = periods.to_vec();
            sorted.sort_unstable();
            sorted
                .into_iter()
                .enumerate()
                .map(|(k, p)| (p, 7 + 4 * (k + 1)))
                .collect()
        }
    };
    pairs.sort_unstable();
    for pair in pairs.windows(2) {
        if pair[0].0 == pair[1].0 {
            return Err(FiltersError::InvalidParameter {
                name: "periods",
                value: pair[0].0 as f64,
                requirement: "distinct seasonal periods — the same period twice would just \
                     re-extract one seasonal in two pieces (statsmodels silently \
                     returns two components for the duplicated period)",
            });
        }
    }
    // Remove periods too long to distinguish from trend: period >= n / 2
    // (statsmodels warns and removes; the kept periods are a prefix of the
    // ascending sort, so the paired windows truncate with them).
    let dropped: Vec<usize> = pairs
        .iter()
        .filter(|&&(p, _)| 2 * p >= n)
        .map(|&(p, _)| p)
        .collect();
    pairs.retain(|&(p, _)| 2 * p < n);
    if pairs.is_empty() {
        let min_p = dropped.iter().copied().min().unwrap_or(0);
        return Err(FiltersError::SeriesTooShort {
            filter: "mstl",
            needed: 2 * min_p + 1,
            got: n,
            why: "MSTL drops any period >= half the series length (too long to tell \
                  seasonality from trend; statsmodels warns and drops the same way), \
                  and every requested period was dropped — supply a longer series \
                  or shorter periods",
        });
    }
    for &(p, _) in &pairs {
        if p < 2 {
            return Err(FiltersError::InvalidParameter {
                name: "periods",
                value: p as f64,
                requirement: "every period must be an integer >= 2 — the number of \
                     observations per seasonal cycle (24 for hourly data with a \
                     daily cycle, 12 for monthly data with a yearly cycle)",
            });
        }
    }
    for &(_, w) in &pairs {
        if w < 3 || w % 2 == 0 {
            return Err(FiltersError::InvalidParameter {
                name: "windows",
                value: w as f64,
                requirement: "each seasonal LOESS window must be an odd integer >= 3 (it counts \
                     observations of one cycle-subseries and needs a centre point); the \
                     MSTL default rule gives 11, 15, 19, ... for the sorted periods",
            });
        }
    }
    let (periods, windows) = pairs.into_iter().unzip();
    Ok((periods, windows, dropped))
}

// -------------------------------------------------------------- interface

/// MSTL decomposition of `y` at the seasonal periods `periods`
/// (observations per cycle, e.g. `[24, 168]` for hourly data with daily
/// and weekly cycles): `y = seasonal_1 + ... + seasonal_K + trend + resid`.
///
/// See the [module docs](self) for the algorithm; parameters, defaults and
/// the period/window resolution mirror `statsmodels.tsa.seasonal.MSTL`,
/// and the output matches it elementwise (pinned at 1e-8; observed
/// ~1e-12). The Box-Cox `lmbda` option of statsmodels is deliberately not
/// implemented — pre-transform `y` if you need it.
///
/// # Errors
///
/// * [`FiltersError::NonFiniteInput`] — NaN/inf in `y` (impute first).
/// * [`FiltersError::InvalidParameter`] — empty or duplicate `periods`; a
///   period `< 2`; `windows` of the wrong length, or a window even or
///   `< 3`; `iterate = 0`; plus everything the forwarded [`StlParams`]
///   can raise (invalid `trend`/`low_pass` window, degree, jump,
///   `inner_iter = 0`).
/// * [`FiltersError::SeriesTooShort`] — every period was `>= n / 2` and
///   therefore dropped.
pub fn mstl(y: &[f64], periods: &[usize], params: &MstlParams) -> Result<MstlResult, FiltersError> {
    check_finite(y)?;
    let n = y.len();
    let (periods, windows, dropped_periods) = resolve_periods_windows(n, periods, &params.windows)?;
    if params.iterate == 0 {
        return Err(FiltersError::InvalidParameter {
            name: "iterate",
            value: 0.0,
            requirement: "a positive integer — the number of seasonal-refinement rounds \
                 (statsmodels' default is 2; with a single period only one round \
                 is ever run). Zero rounds would never fit anything",
        });
    }
    let num = periods.len();
    // statsmodels: a single seasonal component needs no refinement rounds.
    let iterate = if num == 1 { 1 } else { params.iterate };

    let mut seasonal: Vec<Vec<f64>> = vec![vec![0.0; n]; num];
    let mut deseas: Vec<f64> = y.to_vec();
    let mut trend: Vec<f64> = Vec::new();
    let mut weights: Vec<f64> = Vec::new();
    for _ in 0..iterate {
        for i in 0..num {
            // Add this period's current seasonal back, re-extract it from
            // the series deseasonalized of all the others, subtract the
            // refreshed estimate — the exact statsmodels operation order.
            for (d, s) in deseas.iter_mut().zip(&seasonal[i]) {
                *d += *s;
            }
            let sp = StlParams {
                seasonal: windows[i],
                ..params.stl.clone()
            };
            let r = stl(&deseas, periods[i], &sp)?;
            for (d, s) in deseas.iter_mut().zip(&r.seasonal) {
                *d -= *s;
            }
            seasonal[i] = r.seasonal;
            trend = r.trend;
            weights = r.weights;
        }
    }
    let resid: Vec<f64> = deseas.iter().zip(&trend).map(|(&d, &t)| d - t).collect();
    Ok(MstlResult {
        periods,
        windows,
        seasonal,
        trend,
        resid,
        weights,
        iterate,
        dropped_periods,
    })
}

// ------------------------------------------------------ strength measures

/// Per-period Wang-Smith-Hyndman seasonal strengths from an MSTL fit:
/// `max(0, 1 - Var(resid) / Var(seasonal_k + resid))` for each seasonal
/// component, in the order of [`MstlResult::periods`] (sample variances,
/// denominator `n - 1`; a zero-variance denominator yields 0) — the same
/// guarded formula as [`crate::strength_from_components`], applied per
/// component.
///
/// Like `strength_from_components`, this is deliberately unguarded against
/// a constant *input* series (it cannot see the input): the ratio of a
/// constant series' float-noise variances is implementation noise, not a
/// measurement, so do not hand it a decomposition of a constant — the
/// Python binding checks the input and reports no strengths in that case,
/// matching the [`crate::seasonal_strength`] refusal.
pub fn mstl_seasonal_strengths(result: &MstlResult) -> Vec<f64> {
    let vr = sample_var(result.resid.iter().copied());
    result
        .seasonal
        .iter()
        .map(|s| {
            let vsr = sample_var(s.iter().zip(&result.resid).map(|(&s, &r)| s + r));
            if vsr > 0.0 {
                (1.0 - vr / vsr).max(0.0)
            } else {
                0.0
            }
        })
        .collect()
}
