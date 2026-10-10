//! Error types for `tsecon-filters`.

use core::fmt;

/// Errors produced by the trend-cycle filters in this crate.
///
/// All fallible library entry points return `Result<_, FiltersError>`;
/// nothing in the non-test code path panics.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FiltersError {
    /// A filter parameter is outside its valid domain (e.g. `lambda < 0`
    /// for the Hodrick-Prescott filter, or `low < 2` for a band-pass
    /// filter, which would place the upper band edge above the Nyquist
    /// frequency).
    InvalidParameter {
        /// Name of the offending parameter.
        name: &'static str,
        /// The invalid value that was supplied.
        value: f64,
        /// Human-readable statement of the violated constraint.
        requirement: &'static str,
    },
    /// The input series is too short for the requested filter (e.g. a
    /// Baxter-King filter with truncation `K` needs at least `2K + 1`
    /// observations to produce a single output point).
    SeriesTooShort {
        /// Name of the filter that rejected the series.
        filter: &'static str,
        /// Minimum number of observations required.
        needed: usize,
        /// Number of observations supplied.
        got: usize,
        /// Where the requirement comes from, and what to change.
        why: &'static str,
    },
    /// A band-pass frequency band is malformed: the two period bounds are
    /// validated together, so the message names both.
    InvalidBand {
        /// The shorter period bound that was supplied.
        low: f64,
        /// The longer period bound that was supplied.
        high: f64,
    },
    /// The input series contains a NaN or infinite value. Filters propagate
    /// non-finite values in surprising, filter-specific ways (a single NaN
    /// contaminates the entire Hodrick-Prescott trend), so they are
    /// rejected up front.
    NonFiniteInput {
        /// Index of the first non-finite observation.
        index: usize,
    },
    /// The regressor matrix of the Hamilton (2018) regression filter is
    /// numerically rank deficient (e.g. the series is constant, so every
    /// lag column is collinear with the intercept).
    RankDeficient {
        /// Name of the computation that encountered the deficiency.
        what: &'static str,
    },
    /// The input series is constant, so a variance-ratio diagnostic is
    /// undefined: both variances in the strength ratio are pure float
    /// noise from the decomposition, and the resulting number is
    /// implementation noise, not a measurement (audit round 6 measured
    /// `seasonal_strength ~ 0.64` — coincidentally at the `nsdiffs`
    /// threshold — on flat lines).
    ConstantSeries {
        /// Name of the diagnostic that rejected the series.
        what: &'static str,
    },
    /// The seasonal period is too long for the series: STL smooths each
    /// cycle-subseries and needs at least two full cycles, `n >= 2 *
    /// period` (R's `stl()` enforces the same bound).
    PeriodTooLong {
        /// Name of the filter (or diagnostic) that rejected the period.
        filter: &'static str,
        /// The period that was supplied.
        period: usize,
        /// Number of observations supplied.
        n: usize,
    },
    /// An explicit STL/MSTL iteration count asks for more LOESS work than
    /// [`crate::STL_WORK_BUDGET`]: `inner_iter`, `outer_iter` and MSTL's
    /// `iterate` are fixed loop counts that never stop early, so an absurd
    /// count would run for hours (holding the GIL, deaf to Ctrl-C). The
    /// counts of the defaults — at most [`crate::STL_FREE_PASSES`] inner
    /// passes per STL fit over at most two MSTL rounds — are never refused.
    IterationBudget {
        /// Name of the filter that refused the counts.
        filter: &'static str,
        /// The count named as the remedy (`"inner_iter"`, `"outer_iter"`
        /// or `"iterate"`).
        param: &'static str,
        /// Its value.
        value: usize,
        /// The resolved number of inner passes.
        inner_iter: usize,
        /// The resolved number of outer (robustness) passes.
        outer_iter: usize,
        /// The MSTL refinement rounds actually run (`None` for STL, and for
        /// an MSTL with a single period, which runs one round whatever
        /// `iterate` says).
        iterate: Option<usize>,
        /// The LOESS evaluation units the request asks for.
        work: u128,
        /// The largest admissible value of `param`, the others held fixed,
        /// or `None` when no value of any single count fits with the others
        /// held fixed (they must be reduced together).
        max: Option<usize>,
        /// The largest admissible number of inner passes over all rounds
        /// (`iterate x inner_iter x (outer_iter + 1)`) within the budget.
        max_passes: usize,
    },
    /// The series is degenerate for the requested filter in a way the
    /// message itself states fully — e.g. constant *first differences*
    /// (an exact linear ramp) for the Beveridge-Nelson filter, whose AR
    /// stage models the changes of the series, not its level. Unlike
    /// [`FiltersError::RankDeficient`] and
    /// [`FiltersError::ConstantSeries`], no generic explanation is
    /// appended: `what` carries the data property, its consequence, and
    /// the remedy.
    Degenerate {
        /// Complete description of the degeneracy and what to do.
        what: &'static str,
    },
}

impl fmt::Display for FiltersError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FiltersError::InvalidParameter {
                name,
                value,
                requirement,
            } => write!(
                f,
                "invalid parameter `{name}` = {value}: requires {requirement}"
            ),
            FiltersError::SeriesTooShort {
                filter,
                needed,
                got,
                why,
            } => write!(
                f,
                "{filter} needs at least {needed} observation{} but got {got}: {why}",
                if *needed == 1 { "" } else { "s" }
            ),
            FiltersError::InvalidBand { low, high } => write!(
                f,
                "invalid band low = {low}, high = {high}: the band is given in *periods* \
                 (observations per cycle) and needs 2 <= low < high — for quarterly \
                 business-cycle work the usual band is low = 6, high = 32"
            ),
            FiltersError::NonFiniteInput { index } => write!(
                f,
                "the input series has a non-finite value (NaN or inf) at index {index}: \
                 filters spread it in surprising ways — one NaN contaminates the entire \
                 Hodrick-Prescott trend — so drop or impute it first \
                 (pandas: s.dropna() or s.interpolate())"
            ),
            FiltersError::RankDeficient { what } => write!(
                f,
                "{what}: the regressor matrix is numerically rank deficient, which \
                 happens when the series is constant (or constant over the regression \
                 window), so every lag column is collinear with the intercept"
            ),
            FiltersError::ConstantSeries { what } => write!(
                f,
                "{what}: the series is constant (zero sample variance), so a \
                 variance-ratio strength measure is undefined — the ratio of the \
                 decomposition's float-noise variances is implementation noise, not \
                 a measurement. A constant series has no seasonality to measure."
            ),
            FiltersError::PeriodTooLong { filter, period, n } => {
                write!(
                    f,
                    "{filter}: period = {period} needs at least 2 * period = {} observations \
                     but the series has {n}: STL smooths each cycle-subseries (all \
                     Januaries, all Februaries, ...) and needs at least two full cycles to \
                     tell the seasonal from the trend (R's stl() enforces the same bound); ",
                    (*period as u128) * 2
                )?;
                if n / 2 >= 2 {
                    write!(f, "pass period <= {}, or supply a longer series", n / 2)
                } else {
                    write!(
                        f,
                        "no period >= 2 fits a series this short, so supply a longer one"
                    )
                }
            }
            FiltersError::IterationBudget {
                filter,
                param,
                value,
                inner_iter,
                outer_iter,
                iterate,
                work,
                max,
                max_passes,
            } => {
                write!(
                    f,
                    "{filter}: {param} = {value} is past the STL work budget: inner_iter = \
                     {inner_iter} and outer_iter = {outer_iter}"
                )?;
                if let Some(r) = iterate {
                    write!(f, " over iterate = {r} rounds of every period")?;
                }
                write!(
                    f,
                    " ask for about {:.2e} LOESS evaluation units at this series length and \
                     these windows, past the budget of 2^32 units (about half a minute, \
                     checked before any smoothing). These are fixed loop counts that never \
                     stop early; ",
                    *work as f64
                )?;
                match max {
                    Some(m) => write!(f, "pass {param} <= {m}"),
                    None => {
                        let passes = if iterate.is_some() {
                            "iterate * inner_iter * (outer_iter + 1)"
                        } else {
                            "inner_iter * (outer_iter + 1)"
                        };
                        write!(
                            f,
                            "no single count fits with the others held fixed, so reduce \
                             them together to {passes} <= {max_passes}, or to the default \
                             region of at most 32 inner passes per fit over at most two rounds"
                        )
                    }
                }
            }
            FiltersError::Degenerate { what } => write!(f, "{what}"),
        }
    }
}

impl std::error::Error for FiltersError {}

/// Reject series containing NaN or infinities, returning the index of the
/// first offender.
pub(crate) fn check_finite(y: &[f64]) -> Result<(), FiltersError> {
    match y.iter().position(|v| !v.is_finite()) {
        Some(index) => Err(FiltersError::NonFiniteInput { index }),
        None => Ok(()),
    }
}
