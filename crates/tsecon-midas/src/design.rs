//! The mixed-frequency stacked-lag design builder.
//!
//! MIDAS regressions line up a high-frequency predictor against a
//! low-frequency target: each low-frequency period `t` (say a quarter) sees
//! `ratio` high-frequency observations (say three months), and the regression
//! uses the most recent `K` of them as `K` separate stacked columns. This
//! module owns that index bookkeeping — the single most bug-prone part of any
//! MIDAS implementation (off-by-one in the high-frequency lag embedding, R
//! `midasr`'s `mls`).
//!
//! ## Alignment convention (matches `fixtures/midas.json`)
//!
//! High-frequency observations are indexed `0, 1, 2, ...`; low-frequency
//! period `t` (0-indexed) closes at high-frequency index
//! `(t + 1) * ratio - 1` — its last constituent high-frequency observation.
//! The stacked columns are **most-recent-first**: column `k` (for
//! `k = 1..=K`) at period `t` holds
//!
//! ```text
//! stack[k - 1][t] = hf[(t + 1) * ratio - k],
//! ```
//!
//! so column 1 is the last high-frequency obs of period `t`, column 2 the one
//! before it, and so on back `K` steps. Consequently column `c + ratio`
//! equals column `c` lagged one whole low-frequency period — exactly the
//! structure of the fixture's `X_stacked` (columns 4–6 are columns 1–3 lagged
//! one quarter at `ratio = 3`).
//!
//! ## Ragged edge / warm-up
//!
//! The first low-frequency periods lack `K` high-frequency lags of history and
//! are dropped: the first usable period is
//! `first = ceil(K / ratio) - 1`, and the design spans `t = first ..
//! n_low`. This basic builder assumes the high-frequency series is aligned so
//! that `hf.len() >= n_low * ratio` and treats any trailing high-frequency
//! observations beyond `n_low * ratio` as not-yet-usable leads.
//!
//! `// TODO(phase0)`: first-class ragged-edge *leads* (partially observed
//! current period), publication-lag offsets, and release-calendar-driven
//! alignment live in the wider nowcasting module (ROADMAP §8); this builder
//! covers the balanced most-recent-first embedding the MIDAS estimators need.

use crate::error::MidasError;

/// A stacked mixed-frequency design produced by [`stack_high_freq_lags`].
#[derive(Debug, Clone, PartialEq)]
pub struct StackedDesign {
    /// The `K` high-frequency lag columns, most-recent-first (column 0 is the
    /// most recent lag). Each column has length [`nobs`](StackedDesign::nobs).
    pub columns: Vec<Vec<f64>>,
    /// The low-frequency target aligned to [`columns`](StackedDesign::columns):
    /// `low[first ..]`, where `first` is the warm-up offset.
    pub target: Vec<f64>,
    /// Index into the original low-frequency series of the first usable
    /// period (the warm-up offset `ceil(K / ratio) - 1`).
    pub first_low_period: usize,
    /// Number of usable low-frequency periods (rows), i.e. the common length
    /// of `target` and every column.
    pub nobs: usize,
}

/// Build the most-recent-first stacked high-frequency-lag design.
///
/// `hf` is the high-frequency predictor, `low` the low-frequency target,
/// `ratio` the number of high-frequency periods per low-frequency period, and
/// `n_hf_lags = K` the number of stacked lag columns. Returns the `K` columns,
/// the aligned target, and the warm-up offset (see the module docs for the
/// exact `stack[k - 1][t] = hf[(t + 1) * ratio - k]` convention).
///
/// # Errors
///
/// * [`MidasError::InvalidLagCount`] if `n_hf_lags == 0`;
/// * [`MidasError::InvalidWeightParam`] (`name = "ratio"`) if `ratio == 0`;
/// * [`MidasError::NonFinite`] if `hf` or `low` contains a NaN/inf value;
/// * [`MidasError::DimensionMismatch`] if `hf.len() < low.len() * ratio`
///   (the high-frequency series does not cover every low-frequency period);
/// * [`MidasError::SeriesTooShort`] if there is no low-frequency period with
///   `K` high-frequency lags of history.
pub fn stack_high_freq_lags(
    hf: &[f64],
    low: &[f64],
    ratio: usize,
    n_hf_lags: usize,
) -> Result<StackedDesign, MidasError> {
    if n_hf_lags == 0 {
        return Err(MidasError::InvalidLagCount {
            what: "stacked high-frequency design",
            k: n_hf_lags,
            needed: 1,
        });
    }
    if ratio == 0 {
        return Err(MidasError::InvalidWeightParam {
            what: "stacked high-frequency design",
            name: "ratio",
            value: 0.0,
            requirement: "a frequency ratio of at least 1",
        });
    }
    check_finite(hf, "high-frequency series")?;
    check_finite(low, "low-frequency target")?;

    let n_low = low.len();
    let covered = n_low
        .checked_mul(ratio)
        .ok_or(MidasError::DimensionMismatch {
            what: "high-frequency coverage (n_low * ratio overflow)",
            expected: usize::MAX,
            got: hf.len(),
        })?;
    if hf.len() < covered {
        return Err(MidasError::DimensionMismatch {
            what: "high-frequency series length vs low-frequency coverage",
            expected: covered,
            got: hf.len(),
        });
    }

    // Warm-up: first low-frequency period with K high-frequency lags of
    // history. Need (first + 1) * ratio >= n_hf_lags, i.e.
    // first >= ceil(n_hf_lags / ratio) - 1.
    let ceil_div = n_hf_lags.div_ceil(ratio);
    let first = ceil_div - 1;
    if n_low <= first {
        // Minimum HF observations to yield one usable row.
        let needed = ceil_div * ratio;
        return Err(MidasError::SeriesTooShort {
            what: "stacked high-frequency design",
            n: hf.len(),
            needed,
        });
    }
    let nobs = n_low - first;

    let mut columns = vec![Vec::with_capacity(nobs); n_hf_lags];
    for t in first..n_low {
        // Most recent high-frequency index closing low-frequency period t.
        let recent = (t + 1) * ratio - 1;
        for (k, col) in columns.iter_mut().enumerate() {
            // Column k (0-indexed) is lag (k + 1): recent - k.
            col.push(hf[recent - k]);
        }
    }

    Ok(StackedDesign {
        columns,
        target: low[first..].to_vec(),
        first_low_period: first,
        nobs,
    })
}

/// Non-finite guard shared by the mixed-frequency builder.
fn check_finite(x: &[f64], what: &'static str) -> Result<(), MidasError> {
    for (i, &v) in x.iter().enumerate() {
        if !v.is_finite() {
            return Err(MidasError::NonFinite {
                what,
                index: i,
                value: v,
            });
        }
    }
    Ok(())
}
