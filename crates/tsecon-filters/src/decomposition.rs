//! The common trend-cycle return type with explicit index-alignment
//! metadata.
//!
//! Filters that lose observations (Baxter-King loses `K` at each end,
//! Hamilton loses `h + p - 1` at the start) must say so explicitly —
//! silent misalignment is the deadliest bug class in applied macro work
//! (see `docs/roadmap/00-architecture.md`). Every filter in this crate
//! therefore returns a [`Decomposition`] whose [`Alignment`] records
//! exactly which input observations the output components correspond to.

/// Index-alignment metadata: which slice of the input series the output
/// components of a [`Decomposition`] line up with.
///
/// Output element `t` corresponds to input observation
/// `t + lost_start`; the output has `input_len - lost_start - lost_end`
/// elements.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Alignment {
    /// Number of observations dropped from the start of the sample.
    pub lost_start: usize,
    /// Number of observations dropped from the end of the sample.
    pub lost_end: usize,
    /// Length of the input series the filter was applied to.
    pub input_len: usize,
}

impl Alignment {
    /// Alignment for a filter that keeps the full sample.
    pub(crate) fn full(input_len: usize) -> Self {
        Alignment {
            lost_start: 0,
            lost_end: 0,
            input_len,
        }
    }

    /// Index into the input series of the first output element
    /// (equal to `lost_start`).
    pub fn first_index(&self) -> usize {
        self.lost_start
    }

    /// Map an output index to the corresponding input-series index.
    ///
    /// Returns `None` when `output_index` is out of range for the output
    /// length implied by this alignment.
    pub fn input_index(&self, output_index: usize) -> Option<usize> {
        if output_index < self.output_len() {
            Some(output_index + self.lost_start)
        } else {
            None
        }
    }

    /// Number of output observations implied by this alignment.
    pub fn output_len(&self) -> usize {
        self.input_len
            .saturating_sub(self.lost_start)
            .saturating_sub(self.lost_end)
    }
}

/// A trend-cycle decomposition of a univariate series, with explicit
/// alignment metadata.
///
/// `cycle` (and `trend`, when present) have length
/// [`Alignment::output_len`], and element `t` corresponds to input
/// observation `t + alignment.lost_start`.
///
/// `trend` is `None` for filters that only extract a cyclical component
/// (Baxter-King). When both components are present their sum
/// reconstructs the (possibly drift-adjusted — see
/// [`cf_filter`](crate::cf_filter)) input over the aligned range.
#[derive(Debug, Clone, PartialEq)]
pub struct Decomposition {
    /// Trend (growth) component, if the filter produces one.
    pub trend: Option<Vec<f64>>,
    /// Cyclical component.
    pub cycle: Vec<f64>,
    /// Which input observations the components correspond to.
    pub alignment: Alignment,
}
