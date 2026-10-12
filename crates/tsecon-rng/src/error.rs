//! Error type for the `tsecon-rng` crate.

use core::fmt;

/// Errors produced by the `tsecon-rng` crate.
///
/// The generators themselves are infallible (counter arithmetic wraps by
/// design); errors only arise from resource-limit violations in the seeding
/// hierarchy: the 32-bit spawn-key space, and the memory budget on how many
/// children or streams one call materializes at once.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum RngError {
    /// A [`crate::SeedSequence::spawn`] call would push the total number of
    /// children past the `u32` spawn-key space (NumPy stores each child index
    /// as one 32-bit spawn-key word; so do we).
    SpawnLimitExceeded {
        /// Number of children requested by this call.
        requested: usize,
        /// Number of children that can still be spawned from this sequence.
        available: u64,
    },
    /// A [`crate::SeedSequence::spawn`] or [`crate::Stream::substreams`]
    /// call asked to materialize more children (streams) at once than the
    /// memory budget [`crate::MAX_MATERIALIZED_STREAMS`] allows, or the
    /// allocator refused the vector. Checked before anything is allocated,
    /// so the refusal is an error rather than an aborted process; a caller
    /// that needs more streams derives them one at a time with
    /// [`crate::Stream::substream`] / [`crate::SeedSequence::child`].
    StreamBudgetExceeded {
        /// Number of children (streams) requested by this call.
        requested: usize,
        /// The most one call materializes.
        max: usize,
    },
}

impl fmt::Display for RngError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RngError::SpawnLimitExceeded {
                requested,
                available,
            } => write!(
                f,
                "cannot spawn {requested} children: only {available} spawn-key \
                 indices remain (u32 limit)"
            ),
            RngError::StreamBudgetExceeded { requested, max } => write!(
                f,
                "cannot hold {requested} random substreams at once: one call materializes \
                 at most {max} (2^24, a budget of about 2 GiB of generator state), checked \
                 before anything is allocated; request fewer draws or replications"
            ),
        }
    }
}

impl std::error::Error for RngError {}
