//! Error type for the `tsecon-rng` crate.

use core::fmt;

/// Errors produced by the `tsecon-rng` crate.
///
/// The generators themselves are infallible (counter arithmetic wraps by
/// design); errors only arise from resource-limit violations in the seeding
/// hierarchy.
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
        }
    }
}

impl std::error::Error for RngError {}
