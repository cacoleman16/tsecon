//! # tsecon-rng — the reproducibility backbone of tsecon
//!
//! Counter-based random number generation, bit-compatible with NumPy:
//!
//! - [`Philox`]: the Philox-4x64-10 engine of Salmon, Moraes, Dror and Shaw
//!   (2011), matching `numpy.random.Philox` raw output bit for bit, with
//!   O(1) [`Philox::advance`] via counter arithmetic.
//! - [`SeedSequence`]: a faithful port of `numpy.random.SeedSequence`
//!   (entropy pool mixing, `generate_state` for `u32`/`u64`, and `spawn`
//!   with spawn-key propagation for hierarchical independent streams).
//! - [`Stream`]: the user-facing API — `Stream::new(seed)`,
//!   `Stream::from_key_counter`, raw and uniform draws with NumPy
//!   `Generator` output conventions, and [`Stream::substreams`] for
//!   reproducible parallel Monte Carlo.
//!
//! ## Reproducibility contract
//!
//! Every draw in the library is traceable to a user seed: no function here
//! creates entropy from the OS. Substreams are keyed by
//! `(seed, replication_index)` through SeedSequence spawning, so a parallel
//! bootstrap produces bit-identical results at any thread count — move each
//! [`Stream`] into its task (all types are `Clone + Send + Sync`).
//!
//! ```
//! use tsecon_rng::Stream;
//!
//! let mut streams = Stream::substreams(20260716, 4).unwrap();
//! let draws: Vec<f64> = streams.iter_mut().map(|s| s.uniform_f64()).collect();
//! // Same seed, same substream, same draw — forever, on every platform.
//! assert_eq!(draws, Stream::substreams(20260716, 4).unwrap()
//!     .iter_mut().map(|s| s.uniform_f64()).collect::<Vec<_>>());
//! ```

#![warn(missing_docs)]
#![warn(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod error;
mod philox;
mod seedseq;
mod stream;

pub use error::RngError;
pub use philox::Philox;
pub use seedseq::SeedSequence;
pub use stream::Stream;

#[cfg(test)]
mod send_sync_assertions {
    use super::*;

    fn assert_send_sync<T: Send + Sync + Clone>() {}

    #[test]
    fn all_types_are_send_sync_clone() {
        assert_send_sync::<Philox>();
        assert_send_sync::<SeedSequence>();
        assert_send_sync::<Stream>();
        assert_send_sync::<RngError>();
    }
}
