//! # tsecon-bootstrap — the single resampling engine of tsecon
//!
//! Every bootstrap in the library runs through this crate (ROADMAP §5:
//! one owner per capability — six subtly different wild bootstraps is how
//! a library ships contradictory p-values). It provides:
//!
//! - [`BlockScheme`] + [`indices`]: index-based resampling — iid (Efron
//!   1979), moving block (Künsch 1989), circular block (Politis-Romano
//!   1992), and stationary (Politis-Romano 1994) bootstraps, all emitting
//!   indices into the original sample rather than copying data.
//! - [`WildWeights`]: wild-bootstrap weight generators — Rademacher,
//!   Mammen (1993) two-point, and standard normal.
//! - [`optimal_block_length`]: automatic block-length selection via
//!   Politis-White (2004) with the Patton-Politis-White (2009) correction,
//!   so valid dependent-data inference is the default, not a tuning
//!   exercise.
//! - [`par_replicate`] / [`replicate`]: the replication driver — one
//!   SeedSequence-spawned [`tsecon_rng::Stream`] per replication, results
//!   collected in replication order.
//!
//! ## Reproducibility contract
//!
//! For a given seed, every result of this crate is a pure function of its
//! inputs: [`par_replicate`] is **bit-identical at any thread count** and
//! equals [`replicate`], and each scheme documents its exact
//! stream-consumption order. No function creates entropy from the OS.
//!
//! ```
//! use tsecon_bootstrap::{indices, par_replicate, BlockScheme};
//!
//! let scheme = BlockScheme::MovingBlock { block_length: 5 };
//! let reps = par_replicate(20260716, 8, |_, stream| indices(scheme, 40, stream)).unwrap();
//! // Same seed, same replication, same indices — at any thread count.
//! let again = par_replicate(20260716, 8, |_, stream| indices(scheme, 40, stream)).unwrap();
//! assert_eq!(reps, again);
//! ```

#![warn(missing_docs)]
#![warn(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod blocklength;
mod error;
mod replicate;
mod schemes;
mod wild;

pub use blocklength::{optimal_block_length, OptimalBlockLength};
pub use error::BootstrapError;
pub use replicate::{par_replicate, replicate};
pub use schemes::{indices, BlockScheme};
pub use wild::WildWeights;
