//! The replication driver: run a closure once per bootstrap replication
//! (or Monte Carlo draw), each with its own reproducible RNG substream.
//!
//! This is the library's headline reproducibility contract: replication
//! `i` always receives the substream spawned at index `i` from
//! `SeedSequence(seed)`, and results are collected in replication order,
//! so [`par_replicate`] returns *bit-identical* output at any thread
//! count — 1 thread, 14 threads, or whatever rayon's pool happens to be.

use rayon::prelude::*;
use tsecon_rng::Stream;

use crate::error::BootstrapError;

/// Run `f(rep_index, &mut stream)` for `rep_index = 0..n_reps`
/// sequentially, one SeedSequence-spawned substream per replication.
///
/// Produces exactly the same output as [`par_replicate`] with the same
/// `(seed, n_reps, f)` — use this in contexts where rayon must not be
/// touched (e.g. inside an already-parallel outer loop, to avoid
/// oversubscription).
///
/// The closure may be `FnMut` (sequential execution imposes no sharing);
/// any closure accepted by [`par_replicate`] is accepted here too.
///
/// # Errors
///
/// [`BootstrapError::Rng`] if spawning `n_reps` substreams exceeds the
/// SeedSequence spawn limit.
pub fn replicate<T, F>(seed: u64, n_reps: usize, mut f: F) -> Result<Vec<T>, BootstrapError>
where
    F: FnMut(usize, &mut Stream) -> T,
{
    let mut streams = Stream::substreams(seed, n_reps)?;
    Ok(streams
        .iter_mut()
        .enumerate()
        .map(|(i, stream)| f(i, stream))
        .collect())
}

/// Run `f(rep_index, &mut stream)` for `rep_index = 0..n_reps` in parallel
/// over rayon's current thread pool, one SeedSequence-spawned substream
/// per replication.
///
/// # Reproducibility contract
///
/// The output vector is **bit-identical for any thread count**, because
/// replication `i`'s result depends only on `(i, substream_i)` — never on
/// scheduling — and rayon's indexed collect places each result at its
/// replication index. `par_replicate(seed, n, f)` equals
/// `replicate(seed, n, f)` element for element.
///
/// To pin the thread count, run inside an explicit pool:
/// `rayon::ThreadPoolBuilder::new().num_threads(k).build()?.install(|| ...)`.
///
/// `f` must be `Fn + Send + Sync` (shared across worker threads); per-rep
/// mutable state belongs inside the closure body or in the returned `T`.
///
/// # Errors
///
/// [`BootstrapError::Rng`] if spawning `n_reps` substreams exceeds the
/// SeedSequence spawn limit.
///
/// # Example
///
/// ```
/// use tsecon_bootstrap::{indices, par_replicate, replicate, BlockScheme};
///
/// let scheme = BlockScheme::Stationary { p: 0.2 };
/// let draw = |_rep: usize, stream: &mut tsecon_rng::Stream| {
///     indices(scheme, 25, stream)
/// };
/// let parallel = par_replicate(20260716, 32, draw).unwrap();
/// let sequential = replicate(20260716, 32, draw).unwrap();
/// assert_eq!(parallel, sequential);
/// ```
pub fn par_replicate<T, F>(seed: u64, n_reps: usize, f: F) -> Result<Vec<T>, BootstrapError>
where
    T: Send,
    F: Fn(usize, &mut Stream) -> T + Send + Sync,
{
    let streams = Stream::substreams(seed, n_reps)?;
    Ok(streams
        .into_par_iter()
        .enumerate()
        .map(|(i, mut stream)| f(i, &mut stream))
        .collect())
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn zero_replications_is_empty() {
        let out: Vec<u64> = replicate(1, 0, |_, s| s.next_u64()).unwrap();
        assert!(out.is_empty());
        let out: Vec<u64> = par_replicate(1, 0, |_, s| s.next_u64()).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn replications_receive_distinct_substreams() {
        let out: Vec<u64> = replicate(42, 16, |_, s| s.next_u64()).unwrap();
        let mut dedup = out.clone();
        dedup.sort_unstable();
        dedup.dedup();
        assert_eq!(dedup.len(), out.len(), "substreams must not collide");
    }

    #[test]
    fn rep_index_matches_position() {
        let out: Vec<usize> = par_replicate(7, 100, |i, _| i).unwrap();
        assert_eq!(out, (0..100).collect::<Vec<_>>());
    }
}
