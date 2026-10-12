//! The replication driver: run a closure once per bootstrap replication
//! (or Monte Carlo draw), each with its own reproducible RNG substream.
//!
//! This is the library's headline reproducibility contract: replication
//! `i` always receives the substream spawned at index `i` from
//! `SeedSequence(seed)`, and results are collected in replication order,
//! so [`par_replicate`] returns *bit-identical* output at any thread
//! count — 1 thread, 14 threads, or whatever rayon's pool happens to be.
//!
//! ## Memory
//!
//! Replication `i`'s stream is derived inside its own task
//! ([`Stream::substream`]`(seed, i)`, bit-identical to element `i` of
//! [`Stream::substreams`]), so no vector of `n_reps` streams is ever held:
//! that vector used to be the first allocation sized by the replication
//! count, and `n_reps = 2^31` aborted the process on a 154 GB request
//! before any caller could check its own budget. The only allocation the
//! driver itself makes is the result vector, refused beyond
//! [`MAX_REPLICATION_BYTES`] before it is reserved.

use rayon::prelude::*;
use tsecon_rng::Stream;

use crate::error::BootstrapError;

/// The largest result vector one [`replicate`] / [`par_replicate`] call
/// reserves, in bytes: 2^31 (2 GiB) — the house 2^28-double working-set
/// convention, so at most 2^28 replications of an `f64` statistic.
///
/// Checked *before* the vector is reserved, because `try_reserve` alone is
/// not a portable guard (macOS commits an oversized request lazily and the
/// process is killed when it is written); `try_reserve` remains as the
/// second line of defence for a request inside the budget the machine
/// cannot serve. Callers apply their own, tighter budgets naming the user's
/// keyword first (see [`MAX_TEST_REPLICATIONS`]).
pub const MAX_REPLICATION_BYTES: usize = 1 << 31;

/// The most replications a bootstrap *p-value* (one statistic per
/// replication) accepts: 1 000 000.
///
/// The Monte Carlo standard error of a bootstrap p-value from `B`
/// replications is `sqrt(p (1 - p) / B) <= 0.5 / sqrt(B)`, at most 0.0005
/// at the cap, so a larger `B` buys precision past the third decimal at a
/// run time that grows linearly (a request of 2^31 used to abort the process
/// allocating its RNG streams). Shared by the fixed-regressor bootstrap tests (`setar_test`,
/// `threshold_var_test`, `hansen_seo_test`), which refuse a larger `n_boot`
/// by name before any work through [`check_test_replications`].
pub const MAX_TEST_REPLICATIONS: usize = 1_000_000;

/// Whether `n_boot` replications fit [`MAX_TEST_REPLICATIONS`]; allocates
/// nothing.
///
/// # Errors
///
/// [`BootstrapError::TooManyReplications`] (with `max` the cap) when
/// `n_boot > MAX_TEST_REPLICATIONS`.
pub fn check_test_replications(n_boot: usize) -> Result<(), BootstrapError> {
    if n_boot > MAX_TEST_REPLICATIONS {
        return Err(BootstrapError::TooManyReplications {
            n_reps: n_boot,
            max: MAX_TEST_REPLICATIONS,
        });
    }
    Ok(())
}

/// Check `n_reps` against the 2^32 spawn-key space and the result budget,
/// then reserve the result vector — all before any replication runs.
fn reserve_results<T>(n_reps: usize) -> Result<Vec<T>, BootstrapError> {
    // A zero-sized result never allocates; the spawn-key space bounds it.
    let max = MAX_REPLICATION_BYTES
        .checked_div(core::mem::size_of::<T>())
        .unwrap_or(usize::MAX);
    let refused = BootstrapError::TooManyReplications { n_reps, max };
    if n_reps > max {
        return Err(refused);
    }
    Stream::check_substream_count(n_reps)?;
    let mut out = Vec::new();
    out.try_reserve_exact(n_reps).map_err(|_| refused)?;
    Ok(out)
}

/// Replication `rep`'s stream. `reserve_results` bounded the replication
/// count by the 2^32 spawn-key space, so every index fits a `u32`.
#[inline]
fn stream_of(seed: u64, rep: usize) -> Stream {
    Stream::substream(seed, rep as u32)
}

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
/// [`BootstrapError::TooManyReplications`] if the result vector would
/// exceed [`MAX_REPLICATION_BYTES`] (or the allocator refuses it);
/// [`BootstrapError::Rng`] if `n_reps` exceeds the 2^32 SeedSequence spawn
/// limit. Both are checked before any replication runs.
pub fn replicate<T, F>(seed: u64, n_reps: usize, mut f: F) -> Result<Vec<T>, BootstrapError>
where
    F: FnMut(usize, &mut Stream) -> T,
{
    let mut out = reserve_results(n_reps)?;
    for i in 0..n_reps {
        let mut stream = stream_of(seed, i);
        out.push(f(i, &mut stream));
    }
    Ok(out)
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
/// [`BootstrapError::TooManyReplications`] if the result vector would
/// exceed [`MAX_REPLICATION_BYTES`] (or the allocator refuses it);
/// [`BootstrapError::Rng`] if `n_reps` exceeds the 2^32 SeedSequence spawn
/// limit. Both are checked before any replication runs; no vector of
/// streams is allocated (each replication derives its own).
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
    let mut out = reserve_results(n_reps)?;
    // collect_into_vec writes into the reserved capacity (its own reserve
    // is then a no-op), each result at its replication index.
    (0..n_reps)
        .into_par_iter()
        .map(|i| {
            let mut stream = stream_of(seed, i);
            f(i, &mut stream)
        })
        .collect_into_vec(&mut out);
    Ok(out)
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

    /// The lazy driver hands replication `i` exactly the stream the
    /// materialized `Stream::substreams(seed, n)[i]` (the pre-0.11 driver)
    /// did, so every seeded result is unchanged.
    #[test]
    fn lazy_driver_matches_materialized_substreams() {
        let n = 1500;
        let draw = |_: usize, s: &mut Stream| (s.next_u64(), s.uniform_f64(), s.next_u32());
        let want: Vec<(u64, f64, u32)> = Stream::substreams(20260716, n)
            .unwrap()
            .iter_mut()
            .map(|s| (s.next_u64(), s.uniform_f64(), s.next_u32()))
            .collect();
        assert_eq!(par_replicate(20260716, n, draw).unwrap(), want);
        assert_eq!(replicate(20260716, n, draw).unwrap(), want);
    }

    /// Replication counts past the budgets are typed errors raised before
    /// any replication runs or any result is allocated.
    #[test]
    fn oversized_replication_counts_are_refused_before_running() {
        let max_f64 = MAX_REPLICATION_BYTES / 8;
        assert_eq!(max_f64, 1 << 28);
        for n in [max_f64 + 1, 1 << 31, 1 << 40] {
            let want = BootstrapError::TooManyReplications {
                n_reps: n,
                max: max_f64,
            };
            let ran = std::sync::atomic::AtomicUsize::new(0);
            let r = par_replicate(1, n, |_, s| {
                ran.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                s.uniform_f64()
            });
            assert_eq!(r.unwrap_err(), want);
            assert_eq!(replicate(1, n, |_, s| s.uniform_f64()).unwrap_err(), want);
            assert_eq!(ran.into_inner(), 0);
        }
        // A zero-sized result is bounded by the spawn-key space instead.
        let past = (1usize << 32) + 1;
        assert!(matches!(
            par_replicate(1, past, |_, _| ()).unwrap_err(),
            BootstrapError::Rng(tsecon_rng::RngError::SpawnLimitExceeded { .. })
        ));
    }

    #[test]
    fn test_replication_cap_boundary() {
        assert_eq!(MAX_TEST_REPLICATIONS, 1_000_000);
        assert!(check_test_replications(MAX_TEST_REPLICATIONS).is_ok());
        assert_eq!(
            check_test_replications(MAX_TEST_REPLICATIONS + 1).unwrap_err(),
            BootstrapError::TooManyReplications {
                n_reps: MAX_TEST_REPLICATIONS + 1,
                max: MAX_TEST_REPLICATIONS,
            }
        );
    }
}
