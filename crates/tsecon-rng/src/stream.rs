//! The user-facing random stream: a seeded Philox engine plus the NumPy
//! `Generator` output conventions (u32 splitting, 53-bit uniforms).

use crate::error::RngError;
use crate::philox::Philox;
use crate::seedseq::{reserve_children, SeedSequence};

/// 2^-53, the spacing of the 53-bit uniform grid. `(1u64 << 53) as f64` is
/// exact, so this constant is exactly 2^-53.
const UINT53_TO_F64: f64 = 1.0 / (1u64 << 53) as f64;

/// A reproducible random stream: Philox-4x64-10 seeded through
/// [`SeedSequence`], with output conventions bit-identical to NumPy's
/// `Generator(Philox(seed))`.
///
/// - [`Stream::next_u64`] matches `Philox.random_raw()`.
/// - [`Stream::uniform_f64`] matches `Generator.random()` (f64 path).
/// - [`Stream::next_u32`] matches NumPy's `next_uint32`: one buffered `u64`
///   yields the low half first, then the high half.
///
/// `Clone + Send + Sync`: the state is plain data, so streams can be moved
/// into rayon tasks. Cloning duplicates the state — a clone replays the
/// same sequence. For *independent* parallel streams use
/// [`Stream::substreams`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Stream {
    engine: Philox,
    /// Stashed high half of the last `u64` split by `next_u32`.
    half: Option<u32>,
}

impl Stream {
    /// Create a stream from an integer seed, matching
    /// `numpy.random.Philox(seed)`: the 128-bit key is
    /// `SeedSequence(seed).generate_state(2, np.uint64)` and the counter
    /// starts at zero.
    pub fn new(seed: u64) -> Self {
        Self::from_seed_sequence(&SeedSequence::new(u128::from(seed)))
    }

    /// Create a stream keyed by an existing [`SeedSequence`] (e.g. one child
    /// of a `spawn`), matching `numpy.random.Philox(seed=seed_seq)`.
    pub fn from_seed_sequence(seq: &SeedSequence) -> Self {
        Stream {
            engine: Philox::from_key_counter(seq.philox_key(), [0; 4]),
            half: None,
        }
    }

    /// Create a stream from an explicit key and counter, matching
    /// `numpy.random.Philox(key=key, counter=counter)` for values below
    /// 2^128. Integers are split into little-endian 64-bit words.
    pub fn from_key_counter(key: u128, counter: u128) -> Self {
        Self::from_raw_key_counter(
            [key as u64, (key >> 64) as u64],
            [counter as u64, (counter >> 64) as u64, 0, 0],
        )
    }

    /// Create a stream from raw little-endian key and counter words (full
    /// 128-bit key / 256-bit counter control).
    pub fn from_raw_key_counter(key: [u64; 2], counter: [u64; 4]) -> Self {
        Stream {
            engine: Philox::from_key_counter(key, counter),
            half: None,
        }
    }

    /// Spawn `n` independent, reproducible substreams from a seed via
    /// SeedSequence spawning — the parallel Monte Carlo contract: replication
    /// `i` always receives the same stream for a given `seed`, regardless of
    /// how replications are scheduled across threads.
    ///
    /// Equivalent to
    /// `[Philox(seed=c) for c in np.random.SeedSequence(seed).spawn(n)]`;
    /// element `i` is [`Stream::substream`]`(seed, i)`. Each stream is built
    /// straight from its child sequence, so only the `n` streams are held.
    ///
    /// # Errors
    ///
    /// [`RngError::SpawnLimitExceeded`] if `n` exceeds the 32-bit spawn
    /// index space; [`RngError::StreamBudgetExceeded`] if `n` exceeds
    /// [`crate::MAX_MATERIALIZED_STREAMS`] (or the allocator refuses the
    /// vector). Both are checked before anything is allocated. A caller
    /// that would otherwise hold one stream per draw for a large `n` should
    /// derive them lazily with [`Stream::substream`] instead.
    pub fn substreams(seed: u64, n: usize) -> Result<Vec<Stream>, RngError> {
        Self::check_substream_count(n)?;
        let mut streams = reserve_children(n)?;
        for i in 0..n {
            // check_substream_count bounds n by 2^32, so i fits a u32.
            streams.push(Self::substream(seed, i as u32));
        }
        Ok(streams)
    }

    /// Substream `index` of `seed`, derived directly: bit-identical to
    /// element `index` of [`Stream::substreams`]`(seed, n)` for every
    /// `n > index` (NumPy's
    /// `Philox(seed=np.random.SeedSequence(seed, spawn_key=(index,)))`, the
    /// `index`-th child of `SeedSequence(seed).spawn(n)`).
    ///
    /// This is the lazy form of the parallel Monte Carlo contract: a loop
    /// over replications derives replication `i`'s stream inside its own
    /// task, so no vector of `n` streams is ever allocated and the memory
    /// cost is O(1) in the replication count.
    pub fn substream(seed: u64, index: u32) -> Stream {
        Self::from_seed_sequence(&SeedSequence::root_child(u128::from(seed), index))
    }

    /// Whether `n` substreams of one seed fit the 32-bit spawn-key index
    /// space (`n <= 2^32`), so that indices `0..n` all address distinct
    /// children. Allocates nothing.
    ///
    /// # Errors
    ///
    /// [`RngError::SpawnLimitExceeded`] when `n > 2^32`.
    pub fn check_substream_count(n: usize) -> Result<(), RngError> {
        let available = u64::from(u32::MAX) + 1;
        if n as u64 > available {
            return Err(RngError::SpawnLimitExceeded {
                requested: n,
                available,
            });
        }
        Ok(())
    }

    /// Next raw 64-bit value, bit-identical to
    /// `numpy.random.Philox.random_raw()`.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.engine.next_u64()
    }

    /// Next 32-bit value using NumPy's split convention
    /// (`philox_next32`): draw a `u64`, return its low 32 bits, and stash
    /// the high 32 bits for the next call.
    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        if let Some(hi) = self.half.take() {
            return hi;
        }
        let v = self.engine.next_u64();
        self.half = Some((v >> 32) as u32);
        v as u32
    }

    /// Next uniform double on `[0, 1)`, bit-identical to NumPy's
    /// `Generator.random()`: `U = (next_u64 >> 11) * 2^-53`, i.e. a 53-bit
    /// integer scaled onto the unit interval (every value is exactly
    /// representable; 0 is possible, 1 is not).
    #[inline]
    pub fn uniform_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * UINT53_TO_F64
    }

    /// Fill `out` with raw 64-bit draws (equivalent to repeated
    /// [`Stream::next_u64`]).
    pub fn fill_u64(&mut self, out: &mut [u64]) {
        for slot in out {
            *slot = self.next_u64();
        }
    }

    /// Fill `out` with 32-bit draws (equivalent to repeated
    /// [`Stream::next_u32`], including the split-halves convention).
    pub fn fill_u32(&mut self, out: &mut [u32]) {
        for slot in out {
            *slot = self.next_u32();
        }
    }

    /// Fill `out` with uniform doubles on `[0, 1)` (equivalent to repeated
    /// [`Stream::uniform_f64`]).
    pub fn fill_uniform_f64(&mut self, out: &mut [f64]) {
        for slot in out {
            *slot = self.uniform_f64();
        }
    }

    /// Advance the stream as if `delta` calls to [`Stream::next_u64`] had
    /// been made, in O(1) time via counter arithmetic (see
    /// [`Philox::advance`] for exact semantics; the counter wraps modulo
    /// 2^256).
    ///
    /// Any half-`u64` stashed by [`Stream::next_u32`] is discarded, mirroring
    /// NumPy's reset of buffered values on `advance`. `delta` counts 64-bit
    /// draws: `uniform_f64` consumes one draw each, `next_u32` consumes one
    /// draw per *pair* of calls.
    pub fn advance(&mut self, delta: u128) {
        self.half = None;
        self.engine.advance(delta);
    }

    /// The 128-bit Philox key as two little-endian `u64` words (record this
    /// alongside results for exact replay).
    pub fn key(&self) -> [u64; 2] {
        self.engine.key()
    }

    /// The 256-bit Philox counter as four little-endian `u64` words (the
    /// index of the most recently generated block).
    pub fn counter(&self) -> [u64; 4] {
        self.engine.counter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn next_u32_splits_u64_low_then_high() {
        let mut a = Stream::new(99);
        let mut b = Stream::new(99);
        let v = a.next_u64();
        assert_eq!(b.next_u32(), v as u32);
        assert_eq!(b.next_u32(), (v >> 32) as u32);
    }

    #[test]
    fn advance_discards_stashed_half() {
        let mut a = Stream::new(5);
        let _ = a.next_u32(); // stashes a high half
        a.advance(0);
        let mut b = Stream::new(5);
        b.advance(1); // one u64 consumed in `a` too
        assert_eq!(a.next_u64(), b.next_u64());
    }
}
