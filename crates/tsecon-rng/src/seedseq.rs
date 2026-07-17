//! A faithful port of `numpy.random.SeedSequence`.
//!
//! `SeedSequence` turns low-quality user entropy (a seed integer) into
//! high-quality, well-distributed generator state, and supports *spawning*
//! hierarchical children whose states are independent of the parent and of
//! each other. It is the mechanism behind reproducible parallel streams:
//! child `i` is identified purely by `(entropy, spawn_key + (i,))`, so the
//! same seed always yields the same tree of streams regardless of scheduling.
//!
//! Algorithm (Robert Kern's design for NumPy, `numpy/random/bit_generator.pyx`,
//! following Melissa O'Neill's 2015 "Developing a seed_seq Alternative"):
//! entropy and spawn-key words are hashed into a 4-word (128-bit) pool with
//! an LCG-based multiplicative hash and cross-mixed so late words affect
//! early pool words; output state is then squeezed from the pool with a
//! second multiplicative hash. All arithmetic is on `u32` modulo 2^32:
//!
//! ```text
//! hashmix(v):  v ^= c;  c *= MULT_A;  v *= c;  v ^= v >> 16   (c persists)
//! mix(x, y):   r = MIX_MULT_L*x - MIX_MULT_R*y;  r ^= r >> 16
//! ```
//!
//! Verified word-for-word against NumPy 1.26 (`generate_state` for `u32` and
//! `u64`, and spawned children) by the golden tests in this crate.

use crate::error::RngError;

/// Pool size in 32-bit words (NumPy's `DEFAULT_POOL_SIZE`).
const POOL_SIZE: usize = 4;
/// Initial hash constant for entropy pool mixing.
const INIT_A: u32 = 0x43B0_D7E5;
/// Hash multiplier for entropy pool mixing.
const MULT_A: u32 = 0x931E_8875;
/// Initial hash constant for state generation.
const INIT_B: u32 = 0x8B51_F9DD;
/// Hash multiplier for state generation.
const MULT_B: u32 = 0x58F3_8DED;
/// Left multiplier of the two-word mixing function.
const MIX_MULT_L: u32 = 0xCA01_F9DD;
/// Right multiplier of the two-word mixing function.
const MIX_MULT_R: u32 = 0x4973_F715;
/// Xorshift distance (half the word width).
const XSHIFT: u32 = 16;

/// Decompose a non-negative integer into little-endian 32-bit words,
/// mirroring NumPy's `_int_to_uint32_array` (zero maps to a single 0 word).
fn int_to_u32_words(mut n: u128) -> Vec<u32> {
    if n == 0 {
        return vec![0];
    }
    let mut words = Vec::with_capacity(4);
    while n > 0 {
        words.push(n as u32);
        n >>= 32;
    }
    words
}

/// The persistent-constant multiplicative hash used while mixing entropy
/// into the pool. The constant `hash_const` threads through *every* call
/// within one `mix_entropy` pass — this ordering is part of the contract.
struct HashMixer {
    hash_const: u32,
}

impl HashMixer {
    fn new() -> Self {
        HashMixer { hash_const: INIT_A }
    }

    /// `v ^= c; c *= MULT_A; v *= c; v ^= v >> 16`.
    fn hashmix(&mut self, mut value: u32) -> u32 {
        value ^= self.hash_const;
        self.hash_const = self.hash_const.wrapping_mul(MULT_A);
        value = value.wrapping_mul(self.hash_const);
        value ^= value >> XSHIFT;
        value
    }
}

/// `r = MIX_MULT_L*x - MIX_MULT_R*y (mod 2^32); r ^= r >> 16`.
fn mix(x: u32, y: u32) -> u32 {
    let mut r = MIX_MULT_L
        .wrapping_mul(x)
        .wrapping_sub(MIX_MULT_R.wrapping_mul(y));
    r ^= r >> XSHIFT;
    r
}

/// Port of `numpy.random.SeedSequence`: hash-mixes user entropy into a
/// 128-bit pool, generates high-quality `u32`/`u64` state words, and spawns
/// independent children for parallel streams.
///
/// `Clone + Send + Sync`: the type is plain data. `generate_state_*` take
/// `&self`; only [`SeedSequence::spawn`] mutates (it records how many
/// children have been spawned so successive spawns never collide).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SeedSequence {
    /// User entropy as little-endian 32-bit words.
    entropy: Vec<u32>,
    /// Position in the spawn tree: one word per ancestor child-index.
    spawn_key: Vec<u32>,
    /// The mixed 128-bit entropy pool.
    pool: [u32; POOL_SIZE],
    /// Children spawned so far (next child gets this index).
    n_children_spawned: u64,
}

impl SeedSequence {
    /// Create a root `SeedSequence` from an integer seed, matching
    /// `numpy.random.SeedSequence(entropy)` for non-negative `entropy`
    /// below 2^128.
    pub fn new(entropy: u128) -> Self {
        Self::with_spawn_key(int_to_u32_words(entropy), Vec::new())
    }

    /// Create a root `SeedSequence` from raw little-endian 32-bit entropy
    /// words, matching `numpy.random.SeedSequence([w0, w1, ...])`. An empty
    /// slice is valid (NumPy accepts an empty list) and mixes a pool from
    /// zero entropy.
    pub fn from_entropy_words(words: &[u32]) -> Self {
        Self::with_spawn_key(words.to_vec(), Vec::new())
    }

    /// Internal constructor: assemble entropy + spawn key and mix the pool.
    fn with_spawn_key(entropy: Vec<u32>, spawn_key: Vec<u32>) -> Self {
        // NumPy's `get_assembled_entropy`: if there is a spawn key, the run
        // entropy is zero-padded to the pool size first, so that spawn-key
        // words land in the "extra entropy" mixing phase and a short seed
        // can never alias a spawn-key word.
        let mut assembled = entropy.clone();
        if !spawn_key.is_empty() && assembled.len() < POOL_SIZE {
            assembled.resize(POOL_SIZE, 0);
        }
        assembled.extend_from_slice(&spawn_key);

        let mut seq = SeedSequence {
            entropy,
            spawn_key,
            pool: [0; POOL_SIZE],
            n_children_spawned: 0,
        };
        seq.mix_entropy(&assembled);
        seq
    }

    /// Mix assembled entropy words into the pool (NumPy `mix_entropy`).
    fn mix_entropy(&mut self, entropy: &[u32]) {
        let mut hm = HashMixer::new();

        // Seed the pool from the first POOL_SIZE entropy words (hash out
        // zeros if there are fewer).
        for i in 0..POOL_SIZE {
            self.pool[i] = hm.hashmix(entropy.get(i).copied().unwrap_or(0));
        }

        // Cross-mix all pool words so late bits can affect earlier bits.
        for i_src in 0..POOL_SIZE {
            for i_dst in 0..POOL_SIZE {
                if i_src != i_dst {
                    let hashed = hm.hashmix(self.pool[i_src]);
                    self.pool[i_dst] = mix(self.pool[i_dst], hashed);
                }
            }
        }

        // Fold any remaining entropy words into every pool word.
        for &word in entropy.iter().skip(POOL_SIZE) {
            for i_dst in 0..POOL_SIZE {
                let hashed = hm.hashmix(word);
                self.pool[i_dst] = mix(self.pool[i_dst], hashed);
            }
        }
    }

    /// Squeeze state words out of the pool into `out` (NumPy
    /// `generate_state` with `dtype=uint32`). Deterministic in `&self`;
    /// repeated calls return the same words.
    fn generate_state_into(&self, out: &mut [u32]) {
        let mut hash_const = INIT_B;
        for (i, dst) in out.iter_mut().enumerate() {
            let mut v = self.pool[i % POOL_SIZE];
            v ^= hash_const;
            hash_const = hash_const.wrapping_mul(MULT_B);
            v = v.wrapping_mul(hash_const);
            v ^= v >> XSHIFT;
            *dst = v;
        }
    }

    /// `n` well-distributed 32-bit state words, bit-identical to
    /// `numpy.random.SeedSequence.generate_state(n, np.uint32)`.
    pub fn generate_state_u32(&self, n: usize) -> Vec<u32> {
        let mut out = vec![0u32; n];
        self.generate_state_into(&mut out);
        out
    }

    /// `n` well-distributed 64-bit state words, bit-identical to
    /// `numpy.random.SeedSequence.generate_state(n, np.uint64)`: `2n` 32-bit
    /// words are generated and each consecutive pair is packed little-endian
    /// (`lo | hi << 32`).
    pub fn generate_state_u64(&self, n: usize) -> Vec<u64> {
        let words = self.generate_state_u32(2 * n);
        words
            .chunks_exact(2)
            .map(|pair| u64::from(pair[0]) | (u64::from(pair[1]) << 32))
            .collect()
    }

    /// The 128-bit Philox key derived from this sequence, as NumPy's
    /// `Philox(seed=...)` does with `generate_state(2, np.uint64)`.
    pub(crate) fn philox_key(&self) -> [u64; 2] {
        let mut words = [0u32; 4];
        self.generate_state_into(&mut words);
        [
            u64::from(words[0]) | (u64::from(words[1]) << 32),
            u64::from(words[2]) | (u64::from(words[3]) << 32),
        ]
    }

    /// Spawn `n` independent child sequences, matching
    /// `numpy.random.SeedSequence.spawn(n)`.
    ///
    /// Child `i` (counting all children ever spawned from this sequence) is
    /// `SeedSequence(entropy, spawn_key + (i,))`, so spawning is
    /// deterministic, order-stable, and collision-free across successive
    /// calls: `spawn(2)` then `spawn(3)` yields the same five children as a
    /// single `spawn(5)`.
    ///
    /// # Errors
    ///
    /// [`RngError::SpawnLimitExceeded`] if the total number of children
    /// would exceed the 32-bit spawn-key index space.
    pub fn spawn(&mut self, n: usize) -> Result<Vec<SeedSequence>, RngError> {
        let available = (u64::from(u32::MAX) + 1).saturating_sub(self.n_children_spawned);
        if n as u64 > available {
            return Err(RngError::SpawnLimitExceeded {
                requested: n,
                available,
            });
        }
        let mut children = Vec::with_capacity(n);
        for i in 0..n as u64 {
            let mut key = self.spawn_key.clone();
            key.push((self.n_children_spawned + i) as u32);
            children.push(Self::with_spawn_key(self.entropy.clone(), key));
        }
        self.n_children_spawned += n as u64;
        Ok(children)
    }

    /// The user entropy as little-endian 32-bit words.
    pub fn entropy_words(&self) -> &[u32] {
        &self.entropy
    }

    /// This sequence's position in the spawn tree (empty for a root).
    pub fn spawn_key(&self) -> &[u32] {
        &self.spawn_key
    }

    /// Number of children spawned from this sequence so far.
    pub fn children_spawned(&self) -> u64 {
        self.n_children_spawned
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)] // tests may unwrap

    use super::*;

    #[test]
    fn int_to_words_little_endian() {
        assert_eq!(int_to_u32_words(0), vec![0]);
        assert_eq!(int_to_u32_words(42), vec![42]);
        assert_eq!(int_to_u32_words(1 << 32), vec![0, 1]);
        assert_eq!(
            int_to_u32_words(u128::MAX),
            vec![u32::MAX, u32::MAX, u32::MAX, u32::MAX]
        );
    }

    #[test]
    fn generate_state_is_idempotent() {
        let ss = SeedSequence::new(7);
        assert_eq!(ss.generate_state_u32(8), ss.generate_state_u32(8));
        // Prefix property: longer requests extend, never change, the prefix.
        let short = ss.generate_state_u32(4);
        let long = ss.generate_state_u32(8);
        assert_eq!(&long[..4], &short[..]);
    }

    #[test]
    fn u64_state_packs_u32_pairs_little_endian() {
        let ss = SeedSequence::new(123);
        let w32 = ss.generate_state_u32(8);
        let w64 = ss.generate_state_u64(4);
        for i in 0..4 {
            assert_eq!(
                w64[i],
                u64::from(w32[2 * i]) | (u64::from(w32[2 * i + 1]) << 32)
            );
        }
    }

    #[test]
    fn spawn_limit_errors() {
        let mut ss = SeedSequence::new(1);
        ss.n_children_spawned = u64::from(u32::MAX) + 1;
        let err = ss.spawn(1).unwrap_err();
        assert_eq!(
            err,
            RngError::SpawnLimitExceeded {
                requested: 1,
                available: 0
            }
        );
    }
}
