//! Philox counter-based random engine, bit-compatible with `numpy.random.Philox`.
//!
//! NumPy's `Philox` bit generator is the Philox-4x64-10 member of the Philox
//! family of Salmon, Moraes, Dror and Shaw (2011), "Parallel Random Numbers:
//! As Easy as 1, 2, 3" (Proc. SC'11): a 256-bit counter (four little-endian
//! `u64` words), a 128-bit key (two `u64` words), and a 10-round bijective
//! "weak cryptographic" mixing function that maps each counter value to four
//! statistically independent `u64` outputs.
//!
//! One round transforms the counter `(x0, x1, x2, x3)` under key `(k0, k1)` as
//!
//! ```text
//! (x0, x1, x2, x3) -> (hi(M1*x2) ^ x1 ^ k0,  lo(M1*x2),
//!                      hi(M0*x0) ^ x3 ^ k1,  lo(M0*x0))
//! ```
//!
//! where `hi`/`lo` are the high/low 64-bit halves of the 128-bit product, and
//! between rounds the key is bumped by the Weyl constants `k0 += W0`,
//! `k1 += W1`. (The 64-bit Weyl constants extend the well-known 32-bit ones
//! `0x9E3779B9` — the golden ratio — and `0xBB67AE85` — sqrt(3)-1 — used by
//! the 4x32 variant.)
//!
//! Bit-compatibility notes (verified against `numpy.random.Philox`
//! `random_raw`, NumPy 1.26):
//!
//! - The counter is incremented (with carry across the four words, i.e.
//!   modulo 2^256) **before** each block is generated, so a generator whose
//!   stored counter is `c` produces its first block from counter value
//!   `c + 1`.
//! - The four `u64` outputs of a block are consumed in order
//!   `out[0], out[1], out[2], out[3]` before the next block is generated.

/// First 64-bit Philox multiplier (`PHILOX_M4x64_0` in NumPy / Random123).
const PHILOX_M0: u64 = 0xD2E7_470E_E14C_6C93;
/// Second 64-bit Philox multiplier (`PHILOX_M4x64_1` in NumPy / Random123).
const PHILOX_M1: u64 = 0xCA5A_8263_9512_1157;
/// First Weyl key increment: 64-bit extension of the golden ratio 0x9E3779B9.
const PHILOX_W0: u64 = 0x9E37_79B9_7F4A_7C15;
/// Second Weyl key increment: 64-bit extension of sqrt(3)-1 = 0xBB67AE85.
const PHILOX_W1: u64 = 0xBB67_AE85_84CA_A73B;

/// Number of rounds in the standard-strength Philox-4x64-10 recommended by
/// Salmon et al. (2011) and used by NumPy.
const ROUNDS: usize = 10;

/// Outputs buffered per counter block.
const BUFFER_SIZE: usize = 4;

/// 128-bit multiply returning `(hi, lo)` 64-bit halves.
#[inline(always)]
fn mulhilo64(a: u64, b: u64) -> (u64, u64) {
    let p = u128::from(a) * u128::from(b);
    ((p >> 64) as u64, p as u64)
}

/// One Philox-4x64 round (see module docs for the formula).
#[inline(always)]
fn round(ctr: [u64; 4], key: [u64; 2]) -> [u64; 4] {
    let (hi0, lo0) = mulhilo64(PHILOX_M0, ctr[0]);
    let (hi1, lo1) = mulhilo64(PHILOX_M1, ctr[2]);
    [hi1 ^ ctr[1] ^ key[0], lo1, hi0 ^ ctr[3] ^ key[1], lo0]
}

/// The full 10-round Philox-4x64 block function: maps `(counter, key)` to
/// four output words. The key is bumped by the Weyl constants between rounds
/// (9 bumps for 10 rounds), exactly as in Random123's `philox4x64_R(10, ...)`.
#[inline]
pub(crate) fn philox4x64_10(counter: [u64; 4], key: [u64; 2]) -> [u64; 4] {
    let mut ctr = round(counter, key);
    let mut key = key;
    for _ in 1..ROUNDS {
        key[0] = key[0].wrapping_add(PHILOX_W0);
        key[1] = key[1].wrapping_add(PHILOX_W1);
        ctr = round(ctr, key);
    }
    ctr
}

/// Increment a 256-bit little-endian counter by one, with carry (mod 2^256).
#[inline]
fn increment(counter: &mut [u64; 4]) {
    for word in counter.iter_mut() {
        *word = word.wrapping_add(1);
        if *word != 0 {
            return;
        }
    }
}

/// Add `delta` to a 256-bit little-endian counter (mod 2^256).
#[inline]
fn add_to_counter(counter: &mut [u64; 4], delta: u128) {
    let step = [delta as u64, (delta >> 64) as u64, 0u64, 0u64];
    let mut carry = false;
    for (word, s) in counter.iter_mut().zip(step) {
        let (sum, c1) = word.overflowing_add(s);
        let (sum, c2) = sum.overflowing_add(u64::from(carry));
        *word = sum;
        carry = c1 || c2;
    }
}

/// Philox-4x64-10 counter-based generator, bit-compatible with
/// `numpy.random.Philox` raw output.
///
/// State is a 256-bit counter, a 128-bit key, and a four-word output buffer.
/// Distinct keys give statistically independent streams; the counter gives
/// O(1) `advance` within a stream. The period per key is 4 * 2^256 outputs.
///
/// This type is `Clone + Send + Sync` (plain-old-data state), so it can be
/// moved into rayon tasks freely; a `&mut` is required to draw.
///
/// Reference: Salmon, Moraes, Dror and Shaw (2011), "Parallel Random Numbers:
/// As Easy as 1, 2, 3", Proc. SC'11.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Philox {
    key: [u64; 2],
    counter: [u64; 4],
    buffer: [u64; 4],
    /// Next unread index into `buffer`; `BUFFER_SIZE` means "empty, generate
    /// a new block on the next draw" (NumPy's `buffer_pos` convention).
    buffer_pos: usize,
}

impl Philox {
    /// Create an engine from raw little-endian key and counter words,
    /// equivalent to `numpy.random.Philox(key=k, counter=c)` where
    /// `k = key[0] + 2^64*key[1]` and `c = counter[0] + 2^64*counter[1] + ...`.
    ///
    /// The first draw generates the block for counter value `counter + 1`
    /// (NumPy increments before generating).
    pub fn from_key_counter(key: [u64; 2], counter: [u64; 4]) -> Self {
        Philox {
            key,
            counter,
            buffer: [0; 4],
            buffer_pos: BUFFER_SIZE,
        }
    }

    /// The 128-bit key as two little-endian `u64` words.
    pub fn key(&self) -> [u64; 2] {
        self.key
    }

    /// The 256-bit counter as four little-endian `u64` words. This is the
    /// index of the most recently generated block (0 if none yet).
    pub fn counter(&self) -> [u64; 4] {
        self.counter
    }

    /// Next raw 64-bit output, identical to `numpy.random.Philox.random_raw`.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        if self.buffer_pos < BUFFER_SIZE {
            let out = self.buffer[self.buffer_pos];
            self.buffer_pos += 1;
            return out;
        }
        increment(&mut self.counter);
        self.buffer = philox4x64_10(self.counter, self.key);
        self.buffer_pos = 1;
        self.buffer[0]
    }

    /// Advance the engine as if `delta` calls to [`Philox::next_u64`] had
    /// been made, in O(1) time (at most one block generation).
    ///
    /// Semantics: `delta` counts **64-bit draws**, not counter blocks (each
    /// counter block yields four draws). After `g.advance(k)`, `g` produces
    /// exactly the sequence a fresh copy would after `k` discarded draws.
    /// Counter arithmetic wraps modulo 2^256.
    ///
    /// Note this differs from `numpy.random.Philox.advance(delta)`, which
    /// adds `delta` to the raw block counter (i.e. skips `4*delta` draws)
    /// and discards any buffered output.
    pub fn advance(&mut self, delta: u128) {
        let remaining = (BUFFER_SIZE - self.buffer_pos) as u128;
        if delta <= remaining {
            // Consume within the already-generated block.
            self.buffer_pos += delta as usize;
            return;
        }
        let past_buffer = delta - remaining;
        let full_blocks = past_buffer / BUFFER_SIZE as u128;
        let into_next = (past_buffer % BUFFER_SIZE as u128) as usize;
        if into_next == 0 {
            // Lands exactly on a block boundary: the final block would have
            // been fully consumed, so it never needs materializing.
            add_to_counter(&mut self.counter, full_blocks);
            self.buffer_pos = BUFFER_SIZE;
        } else {
            // Lands inside a block: materialize it and mark the first
            // `into_next` outputs as consumed.
            add_to_counter(&mut self.counter, full_blocks + 1);
            self.buffer = philox4x64_10(self.counter, self.key);
            self.buffer_pos = into_next;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counter_carry_propagates() {
        let mut c = [u64::MAX, u64::MAX, 5, 0];
        increment(&mut c);
        assert_eq!(c, [0, 0, 6, 0]);
    }

    #[test]
    fn add_to_counter_carries_into_upper_words() {
        let mut c = [u64::MAX, u64::MAX, u64::MAX, 0];
        add_to_counter(&mut c, 1);
        assert_eq!(c, [0, 0, 0, 1]);
        let mut c = [5, 0, 0, 0];
        add_to_counter(&mut c, u128::MAX);
        // 5 + (2^128 - 1) = 4 + 2^128
        assert_eq!(c, [4, 0, 1, 0]);
    }

    #[test]
    fn advance_zero_is_identity() {
        let mut a = Philox::from_key_counter([1, 2], [0; 4]);
        let b = a.clone();
        a.advance(0);
        assert_eq!(a, b);
    }
}
