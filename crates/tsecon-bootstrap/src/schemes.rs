//! Index-based resampling schemes.
//!
//! Every scheme emits a vector of *indices* into the original sample —
//! resampling never copies data. Downstream code gathers `x[idx[t]]` (or a
//! whole row of a data matrix) itself, so one engine serves every module
//! regardless of the shape of the data being resampled.

use tsecon_rng::Stream;

use crate::error::BootstrapError;

/// A bootstrap resampling scheme, describing how indices into a length-`n`
/// sample are drawn.
///
/// All variants produce exactly `n` indices in `0..n` via
/// [`indices`]. Block schemes concatenate blocks and truncate the final
/// block so the resample always has the original length.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum BlockScheme {
    /// The iid (Efron 1979) bootstrap: each index drawn independently and
    /// uniformly from `0..n`. Appropriate for independent data only.
    Iid,
    /// The moving-block bootstrap of Künsch (1989): blocks of fixed length
    /// `block_length` starting at positions drawn uniformly from
    /// `0..=n - block_length`. Blocks never wrap past the end of the sample,
    /// so observations near the ends appear in fewer blocks.
    MovingBlock {
        /// Fixed block length `l`, `1 <= l <= n`.
        block_length: usize,
    },
    /// The circular-block bootstrap of Politis and Romano (1992): blocks of
    /// fixed length `block_length` starting at positions drawn uniformly
    /// from `0..n`, wrapping around the end of the sample (indices taken
    /// modulo `n`). The wrap-around removes the moving-block edge-effect
    /// bias in the resampled mean.
    CircularBlock {
        /// Fixed block length `l`, `1 <= l <= n`.
        block_length: usize,
    },
    /// The stationary bootstrap of Politis and Romano (1994): block lengths
    /// are geometric with restart probability `p` — the chain starts at a
    /// uniform index, and at each step either restarts at a fresh uniform
    /// index (with probability `p`) or continues to the next observation,
    /// wrapping around the end of the sample. The expected block length is
    /// `1/p`, and the resampled series is stationary conditional on the
    /// data (the property the scheme is named for).
    Stationary {
        /// Per-step restart probability, `0 < p <= 1`; expected block
        /// length `1/p`.
        p: f64,
    },
}

/// Draw one full set of bootstrap indices for a length-`n` sample under
/// `scheme`, consuming randomness from `stream`.
///
/// Returns exactly `n` indices, each in `0..n`. The mapping from the
/// stream's raw draws to indices is fixed and documented per scheme (see
/// [`BlockScheme`]), so a given `(scheme, n, stream state)` always yields
/// the same indices — this is the reproducibility contract every bootstrap
/// in the library inherits.
///
/// Draw order (part of the stability contract):
/// - `Iid`: `n` bounded uniform draws.
/// - `MovingBlock` / `CircularBlock`: one bounded uniform draw per block,
///   in order.
/// - `Stationary`: one bounded uniform draw for the first index, then per
///   step one `[0,1)` uniform (the restart coin) plus, on restart, one
///   bounded uniform draw.
///
/// Bounded uniform draws use bitmask rejection sampling on raw 64-bit
/// output, so they are exactly uniform (no modulo or floating-point bias).
///
/// # Errors
///
/// - [`BootstrapError::EmptySample`] if `n == 0`.
/// - [`BootstrapError::InvalidBlockLength`] if a block length is outside
///   `1..=n`.
/// - [`BootstrapError::InvalidProbability`] if the stationary restart
///   probability is not in `(0, 1]`.
///
/// # References
///
/// Efron (1979); Künsch (1989); Politis and Romano (1992, 1994).
pub fn indices(
    scheme: BlockScheme,
    n: usize,
    stream: &mut Stream,
) -> Result<Vec<usize>, BootstrapError> {
    if n == 0 {
        return Err(BootstrapError::EmptySample);
    }
    match scheme {
        BlockScheme::Iid => Ok((0..n).map(|_| uniform_index(stream, n)).collect()),
        BlockScheme::MovingBlock { block_length } => {
            validate_block_length(block_length, n)?;
            let n_starts = n - block_length + 1;
            let mut out = Vec::with_capacity(n);
            while out.len() < n {
                let start = uniform_index(stream, n_starts);
                let take = block_length.min(n - out.len());
                out.extend(start..start + take);
            }
            Ok(out)
        }
        BlockScheme::CircularBlock { block_length } => {
            validate_block_length(block_length, n)?;
            let mut out = Vec::with_capacity(n);
            while out.len() < n {
                let start = uniform_index(stream, n);
                let take = block_length.min(n - out.len());
                out.extend((0..take).map(|j| (start + j) % n));
            }
            Ok(out)
        }
        BlockScheme::Stationary { p } => {
            if !(p > 0.0 && p <= 1.0) {
                return Err(BootstrapError::InvalidProbability { p });
            }
            let mut out = Vec::with_capacity(n);
            let mut idx = uniform_index(stream, n);
            out.push(idx);
            for _ in 1..n {
                // uniform_f64 < p has probability exactly p when p is a
                // multiple of 2^-53, and within 2^-53 of p otherwise.
                if stream.uniform_f64() < p {
                    idx = uniform_index(stream, n);
                } else {
                    idx = (idx + 1) % n;
                }
                out.push(idx);
            }
            Ok(out)
        }
    }
}

fn validate_block_length(block_length: usize, n: usize) -> Result<(), BootstrapError> {
    if block_length == 0 || block_length > n {
        return Err(BootstrapError::InvalidBlockLength { block_length, n });
    }
    Ok(())
}

/// Exactly uniform draw from `0..n` via bitmask rejection sampling
/// (Lemire 2019 discusses the family; the mask variant needs no
/// multiplication and is branch-cheap): mask raw 64-bit output down to the
/// smallest power-of-two range covering `n`, reject values `>= n`.
/// Expected draws per index < 2. `n == 1` consumes no randomness.
#[inline]
pub(crate) fn uniform_index(stream: &mut Stream, n: usize) -> usize {
    debug_assert!(n > 0);
    if n == 1 {
        return 0;
    }
    let bound = n as u64;
    let mask = u64::MAX >> (bound - 1).leading_zeros();
    loop {
        let v = stream.next_u64() & mask;
        if v < bound {
            // Cast is lossless: v < bound = n as u64, and n fits in usize.
            return v as usize;
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn uniform_index_covers_full_range_and_only_it() {
        let mut stream = Stream::new(7);
        let n = 10;
        let mut seen = [false; 10];
        for _ in 0..1000 {
            let i = uniform_index(&mut stream, n);
            assert!(i < n);
            seen[i] = true;
        }
        assert!(seen.iter().all(|&s| s), "all of 0..10 should appear");
    }

    #[test]
    fn uniform_index_of_one_consumes_no_randomness() {
        let mut a = Stream::new(3);
        let mut b = Stream::new(3);
        assert_eq!(uniform_index(&mut a, 1), 0);
        assert_eq!(a.next_u64(), b.next_u64());
    }

    #[test]
    fn parameter_validation() {
        let mut s = Stream::new(0);
        assert_eq!(
            indices(BlockScheme::Iid, 0, &mut s),
            Err(BootstrapError::EmptySample)
        );
        assert_eq!(
            indices(BlockScheme::MovingBlock { block_length: 0 }, 5, &mut s),
            Err(BootstrapError::InvalidBlockLength {
                block_length: 0,
                n: 5
            })
        );
        assert_eq!(
            indices(BlockScheme::CircularBlock { block_length: 6 }, 5, &mut s),
            Err(BootstrapError::InvalidBlockLength {
                block_length: 6,
                n: 5
            })
        );
        assert_eq!(
            indices(BlockScheme::Stationary { p: 0.0 }, 5, &mut s),
            Err(BootstrapError::InvalidProbability { p: 0.0 })
        );
        assert_eq!(
            indices(BlockScheme::Stationary { p: 1.5 }, 5, &mut s),
            Err(BootstrapError::InvalidProbability { p: 1.5 })
        );
        assert!(matches!(
            indices(BlockScheme::Stationary { p: f64::NAN }, 5, &mut s),
            Err(BootstrapError::InvalidProbability { .. })
        ));
    }

    #[test]
    fn moving_block_with_full_length_is_identity() {
        let mut s = Stream::new(1);
        let out = indices(BlockScheme::MovingBlock { block_length: 8 }, 8, &mut s).unwrap();
        assert_eq!(out, (0..8).collect::<Vec<_>>());
    }
}
