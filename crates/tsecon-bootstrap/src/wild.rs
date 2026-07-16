//! Wild-bootstrap weight generators.
//!
//! The wild bootstrap (Wu 1986; Liu 1988; Mammen 1993) perturbs residuals
//! multiplicatively: `e*_t = w_t * e_t` with iid weights `w_t` satisfying
//! `E[w] = 0` and `E[w^2] = 1`, preserving (conditional) heteroskedasticity
//! in the resample. This module owns the weight distributions; models apply
//! them.

use tsecon_rng::Stream;

/// A wild-bootstrap weight distribution. All variants have mean 0 and
/// variance 1; they differ in their third moment, which controls whether
/// the bootstrap distribution matches skewness.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum WildWeights {
    /// Rademacher weights: `+1` or `-1` with probability 1/2 each
    /// (Liu 1988; recommended by Davidson and Flachaire 2008).
    /// Moments: mean 0, variance 1, third moment 0 — imposes symmetry.
    Rademacher,
    /// Mammen's (1993) two-point distribution:
    /// `w = -(sqrt(5)-1)/2` with probability `(sqrt(5)+1)/(2 sqrt(5))`,
    /// `w =  (sqrt(5)+1)/2` with probability `(sqrt(5)-1)/(2 sqrt(5))`.
    /// Moments: mean 0, variance 1, third moment 1 — matches skewness to
    /// first order (the Edgeworth-correct choice).
    Mammen,
    /// Standard normal weights. Moments: mean 0, variance 1, third
    /// moment 0. Continuous support avoids the discrete atoms of the
    /// two-point laws.
    Normal,
}

impl WildWeights {
    /// Draw a single weight, consuming randomness from `stream`.
    ///
    /// Draw cost (part of the stability contract): `Rademacher` and
    /// `Mammen` consume one 64-bit draw; `Normal` consumes exactly two.
    #[inline]
    pub fn draw(self, stream: &mut Stream) -> f64 {
        match self {
            // uniform_f64 < 0.5 has probability exactly 1/2 (the 53-bit
            // uniform grid splits evenly).
            WildWeights::Rademacher => {
                if stream.uniform_f64() < 0.5 {
                    -1.0
                } else {
                    1.0
                }
            }
            WildWeights::Mammen => {
                let sqrt5 = 5.0_f64.sqrt();
                // P(w = low) = (sqrt(5)+1) / (2 sqrt(5)) ~= 0.7236.
                let p_low = (sqrt5 + 1.0) / (2.0 * sqrt5);
                if stream.uniform_f64() < p_low {
                    (1.0 - sqrt5) / 2.0
                } else {
                    (1.0 + sqrt5) / 2.0
                }
            }
            WildWeights::Normal => standard_normal(stream),
        }
    }

    /// Fill `out` with weights (equivalent to repeated [`WildWeights::draw`]).
    pub fn fill(self, stream: &mut Stream, out: &mut [f64]) {
        for slot in out {
            *slot = self.draw(stream);
        }
    }

    /// Draw `n` weights into a fresh vector (equivalent to repeated
    /// [`WildWeights::draw`]).
    pub fn sample(self, n: usize, stream: &mut Stream) -> Vec<f64> {
        (0..n).map(|_| self.draw(stream)).collect()
    }
}

/// One standard normal draw via the Box-Muller (1958) transform:
/// `z = sqrt(-2 ln u1) * cos(2 pi u2)` with `u1` mapped onto `(0, 1]`
/// so the logarithm is finite. Consumes exactly two 64-bit draws; the sine
/// partner variate is discarded to keep the per-draw stream cost fixed
/// (no hidden cache state).
//
// TODO(phase0): delegate to the shared innovation-distribution zoo
// (tsecon-stats) once its ziggurat sampler lands; Box-Muller is exact but
// roughly 2x slower and its tail stops at ~8.57 sigma (the 2^-53 grid).
#[inline]
fn standard_normal(stream: &mut Stream) -> f64 {
    // uniform_f64 is on [0, 1); 1 - u is on (0, 1], keeping ln() finite.
    let u1 = 1.0 - stream.uniform_f64();
    let u2 = stream.uniform_f64();
    (-2.0 * u1.ln()).sqrt() * (core::f64::consts::TAU * u2).cos()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn draw_costs_are_fixed() {
        // Rademacher and Mammen consume 1 u64; Normal consumes 2. Verify by
        // comparing against a twin stream advanced by hand.
        for (weights, cost) in [
            (WildWeights::Rademacher, 1u128),
            (WildWeights::Mammen, 1),
            (WildWeights::Normal, 2),
        ] {
            let mut a = Stream::new(11);
            let mut b = Stream::new(11);
            let _ = weights.draw(&mut a);
            b.advance(cost);
            assert_eq!(a.next_u64(), b.next_u64(), "{weights:?} cost {cost}");
        }
    }

    #[test]
    fn mammen_support_is_the_two_golden_ratio_points() {
        let sqrt5 = 5.0_f64.sqrt();
        let low = (1.0 - sqrt5) / 2.0;
        let high = (1.0 + sqrt5) / 2.0;
        let mut s = Stream::new(2);
        for _ in 0..1000 {
            let w = WildWeights::Mammen.draw(&mut s);
            assert!(w == low || w == high, "unexpected support point {w}");
        }
    }
}
