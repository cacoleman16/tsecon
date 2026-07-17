//! Model specification for a Markov-switching autoregression.

use crate::error::RegimeError;

/// The upper bound on the expanded state space `k^(order + 1)` that the
/// Hamilton filter materializes. The filter tracks the joint distribution
/// of the last `order + 1` regimes (Hamilton 1989, §2); this cap keeps the
/// per-step work and memory bounded and rejects specifications that would
/// otherwise allocate astronomically.
const MAX_EXPANDED_STATES: usize = 4096;

/// Specification of a `k`-regime Markov-switching autoregression of order
/// `p` (Hamilton 1989).
///
/// The observed series follows, conditional on the latent regime path,
///
/// ```text
/// y_t - mu_{S_t} = sum_{l=1}^{p} phi_l (y_{t-l} - mu_{S_{t-l}}) + e_t,
/// e_t ~ N(0, sigma^2_{S_t}),
/// ```
///
/// where `S_t in {0, ..., k-1}` is a first-order Markov chain with
/// column-stochastic transition matrix `P` (`P[i][j] = P(S_t = i | S_{t-1}
/// = j)`). The per-regime means `mu` always switch. The AR coefficients
/// `phi` switch iff [`switching_ar`](Self::switching_ar); the innovation
/// variance switches iff [`switching_variance`](Self::switching_variance).
///
/// This is the parameterization of `statsmodels`
/// `MarkovAutoregression(k_regimes, order, switching_ar, switching_variance)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MsarSpec {
    /// Number of regimes `k` (at least two).
    pub k_regimes: usize,
    /// Autoregressive order `p` (zero gives a Markov-switching Gaussian
    /// mixture; the filter and smoother support it, but [`fit`] requires
    /// `order >= 1`).
    ///
    /// [`fit`]: crate::MarkovSwitchingAr::fit
    pub order: usize,
    /// Whether the AR coefficients differ across regimes.
    pub switching_ar: bool,
    /// Whether the innovation variance differs across regimes.
    pub switching_variance: bool,
}

impl MsarSpec {
    /// Validates the specification and returns the size `k^(order + 1)` of
    /// the expanded regime state space.
    ///
    /// Errors with [`RegimeError::InvalidSpec`] if there are fewer than two
    /// regimes or if the expanded state space would exceed the internal
    /// cap of 4096 states.
    pub fn expanded_states(&self) -> Result<usize, RegimeError> {
        if self.k_regimes < 2 {
            return Err(RegimeError::InvalidSpec {
                what: "a Markov-switching model needs at least two regimes",
            });
        }
        let mut m: usize = 1;
        for _ in 0..=self.order {
            m = m
                .checked_mul(self.k_regimes)
                .filter(|&m| m <= MAX_EXPANDED_STATES)
                .ok_or(RegimeError::InvalidSpec {
                    what: "expanded state space k^(order+1) exceeds the internal cap of 4096",
                })?;
        }
        Ok(m)
    }

    /// The number of AR-coefficient blocks: `k` when the AR is switching,
    /// otherwise one shared block.
    pub(crate) fn ar_blocks(&self) -> usize {
        if self.switching_ar {
            self.k_regimes
        } else {
            1
        }
    }

    /// The number of variance parameters: `k` when the variance switches,
    /// otherwise one.
    pub(crate) fn variance_blocks(&self) -> usize {
        if self.switching_variance {
            self.k_regimes
        } else {
            1
        }
    }
}
