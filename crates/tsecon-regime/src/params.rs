//! Parameter container for a Markov-switching autoregression.

use crate::error::RegimeError;
use crate::spec::MsarSpec;

/// Tolerance on the column sums of the transition matrix.
const STOCHASTIC_TOL: f64 = 1e-8;

/// The parameters of a [`MsarSpec`] model: the regime transition matrix,
/// per-regime means, AR coefficients (shared or switching), and innovation
/// variances (shared or switching).
///
/// The transition matrix is stored column-stochastic in the `statsmodels`
/// convention `P[i][j] = P(S_t = i | S_{t-1} = j)`, so each **column** is a
/// probability distribution over the destination regime and sums to one.
/// The stationary (ergodic) regime distribution `pi` solves `P pi = pi`.
#[derive(Debug, Clone, PartialEq)]
pub struct MsarParams {
    k: usize,
    order: usize,
    switching_ar: bool,
    switching_variance: bool,
    /// Row-major `k`-by-`k`: `transition[i * k + j] = P(S_t = i | S_{t-1} = j)`.
    transition: Vec<f64>,
    /// Per-regime means `mu`, length `k`.
    means: Vec<f64>,
    /// AR coefficients, length `order` (shared) or `k * order` (switching),
    /// row-major by regime.
    ar: Vec<f64>,
    /// Innovation variances, length `1` (shared) or `k` (switching).
    variances: Vec<f64>,
}

impl MsarParams {
    /// Assembles a parameter set from its components.
    ///
    /// * `transition` is `k` rows of `k` entries with `transition[i][j] =
    ///   P(S_t = i | S_{t-1} = j)`; each column must sum to one (within
    ///   `1e-8`) with non-negative entries.
    /// * `means` has length `k`.
    /// * `ar` has outer length `1` (AR coefficients shared across regimes)
    ///   or `k` (switching); every inner block has the same length `order`.
    /// * `variances` has length `1` (shared) or `k` (switching); every
    ///   entry must be strictly positive.
    ///
    /// The AR order, regime count, and switching flags are inferred from
    /// the shapes and can be checked against a [`MsarSpec`] with
    /// [`matches_spec`](Self::matches_spec).
    pub fn new(
        transition: Vec<Vec<f64>>,
        means: Vec<f64>,
        ar: Vec<Vec<f64>>,
        variances: Vec<f64>,
    ) -> Result<Self, RegimeError> {
        let k = means.len();
        if k < 2 {
            return Err(RegimeError::InvalidSpec {
                what: "a Markov-switching model needs at least two regimes",
            });
        }
        for (i, &m) in means.iter().enumerate() {
            if !m.is_finite() {
                return Err(RegimeError::InvalidParameter {
                    name: "means",
                    value: m,
                    requirement: "every regime mean must be finite",
                });
            }
            let _ = i;
        }

        // Transition matrix: shape and column-stochasticity.
        if transition.len() != k {
            return Err(RegimeError::DimensionMismatch {
                what: "transition matrix rows",
                expected: k,
                actual: transition.len(),
            });
        }
        let mut flat_p = vec![0.0; k * k];
        for (i, row) in transition.iter().enumerate() {
            if row.len() != k {
                return Err(RegimeError::DimensionMismatch {
                    what: "transition matrix row length",
                    expected: k,
                    actual: row.len(),
                });
            }
            for (j, &p) in row.iter().enumerate() {
                if !(p.is_finite() && (0.0..=1.0 + STOCHASTIC_TOL).contains(&p)) {
                    return Err(RegimeError::InvalidParameter {
                        name: "transition",
                        value: p,
                        requirement: "each transition probability must lie in [0, 1]",
                    });
                }
                flat_p[i * k + j] = p;
            }
        }
        for j in 0..k {
            let col_sum: f64 = (0..k).map(|i| flat_p[i * k + j]).sum();
            if (col_sum - 1.0).abs() > STOCHASTIC_TOL {
                return Err(RegimeError::NotStochastic {
                    column: j,
                    sum: col_sum,
                });
            }
        }

        // AR coefficients.
        if ar.is_empty() {
            return Err(RegimeError::InvalidSpec {
                what: "ar must have one block (shared) or k blocks (switching)",
            });
        }
        let switching_ar = ar.len() != 1;
        if switching_ar && ar.len() != k {
            return Err(RegimeError::DimensionMismatch {
                what: "ar outer length (must be 1 or k)",
                expected: k,
                actual: ar.len(),
            });
        }
        let order = ar[0].len();
        let mut flat_ar = Vec::with_capacity(ar.len() * order);
        for block in &ar {
            if block.len() != order {
                return Err(RegimeError::DimensionMismatch {
                    what: "ar block length",
                    expected: order,
                    actual: block.len(),
                });
            }
            for &c in block {
                if !c.is_finite() {
                    return Err(RegimeError::InvalidParameter {
                        name: "ar",
                        value: c,
                        requirement: "every AR coefficient must be finite",
                    });
                }
                flat_ar.push(c);
            }
        }

        // Variances.
        let switching_variance = variances.len() != 1;
        if switching_variance && variances.len() != k {
            return Err(RegimeError::DimensionMismatch {
                what: "variances length (must be 1 or k)",
                expected: k,
                actual: variances.len(),
            });
        }
        for &v in &variances {
            if !(v.is_finite() && v > 0.0) {
                return Err(RegimeError::InvalidParameter {
                    name: "variances",
                    value: v,
                    requirement: "every innovation variance must be strictly positive",
                });
            }
        }

        Ok(Self {
            k,
            order,
            switching_ar,
            switching_variance,
            transition: flat_p,
            means,
            ar: flat_ar,
            variances,
        })
    }

    /// The number of regimes `k`.
    pub fn k_regimes(&self) -> usize {
        self.k
    }

    /// The autoregressive order `p`.
    pub fn order(&self) -> usize {
        self.order
    }

    /// `P(S_t = i | S_{t-1} = j)`.
    #[inline]
    pub(crate) fn transition(&self, i: usize, j: usize) -> f64 {
        self.transition[i * self.k + j]
    }

    /// The transition matrix as `k` rows of `k` entries, `P[i][j] = P(S_t =
    /// i | S_{t-1} = j)`.
    pub fn transition_matrix(&self) -> Vec<Vec<f64>> {
        (0..self.k)
            .map(|i| (0..self.k).map(|j| self.transition(i, j)).collect())
            .collect()
    }

    /// The per-regime means `mu`.
    pub fn means(&self) -> &[f64] {
        &self.means
    }

    /// The AR coefficients for regime `regime` (the shared block when the
    /// AR is not switching).
    #[inline]
    pub(crate) fn ar_coefs(&self, regime: usize) -> &[f64] {
        let block = if self.switching_ar { regime } else { 0 };
        &self.ar[block * self.order..(block + 1) * self.order]
    }

    /// The innovation variance of regime `regime` (the shared variance when
    /// it is not switching).
    #[inline]
    pub(crate) fn variance(&self, regime: usize) -> f64 {
        if self.switching_variance {
            self.variances[regime]
        } else {
            self.variances[0]
        }
    }

    /// The innovation variances, length `1` (shared) or `k` (switching).
    pub fn variances(&self) -> &[f64] {
        &self.variances
    }

    /// The expected regime durations `1 / (1 - p_ii)`, length `k`.
    ///
    /// For a first-order Markov chain the sojourn time in regime `i` is
    /// geometric with success probability `1 - p_ii`, so its mean is `1 /
    /// (1 - p_ii)` (Hamilton 1994, §22.2). An absorbing regime (`p_ii = 1`)
    /// yields `+inf`.
    pub fn expected_durations(&self) -> Vec<f64> {
        (0..self.k)
            .map(|i| 1.0 / (1.0 - self.transition(i, i)))
            .collect()
    }

    /// Whether the parameter shapes match `spec`.
    pub fn matches_spec(&self, spec: &MsarSpec) -> bool {
        self.k == spec.k_regimes
            && self.order == spec.order
            && self.switching_ar == spec.switching_ar
            && self.switching_variance == spec.switching_variance
    }
}
