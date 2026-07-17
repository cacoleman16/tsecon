//! Impulse responses and simulation of the solved system.
//!
//! Both are driven by the solved decision rule and law of motion
//!
//! ```text
//! jump_t              = G . predetermined_t
//! predetermined_{t+1} = P . predetermined_t + Q . z_{t+1}
//! ```

use tsecon_linalg::faer::Mat;

use crate::error::DsgeError;
use crate::solve::DsgeSolution;

/// A path of the solved system: the predetermined and jump variables at each
/// date. `predetermined[t]` has length `n_predetermined`; `jump[t]` has length
/// `n_jump`. Both have the same number of rows (dates).
#[derive(Debug, Clone, PartialEq)]
pub struct Trajectory {
    /// The predetermined-variable path, one row per date.
    pub predetermined: Vec<Vec<f64>>,
    /// The jump-variable path, one row per date.
    pub jump: Vec<Vec<f64>>,
}

impl DsgeSolution {
    /// Impulse response to a one-time unit innovation in shock `shock`, over
    /// `horizon + 1` dates `t = 0, 1, ..., horizon`.
    ///
    /// The economy starts at rest; at `t = 0` a unit innovation `z = e_shock`
    /// hits, so `predetermined_0 = Q e_shock` and `jump_0 = G predetermined_0`.
    /// Thereafter `z = 0`, so `predetermined_{t+1} = P predetermined_t`. Because
    /// `P` is stable the response decays back to zero.
    ///
    /// # Errors
    ///
    /// * [`DsgeError::Simulation`] if `shock >= n_shocks`.
    pub fn impulse_response(&self, shock: usize, horizon: usize) -> Result<Trajectory, DsgeError> {
        if shock >= self.n_shocks() {
            return Err(DsgeError::Simulation {
                what: "shock index out of range (must be < n_shocks)",
            });
        }
        let n_pre = self.n_predetermined();
        // predetermined_0 = Q e_shock  (the `shock`-th column of Q).
        let mut k = vec![0.0f64; n_pre];
        for (i, ki) in k.iter_mut().enumerate() {
            *ki = self.q()[(i, shock)];
        }
        let mut predetermined = Vec::with_capacity(horizon + 1);
        let mut jump = Vec::with_capacity(horizon + 1);
        for _ in 0..=horizon {
            jump.push(mat_vec(self.g(), &k));
            predetermined.push(k.clone());
            k = mat_vec(self.p(), &k);
        }
        Ok(Trajectory {
            predetermined,
            jump,
        })
    }

    /// Simulates the solved system from an initial predetermined state `k0`
    /// under an explicit shock sequence.
    ///
    /// `k0` has length `n_predetermined`. `shocks[t]` (length `n_shocks`) is the
    /// innovation `z_{t+1}` applied between date `t` and `t + 1`. The returned
    /// trajectory has `shocks.len() + 1` dates: `predetermined[0] = k0`,
    /// `jump_t = G predetermined_t`, and
    /// `predetermined_{t+1} = P predetermined_t + Q shocks[t]`.
    ///
    /// # Errors
    ///
    /// * [`DsgeError::Simulation`] if `k0` or any shock vector has the wrong
    ///   length, or contains a non-finite entry.
    pub fn simulate(&self, k0: &[f64], shocks: &[Vec<f64>]) -> Result<Trajectory, DsgeError> {
        let n_pre = self.n_predetermined();
        let n_shocks = self.n_shocks();
        if k0.len() != n_pre {
            return Err(DsgeError::Simulation {
                what: "initial state k0 must have length n_predetermined",
            });
        }
        if k0.iter().any(|v| !v.is_finite()) {
            return Err(DsgeError::Simulation {
                what: "initial state k0 contains a non-finite value",
            });
        }
        let mut k = k0.to_vec();
        let mut predetermined = Vec::with_capacity(shocks.len() + 1);
        let mut jump = Vec::with_capacity(shocks.len() + 1);
        for z in shocks {
            if z.len() != n_shocks {
                return Err(DsgeError::Simulation {
                    what: "each shock vector must have length n_shocks",
                });
            }
            if z.iter().any(|v| !v.is_finite()) {
                return Err(DsgeError::Simulation {
                    what: "a shock vector contains a non-finite value",
                });
            }
            jump.push(mat_vec(self.g(), &k));
            predetermined.push(k.clone());
            // k_{t+1} = P k_t + Q z.
            let mut next = mat_vec(self.p(), &k);
            for (i, ni) in next.iter_mut().enumerate() {
                for (j, &zj) in z.iter().enumerate() {
                    *ni += self.q()[(i, j)] * zj;
                }
            }
            k = next;
        }
        // Final date (after the last shock).
        jump.push(mat_vec(self.g(), &k));
        predetermined.push(k);
        Ok(Trajectory {
            predetermined,
            jump,
        })
    }
}

/// Dense matrix-times-vector, `m (rows x cols) * v (cols)`.
fn mat_vec(m: &Mat<f64>, v: &[f64]) -> Vec<f64> {
    let rows = m.nrows();
    let cols = m.ncols();
    let mut out = vec![0.0f64; rows];
    for (i, oi) in out.iter_mut().enumerate() {
        let mut s = 0.0;
        for (j, &vj) in v.iter().enumerate().take(cols) {
            s += m[(i, j)] * vj;
        }
        *oi = s;
    }
    out
}
