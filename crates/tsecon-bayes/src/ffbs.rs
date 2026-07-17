//! The Carter-Kohn forward-filter backward-sampling (FFBS) simulation
//! smoother: exact joint draws of the state path `alpha_{1..n}` from its
//! smoothing distribution `p(alpha | y)` in a linear-Gaussian state-space
//! model, built on the univariate Kalman filter of `tsecon-ssm`.
//!
//! # Algorithm
//!
//! Carter & Kohn (1994); Frühwirth-Schnatter (1994). A forward pass of the
//! univariate (sequential) filter yields the filtered moments `(a_{t|t},
//! P_{t|t})` and one-step predictions `(a_{t+1|t}, P_{t+1|t})`. The
//! backward pass then samples
//!
//! ```text
//! alpha_n ~ N(a_{n|n}, P_{n|n})
//! alpha_t | alpha_{t+1}, Y_t
//!   ~ N(a_{t|t} + G_t (alpha_{t+1} - a_{t+1|t}),  P_{t|t} - G_t T_t P_{t|t})
//! G_t = P_{t|t} T_t' P_{t+1|t}^+
//! ```
//!
//! which is the exact Gaussian conditional of `alpha_t` given the *full*
//! vector `alpha_{t+1}` (the joint `(alpha_t, alpha_{t+1}) | Y_t` has
//! cross-covariance `P_{t|t} T_t'`); by the Markov property the product of
//! these conditionals is the joint smoothing density, so each call to
//! [`FfbsSampler::draw`] is one independent, exact draw — no MCMC error.
//!
//! # Degenerate states (`R Q R'` singular)
//!
//! With a selection-form disturbance loading `R` (companion/lag states,
//! ARMA stacking), `P_{t+1|t} = T P_{t|t} T' + R Q R'` is singular and the
//! textbook gain `P_{t|t} T' P_{t+1|t}^{-1}` does not exist — the classic
//! crash-or-silently-wrong site (roadmap module 05, warning 11). This
//! implementation uses the Moore-Penrose pseudo-inverse `P_{t+1|t}^+`
//! through a symmetric eigendecomposition with a rank cutoff at
//! `1e-12 x max eigenvalue`: the backward conditional then concentrates
//! exactly on the reachable affine subspace, and randomness enters only
//! through the stochastic subspace — the general, rank-aware form of the
//! "sample only the stochastic rows" device of Kim & Nelson (1999, §8.2),
//! valid for any `R` and any (even singular) filtered covariance. The
//! conditional covariance's square root is likewise taken by
//! eigendecomposition with negative-eigenvalue clipping.
//!
//! # Diffuse initialization
//!
//! The forward filter supports exact-diffuse initialization, but the
//! stored moments during a diffuse period are only the finite parts, so
//! backward sampling *through* an uncollapsed diffuse period is not yet
//! supported: construction fails with [`BayesError::DiffuseNotCollapsed`]
//! unless the diffuse part has collapsed by the end of the first period
//! (`d_diffuse <= 1`, e.g. the local level, whose level prior becomes
//! proper after one observation). Gibbs blocks typically initialize
//! states with `Known` or `Stationary` priors, which are unaffected.
//! // TODO(phase0): diffuse-aware backward pass via the (P_inf, P_star)
//! // two-matrix recursions so multi-period diffuse models can be sampled.
//!
//! The filter's exposed predicted/filtered moments are sufficient for the
//! proper-prior and collapsed-diffuse cases; nothing beyond the public
//! `tsecon-ssm` API is required.

use tsecon_linalg::faer::{Mat, MatRef};
use tsecon_rng::Stream;
use tsecon_ssm::{filter_univariate, FilterOutput, LinearGaussianSSM};

use crate::dense::{frob_sq, mat_vec, std_normal, sym_eigen, symmetrize_in_place};
use crate::error::BayesError;

/// Relative eigenvalue cutoff for the rank decision in the predicted-
/// covariance pseudo-inverse.
const PINV_REL_TOL: f64 = 1e-12;

/// Numerical zero for "has the diffuse covariance collapsed" checks,
/// matching the filter's own diffuse tolerance.
const DIFFUSE_COLLAPSE_TOL: f64 = 1e-10;

/// A prepared FFBS sampler for one model and one observation matrix.
///
/// Construction runs the forward filter once and precomputes every
/// draw-independent backward quantity (gains and conditional-covariance
/// square roots), so each [`FfbsSampler::draw`] costs only `O(n m^2)`
/// flops plus `n m` standard-normal variates — the right shape for Gibbs
/// blocks that redraw states thousands of times per sweep.
#[derive(Debug, Clone)]
pub struct FfbsSampler {
    n: usize,
    m: usize,
    /// Filtered means `a_{t|t}`, `t = 0..n`.
    filtered_mean: Vec<Vec<f64>>,
    /// Predicted means `a_{t+1|t}`, indexed `0..=n`.
    predicted_mean: Vec<Vec<f64>>,
    /// Backward gains `G_t`, `t = 0..n-1`.
    gains: Vec<Mat<f64>>,
    /// Square roots of the backward conditional covariances, `t = 0..n-1`.
    cond_sqrts: Vec<Mat<f64>>,
    /// Square root of `P_{n-1|n-1}` for the terminal draw.
    last_sqrt: Mat<f64>,
    /// The forward filter pass (kept for likelihood access in Gibbs
    /// blocks).
    filter: FilterOutput,
}

impl FfbsSampler {
    /// Runs the forward univariate filter on `y` (`n x p`, NaN = missing)
    /// and precomputes the backward-sampling quantities.
    ///
    /// # Errors
    ///
    /// * everything [`filter_univariate`] can return (wrapped in
    ///   [`BayesError::Ssm`]);
    /// * [`BayesError::DiffuseNotCollapsed`] when an exact-diffuse
    ///   initialization has not collapsed after the first period (see the
    ///   module docs);
    /// * [`BayesError::NoConvergence`] if an eigendecomposition fails (not
    ///   observed for valid covariances).
    pub fn new(model: &LinearGaussianSSM, y: MatRef<'_, f64>) -> Result<Self, BayesError> {
        let fo = filter_univariate(model, y)?;
        let n = fo.filtered_state.len();
        let m = model.state_dim();

        if fo.d_diffuse > 1 {
            return Err(BayesError::DiffuseNotCollapsed {
                periods: fo.d_diffuse,
            });
        }
        // Defensive: the backward pass conditions on predicted covariances
        // at t = 1..n, which must carry no diffuse part.
        for t in 1..=n {
            if frob_sq(fo.predicted_diffuse_state_cov[t].as_ref()) > DIFFUSE_COLLAPSE_TOL {
                return Err(BayesError::DiffuseNotCollapsed {
                    periods: fo.d_diffuse,
                });
            }
        }

        let last_sqrt = sym_eigen(fo.filtered_state_cov[n - 1].as_ref())?.psd_sqrt();

        let mut gains = Vec::with_capacity(n.saturating_sub(1));
        let mut cond_sqrts = Vec::with_capacity(n.saturating_sub(1));
        for t in 0..n.saturating_sub(1) {
            let tr = model.t().at(t);
            let pf = fo.filtered_state_cov[t].as_ref();
            // Cross-covariance Cov(alpha_t, alpha_{t+1} | Y_t) = P_{t|t} T'.
            let cross = pf * tr.transpose();
            let pp = fo.predicted_state_cov[t + 1].as_ref();
            let pinv = sym_eigen(pp)?.pinv(PINV_REL_TOL);
            let g = cross.as_ref() * pinv.as_ref();
            // H_t = P_{t|t} - G_t (P_{t|t} T')'.
            let gct = g.as_ref() * cross.as_ref().transpose();
            let mut h = pf.to_owned();
            for j in 0..m {
                for i in 0..m {
                    h[(i, j)] -= gct[(i, j)];
                }
            }
            symmetrize_in_place(&mut h);
            cond_sqrts.push(sym_eigen(h.as_ref())?.psd_sqrt());
            gains.push(g);
        }

        Ok(Self {
            n,
            m,
            filtered_mean: fo.filtered_state.clone(),
            predicted_mean: fo.predicted_state.clone(),
            gains,
            cond_sqrts,
            last_sqrt,
            filter: fo,
        })
    }

    /// The forward filter output the sampler was built from (e.g. for the
    /// marginal log-likelihood inside a Gibbs sweep).
    pub fn filter(&self) -> &FilterOutput {
        &self.filter
    }

    /// Number of time periods `n`.
    pub fn n_periods(&self) -> usize {
        self.n
    }

    /// State dimension `m`.
    pub fn state_dim(&self) -> usize {
        self.m
    }

    /// One exact draw of the state path from `p(alpha_{1..n} | y)`,
    /// returned as an `n x m` matrix (row `t` is `alpha_t'`).
    ///
    /// Standard-normal variates are inverse-CDF transforms of `stream`
    /// uniforms, consumed in a fixed order (period `n-1` first, then
    /// backward; `m` variates per period) so results are bit-reproducible
    /// for a given stream state.
    ///
    /// # Errors
    ///
    /// [`BayesError::Stats`] / [`BayesError::NoConvergence`] on quantile
    /// failures (not observed in practice).
    pub fn draw(&self, stream: &mut Stream) -> Result<Mat<f64>, BayesError> {
        let (n, m) = (self.n, self.m);
        let mut path = Mat::<f64>::zeros(n, m);
        let mut z = vec![0.0; m];

        // Terminal draw: alpha_{n-1} ~ N(a_{n-1|n-1}, P_{n-1|n-1}).
        for slot in z.iter_mut() {
            *slot = std_normal(stream)?;
        }
        let noise = mat_vec(self.last_sqrt.as_ref(), &z);
        let mut alpha_next: Vec<f64> = self.filtered_mean[n - 1]
            .iter()
            .zip(&noise)
            .map(|(a, e)| a + e)
            .collect();
        for (j, &v) in alpha_next.iter().enumerate() {
            path[(n - 1, j)] = v;
        }

        // Backward pass: alpha_t | alpha_{t+1}, Y_t.
        for t in (0..n.saturating_sub(1)).rev() {
            let resid: Vec<f64> = alpha_next
                .iter()
                .zip(&self.predicted_mean[t + 1])
                .map(|(a, p)| a - p)
                .collect();
            let shift = mat_vec(self.gains[t].as_ref(), &resid);
            for slot in z.iter_mut() {
                *slot = std_normal(stream)?;
            }
            let noise = mat_vec(self.cond_sqrts[t].as_ref(), &z);
            let alpha_t: Vec<f64> = (0..m)
                .map(|j| self.filtered_mean[t][j] + shift[j] + noise[j])
                .collect();
            for (j, &v) in alpha_t.iter().enumerate() {
                path[(t, j)] = v;
            }
            alpha_next = alpha_t;
        }
        Ok(path)
    }
}
