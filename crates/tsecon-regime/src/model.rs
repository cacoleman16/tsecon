//! The Markov-switching autoregression: Hamilton filter, Kim smoother, and
//! EM estimation.

use tsecon_stats::dist::{ContinuousDist, StdNormal};

use crate::error::RegimeError;
use crate::linsolve::solve;
use crate::params::MsarParams;
use crate::results::{FilterResult, FitResult, SmoothResult};
use crate::spec::MsarSpec;

/// Ridge added to the diagonal of an M-step normal-equation matrix, as a
/// fraction of that matrix's own largest diagonal entry.
///
/// The ridge exists only to keep the system solvable when a regime is
/// unoccupied under the smoother, so it must be negligible relative to the
/// information the data actually carry — and "negligible" is a statement
/// about a ratio, not about an absolute size. The mean normal equations are
/// weighted by `1 / sigma^2` and so carry the *inverse square* of the data's
/// scale: measure the same series in persons rather than in thousands of
/// persons and every entry shrinks by `10^-6`, which would inflate a fixed
/// absolute ridge by the same factor and shrink the estimated means toward
/// zero. At unit scale this relative ridge reproduces the magnitude of the
/// absolute `1e-10` it replaced.
const M_STEP_RIDGE_REL: f64 = 1e-12;

/// Floor on an estimated innovation variance, as a fraction of the sample
/// variance of the data.
///
/// The floor keeps `sigma^2` strictly positive when a regime fits its
/// observations exactly. An innovation variance is in units of `y` squared,
/// so the floor has to be too: an absolute floor silently becomes binding
/// once the series is measured in units that make its variance small, and
/// the estimate then stops scaling with the data.
const VARIANCE_FLOOR_REL: f64 = 1e-12;

/// The largest diagonal entry of the row-major `n`-by-`n` matrix `mat`.
///
/// The M-step normal-equation matrices are weighted sums of outer products
/// and hence positive semidefinite, so their largest diagonal entry bounds
/// their spectral norm within a factor of `n` and is the natural scale to
/// measure a ridge against.
fn max_diagonal(mat: &[f64], n: usize) -> f64 {
    (0..n).map(|i| mat[i * n + i]).fold(0.0, f64::max)
}

/// Whether `x` can serve as the scale a tolerance is measured against:
/// strictly positive and finite (`NaN` is not).
#[inline]
fn is_usable_scale(x: f64) -> bool {
    x > 0.0 && x.is_finite()
}

/// Result of running the forward (Hamilton) recursion, retaining the
/// internal expanded-state quantities the smoother and EM need.
struct FilterState {
    /// Number of usable observations `n = T - order`.
    n: usize,
    /// Prediction-error-decomposition log-likelihood.
    loglik: f64,
    /// `n`-by-`k` filtered marginal regime probabilities `P(S_t | Y_t)`.
    filtered_marginal: Vec<Vec<f64>>,
    /// `n`-by-`m` filtered joint probabilities over the expanded state
    /// `(S_t, ..., S_{t-order})`, `P(z_t | Y_t)`.
    filtered_joint: Vec<Vec<f64>>,
    /// `n`-by-`m` one-step predicted joint probabilities `P(z_t | Y_{t-1})`.
    predicted_joint: Vec<Vec<f64>>,
}

/// A `k`-regime Markov-switching autoregression bound to a data series
/// (Hamilton 1989).
///
/// The estimator borrows the data and the [`MsarSpec`]; parameters are
/// supplied per call so a single model can score many parameter vectors
/// (as the EM iterations and any external optimizer do).
///
/// # Example
///
/// ```
/// use tsecon_regime::{MarkovSwitchingAr, MsarParams, MsarSpec};
///
/// let y = [ -1.0, -0.4, 0.2, 1.3, 1.1, 0.6, -0.7, -1.2, 0.1, 0.9 ];
/// let spec = MsarSpec { k_regimes: 2, order: 1,
///                       switching_ar: false, switching_variance: true };
/// // P[i][j] = P(S_t = i | S_{t-1} = j); columns sum to 1.
/// let params = MsarParams::new(
///     vec![vec![0.9, 0.2], vec![0.1, 0.8]],
///     vec![-1.0, 1.0],
///     vec![vec![0.5]],
///     vec![0.5, 1.0],
/// ).unwrap();
/// let model = MarkovSwitchingAr::new(&y, spec).unwrap();
/// let out = model.filter(&params).unwrap();
/// assert!(out.loglik.is_finite());
/// assert_eq!(out.filtered_prob.len(), y.len() - 1);
/// ```
#[derive(Debug, Clone)]
pub struct MarkovSwitchingAr<'a> {
    y: &'a [f64],
    spec: MsarSpec,
    /// Cached expanded-state count `k^(order + 1)`.
    m: usize,
    /// Cached powers of `k`: `pow_k[l] = k^l`, `l = 0..=order + 1`.
    pow_k: Vec<usize>,
    /// Cached sample variance of `y`, in units of `y` squared. This is the
    /// scale the estimated innovation variances live on, and the yardstick
    /// the EM variance floor is measured against.
    y_var: f64,
}

impl<'a> MarkovSwitchingAr<'a> {
    /// Binds the model to data `y` under `spec`.
    ///
    /// Errors if the specification is invalid (see
    /// [`MsarSpec::expanded_states`]), if any observation is non-finite, or
    /// if there are not more than `order` observations.
    pub fn new(y: &'a [f64], spec: MsarSpec) -> Result<Self, RegimeError> {
        let m = spec.expanded_states()?;
        if y.len() <= spec.order {
            return Err(RegimeError::InsufficientData {
                needed: spec.order + 1,
                got: y.len(),
            });
        }
        for &v in y {
            if !v.is_finite() {
                return Err(RegimeError::NonFinite { what: "data y" });
            }
        }
        let k = spec.k_regimes;
        let mut pow_k = vec![1usize; spec.order + 2];
        for l in 1..pow_k.len() {
            pow_k[l] = pow_k[l - 1] * k;
        }
        let n = y.len() as f64;
        let mean_y = y.iter().sum::<f64>() / n;
        let y_var = y.iter().map(|v| (v - mean_y) * (v - mean_y)).sum::<f64>() / n;
        Ok(Self {
            y,
            spec,
            m,
            pow_k,
            y_var,
        })
    }

    /// The `l`-th regime digit of expanded state `a`: `S_{t-l}` when `a`
    /// encodes `(S_t, S_{t-1}, ..., S_{t-order})`.
    #[inline]
    fn digit(&self, a: usize, l: usize) -> usize {
        (a / self.pow_k[l]) % self.spec.k_regimes
    }

    fn check_params(&self, params: &MsarParams) -> Result<(), RegimeError> {
        if !params.matches_spec(&self.spec) {
            return Err(RegimeError::InvalidParameter {
                name: "params",
                value: f64::NAN,
                requirement: "parameter shapes must match the model specification",
            });
        }
        Ok(())
    }

    /// The stationary (ergodic) regime distribution `pi` solving `P pi =
    /// pi`, `sum pi = 1` (Hamilton 1994, eq. 22.2.26). Used as the
    /// `statsmodels` "steady-state" filter initialization.
    fn stationary_distribution(&self, params: &MsarParams) -> Result<Vec<f64>, RegimeError> {
        let k = self.spec.k_regimes;
        // Solve (I - P) pi = 0 with the last equation replaced by the
        // normalization sum(pi) = 1, i.e. A pi = e_k.
        let mut a = vec![0.0; k * k];
        for i in 0..k {
            for j in 0..k {
                a[i * k + j] = if i == j { 1.0 } else { 0.0 } - params.transition(i, j);
            }
        }
        for j in 0..k {
            a[(k - 1) * k + j] = 1.0;
        }
        let mut b = vec![0.0; k];
        b[k - 1] = 1.0;
        let pi = solve(a, b, k, "stationary distribution")?;
        for &p in &pi {
            if !(p.is_finite() && p >= -1e-9) {
                return Err(RegimeError::NonFinite {
                    what: "stationary distribution",
                });
            }
        }
        Ok(pi)
    }

    /// The conditional density `N(y_t; mean_pred, sigma^2_{S_t})` for
    /// expanded state `a` at absolute time `t`, following the Hamilton mean
    /// convention `y_t - mu_{S_t} = sum_l phi_l (y_{t-l} - mu_{S_{t-l}}) + e`.
    #[inline]
    fn state_density(&self, params: &MsarParams, a: usize, t: usize) -> f64 {
        let s0 = a % self.spec.k_regimes;
        let means = params.means();
        let phi = params.ar_coefs(s0);
        let mut mean_pred = means[s0];
        for (l, &phi_l) in phi.iter().enumerate() {
            let s_lag = self.digit(a, l + 1);
            mean_pred += phi_l * (self.y[t - (l + 1)] - means[s_lag]);
        }
        let var = params.variance(s0);
        let sd = var.sqrt();
        let z = (self.y[t] - mean_pred) / sd;
        (StdNormal.ln_pdf(z) - sd.ln()).exp()
    }

    /// The stationary joint distribution of the expanded initial state
    /// `(S_p, S_{p-1}, ..., S_0)`, `P(z_p) = pi_{S_0} * prod_{l=0}^{p-1}
    /// P(S_{t-l} = s_l | S_{t-l-1} = s_{l+1})`.
    fn stationary_joint(&self, params: &MsarParams, pi: &[f64]) -> Vec<f64> {
        let p = self.spec.order;
        let mut joint = vec![0.0; self.m];
        for (a, slot) in joint.iter_mut().enumerate() {
            let mut prob = pi[self.digit(a, p)];
            for l in 0..p {
                prob *= params.transition(self.digit(a, l), self.digit(a, l + 1));
            }
            *slot = prob;
        }
        joint
    }

    /// Runs the Hamilton (1989) forward filter, retaining the expanded-state
    /// quantities the smoother and EM step consume.
    fn run_filter(&self, params: &MsarParams) -> Result<FilterState, RegimeError> {
        self.check_params(params)?;
        let k = self.spec.k_regimes;
        let p = self.spec.order;
        let t_total = self.y.len();
        let n = t_total - p;
        let pi = self.stationary_distribution(params)?;

        let mut filtered_marginal: Vec<Vec<f64>> = Vec::with_capacity(n);
        let mut filtered_joint: Vec<Vec<f64>> = Vec::with_capacity(n);
        let mut predicted_joint: Vec<Vec<f64>> = Vec::with_capacity(n);
        let mut loglik = 0.0;

        for t_idx in 0..n {
            let t = p + t_idx;

            // One-step predicted joint P(z_t | Y_{t-1}).
            let predicted = if t_idx == 0 {
                self.stationary_joint(params, &pi)
            } else {
                let prev = &filtered_joint[t_idx - 1];
                let mut pred = vec![0.0; self.m];
                for (a, slot) in pred.iter_mut().enumerate() {
                    let s0 = a % k;
                    let s1 = self.digit(a, 1);
                    // Predecessor states share digits (s_1, ..., s_p); the
                    // oldest lag S_{t-1-p} is marginalized out.
                    let core: usize = (1..=p).map(|l| self.digit(a, l) * self.pow_k[l - 1]).sum();
                    let mut acc = 0.0;
                    for x in 0..k {
                        acc += prev[core + x * self.pow_k[p]];
                    }
                    *slot = params.transition(s0, s1) * acc;
                }
                pred
            };

            // Update with the observation density.
            let mut joint = vec![0.0; self.m];
            let mut lik_t = 0.0;
            for (a, slot) in joint.iter_mut().enumerate() {
                let val = predicted[a] * self.state_density(params, a, t);
                *slot = val;
                lik_t += val;
            }
            if !(lik_t.is_finite() && lik_t > 0.0) {
                return Err(RegimeError::NonFinite {
                    what: "filter mixture likelihood (regimes explain no observation)",
                });
            }
            loglik += lik_t.ln();

            let mut marginal = vec![0.0; k];
            for (a, slot) in joint.iter_mut().enumerate() {
                *slot /= lik_t;
                marginal[a % k] += *slot;
            }

            filtered_marginal.push(marginal);
            filtered_joint.push(joint);
            predicted_joint.push(predicted);
        }

        Ok(FilterState {
            n,
            loglik,
            filtered_marginal,
            filtered_joint,
            predicted_joint,
        })
    }

    /// The Kim (1994) fixed-interval smoother over the expanded state,
    /// returning `n`-by-`m` smoothed joint probabilities `P(z_t | Y_T)`.
    ///
    /// Backward pass (Kim 1994, eq. 10): `P(z_t = a | Y_T) = P(z_t = a |
    /// Y_t) * sum_b P(z_{t+1} = b | Y_T) P(z_{t+1} = b | z_t = a) / P(z_{t+1}
    /// = b | Y_t)`, where a successor `b = (r0, s0, ..., s_{p-1})` inherits
    /// `a`'s recent lags and `P(z_{t+1} = b | z_t = a) = P(S_{t+1} = r0 |
    /// S_t = s0)`.
    fn run_smoother(&self, params: &MsarParams, fs: &FilterState) -> Vec<Vec<f64>> {
        let k = self.spec.k_regimes;
        let p = self.spec.order;
        let n = fs.n;
        let mut smoothed = vec![Vec::new(); n];
        smoothed[n - 1] = fs.filtered_joint[n - 1].clone();

        for t_idx in (0..n - 1).rev() {
            let next_smoothed = &smoothed[t_idx + 1];
            let next_predicted = &fs.predicted_joint[t_idx + 1];
            let filtered = &fs.filtered_joint[t_idx];
            let mut sm = vec![0.0; self.m];
            for (a, slot) in sm.iter_mut().enumerate() {
                let s0 = a % k;
                // Successor states b = (r0, s0, s1, ..., s_{p-1}); the
                // shared digits are a's non-oldest lags.
                let shifted: usize = (0..p).map(|l| self.digit(a, l) * self.pow_k[l + 1]).sum();
                let mut acc = 0.0;
                for r0 in 0..k {
                    let b = r0 + shifted;
                    let denom = next_predicted[b];
                    if denom > 0.0 {
                        acc += next_smoothed[b] * params.transition(r0, s0) / denom;
                    }
                }
                *slot = filtered[a] * acc;
            }
            smoothed[t_idx] = sm;
        }
        smoothed
    }

    /// Collapses expanded joint probabilities to marginal regime
    /// probabilities `P(S_t = i | .)` by summing over the lag digits.
    fn marginalize(&self, joint: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let k = self.spec.k_regimes;
        joint
            .iter()
            .map(|row| {
                let mut marg = vec![0.0; k];
                for (a, &v) in row.iter().enumerate() {
                    marg[a % k] += v;
                }
                marg
            })
            .collect()
    }

    /// Evaluates the log-likelihood at `params` via the prediction-error
    /// decomposition of the Hamilton filter (Hamilton 1989, eq. 4-5).
    pub fn loglike(&self, params: &MsarParams) -> Result<f64, RegimeError> {
        Ok(self.run_filter(params)?.loglik)
    }

    /// Runs the Hamilton (1989) filter, returning the log-likelihood and
    /// the `n`-by-`k` filtered regime probabilities `P(S_t | Y_t)` where `n
    /// = T - order`.
    pub fn filter(&self, params: &MsarParams) -> Result<FilterResult, RegimeError> {
        let fs = self.run_filter(params)?;
        Ok(FilterResult {
            loglik: fs.loglik,
            filtered_prob: fs.filtered_marginal,
        })
    }

    /// Runs the Hamilton filter followed by the Kim (1994) smoother,
    /// returning the log-likelihood, the filtered probabilities, and the
    /// smoothed regime probabilities `P(S_t | Y_T)`.
    pub fn smooth(&self, params: &MsarParams) -> Result<SmoothResult, RegimeError> {
        let fs = self.run_filter(params)?;
        let smoothed_joint = self.run_smoother(params, &fs);
        let smoothed_prob = self.marginalize(&smoothed_joint);
        Ok(SmoothResult {
            loglik: fs.loglik,
            filtered_prob: fs.filtered_marginal,
            smoothed_prob,
        })
    }

    /// Estimates the parameters by the EM (Baum-Welch) algorithm.
    ///
    /// The E-step is the Hamilton filter and Kim smoother; the M-step
    /// applies the closed-form transition-count update (Hamilton 1990,
    /// Kim & Nelson 1999, §4.3) together with expectation-conditional-
    /// maximization updates of the Gaussian block (means, then AR
    /// coefficients, then variances), each an exact weighted least-squares
    /// solve. This monotonically increases the observed-data log-likelihood
    /// (Dempster, Laird & Rubin 1977).
    ///
    /// Estimation starts from `start` and stops when the log-likelihood
    /// increment falls below `tol` or after `max_iter` iterations. The
    /// initial regime distribution is *not* free: it is the stationary
    /// distribution implied by the estimated transition matrix, matching
    /// the `statsmodels` steady-state convention.
    ///
    /// Because the likelihood surface is multimodal, EM converges to a
    /// *local* optimum determined by `start`; callers should assess fits by
    /// log-likelihood improvement and approximate parameter recovery rather
    /// than exact agreement with any single optimum. `order >= 1` and a
    /// non-constant `y` are required.
    ///
    /// The estimator is **scale-equivariant**: fitting `c * y` from the
    /// correspondingly scaled start (`c * mu`, `c^2 * sigma^2`, the same AR
    /// coefficients and transition matrix) returns `c * mu`, `c^2 *
    /// sigma^2`, and an unchanged transition matrix, AR block, smoothed
    /// probability path, iteration count and convergence flag, with the
    /// log-likelihood shifted by the Jacobian `-n ln c`. Changing the unit
    /// the series is measured in does not change the estimate.
    pub fn fit(
        &self,
        start: &MsarParams,
        max_iter: usize,
        tol: f64,
    ) -> Result<FitResult, RegimeError> {
        self.check_params(start)?;
        if self.spec.order < 1 {
            return Err(RegimeError::InvalidSpec {
                what: "fit requires order >= 1 (a Markov-switching autoregression)",
            });
        }
        // The M-step measures its variance floor against the scale of the
        // data (see `VARIANCE_FLOOR_REL`), which a constant series does not
        // have; such a series also identifies no regimes and drives the
        // likelihood to infinity. Scoring fixed parameters on it
        // (`filter`/`smooth`/`loglike`) remains well defined.
        if !is_usable_scale(self.y_var) {
            return Err(RegimeError::InvalidParameter {
                name: "y",
                value: self.y_var,
                requirement: "a series with strictly positive sample variance \
                              (a constant series identifies no regimes)",
            });
        }

        let mut params = start.clone();
        let mut prev_ll = f64::NEG_INFINITY;
        let mut converged = false;
        let mut iterations = 0;

        for iter in 0..max_iter {
            iterations = iter + 1;
            let fs = self.run_filter(&params)?;
            let ll = fs.loglik;
            if !ll.is_finite() {
                return Err(RegimeError::NonFinite {
                    what: "EM log-likelihood",
                });
            }
            if iter > 0 && (ll - prev_ll) <= tol {
                converged = true;
                break;
            }
            prev_ll = ll;
            let smoothed_joint = self.run_smoother(&params, &fs);
            params = self.m_step(&params, &fs, &smoothed_joint)?;
        }

        // Final consistent E-pass at the returned parameters.
        let fs = self.run_filter(&params)?;
        let smoothed_joint = self.run_smoother(&params, &fs);
        let smoothed_prob = self.marginalize(&smoothed_joint);
        Ok(FitResult {
            params,
            loglik: fs.loglik,
            smoothed_prob,
            iterations,
            converged,
        })
    }

    /// One EM M-step: closed-form transition update and ECM Gaussian
    /// updates. `smoothed` holds the expanded smoothed joint `P(z_t |
    /// Y_T)`; the current `params` supply the working AR/variance values
    /// for the conditional maximizations.
    fn m_step(
        &self,
        current: &MsarParams,
        fs: &FilterState,
        smoothed: &[Vec<f64>],
    ) -> Result<MsarParams, RegimeError> {
        let k = self.spec.k_regimes;
        let n = fs.n;

        // --- Transition matrix: expected transition counts. ---
        let mut trans_num = vec![0.0; k * k]; // [i][j]
        let mut trans_den = vec![0.0; k]; // over j = S_{t-1}
        for row in smoothed.iter().take(n) {
            for (a, &w) in row.iter().enumerate() {
                let s0 = a % k;
                let s1 = self.digit(a, 1);
                trans_num[s0 * k + s1] += w;
                trans_den[s1] += w;
            }
        }
        let mut transition = vec![vec![0.0; k]; k];
        for j in 0..k {
            if trans_den[j] <= 0.0 {
                // Regime never occupied under the smoother; keep the prior
                // column so the matrix stays stochastic.
                for (i, trow) in transition.iter_mut().enumerate() {
                    trow[j] = current.transition(i, j);
                }
            } else {
                for (i, trow) in transition.iter_mut().enumerate() {
                    trow[j] = trans_num[i * k + j] / trans_den[j];
                }
            }
        }

        // --- Gaussian block via ECM (two coordinate passes). ---
        let mut means = current.means().to_vec();
        let mut ar: Vec<Vec<f64>> = (0..self.spec.ar_blocks())
            .map(|b| {
                current
                    .ar_coefs(if self.spec.switching_ar { b } else { 0 })
                    .to_vec()
            })
            .collect();
        let mut variances = (0..self.spec.variance_blocks())
            .map(|b| current.variance(if self.spec.switching_variance { b } else { 0 }))
            .collect::<Vec<_>>();

        for _pass in 0..2 {
            self.update_means(fs, smoothed, &mut means, &ar, &variances)?;
            self.update_ar(fs, smoothed, &means, &mut ar, &variances)?;
            self.update_variances(fs, smoothed, &means, &ar, &mut variances);
        }

        MsarParams::new(transition, means, ar, variances)
    }

    /// Weight `g / sigma^2_{s0}`, residual, and its mean/AR sensitivities
    /// for expanded state `a` at time `t` under working `means`/`ar`.
    #[inline]
    fn variance_for(&self, variances: &[f64], s0: usize) -> f64 {
        if self.spec.switching_variance {
            variances[s0]
        } else {
            variances[0]
        }
    }

    #[inline]
    fn ar_for<'b>(&self, ar: &'b [Vec<f64>], s0: usize) -> &'b [f64] {
        if self.spec.switching_ar {
            &ar[s0]
        } else {
            &ar[0]
        }
    }

    /// ECM update of the regime means: exact weighted-least-squares solve
    /// of the `k`-dimensional normal equations for `mu` holding the AR
    /// coefficients and variances fixed. The residual `e_t(z) = (y_t -
    /// mu_{s0}) - sum_l phi_l (y_{t-l} - mu_{s_l})` is affine in `mu`, so
    /// `mu` solves `M mu = r` with `M = sum g/sigma^2 a a'`, `r = -sum
    /// g/sigma^2 d a`.
    fn update_means(
        &self,
        fs: &FilterState,
        smoothed: &[Vec<f64>],
        means: &mut [f64],
        ar: &[Vec<f64>],
        variances: &[f64],
    ) -> Result<(), RegimeError> {
        let k = self.spec.k_regimes;
        let p = self.spec.order;
        let mut mat = vec![0.0; k * k];
        let mut rhs = vec![0.0; k];

        for (t_idx, row) in smoothed.iter().enumerate().take(fs.n) {
            let t = p + t_idx;
            for (a, &g) in row.iter().enumerate() {
                if g == 0.0 {
                    continue;
                }
                let s0 = a % k;
                let phi = self.ar_for(ar, s0);
                let v = self.variance_for(variances, s0);
                let gw = g / v;

                // d = y_t - sum_l phi_l y_{t-l}; alpha = d(e)/d(mu).
                let mut d = self.y[t];
                let mut alpha = vec![0.0; k];
                alpha[s0] -= 1.0;
                for (l, &phi_l) in phi.iter().enumerate() {
                    d -= phi_l * self.y[t - (l + 1)];
                    alpha[self.digit(a, l + 1)] += phi_l;
                }

                for r in 0..k {
                    if alpha[r] == 0.0 {
                        continue;
                    }
                    rhs[r] -= gw * d * alpha[r];
                    let gwa = gw * alpha[r];
                    for c in 0..k {
                        mat[r * k + c] += gwa * alpha[c];
                    }
                }
            }
        }
        // Ridge for numerical safety against an unoccupied regime, sized
        // relative to the information the matrix itself carries (see
        // `M_STEP_RIDGE_REL`); an absolute ridge here is a `1 / scale(y)^2`
        // quantity and would make the estimate depend on the unit of `y`.
        let scale = max_diagonal(&mat, k);
        if !is_usable_scale(scale) {
            // No observation carries information about any mean at this
            // pass, so `rhs` is zero as well and every `mu` maximizes the
            // objective equally: keep the incumbent means.
            return Ok(());
        }
        for i in 0..k {
            mat[i * k + i] += M_STEP_RIDGE_REL * scale;
        }
        let solved = solve(mat, rhs, k, "M-step means")?;
        means.copy_from_slice(&solved);
        Ok(())
    }

    /// ECM update of the AR coefficients: exact weighted-least-squares solve
    /// of the `order`-dimensional normal equations holding the means and
    /// variances fixed. With `u_t = y_t - mu_{s0}` and `x_{t,l} = y_{t-l} -
    /// mu_{s_l}`, `phi` solves `A phi = b`, `A = sum g/sigma^2 x x'`, `b =
    /// sum g/sigma^2 x u`, pooled over regimes (shared AR) or per regime
    /// (switching AR).
    fn update_ar(
        &self,
        fs: &FilterState,
        smoothed: &[Vec<f64>],
        means: &[f64],
        ar: &mut [Vec<f64>],
        variances: &[f64],
    ) -> Result<(), RegimeError> {
        let k = self.spec.k_regimes;
        let p = self.spec.order;
        if p == 0 {
            return Ok(());
        }
        let blocks = self.spec.ar_blocks();
        let mut mats = vec![vec![0.0; p * p]; blocks];
        let mut rhss = vec![vec![0.0; p]; blocks];

        for (t_idx, row) in smoothed.iter().enumerate().take(fs.n) {
            let t = p + t_idx;
            for (a, &g) in row.iter().enumerate() {
                if g == 0.0 {
                    continue;
                }
                let s0 = a % k;
                let v = self.variance_for(variances, s0);
                let gw = g / v;
                let block = if self.spec.switching_ar { s0 } else { 0 };

                let u = self.y[t] - means[s0];
                let mut x = vec![0.0; p];
                for l in 0..p {
                    x[l] = self.y[t - (l + 1)] - means[self.digit(a, l + 1)];
                }
                let mat = &mut mats[block];
                let rhs = &mut rhss[block];
                for l in 0..p {
                    let gwx = gw * x[l];
                    rhs[l] += gwx * u;
                    for mcol in 0..p {
                        mat[l * p + mcol] += gwx * x[mcol];
                    }
                }
            }
        }

        for (block, ar_block) in ar.iter_mut().enumerate() {
            let mut mat = std::mem::take(&mut mats[block]);
            // The AR normal equations are `sum g/sigma^2 x x'`, in which the
            // two powers of `scale(y)` in `x x'` cancel the two in
            // `sigma^2`, so a rescaling of `y` leaves them alone; their
            // magnitude still depends on the sample size and the
            // signal-to-noise ratio, so the ridge is measured against the
            // matrix rather than fixed (see `M_STEP_RIDGE_REL`).
            let scale = max_diagonal(&mat, p);
            if !is_usable_scale(scale) {
                // The lagged regressors carry no weighted variation; the
                // objective is flat in this block, so keep the incumbent.
                continue;
            }
            for i in 0..p {
                mat[i * p + i] += M_STEP_RIDGE_REL * scale;
            }
            let rhs = std::mem::take(&mut rhss[block]);
            let solved = solve(mat, rhs, p, "M-step AR coefficients")?;
            ar_block.copy_from_slice(&solved);
        }
        Ok(())
    }

    /// ECM update of the innovation variances: the weighted residual second
    /// moment `sigma^2_i = sum g e^2 / sum g` (switching), or its pooled
    /// counterpart (shared). Floored at a negligible fraction of the sample
    /// variance of `y` (see `VARIANCE_FLOOR_REL`), which keeps the floor in
    /// the units the variances are measured in.
    fn update_variances(
        &self,
        fs: &FilterState,
        smoothed: &[Vec<f64>],
        means: &[f64],
        ar: &[Vec<f64>],
        variances: &mut [f64],
    ) {
        let k = self.spec.k_regimes;
        let p = self.spec.order;
        let blocks = self.spec.variance_blocks();
        let mut num = vec![0.0; blocks];
        let mut den = vec![0.0; blocks];

        for (t_idx, row) in smoothed.iter().enumerate().take(fs.n) {
            let t = p + t_idx;
            for (a, &g) in row.iter().enumerate() {
                if g == 0.0 {
                    continue;
                }
                let s0 = a % k;
                let phi = self.ar_for(ar, s0);
                let mut e = self.y[t] - means[s0];
                for (l, &phi_l) in phi.iter().enumerate() {
                    e -= phi_l * (self.y[t - (l + 1)] - means[self.digit(a, l + 1)]);
                }
                let block = if self.spec.switching_variance { s0 } else { 0 };
                num[block] += g * e * e;
                den[block] += g;
            }
        }
        let floor = VARIANCE_FLOOR_REL * self.y_var;
        for b in 0..blocks {
            if den[b] > 0.0 {
                variances[b] = (num[b] / den[b]).max(floor);
            }
        }
    }
}
