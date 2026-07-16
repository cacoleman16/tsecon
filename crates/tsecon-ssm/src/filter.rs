//! Kalman filtering: the univariate (sequential) filter with exact diffuse
//! initialization — the primary path — and a standard matrix filter with a
//! Joseph-form covariance update as the cross-check path.
//!
//! # Univariate (sequential) filtering
//!
//! Following Koopman & Durbin (2000) and Durbin & Koopman (2012, §6.4), the
//! elements of the observation vector `y_t` are brought in one at a time.
//! With `Z_i` the `i`-th row of `Z`, `sigma2_i = H_ii`, and `(a_{t,i},
//! P_{t,i})` the state moments after absorbing elements `1..i-1` of `y_t`:
//!
//! ```text
//! v_{t,i} = y_{t,i} - d_i - Z_i a_{t,i}
//! F_{t,i} = Z_i P_{t,i} Z_i' + sigma2_i,      M_{t,i} = P_{t,i} Z_i'
//! a_{t,i+1} = a_{t,i} + M_{t,i} v_{t,i} / F_{t,i}
//! P_{t,i+1} = P_{t,i} - M_{t,i} M_{t,i}' / F_{t,i}
//! ```
//!
//! followed by the time transition `a_{t+1,1} = c + T a_{t,p+1}`,
//! `P_{t+1,1} = T P_{t,p+1} T' + R Q R'`. No `p x p` inversion ever occurs,
//! missing elements (NaN in `y`) are simply skipped, and the log-likelihood
//! accumulates one scalar prediction-error term per observed element:
//! `-(ln 2*pi + ln F_{t,i} + v_{t,i}^2 / F_{t,i}) / 2`.
//!
//! This decomposition of the joint Gaussian density is exact only when `H`
//! is diagonal; a non-diagonal `H` is rejected with
//! [`SsmError::NonDiagonalH`].
//! // TODO(phase0): LDL' pre-whitening transform to lift the diagonal-H
//! // restriction (Durbin & Koopman 2012, §6.4.1).
//!
//! # Exact diffuse initialization
//!
//! Diffuse states are handled exactly (Koopman 1997; Koopman & Durbin 2003;
//! Durbin & Koopman 2012, ch. 5) via the two-matrix decomposition
//! `P_t = P_{*,t} + kappa P_{inf,t}` with `kappa -> infinity` — never a
//! large-kappa approximation. Per scalar element, with
//! `F_{inf} = Z_i P_inf Z_i'`, `F_* = Z_i P_* Z_i' + sigma2_i`,
//! `M_inf = P_inf Z_i'`, `M_* = P_* Z_i'`, and `F_inf > 0`:
//!
//! ```text
//! K0 = M_inf / F_inf,     K1 = M_* / F_inf - M_inf F_* / F_inf^2
//! a      <- a + K0 v
//! P_*    <- P_* - M_* K0' - M_inf K1'
//! P_inf  <- P_inf - M_inf K0'
//! loglik <- loglik - (ln 2*pi + ln F_inf) / 2
//! ```
//!
//! while an element with `F_inf = 0` gets the ordinary update above. The
//! diffuse period ends at the first `t` whose incoming `P_inf` is
//! numerically zero; conventions (tolerances, the `-(ln 2*pi + ln F_inf)/2`
//! diffuse likelihood contribution) match statsmodels'
//! `use_exact_diffuse=True` filter exactly, so log-likelihoods are directly
//! comparable. (Cross-package caution: some implementations, e.g. following
//! Francke, Koopman & de Vos 2010, omit constants from the diffuse
//! contribution.)

use tsecon_linalg::faer::{Mat, MatRef};
use tsecon_linalg::jittered_cholesky;

use crate::dense::{
    axpy, chol_solve, dot, frob_sq, mat_vec, outer_sub, row_to_vec, sandwich, symmetrize_in_place,
};
use crate::error::SsmError;
use crate::model::LinearGaussianSSM;

/// Numerical tolerance for the diffuse recursions, matching statsmodels'
/// `tolerance_diffuse`: a time step is diffuse while `||P_inf||_F^2`
/// exceeds it, an element takes the diffuse update while `F_inf` exceeds
/// it, and an element with `F_* <= tolerance` is treated as carrying no
/// information (skipped, like a missing value).
pub(crate) const TOLERANCE_DIFFUSE: f64 = 1e-10;

/// `ln(2 pi)`, the per-element likelihood constant.
#[inline]
fn ln_2pi() -> f64 {
    (2.0 * std::f64::consts::PI).ln()
}

/// Per-element filtering record kept for the backward smoothing pass
/// (the univariate smoother re-derives its gains from these).
#[derive(Debug, Clone)]
pub(crate) struct ObsStep {
    /// True when the element was observed *and* informative (`F > 0`);
    /// missing (NaN) and numerically singular elements are both skipped
    /// by filter and smoother alike.
    pub(crate) observed: bool,
    /// Prediction error `v_{t,i}`.
    pub(crate) v: f64,
    /// Finite innovation variance `F_{*,t,i}` (includes `H_ii`).
    pub(crate) f_star: f64,
    /// Diffuse innovation variance `F_{inf,t,i}` (zero outside the
    /// diffuse update branch).
    pub(crate) f_inf: f64,
    /// `M_{*,t,i} = P_{*,t,i} Z_i'`.
    pub(crate) m_star: Vec<f64>,
    /// `M_{inf,t,i} = P_{inf,t,i} Z_i'`.
    pub(crate) m_inf: Vec<f64>,
}

impl ObsStep {
    /// Record for a skipped element (missing or numerically singular).
    fn skipped() -> Self {
        ObsStep {
            observed: false,
            v: 0.0,
            f_star: 0.0,
            f_inf: 0.0,
            m_star: Vec::new(),
            m_inf: Vec::new(),
        }
    }
}

/// Output of the univariate (sequential) Kalman filter.
///
/// Quantities follow the statsmodels reporting conventions: `filtered_*`
/// hold the posterior moments `a_{t|t}`, `P_{t|t}` and `predicted_*` hold
/// the one-step-ahead moments `a_{t+1|t}`, `P_{t+1|t}` (one extra trailing
/// entry for the post-sample prediction). During the diffuse period the
/// covariance entries are the *finite* parts `P_*`; the diffuse parts live
/// in [`FilterOutput::predicted_diffuse_state_cov`].
#[derive(Debug, Clone)]
pub struct FilterOutput {
    /// Joint log-likelihood by prediction-error decomposition, with the
    /// exact-diffuse contribution `-(ln 2*pi + ln F_inf)/2` during diffuse
    /// elements (matches statsmodels `use_exact_diffuse=True`).
    pub loglik: f64,
    /// Number of time periods in the (contiguous, initial) diffuse period:
    /// `t < d_diffuse` was filtered with the exact-diffuse recursions.
    pub d_diffuse: usize,
    /// Predicted state means `a_{t+1|t}`, indexed `0..=n` (entry `t` is
    /// the mean before absorbing `y_t`; entry `n` is the post-sample
    /// one-step prediction).
    pub predicted_state: Vec<Vec<f64>>,
    /// Predicted state covariances `P_{t+1|t}` (finite part `P_*` during
    /// the diffuse period), indexed `0..=n`.
    pub predicted_state_cov: Vec<Mat<f64>>,
    /// Predicted diffuse covariances `P_{inf,t}`, indexed `0..=n`; exactly
    /// zero from the end of the diffuse period onwards.
    pub predicted_diffuse_state_cov: Vec<Mat<f64>>,
    /// Filtered state means `a_{t|t}`, indexed `0..n`.
    pub filtered_state: Vec<Vec<f64>>,
    /// Filtered state covariances `P_{t|t}` (finite part during the
    /// diffuse period), indexed `0..n`.
    pub filtered_state_cov: Vec<Mat<f64>>,
    /// Per-element records, row-major (`t * p + i`), for the smoother.
    pub(crate) steps: Vec<ObsStep>,
}

/// Output of the standard matrix Kalman filter (the cross-check path).
///
/// Same reporting conventions as [`FilterOutput`]; no diffuse quantities
/// because this path requires a proper (finite-covariance) initialization.
#[derive(Debug, Clone)]
pub struct MatrixFilterOutput {
    /// Joint log-likelihood by prediction-error decomposition.
    pub loglik: f64,
    /// Predicted state means `a_{t+1|t}`, indexed `0..=n`.
    pub predicted_state: Vec<Vec<f64>>,
    /// Predicted state covariances `P_{t+1|t}`, indexed `0..=n`.
    pub predicted_state_cov: Vec<Mat<f64>>,
    /// Filtered state means `a_{t|t}`, indexed `0..n`.
    pub filtered_state: Vec<Vec<f64>>,
    /// Filtered state covariances `P_{t|t}`, indexed `0..n`.
    pub filtered_state_cov: Vec<Mat<f64>>,
}

/// Validates the observation matrix: `n x p` shape with `n >= 1`, entries
/// finite or NaN (NaN encodes "missing"; infinities are rejected).
fn validate_observations(model: &LinearGaussianSSM, y: MatRef<'_, f64>) -> Result<(), SsmError> {
    if y.ncols() != model.obs_dim() {
        return Err(SsmError::Dimension {
            what: "observations y must be n x p (one column per observation element)",
            expected: model.obs_dim(),
            got: y.ncols(),
        });
    }
    if y.nrows() == 0 {
        return Err(SsmError::InvalidArgument {
            what: "observations y must contain at least one time period",
        });
    }
    for j in 0..y.ncols() {
        for i in 0..y.nrows() {
            if y[(i, j)].is_infinite() {
                return Err(SsmError::NonFinite {
                    what: "y (entries must be finite or NaN-for-missing)",
                });
            }
        }
    }
    Ok(())
}

/// Runs the univariate (sequential) Kalman filter with exact diffuse
/// initialization — the primary filtering path (see the module docs for
/// the recursions and references).
///
/// `y` is `n x p` with NaN marking a missing element; individual elements
/// of a partially missing `y_t` are used.
///
/// # Errors
///
/// * [`SsmError::NonDiagonalH`] when `H` is not diagonal;
/// * [`SsmError::Dimension`] / [`SsmError::InvalidArgument`] /
///   [`SsmError::NonFinite`] on malformed `y`;
/// * [`SsmError::Linalg`] when the requested initialization cannot be
///   resolved (e.g. stationary initialization of an explosive `T`).
pub fn filter_univariate(
    model: &LinearGaussianSSM,
    y: MatRef<'_, f64>,
) -> Result<FilterOutput, SsmError> {
    if !model.h_is_diagonal() {
        return Err(SsmError::NonDiagonalH);
    }
    validate_observations(model, y)?;
    let n = y.nrows();
    let p = model.obs_dim();
    let ln2pi = ln_2pi();

    let init = model.initial_state()?;
    let mut a = init.a1;
    let mut p_star = init.p_star;
    let mut p_inf = init.p_inf;

    let mut out = FilterOutput {
        loglik: 0.0,
        d_diffuse: 0,
        predicted_state: Vec::with_capacity(n + 1),
        predicted_state_cov: Vec::with_capacity(n + 1),
        predicted_diffuse_state_cov: Vec::with_capacity(n + 1),
        filtered_state: Vec::with_capacity(n),
        filtered_state_cov: Vec::with_capacity(n),
        steps: Vec::with_capacity(n * p),
    };
    // Once P_inf collapses to (numerical) zero it stays zero, so the
    // diffuse period is a contiguous prefix; the flag makes that explicit.
    let mut in_diffuse = true;

    for t in 0..n {
        let z = model.z().at(t);
        let h = model.h().at(t);
        let tr = model.t().at(t);

        let diffuse = in_diffuse && frob_sq(p_inf.as_ref()) > TOLERANCE_DIFFUSE;
        if diffuse {
            out.d_diffuse += 1;
        } else {
            in_diffuse = false;
        }

        out.predicted_state.push(a.clone());
        out.predicted_state_cov.push(p_star.clone());
        out.predicted_diffuse_state_cov.push(p_inf.clone());

        // Sequential scalar updates: (a, p_star, p_inf) morph in place
        // from predicted to filtered moments.
        for i in 0..p {
            let yti = y[(t, i)];
            if yti.is_nan() {
                out.steps.push(ObsStep::skipped());
                continue;
            }
            let zi = row_to_vec(z, i);
            let v = yti - model.obs_intercept()[i] - dot(&zi, &a);
            let m_star = mat_vec(p_star.as_ref(), &zi);
            // Clamp roundoff-negative variances to zero (statsmodels does
            // the same) before branching.
            let f_star = (dot(&zi, &m_star) + h[(i, i)]).max(0.0);
            let (f_inf, m_inf) = if diffuse {
                let mi = mat_vec(p_inf.as_ref(), &zi);
                (dot(&zi, &mi).max(0.0), mi)
            } else {
                (0.0, Vec::new())
            };

            if f_inf > TOLERANCE_DIFFUSE {
                // Exact-diffuse element update (Koopman & Durbin 2003).
                let k0: Vec<f64> = m_inf.iter().map(|x| x / f_inf).collect();
                let f12 = -f_star / f_inf;
                let k1: Vec<f64> = m_star
                    .iter()
                    .zip(&k0)
                    .map(|(ms, k0j)| ms / f_inf + k0j * f12)
                    .collect();
                axpy(&mut a, v, &k0);
                // P_* <- P_* L0' + P_inf L1' = P_* - M_* K0' - M_inf K1'.
                outer_sub(&mut p_star, &m_star, &k0);
                outer_sub(&mut p_star, &m_inf, &k1);
                // P_inf <- P_inf L0' = P_inf - M_inf K0'.
                outer_sub(&mut p_inf, &m_inf, &k0);
                out.loglik -= 0.5 * (ln2pi + f_inf.ln());
                out.steps.push(ObsStep {
                    observed: true,
                    v,
                    f_star,
                    f_inf,
                    m_star,
                    m_inf,
                });
            } else if f_star > TOLERANCE_DIFFUSE {
                // Standard scalar update (Koopman & Durbin 2000).
                let k0: Vec<f64> = m_star.iter().map(|x| x / f_star).collect();
                axpy(&mut a, v, &k0);
                outer_sub(&mut p_star, &m_star, &k0);
                out.loglik -= 0.5 * (ln2pi + f_star.ln() + v * v / f_star);
                out.steps.push(ObsStep {
                    observed: true,
                    v,
                    f_star,
                    f_inf: 0.0,
                    m_star,
                    m_inf: Vec::new(),
                });
            } else {
                // Numerically singular element: carries no information.
                out.steps.push(ObsStep::skipped());
            }
        }

        // Rank-one updates drift off exact symmetry at roundoff level;
        // restore the invariant before the covariances are stored/reused.
        symmetrize_in_place(&mut p_star);
        symmetrize_in_place(&mut p_inf);

        out.filtered_state.push(a.clone());
        out.filtered_state_cov.push(p_star.clone());

        // Time transition: a <- c + T a, P_* <- T P_* T' + R Q R',
        // P_inf <- T P_inf T'.
        let mut a_next = mat_vec(tr, &a);
        axpy(&mut a_next, 1.0, model.state_intercept());
        a = a_next;
        let rqr = model.rqr(t)?;
        let mut ps_next = sandwich(tr, p_star.as_ref());
        ps_next += &rqr;
        p_star = ps_next;
        p_inf = sandwich(tr, p_inf.as_ref());
        symmetrize_in_place(&mut p_star);
        symmetrize_in_place(&mut p_inf);
    }

    out.predicted_state.push(a);
    out.predicted_state_cov.push(p_star);
    out.predicted_diffuse_state_cov.push(p_inf);
    Ok(out)
}

/// Runs the standard matrix Kalman filter with a Joseph-form covariance
/// update — the cross-check path (Durbin & Koopman 2012, §4.3).
///
/// Per period, with the observed subvector `y_t^o` (missing elements
/// dropped) and the correspondingly subset `Z^o`, `d^o`, `H^oo`:
///
/// ```text
/// v_t = y_t^o - d^o - Z^o a_t,      F_t = Z^o P_t Z^o' + H^oo
/// K_t = P_t Z^o' F_t^{-1},          a_{t|t} = a_t + K_t v_t
/// P_{t|t} = (I - K_t Z^o) P_t (I - K_t Z^o)' + K_t H^oo K_t'   (Joseph)
/// a_{t+1} = c + T a_{t|t},          P_{t+1} = T P_{t|t} T' + R Q R'
/// loglik += -(p_t^o ln 2*pi + ln det F_t + v_t' F_t^{-1} v_t) / 2
/// ```
///
/// The Joseph form is stable for any gain (Bucy & Joseph 1968); `F_t` is
/// factorized through the shared jitter-ladder Cholesky, so `ln det F_t`
/// comes from the factor diagonal, never an explicit determinant.
///
/// Unlike the univariate path this filter accepts a non-diagonal `H`, but
/// it requires a proper initialization: exact-diffuse models are rejected
/// with [`SsmError::DiffuseNotSupported`] (use [`filter_univariate`]).
///
/// # Errors
///
/// * [`SsmError::DiffuseNotSupported`] when the initialization has a
///   diffuse part;
/// * [`SsmError::Dimension`] / [`SsmError::InvalidArgument`] /
///   [`SsmError::NonFinite`] on malformed `y`;
/// * [`SsmError::Linalg`] when `F_t` is not positive definite within the
///   jitter ladder, or the initialization cannot be resolved.
pub fn filter_matrix(
    model: &LinearGaussianSSM,
    y: MatRef<'_, f64>,
) -> Result<MatrixFilterOutput, SsmError> {
    validate_observations(model, y)?;
    let init = model.initial_state()?;
    if init.has_diffuse() {
        return Err(SsmError::DiffuseNotSupported {
            what: "the standard matrix Kalman filter",
        });
    }
    let n = y.nrows();
    let p = model.obs_dim();
    let m = model.state_dim();
    let ln2pi = ln_2pi();

    let mut a = init.a1;
    let mut p_mat = init.p_star;

    let mut out = MatrixFilterOutput {
        loglik: 0.0,
        predicted_state: Vec::with_capacity(n + 1),
        predicted_state_cov: Vec::with_capacity(n + 1),
        filtered_state: Vec::with_capacity(n),
        filtered_state_cov: Vec::with_capacity(n),
    };

    for t in 0..n {
        let z = model.z().at(t);
        let h = model.h().at(t);
        let tr = model.t().at(t);

        out.predicted_state.push(a.clone());
        out.predicted_state_cov.push(p_mat.clone());

        let obs: Vec<usize> = (0..p).filter(|&i| !y[(t, i)].is_nan()).collect();
        let k = obs.len();
        if k > 0 {
            // Subset system for the observed elements.
            let z_o = Mat::from_fn(k, m, |r, c| z[(obs[r], c)]);
            let h_oo = Mat::from_fn(k, k, |r, c| h[(obs[r], obs[c])]);
            let v: Vec<f64> = obs
                .iter()
                .map(|&i| y[(t, i)] - model.obs_intercept()[i] - dot(&row_to_vec(z, i), &a))
                .collect();

            // M = P Z_o' (m x k), F = Z_o M + H_oo (k x k).
            let m_mat = p_mat.as_ref() * z_o.as_ref().transpose();
            let mut f = z_o.as_ref() * m_mat.as_ref();
            f += &h_oo;
            symmetrize_in_place(&mut f);
            let chol = jittered_cholesky(f.as_ref())?;

            // K = M F^{-1} (m x k): row r of K solves F x = (row r of M)'.
            let mut kal = Mat::<f64>::zeros(m, k);
            for r in 0..m {
                let x = chol_solve(&chol.factor, &row_to_vec(m_mat.as_ref(), r));
                for (c, xc) in x.iter().enumerate() {
                    kal[(r, c)] = *xc;
                }
            }

            // a_{t|t} = a_t + K v (K already contains F^{-1}).
            axpy(&mut a, 1.0, &mat_vec(kal.as_ref(), &v));
            let f_inv_v = chol_solve(&chol.factor, &v);

            // Joseph update: P <- (I - K Z_o) P (I - K Z_o)' + K H_oo K'.
            let kz = kal.as_ref() * z_o.as_ref();
            let i_kz = Mat::from_fn(m, m, |r, c| {
                let eye = if r == c { 1.0 } else { 0.0 };
                eye - kz[(r, c)]
            });
            let mut p_new = sandwich(i_kz.as_ref(), p_mat.as_ref());
            let kh = kal.as_ref() * h_oo.as_ref();
            let khk = kh.as_ref() * kal.as_ref().transpose();
            p_new += &khk;
            symmetrize_in_place(&mut p_new);
            p_mat = p_new;

            out.loglik -= 0.5 * (k as f64 * ln2pi + chol.log_det() + dot(&v, &f_inv_v));
        }

        out.filtered_state.push(a.clone());
        out.filtered_state_cov.push(p_mat.clone());

        // Time transition.
        let mut a_next = mat_vec(tr, &a);
        axpy(&mut a_next, 1.0, model.state_intercept());
        a = a_next;
        let rqr = model.rqr(t)?;
        let mut p_next = sandwich(tr, p_mat.as_ref());
        p_next += &rqr;
        symmetrize_in_place(&mut p_next);
        p_mat = p_next;
    }

    out.predicted_state.push(a);
    out.predicted_state_cov.push(p_mat);
    Ok(out)
}

impl LinearGaussianSSM {
    /// Univariate (sequential) Kalman filter — the primary path; see
    /// [`filter_univariate`].
    pub fn filter(&self, y: MatRef<'_, f64>) -> Result<FilterOutput, SsmError> {
        filter_univariate(self, y)
    }

    /// Standard matrix Kalman filter with Joseph-form update — the
    /// cross-check path; see [`filter_matrix`].
    pub fn filter_matrix(&self, y: MatRef<'_, f64>) -> Result<MatrixFilterOutput, SsmError> {
        filter_matrix(self, y)
    }

    /// Log-likelihood of `y` by prediction-error decomposition (with the
    /// exact-diffuse correction during diffuse steps), via the univariate
    /// filter.
    pub fn loglike(&self, y: MatRef<'_, f64>) -> Result<f64, SsmError> {
        Ok(filter_univariate(self, y)?.loglik)
    }
}
