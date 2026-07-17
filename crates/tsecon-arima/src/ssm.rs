//! State-space construction: the Harvey canonical form for ARMA(p, q).

use tsecon_linalg::faer::Mat;
use tsecon_ssm::{Initialization, LinearGaussianSSM};

use crate::error::ArimaError;

/// Builds the linear-Gaussian state-space form of a (stationary)
/// ARMA(p, q) process with intercept,
///
/// ```text
/// y_t = c + phi_1 y_{t-1} + ... + phi_p y_{t-p}
///         + eps_t + theta_1 eps_{t-1} + ... + theta_q eps_{t-q},
/// eps_t ~ N(0, sigma2),
/// ```
///
/// in the Harvey canonical form (Harvey 1989, section 3.3; Jones 1980)
/// with state dimension `m = max(p, q + 1)`:
///
/// ```text
/// y_t = Z alpha_t,                    Z = [1, 0, ..., 0],  H = 0
/// alpha_{t+1} = c e_1 + T alpha_t + R eta_t,   eta_t ~ N(0, sigma2)
/// T[i][0] = phi_{i+1} (zero past p),  T[i][i+1] = 1
/// R = [1, theta_1, ..., theta_q, 0, ..., 0]'
/// ```
///
/// This generalizes [`LinearGaussianSSM::ar`] to MA terms with the same
/// conventions: the first state element is `y_t` itself, and the
/// intercept enters the state equation as `c e_1` — exactly statsmodels
/// `SARIMAX(trend='c')`, so `c` is the regression constant, *not* the
/// process mean (the mean is `c / (1 - phi_1 - ... - phi_p)`).
///
/// Initialization is stationary: the unconditional mean
/// `a_1 = (I - T)^{-1} c e_1` and the discrete-Lyapunov unconditional
/// covariance `P_1 = T P_1 T' + sigma2 R R'` (Lütkepohl 2005, section
/// 2.1.4), solved by `tsecon-linalg`. Filtering a model whose AR
/// coefficients are non-stationary therefore fails with a wrapped
/// [`tsecon_linalg::LinalgError::Unstable`].
///
/// ARMA(0, 0) is valid and yields the one-dimensional degenerate system
/// `y_t = c + eps_t`.
///
/// # Errors
///
/// [`ArimaError::NonFinite`] for NaN/infinite coefficients or intercept,
/// [`ArimaError::InvalidArgument`] for `sigma2 <= 0` (or non-finite), and
/// [`ArimaError::Ssm`] if the assembled system fails the engine's
/// validation.
pub fn arma_ssm(
    ar: &[f64],
    ma: &[f64],
    sigma2: f64,
    intercept: f64,
) -> Result<LinearGaussianSSM, ArimaError> {
    if ar.iter().chain(ma.iter()).any(|v| !v.is_finite()) {
        return Err(ArimaError::NonFinite {
            what: "ARMA coefficients",
        });
    }
    if !intercept.is_finite() {
        return Err(ArimaError::NonFinite { what: "intercept" });
    }
    if !sigma2.is_finite() || sigma2 <= 0.0 {
        return Err(ArimaError::InvalidArgument {
            what: "sigma2 must be strictly positive and finite",
        });
    }
    let p = ar.len();
    let q = ma.len();
    let m = p.max(q + 1);

    let t = Mat::from_fn(m, m, |i, j| {
        if j == 0 && i < p {
            ar[i]
        } else if j == i + 1 {
            1.0
        } else {
            0.0
        }
    });
    let z = Mat::from_fn(1, m, |_, j| if j == 0 { 1.0 } else { 0.0 });
    let r = Mat::from_fn(m, 1, |i, _| {
        if i == 0 {
            1.0
        } else if i <= q {
            ma[i - 1]
        } else {
            0.0
        }
    });
    let mut c = vec![0.0; m];
    c[0] = intercept;

    Ok(LinearGaussianSSM::builder(1, m, 1)
        .z(z)
        .h(Mat::zeros(1, 1))
        .t(t)
        .r(r)
        .q(Mat::from_fn(1, 1, |_, _| sigma2))
        .state_intercept(c)
        .initialization(Initialization::Stationary)
        .build()?)
}
