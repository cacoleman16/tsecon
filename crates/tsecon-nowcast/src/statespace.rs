//! The dynamic-factor model as a linear-Gaussian state space, and the
//! fixed-parameter Kalman filter/smoother pass.
//!
//! For `r = k_factors` common factors following a VAR(`p`) and white-noise
//! (`error_order = 0`) idiosyncratic components, the model is
//!
//! ```text
//! y_t = Lambda f_t + e_t,                 e_t ~ N(0, diag(sigma2))   (N x 1)
//! F_t = A_1 F_{t-1} + ... + A_p F_{t-p} + eta_t,  eta_t ~ N(0, Q)   (r x 1)
//! ```
//!
//! cast into the stacked companion state `alpha_t = [F_t', ..., F_{t-p+1}']'`
//! of dimension `m = r p`:
//!
//! ```text
//! Z = [ Lambda | 0 ]  (N x m)          d = 0
//! T = [ A_1  A_2 ... A_p ]  (m x m)    c = 0
//!     [ I_r   0  ...  0  ]
//!     [  0   I_r ...  0  ]
//!     [       ...        ]
//! R = [ I_r ; 0 ]  (m x r)             Q  (r x r)
//! H = diag(sigma2)  (N x N)
//! ```
//!
//! with **stationary** initialization (the unconditional moments of the
//! stationary state, via the discrete Lyapunov equation). This is *exactly*
//! statsmodels' `DynamicFactor(k_factors=r, factor_order=p, error_order=0)`
//! representation: AR coefficients occupy the first `r` rows of `T`, the
//! sub-diagonal identity blocks shift the lag stack, the factor innovation
//! covariance is `Q` (statsmodels normalizes it to `I`), the idiosyncratic
//! variances form the diagonal observation covariance `H`, and both
//! intercepts are zero. Consequently, given the same parameters and panel,
//! [`smooth_fixed`] reproduces statsmodels' Kalman log-likelihood and
//! smoothed states to machine precision (validated in `tests/golden.rs`).

use tsecon_linalg::faer::{Mat, MatRef};
use tsecon_ssm::{smooth_univariate, Initialization, LinearGaussianSSM};

use crate::error::NowcastError;

/// A fixed parameter vector for a single- or multi-factor dynamic factor
/// model with an order-`p` factor VAR and white-noise idiosyncratic
/// components.
///
/// The parameters live on whatever scale the panel is measured on: the
/// reference-exact golden feeds *raw* panel parameters (matching
/// statsmodels), while the two-step estimator (see [`crate::twostep`])
/// produces parameters on the *standardized* scale.
#[derive(Debug, Clone)]
pub struct DfmParams {
    /// Factor loadings `Lambda` (`N x r`): `loadings[(i, k)]` is series
    /// `i`'s loading on factor `k`.
    pub loadings: Mat<f64>,
    /// Stacked factor-VAR coefficients `[A_1 | A_2 | ... | A_p]`
    /// (`r x r p`): `factor_ar[(i, k*r + j)]` is the effect of factor `j`
    /// at lag `k+1` on factor `i`. This is the first `r` rows of the
    /// companion transition matrix.
    pub factor_ar: Mat<f64>,
    /// Factor-innovation covariance `Q` (`r x r`, symmetric PSD).
    /// statsmodels normalizes this to the identity; the two-step estimator
    /// uses the estimated innovation covariance of the PC factors.
    pub factor_cov: Mat<f64>,
    /// Idiosyncratic variances (length `N`); the diagonal of `H`.
    pub idiosyncratic: Vec<f64>,
}

impl DfmParams {
    /// Number of series `N`.
    #[inline]
    pub fn n_series(&self) -> usize {
        self.loadings.nrows()
    }

    /// Number of factors `r`.
    #[inline]
    pub fn n_factors(&self) -> usize {
        self.loadings.ncols()
    }

    /// Factor-VAR order `p`, inferred from the width of [`Self::factor_ar`].
    ///
    /// # Errors
    ///
    /// [`NowcastError::DimensionMismatch`] if the coefficient block width is
    /// not a positive multiple of `r`.
    pub fn factor_order(&self) -> Result<usize, NowcastError> {
        let r = self.n_factors();
        if r == 0 {
            return Err(NowcastError::InvalidArgument {
                what: "n_factors must be at least 1",
            });
        }
        let width = self.factor_ar.ncols();
        if width == 0 || width % r != 0 {
            return Err(NowcastError::DimensionMismatch {
                what: "factor_ar width must be a positive multiple of r",
                expected: r,
                got: width,
            });
        }
        Ok(width / r)
    }

    /// Validates the internal shapes and finiteness of the parameters.
    fn validate(&self) -> Result<usize, NowcastError> {
        let r = self.n_factors();
        let n = self.n_series();
        if n == 0 || r == 0 {
            return Err(NowcastError::InvalidArgument {
                what: "loadings must be N x r with N >= 1 and r >= 1",
            });
        }
        let p = self.factor_order()?;
        if self.factor_ar.nrows() != r {
            return Err(NowcastError::DimensionMismatch {
                what: "factor_ar must have r rows",
                expected: r,
                got: self.factor_ar.nrows(),
            });
        }
        if self.factor_cov.nrows() != r || self.factor_cov.ncols() != r {
            return Err(NowcastError::DimensionMismatch {
                what: "factor_cov must be r x r",
                expected: r,
                got: self.factor_cov.nrows(),
            });
        }
        if self.idiosyncratic.len() != n {
            return Err(NowcastError::DimensionMismatch {
                what: "idiosyncratic must have length N",
                expected: n,
                got: self.idiosyncratic.len(),
            });
        }
        for v in &self.idiosyncratic {
            if !v.is_finite() {
                return Err(NowcastError::NonFinite {
                    what: "idiosyncratic variances",
                });
            }
        }
        if !mat_is_finite(self.loadings.as_ref()) {
            return Err(NowcastError::NonFinite { what: "loadings" });
        }
        if !mat_is_finite(self.factor_ar.as_ref()) {
            return Err(NowcastError::NonFinite { what: "factor_ar" });
        }
        if !mat_is_finite(self.factor_cov.as_ref()) {
            return Err(NowcastError::NonFinite { what: "factor_cov" });
        }
        Ok(p)
    }

    /// Assembles the exact statsmodels-`DynamicFactor` state-space model
    /// (see the module docs) with stationary initialization.
    ///
    /// # Errors
    ///
    /// [`NowcastError::DimensionMismatch`] / [`NowcastError::NonFinite`] /
    /// [`NowcastError::InvalidArgument`] on malformed parameters, and
    /// [`NowcastError::Ssm`] if the state-space builder rejects the model
    /// (e.g. a non-PSD `Q`).
    pub fn state_space(&self) -> Result<LinearGaussianSSM, NowcastError> {
        let p = self.validate()?;
        let r = self.n_factors();
        let n = self.n_series();
        let m = r * p;

        // Z = [Lambda | 0].
        let loadings = &self.loadings;
        let z = Mat::from_fn(n, m, |i, j| if j < r { loadings[(i, j)] } else { 0.0 });

        // H = diag(idiosyncratic).
        let idio = &self.idiosyncratic;
        let h = Mat::from_fn(n, n, |i, j| if i == j { idio[i] } else { 0.0 });

        // T: top r rows are the stacked AR block; the sub-diagonal identity
        // blocks shift the lag stack (statsmodels layout).
        let ar = &self.factor_ar;
        let mut t = Mat::from_fn(m, m, |i, j| if i < r { ar[(i, j)] } else { 0.0 });
        // Sub-diagonal identity: state block s (>= 1) copies block s-1, i.e.
        // row i in [r, m) has a one in column i - r.
        for i in r..m {
            t[(i, i - r)] = 1.0;
        }

        // R = [I_r ; 0].
        let r_sel = Mat::from_fn(m, r, |i, j| if i < r && i == j { 1.0 } else { 0.0 });

        // Q = factor_cov.
        let q = self.factor_cov.clone();

        let model = LinearGaussianSSM::builder(n, m, r)
            .z(z)
            .h(h)
            .t(t)
            .r(r_sel)
            .q(q)
            .initialization(Initialization::Stationary)
            .build()?;
        Ok(model)
    }
}

/// The output of a fixed-parameter Kalman smoother pass.
#[derive(Debug, Clone)]
pub struct DfmSmoothing {
    /// The Gaussian (prediction-error-decomposition) log-likelihood, the
    /// quantity statsmodels reports as `results.llf`.
    pub loglik: f64,
    /// Smoothed state `alpha_hat_{t|T}` for each `t` (length `T`, each of
    /// length `m = r p`). Column-block 0 is the smoothed factor vector.
    pub smoothed_state: Vec<Vec<f64>>,
    /// Smoothed factor vectors `F_hat_{t|T}` (length `T`, each of length
    /// `r`): the first `r` entries of each smoothed state.
    pub smoothed_factors: Vec<Vec<f64>>,
}

/// Runs the Kalman filter/smoother at the fixed [`DfmParams`] over the panel
/// `data` (`T x N`, observations in rows, oldest first; NaN marks a missing
/// element and is skipped in the measurement update — the ragged edge).
///
/// This is the reference-exact path: with statsmodels' parameters on the
/// same panel it reproduces `results.llf` and `results.states.smoothed`.
///
/// # Errors
///
/// * [`NowcastError::DimensionMismatch`] if `data` has the wrong number of
///   columns or is empty;
/// * [`NowcastError::NonFinite`] if `data` holds an infinity;
/// * whatever [`DfmParams::state_space`] and the state-space smoother return
///   (e.g. a non-stationary factor VAR).
pub fn smooth_fixed(
    params: &DfmParams,
    data: MatRef<'_, f64>,
) -> Result<DfmSmoothing, NowcastError> {
    let n = params.n_series();
    let r = params.n_factors();
    if data.nrows() == 0 {
        return Err(NowcastError::EmptyInput {
            what: "panel data has no observations",
        });
    }
    if data.ncols() != n {
        return Err(NowcastError::DimensionMismatch {
            what: "panel data must have N columns",
            expected: n,
            got: data.ncols(),
        });
    }
    // Infinities are rejected here with a crate-local message; NaN is allowed
    // (it encodes a missing observation).
    for j in 0..data.ncols() {
        for i in 0..data.nrows() {
            if data[(i, j)].is_infinite() {
                return Err(NowcastError::NonFinite {
                    what: "panel data (entries must be finite or NaN-for-missing)",
                });
            }
        }
    }

    let model = params.state_space()?;
    let out = smooth_univariate(&model, data)?;

    let smoothed_state = out.smoothed_state.clone();
    let smoothed_factors = smoothed_state
        .iter()
        .map(|s| s.iter().take(r).copied().collect::<Vec<f64>>())
        .collect();

    Ok(DfmSmoothing {
        loglik: out.filter.loglik,
        smoothed_state,
        smoothed_factors,
    })
}

/// True when every entry of `m` is finite.
fn mat_is_finite(m: MatRef<'_, f64>) -> bool {
    for j in 0..m.ncols() {
        for i in 0..m.nrows() {
            if !m[(i, j)].is_finite() {
                return false;
            }
        }
    }
    true
}
