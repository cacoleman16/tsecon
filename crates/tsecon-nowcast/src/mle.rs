//! The one-step (full-information) Gaussian maximum-likelihood estimator of
//! the single-factor dynamic factor model.
//!
//! Where the two-step Doz-Giannone-Reichlin estimator (see [`crate::twostep`])
//! extracts PCA factors, fits a factor VAR, and takes a single Kalman pass, the
//! **one-step MLE** maximises the exact Gaussian Kalman log-likelihood jointly
//! over every DFM parameter. For a single factor (`r = 1`) with an AR(`p`)
//! factor and diagonal white-noise idiosyncratic errors this is *exactly*
//! statsmodels' `DynamicFactor(k_factors=1, factor_order=p, error_order=0)`
//! model, so it is validatable against that reference
//! (`fixtures/nowcast_mle.json`, `tests/mle.rs`).
//!
//! # What is maximised
//!
//! The objective is the prediction-error-decomposition log-likelihood returned
//! by the already-validated [`smooth_fixed`] Kalman filter/smoother — no new
//! filter is written here. statsmodels does not model the panel mean, so the
//! panel is **centred** (column means removed) before fitting; the crate stores
//! that centre and re-adds it when nowcasting levels. The factor-innovation
//! variance is **fixed to 1** for identification (statsmodels' normalisation —
//! the loadings carry the factor scale), leaving `N` loadings, `p` factor-AR
//! coefficients, and `N` idiosyncratic variances free.
//!
//! # Working parameterisation
//!
//! The optimiser (tsecon-optim Nelder-Mead, then a BFGS polish) works in an
//! unconstrained space of dimension `2N + p`:
//!
//! ```text
//! z = [ loadings (N) | ar_working (p) | log_idiosyncratic (N) ]
//! ```
//!
//! * **loadings** enter directly (unbounded);
//! * **factor-AR coefficients** go through the Monahan (1984)
//!   [`StationaryAr`] PACF transform, so every trial point is a *stationary*
//!   AR(`p`) and the stationary state-space initialisation always succeeds;
//! * **idiosyncratic variances** enter as their logs, keeping them positive.
//!
//! The search starts from the two-step estimate (converted to the raw,
//! unit-innovation scale), which is a good, cheap warm start; because the MLE
//! is the maximum and is started there, its log-likelihood cannot fall below
//! the two-step value on the same centred panel.
//!
//! # Validation
//!
//! * **Reference-exact (tight).** Given statsmodels' fitted parameters,
//!   [`smooth_fixed`] reproduces statsmodels' maximised `llf` and smoothed
//!   factor on the centred panel to ~`1e-6`.
//! * **Optimum (honest gap).** `fit_mle` and statsmodels maximise the same
//!   function; the achieved llf lands within a small, reported tolerance of
//!   statsmodels' `llf`. Parameter vectors are *not* tolerance-matched (the
//!   factor is identified only up to sign, and optimisers differ); instead the
//!   crate asserts `llf(MLE) >= llf(two-step)` and, as a property, that the MLE
//!   smoothed factor tracks the simulated truth (`|corr| > 0.9`).

use tsecon_linalg::faer::{Mat, MatRef};
use tsecon_optim::{
    bfgs, nelder_mead, BfgsOptions, NelderMeadOptions, ObjectiveFn, StationaryAr, Transform,
};

use crate::error::NowcastError;
use crate::statespace::{smooth_fixed, DfmParams, DfmSmoothing};

/// A single-factor one-step MLE fit, on the centred (raw-scale) panel.
///
/// The loadings and idiosyncratic variances are on the level scale of the
/// centred panel (the loadings carry the factor scale, since the factor
/// innovation variance is fixed to 1); the factor is mean-zero.
pub(crate) struct MleFit {
    /// The fitted parameters (raw scale, factor innovation variance 1).
    pub params: DfmParams,
    /// Column means removed before fitting (length `N`).
    pub center: Vec<f64>,
    /// The smoother output at the fitted parameters over the centred panel.
    pub smoothing: DfmSmoothing,
    /// The log-likelihood of the two-step warm start on the *same* centred
    /// panel (the optimiser's initial objective). `llf(MLE) >= this` holds by
    /// construction.
    pub two_step_loglik: f64,
}

/// The negative Kalman log-likelihood as a function of the unconstrained
/// working vector `z = [loadings (N) | ar_working (p) | log_idio (N)]`, over a
/// fixed centred panel. Reuses [`smooth_fixed`] for the likelihood.
struct NegLoglik<'a> {
    /// The centred panel `Yc` (`T x N`).
    data: MatRef<'a, f64>,
    n: usize,
    p: usize,
    ar: StationaryAr,
}

impl NegLoglik<'_> {
    /// Maps a working vector to [`DfmParams`], or `None` if it leaves the
    /// feasible region (non-stationary AR, non-finite variances).
    fn params_of(&self, z: &[f64]) -> Option<DfmParams> {
        let (n, p) = (self.n, self.p);
        if z.len() != 2 * n + p {
            return None;
        }
        let loadings_slice = &z[..n];
        let ar_working = &z[n..n + p];
        let log_idio = &z[n + p..];

        // Stationary AR(p) coefficients by construction.
        let ar_nat = self.ar.forward_vec(ar_working).ok()?;
        if ar_nat.iter().any(|v| !v.is_finite()) {
            return None;
        }

        // Positive idiosyncratic variances via exp.
        let mut idiosyncratic = Vec::with_capacity(n);
        for &l in log_idio {
            let v = l.exp();
            if !v.is_finite() || v <= 0.0 {
                return None;
            }
            idiosyncratic.push(v);
        }

        if loadings_slice.iter().any(|v| !v.is_finite()) {
            return None;
        }

        Some(DfmParams {
            loadings: Mat::from_fn(n, 1, |i, _| loadings_slice[i]),
            factor_ar: Mat::from_fn(1, p, |_, j| ar_nat[j]),
            factor_cov: Mat::from_fn(1, 1, |_, _| 1.0),
            idiosyncratic,
        })
    }
}

impl ObjectiveFn for NegLoglik<'_> {
    fn value(&mut self, z: &[f64]) -> f64 {
        let Some(params) = self.params_of(z) else {
            return f64::INFINITY;
        };
        match smooth_fixed(&params, self.data) {
            Ok(sm) if sm.loglik.is_finite() => -sm.loglik,
            _ => f64::INFINITY,
        }
    }
}

/// Centres `data` (`T x N`) by its column means, returning `(centered, means)`.
fn center_columns(data: MatRef<'_, f64>) -> (Mat<f64>, Vec<f64>) {
    let t = data.nrows();
    let n = data.ncols();
    let mut means = vec![0.0; n];
    for (j, mean) in means.iter_mut().enumerate() {
        let mut s = 0.0;
        for i in 0..t {
            s += data[(i, j)];
        }
        *mean = s / t as f64;
    }
    let centered = Mat::from_fn(t, n, |i, j| data[(i, j)] - means[j]);
    (centered, means)
}

/// Builds the unconstrained warm-start working vector from a two-step fit,
/// converting the standardized two-step parameters onto the raw, unit-factor-
/// innovation scale that the MLE optimises on.
///
/// Two-step gives, on the standardized panel `Z_j = (Y_j - center_j) / sd_j`,
/// `Z_j = l_std_j g + e_j` with the PC factor `g` carrying innovation variance
/// `q`. Writing the centred raw panel `Yc_j = sd_j Z_j` and normalising the
/// factor to unit innovation variance (`f = g / sqrt(q)`, same AR coefficients)
/// gives raw loadings `sd_j l_std_j sqrt(q)`, raw idiosyncratic variances
/// `sd_j^2 idio_std_j`, and unchanged AR coefficients.
fn warm_start(two_step_params: &DfmParams, scale: &[f64], p: usize, ar: &StationaryAr) -> Vec<f64> {
    let n = two_step_params.n_series();
    let q = two_step_params.factor_cov[(0, 0)];
    let sq = if q > 0.0 && q.is_finite() {
        q.sqrt()
    } else {
        1.0
    };

    let mut z = vec![0.0; 2 * n + p];
    // Loadings block.
    for i in 0..n {
        z[i] = scale[i] * two_step_params.loadings[(i, 0)] * sq;
    }
    // AR block: map the (raw-scale, unchanged) two-step AR coefficients into
    // working space; if they are non-stationary, fall back to a zero start
    // (ar = 0), which is a valid stationary point.
    let ar_nat: Vec<f64> = (0..p).map(|j| two_step_params.factor_ar[(0, j)]).collect();
    let mut ar_working = vec![0.0; p];
    if ar.inverse(&ar_nat, &mut ar_working).is_ok() && ar_working.iter().all(|v| v.is_finite()) {
        z[n..n + p].copy_from_slice(&ar_working);
    }
    // Log-idiosyncratic block on the raw scale.
    for i in 0..n {
        let idio_raw = scale[i] * scale[i] * two_step_params.idiosyncratic[i];
        let safe = if idio_raw > 0.0 && idio_raw.is_finite() {
            idio_raw
        } else {
            1e-4
        };
        z[n + p + i] = safe.ln();
    }
    z
}

/// Fits the one-step Gaussian MLE of the single-factor DFM to the balanced
/// panel `data` (`T x N`) with an AR(`factor_order`) factor.
///
/// Centres the panel, warm-starts from the two-step estimate, then maximises
/// the exact Kalman log-likelihood with Nelder-Mead followed by a BFGS polish
/// (both from `tsecon-optim`). The factor-innovation variance is fixed to 1.
pub(crate) fn fit_mle_single_factor(
    data: MatRef<'_, f64>,
    factor_order: usize,
) -> Result<MleFit, NowcastError> {
    let t = data.nrows();
    let n = data.ncols();
    if t == 0 || n == 0 {
        return Err(NowcastError::EmptyInput {
            what: "training panel is empty",
        });
    }
    if factor_order == 0 {
        return Err(NowcastError::InvalidArgument {
            what: "factor_order must be at least 1",
        });
    }
    if t <= factor_order {
        return Err(NowcastError::InvalidArgument {
            what: "training panel must have more observations than factor_order",
        });
    }
    for j in 0..n {
        for i in 0..t {
            if !data[(i, j)].is_finite() {
                return Err(NowcastError::NonFinite {
                    what: "training panel (must be balanced and finite)",
                });
            }
        }
    }

    // Centre the panel; the MLE (like statsmodels) does not model the mean.
    let (centered, center) = center_columns(data);

    // Warm start from the two-step estimate (converted to the raw scale).
    let two_step = crate::twostep::Nowcaster::fit_two_step(data, 1, factor_order)?;
    let ar = StationaryAr;
    let z0 = warm_start(two_step.params(), two_step.scale(), factor_order, &ar);

    // The two-step warm start's log-likelihood on the same centred panel: the
    // optimiser's initial objective. llf(MLE) >= this by construction.
    let mut obj = NegLoglik {
        data: centered.as_ref(),
        n,
        p: factor_order,
        ar,
    };
    let two_step_loglik = -obj.value(&z0);
    if !two_step_loglik.is_finite() {
        return Err(NowcastError::NonFinite {
            what: "two-step warm-start log-likelihood",
        });
    }

    // --- Nelder-Mead (primary, derivative-free), then a BFGS polish. ---
    //
    // The simplex search reliably walks the (smooth, well-conditioned) Kalman
    // log-likelihood from the two-step warm start to the optimum; the BFGS
    // polish then refines the last digits and supplies a gradient-norm
    // convergence certificate. Budgets scale with the parameter dimension
    // `dim = 2N + p`; both stages treat the same objective, so the best point
    // of the two is kept.
    let dim = 2 * n + factor_order;
    let nm = nelder_mead(
        &mut obj,
        &z0,
        &NelderMeadOptions {
            x_tol: 1e-8,
            f_tol: 1e-8,
            max_iter: Some(150 * dim),
            max_fevals: Some(150 * dim),
            restarts: 1,
            adaptive: true,
            initial_step: 0.05,
        },
    )?;
    let polished = bfgs(
        &mut obj,
        &nm.x,
        &BfgsOptions {
            grad_tol: 1e-5,
            max_iter: Some(150),
            ..BfgsOptions::default()
        },
    )?;

    let (best_z, best_neg) = if polished.f <= nm.f {
        (polished.x, polished.f)
    } else {
        (nm.x, nm.f)
    };
    if !best_neg.is_finite() {
        return Err(NowcastError::NonFinite {
            what: "optimised log-likelihood",
        });
    }

    let params = obj
        .params_of(&best_z)
        .ok_or(NowcastError::InvalidArgument {
            what: "optimiser returned an infeasible parameter vector",
        })?;
    let smoothing = smooth_fixed(&params, centered.as_ref())?;

    Ok(MleFit {
        params,
        center,
        smoothing,
        two_step_loglik,
    })
}
