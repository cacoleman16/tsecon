//! Whole-curve scenario responses: project a user scenario curve onto the
//! eigenfunctions and push it through the functional local projection.

use crate::error::FuncShockError;
use crate::flp::FlpFit;
use crate::fpca::Fpca;

/// The impulse response of the outcome to a whole-curve scenario, with
/// delta-method standard errors; produced by [`scenario_response`] /
/// [`flp_scenario`].
#[derive(Debug, Clone)]
pub struct ScenarioResponse {
    /// The horizons `0..=H` (copied from the [`FlpFit`]).
    pub horizons: Vec<usize>,
    /// The scenario's weights on the eigenfunctions,
    /// `w_k = <phi_k, delta>` (length `K`).
    pub weights: Vec<f64>,
    /// `response[h] = w' beta_h` — the horizon-`h` response of the outcome
    /// to the whole-curve scenario.
    pub response: Vec<f64>,
    /// `se[h] = sqrt(w' Cov_h w)` — the delta-method standard error using
    /// the JOINT per-horizon score-coefficient covariance (the scenario is a
    /// fixed curve, so this is exact, not an approximation).
    pub se: Vec<f64>,
}

/// Projects a scenario curve `delta` (length `M`, on the same grid as the
/// curves) onto the eigenfunctions: `w_k = <phi_k, delta>` in the discrete
/// (Euclidean) inner product.
///
/// If `delta` lies in the span of the kept eigenfunctions the projection is
/// lossless; otherwise the reconstructed scenario is the best
/// rank-`K` approximation `sum_k w_k phi_k`, and the response below answers
/// for that approximation. Compare `sum_k w_k phi_k` to `delta` to see what
/// the truncation dropped.
///
/// # Errors
///
/// * [`FuncShockError::EmptyInput`] on an empty `eigenfunctions` slice;
/// * [`FuncShockError::DimensionMismatch`] if `delta`'s length differs from
///   the eigenfunction grid;
/// * [`FuncShockError::NonFinite`] on NaN/infinite entries in `delta`.
pub fn scenario_weights(
    eigenfunctions: &[Vec<f64>],
    delta: &[f64],
) -> Result<Vec<f64>, FuncShockError> {
    if eigenfunctions.is_empty() {
        return Err(FuncShockError::EmptyInput {
            what: "eigenfunctions",
        });
    }
    let m = eigenfunctions[0].len();
    if delta.len() != m {
        return Err(FuncShockError::DimensionMismatch {
            what: "scenario curve delta vs the eigenfunction grid",
            expected: m,
            got: delta.len(),
        });
    }
    if delta.iter().any(|v| !v.is_finite()) {
        return Err(FuncShockError::NonFinite {
            what: "delta (scenario curve)",
        });
    }
    Ok(eigenfunctions
        .iter()
        .map(|phi| phi.iter().zip(delta.iter()).map(|(p, d)| p * d).sum())
        .collect())
}

/// The impulse response of the outcome to a whole-curve scenario with
/// weights `w` on the eigenfunctions: `response_h = w' beta_h` with variance
/// `w' Cov_h w` from the JOINT per-horizon covariance in `fit`.
///
/// This is the deliverable that makes the method functional: the IRF of the
/// outcome to the entire curve moving by `delta = sum_k w_k phi_k`, standard
/// errors included.
///
/// # Errors
///
/// * [`FuncShockError::DimensionMismatch`] if `weights.len() != K`;
/// * [`FuncShockError::NonFinite`] on NaN/infinite weights;
/// * [`FuncShockError::NegativeVariance`] if `w' Cov_h w` is materially
///   negative (a non-PSD covariance was supplied); tiny negatives from
///   floating-point roundoff (`> -1e-10 * scale`) are clamped to zero.
pub fn scenario_response(
    fit: &FlpFit,
    weights: &[f64],
) -> Result<ScenarioResponse, FuncShockError> {
    let k = fit.n_factors;
    if weights.len() != k {
        return Err(FuncShockError::DimensionMismatch {
            what: "weights vs the fitted number of score regressors K",
            expected: k,
            got: weights.len(),
        });
    }
    if weights.iter().any(|v| !v.is_finite()) {
        return Err(FuncShockError::NonFinite {
            what: "weights (scenario projection)",
        });
    }

    let mut response = Vec::with_capacity(fit.betas.len());
    let mut se = Vec::with_capacity(fit.betas.len());
    for (h, (beta, cov)) in fit.betas.iter().zip(fit.covs.iter()).enumerate() {
        let r: f64 = weights.iter().zip(beta.iter()).map(|(w, b)| w * b).sum();
        let mut var = 0.0_f64;
        for i in 0..k {
            for j in 0..k {
                var += weights[i] * cov[i * k + j] * weights[j];
            }
        }
        // Bartlett-HAC covariances are PSD; anything materially negative
        // means the caller supplied a broken covariance.
        let scale: f64 = 1.0 + (0..k).map(|i| cov[i * k + i].abs()).sum::<f64>();
        if var < -1e-10 * scale {
            return Err(FuncShockError::NegativeVariance {
                horizon: fit.horizons.get(h).copied().unwrap_or(h),
                value: var,
            });
        }
        response.push(r);
        se.push(var.max(0.0).sqrt());
    }

    Ok(ScenarioResponse {
        horizons: fit.horizons.clone(),
        weights: weights.to_vec(),
        response,
        se,
    })
}

/// Convenience composition: project the scenario curve `delta` onto
/// `fpca`'s eigenfunctions ([`scenario_weights`]) and push the weights
/// through the fitted functional local projection
/// ([`scenario_response`]).
///
/// # Errors
///
/// Everything [`scenario_weights`] and [`scenario_response`] can return; in
/// particular [`FuncShockError::DimensionMismatch`] if `fit` was estimated
/// with a different number of scores than `fpca` kept.
pub fn flp_scenario(
    fpca: &Fpca,
    fit: &FlpFit,
    delta: &[f64],
) -> Result<ScenarioResponse, FuncShockError> {
    let weights = scenario_weights(&fpca.eigenfunctions, delta)?;
    scenario_response(fit, &weights)
}
