//! FVAR scenario responses: the whole-curve scenario traced through a VAR
//! in `[scores, outcome]` (Inoue & Rossi 2021's FVAR route).

use tsecon_linalg::faer::Mat;
use tsecon_var::{Trend, VarSpec};

use crate::error::FuncShockError;

/// The response paths of `[scores, outcome]` to a whole-curve scenario
/// through the estimated FVAR; produced by [`fvar_scenario`].
#[derive(Debug, Clone)]
pub struct FvarScenario {
    /// The horizons `0..=H`.
    pub horizons: Vec<usize>,
    /// `responses[h]` (length `K + 1`): the horizon-`h` responses of the
    /// `K` scores (first) and the outcome (last). At `h = 0` the score
    /// responses equal the scenario weights exactly.
    pub responses: Vec<Vec<f64>>,
    /// `response_outcome[h] = responses[h][K]` — the outcome's IRF to the
    /// whole-curve scenario.
    pub response_outcome: Vec<f64>,
    /// The scenario weights `w` the caller supplied.
    pub weights: Vec<f64>,
    /// The outcome innovation the identification implies at impact: the
    /// Cholesky regression of the outcome's reduced-form innovation on the
    /// score innovations, evaluated at `w` (equals `response_outcome[0]`).
    pub implied_outcome_innovation: f64,
}

/// Scenario response through a VAR containing `[scores, outcome]` — scores
/// ordered FIRST, outcome LAST — estimated with a constant by
/// [`tsecon_var::VarSpec::fit`] and expanded with its IRF machinery.
///
/// The scenario sets the reduced-form innovation of the score block to `w`
/// (the projection of the scenario curve onto the eigenfunctions, from
/// [`crate::scenario_weights`]) and the OUTCOME's own structural shock to
/// zero. With `Theta_h = Psi_h P` the Cholesky-orthogonalized MA
/// coefficients (`P` lower-triangular from the df-adjusted `Sigma_u`), the
/// implementation solves the forward-substitution `P[..K, ..K] z = w` and
/// reports
///
/// ```text
/// responses[h] = Theta_h[:, ..K] z ,
/// ```
///
/// which is algebraically `sum_j z_j * IRF(. <- Cholesky score shock j)`.
/// At `h = 0`, `Theta_0 = P` gives score responses exactly `w`.
///
/// **Identification caveat — read before interpreting.** This is a
/// recursive (Cholesky) identification with the scores ordered first: the
/// outcome may respond to the curve within the period, but the curve does
/// not respond to the outcome's own shock within the period. The impact
/// response of the outcome ([`FvarScenario::implied_outcome_innovation`]) is
/// therefore the in-sample regression of the outcome innovation on the score
/// innovations — a modeling assumption, not a discovery. If announcement-day
/// timing makes the curve predetermined (the Inoue-Rossi high-frequency
/// setting) this is credible; otherwise treat impact responses with care.
/// The functional local projection ([`crate::flp`]) sidesteps the VAR's
/// dynamic extrapolation but shares the same contemporaneous-exogeneity
/// question at `h = 0`.
///
/// # Errors
///
/// * [`FuncShockError::EmptyInput`] / [`FuncShockError::RaggedRow`] /
///   [`FuncShockError::NonFinite`] / [`FuncShockError::DimensionMismatch`]
///   on malformed `scores`, `y`, or `weights` (as in [`crate::flp`]);
/// * [`FuncShockError::Var`] wrapping estimation failures (too few
///   observations for `lags`, singular designs, a non-positive-definite
///   `Sigma_u`).
pub fn fvar_scenario(
    scores: &[Vec<f64>],
    y: &[f64],
    weights: &[f64],
    lags: usize,
    horizon: usize,
) -> Result<FvarScenario, FuncShockError> {
    let t = y.len();
    if t == 0 {
        return Err(FuncShockError::EmptyInput {
            what: "y (outcome)",
        });
    }
    if scores.is_empty() {
        return Err(FuncShockError::EmptyInput {
            what: "scores (T x K)",
        });
    }
    if scores.len() != t {
        return Err(FuncShockError::DimensionMismatch {
            what: "scores rows vs y: one score vector per outcome observation",
            expected: t,
            got: scores.len(),
        });
    }
    let k = scores[0].len();
    if k == 0 {
        return Err(FuncShockError::EmptyInput {
            what: "scores (each row must contain at least one score)",
        });
    }
    for (row, s) in scores.iter().enumerate() {
        if s.len() != k {
            return Err(FuncShockError::RaggedRow {
                what: "scores",
                row,
                expected: k,
                got: s.len(),
            });
        }
        if s.iter().any(|v| !v.is_finite()) {
            return Err(FuncShockError::NonFinite { what: "scores" });
        }
    }
    if y.iter().any(|v| !v.is_finite()) {
        return Err(FuncShockError::NonFinite {
            what: "y (outcome)",
        });
    }
    if weights.len() != k {
        return Err(FuncShockError::DimensionMismatch {
            what: "weights vs the number of score columns K",
            expected: k,
            got: weights.len(),
        });
    }
    if weights.iter().any(|v| !v.is_finite()) {
        return Err(FuncShockError::NonFinite {
            what: "weights (scenario projection)",
        });
    }

    // Endogenous block: scores first, outcome last (the documented ordering).
    let neqs = k + 1;
    let endog = Mat::from_fn(t, neqs, |i, j| if j < k { scores[i][j] } else { y[i] });

    let spec = VarSpec::new(lags, Trend::Constant)?;
    let results = spec.fit(endog.as_ref())?;
    let irf = results.irf(horizon)?;

    // Theta_0 = Psi_0 P = P: forward-substitute P[..K, ..K] z = w.
    let p_chol = &irf.orth_irfs[0];
    let mut z = vec![0.0_f64; k];
    for i in 0..k {
        let mut acc = weights[i];
        for (j, zj) in z.iter().enumerate().take(i) {
            acc -= p_chol[(i, j)] * zj;
        }
        // The Cholesky factor of a positive-definite Sigma_u has a strictly
        // positive diagonal (VarResults::irf already errored otherwise).
        z[i] = acc / p_chol[(i, i)];
    }

    let mut responses = Vec::with_capacity(horizon + 1);
    let mut response_outcome = Vec::with_capacity(horizon + 1);
    for theta in &irf.orth_irfs {
        let resp: Vec<f64> = (0..neqs)
            .map(|i| (0..k).map(|j| theta[(i, j)] * z[j]).sum())
            .collect();
        response_outcome.push(resp[k]);
        responses.push(resp);
    }

    let implied_outcome_innovation = response_outcome[0];
    Ok(FvarScenario {
        horizons: (0..=horizon).collect(),
        responses,
        response_outcome,
        weights: weights.to_vec(),
        implied_outcome_innovation,
    })
}
