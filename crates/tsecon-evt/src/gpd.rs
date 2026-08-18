//! Peaks-over-threshold: generalized Pareto tail fitting with the
//! McNeil-Frey (2000) VaR/ES formulas.
//!
//! ## Model
//!
//! Exceedances `z = y - u > 0` over a high threshold `u` are modeled as
//! GPD(`xi`, `beta`) with density
//!
//! ```text
//! g(z) = (1 / beta) * (1 + xi z / beta)^(-1/xi - 1),   1 + xi z / beta > 0,
//! ```
//!
//! (Pickands 1975; Balkema-de Haan 1974); `xi -> 0` is the exponential
//! limit `g(z) = exp(-z / beta) / beta`. scipy's `genpareto` shape `c` *is*
//! this `xi` (verified numerically in the fixture generator).
//!
//! ## Estimation
//!
//! MLE over the working space `(xi, ln beta)` (the `tsecon-optim`
//! reparameterization pattern; the support constraint is enforced by an
//! infinite-barrier objective): moment-based starting candidates, BFGS,
//! then a tight Nelder-Mead polish. Standard errors are
//! observed-information (inverse numerical Hessian in the *original*
//! `(xi, beta)` parameterization).
//!
//! Known irregularities, reported honestly:
//!
//! * for `xi <= -0.5` the MLE is non-regular (Smith 1985) — standard
//!   errors are still reported when the Hessian permits, but
//!   [`GpdFit::se_valid`] is `false`;
//! * for samples whose true `xi <= -1` (e.g. exceedances of a uniform
//!   variable, `xi = -1`) the MLE does not exist — the likelihood grows
//!   toward the boundary `beta -> -xi * max(z)`; the fit returns the best
//!   point found with `converged` reported by the optimizer, and
//!   `se_valid = false`. The reported optimum is the standard *local
//!   interior* maximum whenever one exists (the global supremum of a GPD
//!   likelihood is always `+inf` at that boundary; every reference
//!   implementation reports the local maximum).
//!
//! ## Tail risk (McNeil-Frey 2000, section 3)
//!
//! With `n` observations, `n_u` exceedances and tail probability `p`
//! (so `P(Y <= VaR_p) = p`), the standard POT estimators are
//!
//! ```text
//! VaR_p = u + (beta / xi) * [ ((1 - p) / (n_u / n))^(-xi) - 1 ]
//! ES_p  = (VaR_p + beta - xi u) / (1 - xi),          xi < 1,
//! ```
//!
//! with the `xi = 0` limits `VaR_p = u + beta ln((n_u/n) / (1-p))`,
//! `ES_p = VaR_p + beta`. `ES` is reported NaN when `xi >= 1` (the
//! conditional mean beyond the quantile is infinite).

use tsecon_optim::{minimize, BfgsOptions, FnObjective, Method, NelderMeadOptions, ObjectiveFn};

use crate::common::{check_series, ln1p_over_xi, np_quantile_linear, observed_info_ses};
use crate::error::EvtError;

/// Minimum number of exceedances required by [`gpd_fit`].
///
/// Below this the two-parameter GPD MLE is noise; the value follows the
/// crate-level docs (a documented floor, not a statistical guarantee —
/// serious tail work wants far more).
pub const MIN_EXCEEDANCES: usize = 10;

/// Division-by-zero guard for the closed-form tail quantile formulas: at
/// `|xi|` below this the exact and limit formulas agree to machine
/// precision, so the limit branch is used.
const VAR_XI_EPS: f64 = 1e-12;

/// Result of a peaks-over-threshold GPD fit — see [`gpd_fit`].
#[derive(Debug, Clone, PartialEq)]
pub struct GpdFit {
    /// The threshold `u` actually used (supplied, or the empirical
    /// quantile of `y`).
    pub threshold: f64,
    /// The quantile the threshold corresponds to: the requested
    /// `quantile` when the threshold was chosen empirically, or the
    /// empirical fraction `P(y <= u)` when an explicit threshold was
    /// supplied — both routes report a comparable number.
    pub threshold_quantile: f64,
    /// Number of observations in `y`.
    pub n: usize,
    /// Number of strict exceedances `y > u`.
    pub n_exceed: usize,
    /// Empirical exceedance rate `n_exceed / n` — the `n_u / n` entering
    /// the McNeil-Frey formulas.
    pub exceed_rate: f64,
    /// Fitted GPD shape (tail index). Positive: heavy tail; zero:
    /// exponential; negative: bounded tail.
    pub xi: f64,
    /// Fitted GPD scale (`> 0`), in the units of `y`.
    pub beta: f64,
    /// Observed-information standard error of `xi` (NaN when the Hessian
    /// failed; see [`GpdFit::se_valid`]).
    pub se_xi: f64,
    /// Observed-information standard error of `beta` (NaN when the
    /// Hessian failed).
    pub se_beta: f64,
    /// Whether the standard errors are *certified*: the observed
    /// information was positive definite **and** `xi > -0.5` (the Smith
    /// 1985 regularity region). When `xi <= -0.5` the numbers are still
    /// reported if computable, but this flag is `false` — the asymptotics
    /// backing them do not hold.
    pub se_valid: bool,
    /// GPD log-likelihood of the exceedances at (`xi`, `beta`) —
    /// comparable to `scipy.stats.genpareto.logpdf(z, xi, 0, beta).sum()`.
    pub loglik: f64,
    /// Whether the final optimizer run reported convergence.
    pub converged: bool,
    /// The tail probabilities the VaR/ES were computed at (echoed).
    pub p_tail: Vec<f64>,
    /// POT tail quantiles `VaR_p` per entry of `p_tail`, in the units and
    /// sign convention of `y` (fit the *losses*: pass `-returns` or
    /// `|returns|` for a loss tail).
    pub var: Vec<f64>,
    /// POT expected shortfall `ES_p` per entry of `p_tail`; NaN where
    /// `xi >= 1` (infinite mean beyond the quantile).
    pub es: Vec<f64>,
}

/// Fits a GPD to the exceedances of `y` over a threshold and computes
/// McNeil-Frey tail VaR/ES.
///
/// * `threshold` — the POT threshold `u`; `None` selects the empirical
///   `quantile` of `y` (numpy `method="linear"` convention). Exceedances
///   are *strict* (`y > u`).
/// * `quantile` — the threshold quantile used when `threshold` is `None`
///   (conventional default 0.90: the top decile becomes exceedances);
///   must lie strictly inside (0, 1). Ignored when an explicit
///   `threshold` is supplied.
/// * `p_tail` — tail probabilities for VaR/ES (e.g. `[0.99, 0.995,
///   0.999]`); each must lie in (0, 1) and reach beyond the threshold
///   (`1 - p < n_exceed / n`). May be empty (no VaR/ES requested).
///
/// # Errors
///
/// [`EvtError::EmptyInput`] / [`EvtError::NonFinite`] on malformed `y` or
/// threshold; [`EvtError::InvalidQuantile`]; [`EvtError::TooFewExceedances`]
/// (fewer than [`MIN_EXCEEDANCES`] strict exceedances);
/// [`EvtError::Degenerate`] (numerically constant exceedances);
/// [`EvtError::InvalidTailProb`] / [`EvtError::TailProbNotBeyondThreshold`];
/// [`EvtError::NoAdmissibleStart`] / [`EvtError::Optim`] from the MLE.
pub fn gpd_fit(
    y: &[f64],
    threshold: Option<f64>,
    quantile: f64,
    p_tail: &[f64],
) -> Result<GpdFit, EvtError> {
    check_series(y, "y")?;
    let n = y.len();

    let (u, threshold_quantile) = match threshold {
        Some(u) => {
            if !u.is_finite() {
                return Err(EvtError::NonFinite {
                    what: "threshold",
                    index: 0,
                });
            }
            let below = y.iter().filter(|&&v| v <= u).count();
            (u, below as f64 / n as f64)
        }
        None => {
            if !(quantile > 0.0 && quantile < 1.0) {
                return Err(EvtError::InvalidQuantile { q: quantile });
            }
            let mut sorted = y.to_vec();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
            (np_quantile_linear(&sorted, quantile), quantile)
        }
    };

    let z: Vec<f64> = y.iter().filter(|&&v| v > u).map(|&v| v - u).collect();
    let n_u = z.len();
    if n_u < MIN_EXCEEDANCES {
        return Err(EvtError::TooFewExceedances {
            n_exceed: n_u,
            min: MIN_EXCEEDANCES,
            threshold: u,
        });
    }
    let z_max = z.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let z_min = z.iter().cloned().fold(f64::INFINITY, f64::min);
    if z_max - z_min <= 8.0 * f64::EPSILON * z_max {
        return Err(EvtError::Degenerate {
            what: "the exceedances over the threshold",
        });
    }

    let exceed_rate = n_u as f64 / n as f64;
    for &p in p_tail {
        if !(p > 0.0 && p < 1.0) {
            return Err(EvtError::InvalidTailProb { p });
        }
        if 1.0 - p >= exceed_rate {
            return Err(EvtError::TailProbNotBeyondThreshold { p, exceed_rate });
        }
    }

    // ---------------------------------------------- starting candidates
    let m = z.iter().sum::<f64>() / n_u as f64;
    let v = z.iter().map(|&zi| (zi - m) * (zi - m)).sum::<f64>() / (n_u as f64 - 1.0);
    // Method-of-moments shape (valid only for xi < 1/2; used as a start).
    let xi_mom = 0.5 * (1.0 - m * m / v);
    let candidates = [xi_mom.clamp(-0.4, 0.9), -0.2, 0.05, 0.2, 0.5];
    let mut start: Option<(f64, [f64; 2])> = None;
    for &xi0 in &candidates {
        // Moment-matched scale, floored proportionally to the mean so the
        // whole start map is equivariant under y -> c * y.
        let mut beta0 = m * (1.0 - xi0).max(0.1);
        if xi0 < 0.0 {
            // Ensure the support constraint 1 + xi z_max / beta > 0 holds.
            beta0 = beta0.max(-xi0 * z_max * 1.05);
        }
        let f = gpd_nll(&z, xi0, beta0);
        if f.is_finite() && start.as_ref().is_none_or(|(bf, _)| f < *bf) {
            start = Some((f, [xi0, beta0.ln()]));
        }
    }
    let (_, w0) = start.ok_or(EvtError::NoAdmissibleStart { what: "gpd_fit" })?;

    // ------------------------------------------------------------- MLE
    let (wb, converged) = refine(|w: &[f64]| gpd_nll(&z, w[0], w[1].exp()), &w0)?;
    let xi = wb[0];
    let beta = wb[1].exp();
    let loglik = -gpd_nll(&z, xi, beta);

    // -------------------------------------- observed-information SEs
    let ses = observed_info_ses(
        |p: &[f64]| gpd_nll(&z, p[0], p[1]),
        &[xi, beta],
        &[xi.abs().max(0.1), beta],
    );
    let (se_xi, se_beta) = match &ses {
        Some(s) => (s[0], s[1]),
        None => (f64::NAN, f64::NAN),
    };
    let se_valid = ses.is_some() && xi > -0.5;

    // ------------------------------------------------- McNeil-Frey tails
    let mut var = Vec::with_capacity(p_tail.len());
    let mut es = Vec::with_capacity(p_tail.len());
    for &p in p_tail {
        let (v_p, e_p) = pot_var_es(u, xi, beta, exceed_rate, p);
        var.push(v_p);
        es.push(e_p);
    }

    Ok(GpdFit {
        threshold: u,
        threshold_quantile,
        n,
        n_exceed: n_u,
        exceed_rate,
        xi,
        beta,
        se_xi,
        se_beta,
        se_valid,
        loglik,
        converged,
        p_tail: p_tail.to_vec(),
        var,
        es,
    })
}

/// GPD log-likelihood of `exceedances` at (`xi`, `beta`) — the quantity
/// [`gpd_fit`] maximizes, exposed for model comparison and testing.
///
/// Returns `-infinity` when any observation lies outside the support
/// (`z < 0`, or `1 + xi z / beta <= 0`), matching scipy's `logpdf`
/// convention.
///
/// # Errors
///
/// [`EvtError::EmptyInput`] / [`EvtError::NonFinite`] on malformed input;
/// [`EvtError::InvalidScale`] unless `beta > 0` and finite;
/// [`EvtError::NonFinite`] if `xi` is not finite.
pub fn gpd_loglik(exceedances: &[f64], xi: f64, beta: f64) -> Result<f64, EvtError> {
    check_series(exceedances, "exceedances")?;
    if !xi.is_finite() {
        return Err(EvtError::NonFinite {
            what: "xi",
            index: 0,
        });
    }
    if !(beta > 0.0 && beta.is_finite()) {
        return Err(EvtError::InvalidScale { scale: beta });
    }
    Ok(-gpd_nll(exceedances, xi, beta))
}

/// Negative GPD log-likelihood; `+infinity` outside the admissible region
/// (support violation, `beta <= 0`, non-finite parameters), which every
/// optimizer in `tsecon-optim` treats as an infeasible trial.
pub(crate) fn gpd_nll(z: &[f64], xi: f64, beta: f64) -> f64 {
    if !beta.is_finite() || beta <= 0.0 || !xi.is_finite() {
        return f64::INFINITY;
    }
    let mut acc = 0.0;
    for &zi in z {
        if zi < 0.0 {
            return f64::INFINITY;
        }
        let a = ln1p_over_xi(xi, zi / beta);
        if !a.is_finite() {
            return f64::INFINITY;
        }
        acc += a;
    }
    let nll = z.len() as f64 * beta.ln() + (1.0 + xi) * acc;
    if nll.is_finite() {
        nll
    } else {
        f64::INFINITY
    }
}

/// The McNeil-Frey (2000) POT tail quantile and expected shortfall at
/// tail probability `p` (see the module docs for the formulas).
fn pot_var_es(u: f64, xi: f64, beta: f64, exceed_rate: f64, p: f64) -> (f64, f64) {
    // ln r with r = (1 - p) / (n_u / n) < 1, so ln r < 0.
    let ln_r = ((1.0 - p) / exceed_rate).ln();
    let var = if xi.abs() < VAR_XI_EPS {
        u - beta * ln_r
    } else {
        u + beta / xi * (-xi * ln_r).exp_m1()
    };
    let es = if xi < 1.0 {
        (var + beta - xi * u) / (1.0 - xi)
    } else {
        f64::NAN
    };
    (var, es)
}

/// Shared two-stage refinement: BFGS from the start (central-difference
/// gradients — the crate's documented numerical route; 2-3 parameter
/// likelihoods make analytic gradients across the limit-branch seam more
/// error-prone than valuable), then a tight Nelder-Mead polish
/// (`x_tol = f_tol = 1e-10`, 2 restarts), keeping the best point found.
/// `converged` is the polisher's verdict.
pub(crate) fn refine<F>(f: F, w0: &[f64]) -> Result<(Vec<f64>, bool), EvtError>
where
    F: FnMut(&[f64]) -> f64,
{
    let mut obj = FnObjective::new(f);
    let mut best_x = w0.to_vec();
    let mut best_f = obj.value(&best_x);
    let mut converged = false;
    if let Ok(r) = minimize(&mut obj, &best_x, &Method::Bfgs(BfgsOptions::default())) {
        if r.f <= best_f {
            best_f = r.f;
            best_x = r.x;
            converged = r.converged;
        }
    }
    let nm = NelderMeadOptions {
        x_tol: 1e-10,
        f_tol: 1e-10,
        restarts: 2,
        ..Default::default()
    };
    let r = minimize(&mut obj, &best_x, &Method::NelderMead(nm))?;
    if r.f <= best_f {
        best_x = r.x;
        converged = r.converged;
    }
    Ok((best_x, converged))
}
