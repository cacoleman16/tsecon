//! Copula fitting (`copula_fit`), family selection (`copula_select`),
//! and the pseudo-observation transform (`pseudo_obs`).
//!
//! ## Estimation
//!
//! * `method = Tau` — **Kendall-tau inversion** (the method-of-moments
//!   route statsmodels' `fit_corr_param` implements): the empirical
//!   tau-b is pushed through the family's closed-form inverse map. For
//!   the t family, tau identifies `rho` only; `nu` is then *profiled*
//!   by MLE at that fixed `rho` (statsmodels offers no `nu` estimate at
//!   all — stated honestly). No standard errors are reported for this
//!   method in this slice (the method-of-moments delta method around
//!   tau's asymptotic variance is deferred): `se` is NaN and `se_valid`
//!   is false.
//! * `method = Mle` — **maximum likelihood** on the copula density,
//!   started from the tau inversion, optimized in an unconstrained
//!   working space (`atanh rho` / `ln nu` / `ln theta` /
//!   `ln(theta - 1)` / raw Frank `theta`) by BFGS with a tight
//!   Nelder-Mead polish — the `tsecon-optim` pattern shared with
//!   `tsecon-evt`. Standard errors are observed-information (inverse
//!   numerical Hessian in the *original* parameterization, unit-safe
//!   steps mirrored by the fixture generator). When the Hessian is not
//!   positive definite — e.g. the t family's `nu` drifting toward its
//!   Gaussian-limit boundary on near-Gaussian data — `se` is NaN and
//!   `se_valid` is false rather than fabricated.
//!
//! Both methods report the same result shape: parameters, log-likelihood,
//! AIC/BIC (`-2 ll + 2k` / `-2 ll + k ln n`), the empirical and implied
//! Kendall tau, and the closed-form tail-dependence pair.

use tsecon_optim::{minimize, BfgsOptions, FnObjective, Method, NelderMeadOptions, ObjectiveFn};

use crate::common::{check_series, check_u, kendall_tau, observed_info_ses, pseudo_obs_col};
use crate::error::CopulaError;
use crate::family::{logpdf_unchecked, param_to_tau, tail_dependence, tau_to_param, Family};

/// Barrier bounds on the t family's degrees of freedom during MLE
/// (mirrored by the fixture generator). At the upper bound the t copula
/// is numerically Gaussian and the likelihood is flat in `nu` — the fit
/// still returns, with uncertified standard errors.
pub const NU_BOUNDS: (f64, f64) = (0.1, 1000.0);

/// Barrier bound on `|theta|` for the Frank family during MLE (mirrored
/// by the fixture generator; well inside the `exp` overflow guard of the
/// evaluators).
pub const FRANK_MLE_THETA_MAX: f64 = 500.0;

/// Tolerance treating the empirical tau as perfectly (anti)monotone.
const TAU_BOUNDARY: f64 = 1e-10;

/// How the dependence parameters were estimated — see the module docs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FitMethod {
    /// Full maximum likelihood on the copula density.
    Mle,
    /// Kendall-tau inversion (t family: `rho` from tau, `nu` profiled).
    Tau,
}

impl FitMethod {
    /// Parse a method name (the Python-surface strings).
    pub fn parse(name: &str) -> Result<Self, CopulaError> {
        match name.to_ascii_lowercase().as_str() {
            "mle" => Ok(Self::Mle),
            "tau" => Ok(Self::Tau),
            _ => Err(CopulaError::UnknownMethod {
                name: name.to_string(),
            }),
        }
    }

    /// Canonical lowercase name.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Mle => "mle",
            Self::Tau => "tau",
        }
    }
}

/// Result of a bivariate copula fit — see [`copula_fit`].
#[derive(Debug, Clone, PartialEq)]
pub struct CopulaFit {
    /// The fitted family.
    pub family: Family,
    /// The estimation method actually used.
    pub method: FitMethod,
    /// Number of paired observations.
    pub n: usize,
    /// Fitted dependence parameters, in [`Family::param_names`] order
    /// (`[rho]`, `[rho, nu]`, or `[theta]`).
    pub params: Vec<f64>,
    /// Observed-information standard errors per parameter (NaN for
    /// `method = Tau`, or when the Hessian failed — see
    /// [`CopulaFit::se_valid`]).
    pub se: Vec<f64>,
    /// Whether the standard errors are certified: MLE with a positive
    /// definite observed information. Always false for `method = Tau`.
    pub se_valid: bool,
    /// Copula log-likelihood at the fitted parameters.
    pub loglik: f64,
    /// Akaike information criterion `-2 loglik + 2 k`.
    pub aic: f64,
    /// Bayesian information criterion `-2 loglik + k ln n`.
    pub bic: f64,
    /// Empirical Kendall tau-b of the data.
    pub tau: f64,
    /// Kendall tau implied by the fitted parameters (closed form).
    pub tau_implied: f64,
    /// Closed-form lower tail-dependence coefficient at the fit.
    pub tail_lower: f64,
    /// Closed-form upper tail-dependence coefficient at the fit.
    pub tail_upper: f64,
    /// Whether the final optimizer run reported convergence (true by
    /// construction for closed-form tau inversions).
    pub converged: bool,
}

/// The average-rank pseudo-observation transform, columnwise:
/// `u = rank / (n + 1)` with average ranks on ties — exactly
/// `scipy.stats.rankdata(x, method="average") / (n + 1)`. This is the
/// one-line companion to [`copula_fit`]: ranks depend only on order, so
/// the result — and any copula fitted to it — is invariant to strictly
/// monotone transforms of each margin (the point of the copula
/// decomposition; property-tested).
///
/// # Errors
///
/// [`CopulaError::EmptyInput`] (no columns, or an empty column);
/// [`CopulaError::NonFinite`]; [`CopulaError::LengthMismatch`];
/// [`CopulaError::TooFewObservations`] (fewer than 2 rows).
pub fn pseudo_obs(columns: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, CopulaError> {
    if columns.is_empty() {
        return Err(CopulaError::EmptyInput { what: "x" });
    }
    let n = columns[0].len();
    for col in columns {
        check_series(col, "x")?;
        if col.len() != n {
            return Err(CopulaError::LengthMismatch {
                n1: n,
                n2: col.len(),
            });
        }
    }
    if n < 2 {
        return Err(CopulaError::TooFewObservations { n, min: 2 });
    }
    Ok(columns.iter().map(|c| pseudo_obs_col(c)).collect())
}

/// Fits a bivariate copula to probability-scale pseudo-observations.
///
/// `u1`, `u2` must already be on the probability scale, strictly inside
/// `(0, 1)` — rank/PIT-transform the raw margins first ([`pseudo_obs`]
/// does this in one call). At least [`crate::MIN_OBS`] pairs are
/// required.
///
/// See the module docs for the two methods; the [`CopulaFit`] result
/// carries parameters, SEs, loglik/AIC/BIC, empirical and implied
/// Kendall tau, and the family's closed-form tail-dependence pair.
///
/// # Errors
///
/// Malformed `u` ([`CopulaError::EmptyInput`] / [`CopulaError::NonFinite`]
/// / [`CopulaError::LengthMismatch`] / [`CopulaError::TooFewObservations`]
/// / [`CopulaError::OutOfUnitInterval`]);
/// [`CopulaError::Degenerate`] (a constant column);
/// [`CopulaError::PerfectDependence`] (|tau| numerically 1);
/// [`CopulaError::NegativeDependence`] (Clayton/Gumbel with tau <= 0);
/// [`CopulaError::NoAdmissibleStart`] / [`CopulaError::Optim`] from the
/// MLE.
pub fn copula_fit(
    u1: &[f64],
    u2: &[f64],
    family: Family,
    method: FitMethod,
) -> Result<CopulaFit, CopulaError> {
    check_u(u1, u2)?;
    let n = u1.len();
    let tau = kendall_tau(u1, u2)?;
    if tau.abs() >= 1.0 - TAU_BOUNDARY {
        return Err(CopulaError::PerfectDependence { tau });
    }
    if matches!(family, Family::Clayton | Family::Gumbel) && tau <= 0.0 {
        return Err(CopulaError::NegativeDependence {
            family: family.name(),
            tau,
        });
    }

    let (params, se, se_valid, converged) = match method {
        FitMethod::Tau => fit_by_tau(u1, u2, family, tau)?,
        FitMethod::Mle => fit_by_mle(u1, u2, family, tau)?,
    };

    let loglik: f64 = logpdf_unchecked(u1, u2, family, &params)?.iter().sum();
    let k = params.len() as f64;
    let tau_implied = param_to_tau(family, &params)?;
    let (tail_lower, tail_upper) = tail_dependence(family, &params)?;
    Ok(CopulaFit {
        family,
        method,
        n,
        params,
        se,
        se_valid,
        loglik,
        aic: -2.0 * loglik + 2.0 * k,
        bic: -2.0 * loglik + k * (n as f64).ln(),
        tau,
        tau_implied,
        tail_lower,
        tail_upper,
        converged,
    })
}

/// Negative copula log-likelihood at `params`, `+infinity` outside the
/// family's domain / MLE barriers or on any non-finite density value —
/// the infinite-barrier objective every optimizer in `tsecon-optim`
/// treats as an infeasible trial. Barriers mirror the fixture
/// generator: t's `nu` in [`NU_BOUNDS`], Frank's `|theta|` at most
/// [`FRANK_MLE_THETA_MAX`].
fn nll(u1: &[f64], u2: &[f64], family: Family, params: &[f64]) -> f64 {
    let ok = match family {
        Family::Gaussian => params[0] > -1.0 && params[0] < 1.0,
        Family::StudentT => {
            params[0] > -1.0
                && params[0] < 1.0
                && params[1] > NU_BOUNDS.0
                && params[1] < NU_BOUNDS.1
        }
        Family::Clayton => params[0] > 0.0,
        Family::Gumbel => params[0] > 1.0,
        Family::Frank => params[0] != 0.0 && params[0].abs() <= FRANK_MLE_THETA_MAX,
    };
    if !ok || params.iter().any(|p| !p.is_finite()) {
        return f64::INFINITY;
    }
    match logpdf_unchecked(u1, u2, family, params) {
        Ok(lp) => {
            let s: f64 = lp.iter().sum();
            if s.is_finite() {
                -s
            } else {
                f64::INFINITY
            }
        }
        Err(_) => f64::INFINITY,
    }
}

/// Kendall-tau inversion (+ profiled `nu` for the t family).
fn fit_by_tau(
    u1: &[f64],
    u2: &[f64],
    family: Family,
    tau: f64,
) -> Result<(Vec<f64>, Vec<f64>, bool, bool), CopulaError> {
    let dep = tau_to_param(family, tau)?;
    let (params, converged) = match family {
        Family::StudentT => {
            let (nu, converged) = profile_nu(u1, u2, dep)?;
            (vec![dep, nu], converged)
        }
        _ => (vec![dep], true),
    };
    let k = params.len();
    Ok((params, vec![f64::NAN; k], false, converged))
}

/// Profile MLE of the t family's `nu` at fixed `rho` (best-of starting
/// candidates, then the shared BFGS + Nelder-Mead refinement over
/// `ln nu`).
fn profile_nu(u1: &[f64], u2: &[f64], rho: f64) -> Result<(f64, bool), CopulaError> {
    let mut start: Option<(f64, f64)> = None;
    for &nu0 in &[2.5, 5.0, 10.0, 20.0, 50.0] {
        let f = nll(u1, u2, Family::StudentT, &[rho, nu0]);
        if f.is_finite() && start.as_ref().is_none_or(|(bf, _)| f < *bf) {
            start = Some((f, nu0));
        }
    }
    let (_, nu0) = start.ok_or(CopulaError::NoAdmissibleStart {
        what: "the t-copula nu profile",
    })?;
    let (wb, converged) = refine(
        |w: &[f64]| nll(u1, u2, Family::StudentT, &[rho, w[0].exp()]),
        &[nu0.ln()],
    )?;
    Ok((wb[0].exp(), converged))
}

/// Full MLE: tau-inverted start, unconstrained working space, BFGS +
/// Nelder-Mead polish, observed-information SEs in the original space.
fn fit_by_mle(
    u1: &[f64],
    u2: &[f64],
    family: Family,
    tau: f64,
) -> Result<(Vec<f64>, Vec<f64>, bool, bool), CopulaError> {
    // Starting point (mirrors the fixture generator's construction).
    let rho_tau = (core::f64::consts::PI * tau / 2.0).sin().clamp(-0.99, 0.99);
    let w0: Vec<f64> = match family {
        Family::Gaussian => vec![rho_tau.atanh()],
        Family::StudentT => {
            let mut start: Option<(f64, f64)> = None;
            for &nu0 in &[2.5, 5.0, 10.0, 20.0, 50.0] {
                let f = nll(u1, u2, family, &[rho_tau, nu0]);
                if f.is_finite() && start.as_ref().is_none_or(|(bf, _)| f < *bf) {
                    start = Some((f, nu0));
                }
            }
            let (_, nu0) = start.ok_or(CopulaError::NoAdmissibleStart {
                what: "the t-copula MLE",
            })?;
            vec![rho_tau.atanh(), nu0.ln()]
        }
        Family::Clayton => vec![(2.0 * tau / (1.0 - tau)).max(0.05).ln()],
        Family::Gumbel => vec![(1.0 / (1.0 - tau) - 1.0).max(1e-3).ln()],
        Family::Frank => {
            let th0 = if tau.abs() > 1e-8 {
                tau_to_param(Family::Frank, tau)?
            } else {
                0.5
            };
            vec![th0]
        }
    };
    let to_params = |w: &[f64]| -> Vec<f64> {
        match family {
            Family::Gaussian => vec![w[0].tanh()],
            Family::StudentT => vec![w[0].tanh(), w[1].exp()],
            Family::Clayton => vec![w[0].exp()],
            Family::Gumbel => vec![w[0].exp() + 1.0],
            Family::Frank => vec![w[0]],
        }
    };
    if !nll(u1, u2, family, &to_params(&w0)).is_finite() {
        return Err(CopulaError::NoAdmissibleStart {
            what: "copula_fit MLE",
        });
    }
    let (wb, converged) = refine(|w: &[f64]| nll(u1, u2, family, &to_params(w)), &w0)?;
    let params = to_params(&wb);

    // Observed-information SEs in the original parameterization, with
    // unit-safe step scales mirrored by the fixture generator.
    let scales: Vec<f64> = match family {
        Family::Gaussian => vec![(1.0 - params[0] * params[0]).max(0.01)],
        Family::StudentT => vec![(1.0 - params[0] * params[0]).max(0.01), params[1]],
        _ => vec![params[0].abs().max(0.1)],
    };
    let ses = observed_info_ses(|p: &[f64]| nll(u1, u2, family, p), &params, &scales);
    let k = params.len();
    let (se, se_valid) = match ses {
        Some(s) => (s, true),
        None => (vec![f64::NAN; k], false),
    };
    Ok((params, se, se_valid, converged))
}

/// Shared two-stage refinement: BFGS from the start (central-difference
/// gradients — the optimizer crate's documented numerical route), then a
/// tight Nelder-Mead polish (`x_tol = f_tol = 1e-10`, 2 restarts),
/// keeping the best point found. `converged` is the polisher's verdict.
fn refine<F>(f: F, w0: &[f64]) -> Result<(Vec<f64>, bool), CopulaError>
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

// ---------------------------------------------------------------------------
// Selection
// ---------------------------------------------------------------------------

/// A family `copula_select` could not fit, with the teaching reason.
#[derive(Debug, Clone, PartialEq)]
pub struct SkippedFamily {
    /// The family that was skipped.
    pub family: Family,
    /// Why (the full teaching error message).
    pub reason: String,
}

/// Result of [`copula_select`].
#[derive(Debug, Clone, PartialEq)]
pub struct CopulaSelect {
    /// One fit per successfully fitted family, in request order.
    pub fits: Vec<CopulaFit>,
    /// Families skipped because their domain excludes this data
    /// (currently: Clayton/Gumbel under non-positive Kendall tau).
    pub skipped: Vec<SkippedFamily>,
    /// Index into `fits` of the AIC winner.
    pub best_aic: usize,
    /// Index into `fits` of the BIC winner.
    pub best_bic: usize,
    /// Fit indices sorted best-first by AIC.
    pub ranking_aic: Vec<usize>,
    /// Fit indices sorted best-first by BIC.
    pub ranking_bic: Vec<usize>,
    /// The teaching verdict — which family wins, by how much, whether
    /// AIC and BIC agree, what the winner implies for tail dependence,
    /// and what was skipped and why.
    pub verdict: String,
}

/// Fits every requested family to the same pseudo-observations and ranks
/// the fits by AIC and BIC, with a teaching verdict.
///
/// Families whose domain excludes the data (Clayton/Gumbel under
/// negative dependence) are *skipped with a reason*, not failed, so a
/// standard menu works on any data; every other error propagates.
///
/// # Errors
///
/// [`CopulaError::EmptyFamilies`] / [`CopulaError::DuplicateFamily`] on a
/// malformed menu; [`CopulaError::AllFamiliesSkipped`] when nothing could
/// be fitted; otherwise as [`copula_fit`].
pub fn copula_select(
    u1: &[f64],
    u2: &[f64],
    families: &[Family],
    method: FitMethod,
) -> Result<CopulaSelect, CopulaError> {
    if families.is_empty() {
        return Err(CopulaError::EmptyFamilies);
    }
    for (i, f) in families.iter().enumerate() {
        if families[..i].contains(f) {
            return Err(CopulaError::DuplicateFamily { family: f.name() });
        }
    }
    let mut fits = Vec::new();
    let mut skipped = Vec::new();
    for &family in families {
        match copula_fit(u1, u2, family, method) {
            Ok(fit) => fits.push(fit),
            Err(e @ CopulaError::NegativeDependence { .. }) => skipped.push(SkippedFamily {
                family,
                reason: e.to_string(),
            }),
            Err(e) => return Err(e),
        }
    }
    if fits.is_empty() {
        let reasons = skipped
            .iter()
            .map(|s| s.reason.clone())
            .collect::<Vec<_>>()
            .join("; ");
        return Err(CopulaError::AllFamiliesSkipped { reasons });
    }
    let mut ranking_aic: Vec<usize> = (0..fits.len()).collect();
    ranking_aic.sort_by(|&a, &b| {
        fits[a]
            .aic
            .partial_cmp(&fits[b].aic)
            .unwrap_or(core::cmp::Ordering::Equal)
    });
    let mut ranking_bic: Vec<usize> = (0..fits.len()).collect();
    ranking_bic.sort_by(|&a, &b| {
        fits[a]
            .bic
            .partial_cmp(&fits[b].bic)
            .unwrap_or(core::cmp::Ordering::Equal)
    });
    let best_aic = ranking_aic[0];
    let best_bic = ranking_bic[0];
    let verdict = build_verdict(&fits, &skipped, &ranking_aic, best_aic, best_bic);
    Ok(CopulaSelect {
        fits,
        skipped,
        best_aic,
        best_bic,
        ranking_aic,
        ranking_bic,
        verdict,
    })
}

fn build_verdict(
    fits: &[CopulaFit],
    skipped: &[SkippedFamily],
    ranking_aic: &[usize],
    best_aic: usize,
    best_bic: usize,
) -> String {
    let best = &fits[best_aic];
    let mut v = format!("{} minimizes AIC ({:.2})", best.family.name(), best.aic);
    if ranking_aic.len() > 1 {
        let runner = &fits[ranking_aic[1]];
        let delta = runner.aic - best.aic;
        v.push_str(&format!(
            ", dAIC {:.2} over the runner-up {}",
            delta,
            runner.family.name()
        ));
        if delta < 2.0 {
            v.push_str(
                " — within 2 AIC the fits are statistically \
                 near-indistinguishable; prefer the family whose tail \
                 behavior matches the economics",
            );
        }
    }
    if best_bic == best_aic {
        v.push_str("; BIC agrees");
    } else {
        v.push_str(&format!(
            "; BIC instead selects {} (BIC charges more for extra parameters)",
            fits[best_bic].family.name()
        ));
    }
    if best.tail_lower == 0.0 && best.tail_upper == 0.0 {
        v.push_str(&format!(
            ". The winner implies Kendall tau {:.3} and NO tail dependence \
             — joint extremes are modeled as asymptotically independent \
             (the classic reason a Gaussian/Frank fit understates joint \
             crashes; if tails matter, compare against t/Clayton/Gumbel)",
            best.tau_implied
        ));
    } else {
        v.push_str(&format!(
            ". The winner implies Kendall tau {:.3} and lower/upper tail \
             dependence {:.3}/{:.3} — joint extremes stay dependent in \
             the limit",
            best.tau_implied, best.tail_lower, best.tail_upper
        ));
    }
    if !skipped.is_empty() {
        let names: Vec<&str> = skipped.iter().map(|s| s.family.name()).collect();
        v.push_str(&format!(
            ". Skipped {}: Kendall tau <= 0 and these families model \
             positive dependence only (rotations are deferred in this \
             slice)",
            names.join(", ")
        ));
    }
    v.push('.');
    v
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;
    use crate::common::MIN_OBS;

    #[test]
    fn method_and_family_parse() {
        assert_eq!(FitMethod::parse("MLE").expect("mle"), FitMethod::Mle);
        assert_eq!(FitMethod::parse("tau").expect("tau"), FitMethod::Tau);
        assert!(matches!(
            FitMethod::parse("moments"),
            Err(CopulaError::UnknownMethod { .. })
        ));
        assert_eq!(Family::parse("Student-T").expect("t"), Family::StudentT);
    }

    #[test]
    fn min_obs_is_enforced() {
        let u1: Vec<f64> = (0..MIN_OBS - 1).map(|i| (i as f64 + 0.5) / 20.0).collect();
        let u2 = u1.clone();
        assert!(matches!(
            copula_fit(&u1, &u2, Family::Gaussian, FitMethod::Tau),
            Err(CopulaError::TooFewObservations { .. })
        ));
    }
}
