//! Automatic model selection over the ETS taxonomy — the candidate-set
//! search of Hyndman et al. (2008, section 7.2) as R's `forecast::ets`
//! runs it: fit every admissible member of the taxonomy by maximum
//! likelihood and keep the one with the smallest information criterion
//! (AICc by default).
//!
//! The candidate set and its admissibility restrictions are R's:
//!
//! * error `A`/`M`, trend `N`/`A`(`d`)/(`M`(`d`) only with
//!   `allow_multiplicative_trend`), seasonal `N`/`A`/`M` (seasonal
//!   candidates only with `seasonal_periods >= 2`);
//! * multiplicative error or seasonal only on strictly positive data;
//! * `restrict` (R's default `TRUE`) drops the combinations with
//!   infinite forecast variance or a mis-specified error scale: additive
//!   error with any multiplicative component (`A,M,*`, `A,*,M`) and
//!   `M,M,A`;
//! * `damped = None` tries both damped and undamped trends.
//!
//! Every candidate's fit is recorded (specification, log-likelihood, the
//! three criteria, convergence, or the error that stopped it) so the
//! selection is auditable, and the winner is refit-free: it is the very
//! fit the search scored.

use crate::error::EtsError;
use crate::fit::{ets_fit, EtsFit, FitOptions, Initialization, Optimizer};
use crate::spec::{Component, ErrorType, EtsSpec};

/// The selection criterion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ic {
    /// Corrected Akaike (the default; Hyndman-Khandakar 2008).
    Aicc,
    /// Akaike.
    Aic,
    /// Schwarz / Bayesian.
    Bic,
}

impl Ic {
    /// The name reported in results.
    pub fn name(self) -> &'static str {
        match self {
            Ic::Aicc => "aicc",
            Ic::Aic => "aic",
            Ic::Bic => "bic",
        }
    }

    fn of(self, fit: &EtsFit) -> f64 {
        match self {
            Ic::Aicc => fit.aicc,
            Ic::Aic => fit.aic,
            Ic::Bic => fit.bic,
        }
    }
}

/// Options of [`auto_ets`].
#[derive(Debug, Clone, PartialEq)]
pub struct AutoEtsOptions {
    /// Selection criterion (default AICc).
    pub ic: Ic,
    /// Seasonal period (`None` or `Some(1)`: non-seasonal candidates
    /// only).
    pub seasonal_periods: Option<usize>,
    /// Include multiplicative (and damped multiplicative) trends (default
    /// `false`, R's `allow.multiplicative.trend`).
    pub allow_multiplicative_trend: bool,
    /// Apply R's admissibility restrictions (default `true`).
    pub restrict: bool,
    /// `None`: try damped and undamped trends; `Some(d)`: only that.
    pub damped: Option<bool>,
    /// Initial-state treatment for every candidate (estimated or
    /// heuristic).
    pub initialization: Initialization,
    /// Numerical search for every candidate.
    pub optimizer: Optimizer,
}

impl Default for AutoEtsOptions {
    fn default() -> Self {
        Self {
            ic: Ic::Aicc,
            seasonal_periods: None,
            allow_multiplicative_trend: false,
            restrict: true,
            damped: None,
            initialization: Initialization::Estimated,
            optimizer: Optimizer::TwoStage,
        }
    }
}

/// One scored candidate.
#[derive(Debug, Clone, PartialEq)]
pub struct Candidate {
    /// The specification.
    pub spec: EtsSpec,
    /// Its compact name (`AAdN`, ...).
    pub short_name: String,
    /// Log-likelihood (NaN when the fit failed).
    pub loglik: f64,
    /// AIC (NaN when the fit failed).
    pub aic: f64,
    /// AICc.
    pub aicc: f64,
    /// BIC.
    pub bic: f64,
    /// The selection criterion's value (`+inf` when the fit failed, so
    /// failures sort last).
    pub ic_value: f64,
    /// Parameters counted by the criteria.
    pub k_params: usize,
    /// Whether the candidate's search converged.
    pub converged: bool,
    /// `"ok"` or `"error"`.
    pub status: &'static str,
    /// The error message when the fit failed.
    pub error: Option<String>,
}

/// The outcome of [`auto_ets`].
#[derive(Debug, Clone, PartialEq)]
pub struct AutoEtsResult {
    /// The selected model's fit (the search's own fit of it, not a refit).
    pub best: EtsFit,
    /// Every candidate, ranked by the criterion (failures last, ties in
    /// enumeration order).
    pub candidates: Vec<Candidate>,
    /// The criterion used.
    pub ic: Ic,
    /// Number of candidates enumerated.
    pub n_candidates: usize,
    /// Number that fitted successfully.
    pub n_fitted: usize,
}

/// Enumerates the admissible candidate specifications for the options
/// and the data's sign (R's loop order: error, trend, seasonal, damped
/// with `TRUE` first).
pub fn candidate_specs(opts: &AutoEtsOptions, data_positive: bool) -> Vec<EtsSpec> {
    let errors = [ErrorType::Additive, ErrorType::Multiplicative];
    let trends: Vec<Component> = if opts.allow_multiplicative_trend {
        vec![
            Component::None,
            Component::Additive,
            Component::Multiplicative,
        ]
    } else {
        vec![Component::None, Component::Additive]
    };
    let m = opts.seasonal_periods.unwrap_or(1);
    let seasonals: Vec<Component> = if m >= 2 {
        vec![
            Component::None,
            Component::Additive,
            Component::Multiplicative,
        ]
    } else {
        vec![Component::None]
    };
    let dampeds: Vec<bool> = match opts.damped {
        None => vec![true, false],
        Some(d) => vec![d],
    };
    let mut out = Vec::new();
    for &error in &errors {
        for &trend in &trends {
            for &seasonal in &seasonals {
                for &damped in &dampeds {
                    if damped && !trend.is_present() {
                        continue;
                    }
                    if opts.restrict {
                        if error == ErrorType::Additive
                            && (trend == Component::Multiplicative
                                || seasonal == Component::Multiplicative)
                        {
                            continue;
                        }
                        if error == ErrorType::Multiplicative
                            && trend == Component::Multiplicative
                            && seasonal == Component::Additive
                        {
                            continue;
                        }
                    }
                    let spec = EtsSpec {
                        error,
                        trend,
                        damped,
                        seasonal,
                        seasonal_periods: if seasonal.is_present() { m } else { 1 },
                    };
                    if !data_positive && spec.needs_positive_data() {
                        continue;
                    }
                    out.push(spec);
                }
            }
        }
    }
    out
}

/// Selects and fits the best ETS model for `y` (see the module docs).
///
/// # Errors
///
/// * [`EtsError::NonFinite`] for data with NaN/inf;
/// * [`EtsError::InvalidOption`] for `seasonal_periods = 0` or a `Known`
///   initialisation (the states differ per candidate; use `Estimated` or
///   `Heuristic`);
/// * [`EtsError::NoCandidates`] if no candidate is admissible or every
///   candidate failed to fit (each failure is reported in the message).
pub fn auto_ets(y: &[f64], opts: &AutoEtsOptions) -> Result<AutoEtsResult, EtsError> {
    if let Some((i, &v)) = y.iter().enumerate().find(|(_, v)| !v.is_finite()) {
        return Err(EtsError::NonFinite {
            what: "y",
            index: i,
            value: v,
        });
    }
    if opts.seasonal_periods == Some(0) {
        return Err(EtsError::InvalidOption {
            name: "seasonal_periods",
            value: 0.0,
            requirement: "the seasonal period must be at least 2 (or None for a \
                          non-seasonal search)",
        });
    }
    if matches!(opts.initialization, Initialization::Known(_)) {
        return Err(EtsError::InvalidOption {
            name: "initialization",
            value: f64::NAN,
            requirement: "auto_ets cannot use known initial states (each candidate has \
                          its own state vector); pass Estimated or Heuristic",
        });
    }
    let data_positive = !y.is_empty() && y.iter().all(|&v| v > 0.0);
    let specs = candidate_specs(opts, data_positive);
    if specs.is_empty() {
        return Err(EtsError::NoCandidates {
            what: "the auto_ets restrictions left no candidate model; relax restrict, \
                   allow_multiplicative_trend or damped"
                .into(),
        });
    }
    let fit_opts = FitOptions {
        initialization: opts.initialization.clone(),
        optimizer: opts.optimizer,
        max_iter: None,
    };
    let mut scored: Vec<(Candidate, Option<EtsFit>)> = Vec::with_capacity(specs.len());
    for spec in specs {
        match ets_fit(&spec, y, &fit_opts) {
            Ok(fit) => {
                let cand = Candidate {
                    spec,
                    short_name: spec.short_name(),
                    loglik: fit.loglik,
                    aic: fit.aic,
                    aicc: fit.aicc,
                    bic: fit.bic,
                    ic_value: opts.ic.of(&fit),
                    k_params: fit.k_params,
                    converged: fit.converged,
                    status: "ok",
                    error: None,
                };
                scored.push((cand, Some(fit)));
            }
            Err(e) => {
                let cand = Candidate {
                    spec,
                    short_name: spec.short_name(),
                    loglik: f64::NAN,
                    aic: f64::NAN,
                    aicc: f64::NAN,
                    bic: f64::NAN,
                    ic_value: f64::INFINITY,
                    k_params: 0,
                    converged: false,
                    status: "error",
                    error: Some(e.to_string()),
                };
                scored.push((cand, None));
            }
        }
    }
    // Stable sort by criterion: failures (+inf) last, ties in enumeration
    // order. A NaN criterion (should not happen) sorts last too.
    scored.sort_by(|a, b| a.0.ic_value.total_cmp(&b.0.ic_value));
    let n_candidates = scored.len();
    let n_fitted = scored.iter().filter(|(c, _)| c.status == "ok").count();
    let (candidates, fits): (Vec<Candidate>, Vec<Option<EtsFit>>) = scored.into_iter().unzip();
    let best = fits
        .into_iter()
        .flatten()
        .next()
        .ok_or_else(|| EtsError::NoCandidates {
            what: format!(
                "every auto_ets candidate failed to fit: {}",
                candidates
                    .iter()
                    .map(|c| format!("{}: {}", c.short_name, c.error.as_deref().unwrap_or("?")))
                    .collect::<Vec<_>>()
                    .join("; ")
            ),
        })?;
    if !best.ic_value_is_finite(opts.ic) {
        return Err(EtsError::NoCandidates {
            what: "no auto_ets candidate has a finite criterion value (the sample is \
                   too short for the AICc of every candidate)"
                .into(),
        });
    }
    Ok(AutoEtsResult {
        best,
        candidates,
        ic: opts.ic,
        n_candidates,
        n_fitted,
    })
}

impl EtsFit {
    fn ic_value_is_finite(&self, ic: Ic) -> bool {
        ic.of(self).is_finite()
    }
}
