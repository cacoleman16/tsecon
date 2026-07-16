//! Lag-order selection by information criteria on a common sample.

use tsecon_linalg::faer::MatRef;

use crate::error::VarError;
use crate::estimate::estimate;
use crate::spec::Trend;

/// Information criteria of one candidate lag order.
#[derive(Debug, Clone, PartialEq)]
pub struct LagOrderCandidate {
    /// The candidate lag order `p`.
    pub lags: usize,
    /// Akaike information criterion.
    pub aic: f64,
    /// Schwarz (Bayesian) information criterion.
    pub bic: f64,
    /// Hannan-Quinn information criterion.
    pub hqic: f64,
    /// Final prediction error.
    pub fpe: f64,
}

/// Result of [`select_order`]: the per-criterion argmin lag orders and
/// the full candidate table.
#[derive(Debug, Clone, PartialEq)]
pub struct LagOrderSelection {
    /// Lag order minimizing the AIC.
    pub aic: usize,
    /// Lag order minimizing the BIC.
    pub bic: usize,
    /// Lag order minimizing the HQIC.
    pub hqic: usize,
    /// Lag order minimizing the FPE.
    pub fpe: usize,
    /// Criteria for every candidate order, ascending in `lags`.
    pub candidates: Vec<LagOrderCandidate>,
}

/// Selects the VAR lag order over candidate orders `p_min..=maxlags` by
/// minimizing each information criterion, statsmodels
/// `VAR.select_order` conventions (Lütkepohl 2005, section 4.3):
///
/// * every candidate is estimated on the **common sample** obtained by
///   dropping the first `maxlags - p` rows before fitting VAR(p), so
///   all fits share the same `n - maxlags` effective observations and
///   their criteria are comparable;
/// * `p_min = 0` (intercept-only baseline) when `trend` is
///   [`Trend::Constant`], `p_min = 1` when there are no deterministic
///   terms;
/// * ties go to the smaller order.
///
/// # Errors
///
/// * [`VarError::InvalidArgument`] if `maxlags == 0`;
/// * anything the per-candidate estimation can return (notably
///   [`VarError::InsufficientObservations`] when
///   `n - maxlags <= k * maxlags + n_trend`).
pub fn select_order(
    endog: MatRef<'_, f64>,
    maxlags: usize,
    trend: Trend,
) -> Result<LagOrderSelection, VarError> {
    if maxlags == 0 {
        return Err(VarError::InvalidArgument {
            what: "maxlags must be at least 1",
        });
    }
    let p_min = usize::from(trend == Trend::None);
    let mut candidates = Vec::with_capacity(maxlags + 1 - p_min);
    for p in p_min..=maxlags {
        let fit = estimate(endog, p, trend, maxlags - p)?;
        candidates.push(LagOrderCandidate {
            lags: p,
            aic: fit.aic,
            bic: fit.bic,
            hqic: fit.hqic,
            fpe: fit.fpe,
        });
    }
    let argmin = |get: fn(&LagOrderCandidate) -> f64| -> usize {
        let mut best = &candidates[0];
        for c in &candidates[1..] {
            // Strict inequality: ties keep the smaller lag order.
            if get(c) < get(best) {
                best = c;
            }
        }
        best.lags
    };
    Ok(LagOrderSelection {
        aic: argmin(|c| c.aic),
        bic: argmin(|c| c.bic),
        hqic: argmin(|c| c.hqic),
        fpe: argmin(|c| c.fpe),
        candidates,
    })
}
