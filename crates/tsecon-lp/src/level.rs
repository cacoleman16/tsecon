//! Single-impulse local projections: [`lp`].

use tsecon_hac::{ols, Kernel, SeType};

use crate::design::{check_finite, horizon_sample, outcome_column, single_impulse_design};
use crate::error::LpError;
use crate::spec::{LpResult, LpSpec, SeKind, SeSpec};

/// Estimate a single-impulse local-projection impulse-response function.
///
/// For each horizon `h in 0..=spec.horizons` this runs the Jordà (2005)
/// regression
///
/// ```text
///   y_{t+h} = beta_h * shock_t + c + sum_{l=1}^{p} phi_l y_{t-l} + u_{t,h}
/// ```
///
/// (with `p = spec.n_lag_controls`) and reports `beta_h` as the horizon-`h`
/// response. Two inference paths are available through [`SeSpec`]:
///
/// * [`SeSpec::LagAugmented`] (**default**) augments the regression with the
///   impulse's own lags `shock_{t-1}, ..., shock_{t-h}` and takes HC1
///   (Eicker-Huber-White) standard errors — Montiel Olea &
///   Plagborg-Møller (2021). See the crate docs for why this dominates HAC.
/// * [`SeSpec::Hac`] leaves the regression un-augmented and takes Newey-West
///   Bartlett HAC standard errors (statsmodels `cov_type="HAC"`,
///   `use_correction=True`); the lag truncation defaults to
///   `maxlags = h + p`.
///
/// `spec.cumulation` controls which side(s) are accumulated:
///
/// * [`Cumulation::Outcome`](crate::Cumulation::Outcome) — the outcome is the
///   cumulated `sum_{j=0}^{h} y_{t+j}` (Ramey-Zubairy), so the reported path
///   is a cumulative response whose standard errors are correct by
///   construction. The impulse stays contemporaneous, so this is a cumulative
///   *impulse response*, not a multiplier.
/// * [`Cumulation::Both`](crate::Cumulation::Both) — the impulse is cumulated
///   too, so `beta_h` is an OLS integral multiplier (cumulated `y` per
///   cumulated `shock`). For the identified version see
///   [`lp_multiplier`](crate::lp_multiplier).
///
/// # Errors
///
/// [`LpError::LengthMismatch`] if `shock` and `y` differ in length,
/// [`LpError::NonFinite`] on NaN/inf input, [`LpError::SeriesTooShort`] or
/// [`LpError::HorizonTooLong`] when a horizon has no usable sample, and
/// [`LpError::Hac`] wrapping any failure of the shared OLS/HAC engine
/// (e.g. a collinear design).
pub fn lp(y: &[f64], shock: &[f64], spec: LpSpec) -> Result<LpResult, LpError> {
    if shock.len() != y.len() {
        return Err(LpError::LengthMismatch {
            what: "impulse (shock) vs outcome (y)",
            expected: y.len(),
            got: shock.len(),
        });
    }
    check_finite(y, "outcome (y)")?;
    check_finite(shock, "impulse (shock)")?;

    let n = y.len();
    let p = spec.n_lag_controls;
    if n <= p {
        return Err(LpError::SeriesTooShort {
            n,
            n_lag_controls: p,
        });
    }

    let se_kind = match spec.se {
        SeSpec::LagAugmented => SeKind::LagAugmentedHc1,
        SeSpec::Hac { .. } => SeKind::HacBartlett,
    };

    let mut horizons = Vec::with_capacity(spec.horizons + 1);
    let mut irf = Vec::with_capacity(spec.horizons + 1);
    let mut se = Vec::with_capacity(spec.horizons + 1);
    let mut nobs_per_h = Vec::with_capacity(spec.horizons + 1);

    for h in 0..=spec.horizons {
        // Lag-augmentation order: the impulse's own lags 1..=h, which makes
        // the horizon-h regression score serially uncorrelated (Montiel Olea
        // & Plagborg-Møller 2021). The HAC path uses no impulse lags.
        let n_shock_lags = match spec.se {
            SeSpec::LagAugmented => h,
            SeSpec::Hac { .. } => 0,
        };

        let (start, nobs) = horizon_sample(n, h, p, n_shock_lags);
        let nparams = 2 + p + n_shock_lags;
        if nobs <= nparams {
            return Err(LpError::HorizonTooLong {
                horizon: h,
                nobs,
                nparams,
            });
        }

        let response = outcome_column(y, h, start, nobs, spec.cumulation.accumulates_outcome());
        let cols = single_impulse_design(
            y,
            shock,
            h,
            start,
            nobs,
            p,
            n_shock_lags,
            spec.cumulation.accumulates_impulse(),
        );

        let fit = ols(&response, &cols)?;
        let se_type = match spec.se {
            SeSpec::LagAugmented => SeType::Hc1,
            SeSpec::Hac { maxlags } => {
                let ml = maxlags.unwrap_or(h + p);
                SeType::Hac {
                    kernel: Kernel::Bartlett,
                    bandwidth: ml as f64,
                    use_correction: true,
                }
            }
        };
        let inf = fit.inference(se_type)?;

        horizons.push(h);
        irf.push(fit.params[0]);
        se.push(inf.bse[0]);
        nobs_per_h.push(nobs);
    }

    Ok(LpResult {
        horizons,
        irf,
        se,
        nobs_per_h,
        se_kind,
    })
}
