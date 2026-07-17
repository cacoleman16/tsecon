//! The Pesaran (2006) common-correlated-effects mean-group (CCE-MG) estimator.
//!
//! Plain mean group (see [`crate::mg`]) is biased when the units share an
//! unobserved common factor `f_t` that is correlated with the regressors:
//!
//! ```text
//! y_it = a_i + b_i' x_it + gamma_i' f_t + e_it,
//! x_it = mu_i + Delta_i' f_t + v_it .
//! ```
//!
//! Omitting `f_t` leaves an omitted-variable term in each per-unit slope that,
//! because `E[gamma_i Delta_i] != 0`, does **not** average away across units.
//!
//! Pesaran's (2006) insight is that the per-period cross-section averages of
//! the observables span the space of the factors: with `zbar_t = (1/N) sum_i
//! (y_it, x_it')`, including `zbar_t` as extra regressors in each unit's
//! equation absorbs `f_t`. The CCE-MG estimator therefore
//!
//! 1. forms the cross-section averages `ybar_t` and `xbar_t` (per time `t`,
//!    over units);
//! 2. runs a per-unit OLS of `y_i` on `[const, x_i, ybar, xbar]`; and
//! 3. MG-averages **only** the own-`x` slope coefficients (the coefficients on
//!    the augmenting cross-section averages are nuisance parameters), reusing
//!    the identical [`crate::mg`] averaging and standard-error formulas.
//!
//! Forming the cross-section averages requires a **balanced** panel: every unit
//! must be observed over the same time index so the averages line up.

use tsecon_hac::ols;

use crate::error::PanelTsError;
use crate::mg::{assemble_mean_group, design_with_const, validate_units, MeanGroup, PanelUnit};

/// Cross-section averages, per time `t`, of `y` and of every regressor column.
///
/// Returns `(ybar, xbar)` where `ybar` has length `T` and `xbar` is `k`
/// columns each of length `T`.
fn cross_section_averages(units: &[PanelUnit], t: usize, k: usize) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = units.len();
    let nf = n as f64;

    let mut ybar = vec![0.0_f64; t];
    for unit in units {
        for (acc, &v) in ybar.iter_mut().zip(unit.y.iter()) {
            *acc += v;
        }
    }
    for acc in ybar.iter_mut() {
        *acc /= nf;
    }

    let mut xbar = vec![vec![0.0_f64; t]; k];
    for unit in units {
        for (col_acc, col) in xbar.iter_mut().zip(unit.x.iter()) {
            for (acc, &v) in col_acc.iter_mut().zip(col.iter()) {
                *acc += v;
            }
        }
    }
    for col_acc in xbar.iter_mut() {
        for acc in col_acc.iter_mut() {
            *acc /= nf;
        }
    }

    (ybar, xbar)
}

/// The Pesaran (2006) common-correlated-effects mean-group estimator.
///
/// Augments each unit's regression with the per-period cross-section averages
/// of `y` and of every regressor, runs the per-unit OLS via [`tsecon_hac::ols`],
/// and MG-averages only the own-`x` slopes. This purges an unobserved common
/// factor that would otherwise bias plain [`crate::mg::mean_group`].
///
/// # Errors
///
/// [`PanelTsError::TooFewUnits`], [`PanelTsError::NoRegressors`],
/// [`PanelTsError::InconsistentRegressors`], or [`PanelTsError::RaggedUnit`]
/// for malformed units; [`PanelTsError::UnbalancedPanel`] if the units span
/// different numbers of periods (the cross-section averages are then
/// undefined); [`PanelTsError::Ols`] wrapping any per-unit OLS failure. Note
/// the augmented design carries `2k + 2` columns, so each unit needs
/// `T > 2k + 2` periods.
pub fn cce_mean_group(units: &[PanelUnit]) -> Result<MeanGroup, PanelTsError> {
    let k = validate_units(units)?;

    // CCE additionally requires a balanced panel.
    let t = units[0].t();
    for (i, unit) in units.iter().enumerate() {
        if unit.t() != t {
            return Err(PanelTsError::UnbalancedPanel {
                unit: i,
                expected: t,
                got: unit.t(),
            });
        }
    }

    let (ybar, xbar) = cross_section_averages(units, t, k);

    let mut slopes = Vec::with_capacity(units.len());
    for (i, unit) in units.iter().enumerate() {
        // Augmented regressors: [x_i, ybar, xbar_1, ..., xbar_k].
        let mut aug = Vec::with_capacity(2 * k + 1);
        aug.extend(unit.x.iter().cloned());
        aug.push(ybar.clone());
        aug.extend(xbar.iter().cloned());

        let design = design_with_const(&aug, t); // [const, x_i, ybar, xbar]
        let fit = ols(&unit.y, &design).map_err(|source| PanelTsError::Ols { unit: i, source })?;
        // params = [const, own-x (k), ybar, xbar (k)]; keep the own-x slopes.
        slopes.push(
            fit.params
                .iter()
                .skip(1)
                .take(k)
                .copied()
                .collect::<Vec<f64>>(),
        );
    }

    Ok(assemble_mean_group(slopes, k))
}
