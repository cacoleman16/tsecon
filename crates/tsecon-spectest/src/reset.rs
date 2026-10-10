//! Ramsey (1969) RESET functional-form test.
//!
//! Fit `y = X b + u` by OLS and obtain the fitted values `yhat`. Refit `y` on
//! the augmented design `[X, yhat^2, ..., yhat^p]` and test the joint
//! significance of the added power terms. Under nonrobust covariance the Wald
//! F equals the sum-of-squares ratio
//!
//! ```text
//! F = [(SSR_r - SSR_u) / q] / [SSR_u / (n - (k + q))]  ~  F(q, n - k - q),
//! ```
//!
//! with `q = p - 1` added powers (`yhat^2 .. yhat^p`), `SSR_r` the residual
//! sum of squares of the original fit, and `SSR_u` that of the augmented fit.
//! With `max_power = 3` this reproduces statsmodels'
//! `linear_reset(res, power=3, test_type="fitted", use_f=True)`.

use tsecon_hac::ols;

use crate::breusch_godfrey::{breusch_godfrey_exceeded_budget, breusch_godfrey_max_nlags};
use crate::common::{f_sf, fitted, ssr, validate};
use crate::error::SpecTestError;
use crate::results::FTest;

/// Ramsey RESET test adding fitted-value powers `yhat^2 .. yhat^max_power` to
/// the regression of `y` on `x_cols`.
///
/// The common choice `max_power = 3` (powers `yhat^2, yhat^3`) matches
/// statsmodels' `linear_reset` default.
///
/// # Errors
///
/// Returns [`SpecTestError::InvalidPower`] if `max_power < 2` (no power terms
/// would be added); [`SpecTestError::DegreesOfFreedom`] if the original fit
/// has no residual degrees of freedom (`n <= k`);
/// [`SpecTestError::ResetPowerTooLarge`] if the augmented fit would have
/// none (`n <= k + q`) or its design would exceed the auxiliary-regression
/// budgets of [`crate::breusch_godfrey_test`] ([`crate::BG_MAX_AUX_CELLS`],
/// [`crate::BG_WORK_BUDGET`]) — checked before any power column is built,
/// naming `max_power` and the largest that fits;
/// [`SpecTestError::ResetPowerOverflow`] if a fitted-value power overflows
/// to infinity (rescale `y`, to which RESET is invariant); the usual
/// empty/dimension/finite validation errors; and propagates
/// [`SpecTestError::SingularDesign`] if a design is collinear.
pub fn reset_test(
    y: &[f64],
    x_cols: &[Vec<f64>],
    max_power: usize,
) -> Result<FTest, SpecTestError> {
    let (n, k) = validate(y, x_cols)?;
    if max_power < 2 {
        return Err(SpecTestError::InvalidPower { max_power });
    }
    if n <= k {
        return Err(SpecTestError::DegreesOfFreedom {
            what: "RESET original regression",
            n,
            k,
        });
    }
    // The augmented fit has p = k + q regressors (q = max_power - 1 added
    // powers): sized against the sample and the auxiliary-regression
    // budgets BEFORE any power column is built (checked, so an absurd
    // max_power can neither overflow nor allocate), naming max_power.
    let q = max_power - 1;
    let p = k.checked_add(q);
    let fits_sample = p.is_some_and(|p| n > p);
    let over_budget =
        fits_sample && p.is_some_and(|p| breusch_godfrey_exceeded_budget(n, p).is_some());
    if !fits_sample || over_budget {
        // The largest q that keeps n > k + q and fits both budgets.
        let max_q = breusch_godfrey_max_nlags(n, k);
        return Err(SpecTestError::ResetPowerTooLarge {
            max_power,
            n,
            k,
            max: (max_q >= 1).then_some(max_q + 1),
            over_budget,
        });
    }
    let p = k + q;

    let base = ols(y, x_cols)?;
    let ssr_r = ssr(&base);
    let yhat = fitted(y, &base);

    // Augment with yhat^2 .. yhat^max_power (max_power < n here, so the
    // exponent fits an i32 cast for any sample a machine holds).
    let mut aug: Vec<Vec<f64>> = x_cols.to_vec();
    for power in 2..=max_power {
        let col: Vec<f64> = yhat.iter().map(|&v| v.powi(power as i32)).collect();
        // A power of a large fitted value overflows: refuse naming
        // max_power rather than letting the OLS blame the data for an inf.
        if col.iter().any(|v| !v.is_finite()) {
            return Err(SpecTestError::ResetPowerOverflow {
                max_power,
                power,
                max_abs_fitted: yhat.iter().fold(0.0_f64, |m, &v| m.max(v.abs())),
            });
        }
        aug.push(col);
    }

    let augmented = ols(y, &aug)?;
    let ssr_u = ssr(&augmented);

    let df_num = q;
    let df_den = n - p;
    let fstat = ((ssr_r - ssr_u) / df_num as f64) / (ssr_u / df_den as f64);
    let pvalue = f_sf(fstat, df_num as f64, df_den as f64)?;

    Ok(FTest {
        fstat,
        df_num,
        df_den,
        pvalue,
    })
}
