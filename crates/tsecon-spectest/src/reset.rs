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
/// would be added); [`SpecTestError::DegreesOfFreedom`] if the augmented fit
/// has no residual degrees of freedom (`n <= k + q`); the usual
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

    let base = ols(y, x_cols)?;
    let ssr_r = ssr(&base);
    let yhat = fitted(y, &base);

    // Augment with yhat^2 .. yhat^max_power.
    let q = max_power - 1;
    let mut aug: Vec<Vec<f64>> = x_cols.to_vec();
    for power in 2..=max_power {
        let col: Vec<f64> = yhat.iter().map(|&v| v.powi(power as i32)).collect();
        aug.push(col);
    }
    let p = k + q;
    if n <= p {
        return Err(SpecTestError::DegreesOfFreedom {
            what: "RESET augmented regression",
            n,
            k: p,
        });
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
