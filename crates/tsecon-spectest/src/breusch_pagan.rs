//! Breusch-Pagan (1979) test for heteroskedasticity, in the Koenker (1981)
//! studentized form.
//!
//! Fit `y = X b + u` by OLS, then regress the squared residuals `u_t^2`
//! directly on the design `X` (a constant plus the `k - 1` regressors). The
//! studentized statistic is
//!
//! ```text
//! LM = n * R^2_aux  ~  chi2(k - 1),
//! ```
//!
//! the sample size times the centered `R^2` of that auxiliary regression. This
//! Koenker version replaces the original Breusch-Pagan `0.5 * ESS/sigma^4`
//! scaling with `n * R^2`, which is robust to non-normal errors; it is the
//! default (`robust=True`) branch of statsmodels' `het_breuschpagan`, and the
//! equivalent F-form is reported alongside.

use tsecon_hac::ols;
use tsecon_stats::chi2_sf;

use crate::common::{aux_regression, f_sf, has_constant, validate};
use crate::error::SpecTestError;
use crate::results::HetTest;

/// The Koenker-studentized Breusch-Pagan test on the regression of `y` on
/// `x_cols`.
///
/// `x_cols` are the design columns with the constant included explicitly; the
/// auxiliary regression of `u^2` uses this same design.
///
/// # Errors
///
/// Returns [`SpecTestError::MissingConstant`] if no design column is constant;
/// [`SpecTestError::DegreesOfFreedom`] if `n <= k`; the usual
/// empty/dimension/finite validation errors; and propagates
/// [`SpecTestError::SingularDesign`] for a collinear design.
pub fn breusch_pagan_test(y: &[f64], x_cols: &[Vec<f64>]) -> Result<HetTest, SpecTestError> {
    let (n, k) = validate(y, x_cols)?;
    if !has_constant(x_cols) {
        return Err(SpecTestError::MissingConstant {
            what: "Breusch-Pagan test",
        });
    }
    if n <= k {
        return Err(SpecTestError::DegreesOfFreedom {
            what: "Breusch-Pagan test original regression",
            n,
            k,
        });
    }

    let fit = ols(y, x_cols)?;
    let u2: Vec<f64> = fit.residuals.iter().map(|u| u * u).collect();

    let af = aux_regression(&u2, x_cols, "Breusch-Pagan auxiliary regression")?;
    let df = k - 1;
    let statistic = n as f64 * af.r2;
    let pvalue = chi2_sf(statistic, df as f64)?;

    let ess = af.tss - af.ssr;
    let f_df_num = k - 1;
    let f_df_den = n - k;
    let fstat = (ess / f_df_num as f64) / (af.ssr / f_df_den as f64);
    let f_pvalue = f_sf(fstat, f_df_num as f64, f_df_den as f64)?;

    Ok(HetTest {
        statistic,
        df,
        pvalue,
        fstat,
        f_df_num,
        f_df_den,
        f_pvalue,
    })
}
