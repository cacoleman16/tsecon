//! White (1980) test for heteroskedasticity of unknown form.
//!
//! Fit `y = X b + u` by OLS, then regress the squared residuals `u_t^2` on the
//! design's columns, their squares, and every pairwise cross-product — the
//! upper-triangular product basis `x_i x_j` for `i <= j`, which (because the
//! design carries a constant) contains the intercept, each regressor, each
//! square, and each interaction. The statistic is
//!
//! ```text
//! LM = n * R^2_aux  ~  chi2(m - 1),
//! ```
//!
//! where `m = k(k+1)/2` is the number of auxiliary regressors including the
//! constant and `R^2_aux` is the centered coefficient of determination of the
//! auxiliary regression. The equivalent F-form (all `m - 1` auxiliary slopes
//! jointly zero) is reported too. Both reproduce statsmodels'
//! `het_white(resid, exog)`.

use tsecon_hac::ols;
use tsecon_stats::chi2_sf;

use crate::common::{aux_regression, f_sf, has_constant, validate};
use crate::error::SpecTestError;
use crate::results::HetTest;

/// White's heteroskedasticity test on the regression of `y` on `x_cols`.
///
/// `x_cols` are the design columns with the constant included explicitly
/// (statsmodels exog convention). The auxiliary design is the set of products
/// `x_i * x_j` for all `i <= j`.
///
/// # Errors
///
/// Returns [`SpecTestError::MissingConstant`] if no design column is constant
/// (the centered `R^2` would not be the LM statistic);
/// [`SpecTestError::DegreesOfFreedom`] if the sample is too short for the
/// original fit (`n <= k`) or the auxiliary fit (`n <= m`); the usual
/// empty/dimension/finite validation errors; and propagates
/// [`SpecTestError::SingularDesign`] if either design is collinear (e.g. a
/// dummy regressor whose square duplicates it).
pub fn white_test(y: &[f64], x_cols: &[Vec<f64>]) -> Result<HetTest, SpecTestError> {
    let (n, k) = validate(y, x_cols)?;
    if !has_constant(x_cols) {
        return Err(SpecTestError::MissingConstant { what: "White test" });
    }
    if n <= k {
        return Err(SpecTestError::DegreesOfFreedom {
            what: "White test original regression",
            n,
            k,
        });
    }

    let fit = ols(y, x_cols)?;
    let u2: Vec<f64> = fit.residuals.iter().map(|u| u * u).collect();

    // Auxiliary design: the upper-triangular products x_i * x_j (i <= j).
    let mut aux: Vec<Vec<f64>> = Vec::with_capacity(k * (k + 1) / 2);
    for i in 0..k {
        for j in i..k {
            let col: Vec<f64> = x_cols[i]
                .iter()
                .zip(x_cols[j].iter())
                .map(|(&a, &b)| a * b)
                .collect();
            aux.push(col);
        }
    }
    let m = aux.len();

    let af = aux_regression(&u2, &aux, "White auxiliary regression")?;
    let df = m - 1;
    let statistic = n as f64 * af.r2;
    let pvalue = chi2_sf(statistic, df as f64)?;

    // F-form of the auxiliary regression (all m-1 slopes jointly zero).
    let ess = af.tss - af.ssr;
    let f_df_num = m - 1;
    let f_df_den = n - m;
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
