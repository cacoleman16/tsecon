//! Chow (1960) test for a structural break at a KNOWN split point.
//!
//! Split the sample at a known index into two regimes, fit `y = X b + u`
//! separately on each and pooled on the whole, and compare the residual sums
//! of squares:
//!
//! ```text
//! F = [(SSR_pooled - SSR_1 - SSR_2) / k]
//!     / [(SSR_1 + SSR_2) / (n - 2k)]      ~  F(k, n - 2k).
//! ```
//!
//! A large `F` says the pooled fit leaves far more unexplained variation than
//! the two regime fits combined, i.e. the coefficient vector differs across
//! the split. `SSR_pooled`, `SSR_1`, and `SSR_2` all come from
//! [`tsecon_hac::ols`].

use tsecon_hac::ols;

use crate::common::{f_sf, ssr, validate};
use crate::error::SpecTestError;

/// Outcome of a Chow structural-break test at a known split.
#[derive(Debug, Clone, PartialEq)]
pub struct ChowTest {
    /// The Chow F statistic.
    pub fstat: f64,
    /// Numerator degrees of freedom, `k` (the number of regressors).
    pub df_num: usize,
    /// Denominator degrees of freedom, `n - 2k`.
    pub df_den: usize,
    /// Upper-tail p-value `P(F(df_num, df_den) > fstat)`.
    pub pvalue: f64,
    /// Residual sum of squares of the pooled (whole-sample) fit.
    pub ssr_pooled: f64,
    /// Residual sum of squares of the first regime (`0..split`).
    pub ssr1: f64,
    /// Residual sum of squares of the second regime (`split..n`).
    pub ssr2: f64,
}

/// Chow test for a break at the known 0-indexed `split`: the first regime is
/// observations `0..split`, the second is `split..n`.
///
/// # Errors
///
/// Returns [`SpecTestError::InvalidSplit`] unless `k < split < n - k` (both
/// sub-samples must have more observations than regressors, leaving `n - 2k`
/// denominator degrees of freedom); the usual empty/dimension/finite
/// validation errors; and propagates [`SpecTestError::SingularDesign`] if any
/// of the three designs is collinear (which a regime split can induce, e.g. a
/// regressor constant within one regime).
pub fn chow_test(y: &[f64], x_cols: &[Vec<f64>], split: usize) -> Result<ChowTest, SpecTestError> {
    let (n, k) = validate(y, x_cols)?;
    // Need split > k and n - split > k so each sub-sample is estimable.
    if split <= k || split >= n - k {
        return Err(SpecTestError::InvalidSplit { split, n, k });
    }

    let slice_cols = |lo: usize, hi: usize| -> Vec<Vec<f64>> {
        x_cols.iter().map(|col| col[lo..hi].to_vec()).collect()
    };

    let pooled = ols(y, x_cols)?;
    let ssr_pooled = ssr(&pooled);

    let x1 = slice_cols(0, split);
    let fit1 = ols(&y[0..split], &x1)?;
    let ssr1 = ssr(&fit1);

    let x2 = slice_cols(split, n);
    let fit2 = ols(&y[split..n], &x2)?;
    let ssr2 = ssr(&fit2);

    let df_num = k;
    let df_den = n - 2 * k;
    let fstat = ((ssr_pooled - ssr1 - ssr2) / df_num as f64) / ((ssr1 + ssr2) / df_den as f64);
    let pvalue = f_sf(fstat, df_num as f64, df_den as f64)?;

    Ok(ChowTest {
        fstat,
        df_num,
        df_den,
        pvalue,
        ssr_pooled,
        ssr1,
        ssr2,
    })
}
