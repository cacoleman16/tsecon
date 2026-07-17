//! Result records shared by more than one test.

/// Outcome of a heteroskedasticity Lagrange-multiplier test (White or
/// Breusch-Pagan): both report an `n * R^2` LM statistic that is
/// chi-square under the null, together with the equivalent F-form of the
/// same auxiliary regression.
#[derive(Debug, Clone, PartialEq)]
pub struct HetTest {
    /// The LM statistic `n * R^2_aux`.
    pub statistic: f64,
    /// Chi-square degrees of freedom: the number of auxiliary regressors
    /// excluding the constant.
    pub df: usize,
    /// Upper-tail p-value `P(chi2(df) > statistic)`.
    pub pvalue: f64,
    /// The F-form of the auxiliary regression (all auxiliary slopes jointly
    /// zero), matching statsmodels' `fvalue`.
    pub fstat: f64,
    /// Numerator degrees of freedom of [`Self::fstat`] (`= df`).
    pub f_df_num: usize,
    /// Denominator degrees of freedom of [`Self::fstat`]
    /// (`= n - #auxiliary regressors`).
    pub f_df_den: usize,
    /// Upper-tail p-value of [`Self::fstat`] under `F(f_df_num, f_df_den)`.
    pub f_pvalue: f64,
}

/// Outcome of an F-form specification test (Ramsey RESET).
#[derive(Debug, Clone, PartialEq)]
pub struct FTest {
    /// The F statistic.
    pub fstat: f64,
    /// Numerator degrees of freedom.
    pub df_num: usize,
    /// Denominator degrees of freedom.
    pub df_den: usize,
    /// Upper-tail p-value `P(F(df_num, df_den) > fstat)`.
    pub pvalue: f64,
}
