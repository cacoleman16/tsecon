//! Fisher-type combination of per-unit ADF p-values.
//!
//! Given the per-unit MacKinnon p-values `p_i`, two closed-form combinations
//! test the joint null "every unit has a unit root":
//!
//! * Maddala-Wu (1999): `P = -2 sum_i ln p_i ~ chi^2(2N)` under the null,
//!   rejecting in the RIGHT tail (a large `P`, i.e. many tiny `p_i`, is
//!   evidence for stationarity). This is the headline `statistic`.
//! * Choi (2001) inverse-normal: `Z = N^{-1/2} sum_i Phi^{-1}(p_i) ~ N(0,1)`,
//!   rejecting in the LEFT tail.
//!
//! Both inherit the accuracy of the underlying ADF: no new distribution
//! theory is introduced, only exact arithmetic on the `p_i`. The p-values
//! are clamped to `[eps, 1 - eps]` before the logarithm and the normal
//! quantile because MacKinnon p-values saturate to exactly `0`/`1` deep in
//! the tails, where `ln 0` and `Phi^{-1}(0)` are undefined.

use tsecon_stats::{chi2_sf, ContinuousDist, StdNormal};

use crate::error::PanelRootError;

/// Clamp bound guarding `ln` and `Phi^{-1}` against saturated p-values.
pub(crate) const CLAMP_EPS: f64 = 1e-16;

/// Output of the Fisher-type combination.
pub(crate) struct FisherOut {
    /// Maddala-Wu statistic `P = -2 sum ln p_i` (the headline statistic).
    pub maddala_wu: f64,
    /// Right-tail chi-squared p-value of `P` on `2N` degrees of freedom.
    pub mw_pvalue: f64,
    /// Choi inverse-normal statistic `Z`.
    pub choi_z: f64,
    /// Left-tail standard-normal p-value of `Z`.
    pub choi_z_pvalue: f64,
    /// The per-unit p-values after clamping to `[eps, 1 - eps]`.
    pub clamped_pvalues: Vec<f64>,
}

/// Combine `pvalues` (one MacKinnon ADF p-value per unit) into the
/// Maddala-Wu and Choi statistics. `pvalues` must be non-empty and finite.
pub(crate) fn fisher_combine(pvalues: &[f64]) -> Result<FisherOut, PanelRootError> {
    let n = pvalues.len();
    let clamped: Vec<f64> = pvalues
        .iter()
        .map(|&p| p.clamp(CLAMP_EPS, 1.0 - CLAMP_EPS))
        .collect();

    let maddala_wu = -2.0 * clamped.iter().map(|&p| p.ln()).sum::<f64>();
    let mw_pvalue = chi2_sf(maddala_wu, 2.0 * n as f64)?;

    let mut z_sum = 0.0;
    for &p in &clamped {
        z_sum += StdNormal.ppf(p)?;
    }
    let choi_z = z_sum / (n as f64).sqrt();
    let choi_z_pvalue = StdNormal.cdf(choi_z);

    Ok(FisherOut {
        maddala_wu,
        mw_pvalue,
        choi_z,
        choi_z_pvalue,
        clamped_pvalues: clamped,
    })
}
