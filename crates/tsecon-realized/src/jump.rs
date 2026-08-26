//! The Barndorff-Nielsen-Shephard ratio jump test — a simple, documented
//! diagnostic for whether a day's price path contains a jump.

use core::f64::consts::PI;

use crate::error::RealizedError;
use crate::measures::{bipower_variation, realized_variance, tripower_quarticity};

/// `theta = pi^2/4 + pi - 5`, the asymptotic-variance constant of the
/// bipower-based jump statistic (Barndorff-Nielsen & Shephard 2004; Huang
/// & Tauchen 2005).
const THETA: f64 = PI * PI / 4.0 + PI - 5.0;

/// The BNS ratio jump statistic, in the Huang & Tauchen (2005) form
///
/// ```text
///                 sqrt(M) * (RV - BV~) / RV
///   z  =  ---------------------------------------------
///          sqrt( theta * max(1, TQ~ / BV~^2) )
/// ```
///
/// with `theta = pi^2/4 + pi - 5`, realized variance `RV`, and the
/// **finite-sample-adjusted** bipower variation and tripower quarticity
///
/// ```text
///   BV~ = (M / (M - 1)) * BV,      TQ~ = (M / (M - 2)) * TQ,
/// ```
///
/// which carry the Huang-Tauchen scalings for the `M - 1` (resp. `M - 2`)
/// products actually summed. Under the null of no jumps the relative jump
/// `(RV - BV~)/RV` is centred at zero and `z` is asymptotically standard
/// normal; a jump inflates `RV` relative to the jump-robust `BV~`, pushing
/// `z` large and positive. This is the "ratio" (as opposed to difference
/// or log) version, which Huang & Tauchen (2005) find best sized in finite
/// samples; the `TQ~ / BV~^2` studentization uses the jump-robust tripower
/// quarticity so the denominator is not itself inflated by the jump being
/// tested for, and is floored at 1 exactly as in that paper.
///
/// The exported [`bipower_variation`] and [`tripower_quarticity`] measures
/// remain the plain Barndorff-Nielsen-Shephard (2004) quantities; the
/// `M/(M-1)` and `M/(M-2)` factors are applied here, inside the test only.
/// (Changed in 0.6: through 0.5.0 the statistic used the unadjusted BNS-2004
/// `BV`/`TQ` while citing Huang-Tauchen; on marginal days the two z-values
/// straddle a critical value — e.g. 1.71 vs 1.58 around the one-sided 5%
/// cutoff 1.645 on an `M = 78` day with one modest jump.)
///
/// Returned as a raw z-score; compare against a normal critical value (e.g.
/// `1.645` at the 5% one-sided level). No golden fixture pins this — it is
/// a diagnostic — but larger `z` means stronger evidence of a jump.
///
/// # Errors
///
/// [`RealizedError::TooFewObservations`] with fewer than three returns
/// (tripower quarticity needs three), [`RealizedError::NonFinite`] on
/// NaN/inf input, and [`RealizedError::DegenerateSeries`] if `RV` or `BV`
/// is zero (the ratio is then undefined).
pub fn bns_jump_ratio(r: &[f64]) -> Result<f64, RealizedError> {
    let rv = realized_variance(r)?;
    let bv = bipower_variation(r)?;
    let tq = tripower_quarticity(r)?;
    if rv <= 0.0 || bv <= 0.0 {
        return Err(RealizedError::DegenerateSeries {
            what: "BNS ratio jump test",
        });
    }
    let m = r.len() as f64;
    // Huang-Tauchen (2005) finite-sample adjustments: the bipower sum has
    // M - 1 terms and the tripower sum M - 2, so the paper scales BV by
    // M/(M-1) and TQ by M/(M-2) before assembling the statistic. Tripower
    // quarticity has already enforced M >= 3, so both denominators are
    // strictly positive.
    let bv_ht = bv * m / (m - 1.0);
    let tq_ht = tq * m / (m - 2.0);
    let relative_jump = (rv - bv_ht) / rv;
    let denom = (THETA * (tq_ht / (bv_ht * bv_ht)).max(1.0)).sqrt();
    Ok(m.sqrt() * relative_jump / denom)
}
