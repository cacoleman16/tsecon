//! The Barndorff-Nielsen-Shephard ratio jump test — a simple, documented
//! diagnostic for whether a day's price path contains a jump.

use core::f64::consts::PI;

use crate::error::RealizedError;
use crate::measures::{bipower_variation, realized_variance, tripower_quarticity};

/// `theta = pi^2/4 + pi - 5`, the asymptotic-variance constant of the
/// bipower-based jump statistic (Barndorff-Nielsen & Shephard 2004; Huang
/// & Tauchen 2005).
const THETA: f64 = PI * PI / 4.0 + PI - 5.0;

/// The BNS ratio jump statistic
///
/// ```text
///                 sqrt(n) * (RV - BV) / RV
///   z  =  ---------------------------------------------
///          sqrt( theta * max(1, TQ / BV^2) )
/// ```
///
/// with `theta = pi^2/4 + pi - 5`, realized variance `RV`, bipower
/// variation `BV`, and tripower quarticity `TQ`. Under the null of no
/// jumps the relative jump `(RV - BV)/RV` is centred at zero and `z` is
/// asymptotically standard normal; a jump inflates `RV` relative to the
/// jump-robust `BV`, pushing `z` large and positive. This is the "ratio"
/// (as opposed to difference or log) version, which Huang & Tauchen (2005)
/// find best sized in finite samples; the `TQ / BV^2` studentization uses
/// the jump-robust tripower quarticity so the denominator is not itself
/// inflated by the jump being tested for, and is floored at 1 exactly as
/// in that paper.
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
    let n = r.len() as f64;
    let relative_jump = (rv - bv) / rv;
    let denom = (THETA * (tq / (bv * bv)).max(1.0)).sqrt();
    Ok(n.sqrt() * relative_jump / denom)
}
