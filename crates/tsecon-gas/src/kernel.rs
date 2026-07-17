//! The observation densities and their scaled scores — the numerical
//! heart of the score-driven recursion.
//!
//! For a time-varying variance `f_t` the observation model is
//! `y_t ~ D(0, f_t)`, i.e. `y_t = sqrt(f_t) * eps_t` with `eps_t` drawn
//! from a **unit-variance** density `D` (so `Var(y_t | f_t) = f_t`). The
//! score-driven update is
//!
//! ```text
//! f_{t+1} = omega + a * s_t + b * f_t,   s_t = S_t * nabla_t,
//! ```
//!
//! where `nabla_t = d/df_t log p(y_t | f_t)` is the score of the
//! observation density and `S_t = I_t^{-1}` is the inverse-Fisher-
//! information scaling (Creal-Koopman-Lucas 2013, the "GAS" default).
//! This module supplies `log p(y_t | f_t)` and `s_t = S_t nabla_t` for the
//! Gaussian and standardized Student-t cases.

use crate::error::GasError;

/// The observation density driving the score recursion.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Density {
    /// Gaussian: `y_t ~ N(0, f_t)`.
    Gaussian,
    /// Standardized (unit-variance) Student-t with `nu` degrees of freedom
    /// (`nu > 2`); nests toward the Gaussian as `nu -> inf`.
    StudentT,
}

impl Density {
    /// Human-readable name, for diagnostics.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Density::Gaussian => "gaussian",
            Density::StudentT => "student-t",
        }
    }

    /// Whether this density needs a degrees-of-freedom parameter.
    #[must_use]
    pub fn needs_dof(self) -> bool {
        matches!(self, Density::StudentT)
    }
}

/// `0.5 * ln(2 * pi)`, the Gaussian log-density constant.
const HALF_LN_2PI: f64 = 0.918_938_533_204_672_8;

/// The observation log-density `log p(y | f)` for a variance `f > 0`.
///
/// # Gaussian
///
/// ```text
/// log p(y | f) = -0.5 ln(2 pi) - 0.5 ln f - 0.5 y^2 / f.
/// ```
///
/// # Standardized Student-t (`nu` dof, unit-variance)
///
/// With `y = sqrt(f) eps`, `eps` the unit-variance t whose density is
/// `f_eps(e) = Gamma((nu+1)/2) / (sqrt((nu-2) pi) Gamma(nu/2))
///            * (1 + e^2/(nu-2))^{-(nu+1)/2}`,
///
/// ```text
/// log p(y | f) = c(nu) - 0.5 ln f
///                - (nu+1)/2 * ln(1 + y^2 / ((nu-2) f)),
/// c(nu) = ln Gamma((nu+1)/2) - ln Gamma(nu/2) - 0.5 ln((nu-2) pi).
/// ```
///
/// The `nu` argument is ignored for the Gaussian.
#[must_use]
pub fn log_density(density: Density, nu: f64, y: f64, f: f64) -> f64 {
    match density {
        Density::Gaussian => -HALF_LN_2PI - 0.5 * f.ln() - 0.5 * y * y / f,
        Density::StudentT => {
            let z = y / f.sqrt();
            // Standardized unit-variance-t log density in z, plus the
            // -0.5 ln f Jacobian from y = sqrt(f) z. tsecon_stats provides
            // the exact standardized-t log pdf.
            -0.5 * f.ln() + std_t_ln_pdf(nu, z)
        }
    }
}

/// Log density of the unit-variance ("standardized") Student-t at `z`,
/// matching [`tsecon_stats::Standardized::student_t`] and, as a formula,
/// `c(nu) - (nu+1)/2 ln(1 + z^2/(nu-2))` with the `c(nu)` above.
fn std_t_ln_pdf(nu: f64, z: f64) -> f64 {
    use tsecon_stats::special::ln_gamma;
    let a = nu - 2.0;
    ln_gamma(0.5 * (nu + 1.0))
        - ln_gamma(0.5 * nu)
        - 0.5 * (a * core::f64::consts::PI).ln()
        - 0.5 * (nu + 1.0) * (z * z / a).ln_1p()
}

/// The scaled score `s_t = S_t nabla_t` with the inverse-information
/// scaling `S_t = I_t^{-1}` (Creal-Koopman-Lucas 2013).
///
/// # Gaussian
///
/// `nabla_t = 0.5 (y^2 - f) / f^2`, Fisher information `I_t = 0.5 / f^2`,
/// so `S_t = 2 f^2` and the scaled score collapses to
///
/// ```text
/// s_t = y^2 - f,
/// ```
///
/// giving the GARCH-like update `f_{t+1} = omega + a (y^2 - f) + b f`.
///
/// # Standardized Student-t
///
/// Writing `eps^2 = y^2 / f` and
/// `g(eps) = (nu eps^2 - (nu-2)) / ((nu-2) + eps^2)`, the raw score is
/// `nabla_t = g(eps) / (2 f)`. Its Fisher information is
/// `I_t = E[g^2] / (4 f^2)` with the closed form `E[g^2] = 2 nu / (nu+3)`
/// (verified by direct integration against the unit-variance-t density in
/// the fixture generator), so `S_t = 4 f^2 (nu+3) / (2 nu)` and
///
/// ```text
/// s_t = ((nu+3)/nu) * f * (nu y^2 - (nu-2) f) / ((nu-2) f + y^2).
/// ```
///
/// As `nu -> inf` this tends to `y^2 - f`, recovering the Gaussian update.
/// The response to a large `|y_t|` is bounded (it tends to
/// `((nu+3)/nu) f (nu - ... )`, saturating), which is the outlier-robust
/// property distinguishing GAS-t from GARCH.
///
/// The `nu` argument is ignored for the Gaussian.
#[must_use]
pub fn scaled_score(density: Density, nu: f64, y: f64, f: f64) -> f64 {
    match density {
        Density::Gaussian => y * y - f,
        Density::StudentT => {
            let y2 = y * y;
            let a = nu - 2.0;
            ((nu + 3.0) / nu) * f * (nu * y2 - a * f) / (a * f + y2)
        }
    }
}

/// Validate the dynamics/density parameters, returning a clear error rather
/// than propagating a NaN through the recursion.
pub(crate) fn validate_params(
    density: Density,
    omega: f64,
    a: f64,
    b: f64,
    nu: f64,
) -> Result<(), GasError> {
    if !(omega.is_finite() && omega > 0.0) {
        return Err(GasError::InvalidParameter {
            name: "omega",
            value: omega,
            requirement: "omega > 0 (finite)",
        });
    }
    if !(a.is_finite() && a >= 0.0) {
        return Err(GasError::InvalidParameter {
            name: "a",
            value: a,
            requirement: "a >= 0 (finite)",
        });
    }
    if !(b.is_finite() && (0.0..1.0).contains(&b)) {
        return Err(GasError::InvalidParameter {
            name: "b",
            value: b,
            requirement: "0 <= b < 1 (finite)",
        });
    }
    if density.needs_dof() && !(nu.is_finite() && nu > 2.0) {
        return Err(GasError::InvalidParameter {
            name: "nu",
            value: nu,
            requirement: "nu > 2 (finite)",
        });
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn gaussian_scaled_score_collapses() {
        // s_t = y^2 - f exactly.
        assert!(
            (scaled_score(Density::Gaussian, f64::NAN, 1.5, 0.8) - (1.5 * 1.5 - 0.8)).abs() < 1e-15
        );
    }

    #[test]
    fn student_t_nests_toward_gaussian() {
        // As nu -> inf the scaled score approaches the Gaussian y^2 - f.
        let (y, f) = (1.3, 0.9);
        let g = scaled_score(Density::Gaussian, f64::NAN, y, f);
        let t_big = scaled_score(Density::StudentT, 1e6, y, f);
        assert!((t_big - g).abs() < 1e-3, "{t_big} vs {g}");
    }

    #[test]
    fn student_t_downweights_outliers() {
        // A huge |y| produces a bounded score (robustness), unlike the
        // Gaussian whose score grows like y^2.
        let f = 1.0;
        let nu = 6.0;
        let s_moderate = scaled_score(Density::StudentT, nu, 2.0, f).abs();
        let s_huge = scaled_score(Density::StudentT, nu, 100.0, f).abs();
        // The t-score saturates: doubling-and-then-some of |y| barely moves it.
        assert!(s_huge < 5.0 * s_moderate + 5.0);
        // The Gaussian score, by contrast, explodes.
        let g_huge = scaled_score(Density::Gaussian, f64::NAN, 100.0, f).abs();
        assert!(g_huge > 100.0 * s_huge);
    }

    #[test]
    fn density_matches_stats_crate() {
        use tsecon_stats::{ContinuousDist, Standardized};
        let (nu, y, f) = (7.0_f64, 0.7_f64, 1.4_f64);
        let z = y / f.sqrt();
        let expected = -0.5 * f.ln() + Standardized::student_t(nu).unwrap().ln_pdf(z);
        assert!((log_density(Density::StudentT, nu, y, f) - expected).abs() < 1e-13);
    }

    #[test]
    fn validate_rejects_bad_params() {
        assert!(validate_params(Density::Gaussian, 0.0, 0.1, 0.9, f64::NAN).is_err());
        assert!(validate_params(Density::Gaussian, 0.1, -0.1, 0.9, f64::NAN).is_err());
        assert!(validate_params(Density::Gaussian, 0.1, 0.1, 1.0, f64::NAN).is_err());
        assert!(validate_params(Density::StudentT, 0.1, 0.1, 0.9, 2.0).is_err());
        assert!(validate_params(Density::StudentT, 0.1, 0.1, 0.9, 6.0).is_ok());
    }
}
