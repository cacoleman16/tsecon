//! Unit-variance ("standardized") wrappers for GARCH innovations.

use super::{ContinuousDist, Ged, StudentT};
use crate::error::StatsError;
use crate::special::ln_gamma;

/// A distribution rescaled to unit variance: if `X ~ D` has standard
/// deviation `s`, then `Z = X / s` and this wrapper is the distribution of
/// `Z`.
///
/// The change of variables gives
///
/// ```text
/// f_Z(z) = s · f_X(s z),   F_Z(z) = F_X(s z),   Q_Z(u) = Q_X(u) / s
/// ```
///
/// This is the form GARCH-type models need for their innovations (mean 0,
/// variance 1). Use the parameter-validated constructors
/// [`Standardized::student_t`] and [`Standardized::ged`]; the scale factors
/// they apply are documented there. ([`super::HansenSkewT`] is already
/// standardized by construction and needs no wrapper.)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Standardized<D> {
    inner: D,
    sd: f64,
}

impl<D: ContinuousDist> Standardized<D> {
    /// Wrap an arbitrary distribution given the standard deviation `sd` of
    /// the *inner* distribution. Prefer the named constructors, which
    /// compute `sd` from the parameters; this escape hatch exists for
    /// distributions added later.
    ///
    /// Errors unless `0 < sd < inf`. Correctness of the wrapper relies on
    /// the caller supplying the true standard deviation of `inner`, and on
    /// `inner` having mean zero.
    pub fn from_parts(inner: D, sd: f64) -> Result<Self, StatsError> {
        if !(sd > 0.0 && sd.is_finite()) {
            return Err(StatsError::InvalidParameter {
                name: "sd",
                value: sd,
                requirement: "0 < sd < inf",
            });
        }
        Ok(Self { inner, sd })
    }

    /// The wrapped (unstandardized) distribution.
    pub fn inner(&self) -> &D {
        &self.inner
    }

    /// The scale factor: the standard deviation of the inner distribution
    /// that draws are divided by.
    pub fn sd(&self) -> f64 {
        self.sd
    }
}

impl Standardized<StudentT> {
    /// The unit-variance Student t with `df > 2` degrees of freedom — the
    /// classic GARCH innovation density (Bollerslev 1987).
    ///
    /// **Scale factor**: a Student t with `df = ν` has variance `ν/(ν-2)`,
    /// so draws are divided by `s = sqrt(ν/(ν-2))`, giving the density
    ///
    /// ```text
    /// f(z) = Γ((ν+1)/2) / (√(π(ν-2)) Γ(ν/2)) · (1 + z²/(ν-2))^(-(ν+1)/2)
    /// ```
    pub fn student_t(df: f64) -> Result<Self, StatsError> {
        if !(df > 2.0 && df.is_finite()) {
            return Err(StatsError::InvalidParameter {
                name: "df",
                value: df,
                requirement: "2 < df < inf (variance must exist)",
            });
        }
        let inner = StudentT::new(df)?;
        let sd = (df / (df - 2.0)).sqrt();
        Ok(Self { inner, sd })
    }
}

impl Standardized<Ged> {
    /// The unit-variance GED with shape `nu > 0` — the GED innovation of
    /// Nelson (1991).
    ///
    /// **Scale factor**: in the scipy `gennorm` parameterization the GED has
    /// variance `Γ(3/ν)/Γ(1/ν)`, so draws are divided by
    /// `s = sqrt(Γ(3/ν)/Γ(1/ν))`.
    pub fn ged(nu: f64) -> Result<Self, StatsError> {
        let inner = Ged::new(nu)?;
        let sd = (0.5 * (ln_gamma(3.0 / nu) - ln_gamma(1.0 / nu))).exp();
        Ok(Self { inner, sd })
    }
}

impl<D: ContinuousDist> ContinuousDist for Standardized<D> {
    fn pdf(&self, x: f64) -> f64 {
        self.sd * self.inner.pdf(self.sd * x)
    }

    fn ln_pdf(&self, x: f64) -> f64 {
        self.sd.ln() + self.inner.ln_pdf(self.sd * x)
    }

    fn cdf(&self, x: f64) -> f64 {
        self.inner.cdf(self.sd * x)
    }

    fn sf(&self, x: f64) -> f64 {
        self.inner.sf(self.sd * x)
    }

    fn ppf(&self, u: f64) -> Result<f64, StatsError> {
        Ok(self.inner.ppf(u)? / self.sd)
    }
}
