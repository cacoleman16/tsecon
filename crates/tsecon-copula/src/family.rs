//! The five copula families of this slice — density, CDF, Kendall-tau
//! maps, and closed-form tail dependence — all bivariate.
//!
//! Formulas (Joe 2014; statsmodels-verified at 1e-12 in the fixture
//! generator before any golden number is written):
//!
//! * **Gaussian** (`rho`): `c(u) = phi2(z1, z2; rho) / (phi(z1) phi(z2))`,
//!   `z = Phi^{-1}(u)`; `C(u) = Phi2(z1, z2; rho)` (Genz BVND);
//!   `tau = 2 arcsin(rho) / pi`; tail dependence `(0, 0)`.
//! * **Student-t** (`rho`, `nu`): the same construction with `t_nu`
//!   margins and the bivariate t density; the CDF has no closed form —
//!   the conditional 1-D integral is used (see [`crate::special`]);
//!   `tau = 2 arcsin(rho) / pi` (free of `nu`); tail dependence
//!   `lambda = 2 t_{nu+1}(-sqrt((nu+1)(1-rho)/(1+rho)))` in *both* tails
//!   (Demarta-McNeil 2005, eq. 15).
//! * **Clayton** (`theta > 0`, the positive-dependence branch; rotations
//!   are deferred, stated honestly): lower-tail dependent;
//!   `C = (u^-theta + v^-theta - 1)^{-1/theta}`; `tau = theta/(theta+2)`;
//!   tails `(2^{-1/theta}, 0)`.
//! * **Gumbel** (`theta >= 1`): upper-tail dependent;
//!   `C = exp(-((-ln u)^theta + (-ln v)^theta)^{1/theta})`;
//!   `tau = (theta-1)/theta`; tails `(0, 2 - 2^{1/theta})`.
//! * **Frank** (`theta != 0`, either sign — the Archimedean family that
//!   covers negative dependence): radially symmetric, tail-independent;
//!   `tau = 1 + 4 (D1(theta) - 1)/theta` (Debye `D1`); tails `(0, 0)`.
//!
//! The Clayton/Gumbel/Frank evaluators work in log-sum-exp form so large
//! `theta` and extreme `u` never overflow.

use tsecon_stats::special::ln_gamma;
use tsecon_stats::{ContinuousDist, StdNormal, StudentT};

use crate::error::CopulaError;
use crate::special::{bvn_cdf, bvt_cdf, debye_1, norm_cdf, tau_frank};

/// The largest admissible `|theta|` for the Frank family (`exp(|theta|)`
/// must not overflow anywhere in the density); `tau_to_param` states the
/// corresponding tau bound in its error.
pub const FRANK_THETA_MAX: f64 = 700.0;

/// A bivariate copula family (this slice is bivariate throughout;
/// `d > 2` for the elliptical families is a documented deferral).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Family {
    /// Gaussian copula — parameter `rho` in (-1, 1); no tail dependence.
    Gaussian,
    /// Student-t copula — parameters `rho` in (-1, 1) and `nu > 0`;
    /// symmetric tail dependence.
    StudentT,
    /// Clayton copula — parameter `theta > 0` (positive-dependence
    /// branch); lower-tail dependence.
    Clayton,
    /// Gumbel copula — parameter `theta >= 1`; upper-tail dependence.
    Gumbel,
    /// Frank copula — parameter `theta != 0` (either sign); no tail
    /// dependence.
    Frank,
}

impl Family {
    /// Parse a family name (the Python-surface strings).
    pub fn parse(name: &str) -> Result<Self, CopulaError> {
        match name.to_ascii_lowercase().as_str() {
            "gaussian" | "normal" => Ok(Self::Gaussian),
            "t" | "student_t" | "student-t" | "studentt" => Ok(Self::StudentT),
            "clayton" => Ok(Self::Clayton),
            "gumbel" => Ok(Self::Gumbel),
            "frank" => Ok(Self::Frank),
            _ => Err(CopulaError::UnknownFamily {
                name: name.to_string(),
            }),
        }
    }

    /// Canonical lowercase name (as reported in results and errors).
    pub fn name(&self) -> &'static str {
        match self {
            Self::Gaussian => "gaussian",
            Self::StudentT => "t",
            Self::Clayton => "clayton",
            Self::Gumbel => "gumbel",
            Self::Frank => "frank",
        }
    }

    /// Number of dependence parameters (2 for t, 1 otherwise).
    pub fn n_params(&self) -> usize {
        match self {
            Self::StudentT => 2,
            _ => 1,
        }
    }

    /// Names of the dependence parameters, in `params` order.
    pub fn param_names(&self) -> &'static [&'static str] {
        match self {
            Self::Gaussian => &["rho"],
            Self::StudentT => &["rho", "nu"],
            _ => &["theta"],
        }
    }
}

/// Validates a parameter vector against the family's domain.
pub(crate) fn validate_params(family: Family, params: &[f64]) -> Result<(), CopulaError> {
    let fam = family.name();
    if params.len() != family.n_params() {
        return Err(CopulaError::WrongParamCount {
            family: fam,
            expected: family.n_params(),
            got: params.len(),
        });
    }
    let bad = |name: &'static str, value: f64, requirement: &'static str| {
        Err(CopulaError::InvalidParameter {
            family: fam,
            name,
            value,
            requirement,
        })
    };
    match family {
        Family::Gaussian => {
            let rho = params[0];
            if !(rho.is_finite() && rho > -1.0 && rho < 1.0) {
                return bad("rho", rho, "-1 < rho < 1");
            }
        }
        Family::StudentT => {
            let (rho, nu) = (params[0], params[1]);
            if !(rho.is_finite() && rho > -1.0 && rho < 1.0) {
                return bad("rho", rho, "-1 < rho < 1");
            }
            if !(nu.is_finite() && nu > 0.0) {
                return bad("nu", nu, "0 < nu < inf");
            }
        }
        Family::Clayton => {
            let th = params[0];
            if !(th.is_finite() && th > 0.0) {
                return bad(
                    "theta",
                    th,
                    "theta > 0 (the positive-dependence branch; rotations \
                     are deferred in this slice)",
                );
            }
        }
        Family::Gumbel => {
            let th = params[0];
            if !(th.is_finite() && th >= 1.0) {
                return bad("theta", th, "theta >= 1 (theta = 1 is independence)");
            }
        }
        Family::Frank => {
            let th = params[0];
            if !(th.is_finite() && th != 0.0 && th.abs() <= FRANK_THETA_MAX) {
                return bad(
                    "theta",
                    th,
                    "theta != 0 and |theta| <= 700 (theta -> 0 is the \
                     independence limit)",
                );
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Per-point log-density kernels (u already validated in (0,1), params in
// domain; each returns -inf only through the caller's non-finite guard).
// ---------------------------------------------------------------------------

#[inline]
fn logpdf_gaussian_z(z1: f64, z2: f64, rho: f64) -> f64 {
    // The cross term is grouped as rho * (z1 z2) so the kernel is
    // bit-exactly exchangeable in (z1, z2) — property-tested.
    let r2 = 1.0 - rho * rho;
    -0.5 * r2.ln() - (rho * rho * (z1 * z1 + z2 * z2) - 2.0 * rho * (z1 * z2)) / (2.0 * r2)
}

#[inline]
fn logpdf_t_x(x1: f64, x2: f64, rho: f64, nu: f64, t_nu: &StudentT) -> f64 {
    // Sums and cross term grouped for bit-exact exchangeability (see
    // the Gaussian kernel).
    let r2 = 1.0 - rho * rho;
    let q = ((x1 * x1 + x2 * x2) - 2.0 * rho * (x1 * x2)) / r2;
    let ln_f2 = ln_gamma((nu + 2.0) / 2.0)
        - ln_gamma(nu / 2.0)
        - (nu * core::f64::consts::PI).ln()
        - 0.5 * r2.ln()
        - (nu + 2.0) / 2.0 * (q / nu).ln_1p();
    ln_f2 - t_nu.ln_pdf(x1) - t_nu.ln_pdf(x2)
}

/// `ln(u1^-theta + u2^-theta - 1)` in overflow-safe log-sum-exp form.
#[inline]
fn clayton_ln_s(u1: f64, u2: f64, theta: f64) -> f64 {
    let a1 = -theta * u1.ln();
    let a2 = -theta * u2.ln();
    let m = a1.max(a2);
    m + ((a1 - m).exp() + (a2 - m).exp() - (-m).exp()).ln()
}

#[inline]
fn logpdf_clayton(u1: f64, u2: f64, theta: f64) -> f64 {
    (1.0 + theta).ln()
        - (1.0 + theta) * (u1.ln() + u2.ln())
        - (2.0 + 1.0 / theta) * clayton_ln_s(u1, u2, theta)
}

/// `ln((-ln u1)^theta + (-ln u2)^theta)` in log-sum-exp form.
#[inline]
fn gumbel_ln_s(x: f64, y: f64, theta: f64) -> f64 {
    let b1 = theta * x.ln();
    let b2 = theta * y.ln();
    let m = b1.max(b2);
    m + ((b1 - m).exp() + (b2 - m).exp()).ln()
}

#[inline]
fn logpdf_gumbel(u1: f64, u2: f64, theta: f64) -> f64 {
    let x = -u1.ln();
    let y = -u2.ln();
    let ln_s = gumbel_ln_s(x, y, theta);
    let a = (ln_s / theta).exp();
    -a + (a + theta - 1.0).ln()
        + (1.0 / theta - 2.0) * ln_s
        + (theta - 1.0) * (x.ln() + y.ln())
        + x
        + y
}

#[inline]
fn logpdf_frank(u1: f64, u2: f64, theta: f64) -> f64 {
    // c = theta b e^{-theta(u+v)} / (b - g1 g2)^2, b = 1 - e^{-theta},
    // g_i = 1 - e^{-theta u_i}; theta*b > 0 for either sign of theta and
    // the denominator enters squared, so |.| is exact.
    let b = -(-theta).exp_m1();
    let g1 = -(-theta * u1).exp_m1();
    let g2 = -(-theta * u2).exp_m1();
    (theta * b).ln() - theta * (u1 + u2) - 2.0 * (b - g1 * g2).abs().ln()
}

// ---------------------------------------------------------------------------
// Public evaluators
// ---------------------------------------------------------------------------

fn check_eval_u(u1: &[f64], u2: &[f64]) -> Result<(), CopulaError> {
    crate::common::check_series(u1, "u[:, 0]")?;
    crate::common::check_series(u2, "u[:, 1]")?;
    if u1.len() != u2.len() {
        return Err(CopulaError::LengthMismatch {
            n1: u1.len(),
            n2: u2.len(),
        });
    }
    for (what, col) in [("u[:, 0]", u1), ("u[:, 1]", u2)] {
        if let Some(index) = col.iter().position(|&v| !(v > 0.0 && v < 1.0)) {
            return Err(CopulaError::OutOfUnitInterval {
                what,
                index,
                value: col[index],
            });
        }
    }
    Ok(())
}

/// Log-density of `family` at each `(u1[i], u2[i])` pair for fixed
/// `params` — the elementwise log-copula-density, comparable to
/// statsmodels `Copula.logpdf` (golden-pinned at 1e-10).
///
/// # Errors
///
/// Malformed `u` (empty, non-finite, length mismatch, outside `(0, 1)`)
/// or `params` outside the family's domain.
pub fn copula_logpdf(
    u1: &[f64],
    u2: &[f64],
    family: Family,
    params: &[f64],
) -> Result<Vec<f64>, CopulaError> {
    check_eval_u(u1, u2)?;
    validate_params(family, params)?;
    logpdf_unchecked(u1, u2, family, params)
}

/// [`copula_logpdf`] without input re-validation — the MLE hot path.
/// Params must already be inside the family domain.
pub(crate) fn logpdf_unchecked(
    u1: &[f64],
    u2: &[f64],
    family: Family,
    params: &[f64],
) -> Result<Vec<f64>, CopulaError> {
    let n = u1.len();
    let mut out = Vec::with_capacity(n);
    match family {
        Family::Gaussian => {
            let rho = params[0];
            for i in 0..n {
                let z1 = StdNormal.ppf(u1[i])?;
                let z2 = StdNormal.ppf(u2[i])?;
                out.push(logpdf_gaussian_z(z1, z2, rho));
            }
        }
        Family::StudentT => {
            let (rho, nu) = (params[0], params[1]);
            let t_nu = StudentT::new(nu).map_err(|_| CopulaError::InvalidParameter {
                family: "t",
                name: "nu",
                value: nu,
                requirement: "0 < nu < inf",
            })?;
            for i in 0..n {
                let x1 = t_nu.ppf(u1[i])?;
                let x2 = t_nu.ppf(u2[i])?;
                out.push(logpdf_t_x(x1, x2, rho, nu, &t_nu));
            }
        }
        Family::Clayton => {
            let th = params[0];
            for i in 0..n {
                out.push(logpdf_clayton(u1[i], u2[i], th));
            }
        }
        Family::Gumbel => {
            let th = params[0];
            for i in 0..n {
                out.push(logpdf_gumbel(u1[i], u2[i], th));
            }
        }
        Family::Frank => {
            let th = params[0];
            for i in 0..n {
                out.push(logpdf_frank(u1[i], u2[i], th));
            }
        }
    }
    Ok(out)
}

/// Density of `family` at each `(u1[i], u2[i])` pair —
/// `exp(copula_logpdf)`.
///
/// # Errors
///
/// As [`copula_logpdf`].
pub fn copula_pdf(
    u1: &[f64],
    u2: &[f64],
    family: Family,
    params: &[f64],
) -> Result<Vec<f64>, CopulaError> {
    Ok(copula_logpdf(u1, u2, family, params)?
        .into_iter()
        .map(f64::exp)
        .collect())
}

/// CDF `C(u1[i], u2[i])` of `family` at fixed `params`.
///
/// Gaussian: Genz BVND (double-precision exact). Student-t: the
/// conditional 1-D integral to ~1e-12 (no closed form exists;
/// statsmodels' `StudentTCopula.cdf` raises `NotImplementedError`).
/// Archimedean: closed forms in log-sum-exp form.
///
/// # Errors
///
/// As [`copula_logpdf`].
pub fn copula_cdf(
    u1: &[f64],
    u2: &[f64],
    family: Family,
    params: &[f64],
) -> Result<Vec<f64>, CopulaError> {
    check_eval_u(u1, u2)?;
    validate_params(family, params)?;
    let n = u1.len();
    let mut out = Vec::with_capacity(n);
    match family {
        Family::Gaussian => {
            let rho = params[0];
            for i in 0..n {
                let z1 = StdNormal.ppf(u1[i])?;
                let z2 = StdNormal.ppf(u2[i])?;
                out.push(bvn_cdf(z1, z2, rho));
            }
        }
        Family::StudentT => {
            let (rho, nu) = (params[0], params[1]);
            let t_nu = StudentT::new(nu).map_err(|_| CopulaError::InvalidParameter {
                family: "t",
                name: "nu",
                value: nu,
                requirement: "0 < nu < inf",
            })?;
            for i in 0..n {
                let x1 = t_nu.ppf(u1[i])?;
                let x2 = t_nu.ppf(u2[i])?;
                out.push(bvt_cdf(x1, x2, rho, nu));
            }
        }
        Family::Clayton => {
            let th = params[0];
            for i in 0..n {
                out.push((-clayton_ln_s(u1[i], u2[i], th) / th).exp());
            }
        }
        Family::Gumbel => {
            let th = params[0];
            for i in 0..n {
                let ln_s = gumbel_ln_s(-u1[i].ln(), -u2[i].ln(), th);
                out.push((-(ln_s / th).exp()).exp());
            }
        }
        Family::Frank => {
            let th = params[0];
            let b = -(-th).exp_m1();
            for i in 0..n {
                let g1 = -(-th * u1[i]).exp_m1();
                let g2 = -(-th * u2[i]).exp_m1();
                out.push(-(-g1 * g2 / b).ln_1p() / th);
            }
        }
    }
    Ok(out)
}

/// Copula log-likelihood: `sum(copula_logpdf)` — the quantity
/// [`crate::copula_fit`] maximizes, exposed for model comparison and
/// testing.
///
/// # Errors
///
/// As [`copula_logpdf`].
pub fn copula_loglik(
    u1: &[f64],
    u2: &[f64],
    family: Family,
    params: &[f64],
) -> Result<f64, CopulaError> {
    Ok(copula_logpdf(u1, u2, family, params)?.iter().sum())
}

// ---------------------------------------------------------------------------
// Kendall-tau maps and tail dependence
// ---------------------------------------------------------------------------

/// The Kendall tau implied by `params` (closed forms; see the module
/// docs). For the t family, tau depends on `rho` only.
///
/// # Errors
///
/// `params` outside the family's domain.
pub fn param_to_tau(family: Family, params: &[f64]) -> Result<f64, CopulaError> {
    validate_params(family, params)?;
    Ok(match family {
        Family::Gaussian | Family::StudentT => 2.0 * params[0].asin() / core::f64::consts::PI,
        Family::Clayton => params[0] / (params[0] + 2.0),
        Family::Gumbel => (params[0] - 1.0) / params[0],
        Family::Frank => tau_frank(params[0]),
    })
}

/// The dependence parameter implied by a Kendall tau (the inverse of
/// [`param_to_tau`]): `rho = sin(pi tau / 2)` for the elliptical
/// families (for the t family this pins `rho` only — `nu` is not
/// identified by tau; [`crate::copula_fit`] profiles it by MLE),
/// `theta = 2 tau / (1 - tau)` for Clayton, `theta = 1 / (1 - tau)` for
/// Gumbel, and a bisection root of the exact Debye-form tau for Frank
/// (to ~1e-14 in log-theta).
///
/// # Errors
///
/// [`CopulaError::TauOutOfRange`] outside the family's invertible range
/// (Clayton needs `0 < tau < 1`, Gumbel `0 <= tau < 1`, Frank
/// `tau != 0` and `|tau| < tau(700) ~ 0.99433`, elliptical
/// `|tau| < 1`); non-finite tau.
pub fn tau_to_param(family: Family, tau: f64) -> Result<f64, CopulaError> {
    let fam = family.name();
    if !tau.is_finite() {
        return Err(CopulaError::TauOutOfRange {
            family: fam,
            tau,
            requirement: "tau must be finite",
        });
    }
    match family {
        Family::Gaussian | Family::StudentT => {
            if tau.abs() >= 1.0 {
                return Err(CopulaError::TauOutOfRange {
                    family: fam,
                    tau,
                    requirement: "-1 < tau < 1",
                });
            }
            Ok((core::f64::consts::PI * tau / 2.0).sin())
        }
        Family::Clayton => {
            if !(tau > 0.0 && tau < 1.0) {
                return Err(CopulaError::TauOutOfRange {
                    family: fam,
                    tau,
                    requirement: "0 < tau < 1 (positive dependence only)",
                });
            }
            Ok(2.0 * tau / (1.0 - tau))
        }
        Family::Gumbel => {
            if !(0.0..1.0).contains(&tau) {
                return Err(CopulaError::TauOutOfRange {
                    family: fam,
                    tau,
                    requirement: "0 <= tau < 1 (positive dependence only)",
                });
            }
            Ok(1.0 / (1.0 - tau))
        }
        Family::Frank => {
            if tau == 0.0 {
                return Err(CopulaError::TauOutOfRange {
                    family: fam,
                    tau,
                    requirement: "tau != 0 (theta = 0 is the independence limit)",
                });
            }
            let tau_max = tau_frank(FRANK_THETA_MAX);
            if tau.abs() >= tau_max {
                return Err(CopulaError::TauOutOfRange {
                    family: fam,
                    tau,
                    requirement: "|tau| < 0.99433 (theta <= 700)",
                });
            }
            let a = tau.abs();
            // tau(theta) is strictly increasing in theta > 0; below the
            // bracket floor the map is its exact linear limit theta = 9 tau.
            if a < tau_frank(1e-10) {
                return Ok(9.0 * tau);
            }
            // Bisection in ln(theta): 100 halvings of a 37-unit log
            // interval land at full double precision.
            let mut lo = (1e-10_f64).ln();
            let mut hi = FRANK_THETA_MAX.ln();
            for _ in 0..100 {
                let mid = 0.5 * (lo + hi);
                if tau_frank(mid.exp()) < a {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            let th = (0.5 * (lo + hi)).exp();
            Ok(if tau < 0.0 { -th } else { th })
        }
    }
}

/// Closed-form lower/upper tail-dependence coefficients `(lambda_L,
/// lambda_U)` at `params` (Joe 2014, ch. 4):
///
/// * Gaussian: `(0, 0)` — *no* tail dependence at any `|rho| < 1`, the
///   classic reason a Gaussian copula understates joint crashes;
/// * t: `lambda = 2 t_{nu+1}(-sqrt((nu+1)(1-rho)/(1+rho)))` in both
///   tails (Demarta-McNeil 2005) — positive even at `rho = 0`;
/// * Clayton: `(2^{-1/theta}, 0)`; Gumbel: `(0, 2 - 2^{1/theta})`;
///   Frank: `(0, 0)`.
///
/// (statsmodels 0.14.6 `StudentTCopula.dependence_tail` mis-evaluates
/// the t formula through an operator-precedence slip — documented in the
/// fixture generator, where the closed form used here is verified by the
/// numeric copula limit instead.)
///
/// # Errors
///
/// `params` outside the family's domain.
pub fn tail_dependence(family: Family, params: &[f64]) -> Result<(f64, f64), CopulaError> {
    validate_params(family, params)?;
    Ok(match family {
        Family::Gaussian | Family::Frank => (0.0, 0.0),
        Family::StudentT => {
            let (rho, nu) = (params[0], params[1]);
            let arg = -((nu + 1.0) * (1.0 - rho) / (1.0 + rho)).sqrt();
            let lam = match StudentT::new(nu + 1.0) {
                Ok(t) => 2.0 * t.cdf(arg),
                Err(_) => f64::NAN,
            };
            (lam, lam)
        }
        Family::Clayton => ((2.0_f64).powf(-1.0 / params[0]), 0.0),
        Family::Gumbel => (0.0, 2.0 - (2.0_f64).powf(1.0 / params[0])),
    })
}

/// The exact standard-normal CDF used throughout (re-exported for the
/// property tests' independence checks).
#[doc(hidden)]
pub fn _norm_cdf(x: f64) -> f64 {
    norm_cdf(x)
}

/// The Debye function `D1` (re-exported for the property tests).
#[doc(hidden)]
pub fn _debye_1(x: f64) -> f64 {
    debye_1(x)
}
