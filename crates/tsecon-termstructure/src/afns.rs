//! Arbitrage-free Nelson-Siegel (AFNS): the Christensen, Diebold & Rudebusch
//! (2011) closed-form yield-adjustment term and an AFNS curve fit.
//!
//! The plain Nelson-Siegel curve ([`crate::fit_nelson_siegel`]) is a
//! *reduced-form* factor model: nothing forces its factor loadings to be
//! consistent with the no-arbitrage restrictions of a dynamic term-structure
//! model. Christensen, Diebold & Rudebusch (2011) show that keeping the three
//! Nelson-Siegel factor loadings **unchanged** and adding a single
//! deterministic, maturity-dependent **yield-adjustment term** `-A(tau)/tau`
//! makes the curve arbitrage-free:
//!
//! ```text
//! y(tau) = L
//!        + S * (1 - e^{-lam tau}) / (lam tau)
//!        + C * [ (1 - e^{-lam tau}) / (lam tau) - e^{-lam tau} ]
//!        - A(tau) / tau.
//! ```
//!
//! The level/slope/curvature loadings are exactly the Nelson-Siegel ones
//! ([`crate::nelson_siegel_loadings`]); only the extra `-A(tau)/tau` term is
//! new. For the **independent-factor** AFNS — a diagonal factor-volatility
//! matrix `Sigma = diag(sigma_11, sigma_22, sigma_33)` — the adjustment has the
//! documented closed form (CDR 2011)
//!
//! ```text
//! A(tau)/tau =
//!     sigma_11^2 * ( tau^2 / 6 )
//!
//!   + sigma_22^2 * [ 1/(2 lam^2)
//!                    - (1 - e^{-lam tau}) / (lam^3 tau)
//!                    + (1 - e^{-2 lam tau}) / (4 lam^3 tau) ]
//!
//!   + sigma_33^2 * [ 1/(2 lam^2)
//!                    + e^{-lam tau} / lam^2
//!                    - (tau e^{-2 lam tau}) / (4 lam)
//!                    - 3 e^{-2 lam tau} / (4 lam^2)
//!                    - 2 (1 - e^{-lam tau}) / (lam^3 tau)
//!                    + 5 (1 - e^{-2 lam tau}) / (8 lam^3 tau) ].
//! ```
//!
//! `A(tau)/tau` is non-negative; through the `sigma_11^2 tau^2/6` term it grows
//! without bound in maturity, so the *signed* adjustment `-A(tau)/tau` added to
//! the curve is negative and its magnitude grows with maturity — the
//! arbitrage-free concavity/convexity effect that pulls long yields down
//! relative to the reduced-form Nelson-Siegel curve. As `Sigma -> 0` the
//! adjustment vanishes and AFNS nests plain Nelson-Siegel exactly.
//!
//! ## References
//!
//! - Christensen, J. H. E., Diebold, F. X., & Rudebusch, G. D. (2011). "The
//!   affine arbitrage-free class of Nelson-Siegel term structure models."
//!   *Journal of Econometrics*, 164(1), 4-20.

use crate::error::TermStructureError;
use crate::fit::fit_nelson_siegel;
use crate::loadings::{check_lambda, check_maturities};
use crate::NsFit;

/// Validate the diagonal factor-volatility triple: each entry finite and
/// non-negative (`0` is allowed and yields the plain Nelson-Siegel curve).
fn check_sigma(sigma_diag: [f64; 3]) -> Result<(), TermStructureError> {
    for (index, &s) in sigma_diag.iter().enumerate() {
        if !s.is_finite() || s < 0.0 {
            return Err(TermStructureError::InvalidSigma { index, value: s });
        }
    }
    Ok(())
}

/// The independent-factor AFNS yield-adjustment term `A(tau)/tau` at a single
/// maturity `tau`, decay `lambda`, and diagonal volatilities.
///
/// Transcribed verbatim from the Christensen-Diebold-Rudebusch (2011) closed
/// form (see the module docs). Returns the **positive** term `A(tau)/tau`; the
/// signed adjustment added to the curve is its negation.
fn c_over_tau_scalar(tau: f64, lambda: f64, sigma_diag: [f64; 3]) -> f64 {
    let [s11, s22, s33] = sigma_diag;
    let lam = lambda;
    let e1 = (-lam * tau).exp();
    let e2 = (-2.0 * lam * tau).exp();
    let lam2 = lam * lam;
    let lam3 = lam2 * lam;

    // sigma_11^2 term: tau^2 / 6.
    let term11 = tau * tau / 6.0;

    // sigma_22^2 term.
    let term22 = 1.0 / (2.0 * lam2) - (1.0 - e1) / (lam3 * tau) + (1.0 - e2) / (4.0 * lam3 * tau);

    // sigma_33^2 term (the full CDR (2011) independent-factor expression).
    let term33 = 1.0 / (2.0 * lam2) + e1 / lam2
        - (tau * e2) / (4.0 * lam)
        - 3.0 * e2 / (4.0 * lam2)
        - 2.0 * (1.0 - e1) / (lam3 * tau)
        + 5.0 * (1.0 - e2) / (8.0 * lam3 * tau);

    s11 * s11 * term11 + s22 * s22 * term22 + s33 * s33 * term33
}

/// The AFNS **signed yield-adjustment** term `-A(tau)/tau` on a maturity grid.
///
/// Computes the Christensen-Diebold-Rudebusch (2011) independent-factor
/// yield-adjustment closed form (module docs) at each maturity and returns the
/// signed adjustment `-A(tau)/tau` that is *added* to the reduced-form
/// Nelson-Siegel curve to make it arbitrage-free. Because `A(tau)/tau >= 0`,
/// every returned entry is `<= 0`, and — through the `sigma_11^2 tau^2/6` term
/// — its magnitude grows with maturity. When `sigma_diag = [0, 0, 0]` the
/// adjustment is identically zero (AFNS nests Nelson-Siegel).
///
/// `sigma_diag = [sigma_11, sigma_22, sigma_33]` are the diagonal entries of
/// the factor-volatility matrix, in the same time units as `lambda` and the
/// maturities.
///
/// # Errors
///
/// [`TermStructureError::EmptyMaturities`] for an empty grid,
/// [`TermStructureError::InvalidMaturity`] for a non-positive/non-finite
/// maturity, [`TermStructureError::InvalidLambda`] for a
/// non-positive/non-finite `lambda`, and [`TermStructureError::InvalidSigma`]
/// for a negative or non-finite volatility.
///
/// # Example
///
/// ```
/// use tsecon_termstructure::afns_yield_adjustment;
/// let maturities = [1.0, 3.0, 5.0, 10.0];
/// let adj = afns_yield_adjustment(&maturities, 0.5, [0.01, 0.008, 0.012]).unwrap();
/// // The adjustment is negative and its magnitude grows with maturity.
/// assert!(adj.iter().all(|&a| a <= 0.0));
/// assert!(adj[3].abs() > adj[0].abs());
/// // Sigma -> 0 nests plain Nelson-Siegel (zero adjustment).
/// let zero = afns_yield_adjustment(&maturities, 0.5, [0.0, 0.0, 0.0]).unwrap();
/// assert!(zero.iter().all(|&a| a == 0.0));
/// ```
pub fn afns_yield_adjustment(
    maturities: &[f64],
    lambda: f64,
    sigma_diag: [f64; 3],
) -> Result<Vec<f64>, TermStructureError> {
    check_maturities(maturities)?;
    check_lambda(lambda, "lambda")?;
    check_sigma(sigma_diag)?;

    Ok(maturities
        .iter()
        .map(|&tau| -c_over_tau_scalar(tau, lambda, sigma_diag))
        .collect())
}

/// A fitted arbitrage-free Nelson-Siegel (AFNS) yield curve.
///
/// Produced by [`fit_afns`]. Holds the Nelson-Siegel factor fit, the diagonal
/// factor volatilities used to build the arbitrage-free adjustment, the signed
/// per-maturity adjustment `-A(tau)/tau`, and the arbitrage-free fitted yields
/// (Nelson-Siegel fitted curve plus the adjustment).
#[derive(Debug, Clone, PartialEq)]
pub struct AfnsFit {
    /// The underlying Nelson-Siegel factor fit. Its `factors`
    /// (`[level, slope, curvature]`), `lambda`, and `rsquared` are the
    /// reduced-form factors recovered *after* removing the arbitrage-free
    /// adjustment from the observed yields, so combined with `adjustment` they
    /// reconstruct the observed curve.
    pub ns: NsFit,
    /// The diagonal factor volatilities `[sigma_11, sigma_22, sigma_33]` used
    /// to build the adjustment.
    pub sigma_diag: [f64; 3],
    /// The signed AFNS yield-adjustment `-A(tau)/tau` at each input maturity
    /// (non-positive, magnitude growing with maturity).
    pub adjustment: Vec<f64>,
    /// The arbitrage-free fitted yields, one per input maturity: the
    /// Nelson-Siegel fitted curve plus the [`AfnsFit::adjustment`].
    pub fitted: Vec<f64>,
}

impl AfnsFit {
    /// The recovered `[level, slope, curvature]` factors of the underlying
    /// Nelson-Siegel fit.
    pub fn factors(&self) -> [f64; 3] {
        self.ns.factors
    }

    /// The arbitrage-free fitted yield at an arbitrary maturity `tau`:
    /// the Nelson-Siegel fitted yield plus the AFNS adjustment `-A(tau)/tau`.
    ///
    /// # Errors
    ///
    /// [`TermStructureError::InvalidMaturity`] for a non-positive/non-finite
    /// maturity.
    pub fn yield_at(&self, maturity: f64) -> Result<f64, TermStructureError> {
        let ns = self.ns.yield_at(maturity)?;
        Ok(ns - c_over_tau_scalar(maturity, self.ns.lambda, self.sigma_diag))
    }
}

/// Fit an arbitrage-free Nelson-Siegel curve at a fixed decay `lambda` and
/// **known** diagonal factor volatilities `sigma_diag`.
///
/// Given the arbitrage-free representation
/// `y(tau) = NS(tau) - A(tau)/tau`, the Nelson-Siegel part is recovered by
/// adding the adjustment back to the observed yields and running the ordinary
/// cross-sectional [`fit_nelson_siegel`] on `y + A(tau)/tau`. The fitted
/// factors are therefore consistent with the arbitrage-free structure, and the
/// returned [`AfnsFit::fitted`] curve — the Nelson-Siegel fit *plus* the signed
/// adjustment `-A(tau)/tau` — reconstructs the observed yields.
///
/// The three Nelson-Siegel factor loadings are reused unchanged (CDR 2011); the
/// only difference from [`fit_nelson_siegel`] is the deterministic
/// arbitrage-free adjustment. When `sigma_diag = [0, 0, 0]` the adjustment is
/// zero and this is exactly the plain Nelson-Siegel fit.
///
/// # Errors
///
/// The validation errors of [`afns_yield_adjustment`]
/// ([`TermStructureError::EmptyMaturities`],
/// [`TermStructureError::InvalidMaturity`],
/// [`TermStructureError::InvalidLambda`],
/// [`TermStructureError::InvalidSigma`]) and any fit error from
/// [`fit_nelson_siegel`] ([`TermStructureError::DimensionMismatch`],
/// [`TermStructureError::Underdetermined`], [`TermStructureError::NonFinite`],
/// [`TermStructureError::SingularDesign`]).
///
/// # Example
///
/// ```
/// use tsecon_termstructure::fit_afns;
/// let maturities = [0.25, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0];
/// let yields = [4.10, 3.99, 4.02, 4.09, 4.25, 4.31, 4.43];
/// let fit = fit_afns(&maturities, &yields, 0.5, [0.01, 0.008, 0.012]).unwrap();
/// // The arbitrage-free curve reconstructs the observed yields.
/// for (f, y) in fit.fitted.iter().zip(yields.iter()) {
///     assert!((f - y).abs() < 0.25);
/// }
/// // The adjustment is non-positive.
/// assert!(fit.adjustment.iter().all(|&a| a <= 0.0));
/// ```
pub fn fit_afns(
    maturities: &[f64],
    yields: &[f64],
    lambda: f64,
    sigma_diag: [f64; 3],
) -> Result<AfnsFit, TermStructureError> {
    // Validate sigma up front (maturities/lambda/dimension are validated by the
    // Nelson-Siegel fit and the adjustment below).
    check_sigma(sigma_diag)?;

    let adjustment = afns_yield_adjustment(maturities, lambda, sigma_diag)?;
    if yields.len() != maturities.len() {
        return Err(TermStructureError::DimensionMismatch {
            what: "AFNS yields vs maturities",
            expected: maturities.len(),
            got: yields.len(),
        });
    }

    // Remove the arbitrage-free adjustment from the observed yields, then fit
    // the reduced-form Nelson-Siegel factors: y = NS - A/tau  =>  NS = y + A/tau
    // = y - adjustment (adjustment = -A/tau).
    let ns_target: Vec<f64> = yields
        .iter()
        .zip(adjustment.iter())
        .map(|(&y, &a)| y - a)
        .collect();
    let ns = fit_nelson_siegel(maturities, &ns_target, lambda)?;

    // Arbitrage-free fitted yields: the Nelson-Siegel fitted curve plus the
    // signed adjustment.
    let ns_fitted = ns.fitted(maturities)?;
    let fitted: Vec<f64> = ns_fitted
        .iter()
        .zip(adjustment.iter())
        .map(|(&f, &a)| f + a)
        .collect();

    Ok(AfnsFit {
        ns,
        sigma_diag,
        adjustment,
        fitted,
    })
}
