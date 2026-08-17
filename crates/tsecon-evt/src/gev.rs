//! Generalized extreme value fitting on block maxima, with return
//! levels.
//!
//! ## Model
//!
//! Block maxima `m` are modeled as GEV(`xi`, `mu`, `sigma`) with CDF
//!
//! ```text
//! F(m) = exp( -(1 + xi (m - mu) / sigma)^(-1/xi) ),   1 + xi (m - mu) / sigma > 0,
//! ```
//!
//! the Fisher-Tippett-Gnedenko limit family (Coles 2001, ch. 3);
//! `xi -> 0` is the Gumbel limit `exp(-exp(-(m - mu)/sigma))`. **Sign
//! convention**: this `xi` is the tail index (positive = heavy Fréchet
//! tail); scipy's `genextreme` shape is `c = -xi` (verified numerically in
//! the fixture generator).
//!
//! ## Estimation
//!
//! MLE over the working space `(xi, mu, ln sigma)` (support constraint by
//! infinite barrier): Gumbel moment starting values
//! (`sigma_0 = s sqrt(6)/pi`, `mu_0 = mean - gamma sigma_0`) with a small
//! shape-candidate grid, BFGS, then a tight Nelder-Mead polish. Standard
//! errors are observed-information in the original `(xi, mu, sigma)`
//! parameterization. The `xi <= -0.5` irregularity (Smith 1985) is
//! reported through [`GevFit::se_valid`] exactly as in [`crate::gpd_fit`];
//! for `xi <= -1` the MLE does not exist and the reported point is the
//! best found.
//!
//! ## Return levels
//!
//! The `T`-block return level (exceeded once every `T` blocks on
//! average) is the `q = 1 - 1/T` GEV quantile
//!
//! ```text
//! z_T = mu + (sigma / xi) * [ (-ln q)^(-xi) - 1 ]        (xi != 0)
//! z_T = mu - sigma * ln(-ln q)                           (xi  = 0)
//! ```
//!
//! (Coles 2001, eq. 3.4). With annual blocks, `z_100` is the "100-year
//! event".

use crate::common::{check_series, ln1p_over_xi, observed_info_ses};
use crate::error::EvtError;
use crate::gpd::refine;

/// Minimum number of block maxima required by [`gev_fit`].
pub const MIN_MAXIMA: usize = 10;

/// Euler-Mascheroni constant (Gumbel mean is `mu + gamma * sigma`).
const EULER_GAMMA: f64 = 0.577_215_664_901_532_9;

/// Division-by-zero guard for the return-level formula (see
/// `crate::gpd`'s tail formulas — same role).
const XI_EPS: f64 = 1e-12;

/// Result of a GEV block-maxima fit — see [`gev_fit`].
#[derive(Debug, Clone, PartialEq)]
pub struct GevFit {
    /// Fitted GEV shape (tail index; scipy's `genextreme` `c` is `-xi`).
    pub xi: f64,
    /// Fitted GEV location.
    pub mu: f64,
    /// Fitted GEV scale (`> 0`).
    pub sigma: f64,
    /// Observed-information standard error of `xi` (NaN when the Hessian
    /// failed; see [`GevFit::se_valid`]).
    pub se_xi: f64,
    /// Observed-information standard error of `mu`.
    pub se_mu: f64,
    /// Observed-information standard error of `sigma`.
    pub se_sigma: f64,
    /// Whether the standard errors are certified: positive-definite
    /// observed information **and** `xi > -0.5` (Smith 1985 regularity).
    pub se_valid: bool,
    /// GEV log-likelihood of the maxima at the fitted parameters —
    /// comparable to
    /// `scipy.stats.genextreme.logpdf(m, -xi, mu, sigma).sum()`.
    pub loglik: f64,
    /// Whether the final optimizer run reported convergence.
    pub converged: bool,
    /// Number of block maxima the fit used.
    pub n_maxima: usize,
    /// The block size, when the maxima were computed from a series
    /// (`None` when pre-computed maxima were supplied).
    pub block_size: Option<usize>,
    /// The return periods the return levels were computed at (echoed).
    pub return_periods: Vec<f64>,
    /// Return levels `z_T` per entry of `return_periods` (the
    /// `1 - 1/T` GEV quantile).
    pub return_levels: Vec<f64>,
}

/// Fits a GEV to block maxima by MLE and computes return levels.
///
/// * `y` — either the block maxima themselves (`block_size = None`) or a
///   raw series to be blocked (`block_size = Some(b)`): consecutive
///   non-overlapping blocks of length `b`, **a trailing partial block is
///   dropped**, and each block contributes its maximum.
/// * `return_periods` — return periods in blocks (each `> 1`), e.g.
///   `[10.0, 50.0, 100.0]`. May be empty.
///
/// # Errors
///
/// [`EvtError::EmptyInput`] / [`EvtError::NonFinite`] on malformed `y`;
/// [`EvtError::InvalidBlockSize`] (`block_size` zero or `> y.len()`);
/// [`EvtError::TooFewMaxima`] (fewer than [`MIN_MAXIMA`] maxima);
/// [`EvtError::Degenerate`] (numerically constant maxima);
/// [`EvtError::InvalidReturnPeriod`];
/// [`EvtError::NoAdmissibleStart`] / [`EvtError::Optim`] from the MLE.
pub fn gev_fit(
    y: &[f64],
    block_size: Option<usize>,
    return_periods: &[f64],
) -> Result<GevFit, EvtError> {
    check_series(y, "y")?;
    let maxima: Vec<f64> = match block_size {
        None => y.to_vec(),
        Some(b) => {
            if b == 0 || b > y.len() {
                return Err(EvtError::InvalidBlockSize {
                    block_size: b,
                    n: y.len(),
                });
            }
            y.chunks_exact(b)
                .map(|c| c.iter().cloned().fold(f64::NEG_INFINITY, f64::max))
                .collect()
        }
    };
    let nm = maxima.len();
    if nm < MIN_MAXIMA {
        return Err(EvtError::TooFewMaxima {
            n_maxima: nm,
            min: MIN_MAXIMA,
        });
    }
    let m_max = maxima.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let m_min = maxima.iter().cloned().fold(f64::INFINITY, f64::min);
    if m_max - m_min <= 8.0 * f64::EPSILON * m_max.abs().max(m_min.abs()) {
        return Err(EvtError::Degenerate {
            what: "the block maxima",
        });
    }
    for &t in return_periods {
        if !(t.is_finite() && t > 1.0) {
            return Err(EvtError::InvalidReturnPeriod { t });
        }
    }

    // ------------------------------- Gumbel moment starts + shape grid
    let mean = maxima.iter().sum::<f64>() / nm as f64;
    let sd =
        (maxima.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>() / (nm as f64 - 1.0)).sqrt();
    let sigma0 = sd * (6.0_f64).sqrt() / core::f64::consts::PI;
    let mu0 = mean - EULER_GAMMA * sigma0;
    let mut start: Option<(f64, [f64; 3])> = None;
    for &xi0 in &[-0.2, -0.05, 0.05, 0.1, 0.3] {
        for &scale_mult in &[1.0, 2.0] {
            let s0 = sigma0 * scale_mult;
            let f = gev_nll(&maxima, xi0, mu0, s0);
            if f.is_finite() && start.as_ref().is_none_or(|(bf, _)| f < *bf) {
                start = Some((f, [xi0, mu0, s0.ln()]));
            }
        }
    }
    let (_, w0) = start.ok_or(EvtError::NoAdmissibleStart { what: "gev_fit" })?;

    // ------------------------------------------------------------- MLE
    let (wb, converged) = refine(|w: &[f64]| gev_nll(&maxima, w[0], w[1], w[2].exp()), &w0)?;
    let xi = wb[0];
    let mu = wb[1];
    let sigma = wb[2].exp();
    let loglik = -gev_nll(&maxima, xi, mu, sigma);

    // -------------------------------------- observed-information SEs
    let ses = observed_info_ses(
        |p: &[f64]| gev_nll(&maxima, p[0], p[1], p[2]),
        &[xi, mu, sigma],
        &[xi.abs().max(0.1), sigma, sigma],
    );
    let (se_xi, se_mu, se_sigma) = match &ses {
        Some(s) => (s[0], s[1], s[2]),
        None => (f64::NAN, f64::NAN, f64::NAN),
    };
    let se_valid = ses.is_some() && xi > -0.5;

    // --------------------------------------------------- return levels
    let return_levels = return_periods
        .iter()
        .map(|&t| {
            let q = 1.0 - 1.0 / t;
            let lnl = (-(q.ln())).ln();
            if xi.abs() < XI_EPS {
                mu - sigma * lnl
            } else {
                mu + sigma / xi * (-xi * lnl).exp_m1()
            }
        })
        .collect();

    Ok(GevFit {
        xi,
        mu,
        sigma,
        se_xi,
        se_mu,
        se_sigma,
        se_valid,
        loglik,
        converged,
        n_maxima: nm,
        block_size,
        return_periods: return_periods.to_vec(),
        return_levels,
    })
}

/// GEV log-likelihood of `maxima` at (`xi`, `mu`, `sigma`) — the quantity
/// [`gev_fit`] maximizes, exposed for model comparison and testing.
///
/// Returns `-infinity` when any observation lies outside the support
/// (`1 + xi (m - mu) / sigma <= 0`), matching scipy's `logpdf`
/// convention.
///
/// # Errors
///
/// [`EvtError::EmptyInput`] / [`EvtError::NonFinite`] on malformed input
/// or non-finite `xi` / `mu`; [`EvtError::InvalidScale`] unless
/// `sigma > 0` and finite.
pub fn gev_loglik(maxima: &[f64], xi: f64, mu: f64, sigma: f64) -> Result<f64, EvtError> {
    check_series(maxima, "maxima")?;
    if !xi.is_finite() {
        return Err(EvtError::NonFinite {
            what: "xi",
            index: 0,
        });
    }
    if !mu.is_finite() {
        return Err(EvtError::NonFinite {
            what: "mu",
            index: 0,
        });
    }
    if !(sigma > 0.0 && sigma.is_finite()) {
        return Err(EvtError::InvalidScale { scale: sigma });
    }
    Ok(-gev_nll(maxima, xi, mu, sigma))
}

/// Negative GEV log-likelihood; `+infinity` outside the admissible
/// region. Written through the shared kernel `a_i = ln(1 + xi t_i)/xi`
/// (with its documented Gumbel-limit branch) as
/// `nll = sum_i [ ln sigma + (1 + xi) a_i + exp(-a_i) ]`.
pub(crate) fn gev_nll(m: &[f64], xi: f64, mu: f64, sigma: f64) -> f64 {
    if !sigma.is_finite() || sigma <= 0.0 || !xi.is_finite() || !mu.is_finite() {
        return f64::INFINITY;
    }
    let ln_sigma = sigma.ln();
    let mut acc = 0.0;
    for &mi in m {
        let t = (mi - mu) / sigma;
        let a = ln1p_over_xi(xi, t);
        if !a.is_finite() {
            return f64::INFINITY;
        }
        acc += ln_sigma + (1.0 + xi) * a + (-a).exp();
    }
    if acc.is_finite() {
        acc
    } else {
        f64::INFINITY
    }
}
