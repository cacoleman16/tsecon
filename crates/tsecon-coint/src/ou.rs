//! Ornstein-Uhlenbeck mean-reversion utilities for spreads ([`ou_fit`],
//! [`spread_zscore`]) — the univariate follow-on to the cointegration
//! layer: once [`crate::engle_granger()`] (or a [`crate::johansen()`] +
//! [`crate::fit_vecm_det`] pass) has produced a stationary spread, this
//! module quantifies *how fast* it mean-reverts and *how far* it currently
//! sits from equilibrium, the two numbers a statistical-arbitrage
//! workflow actually trades on.
//!
//! ## The model and its exact discretization
//!
//! The Ornstein-Uhlenbeck (Vasicek) process
//!
//! ```text
//! dX_t = kappa (mu - X_t) dt + sigma dW_t ,        kappa > 0
//! ```
//!
//! observed at a fixed step `dt` is **exactly** (not approximately) a
//! Gaussian AR(1):
//!
//! ```text
//! X_{t+1} = c + phi X_t + eps_t ,   eps_t ~ iid N(0, eta2)
//! phi  = exp(-kappa dt)
//! c    = mu (1 - phi)
//! eta2 = sigma^2 (1 - phi^2) / (2 kappa)           (-> sigma^2 dt as kappa -> 0)
//! ```
//!
//! so the continuous-time MLE **is** the discrete AR(1) conditional MLE
//! mapped through this bijection — no iterative optimizer is involved.
//! Conditioning on the first observation `X_0` (the standard choice; the
//! unconditional term would require `kappa > 0` a priori, which is exactly
//! what we refuse to assume), the AR(1) conditional Gaussian likelihood is
//! maximized in closed form by the OLS regression of `X_{t+1}` on
//! `(1, X_t)` over the `n = T - 1` transitions, with the variance MLE
//! `eta2 = RSS / n` (no degrees-of-freedom correction — the MLE, which is
//! also what statsmodels `AutoReg(x, lags=1).fit()` reports as `sigma2`).
//! The inverse mapping recovers the continuous-time parameters:
//!
//! ```text
//! kappa  = -ln(phi) / dt
//! mu     = c / (1 - phi)
//! sigma2 = eta2 * 2 kappa / (1 - phi^2)  =  eta2 * (-2 ln phi) / (dt (1 - phi^2))
//! ```
//!
//! ## Standard errors (delta method from the AR(1) information)
//!
//! The AR(1) conditional-MLE asymptotics give, with `W = [1, x_lag]`,
//! `m = mean(x_lag)`, `S_xx = sum((x_lag - m)^2)`:
//!
//! ```text
//! Var(phi_hat)          = eta2 / S_xx
//! Var(c_hat)            = eta2 (1/n + m^2 / S_xx)
//! Cov(c_hat, phi_hat)   = -eta2 m / S_xx
//! Var(eta2_hat)         = 2 eta2^2 / n        (independent of (c, phi) asymptotically)
//! ```
//!
//! (the Gaussian-MLE information evaluated at the estimate, i.e. the OLS
//! covariance with the *MLE* variance `eta2 = RSS/n`). The delta method
//! maps these through the inverse discretization:
//!
//! ```text
//! SE(kappa) = SE(phi) / (phi dt)                            [kappa = -ln(phi)/dt]
//! SE(mu)^2  = g' V g,  g = [1/(1-phi), c/(1-phi)^2],
//!             V = Cov[(c, phi)]                             [mu = c/(1-phi)]
//! sigma2    = eta2 * a(phi),  a(phi) = -2 ln(phi) / (dt (1 - phi^2))
//! a'(phi)   = (-2/dt) * [ (1-phi^2)/phi + 2 phi ln(phi) ] / (1-phi^2)^2
//! Var(sigma2_hat) = (eta2 a'(phi))^2 Var(phi) + a(phi)^2 Var(eta2)
//! SE(sigma) = SE(sigma2) / (2 sigma)
//! ```
//!
//! ## The half-life confidence interval: level scale, a *measured* choice
//!
//! `half_life = ln(2) / kappa`. The interval reported in
//! [`OuFit::half_life_ci`] maps the symmetric normal (level-scale) kappa
//! interval through the monotone `ln(2)/kappa`:
//!
//! ```text
//! kappa_lo = kappa - z SE(kappa),   kappa_hi = kappa + z SE(kappa)
//! half_life_ci = ( ln 2 / kappa_hi ,  ln 2 / kappa_lo )     if kappa_lo > 0
//!              = ( ln 2 / kappa_hi ,  +inf )                if kappa_lo <= 0
//! ```
//!
//! When `kappa_lo <= 0` the data cannot rule out `kappa <= 0` — no mean
//! reversion at all — at this confidence, and the interval says so by
//! letting its upper endpoint go to `+inf` rather than fabricating a
//! finite bound.
//!
//! **Why level scale (and not the log scale).** The a-priori argument
//! runs the other way — a log-scale interval
//! `kappa * exp(±z SE/kappa)` stays positive by construction and matches
//! the right skew of `ln(2)/kappa_hat` — so *both* constructions were
//! measured in the shipped Monte Carlo
//! (`docs/examples/coverage/experiments/ou_kappa_bias_coverage.py`, 2000
//! reps per cell, `kappa in {5, 2, 0.5, 0.1}` on a 5-year daily and a
//! 20-year monthly grid, nominal 95%). The level-scale interval covers
//! closer to nominal in **every** cell (e.g. daily 5y: 0.94 vs 0.89 at
//! `kappa = 5`, 0.82 vs 0.53 at `kappa = 0.5`, 0.71 vs 0.21 at
//! `kappa = 0.1`). The mechanism is the kappa bias below: `kappa_hat`
//! centers *above* the truth, and a multiplicative interval around an
//! upward-biased center never reaches down to a small true kappa, while
//! the level interval — precisely by crossing zero and conceding "maybe
//! no mean reversion" (its `+inf` branch) — does. Neither construction
//! attains nominal in the slow-reversion cells; that residual
//! under-coverage is driven by the bias itself and is tabulated in the
//! cointegration model card rather than hidden.
//!
//! ## Honesty at and past the unit root, and the known kappa bias
//!
//! * `phi_hat >= 1` (`kappa_hat <= 0`): the fitted AR(1) root is at or
//!   over unity — the spread shows **no mean reversion** in this sample.
//!   This is a legitimate, informative estimate (it is how a
//!   non-cointegrated "spread" announces itself), so [`ou_fit`] does
//!   **not** error: it returns the estimate with
//!   [`OuFit::mean_reverting`]` = false`, `half_life = +inf` (the
//!   deviation never halves in expectation), and `half_life_ci = None` /
//!   `stationary_sd = None` (no stationary distribution exists, and the
//!   Gaussian AR(1) asymptotics the delta method relies on break down at
//!   the unit root, so an interval would be a fabrication). At exactly
//!   `phi = 1`, `mu = c/(1-phi)` is unidentified and is reported as NaN,
//!   and `sigma2` uses its `kappa -> 0` limit `eta2 / dt`.
//! * `phi_hat <= 0`: **refused** ([`CointError::InvalidArgument`]) — no
//!   real `kappa` satisfies `exp(-kappa dt) <= 0`, so the continuous-time
//!   parametrization is genuinely unattainable (the series is
//!   anti-persistent at this sampling interval). The error text says what
//!   to do about it.
//! * **Finite-sample bias in `kappa_hat`** (documented, not hidden): the
//!   AR(1) OLS/MLE slope is biased downward, `E[phi_hat] - phi ~
//!   -(1 + 3 phi)/n` (Kendall 1954), which maps through
//!   `kappa = -ln(phi)/dt` to an *upward* bias in `kappa_hat` of roughly
//!   `(1 + 3 phi)/(n phi dt)` — approximately `4 / (n dt)`, four over the
//!   **time span** of the sample, for persistent spreads (Tang & Chen
//!   2009; Yu 2012). Sampling more finely over the same span does not
//!   remove it; only a longer span does. The estimator ships unadjusted —
//!   the MLE is what the closed form and the cross-checks pin — and the
//!   Monte-Carlo bias table in the model card quantifies it so users can
//!   judge their own span.
//!
//! ## References
//!
//! Uhlenbeck & Ornstein (1930), Physical Review 36; Vasicek (1977), JFE 5;
//! Kendall (1954), Biometrika 41; Tang & Chen (2009), Journal of
//! Econometrics 149 (bias of `kappa_hat` is `O(1/(n dt))`); Yu (2012),
//! Journal of Econometrics 169.

use tsecon_stats::special::inv_norm_cdf;

use crate::error::CointError;

/// The result of [`ou_fit`]: continuous-time Ornstein-Uhlenbeck parameters
/// with delta-method standard errors, the AR(1) discretization they map
/// from, and the derived mean-reversion summaries.
#[derive(Debug, Clone, PartialEq)]
pub struct OuFit {
    /// Mean-reversion speed `kappa` (per unit of time implied by `dt`).
    /// `<= 0` when the fitted AR(1) root is at/over unity — see
    /// [`OuFit::mean_reverting`].
    pub kappa: f64,
    /// Delta-method standard error of `kappa`: `SE(phi) / (phi * dt)`.
    pub kappa_se: f64,
    /// Long-run mean `mu = c / (1 - phi)`. NaN when `phi == 1` exactly
    /// (unidentified at the unit root).
    pub mu: f64,
    /// Delta-method standard error of `mu` (uses the full `(c, phi)`
    /// covariance including their negative correlation).
    pub mu_se: f64,
    /// Diffusion scale `sigma` (per square root of the `dt` time unit).
    pub sigma: f64,
    /// Delta-method standard error of `sigma`.
    pub sigma_se: f64,
    /// `ln(2) / kappa` — expected time for a deviation from `mu` to halve.
    /// `+inf` when `kappa <= 0` (the deviation never halves in
    /// expectation).
    pub half_life: f64,
    /// Delta-method confidence interval for the half-life at
    /// [`OuFit::ci_level`]: the level-scale kappa interval
    /// `kappa ± z SE(kappa)` mapped through `ln(2)/kappa` — the
    /// construction the shipped Monte Carlo measured as covering closer
    /// to nominal than the log-scale alternative in every cell (module
    /// docs). The upper endpoint is `+inf` when the kappa interval
    /// crosses zero (the data cannot rule out "no mean reversion" at
    /// this confidence). `None` when `kappa <= 0` itself: no half-life
    /// exists and the Gaussian asymptotics fail at the unit root.
    pub half_life_ci: Option<(f64, f64)>,
    /// The confidence level `half_life_ci` was built at.
    pub ci_level: f64,
    /// Stationary standard deviation `sigma / sqrt(2 kappa)` — the
    /// denominator of the spread z-score. `None` when `kappa <= 0` (no
    /// stationary distribution exists).
    pub stationary_sd: Option<f64>,
    /// `true` iff `kappa > 0` (equivalently `phi < 1`): the fitted
    /// process actually reverts to `mu`.
    pub mean_reverting: bool,
    /// AR(1) slope of the exact discretization, `phi = exp(-kappa dt)`.
    pub phi: f64,
    /// OLS/MLE standard error of `phi`: `sqrt(eta2 / S_xx)`.
    pub phi_se: f64,
    /// AR(1) intercept `c = mu (1 - phi)`.
    pub c: f64,
    /// OLS/MLE standard error of `c`: `sqrt(eta2 (1/n + m^2 / S_xx))`.
    pub c_se: f64,
    /// AR(1) innovation variance MLE `eta2 = RSS / n` (statsmodels
    /// `AutoReg` `sigma2` convention; no degrees-of-freedom correction).
    pub eta2: f64,
    /// Exact conditional Gaussian log-likelihood at the MLE,
    /// `-(n/2) (ln(2 pi eta2) + 1)`.
    pub loglik: f64,
    /// Number of transitions used, `n = len(x) - 1`.
    pub n_obs: usize,
    /// The observation step the fit was performed at.
    pub dt: f64,
}

impl OuFit {
    /// Z-score of `x` against the fitted stationary distribution:
    /// `(x - mu) / stationary_sd`. Delegates to [`spread_zscore`];
    /// refuses (like it) when the fit is not mean-reverting, because no
    /// stationary distribution exists to score against.
    pub fn zscore(&self, x: &[f64]) -> Result<Vec<f64>, CointError> {
        spread_zscore(x, self.kappa, self.mu, self.sigma)
    }
}

/// First non-finite entry of `x`, as a [`CointError::NonFiniteSeries`]
/// whose `what` carries the OU-appropriate consequence and remedy (the
/// located `NonFinite` variant explains a *cointegration-eigenvalue*
/// consequence that does not apply to an AR(1) fit).
fn check_series_finite(x: &[f64], what: &'static str) -> Result<(), CointError> {
    for (i, v) in x.iter().enumerate() {
        if !v.is_finite() {
            return Err(CointError::NonFiniteSeries { what, index: i });
        }
    }
    Ok(())
}

/// Exact-discretization Gaussian MLE of the Ornstein-Uhlenbeck process
/// `dX = kappa (mu - X) dt + sigma dW` from observations `x` at fixed
/// step `dt`, conditioning on the first observation.
///
/// The estimator is the closed-form AR(1) OLS/MLE (`X_{t+1}` on
/// `(1, X_t)`, variance `RSS/n`) mapped through the exact discretization
/// — see the module docs for the mapping, the delta-method standard
/// errors, and the (Monte-Carlo-vetted, level-scale) construction of the
/// half-life interval at confidence `level` (e.g. `0.95`).
///
/// # Errors
///
/// * fewer than 4 observations (3 transitions: the regression estimates
///   an intercept, a slope, and a variance), non-finite `x`, `dt <= 0`,
///   or `level` outside `(0, 1)` — [`CointError::InvalidArgument`] /
///   [`CointError::NonFinite`];
/// * a constant series (the lagged regressor has zero variance) or an
///   exactly deterministic recursion (`RSS == 0`) — the likelihood is
///   degenerate and nothing honest can be reported;
/// * `phi_hat <= 0` — no real `kappa` produces a non-positive
///   `exp(-kappa dt)`, so the continuous-time parametrization is
///   unattainable (the error text explains the sampling-interval cause).
///
/// A fit with `phi_hat >= 1` (`kappa_hat <= 0`) is **not** an error: it
/// is returned honestly with `mean_reverting = false`,
/// `half_life = +inf`, and `half_life_ci = stationary_sd = None`.
pub fn ou_fit(x: &[f64], dt: f64, level: f64) -> Result<OuFit, CointError> {
    check_series_finite(
        x,
        "the spread series x: the OU fit is an AR(1) regression of x_{t+1} on \
         (1, x_t), so one gap invalidates both transitions it enters — drop or \
         impute missing values first (pandas: s.dropna() or s.interpolate())",
    )?;
    if x.len() < 4 {
        return Err(CointError::InvalidArgument {
            what: "ou_fit needs at least 4 observations (3 transitions): the AR(1) \
                   regression behind the exact-discretization MLE estimates an \
                   intercept, a slope, and an innovation variance",
        });
    }
    if !(dt.is_finite() && dt > 0.0) {
        return Err(CointError::InvalidArgument {
            what: "dt must be a finite positive number: it is the time between \
                   consecutive observations in the units kappa is quoted in \
                   (e.g. 1.0/252.0 for daily data and annualized kappa)",
        });
    }
    if !(level.is_finite() && level > 0.0 && level < 1.0) {
        return Err(CointError::InvalidArgument {
            what: "level must lie strictly between 0 and 1 (e.g. 0.95 for a 95% \
                   half-life confidence interval)",
        });
    }

    // ---- AR(1) OLS/MLE over the n = T - 1 transitions (two-pass, centered).
    let n = x.len() - 1;
    let nf = n as f64;
    let lag = &x[..n];
    let lead = &x[1..];
    let m_lag = lag.iter().sum::<f64>() / nf;
    let m_lead = lead.iter().sum::<f64>() / nf;
    let mut sxx = 0.0;
    let mut sxy = 0.0;
    for t in 0..n {
        let d = lag[t] - m_lag;
        sxx += d * d;
        sxy += d * (lead[t] - m_lead);
    }
    if sxx <= 0.0 {
        return Err(CointError::Singular {
            what: "the lagged spread has zero variance (the series is constant over \
                   the sample), so the AR(1) slope — and with it every OU parameter \
                   — is unidentified",
        });
    }
    let phi = sxy / sxx;
    let c = m_lead - phi * m_lag;
    let mut rss = 0.0;
    for t in 0..n {
        let r = lead[t] - c - phi * lag[t];
        rss += r * r;
    }
    let eta2 = rss / nf;
    if eta2 <= 0.0 {
        return Err(CointError::InvalidArgument {
            what: "the AR(1) residuals are exactly zero (the series follows a \
                   deterministic recursion), so the Gaussian likelihood is \
                   degenerate and sigma is unidentified",
        });
    }
    let loglik = -0.5 * nf * ((2.0 * core::f64::consts::PI * eta2).ln() + 1.0);

    // AR(1) information: OLS covariance evaluated with the MLE variance.
    let var_phi = eta2 / sxx;
    let var_c = eta2 * (1.0 / nf + m_lag * m_lag / sxx);
    let cov_c_phi = -eta2 * m_lag / sxx;
    let var_eta2 = 2.0 * eta2 * eta2 / nf;
    let phi_se = var_phi.sqrt();
    let c_se = var_c.sqrt();

    if phi <= 0.0 {
        return Err(CointError::InvalidArgument {
            what: "the fitted AR(1) coefficient is <= 0, which exp(-kappa*dt) cannot \
                   equal for any real kappa: the series is anti-persistent at this \
                   sampling interval (consecutive deviations flip sign), so no \
                   Ornstein-Uhlenbeck process observed at step dt can generate it. \
                   This typically means dt is too coarse relative to the mean \
                   reversion (kappa*dt >> 1) or the spread is close to white noise; \
                   sample more finely, or model the discrete-time AR(1) directly",
        });
    }

    // ---- Inverse of the exact discretization, and the delta method.
    let kappa = if phi == 1.0 { 0.0 } else { -phi.ln() / dt };
    let mu = c / (1.0 - phi); // phi == 1 -> c/0 = +-inf or NaN; normalized below
    let (mu, mu_se) = if phi == 1.0 {
        (f64::NAN, f64::NAN) // unidentified at the unit root
    } else {
        let g1 = 1.0 / (1.0 - phi);
        let g2 = c / ((1.0 - phi) * (1.0 - phi));
        let v = g1 * g1 * var_c + g2 * g2 * var_phi + 2.0 * g1 * g2 * cov_c_phi;
        (mu, v.max(0.0).sqrt())
    };
    // sigma2 = eta2 * a(phi), a(phi) = -2 ln(phi) / (dt (1 - phi^2)),
    // with the kappa -> 0 limit a(1) = 1/dt.
    let (a, a_prime) = if phi == 1.0 {
        (1.0 / dt, f64::NAN)
    } else {
        let om = 1.0 - phi * phi;
        let a = -2.0 * phi.ln() / (dt * om);
        let a_prime = (-2.0 / dt) * ((om / phi + 2.0 * phi * phi.ln()) / (om * om));
        (a, a_prime)
    };
    let sigma2 = eta2 * a;
    let sigma = sigma2.sqrt();
    let sigma_se = if phi == 1.0 {
        f64::NAN
    } else {
        let var_sigma2 = (eta2 * a_prime) * (eta2 * a_prime) * var_phi + a * a * var_eta2;
        var_sigma2.max(0.0).sqrt() / (2.0 * sigma)
    };
    let kappa_se = phi_se / (phi * dt);

    let mean_reverting = kappa > 0.0;
    let (half_life, half_life_ci, stationary_sd) = if mean_reverting {
        let z = inv_norm_cdf(0.5 + level / 2.0)?;
        let kappa_lo = kappa - z * kappa_se;
        let kappa_hi = kappa + z * kappa_se;
        let hl_hi = if kappa_lo > 0.0 {
            core::f64::consts::LN_2 / kappa_lo
        } else {
            // The kappa interval crosses zero: "no mean reversion" cannot
            // be ruled out at this confidence, so no finite upper bound
            // on the half-life exists.
            f64::INFINITY
        };
        (
            core::f64::consts::LN_2 / kappa,
            Some((core::f64::consts::LN_2 / kappa_hi, hl_hi)),
            Some((sigma2 / (2.0 * kappa)).sqrt()),
        )
    } else {
        (f64::INFINITY, None, None)
    };

    Ok(OuFit {
        kappa,
        kappa_se,
        mu,
        mu_se,
        sigma,
        sigma_se,
        half_life,
        half_life_ci,
        ci_level: level,
        stationary_sd,
        mean_reverting,
        phi,
        phi_se,
        c,
        c_se,
        eta2,
        loglik,
        n_obs: n,
        dt,
    })
}

/// Z-score of a spread against the stationary distribution of an
/// Ornstein-Uhlenbeck process with parameters `(kappa, mu, sigma)`:
/// `z_t = (x_t - mu) / (sigma / sqrt(2 kappa))`.
///
/// The stationary law of the OU process is
/// `N(mu, sigma^2 / (2 kappa))`, so `z_t` measures how many long-run
/// standard deviations the spread currently sits from its equilibrium —
/// the entry/exit signal of a mean-reversion strategy.
///
/// # Errors
///
/// `kappa <= 0` is refused ([`CointError::InvalidArgument`]): a
/// non-mean-reverting process has **no** stationary distribution, so the
/// z-score does not exist — the error is the honest answer, not a
/// convenience gap. Non-finite `x`/parameters and `sigma <= 0` are
/// refused likewise.
pub fn spread_zscore(x: &[f64], kappa: f64, mu: f64, sigma: f64) -> Result<Vec<f64>, CointError> {
    check_series_finite(
        x,
        "the spread series x: the z-score (x_t - mu) / stationary_sd is undefined \
         at a missing value — drop or impute it first (pandas: s.dropna() or \
         s.interpolate())",
    )?;
    if x.is_empty() {
        return Err(CointError::InvalidArgument {
            what: "spread_zscore needs at least one observation",
        });
    }
    if !kappa.is_finite() {
        return Err(CointError::InvalidArgument {
            what: "spread_zscore requires a finite kappa: the mean-reversion speed \
                   sets the stationary standard deviation sigma / sqrt(2 kappa), \
                   and NaN or infinity gives no stationary distribution to score \
                   against — pass the kappa from ou_fit",
        });
    }
    if kappa <= 0.0 {
        return Err(CointError::InvalidArgument {
            what: "spread_zscore requires kappa > 0: a process with kappa <= 0 does \
                   not mean-revert and has no stationary distribution, so the \
                   z-score is undefined. If ou_fit reported mean_reverting = False \
                   the spread showed no mean reversion in this sample — re-check \
                   the cointegrating relation (engle_granger / johansen) before \
                   trading it as one",
        });
    }
    if !mu.is_finite() {
        return Err(CointError::InvalidArgument {
            what: "spread_zscore requires a finite mu (the long-run mean)",
        });
    }
    if !sigma.is_finite() {
        return Err(CointError::InvalidArgument {
            what: "spread_zscore requires a finite sigma (the diffusion scale): NaN \
                   or infinity gives no stationary distribution to score against",
        });
    }
    if sigma <= 0.0 {
        return Err(CointError::InvalidArgument {
            what: "spread_zscore requires sigma > 0 (the diffusion scale)",
        });
    }
    let sd = sigma / (2.0 * kappa).sqrt();
    Ok(x.iter().map(|v| (v - mu) / sd).collect())
}
