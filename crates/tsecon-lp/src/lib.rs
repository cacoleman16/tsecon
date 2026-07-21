//! # tsecon-lp — local projections (Jordà 2005)
//!
//! Horizon-by-horizon impulse-response estimation by local projections: for
//! each horizon `h` a separate regression of the (possibly cumulated)
//! outcome `h` periods ahead on the impulse and a set of controls, with the
//! sequence of impulse coefficients tracing out the impulse-response
//! function. This is the tsecon library's dedicated LP module (ROADMAP §7,
//! Tier 1); it owns the LP-specific inference and delegates all robust
//! covariance arithmetic to the single HAC engine in
//! [`tsecon_hac`].
//!
//! ## What this crate provides
//!
//! - [`lp`]: single-impulse LP with two inference paths — **lag-augmented
//!   HC1 (the default)** and Newey-West HAC — plus optional cumulative
//!   (Ramey-Zubairy) responses.
//! - [`lp_iv`]: just-identified LP-IV (external-instrument local
//!   projections) with a linearmodels-convention kernel-HAC covariance and a
//!   first-stage effective-F diagnostic.
//! - [`lp_multiplier`]: the Ramey-Zubairy (2018) one-step **integral
//!   multiplier** — cumulated outcome on cumulated impulse, instrumented.
//! - [`lp_state`]: state-dependent LP with the impulse and controls
//!   interacted with the lagged regime indicator (Ramey-Zubairy 2018).
//! - [`smooth_lp`]: **smooth local projections** (Barnichon-Brownlees 2019)
//!   — the IRF path estimated jointly across horizons as a B-spline
//!   expansion in `h`, with a ridge penalty on the r-th difference of the
//!   basis coefficients (`r = 2` shrinks toward a straight line). The
//!   stacked estimator has the closed form `theta_hat = (X'X + lambda
//!   P)^{-1} X'y`; `lambda` is fixed or chosen by leave-h-block-out
//!   cross-validation, and `lambda = 0` with the default interpolating
//!   basis reproduces the per-horizon HAC-path [`lp`] point estimates
//!   exactly.
//!
//! Module map: `level` ([`lp`]), `iv` ([`lp_iv`], [`lp_multiplier`]),
//! `state` ([`lp_state`]), `smooth` ([`smooth_lp`]), with the shared design
//! bookkeeping in `design`, the spec/result types in `spec`, and the typed
//! error in `error`.
//!
//! ## Cumulative responses vs multipliers
//!
//! [`Cumulation`] names the thing that is easy to get wrong.
//! `Cumulation::Outcome` (the historical `cumulative = true`) accumulates
//! only the left-hand side: it is a cumulative *impulse response*, cumulated
//! `y` per unit of contemporaneous impulse. Dividing it by nothing does not
//! make it a multiplier — with a denominator that never grows, it rises
//! roughly linearly in the horizon by construction. A multiplier needs
//! `Cumulation::Both`, and the identified version of it is
//! [`lp_multiplier`].
//!
//! ## Why lag-augmented HC1 is the default
//!
//! The textbook LP standard error is a Newey-West HAC: the horizon-`h`
//! projection residual is a moving average of the `h` innovations dated
//! `t+1, ..., t+h`, so the OLS score `s_t = shock_t \cdot u_{t,h}` is
//! serially correlated up to order `h`, and one "corrects" for it with a
//! kernel whose bandwidth must grow with `h`. That correction is exactly the
//! problem: HAC inference for LP is sensitive to the bandwidth choice and,
//! near a unit root, the confidence intervals systematically undercover
//! (the effective bandwidth explodes relative to the sample).
//!
//! Montiel Olea & Plagborg-Møller (2021, *Econometrica*, "Local Projection
//! Inference Is Simpler and More Robust Than You Think") show a cleaner
//! route: **augment** the horizon-`h` regression with the impulse's own lags
//! `shock_{t-1}, ..., shock_{t-h}`. Under the augmentation the part of the
//! residual that leaks past shocks into the score is projected out, the
//! score `s_t` becomes serially uncorrelated (a martingale difference
//! sequence), and the ordinary heteroskedasticity-robust (HC1)
//! variance is consistent — no kernel, no bandwidth. The resulting
//! `t`-statistic is asymptotically standard normal *uniformly* over the
//! persistence of the process, so coverage is reliable even in the
//! near-unit-root region where HAC-LP fails. This crate therefore makes
//! lag-augmented HC1 the default ([`SeSpec::LagAugmented`]) and keeps HAC
//! ([`SeSpec::Hac`]) as the statsmodels-compatibility option.
//!
//! ```
//! use tsecon_lp::{lp, LpSpec};
//!
//! // A simple persistent series driven by an observed (white-noise-like)
//! // shock plus a little measurement noise, generated deterministically so
//! // the doctest is reproducible.
//! let n = 200;
//! let mut state = 1u64;
//! let mut draw = || {
//!     state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
//!     (state >> 33) as f64 / (1u64 << 30) as f64 - 1.0 // in [-1, 1)
//! };
//! let shock: Vec<f64> = (0..n).map(|_| draw()).collect();
//! let noise: Vec<f64> = (0..n).map(|_| draw()).collect();
//! let mut y = vec![0.0; n];
//! for t in 1..n {
//!     y[t] = 0.7 * y[t - 1] + shock[t] + 0.3 * noise[t];
//! }
//!
//! // Default: lag-augmented HC1 inference, horizons 0..=6, 4 lag controls.
//! let res = lp(&y, &shock, LpSpec::new(6, 4)).unwrap();
//! assert_eq!(res.irf.len(), 7);
//! assert!(res.irf[0] > 0.0 && res.se[0] > 0.0);
//! ```
//!
//! ## Validation
//!
//! Two independent goldens pin this crate:
//!
//! - `fixtures/lp.json` (`fixtures/generate_phase2_fixtures.py`) — the
//!   per-horizon
//!   estimators: [`lp`]/HAC against statsmodels OLS `cov_type="HAC"` (beta
//!   `1e-10`, se `1e-8`), [`lp_iv`] against linearmodels IV2SLS
//!   `cov_type="kernel"` (beta `1e-8`, se `1e-6`).
//! - `fixtures/smoothlp.json` (`fixtures/generate_smoothlp_fixtures.py`) —
//!   smooth LP: the B-spline basis against
//!   `scipy.interpolate.BSpline.design_matrix` on the same uniform knots
//!   (`1e-10`), the stacked penalized `theta`/IRF/sandwich-SE paths against
//!   plain-NumPy normal equations transcribing the documented closed form
//!   at several `lambda` (`~1e-8` relative), the `lambda = 0` IRF against
//!   *statsmodels* per-horizon OLS (`1e-8`), the raw-LP comparison path
//!   against statsmodels HAC (beta `1e-10`, se `1e-8`), and the
//!   leave-h-block-out CV scores and chosen `lambda` against the same rule
//!   evaluated in NumPy (`~1e-8` relative).
//!
//! `tests/properties.rs` covers the statistical claims (coverage of the
//! lag-augmented default, the Ramey-Zubairy identities, `lambda -> 0` /
//! `lambda -> inf` limits of smooth LP and its MSE gain over raw LP under a
//! smooth true IRF); `tests/validation.rs` covers the errors-that-teach
//! guards.
//!
//! ## References
//!
//! - Jordà, Ò. (2005). "Estimation and Inference of Impulse Responses by
//!   Local Projections." *American Economic Review*.
//! - Montiel Olea, J. L., & Plagborg-Møller, M. (2021). "Local Projection
//!   Inference Is Simpler and More Robust Than You Think." *Econometrica*.
//! - Ramey, V. A., & Zubairy, S. (2018). "Government Spending Multipliers in
//!   Good Times and in Bad." *Journal of Political Economy*.
//! - Montiel Olea, J. L., & Pflueger, C. (2013). "A Robust Test for Weak
//!   Instruments." *Journal of Business & Economic Statistics*.
//! - Barnichon, R., & Brownlees, C. (2019). "Impulse Response Estimation by
//!   Smooth Local Projections." *Review of Economics and Statistics*.
//! - Eilers, P. H. C., & Marx, B. D. (1996). "Flexible Smoothing with
//!   B-splines and Penalties." *Statistical Science*.
//! - Burman, P., Chow, E., & Nolan, D. (1994). "A Cross-Validatory Method
//!   for Dependent Data." *Biometrika*.

#![warn(missing_docs)]
#![warn(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod design;
mod error;
mod iv;
mod level;
mod smooth;
mod spec;
mod state;

pub use error::LpError;
pub use iv::{lp_iv, lp_multiplier};
pub use level::lp;
pub use smooth::{smooth_lp, SmoothLambda, SmoothLpResult, SmoothLpSpec};
pub use spec::{
    Cumulation, LpIvResult, LpMultiplierResult, LpResult, LpSpec, LpStateResult, SeKind, SeSpec,
};
pub use state::lp_state;
