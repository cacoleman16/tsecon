//! # tsecon-regime
//!
//! Markov-switching time-series models for the `tsecon` econometrics
//! library (nonlinear-univariate track; ROADMAP 02 "Markov-switching"
//! rows).
//!
//! The crate fits and scores the `k`-regime Markov-switching autoregression
//! of Hamilton (1989): conditional on a latent first-order Markov chain
//! `S_t`,
//!
//! ```text
//! y_t - mu_{S_t} = sum_{l=1}^{p} phi_l (y_{t-l} - mu_{S_{t-l}}) + e_t,
//! e_t ~ N(0, sigma^2_{S_t}),
//! ```
//!
//! with per-regime means `mu`, common or switching AR coefficients `phi`,
//! and common or switching innovation variances ([`MsarSpec`]). The regime
//! chain has column-stochastic transition matrix `P[i][j] = P(S_t = i |
//! S_{t-1} = j)`.
//!
//! * [`MarkovSwitchingAr::filter`] runs the **Hamilton (1989) filter**,
//!   tracking the joint distribution of the last `p + 1` regimes and
//!   returning the prediction-error-decomposition log-likelihood and
//!   filtered regime probabilities.
//! * [`MarkovSwitchingAr::smooth`] adds the **Kim (1994) fixed-interval
//!   smoother** for the smoothed regime probabilities `P(S_t | Y_T)`.
//! * [`MarkovSwitchingAr::fit`] estimates the parameters by **EM
//!   (Baum-Welch)**: the Hamilton filter and Kim smoother in the E-step,
//!   and closed-form transition counts with expectation-conditional-
//!   maximization Gaussian updates in the M-step, monotonically increasing
//!   the log-likelihood (Dempster, Laird & Rubin 1977; Hamilton 1990).
//! * [`classify`] and [`MsarParams::expected_durations`] summarize a fit:
//!   the most-probable regime per period and the expected sojourn `1 / (1 -
//!   p_ii)`.
//!
//! **Cross-package parity**: the filter/smoother conventions — the
//! steady-state (stationary-distribution) initialization, the switching-
//! mean AR recursion, and the switching-variance likelihood — match
//! `statsmodels` `MarkovAutoregression`, pinned by the golden fixture
//! `fixtures/regime.json` (`k = 2`, `order = 1`, `switching_ar = false`,
//! `switching_variance = true`): the fixed-parameter log-likelihood and the
//! filtered and smoothed regime-1 probabilities all to `1e-6`.
//!
//! Because the likelihood is multimodal, EM converges to a *local* optimum;
//! fits are assessed by log-likelihood improvement and approximate
//! parameter recovery, not exact agreement with a single optimum.
//!
//! ```
//! use tsecon_regime::{MarkovSwitchingAr, MsarParams, MsarSpec};
//!
//! let spec = MsarSpec { k_regimes: 2, order: 1,
//!                       switching_ar: false, switching_variance: true };
//! let params = MsarParams::new(
//!     vec![vec![0.95, 0.10], vec![0.05, 0.90]], // P[i][j] = P(S_t=i|S_{t-1}=j)
//!     vec![-1.0, 1.5],
//!     vec![vec![0.5]],
//!     vec![0.64, 1.44],
//! ).unwrap();
//! // Expected sojourn 1 / (1 - p_ii): 1/0.05 = 20 and 1/0.10 = 10.
//! let dur = params.expected_durations();
//! assert!((dur[0] - 20.0).abs() < 1e-9 && (dur[1] - 10.0).abs() < 1e-9);
//! ```

#![warn(missing_docs)]
#![warn(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod error;
mod linsolve;
mod model;
mod params;
mod results;
mod spec;

pub use error::RegimeError;
pub use model::MarkovSwitchingAr;
pub use params::MsarParams;
pub use results::{classify, FilterResult, FitResult, SmoothResult};
pub use spec::MsarSpec;
