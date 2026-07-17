//! # tsecon-mgarch
//!
//! Multivariate (conditional-correlation) GARCH for the `tsecon`
//! time-series econometrics library (volatility track; ROADMAP 03
//! "multivariate" rows). Two models, both built on the decomposition
//!
//! ```text
//! H_t = D_t R_t D_t,   D_t = diag(sigma_{1,t}, ..., sigma_{k,t}),
//! ```
//!
//! where `H_t` is the conditional covariance, `D_t` holds the univariate
//! GARCH conditional standard deviations, and `R_t` is a conditional
//! correlation matrix:
//!
//! * [`CccGarch`] — **Constant** Conditional Correlation (Bollerslev 1990):
//!   `R_t = R`, the sample correlation matrix of the standardized residuals.
//! * [`DccGarch`] — **Dynamic** Conditional Correlation (Engle 2002): `R_t`
//!   evolves through a scalar GARCH-like recursion on an auxiliary matrix
//!   `Q_t`, correlation-targeted to `Qbar = (1/T) sum_t z_t z_t'`.
//!
//! Both use **two-step (Engle) estimation**: step 1 fits a univariate
//! GARCH to each series — delegated wholesale to [`tsecon_garch`] — and
//! step 2 estimates the correlation structure from the standardized
//! residuals with the univariate parameters held fixed. All covariance and
//! correlation factorizations route through [`tsecon_linalg`]'s shared
//! positive-definiteness-hygiene path (symmetrize, then `L L'` Cholesky with
//! a bounded jitter ladder), so every emitted `H_t` and `R_t` is symmetric
//! and positive-definite.
//!
//! # A note on validation
//!
//! The univariate stage is pinned to Kevin Sheppard's `arch` through the
//! `tsecon-garch` golden fixtures. The **DCC correlation dynamics are not** —
//! there is no third-party DCC reference available in this project. The
//! fixture `fixtures/mgarch.json` therefore ships *simulated* data from a
//! known DCC(a, b) + CCC base together with the true parameters, and the
//! crate validates the DCC path by internal properties instead of a golden
//! comparison:
//!
//! 1. the **CCC special case** — `DccGarch` at `a = b = 0` reproduces the
//!    `CccGarch` log-likelihood to `1e-10`;
//! 2. **positive-definiteness** of every `R_t` on the fixture data;
//! 3. **correlation targeting** — the sample mean of `z_t z_t'` equals
//!    `Qbar`, so `E[Q_t] = Qbar`;
//! 4. **simulation recovery** — on the fixture's single simulated
//!    realization (truth `a = 0.03`, `b = 0.95`), the estimated persistence
//!    `a + b` lands within a loose Monte-Carlo tolerance (`0.05`) of `0.98`.
//!
//! These bounds are honest: property 4 is a single-realization sanity check,
//! not a precision guarantee, and the docs on [`DccGarch`] restate this.
//!
//! ```
//! use tsecon_garch::{DistSpec, GarchSpec, MeanSpec, VolSpec};
//! use tsecon_mgarch::{CccGarch, DccGarch};
//!
//! // Two mildly heteroskedastic, correlated series (deterministic here so
//! // the doctest is reproducible; in practice these are asset returns).
//! let mut seed = 0x2545_F491_4F6C_DD1D_u64;
//! let mut normal = || {
//!     seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
//!     let mut z = seed;
//!     z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
//!     z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
//!     let u1 = (((z ^ (z >> 31)) >> 11) as f64 + 0.5) / (1u64 << 53) as f64;
//!     seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
//!     let mut w = seed;
//!     w = (w ^ (w >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
//!     w = (w ^ (w >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
//!     let u2 = (((w ^ (w >> 31)) >> 11) as f64 + 0.5) / (1u64 << 53) as f64;
//!     (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
//! };
//! let (mut s0, mut s1) = (Vec::new(), Vec::new());
//! let (mut v0, mut v1) = (1.0_f64, 1.0_f64);
//! for _ in 0..400 {
//!     let (e0, common) = (normal(), normal());
//!     let e1 = 0.6 * e0 + 0.8 * normal();
//!     let (x0, x1) = (v0.sqrt() * e0 + 0.2 * common, v1.sqrt() * e1 + 0.2 * common);
//!     v0 = 0.05 + 0.1 * x0 * x0 + 0.85 * v0;
//!     v1 = 0.04 + 0.08 * x1 * x1 + 0.88 * v1;
//!     s0.push(x0);
//!     s1.push(x1);
//! }
//! let series = vec![s0, s1];
//! let spec = GarchSpec {
//!     mean: MeanSpec::Zero,
//!     vol: VolSpec::Garch { p: 1, q: 1 },
//!     dist: DistSpec::Normal,
//! };
//!
//! let ccc = CccGarch::new(spec).fit(&series).unwrap();
//! assert_eq!(ccc.k(), 2);
//! assert!(ccc.loglik.is_finite());
//!
//! let dcc = DccGarch::new(spec).fit(&series).unwrap();
//! assert!(dcc.persistence() >= 0.0 && dcc.persistence() < 1.0);
//! ```

#![warn(missing_docs)]
#![warn(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod ccc;
mod dcc;
mod error;
mod stage;
mod util;

pub use ccc::{CccFit, CccGarch};
pub use dcc::{DccFit, DccGarch};
pub use error::MgarchError;
pub use stage::UnivariateStage;

// Re-export the shared dense backend so callers manipulate the returned
// `Mat<f64>` covariances without pinning their own `faer` version.
pub use tsecon_linalg::faer;
