//! # tsecon-garch
//!
//! Univariate conditional-variance models for the `tsecon` time-series
//! econometrics library (volatility track; ROADMAP 03 "Univariate GARCH
//! core").
//!
//! Models: [`VolSpec::Garch`] (Bollerslev 1986; `q = 0` gives Engle 1982
//! ARCH), [`VolSpec::Gjr`] (Glosten-Jagannathan-Runkle 1993), and
//! [`VolSpec::Egarch`] (Nelson 1991), each under a zero or constant mean
//! ([`MeanSpec`]) with normal or standardized Student-t innovations
//! ([`DistSpec`], the unit-variance t of
//! [`tsecon_stats::Standardized::student_t`], Bollerslev 1987).
//!
//! Estimation is quasi-maximum likelihood (Bollerslev-Wooldridge 1992):
//! an `arch`-style grid start, L-BFGS with central-difference gradients in
//! a reparameterized working space, and a Nelder-Mead polish, all from
//! `tsecon-optim`. Inference offers both classical MLE (inverse numerical
//! Hessian) and Bollerslev-Wooldridge robust sandwich standard errors
//! ([`StdErrors`]). [`GarchResults`] carries the conditional volatility,
//! standardized residuals, information criteria, and analytic multi-step
//! variance forecasts for GARCH/GJR (EGARCH one-step; multi-step is
//! `// TODO(phase0)` pending the simulation engine).
//!
//! **Cross-package parity**: every convention — the RiskMetrics-decay
//! backcast initialization, presample terms, likelihood constants,
//! parameter ordering and names, and the numerical-derivative steps behind
//! the standard errors — matches Kevin Sheppard's `arch` package, pinned
//! by the golden fixture `fixtures/garch.json` (fixed-parameter
//! log-likelihoods to 1e-8 relative, conditional volatilities to 1e-6,
//! robust standard errors to 5e-3).
//!
//! ```
//! use tsecon_garch::{DistSpec, GarchModel, GarchSpec, MeanSpec, VolSpec};
//!
//! // A tiny GARCH(1,1) log-likelihood evaluation at fixed parameters.
//! let y = [0.4, -1.2, 0.3, 0.8, -0.5, 1.4, -0.9, 0.2, -0.1, 0.6,
//!          -0.7, 1.1, 0.05, -0.3, 0.9, -1.5, 0.45, -0.2, 0.75, -0.6];
//! let spec = GarchSpec {
//!     mean: MeanSpec::Zero,
//!     vol: VolSpec::Garch { p: 1, q: 1 },
//!     dist: DistSpec::Normal,
//! };
//! let model = GarchModel::new(&y, spec).unwrap();
//! // params: omega, alpha[1], beta[1] — the arch ordering.
//! let ll = model.loglike(&[0.05, 0.1, 0.8]).unwrap();
//! assert!(ll.is_finite());
//! ```

#![warn(missing_docs)]
#![warn(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod error;
mod inference;
mod model;
mod recursion;
mod results;
mod spec;

pub use error::GarchError;
pub use inference::StdErrors;
pub use model::GarchModel;
pub use results::GarchResults;
pub use spec::{DistSpec, GarchSpec, MeanSpec, VolSpec};
