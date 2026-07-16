//! # tsecon-optim
//!
//! The numerical optimization suite of the `tsecon` time-series
//! econometrics library (foundations layer; ROADMAP 00 "Optimization").
//! ARIMA, GARCH, ETS, and state-space MLE all run on this crate.
//!
//! Contents:
//!
//! * [`nelder_mead`] — derivative-free simplex search with the Gao-Han
//!   (2012) dimension-adaptive coefficients, scipy-compatible termination
//!   (simplex size + f-spread), budgets, and restart support;
//! * [`bfgs`] / [`lbfgs`] — quasi-Newton minimizers (Nocedal-Wright 2006,
//!   chapters 6-7) sharing the strong-Wolfe line search [`strong_wolfe`]
//!   (More-Thuente-style bracketing/zoom, `c1 = 1e-4`, `c2 = 0.9`); the
//!   gradient is the caller's analytic one or the documented
//!   [`central_difference_gradient`];
//! * [`ObjectiveFn`] / [`OptimizeResult`] / [`minimize`] — the one API
//!   every model crate consumes: trait in, typed result (best point,
//!   [`Termination`] reason, budgets) out;
//! * the reparameterization toolkit ([`Transform`],
//!   [`TransformedObjective`]): [`Positive`], [`Bounded`],
//!   [`UnitInterval`], [`Ordered`], and the Monahan (1984) PACF
//!   stationarity transform [`StationaryAr`] (MA invertibility by
//!   duality) — each with exact forward/inverse and the log-Jacobian
//!   needed for densities in working space;
//! * [`multistart`] — best-of-k perturbed starts, the recommended default
//!   for multimodal likelihoods (threshold, Markov-switching, GARCH-M).
//!
//! Failure honesty: optimizers never panic on user input and always return
//! the best point found; non-convergence is reported through
//! [`OptimizeResult::converged`] and [`Termination`], and boundary optima
//! of reparameterized domains surface as line-search failure/divergence in
//! working space rather than a fake interior optimum.
//!
//! ```
//! use tsecon_optim::{minimize, FnObjectiveGrad, Method};
//!
//! // Rosenbrock: f(x, y) = 100 (y - x^2)^2 + (1 - x)^2.
//! let mut rosen = FnObjectiveGrad::new(
//!     |x: &[f64]| 100.0 * (x[1] - x[0] * x[0]).powi(2) + (1.0 - x[0]).powi(2),
//!     |x: &[f64]| {
//!         vec![
//!             -400.0 * x[0] * (x[1] - x[0] * x[0]) - 2.0 * (1.0 - x[0]),
//!             200.0 * (x[1] - x[0] * x[0]),
//!         ]
//!     },
//! );
//! let res = minimize(&mut rosen, &[-1.2, 1.0], &Method::bfgs()).unwrap();
//! assert!(res.converged && (res.x[0] - 1.0).abs() < 1e-6);
//! ```

#![warn(missing_docs)]
#![warn(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod error;
mod linesearch;
mod minimize;
mod neldermead;
mod objective;
mod quasinewton;
mod result;
mod transform;

pub use error::OptimError;
pub use linesearch::{strong_wolfe, WolfeOptions, WolfeResult};
pub use minimize::{minimize, multistart, Method, MultistartResult};
pub use neldermead::{nelder_mead, NelderMeadOptions};
pub use objective::{
    central_difference_gradient, eval_gradient, FnObjective, FnObjectiveGrad, ObjectiveFn,
};
pub use quasinewton::{bfgs, lbfgs, BfgsOptions, LbfgsOptions};
pub use result::{OptimizeResult, Termination};
pub use transform::{
    Bounded, Ordered, Positive, StationaryAr, Transform, TransformedObjective, UnitInterval,
};
