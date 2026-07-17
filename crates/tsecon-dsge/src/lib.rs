//! # tsecon-dsge — a minimal linear rational-expectations (DSGE-lite) solver
//!
//! The Blanchard-Kahn (1980) solution of a linearized rational-expectations
//! model to its state-space policy function. This is the roadmap E5 crate: a
//! deliberately minimal layer — the linear RE *solver*, not a full DSGE
//! estimation suite.
//!
//! ## The model
//!
//! A linearized model is supplied in first-order expectational form
//!
//! ```text
//! A . E_t[y_{t+1}] = B . y_t + C . z_{t+1}
//! ```
//!
//! where `y_t = [k_t ; x_t]` stacks the `n_predetermined` PREDETERMINED
//! (backward-looking) variables `k_t` on top of the NON-PREDETERMINED (jump /
//! forward-looking) variables `x_t`, and `z_{t+1}` is a mean-zero exogenous
//! innovation with `E_t[z_{t+1}] = 0`. Writing `M = A^{-1} B` and
//! `N = A^{-1} C`, the reduced form is `E_t[y_{t+1}] = M y_t + N z`.
//!
//! ## The method
//!
//! Eigen-decompose `M = V L V^{-1}` and order the eigenvalues by modulus. The
//! **Blanchard-Kahn condition** compares the number of eigenvalues *outside*
//! the unit circle, `n_unstable`, to the number of jump variables `n_jump`:
//!
//! | relation | verdict |
//! |---|---|
//! | `n_unstable == n_jump` | unique non-explosive solution |
//! | `n_unstable <  n_jump` | indeterminate (a continuum of stable solutions) |
//! | `n_unstable >  n_jump` | no stable solution (everything explodes) |
//!
//! When unique, the stable-eigenvector columns `V_s = [V_ks ; V_xs]` yield the
//! **policy rule** and **law of motion**
//!
//! ```text
//! jump_t              = G . predetermined_t,      G = V_xs V_ks^{-1}
//! predetermined_{t+1} = P . predetermined_t + Q . z_{t+1},
//!                                                 P = M_kk + M_kx G,   Q = N_k.
//! ```
//!
//! The stable eigenvalues of `M` are exactly the eigenvalues of `P`, so `P` is
//! always stable and the solved system reverts. See [`solve`], [`DsgeSolution`],
//! [`BlanchardKahnVerdict`], and [`verdict`].
//!
//! Because `E_t[z_{t+1}] = 0`, the innovation drops out of the forward-looking
//! solve and re-enters only the predetermined law of motion. This crate's
//! convention therefore requires shocks to load on predetermined (exogenous-
//! state) equations, not on jump equations; a shock on a jump row is rejected
//! with [`DsgeError::ShockOnJump`]. Route shocks through an exogenous AR state,
//! exactly as the Cagan example below does.
//!
//! ## Impulse responses and simulation
//!
//! [`DsgeSolution::impulse_response`] traces the response to a one-time unit
//! innovation; [`DsgeSolution::simulate`] runs the system under an explicit
//! shock sequence. Both return a [`Trajectory`].
//!
//! ## Scope and limitations
//!
//! * **Invertible `A` only.** The solver forms `M = A^{-1} B` explicitly, so
//!   `A` must be non-singular (the "regular" case). A singular pencil needs the
//!   QZ / generalized-Schur generalization, which is out of scope; such a model
//!   returns [`DsgeError::SingularA`]. (`faer` does expose a generalized
//!   eigendecomposition, so this is a deliberate scoping choice, not a backend
//!   limitation.)
//! * **Diagonalizable `M`.** `G` is built from the eigenvectors, so `M` must be
//!   diagonalizable on its stable subspace; a defective repeated root returns
//!   [`DsgeError::SingularStableBlock`].
//! * **Full complex spectrum supported.** `faer` 0.24 exposes the general
//!   (possibly complex) eigendecomposition, so complex eigenvalues — the
//!   oscillatory dynamics of many New-Keynesian blocks — are handled directly;
//!   the imaginary parts cancel in the real policy matrices, which is verified.
//!
//! ## Validation
//!
//! The solver is pinned to a DOCUMENTED CLOSED-FORM golden
//! (`fixtures/tsecon-dsge.json`, `fixtures/generate_tsecon-dsge_fixtures.py`):
//! the Cagan / asset-price model `p_t = a E_t[p_{t+1}] + u_t` with an AR(1)
//! fundamental has the textbook fundamental solution `p_t = u_t / (1 - a rho)`,
//! i.e. `G = 1/(1 - a rho)`, `P = rho`, `Q = sigma`; a two-shock variant gives a
//! diagonal `P` and a multi-column `G`. The Rust reproduces these analytic
//! matrices and the eigenvalues to ~1e-8 (`tests/golden.rs`). The generator
//! types the closed forms straight from the derivation and never calls the Rust
//! solver, and it independently re-derives the eigenvalues via `numpy.linalg`,
//! so the match is non-circular.
//!
//! Property tests (`tests/properties.rs`) additionally check that the solved
//! `P` is stable, that Blanchard-Kahn correctly flags too-few-jump
//! (no-stable-solution) and too-many-jump (indeterminate) mis-specifications,
//! and that impulse responses revert to zero.
//!
//! ```
//! use tsecon_dsge::{solve, LinearReModel};
//!
//! // Cagan p_t = a E_t p_{t+1} + u_t with a = 0.5, u AR(1) rho = 0.6.
//! // y = [u (predetermined); p (jump)]. M = [[rho, 0], [-1/a, 1/a]].
//! let a_mat = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
//! let b_mat = vec![vec![0.6, 0.0], vec![-2.0, 2.0]];
//! let c_mat = vec![vec![1.0], vec![0.0]];
//! let model = LinearReModel::new(&a_mat, &b_mat, &c_mat, 1).unwrap();
//! let sol = solve(&model).unwrap();
//! assert!(sol.verdict().is_unique());
//! // G = 1 / (1 - a rho) = 1 / 0.7.
//! assert!((sol.g()[(0, 0)] - 1.0 / 0.7).abs() < 1e-10);
//! ```
//!
//! ## Module map
//!
//! - [`LinearReModel`] — the model matrices, validation, and reduced form.
//! - [`solve`] / [`DsgeSolution`] — the Blanchard-Kahn decision rule and law of
//!   motion.
//! - [`verdict`] / [`BlanchardKahnVerdict`] — the existence/uniqueness
//!   classification, exposed on its own for determinacy analysis.
//! - [`DsgeSolution::impulse_response`] / [`DsgeSolution::simulate`] /
//!   [`Trajectory`] — responses and simulation of the solved system.
//! - [`DsgeError`] — the crate's error type ("errors that teach").

#![warn(missing_docs)]
#![warn(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod error;
mod model;
mod simulate;
mod solve;

pub use error::DsgeError;
pub use model::LinearReModel;
pub use simulate::Trajectory;
pub use solve::{solve, verdict, BlanchardKahnVerdict, DsgeSolution};
