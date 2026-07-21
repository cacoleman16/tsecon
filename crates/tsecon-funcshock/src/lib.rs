//! # tsecon-funcshock — functional shocks (Inoue-Rossi 2021)
//!
//! The response of an outcome to a shock that is a whole CURVE — the entire
//! yield curve shifting on an announcement day — not a scalar. This is the
//! functional-VAR / functional-local-projection idea of Inoue & Rossi
//! (2021, Quantitative Economics): summarize curve shocks by functional
//! principal components, trace the outcome's response to the component
//! scores, and reconstruct the response to ANY whole-curve scenario from
//! its projection onto the eigenfunctions.
//!
//! ## The method
//!
//! **1. Functional PCA** ([`functional_pca`] / [`Fpca`]). Given a `T x M`
//! panel of curves on a shared grid, demean and eigendecompose the `M x M`
//! covariance `Xc' Xc / T` (population divisor; `faer` self-adjoint
//! eigensolver through [`tsecon_linalg`]). Keep the leading `K`
//! eigenfunctions `phi_k`, scores `s_{t,k} = <X_t - mean, phi_k>`,
//! eigenvalues, and explained-variance shares. The inner product is the
//! discrete (Euclidean) one on the grid — no quadrature weights — matching
//! the discretized Inoue-Rossi implementation. Sign convention: each
//! eigenfunction's largest-|.| entry is positive (first index on ties),
//! identical to the fixture generator, so the golden pin is well-defined.
//!
//! **2. Functional local projection** ([`flp`] / [`FlpFit`]). At each
//! horizon `h`, regress `y_{t+h}` jointly on a constant, ALL `K` scores at
//! `t`, and `p` lags of `y`, via the library's single OLS owner
//! [`tsecon_hac::ols`] with Newey-West Bartlett kernel-HAC covariance
//! (truncation `h + p` by default, `use_correction=True` scaling — the
//! `tsecon-lp` conventions). The JOINT `K x K` coefficient covariance is
//! kept per horizon; the scenario variance needs its off-diagonals.
//!
//! **3. Functional scenario response** ([`scenario_weights`] /
//! [`scenario_response`] / [`flp_scenario`] / [`ScenarioResponse`]). A user
//! scenario curve `delta(M)` ("the whole curve flattens by this shape")
//! projects onto the eigenfunctions, `w_k = <phi_k, delta>`; the IRF of `y`
//! to that whole-curve scenario is `response_h = w' beta_h` with variance
//! `w' Cov_h w`. This is the deliverable that makes the method functional.
//!
//! **4. FVAR scenario** ([`fvar_scenario`] / [`FvarScenario`]). The same
//! reconstruction through a VAR in `[scores, outcome]` (scores first,
//! outcome last) using [`tsecon_var`]'s IRF machinery: set the reduced-form
//! score innovation to `w`, the outcome's own structural shock to zero, and
//! read the response off the Cholesky-orthogonalized MA coefficients.
//! **Identification caveat:** recursive/Cholesky with scores ordered first —
//! the impact response of the outcome is the in-sample regression of its
//! innovation on the score innovations, an assumption that is credible in
//! announcement-day timing and not otherwise; see [`fvar_scenario`].
//!
//! ## Module map
//!
//! - [`functional_pca`] / [`Fpca`] — functional principal components.
//! - [`flp`] / [`FlpFit`] — per-horizon joint score regressions (HAC).
//! - [`scenario_weights`] / [`scenario_response`] / [`flp_scenario`] /
//!   [`ScenarioResponse`] — whole-curve scenario IRFs with SEs.
//! - [`fvar_scenario`] / [`FvarScenario`] — the FVAR route.
//! - [`FuncShockError`] — the crate's error type ("errors that teach").
//!
//! ## Validation
//!
//! The golden fixture `fixtures/tsecon-funcshock.json`
//! (`fixtures/generate_tsecon-funcshock_fixtures.py`) is an INDEPENDENT
//! reference; nothing in it touches this crate. `tests/golden.rs` pins:
//!
//! * [`functional_pca`] to `numpy.linalg.eigh` of the same covariance
//!   (mean curve, all-`M` descending eigenvalues, sign-fixed
//!   eigenfunctions, scores, explained shares) at `1e-10`;
//! * [`flp`] to statsmodels `OLS(...).fit(cov_type="HAC",
//!   maxlags=h+p, use_correction=True)` per horizon — betas, the JOINT
//!   `K x K` covariance, and SEs at `1e-8`;
//! * [`scenario_response`] to the numpy closed form `w' beta_h`,
//!   `sqrt(w' Cov_h w)` at `1e-8`;
//! * [`fvar_scenario`] to statsmodels `VAR([scores, y]).fit(lags,
//!   trend="c")` + `orth_ma_rep` + `scipy` triangular solve at `1e-8`.
//!
//! `tests/properties.rs` checks the invariants (eigenfunction
//! orthonormality, score/eigenvalue identities, the EXACT reconstruction
//! identity — a scenario equal to the `j`-th eigenfunction reproduces the
//! `beta_j` path bit-near-exactly — the FVAR impact identity
//! `responses[0][..K] == w`, and a seeded Monte Carlo recovering
//! `integral B(m) delta(m) dm` for a known functional `B`);
//! `tests/validation.rs` covers the errors that teach.
//!
//! ```
//! use tsecon_funcshock::{flp, flp_scenario, functional_pca};
//!
//! // Toy panel: 40 days of 5-point curves moved by one level factor.
//! let curves: Vec<Vec<f64>> = (0..40)
//!     .map(|t| {
//!         let f = (0.3 * t as f64).sin();
//!         (0..5).map(|m| f + 0.01 * ((t * 5 + m) % 7) as f64).collect()
//!     })
//!     .collect();
//! let y: Vec<f64> = (0..40).map(|t| (0.3 * t as f64).sin() * 0.5).collect();
//!
//! let fpca = functional_pca(&curves, 2).unwrap();
//! let fit = flp(&y, &fpca.scores, 4, 1, None).unwrap();
//! let delta = vec![1.0; 5]; // the whole curve shifts up in parallel
//! let irf = flp_scenario(&fpca, &fit, &delta).unwrap();
//! assert_eq!(irf.response.len(), 5);
//! assert!(irf.se.iter().all(|s| s.is_finite() && *s >= 0.0));
//! ```

#![warn(missing_docs)]
#![warn(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod error;
mod flp;
mod fpca;
mod fvar;
mod scenario;

pub use error::FuncShockError;
pub use flp::{flp, FlpFit};
pub use fpca::{functional_pca, Fpca};
pub use fvar::{fvar_scenario, FvarScenario};
pub use scenario::{flp_scenario, scenario_response, scenario_weights, ScenarioResponse};
