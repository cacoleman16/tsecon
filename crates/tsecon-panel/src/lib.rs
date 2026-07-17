//! # tsecon-panel — panel time series econometrics
//!
//! The first panel slice of the `tsecon` library (roadmap Modules 04/07
//! panel rows and extension E1): fixed-effects estimation with
//! panel-robust covariances, panel local projections, and a mean-group
//! panel VAR. Every fixed-effects numeric convention matches
//! `linearmodels.panel.PanelOLS` (version 7.0; the golden fixture
//! `fixtures/panel.json` arbitrates):
//!
//! * [`PanelData`] — balanced-panel container (`N x T` outcome and
//!   regressor matrices; unbalanced panels are `// TODO(phase0)`, see
//!   `data.rs` for the planned mask design);
//! * [`panel_ols_fe`] / [`FePanelOls::inference`] — the within (entity-
//!   demeaned) estimator with the correct `nobs - k - N` degrees of
//!   freedom and a [`PanelSeType`] menu: nonrobust, clustered by entity
//!   (Arellano 1987), and Driscoll-Kraay (1998) Bartlett-kernel HAC on
//!   per-period score sums (weights from `tsecon-hac`, the library's
//!   single kernel owner);
//! * [`panel_lp`] — panel local projections (Jordà 2005) with fixed
//!   effects, lagged-shock/lagged-outcome controls, Ramey-Zubairy (2018)
//!   cumulative multipliers estimated on the cumulated regressand (so the
//!   standard errors are the cumulative ones — never a cumsum of level
//!   estimates), and the Dhaene-Jochmans (2015) half-panel jackknife as a
//!   Nickell-bias correction option;
//! * [`mean_group_var`] — the Pesaran-Smith (1995) mean-group panel VAR:
//!   per-entity VARs via `tsecon-var`, cross-entity averages of
//!   coefficients and Cholesky-orthogonalized IRFs, with dispersion-based
//!   standard errors `sd / sqrt(N)`.
//!
//! ## Nickell bias (read before running dynamic panels with short T)
//!
//! Fixed effects + lagged outcomes + short T biases dynamic coefficients:
//! the within transformation makes the demeaned lagged outcome mechanically
//! correlated with the demeaned error, giving an incidental-parameter bias
//! of roughly `-(1 + rho)/(T - 1)` for an AR(1) panel (Nickell 1981) that
//! does **not** shrink with the number of entities and is horizon-amplified
//! in local projections. See `lp.rs` for the full discussion and the
//! half-panel jackknife correction ([`PanelLpConfig::jackknife`]).
//!
//! All fallible routines return [`PanelError`]; nothing in this crate
//! panics on user input.

#![warn(missing_docs)]
#![warn(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

pub mod data;
pub mod error;
pub mod fe;
pub mod lp;
pub mod mean_group;

pub use data::PanelData;
pub use error::PanelError;
pub use fe::{panel_ols_fe, FePanelOls, PanelInference, PanelSeType};
pub use lp::{panel_lp, PanelLpConfig, PanelLpResult};
pub use mean_group::{mean_group_var, mg_irf_path, MeanGroupVar};

// Re-export the shared linear-algebra layer (and, through it, the dense
// backend) plus the VAR layer whose `Trend` the mean-group API takes, so
// downstream crates see one faer/tsecon-var version.
pub use tsecon_linalg;
pub use tsecon_var;
