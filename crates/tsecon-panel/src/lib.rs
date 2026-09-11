//! # tsecon-panel — panel time series econometrics
//!
//! The first panel slice of the `tsecon` library (roadmap Modules 04/07
//! panel rows and extension E1): fixed-effects estimation with
//! panel-robust covariances, panel local projections, and a mean-group
//! panel VAR. Every fixed-effects numeric convention matches
//! `linearmodels.panel.PanelOLS` (version 7.0; the golden fixture
//! `fixtures/panel.json` arbitrates):
//!
//! * [`PanelData`] — the panel container (`N x T` outcome and regressor
//!   matrices) with an optional observation mask for unbalanced panels
//!   ([`PanelData::unbalanced`]: entities entering late, leaving early,
//!   internal gaps — the within projections, lag designs and covariances
//!   skip the unobserved cells; validated against `PanelOLS` on the
//!   Arellano-Bond `EmplUK` panel in `fixtures/panel_unbalanced.json`);
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
//!   estimates), and two half-panel Nickell-bias corrections: the
//!   Dhaene-Jochmans (2015) jackknife ([`PanelLpConfig::jackknife`]) and
//!   the Mei-Sheng-Shi (2026) split-panel jackknife with its
//!   adjusted-score standard errors
//!   ([`PanelLpConfig::bias_correction`]);
//! * [`lp_did`] — LP-DiD, the local-projections difference-in-differences
//!   of Dube, Girardi, Jordà & Taylor (2025, J. Applied Econometrics):
//!   per-horizon long-difference regressions on the treatment switch with
//!   period effects, restricted to clean controls (not-yet-treated /
//!   never-treated / stabilized — avoiding TWFE's negative-weight
//!   comparisons), with pre-trend horizons, the equally-weighted-ATT
//!   reweighting, pooled ATTs, absorbing and non-absorbing treatments,
//!   and entity-clustered standard errors in the authors'
//!   fixest/reghdfe convention (see `lpdid.rs`);
//! * [`mean_group_var`] — the Pesaran-Smith (1995) mean-group panel VAR:
//!   per-entity VARs via `tsecon-var`, cross-entity averages of
//!   coefficients and Cholesky-orthogonalized IRFs, with dispersion-based
//!   standard errors `sd / sqrt(N)`.
//! * [`panel_distributed_lag`] — distributed-lag panel regressions, the
//!   climate-impact specification of Dell-Jones-Olken (2012) and
//!   Burke-Hsiang-Miguel (2015): `L` lags of each regressor (and of its
//!   square for a nonlinear response) with entity effects, time effects
//!   and optional entity-specific trends via [`panel_ols_fe_with`]'s
//!   [`FixedEffects`] menu, returning the cumulative (long-run) effect
//!   with its delta-method standard error, and for the quadratic
//!   response the marginal effect at chosen points and the turning point
//!   of the cumulative response (see `distributed_lag.rs`).
//!
//! ## Nickell bias (read before running dynamic panels with short T)
//!
//! Fixed effects + lagged outcomes + short T biases dynamic coefficients:
//! the within transformation makes the demeaned lagged outcome mechanically
//! correlated with the demeaned error, giving an incidental-parameter bias
//! of roughly `-(1 + rho)/(T - 1)` for an AR(1) panel (Nickell 1981) that
//! does **not** shrink with the number of entities and is horizon-amplified
//! in local projections. See `lp.rs` for the full discussion and the two
//! half-panel corrections ([`PanelLpConfig::jackknife`] and
//! [`PanelLpConfig::bias_correction`]).
//!
//! All fallible routines return [`PanelError`]; nothing in this crate
//! panics on user input.

#![warn(missing_docs)]
#![warn(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

pub mod data;
pub mod distributed_lag;
pub mod error;
pub mod fe;
pub mod lp;
pub mod lpdid;
pub mod mean_group;

pub use data::PanelData;
pub use distributed_lag::{panel_distributed_lag, DistributedLagConfig, DistributedLagResult};
pub use error::PanelError;
pub use fe::{
    panel_ols_fe, panel_ols_fe_with, FePanelOls, FixedEffects, PanelInference, PanelSeType,
};
pub use lp::{panel_lp, LpBiasCorrection, PanelLpConfig, PanelLpResult};
pub use lpdid::{lp_did, LpDidConfig, LpDidPooled, LpDidResult};
pub use mean_group::{mean_group_var, mg_irf_path, MeanGroupVar};

// Re-export the shared linear-algebra layer (and, through it, the dense
// backend) plus the VAR layer whose `Trend` the mean-group API takes, so
// downstream crates see one faer/tsecon-var version.
pub use tsecon_linalg;
pub use tsecon_var;
