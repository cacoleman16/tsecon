//! # tsecon-favar
//!
//! Approximate factor models and the factor-augmented VAR (FAVAR) for the
//! `tsecon` time-series econometrics library — the Module 04 factor layer
//! (see ROADMAP). Three public surfaces:
//!
//! * [`FactorModel`] — the static (approximate) factor model of
//!   Stock & Watson (2002): standardize an `n x N` panel (population /
//!   `ddof = 0` scaling), take a thin SVD `Z = U S V'`, and read off the
//!   principal-component factors `F = U S`, the loadings `L = V`, and the
//!   eigenvalues `lambda_k = s_k^2 / n`. This is the only step validated
//!   against an external golden fixture (`fixtures/favar.json`,
//!   eigenvalues / absolute PCs / absolute loadings to `1e-6`).
//! * [`bai_ng`] / [`eigenvalue_ratio`] — factor-number selection: the
//!   Bai & Ng (2002) `IC_p1`/`IC_p2`/`PC_p1`/`PC_p2` criteria and the
//!   Ahn & Horenstein (2013) eigenvalue-ratio estimator. The ER estimator
//!   is the robust choice in small cross-sections; the Bai-Ng criteria are
//!   consistent asymptotically but can over-select when `N` is small and
//!   the idiosyncratic eigenvalues decay slowly (see [`criteria`]).
//! * [`Favar`] — the two-step FAVAR of Bernanke, Boivin & Eliasz (2005):
//!   extract `r` factors, fit a VAR (via `tsecon-var`) on
//!   `[factors, policy]` with the policy variable ordered last, offer the
//!   slow/fast factor rotation, and map the factor VAR's impulse responses
//!   back onto every observed series through the loadings. The two-step
//!   assembly and IRF mapping have no external golden; they are validated
//!   structurally and by simulation.
//!
//! All fallible routines return [`FavarError`]; nothing in this crate
//! panics on user input.

#![warn(missing_docs)]
#![warn(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

pub mod criteria;
pub mod error;
pub mod favar;
pub mod pca;

pub use criteria::{bai_ng, eigenvalue_ratio, BaiNg};
pub use error::FavarError;
pub use favar::Favar;
pub use pca::FactorModel;

// Re-export the factor VAR layer (and, through it, the shared linear
// algebra and dense backend) so downstream crates see one faer version
// and one `Trend` / `VarResults` type.
pub use tsecon_var;
pub use tsecon_var::Trend;
