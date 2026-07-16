//! # tsecon-diag
//!
//! First diagnostics slice for the `tsecon` time-series econometrics
//! library (diagnostics track; see ROADMAP §01): the exploratory
//! second-moment tools and the core residual test battery, with
//! statsmodels as the golden reference.
//!
//! * [`acf`] — sample autocorrelations (biased `n` or adjusted `n - k`
//!   denominator) with Bartlett standard-error bands (Bartlett 1946;
//!   Brockwell & Davis 1991).
//! * [`pacf_yw`] / [`pacf_ols`] — partial autocorrelations via
//!   Yule-Walker/Durbin-Levinson (statsmodels `"ywm"`) and via successive
//!   lag regressions (statsmodels `"ols"`).
//! * [`ljung_box`] — Ljung-Box (1978) and Box-Pierce (1970) portmanteau
//!   whiteness tests over a lag range with chi-squared p-values.
//! * [`jarque_bera`] — Jarque-Bera (1980) normality test with skewness and
//!   (raw) kurtosis.
//! * [`arch_lm`] — Engle's (1982) LM test for ARCH effects.
//!
//! Each test result offers `report(alpha)`, producing a
//! [`DiagnosticReport`] whose interpretation string states the decision in
//! plain language and points at the next diagnostic — the library's
//! "errors that teach" pillar.
//!
//! P-values come from the chi-squared survival function in
//! [`tsecon_stats`]; golden-value tests pin every statistic and p-value
//! against statsmodels 0.14.6 fixtures at `1e-12` relative.

#![warn(missing_docs)]

mod acf;
mod arch;
mod error;
mod normality;
mod ols;
mod portmanteau;
mod report;
mod validate;

pub use acf::{acf, pacf_ols, pacf_yw, AcfResult};
pub use arch::{arch_lm, ArchLmResult};
pub use error::DiagError;
pub use normality::{jarque_bera, JarqueBeraResult};
pub use portmanteau::{ljung_box, PortmanteauResult};
pub use report::DiagnosticReport;
