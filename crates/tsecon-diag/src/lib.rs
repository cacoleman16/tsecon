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
//! * [`adf`] — augmented Dickey-Fuller unit-root test (Said-Dickey 1984)
//!   with fixed or automatic (AIC/BIC/t-stat) lag selection and MacKinnon
//!   (1994, 2010) response-surface p-values and critical values
//!   ([`mackinnon_p`], [`mackinnon_crit`]).
//! * [`kpss`] — KPSS stationarity test (Kwiatkowski et al. 1992) with
//!   legacy and Hobijn-Franses-Ooms automatic Bartlett bandwidths.
//! * [`check_stationarity`] — the joint ADF + KPSS confirmatory decision
//!   workflow, classifying the evidence into a quadrant with a concrete
//!   recommendation (proceed / difference / detrend).
//!
//! Each test result offers `report(alpha)`, producing a
//! [`DiagnosticReport`] whose interpretation string states the decision in
//! plain language and points at the next diagnostic — the library's
//! "errors that teach" pillar; the unit-root workflow returns a
//! [`StationarityReport`] in the same spirit.
//!
//! P-values come from the chi-squared survival function in
//! [`tsecon_stats`]; golden-value tests pin every statistic and p-value
//! against statsmodels 0.14.6 fixtures at `1e-12` relative (`1e-8` for
//! the unit-root layer).

#![warn(missing_docs)]

mod acf;
mod arch;
mod error;
mod mackinnon;
mod normality;
mod ols;
mod portmanteau;
mod report;
mod unitroot;
mod validate;

pub use acf::{acf, pacf_ols, pacf_yw, AcfResult};
pub use arch::{arch_lm, ArchLmResult};
pub use error::DiagError;
pub use mackinnon::{mackinnon_crit, mackinnon_p, AdfCriticalValues};
pub use normality::{jarque_bera, JarqueBeraResult};
pub use portmanteau::{ljung_box, PortmanteauResult};
pub use report::DiagnosticReport;
pub use unitroot::{
    adf, check_stationarity, check_stationarity_at, kpss, AdfLagSelection, AdfRegression,
    AdfResult, KpssCriticalValues, KpssLags, KpssRegression, KpssResult, Recommendation,
    StationarityQuadrant, StationarityReport,
};
