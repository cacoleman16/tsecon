//! # tsecon-panelroot — first-generation panel unit-root tests
//!
//! Three classic tests of the joint null "every cross-section unit has a
//! unit root", behind one [`panel_unit_root`] entry point. All three share a
//! front half — the already-pinned per-unit augmented Dickey-Fuller test
//! ([`tsecon_diag::adf`], matched to statsmodels at `1e-8`) — and differ only
//! in how the per-unit ADF outputs are combined:
//!
//! - **Fisher** ([`PanelRootTest::Fisher`]): Maddala-Wu (1999)
//!   `P = -2 sum ln p_i ~ chi^2(2N)` (right tail) plus Choi's (2001)
//!   inverse-normal `Z ~ N(0,1)` (left tail). A deterministic, exact-arithmetic
//!   combination of the per-unit p-values — it inherits the ADF's validated
//!   accuracy and is the crate's strongest-anchored statistic.
//! - **IPS** ([`PanelRootTest::Ips`]): Im-Pesaran-Shin (2003) — average the
//!   per-unit t-statistics into `t_bar`, then standardize to
//!   `W_tbar ~ N(0,1)` (left tail) with the tabulated mean/variance of
//!   `t_{iT}(p,0)` (Table 3).
//! - **LLC** ([`PanelRootTest::Llc`]): Levin-Lin-Chu (2002) — a pooled ADF
//!   with a common root, a per-unit long-run-variance ratio, a pooled OLS,
//!   and the tabulated `mu*`/`sigma*` bias adjustment (Table 2) giving
//!   `t*_delta ~ N(0,1)` (left tail). Requires a balanced panel.
//!
//! ## Reference and validation
//!
//! The per-unit ADF is reused verbatim from [`tsecon_diag`]; the kernel
//! long-run variance and the auxiliary/pooled OLS from [`tsecon_hac`]; the
//! normal and chi-squared tails from [`tsecon_stats`]. Only the combination
//! layer and the two transcribed moment-table families (see [`tables`]) are
//! new. Conventions follow `plm::purtest` (R): the crate's golden fixtures
//! reproduce `plm`'s `Wtbar`, `levinlin`, `madwu`, and `invnormal` statistics
//! — and, for Fisher, an independent statsmodels/scipy reference — to
//! floating-point precision.
//!
//! ## Input shape
//!
//! Supply the panel as a slice of per-unit series (`&[Vec<f64>]`). IPS and
//! Fisher accept unbalanced panels (each unit uses its own length and lag);
//! LLC requires a common length `T`.

#![warn(missing_docs)]

mod error;
mod fisher;
mod ips;
mod llc;
mod tables;

pub use error::PanelRootError;

use tsecon_diag::{adf, AdfLagSelection, AdfRegression};
use tsecon_hac::Kernel;

/// Which panel unit-root test to run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PanelRootTest {
    /// Levin-Lin-Chu (2002) pooled-ADF common-root test (balanced only).
    Llc,
    /// Im-Pesaran-Shin (2003) standardized average-t test.
    Ips,
    /// Fisher-type p-value combination: Maddala-Wu (1999) with Choi (2001).
    Fisher,
}

/// Options for the Levin-Lin-Chu long-run variance. Ignored by IPS/Fisher.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PanelRootOpts {
    /// Kernel for the per-unit long-run variance of the first differences
    /// (default [`Kernel::Bartlett`], the Levin-Lin-Chu choice).
    pub lrv_kernel: Kernel,
    /// Bandwidth for the long-run variance. `None` uses the Levin-Lin-Chu
    /// rule `round(3.21 T^{1/3})`.
    pub lrv_bandwidth: Option<f64>,
}

impl Default for PanelRootOpts {
    fn default() -> Self {
        Self {
            lrv_kernel: Kernel::Bartlett,
            lrv_bandwidth: None,
        }
    }
}

/// Test-specific extra outputs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PanelRootDetail {
    /// IPS extras.
    Ips {
        /// The unstandardized average of the per-unit t-statistics.
        t_bar: f64,
    },
    /// LLC extras.
    Llc {
        /// The pooled common-root estimate.
        delta_hat: f64,
        /// The unadjusted pooled t-ratio.
        t_delta: f64,
        /// The average long-run/short-run standard-deviation ratio.
        s_n: f64,
        /// The average per-unit usable-row count `T~ = T - p - 1`.
        t_bar_periods: f64,
    },
    /// Fisher extras.
    Fisher {
        /// Choi's (2001) inverse-normal statistic `Z`.
        choi_z: f64,
        /// Left-tail standard-normal p-value of `Z`.
        choi_z_pvalue: f64,
    },
}

/// Result of a panel unit-root test.
///
/// The null is a unit root in every unit; the alternative is stationarity in
/// (a positive fraction of) the panel. For IPS and LLC small statistics
/// favour stationarity (`p_value = Phi(statistic)`, a LEFT tail); for the
/// Maddala-Wu headline of Fisher, a LARGE statistic favours stationarity
/// (`p_value = chi2_sf`, a RIGHT tail).
#[derive(Debug, Clone, PartialEq)]
pub struct PanelRootResult {
    /// Which test produced this result.
    pub test: PanelRootTest,
    /// The headline statistic (`W_tbar` for IPS; `t*_delta` for LLC; the
    /// Maddala-Wu `P` for Fisher).
    pub statistic: f64,
    /// The p-value of `statistic` for the joint unit-root null.
    pub p_value: f64,
    /// Per-unit ADF tau statistics.
    pub per_unit_tstat: Vec<f64>,
    /// Per-unit ADF MacKinnon p-values (clamped to `[1e-16, 1 - 1e-16]` for
    /// Fisher, as used in the combination; raw for IPS/LLC).
    pub per_unit_pvalue: Vec<f64>,
    /// Per-unit ADF augmentation lags.
    pub per_unit_lags: Vec<usize>,
    /// Per-unit ADF effective sample sizes (`T_i - 1 - lag_i`).
    pub per_unit_nobs: Vec<usize>,
    /// The number of cross-section units.
    pub n_units: usize,
    /// The deterministic specification tested.
    pub regression: AdfRegression,
    /// Test-specific extra outputs.
    pub detail: PanelRootDetail,
}

/// Run a first-generation panel unit-root test.
///
/// `units` holds one series per cross-section unit. `test` selects the
/// combination; `regression` fixes the per-unit ADF deterministics (`"n"` is
/// rejected for IPS); `lags` is the per-unit ADF lag rule (a fixed common
/// lag, or an automatic AIC/BIC/t-stat search); `opts` configures the LLC
/// long-run variance.
///
/// # Errors
///
/// * [`PanelRootError::TooFewUnits`] for fewer than two units.
/// * [`PanelRootError::NonFinite`] if any unit contains NaN/inf.
/// * [`PanelRootError::IpsNoConstant`] for `test = Ips` with
///   `regression = NoConstant`.
/// * [`PanelRootError::UnbalancedForLlc`] for `test = Llc` on an unbalanced
///   panel.
/// * [`PanelRootError::UnitTooShort`] / [`PanelRootError::Adf`] /
///   [`PanelRootError::Hac`] / [`PanelRootError::DegeneratePool`] /
///   [`PanelRootError::Stats`] for per-unit or combination failures.
pub fn panel_unit_root(
    units: &[Vec<f64>],
    test: PanelRootTest,
    regression: AdfRegression,
    lags: AdfLagSelection,
    opts: &PanelRootOpts,
) -> Result<PanelRootResult, PanelRootError> {
    let n = units.len();
    if n < 2 {
        return Err(PanelRootError::TooFewUnits { n });
    }
    // Finiteness and (for LLC) balance.
    for (i, y) in units.iter().enumerate() {
        if y.iter().any(|v| !v.is_finite()) {
            return Err(PanelRootError::NonFinite { unit: i });
        }
    }
    if test == PanelRootTest::Ips && regression == AdfRegression::NoConstant {
        return Err(PanelRootError::IpsNoConstant);
    }
    if test == PanelRootTest::Llc {
        let t0 = units[0].len();
        for (i, y) in units.iter().enumerate().skip(1) {
            if y.len() != t0 {
                return Err(PanelRootError::UnbalancedForLlc {
                    unit: i,
                    expected: t0,
                    got: y.len(),
                });
            }
        }
    }

    // Shared front half: per-unit ADF.
    let mut per_unit_tstat = Vec::with_capacity(n);
    let mut per_unit_pvalue = Vec::with_capacity(n);
    let mut per_unit_lags = Vec::with_capacity(n);
    let mut per_unit_nobs = Vec::with_capacity(n);
    for (i, y) in units.iter().enumerate() {
        let r = adf(y, regression, lags).map_err(|e| PanelRootError::Adf { unit: i, source: e })?;
        per_unit_tstat.push(r.statistic);
        per_unit_pvalue.push(r.p_value);
        per_unit_lags.push(r.used_lag);
        per_unit_nobs.push(r.nobs);
    }

    let trend = regression == AdfRegression::ConstantTrend;

    let (statistic, p_value, detail) = match test {
        PanelRootTest::Fisher => {
            let out = fisher::fisher_combine(&per_unit_pvalue)?;
            // Report the clamped p-values actually used in the combination.
            per_unit_pvalue = out.clamped_pvalues;
            (
                out.maddala_wu,
                out.mw_pvalue,
                PanelRootDetail::Fisher {
                    choi_z: out.choi_z,
                    choi_z_pvalue: out.choi_z_pvalue,
                },
            )
        }
        PanelRootTest::Ips => {
            let out = ips::ips_combine(&per_unit_tstat, &per_unit_lags, &per_unit_nobs, trend);
            (
                out.w_tbar,
                out.p_value,
                PanelRootDetail::Ips { t_bar: out.t_bar },
            )
        }
        PanelRootTest::Llc => {
            let out = llc::llc(units, &per_unit_lags, regression, opts)?;
            (
                out.t_star,
                out.p_value,
                PanelRootDetail::Llc {
                    delta_hat: out.delta_hat,
                    t_delta: out.t_delta,
                    s_n: out.s_n,
                    t_bar_periods: out.t_bar_periods,
                },
            )
        }
    };

    Ok(PanelRootResult {
        test,
        statistic,
        p_value,
        per_unit_tstat,
        per_unit_pvalue,
        per_unit_lags,
        per_unit_nobs,
        n_units: n,
        regression,
        detail,
    })
}
