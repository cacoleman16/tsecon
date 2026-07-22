//! Im-Pesaran-Shin (2003) standardized t-bar statistic.
//!
//! The per-unit ADF t-statistics are averaged into
//!
//! ```text
//! t_bar = (1/N) sum_i t_i
//! ```
//!
//! and standardized to a `N(0,1)` statistic using the tabulated mean and
//! variance of `t_{iT}(p_i, 0)` (Im-Pesaran-Shin 2003, Table 3; see
//! [`crate::tables`]):
//!
//! ```text
//! W_tbar = sqrt(N) * (t_bar - (1/N) sum_i E[t_i]) / sqrt((1/N) sum_i Var[t_i]) ~ N(0,1).
//! ```
//!
//! Each unit contributes its own moments keyed by its effective sample size
//! `l_i = T_i - p_i - 1` (its ADF regression `nobs`) and its lag `p_i` — the
//! `plm::purtest` `Wtbar` convention — so heterogeneous lengths and lag
//! orders are handled exactly. The test rejects the panel-unit-root null in
//! the LEFT tail, so `p_value = Phi(W_tbar)`.

use tsecon_stats::{ContinuousDist, StdNormal};

use crate::tables::ips_moments;

/// Output of the IPS combination.
pub(crate) struct IpsOut {
    /// The standardized statistic `W_tbar ~ N(0,1)`.
    pub w_tbar: f64,
    /// Left-tail standard-normal p-value `Phi(W_tbar)`.
    pub p_value: f64,
    /// The unstandardized average `t_bar`.
    pub t_bar: f64,
}

/// Combine the per-unit ADF t-statistics `tstats` (with augmentation lags
/// `lags` and effective sample sizes `nobs`, one per unit) into the IPS
/// `W_tbar` statistic. `trend` selects the intercept-plus-trend moment table
/// (`false` = intercept only).
pub(crate) fn ips_combine(tstats: &[f64], lags: &[usize], nobs: &[usize], trend: bool) -> IpsOut {
    let n = tstats.len();
    let nf = n as f64;
    let t_bar = tstats.iter().sum::<f64>() / nf;

    let mut sum_e = 0.0;
    let mut sum_v = 0.0;
    for i in 0..n {
        let (e, v) = ips_moments(nobs[i] as f64, lags[i], trend);
        sum_e += e;
        sum_v += v;
    }
    let e_bar = sum_e / nf;
    let v_bar = sum_v / nf;

    let w_tbar = nf.sqrt() * (t_bar - e_bar) / v_bar.sqrt();
    let p_value = StdNormal.cdf(w_tbar);

    IpsOut {
        w_tbar,
        p_value,
        t_bar,
    }
}
