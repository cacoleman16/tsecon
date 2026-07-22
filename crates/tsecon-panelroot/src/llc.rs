//! Levin-Lin-Chu (2002) pooled-ADF panel unit-root test.
//!
//! Unlike IPS/Fisher, LLC fits a genuinely different model: a *common*
//! autoregressive root pooled across the panel, with heterogeneous
//! deterministics and short-run dynamics. The six-step procedure (Levin,
//! Lin & Chu 2002, *J. Econometrics* 108, §II) is:
//!
//! 1. For each unit run two auxiliary OLS on the deterministics and the
//!    lagged differences (but NOT the lagged level): regress `dy_t` to get
//!    residuals `ehat`, and `y_{t-1}` to get residuals `vhat`.
//! 2. Normalize both by the unit's ADF regression standard error
//!    `sigma_eps_i`: `etil = ehat / sigma_eps_i`, `vtil = vhat / sigma_eps_i`.
//! 3. Form the long-run/short-run standard-deviation ratio
//!    `s_i = sigma_y_i / sigma_eps_i` from a kernel long-run variance of the
//!    unit's first differences, and average it to `S_N`.
//! 4. Pool: `delta_hat = sum(vtil etil) / sum(vtil^2)`, with the pooled
//!    residual variance and the t-ratio `t_delta`.
//! 5. Bias-adjust with the tabulated `mu*`, `sigma*` (Levin-Lin-Chu 2002,
//!    Table 2; see [`crate::tables`]) to obtain `t*_delta ~ N(0,1)`.
//!
//! The convention matches `plm::purtest(test = "levinlin")`: the per-unit
//! `sigma_eps_i` and the pooled t-ratio use the residual-degrees-of-freedom
//! divisor, the long-run variance uses the Levin-Lin-Chu bandwidth rule
//! `round(3.21 T^{1/3})` with a Bartlett kernel by default, and the `mu*` /
//! `sigma*` adjustments are looked up at the *full* common length `T`. The
//! test rejects the panel-unit-root null in the LEFT tail, so
//! `p_value = Phi(t*_delta)`.

use tsecon_hac::{lrv, ols, Kernel};
use tsecon_stats::{ContinuousDist, StdNormal};

use crate::error::PanelRootError;
use crate::tables::llc_adj;
use crate::PanelRootOpts;

use tsecon_diag::AdfRegression;

/// Output of the LLC pipeline.
pub(crate) struct LlcOut {
    /// Bias-adjusted statistic `t*_delta ~ N(0,1)` (the headline statistic).
    pub t_star: f64,
    /// Left-tail standard-normal p-value `Phi(t*_delta)`.
    pub p_value: f64,
    /// The pooled common-root estimate `delta_hat`.
    pub delta_hat: f64,
    /// The unadjusted pooled t-ratio `t_delta`.
    pub t_delta: f64,
    /// The average long-run/short-run standard-deviation ratio `S_N`.
    pub s_n: f64,
    /// The average per-unit usable-row count `T~` (= `T - p - 1`).
    pub t_bar_periods: f64,
}

/// Number of deterministic columns for a case.
fn ntrend(regression: AdfRegression) -> usize {
    match regression {
        AdfRegression::NoConstant => 0,
        AdfRegression::Constant => 1,
        AdfRegression::ConstantTrend => 2,
    }
}

/// Deterministic columns for `rows` observations: a constant and/or a linear
/// trend `1..=rows`. Because every case that carries a trend also carries a
/// constant, the exact origin of the trend is immaterial to the residuals.
fn determ_cols(regression: AdfRegression, rows: usize) -> Vec<Vec<f64>> {
    let mut cols = Vec::new();
    if ntrend(regression) >= 1 {
        cols.push(vec![1.0; rows]);
    }
    if ntrend(regression) >= 2 {
        cols.push((1..=rows).map(|i| i as f64).collect());
    }
    cols
}

/// Round to the nearest integer, ties to even — matching R's `round`, so the
/// default bandwidth `round(3.21 T^{1/3})` reproduces `plm`'s exactly.
fn round_half_even(x: f64) -> f64 {
    let r = x.round();
    if (x - x.floor() - 0.5).abs() < 1e-9 {
        // Exactly halfway: pick the even neighbour.
        let lower = x.floor();
        if (lower as i64) % 2 == 0 {
            lower
        } else {
            lower + 1.0
        }
    } else {
        r
    }
}

/// The centered first differences a unit contributes to its long-run
/// variance: `dx_t = y_t - y_{t-1}`, demeaned (intercept) or OLS-detrended
/// (trend), matching `plm`'s `longrunvar`.
fn centered_diffs(y: &[f64], regression: AdfRegression) -> Result<Vec<f64>, PanelRootError> {
    let t = y.len();
    let dx: Vec<f64> = (1..t).map(|k| y[k] - y[k - 1]).collect();
    Ok(match regression {
        AdfRegression::NoConstant => dx,
        AdfRegression::Constant => {
            let mean = dx.iter().sum::<f64>() / dx.len() as f64;
            dx.iter().map(|v| v - mean).collect()
        }
        AdfRegression::ConstantTrend => {
            let m = dx.len();
            let cols = vec![
                vec![1.0; m],
                (1..=m).map(|i| i as f64).collect::<Vec<f64>>(),
            ];
            let fit = ols(&dx, &cols).map_err(|e| PanelRootError::Hac {
                unit: usize::MAX,
                source: e,
            })?;
            fit.residuals
        }
    })
}

/// Run the LLC pipeline on a balanced panel. `lags[i]` is the per-unit ADF
/// augmentation from the shared front-half; `regression` and `opts` fix the
/// deterministics and the long-run-variance kernel/bandwidth.
pub(crate) fn llc(
    units: &[Vec<f64>],
    lags: &[usize],
    regression: AdfRegression,
    opts: &PanelRootOpts,
) -> Result<LlcOut, PanelRootError> {
    let n = units.len();
    let t = units[0].len();
    let nt = ntrend(regression);
    let kernel: Kernel = opts.lrv_kernel;
    let default_bw = round_half_even(3.21 * (t as f64).powf(1.0 / 3.0));
    let bandwidth = opts.lrv_bandwidth.unwrap_or(default_bw);

    let mut etil_all: Vec<f64> = Vec::new();
    let mut vtil_all: Vec<f64> = Vec::new();
    let mut s_list: Vec<f64> = Vec::with_capacity(n);
    let mut rows_list: Vec<usize> = Vec::with_capacity(n);

    for (i, y) in units.iter().enumerate() {
        let ell = lags[i];
        // Usable rows: t = ell+1 .. T-1 (0-indexed), giving rows = T-ell-1.
        if t < ell + 2 {
            return Err(PanelRootError::UnitTooShort { unit: i, len: t });
        }
        let rows = t - ell - 1;
        let t0 = ell + 1;
        let k = 1 + ell + nt;
        if rows <= k {
            return Err(PanelRootError::UnitTooShort { unit: i, len: t });
        }

        // Response dy_t and the lagged level y_{t-1}.
        let dy: Vec<f64> = (t0..t).map(|s| y[s] - y[s - 1]).collect();
        let ly1: Vec<f64> = (t0..t).map(|s| y[s - 1]).collect();
        // Lagged differences dy_{t-j}, j = 1..ell.
        let dylags: Vec<Vec<f64>> = (1..=ell)
            .map(|j| {
                (t0..t)
                    .map(|s| y[s - j] - y[s - j - 1])
                    .collect::<Vec<f64>>()
            })
            .collect();

        // --- Step 2: sigma_eps_i from the full ADF regression.
        let mut full: Vec<Vec<f64>> = Vec::with_capacity(k);
        full.push(ly1.clone());
        full.extend(dylags.iter().cloned());
        full.extend(determ_cols(regression, rows));
        let full_fit = ols(&dy, &full).map_err(|e| PanelRootError::Hac { unit: i, source: e })?;
        let rss_full: f64 = full_fit.residuals.iter().map(|r| r * r).sum();
        let sigma_eps = (rss_full / (rows - k) as f64).sqrt();

        // --- Step 1: auxiliary OLS (deterministics + lagged diffs, no level).
        let mut aux: Vec<Vec<f64>> = determ_cols(regression, rows);
        aux.extend(dylags.iter().cloned());
        let (ehat, vhat) = if aux.is_empty() {
            // none + lag 0: nothing to project out.
            (dy.clone(), ly1.clone())
        } else {
            let fit_e = ols(&dy, &aux).map_err(|e| PanelRootError::Hac { unit: i, source: e })?;
            let fit_v = ols(&ly1, &aux).map_err(|e| PanelRootError::Hac { unit: i, source: e })?;
            (fit_e.residuals, fit_v.residuals)
        };

        for k in 0..rows {
            etil_all.push(ehat[k] / sigma_eps);
            vtil_all.push(vhat[k] / sigma_eps);
        }

        // --- Step 3: long-run/short-run ratio s_i.
        let dx = centered_diffs(y, regression).map_err(|e| match e {
            PanelRootError::Hac { source, .. } => PanelRootError::Hac { unit: i, source },
            other => other,
        })?;
        let sigma_y2 =
            lrv(&dx, kernel, bandwidth).map_err(|e| PanelRootError::Hac { unit: i, source: e })?;
        s_list.push(sigma_y2.sqrt() / sigma_eps);
        rows_list.push(rows);
    }

    // --- Step 4: pooled no-intercept regression of etil on vtil.
    let sum_vv: f64 = vtil_all.iter().map(|v| v * v).sum();
    if sum_vv == 0.0 {
        return Err(PanelRootError::DegeneratePool);
    }
    let sum_ve: f64 = vtil_all
        .iter()
        .zip(etil_all.iter())
        .map(|(v, e)| v * e)
        .sum();
    let delta_hat = sum_ve / sum_vv;
    let rss_pool: f64 = etil_all
        .iter()
        .zip(vtil_all.iter())
        .map(|(e, v)| {
            let r = e - delta_hat * v;
            r * r
        })
        .sum();

    let npool: usize = rows_list.iter().sum();
    let tilde_t = npool as f64 / n as f64; // = mean_i (T - p_i - 1)
                                           // Bias-term residual variance (N*T~ = npool divisor, plm convention).
    let sigma_etil2 = rss_pool / npool as f64;
    // Pooled t-ratio SE with the residual-df divisor (plm dfcor = TRUE).
    let sd_delta = (rss_pool / (npool as f64 - 1.0)).sqrt() / sum_vv.sqrt();
    let t_delta = delta_hat / sd_delta;
    let s_n = s_list.iter().sum::<f64>() / n as f64;

    // --- Step 5: bias adjustment with mu*/sigma* keyed by the full T.
    let (mu_star, sigma_star) = llc_adj(t as f64, regression);
    let bias = npool as f64 * s_n / sigma_etil2 * sd_delta * mu_star;
    let t_star = (t_delta - bias) / sigma_star;
    let p_value = StdNormal.cdf(t_star);

    Ok(LlcOut {
        t_star,
        p_value,
        delta_hat,
        t_delta,
        s_n,
        t_bar_periods: tilde_t,
    })
}
