//! Stock-Watson (1993) / Saikkonen (1991) **dynamic OLS** ([`dols`]): the
//! static cointegrating regression augmented with leads and lags of the
//! regressor differences, so that the augmented error is orthogonal to the
//! regressor innovations and OLS on the augmented design is asymptotically
//! mixed normal.
//!
//! The regression (`arch.unitroot.cointegration.DynamicOLS`) is
//!
//! ```text
//! y_t = x_t' beta + d_t' delta + sum_{j=-p}^{q} Delta x_{t+j}' gamma_j + e_t,   t = p + 2 .. T - q,
//! ```
//!
//! with `p` lags and `q` leads (the contemporaneous difference `j = 0` is
//! always included), on the `T - 1 - p - q` rows every term is defined
//! on. The design is ordered `[x, deterministics, Delta x_{t-p}, ...,
//! Delta x_t, ..., Delta x_{t+q}]` (each difference block holds the `k_x`
//! regressors), the trend running `1..nobs` over the regression sample.
//!
//! When `lags` and/or `leads` are not fixed, the pair is chosen by
//! minimising an information criterion on the **common sample of the
//! largest candidate** (`p <= max_lag`, `q <= max_lead`, both defaulting
//! to `ceil(12 (T/100)^(1/4))`; `common = true` restricts to `p = q`):
//!
//! ```text
//! IC(p, q) = ln(RSS/nobs) + n_params * c / nobs,   c = 2 (AIC), 2 ln ln nobs (HQIC), ln nobs (BIC),
//! ```
//!
//! ties going to the smaller lag, then the smaller lead. The chosen model
//! is then refit on its own (larger) sample.
//!
//! Inference (`cov_type`): `"unadjusted"` — `sigma2_HAC (Z'Z/nobs)^{-1} /
//! nobs` with `sigma2_HAC` the kernel long-run variance of the residuals;
//! `"robust"` — the kernel-HAC sandwich `(Z'Z/nobs)^{-1} S_HAC
//! (Z'Z/nobs)^{-1} / nobs` on the scores `z_t e_t`. The kernel window,
//! bandwidth rules and `force_int` are those of [`crate::fmols`]
//! (`arch`'s DOLS defaults `force_int = false`); `df_adjust` scales the
//! covariance by `nobs/(nobs - n_params)`.
//!
//! References: Saikkonen (1991), Econometric Theory 7(1); Stock & Watson
//! (1993), Econometrica 61(4).

use tsecon_hac::Kernel;
use tsecon_linalg::faer::{Mat, MatRef};

use crate::error::CointError;
use crate::fmols::{
    check_yx, det_columns, hcat, inference, inv_spd_scaled, long_run_covariance, lstsq,
    param_names, resolve_bandwidth, static_ols, BandwidthRule, CointTrend,
};

/// Information criterion for the lead/lag search of [`dols`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DolsIc {
    /// Akaike, penalty `2`.
    Aic,
    /// Schwarz / Bayesian, penalty `ln nobs` (the default).
    Bic,
    /// Hannan-Quinn, penalty `2 ln ln nobs`.
    Hqic,
}

impl DolsIc {
    /// Parses `"aic"` / `"bic"` / `"hqic"` (case-insensitive).
    pub fn parse(code: &str) -> Option<Self> {
        match code.to_ascii_lowercase().as_str() {
            "aic" => Some(Self::Aic),
            "bic" => Some(Self::Bic),
            "hqic" => Some(Self::Hqic),
            _ => None,
        }
    }

    /// The canonical spelling.
    #[must_use]
    pub fn code(self) -> &'static str {
        match self {
            Self::Aic => "aic",
            Self::Bic => "bic",
            Self::Hqic => "hqic",
        }
    }

    fn penalty(self, nobs: f64) -> f64 {
        match self {
            Self::Aic => 2.0,
            Self::Hqic => 2.0 * nobs.ln().ln(),
            Self::Bic => nobs.ln(),
        }
    }
}

/// Parameter-covariance estimator of [`dols`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DolsCovType {
    /// `sigma2_HAC (Z'Z)^{-1}`: the residual long-run variance times the
    /// classical bread (`arch`'s `"unadjusted"` / `"homoskedastic"`, the
    /// default).
    Unadjusted,
    /// Kernel-HAC sandwich on the scores (`arch`'s `"robust"` /
    /// `"kernel"`).
    Robust,
}

impl DolsCovType {
    /// Parses `"unadjusted"` / `"homoskedastic"` / `"robust"` / `"kernel"`.
    pub fn parse(code: &str) -> Option<Self> {
        match code.to_ascii_lowercase().as_str() {
            "unadjusted" | "homoskedastic" => Some(Self::Unadjusted),
            "robust" | "kernel" => Some(Self::Robust),
            _ => None,
        }
    }

    /// The canonical spelling.
    #[must_use]
    pub fn code(self) -> &'static str {
        match self {
            Self::Unadjusted => "unadjusted",
            Self::Robust => "robust",
        }
    }
}

/// Options of [`dols`] (defaults are `arch`'s).
#[derive(Debug, Clone, PartialEq)]
pub struct DolsOptions {
    /// Deterministic terms. Default [`CointTrend::Constant`].
    pub trend: CointTrend,
    /// Fixed number of lags, or `None` (default) to select it.
    pub lags: Option<usize>,
    /// Fixed number of leads, or `None` (default) to select it.
    pub leads: Option<usize>,
    /// Restrict the search to `lags == leads` (and, when both are fixed,
    /// require them equal). Default `false`.
    pub common: bool,
    /// Largest lag the search considers; `None` (default) is
    /// `ceil(12 (T/100)^(1/4))`. Ignored when `lags` is fixed.
    pub max_lag: Option<usize>,
    /// Largest lead the search considers; same default. Ignored when
    /// `leads` is fixed.
    pub max_lead: Option<usize>,
    /// Information criterion of the search. Default [`DolsIc::Bic`].
    pub ic: DolsIc,
    /// Covariance estimator. Default [`DolsCovType::Unadjusted`].
    pub cov_type: DolsCovType,
    /// Kernel of the long-run variance / HAC meat. Default
    /// [`Kernel::Bartlett`].
    pub kernel: Kernel,
    /// Explicit bandwidth, or `None` (default) for the automatic rule.
    pub bandwidth: Option<f64>,
    /// Automatic rule when `bandwidth` is `None`. Default
    /// [`BandwidthRule::NeweyWest`].
    pub bandwidth_rule: BandwidthRule,
    /// Ceil the bandwidth to an integer. Default `false` (`arch`'s DOLS
    /// default).
    pub force_int: bool,
    /// Scale the covariance by `nobs/(nobs - n_params)`. Default `false`.
    pub df_adjust: bool,
}

impl Default for DolsOptions {
    fn default() -> Self {
        Self {
            trend: CointTrend::Constant,
            lags: None,
            leads: None,
            common: false,
            max_lag: None,
            max_lead: None,
            ic: DolsIc::Bic,
            cov_type: DolsCovType::Unadjusted,
            kernel: Kernel::Bartlett,
            bandwidth: None,
            bandwidth_rule: BandwidthRule::NeweyWest,
            force_int: false,
            df_adjust: false,
        }
    }
}

/// Result of [`dols`].
#[derive(Debug, Clone, PartialEq)]
pub struct DolsResult {
    /// The cointegrating vector: the `k_x` regressor coefficients, then
    /// the deterministics of `trend`.
    pub params: Vec<f64>,
    /// Asymptotic standard errors of `params`.
    pub se: Vec<f64>,
    /// `params / se`.
    pub tvalues: Vec<f64>,
    /// Two-sided normal p-values.
    pub pvalues: Vec<f64>,
    /// Covariance of `params` (rows).
    pub cov: Vec<Vec<f64>>,
    /// Names of `params` (`x1`, ..., `const`, `trend`, `quadratic_trend`).
    pub param_names: Vec<String>,
    /// Every coefficient of the augmented regression, design order:
    /// `params`, then the difference blocks `Delta x_{t-lags}`, ...,
    /// `Delta x_t`, ..., `Delta x_{t+leads}` (`k_x` columns each).
    pub full_params: Vec<f64>,
    /// Standard errors of `full_params`.
    pub full_se: Vec<f64>,
    /// Covariance of `full_params` (rows).
    pub full_cov: Vec<Vec<f64>>,
    /// Names of `full_params` (`D.x1.LAG2`, `D.x1`, `D.x1.LEAD1`, ...).
    pub full_param_names: Vec<String>,
    /// Residuals of the augmented regression (length `nobs`).
    pub resid: Vec<f64>,
    /// Rows of the augmented regression, `T - 1 - lags - leads`.
    pub nobs: usize,
    /// Sample size `T` of the data.
    pub n_total: usize,
    /// Number of stochastic regressors `k_x`.
    pub n_x: usize,
    /// Number of deterministic columns.
    pub n_det: usize,
    /// Number of regressors of the augmented design.
    pub n_params: usize,
    /// The deterministic specification.
    pub trend: CointTrend,
    /// Lags used.
    pub lags: usize,
    /// Leads used.
    pub leads: usize,
    /// Whether the lead/lag pair came from the information-criterion
    /// search (`false` when both were fixed).
    pub selected: bool,
    /// The criterion of the search.
    pub ic: DolsIc,
    /// The minimised criterion value on the common search sample (`NaN`
    /// when both lags and leads were fixed).
    pub ic_value: f64,
    /// The largest lag the search considered (`lags` when fixed).
    pub max_lag: usize,
    /// The largest lead the search considered (`leads` when fixed).
    pub max_lead: usize,
    /// Echo of `common`.
    pub common: bool,
    /// The covariance estimator.
    pub cov_type: DolsCovType,
    /// The kernel used.
    pub kernel: Kernel,
    /// The bandwidth actually used (on the residuals under
    /// `"unadjusted"`, on the scores under `"robust"`).
    pub bandwidth: f64,
    /// The rule that produced `bandwidth`, or `None` when explicit.
    pub bandwidth_rule: Option<BandwidthRule>,
    /// Echo of `force_int`.
    pub force_int: bool,
    /// Echo of `df_adjust`.
    pub df_adjust: bool,
    /// Kernel long-run variance of the residuals at `bandwidth` (times
    /// `nobs/(nobs - n_params)` under `df_adjust`) — the `sigma2_HAC` of
    /// the unadjusted covariance.
    pub long_run_variance: f64,
    /// `R^2` of the augmented regression (centred when `trend` has a
    /// constant).
    pub rsquared: f64,
    /// Adjusted `R^2`, `1 - (nobs - k_const)/(nobs - n_params) (1 - R^2)`.
    pub rsquared_adj: f64,
    /// Plain static OLS of `y` on `[x, deterministics]` over the full
    /// sample, for comparison (same order as `params`).
    pub ols_params: Vec<f64>,
    /// Classical OLS standard errors of `ols_params` — comparison only,
    /// **not** valid for inference on a cointegrating vector.
    pub ols_se: Vec<f64>,
}

/// The augmented design at `(lags, leads)`: rows `t = lags + 1 ..= n - 1 -
/// leads` of `[x_t, det, Delta x_{t-lags}, ..., Delta x_{t+leads}]` and
/// the matching `y`.
fn design(
    y: &[f64],
    x: MatRef<'_, f64>,
    trend: CointTrend,
    lags: usize,
    leads: usize,
) -> (Mat<f64>, Mat<f64>) {
    let n = y.len();
    let kx = x.ncols();
    let nobs = n - 1 - lags - leads;
    let n_det = trend.n_det();
    let n_blocks = lags + leads + 1;
    let det = det_columns(trend, nobs);
    let rhs = Mat::from_fn(nobs, kx + n_det + n_blocks * kx, |i, j| {
        let t = i + lags + 1;
        if j < kx {
            x[(t, j)]
        } else if j < kx + n_det {
            det[(i, j - kx)]
        } else {
            let c = j - kx - n_det;
            let block = c / kx;
            let col = c % kx;
            // block 0 is Delta x_{t-lags}, block `lags` is Delta x_t.
            let s = t + block - lags;
            x[(s, col)] - x[(s - 1, col)]
        }
    });
    let lhs = Mat::from_fn(nobs, 1, |i, _| y[i + lags + 1]);
    (lhs, rhs)
}

fn full_names(kx: usize, trend: CointTrend, lags: usize, leads: usize) -> Vec<String> {
    let mut names = param_names(kx, trend);
    for b in 0..(lags + leads + 1) {
        for j in 1..=kx {
            let name = if b < lags {
                format!("D.x{j}.LAG{}", lags - b)
            } else if b == lags {
                format!("D.x{j}")
            } else {
                format!("D.x{j}.LEAD{}", b - lags)
            };
            names.push(name);
        }
    }
    names
}

/// Stock-Watson dynamic OLS of `y` (length `T`) on the `T x k_x`
/// regressors `x` — see the module docs for the regression, the lead/lag
/// search and the covariance estimators.
///
/// Reproduces `arch.unitroot.cointegration.DynamicOLS(y, x, trend, lags,
/// leads, common, max_lag, max_lead, method).fit(cov_type, kernel,
/// bandwidth, force_int, df_adjust)` to 1e-10 on the goldens.
///
/// # Errors
///
/// [`CointError::Dimension`] for an empty `x` or mismatched lengths;
/// [`CointError::NonFinite`] / [`CointError::NonFiniteSeries`] on NaN or
/// infinity; [`CointError::InvalidSpec`] when `common` is set with
/// unequal fixed `lags`/`leads` (or unequal `max_lag`/`max_lead`), when
/// the largest candidate leaves no residual degrees of freedom, for a bad
/// bandwidth or the truncated kernel; [`CointError::Singular`] /
/// [`CointError::NotPositiveDefinite`] for a collinear design;
/// [`CointError::Hac`] from the Andrews bandwidth rule.
pub fn dols(y: &[f64], x: MatRef<'_, f64>, opts: &DolsOptions) -> Result<DolsResult, CointError> {
    if opts.kernel == Kernel::Truncated {
        return Err(CointError::InvalidSpec {
            what: "kernel = \"truncated\" is not admissible for a cointegrating regression: \
                   its lag window is not positive semi-definite, so the long-run covariance \
                   it produces can fail to be a covariance, and it has no plug-in bandwidth \
                   rule; pass \"bartlett\", \"parzen\" or \"quadratic-spectral\""
                .to_string(),
        });
    }
    let (n, kx) = check_yx(y, x)?;
    let trend = opts.trend;
    let n_det = trend.n_det();

    if opts.common {
        if let (Some(p), Some(q)) = (opts.lags, opts.leads) {
            if p != q {
                return Err(CointError::InvalidSpec {
                    what: format!(
                        "common = true requires lags == leads but lags = {p} and leads = {q} \
                         were fixed; drop common, or fix them to the same value"
                    ),
                });
            }
        }
        if opts.max_lag != opts.max_lead {
            return Err(CointError::InvalidSpec {
                what: format!(
                    "common = true requires max_lag == max_lead but max_lag = {:?} and \
                     max_lead = {:?} were given; pass the same cap for both (or neither)",
                    opts.max_lag, opts.max_lead
                ),
            });
        }
    }

    let nf = n as f64;
    let default_cap = (12.0 * (nf / 100.0).powf(0.25)).ceil() as usize;
    let (min_lag, max_lag) = match opts.lags {
        Some(p) => (p, p),
        None => (0, opts.max_lag.unwrap_or(default_cap)),
    };
    let (min_lead, max_lead) = match opts.leads {
        Some(q) => (q, q),
        None => (0, opts.max_lead.unwrap_or(default_cap)),
    };
    let both_fixed = opts.lags.is_some() && opts.leads.is_some();

    // The largest candidate must leave residual degrees of freedom.
    let n_params_max = kx + n_det + kx * (max_lag + max_lead + 1);
    let rows_max = n.checked_sub(1 + max_lag + max_lead).unwrap_or(0);
    if rows_max <= n_params_max {
        let describe = |name: &str, fixed: Option<usize>, cap: Option<usize>, used: usize| match (
            fixed, cap,
        ) {
            (Some(v), _) => format!("{name} = {v}"),
            (None, Some(c)) => format!("max_{name} = {c}"),
            (None, None) => format!("max_{name} = {used} (the default ceil(12 (T/100)^(1/4)))"),
        };
        return Err(CointError::InvalidSpec {
            what: format!(
                "the dynamic OLS design is too large for T = {n} observations: with {} and \
                 {} the largest regression keeps T - 1 - {max_lag} - {max_lead} = {rows_max} \
                 row(s) for k_x + n_det + k_x (lags + leads + 1) = {kx} + {n_det} + {kx} x \
                 {} = {n_params_max} regressors, which leaves no residual degrees of \
                 freedom; fix or cap the lags/leads lower, drop regressors, or supply a \
                 longer sample",
                describe("lags", opts.lags, opts.max_lag, max_lag),
                describe("leads", opts.leads, opts.max_lead, max_lead),
                max_lag + max_lead + 1
            ),
        });
    }

    // Lead/lag search on the common sample of the largest candidate.
    let (lags, leads, ic_value) = if both_fixed {
        (max_lag, max_lead, f64::NAN)
    } else {
        let (lhs, rhs) = design(y, x, trend, max_lag, max_lead);
        let nobs = lhs.nrows();
        let nobs_f = nobs as f64;
        let always = kx + n_det;
        let penalty = opts.ic.penalty(nobs_f);
        let mut best = (0usize, 0usize, f64::INFINITY);
        for lag in min_lag..=max_lag {
            for lead in min_lead..=max_lead {
                if opts.common && lag != lead {
                    continue;
                }
                let first_block = max_lag - lag;
                let n_blocks = lag + lead + 1;
                let ncols = always + n_blocks * kx;
                let sub = Mat::from_fn(nobs, ncols, |i, j| {
                    if j < always {
                        rhs[(i, j)]
                    } else {
                        rhs[(i, always + first_block * kx + (j - always))]
                    }
                });
                let b = lstsq(
                    sub.as_ref(),
                    lhs.as_ref(),
                    "a candidate dynamic OLS design is collinear (a regressor is an exact \
                     linear combination of the others, or its differences are)",
                )?;
                let fitted = &sub * &b;
                let rss: f64 = (0..nobs)
                    .map(|i| {
                        let e = lhs[(i, 0)] - fitted[(i, 0)];
                        e * e
                    })
                    .sum();
                let ic = (rss / nobs_f).ln() + ncols as f64 * penalty / nobs_f;
                if ic < best.2 {
                    best = (lag, lead, ic);
                }
            }
        }
        best
    };

    // Final regression on its own sample.
    let (lhs, rhs) = design(y, x, trend, lags, leads);
    let nobs = lhs.nrows();
    let nobs_f = nobs as f64;
    let n_params = rhs.ncols();
    let b = lstsq(
        rhs.as_ref(),
        lhs.as_ref(),
        "the dynamic OLS design is collinear (a regressor is an exact linear combination \
         of the others, or its differences are); drop the redundant column",
    )?;
    let full_params: Vec<f64> = (0..n_params).map(|j| b[(j, 0)]).collect();
    let fitted = &rhs * &b;
    let resid: Vec<f64> = (0..nobs).map(|i| lhs[(i, 0)] - fitted[(i, 0)]).collect();

    // Covariance.
    let zpz = rhs.transpose() * &rhs;
    let sigma_zz = Mat::from_fn(n_params, n_params, |i, j| zpz[(i, j)] / nobs_f);
    let sigma_zz_inv = inv_spd_scaled(
        sigma_zz.as_ref(),
        "Z'Z of the dynamic OLS regression (collinear design)",
    )?;
    let scale = if opts.df_adjust {
        nobs_f / (nobs - n_params) as f64
    } else {
        1.0
    };
    let eps = Mat::from_fn(nobs, 1, |i, _| resid[i]);
    let (bandwidth, bandwidth_rule, full_cov, long_run_variance) = match opts.cov_type {
        DolsCovType::Unadjusted => {
            let (bw, rule) = resolve_bandwidth(
                eps.as_ref(),
                opts.kernel,
                opts.bandwidth,
                opts.bandwidth_rule,
                opts.force_int,
            )?;
            let lr = long_run_covariance(eps.as_ref(), opts.kernel, bw)?;
            let sigma2 = lr.omega[0][0];
            let cov: Vec<Vec<f64>> = (0..n_params)
                .map(|i| {
                    (0..n_params)
                        .map(|j| scale * sigma2 * sigma_zz_inv[(i, j)] / nobs_f)
                        .collect()
                })
                .collect();
            (bw, rule, cov, scale * sigma2)
        }
        DolsCovType::Robust => {
            let scores = Mat::from_fn(nobs, n_params, |i, j| rhs[(i, j)] * resid[i]);
            let (bw, rule) = resolve_bandwidth(
                scores.as_ref(),
                opts.kernel,
                opts.bandwidth,
                opts.bandwidth_rule,
                opts.force_int,
            )?;
            let lr = long_run_covariance(scores.as_ref(), opts.kernel, bw)?;
            let s = lr.omega_mat();
            let sand = &sigma_zz_inv * &s * &sigma_zz_inv;
            let cov: Vec<Vec<f64>> = (0..n_params)
                .map(|i| {
                    (0..n_params)
                        .map(|j| scale * sand[(i, j)] / nobs_f)
                        .collect()
                })
                .collect();
            let lr_eps = long_run_covariance(eps.as_ref(), opts.kernel, bw)?;
            (bw, rule, cov, scale * lr_eps.omega[0][0])
        }
    };

    let ci = kx + n_det;
    let params: Vec<f64> = full_params[..ci].to_vec();
    let cov: Vec<Vec<f64>> = (0..ci).map(|i| full_cov[i][..ci].to_vec()).collect();
    let (se, tvalues, pvalues) = inference(&params, &cov);
    let (full_se, _, _) = inference(&full_params, &full_cov);

    // statsmodels R^2 of the augmented regression.
    let has_constant = trend.has_constant();
    let ssr: f64 = resid.iter().map(|e| e * e).sum();
    let yv: Vec<f64> = (0..nobs).map(|i| lhs[(i, 0)]).collect();
    let tss = if has_constant {
        let mean = yv.iter().sum::<f64>() / nobs_f;
        yv.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>()
    } else {
        yv.iter().map(|v| v * v).sum::<f64>()
    };
    let rsquared = 1.0 - ssr / tss;
    let k_const = if has_constant { 1.0 } else { 0.0 };
    let rsquared_adj = 1.0 - (nobs_f - k_const) / (nobs - n_params) as f64 * (1.0 - rsquared);

    // Static OLS comparison on the full sample.
    let det_full = det_columns(trend, n);
    let z_full = hcat(x, det_full.as_ref());
    let ols = static_ols(y, z_full.as_ref())?;

    Ok(DolsResult {
        params,
        se,
        tvalues,
        pvalues,
        cov,
        param_names: param_names(kx, trend),
        full_params,
        full_se,
        full_cov,
        full_param_names: full_names(kx, trend, lags, leads),
        resid,
        nobs,
        n_total: n,
        n_x: kx,
        n_det,
        n_params,
        trend,
        lags,
        leads,
        selected: !both_fixed,
        ic: opts.ic,
        ic_value,
        max_lag,
        max_lead,
        common: opts.common,
        cov_type: opts.cov_type,
        kernel: opts.kernel,
        bandwidth,
        bandwidth_rule,
        force_int: opts.force_int,
        df_adjust: opts.df_adjust,
        long_run_variance,
        rsquared,
        rsquared_adj,
        ols_params: ols.params,
        ols_se: ols.se,
    })
}
