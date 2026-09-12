//! Regression with random-walk (time-varying-parameter) coefficients by
//! exact-diffuse maximum likelihood: [`tvp_regression`].
//!
//! # Model
//!
//! ```text
//! y_t         = x_t' beta_t + eps_t,       eps_t ~ N(0, sigma2_eps)
//! beta_{t+1}  = beta_t + eta_t,            eta_t ~ N(0, diag(sigma2_beta))
//! beta_1      ~ diffuse
//! ```
//!
//! — the state-space form with the per-period design row `Z_t = x_t'`
//! ([`crate::SystemMatrix::Varying`]), identity transition and
//! selection, and one disturbance variance per coefficient (Harvey 1989,
//! §8.3; Durbin & Koopman 2012, §3.6). With every `sigma2_beta` equal to
//! zero the filter *is* recursive least squares: the filtered
//! coefficients are the expanding-window OLS estimates and the model
//! reduces to statsmodels' `RecursiveLS`, which the golden tests pin.
//!
//! # Estimation and the pile-up check
//!
//! `sigma2_eps` and the `k` state variances are estimated by the shared
//! MLE layer ([`crate::mle`]): BFGS + Nelder-Mead in the square-root
//! working space from a deterministic ladder of starts, on `y` scaled by
//! its standard deviation and each regressor by its largest absolute
//! value, mapped back exactly. Random-walk-coefficient likelihoods are
//! the textbook case of the *pile-up problem* (Shephard & Harvey 1990;
//! Stock & Watson 1998): with a small true variance the MLE lands on
//! exactly zero with substantial probability. Each state variance whose
//! estimate cannot be told from zero — zeroing it lowers the
//! log-likelihood by less than `1e-4` — is therefore flagged in
//! `pile_up` (and `at_boundary`) and reported with a NaN standard error,
//! so a coefficient the data cannot show moving is not presented as
//! estimated to be moving by a tiny amount.
//!
//! `aic` / `bic` follow statsmodels' `df_model = k_params + k_diffuse`
//! convention (`k_diffuse = k`). The observation `y` may contain NaN for
//! missing periods; the regressors may not.

use tsecon_linalg::faer::Mat;

use crate::dense::dot;
use crate::error::SsmError;
use crate::mle::{maximize, ols, variance, ParamKind, ParamSpec};
use crate::model::{Initialization, LinearGaussianSSM};
use crate::smoother::smooth_univariate;

/// Options for [`tvp_regression`].
#[derive(Debug, Clone, PartialEq)]
pub struct TvpOptions {
    /// Prepend a column of ones (a random-walk intercept).
    pub constant: bool,
    /// Evaluate at these parameters `[sigma2_eps, sigma2_beta_1, ...,
    /// sigma2_beta_k]` instead of estimating (`sigma2_eps > 0`, the rest
    /// `>= 0`; all zero gives the recursive-least-squares filter).
    pub fixed_params: Option<Vec<f64>>,
    /// Number of deterministic starts for the search (at least 1).
    pub n_starts: usize,
}

impl Default for TvpOptions {
    fn default() -> Self {
        Self {
            constant: true,
            fixed_params: None,
            n_starts: 3,
        }
    }
}

/// The fitted TVP regression.
#[derive(Debug, Clone, PartialEq)]
pub struct TvpFit {
    /// Coefficient names (`const`, `x1`, ...).
    pub coef_names: Vec<String>,
    /// Number of coefficients (constant included).
    pub k: usize,
    /// Parameter names: `sigma2.irregular`, then `sigma2.<coef>`.
    pub param_names: Vec<String>,
    /// Estimates (or the fixed parameters).
    pub params: Vec<f64>,
    /// Observed-information standard errors (NaN at a boundary, when the
    /// Hessian is singular, or under `fixed_params`).
    pub se: Vec<f64>,
    /// Boundary flag per parameter.
    pub at_boundary: Vec<bool>,
    /// Observation variance.
    pub sigma2_eps: f64,
    /// State (coefficient-innovation) variances.
    pub sigma2_beta: Vec<f64>,
    /// Pile-up flag per coefficient (`at_boundary[1..]`).
    pub pile_up: Vec<bool>,
    /// Exact-diffuse log-likelihood at `params`.
    pub loglik: f64,
    /// `2 (k_params + k) - 2 loglik`.
    pub aic: f64,
    /// `-2 loglik + (k_params + k) ln(nobs)`.
    pub bic: f64,
    /// Number of time periods (missing included).
    pub nobs: usize,
    /// Number of observed (non-NaN) periods.
    pub nobs_observed: usize,
    /// Length of the diffuse period.
    pub nobs_diffuse: usize,
    /// Number of parameters (`k + 1`).
    pub k_params: usize,
    /// False under `fixed_params`.
    pub estimated: bool,
    /// The optimizer's convergence certificate (true under `fixed_params`).
    pub converged: bool,
    /// Optimizer iterations (0 under `fixed_params`).
    pub n_iter: usize,
    /// Log-likelihood evaluations during the search (0 under
    /// `fixed_params`).
    pub n_fevals: usize,
    /// Filtered coefficients `E[beta_t | y_1..y_t]`, `nobs x k`.
    pub beta_filtered: Vec<Vec<f64>>,
    /// Their variances (finite part inside the diffuse period).
    pub beta_filtered_var: Vec<Vec<f64>>,
    /// Smoothed coefficients `E[beta_t | y_1..y_n]`, `nobs x k`.
    pub beta_smoothed: Vec<Vec<f64>>,
    /// Their variances (exact through the diffuse period).
    pub beta_smoothed_var: Vec<Vec<f64>>,
    /// One-step-ahead predictions `x_t' E[beta_t | y_1..y_{t-1}]`.
    pub fitted: Vec<f64>,
    /// One-step prediction errors (NaN at missing periods).
    pub resid: Vec<f64>,
    /// `v_t / sqrt(F_t)` after the diffuse period; NaN inside it and at
    /// missing periods.
    pub std_resid: Vec<f64>,
}

fn invalid(message: String) -> SsmError {
    SsmError::InvalidSpec { message }
}

/// The state-space system for design columns `cols` at
/// `[sigma2_eps, sigma2_beta...]`.
fn build_ssm(cols: &[Vec<f64>], params: &[f64]) -> Result<LinearGaussianSSM, SsmError> {
    let k = cols.len();
    let n = cols[0].len();
    let zs: Vec<Mat<f64>> = (0..n)
        .map(|t| Mat::from_fn(1, k, |_, j| cols[j][t]))
        .collect();
    let eye = Mat::from_fn(k, k, |i, j| if i == j { 1.0 } else { 0.0 });
    let q = Mat::from_fn(k, k, |i, j| if i == j { params[1 + i] } else { 0.0 });
    LinearGaussianSSM::builder(1, k, k)
        .z_varying(zs)
        .h(Mat::from_fn(1, 1, |_, _| params[0]))
        .t(eye.clone())
        .r(eye)
        .q(q)
        .initialization(Initialization::Diffuse)
        .build()
}

fn loglik_value(cols: &[Vec<f64>], y: &[f64], params: &[f64]) -> f64 {
    let model = match build_ssm(cols, params) {
        Ok(m) => m,
        Err(_) => return f64::NAN,
    };
    let ym = Mat::from_fn(y.len(), 1, |i, _| y[i]);
    match model.filter(ym.as_ref()) {
        Ok(fo) => fo.loglik,
        Err(_) => f64::NAN,
    }
}

/// Fits (or evaluates at `fixed_params`) the random-walk-coefficient
/// regression of `y` on the columns `x`; see the module docs.
///
/// # Errors
///
/// [`SsmError::InvalidSpec`] for every malformed input, naming the
/// argument; the filter's own errors when the likelihood cannot be
/// evaluated at `fixed_params`.
pub fn tvp_regression(y: &[f64], x: &[Vec<f64>], opts: &TvpOptions) -> Result<TvpFit, SsmError> {
    let n = y.len();
    if n == 0 {
        return Err(invalid(
            "y is empty; pass at least one observation".to_string(),
        ));
    }
    if y.iter().any(|v| v.is_infinite()) {
        return Err(invalid(
            "y contains an infinity; entries must be finite or NaN (missing)".to_string(),
        ));
    }
    if x.is_empty() && !opts.constant {
        return Err(invalid(
            "x has no columns and constant = false: a regression needs at least one \
             regressor"
                .to_string(),
        ));
    }
    for (j, col) in x.iter().enumerate() {
        if col.len() != n {
            return Err(invalid(format!(
                "x column {j} has length {} but y has {n} periods; every regressor must be \
                 aligned with y",
                col.len()
            )));
        }
        if col.iter().any(|v| !v.is_finite()) {
            return Err(invalid(format!(
                "x column {j} contains a NaN or infinity; regressors must be finite (NaN \
                 marks a missing value in y only)"
            )));
        }
        if col.iter().all(|&v| v == 0.0) {
            return Err(invalid(format!(
                "x column {j} is identically zero, so its coefficient is not identified; \
                 drop the column"
            )));
        }
    }
    let mut cols: Vec<Vec<f64>> = Vec::with_capacity(x.len() + 1);
    let mut coef_names = Vec::with_capacity(x.len() + 1);
    if opts.constant {
        cols.push(vec![1.0; n]);
        coef_names.push("const".to_string());
    }
    for (j, col) in x.iter().enumerate() {
        cols.push(col.clone());
        coef_names.push(format!("x{}", j + 1));
    }
    let k = cols.len();
    let nobs_observed = y.iter().filter(|v| v.is_finite()).count();
    if nobs_observed < k + 2 {
        return Err(invalid(format!(
            "y has {nobs_observed} observed (non-NaN) values but the regression has {k} \
             coefficients; at least k + 2 = {} observations are needed",
            k + 2
        )));
    }
    let mut param_names = vec!["sigma2.irregular".to_string()];
    param_names.extend(coef_names.iter().map(|c| format!("sigma2.{c}")));
    let specs: Vec<ParamSpec> = param_names
        .iter()
        .map(|nm| ParamSpec {
            name: nm.clone(),
            kind: ParamKind::Variance,
        })
        .collect();

    if let Some(fixed) = &opts.fixed_params {
        if fixed.len() != k + 1 {
            return Err(invalid(format!(
                "fixed_params has length {} but the regression has k + 1 = {} parameters \
                 [sigma2_eps, sigma2_beta_1, ..., sigma2_beta_k] (constant {})",
                fixed.len(),
                k + 1,
                if opts.constant {
                    "included"
                } else {
                    "excluded"
                }
            )));
        }
        if !(fixed[0] > 0.0) || !fixed[0].is_finite() {
            return Err(invalid(format!(
                "fixed_params[0] = {} (sigma2_eps) must be a positive finite number",
                fixed[0]
            )));
        }
        for (i, &v) in fixed.iter().enumerate().skip(1) {
            if !(v >= 0.0) || !v.is_finite() {
                return Err(invalid(format!(
                    "fixed_params[{i}] = {v} ({}) must be a finite variance >= 0",
                    param_names[i]
                )));
            }
        }
        let at_boundary: Vec<bool> = fixed.iter().map(|&v| v == 0.0).collect();
        let se = vec![f64::NAN; k + 1];
        return evaluate(
            &cols,
            coef_names,
            param_names,
            y,
            fixed,
            false,
            true,
            (0, 0),
            se,
            at_boundary,
        );
    }
    if opts.n_starts == 0 {
        return Err(invalid(
            "n_starts = 0: the search needs at least one starting value".to_string(),
        ));
    }

    let observed: Vec<f64> = y.iter().copied().filter(|v| v.is_finite()).collect();
    let s = variance(&observed).sqrt();
    if !(s > 0.0) || !s.is_finite() {
        return Err(invalid(
            "y is constant (standard deviation 0): the observation variance would be \
             estimated at zero and the likelihood is unbounded"
                .to_string(),
        ));
    }
    let y_s: Vec<f64> = y.iter().map(|v| v / s).collect();
    let col_scale: Vec<f64> = cols
        .iter()
        .map(|col| col.iter().fold(0.0f64, |m, v| m.max(v.abs())))
        .collect();
    let cols_s: Vec<Vec<f64>> = cols
        .iter()
        .zip(&col_scale)
        .map(|(col, &c)| col.iter().map(|v| v / c).collect())
        .collect();
    // Starts: OLS residual variance for the irregular; a ladder for the
    // state variances.
    let rows: Vec<usize> = (0..n).filter(|&t| y_s[t].is_finite()).collect();
    let beta = ols(&cols_s, &y_s, &rows).unwrap_or_else(|| vec![0.0; k]);
    let resid: Vec<f64> = rows
        .iter()
        .map(|&t| y_s[t] - cols_s.iter().zip(&beta).map(|(c, b)| c[t] * b).sum::<f64>())
        .collect();
    let mut v_r = variance(&resid);
    if !(v_r > 0.0) || !v_r.is_finite() {
        v_r = 1.0;
    }
    let ladder = [1.0, 0.01, 100.0, 1e-4, 1e4];
    let starts: Vec<Vec<f64>> = (0..opts.n_starts)
        .map(|i| {
            let mut p = vec![v_r; k + 1];
            for v in p.iter_mut().skip(1) {
                *v = 0.01 * v_r * ladder[i % ladder.len()];
            }
            p
        })
        .collect();
    let outcome = maximize(&specs, &starts, |p| loglik_value(&cols_s, &y_s, p))?;
    let back = |i: usize, v: f64| -> f64 {
        if i == 0 {
            v * s * s
        } else {
            let c = col_scale[i - 1];
            v * (s / c) * (s / c)
        }
    };
    let params: Vec<f64> = outcome
        .params
        .iter()
        .enumerate()
        .map(|(i, &v)| back(i, v))
        .collect();
    let se: Vec<f64> = outcome
        .se
        .iter()
        .enumerate()
        .map(|(i, &v)| back(i, v))
        .collect();
    evaluate(
        &cols,
        coef_names,
        param_names,
        y,
        &params,
        true,
        outcome.converged,
        (outcome.n_iter, outcome.n_fevals),
        se,
        outcome.at_boundary,
    )
}

#[allow(clippy::too_many_arguments)]
fn evaluate(
    cols: &[Vec<f64>],
    coef_names: Vec<String>,
    param_names: Vec<String>,
    y: &[f64],
    params: &[f64],
    estimated: bool,
    converged: bool,
    (n_iter, n_fevals): (usize, usize),
    se: Vec<f64>,
    at_boundary: Vec<bool>,
) -> Result<TvpFit, SsmError> {
    let n = y.len();
    let k = cols.len();
    let model = build_ssm(cols, params)?;
    let ym = Mat::from_fn(n, 1, |i, _| y[i]);
    let so = smooth_univariate(&model, ym.as_ref())?;
    let fo = &so.filter;
    let diag = |mat: &Mat<f64>| -> Vec<f64> { (0..k).map(|i| mat[(i, i)]).collect() };
    let beta_filtered = fo.filtered_state.clone();
    let beta_filtered_var: Vec<Vec<f64>> = fo.filtered_state_cov.iter().map(diag).collect();
    let beta_smoothed = so.smoothed_state.clone();
    let beta_smoothed_var: Vec<Vec<f64>> = so.smoothed_state_cov.iter().map(diag).collect();

    let mut fitted = vec![f64::NAN; n];
    let mut resid = vec![f64::NAN; n];
    let mut std_resid = vec![f64::NAN; n];
    for t in 0..n {
        let xrow: Vec<f64> = cols.iter().map(|c| c[t]).collect();
        let pred = dot(&xrow, &fo.predicted_state[t]);
        fitted[t] = pred;
        if y[t].is_finite() {
            resid[t] = y[t] - pred;
            let step = &fo.steps[t];
            if t >= fo.d_diffuse && step.observed && step.f_star > 0.0 {
                std_resid[t] = step.v / step.f_star.sqrt();
            }
        }
    }
    let k_params = k + 1;
    let df = (k_params + k) as f64;
    let loglik = fo.loglik;
    Ok(TvpFit {
        coef_names,
        k,
        param_names,
        params: params.to_vec(),
        se,
        pile_up: at_boundary[1..].to_vec(),
        at_boundary,
        sigma2_eps: params[0],
        sigma2_beta: params[1..].to_vec(),
        loglik,
        aic: 2.0 * df - 2.0 * loglik,
        bic: -2.0 * loglik + df * (n as f64).ln(),
        nobs: n,
        nobs_observed: y.iter().filter(|v| v.is_finite()).count(),
        nobs_diffuse: fo.d_diffuse,
        k_params,
        estimated,
        converged,
        n_iter,
        n_fevals,
        beta_filtered,
        beta_filtered_var,
        beta_smoothed,
        beta_smoothed_var,
        fitted,
        resid,
        std_resid,
    })
}
