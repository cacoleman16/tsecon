//! Functional local projections: the outcome regressed jointly on all `K`
//! functional principal-component scores at each horizon.

use tsecon_hac::{ols, Kernel, SeType};

use crate::error::FuncShockError;

/// Per-horizon joint score coefficients with their joint HAC covariance;
/// produced by [`flp`].
#[derive(Debug, Clone)]
pub struct FlpFit {
    /// The horizons `0..=H`.
    pub horizons: Vec<usize>,
    /// Number of score regressors `K`.
    pub n_factors: usize,
    /// `betas[h]` (length `K`): the horizon-`h` coefficients on the scores.
    pub betas: Vec<Vec<f64>>,
    /// `covs[h]`: the JOINT `K x K` HAC covariance of `betas[h]`, row-major
    /// (`covs[h][i * K + j]`). Kept whole — the scenario variance
    /// `w' Cov_h w` needs the off-diagonal terms.
    pub covs: Vec<Vec<f64>>,
    /// `se[h]` (length `K`): `sqrt(diag(covs[h]))`.
    pub se: Vec<Vec<f64>>,
    /// Usable observations at each horizon (`T - h - n_lag_controls`).
    pub nobs: Vec<usize>,
}

/// Functional local projection (Inoue & Rossi 2021; Jordà 2005 mechanics):
/// for each horizon `h in 0..=horizons` run the JOINT regression
///
/// ```text
/// y_{t+h} = c_h + sum_{k=1}^{K} beta_{h,k} s_{t,k}
///               + sum_{l=1}^{p} phi_{h,l} y_{t-l} + u_{t,h}
/// ```
///
/// over `t = p .. T-1-h` and keep the score coefficients `beta_h` with their
/// joint `K x K` kernel-HAC covariance (Newey-West Bartlett, lag truncation
/// `hac_maxlags` or the default `h + p`, with the statsmodels
/// `use_correction=True` small-sample scaling — the same conventions as
/// `tsecon-lp`'s HAC path, via the library's single OLS owner
/// [`tsecon_hac::ols`]).
///
/// `scores[t]` is the length-`K` score vector at time `t` (rows of
/// [`crate::Fpca::scores`]); `y` is the outcome, aligned with the scores.
///
/// # Errors
///
/// * [`FuncShockError::EmptyInput`] for empty `y`/`scores` or zero-width
///   score rows;
/// * [`FuncShockError::DimensionMismatch`] if `scores` and `y` differ in
///   length; [`FuncShockError::RaggedRow`] on unequal score rows;
/// * [`FuncShockError::NonFinite`] on NaN/infinite entries;
/// * [`FuncShockError::SeriesTooShort`] / [`FuncShockError::HorizonTooLong`]
///   when a horizon has no identifiable sample;
/// * [`FuncShockError::Hac`] wrapping OLS-engine failures (e.g. collinear
///   scores).
pub fn flp(
    y: &[f64],
    scores: &[Vec<f64>],
    horizons: usize,
    n_lag_controls: usize,
    hac_maxlags: Option<usize>,
) -> Result<FlpFit, FuncShockError> {
    let t = y.len();
    if t == 0 {
        return Err(FuncShockError::EmptyInput {
            what: "y (outcome)",
        });
    }
    if scores.is_empty() {
        return Err(FuncShockError::EmptyInput {
            what: "scores (T x K)",
        });
    }
    if scores.len() != t {
        return Err(FuncShockError::DimensionMismatch {
            what: "scores rows vs y: one score vector per outcome observation",
            expected: t,
            got: scores.len(),
        });
    }
    let k = scores[0].len();
    if k == 0 {
        return Err(FuncShockError::EmptyInput {
            what: "scores (each row must contain at least one score)",
        });
    }
    for (row, s) in scores.iter().enumerate() {
        if s.len() != k {
            return Err(FuncShockError::RaggedRow {
                what: "scores",
                row,
                expected: k,
                got: s.len(),
            });
        }
        if s.iter().any(|v| !v.is_finite()) {
            return Err(FuncShockError::NonFinite { what: "scores" });
        }
    }
    if y.iter().any(|v| !v.is_finite()) {
        return Err(FuncShockError::NonFinite {
            what: "y (outcome)",
        });
    }
    let p = n_lag_controls;
    if t <= p {
        return Err(FuncShockError::SeriesTooShort {
            n: t,
            n_lag_controls: p,
        });
    }

    let nparams = 1 + k + p;
    let mut out = FlpFit {
        horizons: Vec::with_capacity(horizons + 1),
        n_factors: k,
        betas: Vec::with_capacity(horizons + 1),
        covs: Vec::with_capacity(horizons + 1),
        se: Vec::with_capacity(horizons + 1),
        nobs: Vec::with_capacity(horizons + 1),
    };

    for h in 0..=horizons {
        let nobs = t.saturating_sub(h + p);
        if nobs <= nparams {
            return Err(FuncShockError::HorizonTooLong {
                horizon: h,
                nobs,
                nparams,
            });
        }
        // Sample t = p .. T-1-h; design [const, s_1..s_K, y_{t-1}..y_{t-p}].
        let ts = p..(t - h);
        let response: Vec<f64> = ts.clone().map(|tt| y[tt + h]).collect();
        let mut cols: Vec<Vec<f64>> = Vec::with_capacity(nparams);
        cols.push(vec![1.0; nobs]);
        cols.extend((0..k).map(|j| ts.clone().map(|tt| scores[tt][j]).collect::<Vec<f64>>()));
        for lag in 1..=p {
            cols.push(ts.clone().map(|tt| y[tt - lag]).collect());
        }

        let fit = ols(&response, &cols)?;
        let ml = hac_maxlags.unwrap_or(h + p);
        let inf = fit.inference(SeType::Hac {
            kernel: Kernel::Bartlett,
            bandwidth: ml as f64,
            use_correction: true,
        })?;

        // Score coefficients sit at design indices 1..=K.
        let betas: Vec<f64> = fit.params[1..=k].to_vec();
        let mut cov = vec![0.0_f64; k * k];
        for i in 0..k {
            for j in 0..k {
                cov[i * k + j] = inf.cov[(i + 1) * nparams + (j + 1)];
            }
        }
        let se: Vec<f64> = (0..k).map(|i| cov[i * k + i].max(0.0).sqrt()).collect();

        out.horizons.push(h);
        out.betas.push(betas);
        out.covs.push(cov);
        out.se.push(se);
        out.nobs.push(nobs);
    }
    Ok(out)
}
