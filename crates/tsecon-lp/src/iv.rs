//! Just-identified LP-IV: [`lp_iv`].

use tsecon_hac::{ols, Kernel, SeType};
use tsecon_linalg::faer::linalg::solvers::DenseSolveCore;
use tsecon_linalg::faer::{Mat, Side};

use crate::design::{
    check_finite, const_column, contemporaneous_column, horizon_sample, impulse_column, lag_column,
    outcome_column,
};
use crate::error::LpError;
use crate::spec::{LpIvResult, LpMultiplierResult, LpSpec, SeKind, SeSpec};

/// Estimate an instrumented (LP-IV) impulse-response function.
///
/// For each horizon `h in 0..=spec.horizons` this fits the just-identified
/// two-stage least squares projection
///
/// ```text
///   y_{t+h} = beta_h * impulse_t + c + sum_{l=1}^{p} phi_l y_{t-l} + u_{t,h}
/// ```
///
/// with the endogenous `impulse_t` instrumented by `instrument_t` and the
/// constant and lag controls treated as included (exogenous) instruments.
/// The covariance is the kernel-HAC estimator of the linearmodels `IV2SLS`
/// `cov_type="kernel"` (Bartlett kernel, `bandwidth = h + p` by default,
/// `debiased=False`), reproduced to golden precision.
///
/// Each horizon also reports a first-stage effective-F diagnostic: the
/// HAC-robust first-stage F for the excluded instrument (for a single
/// instrument this is the Montiel Olea & Pflueger 2013 effective F). Values
/// below the rule-of-thumb of 10 flag a weak instrument.
///
/// The impulse-lag augmentation of [`lp`](crate::lp) is a level-LP device;
/// LP-IV keeps the linearmodels kernel-HAC covariance regardless of
/// `spec.se` so its numbers stay comparable to the reference implementation.
///
/// # Errors
///
/// [`LpError::LengthMismatch`] if the inputs differ in length,
/// [`LpError::NonFinite`] on NaN/inf input, [`LpError::SeriesTooShort`] /
/// [`LpError::HorizonTooLong`] when a horizon has no usable sample, and
/// [`LpError::Hac`] wrapping a singular first stage or projection.
pub fn lp_iv(
    y: &[f64],
    impulse: &[f64],
    instrument: &[f64],
    spec: LpSpec,
) -> Result<LpIvResult, LpError> {
    let n = y.len();
    if impulse.len() != n {
        return Err(LpError::LengthMismatch {
            what: "impulse vs outcome (y)",
            expected: n,
            got: impulse.len(),
        });
    }
    if instrument.len() != n {
        return Err(LpError::LengthMismatch {
            what: "instrument vs outcome (y)",
            expected: n,
            got: instrument.len(),
        });
    }
    check_finite(y, "outcome (y)")?;
    check_finite(impulse, "impulse")?;
    check_finite(instrument, "instrument")?;

    let p = spec.n_lag_controls;
    if n <= p {
        return Err(LpError::SeriesTooShort {
            n,
            n_lag_controls: p,
        });
    }

    let mut horizons = Vec::with_capacity(spec.horizons + 1);
    let mut irf = Vec::with_capacity(spec.horizons + 1);
    let mut se = Vec::with_capacity(spec.horizons + 1);
    let mut first_stage_f = Vec::with_capacity(spec.horizons + 1);
    let mut nobs_per_h = Vec::with_capacity(spec.horizons + 1);

    for h in 0..=spec.horizons {
        let (start, nobs) = horizon_sample(n, h, p, 0);
        // Regressors X = [const, y lags..., endog]; instruments
        // Z = [const, y lags..., instrument]. k = p + 2, just identified.
        let k = p + 2;
        if nobs <= k {
            return Err(LpError::HorizonTooLong {
                horizon: h,
                nobs,
                nparams: k,
            });
        }

        let response = outcome_column(y, h, start, nobs, spec.cumulation.accumulates_outcome());
        let exog = exog_columns(y, start, nobs, p);
        let endog = impulse_column(
            impulse,
            h,
            start,
            nobs,
            spec.cumulation.accumulates_impulse(),
        );
        // The instrument stays contemporaneous under every cumulation mode:
        // accumulation belongs to the endogenous variables, not to the
        // identifying variation.
        let instr = contemporaneous_column(instrument, start, nobs);

        let bw = match spec.se {
            SeSpec::Hac {
                maxlags: Some(ml), ..
            } => ml,
            _ => h + p,
        };

        let (beta_endog, se_endog) = iv_kernel(&response, &exog, &endog, &instr, bw)?;
        let f = first_stage_effective_f(&endog, &exog, &instr, bw)?;

        horizons.push(h);
        irf.push(beta_endog);
        se.push(se_endog);
        first_stage_f.push(f);
        nobs_per_h.push(nobs);
    }

    Ok(LpIvResult {
        horizons,
        irf,
        se,
        first_stage_f,
        nobs_per_h,
        se_kind: SeKind::IvKernelHac,
    })
}

/// Exogenous / included-instrument columns `[const, y_{t-1..t-p}]`.
fn exog_columns(y: &[f64], start: usize, nobs: usize, p: usize) -> Vec<Vec<f64>> {
    let mut cols = Vec::with_capacity(1 + p);
    cols.push(const_column(nobs));
    for lag in 1..=p {
        cols.push(lag_column(y, lag, start, nobs));
    }
    cols
}

/// Column-major dense matrix (`nobs x k`) from a slice of columns.
fn mat_from_cols(cols: &[Vec<f64>]) -> Mat<f64> {
    let nobs = cols[0].len();
    let k = cols.len();
    Mat::from_fn(nobs, k, |i, j| cols[j][i])
}

/// The core just-identified 2SLS point estimate and linearmodels kernel-HAC
/// standard error for the (last) endogenous coefficient.
///
/// Reproduces `linearmodels.iv.covariance.KernelCovariance` with
/// `debiased=False`: `cov = M^{-1} S M^{-1}` where `M = X'Z (Z'Z)^{-1} Z'X`,
/// `S = sum_j K(j, bw) (Gamma_j + Gamma_j')` are the (un-normalised) kernel
/// autocovariances of the projected scores `xhat_t * eps_t`, and
/// `xhat = Z (Z'Z)^{-1} Z'X`, `eps = Y - X beta`.
fn iv_kernel(
    response: &[f64],
    exog: &[Vec<f64>],
    endog: &[f64],
    instr: &[f64],
    bw: usize,
) -> Result<(f64, f64), LpError> {
    let nobs = response.len();
    let k = exog.len() + 1;

    // X = [exog..., endog]; Z = [exog..., instrument].
    let mut x_cols: Vec<Vec<f64>> = exog.to_vec();
    x_cols.push(endog.to_vec());
    let mut z_cols: Vec<Vec<f64>> = exog.to_vec();
    z_cols.push(instr.to_vec());

    let xmat = mat_from_cols(&x_cols);
    let zmat = mat_from_cols(&z_cols);
    let ymat = Mat::from_fn(nobs, 1, |i, _| response[i]);

    let ztz = zmat.transpose() * &zmat;
    let ztz_inv = ztz
        .llt(Side::Lower)
        .map_err(|_| singular("LP-IV Z'Z (instruments collinear)"))?
        .inverse();
    let ztx = zmat.transpose() * &xmat;
    let zty = zmat.transpose() * &ymat;

    let a = &ztz_inv * &ztx; // (Z'Z)^{-1} Z'X, k x k
    let ay = &ztz_inv * &zty; // (Z'Z)^{-1} Z'Y, k x 1
    let xtz = ztx.transpose(); // X'Z
    let m = xtz * &a; // X'Z (Z'Z)^{-1} Z'X (symmetric PD), k x k
    let xr = xtz * &ay; // X'Z (Z'Z)^{-1} Z'Y, k x 1

    let m_inv = m
        .llt(Side::Lower)
        .map_err(|_| singular("LP-IV projected design X'PzX (weak/collinear)"))?
        .inverse();
    let beta = &m_inv * &xr; // k x 1

    // Residuals and projected scores.
    let xhat = &zmat * &a; // nobs x k
    let mut scores = vec![0.0_f64; nobs * k];
    for t in 0..nobs {
        let mut fit = 0.0;
        for j in 0..k {
            fit += xmat[(t, j)] * beta[(j, 0)];
        }
        let eps_t = response[t] - fit;
        for j in 0..k {
            scores[t * k + j] = xhat[(t, j)] * eps_t;
        }
    }

    // S = Gamma_0 + sum_{lag>=1} w_lag (Gamma_lag + Gamma_lag'),
    // un-normalised (linearmodels' 1/n cancels the V = M/n scaling).
    let kernel = Kernel::Bartlett;
    let bwf = bw as f64;
    let mut s = vec![0.0_f64; k * k];
    for lag in 0..nobs {
        let w = kernel.weight(lag, bwf);
        if lag > 0 && w == 0.0 {
            break; // Bartlett truncates beyond bw.
        }
        for t in lag..nobs {
            let row_t = &scores[t * k..(t + 1) * k];
            let row_l = &scores[(t - lag) * k..(t - lag + 1) * k];
            for i in 0..k {
                for j in 0..k {
                    let g = row_t[i] * row_l[j];
                    if lag == 0 {
                        s[i * k + j] += g;
                    } else {
                        s[i * k + j] += w * g;
                        s[j * k + i] += w * g;
                    }
                }
            }
        }
    }
    let smeat = Mat::from_fn(k, k, |i, j| s[i * k + j]);

    // cov = M^{-1} S M^{-1}; symmetrise for numerical hygiene.
    let cov = &m_inv * &smeat * &m_inv;
    let endog_idx = k - 1;
    let var: f64 = cov[(endog_idx, endog_idx)];
    if var < 0.0 {
        return Err(LpError::Hac(tsecon_hac::HacError::NumericalBreakdown {
            what: "LP-IV kernel covariance diagonal",
        }));
    }
    Ok((beta[(endog_idx, 0)], var.sqrt()))
}

/// HAC-robust first-stage effective F for the single excluded instrument.
///
/// Regresses the endogenous impulse on `[const, y lags, instrument]` with the
/// same Bartlett HAC (`bw`, `use_correction=True`) and returns the squared
/// robust t-statistic on the instrument — the Montiel Olea & Pflueger (2013)
/// effective F in the single-instrument, just-identified case.
fn first_stage_effective_f(
    endog: &[f64],
    exog: &[Vec<f64>],
    instr: &[f64],
    bw: usize,
) -> Result<f64, LpError> {
    let mut cols: Vec<Vec<f64>> = exog.to_vec();
    cols.push(instr.to_vec());
    let fit = ols(endog, &cols)?;
    let inf = fit.inference(SeType::Hac {
        kernel: Kernel::Bartlett,
        bandwidth: bw as f64,
        use_correction: true,
    })?;
    let idx = cols.len() - 1;
    let t = fit.params[idx] / inf.bse[idx];
    Ok(t * t)
}

fn singular(what: &'static str) -> LpError {
    LpError::Hac(tsecon_hac::HacError::SingularDesign { what })
}

/// Estimate the Ramey-Zubairy (2018) **integral multiplier** by one-step
/// LP-IV.
///
/// For each horizon `h in 0..=spec.horizons` this fits the just-identified
/// two-stage least squares projection
///
/// ```text
///   sum_{j=0}^{h} y_{t+j} = m_h * sum_{j=0}^{h} x_{t+j}
///                         + c + sum_{l=1}^{p} (phi_l y_{t-l} + psi_l x_{t-l})
///                         + u_{t,h}
/// ```
///
/// with the cumulated impulse instrumented by the **contemporaneous**
/// instrument `z_t`. Both sides of the projection are accumulated over the
/// same window, so `m_h` is the extra cumulated outcome per extra cumulated
/// impulse through horizon `h` — a multiplier, in the units of the two
/// series.
///
/// # Why this is not `lp_iv(..., Cumulation::Outcome)`
///
/// [`Cumulation::Outcome`](crate::Cumulation::Outcome) accumulates only the left-hand side, so its
/// coefficient is cumulated `y` per unit of *contemporaneous* `x`. That
/// denominator does not grow with the horizon while the numerator does, so
/// the reported number grows roughly linearly in `h` and is not a multiplier
/// at all. Requesting a multiplier through this function makes the correct
/// object the one you get.
///
/// # Standard errors
///
/// The multiplier is a **single 2SLS coefficient**, not a ratio of two
/// separately estimated responses, so [`LpMultiplierResult::se`] is the
/// kernel-HAC standard error of the reported parameter itself. No
/// delta-method approximation is involved and no leg's standard error is
/// being relabelled. (`cumulative_outcome` / `cumulative_impulse` are
/// reported for transparency and carry no standard errors; by the
/// just-identified IV algebra their ratio equals `multiplier` to numerical
/// precision.)
///
/// `spec.cumulation` and `spec.se` are ignored: the cumulation is fixed at
/// [`Cumulation::Both`](crate::Cumulation::Both) by definition, and the covariance is the
/// linearmodels-convention kernel HAC, except that `SeSpec::Hac { maxlags:
/// Some(m) }` overrides the default `bandwidth = h + p`.
///
/// # Errors
///
/// [`LpError::LengthMismatch`] if the inputs differ in length,
/// [`LpError::NonFinite`] on NaN/inf input, [`LpError::SeriesTooShort`] /
/// [`LpError::HorizonTooLong`] when a horizon has no usable sample, and
/// [`LpError::Hac`] wrapping a singular first stage or projection.
pub fn lp_multiplier(
    y: &[f64],
    impulse: &[f64],
    instrument: &[f64],
    spec: LpSpec,
) -> Result<LpMultiplierResult, LpError> {
    let n = y.len();
    if impulse.len() != n {
        return Err(LpError::LengthMismatch {
            what: "impulse vs outcome (y)",
            expected: n,
            got: impulse.len(),
        });
    }
    if instrument.len() != n {
        return Err(LpError::LengthMismatch {
            what: "instrument vs outcome (y)",
            expected: n,
            got: instrument.len(),
        });
    }
    check_finite(y, "outcome (y)")?;
    check_finite(impulse, "impulse")?;
    check_finite(instrument, "instrument")?;

    let p = spec.n_lag_controls;
    if n <= p {
        return Err(LpError::SeriesTooShort {
            n,
            n_lag_controls: p,
        });
    }

    let cap = spec.horizons + 1;
    let mut horizons = Vec::with_capacity(cap);
    let mut multiplier = Vec::with_capacity(cap);
    let mut se = Vec::with_capacity(cap);
    let mut first_stage_f = Vec::with_capacity(cap);
    let mut cumulative_outcome = Vec::with_capacity(cap);
    let mut cumulative_impulse = Vec::with_capacity(cap);
    let mut nobs_per_h = Vec::with_capacity(cap);

    for h in 0..=spec.horizons {
        let (start, nobs) = horizon_sample(n, h, p, p);
        // X = [const, y lags, x lags, cum endog]; Z swaps the last column for
        // the instrument. k = 2 + 2p, just identified.
        let k = 2 + 2 * p;
        if nobs <= k {
            return Err(LpError::HorizonTooLong {
                horizon: h,
                nobs,
                nparams: k,
            });
        }

        let cum_y = outcome_column(y, h, start, nobs, true);
        let cum_x = impulse_column(impulse, h, start, nobs, true);
        let exog = multiplier_exog_columns(y, impulse, start, nobs, p);
        let instr = contemporaneous_column(instrument, start, nobs);

        let bw = match spec.se {
            SeSpec::Hac {
                maxlags: Some(ml), ..
            } => ml,
            _ => h + p,
        };

        let (m_h, se_h) = iv_kernel(&cum_y, &exog, &cum_x, &instr, bw)?;
        let f = first_stage_effective_f(&cum_x, &exog, &instr, bw)?;
        // Reduced forms, for transparency: the two legs whose ratio is m_h.
        let rf_y = reduced_form_coef(&cum_y, &exog, &instr)?;
        let rf_x = reduced_form_coef(&cum_x, &exog, &instr)?;

        horizons.push(h);
        multiplier.push(m_h);
        se.push(se_h);
        first_stage_f.push(f);
        cumulative_outcome.push(rf_y);
        cumulative_impulse.push(rf_x);
        nobs_per_h.push(nobs);
    }

    Ok(LpMultiplierResult {
        horizons,
        multiplier,
        se,
        first_stage_f,
        cumulative_outcome,
        cumulative_impulse,
        nobs_per_h,
        se_kind: SeKind::IvKernelHac,
    })
}

/// Included-instrument columns for the multiplier design:
/// `[const, y_{t-1..t-p}, x_{t-1..t-p}]`.
///
/// Unlike [`lp_iv`], the multiplier design controls for lags of the impulse
/// as well as the outcome: the denominator is now an endogenous *quantity*
/// whose own dynamics must be soaked up for the ratio to be interpretable
/// (Ramey & Zubairy 2018 condition on lags of both).
fn multiplier_exog_columns(
    y: &[f64],
    impulse: &[f64],
    start: usize,
    nobs: usize,
    p: usize,
) -> Vec<Vec<f64>> {
    let mut cols = Vec::with_capacity(1 + 2 * p);
    cols.push(const_column(nobs));
    for lag in 1..=p {
        cols.push(lag_column(y, lag, start, nobs));
    }
    for lag in 1..=p {
        cols.push(lag_column(impulse, lag, start, nobs));
    }
    cols
}

/// OLS coefficient on the instrument in the reduced form of `target`.
fn reduced_form_coef(target: &[f64], exog: &[Vec<f64>], instr: &[f64]) -> Result<f64, LpError> {
    let mut cols: Vec<Vec<f64>> = exog.to_vec();
    cols.push(instr.to_vec());
    let fit = ols(target, &cols)?;
    Ok(fit.params[cols.len() - 1])
}
