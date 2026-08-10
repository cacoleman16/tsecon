//! Growth-at-risk (Adrian, Boyarchenko & Giannone 2019 AER): conditional
//! quantiles of the `h`-ahead outcome given current conditions.
//!
//! The canonical use regresses `h`-ahead GDP growth on current financial
//! conditions (NFCI) and current growth, per quantile level:
//!
//! ```text
//! Q_tau( y_{t+h} | x_t ) = x_t' beta_tau,
//! x_t = [1, conditions_t, y_t],   t = 0, ..., n - 1 - h,
//! ```
//!
//! then reads the fitted quantiles at every `t` — the last observation is
//! the *current risk read*. Because separately fitted quantile curves can
//! cross in finite samples, the Chernozhukov-Fernandez-Val-Galichon (2010)
//! rearrangement is applied (optionally): at each evaluation point the
//! fitted values are sorted across tau, which is exactly the monotone
//! rearrangement of the estimated conditional quantile function and never
//! increases estimation error.
//!
//! ## Overlapping windows and the standard errors
//!
//! For `h > 1` consecutive regressions share innovations: `y_{t+h}` and
//! `y_{t+h-l}` have `h - l` of them in common, so even under a perfectly
//! specified conditional quantile the check-loss score
//! `psi_t = x_t (tau - 1{u_t < 0})` is an MA(h-1), not a martingale
//! difference. The plain Powell sandwich assumes the latter and is therefore
//! *optimistic by construction* at every horizon the function exists for.
//! `bse` applies the standard remedy — a Newey-West (Bartlett) correction at
//! `h - 1` lags, the exact MA order the overlap induces — and `bse_powell`
//! keeps the uncorrected number.
//!
//! The correction is a large improvement, not a cure. Monte Carlo on an
//! exact-truth, correctly specified, homoskedastic Gaussian design with a
//! persistent conditioner (measured numbers in the model card) puts nominal
//! 95% coverage of a slope at, e.g., `T = 250, tau = 0.05`: 0.60 (Powell) vs
//! 0.69 (corrected) at `h = 12`, and 0.77 vs 0.81 at `h = 4`. The residual
//! gap is *not* serial correlation: it is the upward finite-sample bias of
//! the kernel density estimate `f_hat(0)` that both sandwiches divide by,
//! which grows with the horizon because the overlap shrinks the effective
//! sample. Quote long-horizon bands with that in mind.

use crate::error::QuantileError;
use crate::qreg::{check_finite, fit_one_hac, validate_taus, QuantileFit};

/// Growth-at-risk results; produced by [`growth_at_risk`].
#[derive(Debug, Clone, PartialEq)]
pub struct GrowthAtRisk {
    /// The quantile levels (strictly increasing, as passed).
    pub taus: Vec<f64>,
    /// The forecast horizon `h >= 1`.
    pub horizon: usize,
    /// Per-tau coefficients on `[const, conditions..., y_t]`,
    /// indexed `params[tau_index][coef]`.
    pub params: Vec<Vec<f64>>,
    /// Per-tau standard errors, same indexing: the Powell sandwich with a
    /// Newey-West (Bartlett) correction at [`Self::hac_lags`] lags for the
    /// serial correlation the *overlapping* `h`-step windows induce. At
    /// `horizon = 1` there is no overlap, `hac_lags` is zero, and these are
    /// exactly [`Self::bse_powell`].
    ///
    /// Even corrected these are optimistic at long horizons — see the
    /// measured coverage table in the model card before quoting a band.
    pub bse: Vec<Vec<f64>>,
    /// Per-tau *uncorrected* Powell-sandwich standard errors — what
    /// statsmodels `QuantReg(...).fit(q=tau)` reports for this design, kept
    /// for replication and for the (rare) case of a serially uncorrelated
    /// conditioner, where the HAC correction only adds noise.
    pub bse_powell: Vec<Vec<f64>>,
    /// The Newey-West lag truncation used for [`Self::bse`]: `horizon - 1`,
    /// the exact MA order the overlap induces in the check-loss score.
    pub hac_lags: usize,
    /// Raw fitted conditional quantiles `x_t' beta_tau` at EVERY
    /// `t = 0..n-1` (not just the estimation sample), indexed
    /// `fitted_raw[tau_index][t]`.
    pub fitted_raw: Vec<Vec<f64>>,
    /// Fitted conditional quantiles after the requested treatment:
    /// rearranged (sorted across tau at each `t`) when `rearrange` was
    /// `true`, identical to `fitted_raw` otherwise.
    pub fitted: Vec<Vec<f64>>,
    /// Whether any quantile crossing occurred in `fitted_raw` (some
    /// `fitted_raw[j+1][t] < fitted_raw[j][t]`) — reported regardless of
    /// whether rearrangement was applied.
    pub crossing: bool,
    /// The current risk read: `fitted[..][n-1]` across taus — the
    /// conditional quantiles of `y_{n-1+h}` given the latest conditions.
    pub current: Vec<f64>,
}

/// Conditional quantiles of the `horizon`-ahead outcome given current
/// conditions, with optional monotone rearrangement across tau.
///
/// `conditions` are columns (each of length `n = y.len()`), e.g. an NFCI
/// series; the design is `[const, conditions..., y_t]` in that order. Fits
/// use observations `t = 0..n-1-horizon`; fitted quantiles are evaluated at
/// every `t`, so `current` (the last evaluation point) is a genuine
/// out-of-sample risk read. Coefficients match statsmodels `QuantReg` per
/// tau plus a numpy `sort` rearrangement (see the fixture generator); so do
/// `bse_powell`, while `bse` adds the `horizon - 1`-lag Newey-West
/// correction the overlapping windows require (module docs).
///
/// # Errors
///
/// [`QuantileError::ZeroHorizon`], [`QuantileError::TausNotIncreasing`],
/// plus the shared shape/finiteness/tau/degrees-of-freedom errors of
/// [`crate::quantile_regression`].
pub fn growth_at_risk(
    y: &[f64],
    conditions: &[Vec<f64>],
    horizon: usize,
    taus: &[f64],
    rearrange: bool,
) -> Result<GrowthAtRisk, QuantileError> {
    if horizon == 0 {
        return Err(QuantileError::ZeroHorizon);
    }
    validate_taus(taus)?;
    for (i, pair) in taus.windows(2).enumerate() {
        if pair[1] <= pair[0] {
            return Err(QuantileError::TausNotIncreasing { index: i + 1 });
        }
    }
    if y.is_empty() {
        return Err(QuantileError::EmptyInput { what: "y" });
    }
    let n = y.len();
    for col in conditions {
        if col.len() != n {
            return Err(QuantileError::DimensionMismatch {
                what: "condition series vs y",
                expected: n,
                got: col.len(),
            });
        }
    }
    check_finite(y, "y")?;
    for col in conditions {
        check_finite(col, "condition series")?;
    }
    let k = 2 + conditions.len();
    if n <= horizon || n - horizon <= k {
        return Err(QuantileError::HorizonExhaustsSample {
            horizon,
            n,
            nobs: n.saturating_sub(horizon),
            k,
        });
    }

    // Estimation design over t = 0..n-1-h: [const, conditions..., y_t].
    let nobs = n - horizon;
    let mut cols: Vec<Vec<f64>> = Vec::with_capacity(k);
    cols.push(vec![1.0; nobs]);
    for col in conditions {
        cols.push(col[..nobs].to_vec());
    }
    cols.push(y[..nobs].to_vec());
    let outcome: Vec<f64> = (0..nobs).map(|t| y[t + horizon]).collect();

    // Overlapping windows: y_{t+h} and y_{t+h-l} share h - l innovations for
    // l < h, so the check-loss score x_t (tau - 1{u_t < 0}) is an MA(h-1)
    // even under a correctly specified quantile. h - 1 is that exact order.
    let hac_lags = horizon - 1;
    let fitted_pairs: Vec<(QuantileFit, Vec<f64>)> = taus
        .iter()
        .map(|&tau| fit_one_hac(&outcome, &cols, tau, hac_lags))
        .collect::<Result<_, _>>()?;
    let (fits, bse): (Vec<QuantileFit>, Vec<Vec<f64>>) = fitted_pairs.into_iter().unzip();

    // Fitted quantiles at EVERY t (the last row is the current risk read).
    let fitted_raw: Vec<Vec<f64>> = fits
        .iter()
        .map(|fit| {
            (0..n)
                .map(|t| {
                    let mut acc = fit.params[0];
                    for (j, col) in conditions.iter().enumerate() {
                        acc += fit.params[1 + j] * col[t];
                    }
                    acc + fit.params[k - 1] * y[t]
                })
                .collect()
        })
        .collect();

    let crossing = (1..taus.len()).any(|j| (0..n).any(|t| fitted_raw[j][t] < fitted_raw[j - 1][t]));

    let fitted = if rearrange {
        rearranged(&fitted_raw, n)
    } else {
        fitted_raw.clone()
    };
    let current = fitted.iter().map(|row| row[n - 1]).collect();

    Ok(GrowthAtRisk {
        taus: taus.to_vec(),
        horizon,
        params: fits.iter().map(|f| f.params.clone()).collect(),
        bse,
        bse_powell: fits.iter().map(|f| f.bse.clone()).collect(),
        hac_lags,
        fitted_raw,
        fitted,
        crossing,
        current,
    })
}

/// Chernozhukov-Fernandez-Val-Galichon rearrangement: sort the fitted
/// values across tau at each evaluation point.
fn rearranged(raw: &[Vec<f64>], n: usize) -> Vec<Vec<f64>> {
    let m = raw.len();
    let mut out = vec![vec![0.0_f64; n]; m];
    let mut column = vec![0.0_f64; m];
    for t in 0..n {
        for (j, row) in raw.iter().enumerate() {
            column[j] = row[t];
        }
        column.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
        for (j, &v) in column.iter().enumerate() {
            out[j][t] = v;
        }
    }
    out
}
