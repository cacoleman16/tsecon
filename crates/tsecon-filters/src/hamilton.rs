//! Hamilton (2018) regression filter — the proposed replacement for the
//! Hodrick-Prescott filter — its random-walk special case, and HAC
//! inference on the regression coefficients.

use crate::decomposition::{Alignment, Decomposition};
use crate::error::{check_finite, FiltersError};
use crate::hp::Frequency;
use crate::lin::householder_lstsq;

/// Result of the Hamilton (2018) regression filter: the OLS coefficients
/// together with the trend (fitted values) / cycle (residuals)
/// decomposition.
#[derive(Debug, Clone, PartialEq)]
pub struct HamiltonResult {
    /// OLS coefficients `[intercept, b_1, ..., b_p]` on
    /// `[1, y_{t-h}, y_{t-h-1}, ..., y_{t-h-p+1}]`.
    pub beta: Vec<f64>,
    /// Fitted values (`trend`) and residuals (`cycle`), aligned to input
    /// observations `h + p - 1, ..., n - 1`
    /// (`alignment.lost_start = h + p - 1`).
    pub decomposition: Decomposition,
}

/// Hamilton's recommended `(h, p)` defaults by sampling frequency: the
/// horizon `h` spans two years and `p` one year of lags — `(2, 1)`
/// annual, `(8, 4)` quarterly, `(24, 12)` monthly (Hamilton 2018,
/// section 4).
pub fn hamilton_defaults(freq: Frequency) -> (usize, usize) {
    match freq {
        Frequency::Annual => (2, 1),
        Frequency::Quarterly => (8, 4),
        Frequency::Monthly => (24, 12),
    }
}

/// Hamilton (2018) regression filter.
///
/// Regresses `y_{t}` on a constant and `p` lags of the series dated `h`
/// periods earlier,
///
/// ```text
/// y_t = beta_0 + beta_1 y_{t-h} + beta_2 y_{t-h-1} + ...
///              + beta_p y_{t-h-p+1} + v_t,
/// ```
///
/// estimated by OLS over `t = h + p - 1, ..., n - 1` (0-indexed). The
/// cycle is the residual `v_t` and the trend the fitted value, so
/// `trend + cycle == y` on the aligned range. The first `h + p - 1`
/// observations are lost (`alignment.lost_start = h + p - 1`,
/// `lost_end = 0`).
///
/// Quarterly defaults are `h = 8`, `p = 4` (see [`hamilton_defaults`]).
/// The regression is solved by Householder QR (numerically stable for
/// the highly collinear lag columns; never normal equations). A constant
/// series makes the lag columns collinear with the intercept and returns
/// [`FiltersError::RankDeficient`].
///
/// Reference: Hamilton (2018), "Why You Should Never Use the
/// Hodrick-Prescott Filter", *Review of Economics and Statistics*
/// 100(5).
pub fn hamilton_filter(y: &[f64], h: usize, p: usize) -> Result<HamiltonResult, FiltersError> {
    if h == 0 {
        return Err(FiltersError::InvalidParameter {
            name: "h",
            value: 0.0,
            requirement: "a forecast horizon >= 1 (Hamilton recommends h = 8 for \
                          quarterly data)",
        });
    }
    if p == 0 {
        return Err(FiltersError::InvalidParameter {
            name: "p",
            value: 0.0,
            requirement: "a lag count >= 1 (Hamilton recommends p = 4 for quarterly \
                          data)",
        });
    }
    let n = y.len();
    let lost = h + p - 1;
    // Need at least p + 1 regression rows for the p + 1 coefficients.
    let needed = lost + p + 1;
    if n < needed {
        return Err(FiltersError::SeriesTooShort {
            filter: "hamilton_filter",
            needed,
            got: n,
            why: "the h-step-ahead regression on p lags discards h + p - 1 rows and then \
                  needs p + 1 more to fit its coefficients (h + 2p in total); lower h or \
                  p, or supply a longer series",
        });
    }
    check_finite(y)?;

    let m = n - lost; // regression rows, one per t = lost..n-1
    let k = p + 1; // intercept + p lags

    // Design matrix in column-major storage: [1, y_{t-h}, ..., y_{t-h-p+1}].
    let mut cols: Vec<Vec<f64>> = Vec::with_capacity(k);
    cols.push(vec![1.0; m]);
    for j in 0..p {
        cols.push((lost..n).map(|t| y[t - h - j]).collect());
    }
    let rhs: Vec<f64> = y[lost..].to_vec();

    let beta = householder_lstsq(cols, rhs, "hamilton_filter OLS")?;

    let mut trend = Vec::with_capacity(m);
    let mut cycle = Vec::with_capacity(m);
    for t in lost..n {
        let mut fit = beta[0];
        for j in 0..p {
            fit += beta[j + 1] * y[t - h - j];
        }
        trend.push(fit);
        cycle.push(y[t] - fit);
    }

    Ok(HamiltonResult {
        beta,
        decomposition: Decomposition {
            trend: Some(trend),
            cycle,
            alignment: Alignment {
                lost_start: lost,
                lost_end: 0,
                input_len: n,
            },
        },
    })
}

/// Random-walk special case of the Hamilton (2018) filter.
///
/// When the series is a random walk (with drift), the population
/// coefficients of the [`hamilton_filter`] regression are
/// `beta_1 = 1` and `beta_0 = beta_2 = ... = beta_p = 0`, so the filter
/// reduces to the `h`-period difference
///
/// ```text
/// cycle_t = y_t - y_{t-h},    trend_t = y_{t-h},
/// ```
///
/// for `t = h, ..., n - 1` (`alignment.lost_start = h`, `lost_end = 0`;
/// `trend + cycle == y` exactly on the aligned range). Hamilton (2018,
/// section 6) recommends this variant when the regression sample is
/// short.
pub fn hamilton_filter_random_walk(y: &[f64], h: usize) -> Result<Decomposition, FiltersError> {
    if h == 0 {
        return Err(FiltersError::InvalidParameter {
            name: "h",
            value: 0.0,
            requirement: "a forecast horizon >= 1 (Hamilton recommends h = 8 for \
                          quarterly data)",
        });
    }
    let n = y.len();
    if n < h + 1 {
        return Err(FiltersError::SeriesTooShort {
            filter: "hamilton_filter_random_walk",
            needed: h + 1,
            got: n,
            why: "the h-step-ahead difference discards the first h rows; lower h or \
                  supply a longer series",
        });
    }
    check_finite(y)?;

    let trend: Vec<f64> = y[..n - h].to_vec();
    let cycle: Vec<f64> = (h..n).map(|t| y[t] - y[t - h]).collect();
    Ok(Decomposition {
        trend: Some(trend),
        cycle,
        alignment: Alignment {
            lost_start: h,
            lost_end: 0,
            input_len: n,
        },
    })
}

/// Which standard errors [`hamilton_filter_with_se`] computes for the
/// regression coefficients.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HamiltonSe {
    /// Classical spherical-errors OLS covariance
    /// (statsmodels `cov_type="nonrobust"`). **Wrong for this
    /// regression** except as a comparison point: the residual
    /// `v_{t}` is an `h`-step-ahead forecast error and is serially
    /// correlated by construction (overlapping horizons make it MA of
    /// order `h - 1` even under a correctly specified model), which
    /// classical standard errors ignore.
    NonRobust,
    /// Newey-West (Bartlett-kernel) HAC sandwich covariance from the
    /// shared `tsecon-hac` engine (statsmodels `cov_type="HAC"`).
    Hac {
        /// Lag truncation. `None` uses the **`h`-overlap default
        /// `maxlags = h`**: the `h`-step-ahead forecast error is MA of
        /// order `h - 1` under correct specification, so the bandwidth
        /// must cover at least `h - 1` lags; `h` covers that with one
        /// lag of slack against mild misspecification. (The generic
        /// Newey-West `0.75 n^(1/3)`-style rules are built for unknown
        /// mixing decay and can sit *below* `h - 1` here — for the
        /// quarterly `h = 8` the rule of thumb gives 4 — cutting off
        /// autocorrelation that is known to exist by construction.)
        maxlags: Option<usize>,
        /// Apply the small-sample `n/(n - k)` correction
        /// (statsmodels `use_correction`). tsecon's Python surface
        /// defaults this to `true` (the finite-sample recommendation);
        /// a default statsmodels `cov_type="HAC"` call uses `false`.
        use_correction: bool,
    },
}

/// Coefficient inference for the Hamilton regression, produced by
/// [`hamilton_filter_with_se`]. Slot `j` corresponds to `beta[j]` of the
/// accompanying [`HamiltonResult`] (`[intercept, b_1, ..., b_p]`).
#[derive(Debug, Clone, PartialEq)]
pub struct HamiltonInference {
    /// Standard errors of the regression coefficients.
    pub bse: Vec<f64>,
    /// t-statistics `beta / bse`.
    pub tvalues: Vec<f64>,
    /// Full parameter covariance matrix, `(p+1) x (p+1)` row-major.
    pub cov: Vec<f64>,
    /// The lag truncation actually used (`None` for
    /// [`HamiltonSe::NonRobust`]; the resolved `h`-overlap default when
    /// `maxlags` was `None`).
    pub maxlags: Option<usize>,
}

/// Hamilton (2018) regression filter with standard errors on the
/// regression coefficients.
///
/// The decomposition and `beta` are **bit-identical** to
/// [`hamilton_filter`] — the filter itself is computed by exactly that
/// function; only the inference is added. Standard errors come from the
/// library's single HAC engine (`tsecon-hac`), whose OLS/HAC sandwich
/// matches statsmodels `OLS(...).fit(cov_type=...)` to golden-fixture
/// precision; the engine's own point estimates agree with the filter's
/// Householder QR solve to well under the golden tolerance (the lag
/// columns are highly collinear, so the two solvers differ at the
/// ~1e-9-relative level on the intercept of a trending series; asserted
/// in the crate tests at 5e-8).
///
/// Because the dependent variable is `y_{t+h}` observed at overlapping
/// horizons, the regression residuals are serially correlated *by
/// construction* (MA(`h - 1`) under correct specification), so
/// [`HamiltonSe::Hac`] with the default `maxlags = h` is the
/// recommended setting; [`HamiltonSe::NonRobust`] is provided as the
/// comparison point Hamilton's own Table 2 makes (his standard errors
/// are Newey-West as well).
///
/// # Errors
///
/// Everything [`hamilton_filter`] rejects, plus
/// [`FiltersError::RankDeficient`] if the HAC engine finds the design
/// collinear (it cannot be reached when the filter itself succeeded,
/// short of pathological rounding).
pub fn hamilton_filter_with_se(
    y: &[f64],
    h: usize,
    p: usize,
    se: HamiltonSe,
) -> Result<(HamiltonResult, HamiltonInference), FiltersError> {
    let result = hamilton_filter(y, h, p)?;
    let lost = h + p - 1;
    let n = y.len();
    let m = n - lost;

    // The identical design hamilton_filter regressed on.
    let mut x_cols: Vec<Vec<f64>> = Vec::with_capacity(p + 1);
    x_cols.push(vec![1.0; m]);
    for j in 0..p {
        x_cols.push((lost..n).map(|t| y[t - h - j]).collect());
    }
    let rhs: Vec<f64> = y[lost..].to_vec();

    let fit = tsecon_hac::ols(&rhs, &x_cols).map_err(|_| FiltersError::RankDeficient {
        what: "hamilton_filter_with_se OLS (shared HAC engine)",
    })?;
    let (se_type, maxlags) = match se {
        HamiltonSe::NonRobust => (tsecon_hac::SeType::NonRobust, None),
        HamiltonSe::Hac {
            maxlags,
            use_correction,
        } => {
            let lags = maxlags.unwrap_or(h);
            (
                tsecon_hac::SeType::Hac {
                    kernel: tsecon_hac::Kernel::Bartlett,
                    bandwidth: lags as f64,
                    use_correction,
                },
                Some(lags),
            )
        }
    };
    let inf = fit
        .inference(se_type)
        .map_err(|_| FiltersError::RankDeficient {
            what: "hamilton_filter_with_se covariance (shared HAC engine)",
        })?;
    Ok((
        result,
        HamiltonInference {
            bse: inf.bse,
            tvalues: inf.tvalues,
            cov: inf.cov,
            maxlags,
        },
    ))
}
