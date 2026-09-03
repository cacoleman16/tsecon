//! Post-double-selection LASSO (Belloni, Chernozhukov & Hansen 2014) for
//! the coefficient on a treatment with high-dimensional controls, with
//! Newey-West HAC inference from the workspace's single HAC engine.
//!
//! # The estimator
//!
//! For the partially linear model `y_t = tau d_t + x_t' beta + e_t` with a
//! scalar treatment `d` and `p` controls `x` (possibly `p > n`), naive
//! *single* selection — LASSO `y` on `x`, then OLS `y` on `d` and the
//! selected controls — is invalid: a control with a modest effect on `y`
//! but a strong effect on `d` is dropped by the `y`-equation LASSO, and its
//! omission biases `tau` by (roughly) its `y`-effect times its `d`-loading,
//! a bias of the same order as the standard error, so the interval
//! undercovers. Belloni, Chernozhukov & Hansen (2014, *Review of Economic
//! Studies* 81(2)) repair this with **double** selection:
//!
//! 1. LASSO `y` on `x` → support `S_y`;
//! 2. LASSO `d` on `x` → support `S_d`;
//! 3. OLS `y` on `[d, x_{S_y ∪ S_d}]`, reading `tau` off `d`.
//!
//! Selecting on the treatment equation too makes the estimating equation
//! Neyman-orthogonal to first-stage selection mistakes, so the resulting
//! interval keeps nominal coverage under sparsity as long as the omitted
//! controls are small in *both* equations. The treatment itself is never
//! penalized. `crate::tests::structured_properties` measures the failure
//! and the repair on a seeded design with autocorrelated errors — the
//! numbers are on the model card.
//!
//! # Penalty choice
//!
//! `alpha` is either one value applied to both LASSOs or the crate's
//! per-equation BIC pick: [`crate::regularization_path`] over its default
//! grid (100 log-spaced points from `lambda_max` down three decades) with
//! `BIC = n ln(RSS/n) + ln(n) df`, `df` the nonzero count (Zou, Hastie &
//! Tibshirani 2007). BCH's theoretical penalty (a normal-quantile rule with
//! iteratively estimated loadings) is not implemented; BIC is the common
//! practical substitute in the time-series literature and is what the
//! Monte-Carlo grade below was measured with.
//!
//! # Inference
//!
//! The final OLS takes its sandwich covariance from [`tsecon_hac::ols`]:
//! Newey-West (Bartlett kernel) with lag truncation `hac_lags`, the
//! finite-sample factor `n / (n - k)` applied (statsmodels `cov_type="HAC"`,
//! `cov_kwds={"maxlags": hac_lags, "use_correction": True}`). `hac_lags =
//! None` resolves to the Newey-West rule of thumb
//! [`tsecon_hac::newey_west_maxlags`] `= floor(4 (n/100)^(2/9))`;
//! `hac_lags = 0` switches to the classical spherical-errors covariance
//! `sigma2 (X'X)^{-1}` (statsmodels `cov_type="nonrobust"`). The p-value and
//! the 95% interval use the standard normal reference distribution in both
//! modes (statsmodels `use_t=False`, its default under `cov_type="HAC"`;
//! its nonrobust default would use `t(n - k)` — pass `use_t=False` to
//! match), because the post-double-selection theory is asymptotic.
//!
//! No intercept is fitted: pass centered `y` and `d` and centered
//! (typically standardized) controls, exactly as for every other routine in
//! this crate.

use tsecon_linalg::faer::MatRef;

use crate::coordinate_descent::{lasso, CoordDescentOptions};
use crate::error::MlError;
use crate::path::{regularization_path, PathOptions};
use crate::util::{check_xy, columns};

/// How the LASSO penalty for the two selection equations is chosen.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PdsAlpha {
    /// One penalty applied to both the `y`-on-`x` and `d`-on-`x` LASSOs.
    Fixed(f64),
    /// The per-equation BIC minimizer along the crate's default
    /// regularization path.
    Bic,
}

/// Result of a post-double-selection fit.
#[derive(Debug, Clone, PartialEq)]
pub struct PdsFit {
    /// Estimated treatment effect `tau` (the coefficient on `d`).
    pub coef: f64,
    /// Its standard error (HAC or classical, per `hac_lags`).
    pub se: f64,
    /// `coef / se`.
    pub t_stat: f64,
    /// Two-sided p-value under the standard normal.
    pub p_value: f64,
    /// 95% interval `coef ∓ 1.959963984540054 * se`.
    pub conf_int: (f64, f64),
    /// Controls selected by the LASSO of `y` on `x`, ascending.
    pub support_y: Vec<usize>,
    /// Controls selected by the LASSO of `d` on `x`, ascending.
    pub support_d: Vec<usize>,
    /// `support_y ∪ support_d`, ascending — the controls in the final OLS.
    pub union_support: Vec<usize>,
    /// `union_support.len()`.
    pub n_controls_selected: usize,
    /// Penalty used for the `y` equation.
    pub alpha_y: f64,
    /// Penalty used for the `d` equation.
    pub alpha_d: f64,
    /// The lag truncation actually used (`0` for classical standard
    /// errors).
    pub hac_lags_resolved: usize,
}

/// `z_{0.975}`, the standard normal 97.5% quantile.
const Z_975: f64 = 1.959963984540054;

/// Select the LASSO support of `target` on `x` under `alpha`.
fn select(
    x: MatRef<'_, f64>,
    cols_len: usize,
    target: &[f64],
    alpha: PdsAlpha,
    opts: CoordDescentOptions,
) -> Result<(f64, Vec<usize>), MlError> {
    let (alpha_used, coef) = match alpha {
        PdsAlpha::Fixed(a) => (a, lasso(x, target, a, opts)?.coef),
        PdsAlpha::Bic => {
            let path = regularization_path(
                x,
                target,
                1.0,
                PathOptions {
                    cd: opts,
                    ..PathOptions::default()
                },
            )?;
            let i = path.bic_best();
            (path.lambdas[i], path.coefs[i].clone())
        }
    };
    let support = (0..cols_len).filter(|&j| coef[j] != 0.0).collect();
    Ok((alpha_used, support))
}

/// Post-double-selection LASSO estimate of the coefficient on `d` in
/// `y = tau d + x beta + e`, with Newey-West HAC (or classical) inference.
///
/// `y` and `d` are centered length-`n` vectors, `x` the centered
/// `n x p` control matrix. See the [module docs](self) for the algorithm,
/// the penalty rule, and the standard-error conventions.
///
/// # Errors
///
/// * [`MlError::EmptyInput`] / [`MlError::DimensionMismatch`] /
///   [`MlError::NonFinite`] on malformed `x`, `y` or `d`;
/// * [`MlError::InvalidArgument`] if a fixed `alpha` is negative or
///   non-finite, or the stopping controls are invalid;
/// * [`MlError::InsufficientData`] if the final OLS on
///   `[d, x_union]` has no residual degrees of freedom;
/// * [`MlError::Hac`] if that OLS design is singular or its covariance
///   breaks down;
/// * [`MlError::NoConvergence`] propagated from the selection LASSOs.
pub fn pds_lasso(
    y: &[f64],
    d: &[f64],
    x: MatRef<'_, f64>,
    alpha: PdsAlpha,
    hac_lags: Option<usize>,
    opts: CoordDescentOptions,
) -> Result<PdsFit, MlError> {
    let (n, p) = check_xy(x, y)?;
    if d.len() != n {
        return Err(MlError::DimensionMismatch {
            what: "d length must equal the number of rows of x",
            expected: n,
            got: d.len(),
        });
    }
    if d.iter().any(|v| !v.is_finite()) {
        return Err(MlError::NonFinite { what: "d" });
    }
    if let PdsAlpha::Fixed(a) = alpha {
        if !a.is_finite() || a < 0.0 {
            return Err(MlError::InvalidArgument {
                what: "alpha must be finite and non-negative (or \"bic\")",
            });
        }
    }

    let (alpha_y, support_y) = select(x, p, y, alpha, opts)?;
    let (alpha_d, support_d) = select(x, p, d, alpha, opts)?;
    let mut union_support: Vec<usize> = support_y.iter().chain(&support_d).copied().collect();
    union_support.sort_unstable();
    union_support.dedup();
    let n_controls_selected = union_support.len();

    let k = 1 + n_controls_selected;
    if n <= k {
        return Err(MlError::InsufficientData {
            got: n,
            needed: k + 1,
            what: "the post-double-selection OLS on the treatment and the union of \
                   selected controls needs residual degrees of freedom; raise alpha \
                   to select fewer controls",
        });
    }

    let cols = columns(x);
    let mut design: Vec<Vec<f64>> = Vec::with_capacity(k);
    design.push(d.to_vec());
    for &j in &union_support {
        design.push(cols[j].clone());
    }
    let fit = tsecon_hac::ols(y, &design)?;

    let hac_lags_resolved = hac_lags.unwrap_or_else(|| tsecon_hac::newey_west_maxlags(n));
    let se_type = if hac_lags_resolved == 0 {
        tsecon_hac::SeType::NonRobust
    } else {
        tsecon_hac::SeType::Hac {
            kernel: tsecon_hac::Kernel::Bartlett,
            bandwidth: hac_lags_resolved as f64,
            use_correction: true,
        }
    };
    let inf = fit.inference(se_type)?;
    let coef = fit.params[0];
    let se = inf.bse[0];
    let t_stat = coef / se;
    let p_value = normal_two_sided_p(t_stat);
    Ok(PdsFit {
        coef,
        se,
        t_stat,
        p_value,
        conf_int: (coef - Z_975 * se, coef + Z_975 * se),
        support_y,
        support_d,
        union_support,
        n_controls_selected,
        alpha_y,
        alpha_d,
        hac_lags_resolved,
    })
}

/// Two-sided standard-normal p-value `2 Phi(-|t|) = erfc(|t| / sqrt 2)`.
fn normal_two_sided_p(t: f64) -> f64 {
    if !t.is_finite() {
        return if t.is_nan() { f64::NAN } else { 0.0 };
    }
    erfc(t.abs() / std::f64::consts::SQRT_2)
}

// ---------------------------------------------------------------------------
// erfc: W. J. Cody's rational Chebyshev approximations ("Rational Chebyshev
// approximation for the error function", Math. Comp. 23 (1969) 631-637;
// SPECFUN CALERF, TOMS 715). Maximum error about 1 ulp. Transcribed here
// rather than taken from `tsecon-stats` so this crate's only new dependency
// is the HAC engine.
// ---------------------------------------------------------------------------

#[allow(clippy::excessive_precision)]
const ERF_A: [f64; 5] = [
    3.16112374387056560e0,
    1.13864154151050156e2,
    3.77485237685302021e2,
    3.20937758913846947e3,
    1.85777706184603153e-1,
];
#[allow(clippy::excessive_precision)]
const ERF_B: [f64; 4] = [
    2.36012909523441209e1,
    2.44024637934444173e2,
    1.28261652607737228e3,
    2.84423683343917062e3,
];
#[allow(clippy::excessive_precision)]
const ERFC_C: [f64; 9] = [
    5.64188496988670089e-1,
    8.88314979438837594e0,
    6.61191906371416295e1,
    2.98635138197400131e2,
    8.81952221241769090e2,
    1.71204761263407058e3,
    2.05107837782607147e3,
    1.23033935479799725e3,
    2.15311535474403846e-8,
];
#[allow(clippy::excessive_precision)]
const ERFC_D: [f64; 8] = [
    1.57449261107098347e1,
    1.17693950891312499e2,
    5.37181101862009858e2,
    1.62138957456669019e3,
    3.29079923573345963e3,
    4.36261909014324716e3,
    3.43936767414372164e3,
    1.23033935480374942e3,
];
#[allow(clippy::excessive_precision)]
const ERFC_P: [f64; 6] = [
    3.05326634961232344e-1,
    3.60344899949804439e-1,
    1.25781726111229246e-1,
    1.60837851487422766e-2,
    6.58749161529837803e-4,
    1.63153871373020978e-2,
];
#[allow(clippy::excessive_precision)]
const ERFC_Q: [f64; 5] = [
    2.56852019228982242e0,
    1.87295284992346047e0,
    5.27905102951428412e-1,
    6.05183413124413191e-2,
    2.33520497626869185e-3,
];
const ERF_THRESH: f64 = 0.46875;
#[allow(clippy::excessive_precision)]
const SQRPI: f64 = 5.6418958354775628695e-1;
const ERFC_XBIG: f64 = 26.543;

fn exp_neg_sq(y: f64) -> f64 {
    let q = (y * 16.0).trunc() / 16.0;
    let del = (y - q) * (y + q);
    (-q * q).exp() * (-del).exp()
}

fn erf_small(x: f64) -> f64 {
    let ysq = x * x;
    let mut num = ERF_A[4] * ysq;
    let mut den = ysq;
    for (&a, &b) in ERF_A[..3].iter().zip(ERF_B[..3].iter()) {
        num = (num + a) * ysq;
        den = (den + b) * ysq;
    }
    x * (num + ERF_A[3]) / (den + ERF_B[3])
}

fn erfc_abs(y: f64) -> f64 {
    if y <= 4.0 {
        let mut num = ERFC_C[8] * y;
        let mut den = y;
        for (&c, &d) in ERFC_C[..7].iter().zip(ERFC_D[..7].iter()) {
            num = (num + c) * y;
            den = (den + d) * y;
        }
        exp_neg_sq(y) * (num + ERFC_C[7]) / (den + ERFC_D[7])
    } else if y < ERFC_XBIG {
        let ysq = 1.0 / (y * y);
        let mut num = ERFC_P[5] * ysq;
        let mut den = ysq;
        for (&p, &q) in ERFC_P[..4].iter().zip(ERFC_Q[..4].iter()) {
            num = (num + p) * ysq;
            den = (den + q) * ysq;
        }
        let r = ysq * (num + ERFC_P[4]) / (den + ERFC_Q[4]);
        exp_neg_sq(y) * (SQRPI - r) / y
    } else {
        0.0
    }
}

/// `erfc(x)` for `x >= 0` (the only range the p-value needs).
fn erfc(x: f64) -> f64 {
    if x <= ERF_THRESH {
        1.0 - erf_small(x)
    } else {
        erfc_abs(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn erfc_matches_reference_points() {
        // scipy.special.erfc at 0.3, 1.0, 2.5, 6.0.
        let refs = [
            (0.3, 0.6713732405408726),
            (1.0, 0.15729920705028513),
            (2.5, 0.0004069520174449589),
            (6.0, 2.1519736712498913e-17),
        ];
        for (x, want) in refs {
            let got = erfc(x);
            assert!(
                ((got - want) / want).abs() < 1e-13,
                "erfc({x}) = {got}, want {want}"
            );
        }
        // p-value at |t| = 1.959963984540054 is 0.05.
        assert!((normal_two_sided_p(Z_975) - 0.05).abs() < 1e-12);
    }
}
