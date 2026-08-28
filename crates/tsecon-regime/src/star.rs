//! Smooth-transition autoregression (STAR): the logistic (LSTAR) and
//! exponential (ESTAR) two-regime models of Terasvirta (1994), with the
//! Terasvirta modeling-cycle test battery — the Luukkonen-Saikkonen-
//! Terasvirta (1988) LM3 linearity test and the H03/H02/H01 sequence for
//! choosing between LSTAR and ESTAR.
//!
//! The model is
//!
//! ```text
//! y_t = phi1' x_t + G(gamma, c; s_t) * phi2' x_t + e_t,
//! x_t = (1?, y_{t-1}, ..., y_{t-p})',   s_t = y_{t-d},
//! ```
//!
//! with transition function
//!
//! ```text
//! LSTAR:  G(gamma, c; s) = 1 / (1 + exp(-gamma (s - c))),
//! ESTAR:  G(gamma, c; s) = 1 - exp(-gamma (s - c)^2),
//! ```
//!
//! `gamma > 0` the smoothness, `c` the location. **Gamma-scaling
//! convention**: the transition function takes the *raw* `gamma` — the
//! convention of R `tsDyn::lstar`, whose sigmoid is
//! `plogis(s, location = th, scale = 1/gamma)` with no standardization —
//! not Terasvirta's (1994) scale-free variant, which divides the exponent
//! by `sd(s)` (LSTAR) or `var(s)` (ESTAR). Both are reported:
//! [`StarFit::gamma`] is raw (tsDyn-comparable), and
//! [`StarFit::gamma_standardized`] is `gamma * sd(s)` for LSTAR and
//! `gamma * var(s)` for ESTAR (Terasvirta-comparable; `sd` is the
//! *population* standard deviation of the usable-sample `s_t`, divisor
//! `n`). The internal grid is built in *standardized* units so the search
//! is scale-equivariant, then mapped back to raw `gamma`.
//!
//! **Estimation** ([`star`]) is concentrated nonlinear least squares:
//! for fixed `(gamma, c)` the model is linear in `(phi1, phi2)`, so the
//! concentrated SSR is an OLS on the `2k` columns `[x_t, G_t x_t]`.
//! A `n_gamma x n_c` grid — `gamma` log-spaced over standardized
//! `[0.5, 100]`, `c` on equally spaced order statistics of `s_t` between
//! the `trim` and `1 - trim` quantiles — locates the basin; the best cell
//! is refined by Nelder-Mead over `(ln gamma_std, c / sd(s))`, with `c`
//! confined to the trimmed range and standardized `gamma` boxed to
//! `[0.5, 1000]` (the grid bottom is a hard wall: below it the transition
//! is numerically linear in `s` and `gamma`/`phi2` are separately
//! unidentified). A refined standardized `gamma` at the grid top (100) or
//! above, or pinned at the bottom wall, sets
//! [`StarFit::gamma_at_boundary`] — at the top the logistic is a step at
//! sample resolution and the SSR surface is flat in `gamma` (Terasvirta
//! 1994's large-gamma advice), so read the estimate as a bound, not a
//! point estimate; [`StarFit::converged`] reports the optimizer verdict.
//!
//! **Standard errors** are the Gauss-Newton NLS ones: `sigma2 *
//! (J'J)^{-1}` with the analytic Jacobian over all `2k + 2` parameters
//! `(phi1, phi2, gamma, c)` and `sigma2 = SSR / (n - 2k - 2)`. Near the
//! large-gamma boundary the `gamma` column of `J` degenerates and `J'J`
//! is numerically singular; the fit then reports NaN standard errors with
//! [`StarFit::se_valid`]` = false` rather than inventing curvature.
//!
//! **The modeling cycle** ([`star_test`]; Terasvirta 1994, JASA 89):
//! linearity is tested by the third-order Taylor auxiliary regression
//!
//! ```text
//! y_t = b0' w_t + b1'(xt~ s_t) + b2'(xt~ s_t^2) + b3'(xt~ s_t^3) + u_t,
//! ```
//!
//! where `w_t = (1, lags)` is the null AR design and `xt~` is the lag
//! block (augmented with `y_{t-d}` itself when `d > p`, Terasvirta's
//! redefinition). LM3 tests `b1 = b2 = b3 = 0`: the chi-squared form is
//! `n (SSR0 - SSR3) / SSR0` with `3q` degrees of freedom, and the F form
//! — recommended in small samples (Terasvirta 1994, sec. 4) — is the
//! standard nested-OLS F with `(3q, n - k0 - 3q)` degrees of freedom.
//! Model choice is the H-sequence of nested F tests: H03 (`b3 = 0`), H02
//! (`b2 = 0 | b3 = 0`), H01 (`b1 = 0 | b2 = b3 = 0`); choose **ESTAR
//! when the H02 p-value is strictly the smallest** of the three, LSTAR
//! otherwise. Several candidate delays are ranked by the F-form LM3
//! p-value (smallest wins — Terasvirta's delay-selection rule).
//!
//! References: Luukkonen, Saikkonen & Terasvirta (1988), Biometrika
//! 75(3); Terasvirta (1994), JASA 89(425); Franses & van Dijk (2000),
//! *Non-Linear Time Series Models in Empirical Finance*, ch. 3; van Dijk,
//! Terasvirta & Franses (2002), Econometric Reviews 21(1).

use crate::error::RegimeError;
use crate::linsolve::{chol_solve, cholesky};
use crate::setar::{build_design, ols_qr, Design};
use tsecon_optim::{minimize, FnObjective, Method, NelderMeadOptions};
use tsecon_stats::chi2_sf;

// ------------------------------------------------------------- constants

/// Standardized-gamma grid range (log-spaced). The grid bottom is also
/// the refinement's lower wall: below standardized `0.5` the transition
/// is numerically linear in `s` over the sample, `[x, Gx]` degenerates
/// toward collinearity, and the linear coefficients blow up in opposite
/// pairs — the low-gamma analog of the large-gamma flat valley (tsDyn's
/// `gammaInt` grid similarly starts at 1 raw).
const GAMMA_STD_GRID_LO: f64 = 0.5;
const GAMMA_STD_GRID_HI: f64 = 100.0;
/// Upper refinement cap for standardized gamma (an order of magnitude
/// past the grid top; beyond the top the SSR surface is flat in gamma).
const GAMMA_STD_CAP: f64 = 1000.0;

// ----------------------------------------------------------- transition

/// The STAR transition-function family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StarModel {
    /// Logistic STAR: `G = 1 / (1 + exp(-gamma (s - c)))` — regimes
    /// differ by the *level* of `s_t` (low vs. high).
    Lstar,
    /// Exponential STAR: `G = 1 - exp(-gamma (s - c)^2)` — regimes differ
    /// by the *distance* of `s_t` from `c` (inner vs. outer), symmetric
    /// about `c`.
    Estar,
}

impl StarModel {
    /// The transition function `G(gamma, c; s)`.
    fn g(self, gamma: f64, c: f64, s: f64) -> f64 {
        match self {
            StarModel::Lstar => 1.0 / (1.0 + (-gamma * (s - c)).exp()),
            StarModel::Estar => 1.0 - (-gamma * (s - c) * (s - c)).exp(),
        }
    }

    /// `(dG/dgamma, dG/dc)` at `(gamma, c, s)` — the analytic Jacobian
    /// pieces for the Gauss-Newton standard errors.
    fn dg(self, gamma: f64, c: f64, s: f64) -> (f64, f64) {
        match self {
            StarModel::Lstar => {
                let g = self.g(gamma, c, s);
                let gg = g * (1.0 - g);
                (gg * (s - c), -gamma * gg)
            }
            StarModel::Estar => {
                let e = (-gamma * (s - c) * (s - c)).exp();
                ((s - c) * (s - c) * e, -2.0 * gamma * (s - c) * e)
            }
        }
    }

    /// The Terasvirta standardization factor: `sd(s)` for LSTAR (gamma
    /// multiplies `s - c`), `var(s)` for ESTAR (gamma multiplies
    /// `(s - c)^2`).
    fn scale(self, sd_s: f64) -> f64 {
        match self {
            StarModel::Lstar => sd_s,
            StarModel::Estar => sd_s * sd_s,
        }
    }

    /// Lower-case name used in results and bindings.
    pub fn name(self) -> &'static str {
        match self {
            StarModel::Lstar => "lstar",
            StarModel::Estar => "estar",
        }
    }
}

// ------------------------------------------------------------ validation

fn validate_common(y: &[f64], p: usize) -> Result<(), RegimeError> {
    for &v in y {
        if !v.is_finite() {
            return Err(RegimeError::NonFinite {
                what: "the input series y (STAR requires finite observations)",
            });
        }
    }
    if p == 0 {
        return Err(RegimeError::InvalidSpec {
            what: "STAR requires p >= 1 (the model is an autoregression; \
                   with p = 0 there is no lag to regress on)",
        });
    }
    if !y.is_empty() && y.iter().all(|&v| v == y[0]) {
        return Err(RegimeError::InvalidSpec {
            what: "the series is constant: a smooth-transition autoregression \
                   needs variation in the transition variable y_{t-d}",
        });
    }
    Ok(())
}

fn validate_delay(delay: usize) -> Result<(), RegimeError> {
    if delay == 0 {
        return Err(RegimeError::InvalidParameter {
            name: "delay",
            value: 0.0,
            requirement: "delay >= 1 (the transition variable is the lagged \
                          value y_{t-delay})",
        });
    }
    Ok(())
}

/// Population (divisor `n`) mean and standard deviation of `z`; errors on
/// a degenerate (near-constant) transition variable, for which neither the
/// standardized gamma nor the trimmed `c` grid is meaningful.
fn transition_scale(z: &[f64]) -> Result<(f64, f64), RegimeError> {
    let n = z.len() as f64;
    let mean = z.iter().sum::<f64>() / n;
    let var = z.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>() / n;
    let sd = var.sqrt();
    if !(sd.is_finite() && sd > 1e-10 * mean.abs().max(1.0)) {
        return Err(RegimeError::InvalidSpec {
            what: "the transition variable y_{t-d} is (near-)constant over \
                   the usable sample: the transition function cannot vary, \
                   so gamma and c are unidentified",
        });
    }
    Ok((mean, sd))
}

// ---------------------------------------------------- concentrated  OLS

/// The concentrated design at fixed `(gamma, c)`: `[x_t, G_t x_t]`.
fn concentrated_cols(design: &Design, model: StarModel, gamma: f64, c: f64) -> Vec<Vec<f64>> {
    let mut cols: Vec<Vec<f64>> = Vec::with_capacity(2 * design.k);
    cols.extend(design.cols.iter().cloned());
    for base in &design.cols {
        cols.push(
            base.iter()
                .zip(&design.z)
                .map(|(&x, &s)| x * model.g(gamma, c, s))
                .collect(),
        );
    }
    cols
}

/// Concentrated SSR at fixed `(gamma, c)` by normal equations + Cholesky
/// (fast path for the grid and the Nelder-Mead objective). `None` when the
/// `2k`-column Gram matrix is not positive definite (e.g. `G_t` nearly
/// constant, making `[x, Gx]` collinear).
fn concentrated_ssr(design: &Design, model: StarModel, gamma: f64, c: f64) -> Option<f64> {
    let k2 = 2 * design.k;
    let cols = concentrated_cols(design, model, gamma, c);
    let mut xtx = vec![0.0_f64; k2 * k2];
    let mut xty = vec![0.0_f64; k2];
    let mut yty = 0.0_f64;
    for t in 0..design.n {
        let yt = design.y[t];
        yty += yt * yt;
        for a in 0..k2 {
            let xa = cols[a][t];
            xty[a] += xa * yt;
            for (b, col_b) in cols.iter().enumerate().take(a + 1) {
                xtx[a * k2 + b] += xa * col_b[t];
            }
        }
    }
    for a in 0..k2 {
        for b in (a + 1)..k2 {
            xtx[a * k2 + b] = xtx[b * k2 + a];
        }
    }
    let l = cholesky(&xtx, k2)?;
    let beta = chol_solve(&l, k2, &xty);
    let fitted: f64 = beta.iter().zip(&xty).map(|(&b, &c)| b * c).sum();
    let ssr = (yty - fitted).max(0.0);
    ssr.is_finite().then_some(ssr)
}

// --------------------------------------------------------------- results

/// Output of [`star_eval`]: the concentrated STAR fit at *fixed*
/// `(gamma, c)` — the linear parameters, Gauss-Newton standard errors,
/// and fit statistics. [`star`] reports the same quantities at the
/// estimated `(gamma, c)`.
#[derive(Debug, Clone, PartialEq)]
pub struct StarEval {
    /// Linear-part coefficients `phi1` (`[constant?, lag 1, ..., lag p]`).
    pub coefs_linear: Vec<f64>,
    /// Nonlinear-part (regime-difference) coefficients `phi2`.
    pub coefs_nonlinear: Vec<f64>,
    /// Gauss-Newton standard errors of `phi1` (`sqrt(sigma2 *
    /// diag[(J'J)^{-1}])` over all `2k + 2` parameters); NaN when
    /// [`se_valid`](StarEval::se_valid) is false.
    pub se_linear: Vec<f64>,
    /// Gauss-Newton standard errors of `phi2`.
    pub se_nonlinear: Vec<f64>,
    /// Gauss-Newton standard error of `gamma` (raw parameterization).
    pub se_gamma: f64,
    /// Gauss-Newton standard error of `c`.
    pub se_c: f64,
    /// Whether the `(2k+2)`-parameter `J'J` was positive definite; false
    /// (with NaN standard errors) when it degenerates — typically at
    /// large gamma, where the SSR surface is flat in `gamma`.
    pub se_valid: bool,
    /// Residual sum of squares.
    pub ssr: f64,
    /// Error variance `SSR / (n - 2k - 2)`.
    pub sigma2: f64,
    /// Gaussian conditional log-likelihood at the ML variance
    /// `SSR / n`: `-n/2 (ln(2 pi SSR/n) + 1)`.
    pub loglik: f64,
    /// Akaike criterion `n ln(SSR/n) + 2 m`, `m = 2k + 2`.
    pub aic: f64,
    /// Schwarz criterion `n ln(SSR/n) + m ln(n)`.
    pub bic: f64,
    /// Usable observations `n = T - max(p, d)`.
    pub nobs: usize,
    /// Regressors per part `k = p + constant`.
    pub k: usize,
    /// The transition-function values `G(gamma, c; s_t)`, in time order.
    pub transition: Vec<f64>,
}

/// Output of [`star`]: the concentrated-NLS STAR fit.
#[derive(Debug, Clone, PartialEq)]
pub struct StarFit {
    /// The fitted family (logistic or exponential).
    pub model: StarModel,
    /// Estimated smoothness `gamma` in the **raw** (tsDyn) convention —
    /// the value inside the transition function.
    pub gamma: f64,
    /// Terasvirta-standardized smoothness: `gamma * sd(s)` (LSTAR) or
    /// `gamma * var(s)` (ESTAR), population `sd` over the usable sample.
    pub gamma_standardized: f64,
    /// Estimated location `c`.
    pub c: f64,
    /// The delay used (selected by refined SSR when several candidates
    /// were supplied).
    pub delay: usize,
    /// The concentrated fit at `(gamma, c)` — coefficients, Gauss-Newton
    /// standard errors, SSR, information criteria, transition path.
    pub eval: StarEval,
    /// Population standard deviation of the transition variable `s_t`
    /// over the usable sample (the gamma standardizer).
    pub s_sd: f64,
    /// Whether the Nelder-Mead refinement satisfied its convergence test.
    pub converged: bool,
    /// True when the refined standardized gamma sits at an edge of the
    /// searchable range. At the **top** (`>= 100`) the transition is
    /// numerically a hard threshold at sample resolution, the SSR surface
    /// is flat in gamma, and the reported `gamma` is a *lower bound*
    /// rather than a point estimate (Terasvirta 1994's large-gamma
    /// advice; the refinement itself is capped at standardized 1000). At
    /// the **bottom** (`<= 0.5`, within rounding) the transition is
    /// numerically linear in `s_t` over the sample, `gamma` and `phi2`
    /// are separately unidentified (only their product is), and the
    /// reported `gamma` is an *upper bound*. Either way: do not read
    /// `gamma` (or its standard error) as a precise estimate.
    pub gamma_at_boundary: bool,
    /// The raw-gamma grid actually scanned (standardized log-spaced
    /// `[0.5, 100]` divided by the standardizer), ascending.
    pub grid_gamma: Vec<f64>,
    /// The location grid: equally spaced order statistics of `s_t`
    /// between the `trim` and `1 - trim` quantiles, ascending.
    pub grid_c: Vec<f64>,
    /// Concentrated SSR per grid cell, row-major over
    /// `(gamma index, c index)`; NaN marks cells whose concentrated OLS
    /// was singular (skipped, never selected).
    pub ssr_grid: Vec<f64>,
    /// `(gamma index, c index)` of the best feasible grid cell.
    pub best_cell: (usize, usize),
    /// Objective evaluations spent by the Nelder-Mead refinement.
    pub fevals: usize,
}

/// One delay's Terasvirta test battery from [`star_test`].
#[derive(Debug, Clone, PartialEq)]
pub struct StarTest {
    /// The delay `d` of the transition variable `s_t = y_{t-d}`.
    pub delay: usize,
    /// Usable observations `n = T - max(p, d)`.
    pub nobs: usize,
    /// Size `q` of the interaction block `xt~` (`p`, or `p + 1` when
    /// `d > p` and `y_{t-d}` is appended — Terasvirta's redefinition).
    pub q: usize,
    /// Columns in the null AR design `w_t` (`1 + q` with the constant).
    pub k0: usize,
    /// LM3 statistic, chi-squared form: `n (SSR0 - SSR3) / SSR0`.
    pub lm3_stat: f64,
    /// Chi-squared p-value of `lm3_stat` with `3q` degrees of freedom.
    pub lm3_p_value: f64,
    /// LM3 statistic, F form (recommended in small samples):
    /// `((SSR0 - SSR3)/3q) / (SSR3/(n - k0 - 3q))`.
    pub lm3_f_stat: f64,
    /// F p-value with `(3q, n - k0 - 3q)` degrees of freedom.
    pub lm3_f_p_value: f64,
    /// H03: F test of the cubic block `b3 = 0` in the full auxiliary
    /// regression, `(q, n - k0 - 3q)` degrees of freedom.
    pub h3_f_stat: f64,
    /// p-value of H03.
    pub h3_p_value: f64,
    /// H02: F test of `b2 = 0` given `b3 = 0`, `(q, n - k0 - 2q)`.
    pub h2_f_stat: f64,
    /// p-value of H02.
    pub h2_p_value: f64,
    /// H01: F test of `b1 = 0` given `b2 = b3 = 0`, `(q, n - k0 - q)`.
    pub h1_f_stat: f64,
    /// p-value of H01.
    pub h1_p_value: f64,
    /// SSR of the null AR regression on `w_t`.
    pub ssr0: f64,
    /// SSR after adding the `xt~ s_t` block.
    pub ssr1: f64,
    /// SSR after adding the `xt~ s_t^2` block.
    pub ssr2: f64,
    /// SSR of the full cubic auxiliary regression.
    pub ssr3: f64,
    /// The Terasvirta (1994) verdict: `Estar` when the H02 p-value is
    /// strictly the smallest of H01/H02/H03, `Lstar` otherwise. Only
    /// meaningful when LM3 rejects linearity.
    pub suggested: StarModel,
}

/// Output of [`star_test`]: one [`StarTest`] battery per candidate delay,
/// plus Terasvirta's delay selection.
#[derive(Debug, Clone, PartialEq)]
pub struct StarTestResult {
    /// One battery per candidate delay, in the order supplied.
    pub tests: Vec<StarTest>,
    /// Index into [`tests`](StarTestResult::tests) of the selected delay:
    /// the smallest F-form LM3 p-value (first on ties) — Terasvirta's
    /// rule of choosing `d` where linearity is rejected most strongly.
    pub best: usize,
}

// ------------------------------------------------------------ star_eval

/// Evaluate the concentrated STAR fit at *fixed* transition parameters
/// `(gamma, c)`: OLS of `y_t` on `[x_t, G_t x_t]`, Gauss-Newton standard
/// errors from the full `(2k+2)`-parameter Jacobian, and fit statistics.
///
/// This is the fixed-parameter workhorse behind [`star`], exposed so a
/// published parameterization can be scored directly (e.g. comparing the
/// SSR/log-likelihood of another package's reported estimates — robust to
/// optimizer differences, unlike parameter-level comparison).
///
/// # Errors
///
/// * [`RegimeError::NonFinite`] for NaN/infinite observations or a
///   non-finite `c`.
/// * [`RegimeError::InvalidSpec`] for `p = 0` or a constant series.
/// * [`RegimeError::InvalidParameter`] for `delay = 0` or `gamma <= 0`.
/// * [`RegimeError::InsufficientData`] when `n < 2k + 3`.
/// * [`RegimeError::Singular`] when `[x, Gx]` is collinear (degenerate
///   transition at these parameters).
pub fn star_eval(
    y: &[f64],
    p: usize,
    delay: usize,
    model: StarModel,
    gamma: f64,
    c: f64,
    constant: bool,
) -> Result<StarEval, RegimeError> {
    validate_common(y, p)?;
    validate_delay(delay)?;
    if !(gamma > 0.0 && gamma.is_finite()) {
        return Err(RegimeError::InvalidParameter {
            name: "gamma",
            value: gamma,
            requirement: "gamma > 0 and finite (the transition smoothness)",
        });
    }
    if !c.is_finite() {
        return Err(RegimeError::NonFinite {
            what: "the transition location c",
        });
    }
    let start = p.max(delay);
    let k = p + usize::from(constant);
    check_star_length(y.len(), start, k)?;
    let design = build_design(y, p, delay, start, constant);
    transition_scale(&design.z)?;
    eval_at(&design, model, gamma, c)
}

fn check_star_length(t: usize, start: usize, k: usize) -> Result<(), RegimeError> {
    // 2k linear parameters plus (gamma, c), plus one residual df.
    let needed = 2 * k + 3;
    let n = t.saturating_sub(start);
    if n < needed {
        return Err(RegimeError::InsufficientData {
            needed: start + needed,
            got: t,
        });
    }
    Ok(())
}

/// The full concentrated fit + Gauss-Newton SEs at fixed `(gamma, c)` on a
/// prebuilt design.
fn eval_at(design: &Design, model: StarModel, gamma: f64, c: f64) -> Result<StarEval, RegimeError> {
    let k = design.k;
    let n = design.n;
    let cols = concentrated_cols(design, model, gamma, c);
    let fit = ols_qr(
        &cols,
        &design.y,
        "the concentrated STAR OLS at (gamma, c): the design [x, G x] is \
         collinear — either the lag columns themselves are linearly dependent \
         (a (near-)constant series or a constant stretch over the usable \
         sample) or the transition G(gamma, c; s_t) is numerically constant, \
         making the two regimes' columns coincide. Check the series for \
         constant segments, reduce p, or move (gamma, c) so the transition \
         actually varies over the sample",
    )?;
    let ssr = fit.ssr;
    if !(ssr > 0.0 && ssr.is_finite()) {
        return Err(RegimeError::NonFinite {
            what: "the STAR residual sum of squares (degenerate perfect fit)",
        });
    }
    let m = 2 * k + 2;
    let nf = n as f64;
    let sigma2 = ssr / (nf - m as f64);
    let loglik = -0.5 * nf * ((2.0 * core::f64::consts::PI * ssr / nf).ln() + 1.0);
    let aic = nf * (ssr / nf).ln() + 2.0 * m as f64;
    let bic = nf * (ssr / nf).ln() + m as f64 * nf.ln();

    let coefs_linear = fit.params[..k].to_vec();
    let coefs_nonlinear = fit.params[k..].to_vec();
    let transition: Vec<f64> = design.z.iter().map(|&s| model.g(gamma, c, s)).collect();

    // Gauss-Newton covariance: J columns [x, Gx, (phi2'x) dG/dgamma,
    // (phi2'x) dG/dc]; se = sqrt(sigma2 diag[(J'J)^{-1}]).
    let mut jac: Vec<Vec<f64>> = cols;
    let mut col_gamma = vec![0.0_f64; n];
    let mut col_c = vec![0.0_f64; n];
    for t in 0..n {
        let mut phi2x = 0.0;
        for (j, base) in design.cols.iter().enumerate() {
            phi2x += coefs_nonlinear[j] * base[t];
        }
        let (dgg, dgc) = model.dg(gamma, c, design.z[t]);
        col_gamma[t] = phi2x * dgg;
        col_c[t] = phi2x * dgc;
    }
    jac.push(col_gamma);
    jac.push(col_c);
    let mp = m;
    let mut jtj = vec![0.0_f64; mp * mp];
    for a in 0..mp {
        for b in 0..=a {
            let dot: f64 = jac[a].iter().zip(&jac[b]).map(|(&x, &y)| x * y).sum();
            jtj[a * mp + b] = dot;
            jtj[b * mp + a] = dot;
        }
    }
    let (se_all, se_valid) = match cholesky(&jtj, mp) {
        Some(l) => {
            // diag[(J'J)^{-1}] via mp unit solves.
            let mut diag = vec![0.0_f64; mp];
            let mut ok = true;
            let mut e = vec![0.0_f64; mp];
            for j in 0..mp {
                e.iter_mut().for_each(|v| *v = 0.0);
                e[j] = 1.0;
                let x = chol_solve(&l, mp, &e);
                diag[j] = x[j];
                if !(x[j].is_finite() && x[j] > 0.0) {
                    ok = false;
                }
            }
            if ok {
                (
                    diag.iter()
                        .map(|&d| (sigma2 * d).sqrt())
                        .collect::<Vec<_>>(),
                    true,
                )
            } else {
                (vec![f64::NAN; mp], false)
            }
        }
        None => (vec![f64::NAN; mp], false),
    };

    Ok(StarEval {
        coefs_linear,
        coefs_nonlinear,
        se_linear: se_all[..k].to_vec(),
        se_nonlinear: se_all[k..2 * k].to_vec(),
        se_gamma: se_all[2 * k],
        se_c: se_all[2 * k + 1],
        se_valid,
        ssr,
        sigma2,
        loglik,
        aic,
        bic,
        nobs: n,
        k,
        transition,
    })
}

// ------------------------------------------------------------------ fit

/// The grid for one design: raw gammas, c candidates, and the SSR surface.
struct Grid {
    gammas: Vec<f64>,
    cs: Vec<f64>,
    ssr: Vec<f64>,
    best: Option<(usize, usize, f64)>,
    /// Trimmed c bounds `[c_lo, c_hi]` for the refinement box.
    c_lo: f64,
    c_hi: f64,
    sd_s: f64,
}

fn build_grid(
    design: &Design,
    model: StarModel,
    trim: f64,
    n_gamma: usize,
    n_c: usize,
) -> Result<Grid, RegimeError> {
    let (_, sd_s) = transition_scale(&design.z)?;
    let scale = model.scale(sd_s);
    let n = design.n;

    // Gamma grid: standardized log-spaced [0.5, 100], mapped to raw.
    let lo = GAMMA_STD_GRID_LO.ln();
    let hi = GAMMA_STD_GRID_HI.ln();
    let gammas: Vec<f64> = (0..n_gamma)
        .map(|j| (lo + (hi - lo) * j as f64 / (n_gamma - 1) as f64).exp() / scale)
        .collect();

    // c grid: equally spaced order statistics between the trim and
    // 1 - trim quantile positions (index = floor(pos + 0.5), matching the
    // NumPy transcription's np.floor(pos + 0.5)).
    let mut zs: Vec<f64> = design.z.clone();
    zs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
    let i_lo = (trim * (n - 1) as f64).ceil() as usize;
    let i_hi = ((1.0 - trim) * (n - 1) as f64).floor() as usize;
    if i_hi <= i_lo {
        return Err(RegimeError::InsufficientData {
            needed: design.n + 2,
            got: design.n,
        });
    }
    let cs: Vec<f64> = (0..n_c)
        .map(|j| {
            let pos = i_lo as f64 + (i_hi - i_lo) as f64 * j as f64 / (n_c - 1) as f64;
            zs[(pos + 0.5).floor() as usize]
        })
        .collect();

    let mut ssr = vec![f64::NAN; n_gamma * n_c];
    let mut best: Option<(usize, usize, f64)> = None;
    for (i, &g) in gammas.iter().enumerate() {
        for (j, &c) in cs.iter().enumerate() {
            if let Some(s) = concentrated_ssr(design, model, g, c) {
                ssr[i * n_c + j] = s;
                let better = match best {
                    None => true,
                    Some((_, _, b)) => s < b,
                };
                if better {
                    best = Some((i, j, s));
                }
            }
        }
    }
    Ok(Grid {
        gammas,
        cs,
        ssr,
        best,
        c_lo: zs[i_lo],
        c_hi: zs[i_hi],
        sd_s,
    })
}

/// Fit a two-regime STAR(`p`) by concentrated nonlinear least squares:
/// grid search over `(gamma, c)` (OLS in the linear parameters per cell)
/// followed by Nelder-Mead refinement of the best cell. See the module
/// docs for the model, the gamma-scaling convention (raw gamma in the
/// transition function, tsDyn-style; the standardized value is reported
/// alongside), the grid, and the boundary/convergence flags.
///
/// `delays` lists the candidate delays for `s_t = y_{t-d}`; all
/// candidates share the common usable sample `t >= max(p, max delays)` so
/// SSRs are comparable, and the delay with the smallest refined SSR wins
/// (first on ties, iterating in the given order). `n_gamma >= 2` and
/// `n_c >= 2` set the grid resolution (25 x 25 is the bindings' default).
///
/// # Errors
///
/// * [`RegimeError::NonFinite`] for NaN/infinite observations.
/// * [`RegimeError::InvalidSpec`] for `p = 0`, an empty `delays` list, a
///   constant series, or a degenerate (near-constant) transition
///   variable.
/// * [`RegimeError::InvalidParameter`] for `trim` outside `(0, 0.5)`,
///   a zero delay, or a grid dimension below 2.
/// * [`RegimeError::InsufficientData`] when the usable sample is shorter
///   than `2k + 3` or the trimmed c grid is empty.
/// * [`RegimeError::Singular`] when every grid cell's concentrated OLS is
///   collinear.
#[allow(clippy::too_many_arguments)] // the estimator's full public spec
pub fn star(
    y: &[f64],
    p: usize,
    delays: &[usize],
    model: StarModel,
    trim: f64,
    constant: bool,
    n_gamma: usize,
    n_c: usize,
) -> Result<StarFit, RegimeError> {
    validate_common(y, p)?;
    if delays.is_empty() {
        return Err(RegimeError::InvalidSpec {
            what: "delays must contain at least one candidate delay",
        });
    }
    for &d in delays {
        validate_delay(d)?;
    }
    if !(trim > 0.0 && trim < 0.5) || !trim.is_finite() {
        return Err(RegimeError::InvalidParameter {
            name: "trim",
            value: trim,
            requirement: "0 < trim < 0.5 (the fraction of transition-variable \
                          order statistics excluded at each end of the c grid)",
        });
    }
    if n_gamma < 2 {
        return Err(RegimeError::InvalidParameter {
            name: "n_gamma",
            value: n_gamma as f64,
            requirement: "n_gamma >= 2 grid points for the smoothness gamma",
        });
    }
    if n_c < 2 {
        return Err(RegimeError::InvalidParameter {
            name: "n_c",
            value: n_c as f64,
            requirement: "n_c >= 2 grid points for the location c",
        });
    }
    let max_delay = delays.iter().copied().max().unwrap_or(1);
    let start = p.max(max_delay);
    let k = p + usize::from(constant);
    check_star_length(y.len(), start, k)?;

    struct Best {
        delay: usize,
        design: Design,
        grid: Grid,
        cell: (usize, usize),
        gamma: f64,
        c: f64,
        ssr: f64,
        converged: bool,
        fevals: usize,
    }
    let mut overall: Option<Best> = None;

    for &d in delays {
        let design = build_design(y, p, d, start, constant);
        let grid = build_grid(&design, model, trim, n_gamma, n_c)?;
        let (bi, bj, grid_ssr) = match grid.best {
            Some(b) => b,
            None => {
                return Err(RegimeError::Singular {
                    what: "every (gamma, c) grid cell's concentrated OLS: the \
                           STAR design [x, G x] is collinear at every candidate \
                           — the lag columns are linearly dependent (a \
                           (near-)constant series or a constant stretch over \
                           the usable sample) or the transition variable \
                           y_{t-d} barely varies, so G(gamma, c; s_t) is \
                           numerically constant everywhere. Check the series \
                           for constant segments, reduce p, or lower trim so \
                           the c grid spans more of the transition variable's \
                           range",
                })
            }
        };

        // Nelder-Mead over (ln gamma_std, c / sd_s) from the best cell,
        // boxed by rejection: c within the trimmed range, standardized
        // gamma within [GAMMA_STD_MIN, GAMMA_STD_CAP].
        let scale = model.scale(grid.sd_s);
        let sd_s = grid.sd_s;
        // The box walls carry a tiny slack: the working-coordinate round
        // trip (c / sd) * sd can land an ulp outside the exact trimmed
        // range when the best grid cell sits on its edge.
        let slack = 1e-9 * sd_s;
        let (c_lo, c_hi) = (grid.c_lo - slack, grid.c_hi + slack);
        let x0 = [(grid.gammas[bi] * scale).ln(), grid.cs[bj] / sd_s];
        // The gamma wall gets the same relative slack as the c walls (the
        // grid-bottom cell sits exactly on the wall after the ln/exp
        // round trip).
        let g_lo = GAMMA_STD_GRID_LO * (1.0 - 1e-9);
        let mut objective = FnObjective::new(|x: &[f64]| {
            let gamma_std = x[0].exp();
            let c = x[1] * sd_s;
            if !(g_lo..=GAMMA_STD_CAP).contains(&gamma_std) || !(c_lo..=c_hi).contains(&c) {
                return f64::INFINITY;
            }
            concentrated_ssr(&design, model, gamma_std / scale, c).unwrap_or(f64::INFINITY)
        });
        let res = minimize(
            &mut objective,
            &x0,
            &Method::NelderMead(NelderMeadOptions {
                // One restart guards against simplex collapse along the
                // flat gamma valley (and doubles the default budget).
                restarts: 1,
                ..NelderMeadOptions::default()
            }),
        )
        .map_err(|_| RegimeError::NonFinite {
            what: "the Nelder-Mead refinement of (gamma, c) (non-finite \
                   objective at the starting simplex)",
        })?;
        let (gamma, c, ssr, converged) = if res.f.is_finite() && res.f <= grid_ssr {
            (
                res.x[0].exp() / scale,
                res.x[1] * sd_s,
                res.f,
                res.converged,
            )
        } else {
            // The refinement never improved on the grid (e.g. an
            // infinity-walled start); fall back to the grid cell.
            (grid.gammas[bi], grid.cs[bj], grid_ssr, false)
        };

        let better = match &overall {
            None => true,
            Some(b) => ssr < b.ssr,
        };
        if better {
            overall = Some(Best {
                delay: d,
                design,
                grid,
                cell: (bi, bj),
                gamma,
                c,
                ssr,
                converged,
                fevals: res.fevals,
            });
        }
    }

    let best = match overall {
        Some(b) => b,
        None => {
            return Err(RegimeError::InvalidSpec {
                what: "delays must contain at least one candidate delay",
            })
        }
    };
    let scale = model.scale(best.grid.sd_s);
    let gamma_standardized = best.gamma * scale;
    let eval = eval_at(&best.design, model, best.gamma, best.c)?;

    Ok(StarFit {
        model,
        gamma: best.gamma,
        gamma_standardized,
        c: best.c,
        delay: best.delay,
        eval,
        s_sd: best.grid.sd_s,
        converged: best.converged,
        gamma_at_boundary: gamma_standardized >= GAMMA_STD_GRID_HI
            || gamma_standardized <= GAMMA_STD_GRID_LO * (1.0 + 1e-9),
        grid_gamma: best.grid.gammas,
        grid_c: best.grid.cs,
        ssr_grid: best.grid.ssr,
        best_cell: (best.cell.0, best.cell.1),
        fevals: best.fevals,
    })
}

// ----------------------------------------------------------------- test

/// F survival function via the regularized incomplete beta, evaluating
/// the upper tail directly (`SF(x) = I_{d2/(d2 + d1 x)}(d2/2, d1/2)`) so
/// small p-values keep relative accuracy. Mirrors
/// `tsecon-spectest::common::f_sf`.
fn f_sf(x: f64, d1: f64, d2: f64) -> Result<f64, RegimeError> {
    if x.is_nan() {
        return Err(RegimeError::NonFinite {
            what: "an F statistic in the STAR linearity battery",
        });
    }
    if x <= 0.0 {
        return Ok(1.0);
    }
    if x == f64::INFINITY {
        return Ok(0.0);
    }
    tsecon_stats::special::beta_inc(d2 / 2.0, d1 / 2.0, d2 / (d2 + d1 * x)).map_err(|_| {
        RegimeError::NonFinite {
            what: "the incomplete-beta evaluation of an F p-value",
        }
    })
}

/// Nested-OLS F statistic and p-value for dropping `r` columns:
/// `((ssr_r - ssr_f)/r) / (ssr_f/df2)`.
fn nested_f(ssr_r: f64, ssr_f: f64, r: usize, df2: usize) -> Result<(f64, f64), RegimeError> {
    if df2 == 0 || ssr_f <= 0.0 || ssr_f.is_nan() {
        return Err(RegimeError::InsufficientData { needed: 1, got: 0 });
    }
    let f = ((ssr_r - ssr_f).max(0.0) / r as f64) / (ssr_f / df2 as f64);
    let p = f_sf(f, r as f64, df2 as f64)?;
    Ok((f, p))
}

/// One delay's battery on a prebuilt design (`xt~` = the lag columns of
/// the null design, plus `s_t` itself when `d > p`).
fn battery(y: &[f64], p: usize, delay: usize, start: usize) -> Result<StarTest, RegimeError> {
    // Estimability FIRST, before any design construction: the full
    // auxiliary regression has k0 + 3q columns (q = p, plus one when
    // d > p appends y_{t-d} itself; k0 = 1 + q with the constant) and
    // needs a residual degree of freedom. Checking before `build_design`
    // keeps a delay past the sample (usable rows <= 0) on the same
    // insufficient-data path as every sibling estimator instead of
    // wrapping the row count.
    let q = p + usize::from(delay > p);
    let k0 = 1 + q;
    let n = y.len().saturating_sub(start);
    if n < k0 + 3 * q + 1 {
        return Err(RegimeError::InsufficientData {
            needed: start + k0 + 3 * q + 1,
            got: y.len(),
        });
    }

    let design = build_design(y, p, delay, start, true);
    transition_scale(&design.z)?;
    let s = &design.z;

    // Null design w = [1, lags, (y_{t-d} if d > p)].
    let mut w: Vec<Vec<f64>> = design.cols.clone();
    // The interaction block xt~: the lag columns (indices 1..=p of w),
    // augmented with s when d > p.
    let mut xtilde: Vec<Vec<f64>> = design.cols[1..].to_vec();
    if delay > p {
        w.push(s.clone());
        xtilde.push(s.clone());
    }
    debug_assert_eq!(q, xtilde.len());
    debug_assert_eq!(k0, w.len());

    let block = |m: u32| -> Vec<Vec<f64>> {
        xtilde
            .iter()
            .map(|col| {
                col.iter()
                    .zip(s)
                    .map(|(&x, &si)| x * si.powi(m as i32))
                    .collect()
            })
            .collect()
    };

    let mut cols = w;
    let fit0 = ols_qr(
        &cols,
        &design.y,
        "the null AR regression of the STAR battery",
    )?;
    cols.extend(block(1));
    let fit1 = ols_qr(&cols, &design.y, "the s-block auxiliary regression")?;
    cols.extend(block(2));
    let fit2 = ols_qr(&cols, &design.y, "the s^2-block auxiliary regression")?;
    cols.extend(block(3));
    let fit3 = ols_qr(&cols, &design.y, "the s^3-block auxiliary regression")?;
    let (ssr0, ssr1, ssr2, ssr3) = (fit0.ssr, fit1.ssr, fit2.ssr, fit3.ssr);
    if !(ssr0 > 0.0 && ssr3 > 0.0) {
        return Err(RegimeError::NonFinite {
            what: "an auxiliary-regression SSR in the STAR linearity battery \
                   (degenerate perfect fit)",
        });
    }

    let nf = n as f64;
    let lm3_stat = nf * (ssr0 - ssr3).max(0.0) / ssr0;
    let lm3_p_value = chi2_sf(lm3_stat, 3.0 * q as f64).map_err(|_| RegimeError::NonFinite {
        what: "the chi-squared evaluation of the LM3 p-value",
    })?;
    let (lm3_f_stat, lm3_f_p_value) = nested_f(ssr0, ssr3, 3 * q, n - k0 - 3 * q)?;
    let (h3_f_stat, h3_p_value) = nested_f(ssr2, ssr3, q, n - k0 - 3 * q)?;
    let (h2_f_stat, h2_p_value) = nested_f(ssr1, ssr2, q, n - k0 - 2 * q)?;
    let (h1_f_stat, h1_p_value) = nested_f(ssr0, ssr1, q, n - k0 - q)?;

    let suggested = if h2_p_value < h1_p_value && h2_p_value < h3_p_value {
        StarModel::Estar
    } else {
        StarModel::Lstar
    };

    Ok(StarTest {
        delay,
        nobs: n,
        q,
        k0,
        lm3_stat,
        lm3_p_value,
        lm3_f_stat,
        lm3_f_p_value,
        h3_f_stat,
        h3_p_value,
        h2_f_stat,
        h2_p_value,
        h1_f_stat,
        h1_p_value,
        ssr0,
        ssr1,
        ssr2,
        ssr3,
        suggested,
    })
}

/// The Terasvirta modeling-cycle test battery: the LM3 linearity test
/// against STAR (Luukkonen-Saikkonen-Terasvirta 1988; chi-squared and
/// small-sample F forms) and the H03/H02/H01 nested F sequence for
/// choosing LSTAR vs. ESTAR (Terasvirta 1994), evaluated at each
/// candidate delay.
///
/// Every statistic is a closed-form nested-OLS comparison on the
/// auxiliary regression `y_t` on `[w_t, xt~ s_t, xt~ s_t^2, xt~ s_t^3]`
/// (see the module docs; `xt~` is augmented with `y_{t-d}` when `d > p`).
/// Unlike the SETAR sup-F, the null distribution here IS standard
/// (chi-squared / F) — the auxiliary regression is linear and no nuisance
/// parameter is unidentified — so no bootstrap is needed.
///
/// Each delay's battery is computed on its own usable sample
/// `t >= max(p, d)` (the per-delay convention of the modeling cycle;
/// p-values, not SSRs, are compared across delays).
/// [`StarTestResult::best`] applies Terasvirta's rule: the delay with the
/// smallest F-form LM3 p-value.
///
/// # Errors
///
/// * [`RegimeError::NonFinite`] for NaN/infinite observations.
/// * [`RegimeError::InvalidSpec`] for `p = 0`, an empty `delays` list, a
///   constant series, or a degenerate transition variable.
/// * [`RegimeError::InvalidParameter`] for a zero delay.
/// * [`RegimeError::InsufficientData`] when the usable sample cannot hold
///   the full auxiliary regression with a residual degree of freedom.
/// * [`RegimeError::Singular`] when an auxiliary regression is collinear.
pub fn star_test(y: &[f64], p: usize, delays: &[usize]) -> Result<StarTestResult, RegimeError> {
    validate_common(y, p)?;
    if delays.is_empty() {
        return Err(RegimeError::InvalidSpec {
            what: "delays must contain at least one candidate delay",
        });
    }
    for &d in delays {
        validate_delay(d)?;
    }
    let mut tests = Vec::with_capacity(delays.len());
    for &d in delays {
        let start = p.max(d);
        tests.push(battery(y, p, d, start)?);
    }
    let mut best = 0usize;
    for (i, t) in tests.iter().enumerate() {
        if t.lm3_f_p_value < tests[best].lm3_f_p_value {
            best = i;
        }
    }
    Ok(StarTestResult { tests, best })
}
