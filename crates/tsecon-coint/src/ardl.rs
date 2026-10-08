//! The **ARDL / unrestricted error-correction model** and the
//! Pesaran-Shin-Smith (2001) **bounds test for a level relationship**
//! ([`ardl_bounds_test`]).
//!
//! # The model
//!
//! With `y` the regressand and `x` the `T x k` regressors (any mix of I(0)
//! and I(1) — that indifference is the point of the bounds test), the
//! ARDL(p, q_1, ..., q_k) in its unrestricted error-correction form
//! (statsmodels `UECM`) is
//!
//! ```text
//! Δy_t = d_t'δ + π y_{t-1} + Σ_j θ_j x_{j,t-1}
//!        + Σ_{i=1}^{p-1} φ_i Δy_{t-i} + Σ_j Σ_{l=0}^{q_j-1} β_{j,l} Δx_{j,t-l} + e_t,
//! ```
//!
//! estimated by OLS on rows `t = h .. T-1`, `h = max(p, max_j q_j)` (so
//! `nobs = T - h`), with `p >= 1` and every `q_j >= 1`. The deterministics
//! `d_t` follow the PSS case: none (case 1), a constant (cases 2 and 3), a
//! constant and a linear trend (cases 4 and 5); the trend is `t + 1` over
//! the FULL sample (statsmodels' `DeterministicProcess`), not `1..nobs`
//! over the regression rows as in [`crate::dols`]. Columns are ordered
//! `[const, trend, y.L1, x1.L1..xk.L1, D.y.L1..D.y.L{p-1}, D.x1.L0..
//! D.x1.L{q1-1}, ...]`, statsmodels' names with 1-based `x` indices.
//!
//! OLS runs by Householder QR on the column-equilibrated design;
//! `s^2 = ssr/(nobs - m)`, `Cov = s^2 (Z'Z)^{-1}` (from `R^{-1}`), p-values
//! Student t on `nobs - m`; `llf = -nobs/2 (ln(2π ssr/nobs) + 1)` and
//! `IC = -2 llf + c (m + 1)` (statsmodels' Lütkepohl form, `c = 2`,
//! `ln nobs`, `2 ln ln nobs`), with `nobs` the regression's ACTUAL row count
//! (statsmodels' `ARDL`/`UECM` report `T - p` instead whenever
//! `max q_j > p` — a defect this crate documents and does not inherit).
//!
//! # The bounds test
//!
//! The PSS F statistic is the Wald test that the level block is zero —
//! `[y.L1, x.L1]`, plus the constant in case 2 or the trend in case 4 —
//! divided by the number of restrictions: `F = (Rb)'(R Cov R')^{-1}(Rb)/q`.
//! Its null distribution is non-standard and depends on whether the
//! regressors are I(0) or I(1); PSS bound it between the all-I(0) (lower)
//! and all-I(1) (upper) limits. The critical values and p-values come
//! from the response surface of [`crate::pss`] (statsmodels' 32-million-draw
//! simulation of the PSS design), looked up at `k = n_x`, the number of
//! regressors. statsmodels' own `bounds_test` looks it up at `1 + n_x` —
//! the wrong row (its own simulator `_pss_simulate`, run at the key it
//! passes, reproduces the `n_x` row). The verdict is read off the
//! p-values: a level relationship when `p_upper < alpha`, none when
//! `p_lower >= alpha`, inconclusive in between.
//!
//! The t statistic on `y.L1` (PSS's second test) is reported but no
//! critical values ship for it; `t_statistic` near zero with a rejecting
//! `F` is PSS's "degenerate case #1" (the levels of `x` matter but `y`
//! does not error-correct) and must be checked by the user.
//!
//! # Order selection
//!
//! A `None` order is chosen by minimising AIC / BIC (default) / HQIC over
//! `p in 1..=P` and every `q_j in 1..=Q` (`P = max_lag`, `Q = max_order`,
//! default 4 each, or the fixed values) on the COMMON sample
//! `h* = max(P, Q)`, candidates in `itertools.product` order (`p`
//! outermost, the last regressor's order fastest), the first strict
//! minimum kept. The winner is then refit on its own sample. This is
//! statsmodels' `ardl_select_order` restricted to the subgrid a UECM can
//! represent (`p >= 1`, `q_j >= 1`); statsmodels' unrestricted grid also
//! admits `q_j = 0` and excluded regressors, which its own
//! `UECM.from_ardl` then refuses.
//!
//! References: Pesaran, Shin & Smith (2001), *Journal of Applied
//! Econometrics* 16(3); Pesaran & Shin (1999), in *Econometrics and
//! Economic Theory in the 20th Century*.

use tsecon_linalg::faer::linalg::solvers::SolveLstsq;
use tsecon_linalg::faer::{Mat, MatRef};
use tsecon_stats::{ContinuousDist, StdNormal, StudentT};

use crate::dols::DolsIc;
use crate::error::CointError;
use crate::fmols::{check_yx, inv_spd_scaled};
use crate::pss;

/// Largest number of candidate models the order search may fit
/// (`P * Q^k`); a larger grid is refused by name.
pub const MAX_GRID: u64 = 100_000;

/// Largest UECM design (`rows x regressors` doubles) the search may build:
/// 2^26 cells, 512 MiB per copy (the search holds the design, its
/// equilibrated copy and one candidate copy at a time).
pub const MAX_DESIGN_CELLS: usize = 1 << 26;

/// Default cap of the searched lag order `p` and of every searched `q_j`.
pub const DEFAULT_MAX: usize = 4;

/// Relative size below which a diagonal entry of the equilibrated `R`
/// factor marks the design as numerically singular.
const RANK_TOL: f64 = 1e-11;

/// The five deterministic cases of Pesaran, Shin & Smith (2001).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PssCase {
    /// Case 1: no deterministic terms.
    NoDeterministics,
    /// Case 2: a constant, restricted to the level relation (tested).
    RestrictedConstant,
    /// Case 3: an unrestricted constant (not tested) — the default.
    UnrestrictedConstant,
    /// Case 4: an unrestricted constant and a trend restricted to the level
    /// relation (the trend is tested).
    RestrictedTrend,
    /// Case 5: an unrestricted constant and trend (neither tested).
    UnrestrictedTrend,
}

impl PssCase {
    /// The case numbered `1 ..= 5`, or `None`.
    #[must_use]
    pub fn from_index(case: usize) -> Option<Self> {
        match case {
            1 => Some(Self::NoDeterministics),
            2 => Some(Self::RestrictedConstant),
            3 => Some(Self::UnrestrictedConstant),
            4 => Some(Self::RestrictedTrend),
            5 => Some(Self::UnrestrictedTrend),
            _ => None,
        }
    }

    /// The PSS case number `1 ..= 5`.
    #[must_use]
    pub fn index(self) -> usize {
        match self {
            Self::NoDeterministics => 1,
            Self::RestrictedConstant => 2,
            Self::UnrestrictedConstant => 3,
            Self::RestrictedTrend => 4,
            Self::UnrestrictedTrend => 5,
        }
    }

    /// The deterministics of the regression: `"n"`, `"c"` or `"ct"`.
    #[must_use]
    pub fn trend_code(self) -> &'static str {
        match self.n_det() {
            0 => "n",
            1 => "c",
            _ => "ct",
        }
    }

    /// Number of deterministic columns (0, 1 or 2).
    #[must_use]
    pub fn n_det(self) -> usize {
        match self {
            Self::NoDeterministics => 0,
            Self::RestrictedConstant | Self::UnrestrictedConstant => 1,
            Self::RestrictedTrend | Self::UnrestrictedTrend => 2,
        }
    }

    /// The deterministic column the test restricts with the level block:
    /// the constant (case 2) or the trend (case 4).
    fn restricted_det(self) -> Option<usize> {
        match self {
            Self::RestrictedConstant => Some(0),
            Self::RestrictedTrend => Some(1),
            _ => None,
        }
    }
}

/// The reading of the bounds test at `alpha`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArdlVerdict {
    /// `p_value_upper < alpha`: F exceeds the I(1) bound — a level
    /// relationship whatever the integration order of the regressors.
    LevelRelationship,
    /// `p_value_lower >= alpha`: F is below the I(0) bound — no level
    /// relationship whatever the integration order.
    NoLevelRelationship,
    /// Between the bounds: the answer depends on the integration order of
    /// the regressors.
    Inconclusive,
    /// No tabulated bounds for this many regressors (`n_x > 10`).
    Unavailable,
}

impl ArdlVerdict {
    /// The spelling the Python surface returns.
    #[must_use]
    pub fn code(self) -> &'static str {
        match self {
            Self::LevelRelationship => "level_relationship",
            Self::NoLevelRelationship => "no_level_relationship",
            Self::Inconclusive => "inconclusive",
            Self::Unavailable => "unavailable",
        }
    }
}

/// Options of [`ardl_bounds_test`].
#[derive(Debug, Clone, PartialEq)]
pub struct ArdlOptions {
    /// Fixed ARDL order `p >= 1` of `y`, or `None` (default) to select it.
    pub lags: Option<usize>,
    /// Fixed order `q_j >= 1` of each regressor (one per column of `x`),
    /// or `None` (default) to select them.
    pub order: Option<Vec<usize>>,
    /// The PSS case (default [`PssCase::UnrestrictedConstant`], case 3).
    pub case: PssCase,
    /// Criterion of the order search (default [`DolsIc::Bic`]). Its value
    /// is reported on statsmodels' `-2 llf + c (m + 1)` scale, NOT on
    /// [`crate::dols`]'s `ln(RSS/n) + m c/n` scale.
    pub ic: DolsIc,
    /// Cap of the searched `p` (`None` = [`DEFAULT_MAX`]); refused when
    /// `lags` is fixed.
    pub max_lag: Option<usize>,
    /// Cap of every searched `q_j` (`None` = [`DEFAULT_MAX`]); refused
    /// when `order` is fixed.
    pub max_order: Option<usize>,
    /// Level of the verdict, in `(0, 1)` (default 0.05).
    pub alpha: f64,
}

impl Default for ArdlOptions {
    fn default() -> Self {
        Self {
            lags: None,
            order: None,
            case: PssCase::UnrestrictedConstant,
            ic: DolsIc::Bic,
            max_lag: None,
            max_order: None,
            alpha: 0.05,
        }
    }
}

/// Result of [`ardl_bounds_test`].
#[derive(Debug, Clone, PartialEq)]
pub struct ArdlResult {
    /// The PSS F statistic (Wald / number of restrictions).
    pub statistic: f64,
    /// The t statistic on `y.L1` (no critical values ship for it).
    pub t_statistic: f64,
    /// The PSS case.
    pub case: PssCase,
    /// Number of regressors `k` — the PSS table key.
    pub n_x: usize,
    /// Number of restrictions of the F test (`1 + n_x`, plus one in cases
    /// 2 and 4).
    pub n_restrictions: usize,
    /// The levels of the critical values, `[0.90, 0.95, 0.99, 0.999]`.
    pub crit_levels: [f64; 4],
    /// I(0) (lower) bounds at `crit_levels` (NaN when `n_x > 10`).
    pub crit_lower: [f64; 4],
    /// I(1) (upper) bounds at `crit_levels` (NaN when `n_x > 10`).
    pub crit_upper: [f64; 4],
    /// Response-surface p-value under the I(0) bound.
    pub p_value_lower: f64,
    /// Response-surface p-value under the I(1) bound.
    pub p_value_upper: f64,
    /// The reading at `alpha`, from the p-values.
    pub verdict: ArdlVerdict,
    /// Echo of `alpha`.
    pub alpha: f64,
    /// Why the critical values / p-values are NaN, or `None`.
    pub crit_note: Option<String>,
    /// UECM coefficients in design order (`param_names`).
    pub params: Vec<f64>,
    /// Their OLS standard errors.
    pub se: Vec<f64>,
    /// `params / se`.
    pub tvalues: Vec<f64>,
    /// Two-sided Student-t p-values on `df_resid`.
    pub pvalues: Vec<f64>,
    /// `s^2 (Z'Z)^{-1}` (rows).
    pub cov: Vec<Vec<f64>>,
    /// Names of `params`.
    pub param_names: Vec<String>,
    /// The error-correction coefficient on `y.L1` (`π`).
    pub adjustment: f64,
    /// Number of deterministic columns.
    pub n_det: usize,
    /// Number of UECM regressors `m = n_det + p + Σ (q_j + 1)`.
    pub n_params: usize,
    /// Long-run coefficients `-b_i / π` of the deterministics and the
    /// regressors (statsmodels `ci_params` times -1, without `y`).
    pub long_run: Vec<f64>,
    /// Delta-method standard errors of `long_run`.
    pub long_run_se: Vec<f64>,
    /// `long_run / long_run_se`.
    pub long_run_tvalues: Vec<f64>,
    /// Two-sided normal p-values of `long_run_tvalues`.
    pub long_run_pvalues: Vec<f64>,
    /// Names of `long_run` (`const`, `trend`, `x1`, ...).
    pub long_run_names: Vec<String>,
    /// Residuals `Δy - Z b` (length `nobs`).
    pub resid: Vec<f64>,
    /// Rows of the regression, `T - hold_back`.
    pub nobs: usize,
    /// Sample size `T`.
    pub n_total: usize,
    /// Presample rows, `max(lags, max order)`.
    pub hold_back: usize,
    /// `nobs - n_params`.
    pub df_resid: usize,
    /// Residual sum of squares.
    pub ssr: f64,
    /// Gaussian log-likelihood at `ssr / nobs`.
    pub llf: f64,
    /// `-2 llf + 2 (m + 1)`.
    pub aic: f64,
    /// `-2 llf + ln(nobs) (m + 1)`.
    pub bic: f64,
    /// `-2 llf + 2 ln(ln nobs) (m + 1)`.
    pub hqic: f64,
    /// The ARDL order `p` of `y` used.
    pub lags: usize,
    /// The orders `q_j` used.
    pub order: Vec<usize>,
    /// Whether any order came from the search.
    pub selected: bool,
    /// The search criterion.
    pub ic: DolsIc,
    /// The minimised criterion on the common sample (NaN when both `lags`
    /// and `order` were fixed).
    pub ic_value: f64,
    /// The cap of `p` the search used (`lags` when fixed).
    pub max_lag: usize,
    /// The cap of the `q_j` the search used (`max(order)` when fixed).
    pub max_order: usize,
    /// Candidate models compared (1 when both were fixed).
    pub n_models: usize,
}

/// OLS pieces of one fit.
struct Fit {
    b: Vec<f64>,
    resid: Vec<f64>,
    ssr: f64,
    xtx_inv: Option<Mat<f64>>,
}

/// Power-of-two column scales `2^-round(log2 ||col||)`.
fn column_scales(z: MatRef<'_, f64>) -> Vec<f64> {
    (0..z.ncols())
        .map(|j| {
            let norm = (0..z.nrows())
                .map(|i| z[(i, j)] * z[(i, j)])
                .sum::<f64>()
                .sqrt();
            if norm > 0.0 && norm.is_finite() {
                2f64.powi(-(norm.log2().round() as i32))
            } else {
                1.0
            }
        })
        .collect()
}

/// OLS of `y` on `z` by Householder QR on the equilibrated design;
/// `(Z'Z)^{-1} = S R^{-1} R^{-T} S` when `want_inv`.
fn qr_fit(
    z: MatRef<'_, f64>,
    y: &[f64],
    want_inv: bool,
    what: &'static str,
) -> Result<Fit, CointError> {
    let (n, m) = (z.nrows(), z.ncols());
    let s = column_scales(z);
    let zs = Mat::from_fn(n, m, |i, j| z[(i, j)] * s[j]);
    let qr = zs.qr();
    let r = qr.thin_R();
    let dmax = (0..m).map(|j| r[(j, j)].abs()).fold(0.0_f64, f64::max);
    // `<=` is false for NaN, so a NaN diagonal is caught by the finiteness
    // check of the coefficients below.
    if !(dmax > 0.0 && dmax.is_finite()) || (0..m).any(|j| r[(j, j)].abs() <= RANK_TOL * dmax) {
        return Err(CointError::Singular { what });
    }
    let ym = Mat::from_fn(n, 1, |i, _| y[i]);
    let bs = qr.solve_lstsq(&ym);
    let b: Vec<f64> = (0..m).map(|j| bs[(j, 0)] * s[j]).collect();
    if b.iter().any(|v| !v.is_finite()) {
        return Err(CointError::Singular { what });
    }
    let resid: Vec<f64> = (0..n)
        .map(|i| y[i] - (0..m).map(|j| z[(i, j)] * b[j]).sum::<f64>())
        .collect();
    let ssr = resid.iter().map(|e| e * e).sum();
    let xtx_inv = if want_inv {
        let mut rinv = Mat::<f64>::identity(m, m);
        let rr = r.get(..m, ..m);
        rr.solve_upper_triangular_in_place(&mut rinv);
        let inner = &rinv * rinv.transpose();
        Some(Mat::from_fn(m, m, |i, j| s[i] * s[j] * inner[(i, j)]))
    } else {
        None
    };
    Ok(Fit {
        b,
        resid,
        ssr,
        xtx_inv,
    })
}

/// The UECM design on rows `t = h .. T-1` (`h >= max(p, max q)`):
/// `(Δy, Z)` with the column layout of the module docs.
fn design(
    y: &[f64],
    x: MatRef<'_, f64>,
    n_det: usize,
    p: usize,
    q: &[usize],
    h: usize,
) -> (Vec<f64>, Mat<f64>) {
    let n = y.len();
    let kx = x.ncols();
    let nobs = n - h;
    let m = n_det + p + q.iter().map(|v| v + 1).sum::<usize>();
    // column -> generator
    let mut cols: Vec<Col> = Vec::with_capacity(m);
    for d in 0..n_det {
        cols.push(if d == 0 { Col::Const } else { Col::Trend });
    }
    cols.push(Col::YLevel);
    for j in 0..kx {
        cols.push(Col::XLevel(j));
    }
    for i in 1..p {
        cols.push(Col::DyLag(i));
    }
    for (j, &qj) in q.iter().enumerate() {
        for l in 0..qj {
            cols.push(Col::DxLag(j, l));
        }
    }
    let z = Mat::from_fn(nobs, m, |i, c| {
        let t = i + h;
        match cols[c] {
            Col::Const => 1.0,
            Col::Trend => (t + 1) as f64,
            Col::YLevel => y[t - 1],
            Col::XLevel(j) => x[(t - 1, j)],
            Col::DyLag(l) => y[t - l] - y[t - l - 1],
            Col::DxLag(j, l) => x[(t - l, j)] - x[(t - l - 1, j)],
        }
    });
    let dy: Vec<f64> = (h..n).map(|t| y[t] - y[t - 1]).collect();
    (dy, z)
}

#[derive(Clone, Copy)]
enum Col {
    Const,
    Trend,
    YLevel,
    XLevel(usize),
    DyLag(usize),
    DxLag(usize, usize),
}

fn param_names(n_det: usize, kx: usize, p: usize, q: &[usize]) -> Vec<String> {
    let mut names: Vec<String> = ["const", "trend"][..n_det]
        .iter()
        .map(|s| (*s).to_string())
        .collect();
    names.push("y.L1".to_string());
    for j in 1..=kx {
        names.push(format!("x{j}.L1"));
    }
    for i in 1..p {
        names.push(format!("D.y.L{i}"));
    }
    for (j, &qj) in q.iter().enumerate() {
        for l in 0..qj {
            names.push(format!("D.x{}.L{l}", j + 1));
        }
    }
    names
}

fn gaussian_llf(ssr: f64, nobs: usize) -> f64 {
    let n = nobs as f64;
    -n / 2.0 * ((2.0 * core::f64::consts::PI * ssr / n).ln() + 1.0)
}

fn criterion(ic: DolsIc, llf: f64, nobs: usize, m: usize) -> f64 {
    -2.0 * llf + ic.penalty(nobs as f64) * (m + 1) as f64
}

fn show_list(v: &[usize]) -> String {
    let parts: Vec<String> = v.iter().map(|q| q.to_string()).collect();
    format!("[{}]", parts.join(", "))
}

/// The reading of the p-values at `alpha`.
fn read_verdict(p_lower: f64, p_upper: f64, alpha: f64) -> ArdlVerdict {
    if p_lower.is_nan() || p_upper.is_nan() {
        ArdlVerdict::Unavailable
    } else if p_upper < alpha {
        ArdlVerdict::LevelRelationship
    } else if p_lower >= alpha {
        ArdlVerdict::NoLevelRelationship
    } else {
        ArdlVerdict::Inconclusive
    }
}

/// Validated specification of one call.
struct Spec {
    /// Cap (or fixed value) of `p`.
    p_cap: usize,
    /// Per-regressor cap (or fixed value) of `q_j`.
    q_cap: Vec<usize>,
    /// Common-sample hold-back of the search, `max(P, Q)`.
    h_star: usize,
}

fn validate(n: usize, kx: usize, opts: &ArdlOptions) -> Result<Spec, CointError> {
    let spec = |what: String| CointError::InvalidSpec { what };
    if !(opts.alpha > 0.0 && opts.alpha < 1.0) {
        return Err(spec(format!(
            "alpha = {} is outside (0, 1): it is the significance level the verdict compares \
             the bounds-test p-values with; pass e.g. 0.05",
            fmt_f64(opts.alpha)
        )));
    }
    if opts.lags == Some(0) {
        return Err(spec(
            "lags = 0 is not a valid ARDL order: the error-correction form needs the lagged \
             level y_{t-1}, so the order p of y is at least 1 (lags = 1 keeps the level and no \
             lagged differences of y). statsmodels' UECM silently accepts lags = 0 and builds \
             the lags = 1 design; tsecon refuses instead. Pass lags >= 1, or lags = None to \
             select it"
                .to_string(),
        ));
    }
    if let Some(q) = &opts.order {
        if q.len() != kx {
            return Err(spec(format!(
                "order has {} entr{} but x has {kx} column(s): pass one order q_j >= 1 per \
                 regressor (or a single int for all of them), or order = None to select them",
                q.len(),
                if q.len() == 1 { "y" } else { "ies" }
            )));
        }
        if let Some(j) = q.iter().position(|&v| v == 0) {
            return Err(spec(format!(
                "order[{j}] = 0 is not supported: an order-0 regressor (x_j entering only \
                 through its level, with no current or lagged difference — Stata's ardl grid \
                 admits it) has no unrestricted error-correction form here, and supporting it \
                 is a planned follow-up. Pass q_j >= 1 for every regressor, or order = None to \
                 search q_j in 1..max_order"
            )));
        }
    }
    if let (Some(m), Some(p)) = (opts.max_lag, opts.lags) {
        return Err(spec(format!(
            "max_lag = {m} was given but lags = {p} is fixed: the cap only bounds a searched \
             lag order, so it would be inert; drop max_lag or pass lags = None"
        )));
    }
    if let (Some(m), Some(q)) = (opts.max_order, &opts.order) {
        return Err(spec(format!(
            "max_order = {m} was given but order = {} is fixed: the cap only bounds searched \
             orders, so it would be inert; drop max_order or pass order = None",
            show_list(q)
        )));
    }
    if opts.max_lag == Some(0) {
        return Err(spec(
            "max_lag = 0 leaves the search no candidate: the order p of y runs over \
             1..max_lag, so pass max_lag >= 1 (or None for the default 4)"
                .to_string(),
        ));
    }
    if opts.max_order == Some(0) {
        return Err(spec(
            "max_order = 0 leaves the search no candidate: every order q_j runs over \
             1..max_order (an order-0 regressor is not supported), so pass max_order >= 1 \
             (or None for the default 4)"
                .to_string(),
        ));
    }
    let p_cap = opts.lags.or(opts.max_lag).unwrap_or(DEFAULT_MAX);
    let q_cap: Vec<usize> = match &opts.order {
        Some(q) => q.clone(),
        None => vec![opts.max_order.unwrap_or(DEFAULT_MAX); kx],
    };
    let q_max = q_cap.iter().copied().max().unwrap_or(1);

    // grid guard (saturating: the caps are user input)
    let n_lag: u64 = if opts.lags.is_some() { 1 } else { p_cap as u64 };
    let n_ord: u64 = if opts.order.is_some() {
        1
    } else {
        let base = q_max as u64;
        u32::try_from(kx)
            .ok()
            .and_then(|e| base.checked_pow(e))
            .unwrap_or(u64::MAX)
    };
    let grid = n_lag.saturating_mul(n_ord);
    if grid > MAX_GRID {
        let lag_part = match (opts.lags, opts.max_lag) {
            (Some(p), _) => format!("lags = {p} fixed"),
            (None, Some(m)) => format!("max_lag = {m}"),
            (None, None) => format!("max_lag = {DEFAULT_MAX} (the default)"),
        };
        let ord_part = match opts.max_order {
            Some(m) => format!("max_order = {m}"),
            None => format!("max_order = {DEFAULT_MAX} (the default)"),
        };
        let count = if grid == u64::MAX {
            "more than 1.8e19".to_string()
        } else {
            grid.to_string()
        };
        return Err(spec(format!(
            "the order search would fit {count} candidate models, above the {MAX_GRID} guard: \
             with {lag_part} and {ord_part} the grid is P x Q^k with k = {kx} regressors, so it \
             grows geometrically in k. Fix order (an int for every regressor, e.g. order = 1, \
             or a list) and search only the lags, or lower max_order / max_lag"
        )));
    }

    // the largest candidate must leave residual degrees of freedom
    let h_star = p_cap.max(q_max);
    let n_det = opts.case.n_det();
    let m_max = q_cap.iter().fold(n_det.saturating_add(p_cap), |acc, &q| {
        acc.saturating_add(q.saturating_add(1))
    });
    let rows = n.saturating_sub(h_star);
    if rows <= m_max {
        let lag_part = match (opts.lags, opts.max_lag) {
            (Some(p), _) => format!("lags = {p}"),
            (None, Some(m)) => format!("max_lag = {m}"),
            (None, None) => format!("max_lag = {DEFAULT_MAX} (the default)"),
        };
        let ord_part = match (&opts.order, opts.max_order) {
            (Some(q), _) => format!("order = {}", show_list(q)),
            (None, Some(m)) => format!("max_order = {m}"),
            (None, None) => format!("max_order = {DEFAULT_MAX} (the default)"),
        };
        return Err(spec(format!(
            "the ARDL design is too large for T = {n} observations: with {lag_part} and \
             {ord_part} the largest model keeps T - max(p, q) = {rows} row(s) for \
             n_det + p + sum(q_j + 1) = {m_max} regressors (case {} has {n_det} deterministic \
             column(s)), which leaves no residual degrees of freedom; lower the orders or \
             their caps, drop regressors, or supply a longer sample",
            opts.case.index()
        )));
    }
    let cells = rows.saturating_mul(m_max);
    if cells > MAX_DESIGN_CELLS {
        return Err(spec(format!(
            "the ARDL design would hold {rows} x {m_max} = {cells} entries, above the \
             {MAX_DESIGN_CELLS}-entry (512 MiB) budget the order search may allocate; \
             lower the orders or their caps (lags / order / max_lag / max_order), or \
             estimate on a shorter sample"
        )));
    }
    Ok(Spec {
        p_cap,
        q_cap,
        h_star,
    })
}

fn fmt_f64(v: f64) -> String {
    if v.is_nan() {
        "NaN".to_string()
    } else {
        v.to_string()
    }
}

/// The column indices of candidate `(p, q)` inside the largest design
/// built at `(P, Q_j)`.
fn candidate_columns(
    n_det: usize,
    kx: usize,
    p_cap: usize,
    q_cap: &[usize],
    p: usize,
    q: &[usize],
) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..n_det + 1 + kx).collect();
    let dy0 = n_det + 1 + kx;
    idx.extend(dy0..dy0 + (p - 1));
    let mut start = dy0 + (p_cap - 1);
    for (j, &qj) in q.iter().enumerate() {
        idx.extend(start..start + qj);
        start += q_cap[j];
    }
    idx
}

/// The order search on the common sample `h*`: `(p, q, ic_value,
/// n_models)`.
fn search(
    y: &[f64],
    x: MatRef<'_, f64>,
    opts: &ArdlOptions,
    spec: &Spec,
) -> Result<(usize, Vec<usize>, f64, usize), CointError> {
    let kx = x.ncols();
    let n_det = opts.case.n_det();
    let (dy, z) = design(y, x, n_det, spec.p_cap, &spec.q_cap, spec.h_star);
    let nobs = dy.len();
    // the largest design must be of full rank (every candidate is a column
    // subset of it on the same rows)
    qr_fit(
        z.as_ref(),
        &dy,
        false,
        "the largest ARDL design of the order search (a regressor, its lagged level or a \
         lagged difference is an exact linear combination of the others — a duplicated or \
         constant column of x, or x collinear with the case's deterministics)",
    )?;
    let p_range: Vec<usize> = match opts.lags {
        Some(p) => vec![p],
        None => (1..=spec.p_cap).collect(),
    };
    let fixed_q = opts.order.clone();
    let mut q = match &fixed_q {
        Some(v) => v.clone(),
        None => vec![1; kx],
    };
    let mut best: Option<(usize, Vec<usize>, f64)> = None;
    let mut n_models = 0usize;
    for &p in &p_range {
        if fixed_q.is_none() {
            q.iter_mut().for_each(|v| *v = 1);
        }
        loop {
            let cols = candidate_columns(n_det, kx, spec.p_cap, &spec.q_cap, p, &q);
            let sub = Mat::from_fn(nobs, cols.len(), |i, c| z[(i, cols[c])]);
            let fit = qr_fit(
                sub.as_ref(),
                &dy,
                false,
                "a candidate ARDL design of the order search is collinear",
            )?;
            let ic = criterion(opts.ic, gaussian_llf(fit.ssr, nobs), nobs, cols.len());
            n_models += 1;
            if best.as_ref().is_none_or(|b| ic < b.2) {
                best = Some((p, q.clone(), ic));
            }
            // odometer: the last regressor's order runs fastest
            if fixed_q.is_some() {
                break;
            }
            let mut j = kx;
            let mut carried = true;
            while j > 0 && carried {
                j -= 1;
                if q[j] < spec.q_cap[j] {
                    q[j] += 1;
                    carried = false;
                } else {
                    q[j] = 1;
                }
            }
            if carried {
                break;
            }
        }
    }
    match best {
        Some((p, q, ic)) if ic.is_finite() => Ok((p, q, ic, n_models)),
        _ => Err(CointError::Singular {
            what: "every candidate of the ARDL order search has a non-finite criterion \
                   (a perfect fit: the residual sum of squares is zero)",
        }),
    }
}

/// The ARDL / UECM fit and the Pesaran-Shin-Smith (2001) bounds test of
/// `y` (length `T`) on the `T x k` regressors `x` — see the module docs
/// for the model, the test, the order search and the two statsmodels
/// defects this function does not inherit.
///
/// Reproduces statsmodels 0.15 `UECM(y, p, x, q, trend).fit()` (params,
/// covariance, residuals), `bounds_test(case).statistic` and the levels
/// `ARDL` affine `f_test`, and `ardl_select_order` on the admissible
/// subgrid, to 1e-10 on the goldens (`fixtures/ardl.json`).
///
/// # Errors
///
/// [`CointError::Dimension`] for an empty `x` or mismatched lengths;
/// [`CointError::NonFinite`] / [`CointError::NonFiniteSeries`] on NaN or
/// infinity; [`CointError::InvalidSpec`] (naming the parameter) for
/// `lags = 0`, an `order` of the wrong length or with a zero, `max_lag` /
/// `max_order` with the value they cap fixed (or zero), a search grid
/// above [`MAX_GRID`], a design with no residual degrees of freedom or
/// above [`MAX_DESIGN_CELLS`] entries, or `alpha` outside `(0, 1)`;
/// [`CointError::Singular`] for a collinear design.
pub fn ardl_bounds_test(
    y: &[f64],
    x: MatRef<'_, f64>,
    opts: &ArdlOptions,
) -> Result<ArdlResult, CointError> {
    let (n, kx) = check_yx(y, x)?;
    let spec = validate(n, kx, opts)?;
    let case = opts.case;
    let n_det = case.n_det();

    let both_fixed = opts.lags.is_some() && opts.order.is_some();
    let (p, q, ic_value, n_models) = if both_fixed {
        (spec.p_cap, spec.q_cap.clone(), f64::NAN, 1usize)
    } else {
        search(y, x, opts, &spec)?
    };

    // final fit on the winner's own sample
    let h = q.iter().copied().fold(p, usize::max);
    let (dy, z) = design(y, x, n_det, p, &q, h);
    let nobs = dy.len();
    let m = z.ncols();
    let fit = qr_fit(
        z.as_ref(),
        &dy,
        true,
        "the ARDL / UECM design (a regressor, its lagged level or a lagged difference is an \
         exact linear combination of the others — a duplicated or constant column of x, or x \
         collinear with the case's deterministics); drop the redundant column",
    )?;
    let df_resid = nobs - m;
    let s2 = fit.ssr / df_resid as f64;
    let xtx_inv = fit.xtx_inv.unwrap_or_else(|| Mat::zeros(m, m));
    let cov: Vec<Vec<f64>> = (0..m)
        .map(|i| (0..m).map(|j| s2 * xtx_inv[(i, j)]).collect())
        .collect();
    let b = fit.b;
    let se: Vec<f64> = (0..m).map(|j| cov[j][j].max(0.0).sqrt()).collect();
    let tvalues: Vec<f64> = b.iter().zip(&se).map(|(v, s)| v / s).collect();
    let tdist = StudentT::new(df_resid as f64)?;
    let pvalues: Vec<f64> = tvalues
        .iter()
        .map(|t| {
            if t.is_nan() {
                f64::NAN
            } else {
                2.0 * tdist.sf(t.abs())
            }
        })
        .collect();
    let llf = gaussian_llf(fit.ssr, nobs);
    let aic = criterion(DolsIc::Aic, llf, nobs, m);
    let bic = criterion(DolsIc::Bic, llf, nobs, m);
    let hqic = criterion(DolsIc::Hqic, llf, nobs, m);

    // the PSS F statistic: Wald on the level block (+ restricted det) / q
    let mut rest: Vec<usize> = Vec::with_capacity(kx + 2);
    if let Some(d) = case.restricted_det() {
        rest.push(d);
    }
    rest.extend(n_det..n_det + 1 + kx);
    let nr = rest.len();
    let v_rr = Mat::from_fn(nr, nr, |i, j| cov[rest[i]][rest[j]]);
    let v_inv = inv_spd_scaled(
        v_rr.as_ref(),
        "the covariance block of the bounds-test restrictions (collinear level regressors)",
    )?;
    let rb: Vec<f64> = rest.iter().map(|&i| b[i]).collect();
    let mut wald = 0.0;
    for i in 0..nr {
        for j in 0..nr {
            wald += rb[i] * v_inv[(i, j)] * rb[j];
        }
    }
    let statistic = wald / nr as f64;
    let t_statistic = tvalues[n_det];
    let adjustment = b[n_det];

    // long run: θ_i = -b_i / π, delta-method covariance
    let pi = adjustment;
    let lr_idx: Vec<usize> = (0..n_det).chain(n_det + 1..n_det + 1 + kx).collect();
    let long_run: Vec<f64> = lr_idx.iter().map(|&i| -b[i] / pi).collect();
    let c_pp = cov[n_det][n_det];
    let long_run_se: Vec<f64> = lr_idx
        .iter()
        .map(|&i| {
            let v = cov[i][i] / (pi * pi) - 2.0 * b[i] * cov[i][n_det] / (pi * pi * pi)
                + b[i] * b[i] * c_pp / (pi * pi * pi * pi);
            v.max(0.0).sqrt()
        })
        .collect();
    let long_run_tvalues: Vec<f64> = long_run
        .iter()
        .zip(&long_run_se)
        .map(|(v, s)| v / s)
        .collect();
    let long_run_pvalues: Vec<f64> = long_run_tvalues
        .iter()
        .map(|t| {
            if t.is_nan() {
                f64::NAN
            } else {
                2.0 * StdNormal.sf(t.abs())
            }
        })
        .collect();
    let mut long_run_names: Vec<String> = ["const", "trend"][..n_det]
        .iter()
        .map(|s| (*s).to_string())
        .collect();
    long_run_names.extend((1..=kx).map(|j| format!("x{j}")));

    // critical values and p-values at the corrected key k = n_x
    let ci = case.index();
    let (crit_lower, crit_upper, p_value_lower, p_value_upper, crit_note) = match (
        pss::crit_values(kx, ci, false),
        pss::crit_values(kx, ci, true),
        pss::pvalue(statistic, kx, ci, false),
        pss::pvalue(statistic, kx, ci, true),
    ) {
        (Some(lo), Some(up), Some(pl), Some(pu)) => (lo, up, pl, pu, None),
        _ => (
            [f64::NAN; 4],
            [f64::NAN; 4],
            f64::NAN,
            f64::NAN,
            Some(format!(
                "no tabulated bounds for n_x = {kx} regressors: the PSS response surface \
                 (statsmodels' 32-million-draw simulation) stops at 10, so crit_lower, \
                 crit_upper, p_value_lower and p_value_upper are NaN and the verdict is \
                 \"unavailable\"; the statistic, estimates and long run are still valid"
            )),
        ),
    };
    let verdict = read_verdict(p_value_lower, p_value_upper, opts.alpha);

    Ok(ArdlResult {
        statistic,
        t_statistic,
        case,
        n_x: kx,
        n_restrictions: nr,
        crit_levels: pss::CRIT_LEVELS,
        crit_lower,
        crit_upper,
        p_value_lower,
        p_value_upper,
        verdict,
        alpha: opts.alpha,
        crit_note,
        params: b,
        se,
        tvalues,
        pvalues,
        cov,
        param_names: param_names(n_det, kx, p, &q),
        adjustment,
        n_det,
        n_params: m,
        long_run,
        long_run_se,
        long_run_tvalues,
        long_run_pvalues,
        long_run_names,
        resid: fit.resid,
        nobs,
        n_total: n,
        hold_back: h,
        df_resid,
        ssr: fit.ssr,
        llf,
        aic,
        bic,
        hqic,
        lags: p,
        order: q,
        selected: !both_fixed,
        ic: opts.ic,
        ic_value,
        max_lag: spec.p_cap,
        max_order: spec.q_cap.iter().copied().max().unwrap_or(1),
        n_models,
    })
}
