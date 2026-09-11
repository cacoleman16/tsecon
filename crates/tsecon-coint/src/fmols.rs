//! Single-equation cointegrating regressions with asymptotically valid
//! inference: the Phillips-Hansen (1990) **fully modified OLS** estimator
//! ([`fmols`]) and Park's (1992) **canonical cointegrating regression**
//! ([`ccr`]), plus the shared engine — deterministic designs, the kernel
//! long-run covariance of the residual system, and the automatic bandwidth
//! rules — that the Stock-Watson (1993) dynamic OLS estimator in
//! [`crate::dols`] reuses.
//!
//! # The problem these estimators solve
//!
//! In the static cointegrating regression `y_t = x_t' beta + d_t' delta +
//! e_t` with `x_t` a vector of I(1) regressors, OLS is super-consistent
//! (`T (beta_hat - beta)` is bounded in probability) but its limit
//! distribution is not centred and not Gaussian: serial correlation in
//! `e_t` and correlation between `e_t` and the regressor innovations
//! `Delta x_t` leave a second-order bias and a nuisance-parameter-laden
//! limit, so plain OLS t-statistics are invalid. FM-OLS, CCR and DOLS are
//! three ways of removing the nuisance terms so that the corrected
//! estimator is asymptotically mixed normal and the reported t-statistics
//! are standard normal under the null.
//!
//! # Conventions (all reproduce `arch.unitroot.cointegration`)
//!
//! * `trend` adds the deterministics `"n"` (none), `"c"` (constant),
//!   `"ct"` (constant and linear trend) or `"ctt"` (constant, linear and
//!   quadratic trend); the trend runs `1, 2, ..., n` over the regression
//!   sample. Regressors are ordered **`x` columns first, then the
//!   deterministics** (`const`, `trend`, `quadratic_trend`), as `arch`
//!   orders them.
//! * The residual system `eta_t = (eta_1t, eta_2t')'` is the static OLS
//!   residual `eta_1t` paired with the detrended regressor innovations
//!   `eta_2t` (the first differences of `x_t` after projecting the levels
//!   on the `x_trend` deterministics, or — with `diff = true` — of the
//!   differences on the differenced trend), on the sample `t = 2..n`.
//! * Its long-run covariance uses the kernel weights of
//!   [`tsecon_hac::Kernel`] in `arch`'s lag-window convention: Bartlett
//!   and Parzen weight lag `j` by `k(j / (bandwidth + 1))` **for `j <=
//!   floor(bandwidth)` only** (the window stops at the integer part of the
//!   bandwidth even when the weight there is still positive), quadratic
//!   spectral by `k(j / bandwidth)` at every lag. With an integer
//!   bandwidth the Bartlett/Parzen window coincides with
//!   [`tsecon_hac::lrv`]'s; `fmols`/`ccr` default to `force_int = true`
//!   (`arch`'s default) so the two agree unless a non-integer bandwidth is
//!   requested explicitly with `force_int = false`.
//! * The automatic bandwidth ([`BandwidthRule`]) is chosen on the
//!   unit-weighted column sum of the residual system, as `arch` does:
//!   [`BandwidthRule::NeweyWest`] is `arch`'s own rule (Newey-West 1994
//!   plug-in with `ceil(4 (T/100)^rate)` pilot lags — `ceil`, where
//!   [`tsecon_hac::newey_west_bandwidth`] floors), and
//!   [`BandwidthRule::Andrews`] is the Andrews (1991) AR(1) parametric
//!   plug-in of [`tsecon_hac::andrews_bandwidth_ar1`]. Both return the
//!   Andrews scale `S_T`, used **directly** as the `bandwidth` (so the
//!   Bartlett weights are `1 - j/(S_T + 1)`, `arch`'s reading);
//!   [`tsecon_hac::Kernel::bandwidth_from_scale`] reads `S_T` one lag
//!   tighter — pass an explicit `bandwidth` to pick either reading.
//!   `force_int` ceils the bandwidth (automatic **or** explicit, as `arch`
//!   applies it) and the automatic one is capped at `T - 1`.
//! * `df_adjust` multiplies the parameter covariance by `T/(T - k)` with
//!   `T` the regression sample and `k` its regressors — for CCR this is
//!   the documented scaling of the *conditional* long-run variance
//!   `omega_{1.2}`; `arch` 8.0's `CanonicalCointegratingReg.fit` scales
//!   only `omega_11` by an operator-precedence slip, and the fixture
//!   pins the documented version (see `fixtures/generate_fmols_fixtures.py`).
//!
//! References: Phillips & Hansen (1990), Review of Economic Studies 57(1);
//! Hansen & Phillips (1990), Advances in Econometrics 8; Park (1992),
//! Econometrica 60(1); Andrews (1991), Econometrica 59(3); Newey & West
//! (1994), Review of Economic Studies 61(4).

use tsecon_hac::{andrews_bandwidth_ar1, Kernel};
use tsecon_linalg::faer::linalg::solvers::SolveLstsq;
use tsecon_linalg::faer::{Mat, MatRef};
use tsecon_stats::{ContinuousDist, StdNormal};

use crate::error::CointError;
use crate::linalg::{check_finite, inv_spd};

// ------------------------------------------------------------ trends

/// Deterministic terms of a cointegrating regression (`arch`'s `trend`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CointTrend {
    /// No deterministic terms (`"n"`).
    None,
    /// A constant (`"c"`, the default).
    Constant,
    /// A constant and a linear trend `t = 1..n` (`"ct"`).
    ConstantTrend,
    /// A constant, a linear and a quadratic trend (`"ctt"`).
    ConstantQuadratic,
}

impl CointTrend {
    /// Parses the `arch` spelling (`"n"`, `"c"`, `"ct"`, `"ctt"`).
    pub fn parse(code: &str) -> Option<Self> {
        match code {
            "n" => Some(Self::None),
            "c" => Some(Self::Constant),
            "ct" => Some(Self::ConstantTrend),
            "ctt" => Some(Self::ConstantQuadratic),
            _ => None,
        }
    }

    /// The `arch` spelling.
    #[must_use]
    pub fn code(self) -> &'static str {
        match self {
            Self::None => "n",
            Self::Constant => "c",
            Self::ConstantTrend => "ct",
            Self::ConstantQuadratic => "ctt",
        }
    }

    /// Number of deterministic columns (0, 1, 2 or 3).
    #[must_use]
    pub fn n_det(self) -> usize {
        match self {
            Self::None => 0,
            Self::Constant => 1,
            Self::ConstantTrend => 2,
            Self::ConstantQuadratic => 3,
        }
    }

    /// Whether the design carries an intercept (decides whether `R^2` is
    /// centred).
    #[must_use]
    pub fn has_constant(self) -> bool {
        self.n_det() >= 1
    }

    /// Names of the deterministic columns, in design order.
    #[must_use]
    pub fn det_names(self) -> &'static [&'static str] {
        const NAMES: [&str; 3] = ["const", "trend", "quadratic_trend"];
        &NAMES[..self.n_det()]
    }
}

/// The deterministic columns `[1], [t], [t^2]` with `t = 1..=n`
/// (`arch.utility.timeseries.add_trend`, `np.vander` flipped to
/// increasing powers), as an `n x n_det` matrix.
pub(crate) fn det_columns(trend: CointTrend, n: usize) -> Mat<f64> {
    let k = trend.n_det();
    Mat::from_fn(n, k, |i, j| {
        let t = (i + 1) as f64;
        match j {
            0 => 1.0,
            1 => t,
            _ => t * t,
        }
    })
}

/// Horizontal concatenation `[a, b]`.
pub(crate) fn hcat(a: MatRef<'_, f64>, b: MatRef<'_, f64>) -> Mat<f64> {
    let ka = a.ncols();
    Mat::from_fn(a.nrows(), ka + b.ncols(), |i, j| {
        if j < ka {
            a[(i, j)]
        } else {
            b[(i, j - ka)]
        }
    })
}

/// Rows `from..` of a matrix.
pub(crate) fn rows_from(m: MatRef<'_, f64>, from: usize) -> Mat<f64> {
    Mat::from_fn(m.nrows() - from, m.ncols(), |i, j| m[(i + from, j)])
}

/// First differences, `m[1..] - m[..n-1]` (`(n - 1) x k`).
pub(crate) fn diff_rows(m: MatRef<'_, f64>) -> Mat<f64> {
    Mat::from_fn(m.nrows().saturating_sub(1), m.ncols(), |i, j| {
        m[(i + 1, j)] - m[(i, j)]
    })
}

/// Power-of-two column scales `2^-round(log2 ||col||)` — an exact
/// (rounding-free) equilibration that keeps the trend polynomials from
/// swamping the conditioning of `Z'Z`.
fn column_scales(m: MatRef<'_, f64>) -> Vec<f64> {
    (0..m.ncols())
        .map(|j| {
            let norm = (0..m.nrows())
                .map(|i| m[(i, j)] * m[(i, j)])
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

/// Least-squares coefficients of the multivariate regression of `y` on
/// `x` by Householder QR on the column-equilibrated design. Errors with
/// [`CointError::Singular`] (named by `what`) when the design is collinear.
pub(crate) fn lstsq(
    x: MatRef<'_, f64>,
    y: MatRef<'_, f64>,
    what: &'static str,
) -> Result<Mat<f64>, CointError> {
    if x.ncols() == 0 {
        return Ok(Mat::zeros(0, y.ncols()));
    }
    let s = column_scales(x);
    let xs = Mat::from_fn(x.nrows(), x.ncols(), |i, j| x[(i, j)] * s[j]);
    let bs = xs.qr().solve_lstsq(y);
    let b = Mat::from_fn(bs.nrows(), bs.ncols(), |i, j| bs[(i, j)] * s[i]);
    check_finite(b.as_ref(), what).map_err(|_| CointError::Singular { what })?;
    Ok(b)
}

/// Inverse of the symmetric positive definite cross-product `Z'Z` via
/// Cholesky on the column-equilibrated matrix.
pub(crate) fn inv_spd_scaled(
    m: MatRef<'_, f64>,
    what: &'static str,
) -> Result<Mat<f64>, CointError> {
    let k = m.nrows();
    let s: Vec<f64> = (0..k)
        .map(|j| {
            let d = m[(j, j)];
            if d > 0.0 && d.is_finite() {
                2f64.powi(-((0.5 * d.log2()).round() as i32))
            } else {
                1.0
            }
        })
        .collect();
    let ms = Mat::from_fn(k, k, |i, j| m[(i, j)] * s[i] * s[j]);
    let inv_s = inv_spd(ms.as_ref(), what)?;
    Ok(Mat::from_fn(k, k, |i, j| inv_s[(i, j)] * s[i] * s[j]))
}

// ------------------------------------------------- long-run covariance

/// Automatic bandwidth rule for the kernel long-run covariance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BandwidthRule {
    /// `arch`'s automatic bandwidth: the Newey-West (1994) nonparametric
    /// plug-in `S_T = c (alpha_hat(q) T)^(1/(2q+1))` with `alpha_hat(q) =
    /// (s^(q)/s^(0))^2` from the first `ceil(4 (T/100)^rate)`
    /// autocovariances of the unit-weighted column sum of the residual
    /// system (`rate` = 2/9 Bartlett, 4/25 Parzen, 2/25 quadratic
    /// spectral; `c`, `q` from [`Kernel::andrews_constant`] /
    /// [`Kernel::andrews_q`]). The default.
    NeweyWest,
    /// The Andrews (1991) AR(1) parametric plug-in
    /// ([`tsecon_hac::andrews_bandwidth_ar1`]) on the same unit-weighted
    /// column sum.
    Andrews,
}

impl BandwidthRule {
    /// Parses `"newey-west"` / `"andrews"` (case-insensitive, `-`/`_`
    /// interchangeable).
    pub fn parse(code: &str) -> Option<Self> {
        match code.to_ascii_lowercase().replace(['-', '_'], "").as_str() {
            "neweywest" | "nw" => Some(Self::NeweyWest),
            "andrews" => Some(Self::Andrews),
            _ => None,
        }
    }

    /// The canonical spelling.
    #[must_use]
    pub fn code(self) -> &'static str {
        match self {
            Self::NeweyWest => "newey-west",
            Self::Andrews => "andrews",
        }
    }
}

/// Parses the kernel names accepted by the cointegrating regressions:
/// `"bartlett"`, `"parzen"`, `"quadratic-spectral"` (case-insensitive,
/// `-`/`_`/spaces ignored; `"qs"`, and `arch`'s aliases `"newey-west"`
/// for Bartlett, `"gallant"` for Parzen and `"andrews"` for quadratic
/// spectral are accepted too).
pub fn parse_kernel(code: &str) -> Option<Kernel> {
    match code
        .to_ascii_lowercase()
        .replace(['-', '_', ' '], "")
        .as_str()
    {
        "bartlett" | "neweywest" => Some(Kernel::Bartlett),
        "parzen" | "gallant" => Some(Kernel::Parzen),
        "quadraticspectral" | "qs" | "andrews" => Some(Kernel::QuadraticSpectral),
        _ => None,
    }
}

/// The canonical spelling of a kernel for result echoes.
#[must_use]
pub fn kernel_code(kernel: Kernel) -> &'static str {
    match kernel {
        Kernel::Bartlett => "bartlett",
        Kernel::Parzen => "parzen",
        Kernel::QuadraticSpectral => "quadratic-spectral",
        Kernel::Truncated => "truncated",
    }
}

/// The three long-run covariance pieces of a `k`-variate mean-zero
/// series, in `arch.covariance.kernel` notation.
#[derive(Debug, Clone, PartialEq)]
pub struct LongRunCovariance {
    /// Dimension `k`.
    pub k: usize,
    /// Observations `T` the sums were divided by.
    pub nobs: usize,
    /// Short-run covariance `Sigma = Gamma_0 = eta' eta / T` (`k x k`,
    /// rows).
    pub sigma: Vec<Vec<f64>>,
    /// One-sided long-run covariance `Lambda = sum_{h >= 0} w_h Gamma_h`
    /// with `Gamma_h = sum_t eta_t eta_{t-h}' / T` (`arch`'s
    /// `one_sided`).
    pub lambda: Vec<Vec<f64>>,
    /// Two-sided long-run covariance `Omega = Gamma_0 + sum_{h >= 1} w_h
    /// (Gamma_h + Gamma_h')` (`arch`'s `long_run`).
    pub omega: Vec<Vec<f64>>,
    /// The bandwidth the weights were evaluated at.
    pub bandwidth: f64,
    /// Number of positive lags the window covered (`floor(bandwidth)` for
    /// Bartlett/Parzen, `T - 1` for quadratic spectral, both capped at
    /// `T - 1`).
    pub n_lags: usize,
}

impl LongRunCovariance {
    fn to_mat(m: &[Vec<f64>]) -> Mat<f64> {
        let k = m.len();
        Mat::from_fn(k, k, |i, j| m[i][j])
    }

    /// `sigma` as a matrix.
    pub(crate) fn sigma_mat(&self) -> Mat<f64> {
        Self::to_mat(&self.sigma)
    }

    /// `lambda` as a matrix.
    pub(crate) fn lambda_mat(&self) -> Mat<f64> {
        Self::to_mat(&self.lambda)
    }

    /// `omega` as a matrix.
    pub(crate) fn omega_mat(&self) -> Mat<f64> {
        Self::to_mat(&self.omega)
    }
}

/// Number of positive lags the window covers (see the module docs).
fn window_lags(kernel: Kernel, bandwidth: f64, nobs: usize) -> usize {
    let cap = nobs.saturating_sub(1);
    match kernel {
        Kernel::QuadraticSpectral => cap,
        // `int(bandwidth + 1) - 1` for a non-negative float bandwidth.
        _ => (bandwidth.floor() as usize).min(cap),
    }
}

/// Rejects the truncated kernel (not positive semi-definite; no plug-in
/// bandwidth rule) — the cointegrating regressions need a PSD long-run
/// covariance.
fn check_kernel(kernel: Kernel) -> Result<(), CointError> {
    if kernel == Kernel::Truncated {
        return Err(CointError::InvalidSpec {
            what: "kernel = \"truncated\" is not admissible for a cointegrating regression: \
                   its lag window is not positive semi-definite, so the long-run covariance \
                   it produces can fail to be a covariance, and it has no plug-in bandwidth \
                   rule; pass \"bartlett\", \"parzen\" or \"quadratic-spectral\""
                .to_string(),
        });
    }
    Ok(())
}

/// Kernel long-run covariance of a `T x k` mean-zero series at an
/// explicit bandwidth, in `arch.covariance.kernel` conventions (biased
/// `1/T` autocovariances about zero, lag window per the module docs).
///
/// # Errors
///
/// [`CointError::InvalidSpec`] for a negative or non-finite `bandwidth`
/// or the truncated kernel; [`CointError::Dimension`] with fewer than two
/// rows or no columns; [`CointError::NonFinite`] on NaN/inf input.
pub fn long_run_covariance(
    eta: MatRef<'_, f64>,
    kernel: Kernel,
    bandwidth: f64,
) -> Result<LongRunCovariance, CointError> {
    check_kernel(kernel)?;
    if !bandwidth.is_finite() || bandwidth < 0.0 {
        return Err(CointError::InvalidSpec {
            what: format!(
                "bandwidth = {bandwidth} is invalid: the kernel bandwidth must be a finite \
                 number >= 0 (0 keeps only the contemporaneous covariance; pass None to \
                 select it automatically)"
            ),
        });
    }
    let n = eta.nrows();
    let k = eta.ncols();
    if k == 0 {
        return Err(CointError::Dimension {
            what: "the residual system has no columns",
            expected: 1,
            got: 0,
        });
    }
    if n < 2 {
        return Err(CointError::Dimension {
            what: "the kernel long-run covariance needs at least two observations — rows",
            expected: 2,
            got: n,
        });
    }
    check_finite(eta, "the residual system of the long-run covariance")?;

    let nf = n as f64;
    let n_lags = window_lags(kernel, bandwidth, n);
    // Gamma_0 and the weighted one-sided strict sum, both divided by T.
    let mut sigma = vec![vec![0.0_f64; k]; k];
    for t in 0..n {
        for i in 0..k {
            let ei = eta[(t, i)];
            for j in 0..k {
                sigma[i][j] += ei * eta[(t, j)];
            }
        }
    }
    for row in &mut sigma {
        for v in row.iter_mut() {
            *v /= nf;
        }
    }
    let mut oss = vec![vec![0.0_f64; k]; k];
    for h in 1..=n_lags {
        let w = kernel.weight(h, bandwidth);
        if w == 0.0 {
            continue;
        }
        // Gamma_h[i][j] = sum_{t=h}^{n-1} eta_t[i] eta_{t-h}[j] / T.
        let mut g = vec![vec![0.0_f64; k]; k];
        for t in h..n {
            for i in 0..k {
                let ei = eta[(t, i)];
                for j in 0..k {
                    g[i][j] += ei * eta[(t - h, j)];
                }
            }
        }
        for i in 0..k {
            for j in 0..k {
                oss[i][j] += w * g[i][j] / nf;
            }
        }
    }
    let lambda: Vec<Vec<f64>> = (0..k)
        .map(|i| (0..k).map(|j| sigma[i][j] + oss[i][j]).collect())
        .collect();
    let omega: Vec<Vec<f64>> = (0..k)
        .map(|i| {
            (0..k)
                .map(|j| sigma[i][j] + oss[i][j] + oss[j][i])
                .collect()
        })
        .collect();
    Ok(LongRunCovariance {
        k,
        nobs: n,
        sigma,
        lambda,
        omega,
        bandwidth,
        n_lags,
    })
}

/// Unit-weighted column sum of the residual system — the univariate
/// series both automatic rules are evaluated on.
fn column_sum(eta: MatRef<'_, f64>) -> Vec<f64> {
    (0..eta.nrows())
        .map(|t| (0..eta.ncols()).map(|j| eta[(t, j)]).sum())
        .collect()
}

/// `arch`'s automatic bandwidth (see [`BandwidthRule::NeweyWest`]):
/// returns the raw plug-in scale before the `force_int` / `T - 1`
/// treatment.
fn newey_west_plugin(v: &[f64], kernel: Kernel) -> Result<f64, CointError> {
    let n = v.len();
    let nf = n as f64;
    let rate = match kernel {
        Kernel::Bartlett => 2.0 / 9.0,
        Kernel::Parzen => 4.0 / 25.0,
        _ => 2.0 / 25.0,
    };
    let pilot = (4.0 * (nf / 100.0).powf(rate)).ceil() as usize;
    if pilot >= n {
        return Err(CointError::InvalidSpec {
            what: format!(
                "the automatic bandwidth needs more observations: its pilot window uses \
                 ceil(4 (T/100)^{rate:.4}) = {pilot} lags but the residual system has only \
                 T = {n} rows; supply a longer sample or pass an explicit bandwidth"
            ),
        });
    }
    let q = kernel.andrews_q();
    let mut f0 = 0.0;
    let mut fq = 0.0;
    for j in 0..=pilot {
        let sig: f64 = (j..n).map(|t| v[t] * v[t - j]).sum::<f64>() / nf;
        let scale = if j == 0 { 1.0 } else { 2.0 };
        let jf = j as f64;
        let jq = if q == 1.0 { jf } else { jf * jf };
        f0 += scale * sig;
        fq += scale * jq * sig;
    }
    if f0 == 0.0 || !f0.is_finite() {
        return Err(CointError::InvalidSpec {
            what: "the automatic bandwidth is undefined: the pilot long-run variance of the \
                   residual system is zero (the residuals are identically zero — the \
                   regression fits perfectly, so the series are collinear rather than \
                   cointegrated); pass an explicit bandwidth or drop the redundant series"
                .to_string(),
        });
    }
    let alpha = (fq / f0) * (fq / f0);
    let exponent = 1.0 / (2.0 * q + 1.0);
    Ok(kernel.andrews_constant() * (alpha * nf).powf(exponent))
}

/// Automatic bandwidth of the residual system under `rule`, with
/// `arch`'s finishing: `ceil` when `force_int`, then capped at `T - 1`.
///
/// # Errors
///
/// [`CointError::InvalidSpec`] when the sample is too short for the pilot
/// window or the residuals are identically zero;
/// [`CointError::Hac`] from the Andrews rule (fewer than four rows, a
/// unit AR(1) root, a zero series).
pub fn automatic_bandwidth(
    eta: MatRef<'_, f64>,
    kernel: Kernel,
    rule: BandwidthRule,
    force_int: bool,
) -> Result<f64, CointError> {
    check_kernel(kernel)?;
    check_finite(eta, "the residual system of the automatic bandwidth")?;
    let v = column_sum(eta);
    let raw = match rule {
        BandwidthRule::NeweyWest => newey_west_plugin(&v, kernel)?,
        BandwidthRule::Andrews => andrews_bandwidth_ar1(&v, kernel)?,
    };
    let bw = if force_int { raw.ceil() } else { raw };
    Ok(bw.min(eta.nrows() as f64 - 1.0))
}

/// The bandwidth actually used, given the user's choice: an explicit
/// value (validated, ceiled under `force_int`) or the automatic rule.
/// Returns the bandwidth and the rule that produced it (`None` when it
/// was explicit).
pub(crate) fn resolve_bandwidth(
    eta: MatRef<'_, f64>,
    kernel: Kernel,
    bandwidth: Option<f64>,
    rule: BandwidthRule,
    force_int: bool,
) -> Result<(f64, Option<BandwidthRule>), CointError> {
    match bandwidth {
        Some(b) => {
            if !b.is_finite() || b < 0.0 {
                return Err(CointError::InvalidSpec {
                    what: format!(
                        "bandwidth = {b} is invalid: the kernel bandwidth must be a finite \
                         number >= 0 (0 keeps only the contemporaneous covariance; pass None \
                         to select it automatically)"
                    ),
                });
            }
            Ok((if force_int { b.ceil() } else { b }, None))
        }
        None => Ok((
            automatic_bandwidth(eta, kernel, rule, force_int)?,
            Some(rule),
        )),
    }
}

// --------------------------------------------------------- validation

/// Checks `y` / `x` shapes and finiteness; returns `(n, k_x)`.
pub(crate) fn check_yx(y: &[f64], x: MatRef<'_, f64>) -> Result<(usize, usize), CointError> {
    let n = y.len();
    let kx = x.ncols();
    if kx == 0 {
        return Err(CointError::Dimension {
            what: "x must hold at least one regressor column (shape (T, k) with k >= 1)",
            expected: 1,
            got: 0,
        });
    }
    if x.nrows() != n {
        return Err(CointError::Dimension {
            what: "x and y must cover the same observations: x has a different number of \
                   rows than y has entries (x rows",
            expected: n,
            got: x.nrows(),
        });
    }
    for (i, &v) in y.iter().enumerate() {
        if !v.is_finite() {
            return Err(CointError::NonFiniteSeries {
                what: "y; the cointegrating regressions have no missing-value handling — \
                       drop or impute the gap first",
                index: i,
            });
        }
    }
    check_finite(x, "x")?;
    Ok((n, kx))
}

/// Plain OLS of `y` on `z` with classical standard errors — the
/// comparison estimate every corrected estimator reports alongside.
pub(crate) struct StaticOls {
    pub params: Vec<f64>,
    pub se: Vec<f64>,
    pub resid: Vec<f64>,
}

pub(crate) fn static_ols(y: &[f64], z: MatRef<'_, f64>) -> Result<StaticOls, CointError> {
    let n = z.nrows();
    let k = z.ncols();
    let ym = Mat::from_fn(n, 1, |i, _| y[i]);
    let b = lstsq(
        z,
        ym.as_ref(),
        "the static cointegrating regression design is collinear (a duplicated or constant \
         regressor column, or a constant column beside trend = \"c\"); drop the redundant \
         column",
    )?;
    let fitted = z * &b;
    let resid: Vec<f64> = (0..n).map(|i| y[i] - fitted[(i, 0)]).collect();
    let params: Vec<f64> = (0..k).map(|j| b[(j, 0)]).collect();
    let se = if n > k {
        let zpz = z.transpose() * z;
        let inv = inv_spd_scaled(
            zpz.as_ref(),
            "Z'Z of the static cointegrating regression (collinear design)",
        )?;
        let s2 = resid.iter().map(|e| e * e).sum::<f64>() / (n - k) as f64;
        (0..k).map(|j| (s2 * inv[(j, j)]).sqrt()).collect()
    } else {
        vec![f64::NAN; k]
    };
    Ok(StaticOls { params, se, resid })
}

/// Names of the regressors: `x1..xk` then the deterministics.
pub(crate) fn param_names(kx: usize, trend: CointTrend) -> Vec<String> {
    let mut names: Vec<String> = (1..=kx).map(|j| format!("x{j}")).collect();
    names.extend(trend.det_names().iter().map(|s| s.to_string()));
    names
}

/// `R^2` and adjusted `R^2` as `arch` computes them for the corrected
/// estimators: centred when the design has a constant, `1 - (ssr/(n -
/// nvar)) / (tss/(n - tss_df))` for the adjusted version.
pub(crate) fn r_squared(y: &[f64], resid: &[f64], nvar: usize, has_constant: bool) -> (f64, f64) {
    let n = y.len() as f64;
    let ssr: f64 = resid.iter().map(|e| e * e).sum();
    let (tss, tss_df) = if has_constant {
        let mean = y.iter().sum::<f64>() / n;
        (y.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>(), 1.0)
    } else {
        (y.iter().map(|v| v * v).sum::<f64>(), 0.0)
    };
    let r2 = 1.0 - ssr / tss;
    let r2_adj = 1.0 - (ssr / (n - nvar as f64)) / (tss / (n - tss_df));
    (r2, r2_adj)
}

/// Standard errors, t-statistics and two-sided normal p-values from a
/// parameter covariance.
pub(crate) fn inference(params: &[f64], cov: &[Vec<f64>]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let se: Vec<f64> = (0..params.len())
        .map(|j| cov[j][j].max(0.0).sqrt())
        .collect();
    let t: Vec<f64> = params.iter().zip(&se).map(|(b, s)| b / s).collect();
    let p: Vec<f64> = t.iter().map(|v| 2.0 * StdNormal.sf(v.abs())).collect();
    (se, t, p)
}

// ------------------------------------------------------- FM-OLS / CCR

/// Options of [`fmols`] and [`ccr`] (defaults are `arch`'s).
#[derive(Debug, Clone, PartialEq)]
pub struct CointRegOptions {
    /// Deterministic terms of the cointegrating regression. Default
    /// [`CointTrend::Constant`].
    pub trend: CointTrend,
    /// Deterministic terms the regressors are detrended with before their
    /// innovations enter the long-run covariance (`arch`'s `x_trend`);
    /// must carry at least the terms of `trend`. `None` (default) uses
    /// `trend`.
    pub x_trend: Option<CointTrend>,
    /// Kernel of the long-run covariance. Default [`Kernel::Bartlett`].
    pub kernel: Kernel,
    /// Explicit bandwidth, or `None` (default) for the automatic rule.
    pub bandwidth: Option<f64>,
    /// The automatic rule used when `bandwidth` is `None`. Default
    /// [`BandwidthRule::NeweyWest`].
    pub bandwidth_rule: BandwidthRule,
    /// Ceil the bandwidth to an integer (automatic or explicit). Default
    /// `true` (`arch`'s default for FM-OLS / CCR).
    pub force_int: bool,
    /// Detrend the regressor *differences* on the differenced trend terms
    /// instead of differencing the detrended levels (`arch`'s `diff`);
    /// acts only when `x_trend` carries a trend (`"ct"` / `"ctt"`).
    /// Default `false`.
    pub diff: bool,
    /// Scale the parameter covariance by `T/(T - k)`. Default `false`.
    pub df_adjust: bool,
}

impl Default for CointRegOptions {
    fn default() -> Self {
        Self {
            trend: CointTrend::Constant,
            x_trend: None,
            kernel: Kernel::Bartlett,
            bandwidth: None,
            bandwidth_rule: BandwidthRule::NeweyWest,
            force_int: true,
            diff: false,
            df_adjust: false,
        }
    }
}

/// Which corrected estimator produced a [`CointRegResult`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CointRegEstimator {
    /// Phillips-Hansen fully modified OLS.
    Fmols,
    /// Park canonical cointegrating regression.
    Ccr,
}

impl CointRegEstimator {
    /// Human-readable name.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Fmols => "Fully Modified OLS",
            Self::Ccr => "Canonical Cointegrating Regression",
        }
    }
}

/// Result of [`fmols`] / [`ccr`].
#[derive(Debug, Clone, PartialEq)]
pub struct CointRegResult {
    /// The estimator.
    pub estimator: CointRegEstimator,
    /// Coefficients in design order: the `k_x` regressors, then the
    /// deterministics of `trend`.
    pub params: Vec<f64>,
    /// Asymptotic standard errors `sqrt(diag(cov))`.
    pub se: Vec<f64>,
    /// `params / se` — standard normal under the null (asymptotically).
    pub tvalues: Vec<f64>,
    /// Two-sided normal p-values of `tvalues`.
    pub pvalues: Vec<f64>,
    /// Parameter covariance `omega_{1.2} (Z'Z)^{-1}` (rows).
    pub cov: Vec<Vec<f64>>,
    /// Names of the coefficients (`x1`, ..., `const`, `trend`,
    /// `quadratic_trend`).
    pub param_names: Vec<String>,
    /// Residuals `y_t - z_t' theta` over the full sample (length `T`).
    pub resid: Vec<f64>,
    /// Sample size `T` of the data.
    pub nobs: usize,
    /// Number of stochastic regressors `k_x`.
    pub n_x: usize,
    /// Number of deterministic columns.
    pub n_det: usize,
    /// The deterministic specification.
    pub trend: CointTrend,
    /// The deterministics the regressors were detrended with.
    pub x_trend: CointTrend,
    /// The kernel used.
    pub kernel: Kernel,
    /// The bandwidth actually used (after `force_int` and the `T - 1`
    /// cap).
    pub bandwidth: f64,
    /// The automatic rule that produced `bandwidth`, or `None` when it
    /// was passed explicitly.
    pub bandwidth_rule: Option<BandwidthRule>,
    /// Echo of `force_int`.
    pub force_int: bool,
    /// Echo of `diff`.
    pub diff: bool,
    /// Echo of `df_adjust`.
    pub df_adjust: bool,
    /// The conditional long-run variance `omega_{1.2} = omega_11 -
    /// omega_12 Omega_22^{-1} omega_21` (times `T/(T - k)` under
    /// `df_adjust`) that scales the covariance.
    pub long_run_variance: f64,
    /// The long-run covariance pieces of the residual system `eta`
    /// (`(1 + k_x)`-variate: static residual first, then the regressor
    /// innovations) the correction was built from.
    pub lrcov: LongRunCovariance,
    /// `R^2` of the full-sample residual (centred when `trend` has a
    /// constant).
    pub rsquared: f64,
    /// Adjusted `R^2`.
    pub rsquared_adj: f64,
    /// Plain static OLS coefficients (same order) for comparison.
    pub ols_params: Vec<f64>,
    /// Classical OLS standard errors — reported for the comparison only;
    /// they are **not** valid for inference on a cointegrating vector.
    pub ols_se: Vec<f64>,
}

/// What [`fmols`] and [`ccr`] share (`arch`'s `_common_fit`).
struct CommonFit {
    n: usize,
    kx: usize,
    x_trend: CointTrend,
    ols: StaticOls,
    /// `(n - 1) x (1 + k_x)` residual system.
    eta: Mat<f64>,
    lrcov: LongRunCovariance,
    bandwidth_rule: Option<BandwidthRule>,
}

fn common_fit(
    y: &[f64],
    x: MatRef<'_, f64>,
    opts: &CointRegOptions,
) -> Result<CommonFit, CointError> {
    check_kernel(opts.kernel)?;
    let (n, kx) = check_yx(y, x)?;
    let trend = opts.trend;
    let x_trend = opts.x_trend.unwrap_or(trend);
    if x_trend.n_det() < trend.n_det() {
        return Err(CointError::InvalidSpec {
            what: format!(
                "x_trend = \"{}\" carries fewer deterministic terms than trend = \"{}\": \
                 the regressors must be detrended with at least the terms of the \
                 cointegrating regression, so pass x_trend = \"{}\" or larger (or None \
                 to use trend)",
                x_trend.code(),
                trend.code(),
                trend.code()
            ),
        });
    }
    let nvar = kx + trend.n_det();
    // The static regression needs n > nvar rows; the residual system has
    // n - 1 rows and must leave df_adjust's n - 1 - nvar positive and the
    // automatic rules at least four rows.
    let needed = (nvar + 2).max(5);
    if n < needed {
        return Err(CointError::InvalidSpec {
            what: format!(
                "y has only T = {n} observations but the cointegrating regression with \
                 k_x = {kx} regressor(s) and trend = \"{}\" ({} deterministic column(s)) \
                 needs at least {needed}: the residual system loses one row to \
                 differencing and must keep more rows than the {nvar} regressors; supply \
                 a longer sample, fewer regressors or a smaller trend",
                trend.code(),
                trend.n_det()
            ),
        });
    }

    // Static OLS of y on [x, det(trend)].
    let det = det_columns(trend, n);
    let z = hcat(x, det.as_ref());
    let ols = static_ols(y, z.as_ref())?;

    // Regressor innovations eta_2.
    let tr = det_columns(x_trend, n);
    let eta_2 = if tr.ncols() > 1 && opts.diff {
        let tr_slopes = Mat::from_fn(n, tr.ncols() - 1, |i, j| tr[(i, j + 1)]);
        let delta_tr = diff_rows(tr_slopes.as_ref());
        let delta_x = diff_rows(x);
        let gamma = lstsq(
            delta_tr.as_ref(),
            delta_x.as_ref(),
            "the differenced trend design is collinear",
        )?;
        &delta_x - &delta_tr * &gamma
    } else {
        let eps = if tr.ncols() > 0 {
            let gamma = lstsq(tr.as_ref(), x, "the x_trend design is collinear")?;
            x - &tr * &gamma
        } else {
            x.to_owned()
        };
        diff_rows(eps.as_ref())
    };
    let eta = Mat::from_fn(n - 1, 1 + kx, |i, j| {
        if j == 0 {
            ols.resid[i + 1]
        } else {
            eta_2[(i, j - 1)]
        }
    });
    let (bw, rule) = resolve_bandwidth(
        eta.as_ref(),
        opts.kernel,
        opts.bandwidth,
        opts.bandwidth_rule,
        opts.force_int,
    )?;
    let lrcov = long_run_covariance(eta.as_ref(), opts.kernel, bw)?;
    Ok(CommonFit {
        n,
        kx,
        x_trend,
        ols,
        eta,
        lrcov,
        bandwidth_rule: rule,
    })
}

/// The partitioned pieces `omega_12` (`1 x k_x`), `Omega_22^{-1}` and
/// `omega_{1.2}` of the long-run covariance.
struct OmegaBlocks {
    omega_12: Mat<f64>,
    omega_22_inv: Mat<f64>,
    omega_11_2: f64,
}

fn omega_blocks(lrcov: &LongRunCovariance) -> Result<OmegaBlocks, CointError> {
    let kx = lrcov.k - 1;
    let omega = lrcov.omega_mat();
    let omega_12 = Mat::from_fn(1, kx, |_, j| omega[(0, j + 1)]);
    let omega_22 = Mat::from_fn(kx, kx, |i, j| omega[(i + 1, j + 1)]);
    let omega_22_inv = inv_spd(
        omega_22.as_ref(),
        "Omega_22, the long-run covariance of the regressor innovations; two regressors \
         share the same stochastic trend (a duplicated or linearly dependent column) or \
         the bandwidth is too large for the sample — drop the redundant regressor or \
         pass a smaller bandwidth",
    )?;
    let cond = &omega_12 * &omega_22_inv * omega_12.transpose();
    Ok(OmegaBlocks {
        omega_12,
        omega_22_inv,
        omega_11_2: omega[(0, 0)] - cond[(0, 0)],
    })
}

#[allow(clippy::too_many_arguments)]
fn finish(
    estimator: CointRegEstimator,
    y: &[f64],
    x: MatRef<'_, f64>,
    opts: &CointRegOptions,
    cf: CommonFit,
    params: Vec<f64>,
    cov: Vec<Vec<f64>>,
    long_run_variance: f64,
) -> CointRegResult {
    let trend = opts.trend;
    let n = cf.n;
    let kx = cf.kx;
    let nvar = params.len();
    let det = det_columns(trend, n);
    let z = hcat(x, det.as_ref());
    let resid: Vec<f64> = (0..n)
        .map(|i| y[i] - (0..nvar).map(|j| z[(i, j)] * params[j]).sum::<f64>())
        .collect();
    let (rsquared, rsquared_adj) = r_squared(y, &resid, nvar, trend.has_constant());
    let (se, tvalues, pvalues) = inference(&params, &cov);
    CointRegResult {
        estimator,
        params,
        se,
        tvalues,
        pvalues,
        cov,
        param_names: param_names(kx, trend),
        resid,
        nobs: n,
        n_x: kx,
        n_det: trend.n_det(),
        trend,
        x_trend: cf.x_trend,
        kernel: opts.kernel,
        bandwidth: cf.lrcov.bandwidth,
        bandwidth_rule: cf.bandwidth_rule,
        force_int: opts.force_int,
        diff: opts.diff,
        df_adjust: opts.df_adjust,
        long_run_variance,
        lrcov: cf.lrcov,
        rsquared,
        rsquared_adj,
        ols_params: cf.ols.params,
        ols_se: cf.ols.se,
    }
}

/// Phillips-Hansen (1990) fully modified OLS of `y` (length `T`) on the
/// `T x k_x` regressors `x` and the deterministics of `opts.trend`.
///
/// With `Omega` / `Lambda` the long-run and one-sided long-run covariance
/// of the residual system `eta_t = (eta_1t, eta_2t')'` (see the module
/// docs), the regressand is corrected for the endogeneity of the
/// regressors,
///
/// ```text
/// y+_t = y_t - omega_12 Omega_22^{-1} eta_2t,
/// ```
///
/// and the estimator removes the serial-correlation bias term
/// `lambda+_12 = lambda_12 - omega_12 Omega_22^{-1} Lambda_22`:
///
/// ```text
/// theta_hat = (Z'Z)^{-1} (Z' y+ - T [lambda+_12', 0]'),   Z = [x, det], t = 2..T,
/// Cov(theta_hat) = omega_{1.2} (Z'Z)^{-1},   omega_{1.2} = omega_11 - omega_12 Omega_22^{-1} omega_21.
/// ```
///
/// Reproduces `arch.unitroot.cointegration.FullyModifiedOLS(y, x, trend,
/// x_trend).fit(kernel, bandwidth, force_int, diff, df_adjust)` to 1e-10
/// on the goldens.
///
/// # Errors
///
/// [`CointError::Dimension`] for an empty `x` or mismatched lengths;
/// [`CointError::NonFinite`] / [`CointError::NonFiniteSeries`] on NaN or
/// infinity; [`CointError::InvalidSpec`] for a bad bandwidth, an
/// `x_trend` smaller than `trend`, the truncated kernel or a sample too
/// short for the specification; [`CointError::Singular`] /
/// [`CointError::NotPositiveDefinite`] for collinear regressors;
/// [`CointError::Hac`] from the Andrews bandwidth rule.
pub fn fmols(
    y: &[f64],
    x: MatRef<'_, f64>,
    opts: &CointRegOptions,
) -> Result<CointRegResult, CointError> {
    let cf = common_fit(y, x, opts)?;
    let n = cf.n;
    let kx = cf.kx;
    let m = n - 1;
    let blocks = omega_blocks(&cf.lrcov)?;
    let lambda = cf.lrcov.lambda_mat();
    let lambda_12 = Mat::from_fn(1, kx, |_, j| lambda[(0, j + 1)]);
    let lambda_22 = Mat::from_fn(kx, kx, |i, j| lambda[(i + 1, j + 1)]);
    // lambda+_12 = lambda_12 - omega_12 Omega_22^{-1} Lambda_22.
    let lambda_12_dot = &lambda_12 - &blocks.omega_12 * &blocks.omega_22_inv * &lambda_22;
    // y+ over t = 2..T.
    let eta_2 = Mat::from_fn(m, kx, |i, j| cf.eta[(i, j + 1)]);
    let adj = &eta_2 * &blocks.omega_22_inv * blocks.omega_12.transpose();
    let y_dot = Mat::from_fn(m, 1, |i, _| y[i + 1] - adj[(i, 0)]);

    let det = det_columns(opts.trend, n);
    let z_full = hcat(x, det.as_ref());
    let z = rows_from(z_full.as_ref(), 1);
    let nvar = z.ncols();
    let zpz = z.transpose() * &z;
    let zpz_inv = inv_spd_scaled(
        zpz.as_ref(),
        "Z'Z of the fully modified regression (collinear regressors)",
    )?;
    let zpy = z.transpose() * &y_dot;
    let rhs = Mat::from_fn(nvar, 1, |j, _| {
        if j < kx {
            zpy[(j, 0)] - m as f64 * lambda_12_dot[(0, j)]
        } else {
            zpy[(j, 0)]
        }
    });
    let theta = &zpz_inv * &rhs;
    let params: Vec<f64> = (0..nvar).map(|j| theta[(j, 0)]).collect();
    let scale = if opts.df_adjust {
        m as f64 / (m - nvar) as f64
    } else {
        1.0
    };
    let omega_112 = scale * blocks.omega_11_2;
    let cov: Vec<Vec<f64>> = (0..nvar)
        .map(|i| (0..nvar).map(|j| omega_112 * zpz_inv[(i, j)]).collect())
        .collect();
    Ok(finish(
        CointRegEstimator::Fmols,
        y,
        x,
        opts,
        cf,
        params,
        cov,
        omega_112,
    ))
}

/// Park (1992) canonical cointegrating regression of `y` on `x` and the
/// deterministics of `opts.trend`.
///
/// With `Sigma`, `Lambda`, `Omega` the short-run, one-sided and two-sided
/// long-run covariances of the residual system and `beta_hat` the static
/// OLS coefficients of `x`, the data are transformed,
///
/// ```text
/// x*_t = x_t - (Sigma^{-1} Lambda_2)' eta_t,
/// y*_t = y_t - (Sigma^{-1} Lambda_2 beta_hat + kappa)' eta_t,   kappa = (0, Omega_22^{-1} omega_21')',
/// ```
///
/// (`Lambda_2` the last `k_x` columns of `Lambda`), and `theta_hat` is the
/// OLS of `y*` on `[x*, det]` over `t = 2..T`, with covariance
/// `omega_{1.2} (Z*'Z*)^{-1}`.
///
/// Reproduces `arch.unitroot.cointegration.CanonicalCointegratingReg` to
/// 1e-10 on the goldens, except that `df_adjust` scales the whole
/// conditional long-run variance as documented (see the module docs).
///
/// # Errors
///
/// As [`fmols`].
pub fn ccr(
    y: &[f64],
    x: MatRef<'_, f64>,
    opts: &CointRegOptions,
) -> Result<CointRegResult, CointError> {
    let cf = common_fit(y, x, opts)?;
    let n = cf.n;
    let kx = cf.kx;
    let m = n - 1;
    let blocks = omega_blocks(&cf.lrcov)?;
    let sigma = cf.lrcov.sigma_mat();
    let lambda = cf.lrcov.lambda_mat();
    let sigma_inv = inv_spd(
        sigma.as_ref(),
        "Sigma, the contemporaneous covariance of the residual system; a regressor is an \
         exact linear combination of the others (or of the regressand) — drop the \
         redundant column",
    )?;
    let lambda_2 = Mat::from_fn(kx + 1, kx, |i, j| lambda[(i, j + 1)]);
    let a = &sigma_inv * &lambda_2; // (1 + k_x) x k_x
    let x_star = rows_from(x, 1) - &cf.eta * &a;
    let beta = Mat::from_fn(kx, 1, |i, _| cf.ols.params[i]);
    let kappa_tail = &blocks.omega_22_inv * blocks.omega_12.transpose(); // k_x x 1
    let ab = &a * &beta; // (1 + k_x) x 1
    let b = Mat::from_fn(kx + 1, 1, |i, _| {
        if i == 0 {
            ab[(0, 0)]
        } else {
            ab[(i, 0)] + kappa_tail[(i - 1, 0)]
        }
    });
    let adj = &cf.eta * &b;
    let y_star = Mat::from_fn(m, 1, |i, _| y[i + 1] - adj[(i, 0)]);
    let det = det_columns(opts.trend, m);
    let z_star = hcat(x_star.as_ref(), det.as_ref());
    let nvar = z_star.ncols();
    let theta = lstsq(
        z_star.as_ref(),
        y_star.as_ref(),
        "the canonical cointegrating regression design is collinear",
    )?;
    let params: Vec<f64> = (0..nvar).map(|j| theta[(j, 0)]).collect();
    let zpz = z_star.transpose() * &z_star;
    let zpz_inv = inv_spd_scaled(
        zpz.as_ref(),
        "Z*'Z* of the canonical cointegrating regression (collinear regressors)",
    )?;
    let scale = if opts.df_adjust {
        m as f64 / (m - nvar) as f64
    } else {
        1.0
    };
    let omega_112 = scale * blocks.omega_11_2;
    let cov: Vec<Vec<f64>> = (0..nvar)
        .map(|i| (0..nvar).map(|j| omega_112 * zpz_inv[(i, j)]).collect())
        .collect();
    Ok(finish(
        CointRegEstimator::Ccr,
        y,
        x,
        opts,
        cf,
        params,
        cov,
        omega_112,
    ))
}
