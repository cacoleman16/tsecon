//! Nadaraya-Watson and local-linear kernel regression with a product
//! Gaussian kernel, matching statsmodels `KernelReg(reg_type="lc" | "ll",
//! var_type="c" * k)` exactly, plus dependence-aware bandwidth selection.
//!
//! # Estimators (statsmodels conventions)
//!
//! For `x` of shape `n x k` (`k <= 3`), bandwidths `h_1..h_k`, and the
//! product Gaussian kernel
//!
//! ```text
//! K_h(x_i - x) = prod_j phi((x_ij - x_j) / h_j) / prod_j h_j ,
//! ```
//!
//! the **Nadaraya-Watson** (local constant, statsmodels `"lc"`) estimator
//! of `E[y | x]` is
//!
//! ```text
//! g(x) = sum_i K_h(x_i - x) y_i / sum_i K_h(x_i - x) ,
//! ```
//!
//! and the **local-linear** (statsmodels `"ll"`) estimator is the
//! intercept of the weighted least-squares fit of `y` on `[1, x_i - x]`
//! with weights `K_h(x_i - x)`:
//!
//! ```text
//! g(x) = e_1' (Z'WZ)^+ Z'W y ,   Z = [1, x_i - x],  W = diag(K_h(x_i - x)),
//! ```
//!
//! solved through the Moore-Penrose pseudoinverse with NumPy's `1e-15`
//! relative cutoff, exactly as statsmodels' `_est_loc_linear` does
//! (`np.linalg.pinv(M)`), so a locally rank-deficient design gives the
//! same minimum-norm answer (Fan & Gijbels 1996, ch. 3; Li & Racine 2007,
//! ch. 2). Local linear is the default: it has no boundary bias and its
//! bias does not depend on the design density (Fan 1992).
//!
//! # Bandwidth selection
//!
//! The leave-one-out least-squares criterion (statsmodels
//! `KernelReg.cv_loo`, Li & Racine 2007 eq. 2.26)
//!
//! ```text
//! CV(h) = n^{-1} sum_i ( y_i - g_{-i}(x_i) )^2
//! ```
//!
//! undersmooths badly when the errors are serially correlated: the
//! smoother chases the correlated neighbours' noise, which leave-one-out
//! cannot see (Hart 1991; Opsomer, Wang & Yang 2001). The
//! **leave-block-out** criterion (Chu & Marron 1991's "modified
//! cross-validation"; Hart & Vieu 1990) drops the `2l + 1` observations
//! `j` with `|i - j| <= l` when predicting `y_i`,
//!
//! ```text
//! CV_l(h) = n^{-1} sum_i ( y_i - g_{-(i-l..i+l)}(x_i) )^2 ,
//! ```
//!
//! so the neighbours whose errors are correlated with `e_i` never vote on
//! `y_i`. `l = 0` is leave-one-out. This is the roadmap's dependence-aware
//! default for nonlinear autoregressions; the default block half-width is
//! `ceil(n^(1/3))` (the block-bootstrap rate of Hall, Horowitz & Jing
//! 1995), overridable.
//!
//! Both criteria are minimized the same way: bandwidths are parameterized
//! as a common multiple `c` of the Scott reference `h0_j = 1.06 sd(x_j)
//! n^{-1/(4+k)}` (statsmodels' own starting point), `c` is scanned on a
//! 21-point log-spaced grid over `[0.05, 20]`, the bracket around the grid
//! minimum is refined by golden-section search on `log c`, and for `k >= 2`
//! each column's bandwidth is then refined in turn (a 9-point log grid over
//! `[h_j / 8, 8 h_j]` plus golden section, then narrow golden-section
//! polishing rounds per column and along the common scale until a round
//! stops improving the criterion, at most eight rounds). The search
//! is deterministic and derivative-free; it does **not** reproduce
//! statsmodels' Nelder-Mead `fmin` path — the criterion *value* at any
//! bandwidth matches `cv_loo` to `1e-10`, the *selected* bandwidth is
//! checked by property (its criterion is no worse than at statsmodels'
//! `fmin` optimum). A selected bandwidth that sits on a search wall is
//! reported through [`KernelRegressionFit::bandwidth_at_boundary`].
//!
//! # Effective degrees of freedom
//!
//! Both estimators are linear smoothers `fitted = S y`;
//! [`KernelRegressionFit::effective_df`] is `tr(S)` (Hastie & Tibshirani
//! 1990, sec. 3.5): `S_ii = K_h(0) / sum_j K_h(x_j - x_i)` for
//! Nadaraya-Watson and `S_ii = [(Z'WZ)^+]_{11} K_h(0)` for local linear.
//! It runs from `k + 1` (local linear) or `1` (Nadaraya-Watson) at
//! `h -> infinity` — the global linear or constant fit — to `n` as
//! `h -> 0`.

use tsecon_linalg::faer::MatRef;

use crate::error::MlError;
use crate::kernel_ridge::quoted_list;
use crate::util::{check_xy, columns};

/// The largest number of regressors accepted (the curse of dimensionality
/// makes product-kernel smoothing in more than three dimensions
/// uninformative at econometric sample sizes; use [`crate::kernel_ridge`]
/// for higher-dimensional nonlinear regression).
pub const MAX_COLUMNS: usize = 3;

/// Lower wall of the common-multiplier search, relative to the Scott
/// reference bandwidth.
const C_LO: f64 = 0.05;
/// Upper wall of the common-multiplier search.
const C_HI: f64 = 20.0;
/// Log-spaced grid points between the two walls.
const N_GRID: usize = 21;
/// Golden-section iterations on the common multiplier.
const GOLDEN_ITERS: usize = 25;
/// Per-column refinement, first round: a 9-point log grid over
/// `[h_j / 8, 8 h_j]` then golden section on the bracket.
const COL_SPAN: f64 = 8.0;
const COL_GRID: usize = 9;
const COL_GOLDEN_ITERS: usize = 20;
/// Later rounds: golden section over `[h_j / 2, 2 h_j]` per column and over
/// `[c / 1.5, 1.5 c]` on the common scale (the diagonal move that keeps
/// coordinate descent from zig-zagging down a correlated valley).
const POLISH_SPAN: f64 = 2.0;
const POLISH_SCALE_SPAN: f64 = 1.5;
const POLISH_GOLDEN_ITERS: usize = 16;
/// Upper bound on refinement rounds (the stopping rule below usually ends
/// the search well before).
const COL_ROUNDS: usize = 8;
/// Relative criterion improvement below which a refinement round stops.
const REFINE_REL_TOL: f64 = 1e-9;
/// A bandwidth within this factor of a search wall is "at the boundary".
const WALL_TOL: f64 = 1.01;

/// The local estimator (statsmodels `reg_type`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegressionKind {
    /// Local constant, statsmodels `"lc"`.
    NadarayaWatson,
    /// Local linear, statsmodels `"ll"` (the default).
    LocalLinear,
}

impl RegressionKind {
    /// The accepted names, in the order the teaching error lists them.
    pub const ACCEPTED: &'static [&'static str] = &["local_linear", "nadaraya_watson"];

    /// Parses an estimator name.
    ///
    /// # Errors
    ///
    /// [`MlError::InvalidValue`] listing the accepted names.
    pub fn parse(name: &str) -> Result<Self, MlError> {
        match name {
            "local_linear" => Ok(Self::LocalLinear),
            "nadaraya_watson" => Ok(Self::NadarayaWatson),
            other => Err(MlError::InvalidValue {
                what: format!(
                    "unknown kind {other:?}; accepted values are {} (statsmodels reg_type \
                     \"ll\" and \"lc\" respectively)",
                    quoted_list(Self::ACCEPTED)
                ),
            }),
        }
    }

    /// The name as accepted by [`RegressionKind::parse`].
    pub fn as_str(self) -> &'static str {
        match self {
            Self::LocalLinear => "local_linear",
            Self::NadarayaWatson => "nadaraya_watson",
        }
    }
}

/// The kernel shape. Only the Gaussian is implemented — it is the one
/// statsmodels' `KernelReg` validates against, and its unbounded support
/// keeps every local fit well defined.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegressionKernel {
    /// `phi(u) = exp(-u^2 / 2) / sqrt(2 pi)` per column.
    Gaussian,
}

impl RegressionKernel {
    /// The accepted names.
    pub const ACCEPTED: &'static [&'static str] = &["gaussian"];

    /// Parses a kernel name.
    ///
    /// # Errors
    ///
    /// [`MlError::InvalidValue`] listing the accepted names.
    pub fn parse(name: &str) -> Result<Self, MlError> {
        match name {
            "gaussian" => Ok(Self::Gaussian),
            other => Err(MlError::InvalidValue {
                what: format!(
                    "unknown kernel {other:?}; accepted values are {} (compact-support \
                     kernels are deferred: statsmodels' tricube gives far points full \
                     weight, so there is no reference to validate one against)",
                    quoted_list(Self::ACCEPTED)
                ),
            }),
        }
    }

    /// The name as accepted by [`RegressionKernel::parse`].
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Gaussian => "gaussian",
        }
    }
}

/// How the bandwidth is obtained.
#[derive(Debug, Clone, PartialEq)]
pub enum BandwidthSpec {
    /// Use these bandwidths (one per column of `x`) as given.
    Fixed(Vec<f64>),
    /// Minimize the leave-one-out least-squares criterion.
    LooCv,
    /// Minimize the leave-block-out criterion with half-width `block`
    /// (`None` resolves to `ceil(n^(1/3))`).
    BlockCv {
        /// Number of neighbours on each side of `i` dropped together with
        /// `i` when predicting `y_i` (`2 * block + 1` observations).
        block: Option<usize>,
    },
}

impl BandwidthSpec {
    /// The method name (`"fixed"`, `"loo_cv"`, `"block_cv"`).
    pub fn method_name(&self) -> &'static str {
        match self {
            Self::Fixed(_) => "fixed",
            Self::LooCv => "loo_cv",
            Self::BlockCv { .. } => "block_cv",
        }
    }
}

/// Configuration of a [`kernel_regression`] fit.
#[derive(Debug, Clone, PartialEq)]
pub struct KernelRegressionOptions {
    /// Local estimator.
    pub kind: RegressionKind,
    /// Kernel shape.
    pub kernel: RegressionKernel,
    /// Bandwidth specification.
    pub bandwidth: BandwidthSpec,
}

/// Result of a [`kernel_regression`] fit.
#[derive(Debug, Clone, PartialEq)]
pub struct KernelRegressionFit {
    /// Fitted conditional means at the training rows, length `n`.
    pub fitted: Vec<f64>,
    /// Conditional means at the test rows, present only when `x_test` was
    /// given. `NaN` where no training row carries any weight (a test
    /// point so far from the sample that every kernel value underflows).
    pub predicted: Option<Vec<f64>>,
    /// The bandwidth used, one per column of `x` (resolved by the search
    /// under the CV methods).
    pub bandwidth: Vec<f64>,
    /// Name of the bandwidth method (`"fixed"`, `"loo_cv"`, `"block_cv"`).
    pub bandwidth_method: &'static str,
    /// The block half-width under `"block_cv"` (resolved), `None` otherwise.
    pub block: Option<usize>,
    /// The cross-validation criterion at `bandwidth`: leave-one-out under
    /// `"fixed"` and `"loo_cv"`, leave-block-out under `"block_cv"`.
    pub cv_criterion: f64,
    /// `tr(S)` of the linear smoother (see the [module docs](self)).
    pub effective_df: f64,
    /// The local estimator used.
    pub kind: RegressionKind,
    /// The kernel used.
    pub kernel: RegressionKernel,
    /// `true` when a selected bandwidth ended on a wall of the search
    /// range (within 1%): the criterion was still falling at the edge, so
    /// the reported bandwidth is the search's limit, not an interior
    /// optimum — typically a target with no detectable signal (the
    /// criterion wants `h -> infinity`) or one far rougher than the
    /// reference scale. Always `false` under `"fixed"`.
    pub bandwidth_at_boundary: bool,
    /// Number of criterion evaluations the bandwidth search made
    /// (`0` under `"fixed"`).
    pub n_criterion_evaluations: usize,
}

/// Column-major training data with a scratch weight buffer.
struct Design {
    cols: Vec<Vec<f64>>,
    y: Vec<f64>,
    n: usize,
    k: usize,
}

/// Unnormalized product-Gaussian weights of every training row relative
/// to `x0` (`K(0) = 1`). Every quantity this module reports is invariant
/// to the kernel's normalizing constant, so it is dropped here and
/// restored analytically where a formula needs it.
fn weights_into(d: &Design, x0: &[f64], inv_bw: &[f64], out: &mut [f64]) {
    for (i, w) in out.iter_mut().enumerate() {
        let mut s = 0.0;
        for c in 0..d.k {
            let u = (d.cols[c][i] - x0[c]) * inv_bw[c];
            s += u * u;
        }
        *w = (-0.5 * s).exp();
    }
}

/// Symmetric eigendecomposition of an `m x m` (`m <= 4`) matrix by cyclic
/// Jacobi rotations. Returns the eigenvalues on the diagonal of `a` and
/// the eigenvectors as the columns of `v`. Exact to roundoff for these
/// dimensions (the rotations are orthogonal), which is what makes the
/// pseudoinverse below agree with a LAPACK SVD.
#[allow(clippy::needless_range_loop)] // rotation index arithmetic is the point
fn jacobi_eigen(a: &mut [[f64; 4]; 4], m: usize) -> [[f64; 4]; 4] {
    let mut v = [[0.0; 4]; 4];
    for (r, row) in v.iter_mut().enumerate() {
        row[r] = 1.0;
    }
    for _sweep in 0..100 {
        let mut off = 0.0;
        for p in 0..m {
            for q in (p + 1)..m {
                off += a[p][q] * a[p][q];
            }
        }
        if off == 0.0 {
            break;
        }
        for p in 0..(m - 1) {
            for q in (p + 1)..m {
                let apq = a[p][q];
                if apq == 0.0 {
                    continue;
                }
                // An off-diagonal below the rounding resolution of both
                // diagonals cannot move an eigenvalue: zero it and move on
                // (the classic Numerical Recipes test).
                let g = 100.0 * apq.abs();
                if a[p][p].abs() + g == a[p][p].abs() && a[q][q].abs() + g == a[q][q].abs() {
                    a[p][q] = 0.0;
                    a[q][p] = 0.0;
                    continue;
                }
                let theta = (a[q][q] - a[p][p]) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                // A <- J' A J with J the rotation in the (p, q) plane.
                for r in 0..m {
                    let arp = a[r][p];
                    let arq = a[r][q];
                    a[r][p] = c * arp - s * arq;
                    a[r][q] = s * arp + c * arq;
                }
                for r in 0..m {
                    let apr = a[p][r];
                    let aqr = a[q][r];
                    a[p][r] = c * apr - s * aqr;
                    a[q][r] = s * apr + c * aqr;
                }
                a[p][q] = 0.0;
                a[q][p] = 0.0;
                for row in v.iter_mut().take(m) {
                    let vrp = row[p];
                    let vrq = row[q];
                    row[p] = c * vrp - s * vrq;
                    row[q] = s * vrp + c * vrq;
                }
            }
        }
    }
    v
}

/// `(pinv(M) rhs)[0]` and `pinv(M)[0, 0]` for a symmetric `m x m` matrix
/// (`m <= 4`), with NumPy's `pinv` truncation: eigenvalues of magnitude at
/// most `1e-15 * max |lambda|` are dropped.
fn sym_pinv_first(mut a: [[f64; 4]; 4], m: usize, rhs: &[f64; 4]) -> (f64, f64) {
    let v = jacobi_eigen(&mut a, m);
    let lam_max = (0..m).map(|i| a[i][i].abs()).fold(0.0f64, f64::max);
    let cutoff = 1e-15 * lam_max;
    let mut sol0 = 0.0;
    let mut p00 = 0.0;
    for i in 0..m {
        let lam = a[i][i];
        if lam.abs() > cutoff {
            let proj: f64 = (0..m).map(|r| v[r][i] * rhs[r]).sum();
            sol0 += v[0][i] * proj / lam;
            p00 += v[0][i] * v[0][i] / lam;
        }
    }
    (sol0, p00)
}

/// One local fit at `x0` from precomputed weights `w`, excluding the row
/// range `lo..hi`. Returns `(mean, s_factor)` where `s_factor` is the
/// smoother diagonal per unit weight at `x0` (`1 / sum w` for
/// Nadaraya-Watson, `[(Z'WZ)^+]_{00}` for local linear); the mean is
/// `NaN` when no included row carries weight.
#[allow(clippy::needless_range_loop)] // the exclusion window is an index range
fn local_fit(
    d: &Design,
    kind: RegressionKind,
    w: &[f64],
    x0: &[f64],
    lo: usize,
    hi: usize,
) -> (f64, f64) {
    match kind {
        RegressionKind::NadarayaWatson => {
            let mut sw = 0.0;
            let mut swy = 0.0;
            for i in 0..d.n {
                if i >= lo && i < hi {
                    continue;
                }
                sw += w[i];
                swy += w[i] * d.y[i];
            }
            if sw > 0.0 {
                (swy / sw, 1.0 / sw)
            } else {
                (f64::NAN, f64::NAN)
            }
        }
        RegressionKind::LocalLinear => {
            let m = d.k + 1;
            let mut a = [[0.0f64; 4]; 4];
            let mut rhs = [0.0f64; 4];
            let mut dvec = [0.0f64; 4];
            for i in 0..d.n {
                if i >= lo && i < hi {
                    continue;
                }
                let wi = w[i];
                if wi == 0.0 {
                    continue;
                }
                dvec[0] = 1.0;
                for c in 0..d.k {
                    dvec[c + 1] = d.cols[c][i] - x0[c];
                }
                for r in 0..m {
                    let wr = wi * dvec[r];
                    rhs[r] += wr * d.y[i];
                    for s in r..m {
                        a[r][s] += wr * dvec[s];
                    }
                }
            }
            if a[0][0] <= 0.0 {
                return (f64::NAN, f64::NAN);
            }
            for r in 0..m {
                for s in (r + 1)..m {
                    a[s][r] = a[r][s];
                }
            }
            sym_pinv_first(a, m, &rhs)
        }
    }
}

/// The cross-validation criterion at `bw`: `n^{-1} sum_i (y_i - g_{-B_i}(x_i))^2`
/// with `B_i = {j : |i - j| <= half}` (`half = 0` is leave-one-out). Returns
/// `+inf` when any leave-out prediction is undefined (all weights underflow).
fn criterion_at(d: &Design, kind: RegressionKind, bw: &[f64], half: usize, w: &mut [f64]) -> f64 {
    let inv_bw: Vec<f64> = bw.iter().map(|h| 1.0 / h).collect();
    let mut x0 = vec![0.0; d.k];
    let mut total = 0.0;
    for i in 0..d.n {
        for (c, v) in x0.iter_mut().enumerate() {
            *v = d.cols[c][i];
        }
        weights_into(d, &x0, &inv_bw, w);
        let lo = i.saturating_sub(half);
        let hi = (i + half + 1).min(d.n);
        let (g, _) = local_fit(d, kind, w, &x0, lo, hi);
        if !g.is_finite() {
            return f64::INFINITY;
        }
        let e = d.y[i] - g;
        total += e * e;
    }
    total / d.n as f64
}

/// The leave-block-out cross-validation criterion of a fixed bandwidth,
/// exposed for validation: with `block = 0` this is statsmodels'
/// `KernelReg.cv_loo(bw, func)` (matched at `1e-10` in the golden test).
///
/// # Errors
///
/// As [`kernel_regression`] with `BandwidthSpec::Fixed(bandwidth)`.
pub fn cv_criterion(
    x: MatRef<'_, f64>,
    y: &[f64],
    bandwidth: &[f64],
    kind: RegressionKind,
    block: usize,
) -> Result<f64, MlError> {
    let d = build_design(x, y, kind, block)?;
    check_fixed_bandwidth(bandwidth, d.k)?;
    let mut w = vec![0.0; d.n];
    Ok(criterion_at(&d, kind, bandwidth, block, &mut w))
}

/// Validates the column count and the sample size (the sufficiency check
/// runs before any design work, so an empty `x` reports insufficiency
/// rather than a bare empty-input error), then the entries.
fn build_design(
    x: MatRef<'_, f64>,
    y: &[f64],
    kind: RegressionKind,
    half: usize,
) -> Result<Design, MlError> {
    let k = x.ncols();
    if k == 0 {
        return Err(MlError::EmptyInput { what: "x" });
    }
    if k > MAX_COLUMNS {
        return Err(MlError::InvalidValue {
            what: format!(
                "x has {k} columns but kernel_regression accepts at most {MAX_COLUMNS}: a \
                 product-kernel smoother in more dimensions needs samples that grow \
                 exponentially in k (the curse of dimensionality) and is uninformative at \
                 econometric sizes. Reduce the regressors, or use kernel_ridge for a \
                 higher-dimensional nonlinear fit"
            ),
        });
    }
    let needed = min_observations(k, kind, half);
    if x.nrows() < needed {
        return Err(MlError::InsufficientData {
            needed,
            got: x.nrows(),
            what: "kernel_regression",
        });
    }
    let (n, k) = check_xy(x, y)?;
    Ok(Design {
        cols: columns(x),
        y: y.to_vec(),
        n,
        k,
    })
}

/// The smallest sample the local fits can be computed from once the
/// exclusion window (`2 * half + 1` rows) is removed: `k + 1` remaining
/// rows for local linear, one for Nadaraya-Watson.
fn min_observations(k: usize, kind: RegressionKind, half: usize) -> usize {
    let remaining = match kind {
        RegressionKind::LocalLinear => k + 1,
        RegressionKind::NadarayaWatson => 1,
    };
    remaining + 2 * half + 1
}

fn check_fixed_bandwidth(bw: &[f64], k: usize) -> Result<(), MlError> {
    if bw.len() != k {
        return Err(MlError::DimensionMismatch {
            what: "bandwidth must have one entry per column of x",
            expected: k,
            got: bw.len(),
        });
    }
    for (c, &h) in bw.iter().enumerate() {
        if !h.is_finite() || h <= 0.0 {
            return Err(MlError::InvalidValue {
                what: format!(
                    "bandwidth[{c}]={h} must be finite and positive: the Gaussian bandwidth is \
                     the kernel's standard deviation in the units of column {c} of x, so pass a \
                     positive scale (a fraction of the column's spread is the usual starting \
                     point), or select it by cross-validation with bandwidth_method=\"loo_cv\" \
                     or \"block_cv\""
                ),
            });
        }
    }
    Ok(())
}

/// Scott's reference bandwidths `1.06 sd(x_j) n^{-1/(4+k)}` (population
/// sd, as `np.std`), statsmodels' starting point.
fn scott_reference(d: &Design) -> Result<Vec<f64>, MlError> {
    let n = d.n as f64;
    let rate = n.powf(-1.0 / (4.0 + d.k as f64));
    let mut h0 = Vec::with_capacity(d.k);
    for (c, col) in d.cols.iter().enumerate() {
        let mean = col.iter().sum::<f64>() / n;
        let var = col.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n;
        let sd = var.sqrt();
        if sd.is_nan() || sd <= 0.0 {
            return Err(MlError::InvalidValue {
                what: format!(
                    "column {c} of x is constant, so no bandwidth can be selected for it (the \
                     reference bandwidth 1.06 * sd * n^(-1/(4+k)) is zero and every scale of \
                     it smooths identically). Drop the column, or pass a fixed bandwidth with \
                     bandwidth_method=\"fixed\""
                ),
            });
        }
        h0.push(1.06 * sd * rate);
    }
    Ok(h0)
}

/// A criterion evaluator that counts calls and remembers the best point.
struct Search<'a> {
    d: &'a Design,
    kind: RegressionKind,
    half: usize,
    w: Vec<f64>,
    n_eval: usize,
    best_bw: Vec<f64>,
    best_val: f64,
}

impl Search<'_> {
    fn eval(&mut self, bw: &[f64]) -> f64 {
        self.n_eval += 1;
        let v = criterion_at(self.d, self.kind, bw, self.half, &mut self.w);
        if v < self.best_val {
            self.best_val = v;
            self.best_bw = bw.to_vec();
        }
        v
    }
}

/// Golden-section minimization of `f` over `[a, b]`; every evaluation is
/// recorded by the caller's `f`, so the caller reads the best point back
/// from its own bookkeeping.
fn golden_section(mut f: impl FnMut(f64) -> f64, mut a: f64, mut b: f64, iters: usize) {
    let gr = (5.0f64.sqrt() - 1.0) / 2.0;
    let mut c = b - gr * (b - a);
    let mut d = a + gr * (b - a);
    let mut fc = f(c);
    let mut fd = f(d);
    for _ in 0..iters {
        if fc < fd {
            b = d;
            d = c;
            fd = fc;
            c = b - gr * (b - a);
            fc = f(c);
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + gr * (b - a);
            fd = f(d);
        }
    }
}

/// Minimizes the criterion over a log-spaced grid on one scalar (through
/// `bw_of`), then refines the bracket around the grid minimum by golden
/// section. Returns the best scalar found and whether it sits on a wall.
fn grid_then_golden(
    search: &mut Search<'_>,
    bw_of: &dyn Fn(f64) -> Vec<f64>,
    log_lo: f64,
    log_hi: f64,
    n_grid: usize,
    iters: usize,
) -> Option<(f64, bool)> {
    let step = (log_hi - log_lo) / (n_grid - 1) as f64;
    let mut best_i = None;
    let mut best_v = f64::INFINITY;
    for i in 0..n_grid {
        let v = search.eval(&bw_of(log_lo + step * i as f64));
        if v < best_v {
            best_v = v;
            best_i = Some(i);
        }
    }
    let i = best_i?;
    let a = log_lo + step * i.saturating_sub(1) as f64;
    let b = log_lo + step * (i + 1).min(n_grid - 1) as f64;
    let mut local_best = (log_lo + step * i as f64, best_v);
    golden_section(
        |t| {
            let v = search.eval(&bw_of(t));
            if v < local_best.1 {
                local_best = (t, v);
            }
            v
        },
        a,
        b,
        iters,
    );
    let t = local_best.0;
    let wall = (t - log_lo).abs() <= WALL_TOL.ln() || (log_hi - t).abs() <= WALL_TOL.ln();
    Some((t, wall))
}

/// The full bandwidth search: common multiplier, then (for `k >= 2`)
/// cyclic per-column refinement. Returns the bandwidth, its criterion, the
/// boundary flag, and the evaluation count.
fn select_bandwidth(
    d: &Design,
    kind: RegressionKind,
    half: usize,
) -> Result<(Vec<f64>, f64, bool, usize), MlError> {
    let h0 = scott_reference(d)?;
    let mut search = Search {
        d,
        kind,
        half,
        w: vec![0.0; d.n],
        n_eval: 0,
        best_bw: h0.clone(),
        best_val: f64::INFINITY,
    };

    // Stage 1: a common multiple c of the reference bandwidths.
    let h0_ref = &h0;
    let common = |t: f64| -> Vec<f64> { h0_ref.iter().map(|h| h * t.exp()).collect() };
    let (_, mut at_wall) = grid_then_golden(
        &mut search,
        &common,
        C_LO.ln(),
        C_HI.ln(),
        N_GRID,
        GOLDEN_ITERS,
    )
    .ok_or_else(|| MlError::InvalidValue {
        what: format!(
            "the {} cross-validation criterion is not finite at any bandwidth between \
             {C_LO} and {C_HI} times the Scott reference: every leave-out prediction is \
             undefined, which happens when the excluded window leaves no training row \
             within reach of some x_i (a block too wide for the sample, or rows of x that \
             are exact duplicates). Reduce block, or pass a fixed bandwidth",
            if half == 0 {
                "leave-one-out"
            } else {
                "leave-block-out"
            }
        ),
    })?;

    // Stage 2 (k >= 2): refine each column's bandwidth in turn. The first
    // round scans a wide per-column grid (an irrelevant regressor wants a
    // bandwidth far above the common scale); later rounds polish with
    // narrow golden-section brackets per column and along the common
    // scale, until a round no longer improves the criterion.
    if d.k >= 2 {
        for round in 0..COL_ROUNDS {
            let before = search.best_val;
            let mut round_wall = false;
            for c in 0..d.k {
                let base = search.best_bw.clone();
                let center = base[c].ln();
                let per_col = |t: f64| -> Vec<f64> {
                    let mut bw = base.clone();
                    bw[c] = t.exp();
                    bw
                };
                let (span, grid, iters) = if round == 0 {
                    (COL_SPAN, COL_GRID, COL_GOLDEN_ITERS)
                } else {
                    (POLISH_SPAN, 3, POLISH_GOLDEN_ITERS)
                };
                if let Some((_, wall)) = grid_then_golden(
                    &mut search,
                    &per_col,
                    center - span.ln(),
                    center + span.ln(),
                    grid,
                    iters,
                ) {
                    round_wall |= wall;
                }
            }
            if round > 0 {
                let base = search.best_bw.clone();
                let scaled = |t: f64| -> Vec<f64> { base.iter().map(|h| h * t.exp()).collect() };
                grid_then_golden(
                    &mut search,
                    &scaled,
                    -POLISH_SCALE_SPAN.ln(),
                    POLISH_SCALE_SPAN.ln(),
                    3,
                    POLISH_GOLDEN_ITERS,
                );
            }
            at_wall = round_wall;
            if before.is_finite() && (before - search.best_val) <= REFINE_REL_TOL * before.abs() {
                break;
            }
        }
    }

    Ok((search.best_bw, search.best_val, at_wall, search.n_eval))
}

/// Nadaraya-Watson or local-linear kernel regression of `y` on the columns
/// of `x` (`n x k`, `k <= 3`) with a product Gaussian kernel, at a fixed
/// or cross-validated bandwidth. See the [module docs](self) for the
/// estimators, the two criteria, and the search.
///
/// `x_test` (`m x k`) adds `predicted`.
///
/// # Errors
///
/// * [`MlError::EmptyInput`] / [`MlError::DimensionMismatch`] /
///   [`MlError::NonFinite`] on malformed inputs (`x_test` included; a fixed
///   bandwidth of the wrong length is a `DimensionMismatch`);
/// * [`MlError::InvalidValue`] if `x` has more than [`MAX_COLUMNS`]
///   columns, a fixed bandwidth is not positive, `block` is `0`, a column
///   of `x` is constant under a CV method, or the criterion is undefined
///   everywhere in the search range;
/// * [`MlError::InsufficientData`] if fewer than `k + 2 + 2 * block`
///   (local linear) or `2 + 2 * block` (Nadaraya-Watson) rows remain to
///   fit from — `block = 0` outside `"block_cv"`.
pub fn kernel_regression(
    x: MatRef<'_, f64>,
    y: &[f64],
    x_test: Option<MatRef<'_, f64>>,
    opts: &KernelRegressionOptions,
) -> Result<KernelRegressionFit, MlError> {
    let RegressionKernel::Gaussian = opts.kernel;
    let kind = opts.kind;

    // Resolve the exclusion half-width and the sample requirement first,
    // before any search or design work (the round-10 lesson).
    let (half, block) = match &opts.bandwidth {
        BandwidthSpec::BlockCv { block } => {
            let b = match *block {
                Some(0) => {
                    return Err(MlError::InvalidValue {
                        what: "block=0 under bandwidth_method=\"block_cv\" excludes nothing \
                               beyond the observation itself, which is leave-one-out: pass \
                               bandwidth_method=\"loo_cv\" for that, or block >= 1 (the number \
                               of neighbours dropped on each side of i; None selects \
                               ceil(n^(1/3)))"
                            .to_string(),
                    })
                }
                Some(b) => b,
                None => (x.nrows() as f64).powf(1.0 / 3.0).ceil() as usize,
            };
            (b, Some(b))
        }
        _ => (0, None),
    };
    let d = build_design(x, y, kind, half)?;
    if let Some(xt) = x_test {
        if xt.ncols() != d.k {
            return Err(MlError::DimensionMismatch {
                what: "x_test must have the same number of columns as x",
                expected: d.k,
                got: xt.ncols(),
            });
        }
        for j in 0..xt.ncols() {
            for i in 0..xt.nrows() {
                if !xt[(i, j)].is_finite() {
                    return Err(MlError::NonFinite { what: "x_test" });
                }
            }
        }
    }
    let (bandwidth, at_boundary, n_eval) = match &opts.bandwidth {
        BandwidthSpec::Fixed(bw) => {
            check_fixed_bandwidth(bw, d.k)?;
            (bw.clone(), false, 0)
        }
        BandwidthSpec::LooCv | BandwidthSpec::BlockCv { .. } => {
            let (bw, _, wall, n_eval) = select_bandwidth(&d, kind, half)?;
            (bw, wall, n_eval)
        }
    };

    // Final pass at the resolved bandwidth: fitted values, tr(S), and the
    // criterion, from one weight row per observation.
    let inv_bw: Vec<f64> = bandwidth.iter().map(|h| 1.0 / h).collect();
    let mut w = vec![0.0; d.n];
    let mut x0 = vec![0.0; d.k];
    let mut fitted = Vec::with_capacity(d.n);
    let mut edf = 0.0;
    let mut total = 0.0;
    for i in 0..d.n {
        for (c, v) in x0.iter_mut().enumerate() {
            *v = d.cols[c][i];
        }
        weights_into(&d, &x0, &inv_bw, &mut w);
        let (g, s) = local_fit(&d, kind, &w, &x0, d.n, d.n);
        fitted.push(g);
        // S_ii = s_factor * K(0) with the unnormalized K(0) = 1.
        edf += s;
        let lo = i.saturating_sub(half);
        let hi = (i + half + 1).min(d.n);
        let (g_out, _) = local_fit(&d, kind, &w, &x0, lo, hi);
        let e = d.y[i] - g_out;
        total += e * e;
    }
    let cv_criterion = total / d.n as f64;

    let predicted = x_test.map(|xt| {
        let mut out = Vec::with_capacity(xt.nrows());
        for r in 0..xt.nrows() {
            for (c, v) in x0.iter_mut().enumerate() {
                *v = xt[(r, c)];
            }
            weights_into(&d, &x0, &inv_bw, &mut w);
            out.push(local_fit(&d, kind, &w, &x0, d.n, d.n).0);
        }
        out
    });

    Ok(KernelRegressionFit {
        fitted,
        predicted,
        bandwidth,
        bandwidth_method: opts.bandwidth.method_name(),
        block,
        cv_criterion,
        effective_df: edf,
        kind,
        kernel: opts.kernel,
        bandwidth_at_boundary: at_boundary,
        n_criterion_evaluations: n_eval,
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use tsecon_linalg::faer::Mat;

    #[test]
    fn jacobi_pinv_matches_a_direct_solve_on_a_definite_matrix() {
        // M = [[4, 1, 0], [1, 3, 1], [0, 1, 2]], rhs = [1, 2, 3].
        let mut a = [[0.0; 4]; 4];
        a[0] = [4.0, 1.0, 0.0, 0.0];
        a[1] = [1.0, 3.0, 1.0, 0.0];
        a[2] = [0.0, 1.0, 2.0, 0.0];
        let rhs = [1.0, 2.0, 3.0, 0.0];
        let (s0, p00) = sym_pinv_first(a, 3, &rhs);
        // det = 4*(6-1) - 1*(2-0) = 18; inverse row 0 = [5, -2, 1] / 18.
        assert!((s0 - (5.0 - 4.0 + 3.0) / 18.0).abs() < 1e-14);
        assert!((p00 - 5.0 / 18.0).abs() < 1e-14);
    }

    #[test]
    fn jacobi_pinv_truncates_a_rank_deficient_matrix() {
        // rank-1 M = u u' with u = [1, 2]: pinv = u u' / ||u||^4.
        let mut a = [[0.0; 4]; 4];
        a[0] = [1.0, 2.0, 0.0, 0.0];
        a[1] = [2.0, 4.0, 0.0, 0.0];
        let rhs = [1.0, 0.0, 0.0, 0.0];
        let (s0, p00) = sym_pinv_first(a, 2, &rhs);
        assert!((p00 - 1.0 / 25.0).abs() < 1e-14);
        assert!((s0 - 1.0 / 25.0).abs() < 1e-14);
    }

    #[test]
    fn insufficient_data_names_the_exact_minimum() {
        let x = Mat::from_fn(4, 2, |i, j| (i + 2 * j) as f64);
        let y = vec![0.0, 1.0, 0.5, 0.25];
        let opts = KernelRegressionOptions {
            kind: RegressionKind::LocalLinear,
            kernel: RegressionKernel::Gaussian,
            bandwidth: BandwidthSpec::BlockCv { block: Some(1) },
        };
        let e = kernel_regression(x.as_ref(), &y, None, &opts).unwrap_err();
        // k + 1 = 3 remaining rows after dropping 2*1 + 1 = 3: needed 6.
        assert_eq!(
            e,
            MlError::InsufficientData {
                needed: 6,
                got: 4,
                what: "kernel_regression"
            }
        );
        // n = 6 succeeds.
        let x6 = Mat::from_fn(6, 2, |i, j| (i * i + 2 * j) as f64 * 0.3);
        let y6 = vec![0.0, 1.0, 0.5, 0.25, 0.7, 0.1];
        let fixed = KernelRegressionOptions {
            bandwidth: BandwidthSpec::Fixed(vec![1.0, 1.0]),
            ..opts.clone()
        };
        assert!(kernel_regression(x6.as_ref(), &y6, None, &fixed).is_ok());
    }
}
