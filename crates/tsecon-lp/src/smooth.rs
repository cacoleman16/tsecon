//! Smooth local projections (Barnichon-Brownlees 2019): [`smooth_lp`].
//!
//! Estimates the IRF path *jointly* across horizons as a B-spline expansion
//! in the horizon, with a ridge penalty on the r-th difference of the basis
//! coefficients. The estimator is a closed-form penalized least squares over
//! the stacked per-horizon design; see [`smooth_lp`] for the model, the
//! standard-error construction (and what it conditions on), and the
//! cross-validation rule.

use tsecon_hac::Kernel;
use tsecon_linalg::faer::linalg::solvers::DenseSolveCore;
use tsecon_linalg::faer::{Mat, Side};

use crate::design::check_finite;
use crate::error::LpError;
use crate::spec::{LpSpec, SeKind};

/// How the smoothing parameter `lambda` is chosen in [`smooth_lp`].
#[derive(Debug, Clone, PartialEq)]
pub enum SmoothLambda {
    /// Use the given fixed `lambda >= 0`. `0.0` is the unpenalized stacked
    /// estimator; with the default interpolating basis it reproduces the
    /// per-horizon HAC-path [`lp`](crate::lp) point estimates exactly.
    Fixed(f64),
    /// Choose `lambda` from a grid by leave-h-block-out cross-validation.
    ///
    /// The rule (Burman-Chow-Nolan h-block CV, adapted to the stacked LP):
    /// base times `t = p..n-1` are split into `n_folds` contiguous blocks
    /// (fold `j` covers `p + floor(j*nb/n_folds) <= t < p +
    /// floor((j+1)*nb/n_folds)`, `nb = n - p`). For each fold the *test*
    /// rows are all stacked rows `(h, t)` with `t` in the block; the
    /// *training* rows exclude the block plus a buffer of `horizons +
    /// n_lag_controls` base times on each side — the maximal residual/lag
    /// overlap between a training and a test row, so no information leaks
    /// across the split through the overlapping LP residuals. The score is
    /// the total squared out-of-block prediction error over all folds
    /// divided by the total number of test rows, and the chosen `lambda` is
    /// the grid minimizer (first index on ties).
    CrossValidate {
        /// Candidate `lambda` grid. `None` uses the default log-spaced grid
        /// `10^(k/2)` for `k = -4..=12` (17 values, `1e-2` to `1e6`).
        grid: Option<Vec<f64>>,
        /// Number of contiguous folds (default 5; must be `>= 2`).
        n_folds: usize,
    },
}

impl Default for SmoothLambda {
    fn default() -> Self {
        SmoothLambda::CrossValidate {
            grid: None,
            n_folds: 5,
        }
    }
}

/// Configuration for [`smooth_lp`].
#[derive(Debug, Clone, PartialEq)]
pub struct SmoothLpSpec {
    /// Maximum horizon `H`; the IRF is estimated for `h in 0..=horizons`.
    pub horizons: usize,
    /// Number of lagged-outcome controls per horizon block (as in
    /// [`LpSpec::n_lag_controls`]).
    pub n_lag_controls: usize,
    /// B-spline degree (default 3, cubic). Must satisfy `1 <= degree <
    /// n_basis`.
    pub degree: usize,
    /// Number of B-spline basis functions `K`. `None` (the default) uses the
    /// *interpolating* size `K = horizons + 1`, for which `lambda = 0`
    /// reproduces the per-horizon LP exactly and all smoothing is done by
    /// the penalty. Smaller `K` adds hard pre-smoothing on top of the
    /// penalty. Must satisfy `degree + 1 <= K <= horizons + 1`.
    pub n_basis: Option<usize>,
    /// Order `r` of the difference penalty on the basis coefficients
    /// (default 2). With the uniform basis used here, `r = 2` shrinks the
    /// IRF toward a straight line in the horizon, `r = 1` toward a
    /// constant. Must satisfy `1 <= r < n_basis`.
    pub penalty_order: usize,
    /// How `lambda` is chosen; see [`SmoothLambda`]. Defaults to
    /// cross-validation on the default grid with 5 folds.
    pub lambda: SmoothLambda,
    /// Bartlett bandwidth (lag truncation) for the stacked HAC sandwich.
    /// `None` uses `horizons + n_lag_controls`, the `maxlags = h + p`
    /// convention of the HAC LP path evaluated at the longest horizon.
    pub hac_maxlags: Option<usize>,
}

impl SmoothLpSpec {
    /// A spec with the library defaults: cubic splines, interpolating basis
    /// size, second-difference penalty, cross-validated `lambda`.
    #[must_use]
    pub fn new(horizons: usize, n_lag_controls: usize) -> Self {
        SmoothLpSpec {
            horizons,
            n_lag_controls,
            degree: 3,
            n_basis: None,
            penalty_order: 2,
            lambda: SmoothLambda::default(),
            hac_maxlags: None,
        }
    }

    /// Builder: use a fixed `lambda` instead of cross-validation.
    #[must_use]
    pub fn with_lambda(mut self, lambda: f64) -> Self {
        self.lambda = SmoothLambda::Fixed(lambda);
        self
    }

    /// Builder: cross-validate `lambda` over `grid` (or the default grid
    /// when `None`) with `n_folds` contiguous folds.
    #[must_use]
    pub fn with_cv(mut self, grid: Option<Vec<f64>>, n_folds: usize) -> Self {
        self.lambda = SmoothLambda::CrossValidate { grid, n_folds };
        self
    }

    /// Builder: set the B-spline degree.
    #[must_use]
    pub fn with_degree(mut self, degree: usize) -> Self {
        self.degree = degree;
        self
    }

    /// Builder: set the number of basis functions explicitly.
    #[must_use]
    pub fn with_n_basis(mut self, n_basis: usize) -> Self {
        self.n_basis = Some(n_basis);
        self
    }

    /// Builder: set the difference-penalty order.
    #[must_use]
    pub fn with_penalty_order(mut self, penalty_order: usize) -> Self {
        self.penalty_order = penalty_order;
        self
    }

    /// Builder: fix the Bartlett bandwidth of the stacked HAC sandwich.
    #[must_use]
    pub fn with_hac_maxlags(mut self, maxlags: usize) -> Self {
        self.hac_maxlags = Some(maxlags);
        self
    }
}

/// The result of a smooth local projection ([`smooth_lp`]).
#[derive(Debug, Clone, PartialEq)]
pub struct SmoothLpResult {
    /// Horizons estimated, `[0, 1, ..., H]`.
    pub horizons: Vec<usize>,
    /// Smoothed impulse response at each horizon, `irf_h = B_h' theta`.
    pub irf: Vec<f64>,
    /// Delta-method standard error of the smoothed response,
    /// `sqrt(B_h' V_theta B_h)` with `V` the stacked Bartlett-HAC sandwich.
    /// **Conditional on `lambda`** (treated as fixed even when it was chosen
    /// by cross-validation) and describing the sampling variability of the
    /// *penalized* estimator around its own probability limit — the
    /// shrinkage bias of the penalty is not accounted for, so these are
    /// shrinkage-estimator standard errors, not exact frequentist ones.
    pub se: Vec<f64>,
    /// The smoothing parameter actually used (the fixed value, or the CV
    /// winner).
    pub lambda: f64,
    /// The cross-validation grid (empty when `lambda` was fixed).
    pub cv_grid: Vec<f64>,
    /// Mean squared out-of-block prediction error per grid value (empty when
    /// `lambda` was fixed); `lambda` is the grid value with the smallest
    /// entry.
    pub cv_scores: Vec<f64>,
    /// Estimated basis coefficients `theta` (length `n_basis`).
    pub theta: Vec<f64>,
    /// The B-spline basis matrix, `basis[h][k] = B_k(h)`, `(H+1) x n_basis`.
    pub basis: Vec<Vec<f64>>,
    /// The uniform knot vector underlying the basis (length `n_basis +
    /// degree + 1`).
    pub knots: Vec<f64>,
    /// The raw per-horizon LP point estimates for comparison: exactly
    /// [`lp`](crate::lp) with [`SeSpec::Hac`](crate::SeSpec::Hac)
    /// `{ maxlags: None }` (Newey-West, `maxlags = h + p`) — the un-smoothed
    /// estimator that `lambda -> 0` (with the default interpolating basis)
    /// reproduces.
    pub irf_raw: Vec<f64>,
    /// Standard errors of the raw per-horizon LP path.
    pub se_raw: Vec<f64>,
    /// Effective observations in each horizon block of the stacked design.
    pub nobs_per_h: Vec<usize>,
    /// Always [`SeKind::SmoothStackedHac`].
    pub se_kind: SeKind,
}

/// Estimate a smooth local-projection impulse-response function
/// (Barnichon & Brownlees 2019, *Review of Economics and Statistics*).
///
/// # The model and the closed form
///
/// Each horizon keeps the plain Jordà regression of the HAC LP path (no
/// shock-lag augmentation),
///
/// ```text
///   y_{t+h} = beta_h * shock_t + c_h + sum_{l=1}^{p} phi_{h,l} y_{t-l} + u_{t,h},
/// ```
///
/// but the IRF path is restricted to a B-spline expansion in the horizon,
/// `beta_h = sum_k theta_k B_k(h)`, while the intercept and lag controls
/// stay horizon-specific and unpenalized. Stacking all horizons (each
/// horizon `h` contributing its own usable sample `t = p..n-1-h`) gives a
/// single penalized least-squares problem with the **closed form**
///
/// ```text
///   theta_hat = (X'X + lambda * P)^{-1} X'y,    P = blkdiag(D_r' D_r, 0),
/// ```
///
/// where `D_r` is the r-th difference matrix on the `K` basis coefficients
/// and the zero block leaves the controls unpenalized. The basis is the
/// Eilers-Marx P-spline basis: degree-`d` B-splines on *uniform unclamped*
/// knots spanning `[0, H]`, so that `D_2 theta = 0` is exactly "IRF linear
/// in `h`" — `r = 2` (the default) shrinks the IRF toward a straight line,
/// `r = 1` toward a constant. With the default interpolating size
/// `K = H + 1`, `lambda = 0` reproduces the per-horizon
/// [`lp`](crate::lp)/HAC point estimates exactly, so the whole path from
/// "raw LP" to "line" is indexed by `lambda`.
///
/// # Standard errors — what they condition on
///
/// [`SmoothLpResult::se`] is the delta method through the basis,
/// `se_h = sqrt(B_h' V_theta B_h)`, with `V = A^{-1} M A^{-1}`,
/// `A = X'X + lambda P`, and `M` a Bartlett (Newey-West) HAC estimate over
/// the **base-time-aggregated scores** `g_t = sum_h x_{(h,t)} u_{(h,t)}`
/// with bandwidth `horizons + n_lag_controls` by default (no small-sample
/// correction). Aggregating the stacked rows that share a base period
/// before applying the kernel is what accounts for the cross-horizon
/// correlation of the overlapping LP residuals. Two honest caveats, stated
/// rather than hidden: the penalty makes `theta_hat` a *shrinkage*
/// estimator, so these are standard errors around the estimator's own
/// (shrunk) probability limit and do not account for shrinkage bias; and
/// `lambda` is treated as fixed even when it was cross-validated. Under
/// `lambda -> 0` they reduce to the ordinary stacked-LP HAC.
///
/// # Choosing `lambda`
///
/// [`SmoothLambda::Fixed`] uses the given value; [`SmoothLambda::CrossValidate`]
/// (the default) picks the minimizer of a leave-h-block-out cross-validation
/// score over a grid — contiguous base-time folds with a buffer of
/// `horizons + n_lag_controls` periods excluded around each held-out block,
/// so the overlapping-residual dependence cannot leak into training. The
/// rule is spelled out at [`SmoothLambda::CrossValidate`].
///
/// # Errors
///
/// [`LpError::LengthMismatch`] / [`LpError::NonFinite`] on malformed inputs,
/// [`LpError::SeriesTooShort`] / [`LpError::HorizonTooLong`] when a horizon
/// has no usable sample, [`LpError::SplineConfig`] on an infeasible
/// degree/basis/penalty combination, [`LpError::InvalidLambda`] /
/// [`LpError::EmptyLambdaGrid`] / [`LpError::CvConfig`] on bad smoothing
/// setups, and [`LpError::Hac`] wrapping a singular stacked design.
pub fn smooth_lp(y: &[f64], shock: &[f64], spec: &SmoothLpSpec) -> Result<SmoothLpResult, LpError> {
    let n = y.len();
    if shock.len() != n {
        return Err(LpError::LengthMismatch {
            what: "impulse (shock) vs outcome (y)",
            expected: n,
            got: shock.len(),
        });
    }
    check_finite(y, "outcome (y)")?;
    check_finite(shock, "impulse (shock)")?;

    let hmax = spec.horizons;
    let p = spec.n_lag_controls;
    if n <= p {
        return Err(LpError::SeriesTooShort {
            n,
            n_lag_controls: p,
        });
    }
    // Every horizon block must be individually estimable: its own intercept
    // + p lag controls + the shock column need head room.
    for h in 0..=hmax {
        let nobs = n.saturating_sub(h + p);
        if nobs <= 2 + p {
            return Err(LpError::HorizonTooLong {
                horizon: h,
                nobs,
                nparams: 2 + p,
            });
        }
    }

    let cfg = SplineConfig::validate(spec)?;
    let geo = Geometry::new(hmax, p, cfg.n_basis);

    // Basis and penalty.
    let knots = uniform_knots(hmax, cfg.degree, cfg.n_basis);
    let basis: Vec<Vec<f64>> = (0..=hmax)
        .map(|h| basis_row(&knots, cfg.degree, cfg.n_basis, h as f64))
        .collect();
    let pen = penalty_matrix(cfg.n_basis, cfg.penalty_order, geo.q);

    let stack = Stack {
        y,
        shock,
        basis: &basis,
        geo,
    };

    // Choose lambda.
    let (lambda, cv_grid, cv_scores) = match &spec.lambda {
        SmoothLambda::Fixed(l) => {
            check_lambda(*l)?;
            (*l, Vec::new(), Vec::new())
        }
        SmoothLambda::CrossValidate { grid, n_folds } => {
            let grid = match grid {
                Some(g) => {
                    if g.is_empty() {
                        return Err(LpError::EmptyLambdaGrid);
                    }
                    for &l in g {
                        check_lambda(l)?;
                    }
                    g.clone()
                }
                None => default_grid(),
            };
            let scores = cross_validate(&stack, &pen, &grid, *n_folds)?;
            let mut best = 0usize;
            for (i, &s) in scores.iter().enumerate() {
                if s < scores[best] {
                    best = i;
                }
            }
            (grid[best], grid, scores)
        }
    };

    // Full-sample fit at the chosen lambda.
    let (a0, b) = stack.normal_equations(|_| true);
    let a = add_penalty(&a0, &pen, lambda);
    let a_inv = a.llt(Side::Lower).map_err(|_| singular())?.inverse();
    let theta_full = &a_inv * &b; // q x 1

    // IRF through the basis.
    let k = cfg.n_basis;
    let theta: Vec<f64> = (0..k).map(|i| theta_full[(i, 0)]).collect();
    let irf: Vec<f64> = basis
        .iter()
        .map(|row| row.iter().zip(&theta).map(|(b, t)| b * t).sum())
        .collect();

    // Sandwich SEs: Bartlett HAC over base-time-aggregated scores.
    let bw = spec.hac_maxlags.unwrap_or(hmax + p);
    let m = stack.score_hac(&theta_full, bw);
    let v = &a_inv * &m * &a_inv;
    let mut se = Vec::with_capacity(hmax + 1);
    for row in &basis {
        let mut var = 0.0;
        for (i, &bi) in row.iter().enumerate() {
            for (j, &bj) in row.iter().enumerate() {
                var += bi * v[(i, j)] * bj;
            }
        }
        if var < 0.0 {
            return Err(LpError::Hac(tsecon_hac::HacError::NumericalBreakdown {
                what: "smooth-LP sandwich variance",
            }));
        }
        se.push(var.sqrt());
    }

    // Raw per-horizon LP for comparison (the lambda -> 0 anchor): the HAC
    // path of `lp` with its own maxlags = h + p convention.
    let raw = crate::lp(y, shock, LpSpec::new(hmax, p).with_hac(None))?;

    Ok(SmoothLpResult {
        horizons: (0..=hmax).collect(),
        irf,
        se,
        lambda,
        cv_grid,
        cv_scores,
        theta,
        basis,
        knots,
        irf_raw: raw.irf,
        se_raw: raw.se,
        nobs_per_h: (0..=hmax).map(|h| n - h - p).collect(),
        se_kind: SeKind::SmoothStackedHac,
    })
}

/// Validated spline configuration.
struct SplineConfig {
    degree: usize,
    n_basis: usize,
    penalty_order: usize,
}

impl SplineConfig {
    fn validate(spec: &SmoothLpSpec) -> Result<Self, LpError> {
        let degree = spec.degree;
        let n_basis = spec.n_basis.unwrap_or(spec.horizons + 1);
        let penalty_order = spec.penalty_order;
        let err = |constraint: &'static str| {
            Err(LpError::SplineConfig {
                horizons: spec.horizons,
                degree,
                n_basis,
                penalty_order,
                constraint,
            })
        };
        if degree < 1 {
            return err("degree >= 1 (degree-0 splines are step functions; smoothing needs at least a linear basis)");
        }
        if n_basis < degree + 1 {
            return err("n_basis >= degree + 1 (a degree-d spline needs at least d + 1 basis functions; raise n_basis or lower degree)");
        }
        if n_basis > spec.horizons + 1 {
            return err("n_basis <= horizons + 1 (more basis functions than horizon points cannot be identified; lower n_basis or raise horizons)");
        }
        if penalty_order < 1 || penalty_order >= n_basis {
            return err("1 <= penalty_order < n_basis (the r-th difference matrix needs more than r coefficients; lower penalty_order or raise n_basis)");
        }
        Ok(SplineConfig {
            degree,
            n_basis,
            penalty_order,
        })
    }
}

/// Column layout of the stacked design.
#[derive(Clone, Copy)]
struct Geometry {
    hmax: usize,
    p: usize,
    k: usize,
    /// Total columns: `k` spline columns then `(hmax+1)` control blocks of
    /// `1 + p` columns each.
    q: usize,
}

impl Geometry {
    fn new(hmax: usize, p: usize, k: usize) -> Self {
        Geometry {
            hmax,
            p,
            k,
            q: k + (hmax + 1) * (1 + p),
        }
    }
}

/// The stacked smooth-LP design, materialized row-by-row (each row has only
/// `degree + 1` spline entries plus its `1 + p` control entries, so the
/// normal equations are accumulated sparsely without forming `X`).
struct Stack<'a> {
    y: &'a [f64],
    shock: &'a [f64],
    basis: &'a [Vec<f64>],
    geo: Geometry,
}

impl Stack<'_> {
    /// Sparse entries of row `(h, t)` into `out` as `(column, value)`.
    fn row(&self, h: usize, t: usize, out: &mut Vec<(usize, f64)>) {
        out.clear();
        let g = self.geo;
        for (kcol, &bv) in self.basis[h].iter().enumerate() {
            if bv != 0.0 {
                out.push((kcol, self.shock[t] * bv));
            }
        }
        let off = g.k + h * (1 + g.p);
        out.push((off, 1.0));
        for lag in 1..=g.p {
            out.push((off + lag, self.y[t - lag]));
        }
    }

    /// Accumulate `X'X` and `X'y` over rows whose base time passes `keep`.
    /// Returns `(X'X, X'y)`.
    fn normal_equations<F: Fn(usize) -> bool>(&self, keep: F) -> (Mat<f64>, Mat<f64>) {
        let g = self.geo;
        let n = self.y.len();
        let mut a = Mat::<f64>::zeros(g.q, g.q);
        let mut b = Mat::<f64>::zeros(g.q, 1);
        let mut entries = Vec::with_capacity(g.k + 1 + g.p);
        for h in 0..=g.hmax {
            for t in g.p..n - h {
                if !keep(t) {
                    continue;
                }
                self.row(h, t, &mut entries);
                let yv = self.y[t + h];
                for &(i, vi) in &entries {
                    b[(i, 0)] += vi * yv;
                    for &(j, vj) in &entries {
                        if j >= i {
                            a[(i, j)] += vi * vj;
                        }
                    }
                }
            }
        }
        // Mirror the upper triangle.
        for i in 0..g.q {
            for j in 0..i {
                a[(i, j)] = a[(j, i)];
            }
        }
        (a, b)
    }

    /// Bartlett-HAC "meat" over base-time-aggregated scores
    /// `g_t = sum_h x_{(h,t)} u_{(h,t)}` at the given lag truncation
    /// (`w_l = 1 - l/(bw+1)`, no small-sample correction).
    fn score_hac(&self, theta_full: &Mat<f64>, bw: usize) -> Mat<f64> {
        let g = self.geo;
        let n = self.y.len();
        let nbase = n - g.p;
        let mut scores = vec![0.0_f64; nbase * g.q];
        let mut entries = Vec::with_capacity(g.k + 1 + g.p);
        for h in 0..=g.hmax {
            for t in g.p..n - h {
                self.row(h, t, &mut entries);
                let mut fit = 0.0;
                for &(i, vi) in &entries {
                    fit += vi * theta_full[(i, 0)];
                }
                let u = self.y[t + h] - fit;
                let row = &mut scores[(t - g.p) * g.q..(t - g.p + 1) * g.q];
                for &(i, vi) in &entries {
                    row[i] += vi * u;
                }
            }
        }
        let mut m = Mat::<f64>::zeros(g.q, g.q);
        for lag in 0..=bw.min(nbase.saturating_sub(1)) {
            let w = Kernel::Bartlett.weight(lag, bw as f64);
            if lag > 0 && w == 0.0 {
                break;
            }
            for t in lag..nbase {
                let row_t = &scores[t * g.q..(t + 1) * g.q];
                let row_l = &scores[(t - lag) * g.q..(t - lag + 1) * g.q];
                for i in 0..g.q {
                    if row_t[i] == 0.0 && row_l[i] == 0.0 {
                        continue;
                    }
                    for j in 0..g.q {
                        let gm = row_t[i] * row_l[j];
                        if lag == 0 {
                            m[(i, j)] += gm;
                        } else {
                            m[(i, j)] += w * gm;
                            m[(j, i)] += w * gm;
                        }
                    }
                }
            }
        }
        m
    }
}

/// Uniform (Eilers-Marx) knot vector on `[0, hmax]`:
/// `t_j = (j - degree) * hmax / (n_basis - degree)` for
/// `j = 0..=n_basis + degree`.
fn uniform_knots(hmax: usize, degree: usize, n_basis: usize) -> Vec<f64> {
    let n_seg = n_basis - degree;
    let delta = hmax as f64 / n_seg as f64;
    (0..=n_basis + degree)
        .map(|j| (j as f64 - degree as f64) * delta)
        .collect()
}

/// One row of the B-spline design matrix at `x` by the Cox-de Boor
/// triangular recursion (de Boor 2001, Algorithm A2.2 layout); the last
/// span is treated as right-closed so `x = hmax` evaluates cleanly.
fn basis_row(knots: &[f64], degree: usize, n_basis: usize, x: f64) -> Vec<f64> {
    // Knot span i with t_i <= x < t_{i+1}, clamped to [degree, n_basis - 1].
    let mut span = degree;
    while span + 1 < n_basis && knots[span + 1] <= x {
        span += 1;
    }
    let mut vals = vec![0.0_f64; degree + 1];
    vals[0] = 1.0;
    let mut left = vec![0.0_f64; degree + 1];
    let mut right = vec![0.0_f64; degree + 1];
    for j in 1..=degree {
        left[j] = x - knots[span + 1 - j];
        right[j] = knots[span + j] - x;
        let mut saved = 0.0;
        for r in 0..j {
            let temp = vals[r] / (right[r + 1] + left[j - r]);
            vals[r] = saved + right[r + 1] * temp;
            saved = left[j - r] * temp;
        }
        vals[j] = saved;
    }
    let mut row = vec![0.0_f64; n_basis];
    row[span - degree..=span].copy_from_slice(&vals);
    row
}

/// `blkdiag(D_r' D_r, 0)` as a dense `q x q` matrix, where `D_r` is the
/// r-th difference matrix on the first `k` (spline) coefficients.
fn penalty_matrix(k: usize, r: usize, q: usize) -> Mat<f64> {
    // Rows of D_r: iterated differencing of the identity.
    let mut d: Vec<Vec<f64>> = (0..k)
        .map(|i| {
            let mut row = vec![0.0; k];
            row[i] = 1.0;
            row
        })
        .collect();
    for _ in 0..r {
        d = (1..d.len())
            .map(|i| {
                d[i].iter()
                    .zip(&d[i - 1])
                    .map(|(a, b)| a - b)
                    .collect::<Vec<f64>>()
            })
            .collect();
    }
    let mut pen = Mat::<f64>::zeros(q, q);
    for row in &d {
        for i in 0..k {
            if row[i] == 0.0 {
                continue;
            }
            for j in 0..k {
                pen[(i, j)] += row[i] * row[j];
            }
        }
    }
    pen
}

/// `A0 + lambda * P` (both `q x q`).
fn add_penalty(a0: &Mat<f64>, pen: &Mat<f64>, lambda: f64) -> Mat<f64> {
    let q = a0.nrows();
    Mat::from_fn(q, q, |i, j| a0[(i, j)] + lambda * pen[(i, j)])
}

/// The default cross-validation grid: `10^(k/2)` for `k = -4..=12`.
fn default_grid() -> Vec<f64> {
    (-4..=12).map(|k| 10f64.powf(k as f64 / 2.0)).collect()
}

fn check_lambda(lambda: f64) -> Result<(), LpError> {
    if !lambda.is_finite() || lambda < 0.0 {
        return Err(LpError::InvalidLambda { value: lambda });
    }
    Ok(())
}

fn singular() -> LpError {
    LpError::Hac(tsecon_hac::HacError::SingularDesign {
        what: "smooth-LP stacked normal equations (X'X + lambda P); the design is \
               collinear — check for a constant shock or too many basis functions",
    })
}

/// The leave-h-block-out CV scores for each grid value (mean squared
/// out-of-block prediction error; see [`SmoothLambda::CrossValidate`]).
fn cross_validate(
    stack: &Stack<'_>,
    pen: &Mat<f64>,
    grid: &[f64],
    n_folds: usize,
) -> Result<Vec<f64>, LpError> {
    let g = stack.geo;
    let n = stack.y.len();
    let nb = n - g.p;
    if n_folds < 2 || n_folds > nb {
        return Err(LpError::CvConfig {
            n_folds,
            n_base: nb,
        });
    }
    let buffer = g.hmax + g.p;

    let mut sse = vec![0.0_f64; grid.len()];
    let mut n_test = 0usize;
    let mut entries = Vec::with_capacity(g.k + 1 + g.p);
    for fold in 0..n_folds {
        let lo = g.p + fold * nb / n_folds;
        let hi = g.p + (fold + 1) * nb / n_folds;
        let excl_lo = lo.saturating_sub(buffer);
        let excl_hi = (hi + buffer).min(n);
        let keep = |t: usize| t < excl_lo || t >= excl_hi;
        if (g.p..n).filter(|&t| keep(t)).count() < 2 + g.p {
            return Err(LpError::CvConfig {
                n_folds,
                n_base: nb,
            });
        }
        let (a0, b) = stack.normal_equations(keep);

        // Collect the fold's test rows once: (target, row entries).
        let mut test_rows: Vec<(f64, Vec<(usize, f64)>)> = Vec::new();
        for h in 0..=g.hmax {
            for t in g.p..n - h {
                if t < lo || t >= hi {
                    continue;
                }
                stack.row(h, t, &mut entries);
                test_rows.push((stack.y[t + h], entries.clone()));
            }
        }
        n_test += test_rows.len();

        for (gi, &lambda) in grid.iter().enumerate() {
            let a = add_penalty(&a0, pen, lambda);
            let theta = a
                .llt(Side::Lower)
                .map_err(|_| LpError::CvConfig {
                    n_folds,
                    n_base: nb,
                })?
                .inverse()
                * &b;
            for (target, row) in &test_rows {
                let mut fit = 0.0;
                for &(i, vi) in row {
                    fit += vi * theta[(i, 0)];
                }
                let err = target - fit;
                sse[gi] += err * err;
            }
        }
    }
    if n_test == 0 {
        return Err(LpError::CvConfig {
            n_folds,
            n_base: nb,
        });
    }
    Ok(sse.iter().map(|s| s / n_test as f64).collect())
}
