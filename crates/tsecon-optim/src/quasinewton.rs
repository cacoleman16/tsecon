//! BFGS and L-BFGS quasi-Newton minimizers with a strong-Wolfe line
//! search.
//!
//! * **BFGS** maintains the dense inverse-Hessian approximation `H_k` and
//!   updates it by the rank-two formula (Nocedal-Wright 2006, eq. 6.17)
//!
//!   ```text
//!   H_{k+1} = (I - rho s y') H_k (I - rho y s') + rho s s',
//!   rho = 1 / (y's),  s = x_{k+1} - x_k,  y = g_{k+1} - g_k
//!   ```
//!
//!   with the self-scaling `H_0 <- (y's / y'y) I` applied before the first
//!   update (Nocedal-Wright 2006, eq. 6.20).
//! * **L-BFGS** stores only the last `m` (default 10) pairs `(s, y)` and
//!   computes `H_k g` by the two-loop recursion (Nocedal-Wright 2006,
//!   algorithm 7.4) with the scaling `gamma_k = s'y / y'y` (eq. 7.20).
//!
//! Both use the strong-Wolfe line search of [`crate::strong_wolfe`]
//! (`c1 = 1e-4`, `c2 = 0.9`), which guarantees `y's > 0` on accepted steps
//! so the updates preserve positive definiteness (Nocedal-Wright 2006,
//! section 6.1). The gradient is the caller's analytic one when
//! [`ObjectiveFn::gradient`] supplies it, otherwise the central-difference
//! helper [`crate::central_difference_gradient`].

use crate::error::OptimError;
use crate::linesearch::{strong_wolfe, WolfeOptions};
use crate::objective::{Counted, ObjectiveFn};
use crate::result::{OptimizeResult, Termination};

/// Options for [`bfgs`].
#[derive(Debug, Clone, Copy)]
pub struct BfgsOptions {
    /// Convergence when `||g||_inf <= grad_tol`; default `1e-8`.
    pub grad_tol: f64,
    /// Optional relative function-decrease test: stop with
    /// [`Termination::FunctionTolerance`] when
    /// `f_k - f_{k+1} <= f_tol * max(1, |f_{k+1}|)`. Disabled (0) by
    /// default — on ill-conditioned problems it can trigger long before the
    /// gradient is small.
    pub f_tol: f64,
    /// Optional relative step test: stop with
    /// [`Termination::StepTolerance`] when
    /// `||x_{k+1} - x_k||_inf <= x_tol * max(1, ||x_{k+1}||_inf)`.
    /// Disabled (0) by default.
    pub x_tol: f64,
    /// Iteration budget; `None` (default) means `200 * n`.
    pub max_iter: Option<usize>,
    /// Objective-evaluation budget (central-difference gradient probes
    /// included), checked between iterations; `None` (default) means no
    /// limit beyond `max_iter`.
    pub max_fevals: Option<usize>,
    /// Strong-Wolfe line-search constants (defaults `c1 = 1e-4`,
    /// `c2 = 0.9`).
    pub line_search: WolfeOptions,
}

impl Default for BfgsOptions {
    fn default() -> Self {
        Self {
            grad_tol: 1e-8,
            f_tol: 0.0,
            x_tol: 0.0,
            max_iter: None,
            max_fevals: None,
            line_search: WolfeOptions::default(),
        }
    }
}

/// Options for [`lbfgs`].
#[derive(Debug, Clone, Copy)]
pub struct LbfgsOptions {
    /// Number of correction pairs stored; default 10 (Nocedal-Wright 2006
    /// recommend 3-20; ~10 suits smooth likelihood surfaces).
    pub memory: usize,
    /// Convergence when `||g||_inf <= grad_tol`; default `1e-8`.
    pub grad_tol: f64,
    /// Optional relative function-decrease test (see
    /// [`BfgsOptions::f_tol`]); disabled (0) by default.
    pub f_tol: f64,
    /// Optional relative step test (see [`BfgsOptions::x_tol`]); disabled
    /// (0) by default.
    pub x_tol: f64,
    /// Iteration budget; `None` (default) means `200 * n`.
    pub max_iter: Option<usize>,
    /// Objective-evaluation budget, checked between iterations; `None`
    /// (default) means no limit beyond `max_iter`.
    pub max_fevals: Option<usize>,
    /// Strong-Wolfe line-search constants (defaults `c1 = 1e-4`,
    /// `c2 = 0.9`).
    pub line_search: WolfeOptions,
}

impl Default for LbfgsOptions {
    fn default() -> Self {
        Self {
            memory: 10,
            grad_tol: 1e-8,
            f_tol: 0.0,
            x_tol: 0.0,
            max_iter: None,
            max_fevals: None,
            line_search: WolfeOptions::default(),
        }
    }
}

fn validate_tols(
    grad_tol: f64,
    f_tol: f64,
    x_tol: f64,
    ls: &WolfeOptions,
) -> Result<(), OptimError> {
    if !(grad_tol >= 0.0 && grad_tol.is_finite()) {
        return Err(OptimError::InvalidOption {
            name: "grad_tol",
            value: grad_tol,
            requirement: "0 <= grad_tol < infinity",
        });
    }
    if !(f_tol >= 0.0 && f_tol.is_finite()) {
        return Err(OptimError::InvalidOption {
            name: "f_tol",
            value: f_tol,
            requirement: "0 <= f_tol < infinity",
        });
    }
    if !(x_tol >= 0.0 && x_tol.is_finite()) {
        return Err(OptimError::InvalidOption {
            name: "x_tol",
            value: x_tol,
            requirement: "0 <= x_tol < infinity",
        });
    }
    // Line-search options are re-validated on every call; validate once
    // here for an early, clearer error.
    if !(ls.c1 > 0.0 && ls.c1 < ls.c2 && ls.c2 < 1.0) {
        return Err(OptimError::InvalidOption {
            name: "line_search.c1/c2",
            value: ls.c1,
            requirement: "0 < c1 < c2 < 1",
        });
    }
    Ok(())
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(&x, &y)| x * y).sum()
}

fn inf_norm(a: &[f64]) -> f64 {
    a.iter().fold(0.0, |m, &v| m.max(v.abs()))
}

/// The inverse-Hessian model shared by the BFGS and L-BFGS drivers.
trait InvHessian {
    /// `d = -H g`.
    fn direction(&self, g: &[f64]) -> Vec<f64>;
    /// Incorporates the pair `(s, y)` with `y's > 0`.
    fn update(&mut self, s: &[f64], y: &[f64]);
}

/// Dense BFGS inverse Hessian, row-major `n x n`.
struct DenseInvHessian {
    n: usize,
    h: Vec<f64>,
    first: bool,
}

impl DenseInvHessian {
    fn new(n: usize) -> Self {
        let mut h = vec![0.0; n * n];
        for i in 0..n {
            h[i * n + i] = 1.0;
        }
        Self { n, h, first: true }
    }
}

impl InvHessian for DenseInvHessian {
    fn direction(&self, g: &[f64]) -> Vec<f64> {
        let n = self.n;
        let mut d = vec![0.0; n];
        for (i, di) in d.iter_mut().enumerate() {
            let row = &self.h[i * n..(i + 1) * n];
            *di = -dot(row, g);
        }
        d
    }

    fn update(&mut self, s: &[f64], y: &[f64]) {
        let n = self.n;
        let sy = dot(s, y);
        if self.first {
            // Self-scaling before the first update (NW 2006, eq. 6.20).
            let yy = dot(y, y);
            if yy > 0.0 && sy > 0.0 {
                let scale = sy / yy;
                for i in 0..n {
                    for j in 0..n {
                        self.h[i * n + j] = if i == j { scale } else { 0.0 };
                    }
                }
            }
            self.first = false;
        }
        let rho = 1.0 / sy;
        // hy = H y  (H symmetric).
        let mut hy = vec![0.0; n];
        for (i, hyi) in hy.iter_mut().enumerate() {
            *hyi = dot(&self.h[i * n..(i + 1) * n], y);
        }
        let yhy = dot(y, &hy);
        // H <- H - rho (s hy' + hy s') + rho^2 yhy s s' + rho s s'
        let coef = rho * rho * yhy + rho;
        for i in 0..n {
            for j in 0..n {
                self.h[i * n + j] += -rho * (s[i] * hy[j] + hy[i] * s[j]) + coef * s[i] * s[j];
            }
        }
    }
}

/// L-BFGS two-loop recursion over the last `m` pairs.
struct LbfgsMemory {
    m: usize,
    pairs: std::collections::VecDeque<(Vec<f64>, Vec<f64>, f64)>, // (s, y, rho)
    gamma: f64,
}

impl LbfgsMemory {
    fn new(m: usize) -> Self {
        Self {
            m,
            pairs: std::collections::VecDeque::with_capacity(m),
            gamma: 1.0,
        }
    }
}

impl InvHessian for LbfgsMemory {
    fn direction(&self, g: &[f64]) -> Vec<f64> {
        let mut q = g.to_vec();
        let mut alphas = Vec::with_capacity(self.pairs.len());
        for (s, y, rho) in self.pairs.iter().rev() {
            let alpha = rho * dot(s, &q);
            for (qi, yi) in q.iter_mut().zip(y) {
                *qi -= alpha * yi;
            }
            alphas.push(alpha);
        }
        for qi in q.iter_mut() {
            *qi *= self.gamma;
        }
        for ((s, y, rho), alpha) in self.pairs.iter().zip(alphas.iter().rev()) {
            let beta = rho * dot(y, &q);
            for (qi, si) in q.iter_mut().zip(s) {
                *qi += (alpha - beta) * si;
            }
        }
        for qi in q.iter_mut() {
            *qi = -*qi;
        }
        q
    }

    fn update(&mut self, s: &[f64], y: &[f64]) {
        let sy = dot(s, y);
        let yy = dot(y, y);
        if yy > 0.0 {
            self.gamma = sy / yy;
        }
        if self.pairs.len() == self.m {
            self.pairs.pop_front();
        }
        self.pairs.push_back((s.to_vec(), y.to_vec(), 1.0 / sy));
    }
}

/// Shared quasi-Newton driver.
#[allow(clippy::too_many_arguments)]
fn quasi_newton_drive<F: ObjectiveFn + ?Sized, H: InvHessian>(
    f: &mut F,
    x0: &[f64],
    mut hess: H,
    grad_tol: f64,
    f_tol: f64,
    x_tol: f64,
    max_iter: Option<usize>,
    max_fevals: Option<usize>,
    ls: &WolfeOptions,
) -> Result<OptimizeResult, OptimError> {
    validate_tols(grad_tol, f_tol, x_tol, ls)?;
    let n = x0.len();
    if n == 0 {
        return Err(OptimError::EmptyInput { what: "x0" });
    }
    if x0.iter().any(|v| !v.is_finite()) {
        return Err(OptimError::NonFinite { what: "x0" });
    }
    let max_iter = max_iter.unwrap_or(200 * n);
    let max_fevals = max_fevals.unwrap_or(usize::MAX);

    let mut c = Counted::new(f);
    let mut x = x0.to_vec();
    let mut fx = c.value(&x);
    if !fx.is_finite() {
        return Err(OptimError::NonFinite { what: "f(x0)" });
    }
    let mut g = c.grad(&x)?;
    if g.iter().any(|v| !v.is_finite()) {
        return Err(OptimError::NonFinite {
            what: "gradient at x0",
        });
    }

    let mut iterations = 0usize;
    let termination;
    let mut first_step = true;

    loop {
        if inf_norm(&g) <= grad_tol {
            termination = Termination::GradientTolerance;
            break;
        }
        if iterations >= max_iter {
            termination = Termination::MaxIterations;
            break;
        }
        if c.fevals >= max_fevals {
            termination = Termination::MaxFevals;
            break;
        }
        iterations += 1;

        let mut d = hess.direction(&g);
        let mut dphi0 = dot(&g, &d);
        if dphi0 >= 0.0 || !dphi0.is_finite() || d.iter().any(|v| !v.is_finite()) {
            // Curvature model broke down; fall back to steepest descent.
            d = g.iter().map(|&v| -v).collect();
            dphi0 = -dot(&g, &g);
            if dphi0 >= 0.0 || !dphi0.is_finite() {
                termination = Termination::LineSearchFailed;
                break;
            }
        }
        // First trial step: unit for quasi-Newton directions, scaled on the
        // very first iteration where H = I gives a raw steepest-descent
        // direction (Nocedal-Wright 2006, p. 142).
        let a0 = if first_step {
            (1.0 / inf_norm(&d).max(1e-30)).min(1.0)
        } else {
            1.0
        };
        let res = strong_wolfe(&mut c, &x, &d, fx, &g, a0, ls)?;
        if !res.success {
            // No acceptable step: keep any improvement, then stop.
            if res.step > 0.0 && res.f < fx {
                x = res.x;
                fx = res.f;
            }
            termination = Termination::LineSearchFailed;
            break;
        }
        if res.g.iter().any(|v| !v.is_finite()) {
            if res.f < fx {
                x = res.x;
                fx = res.f;
            }
            termination = Termination::GradientFailed;
            break;
        }

        let s: Vec<f64> = res.x.iter().zip(&x).map(|(&a, &b)| a - b).collect();
        let y: Vec<f64> = res.g.iter().zip(&g).map(|(&a, &b)| a - b).collect();
        let sy = dot(&s, &y);
        // Strong Wolfe guarantees sy > 0 in exact arithmetic; skip the
        // update if rounding voids it (Nocedal-Wright 2006, section 6.1).
        if sy > 1e-12 * inf_norm(&s) * inf_norm(&y) {
            hess.update(&s, &y);
        }
        first_step = false;

        let f_prev = fx;
        x = res.x;
        fx = res.f;
        g = res.g;

        if f_tol > 0.0 && f_prev - fx <= f_tol * fx.abs().max(1.0) {
            termination = Termination::FunctionTolerance;
            break;
        }
        if x_tol > 0.0 && inf_norm(&s) <= x_tol * inf_norm(&x).max(1.0) {
            termination = Termination::StepTolerance;
            break;
        }
    }

    let converged = termination.converged();
    Ok(OptimizeResult {
        x,
        f: fx,
        iterations,
        fevals: c.fevals,
        gevals: c.gevals,
        converged,
        termination,
    })
}

/// Minimizes `f` by BFGS with a strong-Wolfe line search (Nocedal-Wright
/// 2006, algorithm 6.1). See the module docs for the update formulas and
/// [`BfgsOptions`] for the convergence tests.
///
/// The dense `n x n` inverse Hessian makes BFGS the right choice up to a
/// few hundred parameters; beyond that use [`lbfgs`]. The final inverse
/// Hessian is the standard-error seed for MLE consumers — a phase-1
/// accessor will expose it (`// TODO(phase0)` below).
///
/// # Errors
///
/// * [`OptimError::EmptyInput`] — `x0` is empty;
/// * [`OptimError::NonFinite`] — `x0`, `f(x0)`, or the gradient at `x0` is
///   non-finite;
/// * [`OptimError::InvalidOption`] — malformed options;
/// * [`OptimError::DimensionMismatch`] — the analytic gradient has the
///   wrong length.
pub fn bfgs<F: ObjectiveFn + ?Sized>(
    f: &mut F,
    x0: &[f64],
    opts: &BfgsOptions,
) -> Result<OptimizeResult, OptimError> {
    // TODO(phase0): expose the final inverse-Hessian approximation in the
    // result for covariance/SE consumers (with the usual quasi-Newton
    // caveats), plus the Richardson-extrapolated numerical Hessian module.
    quasi_newton_drive(
        f,
        x0,
        DenseInvHessian::new(x0.len()),
        opts.grad_tol,
        opts.f_tol,
        opts.x_tol,
        opts.max_iter,
        opts.max_fevals,
        &opts.line_search,
    )
}

/// Minimizes `f` by L-BFGS with a strong-Wolfe line search: the two-loop
/// recursion over the last `m` correction pairs (Nocedal-Wright 2006,
/// algorithms 7.4/7.5). Memory and per-iteration cost are `O(m n)`, so it
/// scales to high-dimensional problems where the dense [`bfgs`] matrix is
/// prohibitive.
///
/// # Errors
///
/// As for [`bfgs`], plus [`OptimError::InvalidOption`] if `memory == 0`.
pub fn lbfgs<F: ObjectiveFn + ?Sized>(
    f: &mut F,
    x0: &[f64],
    opts: &LbfgsOptions,
) -> Result<OptimizeResult, OptimError> {
    if opts.memory == 0 {
        return Err(OptimError::InvalidOption {
            name: "memory",
            value: 0.0,
            requirement: "memory >= 1",
        });
    }
    quasi_newton_drive(
        f,
        x0,
        LbfgsMemory::new(opts.memory),
        opts.grad_tol,
        opts.f_tol,
        opts.x_tol,
        opts.max_iter,
        opts.max_fevals,
        &opts.line_search,
    )
}
