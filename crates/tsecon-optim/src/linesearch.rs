//! Strong-Wolfe line search (More-Thuente-style bracketing and zoom).
//!
//! Finds a step `a > 0` along a descent direction `d` satisfying the strong
//! Wolfe conditions (Nocedal-Wright 2006, eq. 3.7; More-Thuente 1994):
//!
//! ```text
//! sufficient decrease:  f(x + a d) <= f(x) + c1 a g0'd        (Armijo)
//! strong curvature:     |g(x + a d)'d| <= c2 |g0'd|
//! ```
//!
//! with `0 < c1 < c2 < 1`. The implementation is the bracketing/zoom scheme
//! of Nocedal-Wright 2006 (algorithms 3.5 and 3.6) with safeguarded cubic
//! interpolation — the same structure More-Thuente (1994) uses for its
//! trial-step selection.

use crate::error::OptimError;
use crate::objective::{eval_gradient, ObjectiveFn};

/// Options for [`strong_wolfe`].
#[derive(Debug, Clone, Copy)]
pub struct WolfeOptions {
    /// Sufficient-decrease (Armijo) constant `c1`; default `1e-4`
    /// (Nocedal-Wright 2006, p. 62).
    pub c1: f64,
    /// Curvature constant `c2`; default `0.9`, the standard choice for
    /// quasi-Newton methods (Nocedal-Wright 2006, p. 62).
    pub c2: f64,
    /// Maximum number of trial evaluations (each trial evaluates the
    /// objective and its gradient once); default 30.
    pub max_evals: usize,
    /// Largest step considered; default `1e10`.
    pub step_max: f64,
}

impl Default for WolfeOptions {
    fn default() -> Self {
        Self {
            c1: 1e-4,
            c2: 0.9,
            max_evals: 30,
            step_max: 1e10,
        }
    }
}

impl WolfeOptions {
    fn validate(&self) -> Result<(), OptimError> {
        if !(self.c1 > 0.0 && self.c1 < 1.0) {
            return Err(OptimError::InvalidOption {
                name: "c1",
                value: self.c1,
                requirement: "0 < c1 < 1",
            });
        }
        if !(self.c2 > self.c1 && self.c2 < 1.0) {
            return Err(OptimError::InvalidOption {
                name: "c2",
                value: self.c2,
                requirement: "c1 < c2 < 1",
            });
        }
        if self.max_evals == 0 {
            return Err(OptimError::InvalidOption {
                name: "max_evals",
                value: 0.0,
                requirement: "max_evals >= 1",
            });
        }
        if !(self.step_max > 0.0 && self.step_max.is_finite()) {
            return Err(OptimError::InvalidOption {
                name: "step_max",
                value: self.step_max,
                requirement: "0 < step_max < infinity",
            });
        }
        Ok(())
    }
}

/// The outcome of [`strong_wolfe`].
#[derive(Debug, Clone)]
pub struct WolfeResult {
    /// `true` if `step` satisfies both strong-Wolfe conditions. When
    /// `false`, the best (lowest-`f`) trial point found is returned; `step`
    /// is `0` and `x`/`f`/`g` echo the inputs if no trial improved on them.
    pub success: bool,
    /// The accepted (or best-found) step length.
    pub step: f64,
    /// The objective value at `x`.
    pub f: f64,
    /// The point `x0 + step * d`.
    pub x: Vec<f64>,
    /// The gradient at `x`.
    pub g: Vec<f64>,
    /// Number of trial evaluations performed (each evaluates the objective
    /// once and the gradient once).
    pub evals: usize,
}

/// One trial evaluation: the point, objective, gradient, and directional
/// derivative at step `a`.
struct Trial {
    a: f64,
    f: f64,
    x: Vec<f64>,
    g: Vec<f64>,
    dphi: f64,
}

/// Evaluation context shared by the bracketing loop and zoom.
struct Ctx<'a, F: ?Sized> {
    f: &'a mut F,
    x0: &'a [f64],
    d: &'a [f64],
    f0: f64,
    g0: &'a [f64],
    dphi0: f64,
    c1: f64,
    c2: f64,
    evals: usize,
    max_evals: usize,
    /// Best (lowest finite f) trial seen, for graceful failure.
    best: Option<Trial>,
}

impl<F: ObjectiveFn + ?Sized> Ctx<'_, F> {
    /// phi(a), phi'(a) at trial step `a`. Non-finite objective values are
    /// mapped to `+infinity` (infeasible).
    fn eval(&mut self, a: f64) -> Result<Trial, OptimError> {
        self.evals += 1;
        let x: Vec<f64> = self
            .x0
            .iter()
            .zip(self.d)
            .map(|(&xi, &di)| xi + a * di)
            .collect();
        let mut fv = self.f.value(&x);
        if !fv.is_finite() {
            fv = f64::INFINITY;
        }
        let (g, dphi) = if fv.is_finite() {
            let g = eval_gradient(self.f, &x)?;
            let dphi = dot(&g, self.d);
            (g, dphi)
        } else {
            // Infeasible point: the gradient is not needed (the trial can
            // only become a bracket endpoint through its f value).
            (vec![f64::NAN; x.len()], f64::NAN)
        };
        let t = Trial {
            a,
            f: fv,
            x,
            g,
            dphi,
        };
        if fv.is_finite() && self.best.as_ref().is_none_or(|b| fv < b.f) {
            self.best = Some(Trial {
                a: t.a,
                f: t.f,
                x: t.x.clone(),
                g: t.g.clone(),
                dphi: t.dphi,
            });
        }
        Ok(t)
    }

    fn armijo_ok(&self, t: &Trial) -> bool {
        t.f <= self.f0 + self.c1 * t.a * self.dphi0
    }

    fn curvature_ok(&self, t: &Trial) -> bool {
        t.dphi.is_finite() && t.dphi.abs() <= -self.c2 * self.dphi0
    }

    fn budget_left(&self) -> bool {
        self.evals < self.max_evals
    }

    fn take_failure(self) -> WolfeResult {
        let evals = self.evals;
        match self.best {
            Some(b) if b.f < self.f0 => WolfeResult {
                success: false,
                step: b.a,
                f: b.f,
                x: b.x,
                g: b.g,
                evals,
            },
            _ => WolfeResult {
                success: false,
                step: 0.0,
                f: self.f0,
                x: self.x0.to_vec(),
                g: self.g0.to_vec(),
                evals,
            },
        }
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(&x, &y)| x * y).sum()
}

/// Safeguarded cubic interpolation for the minimizer of the cubic matching
/// `(f, f')` at two points (Nocedal-Wright 2006, eq. 3.59). Falls back to
/// bisection when the cubic is degenerate or the result leaves the
/// safeguarded interior of the interval.
fn interpolate(a_lo: f64, f_lo: f64, d_lo: f64, a_hi: f64, f_hi: f64, d_hi: f64) -> f64 {
    let mid = 0.5 * (a_lo + a_hi);
    if !(f_lo.is_finite() && f_hi.is_finite() && d_lo.is_finite() && d_hi.is_finite()) {
        return mid;
    }
    let d1 = d_lo + d_hi - 3.0 * (f_lo - f_hi) / (a_lo - a_hi);
    let disc = d1 * d1 - d_lo * d_hi;
    if disc < 0.0 {
        return mid;
    }
    let d2 = (a_hi - a_lo).signum() * disc.sqrt();
    let denom = d_hi - d_lo + 2.0 * d2;
    if denom == 0.0 || !denom.is_finite() {
        return mid;
    }
    let a = a_hi - (a_hi - a_lo) * (d_hi + d2 - d1) / denom;
    // Safeguard: keep the trial strictly inside the bracket.
    let lo = a_lo.min(a_hi);
    let hi = a_lo.max(a_hi);
    let width = hi - lo;
    if !a.is_finite() || a <= lo + 0.05 * width || a >= hi - 0.05 * width {
        mid
    } else {
        a
    }
}

/// Strong-Wolfe line search along the descent direction `d` from `x0`.
///
/// `f0` and `g0` are the objective value and gradient at `x0` (so callers
/// that already have them pay no extra evaluation); `initial_step` is the
/// first trial (use 1 for quasi-Newton directions, Nocedal-Wright 2006,
/// p. 59).
///
/// On success the returned step satisfies both conditions with the `c1`,
/// `c2` of `opts`. On numerical failure (`success == false`) the best trial
/// found is returned so the caller can still make progress or terminate
/// gracefully; see [`WolfeResult`].
///
/// # Errors
///
/// * [`OptimError::DimensionMismatch`] — `d`/`g0` length differs from `x0`;
/// * [`OptimError::EmptyInput`] — `x0` is empty;
/// * [`OptimError::NonFinite`] — `f0`, `g0`, `d`, or `x0` non-finite;
/// * [`OptimError::Domain`] — `d` is not a descent direction
///   (`g0'd >= 0`);
/// * [`OptimError::InvalidOption`] — malformed `opts` or
///   `initial_step <= 0`.
pub fn strong_wolfe<F: ObjectiveFn + ?Sized>(
    f: &mut F,
    x0: &[f64],
    d: &[f64],
    f0: f64,
    g0: &[f64],
    initial_step: f64,
    opts: &WolfeOptions,
) -> Result<WolfeResult, OptimError> {
    opts.validate()?;
    if x0.is_empty() {
        return Err(OptimError::EmptyInput { what: "x0" });
    }
    if d.len() != x0.len() {
        return Err(OptimError::DimensionMismatch {
            what: "direction",
            expected: x0.len(),
            actual: d.len(),
        });
    }
    if g0.len() != x0.len() {
        return Err(OptimError::DimensionMismatch {
            what: "g0",
            expected: x0.len(),
            actual: g0.len(),
        });
    }
    if x0.iter().any(|v| !v.is_finite()) {
        return Err(OptimError::NonFinite { what: "x0" });
    }
    if d.iter().any(|v| !v.is_finite()) {
        return Err(OptimError::NonFinite { what: "direction" });
    }
    if g0.iter().any(|v| !v.is_finite()) {
        return Err(OptimError::NonFinite { what: "g0" });
    }
    if !f0.is_finite() {
        return Err(OptimError::NonFinite { what: "f0" });
    }
    if !(initial_step > 0.0 && initial_step.is_finite()) {
        return Err(OptimError::InvalidOption {
            name: "initial_step",
            value: initial_step,
            requirement: "0 < initial_step < infinity",
        });
    }
    let dphi0 = dot(g0, d);
    if dphi0 >= 0.0 {
        return Err(OptimError::Domain {
            name: "g0'd",
            value: dphi0,
            requirement: "a descent direction (g0'd < 0)",
        });
    }

    let mut ctx = Ctx {
        f,
        x0,
        d,
        f0,
        g0,
        dphi0,
        c1: opts.c1,
        c2: opts.c2,
        evals: 0,
        max_evals: opts.max_evals,
        best: None,
    };

    // Bracketing phase (Nocedal-Wright algorithm 3.5).
    let mut prev: Option<Trial> = None;
    let mut a = initial_step.min(opts.step_max);
    let mut first = true;
    loop {
        if !ctx.budget_left() {
            return Ok(ctx.take_failure());
        }
        let t = ctx.eval(a)?;
        let armijo_fail =
            !ctx.armijo_ok(&t) || (!first && prev.as_ref().is_some_and(|p| t.f >= p.f));
        if armijo_fail {
            // Minimum bracketed between the previous (good) point and `t`.
            let lo = match prev {
                Some(p) => p,
                None => Trial {
                    a: 0.0,
                    f: f0,
                    x: x0.to_vec(),
                    g: g0.to_vec(),
                    dphi: dphi0,
                },
            };
            return zoom(ctx, lo, t);
        }
        // Armijo holds and dphi is finite here (f is finite).
        if ctx.curvature_ok(&t) {
            return Ok(WolfeResult {
                success: true,
                step: t.a,
                f: t.f,
                x: t.x,
                g: t.g,
                evals: ctx.evals,
            });
        }
        if !t.dphi.is_finite() || t.dphi >= 0.0 {
            // Positive slope: minimum bracketed between `t` and `prev`.
            let hi = match prev {
                Some(p) => p,
                None => Trial {
                    a: 0.0,
                    f: f0,
                    x: x0.to_vec(),
                    g: g0.to_vec(),
                    dphi: dphi0,
                },
            };
            return zoom(ctx, t, hi);
        }
        if a >= opts.step_max {
            // Still descending at the cap; give the caller the best point.
            return Ok(ctx.take_failure());
        }
        prev = Some(t);
        a = (2.0 * a).min(opts.step_max);
        first = false;
    }
}

/// Zoom phase (Nocedal-Wright algorithm 3.6): shrink `[lo, hi]` (where `lo`
/// satisfies Armijo with the lowest f seen, and the interval brackets a
/// strong-Wolfe point) until an acceptable step is found.
fn zoom<F: ObjectiveFn + ?Sized>(
    mut ctx: Ctx<'_, F>,
    mut lo: Trial,
    mut hi: Trial,
) -> Result<WolfeResult, OptimError> {
    // Relative interval width below which no representable progress exists.
    const MIN_REL_WIDTH: f64 = 1e-14;
    loop {
        if !ctx.budget_left() {
            return Ok(ctx.take_failure());
        }
        let width = (hi.a - lo.a).abs();
        if width <= MIN_REL_WIDTH * lo.a.abs().max(hi.a.abs()).max(1.0) {
            return Ok(ctx.take_failure());
        }
        let a = interpolate(lo.a, lo.f, lo.dphi, hi.a, hi.f, hi.dphi);
        let t = ctx.eval(a)?;
        if !ctx.armijo_ok(&t) || t.f >= lo.f {
            hi = t;
        } else {
            if ctx.curvature_ok(&t) {
                return Ok(WolfeResult {
                    success: true,
                    step: t.a,
                    f: t.f,
                    x: t.x,
                    g: t.g,
                    evals: ctx.evals,
                });
            }
            if !t.dphi.is_finite() {
                // Cannot trust the slope; treat as a bad endpoint.
                hi = t;
                continue;
            }
            if t.dphi * (hi.a - lo.a) >= 0.0 {
                hi = lo;
            }
            lo = t;
        }
    }
}
