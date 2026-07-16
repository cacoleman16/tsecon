//! The objective-function trait, closure adapters, and the
//! central-difference numerical gradient.

use crate::error::OptimError;

/// A scalar objective `f: R^n -> R` to be *minimized*, with an optional
/// analytic gradient.
///
/// Implementors take `&mut self` so model crates can reuse internal
/// workspaces (e.g. Kalman-filter arenas) across evaluations without
/// interior mutability.
///
/// # Non-finite values
///
/// Every optimizer in this crate treats a non-finite objective value (NaN or
/// +/- infinity) as "infeasible point": the trial is rejected and the search
/// continues elsewhere. Objectives should therefore return NaN or infinity
/// for out-of-domain inputs rather than panic.
pub trait ObjectiveFn {
    /// The objective value at `x`.
    fn value(&mut self, x: &[f64]) -> f64;

    /// The analytic gradient at `x`, or `None` if unavailable.
    ///
    /// When this returns `None` (the default), the gradient-based optimizers
    /// fall back to the central-difference helper
    /// [`central_difference_gradient`], which costs `2 n` extra
    /// [`value`](ObjectiveFn::value) evaluations per gradient.
    fn gradient(&mut self, x: &[f64]) -> Option<Vec<f64>> {
        let _ = x;
        None
    }
}

impl<F: ObjectiveFn + ?Sized> ObjectiveFn for &mut F {
    fn value(&mut self, x: &[f64]) -> f64 {
        (**self).value(x)
    }

    fn gradient(&mut self, x: &[f64]) -> Option<Vec<f64>> {
        (**self).gradient(x)
    }
}

/// Adapts a closure `f: FnMut(&[f64]) -> f64` into an [`ObjectiveFn`]
/// without an analytic gradient.
///
/// ```
/// use tsecon_optim::{FnObjective, ObjectiveFn};
/// let mut sphere = FnObjective::new(|x: &[f64]| x.iter().map(|v| v * v).sum());
/// assert_eq!(sphere.value(&[3.0, 4.0]), 25.0);
/// assert!(sphere.gradient(&[3.0, 4.0]).is_none());
/// ```
#[derive(Debug, Clone)]
pub struct FnObjective<F> {
    f: F,
}

impl<F: FnMut(&[f64]) -> f64> FnObjective<F> {
    /// Wraps the closure `f`.
    pub fn new(f: F) -> Self {
        Self { f }
    }
}

impl<F: FnMut(&[f64]) -> f64> ObjectiveFn for FnObjective<F> {
    fn value(&mut self, x: &[f64]) -> f64 {
        (self.f)(x)
    }
}

/// Adapts a value closure and an analytic-gradient closure into an
/// [`ObjectiveFn`].
///
/// ```
/// use tsecon_optim::{FnObjectiveGrad, ObjectiveFn};
/// let mut sphere = FnObjectiveGrad::new(
///     |x: &[f64]| x.iter().map(|v| v * v).sum(),
///     |x: &[f64]| x.iter().map(|v| 2.0 * v).collect(),
/// );
/// assert_eq!(sphere.gradient(&[3.0, 4.0]), Some(vec![6.0, 8.0]));
/// ```
#[derive(Debug, Clone)]
pub struct FnObjectiveGrad<F, G> {
    f: F,
    g: G,
}

impl<F, G> FnObjectiveGrad<F, G>
where
    F: FnMut(&[f64]) -> f64,
    G: FnMut(&[f64]) -> Vec<f64>,
{
    /// Wraps the value closure `f` and gradient closure `g`.
    pub fn new(f: F, g: G) -> Self {
        Self { f, g }
    }
}

impl<F, G> ObjectiveFn for FnObjectiveGrad<F, G>
where
    F: FnMut(&[f64]) -> f64,
    G: FnMut(&[f64]) -> Vec<f64>,
{
    fn value(&mut self, x: &[f64]) -> f64 {
        (self.f)(x)
    }

    fn gradient(&mut self, x: &[f64]) -> Option<Vec<f64>> {
        Some((self.g)(x))
    }
}

/// Central-difference numerical gradient,
/// `g_i = (f(x + h_i e_i) - f(x - h_i e_i)) / (2 h_i)`.
///
/// The step is `h_i = cbrt(eps) * max(1, |x_i|)` with `eps` the f64 machine
/// epsilon — the step that balances the `O(h^2)` truncation error of the
/// central difference against the `O(eps / h)` rounding error, giving a
/// total error of order `eps^(2/3) ~ 4e-11` relative to the local scale of
/// `f` (Nocedal-Wright 2006, section 8.1). The actual step used is the
/// exactly representable `(x_i + h_i) - x_i`.
///
/// Costs `2 n` objective evaluations. Entries are NaN/infinite when `f` is
/// non-finite at a probe point; callers that cannot tolerate that should
/// check `is_finite` on the result.
pub fn central_difference_gradient<F: ObjectiveFn + ?Sized>(f: &mut F, x: &[f64]) -> Vec<f64> {
    let n = x.len();
    let mut grad = vec![0.0; n];
    let mut work = x.to_vec();
    let base = f64::EPSILON.cbrt();
    for i in 0..n {
        let h = base * x[i].abs().max(1.0);
        // Make the step exactly representable so the divisor is exact.
        let xp = x[i] + h;
        let xm = x[i] - h;
        work[i] = xp;
        let fp = f.value(&work);
        work[i] = xm;
        let fm = f.value(&work);
        work[i] = x[i];
        grad[i] = (fp - fm) / (xp - xm);
    }
    grad
}

/// Evaluates the gradient of `f` at `x`: the analytic gradient when
/// [`ObjectiveFn::gradient`] provides one (validated for length), otherwise
/// the [`central_difference_gradient`].
///
/// # Errors
///
/// [`OptimError::DimensionMismatch`] if an analytic gradient of the wrong
/// length is returned.
pub fn eval_gradient<F: ObjectiveFn + ?Sized>(
    f: &mut F,
    x: &[f64],
) -> Result<Vec<f64>, OptimError> {
    match f.gradient(x) {
        Some(g) => {
            if g.len() != x.len() {
                return Err(OptimError::DimensionMismatch {
                    what: "analytic gradient",
                    expected: x.len(),
                    actual: g.len(),
                });
            }
            Ok(g)
        }
        None => Ok(central_difference_gradient(f, x)),
    }
}

/// Internal wrapper that counts objective and analytic-gradient evaluations
/// and maps non-finite objective values to `+infinity` so ordering
/// comparisons are total.
pub(crate) struct Counted<'a, F: ?Sized> {
    inner: &'a mut F,
    /// Number of `value` evaluations (numerical-gradient probes included).
    pub(crate) fevals: usize,
    /// Number of analytic-gradient evaluations.
    pub(crate) gevals: usize,
}

impl<'a, F: ObjectiveFn + ?Sized> Counted<'a, F> {
    pub(crate) fn new(inner: &'a mut F) -> Self {
        Self {
            inner,
            fevals: 0,
            gevals: 0,
        }
    }

    /// Gradient with evaluation counting (analytic if available, else
    /// central differences whose probes count as fevals).
    pub(crate) fn grad(&mut self, x: &[f64]) -> Result<Vec<f64>, OptimError> {
        eval_gradient(self, x)
    }
}

impl<F: ObjectiveFn + ?Sized> ObjectiveFn for Counted<'_, F> {
    fn value(&mut self, x: &[f64]) -> f64 {
        self.fevals += 1;
        let v = self.inner.value(x);
        if v.is_finite() {
            v
        } else {
            f64::INFINITY
        }
    }

    fn gradient(&mut self, x: &[f64]) -> Option<Vec<f64>> {
        let g = self.inner.gradient(x)?;
        self.gevals += 1;
        Some(g)
    }
}
