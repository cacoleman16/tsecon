//! The constrained-domain reparameterization toolkit.
//!
//! Every model crate optimizes in unconstrained space: a [`Transform`] maps
//! the optimizer's working vector `z` in `R^n` to the model's constrained
//! parameter `theta` (positive variances, bounded persistences, ordered
//! thresholds, stationary AR coefficients), and [`TransformedObjective`]
//! lets any optimizer in this crate work through that map.
//!
//! Each transform supplies `forward` (unconstrained to constrained),
//! `inverse` (exact inverse, for turning user-supplied starting values into
//! working coordinates), and `log_jacobian` — `log |det d theta / d z|` of
//! the forward map, needed whenever a *density* moves across the
//! reparameterization (Bayesian priors on `theta` evaluated in `z`;
//! omitting it is the classic silent bug flagged in the roadmap, docs/roadmap
//! 00, "constrained reparameterization toolkit").
//!
//! Boundary honesty: an optimum on the boundary of the constrained domain
//! (a variance at zero, GARCH persistence at one) is pushed to `z` at
//! +/- infinity in working space, so the optimizer reports line-search
//! failure or step stagnation rather than a fake interior optimum. Detect
//! this by checking `|z|` of the solution before mapping back.
//!
//! `// TODO(phase0)`: delta-method Jacobians (`d theta / d z` as a matrix)
//! for standard errors in natural coordinates, and analytic
//! gradient chain-rule support in [`TransformedObjective`].

mod monahan;
mod simple;

pub use monahan::StationaryAr;
pub use simple::{Bounded, Ordered, Positive, UnitInterval};

use crate::error::OptimError;
use crate::objective::ObjectiveFn;

/// A smooth bijection between unconstrained working space `R^n` and a
/// constrained parameter domain.
///
/// All transforms in this crate preserve length (`z` and `theta` have the
/// same number of elements) and accept any `n >= 0` unless documented
/// otherwise; slices of length zero are a valid degenerate case (e.g. an
/// AR(0) block) with `log_jacobian = 0`.
pub trait Transform {
    /// Maps unconstrained `z` into the constrained domain, writing into
    /// `theta`.
    ///
    /// # Errors
    ///
    /// [`OptimError::DimensionMismatch`] if `theta.len() != z.len()`;
    /// [`OptimError::NonFinite`] if `z` contains NaN/infinity.
    fn forward(&self, z: &[f64], theta: &mut [f64]) -> Result<(), OptimError>;

    /// Maps constrained `theta` back to unconstrained space, writing into
    /// `z`. Exact inverse of [`forward`](Transform::forward) up to
    /// rounding.
    ///
    /// # Errors
    ///
    /// [`OptimError::DimensionMismatch`] on length mismatch;
    /// [`OptimError::NonFinite`] on non-finite input; [`OptimError::Domain`]
    /// or [`OptimError::NotStationary`] if `theta` is outside the
    /// constrained domain (including exactly on its boundary, which maps to
    /// infinite `z`).
    fn inverse(&self, theta: &[f64], z: &mut [f64]) -> Result<(), OptimError>;

    /// `log |det (d theta / d z)|` of the forward map at `z`.
    ///
    /// # Errors
    ///
    /// [`OptimError::NonFinite`] if `z` contains NaN/infinity.
    fn log_jacobian(&self, z: &[f64]) -> Result<f64, OptimError>;

    /// Convenience allocating wrapper around [`forward`](Transform::forward).
    ///
    /// # Errors
    ///
    /// As for [`forward`](Transform::forward).
    fn forward_vec(&self, z: &[f64]) -> Result<Vec<f64>, OptimError> {
        let mut theta = vec![0.0; z.len()];
        self.forward(z, &mut theta)?;
        Ok(theta)
    }

    /// Convenience allocating wrapper around [`inverse`](Transform::inverse).
    ///
    /// # Errors
    ///
    /// As for [`inverse`](Transform::inverse).
    fn inverse_vec(&self, theta: &[f64]) -> Result<Vec<f64>, OptimError> {
        let mut z = vec![0.0; theta.len()];
        self.inverse(theta, &mut z)?;
        Ok(z)
    }
}

pub(crate) fn check_lengths(
    what: &'static str,
    expected: usize,
    actual: usize,
) -> Result<(), OptimError> {
    if expected != actual {
        return Err(OptimError::DimensionMismatch {
            what,
            expected,
            actual,
        });
    }
    Ok(())
}

pub(crate) fn check_finite(what: &'static str, x: &[f64]) -> Result<(), OptimError> {
    if x.iter().any(|v| !v.is_finite()) {
        return Err(OptimError::NonFinite { what });
    }
    Ok(())
}

/// Lets any optimizer work in unconstrained space: wraps an objective
/// defined on the constrained domain together with a [`Transform`], and
/// evaluates `z -> f(forward(z))`.
///
/// Two modes:
///
/// * [`new`](TransformedObjective::new) — plain reparameterized
///   minimization (the MLE case): `value(z) = f(theta(z))`.
/// * [`with_log_jacobian`](TransformedObjective::with_log_jacobian) — for
///   minimizing a *negative log density* of `theta` (Bayesian MAP): the
///   change of variables requires
///   `value(z) = f(theta(z)) - log |det d theta / d z|`.
///
/// If the transform rejects `z` or maps it to a non-finite `theta` (e.g.
/// `exp` overflow far out in working space), the value is `+infinity`, which
/// every optimizer here treats as an infeasible trial — no panics, no
/// errors mid-search.
///
/// The wrapper reports no analytic gradient (the optimizers difference
/// numerically in `z`); `// TODO(phase0)`: chain analytic gradients
/// through the transform Jacobian.
///
/// ```
/// use tsecon_optim::{
///     minimize, FnObjective, Method, Positive, Transform, TransformedObjective,
/// };
///
/// // Minimize f(theta) = (theta - 3)^2 subject to theta > 0, working in z.
/// let inner = FnObjective::new(|t: &[f64]| (t[0] - 3.0) * (t[0] - 3.0));
/// let mut obj = TransformedObjective::new(inner, Positive);
/// let res = minimize(&mut obj, &[0.0], &Method::nelder_mead()).unwrap();
/// let theta = obj.constrained(&res.x).unwrap();
/// assert!((theta[0] - 3.0).abs() < 1e-6);
/// ```
#[derive(Debug, Clone)]
pub struct TransformedObjective<F, T> {
    inner: F,
    transform: T,
    include_log_jacobian: bool,
    buf: Vec<f64>,
}

impl<F: ObjectiveFn, T: Transform> TransformedObjective<F, T> {
    /// Plain reparameterized objective: `value(z) = inner(forward(z))`.
    pub fn new(inner: F, transform: T) -> Self {
        Self {
            inner,
            transform,
            include_log_jacobian: false,
            buf: Vec::new(),
        }
    }

    /// Density-aware objective for Bayesian MAP / posterior work:
    /// `value(z) = inner(forward(z)) - log_jacobian(z)`, where `inner` is a
    /// negative log density in `theta`.
    pub fn with_log_jacobian(inner: F, transform: T) -> Self {
        Self {
            inner,
            transform,
            include_log_jacobian: true,
            buf: Vec::new(),
        }
    }

    /// Maps a working-space point (e.g. the optimizer's solution) to the
    /// constrained domain.
    ///
    /// # Errors
    ///
    /// As for [`Transform::forward`].
    pub fn constrained(&self, z: &[f64]) -> Result<Vec<f64>, OptimError> {
        self.transform.forward_vec(z)
    }

    /// The wrapped objective.
    pub fn inner(&self) -> &F {
        &self.inner
    }

    /// The transform in use.
    pub fn transform(&self) -> &T {
        &self.transform
    }

    /// Consumes the wrapper, returning the inner objective.
    pub fn into_inner(self) -> F {
        self.inner
    }
}

impl<F: ObjectiveFn, T: Transform> ObjectiveFn for TransformedObjective<F, T> {
    fn value(&mut self, z: &[f64]) -> f64 {
        self.buf.resize(z.len(), 0.0);
        let mut theta = core::mem::take(&mut self.buf);
        let v = match self.transform.forward(z, &mut theta) {
            Ok(()) if theta.iter().all(|v| v.is_finite()) => {
                let base = self.inner.value(&theta);
                if self.include_log_jacobian {
                    match self.transform.log_jacobian(z) {
                        Ok(lj) => base - lj,
                        Err(_) => f64::INFINITY,
                    }
                } else {
                    base
                }
            }
            _ => f64::INFINITY,
        };
        self.buf = theta;
        v
    }
}
