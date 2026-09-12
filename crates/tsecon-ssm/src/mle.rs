//! Maximum-likelihood plumbing shared by the structural-model
//! ([`crate::uc`]) and TVP-regression ([`crate::tvp`]) layers: the
//! working-space reparameterization, the optimizer schedule, the boundary
//! (variance pile-up) check, and observed-information standard errors.
//!
//! # Working space
//!
//! Following statsmodels' `UnobservedComponents.transform_params`, a
//! variance is optimized as the square of an unconstrained working value
//! (`sigma2 = theta^2`), a bounded parameter (cycle frequency, damping)
//! through a logistic map onto its interval, and a regression coefficient
//! as itself. The square map keeps a variance that wants to be zero at a
//! *regular* interior point of the working space (`theta = 0`, zero
//! gradient) instead of at `-infinity` of a log map, which is what makes a
//! boundary optimum reachable by a quasi-Newton search at all.
//!
//! # Schedule
//!
//! From each start: BFGS with central-difference gradients (the
//! strong-Wolfe line search of `tsecon-optim`), then an adaptive
//! Nelder-Mead polish from the BFGS point. The best of the starts wins.
//! Starts are deterministic scalings of one heuristic start (no
//! randomness, hence no seed), chosen by the calling estimator.
//!
//! # Boundary (pile-up) check
//!
//! A variance is reported *at the boundary* when setting it to exactly
//! zero — every other parameter held at its estimate — lowers the
//! log-likelihood by less than [`BOUNDARY_LL_TOL`] (an absolute
//! log-likelihood difference, hence invariant to the units of `y`): the
//! likelihood cannot tell the estimate from zero, which is the pile-up
//! phenomenon of Shephard & Harvey (1990) and Stock & Watson (1998). A
//! bounded parameter within `1e-6` of the width of its interval from
//! either end is flagged the same way. Flagged parameters get a NaN
//! standard error: the observed information at a boundary is not a
//! curvature.
//!
//! # Standard errors
//!
//! Observed-information standard errors: the inverse of the four-point
//! central-difference Hessian of the negative log-likelihood in the
//! *constrained* parameter space (the `approx_hess3` formulas statsmodels
//! uses for `cov_type="approx"`), restricted to the parameters not at a
//! boundary. A singular or non-finite Hessian gives NaN for every standard
//! error rather than a number computed from a failed inverse.

use tsecon_optim::{minimize, BfgsOptions, FnObjective, Method, NelderMeadOptions, ObjectiveFn};

use crate::error::SsmError;

/// Absolute log-likelihood drop below which a variance set to zero is
/// indistinguishable from its estimate (the pile-up flag).
pub(crate) const BOUNDARY_LL_TOL: f64 = 1e-4;

/// Fraction of a bounded parameter's interval width within which an
/// estimate counts as sitting on the bound.
const BOUNDED_EDGE_FRACTION: f64 = 1e-6;

/// How one parameter is mapped between the working and constrained spaces.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum ParamKind {
    /// A variance: constrained value `theta^2`.
    Variance,
    /// Bounded in `(low, high)` through a logistic map.
    Bounded {
        /// Lower bound.
        low: f64,
        /// Upper bound.
        high: f64,
    },
    /// Unconstrained.
    Free,
}

/// One estimated parameter: its reporting name and its kind.
#[derive(Debug, Clone)]
pub(crate) struct ParamSpec {
    pub(crate) name: String,
    pub(crate) kind: ParamKind,
}

#[inline]
fn logistic(w: f64) -> f64 {
    if w >= 0.0 {
        1.0 / (1.0 + (-w).exp())
    } else {
        let e = w.exp();
        e / (1.0 + e)
    }
}

/// Working-space values to constrained parameters.
pub(crate) fn to_constrained(specs: &[ParamSpec], working: &[f64]) -> Vec<f64> {
    specs
        .iter()
        .zip(working)
        .map(|(s, &w)| match s.kind {
            ParamKind::Variance => w * w,
            ParamKind::Bounded { low, high } => low + (high - low) * logistic(w),
            ParamKind::Free => w,
        })
        .collect()
}

/// Constrained parameters to working-space values (the inverse map;
/// bounded values are clamped strictly inside their interval first).
pub(crate) fn to_working(specs: &[ParamSpec], constrained: &[f64]) -> Vec<f64> {
    specs
        .iter()
        .zip(constrained)
        .map(|(s, &c)| match s.kind {
            ParamKind::Variance => c.max(0.0).sqrt(),
            ParamKind::Bounded { low, high } => {
                let x = ((c - low) / (high - low)).clamp(1e-9, 1.0 - 1e-9);
                (x / (1.0 - x)).ln()
            }
            ParamKind::Free => c,
        })
        .collect()
}

/// The outcome of [`maximize`].
#[derive(Debug, Clone)]
pub(crate) struct MleOutcome {
    /// Constrained parameter estimates.
    pub(crate) params: Vec<f64>,
    /// Whether the winning run's optimizer certified convergence.
    pub(crate) converged: bool,
    /// Iterations summed over every start and stage.
    pub(crate) n_iter: usize,
    /// Objective evaluations summed over every start and stage.
    pub(crate) n_fevals: usize,
    /// Per-parameter boundary (pile-up) flags.
    pub(crate) at_boundary: Vec<bool>,
    /// Observed-information standard errors (NaN at a boundary or when
    /// the Hessian is singular / non-finite).
    pub(crate) se: Vec<f64>,
}

/// Maximizes `loglik` over the constrained parameters from every start in
/// `starts` (constrained coordinates), then runs the boundary check and
/// computes standard errors. `loglik` returns NaN or `-inf` where the
/// likelihood is undefined.
pub(crate) fn maximize<F>(
    specs: &[ParamSpec],
    starts: &[Vec<f64>],
    mut loglik: F,
) -> Result<MleOutcome, SsmError>
where
    F: FnMut(&[f64]) -> f64,
{
    let k = specs.len();
    // (working x, negative loglik, converged, iterations, fevals)
    let mut best: Option<(Vec<f64>, f64, bool, usize, usize)> = None;
    let bfgs = Method::Bfgs(BfgsOptions {
        max_iter: Some(500),
        ..BfgsOptions::default()
    });
    let nm = Method::NelderMead(NelderMeadOptions {
        max_iter: Some(4000),
        ..NelderMeadOptions::default()
    });
    for start in starts {
        let w0 = to_working(specs, start);
        let mut obj = FnObjective::new(|w: &[f64]| {
            let c = to_constrained(specs, w);
            let ll = loglik(&c);
            if ll.is_finite() {
                -ll
            } else {
                f64::INFINITY
            }
        });
        let f0 = obj.value(&w0);
        if !f0.is_finite() {
            continue;
        }
        let (mut x, mut f, mut conv, mut iters, mut fevals) = (w0, f0, false, 0usize, 1usize);
        if let Ok(r) = minimize(&mut obj, &x, &bfgs) {
            iters += r.iterations;
            fevals += r.fevals;
            if r.f.is_finite() && r.f <= f {
                x = r.x;
                f = r.f;
                conv = r.converged;
            }
        }
        if let Ok(r) = minimize(&mut obj, &x, &nm) {
            iters += r.iterations;
            fevals += r.fevals;
            if r.f.is_finite() && r.f <= f {
                x = r.x;
                f = r.f;
                conv = conv || r.converged;
            }
        }
        let better = match &best {
            None => true,
            Some((_, bf, _, _, _)) => f < *bf,
        };
        if better {
            best = Some((x, f, conv, iters, fevals));
        } else if let Some(b) = best.as_mut() {
            b.3 += iters;
            b.4 += fevals;
        }
    }
    let (x, f, converged, n_iter, n_fevals) = best.ok_or_else(|| SsmError::InvalidSpec {
        message: "the log-likelihood is not finite at any starting value: check y for \
                  a (near-)constant series or an unidentified component"
            .to_string(),
    })?;
    let params = to_constrained(specs, &x);
    let ll_hat = -f;

    let mut at_boundary = vec![false; k];
    for (i, spec) in specs.iter().enumerate() {
        at_boundary[i] = match spec.kind {
            ParamKind::Variance => {
                if params[i] <= 0.0 {
                    true
                } else {
                    let mut p0 = params.clone();
                    p0[i] = 0.0;
                    let ll0 = loglik(&p0);
                    ll0.is_finite() && ll_hat - ll0 < BOUNDARY_LL_TOL
                }
            }
            ParamKind::Bounded { low, high } => {
                let edge = BOUNDED_EDGE_FRACTION * (high - low);
                params[i] - low < edge || high - params[i] < edge
            }
            ParamKind::Free => false,
        };
    }
    let se = standard_errors(specs, &params, &at_boundary, &mut loglik);
    Ok(MleOutcome {
        params,
        converged,
        n_iter,
        n_fevals,
        at_boundary,
        se,
    })
}

/// Observed-information standard errors restricted to the parameters not
/// at a boundary (see the module docs).
fn standard_errors<F>(
    specs: &[ParamSpec],
    params: &[f64],
    at_boundary: &[bool],
    loglik: &mut F,
) -> Vec<f64>
where
    F: FnMut(&[f64]) -> f64,
{
    let k = specs.len();
    let mut se = vec![f64::NAN; k];
    let free: Vec<usize> = (0..k).filter(|&i| !at_boundary[i]).collect();
    if free.is_empty() {
        return se;
    }
    let eps4 = f64::EPSILON.powf(0.25);
    let h: Vec<f64> = free
        .iter()
        .map(|&i| {
            let x = params[i];
            match specs[i].kind {
                ParamKind::Variance => eps4 * x.abs(),
                ParamKind::Bounded { low, high } => {
                    let hh = eps4 * x.abs().max(1e-2);
                    hh.min(0.5 * (x - low)).min(0.5 * (high - x))
                }
                ParamKind::Free => eps4 * x.abs().max(1.0),
            }
        })
        .collect();
    if h.iter().any(|&v| !(v.is_finite() && v > 0.0)) {
        return se;
    }
    let m = free.len();
    let mut probe = params.to_vec();
    let mut eval = |di: (usize, f64), dj: (usize, f64)| -> f64 {
        probe.copy_from_slice(params);
        probe[free[di.0]] += di.1;
        probe[free[dj.0]] += dj.1;
        -loglik(&probe)
    };
    let mut hess = vec![vec![0.0; m]; m];
    for i in 0..m {
        for j in i..m {
            let fpp = eval((i, h[i]), (j, h[j]));
            let fpm = eval((i, h[i]), (j, -h[j]));
            let fmp = eval((i, -h[i]), (j, h[j]));
            let fmm = eval((i, -h[i]), (j, -h[j]));
            let v = ((fpp - fpm) - (fmp - fmm)) / (4.0 * h[i] * h[j]);
            if !v.is_finite() {
                return se;
            }
            hess[i][j] = v;
            hess[j][i] = v;
        }
    }
    let inv = match invert_symmetric(&hess) {
        Some(v) => v,
        None => return se,
    };
    for (a, &i) in free.iter().enumerate() {
        let v = inv[a][a];
        se[i] = if v.is_finite() && v > 0.0 {
            v.sqrt()
        } else {
            f64::NAN
        };
    }
    se
}

/// Gauss-Jordan inverse with partial pivoting of a small square matrix;
/// `None` when a pivot is zero or non-finite.
pub(crate) fn invert_symmetric(a: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = a.len();
    let mut m: Vec<Vec<f64>> = a
        .iter()
        .enumerate()
        .map(|(i, row)| {
            let mut r = row.clone();
            r.extend((0..n).map(|j| if i == j { 1.0 } else { 0.0 }));
            r
        })
        .collect();
    for col in 0..n {
        let mut piv = col;
        let mut best = m[col][col].abs();
        for (r, row) in m.iter().enumerate().skip(col + 1) {
            if row[col].abs() > best {
                best = row[col].abs();
                piv = r;
            }
        }
        if !(best.is_finite() && best > 0.0) {
            return None;
        }
        m.swap(col, piv);
        let d = m[col][col];
        for v in m[col].iter_mut() {
            *v /= d;
        }
        let pivot_row = m[col].clone();
        for (r, row) in m.iter_mut().enumerate() {
            if r == col {
                continue;
            }
            let factor = row[col];
            if factor != 0.0 {
                for (v, p) in row.iter_mut().zip(&pivot_row) {
                    *v -= factor * p;
                }
            }
        }
    }
    let inv: Vec<Vec<f64>> = m.into_iter().map(|row| row[n..].to_vec()).collect();
    if inv.iter().flatten().any(|v| !v.is_finite()) {
        return None;
    }
    Some(inv)
}

/// Sample variance (divisor `n`) of a slice; `0.0` for fewer than two
/// values.
pub(crate) fn variance(xs: &[f64]) -> f64 {
    let n = xs.len();
    if n < 2 {
        return 0.0;
    }
    let mean = xs.iter().sum::<f64>() / n as f64;
    xs.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n as f64
}

/// OLS coefficients of `y` on the columns `x_cols` over the rows `rows`
/// (normal equations); `None` when `X'X` is singular.
pub(crate) fn ols(x_cols: &[Vec<f64>], y: &[f64], rows: &[usize]) -> Option<Vec<f64>> {
    let k = x_cols.len();
    if k == 0 {
        return Some(Vec::new());
    }
    let mut xtx = vec![vec![0.0; k]; k];
    let mut xty = vec![0.0; k];
    for &t in rows {
        for i in 0..k {
            let xi = x_cols[i][t];
            xty[i] += xi * y[t];
            for j in 0..k {
                xtx[i][j] += xi * x_cols[j][t];
            }
        }
    }
    let inv = invert_symmetric(&xtx)?;
    Some(
        inv.iter()
            .map(|row| row.iter().zip(&xty).map(|(a, b)| a * b).sum())
            .collect(),
    )
}
