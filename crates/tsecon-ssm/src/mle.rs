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
//! as itself. An autoregressive block is mapped jointly onto the
//! stationarity region through its partial autocorrelations (`tanh` and
//! the Levinson-Durbin recursion, `tsecon_optim::StationaryAr`); statsmodels
//! uses `x / sqrt(1 + x^2)` for the same partial autocorrelations, which
//! changes the search path but not the optimum. The square map keeps a
//! variance that wants to be zero at a *regular* interior point of the
//! working space (`theta = 0`, zero gradient) instead of at `-infinity` of
//! a log map, which is what makes a boundary optimum reachable by a
//! quasi-Newton search at all.
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
//!
//! An autoregressive coefficient's probe step is halved until every probe
//! point `theta +/- h_i +/- h_j` is stationary (the likelihood is undefined
//! outside the region, and one NaN probe would blank every standard
//! error), so an estimate close to the unit root still gets finite
//! standard errors for the other parameters.
//!
//! One identification rule removes more than the flagged parameters (see
//! [`maximize_tied`]): when the disturbance variance of a *stationary*
//! block (an AR component, a stationary-initialized cycle) is flagged at
//! zero, the block is identically zero, so its transition parameters carry
//! exactly zero information; they are dropped from the matrix as well and
//! reported with a NaN standard error instead of making every standard
//! error NaN through a singular Hessian.

use tsecon_optim::{
    minimize, BfgsOptions, FnObjective, Method, NelderMeadOptions, ObjectiveFn, StationaryAr,
    Transform,
};

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
    /// One coefficient of a stationary autoregressive block. The block is
    /// mapped *jointly* (Monahan 1984: `tanh` partial autocorrelations
    /// through the Levinson-Durbin recursion, [`StationaryAr`]), so every
    /// working vector is a stationary AR polynomial. `block_start` marks
    /// the block's first coefficient: a run of `StationaryAr` entries ends
    /// at the next `block_start` or at any other kind, so two adjacent
    /// blocks cannot merge into one polynomial.
    StationaryAr {
        /// True on the first coefficient of the block.
        block_start: bool,
    },
}

/// The `[start, end)` index ranges of the stationary AR blocks.
pub(crate) fn ar_blocks(specs: &[ParamSpec]) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < specs.len() {
        if matches!(specs[i].kind, ParamKind::StationaryAr { .. }) {
            let mut j = i + 1;
            while j < specs.len()
                && specs[j].kind == (ParamKind::StationaryAr { block_start: false })
            {
                j += 1;
            }
            out.push((i, j));
            i = j;
        } else {
            i += 1;
        }
    }
    out
}

/// True when every stationary AR block of `params` is a stationary
/// polynomial (all partial autocorrelations strictly inside `(-1, 1)`).
pub(crate) fn ar_stationary(blocks: &[(usize, usize)], params: &[f64]) -> bool {
    blocks.iter().all(|&(a, b)| {
        let mut z = vec![0.0; b - a];
        StationaryAr.inverse(&params[a..b], &mut z).is_ok()
    })
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

/// Working-space values to constrained parameters. Variance, bounded and
/// free parameters map elementwise; each stationary AR block jointly.
pub(crate) fn to_constrained(specs: &[ParamSpec], working: &[f64]) -> Vec<f64> {
    let mut out: Vec<f64> = specs
        .iter()
        .zip(working)
        .map(|(s, &w)| match s.kind {
            ParamKind::Variance => w * w,
            ParamKind::Bounded { low, high } => low + (high - low) * logistic(w),
            ParamKind::Free => w,
            // Filled in jointly below.
            ParamKind::StationaryAr { .. } => f64::NAN,
        })
        .collect();
    for (a, b) in ar_blocks(specs) {
        // A non-finite working value has no image: leave NaN, which the
        // likelihood reports as undefined.
        let _ = StationaryAr.forward(&working[a..b], &mut out[a..b]);
    }
    out
}

/// Constrained parameters to working-space values (the inverse map;
/// bounded values are clamped strictly inside their interval first, and a
/// non-stationary AR block starts from the origin of its working space,
/// the white-noise polynomial).
pub(crate) fn to_working(specs: &[ParamSpec], constrained: &[f64]) -> Vec<f64> {
    let mut out: Vec<f64> = specs
        .iter()
        .zip(constrained)
        .map(|(s, &c)| match s.kind {
            ParamKind::Variance => c.max(0.0).sqrt(),
            ParamKind::Bounded { low, high } => {
                let x = ((c - low) / (high - low)).clamp(1e-9, 1.0 - 1e-9);
                (x / (1.0 - x)).ln()
            }
            ParamKind::Free => c,
            ParamKind::StationaryAr { .. } => 0.0,
        })
        .collect();
    for (a, b) in ar_blocks(specs) {
        let mut z = vec![0.0; b - a];
        if StationaryAr.inverse(&constrained[a..b], &mut z).is_ok() {
            out[a..b].copy_from_slice(&z);
        }
    }
    out
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
    loglik: F,
) -> Result<MleOutcome, SsmError>
where
    F: FnMut(&[f64]) -> f64,
{
    maximize_tied(specs, starts, &[], loglik)
}

/// [`maximize`] with identification ties for the standard errors: for
/// every `(variance, transition)` entry of `ties`, when the variance
/// parameter is flagged at the boundary, the transition parameters of its
/// block are removed from the information matrix and reported with a NaN
/// standard error. This is the stationary-block case: a stationary
/// component whose disturbance variance is zero is identically zero (its
/// unconditional variance is zero too), so its transition parameters —
/// AR coefficients, a stationary cycle's frequency and damping — carry
/// exactly zero information there, and leaving them in would make the
/// whole Hessian singular. With `ties` empty this is exactly
/// [`maximize`].
pub(crate) fn maximize_tied<F>(
    specs: &[ParamSpec],
    starts: &[Vec<f64>],
    ties: &[(usize, Vec<usize>)],
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
            // The stationarity region is open: its edge has no stationary
            // initialization (the likelihood is undefined there), so the
            // search never reports it and an AR coefficient is never "at a
            // bound".
            ParamKind::Free | ParamKind::StationaryAr { .. } => false,
        };
    }
    let mut excluded = at_boundary.clone();
    for (var, deps) in ties {
        if at_boundary[*var] {
            for &d in deps {
                excluded[d] = true;
            }
        }
    }
    let se = standard_errors(specs, &params, &excluded, &mut loglik);
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
/// excluded (at a boundary, or tied to a flagged variance; see the module
/// docs).
fn standard_errors<F>(
    specs: &[ParamSpec],
    params: &[f64],
    excluded: &[bool],
    loglik: &mut F,
) -> Vec<f64>
where
    F: FnMut(&[f64]) -> f64,
{
    let k = specs.len();
    let mut se = vec![f64::NAN; k];
    let free: Vec<usize> = (0..k).filter(|&i| !excluded[i]).collect();
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
                ParamKind::StationaryAr { .. } => eps4 * x.abs().max(1e-2),
            }
        })
        .collect();
    if h.iter().any(|&v| !(v.is_finite() && v > 0.0)) {
        return se;
    }
    let m = free.len();
    let mut h = h;
    if !shrink_ar_steps(specs, params, &free, &mut h) {
        return se;
    }
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

/// The probe guard for AR coefficients. The Hessian is a four-point
/// central difference in the constrained space, so it evaluates the
/// likelihood at `theta +/- h_i +/- h_j`; next to the stationarity
/// boundary a probe can step outside it, where the likelihood is undefined
/// (NaN), and a single NaN probe would blank every standard error. Halves
/// each AR coefficient's step until every probe of every pair stays
/// stationary. Returns false when no admissible step was found (an
/// estimate within roundoff of the boundary). Without AR coefficients in
/// `free` this does nothing.
fn shrink_ar_steps(specs: &[ParamSpec], params: &[f64], free: &[usize], h: &mut [f64]) -> bool {
    let is_ar: Vec<bool> = free
        .iter()
        .map(|&i| matches!(specs[i].kind, ParamKind::StationaryAr { .. }))
        .collect();
    if !is_ar.iter().any(|&b| b) {
        return true;
    }
    let blocks = ar_blocks(specs);
    let m = free.len();
    let mut probe = params.to_vec();
    for _ in 0..64 {
        let mut bad = vec![false; m];
        for a in 0..m {
            for b in a..m {
                if !(is_ar[a] || is_ar[b]) {
                    continue;
                }
                for (sa, sb) in [(1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)] {
                    probe.copy_from_slice(params);
                    probe[free[a]] += sa * h[a];
                    probe[free[b]] += sb * h[b];
                    if !ar_stationary(&blocks, &probe) {
                        bad[a] |= is_ar[a];
                        bad[b] |= is_ar[b];
                    }
                }
            }
        }
        if !bad.iter().any(|&b| b) {
            return true;
        }
        for (hv, &b) in h.iter_mut().zip(&bad) {
            if b {
                *hv *= 0.5;
            }
        }
    }
    false
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

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(name: &str, kind: ParamKind) -> ParamSpec {
        ParamSpec {
            name: name.to_string(),
            kind,
        }
    }

    fn ar(block_start: bool) -> ParamKind {
        ParamKind::StationaryAr { block_start }
    }

    /// `block_start` delimits adjacent AR blocks, and the joint map is a
    /// bijection onto the stationarity region: an AR(2) start survives the
    /// round trip, a non-stationary one starts from white noise.
    #[test]
    fn ar_blocks_are_delimited_and_mapped_jointly() {
        let specs = vec![
            spec("s", ParamKind::Variance),
            spec("a1", ar(true)),
            spec("a2", ar(false)),
            spec("b1", ar(true)),
            spec("x", ParamKind::Free),
        ];
        assert_eq!(ar_blocks(&specs), vec![(1, 3), (3, 4)]);
        let c = vec![2.0, 1.3, -0.4, 0.5, -1.5];
        let w = to_working(&specs, &c);
        let back = to_constrained(&specs, &w);
        for (a, b) in back.iter().zip(&c) {
            assert!((a - b).abs() < 1e-12, "{back:?} vs {c:?}");
        }
        // (1.3, -0.4) is stationary as one AR(2); as two AR(1)s the first
        // would not be: the blocks really are joint.
        assert!(ar_stationary(&ar_blocks(&specs), &c));
        let w = to_working(&specs, &[2.0, 1.2, 0.0, 0.5, 1.0]);
        assert_eq!(&w[1..3], &[0.0, 0.0]);
        // Working vectors map to stationary polynomials...
        let c = to_constrained(&specs, &[1.0, 3.0, -3.0, 3.0, 0.0]);
        assert!(ar_stationary(&ar_blocks(&specs), &c), "{c:?}");
        // ...until `tanh` saturates to exactly 1 in floating point (|z| >
        // ~19): the image is then the unit-root boundary, which the
        // stationary initialization refuses (NaN likelihood), so the
        // search never reports it.
        let c = to_constrained(&specs, &[1.0, 40.0, -40.0, 3.0, 0.0]);
        assert_eq!(&c[1..3], &[2.0, -1.0]);
        assert!(!ar_stationary(&ar_blocks(&specs), &c));
    }

    /// The probe guard: at an AR estimate 5e-5 from the unit root the
    /// unguarded step (`eps^(1/4) |phi|` ~ 1.2e-4) would probe outside the
    /// stationarity region, where the likelihood is NaN, and blank every
    /// standard error. The guard halves the AR step until every probe is
    /// stationary, and the standard errors are the closed-form ones.
    #[test]
    fn ar_probe_steps_shrink_inside_the_stationarity_region() {
        let specs = vec![spec("w", ParamKind::Free), spec("phi", ar(true))];
        let c = 0.99995;
        let mut f = |p: &[f64]| -> f64 {
            if p[1].abs() >= 1.0 {
                return f64::NAN;
            }
            -(p[0] - 1.0).powi(2) - 1e3 * (p[1] - c).powi(2)
        };
        let se = standard_errors(&specs, &[1.0, c], &[false, false], &mut f);
        assert!((se[0] - 0.5f64.sqrt()).abs() < 1e-6, "{se:?}");
        assert!((se[1] - (1.0 / 2e3f64).sqrt()).abs() < 1e-6, "{se:?}");
    }

    /// The identification tie: a stationary block whose variance piles up
    /// at zero is identically zero, so its transition parameter carries no
    /// information. Untied, the near-singular Hessian reports a garbage
    /// standard error for it (here ~3e18, read off a pivot of order the
    /// leftover variance; exactly at zero it would blank every standard
    /// error); tied, the transition parameter alone gets NaN and the rest
    /// keep their curvature.
    #[test]
    fn a_flagged_block_variance_drops_its_transition_parameters() {
        let specs = vec![
            spec("s", ParamKind::Variance),
            spec("phi", ar(true)),
            spec("w", ParamKind::Free),
        ];
        let ll = |p: &[f64]| -> f64 {
            -10.0 * (p[2] - 1.0).powi(2) - p[0] * (1.0 + (p[1] - 0.3).powi(2))
        };
        let starts = vec![vec![0.5, 0.2, 0.0]];
        let untied = maximize(&specs, &starts, ll).unwrap();
        assert!(untied.at_boundary[0] && !untied.at_boundary[1]);
        assert!(
            untied.se[1].is_nan() || untied.se[1] > 1e6,
            "{:?}",
            untied.se
        );
        let tied = maximize_tied(&specs, &starts, &[(0, vec![1])], ll).unwrap();
        assert_eq!(tied.at_boundary, untied.at_boundary);
        assert!(tied.se[0].is_nan() && tied.se[1].is_nan());
        assert!(
            (tied.se[2] - (1.0 / 20.0f64).sqrt()).abs() < 1e-5,
            "{:?}",
            tied.se
        );
    }
}
