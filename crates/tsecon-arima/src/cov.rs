//! Parameter covariance from the observed information: the inverse of
//! the negative Hessian of the log-likelihood at the reported
//! parameters.
//!
//! ```text
//! Cov(theta_hat) = [ -d^2 loglik / d theta d theta' ]^{-1}
//! ```
//!
//! evaluated by finite differences on the *total* (not per-observation)
//! log-likelihood, in the natural parameter space `[const?, ar.., ma..,
//! sigma2]`. This is the estimator statsmodels reports as
//! `SARIMAX(...).fit(cov_type='approx').bse`.
//!
//! # What the reference actually does, and what this must match
//!
//! statsmodels' `cov_type='approx'` defaults to
//! `approx_complex_step=True`: it differentiates by *complex step*, which
//! carries no subtractive cancellation and no step-size trade-off at all.
//! It is not `numdiff.approx_hess3`, and the agreement measured in
//! `tests/golden_bse.rs` is not step-rule parity — both sides are
//! estimating the same true Hessian, and that true Hessian is the only
//! invariant worth holding. Where a step rule and accuracy conflict,
//! accuracy wins here.
//!
//! Complex step is unavailable to us (the Kalman filter runs on `f64`),
//! so the Hessian is four-point central cross differences,
//!
//! ```text
//! H_ij ~ [ (f(+h_i, +h_j) - f(+h_i, -h_j))
//!        - (f(-h_i, +h_j) - f(-h_i, -h_j)) ] / (4 h_i h_j),
//! ```
//!
//! with steps chosen per coordinate:
//!
//! * **`sigma2` is differentiated with respect to `log sigma2`** and
//!   chain-ruled back (see [`numerical_hessian`]). A variance has no
//!   natural additive scale — only a multiplicative one — and the
//!   absolute-floored rule `h = eps^(1/4) max(|theta_i|, 0.1)` that
//!   statsmodels uses for *all* coordinates silently destroys the
//!   `sigma2` row once `sigma2` falls below ~1e-3, where the step becomes
//!   a large fraction of the parameter itself. Measured against the
//!   closed form `se(sigma2) = sqrt(2 sigma2^2 / n)` on ARIMA(0,1,0)+c,
//!   the floored rule returned 4.6e-2 relative error at
//!   `sigma2 = 9.8e-5` — no error, no NaN, just a wrong number — and hard
//!   errors one decade below that. On a log scale the step is
//!   `eps^(1/4)` *relative* at every magnitude, and the same sweep now
//!   holds 4.6e-7 worst case across `sigma2` from 1e-8 to 1e6; see
//!   `tests/cov_accuracy.rs`. The band that was wrong is where daily log
//!   returns and rates-in-decimals live, so it is not an exotic corner —
//!   and no fixture goes anywhere near it, every one having `sigma2` in
//!   `[0.94, 2e4]`.
//! * The constant and the AR/MA coefficients keep the statsmodels rule
//!   `h_i = eps^(1/4) max(|theta_i|, 0.1)`. AR/MA coefficients are O(1)
//!   by stationarity/invertibility, so the floor never binds hard; and
//!   the log-likelihood is *exactly quadratic* in the constant (the
//!   Kalman gains do not depend on the intercept, so the prediction
//!   errors are affine in it), which means an oversized step there costs
//!   no truncation at all.
//!
//! Measured agreement with the statsmodels fixture is at worst 4.8e-7
//! relative on the well-conditioned cases and 3.5e-6 on the Nile (whose
//! parameter scales span four orders of magnitude); see
//! `tests/golden_bse.rs`. A numerical Hessian is not a closed form and
//! will never agree to 1e-10 — the golden tolerances say what is actually
//! held.
//!
//! There is a hard floor on how small `sigma2` may be that has nothing to
//! do with this module: `tsecon_ssm`'s filter skips any observation whose
//! prediction variance falls below an *absolute* `1e-10`, so at
//! `sigma2 <~ 1e-10` the exact log-likelihood returns a constant (`0.0`,
//! successfully) and every derivative of it is zero. That surfaces here
//! as a `CovarianceFailed` naming the rescaling, never as a number.
//!
//! # Rank, and what "the same as statsmodels" stops meaning
//!
//! The inverse here is a genuine inverse, not a pseudo-inverse.
//! statsmodels uses `pinv_extended`, which truncates small singular
//! values and therefore returns *something* for a rank-deficient
//! information matrix. So the common shorthand — that
//! `cov_params_approx` is "exactly `pinv(-H_total)`" and this crate
//! computes the same object — holds only while `-H_total` is well
//! conditioned. When it is not, the two deliberately diverge:
//! statsmodels reports the pseudo-inverse of a matrix that does not
//! identify the parameters, and this crate reports
//! [`ArimaError::CovarianceFailed`] (see [`invert`] and [`MIN_RCOND`]).
//!
//! What the rank guard does and does not buy. It is a statement about
//! *arithmetic*: an equilibrated condition number past 1e6 means the
//! inverse has no significant digits left, and that is refused. It is
//! not a general test for an unidentified model — a finite-difference
//! Hessian at an unidentified point does not come back exactly singular,
//! it comes back with the flat direction filled in by differencing
//! noise, and the sign of that noise decides the outcome. On an
//! ARMA(1,1) at `theta = -phi` (where the likelihood is provably
//! constant along a line) what the user actually sees is `NaN` standard
//! errors for the unidentified block from a negative variance, or a
//! rank error — never a finite, confident-looking number.
//! `tests/cov_accuracy.rs` pins that.
//!
//! Why the observed information rather than the outer product of
//! gradients (statsmodels' *default* `cov_type='opg'`): OPG and the
//! Hessian are asymptotically equivalent under correct specification but
//! differ visibly in small samples, and the Hessian is the quantity the
//! forecast delta-method correction needs — the two must come from one
//! estimator or the drift term would not be the variance of the
//! constant that the standard errors advertise.

use crate::error::ArimaError;

/// The finite-difference base step, `eps^(1/4) ~ 1.22e-4` — the
/// truncation/roundoff optimum for a central second difference.
#[inline]
fn base_step() -> f64 {
    f64::EPSILON.powf(0.25)
}

/// Smallest admissible reciprocal condition number of the
/// *equilibrated* information matrix in [`invert`].
///
/// Derived from this crate's own accuracy rather than tuned. The
/// numerical Hessian agrees with statsmodels' complex-step Hessian to
/// 2.1e-6 relative in the worst measured case, so the matrix handed to
/// the inversion carries a perturbation of about that size. Below
/// `rcond = 1e-6` the least-curved direction of the information matrix is
/// therefore at or under our own differencing error: whatever comes out
/// of the inverse along that direction is noise, not data, and it comes
/// out finite and plausible-looking, which is worse than an error.
///
/// Measured `rcond` over the six statsmodels golden cases: 1.0 on
/// `rw_drift_010c_T60` (exactly block diagonal), 8.5e-2 on the two
/// ARMA(1,1) fits, 2.4e-1 / 2.9e-1 on the demeaned and AR(2) cases, and
/// **5.1e-4 on `nile_arma11c`**, which is the binding one — 500x clear of
/// the threshold. `tests/cov_accuracy.rs` asserts the margin so that a
/// future change to the step rule cannot quietly eat it.
const MIN_RCOND: f64 = 1e-6;

/// Parameter covariance matrix and the standard errors on its diagonal,
/// in the packed parameter order `[const?, ar.., ma.., sigma2]` (aligned
/// with [`ArimaResults::param_names`](crate::ArimaResults::param_names)).
#[derive(Debug, Clone, PartialEq)]
pub struct ParamCov {
    k: usize,
    cov: Vec<f64>,
    se: Vec<f64>,
    rcond: f64,
}

impl ParamCov {
    /// Number of parameters `k` (the matrix is `k x k`).
    #[inline]
    pub fn k(&self) -> usize {
        self.k
    }

    /// The covariance matrix in row-major order, length `k * k`.
    #[inline]
    pub fn cov(&self) -> &[f64] {
        &self.cov
    }

    /// Standard errors `sqrt(diag(cov))`, length `k` — the statsmodels
    /// `.bse` vector.
    ///
    /// An entry is NaN when its variance came out negative, which means
    /// the numerical Hessian was not negative definite at the reported
    /// parameters (a flat or boundary optimum). That is reported rather
    /// than clipped: a NaN standard error is a signal about the fit, and
    /// silently replacing it with a number would hide it.
    #[inline]
    pub fn se(&self) -> &[f64] {
        &self.se
    }

    /// Covariance entry `(i, j)`, or `None` when either index is out of
    /// range.
    #[inline]
    pub fn get(&self, i: usize, j: usize) -> Option<f64> {
        (i < self.k && j < self.k).then(|| self.cov[i * self.k + j])
    }

    /// Reciprocal condition number of the *equilibrated* observed
    /// information — how much of this covariance survived the inversion.
    ///
    /// 1 is a perfectly conditioned (block-diagonal) information matrix;
    /// smaller is worse, and roughly `-log10(rcond)` decimal digits are
    /// lost from every entry. Anything below [`MIN_RCOND`] is refused
    /// rather than returned, so a `ParamCov` you are holding always has
    /// `rcond >= 1e-6`; the value is exposed because "it inverted" and
    /// "the answer means something" are different claims. For reference,
    /// the statsmodels golden fits in `tests/golden_bse.rs` run from
    /// 8.5e-2 to 1.0, except the Nile ARMA(1,1) at 5.1e-4 — whose AR and
    /// MA terms are nearly redundant on 100 observations.
    #[inline]
    pub fn rcond(&self) -> f64 {
        self.rcond
    }
}

/// Four-point central-difference Hessian of `f` at `x`.
///
/// Coordinates flagged in `log_scale` are differentiated with respect to
/// `u_i = ln(theta_i)` and chain-ruled back; every other coordinate is
/// differentiated directly. Writing `J_i = dtheta_i / du_i` (so `J_i =
/// theta_i` on a log coordinate and `1` otherwise), the map is exact:
///
/// ```text
/// d^2 F / du_i du_j = J_i J_j  d^2 f / dtheta_i dtheta_j
///                     + [i = j] (d^2 theta_i / du_i^2) df / dtheta_i,
/// ```
///
/// and on a log coordinate `d^2 theta_i / du_i^2 = theta_i = J_i`, so the
/// stray term is exactly `dF / du_i` — a quantity the same difference
/// stencil already produces. Hence
///
/// ```text
/// H_ij = ( Hu_ij - [i = j, log] (dF/du_i) ) / (J_i J_j).
/// ```
///
/// At a maximizer `dF/du_i` vanishes and the correction is numerical
/// noise; away from one (`EstimationMethod::Fixed`, or another
/// optimizer's stopping point) it is the difference between the Hessian
/// and something that is not the Hessian, so it is always applied.
///
/// A flagged coordinate that is not strictly positive falls back to the
/// direct rule rather than erroring — `log` of it does not exist, and
/// this function's job is differentiation, not validation.
///
/// `f` is the *negative* total log-likelihood, so the result is already
/// the observed information up to sign conventions handled by the
/// caller.
fn numerical_hessian<F>(mut f: F, x: &[f64], log_scale: &[bool]) -> Result<Vec<f64>, ArimaError>
where
    F: FnMut(&[f64]) -> Result<f64, ArimaError>,
{
    let n = x.len();
    debug_assert_eq!(log_scale.len(), n);
    let qe = base_step();
    // `on_log[i]` is the flag *after* the positivity fallback; `u` is the
    // point in differentiation space; `jac[i] = dtheta_i / du_i`.
    let mut on_log = vec![false; n];
    let mut u = vec![0.0; n];
    let mut jac = vec![1.0; n];
    let mut h = vec![0.0; n];
    for i in 0..n {
        let is_log = log_scale.get(i).copied().unwrap_or(false) && x[i].is_finite() && x[i] > 0.0;
        on_log[i] = is_log;
        if is_log {
            u[i] = x[i].ln();
            jac[i] = x[i];
            // A *relative* step of eps^(1/4) in theta_i, at any scale.
            h[i] = qe;
        } else {
            u[i] = x[i];
            h[i] = qe * x[i].abs().max(0.1);
        }
    }

    let mut hess = vec![0.0; n * n];
    let mut probe = vec![0.0; n];
    // `du` is the perturbed point in differentiation space; `probe` is
    // its image in parameter space, which is what `f` consumes.
    let mut du = vec![0.0; n];
    let mut eval = |probe: &mut Vec<f64>,
                    du: &mut Vec<f64>,
                    di: (usize, f64),
                    dj: (usize, f64)|
     -> Result<f64, ArimaError> {
        du.copy_from_slice(&u);
        du[di.0] += di.1;
        du[dj.0] += dj.1;
        for i in 0..n {
            probe[i] = if on_log[i] { du[i].exp() } else { du[i] };
        }
        f(probe).map_err(|_| ArimaError::CovarianceFailed {
            what: "the log-likelihood is undefined at a finite-difference probe point. \
                   sigma2 is stepped multiplicatively, so it cannot be pushed to zero \
                   or below; the remaining causes are a fit that stopped on the \
                   stationarity/invertibility boundary, or an AR/MA coefficient whose \
                   step eps^(1/4) max(|theta_i|, 0.1) crosses it. Refit from different \
                   starting values, or lower p or q",
        })
    };

    for i in 0..n {
        for j in i..n {
            let fpp = eval(&mut probe, &mut du, (i, h[i]), (j, h[j]))?;
            let fpm = eval(&mut probe, &mut du, (i, h[i]), (j, -h[j]))?;
            let fmp = eval(&mut probe, &mut du, (i, -h[i]), (j, h[j]))?;
            let fmm = eval(&mut probe, &mut du, (i, -h[i]), (j, -h[j]))?;
            // In differentiation space. For i == j the two perturbations
            // land on the same coordinate, so this is the usual
            // (f(+2h) - 2 f(0) + f(-2h)) / (2h)^2 with fpm = fmp = f(0).
            let hu = ((fpp - fpm) - (fmp - fmm)) / (4.0 * h[i] * h[j]);
            let v = if i == j && on_log[i] {
                // dF/du_i from the same four evaluations: fpp = F(u + 2h),
                // fmm = F(u - 2h), so (fpp - fmm) / (4h) is the central
                // first difference at step 2h.
                let grad_u = (fpp - fmm) / (4.0 * h[i]);
                (hu - grad_u) / (jac[i] * jac[i])
            } else {
                hu / (jac[i] * jac[j])
            };
            if !v.is_finite() {
                return Err(ArimaError::CovarianceFailed {
                    what: "the numerical Hessian of the log-likelihood has a non-finite \
                           entry at the reported parameters",
                });
            }
            hess[i * n + j] = v;
            hess[j * n + i] = v;
        }
    }
    Ok(hess)
}

/// Inverts a small symmetric matrix (row-major `n x n`) by Gauss-Jordan
/// elimination with partial pivoting, on the **equilibrated** matrix
/// `B = D^{-1} A D^{-1}`, `D = diag(sqrt(|A_ii|))`.
///
/// Equilibration is what makes a singularity test possible at all. The
/// pivots of a raw information matrix carry the parameters' units: on
/// the Nile ARMA(1,1) the `sigma2` curvature is O(1e-7) beside an AR
/// curvature of O(1e2), a spread of nine orders of magnitude that is
/// pure scale and no ill-conditioning whatsoever. Against that, an
/// absolute pivot threshold is useless — it must be set so low
/// (`1e-300`, i.e. "is the pivot exactly zero") that a matrix of
/// condition 1e18 sails through and returns finite, plausible-looking
/// standard errors. After scaling, `B` has unit magnitude on the
/// diagonal, and the reciprocal condition number
/// `1 / (||B||_inf ||B^{-1}||_inf)` computed below is a pure
/// conditioning number that [`MIN_RCOND`] can be compared against on any
/// series in any units.
///
/// The elimination itself still only rejects an exactly zero or
/// non-finite pivot (division by it would produce infinities before the
/// condition number could be formed); the conditioning verdict is passed
/// on the assembled inverse, which is where it is meaningful.
///
/// `k` is tiny here (constant + p + q + 1), so a dense direct inverse is
/// the right tool; no factorization is reused.
fn invert(a: &[f64], n: usize) -> Result<(Vec<f64>, f64), ArimaError> {
    let singular = ArimaError::CovarianceFailed {
        what: "the numerical Hessian of the log-likelihood is singular or numerically \
               rank-deficient at the reported parameters, so the observed information \
               cannot be inverted — usually a parameter is not identified by this \
               sample (near-cancelling AR and MA roots: lower p or q) or the optimizer \
               stopped on the stationarity/invertibility boundary",
    };
    let flat = ArimaError::CovarianceFailed {
        what: "the log-likelihood has zero or non-finite curvature in one parameter at \
               the reported parameters, so that parameter's standard error is not \
               defined. Either the sample does not identify that parameter, or the \
               series is so far from unit scale that the state-space likelihood has \
               gone numerically flat — the Kalman filter drops observations whose \
               prediction variance falls below 1e-10, so an innovation variance near \
               or below that returns a constant log-likelihood. Rescale the series \
               toward unit variance and refit",
    };

    // Equilibrate. `d_i = sqrt(|A_ii|)` divides the parameter's units out
    // of row and column `i`; a zero diagonal is a parameter the
    // likelihood does not curve in at all.
    let mut d = vec![0.0; n];
    for (i, di) in d.iter_mut().enumerate() {
        let v = a[i * n + i].abs().sqrt();
        if !(v.is_finite() && v > 0.0) {
            return Err(flat);
        }
        *di = v;
    }

    // Augmented [B | I], reduced in place; rows are length 2n.
    let w = 2 * n;
    let mut m = vec![0.0; n * w];
    let mut norm_b = 0.0_f64;
    for i in 0..n {
        let mut row_sum = 0.0;
        for j in 0..n {
            let v = a[i * n + j] / (d[i] * d[j]);
            m[i * w + j] = v;
            row_sum += v.abs();
        }
        norm_b = norm_b.max(row_sum);
        m[i * w + n + i] = 1.0;
    }
    for col in 0..n {
        let mut pivot_row = col;
        for r in col + 1..n {
            if m[r * w + col].abs() > m[pivot_row * w + col].abs() {
                pivot_row = r;
            }
        }
        let pivot = m[pivot_row * w + col];
        if !pivot.is_finite() || pivot == 0.0 {
            return Err(singular);
        }
        if pivot_row != col {
            for c in 0..w {
                m.swap(col * w + c, pivot_row * w + c);
            }
        }
        for c in 0..w {
            m[col * w + c] /= pivot;
        }
        for r in 0..n {
            if r == col {
                continue;
            }
            let factor = m[r * w + col];
            if factor != 0.0 {
                for c in 0..w {
                    m[r * w + c] -= factor * m[col * w + c];
                }
            }
        }
    }
    // The reciprocal condition number of the *equilibrated* matrix, in
    // the infinity norm: `rcond = 1 / (||B|| ||B^{-1}||)`. Exact, not
    // estimated — `n` is at most a handful.
    let mut norm_inv = 0.0_f64;
    for i in 0..n {
        let mut row_sum = 0.0;
        for j in 0..n {
            row_sum += m[i * w + n + j].abs();
        }
        norm_inv = norm_inv.max(row_sum);
    }
    let rcond = 1.0 / (norm_b * norm_inv);
    if !rcond.is_finite() || rcond < MIN_RCOND {
        return Err(singular);
    }

    // Undo the equilibration: A^{-1} = D^{-1} B^{-1} D^{-1}.
    let mut inv = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let v = m[i * w + n + j] / (d[i] * d[j]);
            if !v.is_finite() {
                return Err(singular);
            }
            inv[i * n + j] = v;
        }
    }
    Ok((inv, rcond))
}

/// Builds [`ParamCov`] from the negative total log-likelihood `neg_ll`
/// evaluated at `params`.
///
/// `log_scale[i]` asks for coordinate `i` to be differentiated
/// multiplicatively (see [`numerical_hessian`]); the ARIMA caller sets it
/// for the `sigma2` slot and nothing else.
///
/// # Errors
///
/// [`ArimaError::CovarianceFailed`] when a finite-difference probe leaves
/// the admissible parameter region, when the Hessian has a non-finite
/// entry, or when it is singular or numerically rank-deficient.
pub(crate) fn observed_information<F>(
    neg_ll: F,
    params: &[f64],
    log_scale: &[bool],
) -> Result<ParamCov, ArimaError>
where
    F: FnMut(&[f64]) -> Result<f64, ArimaError>,
{
    let k = params.len();
    // The Hessian of the *negative* log-likelihood is already the
    // observed information; its inverse is the covariance.
    let info = numerical_hessian(neg_ll, params, log_scale)?;
    let (cov, rcond) = invert(&info, k)?;
    // Restore exact symmetry lost to elimination roundoff, so that
    // `get(i, j) == get(j, i)` holds bit-for-bit.
    let mut cov = cov;
    for i in 0..k {
        for j in 0..i {
            let v = 0.5 * (cov[i * k + j] + cov[j * k + i]);
            cov[i * k + j] = v;
            cov[j * k + i] = v;
        }
    }
    let se: Vec<f64> = (0..k).map(|i| cov[i * k + i].sqrt()).collect();
    Ok(ParamCov { k, cov, se, rcond })
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn invert_recovers_identity() {
        let a = [4.0, 1.0, 0.5, 1.0, 3.0, 0.2, 0.5, 0.2, 2.0];
        let (inv, rcond) = invert(&a, 3).unwrap();
        assert!((0.0..=1.0).contains(&rcond), "rcond = {rcond}");
        for i in 0..3 {
            for j in 0..3 {
                let mut s = 0.0;
                for k in 0..3 {
                    s += a[i * 3 + k] * inv[k * 3 + j];
                }
                let target = if i == j { 1.0 } else { 0.0 };
                assert!((s - target).abs() < 1e-12, "prod[{i}][{j}] = {s}");
            }
        }
    }

    /// Equilibration must not cost accuracy on a *well-conditioned but
    /// badly scaled* matrix — the Nile situation, where one parameter's
    /// curvature is 1e8 times another's purely from units.
    ///
    /// The check is on the inverse itself, not on `A A^{-1} - I`: that
    /// residual's off-diagonal entries carry the ratio `s_i / s_j = 1e8`,
    /// so an absolute tolerance on it would be measuring the units rather
    /// than the arithmetic. `A = S C S` has `A^{-1} = S^{-1} C^{-1}
    /// S^{-1}` exactly, so the invariant with the units divided out is
    /// that `s_i s_j A^{-1}_ij` reproduces `C^{-1}` — which is what a
    /// user of `param_cov` actually depends on, entry by entry.
    #[test]
    fn invert_is_accurate_under_extreme_scaling() {
        let c = [1.0, 0.3, 0.2, 0.3, 1.0, -0.1, 0.2, -0.1, 1.0];
        let s = [1.0, 1e-4, 1e4];
        let mut a = [0.0; 9];
        for i in 0..3 {
            for j in 0..3 {
                a[i * 3 + j] = s[i] * c[i * 3 + j] * s[j];
            }
        }
        let (inv_a, rcond_a) = invert(&a, 3).unwrap();
        let (inv_c, rcond_c) = invert(&c, 3).unwrap();
        // The verdict is a property of C, not of the units it is
        // expressed in: equilibration must give back the same number.
        assert!(
            (rcond_a - rcond_c).abs() <= 1e-12 * rcond_c,
            "rcond moved under scaling: {rcond_a} vs {rcond_c}"
        );
        for i in 0..3 {
            for j in 0..3 {
                let got = inv_a[i * 3 + j] * s[i] * s[j];
                let want = inv_c[i * 3 + j];
                assert!(
                    (got - want).abs() <= 1e-12 * want.abs(),
                    "scaled inv[{i}][{j}] = {got} vs {want} (scaling cost accuracy)"
                );
            }
        }
    }

    #[test]
    fn invert_rejects_exactly_singular() {
        let a = [1.0, 2.0, 2.0, 4.0];
        assert!(matches!(
            invert(&a, 2),
            Err(ArimaError::CovarianceFailed { .. })
        ));
    }

    /// The test the old `pivot.abs() < 1e-300` rule could not do: a
    /// matrix that is *nearly* rank deficient rather than exactly so.
    ///
    /// `[[1, 1 - e], [1 - e, 1]]` has eigenvalues `2 - e` and `e`, so
    /// `rcond ~ e/2` and the cut sits at `e = 2e-6`. Both sides of it are
    /// checked: at `e <= 1e-8` the inverse has no significant digits and
    /// must be refused; at `e >= 1e-4` it is ill conditioned but still
    /// carries information, and must come back — and come back *right*,
    /// which is asserted against the exact 2x2 inverse rather than merely
    /// being non-`Err`.
    #[test]
    fn invert_rejects_near_singular_but_keeps_merely_ill_conditioned() {
        for e in [1e-8, 1e-12, 1e-14, 1e-16] {
            let a = [1.0, 1.0 - e, 1.0 - e, 1.0];
            assert!(
                matches!(invert(&a, 2), Err(ArimaError::CovarianceFailed { .. })),
                "near-singular matrix at e = {e:e} was inverted anyway"
            );
        }
        for e in [1e-2, 1e-3, 1e-4] {
            let a = [1.0, 1.0 - e, 1.0 - e, 1.0];
            let (inv, _) = invert(&a, 2).unwrap();
            // Exact inverse: 1/(1 - (1-e)^2) * [[1, -(1-e)], [-(1-e), 1]].
            let det = 1.0 - (1.0 - e) * (1.0 - e);
            for (got, want) in
                inv.iter()
                    .zip([1.0 / det, -(1.0 - e) / det, -(1.0 - e) / det, 1.0 / det])
            {
                assert!(
                    (got - want).abs() <= 1e-6 * want.abs(),
                    "ill-conditioned but usable matrix at e = {e:e}: {got} vs {want}"
                );
            }
        }
    }

    /// Scale invariance is the whole point of equilibration: rescaling a
    /// parameter changes the matrix's raw condition number by orders of
    /// magnitude while changing nothing about whether the model is
    /// identified, so the accept/reject verdict must not move.
    #[test]
    fn singularity_verdict_is_scale_invariant() {
        // Perfectly conditioned once the units are divided out, but with
        // a raw condition number of 1e16.
        let a = [1e-8, 0.0, 0.0, 1e8];
        assert!(invert(&a, 2).is_ok(), "a badly scaled identity was refused");
        // Rank deficient, but with every entry enormous: an absolute
        // pivot test would never fire here.
        let b = [1e8, 1e8, 1e8, 1e8];
        assert!(matches!(
            invert(&b, 2),
            Err(ArimaError::CovarianceFailed { .. })
        ));

        // The same near-singular matrix in three different unit systems
        // must give the same verdict each time.
        let e = 1e-10;
        for s in [[1.0, 1.0], [1e-5, 1e5], [1e6, 1e-3]] {
            let raw = [1.0, 1.0 - e, 1.0 - e, 1.0];
            let scaled = [
                s[0] * raw[0] * s[0],
                s[0] * raw[1] * s[1],
                s[1] * raw[2] * s[0],
                s[1] * raw[3] * s[1],
            ];
            assert!(
                matches!(invert(&scaled, 2), Err(ArimaError::CovarianceFailed { .. })),
                "verdict flipped under the scaling {s:?}"
            );
        }
    }

    /// A quadratic `-loglik = 0.5 (theta - m)' A (theta - m)` has Hessian
    /// `A` exactly, so the covariance is `A^{-1}`: the finite-difference
    /// machinery must reproduce a known closed form.
    #[test]
    fn covariance_of_a_gaussian_quadratic_is_the_inverse_curvature() {
        let a = [2.0, 0.3, 0.3, 5.0];
        let m = [1.0, -2.0];
        let f = |t: &[f64]| {
            let d = [t[0] - m[0], t[1] - m[1]];
            let mut s = 0.0;
            for i in 0..2 {
                for j in 0..2 {
                    s += 0.5 * d[i] * a[i * 2 + j] * d[j];
                }
            }
            Ok(s)
        };
        let pc = observed_information(f, &m, &[false, false]).unwrap();
        let det = a[0] * a[3] - a[1] * a[2];
        let expect = [a[3] / det, -a[1] / det, -a[2] / det, a[0] / det];
        for (i, &want) in expect.iter().enumerate() {
            assert!(
                (pc.cov()[i] - want).abs() < 1e-8,
                "cov[{i}] = {} vs {want}",
                pc.cov()[i]
            );
        }
        assert!((pc.se()[0] - expect[0].sqrt()).abs() < 1e-9);
        assert_eq!(pc.get(0, 1), pc.get(1, 0));
        assert_eq!(pc.get(2, 0), None);
    }

    /// The log-scale chain rule, against a closed form with a *nonzero*
    /// gradient so the `dF/du` correction term is load-bearing.
    ///
    /// `f(s) = a ln s + b / s` has `f'(s) = a/s - b/s^2` and
    /// `f''(s) = -a/s^2 + 2b/s^3`. Differentiating on a log scale without
    /// subtracting `dF/du` would return `f'' + f'/s`, which differs
    /// from the truth whenever `f'(s) != 0` — here by a factor of ~3.
    #[test]
    fn log_scale_hessian_matches_closed_form_away_from_a_stationary_point() {
        let (a, b) = (50.0, 0.5);
        let f = |t: &[f64]| Ok(a * t[0].ln() + b / t[0]);
        // Deliberately not the minimizer (which is s = b / a = 0.01).
        for s in [1e-6, 1e-3, 0.03, 1.0, 1e3] {
            let want = -a / (s * s) + 2.0 * b / (s * s * s);
            let h = numerical_hessian(f, &[s], &[true]).unwrap();
            assert!(
                (h[0] - want).abs() <= 1e-7 * want.abs(),
                "log-scale Hessian at s = {s:e}: {} vs {want} (rel {:e})",
                h[0],
                (h[0] - want).abs() / want.abs()
            );
        }
    }

    /// Scale invariance of the log-scale rule where it matters: the
    /// Gaussian variance Hessian `n / (2 sigma2^2)` at the MLE, swept over
    /// nine decades. The direct rule loses the number entirely below
    /// ~1e-3; this must not.
    #[test]
    fn log_scale_hessian_is_scale_free_for_a_gaussian_variance() {
        let n = 60.0;
        for k in -6..3 {
            let s = 10f64.powi(k);
            // -loglik of n iid N(0, s) draws with sum of squares n * s.
            let f = |t: &[f64]| Ok(0.5 * n * (t[0].ln()) + n * s / (2.0 * t[0]));
            let h = numerical_hessian(f, &[s], &[true]).unwrap();
            let want = n / (2.0 * s * s);
            assert!(
                (h[0] - want).abs() <= 1e-7 * want,
                "sigma2 = 1e{k}: H = {} vs {want} (rel {:e})",
                h[0],
                (h[0] - want).abs() / want
            );
        }
    }

    /// A non-positive coordinate flagged for the log scale falls back to
    /// the direct rule instead of producing NaNs.
    #[test]
    fn log_scale_flag_falls_back_when_the_parameter_is_not_positive() {
        let f = |t: &[f64]| Ok(3.0 * t[0] * t[0]);
        let h = numerical_hessian(f, &[-2.0], &[true]).unwrap();
        assert!((h[0] - 6.0).abs() < 1e-6, "H = {}", h[0]);
    }
}
