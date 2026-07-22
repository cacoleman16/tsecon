//! Historical decomposition of a VAR (Kilian & Lütkepohl 2017, ch. 4): the
//! contribution of each structural shock to each variable at each date, plus
//! the deterministic / initial-condition baseline, satisfying the exact
//! adding-up identity
//!
//! ```text
//! y_{p+t, i} = baseline[t, i] + sum_j HD[t][(i, j)].
//! ```
//!
//! # The decomposition
//!
//! Write the observed series as a deterministic path plus a moving average of
//! reduced-form residuals. The **baseline** `y*_t` iterates the estimated VAR
//! with every in-sample shock set to zero, seeded by the actual pre-sample
//! observations:
//!
//! ```text
//! y*_t = c + sum_{l=1}^p A_l yhat_{t-l},   yhat_{t-l} = data actual if t-l < 0,
//!                                                        y*_{t-l} otherwise.
//! ```
//!
//! The deviation `d_t = y_t - y*_t` then obeys `d_t = sum_{s=0}^t Psi_s u_{t-s}`
//! (reduced-form MA weights `Psi_s`, `Psi_0 = I`), since the deviation is zero
//! through the pre-sample. Substituting `u_{t-s} = P Q eps_{t-s}` gives the
//! **structural** decomposition
//!
//! ```text
//! HD[t][(i, j)] = sum_{s=0}^t Theta_s[(i, j)] E[t - s, j],
//! ```
//!
//! with `Theta_s = Psi_s P Q` the structural IRF ([`tsecon_bayes::cholesky_irf`]
//! times `Q`) and `E` the structural shocks ([`crate::shocks`]). Summing over
//! `j` telescopes back to `sum_s Psi_s u_{t-s} = d_t`, which is why the
//! adding-up identity is exact regardless of the rotation `Q` or the impact
//! normalization.
//!
//! # Orientation invariance
//!
//! Flipping the sign of shock column `j` (`s_j = -1`) negates both
//! `Theta_s[(·, j)]` and `E[·, j]`; their product in `HD[t][(·, j)]` is
//! unchanged. So `|HD|`-based ("most / least important contributor")
//! statements are orientation-free; only the *sign* of a contribution depends
//! on the shock's orientation.

use tsecon_bayes::cholesky_irf;
use tsecon_linalg::faer::{Mat, MatRef};
use tsecon_linalg::jittered_cholesky;

use crate::error::IdentError;
use crate::shocks::{
    build_regressors, orthogonalized_residuals, reduced_form_residuals, structural_shocks,
};
use crate::summary::structural_irf;

/// Extracts the lag matrices `A_1..A_p` and the intercept `c` from the
/// `(1 + n p) x n` regressor-by-equation coefficient matrix `b`.
///
/// `A_l[(i, j)]` is the coefficient of `y_{t-l, j}` in equation `i` — the
/// convention [`tsecon_bayes::cholesky_irf`] uses — and `c[i] = b[(0, i)]`.
pub(crate) fn coefs_and_intercept(
    b: MatRef<'_, f64>,
    n: usize,
    p: usize,
) -> (Vec<Mat<f64>>, Vec<f64>) {
    let coefs: Vec<Mat<f64>> = (1..=p)
        .map(|l| Mat::from_fn(n, n, |i, j| b[(1 + (l - 1) * n + j, i)]))
        .collect();
    let intercept = (0..n).map(|i| b[(0, i)]).collect();
    (coefs, intercept)
}

/// The deterministic + initial-condition baseline path (`T_eff x n`): the VAR
/// iterated forward from the actual pre-sample observations with every
/// in-sample shock zeroed. See the module docs for the recursion.
pub(crate) fn baseline_path(
    coefs: &[Mat<f64>],
    intercept: &[f64],
    data: MatRef<'_, f64>,
    p: usize,
) -> Result<Mat<f64>, IdentError> {
    let n = intercept.len();
    if data.ncols() != n {
        return Err(IdentError::Dimension {
            what: "data columns must equal the number of variables",
            expected: n,
            got: data.ncols(),
        });
    }
    if p == 0 {
        return Err(IdentError::InvalidArgument {
            what: "lag length p must be at least 1",
        });
    }
    if data.nrows() < p + 1 {
        return Err(IdentError::Dimension {
            what: "data must have at least p + 1 rows",
            expected: p + 1,
            got: data.nrows(),
        });
    }
    let t_eff = data.nrows() - p;
    let mut baseline = Mat::<f64>::zeros(t_eff, n);
    for t_idx in 0..t_eff {
        for i in 0..n {
            let mut acc = intercept[i];
            for (l1, a) in coefs.iter().enumerate() {
                let l = l1 + 1;
                let lag_eff = t_idx as isize - l as isize;
                for j in 0..n {
                    let yhat = if lag_eff < 0 {
                        // Pre-sample: use the actual observation at data row
                        // p + t_idx - l (in [0, p-1]).
                        data[((p as isize + lag_eff) as usize, j)]
                    } else {
                        baseline[(lag_eff as usize, j)]
                    };
                    acc += a[(i, j)] * yhat;
                }
            }
            baseline[(t_idx, i)] = acc;
        }
    }
    Ok(baseline)
}

/// The full historical-decomposition tensor `HD[t][(i, j)]` from the structural
/// IRF `theta` (`Theta_s`, length at least `T_eff`) and the structural shocks
/// `E` (`T_eff x n`):
///
/// ```text
/// HD[t][(i, j)] = sum_{s=0}^t theta[s][(i, j)] E[(t - s, j)].
/// ```
///
/// # Errors
///
/// [`IdentError::Dimension`] if `theta.len() < T_eff` (the IRF is truncated
/// before the last date and the decomposition would be incomplete).
pub(crate) fn hd_recursion(
    theta: &[Mat<f64>],
    shocks: MatRef<'_, f64>,
) -> Result<Vec<Mat<f64>>, IdentError> {
    let t_eff = shocks.nrows();
    let n = shocks.ncols();
    if theta.len() < t_eff {
        return Err(IdentError::Dimension {
            what: "structural IRF must reach horizon T_eff - 1 for the decomposition",
            expected: t_eff,
            got: theta.len(),
        });
    }
    let mut hd = Vec::with_capacity(t_eff);
    for t in 0..t_eff {
        let mut m = Mat::<f64>::zeros(n, n);
        for (s, th) in theta.iter().enumerate().take(t + 1) {
            let row = t - s;
            for i in 0..n {
                for j in 0..n {
                    m[(i, j)] += th[(i, j)] * shocks[(row, j)];
                }
            }
        }
        hd.push(m);
    }
    Ok(hd)
}

/// A single cumulated historical-decomposition cell
/// `HD[t][(i, k)] = sum_{s=0}^t theta[s][(i, k)] E[(t - s, k)]`, computed
/// without materializing the full tensor (the hot path in the narrative
/// contribution checks).
pub(crate) fn hd_cell(
    theta: &[Mat<f64>],
    shocks: MatRef<'_, f64>,
    i: usize,
    k: usize,
    t: usize,
) -> f64 {
    let mut acc = 0.0;
    for s in 0..=t {
        acc += theta[s][(i, k)] * shocks[(t - s, k)];
    }
    acc
}

/// Episode contribution of every shock `k` to variable `i` over the inclusive
/// effective-sample window `[t1, t2]`, computed directly from `theta` and `E`.
///
/// Returns a length-`n` vector `C` where `C[k]` is the change in shock `k`'s
/// cumulated contribution to variable `i`:
///
/// * single period (`t1 == t2`): `C[k] = HD[t1][(i, k)]` (the level
///   contribution at that date);
/// * a window (`t1 < t2`): `C[k] = HD[t2][(i, k)] - HD[t1-1][(i, k)]`, with the
///   convention `HD[-1] = 0` when `t1 == 0`.
///
/// This is the Antolín-Díaz & Rubio-Ramírez (2018) episode measure; the
/// "most / least important contributor" restrictions compare `|C[k]|` across
/// shocks (orientation-free), while a contribution-sign restriction fixes the
/// sign of `C[shock]` (orientation-dependent).
pub(crate) fn episode_contribution_cells(
    theta: &[Mat<f64>],
    shocks: MatRef<'_, f64>,
    i: usize,
    t1: usize,
    t2: usize,
) -> Vec<f64> {
    let n = shocks.ncols();
    (0..n)
        .map(|k| {
            let end = hd_cell(theta, shocks, i, k, t2);
            if t1 == t2 || t1 == 0 {
                end
            } else {
                end - hd_cell(theta, shocks, i, k, t1 - 1)
            }
        })
        .collect()
}

/// The historical decomposition of one identified structural model.
#[derive(Debug, Clone)]
pub struct HistoricalDecomposition {
    n_vars: usize,
    t_eff: usize,
    baseline: Mat<f64>,
    hd: Vec<Mat<f64>>,
    shocks: Mat<f64>,
    theta: Vec<Mat<f64>>,
}

impl HistoricalDecomposition {
    /// Number of variables (and structural shocks).
    pub fn n_vars(&self) -> usize {
        self.n_vars
    }

    /// Effective sample length `T_eff = T - p` (decomposition dates
    /// `0..T_eff`).
    pub fn t_eff(&self) -> usize {
        self.t_eff
    }

    /// The deterministic / initial-condition baseline (`T_eff x n`).
    pub fn baseline(&self) -> MatRef<'_, f64> {
        self.baseline.as_ref()
    }

    /// The decomposition tensor as `T_eff` matrices, each `n x n` with
    /// `(i, j)` the contribution of shock `j` to variable `i` at that date.
    pub fn hd(&self) -> &[Mat<f64>] {
        &self.hd
    }

    /// The structural shocks `E` (`T_eff x n`, row `t` = `eps_t'`).
    pub fn shocks(&self) -> MatRef<'_, f64> {
        self.shocks.as_ref()
    }

    /// The structural IRF `Theta_s` used for the decomposition
    /// (length `horizon + 1`).
    pub fn theta(&self) -> &[Mat<f64>] {
        &self.theta
    }

    /// Episode contribution of shock `k` to variable `i` over `[t1, t2]` (see
    /// [`episode_contribution_cells`] for the convention).
    ///
    /// # Errors
    ///
    /// [`IdentError::RestrictionOutOfRange`] if any index exceeds the model
    /// dimensions or `t1 > t2` or `t2 >= T_eff`.
    pub fn episode_contribution(
        &self,
        i: usize,
        k: usize,
        t1: usize,
        t2: usize,
    ) -> Result<f64, IdentError> {
        if i >= self.n_vars {
            return Err(IdentError::RestrictionOutOfRange {
                what: "response variable",
                index: i,
                bound: self.n_vars,
            });
        }
        if k >= self.n_vars {
            return Err(IdentError::RestrictionOutOfRange {
                what: "structural shock",
                index: k,
                bound: self.n_vars,
            });
        }
        if t1 > t2 || t2 >= self.t_eff {
            return Err(IdentError::RestrictionOutOfRange {
                what: "episode window end",
                index: t2,
                bound: self.t_eff,
            });
        }
        Ok(episode_contribution_cells(&self.theta, self.shocks.as_ref(), i, t1, t2)[k])
    }

    /// Largest absolute violation of the adding-up identity
    /// `y_{p+t, i} = baseline[t, i] + sum_j HD[t][(i, j)]` over all `(t, i)`;
    /// should be at the level of accumulated rounding (`< 1e-9`).
    pub fn adding_up_residual(&self, data: MatRef<'_, f64>, p: usize) -> f64 {
        let mut worst = 0.0f64;
        for t in 0..self.t_eff {
            for i in 0..self.n_vars {
                let mut recon = self.baseline[(t, i)];
                for j in 0..self.n_vars {
                    recon += self.hd[t][(i, j)];
                }
                let resid = (data[(p + t, i)] - recon).abs();
                if resid > worst {
                    worst = resid;
                }
            }
        }
        worst
    }
}

/// Computes the historical decomposition of the reduced-form draw `(b, sigma)`
/// under the structural impact `A0 = P Q` (`P = chol(sigma)` lower, `Q` the
/// rotation; pass `Q = I` for the recursive / Cholesky decomposition).
///
/// `horizon` is the MA truncation for `Theta_s`; it must reach `T_eff - 1` so
/// the decomposition is complete (and the adding-up identity exact).
///
/// # Errors
///
/// * [`IdentError::InvalidArgument`] if `horizon < T_eff - 1` (an incomplete
///   decomposition) or `p == 0`;
/// * [`IdentError::Dimension`] on inconsistent `data` / `b` / `sigma` / `q`
///   shapes;
/// * [`IdentError::Bayes`] on a Cholesky-IRF failure;
/// * [`IdentError::Linalg`] if `sigma` is not positive definite.
pub fn decompose(
    data: MatRef<'_, f64>,
    b: MatRef<'_, f64>,
    sigma: MatRef<'_, f64>,
    q: MatRef<'_, f64>,
    p: usize,
    horizon: usize,
) -> Result<HistoricalDecomposition, IdentError> {
    let n = data.ncols();
    if q.nrows() != n || q.ncols() != n {
        return Err(IdentError::Dimension {
            what: "rotation Q must be n x n",
            expected: n,
            got: q.nrows(),
        });
    }
    let (y, x) = build_regressors(data, p)?;
    let t_eff = y.nrows();
    if horizon + 1 < t_eff {
        return Err(IdentError::InvalidArgument {
            what: "horizon must be at least T_eff - 1 for a complete decomposition",
        });
    }
    let u = reduced_form_residuals(y.as_ref(), x.as_ref(), b);
    let p_chol = jittered_cholesky(sigma)?.factor;
    let w = orthogonalized_residuals(u.as_ref(), p_chol.as_ref())?;
    let e = structural_shocks(w.as_ref(), q);
    // cholesky_irf validates the (b, sigma) shapes and finiteness.
    let base = cholesky_irf(b, sigma, p, horizon)?;
    let theta = structural_irf(&base, q);
    let hd = hd_recursion(&theta, e.as_ref())?;
    let (coefs, intercept) = coefs_and_intercept(b, n, p);
    let baseline = baseline_path(&coefs, &intercept, data, p)?;
    Ok(HistoricalDecomposition {
        n_vars: n,
        t_eff,
        baseline,
        hd,
        shocks: e,
        theta,
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    /// A stable VAR(1) coefficient matrix `b` (`1 + n` rows: intercept then
    /// the lag-1 block) and a positive-definite `Sigma = A A'`.
    fn toy() -> (Mat<f64>, Mat<f64>) {
        let n = 3;
        let phi = [[0.5, 0.1, 0.0], [0.0, 0.4, 0.1], [0.1, 0.0, 0.3]];
        let mut b = Mat::<f64>::zeros(1 + n, n);
        b[(0, 0)] = 0.2;
        b[(0, 1)] = -0.1;
        b[(0, 2)] = 0.05;
        for i in 0..n {
            for v in 0..n {
                b[(1 + v, i)] = phi[i][v];
            }
        }
        let a = [[1.0, 0.0, 0.0], [0.4, 0.9, 0.0], [0.2, 0.3, 0.7]];
        let sigma = Mat::from_fn(n, n, |i, j| {
            a[i].iter().zip(a[j].iter()).map(|(x, y)| x * y).sum()
        });
        (b, sigma)
    }

    /// A deterministic pseudo-random data matrix (xorshift, no tsecon RNG).
    fn toy_data(t: usize, n: usize) -> Mat<f64> {
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        let mut data = Mat::<f64>::zeros(t, n);
        for r in 1..t {
            for c in 0..n {
                data[(r, c)] = 0.3 * data[(r - 1, c)] + next();
            }
        }
        data
    }

    #[test]
    fn adding_up_identity_holds_under_cholesky() -> Result<(), IdentError> {
        let (b, sigma) = toy();
        let data = toy_data(60, 3);
        let eye = Mat::<f64>::identity(3, 3);
        let t_eff = 60 - 1;
        let hd = decompose(
            data.as_ref(),
            b.as_ref(),
            sigma.as_ref(),
            eye.as_ref(),
            1,
            t_eff - 1,
        )?;
        let resid = hd.adding_up_residual(data.as_ref(), 1);
        assert!(resid < 1e-9, "adding-up identity violated: {resid}");
        Ok(())
    }

    #[test]
    fn adding_up_identity_holds_under_rotation() -> Result<(), IdentError> {
        // The identity is rotation-invariant: any orthogonal Q reproduces y.
        let (b, sigma) = toy();
        let data = toy_data(50, 3);
        // A proper 3x3 rotation about the z-axis by 0.6 rad.
        let (s, c) = 0.6f64.sin_cos();
        let q = Mat::from_fn(3, 3, |i, j| match (i, j) {
            (0, 0) => c,
            (0, 1) => -s,
            (1, 0) => s,
            (1, 1) => c,
            (2, 2) => 1.0,
            _ => 0.0,
        });
        let t_eff = 50 - 1;
        let hd = decompose(
            data.as_ref(),
            b.as_ref(),
            sigma.as_ref(),
            q.as_ref(),
            1,
            t_eff - 1,
        )?;
        assert!(hd.adding_up_residual(data.as_ref(), 1) < 1e-9);
        Ok(())
    }

    #[test]
    fn hd_orientation_invariance_of_magnitudes() -> Result<(), IdentError> {
        // Flipping shock column signs leaves |HD| unchanged.
        let (b, sigma) = toy();
        let data = toy_data(40, 3);
        let eye = Mat::<f64>::identity(3, 3);
        let flip = Mat::from_fn(3, 3, |i, j| if i == j { [1.0, -1.0, 1.0][i] } else { 0.0 });
        let t_eff = 40 - 1;
        let a = decompose(
            data.as_ref(),
            b.as_ref(),
            sigma.as_ref(),
            eye.as_ref(),
            1,
            t_eff - 1,
        )?;
        let f = decompose(
            data.as_ref(),
            b.as_ref(),
            sigma.as_ref(),
            flip.as_ref(),
            1,
            t_eff - 1,
        )?;
        for t in 0..t_eff {
            for i in 0..3 {
                for j in 0..3 {
                    assert!((a.hd()[t][(i, j)].abs() - f.hd()[t][(i, j)].abs()).abs() < 1e-12);
                }
            }
        }
        Ok(())
    }

    #[test]
    fn episode_contribution_matches_window_difference() -> Result<(), IdentError> {
        let (b, sigma) = toy();
        let data = toy_data(30, 3);
        let eye = Mat::<f64>::identity(3, 3);
        let hd = decompose(
            data.as_ref(),
            b.as_ref(),
            sigma.as_ref(),
            eye.as_ref(),
            1,
            28,
        )?;
        // Window [3, 10]: C_k must equal HD[10] - HD[2].
        let c = hd.episode_contribution(0, 1, 3, 10)?;
        let expect = hd.hd()[10][(0, 1)] - hd.hd()[2][(0, 1)];
        assert!((c - expect).abs() < 1e-12);
        // Single period [5, 5]: C_k = HD[5].
        let c1 = hd.episode_contribution(2, 0, 5, 5)?;
        assert!((c1 - hd.hd()[5][(2, 0)]).abs() < 1e-12);
        // t1 == 0 window: C_k = HD[t2].
        let c0 = hd.episode_contribution(1, 2, 0, 4)?;
        assert!((c0 - hd.hd()[4][(1, 2)]).abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn decompose_rejects_short_horizon() {
        let (b, sigma) = toy();
        let data = toy_data(20, 3);
        let eye = Mat::<f64>::identity(3, 3);
        // T_eff = 19 needs horizon >= 18; pass 5.
        assert!(matches!(
            decompose(
                data.as_ref(),
                b.as_ref(),
                sigma.as_ref(),
                eye.as_ref(),
                1,
                5
            ),
            Err(IdentError::InvalidArgument { .. })
        ));
    }
}
