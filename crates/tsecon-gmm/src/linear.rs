//! Linear instrumental-variables GMM: one-step, two-step efficient, and
//! iterated estimators with heteroskedasticity-robust and HAC weighting, the
//! robust (sandwich) parameter covariance, and the Hansen (1982) J-test of
//! over-identifying restrictions.
//!
//! # Model
//!
//! We estimate `k` parameters `beta` in the linear moment condition
//! `E[z_t (y_t - x_t' beta)] = 0`, where `x_t` are the `k` regressors
//! (some possibly endogenous), `z_t` are the `L >= k` instruments (the
//! included exogenous regressors instrument themselves, statsmodels/
//! linearmodels-style), and `y_t` is the response. Stacking observations,
//! `X` is `n x k`, `Z` is `n x L`, `y` is `n x 1`.
//!
//! The GMM estimator minimizes the criterion `gbar(beta)' W gbar(beta)`
//! where `gbar(beta) = Z'(y - X beta) / n` is the sample moment vector and
//! `W` is an `L x L` positive-definite weighting matrix. For linear moments
//! the minimizer is closed-form:
//!
//! ```text
//! beta_hat(W) = (X'Z W Z'X)^{-1} X'Z W Z'y.
//! ```
//!
//! # Weighting and the efficient two-step estimator
//!
//! The efficient GMM weight is the inverse of the moment-score covariance
//! `S = Avar(sqrt(n) gbar)`. It is unknown, so the two-step estimator
//! (Hansen 1982) plugs in a first-step estimate:
//!
//! * **Step 1** — `W1 = (Z'Z / n)^{-1}` (this makes step 1 numerically the
//!   two-stage least-squares estimator). Estimate `beta1`; residuals
//!   `u1 = y - X beta1`.
//! * **Step 2** — estimate the moment covariance `S(u1)` from the step-1
//!   residuals (see [`GmmWeight`]), set `W2 = S(u1)^{-1}`, and re-estimate
//!   `beta2`.
//!
//! # Covariance and the Hansen J-test — the exact linearmodels convention
//!
//! This crate reproduces `linearmodels` 7.0 `IVGMM(...).fit()` with the
//! default `weight_type="robust"` and `cov_type="robust"` to machine
//! precision. Two conventions matter and were pinned empirically against the
//! golden fixture (`fixtures/gmm.json`):
//!
//! * **Covariance** uses the *general* GMM sandwich, not the efficient
//!   simplification. With `G = Z'X / n` (the `L x k` moment Jacobian), the
//!   estimation weight `W` used in the final step, and the robust moment
//!   covariance `S` **recomputed at the final residuals**,
//!   ```text
//!   Cov(beta) = (1/n) (G' W G)^{-1} (G' W S W G) (G' W G)^{-1}.
//!   ```
//!   Because linearmodels keeps `W = S(u1)^{-1}` (the step-1 weight) while
//!   recomputing `S = S(u2)` at the step-2 residuals, `W != S^{-1}` exactly,
//!   so the sandwich does *not* collapse to `(G' W G)^{-1}/n`; using the
//!   collapsed form reproduces the golden `bse` only to ~5e-5, whereas the
//!   full sandwich matches to ~1e-17.
//! * **Hansen J** uses the *step-2 estimation weight* `W = S(u1)^{-1}`
//!   (the weight actually used to compute `beta2`), evaluated at the step-2
//!   residuals: `J = n * gbar(u2)' W gbar(u2)`. Recomputing `S` at `u2` for
//!   the J-statistic (as one might expect) reproduces the golden only to
//!   ~6e-4; the step-2 weight matches to ~3e-16. Under the null of correct
//!   over-identifying restrictions, `J ~ chi^2(L - k)`.
//!
//! See `tests/golden.rs` for the fixture check documenting these tolerances.

use tsecon_hac::Kernel;
use tsecon_linalg::faer::{Mat, MatRef};
use tsecon_stats::chi2_sf;

use crate::error::GmmError;
use crate::matrix::{
    col_vec, inv_spd, mat_from_cols, mat_from_rowmajor, mat_to_rowmajor, solve_spd,
};

/// How the moment-score covariance `S = Avar(sqrt(n) gbar)` is estimated,
/// both for the efficient two-step weight `W = S^{-1}` and for the robust
/// sandwich covariance meat.
///
/// The moment scores are `g_t = z_t u_t` with `u_t` the estimation
/// residuals; `S` is the (long-run) covariance of `gbar = (1/n) sum_t g_t`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GmmWeight {
    /// Heteroskedasticity-robust White (1980) covariance,
    /// `S = (1/n) sum_t g_t g_t' = (1/n) sum_t z_t z_t' u_t^2`. This is
    /// `linearmodels` `weight_type="robust"` (no small-sample degrees-of-
    /// freedom correction), the convention the golden fixture uses.
    Robust,
    /// Heteroskedasticity-and-autocorrelation-consistent (Newey-West 1987)
    /// covariance, `S = (1/n) [Gamma_0 + sum_{j>=1} w_j (Gamma_j + Gamma_j')]`
    /// with `Gamma_j = sum_{t>j} g_t g_{t-j}'` and kernel weights `w_j` from
    /// [`tsecon_hac::Kernel`] (the library's single kernel owner). Use for
    /// serially correlated moments (e.g. overlapping observations, forecast
    /// errors).
    Hac {
        /// The lag-weighting kernel.
        kernel: Kernel,
        /// Bandwidth in the [`tsecon_hac::Kernel::weight`] convention (for
        /// Bartlett/Parzen/truncated this is the lag-truncation `maxlags`).
        bandwidth: f64,
    },
}

/// The Hansen (1982) J-test of over-identifying restrictions.
///
/// Present only when the model is over-identified (`L > k`); an exactly
/// identified model fits the moments exactly (`gbar = 0`), leaving no
/// restrictions to test.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HansenJ {
    /// The J-statistic `n * gbar' W gbar` at the final estimate, with `W`
    /// the final-step estimation weight.
    pub stat: f64,
    /// Degrees of freedom `L - k` (number of over-identifying restrictions).
    pub dof: usize,
    /// The p-value `P(chi^2(dof) > stat)` (chi-squared survival function).
    pub pval: f64,
}

/// A fitted linear IV-GMM regression.
#[derive(Debug, Clone, PartialEq)]
pub struct GmmFit {
    /// Coefficient estimates `beta`, in the order the regressor columns were
    /// passed.
    pub params: Vec<f64>,
    /// Standard errors `sqrt(diag(Cov(beta)))`, one per parameter.
    pub bse: Vec<f64>,
    /// Parameter covariance `Cov(beta)`, `k x k` row-major (the robust GMM
    /// sandwich; see the module docs).
    pub cov: Vec<f64>,
    /// Residuals `u_t = y_t - x_t' beta`, length `n`.
    pub residuals: Vec<f64>,
    /// Number of observations `n`.
    pub nobs: usize,
    /// Number of moment conditions / instruments `L`.
    pub nmoments: usize,
    /// Number of parameters `k`.
    pub nparams: usize,
    /// Number of GMM estimation steps performed (weight updates + 1): 1 for
    /// one-step / 2SLS, 2 for the two-step estimator, and the iteration count
    /// for the iterated estimator.
    pub steps: usize,
    /// The Hansen J-test, present iff the model is over-identified (`L > k`).
    pub jtest: Option<HansenJ>,
}

/// The robust sandwich covariance (row-major `k x k`), the standard errors,
/// and the (optional, over-identified-only) Hansen J-test — the trio returned
/// by [`Design::cov_and_j`].
type CovAndJ = (Vec<f64>, Vec<f64>, Option<HansenJ>);

/// The assembled design in the several cross-product forms the estimators
/// reuse. Built once, then shared across steps.
struct Design {
    xmat: Mat<f64>, // n x k
    zmat: Mat<f64>, // n x L
    xz: Mat<f64>,   // k x L = X'Z
    zx: Mat<f64>,   // L x k = Z'X
    zy: Mat<f64>,   // L x 1 = Z'y
    zz: Mat<f64>,   // L x L = Z'Z
    y: Vec<f64>,
    n: usize,
    k: usize,
    l: usize,
}

fn check_finite(xs: &[f64], what: &'static str) -> Result<(), GmmError> {
    if xs.iter().any(|v| !v.is_finite()) {
        return Err(GmmError::NonFinite { what });
    }
    Ok(())
}

impl Design {
    /// Validate the inputs and assemble the cross-product matrices.
    fn build(x_cols: &[Vec<f64>], z_cols: &[Vec<f64>], y: &[f64]) -> Result<Self, GmmError> {
        let n = y.len();
        let k = x_cols.len();
        let l = z_cols.len();
        if n == 0 {
            return Err(GmmError::EmptyInput { what: "response y" });
        }
        if k == 0 {
            return Err(GmmError::EmptyInput {
                what: "regressor columns X",
            });
        }
        if l == 0 {
            return Err(GmmError::EmptyInput {
                what: "instrument columns Z",
            });
        }
        for col in x_cols.iter() {
            if col.len() != n {
                return Err(GmmError::DimensionMismatch {
                    what: "regressor column length vs y",
                    expected: n,
                    got: col.len(),
                });
            }
            check_finite(col, "regressor columns X")?;
        }
        for col in z_cols.iter() {
            if col.len() != n {
                return Err(GmmError::DimensionMismatch {
                    what: "instrument column length vs y",
                    expected: n,
                    got: col.len(),
                });
            }
            check_finite(col, "instrument columns Z")?;
        }
        check_finite(y, "response y")?;
        if l < k {
            return Err(GmmError::UnderIdentified {
                moments: l,
                params: k,
            });
        }
        if n <= k {
            return Err(GmmError::DegreesOfFreedom { n, k });
        }

        let xmat = mat_from_cols(x_cols, n);
        let zmat = mat_from_cols(z_cols, n);
        let ymat = col_vec(y);
        let xz = xmat.transpose() * &zmat; // k x L
        let zx = zmat.transpose() * &xmat; // L x k
        let zy = zmat.transpose() * &ymat; // L x 1
        let zz = zmat.transpose() * &zmat; // L x L
        Ok(Self {
            xmat,
            zmat,
            xz,
            zx,
            zy,
            zz,
            y: y.to_vec(),
            n,
            k,
            l,
        })
    }

    /// The step-1 weight `W1 = (Z'Z / n)^{-1}` (makes step 1 the 2SLS
    /// estimator).
    fn initial_weight(&self) -> Result<Mat<f64>, GmmError> {
        let nf = self.n as f64;
        let zz_over_n = Mat::from_fn(self.l, self.l, |i, j| self.zz[(i, j)] / nf);
        inv_spd(zz_over_n.as_ref(), "step-1 weight (Z'Z/n)")
    }

    /// Closed-form point estimate `beta(W) = (X'Z W Z'X)^{-1} X'Z W Z'y`.
    fn point_estimate(&self, w: MatRef<'_, f64>) -> Result<Vec<f64>, GmmError> {
        let xzw = &self.xz * w; // k x L
        let a = &xzw * &self.zx; // k x k (symmetric PD)
        let b = &xzw * &self.zy; // k x 1
        let beta = solve_spd(
            a.as_ref(),
            b.as_ref(),
            "GMM normal equations X'Z W Z'X (weak/collinear instruments?)",
        )?;
        Ok((0..self.k).map(|i| beta[(i, 0)]).collect())
    }

    /// Residuals `u = y - X beta`.
    fn residuals(&self, beta: &[f64]) -> Vec<f64> {
        (0..self.n)
            .map(|t| {
                let mut fit = 0.0;
                for (j, &b) in beta.iter().enumerate() {
                    fit += self.xmat[(t, j)] * b;
                }
                self.y[t] - fit
            })
            .collect()
    }

    /// Robust or HAC moment-score covariance `S` (`L x L`) at residuals `u`.
    fn moment_cov(&self, u: &[f64], weight: GmmWeight) -> Result<Mat<f64>, GmmError> {
        moment_cov(self.zmat.as_ref(), u, self.n, self.l, weight)
    }

    /// The robust GMM sandwich covariance and the Hansen J-test, given the
    /// final-step estimation weight `w`, the moment covariance `s` at the
    /// final residuals, and the final residuals `u` (for `gbar`).
    fn cov_and_j(
        &self,
        w: MatRef<'_, f64>,
        s: MatRef<'_, f64>,
        u: &[f64],
    ) -> Result<CovAndJ, GmmError> {
        let nf = self.n as f64;
        // G = Z'X / n  (L x k moment Jacobian).
        let g = Mat::from_fn(self.l, self.k, |i, j| self.zx[(i, j)] / nf);
        let gt = g.transpose();
        // bread = (G' W G)^{-1}.
        let gtw = gt * w; // k x L
        let bread_arg = &gtw * &g; // k x k
        let bread = inv_spd(
            bread_arg.as_ref(),
            "GMM sandwich bread G'WG (weak/collinear instruments?)",
        )?;
        // meat = G' W S W G.
        let gtws = &gtw * s; // k x L
        let gtwsw = &gtws * w; // k x L
        let meat = &gtwsw * &g; // k x k
                                // Cov = (1/n) bread meat bread.
        let bm = &bread * &meat;
        let cov_mat = Mat::from_fn(self.k, self.k, |i, j| {
            let mut acc = 0.0;
            for m in 0..self.k {
                acc += bm[(i, m)] * bread[(m, j)];
            }
            acc / nf
        });
        let cov = mat_to_rowmajor(cov_mat.as_ref());

        let mut bse = Vec::with_capacity(self.k);
        for i in 0..self.k {
            let v = cov[i * self.k + i];
            if v < 0.0 || !v.is_finite() {
                return Err(GmmError::SingularMatrix {
                    what: "GMM sandwich covariance diagonal (non-PSD moment covariance?)",
                });
            }
            bse.push(v.sqrt());
        }

        // Hansen J = n * gbar' W gbar, gbar = Z'u / n  =>  J = (Z'u)' W (Z'u) / n.
        let jtest = if self.l > self.k {
            let zu = self.zmat.transpose() * &col_vec(u); // L x 1
            let wzu = w * &zu; // L x 1
            let mut quad = 0.0;
            for i in 0..self.l {
                quad += zu[(i, 0)] * wzu[(i, 0)];
            }
            let stat = quad / nf;
            let dof = self.l - self.k;
            let pval = chi2_sf(stat, dof as f64)?;
            Some(HansenJ { stat, dof, pval })
        } else {
            None
        };
        Ok((cov, bse, jtest))
    }
}

/// Robust/HAC moment-score covariance `S` (`L x L`) of `gbar`, from scores
/// `g_t = z_t u_t`. Free function so the nonlinear driver could reuse it.
fn moment_cov(
    zmat: MatRef<'_, f64>,
    u: &[f64],
    n: usize,
    l: usize,
    weight: GmmWeight,
) -> Result<Mat<f64>, GmmError> {
    let nf = n as f64;
    // Scores g_t = z_t * u_t, stored n x L.
    let scores = Mat::from_fn(n, l, |t, j| zmat[(t, j)] * u[t]);

    match weight {
        GmmWeight::Robust => {
            // S = (1/n) sum_t g_t g_t'.
            Ok(Mat::from_fn(l, l, |i, j| {
                let mut acc = 0.0;
                for t in 0..n {
                    acc += scores[(t, i)] * scores[(t, j)];
                }
                acc / nf
            }))
        }
        GmmWeight::Hac { kernel, bandwidth } => {
            if !bandwidth.is_finite() || bandwidth < 0.0 {
                return Err(GmmError::InvalidBandwidth { value: bandwidth });
            }
            // S = (1/n)[Gamma_0 + sum_{j>=1} w_j (Gamma_j + Gamma_j')].
            let mut s = vec![0.0_f64; l * l];
            for lag in 0..n {
                let wj = kernel.weight(lag, bandwidth);
                if lag > 0 && wj == 0.0 && kernel.truncates() {
                    break;
                }
                for t in lag..n {
                    for i in 0..l {
                        let gti = scores[(t, i)];
                        for j in 0..l {
                            let g = gti * scores[(t - lag, j)];
                            if lag == 0 {
                                s[i * l + j] += g;
                            } else {
                                s[i * l + j] += wj * g;
                                s[j * l + i] += wj * g;
                            }
                        }
                    }
                }
            }
            Ok(Mat::from_fn(l, l, |i, j| s[i * l + j] / nf))
        }
    }
}

/// One-step linear GMM with a caller-supplied weighting matrix `weight`.
///
/// Estimates `beta = (X'Z W Z'X)^{-1} X'Z W Z'y` for the given `L x L`
/// weight `W` (row-major), then reports the robust GMM sandwich covariance
/// using `cov_weight` to estimate the moment covariance at the one-step
/// residuals, and the Hansen J-test when over-identified.
///
/// `x_cols` are the `k` regressor columns (include the constant explicitly,
/// statsmodels-style), `z_cols` the `L >= k` instrument columns (the included
/// exogenous regressors appear in both). `weight` must be `L x L` row-major.
///
/// # Errors
///
/// [`GmmError::EmptyInput`] for empty inputs; [`GmmError::DimensionMismatch`]
/// for mismatched column lengths or a mis-sized weight;
/// [`GmmError::UnderIdentified`] if `L < k`; [`GmmError::DegreesOfFreedom`]
/// if `n <= k`; [`GmmError::NonFinite`] on NaN/inf; [`GmmError::SingularMatrix`]
/// if the projected design or moment covariance is singular.
pub fn one_step_gmm(
    x_cols: &[Vec<f64>],
    z_cols: &[Vec<f64>],
    y: &[f64],
    weight: &[f64],
    cov_weight: GmmWeight,
) -> Result<GmmFit, GmmError> {
    let d = Design::build(x_cols, z_cols, y)?;
    let w = mat_from_rowmajor(weight, d.l, "one-step GMM weight matrix (must be L x L)")?;
    finish(&d, w, 1, cov_weight)
}

/// Two-stage least squares as one-step GMM with `W = (Z'Z / n)^{-1}`.
///
/// This is the exactly/over-identified 2SLS point estimator; the reported
/// covariance is the heteroskedasticity-robust ([`GmmWeight::Robust`])
/// sandwich (linearmodels `IV2SLS(...).fit(cov_type="robust")`). When the
/// model is exactly identified (`L == k`) the estimate coincides with the
/// simple IV estimator `(Z'X)^{-1} Z'y` for *any* weight.
///
/// # Errors
///
/// As [`one_step_gmm`].
pub fn two_stage_least_squares(
    x_cols: &[Vec<f64>],
    z_cols: &[Vec<f64>],
    y: &[f64],
) -> Result<GmmFit, GmmError> {
    let d = Design::build(x_cols, z_cols, y)?;
    let w1 = d.initial_weight()?;
    finish(&d, w1, 1, GmmWeight::Robust)
}

/// Two-step efficient linear IV-GMM (Hansen 1982).
///
/// Step 1 uses `W1 = (Z'Z / n)^{-1}` (2SLS); step 2 uses
/// `W2 = S(u1)^{-1}` with the moment covariance estimated from the step-1
/// residuals per `cov_weight`. The covariance and Hansen J follow the exact
/// linearmodels convention documented at the module level (the J-test uses
/// the step-2 weight `W2`, the covariance recomputes `S` at the step-2
/// residuals). With `cov_weight = GmmWeight::Robust` this reproduces
/// `linearmodels` `IVGMM(...).fit()` to machine precision.
///
/// # Errors
///
/// As [`one_step_gmm`], plus propagation from the moment-covariance inverse.
pub fn two_step_gmm(
    x_cols: &[Vec<f64>],
    z_cols: &[Vec<f64>],
    y: &[f64],
    cov_weight: GmmWeight,
) -> Result<GmmFit, GmmError> {
    let d = Design::build(x_cols, z_cols, y)?;
    // Step 1: 2SLS.
    let w1 = d.initial_weight()?;
    let beta1 = d.point_estimate(w1.as_ref())?;
    let u1 = d.residuals(&beta1);
    // Step 2: efficient weight from step-1 residuals.
    let s1 = d.moment_cov(&u1, cov_weight)?;
    let w2 = inv_spd(s1.as_ref(), "step-2 GMM weight S(u1)")?;
    finish(&d, w2, 2, cov_weight)
}

/// Iterated efficient GMM: repeat the (re-weight, re-estimate) loop until the
/// coefficient vector stops moving.
///
/// Starting from the two-step weight, each iteration recomputes
/// `W = S(u)^{-1}` at the current residuals and re-estimates `beta`, stopping
/// when the maximum absolute coefficient change falls below `tol` or
/// `max_iter` weight updates have been performed. At the fixed point
/// `W = S(u)^{-1}` exactly, so the sandwich covariance collapses to the
/// efficient `(G' W G)^{-1} / n`. On well-identified data this typically
/// equals the two-step estimate within a couple of iterations.
///
/// # Errors
///
/// As [`two_step_gmm`]; [`GmmError::InvalidArgument`] if `tol <= 0` or
/// `max_iter == 0`.
pub fn iterated_gmm(
    x_cols: &[Vec<f64>],
    z_cols: &[Vec<f64>],
    y: &[f64],
    cov_weight: GmmWeight,
    tol: f64,
    max_iter: usize,
) -> Result<GmmFit, GmmError> {
    if tol <= 0.0 || !tol.is_finite() {
        return Err(GmmError::InvalidArgument {
            what: "iterated GMM tolerance must be a positive finite number",
        });
    }
    if max_iter == 0 {
        return Err(GmmError::InvalidArgument {
            what: "iterated GMM max_iter must be at least 1",
        });
    }
    let d = Design::build(x_cols, z_cols, y)?;
    // Start at 2SLS.
    let w1 = d.initial_weight()?;
    let mut beta = d.point_estimate(w1.as_ref())?;
    let mut u = d.residuals(&beta);
    let mut steps = 1usize;
    let mut w = w1;
    for _ in 0..max_iter {
        let s = d.moment_cov(&u, cov_weight)?;
        w = inv_spd(s.as_ref(), "iterated GMM weight S(u)")?;
        let beta_new = d.point_estimate(w.as_ref())?;
        steps += 1;
        let delta = beta
            .iter()
            .zip(beta_new.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        beta = beta_new;
        u = d.residuals(&beta);
        if delta < tol {
            break;
        }
    }
    // Covariance/J at the converged estimate, using the final weight `w`.
    let s_final = d.moment_cov(&u, cov_weight)?;
    let (cov, bse, jtest) = d.cov_and_j(w.as_ref(), s_final.as_ref(), &u)?;
    Ok(GmmFit {
        params: beta,
        bse,
        cov,
        residuals: u,
        nobs: d.n,
        nmoments: d.l,
        nparams: d.k,
        steps,
        jtest,
    })
}

/// Shared tail: estimate `beta` with the final weight `w`, compute residuals,
/// the moment covariance at those residuals, and the sandwich cov + J-test.
/// Used by the one-step and 2SLS entry points (where `w` is the final weight).
fn finish(
    d: &Design,
    w: Mat<f64>,
    steps: usize,
    cov_weight: GmmWeight,
) -> Result<GmmFit, GmmError> {
    let beta = d.point_estimate(w.as_ref())?;
    let u = d.residuals(&beta);
    let s = d.moment_cov(&u, cov_weight)?;
    let (cov, bse, jtest) = d.cov_and_j(w.as_ref(), s.as_ref(), &u)?;
    Ok(GmmFit {
        params: beta,
        bse,
        cov,
        residuals: u,
        nobs: d.n,
        nmoments: d.l,
        nparams: d.k,
        steps,
        jtest,
    })
}
