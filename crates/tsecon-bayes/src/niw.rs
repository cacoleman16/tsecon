//! The conjugate Minnesota / Normal-inverse-Wishart BVAR: prior
//! construction, closed-form posterior and marginal likelihood, and
//! posterior sampling of coefficients, covariances, and orthogonalized
//! impulse responses.
//!
//! # Model and conjugate updating
//!
//! The VAR(p) in stacked regression form (Karlsson 2013, Handbook of
//! Economic Forecasting ch. 15, §2; Giannone, Lenza & Primiceri 2015,
//! appendix):
//!
//! ```text
//! y_t' = x_t' B + u_t',   x_t = [1, y_{t-1}', ..., y_{t-p}']',
//! U ~ MN(0, I_T (x) Sigma)                    [rows iid N(0, Sigma)]
//! ```
//!
//! with `n` variables, `k = 1 + n p` regressors, and `T` usable rows after
//! lagging. Under the natural-conjugate prior
//!
//! ```text
//! Sigma ~ IW(S0, v0),
//! vec(B) | Sigma ~ N(vec(B0), Sigma (x) Omega0)
//! ```
//!
//! the posterior is again Normal-inverse-Wishart with
//!
//! ```text
//! Obar = (Omega0^-1 + X'X)^-1
//! Bbar = Obar (Omega0^-1 B0 + X'Y)
//! Sbar = S0 + Y'Y + B0' Omega0^-1 B0 - Bbar' (Omega0^-1 + X'X) Bbar
//! vbar = v0 + T
//! ```
//!
//! and the marginal likelihood is matrix-variate t (Kadiyala & Karlsson
//! 1997, eq. 3.6):
//!
//! ```text
//! ln p(Y) = -(n T / 2) ln pi + (n/2)(ln|Obar| - ln|Omega0|)
//!           + (v0/2) ln|S0| - (vbar/2) ln|Sbar|
//!           + ln Gamma_n(vbar/2) - ln Gamma_n(v0/2)
//! ```
//!
//! with `Gamma_n` the multivariate gamma function. All log-determinants
//! come from Cholesky factors; the `n k x n k` coefficient covariance is
//! never formed (Kronecker identities throughout).
//!
//! # Inverse-Wishart convention
//!
//! `Sigma ~ IW(S, v)` here means density proportional to
//! `|Sigma|^{-(v + n + 1)/2} exp(-tr(S Sigma^{-1}) / 2)` — the *scale*
//! parameterization with mean `S / (v - n - 1)` for `v > n + 1`,
//! equivalently `Sigma^{-1} ~ Wishart(S^{-1}, v)`. This matches R `BVAR`,
//! BEAR, and the GLP replication code; papers using the rate convention
//! swap `S` for `S^{-1}` (a leading cause of cross-package mismatches —
//! see ROADMAP module 05, implementation warning 6).

use tsecon_linalg::faer::{Mat, MatRef};
use tsecon_linalg::{companion_from_var, jittered_cholesky};
use tsecon_rng::Stream;
use tsecon_stats::special::ln_gamma;
use tsecon_stats::{ChiSquared, ContinuousDist};

use crate::dense::{
    backward_solve_in_place, chol_inverse, chol_solve_mat, forward_solve_in_place,
    positive_uniform, std_normal, symmetrize_in_place,
};
use crate::error::BayesError;

/// Lag length of the univariate autoregressions whose residual variances
/// scale the Minnesota prior (the fixture/BEAR convention: AR(4) with
/// intercept, OLS, denominator `T_eff - 5`).
const MINNESOTA_AR_ORDER: usize = 4;

/// Validates a data matrix: finite entries, at least one column.
fn check_data(data: MatRef<'_, f64>) -> Result<(), BayesError> {
    if data.ncols() == 0 {
        return Err(BayesError::InvalidArgument {
            what: "data must have at least one column (variable)",
        });
    }
    for j in 0..data.ncols() {
        for i in 0..data.nrows() {
            if !data[(i, j)].is_finite() {
                return Err(BayesError::NonFinite { what: "data" });
            }
        }
    }
    Ok(())
}

/// Residual variance of an OLS AR(`p`) with intercept fit to `y`,
/// denominator `T_eff - (p + 1)` (the degrees-of-freedom convention of the
/// golden fixture; matches `numpy.linalg.lstsq` up to conditioning).
fn ar_resid_var(y: &[f64], p: usize) -> Result<f64, BayesError> {
    let t = y.len();
    let k = p + 1;
    if t < p + k + 1 {
        return Err(BayesError::InsufficientObservations {
            needed: p + k + 1,
            got: t,
        });
    }
    let t_eff = t - p;
    // Regressors: [1, y_{t-1}, ..., y_{t-p}] for t = p..t-1.
    let x = Mat::from_fn(t_eff, k, |i, j| if j == 0 { 1.0 } else { y[p + i - j] });
    let yy: Vec<f64> = y[p..].to_vec();
    let xtx = x.as_ref().transpose() * x.as_ref();
    let chol = jittered_cholesky(xtx.as_ref())?;
    let mut xty = vec![0.0; k];
    for (j, slot) in xty.iter_mut().enumerate() {
        let mut s = 0.0;
        for (i, &yi) in yy.iter().enumerate() {
            s += x[(i, j)] * yi;
        }
        *slot = s;
    }
    let mut beta = xty;
    forward_solve_in_place(chol.factor.as_ref(), &mut beta);
    backward_solve_in_place(chol.factor.as_ref(), &mut beta);
    let mut rss = 0.0;
    for (i, &yi) in yy.iter().enumerate() {
        let mut fit = 0.0;
        for (j, &bj) in beta.iter().enumerate() {
            fit += x[(i, j)] * bj;
        }
        let r = yi - fit;
        rss += r * r;
    }
    Ok(rss / (t_eff - k) as f64)
}

/// The Minnesota prior in natural-conjugate NIW form (Litterman 1986;
/// Doan, Litterman & Sims 1984; NIW embedding per Kadiyala & Karlsson
/// 1997).
///
/// Prior moments, with regressors ordered `[intercept, lag-1 block
/// (variables in data order), ..., lag-p block]`:
///
/// * `B0`: zero except the own first lag, which is `delta` (0 for
///   growth-rate data, 1 to shrink levels data toward univariate random
///   walks);
/// * `Omega0 = diag(omega)` shared across equations (the Kronecker form),
///   with `omega[0] = lambda0^2` (intercept) and, for lag `l` of variable
///   `j`, `omega = lambda1^2 / (l^(2 lambda3) sigma_j^2)`;
/// * `S0 = diag(sigma_1^2, ..., sigma_n^2)`, `v0 = n + 2` (the smallest
///   integer degrees of freedom giving a finite prior mean `S0 / (v0 - n
///   - 1) = S0`);
/// * `sigma_j^2` are univariate AR(4) OLS residual variances (the common
///   convention; documented because packages differ and results are
///   sensitive — see the module docs of the roadmap).
///
/// Because `Omega0` is shared across equations, the cross-variable scaling
/// `sigma_i^2 / sigma_j^2` of the original Minnesota prior lives in `S0`;
/// the classic cross-variable tightness `lambda2` is not expressible in
/// conjugate form (it breaks the Kronecker structure) and belongs to the
/// independent-NW sampler.
#[derive(Debug, Clone)]
pub struct MinnesotaNiwPrior {
    b0: Mat<f64>,
    omega0_diag: Vec<f64>,
    s0_diag: Vec<f64>,
    v0: f64,
    p: usize,
    n_vars: usize,
}

impl MinnesotaNiwPrior {
    /// Builds the prior from the data (for the AR(4) residual-variance
    /// scales), the lag length `p`, the hyperparameters `lambda0`
    /// (intercept tightness), `lambda1` (overall tightness), `lambda3`
    /// (lag decay), and the own-first-lag prior mean `delta`.
    ///
    /// # Errors
    ///
    /// * [`BayesError::InvalidArgument`] for `p = 0`, non-positive
    ///   `lambda0`/`lambda1`, or negative `lambda3`;
    /// * [`BayesError::NonFinite`] for NaN/infinite data or
    ///   hyperparameters;
    /// * [`BayesError::InsufficientObservations`] when the sample cannot
    ///   support the AR(4) scale regressions;
    /// * [`BayesError::Linalg`] if an AR scale regression is numerically
    ///   singular (e.g. a constant series).
    pub fn new(
        data: MatRef<'_, f64>,
        p: usize,
        lambda0: f64,
        lambda1: f64,
        lambda3: f64,
        delta: f64,
    ) -> Result<Self, BayesError> {
        if p == 0 {
            return Err(BayesError::InvalidArgument {
                what: "lag length p must be at least 1",
            });
        }
        if !(lambda0.is_finite() && lambda1.is_finite() && lambda3.is_finite() && delta.is_finite())
        {
            return Err(BayesError::NonFinite {
                what: "Minnesota hyperparameters (lambda0, lambda1, lambda3, delta)",
            });
        }
        if lambda0 <= 0.0 || lambda1 <= 0.0 {
            return Err(BayesError::InvalidArgument {
                what: "lambda0 and lambda1 must be strictly positive",
            });
        }
        if lambda3 < 0.0 {
            return Err(BayesError::InvalidArgument {
                what: "lambda3 (lag decay) must be nonnegative",
            });
        }
        check_data(data)?;
        let n = data.ncols();
        let k = 1 + n * p;

        let mut s0_diag = Vec::with_capacity(n);
        for j in 0..n {
            let col: Vec<f64> = (0..data.nrows()).map(|i| data[(i, j)]).collect();
            s0_diag.push(ar_resid_var(&col, MINNESOTA_AR_ORDER)?);
        }

        let mut b0 = Mat::<f64>::zeros(k, n);
        for j in 0..n {
            // Own first lag: regressor index 1 + j (lag-1 block).
            b0[(1 + j, j)] = delta;
        }

        let mut omega0_diag = vec![0.0; k];
        omega0_diag[0] = lambda0 * lambda0;
        for l in 1..=p {
            let decay = (l as f64).powf(2.0 * lambda3);
            for (j, &s2) in s0_diag.iter().enumerate() {
                omega0_diag[1 + (l - 1) * n + j] = lambda1 * lambda1 / decay / s2;
            }
        }

        Ok(Self {
            b0,
            omega0_diag,
            s0_diag,
            v0: n as f64 + 2.0,
            p,
            n_vars: n,
        })
    }

    /// Prior coefficient mean `B0` (`k x n`).
    pub fn b0(&self) -> MatRef<'_, f64> {
        self.b0.as_ref()
    }

    /// Diagonal of the shared coefficient prior variance `Omega0` (length
    /// `k`).
    pub fn omega0_diag(&self) -> &[f64] {
        &self.omega0_diag
    }

    /// Diagonal of the inverse-Wishart scale `S0` (the AR(4) residual
    /// variances).
    pub fn s0_diag(&self) -> &[f64] {
        &self.s0_diag
    }

    /// Inverse-Wishart degrees of freedom `v0 = n + 2`.
    pub fn v0(&self) -> f64 {
        self.v0
    }

    /// Lag length `p`.
    pub fn lag_order(&self) -> usize {
        self.p
    }

    /// Number of variables `n`.
    pub fn n_vars(&self) -> usize {
        self.n_vars
    }

    /// Conjugate posterior update on `data` (same variable layout used to
    /// build the prior); see the module docs for the updating equations.
    ///
    /// # Errors
    ///
    /// * [`BayesError::Dimension`] if `data` has a different number of
    ///   columns than the prior was built for;
    /// * [`BayesError::InsufficientObservations`] when fewer than `p + 1`
    ///   rows are supplied;
    /// * [`BayesError::NonFinite`] on NaN/infinite data;
    /// * [`BayesError::Linalg`] if a posterior scale matrix cannot be
    ///   factorized (numerically indefinite — not observed for valid
    ///   priors).
    pub fn posterior(&self, data: MatRef<'_, f64>) -> Result<NiwPosterior, BayesError> {
        check_data(data)?;
        let n = self.n_vars;
        let p = self.p;
        let k = 1 + n * p;
        if data.ncols() != n {
            return Err(BayesError::Dimension {
                what: "data must have the same number of variables as the prior",
                expected: n,
                got: data.ncols(),
            });
        }
        if data.nrows() < p + 1 {
            return Err(BayesError::InsufficientObservations {
                needed: p + 1,
                got: data.nrows(),
            });
        }
        let t_eff = data.nrows() - p;

        // Y (T x n) and X (T x k): intercept, then lag blocks.
        let y = Mat::from_fn(t_eff, n, |i, j| data[(p + i, j)]);
        let x = Mat::from_fn(t_eff, k, |i, j| {
            if j == 0 {
                1.0
            } else {
                let l = (j - 1) / n + 1; // lag 1..p
                let v = (j - 1) % n; // variable within the lag block
                data[(p + i - l, v)]
            }
        });

        // K = Omega0^-1 + X'X and its Cholesky (the only k x k solve).
        let mut kmat = x.as_ref().transpose() * x.as_ref();
        for (i, &w) in self.omega0_diag.iter().enumerate() {
            kmat[(i, i)] += 1.0 / w;
        }
        symmetrize_in_place(&mut kmat);
        let k_chol = jittered_cholesky(kmat.as_ref())?;

        // Bbar = K^-1 (Omega0^-1 B0 + X'Y).
        let mut rhs = x.as_ref().transpose() * y.as_ref();
        for j in 0..n {
            for (i, &w) in self.omega0_diag.iter().enumerate() {
                rhs[(i, j)] += self.b0[(i, j)] / w;
            }
        }
        let b_bar = chol_solve_mat(k_chol.factor.as_ref(), rhs.as_ref());

        // Obar = K^-1 (small k, formed explicitly for reporting/sampling).
        let omega_bar = chol_inverse(k_chol.factor.as_ref());

        // Sbar = S0 + Y'Y + B0' Omega0^-1 B0 - Bbar' K Bbar.
        let mut s_bar = y.as_ref().transpose() * y.as_ref();
        for (j, &s2) in self.s0_diag.iter().enumerate() {
            s_bar[(j, j)] += s2;
        }
        for c in 0..n {
            for r in 0..n {
                let mut s = 0.0;
                for (i, &w) in self.omega0_diag.iter().enumerate() {
                    s += self.b0[(i, r)] * self.b0[(i, c)] / w;
                }
                s_bar[(r, c)] += s;
            }
        }
        let kb = kmat.as_ref() * b_bar.as_ref();
        let btkb = b_bar.as_ref().transpose() * kb.as_ref();
        for c in 0..n {
            for r in 0..n {
                s_bar[(r, c)] -= btkb[(r, c)];
            }
        }
        symmetrize_in_place(&mut s_bar);
        let s_chol = jittered_cholesky(s_bar.as_ref())?;

        let v_bar = self.v0 + t_eff as f64;

        // Matrix-variate-t log marginal likelihood; every hyperparameter-
        // dependent constant retained (roadmap warning 18).
        let ln_pi = std::f64::consts::PI.ln();
        let ln_det_omega_bar = -k_chol.log_det();
        let ln_det_omega0: f64 = self.omega0_diag.iter().map(|w| w.ln()).sum();
        let ln_det_s0: f64 = self.s0_diag.iter().map(|s| s.ln()).sum();
        let ln_det_s_bar = s_chol.log_det();
        let nt = n as f64 * t_eff as f64;
        let log_marginal_likelihood = -0.5 * nt * ln_pi
            + 0.5 * n as f64 * (ln_det_omega_bar - ln_det_omega0)
            + 0.5 * self.v0 * ln_det_s0
            - 0.5 * v_bar * ln_det_s_bar
            + ln_multigamma(0.5 * v_bar, n)?
            - ln_multigamma(0.5 * self.v0, n)?;

        let omega_bar_chol = jittered_cholesky(omega_bar.as_ref())?.factor;

        Ok(NiwPosterior {
            b_bar,
            omega_bar,
            s_bar,
            v_bar,
            log_marginal_likelihood,
            omega_bar_chol,
            s_bar_chol: s_chol.factor,
            p,
            n_vars: n,
        })
    }
}

/// Log of the multivariate gamma function,
/// `ln Gamma_n(a) = n(n-1)/4 ln pi + sum_{j=1..n} ln Gamma(a + (1-j)/2)`
/// (Muirhead 1982, theorem 2.1.12). Requires `a > (n-1)/2`.
fn ln_multigamma(a: f64, n: usize) -> Result<f64, BayesError> {
    if !a.is_finite() || a <= (n as f64 - 1.0) / 2.0 {
        return Err(BayesError::InvalidArgument {
            what: "multivariate gamma requires a > (n - 1)/2",
        });
    }
    let mut s = n as f64 * (n as f64 - 1.0) / 4.0 * std::f64::consts::PI.ln();
    for j in 1..=n {
        s += ln_gamma(a + (1.0 - j as f64) / 2.0);
    }
    Ok(s)
}

/// One joint posterior draw of the VAR coefficients and the innovation
/// covariance.
#[derive(Debug, Clone)]
pub struct NiwDraw {
    /// Coefficient matrix `B` (`k x n`, regressor-by-equation, same layout
    /// as [`NiwPosterior::b_bar`]).
    pub b: Mat<f64>,
    /// Innovation covariance `Sigma` (`n x n`).
    pub sigma: Mat<f64>,
}

/// The Normal-inverse-Wishart posterior of a conjugate BVAR; see the
/// module docs for the updating equations and conventions.
#[derive(Debug, Clone)]
pub struct NiwPosterior {
    b_bar: Mat<f64>,
    omega_bar: Mat<f64>,
    s_bar: Mat<f64>,
    v_bar: f64,
    log_marginal_likelihood: f64,
    omega_bar_chol: Mat<f64>,
    s_bar_chol: Mat<f64>,
    p: usize,
    n_vars: usize,
}

impl NiwPosterior {
    /// Posterior coefficient mean `Bbar` (`k x n`; also the posterior mean
    /// of the matrix-variate-t marginal of `B`).
    pub fn b_bar(&self) -> MatRef<'_, f64> {
        self.b_bar.as_ref()
    }

    /// Posterior coefficient scale `Obar` (`k x k`):
    /// `vec(B) | Sigma, Y ~ N(vec(Bbar), Sigma (x) Obar)`.
    pub fn omega_bar(&self) -> MatRef<'_, f64> {
        self.omega_bar.as_ref()
    }

    /// Posterior inverse-Wishart scale `Sbar` (`n x n`).
    pub fn s_bar(&self) -> MatRef<'_, f64> {
        self.s_bar.as_ref()
    }

    /// Posterior inverse-Wishart degrees of freedom `vbar = v0 + T`.
    pub fn v_bar(&self) -> f64 {
        self.v_bar
    }

    /// Closed-form log marginal likelihood `ln p(Y)` (matrix-variate-t
    /// normalization; module docs).
    pub fn log_marginal_likelihood(&self) -> f64 {
        self.log_marginal_likelihood
    }

    /// Posterior mean of `Sigma`: `Sbar / (vbar - n - 1)` (inverse-Wishart
    /// mean, defined for `vbar > n + 1`).
    ///
    /// # Errors
    ///
    /// [`BayesError::InvalidArgument`] when `vbar <= n + 1` (the mean does
    /// not exist).
    pub fn sigma_posterior_mean(&self) -> Result<Mat<f64>, BayesError> {
        let n = self.n_vars as f64;
        if self.v_bar <= n + 1.0 {
            return Err(BayesError::InvalidArgument {
                what: "inverse-Wishart mean requires vbar > n + 1",
            });
        }
        let denom = self.v_bar - n - 1.0;
        let n = self.n_vars;
        Ok(Mat::from_fn(n, n, |i, j| self.s_bar[(i, j)] / denom))
    }

    /// Lag length `p`.
    pub fn lag_order(&self) -> usize {
        self.p
    }

    /// Number of variables `n`.
    pub fn n_vars(&self) -> usize {
        self.n_vars
    }

    /// One joint draw `(B, Sigma)` from the posterior.
    ///
    /// `Sigma ~ IW(Sbar, vbar)` via the Bartlett decomposition of the
    /// Wishart on `Sbar^-1` (Anderson 2003, theorem 7.2.1): with `Sbar =
    /// Ls Ls'` and `A` lower triangular, `A_jj = sqrt(chi2_{vbar - j})`
    /// (0-indexed `j`), `A_ij ~ N(0,1)` below the diagonal,
    ///
    /// ```text
    /// Sigma^-1 = (Ls^-T A)(Ls^-T A)'   =>   Sigma = M M',  M = Ls A^-T
    /// ```
    ///
    /// so only triangular solves against the cached Cholesky of `Sbar` are
    /// needed — `Sbar` is never inverted.
    ///
    /// Then `vec(B) | Sigma ~ N(vec(Bbar), Sigma (x) Obar)` using the
    /// Kronecker structure: `B = Bbar + Lo Z M'` with `Lo = chol(Obar)`
    /// and `Z` a `k x n` matrix of iid standard normals, since
    /// `Cov(vec(Lo Z M')) = (M M') (x) (Lo Lo') = Sigma (x) Obar`. The
    /// full `nk x nk` covariance is never formed.
    ///
    /// Draw order (fixed, for reproducibility): for each column `j` of
    /// `A`, the chi-squared diagonal then the subdiagonal normals; then
    /// `Z` row-major. All variates are inverse-CDF transforms of
    /// [`Stream`] uniforms (a ziggurat replaces this later; see the crate
    /// docs).
    ///
    /// # Errors
    ///
    /// [`BayesError::Stats`] on quantile-function failures and
    /// [`BayesError::NoConvergence`] if the uniform stream degenerates
    /// (not observed in practice).
    pub fn draw(&self, stream: &mut Stream) -> Result<NiwDraw, BayesError> {
        let (sigma, m_factor) = self.draw_sigma_with_factor(stream)?;
        let k = self.b_bar.nrows();
        let n = self.n_vars;
        let mut z = Mat::<f64>::zeros(k, n);
        for i in 0..k {
            for j in 0..n {
                z[(i, j)] = std_normal(stream)?;
            }
        }
        let loz = self.omega_bar_chol.as_ref() * z.as_ref();
        let mut b = loz.as_ref() * m_factor.as_ref().transpose();
        for j in 0..n {
            for i in 0..k {
                b[(i, j)] += self.b_bar[(i, j)];
            }
        }
        Ok(NiwDraw { b, sigma })
    }

    /// `n_draws` joint posterior draws (see [`NiwPosterior::draw`]).
    ///
    /// # Errors
    ///
    /// As for [`NiwPosterior::draw`].
    pub fn sample(&self, n_draws: usize, stream: &mut Stream) -> Result<Vec<NiwDraw>, BayesError> {
        let mut out = Vec::with_capacity(n_draws);
        for _ in 0..n_draws {
            out.push(self.draw(stream)?);
        }
        Ok(out)
    }

    /// Draws `Sigma ~ IW(Sbar, vbar)` returning both `Sigma` and a square
    /// root `M` with `M M' = Sigma` (see [`NiwPosterior::draw`]).
    fn draw_sigma_with_factor(
        &self,
        stream: &mut Stream,
    ) -> Result<(Mat<f64>, Mat<f64>), BayesError> {
        let n = self.n_vars;
        let mut a = Mat::<f64>::zeros(n, n);
        for j in 0..n {
            let dof = self.v_bar - j as f64;
            let chi2 = ChiSquared::new(dof)?;
            let u = positive_uniform(stream)?;
            a[(j, j)] = chi2.sample_from_uniform(u)?.sqrt();
            for i in (j + 1)..n {
                a[(i, j)] = std_normal(stream)?;
            }
        }
        // M = Ls A^-T, computed as M' = A^-1 Ls' by forward substitution
        // (column c of Ls' is row c of Ls).
        let ls = self.s_bar_chol.as_ref();
        let mut mt = Mat::<f64>::zeros(n, n);
        let mut col = vec![0.0; n];
        for c in 0..n {
            for (r, slot) in col.iter_mut().enumerate() {
                *slot = ls[(c, r)];
            }
            forward_solve_in_place(a.as_ref(), &mut col);
            for (r, &v) in col.iter().enumerate() {
                mt[(r, c)] = v;
            }
        }
        let m_factor = mt.as_ref().transpose().to_owned();
        let mut sigma = m_factor.as_ref() * m_factor.as_ref().transpose();
        symmetrize_in_place(&mut sigma);
        Ok((sigma, m_factor))
    }

    /// Posterior draws of Cholesky-orthogonalized impulse responses:
    /// `n_draws` independent `(B, Sigma)` draws, each mapped through
    /// [`cholesky_irf`] to horizon `horizon`.
    ///
    /// The result is indexed `[draw][horizon]`, each entry an `n x n`
    /// matrix with `(i, j)` the response of variable `i` to structural
    /// shock `j` — retained per draw exactly so pointwise quantile bands
    /// (and later joint bands) can be formed from the raw draws.
    ///
    /// # Errors
    ///
    /// As for [`NiwPosterior::draw`] and [`cholesky_irf`].
    pub fn irf_draws(
        &self,
        n_draws: usize,
        horizon: usize,
        stream: &mut Stream,
    ) -> Result<Vec<Vec<Mat<f64>>>, BayesError> {
        let mut out = Vec::with_capacity(n_draws);
        for _ in 0..n_draws {
            let d = self.draw(stream)?;
            out.push(cholesky_irf(
                d.b.as_ref(),
                d.sigma.as_ref(),
                self.p,
                horizon,
            )?);
        }
        Ok(out)
    }
}

/// Cholesky-orthogonalized impulse responses of one VAR draw to horizon
/// `horizon` (Kilian & Lütkepohl 2017, ch. 4).
///
/// `b` is `k x n` with `k = 1 + n p` in the crate's regressor order
/// (intercept, then lag blocks); `sigma` is the `n x n` innovation
/// covariance. With the companion matrix `F` (via
/// [`tsecon_linalg::companion_from_var`]) and `J = [I_n 0 ... 0]`, the
/// reduced-form MA weights are `Psi_h = J F^h J'` and the orthogonalized
/// responses are
///
/// ```text
/// Theta_h = Psi_h P,   P = chol(sigma) (lower),
/// ```
///
/// so `Theta_h[(i, j)]` is the response of variable `i` at horizon `h` to
/// the `j`-th structural shock (one-standard-deviation impulse; `Theta_0 =
/// P`). The returned vector has length `horizon + 1`.
///
/// # Errors
///
/// * [`BayesError::Dimension`] if `b` and `sigma` shapes are inconsistent
///   with `1 + sigma.nrows() * p` regressors;
/// * [`BayesError::Linalg`] if `sigma` is not positive definite or the
///   companion construction fails;
/// * [`BayesError::NonFinite`] on NaN/infinite entries.
pub fn cholesky_irf(
    b: MatRef<'_, f64>,
    sigma: MatRef<'_, f64>,
    p: usize,
    horizon: usize,
) -> Result<Vec<Mat<f64>>, BayesError> {
    let n = sigma.nrows();
    if sigma.ncols() != n {
        return Err(BayesError::Dimension {
            what: "sigma must be square",
            expected: n,
            got: sigma.ncols(),
        });
    }
    if p == 0 {
        return Err(BayesError::InvalidArgument {
            what: "lag length p must be at least 1",
        });
    }
    let k = 1 + n * p;
    if b.nrows() != k {
        return Err(BayesError::Dimension {
            what: "b must have 1 + n*p rows (intercept plus lag blocks)",
            expected: k,
            got: b.nrows(),
        });
    }
    if b.ncols() != n {
        return Err(BayesError::Dimension {
            what: "b must have one column per variable",
            expected: n,
            got: b.ncols(),
        });
    }
    for j in 0..n {
        for i in 0..k {
            if !b[(i, j)].is_finite() {
                return Err(BayesError::NonFinite { what: "b" });
            }
        }
    }

    // A_l[(i, j)] = coefficient of y_{t-l, j} in equation i.
    let coef_mats: Vec<Mat<f64>> = (1..=p)
        .map(|l| Mat::from_fn(n, n, |i, j| b[(1 + (l - 1) * n + j, i)]))
        .collect();
    let coef_refs: Vec<MatRef<'_, f64>> = coef_mats.iter().map(|m| m.as_ref()).collect();
    let companion = companion_from_var(&coef_refs)?;

    let p_chol = jittered_cholesky(sigma)?.factor;

    let mut out = Vec::with_capacity(horizon + 1);
    let np = n * p;
    let mut f_pow = Mat::<f64>::identity(np, np);
    for _h in 0..=horizon {
        let psi = Mat::from_fn(n, n, |i, j| f_pow[(i, j)]);
        out.push(psi.as_ref() * p_chol.as_ref());
        f_pow = companion.as_ref() * f_pow.as_ref();
    }
    Ok(out)
}
