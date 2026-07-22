//! Stochastic-search variable selection for the reduced-form BVAR
//! (George, Sun & Ni 2008, J. Econometrics 142: "Bayesian stochastic
//! search for VAR model restrictions").
//!
//! # Model
//!
//! The VAR(`p`) in the crate's stacked regression form,
//!
//! ```text
//! Y = X A + E,   X = [1, y_{t-1}', ..., y_{t-p}'],   rows of E iid N(0, Sigma),
//! ```
//!
//! with `n` variables, `k = 1 + n p` regressors, `A` a `k x n`
//! regressor-by-equation coefficient matrix (the layout of
//! [`crate::NiwPosterior::b_bar`]), and `m = n k` coefficients. The
//! innovation precision is factored `Sigma^{-1} = Psi Psi'` with `Psi`
//! *upper* triangular (positive diagonal `psi_jj`, strictly-upper free
//! elements `eta_ij`, `i < j`) — the GSN device that turns the covariance
//! into a triangular sequence of residual-on-residual regressions, so every
//! Gibbs block is a standard conjugate update.
//!
//! # Spike-and-slab prior
//!
//! Each searchable coefficient `alpha_i` (every lag coefficient; intercepts
//! are never searched) carries a two-component normal mixture governed by a
//! latent Bernoulli inclusion indicator `gamma_i`:
//!
//! ```text
//! alpha_i | gamma_i ~ (1 - gamma_i) N(0, tau0_i^2) + gamma_i N(0, tau1_i^2),
//! gamma_i ~ Bernoulli(pi),
//! ```
//!
//! with a tight "spike" `tau0` and a diffuse "slab" `tau1`. The prior scales
//! are set semi-automatically (GSN §3.2): `tau0_i = c0 se_i`,
//! `tau1_i = c1 se_i` with `c0 << 1 << c1` and `se_i` the unrestricted OLS
//! coefficient standard error, so the estimator is scale-invariant and needs
//! no data transformation. The closed-form OLS standard error uses the
//! Kronecker structure `Cov(vec(A_hat)) = Sigma_hat (x) (X'X)^{-1}`, giving
//! `se_{r,c} = sqrt( Sigma_hat_{cc} [(X'X)^{-1}]_{rr} )` without ever forming
//! the `m x m` covariance. The optional spike-and-slab on the off-diagonal
//! precision elements `eta_ij` (enabled by
//! [`SsvsConfig::ssvs_cov`]) selects error-covariance restrictions the same
//! way; the diagonal `psi_jj^2` carries a diffuse `Gamma(a, rate = b)` prior.
//!
//! The posterior inclusion probability of a coefficient is the Monte-Carlo
//! mean of its `gamma_i` over the retained sweeps — the headline output.
//!
//! # Gibbs sampler
//!
//! One sweep, all randomness through [`tsecon_rng::Stream`] uniforms mapped
//! by inverse CDFs (so every draw is reproducible under the library's
//! substream contract):
//!
//! 1. **coefficients** `alpha | Sigma, gamma`: a Gaussian / SUR update with
//!    precision `P = (Sigma^{-1} (x) X'X) + D^{-1}` (`D^{-1}` the mixture
//!    prior precision) and mean solving `P alpha_bar = vec(X'Y Sigma^{-1})`.
//!    `P` is Cholesky-factored once; the posterior mean is a triangular
//!    solve and the draw adds `Lp'^{-1} z` with `z` standard normal;
//! 2. **indicators** `gamma_i | alpha_i`: independent Bernoulli with the
//!    mixture-odds probability;
//! 3. **precision factor** `Psi | A, omega`: column by column, `psi_jj^2` a
//!    Gamma draw and the strictly-upper `eta_j` a Gaussian draw from the
//!    completed-square conditional (GSN eqs. 15–18);
//! 4. **covariance indicators** `omega_ij | eta_ij` (only when
//!    [`SsvsConfig::ssvs_cov`]): independent Bernoulli, as block 2.
//!
//! Per retained (thinned) sweep the Cholesky-orthogonalized impulse
//! responses are formed by [`crate::cholesky_irf`] on the current
//! `(A, Sigma)`; running means of `gamma`, `A`, `Sigma`, and `omega`
//! accumulate over every post-burn sweep.

use tsecon_linalg::faer::{Mat, MatRef};
use tsecon_linalg::jittered_cholesky;
use tsecon_rng::Stream;
use tsecon_stats::special::inv_gamma_p;

use crate::cholesky_irf;
use crate::convergence::{ess_bulk, rhat_rank};
use crate::dense::{
    backward_solve_in_place, chol_inverse, forward_solve_in_place, positive_uniform, std_normal,
    symmetrize_in_place,
};
use crate::error::BayesError;

/// Configuration for [`bvar_ssvs`]. [`Default`] matches the Python
/// `bvar_ssvs` defaults.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SsvsConfig {
    /// Lag length `p >= 1`.
    pub lags: usize,
    /// Total Gibbs sweeps (including burn-in).
    pub n_draws: usize,
    /// Discarded burn-in sweeps (`burn < n_draws`).
    pub burn: usize,
    /// Spike scale factor `c0` (times the OLS standard error); `c0 << 1`.
    pub c0: f64,
    /// Slab scale factor `c1` (times the OLS standard error); `c1 >> 1`.
    pub c1: f64,
    /// Prior inclusion probability `pi` for the coefficients.
    pub prior_inclusion: f64,
    /// Spike-and-slab on the off-diagonal precision elements `eta`.
    pub ssvs_cov: bool,
    /// Covariance spike standard deviation `kappa0` (used when `ssvs_cov`).
    pub kappa0: f64,
    /// Covariance slab standard deviation `kappa1` (also the single diffuse
    /// slab when `ssvs_cov` is off).
    pub kappa1: f64,
    /// Prior inclusion probability `pi_cov` for the off-diagonal precision.
    pub prior_inclusion_cov: f64,
    /// `Gamma(shape = gamma_a, rate = gamma_b)` prior on `psi_jj^2`.
    pub gamma_a: f64,
    /// Rate of the `Gamma` prior on `psi_jj^2`.
    pub gamma_b: f64,
    /// Impulse-response horizon.
    pub horizon: usize,
    /// Keep every `thin`-th post-burn sweep for the IRF draws (`thin >= 1`).
    pub thin: usize,
    /// Number of independent chains (`>= 1`); `> 1` adds convergence
    /// diagnostics from the per-sweep model-size functional.
    pub n_chains: usize,
}

impl Default for SsvsConfig {
    fn default() -> Self {
        Self {
            lags: 2,
            n_draws: 10_000,
            burn: 2_000,
            c0: 0.1,
            c1: 10.0,
            prior_inclusion: 0.5,
            ssvs_cov: false,
            kappa0: 0.1,
            kappa1: 10.0,
            prior_inclusion_cov: 0.5,
            gamma_a: 0.01,
            gamma_b: 0.01,
            horizon: 16,
            thin: 1,
            n_chains: 1,
        }
    }
}

impl SsvsConfig {
    /// Validates the configuration independently of the data.
    ///
    /// # Errors
    ///
    /// [`BayesError::InvalidArgument`] for `lags == 0`, `burn >= n_draws`,
    /// non-positive scale factors (`c0`, `c1`, `kappa0`, `kappa1`,
    /// `gamma_a`, `gamma_b`), prior probabilities outside `[0, 1]`,
    /// `thin == 0`, or `n_chains == 0`.
    pub fn validate(&self) -> Result<(), BayesError> {
        if self.lags == 0 {
            return Err(BayesError::InvalidArgument {
                what: "lag length p must be at least 1",
            });
        }
        if self.burn >= self.n_draws {
            return Err(BayesError::InvalidArgument {
                what: "burn must be strictly less than n_draws",
            });
        }
        if !(self.c0 > 0.0 && self.c1 > 0.0) {
            return Err(BayesError::InvalidArgument {
                what: "spike/slab scale factors c0, c1 must be strictly positive",
            });
        }
        if !(self.kappa0 > 0.0 && self.kappa1 > 0.0) {
            return Err(BayesError::InvalidArgument {
                what: "covariance scales kappa0, kappa1 must be strictly positive",
            });
        }
        if !(self.gamma_a > 0.0 && self.gamma_b > 0.0) {
            return Err(BayesError::InvalidArgument {
                what: "Gamma prior parameters gamma_a, gamma_b must be strictly positive",
            });
        }
        if !(0.0..=1.0).contains(&self.prior_inclusion)
            || !(0.0..=1.0).contains(&self.prior_inclusion_cov)
        {
            return Err(BayesError::InvalidArgument {
                what: "prior inclusion probabilities must lie in [0, 1]",
            });
        }
        if self.thin == 0 {
            return Err(BayesError::InvalidArgument {
                what: "thin must be at least 1",
            });
        }
        if self.n_chains == 0 {
            return Err(BayesError::InvalidArgument {
                what: "n_chains must be at least 1",
            });
        }
        Ok(())
    }
}

/// The result of [`bvar_ssvs`]: posterior inclusion probabilities,
/// coefficient/covariance means, orthogonalized IRF draws, and diagnostics.
#[derive(Debug, Clone)]
pub struct SsvsResult {
    /// Posterior inclusion probability of each coefficient (`k x n`,
    /// regressor-by-equation). The intercept row (`r = 0`) is `1.0` by
    /// construction (intercepts are never searched).
    pub inclusion_prob: Mat<f64>,
    /// Posterior mean of `A` (`k x n`, same layout as
    /// [`crate::NiwPosterior::b_bar`]).
    pub coef_mean: Mat<f64>,
    /// Posterior mean of the innovation covariance `Sigma` (`n x n`).
    pub sigma_mean: Mat<f64>,
    /// Cholesky-orthogonalized IRFs per retained draw, indexed
    /// `[draw][horizon]`, each an `n x n` matrix with `(i, j)` the response
    /// of variable `i` to structural shock `j`.
    pub irf_draws: Vec<Vec<Mat<f64>>>,
    /// Posterior inclusion probability of the off-diagonal precision
    /// elements `eta_ij` (`n x n`, strictly-upper populated, lower/diagonal
    /// `0.0`); `None` unless [`SsvsConfig::ssvs_cov`] is set.
    pub inclusion_prob_cov: Option<Mat<f64>>,
    /// Average number of included searchable coefficients over the retained
    /// sweeps.
    pub mean_model_size: f64,
    /// Median over the retained sweeps of the Gaussian log-likelihood
    /// `ln p(Y | A, Sigma)` at the drawn parameters.
    pub log_marginal_likelihood_median: f64,
    /// Number of retained (post-burn, thinned, pooled over chains) IRF
    /// draws.
    pub n_draws_kept: usize,
    /// Rank-normalized split R-hat of the per-sweep model-size functional
    /// (only when `n_chains > 1` and the functional is non-degenerate).
    pub rhat: Option<f64>,
    /// Bulk effective sample size of the per-sweep model-size functional
    /// (only when `n_chains > 1`).
    pub ess_bulk: Option<f64>,
}

/// Builds the stacked regression design `(X, Y)` from `data` at lag `p`,
/// intercept then lag blocks (variables in data order) — the crate's
/// regressor layout.
pub(crate) fn build_xy(data: MatRef<'_, f64>, p: usize) -> (Mat<f64>, Mat<f64>) {
    let n = data.ncols();
    let k = 1 + n * p;
    let t_eff = data.nrows() - p;
    let y = Mat::from_fn(t_eff, n, |i, j| data[(p + i, j)]);
    let x = Mat::from_fn(t_eff, k, |i, j| {
        if j == 0 {
            1.0
        } else {
            let l = (j - 1) / n + 1;
            let v = (j - 1) % n;
            data[(p + i - l, v)]
        }
    });
    (x, y)
}

/// Assembles the block-1 posterior precision
/// `P = (Sigma^{-1} (x) X'X) + diag(dinv)` (`m x m`, `m = n k`) directly
/// from the `n x n` `sigma_inv`, the `k x k` `xtx`, and the length-`m`
/// prior-precision diagonal `dinv` (index `i = r + c k`).
pub(crate) fn assemble_precision(
    sigma_inv: MatRef<'_, f64>,
    xtx: MatRef<'_, f64>,
    dinv: &[f64],
) -> Mat<f64> {
    let n = sigma_inv.nrows();
    let k = xtx.nrows();
    let m = n * k;
    let mut p = Mat::<f64>::zeros(m, m);
    for c in 0..n {
        for cp in 0..n {
            let s = sigma_inv[(c, cp)];
            for r in 0..k {
                for rp in 0..k {
                    p[(r + c * k, rp + cp * k)] = s * xtx[(r, rp)];
                }
            }
        }
    }
    for (i, &d) in dinv.iter().enumerate() {
        p[(i, i)] += d;
    }
    p
}

/// The block-1 right-hand side `b = vec(X'Y Sigma^{-1})` (length `m`, index
/// `i = r + c k`) from the `k x n` `xty` and the `n x n` `sigma_inv`.
pub(crate) fn coef_rhs(xty: MatRef<'_, f64>, sigma_inv: MatRef<'_, f64>) -> Vec<f64> {
    let k = xty.nrows();
    let n = xty.ncols();
    let mut b = vec![0.0; n * k];
    for c in 0..n {
        for r in 0..k {
            let mut s = 0.0;
            for cp in 0..n {
                s += xty[(r, cp)] * sigma_inv[(cp, c)];
            }
            b[r + c * k] = s;
        }
    }
    b
}

/// Solves the SPD system `(L L') x = b` for the lower Cholesky factor `L`
/// (forward then backward substitution); returns `x`.
pub(crate) fn chol_solve_vec(l: MatRef<'_, f64>, b: &[f64]) -> Vec<f64> {
    let mut x = b.to_vec();
    forward_solve_in_place(l, &mut x);
    backward_solve_in_place(l, &mut x);
    x
}

/// The mixture-odds inclusion probability
/// `[pi phi(x; 0, v1)] / [pi phi(x; 0, v1) + (1 - pi) phi(x; 0, v0)]` for a
/// spike variance `v0`, a slab variance `v1`, and prior `pi`, computed in
/// the numerically stable log-odds form.
pub(crate) fn inclusion_probability(x: f64, v_slab: f64, v_spike: f64, prior: f64) -> f64 {
    if prior <= 0.0 {
        return 0.0;
    }
    if prior >= 1.0 {
        return 1.0;
    }
    // phi_spike/phi_slab = sqrt(v_slab/v_spike) exp(-x^2/2 (1/v_spike - 1/v_slab)).
    let log_pdf_ratio =
        0.5 * (v_slab.ln() - v_spike.ln()) - 0.5 * x * x * (1.0 / v_spike - 1.0 / v_slab);
    let odds = ((1.0 - prior) / prior) * log_pdf_ratio.exp();
    1.0 / (1.0 + odds)
}

/// Deterministic conditional moments of the block-3 precision-factor column
/// `j` (0-indexed), given the residual cross-product `s = E'E` (`n x n`),
/// the strictly-upper prior-precision diagonal `d_j_inv` (length `j`, the
/// reciprocals `1/kappa^2`), the `Gamma(gamma_a, rate = gamma_b)` prior on
/// `psi_jj^2`, and the effective sample size `t`.
///
/// Returns `(gamma_shape, gamma_rate, m_j, eta_mean_unit)` where
/// `psi_jj^2 ~ Gamma(gamma_shape, rate = gamma_rate)`,
/// `m_j = (S_[j] + diag(d_j_inv))^{-1}` is the `j x j` `eta_j` posterior
/// covariance, and `eta_mean_unit = -m_j s_[j]` is the `eta_j` mean per unit
/// `psi_jj` (the actual mean is `psi_jj * eta_mean_unit`). For `j = 0` the
/// covariance is empty and `gamma_rate = gamma_b + S_00/2`.
pub(crate) fn psi_column_moments(
    s: MatRef<'_, f64>,
    j: usize,
    d_j_inv: &[f64],
    gamma_a: f64,
    gamma_b: f64,
    t: usize,
) -> Result<(f64, f64, Mat<f64>, Vec<f64>), BayesError> {
    let shape = gamma_a + 0.5 * t as f64;
    let s_jj = s[(j, j)];
    if j == 0 {
        let rate = gamma_b + 0.5 * s_jj;
        return Ok((shape, rate, Mat::<f64>::zeros(0, 0), Vec::new()));
    }
    // S_[j] + diag(d_j_inv) and its inverse M_j.
    let mut precision = Mat::from_fn(j, j, |a, b| s[(a, b)]);
    for (a, &d) in d_j_inv.iter().enumerate() {
        precision[(a, a)] += d;
    }
    symmetrize_in_place(&mut precision);
    let chol = jittered_cholesky(precision.as_ref())?;
    let m_j = chol_inverse(chol.factor.as_ref());
    // s_[j] = S[0..j, j].
    let s_vec: Vec<f64> = (0..j).map(|a| s[(a, j)]).collect();
    // eta_mean_unit = -M_j s_[j]; quadratic form s_[j]' M_j s_[j].
    let mut m_s = vec![0.0; j];
    for a in 0..j {
        let mut acc = 0.0;
        for (b, &sv) in s_vec.iter().enumerate() {
            acc += m_j[(a, b)] * sv;
        }
        m_s[a] = acc;
    }
    let quad: f64 = s_vec.iter().zip(&m_s).map(|(&sv, &ms)| sv * ms).sum();
    let rate = gamma_b + 0.5 * (s_jj - quad);
    let eta_mean_unit: Vec<f64> = m_s.iter().map(|&v| -v).collect();
    Ok((shape, rate, m_j, eta_mean_unit))
}

/// One block-1 coefficient draw `alpha` (length `m`, index `i = r + c k`)
/// from `N(P^{-1} b, P^{-1})` with `P = (sigma_inv (x) xtx) + diag(dinv)`
/// and `b = vec(xty sigma_inv)`. Exposed for the conditional-moment tests.
pub(crate) fn draw_coefficients(
    sigma_inv: MatRef<'_, f64>,
    xtx: MatRef<'_, f64>,
    xty: MatRef<'_, f64>,
    dinv: &[f64],
    stream: &mut Stream,
) -> Result<Vec<f64>, BayesError> {
    let m = dinv.len();
    let p = assemble_precision(sigma_inv, xtx, dinv);
    let chol = jittered_cholesky(p.as_ref())?;
    let b = coef_rhs(xty, sigma_inv);
    let mean = chol_solve_vec(chol.factor.as_ref(), &b);
    // u ~ N(0, P^{-1}): draw z ~ N(0, I), solve Lp' u = z.
    let mut u = vec![0.0; m];
    for slot in u.iter_mut() {
        *slot = std_normal(stream)?;
    }
    backward_solve_in_place(chol.factor.as_ref(), &mut u);
    Ok(mean.iter().zip(&u).map(|(&mu, &z)| mu + z).collect())
}

/// One Gamma(`shape`, rate = `rate`) draw by inverse-CDF transform:
/// `inv_gamma_p(shape, u) / rate`, `u` a positive stream uniform.
fn gamma_draw(shape: f64, rate: f64, stream: &mut Stream) -> Result<f64, BayesError> {
    let u = positive_uniform(stream)?;
    Ok(inv_gamma_p(shape, u)? / rate)
}

/// Per-coefficient prior state precomputed once from the OLS standard
/// errors: the spike/slab variances and whether the coefficient is searched.
struct PriorScales {
    /// Slab variance `tau1_i^2 = (c1 se_i)^2` (also the intercept diffuse
    /// variance).
    var_slab: Vec<f64>,
    /// Spike variance `tau0_i^2 = (c0 se_i)^2` (unused on intercept rows).
    var_spike: Vec<f64>,
    /// Whether coefficient `i` is searched (`false` on intercept rows).
    searchable: Vec<bool>,
}

/// The unrestricted-OLS pieces: `X'X`, `X'Y`, `(X'X)^{-1}`, `Sigma_hat`, and
/// the semi-automatic prior scales.
struct OlsPieces {
    xtx: Mat<f64>,
    xty: Mat<f64>,
    sigma_hat_inv: Mat<f64>,
    scales: PriorScales,
}

/// Computes the OLS design cross-products and the GSN semi-automatic prior
/// scales; errors when the sample cannot support the unrestricted fit.
fn ols_pieces(
    x: MatRef<'_, f64>,
    y: MatRef<'_, f64>,
    n: usize,
    k: usize,
    c0: f64,
    c1: f64,
) -> Result<OlsPieces, BayesError> {
    let t_eff = x.nrows();
    if t_eff <= k {
        return Err(BayesError::InsufficientObservations {
            needed: k + 1,
            got: t_eff,
        });
    }
    let mut xtx = x.transpose() * x;
    symmetrize_in_place(&mut xtx);
    let xty = x.transpose() * y;
    let xtx_chol = jittered_cholesky(xtx.as_ref())?;
    let xtx_inv = chol_inverse(xtx_chol.factor.as_ref());
    // A_hat = (X'X)^{-1} X'Y, residuals, Sigma_hat = E'E/(T - k).
    let a_hat = {
        let mut cols = Mat::<f64>::zeros(k, n);
        for c in 0..n {
            let rhs: Vec<f64> = (0..k).map(|r| xty[(r, c)]).collect();
            let sol = chol_solve_vec(xtx_chol.factor.as_ref(), &rhs);
            for (r, &v) in sol.iter().enumerate() {
                cols[(r, c)] = v;
            }
        }
        cols
    };
    let resid = y - x * a_hat.as_ref();
    let mut sigma_hat = resid.transpose() * resid.as_ref();
    let denom = (t_eff - k) as f64;
    for c in 0..n {
        for r in 0..n {
            sigma_hat[(r, c)] /= denom;
        }
    }
    symmetrize_in_place(&mut sigma_hat);
    let sigma_hat_chol = jittered_cholesky(sigma_hat.as_ref())?;
    let sigma_hat_inv = chol_inverse(sigma_hat_chol.factor.as_ref());

    // Semi-automatic scales: se_{r,c} = sqrt(Sigma_hat_cc [(X'X)^-1]_rr).
    let m = n * k;
    let mut var_slab = vec![0.0; m];
    let mut var_spike = vec![0.0; m];
    let mut searchable = vec![false; m];
    for c in 0..n {
        for r in 0..k {
            let se2 = sigma_hat[(c, c)] * xtx_inv[(r, r)];
            let i = r + c * k;
            var_slab[i] = c1 * c1 * se2;
            var_spike[i] = c0 * c0 * se2;
            searchable[i] = r >= 1; // intercept (r = 0) never searched
        }
    }
    Ok(OlsPieces {
        xtx,
        xty,
        sigma_hat_inv,
        scales: PriorScales {
            var_slab,
            var_spike,
            searchable,
        },
    })
}

/// Mutable Gibbs state and the running accumulators shared across chains.
struct Accumulators {
    gamma_sum: Vec<f64>, // length m
    a_sum: Mat<f64>,     // k x n
    sigma_sum: Mat<f64>, // n x n
    omega_sum: Mat<f64>, // n x n (strictly upper)
    count: usize,
    model_size_sum: f64,
    log_lik: Vec<f64>, // per post-burn sweep
    irf_draws: Vec<Vec<Mat<f64>>>,
    chain_model_size: Vec<Vec<f64>>, // per chain, per post-burn sweep
}

/// One-chain Gibbs run, folding into `acc`.
#[allow(clippy::too_many_arguments)]
fn run_chain(
    ols: &OlsPieces,
    n: usize,
    k: usize,
    t_eff: usize,
    x: MatRef<'_, f64>,
    y: MatRef<'_, f64>,
    cfg: &SsvsConfig,
    stream: &mut Stream,
    acc: &mut Accumulators,
) -> Result<(), BayesError> {
    let m = n * k;
    let scales = &ols.scales;

    // State: gamma (searchable start included), omega (start in slab),
    // sigma_inv (start at OLS precision).
    let mut gamma = vec![true; m];
    let mut omega = Mat::<bool>::from_fn(n, n, |_, _| true);
    let mut sigma_inv = ols.sigma_hat_inv.clone();
    let ln_2pi = (2.0 * std::f64::consts::PI).ln();

    let mut chain_sizes = Vec::with_capacity(cfg.n_draws - cfg.burn);

    for sweep in 0..cfg.n_draws {
        // ---- Block 1: coefficients alpha | Sigma, gamma. ----
        let mut dinv = vec![0.0; m];
        for i in 0..m {
            let h2 = if scales.searchable[i] {
                if gamma[i] {
                    scales.var_slab[i]
                } else {
                    scales.var_spike[i]
                }
            } else {
                scales.var_slab[i] // intercept diffuse slab
            };
            dinv[i] = 1.0 / h2;
        }
        let alpha = draw_coefficients(
            sigma_inv.as_ref(),
            ols.xtx.as_ref(),
            ols.xty.as_ref(),
            &dinv,
            stream,
        )?;
        let a = Mat::from_fn(k, n, |r, c| alpha[r + c * k]);

        // ---- Block 2: indicators gamma_i | alpha_i (searchable). ----
        for c in 0..n {
            for r in 0..k {
                let i = r + c * k;
                if !scales.searchable[i] {
                    continue;
                }
                let prob = inclusion_probability(
                    alpha[i],
                    scales.var_slab[i],
                    scales.var_spike[i],
                    cfg.prior_inclusion,
                );
                gamma[i] = positive_uniform(stream)? < prob;
            }
        }

        // ---- Block 3: precision factor Psi | A, omega. ----
        let resid = y - x * a.as_ref();
        let mut s = resid.transpose() * resid.as_ref();
        symmetrize_in_place(&mut s);
        let mut psi = Mat::<f64>::zeros(n, n);
        for j in 0..n {
            // Prior precision diagonal d_j_inv for the j preceding etas.
            let d_j_inv: Vec<f64> = (0..j)
                .map(|i| {
                    let var = if cfg.ssvs_cov && !omega[(i, j)] {
                        cfg.kappa0 * cfg.kappa0
                    } else {
                        cfg.kappa1 * cfg.kappa1
                    };
                    1.0 / var
                })
                .collect();
            let (shape, rate, m_j, eta_mean_unit) =
                psi_column_moments(s.as_ref(), j, &d_j_inv, cfg.gamma_a, cfg.gamma_b, t_eff)?;
            let psi2 = gamma_draw(shape, rate, stream)?;
            let psi_jj = psi2.sqrt();
            psi[(j, j)] = psi_jj;
            if j > 0 {
                // eta_j = psi_jj * eta_mean_unit + chol(M_j) z.
                let m_chol = jittered_cholesky(m_j.as_ref())?.factor;
                let mut z = vec![0.0; j];
                for slot in z.iter_mut() {
                    *slot = std_normal(stream)?;
                }
                for a_idx in 0..j {
                    let mut val = psi_jj * eta_mean_unit[a_idx];
                    for b_idx in 0..=a_idx {
                        val += m_chol[(a_idx, b_idx)] * z[b_idx];
                    }
                    psi[(a_idx, j)] = val;
                }
            }
        }
        // Sigma^{-1} = Psi Psi'; Sigma = (Psi Psi')^{-1}.
        let mut new_sigma_inv = psi.as_ref() * psi.transpose();
        symmetrize_in_place(&mut new_sigma_inv);
        let sinv_chol = jittered_cholesky(new_sigma_inv.as_ref())?;
        let sigma = chol_inverse(sinv_chol.factor.as_ref());
        sigma_inv = new_sigma_inv;

        // ---- Block 4: covariance indicators omega_ij | eta_ij. ----
        if cfg.ssvs_cov {
            for j in 0..n {
                for i in 0..j {
                    let prob = inclusion_probability(
                        psi[(i, j)],
                        cfg.kappa1 * cfg.kappa1,
                        cfg.kappa0 * cfg.kappa0,
                        cfg.prior_inclusion_cov,
                    );
                    omega[(i, j)] = positive_uniform(stream)? < prob;
                }
            }
        }

        // ---- Accumulate over post-burn sweeps. ----
        if sweep >= cfg.burn {
            let post = sweep - cfg.burn;
            let mut model_size = 0.0;
            for ((&g, &searchable), gsum) in gamma
                .iter()
                .zip(&scales.searchable)
                .zip(acc.gamma_sum.iter_mut())
            {
                if g && searchable {
                    *gsum += 1.0;
                    model_size += 1.0;
                }
            }
            for c in 0..n {
                for r in 0..k {
                    acc.a_sum[(r, c)] += a[(r, c)];
                }
            }
            for c in 0..n {
                for r in 0..n {
                    acc.sigma_sum[(r, c)] += sigma[(r, c)];
                }
            }
            if cfg.ssvs_cov {
                for j in 0..n {
                    for i in 0..j {
                        if omega[(i, j)] {
                            acc.omega_sum[(i, j)] += 1.0;
                        }
                    }
                }
            }
            acc.count += 1;
            acc.model_size_sum += model_size;
            chain_sizes.push(model_size);

            // Gaussian log-likelihood at (A, Sigma):
            // -0.5 T n ln(2pi) + T sum ln psi_jj - 0.5 tr(Sigma^{-1} S).
            let mut ln_det_half = 0.0;
            for j in 0..n {
                ln_det_half += psi[(j, j)].ln();
            }
            let mut tr = 0.0;
            for c in 0..n {
                for r in 0..n {
                    tr += sigma_inv[(r, c)] * s[(c, r)];
                }
            }
            let log_lik =
                -0.5 * t_eff as f64 * n as f64 * ln_2pi + t_eff as f64 * ln_det_half - 0.5 * tr;
            acc.log_lik.push(log_lik);

            // Store the IRF for retained (thinned) sweeps.
            if post % cfg.thin == 0 {
                let irf = cholesky_irf(a.as_ref(), sigma.as_ref(), cfg.lags, cfg.horizon)?;
                acc.irf_draws.push(irf);
            }
        }
    }
    acc.chain_model_size.push(chain_sizes);
    Ok(())
}

/// SSVS-BVAR posterior estimation by the 4-block Gibbs sampler of George,
/// Sun & Ni (2008); see the module docs for the model, prior, and sampler.
///
/// `data` is `(T, n)` with variables in columns; the seed drives the
/// reproducible [`Stream`] (one substream per chain when
/// [`SsvsConfig::n_chains`] `> 1`).
///
/// # Errors
///
/// * whatever [`SsvsConfig::validate`] returns (invalid hyperparameters);
/// * [`BayesError::NonFinite`] for NaN/infinite data;
/// * [`BayesError::InsufficientObservations`] when `T <= k + p` (the
///   unrestricted OLS standard errors are undefined);
/// * [`BayesError::Linalg`] on a numerically indefinite conditional (not
///   observed for valid inputs).
pub fn bvar_ssvs(
    data: MatRef<'_, f64>,
    cfg: &SsvsConfig,
    seed: u64,
) -> Result<SsvsResult, BayesError> {
    cfg.validate()?;
    let n = data.ncols();
    if n == 0 {
        return Err(BayesError::InvalidArgument {
            what: "data must have at least one column (variable)",
        });
    }
    for c in 0..n {
        for r in 0..data.nrows() {
            if !data[(r, c)].is_finite() {
                return Err(BayesError::NonFinite { what: "data" });
            }
        }
    }
    let p = cfg.lags;
    let k = 1 + n * p;
    if data.nrows() <= k + p {
        return Err(BayesError::InsufficientObservations {
            needed: k + p + 1,
            got: data.nrows(),
        });
    }
    let (x, y) = build_xy(data, p);
    let t_eff = x.nrows();
    let ols = ols_pieces(x.as_ref(), y.as_ref(), n, k, cfg.c0, cfg.c1)?;

    let m = n * k;
    let mut acc = Accumulators {
        gamma_sum: vec![0.0; m],
        a_sum: Mat::<f64>::zeros(k, n),
        sigma_sum: Mat::<f64>::zeros(n, n),
        omega_sum: Mat::<f64>::zeros(n, n),
        count: 0,
        model_size_sum: 0.0,
        log_lik: Vec::new(),
        irf_draws: Vec::new(),
        chain_model_size: Vec::new(),
    };

    let mut streams = Stream::substreams(seed, cfg.n_chains)?;
    for stream in streams.iter_mut() {
        run_chain(
            &ols,
            n,
            k,
            t_eff,
            x.as_ref(),
            y.as_ref(),
            cfg,
            stream,
            &mut acc,
        )?;
    }

    let cf = acc.count as f64;
    let inclusion_prob = Mat::from_fn(k, n, |r, c| {
        if r == 0 {
            1.0
        } else {
            acc.gamma_sum[r + c * k] / cf
        }
    });
    let coef_mean = Mat::from_fn(k, n, |r, c| acc.a_sum[(r, c)] / cf);
    let sigma_mean = Mat::from_fn(n, n, |r, c| acc.sigma_sum[(r, c)] / cf);
    let inclusion_prob_cov = if cfg.ssvs_cov {
        Some(Mat::from_fn(n, n, |r, c| {
            if r < c {
                acc.omega_sum[(r, c)] / cf
            } else {
                0.0
            }
        }))
    } else {
        None
    };
    let mean_model_size = acc.model_size_sum / cf;
    let log_marginal_likelihood_median = median(&mut acc.log_lik);
    let n_draws_kept = acc.irf_draws.len();

    // Convergence diagnostics from the per-sweep model-size functional.
    let (rhat, ess) = if cfg.n_chains > 1 {
        let n_sweeps = acc.chain_model_size[0].len();
        let chains = Mat::from_fn(cfg.n_chains, n_sweeps, |c, s| acc.chain_model_size[c][s]);
        let r = rhat_rank(chains.as_ref()).ok();
        let e = ess_bulk(chains.as_ref()).ok();
        (r, e)
    } else {
        (None, None)
    };

    Ok(SsvsResult {
        inclusion_prob,
        coef_mean,
        sigma_mean,
        irf_draws: acc.irf_draws,
        inclusion_prob_cov,
        mean_model_size,
        log_marginal_likelihood_median,
        n_draws_kept,
        rhat,
        ess_bulk: ess,
    })
}

/// Median of a slice (sorts in place); `0.0` for an empty slice.
fn median(x: &mut [f64]) -> f64 {
    if x.is_empty() {
        return 0.0;
    }
    x.sort_by(f64::total_cmp);
    let n = x.len();
    if n % 2 == 1 {
        x[n / 2]
    } else {
        0.5 * (x[n / 2 - 1] + x[n / 2])
    }
}

// ---------------------------------------------------------------------------
// Closed-form anchor unit tests (need the pub(crate) deterministic pieces).
//
// The deterministic conditional moments are pinned to an independent NumPy
// re-implementation in fixtures/ssvs.json (see
// fixtures/generate_ssvs_fixtures.py); the stochastic block-1 kernel is
// checked by a conditional-moment Monte Carlo against the same closed form.
// ---------------------------------------------------------------------------
#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use serde_json::Value;

    fn load() -> Value {
        let path = format!("{}/../../fixtures/ssvs.json", env!("CARGO_MANIFEST_DIR"));
        let text = std::fs::read_to_string(&path).unwrap();
        serde_json::from_str(&text).unwrap()
    }

    fn as_mat(v: &Value) -> Mat<f64> {
        let rows = v.as_array().unwrap();
        let ncols = rows[0].as_array().unwrap().len();
        Mat::from_fn(rows.len(), ncols, |i, j| {
            rows[i].as_array().unwrap()[j].as_f64().unwrap()
        })
    }

    fn as_vec(v: &Value) -> Vec<f64> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap())
            .collect()
    }

    /// Block-1 precision assembly and posterior mean match the NumPy golden
    /// P = kron(inv(Sigma), X'X) + diag(1/tau^2) and
    /// alpha_bar = solve(P, vec(X'Y inv(Sigma))) to 1e-10.
    #[test]
    fn anchor_block1_precision_and_mean() {
        let fx = load();
        let a = &fx["anchor_block1"];
        let data = as_mat(&a["data"]);
        let p = a["p"].as_u64().unwrap() as usize;
        let n = a["n"].as_u64().unwrap() as usize;
        let k = 1 + n * p;
        let m = n * k;
        let tau = a["tau"].as_f64().unwrap();
        let sigma = as_mat(&a["sigma"]);
        let sigma_chol = jittered_cholesky(sigma.as_ref()).unwrap();
        let sigma_inv = chol_inverse(sigma_chol.factor.as_ref());

        let (x, y) = build_xy(data.as_ref(), p);
        let mut xtx = x.transpose() * x.as_ref();
        symmetrize_in_place(&mut xtx);
        let xty = x.transpose() * y.as_ref();

        let dinv = vec![1.0 / (tau * tau); m];
        let pmat = assemble_precision(sigma_inv.as_ref(), xtx.as_ref(), &dinv);
        let p_ref = as_mat(&a["P"]);
        for i in 0..m {
            for j in 0..m {
                assert!(
                    (pmat[(i, j)] - p_ref[(i, j)]).abs() <= 1e-10 * p_ref[(i, j)].abs().max(1.0),
                    "P[{i},{j}]: {} vs {}",
                    pmat[(i, j)],
                    p_ref[(i, j)]
                );
            }
        }
        let chol = jittered_cholesky(pmat.as_ref()).unwrap();
        let b = coef_rhs(xty.as_ref(), sigma_inv.as_ref());
        let alpha_bar = chol_solve_vec(chol.factor.as_ref(), &b);
        let ab_ref = as_vec(&a["alpha_bar"]);
        for i in 0..m {
            assert!(
                (alpha_bar[i] - ab_ref[i]).abs() <= 1e-10 * ab_ref[i].abs().max(1.0),
                "alpha_bar[{i}]: {} vs {}",
                alpha_bar[i],
                ab_ref[i]
            );
        }
    }

    /// The mixture-odds inclusion probability matches the raw normal-pdf
    /// ratio to 1e-12 at every stored point.
    #[test]
    fn anchor_bernoulli_matches_pdf_ratio() {
        let fx = load();
        for case in fx["anchor_bernoulli"].as_array().unwrap() {
            let x = case["x"].as_f64().unwrap();
            let v_slab = case["v_slab"].as_f64().unwrap();
            let v_spike = case["v_spike"].as_f64().unwrap();
            let prior = case["prior"].as_f64().unwrap();
            let want = case["prob"].as_f64().unwrap();
            let got = inclusion_probability(x, v_slab, v_spike, prior);
            assert!(
                (got - want).abs() <= 1e-12,
                "inclusion_probability({x}, {v_slab}, {v_spike}, {prior}) = {got} vs {want}"
            );
        }
    }

    /// The block-3 precision-factor column moments (Gamma shape/rate, the
    /// eta covariance M_j, and the unit eta mean) match NumPy to 1e-10, for
    /// both an interior column and the first column (no etas).
    #[test]
    fn anchor_block3_column_moments() {
        let fx = load();
        let a = &fx["anchor_block3"];
        let s = as_mat(&a["S"]);
        let gamma_a = a["gamma_a"].as_f64().unwrap();
        let gamma_b = a["gamma_b"].as_f64().unwrap();
        let t = a["T"].as_u64().unwrap() as usize;
        let j = a["col_j"].as_u64().unwrap() as usize;
        let d_j_inv = as_vec(&a["d_j_inv"]);

        let (shape, rate, m_j, eta_mean_unit) =
            psi_column_moments(s.as_ref(), j, &d_j_inv, gamma_a, gamma_b, t).unwrap();
        assert!((shape - a["shape"].as_f64().unwrap()).abs() <= 1e-12);
        assert!(
            (rate - a["rate_j"].as_f64().unwrap()).abs() <= 1e-10 * a["rate_j"].as_f64().unwrap(),
            "rate_j: {rate}"
        );
        let m_ref = as_mat(&a["M_j"]);
        for r in 0..j {
            for c in 0..j {
                assert!(
                    (m_j[(r, c)] - m_ref[(r, c)]).abs() <= 1e-10 * m_ref[(r, c)].abs().max(1.0),
                    "M_j[{r},{c}]: {} vs {}",
                    m_j[(r, c)],
                    m_ref[(r, c)]
                );
            }
        }
        let em_ref = as_vec(&a["eta_mean_unit"]);
        for i in 0..j {
            assert!(
                (eta_mean_unit[i] - em_ref[i]).abs() <= 1e-10 * em_ref[i].abs().max(1.0),
                "eta_mean_unit[{i}]: {} vs {}",
                eta_mean_unit[i],
                em_ref[i]
            );
        }

        // Column 0: no etas, rate = gamma_b + S_00/2.
        let (shape0, rate0, m0, eta0) =
            psi_column_moments(s.as_ref(), 0, &[], gamma_a, gamma_b, t).unwrap();
        assert!((shape0 - a["shape"].as_f64().unwrap()).abs() <= 1e-12);
        assert!(
            (rate0 - a["rate_0"].as_f64().unwrap()).abs() <= 1e-10 * a["rate_0"].as_f64().unwrap()
        );
        assert_eq!(m0.nrows(), 0);
        assert!(eta0.is_empty());
    }

    /// Conditional-moment Monte Carlo on the stochastic block-1 kernel: with
    /// Sigma and gamma (hence the design and prior precision) fixed, the
    /// `draw_coefficients` draws must have sample mean equal to the closed-
    /// form posterior mean and sample variances equal to the diagonal of
    /// P^{-1}, both within 3 Monte-Carlo standard errors — the direct check
    /// that the Gaussian draw kernel targets N(P^{-1} b, P^{-1}).
    #[test]
    fn block1_draw_matches_conditional_moments() {
        let fx = load();
        let a = &fx["anchor_block1"];
        let data = as_mat(&a["data"]);
        let p = a["p"].as_u64().unwrap() as usize;
        let n = a["n"].as_u64().unwrap() as usize;
        let k = 1 + n * p;
        let m = n * k;
        let tau = a["tau"].as_f64().unwrap();
        let sigma = as_mat(&a["sigma"]);
        let sigma_inv = chol_inverse(jittered_cholesky(sigma.as_ref()).unwrap().factor.as_ref());
        let (x, y) = build_xy(data.as_ref(), p);
        let mut xtx = x.transpose() * x.as_ref();
        symmetrize_in_place(&mut xtx);
        let xty = x.transpose() * y.as_ref();
        let dinv = vec![1.0 / (tau * tau); m];

        // Closed-form targets: mean = alpha_bar, cov = P^{-1}.
        let pmat = assemble_precision(sigma_inv.as_ref(), xtx.as_ref(), &dinv);
        let p_chol = jittered_cholesky(pmat.as_ref()).unwrap();
        let target_mean = chol_solve_vec(
            p_chol.factor.as_ref(),
            &coef_rhs(xty.as_ref(), sigma_inv.as_ref()),
        );
        let p_inv = chol_inverse(p_chol.factor.as_ref());

        let n_draws = 60_000usize;
        let mut stream = Stream::new(20260722);
        let mut sum = vec![0.0; m];
        let mut sumsq = vec![0.0; m];
        for _ in 0..n_draws {
            let al = draw_coefficients(
                sigma_inv.as_ref(),
                xtx.as_ref(),
                xty.as_ref(),
                &dinv,
                &mut stream,
            )
            .unwrap();
            for i in 0..m {
                sum[i] += al[i];
                sumsq[i] += al[i] * al[i];
            }
        }
        let nf = n_draws as f64;
        for i in 0..m {
            let mean = sum[i] / nf;
            let var = (sumsq[i] / nf - mean * mean) * nf / (nf - 1.0);
            let se_mean = (p_inv[(i, i)] / nf).sqrt();
            assert!(
                (mean - target_mean[i]).abs() <= 3.0 * se_mean,
                "coef {i} mean {mean} vs {} (3 se = {})",
                target_mean[i],
                3.0 * se_mean
            );
            let se_var = p_inv[(i, i)] * (2.0 / (nf - 1.0)).sqrt();
            assert!(
                (var - p_inv[(i, i)]).abs() <= 3.0 * se_var,
                "coef {i} var {var} vs {} (3 se = {})",
                p_inv[(i, i)],
                3.0 * se_var
            );
        }
    }
}
