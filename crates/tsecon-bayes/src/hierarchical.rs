//! Hierarchical (empirical-Bayes / ML-II) selection of the Minnesota prior
//! tightness, after Giannone, Lenza & Primiceri (2015, REStat "Prior
//! Selection for Vector Autoregressions").
//!
//! # The idea
//!
//! The conjugate Minnesota / Normal-inverse-Wishart BVAR
//! ([`MinnesotaNiwPrior`]) already returns a *proper* closed-form log
//! marginal likelihood `ln p(Y | lambda)` — the matrix-variate-t
//! normalization of Kadiyala & Karlsson (1997), with every
//! hyperparameter-dependent normalizer (`-(n/2) ln|Omega0|`,
//! `+(n/2) ln|Obar|`) retained. GLP's central observation is that this makes
//! the marginal likelihood a genuine function of the tightness `lambda1`
//! (and, optionally, the lag-decay `lambda3`), so *selecting* the prior
//! reduces to a low-dimensional maximization of that same function — no new
//! likelihood algebra is required.
//!
//! This module maximizes
//!
//! ```text
//! g(lambda1) = ln p(Y | lambda1)                        [hyperprior = None]
//! g(lambda1) = ln p(Y | lambda1) + ln p_hyper(lambda1)  [hyperprior = Glp]
//! ```
//!
//! by a coarse log-spaced grid pre-scan of the closed form (robust to the
//! occasional multimodality of the GLP marginal likelihood) followed by an
//! adaptive Nelder-Mead polish in a [`Bounded`]-reparameterized working
//! space seeded at the grid argmax, then refits the conjugate posterior at
//! the optimum. Under `hyperprior = None` the maximized value is an
//! optimality certificate: `ln p(Y | lambda1_opt) >= ln p(Y | lambda1)` for
//! every fixed `lambda1`, up to the optimizer tolerance.
//!
//! # Hyperprior
//!
//! With `hyperprior = Glp` the objective adds GLP's Gamma hyperprior on the
//! tightness, `lambda1 ~ Gamma(shape a, scale s)` with mode `0.2` and
//! standard deviation `0.4`. Solving `mode = (a - 1) s = 0.2` and
//! `var = a s^2 = 0.16` gives `4 a^2 - 9 a + 4 = 0`, so
//! `a = (9 + sqrt 17) / 8` and `s = 0.2 / (a - 1)`. The optimizer sees the
//! log kernel `(a - 1) ln lambda1 - lambda1 / s` (the argmax-irrelevant
//! constant dropped); the reported [`HierarchicalFit::log_posterior`] adds
//! the exact constant `-a ln s - ln Gamma(a)` back for honesty.
//!
//! The [`Bounded`] map is used purely to enforce the box `lo < lambda1 < hi`;
//! it is composed with [`TransformedObjective::new`] (*not*
//! `with_log_jacobian`), because the transform is a domain map, not a change
//! of density — adding the log-Jacobian would shift the argmax off the
//! natural-parameter mode GLP report.

use tsecon_linalg::faer::MatRef;
use tsecon_optim::{
    minimize, Bounded, FnObjective, Method, NelderMeadOptions, Transform, TransformedObjective,
};
use tsecon_stats::special::ln_gamma;

use crate::error::BayesError;
use crate::niw::{MinnesotaNiwPrior, NiwPosterior};

/// Search box for the lag-decay `lambda3` when [`HierarchicalConfig::optimize_lambda3`]
/// is set (the core ships the 1-D `lambda1`-only selection; this 2-D path is
/// behind the flag). `lambda3 >= 0` is required by the prior; the box keeps
/// the decay in the empirically sensible `(1e-4, 5)`.
const LAMBDA3_LO: f64 = 1e-4;
const LAMBDA3_HI: f64 = 5.0;

/// A hyperprior on the Minnesota overall tightness `lambda1`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Hyperprior {
    /// Flat (improper) hyperprior: pure ML-II / empirical Bayes — the
    /// objective is the marginal likelihood itself. The default.
    #[default]
    None,
    /// The Giannone-Lenza-Primiceri (2015) Gamma hyperprior with mode `0.2`
    /// and standard deviation `0.4`; the objective becomes the log posterior
    /// (marginal likelihood times hyperprior), i.e. MAP-II selection.
    Glp,
}

impl Hyperprior {
    /// GLP Gamma shape `a = (9 + sqrt 17) / 8` and scale `s = 0.2 / (a - 1)`
    /// (mode `0.2`, standard deviation `0.4`).
    fn glp_params() -> (f64, f64) {
        let a = (9.0 + 17.0_f64.sqrt()) / 8.0;
        let s = 0.2 / (a - 1.0);
        (a, s)
    }

    /// The log kernel added to `ln p(Y | lambda1)` for the maximization (the
    /// argmax-irrelevant normalizing constant is dropped).
    fn log_kernel(self, lambda1: f64) -> f64 {
        match self {
            Hyperprior::None => 0.0,
            Hyperprior::Glp => {
                let (a, s) = Self::glp_params();
                (a - 1.0) * lambda1.ln() - lambda1 / s
            }
        }
    }

    /// The full log hyperprior density at `lambda1`, including the
    /// normalizing constant `-a ln s - ln Gamma(a)`, used to report the log
    /// posterior honestly.
    fn log_prior(self, lambda1: f64) -> f64 {
        match self {
            Hyperprior::None => 0.0,
            Hyperprior::Glp => {
                let (a, s) = Self::glp_params();
                (a - 1.0) * lambda1.ln() - lambda1 / s - a * s.ln() - ln_gamma(a)
            }
        }
    }
}

/// Configuration for [`bvar_hierarchical`]. [`Default`] matches the Python
/// `bvar_hierarchical` defaults.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HierarchicalConfig {
    /// Intercept tightness (fixed, diffuse); Minnesota `lambda0`.
    pub lambda0: f64,
    /// Lag decay; Minnesota `lambda3` (fixed unless
    /// [`optimize_lambda3`](HierarchicalConfig::optimize_lambda3) is set).
    pub lambda3: f64,
    /// Own-first-lag prior mean `delta` (0 for growth rates, 1 for
    /// levels / random-walk shrinkage).
    pub delta: f64,
    /// Starting tightness and the "fixed-lambda" reference whose marginal
    /// likelihood the optimum is certified against
    /// ([`HierarchicalFit::lambda1_fixed_log_ml`]).
    pub lambda1_init: f64,
    /// Lower edge of the `lambda1` search box (strictly positive).
    pub lambda1_lo: f64,
    /// Upper edge of the `lambda1` search box.
    pub lambda1_hi: f64,
    /// Jointly optimize `lambda3` with `lambda1` (2-D; behind this flag).
    pub optimize_lambda3: bool,
    /// The hyperprior on `lambda1`.
    pub hyperprior: Hyperprior,
    /// Number of log-spaced grid points in the pre-scan (`0` disables it).
    pub n_grid: usize,
    /// Nelder-Mead iteration budget.
    pub max_iter: usize,
    /// Nelder-Mead simplex-size and function-spread tolerance.
    pub tol: f64,
}

impl Default for HierarchicalConfig {
    fn default() -> Self {
        Self {
            lambda0: 100.0,
            lambda3: 1.0,
            delta: 0.0,
            lambda1_init: 0.2,
            lambda1_lo: 1e-4,
            lambda1_hi: 10.0,
            optimize_lambda3: false,
            hyperprior: Hyperprior::None,
            n_grid: 25,
            max_iter: 200,
            tol: 1e-8,
        }
    }
}

/// The result of [`bvar_hierarchical`]: the selected tightness, the
/// refitted conjugate posterior, and the diagnostics needed to inspect and
/// certify the selection.
#[derive(Debug, Clone)]
pub struct HierarchicalFit {
    /// The selected overall tightness `lambda1`.
    pub lambda1: f64,
    /// The selected lag decay `lambda3` (equal to the input unless
    /// [`HierarchicalConfig::optimize_lambda3`] was set).
    pub lambda3: f64,
    /// `ln p(Y | lambda_opt)` alone (excludes the hyperprior).
    pub log_ml: f64,
    /// `ln p(Y | lambda_opt) + ln p_hyper(lambda_opt)`; equals
    /// [`log_ml`](HierarchicalFit::log_ml) when the hyperprior is
    /// [`Hyperprior::None`].
    pub log_posterior: f64,
    /// The conjugate NIW posterior refitted at the optimum (coefficient and
    /// covariance means, sampling, IRFs).
    pub posterior: NiwPosterior,
    /// The pre-scan grid of `lambda1` values (empty when `n_grid = 0`).
    pub grid_lambda1: Vec<f64>,
    /// The closed-form `ln p(Y | lambda1)` at each grid point (the marginal
    /// likelihood profile; excludes the hyperprior).
    pub grid_log_ml: Vec<f64>,
    /// `ln p(Y | lambda1_init)`, the "fixed-lambda" reference the optimum is
    /// certified to dominate.
    pub lambda1_fixed_log_ml: f64,
    /// Whether the Nelder-Mead polish satisfied a convergence test.
    pub converged: bool,
    /// Objective evaluations (grid pre-scan plus polish).
    pub n_evals: usize,
}

/// Selects the Minnesota tightness by maximizing the closed-form marginal
/// likelihood (Giannone-Lenza-Primiceri 2015 empirical-Bayes / ML-II
/// selection) and returns the conjugate BVAR refitted at the optimum.
///
/// `data` is `(T, n)` with variables in columns; `p` is the lag length. See
/// [`HierarchicalConfig`] for the hyperparameters and the module docs for
/// the method.
///
/// # Errors
///
/// * [`BayesError::InvalidArgument`] for `p = 0`, a non-positive
///   `lambda1_lo`, `lambda1_lo >= lambda1_hi`, or a non-positive
///   `lambda1_init`;
/// * [`BayesError::NonFinite`] for non-finite search bounds;
/// * whatever [`MinnesotaNiwPrior::new`] / [`MinnesotaNiwPrior::posterior`]
///   return when the *refit at the optimum* fails (insufficient
///   observations, singular AR scale regressions, non-finite data);
/// * [`BayesError::NoConvergence`] if the optimizer itself errors (as
///   opposed to merely not meeting a tolerance, which is reported through
///   [`HierarchicalFit::converged`]).
pub fn bvar_hierarchical(
    data: MatRef<'_, f64>,
    p: usize,
    cfg: &HierarchicalConfig,
) -> Result<HierarchicalFit, BayesError> {
    if p == 0 {
        return Err(BayesError::InvalidArgument {
            what: "lag length p must be at least 1",
        });
    }
    if !(cfg.lambda1_lo.is_finite() && cfg.lambda1_hi.is_finite()) {
        return Err(BayesError::NonFinite {
            what: "lambda1 search bounds",
        });
    }
    if !(cfg.lambda1_lo > 0.0 && cfg.lambda1_lo < cfg.lambda1_hi) {
        return Err(BayesError::InvalidArgument {
            what: "require 0 < lambda1_lo < lambda1_hi",
        });
    }
    if !(cfg.lambda1_init.is_finite() && cfg.lambda1_init > 0.0) {
        return Err(BayesError::InvalidArgument {
            what: "lambda1_init must be finite and positive",
        });
    }

    let lo = cfg.lambda1_lo;
    let hi = cfg.lambda1_hi;
    let hyper = cfg.hyperprior;

    // Grid pre-scan: the lambda1 marginal-likelihood profile at the fixed
    // lambda3. grid_log_ml stores the raw ML (the returned profile); the
    // Nelder-Mead seed is the grid point maximizing the full objective
    // (ML + hyperprior kernel), which coincides with the ML argmax when the
    // hyperprior is flat.
    let mut grid_lambda1 = Vec::with_capacity(cfg.n_grid);
    let mut grid_log_ml = Vec::with_capacity(cfg.n_grid);
    let mut best_obj = f64::NEG_INFINITY;
    let mut seed_lambda1 = cfg.lambda1_init;
    if cfg.n_grid > 0 {
        let log_lo = lo.ln();
        let log_hi = hi.ln();
        for i in 0..cfg.n_grid {
            let t = if cfg.n_grid == 1 {
                0.0
            } else {
                i as f64 / (cfg.n_grid as f64 - 1.0)
            };
            let lam = (log_lo + t * (log_hi - log_lo)).exp();
            let ml = eval_log_ml(data, p, cfg.lambda0, lam, cfg.lambda3, cfg.delta);
            let obj = ml + hyper.log_kernel(lam);
            if obj > best_obj {
                best_obj = obj;
                seed_lambda1 = lam;
            }
            grid_lambda1.push(lam);
            grid_log_ml.push(ml);
        }
    }
    let n_grid_evals = grid_lambda1.len();

    // Keep the seed strictly inside the open box for the logit map.
    let eps = (hi - lo) * 1e-9;
    let seed_lambda1 = seed_lambda1.clamp(lo + eps, hi - eps);

    let (lambda1_opt, lambda3_opt, converged, polish_evals) = if cfg.optimize_lambda3 {
        optimize_2d(data, p, cfg, hyper, seed_lambda1)?
    } else {
        optimize_1d(data, p, cfg, hyper, seed_lambda1, lo, hi)?
    };

    // Refit the conjugate posterior at the optimum (the returned drop-in
    // BVAR). Errors here are genuine model failures and propagate.
    let prior = MinnesotaNiwPrior::new(data, p, cfg.lambda0, lambda1_opt, lambda3_opt, cfg.delta)?;
    let posterior = prior.posterior(data)?;
    let log_ml = posterior.log_marginal_likelihood();
    let log_posterior = log_ml + hyper.log_prior(lambda1_opt);
    let lambda1_fixed_log_ml = eval_log_ml(
        data,
        p,
        cfg.lambda0,
        cfg.lambda1_init,
        cfg.lambda3,
        cfg.delta,
    );

    Ok(HierarchicalFit {
        lambda1: lambda1_opt,
        lambda3: lambda3_opt,
        log_ml,
        log_posterior,
        posterior,
        grid_lambda1,
        grid_log_ml,
        lambda1_fixed_log_ml,
        converged,
        n_evals: n_grid_evals + polish_evals,
    })
}

/// Closed-form `ln p(Y | lambda1, lambda3)` for a Minnesota-NIW prior; any
/// prior/posterior failure returns `-inf` so the optimizer treats the point
/// as infeasible (the `ObjectiveFn` non-finite contract).
fn eval_log_ml(
    data: MatRef<'_, f64>,
    p: usize,
    lambda0: f64,
    lambda1: f64,
    lambda3: f64,
    delta: f64,
) -> f64 {
    match MinnesotaNiwPrior::new(data, p, lambda0, lambda1, lambda3, delta) {
        Ok(prior) => match prior.posterior(data) {
            Ok(post) => post.log_marginal_likelihood(),
            Err(_) => f64::NEG_INFINITY,
        },
        Err(_) => f64::NEG_INFINITY,
    }
}

/// 1-D polish over `lambda1` in the [`Bounded`] working space, seeded at
/// `seed_lambda1`. Returns `(lambda1_opt, lambda3, converged, fevals)`.
fn optimize_1d(
    data: MatRef<'_, f64>,
    p: usize,
    cfg: &HierarchicalConfig,
    hyper: Hyperprior,
    seed_lambda1: f64,
    lo: f64,
    hi: f64,
) -> Result<(f64, f64, bool, usize), BayesError> {
    let bounded = Bounded::new(lo, hi).map_err(|_| BayesError::InvalidArgument {
        what: "invalid lambda1 search bounds",
    })?;
    let lambda0 = cfg.lambda0;
    let lambda3 = cfg.lambda3;
    let delta = cfg.delta;
    let inner = FnObjective::new(move |theta: &[f64]| {
        let l = theta[0];
        let ml = eval_log_ml(data, p, lambda0, l, lambda3, delta);
        -(ml + hyper.log_kernel(l))
    });
    let mut obj = TransformedObjective::new(inner, bounded);
    let z0 = bounded
        .inverse_vec(&[seed_lambda1])
        .map_err(|_| BayesError::InvalidArgument {
            what: "seed lambda1 outside search box",
        })?;
    let opts = NelderMeadOptions {
        x_tol: cfg.tol,
        f_tol: cfg.tol,
        max_iter: Some(cfg.max_iter),
        ..Default::default()
    };
    let res = minimize(&mut obj, &z0, &Method::NelderMead(opts)).map_err(|_| {
        BayesError::NoConvergence {
            what: "hierarchical lambda1 optimization",
        }
    })?;
    let lambda1_opt = bounded
        .forward_vec(&res.x)
        .map_err(|_| BayesError::NoConvergence {
            what: "hierarchical lambda1 optimization",
        })?[0];
    Ok((lambda1_opt, lambda3, res.converged, res.fevals))
}

/// 2-D polish over `(lambda1, lambda3)` in a manual per-coordinate logistic
/// working space (the optional flagged path). Returns
/// `(lambda1_opt, lambda3_opt, converged, fevals)`.
fn optimize_2d(
    data: MatRef<'_, f64>,
    p: usize,
    cfg: &HierarchicalConfig,
    hyper: Hyperprior,
    seed_lambda1: f64,
) -> Result<(f64, f64, bool, usize), BayesError> {
    let lo1 = cfg.lambda1_lo;
    let hi1 = cfg.lambda1_hi;
    let lambda0 = cfg.lambda0;
    let delta = cfg.delta;
    let mut obj = FnObjective::new(move |z: &[f64]| {
        let l1 = logistic(z[0], lo1, hi1);
        let l3 = logistic(z[1], LAMBDA3_LO, LAMBDA3_HI);
        let ml = eval_log_ml(data, p, lambda0, l1, l3, delta);
        -(ml + hyper.log_kernel(l1))
    });
    let seed_l3 = cfg
        .lambda3
        .clamp(LAMBDA3_LO * 1.000_001, LAMBDA3_HI * 0.999_999);
    let z0 = [
        inv_logistic(seed_lambda1, lo1, hi1),
        inv_logistic(seed_l3, LAMBDA3_LO, LAMBDA3_HI),
    ];
    let opts = NelderMeadOptions {
        x_tol: cfg.tol,
        f_tol: cfg.tol,
        max_iter: Some(cfg.max_iter),
        ..Default::default()
    };
    let res = minimize(&mut obj, &z0, &Method::NelderMead(opts)).map_err(|_| {
        BayesError::NoConvergence {
            what: "hierarchical (lambda1, lambda3) optimization",
        }
    })?;
    let l1 = logistic(res.x[0], lo1, hi1);
    let l3 = logistic(res.x[1], LAMBDA3_LO, LAMBDA3_HI);
    Ok((l1, l3, res.converged, res.fevals))
}

/// Scaled logistic `lo + (hi - lo) / (1 + e^{-z})`, stable for large `|z|`
/// (matches [`Bounded`]'s forward map per coordinate).
fn logistic(z: f64, lo: f64, hi: f64) -> f64 {
    let s = if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let e = z.exp();
        e / (1.0 + e)
    };
    lo + (hi - lo) * s
}

/// Inverse of [`logistic`]: the logit of the rescaled coordinate.
fn inv_logistic(theta: f64, lo: f64, hi: f64) -> f64 {
    (theta - lo).ln() - (hi - theta).ln()
}
