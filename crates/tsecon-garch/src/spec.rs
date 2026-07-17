//! Model specification: mean equation, conditional-variance equation,
//! innovation distribution, parameter layout, and admissibility checks.

use crate::error::GarchError;

/// The conditional-mean equation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeanSpec {
    /// `y_t = eps_t` — no mean parameters (arch's `ZeroMean`).
    Zero,
    /// `y_t = mu + eps_t` — one parameter `mu` (arch's `ConstantMean`).
    Constant,
}

/// The conditional-variance equation.
///
/// Lag-count convention follows Kevin Sheppard's `arch` package: `p` counts
/// symmetric shock terms (`alpha`), `o` counts asymmetric terms (`gamma`),
/// `q` counts variance lags (`beta`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VolSpec {
    /// GARCH(p, q) (Bollerslev 1986):
    ///
    /// ```text
    /// sigma2_t = omega + sum_i alpha_i eps_{t-i}^2 + sum_j beta_j sigma2_{t-j}
    /// ```
    Garch {
        /// Number of `alpha` (squared-shock) lags; `p >= 1`.
        p: usize,
        /// Number of `beta` (variance) lags; `q >= 0` (`q = 0` is ARCH(p),
        /// Engle 1982).
        q: usize,
    },
    /// GJR-GARCH(p, o, q) (Glosten-Jagannathan-Runkle 1993):
    ///
    /// ```text
    /// sigma2_t = omega + sum_i alpha_i eps_{t-i}^2
    ///                  + sum_i gamma_i eps_{t-i}^2 1[eps_{t-i} < 0]
    ///                  + sum_j beta_j sigma2_{t-j}
    /// ```
    Gjr {
        /// Number of `alpha` lags; `p >= 1`.
        p: usize,
        /// Number of `gamma` (threshold) lags; `o >= 1` (with `o = 0` use
        /// [`VolSpec::Garch`]).
        o: usize,
        /// Number of `beta` lags; `q >= 0`.
        q: usize,
    },
    /// EGARCH(p, o, q) in `arch`'s formulation (after Nelson 1991):
    ///
    /// ```text
    /// ln sigma2_t = omega + sum_i alpha_i (|z_{t-i}| - sqrt(2/pi))
    ///                     + sum_i gamma_i z_{t-i}
    ///                     + sum_j beta_j ln sigma2_{t-j},   z = eps / sigma
    /// ```
    ///
    /// The centering constant is `E|z| = sqrt(2/pi)` of the standard
    /// normal regardless of the innovation distribution — an `arch`
    /// convention this crate reproduces for cross-package parity.
    Egarch {
        /// Number of `alpha` (magnitude) lags; `p >= 1`.
        p: usize,
        /// Number of `gamma` (sign) lags; `o >= 0`.
        o: usize,
        /// Number of `beta` (log-variance) lags; `q >= 0`.
        q: usize,
    },
}

impl VolSpec {
    /// The `(p, o, q)` lag counts (`o = 0` for [`VolSpec::Garch`]).
    pub fn lags(&self) -> (usize, usize, usize) {
        match *self {
            VolSpec::Garch { p, q } => (p, 0, q),
            VolSpec::Gjr { p, o, q } | VolSpec::Egarch { p, o, q } => (p, o, q),
        }
    }

    /// Number of variance-equation parameters: `1 + p + o + q`.
    pub fn n_params(&self) -> usize {
        let (p, o, q) = self.lags();
        1 + p + o + q
    }

    /// The largest lag appearing in the recursion.
    pub fn max_lag(&self) -> usize {
        let (p, o, q) = self.lags();
        p.max(o).max(q)
    }
}

/// The innovation distribution (both are mean-zero, unit-variance).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistSpec {
    /// Standard normal innovations; QMLE remains consistent under
    /// misspecification (Bollerslev-Wooldridge 1992). No parameters.
    Normal,
    /// Standardized Student-t innovations with `nu > 2` degrees of freedom
    /// (Bollerslev 1987) — the unit-variance rescaling of Student's t,
    /// [`tsecon_stats::Standardized::student_t`]. One parameter `nu`,
    /// ordered last.
    StudentT,
}

impl DistSpec {
    /// Number of distribution parameters (0 for normal, 1 for Student-t).
    pub fn n_params(&self) -> usize {
        match self {
            DistSpec::Normal => 0,
            DistSpec::StudentT => 1,
        }
    }
}

/// A complete univariate volatility-model specification.
///
/// The parameter vector is ordered exactly as `arch` orders it: mean
/// parameters, then `omega`, `alpha`s, `gamma`s, `beta`s, then distribution
/// parameters. [`GarchSpec::param_names`] spells the layout out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GarchSpec {
    /// The conditional-mean equation.
    pub mean: MeanSpec,
    /// The conditional-variance equation.
    pub vol: VolSpec,
    /// The innovation distribution.
    pub dist: DistSpec,
}

impl GarchSpec {
    /// Checks the lag structure itself (independent of data length).
    ///
    /// # Errors
    ///
    /// [`GarchError::InvalidSpec`] if `p == 0`, or if a GJR specification
    /// has `o == 0`.
    pub fn validate(&self) -> Result<(), GarchError> {
        let (p, o, _q) = self.vol.lags();
        if p == 0 {
            return Err(GarchError::InvalidSpec {
                what: "p >= 1 (at least one shock lag) is required",
            });
        }
        if matches!(self.vol, VolSpec::Gjr { .. }) && o == 0 {
            return Err(GarchError::InvalidSpec {
                what: "GJR requires o >= 1 (use Garch for o = 0)",
            });
        }
        Ok(())
    }

    /// Number of mean-equation parameters (0 or 1).
    pub fn n_mean_params(&self) -> usize {
        match self.mean {
            MeanSpec::Zero => 0,
            MeanSpec::Constant => 1,
        }
    }

    /// Total number of parameters: mean + variance + distribution.
    pub fn n_params(&self) -> usize {
        self.n_mean_params() + self.vol.n_params() + self.dist.n_params()
    }

    /// Parameter names in vector order, matching `arch`'s labels:
    /// `mu`, `omega`, `alpha[1]`, ..., `gamma[1]`, ..., `beta[1]`, ...,
    /// `nu`.
    pub fn param_names(&self) -> Vec<String> {
        let mut names = Vec::with_capacity(self.n_params());
        if matches!(self.mean, MeanSpec::Constant) {
            names.push("mu".to_owned());
        }
        names.push("omega".to_owned());
        let (p, o, q) = self.vol.lags();
        for i in 1..=p {
            names.push(format!("alpha[{i}]"));
        }
        for i in 1..=o {
            names.push(format!("gamma[{i}]"));
        }
        for j in 1..=q {
            names.push(format!("beta[{j}]"));
        }
        if matches!(self.dist, DistSpec::StudentT) {
            names.push("nu".to_owned());
        }
        names
    }

    /// Splits a full parameter vector into
    /// `(mean, omega, alphas, gammas, betas, dist)` slices/values.
    ///
    /// # Errors
    ///
    /// [`GarchError::DimensionMismatch`] if `params.len() != n_params()`.
    #[allow(clippy::type_complexity)]
    pub(crate) fn split_params<'a>(
        &self,
        params: &'a [f64],
    ) -> Result<(&'a [f64], f64, &'a [f64], &'a [f64], &'a [f64], &'a [f64]), GarchError> {
        if params.len() != self.n_params() {
            return Err(GarchError::DimensionMismatch {
                what: "parameter vector",
                expected: self.n_params(),
                actual: params.len(),
            });
        }
        let nm = self.n_mean_params();
        let (p, o, q) = self.vol.lags();
        let mean = &params[..nm];
        let omega = params[nm];
        let alphas = &params[nm + 1..nm + 1 + p];
        let gammas = &params[nm + 1 + p..nm + 1 + p + o];
        let betas = &params[nm + 1 + p + o..nm + 1 + p + o + q];
        let dist = &params[nm + 1 + p + o + q..];
        Ok((mean, omega, alphas, gammas, betas, dist))
    }

    /// The persistence of the variance recursion:
    ///
    /// * GARCH: `sum(alpha) + sum(beta)` (Bollerslev 1986);
    /// * GJR: `sum(alpha) + 0.5 sum(gamma) + sum(beta)` — the `0.5` is
    ///   `P(z < 0)` under the symmetric innovations this crate ships
    ///   (Glosten-Jagannathan-Runkle 1993);
    /// * EGARCH: `sum(beta)` (Nelson 1991, log-variance AR persistence).
    ///
    /// # Errors
    ///
    /// [`GarchError::DimensionMismatch`] if `params.len() != n_params()`.
    pub fn persistence(&self, params: &[f64]) -> Result<f64, GarchError> {
        let (_, _, alphas, gammas, betas, _) = self.split_params(params)?;
        let a: f64 = alphas.iter().sum();
        let g: f64 = gammas.iter().sum();
        let b: f64 = betas.iter().sum();
        Ok(match self.vol {
            VolSpec::Garch { .. } => a + b,
            VolSpec::Gjr { .. } => a + 0.5 * g + b,
            VolSpec::Egarch { .. } => b,
        })
    }

    /// Checks a full parameter vector for admissibility. The constraints
    /// mirror `arch`'s:
    ///
    /// * all parameters finite; `mu` unrestricted;
    /// * GARCH/GJR: `omega > 0`, `alpha_i >= 0`, `beta_j >= 0`,
    ///   `alpha_i + gamma_i >= 0` (with `alpha_i = 0` past lag `p`), and
    ///   persistence strictly below one;
    /// * EGARCH: `omega`, `alpha`, `gamma` unrestricted;
    ///   `|sum(beta)| < 1`;
    /// * Student-t: `nu > 2` (unit variance must exist).
    ///
    /// # Errors
    ///
    /// [`GarchError::DimensionMismatch`] on a wrong-length vector;
    /// [`GarchError::NonFinite`] on NaN/infinity;
    /// [`GarchError::InvalidParameter`] on any violated constraint
    /// (persistence at or above one included — explosive recursions are
    /// rejected, not evaluated).
    pub fn validate_params(&self, params: &[f64]) -> Result<(), GarchError> {
        let (_, omega, alphas, gammas, betas, dist) = self.split_params(params)?;
        if params.iter().any(|v| !v.is_finite()) {
            return Err(GarchError::NonFinite {
                what: "parameter vector",
            });
        }
        match self.vol {
            VolSpec::Garch { .. } | VolSpec::Gjr { .. } => {
                if omega <= 0.0 {
                    return Err(GarchError::InvalidParameter {
                        name: "omega",
                        value: omega,
                        requirement: "omega > 0",
                    });
                }
                for &a in alphas {
                    if a < 0.0 {
                        return Err(GarchError::InvalidParameter {
                            name: "alpha",
                            value: a,
                            requirement: "alpha_i >= 0",
                        });
                    }
                }
                for &b in betas {
                    if b < 0.0 {
                        return Err(GarchError::InvalidParameter {
                            name: "beta",
                            value: b,
                            requirement: "beta_j >= 0",
                        });
                    }
                }
                for (i, &g) in gammas.iter().enumerate() {
                    let a = alphas.get(i).copied().unwrap_or(0.0);
                    if a + g < 0.0 {
                        return Err(GarchError::InvalidParameter {
                            name: "gamma",
                            value: g,
                            requirement: "alpha_i + gamma_i >= 0",
                        });
                    }
                }
            }
            VolSpec::Egarch { .. } => {}
        }
        let pers = self.persistence(params)?;
        let (name, bad) = match self.vol {
            VolSpec::Egarch { .. } => ("sum(beta)", pers.abs() >= 1.0),
            _ => ("persistence", pers >= 1.0),
        };
        if bad {
            return Err(GarchError::InvalidParameter {
                name,
                value: pers,
                requirement: "strictly inside the unit interval / circle",
            });
        }
        if matches!(self.dist, DistSpec::StudentT) {
            let nu = dist[0];
            if nu <= 2.0 {
                return Err(GarchError::InvalidParameter {
                    name: "nu",
                    value: nu,
                    requirement: "nu > 2 (unit variance must exist)",
                });
            }
        }
        Ok(())
    }
}
