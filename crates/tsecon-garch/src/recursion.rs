//! Backcast initialization and the conditional-variance recursions,
//! matching Kevin Sheppard's `arch` package term for term (the fixture
//! `garch.json` pins fixed-parameter log-likelihoods to 1e-8 relative,
//! and machine precision is achieved in practice).

/// Exponentially weighted backcast of the presample variance
/// (`arch`'s `VolatilityProcess.backcast`):
///
/// ```text
/// bc = sum_{i=0}^{tau-1} w_i eps_i^2,   w_i propto 0.94^i,  tau = min(75, T)
/// ```
///
/// — the RiskMetrics decay (J.P. Morgan 1996) over at most the first 75
/// observations, weights normalized to sum to one. `arch` computes this
/// once from the residuals at the mean model's *starting values* (zero for
/// a zero-mean model, the sample mean for a constant-mean model) and holds
/// it fixed through estimation; this crate does the same.
pub(crate) fn backcast(resids: &[f64]) -> f64 {
    let tau = resids.len().min(75);
    let mut wsum = 0.0;
    let mut acc = 0.0;
    let mut w = 1.0;
    for &r in &resids[..tau] {
        acc += w * r * r;
        wsum += w;
        w *= 0.94;
    }
    if wsum > 0.0 {
        acc / wsum
    } else {
        f64::NAN
    }
}

/// GARCH/GJR variance recursion (`arch`'s `garch_recursion`):
///
/// ```text
/// sigma2_t = omega + sum_i alpha_i eps_{t-i}^2
///                  + sum_i gamma_i eps_{t-i}^2 1[eps_{t-i} < 0]
///                  + sum_j beta_j sigma2_{t-j}
/// ```
///
/// Presample values follow `arch` exactly: `eps_{t}^2 -> backcast` for
/// `t < 0`, the asymmetric term `-> 0.5 * backcast` (its expectation under
/// symmetric innovations), and `sigma2_t -> backcast`.
///
/// Writes into `sigma2` (`sigma2.len() == resids.len()`); the caller
/// checks positivity/finiteness (guaranteed for admissible parameters and
/// finite data, but the likelihood re-checks to stay panic-free).
pub(crate) fn garch_recursion(
    omega: f64,
    alphas: &[f64],
    gammas: &[f64],
    betas: &[f64],
    resids: &[f64],
    backcast: f64,
    sigma2: &mut [f64],
) {
    let nobs = resids.len();
    for t in 0..nobs {
        let mut v = omega;
        for (i, &a) in alphas.iter().enumerate() {
            v += a * match t.checked_sub(i + 1) {
                Some(s) => resids[s] * resids[s],
                None => backcast,
            };
        }
        for (i, &g) in gammas.iter().enumerate() {
            v += g * match t.checked_sub(i + 1) {
                Some(s) => {
                    if resids[s] < 0.0 {
                        resids[s] * resids[s]
                    } else {
                        0.0
                    }
                }
                None => 0.5 * backcast,
            };
        }
        for (j, &b) in betas.iter().enumerate() {
            v += b * match t.checked_sub(j + 1) {
                Some(s) => sigma2[s],
                None => backcast,
            };
        }
        sigma2[t] = v;
    }
}

/// EGARCH log-variance recursion (`arch`'s `egarch_recursion`):
///
/// ```text
/// ln sigma2_t = omega + sum_i alpha_i (|z_{t-i}| - sqrt(2/pi))
///                     + sum_i gamma_i z_{t-i}
///                     + sum_j beta_j ln sigma2_{t-j},    z = eps / sigma
/// ```
///
/// Presample conventions per `arch`: the `alpha` and `gamma` terms are
/// *dropped* for `t - i < 0` (no presample standardized residuals), while
/// `ln sigma2_{t-j} -> ln_backcast` (the log of the weighted backcast
/// variance). `ln sigma2_t` is clamped at `ln(f64::MAX)` against overflow,
/// as in `arch`.
pub(crate) fn egarch_recursion(
    omega: f64,
    alphas: &[f64],
    gammas: &[f64],
    betas: &[f64],
    resids: &[f64],
    ln_backcast: f64,
    sigma2: &mut [f64],
) {
    let nobs = resids.len();
    let norm_const = (2.0 / core::f64::consts::PI).sqrt();
    let ln_max = f64::MAX.ln();
    let mut lns2 = vec![0.0_f64; nobs];
    let mut zstd = vec![0.0_f64; nobs];
    for t in 0..nobs {
        let mut v = omega;
        for (i, &a) in alphas.iter().enumerate() {
            if let Some(s) = t.checked_sub(i + 1) {
                v += a * (zstd[s].abs() - norm_const);
            }
        }
        for (i, &g) in gammas.iter().enumerate() {
            if let Some(s) = t.checked_sub(i + 1) {
                v += g * zstd[s];
            }
        }
        for (j, &b) in betas.iter().enumerate() {
            v += b * match t.checked_sub(j + 1) {
                Some(s) => lns2[s],
                None => ln_backcast,
            };
        }
        if v > ln_max {
            v = ln_max;
        }
        lns2[t] = v;
        sigma2[t] = v.exp();
        zstd[t] = resids[t] / sigma2[t].sqrt();
    }
}
