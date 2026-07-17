//! MCMC convergence diagnostics following Vehtari, Gelman, Simpson,
//! Carpenter & Bürkner (2021, Bayesian Analysis): rank-normalized split
//! R-hat, bulk effective sample size, and tail effective sample size —
//! algorithm-for-algorithm the ArviZ / Stan implementations, so values
//! agree numerically with `arviz.rhat(method="rank")`,
//! `arviz.ess(method="bulk"/"tail"/"mean")` on shared draws.
//!
//! Input convention: one scalar quantity, `chains` as an
//! `n_chains x n_draws` matrix (row = chain). All functions:
//!
//! 1. **split** each chain into halves (first `n/2` and last `n/2` draws),
//!    so within-chain trends register as between-chain disagreement;
//! 2. where indicated, **rank-normalize**: pooled average ranks `r` over
//!    all split chains mapped through the normal quantile
//!    `z = Phi^-1((r - 3/8) / (S + 1/4))` (Blom 1958 offsets), making the
//!    diagnostics defined for heavy-tailed quantities;
//! 3. compute the classic statistic on the transformed chains.
//!
//! The ESS autocorrelation sum uses Geyer's (1992) initial positive and
//! initial monotone sequence estimators on the *biased* (divide by `n`)
//! autocovariances — exactly the estimator ArviZ evaluates via FFT; this
//! implementation computes the same sums directly. ESS is capped through
//! the `tau >= 1 / log10(S)` floor, so `ess <= S log10(S)` with
//! `S = n_chains x n_draws` (ArviZ's anti-superefficiency cap; a value
//! above `S` is possible for antithetic chains and is not an error).

use tsecon_linalg::faer::MatRef;
use tsecon_stats::special::inv_norm_cdf;

use crate::error::BayesError;

/// Rank-normalized split R-hat (Vehtari et al. 2021, eq. 4 and §4.1):
/// the maximum of the bulk statistic (split chains, rank-normalized) and
/// the tail statistic (same, after folding around the pooled median,
/// `|x - median|`, which detects variance rather than location
/// disagreement). Matches `arviz.rhat(..., method="rank")`.
///
/// Values near 1 indicate convergence; the paper's threshold is
/// `R-hat < 1.01`.
///
/// # Errors
///
/// * [`BayesError::InvalidArgument`] with fewer than 2 chains or 4 draws
///   per chain;
/// * [`BayesError::NonFinite`] on NaN/infinite draws.
pub fn rhat_rank(chains: MatRef<'_, f64>) -> Result<f64, BayesError> {
    let chains = validate(chains, 2)?;
    let split = split_chains(&chains);
    let bulk = rhat_base(&rank_normalize(&split)?)?;
    let med = pooled_median(&split);
    let folded: Vec<Vec<f64>> = split
        .iter()
        .map(|c| c.iter().map(|x| (x - med).abs()).collect())
        .collect();
    let tail = rhat_base(&rank_normalize(&folded)?)?;
    Ok(bulk.max(tail))
}

/// Bulk effective sample size (Vehtari et al. 2021, §3.2): Geyer-sum ESS
/// of the rank-normalized split chains. Matches
/// `arviz.ess(..., method="bulk")`. The paper recommends requiring
/// `ess_bulk > 100 x n_chains`.
///
/// # Errors
///
/// * [`BayesError::InvalidArgument`] with no chains or fewer than 4 draws
///   per chain;
/// * [`BayesError::NonFinite`] on NaN/infinite draws.
pub fn ess_bulk(chains: MatRef<'_, f64>) -> Result<f64, BayesError> {
    let chains = validate(chains, 1)?;
    let split = split_chains(&chains);
    ess_core(&rank_normalize(&split)?)
}

/// Tail effective sample size (Vehtari et al. 2021, §4.3): the minimum
/// over a lower and an upper tail quantile of the ESS of the indicator
/// chains `1{x <= q_alpha}` (pooled type-7 quantiles), diagnosing how
/// reliably the tails are explored.
///
/// Default quantile pair: `(0.11, 0.89)` — the tails of the 89% central
/// interval, matching `arviz.ess(..., method="tail")` at its default
/// `rcParams["stats.ci_prob"] = 0.89` (ArviZ >= 1.0). The paper's own
/// illustration uses `(0.05, 0.95)`; pass that to [`ess_tail_prob`] if
/// legacy-ArviZ comparability is wanted.
///
/// # Errors
///
/// As for [`ess_bulk`].
pub fn ess_tail(chains: MatRef<'_, f64>) -> Result<f64, BayesError> {
    ess_tail_prob(chains, (0.11, 0.89))
}

/// [`ess_tail`] with an explicit `(lower, upper)` tail-quantile pair,
/// each strictly inside `(0, 1)`.
///
/// # Errors
///
/// As for [`ess_bulk`], plus [`BayesError::InvalidArgument`] for a
/// non-increasing or out-of-range probability pair.
pub fn ess_tail_prob(chains: MatRef<'_, f64>, prob: (f64, f64)) -> Result<f64, BayesError> {
    let (lo, hi) = prob;
    if !(lo > 0.0 && lo < 1.0 && hi > 0.0 && hi < 1.0 && lo < hi) {
        return Err(BayesError::InvalidArgument {
            what: "tail probabilities must satisfy 0 < lower < upper < 1",
        });
    }
    let chains = validate(chains, 1)?;
    let mut result = f64::INFINITY;
    for prob in [lo, hi] {
        let q = pooled_quantile_type7(&chains, prob);
        let indicator: Vec<Vec<f64>> = chains
            .iter()
            .map(|c| c.iter().map(|&x| if x <= q { 1.0 } else { 0.0 }).collect())
            .collect();
        let e = ess_core(&split_chains(&indicator))?;
        if e < result {
            result = e;
        }
    }
    Ok(result)
}

/// Effective sample size for the posterior mean: the Geyer-sum ESS of the
/// split chains *without* rank normalization (matches
/// `arviz.ess(..., method="mean")`). This is the classical `n_eff`
/// entering the Monte Carlo standard error `sd / sqrt(ess_mean)`, used by
/// this crate's Geweke (2004) sampler tests.
///
/// # Errors
///
/// As for [`ess_bulk`].
pub fn ess_mean(chains: MatRef<'_, f64>) -> Result<f64, BayesError> {
    let chains = validate(chains, 1)?;
    ess_core(&split_chains(&chains))
}

// ---------------------------------------------------------------------------
// Internals
// ---------------------------------------------------------------------------

/// Validates shape/finiteness and converts to owned rows.
fn validate(chains: MatRef<'_, f64>, min_chains: usize) -> Result<Vec<Vec<f64>>, BayesError> {
    if chains.nrows() < min_chains {
        return Err(BayesError::InvalidArgument {
            what: "too few chains (rank-normalized R-hat needs >= 2, ESS >= 1)",
        });
    }
    if chains.ncols() < 4 {
        return Err(BayesError::InvalidArgument {
            what: "convergence diagnostics need at least 4 draws per chain",
        });
    }
    let mut out = Vec::with_capacity(chains.nrows());
    for i in 0..chains.nrows() {
        let mut row = Vec::with_capacity(chains.ncols());
        for j in 0..chains.ncols() {
            let v = chains[(i, j)];
            if !v.is_finite() {
                return Err(BayesError::NonFinite { what: "chains" });
            }
            row.push(v);
        }
        out.push(row);
    }
    Ok(out)
}

/// Splits every chain into its first and last `n/2` draws (the middle draw
/// of an odd-length chain is dropped), stacking first halves then second
/// halves.
fn split_chains(chains: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = chains[0].len();
    let half = n / 2;
    let mut out = Vec::with_capacity(2 * chains.len());
    for c in chains {
        out.push(c[..half].to_vec());
    }
    for c in chains {
        out.push(c[n - half..].to_vec());
    }
    out
}

/// Pooled average ranks (ties get the mean rank), mapped through the
/// normal quantile with Blom (1958) offsets:
/// `z = Phi^-1((r - 3/8) / (S + 1/4))`.
fn rank_normalize(chains: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, BayesError> {
    let n = chains[0].len();
    let total = chains.len() * n;
    // (value, flat index), sorted by value; f64::total_cmp is a total
    // order (inputs are validated finite).
    let mut order: Vec<(f64, usize)> = Vec::with_capacity(total);
    for (c, chain) in chains.iter().enumerate() {
        for (i, &v) in chain.iter().enumerate() {
            order.push((v, c * n + i));
        }
    }
    order.sort_by(|a, b| a.0.total_cmp(&b.0));

    // Average ranks over runs of tied values (1-based ranks).
    let mut ranks = vec![0.0; total];
    let mut start = 0;
    while start < total {
        let mut end = start + 1;
        while end < total && order[end].0 == order[start].0 {
            end += 1;
        }
        let avg = (start + 1 + end) as f64 / 2.0; // mean of ranks start+1..=end
        for item in &order[start..end] {
            ranks[item.1] = avg;
        }
        start = end;
    }

    let denom = total as f64 + 0.25;
    let mut out = Vec::with_capacity(chains.len());
    for c in 0..chains.len() {
        let mut row = Vec::with_capacity(n);
        for i in 0..n {
            let u = (ranks[c * n + i] - 0.375) / denom;
            row.push(inv_norm_cdf(u)?);
        }
        out.push(row);
    }
    Ok(out)
}

/// Classic split R-hat on already-transformed chains: with `W` the mean
/// within-chain variance and `B` the between-chain variance,
/// `R-hat = sqrt(((n-1)/n + B/(n W)))` (Gelman & Rubin 1992 as
/// implemented by ArviZ/Stan).
fn rhat_base(chains: &[Vec<f64>]) -> Result<f64, BayesError> {
    let n = chains[0].len() as f64;
    let means: Vec<f64> = chains.iter().map(|c| mean(c)).collect();
    let within = mean(
        &chains
            .iter()
            .map(|c| variance_ddof1(c))
            .collect::<Vec<f64>>(),
    );
    let between = n * variance_ddof1(&means);
    if !within.is_finite() || within <= 0.0 {
        return Err(BayesError::InvalidArgument {
            what: "R-hat undefined: chains have zero within-chain variance",
        });
    }
    Ok(((between / within + n - 1.0) / n).sqrt())
}

/// The Stan/ArviZ effective sample size of already-transformed split
/// chains: biased per-chain autocovariances combined across chains, the
/// Geyer initial positive sequence, the Geyer initial monotone sequence,
/// and the `tau >= 1/log10(S)` floor.
fn ess_core(chains: &[Vec<f64>]) -> Result<f64, BayesError> {
    let m = chains.len();
    let n = chains[0].len();
    let size = (m * n) as f64;

    // Constant chains: ArviZ returns the total draw count.
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for c in chains {
        for &v in c {
            lo = lo.min(v);
            hi = hi.max(v);
        }
    }
    if hi - lo == 0.0 {
        return Ok(size);
    }

    // Biased (divide by n) autocovariances per chain, all lags.
    let acov: Vec<Vec<f64>> = chains.iter().map(|c| autocov_biased(c)).collect();
    let chain_means: Vec<f64> = chains.iter().map(|c| mean(c)).collect();
    let mean_acov = |lag: usize| -> f64 {
        let mut s = 0.0;
        for a in &acov {
            s += a[lag];
        }
        s / m as f64
    };
    let mean_acov0 = mean_acov(0);
    // W (unbiased within variance) and var_plus = W(n-1)/n + B/n.
    let mean_var = mean_acov0 * n as f64 / (n as f64 - 1.0);
    let mut var_plus = mean_acov0;
    if m > 1 {
        var_plus += variance_ddof1(&chain_means);
    }
    if !var_plus.is_finite() || var_plus <= 0.0 {
        return Err(BayesError::InvalidArgument {
            what: "effective sample size undefined: chains have zero variance",
        });
    }

    // Geyer initial positive sequence on rho_t = 1 - (W - acov_t)/var_plus.
    let nd = n as i64;
    let mut rho = vec![0.0f64; n];
    let mut rho_even = 1.0f64;
    rho[0] = rho_even;
    let mut rho_odd = 1.0 - (mean_var - mean_acov(1)) / var_plus;
    rho[1] = rho_odd;
    let mut t: i64 = 1;
    while t < nd - 3 && (rho_even + rho_odd) > 0.0 {
        rho_even = 1.0 - (mean_var - mean_acov((t + 1) as usize)) / var_plus;
        rho_odd = 1.0 - (mean_var - mean_acov((t + 2) as usize)) / var_plus;
        if rho_even + rho_odd >= 0.0 {
            rho[(t + 1) as usize] = rho_even;
            rho[(t + 2) as usize] = rho_odd;
        }
        t += 2;
    }
    let max_t = t - 2;
    // Improve the estimate: keep a trailing positive even term.
    if rho_even > 0.0 {
        rho[(max_t + 1) as usize] = rho_even;
    }
    // Geyer initial monotone sequence: enforce nonincreasing pair sums.
    let mut t: i64 = 1;
    while t <= max_t - 2 {
        let (i, j) = ((t + 1) as usize, (t + 2) as usize);
        let (a, b) = ((t - 1) as usize, t as usize);
        if rho[i] + rho[j] > rho[a] + rho[b] {
            let avg = (rho[a] + rho[b]) / 2.0;
            rho[i] = avg;
            rho[j] = avg;
        }
        t += 2;
    }

    let mut s = 0.0;
    if max_t >= 0 {
        for r in rho.iter().take(max_t as usize + 1) {
            s += r;
        }
    }
    let mut tau = -1.0 + 2.0 * s + rho[(max_t + 1) as usize];
    tau = tau.max(1.0 / size.log10());
    if !tau.is_finite() {
        return Err(BayesError::NonFinite {
            what: "integrated autocorrelation time",
        });
    }
    Ok(size / tau)
}

/// Biased autocovariances `(1/n) sum (x_i - xbar)(x_{i+t} - xbar)` for all
/// lags `t = 0..n-1` — the estimator ArviZ computes via FFT (the biased
/// divide-by-`n` convention Geyer/Stan require for the initial-sequence
/// arguments to hold).
fn autocov_biased(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    let xbar = mean(x);
    let centered: Vec<f64> = x.iter().map(|v| v - xbar).collect();
    let mut out = Vec::with_capacity(n);
    for lag in 0..n {
        let mut s = 0.0;
        for i in 0..(n - lag) {
            s += centered[i] * centered[i + lag];
        }
        out.push(s / n as f64);
    }
    out
}

fn mean(x: &[f64]) -> f64 {
    x.iter().sum::<f64>() / x.len() as f64
}

/// Sample variance with `ddof = 1`.
fn variance_ddof1(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    let m = mean(x);
    x.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / (n - 1.0)
}

/// Pooled median (average of the two middle order statistics for an even
/// count, matching `numpy.median`).
fn pooled_median(chains: &[Vec<f64>]) -> f64 {
    let mut all: Vec<f64> = chains.iter().flatten().copied().collect();
    all.sort_by(f64::total_cmp);
    let s = all.len();
    if s % 2 == 1 {
        all[s / 2]
    } else {
        0.5 * (all[s / 2 - 1] + all[s / 2])
    }
}

/// Pooled type-7 (R default / NumPy `linear`) quantile: with sorted
/// `a_0..a_{S-1}` and `h = (S - 1) q`, interpolate
/// `a_{floor(h)} + (h - floor(h)) (a_{floor(h)+1} - a_{floor(h)})`.
fn pooled_quantile_type7(chains: &[Vec<f64>], q: f64) -> f64 {
    let mut all: Vec<f64> = chains.iter().flatten().copied().collect();
    all.sort_by(f64::total_cmp);
    let s = all.len();
    let h = (s as f64 - 1.0) * q;
    let lo = h.floor() as usize;
    let frac = h - lo as f64;
    if lo + 1 < s {
        all[lo] + frac * (all[lo + 1] - all[lo])
    } else {
        all[s - 1]
    }
}
