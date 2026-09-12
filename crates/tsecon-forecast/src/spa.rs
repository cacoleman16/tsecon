//! White's (2000) Reality Check and Hansen's (2005) test for Superior
//! Predictive Ability (SPA), plus the Romano-Wolf (2005) StepM procedure
//! built on the same bootstrap.
//!
//! # The question
//!
//! A benchmark forecast and `m` competing models produce losses over the
//! same `n` evaluation periods. Pairwise Diebold-Mariano tests answer "does
//! model `k` beat the benchmark?" one model at a time; with many models the
//! best one wins some of those tests by search alone. The Reality Check and
//! SPA ask the joint question — is the BEST model better than the benchmark
//! once the search over all `m` is accounted for — through the null
//! `H0: max_k mu_k <= 0`, where `d_{t,k} = L(benchmark)_t - L(model k)_t`
//! and `mu_k = E[d_{t,k}]` (a positive `mu_k` means model `k` incurs the
//! lower loss).
//!
//! # The statistics
//!
//! With `dbar_k = (1/n) sum_t d_{t,k}` and `omega_k^2` a consistent estimate
//! of the long-run variance `Var(sqrt(n) dbar_k)`,
//!
//! ```text
//! SPA (studentize = true):  T = sqrt(n) max_k dbar_k / omega_k     (Hansen 2005)
//! RC  (studentize = false): V = sqrt(n) max_k dbar_k               (White 2000)
//! ```
//!
//! Hansen writes his statistic as `max(T, 0)`; this crate reports the
//! untruncated maximum and computes every p-value as the fraction of
//! bootstrap replicates `T*_b > T` (the convention of `arch.bootstrap.SPA`).
//! The two agree whenever some model beats the benchmark on average
//! (`T > 0`), which is the only region where a rejection can occur.
//!
//! # The bootstrap and the three re-centrings
//!
//! The null distribution is simulated by a block bootstrap of the whole
//! `n x m` loss-differential panel — rows are resampled together, so the
//! cross-model dependence that shapes the maximum is preserved — and each
//! resampled mean `dbar*_{b,k}` is re-centred so that the simulated world
//! satisfies the null. Hansen (2005) defines three re-centrings `g_k`, which
//! give three p-values that bracket the truth:
//!
//! * **upper** (`g_k = dbar_k`): every model is re-centred to zero, the
//!   least favourable configuration `mu = 0`. This is White's (2000)
//!   original Reality Check p-value; it is conservative when poor models
//!   pad the comparison.
//! * **consistent** (`g_k = dbar_k * 1{dbar_k >= -sqrt(omega_k^2/n * 2 log log n)}`):
//!   models significantly WORSE than the benchmark by Hansen's
//!   `sqrt(2 log log n)` law-of-the-iterated-logarithm threshold are not
//!   re-centred — they keep their negative mean and stop inflating the
//!   simulated maximum. This is the recommended p-value.
//! * **lower** (`g_k = max(dbar_k, 0)`): no model with a negative sample
//!   mean is re-centred — the liberal bound.
//!
//! Replicate `b` under re-centring `g` is
//! `T*_b = sqrt(n) max_k (dbar*_{b,k} - g_k) / omega_k` (divided by
//! `omega_k` only under `studentize`); the p-value is the fraction of
//! replicates strictly above the observed statistic, and the critical values
//! at 90 / 95 / 99% are the corresponding percentiles (NumPy's linear
//! interpolation) of the replicate statistics.
//!
//! # The variance `omega_k^2`
//!
//! By default (`nested = false`) the long-run variance uses the kernel
//! Hansen (2005, eq. 9) derives for the stationary bootstrap with restart
//! probability `q = 1 / block_size`; with `e_{t,k} = d_{t,k} - dbar_k`,
//!
//! ```text
//! omega_k^2 = gamma_0 + 2 sum_{i=1}^{n-1} kappa(n, i) gamma_i,
//! gamma_i   = (1/n) sum_{t=1}^{n-i} e_{t,k} e_{t+i,k},
//! kappa     = (1 - i/n) (1 - q)^i + (i/n) (1 - q)^(n - i).
//! ```
//!
//! It is applied for every resampling scheme with the same `q`, exactly as
//! `arch` does. With `nested = true` the variance is instead the bootstrap
//! variance of the resampled mean of the demeaned panel over the SAME
//! resamples the test uses, `omega_k^2 = n Var_b( mean_t e_{idx_b(t),k} )`
//! (population variance over replications) — the choice `arch` makes when
//! seeded with an integer.
//!
//! # Resampling conventions
//!
//! `block_size` is the (expected) block length. `None` selects the
//! Politis-White (2004) / Patton-Politis-White (2009) optimal length of each
//! loss-differential column via [`tsecon_bootstrap::optimal_block_length`]
//! (the stationary-bootstrap value for [`ResampleScheme::Stationary`], the
//! circular value otherwise), averaged over the columns, rounded to the
//! nearest integer and clamped to `1..=n-1`. The three schemes are the
//! library's [`tsecon_bootstrap::BlockScheme`]s, and their conventions —
//! geometric block lengths with restart probability `1/block_size` and
//! wrap-around for the stationary bootstrap, fixed blocks with wrap-around
//! for the circular bootstrap, fixed blocks from starts in
//! `0..=n-block_size` for the moving block, the last block truncated so the
//! resample has length `n` — are the same as `arch`'s, so the two libraries
//! differ only in the random-number generator. Replication `b` draws its
//! indices from substream `b` of `SeedSequence(seed)` (the library's
//! [`tsecon_bootstrap::par_replicate`] contract), and every reduction runs in
//! replication order after the parallel resampling, so the result is
//! bit-identical at any rayon thread count.
//!
//! # Validation and the `arch` finding
//!
//! `arch 8.0` is the golden. Its `studentize` flag is INERT: the source
//! never divides by the variance it estimates, and the p-values and critical
//! values are identical with the flag on and off (measured in the fixture
//! generator). `arch.bootstrap.SPA` therefore computes the un-studentized
//! Reality Check statistic with Hansen's three re-centrings, and
//! `arch.bootstrap.RealityCheck` is literally `class RealityCheck(SPA):
//! pass`. [`spa_test_with_indices`] reproduces `arch` bit-for-bit under
//! `studentize = false` when fed `arch`'s own resample indices (means,
//! variances, replicate statistics, p-values), and the studentized path is
//! pinned against a NumPy transcription of Hansen's formulas on the same
//! indices. `fixtures/generate_spa_fixtures.py` states the honest grade of
//! each leg.
//!
//! # StepM
//!
//! Romano & Wolf (2005): after the joint test rejects, identify WHICH
//! models beat the benchmark while controlling the family-wise error rate
//! at `size`. Step 1 declares superior every model whose statistic exceeds
//! the `1 - size` quantile of the bootstrap maximum over ALL models (under
//! the consistent re-centring); each later step recomputes that quantile
//! over the models not yet declared superior and adds those now exceeding
//! it, until a step adds nothing. This is `arch.bootstrap.StepM` (with its
//! inert `studentize`, as above), except that `arch` raises when every model
//! is eventually declared superior over two or more steps; here the loop
//! stops with all `m` models in the superior set.
//!
//! # References
//!
//! White, H. (2000). "A Reality Check for Data Snooping." *Econometrica*
//! 68(5), 1097-1126. Hansen, P. R. (2005). "A Test for Superior Predictive
//! Ability." *Journal of Business & Economic Statistics* 23(4), 365-380.
//! Romano, J. P. & Wolf, M. (2005). "Stepwise Multiple Testing as Formalized
//! Data Snooping." *Econometrica* 73(4), 1237-1282. Politis, D. N. & Romano,
//! J. P. (1994). "The Stationary Bootstrap." *JASA* 89, 1303-1313. Politis &
//! White (2004) and Patton, Politis & White (2009) for the block length.

use rayon::prelude::*;
use tsecon_bootstrap::{indices, optimal_block_length, BlockScheme, BootstrapError};
use tsecon_rng::{SeedSequence, Stream};

use crate::error::ForecastError;

/// Block-bootstrap flavour used to resample the loss panel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResampleScheme {
    /// Politis-Romano (1994) stationary bootstrap: geometric block lengths
    /// with mean `block_size` (restart probability `1 / block_size`),
    /// wrapping around the end of the sample.
    Stationary,
    /// Politis-Romano (1992) circular block bootstrap: fixed blocks of
    /// `block_size` consecutive rows, wrapping around.
    CircularBlock,
    /// Künsch (1989) moving-block bootstrap: fixed blocks of `block_size`
    /// consecutive rows starting in `0..=n - block_size`, no wrap-around.
    MovingBlock,
}

impl ResampleScheme {
    /// The scheme's name as the Python surface spells it.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            ResampleScheme::Stationary => "stationary",
            ResampleScheme::CircularBlock => "circular",
            ResampleScheme::MovingBlock => "moving_block",
        }
    }

    pub(crate) fn block_scheme(self, block_size: usize) -> BlockScheme {
        match self {
            ResampleScheme::Stationary => BlockScheme::Stationary {
                p: 1.0 / block_size as f64,
            },
            ResampleScheme::CircularBlock => BlockScheme::CircularBlock {
                block_length: block_size,
            },
            ResampleScheme::MovingBlock => BlockScheme::MovingBlock {
                block_length: block_size,
            },
        }
    }
}

/// Options of [`spa_test`].
#[derive(Debug, Clone, PartialEq)]
pub struct SpaOptions {
    /// (Expected) block length; `None` selects the Politis-White optimal
    /// length (see the [module docs](self)).
    pub block_size: Option<usize>,
    /// Number of bootstrap replications (`>= 1`).
    pub reps: usize,
    /// The resampling scheme.
    pub scheme: ResampleScheme,
    /// Divide each loss differential by its long-run standard deviation
    /// (Hansen's SPA); `false` gives White's Reality Check.
    pub studentize: bool,
    /// Estimate `omega_k^2` by a nested bootstrap over the same resamples
    /// instead of Hansen's kernel.
    pub nested: bool,
    /// Seed of the Philox substream hierarchy.
    pub seed: u64,
}

impl Default for SpaOptions {
    fn default() -> Self {
        SpaOptions {
            block_size: None,
            reps: 1000,
            scheme: ResampleScheme::Stationary,
            studentize: true,
            nested: false,
            seed: 0,
        }
    }
}

/// Result of [`spa_test`] / [`spa_test_with_indices`].
#[derive(Debug, Clone, PartialEq)]
pub struct SpaResult {
    /// Number of evaluation periods.
    pub n: usize,
    /// Number of competing models.
    pub m: usize,
    /// The block length actually used.
    pub block_size: usize,
    /// Whether `block_size` came from the automatic Politis-White rule.
    pub block_size_auto: bool,
    /// The resampling scheme.
    pub scheme: ResampleScheme,
    /// Number of bootstrap replications.
    pub reps: usize,
    /// Whether the statistic was studentized.
    pub studentize: bool,
    /// Whether the variances came from the nested bootstrap.
    pub nested: bool,
    /// `dbar_k = mean_t (benchmark_t - model_{k,t})`; positive favours
    /// model `k`.
    pub mean_loss_diff: Vec<f64>,
    /// `omega_k^2`, the long-run variance of `sqrt(n) dbar_k`.
    pub loss_diff_var: Vec<f64>,
    /// Whether model `k` was re-centred under the consistent p-value
    /// (`false` for models significantly worse than the benchmark).
    pub recentered: Vec<bool>,
    /// The observed statistic, `sqrt(n)`-scaled (see the module docs).
    pub statistic: f64,
    /// Index of the model attaining the maximum (first on ties).
    pub best_model: usize,
    /// Bootstrap p-value under the lower re-centring.
    pub p_value_lower: f64,
    /// Bootstrap p-value under the consistent re-centring (recommended).
    pub p_value_consistent: f64,
    /// Bootstrap p-value under the upper re-centring (White's RC).
    pub p_value_upper: f64,
    /// The confidence levels of the critical values: `[0.90, 0.95, 0.99]`.
    pub crit_levels: [f64; 3],
    /// Critical values at `crit_levels` under the lower re-centring.
    pub crit_lower: [f64; 3],
    /// Critical values at `crit_levels` under the consistent re-centring.
    pub crit_consistent: [f64; 3],
    /// Critical values at `crit_levels` under the upper re-centring.
    pub crit_upper: [f64; 3],
    /// The `reps` replicate statistics under the lower re-centring
    /// (`sqrt(n)`-scaled like `statistic`).
    pub boot_lower: Vec<f64>,
    /// The replicate statistics under the consistent re-centring.
    pub boot_consistent: Vec<f64>,
    /// The replicate statistics under the upper re-centring.
    pub boot_upper: Vec<f64>,
}

/// Options of [`stepm_test`].
#[derive(Debug, Clone, PartialEq)]
pub struct StepmOptions {
    /// Family-wise error rate controlled by the procedure, in `(0, 1)`.
    pub size: f64,
    /// The SPA bootstrap settings.
    pub spa: SpaOptions,
}

impl Default for StepmOptions {
    fn default() -> Self {
        StepmOptions {
            size: 0.05,
            spa: SpaOptions::default(),
        }
    }
}

/// Result of [`stepm_test`] / [`stepm_test_with_indices`].
#[derive(Debug, Clone, PartialEq)]
pub struct StepmResult {
    /// The full-set SPA test the procedure starts from.
    pub spa: SpaResult,
    /// The family-wise error rate.
    pub size: f64,
    /// Indices of the models declared superior to the benchmark, ascending.
    pub superior_models: Vec<usize>,
    /// The models declared superior at each step, in index order; the last
    /// entry is empty when the procedure stopped because a step added
    /// nothing.
    pub steps: Vec<Vec<usize>>,
    /// The `1 - size` bootstrap quantile each step compared against
    /// (`sqrt(n)`-scaled like [`SpaResult::statistic`]).
    pub step_crit_values: Vec<f64>,
}

/// Confidence levels of the reported critical values.
pub(crate) const CRIT_LEVELS: [f64; 3] = [0.90, 0.95, 0.99];

/// Substreams are spawned in chunks of this size so that no vector of
/// `reps` streams is ever allocated; SeedSequence spawning is order-stable
/// across calls, so the chunking does not change which substream a
/// replication receives.
const SPAWN_CHUNK: usize = 1024;

/// Where a bootstrap replication's row indices come from.
pub(crate) enum IndexSource<'a> {
    /// Replication `b` uses substream `b` of `SeedSequence(seed)`.
    Seeded {
        scheme: BlockScheme,
        seed: u64,
        reps: usize,
    },
    /// Replication `b` uses the caller's `resamples[b]`.
    Explicit(&'a [Vec<usize>]),
}

impl IndexSource<'_> {
    pub(crate) fn reps(&self) -> usize {
        match self {
            IndexSource::Seeded { reps, .. } => *reps,
            IndexSource::Explicit(r) => r.len(),
        }
    }
}

/// Fill a `reps x width` row-major buffer with `fill(indices_b, row_b)`,
/// budgeting the buffer with `try_reserve` and drawing the indices as the
/// source dictates. Rows are written by replication index, so the buffer
/// is bit-identical at any thread count.
pub(crate) fn bootstrap_rows<F>(
    what: &'static str,
    source: &IndexSource<'_>,
    n: usize,
    width: usize,
    fill: F,
) -> Result<Vec<f64>, ForecastError>
where
    F: Fn(&[usize], &mut [f64]) + Sync,
{
    let reps = source.reps();
    let total = reps
        .checked_mul(width)
        .ok_or(ForecastError::AllocationRefused {
            what,
            elements: usize::MAX,
        })?;
    let mut data: Vec<f64> = Vec::new();
    data.try_reserve_exact(total)
        .map_err(|_| ForecastError::AllocationRefused {
            what,
            elements: total,
        })?;
    data.resize(total, 0.0);
    match source {
        IndexSource::Explicit(resamples) => {
            for (b, r) in resamples.iter().enumerate() {
                if r.len() != n {
                    return Err(ForecastError::InvalidResample {
                        rep: b,
                        detail: format!(
                            "it has {} indices but the panel has n = {n} rows",
                            r.len()
                        ),
                    });
                }
                if let Some((pos, &bad)) = r.iter().enumerate().find(|(_, &i)| i >= n) {
                    return Err(ForecastError::InvalidResample {
                        rep: b,
                        detail: format!("index {bad} at position {pos} is outside 0..{n}"),
                    });
                }
            }
            data.par_chunks_mut(width)
                .zip(resamples.par_iter())
                .for_each(|(row, idx)| fill(idx, row));
        }
        IndexSource::Seeded { scheme, seed, .. } => {
            let mut root = SeedSequence::new(u128::from(*seed));
            let mut done = 0usize;
            while done < reps {
                let take = SPAWN_CHUNK.min(reps - done);
                let seqs = root
                    .spawn(take)
                    .map_err(|e| ForecastError::Bootstrap(BootstrapError::from(e)))?;
                let slab = &mut data[done * width..(done + take) * width];
                slab.par_chunks_mut(width)
                    .zip(seqs.par_iter())
                    .try_for_each(|(row, seq)| {
                        let mut stream = Stream::from_seed_sequence(seq);
                        let idx = indices(*scheme, n, &mut stream)?;
                        fill(&idx, row);
                        Ok::<(), ForecastError>(())
                    })?;
                done += take;
            }
        }
    }
    Ok(data)
}

/// `out[k] = (1/n) sum_t (cols[k][idx[t]] - shift[k])`, accumulated
/// sequentially in `t` — NumPy's reduction order for `panel[idx].mean(0)`
/// on a C-ordered `n x m` panel, which the fixture generator asserts
/// bit-for-bit before storing any case.
pub(crate) fn resampled_means(
    cols: &[Vec<f64>],
    idx: &[usize],
    shift: Option<&[f64]>,
    out: &mut [f64],
) {
    let n = idx.len() as f64;
    for (k, (col, o)) in cols.iter().zip(out.iter_mut()).enumerate() {
        let s = shift.map_or(0.0, |s| s[k]);
        let mut acc = 0.0;
        for &t in idx {
            acc += col[t] - s;
        }
        *o = acc / n;
    }
}

/// Column mean, accumulated sequentially (NumPy's axis-0 order).
pub(crate) fn column_mean(col: &[f64]) -> f64 {
    let mut acc = 0.0;
    for &v in col {
        acc += v;
    }
    acc / col.len() as f64
}

/// NumPy's pairwise summation of a contiguous 1-D array
/// (`numpy/core/src/umath/loops_utils.h.src`, `pairwise_sum`): plain
/// accumulation below 8 elements, eight interleaved partial sums up to the
/// 128-element block size, recursive halving above it. Used wherever NumPy
/// reduces along a contiguous axis (a row mean over the models), so that
/// the fixture pins are exact rather than approximate.
pub(crate) fn numpy_pairwise_sum(a: &[f64]) -> f64 {
    let n = a.len();
    if n < 8 {
        let mut res = 0.0;
        for &v in a {
            res += v;
        }
        res
    } else if n <= 128 {
        let mut r = [a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]];
        let lim = n - (n % 8);
        let mut i = 8;
        while i < lim {
            for (acc, &v) in r.iter_mut().zip(&a[i..i + 8]) {
                *acc += v;
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for &v in &a[i..] {
            res += v;
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        numpy_pairwise_sum(&a[..n2]) + numpy_pairwise_sum(&a[n2..])
    }
}

/// `numpy.percentile(x, 100 q)` with the default linear interpolation
/// (Hyndman-Fan type 7): virtual index `(n - 1) q`, and NumPy's `_lerp`
/// branch at `gamma >= 0.5`. `sorted` must be ascending and
/// non-empty.
pub(crate) fn numpy_percentile(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    // numpy's `linear` method: virtual index (n - 1) q (its `_QuantileMethods`
    // entry; `_compute_virtual_index` serves the other methods).
    let virt = (n - 1) as f64 * q;
    if virt >= (n - 1) as f64 {
        return sorted[n - 1];
    }
    if virt < 0.0 {
        return sorted[0];
    }
    let prev = virt.floor();
    let lo = prev as usize;
    let hi = (lo + 1).min(n - 1);
    let gamma = virt - prev;
    let (a, b) = (sorted[lo], sorted[hi]);
    let diff = b - a;
    if gamma >= 0.5 {
        b - diff * (1.0 - gamma)
    } else {
        a + diff * gamma
    }
}

fn param_err(
    test: &'static str,
    what: &'static str,
    value: impl std::fmt::Display,
    requirement: impl Into<String>,
) -> ForecastError {
    ForecastError::InvalidMultipleComparisonParam {
        test,
        what,
        value: value.to_string(),
        requirement: requirement.into(),
    }
}

/// Every value of a loss column finite, naming the column in the error.
pub(crate) fn check_finite_column(
    test: &'static str,
    what: &'static str,
    col: &[f64],
    column: Option<usize>,
) -> Result<(), ForecastError> {
    if let Some((t, &v)) = col.iter().enumerate().find(|(_, v)| !v.is_finite()) {
        let value = match column {
            Some(k) => format!("{v} at period {t} of column {k}"),
            None => format!("{v} at period {t}"),
        };
        return Err(param_err(
            test,
            what,
            value,
            "every loss finite; NaN/inf are never skipped silently (that \
             would change the evaluation sample behind your back) — drop or \
             impute them first",
        ));
    }
    Ok(())
}

/// Resolve the block length: validate an explicit one, or average the
/// Politis-White optimal lengths of the panel's columns.
pub(crate) fn resolve_block_size(
    test: &'static str,
    requested: Option<usize>,
    cols: &[Vec<f64>],
    n: usize,
    scheme: ResampleScheme,
) -> Result<(usize, bool), ForecastError> {
    match requested {
        Some(b) => {
            if b == 0 || b >= n {
                return Err(param_err(
                    test,
                    "block_size",
                    b,
                    format!(
                        "1 <= block_size < n = {n}: a block as long as the \
                         sample makes every moving-block resample the original \
                         data (zero bootstrap variance); the usual rule of thumb \
                         is the Politis-White length (block_size=None) or about \
                         n^(1/3)"
                    ),
                ));
            }
            Ok((b, false))
        }
        None => {
            let mut sum = 0.0;
            for (k, col) in cols.iter().enumerate() {
                let obl = optimal_block_length(col).map_err(|e| {
                    param_err(
                        test,
                        "block_size",
                        "None",
                        format!(
                            "the automatic Politis-White block length could not \
                             be computed on loss column {k} ({e}); pass \
                             block_size explicitly (e.g. block_size=5)"
                        ),
                    )
                })?;
                sum += match scheme {
                    ResampleScheme::Stationary => obl.stationary,
                    ResampleScheme::CircularBlock | ResampleScheme::MovingBlock => obl.circular,
                };
            }
            let avg = sum / cols.len() as f64;
            let rounded = if avg.is_finite() { avg.round() } else { 1.0 };
            let b = (rounded.max(1.0) as usize).clamp(1, n - 1);
            Ok((b, true))
        }
    }
}

/// Validate `size` (a probability strictly inside `(0, 1)`).
pub(crate) fn check_size(test: &'static str, size: f64) -> Result<(), ForecastError> {
    if !(size > 0.0 && size < 1.0) {
        return Err(param_err(
            test,
            "size",
            size,
            "0 < size < 1 (the test size / family-wise error rate, e.g. 0.05 \
             or 0.10)",
        ));
    }
    Ok(())
}

/// Validate a replication count.
pub(crate) fn check_reps(
    test: &'static str,
    source: &IndexSource<'_>,
) -> Result<(), ForecastError> {
    let (what, reps) = match source {
        IndexSource::Seeded { reps, .. } => ("reps", *reps),
        IndexSource::Explicit(r) => ("resamples", r.len()),
    };
    if reps == 0 {
        return Err(param_err(
            test,
            what,
            reps,
            "at least one bootstrap replication (1000 is the conventional \
             default; use more for p-values near the decision boundary)",
        ));
    }
    Ok(())
}

/// Hansen's (2005, eq. 9) stationary-bootstrap long-run variance of each
/// column of the demeaned panel `e`, transcribed op-for-op from `arch`.
fn hansen_kernel_variance(e: &[Vec<f64>], n: usize, block_size: usize) -> Vec<f64> {
    let t = n as f64;
    let q = 1.0 / block_size as f64;
    let mut var: Vec<f64> = e
        .iter()
        .map(|col| {
            let mut s = 0.0;
            for &x in col {
                s += x * x;
            }
            s / t
        })
        .collect();
    for i in 1..n {
        let fi = i as f64;
        let kappa =
            ((1.0 - (fi / t)) * ((1.0 - q).powf(fi))) + ((fi / t) * ((1.0 - q).powf(t - fi)));
        for (col, v) in e.iter().zip(var.iter_mut()) {
            let mut s = 0.0;
            for (a, b) in col[..n - i].iter().zip(&col[i..]) {
                s += a * b;
            }
            *v += 2.0 * kappa * s / t;
        }
    }
    var
}

/// Index of the first maximum of `x` (non-empty, finite).
fn argmax_first(x: &[f64]) -> (usize, f64) {
    let mut best = 0usize;
    let mut mx = x[0];
    for (k, &v) in x.iter().enumerate().skip(1) {
        if v > mx {
            mx = v;
            best = k;
        }
    }
    (best, mx)
}

/// The SPA computation plus what StepM needs from it: the per-replicate,
/// per-model consistent-recentred values (unscaled) and the observed
/// per-model values.
struct SpaCore {
    result: SpaResult,
    zc: Vec<f64>,
    obs: Vec<f64>,
}

fn spa_core(
    test: &'static str,
    benchmark: &[f64],
    models: &[Vec<f64>],
    opts: &SpaOptions,
    source: IndexSource<'_>,
    keep_zc: bool,
) -> Result<SpaCore, ForecastError> {
    let n = benchmark.len();
    if n < 3 {
        return Err(param_err(
            test,
            "benchmark_losses",
            format!("{n} periods"),
            "at least 3 evaluation periods (the consistent re-centring \
             threshold uses sqrt(2 log log n), and a bootstrap of one or two \
             periods carries no information)",
        ));
    }
    check_finite_column(test, "benchmark_losses", benchmark, None)?;
    let m = models.len();
    if m == 0 {
        return Err(param_err(
            test,
            "model_losses",
            "0 columns",
            "at least one competing model (a T x m loss panel, one column per \
             model)",
        ));
    }
    for (k, col) in models.iter().enumerate() {
        if col.len() != n {
            return Err(ForecastError::RaggedLosses {
                what: "model_losses",
                index: k,
                expected: n,
                actual: col.len(),
            });
        }
        check_finite_column(test, "model_losses", col, Some(k))?;
    }
    check_reps(test, &source)?;

    // d_{t,k} = benchmark_t - model_{k,t}: positive favours the model.
    let d: Vec<Vec<f64>> = models
        .iter()
        .map(|col| benchmark.iter().zip(col).map(|(&b, &l)| b - l).collect())
        .collect();
    let dbar: Vec<f64> = d.iter().map(|col| column_mean(col)).collect();
    let e: Vec<Vec<f64>> = d
        .iter()
        .zip(&dbar)
        .map(|(col, &mu)| col.iter().map(|&v| v - mu).collect())
        .collect();
    if let Some(k) = e.iter().position(|col| col.iter().all(|&v| v == 0.0)) {
        return Err(ForecastError::ConstantLossColumn {
            what: test,
            index: k,
        });
    }
    let (block_size, block_size_auto) =
        resolve_block_size(test, opts.block_size, &d, n, opts.scheme)?;
    let source = match source {
        IndexSource::Seeded { seed, reps, .. } => IndexSource::Seeded {
            scheme: opts.scheme.block_scheme(block_size),
            seed,
            reps,
        },
        other => other,
    };

    // Bootstrap: the resampled column means of d (and, nested, of e).
    let width = if opts.nested { 2 * m } else { m };
    let nested = opts.nested;
    let rows = bootstrap_rows(test, &source, n, width, |idx, row| {
        let (a, b) = row.split_at_mut(m);
        resampled_means(&d, idx, None, a);
        if nested {
            resampled_means(&d, idx, Some(&dbar), b);
        }
    })?;
    let reps = source.reps();
    let t = n as f64;

    // omega_k^2.
    let omega2: Vec<f64> = if nested {
        (0..m)
            .map(|k| {
                let mut s = 0.0;
                for b in 0..reps {
                    s += rows[b * width + m + k];
                }
                let mean = s / reps as f64;
                let mut v = 0.0;
                for b in 0..reps {
                    let x = rows[b * width + m + k] - mean;
                    v += x * x;
                }
                t * (v / reps as f64)
            })
            .collect()
    } else {
        hansen_kernel_variance(&e, n, block_size)
    };
    if let Some(k) = omega2.iter().position(|&v| v.is_nan() || v <= 0.0) {
        return Err(ForecastError::ConstantLossColumn {
            what: test,
            index: k,
        });
    }
    let omega: Vec<f64> = omega2.iter().map(|v| v.sqrt()).collect();

    // Hansen's consistent re-centring rule.
    let loglog = t.ln().ln();
    let recentered: Vec<bool> = (0..m)
        .map(|k| dbar[k] >= -((omega2[k] / t) * 2.0 * loglog).sqrt())
        .collect();
    let g_lower: Vec<f64> = dbar
        .iter()
        .map(|&v| if v < 0.0 { 0.0 } else { v })
        .collect();
    let g_consistent: Vec<f64> = dbar
        .iter()
        .zip(&recentered)
        .map(|(&v, &r)| if r { v } else { 0.0 })
        .collect();
    let g_upper: Vec<f64> = dbar.clone();
    let g = [&g_lower, &g_consistent, &g_upper];

    // Replicate maxima under the three re-centrings, in replication order.
    let mut boot: [Vec<f64>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for v in boot.iter_mut() {
        v.try_reserve_exact(reps)
            .map_err(|_| ForecastError::AllocationRefused {
                what: test,
                elements: reps,
            })?;
    }
    let mut zc: Vec<f64> = Vec::new();
    if keep_zc {
        zc.try_reserve_exact(reps * m)
            .map_err(|_| ForecastError::AllocationRefused {
                what: test,
                elements: reps * m,
            })?;
    }
    let studentize = opts.studentize;
    for b in 0..reps {
        let row = &rows[b * width..b * width + m];
        for (j, gj) in g.iter().enumerate() {
            let mut mx = f64::NEG_INFINITY;
            for k in 0..m {
                let mut z = row[k] - gj[k];
                if studentize {
                    z /= omega[k];
                }
                if z > mx {
                    mx = z;
                }
                if j == 1 && keep_zc {
                    zc.push(z);
                }
            }
            boot[j].push(mx);
        }
    }
    let obs: Vec<f64> = (0..m)
        .map(|k| {
            if studentize {
                dbar[k] / omega[k]
            } else {
                dbar[k]
            }
        })
        .collect();
    let (best_model, mobs) = argmax_first(&obs);
    let pvals: Vec<f64> = boot
        .iter()
        .map(|v| v.iter().filter(|&&x| x > mobs).count() as f64 / reps as f64)
        .collect();

    // Report on the sqrt(n) scale.
    let sq = t.sqrt();
    let scale = |v: &[f64]| -> Vec<f64> { v.iter().map(|&x| sq * x).collect() };
    let boot_lower = scale(&boot[0]);
    let boot_consistent = scale(&boot[1]);
    let boot_upper = scale(&boot[2]);
    let crit = |v: &[f64]| -> [f64; 3] {
        let mut s = v.to_vec();
        s.sort_by(f64::total_cmp);
        let mut out = [0.0; 3];
        for (o, &level) in out.iter_mut().zip(&CRIT_LEVELS) {
            *o = numpy_percentile(&s, level);
        }
        out
    };
    let result = SpaResult {
        n,
        m,
        block_size,
        block_size_auto,
        scheme: opts.scheme,
        reps,
        studentize,
        nested,
        mean_loss_diff: dbar,
        loss_diff_var: omega2,
        recentered,
        statistic: sq * mobs,
        best_model,
        p_value_lower: pvals[0],
        p_value_consistent: pvals[1],
        p_value_upper: pvals[2],
        crit_levels: CRIT_LEVELS,
        crit_lower: crit(&boot_lower),
        crit_consistent: crit(&boot_consistent),
        crit_upper: crit(&boot_upper),
        boot_lower,
        boot_consistent,
        boot_upper,
    };
    Ok(SpaCore { result, zc, obs })
}

/// White's Reality Check / Hansen's SPA test of `H0: no model beats the
/// benchmark`.
///
/// * `benchmark` — the benchmark's losses over the `n` evaluation periods.
/// * `models` — one loss column per competing model, each of length `n`
///   and index-aligned with `benchmark` (the columns of a `T x m` loss
///   panel).
/// * `opts` — see [`SpaOptions`].
///
/// See the [module docs](self) for the statistics, the three re-centrings,
/// the variance kernel and the resampling conventions.
///
/// # Errors
///
/// [`ForecastError::InvalidMultipleComparisonParam`] (fewer than 3
/// periods, no models, non-finite losses, `reps = 0`, `block_size` outside
/// `1..n`, or an automatic block length that cannot be computed),
/// [`ForecastError::RaggedLosses`] (a model column of the wrong length),
/// [`ForecastError::ConstantLossColumn`] (a model with losses identical to
/// the benchmark's), [`ForecastError::AllocationRefused`] (a
/// `reps x m` buffer beyond memory), and wrapped bootstrap errors.
pub fn spa_test(
    benchmark: &[f64],
    models: &[Vec<f64>],
    opts: &SpaOptions,
) -> Result<SpaResult, ForecastError> {
    let source = IndexSource::Seeded {
        // Replaced by the resolved block scheme inside spa_core.
        scheme: BlockScheme::Iid,
        seed: opts.seed,
        reps: opts.reps,
    };
    Ok(spa_core("spa_test", benchmark, models, opts, source, false)?.result)
}

/// [`spa_test`] with caller-supplied resample indices — one length-`n`
/// index vector per replication — instead of the seeded bootstrap.
///
/// This is the validation entry point: fed the resamples another
/// implementation drew (the fixture stores `arch`'s), it reproduces that
/// implementation's replicate statistics and p-values exactly rather than
/// at Monte Carlo tolerance. `opts.reps` and `opts.seed` are ignored (the
/// number of replications is `resamples.len()`); `opts.block_size` still
/// sets the restart probability of Hansen's variance kernel.
///
/// # Errors
///
/// As [`spa_test`], plus [`ForecastError::InvalidResample`] for an index
/// vector of the wrong length or with an index outside `0..n`.
pub fn spa_test_with_indices(
    benchmark: &[f64],
    models: &[Vec<f64>],
    resamples: &[Vec<usize>],
    opts: &SpaOptions,
) -> Result<SpaResult, ForecastError> {
    let source = IndexSource::Explicit(resamples);
    Ok(spa_core("spa_test", benchmark, models, opts, source, false)?.result)
}

fn stepm_core(core: SpaCore, size: f64) -> Result<StepmResult, ForecastError> {
    const TEST: &str = "stepm_test";
    check_size(TEST, size)?;
    let SpaCore { result, zc, obs } = core;
    let m = result.m;
    let reps = result.reps;
    let sq = (result.n as f64).sqrt();
    let q = 1.0 - size;
    let mut remaining = vec![true; m];
    let mut superior: Vec<usize> = Vec::new();
    let mut steps: Vec<Vec<usize>> = Vec::new();
    let mut crits: Vec<f64> = Vec::new();
    let mut mx = vec![0.0; reps];
    loop {
        // The bootstrap maximum over the models still in play.
        for (b, slot) in mx.iter_mut().enumerate() {
            let row = &zc[b * m..(b + 1) * m];
            let mut best = f64::NEG_INFINITY;
            for (k, &z) in row.iter().enumerate() {
                if remaining[k] && z > best {
                    best = z;
                }
            }
            *slot = best;
        }
        let mut sorted = mx.clone();
        sorted.sort_by(f64::total_cmp);
        let crit = numpy_percentile(&sorted, q);
        let better: Vec<usize> = (0..m).filter(|&k| remaining[k] && obs[k] > crit).collect();
        crits.push(sq * crit);
        let added = better.len();
        for &k in &better {
            remaining[k] = false;
        }
        superior.extend(better.iter().copied());
        steps.push(better);
        if added == 0 || superior.len() == m {
            break;
        }
    }
    superior.sort_unstable();
    Ok(StepmResult {
        spa: result,
        size,
        superior_models: superior,
        steps,
        step_crit_values: crits,
    })
}

/// The Romano-Wolf (2005) StepM procedure: which models beat the benchmark,
/// at family-wise error rate `opts.size`, on the SPA bootstrap.
///
/// Arguments as [`spa_test`]; see the [module docs](self) for the stepwise
/// rule. The returned [`StepmResult::spa`] is the full-set SPA test.
///
/// # Errors
///
/// As [`spa_test`], plus [`ForecastError::InvalidMultipleComparisonParam`]
/// for `size` outside `(0, 1)`.
pub fn stepm_test(
    benchmark: &[f64],
    models: &[Vec<f64>],
    opts: &StepmOptions,
) -> Result<StepmResult, ForecastError> {
    check_size("stepm_test", opts.size)?;
    let source = IndexSource::Seeded {
        scheme: BlockScheme::Iid,
        seed: opts.spa.seed,
        reps: opts.spa.reps,
    };
    let core = spa_core("stepm_test", benchmark, models, &opts.spa, source, true)?;
    stepm_core(core, opts.size)
}

/// [`stepm_test`] with caller-supplied resample indices (see
/// [`spa_test_with_indices`]).
///
/// # Errors
///
/// As [`stepm_test`] and [`spa_test_with_indices`].
pub fn stepm_test_with_indices(
    benchmark: &[f64],
    models: &[Vec<f64>],
    resamples: &[Vec<usize>],
    opts: &StepmOptions,
) -> Result<StepmResult, ForecastError> {
    check_size("stepm_test", opts.size)?;
    let source = IndexSource::Explicit(resamples);
    let core = spa_core("stepm_test", benchmark, models, &opts.spa, source, true)?;
    stepm_core(core, opts.size)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn pairwise_sum_matches_numpy_hand_cases() {
        // Below 8: plain accumulation.
        let a: Vec<f64> = (1..=5).map(|i| i as f64 * 0.1).collect();
        let seq: f64 = a.iter().fold(0.0, |s, &v| s + v);
        assert_eq!(numpy_pairwise_sum(&a).to_bits(), seq.to_bits());
        // 8..=128: eight interleaved partial sums, then the fixed tree.
        let a: Vec<f64> = (0..19).map(|i| ((i * 7919) % 101) as f64 / 13.0).collect();
        let mut r = [a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]];
        for j in 0..8 {
            r[j] += a[8 + j];
        }
        let mut want = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for &v in &a[16..] {
            want += v;
        }
        assert_eq!(numpy_pairwise_sum(&a).to_bits(), want.to_bits());
        // > 128: split at the largest multiple of 8 below n/2.
        let a: Vec<f64> = (0..300).map(|i| ((i * 31) % 97) as f64 / 7.0).collect();
        let want = numpy_pairwise_sum(&a[..144]) + numpy_pairwise_sum(&a[144..]);
        assert_eq!(numpy_pairwise_sum(&a).to_bits(), want.to_bits());
    }

    #[test]
    fn percentile_matches_numpy_linear_hand_cases() {
        // numpy.percentile([1, 2, 3, 4], 90) == 3.7; 50 -> 2.5; 0 -> 1; 100 -> 4.
        let x = [1.0, 2.0, 3.0, 4.0];
        assert!((numpy_percentile(&x, 0.9) - 3.7).abs() < 1e-12);
        assert!((numpy_percentile(&x, 0.5) - 2.5).abs() < 1e-12);
        assert_eq!(numpy_percentile(&x, 0.0), 1.0);
        assert_eq!(numpy_percentile(&x, 1.0), 4.0);
        assert_eq!(numpy_percentile(&[7.0], 0.95), 7.0);
        // numpy.percentile(range(10), 95) == 8.55 (the gamma >= 0.5 branch).
        let y: Vec<f64> = (0..10).map(|i| i as f64).collect();
        assert!((numpy_percentile(&y, 0.95) - 8.55).abs() < 1e-12);
    }

    #[test]
    fn hansen_kernel_reduces_to_the_sample_variance_at_block_one() {
        // q = 1: (1 - q)^i = 0 for i >= 1, so only gamma_0 survives.
        let e = vec![vec![1.0, -2.0, 0.5, 0.5]];
        let v = hansen_kernel_variance(&e, 4, 1);
        assert!((v[0] - (1.0 + 4.0 + 0.25 + 0.25) / 4.0).abs() < 1e-15);
    }
}
