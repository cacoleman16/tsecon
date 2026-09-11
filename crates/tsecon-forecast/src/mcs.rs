//! The Model Confidence Set of Hansen, Lunde & Nason (2011).
//!
//! # The question
//!
//! Given the losses of `m` models over `n` evaluation periods, which models
//! are statistically indistinguishable from the best? The MCS answers with a
//! SET `M*_{1-alpha}` that contains the best model(s) with probability at
//! least `1 - alpha` asymptotically, built by sequential elimination: test
//! the null of equal predictive ability across the models still in the set;
//! if it is rejected at level `alpha`, eliminate the worst model and repeat;
//! otherwise stop. Each eliminated model records the p-value of the step that
//! removed it, and the MCS p-value of a model is the running maximum of
//! those step p-values along the elimination path (so that the set at any
//! size `alpha` is `{k : p_MCS(k) > alpha}`, and the sets are nested in
//! `alpha`).
//!
//! # The two statistics
//!
//! Let `dbar_{ij} = mean_t (L_{t,i} - L_{t,j})` and `dbar_{i.}` the mean of
//! model `i`'s loss minus the cross-sectional mean of the models still in
//! the set. With bootstrap variance estimates of those means,
//!
//! ```text
//! T_R   = max_{i,j in M} |dbar_{ij}| / sqrt(Var*(dbar_{ij}))     (method "R", range)
//! T_max = max_{i in M}    dbar_{i.} / sqrt(Var*(dbar_{i.}))      (method "max")
//! ```
//!
//! Under `T_R` the model eliminated is the `i` of the maximizing pair
//! (the one with the larger loss); under `T_max` it is the model attaining
//! the maximum (every model tied at the maximum is eliminated together).
//! The null distribution of each statistic is the same block bootstrap of
//! the loss panel — one set of resamples reused at every step, so the
//! elimination path is coherent — with the bootstrap means re-centred at the
//! sample means. Op-for-op this is `arch.bootstrap.MCS`: the pairwise
//! bootstrap variances `Var*(dbar_{ij})` are the replicate mean squares of
//! the re-centred resampled differences with `1` added on the diagonal
//! (so `i = j` is `0 / 1`, never `0 / 0`); the `T_max` standard deviations
//! are recomputed at every step from the resampled means re-centred at the
//! cross-sectional mean of the remaining set.
//!
//! # Conventions, budget, determinism
//!
//! `block_size`, the resampling schemes and the automatic Politis-White
//! block length are those of [`crate::spa`] (the length is computed on the
//! loss columns themselves). The `reps x m` matrix of resampled means is the
//! only replication-sized buffer and is budgeted with `try_reserve`; the
//! range method additionally keeps an `m x m` variance matrix. Rows are
//! resampled in parallel and every reduction runs in replication order, so
//! the result is bit-identical at any rayon thread count.
//!
//! # Validation
//!
//! `arch.bootstrap.MCS` is the golden: [`model_confidence_set_with_indices`]
//! fed `arch`'s own resample indices reproduces its statistics, step
//! p-values, elimination order and included set exactly (the fixture
//! generator asserts the summation orders bit-for-bit before storing a
//! case), and the seeded [`model_confidence_set`] is pinned at Monte Carlo
//! tolerance; coverage of the true best model is measured by seeded Monte
//! Carlo in the crate's property tests. `arch` warns and continues with a
//! zero standard deviation (identical loss columns); this crate refuses.
//!
//! # References
//!
//! Hansen, P. R., Lunde, A. & Nason, J. M. (2011). "The Model Confidence
//! Set." *Econometrica* 79(2), 453-497.

use crate::error::ForecastError;
use crate::spa::{
    bootstrap_rows, check_finite_column, check_reps, check_size, column_mean, numpy_pairwise_sum,
    resampled_means, resolve_block_size, IndexSource, ResampleScheme,
};
use tsecon_bootstrap::BlockScheme;

/// The elimination statistic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum McsMethod {
    /// The range statistic `T_R` (`method="R"`), HLN's recommended default.
    Range,
    /// The max statistic `T_max` (`method="max"`).
    Max,
}

impl McsMethod {
    /// The method's name as the Python surface spells it.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            McsMethod::Range => "R",
            McsMethod::Max => "max",
        }
    }
}

/// Options of [`model_confidence_set`].
#[derive(Debug, Clone, PartialEq)]
pub struct McsOptions {
    /// The size `alpha` of the sequential tests; the set is
    /// `{k : p_MCS(k) > size}`.
    pub size: f64,
    /// The elimination statistic.
    pub method: McsMethod,
    /// (Expected) block length; `None` selects the Politis-White length.
    pub block_size: Option<usize>,
    /// Number of bootstrap replications (`>= 1`).
    pub reps: usize,
    /// The resampling scheme.
    pub scheme: ResampleScheme,
    /// Seed of the Philox substream hierarchy.
    pub seed: u64,
}

impl Default for McsOptions {
    fn default() -> Self {
        McsOptions {
            size: 0.10,
            method: McsMethod::Range,
            block_size: None,
            reps: 1000,
            scheme: ResampleScheme::Stationary,
            seed: 0,
        }
    }
}

/// Result of [`model_confidence_set`] / [`model_confidence_set_with_indices`].
#[derive(Debug, Clone, PartialEq)]
pub struct McsResult {
    /// Number of evaluation periods.
    pub n: usize,
    /// Number of models.
    pub m: usize,
    /// The size used.
    pub size: f64,
    /// The elimination statistic used.
    pub method: McsMethod,
    /// The block length actually used.
    pub block_size: usize,
    /// Whether `block_size` came from the automatic Politis-White rule.
    pub block_size_auto: bool,
    /// The resampling scheme.
    pub scheme: ResampleScheme,
    /// Number of bootstrap replications.
    pub reps: usize,
    /// Mean loss of each model.
    pub mean_losses: Vec<f64>,
    /// Models in the confidence set (`p_MCS > size`), ascending.
    pub included: Vec<usize>,
    /// Models eliminated from the set (`p_MCS <= size`), ascending.
    pub excluded: Vec<usize>,
    /// Every model in the order it was eliminated; the survivor(s) of the
    /// last step come last.
    pub elimination_order: Vec<usize>,
    /// The raw p-value of the step that eliminated each model, aligned with
    /// `elimination_order` (`1.0` for the survivors).
    pub step_p_values: Vec<f64>,
    /// The observed statistic (`T_R` or `T_max`) at each elimination step.
    pub statistics: Vec<f64>,
    /// The MCS p-value of each model (index = model): the running maximum
    /// of `step_p_values` along the elimination path.
    pub mcs_p_values: Vec<f64>,
    /// Number of elimination steps performed.
    pub n_steps: usize,
}

const TEST: &str = "model_confidence_set";

fn param_err(
    what: &'static str,
    value: impl std::fmt::Display,
    requirement: impl Into<String>,
) -> ForecastError {
    ForecastError::InvalidMultipleComparisonParam {
        test: TEST,
        what,
        value: value.to_string(),
        requirement: requirement.into(),
    }
}

/// Validate the loss panel: `m >= 2` index-aligned finite columns of
/// `n >= 2` periods. Returns `(n, m)`.
fn validate_losses(losses: &[Vec<f64>]) -> Result<(usize, usize), ForecastError> {
    let m = losses.len();
    if m < 2 {
        return Err(param_err(
            "losses",
            format!("{m} column(s)"),
            "at least two models (a T x m loss panel, one column per model); \
             a confidence set over one model is that model",
        ));
    }
    let n = losses[0].len();
    if n < 2 {
        return Err(param_err(
            "losses",
            format!("{n} period(s)"),
            "at least two evaluation periods per model",
        ));
    }
    for (k, col) in losses.iter().enumerate() {
        if col.len() != n {
            return Err(ForecastError::RaggedLosses {
                what: "losses",
                index: k,
                expected: n,
                actual: col.len(),
            });
        }
        check_finite_column(TEST, "losses", col, Some(k))?;
    }
    Ok((n, m))
}

struct Elimination {
    order: Vec<usize>,
    p_values: Vec<f64>,
    statistics: Vec<f64>,
}

/// The range (`T_R`) elimination on the resampled means `rows` (`reps x m`).
fn eliminate_range(
    rows: &[f64],
    reps: usize,
    m: usize,
    mean: &[f64],
) -> Result<Elimination, ForecastError> {
    // Pairwise bootstrap variances of the re-centred resampled differences,
    // accumulated in replication order, plus the identity on the diagonal.
    let mm = m.checked_mul(m).ok_or(ForecastError::AllocationRefused {
        what: TEST,
        elements: usize::MAX,
    })?;
    let mut var: Vec<f64> = Vec::new();
    var.try_reserve_exact(mm)
        .map_err(|_| ForecastError::AllocationRefused {
            what: TEST,
            elements: mm,
        })?;
    var.resize(mm, 0.0);
    for b in 0..reps {
        let ms = &rows[b * m..(b + 1) * m];
        for i in 0..m {
            for j in 0..m {
                let d = (ms[i] - ms[j]) - (mean[i] - mean[j]);
                var[i * m + j] += d * d;
            }
        }
    }
    for v in var.iter_mut() {
        *v /= reps as f64;
    }
    for i in 0..m {
        var[i * m + i] += 1.0;
    }
    for i in 0..m {
        for j in (i + 1)..m {
            let v = var[i * m + j];
            if v.is_nan() || v <= 0.0 {
                return Err(ForecastError::ZeroBootstrapVariance {
                    what: "model_confidence_set(method=\"R\")",
                    model: i,
                    other: Some(j),
                });
            }
        }
    }
    let sd: Vec<f64> = var.iter().map(|v| v.sqrt()).collect();
    let std_ld: Vec<f64> = (0..mm)
        .map(|ij| (mean[ij / m] - mean[ij % m]) / sd[ij])
        .collect();

    let mut included = vec![true; m];
    let mut order = Vec::with_capacity(m);
    let mut p_values = Vec::with_capacity(m);
    let mut statistics = Vec::with_capacity(m);
    let mut n_incl = m;
    while n_incl > 1 {
        let incl: Vec<usize> = (0..m).filter(|&k| included[k]).collect();
        // Observed T_R: the first maximum in row-major order over the
        // included submatrix; its row is the model eliminated.
        let mut test_stat = f64::NEG_INFINITY;
        let mut loc = 0usize;
        for (a, &i) in incl.iter().enumerate() {
            for &j in &incl {
                let v = std_ld[i * m + j];
                if v > test_stat {
                    test_stat = v;
                    loc = a;
                }
            }
        }
        let mut count = 0usize;
        for b in 0..reps {
            let ms = &rows[b * m..(b + 1) * m];
            let mut mx = f64::NEG_INFINITY;
            for &i in &incl {
                for &j in &incl {
                    let z = ((ms[i] - ms[j]) - (mean[i] - mean[j])) / sd[i * m + j];
                    if z > mx {
                        mx = z;
                    }
                }
            }
            if test_stat < mx {
                count += 1;
            }
        }
        let pval = count as f64 / reps as f64;
        let out = incl[loc];
        order.push(out);
        p_values.push(pval);
        statistics.push(test_stat);
        included[out] = false;
        n_incl -= 1;
    }
    for (k, &inc) in included.iter().enumerate() {
        if inc {
            order.push(k);
            p_values.push(1.0);
        }
    }
    Ok(Elimination {
        order,
        p_values,
        statistics,
    })
}

/// The max (`T_max`) elimination on the resampled centred means `rows`
/// (`reps x m`, each row already centred at its own cross-sectional mean).
fn eliminate_max(
    rows: &[f64],
    reps: usize,
    m: usize,
    mean: &[f64],
) -> Result<Elimination, ForecastError> {
    let mut included = vec![true; m];
    let mut order = Vec::with_capacity(m);
    let mut p_values = Vec::with_capacity(m);
    let mut statistics = Vec::with_capacity(m);
    let mut n_incl = m;
    while n_incl > 1 {
        let incl: Vec<usize> = (0..m).filter(|&k| included[k]).collect();
        let kk = incl.len();
        let kf = kk as f64;
        // Pass 1: the standard deviations of the re-centred resampled means.
        let mut tmp = vec![0.0; kk];
        let mut ss = vec![0.0; kk];
        for b in 0..reps {
            let row = &rows[b * m..(b + 1) * m];
            for (slot, &k) in tmp.iter_mut().zip(&incl) {
                *slot = row[k];
            }
            let grand = numpy_pairwise_sum(&tmp) / kf;
            for (s, &v) in ss.iter_mut().zip(&tmp) {
                let c = v - grand;
                *s += c * c;
            }
        }
        let sd: Vec<f64> = ss.iter().map(|s| (s / reps as f64).sqrt()).collect();
        if let Some(a) = sd.iter().position(|&s| s.is_nan() || s <= 0.0) {
            return Err(ForecastError::ZeroBootstrapVariance {
                what: "model_confidence_set(method=\"max\")",
                model: incl[a],
                other: None,
            });
        }
        // Observed T_max.
        let mut ld: Vec<f64> = incl.iter().map(|&k| mean[k]).collect();
        let gm = numpy_pairwise_sum(&ld) / kf;
        for v in ld.iter_mut() {
            *v -= gm;
        }
        let std_ld: Vec<f64> = ld.iter().zip(&sd).map(|(&v, &s)| v / s).collect();
        let mut test_stat = f64::NEG_INFINITY;
        for &v in &std_ld {
            if v > test_stat {
                test_stat = v;
            }
        }
        // Pass 2: the bootstrap maxima.
        let mut count = 0usize;
        for b in 0..reps {
            let row = &rows[b * m..(b + 1) * m];
            for (slot, &k) in tmp.iter_mut().zip(&incl) {
                *slot = row[k];
            }
            let grand = numpy_pairwise_sum(&tmp) / kf;
            let mut mx = f64::NEG_INFINITY;
            for (&v, &s) in tmp.iter().zip(&sd) {
                let z = (v - grand) / s;
                if z > mx {
                    mx = z;
                }
            }
            if test_stat < mx {
                count += 1;
            }
        }
        let pval = count as f64 / reps as f64;
        for (a, &v) in std_ld.iter().enumerate() {
            if v == test_stat {
                order.push(incl[a]);
                p_values.push(pval);
                included[incl[a]] = false;
                n_incl -= 1;
            }
        }
        statistics.push(test_stat);
    }
    for (k, &inc) in included.iter().enumerate() {
        if inc {
            order.push(k);
            p_values.push(1.0);
        }
    }
    Ok(Elimination {
        order,
        p_values,
        statistics,
    })
}

fn mcs_core(
    losses: &[Vec<f64>],
    opts: &McsOptions,
    source: IndexSource<'_>,
) -> Result<McsResult, ForecastError> {
    let (n, m) = validate_losses(losses)?;
    check_size(TEST, opts.size)?;
    check_reps(TEST, &source)?;
    let (block_size, block_size_auto) =
        resolve_block_size(TEST, opts.block_size, losses, n, opts.scheme)?;
    let source = match source {
        IndexSource::Seeded { seed, reps, .. } => IndexSource::Seeded {
            scheme: opts.scheme.block_scheme(block_size),
            seed,
            reps,
        },
        other => other,
    };
    let mean: Vec<f64> = losses.iter().map(|col| column_mean(col)).collect();

    let (rows, elim) = match opts.method {
        McsMethod::Range => {
            let rows = bootstrap_rows(TEST, &source, n, m, |idx, row| {
                resampled_means(losses, idx, None, row);
            })?;
            let reps = source.reps();
            let elim = eliminate_range(&rows, reps, m, &mean)?;
            (rows, elim)
        }
        McsMethod::Max => {
            // arch: loss_errors = losses - mean; per replicate the resampled
            // mean of loss_errors, centred at its cross-sectional mean.
            let errs: Vec<Vec<f64>> = losses
                .iter()
                .zip(&mean)
                .map(|(col, &mu)| col.iter().map(|&v| v - mu).collect())
                .collect();
            let mf = m as f64;
            let rows = bootstrap_rows(TEST, &source, n, m, |idx, row| {
                resampled_means(&errs, idx, None, row);
                let grand = numpy_pairwise_sum(row) / mf;
                for v in row.iter_mut() {
                    *v -= grand;
                }
            })?;
            let reps = source.reps();
            let elim = eliminate_max(&rows, reps, m, &mean)?;
            (rows, elim)
        }
    };
    let reps = rows.len() / m;

    // MCS p-values: running maximum along the elimination path.
    let mut mcs_p = vec![0.0; m];
    let mut running = f64::NEG_INFINITY;
    for (&k, &p) in elim.order.iter().zip(&elim.p_values) {
        if p > running {
            running = p;
        }
        mcs_p[k] = running;
    }
    let included: Vec<usize> = (0..m).filter(|&k| mcs_p[k] > opts.size).collect();
    let excluded: Vec<usize> = (0..m).filter(|&k| mcs_p[k] <= opts.size).collect();
    let n_steps = elim.statistics.len();
    Ok(McsResult {
        n,
        m,
        size: opts.size,
        method: opts.method,
        block_size,
        block_size_auto,
        scheme: opts.scheme,
        reps,
        mean_losses: mean,
        included,
        excluded,
        elimination_order: elim.order,
        step_p_values: elim.p_values,
        statistics: elim.statistics,
        mcs_p_values: mcs_p,
        n_steps,
    })
}

/// The Hansen-Lunde-Nason (2011) Model Confidence Set.
///
/// * `losses` — one loss column per model (`m >= 2`), each of length `n`
///   and index-aligned (the columns of a `T x m` loss panel).
/// * `opts` — see [`McsOptions`].
///
/// See the [module docs](self) for the statistics, the elimination rule and
/// the p-value construction.
///
/// # Errors
///
/// [`ForecastError::InvalidMultipleComparisonParam`] (fewer than two models
/// or two periods, non-finite losses, `size` outside `(0, 1)`, `reps = 0`,
/// `block_size` outside `1..n`, or an automatic block length that cannot be
/// computed), [`ForecastError::RaggedLosses`],
/// [`ForecastError::ZeroBootstrapVariance`] (identical loss columns),
/// [`ForecastError::AllocationRefused`], and wrapped bootstrap errors.
pub fn model_confidence_set(
    losses: &[Vec<f64>],
    opts: &McsOptions,
) -> Result<McsResult, ForecastError> {
    let source = IndexSource::Seeded {
        scheme: BlockScheme::Iid,
        seed: opts.seed,
        reps: opts.reps,
    };
    mcs_core(losses, opts, source)
}

/// [`model_confidence_set`] with caller-supplied resample indices — one
/// length-`n` index vector per replication — instead of the seeded
/// bootstrap (the validation entry point; see [`crate::spa_test_with_indices`]).
/// `opts.reps` and `opts.seed` are ignored.
///
/// # Errors
///
/// As [`model_confidence_set`], plus [`ForecastError::InvalidResample`].
pub fn model_confidence_set_with_indices(
    losses: &[Vec<f64>],
    resamples: &[Vec<usize>],
    opts: &McsOptions,
) -> Result<McsResult, ForecastError> {
    let source = IndexSource::Explicit(resamples);
    mcs_core(losses, opts, source)
}
