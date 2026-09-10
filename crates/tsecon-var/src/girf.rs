//! Koop-Pesaran-Potter (1996) **generalized impulse responses** by
//! simulation — the one engine every nonlinear model in the library reuses.
//!
//! # The object
//!
//! For a model with history (state) `ω_{t-1}` and a reduced-form innovation
//! `u_t`, the generalized impulse response of Koop, Pesaran and Potter
//! (1996, *Journal of Econometrics* 74) is the difference between two
//! conditional expectations,
//!
//! ```text
//! GIRF(h, δ, ω_{t-1}) = E[y_{t+h} | u_t + δ, ω_{t-1}] - E[y_{t+h} | ω_{t-1}],
//! ```
//!
//! evaluated by Monte Carlo: draw future innovations, simulate the model
//! forward twice from the same history — once with the shock `δ` added to
//! the period-`t` innovation, once without — and average the paired
//! differences. In a linear model the paired difference is `Ψ_h δ` for
//! *every* draw and every history (the future innovations cancel exactly),
//! so the engine reduces to the moving-average impulse response with zero
//! Monte Carlo noise; in a nonlinear model the response depends on the
//! history, the sign and the size of `δ`, and the spread of the
//! history-conditional responses across histories is itself the finding.
//!
//! # What a model must provide — [`GirfModel`]
//!
//! * a **history** of `history_len()` consecutive observations of the `k`
//!   variables (a flat slice, rows oldest first, so row `history_len() - 1`
//!   is `y_{t-1}`) — for a VAR(p) that is `p` rows, for a threshold VAR with
//!   delay `d` it is `max(p, d)` rows because the regime at `t` reads
//!   `y_{t-d}`;
//! * [`GirfModel::state`] — which of `n_states()` innovation covariances
//!   applies to the innovation of the *next* period given the history (one
//!   state for a linear VAR; the regime for a threshold VAR);
//! * [`GirfModel::covariance`] — the `k × k` covariance of each state's
//!   innovation; the engine Cholesky-factors each once;
//! * [`GirfModel::step`] — the law of motion `y_t = f(history, u_t)`.
//!
//! # Conventions (every one of these is transcribed in
//! `fixtures/generate_girf_fixtures.py` and pinned)
//!
//! **Shock.** With `s_0 = state(history)` the state at the shock date,
//!
//! * [`GirfShock::Orthogonal`] — `δ = size · L_{s_0} e_j`, column `j` of
//!   the lower Cholesky factor of `Σ_{s_0}`: a shock of `size` standard
//!   deviations to the `j`-th orthogonalized innovation under the recursive
//!   ordering of the variables (`var_irf(orth=True)`'s shock);
//! * [`GirfShock::Generalized`] — `δ = size · Σ_{s_0} e_j / sqrt(σ_{jj})`,
//!   the Pesaran-Shin (1998) generalized shock: innovation `j` moved by
//!   `size` standard deviations and the others by their conditional
//!   expectation given that move, `E[u | u_j = size·sqrt(σ_jj)]`, with no
//!   ordering assumption;
//! * [`GirfShock::Raw`] — a user-supplied innovation vector `δ`, the same
//!   for every history.
//!
//! `size` is in standard deviations of the innovation of the state the
//! history is in *at the shock date*; when the states' covariances differ,
//! the raw impact of a "one standard deviation" shock differs across states
//! and [`Girf::shock_by_state`] reports each.
//!
//! **Common random numbers.** For history `i` and draw `r` one sequence of
//! `(horizon + 1) · k` standard normals `z_0, …, z_H` is drawn and used by
//! *both* paths. At period `h` each path reads its own state `s_h` from its
//! own window and scales the common draw, `u_h = L_{s_h} z_h`; the shocked
//! path adds `δ` at `h = 0`. Innovations are therefore drawn from the
//! covariance of the state the *simulated* path is in at that period — when
//! a path switches regime its innovation covariance switches with it (the
//! regime-by-regime draw scheme Balke 2000 describes for his TVAR
//! responses). Because the period-0 draw is shared too, the paired
//! difference of a linear model is exactly `Ψ_h δ` for every draw: this is
//! the *perturbation* form of the KPP definition (shock added to the drawn
//! innovation), which coincides with the *realization* form (innovation
//! set to `δ`) in a linear model and differs from it in a nonlinear one by
//! the averaging over the period-`t` draw.
//!
//! **Antithetic variates.** With `antithetic = true`, draws come in pairs
//! `(+z, −z)` sharing one substream; `n_draws` must be even and the
//! Monte Carlo standard error is computed from the pair averages (the
//! only independent units).
//!
//! **Reproducibility.** History `i` owns child `i` of
//! `SeedSequence(seed)`; draw stream `r` of that history is child `r` of
//! child `i`; the normals are Box-Muller `sqrt(-2 ln(1 - U_1)) cos(2π U_2)`
//! on consecutive NumPy-convention uniforms. Histories are simulated in
//! parallel and every reduction runs in history order afterwards, so the
//! result is bit-identical at any rayon thread count and across process
//! restarts. NumPy's `SeedSequence.spawn` / `Philox` / `Generator.random`
//! reproduce the same streams, which is how the fixture generator
//! transcribes the whole engine independently.
//!
//! **Summaries.** `per_history[i]` is the history-conditional GIRF (mean
//! over draws); `girf` its mean over histories; `lower`/`upper` the
//! `bands` quantiles across histories (NumPy `percentile(method="linear")`)
//! — the KPP history-conditional distribution; `mc_se` the Monte Carlo
//! standard error of `girf` (histories fixed, future innovations
//! integrated); `draw_sd` the typical spread of a single realized paired
//! difference across future-innovation draws (root mean over histories of
//! the across-draw variance; zero for a linear model); `draw_lower`/
//! `draw_upper` the mean over histories of the per-history across-draw
//! `bands` quantiles.
//!
//! # References
//!
//! Koop, Pesaran & Potter (1996, JoE 74); Pesaran & Shin (1998, Economics
//! Letters 58); Balke (2000, REStat 82); Kilian & Lütkepohl (2017, ch. 18).

use rayon::prelude::*;
use tsecon_bootstrap::WildWeights;
use tsecon_linalg::faer::{Mat, MatRef};
use tsecon_rng::{SeedSequence, Stream};

use crate::error::VarError;
use crate::irf_bootstrap::percentile_sorted;
use crate::results::{chol_lower, VarResults};

/// Largest accepted `horizon` (a guard against absurd allocations, not a
/// modelling limit).
const MAX_HORIZON: usize = 1_000_000;
/// Largest `n_draws × cells` buffer one history may allocate (2^31 doubles).
const MAX_BUFFER: usize = 1 << 31;
/// Offset added to `seed` to key the history subsample, so it never shares
/// a substream with the simulation (an odd 64-bit constant, like the
/// bias-bootstrap offset in `irf_bootstrap.rs`).
pub const HISTORY_SEED_OFFSET: u64 = 0xD1B5_4A32_D192_ED03;

/// A model the GIRF engine can simulate. See the module docs for the
/// history layout and the state / covariance contract.
pub trait GirfModel: Sync {
    /// Number of variables `k`.
    fn n_vars(&self) -> usize;
    /// Number of consecutive past observations a history carries.
    fn history_len(&self) -> usize;
    /// Number of innovation states (distinct covariances).
    fn n_states(&self) -> usize;
    /// The state whose covariance governs the innovation of the next period,
    /// given `history` (`history_len() * n_vars()` values, rows oldest
    /// first). Must be `< n_states()`.
    fn state(&self, history: &[f64]) -> usize;
    /// The `k × k` innovation covariance of `state` (rows).
    fn covariance(&self, state: usize) -> Vec<Vec<f64>>;
    /// The law of motion: write `y_t` into `out` (length `k`) given the
    /// history and the reduced-form innovation `u_t` (length `k`).
    fn step(&self, history: &[f64], innovation: &[f64], out: &mut [f64]);
}

/// How the shock at the impact period is formed (module docs).
#[derive(Debug, Clone, PartialEq)]
pub enum GirfShock {
    /// `size` standard deviations of the `var`-th orthogonalized (Cholesky,
    /// variable ordering) innovation of the history's state.
    Orthogonal {
        /// 0-based index of the shocked variable.
        var: usize,
        /// Shock size in standard deviations (sign allowed).
        size: f64,
    },
    /// The Pesaran-Shin generalized shock: innovation `var` moved by `size`
    /// standard deviations, the others by their conditional mean.
    Generalized {
        /// 0-based index of the shocked variable.
        var: usize,
        /// Shock size in standard deviations (sign allowed).
        size: f64,
    },
    /// A raw innovation vector (length `k`), identical for every history.
    Raw(Vec<f64>),
}

/// Options of [`girf`].
#[derive(Debug, Clone, PartialEq)]
pub struct GirfOptions {
    /// The shock (module docs).
    pub shock: GirfShock,
    /// Last horizon `H`; every path array has `H + 1` rows.
    pub horizon: usize,
    /// Future-innovation draws per history (even when `antithetic`).
    pub n_draws: usize,
    /// Reproducibility seed.
    pub seed: u64,
    /// Antithetic `(+z, −z)` pairs.
    pub antithetic: bool,
    /// Quantile levels `(lower, upper)` of the across-history and
    /// across-draw bands.
    pub bands: (f64, f64),
}

/// Output of [`girf`]. Path arrays are `[h][variable]`, `h = 0..=horizon`.
#[derive(Debug, Clone, PartialEq)]
pub struct Girf {
    /// Mean over histories of the history-conditional GIRFs.
    pub girf: Vec<Vec<f64>>,
    /// `bands.0` quantile across histories.
    pub lower: Vec<Vec<f64>>,
    /// `bands.1` quantile across histories.
    pub upper: Vec<Vec<f64>>,
    /// The history-conditional GIRFs, `[history][h][variable]`.
    pub per_history: Vec<Vec<Vec<f64>>>,
    /// Monte Carlo standard error of `girf` (histories fixed); `NaN` when
    /// fewer than two effective draws exist.
    pub mc_se: Vec<Vec<f64>>,
    /// Root mean (over histories) across-draw standard deviation of a
    /// single realized paired difference; `NaN` when `n_draws < 2`.
    pub draw_sd: Vec<Vec<f64>>,
    /// Mean over histories of the per-history `bands.0` quantile across
    /// draws.
    pub draw_lower: Vec<Vec<f64>>,
    /// Mean over histories of the per-history `bands.1` quantile across
    /// draws.
    pub draw_upper: Vec<Vec<f64>>,
    /// The state of each history at the shock date.
    pub history_states: Vec<usize>,
    /// The raw innovation vector `δ` added at the impact period for a
    /// history in each state, `[state][variable]`.
    pub shock_by_state: Vec<Vec<f64>>,
    /// Number of histories used.
    pub n_histories: usize,
    /// Draws per history.
    pub n_draws: usize,
    /// Independent draw units per history (`n_draws / 2` when antithetic).
    pub n_effective_draws: usize,
    /// Whether antithetic pairs were used.
    pub antithetic: bool,
    /// The horizon echoed back.
    pub horizon: usize,
    /// The band levels echoed back.
    pub bands: (f64, f64),
}

/// Per-history simulation result, cells flattened as `h * k + j`.
struct HistoryOut {
    mean: Vec<f64>,
    var_eff: Vec<f64>,
    var_draw: Vec<f64>,
    q_lo: Vec<f64>,
    q_hi: Vec<f64>,
    state0: usize,
}

/// Cholesky factors and impact shocks, one per state (flat `k × k`
/// lower-triangular, row-major).
struct States {
    chols: Vec<Vec<f64>>,
    shocks: Vec<Vec<f64>>,
}

/// Run the KPP simulation for `model` from every history in `histories`
/// (each `history_len() * n_vars()` values, rows oldest first).
///
/// # Errors
///
/// * [`VarError::InvalidArgument`] for an empty history set, a model with
///   zero variables / history length / states, `n_draws = 0`, an odd
///   `n_draws` under `antithetic`, a state index outside `0..n_states()`,
///   or a buffer beyond the engine's allocation guard;
/// * [`VarError::InvalidParameter`] for a shocked variable outside `0..k`,
///   a non-finite `size`, `horizon` above the guard, or `bands` outside
///   `0 <= lower < upper <= 1`;
/// * [`VarError::Dimension`] for a history or raw shock of the wrong length,
///   or a covariance that is not `k × k`;
/// * [`VarError::NonFinite`] for non-finite histories, covariances, raw
///   shocks, or simulated paths (an explosive model);
/// * [`VarError::NotPositiveDefinite`] for a state covariance without a
///   Cholesky factor.
pub fn girf<M: GirfModel>(
    model: &M,
    histories: &[Vec<f64>],
    opts: &GirfOptions,
) -> Result<Girf, VarError> {
    let k = model.n_vars();
    let len = model.history_len();
    let n_states = model.n_states();
    validate_model_dims(k, len, n_states)?;
    validate_histories(histories, k, len)?;
    validate_options(opts, k)?;

    let states = build_states(model, k, n_states, &opts.shock)?;

    let n_hist = histories.len();
    let n_draws = opts.n_draws;
    let n_streams = if opts.antithetic {
        n_draws / 2
    } else {
        n_draws
    };
    let cells = (opts.horizon + 1) * k;
    if n_draws.checked_mul(cells).is_none_or(|b| b > MAX_BUFFER)
        || n_hist.checked_mul(cells).is_none_or(|b| b > MAX_BUFFER)
    {
        return Err(VarError::InvalidArgument {
            what: "n_draws x (horizon + 1) x k (and n_histories x (horizon + 1) x k) must \
                   stay below 2^31 values; reduce n_draws, the horizon, or the number of \
                   histories",
        });
    }

    let mut root = SeedSequence::new(u128::from(opts.seed));
    let hist_seqs = root.spawn(n_hist).map_err(|_| spawn_error("histories"))?;

    // One task per history, sequential over its draws; results land at
    // their history index, so nothing depends on scheduling.
    let outs: Vec<Result<HistoryOut, VarError>> = histories
        .par_iter()
        .zip(hist_seqs.par_iter())
        .map(|(hist, seq)| simulate_history(model, hist, seq, &states, opts, n_streams))
        .collect();
    let mut results = Vec::with_capacity(n_hist);
    for out in outs {
        results.push(out?);
    }

    Ok(reduce(results, opts, k, n_draws, n_streams, states.shocks))
}

/// Every lag window of an `n × k` sample of length `history_len`: window
/// `t - history_len` holds rows `t - history_len .. t` (the shock then hits
/// row `t`), for `t = history_len .. n`.
pub fn sample_histories(endog: MatRef<'_, f64>, history_len: usize) -> Vec<Vec<f64>> {
    let (n, k) = (endog.nrows(), endog.ncols());
    if n < history_len + 1 || history_len == 0 {
        return Vec::new();
    }
    (history_len..n)
        .map(|t| {
            let mut h = Vec::with_capacity(history_len * k);
            for row in (t - history_len)..t {
                for j in 0..k {
                    h.push(endog[(row, j)]);
                }
            }
            h
        })
        .collect()
}

/// A seeded subsample of `m` indices out of `0..n` without replacement
/// (a partial Fisher-Yates shuffle driven by `Stream::new(seed +
/// HISTORY_SEED_OFFSET)`, uniform draws floored onto the remaining range),
/// returned in ascending order. Returns `0..n` when `m >= n`.
pub fn subsample_indices(n: usize, m: usize, seed: u64) -> Vec<usize> {
    if m >= n {
        return (0..n).collect();
    }
    let mut idx: Vec<usize> = (0..n).collect();
    let mut stream = Stream::new(seed.wrapping_add(HISTORY_SEED_OFFSET));
    for i in 0..m {
        let remaining = (n - i) as f64;
        let j = i + ((stream.uniform_f64() * remaining) as usize).min(n - i - 1);
        idx.swap(i, j);
    }
    let mut out = idx[..m].to_vec();
    out.sort_unstable();
    out
}

// ------------------------------------------------------------------ linear

/// The linear VAR(p) as a [`GirfModel`]: one state, innovation covariance
/// `sigma`, law of motion `y_t = c + Σ A_i y_{t-i} + u_t`. Its GIRF is the
/// closed-form impulse response (`Ψ_h P e_j` orthogonal, `Ψ_h Σ e_j /
/// sqrt(σ_jj)` generalized) for every draw and every history — the
/// engine's exactness check.
#[derive(Debug, Clone, PartialEq)]
pub struct LinearVarModel {
    k: usize,
    p: usize,
    /// `[lag][row][col]` flattened as `(lag * k + row) * k + col`.
    coefs: Vec<f64>,
    intercept: Vec<f64>,
    sigma: Vec<Vec<f64>>,
}

impl LinearVarModel {
    /// Build from lag matrices `coefs[i]` (`k × k`, rows = equations),
    /// an intercept (length `k`) and a covariance (`k × k`).
    ///
    /// # Errors
    ///
    /// [`VarError::InvalidArgument`] for no lags or `k = 0`;
    /// [`VarError::Dimension`] for mismatched shapes.
    pub fn new(
        coefs: &[Vec<Vec<f64>>],
        intercept: Vec<f64>,
        sigma: Vec<Vec<f64>>,
    ) -> Result<Self, VarError> {
        if coefs.is_empty() {
            return Err(VarError::InvalidArgument {
                what: "the linear GIRF model needs at least one lag matrix",
            });
        }
        let k = coefs[0].len();
        if k == 0 {
            return Err(VarError::InvalidArgument {
                what: "the linear GIRF model has 0 x 0 lag matrices; pass k >= 1 series",
            });
        }
        let mut flat = Vec::with_capacity(coefs.len() * k * k);
        for a in coefs {
            if a.len() != k {
                return Err(VarError::Dimension {
                    what: "every lag matrix must be k x k",
                    expected: k,
                    got: a.len(),
                });
            }
            for row in a {
                if row.len() != k {
                    return Err(VarError::Dimension {
                        what: "every lag matrix must be k x k",
                        expected: k,
                        got: row.len(),
                    });
                }
                flat.extend_from_slice(row);
            }
        }
        if intercept.len() != k {
            return Err(VarError::Dimension {
                what: "the intercept must have one entry per series",
                expected: k,
                got: intercept.len(),
            });
        }
        if sigma.len() != k || sigma.iter().any(|r| r.len() != k) {
            return Err(VarError::Dimension {
                what: "the innovation covariance must be k x k",
                expected: k,
                got: sigma.len(),
            });
        }
        Ok(Self {
            k,
            p: coefs.len(),
            coefs: flat,
            intercept,
            sigma,
        })
    }

    /// The fitted reduced form: `coefs`, `intercept`, and the df-adjusted
    /// `sigma_u` (the covariance `var_irf(orth=True)` factors).
    ///
    /// # Errors
    ///
    /// [`VarError::InvalidArgument`] for a VAR(0) (no dynamics to simulate).
    pub fn from_results(r: &VarResults) -> Result<Self, VarError> {
        if r.spec.lags == 0 {
            return Err(VarError::InvalidArgument {
                what: "a VAR(0) has no dynamics to simulate, so its generalized impulse \
                       response is the impact shock alone; refit with p >= 1",
            });
        }
        let k = r.neqs;
        let coefs: Vec<Vec<Vec<f64>>> = r
            .coefs
            .iter()
            .map(|a| {
                (0..k)
                    .map(|i| (0..k).map(|j| a[(i, j)]).collect())
                    .collect()
            })
            .collect();
        let sigma: Vec<Vec<f64>> = (0..k)
            .map(|i| (0..k).map(|j| r.sigma_u[(i, j)]).collect())
            .collect();
        Self::new(&coefs, r.intercept.clone(), sigma)
    }
}

impl GirfModel for LinearVarModel {
    fn n_vars(&self) -> usize {
        self.k
    }
    fn history_len(&self) -> usize {
        self.p
    }
    fn n_states(&self) -> usize {
        1
    }
    fn state(&self, _history: &[f64]) -> usize {
        0
    }
    fn covariance(&self, _state: usize) -> Vec<Vec<f64>> {
        self.sigma.clone()
    }
    fn step(&self, history: &[f64], innovation: &[f64], out: &mut [f64]) {
        let (k, p) = (self.k, self.p);
        for r in 0..k {
            let mut v = self.intercept[r] + innovation[r];
            for lag in 1..=p {
                let yrow = &history[(p - lag) * k..(p - lag + 1) * k];
                let arow = &self.coefs[((lag - 1) * k + r) * k..((lag - 1) * k + r + 1) * k];
                for c in 0..k {
                    v += arow[c] * yrow[c];
                }
            }
            out[r] = v;
        }
    }
}

/// GIRF of a fitted linear VAR from its own sample's lag windows (all of
/// them, or a seeded subsample of `histories` of them). A convenience over
/// [`girf`] + [`LinearVarModel::from_results`] + [`sample_histories`].
///
/// # Errors
///
/// Those of [`girf`] and [`LinearVarModel::from_results`], plus
/// [`VarError::InvalidArgument`] if the sample has no complete lag window
/// or `histories = Some(0)`.
pub fn var_girf(
    results: &VarResults,
    opts: &GirfOptions,
    histories: Option<usize>,
) -> Result<Girf, VarError> {
    let model = LinearVarModel::from_results(results)?;
    let all = sample_histories(results.endog.as_ref(), model.history_len());
    if all.is_empty() {
        return Err(VarError::InvalidArgument {
            what: "the sample holds no complete lag window to condition on; supply at \
                   least lags + 1 observations",
        });
    }
    let selected: Vec<Vec<f64>> = match histories {
        None => all,
        Some(0) => {
            return Err(VarError::InvalidArgument {
                what: "histories must be >= 1 (the number of lag windows to condition \
                       on) or None for every window in the sample",
            })
        }
        Some(m) => subsample_indices(all.len(), m, opts.seed)
            .into_iter()
            .map(|i| all[i].clone())
            .collect(),
    };
    girf(&model, &selected, opts)
}

// -------------------------------------------------------------- internals

fn validate_model_dims(k: usize, len: usize, n_states: usize) -> Result<(), VarError> {
    if k == 0 {
        return Err(VarError::InvalidArgument {
            what: "the GIRF model reports n_vars() = 0; a model needs at least one variable",
        });
    }
    if len == 0 {
        return Err(VarError::InvalidArgument {
            what: "the GIRF model reports history_len() = 0; a history must hold at least \
                   one past observation (p lags, or max(p, delay) for a threshold model)",
        });
    }
    if n_states == 0 {
        return Err(VarError::InvalidArgument {
            what: "the GIRF model reports n_states() = 0; a model needs at least one \
                   innovation covariance",
        });
    }
    Ok(())
}

fn validate_histories(histories: &[Vec<f64>], k: usize, len: usize) -> Result<(), VarError> {
    if histories.is_empty() {
        return Err(VarError::InvalidArgument {
            what: "no histories to condition on: pass at least one lag window (for a \
                   regime-conditional GIRF this means the chosen regime holds no \
                   observation in the sample — use regime = \"all\" or the other regime)",
        });
    }
    for h in histories {
        if h.len() != len * k {
            return Err(VarError::Dimension {
                what: "every history must hold history_len() x n_vars() values (rows \
                       oldest first)",
                expected: len * k,
                got: h.len(),
            });
        }
        if h.iter().any(|v| !v.is_finite()) {
            return Err(VarError::NonFinite {
                what: "a conditioning history (the sample's lag windows must be finite)",
                at: None,
            });
        }
    }
    Ok(())
}

fn validate_options(opts: &GirfOptions, k: usize) -> Result<(), VarError> {
    match &opts.shock {
        GirfShock::Orthogonal { var, size } | GirfShock::Generalized { var, size } => {
            if *var >= k {
                return Err(VarError::InvalidParameter {
                    name: "shock_var",
                    value: *var as f64,
                    requirement: "shock_var < n_series (the 0-based index of the variable \
                                  whose innovation is shocked)",
                });
            }
            if !size.is_finite() {
                return Err(VarError::InvalidParameter {
                    name: "size",
                    value: *size,
                    requirement: "a finite shock size in standard deviations of the \
                                  innovation (negative for an adverse shock)",
                });
            }
        }
        GirfShock::Raw(delta) => {
            if delta.len() != k {
                return Err(VarError::Dimension {
                    what: "a raw shock vector must have one entry per series",
                    expected: k,
                    got: delta.len(),
                });
            }
            if delta.iter().any(|v| !v.is_finite()) {
                return Err(VarError::NonFinite {
                    what: "the raw shock vector",
                    at: None,
                });
            }
        }
    }
    if opts.horizon > MAX_HORIZON {
        return Err(VarError::InvalidParameter {
            name: "horizon",
            value: opts.horizon as f64,
            requirement: "horizon <= 1_000_000 (responses are simulated period by period; \
                          20-40 is the usual range)",
        });
    }
    if opts.n_draws == 0 {
        return Err(VarError::InvalidParameter {
            name: "n_draws",
            value: 0.0,
            requirement: "n_draws >= 1 (future-innovation draws per history; a linear \
                          model needs 1, a nonlinear model a few hundred)",
        });
    }
    if opts.antithetic && opts.n_draws % 2 == 1 {
        return Err(VarError::InvalidParameter {
            name: "n_draws",
            value: opts.n_draws as f64,
            requirement: "an even n_draws when antithetic = true (draws come in (+z, -z) \
                          pairs); pass an even count or antithetic = false",
        });
    }
    let (lo, hi) = opts.bands;
    if !(lo.is_finite() && hi.is_finite() && lo >= 0.0 && lo < hi && hi <= 1.0) {
        return Err(VarError::InvalidParameter {
            name: "bands",
            value: lo,
            requirement: "0 <= lower < upper <= 1 for the (lower, upper) quantile levels \
                          across histories and draws, e.g. (0.16, 0.84)",
        });
    }
    Ok(())
}

fn build_states<M: GirfModel>(
    model: &M,
    k: usize,
    n_states: usize,
    shock: &GirfShock,
) -> Result<States, VarError> {
    let mut chols = Vec::with_capacity(n_states);
    let mut shocks = Vec::with_capacity(n_states);
    for s in 0..n_states {
        let sigma = model.covariance(s);
        if sigma.len() != k || sigma.iter().any(|r| r.len() != k) {
            return Err(VarError::Dimension {
                what: "an innovation covariance handed to the GIRF engine must be k x k",
                expected: k,
                got: sigma.len(),
            });
        }
        if sigma.iter().flatten().any(|v| !v.is_finite()) {
            return Err(VarError::NonFinite {
                what: "an innovation covariance handed to the GIRF engine",
                at: None,
            });
        }
        let sym = Mat::from_fn(k, k, |i, j| 0.5 * (sigma[i][j] + sigma[j][i]));
        let l = chol_lower(
            sym.as_ref(),
            "an innovation covariance handed to the GIRF engine (a regime's residual \
             covariance)",
        )?;
        let mut flat = vec![0.0f64; k * k];
        for i in 0..k {
            for j in 0..=i {
                flat[i * k + j] = l[(i, j)];
            }
        }
        let delta: Vec<f64> = match shock {
            GirfShock::Orthogonal { var, size } => (0..k).map(|i| size * l[(i, *var)]).collect(),
            GirfShock::Generalized { var, size } => {
                let sd = sym[(*var, *var)].sqrt();
                (0..k).map(|i| size * sym[(i, *var)] / sd).collect()
            }
            GirfShock::Raw(d) => d.clone(),
        };
        chols.push(flat);
        shocks.push(delta);
    }
    Ok(States { chols, shocks })
}

fn spawn_error(what: &'static str) -> VarError {
    let _ = what;
    VarError::InvalidArgument {
        what: "the GIRF substream tree exceeds the SeedSequence spawn limit (2^32 children \
               per level); use fewer histories or draws",
    }
}

/// Simulate every draw of one history: paired paths in lockstep, the
/// per-cell mean / variances / quantiles of the paired differences.
fn simulate_history<M: GirfModel>(
    model: &M,
    hist: &[f64],
    seq: &SeedSequence,
    states: &States,
    opts: &GirfOptions,
    n_streams: usize,
) -> Result<HistoryOut, VarError> {
    let k = model.n_vars();
    let len = model.history_len();
    let n_states = model.n_states();
    let hh = opts.horizon + 1;
    let cells = hh * k;
    let n_draws = opts.n_draws;

    let state0 = model.state(hist);
    if state0 >= n_states {
        return Err(state_error());
    }
    let shock0 = &states.shocks[state0];

    let mut seq = seq.clone();
    let draw_seqs = seq.spawn(n_streams).map_err(|_| spawn_error("draws"))?;

    // Draw values, cell-major: vals[cell * n_draws + r].
    let mut vals = vec![0.0f64; cells * n_draws];
    // Effective (independent) draw values for the MC standard error: the
    // pair means under antithetic sampling, else the draws themselves.
    let mut eff = vec![0.0f64; cells * n_streams];
    let mut eps = vec![0.0f64; cells];
    let mut win_a = vec![0.0f64; len * k];
    let mut win_b = vec![0.0f64; len * k];
    let mut u = vec![0.0f64; k];
    let mut ya = vec![0.0f64; k];
    let mut yb = vec![0.0f64; k];
    let mut diff = vec![0.0f64; cells];

    let mut r = 0usize;
    for (s_idx, dseq) in draw_seqs.iter().enumerate() {
        let mut stream = Stream::from_seed_sequence(dseq);
        for e in eps.iter_mut() {
            *e = WildWeights::Normal.draw(&mut stream);
        }
        let signs: &[f64] = if opts.antithetic {
            &[1.0, -1.0]
        } else {
            &[1.0]
        };
        for &sign in signs {
            simulate_pair(
                model, hist, states, shock0, &eps, sign, &mut win_a, &mut win_b, &mut u, &mut ya,
                &mut yb, &mut diff,
            )?;
            for (cell, &d) in diff.iter().enumerate() {
                vals[cell * n_draws + r] = d;
                eff[cell * n_streams + s_idx] += d / signs.len() as f64;
            }
            r += 1;
        }
    }

    let mut mean = vec![0.0f64; cells];
    let mut var_eff = vec![f64::NAN; cells];
    let mut var_draw = vec![f64::NAN; cells];
    let mut q_lo = vec![0.0f64; cells];
    let mut q_hi = vec![0.0f64; cells];
    for cell in 0..cells {
        let v = &mut vals[cell * n_draws..(cell + 1) * n_draws];
        let m = v.iter().sum::<f64>() / n_draws as f64;
        mean[cell] = m;
        if n_draws >= 2 {
            var_draw[cell] =
                v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (n_draws as f64 - 1.0);
        }
        let e = &eff[cell * n_streams..(cell + 1) * n_streams];
        if n_streams >= 2 {
            let me = e.iter().sum::<f64>() / n_streams as f64;
            var_eff[cell] =
                e.iter().map(|x| (x - me) * (x - me)).sum::<f64>() / (n_streams as f64 - 1.0);
        }
        v.sort_by(f64::total_cmp);
        q_lo[cell] = percentile_sorted(v, opts.bands.0);
        q_hi[cell] = percentile_sorted(v, opts.bands.1);
    }
    Ok(HistoryOut {
        mean,
        var_eff,
        var_draw,
        q_lo,
        q_hi,
        state0,
    })
}

fn state_error() -> VarError {
    VarError::InvalidArgument {
        what: "the GIRF model's state() returned an index >= n_states(); every state must \
               have a covariance",
    }
}

/// The two paths of one draw in lockstep; `diff[h * k + j] = y^with - y^without`.
#[allow(clippy::too_many_arguments)]
fn simulate_pair<M: GirfModel>(
    model: &M,
    hist: &[f64],
    states: &States,
    shock0: &[f64],
    eps: &[f64],
    sign: f64,
    win_a: &mut [f64],
    win_b: &mut [f64],
    u: &mut [f64],
    ya: &mut [f64],
    yb: &mut [f64],
    diff: &mut [f64],
) -> Result<(), VarError> {
    let k = model.n_vars();
    let len = win_a.len() / k;
    let n_states = model.n_states();
    let hh = diff.len() / k;
    win_a.copy_from_slice(hist);
    win_b.copy_from_slice(hist);
    for h in 0..hh {
        let z = &eps[h * k..(h + 1) * k];
        // Shocked path.
        let sa = model.state(win_a);
        if sa >= n_states {
            return Err(state_error());
        }
        scale_innovation(&states.chols[sa], z, sign, k, u);
        if h == 0 {
            for j in 0..k {
                u[j] += shock0[j];
            }
        }
        model.step(win_a, u, ya);
        // Baseline path.
        let sb = model.state(win_b);
        if sb >= n_states {
            return Err(state_error());
        }
        scale_innovation(&states.chols[sb], z, sign, k, u);
        model.step(win_b, u, yb);
        for j in 0..k {
            let d = ya[j] - yb[j];
            if !d.is_finite() {
                return Err(VarError::NonFinite {
                    what: "the simulated GIRF paths: the model is explosive from at least \
                           one history (a regime with a root outside the unit circle) or \
                           the shock drives it to overflow; check the fit's stability or \
                           shorten the horizon",
                    at: None,
                });
            }
            diff[h * k + j] = d;
        }
        if h + 1 < hh {
            shift_window(win_a, ya, len, k);
            shift_window(win_b, yb, len, k);
        }
    }
    Ok(())
}

/// `u = sign * L z` for a flat lower-triangular `L`.
#[inline]
fn scale_innovation(l: &[f64], z: &[f64], sign: f64, k: usize, u: &mut [f64]) {
    for r in 0..k {
        let row = &l[r * k..r * k + r + 1];
        let mut v = 0.0;
        for (c, &lrc) in row.iter().enumerate() {
            v += lrc * z[c];
        }
        u[r] = sign * v;
    }
}

/// Drop the oldest row of the window and append `y`.
#[inline]
fn shift_window(win: &mut [f64], y: &[f64], len: usize, k: usize) {
    if len > 1 {
        win.copy_within(k.., 0);
    }
    win[(len - 1) * k..].copy_from_slice(y);
}

/// The history-order reductions (sequential, so bit-identical at any
/// thread count).
fn reduce(
    results: Vec<HistoryOut>,
    opts: &GirfOptions,
    k: usize,
    n_draws: usize,
    n_streams: usize,
    shocks: Vec<Vec<f64>>,
) -> Girf {
    let n_hist = results.len();
    let nh = n_hist as f64;
    let hh = opts.horizon + 1;
    let cells = hh * k;
    let unflatten =
        |v: &[f64]| -> Vec<Vec<f64>> { (0..hh).map(|h| v[h * k..(h + 1) * k].to_vec()).collect() };

    let mut girf = vec![0.0f64; cells];
    let mut mc_se = vec![0.0f64; cells];
    let mut draw_sd = vec![0.0f64; cells];
    let mut draw_lower = vec![0.0f64; cells];
    let mut draw_upper = vec![0.0f64; cells];
    for r in &results {
        for cell in 0..cells {
            girf[cell] += r.mean[cell];
            mc_se[cell] += r.var_eff[cell];
            draw_sd[cell] += r.var_draw[cell];
            draw_lower[cell] += r.q_lo[cell];
            draw_upper[cell] += r.q_hi[cell];
        }
    }
    for cell in 0..cells {
        girf[cell] /= nh;
        mc_se[cell] = if n_streams >= 2 {
            (mc_se[cell] / n_streams as f64).sqrt() / nh
        } else {
            f64::NAN
        };
        draw_sd[cell] = if n_draws >= 2 {
            (draw_sd[cell] / nh).sqrt()
        } else {
            f64::NAN
        };
        draw_lower[cell] /= nh;
        draw_upper[cell] /= nh;
    }

    let mut lower = vec![0.0f64; cells];
    let mut upper = vec![0.0f64; cells];
    let mut col = vec![0.0f64; n_hist];
    for cell in 0..cells {
        for (i, r) in results.iter().enumerate() {
            col[i] = r.mean[cell];
        }
        col.sort_by(f64::total_cmp);
        lower[cell] = percentile_sorted(&col, opts.bands.0);
        upper[cell] = percentile_sorted(&col, opts.bands.1);
    }

    let history_states: Vec<usize> = results.iter().map(|r| r.state0).collect();
    let per_history: Vec<Vec<Vec<f64>>> = results.iter().map(|r| unflatten(&r.mean)).collect();

    Girf {
        girf: unflatten(&girf),
        lower: unflatten(&lower),
        upper: unflatten(&upper),
        per_history,
        mc_se: unflatten(&mc_se),
        draw_sd: unflatten(&draw_sd),
        draw_lower: unflatten(&draw_lower),
        draw_upper: unflatten(&draw_upper),
        history_states,
        shock_by_state: shocks,
        n_histories: n_hist,
        n_draws,
        n_effective_draws: n_streams,
        antithetic: opts.antithetic,
        horizon: opts.horizon,
        bands: opts.bands,
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn subsample_is_sorted_unique_and_seeded() {
        let a = subsample_indices(50, 10, 3);
        let b = subsample_indices(50, 10, 3);
        assert_eq!(a, b);
        assert_eq!(a.len(), 10);
        assert!(a.windows(2).all(|w| w[0] < w[1]));
        assert!(a.iter().all(|&i| i < 50));
        assert_eq!(subsample_indices(5, 9, 1), vec![0, 1, 2, 3, 4]);
        assert_ne!(subsample_indices(50, 10, 4), a);
    }

    #[test]
    fn sample_histories_are_the_lag_windows() {
        let m = Mat::from_fn(5, 2, |i, j| (i * 10 + j) as f64);
        let h = sample_histories(m.as_ref(), 2);
        assert_eq!(h.len(), 3);
        assert_eq!(h[0], vec![0.0, 1.0, 10.0, 11.0]);
        assert_eq!(h[2], vec![20.0, 21.0, 30.0, 31.0]);
        assert!(sample_histories(m.as_ref(), 5).is_empty());
    }

    #[test]
    fn shift_window_drops_oldest_row() {
        let mut w = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        shift_window(&mut w, &[7.0, 8.0], 3, 2);
        assert_eq!(w, vec![3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let mut w1 = vec![1.0, 2.0];
        shift_window(&mut w1, &[9.0, 9.5], 1, 2);
        assert_eq!(w1, vec![9.0, 9.5]);
    }
}
