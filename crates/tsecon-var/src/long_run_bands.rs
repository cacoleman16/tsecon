//! Confidence bands for **Blanchard-Quah long-run SVAR** impulse responses:
//! a residual bootstrap that re-identifies inside every draw (with optional
//! Kilian bias correction and simultaneous bands), and a delta-method arm
//! built on an analytic Jacobian of the Blanchard-Quah map.
//!
//! [`tsecon_ident::long_run_svar`] is a closed-form point estimator. Its
//! sampling uncertainty comes from two places, and a band has to carry both:
//! the reduced-form dynamics `A_1..A_p` (which drive the moving-average
//! weights *and* the long-run multiplier `C(1)`), and `Sigma_u`. The
//! identification step is therefore re-run on every bootstrap pseudo-sample —
//! re-identifying is not optional: holding `B` at its full-sample value would
//! drop exactly the `C(1)` uncertainty that Faust and Leeper (1997) show
//! dominates long-run schemes.
//!
//! # The bootstrap arm ([`long_run_bootstrap_bands`])
//!
//! The Runkle (1987) / Kilian (1998) residual recursive design, through the
//! same engine as [`crate::bootstrap_irf_bands`] (so the same `(seed,
//! n_boot)` gives the same pseudo-samples):
//!
//! 1. fit the VAR on the full sample; centre its residuals; keep the first
//!    `p` observed rows as initial conditions;
//! 2. with `bias_correct`, estimate Kilian's bias by an inner bootstrap (seed
//!    `seed + 0x9E3779B97F4A7C15`, `n_boot` reps), shrunk by repeated
//!    `delta -= 0.01` until `A_hat - delta * bias` is stable; the outer
//!    pseudo-samples are then generated from the corrected coefficients and
//!    the point estimate is the Blanchard-Quah map at the corrected
//!    coefficients and the full-sample `Sigma_u` (the intercept is not
//!    corrected, so the bootstrap DGP's implied mean moves with the slopes);
//! 3. replication `b` draws `T_eff` iid residual rows from substream `b` of
//!    `SeedSequence(seed).spawn(n_boot)`, rebuilds the pseudo-sample
//!    recursively from the observed first `p` rows (no burn-in), refits, and
//!    re-runs the Blanchard-Quah map with the same `restrictions`,
//!    `normalize` and `impact_signs` (Kilian-corrected coefficients when
//!    `bias_correct`, with the refit's own `Sigma_u`); responses are cumulated
//!    per draw when `cumulative`;
//! 4. `se` is the across-draw SD (divisor `n_boot - 1`), `lower`/`upper` the
//!    Efron percentiles at `alpha/2` and `1 - alpha/2` (NumPy "linear").
//!
//! **Self-test.** The imposed long-run zeros are zero in every draw, so their
//! bands have exactly zero width and `long_run_se == 0` there. A
//! non-degenerate imposed cell would prove the identification was hoisted out
//! of the loop.
//!
//! # The delta-method arm ([`long_run_asymptotic_bands`])
//!
//! Lütkepohl's (2005) delta method with the statsmodels conventions the VAR
//! crate already uses (`Sigma_alpha = (Z'Z)^{-1} ⊗ Sigma_u` on the lag block,
//! `Sigma_sigma = 2 D^+ (Sigma_u ⊗ Sigma_u) D^{+'} / T`, `T = nobs`). The only
//! new algebra is `∂ vec B / ∂ (alpha, vech Sigma_u)`, by forward mode:
//!
//! ```text
//! dD  = -sum_i dA_i,          dC1 = -C1 dD C1,
//! dM  = dC1 S C1' + C1 dS C1' + C1 S dC1',
//! dL  = L Phi(L^{-1} dM L^{-T})   (Phi: strict lower part + half the diagonal),
//! dLR = dL with LR's column permutation and signs,   dB = dD LR + D dLR,
//! ```
//!
//! with the column signs (from `normalize` and `impact_signs`) held at the
//! point estimate's. Then `∂ vec Theta_h = (B' ⊗ I) G_h dalpha + (I ⊗ Psi_h)
//! ∂ vec B` (cumulated when `cumulative`), and `se = sqrt(diag(J V J'))`.
//! The long-run matrix gets `∂ vec LR` the same way; its imposed zeros have
//! an exactly-zero Jacobian row, hence zero width.
//!
//! # Memory budget
//!
//! Every allocation that scales with a user count is sized before it is made
//! and refused beyond [`LONG_RUN_BANDS_BUDGET_VALUES`] (`2^28` doubles,
//! 2 GiB, the shared [`crate::budget::BUDGET_VALUES`]), with checked
//! multiplies so absurd counts cannot overflow. The count is of real
//! storage, not of cells ([`crate::budget`]): a `k x k` faer matrix costs
//! `mf = 8 ceil(k / 8) k + 16` doubles and its Python nested list
//! `4 k^2 + (k + 1)(k + 10)`. With `H1 = horizon + 1`, `cells = H1 k^2` and
//! `K` the band family size (`H1`, `k H1` or `cells` for `band_scope`
//! `"horizon"`, `"shock"`, `"all"`):
//!
//! * bootstrap: each replication keeps its responses, its long-run matrix,
//!   its `impact_signs` cells and its companion radius in its slot of ONE
//!   store of `n_boot * (cells + k^2 + n_anchors + 1)` values (not `H1 + 1`
//!   padded matrices, and no allocation of its own), plus a fixed
//!   per-replication overhead for its RNG substream, its result slots and
//!   its slot lock (`DRAW_OVERHEAD_VALUES`, 48) and one value of the
//!   percentile buffer; plus its
//!   inner estimate (`lags` matrices) with `bias_correct`, its copy of the
//!   band family (`K + 1`) with a simultaneous band, and `cells + k^2` when
//!   the draws are kept (the returned flat arrays);
//! * delta method: `cells * (3 dim + 2 k(k+1)/2 + 8) + H1 * 8 ceil(k^2 / 8)
//!   dim + dim^2` (`dim = lags k^2`: the per-cell loading rows, the `G_h`
//!   Jacobians and the coefficient covariance), plus `3 K^2` and the family
//!   cell lists for a simultaneous band (the family's joint covariance and
//!   its factorization, `((horizon + 1) k^2)^2` under `band_scope="all"`)
//!   and `(K + 1) band_n_sim` uniforms and maxima for sup-t. The sup-t
//!   simulation work `n_families * band_n_sim * K^2` is capped at
//!   [`LONG_RUN_BANDS_SUPT_WORK`] (`2^36`), and the Jacobians `G_h =
//!   sum_{m<h} (A')^{h-1-m} ⊗ Phi_m`, whose work `horizon (horizon + 1) / 2
//!   * k^2 * dim` is QUADRATIC in `horizon`, at [`LONG_RUN_BANDS_DELTA_WORK`]
//!   (`2^28`) — a horizon the storage count admits could otherwise run for
//!   hours;
//! * both arms: [`crate::budget::CUBE_COPIES`] copies of the reported cubes
//!   (`point`, `se`, `lower`, `upper`, the simultaneous pair) as matrices and
//!   Python lists.
//!
//! A refusal names the term that is over: `n_boot` (with the largest
//! replication count that fits at this horizon) or `horizon` on the
//! bootstrap arm; on the delta arm `horizon and lags` for the pointwise
//! working set, `horizon and band_scope` for a family's covariance,
//! `band_n_sim` for the sup-t simulation and `horizon` for the Jacobian
//! work, each with the largest value that fits.
//!
//! # References
//!
//! Blanchard and Quah (1989); Runkle (1987); Kilian (1998); Lütkepohl (2005,
//! ch. 3.7 and app. D); Faust and Leeper (1997); Montiel Olea and
//! Plagborg-Møller (2019) for the sup-t band.

use std::collections::VecDeque;

use tsecon_linalg::faer::{Mat, MatRef};

use crate::error::VarError;
use crate::irf_asymptotic::{
    apply_critical_values, critical_values_from_family_covs, delta_refusal, delta_workspace,
    duplication_matrix, family_dims, kron, pinv_full_col_rank, sup_t_terms, BandMethod, DeltaTerms,
    IrfBandScope, IrfCriticalValues,
};
use crate::irf_bootstrap::{
    bootstrap_refusal, draw_critical_values, fill_slot, spectral_radius_of, summarize, write_block,
    Cube, DrawStore, ResidualBootstrap, DRAW_OVERHEAD_VALUES,
};
use crate::results::VarResults;
use crate::spec::Trend;

/// The working-set budget of one call, in `f64` values (`2^28`, i.e. 2 GiB):
/// the shared [`crate::budget::BUDGET_VALUES`].
pub const LONG_RUN_BANDS_BUDGET_VALUES: usize = crate::budget::BUDGET_VALUES;

/// The simulation-work cap of the delta-method sup-t band:
/// `n_families * band_n_sim * K^2` multiply-adds (`2^36`), the shared
/// [`tsecon_stats::simultaneous::SUP_T_MAX_WORK`].
pub const LONG_RUN_BANDS_SUPT_WORK: u128 = tsecon_stats::simultaneous::SUP_T_MAX_WORK;

/// The work cap of the delta-method Jacobians `G_1..G_horizon`:
/// `horizon (horizon + 1) / 2 * k^2 * lags k^2` multiply-adds (`2^28`), the
/// shared [`crate::irf_asymptotic::DELTA_JACOBIAN_WORK`].
///
/// The `G_h` sums are quadratic in `horizon`, so the storage budget alone
/// admits horizons that run for hours. Measured on the Blanchard-Quah VAR(8)
/// (`k = 2`): `horizon = 2000` (just under the cap) takes about 3.5 s,
/// `horizon = 8000` about 90 s, and `horizon = 30000` over two minutes.
/// Blanchard-Quah horizons are tens of periods.
pub const LONG_RUN_BANDS_DELTA_WORK: u128 = crate::irf_asymptotic::DELTA_JACOBIAN_WORK;

/// What to estimate and how to band it. Field defaults mirror the Python
/// `long_run_svar_bands` signature.
#[derive(Debug, Clone, PartialEq)]
pub struct LongRunBandSpec {
    /// VAR lag order `p` (at least 1).
    pub lags: usize,
    /// Deterministic term of the reduced form.
    pub trend: Trend,
    /// Last impulse-response horizon `H` (`H + 1` horizons are reported).
    pub horizon: usize,
    /// Long-run zero pattern `(variable, shock)`; `None` is the classic
    /// recursive scheme. Must be a column permutation of it.
    pub long_run_zeros: Option<Vec<(usize, usize)>>,
    /// Force a positive impact diagonal instead of a positive long-run one.
    pub normalize_impact: bool,
    /// Per-shock sign anchors `(variable, shock)`, applied after the
    /// normalization (see [`tsecon_ident::apply_impact_signs`]).
    pub impact_signs: Option<Vec<(usize, usize)>>,
    /// Band the cumulated responses (cumulated per draw before percentiles).
    pub cumulative: bool,
    /// Two-sided level: `alpha = 0.1` gives 90% bands.
    pub alpha: f64,
    /// Bootstrap replications (bootstrap arm only).
    pub n_boot: usize,
    /// Resampling seed (bootstrap arm only).
    pub seed: u64,
    /// Kilian (1998) bias correction (bootstrap arm only).
    pub bias_correct: bool,
    /// A simultaneous band over a declared cell family, or `None` for the
    /// pointwise band alone.
    pub band: Option<(BandMethod, IrfBandScope)>,
    /// Seed of the delta-method sup-t simulation (asymptotic arm only).
    pub band_seed: u64,
    /// Draws of the delta-method sup-t simulation (asymptotic arm only).
    pub band_n_sim: usize,
    /// Keep the per-draw responses and long-run matrices (bootstrap arm).
    pub keep_draws: bool,
}

impl LongRunBandSpec {
    /// The Python defaults: constant, `horizon`, recursive scheme, long-run
    /// normalization, no anchors, plain responses, `alpha = 0.1`,
    /// `n_boot = 1000`, `seed = 0`, no bias correction, pointwise only,
    /// `band_seed = 20260807`, `band_n_sim = 100_000`, no draws kept.
    pub fn new(lags: usize, horizon: usize) -> Self {
        LongRunBandSpec {
            lags,
            trend: Trend::Constant,
            horizon,
            long_run_zeros: None,
            normalize_impact: false,
            impact_signs: None,
            cumulative: false,
            alpha: 0.1,
            n_boot: 1000,
            seed: 0,
            bias_correct: false,
            band: None,
            band_seed: 20_260_807,
            band_n_sim: 100_000,
            keep_draws: false,
        }
    }
}

/// Banded Blanchard-Quah responses. Impulse-response cubes have `horizon + 1`
/// entries, each `k x k` with `(i, j)` the response of variable `i` to
/// structural shock `j`; long-run matrices are `k x k` on `LR = C(1) B`.
#[derive(Debug, Clone)]
pub struct LongRunBands {
    /// The structural responses (cumulated when `cumulative`): bit-identical
    /// to [`tsecon_ident::long_run_svar`]'s `irf` (plus the anchors) without
    /// bias correction; the map at the Kilian-corrected coefficients and the
    /// full-sample `Sigma_u` with it.
    pub point: Vec<Mat<f64>>,
    /// Bootstrap SD (divisor `n_boot - 1`) or delta-method standard error.
    pub se: Vec<Mat<f64>>,
    /// Efron `alpha/2` percentile (bootstrap) or `point - z se` (delta).
    pub lower: Vec<Mat<f64>>,
    /// Efron `1 - alpha/2` percentile (bootstrap) or `point + z se` (delta).
    pub upper: Vec<Mat<f64>>,
    /// The impact matrix `B` behind `point`.
    pub impact_point: Mat<f64>,
    /// The long-run matrix `LR = C(1) B` behind `point`.
    pub long_run_point: Mat<f64>,
    /// Standard errors of `LR` (exactly zero on imposed zeros).
    pub long_run_se: Mat<f64>,
    /// Lower band of `LR`.
    pub long_run_lower: Mat<f64>,
    /// Upper band of `LR`.
    pub long_run_upper: Mat<f64>,
    /// Smallest modulus of the roots of `det(I - A_1 z - ... - A_p z^p)` at
    /// the full-sample fit (the `var_fit` convention: stable iff `> 1`; it is
    /// `1 / companion spectral radius`, the INVERSE of the radius that
    /// [`Self::n_unstable_draws`] thresholds).
    pub min_root: f64,
    /// Raw bootstrap refits (before any Kilian correction) whose companion
    /// spectral radius is `>= 1`; `None` on the delta arm.
    pub n_unstable_draws: Option<usize>,
    /// Per `impact_signs` entry: draws whose anchor decided differently
    /// (pre-anchor cell negative vs non-negative) from the point estimate's.
    /// `None` without anchors or on the delta arm.
    pub sign_anchor_mismatches: Option<Vec<usize>>,
    /// Replications used (`None` on the delta arm).
    pub n_boot: Option<usize>,
    /// The two-sided level, echoed.
    pub alpha: f64,
    /// Whether the Kilian correction was applied.
    pub bias_correct: bool,
    /// Whether the responses are cumulated.
    pub cumulative: bool,
    /// Simultaneous-band multipliers, when a band was requested.
    pub simultaneous: Option<IrfCriticalValues>,
    /// `point - c se` of the simultaneous band.
    pub sim_lower: Option<Vec<Mat<f64>>>,
    /// `point + c se` of the simultaneous band.
    pub sim_upper: Option<Vec<Mat<f64>>>,
    /// Per-draw reported responses (cumulated iff `cumulative`,
    /// bias-corrected iff `bias_correct`), when `keep_draws`: one flat array
    /// in row-major `(n_boot, horizon + 1, k, k)` order, cell `(b, h, i, j)`
    /// at `((b (horizon + 1) + h) k + i) k + j`.
    pub irf_draws: Option<Vec<f64>>,
    /// Per-draw long-run matrices, when `keep_draws`: row-major
    /// `(n_boot, k, k)`.
    pub long_run_draws: Option<Vec<f64>>,
}

// ---------------------------------------------------------------------------
// Shared pieces
// ---------------------------------------------------------------------------

/// One identified model: the impact and long-run matrices after
/// normalization and anchoring, and the pre-anchor anchored cells.
struct Identified {
    impact: Mat<f64>,
    long_run: Mat<f64>,
    multiplier: Mat<f64>,
    pre_anchor: Vec<f64>,
}

fn identify(
    coefs: &[Mat<f64>],
    sigma_u: MatRef<'_, f64>,
    spec: &LongRunBandSpec,
) -> Result<Identified, tsecon_ident::IdentError> {
    let refs: Vec<MatRef<'_, f64>> = coefs.iter().map(Mat::as_ref).collect();
    let mut r = tsecon_ident::long_run_svar(
        &refs,
        sigma_u,
        0,
        spec.long_run_zeros.as_deref(),
        spec.normalize_impact,
    )?;
    let pre_anchor = match spec.impact_signs.as_deref() {
        Some(a) => tsecon_ident::apply_impact_signs(&mut r, a)?,
        None => Vec::new(),
    };
    Ok(Identified {
        impact: r.impact,
        long_run: r.long_run,
        multiplier: r.long_run_multiplier,
        pre_anchor,
    })
}

/// [`theta_cube`] written straight into blocks `0..=horizon` of `out`
/// (row-major `k x k` blocks, the [`DrawStore`] layout), bit for bit: the
/// same recursion and products, with each `Theta_h` copied out as it is
/// formed instead of kept as a matrix, so a bootstrap replication never holds
/// a horizon-long cube of matrices.
fn theta_into(coefs: &[Mat<f64>], b: &Mat<f64>, horizon: usize, cumulative: bool, out: &mut [f64]) {
    let k = b.nrows();
    let k2 = k * k;
    let p = coefs.len();
    let mut window: VecDeque<Mat<f64>> = VecDeque::with_capacity(p + 1);
    let psi0 = Mat::<f64>::identity(k, k);
    write_block(out, 0, (psi0.as_ref() * b.as_ref()).as_ref());
    window.push_back(psi0);
    for h in 1..=horizon {
        let mut acc = Mat::<f64>::zeros(k, k);
        for i in 1..=h.min(p) {
            acc += window[window.len() - i].as_ref() * coefs[i - 1].as_ref();
        }
        write_block(out, h, (acc.as_ref() * b.as_ref()).as_ref());
        window.push_back(acc);
        if window.len() > p {
            window.pop_front();
        }
    }
    if cumulative {
        for c in k2..(horizon + 1) * k2 {
            out[c] += out[c - k2];
        }
    }
}

/// `Theta_h = Psi_h B` for `h = 0..=horizon`, cumulated when asked.
///
/// The `Psi` recursion is [`tsecon_ident::long_run_svar`]'s, operation for
/// operation (`acc += Psi_{h-i} A_i` for `i = 1..=min(h, p)`, then
/// `Psi_h * B`), so the result is bit-identical to its `irf`; it keeps only
/// the last `p` `Psi` matrices instead of all `horizon + 1`, so a draw's only
/// horizon-sized allocation is the cube it returns. The cumulation is the
/// `cumulative_irf` running sum.
fn theta_cube(coefs: &[Mat<f64>], b: &Mat<f64>, horizon: usize, cumulative: bool) -> Cube {
    let k = b.nrows();
    let p = coefs.len();
    let mut out: Cube = Vec::with_capacity(horizon + 1);
    let mut window: VecDeque<Mat<f64>> = VecDeque::with_capacity(p + 1);
    let psi0 = Mat::<f64>::identity(k, k);
    out.push(psi0.as_ref() * b.as_ref());
    window.push_back(psi0);
    for h in 1..=horizon {
        let mut acc = Mat::<f64>::zeros(k, k);
        for i in 1..=h.min(p) {
            acc += window[window.len() - i].as_ref() * coefs[i - 1].as_ref();
        }
        out.push(acc.as_ref() * b.as_ref());
        window.push_back(acc);
        if window.len() > p {
            window.pop_front();
        }
    }
    if cumulative {
        for h in 1..out.len() {
            let prev = out[h - 1].clone();
            let cur = &out[h] + &prev;
            out[h] = cur;
        }
    }
    out
}

/// The static text of an identification failure on a bootstrap draw.
fn ident_draw_message(e: &tsecon_ident::IdentError) -> &'static str {
    match e {
        tsecon_ident::IdentError::InvalidArgument { what } => what,
        _ => {
            "the re-estimated VAR could not be Blanchard-Quah identified (a non-finite \
             coefficient or residual covariance)"
        }
    }
}

/// Full-sample identification failures are the caller's data problem.
fn full_sample_error(coefs: &[Mat<f64>], e: &tsecon_ident::IdentError) -> VarError {
    let refs: Vec<MatRef<'_, f64>> = coefs.iter().map(Mat::as_ref).collect();
    if tsecon_ident::long_run_multiplier(&refs).is_err() {
        return VarError::InvalidArgument {
            what: "data: the VAR fitted at these lags has a singular long-run matrix \
                   I - A_1 - ... - A_p (a unit root at frequency zero), so C(1) does not \
                   exist and a long-run restriction identifies nothing; difference the \
                   I(1) series (Blanchard-Quah use output GROWTH) or change lags",
        };
    }
    match e {
        tsecon_ident::IdentError::InvalidArgument { what } => VarError::InvalidArgument { what },
        tsecon_ident::IdentError::RestrictionOutOfRange { what, index, bound } => {
            VarError::Dimension {
                what,
                expected: *bound,
                got: *index,
            }
        }
        _ => VarError::InvalidArgument {
            what: "data: the full-sample VAR could not be Blanchard-Quah identified \
                   (non-finite coefficients or residual covariance)",
        },
    }
}

/// Validation shared by both arms; returns `k`.
fn validate(spec: &LongRunBandSpec, k: usize) -> Result<(), VarError> {
    if spec.lags == 0 {
        return Err(VarError::InvalidArgument {
            what: "lags must be at least 1: a VAR(0) has no dynamics, hence no long-run \
                   multiplier to identify from",
        });
    }
    if !(spec.alpha > 0.0 && spec.alpha < 1.0) {
        return Err(VarError::InvalidParameter {
            name: "alpha",
            value: spec.alpha,
            requirement: "a value strictly inside (0, 1) — alpha = 0.1 gives 90% bands",
        });
    }
    if k < 2 {
        return Err(VarError::InvalidArgument {
            what: "data: long-run identification needs at least 2 variables (columns)",
        });
    }
    tsecon_ident::long_run_ordering(spec.long_run_zeros.as_deref(), k).map_err(|e| match e {
        tsecon_ident::IdentError::RestrictionOutOfRange { index, bound, .. } => {
            VarError::Dimension {
                what: "restrictions: a long-run zero (variable, shock) index must be below \
                       the number of variables k",
                expected: bound,
                got: index,
            }
        }
        tsecon_ident::IdentError::InvalidArgument { what } => VarError::InvalidArgument { what },
        _ => VarError::InvalidArgument {
            what: "restrictions is not a valid long-run zero pattern",
        },
    })?;
    if let Some(a) = spec.impact_signs.as_deref() {
        tsecon_ident::validate_impact_signs(a, k).map_err(|e| match e {
            tsecon_ident::IdentError::RestrictionOutOfRange { index, bound, .. } => {
                VarError::Dimension {
                    what: "impact_signs: a (variable, shock) index must be below the number \
                           of variables k",
                    expected: bound,
                    got: index,
                }
            }
            tsecon_ident::IdentError::InvalidArgument { what } => {
                VarError::InvalidArgument { what }
            }
            _ => VarError::InvalidArgument {
                what: "impact_signs is not a valid list of (variable, shock) anchors",
            },
        })?;
    }
    Ok(())
}

/// The bootstrap working set in `f64` values (see the module docs), or
/// `None` when the count itself overflows.
pub fn bootstrap_budget_values(spec: &LongRunBandSpec, k: usize) -> Option<usize> {
    use crate::budget::{cube_values, mat_values, CUBE_COPIES};
    let k2 = k.checked_mul(k)?;
    let hp1 = spec.horizon.checked_add(1)?;
    let cells = hp1.checked_mul(k2)?;
    // The replication's store slot (responses, long-run matrix, impact_signs
    // cells, radius), its overhead and one percentile-buffer value.
    let mut per_draw = draw_width(spec, k)?.checked_add(DRAW_OVERHEAD_VALUES + 1)?;
    if spec.bias_correct {
        per_draw = per_draw
            .checked_add(spec.lags.checked_mul(mat_values(k)?)?)?
            .checked_add(DRAW_OVERHEAD_VALUES)?;
    }
    if let Some((_, scope)) = spec.band {
        let (_, kf) = family_dims(k, spec.horizon, scope)?;
        per_draw = per_draw.checked_add(kf.checked_add(1)?)?;
    }
    if spec.keep_draws {
        per_draw = per_draw.checked_add(cells.checked_add(k2)?)?;
    }
    spec.n_boot
        .checked_mul(per_draw)?
        .checked_add(cube_values(k, hp1, CUBE_COPIES)?)
}

/// Values of one replication's [`DrawStore`] slot: the responses (blocks
/// `0..=horizon`), the long-run matrix (block `horizon + 1`), the
/// pre-anchor `impact_signs` cells and the companion spectral radius.
fn draw_width(spec: &LongRunBandSpec, k: usize) -> Option<usize> {
    let anchors = spec.impact_signs.as_ref().map_or(0, Vec::len);
    spec.horizon
        .checked_add(2)?
        .checked_mul(k.checked_mul(k)?)?
        .checked_add(anchors)?
        .checked_add(1)
}

/// The delta-method size terms (see the module docs).
pub(crate) fn asymptotic_terms(spec: &LongRunBandSpec, k: usize) -> DeltaTerms {
    let (mut family, mut sim, mut sim_work) = (Some(0usize), Some(0usize), Some(0u128));
    if let Some((method, scope)) = spec.band {
        family = long_run_family_values(k, spec.horizon, scope);
        if method == BandMethod::SupT {
            (sim, sim_work) = match family_dims(k, spec.horizon, scope) {
                Some((n_fam, kf)) => sup_t_terms(n_fam, kf, spec.band_n_sim),
                None => (None, None),
            };
        }
    }
    DeltaTerms {
        base: long_run_base_values(k, spec.lags, spec.horizon),
        family,
        sim,
        sim_work,
        jacobian_work: asymptotic_jacobian_work(spec, k),
    }
}

/// The pointwise delta-method working set: the per-cell loading rows, the
/// `G_h` Jacobians, the coefficient covariance and the reported cubes.
fn long_run_base_values(k: usize, lags: usize, horizon: usize) -> Option<usize> {
    use crate::budget::{cube_values, CUBE_COPIES};
    let k2 = k.checked_mul(k)?;
    let hp1 = horizon.checked_add(1)?;
    let cells = hp1.checked_mul(k2)?;
    let dim = lags.checked_mul(k2)?;
    let half = k.checked_mul(k.checked_add(1)?)? / 2;
    let per_cell = dim
        .checked_mul(3)?
        .checked_add(half.checked_mul(2)?)?
        .checked_add(8)?;
    let jacobians = hp1
        .checked_mul(k2.div_ceil(8).checked_mul(8)?)?
        .checked_mul(dim)?;
    cells
        .checked_mul(per_cell)?
        .checked_add(jacobians)?
        .checked_add(dim.checked_mul(dim)?)?
        .checked_add(cube_values(k, hp1, CUBE_COPIES)?)
}

/// A simultaneous band's family covariance and factor (`3 K^2`) and the
/// family cell lists (three indices per cell).
fn long_run_family_values(k: usize, horizon: usize, scope: IrfBandScope) -> Option<usize> {
    let (_, kf) = family_dims(k, horizon, scope)?;
    let cells = horizon.checked_add(1)?.checked_mul(k.checked_mul(k)?)?;
    kf.checked_mul(kf)?
        .checked_mul(3)?
        .checked_add(cells.checked_mul(3)?)
}

/// Refuses a bootstrap request on `k` variables whose working set
/// ([`bootstrap_budget_values`]) exceeds [`LONG_RUN_BANDS_BUDGET_VALUES`]:
/// naming `n_boot` with the largest replication count that fits at this
/// horizon, or `horizon` when not even 2 replications fit.
///
/// # Errors
///
/// [`VarError::SizeBudget`] naming `n_boot` or `horizon`.
pub fn check_bootstrap_budget(spec: &LongRunBandSpec, k: usize) -> Result<(), VarError> {
    let at = |n_boot: usize, horizon: usize| {
        let s = LongRunBandSpec {
            n_boot,
            horizon,
            ..spec.clone()
        };
        bootstrap_budget_values(&s, k)
    };
    match bootstrap_refusal(at, spec.n_boot, spec.horizon) {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

/// Refuses a delta-method request on `k` variables whose working set,
/// sup-t simulation or Jacobian work exceeds its budget, naming the term
/// that is over with the largest value that fits (see
/// [`long_run_asymptotic_bands`]).
///
/// # Errors
///
/// [`VarError::SizeBudget`] naming `horizon` or `band_n_sim`.
pub fn check_asymptotic_budget(spec: &LongRunBandSpec, k: usize) -> Result<(), VarError> {
    let supt = matches!(spec.band, Some((BandMethod::SupT, _)));
    let at = |horizon: usize, band_n_sim: usize| {
        let s = LongRunBandSpec {
            horizon,
            band_n_sim,
            ..spec.clone()
        };
        asymptotic_terms(&s, k)
    };
    match delta_refusal(at, spec.horizon, spec.band_n_sim, supt) {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

/// The delta-method working set in `f64` values and the sup-t simulation
/// work (see the module docs), or `None` on overflow.
pub fn asymptotic_budget(spec: &LongRunBandSpec, k: usize) -> Option<(usize, u128)> {
    let t = asymptotic_terms(spec, k);
    let values = t.base?.checked_add(t.family?)?.checked_add(t.sim?)?;
    Some((values, t.sim_work?))
}

/// The delta-method Jacobian work `horizon (horizon + 1) / 2 * k^2 * lags k^2`
/// checked against [`LONG_RUN_BANDS_DELTA_WORK`]. Saturating, so an absurd
/// `horizon` cannot overflow the count meant to refuse it.
pub fn asymptotic_jacobian_work(spec: &LongRunBandSpec, k: usize) -> u128 {
    crate::irf_asymptotic::delta_jacobian_work(k, spec.lags, spec.horizon)
}

fn min_root_of(res: &VarResults) -> Result<f64, VarError> {
    Ok(res
        .roots_moduli()?
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min))
}

// ---------------------------------------------------------------------------
// Bootstrap arm
// ---------------------------------------------------------------------------

/// Residual-bootstrap bands for the Blanchard-Quah SVAR fitted to `endog`
/// (`n x k`, observations in rows, oldest first), re-identifying inside every
/// draw. See the module docs for the algorithm, the Kilian branch and the
/// budget.
///
/// # Errors
///
/// * [`VarError::InvalidArgument`] / [`VarError::InvalidParameter`] /
///   [`VarError::Dimension`] for `lags == 0`, `n_boot < 2`, `alpha` outside
///   `(0, 1)`, fewer than 2 variables, a `restrictions` pattern that is not a
///   column permutation of the recursive scheme, or bad `impact_signs`;
/// * [`VarError::SizeBudget`] when the working set exceeds
///   [`LONG_RUN_BANDS_BUDGET_VALUES`] (naming `n_boot`, with the largest count
///   that fits, or `horizon` when no replication count fits), checked before
///   anything is allocated;
/// * [`VarError::InvalidArgument`] when the full-sample VAR cannot be
///   identified (a singular `I - sum A_i`, naming `data` and `lags`);
/// * [`VarError::Replication`] naming the first replication whose re-estimated
///   VAR cannot be identified (singular `I - sum A_i` or a non-PD long-run
///   covariance — a measure-zero event that is propagated, never dropped);
/// * any [`crate::VarSpec::fit`] error on the full sample or a pseudo-sample.
pub fn long_run_bootstrap_bands(
    endog: MatRef<'_, f64>,
    spec: &LongRunBandSpec,
) -> Result<LongRunBands, VarError> {
    let k = endog.ncols();
    validate(spec, k)?;
    if spec.n_boot < 2 {
        return Err(VarError::InvalidArgument {
            what: "n_boot must be at least 2 to form a bootstrap distribution; 1000 is the \
                   default (use 999+ for a sup-t band)",
        });
    }
    check_bootstrap_budget(spec, k)?;
    let (h, cumulative) = (spec.horizon, spec.cumulative);
    let engine = ResidualBootstrap::new(
        endog,
        spec.lags,
        spec.trend,
        spec.n_boot,
        spec.seed,
        spec.bias_correct,
    )?;
    let fit = &engine.fit;
    let point_id = identify(&engine.dgp_coefs, fit.sigma_u.as_ref(), spec)
        .map_err(|e| full_sample_error(&engine.dgp_coefs, &e))?;
    let point = theta_cube(&engine.dgp_coefs, &point_id.impact, h, cumulative);
    let min_root = min_root_of(fit)?;

    // Each replication fills its slot of one store: the responses (blocks
    // 0..=h), the long-run matrix (block h + 1), the pre-anchor cells, then
    // the raw refit's companion radius.
    let cells = (h + 1) * k * k;
    let n_anchor = spec.impact_signs.as_ref().map_or(0, Vec::len);
    let mut draws = DrawStore::new(spec.n_boot, cells + k * k + n_anchor + 1);
    let raw: Vec<Result<(), &'static str>> = {
        let slots = draws.slots();
        engine.replicate(spec.seed, spec.n_boot, |rep, res_b, coefs_b| {
            let radius = spectral_radius_of(&res_b.coefs)?;
            Ok(match identify(coefs_b, res_b.sigma_u.as_ref(), spec) {
                Ok(id) => {
                    fill_slot(&slots, rep, |out| {
                        theta_into(coefs_b, &id.impact, h, cumulative, out);
                        write_block(out, h + 1, id.long_run.as_ref());
                        let tail = &mut out[cells + k * k..];
                        tail[..n_anchor].copy_from_slice(&id.pre_anchor);
                        tail[n_anchor] = radius;
                    })?;
                    Ok(())
                }
                Err(e) => Err(ident_draw_message(&e)),
            })
        })?
    };
    collect_draws(raw)?;
    let tail = |b: usize| &draws.draw(b)[cells + k * k..];

    let n_unstable = (0..spec.n_boot)
        .filter(|&b| tail(b)[n_anchor] >= 1.0)
        .count();
    let mismatches = spec.impact_signs.as_ref().map(|anchors| {
        (0..anchors.len())
            .map(|a| {
                let point_neg = point_id.pre_anchor[a] < 0.0;
                (0..spec.n_boot)
                    .filter(|&b| (tail(b)[a] < 0.0) != point_neg)
                    .count()
            })
            .collect::<Vec<usize>>()
    });

    let (se, lower, upper) = summarize(&draws, 0, h + 1, k, spec.alpha);
    let (lse, llo, lup) = summarize(&draws, h + 1, 1, k, spec.alpha);

    let (simultaneous, sim_lower, sim_upper) = match spec.band {
        None => (None, None, None),
        Some((method, scope)) => {
            let cv = draw_critical_values(
                &draws,
                &point,
                &se,
                k,
                h,
                spec.alpha,
                method,
                scope,
                spec.n_boot,
            )?;
            let (lo, hi) = apply_critical_values(&point, &se, &cv)?;
            (Some(cv), Some(lo), Some(hi))
        }
    };
    let (irf_keep, lr_keep) = if spec.keep_draws {
        let mut irf_flat = Vec::with_capacity(spec.n_boot * cells);
        let mut lr_flat = Vec::with_capacity(spec.n_boot * k * k);
        for b in 0..spec.n_boot {
            let d = draws.draw(b);
            irf_flat.extend_from_slice(&d[..cells]);
            lr_flat.extend_from_slice(&d[cells..cells + k * k]);
        }
        (Some(irf_flat), Some(lr_flat))
    } else {
        (None, None)
    };
    drop(draws);
    Ok(LongRunBands {
        point,
        se,
        lower,
        upper,
        impact_point: point_id.impact,
        long_run_point: point_id.long_run,
        long_run_se: lse.into_iter().next().unwrap_or_else(|| Mat::zeros(k, k)),
        long_run_lower: llo.into_iter().next().unwrap_or_else(|| Mat::zeros(k, k)),
        long_run_upper: lup.into_iter().next().unwrap_or_else(|| Mat::zeros(k, k)),
        min_root,
        n_unstable_draws: Some(n_unstable),
        sign_anchor_mismatches: mismatches,
        n_boot: Some(spec.n_boot),
        alpha: spec.alpha,
        bias_correct: spec.bias_correct,
        cumulative,
        simultaneous,
        sim_lower,
        sim_upper,
        irf_draws: irf_keep,
        long_run_draws: lr_keep,
    })
}

// ---------------------------------------------------------------------------
// Delta-method arm
// ---------------------------------------------------------------------------

/// The Jacobian of `vec B` and `vec LR` with respect to
/// `theta = (vec alpha, vech Sigma_u)`, each `k^2 x (dim + half)`, by forward
/// mode over the `dim + half` unit directions. `alpha` is `vec([A_1, ...,
/// A_p])` (index `a k + e` = equation `e`, regressor `a = l k + v`), and
/// `vech` follows [`duplication_matrix`]'s upper-triangle row-major order.
fn bq_jacobian(
    coefs: &[Mat<f64>],
    sigma_u: MatRef<'_, f64>,
    id: &Identified,
    col_of_pos: &[usize],
) -> (Mat<f64>, Mat<f64>) {
    let k = id.impact.nrows();
    let p = coefs.len();
    let k2 = k * k;
    let dim = p * k2;
    let half = k * (k + 1) / 2;
    let c1 = &id.multiplier;
    let lr = &id.long_run;
    // D = I - sum A_i, as the identification computed it.
    let mut d = Mat::<f64>::identity(k, k);
    for a in coefs {
        for j in 0..k {
            for i in 0..k {
                d[(i, j)] -= a[(i, j)];
            }
        }
    }
    // Undo the column relabelling and signs: L[:, c] = sign_s * LR[:, s] with
    // s = col_of_pos[c] and sign_s the sign of LR's diagonal-position entry
    // (the Cholesky diagonal is strictly positive).
    let signs: Vec<f64> = (0..k)
        .map(|s| {
            let c = col_of_pos.iter().position(|&x| x == s).unwrap_or(s);
            if lr[(c, s)] < 0.0 {
                -1.0
            } else {
                1.0
            }
        })
        .collect();
    let mut l = Mat::<f64>::zeros(k, k);
    for (c, &s) in col_of_pos.iter().enumerate() {
        for i in 0..k {
            l[(i, c)] = signs[s] * lr[(i, s)];
        }
    }
    // L^{-1} by forward substitution (L lower, positive diagonal).
    let mut linv = Mat::<f64>::zeros(k, k);
    for col in 0..k {
        for i in 0..k {
            let mut v = if i == col { 1.0 } else { 0.0 };
            for m in 0..i {
                v -= l[(i, m)] * linv[(m, col)];
            }
            linv[(i, col)] = v / l[(i, i)];
        }
    }
    let s = sigma_u;
    let c1t = c1.transpose().to_owned();
    let mut jb = Mat::<f64>::zeros(k2, dim + half);
    let mut jlr = Mat::<f64>::zeros(k2, dim + half);
    // vech positions (upper triangle, row-major) -> (i, j).
    let mut vech: Vec<(usize, usize)> = Vec::with_capacity(half);
    for i in 0..k {
        for j in i..k {
            vech.push((i, j));
        }
    }
    for dir in 0..dim + half {
        let mut dd = Mat::<f64>::zeros(k, k);
        let mut ds = Mat::<f64>::zeros(k, k);
        if dir < dim {
            let (a, e) = (dir / k, dir % k);
            let v = a % k;
            dd[(e, v)] = -1.0;
        } else {
            let (i, j) = vech[dir - dim];
            ds[(i, j)] = 1.0;
            ds[(j, i)] = 1.0;
        }
        // dC1 = -C1 dD C1
        let t1 = &(c1.as_ref() * dd.as_ref()) * c1;
        let dc1 = Mat::from_fn(k, k, |i, j| -t1[(i, j)]);
        // dM = dC1 S C1' + C1 dS C1' + C1 S dC1'
        let a1 = &(dc1.as_ref() * s) * &c1t;
        let a2 = &(c1.as_ref() * ds.as_ref()) * &c1t;
        let a3 = &(c1.as_ref() * s) * dc1.transpose();
        let dm = Mat::from_fn(k, k, |i, j| a1[(i, j)] + a2[(i, j)] + a3[(i, j)]);
        // X = L^{-1} dM L^{-T}; dL = L Phi(X).
        let x = &(&linv * &dm) * linv.transpose();
        let phi = Mat::from_fn(k, k, |i, j| {
            if i > j {
                x[(i, j)]
            } else if i == j {
                0.5 * x[(i, j)]
            } else {
                0.0
            }
        });
        let dl = &l * &phi;
        // dLR = dL with LR's columns and signs; dB = dD LR + D dLR.
        let mut dlr = Mat::<f64>::zeros(k, k);
        for (c, &sh) in col_of_pos.iter().enumerate() {
            for i in 0..k {
                dlr[(i, sh)] = signs[sh] * dl[(i, c)];
            }
        }
        let db1 = dd.as_ref() * lr.as_ref();
        let db2 = d.as_ref() * dlr.as_ref();
        for j in 0..k {
            for i in 0..k {
                jb[(j * k + i, dir)] = db1[(i, j)] + db2[(i, j)];
                jlr[(j * k + i, dir)] = dlr[(i, j)];
            }
        }
    }
    (jb, jlr)
}

/// The delta-method covariance pieces for a set of cells: each cell's
/// loading rows on `alpha` and on `vech Sigma_u`, pre-multiplied by the
/// corresponding covariance. Every variance and covariance is then a dot
/// product in one fixed order, so the diagonal of any family's joint
/// covariance reproduces `se` bit for bit.
struct CellRows {
    ra: Vec<Vec<f64>>,
    rs: Vec<Vec<f64>>,
    ta: Vec<Vec<f64>>,
    ts: Vec<Vec<f64>>,
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    let mut acc = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        acc += x * y;
    }
    acc
}

fn times_cov(r: &[f64], cov: &Mat<f64>) -> Vec<f64> {
    let n = cov.ncols();
    (0..n)
        .map(|c| {
            let mut acc = 0.0;
            for (a, x) in r.iter().enumerate() {
                acc += x * cov[(a, c)];
            }
            acc
        })
        .collect()
}

impl CellRows {
    fn new(ra: Vec<Vec<f64>>, rs: Vec<Vec<f64>>, cov_a: &Mat<f64>, cov_s: &Mat<f64>) -> Self {
        let ta = ra.iter().map(|r| times_cov(r, cov_a)).collect();
        let ts = rs.iter().map(|r| times_cov(r, cov_s)).collect();
        CellRows { ra, rs, ta, ts }
    }

    /// `Cov(cell a, cell b)` (not symmetrized).
    fn cov(&self, a: usize, b: usize, t: f64) -> f64 {
        dot(&self.ta[a], &self.ra[b]) + dot(&self.ts[a], &self.rs[b]) / t
    }

    /// The variance of cell `a`, clamped at zero.
    fn var(&self, a: usize, t: f64) -> f64 {
        self.cov(a, a, t).max(0.0)
    }

    /// Row-major joint covariance of the listed cells: the diagonal is
    /// [`Self::var`] itself (so it reproduces `se` bit for bit) and the
    /// off-diagonal is explicitly symmetrized, as `joint_cov` does for the
    /// Cholesky-ordered bands.
    fn family_cov(&self, ids: &[usize], t: f64) -> Vec<f64> {
        let kk = ids.len();
        let mut out = vec![0.0f64; kk * kk];
        for a in 0..kk {
            for b in 0..kk {
                out[a * kk + b] = if a == b {
                    self.var(ids[a], t)
                } else {
                    0.5 * (self.cov(ids[a], ids[b], t) + self.cov(ids[b], ids[a], t))
                };
            }
        }
        out
    }
}

/// Checks the per-replication results in replication order, turning the
/// FIRST identification failure into [`VarError::Replication`] naming its
/// index. Nothing is dropped: a draw that fails at a boundary is exactly the
/// kind whose removal would narrow the band.
fn collect_draws(raw: Vec<Result<(), &'static str>>) -> Result<(), VarError> {
    for (index, d) in raw.into_iter().enumerate() {
        if let Err(what) = d {
            return Err(VarError::Replication { index, what });
        }
    }
    Ok(())
}

/// Delta-method bands for the Blanchard-Quah SVAR of a fitted VAR `res`
/// (Lütkepohl 2005 conventions; see the module docs). `spec.n_boot`,
/// `spec.seed` and `spec.keep_draws` are not read; `spec.bias_correct` must be
/// false.
///
/// # Errors
///
/// * the validation errors of [`long_run_bootstrap_bands`] (bar `n_boot`);
/// * [`VarError::InvalidArgument`] for `bias_correct` (the delta method draws
///   nothing to re-centre) or `band_n_sim < 2` under sup-t;
/// * [`VarError::SizeBudget`] naming the term that is over, with the largest
///   value that fits: the pointwise working set (`horizon and lags`), a
///   simultaneous band's family covariance (`horizon and band_scope`), the
///   sup-t simulation (`band_n_sim`) or the Jacobian work beyond
///   [`LONG_RUN_BANDS_DELTA_WORK`] (`horizon`); a pointwise band is never
///   told to change `band_scope` or `band_n_sim`, which it does not read;
/// * [`VarError::InvalidArgument`] when the fit cannot be identified;
/// * [`VarError::NotPositiveDefinite`] from the delta-method building blocks.
pub fn long_run_asymptotic_bands(
    res: &VarResults,
    spec: &LongRunBandSpec,
) -> Result<LongRunBands, VarError> {
    let k = res.neqs;
    validate(spec, k)?;
    if res.spec.lags != spec.lags {
        return Err(VarError::InvalidArgument {
            what: "lags: the fitted VAR's lag order differs from spec.lags",
        });
    }
    if spec.bias_correct {
        return Err(VarError::InvalidArgument {
            what: "bias_correct=True has no effect with method=\"asymptotic\": Kilian's \
                   correction re-centres bootstrap draws, and the delta-method arm draws \
                   none; pass method=\"bootstrap\" to get it",
        });
    }
    if let Some((BandMethod::SupT, _)) = spec.band {
        if spec.band_n_sim < 2 {
            return Err(VarError::InvalidArgument {
                what: "band_n_sim must be at least 2 to simulate a sup-t critical value; \
                       100000 is the default (a quantile in the tail of a maximum)",
            });
        }
    }
    check_asymptotic_budget(spec, k)?;
    let h = spec.horizon;
    let cumulative = spec.cumulative;
    let point_id = identify(&res.coefs, res.sigma_u.as_ref(), spec)
        .map_err(|e| full_sample_error(&res.coefs, &e))?;
    let point = theta_cube(&res.coefs, &point_id.impact, h, cumulative);
    let min_root = min_root_of(res)?;
    let col_of_pos =
        tsecon_ident::long_run_ordering(spec.long_run_zeros.as_deref(), k).map_err(|_| {
            VarError::InvalidArgument {
                what: "restrictions is not a valid long-run zero pattern",
            }
        })?;

    let w = delta_workspace(res, h)?;
    let (k2, dim, t) = (w.k2, w.dim, w.t);
    let half = k * (k + 1) / 2;
    // Sigma_sigma = 2 D^+ (Sigma_u ⊗ Sigma_u) D^{+'} (divided by T at use).
    let dk = duplication_matrix(k);
    let dk_pinv = pinv_full_col_rank(&dk, "duplication matrix")?;
    let sxs = kron(res.sigma_u.as_ref(), res.sigma_u.as_ref());
    let cs_half = &(&dk_pinv * &sxs) * dk_pinv.transpose();
    let cov_sig = Mat::from_fn(half, half, |i, j| 2.0 * cs_half[(i, j)]);

    let (jb, jlr) = bq_jacobian(&res.coefs, res.sigma_u.as_ref(), &point_id, &col_of_pos);
    let b = &point_id.impact;

    // Per-horizon loadings of every IRF cell (cell index (h, i, j) ->
    // h*k2 + j*k + i, the vec order).
    let n_cells = (h + 1) * k2;
    let mut ra: Vec<Vec<f64>> = Vec::with_capacity(n_cells);
    let mut rs: Vec<Vec<f64>> = Vec::with_capacity(n_cells);
    let mut f_acc = Mat::<f64>::zeros(k2, dim);
    let mut xi = Mat::<f64>::zeros(k, k);
    for hh in 0..=h {
        // G_h (or F_h = sum_{i<=h} G_i); G_0 = 0.
        let g: Mat<f64> = if hh == 0 {
            Mat::<f64>::zeros(k2, dim)
        } else if cumulative {
            f_acc = &f_acc + &w.g[hh - 1];
            f_acc.clone()
        } else {
            w.g[hh - 1].clone()
        };
        // Psi_h (or Xi_h = sum_{i<=h} Psi_i).
        let psi: Mat<f64> = if cumulative {
            xi = &xi + &w.phi[hh];
            xi.clone()
        } else {
            w.phi[hh].clone()
        };
        for j in 0..k {
            for i in 0..k {
                // Row j*k + i of (B' ⊗ I) G_h + (I ⊗ Psi_h) J_B.
                let mut row_a = vec![0.0f64; dim];
                let mut row_s = vec![0.0f64; half];
                for (c, slot) in row_a.iter_mut().enumerate() {
                    let mut acc = 0.0;
                    for m in 0..k {
                        acc += b[(m, j)] * g[(m * k + i, c)];
                    }
                    for m in 0..k {
                        acc += psi[(i, m)] * jb[(j * k + m, c)];
                    }
                    *slot = acc;
                }
                for (c, slot) in row_s.iter_mut().enumerate() {
                    let mut acc = 0.0;
                    for m in 0..k {
                        acc += psi[(i, m)] * jb[(j * k + m, dim + c)];
                    }
                    *slot = acc;
                }
                ra.push(row_a);
                rs.push(row_s);
            }
        }
    }
    let cells = CellRows::new(ra, rs, &w.cov_a, &cov_sig);
    let z = tsecon_stats::special::inv_norm_cdf(1.0 - spec.alpha / 2.0)?;
    let idx = |hh: usize, i: usize, j: usize| hh * k2 + j * k + i;
    let mut se = Vec::with_capacity(h + 1);
    let mut lower = Vec::with_capacity(h + 1);
    let mut upper = Vec::with_capacity(h + 1);
    for (hh, pt) in point.iter().enumerate() {
        let s = Mat::from_fn(k, k, |i, j| cells.var(idx(hh, i, j), t).sqrt());
        lower.push(Mat::from_fn(k, k, |i, j| pt[(i, j)] - z * s[(i, j)]));
        upper.push(Mat::from_fn(k, k, |i, j| pt[(i, j)] + z * s[(i, j)]));
        se.push(s);
    }

    // Long-run matrix: loadings straight from J_LR.
    let lr_ra: Vec<Vec<f64>> = (0..k2)
        .map(|r| (0..dim).map(|c| jlr[(r, c)]).collect())
        .collect();
    let lr_rs: Vec<Vec<f64>> = (0..k2)
        .map(|r| (0..half).map(|c| jlr[(r, dim + c)]).collect())
        .collect();
    let lr_cells = CellRows::new(lr_ra, lr_rs, &w.cov_a, &cov_sig);
    let lr_point = point_id.long_run.clone();
    let long_run_se = Mat::from_fn(k, k, |i, j| lr_cells.var(j * k + i, t).sqrt());
    let long_run_lower = Mat::from_fn(k, k, |i, j| lr_point[(i, j)] - z * long_run_se[(i, j)]);
    let long_run_upper = Mat::from_fn(k, k, |i, j| lr_point[(i, j)] + z * long_run_se[(i, j)]);

    let (simultaneous, sim_lower, sim_upper) = match spec.band {
        None => (None, None, None),
        Some((method, scope)) => {
            let cv = critical_values_from_family_covs(
                k,
                h,
                spec.alpha,
                method,
                scope,
                spec.band_seed,
                spec.band_n_sim,
                |fam| {
                    let ids: Vec<usize> = fam.iter().map(|&(hh, i, j)| idx(hh, i, j)).collect();
                    Ok(cells.family_cov(&ids, t))
                },
            )?;
            let (lo, hi) = apply_critical_values(&point, &se, &cv)?;
            (Some(cv), Some(lo), Some(hi))
        }
    };
    Ok(LongRunBands {
        point,
        se,
        lower,
        upper,
        impact_point: point_id.impact,
        long_run_point: lr_point,
        long_run_se,
        long_run_lower,
        long_run_upper,
        min_root,
        n_unstable_draws: None,
        sign_anchor_mismatches: None,
        n_boot: None,
        alpha: spec.alpha,
        bias_correct: false,
        cumulative,
        simultaneous,
        sim_lower,
        sim_upper,
        irf_draws: None,
        long_run_draws: None,
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    #[test]
    fn the_first_failing_replication_is_named_and_nothing_is_dropped() {
        let raw = vec![Ok(()), Ok(()), Err("singular"), Ok(()), Err("later")];
        match collect_draws(raw) {
            Err(VarError::Replication { index, what }) => {
                assert_eq!(index, 2);
                assert_eq!(what, "singular");
                let msg = VarError::Replication { index, what }.to_string();
                assert!(msg.contains("replication 2"), "{msg}");
            }
            other => panic!("expected a Replication error, got {other:?}"),
        }
        assert!(collect_draws(vec![Ok(()), Ok(())]).is_ok());
    }

    #[test]
    fn family_covariance_diagonal_is_the_cell_variance_bit_for_bit() {
        // Arbitrary loadings and SPD covariances: the family covariance's
        // diagonal must be the very number `var` returns (the `se` source).
        let ra: Vec<Vec<f64>> = (0..5)
            .map(|r| {
                (0..4)
                    .map(|c| ((r * 7 + c * 3) % 5) as f64 * 0.37 - 0.6)
                    .collect()
            })
            .collect();
        let rs: Vec<Vec<f64>> = (0..5)
            .map(|r| {
                (0..3)
                    .map(|c| ((r + 2 * c) % 4) as f64 * 0.21 - 0.3)
                    .collect()
            })
            .collect();
        let cov_a = Mat::from_fn(4, 4, |i, j| if i == j { 1.3 } else { 0.2 });
        let cov_s = Mat::from_fn(3, 3, |i, j| if i == j { 0.9 } else { -0.1 });
        let cells = CellRows::new(ra, rs, &cov_a, &cov_s);
        let t = 123.0;
        let ids = [4usize, 0, 3, 1];
        let fam = cells.family_cov(&ids, t);
        for (a, &id) in ids.iter().enumerate() {
            assert_eq!(fam[a * 4 + a].to_bits(), cells.var(id, t).to_bits());
            for b in 0..4 {
                assert_eq!(fam[a * 4 + b].to_bits(), fam[b * 4 + a].to_bits());
            }
        }
    }

    #[test]
    fn a_singular_full_sample_long_run_matrix_is_refused_naming_data_and_lags() {
        // A_1 = I makes I - A_1 = 0: C(1) does not exist.
        let coefs = vec![Mat::<f64>::identity(2, 2)];
        let sigma = Mat::<f64>::identity(2, 2);
        let spec = LongRunBandSpec::new(1, 4);
        let err = match identify(&coefs, sigma.as_ref(), &spec) {
            Err(e) => full_sample_error(&coefs, &e),
            Ok(_) => panic!("a singular I - A_1 must not identify"),
        };
        let msg = err.to_string();
        assert!(msg.contains("data") && msg.contains("lags"), "{msg}");
        assert!(matches!(err, VarError::InvalidArgument { .. }), "{err:?}");
    }

    #[test]
    fn theta_cube_reproduces_the_identification_irf_bit_for_bit() {
        let a1 = Mat::from_fn(2, 2, |i, j| [[0.5, 0.1], [0.2, 0.3]][i][j]);
        let a2 = Mat::from_fn(2, 2, |i, j| [[0.1, 0.0], [0.05, 0.1]][i][j]);
        let coefs = vec![a1, a2];
        let refs: Vec<MatRef<'_, f64>> = coefs.iter().map(Mat::as_ref).collect();
        let sigma = Mat::from_fn(2, 2, |i, j| [[1.0, 0.3], [0.3, 0.8]][i][j]);
        let r = tsecon_ident::long_run_svar(&refs, sigma.as_ref(), 12, None, false).unwrap();
        let cube = theta_cube(&coefs, &r.impact, 12, false);
        for (a, b) in cube.iter().zip(r.irf.iter()) {
            for i in 0..2 {
                for j in 0..2 {
                    assert_eq!(a[(i, j)].to_bits(), b[(i, j)].to_bits());
                }
            }
        }
    }

    #[test]
    fn theta_into_is_theta_cube_flattened_bit_for_bit() {
        let a1 = Mat::from_fn(2, 2, |i, j| [[0.5, 0.1], [0.2, 0.3]][i][j]);
        let a2 = Mat::from_fn(2, 2, |i, j| [[0.1, 0.0], [0.05, 0.1]][i][j]);
        let coefs = vec![a1, a2];
        let b = Mat::from_fn(2, 2, |i, j| [[1.1, 0.0], [0.3, 0.7]][i][j]);
        for cumulative in [false, true] {
            for horizon in [0usize, 1, 2, 7] {
                let cube = theta_cube(&coefs, &b, horizon, cumulative);
                let mut flat = vec![42.0; (horizon + 2) * 4];
                theta_into(&coefs, &b, horizon, cumulative, &mut flat);
                // The trailing block is the caller's (the long-run matrix).
                assert!(flat[(horizon + 1) * 4..].iter().all(|&v| v == 42.0));
                for (h, m) in cube.iter().enumerate() {
                    for i in 0..2 {
                        for j in 0..2 {
                            assert_eq!(flat[h * 4 + i * 2 + j].to_bits(), m[(i, j)].to_bits());
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn draw_overhead_covers_the_per_replication_types() {
        use core::mem::size_of;
        let budget = DRAW_OVERHEAD_VALUES * 8;
        // While the substreams are spawned: a SeedSequence child (two heap
        // buffers of a few words each) beside each stream.
        let spawn =
            size_of::<tsecon_rng::SeedSequence>() + 2 * 32 + size_of::<tsecon_rng::Stream>();
        assert!(spawn <= budget, "{spawn}");
        // While the results are collected: the stream, up to three result
        // slots and the replication's slot lock.
        let lock = size_of::<std::sync::Mutex<&mut [f64]>>();
        let slot = size_of::<Result<Result<(), &'static str>, VarError>>();
        let collect = size_of::<tsecon_rng::Stream>() + 3 * slot + lock;
        assert!(collect <= budget, "{collect}");
    }

    #[test]
    fn budgets_overflow_to_refusal_not_to_a_wrapped_count() {
        let mut spec = LongRunBandSpec::new(2, usize::MAX);
        assert_eq!(bootstrap_budget_values(&spec, 2), None);
        assert_eq!(asymptotic_budget(&spec, 2), None);
        spec.horizon = 10;
        spec.n_boot = usize::MAX;
        assert_eq!(bootstrap_budget_values(&spec, 2), None);
    }
}
