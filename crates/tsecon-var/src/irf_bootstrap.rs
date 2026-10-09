//! Frequentist **bootstrap** confidence bands for VAR impulse responses.
//!
//! This is the banded companion to [`crate::VarResults::irf`], filling the
//! `// TODO(phase0)` hook in `src/irf.rs`. It implements the residual
//! (nonparametric iid) recursive-design bootstrap of Runkle (1987) and
//! Kilian (1998), with an optional Kilian (1998) bias correction:
//!
//! 1. fit the reduced-form VAR by OLS; keep the coefficient matrices
//!    `A_1..A_p`, the intercept, the df-adjusted residual covariance
//!    `sigma_u`, and the (mean-centered) residual rows `U`;
//! 2. for each of `n_boot` replications, resample the residual **rows**
//!    with replacement (iid Efron bootstrap, via the shared
//!    [`tsecon_bootstrap`] engine so the draws are reproducible from a
//!    seed — never system RNG), regenerate a pseudo-sample recursively
//!    from the fitted coefficients conditional on the observed first `p`
//!    observations, refit the VAR, and recompute the impulse responses
//!    (orthogonalized or not, cumulated or not);
//! 3. the standard error is the standard deviation across the draws and
//!    the band endpoints are the `alpha/2` and `1 - alpha/2` **percentiles**
//!    per `(horizon, response, impulse)` cell (Efron percentile interval);
//! 4. the point estimate is the full-sample impulse response, so `point`
//!    matches [`crate::VarResults::irf`] exactly (or the bias-corrected
//!    impulse response when `bias_correct` is set).
//!
//! ## Reproducibility contract
//!
//! Every draw is keyed by `(seed, replication_index)` through
//! [`tsecon_bootstrap`]'s SeedSequence-spawned substreams, so the bands are
//! **bit-identical** for a given seed at any thread count. No function here
//! creates entropy from the OS.
//!
//! ## Bias correction (Kilian 1998)
//!
//! With `bias_correct = true` the mean bias of the least-squares slope
//! estimator is estimated by an inner bootstrap around the fitted
//! coefficients, shrunk toward zero (Kilian's stationarity-preserving
//! adjustment) so the bias-corrected estimate stays inside the unit circle,
//! and subtracted from both the point estimate and every outer bootstrap
//! estimate; the outer pseudo-samples are then generated from the
//! bias-corrected coefficients.
//!
//! # References
//!
//! Runkle (1987); Kilian (1998, *Review of Economics and Statistics*);
//! Lütkepohl (2005, appendix D).

use tsecon_linalg::faer::{Mat, MatRef};
use tsecon_linalg::{companion_from_var, spectral_radius};

use tsecon_bootstrap::{indices, par_replicate, BlockScheme, BootstrapError};
use tsecon_stats::simultaneous;

use crate::error::{NonFiniteCell, VarError};
use crate::estimate::{lag_coefs, SIGMA_U_WHAT};
use crate::irf::ma_rep;
use crate::irf_asymptotic::{apply_critical_values, irf_families, IrfCriticalValues};
use crate::results::{chol_lower, VarResults};
use crate::spec::{Trend, VarSpec};

pub use crate::irf_asymptotic::{BandMethod, IrfBandScope};

/// Bootstrap confidence bands for the impulse responses of a VAR.
///
/// Each of `point`, `se`, `lower`, `upper` has `horizon + 1` entries
/// indexed by horizon `h`, each a `k x k` matrix whose `(i, j)` cell is the
/// response of variable `i` to a shock in variable `j` — the same
/// orientation as [`crate::Irf`].
#[derive(Debug, Clone)]
pub struct IrfBands {
    /// Full-sample impulse response point estimate (bias-corrected when
    /// `bias_correct` was set); matches [`crate::VarResults::irf`] under the
    /// same `orth`/`cumulative` flags when `bias_correct` is false.
    pub point: Vec<Mat<f64>>,
    /// Bootstrap standard error: the sample standard deviation (divisor
    /// `n_boot - 1`) across the bootstrap impulse responses, per cell.
    pub se: Vec<Mat<f64>>,
    /// Lower percentile band at level `alpha` — the `alpha/2` percentile of
    /// the bootstrap draws (Efron percentile method).
    pub lower: Vec<Mat<f64>>,
    /// Upper percentile band at level `alpha` — the `1 - alpha/2`
    /// percentile of the bootstrap draws.
    pub upper: Vec<Mat<f64>>,
    /// Number of bootstrap replications actually used.
    pub n_boot: usize,
    /// The two-sided level echoed back.
    pub alpha: f64,
    /// Whether the Kilian (1998) bias correction was applied.
    pub bias_correct: bool,
    /// Multipliers of the **simultaneous** band, when
    /// [`bootstrap_irf_bands_simultaneous`] produced this object; `None` from
    /// the plain [`bootstrap_irf_bands`].
    pub simultaneous: Option<IrfCriticalValues>,
    /// `point - c·se` of the simultaneous band, one `k x k` matrix per horizon.
    /// `Some` exactly when [`Self::simultaneous`] is.
    ///
    /// **This is a different shape of interval from [`Self::lower`].**
    /// [`Self::lower`] / [`Self::upper`] are Efron *percentile* bounds — they
    /// pick up bootstrap skewness and need not be symmetric about `point`. The
    /// simultaneous band is the sup-t construction `point ± c·se`, symmetric by
    /// definition. So `sim_lower` is **not** a widened percentile bound, and it
    /// is not guaranteed to sit outside `lower` at every cell. What it is
    /// guaranteed to contain is the *symmetric* pointwise band `point ± z·se`,
    /// which is the like-for-like comparator: only the multiplier differs.
    pub sim_lower: Option<Vec<Mat<f64>>>,
    /// `point + c·se` of the simultaneous band. See [`Self::sim_lower`].
    pub sim_upper: Option<Vec<Mat<f64>>>,
}

/// One horizon-indexed impulse-response cube (`horizon + 1` `k x k`
/// matrices).
pub(crate) type Cube = Vec<Mat<f64>>;

/// The offset added to `seed` to key the inner bias-estimation bootstrap,
/// so the bias draws and the outer band draws are independent given the
/// user's seed (a fixed transform keeps the whole procedure reproducible).
/// The constant is the 64-bit golden-ratio odd integer (Knuth's multiplier).
pub(crate) const BIAS_SEED_OFFSET: u64 = 0x9E37_79B9_7F4A_7C15;

/// Residual (nonparametric iid) recursive-design bootstrap confidence bands
/// for the impulse responses of a reduced-form VAR(`lags`) fitted on
/// `endog` (an `n x k` matrix, observations in rows, oldest first).
///
/// * `orth` — orthogonalize the responses through the lower Cholesky factor
///   of `sigma_u` (recursive ordering) exactly as
///   [`crate::VarResults::irf`]'s orthogonalized array; when false the
///   reduced-form forecast-error responses `Psi_h` are used.
/// * `cumulative` — put the bands on the **cumulated** impulse response
///   (the level response when the VAR is estimated in differences); each
///   bootstrap draw is cumulated over the horizon before the percentiles
///   are taken.
/// * `alpha` — two-sided level: the bands are the `alpha/2` and
///   `1 - alpha/2` percentiles.
/// * `n_boot` — number of bootstrap replications.
/// * `seed` — reproducibility seed for the resampling engine.
/// * `bias_correct` — apply the Kilian (1998) bias correction (see the
///   module docs).
///
/// # Errors
///
/// * [`VarError::InvalidArgument`] if `lags == 0`, `horizon`'s implied
///   arrays would be empty, `n_boot < 2`, or `alpha` is not in `(0, 1)`;
/// * any error from [`VarSpec::fit`] on the full sample;
/// * [`VarError::Replication`] naming the first bootstrap replication (of
///   the outer bootstrap, or of the inner bias bootstrap under
///   `bias_correct`) whose pseudo-sample refit fails — a singular residual
///   covariance or regressor cross-product, or an overflow;
/// * [`VarError::NotPositiveDefinite`] if an orthogonalization Cholesky
///   fails.
#[allow(clippy::too_many_arguments)]
pub fn bootstrap_irf_bands(
    endog: MatRef<'_, f64>,
    lags: usize,
    trend: Trend,
    horizon: usize,
    orth: bool,
    cumulative: bool,
    alpha: f64,
    n_boot: usize,
    seed: u64,
    bias_correct: bool,
) -> Result<IrfBands, VarError> {
    run_bootstrap(
        endog,
        lags,
        trend,
        horizon,
        orth,
        cumulative,
        alpha,
        n_boot,
        seed,
        bias_correct,
        None,
    )
}

/// [`bootstrap_irf_bands`] plus a **simultaneous** band over a declared family
/// of `(horizon, response, shock)` cells.
///
/// The percentile bands, the point estimate, the standard errors and the
/// resampling stream are all bit-identical to [`bootstrap_irf_bands`] at the
/// same arguments: this only adds a second, jointly-valid interval alongside
/// them.
///
/// # Why a bootstrap band needs this too
///
/// A pointwise bootstrap band is exactly as anti-conservative about whole paths
/// as an asymptotic one — tsecon's interval-coverage audit measured the
/// bootstrap band containing the whole `h = 0..12` path in 73.5% of samples at
/// `T = 500` against a nominal 90%, versus 72.2% for the asymptotic band. The
/// bootstrap fixes the sampling distribution, not the multiplicity.
///
/// # Method
///
/// [`BandMethod::SupT`] is the route to prefer here, and it needs no simulation
/// and no seed beyond the one already driving the resampling: the bootstrap
/// replications *are* draws of the estimand, so the max-|t| statistic can be
/// evaluated on them directly (Montiel Olea and Plagborg-Møller's workhorse
/// construction). For each replication `b` the statistic
/// `M_b = max_cell |theta*_{b,cell} - theta_hat_cell| / se_cell` is formed over
/// the family, and `c` is the `1 - alpha` quantile of `M_1, ..., M_B`. Centring
/// is at the point estimate rather than at the bootstrap mean, so the band stays
/// centred where the reported `point` is; under bootstrap bias that makes it
/// slightly wider than a mean-centred version.
///
/// `n_boot` should be large — this reads a quantile in the tail of a maximum,
/// so 999+ replications, not 199.
///
/// [`BandMethod::Sidak`] and [`BandMethod::Bonferroni`] are also available and
/// need only `K`; [`BandMethod::Pointwise`] returns the symmetric Wald band
/// `point ± z·se`, which is the honest comparator for the simultaneous band
/// (see [`IrfBands::sim_lower`]).
///
/// # Errors
///
/// Everything [`bootstrap_irf_bands`] can return, plus:
///
/// * [`VarError::NonFinite`] if any bootstrap replication produced a non-finite
///   impulse response — one `NaN` replication would otherwise silently poison
///   the maximum;
/// * [`VarError::Stats`] from the simultaneous-band layer.
#[allow(clippy::too_many_arguments)]
pub fn bootstrap_irf_bands_simultaneous(
    endog: MatRef<'_, f64>,
    lags: usize,
    trend: Trend,
    horizon: usize,
    orth: bool,
    cumulative: bool,
    alpha: f64,
    n_boot: usize,
    seed: u64,
    bias_correct: bool,
    method: BandMethod,
    scope: IrfBandScope,
) -> Result<IrfBands, VarError> {
    run_bootstrap(
        endog,
        lags,
        trend,
        horizon,
        orth,
        cumulative,
        alpha,
        n_boot,
        seed,
        bias_correct,
        Some((method, scope)),
    )
}

#[allow(clippy::too_many_arguments)]
fn run_bootstrap(
    endog: MatRef<'_, f64>,
    lags: usize,
    trend: Trend,
    horizon: usize,
    orth: bool,
    cumulative: bool,
    alpha: f64,
    n_boot: usize,
    seed: u64,
    bias_correct: bool,
    band: Option<(BandMethod, IrfBandScope)>,
) -> Result<IrfBands, VarError> {
    if lags == 0 {
        return Err(VarError::InvalidArgument {
            what: "bootstrap IRF bands need lags >= 1: a VAR(0) has no dynamics, so \
                   every response beyond h = 0 is identically zero",
        });
    }
    if n_boot < 2 {
        return Err(VarError::InvalidArgument {
            what: "n_boot must be at least 2 to form a bootstrap distribution; \
                   1000 is the usual default",
        });
    }
    if !(alpha > 0.0 && alpha < 1.0) {
        return Err(VarError::InvalidParameter {
            name: "alpha",
            value: alpha,
            requirement: "a value strictly inside (0, 1) — alpha = 0.1 gives 90% bands",
        });
    }

    // The shared residual-bootstrap engine: full-sample fit, centred
    // residuals, observed initial rows, the optional Kilian bias term and the
    // coefficients that generate the pseudo-samples.
    let engine = ResidualBootstrap::new(endog, lags, trend, n_boot, seed, bias_correct)?;
    let fit = &engine.fit;
    let k = fit.neqs;

    // Point estimate (matches `var_irf` when bias_correct == false).
    let point = irf_cube(
        &engine.dgp_coefs,
        fit.sigma_u.as_ref(),
        horizon,
        orth,
        cumulative,
    )?;

    // Outer bootstrap: one reproducible substream per replication; the first
    // failed replication, if any, is propagated.
    let draws: Vec<Vec<Mat<f64>>> = engine.replicate(seed, n_boot, |res_b, coefs_b| {
        irf_cube(coefs_b, res_b.sigma_u.as_ref(), horizon, orth, cumulative)
    })?;

    let (se, lower, upper) = summarize(&draws, horizon, k, alpha);

    let (simultaneous, sim_lower, sim_upper) = match band {
        None => (None, None, None),
        Some((method, scope)) => {
            let cv = draw_critical_values(
                &draws, &point, &se, k, horizon, alpha, method, scope, n_boot,
            )?;
            let (lo, hi) = apply_critical_values(&point, &se, &cv)?;
            (Some(cv), Some(lo), Some(hi))
        }
    };

    Ok(IrfBands {
        point,
        se,
        lower,
        upper,
        n_boot,
        alpha,
        bias_correct,
        simultaneous,
        sim_lower,
        sim_upper,
    })
}

/// The residual (iid) recursive-design bootstrap engine shared by every VAR
/// band that re-estimates the reduced form per draw: [`bootstrap_irf_bands`]
/// (Cholesky / reduced-form responses) and
/// [`crate::long_run_bands::long_run_bootstrap_bands`] (Blanchard-Quah,
/// re-identified inside every draw).
///
/// It owns the full-sample fit, the column-centred residuals, the observed
/// initial rows, the optional Kilian (1998) bias term (inner bootstrap keyed
/// by `seed + BIAS_SEED_OFFSET`, stationarity shrinkage by repeated
/// `delta -= 0.01`), and the pseudo-sample loop. Callers supply only the
/// per-draw statistic, so the resampling stream, the recursion and the
/// correction are one code path: the same `(seed, n_boot)` gives every caller
/// the same pseudo-samples.
pub(crate) struct ResidualBootstrap {
    /// The full-sample OLS fit the bootstrap is centred on.
    pub(crate) fit: VarResults,
    /// The coefficients that generate the outer pseudo-samples and define a
    /// bias-corrected point estimate: `correct_coefs(A_hat, bias)` under
    /// `bias_correct`, `A_hat` otherwise.
    pub(crate) dgp_coefs: Vec<Mat<f64>>,
    /// The (shrunk) Kilian bias term, when requested.
    bias: Option<Vec<Mat<f64>>>,
    lags: usize,
    trend: Trend,
    /// Column-centred full-sample residuals.
    uc: Mat<f64>,
    /// The observed first `lags` rows that seed every pseudo-sample.
    init: Mat<f64>,
    /// The full-sample intercept (unchanged by the bias correction).
    intercept: Vec<f64>,
}

impl ResidualBootstrap {
    /// Fit the full sample and prepare the bootstrap DGP. With
    /// `bias_correct`, this runs the inner `n_boot`-replication bias
    /// bootstrap. The caller has validated `lags >= 1` and `n_boot >= 2`.
    pub(crate) fn new(
        endog: MatRef<'_, f64>,
        lags: usize,
        trend: Trend,
        n_boot: usize,
        seed: u64,
        bias_correct: bool,
    ) -> Result<Self, VarError> {
        // Full-sample fit: the reduced form we bootstrap around.
        let fit = VarSpec { lags, trend }.fit(endog)?;
        let k = fit.neqs;
        let te = fit.nobs; // effective residual rows T_eff = n - lags
        let p = lags;

        // Mean-center the residuals (Kilian 1998): with an intercept the column
        // means are already ~0, but centering makes the reduced-form (trend="n")
        // case correct too.
        let uc = center_columns(fit.resid.as_ref());

        // Observed first `p` rows seed every recursive pseudo-sample.
        let init = Mat::from_fn(p, k, |t, j| endog[(t, j)]);
        let intercept = fit.intercept.clone();

        // Optional Kilian bias term, shrunk to preserve stationarity.
        let bias = if bias_correct {
            Some(estimate_bias(
                &fit.coefs, &intercept, &init, &uc, te, seed, n_boot,
            )?)
        } else {
            None
        };

        // The coefficients that (a) generate the outer pseudo-samples and
        // (b) define the point estimate: bias-corrected when requested.
        let dgp_coefs = match &bias {
            Some(b) => correct_coefs(&fit.coefs, b)?,
            None => fit.coefs.clone(),
        };
        Ok(ResidualBootstrap {
            fit,
            dgp_coefs,
            bias,
            lags,
            trend,
            uc,
            init,
            intercept,
        })
    }

    /// Run the outer bootstrap: replication `b` draws `T_eff` iid row indices
    /// from substream `b` of `SeedSequence(seed).spawn(n_boot)`, rebuilds the
    /// pseudo-sample recursively from [`Self::dgp_coefs`], refits the VAR, and
    /// hands `per_draw` the RAW refit plus the coefficients to use (the
    /// Kilian-corrected refit coefficients under `bias_correct`, the raw ones
    /// otherwise). Results come back in replication order; the first failed
    /// replication is propagated. A refit failure is reported as
    /// [`VarError::Replication`] with its index: the full sample passed every
    /// check, so the full-sample messages (which tell the caller to drop a
    /// duplicated column) would be false here.
    pub(crate) fn replicate<T, F>(
        &self,
        seed: u64,
        n_boot: usize,
        per_draw: F,
    ) -> Result<Vec<T>, VarError>
    where
        T: Send,
        F: Fn(&VarResults, &[Mat<f64>]) -> Result<T, VarError> + Send + Sync,
    {
        let te = self.uc.nrows();
        let (lags, trend) = (self.lags, self.trend);
        let raw = par_replicate(seed, n_boot, |rep, stream| -> Result<T, VarError> {
            let idx = indices(BlockScheme::Iid, te, stream).map_err(map_boot_err)?;
            let ustar = gather_rows(&self.uc, &idx);
            let ysim = simulate_recursive(&self.dgp_coefs, &self.intercept, &self.init, &ustar);
            let res_b = VarSpec { lags, trend }
                .fit(ysim.as_ref())
                .map_err(|e| pseudo_sample_error(&e, rep, false))?;
            let out = match &self.bias {
                Some(b) => {
                    let coefs_b = correct_coefs(&res_b.coefs, b)?;
                    per_draw(&res_b, &coefs_b)
                }
                None => per_draw(&res_b, &res_b.coefs),
            };
            // A refit covariance whose ML scaling passed its Cholesky by
            // rounding can fail it at the df-adjusted scaling the statistic
            // uses (an orthogonalisation): the same pseudo-sample failure.
            out.map_err(|e| match e {
                VarError::NotPositiveDefinite { what } if what == SIGMA_U_WHAT => {
                    pseudo_sample_error(&e, rep, false)
                }
                other => other,
            })
        })
        .map_err(map_boot_err)?;
        raw.into_iter().collect()
    }
}

/// Simultaneous critical values from the bootstrap replications themselves.
///
/// Cells with `se == 0` are pinned by construction (the Cholesky zeros of the
/// impact matrix, and the whole `orth = false` impact matrix): they take no part
/// in the maximum and no part in the Šidák/Bonferroni cell count, and keep their
/// zero-width band.
#[allow(clippy::too_many_arguments)]
pub(crate) fn draw_critical_values(
    draws: &[Cube],
    point: &[Mat<f64>],
    se: &[Mat<f64>],
    k: usize,
    horizon: usize,
    alpha: f64,
    method: BandMethod,
    scope: IrfBandScope,
    n_boot: usize,
) -> Result<IrfCriticalValues, VarError> {
    let z = simultaneous::pointwise_critical_value(alpha).map_err(VarError::Stats)?;
    let families = irf_families(k, horizon, scope);
    let n_cells = families[0].0.len();
    let mut values = vec![vec![z; k]; k];
    let mut used = vec![vec![0usize; k]; k];

    // Size the flattened draw matrix from the replications actually in hand,
    // not from the requested `n_boot`: a stale tail would silently contribute
    // zeros to the maximum, which is the one failure mode this routine must
    // never have.
    let n_draws = draws.len();
    debug_assert_eq!(n_draws, n_boot);
    let mut flat = vec![0.0f64; n_draws * n_cells];
    for (cells, grid) in families.iter() {
        let theta: Vec<f64> = cells.iter().map(|&(h, i, j)| point[h][(i, j)]).collect();
        let sd: Vec<f64> = cells.iter().map(|&(h, i, j)| se[h][(i, j)]).collect();
        let n_used = sd.iter().filter(|s| **s > 0.0).count();

        let c = if n_used == 0 {
            z
        } else {
            match method {
                BandMethod::Pointwise => z,
                BandMethod::SupT => {
                    for (b, d) in draws.iter().enumerate() {
                        for (c_idx, &(h, i, j)) in cells.iter().enumerate() {
                            let v = d[h][(i, j)];
                            if !v.is_finite() {
                                return Err(VarError::NonFinite {
                                    what: "the bootstrap impulse-response draws (row = \
                                           replication, column = cell of the band family)",
                                    at: Some(NonFiniteCell {
                                        row: b,
                                        col: c_idx,
                                        why: "an explosive pseudo-sample overflowed its \
                                              responses, and the sup-t maximum is undefined \
                                              with a non-finite draw (dropping the draw would \
                                              narrow the band); change seed or n_boot, or use \
                                              method=\"asymptotic\"",
                                    }),
                                });
                            }
                            flat[b * n_cells + c_idx] = v;
                        }
                    }
                    simultaneous::sup_t_from_draws(&flat, n_draws, &theta, &sd, alpha)
                        .map_err(VarError::Stats)?
                }
                BandMethod::Sidak => {
                    simultaneous::sidak_critical_value(alpha, n_used).map_err(VarError::Stats)?
                }
                BandMethod::Bonferroni => simultaneous::bonferroni_critical_value(alpha, n_used)
                    .map_err(VarError::Stats)?,
            }
        };
        for &(i, j) in grid {
            values[i][j] = c;
            used[i][j] = n_used;
        }
    }

    Ok(IrfCriticalValues {
        values,
        n_cells,
        n_cells_used: used,
        pointwise: z,
        method,
        scope,
        alpha,
    })
}

/// Column-mean-center a `T x k` residual matrix.
fn center_columns(u: MatRef<'_, f64>) -> Mat<f64> {
    let (t, k) = (u.nrows(), u.ncols());
    let tf = t as f64;
    let means: Vec<f64> = (0..k)
        .map(|j| (0..t).map(|i| u[(i, j)]).sum::<f64>() / tf)
        .collect();
    Mat::from_fn(t, k, |i, j| u[(i, j)] - means[j])
}

/// Gather resampled residual rows: `out[t, :] = uc[idx[t], :]`.
fn gather_rows(uc: &Mat<f64>, idx: &[usize]) -> Mat<f64> {
    let k = uc.ncols();
    Mat::from_fn(idx.len(), k, |t, j| uc[(idx[t], j)])
}

/// Regenerate a pseudo-sample recursively from fitted coefficients,
/// conditional on the observed first `p` rows:
/// `y_t = c + sum_i A_i y_{t-i} + u*_{t-p}` for `t = p..n`.
fn simulate_recursive(
    coefs: &[Mat<f64>],
    intercept: &[f64],
    init: &Mat<f64>,
    ustar: &Mat<f64>,
) -> Mat<f64> {
    let p = coefs.len();
    let k = init.ncols();
    let te = ustar.nrows();
    let n = p + te;
    let mut y = Mat::<f64>::zeros(n, k);
    for t in 0..p {
        for j in 0..k {
            y[(t, j)] = init[(t, j)];
        }
    }
    for t in p..n {
        for r in 0..k {
            let mut v = intercept[r] + ustar[(t - p, r)];
            for i in 1..=p {
                let a = &coefs[i - 1];
                for c in 0..k {
                    v += a[(r, c)] * y[(t - i, c)];
                }
            }
            y[(t, r)] = v;
        }
    }
    y
}

/// Impulse-response cube for coefficient matrices `coefs` and residual
/// covariance `sigma_u`: `horizon + 1` matrices, orthogonalized through the
/// lower Cholesky factor when `orth`, cumulated over the horizon when
/// `cumulative`. Replicates the `var_irf` binding's arithmetic exactly.
fn irf_cube(
    coefs: &[Mat<f64>],
    sigma_u: MatRef<'_, f64>,
    horizon: usize,
    orth: bool,
    cumulative: bool,
) -> Result<Vec<Mat<f64>>, VarError> {
    let psi = ma_rep(coefs, horizon)?;
    let mut cube: Vec<Mat<f64>> = if orth {
        let pchol = chol_lower(sigma_u, SIGMA_U_WHAT)?;
        psi.iter().map(|m| m * &pchol).collect()
    } else {
        psi
    };
    if cumulative {
        for h in 1..cube.len() {
            let prev = cube[h - 1].clone();
            let cur = &cube[h] + &prev;
            cube[h] = cur;
        }
    }
    Ok(cube)
}

/// Per-cell bootstrap standard error (sample SD, divisor `n_boot - 1`) and
/// the `alpha/2` / `1 - alpha/2` percentile bands.
pub(crate) fn summarize(
    draws: &[Cube],
    horizon: usize,
    k: usize,
    alpha: f64,
) -> (Cube, Cube, Cube) {
    let hh = horizon + 1;
    let nb = draws.len();
    let nbf = nb as f64;
    let ql = alpha / 2.0;
    let qu = 1.0 - alpha / 2.0;

    let mut se = vec![Mat::<f64>::zeros(k, k); hh];
    let mut lower = vec![Mat::<f64>::zeros(k, k); hh];
    let mut upper = vec![Mat::<f64>::zeros(k, k); hh];
    let mut vals = vec![0.0f64; nb];

    for h in 0..hh {
        for i in 0..k {
            for j in 0..k {
                for (b, d) in draws.iter().enumerate() {
                    vals[b] = d[h][(i, j)];
                }
                let mean = vals.iter().sum::<f64>() / nbf;
                let ss = vals.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>();
                se[h][(i, j)] = (ss / (nbf - 1.0)).sqrt();
                vals.sort_by(f64::total_cmp);
                lower[h][(i, j)] = percentile_sorted(&vals, ql);
                upper[h][(i, j)] = percentile_sorted(&vals, qu);
            }
        }
    }
    (se, lower, upper)
}

/// Linear-interpolated percentile of an ascending slice, matching NumPy's
/// default `numpy.percentile(..., method="linear")`: position `q (n - 1)`,
/// interpolating between the two bracketing order statistics. Shared with
/// the GIRF engine's across-history and across-draw bands.
pub(crate) fn percentile_sorted(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    let pos = q * (n as f64 - 1.0);
    let lo = pos.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    let frac = pos - lo as f64;
    sorted[lo] + frac * (sorted[hi] - sorted[lo])
}

/// Estimate the mean bias `E*[A*] - A_hat` of the LS slope estimator by an
/// inner bootstrap around the fitted coefficients, then shrink it (Kilian's
/// stationarity-preserving adjustment) so that `A_hat - bias` stays inside
/// the unit circle. Each inner refit computes the slopes alone
/// ([`lag_coefs`], bit-identical to the full fit's), so a pseudo-sample
/// whose residual covariance happens to be singular does not stop it.
fn estimate_bias(
    coefs_hat: &[Mat<f64>],
    intercept: &[f64],
    init: &Mat<f64>,
    uc: &Mat<f64>,
    te: usize,
    seed: u64,
    n_inner: usize,
) -> Result<Vec<Mat<f64>>, VarError> {
    let p = coefs_hat.len();
    let k = coefs_hat[0].nrows();
    let lags = p;
    let trend = if intercept.iter().all(|&c| c == 0.0) {
        Trend::None
    } else {
        Trend::Constant
    };
    let bias_seed = seed.wrapping_add(BIAS_SEED_OFFSET);

    let raw = par_replicate(
        bias_seed,
        n_inner,
        |rep, stream| -> Result<Vec<Mat<f64>>, VarError> {
            let idx = indices(BlockScheme::Iid, te, stream).map_err(map_boot_err)?;
            let ustar = gather_rows(uc, &idx);
            let ysim = simulate_recursive(coefs_hat, intercept, init, &ustar);
            lag_coefs(ysim.as_ref(), lags, trend).map_err(|e| pseudo_sample_error(&e, rep, true))
        },
    )
    .map_err(map_boot_err)?;
    let estimates: Vec<Vec<Mat<f64>>> = raw.into_iter().collect::<Result<_, _>>()?;

    // Mean of the inner estimates.
    let mut mean = vec![Mat::<f64>::zeros(k, k); p];
    for est in &estimates {
        for i in 0..p {
            let acc = &mean[i] + &est[i];
            mean[i] = acc;
        }
    }
    let nf = estimates.len() as f64;
    // Raw bias E*[A*] - A_hat.
    let bias_raw: Vec<Mat<f64>> = (0..p)
        .map(|i| Mat::from_fn(k, k, |r, c| mean[i][(r, c)] / nf - coefs_hat[i][(r, c)]))
        .collect();

    // Stationarity-preserving shrinkage (Kilian 1998): reduce delta from 1
    // in 0.01 steps until A_hat - delta * bias_raw is stable.
    let mut delta = 1.0_f64;
    loop {
        let cand = scale_subtract(coefs_hat, &bias_raw, delta);
        if spectral_radius_of(&cand)? < 1.0 {
            return Ok(scale(&bias_raw, delta));
        }
        delta -= 0.01;
        if delta <= 0.0 {
            // No admissible correction: fall back to zero bias.
            return Ok(vec![Mat::<f64>::zeros(k, k); p]);
        }
    }
}

/// `coefs - bias`, with a per-call stationarity shrinkage so the corrected
/// coefficients stay inside the unit circle (used for the point estimate
/// and for every outer bootstrap estimate).
fn correct_coefs(coefs: &[Mat<f64>], bias: &[Mat<f64>]) -> Result<Vec<Mat<f64>>, VarError> {
    let mut delta = 1.0_f64;
    loop {
        let cand = scale_subtract(coefs, bias, delta);
        if spectral_radius_of(&cand)? < 1.0 {
            return Ok(cand);
        }
        delta -= 0.01;
        if delta <= 0.0 {
            return Ok(coefs.to_vec());
        }
    }
}

/// Elementwise `coefs[i] - delta * bias[i]`.
fn scale_subtract(coefs: &[Mat<f64>], bias: &[Mat<f64>], delta: f64) -> Vec<Mat<f64>> {
    let p = coefs.len();
    let k = coefs[0].nrows();
    (0..p)
        .map(|i| Mat::from_fn(k, k, |r, c| coefs[i][(r, c)] - delta * bias[i][(r, c)]))
        .collect()
}

/// Elementwise `delta * bias[i]`.
fn scale(bias: &[Mat<f64>], delta: f64) -> Vec<Mat<f64>> {
    let p = bias.len();
    let k = bias[0].nrows();
    (0..p)
        .map(|i| Mat::from_fn(k, k, |r, c| delta * bias[i][(r, c)]))
        .collect()
}

/// Spectral radius of the companion matrix of a VAR lag polynomial, from
/// the balanced eigenvalue solve ([`tsecon_linalg::balanced_eigenvalues`]),
/// so the Kilian shrink and `n_unstable_draws` do not depend on the units of
/// the series.
pub(crate) fn spectral_radius_of(coefs: &[Mat<f64>]) -> Result<f64, VarError> {
    let refs: Vec<MatRef<'_, f64>> = coefs.iter().map(Mat::as_ref).collect();
    let comp = companion_from_var(&refs)?;
    Ok(spectral_radius(comp.as_ref())?)
}

/// A refit failure on bootstrap pseudo-sample `rep`, as
/// [`VarError::Replication`]. `inner` marks the inner bias bootstrap of
/// `bias_correct` (whose refits compute only the slopes, so its
/// `NotPositiveDefinite` can only be `Z'Z`).
fn pseudo_sample_error(e: &VarError, rep: usize, inner: bool) -> VarError {
    let what = match (e, inner) {
        (VarError::NotPositiveDefinite { what }, false) if *what == SIGMA_U_WHAT => {
            "its refit's residual covariance Sigma_u is singular — the iid resample drew too \
             few distinct residual rows to span the k series, which has positive probability \
             when the residual degrees of freedom T_eff - m are close to k (the full sample \
             is fine); more observations or fewer lags make it vanishingly rare"
        }
        (VarError::NotPositiveDefinite { .. }, false) => {
            "its refit's regressor cross-product Z'Z is singular (the pseudo-sample's lagged \
             values are collinear)"
        }
        (VarError::NotPositiveDefinite { .. }, true) => {
            "it is a replication of the inner bias-estimation bootstrap of bias_correct=True, \
             and its refit's regressor cross-product Z'Z is singular (the pseudo-sample's \
             lagged values are collinear)"
        }
        (VarError::NonFinite { .. }, false) => {
            "its pseudo-sample overflowed to infinity (the VAR it is simulated from is \
             explosive)"
        }
        (VarError::NonFinite { .. }, true) => {
            "it is a replication of the inner bias-estimation bootstrap of bias_correct=True, \
             and its pseudo-sample overflowed to infinity (the fitted VAR is explosive)"
        }
        (_, false) => "its pseudo-sample's VAR refit failed",
        (_, true) => {
            "it is a replication of the inner bias-estimation bootstrap of bias_correct=True, \
             and its pseudo-sample's VAR refit failed"
        }
    };
    VarError::Replication { index: rep, what }
}

/// Map the resampling engine's error into the VAR error type. Only the
/// SeedSequence spawn limit can fire, and only for astronomically large
/// `n_boot`.
fn map_boot_err(_e: BootstrapError) -> VarError {
    VarError::InvalidArgument {
        what: "bootstrap resampling failed: n_boot exceeds the RNG substream \
               limit; use a smaller n_boot",
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn percentile_matches_numpy_linear() {
        // numpy.percentile([1,2,3,4], [25, 50, 75]) == [1.75, 2.5, 3.25]
        let x = [1.0, 2.0, 3.0, 4.0];
        assert!((percentile_sorted(&x, 0.25) - 1.75).abs() < 1e-12);
        assert!((percentile_sorted(&x, 0.50) - 2.5).abs() < 1e-12);
        assert!((percentile_sorted(&x, 0.75) - 3.25).abs() < 1e-12);
        // Endpoints are the extremes.
        assert_eq!(percentile_sorted(&x, 0.0), 1.0);
        assert_eq!(percentile_sorted(&x, 1.0), 4.0);
    }

    #[test]
    fn simulate_recursive_reproduces_original_from_own_residuals() {
        // With the fitted coefficients, the observed initial rows, and the
        // *original* (uncentered) residuals in order, the recursion must
        // reproduce the estimation sample exactly.
        let k = 2usize;
        let a = Mat::from_fn(k, k, |i, j| match (i, j) {
            (0, 0) => 0.5,
            (0, 1) => 0.1,
            (1, 0) => -0.2,
            (1, 1) => 0.3,
            _ => 0.0,
        });
        let intercept = [0.4, -0.2];
        // Build a sample from a known residual sequence, then check the
        // recursion inverts the fit's decomposition y = Zb + U.
        let te = 6usize;
        let u = Mat::from_fn(te, k, |t, j| 0.1 * (t as f64 + 1.0) - 0.05 * j as f64);
        let init = Mat::from_fn(1, k, |_, j| 1.0 + j as f64);
        let y = simulate_recursive(std::slice::from_ref(&a), &intercept, &init, &u);
        // Re-derive residuals from y and coefficients; must equal u.
        for t in 1..(1 + te) {
            for r in 0..k {
                let mut fitted = intercept[r];
                for c in 0..k {
                    fitted += a[(r, c)] * y[(t - 1, c)];
                }
                let resid = y[(t, r)] - fitted;
                assert!((resid - u[(t - 1, r)]).abs() < 1e-12);
            }
        }
    }

    /// Residual rows taking only two distinct values make EVERY pseudo-sample's
    /// refit residual covariance singular (rank <= 1 with an intercept and
    /// k = 2). The Kilian inner bootstrap needs only the slopes, so it must
    /// still succeed (it used to die on the refit's log-determinant); a full
    /// refit of the same pseudo-sample is refused as replication 0, naming
    /// the singular Sigma_u and not a duplicated column.
    #[test]
    fn inner_bias_bootstrap_survives_a_singular_pseudo_sample_covariance() {
        let k = 2usize;
        let a = Mat::from_fn(k, k, |i, j| [[0.5, 0.1], [-0.2, 0.3]][i][j]);
        let intercept = [0.4, -0.2];
        let init = Mat::from_fn(1, k, |_, j| 1.0 + j as f64);
        let te = 6usize;
        let uc = Mat::from_fn(te, k, |t, j| {
            let sign = if t % 2 == 0 { 1.0 } else { -1.0 };
            sign * [0.7, -0.3][j]
        });
        let bias = estimate_bias(std::slice::from_ref(&a), &intercept, &init, &uc, te, 1, 20)
            .expect("the inner bootstrap needs only the slopes");
        assert!(bias
            .iter()
            .all(|m| (0..k).all(|i| (0..k).all(|j| m[(i, j)].is_finite()))));

        // The same pseudo-sample, fully refitted, is a replication failure.
        let idx = vec![0usize, 1, 0, 1, 1, 0];
        let ustar = gather_rows(&uc, &idx);
        let ysim = simulate_recursive(std::slice::from_ref(&a), &intercept, &init, &ustar);
        let e = VarSpec {
            lags: 1,
            trend: Trend::Constant,
        }
        .fit(ysim.as_ref())
        .map_err(|e| pseudo_sample_error(&e, 7, false))
        .unwrap_err();
        assert!(matches!(e, VarError::Replication { index: 7, .. }), "{e:?}");
        let text = e.to_string();
        assert!(
            text.contains("replication 7") && text.contains("Sigma_u is singular"),
            "{text}"
        );
        assert!(!text.contains("redundant column"), "{text}");
    }

    #[test]
    fn center_columns_zeroes_the_means() {
        let u = Mat::from_fn(4, 2, |i, j| (i as f64) + 10.0 * j as f64);
        let c = center_columns(u.as_ref());
        for j in 0..2 {
            let m: f64 = (0..4).map(|i| c[(i, j)]).sum::<f64>();
            assert!(m.abs() < 1e-12);
        }
    }
}
