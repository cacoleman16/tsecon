//! Simultaneous (sup-t) confidence bands for local-projection impulse
//! responses.
//!
//! # What a band here is simultaneous *over*
//!
//! Every band this crate produced before this module was **pointwise**: at each
//! horizon `h` separately, `irf_h ± z * se_h` covers the true `beta_h` with
//! probability `1 - alpha`. That is not the statement a reader makes when they
//! look at a plotted impulse response. They read the *path* — "the response is
//! positive out to two years and back to zero by three" — which is a joint
//! statement about all `H + 1` horizons at once, and a pointwise band does not
//! support it. The shortfall is a multiplicity problem, not a consistency
//! problem: it does not shrink as `T` grows. The library's own interval-coverage
//! audit measured a nominal 90% pointwise IRF band containing the entire
//! 13-horizon path in 72.2% of samples at `T = 500`.
//!
//! Local projections are the cleanest place in the library to see the cost,
//! because the *pointwise* calibration is already good: the same audit put
//! `lp(se = "lag_augmented")` at 0.934 coverage against a nominal 0.95, the best
//! pointwise interval it measured. Whatever joint shortfall shows up in LP is
//! therefore multiplicity, not a mis-specified standard error.
//!
//! **The cell family used here is the horizons of one response.** For
//! [`lp_band`] and [`smooth_lp_band`] the `K` cells are `h = 0, 1, ..., H` of
//! the single impulse response the call estimates, in horizon order. Nothing
//! else is in the family: not a second shock, not a second outcome variable, not
//! the control coefficients. If you estimate two responses and want a band that
//! covers both paths jointly, `K` doubles and the band must be built over the
//! stacked family — this module does not do that for you, and running it twice
//! does **not** give you joint coverage over the union.
//!
//! # How the multiplier is chosen
//!
//! A simultaneous band keeps the pointwise standard errors and only widens the
//! multiplier, `irf_h ± c * se_h` with `c >= z`. The four routes come from
//! [`tsecon_stats::simultaneous`]:
//!
//! | [`BandMethod`] | needs | tightness |
//! |---|---|---|
//! | [`BandMethod::Pointwise`] | nothing | not simultaneous — the `z` baseline |
//! | [`BandMethod::SupT`] | the cross-horizon covariance | tightest |
//! | [`BandMethod::Sidak`] | nothing but `K` | loose |
//! | [`BandMethod::Bonferroni`] | nothing but `K` | loosest |
//!
//! The sup-t construction — take the maximum over cells of the absolute
//! t-statistic and read off its `1 - alpha` quantile — is the method of Montiel
//! Olea and Plagborg-Møller, *Simultaneous confidence bands: Theory,
//! implementation, and an application to SVARs*.
//!
//! # Where the cross-horizon covariance comes from
//!
//! Sup-t needs the *correlation across horizons*, and LP estimates every horizon
//! in a **separate regression**, so before this module the crate kept only the
//! per-horizon variances. [`lp_irf_cov`] builds the missing matrix. It is not a
//! new estimator and it does not touch the reported standard errors; it is the
//! same sandwich, written with the off-diagonal blocks filled in.
//!
//! By Frisch-Waugh-Lovell the horizon-`h` impulse coefficient has the influence
//! representation
//!
//! ```text
//!   beta_hat_h - beta_h  ~=  sum_t psi_{h,t},
//!   psi_{h,t} = xtilde_{h,t} * u_{h,t} / sum_s xtilde_{h,s}^2 ,
//! ```
//!
//! where `xtilde_h` is the horizon-`h` impulse column residualized on every
//! other regressor in that horizon's design (the constant, the lagged-`y`
//! controls, and — on the lag-augmented path — the impulse's own lags `1..=h`),
//! and `u_h` is the horizon-`h` regression residual. Squaring and summing
//! `psi_{h,t}` reproduces the HC0 variance of `beta_hat_h` exactly; taking
//! cross-products of `psi_h` and `psi_k` produces the cross-horizon entry the
//! crate was missing. Horizons use *different* samples (horizon `h` runs
//! `t = max(p, h) ..= n - 1 - h`), so `psi_{h,t}` is defined as zero outside
//! horizon `h`'s own sample; the covariance is then `sum_t psi_t psi_t'` over
//! one common time index, which is a sum of outer products and therefore
//! positive semi-definite by construction.
//!
//! Cost: [`lp_band`] with [`BandMethod::SupT`] runs [`lp`] once and then re-fits
//! every horizon twice inside [`lp_irf_cov`] (the horizon regression again, plus
//! the FWL residualization), so three OLS solves per horizon instead of one. The
//! duplication is deliberate — the returned path is literally `lp`'s output, so
//! no refactor of the estimator can drift the goldens. On top of that sit an
//! `O(M K^2 n)` accumulation, with `M` the Bartlett lag truncation (zero on the
//! default lag-augmented path), and the `O(n_sim * K^2)` Gaussian simulation.
//! Nothing here is paid unless [`BandMethod::SupT`] is requested; the
//! closed-form routes are free.
//!
//! ## Why the covariance is exactly right on the default path
//!
//! On [`SeSpec::LagAugmented`](crate::SeSpec::LagAugmented) the augmentation is
//! what makes this work. Montiel Olea & Plagborg-Møller (2021) show that adding
//! the impulse's own lags makes the horizon-`h` score serially uncorrelated, so
//! the *within*-horizon variance needs no kernel. The same argument kills the
//! cross-horizon autocovariances at non-zero lags, leaving only the
//! contemporaneous term `sum_t psi_{h,t} psi_{k,t}` — which is exactly what
//! [`lp_irf_cov`] accumulates. `sqrt(diag(Sigma))` then reproduces the reported
//! [`LpResult::se`](crate::LpResult::se) to floating-point noise —
//! `tests/simultaneous.rs` measures the worst relative gap at `1.3e-15` to
//! `1.8e-15` across three `(H, p)` configurations and asserts it stays under
//! `1e-10`. That check is what rules out the failure mode where the band's shape
//! and its widths silently come from different estimators.
//!
//! ## The HAC path needs one documented compromise
//!
//! On [`SeSpec::Hac`](crate::SeSpec::Hac) with the default
//! `maxlags = h + n_lag_controls`, **every horizon uses a different bandwidth**.
//! A stacked kernel estimator with per-cell bandwidths is not guaranteed
//! positive semi-definite, so [`lp_irf_cov`] uses a single common Bartlett
//! bandwidth for the whole matrix, `M = H + n_lag_controls` (the largest of the
//! per-horizon values), reported in [`LpIrfCov::bandwidth`]. Consequences:
//!
//! * `sqrt(diag(Sigma))` then does **not** equal the reported per-horizon `se`
//!   at `h < H` — it is the same estimator at a wider window.
//!   [`LpBand::cov_se_max_rel_diff`] reports how far apart they are; on the
//!   fixture at `H = 8`, `p = 4` the worst gap is 12%.
//! * The band is still `irf_h ± c * se_h` using the **reported** `se`. Only the
//!   multiplier comes from `Sigma`, and the multiplier depends on `Sigma` solely
//!   through its correlation matrix (rescaling `Sigma` by any positive diagonal
//!   leaves the sup-t quantile unchanged), so the widths remain the ones the
//!   goldens pin.
//! * Passing an explicit `maxlags` (`LpSpec::with_hac(Some(m))`) removes the
//!   compromise entirely: every horizon then already shares `m`, and the
//!   diagonal matches the reported `se` again.
//!
//! # What it buys, measured
//!
//! `tests/simultaneous.rs` runs a seeded Monte Carlo on a known-truth design
//! (`s_t = 0.9 s_{t-1} + e_t`, `y_t = s_t + w_t`, so the LP estimand is exactly
//! `0.9^h`), 400 replications, `K = 13` horizons, 4 lag controls, nominal 90%
//! bands, default lag-augmented inference. *Joint* coverage means the band
//! contained the **whole** 13-horizon path:
//!
//! | | `T = 240` | `T = 720` | mean `c` |
//! |---|---|---|---|
//! | pointwise | **0.365** | **0.427** | 1.645 |
//! | sup-t | 0.818 | **0.895** | 2.55 |
//! | Šidák | 0.843 | 0.910 | 2.649 |
//! | Bonferroni | 0.843 | 0.910 | 2.665 |
//!
//! Read three things off that table.
//!
//! 1. **The pointwise band is not a path statement.** A nominal 90% pointwise
//!    band held the whole path in 36.5% of samples, and *tripling the sample*
//!    moved that to 42.7%. It is not converging to 0.90, because nothing about
//!    multiplicity gets better with data.
//! 2. **Sup-t fixes it when the standard errors are right.** At `T = 720`, where
//!    the per-horizon marginal coverages sit on nominal (0.863-0.910), the sup-t
//!    joint rate is 0.895 against a nominal 0.90.
//! 3. **Sup-t inherits whatever the pointwise standard errors get wrong.** At
//!    `T = 240` the long-horizon marginals themselves run short (0.820-0.910
//!    against 0.90) and the sup-t joint rate is correspondingly short at 0.818.
//!    The multiplier corrects multiplicity; it cannot correct a finite-sample
//!    standard error. The union-bound routes score slightly higher there only
//!    because they over-pay by enough to cover the standard errors' shortfall by
//!    accident — that is not a reason to prefer them.
//!
//! The tightness gap on this design: at `K = 13`, `alpha = 0.10`, the fixture
//! sup-t value is 2.480 against Šidák's 2.649 and Bonferroni's 2.665, so sup-t
//! is about 6% narrower than either union bound and 51% wider than pointwise.
//! The saving is smaller than it would be on a smoother path — this design's
//! impact residual is pure measurement noise that no other horizon shares, so
//! the `h = 0` cell is only weakly correlated with the rest (adjacent-horizon
//! correlations run 0.16 to 0.95, mean 0.76).
//!
//! # What is *not* wired up
//!
//! [`lp_iv`](crate::lp_iv), [`lp_multiplier`](crate::lp_multiplier) and
//! [`lp_state`](crate::lp_state) have no cross-horizon covariance in this crate,
//! and this module does not fabricate one for them. Use
//! [`closed_form_band`] on their `irf`/`se` paths for a Šidák or Bonferroni
//! band, which needs nothing but `K`; those are honest, valid-under-any-
//! dependence fallbacks and are simply wider than a sup-t band would be.
//!
//! # Pointwise output is untouched
//!
//! [`lp_band`] calls [`lp`] and returns its result verbatim in
//! [`LpBandResult::lp`]; [`smooth_lp_band`] does the same with
//! [`smooth_lp`]. No point estimate and no standard error anywhere in the crate
//! changed, which is what keeps the statsmodels/linearmodels goldens valid.

use tsecon_hac::{ols, HacError, Kernel};
use tsecon_rng::Stream;
use tsecon_stats::simultaneous::{
    band as assemble_band, bonferroni_critical_value, pointwise_critical_value, required_uniforms,
    sidak_critical_value, sup_t_from_cov,
};

use crate::design::{check_finite, horizon_sample, outcome_column, single_impulse_design};
use crate::error::LpError;
use crate::level::lp;
use crate::smooth::{smooth_lp, SmoothLpResult, SmoothLpSpec};
use crate::spec::{LpResult, LpSpec, SeKind, SeSpec};

/// Default number of Gaussian simulations behind a [`BandMethod::SupT`]
/// critical value.
///
/// This is a quantile in the tail of a maximum, so it wants a large sample and
/// it is cheap: `O(n_sim * K^2)`, about 0.2 s at `K = 45` in a release build.
/// Do not go below ~50,000 in production.
pub const DEFAULT_BAND_N_SIM: usize = 100_000;

/// Default seed for the sup-t Gaussian simulation.
///
/// The band is a pure function of this seed, so a run is reproducible and two
/// runs with different seeds differ only by simulation noise. Callers that
/// expose bands to users should expose the seed too.
pub const DEFAULT_BAND_SEED: u64 = 20_260_807;

/// Default coverage level: `alpha = 0.10`, i.e. a 90% band, the usual
/// convention for plotted impulse responses.
pub const DEFAULT_BAND_ALPHA: f64 = 0.10;

/// Which multiplier is applied to the pointwise standard errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BandMethod {
    /// The ordinary pointwise `Phi^{-1}(1 - alpha/2)`. **Not simultaneous** —
    /// this is the status-quo band, kept as the default so that asking for a
    /// band never silently changes an existing plot.
    #[default]
    Pointwise,
    /// Sup-t (Montiel Olea & Plagborg-Møller): the `1 - alpha` quantile of the
    /// maximum absolute t-statistic over the horizons, simulated from the
    /// cross-horizon covariance built by [`lp_irf_cov`]. The tightest of the
    /// simultaneous routes.
    SupT,
    /// Šidák: pointwise at per-horizon level `1 - (1 - alpha)^(1/K)`. Needs only
    /// `K`. Exact under independence across horizons — a condition no impulse
    /// response meets — so in practice conservative.
    Sidak,
    /// Bonferroni: pointwise at per-horizon level `alpha / K`. Needs only `K`,
    /// valid under any dependence, and the loosest of the four.
    Bonferroni,
}

impl BandMethod {
    /// Stable snake-case label, for result objects and plot legends.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            BandMethod::Pointwise => "pointwise",
            BandMethod::SupT => "sup-t",
            BandMethod::Sidak => "sidak",
            BandMethod::Bonferroni => "bonferroni",
        }
    }

    /// Whether this method actually delivers joint coverage over the family.
    #[must_use]
    pub fn is_simultaneous(self) -> bool {
        !matches!(self, BandMethod::Pointwise)
    }
}

/// How to build the band around an already-estimated impulse response.
///
/// Construct with one of [`BandSpec::pointwise`], [`BandSpec::sup_t`],
/// [`BandSpec::sidak`], [`BandSpec::bonferroni`] and adjust the simulation
/// settings with [`BandSpec::with_n_sim`] / [`BandSpec::with_seed`]. `n_sim` and
/// `seed` are ignored by every method except [`BandMethod::SupT`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BandSpec {
    /// Which multiplier to use.
    pub method: BandMethod,
    /// Non-coverage level: `0.10` for a 90% band, `0.05` for 95%.
    pub alpha: f64,
    /// Gaussian simulations behind a sup-t critical value; see
    /// [`DEFAULT_BAND_N_SIM`].
    pub n_sim: usize,
    /// Seed for the sup-t simulation; see [`DEFAULT_BAND_SEED`].
    pub seed: u64,
}

impl Default for BandSpec {
    /// [`BandMethod::Pointwise`] at [`DEFAULT_BAND_ALPHA`] — the status quo.
    fn default() -> Self {
        BandSpec::pointwise(DEFAULT_BAND_ALPHA)
    }
}

impl BandSpec {
    /// A spec with the given method and level, and the default simulation
    /// settings.
    #[must_use]
    pub fn new(method: BandMethod, alpha: f64) -> Self {
        BandSpec {
            method,
            alpha,
            n_sim: DEFAULT_BAND_N_SIM,
            seed: DEFAULT_BAND_SEED,
        }
    }

    /// The pointwise (non-simultaneous) band at level `alpha`.
    #[must_use]
    pub fn pointwise(alpha: f64) -> Self {
        BandSpec::new(BandMethod::Pointwise, alpha)
    }

    /// The sup-t band at level `alpha`.
    #[must_use]
    pub fn sup_t(alpha: f64) -> Self {
        BandSpec::new(BandMethod::SupT, alpha)
    }

    /// The Šidák band at level `alpha`.
    #[must_use]
    pub fn sidak(alpha: f64) -> Self {
        BandSpec::new(BandMethod::Sidak, alpha)
    }

    /// The Bonferroni band at level `alpha`.
    #[must_use]
    pub fn bonferroni(alpha: f64) -> Self {
        BandSpec::new(BandMethod::Bonferroni, alpha)
    }

    /// Builder: set the number of Gaussian simulations for the sup-t route.
    #[must_use]
    pub fn with_n_sim(mut self, n_sim: usize) -> Self {
        self.n_sim = n_sim;
        self
    }

    /// Builder: set the seed for the sup-t simulation.
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
}

/// The cross-horizon covariance matrix of an LP impulse-response path.
///
/// This is the object the crate did not have before: LP fits every horizon in
/// its own regression, so only the diagonal of this matrix was ever formed. See
/// the module docs for the influence-function construction and, on the HAC path,
/// for the common-bandwidth compromise.
#[derive(Debug, Clone, PartialEq)]
pub struct LpIrfCov {
    /// Horizons in cell order, `[0, 1, ..., H]`. Row/column `i` of
    /// [`LpIrfCov::cov`] is horizon `horizons[i]`.
    pub horizons: Vec<usize>,
    /// The `K x K` covariance, **row-major**: `Sigma_ij = cov[i * K + j]`,
    /// `K = horizons.len()`. Symmetric and positive semi-definite.
    pub cov: Vec<f64>,
    /// `sqrt(diag(cov))`. Equal to the reported
    /// [`LpResult::se`](crate::LpResult::se) to floating-point noise on the
    /// lag-augmented path; see the module docs for when the HAC path differs.
    pub se: Vec<f64>,
    /// The single Bartlett lag truncation used for the whole matrix. `0.0` on
    /// the lag-augmented path (the score is serially uncorrelated there, so no
    /// kernel is needed).
    pub bandwidth: f64,
    /// Which standard-error construction the matrix corresponds to.
    pub se_kind: SeKind,
}

/// A confidence band over the horizons of one impulse response.
#[derive(Debug, Clone, PartialEq)]
pub struct LpBand {
    /// Which multiplier was applied. Report this next to the band: a band is
    /// not interpretable without knowing whether it is pointwise.
    pub method: BandMethod,
    /// The non-coverage level the band was built at.
    pub alpha: f64,
    /// The multiplier `c` in `irf_h ± c * se_h`.
    pub critical_value: f64,
    /// The pointwise `Phi^{-1}(1 - alpha/2)` for comparison. The ratio
    /// `critical_value / pointwise_critical_value` is exactly what simultaneity
    /// costs on this path.
    pub pointwise_critical_value: f64,
    /// Lower edge of the band, one entry per horizon.
    pub lower: Vec<f64>,
    /// Upper edge of the band, one entry per horizon.
    pub upper: Vec<f64>,
    /// `K`: the number of cells the band covers — here, the number of horizons.
    /// Report it: the same `alpha` over a different family is a different band.
    pub n_cells: usize,
    /// Cells with a strictly positive standard error, i.e. those that actually
    /// entered the maximum. Equals `n_cells` for LP (no horizon is pinned by a
    /// normalization).
    pub n_cells_used: usize,
    /// Gaussian simulations behind the critical value; `0` for every route
    /// except [`BandMethod::SupT`].
    pub n_sim: usize,
    /// Seed behind the critical value; meaningful only for
    /// [`BandMethod::SupT`].
    pub seed: u64,
    /// The cross-horizon covariance the sup-t multiplier came from; `None` for
    /// the closed-form and pointwise routes, which never build one.
    pub cov: Option<LpIrfCov>,
    /// The largest relative gap between `sqrt(diag(cov))` and the reported
    /// standard errors the band is actually built from, `max_h |se_cov_h -
    /// se_h| / se_h`. `None` when no covariance was built.
    ///
    /// Near machine epsilon on the lag-augmented path — that is the check that
    /// the covariance and the reported standard errors are the same estimator.
    /// Materially non-zero on the HAC path with the default horizon-growing
    /// `maxlags`, for the documented reason (a single common bandwidth), and
    /// harmless there because the multiplier only uses the correlation matrix.
    pub cov_se_max_rel_diff: Option<f64>,
}

/// [`lp`] plus a band over its horizons.
#[derive(Debug, Clone, PartialEq)]
pub struct LpBandResult {
    /// The local-projection result, **bit-identical** to calling [`lp`] with
    /// the same arguments — this function calls it and passes the result
    /// through untouched.
    pub lp: LpResult,
    /// The band over `lp.irf` / `lp.se`.
    pub band: LpBand,
}

/// [`smooth_lp`] plus a band over its horizons.
#[derive(Debug, Clone, PartialEq)]
pub struct SmoothLpBandResult {
    /// The smooth-LP result, **bit-identical** to calling [`smooth_lp`] with
    /// the same arguments.
    pub smooth: SmoothLpResult,
    /// The band over `smooth.irf` / `smooth.se`.
    pub band: LpBand,
}

// ---------------------------------------------------------------------------
// Entry points
// ---------------------------------------------------------------------------

/// Estimate a local projection and put a band — pointwise or simultaneous —
/// around its horizon path.
///
/// The band's cell family is **the horizons `0..=spec.horizons` of the single
/// response this call estimates**, and nothing else. `spec` behaves exactly as
/// it does for [`lp`]; `band` selects the multiplier.
///
/// [`BandMethod::SupT`] builds the cross-horizon covariance with
/// [`lp_irf_cov`] (one extra OLS per horizon) and simulates
/// `band.n_sim` Gaussian draws from a `band.seed`-seeded Philox stream, so the
/// band is a pure function of the seed. The closed-form routes need nothing but
/// `K` and cost nothing.
///
/// ```
/// use tsecon_lp::{lp, lp_band, BandMethod, BandSpec, LpSpec};
///
/// let n = 400;
/// let mut state = 1u64;
/// let mut draw = || {
///     state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
///     (state >> 33) as f64 / (1u64 << 30) as f64 - 1.0
/// };
/// let shock: Vec<f64> = (0..n).map(|_| draw()).collect();
/// let noise: Vec<f64> = (0..n).map(|_| draw()).collect();
/// let mut y = vec![0.0; n];
/// for t in 1..n {
///     y[t] = 0.7 * y[t - 1] + shock[t] + 0.5 * noise[t];
/// }
///
/// let spec = LpSpec::new(8, 4);
/// let plain = lp(&y, &shock, spec).unwrap();
/// let out = lp_band(&y, &shock, spec, BandSpec::sup_t(0.10).with_n_sim(20_000)).unwrap();
///
/// // The point estimates and standard errors are untouched.
/// assert_eq!(out.lp.irf, plain.irf);
/// assert_eq!(out.lp.se, plain.se);
///
/// // Only the multiplier changed, and it can only have grown.
/// assert_eq!(out.band.method, BandMethod::SupT);
/// assert_eq!(out.band.n_cells, 9);
/// assert!(out.band.critical_value >= out.band.pointwise_critical_value);
/// ```
///
/// # Errors
///
/// Everything [`lp`] can return, plus [`LpError::Band`] for an invalid `alpha`
/// or `n_sim`, or if the assembled covariance is rejected by the shared sup-t
/// routine.
pub fn lp_band(
    y: &[f64],
    shock: &[f64],
    spec: LpSpec,
    band: BandSpec,
) -> Result<LpBandResult, LpError> {
    let res = lp(y, shock, spec)?;
    let cov = match band.method {
        BandMethod::SupT => Some(lp_irf_cov(y, shock, spec)?),
        _ => None,
    };
    let b = build_band(&res.irf, &res.se, band, cov)?;
    Ok(LpBandResult { lp: res, band: b })
}

/// Estimate a smooth local projection and put a band around its horizon path.
///
/// Smooth LP is the one estimator in this crate that already had the full
/// cross-horizon covariance: the IRF is `irf_h = B_h' theta` for a single
/// jointly-estimated spline coefficient vector, so the delta-method covariance
/// of the whole path is `B V B'` and is reported in
/// [`SmoothLpResult::cov`]. [`BandMethod::SupT`] here therefore needs no extra
/// estimation and no compromise: `sqrt(diag(cov))` **is** the reported
/// [`SmoothLpResult::se`], bit for bit.
///
/// The usual smooth-LP caveat still applies and is not a band problem: those
/// standard errors are conditional on the chosen `lambda` and ignore the
/// penalty's shrinkage bias, so the band is centred on a shrunk estimator.
///
/// # Errors
///
/// Everything [`smooth_lp`] can return, plus [`LpError::Band`] as in
/// [`lp_band`].
pub fn smooth_lp_band(
    y: &[f64],
    shock: &[f64],
    spec: &SmoothLpSpec,
    band: BandSpec,
) -> Result<SmoothLpBandResult, LpError> {
    let res = smooth_lp(y, shock, spec)?;
    let cov = match band.method {
        BandMethod::SupT => Some(LpIrfCov {
            horizons: res.horizons.clone(),
            se: res.se.clone(),
            cov: res.cov.clone(),
            bandwidth: spec
                .hac_maxlags
                .unwrap_or(spec.horizons + spec.n_lag_controls) as f64,
            se_kind: res.se_kind,
        }),
        _ => None,
    };
    let b = build_band(&res.irf, &res.se, band, cov)?;
    Ok(SmoothLpBandResult {
        smooth: res,
        band: b,
    })
}

/// A band from a point estimate and its pointwise standard errors, using only
/// the closed-form routes.
///
/// This is the escape hatch for the estimators that have no cross-horizon
/// covariance in this crate — [`lp_iv`](crate::lp_iv),
/// [`lp_multiplier`](crate::lp_multiplier),
/// [`lp_state`](crate::lp_state) — and for any other path a caller assembles
/// itself. Šidák and Bonferroni need nothing but `K = theta_hat.len()`, are
/// valid under arbitrary dependence across cells, and are simply wider than a
/// sup-t band would be.
///
/// The cell family is whatever you pass in, in the order you pass it. That is a
/// user-visible choice: every cell you add widens the band for every other cell.
///
/// # Errors
///
/// [`LpError::Band`] if `spec.method` is [`BandMethod::SupT`] (there is no
/// covariance to simulate from — call [`lp_band`] or [`smooth_lp_band`]
/// instead), if `alpha` is out of range, or if the inputs disagree in length or
/// contain non-finite values.
pub fn closed_form_band(theta_hat: &[f64], se: &[f64], spec: BandSpec) -> Result<LpBand, LpError> {
    if spec.method == BandMethod::SupT {
        return Err(LpError::Band {
            what: "sup-t was requested without a covariance matrix; closed_form_band only \
                   serves the pointwise/sidak/bonferroni routes — use lp_band or \
                   smooth_lp_band for sup-t, or pick BandMethod::Sidak here",
        });
    }
    build_band(theta_hat, se, spec, None)
}

/// The cross-horizon covariance matrix of the [`lp`] impulse-response path.
///
/// This is the matrix a sup-t band needs and that per-horizon LP does not
/// otherwise produce. See the module docs for the Frisch-Waugh-Lovell influence
/// representation it is built from, and for the common-bandwidth compromise the
/// HAC path requires.
///
/// The returned matrix is symmetric by construction and positive semi-definite:
/// on the lag-augmented path it is literally `sum_t psi_t psi_t'`, and on the
/// HAC path it is a Bartlett Newey-West estimator at a single bandwidth, whose
/// Fejér-kernel representation is non-negative.
///
/// # Errors
///
/// The same input and horizon errors as [`lp`] (including
/// [`LpError::InvalidSeForCumulation`] for lag-augmented inference under
/// [`Cumulation::Both`](crate::Cumulation::Both)), plus
/// [`HacError::SingularDesign`] (wrapped in [`LpError::Hac`]) if a horizon's
/// impulse column is fully explained by the other regressors, which would make
/// the influence function `0/0`.
pub fn lp_irf_cov(y: &[f64], shock: &[f64], spec: LpSpec) -> Result<LpIrfCov, LpError> {
    spec.check_se_supports_cumulation()?;
    if shock.len() != y.len() {
        return Err(LpError::LengthMismatch {
            what: "impulse (shock) vs outcome (y)",
            expected: y.len(),
            got: shock.len(),
        });
    }
    check_finite(y, "outcome (y)")?;
    check_finite(shock, "impulse (shock)")?;

    let n = y.len();
    let p = spec.n_lag_controls;
    if n <= p {
        return Err(LpError::SeriesTooShort {
            n,
            n_lag_controls: p,
        });
    }

    let k_cells = spec.horizons + 1;
    // psi[h] lives on the *original* time index and is zero outside horizon
    // h's own regression sample, so horizons with different samples can be
    // cross-multiplied on one common index without any alignment bookkeeping.
    let mut psi = vec![vec![0.0_f64; n]; k_cells];
    // sqrt of the HC1/statsmodels n/(n - k) inflation, applied as
    // D Sigma D so the diagonal keeps matching the reported standard errors
    // and the matrix stays positive semi-definite.
    let mut dof_root = vec![1.0_f64; k_cells];
    let mut bandwidth = 0.0_f64;

    for h in 0..=spec.horizons {
        let n_shock_lags = match spec.se {
            SeSpec::LagAugmented => h,
            SeSpec::Hac { .. } => 0,
        };
        let (start, nobs) = horizon_sample(n, h, p, n_shock_lags);
        let nparams = 2 + p + n_shock_lags;
        if nobs <= nparams {
            return Err(LpError::HorizonTooLong {
                horizon: h,
                nobs,
                nparams,
            });
        }

        let response = outcome_column(y, h, start, nobs, spec.cumulation.accumulates_outcome());
        let cols = single_impulse_design(
            y,
            shock,
            h,
            start,
            nobs,
            p,
            n_shock_lags,
            spec.cumulation.accumulates_impulse(),
        );

        let fit = ols(&response, &cols)?;
        // Frisch-Waugh-Lovell: residualize the impulse column (column 0) on
        // the constant, the lag controls, and the impulse lags. The influence
        // of observation t on beta_h is then xtilde_t * u_t / sum_s xtilde_s^2.
        let x_fit = ols(&cols[0], &cols[1..])?;
        let ssr: f64 = x_fit.residuals.iter().map(|v| v * v).sum();
        if !ssr.is_finite() || ssr <= 0.0 {
            return Err(LpError::Hac(HacError::SingularDesign {
                what: "local-projection impulse column residualized to zero against its own \
                       controls, so the cross-horizon influence function is undefined",
            }));
        }
        for (i, (&xt, &u)) in x_fit.residuals.iter().zip(&fit.residuals).enumerate() {
            psi[h][start + i] = xt * u / ssr;
        }

        dof_root[h] = (nobs as f64 / (nobs - nparams) as f64).sqrt();
        if let SeSpec::Hac { maxlags } = spec.se {
            let ml = maxlags.unwrap_or(h + p) as f64;
            if ml > bandwidth {
                bandwidth = ml;
            }
        }
    }

    // Bartlett-weighted stacked meat. On the lag-augmented path bandwidth is
    // 0, so only the lag-0 term survives and the matrix is sum_t psi_t psi_t'.
    let max_lag = (bandwidth.floor() as usize).min(n.saturating_sub(1));
    let mut cov = vec![0.0_f64; k_cells * k_cells];
    for lag in 0..=max_lag {
        let w = Kernel::Bartlett.weight(lag, bandwidth);
        if lag > 0 && w == 0.0 {
            break;
        }
        for i in 0..k_cells {
            for j in 0..k_cells {
                let mut g = 0.0_f64;
                for t in lag..n {
                    g += psi[i][t] * psi[j][t - lag];
                }
                if lag == 0 {
                    cov[i * k_cells + j] += g;
                } else {
                    cov[i * k_cells + j] += w * g;
                    cov[j * k_cells + i] += w * g;
                }
            }
        }
    }

    // Degrees-of-freedom inflation, then an exact symmetrization: the lag > 0
    // accumulation above adds the same two terms to (i, j) and (j, i) in
    // opposite order, which can differ in the last bit, and the shared sup-t
    // routine checks symmetry.
    for i in 0..k_cells {
        for j in 0..k_cells {
            cov[i * k_cells + j] *= dof_root[i] * dof_root[j];
        }
    }
    for i in 0..k_cells {
        for j in (i + 1)..k_cells {
            let v = 0.5 * (cov[i * k_cells + j] + cov[j * k_cells + i]);
            cov[i * k_cells + j] = v;
            cov[j * k_cells + i] = v;
        }
    }

    let mut se = Vec::with_capacity(k_cells);
    for i in 0..k_cells {
        let v = cov[i * k_cells + i];
        if v < 0.0 {
            return Err(LpError::Hac(HacError::NumericalBreakdown {
                what: "cross-horizon local-projection covariance diagonal",
            }));
        }
        se.push(v.sqrt());
    }

    let se_kind = match spec.se {
        SeSpec::LagAugmented => SeKind::LagAugmentedHc1,
        SeSpec::Hac { .. } => SeKind::HacBartlett,
    };

    Ok(LpIrfCov {
        horizons: (0..=spec.horizons).collect(),
        cov,
        se,
        bandwidth,
        se_kind,
    })
}

// ---------------------------------------------------------------------------
// Internals
// ---------------------------------------------------------------------------

/// Turn a point estimate, its pointwise standard errors, and (for sup-t) a
/// covariance matrix into an [`LpBand`].
fn build_band(
    theta_hat: &[f64],
    se: &[f64],
    spec: BandSpec,
    cov: Option<LpIrfCov>,
) -> Result<LpBand, LpError> {
    let k = theta_hat.len();
    if se.len() != k {
        return Err(LpError::LengthMismatch {
            what: "band point estimate vs standard errors",
            expected: k,
            got: se.len(),
        });
    }
    if !(spec.alpha > 0.0 && spec.alpha < 1.0) {
        return Err(LpError::Band {
            what: "band alpha must satisfy 0 < alpha < 1 (0.10 for a 90% band, \
                   0.05 for a 95% band)",
        });
    }
    let z = pointwise_critical_value(spec.alpha).map_err(band_err)?;

    let (critical_value, n_sim) = match spec.method {
        BandMethod::Pointwise => (z, 0),
        BandMethod::Sidak => (sidak_critical_value(spec.alpha, k).map_err(band_err)?, 0),
        BandMethod::Bonferroni => (
            bonferroni_critical_value(spec.alpha, k).map_err(band_err)?,
            0,
        ),
        BandMethod::SupT => {
            let Some(c) = cov.as_ref() else {
                return Err(LpError::Band {
                    what: "sup-t needs the cross-horizon covariance and none was built",
                });
            };
            if spec.n_sim < 2 {
                return Err(LpError::Band {
                    what: "sup-t needs n_sim >= 2 Gaussian simulations, and at least ~50,000 \
                           in production: the critical value is a quantile in the tail of a \
                           maximum",
                });
            }
            let mut uniforms = vec![0.0_f64; required_uniforms(k, spec.n_sim)];
            Stream::new(spec.seed).fill_uniform_f64(&mut uniforms);
            let cv = sup_t_from_cov(&c.cov, k, spec.alpha, &uniforms).map_err(band_err)?;
            (cv, spec.n_sim)
        }
    };

    let assembled = assemble_band(theta_hat, se, critical_value).map_err(band_err)?;

    // How far the covariance's own standard errors are from the ones the band
    // is actually built on. Machine noise on the lag-augmented path; see the
    // module docs for the HAC path.
    let cov_se_max_rel_diff = cov.as_ref().map(|c| {
        c.se.iter()
            .zip(se)
            .map(|(a, b)| if *b > 0.0 { (a - b).abs() / b } else { 0.0 })
            .fold(0.0_f64, f64::max)
    });

    Ok(LpBand {
        method: spec.method,
        alpha: spec.alpha,
        critical_value: assembled.critical_value,
        pointwise_critical_value: z,
        lower: assembled.lower,
        upper: assembled.upper,
        n_cells: k,
        n_cells_used: assembled.n_cells_used,
        n_sim,
        seed: spec.seed,
        cov,
        cov_se_max_rel_diff,
    })
}

/// Map a `tsecon-stats` rejection into the crate's error type. The shared
/// routine's own message is not carried through because [`LpError`] is
/// `'static`-message-shaped; the variants it can raise here are all
/// caller-configuration problems, so the text names them directly.
fn band_err(e: tsecon_stats::StatsError) -> LpError {
    match e {
        tsecon_stats::StatsError::Domain { name: "sigma", .. } => LpError::Band {
            what: "the cross-horizon covariance was rejected as asymmetric or not \
                   positive semi-definite; this is an internal invariant of lp_irf_cov \
                   and should be reported as a bug",
        },
        tsecon_stats::StatsError::Domain { name: "se", .. } => LpError::Band {
            what: "every standard error in the band family must be finite and >= 0, and \
                   at least one must be strictly positive",
        },
        _ => LpError::Band {
            what: "the simultaneous-band routine rejected its arguments; check alpha \
                   (0 < alpha < 1), n_sim (>= 2), and that the response path is non-empty",
        },
    }
}
