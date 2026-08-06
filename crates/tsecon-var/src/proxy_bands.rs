//! Jentsch-Lunsford **moving-block bootstrap** confidence bands for proxy
//! SVARs (SVAR-IV), plus the invalid-but-reachable wild bootstrap that
//! Mertens-Ravn (2013) and Gertler-Karadi (2015) actually report.
//!
//! [`tsecon_ident::proxy_svar`] returns a point estimate and nothing else.
//! The obvious way to band it — the recursive wild bootstrap of the two
//! papers above — is **not asymptotically valid for this estimand**, which
//! is the point of Jentsch and Lunsford (2019). This module implements the
//! moving-block (MBB) alternative and, behind an explicit flag, the wild
//! arm for reproducing published bands.
//!
//! # Why the wild bootstrap fails here
//!
//! Identification runs entirely through the contemporaneous cross-moment
//! `gamma = E[m_t u_t'] = phi * h_col`. Apply a common Rademacher draw
//! `e_t in {-1, +1}` to the residual **and** the proxy, as the shipped
//! replication codes do, and
//!
//! ```text
//! m*_t u*_t' = (e_t m_t)(e_t uhat_t)' = e_t^2 m_t uhat_t' = m_t uhat_t'
//! ```
//!
//! *pointwise*. The identifying moment is therefore **bit-identical in
//! every draw** (measured here: 200/200 draws, max deviation exactly
//! `0.000e+00`; see `proxy_bands_props.rs::wild_rademacher_freezes_the_
//! identifying_moment`), so the sampling uncertainty of the identification
//! step is simply missing from the bands. The symptom is intervals that are
//! too short at every horizon, with a deficit that does **not** vanish as
//! `T` grows — a first-order failure, not a finite-sample one.
//!
//! # What the moving-block bootstrap does instead
//!
//! The resampling unit is the **joint pair** `z_t = (uhat_t', m_t')'`. One
//! set of block starts indexes the residual columns and the proxy columns
//! simultaneously, so the time-`t` pairing survives resampling and the
//! bootstrap reproduces the sampling variability of the cross-moment. Draw
//! separate indices for `u` and `m` (or leave `m` unresampled) and the
//! bootstrap DGP has `E*[m*_t u*_t'] = 0`: you would be sampling under the
//! null of **zero instrument relevance**, `rho*` becomes a ratio of two
//! mean-zero noise terms with no finite moments, and the bands are not
//! slightly wrong but a different object entirely. Measured on synthetic
//! data with a known proxy: mean `gamma*[norm_var]` of `+0.0006` against a
//! sample value of `+0.5557`, with the denominator changing sign in 50.7%
//! of draws versus 0.0% under joint blocking.
//!
//! The algorithm, following Jentsch and Lunsford (2019) and, for the
//! VAR-side block machinery, Brüggemann, Jentsch and Trenkler (2016) and
//! Künsch (1989):
//!
//! 1. stack `(uhat_t, m_t)` and draw `N = ceil(T / ell)` blocks of length
//!    `ell` from the `T - ell + 1` **overlapping** candidate starts, with
//!    replacement, laid end to end and truncated to exactly `T` rows;
//! 2. subtract the **position-specific** (Künsch/BJT) means: for
//!    within-block position `s`, `ubar_s` is the average of `uhat_{i+s-1}`
//!    over all candidate starts `i`, and likewise `mbar_s` over the finite
//!    proxy entries. See [`position_centering`] — and note that centering
//!    by the *grand* mean instead is a silent no-op, since OLS with an
//!    intercept already forces `sum_t uhat_t = 0` (measured
//!    `max|mean(uhat)|` = `1.5e-16`, against position-wise means of
//!    `4.19e-02`, some 7.1% of the residual standard deviation);
//! 3. reconstruct `y*` **recursively** from the fitted coefficients, with
//!    the `p` actual observed presample rows as initial conditions and no
//!    burn-in (burn-in would simulate the stationary distribution of the
//!    *estimated* VAR, which is ill-defined and numerically explosive when
//!    the estimated system has roots near unity — as the tax and monetary
//!    VARs this method is used on do);
//! 4. **re-estimate** the identical specification by plain OLS on `y*`, so
//!    the bands carry the reduced-form coefficient uncertainty as well;
//! 5. recompute `gamma*` from the re-estimated residuals paired with the
//!    resampled proxy, and **re-impose the unit-effect normalization inside
//!    the draw**: `rho* = gamma* / gamma*[norm_var]`, `b* = unit * rho*`;
//! 6. propagate `theta*_h = Psi*_h b*` and take quantiles.
//!
//! Step 5 is mandatory rather than cosmetic, and it has a free self-test:
//! `b*[norm_var] == unit` **exactly** in every draw, so the `h = 0` band
//! for `norm_var` is degenerate at `unit` with zero width. A non-degenerate
//! cell there proves the normalization was hoisted out of the loop. No
//! sign-fixing rule is applied: when `gamma*[norm_var]` changes sign the
//! whole response flips, and that is the correct handling of a genuinely
//! bimodal ratio — truncating one lobe would visibly narrow the bands.
//!
//! # Failed draws are counted, never dropped
//!
//! A draw fails when the resampled proxy has too few finite entries, no
//! variance, or a `gamma*[norm_var]` at the floating-point floor. Those are
//! exactly the near-zero-denominator tail of the ratio distribution, so
//! silently discarding them *shrinks* the interval. [`ProxyBands`] reports
//! `n_failed`, the per-reason [`BandFailures`] breakdown, and a warning
//! when failures exceed 1% of `n_boot`.
//!
//! # Intervals
//!
//! Both are returned. [`ProxyBands::lower`]/[`ProxyBands::upper`] are
//! **Hall's** (basic, reverse-percentile) interval, which reflects the
//! bootstrap *deviations* around the point estimate;
//! [`ProxyBands::lower_efron`]/[`ProxyBands::upper_efron`] are the plain
//! Efron percentile interval, which is what Mertens-Ravn and Gertler-Karadi
//! report. The two differ materially when the bootstrap distribution is
//! skewed or off-center, which is the normal case for a ratio estimand.
//! **Unverified:** that Jentsch and Lunsford specifically recommend the Hall
//! form (on the grounds that block-bootstrap draws are not centered at the
//! point estimate) is a recollection that has not been checked against the
//! paper; both are exposed so the choice is the caller's and is visible.
//! Neither a normal-approximation nor a delta-method interval is offered.
//!
//! # Measured coverage
//!
//! Nominal 90% Hall bands on a known-truth proxy-SVAR DGP (VAR(1), `n = 2`,
//! `T = 299`, `ell = 21`, `B = 199`, 150 replications, Monte-Carlo standard
//! error ~0.025; `proxy_bands_props.rs`):
//!
//! ```text
//!                    h = 0  h = 1  h = 2  h = 3  h = 4
//! moving block       0.860  0.847  0.793  0.780  0.807
//! wild               0.113  0.827  0.847  0.833  0.820
//! Cholesky reference 0.927  0.893  0.840  0.833  0.833
//! ```
//!
//! Three things this says, none of which should be sanded off:
//!
//! * **The wild bootstrap collapses at impact** — 0.113 against a nominal
//!   0.90, with a mean width of 0.018 against the moving block's 0.173. At
//!   `h = 0` the identification step *is* the entire variance, and the
//!   common Rademacher draw freezes it.
//! * **At `h >= 1` the wild arm is not uniformly worse on this DGP.** Once
//!   reduced-form coefficient uncertainty enters it dominates, and the two
//!   arms are comparable. The horizon profile of the wild bootstrap's
//!   under-coverage is DGP-dependent and no universal direction is claimed
//!   here; the impact-horizon collapse is the robust part, and impact is
//!   where proxy SVARs are usually read.
//! * **The moving block's own shortfall at longer horizons is consistent
//!   with the reduced-form VAR bootstrap's**, and the Cholesky row is what
//!   bounds that. Read it carefully, because it is **not** a controlled
//!   comparison. It holds the *estimand* fixed — the test DGP's `H` is lower
//!   triangular with `H[0][0] = 1`, so recursive and proxy identification
//!   target one population quantity — and it holds the replications fixed.
//!   It does **not** hold the procedure fixed: [`crate::bootstrap_irf_bands`]
//!   reports **Efron** percentile bands from an **i.i.d. residual**
//!   bootstrap, while the proxy row is **Hall** from a **moving block**. Its
//!   0.927/0.893/0.840/0.833/0.833 therefore differ from the proxy row in
//!   three things at once (identification layer, interval type, resampling
//!   scheme), and the ~0.07 gap **cannot be attributed to the identification
//!   layer**. What the row does license is the weaker, still useful claim
//!   that a shortfall of this size at these horizons is already present in
//!   this crate's validated reduced-form bootstrap on the same data, so the
//!   proxy layer need not be its cause. Isolating the identification layer
//!   would need a fourth arm (Hall + moving block + recursive
//!   identification) that is not implemented. Independently, this library's
//!   coverage audit records the residual VAR bootstrap at 0.848 for impact
//!   and 0.410 at `h = 12` on a persistent VAR without a bias correction.
//!   These bands offer **no bias correction**; that is the cost, and it is
//!   stated rather than tuned away.
//!
//! Left and right non-coverage are measured separately, because they are not
//! symmetric: the right-hand miss rate rises from 0.067 at impact to 0.147
//! at `h = 4` while the left-hand rate does not, which is the skew of a
//! ratio estimand showing through. A two-sided total alone would hide it.
//!
//! At `T = 99` the block-length rule returns `ell = 16` — a sixth of the
//! sample, only seven blocks — and impact coverage falls to 0.78 against
//! 0.83 at `ell = 4`. That is a real property of the method at short
//! samples, not a defect, and it is the reason
//! [`proxy_svar_band_block_sensitivity`] is part of the API rather than an
//! optional extra.
//!
//! # Unverified details (do not repeat these as checked facts)
//!
//! * the block-length constant in [`default_block_length`],
//!   `ell = round(5.03 * T^{1/4})`, is a recollection of Jentsch-Lunsford's
//!   rule; only the *rate* requirement (`ell -> inf`, `ell/T -> 0`) is not
//!   in doubt. [`proxy_svar_band_block_sensitivity`] exists so the caller
//!   can see how much the choice matters on their data;
//! * Jentsch and Lunsford are recalled to apply a deterministic **scale**
//!   adjustment to the resampled proxy on top of the centering. Its
//!   algebraic form is not known here and is therefore **not implemented**.
//!   This is safe for the bands and only for the bands: any positive scalar
//!   on `m*` cancels exactly in `rho* = gamma*_j / gamma*_norm`, so it
//!   cannot move an impulse response — it can only shift the reported
//!   bootstrap `F*` and `reliability*`, which is why those are labelled
//!   diagnostics here;
//! * whether Jentsch and Lunsford center the **proxy** position-wise as well
//!   as the residuals. This module does: `mbar_s` is subtracted from `m*`
//!   exactly as `ubar_s` is from `u*` (see [`position_centering`]).
//!   Confidence is high that the *residuals* get the position-wise
//!   treatment and lower that `m_t` gets the identical one. The stake is
//!   small but real: a *constant* shift of `m*` is annihilated by the moment
//!   (the re-estimated residuals are demeaned over the overlap), so only the
//!   position-**dependent** part of `mbar_s` can move anything, and that is
//!   an `O(ell / T)` end effect;
//! * the block count and the truncation. This module lays `N = ceil(T/ell)`
//!   blocks end to end and cuts the concatenation to **exactly `T`** rows
//!   (see [`block_indices_from_starts`]). Whether Jentsch and Lunsford
//!   instead retain all `N * ell` rows is **not verified**, and the choice
//!   is not free: keeping `N * ell` rescales the bootstrap distribution by
//!   `sqrt(T / (N * ell))`, which is a persistent coverage offset rather
//!   than Monte-Carlo error — it does not shrink as `n_boot` grows;
//! * that the presample rows are fixed at their actual observed values with
//!   no burn-in is inherited from Brüggemann-Jentsch-Trenkler (2016) rather
//!   than verified against Jentsch-Lunsford's own text;
//! * that explosive `Ahat*` draws are not screened is inferred from the
//!   design (a screen would break the bootstrap's job of replicating the
//!   estimator's finite-sample behaviour), not read;
//! * the `k > 1` (multi-instrument) case is **not implemented**: this
//!   module is single-proxy, single-target-shock, like
//!   [`tsecon_ident::proxy_svar`] itself.
//!
//! These asymptotics are **strong-instrument** asymptotics. The MBB is not
//! a weak-instrument fix; when `gamma*[norm_var]` comes close to zero (watch
//! [`ProxyBands::gamma_norm_draws`]) the unit-effect object is itself badly
//! behaved and a weak-IV-robust set (Montiel Olea, Stock and Watson 2021) is
//! the right tool, not a wider bootstrap band. The bands are also
//! **pointwise**: they are not a joint band over the horizon path.
//!
//! # References
//!
//! Jentsch, C. and K. G. Lunsford (2019), "The Dynamic Effects of Personal
//! Income Tax Changes on Macroeconomic Aggregates: A Reassessment,"
//! *American Economic Review* 109(7): 2655-2678 — the primary reference, and
//! the source of the invalidity result for the wild bootstrap. Their
//! companion theory paper (*Journal of Business & Economic Statistics*,
//! earlier as Federal Reserve Bank of Cleveland WP 16-19) carries the proof;
//! its volume and page details are **unverified here**. Block machinery:
//! Künsch (1989), *Annals of Statistics* 17(3): 1217-1241; Brüggemann,
//! Jentsch and Trenkler (2016), *Journal of Econometrics* 191(1): 69-85
//! (exact title unverified). What the wild arm reproduces: Mertens and Ravn
//! (2013), *AER* 103(4); Gertler and Karadi (2015), *AEJ: Macro* 7(1).

use tsecon_bootstrap::{indices, par_replicate, BlockScheme, BootstrapError, WildWeights};
use tsecon_ident::proxy::proxy_svar;
use tsecon_linalg::faer::{Mat, MatRef};
use tsecon_rng::Stream;

use crate::error::VarError;
use crate::irf::ma_rep;
use crate::spec::{Trend, VarSpec};

/// Which bootstrap generates the draws.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ProxyBandMethod {
    /// The Jentsch-Lunsford (2019) moving-block bootstrap: the joint pair
    /// `(uhat_t, m_t)` is resampled in overlapping blocks under a single set
    /// of block starts. This is the asymptotically valid choice and the
    /// default.
    MovingBlock,
    /// The recursive **wild** bootstrap with a common Rademacher draw on the
    /// residuals and the proxy — what Mertens-Ravn (2013) and Gertler-Karadi
    /// (2015) report.
    ///
    /// **Not asymptotically valid for this estimand.** Provided so published
    /// bands can be reproduced; every result carries
    /// [`ProxyBands::asymptotically_valid`] `== false` and a
    /// [`ProxyBands::validity_note`] saying why. See the module docs.
    Wild,
}

/// Everything about a band computation except the data.
///
/// Construct with [`ProxyBandSpec::default`] and override fields, so adding
/// an option later does not break callers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProxyBandSpec {
    /// VAR lag order `p`; must be at least 1.
    pub lags: usize,
    /// Deterministic terms, passed through to [`VarSpec`] unchanged in every
    /// bootstrap refit.
    pub trend: Trend,
    /// Maximum impulse-response horizon `H`; responses are returned for
    /// `h = 0..=H`.
    pub horizon: usize,
    /// Index of the variable whose impact response is normalized to `unit`.
    pub norm_var: usize,
    /// Size of the normalized impact on `norm_var` (the unit-effect
    /// normalization; `+1.0` is the usual choice).
    pub unit: f64,
    /// Two-sided level: bands are the `alpha/2` and `1 - alpha/2` points.
    /// `0.10` gives 90% bands.
    pub alpha: f64,
    /// Number of bootstrap replications. Jentsch-Lunsford's guidance is at
    /// least 2000 for 90% bands and more for 95%.
    pub n_boot: usize,
    /// Seed for the resampling engine; the bands are bit-identical for a
    /// given seed at any thread count.
    pub seed: u64,
    /// Which bootstrap to run.
    pub method: ProxyBandMethod,
    /// Moving-block length `ell`; `None` uses [`default_block_length`].
    ///
    /// [`ProxyBandMethod::Wild`] draws no blocks, so this value does not
    /// affect its bands — but it is **still range-checked** (`1 <= ell < T`)
    /// and still echoed as [`ProxyBands::block_length`] under that method.
    /// Validating both arms alike is deliberate: it means flipping `method`
    /// on an otherwise fixed spec can never turn a rejected configuration
    /// into an accepted one, and it keeps the echoed value meaningful. A
    /// wild-arm caller who does not care about blocking should leave this
    /// `None`.
    pub block_length: Option<usize>,
    /// Use the HC1-robust first-stage `F` (the Montiel Olea-Pflueger
    /// effective `F`) rather than the classical one, for the point estimate
    /// and every draw.
    pub robust_f: bool,
}

impl Default for ProxyBandSpec {
    fn default() -> Self {
        Self {
            lags: 2,
            trend: Trend::Constant,
            horizon: 12,
            norm_var: 0,
            unit: 1.0,
            alpha: 0.10,
            n_boot: 2000,
            seed: 0,
            method: ProxyBandMethod::MovingBlock,
            block_length: None,
            robust_f: true,
        }
    }
}

/// Why draws failed, counted by reason.
///
/// These counts are part of the output rather than an internal detail: the
/// failing draws are exactly the near-zero-denominator tail of the ratio
/// distribution, so a procedure that dropped them quietly would report
/// intervals that are systematically too narrow, worsening as the
/// instrument weakens.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BandFailures {
    /// Fewer than three finite proxy entries survived into the draw (the
    /// availability pattern is itself resampled, so this can happen when the
    /// proxy is mostly `NaN`).
    pub too_few_proxy_obs: usize,
    /// The resampled proxy was constant over its overlap, so there is no
    /// first stage.
    pub zero_proxy_variance: usize,
    /// `gamma*[norm_var]` was at the floating-point floor relative to the
    /// draw's own moment scale, so `rho*` could not be formed. This is the
    /// count that matters: it is the tail, not noise.
    pub near_zero_gamma_norm: usize,
    /// The re-estimation on the bootstrap sample failed: the OLS refit or
    /// the MA recursion built from its coefficients returned an error (for
    /// example a singular regressor cross-product). The draw never reached
    /// the identification step.
    pub refit_failed: usize,
    /// The refit succeeded but [`tsecon_ident::proxy_svar`] rejected the
    /// draw — realistically a `sigma_u*` that is not positive definite.
    ///
    /// Kept apart from `refit_failed` on purpose: merging them points a user
    /// at the VAR when the problem is the covariance. The pre-checks above
    /// (`too_few_proxy_obs`, `zero_proxy_variance`, `near_zero_gamma_norm`)
    /// already skim off the argument errors `proxy_svar` would otherwise
    /// report undifferentiated, so this counter is what remains.
    pub identification_failed: usize,
    /// The draw produced a non-finite quantity: a `Psi*_h` that overflowed
    /// the MA recursion, or an impulse response `theta*_h = Psi*_h b*` that
    /// overflowed once the per-draw normalization `b* = unit * rho*` was
    /// applied. Explosive `Ahat*` draws are *not* screened out; only
    /// overflow is caught.
    ///
    /// In practice the second route is the reachable one. An `Ahat*`
    /// explosive enough to overflow `Psi*_h` needs a full-sample `Ahat` that
    /// is explosive too, and [`proxy_svar_bands`] computes the **point**
    /// estimate at the same horizon first — so such a sample errors out of
    /// [`proxy_svar_bands`] entirely rather than producing counted draws.
    /// Measured: on a near-unit-root sample it took `horizon = 16000` to
    /// make even one draw in a hundred overflow before the point estimate
    /// did. The guard is nonetheless load-bearing, because a non-finite
    /// `theta*` admitted to the quantiles would poison every cell of the
    /// band, not just its own draw.
    pub non_finite: usize,
}

impl BandFailures {
    /// Total number of failed draws.
    pub fn total(&self) -> usize {
        self.too_few_proxy_obs
            + self.zero_proxy_variance
            + self.near_zero_gamma_norm
            + self.refit_failed
            + self.identification_failed
            + self.non_finite
    }
}

/// Bootstrap bands for a single-instrument proxy-SVAR impulse response.
///
/// Every response array is `(horizon + 1)` rows of `n` entries: row `h`,
/// column `i` is the response of variable `i` at horizon `h` to the
/// identified shock, normalized so variable `norm_var` moves by `unit` on
/// impact. The bands are **pointwise**, not a joint band over the path.
#[derive(Debug, Clone)]
pub struct ProxyBands {
    /// Full-sample point estimate `theta_h = Psi_h b`; identical to
    /// [`tsecon_ident::proxy_svar`]'s `irf` on the same inputs.
    pub point: Vec<Vec<f64>>,
    /// Lower **Hall** (basic / reverse-percentile) endpoint,
    /// `2*theta_hat - Q_{1-alpha/2}(theta*)`. The recommended band.
    pub lower: Vec<Vec<f64>>,
    /// Upper Hall endpoint, `2*theta_hat - Q_{alpha/2}(theta*)`.
    pub upper: Vec<Vec<f64>>,
    /// Lower **Efron** percentile endpoint, `Q_{alpha/2}(theta*)` — what
    /// Mertens-Ravn and Gertler-Karadi report. Differs materially from the
    /// Hall endpoint under skew.
    pub lower_efron: Vec<Vec<f64>>,
    /// Upper Efron percentile endpoint, `Q_{1-alpha/2}(theta*)`.
    pub upper_efron: Vec<Vec<f64>>,
    /// Bootstrap standard deviation per cell (divisor `n_used - 1`) over the
    /// non-failed draws. A summary statistic only: the ratio estimand is
    /// heavy-tailed, so `point +/- z*se` is **not** a valid interval and is
    /// deliberately not returned.
    pub se: Vec<Vec<f64>>,
    /// Replications requested.
    pub n_boot: usize,
    /// Replications that produced a usable draw; equals `n_boot - n_failed`.
    pub n_used: usize,
    /// Replications that failed. Equals [`BandFailures::total`].
    pub n_failed: usize,
    /// Failure breakdown by reason.
    pub failures: BandFailures,
    /// Set when `n_failed` exceeds 1% of `n_boot`: at that failure rate the
    /// instrument is too weak for the bands to be trustworthy.
    pub failure_warning: Option<String>,
    /// Moving-block length `ell` — the value [`ProxyBandSpec::block_length`]
    /// supplied, or [`default_block_length`] when it was `None`.
    ///
    /// Echoed under **both** methods. It is the length the draws actually
    /// used under [`ProxyBandMethod::MovingBlock`]; under
    /// [`ProxyBandMethod::Wild`], which draws no blocks, it is the
    /// configured-or-defaulted value carried through unused. It is never the
    /// sample length `T`.
    pub block_length: usize,
    /// Two-sided level echoed back.
    pub alpha: f64,
    /// Which bootstrap produced the draws.
    pub method: ProxyBandMethod,
    /// `false` for [`ProxyBandMethod::Wild`]. Read this before quoting the
    /// interval.
    pub asymptotically_valid: bool,
    /// One-line statement of the validity status and its citation.
    pub validity_note: &'static str,
    /// `gamma*[norm_var]` per draw (`NaN` for a failed draw) — the fragility
    /// diagnostic. Mass near zero means the unit-effect estimand itself is
    /// badly behaved and weak-IV-robust inference is called for.
    pub gamma_norm_draws: Vec<f64>,
    /// Bootstrap first-stage `F` per draw (`NaN` when failed). A
    /// **diagnostic**, not an input to the bands: an unimplemented
    /// Jentsch-Lunsford proxy rescaling would move this and nothing else.
    pub first_stage_f_draws: Vec<f64>,
    /// Bootstrap Stock-Watson reliability per draw (`NaN` when failed); same
    /// diagnostic caveat as `first_stage_f_draws`.
    pub reliability_draws: Vec<f64>,
    /// `rho* = gamma* / gamma*[norm_var]` per draw: one row of `n` entries
    /// per replication, all `NaN` for a failed draw.
    ///
    /// The **scale-free** companion to `gamma_norm_draws`. `gamma*[norm_var]`
    /// says how close the denominator came to zero in the units of the data;
    /// `rho*` says what that did to the estimand, since `b* = unit * rho*`
    /// and `rho*[norm_var] == 1` exactly in every surviving draw. It is also
    /// the quantity in which the unimplemented Jentsch-Lunsford proxy
    /// rescaling provably cancels, so it is the diagnostic that is immune to
    /// that open question. Heavy tails and sign changes here are the
    /// fragility that no wider band can repair — read it alongside
    /// `point_first_stage_f`.
    pub rho_draws: Vec<Vec<f64>>,
    /// Full-sample `gamma[norm_var]`, for comparison with
    /// `gamma_norm_draws`. Under joint blocking the draws sit around this
    /// value; centered on zero instead is the signature of independent
    /// resampling of `u` and `m`.
    pub point_gamma_norm: f64,
    /// Full-sample first-stage `F`. Below 10 is the conventional weak-
    /// instrument flag.
    pub point_first_stage_f: f64,
    /// Full-sample Stock-Watson reliability.
    pub point_reliability: f64,
    /// Finite proxy observations in the full sample, `|O|`.
    pub n_proxy: usize,
}

/// Position-specific (Künsch/Brüggemann-Jentsch-Trenkler) centering terms
/// for a moving-block bootstrap of block length `ell`.
///
/// Because candidate blocks overlap, an observation near `t = 1` or `t = T`
/// appears in strictly fewer of them than an interior observation, so the
/// naive block bootstrap does **not** satisfy `E*[u*_t] = 0` even though
/// `sum_t uhat_t = 0` holds by OLS. Averaging **across candidate starts at a
/// fixed within-block position** — not across time — is the whole content of
/// the fix; averaging across time gives the grand mean, which is exactly
/// zero and therefore does nothing at all.
///
/// **Unverified:** that Jentsch and Lunsford apply the *identical*
/// position-wise treatment to the **proxy** (`m_bar`) as to the residuals
/// (`u_bar`) is a recollection, not a checked fact; confidence is high for
/// the residuals and lower for `m_t`. This module centers both. The stake is
/// bounded: a *constant* shift of `m*` is annihilated by the identifying
/// moment, since the re-estimated residuals are demeaned over the overlap
/// before the cross-product is taken, so only the position-**dependent**
/// part of `m_bar` can move a band, and that is an `O(ell / T)` end effect.
#[derive(Debug, Clone)]
pub struct PositionCentering {
    /// `ell x n`: entry `(s, j)` is `ubar_s[j]`, the mean of `uhat_{i+s}[j]`
    /// over all `T - ell + 1` candidate block starts `i`.
    pub u_bar: Mat<f64>,
    /// Length `ell`: `mbar_s`, the mean of the **finite** `m_{i+s}` over the
    /// same candidate starts. Zero at a position where no candidate block
    /// supplies a finite proxy value.
    pub m_bar: Vec<f64>,
    /// Length `ell`: how many candidate starts contributed a finite proxy
    /// value at each position (the denominator behind `m_bar`).
    pub m_count: Vec<usize>,
    /// The block length these terms belong to.
    pub block_length: usize,
}

/// Jentsch-Lunsford's block-length rule `ell = round(5.03 * T^{1/4})`,
/// clamped to `1 <= ell <= T - 1` so at least two candidate blocks exist.
///
/// For quarterly macro samples (`T ~ 225`) this gives `ell ~ 19-20`, about
/// five years.
///
/// **Unverified.** The constant `5.03` and the exponent `1/4` are recalled
/// from Jentsch-Lunsford, not checked against the paper, and neither is the
/// rounding convention. What is *not* in doubt is the rate requirement that
/// `ell -> infinity` with `ell / T -> 0`. Use
/// [`proxy_svar_band_block_sensitivity`] to see how much the choice moves
/// the bands on a given sample; a smooth, modest response is the expected
/// picture, and a discontinuous jump means a bug in the block construction.
pub fn default_block_length(t: usize) -> usize {
    if t < 3 {
        return 1;
    }
    let raw = (5.03 * (t as f64).powf(0.25)).round();
    // `raw` is finite and positive for t >= 3, so the cast is well defined.
    let ell = raw as usize;
    ell.clamp(1, t - 1)
}

/// Position-specific centering terms for residuals and proxy at block length
/// `block_length`; see [`PositionCentering`].
///
/// Exposed (rather than kept private) because it carries the one part of the
/// algorithm that a plausible wrong implementation silently no-ops, and the
/// test that catches that is an exact enumeration over all `T - ell + 1`
/// candidate blocks — which needs these numbers.
///
/// # Errors
///
/// [`VarError::InvalidArgument`] if `block_length` is zero or leaves fewer
/// than two candidate block starts.
pub fn position_centering(
    u: MatRef<'_, f64>,
    proxy: &[f64],
    block_length: usize,
) -> Result<PositionCentering, VarError> {
    let t = u.nrows();
    let n = u.ncols();
    if block_length == 0 || block_length >= t {
        return Err(VarError::InvalidArgument {
            what: "the moving-block length must satisfy 1 <= ell < T so that at least two \
                   overlapping candidate blocks exist; shorten the block or lengthen the sample",
        });
    }
    if proxy.len() != t {
        return Err(VarError::Dimension {
            what: "the proxy must have one entry per residual row",
            expected: t,
            got: proxy.len(),
        });
    }
    let n_starts = t - block_length + 1;
    let starts = n_starts as f64;

    let u_bar = Mat::from_fn(block_length, n, |s, j| {
        let mut acc = 0.0;
        for i in 0..n_starts {
            acc += u[(i + s, j)];
        }
        acc / starts
    });

    let mut m_bar = vec![0.0f64; block_length];
    let mut m_count = vec![0usize; block_length];
    for (s, (bar, cnt)) in m_bar.iter_mut().zip(m_count.iter_mut()).enumerate() {
        let mut acc = 0.0;
        let mut k = 0usize;
        for i in 0..n_starts {
            let v = proxy[i + s];
            if v.is_finite() {
                acc += v;
                k += 1;
            }
        }
        *cnt = k;
        // A position with no finite proxy value has nothing to center; the
        // resampled entry there is NaN regardless, so 0.0 is inert.
        *bar = if k == 0 { 0.0 } else { acc / k as f64 };
    }

    Ok(PositionCentering {
        u_bar,
        m_bar,
        m_count,
        block_length,
    })
}

/// The wild-bootstrap pair `(u*, m*)` under a **common** multiplier
/// sequence: `u*_t = e_t * uhat_t` and `m*_t = e_t * m_t`, with `NaN` proxy
/// entries propagating unchanged.
///
/// This is what Mertens-Ravn's and Gertler-Karadi's shipped code is believed
/// to do (itself an **unverified** claim about their replication archives),
/// and it is exactly why the wild bootstrap fails: `m*_t u*_t' = e_t^2 m_t
/// uhat_t' = m_t uhat_t'` pointwise, so the identifying moment is frozen.
///
/// Public so that the invalidity can be demonstrated against this crate's
/// own code path rather than a re-implementation of it in a test.
///
/// `weights` is expected to have one entry per row; extra entries are
/// ignored and a short `weights` leaves the remaining rows unscaled (`1.0`),
/// which cannot happen from inside this module.
pub fn wild_common_draw(
    u: MatRef<'_, f64>,
    proxy: &[f64],
    weights: &[f64],
) -> (Mat<f64>, Vec<f64>) {
    let t = u.nrows();
    let n = u.ncols();
    let w = |i: usize| weights.get(i).copied().unwrap_or(1.0);
    let ustar = Mat::from_fn(t, n, |i, j| w(i) * u[(i, j)]);
    let mstar: Vec<f64> = proxy.iter().enumerate().map(|(i, &m)| w(i) * m).collect();
    (ustar, mstar)
}

/// Moving-block (or wild) bootstrap bands for a single-instrument proxy
/// SVAR, resampling the **joint** `(uhat_t, m_t)` pair and re-running the
/// identification inside every draw.
///
/// `endog` is the `n_obs x n` data matrix, observations in rows, oldest
/// first. `proxy` is the external instrument, either already aligned to the
/// residual sample (length `n_obs - lags`) or supplied at full length
/// (`n_obs`, in which case the first `lags` presample entries are dropped),
/// with `NaN` marking dates where the instrument is unavailable.
///
/// See the module documentation for the algorithm, what is verified, and
/// what is not.
///
/// # Errors
///
/// * [`VarError::InvalidArgument`] if `lags == 0`, `n_boot < 2`, the proxy
///   length matches neither convention, the block length is out of range, or
///   fewer than two draws survive;
/// * [`VarError::InvalidParameter`] if `alpha` is not strictly inside
///   `(0, 1)`;
/// * any error from [`VarSpec::fit`] or [`tsecon_ident::proxy_svar`] on the
///   **full sample** — a failure there is a data problem, not a draw, and is
///   propagated rather than counted.
pub fn proxy_svar_bands(
    endog: MatRef<'_, f64>,
    proxy: &[f64],
    spec: &ProxyBandSpec,
) -> Result<ProxyBands, VarError> {
    let prep = Prep::new(endog, proxy, spec)?;
    let ell = prep.block_length;
    let t = prep.t;

    let raw = match spec.method {
        ProxyBandMethod::MovingBlock => par_replicate(spec.seed, spec.n_boot, |_rep, stream| {
            // One block-start draw per block, from the T - ell + 1
            // overlapping candidates, with replacement; the shared engine
            // lays the blocks end to end and truncates to exactly T.
            let idx = indices(BlockScheme::MovingBlock { block_length: ell }, t, stream)
                .map_err(map_boot_err)?;
            let (ustar, mstar) = prep.mbb_pair(&idx);
            Ok(prep.identify(&ustar, &mstar))
        }),
        ProxyBandMethod::Wild => par_replicate(spec.seed, spec.n_boot, |_rep, stream| {
            // THE SAME Rademacher draw multiplies the residual row and the
            // proxy entry — which is precisely what freezes the identifying
            // moment. See the module docs.
            let w = WildWeights::Rademacher.sample(t, stream);
            let (ustar, mstar) = wild_common_draw(prep.uhat.as_ref(), &prep.proxy, &w);
            Ok(prep.identify(&ustar, &mstar))
        }),
    }
    .map_err(map_boot_err)?;
    let draws: Vec<DrawResult> = raw.into_iter().collect::<Result<_, VarError>>()?;

    prep.assemble(draws, spec)
}

/// The deterministic core of [`proxy_svar_bands`]: identical arithmetic,
/// but with the moving-block starts supplied explicitly instead of drawn
/// from a seeded stream.
///
/// `starts` has one row per replication, each row holding the `0`-based
/// starting index of every block in that replication (any row shorter than
/// `ceil(T / ell)` is an error; extra entries are ignored, since the
/// concatenation stops at `T`). Every start must lie in `0..=T-ell`.
///
/// This exists so the bands can be pinned against a reference
/// implementation: a NumPy transcription of the algorithm cannot reproduce
/// this library's RNG, but it can be handed the same block starts. It is
/// also the honest way to audit the procedure, or to reproduce a published
/// band from a saved index matrix.
///
/// [`ProxyBands::method`] is reported as [`ProxyBandMethod::MovingBlock`]
/// and `spec.method` / `spec.seed` are ignored.
///
/// # Errors
///
/// As [`proxy_svar_bands`], plus [`VarError::InvalidArgument`] if a row of
/// `starts` is too short or holds an out-of-range start.
pub fn proxy_svar_bands_from_starts(
    endog: MatRef<'_, f64>,
    proxy: &[f64],
    spec: &ProxyBandSpec,
    starts: &[Vec<usize>],
) -> Result<ProxyBands, VarError> {
    if starts.len() < 2 {
        return Err(VarError::InvalidArgument {
            what: "at least two replications of block starts are needed to form a bootstrap \
                   distribution",
        });
    }
    let mut spec = *spec;
    spec.n_boot = starts.len();
    spec.method = ProxyBandMethod::MovingBlock;

    let prep = Prep::new(endog, proxy, &spec)?;
    let ell = prep.block_length;
    let n_blocks = prep.t.div_ceil(ell);
    let max_start = prep.t - ell;

    let mut draws = Vec::with_capacity(starts.len());
    for row in starts {
        if row.len() < n_blocks {
            return Err(VarError::InvalidArgument {
                what: "each replication needs ceil(T / block_length) block starts",
            });
        }
        if row.iter().any(|&s| s > max_start) {
            return Err(VarError::InvalidArgument {
                what: "a block start is past T - block_length: starts index the overlapping \
                       candidate blocks 0..=T-ell, and blocks never wrap",
            });
        }
        let idx = block_indices_from_starts(row, ell, prep.t);
        let (ustar, mstar) = prep.mbb_pair(&idx);
        draws.push(prep.identify(&ustar, &mstar));
    }
    prep.assemble(draws, &spec)
}

/// Bands at `ell/2`, `ell`, and `2*ell` — Jentsch-Lunsford's block-length
/// sensitivity check, returned in that order.
///
/// The bands should move **smoothly and modestly**. A discontinuous jump
/// points at a bug in the block construction or the truncation to `T`; a
/// strong monotone trend in width says `ell` is in the wrong regime for this
/// `T`. Lengths are clamped into `1..=T-1` and may therefore coincide on a
/// short sample; each result reports the [`ProxyBands::block_length`] it
/// actually used.
///
/// The three runs use **genuine common random numbers**, so the only thing
/// that differs between them is `ell`.
///
/// This costs something and is worth it. Sharing `spec.seed` across three
/// calls to [`proxy_svar_bands`] would *not* share the draws: block starts
/// come from `uniform_index(stream, T - ell + 1)`, and changing that bound
/// changes how many 64-bit words bitmask rejection consumes, so the three
/// streams diverge at the first rejection and the three sets of bands then
/// differ by independent resampling noise as well as by `ell`. That noise is
/// exactly what would hide, or manufacture, the discontinuity this check
/// exists to detect. Instead one `(n_boot, N_max)` matrix of `[0, 1)`
/// uniforms is drawn once from `spec.seed` and mapped to starts by
/// `floor(u * (T - ell + 1))` for each `ell`, then fed to
/// [`proxy_svar_bands_from_starts`]: block `j` of replication `r` comes from
/// the same uniform in all three runs. The trade is that the mapping is an
/// inverse transform rather than the exactly-uniform bitmask rejection
/// [`tsecon_bootstrap::indices`] uses (a bias below `2^-53`), and that the
/// three results are therefore **not** equal to [`proxy_svar_bands`] at the
/// same seed and block length.
///
/// # Errors
///
/// As [`proxy_svar_bands`].
pub fn proxy_svar_band_block_sensitivity(
    endog: MatRef<'_, f64>,
    proxy: &[f64],
    spec: &ProxyBandSpec,
) -> Result<Vec<ProxyBands>, VarError> {
    let t = effective_sample_len(endog.nrows(), spec.lags)?;
    let base = match spec.block_length {
        Some(l) => l,
        None => default_block_length(t),
    };
    let hi = t.saturating_sub(1).max(1);
    let candidates = [
        (base / 2).clamp(1, hi),
        base.clamp(1, hi),
        (base * 2).clamp(1, hi),
    ];
    // The shortest block length needs the most blocks; drawing that many
    // uniforms per replication lets every run take the same prefix, so block
    // j of replication r is driven by the same uniform in all three.
    let n_max = t.div_ceil(candidates.iter().copied().min().unwrap_or(1).max(1));
    let mut stream = Stream::new(spec.seed);
    let uniforms: Vec<Vec<f64>> = (0..spec.n_boot)
        .map(|_| (0..n_max).map(|_| stream.uniform_f64()).collect())
        .collect();

    let mut out = Vec::with_capacity(3);
    for ell in candidates {
        let n_starts = t + 1 - ell.min(t);
        let starts: Vec<Vec<usize>> = uniforms
            .iter()
            .map(|row| {
                row.iter()
                    // `u < 1` so the product is below `n_starts`; the `min`
                    // is belt and braces against a rounding-up cast.
                    .map(|&u| ((u * n_starts as f64) as usize).min(n_starts - 1))
                    .collect()
            })
            .collect();
        let mut s = *spec;
        s.block_length = Some(ell);
        s.method = ProxyBandMethod::MovingBlock;
        out.push(proxy_svar_bands_from_starts(endog, proxy, &s, &starts)?);
    }
    Ok(out)
}

// ---------------------------------------------------------------- internals

/// Why one replication produced no usable impulse response.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FailureKind {
    TooFewProxyObs,
    ZeroProxyVariance,
    NearZeroGammaNorm,
    RefitFailed,
    IdentificationFailed,
    NonFinite,
}

/// One replication's output. A failed draw carries `NaN` throughout and is
/// counted, never dropped.
struct DrawResult {
    /// `(H+1) x n` impulse response, or `NaN` everywhere when failed.
    theta: Vec<Vec<f64>>,
    gamma_norm: f64,
    /// `rho* = gamma*/gamma*[norm_var]`, length `n`, or `NaN` when failed.
    rho: Vec<f64>,
    first_stage_f: f64,
    reliability: f64,
    failure: Option<FailureKind>,
}

/// Everything the per-draw closure needs, computed once.
struct Prep {
    lags: usize,
    trend: Trend,
    horizon: usize,
    norm_var: usize,
    unit: f64,
    robust_f: bool,
    n: usize,
    t: usize,
    block_length: usize,
    /// Fitted coefficient matrices `A_1..A_p` used to regenerate `y*`.
    coefs: Vec<Mat<f64>>,
    intercept: Vec<f64>,
    /// The `p` **actual observed** presample rows — no burn-in.
    init: Mat<f64>,
    uhat: Mat<f64>,
    proxy: Vec<f64>,
    centering: PositionCentering,
    point: Vec<Vec<f64>>,
    point_gamma_norm: f64,
    point_first_stage_f: f64,
    point_reliability: f64,
    n_proxy: usize,
}

impl Prep {
    fn new(endog: MatRef<'_, f64>, proxy: &[f64], spec: &ProxyBandSpec) -> Result<Self, VarError> {
        if spec.lags == 0 {
            return Err(VarError::InvalidArgument {
                what: "proxy-SVAR bands need lags >= 1: the bootstrap regenerates the sample \
                       from the VAR's own dynamics, and a VAR(0) has none",
            });
        }
        if spec.n_boot < 2 {
            return Err(VarError::InvalidArgument {
                what: "n_boot must be at least 2 to form a bootstrap distribution; \
                       Jentsch-Lunsford's guidance is 2000 or more for 90% bands",
            });
        }
        if !(spec.alpha > 0.0 && spec.alpha < 1.0) {
            return Err(VarError::InvalidParameter {
                name: "alpha",
                value: spec.alpha,
                requirement: "a value strictly inside (0, 1) — alpha = 0.1 gives 90% bands",
            });
        }

        let n_obs = endog.nrows();
        let t = effective_sample_len(n_obs, spec.lags)?;

        // Accept the proxy aligned to the residual sample, or at full length
        // with the presample rows still attached (the convention the
        // `proxy_svar` binding already accepts).
        let aligned: Vec<f64> = if proxy.len() == t {
            proxy.to_vec()
        } else if proxy.len() == n_obs {
            proxy[spec.lags..].to_vec()
        } else {
            return Err(VarError::Dimension {
                what: "the proxy must be aligned to the residual sample (n_obs - lags entries) \
                       or supplied at full length (n_obs entries)",
                expected: t,
                got: proxy.len(),
            });
        };

        let fit = VarSpec {
            lags: spec.lags,
            trend: spec.trend,
        }
        .fit(endog)?;
        let n = fit.neqs;
        if spec.norm_var >= n {
            return Err(VarError::Dimension {
                what: "norm_var must index one of the modelled series",
                expected: n,
                got: spec.norm_var,
            });
        }

        let psi = ma_rep(&fit.coefs, spec.horizon)?;
        // Full-sample identification: any failure here is a data problem,
        // so it propagates instead of being counted as a failed draw.
        let point = proxy_svar(
            fit.resid.as_ref(),
            &aligned,
            &psi,
            fit.sigma_u.as_ref(),
            spec.norm_var,
            spec.unit,
            spec.robust_f,
        )
        .map_err(|e| VarError::InvalidArgument {
            what: ident_message(&e),
        })?;

        let block_length = match spec.block_length {
            Some(l) => l,
            None => default_block_length(t),
        };
        // The wild arm uses no blocks, but the centering terms are still
        // built (cheaply) so one Prep serves both arms; `position_centering`
        // is also where the block length is validated.
        let centering = position_centering(fit.resid.as_ref(), &aligned, block_length)?;

        let init = Mat::from_fn(spec.lags, n, |t0, j| endog[(t0, j)]);

        Ok(Self {
            lags: spec.lags,
            trend: spec.trend,
            horizon: spec.horizon,
            norm_var: spec.norm_var,
            unit: spec.unit,
            robust_f: spec.robust_f,
            n,
            t,
            block_length,
            coefs: fit.coefs.clone(),
            intercept: fit.intercept.clone(),
            init,
            uhat: fit.resid.clone(),
            proxy: aligned,
            centering,
            point_gamma_norm: point.cov_um[spec.norm_var],
            point_first_stage_f: point.first_stage_f,
            point_reliability: point.reliability,
            n_proxy: point.n_proxy,
            point: point.irf,
        })
    }

    /// Gather the resampled pair under one index vector, subtracting the
    /// **position-specific** means. The same `idx` drives both blocks of
    /// `z_t = (uhat_t', m_t')'` — that joint pairing is the whole point.
    fn mbb_pair(&self, idx: &[usize]) -> (Mat<f64>, Vec<f64>) {
        let ell = self.block_length;
        let ustar = Mat::from_fn(idx.len(), self.n, |t, j| {
            // Blocks are laid end to end, so within-block position is t % ell.
            self.uhat[(idx[t], j)] - self.centering.u_bar[(t % ell, j)]
        });
        let mstar: Vec<f64> = idx
            .iter()
            .enumerate()
            // A NaN source entry stays NaN: the instrument's availability
            // pattern is resampled along with its values, so the number of
            // informative proxy observations varies across draws. The series
            // is never compacted — that would destroy the date alignment
            // with uhat* and reproduce the independent-resampling failure.
            .map(|(t, &i)| self.proxy[i] - self.centering.m_bar[t % ell])
            .collect();
        (ustar, mstar)
    }

    /// Steps 6-10: reconstruct, re-estimate, re-identify, re-normalize.
    fn identify(&self, ustar: &Mat<f64>, mstar: &[f64]) -> DrawResult {
        let ysim = simulate_recursive(&self.coefs, &self.intercept, &self.init, ustar);
        let fitted = match (VarSpec {
            lags: self.lags,
            trend: self.trend,
        })
        .fit(ysim.as_ref())
        {
            Ok(f) => f,
            // An explosive Ahat* can overflow the recursion; that is caught
            // as non-finite rather than screened out in advance.
            Err(VarError::NonFinite { .. }) => return self.failed(FailureKind::NonFinite),
            Err(_) => return self.failed(FailureKind::RefitFailed),
        };
        let psi = match ma_rep(&fitted.coefs, self.horizon) {
            Ok(p) => p,
            Err(_) => return self.failed(FailureKind::RefitFailed),
        };
        if psi
            .iter()
            .any(|m| (0..self.n).any(|i| (0..self.n).any(|j| !m[(i, j)].is_finite())))
        {
            return self.failed(FailureKind::NonFinite);
        }

        // Failure classification happens here, ahead of `proxy_svar`, so the
        // reasons can be counted separately; `proxy_svar` reports them as one
        // undifferentiated argument error. The moment is recomputed inside
        // `proxy_svar` from the same formula — that call, not this block, is
        // the single source of truth for the identification arithmetic.
        let overlap: Vec<usize> = (0..self.t).filter(|&r| mstar[r].is_finite()).collect();
        if overlap.len() < 3 {
            return self.failed(FailureKind::TooFewProxyObs);
        }
        let no = overlap.len() as f64;
        let mbar = overlap.iter().map(|&r| mstar[r]).sum::<f64>() / no;
        let smm = overlap
            .iter()
            .map(|&r| (mstar[r] - mbar) * (mstar[r] - mbar))
            .sum::<f64>();
        if smm == 0.0 {
            return self.failed(FailureKind::ZeroProxyVariance);
        }
        let resid = fitted.resid.as_ref();
        let mut gamma = vec![0.0f64; self.n];
        for (j, g) in gamma.iter_mut().enumerate() {
            let ubar = overlap.iter().map(|&r| resid[(r, j)]).sum::<f64>() / no;
            *g = overlap
                .iter()
                .map(|&r| (mstar[r] - mbar) * (resid[(r, j)] - ubar))
                .sum::<f64>()
                / no;
        }
        let g_norm = gamma[self.norm_var];
        // A floating-point floor, NOT a screening rule: draws with a small
        // but representable denominator are genuine tail draws of a
        // heavy-tailed ratio and are kept, sign flips and all.
        let scale = gamma.iter().fold(0.0f64, |a, g| a.max(g.abs()));
        if !g_norm.is_finite() || g_norm == 0.0 || g_norm.abs() <= 1e-12 * scale {
            return self.failed(FailureKind::NearZeroGammaNorm);
        }

        // Per-draw re-imposition of the unit-effect normalization: rho* =
        // gamma*/gamma*[norm_var] and b* = unit * rho*, the same map the
        // point estimate applies. Both the SCALE and the SIGN are part of
        // the estimand, so a draw whose gamma*[norm_var] is negative flips
        // every response — correct, and deliberately not "fixed".
        let res = match proxy_svar(
            resid,
            mstar,
            &psi,
            fitted.sigma_u.as_ref(),
            self.norm_var,
            self.unit,
            self.robust_f,
        ) {
            Ok(r) => r,
            // The refit SUCCEEDED to get here — the pre-checks above have
            // already taken every argument error `proxy_svar` reports, so
            // what is left is the identification itself (realistically a
            // sigma_u* that is not positive definite). Counting it as
            // `refit_failed` would point a reader at the VAR when the
            // problem is the covariance.
            Err(_) => return self.failed(FailureKind::IdentificationFailed),
        };
        if res.irf.iter().any(|row| row.iter().any(|v| !v.is_finite())) {
            return self.failed(FailureKind::NonFinite);
        }

        DrawResult {
            theta: res.irf,
            gamma_norm: g_norm,
            // Free: gamma* is already in hand and rho*[norm_var] == 1
            // exactly. Reported because it is the SCALE-FREE reading of the
            // same fragility that `gamma_norm` reports in data units, and
            // because the unimplemented Jentsch-Lunsford proxy rescaling
            // cancels in it exactly.
            rho: gamma.iter().map(|g| g / g_norm).collect(),
            first_stage_f: res.first_stage_f,
            reliability: res.reliability,
            failure: None,
        }
    }

    fn failed(&self, kind: FailureKind) -> DrawResult {
        DrawResult {
            theta: vec![vec![f64::NAN; self.n]; self.horizon + 1],
            gamma_norm: f64::NAN,
            rho: vec![f64::NAN; self.n],
            first_stage_f: f64::NAN,
            reliability: f64::NAN,
            failure: Some(kind),
        }
    }

    /// Step 11-12: count the failures, then form both intervals from the
    /// draws that survived.
    fn assemble(
        self,
        draws: Vec<DrawResult>,
        spec: &ProxyBandSpec,
    ) -> Result<ProxyBands, VarError> {
        let hh = self.horizon + 1;
        let n = self.n;
        let n_boot = draws.len();

        let mut failures = BandFailures::default();
        for d in &draws {
            match d.failure {
                None => {}
                Some(FailureKind::TooFewProxyObs) => failures.too_few_proxy_obs += 1,
                Some(FailureKind::ZeroProxyVariance) => failures.zero_proxy_variance += 1,
                Some(FailureKind::NearZeroGammaNorm) => failures.near_zero_gamma_norm += 1,
                Some(FailureKind::RefitFailed) => failures.refit_failed += 1,
                Some(FailureKind::IdentificationFailed) => failures.identification_failed += 1,
                Some(FailureKind::NonFinite) => failures.non_finite += 1,
            }
        }
        let n_failed = failures.total();
        let n_used = n_boot - n_failed;
        // The invariant the spec asks to be assertable rather than inferred.
        debug_assert_eq!(n_used, draws.iter().filter(|d| d.failure.is_none()).count());
        if n_used < 2 {
            return Err(VarError::InvalidArgument {
                what: "fewer than two bootstrap draws survived: the instrument is too weak (or \
                       too sparsely available) for this sample — inspect gamma[norm_var] and the \
                       first-stage F before trusting any interval here",
            });
        }

        let ql = spec.alpha / 2.0;
        let qu = 1.0 - spec.alpha / 2.0;
        let mut lower = vec![vec![0.0f64; n]; hh];
        let mut upper = vec![vec![0.0f64; n]; hh];
        let mut lower_efron = vec![vec![0.0f64; n]; hh];
        let mut upper_efron = vec![vec![0.0f64; n]; hh];
        let mut se = vec![vec![0.0f64; n]; hh];
        let mut vals = Vec::with_capacity(n_used);

        for h in 0..hh {
            for i in 0..n {
                vals.clear();
                vals.extend(
                    draws
                        .iter()
                        .filter(|d| d.failure.is_none())
                        .map(|d| d.theta[h][i]),
                );
                let mean = vals.iter().sum::<f64>() / n_used as f64;
                let ss = vals.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>();
                se[h][i] = (ss / (n_used as f64 - 1.0)).sqrt();
                vals.sort_by(f64::total_cmp);
                let qlo = percentile_sorted(&vals, ql);
                let qhi = percentile_sorted(&vals, qu);
                let point = self.point[h][i];
                // Hall (basic / reverse-percentile): the interval of the
                // bootstrap DEVIATIONS reflected around the point estimate.
                lower[h][i] = 2.0 * point - qhi;
                upper[h][i] = 2.0 * point - qlo;
                // Efron percentile: the raw bootstrap quantiles.
                lower_efron[h][i] = qlo;
                upper_efron[h][i] = qhi;
            }
        }

        let failure_warning = if n_failed * 100 > n_boot {
            Some(format!(
                "{n_failed} of {n_boot} bootstrap draws failed ({:.1}%), \
                 {} of them because gamma*[norm_var] hit the floating-point floor. Above ~1% the \
                 bands are not trustworthy at this instrument strength: the failures are the \
                 near-zero-denominator tail of a heavy-tailed ratio, so no interval built from \
                 the survivors is honest. Report a weak-instrument-robust set instead.",
                100.0 * n_failed as f64 / n_boot as f64,
                failures.near_zero_gamma_norm,
            ))
        } else {
            None
        };

        let (asymptotically_valid, validity_note) = match spec.method {
            ProxyBandMethod::MovingBlock => (
                true,
                "Moving-block bootstrap of Jentsch-Lunsford (2019, AER 109(7)): the joint \
                 (uhat_t, m_t) pair is resampled in overlapping blocks and the identification is \
                 re-run in every draw, so the sampling variability of the identifying moment is \
                 represented. Asymptotically valid under STRONG-instrument asymptotics; it is \
                 not a weak-instrument fix.",
            ),
            ProxyBandMethod::Wild => (
                false,
                "WILD BOOTSTRAP — NOT ASYMPTOTICALLY VALID for proxy SVARs (Jentsch and \
                 Lunsford 2019, AER 109(7)). With the common Rademacher draw used here and by \
                 Mertens-Ravn (2013) / Gertler-Karadi (2015), m*_t u*_t' = m_t uhat_t' \
                 pointwise, so the identifying moment is bit-identical across draws and its \
                 sampling uncertainty is missing. Intervals are too short at every horizon and \
                 the deficit does not vanish as T grows. Provided only to reproduce published \
                 bands; use MovingBlock for inference.",
            ),
        };

        Ok(ProxyBands {
            point: self.point,
            lower,
            upper,
            lower_efron,
            upper_efron,
            se,
            n_boot,
            n_used,
            n_failed,
            failures,
            failure_warning,
            block_length: self.block_length,
            alpha: spec.alpha,
            method: spec.method,
            asymptotically_valid,
            validity_note,
            gamma_norm_draws: draws.iter().map(|d| d.gamma_norm).collect(),
            first_stage_f_draws: draws.iter().map(|d| d.first_stage_f).collect(),
            reliability_draws: draws.iter().map(|d| d.reliability).collect(),
            rho_draws: draws.into_iter().map(|d| d.rho).collect(),
            point_gamma_norm: self.point_gamma_norm,
            point_first_stage_f: self.point_first_stage_f,
            point_reliability: self.point_reliability,
            n_proxy: self.n_proxy,
        })
    }
}

/// Effective residual-sample length `T = n_obs - lags`, with the guard that
/// makes the subtraction meaningful.
fn effective_sample_len(n_obs: usize, lags: usize) -> Result<usize, VarError> {
    if n_obs <= lags + 2 {
        return Err(VarError::InvalidArgument {
            what: "the sample is too short for a proxy-SVAR bootstrap: after lagging there must \
                   be at least three residual rows for the instrument's first stage",
        });
    }
    Ok(n_obs - lags)
}

/// Lay `ceil(T / ell)` blocks of length `ell` end to end from the supplied
/// starts and truncate to exactly `T` — byte for byte what
/// [`tsecon_bootstrap::indices`] does for
/// [`BlockScheme::MovingBlock`], with the starts given rather than drawn.
///
/// Keeping exactly `T` observations matters: retaining all `N * ell` of them
/// would rescale the bootstrap distribution by `sqrt(T / (N * ell))`, a
/// persistent coverage offset that no increase in `n_boot` removes.
///
/// **Unverified:** that `N = ceil(T / ell)` blocks are laid end to end and
/// the concatenation is cut to exactly `T` — rather than all `N * ell` rows
/// being retained — is the convention this module implements, and it is a
/// recollection of Jentsch-Lunsford rather than something read from the
/// paper. The `sqrt(T / (N * ell))` rescaling above is exactly what is at
/// stake if the other convention is theirs, so this one is not free to get
/// wrong: it is a systematic offset, not Monte-Carlo error, and it does not
/// shrink as `n_boot` grows.
fn block_indices_from_starts(starts: &[usize], ell: usize, t: usize) -> Vec<usize> {
    let mut out = Vec::with_capacity(t);
    for &start in starts {
        if out.len() >= t {
            break;
        }
        let take = ell.min(t - out.len());
        out.extend(start..start + take);
    }
    out
}

/// Regenerate a pseudo-sample recursively from the fitted coefficients,
/// conditional on the `p` **actual observed** presample rows and with no
/// burn-in: `y*_t = c + sum_i A_i y*_{t-i} + u*_t`.
///
/// Deliberately a local copy of the recursion in
/// [`crate::irf_bootstrap`] rather than a shared helper: that module's
/// version is private, and the proxy bootstrap must be able to state its own
/// initial-condition convention (observed presample, no burn-in) in one
/// place, since burn-in would simulate the estimated VAR's stationary
/// distribution — ill-defined when the estimated system has roots near unity.
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

/// Linear-interpolated percentile of an ascending slice, matching NumPy's
/// default `numpy.percentile(..., method="linear")` so the golden fixture
/// and this crate agree cell for cell. (A local copy of
/// [`crate::irf_bootstrap`]'s private helper.)
fn percentile_sorted(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return f64::NAN;
    }
    if n == 1 {
        return sorted[0];
    }
    let pos = q * (n as f64 - 1.0);
    let lo = pos.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    let frac = pos - lo as f64;
    sorted[lo] + frac * (sorted[hi] - sorted[lo])
}

/// Full-sample identification failures are the caller's data problem, so they
/// are reported with a message that says what to change.
fn ident_message(e: &tsecon_ident::IdentError) -> &'static str {
    match e {
        // The identification layer's own text already says what to change,
        // and it is `&'static str`, so it survives the change of error type
        // intact. This is the branch that fires for a misaligned, too-sparse,
        // constant, or irrelevant proxy.
        tsecon_ident::IdentError::InvalidArgument { what } => what,
        _ => {
            "the proxy SVAR could not be identified on the FULL sample, so there is nothing to \
             band: check that the proxy is aligned to the residual rows, has at least three \
             finite observations overlapping them, varies over that overlap, and has nonzero \
             covariance with the norm_var residual"
        }
    }
}

/// Map the resampling engine's error into the VAR error type. Only the
/// SeedSequence spawn limit can fire, and only for astronomically large
/// `n_boot`.
fn map_boot_err(_e: BootstrapError) -> VarError {
    VarError::InvalidArgument {
        what: "bootstrap resampling failed: n_boot exceeds the RNG substream limit; use a \
               smaller n_boot",
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    /// The block-length rule is the documented formula, clamped so at least
    /// two candidate blocks exist.
    #[test]
    fn block_length_follows_the_rule_and_the_clamp() {
        // round(5.03 * 225^0.25) = round(5.03 * 3.87298) = round(19.48) = 19.
        assert_eq!(default_block_length(225), 19);
        // round(5.03 * 100^0.25) = round(15.905) = 16.
        assert_eq!(default_block_length(100), 16);
        // Clamped below T so T - ell + 1 >= 2.
        for t in 3..12 {
            let ell = default_block_length(t);
            assert!(ell >= 1 && ell < t, "T={t} gave ell={ell}");
        }
    }

    /// Blocks are laid end to end and truncated to exactly `T`.
    #[test]
    fn indices_from_starts_truncate_to_t() {
        let idx = block_indices_from_starts(&[0, 4, 2], 3, 8);
        assert_eq!(idx, vec![0, 1, 2, 4, 5, 6, 2, 2 + 1]);
        assert_eq!(idx.len(), 8);
    }

    /// The explicit-starts index layout is identical to what the shared
    /// resampling engine produces, so the golden path and the seeded path
    /// cannot silently diverge.
    #[test]
    fn explicit_starts_match_the_shared_engine() {
        use tsecon_rng::Stream;
        let (t, ell) = (23usize, 5usize);
        let mut stream = Stream::new(4242);
        let engine = indices(
            BlockScheme::MovingBlock { block_length: ell },
            t,
            &mut stream,
        )
        .expect("moving-block indices");
        // Recover the starts the engine used, then rebuild from them.
        let starts: Vec<usize> = engine.iter().step_by(ell).copied().collect();
        assert_eq!(block_indices_from_starts(&starts, ell, t), engine);
    }

    /// Position-wise centering makes the *candidate-block average* at every
    /// within-block position exactly zero — and the uncentered version is
    /// not zero, so the check has teeth.
    #[test]
    fn position_centering_zeroes_each_within_block_position() {
        let (t, n, ell) = (17usize, 2usize, 4usize);
        // A trending, position-dependent residual field: nothing here is
        // symmetric, so an end-effect really exists.
        let u = Mat::from_fn(t, n, |i, j| {
            0.7 * (i as f64) - 0.03 * (i as f64) * (i as f64) + 2.0 * j as f64
        });
        let proxy: Vec<f64> = (0..t).map(|i| 0.5 * i as f64 - 1.0).collect();
        let c = position_centering(u.as_ref(), &proxy, ell).expect("centering");
        let n_starts = t - ell + 1;
        let mut max_centered = 0.0f64;
        let mut max_raw = 0.0f64;
        for s in 0..ell {
            for j in 0..n {
                let raw: f64 = (0..n_starts).map(|i| u[(i + s, j)]).sum::<f64>() / n_starts as f64;
                max_raw = max_raw.max(raw.abs());
                let centered: f64 = (0..n_starts)
                    .map(|i| u[(i + s, j)] - c.u_bar[(s, j)])
                    .sum::<f64>()
                    / n_starts as f64;
                max_centered = max_centered.max(centered.abs());
            }
        }
        assert!(max_centered < 1e-13, "centered mean {max_centered:e}");
        assert!(
            max_raw > 1.0,
            "uncentered mean {max_raw:e} — test has no teeth"
        );
    }

    /// A common multiplier freezes the identifying moment exactly: this is
    /// the arithmetic that invalidates the wild bootstrap.
    #[test]
    fn wild_common_draw_leaves_the_cross_moment_bit_identical() {
        let (t, n) = (24usize, 3usize);
        let u = Mat::from_fn(t, n, |i, j| ((i * 7 + j * 3) % 11) as f64 - 5.0);
        let proxy: Vec<f64> = (0..t).map(|i| ((i * 5) % 7) as f64 - 3.0).collect();
        let base = cross_moment(u.as_ref(), &proxy);
        for seed in 0..16u64 {
            // Deterministic +/-1 patterns; the claim is algebraic, so any
            // sign vector must reproduce it.
            let w: Vec<f64> = (0..t)
                .map(|i| {
                    if (seed >> (i % 6)) & 1 == 1 {
                        -1.0
                    } else {
                        1.0
                    }
                })
                .collect();
            let (us, ms) = wild_common_draw(u.as_ref(), &proxy, &w);
            let got = cross_moment(us.as_ref(), &ms);
            for (a, b) in got.iter().zip(base.iter()) {
                assert_eq!(a.to_bits(), b.to_bits(), "seed {seed}: moment moved");
            }
        }
    }

    /// `sum_t m_t u_t'` (uncentered, the object the algebra is about).
    fn cross_moment(u: MatRef<'_, f64>, m: &[f64]) -> Vec<f64> {
        (0..u.ncols())
            .map(|j| (0..u.nrows()).map(|i| m[i] * u[(i, j)]).sum::<f64>())
            .collect()
    }

    #[test]
    fn percentile_matches_numpy_linear() {
        let x = [1.0, 2.0, 3.0, 4.0];
        assert!((percentile_sorted(&x, 0.25) - 1.75).abs() < 1e-12);
        assert!((percentile_sorted(&x, 0.50) - 2.5).abs() < 1e-12);
        assert!((percentile_sorted(&x, 0.75) - 3.25).abs() < 1e-12);
    }

    #[test]
    fn failure_total_sums_every_reason() {
        let f = BandFailures {
            too_few_proxy_obs: 1,
            zero_proxy_variance: 2,
            near_zero_gamma_norm: 3,
            refit_failed: 4,
            identification_failed: 5,
            non_finite: 6,
        };
        // Distinct values, so a counter omitted from `total` cannot hide.
        assert_eq!(f.total(), 21);
    }
}
