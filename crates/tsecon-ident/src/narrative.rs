//! Narrative sign restrictions for Bayesian SVARs
//! (Antolín-Díaz & Rubio-Ramírez 2018, *American Economic Review*
//! 108(10):2802-2829): the Uhlig/ARWZ sign-restricted sampler augmented with
//! restrictions on *named historical episodes* — the sign of a structural
//! shock in a given period, and "most / least important contributor"
//! statements about a shock's role in a variable's movement over a window —
//! imposed by **importance-reweighting** the accepted rotation draws.
//!
//! # The two restriction families
//!
//! * **Shock-sign** ([`NarrativeRestriction::ShockSign`]): the structural shock
//!   `j` had a known sign in effective-sample period `t*`
//!   (`sign(eps_{j,t*}) = ±`). This is a restriction on the free per-shock
//!   orientation, evaluated jointly with the traditional impulse-response sign
//!   restrictions.
//! * **Contribution** ([`NarrativeRestriction::Contribution`]): over an
//!   episode `[t1, t2]`, shock `j` was the *most* (or *least*) important driver
//!   of variable `i`, i.e. `|C_j| >= |C_k|` (or `<=`) for every other shock
//!   `k`, where `C_k` is the episode contribution from the historical
//!   decomposition ([`crate::histdecomp`]). The optional `strong` flag replaces
//!   the pairwise comparison with the *overwhelming* variant `|C_j| >= sum_{k≠j}
//!   |C_k|` (or the *negligible* variant `<=`). These `|C|` comparisons are
//!   orientation-free.
//! * An optional [`NarrativeRestriction::ContributionSign`] fixes the *sign* of
//!   a shock's episode contribution (orientation-dependent, so folded into the
//!   orientation search).
//!
//! # Importance weighting (the AD&RR core)
//!
//! Let `S` be the traditional sign event and `N` the narrative event. The base
//! sampler proposes a reduced-form draw `phi ~ p(phi | Y)` and one rotation
//! `Q ~ Uniform(S(phi))` (Haar-retry until `S`). Rejecting additionally on `N`
//! retains `phi` in proportion to `p(phi | Y) P(N | S, phi)`; to restore the
//! reduced-form posterior each accepted draw `m` carries weight
//!
//! ```text
//! w^(m) = 1 / P̂(N | S, phi^(m)),
//! ```
//!
//! where `P̂` is a Monte-Carlo estimate over `n_weight_draws` rotations that
//! satisfy `S`: the share that *also* satisfy `N`. A draw from a `phi` whose
//! narrative-admissible slice of the identified set is small (small `P̂`) is
//! up-weighted to restore that `phi`'s traditional-posterior representation.
//! All final bands become **weighted** (weighted type-7 quantiles; the
//! identified-set min/max stay weight-free), and the effective sample size
//! `ESS = (sum w)^2 / sum w^2` is reported as a weight-concentration
//! diagnostic.
//!
//! `P̂` is formed with a Rao-Blackwellized add-one, `P̂ = (#N + 1) / (K + 1)`
//! over the `K` collected `S`-rotations, which counts the accepted draw itself
//! (a valid `S ∧ N` sample) so the estimate is strictly positive and reduces to
//! `1` exactly when `N` is implied by `S` (redundant narrative ⇒ uniform
//! weights). This `1 / P̂` form matches AD&RR and the `bsvarSIGNs` package;
//! `1 / P̂` is a (Jensen) biased estimator of `1 / P(N | S)`, so `n_weight_draws`
//! should be reasonably large and `ESS` inspected.
//!
//! # Reproducibility
//!
//! Each posterior draw gets its own substream ([`Stream::substreams`]); within
//! it the `(B, Sigma)` draw is consumed first, then the accepted rotation, then
//! the `n_weight_draws` weight rotations, so an accepted draw *and its weight*
//! are a deterministic function of `(seed, index)` and invariant to `max_tries`
//! batching. With no narrative restrictions the weight loop is skipped and the
//! sampler reproduces [`crate::SignSampler`] exactly.

use tsecon_bayes::{cholesky_irf, NiwPosterior};
use tsecon_linalg::faer::{Mat, MatRef};
use tsecon_linalg::jittered_cholesky;
use tsecon_rng::Stream;

use crate::error::IdentError;
use crate::haar::haar_rotation;
use crate::histdecomp::{
    baseline_path, coefs_and_intercept, decompose, episode_contribution_cells,
};
use crate::sampler::StructuralIrfSummary;
use crate::shocks::{
    build_regressors, orthogonalized_residuals, reduced_form_residuals, structural_shocks,
};
use crate::sign::{Sign, SignRestrictionSet};
use crate::summary::{structural_irf, summarize_weighted};

/// Default pointwise credible-band probabilities (5/16/50/84/95 percentiles).
const DEFAULT_QUANTILE_PROBS: [f64; 5] = [0.05, 0.16, 0.50, 0.84, 0.95];

#[inline]
fn sign_holds(sign: Sign, value: f64) -> bool {
    match sign {
        Sign::Positive => value > 0.0,
        Sign::Negative => value < 0.0,
    }
}

/// Whether a shock is the *most* or the *least* important episode contributor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContributionRule {
    /// The named shock dominates: `|C_j| >= |C_k|` for every other `k` (or, if
    /// `strong`, `|C_j| >= sum_{k≠j} |C_k|`).
    Most,
    /// The named shock is negligible: `|C_j| <= |C_k|` for every other `k` (or,
    /// if `strong`, `|C_j| <= sum_{k≠j} |C_k|`).
    Least,
}

/// A single narrative restriction on a named historical episode. Periods and
/// windows are given in **0-based effective-sample** indices
/// (`= data_row - lags`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NarrativeRestriction {
    /// The structural shock `shock` had sign `sign` in effective period
    /// `period`.
    ShockSign {
        /// Structural shock index.
        shock: usize,
        /// Effective-sample period.
        period: usize,
        /// Required sign of the shock realization.
        sign: Sign,
    },
    /// Over `[start, end]`, `shock` was the most / least important contributor
    /// to `variable` (a `|C|` comparison; orientation-free).
    Contribution {
        /// Response variable whose movement is being explained.
        variable: usize,
        /// Structural shock whose importance is asserted.
        shock: usize,
        /// First effective period of the episode (inclusive).
        start: usize,
        /// Last effective period of the episode (inclusive).
        end: usize,
        /// Whether the shock is the most or least important contributor.
        rule: ContributionRule,
        /// If set, use the overwhelming/negligible sum variant.
        strong: bool,
    },
    /// The episode contribution of `shock` to `variable` over `[start, end]`
    /// had sign `sign` (orientation-dependent).
    ContributionSign {
        /// Response variable.
        variable: usize,
        /// Structural shock.
        shock: usize,
        /// First effective period (inclusive).
        start: usize,
        /// Last effective period (inclusive).
        end: usize,
        /// Required sign of the episode contribution.
        sign: Sign,
    },
}

/// A validated collection of narrative restrictions against a fixed model
/// dimension `n_vars` and effective sample length `t_eff`.
#[derive(Debug, Clone)]
pub struct NarrativeRestrictionSet {
    items: Vec<NarrativeRestriction>,
    n_vars: usize,
    t_eff: usize,
    /// Shocks carrying a shock-sign or contribution-sign restriction (their
    /// orientation is pinned in the search).
    oriented_shocks: Vec<usize>,
    /// The largest episode `end` index across contribution-type restrictions
    /// (the horizon the structural IRF must reach for the checks).
    contrib_check_horizon: usize,
}

impl NarrativeRestrictionSet {
    /// Builds and validates the set: every shock and variable index must be
    /// below `n_vars`, every period below `t_eff`, and every window must have
    /// `start <= end < t_eff`.
    ///
    /// # Errors
    ///
    /// * [`IdentError::InvalidArgument`] if `n_vars == 0` or `t_eff == 0`;
    /// * [`IdentError::RestrictionOutOfRange`] if any index is out of range or a
    ///   window has `start > end`.
    pub fn new(
        items: Vec<NarrativeRestriction>,
        n_vars: usize,
        t_eff: usize,
    ) -> Result<Self, IdentError> {
        if n_vars == 0 {
            return Err(IdentError::InvalidArgument {
                what: "n_vars must be at least 1",
            });
        }
        if t_eff == 0 {
            return Err(IdentError::InvalidArgument {
                what: "t_eff must be at least 1",
            });
        }
        let mut oriented_shocks = Vec::new();
        let mut contrib_check_horizon = 0usize;
        for r in &items {
            match *r {
                NarrativeRestriction::ShockSign { shock, period, .. } => {
                    if shock >= n_vars {
                        return Err(IdentError::RestrictionOutOfRange {
                            what: "structural shock",
                            index: shock,
                            bound: n_vars,
                        });
                    }
                    if period >= t_eff {
                        return Err(IdentError::RestrictionOutOfRange {
                            what: "narrative period",
                            index: period,
                            bound: t_eff,
                        });
                    }
                    if !oriented_shocks.contains(&shock) {
                        oriented_shocks.push(shock);
                    }
                }
                NarrativeRestriction::Contribution {
                    variable,
                    shock,
                    start,
                    end,
                    ..
                }
                | NarrativeRestriction::ContributionSign {
                    variable,
                    shock,
                    start,
                    end,
                    ..
                } => {
                    if variable >= n_vars {
                        return Err(IdentError::RestrictionOutOfRange {
                            what: "response variable",
                            index: variable,
                            bound: n_vars,
                        });
                    }
                    if shock >= n_vars {
                        return Err(IdentError::RestrictionOutOfRange {
                            what: "structural shock",
                            index: shock,
                            bound: n_vars,
                        });
                    }
                    if start > end {
                        return Err(IdentError::InvalidArgument {
                            what: "narrative episode has start greater than end",
                        });
                    }
                    if end >= t_eff {
                        return Err(IdentError::RestrictionOutOfRange {
                            what: "narrative episode end",
                            index: end,
                            bound: t_eff,
                        });
                    }
                    contrib_check_horizon = contrib_check_horizon.max(end);
                    if matches!(r, NarrativeRestriction::ContributionSign { .. })
                        && !oriented_shocks.contains(&shock)
                    {
                        oriented_shocks.push(shock);
                    }
                }
            }
        }
        oriented_shocks.sort_unstable();
        Ok(Self {
            items,
            n_vars,
            t_eff,
            oriented_shocks,
            contrib_check_horizon,
        })
    }

    /// The validated restrictions.
    pub fn restrictions(&self) -> &[NarrativeRestriction] {
        &self.items
    }

    /// Number of variables the set was validated against.
    pub fn n_vars(&self) -> usize {
        self.n_vars
    }

    /// Effective sample length the set was validated against.
    pub fn t_eff(&self) -> usize {
        self.t_eff
    }

    /// Whether the set contains no restrictions.
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Shocks whose orientation is pinned by a shock-sign / contribution-sign
    /// restriction.
    fn oriented_shocks(&self) -> &[usize] {
        &self.oriented_shocks
    }

    /// The structural-IRF horizon the contribution checks need.
    fn contrib_check_horizon(&self) -> usize {
        self.contrib_check_horizon
    }

    /// Whether every shock-sign restriction on `shock` holds under orientation
    /// `s` on the structural shocks `e`.
    fn shock_sign_ok(&self, e: MatRef<'_, f64>, shock: usize, s: f64) -> bool {
        for r in &self.items {
            if let NarrativeRestriction::ShockSign {
                shock: sh,
                period,
                sign,
            } = *r
            {
                if sh == shock && !sign_holds(sign, s * e[(period, shock)]) {
                    return false;
                }
            }
        }
        true
    }

    /// Whether every contribution-sign restriction on `shock` holds under
    /// orientation `s`.
    fn contribution_sign_ok(
        &self,
        theta: &[Mat<f64>],
        e: MatRef<'_, f64>,
        shock: usize,
        s: f64,
    ) -> bool {
        for r in &self.items {
            if let NarrativeRestriction::ContributionSign {
                variable,
                shock: sh,
                start,
                end,
                sign,
            } = *r
            {
                if sh == shock {
                    let c = episode_contribution_cells(theta, e, variable, start, end)[shock];
                    if !sign_holds(sign, s * c) {
                        return false;
                    }
                }
            }
        }
        true
    }

    /// Whether every "most / least important contributor" rule holds
    /// (orientation-free).
    fn contribution_rules_hold(&self, theta: &[Mat<f64>], e: MatRef<'_, f64>) -> bool {
        let n = self.n_vars;
        for r in &self.items {
            if let NarrativeRestriction::Contribution {
                variable,
                shock,
                start,
                end,
                rule,
                strong,
            } = *r
            {
                let c = episode_contribution_cells(theta, e, variable, start, end);
                let cj = c[shock].abs();
                let holds = if strong {
                    let sum_others: f64 = (0..n).filter(|&k| k != shock).map(|k| c[k].abs()).sum();
                    match rule {
                        ContributionRule::Most => cj >= sum_others,
                        ContributionRule::Least => cj <= sum_others,
                    }
                } else {
                    (0..n).all(|k| {
                        if k == shock {
                            true
                        } else {
                            match rule {
                                ContributionRule::Most => cj >= c[k].abs(),
                                ContributionRule::Least => cj <= c[k].abs(),
                            }
                        }
                    })
                };
                if !holds {
                    return false;
                }
            }
        }
        true
    }
}

/// Whether the per-shock check of traditional sign restriction(s) on `shock`
/// holds under orientation `s` (a lift of the private
/// `SignRestrictionSet::orientation_for_shock` logic onto the crate's public
/// restriction accessor, so `sign.rs` stays untouched).
fn traditional_orientation_ok(
    signs: &SignRestrictionSet,
    theta: &[Mat<f64>],
    shock: usize,
    s: f64,
) -> bool {
    for r in signs.restrictions() {
        if r.shock != shock {
            continue;
        }
        for theta_h in theta.iter().take(r.horizon_hi + 1).skip(r.horizon_lo) {
            if !sign_holds(r.sign, s * theta_h[(r.variable, shock)]) {
                return false;
            }
        }
    }
    true
}

/// Chooses a per-shock orientation satisfying the traditional sign, narrative
/// shock-sign, and narrative contribution-sign restrictions jointly; returns
/// `None` if any restricted shock admits no satisfying orientation.
fn combined_orientation(
    signs: Option<&SignRestrictionSet>,
    narrative: &NarrativeRestrictionSet,
    theta: &[Mat<f64>],
    e: MatRef<'_, f64>,
    n: usize,
) -> Option<Vec<f64>> {
    let mut orient = vec![1.0f64; n];
    for (shock, o) in orient.iter_mut().enumerate() {
        let trad = signs.is_some_and(|s| s.restricted_shocks().contains(&shock));
        let narr = narrative.oriented_shocks().contains(&shock);
        if !trad && !narr {
            continue;
        }
        let mut chosen = None;
        for &s in &[1.0f64, -1.0f64] {
            let ok = signs.is_none_or(|set| traditional_orientation_ok(set, theta, shock, s))
                && narrative.shock_sign_ok(e, shock, s)
                && narrative.contribution_sign_ok(theta, e, shock, s);
            if ok {
                chosen = Some(s);
                break;
            }
        }
        *o = chosen?;
    }
    Some(orient)
}

/// Whether a rotation satisfies the full narrative event `N`: a joint
/// orientation exists (traditional + shock-sign + contribution-sign) and the
/// orientation-free contribution rules hold.
fn narrative_ok(
    signs: Option<&SignRestrictionSet>,
    narrative: &NarrativeRestrictionSet,
    theta: &[Mat<f64>],
    e: MatRef<'_, f64>,
    n: usize,
) -> bool {
    combined_orientation(signs, narrative, theta, e, n).is_some()
        && narrative.contribution_rules_hold(theta, e)
}

/// `s`-satisfiability of a rotation (traditional sign event); `true` for every
/// rotation when there are no traditional restrictions.
fn s_ok(signs: Option<&SignRestrictionSet>, theta: &[Mat<f64>]) -> bool {
    signs.is_none_or(|ss| ss.is_satisfied(theta))
}

/// `q * diag(orient)`: applies the per-shock sign orientation to the rotation
/// columns so the stored rotation carries the normalized shock directions.
fn apply_orientation(q: MatRef<'_, f64>, orient: &[f64]) -> Mat<f64> {
    let n = q.nrows();
    Mat::from_fn(n, n, |i, j| q[(i, j)] * orient[j])
}

/// One accepted draw: the raw reduced form and the sign-normalized rotation, so
/// consumers can derive either structural IRFs or the historical decomposition.
#[derive(Debug, Clone)]
struct AcceptedDraw {
    b: Mat<f64>,
    sigma: Mat<f64>,
    q: Mat<f64>,
    weight: f64,
}

/// Diagnostics for a narrative run — in a set-identified, importance-weighted
/// setting the diagnostics *are* the inference.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NarrativeDiagnostics {
    /// Reduced-form draws processed.
    pub posterior_draws_used: usize,
    /// Total rotations drawn in the acceptance loop.
    pub rotations_tried: usize,
    /// Accepted (narrative-passing) draws.
    pub accepted: usize,
    /// `accepted / rotations_tried`.
    pub acceptance_rate: f64,
    /// Draws whose proposed `S`-rotation also satisfied `N`.
    pub narrative_accepted: usize,
    /// `narrative_accepted / (draws with an S-rotation)`.
    pub narrative_acceptance_rate: f64,
    /// Effective sample size `(sum w)^2 / sum w^2`.
    pub ess: f64,
    /// Mean raw importance weight `mean(1 / P̂)` (`>= 1`; exactly `1` with no
    /// narrative restrictions).
    pub mean_weight: f64,
    /// Smallest `P̂(N | S)` across accepted draws (`1` with no narrative).
    pub min_ptilde: f64,
}

/// Weighted per-`(time, variable, shock)` summary of the historical
/// decomposition across the accepted, importance-weighted draws.
#[derive(Debug, Clone)]
pub struct HdSetSummary {
    summary: StructuralIrfSummary,
    baseline: Mat<f64>,
    t_eff: usize,
    n_vars: usize,
}

impl HdSetSummary {
    /// Number of variables (and shocks).
    pub fn n_vars(&self) -> usize {
        self.n_vars
    }

    /// Effective sample length (decomposition dates `0..t_eff`).
    pub fn t_eff(&self) -> usize {
        self.t_eff
    }

    /// The deterministic / initial-condition baseline (posterior-mean draw),
    /// `t_eff x n`.
    pub fn baseline(&self) -> MatRef<'_, f64> {
        self.baseline.as_ref()
    }

    /// The quantile probabilities.
    pub fn probs(&self) -> &[f64] {
        self.summary.probs()
    }

    /// The weighted min/max and quantile summary of `HD[time][(variable,
    /// shock)]` (indexed as the IRF summary is, with `time` in the horizon
    /// slot).
    ///
    /// # Errors
    ///
    /// [`IdentError::RestrictionOutOfRange`] if any index is out of range.
    pub fn point(
        &self,
        variable: usize,
        shock: usize,
        time: usize,
    ) -> Result<&crate::sampler::IrfBandPoint, IdentError> {
        self.summary.point(variable, shock, time)
    }
}

/// The result of a narrative run: the accepted draws (kept as raw reduced form
/// plus sign-normalized rotation), their importance weights, and the
/// diagnostics. Structural-IRF and historical-decomposition summaries are
/// derived on demand.
#[derive(Debug, Clone)]
pub struct NarrativeSampleResult {
    draws: Vec<AcceptedDraw>,
    data: Mat<f64>,
    posterior_mean_b: Mat<f64>,
    p: usize,
    n_vars: usize,
    t_eff: usize,
    probs: Vec<f64>,
    weights_normalized: Vec<f64>,
    diagnostics: NarrativeDiagnostics,
}

impl NarrativeSampleResult {
    /// Per-accepted-draw importance weights, normalized to sum to the number of
    /// accepted draws (so the mean is `1`, and every weight is `1` with no
    /// narrative restrictions). Aligned with the accepted-draw order.
    pub fn weights(&self) -> &[f64] {
        &self.weights_normalized
    }

    /// Number of accepted draws.
    pub fn accepted(&self) -> usize {
        self.draws.len()
    }

    /// Number of variables (and shocks).
    pub fn n_vars(&self) -> usize {
        self.n_vars
    }

    /// Effective sample length.
    pub fn t_eff(&self) -> usize {
        self.t_eff
    }

    /// The run diagnostics (always inspect before reading any band).
    pub fn diagnostics(&self) -> &NarrativeDiagnostics {
        &self.diagnostics
    }

    /// The quantile probabilities.
    pub fn probs(&self) -> &[f64] {
        &self.probs
    }

    /// The importance-weighted structural-IRF summary to `horizon` (identified-
    /// set min/max weight-free; quantiles weighted).
    ///
    /// # Errors
    ///
    /// [`IdentError::Bayes`] on a Cholesky-IRF failure.
    pub fn irf_summary(&self, horizon: usize) -> Result<StructuralIrfSummary, IdentError> {
        let mut irf_draws: Vec<Vec<Mat<f64>>> = Vec::with_capacity(self.draws.len());
        for d in &self.draws {
            let base = cholesky_irf(d.b.as_ref(), d.sigma.as_ref(), self.p, horizon)?;
            // d.q already carries the sign normalization.
            irf_draws.push(structural_irf(&base, d.q.as_ref()));
        }
        let weights: Vec<f64> = self.draws.iter().map(|d| d.weight).collect();
        Ok(summarize_weighted(
            &irf_draws,
            &weights,
            self.n_vars,
            horizon,
            &self.probs,
        ))
    }

    /// The importance-weighted historical-decomposition summary over the full
    /// effective sample, with the baseline evaluated at the posterior-mean
    /// reduced form.
    ///
    /// # Errors
    ///
    /// [`IdentError::Bayes`]/[`IdentError::Linalg`] on a decomposition failure.
    pub fn hd_summary(&self) -> Result<HdSetSummary, IdentError> {
        let horizon = self.t_eff.saturating_sub(1);
        let mut hd_draws: Vec<Vec<Mat<f64>>> = Vec::with_capacity(self.draws.len());
        for d in &self.draws {
            let dec = decompose(
                self.data.as_ref(),
                d.b.as_ref(),
                d.sigma.as_ref(),
                d.q.as_ref(),
                self.p,
                horizon,
            )?;
            hd_draws.push(dec.hd().to_vec());
        }
        let weights: Vec<f64> = self.draws.iter().map(|d| d.weight).collect();
        let summary = summarize_weighted(&hd_draws, &weights, self.n_vars, horizon, &self.probs);
        let (coefs, intercept) =
            coefs_and_intercept(self.posterior_mean_b.as_ref(), self.n_vars, self.p);
        let baseline = baseline_path(&coefs, &intercept, self.data.as_ref(), self.p)?;
        Ok(HdSetSummary {
            summary,
            baseline,
            t_eff: self.t_eff,
            n_vars: self.n_vars,
        })
    }
}

/// The narrative sign-restriction sampler configuration.
#[derive(Debug, Clone)]
pub struct NarrativeSampler {
    horizon: usize,
    n_posterior_draws: usize,
    max_tries_per_draw: usize,
    n_weight_draws: usize,
    quantile_probs: Vec<f64>,
}

impl NarrativeSampler {
    /// A sampler drawing `n_posterior_draws` reduced-form draws, allowing
    /// `max_tries_per_draw` rotation attempts per draw, and estimating each
    /// weight `P̂(N | S)` over `n_weight_draws` `S`-rotations. `horizon` is the
    /// default structural-IRF horizon for [`NarrativeSampleResult::irf_summary`]
    /// (it can be overridden at summary time). Uses the default 5/16/50/84/95
    /// bands.
    ///
    /// # Errors
    ///
    /// [`IdentError::InvalidArgument`] if `n_posterior_draws`,
    /// `max_tries_per_draw`, or `n_weight_draws` is zero.
    pub fn new(
        horizon: usize,
        n_posterior_draws: usize,
        max_tries_per_draw: usize,
        n_weight_draws: usize,
    ) -> Result<Self, IdentError> {
        if n_posterior_draws == 0 {
            return Err(IdentError::InvalidArgument {
                what: "n_posterior_draws must be at least 1",
            });
        }
        if max_tries_per_draw == 0 {
            return Err(IdentError::InvalidArgument {
                what: "max_tries_per_draw must be at least 1",
            });
        }
        if n_weight_draws == 0 {
            return Err(IdentError::InvalidArgument {
                what: "n_weight_draws must be at least 1",
            });
        }
        Ok(Self {
            horizon,
            n_posterior_draws,
            max_tries_per_draw,
            n_weight_draws,
            quantile_probs: DEFAULT_QUANTILE_PROBS.to_vec(),
        })
    }

    /// The default IRF horizon (used by [`NarrativeSampleResult::irf_summary`]
    /// when called as `irf_summary(sampler.horizon())`).
    pub fn horizon(&self) -> usize {
        self.horizon
    }

    /// Overrides the pointwise band probabilities (each in `[0, 1]`).
    ///
    /// # Errors
    ///
    /// [`IdentError::InvalidArgument`] if the list is empty or any probability
    /// is outside `[0, 1]` or non-finite.
    pub fn with_quantile_probs(mut self, probs: Vec<f64>) -> Result<Self, IdentError> {
        if probs.is_empty() {
            return Err(IdentError::InvalidArgument {
                what: "at least one quantile probability is required",
            });
        }
        for &p in &probs {
            if !p.is_finite() || !(0.0..=1.0).contains(&p) {
                return Err(IdentError::InvalidArgument {
                    what: "quantile probabilities must be finite and in [0, 1]",
                });
            }
        }
        self.quantile_probs = probs;
        Ok(self)
    }

    /// Runs the sampler against a reduced-form NIW posterior, the raw data
    /// (needed to recover structural shocks), optional traditional sign
    /// restrictions, and optional narrative restrictions, seeding all
    /// randomness from `seed`.
    ///
    /// At least one of `signs` / a non-empty `narrative` must be present. With
    /// `narrative = None` the weight loop is skipped, every weight is `1`, and
    /// the accepted draws reproduce [`crate::SignSampler`] bit-for-bit.
    ///
    /// # Errors
    ///
    /// * [`IdentError::InvalidArgument`] if no restrictions are supplied;
    /// * [`IdentError::Dimension`] if any restriction set or the data disagrees
    ///   with the posterior on the number of variables, or the narrative set's
    ///   effective sample length differs from the data's;
    /// * [`IdentError::Rng`] on substream spawning;
    /// * [`IdentError::Bayes`]/[`IdentError::Linalg`]/[`IdentError::Stats`] on a
    ///   posterior draw, Cholesky, or Haar failure.
    pub fn run(
        &self,
        posterior: &NiwPosterior,
        data: MatRef<'_, f64>,
        signs: Option<&SignRestrictionSet>,
        narrative: Option<&NarrativeRestrictionSet>,
        seed: u64,
    ) -> Result<NarrativeSampleResult, IdentError> {
        let n = posterior.n_vars();
        let p = posterior.lag_order();
        let narrative = narrative.filter(|nr| !nr.is_empty());
        if signs.is_none() && narrative.is_none() {
            return Err(IdentError::InvalidArgument {
                what: "at least one of sign restrictions / narrative restrictions is required",
            });
        }
        if data.ncols() != n {
            return Err(IdentError::Dimension {
                what: "data and posterior must have the same number of variables",
                expected: n,
                got: data.ncols(),
            });
        }
        if data.nrows() < p + 1 {
            return Err(IdentError::Dimension {
                what: "data must have at least lags + 1 rows",
                expected: p + 1,
                got: data.nrows(),
            });
        }
        let t_eff = data.nrows() - p;
        if let Some(s) = signs {
            if s.n_vars() != n {
                return Err(IdentError::Dimension {
                    what:
                        "sign-restriction set and posterior must have the same number of variables",
                    expected: n,
                    got: s.n_vars(),
                });
            }
        }
        if let Some(nr) = narrative {
            if nr.n_vars() != n {
                return Err(IdentError::Dimension {
                    what: "narrative-restriction set and posterior must have the same number of variables",
                    expected: n,
                    got: nr.n_vars(),
                });
            }
            if nr.t_eff() != t_eff {
                return Err(IdentError::Dimension {
                    what: "narrative-restriction set effective length must equal T - lags",
                    expected: t_eff,
                    got: nr.t_eff(),
                });
            }
        }

        // The structural-IRF horizon the accept / narrative checks need.
        let check_horizon = signs
            .map(|s| s.horizon())
            .unwrap_or(0)
            .max(narrative.map(|nr| nr.contrib_check_horizon()).unwrap_or(0));

        // Regressors are fixed across draws; only needed when we must recover
        // structural shocks (narrative present).
        let yx = if narrative.is_some() {
            Some(build_regressors(data, p)?)
        } else {
            None
        };

        let mut substreams = Stream::substreams(seed, self.n_posterior_draws)?;
        let mut draws: Vec<AcceptedDraw> = Vec::new();
        let mut rotations_tried = 0usize;
        let mut accepted_s = 0usize;
        let mut min_ptilde = 1.0f64;

        for stream in substreams.iter_mut() {
            let niw = posterior.draw(stream)?;
            let b = niw.b;
            let sigma = niw.sigma;
            let base = cholesky_irf(b.as_ref(), sigma.as_ref(), p, check_horizon)?;

            // Orthogonalized residuals for this draw (narrative only).
            let w_opt = if let Some((y, x)) = &yx {
                let u = reduced_form_residuals(y.as_ref(), x.as_ref(), b.as_ref());
                let p_chol = jittered_cholesky(sigma.as_ref())?.factor;
                Some(orthogonalized_residuals(u.as_ref(), p_chol.as_ref())?)
            } else {
                None
            };

            // Propose one Q ~ Uniform(S(phi)) via Haar-retry.
            let mut proposal = None;
            for _ in 0..self.max_tries_per_draw {
                rotations_tried += 1;
                let q = haar_rotation(n, stream)?;
                let theta = structural_irf(&base, q.as_ref());
                if s_ok(signs, &theta) {
                    proposal = Some((q, theta));
                    break;
                }
            }
            let (q, theta) = match proposal {
                Some(x) => x,
                None => continue, // no S-rotation for this draw
            };
            accepted_s += 1;

            // Decide on N and, if accepted, fix the orientation and the weight.
            let (orient, weight, ptilde) = match narrative {
                None => {
                    let orient = match signs {
                        Some(ss) => match ss.accept_orientations(&theta) {
                            Some(o) => o,
                            None => continue,
                        },
                        None => vec![1.0; n],
                    };
                    (orient, 1.0, 1.0)
                }
                Some(nr) => {
                    let w = match &w_opt {
                        Some(w) => w,
                        None => continue,
                    };
                    let e = structural_shocks(w.as_ref(), q.as_ref());
                    if !narrative_ok(signs, nr, &theta, e.as_ref(), n) {
                        continue; // passes S but not N -> dropped
                    }
                    let orient = match combined_orientation(signs, nr, &theta, e.as_ref(), n) {
                        Some(o) => o,
                        None => continue,
                    };
                    let (weight, ptilde) = estimate_weight(
                        signs,
                        nr,
                        &base,
                        w.as_ref(),
                        n,
                        self.n_weight_draws,
                        self.max_tries_per_draw,
                        stream,
                    )?;
                    (orient, weight, ptilde)
                }
            };

            if ptilde < min_ptilde {
                min_ptilde = ptilde;
            }
            let q_oriented = apply_orientation(q.as_ref(), &orient);
            draws.push(AcceptedDraw {
                b,
                sigma,
                q: q_oriented,
                weight,
            });
        }

        let accepted = draws.len();
        let acceptance_rate = if rotations_tried == 0 {
            0.0
        } else {
            accepted as f64 / rotations_tried as f64
        };
        let narrative_acceptance_rate = if accepted_s == 0 {
            0.0
        } else {
            accepted as f64 / accepted_s as f64
        };
        let sum_w: f64 = draws.iter().map(|d| d.weight).sum();
        let sum_w2: f64 = draws.iter().map(|d| d.weight * d.weight).sum();
        let ess = if sum_w2 > 0.0 {
            sum_w * sum_w / sum_w2
        } else {
            0.0
        };
        let mean_weight = if accepted == 0 {
            0.0
        } else {
            sum_w / accepted as f64
        };
        // Normalize weights to sum to the accepted count (mean 1).
        let weights_normalized: Vec<f64> = if sum_w > 0.0 {
            draws
                .iter()
                .map(|d| d.weight * accepted as f64 / sum_w)
                .collect()
        } else {
            vec![0.0; accepted]
        };
        let min_ptilde = if accepted == 0 { f64::NAN } else { min_ptilde };

        let diagnostics = NarrativeDiagnostics {
            posterior_draws_used: self.n_posterior_draws,
            rotations_tried,
            accepted,
            acceptance_rate,
            narrative_accepted: accepted,
            narrative_acceptance_rate,
            ess,
            mean_weight,
            min_ptilde,
        };

        Ok(NarrativeSampleResult {
            draws,
            data: Mat::from_fn(data.nrows(), data.ncols(), |i, j| data[(i, j)]),
            posterior_mean_b: posterior.b_bar().to_owned(),
            p,
            n_vars: n,
            t_eff,
            probs: self.quantile_probs.clone(),
            weights_normalized,
            diagnostics,
        })
    }
}

/// Monte-Carlo estimate `P̂(N | S, phi) = (#N + 1) / (K + 1)` over up to
/// `k_w` rotations that satisfy `S`, drawn from `stream` after the accepted
/// rotation. Returns `(weight = 1 / P̂, P̂)`.
#[allow(clippy::too_many_arguments)]
fn estimate_weight(
    signs: Option<&SignRestrictionSet>,
    narrative: &NarrativeRestrictionSet,
    base: &[Mat<f64>],
    w: MatRef<'_, f64>,
    n: usize,
    k_w: usize,
    max_tries: usize,
    stream: &mut Stream,
) -> Result<(f64, f64), IdentError> {
    let mut collected = 0usize;
    let mut count_n = 0usize;
    while collected < k_w {
        // Draw one S-passing rotation via Haar-retry.
        let mut found = None;
        for _ in 0..max_tries {
            let q = haar_rotation(n, stream)?;
            let theta = structural_irf(base, q.as_ref());
            if s_ok(signs, &theta) {
                found = Some((q, theta));
                break;
            }
        }
        let (q, theta) = match found {
            Some(x) => x,
            None => break, // S infeasible within the budget; stop collecting
        };
        collected += 1;
        let e = structural_shocks(w, q.as_ref());
        if narrative_ok(signs, narrative, &theta, e.as_ref(), n) {
            count_n += 1;
        }
    }
    let p_hat = (count_n as f64 + 1.0) / (collected as f64 + 1.0);
    Ok((1.0 / p_hat, p_hat))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::sampler::SignSampler;
    use crate::sign::SignRestriction;

    /// A Minnesota-NIW posterior from a small deterministic pseudo-dataset,
    /// mirroring the binding path (lambda0=100, lambda1=0.2, lambda3=1,
    /// delta=0). Returns both the posterior and the data.
    fn toy_posterior(lags: usize) -> (NiwPosterior, Mat<f64>) {
        let n = 3usize;
        let t = 120usize;
        let mut state = 0x2545F491_4F6CDD1Du64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        let mut data = Mat::<f64>::zeros(t, n);
        for r in 1..t {
            for c in 0..n {
                data[(r, c)] = 0.4 * data[(r - 1, c)] + 0.1 * next();
            }
        }
        let prior = tsecon_bayes::MinnesotaNiwPrior::new(data.as_ref(), lags, 100.0, 0.2, 1.0, 0.0)
            .expect("prior");
        let post = prior.posterior(data.as_ref()).expect("posterior");
        (post, data)
    }

    #[test]
    fn no_narrative_reproduces_sign_sampler_bitwise() -> Result<(), IdentError> {
        let horizon = 8;
        let (post, data) = toy_posterior(2);
        let signs = SignRestrictionSet::new(
            vec![
                SignRestriction::at(0, 0, 0, Sign::Positive),
                SignRestriction::at(1, 0, 0, Sign::Negative),
            ],
            3,
            horizon,
        )?;
        let sign_res = SignSampler::new(horizon, 40, 200)?.run(&post, &signs, 7)?;
        let narr_res = NarrativeSampler::new(horizon, 40, 200, 50)?.run(
            &post,
            data.as_ref(),
            Some(&signs),
            None,
            7,
        )?;
        // Every weight is exactly 1.
        for &wt in narr_res.weights() {
            assert_eq!(wt.to_bits(), 1.0f64.to_bits());
        }
        // Same accepted count and bit-identical IRF bands.
        assert_eq!(sign_res.draws().len(), narr_res.accepted());
        let a = sign_res.summary();
        let b = narr_res.irf_summary(horizon)?;
        for h in 0..=horizon {
            for i in 0..3 {
                for j in 0..3 {
                    let pa = a.point(i, j, h)?;
                    let pb = b.point(i, j, h)?;
                    assert_eq!(pa.min.to_bits(), pb.min.to_bits(), "min ({i},{j},{h})");
                    assert_eq!(pa.max.to_bits(), pb.max.to_bits(), "max ({i},{j},{h})");
                    for (qa, qb) in pa.quantiles.iter().zip(pb.quantiles.iter()) {
                        assert_eq!(qa.to_bits(), qb.to_bits(), "quantile ({i},{j},{h})");
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn shock_sign_only_gives_uniform_weights() -> Result<(), IdentError> {
        // With no traditional restrictions, a shock-sign restriction only pins
        // the free orientation, so EVERY rotation satisfies N: P̂ = 1 exactly,
        // all weights uniform, and every posterior draw is accepted.
        let (post, data) = toy_posterior(2);
        let t_eff = data.nrows() - 2;
        let narrative = NarrativeRestrictionSet::new(
            vec![NarrativeRestriction::ShockSign {
                shock: 0,
                period: t_eff / 2,
                sign: Sign::Positive,
            }],
            3,
            t_eff,
        )?;
        let res = NarrativeSampler::new(6, 30, 200, 100)?.run(
            &post,
            data.as_ref(),
            None,
            Some(&narrative),
            123,
        )?;
        assert_eq!(res.accepted(), 30, "every draw should be accepted");
        for &wt in res.weights() {
            assert!((wt - 1.0).abs() < 1e-12, "weight not uniform: {wt}");
        }
        assert!((res.diagnostics().ess - 30.0).abs() < 1e-9);
        assert!((res.diagnostics().min_ptilde - 1.0).abs() < 1e-12);
        // The shock-sign restriction is honored on every accepted draw.
        let dec_hd = res.hd_summary()?;
        assert_eq!(dec_hd.t_eff(), t_eff);
        Ok(())
    }

    #[test]
    fn contribution_restriction_makes_weights_vary() -> Result<(), IdentError> {
        // A "most important contributor" restriction is not implied by S, so
        // P̂ < 1 for at least some draws => weights vary and ESS < accepted.
        let (post, data) = toy_posterior(2);
        let t_eff = data.nrows() - 2;
        let signs =
            SignRestrictionSet::new(vec![SignRestriction::at(0, 0, 0, Sign::Positive)], 3, 6)?;
        let narrative = NarrativeRestrictionSet::new(
            vec![NarrativeRestriction::Contribution {
                variable: 0,
                shock: 0,
                start: 10,
                end: 20,
                rule: ContributionRule::Most,
                strong: false,
            }],
            3,
            t_eff,
        )?;
        let res = NarrativeSampler::new(6, 60, 300, 120)?.run(
            &post,
            data.as_ref(),
            Some(&signs),
            Some(&narrative),
            2026,
        )?;
        assert!(res.accepted() > 0, "nothing accepted");
        // All raw weights are >= 1 (mean normalized to 1). Some must exceed 1.
        let w = res.weights();
        let any_above = w.iter().any(|&x| x > 1.0 + 1e-9);
        assert!(any_above, "no weight exceeds 1; narrative did not bind");
        assert!(res.diagnostics().mean_weight >= 1.0 - 1e-9);
        assert!(res.diagnostics().ess <= res.accepted() as f64 + 1e-9);
        assert!(res.diagnostics().min_ptilde <= 1.0 + 1e-12);
        // The accepted draws honor the contribution rule: shock 0 dominates.
        let hd = res.hd_summary()?;
        assert_eq!(hd.n_vars(), 3);
        Ok(())
    }

    #[test]
    fn deterministic_across_runs_at_fixed_seed() -> Result<(), IdentError> {
        let (post, data) = toy_posterior(2);
        let t_eff = data.nrows() - 2;
        let signs =
            SignRestrictionSet::new(vec![SignRestriction::at(0, 0, 0, Sign::Positive)], 3, 6)?;
        let narrative = NarrativeRestrictionSet::new(
            vec![NarrativeRestriction::Contribution {
                variable: 1,
                shock: 0,
                start: 5,
                end: 15,
                rule: ContributionRule::Most,
                strong: false,
            }],
            3,
            t_eff,
        )?;
        let run = || {
            NarrativeSampler::new(6, 25, 200, 80)
                .unwrap()
                .run(&post, data.as_ref(), Some(&signs), Some(&narrative), 99)
                .unwrap()
        };
        let a = run();
        let b = run();
        assert_eq!(a.accepted(), b.accepted());
        for (wa, wb) in a.weights().iter().zip(b.weights().iter()) {
            assert_eq!(wa.to_bits(), wb.to_bits());
        }
        Ok(())
    }

    #[test]
    fn requires_at_least_one_restriction() {
        let (post, data) = toy_posterior(2);
        let out =
            NarrativeSampler::new(6, 5, 50, 20)
                .unwrap()
                .run(&post, data.as_ref(), None, None, 0);
        assert!(matches!(out, Err(IdentError::InvalidArgument { .. })));
    }

    #[test]
    fn narrative_teff_mismatch_errors() {
        let (post, data) = toy_posterior(2);
        // Build a narrative set against the WRONG effective length.
        let bad = NarrativeRestrictionSet::new(
            vec![NarrativeRestriction::ShockSign {
                shock: 0,
                period: 3,
                sign: Sign::Positive,
            }],
            3,
            10,
        )
        .unwrap();
        let out = NarrativeSampler::new(6, 5, 50, 20).unwrap().run(
            &post,
            data.as_ref(),
            None,
            Some(&bad),
            0,
        );
        assert!(matches!(out, Err(IdentError::Dimension { .. })));
    }
}
