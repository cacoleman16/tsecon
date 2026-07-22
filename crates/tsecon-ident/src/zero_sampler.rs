//! The zero + sign restricted Bayesian SVAR sampler
//! (Rubio-Ramirez-Waggoner-Zha 2010 exact-zero column recursion; Arias,
//! Rubio-Ramirez & Waggoner 2018 importance weighting): a superset of the
//! sign-only [`crate::SignSampler`].
//!
//! For each of `n_posterior_draws` reduced-form draws `(B, Sigma)` from a
//! `tsecon-bayes` Normal-inverse-Wishart posterior:
//!
//! 1. compute the Cholesky-orthogonalized base `Theta^chol_h = Psi_h P` once
//!    ([`tsecon_bayes::cholesky_irf`]);
//! 2. draw a rotation `Q` that satisfies every zero restriction *by
//!    construction* via [`zero_constrained_rotation`] (the RWZ column
//!    recursion) — the zeros are never in a rejection loop;
//! 3. form `Theta_h = Theta^chol_h Q`; if sign restrictions are present,
//!    retry (redrawing `Q`) up to `max_tries_per_draw` times until some
//!    per-shock orientation satisfies them, else accept immediately with the
//!    positive-diagonal normalization (each shock's own impact positive);
//! 4. store the sign-normalized structural IRF and its ARW importance weight.
//!
//! # Reproducibility
//!
//! Each posterior draw gets its own substream ([`Stream::substreams`]);
//! within it the `(B, Sigma)` draw is consumed first, then the rotation
//! attempts, so an accepted draw is a deterministic function of `(seed,
//! index)` and invariant to `max_tries` batching — identical to
//! [`crate::SignSampler`].
//!
//! # Recursive recovery (the strong golden)
//!
//! With strict-upper-triangle impact zeros and no sign restrictions, every
//! null space is one-dimensional, the weight is one, and each draw's
//! structural IRF collapses to that draw's Cholesky IRF deterministically —
//! so `zero_sign_svar` reproduces `var_irf(orth=True)` to machine precision.
//!
//! # Set identification and the ARW weight, honestly
//!
//! The `set_min`/`set_max` envelope (min/max over accepted draws) is the
//! support of the identified set — *weight-invariant* and the defensible,
//! prior-robust object (Baumeister-Hamilton 2015 caveat, documented in the
//! crate root). The pointwise `quantiles` blend data with prior even after
//! ARW weighting and are descriptive. The ARW importance weight is exactly
//! one for impact-only zero patterns; see [`crate::zero`] for the horizon
//! `>= 1` limitation.

use tsecon_bayes::{cholesky_irf, NiwPosterior};
use tsecon_linalg::faer::Mat;
use tsecon_rng::Stream;

use crate::error::IdentError;
use crate::sampler::{IrfBandPoint, SignRestrictionDiagnostics};
use crate::sign::SignRestrictionSet;
use crate::zero::{structural_irf, zero_constrained_rotation, ZeroRestrictionSet};

/// Default pointwise credible-band probabilities (5/16/50/84/95 percentiles:
/// median plus the 68% and 90% equal-tailed bands).
const DEFAULT_QUANTILE_PROBS: [f64; 5] = [0.05, 0.16, 0.50, 0.84, 0.95];

/// The result of a zero + sign restricted run: the accepted, sign-normalized
/// structural IRF draws, their ARW importance weights, the mandatory
/// diagnostics, and the pointwise weighted summary.
#[derive(Debug, Clone)]
pub struct ZeroSignSampleResult {
    draws: Vec<Vec<Mat<f64>>>,
    weights: Vec<f64>,
    diagnostics: SignRestrictionDiagnostics,
    ess: f64,
    probs: Vec<f64>,
    n_vars: usize,
    horizon: usize,
    /// Row-major over `[horizon][variable][shock]`.
    points: Vec<IrfBandPoint>,
}

impl ZeroSignSampleResult {
    /// The accepted structural IRF draws, indexed `[draw][horizon]`, each an
    /// `n x n` matrix with `(i, j)` the sign-normalized response of variable
    /// `i` to structural shock `j`.
    pub fn draws(&self) -> &[Vec<Mat<f64>>] {
        &self.draws
    }

    /// Per-accepted-draw ARW importance weights, normalized to sum to one
    /// (all equal `1 / n_accepted` when every zero is impact-only). Aligned
    /// with [`ZeroSignSampleResult::draws`].
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// The run diagnostics (always inspect before reading any band).
    pub fn diagnostics(&self) -> &SignRestrictionDiagnostics {
        &self.diagnostics
    }

    /// Effective sample size `(sum w)^2 / sum w^2` in `(0, n_accepted]` (equal
    /// to `n_accepted` when weights are constant); a weight-concentration
    /// diagnostic — a low value means the weighted bands are dominated by a
    /// few draws.
    pub fn ess(&self) -> f64 {
        self.ess
    }

    /// The quantile probabilities the bands were computed at.
    pub fn probs(&self) -> &[f64] {
        &self.probs
    }

    /// Number of variables (and shocks).
    pub fn n_vars(&self) -> usize {
        self.n_vars
    }

    /// Maximum horizon index (cells exist for `0..=horizon`).
    pub fn horizon(&self) -> usize {
        self.horizon
    }

    /// The summary of the response of `variable` to `shock` at `horizon`:
    /// identified-set min/max (weight-invariant) and pointwise weighted
    /// quantiles.
    ///
    /// # Errors
    ///
    /// [`IdentError::RestrictionOutOfRange`] if any index is out of range.
    pub fn summary_point(
        &self,
        variable: usize,
        shock: usize,
        horizon: usize,
    ) -> Result<&IrfBandPoint, IdentError> {
        if variable >= self.n_vars {
            return Err(IdentError::RestrictionOutOfRange {
                what: "response variable",
                index: variable,
                bound: self.n_vars,
            });
        }
        if shock >= self.n_vars {
            return Err(IdentError::RestrictionOutOfRange {
                what: "structural shock",
                index: shock,
                bound: self.n_vars,
            });
        }
        if horizon > self.horizon {
            return Err(IdentError::RestrictionOutOfRange {
                what: "horizon",
                index: horizon,
                bound: self.horizon + 1,
            });
        }
        let idx = (horizon * self.n_vars + variable) * self.n_vars + shock;
        Ok(&self.points[idx])
    }
}

/// The zero + sign restricted sampler configuration.
#[derive(Debug, Clone)]
pub struct ZeroSignSampler {
    horizon: usize,
    n_posterior_draws: usize,
    max_tries_per_draw: usize,
    quantile_probs: Vec<f64>,
    weighted: bool,
}

impl ZeroSignSampler {
    /// A sampler drawing `n_posterior_draws` reduced-form draws, computing
    /// structural IRFs to `horizon`, allowing `max_tries_per_draw` rotation
    /// attempts per draw (only sign restrictions can reject — zeros are
    /// satisfied by construction). Uses the default 5/16/50/84/95 bands and
    /// applies ARW importance weights to the quantiles.
    ///
    /// # Errors
    ///
    /// [`IdentError::InvalidArgument`] if `n_posterior_draws` or
    /// `max_tries_per_draw` is zero.
    pub fn new(
        horizon: usize,
        n_posterior_draws: usize,
        max_tries_per_draw: usize,
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
        Ok(Self {
            horizon,
            n_posterior_draws,
            max_tries_per_draw,
            quantile_probs: DEFAULT_QUANTILE_PROBS.to_vec(),
            weighted: true,
        })
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

    /// Whether to apply ARW importance weights to the pointwise quantiles
    /// (`true` by default). The `set_min`/`set_max` envelope is unaffected —
    /// it is weight-invariant. With impact-only zeros the weights are constant
    /// and this flag has no effect.
    pub fn with_weighting(mut self, weighted: bool) -> Self {
        self.weighted = weighted;
        self
    }

    /// Runs the sampler against a reduced-form NIW posterior, optional sign
    /// restrictions, and a validated zero-restriction set, seeding all
    /// randomness from `seed`.
    ///
    /// # Errors
    ///
    /// * [`IdentError::Dimension`] if the posterior and a restriction set
    ///   disagree on the number of variables;
    /// * [`IdentError::InvalidArgument`] if a restriction set's horizon
    ///   differs from the sampler's;
    /// * [`IdentError::Rng`] if substream spawning fails;
    /// * [`IdentError::Bayes`] on a posterior-draw or Cholesky-IRF failure;
    /// * [`IdentError::Stats`]/[`IdentError::NoConvergence`] on a rotation
    ///   draw failure.
    pub fn run(
        &self,
        posterior: &NiwPosterior,
        signs: Option<&SignRestrictionSet>,
        zeros: &ZeroRestrictionSet,
        seed: u64,
    ) -> Result<ZeroSignSampleResult, IdentError> {
        let n = posterior.n_vars();
        if zeros.n_vars() != n {
            return Err(IdentError::Dimension {
                what: "zero-restriction set and posterior must have the same number of variables",
                expected: n,
                got: zeros.n_vars(),
            });
        }
        if zeros.horizon() != self.horizon {
            return Err(IdentError::InvalidArgument {
                what: "zero-restriction set horizon must equal the sampler horizon",
            });
        }
        if let Some(s) = signs {
            if s.n_vars() != n {
                return Err(IdentError::Dimension {
                    what:
                        "sign-restriction set and posterior must have the same number of variables",
                    expected: n,
                    got: s.n_vars(),
                });
            }
            if s.horizon() != self.horizon {
                return Err(IdentError::InvalidArgument {
                    what: "sign-restriction set horizon must equal the sampler horizon",
                });
            }
        }
        let p = posterior.lag_order();

        let mut substreams = Stream::substreams(seed, self.n_posterior_draws)?;

        let mut draws: Vec<Vec<Mat<f64>>> = Vec::new();
        let mut weights_raw: Vec<f64> = Vec::new();
        let mut rotations_tried = 0usize;

        for stream in substreams.iter_mut() {
            let niw = posterior.draw(stream)?;
            let base = cholesky_irf(niw.b.as_ref(), niw.sigma.as_ref(), p, self.horizon)?;

            match signs {
                None => {
                    // Zeros are satisfied by construction: one rotation, always
                    // accepted, positive-diagonal normalized.
                    rotations_tried += 1;
                    let (q, w) = zero_constrained_rotation(&base, zeros, stream)?;
                    let candidate = structural_irf(&base, q.as_ref());
                    let orient = positive_diagonal_orient(&candidate, n);
                    draws.push(normalize(candidate, &orient));
                    weights_raw.push(w);
                }
                Some(sign_set) => {
                    for _try in 0..self.max_tries_per_draw {
                        rotations_tried += 1;
                        let (q, w) = zero_constrained_rotation(&base, zeros, stream)?;
                        let candidate = structural_irf(&base, q.as_ref());
                        if let Some(orient) = sign_set.accept_orientations(&candidate) {
                            draws.push(normalize(candidate, &orient));
                            weights_raw.push(w);
                            break;
                        }
                    }
                }
            }
        }

        let accepted = draws.len();
        let acceptance_rate = if rotations_tried == 0 {
            0.0
        } else {
            accepted as f64 / rotations_tried as f64
        };
        let diagnostics = SignRestrictionDiagnostics {
            posterior_draws_used: self.n_posterior_draws,
            rotations_tried,
            accepted,
            acceptance_rate,
        };

        let sum_w: f64 = weights_raw.iter().sum();
        let sum_w2: f64 = weights_raw.iter().map(|w| w * w).sum();
        let ess = if sum_w2 > 0.0 {
            sum_w * sum_w / sum_w2
        } else {
            0.0
        };
        let weights: Vec<f64> = if sum_w > 0.0 {
            weights_raw.iter().map(|w| w / sum_w).collect()
        } else {
            Vec::new()
        };

        let points = summarize(
            &draws,
            &weights,
            n,
            self.horizon,
            &self.quantile_probs,
            self.weighted,
        );

        Ok(ZeroSignSampleResult {
            draws,
            weights,
            diagnostics,
            ess,
            probs: self.quantile_probs.clone(),
            n_vars: n,
            horizon: self.horizon,
            points,
        })
    }
}

/// Positive-diagonal sign orientation: `orient[j] = -1` iff the impact
/// response of shock `j` on its own variable is negative, so every shock's
/// own impact is non-negative (the recursive convention). Zero impact keeps
/// `+1`.
fn positive_diagonal_orient(irf: &[Mat<f64>], n: usize) -> Vec<f64> {
    let mut orient = vec![1.0f64; n];
    for (j, o) in orient.iter_mut().enumerate().take(n) {
        if irf[0][(j, j)] < 0.0 {
            *o = -1.0;
        }
    }
    orient
}

/// Applies per-shock sign orientations in place: column `j` scaled by
/// `orient[j]` at every horizon. Duplicated from `sampler.rs` to keep this
/// module collision-free with parallel builders.
fn normalize(mut irf: Vec<Mat<f64>>, orient: &[f64]) -> Vec<Mat<f64>> {
    let n = orient.len();
    for m in irf.iter_mut() {
        for (j, &s) in orient.iter().enumerate().take(n) {
            if s != 1.0 {
                for i in 0..n {
                    m[(i, j)] *= s;
                }
            }
        }
    }
    irf
}

/// Weighted type-7 quantile: values `v` and weights `w` are sorted jointly by
/// value; the plotting positions
/// `pp_k = m (C_k - w_k) / (W (m - 1))` (with `C_k` the inclusive cumulative
/// weight and `W` the total) reduce *exactly* to `k / (m - 1)` for equal
/// weights, so this coincides with the NumPy-default type-7 quantile in the
/// unweighted case and interpolates linearly between order statistics
/// otherwise.
fn weighted_quantile(v: &[f64], w: &[f64], p: f64) -> f64 {
    let m = v.len();
    if m == 0 {
        return f64::NAN;
    }
    if m == 1 {
        return v[0];
    }
    let total: f64 = w.iter().sum();
    if total.is_nan() || total <= 0.0 {
        // Degenerate weights: fall back to an unweighted type-7 position.
        let pos = p * (m - 1) as f64;
        let lo = pos.floor() as usize;
        if lo >= m - 1 {
            return v[m - 1];
        }
        let frac = pos - lo as f64;
        return v[lo] + frac * (v[lo + 1] - v[lo]);
    }
    let mm = m as f64;
    let denom = total * (mm - 1.0);
    let mut pp = vec![0.0f64; m];
    let mut cum = 0.0;
    for k in 0..m {
        cum += w[k];
        pp[k] = mm * (cum - w[k]) / denom;
    }
    if p <= pp[0] {
        return v[0];
    }
    if p >= pp[m - 1] {
        return v[m - 1];
    }
    for k in 0..m - 1 {
        if p >= pp[k] && p <= pp[k + 1] {
            let span = pp[k + 1] - pp[k];
            if span <= 0.0 {
                return v[k + 1];
            }
            let frac = (p - pp[k]) / span;
            return v[k] + frac * (v[k + 1] - v[k]);
        }
    }
    v[m - 1]
}

/// Builds the per-cell min/max (weight-invariant) and weighted-quantile
/// summary. When `weighted` is false the quantiles use equal weights (pure
/// type-7); the min/max are always the raw accepted extremes.
fn summarize(
    draws: &[Vec<Mat<f64>>],
    weights: &[f64],
    n_vars: usize,
    horizon: usize,
    probs: &[f64],
    weighted: bool,
) -> Vec<IrfBandPoint> {
    let n_cells = (horizon + 1) * n_vars * n_vars;
    let mut points = Vec::with_capacity(n_cells);

    let m = draws.len();
    if m == 0 {
        for _ in 0..n_cells {
            points.push(IrfBandPoint {
                min: f64::NAN,
                max: f64::NAN,
                quantiles: vec![f64::NAN; probs.len()],
            });
        }
        return points;
    }

    let equal = vec![1.0f64 / m as f64; m];
    let use_w: &[f64] = if weighted && weights.len() == m {
        weights
    } else {
        &equal
    };

    let mut pairs: Vec<(f64, f64)> = Vec::with_capacity(m);
    for h in 0..=horizon {
        for i in 0..n_vars {
            for j in 0..n_vars {
                pairs.clear();
                for (d, &wt) in draws.iter().zip(use_w.iter()) {
                    pairs.push((d[h][(i, j)], wt));
                }
                pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(core::cmp::Ordering::Equal));
                let min = pairs[0].0;
                let max = pairs[m - 1].0;
                let vals: Vec<f64> = pairs.iter().map(|p| p.0).collect();
                let ws: Vec<f64> = pairs.iter().map(|p| p.1).collect();
                let quantiles = probs
                    .iter()
                    .map(|&pp| weighted_quantile(&vals, &ws, pp))
                    .collect();
                points.push(IrfBandPoint {
                    min,
                    max,
                    quantiles,
                });
            }
        }
    }
    points
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::sign::{Sign, SignRestriction};
    use crate::zero::ZeroRestriction;

    /// Builds a Minnesota-NIW posterior from a small simulated 3-var dataset,
    /// mirroring the binding path (lambda0=100, lambda1=0.2, lambda3=1,
    /// delta=0), so the sampler tests exercise a realistic posterior.
    fn toy_posterior(lags: usize) -> NiwPosterior {
        // Deterministic pseudo-data via a simple LCG (no tsecon RNG needed for
        // a fixture-free smoke posterior).
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
        prior.posterior(data.as_ref()).expect("posterior")
    }

    fn recursive_zeros(n: usize, horizon: usize) -> ZeroRestrictionSet {
        let mut rs = Vec::new();
        for j in 0..n {
            for i in 0..j {
                rs.push(ZeroRestriction::at(i, j, 0));
            }
        }
        ZeroRestrictionSet::new(rs, n, horizon).expect("zeros")
    }

    #[test]
    fn recursive_no_signs_is_lower_triangular_and_unit_weight() -> Result<(), IdentError> {
        let horizon = 6;
        let posterior = toy_posterior(2);
        let zeros = recursive_zeros(3, horizon);
        let res = ZeroSignSampler::new(horizon, 40, 100)?.run(&posterior, None, &zeros, 2026)?;
        let diag = res.diagnostics();
        assert_eq!(diag.accepted, 40);
        assert_eq!(diag.posterior_draws_used, 40);
        assert!((diag.acceptance_rate - 1.0).abs() < 1e-15);
        // Every accepted draw: impact is lower-triangular, positive diagonal.
        for d in res.draws() {
            for i in 0..3 {
                for j in 0..3 {
                    if i < j {
                        assert!(d[0][(i, j)].abs() < 1e-9, "impact upper triangle nonzero");
                    }
                }
                assert!(d[0][(i, i)] >= 0.0, "impact diagonal negative");
            }
        }
        // Weights all equal (impact-only) and ESS == n_accepted.
        let w0 = res.weights()[0];
        for &w in res.weights() {
            assert!((w - w0).abs() < 1e-15);
        }
        assert!((res.ess() - 40.0).abs() < 1e-9);
        // Normalized weights sum to 1.
        let sw: f64 = res.weights().iter().sum();
        assert!((sw - 1.0).abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn zeros_and_signs_are_both_respected() -> Result<(), IdentError> {
        let horizon = 6;
        let posterior = toy_posterior(2);
        // Zero: shock 0 has no impact on variable 2. Sign: shock 0 raises
        // variable 0 on impact.
        let zeros = ZeroRestrictionSet::new(vec![ZeroRestriction::at(2, 0, 0)], 3, horizon)?;
        let signs = SignRestrictionSet::new(
            vec![SignRestriction::at(0, 0, 0, Sign::Positive)],
            3,
            horizon,
        )?;
        let res =
            ZeroSignSampler::new(horizon, 60, 200)?.run(&posterior, Some(&signs), &zeros, 11)?;
        assert!(res.diagnostics().accepted > 0, "nothing accepted");
        for d in res.draws() {
            assert!(d[0][(2, 0)].abs() < 1e-10, "zero restriction violated");
            assert!(d[0][(0, 0)] > 0.0, "sign restriction violated");
        }
        Ok(())
    }

    #[test]
    fn dimension_mismatch_errors() {
        let horizon = 4;
        let posterior = toy_posterior(2);
        // Zero set built for 4 variables against a 3-variable posterior.
        let zeros = ZeroRestrictionSet::new(Vec::new(), 4, horizon).expect("zeros");
        let out = ZeroSignSampler::new(horizon, 10, 50)
            .expect("sampler")
            .run(&posterior, None, &zeros, 0);
        assert!(matches!(out, Err(IdentError::Dimension { .. })));
    }

    #[test]
    fn horizon_mismatch_errors() {
        let posterior = toy_posterior(2);
        let zeros = ZeroRestrictionSet::new(Vec::new(), 3, 5).expect("zeros");
        let out = ZeroSignSampler::new(8, 10, 50)
            .expect("sampler")
            .run(&posterior, None, &zeros, 0);
        assert!(matches!(out, Err(IdentError::InvalidArgument { .. })));
    }

    #[test]
    fn unweighted_matches_weighted_under_equal_weights() -> Result<(), IdentError> {
        let horizon = 4;
        let posterior = toy_posterior(2);
        let zeros = recursive_zeros(3, horizon);
        let weighted = ZeroSignSampler::new(horizon, 30, 100)?
            .with_weighting(true)
            .run(&posterior, None, &zeros, 5)?;
        let unweighted = ZeroSignSampler::new(horizon, 30, 100)?
            .with_weighting(false)
            .run(&posterior, None, &zeros, 5)?;
        // Impact-only weights are constant, so the two agree exactly.
        for h in 0..=horizon {
            for i in 0..3 {
                for j in 0..3 {
                    let a = weighted.summary_point(i, j, h)?;
                    let b = unweighted.summary_point(i, j, h)?;
                    for (qa, qb) in a.quantiles.iter().zip(b.quantiles.iter()) {
                        assert!((qa - qb).abs() < 1e-12);
                    }
                    assert!((a.min - b.min).abs() < 1e-15);
                    assert!((a.max - b.max).abs() < 1e-15);
                }
            }
        }
        Ok(())
    }

    #[test]
    fn weighted_quantile_reduces_to_type7() {
        // Equal weights must reproduce NumPy type-7:
        // np.quantile([1,2,3,4], [0,.25,.5,.75,1]) = [1,1.75,2.5,3.25,4].
        let v = [1.0, 2.0, 3.0, 4.0];
        let w = [0.25, 0.25, 0.25, 0.25];
        assert!((weighted_quantile(&v, &w, 0.0) - 1.0).abs() < 1e-15);
        assert!((weighted_quantile(&v, &w, 0.25) - 1.75).abs() < 1e-15);
        assert!((weighted_quantile(&v, &w, 0.5) - 2.5).abs() < 1e-15);
        assert!((weighted_quantile(&v, &w, 0.75) - 3.25).abs() < 1e-15);
        assert!((weighted_quantile(&v, &w, 1.0) - 4.0).abs() < 1e-15);
    }

    #[test]
    fn deterministic_across_runs_at_fixed_seed() -> Result<(), IdentError> {
        let horizon = 5;
        let posterior = toy_posterior(2);
        let zeros = recursive_zeros(3, horizon);
        let a = ZeroSignSampler::new(horizon, 25, 80)?.run(&posterior, None, &zeros, 99)?;
        let b = ZeroSignSampler::new(horizon, 25, 80)?.run(&posterior, None, &zeros, 99)?;
        for (da, db) in a.draws().iter().zip(b.draws().iter()) {
            for (ma, mb) in da.iter().zip(db.iter()) {
                for i in 0..3 {
                    for j in 0..3 {
                        assert_eq!(ma[(i, j)].to_bits(), mb[(i, j)].to_bits());
                    }
                }
            }
        }
        Ok(())
    }
}
