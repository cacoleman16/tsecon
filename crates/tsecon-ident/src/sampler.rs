//! The Bayesian sign-restriction sampler: Uhlig (2005) rejection sampling
//! over Haar rotations, drawing reduced-form parameters from a
//! `tsecon-bayes` Normal-inverse-Wishart posterior
//! (Rubio-Ramirez, Waggoner & Zha 2010, "Structural Vector Autoregressions:
//! Theory of Identification and Algorithms for Inference").
//!
//! # Algorithm
//!
//! For each of `n_posterior_draws` reduced-form draws `(B, Sigma)` from the
//! posterior:
//!
//! 1. compute the Cholesky-orthogonalized IRF `Theta^chol_h = Psi_h P` once
//!    (`P = chol(Sigma)`), caching the companion-form recursion so no IRF is
//!    ever recomputed per rotation (roadmap module 06, implementation
//!    warning 11);
//! 2. draw a Haar rotation `Q` and form the candidate structural IRF
//!    `Theta_h = Theta^chol_h Q`;
//! 3. accept if some per-shock sign choice makes every restriction hold,
//!    retrying up to `max_tries_per_draw` times and counting every attempt;
//! 4. on acceptance, store the sign-normalized structural IRF.
//!
//! Acceptance rates decay exponentially in the number of restrictions and
//! are an identification diagnostic in their own right — hence the
//! mandatory [`SignRestrictionDiagnostics`].
//!
//! # Reproducibility and batching invariance
//!
//! Each posterior draw is given its own independent substream (the
//! library-wide parallel Monte Carlo contract,
//! [`Stream::substreams`]). Draw `i` uses substream `i` first for the
//! `(B, Sigma)` draw — which consumes a fixed amount of randomness — then
//! for its rotation attempts. Because a draw's randomness is private to its
//! substream, the accepted rotation for each draw is a deterministic
//! function of `(seed, i)` alone: bit-identical across runs at the same
//! seed, and invariant to `max_tries_per_draw` (a larger budget only lets
//! more draws succeed; it never changes the rotation a succeeding draw
//! settles on). Accepted-set quantiles are therefore stable under
//! `max_tries` batching at a fixed seed.
//!
//! # The Haar prior is informative (Baumeister-Hamilton)
//!
//! The uniform (Haar) prior on rotations is **not** flat over impulse
//! responses: it is informative about the interior of the identified set,
//! and that information does not vanish as the sample grows
//! (Baumeister & Hamilton 2015, Econometrica; roadmap module 06,
//! implementation warning 3). Pointwise posterior quantiles from this
//! sampler therefore mix data information with prior artifact and must not
//! be read as if the prior were uninformative. To let users separate the
//! two, every summary reports, per `(variable, shock, horizon)`, the **min
//! and max across accepted draws** — an estimate of the identified-set
//! *width* — alongside the pointwise quantiles that reflect *sampling*
//! uncertainty. Prior-robust bounds that remove the Haar artifact entirely
//! (Giacomini & Kitagawa 2021) are the principled next step and are
//! `TODO(phase0)` in this crate.

use tsecon_bayes::{cholesky_irf, NiwPosterior};
use tsecon_linalg::faer::Mat;
use tsecon_rng::Stream;

use crate::error::IdentError;
use crate::haar::haar_rotation;
use crate::sign::SignRestrictionSet;
use crate::summary::{normalize, structural_irf, summarize};

/// Default pointwise credible-band probabilities: the 5/16/50/84/95
/// percentiles (the 68% and 90% equal-tailed bands plus the median, the
/// applied-SVAR convention).
const DEFAULT_QUANTILE_PROBS: [f64; 5] = [0.05, 0.16, 0.50, 0.84, 0.95];

/// Mandatory diagnostics for a sign-restriction run. In set-identified
/// settings the diagnostics *are* the inference: a low acceptance rate
/// means the restrictions fight the reduced-form dynamics (or contradict
/// each other), and a run that accepts nothing has identified nothing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SignRestrictionDiagnostics {
    /// Number of posterior `(B, Sigma)` draws processed.
    pub posterior_draws_used: usize,
    /// Total Haar rotations drawn and checked across all posterior draws.
    pub rotations_tried: usize,
    /// Number of posterior draws that yielded an accepted rotation.
    pub accepted: usize,
    /// `accepted / rotations_tried` (0 when no rotation was ever tried).
    pub acceptance_rate: f64,
}

/// Pointwise summary of one `(variable, shock, horizon)` cell across the
/// accepted structural IRF draws.
#[derive(Debug, Clone)]
pub struct IrfBandPoint {
    /// Minimum accepted response — a lower estimate of the identified-set
    /// edge (NaN when no draw was accepted).
    pub min: f64,
    /// Maximum accepted response — an upper estimate of the identified-set
    /// edge (NaN when no draw was accepted).
    pub max: f64,
    /// Pointwise quantiles aligned with [`StructuralIrfSummary::probs`]
    /// (all NaN when no draw was accepted).
    pub quantiles: Vec<f64>,
}

/// Per-cell min/max and pointwise quantiles of the accepted structural IRF
/// draws. Min/max estimate identified-set width; quantiles estimate
/// sampling uncertainty (see the module docs on the Haar-prior caveat).
#[derive(Debug, Clone)]
pub struct StructuralIrfSummary {
    n_vars: usize,
    horizon: usize,
    probs: Vec<f64>,
    /// Row-major over `[horizon][variable][shock]`.
    points: Vec<IrfBandPoint>,
}

impl StructuralIrfSummary {
    /// Assembles a summary from its already-computed parts. `points` must be
    /// row-major over `[horizon][variable][shock]` with
    /// `(horizon + 1) * n_vars * n_vars` entries. Used by the `summary`
    /// module's (weighted and unweighted) band builders so the private field
    /// layout stays encapsulated here.
    pub(crate) fn from_parts(
        n_vars: usize,
        horizon: usize,
        probs: Vec<f64>,
        points: Vec<IrfBandPoint>,
    ) -> Self {
        Self {
            n_vars,
            horizon,
            probs,
            points,
        }
    }

    /// Number of variables (and shocks).
    pub fn n_vars(&self) -> usize {
        self.n_vars
    }

    /// Maximum horizon index (cells exist for `0..=horizon`).
    pub fn horizon(&self) -> usize {
        self.horizon
    }

    /// The quantile probabilities the bands were computed at.
    pub fn probs(&self) -> &[f64] {
        &self.probs
    }

    /// The summary of the response of `variable` to `shock` at `horizon`.
    ///
    /// # Errors
    ///
    /// [`IdentError::RestrictionOutOfRange`] if any index is out of range.
    pub fn point(
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

/// The result of a sign-restriction run: the accepted, sign-normalized
/// structural IRF draws, the mandatory diagnostics, and the pointwise
/// summary.
#[derive(Debug, Clone)]
pub struct SignSampleResult {
    draws: Vec<Vec<Mat<f64>>>,
    diagnostics: SignRestrictionDiagnostics,
    summary: StructuralIrfSummary,
}

impl SignSampleResult {
    /// The accepted structural IRF draws, indexed `[draw][horizon]`, each an
    /// `n x n` matrix with `(i, j)` the sign-normalized response of variable
    /// `i` to structural shock `j`.
    pub fn draws(&self) -> &[Vec<Mat<f64>>] {
        &self.draws
    }

    /// The run diagnostics (always inspect these before reading any band).
    pub fn diagnostics(&self) -> &SignRestrictionDiagnostics {
        &self.diagnostics
    }

    /// The pointwise min/max and quantile summary.
    pub fn summary(&self) -> &StructuralIrfSummary {
        &self.summary
    }
}

/// The Uhlig (2005) rejection sampler configuration.
#[derive(Debug, Clone)]
pub struct SignSampler {
    horizon: usize,
    n_posterior_draws: usize,
    max_tries_per_draw: usize,
    quantile_probs: Vec<f64>,
}

impl SignSampler {
    /// A sampler drawing `n_posterior_draws` reduced-form draws, computing
    /// structural IRFs to `horizon`, and allowing `max_tries_per_draw`
    /// rotation attempts per draw. Uses the default 5/16/50/84/95 bands.
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
        })
    }

    /// Overrides the pointwise band probabilities (each in `[0, 1]`).
    ///
    /// # Errors
    ///
    /// [`IdentError::InvalidArgument`] if the list is empty or any
    /// probability is outside `[0, 1]` or non-finite.
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

    /// Runs the sampler against a reduced-form NIW posterior and a validated
    /// restriction set, seeding all randomness from `seed`.
    ///
    /// # Errors
    ///
    /// * [`IdentError::Dimension`] if the posterior and restriction set
    ///   disagree on the number of variables;
    /// * [`IdentError::InvalidArgument`] if the restriction set's horizon
    ///   differs from the sampler's;
    /// * [`IdentError::Rng`] if substream spawning fails;
    /// * [`IdentError::Bayes`] on a posterior-draw or Cholesky-IRF failure;
    /// * [`IdentError::Stats`]/[`IdentError::NoConvergence`] on a Haar draw
    ///   failure.
    pub fn run(
        &self,
        posterior: &NiwPosterior,
        restrictions: &SignRestrictionSet,
        seed: u64,
    ) -> Result<SignSampleResult, IdentError> {
        let n = posterior.n_vars();
        if restrictions.n_vars() != n {
            return Err(IdentError::Dimension {
                what: "restriction set and posterior must have the same number of variables",
                expected: n,
                got: restrictions.n_vars(),
            });
        }
        if restrictions.horizon() != self.horizon {
            return Err(IdentError::InvalidArgument {
                what: "restriction set horizon must equal the sampler horizon",
            });
        }
        let p = posterior.lag_order();

        // One independent substream per posterior draw: this is what makes
        // the accepted rotation of each draw a deterministic function of
        // (seed, draw index) and hence invariant to max_tries batching.
        let mut substreams = Stream::substreams(seed, self.n_posterior_draws)?;

        let mut draws: Vec<Vec<Mat<f64>>> = Vec::new();
        let mut rotations_tried = 0usize;

        for stream in substreams.iter_mut() {
            let niw = posterior.draw(stream)?;
            let base = cholesky_irf(niw.b.as_ref(), niw.sigma.as_ref(), p, self.horizon)?;

            for _try in 0..self.max_tries_per_draw {
                rotations_tried += 1;
                let q = haar_rotation(n, stream)?;
                let candidate = structural_irf(&base, q.as_ref());
                if let Some(orient) = restrictions.accept_orientations(&candidate) {
                    draws.push(normalize(candidate, &orient));
                    break;
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

        let summary = summarize(&draws, n, self.horizon, &self.quantile_probs);

        Ok(SignSampleResult {
            draws,
            diagnostics,
            summary,
        })
    }
}
