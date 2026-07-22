//! # tsecon-ident
//!
//! Structural identification for the tsecon time-series econometrics
//! library (roadmap module 06): the layer that turns reduced-form VAR
//! dynamics into economically interpretable structural shocks. This crate
//! is the first slice — the rotation kernel plus Bayesian sign-restricted
//! SVARs — that the rest of the identification stack (zero, narrative,
//! elasticity restrictions; robust-Bayes bounds; proxy SVARs) will build
//! on.
//!
//! Contents:
//!
//! * [`haar_rotation`] — the Haar-uniform rotation kernel: an orthogonal
//!   `Q` drawn uniformly on `O(m)` via QR of a standard-normal matrix with
//!   the Stewart (1980) / Mezzadri (2007) R-diagonal sign fix. Every
//!   rotation-based identification scheme in the library draws its rotations
//!   here, so a single vetted kernel prevents two subtly different (and one
//!   silently biased) sign-restriction samplers from coexisting;
//! * [`SignRestriction`] / [`Sign`] / [`SignRestrictionSet`] — sign
//!   restrictions on structural impulse responses and the early-exit checker
//!   that evaluates a candidate structural IRF set against them, choosing the
//!   free per-shock sign so the restricted responses carry the user's signs;
//! * [`SignSampler`] — Uhlig (2005) rejection sampling over Haar rotations,
//!   drawing reduced-form parameters from a `tsecon-bayes` Normal-
//!   inverse-Wishart posterior (Rubio-Ramirez, Waggoner & Zha 2010), with
//!   mandatory [`SignRestrictionDiagnostics`], per-shock sign normalization,
//!   and a [`StructuralIrfSummary`] that reports identified-set min/max
//!   bounds alongside pointwise quantiles.
//!
//! # Set identification, honestly
//!
//! Sign restrictions **set-identify**: the data plus the restrictions pin
//! down a *set* of structural models, not a point. Two facts govern every
//! summary this crate emits, and both are documented at the point of use:
//!
//! * **The Haar prior is informative** (Baumeister & Hamilton 2015). The
//!   uniform prior on rotations is not flat over impulse responses; its
//!   information about the interior of the identified set does not vanish
//!   asymptotically. Pointwise posterior quantiles therefore blend data with
//!   prior artifact. To let users see the split, every
//!   [`StructuralIrfSummary`] reports per-`(variable, shock, horizon)` **min
//!   and max across accepted draws** (identified-set width) next to the
//!   pointwise quantiles (sampling uncertainty). Prior-robust
//!   Giacomini-Kitagawa (2021) bounds, which remove the artifact entirely,
//!   are `TODO(phase0)`.
//! * **Median IRFs mix models** (Fry & Pagan 2011). A pointwise median
//!   stitches together responses from mutually inconsistent structural
//!   models; treat the median band as a descriptive summary of the accepted
//!   set, not as a single coherent model. (A Fry-Pagan median-target
//!   rotation is a planned companion.)
//!
//! Randomness enters exclusively through [`tsecon_rng::Stream`] uniforms
//! mapped by inverse CDFs, so every draw is reproducible under the
//! library-wide substream contract, and accepted-set summaries are invariant
//! to `max_tries` batching at a fixed seed.
//!
//! All fallible routines return [`IdentError`]; nothing in this crate
//! panics on user input.

#![warn(missing_docs)]
#![warn(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

pub mod error;
pub mod fry_pagan;
pub mod haar;
pub mod hetero;
pub mod histdecomp;
pub mod long_run;
pub mod max_share;
pub mod narrative;
pub mod nongaussian;
pub mod proxy;
pub mod robust_bounds;
pub mod sampler;
/// Structural-shock extraction primitives (`E = Q' P^-1 U`). Internal to the
/// crate; shared by `histdecomp` and `narrative`.
pub(crate) mod shocks;
pub mod sign;
pub mod structural_fevd;
/// Shared structural-IRF construction and (weighted) band-summary helpers
/// reused across identification schemes. Internal to the crate.
pub(crate) mod summary;
pub mod zero;
pub mod zero_sampler;

pub use error::IdentError;
pub use fry_pagan::{median_target, MedianTargetResult};
pub use haar::haar_rotation;
pub use hetero::{box_m_test, hetero_decompose, BoxMResult, HeteroDecomp, SignConvention};
pub use histdecomp::{decompose, HistoricalDecomposition};
pub use long_run::{long_run_multiplier, long_run_svar, LongRunSvar};
pub use max_share::{max_share_shock, MaxShareResult, MaxShareSign, MaxShareWeighting};
pub use narrative::{
    ContributionRule, HdSetSummary, NarrativeDiagnostics, NarrativeRestriction,
    NarrativeRestrictionSet, NarrativeSampleResult, NarrativeSampler,
};
pub use nongaussian::{nongaussian_svar, Contrast, NonGaussianSvar, OrderBy};
pub use proxy::{proxy_svar, ProxySvarResult};
pub use robust_bounds::{
    identified_set_bounds, robust_svar_bounds, robust_svar_bounds_default, RobustBoundPoint,
    RobustBounds, RobustBoundsDiagnostics,
};
pub use sampler::{
    IrfBandPoint, SignRestrictionDiagnostics, SignSampleResult, SignSampler, StructuralIrfSummary,
};
pub use sign::{Sign, SignRestriction, SignRestrictionSet};
pub use structural_fevd::{structural_fevd, structural_fevd_from_theta};
pub use zero::{zero_constrained_rotation, ZeroRestriction, ZeroRestrictionSet};
pub use zero_sampler::{ZeroSignSampleResult, ZeroSignSampler};

// Re-export the reduced-form posterior type so consumers wire one stack.
pub use tsecon_bayes;
