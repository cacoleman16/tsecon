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
pub mod haar;
pub mod sampler;
pub mod sign;

pub use error::IdentError;
pub use haar::haar_rotation;
pub use sampler::{
    IrfBandPoint, SignRestrictionDiagnostics, SignSampleResult, SignSampler, StructuralIrfSummary,
};
pub use sign::{Sign, SignRestriction, SignRestrictionSet};

// Re-export the reduced-form posterior type so consumers wire one stack.
pub use tsecon_bayes;
