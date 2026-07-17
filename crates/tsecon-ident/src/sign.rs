//! Sign restrictions on structural impulse responses and the accept/reject
//! checker that evaluates a candidate structural IRF set against them.
//!
//! A structural IRF set is `Theta_0, ..., Theta_H`, each an `n x n` matrix
//! with `Theta_h[(i, j)]` the response of variable `i` at horizon `h` to
//! structural shock `j`. Under the rotation parameterization
//! `Theta_h = Psi_h P Q` with `Psi_h` the reduced-form MA weights,
//! `P = chol(Sigma)`, and `Q` an orthogonal rotation.

use tsecon_linalg::faer::Mat;

use crate::error::IdentError;

/// The sign a restricted impulse response must take.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sign {
    /// The response must be strictly positive over the restricted horizons.
    Positive,
    /// The response must be strictly negative over the restricted horizons.
    Negative,
}

impl Sign {
    /// Whether a response value satisfies this sign. Exact zeros (a
    /// measure-zero event for a continuous rotation draw) fail both signs,
    /// so a restriction is never vacuously satisfied by a flat response.
    #[inline]
    fn holds(self, value: f64) -> bool {
        match self {
            Sign::Positive => value > 0.0,
            Sign::Negative => value < 0.0,
        }
    }
}

/// A single sign restriction: the response of `variable` to `shock` must
/// have sign `sign` at every horizon in `horizon_lo..=horizon_hi`
/// (inclusive).
///
/// Indices are zero-based. A restriction constrains one `(variable, shock)`
/// cell over a contiguous horizon band; several restrictions on the same
/// shock (across variables and horizons) compose into the shock's full sign
/// pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SignRestriction {
    /// Zero-based index of the response variable `i`.
    pub variable: usize,
    /// Zero-based index of the structural shock `j`.
    pub shock: usize,
    /// First horizon (inclusive) the sign is imposed at.
    pub horizon_lo: usize,
    /// Last horizon (inclusive) the sign is imposed at.
    pub horizon_hi: usize,
    /// The required sign.
    pub sign: Sign,
}

impl SignRestriction {
    /// A restriction imposing `sign` on the response of `variable` to
    /// `shock` at the single horizon `horizon`.
    pub fn at(variable: usize, shock: usize, horizon: usize, sign: Sign) -> Self {
        Self {
            variable,
            shock,
            horizon_lo: horizon,
            horizon_hi: horizon,
            sign,
        }
    }

    /// A restriction imposing `sign` over the inclusive horizon band
    /// `lo..=hi`.
    pub fn over(variable: usize, shock: usize, lo: usize, hi: usize, sign: Sign) -> Self {
        Self {
            variable,
            shock,
            horizon_lo: lo,
            horizon_hi: hi,
            sign,
        }
    }
}

/// A validated collection of sign restrictions against a fixed model
/// dimension `n_vars` and impulse-response horizon `horizon` (the maximum
/// horizon index, so IRF matrices `Theta_0..=Theta_horizon` exist).
///
/// Validation is done once, at construction, so the per-rotation checker in
/// the hot rejection loop can index without bounds surprises.
#[derive(Debug, Clone)]
pub struct SignRestrictionSet {
    restrictions: Vec<SignRestriction>,
    n_vars: usize,
    horizon: usize,
    /// Sorted, deduplicated shock indices that carry at least one
    /// restriction — the shocks the sampler will sign-normalize.
    restricted_shocks: Vec<usize>,
}

impl SignRestrictionSet {
    /// Builds and validates the set: every variable and shock index must be
    /// below `n_vars`, and every horizon must be at most `horizon`.
    ///
    /// # Errors
    ///
    /// * [`IdentError::InvalidArgument`] if `n_vars == 0` or the restriction
    ///   list is empty;
    /// * [`IdentError::RestrictionOutOfRange`] if any variable, shock, or
    ///   horizon index is out of range, or a band has `lo > hi`.
    pub fn new(
        restrictions: Vec<SignRestriction>,
        n_vars: usize,
        horizon: usize,
    ) -> Result<Self, IdentError> {
        if n_vars == 0 {
            return Err(IdentError::InvalidArgument {
                what: "n_vars must be at least 1",
            });
        }
        if restrictions.is_empty() {
            return Err(IdentError::InvalidArgument {
                what: "at least one sign restriction is required to identify a set",
            });
        }
        let mut restricted_shocks = Vec::new();
        for r in &restrictions {
            if r.variable >= n_vars {
                return Err(IdentError::RestrictionOutOfRange {
                    what: "response variable",
                    index: r.variable,
                    bound: n_vars,
                });
            }
            if r.shock >= n_vars {
                return Err(IdentError::RestrictionOutOfRange {
                    what: "structural shock",
                    index: r.shock,
                    bound: n_vars,
                });
            }
            if r.horizon_lo > r.horizon_hi {
                return Err(IdentError::InvalidArgument {
                    what: "sign restriction has horizon_lo greater than horizon_hi",
                });
            }
            if r.horizon_hi > horizon {
                return Err(IdentError::RestrictionOutOfRange {
                    what: "restriction horizon",
                    index: r.horizon_hi,
                    bound: horizon + 1,
                });
            }
            if !restricted_shocks.contains(&r.shock) {
                restricted_shocks.push(r.shock);
            }
        }
        restricted_shocks.sort_unstable();
        Ok(Self {
            restrictions,
            n_vars,
            horizon,
            restricted_shocks,
        })
    }

    /// The validated restrictions.
    pub fn restrictions(&self) -> &[SignRestriction] {
        &self.restrictions
    }

    /// Number of variables the set was validated against.
    pub fn n_vars(&self) -> usize {
        self.n_vars
    }

    /// Maximum horizon index the set was validated against.
    pub fn horizon(&self) -> usize {
        self.horizon
    }

    /// The shock indices that carry at least one restriction (sorted,
    /// deduplicated). These are the shocks the sampler orients and reports;
    /// unrestricted shocks are the nuisance orthogonal complement and are
    /// not sign-identified.
    pub fn restricted_shocks(&self) -> &[usize] {
        &self.restricted_shocks
    }

    /// Chooses a sign orientation `s_j in {+1, -1}` for shock column `j` so
    /// that, applied to `irf`, every restriction on shock `j` holds; returns
    /// `None` if neither orientation works.
    ///
    /// Because a column and its negation are both valid orthonormal shock
    /// directions (negating a column of `Q` keeps `Q` orthogonal), the
    /// shock's sign is free and is chosen here to match the user's specified
    /// pattern. At most one orientation can satisfy a non-degenerate
    /// restriction pattern, so this is the sign normalization: after it, the
    /// shock's restricted responses carry exactly the signs the user asked
    /// for. Reporting a rotation *without* this per-shock sign choice is
    /// incoherent — `Q` and its column-sign flips are observationally
    /// identical, so pointwise bands over unnormalized draws would average a
    /// response against its own negation and collapse toward zero.
    fn orientation_for_shock(&self, irf: &[Mat<f64>], shock: usize) -> Option<f64> {
        for &s in &[1.0f64, -1.0f64] {
            let mut ok = true;
            'restr: for r in &self.restrictions {
                if r.shock != shock {
                    continue;
                }
                for theta in &irf[r.horizon_lo..=r.horizon_hi] {
                    if !r.sign.holds(s * theta[(r.variable, shock)]) {
                        ok = false;
                        break 'restr;
                    }
                }
            }
            if ok {
                return Some(s);
            }
        }
        None
    }

    /// Evaluates a candidate structural IRF set against all restrictions,
    /// returning, on acceptance, the per-shock sign orientations (`+1`/`-1`)
    /// that normalize every restricted shock; returns `None` if any
    /// restricted shock admits no satisfying orientation.
    ///
    /// The returned vector has one entry per variable/shock index `0..n`;
    /// unrestricted shocks get `+1` (their sign is not identified, so the
    /// choice is a convention and they must not be interpreted). Restriction
    /// checking early-exits on the first shock that cannot be oriented — the
    /// acceptance-rate-preserving fast path of Uhlig (2005) rejection
    /// sampling.
    ///
    /// `irf` must have at least `horizon + 1` entries, each at least
    /// `n_vars x n_vars`; the sampler guarantees this, so no bounds error is
    /// surfaced here.
    pub fn accept_orientations(&self, irf: &[Mat<f64>]) -> Option<Vec<f64>> {
        let mut orient = vec![1.0f64; self.n_vars];
        for &shock in &self.restricted_shocks {
            match self.orientation_for_shock(irf, shock) {
                Some(s) => orient[shock] = s,
                None => return None,
            }
        }
        Some(orient)
    }

    /// Whether the candidate structural IRF set satisfies every restriction
    /// under some per-shock sign choice.
    pub fn is_satisfied(&self, irf: &[Mat<f64>]) -> bool {
        self.accept_orientations(irf).is_some()
    }
}
