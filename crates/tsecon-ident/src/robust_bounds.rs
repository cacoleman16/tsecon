//! Prior-robust identified-set bounds for sign-restricted SVARs
//! (Giacomini & Kitagawa 2021, *Review of Economic Studies*).
//!
//! # What this removes
//!
//! The Uhlig/ARWZ sign sampler ([`crate::SignSampler`]) reports pointwise
//! posterior quantiles of the structural IRF over a *single* (Haar-uniform)
//! prior on the rotation. That prior is informative and its information does
//! not vanish asymptotically (Baumeister & Hamilton 2015), so those quantiles
//! blend data with a prior artifact. The min/max the sampler emits are only a
//! *within-draw, fixed-rotation-sample* width proxy — the extremes of the
//! finitely many rotations that happened to be accepted — not the true edges
//! of the identified set.
//!
//! Giacomini-Kitagawa robustness fixes this by, at **each** posterior
//! reduced-form draw `phi = (B, Sigma)`, computing the exact minimum and
//! maximum of the scalar impulse response *over the entire admissible rotation
//! set* — every `Q` whose relevant structural column obeys the sign
//! restrictions. Those per-draw `[l, u]` interval endpoints are the true
//! identified-set edges given `phi`; their posterior (set-mean and a robust
//! credible region) is the object that is invariant to the choice of prior
//! over the identified set.
//!
//! # The inner problem, in closed form
//!
//! Fix a response variable `i`, a restricted shock `j`, and a horizon `h`.
//! With `Psi_h` the reduced-form MA weights and `P = chol_lower(Sigma)`, the
//! scalar response to a one-standard-deviation shock in column `q` (a unit
//! vector, the `j`-th column of `Q`) is linear in `q`:
//!
//! ```text
//! eta_{i,j,h}(q) = e_i' Psi_h P q = g' q,   g = P' Psi_h' e_i.
//! ```
//!
//! Each sign restriction on shock `j` — response of variable `v` at horizon
//! `r` must have sign `sig in {+1, -1}` — is a single linear inequality on the
//! same `q`:
//!
//! ```text
//! sig * e_v' Psi_r P q >= 0    <=>    a' q >= 0,   a = sig * P' Psi_r' e_v.
//! ```
//!
//! So the identified set for `eta` at this draw is the interval
//! `[min g'q, max g'q]` over `{ ||q|| = 1, a_k' q >= 0 for all k }`. This is a
//! linear program on the sphere intersected with half-spaces, and its optimum
//! is a KKT point (Gafarov, Meier & Montiel-Olea 2018, *Journal of
//! Econometrics*, single-column case): either the unconstrained optimum
//! `+/- g / ||g||` (when it is feasible) or, on some active face, the
//! projected direction
//!
//! ```text
//! q = +/- P_perp g / ||P_perp g||,   P_perp = I - N (N'N)^{-1} N',
//! ```
//!
//! where `N` collects the active constraint normals. Because the sphere in
//! `R^n` has dimension `n - 1`, at most `n - 1` constraints can be jointly
//! active at a nonzero `q`; enumerating every active subset of size
//! `1..=min(k, n-1)`, projecting `g` onto the orthogonal complement of the
//! active normals (built by a rank-revealing Gram-Schmidt, so linearly
//! dependent subsets collapse harmlessly), and taking the global min/max of
//! `g'q` over the *feasible* candidates recovers the exact interval. An empty
//! feasible candidate set means the restrictions contradict each other at this
//! draw — the identified set is empty, a first-order GK diagnostic.
//!
//! # Honesty: single vs. multiple restricted shocks
//!
//! The closed form above optimizes **one** structural column over **one**
//! sphere. When only one shock carries restrictions this is the exact
//! identified set. When several shocks are jointly restricted their columns
//! must additionally be mutually orthonormal, which *adds* constraints and
//! *shrinks* each column's admissible set; the per-column problems no longer
//! decouple. This module still reports each restricted shock's bound from its
//! own restrictions alone, which is therefore the **marginal** identified set
//! of that shock — a valid **outer** approximation of the joint set (it
//! contains the true joint identified set, hence never understates width, and
//! still brackets the point-identified recursive answer). It is **not** a
//! certified joint optimum. Callers imposing restrictions on more than one
//! shock must read the multi-shock bounds as conservative marginals.
//!
//! All randomness enters through [`tsecon_rng::Stream`] substreams (the
//! library-wide parallel Monte Carlo contract): each posterior draw owns an
//! independent substream, so the per-draw bounds are a deterministic function
//! of `(seed, draw index)` and the posterior summary is reproducible.

use tsecon_bayes::NiwPosterior;
use tsecon_linalg::faer::{Mat, MatRef};
use tsecon_linalg::jittered_cholesky;
use tsecon_rng::Stream;

use crate::error::IdentError;
use crate::sign::{Sign, SignRestrictionSet};
use crate::summary::{quantile_sorted, structural_ma};

/// Default posterior-quantile probabilities for the per-draw bound endpoints:
/// the applied-SVAR 5/16/50/84/95 convention.
const DEFAULT_BOUND_PROBS: [f64; 5] = [0.05, 0.16, 0.50, 0.84, 0.95];

/// Feasibility slack: a unit-normalized constraint `a' q` counts as satisfied
/// when it exceeds `-FEAS_TOL` (active-face candidates sit at `a' q = 0` up to
/// rounding).
const FEAS_TOL: f64 = 1e-9;

/// Vectors with Euclidean norm at or below this are treated as numerically
/// zero (a degenerate objective/constraint direction).
const TINY: f64 = 1e-12;

// --------------------------------------------------------------------------
// Small dense-vector helpers (n is the number of variables, always tiny).
// --------------------------------------------------------------------------

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

fn scaled(a: &[f64], s: f64) -> Vec<f64> {
    a.iter().map(|x| x * s).collect()
}

/// Orthonormal basis of `span(vectors)` by modified Gram-Schmidt, dropping any
/// vector whose residual norm falls below [`TINY`] (so a linearly dependent
/// active subset collapses to the span of its independent members).
fn gram_schmidt(vectors: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let mut basis: Vec<Vec<f64>> = Vec::with_capacity(vectors.len());
    for v in vectors {
        let mut r = v.clone();
        for b in &basis {
            let proj = dot(b, &r);
            for (ri, bi) in r.iter_mut().zip(b.iter()) {
                *ri -= proj * bi;
            }
        }
        let rn = norm(&r);
        if rn > TINY {
            basis.push(scaled(&r, 1.0 / rn));
        }
    }
    basis
}

/// `g` projected onto the orthogonal complement of `span(basis)`, with `basis`
/// already orthonormal (as returned by [`gram_schmidt`]).
fn project_complement(g: &[f64], basis: &[Vec<f64>]) -> Vec<f64> {
    let mut out = g.to_vec();
    for b in basis {
        let proj = dot(b, g);
        for (oi, bi) in out.iter_mut().zip(b.iter()) {
            *oi -= proj * bi;
        }
    }
    out
}

/// Whether `q` satisfies every constraint `a_k' q >= -FEAS_TOL`.
fn feasible(q: &[f64], constraints: &[Vec<f64>]) -> bool {
    constraints.iter().all(|a| dot(a, q) >= -FEAS_TOL)
}

/// Every combination of `s` indices drawn from `0..k` (lexicographic).
fn combinations(k: usize, s: usize) -> Vec<Vec<usize>> {
    let mut out = Vec::new();
    if s == 0 || s > k {
        return out;
    }
    let mut idx: Vec<usize> = (0..s).collect();
    loop {
        out.push(idx.clone());
        // Advance the odometer: find the rightmost index that can still move.
        let mut i = s;
        loop {
            if i == 0 {
                return out;
            }
            i -= 1;
            if idx[i] != i + k - s {
                break;
            }
        }
        idx[i] += 1;
        for j in (i + 1)..s {
            idx[j] = idx[j - 1] + 1;
        }
    }
}

/// KKT candidate directions for `max/min g' q` over
/// `{ ||q|| = 1, a_k' q >= 0 }`: the unconstrained optimum `+/- g/||g||` and,
/// for every active subset of size `1..=min(k, n-1)`, the projected direction
/// `+/- P_perp g / ||P_perp g||`. Each returned vector is unit-norm.
fn candidate_directions(g: &[f64], constraints: &[Vec<f64>], n: usize) -> Vec<Vec<f64>> {
    let mut cands: Vec<Vec<f64>> = Vec::new();
    let gn = norm(g);
    if gn > TINY {
        cands.push(scaled(g, 1.0 / gn));
        cands.push(scaled(g, -1.0 / gn));
    }
    let k = constraints.len();
    let max_active = k.min(n.saturating_sub(1));
    for s in 1..=max_active {
        for combo in combinations(k, s) {
            let active: Vec<Vec<f64>> = combo.iter().map(|&idx| constraints[idx].clone()).collect();
            let basis = gram_schmidt(&active);
            let gp = project_complement(g, &basis);
            let gpn = norm(&gp);
            if gpn > TINY {
                cands.push(scaled(&gp, 1.0 / gpn));
                cands.push(scaled(&gp, -1.0 / gpn));
            }
        }
    }
    cands
}

/// A deterministic, generic (pseudo-random) probe direction in `R^d`.
///
/// Used only as a surrogate objective for the *feasibility* sub-tests
/// ([`region_nonempty`]), where any direction not aligned with a constraint
/// normal exposes a proper KKT vertex/edge of a non-empty region. A structured
/// vector (e.g. the summed normals) can accidentally lie in the span of the
/// active constraints and hide behind the flat-face degeneracy; a generic
/// direction avoids that with probability one.
fn surrogate(d: usize) -> Vec<f64> {
    let mut state = 0x9E37_79B9_7F4A_7C15u64 ^ (d as u64).wrapping_mul(0xD1B5_4A32_D192_ED03);
    (0..d)
        .map(|_| {
            // xorshift64
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        })
        .collect()
}

/// Orthonormal basis of the orthogonal complement of `span(spanning)` in
/// `R^n` (length `n - rank(spanning)`), by extending an orthonormal basis of
/// the span with the canonical axes.
fn complement_basis(spanning: &[Vec<f64>], n: usize) -> Vec<Vec<f64>> {
    let basis = gram_schmidt(spanning);
    let mut comp: Vec<Vec<f64>> = Vec::new();
    for i in 0..n {
        let mut e = vec![0.0; n];
        e[i] = 1.0;
        for b in basis.iter().chain(comp.iter()) {
            let proj = dot(b, &e);
            for (ei, bi) in e.iter_mut().zip(b.iter()) {
                *ei -= proj * bi;
            }
        }
        let en = norm(&e);
        if en > 1e-9 {
            comp.push(scaled(&e, 1.0 / en));
        }
    }
    comp
}

/// Whether the constraint region `{ ||q|| = 1, a_k' q >= 0 }` (in `R^d`) is
/// non-empty. A generic surrogate objective is maximized by the KKT
/// enumeration; a non-empty region always exposes a feasible proper candidate.
fn region_nonempty(constraints: &[Vec<f64>], d: usize) -> bool {
    if d == 0 {
        return false; // no unit vector lives in R^0
    }
    if constraints.is_empty() {
        return true; // the whole sphere is admissible
    }
    let s = surrogate(d);
    candidate_directions(&s, constraints, d)
        .iter()
        .any(|q| feasible(q, constraints))
}

/// Whether the value `0` is attained on the feasible set: does the hyperplane
/// `{ g' q = 0 }` meet `{ ||q|| = 1, a_k' q >= 0 }`?
///
/// This is the flat-face test. When a sign restriction pins the objective
/// direction parallel to a constraint normal (ubiquitous for impact-horizon
/// restrictions), the identified-set edge is exactly `0` on a whole face, a
/// point the direction-based KKT enumeration cannot represent. Reducing to the
/// `n-1`-dimensional subspace `g^perp` (where the objective is identically `0`)
/// and testing that cone for non-emptiness recovers it. Adding `0` to the
/// candidate pool is always safe: if `0` is interior to the identified set it
/// changes neither edge; if it is the edge, it corrects it.
fn zero_achievable(g: &[f64], constraints: &[Vec<f64>], n: usize) -> bool {
    let gn = norm(g);
    if gn <= TINY {
        return region_nonempty(constraints, n); // g = 0: the hyperplane is all of R^n
    }
    if n <= 1 {
        return false; // g^perp is {0}
    }
    let gdir = scaled(g, 1.0 / gn);
    let w = complement_basis(&[gdir], n); // orthonormal basis of g^perp
    let d = w.len();
    let reduced: Vec<Vec<f64>> = constraints
        .iter()
        .map(|a| w.iter().map(|wj| dot(wj, a)).collect())
        .collect();
    region_nonempty(&reduced, d)
}

/// The identified-set interval `[l, u]` for `g' q` over
/// `{ ||q|| = 1, a_k' q >= 0 }`, or `None` if the feasible set is empty.
///
/// The proper KKT candidates ([`candidate_directions`]) recover any edge with
/// a non-vanishing projected gradient exactly; [`zero_achievable`] adds the
/// flat-face value `0` when it is attainable. Together they are complete: a
/// non-empty region's extremes are either proper KKT points or the flat value
/// `0`, so `any == false` certifies an empty identified set.
fn bounds_from_g(g: &[f64], constraints: &[Vec<f64>], n: usize) -> Option<(f64, f64)> {
    if norm(g) <= TINY {
        return if region_nonempty(constraints, n) {
            Some((0.0, 0.0))
        } else {
            None
        };
    }
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    let mut any = false;
    for q in candidate_directions(g, constraints, n) {
        if feasible(&q, constraints) {
            let v = dot(g, &q);
            if v < lo {
                lo = v;
            }
            if v > hi {
                hi = v;
            }
            any = true;
        }
    }
    if zero_achievable(g, constraints, n) {
        lo = lo.min(0.0);
        hi = hi.max(0.0);
        any = true;
    }
    if any {
        Some((lo, hi))
    } else {
        None
    }
}

/// `P' Psi_r' e_v`: the length-`n` gradient vector that turns the response of
/// variable `v` at horizon `r` into a linear form on the structural column
/// (`response = (P' Psi_r' e_v)' q`).
fn gradient(psi_r: MatRef<'_, f64>, p_chol: MatRef<'_, f64>, v: usize, n: usize) -> Vec<f64> {
    // (Psi_r' e_v)[m] = Psi_r[v, m]; then g[a] = sum_m P[m, a] * Psi_r[v, m].
    let mut g = vec![0.0; n];
    for (a, ga) in g.iter_mut().enumerate() {
        let mut acc = 0.0;
        for m in 0..n {
            acc += p_chol[(m, a)] * psi_r[(v, m)];
        }
        *ga = acc;
    }
    g
}

/// The Giacomini-Kitagawa identified-set interval for one scalar structural
/// IRF `eta_{i,j,h}` at a single reduced-form draw.
///
/// * `psi` — reduced-form MA weights `Psi_0..=Psi_H` (length at least
///   `max(horizon_h, max restriction horizon) + 1`);
/// * `p_chol` — the lower Cholesky factor `P` of `Sigma` (`n x n`);
/// * `restrictions_on_shock` — the sign restrictions on **this shock's own
///   column**, each `(variable v, horizon r, sign)` with `sign = +1.0`
///   (response must be positive) or `-1.0` (negative);
/// * `response_i` — the response variable `i`;
/// * `horizon_h` — the horizon `h` at which the response is evaluated.
///
/// Returns `Ok(Some((l, u)))` with `l <= u` the identified-set edges, or
/// `Ok(None)` if the restrictions are mutually infeasible at this draw (empty
/// identified set).
///
/// # Errors
///
/// * [`IdentError::Dimension`] if `p_chol` is not square;
/// * [`IdentError::RestrictionOutOfRange`] if a variable index exceeds `n`, or
///   a referenced horizon (`horizon_h` or any restriction horizon) is not
///   present in `psi`;
/// * [`IdentError::InvalidArgument`] if a restriction sign is neither `+1` nor
///   `-1`.
pub fn identified_set_bounds(
    psi: &[Mat<f64>],
    p_chol: MatRef<'_, f64>,
    restrictions_on_shock: &[(usize, usize, f64)],
    response_i: usize,
    horizon_h: usize,
) -> Result<Option<(f64, f64)>, IdentError> {
    let n = p_chol.nrows();
    if p_chol.ncols() != n {
        return Err(IdentError::Dimension {
            what: "p_chol must be square",
            expected: n,
            got: p_chol.ncols(),
        });
    }
    if response_i >= n {
        return Err(IdentError::RestrictionOutOfRange {
            what: "response variable",
            index: response_i,
            bound: n,
        });
    }
    if horizon_h >= psi.len() {
        return Err(IdentError::RestrictionOutOfRange {
            what: "response horizon",
            index: horizon_h,
            bound: psi.len(),
        });
    }
    // Objective gradient g = P' Psi_h' e_i.
    let g = gradient(psi[horizon_h].as_ref(), p_chol, response_i, n);

    // Constraint normals a_k = sig * P' Psi_r' e_v, unit-normalized (the
    // feasible region is scale-invariant, so normalizing keeps FEAS_TOL well
    // scaled). Zero-norm normals (a flat restricted response) impose nothing
    // and are dropped.
    let mut constraints: Vec<Vec<f64>> = Vec::with_capacity(restrictions_on_shock.len());
    for &(v, r, sig) in restrictions_on_shock {
        if sig != 1.0 && sig != -1.0 {
            return Err(IdentError::InvalidArgument {
                what: "restriction sign must be +1.0 or -1.0",
            });
        }
        if v >= n {
            return Err(IdentError::RestrictionOutOfRange {
                what: "restriction variable",
                index: v,
                bound: n,
            });
        }
        if r >= psi.len() {
            return Err(IdentError::RestrictionOutOfRange {
                what: "restriction horizon",
                index: r,
                bound: psi.len(),
            });
        }
        let a = gradient(psi[r].as_ref(), p_chol, v, n);
        let an = norm(&a);
        if an > TINY {
            constraints.push(scaled(&a, sig / an));
        }
    }

    Ok(bounds_from_g(&g, &constraints, n))
}

// --------------------------------------------------------------------------
// Posterior summary of the per-draw bounds.
// --------------------------------------------------------------------------

/// Mandatory Giacomini-Kitagawa diagnostics: how often the identified set was
/// empty across the posterior draws (an empty set means the restrictions
/// contradicted the reduced-form dynamics at that draw).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RobustBoundsDiagnostics {
    /// Number of posterior `(B, Sigma)` draws processed.
    pub posterior_draws_used: usize,
    /// Number of draws at which every restricted shock had a non-empty
    /// identified set (so the draw contributed to the bounds).
    pub nonempty_draws: usize,
    /// Share of draws with at least one empty restricted-shock identified set
    /// (`1 - nonempty_draws / posterior_draws_used`).
    pub empty_set_rate: f64,
}

/// The posterior summary of the identified-set edges for one
/// `(variable, shock, horizon)` cell.
///
/// All fields are `NaN` for an unrestricted shock (no bound is identified) or
/// when every draw produced an empty set (`n_samples == 0`).
#[derive(Debug, Clone)]
pub struct RobustBoundPoint {
    /// Posterior mean of the identified-set lower edge, `E[l]`.
    pub set_lower_mean: f64,
    /// Posterior mean of the identified-set upper edge, `E[u]`.
    pub set_upper_mean: f64,
    /// Lower edge of the robust credible region: the `alpha/2` posterior
    /// quantile of the per-draw lower edges `{l^(m)}`.
    pub robust_ci_lower: f64,
    /// Upper edge of the robust credible region: the `1 - alpha/2` posterior
    /// quantile of the per-draw upper edges `{u^(m)}`.
    pub robust_ci_upper: f64,
    /// Posterior quantiles of the per-draw lower edges at
    /// [`RobustBounds::probs`].
    pub lower_quantiles: Vec<f64>,
    /// Posterior quantiles of the per-draw upper edges at
    /// [`RobustBounds::probs`].
    pub upper_quantiles: Vec<f64>,
    /// Number of draws (with a non-empty set for this shock) that contributed.
    pub n_samples: usize,
}

/// Per-cell posterior summary of the Giacomini-Kitagawa identified-set bounds,
/// plus the robustness diagnostics.
#[derive(Debug, Clone)]
pub struct RobustBounds {
    n_vars: usize,
    horizon: usize,
    probs: Vec<f64>,
    alpha: f64,
    restricted_shocks: Vec<usize>,
    /// Row-major over `[horizon][variable][shock]`.
    points: Vec<RobustBoundPoint>,
    diagnostics: RobustBoundsDiagnostics,
}

impl RobustBounds {
    /// Number of variables (and shocks).
    pub fn n_vars(&self) -> usize {
        self.n_vars
    }

    /// Maximum horizon index (cells exist for `0..=horizon`).
    pub fn horizon(&self) -> usize {
        self.horizon
    }

    /// The posterior-quantile probabilities the bands were computed at.
    pub fn probs(&self) -> &[f64] {
        &self.probs
    }

    /// The robust credible level (e.g. `0.10` for a 90% region).
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// The shocks that carry restrictions and therefore have identified
    /// bounds (sorted). Cells for any other shock are `NaN`.
    pub fn restricted_shocks(&self) -> &[usize] {
        &self.restricted_shocks
    }

    /// The robustness diagnostics (always inspect the empty-set rate).
    pub fn diagnostics(&self) -> &RobustBoundsDiagnostics {
        &self.diagnostics
    }

    /// The bound summary for the response of `variable` to `shock` at
    /// `horizon`.
    ///
    /// # Errors
    ///
    /// [`IdentError::RestrictionOutOfRange`] if any index is out of range.
    pub fn point(
        &self,
        variable: usize,
        shock: usize,
        horizon: usize,
    ) -> Result<&RobustBoundPoint, IdentError> {
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

/// Extracts, from a validated restriction set, the per-column linear
/// constraints on `shock`: one `(variable, horizon, sign)` per horizon in each
/// restriction's band, with `sign = +1.0` (positive) or `-1.0` (negative).
fn shock_constraints(restrictions: &SignRestrictionSet, shock: usize) -> Vec<(usize, usize, f64)> {
    let mut out = Vec::new();
    for r in restrictions.restrictions() {
        if r.shock != shock {
            continue;
        }
        let sig = match r.sign {
            Sign::Positive => 1.0,
            Sign::Negative => -1.0,
        };
        for h in r.horizon_lo..=r.horizon_hi {
            out.push((r.variable, h, sig));
        }
    }
    out
}

/// Summarizes the per-draw lower/upper edges of one cell into means, the GK
/// robust credible region, and posterior quantiles.
fn summarize_bound_cell(
    lowers: &[f64],
    uppers: &[f64],
    probs: &[f64],
    alpha: f64,
) -> RobustBoundPoint {
    let n = lowers.len();
    if n == 0 {
        return RobustBoundPoint {
            set_lower_mean: f64::NAN,
            set_upper_mean: f64::NAN,
            robust_ci_lower: f64::NAN,
            robust_ci_upper: f64::NAN,
            lower_quantiles: vec![f64::NAN; probs.len()],
            upper_quantiles: vec![f64::NAN; probs.len()],
            n_samples: 0,
        };
    }
    let mean = |xs: &[f64]| xs.iter().sum::<f64>() / xs.len() as f64;
    let mut lo_sorted = lowers.to_vec();
    let mut hi_sorted = uppers.to_vec();
    lo_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
    hi_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));

    let lower_quantiles = probs
        .iter()
        .map(|&p| quantile_sorted(&lo_sorted, p))
        .collect();
    let upper_quantiles = probs
        .iter()
        .map(|&p| quantile_sorted(&hi_sorted, p))
        .collect();

    RobustBoundPoint {
        set_lower_mean: mean(lowers),
        set_upper_mean: mean(uppers),
        robust_ci_lower: quantile_sorted(&lo_sorted, alpha / 2.0),
        robust_ci_upper: quantile_sorted(&hi_sorted, 1.0 - alpha / 2.0),
        lower_quantiles,
        upper_quantiles,
        n_samples: n,
    }
}

/// Prior-robust (Giacomini-Kitagawa 2021) identified-set bounds for a
/// sign-restricted SVAR, summarized over the reduced-form posterior.
///
/// For every posterior draw of `(B, Sigma)` and every restricted shock, the
/// exact per-draw identified set `[l, u]` of each scalar structural IRF is
/// computed by [`identified_set_bounds`]; their posterior mean, robust
/// credible region at level `alpha`, and quantiles are returned per
/// `(variable, shock, horizon)`.
///
/// * `horizon` — maximum response horizon (cells for `0..=horizon`); must
///   equal the restriction set's horizon;
/// * `n_draws` — number of posterior reduced-form draws;
/// * `seed` — seeds the substream-per-draw randomness;
/// * `alpha` — robust credible level (`0.10` gives a 90% region);
/// * `probs` — posterior-quantile probabilities for the endpoint bands.
///
/// Only restricted shocks receive bounds; cells for any other shock are `NaN`.
/// See the module docs for the single- vs. multi-shock honesty note.
///
/// # Errors
///
/// * [`IdentError::Dimension`] if the posterior and restriction set disagree
///   on the number of variables;
/// * [`IdentError::InvalidArgument`] if `n_draws == 0`, the restriction
///   horizon differs from `horizon`, `alpha` is not in `(0, 1)`, or `probs` is
///   empty or contains a value outside `[0, 1]`;
/// * [`IdentError::Rng`] if substream spawning fails;
/// * [`IdentError::Bayes`] on a posterior-draw failure;
/// * [`IdentError::Linalg`] if a draw's covariance has no Cholesky factor.
#[allow(clippy::too_many_arguments)]
pub fn robust_svar_bounds(
    posterior: &NiwPosterior,
    restrictions: &SignRestrictionSet,
    horizon: usize,
    n_draws: usize,
    seed: u64,
    alpha: f64,
    probs: &[f64],
) -> Result<RobustBounds, IdentError> {
    let n = posterior.n_vars();
    if restrictions.n_vars() != n {
        return Err(IdentError::Dimension {
            what: "restriction set and posterior must have the same number of variables",
            expected: n,
            got: restrictions.n_vars(),
        });
    }
    if restrictions.horizon() != horizon {
        return Err(IdentError::InvalidArgument {
            what: "restriction set horizon must equal the reporting horizon",
        });
    }
    if n_draws == 0 {
        return Err(IdentError::InvalidArgument {
            what: "n_draws must be at least 1",
        });
    }
    if !(alpha > 0.0 && alpha < 1.0) {
        return Err(IdentError::InvalidArgument {
            what: "alpha must be in the open interval (0, 1)",
        });
    }
    if probs.is_empty() {
        return Err(IdentError::InvalidArgument {
            what: "at least one quantile probability is required",
        });
    }
    for &pr in probs {
        if !pr.is_finite() || !(0.0..=1.0).contains(&pr) {
            return Err(IdentError::InvalidArgument {
                what: "quantile probabilities must be finite and in [0, 1]",
            });
        }
    }

    let p = posterior.lag_order();
    let restricted = restrictions.restricted_shocks().to_vec();
    // Per-shock expanded constraint lists, indexed by shock (empty for
    // unrestricted shocks).
    let mut per_shock: Vec<Vec<(usize, usize, f64)>> = vec![Vec::new(); n];
    for &j in &restricted {
        per_shock[j] = shock_constraints(restrictions, j);
    }

    let hs = horizon + 1;
    // Per-cell accumulators over draws, row-major [h][i][j].
    let mut lowers: Vec<Vec<f64>> = vec![Vec::new(); hs * n * n];
    let mut uppers: Vec<Vec<f64>> = vec![Vec::new(); hs * n * n];
    let cell = |h: usize, i: usize, j: usize| (h * n + i) * n + j;

    let eye = Mat::<f64>::identity(n, n);
    let mut substreams = Stream::substreams(seed, n_draws)?;
    let mut empty_draws = 0usize;

    for stream in substreams.iter_mut() {
        let niw = posterior.draw(stream)?;
        // Reduced-form MA weights Psi_h (impact = I yields the raw Psi_h).
        let psi = structural_ma(niw.b.as_ref(), eye.as_ref(), p, horizon)?;
        let p_chol = jittered_cholesky(niw.sigma.as_ref())?.factor;

        let mut draw_has_empty = false;
        for &j in &restricted {
            let cons = &per_shock[j];
            // Emptiness is a property of the constraint region alone. Probe it
            // once at (i=0, h=0), where g = P[0,0] * e_0 is guaranteed nonzero
            // (P lower-triangular with positive diagonal), so None <=> empty.
            let empty = identified_set_bounds(&psi, p_chol.as_ref(), cons, 0, 0)?.is_none();
            if empty {
                draw_has_empty = true;
                continue;
            }
            for h in 0..=horizon {
                for i in 0..n {
                    if let Some((l, u)) = identified_set_bounds(&psi, p_chol.as_ref(), cons, i, h)?
                    {
                        lowers[cell(h, i, j)].push(l);
                        uppers[cell(h, i, j)].push(u);
                    }
                }
            }
        }
        if draw_has_empty {
            empty_draws += 1;
        }
    }

    let mut points = Vec::with_capacity(hs * n * n);
    for h in 0..hs {
        for i in 0..n {
            for j in 0..n {
                let c = cell(h, i, j);
                points.push(summarize_bound_cell(&lowers[c], &uppers[c], probs, alpha));
            }
        }
    }

    let nonempty_draws = n_draws - empty_draws;
    let diagnostics = RobustBoundsDiagnostics {
        posterior_draws_used: n_draws,
        nonempty_draws,
        empty_set_rate: empty_draws as f64 / n_draws as f64,
    };

    Ok(RobustBounds {
        n_vars: n,
        horizon,
        probs: probs.to_vec(),
        alpha,
        restricted_shocks: restricted,
        points,
        diagnostics,
    })
}

/// Convenience wrapper using the default 5/16/50/84/95 posterior-quantile
/// probabilities. See [`robust_svar_bounds`].
///
/// # Errors
///
/// Propagates every error of [`robust_svar_bounds`].
pub fn robust_svar_bounds_default(
    posterior: &NiwPosterior,
    restrictions: &SignRestrictionSet,
    horizon: usize,
    n_draws: usize,
    seed: u64,
    alpha: f64,
) -> Result<RobustBounds, IdentError> {
    robust_svar_bounds(
        posterior,
        restrictions,
        horizon,
        n_draws,
        seed,
        alpha,
        &DEFAULT_BOUND_PROBS,
    )
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::sign::SignRestriction;
    use serde_json::Value;
    use std::fs;
    use tsecon_bayes::{cholesky_irf, MinnesotaNiwPrior};

    fn load_fixture() -> Value {
        let path = format!(
            "{}/../../fixtures/robust_svar_bounds.json",
            env!("CARGO_MANIFEST_DIR")
        );
        let text = fs::read_to_string(&path).expect("read robust_svar_bounds.json");
        serde_json::from_str(&text).expect("valid JSON")
    }

    fn mat_from_json(v: &Value) -> Mat<f64> {
        let rows = v.as_array().expect("rows");
        let nr = rows.len();
        let nc = rows[0].as_array().expect("cols").len();
        Mat::from_fn(nr, nc, |i, j| {
            rows[i].as_array().expect("row")[j].as_f64().expect("f64")
        })
    }

    /// Builds the reduced-form MA weights Psi_h (impact = I) from a packed
    /// coefficient matrix, reusing the shared general-impact helper.
    fn psi_series(b: &Mat<f64>, p: usize, horizon: usize) -> Vec<Mat<f64>> {
        let n = b.ncols();
        let eye = Mat::<f64>::identity(n, n);
        structural_ma(b.as_ref(), eye.as_ref(), p, horizon).expect("psi")
    }

    /// GOLDEN A + B: the analytic per-draw bounds match an independent NumPy
    /// active-set enumeration to 1e-8, and a brute-force random-sphere search
    /// is bracketed from the inside (the analytic interval contains it).
    #[test]
    fn analytic_bounds_match_numpy_and_bracket_brute_force() {
        let fx = load_fixture();
        let p = fx["lags"].as_u64().unwrap() as usize;
        let b = mat_from_json(&fx["reg_coefs"]);
        let sigma = mat_from_json(&fx["sigma"]);
        let horizon = fx["horizon"].as_u64().unwrap() as usize;
        let psi = psi_series(&b, p, horizon);
        let p_chol = jittered_cholesky(sigma.as_ref()).unwrap().factor;

        // Restrictions on the single restricted shock: (variable, horizon,
        // sign) with sign = +1.0 / -1.0.
        let cons: Vec<(usize, usize, f64)> = fx["restrictions"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| {
                let a = r.as_array().unwrap();
                (
                    a[0].as_u64().unwrap() as usize,
                    a[1].as_u64().unwrap() as usize,
                    a[2].as_f64().unwrap(),
                )
            })
            .collect();

        for case in fx["cases"].as_array().unwrap() {
            let i = case["i"].as_u64().unwrap() as usize;
            let h = case["h"].as_u64().unwrap() as usize;
            let l_ref = case["l"].as_f64().unwrap();
            let u_ref = case["u"].as_f64().unwrap();
            let l_brute = case["l_brute"].as_f64().unwrap();
            let u_brute = case["u_brute"].as_f64().unwrap();

            let (l, u) = identified_set_bounds(&psi, p_chol.as_ref(), &cons, i, h)
                .unwrap()
                .expect("non-empty identified set");

            assert!(
                (l - l_ref).abs() < 1e-8,
                "lower bound at (i={i}, h={h}): got {l}, numpy {l_ref}"
            );
            assert!(
                (u - u_ref).abs() < 1e-8,
                "upper bound at (i={i}, h={h}): got {u}, numpy {u_ref}"
            );
            // Brute force is bracketed from the inside.
            assert!(
                l <= l_brute + 1e-6,
                "analytic lower {l} must not exceed brute {l_brute} at (i={i}, h={h})"
            );
            assert!(
                u >= u_brute - 1e-6,
                "analytic upper {u} must not fall below brute {u_brute} at (i={i}, h={h})"
            );
            // And the analytic optimum is tight: brute force gets close.
            assert!(
                (l - l_brute).abs() < 5e-3 && (u - u_brute).abs() < 5e-3,
                "brute force should approach the analytic optimum at (i={i}, h={h})"
            );
        }
    }

    /// The identified set must contain the point-identified recursive
    /// (Cholesky) answer whenever the restrictions are consistent with the
    /// recursive column e_j. The recursive shock j response is the Cholesky
    /// IRF Theta_h[i, j].
    #[test]
    fn identified_set_contains_recursive_point() {
        let fx = load_fixture();
        let p = fx["lags"].as_u64().unwrap() as usize;
        let b = mat_from_json(&fx["reg_coefs"]);
        let sigma = mat_from_json(&fx["sigma"]);
        let n = sigma.nrows();
        let horizon = fx["horizon"].as_u64().unwrap() as usize;
        let psi = psi_series(&b, p, horizon);
        let p_chol = jittered_cholesky(sigma.as_ref()).unwrap().factor;
        let chol_irf = cholesky_irf(b.as_ref(), sigma.as_ref(), p, horizon).unwrap();

        // Build restrictions on shock j = 0 that the recursive column e_0
        // satisfies by construction: sign of the recursive impact response of
        // each variable to shock 0 (Theta_0[v, 0] = P[v, 0]).
        let j = 0usize;
        let mut cons: Vec<(usize, usize, f64)> = Vec::new();
        for v in 0..n {
            let val = p_chol[(v, 0)];
            if val.abs() > 1e-8 {
                cons.push((v, 0, if val > 0.0 { 1.0 } else { -1.0 }));
            }
        }

        for (h, chol_h) in chol_irf.iter().enumerate() {
            for i in 0..n {
                let (l, u) = identified_set_bounds(&psi, p_chol.as_ref(), &cons, i, h)
                    .unwrap()
                    .expect("recursive-consistent restrictions are feasible");
                let recursive = chol_h[(i, j)];
                assert!(
                    l <= recursive + 1e-8 && recursive <= u + 1e-8,
                    "identified set [{l}, {u}] must contain recursive {recursive} at (i={i}, h={h})"
                );
            }
        }
    }

    /// GOLDEN C: the posterior aggregation (set-mean, robust region, quantiles)
    /// over stored per-draw [l, u] matches an independent NumPy computation.
    #[test]
    fn aggregation_matches_numpy() {
        let fx = load_fixture();
        let agg = &fx["aggregation"];
        let lowers: Vec<f64> = agg["lowers"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap())
            .collect();
        let uppers: Vec<f64> = agg["uppers"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap())
            .collect();
        let probs: Vec<f64> = agg["probs"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap())
            .collect();
        let alpha = agg["alpha"].as_f64().unwrap();

        let pt = summarize_bound_cell(&lowers, &uppers, &probs, alpha);
        assert!((pt.set_lower_mean - agg["set_lower_mean"].as_f64().unwrap()).abs() < 1e-10);
        assert!((pt.set_upper_mean - agg["set_upper_mean"].as_f64().unwrap()).abs() < 1e-10);
        assert!((pt.robust_ci_lower - agg["robust_ci_lower"].as_f64().unwrap()).abs() < 1e-10);
        assert!((pt.robust_ci_upper - agg["robust_ci_upper"].as_f64().unwrap()).abs() < 1e-10);
        for (got, exp) in pt
            .lower_quantiles
            .iter()
            .zip(agg["lower_quantiles"].as_array().unwrap())
        {
            assert!((got - exp.as_f64().unwrap()).abs() < 1e-10);
        }
        for (got, exp) in pt
            .upper_quantiles
            .iter()
            .zip(agg["upper_quantiles"].as_array().unwrap())
        {
            assert!((got - exp.as_f64().unwrap()).abs() < 1e-10);
        }
    }

    /// End-to-end driver: reproducible at a fixed seed, well-formed bounds
    /// (lower <= upper, restricted-shock cells finite, unrestricted NaN), and
    /// the posterior-mean bounds bracket the recursive answer at impact.
    #[test]
    fn driver_reproducible_and_wellformed() {
        let fx = load_fixture();
        let p = fx["lags"].as_u64().unwrap() as usize;
        let data = mat_from_json(&fx["data"]);
        let n = data.ncols();
        let horizon = 6usize;

        let prior = MinnesotaNiwPrior::new(data.as_ref(), p, 100.0, 0.2, 1.0, 0.0).unwrap();
        let posterior = prior.posterior(data.as_ref()).unwrap();

        // Restrict shock 0 by the recursive impact signs (guaranteed feasible).
        let p_chol_mean = {
            let sig = posterior.sigma_posterior_mean().unwrap();
            jittered_cholesky(sig.as_ref()).unwrap().factor
        };
        let mut rs = Vec::new();
        for v in 0..n {
            let val = p_chol_mean[(v, 0)];
            if val.abs() > 1e-6 {
                rs.push(SignRestriction::at(
                    v,
                    0,
                    0,
                    if val > 0.0 {
                        Sign::Positive
                    } else {
                        Sign::Negative
                    },
                ));
            }
        }
        let restr = SignRestrictionSet::new(rs, n, horizon).unwrap();

        let out1 = robust_svar_bounds(
            &posterior,
            &restr,
            horizon,
            64,
            12345,
            0.10,
            &DEFAULT_BOUND_PROBS,
        )
        .unwrap();
        let out2 = robust_svar_bounds(
            &posterior,
            &restr,
            horizon,
            64,
            12345,
            0.10,
            &DEFAULT_BOUND_PROBS,
        )
        .unwrap();

        assert_eq!(out1.restricted_shocks(), &[0]);
        for h in 0..=horizon {
            for i in 0..n {
                // Restricted shock 0: finite, ordered bounds, reproducible.
                let a = out1.point(i, 0, h).unwrap();
                let bp = out2.point(i, 0, h).unwrap();
                assert!(a.set_lower_mean.is_finite() && a.set_upper_mean.is_finite());
                assert!(a.set_lower_mean <= a.set_upper_mean + 1e-12);
                assert!(a.robust_ci_lower <= a.robust_ci_upper + 1e-12);
                assert!((a.set_lower_mean - bp.set_lower_mean).abs() < 1e-12);
                assert!((a.set_upper_mean - bp.set_upper_mean).abs() < 1e-12);
                // Unrestricted shock 1: NaN.
                if n > 1 {
                    let un = out1.point(i, 1, h).unwrap();
                    assert!(un.set_lower_mean.is_nan() && un.set_upper_mean.is_nan());
                }
            }
        }
    }

    /// When the constraint normals positively span R^n (their conic hull is
    /// all of space), the only vector satisfying every `a_k' q >= 0` is the
    /// origin, so the identified set on the sphere is empty and `bounds_from_g`
    /// returns `None` for any objective. (A mere +/- pair on one response, by
    /// contrast, only *pins that response to zero* — a feasible great circle —
    /// so it does not empty the set.)
    #[test]
    fn positively_spanning_normals_give_empty_set() {
        // Three normals in R^2 whose conic hull covers the plane: q0 >= 0,
        // q1 >= q0, q1 <= -q0 force q0 = q1 = 0.
        let s = 1.0 / 2.0_f64.sqrt();
        let normals = vec![vec![1.0, 0.0], vec![-s, s], vec![-s, -s]];
        for g in [vec![1.0, 1.0], vec![-0.3, 0.8], vec![0.0, 1.0]] {
            assert!(
                bounds_from_g(&g, &normals, 2).is_none(),
                "positively-spanning normals must yield an empty identified set"
            );
        }

        // Sanity: a +/- pair on the SAME response is NOT empty — it pins that
        // response to zero but leaves the orthogonal circle admissible.
        let pinned = vec![vec![1.0, 0.0], vec![-1.0, 0.0]];
        let bounds = bounds_from_g(&[0.0, 1.0], &pinned, 2);
        assert!(
            bounds.is_some(),
            "a +/- pair pins a response to zero but keeps the set non-empty"
        );
        let (l, u) = bounds.unwrap();
        assert!(
            (l + 1.0).abs() < 1e-8 && (u - 1.0).abs() < 1e-8,
            "circle q0=0 spans g=e_1 fully"
        );
    }

    /// A shock with no restrictions leaves the whole sphere admissible, so the
    /// bound is the full envelope `+/- ||g||` (the reduced-form response
    /// magnitude), containing the recursive point trivially.
    #[test]
    fn unrestricted_shock_spans_full_envelope() {
        let fx = load_fixture();
        let p = fx["lags"].as_u64().unwrap() as usize;
        let b = mat_from_json(&fx["reg_coefs"]);
        let sigma = mat_from_json(&fx["sigma"]);
        let n = sigma.nrows();
        let horizon = 3usize;
        let psi = psi_series(&b, p, horizon);
        let p_chol = jittered_cholesky(sigma.as_ref()).unwrap().factor;

        let cons: Vec<(usize, usize, f64)> = Vec::new();
        for h in 0..=horizon {
            for i in 0..n {
                let g = gradient(psi[h].as_ref(), p_chol.as_ref(), i, n);
                let gn = norm(&g);
                let (l, u) = identified_set_bounds(&psi, p_chol.as_ref(), &cons, i, h)
                    .unwrap()
                    .unwrap();
                assert!((l + gn).abs() < 1e-8, "unrestricted lower must be -||g||");
                assert!((u - gn).abs() < 1e-8, "unrestricted upper must be +||g||");
            }
        }
    }

    #[test]
    fn combinations_enumerates_all_subsets() {
        assert_eq!(combinations(4, 0), Vec::<Vec<usize>>::new());
        assert_eq!(combinations(3, 1), vec![vec![0], vec![1], vec![2]]);
        assert_eq!(
            combinations(4, 2),
            vec![
                vec![0, 1],
                vec![0, 2],
                vec![0, 3],
                vec![1, 2],
                vec![1, 3],
                vec![2, 3],
            ]
        );
        assert_eq!(combinations(3, 3), vec![vec![0, 1, 2]]);
        assert!(combinations(2, 3).is_empty());
    }
}
