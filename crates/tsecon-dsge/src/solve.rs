//! The Blanchard-Kahn (1980) solution of the linear rational-expectations model.

use core::fmt;

use tsecon_linalg::faer::c64;
use tsecon_linalg::faer::linalg::solvers::{DenseSolveCore, FullPivLu};
use tsecon_linalg::faer::Mat;

use crate::error::DsgeError;
use crate::model::LinearReModel;

/// Below this the reciprocal-condition proxy of a block is treated as singular.
const SINGULAR_TOL: f64 = 1e-12;
/// Half-width of the dead band around the unit circle: eigenvalues with
/// `| |lambda| - 1 | <= UNIT_BAND` are treated as unit roots (BK undefined).
const UNIT_BAND: f64 = 1e-9;
/// Largest imaginary part tolerated in a policy matrix that should be real.
const IMAG_TOL: f64 = 1e-8;
/// Largest jump-row entry of `N` tolerated before a shock is deemed to load on
/// a jump equation.
const SHOCK_ON_JUMP_TOL: f64 = 1e-9;

/// The Blanchard-Kahn existence/uniqueness verdict.
///
/// Writing `n_unstable` for the number of eigenvalues of `M = A^{-1} B` outside
/// the unit circle and `n_jump` for the number of non-predetermined variables:
///
/// * `n_unstable == n_jump` -> a unique non-explosive solution ([`Self::Unique`]);
/// * `n_unstable <  n_jump` -> a continuum of stable solutions
///   ([`Self::Indeterminate`]);
/// * `n_unstable >  n_jump` -> no stable solution ([`Self::NoStableSolution`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlanchardKahnVerdict {
    /// Exactly as many unstable roots as jump variables: unique and stable.
    Unique {
        /// Number of unstable eigenvalues (`= n_jump`).
        n_unstable: usize,
        /// Number of jump variables.
        n_jump: usize,
    },
    /// Too few unstable roots: the stable manifold is under-determined.
    Indeterminate {
        /// Number of unstable eigenvalues (`< n_jump`).
        n_unstable: usize,
        /// Number of jump variables.
        n_jump: usize,
    },
    /// Too many unstable roots: no non-explosive path exists.
    NoStableSolution {
        /// Number of unstable eigenvalues (`> n_jump`).
        n_unstable: usize,
        /// Number of jump variables.
        n_jump: usize,
    },
}

impl BlanchardKahnVerdict {
    /// True iff the model has a unique non-explosive solution.
    #[must_use]
    pub fn is_unique(&self) -> bool {
        matches!(self, Self::Unique { .. })
    }
}

impl fmt::Display for BlanchardKahnVerdict {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unique { n_unstable, n_jump } => write!(
                f,
                "unique stable solution ({n_unstable} unstable eigenvalue(s) = \
                 {n_jump} jump variable(s))"
            ),
            Self::Indeterminate { n_unstable, n_jump } => write!(
                f,
                "indeterminate: {n_unstable} unstable eigenvalue(s) < {n_jump} \
                 jump variable(s), so a continuum of stable solutions exists — add \
                 a jump variable or a forward-looking equation"
            ),
            Self::NoStableSolution { n_unstable, n_jump } => write!(
                f,
                "no stable solution: {n_unstable} unstable eigenvalue(s) > \
                 {n_jump} jump variable(s), so every path explodes — remove a jump \
                 variable or add a predetermined state"
            ),
        }
    }
}

/// The solved decision rule and state-space law of motion.
///
/// Recovered from a model satisfying the Blanchard-Kahn condition:
///
/// ```text
/// jump_t          = G . predetermined_t
/// predetermined_{t+1} = P . predetermined_t + Q . z_{t+1}
/// ```
///
/// with `G` (`n_jump x n_predetermined`), `P` (`n_predetermined x
/// n_predetermined`), and `Q` (`n_predetermined x n_shocks`).
#[derive(Debug, Clone)]
pub struct DsgeSolution {
    g: Mat<f64>,
    p: Mat<f64>,
    q: Mat<f64>,
    eigenvalues: Vec<c64>,
    verdict: BlanchardKahnVerdict,
    n_pre: usize,
    n_jump: usize,
    n_shocks: usize,
}

impl DsgeSolution {
    /// The policy / decision rule `G`: `jump_t = G predetermined_t`.
    #[must_use]
    pub fn g(&self) -> &Mat<f64> {
        &self.g
    }

    /// The transition `P` of the predetermined law of motion.
    #[must_use]
    pub fn p(&self) -> &Mat<f64> {
        &self.p
    }

    /// The shock loading `Q` of the predetermined law of motion.
    #[must_use]
    pub fn q(&self) -> &Mat<f64> {
        &self.q
    }

    /// The eigenvalues of `M = A^{-1} B`, in the solver's stable-first order.
    #[must_use]
    pub fn eigenvalues(&self) -> &[c64] {
        &self.eigenvalues
    }

    /// The Blanchard-Kahn verdict (always [`BlanchardKahnVerdict::Unique`] for a
    /// solution that was successfully constructed).
    #[must_use]
    pub fn verdict(&self) -> BlanchardKahnVerdict {
        self.verdict
    }

    /// The number of predetermined variables.
    #[must_use]
    pub fn n_predetermined(&self) -> usize {
        self.n_pre
    }

    /// The number of jump variables.
    #[must_use]
    pub fn n_jump(&self) -> usize {
        self.n_jump
    }

    /// The number of shocks.
    #[must_use]
    pub fn n_shocks(&self) -> usize {
        self.n_shocks
    }
}

/// Classifies a model under Blanchard-Kahn without requiring uniqueness.
///
/// Eigen-decomposes `M = A^{-1} B` and returns the verdict together with the
/// eigenvalues (unsorted, in faer's order). Use this to probe a model's
/// determinacy; use [`solve`] to obtain the decision rule when it is unique.
///
/// # Errors
///
/// * [`DsgeError::SingularA`] if `A` is not invertible;
/// * [`DsgeError::EigenFailed`] if the eigendecomposition does not converge;
/// * [`DsgeError::UnitRoot`] if an eigenvalue lies on the unit circle.
pub fn verdict(model: &LinearReModel) -> Result<(BlanchardKahnVerdict, Vec<c64>), DsgeError> {
    let rf = model.reduced_form()?;
    let eig = rf.m.as_ref().eigen().map_err(|_| DsgeError::EigenFailed)?;
    let s = eig.S();
    let col = s.column_vector();
    let n = model.n_variables();
    let mut eigenvalues = Vec::with_capacity(n);
    let mut n_unstable = 0usize;
    for i in 0..n {
        let lambda = col[i];
        let modulus = lambda.re.hypot(lambda.im);
        if (modulus - 1.0).abs() <= UNIT_BAND {
            return Err(DsgeError::UnitRoot { modulus });
        }
        if modulus > 1.0 {
            n_unstable += 1;
        }
        eigenvalues.push(lambda);
    }
    let n_jump = model.n_jump();
    let v = classify(n_unstable, n_jump);
    Ok((v, eigenvalues))
}

/// Solves the model for its decision rule and law of motion by the
/// Blanchard-Kahn method.
///
/// The steps: form `M = A^{-1} B` and `N = A^{-1} C`; eigen-decompose
/// `M = V L V^{-1}`; count the eigenvalues outside the unit circle and check
/// the Blanchard-Kahn condition; from the STABLE eigenvector columns
/// `V_s = [V_ks ; V_xs]` (predetermined rows on top, jump rows below) form the
/// policy rule `G = V_xs V_ks^{-1}`; then `P = M_kk + M_kx G` and `Q = N_k`
/// (the predetermined rows of `N`). The stable eigenvalues of `M` are exactly
/// the eigenvalues of `P`, so `P` is guaranteed stable.
///
/// The eigenvector arithmetic is complex; because the model is real the
/// imaginary parts cancel in `G` and `P`, which are checked to be real to
/// [`IMAG_TOL`].
///
/// # Errors
///
/// * [`DsgeError::SingularA`] if `A` is not invertible;
/// * [`DsgeError::EigenFailed`] if the eigendecomposition does not converge;
/// * [`DsgeError::UnitRoot`] if an eigenvalue lies on the unit circle;
/// * [`DsgeError::BlanchardKahn`] if the model is indeterminate or has no
///   stable solution;
/// * [`DsgeError::ShockOnJump`] if a shock loads on a jump equation;
/// * [`DsgeError::SingularStableBlock`] / [`DsgeError::ComplexSolution`] on a
///   defective/pathological eigenspace.
pub fn solve(model: &LinearReModel) -> Result<DsgeSolution, DsgeError> {
    let n = model.n_variables();
    let n_pre = model.n_predetermined();
    let n_jump = model.n_jump();
    let n_shocks = model.n_shocks();

    let rf = model.reduced_form()?;
    let eig = rf.m.as_ref().eigen().map_err(|_| DsgeError::EigenFailed)?;
    let s = eig.S();
    let evals = s.column_vector();
    let u = eig.U();

    // Partition eigenvalue indices into stable / unstable, rejecting unit roots.
    let mut stable_idx = Vec::new();
    let mut n_unstable = 0usize;
    let mut ordered_eigs = Vec::with_capacity(n);
    for i in 0..n {
        let lambda = evals[i];
        let modulus = lambda.re.hypot(lambda.im);
        if (modulus - 1.0).abs() <= UNIT_BAND {
            return Err(DsgeError::UnitRoot { modulus });
        }
        if modulus < 1.0 {
            stable_idx.push(i);
        } else {
            n_unstable += 1;
        }
    }
    // Stable-first ordering for the reported eigenvalue list.
    for &i in &stable_idx {
        ordered_eigs.push(evals[i]);
    }
    for i in 0..n {
        let lambda = evals[i];
        if lambda.re.hypot(lambda.im) > 1.0 {
            ordered_eigs.push(lambda);
        }
    }

    let bk = classify(n_unstable, n_jump);
    if !bk.is_unique() {
        return Err(DsgeError::BlanchardKahn(bk));
    }
    // Uniqueness guarantees n_stable == n_pre and n_unstable == n_jump.
    debug_assert_eq!(stable_idx.len(), n_pre);

    // Guard the crate's convention: the innovation must not load on a jump row
    // of N = A^{-1} C. (For n_pre == n there are no jump rows and this is void.)
    let mut jump_shock = 0.0f64;
    for i in n_pre..n {
        for j in 0..n_shocks {
            jump_shock = jump_shock.max(rf.n_mat[(i, j)].abs());
        }
    }
    if jump_shock > SHOCK_ON_JUMP_TOL {
        return Err(DsgeError::ShockOnJump {
            magnitude: jump_shock,
        });
    }

    // Build G = V_xs V_ks^{-1} in complex arithmetic, then take its real part.
    let g = if n_jump == 0 {
        Mat::<f64>::zeros(0, n_pre)
    } else {
        policy_rule(u, &stable_idx, n_pre, n_jump)?
    };

    // P = M_kk + M_kx G  (predetermined block of the transition), real.
    let mut p = Mat::<f64>::zeros(n_pre, n_pre);
    for i in 0..n_pre {
        for j in 0..n_pre {
            let mut v = rf.m[(i, j)];
            for l in 0..n_jump {
                v += rf.m[(i, n_pre + l)] * g[(l, j)];
            }
            p[(i, j)] = v;
        }
    }

    // Q = N_k, the predetermined rows of N.
    let mut q = Mat::<f64>::zeros(n_pre, n_shocks);
    for i in 0..n_pre {
        for j in 0..n_shocks {
            q[(i, j)] = rf.n_mat[(i, j)];
        }
    }

    Ok(DsgeSolution {
        g,
        p,
        q,
        eigenvalues: ordered_eigs,
        verdict: bk,
        n_pre,
        n_jump,
        n_shocks,
    })
}

/// Maps `(n_unstable, n_jump)` to a Blanchard-Kahn verdict.
fn classify(n_unstable: usize, n_jump: usize) -> BlanchardKahnVerdict {
    use core::cmp::Ordering::{Equal, Greater, Less};
    match n_unstable.cmp(&n_jump) {
        Equal => BlanchardKahnVerdict::Unique { n_unstable, n_jump },
        Less => BlanchardKahnVerdict::Indeterminate { n_unstable, n_jump },
        Greater => BlanchardKahnVerdict::NoStableSolution { n_unstable, n_jump },
    }
}

/// Forms the real policy matrix `G = V_xs V_ks^{-1}` from the stable eigenvector
/// columns of `U` (`u`), where `V_ks` is the predetermined-row block and `V_xs`
/// the jump-row block.
fn policy_rule(
    u: tsecon_linalg::faer::MatRef<'_, c64>,
    stable_idx: &[usize],
    n_pre: usize,
    n_jump: usize,
) -> Result<Mat<f64>, DsgeError> {
    // V_ks: n_pre x n_pre (predetermined rows, stable columns).
    let mut vks = Mat::<c64>::zeros(n_pre, n_pre);
    for (c, &col) in stable_idx.iter().enumerate() {
        for r in 0..n_pre {
            vks[(r, c)] = u[(r, col)];
        }
    }
    // V_xs: n_jump x n_pre (jump rows, stable columns).
    let mut vxs = Mat::<c64>::zeros(n_jump, n_pre);
    for (c, &col) in stable_idx.iter().enumerate() {
        for r in 0..n_jump {
            vxs[(r, c)] = u[(n_pre + r, col)];
        }
    }

    let lu = FullPivLu::new(vks.as_ref());
    let uf = lu.U();
    let mut max_piv = 0.0f64;
    let mut min_piv = f64::INFINITY;
    for i in 0..n_pre {
        let v = uf[(i, i)].re.hypot(uf[(i, i)].im);
        max_piv = max_piv.max(v);
        min_piv = min_piv.min(v);
    }
    if max_piv == 0.0 || min_piv / max_piv < SINGULAR_TOL {
        return Err(DsgeError::SingularStableBlock);
    }
    let vks_inv = lu.inverse();
    let g_c = vxs.as_ref() * vks_inv.as_ref();

    // Extract the real part, checking the imaginary residual cancels.
    let mut g = Mat::<f64>::zeros(n_jump, n_pre);
    let mut max_imag = 0.0f64;
    for i in 0..n_jump {
        for j in 0..n_pre {
            let z = g_c[(i, j)];
            max_imag = max_imag.max(z.im.abs());
            g[(i, j)] = z.re;
        }
    }
    if max_imag > IMAG_TOL {
        return Err(DsgeError::ComplexSolution { imag: max_imag });
    }
    Ok(g)
}
