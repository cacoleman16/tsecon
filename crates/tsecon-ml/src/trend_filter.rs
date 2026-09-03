//! L1 trend filtering (Kim, Koh & Boyd 2009) and its squared-penalty
//! counterpart — the Hodrick-Prescott filter — solved on the banded dual.
//!
//! # Objective
//!
//! For a series `y` of length `n`, a difference order `k ∈ {1, 2}`, and a
//! penalty weight `lam >= 0`, the trend `x` minimizes
//!
//! ```text
//! penalty = L1:   (1/2) ||y - x||_2^2 + lam * ||D_k x||_1
//! penalty = L2:   (1/2) ||y - x||_2^2 + (lam / 2) * ||D_k x||_2^2
//! ```
//!
//! where `D_k` is the `(n - k) x n` `k`-th order difference operator (rows
//! `(-1, 1)` for `k = 1`, `(1, -2, 1)` for `k = 2`). With `k = 2` the `L1`
//! problem is exactly Kim, Koh & Boyd's (2009) *l1 trend filtering*
//! (their eq. 3, same `1/2` and same `lam`): the `L1` norm on second
//! differences produces a **piecewise-linear** trend whose kinks — the
//! *knots* — are chosen by the data, the way the LASSO chooses variables.
//! `k = 1` penalizes first differences and gives a **piecewise-constant**
//! trend (the fused LASSO on the level; Tibshirani et al. 2005; Rudin,
//! Osher & Fatemi's total-variation denoising).
//!
//! The `L2` form with `k = 2` is the Hodrick-Prescott filter under its own
//! parametrization: `||y - x||^2 + lam ||D_2 x||^2` and
//! `(1/2)||y - x||^2 + (lam/2)||D_2 x||^2` have the same minimizer, so
//! `lam` here **is** the HP smoothing parameter (1600 for quarterly data).
//! The two penalties therefore share one `lam` scale on the data-fit side
//! and differ only in the shape of the penalty, which is what makes the
//! comparison between the two trends meaningful.
//!
//! # Algorithm
//!
//! Everything is `O(n)` per step and forms no dense matrix.
//!
//! * **`L2`**: the normal equations `(I + lam D_k' D_k) x = y` — a
//!   symmetric positive-definite system of bandwidth `k` — are solved by a
//!   banded `L D L'` factorization. This is the same computation
//!   `tsecon-filters`' `hp_filter` performs (for `k = 2`), so the two
//!   surfaces must agree to rounding; the Python suite asserts it at
//!   `1e-10`.
//! * **`L1`**: the primal-dual interior-point method of Kim, Koh & Boyd
//!   (2009, section 5) on the **dual** problem
//!
//!   ```text
//!   minimize   (1/2) z' D D' z - z' D y     subject to   ||z||_inf <= lam,
//!   ```
//!
//!   from which the trend is recovered as `x = y - D' z`. `D D'` is banded
//!   (bandwidth `k`), so each Newton step — a solve with
//!   `D D' + diag(positive)` — is a banded `L D L'` factorization in
//!   `O(n)`, and a backtracking line search on the primal-dual residual
//!   keeps the iterate strictly inside the box. Every iterate is
//!   therefore dual feasible, and the reported `duality_gap` is a genuine
//!   certificate: `P(x) - G(z)` with `P` the primal objective at the
//!   returned trend and `G` the dual objective at a feasible `z`, so
//!   `P(x) - P* <= duality_gap` (weak duality). The solver stops when
//!   `duality_gap <= tol * P(x)` — a **relative** certificate, invariant
//!   to rescaling `y` (the objective scales as `y^2`, and so does the gap).
//! * **Polish.** After the interior-point loop the active set is read off
//!   the multipliers (a bound is active where its multiplier exceeds its
//!   slack), the equality-constrained problem on that set is solved
//!   exactly by one more banded solve, and the result is kept only if its
//!   own certified gap is smaller. On a correctly identified active set
//!   this lands the trend at machine precision with the inactive
//!   differences exactly zero, which is what makes `knots` crisp; when
//!   identification fails the interior-point iterate is returned with its
//!   honest gap.
//! * **Closed-form limits.** `lam = 0` returns `x = y`. The dual optimum
//!   `z* = (D D')^{-1} D y` is computed first (one banded solve); its
//!   infinity norm is `lam_max`, the smallest penalty at which the trend
//!   collapses to the ordinary-least-squares polynomial of degree `k - 1`
//!   (the projection of `y` onto the null space of `D_k`). For
//!   `lam >= lam_max` that projection, `y - D' z*`, is returned directly.
//!
//! `converged = false` means the relative gap did not fall below `tol` —
//! either the iteration budget ran out or the iteration **stalled**: the
//! certificate is evaluated in floating point, and its floor is roughly
//! `eps * lam^2 * n * ||D D'|| / P(x)` (the `L1` term pays `lam` on the
//! rounding residue of every inactive difference), measured at 1e-11 to
//! 1e-15 relative on the fixture cases. A `tol` below that floor cannot be
//! certified; ten iterations without a 1% improvement on the best gap end
//! the loop rather than burn the budget, and the last iterate is still
//! returned with its honest certified gap, because for a convex problem a
//! bounded gap is a usable answer. The default `tol = 1e-8` sits two to
//! three decades above the floor.
//!
//! # Knots
//!
//! `knots` lists the indices `i` (into `D_k x`, so `0..n-k`) where
//! `|(D_k x)_i| > max(1e-6 * max_j |(D_k y)_j|, 1e-12 * max_j |y_j|)` — a
//! kink counts when it is at least a millionth of the largest kink in the
//! raw data, with a rounding floor. Under `L1` the inactive differences
//! are exactly zero after a successful polish, so the threshold only
//! resolves the interior-point fallback; under `L2` no difference is ever
//! exactly zero, so `knots` lists (nearly) every index — the count is
//! meaningful for the `L1` trend only.
//!
//! References: Kim, Koh & Boyd (2009), "l1 Trend Filtering", *SIAM
//! Review* 51(2); Tibshirani (2014), "Adaptive piecewise polynomial
//! estimation via trend filtering", *Annals of Statistics* 42(1);
//! Hodrick & Prescott (1997); Phillips & Shi (2021), "Boosting: Why You
//! Can Use the HP Filter", *International Economic Review* 62(2).

use crate::error::MlError;

/// Shape of the penalty on the `order`-th differences of the trend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Penalty {
    /// `lam * ||D x||_1` — sparse differences, a piecewise-polynomial
    /// trend with data-chosen knots (Kim, Koh & Boyd 2009).
    L1,
    /// `(lam/2) * ||D x||_2^2` — the Hodrick-Prescott filter for
    /// `order = 2`, in closed form.
    L2,
}

/// Configuration of [`l1_trend_filter`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrendFilterOptions {
    /// Difference order: `1` (piecewise-constant trend) or `2`
    /// (piecewise-linear trend, the Kim-Koh-Boyd filter).
    pub order: usize,
    /// [`Penalty::L1`] (trend filtering) or [`Penalty::L2`]
    /// (Hodrick-Prescott).
    pub penalty: Penalty,
    /// Relative duality-gap tolerance for the `L1` interior-point solver:
    /// it stops when `duality_gap <= tol * objective`. Values below the
    /// certificate's floating-point floor (~1e-11) return `converged =
    /// false` after the stall detector fires. Inert under `L2`, which is a
    /// closed-form solve.
    pub tol: f64,
    /// Interior-point iteration budget (`L1` only). Each iteration is one
    /// Newton step; typical problems converge in 20-60.
    pub max_iter: usize,
}

impl Default for TrendFilterOptions {
    fn default() -> Self {
        Self {
            order: 2,
            penalty: Penalty::L1,
            tol: 1e-8,
            max_iter: 10_000,
        }
    }
}

/// Result of [`l1_trend_filter`].
#[derive(Debug, Clone, PartialEq)]
pub struct TrendFilterFit {
    /// The estimated trend `x`, length `n`.
    pub trend: Vec<f64>,
    /// `y - trend`, length `n`.
    pub cycle: Vec<f64>,
    /// Indices `i` in `0..n-order` where `(D x)_i` is nonzero beyond the
    /// documented threshold (see the [module docs](self#knots)).
    pub knots: Vec<usize>,
    /// `P(trend) - G(z)`: the primal objective at the returned trend minus
    /// the dual objective at a dual-feasible point — an upper bound on how
    /// far `objective` sits above the true minimum.
    pub duality_gap: f64,
    /// The objective value at `trend` (see the [module docs](self#objective)
    /// for the two conventions).
    pub objective: f64,
    /// `true` when `duality_gap <= tol * objective` (always `true` for the
    /// closed-form `L2`, `lam = 0`, and `lam >= lam_max` paths).
    pub converged: bool,
    /// Interior-point iterations performed (`0` on every closed-form path).
    pub n_iter: usize,
    /// `||(D D')^{-1} D y||_inf`: the smallest `lam` at which the `L1`
    /// trend is the least-squares polynomial of degree `order - 1`.
    pub lam_max: f64,
}

/// Row stencil of `D_k`: the `k`-fold convolution of `(-1, 1)`.
fn stencil(k: usize) -> Vec<f64> {
    let mut c = vec![1.0];
    for _ in 0..k {
        let mut next = vec![0.0; c.len() + 1];
        for (i, &ci) in c.iter().enumerate() {
            next[i] -= ci;
            next[i + 1] += ci;
        }
        c = next;
    }
    c
}

/// `D_k x`: `k` successive first differences.
fn diff_k(x: &[f64], k: usize) -> Vec<f64> {
    let mut v = x.to_vec();
    for _ in 0..k {
        v = v.windows(2).map(|w| w[1] - w[0]).collect();
    }
    v
}

/// `D_k' z`: `k` successive transposed first differences
/// (`(D_1' w)_j = w_{j-1} - w_j` with `w_{-1} = w_L = 0`).
fn diff_k_transpose(z: &[f64], k: usize) -> Vec<f64> {
    let mut v = z.to_vec();
    for _ in 0..k {
        let l = v.len();
        let mut out = vec![0.0; l + 1];
        for (j, o) in out.iter_mut().enumerate() {
            let a = if j >= 1 { v[j - 1] } else { 0.0 };
            let b = if j < l { v[j] } else { 0.0 };
            *o = a - b;
        }
        v = out;
    }
    v
}

/// Symmetric positive-definite banded matrix in lower-band storage:
/// `low[i][bw - (i - j)] = A[i][j]` for `j` in `[i - bw, i]`.
struct Banded {
    n: usize,
    bw: usize,
    low: Vec<Vec<f64>>,
}

/// Its `L D L'` factorization (unit-lower `L` of the same bandwidth).
struct BandedLdl {
    n: usize,
    bw: usize,
    l: Vec<Vec<f64>>,
    d: Vec<f64>,
}

impl Banded {
    fn zeros(n: usize, bw: usize) -> Self {
        Self {
            n,
            bw,
            low: vec![vec![0.0; bw + 1]; n],
        }
    }

    /// Adds `v` to `A[i][j]` (`i >= j`, `i - j <= bw`).
    fn add(&mut self, i: usize, j: usize, v: f64) {
        self.low[i][self.bw - (i - j)] += v;
    }

    fn get(&self, i: usize, j: usize) -> f64 {
        self.low[i][self.bw - (i - j)]
    }

    /// `A v` (symmetric banded matrix-vector product).
    fn matvec(&self, v: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; self.n];
        for i in 0..self.n {
            let lo = i.saturating_sub(self.bw);
            for j in lo..=i {
                let a = self.get(i, j);
                out[i] += a * v[j];
                if j != i {
                    out[j] += a * v[i];
                }
            }
        }
        out
    }

    /// `L D L'` factorization; `None` if a pivot is not positive (cannot
    /// happen in exact arithmetic for the SPD systems built here — it
    /// guards against pathological rounding only).
    fn factor(&self) -> Option<BandedLdl> {
        let (n, bw) = (self.n, self.bw);
        let mut l = vec![vec![0.0; bw + 1]; n];
        let mut d = vec![0.0; n];
        for i in 0..n {
            let lo = i.saturating_sub(bw);
            let mut di = self.get(i, i);
            for j in lo..i {
                let lij = l[i][bw - (i - j)];
                di -= lij * lij * d[j];
            }
            if !(di.is_finite() && di > 0.0) {
                return None;
            }
            d[i] = di;
            let hi = (i + bw + 1).min(n);
            for r in (i + 1)..hi {
                let mut v = self.get(r, i);
                let rlo = r.saturating_sub(bw);
                for j in rlo..i {
                    v -= l[r][bw - (r - j)] * l[i][bw - (i - j)] * d[j];
                }
                l[r][bw - (r - i)] = v / di;
            }
        }
        Some(BandedLdl { n, bw, l, d })
    }
}

impl BandedLdl {
    /// Solves `A x = b`.
    fn solve(&self, b: &[f64]) -> Vec<f64> {
        let (n, bw) = (self.n, self.bw);
        let mut x = b.to_vec();
        // Forward: L u = b.
        for i in 0..n {
            let lo = i.saturating_sub(bw);
            for j in lo..i {
                x[i] -= self.l[i][bw - (i - j)] * x[j];
            }
        }
        // Diagonal.
        for (xi, di) in x.iter_mut().zip(&self.d) {
            *xi /= di;
        }
        // Back: L' x = u.
        for i in (0..n).rev() {
            let hi = (i + bw + 1).min(n);
            for r in (i + 1)..hi {
                x[i] -= self.l[r][bw - (r - i)] * x[r];
            }
        }
        x
    }
}

/// `g[d] = sum_j c_j c_{j+d}`: the entries of `D_k D_k'` by offset.
fn gram_offsets(c: &[f64]) -> Vec<f64> {
    (0..c.len())
        .map(|d| (0..c.len() - d).map(|j| c[j] * c[j + d]).sum())
        .collect()
}

/// `D_k D_k'` (`m x m`, bandwidth `k`) in banded storage.
fn ddt_banded(m: usize, g: &[f64]) -> Banded {
    let k = g.len() - 1;
    let mut a = Banded::zeros(m, k);
    for i in 0..m {
        for (d, &gd) in g.iter().enumerate().take(k.min(i) + 1) {
            a.add(i, i - d, gd);
        }
    }
    a
}

fn norm2_sq(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// The `L1` certificate at dual point `z`: `(trend, D trend, P(trend),
/// G(clip(z)))`. `z` is clipped to the box for the dual objective (so the
/// bound is valid even for a polish candidate that strays outside), while
/// the trend is `y - D' z` for the unclipped `z`.
fn certify_l1(
    y: &[f64],
    dy: &[f64],
    z: &[f64],
    lam: f64,
    k: usize,
) -> (Vec<f64>, Vec<f64>, f64, f64) {
    let dtz = diff_k_transpose(z, k);
    let x: Vec<f64> = y.iter().zip(&dtz).map(|(yi, r)| yi - r).collect();
    let dx = diff_k(&x, k);
    let pobj = 0.5 * norm2_sq(&dtz) + lam * dx.iter().map(|v| v.abs()).sum::<f64>();
    let zc: Vec<f64> = z.iter().map(|v| v.clamp(-lam, lam)).collect();
    let dtzc = diff_k_transpose(&zc, k);
    let dobj = -0.5 * norm2_sq(&dtzc) + dot(&zc, dy);
    (x, dx, pobj, dobj)
}

/// Indices where `|dx_i|` exceeds the documented knot threshold.
fn find_knots(dx: &[f64], dy: &[f64], y: &[f64]) -> Vec<usize> {
    let dy_max = dy.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let y_max = y.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let thr = (1e-6 * dy_max).max(1e-12 * y_max);
    dx.iter()
        .enumerate()
        .filter(|(_, v)| v.abs() > thr)
        .map(|(i, _)| i)
        .collect()
}

/// Kim-Koh-Boyd primal-dual interior-point iteration on the dual box QP.
/// Returns `(z, mu1, mu2, n_iter)`; `z` is strictly inside the box.
fn ipm_l1(
    dy: &[f64],
    ddt: &Banded,
    y: &[f64],
    lam: f64,
    k: usize,
    tol: f64,
    max_iter: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, usize) {
    const MU: f64 = 2.0;
    const ALPHA: f64 = 0.01;
    const BETA: f64 = 0.5;
    const MAX_LS: usize = 20;
    const STALL_ITERS: usize = 10;

    let m = dy.len();
    let mut z = vec![0.0; m];
    let mut mu1 = vec![1.0; m];
    let mut mu2 = vec![1.0; m];
    let mut t = 1e-10f64;
    let mut step = f64::INFINITY;
    let mut n_iter = 0usize;

    let (_, _, mut pobj, mut dobj) = certify_l1(y, dy, &z, lam, k);
    let mut gap = pobj - dobj;
    // Stall detector: the certificate has a floating-point floor (see the
    // module docs), below which the Newton steps collapse without moving
    // the gap. Ten iterations without a 1% improvement on the best gap end
    // the loop; `converged` then reports honestly against `tol`.
    let mut best_gap = gap;
    let mut stalled = 0usize;

    loop {
        if gap <= tol * pobj || n_iter >= max_iter || stalled >= STALL_ITERS {
            break;
        }
        n_iter += 1;
        if step >= 0.2 {
            t = (2.0 * m as f64 * MU / gap).max(1.2 * t);
        }
        let inv_t = 1.0 / t;

        let ddtz = ddt.matvec(&z);
        let f1: Vec<f64> = z.iter().map(|v| v - lam).collect();
        let f2: Vec<f64> = z.iter().map(|v| -v - lam).collect();

        // Newton system S dz = r with S = D D' + diag(-mu1/f1 - mu2/f2).
        let mut s = Banded {
            n: ddt.n,
            bw: ddt.bw,
            low: ddt.low.clone(),
        };
        for i in 0..m {
            s.add(i, i, -mu1[i] / f1[i] - mu2[i] / f2[i]);
        }
        let r: Vec<f64> = (0..m)
            .map(|i| -ddtz[i] + dy[i] + inv_t / f1[i] - inv_t / f2[i])
            .collect();
        let dz = match s.factor() {
            Some(f) => f.solve(&r),
            None => break,
        };
        let dmu1: Vec<f64> = (0..m)
            .map(|i| -(mu1[i] + (inv_t + dz[i] * mu1[i]) / f1[i]))
            .collect();
        let dmu2: Vec<f64> = (0..m)
            .map(|i| -(mu2[i] + (inv_t - dz[i] * mu2[i]) / f2[i]))
            .collect();

        // Residual norm at the current point.
        let res_norm = {
            let mut acc = 0.0;
            for i in 0..m {
                let rd = ddtz[i] - dy[i] + mu1[i] - mu2[i];
                let rc1 = -mu1[i] * f1[i] - inv_t;
                let rc2 = -mu2[i] * f2[i] - inv_t;
                acc += rd * rd + rc1 * rc1 + rc2 * rc2;
            }
            acc.sqrt()
        };

        // Largest step keeping mu > 0, then backtrack on the residual.
        step = 1.0;
        for i in 0..m {
            if dmu1[i] < 0.0 {
                step = step.min(0.99 * (-mu1[i] / dmu1[i]));
            }
            if dmu2[i] < 0.0 {
                step = step.min(0.99 * (-mu2[i] / dmu2[i]));
            }
        }
        let mut newz = z.clone();
        let mut newmu1 = mu1.clone();
        let mut newmu2 = mu2.clone();
        for _ in 0..MAX_LS {
            for i in 0..m {
                newz[i] = z[i] + step * dz[i];
                newmu1[i] = mu1[i] + step * dmu1[i];
                newmu2[i] = mu2[i] + step * dmu2[i];
            }
            let interior = newz.iter().all(|v| v.abs() < lam);
            if interior {
                let new_ddtz = ddt.matvec(&newz);
                let mut acc = 0.0;
                for i in 0..m {
                    let nf1 = newz[i] - lam;
                    let nf2 = -newz[i] - lam;
                    let rd = new_ddtz[i] - dy[i] + newmu1[i] - newmu2[i];
                    let rc1 = -newmu1[i] * nf1 - inv_t;
                    let rc2 = -newmu2[i] * nf2 - inv_t;
                    acc += rd * rd + rc1 * rc1 + rc2 * rc2;
                }
                if acc.sqrt() <= (1.0 - ALPHA * step) * res_norm {
                    break;
                }
            }
            step *= BETA;
        }
        // Accept only an interior point: a fully exhausted line search that
        // still sits outside the box keeps the previous iterate.
        if newz.iter().all(|v| v.abs() < lam) {
            z.clone_from(&newz);
            mu1.clone_from(&newmu1);
            mu2.clone_from(&newmu2);
        }

        let c = certify_l1(y, dy, &z, lam, k);
        pobj = c.2;
        dobj = c.3;
        gap = pobj - dobj;
        if gap < 0.99 * best_gap {
            best_gap = gap;
            stalled = 0;
        } else {
            stalled += 1;
        }
    }
    (z, mu1, mu2, n_iter)
}

/// Active-set polish: reads the active bounds off the multipliers and
/// solves the equality-constrained problem on the inactive set exactly.
/// Returns the candidate dual point (possibly slightly outside the box —
/// the caller certifies it before accepting).
fn polish_l1(
    dy: &[f64],
    g: &[f64],
    z: &[f64],
    mu1: &[f64],
    mu2: &[f64],
    lam: f64,
) -> Option<Vec<f64>> {
    let m = z.len();
    let k = g.len() - 1;
    let mut zp = vec![0.0; m];
    let mut inactive = Vec::with_capacity(m);
    let mut is_active = vec![false; m];
    for i in 0..m {
        if mu1[i] > lam - z[i] {
            zp[i] = lam;
            is_active[i] = true;
        } else if mu2[i] > lam + z[i] {
            zp[i] = -lam;
            is_active[i] = true;
        } else {
            inactive.push(i);
        }
    }
    if inactive.is_empty() {
        return Some(zp);
    }
    // Compressed banded system on the inactive set: adjacent inactive
    // indices within k of each other couple through g[|i - j|]; the
    // compressed bandwidth never exceeds k.
    let q = inactive.len();
    let mut a = Banded::zeros(q, k);
    let mut rhs = vec![0.0; q];
    for (p, &i) in inactive.iter().enumerate() {
        let lo = p.saturating_sub(k);
        for (b, &j) in inactive.iter().enumerate().take(p + 1).skip(lo) {
            let d = i - j;
            if d <= k {
                a.add(p, b, g[d]);
            }
        }
        let mut r = dy[i];
        let jlo = i.saturating_sub(k);
        let jhi = (i + k).min(m - 1);
        for j in jlo..=jhi {
            if is_active[j] {
                let d = i.abs_diff(j);
                r -= g[d] * zp[j];
            }
        }
        rhs[p] = r;
    }
    let sol = a.factor()?.solve(&rhs);
    for (p, &i) in inactive.iter().enumerate() {
        zp[i] = sol[p];
    }
    Some(zp)
}

/// `L2` (Hodrick-Prescott-type) closed form: solves
/// `(I + lam D_k' D_k) x = y` by a bandwidth-`k` `L D L'` factorization.
fn solve_l2(y: &[f64], lam: f64, c: &[f64]) -> Option<Vec<f64>> {
    let n = y.len();
    let k = c.len() - 1;
    let mut a = Banded::zeros(n, k);
    for i in 0..n {
        a.add(i, i, 1.0);
    }
    for r in 0..n - k {
        for ai in 0..=k {
            for bi in 0..=ai {
                a.add(r + ai, r + bi, lam * c[ai] * c[bi]);
            }
        }
    }
    Some(a.factor()?.solve(y))
}

/// L1 trend filtering (Kim, Koh & Boyd 2009) — or, with
/// [`Penalty::L2`], the Hodrick-Prescott filter — on the series `y`.
///
/// Minimizes `(1/2)||y - x||^2 + lam * ||D_k x||_1` (`L1`) or
/// `(1/2)||y - x||^2 + (lam/2) * ||D_k x||^2` (`L2`) over the trend `x`,
/// with `D_k` the `k = opts.order` difference operator; see the
/// [module docs](self) for the algorithm, the certificate, and the knot
/// rule.
///
/// # Errors
///
/// * [`MlError::InsufficientData`] if `y` has fewer than `order + 1`
///   observations (no penalized difference exists);
/// * [`MlError::NonFinite`] on a NaN or infinite entry of `y`;
/// * [`MlError::InvalidArgument`] if `lam` is negative or non-finite,
///   `order` is not `1` or `2`, `tol` is not finite and positive, or
///   `max_iter` is `0`;
/// * [`MlError::DecompositionFailed`] if a banded factorization loses
///   positive definiteness to rounding (unreachable for finite input).
pub fn l1_trend_filter(
    y: &[f64],
    lam: f64,
    opts: TrendFilterOptions,
) -> Result<TrendFilterFit, MlError> {
    let k = opts.order;
    if k != 1 && k != 2 {
        return Err(MlError::InvalidArgument {
            what: "order must be 1 (piecewise-constant trend) or 2 (piecewise-linear trend)",
        });
    }
    let n = y.len();
    if n < k + 1 {
        return Err(MlError::InsufficientData {
            needed: k + 1,
            got: n,
        });
    }
    if y.iter().any(|v| !v.is_finite()) {
        return Err(MlError::NonFinite { what: "y" });
    }
    if !lam.is_finite() || lam < 0.0 {
        return Err(MlError::InvalidArgument {
            what: "lam must be finite and non-negative",
        });
    }
    if !opts.tol.is_finite() || opts.tol <= 0.0 {
        return Err(MlError::InvalidArgument {
            what: "tol must be finite and positive",
        });
    }
    if opts.max_iter == 0 {
        return Err(MlError::InvalidArgument {
            what: "max_iter must be at least 1",
        });
    }

    let m = n - k;
    let c = stencil(k);
    let g = gram_offsets(&c);
    let dy = diff_k(y, k);
    let ddt = ddt_banded(m, &g);
    let ddt_ldl = ddt.factor().ok_or(MlError::DecompositionFailed {
        what: "trend-filter D D' factorization",
    })?;
    let z_star = ddt_ldl.solve(&dy);
    let lam_max = z_star.iter().fold(0.0f64, |acc, v| acc.max(v.abs()));

    let finish = |trend: Vec<f64>,
                  dx: Vec<f64>,
                  gap: f64,
                  objective: f64,
                  n_iter: usize,
                  converged: bool| {
        let cycle: Vec<f64> = y.iter().zip(&trend).map(|(a, b)| a - b).collect();
        let knots = find_knots(&dx, &dy, y);
        TrendFilterFit {
            trend,
            cycle,
            knots,
            duality_gap: gap,
            objective,
            converged,
            n_iter,
            lam_max,
        }
    };

    match opts.penalty {
        Penalty::L2 => {
            let trend = solve_l2(y, lam, &c).ok_or(MlError::DecompositionFailed {
                what: "trend-filter (I + lam D'D) factorization",
            })?;
            let dx = diff_k(&trend, k);
            let resid: Vec<f64> = y.iter().zip(&trend).map(|(a, b)| a - b).collect();
            let objective = 0.5 * norm2_sq(&resid) + 0.5 * lam * norm2_sq(&dx);
            // Dual certificate at v = lam * D x: G(v) = -(1/2)||D'v||^2 +
            // v'Dy - ||v||^2 / (2 lam); at the exact solution P = G.
            let gap = if lam > 0.0 {
                let v: Vec<f64> = dx.iter().map(|d| lam * d).collect();
                let dtv = diff_k_transpose(&v, k);
                let dobj = -0.5 * norm2_sq(&dtv) + dot(&v, &dy) - norm2_sq(&v) / (2.0 * lam);
                objective - dobj
            } else {
                0.0
            };
            Ok(finish(trend, dx, gap, objective, 0, true))
        }
        Penalty::L1 => {
            if lam == 0.0 {
                let trend = y.to_vec();
                let dx = dy.clone();
                return Ok(finish(trend, dx, 0.0, 0.0, 0, true));
            }
            if lam >= lam_max {
                // The dual optimum is interior: the trend is the projection
                // of y onto the degree-(k-1) polynomials.
                let (x, dx, pobj, dobj) = certify_l1(y, &dy, &z_star, lam, k);
                return Ok(finish(x, dx, pobj - dobj, pobj, 0, true));
            }
            let (z, mu1, mu2, n_iter) = ipm_l1(&dy, &ddt, y, lam, k, opts.tol, opts.max_iter);
            let (mut x, mut dx, mut pobj, mut dobj) = certify_l1(y, &dy, &z, lam, k);
            let mut gap = pobj - dobj;
            if let Some(zp) = polish_l1(&dy, &g, &z, &mu1, &mu2, lam) {
                let (xp, dxp, pp, dp) = certify_l1(y, &dy, &zp, lam, k);
                let gp = pp - dp;
                if gp.is_finite() && gp < gap {
                    x = xp;
                    dx = dxp;
                    pobj = pp;
                    dobj = dp;
                    gap = gp;
                }
            }
            let _ = dobj;
            let converged = gap <= opts.tol * pobj;
            Ok(finish(x, dx, gap, pobj, n_iter, converged))
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    #[test]
    fn stencils_and_gram_match_the_closed_forms() {
        assert_eq!(stencil(1), vec![-1.0, 1.0]);
        assert_eq!(stencil(2), vec![1.0, -2.0, 1.0]);
        assert_eq!(gram_offsets(&stencil(1)), vec![2.0, -1.0]);
        assert_eq!(gram_offsets(&stencil(2)), vec![6.0, -4.0, 1.0]);
    }

    #[test]
    fn difference_operators_are_adjoint() {
        // <D x, z> == <x, D' z> for both orders.
        let x: Vec<f64> = (0..9).map(|i| ((i * 7) % 5) as f64 - 1.5).collect();
        for k in [1usize, 2] {
            let z: Vec<f64> = (0..9 - k).map(|i| (i as f64).sin()).collect();
            let lhs = dot(&diff_k(&x, k), &z);
            let rhs = dot(&x, &diff_k_transpose(&z, k));
            assert!((lhs - rhs).abs() < 1e-12, "k={k}: {lhs} vs {rhs}");
        }
    }

    #[test]
    fn banded_solve_matches_dense_elimination() {
        let n = 7;
        let mut a = Banded::zeros(n, 2);
        for i in 0..n {
            a.add(i, i, 5.0 + i as f64);
            if i >= 1 {
                a.add(i, i - 1, -1.5);
            }
            if i >= 2 {
                a.add(i, i - 2, 0.25);
            }
        }
        let b: Vec<f64> = (0..n).map(|i| i as f64 - 2.0).collect();
        let x = a.factor().unwrap().solve(&b);
        let ax = a.matvec(&x);
        for i in 0..n {
            assert!((ax[i] - b[i]).abs() < 1e-12);
        }
    }
}
