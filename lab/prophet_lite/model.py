"""Core numerics for prophet_lite: design matrices and MAP estimation.

Model (Taylor & Letham 2017, "Forecasting at Scale")
----------------------------------------------------
    y(t) = g(t) + s(t) + beta' x(t) + eps_t,      eps_t ~ N(0, sigma^2) iid,

with

* piecewise-LINEAR trend
      g(t) = m + k t + sum_j delta_j (t - s_j)_+,
  where s_1 < ... < s_S are candidate changepoints placed uniformly over the
  first ``changepoint_range`` (default 80%) of the sample, and t is calendar
  time rescaled to [0, 1] over the history.  This hinge form is algebraically
  identical to the paper's a(t)' delta slope-adjustment form with the offset
  gamma_j = -s_j delta_j that makes the trend continuous.

* Fourier seasonality
      s(t) = sum_{(P,K)} sum_{n=1..K} [ a_n sin(2 pi n u / P) + b_n cos(2 pi n u / P) ],
  where u is time in the series' native units (days for dated series, the
  integer index otherwise) and P is the period in those units.

* optional extra regressors x(t), standardized as in the reference
  implementation, entering linearly with unpenalized coefficients.

Priors and MAP estimation
-------------------------
Prophet puts a Laplace(0, tau) prior on each changepoint rate adjustment
delta_j (default tau = 0.05) and effectively flat priors on (m, k, beta) and
sigma > 0.  Writing y_s = y / y_scale (y_scale = max |y|, as in the reference
implementation) and X theta for the full linear predictor, the negative log
posterior is, up to constants,

    n log sigma + ||y_s - X theta||^2 / (2 sigma^2) + ||delta||_1 / tau.

Block minimization gives the two exact subproblems we alternate:

* theta-block (sigma fixed): multiply by sigma^2 ->
      min_theta  0.5 ||y_s - X theta||^2 + lam ||delta||_1,   lam = sigma^2 / tau,
  an L1-penalized least-squares problem where only the changepoint block is
  penalized;
* sigma-block (theta fixed): sigma^2 = RSS / n (the profile MLE).

Each block is solved exactly, the joint objective decreases monotonically,
and the alternation converges to a stationary point of the joint MAP problem
(in practice < 10 iterations on every DGP in ``tests.py``; the iteration
count and convergence flag are reported, not hidden).

Solver choice (documented per the lab ground rules)
---------------------------------------------------
The inner theta-problem is solved EXACTLY, not with L-BFGS-B on a smoothed
L1.  Two-step scheme:

1. Frisch-Waugh-Lovell partialling-out: the unpenalized block
   X_u = [1, t, Fourier, extras] is projected out of both y_s and the hinge
   matrix X_d once (dense least squares via SVD).  For a convex objective in
   which X_u's coefficients are unrestricted, minimizing over them first
   reduces the problem to a pure lasso in delta on the residualized data —
   the same device glmnet uses for the unpenalized intercept.
2. Cyclic coordinate descent with soft-thresholding on the residualized
   lasso (Friedman, Hastie & Tibshirani 2010, "Regularization Paths for
   Generalized Linear Models via Coordinate Descent", J. Stat. Software 33).
   The problem is convex, so CD converges to the global minimum; we verify
   the KKT conditions after convergence and report the gap (``kkt_gap``).

Why not L-BFGS-B on a smoothed |.|: smoothing destroys exact zeros in delta,
which the method's semantics rely on (an "active changepoint" is delta_j != 0,
and the tau-path behaviour tested in tests.py needs exact sparsity), and it
introduces a smoothing width to tune.  With S ~ 25 penalized coordinates the
exact CD solve is microseconds; there is no speed argument for smoothing.
(Prophet itself hands the non-smooth posterior to Stan's L-BFGS, which works
but returns near-zeros rather than zeros; our exact solver is the cleaner
statement of the same MAP problem.)

Provenance / licensing
----------------------
Implemented from scratch from the published description: Taylor SJ &
Letham B (2018), "Forecasting at Scale", The American Statistician 72(1)
37-45 (preprint: PeerJ Preprints 5:e3190v2, 2017).  The reference
implementation (facebook/prophet) is MIT-licensed; no code was copied from
it — this module re-derives the estimator from the paper, which carries no
IP restriction.  Coordinate-descent lasso follows Friedman, Hastie &
Tibshirani (2010), a published public algorithm.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "fourier_features",
    "make_changepoint_times",
    "hinge_design",
    "piecewise_linear_trend",
    "lasso_cd",
    "map_fit",
]


# ----------------------------------------------------------------------------
# design matrices
# ----------------------------------------------------------------------------

def fourier_features(u, period, order):
    """Fourier design matrix for one seasonal component.

    Columns are [sin(2 pi 1 u/P), cos(2 pi 1 u/P), ..., sin(2 pi K u/P),
    cos(2 pi K u/P)] — the truncated Fourier series of eq. (5) in Taylor &
    Letham (2017).

    Parameters
    ----------
    u : (n,) array
        Time in native units (days for dated series, index otherwise).
    period : float
        Period P in the same units as ``u``.
    order : int
        Number of harmonics K (Prophet defaults: yearly K=10, weekly K=3).

    Returns
    -------
    (n, 2*order) ndarray
    """
    u = np.asarray(u, dtype=float)
    k = np.arange(1, int(order) + 1)
    ang = 2.0 * np.pi * np.outer(u, k) / float(period)  # (n, K)
    out = np.empty((u.shape[0], 2 * len(k)))
    out[:, 0::2] = np.sin(ang)
    out[:, 1::2] = np.cos(ang)
    return out


def make_changepoint_times(t, n_changepoints, changepoint_range=0.8):
    """Candidate changepoint locations, mirroring the reference implementation.

    Changepoints are placed at observed time points: indices
    ``linspace(0, hist_size - 1, n_changepoints + 1)`` rounded, dropping the
    first (t = 0 carries the base slope k), where
    ``hist_size = floor(changepoint_range * n)``.  ``n_changepoints`` is
    reduced if the sample cannot support it.

    Returns
    -------
    cp_t : (S,) ndarray of scaled times in (0, 1)
    cp_idx : (S,) ndarray of integer sample indices
    """
    t = np.asarray(t, dtype=float)
    n = t.shape[0]
    hist_size = int(np.floor(n * float(changepoint_range)))
    n_cp = int(n_changepoints)
    if n_cp + 1 > hist_size:
        n_cp = max(hist_size - 1, 0)
    if n_cp <= 0:
        return np.empty(0), np.empty(0, dtype=int)
    cp_idx = np.linspace(0, hist_size - 1, n_cp + 1).round().astype(int)[1:]
    cp_idx = np.unique(cp_idx)
    return t[cp_idx], cp_idx


def hinge_design(t, changepoints_t):
    """Hinge (rectified-linear) basis (t - s_j)_+ for the piecewise trend."""
    t = np.asarray(t, dtype=float)
    cp = np.asarray(changepoints_t, dtype=float)
    if cp.size == 0:
        return np.empty((t.shape[0], 0))
    return np.maximum(t[:, None] - cp[None, :], 0.0)


def piecewise_linear_trend(t, m, k, delta, changepoints_t):
    """Evaluate g(t) = m + k t + sum_j delta_j (t - s_j)_+ (scaled units)."""
    t = np.asarray(t, dtype=float)
    g = m + k * t
    if np.size(delta):
        g = g + hinge_design(t, changepoints_t) @ np.asarray(delta, dtype=float)
    return g


# ----------------------------------------------------------------------------
# exact lasso solver (coordinate descent)
# ----------------------------------------------------------------------------

def lasso_cd(X, y, lam, delta0=None, max_iter=10000, tol=1e-10):
    """Exact lasso via cyclic coordinate descent with soft-thresholding.

    Minimizes 0.5 ||y - X d||^2 + lam ||d||_1.  Coordinate update
    (Friedman, Hastie & Tibshirani 2010, eq. 5-6):

        d_j <- soft(x_j' r + ||x_j||^2 d_j, lam) / ||x_j||^2,
        soft(z, g) = sign(z) max(|z| - g, 0).

    The objective is convex, so cyclic CD converges to a global minimizer.
    Columns with (numerically) zero norm — hinges made collinear with the
    unpenalized block by the FWL projection — are pinned at zero.

    Returns
    -------
    d : (p,) solution (exact zeros for inactive coordinates)
    n_sweeps : int
    converged : bool
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, p = X.shape
    d = np.zeros(p) if delta0 is None else np.array(delta0, dtype=float)
    if p == 0:
        return d, 0, True
    norms = np.einsum("ij,ij->j", X, X)
    r = y - X @ d
    converged = False
    sweep = 0
    for sweep in range(1, int(max_iter) + 1):
        max_change = 0.0
        for j in range(p):
            if norms[j] <= 1e-12:
                if d[j] != 0.0:
                    r += X[:, j] * d[j]
                    d[j] = 0.0
                continue
            old = d[j]
            z = X[:, j] @ r + norms[j] * old
            new = np.sign(z) * max(abs(z) - lam, 0.0) / norms[j]
            if new != old:
                r += X[:, j] * (old - new)
                d[j] = new
                max_change = max(max_change, abs(new - old))
        if max_change < tol * max(1.0, np.max(np.abs(d))):
            converged = True
            break
    return d, sweep, converged


def _kkt_gap(X, y, d, lam):
    """Max KKT violation of the lasso solution (0 at an exact optimum).

    Stationarity: x_j'(y - X d) = lam * sign(d_j) on the active set and
    |x_j'(y - X d)| <= lam on the inactive set.
    """
    if X.shape[1] == 0:
        return 0.0
    g = X.T @ (y - X @ d)
    active = d != 0.0
    gap_inactive = max(np.max(np.abs(g[~active])) - lam, 0.0) if np.any(~active) else 0.0
    gap_active = np.max(np.abs(g[active] - lam * np.sign(d[active]))) if np.any(active) else 0.0
    return float(max(gap_inactive, gap_active))


# ----------------------------------------------------------------------------
# MAP fit
# ----------------------------------------------------------------------------

def map_fit(y_s, t, changepoints_t, X_unpen, tau,
            max_sigma_iter=50, cd_max_iter=10000, cd_tol=1e-10):
    """Joint MAP estimate of (m, k, delta, beta, sigma) by exact block descent.

    Alternates (see module docstring for the derivation):

      1. delta | sigma : lasso with lam = sigma^2 / tau on FWL-residualized
         data (exact coordinate descent);
      2. (m, k, beta) | delta : dense least squares;
      3. sigma^2 = RSS / n (profile MLE, exact sigma-block minimizer).

    The FWL projection of the hinge block on the orthogonal complement of
    ``X_unpen`` is computed once (it does not depend on lam).

    Parameters
    ----------
    y_s : (n,) scaled observations (y / y_scale).
    t : (n,) scaled time in [0, 1].
    changepoints_t : (S,) candidate changepoints (scaled).
    X_unpen : (n, q) unpenalized design [1, t, Fourier..., extras...].
    tau : float
        Laplace prior scale on delta (Prophet's changepoint_prior_scale;
        default 0.05 upstream).  LARGER tau = weaker penalty = more active
        changepoints; tau -> 0 shrinks all deltas to exactly zero.

    Returns
    -------
    dict with keys
        b_unpen (q,), delta (S,), sigma_scaled, lam, rss, n_sigma_iter,
        converged, cd_converged, kkt_gap, n_active
    """
    y_s = np.asarray(y_s, dtype=float)
    t = np.asarray(t, dtype=float)
    n = y_s.shape[0]
    Xd = hinge_design(t, changepoints_t)
    S = Xd.shape[1]

    # FWL: residualize y and the hinge block against the unpenalized block.
    # lstsq (SVD) handles possible rank deficiency with the min-norm solution.
    stacked = np.column_stack([y_s, Xd]) if S else y_s[:, None]
    coef, *_ = np.linalg.lstsq(X_unpen, stacked, rcond=None)
    resid = stacked - X_unpen @ coef
    y_til, Xd_til = resid[:, 0], resid[:, 1:]

    # initial sigma^2 from the fully unpenalized (min-norm) fit
    full = np.column_stack([X_unpen, Xd]) if S else X_unpen
    c0, *_ = np.linalg.lstsq(full, y_s, rcond=None)
    rss0 = float(np.sum((y_s - full @ c0) ** 2))
    sigma2 = max(rss0 / n, 1e-16)

    lam = sigma2 / float(tau)
    delta = np.zeros(S)
    converged = False
    cd_ok = True
    it = 0
    for it in range(1, int(max_sigma_iter) + 1):
        delta, _, ok = lasso_cd(Xd_til, y_til, lam, delta0=delta,
                                max_iter=cd_max_iter, tol=cd_tol)
        cd_ok = cd_ok and ok
        partial = y_s - (Xd @ delta if S else 0.0)
        b_u, *_ = np.linalg.lstsq(X_unpen, partial, rcond=None)
        rss = float(np.sum((partial - X_unpen @ b_u) ** 2))
        sigma2 = max(rss / n, 1e-16)
        lam_new = sigma2 / float(tau)
        if abs(lam_new - lam) <= 1e-10 + 1e-8 * lam:
            lam = lam_new
            converged = True
            break
        lam = lam_new

    # final exact polish at the converged lam + KKT certificate
    delta, _, ok = lasso_cd(Xd_til, y_til, lam, delta0=delta,
                            max_iter=cd_max_iter, tol=cd_tol)
    cd_ok = cd_ok and ok
    kkt = _kkt_gap(Xd_til, y_til, delta, lam)
    partial = y_s - (Xd @ delta if S else 0.0)
    b_u, *_ = np.linalg.lstsq(X_unpen, partial, rcond=None)
    rss = float(np.sum((partial - X_unpen @ b_u) ** 2))
    sigma2 = max(rss / n, 1e-16)

    return {
        "b_unpen": b_u,
        "delta": delta,
        "sigma_scaled": float(np.sqrt(sigma2)),
        "lam": float(lam),
        "rss": rss,
        "n_sigma_iter": it,
        "converged": bool(converged),
        "cd_converged": bool(cd_ok),
        "kkt_gap": kkt,
        "n_active": int(np.count_nonzero(delta)),
    }
