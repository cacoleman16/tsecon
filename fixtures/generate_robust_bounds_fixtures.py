"""Golden fixtures for robust_svar_bounds (tsecon-ident): Giacomini-Kitagawa
(2021) prior-robust identified-set bounds for a sign-restricted SVAR.

METHOD
======
For a single reduced-form draw phi = (B, Sigma), the scalar structural impulse
response of variable i to a one-standard-deviation shock in structural column q
(a unit vector) at horizon h is LINEAR in q:

    eta_{i,h}(q) = e_i' Psi_h P q = g' q,     g = P' Psi_h' e_i,

with Psi_h the reduced-form MA weights and P = chol_lower(Sigma). Each sign
restriction on THIS shock's column -- response of variable v at horizon r must
have sign sig in {+1, -1} -- is one linear inequality on the same q:

    sig * e_v' Psi_r P q >= 0   <=>   a' q >= 0,   a = sig * P' Psi_r' e_v.

The identified set for eta is the interval [min g'q, max g'q] over
{ ||q|| = 1, a_k' q >= 0 for all k }. This is a linear program on the sphere
intersected with half-spaces; its optimum is a KKT point (Gafarov, Meier &
Montiel-Olea 2018, J.Econometrics, single-column case): either the
unconstrained optimum +/- g/||g|| (when feasible) or, on some active face,
+/- P_perp g / ||P_perp g|| with P_perp = I - N (N'N)^{-1} N', N the active
constraint normals. Since the unit sphere in R^n has dimension n-1, at most
n-1 constraints can be jointly active; enumerating every active subset of size
1..min(k, n-1), projecting g onto the complement of the active normals, and
taking the global min/max of g'q over the FEASIBLE candidates gives the exact
interval. No feasible candidate => the identified set is empty.

VALIDATION STRATEGY
===================
Every number is produced by an INDEPENDENT NumPy reference -- numpy.linalg
for the OLS reduced form, cholesky, and the active-set enumeration below --
never by the tsecon Rust crate, so reproducing these numbers in Rust is a
genuine cross-implementation check. This file NEVER imports tsecon.

GOLDEN A (analytic, 1e-8): the active-set bounds [l, u] for several (i, h).
GOLDEN B (brute-force bracket): >= 1e6 random unit vectors filtered by the
    inequalities must be bracketed from the inside by [l, u] (l <= l_brute,
    u >= u_brute) and approach it (tightness).
GOLDEN C (aggregation, 1e-10): set-mean and robust-region quantiles are a
    NumPy aggregation over stored per-draw [l, u].

Run with the project venv:
    .venv/bin/python fixtures/generate_robust_bounds_fixtures.py
"""

import json

import numpy as np
import scipy

OUT = "fixtures/robust_svar_bounds.json"

SEED = 20260722
K = 3          # number of variables
LAGS = 2
T = 300
HORIZON = 8


# --------------------------------------------------------------------------- #
# Structural DGP: a stable k=3 VAR(2).
# --------------------------------------------------------------------------- #
A1 = np.array([[0.50, 0.10, 0.00], [0.05, 0.40, 0.10], [0.00, 0.05, 0.55]])
A2 = np.array([[-0.10, 0.00, 0.05], [0.00, -0.05, 0.00], [0.05, 0.00, -0.10]])
P_TRUE = np.array([[1.00, 0.00, 0.00], [0.40, 0.90, 0.00], [0.20, 0.30, 0.70]])


def simulate(seed, t, a1, a2, p, k, burn=200):
    rng = np.random.default_rng(seed)
    n = t + burn
    eps = rng.standard_normal((n, k))
    y = np.zeros((n, k))
    for tt in range(2, n):
        y[tt] = a1 @ y[tt - 1] + a2 @ y[tt - 2] + p @ eps[tt]
    return y[burn:]


def ols_var(data, p):
    """Independent OLS reduced form; df-adjusted Sigma matching VarResults.

    Returns packed B (rows = 1 + k*p in order [const, lag1, ..., lagp];
    cols = k) and Sigma.
    """
    t, k = data.shape
    design = np.column_stack(
        [np.ones(t - p)] + [data[p - l : t - l] for l in range(1, p + 1)]
    )
    yt = data[p:]
    b, *_ = np.linalg.lstsq(design, yt, rcond=None)
    u = yt - design @ b
    nobs = t - p
    n_params = 1 + k * p
    sigma = u.T @ u / (nobs - n_params)
    return b, sigma


def psi_series(b, p, k, horizon):
    """Reduced-form MA weights Psi_s = J F^s J', s = 0..horizon."""
    # A_l[i, j] = coefficient of y_{t-l, j} in equation i.
    a_mats = [b[1 + l * k : 1 + (l + 1) * k, :].T for l in range(p)]
    npp = k * p
    comp = np.zeros((npp, npp))
    for l in range(p):
        comp[:k, l * k : (l + 1) * k] = a_mats[l]
    if p > 1:
        comp[k:, : k * (p - 1)] = np.eye(k * (p - 1))
    psis = []
    f_pow = np.eye(npp)
    for _h in range(horizon + 1):
        psis.append(f_pow[:k, :k].copy())
        f_pow = comp @ f_pow
    return psis


def gradient(psi_r, p_chol, v):
    """g = P' Psi_r' e_v  (length k)."""
    return p_chol.T @ psi_r[v, :]


FEAS_TOL = 1e-9
TINY = 1e-12


def candidate_directions(g, normals, d):
    """Unit KKT candidate directions: +/- g/||g|| and, for every active subset
    of size 1..min(k, d-1), +/- P_perp g / ||P_perp g||."""
    from itertools import combinations

    cands = []
    gn = np.linalg.norm(g)
    if gn > TINY:
        cands.append(g / gn)
        cands.append(-g / gn)
    kk = len(normals)
    max_active = min(kk, d - 1)
    for s in range(1, max_active + 1):
        for combo in combinations(range(kk), s):
            n_mat = np.array([normals[idx] for idx in combo]).T  # d x s
            q_basis, r_basis = np.linalg.qr(n_mat)
            rank = int(np.sum(np.abs(np.diag(r_basis)) > TINY))
            basis = q_basis[:, :rank]
            gp = g - basis @ (basis.T @ g)
            gpn = np.linalg.norm(gp)
            if gpn > TINY:
                cands.append(gp / gpn)
                cands.append(-gp / gpn)
    return cands


def _surrogate(d):
    """Deterministic generic probe direction (a fixed-seed normal draw)."""
    return np.random.default_rng(1234 + d).standard_normal(d)


def region_nonempty(normals, d):
    """Is {||z||=1, c' z >= 0} non-empty in R^d? A generic surrogate objective
    exposes a proper KKT candidate of any non-empty cone-cap."""
    if d == 0:
        return False
    if len(normals) == 0:
        return True
    s = _surrogate(d)
    for q in candidate_directions(s, normals, d):
        if all(np.dot(a, q) >= -FEAS_TOL for a in normals):
            return True
    return False


def complement_basis(spanning, n):
    """Orthonormal basis of the complement of span(spanning) in R^n."""
    q_basis, _ = np.linalg.qr(np.array(spanning).T)
    comp = []
    have = [q_basis[:, j] for j in range(q_basis.shape[1])]
    for i in range(n):
        e = np.zeros(n)
        e[i] = 1.0
        for b in have + comp:
            e = e - np.dot(b, e) * b
        en = np.linalg.norm(e)
        if en > 1e-9:
            comp.append(e / en)
    return comp


def zero_achievable(g, normals, n):
    """Does the hyperplane {g' q = 0} meet {||q||=1, c' q >= 0}?"""
    gn = np.linalg.norm(g)
    if gn <= TINY:
        return region_nonempty(normals, n)
    if n <= 1:
        return False
    w = complement_basis([g / gn], n)  # basis of g^perp
    d = len(w)
    reduced = [np.array([np.dot(wj, a) for wj in w]) for a in normals]
    return region_nonempty(reduced, d)


def analytic_bounds(g, normals, k):
    """Exact [l, u] of g'q over {||q||=1, a' q >= 0} by KKT active-set
    enumeration plus the flat-face (value-0) correction. `normals` are the
    (already unit-normalized) constraint normals a_k. Returns (l, u) or None if
    the feasible set is empty."""
    gn = np.linalg.norm(g)
    if gn <= TINY:
        return (0.0, 0.0) if region_nonempty(normals, k) else None
    lo, hi, any_feas = np.inf, -np.inf, False
    for q in candidate_directions(g, normals, k):
        if all(np.dot(a, q) >= -FEAS_TOL for a in normals):
            val = float(np.dot(g, q))
            lo = min(lo, val)
            hi = max(hi, val)
            any_feas = True
    if zero_achievable(g, normals, k):
        lo = min(lo, 0.0)
        hi = max(hi, 0.0)
        any_feas = True
    if not any_feas:
        return None
    return lo, hi


def brute_bounds(g, normals, k, n_draws=2_000_000, seed=7):
    """Random-sphere lower bracket of [l, u]: min/max of g'q over sampled unit
    vectors satisfying all a' q >= 0."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n_draws, k))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    if normals:
        n_mat = np.array(normals)  # m x k
        mask = np.all(x @ n_mat.T >= 0.0, axis=1)
        x = x[mask]
    if x.shape[0] == 0:
        return None
    vals = x @ g
    return float(vals.min()), float(vals.max())


def np_quantile_type7(sorted_vals, p):
    return float(np.quantile(sorted_vals, p, method="linear"))


def main():
    data = simulate(SEED, T, A1, A2, P_TRUE, K)
    b, sigma = ols_var(data, LAGS)
    p_chol = np.linalg.cholesky(sigma)
    psis = psi_series(b, LAGS, K, HORIZON)

    # ------------------------------------------------------------------ #
    # Restrictions on the single restricted shock (column j=0):
    #   response of var 0 to the shock at impact  > 0,
    #   response of var 1 to the shock at impact  < 0,
    #   response of var 0 at horizon 2            > 0.
    # Stored as (variable, horizon, sign) with sign in {+1.0, -1.0}.
    # ------------------------------------------------------------------ #
    restr_spec = [(0, 0, 1.0), (1, 0, -1.0), (0, 2, 1.0)]
    normals = []
    for (v, r, sig) in restr_spec:
        a = sig * gradient(psis[r], p_chol, v)
        an = np.linalg.norm(a)
        if an > 1e-12:
            normals.append(a / an)

    cases = []
    for i in range(K):
        for h in [0, 1, 2, 4, HORIZON]:
            g = gradient(psis[h], p_chol, i)
            ab = analytic_bounds(g, normals, K)
            assert ab is not None, f"empty identified set at (i={i}, h={h})"
            l, u = ab
            bb = brute_bounds(g, normals, K)
            assert bb is not None
            lb, ub = bb
            cases.append(
                {
                    "i": i,
                    "h": h,
                    "l": float(l),
                    "u": float(u),
                    "l_brute": float(lb),
                    "u_brute": float(ub),
                }
            )

    # ------------------------------------------------------------------ #
    # GOLDEN C: aggregation over stored per-draw [l, u]. Synthetic but the
    # exact NumPy math the Rust summarizer must reproduce.
    # ------------------------------------------------------------------ #
    rng = np.random.default_rng(4242)
    m = 250
    lowers = np.sort(rng.normal(-1.0, 0.4, m))  # not required sorted; test anyway
    uppers = rng.normal(1.5, 0.5, m)
    lowers = lowers.tolist()
    uppers = uppers.tolist()
    probs = [0.05, 0.16, 0.50, 0.84, 0.95]
    alpha = 0.10
    lo_sorted = np.sort(lowers)
    hi_sorted = np.sort(uppers)
    aggregation = {
        "lowers": [float(x) for x in lowers],
        "uppers": [float(x) for x in uppers],
        "probs": probs,
        "alpha": alpha,
        "set_lower_mean": float(np.mean(lowers)),
        "set_upper_mean": float(np.mean(uppers)),
        "robust_ci_lower": np_quantile_type7(lo_sorted, alpha / 2.0),
        "robust_ci_upper": np_quantile_type7(hi_sorted, 1.0 - alpha / 2.0),
        "lower_quantiles": [np_quantile_type7(lo_sorted, p) for p in probs],
        "upper_quantiles": [np_quantile_type7(hi_sorted, p) for p in probs],
    }

    fixture = {
        "_meta": {
            "description": "Golden fixtures for robust_svar_bounds "
            "(Giacomini-Kitagawa 2021 prior-robust identified-set bounds; "
            "Gafarov-Meier-Montiel-Olea 2018 single-column closed form).",
            "references": {
                "reduced_form": "numpy.linalg.lstsq OLS; "
                "Sigma = U'U/(nobs - (1+k*p)) (df-adjusted)",
                "ma_weights": "Psi_s = J F^s J' from the companion matrix",
                "cholesky": "numpy.linalg.cholesky (lower)",
                "bounds": "independent KKT active-set enumeration over the "
                "sphere-cap; brute force = 2e6 random unit vectors",
                "aggregation": "numpy.mean and numpy.quantile(method='linear', "
                "type-7) over stored per-draw [l, u]",
            },
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "seed": SEED,
        },
        "k": K,
        "lags": LAGS,
        "horizon": HORIZON,
        "data": data.tolist(),        # T x k (for the end-to-end driver test)
        "reg_coefs": b.tolist(),      # (1 + k*p) x k packed OLS coefficients
        "sigma": sigma.tolist(),      # k x k df-adjusted innovation covariance
        "restrictions": restr_spec,   # (variable, horizon, sign) on shock 0
        "cases": cases,               # analytic + brute bounds per (i, h)
        "aggregation": aggregation,
    }

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(fixture, f, indent=1)
    print(f"wrote {OUT}")
    for c in cases:
        print(
            f"  i={c['i']} h={c['h']}: "
            f"analytic [{c['l']:+.6f}, {c['u']:+.6f}]  "
            f"brute [{c['l_brute']:+.6f}, {c['u_brute']:+.6f}]"
        )


if __name__ == "__main__":
    main()
