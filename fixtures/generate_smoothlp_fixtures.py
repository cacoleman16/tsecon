"""Golden fixtures for smooth local projections (Barnichon-Brownlees 2019).

VALIDATION STRATEGY
===================
Nothing here imports tsecon: every stored number comes from an INDEPENDENT
path — scipy.interpolate.BSpline for the basis, statsmodels OLS/HAC for the
per-horizon anchors, and plain NumPy normal equations for the stacked
penalized estimator, transcribing the closed form stated below. Reproducing
these numbers in Rust is therefore a genuine cross-implementation check.
All series are DERIVED from a seeded RNG; nothing is a redistributed dataset.

THE ESTIMATOR (Barnichon & Brownlees 2019, REStat, "Impulse Response
Estimation by Smooth Local Projections")
----------------------------------------------------------------------
Per-horizon local projection (the plain Jorda/HAC design, no shock-lag
augmentation):

    y_{t+h} = beta_h * e_t + c_h + sum_{l=1}^{p} phi_{h,l} y_{t-l} + u_{t,h}

Smooth LP restricts the IRF path to a B-spline expansion in the horizon,

    beta_h = sum_{k=1}^{K} theta_k B_k(h),

with the controls left horizon-specific and unpenalized, and estimates all
horizons JOINTLY by penalized least squares over the stacked design

    theta_hat = argmin  ||Y - X theta||^2 + lambda * ||D_r theta_B||^2
              = (X'X + lambda * P)^{-1} X'Y,       P = blkdiag(D_r'D_r, 0),

where a stacked row (h, t) has the K spline columns e_t * B_k(h) followed by
the horizon-h block [1, y_{t-1}, ..., y_{t-p}] (blocks for other horizons
zero), and D_r is the r-th difference matrix on the K basis coefficients.
r = 2 shrinks the IRF toward a straight line in h (Eilers-Marx P-spline).

BASIS (uniform / Eilers-Marx, evaluated by scipy — the independent path)
------------------------------------------------------------------------
Degree-d B-splines with UNIFORM UNCLAMPED knots: n_seg = K - d segments of
width Delta = H / n_seg, knot vector t_j = (j - d) * Delta for
j = 0, ..., K + d, basis evaluated at the integer horizons 0..H via
scipy.interpolate.BSpline.design_matrix. Uniform knots are what make
"theta linear in k  <=>  IRF linear in h" exact, which is the r = 2
shrinkage target. With K = H + 1 the basis interpolates (Schoenberg-Whitney
holds at the integer grid), so lambda = 0 reproduces per-horizon OLS exactly
— that identity is asserted in-file against statsmodels.

STANDARD ERRORS (delta method through the basis; conditional on lambda)
-----------------------------------------------------------------------
The sandwich for the stacked penalized estimator, holding the penalty fixed:

    A = X'X + lambda * P
    g_t = sum_{h : row (h,t) exists} x_{(h,t)} * u_{(h,t)}   (score by base time)
    M = Gamma_0 + sum_{l=1}^{bw} w_l (Gamma_l + Gamma_l'),
        w_l = 1 - l/(bw+1) (Bartlett), Gamma_l = sum_t g_t g_{t-l}',
        bw = H + p (the maxlags = h + p convention at the longest horizon)
    V = A^{-1} M A^{-1},    se(irf_h) = sqrt(B_h' V_thetatheta B_h).

No small-sample correction is applied. These SEs CONDITION on lambda (fixed,
not data-chosen) and describe the sampling variability of the penalized
(shrunk) estimator around its own probability limit — shrinkage bias is not
accounted for. The crate documents the same caveat; the fixture pins the
formula so both implementations agree to ~1e-8.

CROSS-VALIDATION (leave-h-block-out; Burman-Chow-Nolan style)
-------------------------------------------------------------
Base times t = p..n-1 are split into n_folds contiguous blocks
(fold j covers p + floor(j*nb/n_folds) <= t < p + floor((j+1)*nb/n_folds),
nb = n - p). For fold j the TEST rows are all stacked rows (h, t) with t in
the block; the TRAINING rows exclude the block plus a buffer of H + p base
times on each side (the maximal residual/lag overlap between a training row
and a test row). score(lambda) = total squared prediction error over all
test rows across folds / total test rows; the chosen lambda is the grid
minimizer (first index on ties).

WHAT IS STORED
--------------
case_a (n=240, H=8, p=2, degree=3, K=9=H+1, r=2, bw=10):
  y, e, knots, basis B (9x9, scipy), per-horizon statsmodels anchors
  (beta at 1e-10, HAC se maxlags=h+p use_correction=True at 1e-8),
  smooth {theta, irf, se} at lambda in {0, 1, 50, 5000} (~1e-8), and the CV
  block (grid, per-lambda scores at ~1e-8, chosen lambda).
case_b (n=200, H=6, p=1, degree=2, K=5<H+1, r=1, bw=7):
  y, e, knots, basis (7x5), smooth {theta, irf, se} at lambda in {0, 2, 200}.

Run with the project venv:
    .venv/bin/python fixtures/generate_smoothlp_fixtures.py
"""

import json

import numpy as np
import scipy
import statsmodels
import statsmodels.api as sm
from scipy.interpolate import BSpline


def make_dgp(seed, n, burn, noise_sd):
    """y_t = sum_j psi_j e_{t-j} + noise, psi a smooth hump (truncated at 30)."""
    rng = np.random.default_rng(seed)
    jmax = 30
    psi = np.array([(1.0 + 0.8 * j) * np.exp(-0.35 * j) for j in range(jmax + 1)])
    e = rng.standard_normal(n + burn)
    w = rng.standard_normal(n + burn)
    y = np.convolve(e, psi)[: n + burn] + noise_sd * w
    return y[burn:].copy(), e[burn:].copy(), psi


def uniform_knots(hmax, degree, n_basis):
    n_seg = n_basis - degree
    delta = hmax / n_seg
    return np.array([(j - degree) * delta for j in range(n_basis + degree + 1)])


def basis_matrix(hmax, degree, n_basis):
    knots = uniform_knots(hmax, degree, n_basis)
    hgrid = np.arange(hmax + 1, dtype=float)
    b = BSpline.design_matrix(hgrid, knots, degree).toarray()
    assert b.shape == (hmax + 1, n_basis)
    assert np.allclose(b.sum(axis=1), 1.0, atol=1e-12), "partition of unity"
    return knots, b


def stacked(y, e, b, hmax, p, n_basis, base_mask=None):
    """Stacked rows (h, t); columns [K spline cols | per-horizon control blocks]."""
    n = len(y)
    q = n_basis + (hmax + 1) * (1 + p)
    rows_x, rows_y, rows_t = [], [], []
    for h in range(hmax + 1):
        for t in range(p, n - h):
            if base_mask is not None and not base_mask[t]:
                continue
            x = np.zeros(q)
            x[:n_basis] = e[t] * b[h]
            off = n_basis + h * (1 + p)
            x[off] = 1.0
            for lag in range(1, p + 1):
                x[off + lag] = y[t - lag]
            rows_x.append(x)
            rows_y.append(y[t + h])
            rows_t.append(t)
    return np.array(rows_x), np.array(rows_y), np.array(rows_t)


def penalty(n_basis, q, r):
    d = np.diff(np.eye(n_basis), n=r, axis=0)
    pfull = np.zeros((q, q))
    pfull[:n_basis, :n_basis] = d.T @ d
    return pfull


def solve_theta(xmat, yvec, pfull, lam):
    a = xmat.T @ xmat + lam * pfull
    return np.linalg.solve(a, xmat.T @ yvec), a


def sandwich_se(xmat, yvec, tvec, a, theta, b, n, p, n_basis, bw):
    """Bartlett-HAC sandwich over base-time-aggregated scores (see docstring)."""
    u = yvec - xmat @ theta
    g = np.zeros((n, xmat.shape[1]))
    np.add.at(g, tvec, xmat * u[:, None])
    g = g[p:]
    m = g.T @ g
    for lag in range(1, bw + 1):
        w = 1.0 - lag / (bw + 1.0)
        gam = g[lag:].T @ g[:-lag]
        m += w * (gam + gam.T)
    a_inv = np.linalg.inv(a)
    v = a_inv @ m @ a_inv
    vb = v[:n_basis, :n_basis]
    return np.sqrt(np.einsum("hk,kl,hl->h", b, vb, b))


def cv_scores(y, e, b, hmax, p, n_basis, pfull, grid, n_folds, buffer):
    n = len(y)
    nb = n - p
    sse = np.zeros(len(grid))
    n_test = 0
    for j in range(n_folds):
        lo = p + (j * nb) // n_folds
        hi = p + ((j + 1) * nb) // n_folds
        train = np.zeros(n, dtype=bool)
        train[p:n] = True
        train[max(0, lo - buffer) : min(n, hi + buffer)] = False
        test = np.zeros(n, dtype=bool)
        test[lo:hi] = True
        xtr, ytr, _ = stacked(y, e, b, hmax, p, n_basis, train)
        xte, yte, _ = stacked(y, e, b, hmax, p, n_basis, test)
        n_test += len(yte)
        for gi, lam in enumerate(grid):
            th, _ = solve_theta(xtr, ytr, pfull, lam)
            sse[gi] += float(((yte - xte @ th) ** 2).sum())
    return sse / n_test


def per_horizon_statsmodels(y, e, hmax, p):
    """Plain per-horizon LP with statsmodels HAC (maxlags=h+p, correction)."""
    n = len(y)
    out = []
    for h in range(hmax + 1):
        t = np.arange(p, n - h)
        cols = [e[t], np.ones(len(t))] + [y[t - lag] for lag in range(1, p + 1)]
        xh = np.column_stack(cols)
        res = sm.OLS(y[t + h], xh).fit(
            cov_type="HAC", cov_kwds={"maxlags": h + p, "use_correction": True}
        )
        out.append(
            {
                "beta": float(res.params[0]),
                "se_hac": float(res.bse[0]),
                "maxlags": h + p,
                "nobs": int(len(t)),
            }
        )
    return out


def build_case(seed, n, noise_sd, hmax, p, degree, n_basis, r, lambdas, cv=None):
    y, e, _psi = make_dgp(seed, n, burn=60, noise_sd=noise_sd)
    knots, b = basis_matrix(hmax, degree, n_basis)
    q = n_basis + (hmax + 1) * (1 + p)
    xmat, yvec, tvec = stacked(y, e, b, hmax, p, n_basis)
    pfull = penalty(n_basis, q, r)
    bw = hmax + p

    smooth = []
    for lam in lambdas:
        theta, a = solve_theta(xmat, yvec, pfull, lam)
        irf = b @ theta[:n_basis]
        se = sandwich_se(xmat, yvec, tvec, a, theta, b, n, p, n_basis, bw)
        smooth.append(
            {
                "lambda": lam,
                "theta": theta[:n_basis].tolist(),
                "irf": irf.tolist(),
                "se": se.tolist(),
            }
        )

    case = {
        "n": n,
        "horizons": hmax,
        "n_lag_controls": p,
        "degree": degree,
        "n_basis": n_basis,
        "penalty_order": r,
        "hac_bandwidth": bw,
        "y": y.tolist(),
        "e": e.tolist(),
        "knots": knots.tolist(),
        "basis": b.tolist(),
        "smooth": smooth,
    }

    if n_basis == hmax + 1:
        # Interpolating basis: lambda = 0 must equal per-horizon OLS exactly.
        perh = per_horizon_statsmodels(y, e, hmax, p)
        lam0 = next(s for s in smooth if s["lambda"] == 0.0)
        gap = max(abs(a_ - b_["beta"]) for a_, b_ in zip(lam0["irf"], perh))
        assert gap < 1e-10, f"lambda=0 vs per-horizon OLS: max gap {gap}"
        case["perh"] = perh

    if cv is not None:
        grid, n_folds = cv
        buffer = hmax + p
        scores = cv_scores(y, e, b, hmax, p, n_basis, pfull, grid, n_folds, buffer)
        case["cv"] = {
            "grid": list(grid),
            "n_folds": n_folds,
            "buffer": buffer,
            "scores": scores.tolist(),
            "lambda_chosen": grid[int(np.argmin(scores))],
        }
    return case


def main():
    fixture = {
        "_meta": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "statsmodels": statsmodels.__version__,
            "design_note": (
                "Smooth local projections (Barnichon-Brownlees 2019): "
                "theta_hat = (X'X + lambda*P)^{-1} X'y over the stacked "
                "per-horizon design with uniform Eilers-Marx B-spline basis "
                "(scipy BSpline.design_matrix), P = blkdiag(D_r'D_r, 0). "
                "SEs: Bartlett-HAC sandwich over base-time-aggregated scores, "
                "bw = H + p, no small-sample correction, conditional on "
                "lambda. CV: leave-h-block-out, contiguous folds, buffer "
                "H + p. Per-horizon anchors from statsmodels OLS/HAC "
                "(maxlags = h + p, use_correction=True). See module "
                "docstring for the full formulas."
            ),
        },
        "case_a": build_case(
            seed=20260721,
            n=240,
            noise_sd=0.4,
            hmax=8,
            p=2,
            degree=3,
            n_basis=9,
            r=2,
            lambdas=[0.0, 1.0, 50.0, 5000.0],
            cv=([0.5, 5.0, 50.0, 500.0, 5000.0], 4),
        ),
        "case_b": build_case(
            seed=907,
            n=200,
            noise_sd=0.8,
            hmax=6,
            p=1,
            degree=2,
            n_basis=5,
            r=1,
            lambdas=[0.0, 2.0, 200.0],
        ),
    }
    out = "fixtures/smoothlp.json"
    with open(out, "w") as f:
        json.dump(fixture, f)
    print(f"wrote {out}")
    print("case_a cv scores:", fixture["case_a"]["cv"]["scores"])
    print("case_a lambda chosen:", fixture["case_a"]["cv"]["lambda_chosen"])


if __name__ == "__main__":
    main()
