"""Golden fixtures for the SSVS-BVAR (George, Sun & Ni 2008, J. Econometrics
142, "Bayesian stochastic search for VAR model restrictions").

There is no bit-exact external reference for the full stochastic-search
posterior that runs in this venv (R BVAR/bvars are not runnable and use
different priors anyway), so validation is a LAYERED stack. This generator
(NumPy only — it NEVER imports tsecon) writes:

  * `mc_recovery` — a stable sparse VAR(2) DGP: the lag matrices A_l, a fixed
    Sigma (via its lower Cholesky), and the true-nonzero / true-zero index
    masks in the crate's regressor-by-equation coefficient layout. The Rust
    test SIMULATES the data itself from a `tsecon_rng::Stream` (matching the
    crate's property-test style), runs `bvar_ssvs`, and asserts inclusion
    probabilities near 1 on the true-nonzeros and near 0 on the true-zeros.

  * `anchor_block1` — the deterministic block-1 Gaussian/SUR conditional:
    a small data matrix, an injected Sigma, and a single prior scale `tau`
    (the c0 == c1 collapse to one Gaussian), with the NumPy golden precision
    P = kron(inv(Sigma), X'X) + diag(1/tau^2) and mean
    alpha_bar = solve(P, vec(X'Y inv(Sigma))) in column-major (vec) order.

  * `anchor_bernoulli` — the block-2/4 mixture-odds inclusion probability at
    several points, computed from the raw ratio of normal pdfs.

  * `anchor_block3` — the deterministic block-3 precision-factor conditional
    moments (Gamma shape/rate, the eta covariance M_j, and the unit eta
    mean) for a chosen column, from a stored residual cross-product S.

The design layout (must match the Rust `build_xy`): X = [1, y_{t-1}, ...,
y_{t-p}], regressor index for lag l, variable v is 1 + (l-1) n + v; vec()
stacks A (k x n) column-major, index i = r + c k.

Run with the project venv:
    .venv/bin/python fixtures/generate_ssvs_fixtures.py
"""
import json
import platform
from pathlib import Path

import numpy as np

OUT = Path(__file__).parent
META = {
    "numpy": np.__version__,
    "python": platform.python_version(),
    "authored": "independent NumPy closed-form conditional moments + sparse-VAR "
    "DGP metadata for SSVS-BVAR (George-Sun-Ni 2008); data simulated in-Rust "
    "from a tsecon_rng::Stream, so no cross-language RNG contract is needed",
}


def build_design(data, p):
    """X = [1, y_{t-1}, ..., y_{t-p}], matching the Rust build_xy layout."""
    n = data.shape[1]
    T = data.shape[0] - p
    Y = data[p:]
    cols = [np.ones(T)]
    for l in range(1, p + 1):
        for v in range(n):
            cols.append(data[p - l : data.shape[0] - l, v])
    X = np.column_stack(cols)
    return Y, X


def companion_spectral_radius(A_lags, n, p):
    """Largest eigenvalue modulus of the VAR companion form."""
    comp = np.zeros((n * p, n * p))
    comp[:n, :] = np.hstack(A_lags)
    if p > 1:
        comp[n:, : n * (p - 1)] = np.eye(n * (p - 1))
    return float(np.max(np.abs(np.linalg.eigvals(comp))))


def mc_recovery():
    """A stable, sparse VAR(2) with n = 3: own lags plus two cross terms in
    lag 1 and one own second lag; everything else exactly zero."""
    n, p = 3, 2
    A1 = np.array(
        [
            [0.5, 0.3, 0.0],
            [0.0, 0.4, 0.0],
            [0.4, 0.0, 0.5],
        ]
    )
    A2 = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.3, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    A_lags = [A1, A2]
    rho = companion_spectral_radius(A_lags, n, p)
    assert rho < 0.95, f"DGP not comfortably stationary: rho = {rho}"

    Sigma = np.array(
        [
            [1.00, 0.20, 0.10],
            [0.20, 1.00, 0.15],
            [0.10, 0.15, 1.00],
        ]
    )
    L = np.linalg.cholesky(Sigma)  # lower, L L' = Sigma

    # True coefficient matrix in regressor-by-equation layout (k x n):
    # row r = 1 + (l-1) n + v carries A_l[eq, v] in column eq.
    k = 1 + n * p
    true_coef = np.zeros((k, n))
    for l in range(1, p + 1):
        A_l = A_lags[l - 1]
        for v in range(n):
            r = 1 + (l - 1) * n + v
            for eq in range(n):
                true_coef[r, eq] = A_l[eq, v]
    mask = (true_coef != 0.0).astype(int)  # searchable-nonzero mask (row 0 = 0)

    return {
        "n": n,
        "p": p,
        "T": 600,
        "burn": 100,
        "sim_seed": 20260722,
        "A_lags": [A.tolist() for A in A_lags],
        "sigma_chol": L.tolist(),
        "spectral_radius": rho,
        "true_coef": true_coef.tolist(),
        "true_nonzero_mask": mask.tolist(),
    }


def anchor_block1():
    """Deterministic block-1 conditional on a small design and injected Sigma,
    single prior scale tau (the c0 == c1 collapse)."""
    rng = np.random.default_rng(4242)
    n, p = 2, 1
    T_raw = 32
    data = np.cumsum(rng.standard_normal((T_raw, n)) * 0.5, axis=0)
    Y, X = build_design(data, p)
    k = X.shape[1]  # 1 + n p = 3
    m = n * k

    Sigma = np.array([[1.3, 0.4], [0.4, 0.9]])
    Sinv = np.linalg.inv(Sigma)
    tau = 2.5  # single Gaussian prior sd for every coefficient

    XtX = X.T @ X
    XtY = X.T @ Y
    # vec() is column-major (order='F'); i = r + c*k.
    b = (XtY @ Sinv).flatten(order="F")
    P = np.kron(Sinv, XtX) + np.eye(m) * (1.0 / tau**2)
    alpha_bar = np.linalg.solve(P, b)

    return {
        "n": n,
        "p": p,
        "data": data.tolist(),
        "sigma": Sigma.tolist(),
        "tau": tau,
        "P": P.tolist(),
        "alpha_bar": alpha_bar.tolist(),
    }


def anchor_bernoulli():
    """Mixture-odds inclusion probability from the raw ratio of normal pdfs."""

    def npdf(x, var):
        return np.exp(-0.5 * x * x / var) / np.sqrt(2.0 * np.pi * var)

    cases = []
    for x, v_slab, v_spike, prior in [
        (0.05, 1.0, 0.01, 0.5),
        (0.8, 1.0, 0.01, 0.5),
        (0.3, 4.0, 0.04, 0.3),
        (-1.2, 9.0, 0.09, 0.5),
        (0.0, 2.0, 0.02, 0.5),
    ]:
        p1 = prior * npdf(x, v_slab)
        p0 = (1.0 - prior) * npdf(x, v_spike)
        prob = p1 / (p1 + p0)
        cases.append(
            {
                "x": x,
                "v_slab": v_slab,
                "v_spike": v_spike,
                "prior": prior,
                "prob": float(prob),
            }
        )
    return cases


def anchor_block3():
    """Deterministic block-3 precision-factor column moments from a stored
    residual cross-product S."""
    S = np.array(
        [
            [10.0, 2.0, 1.0],
            [2.0, 8.0, 1.5],
            [1.0, 1.5, 6.0],
        ]
    )
    gamma_a, gamma_b, T = 0.01, 0.01, 200
    kappa1 = 10.0
    shape = gamma_a + 0.5 * T

    # Column j = 2 (0-indexed): two preceding etas, diffuse (slab) prior.
    j = 2
    s_prev = S[:j, j]  # [S_02, S_12]
    S_prev = S[:j, :j]
    d_j_inv = np.array([1.0 / kappa1**2, 1.0 / kappa1**2])
    M_j = np.linalg.inv(S_prev + np.diag(d_j_inv))
    quad = s_prev @ M_j @ s_prev
    rate_j = gamma_b + 0.5 * (S[j, j] - quad)
    eta_mean_unit = (-M_j @ s_prev).tolist()

    # Column j = 0: no etas.
    rate_0 = gamma_b + 0.5 * S[0, 0]

    return {
        "S": S.tolist(),
        "gamma_a": gamma_a,
        "gamma_b": gamma_b,
        "T": T,
        "kappa1": kappa1,
        "col_j": j,
        "d_j_inv": d_j_inv.tolist(),
        "shape": shape,
        "rate_j": float(rate_j),
        "M_j": M_j.tolist(),
        "eta_mean_unit": eta_mean_unit,
        "rate_0": float(rate_0),
    }


def gen():
    dump = {
        "_meta": META,
        "mc_recovery": mc_recovery(),
        "anchor_block1": anchor_block1(),
        "anchor_bernoulli": anchor_bernoulli(),
        "anchor_block3": anchor_block3(),
    }
    path = OUT / "ssvs.json"
    path.write_text(json.dumps(dump, indent=1), encoding="utf-8")
    print(f"wrote {path} ({path.stat().st_size} bytes)")
    mc = dump["mc_recovery"]
    print(f"  DGP spectral radius = {mc['spectral_radius']:.4f}")
    nz = int(np.sum(mc["true_nonzero_mask"]))
    print(f"  searchable coefs: {(len(mc['true_coef']) - 1) * mc['n']}, true-nonzero: {nz}")


if __name__ == "__main__":
    gen()
