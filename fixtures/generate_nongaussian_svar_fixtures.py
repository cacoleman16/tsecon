"""Golden fixtures for nongaussian_svar (tsecon-ident): non-Gaussian /
independent-component structural VAR identification.

METHOD
======
Point-identify the SVAR impact matrix B in u_t = B eps_t from the reduced-form
residuals ALONE -- no sign, zero, long-run, or proxy restriction -- by
exploiting that the structural shocks eps_t are mutually INDEPENDENT and
NON-GAUSSIAN (at most one Gaussian). Second moments fix B only up to an
orthogonal rotation (B and B Q both give Sigma_u = B B'); the higher-order
moments break that rotational indeterminacy up to column SIGN and PERMUTATION
(pure conventions). References: Lanne-Meitz-Saikkonen (2017);
Gourieroux-Monfort-Renne (2017); the FastICA fixed point of Hyvarinen (1999),
Hyvarinen-Oja (2000).

  1. WHITEN.  W = Sigma_u^{-1/2} (symmetric inverse sqrt from the eigen-
     decomposition of Sigma_u).  z_t = W u_t, so Cov(z) = I.
  2. ROTATE.  Symmetric (parallel) FastICA with the log-cosh contrast
     g(u) = tanh(u): find the orthogonal unmixing W_ica maximizing the
     non-Gaussianity of the rows of W_ica z.  Deterministic: identity init, no
     RNG in the fixed point.  Symmetric decorrelation W <- (W W')^{-1/2} W.
  3. UNWHITEN.  Q = W_ica'; B = Sigma_u^{1/2} Q  (so B B' = Sigma_u exactly).
  4. CONVENTIONS.  Order columns by DESCENDING |excess kurtosis| of the
     recovered shocks (most non-Gaussian, hence most strongly identified,
     first; ties -> smaller raw index), then sign each column so its largest-
     magnitude entry is positive.
  Structural IRF Theta_h = Psi_h B (Psi_h the reduced-form MA weights).

VALIDATION STRATEGY
===================
Every number this file writes comes from an INDEPENDENT numpy pipeline --
numpy.linalg.lstsq for the OLS reduced form, numpy.linalg.eigh for the
symmetric inverse sqrt AND the FastICA decorrelation, numpy.tanh for the
contrast -- NEVER from the tsecon Rust crate, so reproducing these numbers in
Rust is a genuine cross-implementation check of the SAME deterministic FastICA
on the SAME whitened residuals.  A secondary sanity check (printed, not stored)
confirms this self-contained FastICA agrees with sklearn.decomposition.FastICA
(whiten=False, identity w_init, fun='logcosh', parallel) -- i.e. the reference
is a faithful FastICA, not a bespoke re-derivation.

This is STATISTICAL identification: it FAILS if the shocks are Gaussian, and
column order/sign are conventions.  The MC target below verifies that the
estimator recovers the TRUE B up to sign+permutation in a large sample (a
statistical property, not exact algebra).

THE DGP
-------
k = 3 VAR(1) with a constant.  u_t = B_TRUE eps_t with three INDEPENDENT,
standardized (unit-variance) non-Gaussian shocks of DISTINCT and well-separated
excess kurtosis, all with stable finite-sample sample-kurtosis (so the ordering
is unambiguous and reproducible), and a deliberate super-/sub-Gaussian mix:
  eps_0 ~ Exponential-1  (skewed; excess kurtosis 6),
  eps_1 ~ Laplace        (excess kurtosis 3),
  eps_2 ~ Uniform        (sub-Gaussian; excess kurtosis -1.2).
T = 10000 after a 500-row burn-in.  Data are DERIVED from a seeded RNG
(numpy.random.default_rng); nothing is a redistributed dataset.  This file
NEVER imports tsecon.

WHAT IS STORED  (lags p = 1, horizon H = 12, trend = "c")
---------------------------------------------------------
  data            : T x 3 estimation sample
  reg_coefs       : (1 + k*p) x k packed OLS coefficients (const row, then A_1)
  sigma           : k x k df-adjusted residual covariance U'U/(nobs - (1+k*p))
                    (== VarResults.sigma_u; whitening + IRF input)
  resid           : nobs x k reduced-form residuals U (whitening input)
  max_iter, tol   : FastICA controls used by BOTH sides
  B               : k x k reference impact matrix (ordered + sign-canonical)
  rotation        : k x k reference whitened rotation Q (ordered + signed)
  shock_kurtosis  : length-k excess kurtosis of the recovered shocks (in order)
  order           : length-k permutation (raw FastICA index at each position)
  converged, n_iter
  structural_irf  : (H+1) x k x k, Theta_h = Psi_h @ B
  mc_b_true       : k x k TRUE B_TRUE (raw; the crate test aligns columns by
                    matching up to sign+permutation, then compares < 0.05)
  mc_kurt_true    : length-k theoretical excess kurtosis [6, 3, 2]

Array [h][i][j] = response of variable i to shock j at horizon h (== tsecon
.var_irf and statsmodels layout).

Run with the project venv:
    .venv/bin/python fixtures/generate_nongaussian_svar_fixtures.py
"""

import json
import os

import numpy as np

SEED = 20260722
OUT = os.path.join(os.path.dirname(__file__), "nongaussian_svar.json")

K = 3
T = 10000
BURN = 500
LAGS = 1
HORIZON = 12
MAX_ITER = 400
TOL = 1e-10

A1 = np.array([[0.5, 0.1, 0.0], [0.0, 0.4, 0.1], [0.1, 0.0, 0.3]], dtype=float)
C = np.array([0.2, -0.1, 0.05], dtype=float)
B_TRUE = np.array(
    [[1.0, 0.5, -0.3], [0.4, 1.0, 0.2], [-0.2, 0.3, 1.0]], dtype=float
)
# Theoretical excess kurtosis of the three standardized shocks.
MC_KURT_TRUE = [6.0, 3.0, -1.2]


def draw_shocks(rng, n):
    """n x 3 independent, standardized (unit-variance) non-Gaussian shocks:
    Exponential-1 (excess kurt 6), Laplace (3), Uniform (-1.2)."""
    expo = rng.standard_exponential(size=n) - 1.0  # mean 0, var 1
    lap = rng.laplace(loc=0.0, scale=1.0 / np.sqrt(2.0), size=n)  # var = 1
    rad = np.sqrt(3.0)
    unif = rng.uniform(-rad, rad, size=n)  # var = 1
    return np.column_stack([expo, lap, unif])


def simulate(rng):
    total = BURN + T
    eps = draw_shocks(rng, total)
    y = np.zeros(K)
    data = np.zeros((T, K))
    for t in range(total):
        u = B_TRUE @ eps[t]
        y = C + A1 @ y + u
        if t >= BURN:
            data[t - BURN] = y
    return data


def ols_var(data, p):
    """Independent OLS reduced form (df-adjusted Sigma matching
    VarResults.sigma_u).  Returns packed B ((1+k*p) x k), residuals U, Sigma."""
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
    return b, u, sigma


def ma_rep(b, p, k, horizon):
    """Reduced-form MA weights Psi_0..Psi_horizon; Psi_0 = I."""
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


def sym_matrix_power(m, power):
    """Symmetric matrix^power via eigh: V diag(lambda^power) V'."""
    w, v = np.linalg.eigh(m)
    if np.any(w <= 0.0):
        raise ValueError("matrix is not positive definite")
    return (v * (w ** power)) @ v.T


def sym_decorrelate(w):
    """(W W')^{-1/2} W."""
    s, u = np.linalg.eigh(w @ w.T)
    return (u * (1.0 / np.sqrt(s))) @ u.T @ w


def fastica_symmetric(z, max_iter, tol):
    """Deterministic symmetric (parallel) FastICA, log-cosh contrast, identity
    init.  z is (T x k) whitened+centered.  Returns (W_ica, converged, n_iter);
    rows of W_ica are the components (sklearn `_ica_par` update, verbatim)."""
    t, k = z.shape
    w = np.eye(k)
    converged = False
    n_iter = 0
    for it in range(max_iter):
        n_iter = it + 1
        y = w @ z.T  # (k x T)
        g = np.tanh(y)  # (k x T)
        gp = (1.0 - g * g).mean(axis=1)  # length k
        w1 = (g @ z) / t - gp[:, np.newaxis] * w
        w1 = sym_decorrelate(w1)
        lim = np.max(np.abs(np.abs(np.diag(w1 @ w.T)) - 1.0))
        w = w1
        if lim < tol:
            converged = True
            break
    return w, converged, n_iter


def excess_kurtosis(x):
    """Population excess kurtosis (divisor N)."""
    xc = x - x.mean()
    m2 = np.mean(xc ** 2)
    m4 = np.mean(xc ** 4)
    return float(m4 / (m2 * m2) - 3.0)


def sklearn_crosscheck(z, w_ica, max_iter, tol):
    """Print-only: confirm the self-contained FastICA matches sklearn's
    FastICA(whiten=False, w_init=I, fun='logcosh', parallel) up to the usual
    sign/permutation of components."""
    try:
        from sklearn.decomposition import FastICA
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"  [sklearn cross-check skipped: {exc}]")
        return
    k = z.shape[1]
    try:
        ica = FastICA(
            n_components=k,
            algorithm="parallel",
            whiten=False,
            fun="logcosh",
            w_init=np.eye(k),
            max_iter=max_iter,
            tol=tol,
        )
        ica.fit(z)
        w_sk = ica.components_  # (k x k) unmixing on the already-whitened z
    except Exception as exc:  # pragma: no cover
        print(f"  [sklearn cross-check skipped: {exc}]")
        return
    # Align sklearn rows to ours up to sign+permutation via best |cosine|.
    used = set()
    max_diff = 0.0
    for i in range(k):
        best_j, best_c, best_s = -1, -1.0, 1.0
        for j in range(k):
            if j in used:
                continue
            c = float(w_ica[i] @ w_sk[j])
            if abs(c) > best_c:
                best_c, best_j, best_s = abs(c), j, np.sign(c) or 1.0
        used.add(best_j)
        max_diff = max(max_diff, float(np.max(np.abs(w_ica[i] - best_s * w_sk[best_j]))))
    print(f"  sklearn FastICA cross-check: max aligned |W_ica - W_sklearn| = {max_diff:.2e}")


def identify(u, sigma, order_by="kurtosis"):
    """The full reference identification on residuals U + covariance Sigma."""
    w_half = sym_matrix_power(sigma, 0.5)
    w_inv = sym_matrix_power(sigma, -0.5)
    z = u @ w_inv  # (nobs x k), z_t = W u_t (W symmetric)
    z = z - z.mean(axis=0, keepdims=True)  # center columns

    w_ica, converged, n_iter = fastica_symmetric(z, MAX_ITER, TOL)

    q_raw = w_ica.T  # k x k
    b_raw = w_half @ q_raw  # k x k
    sources = z @ w_ica.T  # (nobs x k), column j = recovered shock j
    kurt_raw = np.array([excess_kurtosis(sources[:, j]) for j in range(K)])

    if order_by == "kurtosis":
        key = np.abs(kurt_raw)
    elif order_by == "colnorm":
        key = np.linalg.norm(b_raw, axis=0)
    else:
        raise ValueError(order_by)
    # Descending by key, ties -> smaller raw index (stable).
    order = sorted(range(K), key=lambda j: (-key[j], j))

    b = np.zeros((K, K))
    rot = np.zeros((K, K))
    kurt = np.zeros(K)
    for pos, src in enumerate(order):
        istar = int(np.argmax(np.abs(b_raw[:, src])))  # first max on ties
        flip = -1.0 if b_raw[istar, src] < 0.0 else 1.0
        b[:, pos] = flip * b_raw[:, src]
        rot[:, pos] = flip * q_raw[:, src]
        kurt[pos] = kurt_raw[src]

    return {
        "B": b,
        "rotation": rot,
        "shock_kurtosis": kurt,
        "order": order,
        "converged": bool(converged),
        "n_iter": int(n_iter),
        "w_ica": w_ica,
        "z": z,
    }


def main():
    rng = np.random.default_rng(SEED)
    data = simulate(rng)
    b_coefs, u, sigma = ols_var(data, LAGS)

    ref = identify(u, sigma, order_by="kurtosis")
    b = ref["B"]

    # Structural IRF Theta_h = Psi_h @ B.
    psis = ma_rep(b_coefs, LAGS, K, HORIZON)
    structural_irf = [psi @ b for psi in psis]

    # Property sanity: B B' == Sigma (whitening is exact).
    bbt = b @ b.T
    assert np.max(np.abs(bbt - sigma)) < 1e-9, np.max(np.abs(bbt - sigma))

    sklearn_crosscheck(ref["z"], ref["w_ica"], MAX_ITER, TOL)

    fixture = {
        "_meta": {
            "description": "Golden fixtures for nongaussian_svar (non-Gaussian / "
            "independent-component SVAR identification; Lanne-Meitz-Saikkonen 2017, "
            "Gourieroux-Monfort-Renne 2017, FastICA/Hyvarinen).",
            "references": {
                "reduced_form": "numpy.linalg.lstsq OLS; Sigma = U'U/(nobs - (1+k*p)) "
                "(df-adjusted, == VarResults.sigma_u)",
                "whiten": "W = Sigma^{-1/2} via numpy.linalg.eigh; z = U W (centered)",
                "fastica": "symmetric parallel FastICA, log-cosh g(u)=tanh(u), identity "
                "init, symmetric decorrelation (W W')^{-1/2} W via numpy.linalg.eigh; "
                "sklearn FastICA(whiten=False, w_init=I) cross-checked at generation",
                "unwhiten": "B = Sigma^{1/2} (W_ica'); columns ordered by descending "
                "|excess kurtosis| then signed max-abs-positive",
            },
            "numpy": np.__version__,
            "seed": SEED,
        },
        "k": K,
        "lags": LAGS,
        "horizon": HORIZON,
        "trend": "c",
        "max_iter": MAX_ITER,
        "tol": TOL,
        "data": data.tolist(),
        "reg_coefs": b_coefs.tolist(),
        "sigma": sigma.tolist(),
        "resid": u.tolist(),
        "B": b.tolist(),
        "rotation": ref["rotation"].tolist(),
        "shock_kurtosis": ref["shock_kurtosis"].tolist(),
        "order": [int(o) for o in ref["order"]],
        "converged": ref["converged"],
        "n_iter": ref["n_iter"],
        "structural_irf": [m.tolist() for m in structural_irf],
        "mc_b_true": B_TRUE.tolist(),
        "mc_kurt_true": MC_KURT_TRUE,
    }

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(fixture, f, indent=1)
    print(f"wrote {OUT}")
    print(
        f"  converged={ref['converged']} n_iter={ref['n_iter']} order={ref['order']}"
    )
    print(f"  shock_kurtosis (recovered) = {np.round(ref['shock_kurtosis'], 4).tolist()}")
    print(f"  theoretical               = {MC_KURT_TRUE}")
    print(f"  max |B B' - Sigma| = {np.max(np.abs(bbt - sigma)):.2e}")


if __name__ == "__main__":
    main()
