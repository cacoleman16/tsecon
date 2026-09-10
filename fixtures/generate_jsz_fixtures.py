"""Golden fixtures for the JSZ canonical Gaussian affine term-structure model
(roadmap E2 / build-later "JSZ canonical affine term structure").

VALIDATION-FIRST / NON-CIRCULAR: this generator does NOT call the tsecon Rust
crate. It transcribes the DOCUMENTED model, recursions and likelihood of

    Joslin, S., Singleton, K. J., & Zhu, H. (2011). "A New Perspective on
    Gaussian Dynamic Term Structure Models." Review of Financial Studies,
    24(3), 926-970.

directly into NumPy, pins the one step that an independent package computes
(the P-measure VAR(1) of the yield portfolios) to statsmodels `VAR(1)`, and
checks the AFNS special case against the Christensen-Diebold-Rudebusch (2011)
closed form (the same formula `generate_afns_fixtures.py` transcribes). The
honest grade of each block is stated with it.

The model (all conventions the Rust implementation must match)
--------------------------------------------------------------
Latent state X_t (N x 1), risk-neutral (Q) dynamics in the JSZ canonical form

    X_{t+1} = K0^Q + K1^Q X_t + Sigma_X eps_{t+1},   K0^Q = (kinf, 0, .., 0)',
    K1^Q = J(lambda^Q): diagonal with lambda_1 >= ... >= lambda_N, a 1 on the
    superdiagonal wherever two consecutive eigenvalues are EQUAL (a Jordan
    block; the AFNS case (1, rho, rho) is one),   r_t = iota' X_t.

Riccati recursions (per period; A_0 = 0, B_0 = 0):

    A_{n+1} = A_n + K0^Q' B_n + 1/2 B_n' SigmaX B_n,     (SigmaX = Sigma_X Sigma_X')
    B_{n+1} = K1^Q' B_n - iota,
    yield(n) = -(A_n + B_n' X_t) / n,   annualized: times periods_per_year.

A_n is affine in kinf: A_n = kinf * alpha_n + gamma_n with alpha_{n+1} =
alpha_n + B_n[0] and gamma the convexity sum. In the loadings block below the
recursion is evaluated in this literal canonical form; the Rust FIT evaluates
an equivalent "Jordan-chain" representation (ones on the whole superdiagonal;
continuous through coinciding eigenvalues) whose loadings differ by a state
rotation C with C e_1 = e_1 — every quantity stored here is representation-
free (intercepts, portfolio-rotation objects, likelihood values), except the
literal-form loadings `b`, which the Rust `jsz_loadings` also reports.

Rotation to observed portfolios P_t = W y_t (W: N x M, full row rank):

    b_x = -B_n/n (M x N),  D = W b_x,   b_p = b_x D^{-1},
    a_x = -A_n/n * ppy,    c = W a_x,   a_p = a_x - b_p c,
    fitted y_t = a_p + b_p P_t   (so W a_p = 0, W b_p = I: P priced exactly),
    SigmaX (per period) = D^{-1} (Sigma_P / ppy^2) D^{-T}   (Sigma_P annualized).

Likelihood (JSZ factorization). P follows the VAR(1) P_{t+1} = mu + Phi P_t +
u_{t+1}, u ~ N(0, Sigma_P); the M - N yield directions orthogonal to the rows
of W carry iid N(0, sigma_e^2) errors. With (mu, Phi) at their OLS values
(the maximizer for ANY Sigma_P — that is the concentration), U = OLS
residuals (T-1 x N), S = U'U, E = y - a_p - P b_p' (T x M, W E' = 0):

    llk_P = -(T-1) N/2 ln(2 pi) - (T-1)/2 ln det Sigma_P - 1/2 tr(Sigma_P^{-1} S)
    llk_Q = -T (M-N)/2 ln(2 pi sigma_e^2) - sum(E^2) / (2 sigma_e^2)
    llf   = llk_P + llk_Q + (T-1)/2 ln det(W W')

The last term is the Jacobian of y -> (W y, W_perp y) with W_perp an
orthonormal basis of the complement; it makes llf the log-density of the
yield panel (conditional on the first cross-section's portfolios) and hence
invariant to the basis W of the portfolio space (W -> G W changes llk_P by
-(T-1) ln|det G| and the Jacobian by +(T-1) ln|det G|). For an orthonormal W
it is zero and llf is the JSZ replication code's llkP + llkQ term for term.
The generator asserts the invariance numerically before storing anything.

Concentration used by the MLE: given (lambda^Q, Sigma_P), the pricing errors
are affine in kinf, E = U0 - kinf v with U0 = Pi (y - a0), v = Pi a1, Pi = I -
b_p W, so kinf_hat = <mean_t U0_t, v> / <v, v>; and sigma_e^2 = SSE / (T (M-N)).
The numerical search runs over (lambda^Q, chol Sigma_P) only.

Term premium (exactly the acm_term_premium convention): risk-neutral yields
re-run the same recursion in the P rotation with the P-measure (mu, Phi) in
place of the Q dynamics (convexity kept); term_premium = fitted - risk-neutral.

Blocks and grades
-----------------
- "loadings": DOCUMENTED-FORMULA golden of the recursions at stated
  parameters (distinct eigenvalues, the AFNS Jordan block, N = 2, N = 4 with
  an interior tie). Pure arithmetic; pinned at 1e-12.
- "afns": the AFNS special case lambda^Q = (1, e^{-lam dt}, e^{-lam dt}). (i) The
  JSZ yield loadings span EXACTLY the Nelson-Siegel loadings at every period
  length dt (a 3 x 3 rotation C with residual at machine precision is stored).
  (ii) The intercept -A_n/(n dt) at kinf = 0 and Sigma diagonal in the NS
  rotation is the discrete-time convexity term, a left Riemann sum of the
  integral CDR (2011) evaluate in closed form; it converges to the CDR
  closed form -A(tau)/tau at FIRST ORDER in dt (the error falls by ~4 when
  dt is quartered). Both sides are stored at dt = 1/12, 1/48, 1/192 with the
  measured gap; the Rust test reproduces both (the CDR side through the
  crate's own afns_yield_adjustment) and checks the convergence rate.
- "sim": a simulated canonical model (portfolios priced exactly, sigma_e =
  0.2bp elsewhere). Stored: the panel and the truth; the statsmodels VAR(1)
  of P = W y (INDEPENDENT-PACKAGE golden: params, stderr, sigma_u_mle);
  the likelihood at the truth and at a second stated point (documented-
  formula golden); the NumPy/SciPy maximum-likelihood estimate (cross-
  optimizer target: two optimizers on one likelihood must agree to their
  stopping tolerances); and the measured recovery of the truth by the MLE.
- "gsw": the real Gurkaynak-Sack-Wright zero-coupon panel (computed from
  the vendored NSS parameters fixtures/gsw_nss_params.csv, Federal Reserve
  Board data, 1990-01..2007-12 — JSZ's own sample window), maturities 6m,
  1y, 2y, 3y, 5y, 7y, 10y, three principal-component portfolios. Stored:
  the PCA weights, the statsmodels VAR(1) golden, the NumPy MLE (cross-
  optimizer target), and the illustration numbers quoted in the docs.

Run with the project venv:
    .venv/bin/python fixtures/generate_jsz_fixtures.py
"""
import csv
import json
import platform
from pathlib import Path

import numpy as np
import scipy
import statsmodels
from scipy.optimize import minimize
from statsmodels.tsa.api import VAR

OUT = Path(__file__).parent
full = lambda a: [float(x) for x in np.asarray(a, dtype=float).ravel()]
mat2 = lambda a: [[float(x) for x in row] for row in np.asarray(a, dtype=float)]


# ---------------------------------------------------------------------------
# The recursions (literal canonical form) and the rotation.
# ---------------------------------------------------------------------------
def k1_matrix(lam):
    lam = np.asarray(lam, float)
    K1 = np.diag(lam)
    for k in range(1, len(lam)):
        if lam[k] == lam[k - 1]:
            K1[k - 1, k] = 1.0
    return K1


def recursions(lam, kinf, SigX, nmax):
    """A[n], B[n], alpha[n] for n = 0..nmax (per-period log-price coefficients)."""
    N = len(lam)
    K1 = k1_matrix(lam)
    K0 = np.zeros(N)
    K0[0] = kinf
    A = np.zeros(nmax + 1)
    B = np.zeros((nmax + 1, N))
    alpha = np.zeros(nmax + 1)
    for n in range(nmax):
        A[n + 1] = A[n] + K0 @ B[n] + 0.5 * B[n] @ SigX @ B[n]
        alpha[n + 1] = alpha[n] + B[n, 0]
        B[n + 1] = K1.T @ B[n] - 1.0
    return A, B, alpha


def yield_loadings(lam, kinf, SigX, mats, ppy):
    mats = np.asarray(mats, int)
    A, B, _ = recursions(lam, kinf, SigX, int(mats.max()))
    return -A[mats] / mats * ppy, -B[mats] / mats[:, None]


def rotate(lam, kinf, SigP_ann, W, mats, ppy):
    """(a_p, b_p, D, c, SigX) of the documented rotation algebra."""
    mats = np.asarray(mats, int)
    N = len(lam)
    _, bX = yield_loadings(lam, 0.0, np.zeros((N, N)), mats, ppy)
    D = W @ bX
    Dinv = np.linalg.inv(D)
    SigX = Dinv @ (SigP_ann / ppy ** 2) @ Dinv.T
    aX, _ = yield_loadings(lam, kinf, SigX, mats, ppy)
    c = W @ aX
    bP = bX @ Dinv
    aP = aX - bP @ c
    return aP, bP, D, c, SigX


def p_var_ols(P):
    T = P.shape[0]
    Z = np.column_stack([np.ones(T - 1), P[:-1]])
    coef, *_ = np.linalg.lstsq(Z, P[1:], rcond=None)
    U = P[1:] - Z @ coef
    return coef[0], coef[1:].T, U


def loglik(Y, mats, W, lam, kinf, SigP_ann, sigma_e, ppy):
    """The documented llf = llk_P + llk_Q + Jacobian at GIVEN parameters
    (P-VAR at OLS)."""
    T, M = Y.shape
    N = len(lam)
    P = Y @ W.T
    mu, phi, U = p_var_ols(P)
    S = U.T @ U
    sign, logdet = np.linalg.slogdet(SigP_ann)
    assert sign > 0
    llk_p = (-(T - 1) * N / 2 * np.log(2 * np.pi) - (T - 1) / 2 * logdet
             - 0.5 * np.trace(np.linalg.solve(SigP_ann, S)))
    aP, bP, *_ = rotate(lam, kinf, SigP_ann, W, mats, ppy)
    E = Y - aP - P @ bP.T
    assert np.abs(W @ E.T).max() < 1e-12 * max(1.0, np.abs(Y).max())
    llk_q = -T * (M - N) / 2 * np.log(2 * np.pi * sigma_e ** 2) - (E ** 2).sum() / (2 * sigma_e ** 2)
    jac = 0.5 * (T - 1) * np.linalg.slogdet(W @ W.T)[1]
    return llk_p + llk_q + jac, dict(llk_p=llk_p, llk_q=llk_q, jac=jac, mu=mu, phi=phi)


def concentrated(Y, mats, W, lam, SigP_ann, ppy):
    """llf with kinf and sigma_e profiled out (documented closed forms)."""
    T, M = Y.shape
    N = len(lam)
    P = Y @ W.T
    _, _, U = p_var_ols(P)
    S = U.T @ U
    sign, logdet = np.linalg.slogdet(SigP_ann)
    if sign <= 0:
        return -np.inf, None
    llk_p = (-(T - 1) * N / 2 * np.log(2 * np.pi) - (T - 1) / 2 * logdet
             - 0.5 * np.trace(np.linalg.solve(SigP_ann, S)))
    mats = np.asarray(mats, int)
    nmax = int(mats.max())
    _, B, alpha = recursions(lam, 0.0, np.zeros((N, N)), nmax)
    bX = -B[mats] / mats[:, None]
    D = W @ bX
    Dinv = np.linalg.inv(D)
    SigX = Dinv @ (SigP_ann / ppy ** 2) @ Dinv.T
    A0, _, _ = recursions(lam, 0.0, SigX, nmax)
    a0 = -A0[mats] / mats * ppy
    a1 = -alpha[mats] / mats * ppy
    bP = bX @ Dinv
    Pi = np.eye(M) - bP @ W
    Ubar = Pi @ (Y.mean(axis=0) - a0)
    v = Pi @ a1
    kinf = (Ubar @ v) / (v @ v)
    aP = Pi @ a0 + kinf * v
    E = Y - aP - P @ bP.T
    dfq = T * (M - N)
    s2 = (E ** 2).sum() / dfq
    llk_q = -dfq / 2 * (np.log(2 * np.pi * s2) + 1.0)
    jac = 0.5 * (T - 1) * np.linalg.slogdet(W @ W.T)[1]
    return llk_p + llk_q + jac, dict(kinf=kinf, sigma_e=np.sqrt(s2), aP=aP, bP=bP)


# ---------------------------------------------------------------------------
# The NumPy/SciPy maximum-likelihood estimator (cross-optimizer target).
# ---------------------------------------------------------------------------
def pack(lam, SigP, scale):
    N = len(lam)
    th = [lam[0]] + list(np.log(-np.diff(lam)))
    L = np.linalg.cholesky(SigP) / scale
    for i in range(N):
        for j in range(i + 1):
            th.append(np.log(L[i, i]) if i == j else L[i, j])
    return np.array(th)


def unpack(th, N, scale):
    lam = np.empty(N)
    lam[0] = th[0]
    for k in range(1, N):
        lam[k] = lam[k - 1] - np.exp(th[k])
    L = np.zeros((N, N))
    p = N
    for i in range(N):
        for j in range(i + 1):
            L[i, j] = np.exp(th[p]) if i == j else th[p]
            p += 1
    L *= scale
    return lam, L @ L.T


def mle(Y, mats, W, ppy, n_starts=8, seed=0):
    """Seeded multi-start maximum likelihood, the same design as the Rust fit:
    start 0 is JSZ's recommendation (OLS eigenvalues, ordered, gaps >= 1e-3,
    OLS covariance); starts 1.. keep the covariance and redraw the eigenvalue
    pattern (lambda_1 shifted by U(-0.02, 0.03), gaps U(0.01, 0.12)). Every
    start runs a loose BFGS; the best basin is polished by BFGS, Nelder-Mead
    and BFGS again. A cross-optimizer TARGET, not an exact golden: SciPy's
    optimizers stop at their own tolerances. The surface IS multimodal on
    the real GSW panel: from the JSZ start alone (the P-feedback matrix has
    a complex pair, so two starting eigenvalues nearly tie) a single run can
    end ~190 log-likelihood points below the best mode."""
    T, M = Y.shape
    N = W.shape[0]
    P = Y @ W.T
    _, phi, U = p_var_ols(P)
    Sig_ols = U.T @ U / (T - 1)
    lam0 = np.sort(np.real(np.linalg.eigvals(phi)))[::-1]
    for k in range(1, N):
        if lam0[k] > lam0[k - 1] - 1e-3:
            lam0[k] = lam0[k - 1] - 1e-3
    scale = np.sqrt(np.trace(Sig_ols) / N)
    th0 = pack(lam0, Sig_ols, scale)
    f0 = concentrated(Y, mats, W, lam0, Sig_ols, ppy)[0]

    def obj(th):
        lam, SigP = unpack(th, N, scale)
        try:
            val = concentrated(Y, mats, W, lam, SigP, ppy)[0]
        except np.linalg.LinAlgError:
            return np.inf
        return np.inf if not np.isfinite(val) else -(val - f0)

    rng = np.random.default_rng(seed)
    starts = [th0]
    for _ in range(1, n_starts):
        th = th0.copy()
        th[0] = np.clip(th0[0] + (0.05 * rng.uniform() - 0.02), -0.5, 1.02)
        for k in range(1, N):
            th[k] = np.log(0.01 + 0.11 * rng.uniform())
        starts.append(th)
    basins = []
    for th in starts:
        r = minimize(obj, th, method="BFGS", options=dict(maxiter=300, gtol=1e-4))
        if np.isfinite(r.fun):
            basins.append((r.fun, r.x, r.nfev))
    basins.sort(key=lambda b: b[0])
    modes = [float(-b[0] + f0) for b in basins]
    best, x, nfev = basins[0][2] and None, basins[0][1], sum(b[2] for b in basins)
    for method in ("BFGS", "Nelder-Mead", "BFGS"):
        opts = (dict(maxiter=20000, gtol=1e-8) if method == "BFGS"
                else dict(maxiter=30000, maxfev=30000, xatol=1e-9, fatol=1e-10))
        r = minimize(obj, x, method=method, options=opts)
        nfev += int(r.nfev)
        if np.isfinite(r.fun) and (best is None or r.fun < best.fun):
            best = r
        x = best.x
    lam, SigP = unpack(best.x, N, scale)
    llf, extra = concentrated(Y, mats, W, lam, SigP, ppy)
    return dict(lambda_q=lam, sigma=SigP, llf=llf, kinf=extra["kinf"],
                sigma_e=extra["sigma_e"], sigma_ols=Sig_ols, lambda_start=lam0,
                n_fev=int(nfev), basin_llfs=modes)


def pca_w(Y, N):
    """Default portfolio weights: first N right singular vectors of the
    demeaned panel, each row's largest-magnitude entry positive."""
    Yd = Y - Y.mean(axis=0)
    _, _, vt = np.linalg.svd(Yd, full_matrices=False)
    W = vt[:N].copy()
    for row in W:
        if row[np.argmax(np.abs(row))] < 0:
            row *= -1.0
    return W


def statsmodels_var(P):
    res = VAR(P).fit(1, trend="c")
    return dict(params=mat2(res.params), stderr=mat2(res.stderr),
                sigma_u_mle=mat2(res.sigma_u_mle), llf=float(res.llf),
                nobs=int(res.nobs))


# ---------------------------------------------------------------------------
# Block 1: loadings golden.
# ---------------------------------------------------------------------------
def gen_loadings():
    cases = []
    rho = float(np.exp(-0.0609))
    L3 = np.array([[2.0e-4, 0, 0], [0.5e-4, 1.5e-4, 0], [0.2e-4, -0.3e-4, 1.0e-4]])
    specs = [
        ("distinct3", [0.995, 0.95, 0.85], 2.0e-5, L3 @ L3.T,
         [1, 2, 3, 6, 12, 24, 36, 60, 84, 120], 12.0),
        ("afns_jordan", [1.0, rho, rho], 0.0, np.diag([1.0e-8, 4.0e-8, 9.0e-8]),
         [1, 3, 6, 12, 24, 36, 60, 84, 120], 1.0),
        ("two_factor", [0.99, 0.9], 1.0e-5, np.array([[3.0e-8, 1.0e-8], [1.0e-8, 5.0e-8]]),
         [1, 4, 12, 40, 120], 4.0),
        ("tie_middle4", [0.99, 0.9, 0.9, 0.7], 1.0e-5, np.diag([1.0e-8, 2.0e-8, 3.0e-8, 4.0e-8]),
         [1, 2, 6, 12, 24, 60, 120], 12.0),
    ]
    for name, lam, kinf, SigX, mats, ppy in specs:
        a, b = yield_loadings(lam, kinf, SigX, mats, ppy)
        N = len(lam)
        K0 = np.zeros(N)
        K0[0] = kinf
        cases.append(dict(name=name, lambda_q=full(lam), k_inf_q=float(kinf),
                          sigma_x=mat2(SigX), maturities=[int(m) for m in mats],
                          periods_per_year=float(ppy), a=full(a), b=mat2(b),
                          k0_q=full(K0), k1_q=mat2(k1_matrix(lam))))
    return cases


# ---------------------------------------------------------------------------
# Block 2: the AFNS special case vs Nelson-Siegel loadings and the CDR
# closed form.
# ---------------------------------------------------------------------------
def ns_loadings(tau, lam):
    x = lam * tau
    g = (1 - np.exp(-x)) / x
    return np.column_stack([np.ones_like(tau), g, g - np.exp(-x)])


def cdr_adjustment(tau, lam, s):
    """-A(tau)/tau of CDR (2011), independent-factor case — the same closed
    form generate_afns_fixtures.py transcribes."""
    s11, s22, s33 = s
    e1, e2 = np.exp(-lam * tau), np.exp(-2 * lam * tau)
    t11 = tau ** 2 / 6
    t22 = 1 / (2 * lam ** 2) - (1 - e1) / (lam ** 3 * tau) + (1 - e2) / (4 * lam ** 3 * tau)
    t33 = (1 / (2 * lam ** 2) + e1 / lam ** 2 - tau * e2 / (4 * lam) - 3 * e2 / (4 * lam ** 2)
           - 2 * (1 - e1) / (lam ** 3 * tau) + 5 * (1 - e2) / (8 * lam ** 3 * tau))
    return -(s11 ** 2 * t11 + s22 ** 2 * t22 + s33 ** 2 * t33)


def gen_afns():
    lam_annual = 0.5
    sig_annual = np.array([0.006, 0.010, 0.014])
    tau_years = np.array([0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
    ns = ns_loadings(tau_years, lam_annual)
    cdr = cdr_adjustment(tau_years, lam_annual, sig_annual)
    grids = []
    for ppy in (12, 48, 192):
        dt = 1.0 / ppy
        rho = float(np.exp(-lam_annual * dt))
        mats = np.rint(tau_years * ppy).astype(int)
        assert np.allclose(mats, tau_years * ppy)
        lam = [1.0, rho, rho]
        _, b = yield_loadings(lam, 0.0, np.zeros((3, 3)), mats, float(ppy))
        # (i) exact span: b C = NS.
        C, *_ = np.linalg.lstsq(b, ns, rcond=None)
        span_resid = float(np.abs(b @ C - ns).max())
        assert span_resid < 1e-10, span_resid
        # (ii) intercept: X (per-period rate units) = dt * C F, F the NS
        # factors in annual units with per-period innovation cov Sigma_ann dt.
        SigX = dt ** 3 * C @ np.diag(sig_annual ** 2) @ C.T
        a, _ = yield_loadings(lam, 0.0, SigX, mats, float(ppy))
        gap = float(np.abs(a - cdr).max())
        grids.append(dict(periods_per_year=float(ppy), rho=rho, maturities=[int(m) for m in mats],
                          rotation_c=mat2(C), span_residual=span_resid, sigma_x=mat2(SigX),
                          a_jsz=full(a), a_cdr=full(cdr), max_abs_gap=gap))
    ratios = [grids[i + 1]["max_abs_gap"] / grids[i]["max_abs_gap"] for i in range(2)]
    print("  afns: span residuals", [g["span_residual"] for g in grids],
          "gaps", [g["max_abs_gap"] for g in grids], "ratios (expect ~0.25)", ratios)
    assert all(0.2 < r < 0.32 for r in ratios), ratios
    return dict(lambda_annual=lam_annual, sigma_annual=full(sig_annual),
                tau_years=full(tau_years), ns_loadings=mat2(ns), grids=grids,
                gap_ratios=ratios)


# ---------------------------------------------------------------------------
# Block 3: simulated canonical model.
# ---------------------------------------------------------------------------
SIM_MATS = np.array([1, 3, 6, 12, 24, 36, 48, 60, 84, 120])
SIM_T, SIM_PPY, SIM_N = 500, 12.0, 3
SIM_LAM = np.array([0.995, 0.96, 0.85])
SIM_KINF = 2.0e-5
SIM_SIGMA_E = 2.0e-5


def orthonormal_seed_weights(mats):
    raw = np.array([np.ones(len(mats)), mats / 120.0, (mats / 24.0) * np.exp(-mats / 24.0)])
    q, _ = np.linalg.qr(raw.T)
    return q.T


def simulate_panel(seed):
    """Two-pass construction: seed rotation -> exact panel -> PCA weights of
    the exact panel -> re-express the P dynamics in that rotation -> final
    panel with errors orthogonal to the PCA rows."""
    rng = np.random.default_rng(seed)
    mats = SIM_MATS
    W0 = orthonormal_seed_weights(mats.astype(float))
    phi0 = np.array([[0.980, 0.010, -0.010], [0.005, 0.930, 0.020], [0.000, -0.010, 0.850]])
    chol0 = np.array([[0.0060, 0, 0], [0.0010, 0.0030, 0], [-0.0003, 0.0005, 0.0015]])
    Sig0 = chol0 @ chol0.T
    target = 0.03 + 0.03 * (1 - np.exp(-mats / 36.0))          # 3% -> ~6% mean curve
    EP0 = W0 @ target
    mu0 = (np.eye(3) - phi0) @ EP0

    def draw(W, mu, phi, chol, aP, bP, sigma_e):
        N = W.shape[0]
        x = np.linalg.solve(np.eye(N) - phi, mu)
        for _ in range(300):
            x = mu + phi @ x + chol @ rng.standard_normal(N)
        P = np.empty((SIM_T, N))
        for t in range(SIM_T):
            x = mu + phi @ x + chol @ rng.standard_normal(N)
            P[t] = x
        Yex = aP + P @ bP.T
        E = sigma_e * rng.standard_normal(Yex.shape)
        E = E - (E @ W.T) @ W                                   # W E' = 0 exactly (W orthonormal)
        return Yex + E, P, Yex

    aP0, bP0, *_ = rotate(SIM_LAM, SIM_KINF, Sig0, W0, mats, SIM_PPY)
    _, _, Yex0 = draw(W0, mu0, phi0, chol0, aP0, bP0, 0.0)
    W = pca_w(Yex0, SIM_N)
    G = W @ bP0                                                  # P_new = g + G P_old
    g = W @ aP0
    phi = G @ phi0 @ np.linalg.inv(G)
    mu = g + G @ mu0 - phi @ g
    Sig = G @ Sig0 @ G.T
    chol = np.linalg.cholesky(Sig)
    aP, bP, *_ = rotate(SIM_LAM, SIM_KINF, Sig, W, mats, SIM_PPY)
    Y, P, Yex = draw(W, mu, phi, chol, aP, bP, SIM_SIGMA_E)
    assert np.abs(Y @ W.T - P).max() < 1e-12
    return Y, dict(W=W, mu=mu, phi=phi, sigma=Sig, aP=aP, bP=bP)


def gen_sim():
    Y, tr = simulate_panel(20260910)
    mats = SIM_MATS
    W = tr["W"]
    P = Y @ W.T
    print("  sim panel mean (%):", np.round(Y.mean(axis=0) * 100, 2))
    var = statsmodels_var(P)
    # Cross-check the OLS transcription against statsmodels.
    mu, phi, U = p_var_ols(P)
    assert np.abs(np.array(var["params"])[0] - mu).max() < 1e-12
    assert np.abs(np.array(var["params"])[1:].T - phi).max() < 1e-12
    assert np.abs(np.array(var["sigma_u_mle"]) - U.T @ U / (SIM_T - 1)).max() < 1e-14

    llf_truth, parts = loglik(Y, mats, W, SIM_LAM, SIM_KINF, tr["sigma"], SIM_SIGMA_E, SIM_PPY)
    # Invariance of the documented llf to the basis of the portfolio space.
    G = np.array([[2.0, 0.5, -0.3], [0.1, -1.5, 0.2], [0.4, 0.3, 3.0]])
    W2 = G @ W
    llf_rot, _ = loglik(Y, mats, W2, SIM_LAM, SIM_KINF, G @ tr["sigma"] @ G.T, SIM_SIGMA_E, SIM_PPY)
    assert abs(llf_rot - llf_truth) < 1e-7 * abs(llf_truth), (llf_rot, llf_truth)
    # A second, arbitrary stated point.
    lam2, kinf2, se2 = np.array([0.99, 0.95, 0.80]), 1.0e-5, 1.0e-4
    Sig2 = np.array(var["sigma_u_mle"])
    llf_point2, _ = loglik(Y, mats, W, lam2, kinf2, Sig2, se2, SIM_PPY)

    est = mle(Y, mats, W, SIM_PPY)
    rec = dict(lambda_max_abs_err=float(np.abs(est["lambda_q"] - SIM_LAM).max()),
               k_inf_rel_err=float(abs(est["kinf"] - SIM_KINF) / SIM_KINF),
               sigma_e_rel_err=float(abs(est["sigma_e"] - SIM_SIGMA_E) / SIM_SIGMA_E),
               sigma_max_rel_err=float(np.abs(est["sigma"] - tr["sigma"]).max() / np.abs(tr["sigma"]).max()),
               sigma_ols_max_rel_err=float(np.abs(est["sigma_ols"] - tr["sigma"]).max() / np.abs(tr["sigma"]).max()),
               llf_mle_minus_truth=float(est["llf"] - llf_truth))
    print("  sim MLE:", json.dumps({k: float(v) for k, v in rec.items()}), "nfev", est["n_fev"])
    print("  sim basin llfs (loose BFGS per start, best first):", np.round(est["basin_llfs"], 2))
    print("  sim lambda_hat", est["lambda_q"], "start", est["lambda_start"])
    return dict(
        periods_per_year=SIM_PPY, n_factors=SIM_N, maturities=[int(m) for m in mats],
        yields=mat2(Y),
        truth=dict(w=mat2(W), lambda_q=full(SIM_LAM), k_inf_q=SIM_KINF, sigma_e=SIM_SIGMA_E,
                   sigma=mat2(tr["sigma"]), mu_p=full(tr["mu"]), phi_p=mat2(tr["phi"]),
                   a_p=full(tr["aP"]), b_p=mat2(tr["bP"])),
        statsmodels_var=var,
        loglik=dict(at_truth=float(llf_truth), llk_p=float(parts["llk_p"]),
                    llk_q=float(parts["llk_q"]), jacobian=float(parts["jac"]),
                    rotated=dict(g=mat2(G), llf=float(llf_rot)),
                    point2=dict(lambda_q=full(lam2), k_inf_q=kinf2, sigma=mat2(Sig2),
                                sigma_e=se2, llf=float(llf_point2))),
        mle=dict(lambda_q=full(est["lambda_q"]), k_inf_q=float(est["kinf"]),
                 sigma_e=float(est["sigma_e"]), sigma=mat2(est["sigma"]), llf=float(est["llf"]),
                 lambda_start=full(est["lambda_start"]), n_fev=est["n_fev"],
                 basin_llfs=est["basin_llfs"]),
        recovery=rec,
    )


# ---------------------------------------------------------------------------
# Block 4: the real GSW panel, JSZ's 1990-2007 window.
# ---------------------------------------------------------------------------
GSW_MATS = np.array([6, 12, 24, 36, 60, 84, 120])


def read_csv_comments(path):
    with open(path) as f:
        rows = [row for row in csv.reader(f) if not row[0].startswith("#")]
    return rows[0], rows[1:]


def gsw_panel():
    header, data = read_csv_comments(OUT / "gsw_nss_params.csv")
    cols = {name: i for i, name in enumerate(header)}
    data = [row for row in data if "1990-01" <= row[cols["DATE"]][:7] <= "2007-12"]
    dates = [row[cols["DATE"]] for row in data]
    par = {name: np.array([float(row[cols[name]]) for row in data])
           for name in ("BETA0", "BETA1", "BETA2", "BETA3", "TAU1", "TAU2")}
    n_years = GSW_MATS / 12.0
    t1 = n_years[None, :] / par["TAU1"][:, None]
    g1 = (1.0 - np.exp(-t1)) / t1
    tau2 = par["TAU2"][:, None]
    safe_tau2 = np.where(tau2 > 0.0, tau2, 1.0)
    t2 = n_years[None, :] / safe_tau2
    sv = np.where(tau2 > 0.0, (1.0 - np.exp(-t2)) / t2 - np.exp(-t2), 0.0)
    Y = (par["BETA0"][:, None] + par["BETA1"][:, None] * g1
         + par["BETA2"][:, None] * (g1 - np.exp(-t1)) + par["BETA3"][:, None] * sv) / 100.0
    return Y, dates


def gen_gsw():
    Y, dates = gsw_panel()
    T = Y.shape[0]
    W = pca_w(Y, 3)
    P = Y @ W.T
    var = statsmodels_var(P)
    est = mle(Y, GSW_MATS, W, 12.0)
    llf_check, extra = concentrated(Y, GSW_MATS, W, est["lambda_q"], est["sigma"], 12.0)
    fitted = extra["aP"] + P @ extra["bP"].T
    rmse_bp = np.sqrt(((Y - fitted) ** 2).mean(axis=0)) * 1e4
    # Term premium (ACM convention): risk-neutral recursion under the P VAR.
    mu, phi, _ = p_var_ols(P)
    aP, bP, D, c, _ = rotate(est["lambda_q"], est["kinf"], est["sigma"], W, GSW_MATS, 12.0)
    Dinv = np.linalg.inv(D)
    rho1 = Dinv.T @ np.ones(3)
    rho0 = -rho1 @ c / 12.0          # c is annualized; the recursion is per period
    nmax = int(GSW_MATS.max())
    A = np.zeros(nmax + 1)
    B = np.zeros((nmax + 1, 3))
    Sig_pp = est["sigma"] / 144.0
    for n in range(nmax):
        A[n + 1] = A[n] + (mu / 12.0) @ B[n] + 0.5 * B[n] @ Sig_pp @ B[n] - rho0
        B[n + 1] = phi.T @ B[n] - rho1
    rn = -A[GSW_MATS] / GSW_MATS * 12.0 + P @ (-B[GSW_MATS] / GSW_MATS[:, None]).T
    tp = fitted - rn
    j10 = list(GSW_MATS).index(120)
    illus = dict(n_dates=int(T), first=dates[0], last=dates[-1],
                 lambda_q=full(est["lambda_q"]), k_inf_q=float(est["kinf"]),
                 sigma_e_bp=float(est["sigma_e"] * 1e4), rmse_bp=full(rmse_bp),
                 llf=float(est["llf"]),
                 tp10_mean_pp=float(tp[:, j10].mean() * 100), tp10_min_pp=float(tp[:, j10].min() * 100),
                 tp10_max_pp=float(tp[:, j10].max() * 100),
                 tp10_first_pp=float(tp[0, j10] * 100), tp10_last_pp=float(tp[-1, j10] * 100),
                 q_half_life_years=float(np.log(0.5) / np.log(est["lambda_q"][0]) / 12.0),
                 phi_p_eigenvalues=full(np.sort(np.real(np.linalg.eigvals(phi)))[::-1]))
    print("  gsw basin llfs (loose BFGS per start, best first):", np.round(est["basin_llfs"], 2))
    print("  gsw illustration:", json.dumps(illus))
    return dict(periods_per_year=12.0, n_factors=3, maturities=[int(m) for m in GSW_MATS],
                dates=dates, yields=mat2(Y), w=mat2(W), statsmodels_var=var,
                mle=dict(lambda_q=full(est["lambda_q"]), k_inf_q=float(est["kinf"]),
                         sigma_e=float(est["sigma_e"]), sigma=mat2(est["sigma"]),
                         llf=float(est["llf"]), lambda_start=full(est["lambda_start"]),
                         n_fev=est["n_fev"], basin_llfs=est["basin_llfs"]),
                fitted_row0=full(fitted[0]), fitted_row_last=full(fitted[-1]),
                term_premium_120=full(tp[:, j10]), illustration=illus)


def main():
    loadings = gen_loadings()
    afns = gen_afns()
    sim = gen_sim()
    gsw = gen_gsw()
    out = {
        "_meta": {
            "numpy": np.__version__, "scipy": scipy.__version__,
            "statsmodels": statsmodels.__version__, "python": platform.python_version(),
            "reference": "Joslin, Singleton & Zhu (2011), Review of Financial Studies 24(3), "
                         "926-970: the canonical Gaussian DTSM and its concentrated likelihood.",
            "note": "Non-circular (no tsecon call). 'loadings': documented-formula golden of the "
                    "Riccati recursions (1e-12). 'afns': the AFNS special case — the JSZ loadings "
                    "span the Nelson-Siegel loadings exactly (stored rotation), and the discrete "
                    "convexity intercept converges at first order in the period length to the "
                    "CDR (2011) closed form (both sides stored per period length). 'sim': "
                    "simulated canonical model — statsmodels VAR(1) of the portfolios "
                    "(independent-package golden), the documented likelihood at two stated "
                    "points, the SciPy MLE as a cross-optimizer target, measured recovery. "
                    "'gsw': the real GSW panel 1990-2007 (JSZ's window) — PCA weights, "
                    "statsmodels VAR(1) golden, SciPy MLE target, illustration numbers. "
                    "Yields are annualized continuously-compounded DECIMALS; maturities are "
                    "integer periods (months); lambda_q and k_inf_q are per period.",
        },
        "loadings": loadings, "afns": afns, "sim": sim, "gsw": gsw,
    }
    (OUT / "jsz.json").write_text(json.dumps(out, separators=(",", ":")))
    print("wrote jsz.json")


if __name__ == "__main__":
    main()
