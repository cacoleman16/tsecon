"""Golden fixtures for tsecon-funcshock: functional shocks (Inoue-Rossi 2021).

VALIDATION STRATEGY
===================
Every number this file writes is produced by an INDEPENDENT reference —
numpy.linalg.eigh, statsmodels OLS with HAC covariance, and statsmodels VAR —
never by the tsecon Rust crate, so reproducing these numbers in Rust is a
genuine cross-implementation check, never circular. All data are DERIVED from
seeded numpy DGPs (no redistributed datasets).

THE ESTIMAND
------------
The response of an outcome to a shock that is a whole CURVE — e.g. the entire
yield curve shifting on an announcement day — following Inoue & Rossi (2021,
Quantitative Economics, "The effects of conventional and unconventional
monetary policy: a new approach"): summarize curve shocks by functional
principal components, trace the outcome response to the PC scores, and
reconstruct the response to any user scenario curve from its projection onto
the eigenfunctions.

BLOCKS AND THEIR REFERENCES
---------------------------
1. functional_pca.  REFERENCE: numpy.linalg.eigh on the M x M covariance of
   the demeaned T x M curve panel, POPULATION divisor T (cov = Xc'Xc / T).
   Eigenvalues are reported in DESCENDING order.  SIGN CONVENTION (documented
   identically in the Rust crate so the pin is well-defined): each
   eigenvector's sign is fixed so that its entry of largest absolute value is
   positive; ties broken by the FIRST such index (numpy argmax convention).
   Scores are Xc @ phi (the discrete/Euclidean inner product on the maturity
   grid — no quadrature weights; this matches the discretized implementation
   of Inoue-Rossi).  total_variance is trace(cov); explained shares are
   lambda_k / trace(cov).

2. flp (functional local projection).  REFERENCE: statsmodels
   `OLS(y_{t+h}, [const, S_t (all K scores), y_{t-1..t-p}]).fit(cov_type="HAC",
   cov_kwds={"maxlags": h + p, "use_correction": True})` at each horizon
   h = 0..H — a JOINT regression on all K scores, keeping the joint K x K
   HAC coefficient covariance per horizon.  The horizon-h sample is
   t = p .. T-1-h (nobs = T - h - p).  The Bartlett lag truncation h + p is
   the same default the tsecon-lp crate documents for its HAC path.

3. scenario (functional response).  REFERENCE: plain numpy linear algebra on
   block 2's outputs — weights w = phi' delta (projection of the scenario
   curve onto the eigenfunctions), response_h = w' beta_h, and
   var_h = w' Cov_h w — the closed form of the delta-method variance for a
   fixed (non-estimated) scenario.

4. fvar_scenario.  REFERENCE: statsmodels `VAR([scores, y]).fit(lags,
   trend="c")` (scores ordered FIRST, outcome LAST), orthogonalized MA
   coefficients Theta_h = Psi_h P with P = cholesky(sigma_u) (df-adjusted
   sigma_u, the statsmodels convention).  The scenario response sets the
   reduced-form score innovation equal to w and the OUTCOME's own structural
   shock to zero (recursive/Cholesky identification, scores first):
       z solves  P[:K,:K] z = w      (forward substitution),
       response_h = Theta_h[:, :K] @ z.
   At h = 0 the score responses equal w exactly (Psi_0 = I), and the outcome
   response is the Cholesky regression of the outcome innovation on the score
   innovations evaluated at w — the honest, documented identification caveat.

Run with the project venv:
    .venv/bin/python fixtures/generate_tsecon-funcshock_fixtures.py
"""

import json

import numpy as np
import scipy
import statsmodels
import statsmodels.api as sm
from scipy import linalg as sla
from statsmodels.tsa.api import VAR

OUT = "fixtures/tsecon-funcshock.json"


# --------------------------------------------------------------------------- #
# Curve DGP: Nelson-Siegel-style level/slope/curvature factors + noise.
# --------------------------------------------------------------------------- #
def ar1(rng, n, rho, sd):
    e = rng.normal(scale=sd, size=n)
    x = np.empty(n)
    x[0] = e[0] / np.sqrt(1.0 - rho * rho)
    for t in range(1, n):
        x[t] = rho * x[t - 1] + e[t]
    return x


def ns_loadings(m):
    """Level / slope / curvature loadings on an M-point maturity grid."""
    tau = np.linspace(0.25, 10.0, m)
    lam = 0.7
    level = np.ones(m)
    slope = (1.0 - np.exp(-lam * tau)) / (lam * tau)
    curv = slope - np.exp(-lam * tau)
    return level, slope, curv


def make_curves(rng, t, m):
    level, slope, curv = ns_loadings(m)
    f1 = ar1(rng, t, 0.60, 1.0)
    f2 = ar1(rng, t, 0.40, 0.7)
    f3 = ar1(rng, t, 0.30, 0.4)
    noise = 0.05 * rng.normal(size=(t, m))
    x = np.outer(f1, level) + np.outer(f2, slope) + np.outer(f3, curv) + noise
    return x


# --------------------------------------------------------------------------- #
# Block 1: functional PCA reference (numpy.linalg.eigh).
# --------------------------------------------------------------------------- #
def fpca_reference(x, k):
    t, m = x.shape
    mean = x.mean(axis=0)
    xc = x - mean
    cov = xc.T @ xc / t  # population divisor T
    evals, evecs = np.linalg.eigh(cov)  # ascending
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]
    # Sign convention: largest-|.| entry positive, first index on ties.
    for j in range(m):
        i = int(np.argmax(np.abs(evecs[:, j])))
        if evecs[i, j] < 0.0:
            evecs[:, j] = -evecs[:, j]
    phi = evecs[:, :k]  # M x K
    scores = xc @ phi  # T x K
    total = float(np.trace(cov))
    return {
        "mean_curve": mean.tolist(),
        "eigenvalues": evals.tolist(),  # all M, descending
        "eigenfunctions": phi.T.tolist(),  # K rows, each length M
        "scores": scores.tolist(),  # T x K
        "total_variance": total,
        "explained": (evals[:k] / total).tolist(),
    }


def fpca_case(name, seed, t, m, k):
    rng = np.random.default_rng(seed)
    x = make_curves(rng, t, m)
    ref = fpca_reference(x, k)
    return {"name": name, "t": t, "m": m, "n_factors": k, "curves": x.tolist(), **ref}


# --------------------------------------------------------------------------- #
# Block 2: functional local projection reference (statsmodels OLS-HAC).
# --------------------------------------------------------------------------- #
def flp_reference(y, s, horizons, p):
    t, k = s.shape
    betas, covs, ses, nobs = [], [], [], []
    for h in range(horizons + 1):
        idx = np.arange(p, t - h)
        response = y[idx + h]
        cols = [np.ones(len(idx))]
        cols.extend(s[idx, j] for j in range(k))
        cols.extend(y[idx - lag] for lag in range(1, p + 1))
        design = np.column_stack(cols)
        ml = h + p  # Bartlett lag truncation, tsecon-lp's documented default
        res = sm.OLS(response, design).fit(
            cov_type="HAC", cov_kwds={"maxlags": ml, "use_correction": True}
        )
        b = np.asarray(res.params)[1 : 1 + k]
        cov = np.asarray(res.cov_params())[1 : 1 + k, 1 : 1 + k]
        betas.append(b.tolist())
        covs.append(cov.tolist())
        ses.append(np.sqrt(np.diag(cov)).tolist())
        nobs.append(int(len(idx)))
    return betas, covs, ses, nobs


# --------------------------------------------------------------------------- #
# Main: one coherent pipeline case (fpca -> flp -> scenario -> fvar) plus a
# small full-rank fpca case.
# --------------------------------------------------------------------------- #
def main():
    t, m, k = 240, 12, 3
    seed = 20260721
    rng = np.random.default_rng(seed)
    curves = make_curves(rng, t, m)
    fpca = fpca_reference(curves, k)
    scores = np.array(fpca["scores"])  # T x K
    phi = np.array(fpca["eigenfunctions"]).T  # M x K

    # Outcome responding to the curve through its scores, with persistence.
    y = np.empty(t)
    e = rng.normal(scale=0.3, size=t)
    y[0] = e[0]
    for tt in range(1, t):
        y[tt] = (
            0.2
            + 0.5 * y[tt - 1]
            + 0.8 * scores[tt - 1, 0]
            - 0.4 * scores[tt - 1, 1]
            + e[tt]
        )

    horizons, p = 6, 2
    betas, covs, ses, nobs = flp_reference(y, scores, horizons, p)

    # Scenario: the whole curve flattens (short end up, long end down).
    level, slope, curv = ns_loadings(m)
    delta = 0.25 * level - 0.6 * slope + 0.1 * curv
    w = phi.T @ delta  # K weights
    response = [float(np.dot(w, b)) for b in np.array(betas)]
    se = [float(np.sqrt(w @ np.array(c) @ w)) for c in covs]

    # FVAR scenario: VAR([scores, y]) with scores FIRST, Cholesky.
    lags, var_horizon = 2, 8
    endog = np.column_stack([scores, y])
    res = VAR(endog).fit(lags, trend="c")
    orth = res.orth_ma_rep(var_horizon)  # (H+1) x (K+1) x (K+1)
    chol = np.linalg.cholesky(res.sigma_u)  # df-adjusted sigma_u
    z = sla.solve_triangular(chol[:k, :k], w, lower=True)
    responses = [(orth[h][:, :k] @ z).tolist() for h in range(var_horizon + 1)]
    response_outcome = [r[k] for r in responses]

    fixture = {
        "_meta": {
            "description": "Golden fixtures for tsecon-funcshock (functional shocks, Inoue-Rossi 2021).",
            "references": {
                "functional_pca": "numpy.linalg.eigh of cov = Xc'Xc/T; sign: largest-|.| entry positive (first index on ties)",
                "flp": 'statsmodels OLS(y_{t+h} ~ const + all K scores + p lags of y).fit(cov_type="HAC", maxlags=h+p, use_correction=True)',
                "scenario": "numpy: w = phi' delta, response = w'beta_h, var = w' Cov_h w",
                "fvar": 'statsmodels VAR([scores, y]).fit(lags, trend="c"); Theta_h[:, :K] @ solve(P[:K,:K], w), P = cholesky(sigma_u)',
            },
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "statsmodels": statsmodels.__version__,
            "seed": seed,
        },
        "fpca": [
            fpca_case("main", seed=101, t=240, m=12, k=3),
            fpca_case("full_rank_small", seed=102, t=40, m=5, k=5),
        ],
        "pipeline": {
            "t": t,
            "m": m,
            "n_factors": k,
            "curves": curves.tolist(),
            "y": y.tolist(),
            "flp": {
                "horizons": horizons,
                "n_lag_controls": p,
                "betas": betas,
                "covs": covs,
                "se": ses,
                "nobs": nobs,
            },
            "scenario": {
                "delta": delta.tolist(),
                "weights": w.tolist(),
                "response": response,
                "se": se,
            },
            "fvar": {
                "lags": lags,
                "horizon": var_horizon,
                "responses": responses,  # (H+1) x (K+1): scores first, outcome last
                "response_outcome": response_outcome,
            },
        },
    }

    with open(OUT, "w") as f:
        json.dump(fixture, f, indent=1)
    print(f"wrote {OUT}")
    print(
        f"fpca cases: {len(fixture['fpca'])}; pipeline: T={t} M={m} K={k} "
        f"H={horizons} p={p} var_lags={lags} var_H={var_horizon}"
    )


if __name__ == "__main__":
    main()
