"""Golden fixtures for var_irf_bands — the ASYMPTOTIC (Lutkepohl 1990
delta-method) IRF standard errors.

VALIDATION STRATEGY
===================
Nothing here imports tsecon. Every stored number is the INDEPENDENT
statsmodels reference: a reduced-form VAR(2) is fitted with
`statsmodels.tsa.api.VAR(...).fit(2, trend="c")` and its
`IRAnalysis.stderr(orth=...)` / `cum_effect_stderr(orth=...)` are
transcribed verbatim. These are the closed-form delta-method standard
errors of the impulse responses (Lutkepohl 2005, ch. 3.7). Reproducing
them in Rust from the fitted (Z'Z)^{-1}, sigma_u, and coefficient
matrices is therefore a genuine cross-implementation check. The data are
DERIVED from a seeded RNG; nothing is a redistributed dataset.

THE DGP
-------
A deliberately stable VAR(2), k = 3, n = 300 (after a 100-row burn-in),
Gaussian innovations with a full covariance. Stability is asserted in
file (all companion eigenvalues strictly inside the unit circle) so the
asymptotics are the standard stationary ones.

WHAT IS STORED (horizon H = 10, trend = "c")
--------------------------------------------
  data               : the n x 3 estimation sample
  point_nonorth      : irf.irfs                       [(H+1) x k x k]
  point_orth         : irf.orth_irfs                  [(H+1) x k x k]
  stderr_nonorth     : irf.stderr(orth=False)         [(H+1) x k x k]
  stderr_orth        : irf.stderr(orth=True)          [(H+1) x k x k]
  cum_stderr_nonorth : irf.cum_effect_stderr(orth=False)
  cum_stderr_orth    : irf.cum_effect_stderr(orth=True)

Layout convention (matches tsecon.var_irf and statsmodels): array
[h][i][j] is the response of variable i to a shock in variable j at
horizon h; the standard-error arrays carry the delta-method SE of that
same cell.
"""

import json
import os

import numpy as np
import statsmodels.tsa.api as tsa

SEED = 20260722
LAGS = 2
HORIZON = 10
N = 300
BURN = 100
K = 3

# Stable VAR(2) coefficient matrices A_1, A_2 (rows = equations).
A1 = np.array(
    [
        [0.5, 0.10, 0.00],
        [0.2, 0.30, 0.10],
        [0.0, 0.20, 0.40],
    ]
)
A2 = np.array(
    [
        [0.10, 0.00, 0.05],
        [0.00, 0.10, 0.00],
        [0.05, 0.00, 0.10],
    ]
)
C = np.array([0.5, 0.2, 0.1])
SIGMA = np.array(
    [
        [1.0, 0.3, 0.2],
        [0.3, 1.0, 0.1],
        [0.2, 0.1, 1.0],
    ]
)


def companion(a1, a2):
    """VAR(2) companion matrix (Lutkepohl comp form)."""
    top = np.hstack([a1, a2])
    bottom = np.hstack([np.eye(K), np.zeros((K, K))])
    return np.vstack([top, bottom])


def simulate():
    """Seeded VAR(2) path; returns the post-burn-in n x k sample."""
    rng = np.random.default_rng(SEED)
    total = N + BURN + LAGS
    shocks = rng.multivariate_normal(np.zeros(K), SIGMA, size=total)
    y = np.zeros((total, K))
    # seed the first two rows at the process mean
    mu = np.linalg.solve(np.eye(K) - A1 - A2, C)
    y[0] = mu
    y[1] = mu
    for t in range(2, total):
        y[t] = C + A1 @ y[t - 1] + A2 @ y[t - 2] + shocks[t]
    return y[BURN + LAGS :]


def main():
    comp = companion(A1, A2)
    eigmod = np.abs(np.linalg.eigvals(comp))
    assert eigmod.max() < 1.0, f"DGP not stable: max |eig| = {eigmod.max()}"

    data = simulate()
    assert data.shape == (N, K)

    res = tsa.VAR(data).fit(LAGS, trend="c")
    irf = res.irf(HORIZON)

    fixture = {
        "_meta": {
            "description": "Asymptotic (Lutkepohl 1990 delta-method) VAR IRF "
            "standard errors — statsmodels IRAnalysis.stderr / "
            "cum_effect_stderr, orth=False and orth=True.",
            "generator": "fixtures/generate_var_irf_bands_fixtures.py",
            "seed": SEED,
            "lags": LAGS,
            "horizon": HORIZON,
            "trend": "c",
            "n": N,
            "k": K,
            "nobs": int(res.nobs),
            "dgp_max_companion_eig": float(eigmod.max()),
            "source": "statsmodels only; no tsecon import (independent golden)",
        },
        "data": data.tolist(),
        "point_nonorth": irf.irfs.tolist(),
        "point_orth": irf.orth_irfs.tolist(),
        "stderr_nonorth": irf.stderr(orth=False).tolist(),
        "stderr_orth": irf.stderr(orth=True).tolist(),
        "cum_stderr_nonorth": irf.cum_effect_stderr(orth=False).tolist(),
        "cum_stderr_orth": irf.cum_effect_stderr(orth=True).tolist(),
    }

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "var_irf_bands.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(fixture, f, indent=2)
    print(f"wrote {out}")
    print(f"nobs={res.nobs}  max|eig|={eigmod.max():.4f}")
    print(f"stderr_nonorth[1] =\n{irf.stderr(orth=False)[1]}")
    print(f"stderr_orth[0] =\n{irf.stderr(orth=True)[0]}")


if __name__ == "__main__":
    main()
