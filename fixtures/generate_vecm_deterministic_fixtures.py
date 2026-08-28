"""Golden fixture for the VECM deterministic-terms cases (field item 12).

Seeded *drifting* cointegrated log-level data — three series driven by two
independent stochastic trends with drift and very different level offsets, so
exactly one cointegrating relation exists and the no-deterministic fit visibly
rotates `beta` away from the constant-adjusted cointegrating space (the
reporter's scenario: the betas of `vecm` (deterministic = "n") and `johansen`
(det_order = 0, an unrestricted constant) end up a cosine of ~0.63 apart on
this draw; the reporter measured ~0.57 on theirs).

Pins statsmodels `VECM(k_ar_diff=1, coint_rank=1, deterministic=...)` for both
supported cases ("n" and "co": alpha, beta, gamma, det_coef, sigma_u, llf) and
`coint_johansen(det_order=0, k_ar_diff=1)` (eigenvalues and eigenvectors) on
the same data, arbitrating that `deterministic="co"` reproduces the Johansen
cointegrating space exactly while `"n"` does not.

Run with the project venv: .venv/bin/python fixtures/generate_vecm_deterministic_fixtures.py
"""
import json
import platform
from pathlib import Path

import numpy as np

OUT = Path(__file__).parent
full = lambda a: [float(x) for x in np.asarray(a).ravel()]


def gen():
    import statsmodels
    from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen

    # Drifting cointegrated log levels: two independent stochastic trends
    # with drift; only y3 - 0.5*y1 - 0.5*y2 (plus a constant) is stationary,
    # so the true cointegrating rank is 1. The large level offsets (0.1 vs
    # 12.0 vs 0.5) are what force the deterministic="n" fit to twist beta.
    n = 400
    rng = np.random.default_rng(1)
    t1 = np.cumsum(0.002 + 0.02 * rng.standard_normal(n))
    t2 = np.cumsum(0.001 + 0.02 * rng.standard_normal(n))
    y1 = 0.1 + t1 + 0.03 * rng.standard_normal(n)
    y2 = 12.0 + t2 + 0.03 * rng.standard_normal(n)
    y3 = 0.5 + 0.5 * t1 + 0.5 * t2 + 0.03 * rng.standard_normal(n)
    data = np.column_stack([y1, y2, y3])

    fit_n = VECM(data, k_ar_diff=1, coint_rank=1, deterministic="n").fit()
    fit_co = VECM(data, k_ar_diff=1, coint_rank=1, deterministic="co").fit()
    joh = coint_johansen(data, det_order=0, k_ar_diff=1)

    def vecm_block(fit, with_det):
        block = {
            "alpha": [full(row) for row in fit.alpha],
            "beta": [full(row) for row in fit.beta],
            "gamma": [full(row) for row in fit.gamma],
            "sigma_u": [full(row) for row in fit.sigma_u],
            "llf": float(fit.llf),
        }
        if with_det:
            block["det_coef"] = [full(row) for row in fit.det_coef]
        return block

    def cosine(a, b):
        a = np.asarray(a, float).ravel()
        b = np.asarray(b, float).ravel()
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

    out = {
        "_meta": {
            "statsmodels": statsmodels.__version__,
            "numpy": np.__version__,
            "python": platform.python_version(),
            "note": "VECM(k_ar_diff=1, coint_rank=1, deterministic='n'|'co') and "
                    "coint_johansen(det_order=0, k_ar_diff=1) on seeded drifting "
                    "cointegrated data (rank 1). deterministic='co' spans the same "
                    "cointegrating space as coint_johansen det_order=0; "
                    "deterministic='n' does not (beta cosine pinned below).",
        },
        # Series-major, like coint.json: k lists of length T.
        "data": [full(data[:, j]) for j in range(3)],
        "k_ar_diff": 1,
        "coint_rank": 1,
        "vecm_n": vecm_block(fit_n, with_det=False),
        "vecm_co": vecm_block(fit_co, with_det=True),
        "johansen": {
            "det_order": 0,
            "eig": full(joh.eig),
            "evec": [full(row) for row in joh.evec],
        },
        # The reporter's divergence, pinned as documented behavior: the angle
        # between the "n" and "co" cointegrating vectors on this draw.
        "beta_cosine_n_co": cosine(fit_n.beta, fit_co.beta),
    }
    (OUT / "vecm_deterministic.json").write_text(json.dumps(out, separators=(",", ":")))
    print("wrote vecm_deterministic.json  beta_cosine_n_co =", out["beta_cosine_n_co"])


if __name__ == "__main__":
    gen()
