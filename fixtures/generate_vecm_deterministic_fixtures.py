"""Golden fixture for the VECM deterministic-terms cases (field item 12 + the
restricted-cases follow-up).

Three seeded datasets:

1. *Drifting* cointegrated log-level data — three series driven by two
   independent stochastic trends with drift and very different level offsets,
   so exactly one cointegrating relation exists and the no-deterministic fit
   visibly rotates `beta` away from the constant-adjusted cointegrating space
   (the reporter's scenario: the betas of `vecm` (deterministic = "n") and
   `johansen` (det_order = 0, an unrestricted constant) end up a cosine of
   ~0.63 apart on this draw; the reporter measured ~0.57 on theirs). Pins
   statsmodels `VECM(k_ar_diff=1, coint_rank=1, deterministic=...)` for "n"
   and "co" (alpha, beta, gamma, det_coef, sigma_u, llf) and
   `coint_johansen(det_order=0, k_ar_diff=1)` (eigenvalues and eigenvectors)
   on the same data, arbitrating that `deterministic="co"` reproduces the
   Johansen cointegrating space exactly while `"n"` does not.

2. *Trending* cointegrated data (`trending` block) — two drifting stochastic
   trends plus a third series whose equilibrium relation to them is
   stationary around a constant **plus a linear trend**, so the restricted
   cases genuinely differ from the unrestricted ones and from each other.
   Pins every statsmodels deterministic case ("n", "co", "ci", "lo", "li",
   "colo", "coli", "cilo", "cili") at k_ar_diff=2: alpha, beta,
   det_coef_coint (the widened-beta rows), gamma, det_coef, sigma_u, llf.
   Also pins `coint_johansen(det_order=1)` on the same data and the measured
   cosine between its first eigenvector and the "colo" beta (~1 but NOT
   exact — coint_johansen's det_order=1 detrends the levels over the full
   sample before partialling, a different finite-sample projection than the
   VECM's joint one; the det_order=0 <-> "co" correspondence in dataset 1 is
   the exact one), plus the cross-case beta cosines documenting that the
   case choice changes the answer on this draw.

3. *Seasonal* cointegrated quarterly data (`seasonal` block) — a cointegrated
   pair with a deterministic quarterly pattern. Pins
   `VECM(..., deterministic="co", seasons=4, first_season=2)` and
   `VECM(..., deterministic="ci", seasons=4)` (alpha, beta, det_coef_coint,
   gamma, det_coef — the centered-seasonal-dummy columns included — sigma_u,
   llf), exercising both the seasons machinery and a nonzero first_season.

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

    def full_block(fit):
        """Every estimate the restricted cases add, statsmodels' own split:
        beta (variable rows) + det_coef_coint (deterministic rows of the
        widened cointegrating matrix), det_coef (short-run deterministics)."""
        return {
            "alpha": [full(row) for row in fit.alpha],
            "beta": [full(row) for row in fit.beta],
            "det_coef_coint": [full(row) for row in fit.det_coef_coint],
            "gamma": [full(row) for row in fit.gamma],
            "det_coef": [full(row) for row in fit.det_coef],
            "sigma_u": [full(row) for row in fit.sigma_u],
            "llf": float(fit.llf),
        }

    # ---- dataset 2: trending cointegrated data, every deterministic case.
    # Two drifting stochastic trends; y3's equilibrium relation to y1/y2 is
    # stationary AR(1) noise around a constant + linear trend, so the
    # restricted trend ("li"-type) is the correctly specified term and the
    # deterministic case visibly moves beta.
    n2 = 420
    rng2 = np.random.default_rng(7)
    tt = np.arange(n2)
    w1 = np.cumsum(0.010 + 0.05 * rng2.standard_normal(n2))
    w2 = np.cumsum(-0.004 + 0.05 * rng2.standard_normal(n2))
    e = np.zeros(n2)
    for i in range(1, n2):
        e[i] = 0.5 * e[i - 1] + 0.08 * rng2.standard_normal()
    y1t = 1.0 + w1 + 0.05 * rng2.standard_normal(n2)
    y2t = -2.0 + w2 + 0.05 * rng2.standard_normal(n2)
    y3t = 0.5 + 0.0025 * tt + 0.6 * w1 + 0.4 * w2 + e
    data2 = np.column_stack([y1t, y2t, y3t])

    all_cases = ["n", "co", "ci", "lo", "li", "colo", "coli", "cilo", "cili"]
    trending_cases = {
        det: full_block(VECM(data2, k_ar_diff=2, coint_rank=1, deterministic=det).fit())
        for det in all_cases
    }
    joh1 = coint_johansen(data2, det_order=1, k_ar_diff=2)
    beta_colo = np.array(trending_cases["colo"]["beta"])[:, 0]
    joh1_beta = joh1.evec[:, 0] / joh1.evec[0, 0]
    trending = {
        "data": [full(data2[:, j]) for j in range(3)],
        "k_ar_diff": 2,
        "coint_rank": 1,
        "cases": trending_cases,
        "johansen_det1": {
            "det_order": 1,
            "eig": full(joh1.eig),
            "evec": [full(row) for row in joh1.evec],
        },
        # Measured correspondences on this draw, pinned as documentation:
        # colo <-> coint_johansen(det_order=1) is the *asymptotic* analogue
        # of the exact co <-> det_order=0 identity (the projections differ
        # in finite samples), and the cross-case cosines show the case
        # choice changes beta.
        "beta_cosines": {
            "colo_joh1": cosine(beta_colo, joh1_beta),
            "co_coli": cosine(
                np.array(trending_cases["co"]["beta"])[:, 0],
                np.array(trending_cases["coli"]["beta"])[:, 0],
            ),
            "ci_cili": cosine(
                np.array(trending_cases["ci"]["beta"])[:, 0],
                np.array(trending_cases["cili"]["beta"])[:, 0],
            ),
        },
    }

    # ---- dataset 3: seasonal quarterly cointegrated pair.
    n3 = 300
    rng3 = np.random.default_rng(11)
    common = np.cumsum(0.01 + 0.10 * rng3.standard_normal(n3))
    seas1 = np.array([0.5, -0.3, 0.8, -1.0])
    seas2 = np.array([-0.2, 0.6, -0.4, 0.0])
    q = np.arange(n3) % 4
    y1s = common + seas1[q] + 0.10 * rng3.standard_normal(n3)
    y2s = 0.8 * common + 0.5 + seas2[q] + 0.10 * rng3.standard_normal(n3)
    data3 = np.column_stack([y1s, y2s])
    fit_co_s4 = VECM(
        data3, k_ar_diff=1, coint_rank=1, deterministic="co", seasons=4, first_season=2
    ).fit()
    fit_ci_s4 = VECM(
        data3, k_ar_diff=1, coint_rank=1, deterministic="ci", seasons=4
    ).fit()
    seasonal = {
        "data": [full(data3[:, j]) for j in range(2)],
        "k_ar_diff": 1,
        "coint_rank": 1,
        "seasons": 4,
        "co_s4_fs2": {"first_season": 2, **full_block(fit_co_s4)},
        "ci_s4_fs0": {"first_season": 0, **full_block(fit_ci_s4)},
    }

    out = {
        "_meta": {
            "statsmodels": statsmodels.__version__,
            "numpy": np.__version__,
            "python": platform.python_version(),
            "note": "VECM(k_ar_diff=1, coint_rank=1, deterministic='n'|'co') and "
                    "coint_johansen(det_order=0, k_ar_diff=1) on seeded drifting "
                    "cointegrated data (rank 1). deterministic='co' spans the same "
                    "cointegrating space as coint_johansen det_order=0; "
                    "deterministic='n' does not (beta cosine pinned below). "
                    "'trending': every deterministic case (n/co/ci/lo/li/colo/"
                    "coli/cilo/cili) at k_ar_diff=2 on trending cointegrated "
                    "data, with coint_johansen(det_order=1) and measured "
                    "cross-case beta cosines. 'seasonal': seasons=4 fits "
                    "(centered seasonal dummies, incl. first_season=2).",
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
        "trending": trending,
        "seasonal": seasonal,
    }
    (OUT / "vecm_deterministic.json").write_text(json.dumps(out, separators=(",", ":")))
    print("wrote vecm_deterministic.json  beta_cosine_n_co =", out["beta_cosine_n_co"])
    print("  trending beta_cosines:", trending["beta_cosines"])


if __name__ == "__main__":
    gen()
