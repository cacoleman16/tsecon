"""Monte-Carlo size and power study for tsecon.variance_ratio.

Runs the shipped test (not a transcription) on simulated random walks and
reports rejection rates at the nominal 5% level. Its output is quoted
verbatim in docs/reference/model-cards/unit-root-cointegration-tests.md
(the `variance_ratio` section).

    python scripts/mc_variance_ratio_size.py

Design
------
Levels y of length n = 500 (499 increments), horizons q in (2, 4, 8, 16),
arch's default call (trend "c", overlapping, debiased). For each draw:

* per-horizon rejection: two-sided `pvalue < 0.05`, with the
  heteroskedasticity-robust z* (`robust=None` -> True) and with the
  homoskedastic z (`robust=False`);
* Chow-Denning rejection: `cd_pvalue < 0.05` over all four horizons (the
  SMM(inf) / Sidak bound), for both standardizations.

DGPs for the increments:

* iid N(0, 1)                       — the null with homoskedasticity;
* GARCH(1, 1), omega 0.05, alpha 0.15, beta 0.80 — the null with
  volatility clustering (martingale differences, not iid);
* AR(1) phi = 0.1 iid innovations   — a weak momentum alternative (power).

Every replication is a fresh draw from one seeded numpy default_rng
(seed 12345, 600 replications per DGP); the Monte-Carlo standard error of a
rate near 5% is about 0.009.
"""
from __future__ import annotations

import numpy as np

import tsecon

SEED = 12345
REPS = 600
N = 500
QS = (2, 4, 8, 16)


def iid(rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal(N - 1)


def garch(rng: np.random.Generator, omega=0.05, alpha=0.15, beta=0.80) -> np.ndarray:
    e = np.empty(N - 1)
    s2 = omega / (1 - alpha - beta)
    z = rng.standard_normal(N - 1)
    for t in range(N - 1):
        e[t] = np.sqrt(s2) * z[t]
        s2 = omega + alpha * e[t] ** 2 + beta * s2
    return e


def ar1(rng: np.random.Generator, phi=0.1) -> np.ndarray:
    z = rng.standard_normal(N - 1)
    u = np.empty(N - 1)
    prev = 0.0
    for t in range(N - 1):
        prev = phi * prev + z[t]
        u[t] = prev
    return u


DGPS = (
    ("iid N(0,1) increments (null)", iid),
    ("GARCH(1,1) a=0.15 b=0.80 increments (null)", garch),
    ("AR(1) phi=0.1 increments (alternative)", ar1),
)


def main() -> None:
    rng = np.random.default_rng(SEED)
    print(f"tsecon {tsecon.__version__}; n = {N}, q = {QS}, {REPS} reps per DGP, "
          f"seed {SEED}, nominal level 5%")
    for label, draw in DGPS:
        rej = {True: np.zeros(len(QS)), False: np.zeros(len(QS))}
        cd = {True: 0, False: 0}
        for _ in range(REPS):
            y = np.concatenate([[0.0], np.cumsum(draw(rng))])
            for robust in (True, False):
                r = tsecon.variance_ratio(y, list(QS), robust=robust)
                rej[robust] += r["pvalue"] < 0.05
                cd[robust] += r["cd_pvalue"] < 0.05
        print(label)
        for robust, name in ((True, "robust z*    "), (False, "homoskedastic")):
            rates = " / ".join(f"{v / REPS:.3f}" for v in rej[robust])
            print(f"  {name}  per-horizon q=2/4/8/16: {rates}   "
                  f"Chow-Denning: {cd[robust] / REPS:.3f}")


if __name__ == "__main__":
    main()
