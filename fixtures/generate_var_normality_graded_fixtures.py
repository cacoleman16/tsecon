"""Golden fixtures for the SYMMETRIC-ROOT multivariate Jarque-Bera on residual
matrices whose columns are on very different scales (`*_symmetric` keys of
`multivariate_normality` and `var_diagnostics`; crate `tsecon-var`,
`normality_test_symmetric`).

    PYTHONPATH=<dir with mpmath> .venv/bin/python fixtures/generate_var_normality_graded_fixtures.py

The statistic (JMulTi's "Doornik & Hansen (1994)" block) orthogonalises the
centred residuals with the symmetric inverse root S^{-1/2} of their ML
covariance S = u'u / T, so it depends on the units of the columns. With
column scales d_j, S = D C D is GRADED: its eigenvalues span the squared
scale ratio. The statistic itself is well conditioned in the data (a
relative 2^-52 perturbation of every entry moves it by about 1e-15), but an
eigensolver that is accurate only relative to the largest eigenvalue loses
digits in proportion to trace / lambda_min: the unbalanced solve was off by
2.0e-10 at a column-scale ratio of 1e3, 1.3e-6 at 1e5 and 7.0e-5 at 3e5, all
inputs the 1e-12 x trace floor accepts.

Reference: mpmath at 60 significant digits on the EXACT float64 inputs —
column means, S, its symmetric eigendecomposition (`mp.eigsy`), S^{-1/2},
the orthogonalised rows, b1, excess b2, the statistics and the chi-squared
p-values (`mp.gammainc`, regularized upper) — the transcription in
fixtures/generate_var_diag_fixtures.py carried out in extended precision.

Cases: seeded skewed panels X0 (numpy default_rng(0); k = 2: exponential
draws through [[1, 0.3], [0.5, 1]]; k = 3, 4: exponential draws through
I + 0.3 U(0, 1), the auditor's design), T = 100, and the inputs are
X0 * scales (one multiplication per entry, so a consumer rebuilds them bit
for bit from the stored X0 and scales):

* k = 2 at scale ratios 1, 1e2, 1e3, 1e4, 1e5, 10^5.5 and 10^5.75, both
  column orders;
* k = 3 and k = 4 at ratio 10^5.5 (geometric scales), every column order
  (the old solver's loss depended on the order).

`lam_min_over_trace` (from mpmath) shows every case is above the 1e-12
floor. Never imports tsecon.
"""

from __future__ import annotations

import itertools
import json
import platform
from pathlib import Path

import mpmath as mp
import numpy as np

mp.mp.dps = 60

OUT = Path(__file__).with_name("var_normality_graded.json")


def mp_symmetric_jb(x: np.ndarray) -> dict:
    n, k = x.shape
    m = mp.matrix([[mp.mpf(float(v)) for v in row] for row in x])
    mean = [mp.fsum(m[i, j] for i in range(n)) / n for j in range(k)]
    u = mp.matrix(n, k)
    for i in range(n):
        for j in range(k):
            u[i, j] = m[i, j] - mean[j]
    s = (u.T * u) / n
    e, q = mp.eigsy(s)
    root = q * mp.diag([1 / mp.sqrt(v) for v in e]) * q.T
    w = root * u.T  # k x n
    b1 = [mp.fsum(w[j, i] ** 3 for i in range(n)) / n for j in range(k)]
    b2 = [mp.fsum(w[j, i] ** 4 for i in range(n)) / n - 3 for j in range(k)]
    lam_s = n * mp.fsum(b**2 for b in b1) / 6
    lam_k = n * mp.fsum(b**2 for b in b2) / 24
    stat = lam_s + lam_k

    def sf(xv, df):
        return mp.gammainc(mp.mpf(df) / 2, xv / 2, mp.inf, regularized=True)

    trace = mp.fsum(s[j, j] for j in range(k))
    return {
        "jarque_bera_symmetric": float(stat),
        "jarque_bera_symmetric_pvalue": float(sf(stat, 2 * k)),
        "skewness_symmetric": float(lam_s),
        "skewness_symmetric_pvalue": float(sf(lam_s, k)),
        "kurtosis_symmetric": float(lam_k),
        "kurtosis_symmetric_pvalue": float(sf(lam_k, k)),
        "skewness_symmetric_components": [float(b) for b in b1],
        "kurtosis_symmetric_components": [float(b) for b in b2],
        "lam_min_over_trace": float(min(e) / trace),
    }


def main() -> None:
    rng = np.random.default_rng(0)
    panels = {
        2: rng.exponential(size=(100, 2)) @ np.array([[1.0, 0.3], [0.5, 1.0]]).T,
    }
    rng = np.random.default_rng(0)
    for k in (3, 4):
        a = np.eye(k) + 0.3 * rng.uniform(size=(k, k))
        panels[k] = rng.exponential(size=(100, k)) @ a.T

    cases = []
    for logr in (0.0, 2.0, 3.0, 4.0, 5.0, 5.5, 5.75):
        for flip in (False, True):
            sc = np.array([10 ** (logr / 2), 10 ** (-logr / 2)])
            if flip:
                sc = sc[::-1]
            cases.append({"k": 2, "log10_ratio": logr, "scales": sc.tolist()})
    for k in (3, 4):
        base = np.logspace(5.5 / 2, -5.5 / 2, k)
        for perm in itertools.permutations(range(k)):
            cases.append(
                {"k": k, "log10_ratio": 5.5, "scales": base[list(perm)].tolist()}
            )

    for c in cases:
        x = panels[c["k"]] * np.array(c["scales"])
        c.update(mp_symmetric_jb(x))
        assert c["lam_min_over_trace"] > 1e-12, c
        print(
            f"k={c['k']} ratio 1e{c['log10_ratio']:<5} lam_min/tr "
            f"{c['lam_min_over_trace']:.1e}  JB_sym {c['jarque_bera_symmetric']:.12f}"
        )

    out = {
        "_meta": {
            "generator": "fixtures/generate_var_normality_graded_fixtures.py",
            "numpy": np.__version__,
            "mpmath": mp.__version__,
            "mp_dps": mp.mp.dps,
            "python": platform.python_version(),
        },
        "panels": {str(k): v.tolist() for k, v in panels.items()},
        "cases": cases,
    }
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(f"{len(cases)} cases -> {OUT.name}")


if __name__ == "__main__":
    main()
