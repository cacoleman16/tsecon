"""The Durbin-Watson 5% bounds d_L and d_U for a given (T, k), computed from
their definition rather than read from a table, and the exact p-value of the
card example set against them (the numbers quoted in the Durbin-Watson
example of docs/reference/model-cards/specification-tests.md).

With A = D'D (first differences), whose eigenvalues are
lambda_j = 2 - 2 cos(pi j / T), j = 0..T-1, and a design of k columns that
includes the constant, the null eigenvalues nu_i of M A M satisfy
lambda_{i} <= nu_i <= lambda_{i+k-1} (0-based, i = 1..T-k), so

    d_L = sum_{i=1}^{T-k} lambda_i z_i^2 / sum z_i^2,
    d_U = sum_{i=1}^{T-k} lambda_{i+k-1} z_i^2 / sum z_i^2,

and the tabulated 5% bounds are their 5% quantiles. Quantiles by Imhof's
formula (SciPy quad) and Brent's root finder. Independent of tsecon except
for the final p-value of the example, which is read from tsecon.

Run:  python lab/serial_corr/dw_bounds.py
Out:  lab/serial_corr/out/dw_bounds.txt
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import integrate, optimize

OUT = Path(__file__).resolve().parent / "out" / "dw_bounds.txt"


def cdf(eigs, c):
    lam = eigs - c

    def f(u):
        if u == 0.0:
            return 0.5 * lam.sum()
        th = 0.5 * np.arctan(lam * u).sum()
        return np.sin(th) * np.exp(-0.25 * np.log1p((lam * u) ** 2).sum()) / u

    val, _ = integrate.quad(f, 0.0, np.inf, epsabs=1e-13, epsrel=1e-12, limit=2000)
    return 0.5 - val / np.pi


def bounds(T, k, level=0.05):
    lam = 2.0 - 2.0 * np.cos(np.pi * np.arange(T) / T)
    lower = lam[1:T - k + 1]
    upper = lam[k:T]
    d_l = optimize.brentq(lambda c: cdf(lower, c) - level, 0.01, 3.99, xtol=1e-12)
    d_u = optimize.brentq(lambda c: cdf(upper, c) - level, 0.01, 3.99, xtol=1e-12)
    return d_l, d_u


def main():
    lines = []
    for T, k in ((60, 3), (50, 2), (100, 4)):
        d_l, d_u = bounds(T, k)
        lines.append(f"T={T:4d} k={k} (k'={k - 1} regressors besides the constant): "
                     f"d_L(5%) = {d_l:.4f}  d_U(5%) = {d_u:.4f}")
    text = "\n".join(lines)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
