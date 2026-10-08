"""Grading the exact Durbin-Watson p-value of `tsecon.durbin_watson_test`
(the numbers quoted in the specification-tests model card).

1. NULL DISTRIBUTION. For three fixed designs, simulate the null
   distribution of d with NumPy (u iid N(0, 1); e = M u; d = e'Ae / e'e,
   N = 200 000 draws) and compare the empirical CDF P(d <= c) with the exact
   p-value tsecon reports for an observed d = c, at the 1/5/10/50/90/95/99%
   points of the simulated distribution. Reported: the largest
   |empirical - exact| in units of the Monte-Carlo standard error.

2. EXACT vs APPROXIMATE under non-normal errors. The p-value is exact for
   Gaussian errors only; the size at 5% under Student-t(3), t(5) and
   centered chi2(1) errors (fixed X, T = 30 and 100, R = 4000) shows how far
   it moves.

Run:  python lab/serial_corr/dw_exact_mc.py   (needs the built tsecon wheel)
Out:  lab/serial_corr/out/dw_exact_mc.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import tsecon

OUT = Path(__file__).resolve().parent / "out" / "dw_exact_mc.txt"


def null_draws(x, n_draws, rng):
    q, _ = np.linalg.qr(x)
    out = np.empty(n_draws)
    step = 20_000
    for s in range(0, n_draws, step):
        z = rng.standard_normal((min(step, n_draws - s), x.shape[0]))
        e = z - (z @ q) @ q.T
        out[s:s + len(z)] = np.sum(np.diff(e, axis=1) ** 2, axis=1) / np.sum(e**2, axis=1)
    return out


def y_with_statistic(x, c, rng):
    """A response whose OLS residuals have Durbin-Watson statistic c (bisect on
    the AR coefficient of a fixed innovation path, then use it as y: the
    residual of y on x is its own residual since y is built in the residual
    space)."""
    n = x.shape[0]
    q, _ = np.linalg.qr(x)
    e0 = rng.standard_normal(n)

    def dw_of(phi):
        u = np.empty(n)
        u[0] = e0[0]
        for t in range(1, n):
            u[t] = phi * u[t - 1] + e0[t]
        r = u - q @ (q.T @ u)
        return np.sum(np.diff(r) ** 2) / np.sum(r**2), r

    lo, hi = -0.99, 0.99
    d_lo, _ = dw_of(lo)
    d_hi, _ = dw_of(hi)
    if not (d_hi <= c <= d_lo):
        return None
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        d_mid, r = dw_of(mid)
        if d_mid > c:
            lo = mid
        else:
            hi = mid
    return dw_of(0.5 * (lo + hi))[1]


def designs(rng):
    n1 = 20
    x1 = np.ones((n1, 1))
    n2 = 40
    x2 = np.column_stack([np.ones(n2), rng.standard_normal(n2), np.arange(n2) / n2])
    n3 = 80
    x3 = np.column_stack([np.ones(n3), np.cumsum(rng.standard_normal(n3)), rng.standard_normal(n3),
                          np.sin(np.arange(n3) / 3.0)])
    return {"T=20, constant only": x1, "T=40, const+regressor+trend": x2,
            "T=80, const+random walk+iid+sine": x3}


def part1(rng, lines):
    lines.append("1. Exact p-value vs simulated null CDF (N = 200000 draws per design)")
    for name, x in designs(rng).items():
        d = null_draws(x, 200_000, rng)
        worst = 0.0
        rows = []
        for q in (0.01, 0.05, 0.10, 0.50, 0.90, 0.95, 0.99):
            c = float(np.quantile(d, q))
            y = y_with_statistic(x, c, rng)
            if y is None:
                continue
            r = tsecon.durbin_watson_test(y, x)
            emp = float(np.mean(d <= r["statistic"]))
            se = np.sqrt(max(r["pvalue"] * (1 - r["pvalue"]), 1e-12) / d.size)
            z = (emp - r["pvalue"]) / se
            worst = max(worst, abs(z))
            rows.append(f"    d={r['statistic']:.4f}  exact P(d<=c)={r['pvalue']:.5f}  "
                        f"simulated={emp:.5f}  z={z:+.2f}")
        lines.append(f"  {name}: largest |z| = {worst:.2f}")
        lines.extend(rows)
    lines.append("")


def part2(rng, lines):
    lines.append("2. Size at 5% of the exact test (rho > 0) under non-normal errors, R = 4000")
    shocks = {
        "normal": lambda n: rng.standard_normal(n),
        "t(5)": lambda n: rng.standard_t(5, n),
        "t(3)": lambda n: rng.standard_t(3, n),
        "chi2(1) centered": lambda n: rng.chisquare(1, n) - 1.0,
    }
    for n in (30, 100):
        x = np.column_stack([np.ones(n), rng.standard_normal(n), np.arange(n) / n])
        cells = []
        for name, draw in shocks.items():
            rej = 0
            for _ in range(4000):
                y = x @ np.array([1.0, 0.5, -0.5]) + draw(n)
                rej += tsecon.durbin_watson_test(y, x)["pvalue"] < 0.05
            cells.append(f"{name}={rej / 4000:.4f}")
        lines.append(f"  T = {n}: " + "  ".join(cells))
    lines.append("  (MC se 0.0034 at a true 0.05)")


def main():
    rng = np.random.default_rng(20261008)
    lines = [f"tsecon {tsecon.__version__}, numpy {np.__version__}, python {sys.version.split()[0]}", ""]
    part1(rng, lines)
    part2(rng, lines)
    text = "\n".join(lines)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
