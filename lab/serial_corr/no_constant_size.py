"""Seeded Monte Carlo: size of `breusch_godfrey_test` on a regression through
the origin (a design with no constant column), the case where the auxiliary
regression gains a constant (`constant_added`). Quoted in the
specification-tests model card.

statsmodels' `acorr_breusch_godfrey`, which the test is pinned to, forms
LM = T * R^2 with the CENTERED R^2 of the auxiliary regression of u on
[X, 1, u_{-1}..u_{-p}]. With a constant in X the lag columns are the only
thing the regression adds, so LM tests them alone. Without one, the added
constant also explains part of u (its projection on M_X 1, not just its
mean), and LM counts that too; the F-form compares with the regression on
[X, 1] and tests the lags alone.

DGP: y_t = a + 0.5 x_t + e_t, e iid N(0, 1), regressed on [x] alone (no
constant), x drawn once per cell and held fixed. Two regressors:
  * x = 2 + N(0, 1)  (a regressor with a nonzero mean: M_X 1 is far from
    constant), and
  * x = N(0, 1)      (mean near zero: M_X 1 is close to the constant).
Two intercepts: a = 0 (the through-origin model is correct) and a = 0.5
(an omitted intercept). Errors are iid, so every rejection is a size error.

Rejection rates at the 5% level for nlags = 1 and 4, R = 2000 replications
per cell (Monte-Carlo standard error 0.49 percentage points at a true 5%).

Run:  python lab/serial_corr/no_constant_size.py   (needs the built wheel)
Out:  lab/serial_corr/out/no_constant_size.txt
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

import tsecon

R = 2000
OUT = Path(__file__).resolve().parent / "out" / "no_constant_size.txt"


def main():
    t0 = time.perf_counter()
    lines = [
        f"Breusch-Godfrey on a regression through the origin, iid N(0,1) errors: "
        f"rejection rates at 5%, R = {R} (MC se 0.49 pp at 5%). tsecon {tsecon.__version__}, "
        f"numpy {np.__version__}, python {sys.version.split()[0]}.",
        "",
        f"  {'T':>4s}  {'regressor':14s} {'intercept':>9s}   "
        f"{'LM p=1':>7s} {'F p=1':>7s}   {'LM p=4':>7s} {'F p=4':>7s}",
    ]
    seed = 20264000
    for n in (50, 200):
        for label, mean in (("mean 2", 2.0), ("mean 0", 0.0)):
            for a in (0.0, 0.5):
                seed += 1
                rng = np.random.default_rng(seed)
                x = mean + rng.standard_normal(n)
                X = x[:, None]
                rej = {"LM1": 0, "F1": 0, "LM4": 0, "F4": 0}
                for _ in range(R):
                    y = a + 0.5 * x + rng.standard_normal(n)
                    for p in (1, 4):
                        r = tsecon.breusch_godfrey_test(y, X, nlags=p)
                        assert r["constant_added"]
                        rej[f"LM{p}"] += r["pvalue"] < 0.05
                        rej[f"F{p}"] += r["f_pvalue"] < 0.05
                lines.append(
                    f"  {n:4d}  x {label:12s} {a:9.1f}   "
                    f"{rej['LM1'] / R:7.3f} {rej['F1'] / R:7.3f}   "
                    f"{rej['LM4'] / R:7.3f} {rej['F4'] / R:7.3f}"
                )
    lines += ["", f"elapsed {time.perf_counter() - t0:.0f} s"]
    text = "\n".join(lines)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
