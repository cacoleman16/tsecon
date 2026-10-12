"""The two runnable examples of the serial-correlation sections of
docs/reference/model-cards/specification-tests.md, verbatim; the printed
lines are pasted into the card as its expected output.

Run:  python lab/serial_corr/card_examples.py   (needs the built tsecon wheel)
Out:  lab/serial_corr/out/card_examples.txt
"""
import contextlib
import io
from pathlib import Path

OUT = Path(__file__).resolve().parent / "out" / "card_examples.txt"

BG_EXAMPLE = '''
import numpy as np, tsecon

rng = np.random.default_rng(4)
n = 200
x1 = rng.normal(size=n)
X = np.column_stack([np.ones(n), x1])              # explicit constant column

u_iid = rng.normal(size=n)                          # no serial correlation
u_ar = np.empty(n)                                  # AR(2) errors: 0.3, 0.3
u_ar[:2] = rng.normal(size=2)
for t in range(2, n):
    u_ar[t] = 0.3 * u_ar[t - 1] + 0.3 * u_ar[t - 2] + rng.normal()

for label, u in [("iid errors", u_iid), ("AR(2) errors", u_ar)]:
    y = 1.0 + 0.8 * x1 + u
    bg = tsecon.breusch_godfrey_test(y, X, nlags=4)
    print(f"{label:>12}: LM={bg['statistic']:6.2f} (df={bg['df']}) p={bg['pvalue']:.4f}  |  "
          f"F={bg['fstat']:5.2f} df=({bg['f_df_num']}, {bg['f_df_den']}) p={bg['f_pvalue']:.4f}")
'''

DW_EXAMPLE = '''
import numpy as np, tsecon

rng = np.random.default_rng(5)
n = 60
x1 = rng.normal(size=n)
trend = np.arange(n) / n
X = np.column_stack([np.ones(n), x1, trend])       # explicit constant column

e = rng.normal(size=n)
u = np.empty(n)
u[0] = e[0] / np.sqrt(1 - 0.4**2)
for t in range(1, n):
    u[t] = 0.4 * u[t - 1] + e[t]                    # AR(1) errors, rho = 0.4

for label, err in [("iid errors", e), ("AR(1) errors", u)]:
    y = 1.0 + 0.5 * x1 - 1.0 * trend + err
    dw = tsecon.durbin_watson_test(y, X)            # H1: rho > 0
    two = tsecon.durbin_watson_test(y, X, alternative="two_sided")
    print(f"{label:>12}: d={dw['statistic']:.3f}  exact p(rho>0)={dw['pvalue']:.4f}  "
          f"two-sided p={two['pvalue']:.4f}  (T={dw['nobs']}, T-k={dw['df_resid']})")
'''


def main():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        for src in (BG_EXAMPLE, DW_EXAMPLE):
            exec(compile(src, "<card example>", "exec"), {})
            print()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(buf.getvalue(), encoding="utf-8")
    print(buf.getvalue())


if __name__ == "__main__":
    main()
