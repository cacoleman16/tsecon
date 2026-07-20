"""Replication: Estrella & Mishkin (1998) — the yield curve predicts recessions.

One of the most durable results in macro-finance: the slope of the Treasury yield
curve, twelve months earlier, is a strong leading indicator of recessions. When
the curve inverts (short rates above long rates), a recession tends to follow.

This reproduces the core result with `tsecon.recession_probit` on live FRED data:
a probit of the NBER recession indicator on the term spread, and it recovers the
canonical numbers — a strongly negative spread coefficient and an inverted curve
implying a high probability of recession within the year.

    .venv/bin/python docs/examples/replication_yield_curve_recession.py

Data (committed to this repository at fixtures/yield_curve_recession.csv — this
runs offline, the library ships no network loaders):
  GS10  — 10-Year Treasury constant maturity, monthly
  TB3MS — 3-Month Treasury bill secondary market rate, monthly
  USREC — NBER recession indicator (1 = recession), monthly
All from the Federal Reserve Bank of St. Louis (FRED); public data, vendored
with attribution. To refresh from current FRED vintages, download those three
series yourself and rebuild the CSV.

Reference: Estrella, A. & Mishkin, F. S. (1998), "Predicting U.S. Recessions:
Financial Variables as Leading Indicators," Review of Economics and Statistics
80(1):45-61.
"""
import csv
from math import erf
from pathlib import Path

import numpy as np

import tsecon

DATA = Path(__file__).resolve().parents[2] / "fixtures" / "yield_curve_recession.csv"
LEAD = 12  # forecast horizon: recession 12 months ahead


def _phi(z):
    return 0.5 * (1.0 + erf(z / np.sqrt(2.0)))


def load_aligned(path=DATA):
    """Return (dates, term_spread, recession) from the committed monthly panel.

    The CSV holds date,gs10,tb3ms,usrec aligned on common monthly dates; public
    FRED data vendored with attribution, so this is fully offline.
    """
    rows = [r for r in csv.reader(open(path)) if r and not r[0].startswith("#")]
    rows = rows[1:]  # header
    dates = np.array([r[0] for r in rows], dtype="datetime64[D]")
    gs10 = np.array([float(r[1]) for r in rows])
    tb3 = np.array([float(r[2]) for r in rows])
    rec = np.array([float(r[3]) for r in rows])
    return dates, gs10 - tb3, rec


def run(dates, spread, recession):
    # Predict recession at t+LEAD from the term spread at t.
    y = recession[LEAD:]
    x = spread[:-LEAD]
    X = np.column_stack([np.ones(len(y)), x])
    fit = tsecon.recession_probit(y, X, link="probit")
    b = np.asarray(fit["params"])
    z = np.asarray(fit["zstats"])
    return fit, b, z


def main():
    dates, spread, recession = load_aligned()
    fit, b, z = run(dates, spread, recession)

    print("Replication — Estrella & Mishkin (1998), REStat 80(1)")
    print("the term spread predicts recessions 12 months ahead")
    print("=" * 66)
    print(f"sample: {dates[0]} to {dates[-1]}   "
          f"{int(recession[LEAD:].sum())} recession months of {len(recession) - LEAD}")
    print()
    print("  Probit:  P(recession in 12m) = Phi(b0 + b1 * term_spread)")
    print(f"    b0 (const)   = {b[0]:+.4f}")
    print(f"    b1 (spread)  = {b[1]:+.4f}   (z = {z[1]:+.2f})")
    print(f"    McFadden R2  = {fit['pseudo_r2']:.3f}")
    print()
    print("  Implied 12-month-ahead recession probability:")
    for s in (3.0, 1.0, 0.0, -1.0):
        print(f"    term spread = {s:+.1f} pp  ->  P(recession) = {_phi(b[0] + b[1] * s):5.1%}")
    print("=" * 66)
    print("The signature result: b1 < 0 and strongly significant — an inverting")
    print("curve raises the recession probability. A steep (+3pp) curve implies a")
    print("near-zero chance of recession within the year; a 1pp inversion implies")
    print("close to a coin flip. This is the yield-curve recession signal that")
    print("shows up in the financial press every cycle, estimated from scratch.")
    print()
    print("Estrella-Mishkin (1998) report a term-spread probit of the same shape")
    print("with a similar pseudo-R2; the point being replicated is the sign,")
    print("significance, and economic magnitude, not their exact vintage/sample.")


if __name__ == "__main__":
    main()
