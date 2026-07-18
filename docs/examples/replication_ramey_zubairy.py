"""Replication: Ramey & Zubairy (2018) government-spending multipliers.

The gallery's other pages run on synthetic data, where the truth is known by
construction. This one runs on the **authors' own data** and targets a
**published number**: RZ report integral government-spending multipliers of
roughly 0.6-0.8 — below one — in US historical data.

Data: Ramey & Zubairy (2018), "Government Spending Multipliers in Good Times and
in Bad: Evidence from US Historical Data", *Journal of Political Economy*
126(2):850-901. Downloaded on first use from the authors' replication archive by
`tsecon.datasets.ramey_zubairy()` and cached; nothing is vendored here.

    .venv/bin/python docs/examples/replication_ramey_zubairy.py

WHAT THIS IS, AND IS NOT
------------------------
This reproduces RZ's *headline integral multiplier* using their data, their
military-news instrument, and their Gordon-Krenn normalisation. It is NOT a
line-by-line port of their Stata code: their published tables involve sample
splits, lag choices and standard-error conventions this script does not
replicate exactly. The claim being checked is the one that matters
economically — that the multiplier is well below one — not bitwise equality.
"""
import numpy as np

import tsecon
from tsecon import datasets as ds


def rule(width=70, ch="-"):
    print(ch * width)


def build_variables(rz):
    """RZ / Gordon-Krenn normalisation: real quantities per unit of potential GDP.

    Dividing by potential output (rather than logging) is what puts the
    estimated coefficient in dollar-for-dollar units, so the regression
    coefficient *is* a multiplier rather than an elasticity.
    """
    s = rz["series"]
    pgdp, pot = s["pgdp"], s["rgdp_potcbo"]
    g = (s["ngov"] / pgdp) / pot          # real government spending / potential
    y = s["rgdp"] / pot                   # real GDP / potential
    # The news shock is a nominal present value; scale by LAGGED nominal
    # potential output so it is a share of the economy agents already knew about.
    nominal_potential = pgdp * pot
    newsy = s["news"] / np.roll(nominal_potential, 1)
    newsy[0] = np.nan
    return g, y, newsy


def integral_multiplier(g, y, shock, horizons=20, n_lag_controls=4):
    """RZ's integral multiplier: cumulative dY divided by cumulative dG.

    Both responses come from cumulative local projections on the same shock, so
    the ratio is the extra dollar of output per extra dollar of government
    spending accumulated through horizon h.

    Note this is deliberately NOT `lp_iv(..., cumulative=True)`: that cumulates
    only the OUTCOME, giving output per unit of *contemporaneous* spending,
    which grows without bound in the horizon and is not a multiplier.
    """
    cum_y = np.asarray(
        tsecon.lp(y, shock, horizons=horizons, n_lag_controls=n_lag_controls,
                  cumulative=True)["irf"]
    )
    cum_g = np.asarray(
        tsecon.lp(g, shock, horizons=horizons, n_lag_controls=n_lag_controls,
                  cumulative=True)["irf"]
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        mult = np.where(cum_g != 0.0, cum_y / cum_g, np.nan)
    return cum_y, cum_g, mult


def report(label, quarter, mask, g, y, newsy, horizons=20):
    gg, yy, nn = g[mask], y[mask], newsy[mask]
    cum_y, cum_g, mult = integral_multiplier(gg, yy, nn, horizons=horizons)
    print(f"\n{label}")
    print(f"  sample {quarter[mask][0]:.2f} to {quarter[mask][-1]:.2f}   "
          f"n = {int(mask.sum())} quarters")
    print(f"  {'h':>3} | {'cum dY':>8} | {'cum dG':>8} | {'multiplier':>10}")
    rule(46)
    for h in (2, 4, 8, 12, 16, 20):
        print(f"  {h:>3} | {cum_y[h]:>8.3f} | {cum_g[h]:>8.3f} | {mult[h]:>10.3f}")
    return mult


def main():
    print("Replication — Ramey & Zubairy (2018), JPE 126(2)")
    print("government-spending multipliers from US historical data")
    rule(70, "=")

    rz = ds.ramey_zubairy()
    print(f"data: {rz['source']}")
    print(f"      {len(rz['quarter'])} quarters, {len(rz['names'])} series")
    print(f"      sha256 {rz['sha256'][:16]}...")

    quarter = rz["quarter"]
    g, y, newsy = build_variables(rz)
    complete = ~np.isnan(g + y + newsy)

    full = report("FULL HISTORICAL SAMPLE (RZ's headline)",
                  quarter, complete, g, y, newsy)
    report("POSTWAR SUBSAMPLE (1947 onward)",
           quarter, complete & (quarter >= 1947.0), g, y, newsy)

    print()
    rule(70, "=")
    print("Published benchmark: RZ report integral multipliers of about")
    print("0.6-0.8 in the historical sample — the central claim being that the")
    print("multiplier is BELOW ONE, so a dollar of government spending buys")
    print("less than a dollar of output.")
    band = [m for h, m in zip(range(21), full) if h in (4, 8, 12, 16, 20)]
    print(f"\nThis replication, h = 4..20:  {np.min(band):.2f} to {np.max(band):.2f}")
    inside = all(0.5 <= m <= 0.9 for m in band)
    print(f"Inside the published 0.6-0.8 neighbourhood: {inside}")
    print("\nThe postwar subsample is far noisier and its long-horizon ratios are")
    print("unstable — RZ make the same point: the identifying variation in the")
    print("military-news series is overwhelmingly pre-1950, which is exactly why")
    print("their headline estimates use the long historical sample.")


if __name__ == "__main__":
    main()
