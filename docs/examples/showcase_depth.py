"""The tsecon gallery, depth wing: the Phase-3/4 methods — score-driven
volatility, and (as they land) connectedness, factor-augmented VARs, the
term structure, and nowcasting.

Run with the project venv (tsecon + matplotlib installed there):
    .venv/bin/python docs/examples/showcase_depth.py

Figures land in docs/examples/img/ in the house style (Module 13).
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = Path(__file__).parents[2]
sys.path.insert(0, str(REPO / "prototypes" / "viz"))
import tsecon_style as ts  # noqa: E402
import tsecon  # noqa: E402

IMG = Path(__file__).parent / "img"
IMG.mkdir(exist_ok=True)


def save(fig, name):
    fig.savefig(IMG / name)
    plt.close(fig)
    print("wrote", IMG / name)


# ------------------------------------------------------------------
# D1. GAS score-driven volatility: the Student-t score is robust to jumps
# ------------------------------------------------------------------
def section_gas():
    rng = np.random.default_rng(20260717)
    n = 500
    # A GARCH-like series with genuine volatility clustering...
    h = np.empty(n)
    r = np.empty(n)
    h[0] = 1.0
    r[0] = rng.standard_normal()
    for t in range(1, n):
        h[t] = 0.05 + 0.08 * r[t - 1] ** 2 + 0.90 * h[t - 1]
        r[t] = np.sqrt(h[t]) * rng.standard_normal()
    # ...plus a few isolated jumps (fat-tailed shocks, not regime changes).
    jumps = [140, 300, 410]
    for j in jumps:
        r[j] += np.sign(rng.standard_normal() + 0.1) * 8.0

    g = tsecon.gas_volatility(r, density="gaussian")
    st = tsecon.gas_volatility(r, density="student_t")
    vol_g = np.sqrt(np.asarray(g["variance"]))
    vol_t = np.sqrt(np.asarray(st["variance"]))
    x = np.arange(n)

    with ts.theme():
        fig, (ax0, ax1) = plt.subplots(
            2, 1, figsize=(ts.WIDTH_DOUBLE, 3.5), sharex=True,
            gridspec_kw={"height_ratios": [1.0, 1.15]},
        )
        # Top: the return series, jumps flagged.
        ax0.plot(x, r, color=ts.INK_2, lw=0.5)
        ts.zero_line(ax0)
        ax0.plot(jumps, r[jumps], "o", color=ts.SERIES["red"], ms=4.5, zorder=5,
                 markeredgecolor=ts.SURFACE, markeredgewidth=0.6)
        ax0.set_ylabel("Return", fontsize=8.5, color=ts.INK)
        ax0.tick_params(labelsize=7.5)

        # Bottom: the two conditional-volatility paths.
        ax1.plot(x, vol_g, color=ts.SERIES["blue"], lw=1.5, zorder=4)
        ax1.plot(x, vol_t, color=ts.SERIES["red"], lw=1.5, zorder=5)
        for j in jumps:
            ax1.axvline(j, color=ts.REF, lw=0.7, ls=(0, (2, 2)), zorder=1)
        ax1.set_ylabel("Conditional volatility", fontsize=8.5, color=ts.INK)
        ax1.set_xlabel("Time", fontsize=8.5, color=ts.INK_2)
        ax1.set_xlim(0, n - 1)
        ax1.tick_params(labelsize=7.5)
        # Sits in the calm top-left, clear of every volatility spike.
        ax1.annotate(
            f"Student-t estimated $\\nu$ = {st['nu']:.1f}  (heavy tails)",
            xy=(0.015, 0.95), xycoords="axes fraction", ha="left", va="top",
            fontsize=7.5, color=ts.SERIES["red"],
        )

        handles = [
            Line2D([0], [0], color=ts.SERIES["blue"], lw=1.5, label="Gaussian GAS"),
            Line2D([0], [0], color=ts.SERIES["red"], lw=1.5, label="Student-t GAS"),
        ]
        fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
                   fontsize=8, handlelength=1.6, columnspacing=1.8,
                   bbox_to_anchor=(0.5, 0.01))
        fig.suptitle("Student-t GAS shrugs off the jumps a Gaussian GAS chases",
                     x=0.005, ha="left", fontsize=11.5, fontweight="semibold",
                     color=ts.INK)
        fig.tight_layout(rect=(0, 0.07, 1, 0.92))
        ts.stamp(fig, "GAS(1,1) score-driven volatility · tsecon.gas_volatility "
                      "(Creal-Koopman-Lucas 2013) · the Student-t score down-weights "
                      "each observation by 1/(1 + y^2/((nu-2)f)), so isolated jumps "
                      "barely move the volatility estimate, while the Gaussian score "
                      "(y^2 - f) reacts one-for-one to the squared jump")
        save(fig, "depth-gas-volatility.png")


ALL = [section_gas]

if __name__ == "__main__":
    only = sys.argv[1:] if len(sys.argv) > 1 else None
    for fn in ALL:
        if only and not any(k in fn.__name__ for k in only):
            continue
        fn()
