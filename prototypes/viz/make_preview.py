"""Generate the visual-identity preview figures (Module 13 prototype).

Four canonical charts on synthetic/real data, written to docs/assets/viz-preview/.
Run:  python3 prototypes/viz/make_preview.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
import tsecon_style as ts

OUT = Path(__file__).parents[2] / "docs" / "assets" / "viz-preview"
OUT.mkdir(parents=True, exist_ok=True)
FIXTURES = Path(__file__).parents[2] / "fixtures"
rng = np.random.default_rng(20260716)

# ----------------------------------------------------------------- coverage
# Single source of truth for every uncertainty band in these previews: each
# band's WIDTH and its LABEL both derive from these levels, so changing the
# percents here changes every figure and its annotations consistently.
COVERAGE = {
    "ci": 0.95,                        # white-noise / residual-ACF bands
    "irf": [0.68, 0.90],               # nested IRF bands, inner -> outer
    "fan": [0.30, 0.50, 0.70, 0.90],   # fan chart, inner -> outer
}


def z_two_sided(level):
    """Exact two-sided normal multiplier: z = Phi^-1((1 + level) / 2).

    E.g. 0.95 -> 1.9600, 0.90 -> 1.6449, 0.68 -> 0.9945, 0.30 -> 0.3853.
    """
    from scipy.stats import norm
    return float(norm.ppf((1.0 + level) / 2.0))


def pct(level):
    """Format a coverage level as a percent label: 0.90 -> '90%'."""
    return f"{100 * level:g}%"


def pct_list(levels):
    """Format levels as a joint label: [0.68, 0.90] -> '68 / 90%'."""
    return " / ".join(f"{100 * lv:g}" for lv in sorted(levels)) + "%"


def irf(h, peak, decay, delay=0.0, osc=0.0):
    t = np.arange(h + 1, dtype=float)
    r = peak * (t / max(delay, 1e-9)) ** (delay > 0) * np.exp(-decay * np.maximum(t - delay, 0))
    if delay > 0:
        r = peak * (t / delay) * np.exp(1 - t / delay) * np.exp(-0.0 * t)
        r = peak * (t / delay) ** 1.4 * np.exp(1.4 * (1 - t / delay))
    if osc:
        r *= np.cos(osc * t) * 0.5 + 0.5
    return r


# ------------------------------------------------------------- 1. IRF grid
def fig_irf_grid():
    h = 20
    t = np.arange(h + 1)
    shocks = ["Monetary policy shock (+25bp)", "Demand shock (+1σ)"]
    variables = ["Output gap (%)", "Inflation (pp)", "Policy rate (pp)"]
    responses = {
        (0, 0): -irf(h, 0.42, 1, delay=5),
        (0, 1): -irf(h, 0.25, 1, delay=7),
        (0, 2): 0.25 * np.exp(-0.25 * t),
        (1, 0): irf(h, 0.55, 1, delay=3),
        (1, 1): irf(h, 0.35, 1, delay=5),
        (1, 2): irf(h, 0.30, 1, delay=6),
    }
    with ts.theme():
        fig, axes = plt.subplots(2, 3, figsize=(ts.WIDTH_DOUBLE, 3.6), sharex=True)
        for i, shock in enumerate(shocks):
            for j, var in enumerate(variables):
                ax = axes[i, j]
                point = responses[(i, j)]
                se = 0.10 * (1 + 0.08 * t) * np.max(np.abs(point) + 0.1)
                ts.zero_line(ax)
                irf_levels = sorted(COVERAGE["irf"])  # inner -> outer
                ts.nested_bands(ax, t, point, [z_two_sided(lv) * se for lv in irf_levels],
                                irf_levels, color_steps=["#cde2fb", "#9ec5f4"])
                ax.plot(t, point, color=ts.SERIES["blue"], lw=1.8, zorder=3)
                if i == 0:
                    ax.set_title(var, fontsize=9, loc="center", color=ts.INK_2, fontweight="normal")
                if j == 0:
                    ax.set_ylabel(shock.split(" (")[0], fontsize=8.5, color=ts.INK)
                ax.set_xlim(0, h)
                ax.tick_params(labelsize=7.5)
        for ax in axes[1]:
            ax.set_xlabel("Horizon (quarters)", fontsize=8)
        fig.suptitle(
            "A monetary tightening lowers output and prices; a demand shock raises both",
            x=0.002, ha="left", fontsize=11.5, fontweight="semibold", color=ts.INK,
        )
        fig.tight_layout(rect=(0, 0.015, 1, 0.94))
        ts.stamp(fig, f"Bands: {pct_list(COVERAGE['irf'])} (asymptotic) · Identification: external instrument · Synthetic preview data · tsecon 0.0.1")
        fig.savefig(OUT / "irf-panel-grid.png")
        plt.close(fig)


# ------------------------------------------------------------- 2. fan chart
def fig_fan_chart():
    n_hist, n_fc = 40, 12
    x_hist = np.arange(n_hist)
    x_fc = np.arange(n_hist - 1, n_hist + n_fc)
    level = 2.0 + 0.8 * np.sin(x_hist / 6) + rng.normal(0, 0.45, n_hist).cumsum() * 0.14
    level[22:27] -= np.array([0.5, 1.3, 1.8, 1.2, 0.6])
    point = level[-1] + 0.75 * (2.1 - level[-1]) * (1 - 0.78 ** np.arange(n_fc + 1))
    se = 0.28 * np.sqrt(np.arange(n_fc + 1) + 1e-9)

    with ts.theme():
        fig, ax = plt.subplots(figsize=(ts.WIDTH_DOUBLE, 2.9))
        ts.shade_period(ax, 21.5, 27.5, "recession")
        # Darkest at the center (most likely), fading outward — BoE convention.
        fan_levels = sorted(COVERAGE["fan"], reverse=True)  # widest first, lightest step
        for cov, step in zip(fan_levels, ts.SEQ_BLUE):
            z = z_two_sided(cov)
            ax.fill_between(x_fc, point - z * se, point + z * se, color=step, lw=0, zorder=2)
        ax.plot(x_hist, level, color=ts.INK, lw=1.6, zorder=3)
        ax.plot(x_fc, point, color=ts.SURFACE, lw=1.6, ls=(0, (4, 2)), zorder=3)
        ticks = np.arange(0, n_hist + n_fc, 8)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{2016 + (2 + k) // 4}Q{(2 + k) % 4 + 1}" for k in ticks])
        ax.axvline(n_hist - 1, color=ts.REF, lw=0.9, zorder=1.5)
        ax.annotate("forecast origin\n2026Q2", xy=(n_hist - 1, ax.get_ylim()[0]),
                    xytext=(n_hist - 0.4, 0.02), textcoords=("data", "axes fraction"),
                    fontsize=7.5, color=ts.MUTED, va="bottom")
        z_outer = z_two_sided(fan_levels[0])
        ax.annotate(f"{pct(fan_levels[0])} band",
                    xy=(x_fc[-1] + 0.25, point[-1] + z_outer * se[-1] * 0.82),
                    fontsize=7.5, color=ts.INK_2, va="center")
        ax.annotate("median path", xy=(x_fc[-1] + 0.25, point[-1]), fontsize=7.5,
                    color=ts.SEQ_BLUE[5], va="center")
        ax.set_xlim(0, n_hist + n_fc + 5.5)
        ax.set_ylabel("Four-quarter GDP growth (%)")
        ax.set_title("Growth recovers toward 2% over the forecast horizon")
        ts.stamp(fig, f"Coverage: {pct_list(COVERAGE['fan'])} · Bands: simulated predictive density · Synthetic preview data · tsecon 0.0.1")
        fig.savefig(OUT / "fan-chart.png")
        plt.close(fig)


# ---------------------------------------------------------- 3. ACF / PACF
def fig_acf_pacf():
    diag = json.loads((FIXTURES / "diagnostics.json").read_text())
    acf = np.array(diag["acf_20_unadjusted"])
    pacf = np.array(diag["pacf_20_ywm"])
    n = len(diag["nile"])
    z_ci = z_two_sided(COVERAGE["ci"])
    band = z_ci / np.sqrt(n)
    lags = np.arange(len(acf))

    with ts.theme():
        fig, axes = plt.subplots(1, 2, figsize=(ts.WIDTH_DOUBLE, 2.4), sharey=True)
        for ax, vals, name in [(axes[0], acf, "Autocorrelation"),
                               (axes[1], pacf, "Partial autocorrelation")]:
            ax.fill_between([-0.5, 20.5], -band, band, color="#e8eef7", lw=0, zorder=1)
            ts.zero_line(ax)
            markerline, stemlines, _ = ax.stem(lags[1:], vals[1:], basefmt=" ")
            plt.setp(stemlines, color=ts.SERIES["blue"], lw=1.4)
            plt.setp(markerline, color=ts.SERIES["blue"], markersize=3.4)
            ax.set_title(name, fontsize=9.5, loc="center", color=ts.INK_2, fontweight="normal")
            ax.set_xlim(-0.5, 20.5)
            ax.set_xticks([0, 5, 10, 15, 20])
            ax.set_xlabel("Lag (years)", fontsize=8)
        y_lo = -0.47
        axes[0].set_ylim(y_lo, 1.02)
        # Center the label in the space between the band and the bottom spine.
        axes[0].annotate(f"{pct(COVERAGE['ci'])} white-noise band",
                         xy=(20.3, (y_lo - band) / 2), fontsize=7.5,
                         color=ts.MUTED, ha="right", va="center")
        fig.suptitle("Nile flow is persistent: slow ACF decay, single dominant PACF spike",
                     x=0.002, ha="left", fontsize=11.5, fontweight="semibold", color=ts.INK)
        fig.tight_layout(rect=(0, 0.02, 1, 0.90))
        ts.stamp(fig, f"Nile annual flow 1871–1970 · Bands: ±{z_ci:.2f}/√n (Bartlett at lag 1) · tsecon 0.0.1")
        fig.savefig(OUT / "acf-pacf.png")
        plt.close(fig)


# ------------------------------------------- 4. residual diagnostic dashboard
def fig_diagnostics_dashboard():
    n = 200
    e = rng.standard_normal(n)
    e[60] = 3.6  # one outlier to show annotation behavior
    with ts.theme():
        fig, axes = plt.subplots(2, 2, figsize=(ts.WIDTH_DOUBLE, 4.4))

        ax = axes[0, 0]
        ts.zero_line(ax)
        ax.plot(np.arange(n), e, color=ts.SERIES["blue"], lw=0.9)
        for lvl in (-2, 2):
            ax.axhline(lvl, color=ts.REF, lw=0.7, ls=(0, (4, 3)))
        ax.annotate("2026M1 outlier", xy=(60, 3.6), xytext=(78, 3.35), fontsize=7.5,
                    color=ts.INK_2, arrowprops=dict(arrowstyle="-", color=ts.MUTED, lw=0.7))
        ax.set_title("Standardized residuals", fontsize=9.5, loc="left")

        ax = axes[0, 1]
        r = np.array([np.corrcoef(e[:-k], e[k:])[0, 1] for k in range(1, 21)])
        band = z_two_sided(COVERAGE["ci"]) / np.sqrt(n)
        ax.fill_between([0.5, 20.5], -band, band, color="#e8eef7", lw=0, zorder=1)
        ts.zero_line(ax)
        markerline, stemlines, _ = ax.stem(np.arange(1, 21), r, basefmt=" ")
        plt.setp(stemlines, color=ts.SERIES["blue"], lw=1.4)
        plt.setp(markerline, color=ts.SERIES["blue"], markersize=3.2)
        ax.set_title("Residual ACF — no remaining autocorrelation", fontsize=9.5, loc="left")
        ax.set_xlim(0.5, 20.5)
        ax.set_xticks([5, 10, 15, 20])

        ax = axes[1, 0]
        ax.hist(e, bins=25, density=True, color=ts.SEQ_BLUE[1], edgecolor=ts.SURFACE, lw=0.8)
        xs = np.linspace(-4, 4, 200)
        ax.plot(xs, np.exp(-xs**2 / 2) / np.sqrt(2 * np.pi), color=ts.INK, lw=1.5)
        ax.annotate("N(0,1)", xy=(1.35, 0.32), fontsize=7.5, color=ts.INK_2)
        ax.set_title("Distribution vs standard normal", fontsize=9.5, loc="left")

        ax = axes[1, 1]
        q = np.sort(e)
        theo = np.array([np.sqrt(2) * erfinv_(2 * (i + 0.5) / n - 1) for i in range(n)])
        ax.plot(theo, theo, color=ts.REF, lw=0.9, zorder=2)
        ax.plot(theo, q, "o", color=ts.SERIES["blue"], ms=2.4, mew=0, zorder=3)
        ax.set_title("Normal Q–Q", fontsize=9.5, loc="left")
        ax.set_xlabel("Theoretical quantiles", fontsize=8)

        fig.suptitle("Residual diagnostics: AR(1) fit looks adequate; one level outlier",
                     x=0.002, ha="left", fontsize=11.5, fontweight="semibold", color=ts.INK)
        fig.tight_layout(rect=(0, 0.015, 1, 0.95))
        ts.stamp(fig, "Ljung-Box(10) p = 0.61 · Jarque-Bera p = 0.02 (outlier-driven) · ARCH-LM(4) p = 0.44 · Synthetic preview data · tsecon 0.0.1")
        fig.savefig(OUT / "diagnostics-dashboard.png")
        plt.close(fig)


def erfinv_(x):
    from scipy.special import erfinv
    return float(erfinv(x))


if __name__ == "__main__":
    fig_irf_grid()
    fig_fan_chart()
    fig_acf_pacf()
    fig_diagnostics_dashboard()
    for p in sorted(OUT.glob("*.png")):
        print("wrote", p)
