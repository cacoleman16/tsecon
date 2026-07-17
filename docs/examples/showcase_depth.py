"""The tsecon gallery, depth wing: the Phase-3/4 methods — score-driven
volatility, and (as they land) connectedness, factor-augmented VARs, the
term structure, and nowcasting.

Run with the project venv (tsecon + matplotlib installed there):
    .venv/bin/python docs/examples/showcase_depth.py

Figures land in docs/examples/img/ in the house style (Module 13).
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

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


# ------------------------------------------------------------------
# D2. Diebold-Yilmaz connectedness: who spills over to whom
# ------------------------------------------------------------------
def section_connectedness():
    rng = np.random.default_rng(11)
    k, n = 8, 500
    # Moderate own-persistence and a ring of cross-links in the dynamics, plus
    # a strong COMMON shock (correlated innovations) — so a large share of each
    # variable's forecast-error variance genuinely comes from the others.
    a = 0.25 * np.eye(k)
    for i in range(k):
        a[i, (i + 1) % k] = 0.15
        a[i, (i - 1) % k] = 0.10
    bload = rng.uniform(0.5, 1.0, k)  # loadings on one common shock
    y = np.zeros((n, k))
    for t in range(1, n):
        eps = bload * rng.standard_normal() + 0.7 * rng.standard_normal(k)
        y[t] = a @ y[t - 1] + eps
    res = tsecon.connectedness(y, lags=1, horizon=10)
    gfevd = 100.0 * np.array(res["gfevd"])  # rows sum to 100
    net = np.array(res["net"])
    names = [f"V{i+1}" for i in range(k)]

    with ts.theme():
        fig, (axh, axn) = plt.subplots(
            1, 2, figsize=(ts.WIDTH_DOUBLE, 3.3),
            gridspec_kw={"width_ratios": [1.35, 1.0]},
        )
        # Spillover heatmap: entry (i, j) = share of i's variance from j.
        im = axh.imshow(gfevd, cmap="Blues", vmin=0, vmax=gfevd.max(), aspect="auto")
        axh.set_xticks(range(k))
        axh.set_xticklabels(names, fontsize=7)
        axh.set_yticks(range(k))
        axh.set_yticklabels(names, fontsize=7)
        axh.set_xlabel("shock from", fontsize=8, color=ts.INK_2)
        axh.set_ylabel("variance of", fontsize=8, color=ts.INK_2)
        axh.set_title("Directional spillover table (%)", fontsize=9.5,
                      color=ts.INK_2, fontweight="normal")
        cb = fig.colorbar(im, ax=axh, fraction=0.045, pad=0.03)
        cb.ax.tick_params(labelsize=6.5)
        cb.outline.set_visible(False)

        # Net connectedness: transmitters (positive) vs receivers (negative).
        order = np.argsort(net)
        colors = [ts.SERIES["red"] if v >= 0 else ts.SERIES["blue"] for v in net[order]]
        axn.barh(range(k), net[order], color=colors, height=0.66)
        axn.axvline(0, color=ts.REF, lw=0.9)
        axn.set_yticks(range(k))
        axn.set_yticklabels([names[i] for i in order], fontsize=7)
        axn.set_xlabel("net connectedness (to − from, %)", fontsize=8, color=ts.INK_2)
        axn.set_title("Net transmitters vs receivers", fontsize=9.5,
                      color=ts.INK_2, fontweight="normal")
        axn.tick_params(labelsize=7.5)

        fig.suptitle(f"Total connectedness {res['total']:.0f}% — a tightly linked system",
                     x=0.005, ha="left", fontsize=11.5, fontweight="semibold", color=ts.INK)
        fig.tight_layout(rect=(0, 0.0, 1, 0.92))
        ts.stamp(fig, "Diebold-Yilmaz connectedness · tsecon.connectedness · row-normalized "
                      "generalized FEVD of a VAR(1) at horizon 10 · left: what share of each "
                      "variable's forecast-error variance comes from each shock (rows sum to "
                      "100%); right: net = what a variable sends minus what it receives")
        save(fig, "depth-connectedness.png")


# ------------------------------------------------------------------
# D3. FAVAR: one policy shock, many responses
# ------------------------------------------------------------------
def section_favar():
    rng = np.random.default_rng(23)
    n, big_n, r = 260, 24, 2
    f = np.zeros((n, r))
    for t in range(1, n):
        f[t] = np.array([0.7, 0.5]) * f[t - 1] + rng.standard_normal(r)
    load = rng.standard_normal((big_n, r))
    x = f @ load.T + 0.5 * rng.standard_normal((n, big_n))
    policy = 0.6 * f[:, 0] + 0.3 * rng.standard_normal(n)
    res = tsecon.favar(x, policy, n_factors=r, lags=2, horizon=24)
    irf_panel = np.array(res["irf_panel"])  # big_n x (H+1)
    irf_policy = np.array(res["irf_policy"])
    h = np.arange(irf_panel.shape[1])

    with ts.theme():
        fig, ax = plt.subplots(figsize=(ts.WIDTH_DOUBLE, 3.1))
        ts.zero_line(ax)
        # Every information-panel series' response, faint.
        for row in irf_panel:
            ax.plot(h, row, color=ts.SERIES["blue"], lw=0.5, alpha=0.28, zorder=2)
        # Two illustrative series highlighted, and the policy variable itself.
        strongest = int(np.argmax(np.abs(irf_panel[:, 1])))
        weakest = int(np.argmin(np.abs(irf_panel).sum(axis=1)))
        ax.plot(h, irf_panel[strongest], color=ts.SERIES["violet"], lw=1.6,
                zorder=4, label=f"series {strongest+1} (strong loader)")
        ax.plot(h, irf_panel[weakest], color=ts.SERIES["aqua"], lw=1.6,
                zorder=4, label=f"series {weakest+1} (weak loader)")
        ax.plot(h, irf_policy, color=ts.INK, lw=2.0, zorder=5, label="policy variable")
        ax.set_xlim(0, h[-1])
        ax.set_xlabel("Horizon", fontsize=8.5, color=ts.INK_2)
        ax.set_ylabel("Response to the policy shock", fontsize=8.5, color=ts.INK)
        ax.tick_params(labelsize=7.5)
        ax.legend(loc="upper right", fontsize=7.5, frameon=False, handlelength=1.6,
                  labelspacing=0.3)
        fig.suptitle("One factor-identified policy shock, twenty-four responses",
                     x=0.005, ha="left", fontsize=11.5, fontweight="semibold", color=ts.INK)
        fig.tight_layout(rect=(0, 0.0, 1, 0.92))
        ts.stamp(fig, "Factor-augmented VAR · tsecon.favar (Bernanke-Boivin-Eliasz 2005) · a "
                      "recursive shock to the policy variable, propagated through the factor VAR "
                      "and mapped back onto every series in the 24-variable information panel "
                      "through the estimated loadings (faint lines); two series highlighted")
        save(fig, "depth-favar.png")


# ------------------------------------------------------------------
# D4. Dynamic Nelson-Siegel: level, slope, curvature over time
# ------------------------------------------------------------------
def section_term_structure():
    fx = json.loads((REPO / "fixtures" / "termstructure.json").read_text())
    panel = np.array(fx["yields_panel"])   # n_dates x n_maturities
    mats = np.array(fx["maturities"])
    lam = fx["lambda"]
    res = tsecon.dynamic_ns(panel, mats, decay=lam)
    level = np.array(res["level"])
    slope = np.array(res["slope"])
    curv = np.array(res["curvature"])
    t = np.arange(len(level))

    date = 100
    fit = tsecon.nelson_siegel(mats, panel[date], decay=lam)
    fitted = panel[date] - np.array(fit["residuals"])

    with ts.theme():
        fig, (axf, axc) = plt.subplots(
            1, 2, figsize=(ts.WIDTH_DOUBLE, 3.0),
            gridspec_kw={"width_ratios": [1.5, 1.0]},
        )
        # Left: the three factor series.
        axf.plot(t, level, color=ts.SERIES["blue"], lw=1.4, label="level")
        axf.plot(t, slope, color=ts.SERIES["red"], lw=1.4, label="slope")
        axf.plot(t, curv, color=ts.SERIES["yellow"], lw=1.4, label="curvature")
        axf.axvline(date, color=ts.REF, lw=0.8, ls=(0, (2, 2)))
        axf.set_xlim(0, len(level) - 1)
        axf.set_xlabel("Month", fontsize=8.5, color=ts.INK_2)
        axf.set_ylabel("Factor", fontsize=8.5, color=ts.INK)
        axf.tick_params(labelsize=7.5)
        # Parked in the empty band between the level (~5) and the slope /
        # curvature (~0) clusters, clear of every line.
        axf.legend(loc="center", bbox_to_anchor=(0.5, 0.66), fontsize=7.5,
                   frameon=False, ncol=3, handlelength=1.3, columnspacing=1.0)
        axf.set_title("The three factors through time", fontsize=9.5,
                      color=ts.INK_2, fontweight="normal")

        # Right: observed vs fitted curve on the marked date.
        axc.plot(mats, panel[date], "o", color=ts.INK_2, ms=4, label="observed")
        axc.plot(mats, fitted, color=ts.SERIES["blue"], lw=1.8, label="Nelson-Siegel fit")
        axc.set_xlabel("Maturity (months)", fontsize=8.5, color=ts.INK_2)
        axc.set_ylabel("Yield (%)", fontsize=8.5, color=ts.INK)
        axc.tick_params(labelsize=7.5)
        axc.legend(loc="lower right", fontsize=7.5, frameon=False, handlelength=1.4)
        axc.set_title(f"Fit on month {date}  ($R^2$ = {fit['rsquared']:.3f})",
                      fontsize=9.5, color=ts.INK_2, fontweight="normal")

        fig.suptitle("The yield curve in three numbers — level, slope, curvature",
                     x=0.005, ha="left", fontsize=11.5, fontweight="semibold", color=ts.INK)
        fig.tight_layout(rect=(0, 0.0, 1, 0.92))
        ts.stamp(fig, "Dynamic Nelson-Siegel · tsecon.dynamic_ns / tsecon.nelson_siegel "
                      "(Diebold-Li 2006) · each date's yield curve is compressed to a level, "
                      "slope, and curvature factor at the fixed decay λ = 0.0609; the right "
                      "panel shows the three-factor fit reconstructing a single date's curve")
        save(fig, "depth-term-structure.png")


# ------------------------------------------------------------------
# D5. Realized volatility: separating the jumps from the diffusion
# ------------------------------------------------------------------
def section_realized():
    rng = np.random.default_rng(31)
    n_days, m = 250, 79
    iv = np.empty(n_days)
    iv[0] = 1.0
    for d in range(1, n_days):
        iv[d] = 0.03 + 0.94 * iv[d - 1] + 0.10 * rng.standard_normal() ** 2
    rv = np.empty(n_days)
    bv = np.empty(n_days)
    ratio = np.empty(n_days)
    jump_flag = np.zeros(n_days, dtype=bool)
    intraday = np.sqrt(iv[:, None] / m) * rng.standard_normal((n_days, m))
    jday = rng.random(n_days) < 0.06
    intraday[jday, 0] += rng.standard_normal(jday.sum()) * 1.4
    for d in range(n_days):
        r = intraday[d]
        rv[d] = tsecon.realized_measures(r)["rv"]
        bv[d] = tsecon.realized_measures(r)["bipower"]
        ratio[d] = tsecon.bns_jump_test(r)["ratio"]
    jump_flag = ratio > 1.96  # ~5% BNS jump test
    t = np.arange(n_days)

    with ts.theme():
        fig, ax = plt.subplots(figsize=(ts.WIDTH_DOUBLE, 3.0))
        # Total realized variance vs the jump-robust continuous part.
        ax.fill_between(t, bv, rv, where=rv >= bv, color=ts.SERIES["red"], alpha=0.25,
                        lw=0, zorder=2, label="jump contribution")
        ax.plot(t, rv, color=ts.INK_2, lw=0.9, zorder=3, label="realized variance")
        ax.plot(t, bv, color=ts.SERIES["blue"], lw=1.2, zorder=4,
                label="bipower (continuous)")
        # Flag the days the BNS test calls a jump.
        ax.plot(t[jump_flag], rv[jump_flag], "v", color=ts.SERIES["red"], ms=5,
                zorder=5, markeredgecolor=ts.SURFACE, markeredgewidth=0.5,
                label="BNS jump day")
        ax.set_xlim(0, n_days - 1)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Day", fontsize=8.5, color=ts.INK_2)
        ax.set_ylabel("Daily variance", fontsize=8.5, color=ts.INK)
        ax.tick_params(labelsize=7.5)
        ax.legend(loc="upper left", fontsize=7.5, frameon=False, handlelength=1.4,
                  labelspacing=0.3)
        fig.suptitle("Realized variance splits into a smooth part and jumps",
                     x=0.005, ha="left", fontsize=11.5, fontweight="semibold", color=ts.INK)
        fig.tight_layout(rect=(0, 0.0, 1, 0.92))
        ts.stamp(fig, "Realized measures · tsecon.realized_measures / tsecon.bns_jump_test "
                      "(Barndorff-Nielsen & Shephard 2004; Huang-Tauchen 2005) · bipower "
                      "variation is robust to jumps, so the gap between realized variance and "
                      "bipower is the jump contribution; markers are days the ratio test flags")
        save(fig, "depth-realized.png")


# ------------------------------------------------------------------
# D6. Mean group vs CCE: the common-factor bias, and its cure
# ------------------------------------------------------------------
def section_panel():
    fx = json.loads((REPO / "fixtures" / "tsecon-panelts.json").read_text())
    ys = [np.array(u) for u in fx["y"]]
    x = fx["x"]
    nunits = fx["design"]["N"]
    xs = [np.column_stack([x[0][i], x[1][i]]) for i in range(nunits)]
    mg = tsecon.panel_mean_group(ys, xs, method="mg")
    cce = tsecon.panel_mean_group(ys, xs, method="cce")
    theta0 = fx["true_mean_slopes"][0]        # first slope's truth
    per_unit_mg = np.array(mg["coef_per_unit"])[:, 0]
    per_unit_cce = np.array(cce["coef_per_unit"])[:, 0]

    with ts.theme():
        fig, ax = plt.subplots(figsize=(ts.WIDTH_DOUBLE, 3.0))
        rng = np.random.default_rng(5)
        jit = rng.uniform(-0.12, 0.12, nunits)
        ax.scatter(per_unit_mg, np.zeros(nunits) + 1 + jit, s=18,
                   color=ts.SERIES["blue"], alpha=0.6, zorder=3, label="per-unit estimate")
        ax.scatter(per_unit_cce, np.zeros(nunits) + 0 + jit, s=18,
                   color=ts.SERIES["aqua"], alpha=0.6, zorder=3)
        ax.axvline(theta0, color=ts.REF, lw=1.2, ls=(0, (3, 2)), zorder=2)
        ax.annotate("true mean slope", xy=(theta0, 1.62), ha="center", va="bottom",
                    fontsize=7.5, color=ts.INK_2)
        # Group means with their SE whiskers.
        ax.errorbar(mg["coef"][0], 1, xerr=1.96 * mg["se"][0], fmt="D",
                    color=ts.SERIES["blue"], ms=6, capsize=3, zorder=5)
        ax.errorbar(cce["coef"][0], 0, xerr=1.96 * cce["se"][0], fmt="D",
                    color=ts.SERIES["aqua"], ms=6, capsize=3, zorder=5)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["CCE-MG", "Mean group"], fontsize=9)
        ax.set_ylim(-0.6, 1.9)
        ax.set_xlabel("Slope on the first regressor", fontsize=8.5, color=ts.INK)
        ax.tick_params(labelsize=7.5)
        fig.suptitle("A common factor biases mean group; CCE-MG corrects it",
                     x=0.005, ha="left", fontsize=11.5, fontweight="semibold", color=ts.INK)
        fig.tight_layout(rect=(0, 0.0, 1, 0.92))
        ts.stamp(fig, "Heterogeneous panel · tsecon.panel_mean_group (Pesaran-Smith 1995 MG; "
                      "Pesaran 2006 CCE-MG) · dots are the per-unit slope estimates, diamonds "
                      "the group averages with 95% bands; plain MG is pulled off the true mean "
                      "slope by the unobserved common factor, while CCE-MG sits on it")
        save(fig, "depth-panel-mg-cce.png")


ALL = [
    section_gas,
    section_connectedness,
    section_favar,
    section_term_structure,
    section_realized,
    section_panel,
]

if __name__ == "__main__":
    only = sys.argv[1:] if len(sys.argv) > 1 else None
    for fn in ALL:
        if only and not any(k in fn.__name__ for k in only):
            continue
        fn()
