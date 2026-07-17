"""The tsecon gallery, structural & shrinkage wing: identification under
weak assumptions, and regularization that selects.

Run with the project venv (tsecon + matplotlib installed there):
    .venv/bin/python docs/examples/showcase_structural.py

Figures land in docs/examples/img/ in the house style (Module 13).
Each section mirrors a subsection of the "Structural & shrinkage" block in
docs/examples/README.md.
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

REPO = Path(__file__).parents[2]
sys.path.insert(0, str(REPO / "prototypes" / "viz"))
import tsecon_style as ts  # noqa: E402
import tsecon  # noqa: E402

IMG = Path(__file__).parent / "img"
IMG.mkdir(exist_ok=True)
rng = np.random.default_rng(20260716)

Z90 = 1.6449  # two-sided 90% normal multiplier


def save(fig, name):
    fig.savefig(IMG / name)
    plt.close(fig)
    print("wrote", IMG / name)


# ------------------------------------------------------------------
# S1. Sign-restricted SVAR: the identified set vs posterior uncertainty
# ------------------------------------------------------------------
def section_sign_svar():
    data = np.array(json.loads((REPO / "fixtures" / "bvar_niw.json").read_text())["data"])
    names = ["GDP growth", "Consumption", "Investment"]
    H = 12
    # A demand shock (shock 0) that raises all three variables on impact.
    restr = [(0, 0, 0, "+"), (1, 0, 0, "+"), (2, 0, 0, "+")]
    res = tsecon.sign_restricted_svar(data, restrictions=restr, lags=2, horizon=H,
                                      n_draws=800, seed=20260716)
    q = np.asarray(res["quantiles"])          # (H+1, var, shock, prob)  probs 5/16/50/84/95
    set_min = np.asarray(res["set_min"])       # (H+1, var, shock)
    set_max = np.asarray(res["set_max"])
    acc = res["diagnostics"]["acceptance_rate"]

    t = np.arange(H + 1)
    # Colour ladder: lightest = widest (the identified set), darkening inward.
    C_SET, C_90, C_68, C_MED = ts.SEQ_BLUE[0], ts.SEQ_BLUE[2], ts.SEQ_BLUE[4], ts.SEQ_BLUE[6]

    with ts.theme():
        fig, axes = plt.subplots(1, 3, figsize=(ts.WIDTH_DOUBLE, 2.9))  # not sharey: scales differ
        for i, ax in enumerate(axes):
            ts.zero_line(ax)
            ax.fill_between(t, set_min[:, i, 0], set_max[:, i, 0], color=C_SET, lw=0, zorder=2)
            ax.fill_between(t, q[:, i, 0, 0], q[:, i, 0, 4], color=C_90, lw=0, zorder=3)
            ax.fill_between(t, q[:, i, 0, 1], q[:, i, 0, 3], color=C_68, lw=0, zorder=4)
            ax.plot(t, q[:, i, 0, 2], color=C_MED, lw=1.8, zorder=5)
            ax.set_title(names[i], fontsize=9.5, loc="center", color=ts.INK_2,
                         fontweight="normal")
            ax.set_xlim(0, H)
            ax.set_xticks([0, 4, 8, 12])
            ax.set_xlabel("Horizon (quarters)", fontsize=8)
            ax.tick_params(labelsize=7.5)
            lo, hi = float(set_min[:, i, 0].min()), float(set_max[:, i, 0].max())
            ax.set_ylim(lo - 0.10 * (hi - lo), hi + 0.10 * (hi - lo))
        axes[0].set_ylabel("Response to demand shock", fontsize=8.5, color=ts.INK)

        handles = [
            Patch(facecolor=C_SET, label="identified set (model ambiguity)"),
            Patch(facecolor=C_90, label="90% posterior band"),
            Patch(facecolor=C_68, label="68% posterior band"),
            Line2D([0], [0], color=C_MED, lw=1.8, label="posterior median"),
        ]
        # One shared horizontal legend below the panels — never over the data.
        fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
                   fontsize=7.5, handlelength=1.4, columnspacing=1.6,
                   handletextpad=0.5, bbox_to_anchor=(0.5, 0.015))
        fig.suptitle("A demand shock lifts all three — but the identified set dwarfs "
                     "posterior uncertainty",
                     x=0.005, ha="left", fontsize=11.5, fontweight="semibold", color=ts.INK)
        # Reserve the top strip for the title and the bottom strip for the legend.
        fig.tight_layout(rect=(0, 0.10, 1, 0.90))
        ts.stamp(fig, "Sign-restricted Bayesian SVAR(2) on US GDP/consumption/investment growth · "
                      "tsecon.sign_restricted_svar (800 draws, seed 20260716, "
                      f"{acc:.0%} rotation acceptance) · outer band = identified set (Haar prior over "
                      "rotations shapes its interior; Baumeister-Hamilton 2015 caveat), inner bands = posterior")
        save(fig, "struct-sign-svar.png")


# ------------------------------------------------------------------
# S2. Local projection: recovering a known IRF with honest bands
# ------------------------------------------------------------------
def section_lp():
    n, H = 480, 16
    phi1, phi2 = 1.1, -0.3
    # y is an AR(2) driven by an OBSERVED shock plus independent own-noise, so the
    # regressor block stays full rank (a pure AR(2) makes lag-augmentation singular);
    # the response of y to a unit observed shock is still the analytic psi-weights.
    shock = rng.standard_normal(n + 100)
    v = rng.standard_normal(n + 100) * 0.6
    y = np.zeros(n + 100)
    for tt in range(2, n + 100):
        y[tt] = phi1 * y[tt - 1] + phi2 * y[tt - 2] + shock[tt] + v[tt]
    y, shock = y[100:], shock[100:]

    psi = np.zeros(H + 1)
    psi[0] = 1.0
    for h in range(1, H + 1):
        psi[h] = phi1 * psi[h - 1] + (phi2 * psi[h - 2] if h >= 2 else 0.0)

    la = tsecon.lp(y, shock, horizons=H, se="lag_augmented")
    hc = tsecon.lp(y, shock, horizons=H, se="hac")
    irf_la, se_la = np.asarray(la["irf"]), np.asarray(la["se"])
    irf_hc, se_hc = np.asarray(hc["irf"]), np.asarray(hc["se"])
    hg = np.arange(H + 1)

    C_HAC = ts.SERIES["violet"]
    with ts.theme():
        fig, ax = plt.subplots(figsize=(ts.WIDTH_DOUBLE, 2.9))
        ts.zero_line(ax)
        # lag-augmented band: solid light fill (the library default, drawn prominent).
        ax.fill_between(hg, irf_la - Z90 * se_la, irf_la + Z90 * se_la,
                        color=ts.SEQ_BLUE[0], lw=0, zorder=2)
        # HAC band: hatched, no solid fill, so it reads distinctly through the overlap.
        ax.fill_between(hg, irf_hc - Z90 * se_hc, irf_hc + Z90 * se_hc,
                        facecolor="none", hatch="////", edgecolor=C_HAC, lw=0.0, zorder=3)
        ax.plot(hg, psi, color=ts.SERIES["red"], lw=1.3, ls=(0, (3, 2)), zorder=5)
        ax.plot(hg, irf_la, "o-", color=ts.SERIES["blue"], lw=1.7, ms=3.6, zorder=6)
        ax.plot(hg, irf_hc, color=C_HAC, lw=1.1, ls=(0, (1, 1.4)), zorder=6)
        ax.set_xlim(0, H)
        ax.set_xticks(range(0, H + 1, 4))
        ax.set_xlabel("Horizon", fontsize=8)
        ax.set_ylabel("Response of y to a unit shock")
        lo_y, hi_y = ax.get_ylim()
        ax.set_ylim(lo_y, hi_y + 0.16 * (hi_y - lo_y))  # headroom keeps the legend off the hump

        handles = [
            Line2D([0], [0], color=ts.SERIES["red"], lw=1.3, ls=(0, (3, 2)),
                   label="true IRF (analytic ψ-weights)"),
            Line2D([0], [0], color=ts.SERIES["blue"], lw=1.7, marker="o", ms=3.6,
                   label="lag-augmented IRF (default)"),
            Line2D([0], [0], color=C_HAC, lw=1.1, ls=(0, (1, 1.4)), label="HAC IRF"),
            Patch(facecolor=ts.SEQ_BLUE[0], label="lag-augmented 90% band"),
            Patch(facecolor="none", hatch="////", edgecolor=C_HAC, label="HAC 90% band"),
        ]
        ax.legend(handles=handles, loc="upper right", fontsize=7.3, handlelength=1.6,
                  labelspacing=0.35, borderaxespad=0.5)
        ax.set_title("Local projection recovers the true impulse response; "
                     "lag-augmented is the honest default")
        fig.tight_layout()
        ts.stamp(fig, "AR(2), φ = (1.1, −0.3) + independent own-noise, shock observed, n = 480 · "
                      "tsecon.lp(se=\"lag_augmented\" default / \"hac\") · bands: ±1.6449 × se (90%) · "
                      "lag augmentation: Montiel Olea-Plagborg-Møller (2021)")
        save(fig, "struct-lp-vs-truth.png")


# ------------------------------------------------------------------
# S3. Penalized regression: the lasso path — shrinkage as selection
# ------------------------------------------------------------------
def section_lasso_path():
    n, p, n_grid = 140, 60, 60
    true_idx = np.array([2, 9, 20, 41, 53])
    true_beta = np.array([3.0, -2.2, 1.6, 2.6, -1.3])
    # A dedicated stream (same master seed) keeps this design fixed regardless of
    # which sections ran before it — the figure regenerates bit-for-bit either way.
    rng_l = np.random.default_rng(20260716)
    X = rng_l.standard_normal((n, p))
    Xs = (X - X.mean(0)) / X.std(0)          # standardize: put every coef on one scale
    beta = np.zeros(p)
    beta[true_idx] = true_beta
    y = Xs @ beta + rng_l.standard_normal(n) * 1.0
    y = y - y.mean()

    alpha_max = np.max(np.abs(Xs.T @ y)) / n  # smallest alpha that zeroes every coefficient
    alphas = np.logspace(np.log10(alpha_max * 0.008), np.log10(alpha_max * 1.05), n_grid)
    paths = np.array([tsecon.lasso(Xs, y, alpha=a)["coef"] for a in alphas])
    la = np.log(alphas)

    # "Sensible" alpha: the largest penalty that still keeps every true feature but
    # has pruned the noise down to the true support size (a parsimonious selection).
    active = (np.abs(paths) > 1e-6).sum(1)
    k = len(true_idx)
    sel_idx = int(np.argmax(active <= k))      # first grid point pruned to <= k features
    a_star = alphas[sel_idx]
    kept = np.where(np.abs(paths[sel_idx]) > 1e-6)[0]
    recovers = set(kept.tolist()) == set(true_idx.tolist())

    is_true = np.zeros(p, dtype=bool)
    is_true[true_idx] = True
    with ts.theme():
        fig, ax = plt.subplots(figsize=(ts.WIDTH_DOUBLE, 3.0))
        ts.zero_line(ax)
        for j in range(p):                     # noise underneath, muted and thin
            if not is_true[j]:
                ax.plot(la, paths[:, j], color=ts.MUTED, lw=0.7, alpha=0.55, zorder=2)
        for j in range(p):                     # signal on top, house blue
            if is_true[j]:
                ax.plot(la, paths[:, j], color=ts.SERIES["blue"], lw=1.6, zorder=4)
        ax.axvline(np.log(a_star), color=ts.REF, lw=0.9, ls=(0, (4, 2)), zorder=3)

        ax.set_xlim(la[0], la[-1] + 0.05 * (la[-1] - la[0]))
        lo_y, hi_y = ax.get_ylim()
        ax.set_ylim(lo_y, hi_y + 0.20 * (hi_y - lo_y))   # headroom for legend + marker note
        note = f"α = {a_star:.2f}: pruned to {int(active[sel_idx])} features"
        if recovers:
            note += "\n(exactly the true support)"
        ax.annotate(note, xy=(np.log(a_star), hi_y + 0.16 * (hi_y - lo_y)),
                    xytext=(np.log(a_star) - 0.12, hi_y + 0.16 * (hi_y - lo_y)),
                    fontsize=7.3, color=ts.INK_2, ha="right", va="top")
        ax.set_xlabel("log penalty  log(α)   (left: weaker · right: stronger)", fontsize=8)
        ax.set_ylabel("Lasso coefficient")

        handles = [
            Line2D([0], [0], color=ts.SERIES["blue"], lw=1.6,
                   label=f"true nonzero coefficients ({k})"),
            Line2D([0], [0], color=ts.MUTED, lw=0.9, label=f"noise features (β = 0, {p - k})"),
        ]
        ax.legend(handles=handles, loc="upper right", fontsize=7.5, handlelength=1.7,
                  borderaxespad=0.6)
        ax.set_title("The lasso path: the penalty zeroes the noise first, the signal last — "
                     "sparsity as selection")
        fig.tight_layout()
        ts.stamp(fig, f"Synthetic design, n = {n}, p = {p}, {k} true nonzeros among noise · "
                      "tsecon.lasso across a 60-point α grid (standardized columns, y demeaned) · "
                      "dashed marker: the α that prunes to the true support")
        save(fig, "struct-lasso-path.png")


ALL = [section_sign_svar, section_lp, section_lasso_path]

if __name__ == "__main__":
    only = sys.argv[1:] if len(sys.argv) > 1 else None
    for fn in ALL:
        if only and not any(k in fn.__name__ for k in only):
            continue
        fn()
