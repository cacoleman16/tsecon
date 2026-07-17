"""The tsecon gallery, advanced wing: methods composed from the primitives.

Run with the project venv (tsecon + matplotlib installed there):
    .venv/bin/python docs/examples/showcase_advanced.py

Figures land in docs/examples/img/ in the house style (Module 13).
Each section mirrors a subsection of the "Advanced" block in
docs/examples/README.md.
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

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
# A1. Local projections: Jordà (2005) from tsecon primitives
# ------------------------------------------------------------------
def section_local_projection():
    n, H, K = 480, 12, 4
    phi1, phi2 = 1.3, -0.4
    eps = rng.standard_normal(n + 100)
    y = np.zeros(n + 100)
    for t in range(2, n + 100):
        y[t] = phi1 * y[t - 1] + phi2 * y[t - 2] + eps[t]
    y, eps = y[100:], eps[100:]

    # The true IRF, analytically: psi-weights of the AR(2).
    psi = np.zeros(H + 1)
    psi[0] = 1.0
    for h in range(1, H + 1):
        psi[h] = phi1 * psi[h - 1] + (phi2 * psi[h - 2] if h >= 2 else 0.0)

    # LP at each horizon: y_{t+h} on the shock + 4 lags of y, HAC errors
    # (the h-step regression error is MA(h) by construction).
    b, s = np.zeros(H + 1), np.zeros(H + 1)
    for h in range(H + 1):
        t_idx = np.arange(K, n - h)
        Y = y[t_idx + h]
        X = np.column_stack([np.ones(len(t_idx)), eps[t_idx]]
                            + [y[t_idx - lag] for lag in range(1, K + 1)])
        r = tsecon.ols(Y, X, se_type="hac", maxlags=h + 1)
        b[h], s[h] = r["params"][1], r["bse"][1]

    with ts.theme():
        fig, ax = plt.subplots(figsize=(ts.WIDTH_DOUBLE, 2.8))
        hgrid = np.arange(H + 1)
        ts.zero_line(ax)
        ax.fill_between(hgrid, b - Z90 * s, b + Z90 * s, color=ts.SEQ_BLUE[0], lw=0, zorder=2)
        ax.plot(hgrid, b, "o-", color=ts.SERIES["blue"], lw=1.7, ms=4, zorder=4,
                label="local projection (point estimate)")
        ax.plot(hgrid, psi, color=ts.SERIES["red"], lw=1.3, ls=(0, (3, 2)), zorder=3,
                label="true IRF (analytic ψ-weights)")
        ax.annotate("90% HAC band", xy=(1.4, b[1] + Z90 * s[1] + 0.06),
                    fontsize=7.5, color=ts.INK_2)
        ax.legend(loc="upper right", fontsize=7.5)
        ax.set_xticks(range(0, H + 1, 2))
        ax.set_xlabel("Horizon", fontsize=8)
        ax.set_ylabel("Response of y to a unit shock")
        ax.set_title("Local projections recover the true impulse response, "
                     "one regression per horizon")
        fig.tight_layout()
        ts.stamp(fig, "AR(2), φ = (1.3, −0.4), n = 480, shock observed · one tsecon.ols(se_type=\"hac\", "
                      "maxlags=h+1) per horizon, 4 lag controls · bands: ±1.645 × HAC se · "
                      "the dedicated LP module (roadmap 07) adds lag augmentation and sup-t bands")
        save(fig, "adv-local-projection.png")


# ------------------------------------------------------------------
# A2. News impact curves: GARCH vs GJR (Engle-Ng 1993)
# ------------------------------------------------------------------
def section_news_impact():
    ret = np.array(json.loads((REPO / "fixtures" / "garch.json").read_text())["returns"])
    g = tsecon.garch_fit(ret, vol="garch", mean="zero", dist="normal")
    gj = tsecon.garch_fit(ret, vol="gjr", mean="zero", dist="normal")
    om, al, be = np.asarray(g["params"])
    omj, alj, gaj, bej = np.asarray(gj["params"])
    se_gam = np.asarray(gj["se_robust"])[2]

    # sigma^2(eps) holding lagged variance at its unconditional level.
    s2_g = om / (1 - al - be)
    s2_j = omj / (1 - alj - gaj / 2 - bej)   # E[eps^2 I(eps<0)] = sigma^2/2 under symmetry
    e = np.linspace(-5, 5, 401)
    nic_g = om + al * e**2 + be * s2_g
    nic_j = omj + (alj + gaj * (e < 0)) * e**2 + bej * s2_j

    with ts.theme():
        fig, ax = plt.subplots(figsize=(ts.WIDTH_DOUBLE, 2.9))
        ax.axvline(0.0, color=ts.REF, lw=0.9, zorder=1.5)
        ax.plot(e, nic_g, color=ts.SERIES["blue"], lw=1.8)
        ax.plot(e, nic_j, color=ts.SERIES["red"], lw=1.6, ls=(0, (4, 2)))
        ax.annotate("GJR(1,1,1):\nkinked at zero", xy=(5.15, nic_j[-1]), fontsize=7.5,
                    color=ts.SERIES["red"], va="center", ha="left")
        ax.annotate("GARCH(1,1):\nsymmetric parabola", xy=(5.15, nic_g[-1] - 0.06), fontsize=7.5,
                    color=ts.SERIES["blue"], va="top", ha="left")
        ax.annotate("bad news:\nslope α + γ", xy=(-3.9, 2.74), fontsize=7.5,
                    color=ts.SERIES["red"], ha="center", va="top")
        ax.annotate("good news:\nslope α", xy=(3.9, 2.74), fontsize=7.5,
                    color=ts.SERIES["red"], ha="center", va="top")
        ax.annotate(f"γ̂ = {gaj:.3f} ({se_gam:.3f}) — statistically zero.\n"
                    "These returns are simulated from a symmetric GARCH,\n"
                    "and the news impact curve shows it honestly; on equity\n"
                    "data γ > 0 tilts the red curve up for bad news.",
                    xy=(0.03, 0.95), xycoords="axes fraction", fontsize=7.5,
                    color=ts.INK_2, va="top")
        ax.set_xlim(-5.3, 8.4)
        ax.set_xticks([-4, -2, 0, 2, 4])
        ax.set_xlabel("Shock ε (today's return)", fontsize=8)
        ax.set_ylabel("Tomorrow's variance σ²(ε)")
        ax.set_title("News impact curves: what does today's shock do to tomorrow's variance?")
        fig.tight_layout()
        ts.stamp(fig, "tsecon.garch_fit(vol=\"garch\" / \"gjr\") on the fixture returns · "
                      "σ²(ε) evaluated at the unconditional lagged variance (Engle-Ng 1993) · "
                      "γ̂ reported with Bollerslev-Wooldridge robust se")
        save(fig, "adv-news-impact.png")


# ------------------------------------------------------------------
# A3. The volatility term structure: mean reversion made visible
# ------------------------------------------------------------------
def section_vol_term_structure():
    ret = np.array(json.loads((REPO / "fixtures" / "garch.json").read_text())["returns"])
    r = tsecon.garch_fit(ret, vol="garch", mean="zero", dist="normal", forecast_horizon=120)
    om, al, be = np.asarray(r["params"])
    s2_bar = om / (1 - al - be)
    pers = al + be
    half_life = np.log(0.5) / np.log(pers)

    h = np.arange(1, 121)
    vf = np.asarray(r["variance_forecast"])          # from the actual end-of-sample state

    # Closed-form GARCH forecast recursion from the FITTED params at three
    # hypothetical starting variances (the API forecasts only from the
    # end-of-sample state, so these are analytic what-ifs, same formula):
    # E[σ²_{t+h}] = σ̄² + (α+β)^{h-1} (σ²_{t+1} − σ̄²).
    scen = {c: s2_bar + pers ** (h - 1) * (c * s2_bar - s2_bar) for c in [0.5, 1.0, 3.0]}

    with ts.theme():
        fig, ax = plt.subplots(figsize=(ts.WIDTH_DOUBLE, 2.9))
        ax.axhline(np.sqrt(s2_bar), color=ts.REF, lw=0.9, zorder=1.5)
        ax.annotate("long-run volatility — start here, stay flat",
                    xy=(6, np.sqrt(s2_bar) - 0.035), fontsize=7.5, color=ts.MUTED, va="top")
        for c in [3.0, 1.0, 0.5]:
            ax.plot(h, np.sqrt(scen[c]), color=ts.MUTED, lw=1.1, ls=(0, (3, 2)), zorder=2)
        ax.annotate("what if vol started at 3× the long-run variance?",
                    xy=(9, np.sqrt(scen[3.0][8]) + 0.05), fontsize=7.5, color=ts.MUTED,
                    ha="left", va="bottom")
        ax.annotate("...at 0.5×? it climbs back up",
                    xy=(20, 1.185), fontsize=7.5, color=ts.MUTED, ha="left", va="top")
        ax.plot(h, np.sqrt(vf), color=ts.SERIES["blue"], lw=2.0, zorder=4)
        ax.annotate("forecast from the end-of-sample state", xy=(4, np.sqrt(vf[0]) + 0.03),
                    fontsize=7.5, color=ts.SERIES["blue"], va="bottom")
        ax.axvline(half_life, color=ts.REF, lw=0.9, zorder=1.5)
        ax.annotate(f"half-life ≈ {half_life:.0f} days", xy=(half_life + 2.5, 2.45),
                    fontsize=7.5, color=ts.MUTED, va="top")
        ax.set_xlim(0, 126)
        ax.set_xticks([1, 30, 60, 90, 120])
        ax.set_xlabel("Forecast horizon (days)", fontsize=8)
        ax.set_ylabel("Volatility  $\\sqrt{\\mathrm{E}[\\sigma^2_{t+h}]}$")
        ax.set_title("The volatility term structure: every state decays to the "
                     "long-run anchor at rate α + β")
        fig.tight_layout()
        ts.stamp(fig, f"tsecon.garch_fit(forecast_horizon=120) · persistence α + β = {pers:.4f} · "
                      "dashed what-if curves: the closed-form recursion "
                      "σ²(h) = σ̄² + (α+β)^(h−1) (σ²(1) − σ̄²) evaluated in numpy from the fitted (ω, α, β)")
        save(fig, "adv-vol-term-structure.png")


# ------------------------------------------------------------------
# A4. BVAR prior tightness in one picture: shrinkage vs evidence
# ------------------------------------------------------------------
def section_bvar_shrinkage():
    data = np.array(json.loads((REPO / "fixtures" / "bvar_niw.json").read_text())["data"])
    H, lambdas = 12, [0.05, 0.2, 1.0]
    fits = {}
    for lam in lambdas:
        draws = np.array(tsecon.bvar_irf_draws(data, lags=2, horizon=H, n_draws=500,
                                               seed=42, lambda1=lam))
        gg = draws[:, :, 0, 0]  # GDP-growth response to a GDP-growth shock
        q05, q50, q95 = np.quantile(gg, [0.05, 0.5, 0.95], axis=0)
        lml = tsecon.bvar_fit(data, lags=2, lambda1=lam)["log_marginal_likelihood"]
        fits[lam] = (q05, q50, q95, lml)
    best = max(lambdas, key=lambda l: fits[l][3])
    labels = {0.05: "tight", 0.2: "the evidence's choice", 1.0: "loose"}

    with ts.theme():
        fig, axes = plt.subplots(1, 3, figsize=(ts.WIDTH_DOUBLE, 2.5), sharey=True)
        t = np.arange(H + 1)
        for ax, lam in zip(axes, lambdas):
            q05, q50, q95, lml = fits[lam]
            primary = lam == best
            band = ts.SEQ_BLUE[0] if primary else ts.SHADE
            line = ts.SERIES["blue"] if primary else ts.MUTED
            ts.zero_line(ax)
            ax.fill_between(t, q05, q95, color=band, lw=0, zorder=2)
            ax.plot(t, q50, color=line, lw=1.9 if primary else 1.4, zorder=4)
            ax.set_title(f"$\\lambda_1$ = {lam:g} — {labels[lam]}", fontsize=8.5, loc="left",
                         color=ts.INK if primary else ts.INK_2)
            ax.annotate(f"log ML = {lml:.1f}" + ("\nevidence-preferred" if primary else ""),
                        xy=(0.96, 0.84), xycoords="axes fraction", fontsize=8, ha="right",
                        va="top", color=ts.SERIES["blue"] if primary else ts.MUTED,
                        fontweight="semibold" if primary else "normal")
            ax.set_xticks([0, 4, 8, 12])
            ax.set_xlabel("Horizon (quarters)", fontsize=8)
        axes[0].set_ylabel("GDP-growth response\nto a GDP-growth shock", fontsize=8)
        fig.suptitle("Minnesota prior tightness: too tight ignores the data, too loose "
                     "overfits — the marginal likelihood arbitrates",
                     x=0.002, ha="left", fontsize=11.5, fontweight="semibold", color=ts.INK)
        fig.tight_layout(rect=(0, 0.02, 1, 0.90))
        ts.stamp(fig, "tsecon.bvar_irf_draws (500 posterior draws, seed 42, horizon 12) · "
                      "bands: 90% credible ([0.05, 0.95] posterior quantiles) · "
                      "log marginal likelihood from tsecon.bvar_fit — evidence peaks at λ₁ = "
                      f"{best:g}")
        save(fig, "adv-bvar-shrinkage.png")


ALL = [section_local_projection, section_news_impact, section_vol_term_structure,
       section_bvar_shrinkage]

if __name__ == "__main__":
    only = sys.argv[1:] if len(sys.argv) > 1 else None
    for fn in ALL:
        if only and not any(k in fn.__name__ for k in only):
            continue
        fn()
