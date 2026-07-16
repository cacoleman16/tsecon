"""The tsecon gallery: every working method on synthetic data, with figures.

Run with the project venv (tsecon + matplotlib installed there):
    .venv/bin/python docs/examples/showcase.py

Figures land in docs/examples/img/ in the house style (Module 13).
Each section mirrors a section of docs/examples/README.md.
"""
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


def save(fig, name):
    fig.savefig(IMG / name)
    plt.close(fig)
    print("wrote", IMG / name)


# ------------------------------------------------------------------
# 1. Exploration: ACF/PACF identify the order of a process
# ------------------------------------------------------------------
def section_acf():
    n = 400
    e = rng.standard_normal(n + 50)
    y = np.empty(n + 50)
    y[:2] = 0
    for t in range(2, n + 50):
        y[t] = 1.3 * y[t - 1] - 0.4 * y[t - 2] + e[t]
    y = y[50:]

    r = tsecon.acf(y, nlags=24)
    p = tsecon.pacf(y, nlags=24)
    band = 1.96 / np.sqrt(n)

    with ts.theme():
        fig, axes = plt.subplots(1, 2, figsize=(ts.WIDTH_DOUBLE, 2.4), sharey=True)
        for ax, vals, name in [(axes[0], r["acf"], "Autocorrelation"),
                               (axes[1], p, "Partial autocorrelation")]:
            ax.fill_between([-0.5, 24.5], -band, band, color="#e8eef7", lw=0, zorder=1)
            ts.zero_line(ax)
            mk, st, _ = ax.stem(np.arange(1, len(vals)), np.asarray(vals)[1:], basefmt=" ")
            plt.setp(st, color=ts.SERIES["blue"], lw=1.4)
            plt.setp(mk, color=ts.SERIES["blue"], markersize=3.2)
            ax.set_title(name, fontsize=9.5, loc="center", color=ts.INK_2, fontweight="normal")
            ax.set_xticks([0, 6, 12, 18, 24])
            ax.set_xlabel("Lag", fontsize=8)
        fig.suptitle("An AR(2): geometric ACF decay, PACF cuts off after lag 2",
                     x=0.002, ha="left", fontsize=11.5, fontweight="semibold", color=ts.INK)
        fig.tight_layout(rect=(0, 0.02, 1, 0.90))
        ts.stamp(fig, "Synthetic AR(2), phi = (1.3, -0.4), n = 400 · tsecon.acf / tsecon.pacf")
        save(fig, "01-acf-pacf.png")


# ------------------------------------------------------------------
# 2. The stationarity workflow: three series, three verdicts
# ------------------------------------------------------------------
def section_stationarity():
    n = 300
    stationary = np.empty(n)
    stationary[0] = 0.0
    e = rng.standard_normal(n)
    for t in range(1, n):
        stationary[t] = 0.6 * stationary[t - 1] + e[t]
    random_walk = np.cumsum(0.08 + rng.standard_normal(n))
    differenced = np.diff(random_walk)

    series = [("Stationary AR(1)", stationary), ("Random walk with drift", random_walk),
              ("...the same walk, differenced", differenced)]
    with ts.theme():
        fig, axes = plt.subplots(1, 3, figsize=(ts.WIDTH_DOUBLE, 2.3))
        for ax, (name, y) in zip(axes, series):
            rep = tsecon.check_stationarity(y)
            ax.plot(y, color=ts.SERIES["blue"], lw=1.0)
            ax.set_title(name, fontsize=9.5, loc="left")
            verdict = rep["quadrant"]
            rec = rep["recommendation"]
            color = ts.SERIES["green"] if rec == "Proceed" else ts.SERIES["red"]
            ax.annotate(f"{verdict}: {rec}", xy=(0.02, 0.03), xycoords="axes fraction",
                        fontsize=8, color=color, fontweight="semibold")
            ax.annotate(f"ADF p = {rep['adf_p_value']:.3f}\nKPSS p = {rep['kpss_p_value']:.3f}",
                        xy=(0.02, 0.83), xycoords="axes fraction", fontsize=7, color=ts.MUTED)
        fig.suptitle("check_stationarity(): the ADF + KPSS confirmatory quadrant, decided for you",
                     x=0.002, ha="left", fontsize=11.5, fontweight="semibold", color=ts.INK)
        fig.tight_layout(rect=(0, 0.02, 1, 0.90))
        ts.stamp(fig, "ADF (MacKinnon p-values) + KPSS (auto bandwidth), alpha = 0.05 · tsecon.check_stationarity")
        save(fig, "02-stationarity.png")


# ------------------------------------------------------------------
# 3. Robust standard errors: why HAC matters with autocorrelated errors
# ------------------------------------------------------------------
def section_hac():
    n = 200
    n_mc = 3000
    beta_true = 0.5

    def draw(local_rng):
        x = np.empty(n); u = np.empty(n)
        x[0] = u[0] = 0.0
        e = local_rng.standard_normal((2, n))
        for t in range(1, n):
            x[t] = 0.7 * x[t - 1] + e[0, t]
            u[t] = 0.7 * u[t - 1] + e[1, t]
        y = 1.0 + beta_true * x + u
        return y, x

    # A representative draw for the interval panel: the naive CI excludes the
    # truth, the HAC CI contains it (the textbook failure mode, made visible).
    results = None
    for seed in range(200):
        y, x = draw(np.random.default_rng(seed))
        X = np.column_stack([np.ones(n), x])
        cand = {se: (lambda r: (r["params"][1], r["bse"][1]))(tsecon.ols(y, X, se_type=se))
                for se in ["nonrobust", "hc1", "hac"]}
        b, s_naive = cand["nonrobust"]
        _, s_hac = cand["hac"]
        if abs(b - beta_true) > 1.96 * s_naive and abs(b - beta_true) < 1.96 * s_hac:
            results = cand
            break
    assert results is not None

    # Monte Carlo coverage of the 95% CI under each SE type.
    cover = {k: 0 for k in results}
    for rep in range(n_mc):
        ym, xm = draw(np.random.default_rng(10_000 + rep))
        Xm = np.column_stack([np.ones(n), xm])
        for se in cover:
            r = tsecon.ols(ym, Xm, se_type=se)
            b, s = r["params"][1], r["bse"][1]
            cover[se] += (b - 1.96 * s <= beta_true <= b + 1.96 * s)
    coverage = {k: v / n_mc for k, v in cover.items()}

    labels = {"nonrobust": "OLS (iid)", "hc1": "White HC1", "hac": "Newey-West HAC"}
    with ts.theme():
        fig, axes = plt.subplots(1, 2, figsize=(ts.WIDTH_DOUBLE, 2.5))
        ax = axes[0]
        for i, se in enumerate(["nonrobust", "hc1", "hac"]):
            b, s = results[se]
            ax.errorbar(b, i, xerr=1.96 * s, fmt="o", color=ts.SERIES["blue"],
                        ecolor=ts.SERIES["blue"], elinewidth=2.2, capsize=3, ms=5)
            ax.annotate(labels[se], xy=(b, i + 0.22), fontsize=8, color=ts.INK_2, ha="center")
        ax.axvline(beta_true, color=ts.REF, lw=0.9)
        ax.annotate("true β", xy=(beta_true, 2.45), fontsize=7.5, color=ts.MUTED,
                    ha="left", xytext=(beta_true + 0.005, 2.45))
        b, s = results["hac"]
        lo = min(beta_true, b - 1.96 * s) - 0.04
        hi = max(b + 1.96 * s for b, s in results.values()) + 0.04
        ax.set_xlim(lo, hi)
        ax.set_ylim(-0.6, 2.9); ax.set_yticks([])
        ax.set_title("Same estimate, honest vs dishonest intervals", fontsize=9.5, loc="left")

        ax = axes[1]
        ts.despine_x_only(ax)
        names = [labels[k] for k in ["nonrobust", "hc1", "hac"]]
        vals = [coverage[k] for k in ["nonrobust", "hc1", "hac"]]
        colors = [ts.SERIES["red"], ts.SERIES["yellow"], ts.SERIES["blue"]]
        bars = ax.barh(names, vals, color=colors, height=0.55)
        ax.axvline(0.95, color=ts.REF, lw=0.9)
        ax.annotate("nominal 95%", xy=(0.95, 2.55), fontsize=7.5, color=ts.MUTED, ha="center")
        for bar, v in zip(bars, vals):
            ax.annotate(f"{v:.0%}", xy=(v + 0.015, bar.get_y() + bar.get_height() / 2),
                        fontsize=8, color=ts.INK_2, ha="left", va="center", fontweight="semibold")
        ax.set_xlim(0, 1.12)
        ax.set_title(f"CI coverage over {n_mc:,} simulations", fontsize=9.5, loc="left")
        fig.suptitle("With autocorrelated errors, iid standard errors lie — HAC restores coverage",
                     x=0.002, ha="left", fontsize=11.5, fontweight="semibold", color=ts.INK)
        fig.tight_layout(rect=(0, 0.02, 1, 0.90))
        ts.stamp(fig, "y = 1 + 0.5x + u, x and u AR(0.7), n = 200 · tsecon.ols(se_type=...) · Monte Carlo via the Rust core")
        save(fig, "03-robust-se.png")


# ------------------------------------------------------------------
# 4. Bootstrap: iid resampling breaks dependence; block schemes keep it
# ------------------------------------------------------------------
def section_bootstrap():
    n, B = 300, 4000
    e = rng.standard_normal(n)
    y = np.empty(n); y[0] = 0.0
    for t in range(1, n):
        y[t] = 0.8 * y[t - 1] + e[t]

    ybar = y.mean()
    # True sampling std of the mean of an AR(0.8) (long-run variance / n).
    true_se = np.sqrt(tsecon.long_run_variance(y, kernel="bartlett") / n)

    means = {"iid": [], "stationary": []}
    opt = tsecon.optimal_block_length(y)
    p_star = 1.0 / opt["stationary"]
    for b in range(B):
        i_iid = tsecon.bootstrap_indices(n, scheme="iid", seed=b)
        i_blk = tsecon.bootstrap_indices(n, scheme="stationary", seed=b, p=p_star)
        means["iid"].append(y[i_iid].mean())
        means["stationary"].append(y[i_blk].mean())

    with ts.theme():
        fig, ax = plt.subplots(figsize=(ts.WIDTH_DOUBLE, 2.7))
        bins = np.linspace(ybar - 4 * true_se, ybar + 4 * true_se, 55)
        ax.hist(means["iid"], bins=bins, density=True, alpha=0.85, color=ts.SEQ_BLUE[0],
                edgecolor=ts.SURFACE, lw=0.4, label="iid bootstrap (breaks dependence)")
        ax.hist(means["stationary"], bins=bins, density=True, alpha=0.72, color=ts.SEQ_BLUE[3],
                edgecolor=ts.SURFACE, lw=0.4,
                label=f"stationary block bootstrap (E[block] ≈ {opt['stationary']:.1f})")
        xs = np.linspace(bins[0], bins[-1], 300)
        ax.plot(xs, np.exp(-((xs - ybar) / true_se) ** 2 / 2) / (true_se * np.sqrt(2 * np.pi)),
                color=ts.INK, lw=1.5, label="correct asymptotic distribution")
        ax.legend(loc="upper left", fontsize=7.5)
        ax.set_title("Bootstrap distributions of the sample mean of a persistent series")
        ax.set_yticks([])
        fig.tight_layout()
        ts.stamp(fig, f"AR(0.8), n = 300, B = {B:,} replications · tsecon.bootstrap_indices / optimal_block_length · "
                      "iid bootstrap understates the variance ~3x")
        save(fig, "04-bootstrap.png")


# ------------------------------------------------------------------
# 5. Kalman filtering & smoothing with missing data
# ------------------------------------------------------------------
def section_kalman():
    n = 160
    level = np.cumsum(rng.standard_normal(n) * 0.7) + 20
    y = level + rng.standard_normal(n) * 2.2
    y_obs = y.copy()
    y_obs[95:120] = np.nan  # a gap: sensor outage / unreleased data

    r = tsecon.local_level_smooth(y_obs, sigma2_eps=2.2**2, sigma2_eta=0.7**2)
    smo, var = np.asarray(r["smoothed_state"]), np.asarray(r["smoothed_state_var"])

    with ts.theme():
        fig, ax = plt.subplots(figsize=(ts.WIDTH_DOUBLE, 2.8))
        ts.shade_period(ax, 95, 119, "missing data")
        band = 1.96 * np.sqrt(var)
        ax.fill_between(np.arange(n), smo - band, smo + band, color=ts.SEQ_BLUE[0], lw=0, zorder=2)
        ax.plot(np.arange(n), y_obs, "o", ms=2.1, mew=0, color=ts.MUTED, zorder=3, label="observed")
        ax.plot(np.arange(n), smo, color=ts.SEQ_BLUE[5], lw=1.8, zorder=4, label="smoothed level")
        ax.plot(np.arange(n), level, color=ts.SERIES["red"], lw=1.0, ls=(0, (3, 2)), zorder=4,
                label="true latent level")
        ax.legend(loc="upper left", fontsize=7.5, ncol=3)
        ax.set_title("The state-space engine bridges a 25-period gap, widening its bands honestly")
        fig.tight_layout()
        ts.stamp(fig, "Local level model, exact diffuse initialization · tsecon.local_level_smooth · "
                      "bands: 95% smoothed-state intervals")
        save(fig, "05-kalman.png")


ALL = [section_acf, section_stationarity, section_hac, section_bootstrap, section_kalman]

if __name__ == "__main__":
    only = sys.argv[1:] if len(sys.argv) > 1 else None
    for fn in ALL:
        if only and not any(k in fn.__name__ for k in only):
            continue
        fn()
