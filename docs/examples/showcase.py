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

# ----------------------------------------------------------------- coverage
# Single source of truth for every uncertainty band in this gallery: each
# band's WIDTH and its LABEL both derive from these levels, so changing the
# percents here changes every figure and its annotations consistently.
COVERAGE = {
    "ci": 0.95,                  # frequentist intervals: white-noise band, OLS CIs, Kalman band
    "bvar": [0.68, 0.90],        # BVAR credible bands (posterior quantile pairs), inner -> outer
    "fan": [0.40, 0.68, 0.90],   # ARIMA fan chart, inner -> outer
}


def z_two_sided(level):
    """Exact two-sided normal multiplier: z = Phi^-1((1 + level) / 2).

    E.g. 0.95 -> 1.9600, 0.90 -> 1.6449, 0.68 -> 0.9945, 0.40 -> 0.5244.
    """
    from scipy.stats import norm
    return float(norm.ppf((1.0 + level) / 2.0))


def pct(level):
    """Format a coverage level as a percent label: 0.90 -> '90%'."""
    return f"{100 * level:g}%"


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
    band = z_two_sided(COVERAGE["ci"]) / np.sqrt(n)

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
        ts.stamp(fig, "Synthetic AR(2), phi = (1.3, -0.4), n = 400 · tsecon.acf / tsecon.pacf · "
                      f"shaded: {pct(COVERAGE['ci'])} white-noise band")
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
            # Reserve head- and foot-room so the annotations never sit on the data.
            lo_y, hi_y = float(np.min(y)), float(np.max(y))
            span = hi_y - lo_y
            ax.set_ylim(lo_y - 0.24 * span, hi_y + 0.42 * span)
            verdict = rep["quadrant"]
            rec = rep["recommendation"]
            color = ts.SERIES["green"] if rec == "Proceed" else ts.SERIES["red"]
            ax.annotate(f"{verdict}: {rec}", xy=(0.02, 0.04), xycoords="axes fraction",
                        fontsize=8, color=color, fontweight="semibold")
            ax.annotate(f"ADF p = {rep['adf_p_value']:.3f}\nKPSS p = {rep['kpss_p_value']:.3f}",
                        xy=(0.02, 0.97), xycoords="axes fraction", fontsize=7,
                        color=ts.MUTED, va="top")
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
    ci = COVERAGE["ci"]
    z = z_two_sided(ci)

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
        if abs(b - beta_true) > z * s_naive and abs(b - beta_true) < z * s_hac:
            results = cand
            break
    assert results is not None

    # Monte Carlo coverage of the CI (level set by COVERAGE["ci"]) per SE type.
    cover = {k: 0 for k in results}
    for rep in range(n_mc):
        ym, xm = draw(np.random.default_rng(10_000 + rep))
        Xm = np.column_stack([np.ones(n), xm])
        for se in cover:
            r = tsecon.ols(ym, Xm, se_type=se)
            b, s = r["params"][1], r["bse"][1]
            cover[se] += (b - z * s <= beta_true <= b + z * s)
    coverage = {k: v / n_mc for k, v in cover.items()}

    labels = {"nonrobust": "OLS (iid)", "hc1": "White HC1", "hac": "Newey-West HAC"}
    with ts.theme():
        fig, axes = plt.subplots(1, 2, figsize=(ts.WIDTH_DOUBLE, 2.5))
        ax = axes[0]
        for i, se in enumerate(["nonrobust", "hc1", "hac"]):
            b, s = results[se]
            ax.errorbar(b, i, xerr=z * s, fmt="o", color=ts.SERIES["blue"],
                        ecolor=ts.SERIES["blue"], elinewidth=2.2, capsize=3, ms=5)
            ax.annotate(labels[se], xy=(b, i + 0.22), fontsize=8, color=ts.INK_2, ha="center")
        ax.axvline(beta_true, color=ts.REF, lw=0.9)
        ax.annotate("true β", xy=(beta_true, 2.45), fontsize=7.5, color=ts.MUTED,
                    ha="left", xytext=(beta_true + 0.005, 2.45))
        b, s = results["hac"]
        lo = min(beta_true, b - z * s) - 0.04
        hi = max(b + z * s for b, s in results.values()) + 0.04
        ax.set_xlim(lo, hi)
        ax.set_ylim(-0.6, 2.9); ax.set_yticks([])
        ax.set_title("Same estimate, honest vs dishonest intervals", fontsize=9.5, loc="left")

        ax = axes[1]
        ts.despine_x_only(ax)
        names = [labels[k] for k in ["nonrobust", "hc1", "hac"]]
        vals = [coverage[k] for k in ["nonrobust", "hc1", "hac"]]
        colors = [ts.SERIES["red"], ts.SERIES["yellow"], ts.SERIES["blue"]]
        bars = ax.barh(names, vals, color=colors, height=0.55)
        ax.axvline(ci, color=ts.REF, lw=0.9)
        # Extra headroom keeps the label inside the axes (annotate clips it otherwise).
        ax.set_ylim(-0.55, 3.05)
        ax.annotate(f"nominal {pct(ci)}", xy=(ci, 2.55), fontsize=7.5, color=ts.MUTED, ha="center")
        for bar, v in zip(bars, vals):
            ax.annotate(f"{v:.0%}", xy=(v + 0.015, bar.get_y() + bar.get_height() / 2),
                        fontsize=8, color=ts.INK_2, ha="left", va="center", fontweight="semibold")
        ax.set_xlim(0, 1.12)
        ax.set_title(f"{pct(ci)} CI coverage over {n_mc:,} simulations", fontsize=9.5, loc="left")
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
        band = z_two_sided(COVERAGE["ci"]) * np.sqrt(var)
        ax.fill_between(np.arange(n), smo - band, smo + band, color=ts.SEQ_BLUE[0], lw=0, zorder=2)
        ax.plot(np.arange(n), y_obs, "o", ms=2.1, mew=0, color=ts.MUTED, zorder=3, label="observed")
        ax.plot(np.arange(n), smo, color=ts.SEQ_BLUE[5], lw=1.8, zorder=4, label="smoothed level")
        ax.plot(np.arange(n), level, color=ts.SERIES["red"], lw=1.0, ls=(0, (3, 2)), zorder=4,
                label="true latent level")
        ax.legend(loc="upper left", fontsize=7.5, ncol=3)
        lo_y, hi_y = ax.get_ylim()
        ax.set_ylim(lo_y, hi_y + 0.16 * (hi_y - lo_y))  # headroom: legend clear of the dots
        ax.set_title("The state-space engine bridges a 25-period gap, widening its bands honestly")
        fig.tight_layout()
        ts.stamp(fig, "Local level model, exact diffuse initialization · tsecon.local_level_smooth · "
                      f"bands: {pct(COVERAGE['ci'])} smoothed-state intervals")
        save(fig, "05-kalman.png")


# ------------------------------------------------------------------
# 6. VAR: impulse responses and variance decomposition
# ------------------------------------------------------------------
def section_var():
    # A small synthetic macro system with a clear causal story:
    # "demand" moves first, "output" responds with a lag, "rate" leans against both.
    n = 400
    e = rng.standard_normal((n + 100, 3))
    y = np.zeros((n + 100, 3))
    A1 = np.array([[0.5, 0.0, 0.0], [0.35, 0.45, -0.15], [0.15, 0.25, 0.6]])
    A2 = np.array([[0.1, 0.0, 0.0], [0.1, 0.1, -0.05], [0.0, 0.1, 0.1]])
    for t in range(2, n + 100):
        y[t] = A1 @ y[t - 1] + A2 @ y[t - 2] + e[t]
    data = y[100:]
    names = ["Demand", "Output", "Policy rate"]

    irf = np.array(tsecon.var_irf(data, lags=2, horizon=16, orth=True))
    fevd = np.array(tsecon.var_fevd(data, lags=2, horizon=16))

    with ts.theme():
        fig, axes = plt.subplots(3, 3, figsize=(ts.WIDTH_DOUBLE, 5.2), sharex=True)
        for i in range(3):        # responding variable
            for j in range(3):    # shock
                ax = axes[i, j]
                ts.zero_line(ax)
                ax.plot(irf[:, i, j], color=ts.SERIES["blue"], lw=1.7)
                if i == 0:
                    ax.set_title(f"{names[j]} shock", fontsize=9, loc="center",
                                 color=ts.INK_2, fontweight="normal")
                if j == 0:
                    ax.set_ylabel(names[i], fontsize=8.5, color=ts.INK)
                ax.tick_params(labelsize=7.5)
        for ax in axes[2]:
            ax.set_xlabel("Horizon", fontsize=8)
        fig.suptitle("The full IRF grid of a three-variable VAR(2), Cholesky-identified",
                     x=0.002, ha="left", fontsize=11.5, fontweight="semibold", color=ts.INK)
        fig.tight_layout(rect=(0, 0.012, 1, 0.955))
        ts.stamp(fig, "Synthetic VAR(2), n = 400 · tsecon.var_irf(orth=True) · matches statsmodels at 1e-8 · "
                      "point IRFs — for bands see tsecon.bvar_irf_draws or tsecon.lp")
        save(fig, "06-var-irf.png")

        # FEVD as stacked areas for the output variable.
        fig, ax = plt.subplots(figsize=(ts.WIDTH_DOUBLE, 2.5))
        h = np.arange(1, 17)
        shares = fevd[1]  # output's forecast-error variance: (horizon, shock)
        ax.stackplot(h, shares.T * 100, colors=[ts.SEQ_BLUE[1], ts.SEQ_BLUE[3], ts.SEQ_BLUE[5]], lw=0)
        ax.set_xlim(1, 17.8)
        ax.set_ylim(0, 100)
        ax.set_ylabel("Share of forecast-error variance (%)")
        ax.set_xlabel("Horizon", fontsize=8)
        cum = np.concatenate([[0.0], np.cumsum(shares[-1]) * 100])
        for kk, nm in enumerate(names):
            mid = (cum[kk] + cum[kk + 1]) / 2
            ax.annotate(f"{nm} shock", xy=(16.15, mid), fontsize=8,
                        color=ts.INK_2 if kk < 2 else ts.SURFACE, va="center",
                        ha="left" if kk < 2 else "right",
                        xytext=(16.15, mid) if kk < 2 else (15.8, mid))
        ax.set_title("What drives output? Variance decomposition across horizons")
        fig.tight_layout()
        ts.stamp(fig, "tsecon.var_fevd · rows sum to 100% by construction (property-tested)")
        save(fig, "07-var-fevd.png")


# ------------------------------------------------------------------
# 7. Trend-cycle filters on real US GDP
# ------------------------------------------------------------------
def section_filters():
    import json as _json
    gdp = np.array(_json.loads((REPO / "fixtures" / "filters.json").read_text())["y_100_log_realgdp"])
    q = np.arange(len(gdp)) / 4 + 1959.25  # quarterly index, macrodata sample

    hp = tsecon.hp_filter(gdp, lamb=1600.0)
    ham = tsecon.hamilton_filter(gdp, h=8, p=4)
    bk = tsecon.bk_filter(gdp, low=6, high=32, k=12)

    with ts.theme():
        fig, axes = plt.subplots(2, 1, figsize=(ts.WIDTH_DOUBLE, 4.4), sharex=True)
        ax = axes[0]
        ax.plot(q, gdp, color=ts.MUTED, lw=1.0, label="100 x log real GDP")
        ax.plot(q, hp["trend"], color=ts.SEQ_BLUE[5], lw=1.7, label="HP trend (λ = 1600)")
        ax.legend(loc="upper left", fontsize=7.5)
        ax.set_title("Trend extraction", fontsize=9.5, loc="left")

        ax = axes[1]
        ts.zero_line(ax)
        ax.plot(q, hp["cycle"], color=ts.SERIES["blue"], lw=1.4, label="HP cycle")
        hi = ham["first_index"]
        ax.plot(q[hi:], ham["cycle"], color=ts.SERIES["red"], lw=1.4,
                label="Hamilton (2018) cycle")
        bi = bk["first_index"]
        ax.plot(q[bi:bi + len(bk["cycle"])], bk["cycle"], color=ts.SERIES["yellow"], lw=1.2,
                label="Baxter-King 6-32q")
        ax.legend(loc="lower left", fontsize=7.5, ncol=3)
        lo_y, hi_y = ax.get_ylim()
        ax.set_ylim(lo_y - 0.22 * (hi_y - lo_y), hi_y)  # footroom: legend clear of the 1975 dip
        ax.set_title("Three views of the business cycle — they disagree, and that is the point",
                     fontsize=9.5, loc="left")
        ax.set_xlabel("Year", fontsize=8)
        fig.suptitle("Trend-cycle decomposition: HP, Hamilton, and band-pass filters",
                     x=0.002, ha="left", fontsize=11.5, fontweight="semibold", color=ts.INK)
        fig.tight_layout(rect=(0, 0.015, 1, 0.95))
        ts.stamp(fig, "US real GDP (statsmodels macrodata) · tsecon.hp_filter / hamilton_filter / bk_filter · "
                      "filters that lose observations report their alignment explicitly")
        save(fig, "08-filters.png")


# ------------------------------------------------------------------
# 8. Forecast evaluation: benchmarks, accuracy, and the DM test
# ------------------------------------------------------------------
def section_forecast_eval():
    # Synthetic quarterly series: trend + seasonality + AR noise.
    n, h = 140, 20
    t = np.arange(n + h)
    season = 4.0 * np.array([1.0, -0.4, 0.6, -1.2])[t % 4]
    noise = np.empty(n + h)
    noise[0] = 0.0
    e = rng.standard_normal(n + h)
    for i in range(1, n + h):
        noise[i] = 0.6 * noise[i - 1] + e[i] * 1.5
    y = 50 + 0.3 * t + season + noise
    train, test = y[:n], y[n:]

    fc_theta = tsecon.theta_forecast(train, steps=h, period=4)
    fc_naive = np.full(h, train[-1])
    fc_snaive = np.tile(train[-4:], h // 4 + 1)[:h]

    e_theta = test - fc_theta
    e_snaive = test - fc_snaive
    dm = tsecon.dm_test(e_snaive, e_theta, h=1, loss="squared")

    acc = {name: tsecon.accuracy(test, f, insample=train, period=4)
           for name, f in [("Theta", fc_theta), ("Seasonal naive", fc_snaive), ("Naive", fc_naive)]}

    with ts.theme():
        fig, axes = plt.subplots(1, 2, figsize=(ts.WIDTH_DOUBLE, 2.7),
                                 gridspec_kw={"width_ratios": [1.7, 1]})
        ax = axes[0]
        ax.plot(np.arange(n - 40, n), train[-40:], color=ts.INK, lw=1.4)
        ax.plot(np.arange(n, n + h), test, color=ts.MUTED, lw=1.2, ls=(0, (2, 2)), label="actual")
        ax.plot(np.arange(n, n + h), fc_theta, color=ts.SERIES["blue"], lw=1.6, label="Theta")
        ax.plot(np.arange(n, n + h), fc_snaive, color=ts.SERIES["yellow"], lw=1.4, label="seasonal naive")
        ax.plot(np.arange(n, n + h), fc_naive, color=ts.SERIES["red"], lw=1.2, label="naive")
        ax.axvline(n - 0.5, color=ts.REF, lw=0.9)
        ax.legend(loc="upper left", fontsize=7, ncol=2)
        ax.set_title("Three benchmarks against held-out data", fontsize=9.5, loc="left")

        ax = axes[1]
        ts.despine_x_only(ax)
        names = list(acc.keys())[::-1]
        vals = [acc[k]["mase"] for k in names]
        best = min(vals)
        colors = [ts.SERIES["blue"] if v == best else ts.SEQ_BLUE[1] for v in vals]
        bars = ax.barh(names, vals, color=colors, height=0.55)
        ax.axvline(1.0, color=ts.REF, lw=0.9)
        # Footroom keeps the two-line label inside the axes (annotate clips it otherwise).
        ax.set_ylim(-1.15, 2.6)
        ax.annotate("MASE = 1\n(in-sample naive)", xy=(1.12, -0.45), fontsize=6.5,
                    color=ts.MUTED, ha="left", va="top")
        for bar, v in zip(bars, vals):
            ax.annotate(f"{v:.2f}", xy=(v + 0.03, bar.get_y() + bar.get_height() / 2),
                        fontsize=8, color=ts.INK_2, ha="left", va="center")
        ax.set_xlim(0, max(vals) * 1.25)
        ax.set_title("MASE (lower is better)", fontsize=9.5, loc="left")
        fig.suptitle("Forecast evaluation: accuracy measures plus a formal test of the difference",
                     x=0.002, ha="left", fontsize=11.5, fontweight="semibold", color=ts.INK)
        fig.tight_layout(rect=(0, 0.015, 1, 0.90))
        ts.stamp(fig, f"tsecon.theta_forecast / accuracy / dm_test · DM (HLN) seasonal-naive vs Theta: "
                      f"stat = {dm['hln_stat']:.2f}, p = {dm['p_value']:.3f}")
        save(fig, "09-forecast-eval.png")


# ------------------------------------------------------------------
# 9. GARCH: conditional volatility and risk
# ------------------------------------------------------------------
def section_garch():
    import json as _json
    ret = np.array(_json.loads((REPO / "fixtures" / "garch.json").read_text())["returns"])
    r = tsecon.garch_fit(ret, vol="garch", mean="zero", dist="normal", forecast_horizon=60)
    vol = np.asarray(r["conditional_volatility"])
    names, params, se = r["param_names"], np.asarray(r["params"]), np.asarray(r["se_robust"])

    with ts.theme():
        fig, axes = plt.subplots(2, 1, figsize=(ts.WIDTH_DOUBLE, 4.2), sharex=False,
                                 gridspec_kw={"height_ratios": [1, 1.15]})
        ax = axes[0]
        ax.plot(ret, color=ts.MUTED, lw=0.5)
        ax.set_title("Returns: volatility clusters — calm and stormy regimes alternate",
                     fontsize=9.5, loc="left")
        ax = axes[1]
        ax.plot(vol, color=ts.SERIES["blue"], lw=1.2, label="conditional volatility (fitted)")
        n = len(ret)
        fc = np.sqrt(np.asarray(r["variance_forecast"]))
        ax.plot(np.arange(n, n + len(fc)), fc, color=ts.SEQ_BLUE[6], lw=1.6, ls=(0, (4, 2)),
                label="60-day volatility forecast")
        lr = np.sqrt(params[0] / (1 - params[1] - params[2]))
        ax.axhline(lr, color=ts.REF, lw=0.9)
        # Label the reference line in clear space to the right of the forecast.
        ax.annotate("long-run level", xy=(n + len(fc) + 25, lr + 0.025), fontsize=7.5,
                    color=ts.MUTED, va="bottom", ha="left")
        ax.set_xlim(-40, n + len(fc) + 340)
        ax.legend(loc="upper right", fontsize=7.5)
        lo_y, hi_y = ax.get_ylim()
        ax.set_ylim(lo_y, hi_y + 0.30 * (hi_y - lo_y))  # headroom: legend clear of vol spikes
        ax.set_title("GARCH(1,1): spikes decay geometrically toward the long-run level; "
                     "forecasts mean-revert", fontsize=9.5, loc="left")
        param_str = "  ".join(f"{nm} = {p:.3f} ({s:.3f})" for nm, p, s in zip(names, params, se))
        fig.suptitle("Volatility is forecastable: the GARCH workhorse",
                     x=0.002, ha="left", fontsize=11.5, fontweight="semibold", color=ts.INK)
        fig.tight_layout(rect=(0, 0.015, 1, 0.95))
        ts.stamp(fig, f"tsecon.garch_fit · QMLE estimates (Bollerslev-Wooldridge robust SEs): {param_str} · "
                      "matches the arch package at machine precision")
        save(fig, "10-garch.png")


# ------------------------------------------------------------------
# 10. Bayesian VAR: posterior impulse responses with credible bands
# ------------------------------------------------------------------
def section_bvar():
    import json as _json
    data = np.array(_json.loads((REPO / "fixtures" / "bvar_niw.json").read_text())["data"])
    names = ["GDP growth", "Consumption", "Investment"]
    H = 12
    draws = np.array(tsecon.bvar_irf_draws(data, lags=2, horizon=H, n_draws=800, seed=42,
                                           lambda1=0.2))
    # Credible bands: quantile pairs derive from COVERAGE["bvar"] so that a
    # level L band is exactly [(1-L)/2, (1+L)/2] of the posterior draws.
    levels = sorted(COVERAGE["bvar"], reverse=True)  # outer (widest) first
    bands = {lv: np.quantile(draws, [(1 - lv) / 2, (1 + lv) / 2], axis=0) for lv in levels}
    q50 = np.quantile(draws, 0.5, axis=0)

    with ts.theme():
        fig, axes = plt.subplots(3, 3, figsize=(ts.WIDTH_DOUBLE, 5.2), sharex=True)
        t = np.arange(H + 1)
        for i in range(3):
            for j in range(3):
                ax = axes[i, j]
                ts.zero_line(ax)
                for k, lv in enumerate(levels):
                    qlo, qhi = bands[lv]
                    ax.fill_between(t, qlo[:, i, j], qhi[:, i, j], color=ts.SEQ_BLUE[k], lw=0)
                ax.plot(t, q50[:, i, j], color=ts.SEQ_BLUE[5], lw=1.7)
                if i == 0:
                    ax.set_title(f"{names[j]} shock", fontsize=9, loc="center",
                                 color=ts.INK_2, fontweight="normal")
                if j == 0:
                    ax.set_ylabel(names[i], fontsize=8.5, color=ts.INK)
                ax.tick_params(labelsize=7.5)
        for ax in axes[2]:
            ax.set_xlabel("Horizon (quarters)", fontsize=8)
        band_label = " / ".join(f"{100 * lv:g}" for lv in sorted(COVERAGE["bvar"])) + "%"
        fig.suptitle(f"Bayesian VAR: posterior impulse responses with {band_label} credible bands",
                     x=0.002, ha="left", fontsize=11.5, fontweight="semibold", color=ts.INK)
        fig.tight_layout(rect=(0, 0.012, 1, 0.955))
        ts.stamp(fig, "Minnesota-NIW conjugate BVAR(2) on US GDP/consumption/investment growth · "
                      "tsecon.bvar_irf_draws (800 posterior draws, seed-reproducible) · "
                      "closed-form posterior validated at 1e-13")
        save(fig, "11-bvar-irf.png")


# ------------------------------------------------------------------
# 11. ARIMA: fit, forecast, fan chart — end to end
# ------------------------------------------------------------------
def section_arima():
    n, h = 160, 16
    e = rng.standard_normal(n + 60)
    g = np.empty(n + 60)
    g[0] = 0.0
    for t in range(1, n + 60):          # ARMA(1,1) growth around 0.5
        g[t] = 0.5 * 0.55 + 0.45 * g[t - 1] + e[t] * 0.9 + 0.35 * e[t - 1]
    level = 100 + np.cumsum(g[60:])      # integrate: an ARIMA(1,1,1)+drift level

    r = tsecon.arima_fit(level, p=1, d=1, q=1, constant=True, forecast_steps=h)
    mean, se = np.asarray(r["forecast_mean"]), np.asarray(r["forecast_se"])

    with ts.theme():
        fig, ax = plt.subplots(figsize=(ts.WIDTH_DOUBLE, 2.9))
        x_h, x_f = np.arange(n), np.arange(n - 1, n + h)
        m = np.concatenate([[level[-1]], mean])
        s = np.concatenate([[0.0], se])
        fan_levels = sorted(COVERAGE["fan"], reverse=True)  # widest first, lightest step
        for cov, step in zip(fan_levels, ts.SEQ_BLUE):
            z = z_two_sided(cov)
            ax.fill_between(x_f, m - z * s, m + z * s, color=step, lw=0, zorder=2)
        ax.plot(x_h[-90:], level[-90:], color=ts.INK, lw=1.5, zorder=3)
        ax.plot(x_f, m, color=ts.SEQ_BLUE[6], lw=1.6, ls=(0, (4, 2)), zorder=3)
        ax.axvline(n - 1, color=ts.REF, lw=0.9, zorder=1.5)
        z_outer = z_two_sided(fan_levels[0])
        ax.annotate(f"{pct(fan_levels[0])} band",
                    xy=(x_f[-1] + 0.4, m[-1] + z_outer * s[-1] * 0.9),
                    fontsize=7.5, color=ts.INK_2, va="center")
        ax.annotate("point forecast", xy=(x_f[-1] + 0.4, m[-1]), fontsize=7.5,
                    color=ts.SEQ_BLUE[6], va="center")
        ax.set_xlim(n - 92, n + h + 7)
        names = ", ".join(f"{k} = {v:.3f}" for k, v in zip(r["param_names"], r["params"]))
        ax.set_title("ARIMA(1,1,1) with drift: fitted by exact MLE, forecast with honest fan")
        fig.tight_layout()
        ts.stamp(fig, f"tsecon.arima_fit (exact MLE via the Kalman engine) · {names} · "
                      "undifferencing carries exact cumulative variance — the fan widens like sqrt(h)")
        save(fig, "12-arima-fan.png")


ALL = [section_acf, section_stationarity, section_hac, section_bootstrap, section_kalman,
       section_var, section_filters, section_forecast_eval, section_garch, section_bvar,
       section_arima]

if __name__ == "__main__":
    only = sys.argv[1:] if len(sys.argv) > 1 else None
    for fn in ALL:
        if only and not any(k in fn.__name__ for k in only):
            continue
        fn()
