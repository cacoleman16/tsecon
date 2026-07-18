"""The tsecon gallery, extensions wing: the roadmap-E methods — recession
probability, survey expectations (information rigidity + disagreement), long
memory, the CUSUM stability test, predictive regressions / IVX, the
arbitrage-free Nelson-Siegel adjustment, and the linear rational-expectations
(DSGE-lite) saddle path.

Run with the project venv (tsecon + matplotlib installed there):
    .venv/bin/python docs/examples/showcase_extensions.py [section-substring ...]

Figures land in docs/examples/img/ in the house style (Module 13).
"""
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


def _runs(mask):
    """Yield (start, end) index spans where a 0/1 mask is 1 — for shading."""
    mask = np.asarray(mask).astype(bool)
    edges = np.diff(np.concatenate([[0], mask.view(np.int8), [0]]))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0] - 1
    return list(zip(starts, ends))


# ------------------------------------------------------------------
# X1. Recession probit: an inverted term spread leads recessions
# ------------------------------------------------------------------
def section_recession():
    rng = np.random.default_rng(20260717)
    n = 264  # 22 years of monthly data
    t = np.arange(n)
    # A slow-moving term spread that dips below zero ahead of each downturn.
    spread = 1.4 + 0.9 * np.sin(t / 21.0) + np.cumsum(rng.standard_normal(n)) * 0.06
    lead = 9  # the spread leads the recession by ~9 months
    latent = -1.6 * np.roll(spread, lead) + 0.5 * rng.standard_normal(n)
    latent[:lead] = -3.0
    y = (latent > -0.4).astype(float)

    X = np.column_stack([np.ones(n), spread])
    fit = tsecon.recession_probit(y, X, link="probit")
    prob = np.asarray(fit["probabilities"])
    spans = _runs(y)

    with ts.theme():
        fig, (ax0, ax1) = plt.subplots(
            2, 1, figsize=(ts.WIDTH_DOUBLE, 3.5), sharex=True,
            gridspec_kw={"height_ratios": [1.0, 1.0]},
        )
        # Top: the term spread, with realized recessions shaded.
        for s, e in spans:
            ax0.axvspan(s, e, color=ts.SHADE, zorder=0, lw=0)
        ax0.plot(t, spread, color=ts.SERIES["blue"], lw=1.4, zorder=4)
        ts.zero_line(ax0)
        ax0.set_ylabel("Term spread", fontsize=8.5, color=ts.INK)
        ax0.tick_params(labelsize=7.5)
        ax0.annotate("shaded = realized recession", xy=(0.015, 0.06),
                     xycoords="axes fraction", ha="left", va="bottom",
                     fontsize=7.5, color=ts.MUTED)

        # Bottom: the fitted recession probability, same shading.
        for s, e in spans:
            ax1.axvspan(s, e, color=ts.SHADE, zorder=0, lw=0)
        ax1.plot(t, prob, color=ts.SERIES["red"], lw=1.6, zorder=5)
        ax1.axhline(0.5, color=ts.REF, lw=0.8, ls=(0, (2, 2)), zorder=2)
        ax1.set_ylabel("P(recession)", fontsize=8.5, color=ts.INK)
        ax1.set_xlabel("Month", fontsize=8.5, color=ts.INK_2)
        ax1.set_ylim(-0.03, 1.03)
        ax1.set_xlim(0, n - 1)
        ax1.tick_params(labelsize=7.5)

        fig.suptitle("A probit turns a leading spread into a recession probability",
                     x=0.005, ha="left", fontsize=11.5, fontweight="semibold",
                     color=ts.INK)
        fig.tight_layout(rect=(0, 0.0, 1, 0.93))
        ts.stamp(fig, "Synthetic monthly data · tsecon.recession_probit (Estrella-Mishkin "
                      "1998 style probit of a 0/1 recession indicator on the term spread) · "
                      f"McFadden pseudo-R2 = {fit['pseudo_r2']:.2f}; the fitted probability "
                      "climbs through 0.5 ahead of each shaded downturn, exactly as an "
                      "inverted yield curve is meant to warn")
        save(fig, "ext-recession-probit.png")


# ------------------------------------------------------------------
# X2. Survey expectations: information rigidity (CG) + disagreement
# ------------------------------------------------------------------
def section_survey():
    rng = np.random.default_rng(4)
    T = 180
    # Coibion-Gorodnichenko: under sticky information the mean forecast error
    # is predictable from the mean forecast revision (slope > 0).
    revision = rng.standard_normal(T)
    error = 0.55 * revision + 0.35 * rng.standard_normal(T)
    cg = tsecon.cg_regression(error, revision)
    slope, rigidity = cg["slope"], cg["implied_rigidity"]

    # A forecaster panel whose disagreement widens through a turbulent middle.
    n_periods = 140
    panel = []
    for p in range(n_periods):
        turbulence = 0.25 + 0.9 * np.exp(-((p - 70) ** 2) / 500.0)
        k = int(rng.integers(18, 32))
        panel.append(rng.normal(2.0 + 0.004 * p, turbulence, size=k))
    dis = tsecon.forecast_disagreement(panel)
    disp = np.asarray(dis["std"])
    pp = np.arange(n_periods)

    with ts.theme():
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(ts.WIDTH_DOUBLE, 2.9))

        # Left: the CG rigidity regression.
        ax0.scatter(revision, error, s=11, color=ts.SERIES["blue"],
                    alpha=0.6, edgecolor="none", zorder=3)
        xs = np.linspace(revision.min(), revision.max(), 50)
        ax0.plot(xs, cg["intercept"] + slope * xs, color=ts.SERIES["red"],
                 lw=1.8, zorder=5)
        # FIRE benchmark: a flat (zero-slope) line through the intercept.
        ax0.plot(xs, np.full_like(xs, cg["intercept"]), color=ts.REF,
                 lw=0.9, ls=(0, (2, 2)), zorder=2)
        ax0.set_xlabel("Mean forecast revision", fontsize=8.5, color=ts.INK)
        ax0.set_ylabel("Mean forecast error", fontsize=8.5, color=ts.INK)
        ax0.tick_params(labelsize=7.5)
        ax0.annotate(f"CG slope = {slope:.2f}\nimplied rigidity = {rigidity:.2f}",
                     xy=(0.04, 0.96), xycoords="axes fraction", ha="left", va="top",
                     fontsize=8, color=ts.SERIES["red"])
        ax0.set_title("Information rigidity (CG)", fontsize=9.5, color=ts.INK_2, loc="left")

        # Right: cross-sectional disagreement over time.
        ax1.fill_between(pp, 0, disp, color=ts.SEQ_BLUE[2], lw=0, zorder=2)
        ax1.plot(pp, disp, color=ts.SERIES["blue"], lw=1.4, zorder=4)
        ax1.set_xlabel("Period", fontsize=8.5, color=ts.INK)
        ax1.set_ylabel("Forecaster std. dev.", fontsize=8.5, color=ts.INK)
        ax1.set_xlim(0, n_periods - 1)
        ax1.set_ylim(0, disp.max() * 1.12)
        ax1.tick_params(labelsize=7.5)
        ax1.set_title("Disagreement widens in turbulence", fontsize=9.5,
                      color=ts.INK_2, loc="left")

        fig.suptitle("Survey expectations: are forecasts fully rational, and do forecasters agree?",
                     x=0.005, ha="left", fontsize=11.5, fontweight="semibold", color=ts.INK)
        fig.tight_layout(rect=(0, 0.0, 1, 0.91))
        ts.stamp(fig, "Synthetic survey panel · tsecon.cg_regression (Coibion-Gorodnichenko "
                      "2015, HAC standard errors) and tsecon.forecast_disagreement · the "
                      "dashed line is the full-information rational-expectations null (zero "
                      "slope, no predictable error); the positive fitted slope is the "
                      "sticky-information departure from it")
        save(fig, "ext-survey.png")


# ------------------------------------------------------------------
# X3. Long memory: fractional differencing whitens a persistent series
# ------------------------------------------------------------------
def section_long_memory():
    rng = np.random.default_rng(7)
    n = 4000
    d_true = 0.40  # stationary long memory (0 < d < 0.5)
    x = tsecon.frac_integrate(rng.standard_normal(n), d_true)
    x = np.asarray(x)

    est = tsecon.long_memory_d(x, method="local_whittle")
    d_hat = est["d"]
    xd = np.asarray(tsecon.frac_diff(x, d_hat))  # differencing by the estimate

    nlags = 40
    acf_x = np.asarray(tsecon.acf(x, nlags=nlags)["acf"])[1:]
    acf_d = np.asarray(tsecon.acf(xd, nlags=nlags)["acf"])[1:]
    lags = np.arange(1, nlags + 1)
    se = 1.96 / np.sqrt(len(x))

    with ts.theme():
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(ts.WIDTH_DOUBLE, 2.9), sharey=True)

        for ax, acf, title, col in (
            (ax0, acf_x, f"Long-memory series (d = {d_true:.2f})", ts.SERIES["violet"]),
            (ax1, acf_d, f"After (1-L)^d, d_hat = {d_hat:.2f}", ts.SERIES["aqua"]),
        ):
            ax.axhspan(-se, se, color=ts.SHADE, zorder=0, lw=0)
            ts.zero_line(ax)
            ax.vlines(lags, 0, acf, color=col, lw=2.2, zorder=4)
            ax.set_xlabel("Lag", fontsize=8.5, color=ts.INK)
            ax.set_xlim(0, nlags + 1)
            ax.tick_params(labelsize=7.5)
            ax.set_title(title, fontsize=9.5, color=ts.INK_2, loc="left")
        ax0.set_ylabel("Autocorrelation", fontsize=8.5, color=ts.INK)
        ax0.annotate("slow hyperbolic decay", xy=(0.5, 0.9), xycoords="axes fraction",
                     ha="center", va="top", fontsize=7.5, color=ts.MUTED)
        ax1.annotate("whitened (within the band)", xy=(0.5, 0.9), xycoords="axes fraction",
                     ha="center", va="top", fontsize=7.5, color=ts.MUTED)

        fig.suptitle("Long memory: a hyperbolically-decaying ACF, and the filter that removes it",
                     x=0.005, ha="left", fontsize=11.5, fontweight="semibold", color=ts.INK)
        fig.tight_layout(rect=(0, 0.0, 1, 0.91))
        ts.stamp(fig, "Synthetic ARFIMA(0,d,0) · tsecon.long_memory_d (Robinson 1995 local "
                      "Whittle) recovers d from the raw series; tsecon.frac_diff then applies "
                      f"(1-L)^d_hat · the estimate d_hat = {d_hat:.2f} (se {est['se']:.2f}) "
                      f"covers the true d = {d_true:.2f}, and the shaded 95% band shows the "
                      "differenced series is left white")
        save(fig, "ext-long-memory.png")


# ------------------------------------------------------------------
# X4. CUSUM: a mid-sample break drives the path out of its bounds
# ------------------------------------------------------------------
def section_cusum():
    rng = np.random.default_rng(11)
    n = 160
    x1 = rng.standard_normal(n)
    X = np.column_stack([np.ones(n), x1])

    beta = np.array([0.5, 1.0])
    y_stable = X @ beta + 0.6 * rng.standard_normal(n)
    # A break: the intercept and slope both shift halfway through.
    y_break = y_stable.copy()
    half = n // 2
    y_break[half:] += (X[half:] @ np.array([2.2, 1.4]))

    cs = tsecon.cusum_test(y_stable, X)
    cb = tsecon.cusum_test(y_break, X)
    path_s = np.asarray(cs["path"])
    path_b = np.asarray(cb["path"])
    up = np.asarray(cs["bound_upper"])
    lo = np.asarray(cs["bound_lower"])
    k = n - len(path_s)  # recursion starts after the first k obs
    xr = np.arange(k, n)

    breached = bool(np.any(path_b > up) or np.any(path_b < lo))

    with ts.theme():
        fig, ax = plt.subplots(figsize=(ts.WIDTH_DOUBLE, ts.WIDTH_DOUBLE * 0.42))
        ax.fill_between(xr, lo, up, color=ts.SHADE, zorder=0, lw=0)
        ax.plot(xr, up, color=ts.REF, lw=1.0, zorder=2)
        ax.plot(xr, lo, color=ts.REF, lw=1.0, zorder=2)
        ts.zero_line(ax)
        ax.plot(xr, path_s, color=ts.SERIES["blue"], lw=1.8, zorder=5)
        ax.plot(xr, path_b, color=ts.SERIES["red"], lw=1.8, zorder=6)
        ax.axvline(half, color=ts.INK_2, lw=0.7, ls=(0, (2, 2)), zorder=3)
        ax.annotate("break introduced here", xy=(half, lo[0]), xytext=(half + 3, lo[0] * 0.9),
                    fontsize=7.5, color=ts.INK_2, ha="left", va="bottom")
        ax.set_xlabel("Observation", fontsize=8.5, color=ts.INK)
        ax.set_ylabel("CUSUM of recursive residuals", fontsize=8.5, color=ts.INK)
        ax.set_xlim(k, n - 1)
        ax.tick_params(labelsize=7.5)

        handles = [
            Line2D([0], [0], color=ts.SERIES["blue"], lw=1.8, label="Stable coefficients"),
            Line2D([0], [0], color=ts.SERIES["red"], lw=1.8, label="Mid-sample break"),
            Patch(facecolor=ts.SHADE, edgecolor=ts.REF, label="5% stability band"),
        ]
        ax.legend(handles=handles, loc="upper left", fontsize=7.5, frameon=False,
                  ncol=1, borderaxespad=0.3)
        fig.suptitle("CUSUM: a stable fit stays inside the band; a break walks out of it",
                     x=0.005, ha="left", fontsize=11.5, fontweight="semibold", color=ts.INK)
        fig.tight_layout(rect=(0, 0.0, 1, 0.92))
        ts.stamp(fig, "Synthetic regression · tsecon.cusum_test (Brown-Durbin-Evans 1975, "
                      "a = 0.948 5% boundary) · the cumulative sum of recursive residuals is a "
                      "random walk near zero under stable coefficients (blue), but the "
                      f"halfway parameter shift sends it decisively past the band (red — "
                      f"boundary breached: {breached})")
        save(fig, "ext-cusum.png")


# ------------------------------------------------------------------
# X5. Predictive regressions / IVX: correct size near a unit root
# ------------------------------------------------------------------
def section_ivx():
    rng = np.random.default_rng(20260101)
    reps, n, corr_ue = 600, 250, -0.9
    rhos = [0.90, 0.95, 0.99, 1.00]
    chi2_95, z_95 = 3.841, 1.96
    ivx_rej, ols_rej = [], []
    for rho in rhos:
        ivx_hits = ols_hits = 0
        for _ in range(reps):
            e = rng.standard_normal(n)
            x = np.zeros(n)
            for t in range(1, n):
                x[t] = rho * x[t - 1] + e[t]
            u = corr_ue * e + np.sqrt(1 - corr_ue ** 2) * rng.standard_normal(n)
            r = u  # TRUE null: beta = 0
            fit = tsecon.predictive_regression(r, x)
            ivx_hits += fit["ivx"]["wald"] > chi2_95
            ols_hits += abs(fit["ols"]["tstat"]) > z_95
        ivx_rej.append(ivx_hits / reps)
        ols_rej.append(ols_hits / reps)

    idx = np.arange(len(rhos))
    with ts.theme():
        fig, ax = plt.subplots(figsize=(ts.WIDTH_DOUBLE, ts.WIDTH_DOUBLE * 0.42))
        w = 0.38
        ax.bar(idx - w / 2, ols_rej, w, color=ts.SERIES["red"], zorder=3,
               label="naive OLS t-test")
        ax.bar(idx + w / 2, ivx_rej, w, color=ts.SERIES["blue"], zorder=3,
               label="IVX Wald test")
        ax.axhline(0.05, color=ts.REF, lw=1.0, ls=(0, (3, 2)), zorder=4)
        ax.annotate("nominal 5%", xy=(len(rhos) - 0.5, 0.05), xytext=(0, 3),
                    textcoords="offset points", fontsize=7.5, color=ts.MUTED,
                    ha="right", va="bottom")
        ax.set_xticks(idx)
        ax.set_xticklabels([f"{r:.2f}" for r in rhos])
        ax.set_xlabel("Predictor persistence  ρ", fontsize=8.5, color=ts.INK)
        ax.set_ylabel("Rejection rate (size)", fontsize=8.5, color=ts.INK)
        ax.set_ylim(0, max(ols_rej) * 1.15)
        ax.tick_params(labelsize=7.5)
        ax.legend(loc="upper left", fontsize=7.5, frameon=False)
        fig.suptitle("IVX keeps its size up to the unit root; the OLS t-test does not",
                     x=0.005, ha="left", fontsize=11.5, fontweight="semibold", color=ts.INK)
        fig.tight_layout(rect=(0, 0.0, 1, 0.92))
        ts.stamp(fig, "Monte Carlo, 600 reps · tsecon.predictive_regression with a persistent, "
                      "endogenous predictor and a TRUE no-predictability null · as ρ climbs to "
                      "the exact unit root the naive OLS t-test rejects far too often (red bars "
                      "well above 5%), while the IVX Wald test (Kostakis-Magdalinos-"
                      "Stamatogiannis 2015) holds near its nominal level (blue)")
        save(fig, "ext-ivx-size.png")


# ------------------------------------------------------------------
# X6. AFNS: the arbitrage-free yield adjustment pulls long yields down
# ------------------------------------------------------------------
def section_afns():
    maturities = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30], float)
    lam = 0.6
    sig_low = np.array([0.007, 0.006, 0.009])
    sig_high = np.array([0.011, 0.009, 0.014])
    adj_low = np.asarray(tsecon.afns_adjustment(maturities, sig_low, decay=lam)) * 1e4  # bps
    adj_high = np.asarray(tsecon.afns_adjustment(maturities, sig_high, decay=lam)) * 1e4

    # A plain Nelson-Siegel curve (level/slope/curvature loadings), and the
    # AFNS curve = NS + adjustment, both in percent.
    L, S, C = 0.045, -0.02, 0.02
    tau = maturities
    load_s = (1 - np.exp(-lam * tau)) / (lam * tau)
    load_c = load_s - np.exp(-lam * tau)
    ns = L + S * load_s + C * load_c
    afns = ns + np.asarray(tsecon.afns_adjustment(maturities, sig_high, decay=lam))

    with ts.theme():
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(ts.WIDTH_DOUBLE, 2.9))

        # Left: the signed adjustment term, in basis points, vs maturity.
        ax0.plot(maturities, adj_low, "o-", color=ts.SERIES["aqua"], lw=1.6, ms=4,
                 zorder=4, label="lower factor vol")
        ax0.plot(maturities, adj_high, "o-", color=ts.SERIES["violet"], lw=1.6, ms=4,
                 zorder=5, label="higher factor vol")
        ts.zero_line(ax0)
        ax0.set_xlabel("Maturity (years)", fontsize=8.5, color=ts.INK)
        ax0.set_ylabel("Yield adjustment (bps)", fontsize=8.5, color=ts.INK)
        ax0.tick_params(labelsize=7.5)
        ax0.legend(loc="lower left", fontsize=7.5, frameon=False)
        ax0.set_title("−A(τ)/τ deepens with maturity", fontsize=9.5, color=ts.INK_2, loc="left")

        # Right: the reduced-form NS curve vs the arbitrage-free AFNS curve.
        ax1.plot(maturities, ns * 100, "o-", color=ts.SERIES["blue"], lw=1.6, ms=4,
                 zorder=4, label="Nelson-Siegel")
        ax1.plot(maturities, afns * 100, "o-", color=ts.SERIES["red"], lw=1.6, ms=4,
                 zorder=5, label="AFNS (arbitrage-free)")
        ax1.set_xlabel("Maturity (years)", fontsize=8.5, color=ts.INK)
        ax1.set_ylabel("Yield (%)", fontsize=8.5, color=ts.INK)
        ax1.tick_params(labelsize=7.5)
        ax1.legend(loc="lower center", fontsize=7.5, frameon=False)
        ax1.set_title("The long end is pulled down", fontsize=9.5, color=ts.INK_2, loc="left")

        fig.suptitle("Arbitrage-free Nelson-Siegel: same loadings, one extra term",
                     x=0.005, ha="left", fontsize=11.5, fontweight="semibold", color=ts.INK)
        fig.tight_layout(rect=(0, 0.0, 1, 0.91))
        ts.stamp(fig, "tsecon.afns_adjustment (Christensen-Diebold-Rudebusch 2011, "
                      "independent-factor closed form) · the adjustment −A(τ)/τ is zero at the "
                      "short end and grows negative with maturity and factor volatility; added "
                      "to the reduced-form Nelson-Siegel curve it restores the no-arbitrage "
                      "restriction, pulling long yields down")
        save(fig, "ext-afns.png")


# ------------------------------------------------------------------
# X7. DSGE-lite: the Cagan saddle path responds to a fundamentals shock
# ------------------------------------------------------------------
def section_dsge():
    a, rho = 0.7, 0.6
    A = np.array([[1.0, 0.0], [0.0, a]])
    B = np.array([[rho, 0.0], [-1.0, 1.0]])
    C = np.array([[1.0], [0.0]])
    sol = tsecon.dsge_solve(A, B, C, n_predetermined=1)
    G = np.asarray(sol["g"], float)
    P = np.asarray(sol["p"], float)
    Q = np.asarray(sol["q"], float)

    # Impulse response to a one-time unit fundamentals innovation, traced by
    # iterating the returned law of motion x_{t+1}=P x_t + Q eps and policy m=G x.
    H = 24
    x = np.zeros(H)
    m = np.zeros(H)
    x[0] = float(Q[0, 0])          # unit shock hits the predetermined state at t=0
    m[0] = float(G[0, 0]) * x[0]
    for t in range(1, H):
        x[t] = float(P[0, 0]) * x[t - 1]
        m[t] = float(G[0, 0]) * x[t]
    h = np.arange(H)

    with ts.theme():
        fig, ax = plt.subplots(figsize=(ts.WIDTH_DOUBLE, ts.WIDTH_DOUBLE * 0.42))
        ts.zero_line(ax)
        ax.plot(h, x, "o-", color=ts.SERIES["blue"], lw=1.6, ms=4, zorder=4,
                label="fundamental  x (predetermined)")
        ax.plot(h, m, "o-", color=ts.SERIES["red"], lw=1.6, ms=4, zorder=5,
                label="price level  m (jump)")
        ax.set_xlabel("Horizon (periods after the shock)", fontsize=8.5, color=ts.INK)
        ax.set_ylabel("Response to a unit innovation", fontsize=8.5, color=ts.INK)
        ax.set_xlim(0, H - 1)
        ax.tick_params(labelsize=7.5)
        ax.legend(loc="upper right", fontsize=7.5, frameon=False)
        ax.annotate(f"impact multiplier  G = 1/(1−aρ) = {G[0,0]:.3f}",
                    xy=(0.03, 0.95), xycoords="axes fraction", ha="left", va="top",
                    fontsize=8, color=ts.SERIES["red"])
        fig.suptitle("A rational-expectations saddle path: the jump variable leads on impact",
                     x=0.005, ha="left", fontsize=11.5, fontweight="semibold", color=ts.INK)
        fig.tight_layout(rect=(0, 0.0, 1, 0.92))
        ts.stamp(fig, "Cagan money-demand model solved by tsecon.dsge_solve (Blanchard-Kahn "
                      "1980) · one unstable root equals one jump variable, so the model has a "
                      "unique stable solution; the price level jumps to G times the shock on "
                      "impact and then both series decay back at the fundamental's AR root ρ = "
                      f"{rho}, the textbook present-value multiplier G = {G[0,0]:.3f}")
        save(fig, "ext-dsge-saddle.png")


# ------------------------------------------------------------------
# X8. Spectral analysis: raw periodogram vs Welch, and coherence
# ------------------------------------------------------------------
def section_spectral():
    rng = np.random.default_rng(20260714)
    n = 1024
    t = np.arange(n)
    period = 20.0            # a cycle at frequency 1/20 = 0.05
    f0 = 1.0 / period
    # A series with one strong periodic component buried in noise...
    x = 1.3 * np.sin(2 * np.pi * f0 * t) + 1.0 * rng.standard_normal(n)
    # ...and a second series sharing that cycle (phase-shifted) plus its own noise.
    y = 1.1 * np.sin(2 * np.pi * f0 * t + 0.9) + 1.0 * rng.standard_normal(n)

    pg = tsecon.periodogram(x)
    wl = tsecon.welch(x)
    co = tsecon.coherence(x, y)
    fp, psd_p = np.asarray(pg["freqs"]), np.asarray(pg["psd"])
    fw, psd_w = np.asarray(wl["freqs"]), np.asarray(wl["psd"])
    fc, coh = np.asarray(co["freqs"]), np.asarray(co["coherence"])

    with ts.theme():
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(ts.WIDTH_DOUBLE, 2.9))

        # Left: raw periodogram (spiky, inconsistent) vs Welch (smooth, consistent).
        ax0.plot(fp, psd_p, color=ts.INK_2, lw=0.6, zorder=3, alpha=0.8)
        ax0.plot(fw, psd_w, color=ts.SERIES["red"], lw=1.8, zorder=5)
        ax0.axvline(f0, color=ts.REF, lw=0.8, ls=(0, (2, 2)), zorder=2)
        ax0.set_yscale("log")
        ax0.set_xlim(0, 0.5)
        ax0.set_xlabel("Frequency (cycles/period)", fontsize=8.5, color=ts.INK)
        ax0.set_ylabel("Power spectral density", fontsize=8.5, color=ts.INK)
        ax0.tick_params(labelsize=7.5)
        ax0.annotate("cycle at f = 0.05", xy=(f0, psd_w.max()), xytext=(0.09, psd_w.max()),
                     fontsize=7.5, color=ts.MUTED, va="center")
        ax0.set_title("Periodogram vs Welch PSD", fontsize=9.5, color=ts.INK_2, loc="left")
        handles = [
            Line2D([0], [0], color=ts.INK_2, lw=0.8, label="raw periodogram"),
            Line2D([0], [0], color=ts.SERIES["red"], lw=1.8, label="Welch (averaged)"),
        ]
        ax0.legend(handles=handles, loc="lower left", fontsize=7.5, frameon=False)

        # Right: magnitude-squared coherence — high only at the shared frequency.
        ax1.fill_between(fc, 0, coh, color=ts.SEQ_BLUE[2], lw=0, zorder=2)
        ax1.plot(fc, coh, color=ts.SERIES["blue"], lw=1.4, zorder=4)
        ax1.axvline(f0, color=ts.REF, lw=0.8, ls=(0, (2, 2)), zorder=3)
        ax1.set_xlim(0, 0.5)
        ax1.set_ylim(0, 1.02)
        ax1.set_xlabel("Frequency (cycles/period)", fontsize=8.5, color=ts.INK)
        ax1.set_ylabel("Coherence  $|\\gamma|^2$", fontsize=8.5, color=ts.INK)
        ax1.tick_params(labelsize=7.5)
        ax1.annotate("shared cycle", xy=(f0, coh[np.argmin(np.abs(fc - f0))]),
                     xytext=(0.1, 0.85), fontsize=7.5, color=ts.MUTED, va="center")
        ax1.set_title("Coherence between two series", fontsize=9.5, color=ts.INK_2, loc="left")

        fig.suptitle("Spectral analysis: finding a cycle, and shared rhythm between series",
                     x=0.005, ha="left", fontsize=11.5, fontweight="semibold", color=ts.INK)
        fig.tight_layout(rect=(0, 0.0, 1, 0.91))
        ts.stamp(fig, "Synthetic series with a period-20 cycle · tsecon.periodogram / "
                      "tsecon.welch / tsecon.coherence (match scipy.signal to ~1e-15) · the raw "
                      "periodogram (grey) is a spiky, inconsistent PSD estimate; Welch averaging "
                      "(red) resolves the f = 0.05 peak cleanly, and the coherence spikes to ~1 "
                      "only at the frequency the two series share")
        save(fig, "ext-spectral.png")


ALL = [
    section_recession,
    section_survey,
    section_long_memory,
    section_cusum,
    section_ivx,
    section_afns,
    section_dsge,
    section_spectral,
]

if __name__ == "__main__":
    only = sys.argv[1:] if len(sys.argv) > 1 else None
    for fn in ALL:
        if only and not any(k in fn.__name__ for k in only):
            continue
        fn()
