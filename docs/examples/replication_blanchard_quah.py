"""Replication: Blanchard & Quah (1989) - supply and demand disturbances in US output and unemployment.

Blanchard and Quah's 1989 *American Economic Review* paper is the founding
application of long-run identification. A bivariate VAR in real output growth
and the unemployment rate is decomposed into two structural disturbances by a
single restriction: the "demand" disturbance has **no permanent effect on the
level of output**, so the "supply" disturbance is the only source of output's
stochastic trend. The published findings, read off their Figures 1-2 and their
variance-decomposition table: demand disturbances have a **hump-shaped**
effect on output that peaks after about a year and **vanishes** after a few
years; the effect on unemployment is, up to scale, a **mirror image** of that
on output; the effect of supply disturbances on output **builds steadily to a
permanent plateau**; a favourable supply disturbance **raises unemployment on
impact** before lowering it; and demand disturbances account for **most of the
forecast-error variance** of output at short horizons and of unemployment at
every horizon.

Data: the bundled `statsmodels.datasets.macrodata` (US real GDP and the
civilian unemployment rate, quarterly 1959Q1-2009Q3, US-government statistics,
public domain), committed as a two-column extract at
fixtures/macrodata_bq.csv so the CI guard runs offline like every other
replication page (tsecon ships no data loaders). Regenerate the extract from
the bundle with `--write-fixture`; the CI guard checks equality whenever
statsmodels is importable.

    .venv/bin/python docs/examples/replication_blanchard_quah.py

WHAT THIS IS, AND IS NOT
------------------------
This is a **design replication at figure-reading resolution**, not a numerical
one. Blanchard-Quah used real GNP and unemployment for 1948Q2-1987Q4 (their
estimation sample 1950Q2-1987Q4); that data vintage is not reachable from the
build environment (FRED and the institutional archives refuse the request), and
the bundled series is a later-vintage real GDP over 1959-2009 - a different
measure, a different deflator base, and a sample that adds the Great
Moderation and the 2008-09 recession while losing the 1950s. The page therefore
reproduces the paper's *design* - the same transformations, the same lag
order, the same restriction, the same objects (cumulated IRFs, variance
decompositions, historical decompositions) - and pins the *qualitative*
findings above with stated numerical bands. It does not claim parity with the
1989 tables, and every threshold in the CI guard is a shape claim, not a digit.

Every object comes from a native callable (since 0.11.0):

* the confidence bands are `tsecon.long_run_svar_bands`: a residual (iid)
  bootstrap of the reduced-form VAR that re-identifies inside every draw
  (Efron percentiles; optionally Kilian's bias correction), plus the
  delta-method arm. BQ report one-standard-deviation bands from a comparable
  Monte Carlo; the 68% percentile band below is the analogue.
* the Blanchard-Quah historical decomposition is
  `tsecon.historical_decomposition(identification="impact",
  impact=long_run_svar(...)["impact"])`.

Both were built in this script from shipped primitives before 0.11.0; those
constructions are kept as cross-checks (`bootstrap_draws_by_hand`,
`historical_decomposition_by_hand`) and agree with the native callables to
~1e-14 - not bit for bit, because the hand-built recursion sums in a different
order.

Sign convention: tsecon normalizes the long-run matrix to a positive diagonal,
which labels the demand disturbance by its *cumulative* effect on
unemployment. Blanchard-Quah sign it as expansionary - output rises on impact,
unemployment falls - which is `impact_signs=[(0, 1)]` (anchor the demand
column on output's impact response). A sign convention, not a different
model; the FEVD shares are invariant to it.
"""
import csv
import sys
from pathlib import Path

import numpy as np

import tsecon

DATA = Path(__file__).resolve().parents[2] / "fixtures" / "macrodata_bq.csv"

LAGS = 8              # Blanchard-Quah's lag order (their Section II)
HORIZON = 40          # ten years - the horizon of their Figures 1-2
MEAN_BREAK = "1974Q1"  # BQ demean output growth separately before/after the
                      # post-1973 productivity slowdown (their Section II)
N_BOOT = 1000
SEED = 20260911
PROBS = (0.05, 0.16, 0.84, 0.95)   # 90% and 68% percentile bands
SHOCKS = ("supply", "demand")
ANCHOR = [(0, 1)]     # BQ's sign: a demand disturbance raises output on impact
VARIABLES = ("output", "unemployment")

# NBER business-cycle reference dates (peak quarter, trough quarter) inside
# the sample - public reference chronology, used only to shade the figures
# and to tabulate the recession decomposition.
NBER_RECESSIONS = [
    ("1960Q2", "1961Q1"), ("1969Q4", "1970Q4"), ("1973Q4", "1975Q1"),
    ("1980Q1", "1980Q3"), ("1981Q3", "1982Q4"), ("1990Q3", "1991Q1"),
    ("2001Q1", "2001Q4"), ("2007Q4", "2009Q2"),
]


# ---------------------------------------------------------------------- data
def load_macrodata(path=DATA):
    """Read the committed macrodata extract (quarter, realgdp, unemp).

    Public-domain US statistical data, vendored with attribution in the CSV
    header - no download, no loader.
    """
    rows = [r for r in csv.reader(open(path)) if r and not r[0].startswith("#")]
    assert rows[0] == ["quarter", "realgdp", "unemp"], rows[0]
    body = rows[1:]
    return {
        "quarter": [r[0] for r in body],
        "realgdp": np.array([float(r[1]) for r in body]),
        "unemp": np.array([float(r[2]) for r in body]),
    }


def write_fixture(path=DATA):
    """Regenerate the extract verbatim from the statsmodels bundle."""
    import statsmodels
    import statsmodels.api as sm

    d = sm.datasets.macrodata.load_pandas().data
    header = (
        "# Blanchard-Quah (1989) design replication inputs: US real GDP and the unemployment "
        "rate, quarterly 1959Q1-2009Q3 (203 quarters).\n"
        "# realgdp = real gross domestic product, billions of chained 2005 US dollars, "
        "seasonally adjusted annual rate;\n"
        "# unemp = civilian unemployment rate, percent, seasonally adjusted.\n"
        "# Source: US-government statistical data (BEA / BLS via FRED), public domain. Copied "
        "verbatim from the\n"
        f"# bundled statsmodels {statsmodels.__version__} dataset "
        "`statsmodels.datasets.macrodata` (columns year, quarter, realgdp, unemp),\n"
        '# whose own COPYRIGHT note reads "This is public domain." Written by\n'
        "# `python docs/examples/replication_blanchard_quah.py --write-fixture`; "
        "test_replication_blanchard_quah.py\n"
        "# re-checks equality against the bundle whenever statsmodels is importable.\n"
    )
    with open(path, "w") as f:
        f.write(header + "quarter,realgdp,unemp\n")
        for y, q, r, u in zip(d["year"], d["quarter"], d["realgdp"], d["unemp"]):
            f.write(f"{int(y)}Q{int(q)},{r:.3f},{u:.1f}\n")
    print("wrote", path)


def qnum(label):
    """'1974Q1' -> 1974.0 (decimal year at the start of the quarter)."""
    y, q = label.split("Q")
    return int(y) + (int(q) - 1) / 4.0


def transform(d, mean_break=MEAN_BREAK, detrend_unemployment=True):
    """Blanchard-Quah's data treatment.

    growth = 100 * diff(log(real GDP)), demeaned separately before and after
    `mean_break` (BQ: the post-1973 productivity slowdown - a single
    sample mean would let the long-run restriction read the slowdown as one
    enormous negative supply disturbance); unemployment linearly detrended
    (BQ's Section II). The VAR carries a constant, so demeaning as such is
    inert; only the break and the trend change anything. Pass
    `mean_break=None` / `detrend_unemployment=False` for the sensitivity table.
    """
    growth = 100.0 * np.diff(np.log(d["realgdp"]))
    unemp = d["unemp"][1:].astype(float)
    quarter = d["quarter"][1:]
    years = np.array([qnum(q) for q in quarter])
    if mean_break is not None:
        post = years >= qnum(mean_break)
        growth = growth - np.where(post, growth[post].mean(), growth[~post].mean())
    else:
        growth = growth - growth.mean()
    if detrend_unemployment:
        t = np.arange(len(unemp), dtype=float)
        Z = np.column_stack([np.ones_like(t), t])
        unemp = unemp - Z @ np.linalg.lstsq(Z, unemp, rcond=None)[0]
    else:
        unemp = unemp - unemp.mean()
    return {
        "quarter": quarter, "years": years, "growth": growth, "unemp": unemp,
        "data": np.column_stack([growth, unemp]),
    }


# ----------------------------------------------------------------- estimation
def level_objects(irf):
    """The objects BQ plot from plain structural responses `irf[..., h, i, j]`
    (any leading axes, e.g. bootstrap draws): output's *cumulated* response
    (the level) in row 0, the unemployment rate's plain response in row 1."""
    irf = np.asarray(irf)
    resp = irf.copy()
    resp[..., 0, :] = np.cumsum(irf[..., 0, :], axis=-2)
    return resp


def level_fevd(cum):
    """Share of the h-step forecast-error variance of the *level* of a
    differenced variable: sum_{s<=h} C_s^2 normalized across shocks, with C_s
    the cumulated structural responses (BQ's output decomposition)."""
    c2 = np.cumsum(cum ** 2, axis=0)
    return c2 / c2.sum(axis=2, keepdims=True)


def fit_bq(data, lags=LAGS, horizon=HORIZON):
    """One `long_run_svar` call plus the derived objects the page reports."""
    raw = tsecon.long_run_svar(data, lags=lags, horizon=horizon, impact_signs=ANCHOR)
    resp = level_objects(raw["irf"])
    cum = np.asarray(raw["cumulative_irf"])
    var = tsecon.var_fit(data, lags=lags)
    return {
        "raw": raw,
        "impact": np.asarray(raw["impact"]),
        "long_run": np.asarray(raw["long_run"]),
        "long_run_multiplier": np.asarray(raw["long_run_multiplier"]),
        "responses": resp,                       # [h][variable][shock]
        "fevd_growth": np.asarray(raw["fevd"]),  # output GROWTH and unemployment
        "fevd_level": level_fevd(cum),           # output LEVEL (row 0 is the one to read)
        "var": var,
        "lags": lags,
        "horizon": horizon,
    }


def bootstrap_bands(data, lags=LAGS, horizon=HORIZON, n_boot=N_BOOT, seed=SEED,
                    probs=PROBS, bias_correct=False):
    """Native residual-bootstrap bands for the level responses.

    `tsecon.long_run_svar_bands` refits the reduced-form VAR on every
    pseudo-sample (iid resampling of the centred residuals from substream b of
    `SeedSequence(seed).spawn(n_boot)`, recursive from the first `lags`
    observations) and re-runs the Blanchard-Quah map with the same anchor.
    The level objects are formed per draw from `irf_draws`, then percentiles;
    the same seed always yields bit-identical output. Returns `quantiles`
    `[h][variable][shock][prob]`, the level draws and the native result.
    """
    nat = tsecon.long_run_svar_bands(
        data, lags=lags, horizon=horizon, impact_signs=ANCHOR, n_boot=n_boot, seed=seed,
        bias_correct=bias_correct, return_draws=True,
    )
    draws = level_objects(nat["irf_draws"])          # (n_boot, H+1, k, k)
    q = np.quantile(draws, probs, axis=0)            # (len(probs), H+1, k, k)
    return {
        "quantiles": np.moveaxis(q, 0, -1), "probs": tuple(probs),
        "n_boot": n_boot, "seed": seed, "draws": draws, "native": nat,
        "point": level_objects(nat["point"]),
    }


def delta_bands(data, lags=LAGS, horizon=HORIZON, probs=PROBS):
    """Native delta-method bands for the level objects: the asymptotic arm of
    `long_run_svar_bands`, cumulated for output and plain for unemployment.
    Symmetric `point +/- z se`; `quantiles` in the bootstrap layout."""
    out = np.empty((horizon + 1, 2, 2, len(probs)))
    for cumulative, row in ((True, 0), (False, 1)):
        for a, (lo, hi) in enumerate(zip(probs[:2], probs[::-1][:2])):
            r = tsecon.long_run_svar_bands(data, lags=lags, horizon=horizon, impact_signs=ANCHOR,
                                           method="asymptotic", cumulative=cumulative,
                                           alpha=2.0 * lo)
            out[:, row, :, a] = np.asarray(r["lower"])[:, row, :]
            out[:, row, :, len(probs) - 1 - a] = np.asarray(r["upper"])[:, row, :]
    return {"quantiles": out, "probs": tuple(probs)}


def spawn_indices(seed, n_boot, n):
    """NumPy replay of the native resampling stream: replication b is
    Philox(SeedSequence(seed).spawn(n_boot)[b]), each index drawn by bitmask
    rejection on one raw 64-bit output per attempt (tsecon_rng /
    tsecon_bootstrap; fixtures/philox.json pins the equivalence)."""
    mask = (1 << (n - 1).bit_length()) - 1
    out = []
    for child in np.random.SeedSequence(seed).spawn(n_boot):
        bg = np.random.Philox(child)
        idx = []
        while len(idx) < n:
            v = int(bg.random_raw()) & mask
            if v < n:
                idx.append(v)
        out.append(np.array(idx))
    return out


def bootstrap_draws_by_hand(data, lags=LAGS, horizon=HORIZON, n_boot=40, seed=SEED):
    """The pre-0.11.0 page construction, kept as a cross-check of the native
    bands: refit with `var_fit`, resample its centred residuals with the
    replayed native indices, rebuild each pseudo-sample as `z @ params + e`,
    and re-run `long_run_svar` with the same anchor. Returns plain `irf`
    draws (n_boot, H+1, k, k), comparable with `irf_draws`."""
    fit = tsecon.var_fit(data, lags=lags)
    params = np.asarray(fit["params"])        # rows: const, lag-1 block, lag-2 block, ...
    resid = np.asarray(fit["resid"])
    resid = resid - resid.mean(axis=0)
    T, k = resid.shape
    draws = np.empty((n_boot, horizon + 1, k, k))
    for b, idx in enumerate(spawn_indices(seed, n_boot, T)):
        e = resid[idx]
        y = np.empty((T + lags, k))
        y[:lags] = data[:lags]
        for t in range(lags, T + lags):
            z = np.concatenate([[1.0]] + [y[t - i] for i in range(1, lags + 1)])
            y[t] = z @ params + e[t - lags]
        draws[b] = tsecon.long_run_svar(y, lags=lags, horizon=horizon, impact_signs=ANCHOR)["irf"]
    return draws


def historical_decomposition_bq(data, lags=LAGS, fit=None):
    """The Blanchard-Quah historical decomposition, natively:
    `historical_decomposition(identification="impact")` with the anchored
    `long_run_svar` impact matrix (same lags, trend "c"). The `baseline`
    (deterministic + initial-condition path) is identification-invariant, so it
    is checked against the Cholesky decomposition's, and the pre-0.11.0
    hand-built decomposition is kept as a cross-check (`hand_check`)."""
    fit = fit or fit_bq(data, lags=lags, horizon=1)
    lib = tsecon.historical_decomposition(data, lags=lags, identification="impact",
                                          impact=fit["raw"]["impact"])
    baseline, hd = np.asarray(lib["baseline"]), np.asarray(lib["hd"])
    T = hd.shape[0]
    chol = tsecon.historical_decomposition(data, lags=lags, identification="cholesky")
    baseline_check = float(np.abs(baseline - np.asarray(chol["baseline"])).max())
    total_check = float(np.abs(hd.sum(axis=2) - np.asarray(chol["hd"]).sum(axis=2)).max())
    hand = historical_decomposition_by_hand(data, lags=lags, fit=fit)
    return {
        "times": np.arange(T) + lags,   # rows of `data`
        "baseline": baseline, "hd": hd, "shocks": np.asarray(lib["shocks"]),
        "shock_covariance": np.asarray(lib["shock_covariance"]),
        "baseline_check": baseline_check, "total_check": total_check,
        "hand_check": float(np.abs(hand["hd"] - hd).max()),
        # the level objects BQ plot: cumulated output contributions, plain
        # unemployment contributions - both net of the baseline
        "output_level": {
            "actual": np.cumsum(data[lags:, 0] - baseline[:, 0]),
            "supply": np.cumsum(hd[:, 0, 0]), "demand": np.cumsum(hd[:, 0, 1]),
        },
        "unemployment": {
            "actual": data[lags:, 1] - baseline[:, 1],
            "supply": hd[:, 1, 0], "demand": hd[:, 1, 1],
        },
    }


def historical_decomposition_by_hand(data, lags=LAGS, fit=None):
    """The pre-0.11.0 construction from shipped primitives, kept as a
    cross-check: structural shocks eps_t = B^-1 u_t from the VAR residuals and
    contributions sum_{s<=t} Theta_s[i, j] eps_{t-s, j}, with Theta_s the
    structural moving-average coefficients `long_run_svar` returns as `irf`."""
    fit = fit or fit_bq(data, lags=lags, horizon=1)
    resid = np.asarray(fit["var"]["resid"])
    T, k = resid.shape
    long = tsecon.long_run_svar(data, lags=lags, horizon=T - 1, impact_signs=ANCHOR)
    irf = np.asarray(long["irf"])
    shocks = np.linalg.solve(np.asarray(long["impact"]), resid.T).T
    hd = np.zeros((T, k, k))
    for t in range(T):
        hd[t] = np.einsum("sij,sj->ij", irf[:t + 1], shocks[t::-1])
    return {"hd": hd, "shocks": shocks}


def recession_table(tr, hd, episodes=NBER_RECESSIONS):
    """Peak-to-trough change in detrended unemployment split by disturbance."""
    quarters = tr["quarter"]
    first = int(hd["times"][0])
    rows = []
    for peak, trough in episodes:
        if peak not in quarters or trough not in quarters:
            continue
        a, b = quarters.index(peak) - first, quarters.index(trough) - first
        if a < 0:
            continue
        u = hd["unemployment"]
        rows.append({
            "peak": peak, "trough": trough,
            "actual": u["actual"][b] - u["actual"][a],
            "demand": u["demand"][b] - u["demand"][a],
            "supply": u["supply"][b] - u["supply"][a],
        })
    return rows


def bq_by_hand(data, lags=LAGS):
    """Independent NumPy transcription of the Blanchard-Quah closed form on a
    statsmodels VAR fit: C(1) = (I - sum A_i)^-1, LR = chol(C(1) Sigma C(1)'),
    B = C(1)^-1 LR (Sigma df-adjusted, statsmodels `sigma_u`), with BQ's sign
    (the demand column negated when output's impact response is negative,
    which is what `impact_signs=[(0, 1)]` does). None when statsmodels is not
    installed."""
    try:
        from statsmodels.tsa.api import VAR
    except ImportError:
        return None
    res = VAR(data).fit(lags, trend="c")
    k = data.shape[1]
    C1 = np.linalg.inv(np.eye(k) - np.asarray(res.coefs).sum(axis=0))
    LR = np.linalg.cholesky(C1 @ np.asarray(res.sigma_u) @ C1.T)
    B = np.linalg.solve(C1, LR)
    if B[0, 1] < 0.0:
        B[:, 1], LR[:, 1] = -B[:, 1], -LR[:, 1]
    return {"impact": B, "long_run": LR, "long_run_multiplier": C1}


def lag_ladder(data, max_lags=8):
    rows = []
    for p in range(1, max_lags + 1):
        f = tsecon.var_fit(data, lags=p)
        rows.append({"lags": p, "aic": f["aic"], "bic": f["bic"], "hqic": f["hqic"],
                     "min_root": f["min_root"], "is_stable": f["is_stable"]})
    return rows


def sensitivity(d, lag_orders=(2, 3, 8)):
    """The BQ treatment against its ingredients, at BQ's 8 lags and the
    information-criterion orders. Every cell re-runs `fit_bq`."""
    treatments = [
        ("constant only", dict(mean_break=None, detrend_unemployment=False)),
        ("mean break only", dict(mean_break=MEAN_BREAK, detrend_unemployment=False)),
        ("detrended u only", dict(mean_break=None, detrend_unemployment=True)),
        ("both (BQ)", dict(mean_break=MEAN_BREAK, detrend_unemployment=True)),
    ]
    out = []
    for name, kw in treatments:
        X = transform(d, **kw)["data"]
        for p in lag_orders:
            f = fit_bq(X, lags=p)
            r = f["responses"]
            out.append({
                "treatment": name, "lags": p,
                "yd_impact": r[0, 0, 1], "yd_peak": r[:, 0, 1].max(),
                "yd_peak_h": int(r[:, 0, 1].argmax()), "yd_h40": r[40, 0, 1],
                "ys_impact": r[0, 0, 0], "ys_lr": f["long_run"][0, 0],
                "ud_impact": r[0, 1, 1], "us_impact": r[0, 1, 0],
                "fevd_y_demand_h3": f["fevd_level"][3, 0, 1],
                "fevd_u_demand_h3": f["fevd_growth"][3, 1, 1],
            })
    return out


# -------------------------------------------------------------------- figures
def make_figures(tr, fit, boot, hd):
    """House-style figures (Module 13) into docs/examples/img/."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
    except ImportError:
        print("matplotlib not installed - figures skipped")
        return
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo / "prototypes" / "viz"))
    import tsecon_style as ts

    img = Path(__file__).resolve().parent / "img"
    img.mkdir(exist_ok=True)
    H = fit["horizon"]
    h = np.arange(H + 1)
    resp, q = fit["responses"], boot["quantiles"]
    C_90, C_68, C_PT = ts.SEQ_BLUE[1], ts.SEQ_BLUE[3], ts.SEQ_BLUE[6]
    stamp_data = ("US real GDP growth (1974Q1 mean break) and detrended unemployment, "
                  "1959Q2-2009Q3, statsmodels macrodata (public domain)")

    # --- 1. the IRF grid: BQ's Figures 1-2 ---------------------------------
    with ts.theme():
        fig, axes = plt.subplots(2, 2, figsize=(ts.WIDTH_DOUBLE, 4.6), sharex=True)
        row_labels = ["Output level (%)", "Unemployment rate (pp)"]
        col_labels = ["Supply disturbance", "Demand disturbance"]
        for i in range(2):
            for j in range(2):
                ax = axes[i, j]
                ts.zero_line(ax)
                ax.fill_between(h, q[:, i, j, 0], q[:, i, j, 3], color=C_90, lw=0, zorder=2)
                ax.fill_between(h, q[:, i, j, 1], q[:, i, j, 2], color=C_68, lw=0, zorder=3)
                ax.plot(h, resp[:, i, j], color=C_PT, lw=1.8, zorder=4)
                if i == 0:
                    ax.set_title(col_labels[j], fontsize=9.5, loc="center",
                                 color=ts.INK_2, fontweight="normal")
                if j == 0:
                    ax.set_ylabel(row_labels[i], fontsize=8.5, color=ts.INK)
                ax.set_xlim(0, H)
                ax.set_xticks([0, 8, 16, 24, 32, 40])
                ax.tick_params(labelsize=7.5)
        for ax in axes[1]:
            ax.set_xlabel("Quarters after the disturbance", fontsize=8)
        handles = [Patch(facecolor=C_90, label="90% bootstrap band"),
                   Patch(facecolor=C_68, label="68% bootstrap band"),
                   Line2D([0], [0], color=C_PT, lw=1.8, label="point estimate")]
        fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
                   fontsize=7.5, handlelength=1.4, columnspacing=1.6,
                   handletextpad=0.5, bbox_to_anchor=(0.5, 0.012))
        fig.suptitle("Demand disturbances are transitory, supply disturbances permanent: "
                     "Blanchard-Quah on 1959-2009 US data",
                     x=0.005, ha="left", fontsize=11.5, fontweight="semibold", color=ts.INK)
        fig.tight_layout(rect=(0, 0.07, 1, 0.94))
        ts.stamp(fig, f"{stamp_data} · tsecon.long_run_svar, VAR({fit['lags']}), long-run "
                      "restriction: demand has no permanent effect on output · bands: "
                      "tsecon.long_run_svar_bands, residual bootstrap re-identified in every "
                      f"draw, {boot['n_boot']} replications, seed {boot['seed']}, Efron "
                      "percentile · output = cumulated response, unemployment = level response")
        fig.savefig(img / "repl-bq-irf.png")
        plt.close(fig)
        print("wrote", img / "repl-bq-irf.png")

    # --- 2. variance decompositions -----------------------------------------
    with ts.theme():
        fig, axes = plt.subplots(1, 2, figsize=(ts.WIDTH_DOUBLE, 2.6), sharey=True)
        steps = np.arange(1, H + 2)
        panels = [("Output level", fit["fevd_level"][:, 0, :]),
                  ("Unemployment rate", fit["fevd_growth"][:, 1, :])]
        for ax, (name, shares) in zip(axes, panels):
            demand, supply = shares[:, 1] * 100, shares[:, 0] * 100
            ax.stackplot(steps, demand, supply, colors=[C_68, C_90], lw=0)
            ax.set_xlim(1, H + 1)
            ax.set_ylim(0, 100)
            ax.set_xticks([1, 8, 16, 24, 32, 40])
            ax.set_title(name, fontsize=9.5, loc="center", color=ts.INK_2, fontweight="normal")
            ax.set_xlabel("Forecast horizon (quarters)", fontsize=8)
            ax.tick_params(labelsize=7.5)
            ax.annotate("demand", xy=(20, demand[19] / 2), fontsize=8, color=ts.SURFACE,
                        ha="center", va="center", fontweight="semibold")
            ax.annotate("supply", xy=(36, demand[35] + supply[35] / 2), fontsize=8,
                        color=ts.INK_2, ha="center", va="center")
        axes[0].set_ylabel("Share of forecast-error variance (%)", fontsize=8.5, color=ts.INK)
        fig.suptitle("Demand disturbances dominate output's short-run variance and "
                     "unemployment's throughout",
                     x=0.005, ha="left", fontsize=11.5, fontweight="semibold", color=ts.INK)
        fig.tight_layout(rect=(0, 0.02, 1, 0.92))
        ts.stamp(fig, f"{stamp_data} · shares from tsecon.long_run_svar: `fevd` for the "
                      "unemployment rate, cumulated structural responses for the output level "
                      "(BQ's object) · rows sum to 100% by construction")
        fig.savefig(img / "repl-bq-fevd.png")
        plt.close(fig)
        print("wrote", img / "repl-bq-fevd.png")

    # --- 3. historical decomposition ----------------------------------------
    with ts.theme():
        fig, axes = plt.subplots(2, 1, figsize=(ts.WIDTH_DOUBLE, 4.4), sharex=True)
        yrs = tr["years"][hd["times"]]
        panels = [("Output level, % deviation (net of the deterministic baseline)",
                   hd["output_level"]),
                  ("Unemployment rate, pp deviation from trend (net of the baseline)",
                   hd["unemployment"])]
        for ax, (name, comp) in zip(axes, panels):
            for peak, trough in NBER_RECESSIONS:
                ts.shade_period(ax, qnum(peak), qnum(trough) + 0.25)
            ts.zero_line(ax)
            ax.plot(yrs, comp["actual"], color=ts.INK_2, lw=1.1, zorder=3)
            ax.plot(yrs, comp["demand"], color=ts.SERIES["blue"], lw=1.7, zorder=4)
            ax.set_title(name, fontsize=9.5, loc="left", color=ts.INK_2, fontweight="normal")
            ax.set_xlim(yrs[0] - 0.5, yrs[-1] + 0.5)
            ax.set_xticks(np.arange(1965, 2010, 5))
            ax.tick_params(labelsize=7.5)
        handles = [Line2D([0], [0], color=ts.INK_2, lw=1.1, label="actual"),
                   Line2D([0], [0], color=ts.SERIES["blue"], lw=1.7,
                          label="demand-disturbance component"),
                   Patch(facecolor=ts.SHADE, label="NBER recession")]
        fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
                   fontsize=7.5, handlelength=1.4, columnspacing=1.6,
                   handletextpad=0.5, bbox_to_anchor=(0.5, 0.012))
        fig.suptitle("Recessions are demand: the Blanchard-Quah historical decomposition",
                     x=0.005, ha="left", fontsize=11.5, fontweight="semibold", color=ts.INK)
        fig.tight_layout(rect=(0, 0.07, 1, 0.94))
        ts.stamp(fig, f"{stamp_data} · tsecon.historical_decomposition("
                      "identification=\"impact\") with the tsecon.long_run_svar impact matrix; "
                      "baseline identical to the Cholesky decomposition's (identification-"
                      "invariant) · effective sample 1961Q2-2009Q3")
        fig.savefig(img / "repl-bq-hd.png")
        plt.close(fig)
        print("wrote", img / "repl-bq-hd.png")


# ----------------------------------------------------------------------- main
def rule(width=78, ch="-"):
    print(ch * width)


def main(n_boot=N_BOOT, seed=SEED, figures=True):
    print("Replication - Blanchard & Quah (1989), American Economic Review 79(4)")
    print("long-run SVAR in US output growth and unemployment: supply and demand disturbances")
    rule(78, "=")

    d = load_macrodata()
    tr = transform(d)
    X = tr["data"]
    print(f"data: statsmodels macrodata extract (committed) - {len(d['quarter'])} quarters "
          f"{d['quarter'][0]}-{d['quarter'][-1]}; growth sample {tr['quarter'][0]}-"
          f"{tr['quarter'][-1]} (n = {len(X)})")
    print(f"      output growth = 100*dlog(real GDP), demeaned before/after {MEAN_BREAK} "
          "(BQ's break); unemployment linearly detrended (BQ)")
    print("      design replication at figure-reading resolution: BQ's 1948-1987 GNP vintage")
    print("      is unreachable; no numerical parity with the 1989 tables is claimed.")

    # --- lag order ------------------------------------------------------------
    print(f"\nLag order: BQ use {LAGS}. Information criteria on this sample:")
    ladder = lag_ladder(X)
    print("  lags |     AIC      BIC      HQ  | min |root| (stable iff > 1)")
    for r in ladder:
        print(f"  {r['lags']:>4} | {r['aic']:8.4f} {r['bic']:8.4f} {r['hqic']:8.4f} | "
              f"{r['min_root']:.3f}")
    picks = {c: min(ladder, key=lambda r: r[c])["lags"] for c in ("aic", "bic", "hqic")}
    print(f"  selected: AIC {picks['aic']}, BIC {picks['bic']}, HQ {picks['hqic']} - "
          f"the page follows BQ's {LAGS}; the sensitivity table re-runs the IC orders.")

    # --- the identification --------------------------------------------------
    fit = fit_bq(X)
    r = fit["responses"]
    LR, B = fit["long_run"], fit["impact"]
    print(f"\nVAR({LAGS}) with constant: stable = {fit['var']['is_stable']}, "
          f"min |root| = {fit['var']['min_root']:.3f}")
    print("Long-run matrix C(1)B (rows: output level, unemployment; cols: supply, demand):")
    print(f"  [[{LR[0, 0]:+.4f}  {LR[0, 1]:+.4f}]   <- the imposed zero: demand has no")
    print(f"   [{LR[1, 0]:+.4f}  {LR[1, 1]:+.4f}]]     permanent effect on output")
    print("Impact matrix B (one-standard-deviation disturbances):")
    print(f"  [[{B[0, 0]:+.4f}  {B[0, 1]:+.4f}]")
    print(f"   [{B[1, 0]:+.4f}  {B[1, 1]:+.4f}]]")
    print("  sign convention: impact_signs=[(0, 1)] anchors the demand column so a demand "
          "disturbance raises output on impact (BQ)")

    # --- the responses ------------------------------------------------------
    rule(78, "=")
    print("Level responses (output: cumulated; unemployment: plain), selected horizons")
    hs = [0, 2, 4, 8, 12, 20, 40]
    print("  response                  | " + " ".join(f"h={h:<3d}" for h in hs) + " | extremum (h)")
    rule()
    for i, vn in enumerate(VARIABLES):
        for j, sn in enumerate(SHOCKS):
            path = r[:, i, j]
            ext = int(np.argmax(np.abs(path)))
            print(f"  {vn:<12} <- {sn:<8} | " + " ".join(f"{path[h]:+.3f}" for h in hs)
                  + f" | {path[ext]:+.3f} ({ext})")

    yd, ys, ud, us = r[:, 0, 1], r[:, 0, 0], r[:, 1, 1], r[:, 1, 0]
    print("\nThe paper's findings, read at figure resolution:")
    print(f"  (1) demand -> output is hump-shaped and transitory: impact {yd[0]:+.3f}, peak "
          f"{yd.max():+.3f} at h = {int(yd.argmax())}, h = 20 {yd[20]:+.3f}, h = 40 "
          f"{yd[40]:+.3f} (zero at infinity by construction)")
    print(f"  (2) supply -> output is permanent and positive: impact {ys[0]:+.3f}, h = 40 "
          f"{ys[40]:+.3f}, long run {LR[0, 0]:+.3f}, minimum over h = 0..40 {ys.min():+.3f}")
    print(f"  (3) demand -> unemployment mirrors output: impact {ud[0]:+.3f}, trough "
          f"{ud.min():+.3f} at h = {int(ud.argmin())}, h = 40 {ud[40]:+.3f}")
    print(f"  (4) supply -> unemployment rises on impact ({us[0]:+.3f}), stays positive "
          f"through h = 20 ({us[20]:+.3f}) and dies out (h = 40 {us[40]:+.3f}; minimum "
          f"{us.min():+.3f} at h = {int(us.argmin())}) - BQ's later sign reversal is not "
          "visible on this vintage")

    # --- bootstrap bands -----------------------------------------------------
    boot = bootstrap_bands(X, n_boot=n_boot, seed=seed)
    q = boot["quantiles"]
    nat = boot["native"]
    print(f"\nNative residual-bootstrap percentile bands, tsecon.long_run_svar_bands "
          f"({n_boot} replications, seed {seed}):")
    print(f"  re-identified in every draw; anchor mismatches {nat['sign_anchor_mismatches']}, "
          f"explosive raw refits {nat['n_unstable_draws']}, full-sample min |root| "
          f"{nat['min_root']:.3f}")
    print("  response                  |  h  |   point |   68% band        |   90% band")
    rule()
    for i, vn in enumerate(VARIABLES):
        for j, sn in enumerate(SHOCKS):
            for h in (0, 4, 12, 40):
                print(f"  {vn:<12} <- {sn:<8} | {h:>3} | {r[h, i, j]:+.3f}  | "
                      f"[{q[h, i, j, 1]:+.3f}, {q[h, i, j, 2]:+.3f}] | "
                      f"[{q[h, i, j, 0]:+.3f}, {q[h, i, j, 3]:+.3f}]")

    kil = bootstrap_bands(X, n_boot=n_boot, seed=seed, bias_correct=True)
    dlt = delta_bands(X)
    qk, qd = kil["quantiles"], dlt["quantiles"]
    print(f"\nThe three arms side by side, 90% bands ({n_boot} replications, seed {seed}; "
          "Kilian = bias_correct=True, whose point moves to the corrected coefficients):")
    print("  response                  |  h  | Efron             | Efron+Kilian (point)       "
          "| delta")
    rule()
    for i, vn in enumerate(VARIABLES):
        for j, sn in enumerate(SHOCKS):
            for h in (0, 4, 12, 40):
                print(f"  {vn:<12} <- {sn:<8} | {h:>3} | [{q[h, i, j, 0]:+.3f}, {q[h, i, j, 3]:+.3f}] | "
                      f"[{qk[h, i, j, 0]:+.3f}, {qk[h, i, j, 3]:+.3f}] ({kil['point'][h, i, j]:+.3f}) | "
                      f"[{qd[h, i, j, 0]:+.3f}, {qd[h, i, j, 3]:+.3f}]")
    lr_e, lr_k = nat, kil["native"]
    print(f"  long-run matrix [1][0] 90% band: Efron [{lr_e['long_run_lower'][1][0]:+.3f}, "
          f"{lr_e['long_run_upper'][1][0]:+.3f}], Kilian [{lr_k['long_run_lower'][1][0]:+.3f}, "
          f"{lr_k['long_run_upper'][1][0]:+.3f}]")

    hand = bootstrap_draws_by_hand(X, n_boot=40, seed=seed)
    native40 = tsecon.long_run_svar_bands(X, lags=LAGS, horizon=HORIZON, impact_signs=ANCHOR,
                                          n_boot=40, seed=seed, return_draws=True)["irf_draws"]
    print(f"  cross-check: the pre-0.11.0 hand-built bootstrap on the replayed native stream "
          f"(40 draws) agrees with irf_draws to {np.abs(hand - native40).max():.1e}")

    # --- variance decompositions --------------------------------------------
    rule(78, "=")
    print("Share of forecast-error variance due to the DEMAND disturbance (%)")
    print("  horizon (steps) |  output level | output growth | unemployment")
    rule()
    for h in (1, 2, 4, 8, 12, 20, 40):
        print(f"  {h:>15} | {100 * fit['fevd_level'][h - 1, 0, 1]:13.1f} | "
              f"{100 * fit['fevd_growth'][h - 1, 0, 1]:13.1f} | "
              f"{100 * fit['fevd_growth'][h - 1, 1, 1]:12.1f}")
    print("  (5) demand dominates output's short-horizon variance and unemployment's at every")
    print("      horizon; its output share declines with the horizon, as in BQ's table.")

    # --- historical decomposition -------------------------------------------
    rule(78, "=")
    hd = historical_decomposition_bq(X, fit=fit)
    print("Historical decomposition (tsecon.historical_decomposition(identification=\"impact\")"
          " with the anchored long_run_svar impact):")
    print(f"  baseline vs the Cholesky decomposition's (identification-invariant): "
          f"max |diff| = {hd['baseline_check']:.1e}; total shock contribution: "
          f"{hd['total_check']:.1e}; vs the pre-0.11.0 hand-built decomposition: "
          f"{hd['hand_check']:.1e}")
    sc = hd["shock_covariance"]
    print(f"  shock_covariance diag = [{sc[0, 0]:.6f}, {sc[1, 1]:.6f}] = (T-m)/T, off-diagonal "
          f"{sc[0, 1]:+.1e} (the df-adjusted Sigma behind long_run_svar's impact)")
    u = hd["unemployment"]
    print(f"  corr(actual detrended unemployment, demand component) = "
          f"{np.corrcoef(u['actual'], u['demand'])[0, 1]:.3f}; with the supply component "
          f"{np.corrcoef(u['actual'], u['supply'])[0, 1]:.3f}")
    print("\n  NBER recession      | rise in unemployment (pp): actual  demand  supply")
    rule()
    for row in recession_table(tr, hd):
        print(f"  {row['peak']}-{row['trough']}       | "
              f"{row['actual']:+7.2f} {row['demand']:+7.2f} {row['supply']:+7.2f}")

    # --- cross-check ---------------------------------------------------------
    rule(78, "=")
    hand = bq_by_hand(X)
    if hand is None:
        print("statsmodels not installed - the NumPy closed-form cross-check was skipped")
    else:
        raw = fit["raw"]
        print("Cross-check: NumPy transcription of the closed form on a statsmodels VAR fit")
        for key in ("impact", "long_run", "long_run_multiplier"):
            diff = np.abs(np.asarray(raw[key]) - hand[key]).max()
            print(f"  max |tsecon - NumPy| {key:<20} = {diff:.1e}")

    # --- sensitivity ---------------------------------------------------------
    rule(78, "=")
    print("Sensitivity: the data treatment and the lag order (every cell a full re-fit)")
    print("  treatment         | p | y<-d impact  peak(h)   h=40 | y<-s impact  LR | "
          "u<-d h0 | u<-s h0 | demand share h=4: y   u")
    rule()
    for s in sensitivity(d):
        print(f"  {s['treatment']:<17} | {s['lags']} | {s['yd_impact']:+.2f}      "
              f"{s['yd_peak']:+.2f}({s['yd_peak_h']})  {s['yd_h40']:+.3f} | "
              f"{s['ys_impact']:+.2f}     {s['ys_lr']:+.2f} | {s['ud_impact']:+.2f}   | "
              f"{s['us_impact']:+.2f}   | {100 * s['fevd_y_demand_h3']:5.0f} "
              f"{100 * s['fevd_u_demand_h3']:5.0f}")
    print("  The 1974Q1 break in mean growth is what BQ's design hinges on: without it the")
    print("  post-1973 slowdown is read as supply, and demand's output share collapses.")

    if figures:
        rule(78, "=")
        make_figures(tr, fit, boot, hd)

    rule(78, "=")
    print("Published benchmark (BQ 1989, Figures 1-2 and their variance-decomposition table,")
    print("at figure resolution): hump-shaped, vanishing demand effects on output that peak")
    print("after about a year; a mirror-image unemployment response; a supply effect on")
    print("output that builds to a permanent plateau; supply raising unemployment on impact;")
    print("demand dominating short-horizon output variance and unemployment variance")
    print("throughout. All of that reproduces above on the 1959-2009 vintage.")


if __name__ == "__main__":
    if "--write-fixture" in sys.argv[1:]:
        write_fixture()
    else:
        main()
