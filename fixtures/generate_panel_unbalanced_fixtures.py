"""Golden fixtures for the observation mask of `tsecon-panel` — the within
(fixed-effects) estimator, the distributed-lag design and the panel local
projection on UNBALANCED panels (entities entering late, leaving early, and
with internal gaps), every cell outside the mask ignored.

Run with the project venv (this script never imports tsecon):
    .venv/bin/python fixtures/generate_panel_unbalanced_fixtures.py

Reference and grade (honest):

  * INDEPENDENT-PACKAGE leg — every stored slope, standard error,
    t-statistic and full parameter covariance comes from linearmodels'
    `PanelOLS` (Kevin Sheppard's package; version in `_meta`) fitted on
    the long (entity, time)-indexed frame that holds ONLY the observed
    cells, with `entity_effects` / `time_effects` per case and the
    entity-trend variant through explicit entity x trend regressors
    (`1[entity = i] * t`, `t` the calendar period index; N - 1 columns
    under time effects because the common linear trend lies in the span
    of the time dummies, N otherwise). On an unbalanced panel PanelOLS
    sweeps two-way effects out by the Frisch-Waugh-Lovell route
    (`PanelData._demean_both`: demean by one dimension, then residualise
    on the demeaned dummies of the other), i.e. the exact joint
    projection; the crate does the same with the roles fixed (per-entity
    projection first, time dummies partialled out by least squares), so
    the two agree to rounding and the Rust golden test pins 1e-10
    relative. Three covariances per case: `cov_type="unadjusted"`,
    `"clustered"` by entity (linearmodels' `auto_df` rule: the absorbed
    effects are exempt from the small-sample factor only under
    entity-only effects), and `"kernel"` (Bartlett, bandwidth 4 — the lag
    truncation `panel_fe` uses; the per-period score sums run over the
    entities observed in that period). `nobs` and `df_resid` are pinned
    exactly. PanelOLS keeps singleton observations by default
    (`singletons=True`), so nothing is dropped on either side.

  * DOCUMENTED-FORMULA leg — for the distributed-lag cases the cumulative
    effect, its delta-method standard error and 95% interval, and the
    quadratic marginal effects / turning point are the NumPy transcription
    of the crate's documented formulas evaluated on linearmodels'
    covariance (the same transcription as `generate_panel_dl_fixtures.py`;
    not an independent authority for the delta method itself).

  * The panel local projection has no third-party implementation; its
    per-horizon regression IS a PanelOLS with entity effects on the rows
    where the horizon-h target (or the cumulated target) and every lag are
    observed, so the `lp` cases pin `panel_lp`'s `irf` (the coefficient on
    the contemporaneous shock), `se` and `nobs` per horizon against
    linearmodels on exactly that design — an independent-package golden
    for the masked LP path with `bias_correction="none"` (the two
    half-panel jackknives are refused on unbalanced panels).

Data:

  * `empluk` — the Arellano-Bond (1991) UK firm panel `EmplUK` from the R
    package `plm` (140 firms, 1976-1984, 1031 firm-years: 103 firms
    observed 7 years, 23 for 8, 14 for 9 — the textbook unbalanced panel),
    fetched with `statsmodels.datasets.get_rdataset("EmplUK", "plm")`.
    Only transformations (log employment, log wage, log capital) of the
    observed cells and the fitted statistics are stored, laid out as
    N x T calendar arrays with `null` outside the mask.
  * `synthetic` — seeded (20260911) N = 20, T = 30 panel: entity i is
    observed from a random entry period (0..7) to a random exit period
    (T - 8 .. T), with 8% of the remaining cells punched out at random
    (internal gaps); a temperature-like AR(1) regressor with entity
    climatologies, a second regressor, entity and period effects, entity
    trends, AR(1) errors plus a common period component; true lag
    polynomials as in `generate_panel_dl_fixtures.py`. Every period keeps
    at least one entity and every entity keeps at least 6 cells.
  * `lp_synthetic` — seeded N = 12, T = 48 panel with entity effects, a
    common shock with response `0.8 * 0.6^h`, AR(0.3) noise, the same
    entry/exit/gap masking.
"""
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import linearmodels
import statsmodels.api as sm
from linearmodels.panel import PanelOLS

OUT = Path(__file__).parent
META = {
    "linearmodels": linearmodels.__version__,
    "numpy": np.__version__,
    "pandas": pd.__version__,
    "statsmodels": sm.__version__,
    "python": platform.python_version(),
}
Z975 = 1.959963984540054
BW = 4
SEED = 20260911


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def to_json_array(a):
    """N x T array with NaN -> null."""
    return [[None if not np.isfinite(v) else float(v) for v in row] for row in a]


def long_frame(y, regs, mask):
    """(entity, time)-indexed frame of the observed cells only."""
    n, t = y.shape
    ent, tim = np.nonzero(mask)
    df = pd.DataFrame({"entity": ent, "time": tim, "y": y[mask]})
    for name, x in regs:
        df[name] = x[mask]
    return df.set_index(["entity", "time"])


def fit_effects(df, names, entity, time, trends, n_entities):
    exog = df[names].copy()
    if trends:
        ent = df.index.get_level_values("entity").to_numpy()
        tim = df.index.get_level_values("time").to_numpy().astype(float)
        n_cols = n_entities - 1 if time else n_entities
        trend_cols = pd.DataFrame(
            {f"trend_{i}": (ent == i) * tim for i in range(n_cols)}, index=df.index
        )
        exog = pd.concat([exog, trend_cols], axis=1)
    mod = PanelOLS(df["y"], exog, entity_effects=entity, time_effects=time)
    out = {}
    for key, kw in [
        ("nonrobust", {"cov_type": "unadjusted"}),
        ("cluster_entity", {"cov_type": "clustered", "cluster_entity": True}),
        ("driscoll_kraay", {"cov_type": "kernel", "kernel": "bartlett", "bandwidth": BW}),
    ]:
        r = mod.fit(**kw)
        cov = r.cov.loc[names, names].to_numpy()
        out[key] = {
            "params": [float(r.params[n]) for n in names],
            "bse": [float(r.std_errors[n]) for n in names],
            "tstats": [float(r.tstats[n]) for n in names],
            "cov": cov.tolist(),
            "nobs": int(r.nobs),
            "df_resid": int(r.df_resid),
        }
    return out


def lag_design(y, regressors, mask, L, powers):
    """Lagged design on the observed cells: row (i, t) for t >= L when the
    cells t - L .. t of entity i are all observed; columns regressor-major
    then power then lag — the crate's column order. Returns the frame and
    the lagged mask (N x (T - L))."""
    n, t_len = y.shape
    tu = t_len - L
    lag_mask = np.zeros((n, tu), dtype=bool)
    for i in range(n):
        for s in range(tu):
            lag_mask[i, s] = mask[i, s : s + L + 1].all()
    cols, names = [], []
    for name, x in regressors:
        for p in range(1, powers + 1):
            for l in range(L + 1):
                cols.append((name if p == 1 else f"{name}^2", l, x[:, L - l : t_len - l] ** p))
                names.append(f"{name}_L{l}" if p == 1 else f"{name}^2_L{l}")
    regs = [(nm, c[2]) for nm, c in zip(names, cols)]
    df = long_frame(y[:, L:], regs, lag_mask)
    return df, names, lag_mask


def delta_method(params, cov, L, powers, eval_points):
    """The documented delta-method transcription (see the docstring)."""
    params = np.asarray(params)
    cov = np.asarray(cov)
    nl = L + 1
    out = {"cumulative_effect": [], "cumulative_se": [], "cumulative_ci_low": [],
           "cumulative_ci_high": []}
    for p in range(1, powers + 1):
        sl = slice((p - 1) * nl, p * nl)
        b = float(params[sl].sum())
        se = float(np.sqrt(np.ones(nl) @ cov[sl, sl] @ np.ones(nl)))
        out["cumulative_effect"].append(b)
        out["cumulative_se"].append(se)
        out["cumulative_ci_low"].append(b - Z975 * se)
        out["cumulative_ci_high"].append(b + Z975 * se)
    if powers == 2:
        b1, b2 = out["cumulative_effect"]
        v = cov[: 2 * nl, : 2 * nl]
        me, mse = [], []
        for x in eval_points:
            g = np.concatenate([np.ones(nl), 2.0 * x * np.ones(nl)])
            me.append(b1 + 2.0 * b2 * x)
            mse.append(float(np.sqrt(g @ v @ g)))
        g = np.concatenate([-np.ones(nl) / (2.0 * b2), np.ones(nl) * b1 / (2.0 * b2 ** 2)])
        out["eval_points"] = list(map(float, eval_points))
        out["marginal_effect"] = me
        out["marginal_se"] = mse
        out["turning_point"] = -b1 / (2.0 * b2)
        out["turning_point_se"] = float(np.sqrt(g @ v @ g))
    return out


def punch_mask(rng, n, t_len, hole_rate, min_cells):
    """Entry/exit windows plus random internal gaps; every period keeps an
    entity and every entity keeps `min_cells` cells."""
    while True:
        mask = np.zeros((n, t_len), dtype=bool)
        for i in range(n):
            a = rng.integers(0, 8)
            b = t_len - rng.integers(0, 8)
            mask[i, a:b] = True
        holes = rng.random((n, t_len)) < hole_rate
        mask &= ~holes
        if mask.sum(axis=1).min() >= min_cells and mask.sum(axis=0).min() >= 1:
            return mask


# ---------------------------------------------------------------------------
# EmplUK (plm), the Arellano-Bond firm panel
# ---------------------------------------------------------------------------
def empluk_arrays():
    d = sm.datasets.get_rdataset("EmplUK", "plm").data
    firms = sorted(d.firm.unique())
    years = sorted(d.year.unique())
    n, t_len = len(firms), len(years)
    fi = {f: i for i, f in enumerate(firms)}
    yi = {yv: t for t, yv in enumerate(years)}
    mask = np.zeros((n, t_len), dtype=bool)
    arrs = {k: np.full((n, t_len), np.nan) for k in ("log_emp", "log_wage", "log_capital")}
    for row in d.itertuples(index=False):
        i, t = fi[row.firm], yi[row.year]
        mask[i, t] = True
        arrs["log_emp"][i, t] = np.log(row.emp)
        arrs["log_wage"][i, t] = np.log(row.wage)
        arrs["log_capital"][i, t] = np.log(row.capital)
    assert mask.sum() == 1031
    return arrs, mask, n, t_len, [int(v) for v in years]


# ---------------------------------------------------------------------------
# synthetic unbalanced panel (the DL DGP of generate_panel_dl_fixtures.py)
# ---------------------------------------------------------------------------
def synthetic(rng, n, t_len):
    mu = rng.normal(20.0, 4.0, n)
    x = np.empty((n, t_len))
    for i in range(n):
        x[i, 0] = mu[i] + rng.standard_normal()
        for t in range(1, t_len):
            x[i, t] = mu[i] + 0.3 * (x[i, t - 1] - mu[i]) + rng.standard_normal()
    z = rng.normal(100.0, 20.0, n)[:, None] + rng.normal(0.0, 10.0, (n, t_len))
    alpha = rng.normal(0.0, 1.0, n)
    delta = rng.normal(0.0, 0.5, t_len)
    gtrend = rng.normal(0.0, 0.02, n)
    common = rng.normal(0.0, 0.5, t_len)
    e = np.empty((n, t_len))
    for i in range(n):
        e[i, 0] = rng.standard_normal()
        for t in range(1, t_len):
            e[i, t] = 0.4 * e[i, t - 1] + rng.standard_normal()
    e = e + common[None, :]
    beta = (1.0, -0.6, 0.3)
    gamma = (-0.02, 0.01, -0.005)
    y = alpha[:, None] + delta[None, :] + gtrend[:, None] * np.arange(t_len)[None, :] + e
    for l, b in enumerate(beta):
        y[:, 2:] += b * x[:, 2 - l : t_len - l]
    for l, g in enumerate(gamma):
        y[:, 2:] += g * x[:, 2 - l : t_len - l] ** 2
    y[:, 2:] += 0.05 * z[:, 2:]
    return x, z, y


def lp_panel(rng, n, t_len):
    shock = rng.standard_normal(t_len)
    alpha = rng.normal(0.0, 2.0, n)
    psi = 0.8 * 0.6 ** np.arange(8)
    y = np.empty((n, t_len))
    for i in range(n):
        u = np.empty(t_len)
        u[0] = rng.standard_normal()
        for t in range(1, t_len):
            u[t] = 0.3 * u[t - 1] + rng.standard_normal()
        y[i] = alpha[i] + np.convolve(shock, psi)[:t_len] + u
    return shock, y


def lp_design(y, shock, mask, h, n_lags, cumulative):
    """Rows (i, t) with t in [n_lags, T - h) whose target (y_{t+h}, or the
    cumulated y_t..y_{t+h}) and lags y_{t-1..t-n_lags} are observed; columns
    [shock_t, shock_{t-1..}, y_{t-1..}] — the crate's column order."""
    n, t_len = y.shape
    rows = []
    for i in range(n):
        for t in range(n_lags, t_len - h):
            target_ok = mask[i, t : t + h + 1].all() if cumulative else mask[i, t + h]
            lags_ok = mask[i, t - n_lags : t].all() if n_lags else True
            if not (target_ok and lags_ok):
                continue
            target = y[i, t : t + h + 1].sum() if cumulative else y[i, t + h]
            row = {"entity": i, "time": t, "y": target, "shock": shock[t]}
            for l in range(1, n_lags + 1):
                row[f"shock_L{l}"] = shock[t - l]
            for l in range(1, n_lags + 1):
                row[f"y_L{l}"] = y[i, t - l]
            rows.append(row)
    df = pd.DataFrame(rows).set_index(["entity", "time"])
    names = ["shock"] + [f"shock_L{l}" for l in range(1, n_lags + 1)] + [f"y_L{l}" for l in range(1, n_lags + 1)]
    return df, names


def main():
    fx = {"_meta": META, "datasets": {}, "fe_cases": [], "dl_cases": [], "lp_cases": []}

    # ---- EmplUK
    arrs, mask, n, t_len, years = empluk_arrays()
    fx["datasets"]["empluk"] = {
        "source": "statsmodels.datasets.get_rdataset('EmplUK', 'plm') — Arellano-Bond (1991)",
        "n_entities": n, "n_periods": t_len, "years": years, "n_obs": int(mask.sum()),
        "mask": mask.astype(int).tolist(),
        "log_emp": to_json_array(arrs["log_emp"]),
        "log_wage": to_json_array(arrs["log_wage"]),
        "log_capital": to_json_array(arrs["log_capital"]),
    }
    effects_menu = [
        ("entity", True, False, False),
        ("two_way", True, True, False),
        ("time_only", False, True, False),
        ("entity_trends", True, False, True),
        ("entity_trends_time", True, True, True),
    ]
    regs = [("log_wage", arrs["log_wage"]), ("log_capital", arrs["log_capital"])]
    df = long_frame(arrs["log_emp"], regs, mask)
    for label, ent, tim, tr in effects_menu:
        fx["fe_cases"].append({
            "name": f"empluk/{label}", "dataset": "empluk", "outcome": "log_emp",
            "regressors": ["log_wage", "log_capital"],
            "entity_effects": ent, "time_effects": tim, "entity_trends": tr,
            "fits": fit_effects(df, ["log_wage", "log_capital"], ent, tim, tr, n),
        })
    for label, L, powers, ent, tim, tr, regnames, pts in [
        ("empluk/dl_L1_two_way", 1, 1, True, True, False, ["log_wage", "log_capital"], None),
        ("empluk/dl_L2_quad_two_way", 2, 2, True, True, False, ["log_wage"], [2.5, 3.0]),
        ("empluk/dl_L1_trends_time", 1, 1, True, True, True, ["log_wage"], None),
    ]:
        rr = [(nm, arrs[nm]) for nm in regnames]
        dfl, names, lag_mask = lag_design(arrs["log_emp"], rr, mask, L, powers)
        fits = fit_effects(dfl, names, ent, tim, tr, n)
        case = {
            "name": label, "dataset": "empluk", "outcome": "log_emp", "regressors": regnames,
            "lags": L, "powers": powers, "entity_effects": ent, "time_effects": tim,
            "entity_trends": tr, "eval_points": pts, "names": names,
            "n_obs_lagged": int(lag_mask.sum()), "fits": fits, "delta": {},
        }
        for key, f in fits.items():
            case["delta"][key] = delta_method(f["params"], f["cov"], L, powers, pts or [])
        fx["dl_cases"].append(case)

    # ---- synthetic unbalanced panel
    rng = np.random.default_rng(SEED)
    n, t_len = 20, 30
    x, z, y = synthetic(rng, n, t_len)
    mask = punch_mask(rng, n, t_len, 0.08, 6)
    ynan = np.where(mask, y, np.nan)
    fx["datasets"]["synthetic"] = {
        "seed": SEED, "n_entities": n, "n_periods": t_len, "n_obs": int(mask.sum()),
        "mask": mask.astype(int).tolist(),
        "y": to_json_array(ynan), "x": to_json_array(np.where(mask, x, np.nan)),
        "z": to_json_array(np.where(mask, z, np.nan)),
        "note": "cells outside the mask are null; the crate must ignore them",
    }
    regs = [("x", x), ("z", z)]
    df = long_frame(y, regs, mask)
    for label, ent, tim, tr in effects_menu:
        fx["fe_cases"].append({
            "name": f"synthetic/{label}", "dataset": "synthetic", "outcome": "y",
            "regressors": ["x", "z"],
            "entity_effects": ent, "time_effects": tim, "entity_trends": tr,
            "fits": fit_effects(df, ["x", "z"], ent, tim, tr, n),
        })
    for label, L, powers, ent, tim, tr, regnames, pts in [
        ("synthetic/dl_L0_two_way", 0, 1, True, True, False, ["x"], None),
        ("synthetic/dl_L2_two_way", 2, 1, True, True, False, ["x"], None),
        ("synthetic/dl_L1_two_reg", 1, 1, True, True, False, ["x", "z"], None),
        ("synthetic/dl_L2_quad_two_way", 2, 2, True, True, False, ["x"], [15.0, 20.0, 25.0]),
        ("synthetic/dl_L1_quad_default_points", 1, 2, True, True, False, ["x"], "mean"),
        ("synthetic/dl_L1_entity", 1, 1, True, False, False, ["x"], None),
        ("synthetic/dl_L1_time_only", 1, 1, False, True, False, ["x"], None),
        ("synthetic/dl_L1_trends", 1, 1, True, False, True, ["x"], None),
        ("synthetic/dl_L2_trends_time", 2, 1, True, True, True, ["x"], None),
    ]:
        rr = [(nm, {"x": x, "z": z}[nm]) for nm in regnames]
        dfl, names, lag_mask = lag_design(y, rr, mask, L, powers)
        fits = fit_effects(dfl, names, ent, tim, tr, n)
        if pts == "mean":
            # the crate's default: the pooled mean of the raw regressor over
            # the OBSERVED cells of the original (unlagged) panel
            pts_used = [float(x[mask].mean())]
        else:
            pts_used = pts
        case = {
            "name": label, "dataset": "synthetic", "outcome": "y", "regressors": regnames,
            "lags": L, "powers": powers, "entity_effects": ent, "time_effects": tim,
            "entity_trends": tr, "eval_points": pts, "names": names,
            "n_obs_lagged": int(lag_mask.sum()), "fits": fits, "delta": {},
        }
        if pts == "mean":
            case["eval_points_used"] = pts_used
        for key, f in fits.items():
            case["delta"][key] = delta_method(f["params"], f["cov"], L, powers, pts_used or [])
        fx["dl_cases"].append(case)

    # ---- panel local projection on a masked panel
    rng = np.random.default_rng(SEED + 1)
    n, t_len = 12, 48
    shock, y = lp_panel(rng, n, t_len)
    mask = punch_mask(rng, n, t_len, 0.08, 12)
    fx["datasets"]["lp_synthetic"] = {
        "seed": SEED + 1, "n_entities": n, "n_periods": t_len, "n_obs": int(mask.sum()),
        "mask": mask.astype(int).tolist(),
        "y": to_json_array(np.where(mask, y, np.nan)), "shock": shock.tolist(),
        "true_irf_psi": (0.8 * 0.6 ** np.arange(8)).tolist(),
    }
    for label, hmax, n_lags, cumulative in [
        ("lp/h3_lags2", 3, 2, False),
        ("lp/h2_lags0", 2, 0, False),
        ("lp/h3_lags1_cumulative", 3, 1, True),
    ]:
        per_h = []
        for h in range(hmax + 1):
            dfh, names = lp_design(y, shock, mask, h, n_lags, cumulative)
            fits = fit_effects(dfh, names, True, False, False, n)
            per_h.append({
                "h": h, "nobs": fits["nonrobust"]["nobs"],
                "irf": {k: f["params"][0] for k, f in fits.items()},
                "se": {k: f["bse"][0] for k, f in fits.items()},
                "params": fits["nonrobust"]["params"],
            })
        fx["lp_cases"].append({
            "name": label, "dataset": "lp_synthetic", "max_horizon": hmax,
            "n_lag_controls": n_lags, "cumulative": cumulative, "horizons": per_h,
        })

    path = OUT / "panel_unbalanced.json"
    path.write_text(json.dumps(fx, separators=(",", ":")))
    print(f"wrote {path} ({path.stat().st_size / 1024:.0f} KB): "
          f"{len(fx['fe_cases'])} fe cases, {len(fx['dl_cases'])} dl cases, {len(fx['lp_cases'])} lp cases")


if __name__ == "__main__":
    main()
