"""Golden fixtures for the static-panel menu of `tsecon-panel` — random
effects (`panel_re`), the between estimator (`panel_between`), first
differences (`panel_fd`) and Fama-MacBeth (`panel_fama_macbeth`) — on
balanced and unbalanced (masked) panels.

Run with the reference venv (this script never imports tsecon):
    .venv/bin/python fixtures/generate_panel_static_fixtures.py

The `_meta` block pins the versions this was run with (numpy 2.5.3, pandas
3.0.6, Python 3.13 at the time of writing); they legitimately differ from
`panel_unbalanced.json`'s pins (numpy 2.4.6, pandas 3.0.5, Python 3.11.15) —
the EmplUK INPUTS are read from that fixture, not refetched, so the inputs
are bit-identical to `panel_fe`'s and only derived statistics are stored
here.

Reference and grade (honest):

  * INDEPENDENT-PACKAGE leg — every stored coefficient, standard error,
    t-statistic and full covariance comes from linearmodels (Kevin
    Sheppard's package; version in `_meta`) with `debiased=True` (its
    default): `RandomEffects.fit(small_sample=..., cov_type=...)`,
    `BetweenOLS.fit(reweight=..., cov_type=...)`,
    `FirstDifferenceOLS.fit(cov_type=...)` and
    `FamaMacBeth.fit(cov_type=...)`, with `cov_type` one of
    `"unadjusted"`, `"robust"` (tsecon's `"hc1"`), `"clustered"` with
    `cluster_entity=True`, and `"kernel"` with `kernel="bartlett"`,
    `bandwidth=4` (linearmodels truncates `int(bandwidth)` and tsecon
    weights `1 - j/(bandwidth + 1)`; the two agree at integer bandwidths,
    and linearmodels' own `bandwidth=None` is an automatic rule while
    tsecon's None means 4.0, so 4 is passed explicitly). Every frame holds
    only the observed cells and is built TIME-MAJOR: linearmodels 7.0's
    `PanelData` orders its time axis by first appearance in row order, so
    an entity-major EmplUK frame (whose first firm lacks 1976) puts 1976
    after 1983, and `FirstDifferenceOLS` then differences non-adjacent
    years (808 rows instead of 891) and the Driscoll-Kraay lags run in the
    wrong order. The script asserts the sorted axis for every frame.
  * The FD-with-intercept variant (`panel_fd(constant=True)`, the plm
    `model="fd"` / Stata `regress D.y D.x` default) is pinned against
    linearmodels `PooledOLS` on the hand-differenced frame (calendar
    adjacency: a row needs t and t - 1 observed); the same leg WITHOUT the
    constant is asserted equal to `FirstDifferenceOLS` (second leg).
  * Second legs asserted here (not stored): statsmodels OLS on the entity
    means equals `BetweenOLS` and its HC1 equals `"robust"`; the between
    `"clustered"` covariance equals `"robust"` (one row per cluster);
    Fama-MacBeth's standard error equals sd(beta_t, ddof=1)/sqrt(T_v);
    `PooledOLS` equals `RandomEffects` when sigma2_effects is truncated at
    0; `small_sample` is a no-op when every entity has the same count; and
    the Mundlak / correlated-random-effects identity — random effects on
    [const, x, mean_i(x)] reproduces `PanelOLS(entity_effects=True)`
    slopes (stored with the cluster-robust Wald on the mean coefficients).
  * DOCUMENTED divergences (stored under `divergences`, pinned by the Rust
    property tests): with an interior period missing for EVERYONE
    linearmodels differences across the hole and lags its Driscoll-Kraay
    scores over the present periods, where tsecon uses calendar adjacency
    and calendar lags. The calendar answers stored there are a NumPy
    transcription of the documented formulas (a documented-formula leg,
    not an independent package).
  * linearmodels CRASHES (numpy LinAlgError) on `small_sample=True` when
    the between design is rank-deficient on an unbalanced panel (EmplUK
    plus year dummies); tsecon refuses that combination by name, so it has
    no golden — the script asserts the crash.

Data:

  * `empluk` — the Arellano-Bond (1991) EmplUK panel read from
    `fixtures/panel_unbalanced.json` (`datasets.empluk`: log employment,
    log wage, log capital, the N x T mask; 140 firms, 1976-1984, 1031
    firm-years). `empluk_yeardummies` adds the 8 dummies for 1977-1984
    (the between design is then rank 7 of 11); `empluk_mundlak` adds the
    entity means of log wage and log capital over each firm's observed
    years.
  * `ragged` — seeded (20261008) N = 40, T = 12: entry/exit windows plus
    10% internal gaps, a random effect independent of the regressors.
    `ragged_z` adds a time-invariant regressor z_i (seed 99) to the
    outcome and the design.
  * `balanced` — the same DGP at seed 7, N = 25, T = 8, every cell
    observed.
  * `re_truncated` — seed 0, N = 30, T = 6, a negative intraclass
    correlation, so the Swamy-Arora effect variance is truncated at 0.
  * `re_common_trend` — seed 11, N = 20, T = 8, balanced, with a common
    linear trend among the regressors (the between design is collinear
    with the constant; the within and GLS steps identify the trend).
  * `fm_long` — seed 2026, N = 60, T = 48, persistent (AR(1)) period
    coefficients, 5% internal gaps, and period 20 observing only 2
    entities (< K = 3), so Fama-MacBeth skips it.
"""
import json
import platform
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import statsmodels
import statsmodels.api as sm
import linearmodels
from linearmodels.panel import (
    BetweenOLS,
    FamaMacBeth,
    FirstDifferenceOLS,
    PanelOLS,
    PooledOLS,
    RandomEffects,
)
from linearmodels.panel.data import PanelData as LmPanelData

warnings.simplefilter("ignore")

OUT = Path(__file__).parent
META = {
    "linearmodels": linearmodels.__version__,
    "numpy": np.__version__,
    "pandas": pd.__version__,
    "scipy": scipy.__version__,
    "statsmodels": statsmodels.__version__,
    "python": platform.python_version(),
}
BW = 4
COVS = [
    ("nonrobust", {"cov_type": "unadjusted"}),
    ("hc1", {"cov_type": "robust"}),
    ("cluster", {"cov_type": "clustered", "cluster_entity": True}),
    ("driscoll_kraay", {"cov_type": "kernel", "kernel": "bartlett", "bandwidth": BW}),
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def arr(rows):
    return np.array([[np.nan if v is None else v for v in r] for r in rows], dtype=float)


def to_json_array(a, mask=None):
    """N x T array with NaN (or masked-out cells) -> null."""
    out = []
    for i, row in enumerate(a):
        out.append([
            None if (mask is not None and not mask[i, t]) or not np.isfinite(v) else float(v)
            for t, v in enumerate(row)
        ])
    return out


def rel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300)))


def frame(y, regs, mask, const):
    """(entity, time)-indexed frame of the observed cells, TIME-MAJOR row
    order so linearmodels' first-appearance time axis is the calendar."""
    ent, tim = np.nonzero(mask)
    df = pd.DataFrame({"entity": ent, "time": tim, "y": y[mask]})
    if const:
        df["const"] = 1.0
    for name, x in regs:
        df[name] = x[mask]
    df = df.sort_values(["time", "entity"], kind="stable").set_index(["entity", "time"])
    axis = list(LmPanelData(df[["y"]]).panel.major_axis)
    assert axis == sorted(axis), axis
    return df


def pack(r, names):
    cov = r.cov.loc[names, names].to_numpy()
    return {
        "params": [float(r.params[n]) for n in names],
        "bse": [float(r.std_errors[n]) for n in names],
        "tvalues": [float(r.tstats[n]) for n in names],
        "cov": cov.tolist(),
    }


def punch_mask(rng, n, t_len, hole_rate, min_cells):
    while True:
        mask = np.zeros((n, t_len), dtype=bool)
        for i in range(n):
            a = rng.integers(0, 4)
            b = t_len - rng.integers(0, 4)
            mask[i, a:b] = True
        mask &= ~(rng.random((n, t_len)) < hole_rate)
        if mask.sum(axis=1).min() >= min_cells and mask.sum(axis=0).min() >= 3:
            return mask


# ---------------------------------------------------------------------------
# datasets
# ---------------------------------------------------------------------------
def empluk():
    ds = json.loads((OUT / "panel_unbalanced.json").read_text())["datasets"]["empluk"]
    mask = np.array(ds["mask"], dtype=bool)
    return arr(ds["log_emp"]), [("log_wage", arr(ds["log_wage"])), ("log_capital", arr(ds["log_capital"]))], mask


def year_dummies(mask):
    """One N x T 0/1 dummy per calendar period after the first."""
    n, t_len = mask.shape
    return [(f"d{c}", np.tile((np.arange(t_len) == c).astype(float), (n, 1))) for c in range(1, t_len)]


def entity_means(x, mask):
    """N x T array holding each entity's mean of x over its observed cells."""
    m = np.where(mask, x, 0.0).sum(1) / mask.sum(1)
    return np.repeat(m[:, None], mask.shape[1], axis=1)


def ragged(seed=20261008, n=40, t_len=12):
    rng = np.random.default_rng(seed)
    u = rng.normal(0, 1.0, n)  # random effect, independent of x
    x1 = rng.normal(0, 1, (n, t_len)) + 0.5 * rng.normal(0, 1, n)[:, None]
    x2 = rng.normal(2, 1, (n, t_len))
    e = rng.normal(0, 0.8, (n, t_len))
    y = 1.0 + 0.7 * x1 - 0.4 * x2 + u[:, None] + e
    mask = punch_mask(rng, n, t_len, 0.1, 4)
    return y, [("x1", x1), ("x2", x2)], mask


def ragged_z():
    y, regs, mask = ragged()
    rng = np.random.default_rng(99)
    z = np.repeat(rng.normal(0, 1, mask.shape[0])[:, None], mask.shape[1], axis=1)
    return y + 0.5 * z, regs + [("z", z)], mask


def balanced():
    y, regs, _ = ragged(seed=7, n=25, t_len=8)
    return y, regs, np.ones((25, 8), dtype=bool)


def re_truncated():
    rng = np.random.default_rng(0)
    n, t_len = 30, 6
    x = rng.normal(0, 1, (n, t_len))
    eps = rng.normal(0, 1, (n, t_len))
    e = eps - 0.8 * eps.mean(1, keepdims=True)  # negative intraclass correlation
    y = 1.0 + 0.5 * x + e
    return y, [("x", x)], np.ones((n, t_len), dtype=bool)


def re_common_trend():
    rng = np.random.default_rng(11)
    n, t_len = 20, 8
    trend = np.tile(np.arange(t_len, dtype=float), (n, 1))
    x = rng.normal(0, 1, (n, t_len))
    y = 1.0 + 0.3 * x + 0.1 * trend + rng.normal(0, 1, n)[:, None] + rng.normal(0, 1, (n, t_len))
    return y, [("x", x), ("trend", trend)], np.ones((n, t_len), dtype=bool)


def fm_long():
    rng = np.random.default_rng(2026)
    n, t_len = 60, 48
    beta_bar = np.array([0.5, 1.0, -0.5])
    dev = np.zeros((t_len, 3))
    for t in range(1, t_len):
        dev[t] = 0.7 * dev[t - 1] + 0.2 * rng.standard_normal(3)
    x1 = rng.normal(0, 1, (n, t_len)) + rng.normal(0, 0.5, n)[:, None]
    x2 = rng.normal(0, 1, (n, t_len))
    firm = rng.normal(0, 0.5, n)[:, None]  # persistent firm effect
    b = beta_bar + dev
    y = b[:, 0][None, :] + b[:, 1][None, :] * x1 + b[:, 2][None, :] * x2 + firm + rng.normal(0, 1, (n, t_len))
    mask = ~(rng.random((n, t_len)) < 0.05)
    mask[:, 20] = False
    mask[:2, 20] = True  # 2 entities < K = 3: the period is skipped
    return y, [("x1", x1), ("x2", x2)], mask


# ---------------------------------------------------------------------------
# estimators
# ---------------------------------------------------------------------------
def fit_re(y, regs, mask, const, small_sample):
    df = frame(y, regs, mask, const)
    names = (["const"] if const else []) + [nm for nm, _ in regs]
    mod = RandomEffects(df["y"], df[names])
    out = {}
    for key, kw in COVS:
        r = mod.fit(small_sample=small_sample, **kw)
        out[key] = pack(r, names)
    vd = r.variance_decomposition
    theta = np.full(mask.shape[0], np.nan)
    th = r.theta
    for ent, val in zip(th.index, th.to_numpy()[:, 0]):
        theta[int(ent)] = float(val)
    res = {
        "se": out,
        "nobs": int(r.nobs),
        "n_entities": int((mask.sum(1) > 0).sum()),
        "df_resid": int(r.df_resid),
        "sigma2_eps": float(vd["Residual"]),
        "sigma2_effects": float(vd["Effects"]),
        "rho": float(vd["Percent due to Effects"]),
        "theta": [None if not np.isfinite(v) else float(v) for v in theta],
        "names": names,
    }
    assert res["df_resid"] == res["nobs"] - len(names)
    return res, mod


def fit_be(y, regs, mask, const, reweight):
    df = frame(y, regs, mask, const)
    names = (["const"] if const else []) + [nm for nm, _ in regs]
    mod = BetweenOLS(df["y"], df[names])
    out = {}
    for key, kw in COVS[:3]:
        if key == "cluster":
            ent = pd.DataFrame({"c": df.index.get_level_values(0).to_numpy()}, index=df.index)
            r = mod.fit(reweight=reweight, cov_type="clustered", clusters=ent)
        else:
            r = mod.fit(reweight=reweight, **kw)
        out[key] = pack(r, names)
    # second leg: statsmodels OLS on the (sqrt-weighted) entity means; HC1 == robust,
    # and clustering by entity == robust (one row per cluster)
    present = mask.sum(1) > 0
    ti = mask.sum(1)[present].astype(float)
    ybar = np.where(mask, y, 0.0).sum(1)[present] / ti
    xbar = [np.ones(present.sum())] if const else []
    xbar += [np.where(mask, x, 0.0).sum(1)[present] / ti for _, x in regs]
    xbar = np.column_stack(xbar)
    rw = np.sqrt(ti / ti.mean()) if reweight else np.ones_like(ti)
    sm_fit = sm.OLS(ybar * rw, xbar * rw[:, None]).fit(cov_type="HC1")
    assert rel(sm_fit.params, out["hc1"]["params"]) < 1e-10
    assert rel(sm_fit.cov_params(), out["hc1"]["cov"]) < 1e-9
    assert rel(out["cluster"]["cov"], out["hc1"]["cov"]) < 1e-9
    return {"se": out, "nobs": int(r.nobs), "n_entities": int(present.sum()),
            "df_resid": int(r.df_resid), "names": names}


def hand_fd(y, regs, mask):
    """Calendar-adjacent first differences: rows (i, t), t >= 1, with t and
    t - 1 both observed, entity-major."""
    n, t_len = mask.shape
    rows = [(i, t) for i in range(n) for t in range(1, t_len) if mask[i, t] and mask[i, t - 1]]
    rows = np.array(rows)
    dy = np.array([y[i, t] - y[i, t - 1] for i, t in rows])
    dx = [(nm, np.array([x[i, t] - x[i, t - 1] for i, t in rows])) for nm, x in regs]
    return rows, dy, dx


def fd_frame(rows, dy, dx, const):
    df = pd.DataFrame({"entity": rows[:, 0], "time": rows[:, 1], "dy": dy})
    if const:
        df["const"] = 1.0
    for nm, v in dx:
        df[nm] = v
    df = df.sort_values(["time", "entity"], kind="stable").set_index(["entity", "time"])
    axis = list(LmPanelData(df[["dy"]]).panel.major_axis)
    assert axis == sorted(axis), axis
    return df


def fit_fd(y, regs, mask, const):
    names = (["const"] if const else []) + [nm for nm, _ in regs]
    rows, dy, dx = hand_fd(y, regs, mask)
    # every t >= 1 carries a difference row, so linearmodels' present-period
    # Driscoll-Kraay lags are the calendar ones
    assert sorted(set(rows[:, 1])) == list(range(1, mask.shape[1]))
    hdf = fd_frame(rows, dy, dx, const)
    pmod = PooledOLS(hdf["dy"], hdf[names])
    pooled = {key: pack(pmod.fit(**kw), names) for key, kw in COVS}
    if not const:
        df = frame(y, regs, mask, False)
        mod = FirstDifferenceOLS(df["y"], df[names])
        out = {}
        for key, kw in COVS:
            r = mod.fit(**kw)
            out[key] = pack(r, names)
            # second leg: PooledOLS on the hand-differenced frame
            assert rel(out[key]["params"], pooled[key]["params"]) < 1e-10, key
            assert rel(out[key]["cov"], pooled[key]["cov"]) < 1e-9, key
        nobs = int(r.nobs)
        assert nobs == len(rows), (nobs, len(rows))
    else:
        out = pooled
        nobs = len(rows)
    return {"se": out, "nobs": nobs, "n_entities": int(len(set(rows[:, 0]))),
            "df_resid": nobs - len(names), "names": names}


def fit_fm(y, regs, mask, const):
    df = frame(y, regs, mask, const)
    names = (["const"] if const else []) + [nm for nm, _ in regs]
    mod = FamaMacBeth(df["y"], df[names])
    out = {}
    for key, kw in [("fama_macbeth", {"cov_type": "unadjusted"}),
                    ("hac", {"cov_type": "kernel", "kernel": "bartlett", "bandwidth": BW})]:
        r = mod.fit(**kw)
        out[key] = pack(r, names)
    ap = r.all_params.reindex(range(mask.shape[1]))[names].to_numpy()
    used = np.isfinite(ap).all(1)
    b = ap[used]
    tv = len(b)
    # second leg: the closed-form Fama-MacBeth standard error
    assert rel(np.std(b, axis=0, ddof=1) / np.sqrt(tv), out["fama_macbeth"]["bse"]) < 1e-10
    assert rel(b.mean(0), out["fama_macbeth"]["params"]) < 1e-12
    return {
        "se": out,
        "nobs": int(r.nobs),
        "n_periods_used": int(tv),
        "periods_used": [bool(v) for v in used],
        "params_per_period": [[None if not np.isfinite(v) else float(v) for v in row] for row in ap],
        "names": names,
    }


# ---------------------------------------------------------------------------
# documented-formula transcription for the divergence cases
# ---------------------------------------------------------------------------
def calendar_dk_cov(x, e, periods, n_periods, bw):
    n, k = x.shape
    s = x * e[:, None]
    a = np.zeros((n_periods, k))
    for r, p in enumerate(periods):
        a[p] += s[r]
    meat = a.T @ a
    for j in range(1, int(bw) + 1):
        w = 1 - j / (bw + 1)
        g = a[j:].T @ a[:-j]
        meat += w * (g + g.T)
    bread = np.linalg.inv(x.T @ x)
    return n / (n - k) * bread @ meat @ bread


def np_re_calendar_dk(y, regs, mask, bw):
    """RE (small_sample=False, constant) with the calendar Driscoll-Kraay —
    the documented-formula transcription of tsecon's convention."""
    cells = np.argwhere(mask)
    ent = cells[:, 0]
    yv = y[mask]
    xv = np.column_stack([np.ones(len(yv))] + [x[mask] for _, x in regs])
    ents = np.unique(ent)
    pos = np.searchsorted(ents, ent)
    N, n, kk = len(ents), len(yv), xv.shape[1]
    ti = np.bincount(pos).astype(float)
    ybar = np.bincount(pos, yv) / ti
    xbar = np.column_stack([np.bincount(pos, xv[:, j]) / ti for j in range(kk)])
    yd, xd = yv - ybar[pos], xv - xbar[pos]
    ew = yd - xd @ np.linalg.lstsq(xd, yd, rcond=None)[0]
    s2e = ew @ ew / (n - kk - N + 1)
    ub = ybar - xbar @ np.linalg.lstsq(xbar, ybar, rcond=None)[0]
    tbar = N / np.sum(1 / ti)
    s2u = max(0.0, ub @ ub / (N - kk) - s2e / tbar)
    theta = 1 - np.sqrt(s2e / (ti * s2u + s2e))
    ys = yv - theta[pos] * ybar[pos]
    xs = xv - theta[pos][:, None] * xbar[pos]
    b = np.linalg.lstsq(xs, ys, rcond=None)[0]
    e = ys - xs @ b
    cov = calendar_dk_cov(xs, e, cells[:, 1], mask.shape[1], bw)
    return b, cov


def main():
    fx = {"_meta": META, "datasets": {}, "re_cases": [], "be_cases": [], "fd_cases": [],
          "fm_cases": [], "mundlak": {}, "divergences": {}}

    data = {
        "empluk": empluk(),
        "ragged": ragged(),
        "ragged_z": ragged_z(),
        "balanced": balanced(),
        "re_truncated": re_truncated(),
        "re_common_trend": re_common_trend(),
        "fm_long": fm_long(),
    }
    ey, eregs, emask = data["empluk"]
    data["empluk_yeardummies"] = (ey, eregs + year_dummies(emask), emask)
    data["empluk_mundlak"] = (
        ey,
        eregs + [("mean_log_wage", entity_means(eregs[0][1], emask)),
                 ("mean_log_capital", entity_means(eregs[1][1], emask))],
        emask,
    )

    fx["datasets"]["empluk"] = {
        "source": "fixtures/panel_unbalanced.json datasets.empluk (log_emp on log_wage, "
                  "log_capital)",
    }
    fx["datasets"]["empluk_yeardummies"] = {
        "source": "empluk plus one 0/1 dummy per calendar period after the first (1977-1984), "
                  "regressors in the order log_wage, log_capital, d1..d8",
    }
    fx["datasets"]["empluk_mundlak"] = {
        "source": "empluk plus each firm's mean of log_wage and of log_capital over its observed "
                  "years, regressors in the order log_wage, log_capital, mean_log_wage, "
                  "mean_log_capital",
    }
    for name in ("ragged", "ragged_z", "balanced", "re_truncated", "re_common_trend", "fm_long"):
        y, regs, mask = data[name]
        fx["datasets"][name] = {
            "n_entities": int(mask.shape[0]),
            "n_periods": int(mask.shape[1]),
            "mask": mask.astype(int).tolist(),
            "outcome": to_json_array(y, mask),
            "regressor_names": [nm for nm, _ in regs],
            "regressors": [to_json_array(x, mask) for _, x in regs],
        }

    # ---------------- random effects
    re_specs = [
        ("empluk", True, False), ("empluk", True, True),
        ("ragged", True, False), ("ragged", True, True),
        ("balanced", True, False), ("balanced", True, True),
        ("ragged_z", True, False), ("ragged_z", True, True),
        ("re_truncated", True, False), ("re_common_trend", True, False),
        ("empluk_yeardummies", True, False), ("empluk_mundlak", True, False),
    ]
    for ds, const, ss in re_specs:
        y, regs, mask = data[ds]
        res, mod = fit_re(y, regs, mask, const, ss)
        fx["re_cases"].append({"name": f"{ds}/ss={ss}", "dataset": ds, "regressors": "all",
                               "constant": const, "small_sample": ss, **res})
    # constant=False (the n - K - N + 1 quirk without a constant) on the ragged panel
    y, regs, mask = data["ragged"]
    res, _ = fit_re(y, regs, mask, False, False)
    fx["re_cases"].append({"name": "ragged/no_constant", "dataset": "ragged", "regressors": "all",
                           "constant": False, "small_sample": False, **res})
    # intercept-only RE on EmplUK: the variance-components / ICC model
    res, _ = fit_re(ey, [], emask, True, False)
    fx["re_cases"].append({"name": "empluk/intercept_only", "dataset": "empluk", "regressors": "none",
                           "constant": True, "small_sample": False, **res})

    by = {c["name"]: c for c in fx["re_cases"]}
    # small_sample is a no-op on a balanced panel
    for key, _ in COVS:
        assert by["balanced/ss=True"]["se"][key]["params"] == by["balanced/ss=False"]["se"][key]["params"]
    # truncation: sigma2_effects == 0 exactly, theta == 0, RE == pooled OLS
    trunc = by["re_truncated/ss=False"]
    assert trunc["sigma2_effects"] == 0.0 and all(t == 0.0 for t in trunc["theta"]), trunc["sigma2_effects"]
    y, regs, mask = data["re_truncated"]
    df = frame(y, regs, mask, True)
    pooled = PooledOLS(df["y"], df[["const", "x"]]).fit()
    assert rel(pooled.params.values, trunc["se"]["nonrobust"]["params"]) < 1e-12
    # small_sample=True crashes linearmodels on the rank-deficient between design
    y, regs, mask = data["empluk_yeardummies"]
    df = frame(y, regs, mask, True)
    names = ["const"] + [nm for nm, _ in regs]
    try:
        RandomEffects(df["y"], df[names]).fit(small_sample=True)
        raise AssertionError("expected linearmodels to raise on singular M")
    except np.linalg.LinAlgError:
        pass
    xbar = df[names].groupby(level=0).mean().to_numpy()
    fx["datasets"]["empluk_yeardummies"]["between_rank"] = int(np.linalg.matrix_rank(xbar))

    # ---------------- Mundlak / CRE identity (RE on [const, x, xbar_i] == FE slopes)
    y, regs, mask = data["empluk"]
    df = frame(y, regs, mask, False)
    fe = PanelOLS(df["y"], df[["log_wage", "log_capital"]], entity_effects=True).fit()
    mund = by["empluk_mundlak/ss=False"]
    gap = rel(mund["se"]["cluster"]["params"][1:3], fe.params.values)
    assert gap < 1e-12, gap
    b = np.array(mund["se"]["cluster"]["params"][3:5])
    v = np.array(mund["se"]["cluster"]["cov"])[3:5, 3:5]
    fx["mundlak"] = {
        "dataset": "empluk_mundlak",
        "fe_slopes": [float(v_) for v_ in fe.params.values],
        "re_slopes_max_rel_gap_to_fe": gap,
        "cluster_wald_mean_coefficients": float(b @ np.linalg.solve(v, b)),
        "wald_df": 2,
    }

    # ---------------- between
    for ds in ("empluk", "ragged", "balanced"):
        y, regs, mask = data[ds]
        for rw in (False, True):
            res = fit_be(y, regs, mask, True, rw)
            fx["be_cases"].append({"name": f"{ds}/reweight={rw}", "dataset": ds, "regressors": "all",
                                   "constant": True, "reweight": rw, **res})
    res = fit_be(ey, [], emask, True, False)
    fx["be_cases"].append({"name": "empluk/intercept_only", "dataset": "empluk", "regressors": "none",
                           "constant": True, "reweight": False, **res})

    # ---------------- first differences
    for ds in ("empluk", "ragged", "balanced"):
        y, regs, mask = data[ds]
        for const in (False, True):
            if const and ds == "balanced":
                continue
            res = fit_fd(y, regs, mask, const)
            fx["fd_cases"].append({"name": f"{ds}/constant={const}", "dataset": ds, "regressors": "all",
                                   "constant": const, **res})
    assert by and fx["fd_cases"][0]["nobs"] == 891

    # ---------------- Fama-MacBeth
    for ds in ("empluk", "ragged", "fm_long"):
        y, regs, mask = data[ds]
        res = fit_fm(y, regs, mask, True)
        fx["fm_cases"].append({"name": ds, "dataset": ds, "regressors": "all", "constant": True, **res})
    assert fx["fm_cases"][2]["periods_used"][20] is False

    # ---------------- documented divergences: a period nobody observes
    y, regs, mask = data["empluk"]
    hole = mask.copy()
    hole[:, 4] = False  # 1980 dropped for everyone
    df = frame(y, regs, hole, False)
    lm_fd = FirstDifferenceOLS(df["y"], df[["log_wage", "log_capital"]]).fit()
    rows, dy, dx = hand_fd(y, regs, hole)
    dxm = np.column_stack([v for _, v in dx])
    cal = np.linalg.lstsq(dxm, dy, rcond=None)[0]
    dfc = frame(y, regs, hole, True)
    lm_re = RandomEffects(dfc["y"], dfc[["const", "log_wage", "log_capital"]]).fit(
        cov_type="kernel", kernel="bartlett", bandwidth=BW)
    b_cal, cov_cal = np_re_calendar_dk(y, regs, hole, BW)
    assert rel(b_cal, lm_re.params.values) < 1e-10
    fx["divergences"]["empluk_without_1980"] = {
        "dataset": "empluk with calendar period 4 (1980) unobserved for every firm",
        "dropped_period": 4,
        "fd_linearmodels_nobs": int(lm_fd.nobs),
        "fd_linearmodels_params": [float(v_) for v_ in lm_fd.params.values],
        "fd_calendar_nobs": int(len(rows)),
        "fd_calendar_params": [float(v_) for v_ in cal],
        "re_params": [float(v_) for v_ in lm_re.params.values],
        "re_dk_linearmodels_bse": [float(v_) for v_ in lm_re.std_errors.values],
        "re_dk_calendar_bse": [float(v_) for v_ in np.sqrt(np.diag(cov_cal))],
        "re_dk_calendar_cov": cov_cal.tolist(),
    }

    path = OUT / "panel_static.json"
    path.write_text(json.dumps(fx, separators=(",", ":")))
    print(f"wrote {path} ({path.stat().st_size} bytes)")
    for c in fx["re_cases"]:
        print("RE", c["name"], np.round(c["se"]["cluster"]["params"][:4], 8), "s2u", c["sigma2_effects"])
    for c in fx["be_cases"]:
        print("BE", c["name"], np.round(c["se"]["hc1"]["bse"], 8))
    for c in fx["fd_cases"]:
        print("FD", c["name"], c["nobs"], np.round(c["se"]["cluster"]["params"], 8), np.round(c["se"]["cluster"]["bse"], 8))
    for c in fx["fm_cases"]:
        print("FM", c["name"], c["n_periods_used"], np.round(c["se"]["fama_macbeth"]["params"], 8),
              np.round(c["se"]["hac"]["bse"], 8))
    print("Mundlak", fx["mundlak"])
    print("divergences", {k: v for k, v in fx["divergences"]["empluk_without_1980"].items() if "cov" not in k})


if __name__ == "__main__":
    main()
