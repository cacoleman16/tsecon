"""Golden fixtures for `panel_distributed_lag` — the distributed-lag panel
regression of the climate-impact literature (Dell-Jones-Olken 2012,
AEJ:Macro; Burke-Hsiang-Miguel 2015, Nature):

    y_it = sum_{l=0..L} beta_l x_{i,t-l} [+ sum_l gamma_l x^2_{i,t-l}]
           + alpha_i + delta_t [+ g_i t] + e_it

Run with the project venv:
    .venv/bin/python fixtures/generate_panel_dl_fixtures.py

Reference and grade (honest):

  * INDEPENDENT-PACKAGE leg — every case's slopes, standard errors,
    t-statistics and full parameter covariance come from linearmodels'
    `PanelOLS` (Kevin Sheppard's package, version recorded in `_meta`) on
    the explicitly lagged, balanced design, with `entity_effects` /
    `time_effects` set per case and the entity-trend variant built from
    explicit entity x trend regressors (N - 1 columns when time effects
    are present: the common linear trend already lies in the span of the
    time dummies, so the N-th column is collinear; N columns otherwise).
    Three covariance estimators per case: `cov_type="unadjusted"`,
    `"clustered"` by entity, and `"kernel"` (Bartlett, bandwidth 4 —
    the lag-truncation value `panel_fe` uses). The Rust golden test
    (`crates/tsecon-panel/tests/panel_dl_golden.rs`) pins tsecon at
    1e-10 relative; the inputs are stored at full double precision so the
    comparison is a from-scratch fit on identical numbers.

  * DOCUMENTED-FORMULA leg — the cumulative effect, its delta-method
    standard error, the 95% interval, and (quadratic case) the marginal
    effect and turning point are transcribed here in NumPy from the
    formulas the crate documents, evaluated on linearmodels' covariance:

        idx(p, l)  = (p - 1) * (L + 1) + l          (one regressor; power p, lag l)
        B_p        = sum_l beta[idx(p, l)]
        se(B_p)    = sqrt(1' V_p 1),   V_p the (L+1)x(L+1) block of power p
        ci         = B_p -/+ 1.959963984540054 * se(B_p)
        marginal(x)= B_1 + 2 B_2 x
        grad_m     = [1, ..., 1 | 2x, ..., 2x]     (2(L+1) entries, powers 1 | 2)
        se_m       = sqrt(grad_m' V grad_m),  V the joint 2(L+1) block
        x*         = -B_1 / (2 B_2)
        grad_tp    = [-1/(2 B_2), ... | B_1/(2 B_2^2), ...]
        se(x*)     = sqrt(grad_tp' V grad_tp)

    This leg checks that the crate's delta method is the documented one;
    it is NOT an independent authority for the delta method itself
    (there is no third-party panel distributed-lag package in Python
    that reports cumulative effects). The statistical claims — that the
    cumulative effect recovers the long-run impact and that its interval
    covers at the nominal rate under clustering / Driscoll-Kraay — are
    measured separately in the seeded Monte Carlo of
    `crates/tsecon-panel/tests/panel_dl_properties.rs`.

Data-generating process (seed 20260910): N = 24 entities, T = 36 periods;
a temperature-like regressor x_it = mu_i + 0.3 (x_{i,t-1} - mu_i) + N(0,1)
with entity climatologies mu_i ~ N(20, 4); a second precipitation-like
regressor for the two-regressor case; entity effects alpha_i ~ N(0, 1),
period effects delta_t ~ N(0, 0.5), entity trends g_i ~ N(0, 0.02) in the
trend case; errors AR(1) with rho = 0.4 within each entity plus a common
period component (so entity clustering and Driscoll-Kraay differ). True
lag polynomials: linear beta = (1.0, -0.6, 0.3, -0.1) truncated to L + 1
terms; quadratic beta = (0.8, -0.4, 0.2), gamma = (-0.02, 0.01, -0.005).
The fixture stores derived numbers and the simulated inputs only.
"""
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import linearmodels
from linearmodels.panel import PanelOLS

OUT = Path(__file__).parent
META = {
    "linearmodels": linearmodels.__version__,
    "numpy": np.__version__,
    "pandas": pd.__version__,
    "python": platform.python_version(),
}
Z975 = 1.959963984540054
SEED = 20260910
N, T = 24, 36
BW = 4


def simulate(rng):
    mu = rng.normal(20.0, 4.0, N)
    x = np.empty((N, T))
    for i in range(N):
        x[i, 0] = mu[i] + rng.standard_normal()
        for t in range(1, T):
            x[i, t] = mu[i] + 0.3 * (x[i, t - 1] - mu[i]) + rng.standard_normal()
    z = rng.normal(100.0, 20.0, N)[:, None] + rng.normal(0.0, 10.0, (N, T))
    alpha = rng.normal(0.0, 1.0, N)
    delta = rng.normal(0.0, 0.5, T)
    gtrend = rng.normal(0.0, 0.02, N)
    common = rng.normal(0.0, 0.5, T)
    e = np.empty((N, T))
    for i in range(N):
        e[i, 0] = rng.standard_normal()
        for t in range(1, T):
            e[i, t] = 0.4 * e[i, t - 1] + rng.standard_normal()
    e = e + common[None, :]
    return x, z, alpha, delta, gtrend, e


def build_outcome(x, z, alpha, delta, gtrend, e, beta, gamma=None, beta_z=None, trends=False):
    y = alpha[:, None] + delta[None, :] + e
    if trends:
        y = y + gtrend[:, None] * np.arange(T)[None, :]
    lags = len(beta)
    for l, b in enumerate(beta):
        y[:, lags - 1:] += b * x[:, lags - 1 - l:T - l]
    if gamma is not None:
        for l, g in enumerate(gamma):
            y[:, lags - 1:] += g * x[:, lags - 1 - l:T - l] ** 2
    if beta_z is not None:
        for l, b in enumerate(beta_z):
            y[:, lags - 1:] += b * z[:, lags - 1 - l:T - l]
    return y


def lag_design(y, regressors, L, powers):
    """Balanced lagged design: rows (entity, period L..T), columns
    regressor-major then power then lag — the crate's column order."""
    tu = T - L
    cols, names = [], []
    for j, (name, x) in enumerate(regressors):
        for p in range(1, powers + 1):
            for l in range(L + 1):
                cols.append((x[:, L - l:T - l] ** p).reshape(-1))
                names.append(f"{name}_L{l}" if p == 1 else f"{name}^2_L{l}")
    ent = np.repeat(np.arange(N), tu)
    tim = np.tile(np.arange(tu), N)
    df = pd.DataFrame(np.column_stack(cols), columns=names)
    df["y"] = y[:, L:].reshape(-1)
    df["entity"] = ent
    df["time"] = tim
    return df.set_index(["entity", "time"]), names


def fit_case(df, names, entity, time, trends):
    exog = df[names].copy()
    if trends:
        # Explicit entity x trend columns; drop one under time effects
        # (the common trend is in the span of the time dummies).
        ent = df.index.get_level_values("entity").to_numpy()
        tim = df.index.get_level_values("time").to_numpy().astype(float)
        n_cols = N - 1 if time else N
        for i in range(n_cols):
            exog[f"trend_{i}"] = (ent == i) * tim
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


def main():
    rng = np.random.default_rng(SEED)
    x, z, alpha, delta, gtrend, e = simulate(rng)
    beta4 = [1.0, -0.6, 0.3, -0.1]
    beta_q = [0.8, -0.4, 0.2]
    gamma_q = [-0.02, 0.01, -0.005]
    beta_z = [0.05, -0.02]
    y_lin = build_outcome(x, z, alpha, delta, gtrend, e, beta4)
    y_quad = build_outcome(x, z, alpha, delta, gtrend, e, beta_q, gamma=gamma_q)
    y_trend = build_outcome(x, z, alpha, delta, gtrend, e, beta4[:3], trends=True)
    y_two = build_outcome(x, z, alpha, delta, gtrend, e, beta4[:2], beta_z=beta_z)

    cases = [
        # name, outcome, regressors, L, powers, entity, time, trends, eval_points
        ("linear_L3_twoway", y_lin, [("x", x)], 3, 1, True, True, False, None),
        ("linear_L0_twoway", y_lin, [("x", x)], 0, 1, True, True, False, None),
        ("linear_L1_entity_only", y_lin, [("x", x)], 1, 1, True, False, False, None),
        ("linear_L1_time_only", y_lin, [("x", x)], 1, 1, False, True, False, None),
        ("linear_L2_trends_twoway", y_trend, [("x", x)], 2, 1, True, True, True, None),
        ("linear_L1_trends_entity_only", y_trend, [("x", x)], 1, 1, True, False, True, None),
        ("linear_L1_two_regressors", y_two, [("x", x), ("z", z)], 1, 1, True, True, False, None),
        ("quadratic_L2_twoway", y_quad, [("x", x)], 2, 2, True, True, False, [15.0, 20.0, 25.0]),
        ("quadratic_L1_mean_eval", y_quad, [("x", x)], 1, 2, True, True, False, "mean"),
    ]
    outcome_key = {
        "linear_L3_twoway": "y_lin", "linear_L0_twoway": "y_lin",
        "linear_L1_entity_only": "y_lin", "linear_L1_time_only": "y_lin",
        "linear_L2_trends_twoway": "y_trend", "linear_L1_trends_entity_only": "y_trend",
        "linear_L1_two_regressors": "y_two", "quadratic_L2_twoway": "y_quad",
        "quadratic_L1_mean_eval": "y_quad",
    }
    fx_cases = []
    for name, y, regs, L, powers, entity, time, trends, ev in cases:
        df, names = lag_design(y, regs, L, powers)
        fits = fit_case(df, names, entity, time, trends)
        if ev == "mean":
            ev_used = [float(regs[0][1].mean())]
        else:
            ev_used = ev
        derived = {}
        if len(regs) == 1:
            for key, f in fits.items():
                derived[key] = delta_method(f["params"], f["cov"], L, powers, ev_used or [])
        fx_cases.append({
            "name": name,
            "outcome": outcome_key[name],
            "regressors": [r[0] for r in regs],
            "lags": L,
            "powers": powers,
            "entity_effects": entity,
            "time_effects": time,
            "entity_trends": trends,
            "eval_points": None if ev is None else ("mean" if ev == "mean" else ev),
            "eval_points_used": ev_used,
            "names": names,
            "bandwidth": BW,
            "fits": fits,
            "delta": derived,
        })

    out = {
        "_meta": META,
        "seed": SEED,
        "n_entities": N,
        "n_periods": T,
        "z975": Z975,
        "true": {"beta_linear": beta4, "beta_quadratic": beta_q, "gamma_quadratic": gamma_q,
                 "beta_z": beta_z},
        "inputs": {
            "x": x.tolist(),
            "z": z.tolist(),
            "y_lin": y_lin.tolist(),
            "y_quad": y_quad.tolist(),
            "y_trend": y_trend.tolist(),
            "y_two": y_two.tolist(),
        },
        "cases": fx_cases,
    }
    path = OUT / "panel_dl.json"
    path.write_text(json.dumps(out, separators=(",", ":")))
    print(f"wrote {path} ({path.stat().st_size / 1024:.0f} KB), {len(fx_cases)} cases")
    for c in fx_cases:
        f = c["fits"]["cluster_entity"]
        d = c["delta"].get("cluster_entity", {})
        print(f"  {c['name']:32s} nobs={f['nobs']:5d} df_resid={f['df_resid']:5d} "
              f"cum={d.get('cumulative_effect')} se={d.get('cumulative_se')}")


if __name__ == "__main__":
    main()
