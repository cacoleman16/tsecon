"""Experiment 1 — point-forecast horse race.

prophet_lite vs tsecon SARIMA (arima_fit + seasonal), tsecon theta, and
seasonal-naive / mean baselines, by rolling-origin (expanding-window)
evaluation with a full refit of every model at every origin.

Datasets
  (a) synthetic piecewise-trend + seasonal + outliers (prophet's home turf)
  (b) statsmodels co2, monthly means (interpolated) — strong stable seasonal
  (c) statsmodels macrodata realgdp growth (400*dlog) — NO seasonality,
      where a trend/seasonality model should LOSE to ARIMA.

tsecon.backtest only exposes its built-in forecasters (naive/drift/mean/
seasonal_naive/theta), so the loop here is manual; tsecon is still used
for arima_fit, theta_forecast, dm_test and the DGP-free metrics.

Metrics: RMSE/MAE at h in {1, 6, 12}; Diebold-Mariano (tsecon.dm_test,
squared loss, HLN correction) for the headline pairs.

Runtime: ~5 min.  Seeded; deterministic.
"""

from __future__ import annotations

import time

import numpy as np

import common as C
import tsecon
from prophet_lite import fit as prophet_fit

HS = (1, 6, 12)
H = 12


def snaive_forecast(train, h, period):
    reps = int(np.ceil(h / period))
    cyc = np.tile(train[-period:], reps)
    return cyc[:h]


def rolling_race(y, origins, models, period):
    """models: dict name -> callable(train) -> (H,) forecast.
    Returns errors[name][h-1] = list of (y_true - fc) across origins."""
    err = {name: [[] for _ in range(H)] for name in models}
    for o in origins:
        train = y[:o]
        for name, fn in models.items():
            fc = np.asarray(fn(train), float)
            for h in range(1, H + 1):
                err[name][h - 1].append(y[o + h - 1] - fc[h - 1])
    return {name: [np.array(v) for v in e] for name, e in err.items()}


def metric_rows(err, names):
    rows = []
    for name in names:
        row = [name]
        for h in HS:
            e = err[name][h - 1]
            row += [C.rmse(e), C.mae(e)]
        rows.append(row)
    return rows


def dm_rows(err, pairs):
    rows = []
    for (a, b) in pairs:
        for h in (1, 12):
            e1, e2 = err[a][h - 1], err[b][h - 1]
            try:
                d = tsecon.dm_test(e1, e2, h=h, loss="squared")
                rows.append([f"{a} vs {b}", h, "%.2f" % d["hln_stat"],
                             "%.4f" % d["p_value"],
                             "%+.3f" % d["mean_loss_diff"]])
            except ValueError:
                # tsecon.dm_test refuses when the rectangular-window LRV is
                # not positive (small n, long h) — fall back to a Bartlett
                # (Newey-West) t on the squared-loss differential, flagged.
                t, p = C.nw_tstat(e1 ** 2 - e2 ** 2, lags=h - 1)
                rows.append([f"{a} vs {b} (NW fallback)", h, "%.2f" % t,
                             "%.4f" % p,
                             "%+.3f" % float(np.mean(e1 ** 2 - e2 ** 2))])
    return rows


def make_models_seasonal(sarima_order, period, prophet_K, prophet_kw=None):
    prophet_kw = prophet_kw or {}

    def m_prophet(train):
        r = prophet_fit(train, seasonality=[(period, prophet_K)], **prophet_kw)
        return r.forecast(H, level=[0.8], n_draws=50, seed=0)["mean"]

    def m_sarima(train):
        p, d, q, P, D, Q = sarima_order
        r = tsecon.arima_fit(train, p=p, d=d, q=q,
                             seasonal=(P, D, Q, period), constant=False,
                             forecast_steps=H)
        return r["forecast_mean"]

    def m_theta(train):
        return tsecon.theta_forecast(train, H, period=period)

    def m_snaive(train):
        return snaive_forecast(train, H, period)

    return {"prophet_lite": m_prophet, "sarima": m_sarima,
            "theta": m_theta, "seasonal_naive": m_snaive}


def main():
    t_start = time.time()
    parts = []

    # ---------------- (a) synthetic home turf ------------------------------
    y, _ = C.piecewise_seasonal(240, seed=20260817)
    origins = list(range(120, 229, 2))          # 55 origins, expanding window
    models = make_models_seasonal((0, 1, 1, 0, 1, 1), 12, prophet_K=3)
    err_a = rolling_race(y, origins, models, 12)

    hdr = ["model", "RMSE h=1", "MAE h=1", "RMSE h=6", "MAE h=6",
           "RMSE h=12", "MAE h=12"]
    tab_a = C.md_table(hdr, metric_rows(err_a, models))
    dm_a = C.md_table(["pair (squared loss)", "h", "DM (HLN)", "p", "mean d"],
                      dm_rows(err_a, [("prophet_lite", "sarima"),
                                      ("prophet_lite", "theta")]))
    parts.append("### (a) Synthetic piecewise-trend + seasonal + outliers "
                 f"(T=240, {len(origins)} origins, expanding, refit each origin)\n\n"
                 + tab_a + "\n\n" + dm_a)

    # ---------------- (b) co2 monthly --------------------------------------
    import statsmodels.api as sm
    co2 = (sm.datasets.co2.load_pandas().data["co2"]
           .resample("MS").mean().interpolate().to_numpy())
    origins_b = list(range(400, 515, 6))        # 20 origins
    models_b = make_models_seasonal((0, 1, 1, 0, 1, 1), 12, prophet_K=5)
    err_b = rolling_race(co2, origins_b, models_b, 12)
    tab_b = C.md_table(hdr, metric_rows(err_b, models_b))
    dm_b = C.md_table(["pair (squared loss)", "h", "DM (HLN)", "p", "mean d"],
                      dm_rows(err_b, [("prophet_lite", "sarima"),
                                      ("prophet_lite", "theta")]))
    parts.append(f"### (b) CO2 monthly means, interpolated (T={len(co2)}, "
                 f"{len(origins_b)} origins; integer index + (12,5) Fourier "
                 "seasonality since calendar-monthly spacing is irregular "
                 "in days)\n\n" + tab_b + "\n\n" + dm_b
                 + "\n\nNote: only 20 origins (SARIMA+prophet refits are "
                 "expensive at T≈500), so the DM tests here are low-powered; "
                 "read signs, not significance.")

    # ---------------- (c) realgdp growth (no seasonality) ------------------
    mac = sm.datasets.macrodata.load_pandas().data
    g = 400.0 * np.diff(np.log(mac["realgdp"].to_numpy()))
    origins_c = list(range(120, 191, 1))        # 71 origins

    def m_prophet_g(train):
        r = prophet_fit(train)                   # integer index, trend only
        return r.forecast(H, level=[0.8], n_draws=50, seed=0)["mean"]

    def m_ar1(train):
        r = tsecon.arima_fit(train, p=1, d=0, q=0, constant=True,
                             forecast_steps=H)
        return r["forecast_mean"]

    def m_theta_g(train):
        return tsecon.theta_forecast(train, H, period=1)

    def m_mean(train):
        return np.full(H, float(np.mean(train)))

    models_c = {"prophet_lite": m_prophet_g, "ar1": m_ar1,
                "theta": m_theta_g, "mean": m_mean}
    err_c = rolling_race(g, origins_c, models_c, 1)
    tab_c = C.md_table(hdr, metric_rows(err_c, models_c))
    dm_c = C.md_table(["pair (squared loss)", "h", "DM (HLN)", "p", "mean d"],
                      dm_rows(err_c, [("prophet_lite", "ar1"),
                                      ("prophet_lite", "mean")]))
    parts.append("### (c) Real GDP growth, quarterly 400·dlog (T="
                 f"{len(g)}, {len(origins_c)} origins) — no seasonality; "
                 "trend models should lose here\n\n" + tab_c + "\n\n" + dm_c)

    md = ("## Experiment 1 — point-forecast horse race "
          "(rolling origin, expanding window, refit every origin)\n\n"
          + "\n\n".join(parts)
          + f"\n\n_Runtime: {time.time() - t_start:.0f} s. Seeds fixed; "
          "rerun with `python exp01_point_horse_race.py`._")
    payload = {
        "a": {m: {f"h{h}": {"rmse": C.rmse(err_a[m][h - 1]),
                            "mae": C.mae(err_a[m][h - 1])} for h in HS}
              for m in err_a},
        "b": {m: {f"h{h}": {"rmse": C.rmse(err_b[m][h - 1]),
                            "mae": C.mae(err_b[m][h - 1])} for h in HS}
              for m in err_b},
        "c": {m: {f"h{h}": {"rmse": C.rmse(err_c[m][h - 1]),
                            "mae": C.mae(err_c[m][h - 1])} for h in HS}
              for m in err_c},
    }
    C.write_results("exp01", md, payload)


if __name__ == "__main__":
    main()
