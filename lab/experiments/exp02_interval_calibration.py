"""Experiment 2 — interval calibration on the home-turf DGP.

Empirical coverage of prophet_lite's simulation-based 80/95% intervals vs
tsecon SARIMA's parametric Gaussian intervals, over 300 seeded
replications of the piecewise-trend + seasonal + outliers DGP.

Design (per replication r = 0..299):
  * simulate T = 132: the first 120 observations are the training window
    (3% outlier contamination lives ONLY there), the last 12 are the
    clean evaluation future — so the exercise isolates how estimation-
    window outliers distort each method's intervals, rather than asking
    either model to predict future outliers it has no model for;
  * prophet_lite: fit with (12,3) Fourier seasonality, simulate 500
    predictive draws (seed=r), take the 80/95% quantile intervals;
  * SARIMA (0,1,1)(0,1,1)_12: tsecon.arima_fit with conf_alpha=0.05;
    the 80% band is rebuilt from the same Gaussian forecast_se with
    z = 1.2816 (exactly what conf_alpha=0.2 would return, one fit);
  * record containment of the realized future and interval widths.

House-style diagnostics: coverage with its Monte-Carlo binomial se
sqrt(p(1-p)/R) per horizon; pooled coverage across all 12 horizons.

Runtime: ~5 min.  Seeded; deterministic.
"""

from __future__ import annotations

import time

import numpy as np

import common as C
import tsecon
from prophet_lite import fit as prophet_fit

R = 300
TRAIN = 120
H = 12
Z80 = 1.2815515655446004


def main():
    t_start = time.time()
    cov = {("prophet_lite", lv): np.zeros((R, H)) for lv in (0.8, 0.95)}
    cov.update({("sarima", lv): np.zeros((R, H)) for lv in (0.8, 0.95)})
    wid = {k: np.zeros((R, H)) for k in cov}
    p_conv = 0

    for r in range(R):
        y, _ = C.piecewise_seasonal(TRAIN + H, seed=1000 + r,
                                    outlier_until=TRAIN)
        train, future = y[:TRAIN], y[TRAIN:]

        pres = prophet_fit(train, seasonality=[(12, 3)])
        p_conv += bool(pres["converged"])
        fc = pres.forecast(H, level=[0.8, 0.95], n_draws=500, seed=r)
        for lv in (0.8, 0.95):
            lo, up = fc["lower"][str(lv)], fc["upper"][str(lv)]
            cov[("prophet_lite", lv)][r] = (future >= lo) & (future <= up)
            wid[("prophet_lite", lv)][r] = up - lo

        a = tsecon.arima_fit(train, p=0, d=1, q=1, seasonal=(0, 1, 1, 12),
                             constant=False, forecast_steps=H,
                             conf_alpha=0.05)
        mean, se = a["forecast_mean"], a["forecast_se"]
        bands = {0.95: (a["forecast_lower"], a["forecast_upper"]),
                 0.8: (mean - Z80 * se, mean + Z80 * se)}
        for lv in (0.8, 0.95):
            lo, up = bands[lv]
            cov[("sarima", lv)][r] = (future >= lo) & (future <= up)
            wid[("sarima", lv)][r] = up - lo

    mc_se = lambda p: np.sqrt(np.maximum(p * (1 - p), 1e-12) / R)
    rows = []
    for name in ("prophet_lite", "sarima"):
        for lv in (0.8, 0.95):
            c = cov[(name, lv)]
            w = wid[(name, lv)]
            row = [name, f"{int(lv*100)}%"]
            for h in (1, 6, 12):
                p = float(c[:, h - 1].mean())
                row.append("%.3f (%.3f)" % (p, mc_se(p)))
            row.append("%.3f" % float(c.mean()))
            row.append("%.2f" % float(w.mean()))
            rows.append(row)
    tab = C.md_table(["model", "nominal", "cov h=1 (se)", "cov h=6 (se)",
                      "cov h=12 (se)", "pooled h=1..12", "mean width"], rows)

    md = ("## Experiment 2 — interval calibration, "
          f"{R} seeded replications of the home-turf DGP\n\n"
          "Training window T=120 with 3% outliers (6-10 sigma); the 12-step "
          "future is clean.  prophet_lite intervals: 500 predictive draws "
          "(future-changepoint bootstrap + Gaussian noise).  SARIMA "
          "(0,1,1)(0,1,1)_12 intervals: parametric Gaussian "
          "(innovation+filtering uncertainty, parameters treated as known "
          "— tsecon's documented statsmodels-matching default).  Binomial "
          f"MC standard errors in parentheses (R={R}).\n\n" + tab
          + f"\n\nprophet_lite fits converged: {p_conv}/{R}."
          + f"\n\n_Runtime: {time.time() - t_start:.0f} s._")
    payload = {f"{name}_{lv}": {"per_h": cov[(name, lv)].mean(0),
                                "pooled": float(cov[(name, lv)].mean()),
                                "width": wid[(name, lv)].mean(0)}
               for (name, lv) in cov}
    C.write_results("exp02", md, payload)


if __name__ == "__main__":
    main()
