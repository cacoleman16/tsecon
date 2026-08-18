"""Experiment 4 — dynamic 5% tail forecasting on a GARCH(1,1)-t DGP.

lab/laplace AL-GAS dynamic quantile vs
  * tsecon.quantile_regression on a constant (the static 5% quantile),
  * GARCH-implied quantiles: tsecon.garch_fit (zero mean) with Student-t
    and with normal innovations; quantile_t = sigma_t * F^{-1}(0.05) of
    the fitted standardized innovation distribution.

Protocol (true out-of-sample): T=3000, parameters estimated on the first
2000 observations only, then FROZEN; each model's one-step-ahead 5%
quantile path is run forward through the 1000-observation test set
(recursions see past y only — no look-ahead; AL-GAS q0 and its smoothing
bandwidth come from the training window).

The DGP is exactly a GARCH(1,1)-t, so the GARCH-t model is correctly
specified and SHOULD win — the honest question is how close the
semiparametric AL-GAS gets without a volatility model, and how badly the
static quantile and the normal-innovation GARCH miss.

Metrics on the test set: mean pinball score (tau=0.05), hit rate with
Kupiec unconditional-coverage LR p-value, RMSE against the TRUE
conditional quantile path (known by construction), and a Newey-West
loss-differential test on pinball scores vs AL-GAS (tsecon.dm_test only
covers squared/absolute loss, so the pinball DM analogue is computed
directly).  5 seeds; per-seed metrics are averaged, Kupiec is reported
per seed as reject/accept counts.

Runtime: ~1 min.  Seeded; deterministic.
"""

from __future__ import annotations

import time

import numpy as np
from scipy.stats import t as tdist
from scipy.stats import norm

import common as C
import tsecon
from al_gas import al_gas_filter, fit_al_gas

TAU = 0.05
T_TOTAL, T_TRAIN = 3000, 2000
NU_TRUE = 5.0
SEEDS = (0, 1, 2, 3, 4)


def std_t_quantile(tau, nu):
    """tau-quantile of the unit-variance Student-t."""
    return float(tdist.ppf(tau, nu) * np.sqrt((nu - 2.0) / nu))


def main():
    t_start = time.time()
    models = ["al_gas", "garch_t", "garch_norm", "static_qr"]
    agg = {m: {"pinball": [], "hit": [], "kupiec_p": [], "rmse_q": []}
           for m in models}
    nw = {m: [] for m in models if m != "al_gas"}

    for seed in SEEDS:
        y, sigma = C.simulate_garch_t(T_TOTAL, seed=3000 + seed, nu=NU_TRUE)
        train, test = y[:T_TRAIN], y[T_TRAIN:]
        q_true = sigma * std_t_quantile(TAU, NU_TRUE)

        paths = {}

        # AL-GAS: fit on train, freeze params, filter the full series
        g = fit_al_gas(train, TAU)
        n0 = max(25, T_TRAIN // 10)
        q0 = float(np.quantile(train[:n0], TAU))    # same init as the fit
        q_path, _ = al_gas_filter(y, TAU, g.omega, g.a, g.b, q0,
                                  bandwidth=g.bandwidth)
        paths["al_gas"] = q_path

        # GARCH with t and with normal innovations, params frozen at train
        for dist, name in (("t", "garch_t"), ("normal", "garch_norm")):
            gf = tsecon.garch_fit(train, vol="garch", mean="zero", dist=dist,
                                  p=1, o=0, q=1)
            par = dict(zip(gf["param_names"], gf["params"]))
            sig = C.garch_sigma_path(y, par["omega"], par["alpha[1]"],
                                     par["beta[1]"])
            if dist == "t":
                zq = std_t_quantile(TAU, par["nu"])
            else:
                zq = float(norm.ppf(TAU))
            paths[name] = sig * zq

        # static: quantile regression on a constant, train only
        qr = tsecon.quantile_regression(train, np.ones((T_TRAIN, 1)),
                                        taus=[TAU])
        paths["static_qr"] = np.full(T_TOTAL, float(np.ravel(qr["params"])[0]))

        loss = {}
        for m in models:
            q = paths[m][T_TRAIN:]
            u = test - q
            loss[m] = u * (TAU - (u < 0.0))
            hits = test <= q
            hit_rate, _, kp = C.kupiec(hits, TAU)
            agg[m]["pinball"].append(float(loss[m].mean()))
            agg[m]["hit"].append(hit_rate)
            agg[m]["kupiec_p"].append(kp)
            agg[m]["rmse_q"].append(C.rmse(q - q_true[T_TRAIN:]))
        for m in nw:
            nw[m].append(C.nw_tstat(loss[m] - loss["al_gas"]))

    label = {"al_gas": "AL-GAS dynamic quantile (lab)",
             "garch_t": "GARCH(1,1)-t implied (tsecon, correctly specified)",
             "garch_norm": "GARCH(1,1)-normal implied (tsecon)",
             "static_qr": "static quantile_regression (tsecon)"}
    rows = []
    for m in models:
        a = agg[m]
        rows.append([label[m],
                     "%.4f" % np.mean(a["pinball"]),
                     "%.3f" % np.mean(a["hit"]),
                     "%d/%d" % (sum(p < 0.05 for p in a["kupiec_p"]),
                                len(SEEDS)),
                     "%.3f" % np.mean(a["rmse_q"])])
    tab = C.md_table(["model", "mean pinball (tau=.05)", "mean hit rate",
                      "Kupiec rej. @5%", "RMSE vs true quantile"], rows)

    nwrows = []
    for m in nw:
        ts = [t for t, _ in nw[m]]
        ps = [p for _, p in nw[m]]
        nwrows.append([f"{label[m]} - AL-GAS", "%.2f" % np.mean(ts),
                       "%d/%d" % (sum(p < 0.05 for p in ps), len(SEEDS))])
    tab2 = C.md_table(["pinball loss differential (NW t, mean over seeds)",
                       "mean t", "signif @5%"], nwrows)

    md = ("## Experiment 4 — 5% tail forecasting, GARCH(1,1)-t DGP\n\n"
          f"T={T_TOTAL} (train {T_TRAIN}, test {T_TOTAL - T_TRAIN}), "
          f"omega=0.05 alpha=0.10 beta=0.85 nu={NU_TRUE:g}; parameters "
          f"frozen at the training fit; {len(SEEDS)} seeds, metrics "
          "averaged over seeds (Kupiec = count of seeds where the 5% "
          "unconditional-coverage test REJECTS).  Positive NW t means the "
          "row model has HIGHER pinball loss than AL-GAS.\n\n"
          + tab + "\n\n" + tab2
          + f"\n\n_Runtime: {time.time() - t_start:.0f} s._")
    payload = {m: {k: v for k, v in agg[m].items()} for m in models}
    C.write_results("exp04", md, payload)


if __name__ == "__main__":
    main()
