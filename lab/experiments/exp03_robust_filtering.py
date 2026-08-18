"""Experiment 3 — robust trend filtering under additive outliers.

lab/laplace robust_filter DCS local level (Student-t and Laplace scores)
vs its own nested DCS-Gaussian (= steady-state Kalman) and vs tsecon's
exact-diffuse Kalman filter/smoother (`tsecon.local_level_smooth`) run at
Gaussian UC-MLE variances (statsmodels UnobservedComponents 'llevel' MLE
— the realistic contaminated-pipeline choice).

DGP: Gaussian local level (sigma_eta=0.1, sigma_eps=1.0, T=500) with
0 / 5 / 10% additive outliers at 8 sd (robust_filter.simulate_local_level).
30 replications per contamination level.

Fair-timing note: the DCS recursion mu_{t+1} = mu_t + kappa u_t delivers
the ONE-STEP-PREDICTED level (mu_t uses y up to t-1).  The Kalman
comparison therefore uses the predicted state a_{t|t-1} (= filtered state
shifted by one for the local level); the Kalman SMOOTHER column uses the
full sample and is reported as the (look-ahead) reference bound, not as a
competitor.  RMSE is measured against the CLEAN true level, discarding a
20-observation burn-in.

Also reports: the mean fitted gain kappa by density (the Gaussian
gain-collapse failure mode), and the clean-data nesting check
DCS-Gaussian kappa vs the steady-state Kalman gain at UC-MLE variances.

Runtime: ~1.5 min.  Seeded; deterministic.
"""

from __future__ import annotations

import time
import warnings

import numpy as np

import common as C
import tsecon
from robust_filter import (fit_dcs_local_level, simulate_local_level,
                           steady_state_gain)

T = 500
BURN = 20
REPS = 30
FRACS = (0.0, 0.05, 0.10)
OUT_SIZE = 8.0


def uc_mle_variances(y):
    """Gaussian local-level MLE variances via statsmodels (sigma2_eps, sigma2_eta)."""
    import statsmodels.api as sm
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f = sm.tsa.UnobservedComponents(y, "llevel").fit(disp=0)
    return float(f.params[0]), float(f.params[1])


def main():
    t_start = time.time()
    methods = ["dcs_t", "dcs_laplace", "dcs_gaussian", "kalman_pred",
               "kalman_smooth"]
    rmse = {(m, f): [] for m in methods for f in FRACS}
    kappas = {(m, f): [] for m in ("dcs_t", "dcs_laplace", "dcs_gaussian")
              for f in FRACS}
    nest_rows = []

    for f in FRACS:
        for r in range(REPS):
            y, mu_true, _ = simulate_local_level(
                T, 0.1, 1.0, outlier_frac=f, outlier_size=OUT_SIZE,
                seed=7000 + r)
            sl = slice(BURN, T)

            fits = {}
            for dens, name in (("t", "dcs_t"), ("laplace", "dcs_laplace"),
                               ("gaussian", "dcs_gaussian")):
                res = fit_dcs_local_level(y, density=dens)
                fits[name] = res
                rmse[(name, f)].append(
                    C.rmse(res.mu[sl] - mu_true[sl]))
                kappas[(name, f)].append(res.kappa)

            s2_eps, s2_eta = uc_mle_variances(y)
            ks = tsecon.local_level_smooth(y, s2_eps, s2_eta)
            filt = np.asarray(ks["filtered_state"])
            pred = np.empty(T)          # a_{t|t-1} = a_{t-1|t-1} (local level)
            pred[0] = y[0]              # diffuse start
            pred[1:] = filt[:-1]
            rmse[("kalman_pred", f)].append(C.rmse(pred[sl] - mu_true[sl]))
            rmse[("kalman_smooth", f)].append(
                C.rmse(np.asarray(ks["smoothed_state"])[sl] - mu_true[sl]))

            if f == 0.0 and r < 5:      # clean-data nesting check
                gain = steady_state_gain(s2_eta, s2_eps)
                nest_rows.append(
                    [r, fits["dcs_gaussian"].kappa, gain,
                     abs(fits["dcs_gaussian"].kappa - gain),
                     C.rmse(fits["dcs_gaussian"].mu[sl] - pred[sl])])

    label = {"dcs_t": "DCS-t (robust)", "dcs_laplace": "DCS-Laplace (robust)",
             "dcs_gaussian": "DCS-Gaussian (nested control)",
             "kalman_pred": "tsecon Kalman predicted @ UC-MLE",
             "kalman_smooth": "tsecon Kalman SMOOTHED @ UC-MLE (look-ahead ref)"}
    rows = []
    for m in methods:
        row = [label[m]]
        for f in FRACS:
            v = np.array(rmse[(m, f)])
            row.append("%.3f (%.3f)" % (v.mean(), v.std(ddof=1)))
        rows.append(row)
    tab1 = C.md_table(["method (one-step-predicted level unless noted)",
                       "RMSE 0% (sd)", "RMSE 5% (sd)", "RMSE 10% (sd)"], rows)

    krows = []
    for m in ("dcs_gaussian", "dcs_t", "dcs_laplace"):
        krows.append([label[m]] + ["%.4f" % np.mean(kappas[(m, f)])
                                   for f in FRACS])
    tab2 = C.md_table(["method", "mean kappa 0%", "mean kappa 5%",
                       "mean kappa 10%"], krows)

    tab3 = C.md_table(["rep", "DCS-Gaussian kappa", "steady-state Kalman gain",
                       "|diff|", "path RMSE vs Kalman predicted"],
                      [[r, "%.4f" % k, "%.4f" % g, "%.1e" % d, "%.4f" % p]
                       for r, k, g, d, p in nest_rows])

    md = ("## Experiment 3 — robust trend filtering under additive outliers\n\n"
          f"Local level, sigma_eta=0.1, sigma_eps=1.0, T={T}, outliers at "
          f"{OUT_SIZE} sd, {REPS} replications per contamination level; RMSE "
          "of the one-step-predicted level against the clean truth "
          f"(mean over reps, sd in parentheses), burn-in {BURN}.\n\n"
          + tab1
          + "\n\n### Fitted gain kappa (the Gaussian gain-collapse "
          "failure mode)\n\n" + tab2
          + "\n\n### Nesting check on clean data (first 5 reps): "
          "DCS-Gaussian = steady-state Kalman\n\n" + tab3
          + f"\n\n_Runtime: {time.time() - t_start:.0f} s._")
    payload = {"rmse": {f"{m}_{f}": np.array(rmse[(m, f)]).mean()
                        for m in methods for f in FRACS},
               "kappa": {f"{m}_{f}": float(np.mean(kappas[(m, f)]))
                         for m in ("dcs_t", "dcs_laplace", "dcs_gaussian")
                         for f in FRACS}}
    C.write_results("exp03", md, payload)


if __name__ == "__main__":
    main()
