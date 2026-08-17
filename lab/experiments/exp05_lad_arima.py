"""Experiment 5 (supplementary) — LAD/median ARMA under heavy tails.

lab/laplace al_arima: Laplace-innovation (LAD) CSS vs the module's own
identical-pipeline Gaussian-CSS twin, on one-step out-of-sample point
forecasts.  The twin comparison isolates the innovation-distribution
choice (same recursion, same optimizer, same starting values); a one-rep
sanity check confirms the Gaussian twin tracks tsecon.arima_fit's exact
MLE, so "Gaussian CSS" is a fair stand-in for the shipped baseline
without paying 1000 refits.

Design: ARMA(1,1), phi=0.6, theta=0.3, mean 0.5; innovations Student-t
(df=2.5), Laplace, Gaussian; T=350 (train 300, test 50 one-step
forecasts with parameters frozen); 20 replications per innovation type.

Under t(2.5) and Laplace innovations LAD should win the point-forecast
RMSE/MAE; under Gaussian innovations it should LOSE (LAD's asymptotic
relative efficiency pi/2) — both directions are reported.

Runtime: ~1 min.  Seeded; deterministic.
"""

from __future__ import annotations

import time

import numpy as np

import common as C
import tsecon
from al_arima import arma_css_resid, fit_arma_css, simulate_arma

T_TRAIN, T_TEST = 300, 50
REPS = 20
INNOVS = ("t", "laplace", "gaussian")


def one_step_preds(y, m, phi, theta):
    """One-step-ahead predictions with frozen params: pred = y - CSS resid."""
    e = arma_css_resid(y, m, np.atleast_1d(phi), np.atleast_1d(theta))
    return y - e


def main():
    t_start = time.time()
    rows = []
    payload = {}
    for innov in INNOVS:
        errs = {"laplace": {"rmse": [], "mae": []},
                "gaussian": {"rmse": [], "mae": []}}
        for r in range(REPS):
            y = simulate_arma(T_TRAIN + T_TEST, [0.6], [0.3], 0.5,
                              innov=innov, df=2.5, seed=5000 + r)
            train = y[:T_TRAIN]
            for fam in ("laplace", "gaussian"):
                f = fit_arma_css(train, 1, 1, innovations=fam)
                pred = one_step_preds(y, f.mean, f.phi, f.theta)
                e = y[T_TRAIN:] - pred[T_TRAIN:]
                errs[fam]["rmse"].append(C.rmse(e))
                errs[fam]["mae"].append(C.mae(e))
        row = [f"{innov} innovations"]
        for met in ("rmse", "mae"):
            l = float(np.mean(errs["laplace"][met]))
            g = float(np.mean(errs["gaussian"][met]))
            row += ["%.4f" % l, "%.4f" % g, "%.3f" % (l / g)]
        rows.append(row)
        payload[innov] = {fam: {met: float(np.mean(errs[fam][met]))
                                for met in ("rmse", "mae")}
                          for fam in errs}

    tab = C.md_table(["DGP", "LAD RMSE", "Gauss RMSE", "ratio",
                      "LAD MAE", "Gauss MAE", "ratio"], rows)

    # one-rep sanity: Gaussian CSS twin vs tsecon exact-MLE arima_fit
    y = simulate_arma(T_TRAIN, [0.6], [0.3], 0.5, innov="gaussian",
                      seed=5999)
    fg = fit_arma_css(y, 1, 1, innovations="gaussian")
    a = tsecon.arima_fit(y, p=1, d=0, q=1, constant=True)
    pn = list(a["param_names"]) if "param_names" in a else []
    sanity = (f"Gaussian-CSS twin phi={float(fg.phi[0]):.4f}, "
              f"theta={float(fg.theta[0]):.4f} vs tsecon.arima_fit exact MLE "
              f"params={np.round(np.asarray(a['params']), 4).tolist()}"
              + (f" ({pn})" if pn else ""))

    md = ("## Experiment 5 (supplementary) — LAD/median ARMA one-step "
          "forecasts under heavy tails\n\n"
          f"ARMA(1,1) phi=0.6 theta=0.3 mean=0.5, train {T_TRAIN} / test "
          f"{T_TEST} one-step forecasts with frozen parameters, {REPS} "
          "replications per innovation type; ratio < 1 favours LAD.\n\n"
          + tab + "\n\n" + sanity
          + f"\n\n_Runtime: {time.time() - t_start:.0f} s._")
    C.write_results("exp05", md, payload)


if __name__ == "__main__":
    main()
