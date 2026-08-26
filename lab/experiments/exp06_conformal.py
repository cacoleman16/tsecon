"""Experiment 6 — conformal interval wrappers head to head (split / EnbPI / ACI).

The 0.5.0 conformal stream shipped three distribution-free interval
wrappers in `tsecon.conformal_forecast` / `tsecon.conformal_backtest`:

  * split      — finite-sample-corrected residual-quantile calibration,
                 rolling recalibration in the online evaluator;
  * enbpi      — Xu-Xie (ICML 2021, Alg. 1) bootstrap-ensemble AR
                 learners, leave-one-out residuals, sliding window,
                 width-minimizing beta line search;
  * aci        — Gibbs-Candes (NeurIPS 2021) adaptive level recursion
                 alpha_{t+1} = alpha_t + gamma (alpha - err_t) on top of
                 the same trailing scores as split.

This experiment grades them on one realistic stationary setting and one
shift setting, all one-step-ahead at nominal 90%:

  A. GARCH(1,1)-t returns (fat tails + volatility clustering; the
     common.py lab DGP, omega=.05 alpha=.10 beta=.85 nu=5): T = 500,
     online evaluation over the last 150 points, R = 100 seeds.
     Reported per method: mean realized coverage (with the across-seed
     MC se of the mean), median interval width, and the Kupiec 5%
     unconditional-coverage rejection rate across seeds (a calibrated
     10%-miss hit sequence should reject ~5% of the time).

  B. Variance-shift AR(1) (phi=.5; innovation sd 1 -> 3 at a point
     one-third into the evaluation window): T = 400, eval 120, R = 100.
     Reported: coverage over the post-shift stretch only — the published
     ACI claim is that the level recursion recovers coverage there while
     a fixed-level method under-covers. gamma is run at the paper's
     0.005 and at 0.05 to show the adaptation-speed tradeoff on a
     window this short.

Not part of the public tsecon API; research bar (seeded, honest), not
the library's golden bar. Runtime ~2-3 min.

Run with a venv holding the 0.5.0-dev tsecon build (the conformal entry
points do not exist in a 0.4.0 wheel).
"""

from __future__ import annotations

import time

import numpy as np

import common as C
import tsecon

R = 100
ALPHA = 0.1


def ar1_var_shift(T, seed, shift_at, phi=0.5, sigma_post=3.0):
    rng = np.random.default_rng(seed)
    y = np.empty(T)
    prev = 0.0
    for t in range(T):
        s = sigma_post if t >= shift_at else 1.0
        prev = phi * prev + s * rng.standard_normal()
        y[t] = prev
    return y


def setting_a():
    """GARCH-t returns, stationary: coverage / width / Kupiec."""
    T, n_eval, calib = 500, 150, 200
    methods = {
        "split": dict(method="split", base="ar", lags=1, calib=calib),
        "aci (gamma=.005)": dict(method="aci", base="ar", lags=1,
                                 calib=calib, gamma=0.005),
        "enbpi (B=25)": dict(method="enbpi", base="ar", lags=1,
                             n_boot=25, batch=1),
    }
    cov = {k: [] for k in methods}
    width = {k: [] for k in methods}
    kupiec_rej = {k: 0 for k in methods}
    for r in range(R):
        y, _sigma = C.simulate_garch_t(T, seed=6000 + r)
        for name, kw in methods.items():
            kw = dict(kw)
            kw["seed"] = 6000 + r if kw["method"] == "enbpi" else kw.get("seed", 0)
            out = tsecon.conformal_backtest(
                y, horizon=1, alpha=ALPHA, n_eval=n_eval, **kw
            )
            err = np.asarray(out["err"][0], dtype=bool)
            cov[name].append(1.0 - err.mean())
            lo = np.asarray(out["lower"][0])
            up = np.asarray(out["upper"][0])
            width[name].append(float(np.median(up - lo)))
            # Kupiec on the miss sequence at tau = alpha.
            _rate, _lr, p = C.kupiec(err.astype(float), ALPHA)
            kupiec_rej[name] += p < 0.05
    rows = []
    for name in methods:
        c = np.asarray(cov[name])
        rows.append([
            name,
            c.mean(),
            c.std(ddof=1) / np.sqrt(R),
            float(np.median(width[name])),
            kupiec_rej[name] / R,
        ])
    return rows


def setting_b():
    """Variance shift inside the evaluation window: post-shift coverage."""
    T, n_eval, calib = 400, 120, 100
    shift_at = T - n_eval + 40 - 1  # one third into the eval window
    methods = {
        "split": dict(method="split", base="ar", lags=1, calib=calib),
        "aci (gamma=.005)": dict(method="aci", base="ar", lags=1,
                                 calib=calib, gamma=0.005),
        "aci (gamma=.05)": dict(method="aci", base="ar", lags=1,
                                calib=calib, gamma=0.05),
        "enbpi (B=25)": dict(method="enbpi", base="ar", lags=1,
                             n_boot=25, batch=1),
    }
    post = {k: [] for k in methods}
    full = {k: [] for k in methods}
    for r in range(R):
        y = ar1_var_shift(T, seed=7000 + r, shift_at=shift_at)
        for name, kw in methods.items():
            kw = dict(kw)
            kw["seed"] = 7000 + r if kw["method"] == "enbpi" else kw.get("seed", 0)
            out = tsecon.conformal_backtest(
                y, horizon=1, alpha=ALPHA, n_eval=n_eval, **kw
            )
            err = np.asarray(out["err"][0], dtype=bool)
            origins = np.asarray(out["origins"])
            post_mask = origins + 1 >= shift_at
            post[name].append(1.0 - err[post_mask].mean())
            full[name].append(1.0 - err.mean())
    rows = []
    for name in methods:
        p = np.asarray(post[name])
        f = np.asarray(full[name])
        rows.append([
            name,
            p.mean(),
            p.std(ddof=1) / np.sqrt(R),
            f.mean(),
        ])
    return rows


def main():
    t0 = time.time()
    print("exp06: conformal wrappers (split / EnbPI / ACI), nominal 90%")

    rows_a = setting_a()
    tbl_a = C.md_table(
        ["method", "coverage", "se(mean)", "median width", "Kupiec rej @5%"],
        rows_a,
    )
    print("\nSetting A - GARCH(1,1)-t returns, T=500, eval=150, R=100\n")
    print(tbl_a)

    rows_b = setting_b()
    tbl_b = C.md_table(
        ["method", "post-shift coverage", "se(mean)", "full-window coverage"],
        rows_b,
    )
    print("\nSetting B - variance shift (sd 1 -> 3) inside the eval window, "
          "T=400, eval=120, post-shift stretch=80, R=100\n")
    print(tbl_b)

    dt = time.time() - t0
    md = (
        "# exp06 — conformal interval wrappers (split / EnbPI / ACI)\n\n"
        "One-step-ahead intervals at nominal 90%; all methods share the "
        "AR(1) least-squares base.\n\n"
        "## Setting A — GARCH(1,1)-t returns (T=500, eval=150, R=100)\n\n"
        f"{tbl_a}\n\n"
        "## Setting B — variance shift sd 1→3 inside the eval window "
        "(T=400, eval=120, post-shift 80, R=100)\n\n"
        f"{tbl_b}\n\n"
        f"_runtime {dt:.0f} s_\n"
    )
    C.write_results("exp06", md, {
        "setting_a": rows_a,
        "setting_b": rows_b,
        "R": R,
        "alpha": ALPHA,
        "runtime_s": dt,
    })
    print(f"\nruntime {dt:.0f} s; wrote results/exp06.md + exp06.json")


if __name__ == "__main__":
    main()
