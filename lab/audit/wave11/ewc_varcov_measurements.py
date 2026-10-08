"""Measurements behind every number the 0.11 EWC and VAR-forecast-covariance
docs quote (the robust-inference and VAR/SVAR model cards, the `ols` and
`var_forecast` docstrings, the CHANGELOG). Everything runs through the built
`tsecon` extension itself — no NumPy proxy of the estimators.

Sections (seeded; the output is committed next to this file):

1. EWC vs Newey-West slope coverage. `ols(se_type="ewc")` with Student-t_B
   critical values against `ols(se_type="hac")` (the tsecon default:
   Bartlett, Newey-West rule-of-thumb `maxlags`, `use_correction=True`) with
   normal critical values, on y = 1 + 0 x + u with x and u independent
   AR(1)s (rho = 0.7, 200-period burn-in), T = 100 and T = 400, 3000
   replications each — the design of the design review's NumPy study, now
   measured on the library.
2. `var_forecast(param_uncertainty=True)` marginal coverage, paired with the
   default plug-in band on the SAME draws: the interval-coverage audit's
   experiment 3 (`docs/examples/coverage/forecast_intervals.py`: VAR(1),
   A = [[.7,.15],[.1,.6]], Sigma = [[1,.4],[.4,1]], T = 100, 12 horizons,
   seed 20260729 stream 3, 6000 replications), fitted at the true order and
   with lags = 4. The default arm reproduces the audit's published plug-in
   number on the identical draws.
3. The levels caveat: a VAR(2) in 100 x log levels of real GDP, consumption
   and investment (statsmodels macrodata) — largest root, cond(Z'Z), and the
   cov / mse inflation at h = 12 with the term on.
4. Cost of the term: wall time of `var_forecast(..., param_uncertainty=True)`
   on the var.json VAR(2) (k = 3) at steps 12, 200, 500, 1000, and the
   largest `steps` the work budget admits there.

Run from the repository root with the release extension importable:

    python lab/audit/wave11/ewc_varcov_measurements.py [--quick] \
        | tee lab/audit/wave11/out/ewc_varcov_measurements.txt
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy import stats

import tsecon

REPO = Path(__file__).resolve().parents[3]


# ------------------------------------------------------------ 1. EWC vs NW
def ar1(rng: np.random.Generator, n: int, rho: float, burn: int = 200) -> np.ndarray:
    e = rng.standard_normal(n + burn)
    x = np.empty(n + burn)
    x[0] = e[0]
    for t in range(1, n + burn):
        x[t] = rho * x[t - 1] + e[t]
    return x[burn:]


def ewc_vs_hac(T: int, reps: int, rho: float = 0.7, seed: int = 7) -> dict:
    rng = np.random.default_rng([seed, T])
    z = stats.norm.ppf(0.975)
    hit_hac = hit_ewc = 0
    w_hac = w_ewc = 0.0
    b_used = None
    for _ in range(reps):
        x = ar1(rng, T, rho)
        u = ar1(rng, T, rho)
        y = 1.0 + 0.0 * x + u
        X = np.column_stack([np.ones(T), x])
        h = tsecon.ols(y, X, se_type="hac")
        e = tsecon.ols(y, X, se_type="ewc")
        b_used = e["ewc_b"]
        tb = stats.t.ppf(0.975, b_used)
        hit_hac += abs(h["params"][1]) <= z * h["bse"][1]
        hit_ewc += abs(e["params"][1]) <= tb * e["bse"][1]
        # the p-value route agrees with the critical-value route
        assert (e["pvalues"][1] >= 0.05) == (abs(e["tvalues"][1]) <= tb)
        w_hac += 2 * z * h["bse"][1]
        w_ewc += 2 * tb * e["bse"][1]
    return {
        "T": T,
        "rho": rho,
        "reps": reps,
        "ewc_b": b_used,
        "nw_maxlags": int(math.floor(4 * (T / 100) ** (2 / 9))),
        "cover_hac": hit_hac / reps,
        "cover_ewc": hit_ewc / reps,
        "width_hac": w_hac / reps,
        "width_ewc": w_ewc / reps,
        "mc_se": math.sqrt(0.95 * 0.05 / reps),
    }


# --------------------------------------------- 2. VAR param-uncertainty arm
def load_forecast_intervals():
    path = REPO / "docs" / "examples" / "coverage" / "forecast_intervals.py"
    spec = importlib.util.spec_from_file_location("forecast_intervals", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def var_param_arm(fi, fit_lags: int, reps: int, t_obs: int = 100, h_max: int = 12) -> dict:
    rng = fi._rng(3)  # the audit's experiment-3 stream
    data = fi.sim_var1(rng, reps, t_obs + h_max, fi.VAR_A, fi.VAR_SIGMA)
    hit = {"plugin": np.zeros((reps, h_max, 2)), "param": np.zeros((reps, h_max, 2))}
    for r in range(reps):
        train, future = data[r, :t_obs, :], data[r, t_obs:, :]
        a = tsecon.var_forecast(train, lags=fit_lags, steps=h_max, alpha=fi.ALPHA)
        b = tsecon.var_forecast(train, lags=fit_lags, steps=h_max, alpha=fi.ALPHA,
                                param_uncertainty=True)
        for name, fc in (("plugin", a), ("param", b)):
            lo = np.asarray(fc["lower"])
            hi = np.asarray(fc["upper"])
            hit[name][r] = (future >= lo) & (future <= hi)
    out = {"fit_lags": fit_lags, "reps": reps, "t_obs": t_obs, "h_max": h_max}
    for name, hits in hit.items():
        per_h = hits.mean(axis=(0, 2))
        pooled_rep = hits.mean(axis=(1, 2))
        out[name] = {
            "pooled": float(hits.mean()),
            "pooled_se": float(pooled_rep.std(ddof=1) / math.sqrt(reps)),
            "worst_h": float(per_h.min()),
            "worst_h_index": int(per_h.argmin()) + 1,
            "h1": float(per_h[0]),
            "h12": float(per_h[-1]),
        }
    d = (hit["param"] - hit["plugin"]).mean(axis=(1, 2))
    out["paired_diff"] = float(d.mean())
    out["paired_diff_se"] = float(d.std(ddof=1) / math.sqrt(reps))
    return out


# -------------------------------------------------------- 3. levels caveat
def levels_caveat() -> dict:
    import statsmodels.api as sm  # data loading only

    mac = sm.datasets.macrodata.load_pandas().data
    lv = 100.0 * np.log(mac[["realgdp", "realcons", "realinv"]].to_numpy(dtype=float))
    fit = tsecon.var_fit(lv, lags=2)
    plain = tsecon.var_forecast(lv, lags=2, steps=12)
    wide = tsecon.var_forecast(lv, lags=2, steps=12, param_uncertainty=True)
    ratio = np.diag(np.asarray(wide["cov"])[11]) / np.diag(np.asarray(plain["cov"])[11])
    p = 2
    Z = np.column_stack([np.ones(len(lv) - p)] + [lv[p - i:len(lv) - i] for i in range(1, p + 1)])
    return {
        "min_root": fit["min_root"],
        "largest_companion_modulus": 1.0 / fit["min_root"],
        "cond_zz": float(np.linalg.cond(Z.T @ Z)),
        "cov_over_mse_h12": ratio.tolist(),
    }


def growth_ratio() -> dict:
    data = np.asarray(json.loads((REPO / "fixtures" / "var.json").read_text())["data_100dlog_gdp_cons_inv"])
    plain = tsecon.var_forecast(data, lags=2, steps=12)
    wide = tsecon.var_forecast(data, lags=2, steps=12, param_uncertainty=True)
    ratio = np.diag(np.asarray(wide["cov"])[11]) / np.diag(np.asarray(plain["cov"])[11])
    return {"cov_over_mse_h12": ratio.tolist()}


# ------------------------------------------------------------------ 4. cost
def cost() -> dict:
    data = np.asarray(json.loads((REPO / "fixtures" / "var.json").read_text())["data_100dlog_gdp_cons_inv"])
    out = {}
    for steps in (12, 200, 500, 1000):
        best = float("inf")
        for _ in range(3 if steps <= 500 else 1):
            t0 = time.perf_counter()
            tsecon.var_forecast(data, lags=2, steps=steps, param_uncertainty=True)
            best = min(best, time.perf_counter() - t0)
        out[str(steps)] = best
    # Largest admitted steps per system size: the crate's work count
    # (forecast.rs `param_uncertainty_work`) against its 2^34 budget,
    # transcribed, then confirmed against the extension on both sides and
    # timed at the cap.
    def work(s: int, k: int, m: int) -> int:
        return s ** 3 * (k * k + 8) // 3 + s * s * k ** 3 // 2 + 2 * s * m ** 3 + s * s * m * m

    for k, lags in ((3, 2), (2, 1), (1, 1)):
        m = 1 + k * lags
        lo = 1
        while work(lo + 1, k, m) <= 2 ** 34:
            lo += 1
        x = data[:, :k]
        try:
            tsecon.var_forecast(x, lags=lags, steps=lo + 1, param_uncertainty=True)
            refused = False
        except ValueError as exc:
            refused = "steps" in str(exc)
        t0 = time.perf_counter()
        tsecon.var_forecast(x, lags=lags, steps=lo, param_uncertainty=True)
        out[f"cap_k{k}_lags{lags}"] = {
            "largest_admitted_steps": lo,
            "seconds_at_cap": time.perf_counter() - t0,
            "refused_at_cap_plus_one_naming_steps": refused,
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    ewc_reps = 300 if args.quick else 3000
    var_reps = 300 if args.quick else 6000
    res: dict = {"tsecon": tsecon.__version__}

    print("== 1. EWC (t_B) vs Newey-West (normal) slope coverage, nominal 0.95 ==")
    res["ewc"] = [ewc_vs_hac(T, ewc_reps) for T in (100, 400)]
    for r in res["ewc"]:
        print(f"T={r['T']} rho={r['rho']} reps={r['reps']} ewc_b={r['ewc_b']} "
              f"nw_maxlags={r['nw_maxlags']}: HAC {r['cover_hac']:.3f} (width {r['width_hac']:.3f})  "
              f"EWC {r['cover_ewc']:.3f} (width {r['width_ewc']:.3f})  MC se {r['mc_se']:.3f}")

    print("\n== 2. var_forecast marginal coverage, plug-in vs param_uncertainty=True (same draws) ==")
    fi = load_forecast_intervals()
    res["var"] = [var_param_arm(fi, lags, var_reps) for lags in (1, 4)]
    for r in res["var"]:
        a, b = r["plugin"], r["param"]
        print(f"lags={r['fit_lags']} T={r['t_obs']} reps={r['reps']}: pooled {a['pooled']:.4f} -> "
              f"{b['pooled']:.4f} (paired diff {r['paired_diff']:+.4f} se {r['paired_diff_se']:.4f}); "
              f"worst horizon {a['worst_h']:.4f} (h={a['worst_h_index']}) -> {b['worst_h']:.4f} "
              f"(h={b['worst_h_index']}); h1 {a['h1']:.4f} -> {b['h1']:.4f}; "
              f"h12 {a['h12']:.4f} -> {b['h12']:.4f}; pooled MC se {a['pooled_se']:.4f}")

    print("\n== 3. cov/mse at h = 12: growth rates (var.json) vs log levels ==")
    res["growth"] = growth_ratio()
    print("growth-rate VAR(2): cov/mse diag h12 =", np.round(res["growth"]["cov_over_mse_h12"], 4))
    res["levels"] = levels_caveat()
    lv = res["levels"]
    print(f"levels VAR(2): largest companion modulus {lv['largest_companion_modulus']:.4f}, "
          f"cond(Z'Z) {lv['cond_zz']:.2e}, cov/mse diag h12 = {np.round(lv['cov_over_mse_h12'], 4)}")

    print("\n== 4. cost of param_uncertainty=True (k = 3 VAR(2), best of 3) ==")
    res["cost"] = cost()
    c = res["cost"]
    for steps in ("12", "200", "500", "1000"):
        print(f"steps={steps}: {c[steps]:.4f} s")
    for key, v in c.items():
        if key.startswith("cap_"):
            print(f"{key}: largest admitted steps {v['largest_admitted_steps']} "
                  f"({v['seconds_at_cap']:.2f} s); refused at +1 naming steps: "
                  f"{v['refused_at_cap_plus_one_naming_steps']}")

    out = Path(__file__).resolve().parent / "out" / "ewc_varcov_measurements.json"
    if not args.quick:
        out.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
        print(f"\nwrote {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
