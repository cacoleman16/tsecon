"""Golden fixtures for the Phillips-Perron and Phillips-Ouliaris tests.

Reference implementations (this venv):
  * statistics    -> arch 8.0.0
      arch.unitroot.PhillipsPerron (Z-tau, Z-alpha)
      arch.unitroot.cointegration.phillips_ouliaris (Zt, Za)
  * p-value / crit maps -> the MacKinnon response surfaces
      Phillips-Perron Z-tau : arch/statsmodels ADF-t (N = 1) surfaces
      Phillips-Perron Z-alpha: arch ADF-z (N = 1) surfaces
      Phillips-Ouliaris  Zt : statsmodels cointegration surfaces indexed by
                              N = 1 + ncols(x) -- the route statsmodels
                              `coint` takes (mackinnonp N, mackinnoncrit at
                              nobs = T - 1). NB: arch's own PO p-value uses a
                              distinct proprietary simulation; the library
                              deliberately adopts the published MacKinnon-N
                              surfaces, so the PO p-value/crit here are
                              computed from statsmodels, not from arch.

This generator NEVER imports tsecon. Doubles are written with json's
shortest round-trip repr, which the Rust golden test parses to identical
bits (serde_json `float_roundtrip`).

Run:  python fixtures/generate_phillips_fixtures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import statsmodels.api as sm
from arch.unitroot import PhillipsPerron
from arch.unitroot.cointegration import phillips_ouliaris
from arch.unitroot.critical_values.dickey_fuller import (
    adf_z_cv_approx,
    adf_z_large_p,
    adf_z_small_p,
    adf_z_star,
)
from arch.unitroot.unitroot import mackinnoncrit as arch_mackinnoncrit
from arch.unitroot.unitroot import mackinnonp as arch_mackinnonp
from statsmodels.tsa.adfvalues import mackinnoncrit, mackinnonp

OUT = Path(__file__).resolve().parent / "phillips.json"


# --------------------------------------------------------------- series

def nile_series() -> np.ndarray:
    return sm.datasets.nile.load_pandas().data["volume"].to_numpy(dtype=float)


def random_walk(seed: int, n: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.standard_normal(n))


# ------------------------------------------------------- Phillips-Perron

def pp_case(y: np.ndarray, name: str, regression: str, test_type: str, lags):
    tau = PhillipsPerron(y, trend=regression, test_type="tau", lags=lags)
    rho = PhillipsPerron(y, trend=regression, test_type="rho", lags=lags)
    selected = tau if test_type == "tau" else rho
    cv = selected.critical_values
    return {
        "series": name,
        "regression": regression,
        "test_type": test_type,
        "lags": None if lags is None else int(lags),
        "stat": float(selected.stat),
        "ztau": float(tau.stat),
        "zalpha": float(rho.stat),
        "pvalue": float(selected.pvalue),
        "lags_used": int(selected.lags),
        "nobs": int(selected.nobs),
        "crit": {k: float(cv[k]) for k in ("1%", "5%", "10%")},
    }


def gen_pp(series: dict[str, np.ndarray]):
    cases = []
    for name in ("nile", "rw0", "rw1"):
        y = series[name]
        for regression in ("n", "c", "ct"):
            for test_type in ("tau", "rho"):
                cases.append(pp_case(y, name, regression, test_type, None))
    # Explicit-bandwidth cases pin the fixed-lag path.
    cases.append(pp_case(series["rw0"], "rw0", "c", "tau", 8))
    cases.append(pp_case(series["rw0"], "rw0", "ct", "rho", 10))
    return cases


# ----------------------------------------------------- Phillips-Ouliaris

def build_po_systems():
    """Two seeded systems x {m = 1, 2, 3}: one cointegrated (stationary
    residual), one not (random-walk residual)."""
    T = 150
    systems: dict[str, dict] = {}
    for m in (1, 2, 3):
        rng = np.random.default_rng(100 + m)
        x = np.cumsum(rng.standard_normal((T, m)), axis=0)
        beta = np.arange(1, m + 1, dtype=float)
        # Cointegrated: y = x beta + stationary noise.
        y_co = x @ beta + 0.5 * rng.standard_normal(T)
        # Not cointegrated: y = x beta + an independent random walk error.
        y_no = x @ beta + np.cumsum(rng.standard_normal(T))
        systems[f"coint_m{m}"] = {
            "m": m,
            "x": [x[:, j].tolist() for j in range(m)],
            "y": y_co.tolist(),
        }
        systems[f"noco_m{m}"] = {
            "m": m,
            "x": [x[:, j].tolist() for j in range(m)],
            "y": y_no.tolist(),
        }
    return systems


def po_case(system: str, sysdata: dict, trend: str, test_type: str, bandwidth: int):
    m = sysdata["m"]
    n_vars = m + 1
    x = np.column_stack([np.asarray(c, float) for c in sysdata["x"]])
    y = np.asarray(sysdata["y"], float)
    T = y.shape[0]
    res = phillips_ouliaris(
        y, x, trend=trend, test_type=test_type, bandwidth=bandwidth, force_int=True
    )
    case = {
        "system": system,
        "m": m,
        "n_vars": n_vars,
        "trend": trend,
        "test_type": test_type,
        "bandwidth": int(bandwidth),
        "stat": float(res.stat),
        "lags": int(bandwidth),
        "nobs": T,
    }
    if test_type == "Zt":
        # MacKinnon-N surfaces (statsmodels), NOT arch's proprietary PO p-value.
        case["pvalue"] = float(mackinnonp(res.stat, regression=trend, N=n_vars))
        if trend in ("c", "ct"):
            cv = np.asarray(mackinnoncrit(N=n_vars, regression=trend, nobs=T - 1))
            case["crit"] = {"1%": float(cv[0]), "5%": float(cv[1]), "10%": float(cv[2])}
        else:
            case["crit"] = None
    else:  # Za: statistic only.
        case["pvalue"] = None
        case["crit"] = None
    return case


def gen_po(systems: dict[str, dict]):
    cases = []
    for m in (1, 2, 3):
        for stype in ("coint", "noco"):
            system = f"{stype}_m{m}"
            for trend in ("c", "ct"):
                cases.append(po_case(system, systems[system], trend, "Zt", 5))
    # A couple of Za (statistic-only) cases and a varied bandwidth.
    cases.append(po_case("coint_m2", systems["coint_m2"], "c", "Za", 5))
    cases.append(po_case("noco_m3", systems["noco_m3"], "ct", "Za", 8))
    cases.append(po_case("coint_m1", systems["coint_m1"], "c", "Zt", 8))
    # n-trend Zt: p-value available (N<=6), crit unavailable (matches coint).
    cases.append(po_case("coint_m2", systems["coint_m2"], "n", "Zt", 5))
    return cases


# ------------------------------------------------ table-map (transcription)

def gen_adf_z_map():
    grid = [-30.0, -18.0, -12.0, -9.0, -6.0, -5.0, -3.0, -1.79146, -1.0, 0.0, 1.0, 5.0]
    nobs = 100
    out = {}
    for reg in ("n", "c", "ct"):
        pvals = [float(arch_mackinnonp(s, regression=reg, dist_type="adf-z")) for s in grid]
        cv = np.asarray(arch_mackinnoncrit(regression=reg, nobs=nobs, dist_type="adf-z"))
        out[reg] = {
            "stat_grid": grid,
            "pvalues": pvals,
            "nobs": nobs,
            "crit": [float(cv[0]), float(cv[1]), float(cv[2])],
        }
    return out


def gen_coint_p_map():
    grid = [-8.0, -6.0, -5.0, -4.0, -3.5, -3.0, -2.0, -1.5, -1.0, 0.0, 0.5, 1.0]
    out = {}
    for reg in ("n", "c", "ct"):
        out[reg] = {}
        for N in range(2, 7):
            out[reg][str(N)] = {
                "stat_grid": grid,
                "pvalues": [float(mackinnonp(s, regression=reg, N=N)) for s in grid],
            }
    return out


def gen_coint_crit_map():
    nobs = 149
    out = {}
    for reg in ("c", "ct"):
        out[reg] = {}
        for N in range(2, 13):
            cv = np.asarray(mackinnoncrit(N=N, regression=reg, nobs=nobs))
            out[reg][str(N)] = {"nobs": nobs, "crit": [float(cv[0]), float(cv[1]), float(cv[2])]}
    return out


# ------------------------------------------------------------------ main

def main():
    series = {
        "nile": nile_series(),
        "rw0": random_walk(0, 200),
        "rw1": random_walk(1, 80),
    }
    systems = build_po_systems()

    fixture = {
        "series": {k: v.tolist() for k, v in series.items()},
        "pp": gen_pp(series),
        "po_series": systems,
        "po": gen_po(systems),
        "adf_z_map": gen_adf_z_map(),
        "coint_p_map": gen_coint_p_map(),
        "coint_crit_map": gen_coint_crit_map(),
        # A light provenance note kept out of the numeric asserts.
        "_meta": {
            "adf_z_star": {k: float(adf_z_star[k]) for k in ("n", "c", "ct")},
            "note": "PP stats from arch.PhillipsPerron; PO stats from arch "
            "phillips_ouliaris; PO p-value/crit from statsmodels MacKinnon-N "
            "surfaces (statsmodels coint route), not arch's proprietary PO "
            "p-value.",
        },
    }
    # Touch the transcribed adf-z coefficient tables so a table drift in a
    # future arch release is visible in the provenance block.
    fixture["_meta"]["adf_z_small_p_c"] = list(map(float, adf_z_small_p["c"]))
    fixture["_meta"]["adf_z_large_p_c"] = list(map(float, adf_z_large_p["c"]))
    fixture["_meta"]["adf_z_cv_c"] = [list(map(float, r)) for r in adf_z_cv_approx["c"]]

    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(fixture, fh, indent=1)
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes); "
          f"{len(fixture['pp'])} PP cases, {len(fixture['po'])} PO cases")


if __name__ == "__main__":
    main()
