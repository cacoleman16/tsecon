"""Golden fixtures for the DF-GLS (Elliott-Rothenberg-Stock 1996) unit-root
test.

Reference implementation (this venv):
  * statistic, selected lag, nobs -> arch 8.0.0, arch.unitroot.DFGLS
      (GLS detrending at cbar = -7.0 for trend "c", -13.5 for "ct";
       Perron-Qu 2007 lag selection on the OLS-detrended series with no
       deterministics; trendless ADF regression on the GLS-detrended
       series).
  * p-value / critical-value maps -> the arch "dfgls" response surfaces
      (arch.unitroot.critical_values.dfgls), computed by Kevin Sheppard
      following the MacKinnon (1994, 2010) response-surface methodology
      from novel simulations. These are TRANSCRIBED SURFACES, not an
      independently published table: the honest grading of the p-value/CV
      layer is "matches arch's simulated surfaces bit-for-bit", not
      "matches ERS 1996 Table 1". The statistic itself is a strong golden
      (independent implementation, pinned at 1e-10 relative).

Grading used by the Rust golden (crates/tsecon-diag/tests/dfgls_golden.rs):
  * statistic       : 1e-10 relative (independent QR vs numpy lstsq/pinv)
  * selected lag    : exact
  * nobs            : exact
  * critical values : 1e-12 relative (identical Horner evaluation of the
                      same transcribed doubles -- effectively bit-equal)
  * p-values        : atol 1e-12 + rtol 1e-8 (tsecon-stats vs scipy
                      normal-CDF difference in the deep tail)

This generator NEVER imports tsecon. Doubles are written with json's
shortest round-trip repr, which the Rust golden test parses to identical
bits (serde_json `float_roundtrip`).

Run:  /path/to/reference-venv/python fixtures/generate_dfgls_fixtures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import statsmodels.api as sm
from arch.unitroot import DFGLS
from arch.unitroot.critical_values.dfgls import (
    dfgls_cv_approx,
    dfgls_large_p,
    dfgls_small_p,
    dfgls_tau_max,
    dfgls_tau_min,
    dfgls_tau_star,
)
from arch.unitroot.unitroot import mackinnoncrit as arch_mackinnoncrit
from arch.unitroot.unitroot import mackinnonp as arch_mackinnonp

OUT = Path(__file__).resolve().parent / "dfgls.json"


# --------------------------------------------------------------- series

def nile_series() -> np.ndarray:
    return sm.datasets.nile.load_pandas().data["volume"].to_numpy(dtype=float)


def random_walk(seed: int, n: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.standard_normal(n))


def trend_stationary(seed: int, n: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 0.5 * np.arange(n, dtype=float) + 5.0 * rng.standard_normal(n)


def white_noise(seed: int, n: int) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(n)


# ---------------------------------------------------------------- cases

def case(y, name, trend, lags=None, max_lags=None, method="aic"):
    res = DFGLS(y, trend=trend, lags=lags, max_lags=max_lags, method=method)
    cv = res.critical_values
    return {
        "series": name,
        "trend": trend,
        "lags": None if lags is None else int(lags),
        "max_lags": None if max_lags is None else int(max_lags),
        "method": method,
        "stat": float(res.stat),
        "pvalue": float(res.pvalue),
        "lags_used": int(res.lags),
        "nobs": int(res.regression.nobs),
        "crit": {k: float(cv[k]) for k in ("1%", "5%", "10%")},
    }


def gen_cases(series):
    cases = []
    # Automatic AIC selection, both trend cases, every series.
    for name in ("nile", "rw0", "rw1", "rw2", "trend_stat", "noise"):
        for trend in ("c", "ct"):
            cases.append(case(series[name], name, trend))
    # Fixed-lag cases (pin the lags= path).
    cases.append(case(series["nile"], "nile", "c", lags=3))
    cases.append(case(series["rw0"], "rw0", "ct", lags=5))
    # Explicit max_lags cap.
    cases.append(case(series["rw0"], "rw0", "c", max_lags=6))
    # BIC and t-stat selection.
    cases.append(case(series["nile"], "nile", "ct", method="bic"))
    cases.append(case(series["rw1"], "rw1", "c", method="t-stat"))
    return cases


# ------------------------------------------------ table-map (transcription)

def gen_dfgls_map():
    """P-value and critical-value grids straight off arch's dfgls response
    surfaces, pinning the Rust transcription (incl. the tau_star boundary
    and the tau_min/tau_max saturation on both sides)."""
    grids = {
        # star("c") = -0.4795076091714674; min -17.561..., max 13.365...
        "c": [-20.0, -17.561302895074206, -15.0, -10.0, -6.0, -4.0, -3.0,
              -2.5, -2.0, -1.5, -1.0, -0.4795076091714674, -0.2, 0.0, 1.0,
              5.0, 13.365361509140614, 14.0],
        # star("ct") = -2.1960404365401298; min -13.681..., max 8.737...
        "ct": [-15.0, -13.681153542634465, -12.0, -8.0, -6.0, -4.5, -3.5,
               -3.0, -2.5, -2.1960404365401298, -2.0, -1.0, 0.0, 1.0, 5.0,
               8.73743383728356, 9.0],
    }
    out = {}
    for trend in ("c", "ct"):
        grid = grids[trend]
        pvals = [
            float(arch_mackinnonp(s, regression=trend, dist_type="dfgls"))
            for s in grid
        ]
        crit = {}
        for nobs in (98, 199, 500):
            cv = np.asarray(
                arch_mackinnoncrit(regression=trend, nobs=nobs, dist_type="dfgls")
            )
            crit[str(nobs)] = [float(cv[0]), float(cv[1]), float(cv[2])]
        out[trend] = {"stat_grid": grid, "pvalues": pvals, "crit": crit}
    return out


# ------------------------------------------------------------------ main

def main():
    series = {
        "nile": nile_series(),
        "rw0": random_walk(0, 200),
        "rw1": random_walk(1, 80),
        "rw2": random_walk(2, 150),
        "trend_stat": trend_stationary(42, 150),
        "noise": white_noise(7, 250),
    }

    fixture = {
        "series": {k: v.tolist() for k, v in series.items()},
        "cases": gen_cases(series),
        "dfgls_map": gen_dfgls_map(),
        # Provenance: the transcribed surface constants, so a drift in a
        # future arch release is visible here rather than only in a
        # mysterious test failure.
        "_meta": {
            "reference": "arch 8.0.0 arch.unitroot.DFGLS; surfaces from "
            "arch.unitroot.critical_values.dfgls (Sheppard's MacKinnon-style "
            "simulations -- transcribed, not independently published)",
            "dfgls_tau_star": {k: float(dfgls_tau_star[k]) for k in ("c", "ct")},
            "dfgls_tau_min": {k: float(dfgls_tau_min[k]) for k in ("c", "ct")},
            "dfgls_tau_max": {k: float(dfgls_tau_max[k]) for k in ("c", "ct")},
            "dfgls_small_p": {k: list(map(float, dfgls_small_p[k])) for k in ("c", "ct")},
            "dfgls_large_p": {k: list(map(float, dfgls_large_p[k])) for k in ("c", "ct")},
            "dfgls_cv_approx": {
                k: [list(map(float, row)) for row in dfgls_cv_approx[k]]
                for k in ("c", "ct")
            },
        },
    }

    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(fixture, fh, indent=1)
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes); "
          f"{len(fixture['cases'])} DFGLS cases")


if __name__ == "__main__":
    main()
