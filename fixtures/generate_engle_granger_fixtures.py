"""Golden fixtures for the Engle-Granger two-step cointegration test.

Reference implementation (this venv): statsmodels 0.14.6,
``statsmodels.tsa.stattools.coint(y0, y1, trend, maxlag, autolag)``, which
returns ``(stat, pvalue, crit)`` where

  * ``stat``   is ``adfuller(OLS(y0, add_trend(y1, trend)).resid,
    regression="n", maxlag=..., autolag=...)[0]``;
  * ``pvalue`` is ``mackinnonp(stat, regression=trend, N=k)`` -- the
    MacKinnon (1994) *cointegration* surface indexed by the number of
    series ``N = k``, NOT the standard (N = 1) ADF surface;
  * ``crit``   is ``mackinnoncrit(N=k, regression=trend, nobs=T-1)`` -- the
    MacKinnon (2010) finite-sample surface, evaluated at ``T - 1`` (the -1
    is statsmodels matching Stata's ``egranger``), and ``[nan, nan, nan]``
    for ``trend="n"`` (no published 2010 no-constant cointegration table).

Two cases are pinned from ``coint``'s own internals rather than from
``coint`` itself, because statsmodels cannot compute them:

  * ``N > 6`` -- ``mackinnonp`` indexes a 6-row table, so ``coint`` raises
    ``IndexError``. The fixture stores ``pvalue = null`` (tsecon returns
    NaN) and still pins the statistic and the 2010 critical values, which
    are published up to ``N = 12``.

The step-1 coefficients are re-ordered from statsmodels' design
(``[x..., const, trend]``, ``add_trend(prepend=False)``) into tsecon's
(``[const, trend, x...]``) -- a column permutation, so the fit and the
residuals are identical.

This generator NEVER imports tsecon. Doubles are written with json's
shortest round-trip repr, which the Rust golden test parses to identical
bits (serde_json `float_roundtrip`).

Run:  python fixtures/generate_engle_granger_fixtures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.adfvalues import mackinnoncrit, mackinnonp
from statsmodels.tsa.tsatools import add_trend
from statsmodels.tsa.stattools import adfuller, coint

OUT = Path(__file__).resolve().parent / "engle_granger.json"

SQRTEPS = np.sqrt(np.finfo(np.double).eps)


# --------------------------------------------------------------- systems


def simulated_system(seed: int, t: int, k: int, cointegrated: bool) -> np.ndarray:
    """A `T x k` I(1) system whose first column is regressed on the rest.

    ``cointegrated``: the error in ``y0 = X beta + e`` is white noise
    (residuals stationary) instead of an independent random walk.
    """
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.standard_normal((t, k - 1)), axis=0)
    beta = np.arange(1, k, dtype=float) / 2.0
    err = (
        0.5 * rng.standard_normal(t)
        if cointegrated
        else np.cumsum(rng.standard_normal(t))
    )
    # A drift in y0 makes the "ct" specification bite.
    y0 = x @ beta + err + 0.02 * np.arange(1, t + 1)
    return np.column_stack([y0, x])


def macro_system() -> np.ndarray:
    """Real data: log real consumption on log real GDP and log real
    investment (statsmodels' bundled `macrodata`, 1959Q1-2009Q3, T = 203) --
    the textbook Engle-Granger application."""
    d = sm.datasets.macrodata.load_pandas().data
    return np.column_stack(
        [
            np.log(d["realcons"].to_numpy(dtype=float)),
            np.log(d["realgdp"].to_numpy(dtype=float)),
            np.log(d["realinv"].to_numpy(dtype=float)),
        ]
    )


def build_systems() -> dict[str, np.ndarray]:
    return {
        "co_k2": simulated_system(11, 200, 2, True),
        "no_k2": simulated_system(12, 200, 2, False),
        "co_k3": simulated_system(13, 150, 3, True),
        "no_k4": simulated_system(14, 120, 4, False),
        "co_k6": simulated_system(15, 250, 6, True),
        "co_k7": simulated_system(16, 250, 7, True),
        "macro_k3": macro_system(),
    }


# ------------------------------------------------------------ one golden


def eg_case(system: str, data: np.ndarray, trend: str, autolag, maxlag):
    t, k = data.shape
    y0 = data[:, 0]
    y1 = data[:, 1:]

    # coint()'s step 1, verbatim.
    xx = y1 if trend == "n" else add_trend(y1, trend=trend, prepend=False)
    res_co = OLS(y0, xx).fit()
    assert res_co.rsquared < 1 - 100 * SQRTEPS, "collinear step-1 fit"

    # coint()'s step 2, verbatim: residual ADF with no deterministic term.
    res_adf = adfuller(res_co.resid, maxlag=maxlag, autolag=autolag, regression="n")
    stat, used_lag, adf_nobs = float(res_adf[0]), int(res_adf[2]), int(res_adf[3])

    pvalue = float(mackinnonp(stat, regression=trend, N=k)) if k <= 6 else None
    if trend == "n":
        crit = None  # 2010 no-constant cointegration surface does not exist
    else:
        cv = np.asarray(mackinnoncrit(N=k, regression=trend, nobs=t - 1))
        crit = {"1%": float(cv[0]), "5%": float(cv[1]), "10%": float(cv[2])}

    # Cross-check against the public entry point wherever it is callable
    # (coint raises IndexError for N > 6).
    if k <= 6:
        c_stat, c_p, c_crit = coint(
            y0, y1, trend=trend, maxlag=maxlag, autolag=autolag
        )
        assert float(c_stat) == stat, (c_stat, stat)
        assert float(c_p) == pvalue, (c_p, pvalue)
        c_crit = np.asarray(c_crit, dtype=float)
        if crit is None:
            assert np.all(np.isnan(c_crit)), c_crit
        else:
            assert np.array_equal(c_crit, np.asarray(list(crit.values()))), c_crit

    # statsmodels design is [x..., const, trend]; tsecon's is [const, trend, x...].
    params = np.asarray(res_co.params, dtype=float)
    coefs = list(params[k - 1 :]) + list(params[: k - 1])

    return {
        "system": system,
        "trend": trend,
        "autolag": autolag,
        "maxlag": None if maxlag is None else int(maxlag),
        "n_vars": k,
        "nobs": t,
        "stat": stat,
        "pvalue": pvalue,
        "crit": crit,
        "used_lag": used_lag,
        "adf_nobs": adf_nobs,
        "coint_coefs": [float(c) for c in coefs],
        "rsquared": float(res_co.rsquared),
    }


def gen_cases(systems: dict[str, np.ndarray]):
    cases = []
    # Every system x every trend, at the statsmodels default lag rule.
    for name in ("co_k2", "no_k2", "co_k3", "no_k4", "co_k6", "co_k7", "macro_k3"):
        for trend in ("n", "c", "ct"):
            cases.append(eg_case(name, systems[name], trend, "aic", None))
    # Alternative lag rules and the fixed-lag (autolag=None) path.
    cases.append(eg_case("co_k2", systems["co_k2"], "c", "bic", None))
    cases.append(eg_case("no_k2", systems["no_k2"], "ct", "bic", None))
    cases.append(eg_case("co_k3", systems["co_k3"], "c", "t-stat", None))
    cases.append(eg_case("macro_k3", systems["macro_k3"], "c", "t-stat", None))
    cases.append(eg_case("co_k2", systems["co_k2"], "c", None, 4))
    cases.append(eg_case("no_k4", systems["no_k4"], "ct", None, 0))
    cases.append(eg_case("macro_k3", systems["macro_k3"], "ct", None, 6))
    # A capped maxlag with a search on top.
    cases.append(eg_case("co_k6", systems["co_k6"], "c", "aic", 5))
    return cases


# ------------------------------------------------ surface map (N-indexed)


def gen_crit_map():
    """The 2010 critical-value surfaces at the exact `nobs` the golden cases
    use, so a mismatch localizes to the surface rather than the statistic."""
    out = {}
    for trend in ("c", "ct"):
        out[trend] = {}
        for n_vars, nobs in ((2, 199), (3, 149), (4, 119), (6, 249), (7, 249), (3, 202)):
            cv = np.asarray(mackinnoncrit(N=n_vars, regression=trend, nobs=nobs))
            out[trend][f"{n_vars}_{nobs}"] = {
                "n_vars": n_vars,
                "nobs": nobs,
                "crit": [float(cv[0]), float(cv[1]), float(cv[2])],
            }
    return out


# ------------------------------------------------------------------ main


def main():
    systems = build_systems()
    payload = {
        "_source": "statsmodels 0.14.6 statsmodels.tsa.stattools.coint",
        "systems": {
            name: {
                "n_vars": int(d.shape[1]),
                "nobs": int(d.shape[0]),
                # series-major: k lists of T observations (the layout the
                # Rust `as_endog` helper transposes into T x k).
                "data": [d[:, j].tolist() for j in range(d.shape[1])],
            }
            for name, d in systems.items()
        },
        "cases": gen_cases(systems),
        "crit_map": gen_crit_map(),
    }
    OUT.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KiB, "
          f"{len(payload['cases'])} cases)")


if __name__ == "__main__":
    main()
