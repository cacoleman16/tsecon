"""Golden fixtures for the two preprocessing advisors: `ndiffs` (how many
differences a series needs) and `box_cox_lambda` (the variance-stabilising
Box-Cox lambda).

Reference implementations (this venv):

  * ndiffs, per-step evidence -> statsmodels 0.14.6 / arch 8.0.0
      test="kpss" : statsmodels.tsa.stattools.kpss(y, "c", nlags="auto")
      test="adf"  : statsmodels.tsa.stattools.adfuller(y, regression="c",
                    autolag="AIC")
      test="pp"   : arch.unitroot.PhillipsPerron(y, trend="c",
                    test_type="tau", lags=None)
    The *statistics* are therefore a strong third-party golden. The
    *sequential rule* itself (difference until the test stops calling for a
    difference, capped at `max_d`) has no third-party implementation in this
    venv (no R `forecast`, no pmdarima, no sktime), so it is transcribed here
    from its published definition -- Hyndman & Khandakar (2008), JSS 27(3),
    sec. 3.2, as implemented by `forecast::ndiffs`:

        KPSS (H0 stationary)      : difference while p <  alpha
        ADF / PP (H0 unit root)   : difference while p >  alpha

    with a constant-series short circuit and the `max_d` cap. Grade: the
    per-step numbers are a third-party golden; the rule is a
    documented-rule transcription.

  * box_cox_lambda(method="mle") -> scipy 1.18.0 (STRONG golden)
      scipy.stats.boxcox_llf                -> the objective, on a lambda grid
      scipy.stats.boxcox_normmax(method="mle")
                                            -> the argmax found by SciPy's own
                                               (unbounded) Brent search
      scipy.optimize.minimize_scalar(..., method="bounded", xatol=1e-12)
                                            -> the argmax on the *bounded*
                                               interval the library documents,
                                               to a tolerance far tighter than
                                               `boxcox_normmax`'s 1.48e-8
    NB `boxcox_normmax`'s `brack=(-2, 2)` is a *starting bracket* for a
    downhill search, not a constraint; `tsecon.box_cox_lambda`'s `bounds`
    are hard bounds. The two agree whenever the optimum is interior, which
    is why one deliberately-exterior case (`bound_case`) is included with
    its unbounded SciPy optimum recorded for contrast.

  * box_cox_lambda(method="guerrero") -> no reference implementation exists
    in this venv (R `forecast` is not available). The objective is
    transcribed here from its published definition -- Guerrero (1993),
    "Time-series analysis supported by power transformations", J.
    Forecasting 12, 37-48, in the grouped form used by
    `forecast::BoxCox.lambda`:

        split the last floor(n/period)*period observations into consecutive
        groups of `period`; for group k with mean m_k and sample sd s_k
        (ddof = 1), form r_k = s_k / m_k^(1 - lambda); minimise
        CV(lambda) = sd(r) / mean(r)   (sd again ddof = 1).

    Grade: documented-formula golden, validated against an independent
    NumPy transcription (this file) rather than a third-party library.

This generator NEVER imports tsecon. Doubles are written with json's
shortest round-trip repr, which the Rust golden test parses to identical
bits (serde_json `float_roundtrip`).

Run:  .venv/bin/python fixtures/generate_advisors_fixtures.py
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import scipy
import statsmodels
from arch.unitroot import PhillipsPerron
from scipy import optimize
from scipy.stats import boxcox_llf, boxcox_normmax
from statsmodels.tsa.stattools import adfuller, kpss

OUT = Path(__file__).resolve().parent / "advisors.json"

warnings.simplefilter("ignore")  # statsmodels' p-value interpolation notes


# ------------------------------------------------------------------ series

def ndiffs_series() -> dict[str, np.ndarray]:
    """Seeded processes with a known integration order."""
    out: dict[str, np.ndarray] = {}
    out["white_noise"] = np.random.default_rng(7).standard_normal(200)          # I(0)
    out["random_walk"] = np.cumsum(np.random.default_rng(11).standard_normal(200))  # I(1)
    out["i2"] = np.cumsum(np.cumsum(np.random.default_rng(13).standard_normal(220)))  # I(2)

    e = np.random.default_rng(17).standard_normal(200)
    ar1 = np.zeros(200)
    for t in range(1, 200):
        ar1[t] = 0.6 * ar1[t - 1] + e[t]
    out["ar1"] = ar1                                                            # I(0)

    out["drift_walk"] = np.cumsum(0.3 + np.random.default_rng(23).standard_normal(150))
    return out


def boxcox_series() -> dict[str, np.ndarray]:
    """Strictly positive seeded series with different amounts of skew."""
    rng = np.random.default_rng(20260729)
    out: dict[str, np.ndarray] = {}
    out["lognormal"] = np.exp(0.5 * rng.standard_normal(200) + 2.0)
    out["gamma"] = rng.gamma(2.0, 3.0, 180) + 0.5
    out["trendy"] = 100.0 + np.arange(1, 241) * 0.8 + 5.0 * rng.standard_normal(240)
    out["airline_like"] = (
        np.exp(np.linspace(2.0, 4.0, 144))
        * (1.0 + 0.2 * np.sin(2 * np.pi * np.arange(144) / 12))
        * np.exp(0.05 * rng.standard_normal(144))
    )
    # Left-skewed by construction: x = w^(1/3) with w normal, so the
    # normalising power is lambda = 3 -- OUTSIDE the default (-2, 2) bounds.
    # A bounded optimiser must return the upper bound and say so.
    out["bound_case"] = (30.0 + 3.0 * rng.standard_normal(200)) ** (1.0 / 3.0)
    return out


# ------------------------------------------------------------------ ndiffs

def step_evidence(y: np.ndarray, test: str, alpha: float) -> dict:
    """One differencing order's worth of test evidence, from the reference."""
    if test == "kpss":
        stat, p, lags, _ = kpss(y, regression="c", nlags="auto")
        needs = bool(p < alpha)          # H0 stationary: difference when rejected
    elif test == "adf":
        stat, p, lags, nobs, _, _ = adfuller(y, regression="c", autolag="AIC")
        needs = bool(p > alpha)          # H0 unit root: difference unless rejected
    elif test == "pp":
        r = PhillipsPerron(y, trend="c", test_type="tau", lags=None)
        stat, p, lags = r.stat, r.pvalue, r.lags
        needs = bool(p > alpha)
    else:
        raise ValueError(test)
    return {
        "n": int(y.size),
        "stat": float(stat),
        "p": float(p),
        "lags": int(lags),
        "needs_differencing": needs,
    }


def ndiffs_case(name: str, y: np.ndarray, test: str, alpha: float, max_d: int) -> dict:
    """The documented sequential rule, transcribed (see module docstring)."""
    cur = np.asarray(y, float)
    d = 0
    steps = []
    while True:
        if np.all(cur == cur[0]):
            reason = "Constant"
            break
        ev = step_evidence(cur, test, alpha)
        ev["d"] = d
        steps.append(ev)
        if not ev["needs_differencing"]:
            reason = "Stationary"
            break
        if d >= max_d:
            reason = "MaxD"
            break
        d += 1
        cur = np.diff(cur)
    return {
        "series": name,
        "test": test,
        "alpha": alpha,
        "max_d": max_d,
        "d": d,
        "reason": reason,
        "steps": steps,
    }


def gen_ndiffs(series: dict[str, np.ndarray]) -> list[dict]:
    cases = []
    for name in ("white_noise", "random_walk", "i2", "ar1", "drift_walk"):
        for test in ("kpss", "adf", "pp"):
            cases.append(ndiffs_case(name, series[name], test, 0.05, 2))
    # alpha sensitivity: ar1's KPSS p-value (0.0597) straddles 5% and 10%.
    cases.append(ndiffs_case("ar1", series["ar1"], "kpss", 0.10, 2))
    # the max_d cap: an I(2) series told it may difference only once.
    cases.append(ndiffs_case("i2", series["i2"], "kpss", 0.05, 1))
    cases.append(ndiffs_case("i2", series["i2"], "adf", 0.05, 1))
    # max_d = 3 leaves room past the true order (rule must stop on its own).
    cases.append(ndiffs_case("i2", series["i2"], "kpss", 0.05, 3))
    return cases


# ---------------------------------------------------------------- box-cox

LAMBDA_GRID = [-2.0, -1.75, -1.5, -1.0, -0.5, -0.25, -0.05, 0.0, 0.05, 0.25,
               0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]


def gen_boxcox_llf(series: dict[str, np.ndarray]) -> list[dict]:
    """scipy.stats.boxcox_llf on a lambda grid -- the objective, pinned."""
    return [
        {
            "series": name,
            "lambdas": list(LAMBDA_GRID),
            "llf": [float(boxcox_llf(lmb, x)) for lmb in LAMBDA_GRID],
        }
        for name, x in series.items()
    ]


def gen_boxcox_mle(series: dict[str, np.ndarray]) -> list[dict]:
    cases = []
    for name, x in series.items():
        for lower, upper in ((-2.0, 2.0), (-1.0, 1.0)):
            res = optimize.minimize_scalar(
                lambda L: -boxcox_llf(L, x),
                bounds=(lower, upper),
                method="bounded",
                options={"xatol": 1e-12},
            )
            lam_b = float(res.x)
            cases.append({
                "series": name,
                "lower": lower,
                "upper": upper,
                # SciPy's own bounded argmax at xatol = 1e-12.
                "lambda_bounded": lam_b,
                # SciPy's public entry point (Brent, xtol 1.48e-8, and NOT
                # constrained to `brack`), for the interior cases.
                "lambda_normmax": float(boxcox_normmax(x, method="mle")),
                "llf_at_opt": float(boxcox_llf(lam_b, x)),
                "llf_at_zero": float(boxcox_llf(0.0, x)),
                "llf_at_one": float(boxcox_llf(1.0, x)),
                "at_bound": bool(min(abs(lam_b - lower), abs(lam_b - upper)) < 1e-6),
            })
    return cases


# --------------------------------------------------------------- guerrero

def guerrero_cv(lmb: float, x: np.ndarray, period: int) -> float:
    """Independent NumPy transcription of the Guerrero (1993) grouped
    coefficient-of-variation criterion (see the module docstring)."""
    x = np.asarray(x, float)
    n = x.size
    ngroups = n // period
    nobst = ngroups * period
    blocks = x[n - nobst:].reshape(ngroups, period)   # each ROW is one group
    mu = blocks.mean(axis=1)
    sd = blocks.std(axis=1, ddof=1)
    ratio = sd / mu ** (1.0 - lmb)
    return float(ratio.std(ddof=1) / ratio.mean())


def gen_guerrero_cv(series: dict[str, np.ndarray]) -> list[dict]:
    out = []
    for name, x in series.items():
        for period in (2, 4, 12):
            if x.size < 2 * period:
                continue
            out.append({
                "series": name,
                "period": period,
                "lambdas": list(LAMBDA_GRID),
                "cv": [guerrero_cv(lmb, x, period) for lmb in LAMBDA_GRID],
            })
    return out


def gen_guerrero_opt(series: dict[str, np.ndarray]) -> list[dict]:
    out = []
    for name, x in series.items():
        for period, lower, upper in ((2, -2.0, 2.0), (12, -2.0, 2.0), (2, -1.0, 2.0)):
            if x.size < 2 * period:
                continue
            res = optimize.minimize_scalar(
                lambda L: guerrero_cv(L, x, period),
                bounds=(lower, upper),
                method="bounded",
                options={"xatol": 1e-12},
            )
            out.append({
                "series": name,
                "period": period,
                "lower": lower,
                "upper": upper,
                "lambda": float(res.x),
                "cv": float(res.fun),
            })
    return out


# ------------------------------------------------------------------- main

def main() -> None:
    nds = ndiffs_series()
    bcs = boxcox_series()

    fixture = {
        "_meta": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "statsmodels": statsmodels.__version__,
            "arch": __import__("arch").__version__,
            "note": (
                "ndiffs per-step statistics: statsmodels kpss/adfuller and arch "
                "PhillipsPerron (third-party golden); the sequential rule: "
                "documented-rule transcription (Hyndman-Khandakar 2008 / "
                "forecast::ndiffs). box_cox_lambda MLE: scipy boxcox_llf + "
                "boxcox_normmax (third-party golden). Guerrero: "
                "documented-formula transcription (Guerrero 1993), no "
                "third-party implementation available in this venv."
            ),
        },
        "ndiffs_series": {k: v.tolist() for k, v in nds.items()},
        "ndiffs": gen_ndiffs(nds),
        "boxcox_series": {k: v.tolist() for k, v in bcs.items()},
        "boxcox_llf": gen_boxcox_llf(bcs),
        "boxcox_mle": gen_boxcox_mle(bcs),
        "guerrero_cv": gen_guerrero_cv(bcs),
        "guerrero_opt": gen_guerrero_opt(bcs),
    }

    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(fixture, fh, indent=1)
    print(
        f"wrote {OUT} ({OUT.stat().st_size} bytes); "
        f"{len(fixture['ndiffs'])} ndiffs cases, "
        f"{len(fixture['boxcox_llf'])} llf grids, "
        f"{len(fixture['boxcox_mle'])} mle cases, "
        f"{len(fixture['guerrero_cv'])} guerrero grids, "
        f"{len(fixture['guerrero_opt'])} guerrero optima"
    )


if __name__ == "__main__":
    main()
