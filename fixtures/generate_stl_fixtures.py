"""Golden fixtures for STL decomposition, the Wang-Smith-Hyndman
seasonal/trend strength measures, and the `nsdiffs` seasonal-differencing
advisor.

Reference implementations (this venv):

  * stl components -> statsmodels 0.14.6 (STRONG golden)
      statsmodels.tsa.seasonal.STL(y, period=..., ...).fit()
    The compiled Cython port of the netlib Fortran `stl.f` (with the
    corrected partitioned-sort median in the robustness weights). The
    seasonal/trend/resid arrays (and robustness weights where the outer
    loop runs) are pinned ELEMENTWISE. The algorithm is deterministic, so
    the Rust port is expected to agree to ~1e-12; the test tolerance is
    1e-8.

  * seasonal_strength -> statsmodels STL components + a transcribed formula
    The components come from statsmodels (third-party golden), but the
    strength measures themselves have no reference implementation in this
    venv (no R `forecast`/`tsfeatures`/`feasts`), so the formula is
    transcribed here from its published definition -- Wang, Smith &
    Hyndman (2006), Data Mining and Knowledge Discovery 13, 335-364, in
    the form used by tsfeatures/feasts and FPP3 sec. 4.3:

        strength_seasonal = max(0, 1 - var(resid) / var(seasonal + resid))
        strength_trend    = max(0, 1 - var(resid) / var(trend + resid))

    with SAMPLE variances (ddof=1, R's `var`). Grade: components are a
    third-party golden; the formula is a documented-formula transcription.

  * nsdiffs -> transcribed published rule (like generate_advisors_fixtures)
    The Hyndman-Khandakar seasonal-strength rule, as implemented by
    `forecast::nsdiffs(test = "seas")`: D = 1 if the STL seasonal strength
    is >= 0.64, iterated on the seasonally-differenced series up to max_d
    (forecast's default max.D = 1). No R is available in this venv, so the
    rule is transcribed from the forecast package documentation/source;
    the per-step strengths are computed from statsmodels STL fits at
    default parameters. Grade: per-step numbers third-party + documented
    formula; the 0.64 threshold and sequential rule are a documented-rule
    transcription.

This generator NEVER imports tsecon. Doubles are written with json's
shortest round-trip repr, which the Rust golden test parses to identical
bits (serde_json `float_roundtrip`).

Run:  python fixtures/generate_stl_fixtures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.seasonal import STL

OUT = Path(__file__).resolve().parent / "stl.json"

SEAS_THRESHOLD = 0.64  # forecast::nsdiffs(test="seas")


# --------------------------------------------------------------- series

def co2_monthly() -> np.ndarray:
    """The classic STL example: Mauna Loa CO2, aggregated weekly -> monthly
    with forward-fill, exactly as the statsmodels STL docstring does."""
    data = sm.datasets.co2.load_pandas().data
    return data.resample("ME").mean().ffill()["co2"].to_numpy(dtype=float)


def synthetic_monthly() -> np.ndarray:
    """Seeded monthly series: slowly growing seasonal, smooth trend, noise,
    and three planted outliers (so robust=True has work to do)."""
    rng = np.random.default_rng(20260817)
    t = np.arange(180, dtype=float)
    seasonal = (1.0 + 0.004 * t) * np.sin(2 * np.pi * t / 12.0) + 0.4 * np.cos(
        4 * np.pi * t / 12.0
    )
    trend = 10.0 + 0.05 * t + 2.0 * np.sin(t / 60.0)
    y = trend + seasonal + 0.5 * rng.standard_normal(t.shape[0])
    y[30] += 6.0
    y[77] -= 8.0
    y[150] += 5.0
    return y


def realgdp_quarterly() -> np.ndarray:
    """100 log US real GDP (statsmodels macrodata), quarterly, n = 203 --
    seasonally adjusted at the source, so its seasonal strength is low."""
    md = sm.datasets.macrodata.load_pandas().data
    return 100.0 * np.log(md["realgdp"].to_numpy(dtype=float))


def white_noise() -> np.ndarray:
    rng = np.random.default_rng(7)
    return rng.standard_normal(120)


# ------------------------------------------------------------------ STL

# (name, STL kwargs, fit kwargs) -- applied to every series. Covers:
# defaults; robust; a large "periodic-ish" seasonal window; seasonal_deg=0;
# non-unit jumps; and explicit windows with explicit inner/outer counts.
SHARED_CONFIGS = [
    ("defaults", {}, {}),
    ("robust", {"robust": True}, {}),
    ("periodic", {"seasonal": 51}, {}),
    ("sdeg0", {"seasonal_deg": 0}, {}),
    (
        "jumps",
        {"seasonal_jump": 3, "trend_jump": 2, "low_pass_jump": 2},
        {},
    ),
    (
        "explicit",
        {"seasonal": 9, "trend": 25, "low_pass": 15, "robust": True},
        {"inner_iter": 3, "outer_iter": 4},
    ),
]

# Extra branch coverage on the small synthetic series only (keeps the
# fixture size moderate): periodic window + jump (the len>=n LOESS branch
# with interpolation), and degree-0 trend/low-pass.
EXTRA_CONFIGS = {
    "synthetic_m": [
        ("periodic_jump", {"seasonal": 35, "seasonal_jump": 4}, {}),
        ("tdeg0", {"trend_deg": 0, "low_pass_deg": 0}, {}),
    ]
}


def stl_case(series_name, y, period, cfg_name, kwargs, fit_kwargs):
    model = STL(y, period=period, **kwargs)
    res = model.fit(**fit_kwargs)
    config = model.config  # resolved windows/degrees/jumps
    robustish = fit_kwargs.get(
        "outer_iter", 15 if kwargs.get("robust", False) else 0
    ) > 0
    case = {
        "series": series_name,
        "period": period,
        "config_name": cfg_name,
        "kwargs": kwargs,
        "fit_kwargs": fit_kwargs,
        "resolved": {k: (int(v) if not isinstance(v, bool) else v) for k, v in config.items()},
        "seasonal": np.asarray(res.seasonal).tolist(),
        "trend": np.asarray(res.trend).tolist(),
        "resid": np.asarray(res.resid).tolist(),
        # Robustness weights only where the outer loop runs (all-ones
        # otherwise; the Rust property tests cover that case).
        "weights": np.asarray(res.weights).tolist() if robustish else None,
    }
    return case


def gen_stl(series: dict[str, tuple[np.ndarray, int]]):
    cases = []
    for name, (y, period) in series.items():
        for cfg_name, kwargs, fit_kwargs in SHARED_CONFIGS + EXTRA_CONFIGS.get(name, []):
            cases.append(stl_case(name, y, period, cfg_name, kwargs, fit_kwargs))
    return cases


# ------------------------------------------------- strength (transcribed)

def strengths(y: np.ndarray, period: int) -> tuple[float, float]:
    """Wang-Smith-Hyndman strengths from a default statsmodels STL fit,
    sample variances (ddof=1)."""
    res = STL(y, period=period).fit()
    resid = np.asarray(res.resid)
    seasonal = np.asarray(res.seasonal)
    trend = np.asarray(res.trend)
    vr = resid.var(ddof=1)
    vsr = (seasonal + resid).var(ddof=1)
    vtr = (trend + resid).var(ddof=1)
    s = max(0.0, 1.0 - vr / vsr) if vsr > 0 else 0.0
    t = max(0.0, 1.0 - vr / vtr) if vtr > 0 else 0.0
    return float(s), float(t)


def gen_strength(series: dict[str, tuple[np.ndarray, int]]):
    cases = []
    for name, (y, period) in series.items():
        s, t = strengths(y, period)
        cases.append(
            {
                "series": name,
                "period": period,
                "seasonal_strength": s,
                "trend_strength": t,
            }
        )
    return cases


# --------------------------------------------------- nsdiffs (transcribed)

def nsdiffs_case(name: str, y: np.ndarray, period: int, max_d: int):
    """forecast::nsdiffs(test='seas') sequential rule: D += 1 and
    seasonally difference while seasonal strength >= 0.64, up to max_d."""
    cur = np.asarray(y, dtype=float)
    d = 0
    steps = []
    while True:
        if np.all(cur == cur[0]):
            stop = "Constant"
            break
        if cur.shape[0] < 2 * period:
            stop = "TooShort"
            break
        s, t = strengths(cur, period)
        needs = bool(s >= SEAS_THRESHOLD)
        steps.append(
            {
                "d": d,
                "n": int(cur.shape[0]),
                "seasonal_strength": s,
                "trend_strength": t,
                "needs_differencing": needs,
            }
        )
        if not needs:
            stop = "WeakSeasonality"
            break
        if d >= max_d:
            stop = "MaxD"
            break
        d += 1
        cur = cur[period:] - cur[:-period]
    return {
        "series": name,
        "period": period,
        "max_d": max_d,
        "d": d,
        "stop": stop,
        "steps": steps,
    }


def gen_nsdiffs(series: dict[str, tuple[np.ndarray, int]]):
    cases = []
    cases.append(nsdiffs_case("co2", series["co2"][0], 12, 1))
    cases.append(nsdiffs_case("co2", series["co2"][0], 12, 2))
    cases.append(nsdiffs_case("synthetic_m", series["synthetic_m"][0], 12, 2))
    cases.append(nsdiffs_case("realgdp_q", series["realgdp_q"][0], 4, 1))
    cases.append(nsdiffs_case("noise", series["noise"][0], 12, 1))
    return cases


# ------------------------------------------------------------------ main

def main():
    series = {
        "co2": (co2_monthly(), 12),
        "synthetic_m": (synthetic_monthly(), 12),
        "realgdp_q": (realgdp_quarterly(), 4),
        "noise": (white_noise(), 12),
    }
    stl_series = {k: v for k, v in series.items() if k != "noise"}

    fixture = {
        "series": {k: v[0].tolist() for k, v in series.items()},
        "periods": {k: v[1] for k, v in series.items()},
        "stl": gen_stl(stl_series),
        "strength": gen_strength(series),
        "nsdiffs": gen_nsdiffs(series),
        "_meta": {
            "statsmodels": __import__("statsmodels").__version__,
            "seas_threshold": SEAS_THRESHOLD,
            "note": "STL arrays are elementwise statsmodels 0.14.6 output "
            "(strong third-party golden). strength/nsdiffs use statsmodels "
            "STL components with the Wang-Smith-Hyndman formula (ddof=1) "
            "and the forecast::nsdiffs(test='seas') 0.64 rule transcribed "
            "from their published definitions -- documented-formula/rule "
            "grading, as in generate_advisors_fixtures.py.",
        },
    }

    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(fixture, fh, indent=1)
    print(
        f"wrote {OUT} ({OUT.stat().st_size} bytes); "
        f"{len(fixture['stl'])} STL cases, {len(fixture['strength'])} strength "
        f"cases, {len(fixture['nsdiffs'])} nsdiffs cases"
    )


if __name__ == "__main__":
    main()
