"""Golden fixtures for MSTL — Multiple Seasonal-Trend decomposition using
LOESS (Bandara-Hyndman-Bergmeir 2021).

Reference implementation (this venv):

  * mstl components -> statsmodels 0.14.6 (STRONG third-party golden)
      statsmodels.tsa.seasonal.MSTL(y, periods=..., windows=...,
                                    iterate=..., stl_kwargs=...).fit()
    MSTL is a thin, deterministic driver over the compiled STL Cython port
    (the netlib Fortran `stl.f`): sort periods ascending, drop any period
    >= n/2 (with a UserWarning), then `iterate` rounds of per-period STL
    re-extraction on the running deseasonalized series; trend and
    robustness weights come from the final STL fit. The trend, EVERY
    per-period seasonal, resid, and (where the outer loop runs) the
    robustness weights are pinned ELEMENTWISE. The algorithm is
    deterministic, so the Rust port is expected to agree to ~1e-12; the
    test tolerance is 1e-8. Grade: strong third-party golden
    (statsmodels MSTL, elementwise).

  * the degenerate single-period case additionally asserts, AT GENERATION
    TIME, that statsmodels MSTL(periods=12) equals statsmodels
    STL(seasonal=11) elementwise (window 11 = the MSTL default rule
    7 + 4*1) — provenance that MSTL truly degenerates to STL. The
    matching *internal-consistency* claim (tsecon.mstl == tsecon.stl,
    bitwise) is asserted in the Rust/Python tests directly and graded
    separately from the third-party golden: it needs no fixture data.

  * lmbda/Box-Cox is scoped OUT of the tsecon port, so no case here sets
    it (statsmodels' default lmbda=None means no transform).

This generator NEVER imports tsecon. Doubles are written with json's
shortest round-trip repr, which the Rust golden test parses to identical
bits (serde_json `float_roundtrip`).

Run:  python fixtures/generate_mstl_fixtures.py
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
from statsmodels.tsa.seasonal import MSTL, STL

OUT = Path(__file__).resolve().parent / "mstl.json"


# --------------------------------------------------------------- series

def two_seasonal_hourly() -> np.ndarray:
    """Seeded hourly-like series, n = 672 (4 weeks): daily (24) and weekly
    (168) cycles with slowly-varying amplitudes, a smooth trend, noise,
    and two planted outliers so robust=True has work to do."""
    rng = np.random.default_rng(20260823)
    t = np.arange(672, dtype=float)
    daily = (1.0 + 0.001 * t) * np.sin(2 * np.pi * t / 24.0)
    weekly = 2.0 * np.sin(2 * np.pi * t / 168.0) + 0.5 * np.cos(
        4 * np.pi * t / 168.0
    )
    trend = 20.0 + 0.01 * t + 3.0 * np.sin(t / 300.0)
    y = trend + daily + weekly + 0.4 * rng.standard_normal(t.shape[0])
    y[100] += 7.0
    y[450] -= 6.0
    return y


def three_seasonal_awkward() -> np.ndarray:
    """Seeded series with three awkward, pairwise co-prime-ish periods
    (5, 12, 31) that never nest, n = 400."""
    rng = np.random.default_rng(7151)
    t = np.arange(400, dtype=float)
    s5 = 0.8 * np.sin(2 * np.pi * t / 5.0)
    s12 = 1.5 * np.sin(2 * np.pi * t / 12.0) + 0.3 * np.cos(4 * np.pi * t / 12.0)
    s31 = 1.1 * np.sin(2 * np.pi * t / 31.0)
    trend = 5.0 + 0.02 * t
    return trend + s5 + s12 + s31 + 0.3 * rng.standard_normal(t.shape[0])


def monthly_single() -> np.ndarray:
    """Seeded monthly series, n = 180 — the degenerate single-period case."""
    rng = np.random.default_rng(99)
    t = np.arange(180, dtype=float)
    seasonal = 2.0 * np.sin(2 * np.pi * t / 12.0)
    trend = 10.0 + 0.05 * t
    return trend + seasonal + 0.5 * rng.standard_normal(t.shape[0])


def short_droppy() -> np.ndarray:
    """Seeded series, n = 120, used with periods (12, 60): 60 >= 120/2, so
    statsmodels warns and drops it, decomposing with period 12 only."""
    rng = np.random.default_rng(4242)
    t = np.arange(120, dtype=float)
    return (
        3.0
        + 0.03 * t
        + 1.4 * np.sin(2 * np.pi * t / 12.0)
        + 0.3 * rng.standard_normal(t.shape[0])
    )


# ------------------------------------------------------------------ MSTL

# (series, case name, MSTL kwargs). `periods` are given deliberately
# unsorted in some cases to pin the ascending sort (paired windows must
# travel with their period). Covers: default windows/iterate; robust;
# explicit unsorted windows; forwarded stl_kwargs incl. fit kwargs
# (inner_iter/outer_iter); iterate=1 and 4; three seasons; the degenerate
# single period (scalar and the drop case).
CASES = [
    ("two_seasonal", "defaults", {"periods": (24, 168)}),
    ("two_seasonal", "robust", {"periods": (24, 168), "stl_kwargs": {"robust": True}}),
    (
        "two_seasonal",
        "windows_unsorted",
        # periods reversed AND windows paired to them: (168, 35), (24, 25)
        # must sort to periods (24, 168) with windows (25, 35).
        {"periods": (168, 24), "windows": (35, 25)},
    ),
    (
        "two_seasonal",
        "stl_kwargs",
        {
            "periods": (24, 168),
            "stl_kwargs": {
                "trend": 201,
                "seasonal_deg": 0,
                "low_pass_jump": 2,
                "robust": True,
                "inner_iter": 3,
                "outer_iter": 2,
            },
        },
    ),
    ("two_seasonal", "iterate1", {"periods": (24, 168), "iterate": 1}),
    ("two_seasonal", "iterate4", {"periods": (24, 168), "iterate": 4}),
    ("three_seasonal", "defaults_unsorted", {"periods": (31, 5, 12)}),
    (
        "three_seasonal",
        "windows_iterate3",
        {"periods": (5, 12, 31), "windows": (9, 13, 23), "iterate": 3},
    ),
    ("single", "scalar_period", {"periods": 12}),
    ("droppy", "period_dropped", {"periods": (12, 60)}),
]


def mstl_case(series_name: str, y: np.ndarray, cfg_name: str, kwargs: dict):
    # statsmodels' MSTL.fit() pops inner_iter/outer_iter out of the CALLER'S
    # stl_kwargs dict in place, so snapshot what was requested (and hand MSTL
    # a copy) BEFORE fitting — otherwise the recorded kwargs silently lose
    # those keys and the golden test replays the wrong iteration counts.
    stl_kwargs = dict(kwargs.get("stl_kwargs", {}))
    call_kwargs = dict(kwargs)
    if "stl_kwargs" in call_kwargs:
        call_kwargs["stl_kwargs"] = dict(stl_kwargs)
    expect_drop = cfg_name == "period_dropped"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = MSTL(y, **call_kwargs)
        res = model.fit()
    drop_warnings = [
        w for w in caught if "larger than half" in str(w.message)
    ]
    assert bool(drop_warnings) == expect_drop, (series_name, cfg_name)

    resolved_periods = [int(p) for p in model.periods]
    resolved_windows = [int(w) for w in model.windows]
    req = kwargs["periods"]
    requested = [req] if isinstance(req, int) else [int(p) for p in req]
    dropped = sorted(set(requested) - set(resolved_periods))

    seasonal = np.asarray(res.seasonal)
    if seasonal.ndim == 1:  # squeezed single-period case
        seasonal_cols = [seasonal.tolist()]
    else:  # (n, K) -> K lists in resolved-period order
        seasonal_cols = [seasonal[:, k].tolist() for k in range(seasonal.shape[1])]

    robustish = (
        stl_kwargs.get("outer_iter", 15 if stl_kwargs.get("robust", False) else 0) > 0
    )
    return {
        "series": series_name,
        "config_name": cfg_name,
        "periods_arg": kwargs["periods"] if isinstance(kwargs["periods"], int)
        else list(kwargs["periods"]),
        "windows_arg": list(kwargs["windows"]) if "windows" in kwargs else None,
        "iterate": kwargs.get("iterate", 2),
        "stl_kwargs": stl_kwargs,
        "resolved_periods": resolved_periods,
        "resolved_windows": resolved_windows,
        "dropped_periods": dropped,
        "seasonal": seasonal_cols,
        "trend": np.asarray(res.trend).tolist(),
        "resid": np.asarray(res.resid).tolist(),
        # Robustness weights only where the outer loop runs (all-ones
        # otherwise; the Rust property tests cover that case).
        "weights": np.asarray(res.weights).tolist() if robustish else None,
    }


def check_single_period_degenerates_to_stl(y: np.ndarray) -> None:
    """Provenance: statsmodels MSTL with one period IS statsmodels STL with
    seasonal window 11 (= 7 + 4*1, the MSTL default rule)."""
    m = MSTL(y, periods=12).fit()
    s = STL(y, period=12, seasonal=11).fit()
    np.testing.assert_array_equal(np.asarray(m.seasonal), np.asarray(s.seasonal))
    np.testing.assert_array_equal(np.asarray(m.trend), np.asarray(s.trend))
    # MSTL's resid is (y - seasonal) - trend, the same order STL uses.
    np.testing.assert_array_equal(np.asarray(m.resid), np.asarray(s.resid))


# ------------------------------------------------------------------ main

def main():
    series = {
        "two_seasonal": two_seasonal_hourly(),
        "three_seasonal": three_seasonal_awkward(),
        "single": monthly_single(),
        "droppy": short_droppy(),
    }
    check_single_period_degenerates_to_stl(series["single"])

    cases = [
        mstl_case(name, series[name], cfg_name, kwargs)
        for name, cfg_name, kwargs in CASES
    ]
    fixture = {
        "series": {k: v.tolist() for k, v in series.items()},
        "cases": cases,
        "_meta": {
            "statsmodels": __import__("statsmodels").__version__,
            "note": "MSTL trend / per-period seasonal / resid (and weights "
            "where the outer loop runs) are elementwise statsmodels 0.14.6 "
            "output — strong third-party golden. The single-period case is "
            "verified at generation time to equal statsmodels STL with "
            "seasonal window 11; the tsecon-internal mstl==stl bitwise "
            "check lives in the Rust/Python tests (graded separately as "
            "internal consistency). lmbda/Box-Cox is scoped out of the "
            "tsecon port and never set here.",
        },
    }

    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(fixture, fh, indent=1)
    print(
        f"wrote {OUT} ({OUT.stat().st_size} bytes); {len(cases)} MSTL cases"
    )


if __name__ == "__main__":
    main()
