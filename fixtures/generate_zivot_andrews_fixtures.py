"""Golden fixtures for the Zivot-Andrews (1992) one-break unit-root test.

Reference implementations (this venv):

  * statistic / break index / selected lag / p-value / critical values
      -> statsmodels 0.14.6, statsmodels.tsa.stattools.zivot_andrews
    This is the PRIMARY reference. Its algorithm is the Baum (2004/2015)
    approximation: a single up-front adfuller(regression="ct") autolag pass
    picks the augmentation lag for ALL candidate break regressions (the
    original paper re-selects per candidate); trimming is int(n*trim);
    candidate periods bp = trimcnt+1 ..= n-trimcnt; the intercept dummy is
    DU_t = 1{t >= bp} and the reported bpidx = bp - 1 is the LAST pre-break
    index; the "t" model's trend ramp starts one observation earlier than
    the "ct" model's (a reference quirk tsecon replicates on purpose).

  * cross-check -> arch 8.0.0, arch.unitroot.ZivotAndrews
    arch implements the SAME Baum algorithm (shared code lineage with the
    statsmodels contribution), so agreement here is a consistency check on
    the transcription of the algorithm, NOT an independent second
    derivation. Every case where the arch API can express the options is
    asserted to agree with statsmodels to 1e-10 relative and flagged
    `arch_agrees` in the fixture. Honest grade: ONE independent reference
    (statsmodels) plus one same-lineage corroboration (arch).

  * p-value / critical-value map -> the simulated null-distribution table
    inside statsmodels (100,000 MC replications of 2,000 obs), which
    tsecon transcribes (BSD-3, with attribution — the MacKinnon-surface
    precedent). `p_map` pins the interpolation on a statistic grid
    straight from statsmodels' _za_crit, so a transcription typo in the
    Rust table cannot hide behind the case statistics.

This generator NEVER imports tsecon. Doubles are written with json's
shortest round-trip repr, which the Rust golden test parses to identical
bits (serde_json float_roundtrip).

Run:  python fixtures/generate_zivot_andrews_fixtures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import statsmodels.api as sm
from arch.unitroot import ZivotAndrews
from statsmodels.tsa.stattools import zivot_andrews

OUT = Path(__file__).resolve().parent / "zivot_andrews.json"


# --------------------------------------------------------------- series

def build_series() -> dict[str, np.ndarray]:
    lrgdp = np.log(
        sm.datasets.macrodata.load_pandas().data["realgdp"].to_numpy(dtype=float)
    )
    nile = sm.datasets.nile.load_pandas().data["volume"].to_numpy(dtype=float)

    rng = np.random.default_rng(11)
    rw_nobreak = np.cumsum(rng.standard_normal(200))

    # A unit root WITH a level break: the classic ZA mis-specification
    # trap (break under the null), kept as a teaching case.
    rw_break = rw_nobreak.copy()
    rw_break[120:] += 8.0

    # Break-stationary: i.i.d. noise with a large level shift at t = 100,
    # so the last pre-break index is 99. ZA should reject and localize.
    rng2 = np.random.default_rng(7)
    stat_break = rng2.standard_normal(200)
    stat_break[100:] += 10.0

    return {
        "lrgdp": lrgdp,
        "nile": nile,
        "rw_nobreak": rw_nobreak,
        "rw_break": rw_break,
        "stat_break": stat_break,
    }


# ----------------------------------------------------------------- cases

def arch_cross_check(y, regression, trim, autolag, maxlag, lags, sm_res):
    """Run arch.unitroot.ZivotAndrews with the equivalent options and
    assert it matches statsmodels; returns True when the check ran."""
    if autolag is not None:
        a = ZivotAndrews(y, trend=regression, trim=trim, method=autolag,
                         max_lags=maxlag)
    elif lags is not None:
        a = ZivotAndrews(y, trend=regression, trim=trim, lags=lags)
    else:
        # arch has no exact "autolag=None, maxlag=None" spelling; skip.
        return False
    assert np.isclose(a.stat, sm_res[0], rtol=1e-10, atol=0.0), (
        f"arch stat {a.stat!r} != statsmodels {sm_res[0]!r}")
    assert np.isclose(a.pvalue, sm_res[1], rtol=1e-8, atol=1e-12), (
        f"arch pvalue {a.pvalue!r} != statsmodels {sm_res[1]!r}")
    assert int(a.lags) == int(sm_res[3]), (
        f"arch lags {a.lags} != statsmodels {sm_res[3]}")
    return True


def za_case(series: dict[str, np.ndarray], name: str, regression: str,
            trim: float = 0.15, autolag: str | None = "aic",
            maxlag: int | None = None, lags: int | None = None):
    y = series[name]
    # statsmodels folds "fixed lag" into maxlag-with-autolag=None.
    sm_maxlag = lags if (autolag is None and lags is not None) else maxlag
    res = zivot_andrews(y, trim=trim, maxlag=sm_maxlag,
                        regression=regression, autolag=autolag)
    stat, pvalue, cvdict, baselag, bpidx = res
    return {
        "series": name,
        "regression": regression,
        "trim": trim,
        "autolag": autolag,
        "maxlag": maxlag,
        "lags": lags,
        "stat": float(stat),
        "pvalue": float(pvalue),
        "crit": {k: float(cvdict[k]) for k in ("1%", "5%", "10%")},
        "baselag": int(baselag),
        "bpidx": int(bpidx),
        "nobs": int(y.shape[0]),
        "arch_agrees": arch_cross_check(y, regression, trim, autolag,
                                        sm_maxlag, lags, res),
    }


def gen_cases(series: dict[str, np.ndarray]):
    cases = []
    # Full grid: every series x every model, default trim/autolag.
    for name in series:
        for regression in ("c", "t", "ct"):
            cases.append(za_case(series, name, regression))
    # Fixed-lag path (autolag=None, explicit lag).
    cases.append(za_case(series, "lrgdp", "c", autolag=None, lags=3))
    cases.append(za_case(series, "rw_nobreak", "ct", autolag=None, lags=5))
    cases.append(za_case(series, "nile", "t", autolag=None, lags=2))
    # Truncated-Schwert default path (autolag=None, no lag): statsmodels
    # baselags = int(12*(n/100)^{1/4}); needs trim big enough for the lag.
    cases.append(za_case(series, "rw_nobreak", "c", trim=0.20, autolag=None))
    # Other information criteria.
    cases.append(za_case(series, "lrgdp", "ct", autolag="bic"))
    cases.append(za_case(series, "nile", "c", autolag="t-stat"))
    # A capped autolag search and non-default trims.
    cases.append(za_case(series, "lrgdp", "c", maxlag=5))
    cases.append(za_case(series, "stat_break", "c", trim=0.05))
    cases.append(za_case(series, "rw_break", "ct", trim=0.25))
    return cases


# --------------------------------------------- p-value/crit interpolation map

def gen_p_map():
    """Pin the transcribed null table + interpolation on a statistic grid,
    straight from statsmodels' _za_crit (clamps at both table ends
    included)."""
    grid = [-90.0, -7.0, -6.0, -5.5, -5.27644, -5.0, -4.81067, -4.5, -4.0,
            -3.5, -3.0, -2.5, -2.0, -1.5, 0.0]
    out = {}
    for reg in ("c", "t", "ct"):
        pvals, cvs = [], None
        for s in grid:
            p, cv = zivot_andrews._za_crit(float(s), reg)
            pvals.append(float(p))
            cvs = cv
        out[reg] = {
            "stat_grid": grid,
            "pvalues": pvals,
            "crit": {k: float(cvs[k]) for k in ("1%", "5%", "10%")},
        }
    return out


# ------------------------------------------------------------------ main

def main():
    series = build_series()
    cases = gen_cases(series)
    fixture = {
        "series": {k: v.tolist() for k, v in series.items()},
        "za": cases,
        "p_map": gen_p_map(),
        "_meta": {
            "reference": "statsmodels 0.14.6 zivot_andrews (primary); "
            "arch 8.0.0 ZivotAndrews cross-check (same Baum code lineage, "
            "not independent)",
            "table": "null-distribution quantiles simulated by statsmodels "
            "(100,000 MC reps, 2,000 obs), transcribed into tsecon with "
            "attribution (BSD-3)",
            "n_arch_checked": sum(c["arch_agrees"] for c in cases),
        },
    }
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(fixture, fh, indent=1)
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes); {len(fixture['za'])} "
          f"cases, {fixture['_meta']['n_arch_checked']} arch-cross-checked")


if __name__ == "__main__":
    main()
