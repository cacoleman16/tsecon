"""Run every interval-coverage family and print ONE consolidated report.

    .venv/bin/python docs/examples/coverage/run_all.py            # full run, ~30 min
    .venv/bin/python docs/examples/coverage/run_all.py --quick    # smoke run, ~3 min
    .venv/bin/python docs/examples/coverage/run_all.py --summary  # consolidated table only
    .venv/bin/python docs/examples/coverage/run_all.py --only irf_bands,lp_family
    .venv/bin/python docs/examples/coverage/run_all.py --markdown tables.md

WHAT THIS FILE ADDS
-------------------
The seven modules under this directory each measure the coverage of one
family of intervals and print their own report. This runner executes all
seven in one process and then answers the question none of them can answer
alone: across every interval `tsecon` ships, WHICH ONES KEEP THEIR PROMISE?

Every number in the consolidated tables is harvested from the structured
results the family modules return -- nothing is transcribed by hand, so the
tables cannot go stale while the measurements move underneath them. Each row
names TWO measured cells:

  favourable  a design the interval is entitled to do well on (assumptions
              met, enough data). If a row misses HERE, the problem is not the
              user's data.
  stress      a design that pushes the same interval where applied work
              actually goes (persistence, weak identification, a short
              sample, a long horizon, a misspecified lag order).

The verdict column is computed from the STRESS cell against its own nominal
level, using the Monte Carlo standard error of the measurement itself:

    dev = (coverage - nominal) / mc_se,   mc_se = sqrt(p(1-p)/reps)

    |dev| <= 3   ->  AT NOMINAL   (indistinguishable from the promise)
    dev  <  -3   ->  UNDER        (covers less often than it claims)
    dev  >  +3   ->  OVER         (conservative: honest, but wider than needed)

A verdict is a statement about statistical distinguishability, not importance,
so the table also prints the raw gap in percentage points. 0.941 against a
0.95 promise at reps=3000 is "UNDER" by this rule and is 0.9pp; 0.588 is also
"UNDER" and is 36pp. Read both columns.

WHAT THE KIND COLUMN MEANS -- read this before reading a verdict
----------------------------------------------------------------
Not everything that looks like an interval makes a repeated-sampling promise.
Four kinds appear, and only the first two are owed nominal coverage:

  CI    a frequentist confidence interval for a PARAMETER. A 95% CI must
        contain the true parameter in 95% of samples. Verdict applies.
  PRED  a predictive interval for a future REALISATION. Same repeated-sampling
        promise, different target. Verdict applies.
  CRED  a Bayesian credible band. It is a statement about the posterior. It
        makes NO frequentist coverage promise, so a shortfall is not a defect
        -- it is a measurement of the prior. Verdict is suppressed and the row
        is reported as a diagnostic.
  SET   set-identified bounds (sign restrictions). Not an interval about a
        point at all; the meaningful question is whether the identified SET
        contains the truth. Verdict suppressed.

CAUSE says which of three things a miss is, because they call for different
responses:

  APPROXIMATION  the formula is right and its asymptotics have not arrived at
                 this sample size / horizon / persistence. No bug fix exists.
                 Widen deliberately, or use a different interval.
  ESTIMATOR      the estimator is wrong for the job, or is off-centre, so no
                 standard error can rescue it. Fixable by the caller (or not
                 at all, when it is inconsistency).
  CONVENTION     a deliberate library convention (a default bandwidth, a
                 degrees-of-freedom choice, a discreteness padding) with a
                 measurable coverage cost. Correct code, documented default,
                 and the cost is worth publishing.
  API GAP        the interval that would fix it is not exposed at all (HC2/HC3,
                 an Anderson-Rubin set, a simultaneous sup-t band). Nothing the
                 caller can pass helps.
  READING        the interval is fine and the reader's question was different
                 (pointwise band read as a joint band; a credible band read as
                 a confidence interval).

Reproducibility: every family seeds from the same master seed and prints it.
The runner adds no randomness of its own, so `run_all.py` and the seven
`python <family>.py` invocations produce the same numbers. The runner exits
non-zero if any family fails its own assertions, or if any probe below can no
longer find the number it reports -- so a schema change is loud rather than
silently dropping a row.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import io
import math
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence

# --------------------------------------------------------------------------
# the families
# --------------------------------------------------------------------------

FAMILIES: list[tuple[str, str, str]] = [
    ("regression_se", "Regression standard errors",
     "ols / iv_gmm / har_rv / recession_probit / quantile_regression"),
    ("irf_bands", "VAR impulse-response bands",
     "var_irf_bands: asymptotic (delta-method) and bootstrap"),
    ("lp_family", "Local projections",
     "lp / lp_iv / lp_state / lp_multiplier / smooth_lp"),
    ("forecast_intervals", "Predictive intervals",
     "arima_fit / var_forecast forecast bands; theta_forecast, backtest"),
    ("bayes_and_sets", "Bayesian bands and identified sets",
     "bvar_irf_draws / bvar_ssvs / sign+zero+narrative SVAR / bai_perron"),
    ("quantile_panel_lp", "Quantile, panel and cumulative local projections",
     'quantile_lp / panel_lp (Driscoll-Kraay, SPJ) / lp(cumulative="both")'),
    ("factor_midas", "Factor models and mixed frequency",
     "favar two-step bands / umidas; weighted_midas, dfm_nowcast, "
     "nelson_siegel ship no interval"),
]

KIND_NOTE = {
    "CI": "frequentist confidence interval for a parameter",
    "PRED": "predictive interval for a future realisation",
    "CRED": "Bayesian credible band -- NO frequentist promise",
    "SET": "set-identified bounds -- not an interval about a point",
    "NONE": "the library ships no interval for this surface",
}


class ProbeError(Exception):
    """A probe could not find the number it is supposed to report."""


# --------------------------------------------------------------------------
# accessors -- deliberately strict, so a schema change fails loudly
# --------------------------------------------------------------------------

def mc_se(p: float, reps: int) -> float:
    return math.sqrt(max(p * (1.0 - p), 0.0) / reps)


def pick(rows: Sequence[dict], **kw) -> dict:
    """The unique row matching every key=value pair."""
    hits = [r for r in rows
            if all(k in r and r[k] == v for k, v in kw.items())]
    if len(hits) != 1:
        raise ProbeError(f"pick({kw}) matched {len(hits)} rows, expected 1")
    return hits[0]


def cm(x: Any, reps: int | None = None) -> tuple[float, float]:
    """Coverage and its MC standard error, from whatever shape it arrives in."""
    if isinstance(x, dict) and "cover" in x:
        cover = float(x["cover"])
        if "mcse" in x and x["mcse"] is not None:
            return cover, float(x["mcse"])
        if reps is None:
            raise ProbeError("no mcse and no reps")
        return cover, mc_se(cover, reps)
    cover = float(x)
    if reps is None:
        raise ProbeError("bare coverage with no reps to build an MC se from")
    return cover, mc_se(cover, reps)


def argworst(values: Sequence[float]) -> int:
    return min(range(len(values)), key=lambda i: values[i])


def argbest(values: Sequence[float]) -> int:
    return max(range(len(values)), key=lambda i: values[i])


def lp_row(res: dict, exp: str, arm: str, h: int) -> tuple[float, float, str]:
    row = pick(res[exp]["rows"], arm=arm, h=h)
    return float(row["cov95"]), float(row["mcse"]), f"{arm}, h={h}"


def _lp_arm(res: dict, exp: str, arm: str) -> list[dict]:
    rows = [r for r in res[exp]["rows"] if r["arm"] == arm]
    if not rows:
        raise ProbeError(f"no rows for arm {arm!r} in {exp}")
    return rows


def lp_worst(res: dict, exp: str, arm: str) -> tuple[float, float, str]:
    row = min(_lp_arm(res, exp, arm), key=lambda r: r["cov95"])
    return (float(row["cov95"]), float(row["mcse"]),
            f"{arm}, worst h ({int(row['h'])})")


def lp_best(res: dict, exp: str, arm: str) -> tuple[float, float, str]:
    row = max(_lp_arm(res, exp, arm), key=lambda r: r["cov95"])
    return (float(row["cov95"]), float(row["mcse"]),
            f"{arm}, best h ({int(row['h'])})")


def lp_extreme(res: dict, exp: str, arm: str) -> tuple[float, float, str]:
    """The horizon furthest from nominal in EITHER direction (for over-coverers)."""
    nom = float(res[exp]["meta"]["nominal"])
    row = max(_lp_arm(res, exp, arm), key=lambda r: abs(r["cov95"] - nom))
    return (float(row["cov95"]), float(row["mcse"]),
            f"{arm}, worst h ({int(row['h'])})")


def lp_closest(res: dict, exp: str, arm: str) -> tuple[float, float, str]:
    """The horizon CLOSEST to nominal -- the kindest reading of an arm."""
    nom = float(res[exp]["meta"]["nominal"])
    row = min(_lp_arm(res, exp, arm), key=lambda r: abs(r["cov95"] - nom))
    return (float(row["cov95"]), float(row["mcse"]),
            f"{arm}, closest h ({int(row['h'])})")


def fc_at(res: dict, exp: str, variant: str, h: int) -> tuple[float, float, str]:
    e = res[exp]
    hs = list(e["horizons"])
    if h not in hs:
        raise ProbeError(f"{exp}: horizon {h} not in {hs}")
    i = hs.index(h)
    return float(e["coverage"][variant][i]), float(e["mc_se"][variant][i]), f"h={h}"


def fc_worst(res: dict, exp: str, variant: str) -> tuple[float, float, str]:
    e = res[exp]
    cov = list(e["coverage"][variant])
    i = argworst(cov)
    return float(cov[i]), float(e["mc_se"][variant][i]), f"worst h ({e['horizons'][i]})"


def irf_curve(res: dict, exp: str, n: int, arm: str) -> Sequence[float]:
    """The coverage-by-horizon array, whichever way round the experiment nests.

    exp1/exp5 store {"coverage": {arm: [...]}}; exp3 stores {arm: {"coverage":
    [...]}}. Both layouts are legitimate and both are read here rather than
    assumed, so neither module has to change to keep this runner honest.
    """
    block = res[exp]["by_n"][n]
    if "coverage" in block and arm in block["coverage"]:
        return block["coverage"][arm]
    if arm in block and "coverage" in block[arm]:
        return block[arm]["coverage"]
    raise ProbeError(f"{exp}: no coverage curve for arm {arm!r} at n={n}")


def irf_at(res: dict, exp: str, n: int, arm: str, h: int,
           reps: int) -> tuple[float, float, str]:
    arr = irf_curve(res, exp, n, arm)
    if h >= len(arr):
        raise ProbeError(f"{exp}: h={h} beyond horizon {len(arr) - 1}")
    c = float(arr[h])
    return c, mc_se(c, reps), f"n={n}, h={h}"


def irf_last(res: dict, exp: str, n: int, arm: str,
             reps: int) -> tuple[float, float, str]:
    arr = irf_curve(res, exp, n, arm)
    c = float(arr[-1])
    return c, mc_se(c, reps), f"n={n}, h={len(arr) - 1}"


def ns(res: dict, exp: str) -> list[int]:
    return sorted(res[exp]["by_n"])


# --------------------------------------------------------------------------
# the probe registry
# --------------------------------------------------------------------------

@dataclass
class Probe:
    family: str
    surface: str          # the tsecon call
    option: str           # which interval of that call
    kind: str             # CI / PRED / CRED / SET / NONE
    nominal: float
    fav_label: str
    fav: Callable[[dict], tuple[float, float, str]] | None
    str_label: str
    stress: Callable[[dict], tuple[float, float, str]] | None
    cause: str
    action: str           # what a user should do; "" when nothing is wrong
    card: str             # where the caveat belongs in the docs


def probes_regression_se() -> list[Probe]:
    def ols(dgp, col):
        def f(R):
            row = pick(R["ols"]["rows"], dgp=dgp)
            c, m = cm(row["cover"][col], R["ols"]["reps"])
            return c, m, f"{dgp}, T={R['ols']['n']}"
        return f

    def lev(n, col):
        def f(R):
            row = pick(R["leverage"]["rows"], n=n)
            c, m = cm(row["cover"][col], R["leverage"]["reps"])
            return c, m, f"x~chi2(1), sd(e|x)=x, T={n}"
        return f

    def hacs(phi, col):
        def f(R):
            row = pick(R["hac_slope"]["rows"], phi=phi)
            c, m = cm(row["cover"][col], R["hac_slope"]["reps"])
            return c, m, f"x,e AR(1) phi={phi}, T={R['hac_slope']['n']}"
        return f

    def iv(pi, method, key="cover"):
        def f(R):
            row = pick(R["iv"]["rows"], pi=pi, method=method)
            c, m = cm(row[key], R["iv"]["reps"])
            return c, m, f"median first-stage F = {row['first_stage_f']:.1f}"
        return f

    def ivhac(col):
        def f(R):
            c, m = cm(R["iv_hac"]["cover"][col], R["iv_hac"]["reps"])
            return c, m, f"AR(1) errors phi={R['iv_hac']['phi']}, {col}"
        return f

    def har(het, maxlags, j, name):
        def f(R):
            row = pick(R["har"]["rows"], het=het, maxlags=maxlags)
            c, m = cm(row["cover"][j], R["har"]["reps"])
            errs = "het" if het else "iid"
            return c, m, f"{errs} innovations, maxlags={maxlags}, {name}"
        return f

    def probit(case, n):
        def f(R):
            row = pick(R["probit"]["rows"], case=case, n=n)
            c, m = cm(row["cover"], row["n_ok"])
            extra = (f", {100 * row['fail_share']:.0f}% no MLE"
                     if row["fail_share"] > 0 else "")
            return c, m, f"{case.strip()}, T={n}{extra}"
        return f

    def qr(design, tau, key="cover_slope"):
        def f(R):
            row = pick(R["quantile"]["rows"], design=design)
            taus = list(row["taus"])
            if tau not in taus:
                raise ProbeError(f"tau {tau} not in {taus}")
            c, m = cm(row[key][taus.index(tau)], R["quantile"]["reps"])
            return c, m, f"{design.strip()}, tau={tau}"
        return f

    guide = "../../guide/03-inference-toolkit.md"
    hacbook = "../../cookbook/hac-standard-errors.md"
    return [
        Probe("regression_se", "tsecon.ols", 'se_type="nonrobust"', "CI", 0.95,
              "assumptions met", ols("iid Gaussian", "nonrobust"),
              "heteroskedasticity", ols("heteroskedastic sd=|x|", "nonrobust"),
              "ESTIMATOR",
              "use se_type='hc1'; the nonrobust SE is inconsistent here, so "
              "more data does not help", guide),
        Probe("regression_se", "tsecon.ols", 'se_type="hc1"', "CI", 0.95,
              "heteroskedasticity (what HC is for)",
              ols("heteroskedastic sd=|x|", "hc1"),
              "serial correlation", ols("AR(1) errors+regressor .7", "hc1"),
              "ESTIMATOR",
              "HC is heteroskedasticity-robust, NOT serial-correlation robust; "
              "use se_type='hac'", hacbook),
        Probe("regression_se", "tsecon.ols", 'se_type="hc1"; small T, leverage',
              "CI", 0.95,
              "large sample", lev(1600, "hc1"),
              "T=25, high leverage", lev(25, "hc1"),
              "ESTIMATOR",
              "hc1's n/(n-k) factor is blind to leverage; se_type='hc3' covers "
              "0.863 on the same draws where hc1 covers 0.682 at T=25 (still "
              "short of nominal -- prefer hc3 at small n)", guide),
        Probe("regression_se", "tsecon.ols", 'se_type="hc3"; small T, leverage',
              "CI", 0.95,
              "large sample", lev(1600, "hc3"),
              "T=25, high leverage", lev(25, "hc3"),
              "APPROXIMATION",
              "the leverage correction recovers most of the small-T gap but "
              "not all of it; the SE distribution is skewed, so mean se/sd of "
              "0.942 overstates the typical interval", guide),
        Probe("regression_se", "tsecon.ols", 'se_type="hac"', "CI", 0.95,
              "no serial correlation", hacs(0.0, "hac auto"),
              "near-unit-root regressor AND errors", hacs(0.95, "hac auto"),
              "APPROXIMATION",
              "T=200 cannot estimate a long-run variance this large; lengthen "
              "the bandwidth and treat the interval as indicative", hacbook),
        Probe("regression_se", "tsecon.iv_gmm", "2sls / 2step / iterated",
              "CI", 0.95,
              "strong instruments", iv(0.60, "2sls"),
              "weak instruments", iv(0.05, "2sls"),
              "ESTIMATOR",
              "read the reported first_stage F before the SE; no "
              "weak-instrument-robust (Anderson-Rubin) set is exposed",
              "../../reference/model-cards/gmm.md"),
        Probe("regression_se", "tsecon.iv_gmm", 'weight="hac"', "CI", 0.95,
              "bandwidth passed explicitly", ivhac("hac bw=10"),
              "the automatic default", ivhac("hac auto (NW rule)"),
              "APPROXIMATION",
              "the Newey-West default (4 lags at T=250) lifts coverage from "
              "0.632 to 0.842, and bandwidth=10 reaches 0.868 -- neither "
              "reaches nominal, so the default is not a remedy",
              "../../reference/model-cards/gmm.md"),
        Probe("regression_se", "tsecon.har_rv", "HAC SEs on the three slopes",
              "CI", 0.95,
              "library default bandwidth",
              har(False, 5, 1, "b_daily"),
              "maxlags=22 on a white-noise score",
              har(True, 22, 3, "b_monthly"),
              "APPROXIMATION",
              "bandwidth is not free: extra lags shrink the SE when the score "
              "has no serial correlation to soak up",
              "../../reference/model-cards/realized-vol.md"),
        Probe("regression_se", "tsecon.har_rv", "HAC SE on the CONSTANT",
              "CI", 0.95,
              "maxlags=0", har(True, 0, 0, "const"),
              "maxlags=22", har(True, 22, 0, "const"),
              "ESTIMATOR",
              "the least-squares persistence bias at sum(b)=0.95 is absorbed "
              "entirely by the intercept; do not read the HAR constant as if "
              "it were as reliable as the slopes",
              "../../reference/model-cards/realized-vol.md"),
        Probe("regression_se", "tsecon.recession_probit", "Wald interval",
              "CI", 0.95,
              "common events", probit("probit common phi=0.9", 250),
              "rare events, T=100", probit("probit rare   phi=0.9", 100),
              "ESTIMATOR",
              "coverage is measured on the samples that HAVE a finite MLE; a "
              "quarter of them do not, so read the failure share with it",
              "../../reference/model-cards/recession.md"),
        Probe("regression_se", "tsecon.quantile_regression",
              "Powell sandwich, slope", "CI", 0.95,
              "median", qr("location-scale T=200", 0.5),
              "tau=0.05 at T=200", qr("location-scale T=200", 0.05),
              "APPROXIMATION",
              "the sandwich needs a density estimated from the few points near "
              "an extreme quantile; bootstrap the quantile process instead",
              "../../reference/model-cards/quantile.md"),
        Probe("regression_se", "tsecon.quantile_regression",
              "Powell sandwich, intercept", "CI", 0.95,
              "x=0 inside the design", qr("homoskedastic  T=200", 0.5,
                                          "cover_icpt"),
              "x=0 at the edge of the support",
              qr("location-scale T=200", 0.5, "cover_icpt"),
              "APPROXIMATION",
              "over-covers: with x ~ U(0,2) the intercept is an extrapolation "
              "and its sandwich SE is conservative -- usually not the "
              "quantity of interest anyway",
              "../../reference/model-cards/quantile.md"),
    ]


def probes_irf_bands() -> list[Probe]:
    def horizon_profile(which, arm):
        def f(out):
            R, reps = out["results"], out["results"]["exp1"]["reps"]
            sizes = ns(R, "exp1")
            n = sizes[-1] if which == "big" else sizes[0]
            if which == "big":
                return irf_at(R, "exp1", n, arm, 0, reps)
            return irf_last(R, "exp1", n, arm, reps)
        return f

    def persist(arm, where):
        def f(out):
            R, reps = out["results"], out["results"]["exp3"]["reps"]
            n = ns(R, "exp3")[0]
            if where == "impact":
                return irf_at(R, "exp3", n, arm, 0, reps)
            return irf_last(R, "exp3", n, arm, reps)
        return f

    def grid_last(cfg, arm):
        def f(out):
            e = out["results"]["exp2"]
            arr = e["grid"][cfg]["coverage"][arm]
            c = float(arr[-1])
            return c, mc_se(c, e["reps"]), f"{cfg}, h={len(arr) - 1}"
        return f

    def misspec(which, h=4):
        def f(out):
            R = out["results"]
            e = R["exp4"]
            n = ns(R, "exp4")[-1]
            key = [k for k in e["by_n"][n] if which in k]
            if len(key) != 1:
                raise ProbeError(f"exp4 arm {which!r} -> {key}")
            arr = e["by_n"][n][key[0]]["coverage"]["asymptotic"]
            c = float(arr[h])
            return c, mc_se(c, e["reps"]), f"n={n}, {key[0]}, h={h}"
        return f

    def joint(out):
        R = out["results"]
        n = ns(R, "exp1")[-1]
        e = R["exp1"]
        c = float(e["by_n"][n]["joint_coverage"]["asymptotic"])
        hz = len(e["by_n"][n]["coverage"]["asymptotic"]) - 1
        return c, mc_se(c, e["reps"]), f"n={n}, all of h=0..{hz} at once"

    card = "../../reference/model-cards/var-svar.md"
    return [
        Probe("irf_bands", "tsecon.var_irf_bands", 'method="asymptotic"',
              "CI", 0.90,
              "impact, largest sample", horizon_profile("big", "asymptotic"),
              "longest horizon, T=100", horizon_profile("small", "asymptotic"),
              "APPROXIMATION",
              "the SE is right (mean se / MC sd ~ 0.96) -- the standardised "
              "statistic is badly skewed at long horizons, so the Wald band is "
              "one-sidedly wrong. Prefer bootstrap, or cumulative responses",
              card),
        Probe("irf_bands", "tsecon.var_irf_bands", 'method="bootstrap"',
              "CI", 0.90,
              "impact, persistent VAR", persist("bootstrap", "impact"),
              "longest horizon, root 0.95, T=100",
              persist("bootstrap", "last"),
              "ESTIMATOR",
              "the percentile bootstrap band sits BELOW an already "
              "downward-biased estimate: pass bias_correct=True on a "
              "persistent VAR", card),
        Probe("irf_bands", "tsecon.var_irf_bands",
              'method="bootstrap", bias_correct=True', "CI", 0.90,
              "longest horizon, root 0.95", persist("bootstrap+bc", "last"),
              "impact", persist("bootstrap+bc", "impact"),
              "APPROXIMATION",
              "Kilian's correction buys ~50 coverage points at long horizons "
              "and costs a little at impact -- take the trade on a persistent "
              "VAR", card),
        Probe("irf_bands", "tsecon.var_irf_bands", "cumulative=True",
              "CI", 0.90,
              "cumulative, longest horizon",
              grid_last("orth=True,cumulative=True", "asymptotic"),
              "per-horizon, longest horizon",
              grid_last("orth=True,cumulative=False", "asymptotic"),
              "APPROXIMATION",
              "on this DGP the running sum is dominated by the early, "
              "well-estimated horizons, so it stays far closer to nominal "
              "than the per-horizon band", card),
        Probe("irf_bands", "tsecon.var_irf_bands", "with the lag order wrong",
              "CI", 0.90,
              "correct lag order", misspec("correct"),
              "VAR(4) truth fitted as VAR(1)", misspec("misspecified"),
              "ESTIMATOR",
              "inconsistency, not a band problem: coverage gets WORSE as T "
              "grows. Choose lags on the data (var_lag_order) before reading "
              "any band", card),
        Probe("irf_bands", "tsecon.var_irf_bands",
              "pointwise band read as a JOINT band", "CI", 0.90,
              "one horizon (what it promises)",
              horizon_profile("big", "asymptotic"),
              "the whole path at once (what it does not)", joint,
              "READING",
              "a pointwise band makes no joint promise; pass band=\"sup-t\" "
              "for a simultaneous one (measured in irf_bands.py exp 7)", card),
    ]


def probes_lp_family() -> list[Probe]:
    card = "../../reference/model-cards/local-projections.md"
    return [
        Probe("lp_family", "tsecon.lp", 'se="lag_augmented" (the default)',
              "CI", 0.95,
              "impact",
              lambda R: lp_row(R, "lag_augmented_vs_hac", "lag_augmented", 0),
              "worst horizon, T=200",
              lambda R: lp_worst(R, "lag_augmented_vs_hac", "lag_augmented"),
              "APPROXIMATION",
              "nothing to do -- this is the best-calibrated interval in the "
              "family and the reason lag augmentation is the default", card),
        Probe("lp_family", "tsecon.lp", 'se="hac"', "CI", 0.95,
              "T=800, longest horizon",
              lambda R: lp_row(R, "sample_size", "T=800 hac", 12),
              "T=100, longest horizon",
              lambda R: lp_row(R, "sample_size", "T=100 hac", 12),
              "APPROXIMATION",
              "keep the default se='lag_augmented': it covers better at every "
              "horizon on the same draws (paired gap +2.7pp at h>=6)", card),
        Probe("lp_family", "tsecon.lp_iv", "strong instrument", "CI", 0.95,
              "impact", lambda R: lp_best(R, "lp_iv", "strong iv"),
              "worst horizon", lambda R: lp_worst(R, "lp_iv", "strong iv"),
              "CONVENTION",
              "the kernel covariance follows linearmodels' debiased=False "
              "convention and smooths p lags even at h=0; subtract a couple of "
              "points from the nominal level before quoting it", card),
        Probe("lp_family", "tsecon.lp_iv", "weak instrument", "CI", 0.95,
              "kindest horizon of the weak arm itself",
              lambda R: lp_closest(R, "lp_iv", "weak iv"),
              "median first-stage F < 4",
              lambda R: lp_extreme(R, "lp_iv", "weak iv"),
              "APPROXIMATION",
              "OVER-covers while the median interval width explodes ~5x -- "
              "Dufour (1997): under weak identification no bounded set can be "
              "honest. Report first_stage_f; the interval is uninformative, "
              "not wrong", card),
        Probe("lp_family", "tsecon.lp_state", "per-regime response",
              "CI", 0.95,
              "the quiet regime",
              lambda R: lp_best(R, "lp_state", "state0 lag_augmented"),
              "the persistent regime",
              lambda R: lp_worst(R, "lp_state", "state1 lag_augmented"),
              "ESTIMATOR",
              "the interacted design identifies each regime off roughly half "
              "a sample; state-dependent LP needs more data than linear LP "
              "for the same interval to mean the same thing", card),
        Probe("lp_family", "tsecon.lp_multiplier", "integral multiplier",
              "CI", 0.95,
              "impact", lambda R: lp_row(R, "lp_multiplier", "multiplier", 0),
              "widest accumulation window",
              lambda R: lp_worst(R, "lp_multiplier", "multiplier"),
              "CONVENTION",
              "well centred and strongly instrumented, but se/sd ~ 0.9: the "
              "honest critical value at T=240 is nearer 2.2 than 1.96", card),
        Probe("lp_family", "tsecon.smooth_lp", 'lam="cv" (the default)',
              "CI", 0.95,
              "unpenalized anchor, impact",
              lambda R: lp_row(R, "smooth_lp", "lam=0", 0),
              "impact response", lambda R: lp_row(R, "smooth_lp", "lam=cv", 0),
              "ESTIMATOR",
              "by design: the penalty buys bias, and se conditions on the "
              "selected lambda. A smooth-LP band is a band around the "
              "PENALIZED estimand -- do not read it as a CI for the raw IRF",
              card),
    ]


def probes_forecast() -> list[Probe]:
    arima = "../../reference/model-cards/arima.md"
    var = "../../reference/model-cards/var-svar.md"
    fore = "../../reference/model-cards/forecasting.md"
    return [
        Probe("forecast_intervals", "tsecon.arima_fit",
              "forecast_lower / forecast_upper", "PRED", 0.95,
              "mild persistence (phi=0.5), h=1",
              lambda R: fc_at(R, "exp1_ar1_phi0.5", "library", 1),
              "phi=0.9, worst horizon, T=100",
              lambda R: fc_worst(R, "exp1_ar1_phi0.9", "library"),
              "APPROXIMATION",
              "a plug-in band: the same formula at the TRUE parameters covers "
              "94.6% on the same draws, so the gap is the price of estimating "
              "phi and sigma, not a wrong SE", arima),
        Probe("forecast_intervals", "tsecon.arima_fit",
              "d=1 (random walk with drift)", "PRED", 0.95,
              "T=100, h=1",
              lambda R: fc_at(R, "exp6_rw_drift_T100_h12", "library", 1),
              "T=60, h=24",
              lambda R: fc_worst(R, "exp6_rw_drift_T60_h24", "library"),
              "APPROXIMATION",
              "forecast_se is exactly sigma*sqrt(h): the h^2/(T-1) "
              "drift-uncertainty term is omitted. The shortfall matches its "
              "closed form 2*Phi(z/sqrt(1+h/(T-1)))-1 to within a point",
              arima),
        Probe("forecast_intervals", "tsecon.var_forecast", "lower / upper",
              "PRED", 0.95,
              "T=800, h=1",
              lambda R: fc_at(R, "exp3_var_T800_lags1", "library", 1),
              "T=100, worst horizon",
              lambda R: fc_worst(R, "exp3_var_T100_lags1", "library"),
              "APPROXIMATION",
              "same plug-in story: at T=800 the gap to the oracle band is "
              "+0.2pp, at T=100 it is +2.4pp. Estimation error, not bias",
              var),
        Probe("forecast_intervals", "tsecon.var_forecast",
              "marginal bands read as a JOINT band", "PRED", 0.95,
              "one horizon, one series",
              lambda R: fc_at(R, "exp3_var_T100_lags1", "library", 1),
              "12 horizons x 2 series at once",
              lambda R: (float(R["exp3_var_T100_lags1"]["joint_all_horizons"]
                               ["library"]),
                         mc_se(float(R["exp3_var_T100_lags1"]
                                     ["joint_all_horizons"]["library"]),
                               R["exp3_var_T100_lags1"]["reps"]),
                         "every horizon and series inside simultaneously"),
              "READING",
              "the bands are marginal by construction; a fan chart is not a "
              "joint statement about the path", var),
        Probe("forecast_intervals", "tsecon.theta_forecast / tsecon.backtest",
              "no interval is returned", "NONE", 0.95,
              "", None, "", None, "READING",
              "both return point paths only. Any band you report around them "
              "is your own construction, and its coverage is your claim, not "
              "the library's", fore),
    ]


def probes_bayes() -> list[Probe]:
    def bvar(design, h, cell):
        def f(out):
            e = out["results"]["exp1"]
            labels = list(e["labels"])
            if design not in labels:
                raise ProbeError(f"{design!r} not in {labels}")
            arr = e["cov"][labels.index(design)]
            if h >= len(arr):
                raise ProbeError(f"h={h} beyond {len(arr) - 1}")
            i, j = cell
            c = float(arr[h][i][j])
            return c, mc_se(c, e["reps"]), f"prior '{design}', h={h}"
        return f

    def bvar_last(design, cell):
        def f(out):
            e = out["results"]["exp1"]
            arr = e["cov"][list(e["labels"]).index(design)]
            i, j = cell
            c = float(arr[-1][i][j])
            return c, mc_se(c, e["reps"]), f"prior '{design}', h={len(arr) - 1}"
        return f

    def impact(which):
        def f(out):
            e = out["results"]["exp1b"]
            if which == "exact":
                c = float(e["cov_exact"])
                lab = "exact chi-square interval for the same scalar"
            else:
                c = float(e["cov_bvar"][which])
                lab = f"credible band, lambda1={which}"
            return c, mc_se(c, e["reps"]), lab
        return f

    def setobj(exp, obj, lam, h, cell):
        def f(out):
            e = out["results"][exp]
            key = (obj, lam) if exp == "exp2" else lam
            arr = e["cov"][key]
            if h >= len(arr):
                raise ProbeError(f"{exp}: h={h} beyond {len(arr) - 1}")
            i, j = cell
            c = float(arr[h][i][j])
            return c, mc_se(c, e["reps"]), f"lambda1={lam}, h={h}"
        return f

    def brk(exp, T, delta, key):
        def f(out):
            cell = pick(out["results"][exp]["cells"], T=T, delta=delta)
            reps = cell["n_detected"] if key.startswith("cond") else cell["reps"]
            c = float(cell[key])
            return c, mc_se(c, reps), (f"break/sigma={delta}, T={T}"
                                       + (", cond. on detection"
                                          if key.startswith("cond")
                                          else f", detection {cell['detect_rate']:.2f}"))
        return f

    bay = "../../reference/model-cards/bayesian.md"
    ident = "../../reference/model-cards/structural-identification.md"
    brkcard = "../../reference/model-cards/structural-breaks.md"
    return [
        Probe("bayes_and_sets", "tsecon.bvar_irf_draws",
              "5th/95th posterior percentile band", "CRED", 0.90,
              "prior mean near the truth, tight",
              bvar("oracle-tight", 4, (0, 0)),
              "library-default Minnesota prior (delta=0)",
              bvar("default", 4, (0, 0)),
              "ESTIMATOR",
              "the shortfall is the PRIOR, not a defect: delta=0 means white "
              "noise and the truth has own lags 0.85, so the band is in the "
              "wrong place. Set delta, or use bvar_hierarchical", bay),
        Probe("bayes_and_sets", "tsecon.bvar_ssvs",
              "spike-and-slab credible band", "CRED", 0.90,
              "impact", bvar("SSVS spike-slab", 0, (0, 0)),
              "long horizon, true-but-small cross coefficient",
              bvar_last("SSVS spike-slab", (1, 0)),
              "ESTIMATOR",
              "the spike does what it is for and zeroes a true 0.03 cross lag; "
              "the band then sits around zero. Expected behaviour, not "
              "calibration", bay),
        Probe("bayes_and_sets", "tsecon.bvar_irf_draws",
              "impact band vs an EXACT interval", "CRED", 0.90,
              "exact chi-square reference (validates the harness)",
              impact("exact"),
              "credible band, prior mean exactly right", impact(5.0),
              "CONVENTION",
              "even a perfect prior mean leaves ~3pp: the conjugate NIW "
              "posterior df exceeds the residual df. This is why a credible "
              "band is not a confidence interval", bay),
        Probe("bayes_and_sets", "tsecon.sign_restricted_svar",
              "pointwise 5-95 band over rotations", "CRED", 0.90,
              "near-diffuse reduced-form prior",
              setobj("exp2", "pointwise band", 5.0, 3, (0, 0)),
              "library-default lambda1=0.2",
              setobj("exp2", "pointwise band", 0.2, 3, (0, 0)),
              "READING",
              "a Haar-prior posterior summary that mixes mutually "
              "inconsistent structural models -- it is neither a confidence "
              "interval nor the identified set (see fry_pagan_svar)", ident),
        Probe("bayes_and_sets", "tsecon.robust_svar_bounds",
              "Giacomini-Kitagawa robust region", "CRED", 0.90,
              "near-diffuse reduced-form prior",
              setobj("exp2", "robust CI", 5.0, 3, (0, 0)),
              "library-default lambda1=0.2",
              setobj("exp2", "robust CI", 0.2, 3, (0, 0)),
              "ESTIMATOR",
              "this is the one set-identified object that aims at 1-alpha "
              "containment, and it delivers under a diffuse prior. It is "
              "robust to the ROTATION prior only -- it inherits the Minnesota "
              "prior on the reduced form", ident),
        Probe("bayes_and_sets", "tsecon.sign_restricted_svar",
              "set envelope (min/max over draws)", "SET", 0.90,
              "near-diffuse reduced-form prior",
              setobj("exp2", "set envelope", 5.0, 3, (0, 0)),
              "library-default lambda1=0.2",
              setobj("exp2", "set envelope", 0.2, 3, (0, 0)),
              "READING",
              "the union over the posterior of the identified set. Wider than "
              "any credible object and near-100% containment certifies very "
              "little -- at impact a sign restriction leaves the set open "
              "down to zero", ident),
        Probe("bayes_and_sets", "tsecon.zero_sign_svar",
              "band at a TRUE point-identifying zero", "CRED", 0.90,
              "near-diffuse reduced-form prior",
              setobj("exp2c", None, 5.0, 3, (0, 0)),
              "library-default lambda1=0.2",
              setobj("exp2c", None, 0.2, 3, (0, 0)),
              "ESTIMATOR",
              "the zero pins the rotation, so this band IS about a point: the "
              "cleanest reading of what the Minnesota prior costs a "
              "frequentist reader", ident),
        Probe("bayes_and_sets", "tsecon.bai_perron",
              "break-date CI, conditional on detection", "CI", 0.95,
              "break/sigma=1, T=800", brk("exp3b", 800, 1.0, "cond95"),
              "break/sigma=0.5, T=200", brk("exp3", 200, 0.5, "cond95"),
              "APPROXIMATION",
              "the finite-sample quality of Bai's argmax limit distribution; "
              "it improves in T (0.877 -> 0.944 at T=200 -> 800) while the "
              "interval WIDTH does not shrink, exactly as fixed-break "
              "asymptotics predict", brkcard),
        Probe("bayes_and_sets", "tsecon.bai_perron",
              "break-date CI, UNconditional", "CI", 0.95,
              "break/sigma=3", brk("exp3", 200, 3.0, "uncond95"),
              "break/sigma=0.25", brk("exp3", 200, 0.25, "uncond95"),
              "ESTIMATOR",
              "detection itself collapses to 0.30, so the rate a user faces "
              "is 0.23. A break-date CI is only meaningful once the break is "
              "detectable", brkcard),
        Probe("bayes_and_sets", "tsecon.bai_perron",
              "break-date CI at a LARGE break", "CI", 0.95,
              "break/sigma=1", brk("exp3", 200, 1.0, "cond95"),
              "break/sigma=3", brk("exp3", 200, 3.0, "cond95"),
              "CONVENTION",
              "over-covers: the half-width is ceil(c/scale) PLUS ONE index on "
              "each side, and at a large break that discreteness padding "
              "dominates", brkcard),
    ]


def rows_prefix(res: dict, exp: str, prefix: str) -> list[dict]:
    """Every row of `exp` whose arm starts with `prefix` (>= 1 required)."""
    rows = [r for r in res[exp]["rows"] if r["arm"].startswith(prefix)]
    if not rows:
        raise ProbeError(f"{exp}: no rows with arm prefix {prefix!r}")
    return rows


def worst_prefix(res: dict, exp: str, prefix: str,
                 label: str | None = None) -> tuple[float, float, str]:
    """The worst-covering cell over every arm matching `prefix`."""
    row = min(rows_prefix(res, exp, prefix), key=lambda r: r["cov95"])
    base = label if label is not None else row["arm"]
    return (float(row["cov95"]), float(row["mcse"]),
            f"{base}, worst h ({int(row['h'])})")


def probes_quantile_panel() -> list[Probe]:
    qcard = "../../reference/model-cards/quantile.md"
    pcard = "../../reference/model-cards/panel.md"
    lcard = "../../reference/model-cards/local-projections.md"
    cook = "../../cookbook/panel-lp-standard-errors.md"

    def umworst(R):
        row = min((r for r in R["quantile_iid"]["rows"]), key=lambda r: r["cov95"])
        return (float(row["cov95"]), float(row["mcse"]),
                f"{row['arm']}, worst h ({int(row['h'])})")

    def spj_cell(T, h):
        def f(R):
            row = pick(R["panel_spj"]["rows"], arm=f"T={T} spj", h=h)
            return (float(row["cov95"]), float(row["mcse"]),
                    f"SPJ, N=50, T={T}, h={h}")
        return f

    def cum_cell(T, h):
        def f(R):
            row = pick(R["lp_cumulative"]["rows"],
                       arm=f"T={T} both (hac default)", h=h)
            return (float(row["cov95"]), float(row["mcse"]),
                    f"T={T}, h={h} (the pre-fix 0.507 cell)")
        return f

    def cum_extreme(T):
        def f(R):
            nom = float(R["lp_cumulative"]["meta"]["nominal"])
            rows = rows_prefix(R, "lp_cumulative", f"T={T} both")
            row = max(rows, key=lambda r: abs(r["cov95"] - nom))
            return (float(row["cov95"]), float(row["mcse"]),
                    f"T={T}, most extreme h ({int(row['h'])})")
        return f

    return [
        Probe("quantile_panel_lp", "tsecon.quantile_lp",
              "Powell sandwich, identified iid shock", "CI", 0.95,
              "median tau, impact, T=400",
              lambda R: lp_row(R, "quantile_iid", "iid T=400 tau=0.50", 0),
              "worst (tau, h, T) cell of the whole iid grid", umworst,
              "APPROXIMATION",
              "close to nominal at every tau, horizon and T measured -- the "
              "quantile card's transferred growth_at_risk warning does NOT "
              "bind on an identified iid shock, because an iid impulse makes "
              "the check-loss score serially uncorrelated (the same "
              "mechanism that makes lag-augmented LP work). The residual "
              "~1-2pp is the Powell kernel density estimate", qcard),
        Probe("quantile_panel_lp", "tsecon.quantile_lp",
              "persistent regressor (phi=0.8)", "CI", 0.95,
              "default lag controls (p=4) whiten the score",
              lambda R: worst_prefix(R, "quantile_persistent",
                                     "persistent p=4", "p=4"),
              "no lag controls (p=0): nothing whitens",
              lambda R: worst_prefix(R, "quantile_persistent",
                                     "persistent p=0", "p=0"),
              "ESTIMATOR",
              "the Powell sandwich is heteroskedasticity-robust, NOT HAC -- "
              "exactly as the card says -- and the default lag controls are "
              "what stand between a user and the growth_at_risk-shaped "
              "decay: strip them and se/sd falls to ~0.65 at the far "
              "horizon. Keep n_lag_controls >= the regressor's AR order",
              qcard),
        Probe("quantile_panel_lp", "tsecon.panel_lp",
              'se_type="driscoll_kraay" (the default)', "CI", 0.95,
              "T=80, impact",
              lambda R: lp_row(R, "panel_dk", "N=50 T=80 dk", 0),
              "T=40, worst horizon (N cannot help)",
              lambda R: worst_prefix(R, "panel_dk", "N=50 T=40 dk",
                                     "N=50 T=40"),
              "APPROXIMATION",
              "with a common shock the effective sample is T, not N*T: "
              "quintupling N moves pooled coverage by under a point while "
              "doubling T buys ~5pp -- the cookbook's caveat, measured. "
              "Driscoll-Kraay is a T-asymptotic estimator; at T=40 read the "
              "bands as indicative. (cluster-by-entity on the same draws is "
              "strictly worse under the common factor)", cook),
        Probe("quantile_panel_lp", "tsecon.panel_lp",
              'bias_correction="spj" at short T', "CI", 0.95,
              "T=40, h=2", spj_cell(40, 2),
              "T=20, h=2 (the card's headline cell)", spj_cell(20, 2),
              "APPROXIMATION",
              "the split-panel jackknife removes most of the Nickell bias "
              "and neither it nor uncorrected FE reaches nominal at T=20, "
              "because Driscoll-Kraay itself is a short-T approximation -- "
              "the card's own caveat, re-measured at ~8x its replication "
              "count (and at these seeds the T=20 coverage GAIN over FE is "
              "smaller than the card's 300-rep point numbers suggest; the "
              "bias reduction is unambiguous)", pcard),
        Probe("quantile_panel_lp", "tsecon.lp",
              'cumulative="both" (default se resolves to "hac")', "CI", 0.95,
              "T=400, h=12 (the repaired defect cell)", cum_cell(400, 12),
              "T=1600, where the h+p bandwidth overshoots",
              cum_extreme(1600),
              "APPROXIMATION",
              "the official post-fix numbers for the audit's most serious "
              "finding: the pre-fix HC1 default covered 0.507 at h=12 and "
              "was flat in T; the 0.3.0 mode-dependent HAC default restores "
              "h=12 to ~0.93-0.94 at T=400, and at T=1600 the residual "
              "deviation flips mildly CONSERVATIVE (the Bartlett bandwidth "
              "h + p is generous once T is large relative to the overlap). "
              "se=\"lag_augmented\" with this mode now raises (asserted "
              "every run)", lcard),
    ]


def probes_factor_midas() -> list[Probe]:
    guide = "../../guide/07-multivariate.md"
    mcard = "../../reference/model-cards/nowcasting-midas.md"
    tcard = "../../reference/model-cards/term-structure.md"

    def um(arm, k, design):
        def f(R):
            row = pick(R["umidas"]["rows"], arm=arm, h=k)
            return float(row["cov95"]), float(row["mcse"]), design
        return f

    return [
        Probe("factor_midas", "tsecon.favar + tsecon.var_irf_bands",
              "two-step bands conditioned on F-hat", "CI", 0.90,
              "rich clean panel (N=100, T=200), impact",
              lambda R: lp_row(R, "favar", "N=100 T=200 F-hat", 0),
              "small noisy panel (N=20), T=800",
              lambda R: worst_prefix(R, "favar", "N=20 T=800 F-hat",
                                     "N=20 T=800"),
              "ESTIMATOR",
              "favar ships no band; this is the guide's own construction "
              "and its warned hazard, priced. On a rich panel the F-hat "
              "bands track the infeasible true-factor bands; on a small "
              "noisy one they lose a further ~15-19pp at long horizons -- "
              "and growing T makes it WORSE while the oracle improves, the "
              "generated-regressor signature. Bootstrap the two-step "
              "procedure, or grow N before T", guide),
        Probe("factor_midas", "tsecon.umidas",
              'se_type="hac" (the default)', "CI", 0.95,
              "iid errors, T=300, most recent HF lag",
              um("fav T=300 ph=0.5 pu=0.0", 1,
                 "iid errors, T=300, k=1 (most recent lag)"),
              "AR(1) errors phi=0.7, T=150, the intercept",
              um("stress T=150 ph=0.9 pu=0.7", 0,
                 "AR(1) errors phi=0.7, T=150, intercept"),
              "APPROXIMATION",
              "the HF-lag coefficients hold ~0.92+ even under persistent "
              "errors and near-collinear lag columns; the INTERCEPT "
              "inherits the error's full serial correlation and drops to "
              "~0.82 -- the same constant-under-persistence mechanism this "
              "page documents for har_rv. Quote the constant with care or "
              "lengthen maxlags", mcard),
        Probe("factor_midas", "tsecon.weighted_midas",
              "no interval is returned", "NONE", 0.95,
              "NLS point fit, weights and fit diagnostics only", None,
              "", None, "READING",
              "no se/bse/band key exists (the key set is verified every "
              "run). If you need an interval on a mixed-frequency slope, "
              "umidas ships HAC bse -- that is the supported route", mcard),
        Probe("factor_midas", "tsecon.dfm_nowcast",
              "no interval is returned", "NONE", 0.95,
              "point nowcast + smoothed factor path only", None,
              "", None, "READING",
              "the Kalman smoother's variance is computed internally but "
              "not exposed; any band around a nowcast is your own "
              "construction", mcard),
        Probe("factor_midas", "tsecon.nelson_siegel",
              "no interval is returned", "NONE", 0.95,
              "factors, fitted lambda, residuals and R^2 only", None,
              "", None, "READING",
              "no factor covariance is exposed; the same holds for the "
              "other term-structure fits (svensson, dynamic_ns)", tcard),
    ]


PROBE_BUILDERS: dict[str, Callable[[], list[Probe]]] = {
    "regression_se": probes_regression_se,
    "irf_bands": probes_irf_bands,
    "lp_family": probes_lp_family,
    "forecast_intervals": probes_forecast,
    "bayes_and_sets": probes_bayes,
    "quantile_panel_lp": probes_quantile_panel,
    "factor_midas": probes_factor_midas,
}


# --------------------------------------------------------------------------
# verdicts
# --------------------------------------------------------------------------

def verdict(cover: float, mcse: float, nominal: float, kind: str) -> str:
    if kind in ("CRED", "SET"):
        return "n/a"
    if kind == "NONE":
        return "no band"
    if mcse <= 0:
        return "exact"
    dev = (cover - nominal) / mcse
    if dev < -3.0:
        return "UNDER"
    if dev > 3.0:
        return "OVER"
    return "at nominal"


def dev_str(cover: float, mcse: float, nominal: float) -> str:
    if mcse <= 0:
        return "  --"
    return f"{(cover - nominal) / mcse:+5.1f}"


# --------------------------------------------------------------------------
# the report
# --------------------------------------------------------------------------

def rule(width: int = 158, ch: str = "-") -> str:
    return ch * width


def header(text: str, width: int = 158) -> None:
    print()
    print(rule(width, "="))
    print(text)
    print(rule(width, "="))


def harvest(probe: Probe, results: dict) -> dict:
    row: dict[str, Any] = {"probe": probe}
    for slot, getter in (("fav", probe.fav), ("stress", probe.stress)):
        if getter is None:
            row[slot] = None
            continue
        try:
            cover, mcse, design = getter(results)
        except ProbeError:
            raise
        except Exception as exc:  # a schema change, surfaced not swallowed
            raise ProbeError(f"{probe.surface} [{probe.option}] {slot}: "
                             f"{type(exc).__name__}: {exc}") from exc
        row[slot] = {"cover": cover, "mcse": mcse, "design": design}
    return row


def print_table(rows: list[dict], slot: str, title: str, note: str,
                sort_by_gap: bool) -> None:
    header(title)
    print(note)
    print()
    live = [r for r in rows if r[slot] is not None]
    if sort_by_gap:
        live = sorted(live, key=lambda r: (r[slot]["cover"]
                                           - r["probe"].nominal))
    fmt = ("  {surf:<30} {opt:<44} {kind:<5} {nom:>4} {design:<46} "
           "{cov:>14} {dev:>6} {gap:>7} {vd:<11}")
    print(fmt.format(surf="surface", opt="interval / option", kind="kind",
                     nom="nom", design="design measured",
                     cov="coverage +- se", dev="dev", gap="gap pp",
                     vd="verdict"))
    print("  " + rule(156))
    for r in live:
        p, c = r["probe"], r[slot]
        print(fmt.format(
            surf=p.surface, opt=p.option, kind=p.kind,
            nom=f"{p.nominal:.2f}", design=c["design"][:46],
            cov=f"{c['cover']:.3f} +-{c['mcse']:.3f}",
            dev=dev_str(c["cover"], c["mcse"], p.nominal),
            gap=f"{100 * (c['cover'] - p.nominal):+.1f}",
            vd=verdict(c["cover"], c["mcse"], p.nominal, p.kind),
        ))
    dead = [r for r in rows if r[slot] is None]
    for r in dead:
        p = r["probe"]
        reason = p.fav_label or "the library returns a point path only"
        print(f"  {p.surface:<30} {p.option:<44} {p.kind:<5}    -- "
              f"{reason[:46]:<46} "
              f"{'--':>14} {'--':>6} {'--':>7} {'no band':<11}")


def print_honest_list(rows: list[dict]) -> None:
    header("EVERY INTERVAL THAT DOES NOT HIT ITS NOMINAL RATE, AND WHAT TO DO")
    print("Three groups, because they are three different findings. Group A is")
    print("the one that matters most: an interval that misses even where it is")
    print("entitled to do well. Group B covers when its assumptions hold and")
    print("loses coverage under stress -- which is what asymptotics DO, and the")
    print("deliverable there is the size of the loss, not an alarm. Group C")
    print("makes no frequentist promise at all, so nothing in it is a defect.")
    print()
    print("By construction every stress design was CHOSEN to be stressful, so a")
    print("miss in group B is expected; do not read the group B count as a")
    print("failure rate. The group A count is the one to read that way.")

    group_a, group_b, group_c = [], [], []
    for r in rows:
        p = r["probe"]
        if r["stress"] is None:
            continue
        vd_s = verdict(r["stress"]["cover"], r["stress"]["mcse"],
                       p.nominal, p.kind)
        if p.kind in ("CRED", "SET"):
            group_c.append((r, vd_s))
            continue
        vd_f = (verdict(r["fav"]["cover"], r["fav"]["mcse"], p.nominal, p.kind)
                if r["fav"] is not None else "at nominal")
        if vd_f in ("UNDER", "OVER"):
            group_a.append((r, vd_f))
        elif vd_s in ("UNDER", "OVER"):
            group_b.append((r, vd_s))

    n_freq = len([r for r in rows if r["probe"].kind in ("CI", "PRED")
                  and r["stress"] is not None])
    group_a.sort(key=lambda t: t[0]["fav"]["cover"] - t[0]["probe"].nominal)
    group_b.sort(key=lambda t: t[0]["stress"]["cover"] - t[0]["probe"].nominal)
    group_c.sort(key=lambda t: t[0]["stress"]["cover"] - t[0]["probe"].nominal)

    def emit(items: list[tuple[dict, str]], banner: str, slot: str) -> None:
        print()
        print(rule())
        print(banner)
        print(rule())
        if not items:
            print("\n  (none)")
        for r, vd in items:
            p, c = r["probe"], r[slot]
            tag = vd if p.kind not in ("CRED", "SET") else f"{p.kind} diagnostic"
            other = "stress" if slot == "fav" else None
            print()
            print(f"  {p.surface}  [{p.option}]")
            print(f"      {tag}: {c['cover']:.3f} +- {c['mcse']:.3f} against a "
                  f"{p.nominal:.2f} promise   ({100 * (c['cover'] - p.nominal):+.1f}pp)")
            print(f"      design    : {c['design']}")
            if other and r[other] is not None:
                o = r[other]
                print(f"      and under stress ({o['design']}): "
                      f"{o['cover']:.3f} +- {o['mcse']:.3f}")
            print(f"      cause     : {p.cause}")
            print(f"      do this   : {p.action}")

    emit(group_a,
         f"A. MISSES EVEN IN THE FAVOURABLE DESIGN -- {len(group_a)} of {n_freq} "
         f"frequentist intervals measured",
         "fav")
    emit(group_b,
         f"B. AT NOMINAL WHEN ENTITLED, OFF UNDER STRESS -- {len(group_b)} of "
         f"{n_freq}. The number to quote is the SIZE of the loss",
         "stress")
    emit(group_c,
         f"C. NO FREQUENTIST PROMISE ({len(group_c)}) -- diagnostics, not "
         f"defects. A shortfall here measures the PRIOR or the identified SET",
         "stress")


# --------------------------------------------------------------------------
# markdown emission -- the page's Table 1 / Table 2 rows, regenerated from
# the harvested structured results so nothing on the page is hand-typed
# --------------------------------------------------------------------------

def _md_surface(surface: str) -> str:
    return " ".join(f"`{tok}`" if tok.startswith("tsecon.") else tok
                    for tok in surface.split(" "))


def _md_escape(text: str) -> str:
    return text.replace("|", "\\|")


def _md_row(probe: Probe, cell: dict | None) -> str:
    if cell is None:
        reason = probe.fav_label or "the library returns a point path only"
        return (f"| {_md_surface(probe.surface)} | "
                f"{_md_escape(probe.option)} | {probe.kind} | — | "
                f"{_md_escape(reason)} | — | — | — | no band |")
    vd = verdict(cell["cover"], cell["mcse"], probe.nominal, probe.kind)
    vd_md = f"**{vd}**" if vd in ("UNDER", "OVER") else vd
    dev = ((cell["cover"] - probe.nominal) / cell["mcse"]
           if cell["mcse"] > 0 else float("nan"))
    return (f"| {_md_surface(probe.surface)} | {_md_escape(probe.option)} | "
            f"{probe.kind} | {probe.nominal:.2f} | "
            f"{_md_escape(cell['design'])} | "
            f"{cell['cover']:.3f} ± {cell['mcse']:.3f} | {dev:+.1f} | "
            f"{100 * (cell['cover'] - probe.nominal):+.1f} | {vd_md} |")


def markdown_tables(rows: list[dict]) -> str:
    """Both page tables as markdown, from the harvested rows."""
    head = ("| surface | interval / option | kind | nom | design measured | "
            "coverage ± MC se | dev | gap pp | verdict |\n"
            "|---|---|---|---|---|---|---|---|---|")
    out = ["### Table 1 rows (registry order)", head]
    live = [r for r in rows if r["fav"] is not None]
    dead = [r for r in rows if r["fav"] is None]
    for r in live:
        out.append(_md_row(r["probe"], r["fav"]))
    for r in dead:
        out.append(_md_row(r["probe"], None))
    out += ["", "### Table 2 rows (sorted worst first)", head]
    live2 = sorted((r for r in rows if r["stress"] is not None),
                   key=lambda r: r["stress"]["cover"] - r["probe"].nominal)
    for r in live2:
        out.append(_md_row(r["probe"], r["stress"]))
    for r in rows:
        if r["stress"] is None:
            out.append(_md_row(r["probe"], None))
    # the registry counts the page's headline and group headers must match
    n = len(rows)
    n_freq = len([r for r in rows if r["probe"].kind in ("CI", "PRED")
                  and r["stress"] is not None])
    n_diag = len([r for r in rows if r["probe"].kind in ("CRED", "SET")])
    n_fn = len({fn for r in rows
                for fn in r["probe"].surface.replace(" + ", " / ")
                .split(" / ")})
    grp_a = grp_b = 0
    for r in rows:
        p = r["probe"]
        if p.kind not in ("CI", "PRED") or r["stress"] is None:
            continue
        vd_f = (verdict(r["fav"]["cover"], r["fav"]["mcse"], p.nominal,
                        p.kind) if r["fav"] is not None else "at nominal")
        vd_s = verdict(r["stress"]["cover"], r["stress"]["mcse"], p.nominal,
                       p.kind)
        if vd_f in ("UNDER", "OVER"):
            grp_a += 1
        elif vd_s in ("UNDER", "OVER"):
            grp_b += 1
    out += ["", f"<!-- counts: {n} probes across {n_fn} functions; "
                f"{n_freq} frequentist (CI+PRED); group A {grp_a}; "
                f"group B {grp_b}; CRED/SET {n_diag}; "
                f"no-band {n - n_freq - n_diag} -->"]
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def run(quick: bool = False, only: list[str] | None = None,
        summary: bool = False, markdown: str | None = None) -> int:
    t0 = time.perf_counter()
    keys = [k for k, _, _ in FAMILIES if only is None or k in only]
    if not keys:
        raise SystemExit(f"--only matched nothing; pick from "
                         f"{[k for k, _, _ in FAMILIES]}")

    header("tsecon INTERVAL COVERAGE -- CONSOLIDATED RUN")
    print(f"families      : {', '.join(keys)}")
    print(f"mode          : {'QUICK smoke run' if quick else 'full run'}")
    if quick:
        print("                QUICK multiplies every Monte Carlo standard error")
        print("                by 2-4x. Read the full run for any number you quote.")
    print(f"python        : {sys.version.split()[0]}")
    print("each family seeds every draw from one master seed, printed in its")
    print("own header below; this runner adds no randomness of its own.")

    status: list[dict] = []
    rows: list[dict] = []
    probe_errors: list[str] = []

    for key in keys:
        title, what = next((t, w) for k, t, w in FAMILIES if k == key)
        mod = importlib.import_module(key)
        buf = io.StringIO()
        t1 = time.perf_counter()
        failed = None
        results = None
        try:
            with contextlib.redirect_stdout(buf):
                results = mod.run(quick=quick)
        except Exception as exc:
            failed = f"{type(exc).__name__}: {exc}"
        elapsed = time.perf_counter() - t1
        text = buf.getvalue()
        if not summary:
            header(f"{key}.py -- {title}")
            print(f"({what})")
            print(text, end="" if text.endswith("\n") else "\n")
            if failed:
                print(f"!! {key} RAISED: {failed}")
        n_pass, n_fail = text.count("[PASS]"), text.count("[FAIL]")
        if n_pass == 0 and results is not None and isinstance(results, dict):
            n_pass = len(results.get("claims", []) or [])
        status.append({"key": key, "title": title, "elapsed": elapsed,
                       "pass": n_pass, "fail": n_fail, "error": failed})
        if results is None:
            probe_errors.append(f"{key}: family failed, {len(PROBE_BUILDERS[key]())} "
                                f"probes not measured")
            continue
        for probe in PROBE_BUILDERS[key]():
            try:
                rows.append(harvest(probe, results))
            except ProbeError as exc:
                probe_errors.append(f"{key}: {exc}")

    print_table(rows, "fav",
                "TABLE 1 -- THE FAVOURABLE CASE: does the interval cover when "
                "it is entitled to?",
                "A miss HERE is not the user's data. `dev` is (coverage - "
                "nominal) / MC se of the measurement.",
                sort_by_gap=False)
    print_table(rows, "stress",
                "TABLE 2 -- THE STRESS CASE: where applied work actually goes",
                "Sorted worst first. Read `gap pp` next to `verdict`: both "
                "0.941 and 0.588 are 'UNDER' at reps=3000, and they are not "
                "the same finding.",
                sort_by_gap=True)
    print_honest_list(rows)

    if markdown is not None:
        with open(markdown, "w", encoding="utf-8") as fh:
            fh.write(markdown_tables(rows))
        print()
        print(f"  markdown table rows written to {markdown}")

    header("FAMILY STATUS")
    print(f"  {'module':<22} {'assertions':>12} {'runtime':>9}   result")
    print("  " + rule(70))
    for s in status:
        got = "OK" if not s["error"] else f"FAILED -- {s['error']}"
        print(f"  {s['key'] + '.py':<22} {str(s['pass']) + ' pass':>12} "
              f"{s['elapsed']:>8.1f}s   {got}")
    total = time.perf_counter() - t0
    print()
    print(f"  {len(rows)} probes harvested from {len(keys)} families in "
          f"{total:.1f}s")
    print(f"  reproduce: .venv/bin/python docs/examples/coverage/run_all.py"
          f"{' --quick' if quick else ''}")

    bad = [s for s in status if s["error"] or s["fail"]]
    if probe_errors:
        print()
        print("  PROBE ERRORS (a returned-results schema moved under a probe):")
        for e in probe_errors:
            print(f"    - {e}")
    if bad or probe_errors:
        print()
        print(f"  EXIT NON-ZERO: {len(bad)} family failure(s), "
              f"{len(probe_errors)} probe error(s)")
        return 1
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true",
                    help="smoke run: every family cuts its replication count")
    ap.add_argument("--summary", action="store_true",
                    help="print only the consolidated tables, not the five "
                         "per-family reports")
    ap.add_argument("--only", default=None,
                    help="comma-separated family keys, e.g. irf_bands,lp_family")
    ap.add_argument("--markdown", default=None, metavar="PATH",
                    help="also write the page's Table 1 / Table 2 rows as "
                         "markdown, generated from the harvested results")
    args = ap.parse_args()
    only = [s.strip() for s in args.only.split(",")] if args.only else None
    sys.exit(run(quick=args.quick, only=only, summary=args.summary,
                 markdown=args.markdown))


if __name__ == "__main__":
    main()
