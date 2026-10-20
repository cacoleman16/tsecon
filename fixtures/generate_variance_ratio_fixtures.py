"""Golden fixtures for the Lo-MacKinlay variance-ratio test and the
Chow-Denning multiple-horizon joint test (`tsecon.variance_ratio`).

Reference implementations (this venv; versions pinned in ``_meta``):

* single horizon -> ``arch.unitroot.VarianceRatio`` (arch 8.0.0):
  ``.vr``, ``.stat``, ``.critical_values`` and ``.pvalue`` for every case.
  arch takes the LEVEL series ``y``; the increments are ``diff(y)``.
* p-value -> pinned TWICE. The primary target ``pvalue_sf`` is the accurate
  two-sided normal tail ``2 * scipy.stats.norm.sf(|arch stat|)`` (what tsecon
  computes, through ``erfc``); the secondary ``pvalue_arch`` is arch's own
  ``2 - 2 * norm.cdf(|stat|)``, which loses ~1e-10 relative accuracy at
  |z| = 5 and underflows to exactly 0.0 beyond |z| = 8.3. The ``tail`` series
  (AR(0.6) increments, n = 2000) puts |z| = 21 into the grid, where
  ``pvalue_sf`` is ~4e-98 and ``pvalue_arch`` is 0.0, so the accuracy claim is
  actually exercised.
* ``stat_variance`` -> the closed forms (Campbell, Lo & MacKinlay 1997):
  ``2 (q - 1)`` for non-overlapping blocks, ``2 (2q - 1)(q - 1) / (3q)``
  overlapping and homoskedastic (CLM 2.4.39), and the robust
  ``theta = sum_{k=1}^{q-1} 4 (1 - k/q)^2 delta_k``,
  ``delta_k = nq sum z2_t z2_{t-k} / (sum z2)^2`` on the drift-centred,
  non-debiased squared increments (CLM 2.4.43). Cross-checked here against
  arch's private ``_stat_variance`` (asserted to 1e-12 relative) but never
  read from it.
* Chow-Denning (1993) -> a documented closed form composed over arch's
  per-horizon statistics: ``CD = max_i |z_i|``; the studentized maximum
  modulus bound SMM(alpha; m, inf), i.e. the Sidak critical value
  ``Phi^{-1}(1 - (1 - (1 - alpha)^(1/m)) / 2)``; and its inversion, the
  p-value ``1 - (1 - 2 Phi(-CD))^m``, evaluated as
  ``-expm1(m * log1p(-p_min))`` (``p_min`` exactly when m = 1).
* cross-implementation leg (not an independent run: R is not available in
  this container) -> CRAN vrtest 1.2 (2023-08-31), ``R/LM_stat.R`` and
  ``R/Chow.Denning.R`` as published on the github.com/cran/vrtest mirror
  (``master``). Their NumPy transcriptions below (``vrtest_lm_stat``,
  ``vrtest_cd_crit``) are run against arch(debiased=False, trend="c",
  overlap=True) — the configuration vrtest implements, on returns — and the
  worst deviations are recorded in ``_meta``.

Grid (``cases``): 5 series x q in {2, 4, 5, 8, 16} x trend in {"n", "c"} x
(overlap, debiased, robust) in {TTT, TTF, TFT, TFF, FFF} = 250 cases. The
(F, T, T) combination is NOT a tsecon call (tsecon refuses debiased/robust
under overlap=False because they have no effect there); this generator
instead asserts arch's bit-identity across those flags under overlap=False
and records that in ``_meta`` as the reason.

``chow_denning``: 5 series x horizon sets {(2, 4, 8, 16), (2, 5, 10), (4,)}
x flags {TTT, TFF, FFF}, trend "c" = 45 cases.

``corners``: arch on the one-q-sum samples (overlap, debiased=False,
n = q + 1; overlap=False, n = 2q) for BOTH trends — arch returns vr = 0.0
under trend "c" and a finite single-squared-sum ratio under trend "n";
tsecon refuses all of them (fewer than two q-period sums).

``nobs`` is tsecon's convention (increments used, n - 1 - n_dropped);
``arch_nobs`` is arch's ``.nobs`` (len(y) before truncation).

Series: seeded ``numpy.random.default_rng`` draws through known DGPs, plus
the log of statsmodels' public-domain US real GDP (macrodata) — a
transformation, not the raw data. This generator NEVER imports tsecon.

Run:  python fixtures/generate_variance_ratio_fixtures.py
"""

from __future__ import annotations

import itertools
import json
import platform
import warnings
from pathlib import Path

import arch
import numpy as np
import scipy
import statsmodels
import statsmodels.api as sm
from arch.unitroot import VarianceRatio
from scipy.stats import norm

OUT = Path(__file__).resolve().parent / "variance_ratio.json"

VRTEST_SOURCE = (
    "CRAN vrtest 1.2 (2023-08-31), R/LM_stat.R and R/Chow.Denning.R, "
    "https://github.com/cran/vrtest (master); source transcribed, not executed"
)


# ------------------------------------------------------------------ series

def rw(seed: int, n: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.concatenate([[0.0], np.cumsum(rng.standard_normal(n - 1))])


def garch_rw(seed: int, n: int, omega=0.05, a=0.10, b=0.85) -> np.ndarray:
    """Random walk with drift 0.02 and GARCH(1, 1) increments."""
    rng = np.random.default_rng(seed)
    e = np.empty(n - 1)
    s2 = omega / (1 - a - b)
    for t in range(n - 1):
        e[t] = np.sqrt(s2) * rng.standard_normal()
        s2 = omega + a * e[t] ** 2 + b * s2
    return np.concatenate([[0.0], np.cumsum(0.02 + e)])


def ar_rw(seed: int, n: int, phi: float = 0.3) -> np.ndarray:
    """Integrated AR(1): increments u_t = phi u_{t-1} + e_t (momentum)."""
    rng = np.random.default_rng(seed)
    u = np.empty(n - 1)
    prev = 0.0
    for t in range(n - 1):
        prev = phi * prev + rng.standard_normal()
        u[t] = prev
    return np.concatenate([[0.0], np.cumsum(u)])


SERIES = {
    "rw500": rw(0, 500),
    "garch1000": garch_rw(1, 1000),
    "ar03_400": ar_rw(2, 400),
    "log_realgdp": np.log(
        sm.datasets.macrodata.load_pandas().data["realgdp"].to_numpy(dtype=float)
    ),
    # |z| = 21 at q = 2: the deep tail where arch's 2 - 2 Phi underflows.
    "tail": ar_rw(3, 2000, phi=0.6),
}

LAGS = (2, 4, 5, 8, 16)
TRENDS = ("n", "c")
# (overlap, debiased, robust)
FLAGS = [
    (True, True, True),
    (True, True, False),
    (True, False, True),
    (True, False, False),
    (False, False, False),
]
CD_LAGS = ((2, 4, 8, 16), (2, 5, 10), (4,))
CD_FLAGS = [(True, True, True), (True, False, False), (False, False, False)]
CD_ALPHAS = {"1%": 0.01, "5%": 0.05, "10%": 0.10}


# --------------------------------------------------------------- closed forms

def stat_variance_closed_form(y: np.ndarray, q: int, trend: str, overlap: bool,
                              robust: bool) -> float:
    """CLM 2.4.39 / 2.4.43 and the block variance; see the module docstring."""
    if not overlap:
        return 2.0 * (q - 1)
    if not robust:
        return (2 * (2 * q - 1) * (q - 1)) / (3 * q)
    nq = y.size - 1
    mu = 0.0 if trend == "n" else (y[-1] - y[0]) / nq
    z2 = (np.diff(y) - mu) ** 2
    scale = z2.sum() ** 2
    return float(sum(4 * (1 - k / q) ** 2 * nq * (z2[k:] @ z2[:-k]) / scale
                     for k in range(1, q)))


def arch_vr(y: np.ndarray, q: int, trend: str, overlap: bool, debiased: bool,
            robust: bool) -> VarianceRatio:
    with warnings.catch_warnings():
        # arch warns (InvalidLengthWarning) when overlap=False truncates.
        warnings.simplefilter("ignore")
        r = VarianceRatio(y, lags=q, trend=trend, debiased=debiased,
                          robust=robust, overlap=overlap)
        r.stat  # noqa: B018 -- force the computation inside the filter
    return r


def vr_case(name: str, q: int, trend: str, ov: bool, deb: bool, rob: bool) -> dict:
    y = SERIES[name]
    r = arch_vr(y, q, trend, ov, deb, rob)
    dropped = 0 if ov else (y.size - 1) % q
    y_used = y if dropped == 0 else y[:-dropped]
    sv = stat_variance_closed_form(y_used, q, trend, ov, rob)
    assert abs(sv - r._stat_variance) <= 1e-12 * abs(r._stat_variance), (name, q, trend)
    stat = float(r.stat)
    return {
        "series": name, "lags": q, "trend": trend,
        "overlap": ov, "debiased": deb, "robust": rob,
        "vr": float(r.vr), "stat": stat,
        "pvalue_sf": float(2.0 * norm.sf(abs(stat))),
        "pvalue_arch": float(r.pvalue),
        "stat_variance": sv,
        "nobs": int(y.size - 1 - dropped), "n_dropped": int(dropped),
        "arch_nobs": int(r.nobs),
        "crit": {k: float(v) for k, v in r.critical_values.items()},
    }


def cd_crit(m: int) -> dict:
    """SMM(alpha; m, inf) = the Sidak critical value (exact pointwise at m=1)."""
    out = {}
    for label, a in CD_ALPHAS.items():
        per = a if m == 1 else -np.expm1(np.log1p(-a) / m)
        out[label] = float(norm.ppf(1.0 - per / 2.0))
    return out


def cd_case(name: str, lags: tuple, ov: bool, deb: bool, rob: bool) -> dict:
    per = [vr_case(name, q, "c", ov, deb, rob) for q in lags]
    z = np.array([c["stat"] for c in per])
    m = len(lags)
    i = int(np.argmax(np.abs(z)))  # first argmax in the given order
    cd = float(abs(z[i]))
    pmin = 2.0 * norm.sf(cd)
    p = pmin if m == 1 else float(-np.expm1(m * np.log1p(-pmin)))
    return {
        "series": name, "lags": list(lags), "trend": "c",
        "overlap": ov, "debiased": deb, "robust": rob,
        "vr": [c["vr"] for c in per], "stat": [c["stat"] for c in per],
        "pvalue_sf": [c["pvalue_sf"] for c in per],
        "stat_variance": [c["stat_variance"] for c in per],
        "nobs": [c["nobs"] for c in per], "n_dropped": [c["n_dropped"] for c in per],
        "cd_stat": cd, "cd_lag": int(lags[i]), "cd_pvalue": float(p),
        "cd_crit": cd_crit(m),
    }


# ------------------------------------------- vrtest 1.2 transcription (leg)

def vrtest_lm_stat(r: np.ndarray, k: int) -> tuple[float, float]:
    """NumPy transcription of vrtest 1.2 ``LM_stat(y, k)`` (y = RETURNS):

        y1 <- (y-mean(y))^2; n <- nrow(y); m <- mean(y)
        vr1 <- sum((y-m)^2)/n
        flt <- filter(y, rep(1,k), method="convolution"); flt <- flt[!is.na(flt)]
        summ <- sum((flt - k*m)^2); vr2 <- summ/(n*k); vr <- vr2/vr1
        m1 <- sqrt(n)*(vr-1)/sqrt(2*(2*k-1)*(k-1)/(3*k))
        w <- 4*(1-(1:(k-1))/k)^2
        dvec[j] <- sum(y1[(j+1):n]*y1[1:(n-j)])/(sum(y1)^2)
        m2 <- sqrt(n)*(vr-1)*((n*crossprod(w,dvec))^(-.5))
    """
    n = r.size
    m = r.mean()
    y1 = (r - m) ** 2
    vr1 = np.sum((r - m) ** 2) / n
    flt = np.convolve(r, np.ones(k), mode="valid")
    vr2 = np.sum((flt - k * m) ** 2) / (n * k)
    vr = vr2 / vr1
    m1 = np.sqrt(n) * (vr - 1) / np.sqrt(2 * (2 * k - 1) * (k - 1) / (3 * k))
    w = 4 * (1 - np.arange(1, k) / k) ** 2
    dvec = np.array([np.sum(y1[j:] * y1[: n - j]) / np.sum(y1) ** 2 for j in range(1, k)])
    m2 = np.sqrt(n) * (vr - 1) * (n * (w @ dvec)) ** -0.5
    return float(m1), float(m2)


def vrtest_cd_crit(m: int) -> list[float]:
    """vrtest 1.2 ``Chow.Denning``: per <- 0.5*(1-(1-alpha)^(1/m));
    crit <- qnorm(1-per), alpha = c(0.1, 0.05, 0.01)."""
    alpha = np.array([0.1, 0.05, 0.01])
    per = 0.5 * (1 - (1 - alpha) ** (1 / m))
    return [float(v) for v in norm.ppf(1 - per)]


def vrtest_leg() -> dict:
    worst_stat = 0.0
    for name, y in SERIES.items():
        r = np.diff(y)
        for q in LAGS:
            m1, m2 = vrtest_lm_stat(r, q)
            a1 = float(arch_vr(y, q, "c", True, False, False).stat)
            a2 = float(arch_vr(y, q, "c", True, False, True).stat)
            for mine, ref in ((m1, a1), (m2, a2)):
                worst_stat = max(worst_stat, abs(mine - ref) / (1e-12 + abs(ref)))
    worst_crit = 0.0
    for m in (1, 2, 3, 4, 5, 8):
        c = cd_crit(m)
        v = vrtest_cd_crit(m)
        for ours, theirs in zip((c["10%"], c["5%"], c["1%"]), v, strict=True):
            worst_crit = max(worst_crit, abs(ours - theirs) / abs(theirs))
    assert worst_stat < 1e-12 and worst_crit < 1e-12, (worst_stat, worst_crit)
    return {"source": VRTEST_SOURCE,
            "max_rel_dev_lm_stat_vs_arch_debiased_false": worst_stat,
            "max_rel_dev_cd_crit_vs_closed_form": worst_crit}


# ------------------------------------------------------------- inertness

def assert_nonoverlap_flags_inert() -> int:
    """arch silently ignores debiased/robust under overlap=False: assert the
    outputs are bit-identical across all four flag pairs."""
    count = 0
    for name, y in SERIES.items():
        for q, trend in itertools.product(LAGS, TRENDS):
            outs = {
                (deb, rob): (lambda r: (r.vr, r.stat, r.pvalue))(
                    arch_vr(y, q, trend, False, deb, rob))
                for deb in (True, False) for rob in (True, False)
            }
            assert len(set(outs.values())) == 1, (name, q, trend)
            count += 1
    return count


def corners() -> list[dict]:
    """arch on samples with ONE q-period sum, both trends (tsecon refuses)."""
    out = []
    for trend in ("c", "n"):
        for q in (2, 4):
            y_ov = rw(5, q + 1)
            a = arch_vr(y_ov, q, trend, True, False, False)
            out.append({"trend": trend, "lags": q, "overlap": True, "debiased": False,
                        "robust": False, "y": y_ov.tolist(),
                        "arch_vr": float(a.vr), "arch_stat": float(a.stat)})
            y_no = rw(5, 2 * q)
            b = arch_vr(y_no, q, trend, False, False, False)
            out.append({"trend": trend, "lags": q, "overlap": False, "debiased": False,
                        "robust": False, "y": y_no.tolist(),
                        "arch_vr": float(b.vr), "arch_stat": float(b.stat)})
    # The trend-"c" corners are degenerate (vr == 0) in arch; trend "n" is not.
    assert all(c["arch_vr"] == 0.0 for c in out if c["trend"] == "c")
    assert all(np.isfinite(c["arch_vr"]) and c["arch_vr"] > 0 for c in out if c["trend"] == "n")
    return out


def main() -> None:
    cases = [vr_case(n, q, t, *f) for n, q, t, f in
             itertools.product(SERIES, LAGS, TRENDS, FLAGS)]
    cd = [cd_case(n, lags, *f) for n, lags, f in
          itertools.product(SERIES, CD_LAGS, CD_FLAGS)]
    n_inert = assert_nonoverlap_flags_inert()
    leg = vrtest_leg()
    fx = {
        "series": {k: v.tolist() for k, v in SERIES.items()},
        "cases": cases,
        "chow_denning": cd,
        "corners": corners(),
        "_meta": {
            "arch": arch.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "statsmodels": statsmodels.__version__,
            "python": platform.python_version(),
            "reference": "arch.unitroot.VarianceRatio (vr, stat, critical_values, "
                         "pvalue_arch); pvalue_sf = 2*scipy.stats.norm.sf(|stat|); "
                         "stat_variance from the CLM closed forms; Chow-Denning "
                         "SMM(inf)/Sidak closed form over arch's per-horizon stats",
            "nonoverlap_flags_inert": (
                f"asserted: arch's vr/stat/pvalue are bit-identical across "
                f"debiased x robust under overlap=False on all {n_inert} "
                f"(series, q, trend) cells, so the (F,T,T) row is omitted; "
                f"tsecon refuses an explicit debiased/robust=True there"),
            "vrtest_cross_implementation": leg,
        },
    }
    OUT.write_text(json.dumps(fx, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {OUT.name}: {OUT.stat().st_size} bytes, {len(cases)} VR cases, "
          f"{len(cd)} Chow-Denning cases, {len(fx['corners'])} corners")
    gap = max(abs(c["pvalue_arch"] - c["pvalue_sf"]) / c["pvalue_sf"]
              for c in cases if c["pvalue_arch"] > 0)
    n_zero = sum(c["pvalue_arch"] == 0.0 for c in cases)
    print(f"  max relative |arch p - 2*sf| where arch p > 0: {gap:.3g}; "
          f"arch p underflows to 0.0 in {n_zero} cases")
    print(f"  vrtest leg: {leg}")
    for c in cases:
        if (c["series"], c["lags"], c["trend"], c["overlap"], c["debiased"],
                c["robust"]) in {("rw500", 4, "c", True, True, True),
                                 ("log_realgdp", 4, "c", True, True, True),
                                 ("ar03_400", 2, "c", True, True, True),
                                 ("tail", 2, "c", True, True, True)}:
            print(f"  {c['series']:12s} q={c['lags']} vr={c['vr']!r} stat={c['stat']!r} "
                  f"p_sf={c['pvalue_sf']!r} p_arch={c['pvalue_arch']!r}")
    for c in cd:
        if c["lags"] == [2, 4, 8, 16] and (c["overlap"], c["debiased"], c["robust"]) == (
                True, True, True):
            print(f"  CD {c['series']:12s} z={np.round(c['stat'], 3).tolist()} "
                  f"cd={c['cd_stat']!r} lag={c['cd_lag']} p={c['cd_pvalue']!r}")


if __name__ == "__main__":
    main()
