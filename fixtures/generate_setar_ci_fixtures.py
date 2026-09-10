"""Golden fixtures for the Hansen (1997/2000) likelihood-ratio threshold
confidence set (`setar_threshold_ci`).

Reference: NO third-party implementation of the Hansen threshold confidence
set runs in this venv (no R `tsDyn`, no Hansen GAUSS/MATLAB `thresh`
programs — his site is unreachable from the build container), so the
reference is built HERE, in NumPy, as an independent transcription of the
published construction. Two kinds of values are pinned:

DOCUMENTED-FORMULA golden (closed forms, pinned at full double precision):

  * critical value  c(level) = -2 ln(1 - sqrt(level)), the `level`
                    quantile of P(xi <= x) = (1 - exp(-x/2))^2 — Hansen
                    (2000, Econometrica 68(3)) Table 1, which this
                    generator ASSERTS it reproduces to the printed two
                    decimals: 0.80 -> 4.50, 0.85 -> 5.10, 0.90 -> 5.94,
                    0.925 -> 6.53, 0.95 -> 7.35, 0.975 -> 8.75,
                    0.99 -> 10.59.
  * p-value         p(x) = 1 - (1 - exp(-x/2))^2, evaluated in the
                    cancellation-free form exp(-x/2) (2 - exp(-x/2)); the
                    naive form is asserted to agree to 1e-12 absolute.

CROSS-IMPLEMENTATION golden (this file's NumPy against the Rust, at 1e-10):

  * the SETAR fit   the transcription of `generate_setar_fixtures.py`
                    (imported, not re-typed): concentrated LS over the
                    trimmed unique order statistics of y_{t-d}; the
                    `thresholds` / `ssr_path` / `threshold` stored here are
                    therefore ALSO what `setar` itself must return — the
                    Rust golden asserts bit-identity with `setar`.
  * LR profile      LR_n(gamma) = n (S_n(gamma) - S_min) / S_min with
                    S_min = min of the profile (Hansen 2000, eq. for LR_n;
                    Hansen 1997 for the TAR).
  * eta^2           Hansen (2000 §3.4) as in his own programs: at gamma^,
                    regime OLS fits give delta^ = b1 - b2 and residuals e^;
                    r1 = (x'delta^)^2 and r2 = e^2 (x'delta^)^2 are each
                    regressed by least squares on the quadratic
                    [1, q, q^2] in the threshold variable q = y_{t-d}; with
                    g1, g2 the fitted values at q = gamma^,
                    eta^2 = (g2 / g1) / sigma^2, sigma^2 = S_min / n.
                    (The Rust fits the same quadratic in the centered basis
                    [1, q - gamma^, (q - gamma^)^2] — the same column space,
                    so identical fitted values; the 1e-10 pin covers the
                    algebra.) eta^2 = 1 when het_robust is off.
  * the set         {gamma_i : LR_i <= eta^2 c(level)} over the grid,
                    reported as maximal runs of consecutive in-set
                    candidates [gamma_start, gamma_end] (grid endpoints,
                    as Hansen's programs report), its hull, connectedness,
                    and the count.
  * null threshold  LR_n is a step function, constant on
                    [gamma_i, gamma_{i+1}), so gamma_0 is evaluated at the
                    largest candidate <= gamma_0: lr_at_null, the used
                    candidate, and p(lr_at_null / eta^2).
  * slope union     Hansen (2000 §3.3): over every candidate in the
                    `slope_region_level` set (his applied convention: 80%),
                    the conventional per-regime intervals
                    b_j(gamma) +/- z_{1-(1-slope_level)/2} se_j(gamma) with
                    classical per-regime SEs sqrt(SSR_j/(n_j - k)
                    diag[(X_j'X_j)^{-1}]) or, under het_robust, White HC0
                    SEs; the union is [min lower, max upper] per
                    coefficient.

Grade (honest): documented-formula golden for the critical values and the
p-value function; CROSS-IMPLEMENTATION transcription golden for everything
else — NOT a third-party golden. What the pin proves: the Rust reproduces
an independent dense-NumPy reading of Hansen's published construction and
of his programs' eta^2 convention at 1e-10. Whether the set COVERS the true
threshold at its nominal rate is a statistical claim the pin cannot make;
it is MEASURED by the crate's seeded Monte Carlo property tests
(`setar_ci_properties.rs`), whose numbers the model card quotes.

This generator NEVER imports tsecon. Doubles are written with json's
shortest round-trip repr, which the Rust golden test parses to identical
bits (serde_json `float_roundtrip`).

Run:  .venv/bin/python fixtures/generate_setar_ci_fixtures.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_setar_fixtures import (  # noqa: E402  (sibling generator)
    design,
    fit_setar,
    make_series,
    ols,
)

OUT = Path(__file__).resolve().parent / "setar_ci.json"

# Hansen (2000) Table 1, as printed.
HANSEN_TABLE_1 = {
    "0.8": 4.50,
    "0.85": 5.10,
    "0.9": 5.94,
    "0.925": 6.53,
    "0.95": 7.35,
    "0.975": 8.75,
    "0.99": 10.59,
}


# ------------------------------------------------------------ closed forms

def crit(level):
    return -2.0 * math.log(1.0 - math.sqrt(level))


def pvalue(x):
    e = math.exp(-x / 2.0)
    stable = e * (2.0 - e)
    naive = 1.0 - (1.0 - e) ** 2
    assert abs(stable - naive) <= 1e-12, (x, stable, naive)
    return stable


# ------------------------------------------------------------------ series

def sim_setar_het(seed, T, gamma, low, high, burn=100):
    """SETAR(1), d = 1, with conditional heteroskedasticity keyed to the
    threshold variable: e_t = (0.6 + 0.4 |y_{t-1}|) eps_t. This makes
    Hansen's eta^2 differ from one, so the correction has something to do."""
    rng = np.random.default_rng(seed)
    y = np.zeros(T + burn + 1)
    eps = rng.standard_normal(T + burn + 1)
    for t in range(1, T + burn + 1):
        c = low if y[t - 1] <= gamma else high
        y[t] = c[0] + c[1] * y[t - 1] + (0.6 + 0.4 * abs(y[t - 1])) * eps[t]
    return y[burn + 1:]


# ------------------------------------- the transcription (see module doc)

def hc0_se(X, e):
    xxi = np.linalg.inv(X.T @ X)
    meat = (X * (e ** 2)[:, None]).T @ X
    return np.sqrt(np.diag(xxi @ meat @ xxi))


def classical_se(X, e, k):
    n_j = X.shape[0]
    return np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * float(e @ e) / (n_j - k))


def threshold_ci(y, p, delays, trim, constant, level, het_robust,
                 slope_level=None, slope_region_level=0.80,
                 null_threshold=None):
    fit = fit_setar(y, p, delays, trim, constant)
    d = fit["delay"]
    start = max(p, max(delays))
    k = p + int(constant)
    X, yy, z = design(y, p, d, start, constant)
    n = yy.size
    gammas = np.array(fit["thresholds"])
    path = np.array(fit["ssr_path"])
    best = int(np.argmin(path))
    gamma_hat = float(gammas[best])
    assert gamma_hat == fit["threshold"]
    s_min = float(path[best])
    lr = n * (path - s_min) / s_min

    c = crit(level)
    if het_robust:
        lo = z <= gamma_hat
        b1, _ = ols(X[lo], yy[lo])
        b2, _ = ols(X[~lo], yy[~lo])
        e = np.empty(n)
        e[lo] = yy[lo] - X[lo] @ b1
        e[~lo] = yy[~lo] - X[~lo] @ b2
        xd = X @ (b1 - b2)
        r1 = xd ** 2
        r2 = (e ** 2) * (xd ** 2)
        Q = np.column_stack([np.ones(n), z, z ** 2])
        m1 = np.linalg.lstsq(Q, r1, rcond=None)[0]
        m2 = np.linalg.lstsq(Q, r2, rcond=None)[0]
        qh = np.array([1.0, gamma_hat, gamma_hat ** 2])
        g1 = float(qh @ m1)
        g2 = float(qh @ m2)
        eta2 = (g2 / g1) / (s_min / n)
        assert eta2 > 0.0 and np.isfinite(eta2), eta2
    else:
        eta2 = 1.0
    c_scaled = eta2 * c

    in_set = lr <= c_scaled
    intervals = []
    i = 0
    G = gammas.size
    while i < G:
        if in_set[i]:
            s = i
            while i + 1 < G and in_set[i + 1]:
                i += 1
            intervals.append([float(gammas[s]), float(gammas[i])])
        i += 1
    assert in_set[best]
    out = {
        "threshold": gamma_hat,
        "delay": int(d),
        "nobs": int(n),
        "k": int(k),
        "level": level,
        "lr_crit": c,
        "lr_crit_scaled": c_scaled,
        "eta2": eta2,
        "het_robust": bool(het_robust),
        "thresholds": gammas.tolist(),
        "ssr_path": path.tolist(),
        "lr_stat": lr.tolist(),
        "in_set": [bool(b) for b in in_set],
        "intervals": intervals,
        "is_connected": len(intervals) == 1,
        "ci_low": intervals[0][0],
        "ci_high": intervals[-1][1],
        "n_in_set": int(in_set.sum()),
    }

    if null_threshold is not None:
        assert gammas[0] <= null_threshold <= gammas[-1]
        idx = int(np.searchsorted(gammas, null_threshold, side="right")) - 1
        lr0 = float(lr[idx])
        out.update(
            null_threshold=null_threshold,
            null_threshold_used=float(gammas[idx]),
            lr_at_null=lr0,
            pvalue_at_threshold=pvalue(lr0 / eta2),
        )
    else:
        out.update(null_threshold=None, null_threshold_used=None,
                   lr_at_null=None, pvalue_at_threshold=None)

    if slope_level is not None:
        zq = float(norm.ppf(1.0 - (1.0 - slope_level) / 2.0))
        region = lr <= eta2 * crit(slope_region_level)
        low_lower = np.full(k, np.inf)
        low_upper = np.full(k, -np.inf)
        high_lower = np.full(k, np.inf)
        high_upper = np.full(k, -np.inf)
        reg_g = gammas[region]
        for g in reg_g:
            lo = z <= g
            for Xj, yj, lower, upper in [
                (X[lo], yy[lo], low_lower, low_upper),
                (X[~lo], yy[~lo], high_lower, high_upper),
            ]:
                bj, _ = ols(Xj, yj)
                ej = yj - Xj @ bj
                se = hc0_se(Xj, ej) if het_robust else classical_se(Xj, ej, k)
                lower[:] = np.minimum(lower, bj - zq * se)
                upper[:] = np.maximum(upper, bj + zq * se)
        out["slope"] = {
            "level": slope_level,
            "region_level": slope_region_level,
            "region_low": float(reg_g.min()),
            "region_high": float(reg_g.max()),
            "n_region": int(region.sum()),
            "low_lower": low_lower.tolist(),
            "low_upper": low_upper.tolist(),
            "high_lower": high_lower.tolist(),
            "high_upper": high_upper.tolist(),
        }
    else:
        out["slope"] = None
    return out


# ------------------------------------------------------------------- main

def main():
    series = make_series()
    series["setar_het"] = sim_setar_het(20260910, 400, 0.0,
                                        low=[1.0, 0.5], high=[-1.0, 0.2])

    # Closed forms.
    levels = [0.5, 0.8, 0.85, 0.9, 0.925, 0.95, 0.975, 0.99, 0.999]
    crits = [{"level": lv, "crit": crit(lv)} for lv in levels]
    for key, printed in HANSEN_TABLE_1.items():
        got = crit(float(key))
        assert round(got, 2) == printed, (key, got, printed)
    xs = [0.0, 0.5, 1.0, 2.0, 4.5, 5.94, 7.35, 10.59, 20.0, 40.0, 80.0]
    pvals = [{"x": x, "p": pvalue(x)} for x in xs]
    # The p-value at the level-alpha critical value is 1 - level, exactly
    # the inversion the set relies on.
    for lv in levels:
        assert abs(pvalue(crit(lv)) - (1.0 - lv)) < 1e-12

    cases = []
    for (name, p, delays, trim, constant, level, het, sl, rl, g0) in [
        ("setar_strong", 1, [1], 0.15, True, 0.95, False, 0.95, 0.80, 0.1),
        ("setar_strong", 1, [1], 0.15, True, 0.90, True, None, 0.80, 0.0),
        ("setar_d2", 1, [1, 2, 3], 0.15, True, 0.95, True, 0.90, 0.90, 0.3),
        ("linear_ar1", 1, [1], 0.15, True, 0.95, False, None, 0.80, None),
        # No threshold effect here (delta^ ~ 0): with het_robust the
        # quadratic fit of e^2 (x'delta)^2 is NEGATIVE at gamma^, eta^2 is
        # not identified, and the Rust refuses with a teaching error (a
        # refusal the tests pin on this very series); the homoskedastic
        # set is what is stored.
        ("linear_ar2", 2, [1, 2], 0.15, False, 0.80, False, 0.95, 0.80, None),
        ("setar_het", 1, [1], 0.10, True, 0.95, True, 0.95, 0.80, 0.0),
        ("setar_het", 1, [1], 0.10, True, 0.95, False, 0.95, 0.80, 0.0),
    ]:
        case = threshold_ci(series[name], p, delays, trim, constant, level,
                            het, sl, rl, g0)
        case.update(series=name, p=p, delays=delays, trim=trim,
                    constant=constant, slope_level=sl,
                    slope_region_level=rl if sl is not None else None)
        cases.append(case)

    fixture = {
        "_meta": {
            "numpy": np.__version__,
            "note": (
                "Hansen (1997/2000) threshold LR confidence sets: closed-form "
                "critical values and p-value function (documented-formula "
                "golden, Table 1 reproduced to the printed decimals) plus a "
                "cross-implementation NumPy transcription of the LR profile, "
                "Hansen's eta^2 programs convention, the interval/hull "
                "construction, the null-threshold inversion and the slope "
                "unions — no third-party threshold-CI code runs in this "
                "venv. Coverage is measured by Monte Carlo property tests, "
                "not pinned here."
            ),
            "hansen_2000_table_1": HANSEN_TABLE_1,
        },
        "critical_values": crits,
        "pvalues": pvals,
        "series": {kk: np.asarray(vv).tolist() for kk, vv in series.items()},
        "cases": cases,
    }

    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(fixture, fh, indent=1)
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes); {len(cases)} cases")
    print("  setar_het std = %.3f" % np.std(series["setar_het"]))
    for c in cases:
        sl = c["slope"]
        print(f"  {c['series']:13s} p={c['p']} d={c['delay']} lvl={c['level']} "
              f"het={int(c['het_robust'])} gamma={c['threshold']: .4f} "
              f"eta2={c['eta2']:.4f} set={c['intervals']} "
              f"n_in={c['n_in_set']}/{len(c['thresholds'])} "
              f"p0={c['pvalue_at_threshold']}"
              + (f" slope_region=({sl['region_low']:.3f},{sl['region_high']:.3f}) "
                 f"n={sl['n_region']}" if sl else ""))


if __name__ == "__main__":
    main()
