"""tsecon.check_series — the Module 01 flagship one-call diagnostic battery.

Pure-Python composition over the compiled primitives: descriptives and
outlier screening, the ADF+KPSS confirmatory quadrant, Ljung-Box/ACF/PACF
serial-correlation evidence, ARCH-LM, Jarque-Bera, a sup-F / Bai-Perron
mean-shift scan, GPH long memory, and seasonality evidence — plus, for a
2D ``(n, k)`` panel, per-series integration, Johansen cointegration, and
VAR lag selection with a stability check. The report ends in an ordered
list of *recommendations* that route to concrete tsecon calls.

Design stance (docs/guide/02, "The battery itself"): evidence is reported
in **families** with assumptions stated, and the multiple-testing
arithmetic is shown explicitly — the battery never silently corrects
p-values. Everything returned is JSON-serializable (plain floats, ints,
strings, lists, dicts, None) and the implementation is fully
deterministic: no randomness anywhere.

This is the friendly pure-Python front door: input is coerced with
``np.asarray(data, dtype=float64)`` (plain lists are fine here), so by the
time the compiled functions are called the strict-float64 rule at the Rust
boundary is already satisfied.
"""
from __future__ import annotations

import numpy as np

from . import _core

__all__ = ["check_series"]

# bai_perron's dynamic program is O(n^2); above this the scan is opt-in.
_MAX_BAI_PERRON_N = 5000


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _py(obj):
    """Recursively convert numpy scalars/arrays so json.dumps always works."""
    if isinstance(obj, dict):
        return {str(k): _py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_py(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _py(obj.tolist())
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


def _rec(topic, finding, evidence, suggestion, functions, caveat):
    """One recommendation entry; every key documented, evidence never empty."""
    return {
        "topic": topic,
        "finding": finding,
        "evidence": evidence,
        "suggestion": suggestion,
        "functions": functions,
        "caveat": caveat,
    }


def _initial_run(significant_lags):
    """Length of the initial consecutive run 1, 2, 3, ... of significant lags."""
    run = 0
    for lag in significant_lags:
        if lag == run + 1:
            run += 1
        else:
            break
    return run


def _validate_alpha(alpha):
    """Reject alpha values the clamped KPSS p-value cannot serve honestly."""
    a = float(alpha)
    if not 0.01 < a <= 0.10:
        raise ValueError(
            f"check_series needs alpha in (0.01, 0.10], got alpha={alpha!r}. "
            f"The compiled KPSS p-value is interpolated from its critical-value "
            f"table and clamped to [0.01, 0.10], so alpha <= 0.01 could never "
            f"let KPSS reject and alpha > 0.10 would make it always reject — "
            f"the confirmatory quadrant would be decided by the clamp, not the "
            f"data. Use alpha in {{0.05, 0.10}}, or run kpss(y) directly and "
            f"read its statistic against the reported critical values."
        )
    return a


def _detrend(y):
    """OLS-detrend: return (residuals, intercept, slope) of y on [1, t]."""
    n = y.size
    x = np.column_stack([np.ones(n), np.arange(n, dtype=np.float64)])
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    return y - x @ beta, float(beta[0]), float(beta[1])


def _validate(data):
    """Coerce, squeeze (n,1)->(n,), and raise teaching errors for bad input."""
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim > 2:
        raise ValueError(
            f"check_series expects a 1D series (n,) or a 2D panel (n, k); got a "
            f"{arr.ndim}D array with shape {arr.shape}. Reshape or slice the "
            f"array down to observations-by-series first."
        )
    if arr.ndim == 0:
        raise ValueError(
            "check_series expects a series of observations, got a scalar. "
            "Pass a 1D array/list (n,) or a 2D panel (n, k)."
        )
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr[:, 0]
    nan_count = int(np.isnan(arr).sum())
    if nan_count:
        first = int(np.argwhere(np.isnan(arr))[0][0])
        raise ValueError(
            f"check_series needs complete data: found {nan_count} NaN value(s), "
            f"the first at index {first}. The unit-root, portmanteau, and break "
            f"tests in this battery have no missing-value handling, so gaps "
            f"would silently corrupt every p-value. Impute or trim the gaps "
            f"first (state-space/Kalman imputation is on the Module 01 roadmap)."
        )
    n = int(arr.shape[0])
    if n < 20:
        raise ValueError(
            f"check_series needs at least 20 observations, got n={n}. Below "
            f"that the ADF/KPSS unit-root tests and the Ljung-Box portmanteau "
            f"have essentially no power and their asymptotic p-values are not "
            f"trustworthy — collect a longer sample, or inspect the handful of "
            f"values directly."
        )
    return arr


# --------------------------------------------------------------------------- #
# univariate battery
# --------------------------------------------------------------------------- #
def _descriptives(y):
    c = y - y.mean()
    m2 = float(np.mean(c**2))
    skew = float(np.mean(c**3) / m2**1.5) if m2 > 0 else 0.0
    exk = float(np.mean(c**4) / m2**2 - 3.0) if m2 > 0 else 0.0
    return {
        "mean": float(y.mean()),
        "sd": float(y.std(ddof=1)),
        "skew": skew,
        "excess_kurtosis": exk,
        "min": float(y.min()),
        "max": float(y.max()),
    }


def _outliers(y):
    med = float(np.median(y))
    mad = float(np.median(np.abs(y - med)))
    if mad > 0:
        z = 0.6745 * (y - med) / mad
        idx = np.flatnonzero(np.abs(z) > 3.5)
    else:
        idx = np.array([], dtype=int)
    return {
        "method": "modified z-score (median/MAD, |z|>3.5, Iglewicz-Hoaglin)",
        "count": int(idx.size),
        "indices": [int(i) for i in idx[:20]],
        "caveat": (
            "A robust unconditional screen on the level series; it flags level "
            "shifts and fat tails alike. `indices` lists at most the first 20 "
            "flagged positions; `count` is the full total. Model-based "
            "(Chen-Liu) additive/innovational outlier detection is a roadmap "
            "item."
            + (" MAD was zero, so the screen was skipped." if mad == 0 else "")
        ),
    }


def _serial_correlation(z, scale, lags, alpha, tests_run):
    m = z.size
    lb_lags = int(lags) if lags is not None else min(10, m // 5)
    lb_lags = max(1, min(lb_lags, m - 2))
    lb = _core.ljung_box(z, nlags=lb_lags)
    lb_p = float(np.asarray(lb["lb_pvalue"])[-1])
    tests_run.append(
        {"name": f"ljung_box (lags 1..{lb_lags})", "pvalue": lb_p, "alpha": alpha}
    )

    n_acf = min(20, m - 1)
    acf_vals = np.asarray(_core.acf(z, nlags=n_acf)["acf"])[1:]
    # the compiled pacf requires nlags < n/2 STRICTLY; (m - 1) // 2 is the
    # largest valid choice for both parities (m // 2 crashes for even m <= 40)
    n_pacf = max(1, min(20, (m - 1) // 2))
    pacf_vals = np.asarray(_core.pacf(z, nlags=n_pacf))[1:]
    band = 1.96 / np.sqrt(m)
    sig_acf = [int(i + 1) for i in np.flatnonzero(np.abs(acf_vals) > band)]
    sig_pacf = [int(i + 1) for i in np.flatnonzero(np.abs(pacf_vals) > band)]

    run_p, run_q = _initial_run(sig_pacf), _initial_run(sig_acf)
    if run_p == 0 and run_q == 0:
        p_s, q_s = 0, 0
    elif run_p > 0 and (run_q == 0 or run_p <= run_q):
        p_s, q_s = min(run_p, 5), 0  # PACF cuts sooner -> AR signature
    else:
        p_s, q_s = 0, min(run_q, 5)  # ACF cuts sooner -> MA signature
    return {
        "computed_on": scale,
        "ljung_box": {
            "lags": lb["lags"],
            "lb_stat": lb["lb_stat"],
            "lb_pvalue": lb["lb_pvalue"],
        },
        "acf": acf_vals,
        "pacf": pacf_vals,
        "conf_band": float(band),
        "band_note": "±1.96/sqrt(n) white-noise band on the analysis object",
        "significant_acf_lags": sig_acf,
        "significant_pacf_lags": sig_pacf,
        "suggested_arma_orders": {
            "p": int(p_s),
            "q": int(q_s),
            "note": (
                "Cutoff heuristics (PACF run -> AR order, ACF run -> MA order): "
                "starting points for arima_fit plus AIC/BIC comparison over a "
                "small order grid, not a verdict."
            ),
        },
    }, lb_p


def _breaks(z, scale, alpha, max_breaks, trim, tests_run):
    m = z.size
    out = {
        "computed_on": scale,
        "design": "intercept-only (a mean-shift scan)",
        "sup_f": None,
        "bai_perron": None,
        "skipped_reason": None,
        "caveat": (
            "Bai (1997) break-date confidence intervals in their homogeneous "
            "case only. The scan assumes serially uncorrelated errors under the "
            "null; strong autocorrelation inflates the sup-F, and an unremoved "
            "deterministic trend masquerades as a cascade of mean shifts — so "
            "read a rejection on a persistent or trending series with care."
        ),
    }
    x = np.ones((m, 1))
    try:
        sf = _core.sup_f_test(z, x, trim=trim)
    except Exception as exc:  # tiny samples can defeat the trimmed search
        out["skipped_reason"] = f"sup_f_test failed on this sample: {exc}"
        return out, False
    rejected = bool(sf["p_value"] < alpha)
    out["sup_f"] = {
        "stat": float(sf["stat"]),
        "p_value": float(sf["p_value"]),
        "break_date": int(sf["break_date"]),
        "rejected": rejected,
    }
    tests_run.append(
        {"name": "sup_f_test", "pvalue": float(sf["p_value"]), "alpha": alpha}
    )
    if not rejected:
        out["skipped_reason"] = (
            f"bai_perron skipped: sup-F did not reject at alpha={alpha}"
        )
    elif m > _MAX_BAI_PERRON_N:
        out["skipped_reason"] = (
            f"bai_perron skipped: n={m} > {_MAX_BAI_PERRON_N} and the dynamic "
            f"program is O(n^2) — run tsecon.bai_perron directly if you want "
            f"the full multi-break scan."
        )
    else:
        bp = _core.bai_perron(z, x, max_breaks=max_breaks, trim=trim)
        out["bai_perron"] = {
            "n_breaks": int(bp["n_breaks"]),
            "break_dates": bp["break_dates"],
            "ci_90": {"lower": bp["ci_lower_90"], "upper": bp["ci_upper_90"]},
            "ci_95": {"lower": bp["ci_lower_95"], "upper": bp["ci_upper_95"]},
            "regime_starts": bp["regime_starts"],
            "regime_ends": bp["regime_ends"],
            "regime_params": bp["params"],
            "regime_bse": bp["bse"],
            "sup_f_seq": bp["sup_f_seq"],
            "sup_f_crit": bp["sup_f_crit"],
        }
    return out, rejected


def _long_memory(y, diff_used, detrended=None):
    target = y if detrended is None else detrended
    computed_on = "level" if detrended is None else "detrended_level"
    label = "level" if detrended is None else "detrended level"
    lm = _core.long_memory_d(target)
    d, se = float(lm["d"]), float(lm["se"])
    out = {
        "computed_on": computed_on,
        "gph": {"d": d, "se": se, "m": int(lm["m"])},
        "gph_on_differences": None,
        "joint_interpretation": None,
    }
    fire = None
    if diff_used:
        lmd = _core.long_memory_d(np.diff(y))
        dd, dse = float(lmd["d"]), float(lmd["se"])
        out["gph_on_differences"] = {"d": dd, "se": dse, "m": int(lmd["m"])}
        if abs(d - 1.0) <= 1.64 * se:
            lead = (
                f"GPH on the level gives d={d:.3f} (se {se:.3f}), "
                f"statistically indistinguishable from 1 — consistent with "
                f"the unit-root verdict. "
            )
        else:
            lead = (
                f"GPH on the level gives d={d:.3f} (se {se:.3f}), "
                f"significantly {'below' if d < 1.0 else 'above'} 1 at the 5% "
                f"level — the unit-root verdict's integer difference may be "
                f"the wrong filter (a fractional d is plausible); read the "
                f"differences next. "
            )
        text = lead + f"On the first differences d={dd:.3f} (se {dse:.3f}): "
        if dd + 1.64 * dse < 0:
            text += (
                "significantly negative — integer differencing may be too much "
                "(over-differencing); a fractional filter frac_diff(y, d) with "
                f"d ≈ {1 + dd:.2f} on the level is the alternative."
            )
            fire = ("over_difference", dd, dse)
        elif dd - 1.64 * dse > 0 and dd >= 0.1:
            text += (
                "still significantly positive — a fractional alternative "
                f"(0<d<0.5 on the differences) via frac_diff is worth testing."
            )
            fire = ("fractional", dd, dse)
        else:
            text += "close to 0, as expected after differencing an I(1) series."
    else:
        text = f"GPH on the {label} gives d={d:.3f} (se {se:.3f}); "
        if d >= 0.1 and d - 1.64 * se > 0:
            text += (
                "significantly positive: the ACF may decay like a power law "
                "rather than geometrically (long memory), which short-memory "
                "ARMA fits will chase with spuriously high orders."
            )
            fire = ("long_memory", d, se)
        else:
            text += "no evidence of long memory at this bandwidth."
    out["joint_interpretation"] = text
    return out, fire


def _seasonality(z, scale, seasonal_period):
    m = z.size
    out = {
        "seasonal_period": int(seasonal_period) if seasonal_period else None,
        "computed_on": scale,
        "acf_at_seasonal_lags": None,
        "periodogram_ordinate": None,
        "detected_period": None,
        "note": None,
    }
    pg = _core.periodogram(z)
    freqs, psd = np.asarray(pg["freqs"]), np.asarray(pg["psd"])
    if seasonal_period:
        s = int(seasonal_period)
        usable = [lag for lag in (s, 2 * s, 3 * s) if lag < m]
        if usable:
            av = np.asarray(_core.acf(z, nlags=max(usable))["acf"])
            out["acf_at_seasonal_lags"] = [
                {"lag": lag, "acf": float(av[lag])} for lag in usable
            ]
        j = int(np.argmin(np.abs(freqs - 1.0 / s)))
        out["periodogram_ordinate"] = {
            "frequency": float(freqs[j]),
            "period": float(1.0 / freqs[j]) if freqs[j] > 0 else None,
            "psd": float(psd[j]),
        }
        out["note"] = (
            f"Seasonal evidence at period {s}: the ACF at lags s, 2s, 3s and "
            f"the periodogram ordinate nearest frequency 1/{s}. Compare the "
            f"seasonal-lag ACF values against the white-noise band."
        )
    else:
        pos = np.flatnonzero(freqs > 0)
        j = int(pos[np.argmax(psd[pos])])
        f = float(freqs[j])
        if f >= 2.0 / m:  # at least two full cycles in sample, else 'trend'
            out["detected_period"] = float(1.0 / f)
            out["note"] = (
                "detected_period is simply the argmax periodogram ordinate — a "
                "heuristic, not a test. Check the ACF at that lag before "
                "believing it; pass seasonal_period to get seasonal evidence."
            )
        else:
            out["note"] = (
                "The periodogram peak sits at a trivial very-low frequency "
                "(fewer than two cycles in sample) — that is trend/persistence, "
                "not seasonality, so detected_period is None."
            )
    return out


def _univariate(y, seasonal_period, lags, alpha, max_breaks, trim):
    n = y.size
    tests_run = []
    report = {"kind": "univariate", "n": n, "alpha": float(alpha)}
    report["descriptives"] = _descriptives(y)
    report["outliers"] = _outliers(y)

    # --- stationarity family: the verbatim confirmatory-quadrant workflow ---
    st = _core.check_stationarity(y, alpha=alpha)
    report["stationarity"] = dict(st)
    tests_run.append(
        {"name": "adf", "pvalue": float(st["adf_p_value"]), "alpha": alpha}
    )
    tests_run.append(
        {"name": "kpss", "pvalue": float(st["kpss_p_value"]), "alpha": alpha}
    )

    diff_used = st["recommendation"] == "Difference"
    detrend_used = st["recommendation"] == "Detrend"
    if diff_used:
        scale, z = "first_difference", np.diff(y)
        rationale = (
            f"The ADF+KPSS quadrant is '{st['quadrant']}' and the workflow says "
            f"to difference, so all downstream dependence/ARCH/normality/break "
            f"tests run on the first differences (n={z.size})."
        )
    elif detrend_used:
        z, tr_a, tr_b = _detrend(y)
        scale = "detrended_level"
        rationale = (
            f"The ADF+KPSS quadrant is '{st['quadrant']}' and the workflow says "
            f"to detrend, so all downstream dependence/ARCH/normality/break/"
            f"long-memory tests run on the OLS-detrended level (fitted trend "
            f"{tr_a:+.4g} {tr_b:+.4g}*t): a deterministic trend left in place "
            f"reproduces the signature of breaks, ARCH, and long memory."
        )
    else:
        scale, z = "level", y
        rationale = (
            f"The ADF+KPSS quadrant is '{st['quadrant']}' "
            f"({st['recommendation']}), so downstream tests run on the level "
            f"series."
        )
    report["analysis_scale"] = {"scale": scale, "rationale": rationale}

    # --- dependence / ARCH / normality / breaks on the analysis object ---
    report["serial_correlation"], lb_p = _serial_correlation(
        z, scale, lags, alpha, tests_run
    )
    arch_nlags = max(1, min(4, z.size // 5))
    arch = _core.arch_lm(z, nlags=arch_nlags)
    report["arch_effects"] = {
        "computed_on": scale,
        "statistic": float(arch["statistic"]),
        "p_value": float(arch["p_value"]),
        "df": int(arch["df"]),
        "nobs": int(arch["nobs"]),
        "rejected": bool(arch["p_value"] < alpha),
    }
    tests_run.append(
        {"name": "arch_lm", "pvalue": float(arch["p_value"]), "alpha": alpha}
    )
    jb = _core.jarque_bera(z)
    report["normality"] = {
        "computed_on": scale,
        "statistic": float(jb["statistic"]),
        "p_value": float(jb["p_value"]),
        "skewness": float(jb["skewness"]),
        "excess_kurtosis": float(jb["kurtosis"]) - 3.0,
        "rejected": bool(jb["p_value"] < alpha),
    }
    tests_run.append(
        {"name": "jarque_bera", "pvalue": float(jb["p_value"]), "alpha": alpha}
    )
    report["breaks"], breaks_fired = _breaks(
        z, scale, alpha, max_breaks, trim, tests_run
    )

    # --- long memory on the level (detrended when the verdict says trend),
    #     seasonality on the analysis object ---
    report["long_memory"], lm_fire = _long_memory(
        y, diff_used, detrended=z if detrend_used else None
    )
    report["seasonality"] = _seasonality(z, scale, seasonal_period)

    report["tests_run"] = tests_run
    n_tests = len(tests_run)
    n_true_null = sum(1 for t in tests_run if not t["name"].startswith("adf"))
    report["multiple_testing"] = {
        "n_tests": n_tests,
        "n_true_null": n_true_null,
        "alpha": float(alpha),
        "expected_false_rejections": float(n_true_null * alpha),
        "note": (
            f"{n_tests} hypothesis tests ran; each test whose null actually "
            f"holds contributes alpha={alpha} to the expected false-alarm "
            f"count. On a clean stationary series that is {n_true_null} tests "
            f"(ADF's null is a unit root, so an ADF rejection there is "
            f"correct, not a false alarm) — about {n_true_null * alpha:.2f} "
            f"spurious rejections. tsecon reports the families and this "
            f"arithmetic instead of silently multiple-testing-correcting the "
            f"p-values."
        ),
    }

    report["recommendations"] = _univariate_recommendations(
        report, st, lb_p, breaks_fired, lm_fire, seasonal_period, alpha
    )
    return report


def _univariate_recommendations(
    report, st, lb_p, breaks_fired, lm_fire, seasonal_period, alpha
):
    recs = []
    quadrant = st["quadrant"]
    power_caveat = (
        "Near the unit circle, stationary and integrated processes are nearly "
        "observationally equivalent in finite samples — ADF/KPSS organize the "
        "evidence, they cannot manufacture information."
    )
    ur_evidence = {
        "quadrant": quadrant,
        "adf_p_value": float(st["adf_p_value"]),
        "kpss_p_value": float(st["kpss_p_value"]),
        "alpha": float(alpha),
    }

    # 1. unit-root / differencing routing
    if st["recommendation"] == "Difference":
        caveat = power_caveat
        if breaks_fired:
            caveat = (
                "Perron (1989) caution: a one-time break in an otherwise "
                "stationary series masquerades as a unit root, and the break "
                "scan also fired here — check bai_perron's dates before "
                "committing to differences. " + power_caveat
            )
        finding = (
            f"quadrant '{quadrant}': the tests agree the series looks I(1)"
            if quadrant == "UnitRoot"
            else f"quadrant '{quadrant}': neither test rejects — too little "
            f"information; the conservative default is to difference"
        )
        recs.append(
            _rec(
                "unit_root",
                finding,
                ur_evidence,
                "Model in first differences — arima_fit(y, d=1, ...) "
                "differences internally — and re-run check_series on the "
                "differences to pick the short-run orders.",
                ["check_stationarity", "arima_fit"],
                caveat,
            )
        )
        pr_finding = (
            "the series is highly persistent"
            if quadrant == "UnitRoot"
            else "IF the series is highly persistent — this sample cannot "
            "tell (quadrant 'Inconclusive': neither test rejects) — "
            "predictive regressions on it are size-distorted"
        )
        recs.append(
            _rec(
                "persistent_regressor",
                pr_finding,
                ur_evidence,
                "If this series will be a REGRESSOR (e.g. predicting returns), "
                "standard t-tests are size-distorted under persistence: use "
                "predictive_regression (Stambaugh + IVX) for one predictor or "
                "ivx_test for a joint test.",
                ["predictive_regression", "ivx_test"],
                "Only relevant when the series sits on the right-hand side; "
                "irrelevant for modeling it as the outcome.",
            )
        )
    elif quadrant == "Conflict":
        recs.append(
            _rec(
                "trend_or_break",
                "both tests reject: not clean I(0) or I(1) — trend, breaks, or "
                "long memory are the usual culprits",
                ur_evidence,
                "The battery has OLS-detrended the series before every "
                "downstream family (see analysis_scale), because a "
                "deterministic trend left in place reproduces the exact "
                "signature of breaks, ARCH, and long memory. Confirm the "
                "trend reading with adf(y, regression='ct') and "
                "kpss(y, regression='ct'), and read the break-scan and "
                "long-memory families below (computed on the detrended "
                "level) before resorting to differencing.",
                ["adf", "kpss", "sup_f_test", "bai_perron"],
                power_caveat,
            )
        )

    # 2. breaks
    br = report["breaks"]
    if breaks_fired:
        evidence = {
            "sup_f_stat": br["sup_f"]["stat"],
            "sup_f_p_value": br["sup_f"]["p_value"],
            "break_date": br["sup_f"]["break_date"],
        }
        if br["bai_perron"] is not None:
            evidence["n_breaks"] = br["bai_perron"]["n_breaks"]
            evidence["break_dates"] = br["bai_perron"]["break_dates"]
        recs.append(
            _rec(
                "structural_break",
                f"sup-F rejects a constant mean on the {br['computed_on']} "
                f"(p={br['sup_f']['p_value']:.3g})",
                evidence,
                "Full-sample estimates average across regimes and can be badly "
                "biased. Date the breaks with bai_perron (or chow_test when "
                "history hands you the date) and consider per-regime fits or "
                "post-break samples.",
                ["bai_perron", "chow_test"],
                br["caveat"],
            )
        )

    # 3. serial correlation -> ARMA starting orders
    sc = report["serial_correlation"]
    detrended_scale = report["analysis_scale"]["scale"] == "detrended_level"
    ol = report["outliers"]
    outlier_note = (
        f" {ol['count']} outlier(s) were flagged by the robust screen (first "
        f"at index {ol['indices'][0]}); a few extreme points can drive this "
        f"rejection — inspect them before modeling the tails."
        if ol["count"]
        else ""
    )
    if lb_p < alpha:
        orders = sc["suggested_arma_orders"]
        d_note = 1 if report["analysis_scale"]["scale"] == "first_difference" else 0
        if detrended_scale:
            arma_suggestion = (
                f"Start from arima_fit(y_dt, p={orders['p']}, d=0, "
                f"q={orders['q']}) where y_dt is the OLS-detrended series the "
                f"battery analyzed (subtract the fitted trend in "
                f"analysis_scale from y first), and compare AIC/BIC over a "
                f"small order grid — the ACF/PACF cutoff heuristics are "
                f"starting points, not a verdict."
            )
        else:
            arma_suggestion = (
                f"Start from arima_fit(y, p={orders['p']}, d={d_note}, "
                f"q={orders['q']}) and compare AIC/BIC over a small order grid "
                f"— the ACF/PACF cutoff heuristics are starting points, not a "
                f"verdict."
            )
        recs.append(
            _rec(
                "arma_orders",
                f"Ljung-Box rejects whiteness on the {sc['computed_on']} "
                f"(p={lb_p:.3g})",
                {
                    "ljung_box_p_value": lb_p,
                    "suggested_p": orders["p"],
                    "suggested_q": orders["q"],
                    "significant_acf_lags": sc["significant_acf_lags"],
                    "significant_pacf_lags": sc["significant_pacf_lags"],
                },
                arma_suggestion,
                ["arima_fit", "acf", "pacf"],
                "Order heuristics assume a clean cutoff pattern; mixed ARMA "
                "signatures need the IC comparison to disambiguate.",
            )
        )

    # 4. ARCH (with the fat-tails escalation)
    ar, nm = report["arch_effects"], report["normality"]
    if ar["rejected"]:
        fat = nm["rejected"] and nm["excess_kurtosis"] > 0
        garch_target = {
            "level": "y",
            "first_difference": "np.diff(y)",
            "detrended_level": "y_dt",
        }.get(ar["computed_on"], "y")
        suggestion = (
            f"Conditional heteroskedasticity: model the variance with "
            f"garch_fit({garch_target})."
        )
        if detrended_scale:
            suggestion += (
                " (y_dt is the OLS-detrended series the battery analyzed — "
                "subtract the fitted trend in analysis_scale from y first; "
                "ARCH-LM on the trending level itself is not trustworthy.)"
            )
        elif ar["computed_on"] == "first_difference":
            suggestion += (
                " (the battery analyzed the first differences — fit the GARCH "
                "to the differenced/return series, not the integrated level.)"
            )
        if fat:
            suggestion += (
                f" Normality also rejects with fat tails, so use "
                f"garch_fit({garch_target}, dist='t') — Gaussian QMLE point "
                f"estimates are consistent but t innovations restore honest "
                f"density forecasts."
            )
        recs.append(
            _rec(
                "arch",
                f"ARCH-LM rejects homoskedasticity on the {ar['computed_on']} "
                f"(p={ar['p_value']:.3g})",
                {
                    "arch_lm_p_value": ar["p_value"],
                    "jarque_bera_p_value": nm["p_value"],
                    "excess_kurtosis": nm["excess_kurtosis"],
                },
                suggestion,
                ["garch_fit", "arch_lm"],
                "ARCH-LM on the raw series can also pick up neglected mean "
                "dynamics; re-check on ARMA residuals if serial correlation "
                "also fired." + outlier_note,
            )
        )
    elif nm["rejected"]:
        # 5. non-normality without ARCH
        recs.append(
            _rec(
                "normality",
                f"Jarque-Bera rejects normality on the {nm['computed_on']} "
                f"(p={nm['p_value']:.3g}) without ARCH effects",
                {
                    "jarque_bera_p_value": nm["p_value"],
                    "skewness": nm["skewness"],
                    "excess_kurtosis": nm["excess_kurtosis"],
                },
                "Point forecasts are unaffected, but Gaussian prediction "
                "intervals and density forecasts are not trustworthy — re-run "
                "jarque_bera on model residuals (unconditional non-normality "
                "often shrinks once dynamics are modeled) and prefer "
                "bootstrapped intervals if it persists.",
                ["jarque_bera"],
                "JB is an asymptotic chi-square(2) test; it over-rejects in "
                "small samples and says nothing about which moments matter for "
                "your loss." + outlier_note,
            )
        )

    # 6. long memory
    if lm_fire is not None:
        kind, d, se = lm_fire
        lm_scale = report["long_memory"]["computed_on"]
        lm_caveat = (
            "GPH is a small-bandwidth semiparametric estimator: short-memory "
            "AR dynamics bias it upward, and level shifts and unremoved "
            "deterministic trends masquerade as long memory. Cross-check with "
            "long_memory_d(method='local_whittle') and the break scan."
        )
        if kind == "long_memory":
            lm_target = "y_dt" if lm_scale == "detrended_level" else "y"
            lm_suggestion = (
                f"Long memory: short-memory ARMA fits will chase the "
                f"hyperbolic ACF with inflated orders. Prefilter with "
                f"frac_diff({lm_target}, d={d:.2f}) and model the residual, or "
                f"re-estimate d with long_memory_d(method='local_whittle') "
                f"to check robustness."
            )
            if lm_scale == "detrended_level":
                lm_suggestion += (
                    " (y_dt is the OLS-detrended series the battery analyzed "
                    "— subtract the fitted trend in analysis_scale from y "
                    "first.)"
                )
            recs.append(
                _rec(
                    "long_memory",
                    f"GPH d={d:.3f} (se {se:.3f}) on the "
                    f"{lm_scale.replace('_', ' ')} is significantly positive",
                    {"d": d, "se": se, "computed_on": lm_scale},
                    lm_suggestion,
                    ["long_memory_d", "frac_diff"],
                    lm_caveat,
                )
            )
        elif kind == "over_difference":
            recs.append(
                _rec(
                    "long_memory",
                    f"GPH d={d:.3f} (se {se:.3f}) on the differences is "
                    f"significantly negative — integer differencing looks like "
                    f"too much",
                    {"d_on_differences": d, "se": se},
                    f"The level may be fractionally integrated with d<1: apply "
                    f"frac_diff(y, d={1 + d:.2f}) to the level instead of a "
                    f"first difference, then model the filtered series.",
                    ["long_memory_d", "frac_diff"],
                    lm_caveat,
                )
            )
        else:  # fractional memory remaining in the differences
            recs.append(
                _rec(
                    "long_memory",
                    f"GPH d={d:.3f} (se {se:.3f}) on the differences is still "
                    f"significantly positive",
                    {"d_on_differences": d, "se": se},
                    f"A fractional alternative fits the evidence: filter the "
                    f"level with frac_diff(y, d={1 + d:.2f}) (frac_integrate "
                    f"inverts it) rather than stopping at d=1.",
                    ["long_memory_d", "frac_diff", "frac_integrate"],
                    lm_caveat,
                )
            )

    # 7. seasonality — with the honest capability gap
    if seasonal_period:
        se_fam = report["seasonality"]
        recs.append(
            _rec(
                "seasonality",
                f"seasonal evidence reported at period {int(seasonal_period)}",
                {
                    "seasonal_period": int(seasonal_period),
                    "acf_at_seasonal_lags": se_fam["acf_at_seasonal_lags"],
                    "periodogram_ordinate": se_fam["periodogram_ordinate"],
                },
                "Plain speech: tsecon ships no seasonal ARIMA and no X-13 — "
                "both are roadmap, and that gap is real. Today, either add "
                "explicit seasonal-lag terms (regress on lag-s terms / seasonal "
                "dummies, or arima_fit on the seasonally differenced series you "
                "construct upstream) or deseasonalize upstream (e.g. X-13/STL "
                "elsewhere) before modeling here.",
                ["acf", "periodogram", "arima_fit"],
                "Compare the seasonal-lag ACF against the white-noise band "
                "before investing in seasonal structure; a periodogram spike "
                "alone can be a harmonic of a lower frequency.",
            )
        )

    # 8. the clean bill of health (the seasonality entry is informational —
    #    it fires whenever seasonal_period is given — so it does not count
    #    against a clean bill)
    if not [r for r in recs if r["topic"] != "seasonality"]:
        mt = report["multiple_testing"]
        recs.append(
            _rec(
                "no_red_flags",
                f"no family rejected at alpha={alpha}",
                {
                    "n_tests": mt["n_tests"],
                    "n_true_null": mt["n_true_null"],
                    "alpha": mt["alpha"],
                    "expected_false_rejections": mt["expected_false_rejections"],
                },
                "No red flags at alpha — and remember the expected-false-alarm "
                f"arithmetic: the {mt['n_true_null']} true-null tests at "
                f"alpha={alpha} would produce about "
                f"{mt['expected_false_rejections']:.2f} spurious rejections on "
                f"pure noise, so a lone marginal p-value on a future re-run is "
                f"not a discovery. A low-order arima_fit (or a white-noise "
                f"mean model) is the natural baseline.",
                ["arima_fit"],
                "Absence of evidence at alpha is not proof of simplicity — the "
                "tests have limited power in short samples.",
            )
        )
    return recs


# --------------------------------------------------------------------------- #
# multivariate battery
# --------------------------------------------------------------------------- #
def _multivariate(data, lags, alpha):
    n, k = int(data.shape[0]), int(data.shape[1])
    tests_run = []
    report = {"kind": "multivariate", "n": n, "k": k, "alpha": float(alpha)}

    per_series = []
    n_i0 = n_diff = 0
    for j in range(k):
        st = _core.check_stationarity(np.ascontiguousarray(data[:, j]), alpha=alpha)
        per_series.append(
            {
                "index": j,
                "verdict": st["quadrant"],
                "recommendation": st["recommendation"],
            }
        )
        tests_run.append(
            {
                "name": f"adf (series {j})",
                "pvalue": float(st["adf_p_value"]),
                "alpha": alpha,
            }
        )
        tests_run.append(
            {
                "name": f"kpss (series {j})",
                "pvalue": float(st["kpss_p_value"]),
                "alpha": alpha,
            }
        )
        n_i0 += st["quadrant"] == "Stationary"
        n_diff += st["recommendation"] == "Difference"
    report["per_series"] = per_series

    all_i0 = n_i0 == k
    all_i1 = n_diff == k
    mixed = not (all_i0 or all_i1)
    report["integration_summary"] = {
        "n_stationary": int(n_i0),
        "n_difference_recommended": int(n_diff),
        "all_stationary": bool(all_i0),
        "all_i1": bool(all_i1),
        "mixed": bool(mixed),
        "text": (
            "every series looks I(0): a levels VAR is the natural starting point"
            if all_i0
            else "every series draws a 'Difference' verdict (I(1) or "
            "inconclusive): test for cointegration before differencing away "
            "the levels information"
            if all_i1
            else "integration orders are mixed: a levels VAR mixing I(0) and "
            "I(1) series risks spurious dynamics and nonstandard inference"
        ),
    }

    # --- cointegration (only meaningful when every series is I(1)) ---
    rank = None
    if all_i1:
        try:
            jo = _core.johansen(data, k_ar_diff=1)
            rank_raw = jo["rank_trace_5pct"]
            if rank_raw is None:
                report["cointegration"] = {
                    "skipped_reason": (
                        "johansen ran but could not deliver a rank"
                        + (
                            f": its trace-test critical values are tabulated "
                            f"only up to 12 series and this panel has k={k}"
                            if k > 12
                            else " on this panel"
                        )
                        + " — the cointegrating rank is UNDETERMINED, not "
                        "zero. Test an economically motivated subset "
                        "(johansen on fewer columns, or another k_ar_diff) "
                        "or compress the panel with factor_model first."
                    )
                }
            else:
                rank = int(rank_raw)
                report["cointegration"] = {
                    "method": (
                        "Johansen trace test at 5% (k_ar_diff=1; deterministic "
                        "case: unrestricted constant in the data, no trend in "
                        "the cointegrating relation — statsmodels det_order=0)"
                    ),
                    "rank": rank,
                    "trace_stat": jo["trace_stat"],
                    "trace_crit_90_95_99": jo["trace_crit_90_95_99"],
                    "max_eig_stat": jo["max_eig_stat"],
                    "rank_max_eig_5pct": jo["rank_max_eig_5pct"],
                    "interpretation": (
                        f"trace rank = {rank}: "
                        + (
                            "no cointegration found — a VAR in first "
                            "differences loses nothing"
                            if rank == 0
                            else f"{rank} cointegrating relation(s) — "
                            f"differencing everything would discard the "
                            f"error-correction terms; route through vecm"
                        )
                    ),
                }
        except Exception as exc:
            report["cointegration"] = {
                "skipped_reason": (
                    f"johansen failed on this panel ({exc}) — the "
                    f"cointegrating rank is UNDETERMINED, not zero. Re-run "
                    f"tsecon.johansen directly on a subset of the columns or "
                    f"with a different k_ar_diff."
                )
            }
    else:
        report["cointegration"] = {
            "skipped_reason": (
                "johansen skipped: the trace test assumes every series is "
                "I(1), and the per-series verdicts disagree"
            )
        }

    # --- VAR lag selection on the appropriate scale ---
    if all_i1 and rank == 0:
        var_data, var_scale = np.diff(data, axis=0), "first_difference"
        scale_note = "all series I(1) with no cointegration: VAR on differences"
    elif all_i1:
        var_data, var_scale = data, "level"
        scale_note = (
            "cointegrated (or rank undetermined) I(1) system: lag selection on "
            "levels; route through vecm with k_ar_diff = selected_lag - 1"
        )
    else:
        var_data, var_scale = data, "level"
        scale_note = (
            "levels VAR (all stationary)"
            if all_i0
            else "levels used despite mixed integration orders — see the "
            "mixed_integration warning"
        )

    nv = var_data.shape[0]
    cap = int(lags) if lags is not None else 8
    max_lag = 0
    for p in range(1, cap + 1):
        if nv - p - (k * p + 1) >= 10:  # >=10 residual dof per equation
            max_lag = p
    sel = None
    if cap < 1:
        report["var_lag_selection"] = {
            "skipped_reason": (
                f"VAR lag search skipped: lags={cap} caps the search below 1. "
                f"For 2D input the `lags` argument is the lag-search cap "
                f"(default 8), not a Ljung-Box horizon — pass lags>=1 or omit "
                f"it."
            )
        }
        report["stability"] = {"skipped_reason": "no VAR was fit"}
    elif max_lag == 0:
        report["var_lag_selection"] = {
            "skipped_reason": (
                f"no VAR lag is feasible with n={nv}, k={k}: even lags=1 "
                f"leaves fewer than 10 residual degrees of freedom per equation"
            )
        }
        report["stability"] = {"skipped_reason": "no VAR was fit"}
    else:
        try:
            aic, bic, hqic = [], [], []
            for p in range(1, max_lag + 1):
                fit = _core.var_fit(var_data, lags=p)
                aic.append(float(fit["aic"]))
                bic.append(float(fit["bic"]))
                hqic.append(float(fit["hqic"]))
            sel = int(np.argmin(bic)) + 1
            best = _core.var_fit(var_data, lags=sel)
        except Exception as exc:
            sel = None
            report["var_lag_selection"] = {
                "skipped_reason": (
                    f"var_fit failed on this panel: {exc}. The usual cause is "
                    f"(near-)collinear columns — a VAR cannot be estimated "
                    f"when one series is (close to) a linear combination of "
                    f"the others; drop duplicate or linearly dependent series "
                    f"and re-run check_series."
                )
            }
            report["stability"] = {"skipped_reason": "no VAR was fit"}
        else:
            report["var_lag_selection"] = {
                "scale": var_scale,
                "scale_note": scale_note,
                "lags_tried": list(range(1, max_lag + 1)),
                "aic": aic,
                "bic": bic,
                "hqic": hqic,
                "selected_by_bic": sel,
            }
            report["stability"] = {
                "scale": var_scale,
                "lags": sel,
                "is_stable": bool(best["is_stable"]),
                "min_root": float(best["min_root"]),
                "note": (
                    "stable iff min_root > 1 (moduli of the reciprocal "
                    "characteristic roots)."
                    + (
                        " A levels VAR of a cointegrated system carries a "
                        "unit root by construction — instability here is "
                        "expected; use vecm."
                        if all_i1 and (rank or 0) > 0
                        else ""
                    )
                ),
            }

    report["tests_run"] = tests_run
    n_tests = len(tests_run)
    n_true_null = sum(1 for t in tests_run if not t["name"].startswith("adf"))
    report["multiple_testing"] = {
        "n_tests": n_tests,
        "n_true_null": n_true_null,
        "alpha": float(alpha),
        "expected_false_rejections": float(n_true_null * alpha),
        "note": (
            f"{n_tests} per-series hypothesis tests ran; each test whose null "
            f"actually holds contributes alpha={alpha} to the expected "
            f"false-alarm count — about {n_true_null * alpha:.2f} spurious "
            f"rejections when every series is clean and stationary (ADF's "
            f"null is a unit root, so its rejections are excluded from the "
            f"count). The battery shows the arithmetic instead of silently "
            f"correcting."
        ),
    }
    report["recommendations"] = _multivariate_recommendations(
        report, all_i0, all_i1, mixed, rank, sel, k, n, alpha
    )
    return report


def _multivariate_recommendations(report, all_i0, all_i1, mixed, rank, sel, k, n, alpha):
    recs = []
    lag_txt = f"lags={sel}" if sel else "lags=1"
    stab = report["stability"]

    if all_i0:
        recs.append(
            _rec(
                "var",
                "every series looks stationary: a levels VAR is well posed",
                {
                    "selected_by_bic": sel,
                    "is_stable": stab.get("is_stable"),
                    "min_root": stab.get("min_root"),
                },
                f"Fit var_fit(data, {lag_txt}) (BIC choice; compare AIC/HQIC "
                f"in var_lag_selection), then var_irf for shock propagation "
                f"and var_fevd for variance shares.",
                ["var_fit", "var_irf", "var_fevd"],
                "BIC prefers parsimony; if residual autocorrelation survives, "
                "revisit with the AIC choice.",
            )
        )
        if stab.get("is_stable") is False:
            recs.append(
                _rec(
                    "var_stability",
                    "the BIC-selected VAR is not stable despite stationary "
                    "verdicts",
                    {"min_root": stab.get("min_root"), "lags": sel},
                    "Treat the IRFs with suspicion: near-unit roots or breaks "
                    "may sit in the system — re-check per-series verdicts and "
                    "the break scan (run check_series on each column).",
                    ["var_fit", "check_stationarity"],
                    "Stationary univariate verdicts do not guarantee a stable "
                    "joint system in finite samples.",
                )
            )
    elif all_i1 and rank is not None and rank > 0:
        co = report["cointegration"]
        recs.append(
            _rec(
                "cointegration",
                f"every series is I(1) and Johansen finds rank {rank}",
                {"rank": rank, "trace_stat": co.get("trace_stat")},
                f"Model the system as a VECM: vecm(data, k_ar_diff="
                f"{max((sel or 1) - 1, 0)}, coint_rank={rank}). Differencing "
                f"everything would throw away the error-correction terms that "
                f"tie the levels together.",
                ["johansen", "vecm"],
                "Johansen's rank is sensitive to the lag order and "
                "deterministic terms; k_ar_diff=1 and an unrestricted constant "
                "in the data (no trend in the cointegrating relation) were "
                "assumed here. The trace test uses asymptotic critical values "
                "and over-rejects in small samples and as k grows — treat a "
                "marginal rank call at this n as provisional.",
            )
        )
    elif all_i1 and rank == 0:
        recs.append(
            _rec(
                "difference_first",
                "every series is I(1) and Johansen finds no cointegration",
                {"rank": rank, "selected_by_bic": sel},
                f"Difference each series and fit the VAR on the differences — "
                f"var_fit(np.diff(data, axis=0), {lag_txt}) — then var_irf/"
                f"var_fevd as usual.",
                ["var_fit", "var_irf", "var_fevd"],
                "A levels VAR on uncointegrated I(1) data risks spurious "
                "regression; the rank-0 verdict is itself uncertain, so re-test "
                "with other k_ar_diff values if theory expects a long-run link.",
            )
        )
    elif all_i1:  # rank is None: Johansen could not be read — say so honestly
        co = report["cointegration"]
        recs.append(
            _rec(
                "cointegration_undetermined",
                "every series draws a Difference verdict but the Johansen "
                "test could not deliver a cointegrating rank",
                {
                    "rank": None,
                    "k": k,
                    "skipped_reason": co.get("skipped_reason"),
                },
                "Settle the rank before choosing levels vs differences: run "
                "johansen on an economically motivated subset of the columns "
                "(or with a different k_ar_diff), or compress the panel with "
                "factor_model. Until then the battery keeps VAR lag selection "
                "on levels rather than risk differencing away "
                "error-correction terms.",
                ["johansen", "factor_model", "var_fit"],
                "Differencing an actually-cointegrated system discards the "
                "error-correction terms, while a levels VAR of an "
                "uncointegrated I(1) system risks spurious regression — "
                "neither route is safe until the rank is determined.",
            )
        )
    if mixed:
        recs.append(
            _rec(
                "mixed_integration",
                "integration orders are mixed across series",
                {
                    "n_stationary": report["integration_summary"]["n_stationary"],
                    "n_difference_recommended": report["integration_summary"][
                        "n_difference_recommended"
                    ],
                    "per_series": report["per_series"],
                },
                "Difference the I(1) series (and only those) to a common order "
                "before a VAR, or model levels knowing inference on some "
                "coefficients is nonstandard. Re-run check_stationarity per "
                "series after transforming to confirm balance.",
                ["check_stationarity", "var_fit"],
                "Mixing I(0) and I(1) variables in a levels VAR makes some "
                "Wald tests nonstandard (Sims-Stock-Watson); balance the "
                "orders first unless you know why you should not.",
            )
        )

    recs.append(
        _rec(
            "single_shock_irf",
            "IRFs for one identified shock do not need the whole VAR",
            {"k": k, "selected_by_bic": sel},
            "If the question is the dynamic response to a single identified "
            "shock, local projections — lp(y, shock) or lp_iv with an "
            "instrument — are robust to VAR misspecification at the cost of "
            "wider bands.",
            ["lp", "lp_iv"],
            "LP and VAR IRFs estimate the same object under correct "
            "specification; large divergence is itself a specification "
            "diagnostic.",
        )
    )
    if k >= 3:
        recs.append(
            _rec(
                "spillovers",
                f"with k={k} series, pairwise spillover questions arise",
                {"k": k},
                "For 'who transmits to whom' questions, connectedness computes "
                "the Diebold-Yilmaz spillover table from the VAR's generalized "
                "FEVD.",
                ["connectedness"],
                "Connectedness inherits the VAR's lag choice and any "
                "instability; fix those first.",
            )
        )
    if k > 12:
        recs.append(
            _rec(
                "dimensionality",
                f"k={k} exceeds what an unrestricted VAR estimates well",
                {"k": k, "n": n},
                "An unrestricted VAR spends k*p+1 coefficients per equation — "
                "compress the panel with factor_model (Bai-Ng selection) or "
                "estimate a favar instead of a raw VAR.",
                ["favar", "factor_model"],
                "Factor approaches change the interpretation of shocks: they "
                "are shocks to factors, not to named series.",
            )
        )
    return recs


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def check_series(
    data,
    seasonal_period=None,
    lags=None,
    alpha=0.05,
    max_breaks=5,
    trim=0.15,
):
    """One-call diagnostic battery with model recommendations.

    See the module docstring for the design stance; the returned dict is
    plain JSON-serializable Python with ``kind`` either ``"univariate"``
    (1D input) or ``"multivariate"`` (2D ``(n, k)`` input, k >= 2; a
    ``(n, 1)`` column squeezes to univariate).

    ``lags`` means different things by input shape: for 1D input it is the
    Ljung-Box horizon (default ``min(10, n // 5)``); for 2D input it caps
    the VAR lag search (default 8). ``alpha`` must lie in (0.01, 0.10] —
    the compiled KPSS p-value is clamped to that range, so values outside
    it would decide the confirmatory quadrant by the clamp rather than the
    data. ``seasonal_period`` must be an integer >= 2 with at least two
    full cycles in sample.
    """
    alpha = _validate_alpha(alpha)
    arr = _validate(data)
    if seasonal_period is not None:
        s = int(seasonal_period)
        if s != seasonal_period or s < 2:
            raise ValueError(
                f"seasonal_period must be an integer number of observations "
                f"per cycle, at least 2 (e.g. 4 for quarterly data, 12 for "
                f"monthly); got {seasonal_period!r}. Omit it to let the "
                f"periodogram scan for a dominant period instead."
            )
        if arr.ndim == 1 and arr.shape[0] < 2 * s:
            raise ValueError(
                f"check_series needs at least two full seasonal cycles to "
                f"say anything about seasonality: n={arr.shape[0]} < "
                f"2*seasonal_period={2 * s}. Collect a longer sample or omit "
                f"seasonal_period."
            )
    if arr.ndim == 1:
        report = _univariate(
            np.ascontiguousarray(arr), seasonal_period, lags, alpha, max_breaks, trim
        )
    else:
        report = _multivariate(np.ascontiguousarray(arr), lags, alpha)
    return _py(report)
