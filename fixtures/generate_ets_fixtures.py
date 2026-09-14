"""Golden fixtures for the innovations state-space exponential-smoothing
family (`ets_fit`, `auto_ets`; crate `tsecon-ets`).

VALIDATION-FIRST / NON-CIRCULAR: this generator never imports tsecon. Its
references, each with its honest grade:

INDEPENDENT PACKAGE (statsmodels 0.15 `tsa.exponential_smoothing.ets.ETSModel`
and `tsa.holtwinters.ExponentialSmoothing`), pinned at 1e-10 unless stated:

  * `fixed[*].statsmodels` — for the TWENTY models WITHOUT a multiplicative
    seasonal (every error x trend x damped combination with seasonal N or A):
    `ETSModel(initialization_method="known", ...).smooth(params)` at stated
    smoothing parameters and initial states — `llf`, `fittedvalues`, `resid`,
    the `level` / `slope` / `season` state paths, `forecast(h)`, and
    `simulate(h, anchor="end", random_errors=E)` along stated innovation
    paths; for the SIX class-1 models (A,N,N) (A,A,N) (A,Ad,N) (A,N,A)
    (A,A,A) (A,Ad,A) also `get_prediction(...).var_pred_mean` (statsmodels'
    exact method).
  * `fixed[*].statsmodels_simulate_start` — for ALL THIRTY models,
    `simulate(h, anchor="start", random_errors=E)`: the innovations recursion
    run forward from the INITIAL state along stated innovations. statsmodels'
    `simulate` is written in the Hyndman innovations form (its `kappa`
    terms), so this leg is exact for the multiplicative-seasonal models too.
  * `heuristic[*]` — `holtwinters.ExponentialSmoothing(...,
    initialization_method="heuristic")` `initial_level` / `initial_trend` /
    `initial_seasons` (the Hyndman 2008 section 2.6.1 heuristic), cross-
    checked here against `ETSModel(initialization_method="heuristic")` before
    storing; and `_initialization_simple` for a sample too short for it.
  * `mle[*]` — `ETSModel(...).fit()` (L-BFGS-B, statsmodels' default): the
    fitted smoothing parameters, initial states, `llf`, and the criteria with
    statsmodels' parameter count. Two optimisers on one likelihood agree to
    their stopping tolerances, not bitwise: the Rust golden asserts the
    crate's optimum matches-or-beats the statsmodels log-likelihood at a
    stated slack and its parameters lie within a stated distance.

DOCUMENTED-FORMULA transcription (NumPy, this file), pinned at 1e-12:

  * `fixed[*].transcription` — for ALL THIRTY models (per-period paths
    stored only for the ten multiplicative-seasonal models; for the other
    twenty they equal the stored statsmodels paths and are asserted so
    before being dropped), the recursion of
    Hyndman, Koehler, Ord & Snyder (2008, "Forecasting with Exponential
    Smoothing", Tables 2.2/2.3), written as R's forecast::ets `etscalc.c`
    `update` does (function `hyndman_recursion` below): with
    q = l_{t-1} (+ phi b_{t-1} | * b_{t-1}^phi), s = s_{t-m},
    f = q (+ s | * s), p = y_t (- s | / s):
        l_t = q + alpha (p - q)
        b_t = phi b_{t-1} + (beta/alpha)(l_t - l_{t-1})            [additive]
            = b_{t-1}^phi + (beta/alpha)(l_t / l_{t-1} - b_{t-1}^phi) [mult.]
        s_t = s + gamma ((y_t - q) - s)   [additive] | s + gamma (y_t / q - s)
        e_t = y_t - f  [additive error] | (y_t - f) / f  [multiplicative]
    loglik = -(n/2)(ln(2 pi sigma2) + 1) - [mult. error] sum ln|f_t|,
    sigma2 = mean(e_t^2) (Ord-Koehler-Snyder 1997; statsmodels' `loglike`;
    R's `ets` omits the constant -(n/2)(ln(2 pi / n) + 1)). The point
    forecast is the same recursion with y_t := f_t. The generator ASSERTS
    that this transcription reproduces statsmodels at 1e-10 on the twenty
    models where statsmodels' Cython smoother follows the innovations form,
    and RECORDS (`statsmodels_smoother_gap`) the measured gap on the ten
    multiplicative-seasonal models, where statsmodels' smoother updates the
    seasonal with the post-update level l_t and gamma* = gamma / (1 - alpha)
    (the classical Holt-Winters recursion, which agrees with the innovations
    form only to first order in the error) — a convention difference, stated,
    not a golden.
  * `fixed[*].class1_variance_table61` — Hyndman et al. (2008) Table 6.1,
    the relative h-step forecast-error variance v_h / sigma2 with
    k = floor((h - 1) / m):
      (A,N,N)  1 + alpha^2 (h - 1)
      (A,A,N)  1 + (h-1)[alpha^2 + alpha beta h + beta^2 h (2h - 1) / 6]
      (A,Ad,N) 1 + alpha^2 (h-1)
               + beta phi h / (1-phi)^2 [2 alpha (1-phi) + beta phi]
               - beta phi (1-phi^h) / ((1-phi)^2 (1-phi^2))
                 [2 alpha (1-phi^2) + beta phi (1 + 2 phi - phi^h)]
      (A,N,A)  1 + alpha^2 (h-1) + gamma k (2 alpha + gamma)
      (A,A,A)  1 + (h-1)[alpha^2 + alpha beta h + beta^2 h (2h-1) / 6]
               + gamma k [2 alpha + gamma + beta m (k + 1)]
      (A,Ad,A) 1 + alpha^2 (h-1) + gamma k (2 alpha + gamma)
               + beta phi h / (1-phi)^2 [2 alpha (1-phi) + beta phi]
               - beta phi (1-phi^h) / ((1-phi)^2 (1-phi^2))
                 [2 alpha (1-phi^2) + beta phi (1 + 2 phi - phi^h)]
               + 2 beta gamma phi / ((1-phi)(1-phi^m))
                 [k (1-phi^m) - phi^m (1 - phi^{m k})]
    The generator ASSERTS each closed form against the general linear
    formula v_h = sigma2 [1 + sum_{j<h} (w' F^{j-1} g)^2] built from the
    explicit (w, F, g) matrices (function `class1_general_variance`) at
    1e-12 before storing, and against statsmodels' exact `get_prediction`.
  * `candidates[*]` — the admissible candidate set of R's forecast::ets
    (`ets.R`, the errortype/trendtype/seasontype/damped loop with
    `restrict` and `allow.multiplicative.trend`), enumerated here for the
    documented option combinations.

Seasonal-state convention (all blocks): `initial_seasonal[j]` is the seasonal
state in force for observation j (Hyndman's s_{j-m}); this is statsmodels'
`initial_seasonal=` argument order and its `results.initial_seasonal`.
Under the estimated initialisation statsmodels pins the index applied to
observation 0 at 0 (additive) / 1 (multiplicative); the crate normalises
the indices to sum to zero / average one (R's convention) and counts m - 1
free seasonal states. The two are equivalent identifications: `mle[*]`
stores statsmodels' raw states AND the same states converted to the crate's
normalisation (`converted_initial_*`), which leaves every fitted value
unchanged; the criteria are stored with statsmodels' count (m seasonal
states) and the crate's count is stated (`k_params_crate`).

Data (derived transformations only, following the fixture policy):

  * `sim_aadn`, `sim_mam4`: seeded NumPy draws through stated ETS DGPs
    (the innovations form above).
  * `co2_monthly`: monthly means of the bundled `statsmodels.datasets.co2`
    (Mauna Loa CO2, weekly, NOAA public data), the missing months
    linearly interpolated — the innovations form has no missing-value
    mechanism, so the fixture documents how the gaps were treated.
  * `airline`: the Box & Jenkins (1976) Series G monthly airline passengers
    (`get_rdataset("AirPassengers", "datasets")`), public domain and already
    embedded in this repository's `sarima.json`.
  * `log_ukgas`: the natural logarithm of `get_rdataset("UKgas",
    "datasets")` (quarterly UK gas consumption 1960-1986, Durbin & Koopman's
    textbook series), the transformation under which its seasonal is
    additive — stored only in logs.

Doubles are written with json's shortest round-trip repr, which the Rust
golden test parses to identical bits (serde_json `float_roundtrip`).

Run:  .venv/bin/python fixtures/generate_ets_fixtures.py
"""

from __future__ import annotations

import json
import math
import platform
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import statsmodels
import statsmodels.api as sm
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.exponential_smoothing.initialization import (
    _initialization_heuristic,
    _initialization_simple,
)
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.simplefilter("ignore")

OUT = Path(__file__).resolve().parent / "ets.json"
H = 12
N_PATHS = 4


# ------------------------------------------------------------ the recursion

def hyndman_recursion(y, error, trend, damped, seasonal, m, alpha, beta, gamma, phi, l0, b0, s0):
    """Hyndman et al. (2008) Tables 2.2/2.3 in the etscalc.c `update` form.

    Returns fitted, level, trend, seasonal paths, residuals, loglik, sigma2,
    and the final state (seasonal in time order for the forecast steps).
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    if not damped:
        phi = 1.0
    l, b = float(l0), (float(b0) if trend else 0.0)
    ring = list(s0) if seasonal else []
    fitted = np.empty(n)
    L = np.empty(n)
    B = np.empty(n)
    S = np.empty(n)
    resid = np.empty(n)
    for t in range(n):
        if trend is None:
            q, phib = l, 0.0
        elif trend == "add":
            phib = phi * b
            q = l + phib
        else:
            phib = b ** phi if damped else b
            q = l * phib
        s = ring[t % m] if seasonal else 0.0
        if seasonal is None:
            f, p = q, y[t]
        elif seasonal == "add":
            f, p = q + s, y[t] - s
        else:
            f, p = q * s, y[t] / s
        fitted[t] = f
        resid[t] = (y[t] - f) / f if error == "mul" else y[t] - f
        l_new = q + alpha * (p - q)
        if trend is None:
            b_new = b
        elif trend == "add":
            b_new = phib + (beta / alpha) * ((l_new - l) - phib)
        else:
            b_new = phib + (beta / alpha) * (l_new / l - phib)
        if seasonal is None:
            s_new = 0.0
        elif seasonal == "add":
            s_new = s + gamma * ((y[t] - q) - s)
        else:
            s_new = s + gamma * (y[t] / q - s)
        L[t], B[t], S[t] = l_new, b_new, s_new
        l, b = l_new, b_new
        if seasonal:
            ring[t % m] = s_new
    sigma2 = float(np.mean(resid ** 2))
    ll = -0.5 * n * (math.log(2 * math.pi * sigma2) + 1.0)
    if error == "mul":
        ll -= float(np.sum(np.log(np.abs(fitted))))
    final_seasonal = [ring[(n + j) % m] for j in range(m)] if seasonal else None
    return fitted, L, B, S, resid, ll, sigma2, (l, b if trend else None, final_seasonal)


def hyndman_forecast(error, trend, damped, seasonal, m, alpha, beta, gamma, phi, state, h, errors=None):
    """Zero-error (or given-innovation) path from `state`; errors in the
    error's own units (additive or relative)."""
    l, b, s0 = state
    if not damped:
        phi = 1.0
    b = b if trend else 0.0
    ring = list(s0) if seasonal else []
    out = np.empty(h)
    for j in range(h):
        if trend is None:
            q, phib = l, 0.0
        elif trend == "add":
            phib = phi * b
            q = l + phib
        else:
            phib = b ** phi if damped else b
            q = l * phib
        s = ring[j % m] if seasonal else 0.0
        f = q if seasonal is None else (q + s if seasonal == "add" else q * s)
        e = 0.0 if errors is None else errors[j]
        yj = f * (1.0 + e) if error == "mul" else f + e
        out[j] = yj
        p = yj if seasonal is None else (yj - s if seasonal == "add" else yj / s)
        l_new = q + alpha * (p - q)
        if trend is None:
            b_new = b
        elif trend == "add":
            b_new = phib + (beta / alpha) * ((l_new - l) - phib)
        else:
            b_new = phib + (beta / alpha) * (l_new / l - phib)
        if seasonal:
            tt = yj - q if seasonal == "add" else yj / q
            ring[j % m] = s + gamma * (tt - s)
        l, b = l_new, b_new
    return out


# ------------------------------------------------------- class-1 variances

def table61(model, h, alpha, beta, gamma, phi, m):
    h = np.asarray(h, dtype=float)
    k = np.floor((h - 1) / m) if m > 1 else np.zeros_like(h)
    if model == "ANN":
        return 1 + alpha ** 2 * (h - 1)
    if model == "AAN":
        return 1 + (h - 1) * (alpha ** 2 + alpha * beta * h + beta ** 2 * h * (2 * h - 1) / 6)
    damp = lambda: (  # noqa: E731
        (beta * phi * h) / (1 - phi) ** 2 * (2 * alpha * (1 - phi) + beta * phi)
        - (beta * phi * (1 - phi ** h)) / ((1 - phi) ** 2 * (1 - phi ** 2))
        * (2 * alpha * (1 - phi ** 2) + beta * phi * (1 + 2 * phi - phi ** h))
    )
    if model == "AAdN":
        return 1 + alpha ** 2 * (h - 1) + damp()
    if model == "ANA":
        return 1 + alpha ** 2 * (h - 1) + gamma * k * (2 * alpha + gamma)
    if model == "AAA":
        return (1 + (h - 1) * (alpha ** 2 + alpha * beta * h + beta ** 2 / 6 * h * (2 * h - 1))
                + gamma * k * (2 * alpha + gamma + beta * m * (k + 1)))
    if model == "AAdA":
        return (1 + alpha ** 2 * (h - 1) + gamma * k * (2 * alpha + gamma) + damp()
                + (2 * beta * gamma * phi) / ((1 - phi) * (1 - phi ** m))
                * (k * (1 - phi ** m) - phi ** m * (1 - phi ** (m * k))))
    raise ValueError(model)


def class1_general_variance(model, h, alpha, beta, gamma, phi, m):
    """v_h / sigma2 = 1 + sum_{j=1}^{h-1} (w' F^{j-1} g)^2 from explicit
    matrices, state x = (l, b, s_t, s_{t-1}, ..., s_{t-m+1})."""
    has_b = model[1] == "A"
    has_s = model[-1] == "A"
    if "d" not in model:
        phi = 1.0
    k = 1 + int(has_b) + (m if has_s else 0)
    w = np.zeros(k)
    F = np.zeros((k, k))
    g = np.zeros(k)
    w[0] = 1.0
    F[0, 0] = 1.0
    g[0] = alpha
    if has_b:
        w[1] = phi
        F[0, 1] = phi
        F[1, 1] = phi
        g[1] = beta
    if has_s:
        o = 1 + int(has_b)
        w[o + m - 1] = 1.0            # y_{t+1} uses s_{t-m+1}
        F[o, o + m - 1] = 1.0         # new s_{t+1} = s_{t-m+1} (+ gamma e)
        for i in range(1, m):
            F[o + i, o + i - 1] = 1.0  # shift
        g[o] = gamma
    out = []
    for hh in h:
        acc = 1.0
        Fp = np.eye(k)
        for _ in range(1, hh):
            acc += float(w @ Fp @ g) ** 2
            Fp = Fp @ F
        out.append(acc)
    return np.array(out)


# --------------------------------------------------------------- the data

def load_series():
    rng = np.random.default_rng(20260911)
    # ETS(A,Ad,N): l0 = 100, b0 = 0.5, alpha 0.4, beta 0.1, phi 0.9, sigma 2.
    n = 200
    e = 2.0 * rng.standard_normal(n)
    l, b = 100.0, 0.5
    y = np.empty(n)
    for t in range(n):
        q = l + 0.9 * b
        y[t] = q + e[t]
        l = q + 0.4 * e[t]
        b = 0.9 * b + 0.1 * e[t]
    sim_aadn = y
    # ETS(M,A,M) m = 4: l0 = 100, b0 = 0.5, s0 = (0.9, 1.1, 1.05, 0.95),
    # alpha 0.3, beta 0.05, gamma 0.1, sigma 0.04 (relative).
    n = 160
    e = 0.04 * rng.standard_normal(n)
    l, b = 100.0, 0.5
    ring = [0.9, 1.1, 1.05, 0.95]
    y = np.empty(n)
    for t in range(n):
        q = l + b
        s = ring[t % 4]
        f = q * s
        y[t] = f * (1 + e[t])
        l = q * (1 + 0.3 * e[t])
        b = b + 0.05 * q * e[t]
        ring[t % 4] = s * (1 + 0.1 * e[t])
    sim_mam4 = y
    co2 = sm.datasets.co2.load_pandas().data["co2"]
    monthly = co2.resample("MS").mean()
    n_missing = int(monthly.isna().sum())
    co2_monthly = monthly.interpolate(method="linear").to_numpy(dtype=float)
    assert np.all(np.isfinite(co2_monthly))
    airline = sm.datasets.get_rdataset("AirPassengers", "datasets").data["value"].to_numpy(dtype=float)
    ukgas = sm.datasets.get_rdataset("UKgas", "datasets").data["value"].to_numpy(dtype=float)
    log_ukgas = np.log(ukgas)
    return {
        "sim_aadn": (sim_aadn, 1),
        "sim_mam4": (sim_mam4, 4),
        "co2_monthly": (co2_monthly, 12),
        "airline": (airline, 12),
        "log_ukgas": (log_ukgas, 4),
    }, n_missing


SPEC_LETTERS = {"A": "add", "M": "mul", "N": None}


def parse(short):
    """'MAdM' -> (error, trend, damped, seasonal)."""
    error = SPEC_LETTERS[short[0]]
    damped = "d" in short
    body = short[1:].replace("d", "")
    trend = SPEC_LETTERS[body[0]]
    seasonal = SPEC_LETTERS[body[1]]
    return error, trend, damped, seasonal


def sm_params(short, alpha, beta, gamma, phi):
    error, trend, damped, seasonal = parse(short)
    p = [alpha]
    if trend:
        p.append(beta)
    if seasonal:
        p.append(gamma)
    if damped:
        p.append(phi)
    return np.array(p)


def fixed_case(name, short, series_name, series, m, alpha, beta, gamma, phi, l0, b0, s0, rng):
    error, trend, damped, seasonal = parse(short)
    y = series
    n = len(y)
    mm = m if seasonal else 1
    fitted, L, B, S, resid, ll, sigma2, final = hyndman_recursion(
        y, error, trend, damped, seasonal, mm, alpha, beta, gamma, phi, l0, b0, s0)
    fc = hyndman_forecast(error, trend, damped, seasonal, mm, alpha, beta, gamma, phi, final, H)
    errors = rng.standard_normal((N_PATHS, H)) * math.sqrt(sigma2)
    paths_end = [hyndman_forecast(error, trend, damped, seasonal, mm, alpha, beta, gamma, phi,
                                  final, H, errors=errors[i]).tolist() for i in range(N_PATHS)]
    init_state = (l0, b0 if trend else None, list(s0) if seasonal else None)
    paths_start = [hyndman_forecast(error, trend, damped, seasonal, mm, alpha, beta, gamma, phi,
                                    init_state, H, errors=errors[i]).tolist() for i in range(N_PATHS)]
    case = {
        "name": name,
        "short_name": short,
        "series": series_name,
        "error": error, "trend": trend, "damped": damped, "seasonal": seasonal,
        "seasonal_periods": m if seasonal else None,
        "alpha": alpha, "beta": beta if trend else None, "gamma": gamma if seasonal else None,
        "phi": phi if damped else None,
        "initial_level": l0, "initial_trend": b0 if trend else None,
        "initial_seasonal": list(s0) if seasonal else None,
        "h": H,
        "errors": errors.tolist(),
        "transcription": {
            "loglik": ll, "sigma2": sigma2,
            "fitted": fitted.tolist(), "resid": resid.tolist(),
            "level": L.tolist(), "trend": B.tolist() if trend else None,
            "seasonal": S.tolist() if seasonal else None,
            "final_level": final[0], "final_trend": final[1], "final_seasonal": final[2],
            "forecast": fc.tolist(),
            "paths_from_end": paths_end,
            "paths_from_start": paths_start,
        },
    }
    # statsmodels
    kw = dict(error=error, trend=trend, damped_trend=damped, seasonal=seasonal,
              initialization_method="known", initial_level=l0)
    if seasonal:
        kw["seasonal_periods"] = m
        kw["initial_seasonal"] = np.array(s0, dtype=float)
    if trend:
        kw["initial_trend"] = b0
    mod = ETSModel(pd.Series(y), **kw)  # a pandas index: get_prediction needs one
    params = sm_params(short, alpha, beta, gamma, phi)
    res = mod.smooth(params)
    sim_start = res.simulate(H, anchor="start", repetitions=N_PATHS, random_errors=errors.T)
    sim_start = np.asarray(sim_start).T  # (paths, H)
    case["statsmodels_simulate_start"] = sim_start.tolist()
    if seasonal != "mul":
        # statsmodels is exact here and pins the paths itself; the
        # transcription's per-period paths would duplicate them (they are
        # asserted equal below), so only its scalars, final state, forecast
        # and simulated paths are stored — the fixture stays small.
        for key in ("fitted", "resid", "level", "trend", "seasonal"):
            case["transcription"][key] = None
    # The innovations-form simulator agrees with the transcription for every
    # model (statsmodels' simulate carries the kappa terms of the innovations
    # form) — asserted, then stored.
    assert np.allclose(sim_start, np.array(paths_start), rtol=1e-10, atol=1e-10), short
    if seasonal != "mul":
        block = {
            "loglik": float(res.llf),
            "fitted": np.asarray(res.fittedvalues).tolist(),
            "resid": np.asarray(res.resid).tolist(),
            "level": np.asarray(res.level).tolist(),
            "trend": np.asarray(res.slope).tolist() if trend else None,
            "seasonal": np.asarray(res.season).tolist() if seasonal else None,
            "forecast": np.asarray(res.forecast(H)).tolist(),
            "mse": float(res.mse),
        }
        sim_end = np.asarray(res.simulate(H, anchor="end", repetitions=N_PATHS,
                                          random_errors=errors.T)).T
        block["simulate_end"] = sim_end.tolist()
        assert abs(block["loglik"] - ll) <= 1e-9 * max(1.0, abs(ll)), (short, block["loglik"], ll)
        assert np.allclose(block["fitted"], fitted, rtol=1e-10, atol=1e-10), short
        assert np.allclose(block["forecast"], fc, rtol=1e-10, atol=1e-10), short
        assert np.allclose(sim_end, np.array(paths_end), rtol=1e-10, atol=1e-10), short
        if short in ("ANN", "AAN", "AAdN", "ANA", "AAA", "AAdA"):
            pr = res.get_prediction(start=n, end=n + H - 1)
            var = np.asarray(pr.var_pred_mean)
            hs = np.arange(1, H + 1)
            t61 = table61(short, hs, alpha, beta, gamma, phi, mm)
            gen = class1_general_variance(short, hs, alpha, beta, gamma, phi, mm)
            assert np.allclose(t61, gen, rtol=1e-12, atol=1e-12), (short, t61, gen)
            assert np.allclose(var, sigma2 * t61, rtol=1e-10), short
            block["forecast_variance"] = var.tolist()
            case["class1_variance_table61"] = (sigma2 * t61).tolist()
            case["class1_relative_variance_general"] = gen.tolist()
        case["statsmodels"] = block
    else:
        case["statsmodels"] = None
        case["statsmodels_smoother_gap"] = {
            "note": "statsmodels' Cython smoother updates a multiplicative seasonal "
                    "with the post-update level and gamma/(1-alpha) (classical "
                    "Holt-Winters), not the innovations form; measured gap vs the "
                    "transcription, recorded and not gated",
            "loglik_statsmodels": float(res.llf),
            "loglik_transcription": ll,
            "max_abs_fitted_gap": float(np.max(np.abs(np.asarray(res.fittedvalues) - fitted))),
        }
    return case


def fixed_block(series, rng):
    cases = []
    y_ns, _ = series["sim_aadn"]
    for short in ["ANN", "AAN", "AAdN", "AMN", "AMdN", "MNN", "MAN", "MAdN", "MMN", "MMdN"]:
        _, trend, _, _ = parse(short)
        b0 = 1.005 if trend == "mul" else 0.5
        cases.append(fixed_case(f"{short}__sim_aadn", short, "sim_aadn", y_ns, 1,
                                0.3, 0.1, 0.0, 0.9, 100.0, b0, None, rng))
    y_co2, m12 = series["co2_monthly"]
    s12 = np.array([-0.1, 0.5, 1.0, 1.9, 2.3, 1.6, 0.2, -1.7, -3.2, -3.1, -1.9, 0.5])
    s12 = (s12 - s12.mean()).tolist()
    for short in ["ANA", "AAA", "AAdA", "AMA", "AMdA", "MNA", "MAA", "MAdA", "MMA", "MMdA"]:
        _, trend, _, _ = parse(short)
        b0 = 1.0003 if trend == "mul" else 0.08
        cases.append(fixed_case(f"{short}__co2_monthly", short, "co2_monthly", y_co2, m12,
                                0.3, 0.05, 0.1, 0.95, 315.0, b0, s12, rng))
    y_uk, m4 = series["log_ukgas"]
    s4 = np.array([0.15, -0.05, -0.25, 0.15])
    s4 = (s4 - s4.mean()).tolist()
    for short in ["ANA", "AAA", "AAdA"]:
        cases.append(fixed_case(f"{short}__log_ukgas", short, "log_ukgas", y_uk, m4,
                                0.4, 0.05, 0.2, 0.9, 5.0, 0.01, s4, rng))
    y_air, _ = series["airline"]
    s_air = np.array([0.9, 0.85, 1.0, 1.05, 1.02, 1.1, 1.2, 1.25, 1.1, 0.95, 0.85, 0.9])
    s_air = (s_air / s_air.mean()).tolist()
    for short in ["ANM", "AAM", "AAdM", "AMM", "AMdM", "MNM", "MAM", "MAdM", "MMM", "MMdM"]:
        _, trend, _, _ = parse(short)
        b0 = 1.012 if trend == "mul" else 1.5
        cases.append(fixed_case(f"{short}__airline", short, "airline", y_air, 12,
                                0.3, 0.05, 0.2, 0.95, 120.0, b0, s_air, rng))
    y_m4, _ = series["sim_mam4"]
    s_m4 = [0.9, 1.1, 1.05, 0.95]
    for short in ["MAM", "MAdM", "MNM"]:
        cases.append(fixed_case(f"{short}__sim_mam4", short, "sim_mam4", y_m4, 4,
                                0.3, 0.05, 0.1, 0.9, 100.0, 0.5, s_m4, rng))
    return cases


def heuristic_block(series):
    out = []
    combos = [(t, s) for t in (None, "add", "mul") for s in (None, "add", "mul")]
    for sname in ["sim_aadn", "co2_monthly", "airline", "log_ukgas"]:
        y, m = series[sname]
        for trend, seasonal in combos:
            if seasonal and m == 1:
                continue
            kw = dict(trend=trend, seasonal=seasonal, initialization_method="heuristic")
            if seasonal:
                kw["seasonal_periods"] = m
            hw = ExponentialSmoothing(y, **kw)
            # optimized=False runs the recursion at fixed smoothing values and
            # reports the (heuristic, not estimated) initial states.
            fit_kw = dict(smoothing_level=0.5, optimized=False)
            if trend:
                fit_kw["smoothing_trend"] = 0.1
            if seasonal:
                fit_kw["smoothing_seasonal"] = 0.1
            p = hw.fit(**fit_kw).params
            lvl, tr, seas = _initialization_heuristic(y, trend=trend, seasonal=seasonal,
                                                      seasonal_periods=m if seasonal else None)
            ets_kw = dict(error="add", trend=trend, seasonal=seasonal, initialization_method="heuristic")
            if seasonal:
                ets_kw["seasonal_periods"] = m
            em = ETSModel(y, **ets_kw)
            assert abs(p["initial_level"] - lvl) < 1e-10 and abs(em.initial_level - lvl) < 1e-10
            if trend:
                assert abs(p["initial_trend"] - tr) < 1e-10 and abs(em.initial_trend - tr) < 1e-10
            if seasonal:
                assert np.allclose(p["initial_seasons"], seas, atol=1e-10)
                assert np.allclose(em.initial_seasonal, seas, atol=1e-10)
            out.append({
                "series": sname, "trend": trend, "seasonal": seasonal,
                "seasonal_periods": m if seasonal else None,
                "initial_level": float(p["initial_level"]),
                "initial_trend": float(p["initial_trend"]) if trend else None,
                "initial_seasonal": np.asarray(p["initial_seasons"], dtype=float).tolist() if seasonal else None,
                "method": "heuristic",
            })
    # the simple rule on a sample too short for the heuristic
    y, m = series["sim_mam4"]
    y_short = y[:12]
    for trend, seasonal in [(None, "add"), ("add", "mul"), ("mul", "mul"), (None, None), ("add", None), ("mul", None)]:
        lvl, tr, seas = _initialization_simple(y_short, trend=trend, seasonal=seasonal,
                                               seasonal_periods=4 if seasonal else None)
        out.append({
            "series": "sim_mam4[:12]", "trend": trend, "seasonal": seasonal,
            "seasonal_periods": 4 if seasonal else None,
            "initial_level": float(lvl),
            "initial_trend": float(tr) if trend else None,
            "initial_seasonal": np.asarray(seas, dtype=float).tolist() if seasonal else None,
            "method": "simple",
        })
    return out


def convert_normalisation(seasonal, trend, level, trend0, seas):
    """statsmodels' identification (index for observation 0 pinned at 0/1)
    to the crate's (sum zero / average one); fitted values unchanged."""
    if seas is None:
        return level, trend0, None
    s = np.asarray(seas, dtype=float)
    c = float(s.mean())
    if seasonal == "mul":
        return level * c, (trend0 * c if trend == "add" else trend0), (s / c).tolist()
    return level + c, trend0, (s - c).tolist()


def mle_block(series):
    plan = [
        ("sim_aadn", ["ANN", "AAN", "AAdN", "MNN", "MAN", "MAdN", "AMN", "MMN", "MMdN"], "estimated"),
        ("log_ukgas", ["ANA", "AAA", "AAdA"], "estimated"),
        ("log_ukgas", ["AAA", "ANA"], "heuristic"),
        ("co2_monthly", ["AAA", "AAdA"], "estimated"),
        ("airline", ["MNA", "MAA", "MAdA"], "estimated"),
        ("sim_aadn", ["AAdN", "MAN"], "heuristic"),
    ]
    out = []
    for sname, shorts, init in plan:
        y, m = series[sname]
        for short in shorts:
            error, trend, damped, seasonal = parse(short)
            kw = dict(error=error, trend=trend, damped_trend=damped, seasonal=seasonal,
                      initialization_method=init)
            if seasonal:
                kw["seasonal_periods"] = m
            mod = ETSModel(y, **kw)
            res = mod.fit(disp=False, maxiter=5000)
            names = list(mod.param_names)
            p = dict(zip(names, np.asarray(res.params, dtype=float).tolist()))
            seas = np.asarray(res.initial_seasonal, dtype=float).tolist() if seasonal else None
            lvl = float(res.initial_level)
            tr = float(res.initial_trend) if trend else None
            c_lvl, c_tr, c_seas = convert_normalisation(seasonal, trend, lvl, tr, seas)
            # the conversion leaves the likelihood unchanged: re-evaluate
            if init == "estimated":
                chk_kw = dict(kw, initialization_method="known", initial_level=c_lvl)
                if trend:
                    chk_kw["initial_trend"] = c_tr
                if seasonal:
                    chk_kw["initial_seasonal"] = np.array(c_seas)
                chk = ETSModel(y, **chk_kw).smooth(
                    sm_params(short, p["smoothing_level"], p.get("smoothing_trend", 0.0),
                              p.get("smoothing_seasonal", 0.0), p.get("damping_trend", 1.0)))
                assert abs(chk.llf - res.llf) < 1e-8 * max(1.0, abs(res.llf)), (short, chk.llf, res.llf)
            k_sm = int(res.df_model)
            k_crate = (1 + int(bool(trend)) + int(bool(seasonal)) + int(damped)
                       + (1 + int(bool(trend)) + (m - 1 if seasonal else 0) if init == "estimated" else 0)
                       + 1)
            out.append({
                "series": sname, "short_name": short, "initialization": init,
                "error": error, "trend": trend, "damped": damped, "seasonal": seasonal,
                "seasonal_periods": m if seasonal else None,
                "alpha": p["smoothing_level"],
                "beta": p.get("smoothing_trend"),
                "gamma": p.get("smoothing_seasonal"),
                "phi": p.get("damping_trend"),
                "initial_level": lvl, "initial_trend": tr, "initial_seasonal": seas,
                "converted_initial_level": c_lvl, "converted_initial_trend": c_tr,
                "converted_initial_seasonal": c_seas,
                "loglik": float(res.llf), "sigma2": float(res.mse),
                "aic": float(res.aic), "aicc": float(res.aicc), "bic": float(res.bic),
                "k_params_statsmodels": k_sm, "k_params_crate": k_crate,
                "converged": bool(res.mle_retvals.get("converged", True)),
                "forecast": np.asarray(res.forecast(H)).tolist(),
                "nobs": int(len(y)),
            })
    return out


def r_candidates(seasonal_periods, data_positive, allow_mult_trend, restrict, damped):
    """The candidate loop of forecast::ets (ets.R), enumerated."""
    errors = ["A", "M"]
    trends = ["N", "A", "M"] if allow_mult_trend else ["N", "A"]
    seasonals = ["N", "A", "M"] if (seasonal_periods or 1) >= 2 else ["N"]
    dampeds = [True, False] if damped is None else [damped]
    out = []
    for e in errors:
        for t in trends:
            for s in seasonals:
                for d in dampeds:
                    if t == "N" and d:
                        continue
                    if restrict:
                        if e == "A" and (t == "M" or s == "M"):
                            continue
                        if e == "M" and t == "M" and s == "A":
                            continue
                    if not data_positive and (e == "M" or t == "M" or s == "M"):
                        continue
                    out.append(e + t + ("d" if d else "") + s)
    return out


def candidates_block():
    out = []
    for m in (None, 4):
        for pos in (True, False):
            for amt in (False, True):
                for restrict in (True, False):
                    for damped in (None, True, False):
                        out.append({
                            "seasonal_periods": m, "data_positive": pos,
                            "allow_multiplicative_trend": amt, "restrict": restrict,
                            "damped": damped,
                            "candidates": r_candidates(m, pos, amt, restrict, damped),
                        })
    return out


def main():
    series, co2_missing = load_series()
    rng = np.random.default_rng(7)
    fixed = fixed_block(series, rng)
    heuristic = heuristic_block(series)
    mle = mle_block(series)
    cands = candidates_block()
    fx = {
        "_meta": {
            "generator": "fixtures/generate_ets_fixtures.py",
            "statsmodels": statsmodels.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pandas": pd.__version__,
            "python": platform.python_version(),
            "co2_missing_months_interpolated": co2_missing,
            "h": H,
            "n_paths": N_PATHS,
            "seasonal_convention": "initial_seasonal[j] is the seasonal state in force for "
                                   "observation j; final_seasonal[j] the state for forecast "
                                   "step j",
            "sim_dgps": {
                "sim_aadn": "ETS(A,Ad,N): l0 100, b0 0.5, alpha 0.4, beta 0.1, phi 0.9, sigma 2, "
                            "n 200, default_rng(20260911)",
                "sim_mam4": "ETS(M,A,M) m 4: l0 100, b0 0.5, s0 (0.9, 1.1, 1.05, 0.95), alpha 0.3, "
                            "beta 0.05, gamma 0.1, sigma 0.04, n 160, same stream",
            },
        },
        "series": {k: v[0].tolist() for k, v in series.items()},
        "periods": {k: v[1] for k, v in series.items()},
        "fixed": fixed,
        "heuristic": heuristic,
        "mle": mle,
        "candidates": cands,
    }
    OUT.write_text(json.dumps(fx, indent=1) + "\n")
    n_exact = sum(1 for c in fixed if c["statsmodels"] is not None)
    print(f"wrote {OUT}: {len(fixed)} fixed cases ({n_exact} statsmodels-exact), "
          f"{len(heuristic)} heuristic, {len(mle)} mle, {len(cands)} candidate sets; "
          f"co2 months interpolated: {co2_missing}")
    for c in fixed:
        if c["statsmodels"] is None:
            g = c["statsmodels_smoother_gap"]
            print(f"  {c['name']}: statsmodels smoother gap loglik "
                  f"{g['loglik_statsmodels']:.6f} vs {g['loglik_transcription']:.6f}, "
                  f"max|fitted gap| {g['max_abs_fitted_gap']:.4g}")
    for c in mle:
        print(f"  mle {c['series']} {c['short_name']} ({c['initialization']}): llf {c['loglik']:.6f} "
              f"alpha {c['alpha']:.5f} converged {c['converged']}")


if __name__ == "__main__":
    main()
