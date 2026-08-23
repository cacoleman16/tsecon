"""Candidate-level goldens for auto_arima from statsmodels SARIMAX.

An INDEPENDENT reference: nothing here imports tsecon. Every number is
produced by statsmodels' own state-space code.

    .venv-wt/bin/python fixtures/generate_auto_arima_fixtures.py

What is pinned and why
----------------------
`auto_arima` is a *selection loop* over fits the crate already computes;
its published grading is Monte-Carlo order recovery, **not** R or
pmdarima parity (the two disagree with each other on real series, so
either "parity" would pin an implementation accident). What CAN be pinned
to an independent reference is the candidate level: for a handful of
(series, order) pairs this fixture records, per candidate,

- ``loglike_fixed``: statsmodels ``SARIMAX(order, trend,
  simple_differencing=True).loglike(params)`` at the recorded
  (Nelder-Mead-polished) MLE parameters — an exact-likelihood value the
  crate must reproduce at 1e-8 relative;
- ``aicc_fixed`` / ``aic_fixed`` / ``bic_fixed``: the information
  criteria implied by ``loglike_fixed`` under the shared conventions
  (``k`` counts the constant, AR, MA, seasonal AR/MA, and ``sigma2``;
  ``n`` is the post-differencing sample the likelihood is computed on;
  ``AICc = AIC + 2k(k+1)/(n-k-1)``) — the numbers the selection loop
  actually compares, pinned at the same 1e-8;
- ``loglike_fit`` / ``aicc_fit``: statsmodels' own polished-MLE optimum,
  kept as a **match-or-beat floor** for the crate's free fit (ARMA
  likelihoods are multimodal and both optimizers are local, so equality
  gates on free fits are the classic pmdarima-vs-R trap — the crate's
  Nile golden documents a live statsmodels stall).

``res.aicc`` from statsmodels is asserted (not just assumed) to equal the
manual formula before anything is stored, so the fixture cannot silently
encode a convention drift.

All series are seeded ``default_rng`` draws through known DGPs, embedded
verbatim; no external data is used.
"""

import json
import platform
from pathlib import Path

import numpy as np
import scipy
import statsmodels
import statsmodels.api as sm

OUT = Path(__file__).resolve().parent


def polished_fit(model):
    """Fit, then Nelder-Mead polish toward a genuine stationary point.

    statsmodels' default L-BFGS stopping point is routinely not a
    stationary point (the Nile ARMA(1,1) golden in this repo documents a
    live example); the polish makes the recorded parameters a meaningful
    match-or-beat floor rather than an optimizer accident.
    """
    res = model.fit(disp=False)
    res = model.fit(
        start_params=res.params,
        method="nm",
        maxiter=20000,
        disp=False,
        xtol=1e-12,
        ftol=1e-12,
    )
    return res


def simulate_arma(rng, n, ar, ma, sigma, constant=0.0, burn=500):
    """y_t = c + sum phi_i y_{t-i} + e_t + sum theta_j e_{t-j}."""
    total = n + burn
    e = sigma * rng.standard_normal(total)
    y = np.zeros(total)
    for t in range(total):
        v = constant + e[t]
        for i, phi in enumerate(ar):
            if t > i:
                v += phi * y[t - 1 - i]
        for j, th in enumerate(ma):
            if t > j:
                v += th * e[t - 1 - j]
        y[t] = v
    return y[burn:]


def case(y, order, seasonal_order, trend):
    """One (series, order) candidate: fixed-parameter and fit targets."""
    model = sm.tsa.SARIMAX(
        y,
        order=order,
        seasonal_order=seasonal_order,
        trend=trend,
        simple_differencing=True,
    )
    res = polished_fit(model)
    params = np.asarray(res.params)
    k = params.size  # constant? + ar + ma + sar + sma + sigma2
    nobs = int(res.nobs)

    ll_fixed = float(model.loglike(params))
    aic_fixed = -2.0 * ll_fixed + 2.0 * k
    bic_fixed = -2.0 * ll_fixed + k * np.log(nobs)
    denom = nobs - k - 1
    assert denom > 0, f"AICc undefined for order={order}: nobs={nobs}, k={k}"
    aicc_fixed = aic_fixed + 2.0 * k * (k + 1) / denom

    # Guard against convention drift: statsmodels' own aicc must equal
    # the manual formula at ITS fitted loglik.
    aicc_manual_fit = float(res.aic + 2.0 * k * (k + 1) / denom)
    assert abs(float(res.aicc) - aicc_manual_fit) < 1e-8, (
        f"statsmodels aicc convention drifted: {res.aicc} vs {aicc_manual_fit}"
    )

    return {
        "order": list(order),
        "seasonal_order": list(seasonal_order),
        "trend": trend,
        "k_params": k,
        "nobs": nobs,
        "params": params.tolist(),
        "param_names": list(res.param_names),
        "loglike_fixed": ll_fixed,
        "aic_fixed": aic_fixed,
        "bic_fixed": bic_fixed,
        "aicc_fixed": aicc_fixed,
        "loglike_fit": float(res.llf),
        "aicc_fit": float(res.aicc),
        "converged": bool(res.mle_retvals.get("converged", True)),
    }


def main():
    rng = np.random.default_rng(20260823)

    # --- Series 1: AR(1), phi = 0.6, n = 200. ---
    y_ar1 = simulate_arma(rng, 200, [0.6], [], 1.0)

    # --- Series 2: ARMA(1,1) around a mean, phi = 0.5, theta = 0.4. ---
    y_arma11 = simulate_arma(rng, 300, [0.5], [0.4], 1.2, constant=1.0)

    # --- Series 3: ARIMA(1,1,1) — an integrated ARMA(1,1). ---
    x = simulate_arma(rng, 301, [0.5], [-0.3], 1.0)
    y_i1 = np.cumsum(x)

    # --- Series 4: quarterly SARIMA (1,0,0)(1,0,0)_4. ---
    # Multiplied-out AR polynomial: (1 - 0.5L)(1 - 0.6L^4).
    y_sarima = simulate_arma(rng, 240, [0.5, 0.0, 0.0, 0.6, -0.30], [], 1.0)

    fixture = {
        "meta": {
            "generator": "generate_auto_arima_fixtures.py",
            "purpose": (
                "candidate-level pins for auto_arima: fixed-parameter "
                "loglik/AICc at 1e-8, polished free fits as "
                "match-or-beat floors; the selection loop itself is "
                "graded by MC order recovery, not third-party parity"
            ),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "statsmodels": statsmodels.__version__,
            "python": platform.python_version(),
        },
        "series": {
            "ar1": y_ar1.tolist(),
            "arma11c": y_arma11.tolist(),
            "arima111": y_i1.tolist(),
            "sarima": y_sarima.tolist(),
        },
        "cases": {
            # Series ar1: the true order plus two HK-search neighbors.
            "ar1__100c": None,
            "ar1__001c": None,
            "ar1__202c": None,
            # Series arma11c: truth and one neighbor.
            "arma11c__101c": None,
            "arma11c__100c": None,
            # Series arima111: truth (no constant: d = 1 drift ambiguity
            # left out deliberately) and the (0,1,1) neighbor.
            "arima111__111": None,
            "arima111__011": None,
            # Seasonal: the true (1,0,0)(1,0,0)_4 and one neighbor.
            "sarima__100_100c": None,
            "sarima__100_000c": None,
        },
    }

    fixture["cases"]["ar1__100c"] = case(y_ar1, (1, 0, 0), (0, 0, 0, 0), "c")
    fixture["cases"]["ar1__001c"] = case(y_ar1, (0, 0, 1), (0, 0, 0, 0), "c")
    fixture["cases"]["ar1__202c"] = case(y_ar1, (2, 0, 2), (0, 0, 0, 0), "c")
    fixture["cases"]["arma11c__101c"] = case(y_arma11, (1, 0, 1), (0, 0, 0, 0), "c")
    fixture["cases"]["arma11c__100c"] = case(y_arma11, (1, 0, 0), (0, 0, 0, 0), "c")
    fixture["cases"]["arima111__111"] = case(y_i1, (1, 1, 1), (0, 0, 0, 0), "n")
    fixture["cases"]["arima111__011"] = case(y_i1, (0, 1, 1), (0, 0, 0, 0), "n")
    fixture["cases"]["sarima__100_100c"] = case(
        y_sarima, (1, 0, 0), (1, 0, 0, 4), "c"
    )
    fixture["cases"]["sarima__100_000c"] = case(
        y_sarima, (1, 0, 0), (0, 0, 0, 0), "c"
    )

    out = OUT / "auto_arima.json"
    out.write_text(json.dumps(fixture, indent=1))
    print(f"wrote {out}")
    for name, c in fixture["cases"].items():
        print(
            f"  {name}: ll_fixed={c['loglike_fixed']:.6f} "
            f"aicc_fixed={c['aicc_fixed']:.6f} aicc_fit={c['aicc_fit']:.6f} "
            f"converged={c['converged']}"
        )


if __name__ == "__main__":
    main()
