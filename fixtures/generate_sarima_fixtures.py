"""Seasonal ARIMA (SARIMA) goldens from statsmodels SARIMAX.

An INDEPENDENT reference: nothing here imports tsecon. Every number is
produced by statsmodels' own state-space code.

    .venv/bin/python fixtures/generate_sarima_fixtures.py

What is pinned and why
----------------------
Fixed-parameter exact log-likelihoods under
``simple_differencing=True`` (both ``d`` and ``D`` are differenced out
up front, matching the crate's convention -- ``d + D*s`` observations
are lost), full MLE fits as match-or-beat log-likelihood targets with
the fitted parameters recorded for parity gates, ``cov_type='approx'``
standard errors at the recorded parameters, and levels forecasts.

The forecast leg uses ``simple_differencing=False`` (the levels
state-space form) *at the same ARMA parameters*: with simple
differencing statsmodels forecasts the differenced series only, while
tsecon re-cumulates to levels. For an invertible model the two levels
forecasts agree up to initialization effects that decay geometrically
in the sample size -- the recorded ``forecast_agreement_note`` states
the measured gap between the two statsmodels conventions is what the
comparison tolerance must absorb.

The airline passenger series (Box & Jenkins 1976, Series G: monthly
totals of international airline passengers, 1949-1960, in thousands) is
embedded verbatim; it is the canonical SARIMA validation target.
"""

import json
import platform
from pathlib import Path

import numpy as np
import scipy
import statsmodels
import statsmodels.api as sm
from statsmodels.tools import numdiff


def polished_fit(model):
    """Fit, then Nelder-Mead polish toward a genuine stationary point.

    statsmodels' default L-BFGS stopping point is routinely *not* a
    stationary point (this repo's Nile ARMA(1,1) golden documents a live
    example, and the first cut of this fixture had |grad| up to 0.96 at
    the 'fitted' parameters). Standard errors are curvature at the
    reported point, and at a non-stationary point they become
    step-rule-sensitive at the 5e-3 level. The polish shrinks that
    sensitivity; what removes the confound entirely is ``bse_hess3``
    below -- a real four-point finite-difference Hessian, the same
    *kind* of estimator tsecon computes, recorded next to the canonical
    complex-step ``bse_approx``.
    """
    res = model.fit(disp=False, cov_type="approx")
    res = model.fit(
        start_params=res.params,
        method="nm",
        maxiter=20000,
        disp=False,
        cov_type="approx",
        xtol=1e-12,
        ftol=1e-12,
    )
    # Parameter-scaled score, informational only (a raw gradient is
    # meaningless across parameters whose scales differ by orders of
    # magnitude -- d loglik / d sigma2 is huge in raw units when sigma2
    # is small).
    grad = numdiff.approx_fprime_cs(res.params, model.loglike)
    scaled = float(np.max(np.abs(grad) * np.maximum(np.abs(res.params), 1e-8)))
    return res, scaled


def bse_hess3(model, params):
    """sqrt diag inv(-H) from statsmodels' real four-point Hessian."""
    h = numdiff.approx_hess3(np.asarray(params), model.loglike)
    return np.sqrt(np.diag(np.linalg.inv(-h)))

OUT = Path(__file__).resolve().parent

META = {
    "statsmodels": statsmodels.__version__,
    "scipy": scipy.__version__,
    "numpy": np.__version__,
    "python": platform.python_version(),
}

# Box & Jenkins (1976) Series G -- monthly international airline
# passengers (thousands), January 1949 to December 1960. Public data;
# the same numbers ship with R as ``AirPassengers``.
AIRLINE = [
    112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118,
    115, 126, 141, 135, 125, 149, 170, 170, 158, 133, 114, 140,
    145, 150, 178, 163, 172, 178, 199, 199, 184, 162, 146, 166,
    171, 180, 193, 181, 183, 218, 230, 242, 209, 191, 172, 194,
    196, 196, 236, 235, 229, 243, 264, 272, 237, 211, 180, 201,
    204, 188, 235, 227, 234, 264, 302, 293, 259, 229, 203, 229,
    242, 233, 267, 269, 270, 315, 364, 347, 312, 274, 237, 278,
    284, 277, 317, 313, 318, 374, 413, 405, 355, 306, 271, 306,
    315, 301, 356, 348, 355, 422, 465, 467, 404, 347, 305, 336,
    340, 318, 362, 348, 363, 435, 491, 505, 404, 359, 310, 337,
    360, 342, 406, 396, 420, 472, 548, 559, 463, 407, 362, 405,
    417, 391, 419, 461, 472, 535, 622, 606, 508, 461, 390, 432,
]


def airline_case():
    """ARIMA(0,1,1)(0,1,1)_12 on log airline passengers -- the airline
    model, the canonical SARIMA cross-package target."""
    y = np.log(np.asarray(AIRLINE, dtype=float))
    order, seasonal = (0, 1, 1), (0, 1, 1, 12)

    # Fixed-parameter log-likelihood: a deterministic golden at round
    # numbers inside the invertibility region.
    fixed = np.array([-0.3, -0.5, 0.0015])
    m_simple = sm.tsa.SARIMAX(
        y, order=order, seasonal_order=seasonal, trend="n", simple_differencing=True
    )
    ll_fixed = float(m_simple.loglike(fixed))

    # Full MLE fit: a match-or-beat target with parameter parity,
    # polished to a genuine stationary point.
    r, grad_norm = polished_fit(m_simple)

    # Levels forecasts at the FITTED simple-differencing parameters,
    # through the levels state-space form (exact diffuse
    # initialization). Recorded so the crate's re-cumulated forecast can
    # be compared against an independent levels implementation.
    m_levels = sm.tsa.SARIMAX(
        y, order=order, seasonal_order=seasonal, trend="n", simple_differencing=False
    )
    fc = m_levels.smooth(r.params).get_forecast(24)

    return {
        "note": "log airline passengers, ARIMA(0,1,1)(0,1,1)_12, trend='n'",
        "y_is_log_airline": True,
        "fixed_params_theta_Theta_sigma2": fixed.tolist(),
        "loglike_fixed_simple_diff": ll_fixed,
        "fit_params": r.params.tolist(),
        "fit_param_names": list(r.model.param_names),
        "fit_loglike": float(r.llf),
        "fit_aic": float(r.aic),
        "fit_bic": float(r.bic),
        "fit_bse_approx": np.asarray(r.bse).tolist(),
        "fit_bse_hess3": bse_hess3(m_simple, r.params).tolist(),
        "fit_score_scaled_max": grad_norm,
        "nobs_effective": int(r.nobs),
        "forecast_mean_24_levels_ssm": fc.predicted_mean.tolist(),
        "forecast_se_24_levels_ssm": fc.se_mean.tolist(),
        "forecast_agreement_note": (
            "predicted_mean/se_mean come from the simple_differencing=False levels "
            "state-space form smoothed at the simple-differencing fit's parameters. "
            "A simple-differencing implementation that re-cumulates with exact "
            "anchors agrees up to initialization effects decaying like the MA "
            "roots to the n-th power; compare with a tolerance, not bit-exactly."
        ),
    }


def quarterly_sar_case(rng):
    """SARMA(1,0,0)(1,0,0)_4 with a constant on a simulated quarterly
    series: fixed-parameter loglike, MLE fit, and approx bse."""
    n, s = 240, 4
    phi, cap_phi, const, sigma = 0.5, 0.35, 0.8, 1.0
    burn = 200
    e = rng.standard_normal(n + burn) * sigma
    # Multiplicative SAR: (1 - phi L)(1 - Phi L^4) y = const + e.
    ar_full = np.zeros(1 + s)
    ar_full[0] = phi
    ar_full[s - 1] += cap_phi
    ar_full[s] -= phi * cap_phi
    y = np.zeros(n + burn)
    for t in range(n + burn):
        acc = const + e[t]
        for i, a in enumerate(ar_full):
            if t > i:
                acc += a * y[t - 1 - i]
        y[t] = acc
    y = y[burn:]

    order, seasonal = (1, 0, 0), (1, 0, 0, 4)
    fixed = np.array([0.8, 0.5, 0.35, 1.0])  # [const, ar.L1, ar.S.L4, sigma2]
    m = sm.tsa.SARIMAX(
        y, order=order, seasonal_order=seasonal, trend="c", simple_differencing=True
    )
    ll_fixed = float(m.loglike(fixed))
    r, grad_norm = polished_fit(m)
    fc = m.smooth(r.params).get_forecast(12)

    return {
        "note": "simulated multiplicative SAR(1)x(1)_4 with constant, trend='c'",
        "y": y.tolist(),
        "true_const_phi_Phi_sigma2": [const, phi, cap_phi, sigma**2],
        "fixed_params_const_phi_Phi_sigma2": fixed.tolist(),
        "loglike_fixed": ll_fixed,
        "fit_params": r.params.tolist(),
        "fit_param_names": list(r.model.param_names),
        "fit_loglike": float(r.llf),
        "fit_bse_approx": np.asarray(r.bse).tolist(),
        "fit_bse_hess3": bse_hess3(m, r.params).tolist(),
        "fit_score_scaled_max": grad_norm,
        "nobs_effective": int(r.nobs),
        "forecast_mean_12": fc.predicted_mean.tolist(),
        "forecast_se_12": fc.se_mean.tolist(),
    }


def mixed_sarima_case(rng):
    """SARIMA(1,1,1)(1,1,1)_4: the full mixed specification at fixed
    parameters (loglike only -- a pure differencing + expansion gate)."""
    n, s = 200, 4
    burn = 150
    e = rng.standard_normal(n + burn + s + 1)
    x = np.zeros(n + burn + s + 1)
    for t in range(1, len(x)):
        x[t] = 0.4 * x[t - 1] + e[t] + 0.3 * e[t - 1]
    w = np.cumsum(x)  # one regular integration
    y = w.copy()      # one seasonal integration
    for t in range(s, len(y)):
        y[t] += y[t - s]
    y = y[burn:burn + n]

    fixed = np.array([0.4, 0.3, 0.2, -0.3, 1.0])
    m = sm.tsa.SARIMAX(
        y,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, s),
        trend="n",
        simple_differencing=True,
    )
    ll_fixed = float(m.loglike(fixed))
    return {
        "note": "SARIMA(1,1,1)(1,1,1)_4 at fixed params on a simulated integrated series",
        "y": y.tolist(),
        "fixed_params_phi_theta_Phi_Theta_sigma2": fixed.tolist(),
        "loglike_fixed": ll_fixed,
        "nobs_effective": int(m.nobs),
    }


def main():
    rng = np.random.default_rng(20260813)
    blob = {
        "_meta": META,
        "_about": (
            "statsmodels SARIMAX seasonal goldens under simple_differencing=True "
            "(fixed-parameter log-likelihoods, MLE fits with cov_type='approx' "
            "standard errors) plus levels forecasts from the "
            "simple_differencing=False state-space form at the same parameters."
        ),
        "airline_011_011_12": airline_case(),
        "quarterly_sar_c": quarterly_sar_case(rng),
        "mixed_111_111_4": mixed_sarima_case(rng),
    }
    (OUT / "sarima.json").write_text(json.dumps(blob, indent=1) + "\n")
    print("wrote sarima.json")
    for name in ("airline_011_011_12", "quarterly_sar_c", "mixed_111_111_4"):
        case = blob[name]
        ll = case.get("loglike_fixed", case.get("loglike_fixed_simple_diff"))
        print(f"  {name:22s} loglike_fixed={ll:.6f}")


if __name__ == "__main__":
    main()
