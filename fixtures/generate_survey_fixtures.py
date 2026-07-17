"""Golden fixtures for tsecon-survey: the survey-expectations toolkit.

Roadmap E6. Three estimators, all OLS with HAC / Newey-West inference plus
two purely-numerical dispersion measures:

  1. The Coibion-Gorodnichenko (2015, AER 105:2644-2678) information-rigidity
     regression of the mean forecast ERROR on the mean forecast REVISION.
  2. Forecast DISAGREEMENT: cross-sectional dispersion (std / IQR / quartiles)
     of individual forecasts over time.
  3. A forecast-EFFICIENCY / rationality test (Mincer-Zarnowitz in
     error-on-forecast form): regress the forecast error on the forecast and
     jointly test that ALL coefficients are zero, by a HAC Wald test.

VALIDATION STRATEGY (this is an INDEPENDENT-reference golden).
Everything here is OLS with a Bartlett/Newey-West HAC covariance, so the CG
regression and the Mincer-Zarnowitz Wald test are checked against
*statsmodels* `OLS(...).fit(cov_type="HAC", cov_kwds={"maxlags": L,
"use_correction": ...})` — a fully independent implementation of the same
estimand (target (a): independent reference). The disagreement measures are
checked against *numpy* `np.std(ddof=0)` and `np.percentile(method="linear")`
(target (a) again). The two derived scalars that statsmodels does not report
directly are DOCUMENTED CLOSED FORMS written out below (target (b)):

    implied_rigidity  = slope / (1 + slope)          [CG 2015, sticky-info /
        noisy-info map: slope beta = lambda/(1-lambda) => lambda = beta/(1+beta)]
    IQR               = percentile(75) - percentile(25)

No call to the Rust crate is made here, so the check is non-circular.

Run with the project venv:
    .venv/bin/python fixtures/generate_survey_fixtures.py
"""

import json

import numpy as np
import statsmodels.api as sm
from scipy import stats


def hac_ols(y, xcols, maxlags, use_correction):
    """statsmodels OLS with a Bartlett/Newey-West HAC covariance.

    `xcols` are the design columns WITHOUT the constant; a constant is
    prepended (statsmodels `add_constant`, so param[0] is the intercept).
    `use_t=False` => normal-based p-values (the HAC default in statsmodels).
    Returns a dict of the reported quantities plus the fitted results object.
    """
    y = np.asarray(y, float)
    X = np.column_stack([np.ones_like(y)] + [np.asarray(c, float) for c in xcols])
    model = sm.OLS(y, X)
    res = model.fit(
        cov_type="HAC",
        cov_kwds={"maxlags": maxlags, "use_correction": use_correction, "kernel": "bartlett"},
        use_t=False,
    )
    return res


def cg_regression(errors, revisions, maxlags, use_correction):
    """CG (2015) regression: error_t = c + beta * revision_t + u_t (OLS-HAC)."""
    res = hac_ols(errors, [revisions], maxlags, use_correction)
    intercept, slope = res.params
    # Documented closed form for the implied degree of information rigidity:
    # under sticky information beta = lambda/(1-lambda) so lambda = beta/(1+beta);
    # the noisy-information (Kalman-gain) map gives the same beta/(1+beta).
    implied_rigidity = slope / (1.0 + slope)
    return {
        "errors": list(map(float, errors)),
        "revisions": list(map(float, revisions)),
        "maxlags": int(maxlags),
        "use_correction": bool(use_correction),
        "intercept": float(intercept),
        "slope": float(slope),
        "se_intercept": float(res.bse[0]),
        "se_slope": float(res.bse[1]),
        "t_intercept": float(res.tvalues[0]),
        "t_slope": float(res.tvalues[1]),
        "p_intercept": float(res.pvalues[0]),
        "p_slope": float(res.pvalues[1]),
        "r_squared": float(res.rsquared),
        "implied_rigidity": float(implied_rigidity),
        "nobs": int(res.nobs),
        "use_t": bool(res.use_t),
    }


def efficiency_test(errors, regressors, maxlags, use_correction):
    """Mincer-Zarnowitz efficiency test in error-on-forecast form.

    Regress error_t on a constant and the `regressors` (e.g. the forecast),
    then jointly test H0: ALL coefficients (intercept + slopes) are zero via a
    HAC Wald test.  `res.wald_test(I_k, use_f=False)` returns the chi-square(k)
    statistic W = b' V^{-1} b and its p-value, with V the HAC covariance.
    """
    res = hac_ols(errors, regressors, maxlags, use_correction)
    k = len(res.params)
    wt = res.wald_test(np.eye(k), use_f=False, scalar=True)
    wald = float(np.ravel(wt.statistic)[0])
    wald_pvalue = float(wt.pvalue)
    return {
        "errors": list(map(float, errors)),
        "regressors": [list(map(float, c)) for c in regressors],
        "maxlags": int(maxlags),
        "use_correction": bool(use_correction),
        "params": list(map(float, res.params)),
        "bse": list(map(float, res.bse)),
        "tvalues": list(map(float, res.tvalues)),
        "pvalues": list(map(float, res.pvalues)),
        "r_squared": float(res.rsquared),
        "wald": wald,
        "wald_df": int(k),
        "wald_pvalue": wald_pvalue,
        "nobs": int(res.nobs),
    }


def disagreement(panel, ddof):
    """Cross-sectional dispersion of individual forecasts, period by period.

    `panel` is a list of per-period cross-sections (ragged allowed).  For each
    period compute numpy population/sample std (given ddof), the 25/50/75
    percentiles (numpy default linear interpolation), and IQR = p75 - p25.
    """
    std, p25, p50, p75, iqr, counts = [], [], [], [], [], []
    for row in panel:
        a = np.asarray(row, float)
        std.append(float(np.std(a, ddof=ddof)))
        q25, q50, q75 = np.percentile(a, [25.0, 50.0, 75.0])  # method='linear'
        p25.append(float(q25))
        p50.append(float(q50))
        p75.append(float(q75))
        iqr.append(float(q75 - q25))  # documented: IQR = P75 - P25
        counts.append(int(a.size))
    return {
        "panel": [list(map(float, row)) for row in panel],
        "ddof": int(ddof),
        "std": std,
        "p25": p25,
        "p50": p50,
        "p75": p75,
        "iqr": iqr,
        "counts": counts,
    }


def build_cg_series(mean_forecast, actual, h):
    """Documented-formula construction of the CG error/revision series.

    Fixed-horizon setup: mean_forecast[t] is the h-step-ahead forecast made at
    time t of the outcome actual[t+h].  For each usable t (with t-1 >= 0 and
    t+h <= n-1):
        error_t    = actual[t + h] - mean_forecast[t]
        revision_t = mean_forecast[t] - mean_forecast[t - 1]
    Aligned over t = 1 .. n-1-h (inclusive).
    """
    f = np.asarray(mean_forecast, float)
    y = np.asarray(actual, float)
    n = f.size
    errors, revisions = [], []
    for t in range(1, n - h):
        errors.append(float(y[t + h] - f[t]))
        revisions.append(float(f[t] - f[t - 1]))
    return {
        "mean_forecast": list(map(float, f)),
        "actual": list(map(float, y)),
        "h": int(h),
        "errors": errors,
        "revisions": revisions,
    }


def main():
    rng = np.random.default_rng(20260717)

    # ------------------------------------------------------------------
    # (1) CG regression: mean forecast error on mean forecast revision.
    # Simulate a positively-autocorrelated error and a revision that carries
    # information rigidity (error loads positively on the revision), so the HAC
    # correction genuinely differs from OLS.
    # ------------------------------------------------------------------
    n = 180
    rev = np.zeros(n)
    for t in range(1, n):
        rev[t] = 0.5 * rev[t - 1] + rng.normal(0.0, 1.0)
    u = np.zeros(n)
    for t in range(1, n):
        u[t] = 0.6 * u[t - 1] + rng.normal(0.0, 0.8)
    beta_true, c_true = 0.7, 0.1
    err = c_true + beta_true * rev + u
    L_cg = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))  # NW rule of thumb
    cg = cg_regression(err, rev, maxlags=L_cg, use_correction=True)

    # A second CG case with a non-default (larger) maxlags and no small-sample
    # correction, to exercise both flags.
    cg_alt = cg_regression(err, rev, maxlags=8, use_correction=False)

    # ------------------------------------------------------------------
    # (1b) Documented-formula construction of the error/revision series from a
    # fixed-horizon mean-forecast panel + realized actual.
    # ------------------------------------------------------------------
    m = 60
    actual = np.cumsum(rng.normal(0.05, 1.0, m))
    mean_fc = actual + rng.normal(0.0, 0.5, m)  # noisy forecasts of the level
    cg_build = build_cg_series(mean_fc, actual, h=2)

    # ------------------------------------------------------------------
    # (2) Efficiency / Mincer-Zarnowitz: error on the forecast; joint HAC Wald
    # that intercept and slope are both zero.
    # ------------------------------------------------------------------
    ne = 200
    forecast = np.cumsum(rng.normal(0.0, 1.0, ne)) * 0.3 + rng.normal(0.0, 1.0, ne)
    # A rational-ish error: mostly noise, mild bias + tiny predictability.
    e_err = np.zeros(ne)
    for t in range(1, ne):
        e_err[t] = 0.4 * e_err[t - 1] + rng.normal(0.0, 1.0)
    fe = 0.15 + 0.05 * forecast + e_err
    L_eff = int(np.floor(4.0 * (ne / 100.0) ** (2.0 / 9.0)))
    eff = efficiency_test(fe, [forecast], maxlags=L_eff, use_correction=True)

    # A multi-regressor efficiency test: error on forecast AND a lagged signal.
    lag_signal = np.roll(forecast, 1)
    lag_signal[0] = 0.0
    eff_multi = efficiency_test(fe, [forecast, lag_signal], maxlags=6, use_correction=True)

    # ------------------------------------------------------------------
    # (3) Disagreement: a forecaster panel.  Balanced main panel + a ragged
    # panel to exercise variable cross-section sizes.
    # ------------------------------------------------------------------
    n_periods, n_fc = 40, 25
    common = np.cumsum(rng.normal(0.0, 1.0, n_periods))
    spread = 0.5 + 0.4 * np.abs(np.sin(np.arange(n_periods) / 3.0))
    balanced = [
        list(common[t] + spread[t] * rng.normal(0.0, 1.0, n_fc)) for t in range(n_periods)
    ]
    disag_pop = disagreement(balanced, ddof=0)   # numpy default population std
    disag_sample = disagreement(balanced, ddof=1)  # sample std

    ragged = [
        list(rng.normal(0.0, 1.0, k)) for k in [3, 5, 4, 7, 2, 6, 9, 4]
    ]
    disag_ragged = disagreement(ragged, ddof=0)

    fixtures = {
        "_meta": {
            "description": "tsecon-survey golden fixtures (roadmap E6)",
            "reference": "statsmodels OLS cov_type=HAC (bartlett) + numpy std/percentile",
            "statsmodels_use_t": False,
            "note": "CG and MZ regressions match statsmodels HAC; disagreement "
            "matches numpy; implied_rigidity=slope/(1+slope) and IQR=P75-P25 "
            "are documented closed forms.",
        },
        "cg": cg,
        "cg_alt": cg_alt,
        "cg_build": cg_build,
        "efficiency": eff,
        "efficiency_multi": eff_multi,
        "disagreement_pop": disag_pop,
        "disagreement_sample": disag_sample,
        "disagreement_ragged": disag_ragged,
    }

    out = "fixtures/tsecon-survey.json"
    with open(out, "w") as fh:
        json.dump(fixtures, fh, indent=2)
    print(f"wrote {out}")
    # Small console echo for a sanity glance.
    print("CG slope:", cg["slope"], "se:", cg["se_slope"], "rigidity:", cg["implied_rigidity"])
    print("MZ wald:", eff["wald"], "df:", eff["wald_df"], "p:", eff["wald_pvalue"])
    print("use_t(HAC):", cg["use_t"])


if __name__ == "__main__":
    main()
