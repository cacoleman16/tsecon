"""Golden fixtures for tsecon-quantile: quantile regression, quantile local
projections, and growth-at-risk (roadmap frontier slice).

VALIDATION STRATEGY
===================
Every number this file writes is produced by an INDEPENDENT reference:
statsmodels' `QuantReg` (a completely separate code path from the tsecon Rust
crate) plus plain-numpy assembly of the LP designs and the rearrangement.
Nothing here imports tsecon, so reproducing these numbers in Rust is a genuine
cross-implementation check, never circular. All data are DERIVED from seeded
numpy Generator draws — no redistributed datasets.

ESTIMATORS AND THEIR REFERENCES
-------------------------------
1. Linear quantile regression.  REFERENCE: statsmodels
   `QuantReg(endog, exog).fit(q=tau)` with ALL defaults — the
   Schnabel-Koenker IRLS with a 1e-6 residual floor, `p_tol=1e-6`,
   `max_iter=1000`, and the `vcov="robust"` kernel sandwich
   (Powell 1991 / Greene 2008 form): Epanechnikov kernel density of the
   residuals at zero with the Hall-Sheather (1988) bandwidth mapped through
   `min(std(y), IQR(e)/1.34) * (ppf(q+h) - ppf(q-h))` as in Stata 12.
   We store `params`, `bse`, `iterations`, `bandwidth`, `sparsity` verbatim
   for several taus on three DGPs (heteroskedastic normal, Student-t(3),
   and skewed chi-square errors).

2. Quantile local projections.  REFERENCE: the SAME statsmodels `QuantReg`
   run per horizon on a design assembled with plain numpy in exactly the
   tsecon-lp column convention `[shock_t, const, y_{t-1..t-p},
   shock_{t-1..t-p}]` with outcome `y_{t+h}` over
   `t = p, ..., n - 1 - h`.  We store the impulse coefficient (column 0)
   and its robust bse for every (tau, horizon).

3. Growth-at-risk (Adrian-Boyarchenko-Giannone 2019 AER).  REFERENCE:
   statsmodels `QuantReg` of `y_{t+h}` on `[const, conditions_t, y_t]`
   per tau, fitted values `x_t' beta_tau` evaluated with numpy at EVERY
   `t = 0..n-1` (the last row is the "current risk read"), and the
   Chernozhukov-Fernandez-Val-Galichon (2010) rearrangement implemented as
   `np.sort` across the tau axis at each evaluation point.  We store raw and
   rearranged fitted paths, whether any crossing occurred, and the
   rearranged quantiles at the last observation.

Run with the project venv:
    .venv/bin/python fixtures/generate_tsecon-quantile_fixtures.py
"""

import json
import warnings

import numpy as np
import scipy
import statsmodels
from statsmodels.regression.quantile_regression import QuantReg

OUT = "fixtures/tsecon-quantile.json"


def fit_quantreg(y, X, tau):
    """statsmodels QuantReg with all defaults; returns the raw pieces."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = QuantReg(y, X).fit(q=tau)
    return {
        "tau": tau,
        "params": list(res.params),
        "bse": list(res.bse),
        "iterations": int(res.iterations),
        "bandwidth": float(res.bandwidth),
        "sparsity": float(res.sparsity),
    }


# ---------------------------------------------------------------------------
# 1. Plain quantile regression
# ---------------------------------------------------------------------------

def qreg_cases():
    cases = []

    # (a) heteroskedastic normal errors, 2 slopes + const, n = 200.
    rng = np.random.default_rng(20260721)
    n = 200
    x1 = rng.standard_normal(n)
    x2 = rng.uniform(-2.0, 2.0, n)
    scale = 0.5 + 0.4 * np.abs(x1)
    y = 1.0 + 2.0 * x1 - 0.5 * x2 + scale * rng.standard_normal(n)
    X = np.column_stack([np.ones(n), x1, x2])
    taus = [0.05, 0.25, 0.5, 0.75, 0.95]
    cases.append({
        "name": "hetero_normal",
        "y": list(y),
        "columns": [list(c) for c in X.T],
        "fits": [fit_quantreg(y, X, t) for t in taus],
    })

    # (b) heavy-tailed t(3) errors, one slope, n = 150.
    rng = np.random.default_rng(7)
    n = 150
    x1 = rng.standard_normal(n)
    y = -0.5 + 1.5 * x1 + rng.standard_t(3, n)
    X = np.column_stack([np.ones(n), x1])
    cases.append({
        "name": "student_t3",
        "y": list(y),
        "columns": [list(c) for c in X.T],
        "fits": [fit_quantreg(y, X, t) for t in [0.1, 0.5, 0.9]],
    })

    # (c) skewed chi-square(2) errors, 2 slopes, n = 120.
    rng = np.random.default_rng(99)
    n = 120
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    y = 0.3 + 0.8 * x1 + 0.2 * x2 + (rng.chisquare(2, n) - 2.0)
    X = np.column_stack([np.ones(n), x1, x2])
    cases.append({
        "name": "skew_chi2",
        "y": list(y),
        "columns": [list(c) for c in X.T],
        "fits": [fit_quantreg(y, X, t) for t in [0.25, 0.5, 0.75]],
    })
    return cases


# ---------------------------------------------------------------------------
# 2. Quantile local projections
# ---------------------------------------------------------------------------

def lp_design(y, shock, h, p):
    """tsecon-lp column convention: [shock_t, const, y lags 1..p, shock lags
    1..p], outcome y_{t+h}, sample t = p .. n-1-h."""
    n = len(y)
    start = p
    nobs = n - h - start
    t = np.arange(start, start + nobs)
    cols = [shock[t], np.ones(nobs)]
    for lag in range(1, p + 1):
        cols.append(y[t - lag])
    for lag in range(1, p + 1):
        cols.append(shock[t - lag])
    return y[t + h], np.column_stack(cols)


def qlp_case():
    rng = np.random.default_rng(314159)
    n = 240
    shock = rng.standard_normal(n)
    eps = 0.8 * rng.standard_normal(n)
    y = np.zeros(n)
    for t in range(n):
        prev = y[t - 1] if t > 0 else 0.0
        sprev = shock[t - 1] if t > 0 else 0.0
        y[t] = 0.5 * prev + 1.0 * shock[t] + 0.3 * sprev + eps[t]
    p = 2
    max_h = 4
    taus = [0.1, 0.5, 0.9]
    irf = []   # [tau][h]
    se = []
    for tau in taus:
        irf_row, se_row = [], []
        for h in range(max_h + 1):
            yy, X = lp_design(y, shock, h, p)
            f = fit_quantreg(yy, X, tau)
            irf_row.append(f["params"][0])
            se_row.append(f["bse"][0])
        irf.append(irf_row)
        se.append(se_row)
    return {
        "name": "ar1_shock",
        "y": list(y),
        "shock": list(shock),
        "n_lag_controls": p,
        "horizons": max_h,
        "taus": taus,
        "irf": irf,
        "se": se,
    }


# ---------------------------------------------------------------------------
# 3. Growth-at-risk
# ---------------------------------------------------------------------------

def gar_case(name, seed, n, horizon, taus):
    """Location-scale DGP: the condition x shifts the volatility of future
    growth (the ABG mechanism), so lower quantiles react more than the
    median by construction."""
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    y = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.8 * x[t - 1] + 0.5 * rng.standard_normal()
        scale = 0.4 * np.exp(0.4 * x[t - 1])
        y[t] = 0.2 + 0.3 * y[t - 1] - 0.4 * x[t - 1] + scale * rng.standard_normal()

    # Estimation sample: t = 0 .. n-1-h; regressors [const, x_t, y_t].
    h = horizon
    t = np.arange(0, n - h)
    X = np.column_stack([np.ones(n - h), x[t], y[t]])
    yy = y[t + h]
    Xall = np.column_stack([np.ones(n), x, y])

    params, bse, fitted_raw = [], [], []
    for tau in taus:
        f = fit_quantreg(yy, X, tau)
        params.append(f["params"])
        bse.append(f["bse"])
        fitted_raw.append(list(Xall @ np.asarray(f["params"])))
    raw = np.asarray(fitted_raw)               # (n_tau, n)
    rearranged = np.sort(raw, axis=0)          # CFG rearrangement per t
    crossing = bool(np.any(np.diff(raw, axis=0) < 0.0))
    return {
        "name": name,
        "y": list(y),
        "conditions": [list(x)],
        "horizon": h,
        "taus": list(taus),
        "params": params,
        "bse": bse,
        "fitted_raw": [list(r) for r in raw],
        "fitted_rearranged": [list(r) for r in rearranged],
        "crossing": crossing,
        "current": list(rearranged[:, -1]),
    }


def main():
    fx = {
        "_meta": {
            "generator": "fixtures/generate_tsecon-quantile_fixtures.py",
            "reference": "statsmodels QuantReg (IRLS + robust kernel sandwich, "
                         "Epanechnikov kernel, Hall-Sheather bandwidth); numpy "
                         "design assembly; np.sort rearrangement",
            "statsmodels": statsmodels.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "qreg": qreg_cases(),
        "qlp": qlp_case(),
        "gar": [
            gar_case("gar_h1", 2718, 200, 1, [0.05, 0.25, 0.5, 0.75, 0.95]),
            # Dense tau grid on a short sample: raw fits DO cross here, so
            # this case pins the rearrangement doing real work. Taus stay in
            # [0.1, 0.9]: at n-h = 76 usable observations the Hall-Sheather
            # offset pushes tau = 0.05 outside (0, 1), where statsmodels
            # silently returns NaN bse (the Rust crate raises
            # DegenerateBandwidth instead — covered in validation tests).
            gar_case(
                "gar_h4_dense",
                1618,
                80,
                4,
                [0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9],
            ),
        ],
    }
    with open(OUT, "w") as f:
        json.dump(fx, f, indent=1)
    n_gar_cross = sum(c["crossing"] for c in fx["gar"])
    print(f"wrote {OUT}; gar cases with crossing: {n_gar_cross}/2")


if __name__ == "__main__":
    main()
