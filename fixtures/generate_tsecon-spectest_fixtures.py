"""Golden fixtures for tsecon-spectest: specification / diagnostic tests (roadmap E9).

VALIDATION STRATEGY
===================
Every number this file writes is produced by an INDEPENDENT reference — either
statsmodels (a completely separate code path from the tsecon Rust crate) or a
DOCUMENTED closed-form formula evaluated with plain numpy. Nothing here calls
the tsecon Rust crate, so reproducing these numbers in Rust is a genuine
cross-implementation check, never circular.

TESTS AND THEIR REFERENCES
--------------------------
1. White (1980) heteroskedasticity test.  REFERENCE: statsmodels
   `statsmodels.stats.diagnostic.het_white(resid, exog)`, which regresses the
   squared OLS residuals on the design's columns, their squares, and all
   pairwise cross-products (the `numpy.triu_indices` product basis), and
   returns `LM = n * R^2_aux ~ chi2(m - 1)` where `m` is the number of
   auxiliary regressors INCLUDING the constant, plus the overall F-form of the
   auxiliary regression.  We store statsmodels' `(lm, lm_pvalue, fvalue,
   f_pvalue)` verbatim.

2. Breusch-Pagan (1979), Koenker (1981) studentized version.  REFERENCE:
   statsmodels `het_breuschpagan(resid, exog, robust=True)` (robust=True is the
   Koenker studentized default): regress squared residuals on the design,
   `LM = n * R^2_aux ~ chi2(m - 1)`, plus the F-form.  Stored verbatim.

3. Ramsey (1969) RESET functional-form test.  REFERENCE: statsmodels
   `linear_reset(res, power=3, test_type="fitted", use_f=True)`, i.e. refit
   `y` on `[X, yhat^2, yhat^3]` and F-test the joint significance of the two
   added power terms.  With nonrobust covariance the Wald F equals
   `F = ((SSR_r - SSR_u)/q) / (SSR_u/(n - p))`, q=2, p = k+2; we cross-check
   that identity in-file and store statsmodels' `(fvalue, pvalue, df_num,
   df_den)`.

4. Chow (1960) structural-break test at a KNOWN split.  REFERENCE: a Chow F
   assembled from statsmodels OLS residual sums of squares,
   `F = [(SSR_pooled - SSR_1 - SSR_2)/k] / [(SSR_1 + SSR_2)/(n - 2k)]
        ~ F(k, n - 2k)`,
   with the p-value from `scipy.stats.f.sf`.  SSR_pooled/SSR_1/SSR_2 are
   `sm.OLS(...).fit().ssr` on the full sample and the two sub-samples.

5. CUSUM parameter-stability test (Brown, Durbin & Evans 1975).  REFERENCE: a
   DOCUMENTED closed-form recursion evaluated with plain numpy (no statsmodels).
   The recursive residuals are, for r = k, k+1, ..., n-1 (0-indexed; the r-th
   uses observations 0..r to predict observation r):

       b_r      = argmin over the first r observations  (OLS on X[:r], y[:r])
       f_r      = 1 + x_r' (X[:r]' X[:r])^{-1} x_r
       w_r      = (y_r - x_r' b_r) / sqrt(f_r)                 (recursive residual)

   Under the null of stable coefficients the w_r are iid N(0, sigma^2), and it
   is an algebraic identity that sum_r w_r^2 = SSR_full, so

       sigma    = sqrt(SSR_full / (n - k)).

   The standardized CUSUM path is the running sum

       W_t = (1/sigma) * sum_{r=k}^{t} w_r,    t = k, ..., n-1   (length n - k),

   and the Brown-Durbin-Evans 5% significance boundary is the pair of straight
   lines through (+/- a*sqrt(n-k)) at the first CUSUM point and (+/- 3a*sqrt(n-k))
   at the last, with the DOCUMENTED constant a = 0.948 for the 5% level:

       bound_upper[i] = a * ( sqrt(n-k) + 2*(i+1)/sqrt(n-k) ),   i = 0..n-k-1
       bound_lower[i] = -bound_upper[i].

   We evaluate the recursion by refitting OLS on each expanding window (numpy
   `lstsq` / `solve`) — deliberately a different code path from the tsecon
   crate's recursive-least-squares update — and store `w`, the CUSUM path, the
   bounds, sigma, and a.

Run with the project venv:
    .venv/bin/python fixtures/generate_tsecon-spectest_fixtures.py
"""

import json

import numpy as np
import statsmodels
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan, het_white, linear_reset

OUT = "fixtures/tsecon-spectest.json"


def ar1(rng, n, rho, sd=1.0):
    """A stationary AR(1) leading-indicator column."""
    x = np.zeros(n)
    x[0] = rng.normal(scale=sd / np.sqrt(1 - rho * rho))
    for t in range(1, n):
        x[t] = rho * x[t - 1] + rng.normal(scale=sd)
    return x


def cols(X):
    """statsmodels design -> list of columns (const first), JSON-friendly."""
    return [X[:, j].tolist() for j in range(X.shape[1])]


# --------------------------------------------------------------------------- #
# White / Breusch-Pagan heteroskedasticity cases.
# --------------------------------------------------------------------------- #
def het_case(name, seed, n, hetero):
    rng = np.random.default_rng(seed)
    x1 = ar1(rng, n, 0.5)
    x2 = rng.normal(size=n)
    X = np.column_stack([np.ones(n), x1, x2])
    beta = np.array([1.0, 0.7, -0.4])
    if hetero:
        # Variance rises with x1: a designed heteroskedasticity White/BP detect.
        scale = np.exp(0.6 * x1)
    else:
        scale = np.ones(n)
    y = X @ beta + scale * rng.normal(size=n)

    res = sm.OLS(y, X).fit()
    lm_w, lmp_w, f_w, fp_w = (float(v) for v in het_white(res.resid, X))
    lm_b, lmp_b, f_b, fp_b = (float(v) for v in het_breuschpagan(res.resid, X, robust=True))

    # df bookkeeping mirrors the closed forms the Rust reproduces.
    p = X.shape[1]
    m_white = p * (p + 1) // 2  # number of triu products (incl. constant)
    return {
        "name": name,
        "n": n,
        "columns": cols(X),
        "y": y.tolist(),
        "white": {
            "statistic": lm_w,
            "df": m_white - 1,
            "pvalue": lmp_w,
            "fstat": f_w,
            "f_df_num": m_white - 1,
            "f_df_den": n - m_white,
            "f_pvalue": fp_w,
        },
        "breusch_pagan": {
            "statistic": lm_b,
            "df": p - 1,
            "pvalue": lmp_b,
            "fstat": f_b,
            "f_df_num": p - 1,
            "f_df_den": n - p,
            "f_pvalue": fp_b,
        },
    }


# --------------------------------------------------------------------------- #
# Ramsey RESET cases.
# --------------------------------------------------------------------------- #
def reset_case(name, seed, n, misspecified):
    rng = np.random.default_rng(seed)
    x1 = ar1(rng, n, 0.4)
    x2 = rng.normal(size=n)
    X = np.column_stack([np.ones(n), x1, x2])
    if misspecified:
        # A quadratic term in x1 is omitted from the fitted model: RESET rejects.
        y = 1.0 + x1 + 0.6 * x1 * x1 - 0.4 * x2 + rng.normal(size=n)
    else:
        y = 1.0 + x1 - 0.4 * x2 + rng.normal(size=n)

    res = sm.OLS(y, X).fit()
    rr = linear_reset(res, power=3, test_type="fitted", use_f=True)

    # Independent identity check: nonrobust Wald F equals the SSR-ratio F.
    yhat = res.fittedvalues
    Xa = np.column_stack([X, yhat**2, yhat**3])
    resa = sm.OLS(y, Xa).fit()
    q = 2
    p = Xa.shape[1]
    dfden = n - p
    F_manual = ((res.ssr - resa.ssr) / q) / (resa.ssr / dfden)
    assert abs(F_manual - float(rr.fvalue)) < 1e-8, (F_manual, rr.fvalue)

    return {
        "name": name,
        "n": n,
        "columns": cols(X),
        "y": y.tolist(),
        "reset": {
            "fstat": float(rr.fvalue),
            "df_num": int(rr.df_num),
            "df_den": int(rr.df_denom),
            "pvalue": float(rr.pvalue),
        },
    }


# --------------------------------------------------------------------------- #
# Chow structural-break cases.
# --------------------------------------------------------------------------- #
def chow_case(name, seed, n, split, broken):
    rng = np.random.default_rng(seed)
    x1 = ar1(rng, n, 0.5)
    x2 = rng.normal(size=n)
    X = np.column_stack([np.ones(n), x1, x2])
    beta1 = np.array([1.0, 0.6, -0.3])
    beta2 = np.array([1.0, 0.6, -0.3]) if not broken else np.array([2.2, -0.5, 0.8])
    y = np.empty(n)
    y[:split] = X[:split] @ beta1 + rng.normal(size=split)
    y[split:] = X[split:] @ beta2 + rng.normal(size=n - split)

    k = X.shape[1]
    ssr_p = float(sm.OLS(y, X).fit().ssr)
    ssr1 = float(sm.OLS(y[:split], X[:split]).fit().ssr)
    ssr2 = float(sm.OLS(y[split:], X[split:]).fit().ssr)
    df_num = k
    df_den = n - 2 * k
    F = ((ssr_p - ssr1 - ssr2) / df_num) / ((ssr1 + ssr2) / df_den)
    pvalue = float(stats.f.sf(F, df_num, df_den))

    return {
        "name": name,
        "n": n,
        "split": split,
        "columns": cols(X),
        "y": y.tolist(),
        "chow": {
            "fstat": float(F),
            "df_num": df_num,
            "df_den": df_den,
            "pvalue": pvalue,
            "ssr_pooled": ssr_p,
            "ssr1": ssr1,
            "ssr2": ssr2,
        },
    }


# --------------------------------------------------------------------------- #
# CUSUM (Brown-Durbin-Evans) cases — DOCUMENTED-formula golden, plain numpy.
# --------------------------------------------------------------------------- #
def recursive_residuals(X, y):
    """Recursive residuals w_r for r = k..n-1 by refitting OLS on each expanding
    window (the plain textbook definition; independent of any RLS update)."""
    n, k = X.shape
    w = np.empty(n - k)
    for r in range(k, n):
        Xr = X[:r]
        yr = y[:r]
        # OLS on the first r observations.
        b, *_ = np.linalg.lstsq(Xr, yr, rcond=None)
        xtx_inv = np.linalg.inv(Xr.T @ Xr)
        xr = X[r]
        f = 1.0 + xr @ xtx_inv @ xr
        w[r - k] = (y[r] - xr @ b) / np.sqrt(f)
    return w


def cusum_case(name, seed, n, broken):
    rng = np.random.default_rng(seed)
    x1 = ar1(rng, n, 0.5)
    X = np.column_stack([np.ones(n), x1])
    k = X.shape[1]
    if broken:
        beta1 = np.array([0.0, 0.5])
        beta2 = np.array([3.0, 0.5])
        y = np.empty(n)
        h = n // 2
        y[:h] = X[:h] @ beta1 + rng.normal(size=h)
        y[h:] = X[h:] @ beta2 + rng.normal(size=n - h)
    else:
        y = X @ np.array([1.0, 0.5]) + rng.normal(size=n)

    w = recursive_residuals(X, y)
    # sigma from the full-sample OLS: SSR_full/(n-k); identically == sum(w^2).
    res = sm.OLS(y, X).fit()
    sigma = float(np.sqrt(res.ssr / (n - k)))
    assert abs(float((w * w).sum()) - float(res.ssr)) < 1e-6, "sum(w^2) != SSR"

    cusum = (np.cumsum(w) / sigma).tolist()
    nk = n - k
    a = 0.948
    sq = np.sqrt(nk)
    idx = np.arange(1, nk + 1)
    bound_upper = (a * (sq + 2.0 * idx / sq)).tolist()
    bound_lower = (-a * (sq + 2.0 * idx / sq)).tolist()

    return {
        "name": name,
        "n": n,
        "columns": cols(X),
        "y": y.tolist(),
        "cusum": {
            "recursive_residuals": w.tolist(),
            "path": cusum,
            "bound_upper": bound_upper,
            "bound_lower": bound_lower,
            "sigma": sigma,
            "a": a,
        },
    }


def main():
    het = [
        het_case("homoskedastic", seed=11, n=120, hetero=False),
        het_case("heteroskedastic", seed=12, n=160, hetero=True),
    ]
    reset = [
        reset_case("correct", seed=21, n=120, misspecified=False),
        reset_case("misspecified", seed=22, n=120, misspecified=True),
    ]
    chow = [
        chow_case("stable", seed=31, n=120, split=60, broken=False),
        chow_case("break", seed=32, n=140, split=70, broken=True),
    ]
    cusum = [
        cusum_case("stable", seed=41, n=100, broken=False),
        cusum_case("break", seed=42, n=120, broken=True),
    ]

    fixture = {
        "_meta": {
            "description": "Golden fixtures for tsecon-spectest (roadmap E9).",
            "references": {
                "white": "statsmodels.stats.diagnostic.het_white",
                "breusch_pagan": "statsmodels.stats.diagnostic.het_breuschpagan (robust=True, Koenker)",
                "reset": "statsmodels.stats.diagnostic.linear_reset(power=3, use_f=True)",
                "chow": "Chow F from statsmodels OLS SSRs + scipy.stats.f.sf",
                "cusum": "documented Brown-Durbin-Evans recursion (numpy), a=0.948 (5%)",
            },
            "statsmodels": statsmodels.__version__,
        },
        "white_breusch_pagan": het,
        "reset": reset,
        "chow": chow,
        "cusum": cusum,
    }

    with open(OUT, "w") as f:
        json.dump(fixture, f, indent=2)
    print(f"wrote {OUT}")
    print(
        "cases:",
        f"{len(het)} het, {len(reset)} reset, {len(chow)} chow, {len(cusum)} cusum",
    )


if __name__ == "__main__":
    main()
