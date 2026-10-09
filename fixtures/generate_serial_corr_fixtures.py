"""Golden fixtures for the serial-correlation tests of tsecon-spectest:
`breusch_godfrey_test` and `durbin_watson_test` (`fixtures/serial_corr.json`).

VALIDATION STRATEGY
===================
Every number written here comes from an INDEPENDENT reference: statsmodels
(a separate code path from the tsecon Rust crate) or a DOCUMENTED formula
transcribed below in NumPy/SciPy. This generator NEVER imports tsecon.

1. Breusch-Godfrey LM test (Breusch 1978, Godfrey 1978).  REFERENCE:
   statsmodels `acorr_breusch_godfrey(OLS(y, X).fit(), nlags=p,
   result_object=True)`, stored verbatim (`lm`, `lmpval`, `fval`, `fpval`).
   The statsmodels convention, read from its source (0.15.0) and pinned
   here:

   * the OLS residuals u_1..u_T are PADDED with p zeros in front, so the
     auxiliary regression keeps all T observations and the lag-j column at
     time t is u_{t-j} (0 when t - j < 1) — no observations are dropped;
   * the default order is `nlags = min(10, T // 5)` (stored as `None` cases
     with the resolved value in `nlags`);
   * the auxiliary regression is u on [X, u_{-1}, ..., u_{-p}]; when the
     design does NOT span a constant (statsmodels `k_constant == 0`: no
     constant column and no implicit one, e.g. complementary dummies) a
     column of ones is ADDED to the auxiliary design (`constant_added`);
   * LM = T * R^2 (the CENTERED R^2 of the auxiliary regression, which
     always has an intercept), compared with chi2(p);
   * F = the Wald/F test that the p lag coefficients are jointly zero in the
     auxiliary regression, F(p, T - k_aux), k_aux = k + p (+ 1 when the
     constant was added) — asserted equal to statsmodels' `df_resid`.

2. Durbin-Watson statistic (Durbin & Watson 1950, 1951).  REFERENCE:
   statsmodels `durbin_watson(resid)` =
   sum_{t=2}^T (u_t - u_{t-1})^2 / sum_{t=1}^T u_t^2.

3. Durbin-Watson EXACT p-value (Imhof 1961).  REFERENCE: the documented
   formula, transcribed here (no runnable package reference exists in the
   fixture environment).  Under H0 (u ~ N(0, s^2 I)) the residuals are
   e = M u with M = I - X (X'X)^{-1} X', so

       d = e'A e / e'e,  A = D'D (D the (T-1) x T first-difference matrix),
       P(d <= c) = P( sum_i (nu_i - c) z_i^2 <= 0 ),  z_i iid N(0, 1),

   where nu_1..nu_{T-k} are the eigenvalues of Z'A Z, Z an orthonormal basis
   of the orthogonal complement of col(X) (here: the last T - k columns of
   numpy's COMPLETE QR of X; LAPACK `eigvalsh`). For Q = sum lam_i z_i^2,
   Imhof (1961, eq. 3.2) gives

       P(Q <= 0) = 1/2 - (1/pi) int_0^inf sin(theta(u)) / (u rho(u)) du,
       theta(u) = (1/2) sum_i arctan(lam_i u),
       rho(u)   = prod_i (1 + lam_i^2 u^2)^{1/4},

   integrated with SciPy `quad` (QUADPACK QAGI) at epsabs 1e-14 and
   cross-checked against a second route (QAGS on [0, U] in pieces plus
   Imhof's tail bound) to 1e-12 before anything is stored. The p-values are
   `pvalue_positive = P(d <= d_obs)` (H1: rho > 0, the classical test),
   `pvalue_negative = P(d >= d_obs)` and
   `pvalue_two_sided = min(1, 2 min(...))`. The formula itself is graded by
   a seeded Monte Carlo of the null distribution in
   `lab/serial_corr/dw_exact_mc.py` (not stored here).

Data: seeded simulated regressions (`numpy.random.default_rng`), AR(1) and
iid errors, with and without extra regressors, two designs without a
constant (one with a large omitted intercept), an implicit-constant (complementary dummies) design, a short and a long
sample, and three US macro regressions built from statsmodels' bundled
`macrodata` by transformation only (quarterly log-differences x 100/400,
first differences), so the fixture stores derived series, not the dataset.

Run:  .venv/bin/python fixtures/generate_serial_corr_fixtures.py
"""

from __future__ import annotations

import json
import platform
import warnings
from pathlib import Path

import numpy as np
import scipy
import statsmodels
import statsmodels.api as sm
from scipy import integrate
from statsmodels.stats.diagnostic import acorr_breusch_godfrey
from statsmodels.stats.stattools import durbin_watson

OUT = Path(__file__).resolve().parent / "serial_corr.json"


# --------------------------------------------------------------- data

def ar1_errors(rng, n, rho):
    """Stationary AR(1) errors (unit innovation variance)."""
    e = rng.standard_normal(n)
    u = np.empty(n)
    u[0] = e[0] / np.sqrt(1.0 - rho * rho)
    for t in range(1, n):
        u[t] = rho * u[t - 1] + e[t]
    return u


def ar1_regressor(rng, n, rho):
    x = np.empty(n)
    x[0] = rng.standard_normal() / np.sqrt(1.0 - rho * rho)
    for t in range(1, n):
        x[t] = rho * x[t - 1] + rng.standard_normal()
    return x


def macro():
    """Transformations of statsmodels' macrodata (1959Q1-2009Q3, 203 rows)."""
    md = sm.datasets.macrodata.load_pandas().data
    dl = lambda c: 100.0 * np.diff(np.log(md[c].to_numpy(dtype=float)))  # noqa: E731
    return {
        "cons_growth": dl("realcons"),
        "income_growth": dl("realdpi"),
        "gdp_growth": dl("realgdp"),
        "inv_growth": dl("realinv"),
        "inflation": 4.0 * dl("cpi"),
        "log_unemp": 100.0 * np.log(md["unemp"].to_numpy(dtype=float))[1:],
        "d_realint": np.diff(md["realint"].to_numpy(dtype=float)),
    }


# --------------------------------------------- Durbin-Watson exact p-value

def dw_null_eigenvalues(X):
    """nu_1..nu_{T-k}: eigenvalues of Z'AZ, Z the orthogonal complement of
    col(X) from a COMPLETE QR (an independent route from the crate's)."""
    T, k = X.shape
    Q, _ = np.linalg.qr(X, mode="complete")
    Z = Q[:, k:]
    D = np.diff(np.eye(T), axis=0)  # (T-1) x T first differences
    A = D.T @ D
    B = Z.T @ A @ Z
    return np.linalg.eigvalsh(0.5 * (B + B.T))


def _imhof_integrand(lam):
    def f(u):
        if u == 0.0:
            return 0.5 * float(np.sum(lam))
        theta = 0.5 * np.sum(np.arctan(lam * u))
        log_rho = 0.25 * np.sum(np.log1p((lam * u) ** 2))
        return float(np.sin(theta) * np.exp(-log_rho) / u)
    return f


def imhof_le0(lam):
    """P(sum lam_i z_i^2 <= 0) by Imhof's formula; two quadrature routes."""
    f = _imhof_integrand(lam)
    with warnings.catch_warnings():
        warnings.simplefilter("error", integrate.IntegrationWarning)
        i1, err1 = integrate.quad(f, 0.0, np.inf, epsabs=1e-14, epsrel=1e-13, limit=5000)
    # Route 2: finite range in pieces + Imhof's tail bound
    #   |int_U^inf| <= 1 / (pi (m/2) U^{m/2} prod |lam_i|^{1/2})  (Imhof 1961, 3.3)
    big = lam[np.abs(lam) > 1e-3 * np.max(np.abs(lam))]
    m = big.size
    log_u = (np.log(1e17) - np.log(np.pi * m / 2.0) - 0.5 * np.sum(np.log(np.abs(big)))) / (m / 2.0)
    U = float(np.exp(log_u))
    edges = np.concatenate([[0.0], np.geomspace(1e-3, U, 60)])
    i2 = 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        # Pieces whose integral is below the noise floor can trip QUADPACK's
        # round-off detector; the agreement assertion below is the check.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", integrate.IntegrationWarning)
            v, _ = integrate.quad(f, a, b, epsabs=1e-16, epsrel=1e-13, limit=2000)
        i2 += v
    assert abs(i1 - i2) < 1e-12, (i1, i2, err1)
    return 0.5 - i1 / np.pi


def dw_exact(X, d_obs):
    nu = dw_null_eigenvalues(X)
    p_le = imhof_le0(nu - d_obs)
    p_le = min(1.0, max(0.0, p_le))
    p_ge = min(1.0, max(0.0, 1.0 - p_le))
    return {
        "pvalue_positive": p_le,
        "pvalue_negative": p_ge,
        "pvalue_two_sided": min(1.0, 2.0 * min(p_le, p_ge)),
    }


# --------------------------------------------------------------- cases

def build_case(name, y, X, nlags_list, note):
    T, k = X.shape
    res = sm.OLS(y, X).fit()
    bg = []
    for p in nlags_list:
        r = acorr_breusch_godfrey(res, nlags=p, store=True)
        resolved = r.res_store.usedlag
        resols = r.res_store.resols
        added = 0 if res.k_constant else 1
        k_aux = k + resolved + added
        assert resols.model.exog.shape[1] == k_aux
        assert int(round(resols.df_resid)) == T - k_aux
        assert resolved == (min(10, T // 5) if p is None else p)
        bg.append({
            "nlags_arg": p,
            "nlags": int(resolved),
            "statistic": float(r.lm),
            "df": int(resolved),
            "pvalue": float(r.lmpval),
            "fstat": float(r.fval),
            "f_df_num": int(resolved),
            "f_df_den": int(T - k_aux),
            "f_pvalue": float(r.fpval),
            "constant_added": bool(added),
        })
    d = float(durbin_watson(res.resid))
    # identity check against the textbook definition
    u = res.resid
    assert abs(d - np.sum(np.diff(u) ** 2) / np.sum(u ** 2)) < 1e-12
    dw = {"statistic": d, "nobs": int(T), "df_resid": int(T - k), **dw_exact(X, d)}
    return {
        "name": name,
        "note": note,
        "nobs": int(T),
        "k": int(k),
        "y": y.tolist(),
        "columns": [X[:, j].tolist() for j in range(k)],
        "breusch_godfrey": bg,
        "durbin_watson": dw,
    }


def main():
    cases = []

    # 1. AR(1) errors, two extra regressors.
    rng = np.random.default_rng(20261001)
    T = 120
    X = np.column_stack([np.ones(T), ar1_regressor(rng, T, 0.5), rng.standard_normal(T)])
    y = X @ np.array([1.0, 0.7, -0.4]) + ar1_errors(rng, T, 0.5)
    cases.append(build_case("ar1_rho05_two_regressors", y, X, [None, 1, 2, 4],
                            "AR(1) errors rho=0.5, const + AR(1) regressor + iid regressor"))

    # 2. iid errors, two extra regressors.
    rng = np.random.default_rng(20261002)
    T = 100
    X = np.column_stack([np.ones(T), ar1_regressor(rng, T, 0.6), rng.standard_normal(T)])
    y = X @ np.array([0.5, -0.3, 0.8]) + rng.standard_normal(T)
    cases.append(build_case("iid_two_regressors", y, X, [None, 1, 3, 6],
                            "iid N(0,1) errors, const + AR(1) regressor + iid regressor"))

    # 3. Constant only (no extra regressors), strongly autocorrelated errors.
    rng = np.random.default_rng(20261003)
    T = 60
    X = np.ones((T, 1))
    y = 2.0 + ar1_errors(rng, T, 0.8)
    cases.append(build_case("ar1_rho08_const_only", y, X, [None, 1, 2],
                            "mean model (constant only), AR(1) errors rho=0.8"))

    # 4. Constant only, iid errors.
    rng = np.random.default_rng(20261004)
    T = 40
    X = np.ones((T, 1))
    y = -1.0 + rng.standard_normal(T)
    cases.append(build_case("iid_const_only", y, X, [None, 1, 4],
                            "mean model (constant only), iid errors"))

    # 5. Negative autocorrelation, constant + linear trend.
    rng = np.random.default_rng(20261005)
    T = 90
    X = np.column_stack([np.ones(T), np.arange(T) / T])
    y = X @ np.array([0.2, 1.5]) + ar1_errors(rng, T, -0.6)
    cases.append(build_case("ar1_negative_rho_trend", y, X, [None, 1, 2],
                            "AR(1) errors rho=-0.6, const + linear trend"))

    # 6. Design WITHOUT a constant: statsmodels adds one to the auxiliary design.
    rng = np.random.default_rng(20261006)
    T = 80
    X = np.column_stack([ar1_regressor(rng, T, 0.3), rng.standard_normal(T)])
    y = 0.4 + X @ np.array([0.5, -0.3]) + ar1_errors(rng, T, 0.3)
    cases.append(build_case("no_constant_design", y, X, [None, 2],
                            "regression through the origin (no constant column), AR(1) errors rho=0.3"))

    # 7. Implicit constant: two complementary dummies + a regressor.
    rng = np.random.default_rng(20261007)
    T = 96
    d1 = (np.arange(T) < T // 2).astype(float)
    X = np.column_stack([d1, 1.0 - d1, rng.standard_normal(T)])
    y = X @ np.array([1.0, 2.0, 0.5]) + ar1_errors(rng, T, 0.4)
    cases.append(build_case("implicit_constant_dummies", y, X, [None, 3],
                            "two complementary regime dummies (implicit constant) + regressor, rho=0.4"))

    # 7b. No constant and a large omitted intercept: the added auxiliary
    # constant explains almost all of the residuals, the stress case for the
    # restricted sum of squares of the F-form (statsmodels is accurate here to
    # about 1e-11, checked against a 50-digit evaluation during review).
    rng = np.random.default_rng(20261013)
    T = 80
    X = (3.0 + ar1_regressor(rng, T, 0.3))[:, None]
    y = 5.0 + 0.5 * X[:, 0] + 0.01 * ar1_errors(rng, T, 0.3)
    cases.append(build_case("no_constant_omitted_intercept", y, X, [1, 2],
                            "through the origin on a regressor with mean 3, true intercept 5, "
                            "errors 0.01 x AR(1) rho=0.3: the added constant explains nearly all of u"))

    # 8. Short sample.
    rng = np.random.default_rng(20261008)
    T = 15
    X = np.column_stack([np.ones(T), rng.standard_normal(T)])
    y = X @ np.array([1.0, 0.5]) + ar1_errors(rng, T, 0.5)
    cases.append(build_case("short_sample", y, X, [None, 1],
                            "T = 15, const + regressor, AR(1) errors rho=0.5"))

    # 9. Long sample, three extra regressors, weak autocorrelation.
    rng = np.random.default_rng(20261009)
    T = 500
    X = np.column_stack([np.ones(T), ar1_regressor(rng, T, 0.7), rng.standard_normal(T),
                         ar1_regressor(rng, T, -0.2)])
    y = X @ np.array([0.3, 0.5, -0.5, 0.25]) + ar1_errors(rng, T, 0.2)
    cases.append(build_case("long_sample_weak_ar1", y, X, [None, 12],
                            "T = 500, const + 3 regressors, AR(1) errors rho=0.2"))

    # 10-12. US macro regressions (transformations of statsmodels macrodata).
    m = macro()
    T = m["cons_growth"].size
    X = np.column_stack([np.ones(T), m["income_growth"]])
    cases.append(build_case("macro_consumption_on_income", m["cons_growth"], X, [None, 1, 4, 8],
                            "100 dlog realcons on const + 100 dlog realdpi (statsmodels macrodata)"))
    X = np.column_stack([np.ones(T), m["log_unemp"]])
    cases.append(build_case("macro_inflation_on_unemployment", m["inflation"], X, [None, 1, 4],
                            "400 dlog cpi on const + 100 log unemp (a static Phillips curve; statsmodels macrodata)"))
    X = np.column_stack([np.ones(T), m["gdp_growth"], m["d_realint"]])
    cases.append(build_case("macro_investment_accelerator", m["inv_growth"], X, [None, 4],
                            "100 dlog realinv on const + 100 dlog realgdp + diff realint (statsmodels macrodata)"))

    fixture = {
        "_meta": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "statsmodels": statsmodels.__version__,
            "python": platform.python_version(),
            "references": {
                "breusch_godfrey": "statsmodels.stats.diagnostic.acorr_breusch_godfrey(OLS fit, nlags) "
                                   "— zero-padded lagged residuals, default nlags=min(10, T//5), "
                                   "a constant added to the auxiliary design when the model has none",
                "durbin_watson": "statsmodels.stats.stattools.durbin_watson(resid)",
                "durbin_watson_exact_pvalue": "documented Imhof (1961) formula on the eigenvalues of "
                                              "Z'AZ (numpy complete QR + LAPACK eigvalsh), "
                                              "scipy.integrate.quad; two quadrature routes agree to 1e-12",
            },
            "note": "Macro cases are transformations of statsmodels' macrodata; simulated cases are "
                    "seeded numpy default_rng draws. Generated without importing tsecon.",
        },
        "cases": cases,
    }
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(fixture, fh, indent=1)
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes); {len(cases)} cases")
    for c in cases:
        dw = c["durbin_watson"]
        bgs = "  ".join(f"p={b['nlags']}:LM={b['statistic']:.3f}(p={b['pvalue']:.3g})" for b in c["breusch_godfrey"])
        print(f"  {c['name']:34s} T={c['nobs']:3d} k={c['k']} DW={dw['statistic']:.4f} "
              f"p+={dw['pvalue_positive']:.3g}  {bgs}")


if __name__ == "__main__":
    main()
