"""Golden fixtures for the Phase-2 slices: ML (sklearn), LP (statsmodels OLS-HAC
+ linearmodels IV2SLS), and forecast-evaluation depth (self-authored CW/GW,
formulas documented inline).

Run with the project venv:  .venv/bin/python fixtures/generate_phase2_fixtures.py
"""
import json
import platform
from pathlib import Path

import numpy as np

OUT = Path(__file__).parent
r6 = lambda a: [round(float(x), 10) for x in np.asarray(a).ravel()]


# ------------------------------------------------------------------ ML
def gen_ml():
    import sklearn
    from sklearn.linear_model import ElasticNet, Lasso, Ridge

    rng = np.random.default_rng(19)
    n, k = 200, 12
    X = rng.standard_normal((n, k))
    beta = np.zeros(k)
    beta[:4] = [1.5, -0.9, 0.6, 0.3]
    y = X @ beta + rng.standard_normal(n) * 1.2

    # Conventions pinned: features standardized HERE (mean 0, sd 1 with ddof=0),
    # y demeaned HERE; sklearn fit_intercept=False on the pre-processed data.
    Xs = (X - X.mean(0)) / X.std(0)
    yc = y - y.mean()

    cases = []
    for name, est in [
        ("ridge_a1", Ridge(alpha=1.0, fit_intercept=False)),
        ("ridge_a10", Ridge(alpha=10.0, fit_intercept=False)),
        ("lasso_a01", Lasso(alpha=0.1, fit_intercept=False, tol=1e-12, max_iter=1_000_000)),
        ("lasso_a02", Lasso(alpha=0.2, fit_intercept=False, tol=1e-12, max_iter=1_000_000)),
        ("enet_a01_l05", ElasticNet(alpha=0.1, l1_ratio=0.5, fit_intercept=False, tol=1e-12, max_iter=1_000_000)),
    ]:
        est.fit(Xs, yc)
        cases.append({"name": name, "params": est.get_params(), "coef": r6(est.coef_)})
        cases[-1]["params"] = {k2: v for k2, v in cases[-1]["params"].items()
                               if k2 in ("alpha", "l1_ratio")}

    out = {
        "_meta": {"sklearn": sklearn.__version__, "numpy": np.__version__,
                  "python": platform.python_version(),
                  "objective_note": ("sklearn Lasso/ElasticNet minimize "
                                     "(1/(2n))||y-Xb||^2 + alpha*l1_ratio*||b||_1 "
                                     "+ 0.5*alpha*(1-l1_ratio)*||b||^2; Ridge minimizes "
                                     "||y-Xb||^2 + alpha*||b||^2 (NO 1/n) — match exactly.")},
        "X_standardized": [[round(float(v), 10) for v in row] for row in Xs],
        "y_centered": r6(yc),
        "true_beta": r6(beta),
        "cases": cases,
    }
    (OUT / "ml.json").write_text(json.dumps(out, separators=(",", ":")))
    print("wrote ml.json")


# ------------------------------------------------------------------ LP
def gen_lp():
    import statsmodels
    import statsmodels.api as sm
    import linearmodels
    from linearmodels.iv import IV2SLS

    rng = np.random.default_rng(23)
    n = 400
    # Observed structural shock e with known IRF on y: psi_h = 0.9^h; plus an
    # instrument z correlated with a mismeasured/endogenous impulse x.
    e = rng.standard_normal(n)
    y = np.zeros(n)
    for t in range(n):
        for h in range(min(t + 1, 24)):
            y[t] += (0.9 ** h) * e[t - h]
    y += 0.6 * rng.standard_normal(n)
    x = e + 0.5 * rng.standard_normal(n)              # endogenous impulse proxy
    z = e + 0.5 * rng.standard_normal(n)              # instrument

    H = 8
    NLAG_CONTROLS = 4

    def controls(t0):
        # lagged y controls y_{t-1..t-4}, aligned so row t uses lags before t
        C = np.column_stack([y[NLAG_CONTROLS - j - 1: n - j - 1 - t0] for j in range(NLAG_CONTROLS)])
        return C

    ols_lp, iv_lp = [], []
    for h in range(H + 1):
        yy = y[NLAG_CONTROLS + h:]
        ee = e[NLAG_CONTROLS: n - h]
        C = np.column_stack([np.ones(len(yy))] + [y[NLAG_CONTROLS - 1 - j: n - h - 1 - j] for j in range(NLAG_CONTROLS)])
        X = np.column_stack([ee, C])
        r = sm.OLS(yy, X).fit(cov_type="HAC", cov_kwds={"maxlags": h + 4, "use_correction": True})
        ols_lp.append({"h": h, "beta": float(r.params[0]), "se_hac": float(r.bse[0]),
                       "maxlags": h + 4, "nobs": int(r.nobs)})

        xx = x[NLAG_CONTROLS: n - h]
        zz = z[NLAG_CONTROLS: n - h]
        iv = IV2SLS(yy, C, xx.reshape(-1, 1), zz.reshape(-1, 1)).fit(
            cov_type="kernel", kernel="bartlett", bandwidth=h + 4)
        iv_lp.append({"h": h, "beta": float(iv.params.iloc[-1]), "se_kernel": float(iv.std_errors.iloc[-1]),
                      "bandwidth": h + 4, "nobs": int(iv.nobs)})

    out = {
        "_meta": {"statsmodels": statsmodels.__version__, "linearmodels": linearmodels.__version__,
                  "numpy": np.__version__,
                  "design_note": ("y_t = sum_h 0.9^h e_{t-h} + noise; LP of y_{t+h} on e_t with "
                                  "4 lagged-y controls + intercept, HAC(maxlags=h+4, use_correction). "
                                  "LP-IV: same regression with endogenous x_t instrumented by z_t, "
                                  "linearmodels IV2SLS kernel(bartlett, bandwidth=h+4). Regressor order "
                                  "in the OLS design: [e_t, const, y_{t-1..t-4}].")},
        "y": r6(y), "e": r6(e), "x": r6(x), "z": r6(z),
        "n_lag_controls": NLAG_CONTROLS,
        "true_irf": r6(0.9 ** np.arange(H + 1)),
        "ols_lp": ols_lp,
        "iv_lp": iv_lp,
    }
    (OUT / "lp.json").write_text(json.dumps(out, separators=(",", ":")))
    print("wrote lp.json")


# ------------------------------------------- forecast evaluation depth
def gen_forecast_eval2():
    """Self-authored reference values for Clark-West and Giacomini-White.

    Clark-West (2007): for nested models with forecast errors e1 (small) and
    e2 (large), f_t = e1^2 - e2^2 + (yhat1 - yhat2)^2; CW stat = mean(f) /
    sqrt(LRV_bartlett(f, L)/n), one-sided normal p-value.

    Giacomini-White (2006), unconditional case with test function h_t = 1:
    equivalent to a DM-type test on d_t = L1_t - L2_t with variance from the
    Bartlett LRV; GW stat = n * dbar' * Shat^-1 * dbar ~ chi2(1) (scalar case).
    """
    rng = np.random.default_rng(31)
    n = 150
    yhat1 = rng.standard_normal(n) * 0.2          # nested small model forecast
    yhat2 = yhat1 + rng.standard_normal(n) * 0.3  # larger model forecast
    ytrue = 0.1 * rng.standard_normal(n) + yhat1 * 0.9
    e1, e2 = ytrue - yhat1, ytrue - yhat2
    L = 3

    def bartlett_lrv(v, lags):
        v = v - v.mean()
        g = [float(v[: len(v) - k] @ v[k:] / len(v)) for k in range(lags + 1)]
        return g[0] + 2 * sum((1 - k / (lags + 1)) * g[k] for k in range(1, lags + 1))

    f = e1**2 - e2**2 + (yhat1 - yhat2) ** 2
    cw_stat = f.mean() / np.sqrt(bartlett_lrv(f, L) / n)
    from scipy import stats as sps
    cw_p = float(sps.norm.sf(cw_stat))

    d = e1**2 - e2**2
    gw_stat = n * d.mean() ** 2 / bartlett_lrv(d, L)
    gw_p = float(sps.chi2.sf(gw_stat, 1))

    out = {
        "_meta": {"authored": "documented CW/GW formulas above", "numpy": np.__version__},
        "ytrue": r6(ytrue), "yhat1": r6(yhat1), "yhat2": r6(yhat2),
        "lrv_lags": L,
        "clark_west": {"stat": float(cw_stat), "pvalue_one_sided": cw_p},
        "giacomini_white_uncond": {"stat": float(gw_stat), "pvalue_chi2_1": gw_p},
    }
    (OUT / "forecast_eval2.json").write_text(json.dumps(out, separators=(",", ":")))
    print("wrote forecast_eval2.json")


if __name__ == "__main__":
    gen_ml()
    gen_lp()
    gen_forecast_eval2()
