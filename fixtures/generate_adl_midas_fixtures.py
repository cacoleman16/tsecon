"""Golden fixtures for `adl_midas`, the autoregressive-distributed-lag MIDAS
regression (crate `tsecon-midas`, `adl_midas`; Andreou, Ghysels & Kourtellos
2013).

WHAT IS PINNED
--------------
ADL-MIDAS is exactly OLS on a stacked design. For a low-frequency target y
(length n), P = ar_lags autoregressive lags and K stacked high-frequency lag
columns hf_lags (n x K, row t aligned with y_t), the regression runs over the
rows t = P..n-1:

    y_t = c + sum_{p=1..P} rho_p y_{t-p} + sum_{k=1..K} b_k hf_lags[t, k-1] + u_t

with parameter order [const, y.L1..y.LP, hf1..hfK]. The design is assembled
here INDEPENDENTLY of the crate, with `pandas.Series.shift` for the y lags and
`iloc[P:]` for the trimming, and fitted with statsmodels 0.15.0 `OLS`:

  * `cov_type="nonrobust"` (sigma2 = RSS / (n - P - k));
  * `cov_type="HAC"` with `cov_kwds={"maxlags": L, "use_correction": True}` —
    Bartlett, with the n/(n - k) small-sample factor that tsecon's MIDAS
    bindings fix on (statsmodels' own default is use_correction=False) — at
    the binding's default L = floor(4 ((n - P)/100)^(2/9)) on the EFFECTIVE
    number of rows n - P, and at L = 8.

Case `n275_p4` is chosen so the default bandwidth discriminates: the
effective-rows rule gives L = 4 while the same rule on the raw length n gives
5 (both stored).

The high-frequency lag columns follow the crate's alignment convention
(`stack_high_freq_lags`): most-recent-first, column k (1-based) of period t is
hf[off + (t + 1) * ratio - k], so column 1 is the LAST high-frequency
observation of period t (high-frequency lag 0) — which is why the binding
names the coefficients hf1..hfK rather than hf.L1..hf.LK. A pre-sample offset
off = K keeps every row complete. ratio = 3 (months per quarter).

LEVEL TARGETS (`level_cases`)
-----------------------------
A persistent target in levels (a log-GDP level near 9.5, an index near 100)
makes the lagged-y columns nearly collinear with the constant: cond(X) of
1e3 to 1e5, so cond(X'X) of 1e6 to 1e10. These cases pin the solver's
accuracy where it matters. Each stores, besides statsmodels' OLS fit
(params, residuals, R^2, nonrobust and HAC bse at the effective-rows default
bandwidth), an EXACT reference: the least-squares solution of the very same
float64 design and target in rational arithmetic (`fractions.Fraction`
normal equations, solved exactly — the forming of X'X loses nothing there),
with the exact nonrobust and HAC covariances (Bartlett weights 1 - j/(L+1),
the n/(n-k) factor) and their square roots taken at 40 digits (`decimal`).
`sm_vs_exact` records how far statsmodels' pinv solve is from it.

  * `loggdp_level_p2` — y = 9.5 + cumsum(0.005 + 0.3 x1 + 0.008 e), the hf
    regressors x ~ 0.01 N(0, 1) (n = 120, K = 3), ar_lags = 2;
  * `index100_p2` — y = 100 + 0.1 u, u an AR(2) driven by 0.3 x1 + e
    (n = 200, K = 3), ar_lags = 2.

Data: seeded NumPy `default_rng` draws (no dataset).

This generator NEVER imports tsecon. Doubles are written with json's shortest
round-trip repr (serde_json `float_roundtrip` on the Rust side).

Run:  .venv/bin/python fixtures/generate_adl_midas_fixtures.py
"""

from __future__ import annotations

import decimal
import json
import platform
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import statsmodels
import statsmodels.api as sm

HERE = Path(__file__).resolve().parent
OUT = HERE / "adl_midas.json"
SEED = 20261009
RATIO = 3


def nw_rule(n: int) -> int:
    return int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))


def simulate(rng, n, k, rho, intercept, het):
    hf = rng.standard_normal(n * RATIO + k)
    hf_lags = np.empty((n, k))
    for t in range(n):
        for col in range(1, k + 1):
            hf_lags[t, col - 1] = hf[k + (t + 1) * RATIO - col]
    w = 0.9 * 0.75 ** np.arange(k)
    y = np.zeros(n)
    for t in range(n):
        ar = sum(rho[p] * y[t - 1 - p] for p in range(len(rho)) if t - 1 - p >= 0)
        scale = 0.6 * (1.0 + het * abs(hf_lags[t, 0]))
        y[t] = intercept + ar + hf_lags[t] @ w + scale * rng.standard_normal()
    return y, hf_lags


def fit_case(name, y, hf_lags, ar_lags):
    n, k = hf_lags.shape
    df = pd.DataFrame({"y": y})
    for p in range(1, ar_lags + 1):
        df[f"y.L{p}"] = df["y"].shift(p)
    for col in range(k):
        df[f"hf{col + 1}"] = hf_lags[:, col]
    df = df.iloc[ar_lags:]
    design = sm.add_constant(df.drop(columns="y"), has_constant="add")
    names = list(design.columns)
    assert names == ["const"] + [f"y.L{p}" for p in range(1, ar_lags + 1)] + [f"hf{c + 1}" for c in range(k)]
    yy = df["y"].to_numpy()
    xx = design.to_numpy()
    nobs = len(yy)
    assert nobs == n - ar_lags
    default_l = nw_rule(nobs)
    fits = {}
    ols = sm.OLS(yy, xx).fit()
    fits["nonrobust"] = {
        "se_type": "nonrobust",
        "maxlags": None,
        "bse": ols.bse.tolist(),
        "tvalues": ols.tvalues.tolist(),
    }
    for label, lags in (("hac_default", default_l), ("hac_8", 8)):
        hac = sm.OLS(yy, xx).fit(cov_type="HAC", cov_kwds={"maxlags": lags, "use_correction": True})
        assert np.array_equal(hac.params, ols.params)
        fits[label] = {
            "se_type": "hac",
            "maxlags": int(lags),
            "bse": hac.bse.tolist(),
            "tvalues": hac.tvalues.tolist(),
        }
    return {
        "name": name,
        "ar_lags": ar_lags,
        "n": n,
        "K": k,
        "ratio": RATIO,
        "y": y.tolist(),
        "hf_lags": hf_lags.tolist(),
        "param_names": names,
        "params": ols.params.tolist(),
        "residuals": ols.resid.tolist(),
        "rsquared": float(ols.rsquared),
        "nobs": nobs,
        "first_index": ar_lags,
        "default_maxlags": default_l,
        "maxlags_rule_on_raw_n": nw_rule(n),
        "fits": fits,
    }


def exact_solve(a, b):
    """Gauss-Jordan elimination in exact rational arithmetic; `a` is k x k."""
    k = len(a)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for c in range(k):
        piv = next(r for r in range(c, k) if m[r][c] != 0)
        m[c], m[piv] = m[piv], m[c]
        for r in range(k):
            if r != c and m[r][c] != 0:
                f = m[r][c] / m[c][c]
                m[r] = [m[r][j] - f * m[c][j] for j in range(k + 1)]
    return [m[i][k] / m[i][i] for i in range(k)]


def exact_inverse(a):
    k = len(a)
    cols = [exact_solve(a, [Fraction(int(i == j)) for i in range(k)]) for j in range(k)]
    return [[cols[j][i] for j in range(k)] for i in range(k)]


def frac_sqrt(x: Fraction) -> float:
    with decimal.localcontext() as ctx:
        ctx.prec = 40
        return float((decimal.Decimal(x.numerator) / decimal.Decimal(x.denominator)).sqrt())


def exact_fit(xx, yy, maxlags):
    """Exact least squares of the float64 design: params, nonrobust and HAC
    (Bartlett, use_correction=True) standard errors."""
    n, k = xx.shape
    xf = [[Fraction(v) for v in row] for row in xx.tolist()]
    yf = [Fraction(v) for v in yy.tolist()]
    xtx = [[sum(xf[t][i] * xf[t][j] for t in range(n)) for j in range(k)] for i in range(k)]
    xty = [sum(xf[t][i] * yf[t] for t in range(n)) for i in range(k)]
    beta = exact_solve(xtx, xty)
    resid = [yf[t] - sum(xf[t][i] * beta[i] for i in range(k)) for t in range(n)]
    bread = exact_inverse(xtx)
    sigma2 = sum(u * u for u in resid) / (n - k)
    bse_nr = [frac_sqrt(sigma2 * bread[i][i]) for i in range(k)]
    scores = [[xf[t][i] * resid[t] for i in range(k)] for t in range(n)]
    meat = [[Fraction(0)] * k for _ in range(k)]
    for lag in range(0, maxlags + 1):
        w = Fraction(1) if lag == 0 else 1 - Fraction(lag, maxlags + 1)
        for t in range(lag, n):
            st, sl = scores[t], scores[t - lag]
            for i in range(k):
                for j in range(k):
                    g = w * st[i] * sl[j]
                    meat[i][j] += g
                    if lag > 0:
                        meat[j][i] += g
    tmp = [[sum(meat[i][l] * bread[l][j] for l in range(k)) for j in range(k)] for i in range(k)]
    cov = [[sum(bread[i][l] * tmp[l][j] for l in range(k)) for j in range(k)] for i in range(k)]
    corr = Fraction(n, n - k)
    bse_hac = [frac_sqrt(cov[i][i] * corr) for i in range(k)]
    return [float(b) for b in beta], [float(u) for u in resid], bse_nr, bse_hac


def level_case(name, y, hf_lags, ar_lags):
    case = fit_case(name, y, hf_lags, ar_lags)
    # The same design fit_case built (pandas.shift), rebuilt for the exact solve.
    df = pd.DataFrame({"y": y})
    for p in range(1, ar_lags + 1):
        df[f"y.L{p}"] = df["y"].shift(p)
    for col in range(hf_lags.shape[1]):
        df[f"hf{col + 1}"] = hf_lags[:, col]
    df = df.iloc[ar_lags:]
    xx = sm.add_constant(df.drop(columns="y"), has_constant="add").to_numpy()
    yy = df["y"].to_numpy()
    lags = case["default_maxlags"]
    params, resid, bse_nr, bse_hac = exact_fit(xx, yy, lags)
    case["cond_x"] = float(np.linalg.cond(xx))
    case["exact"] = {
        "params": params,
        "residuals": resid,
        "bse_nonrobust": bse_nr,
        "bse_hac_default": bse_hac,
    }
    rel = lambda a, e: float(np.max(np.abs(np.asarray(a) - np.asarray(e)) / np.abs(np.asarray(e))))
    case["sm_vs_exact"] = {
        "params": rel(case["params"], params),
        "bse_nonrobust": rel(case["fits"]["nonrobust"]["bse"], bse_nr),
        "bse_hac_default": rel(case["fits"]["hac_default"]["bse"], bse_hac),
    }
    del case["fits"]["hac_8"]
    return case


def main() -> None:
    rng = np.random.default_rng(SEED)
    y1, x1 = simulate(rng, 160, 6, [0.5, -0.2], 0.7, 0.0)
    y2, x2 = simulate(rng, 275, 6, [0.4, -0.1, 0.15, -0.05], 0.3, 0.8)
    cases = [fit_case("n160_p2", y1, x1, 2), fit_case("n275_p4", y2, x2, 4)]
    assert cases[1]["default_maxlags"] == 4 and cases[1]["maxlags_rule_on_raw_n"] == 5

    # Level targets (the solver-accuracy cases), from their own seeded stream
    # so the cases above are unchanged.
    lrng = np.random.default_rng(SEED + 1)
    n = 120
    x3 = lrng.standard_normal((n, 3)) * 0.01
    y3 = 9.5 + np.cumsum(0.005 + 0.3 * x3[:, 0] + 0.008 * lrng.standard_normal(n))
    n = 200
    x4 = lrng.standard_normal((n, 3))
    u = np.zeros(n)
    e = lrng.standard_normal(n)
    for t in range(2, n):
        u[t] = 0.6 * u[t - 1] - 0.2 * u[t - 2] + 0.3 * x4[t, 0] + e[t]
    y4 = 100.0 + 0.1 * u
    level_cases = [level_case("loggdp_level_p2", y3, x3, 2), level_case("index100_p2", y4, x4, 2)]
    fixture = {
        "_meta": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "statsmodels": statsmodels.__version__,
            "pandas": pd.__version__,
            "seed": SEED,
            "note": (
                "ADL-MIDAS = OLS of y_t on [1, y_{t-1..t-P}, hf_lags[t, :]] over t = P..n-1; "
                "design built with pandas.shift, fitted with statsmodels OLS: nonrobust, and "
                "HAC (Bartlett, use_correction=True) at the default maxlags "
                "floor(4((n-P)/100)^(2/9)) and at maxlags=8. hf_lags column 1 is high-"
                "frequency lag 0 (the period's last observation). level_cases: persistent "
                "level targets with an exact rational least-squares reference (exact) and "
                "statsmodels' distance from it (sm_vs_exact)."
            ),
        },
        "cases": cases,
        "level_cases": level_cases,
    }
    OUT.write_text(json.dumps(fixture, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")
    for c in cases:
        print(f"  {c['name']}: nobs={c['nobs']} default maxlags={c['default_maxlags']} "
              f"(raw-n rule {c['maxlags_rule_on_raw_n']}) params[0]={c['params'][0]!r} "
              f"R2={c['rsquared']!r} hac bse[1]={c['fits']['hac_default']['bse'][1]!r}")
    for c in level_cases:
        print(f"  {c['name']}: cond(X)={c['cond_x']:.2e} statsmodels vs exact {c['sm_vs_exact']}")


if __name__ == "__main__":
    main()
