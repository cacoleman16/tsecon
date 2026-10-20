"""Golden fixtures for the ARBITRARY-IMPACT historical decomposition
(`historical_decomposition(identification="impact", impact=A0)`, the Rust
`tsecon_ident::decompose_with_impact`).

    .venv/bin/python fixtures/generate_historical_decomposition_impact_fixtures.py

VALIDATION STRATEGY
===================
Nothing here imports tsecon. The reduced form is statsmodels'
`VAR(data).fit(p, trend="c")` (an independent OLS), and everything after it is
a documented closed form transcribed in NumPy:

    U        = data[p:] - X @ params                (reduced-form residuals)
    E        = solve(A0, U')'                       (structural shocks, eps_t = A0^-1 u_t)
    Theta_s  = Psi_s A0,  Psi_0 = I,  Psi_h = sum_{i<=min(h,p)} Psi_{h-i} A_i
    HD[t,i,j] = sum_{s<=t} Theta_s[i,j] E[t-s,j]
    baseline = the VAR iterated from the actual pre-sample with every shock zeroed
    shock_covariance = E'E / T_eff

The adding-up identity y_{p+t} = baseline_t + sum_j HD[t][:, j] holds for ANY
invertible A0 (summing over j telescopes Psi_s A0 A0^-1 u back to Psi_s u); it
is asserted here at 1e-10 and stored as `adding_up_residual`.

CASES
-----
* `bq_var8` - the Blanchard-Quah design-replication data (fixtures/
  macrodata_bq.csv, transformed exactly as docs/examples/
  replication_blanchard_quah.py does: 100*dlog real GDP demeaned separately
  before/after 1974Q1, unemployment linearly detrended), VAR(8) with a
  constant, and the Blanchard-Quah impact matrix by the NumPy closed form
  B = D chol(C(1) Sigma_u C(1)'), Sigma_u df-adjusted, with the demand column
  sign-anchored so output rises on impact (`impact_signs=[(0, 1)]`). Stores the
  transformed data (a transformation of the public-domain macrodata series,
  per fixtures/README.md), hd, baseline, shocks, theta[0..3],
  shock_covariance and the seven NBER-recession rows of the replication page.
  `b` is statsmodels' `params` (intercept row, then lag blocks), the reduced
  form the Rust closed form is fed; the Python reload instead re-fits through
  tsecon's own OLS, so it also measures the OLS gap.
* `random_3var_p2` - a seeded stable 3-variable VAR(2) and a random
  NON-orthogonal, NON-covariance-consistent A0 (A0 A0' != Sigma): the identity
  must still hold and E'E/T is then far from the identity.

The cross-implementation gap is the statsmodels-vs-faer OLS difference
(~1e-13 on the VAR(8)), so the Rust golden compares at rtol 1e-8 / atol 1e-10.
Margins: the anchored impact cell is asserted |B[0,1]| > 1e-8 so the sign
anchor is never a knife-edge decision.
"""

import csv
import json
import platform
from pathlib import Path

import numpy as np
import scipy
import statsmodels
from statsmodels.tsa.api import VAR

HERE = Path(__file__).resolve().parent
OUT = HERE / "historical_decomposition_impact.json"
CSV = HERE / "macrodata_bq.csv"

META = {
    "numpy": np.__version__,
    "scipy": scipy.__version__,
    "statsmodels": statsmodels.__version__,
    "python": platform.python_version(),
}

NBER_RECESSIONS = [
    ("1960Q2", "1961Q1"), ("1969Q4", "1970Q4"), ("1973Q4", "1975Q1"),
    ("1980Q1", "1980Q3"), ("1981Q3", "1982Q4"), ("1990Q3", "1991Q1"),
    ("2001Q1", "2001Q4"), ("2007Q4", "2009Q2"),
]


def qnum(label):
    y, q = label.split("Q")
    return int(y) + (int(q) - 1) / 4.0


def bq_data():
    """The replication page's transform of the committed macrodata extract."""
    rows = [r for r in csv.reader(open(CSV)) if r and not r[0].startswith("#")]
    assert rows[0] == ["quarter", "realgdp", "unemp"]
    body = rows[1:]
    quarter = [r[0] for r in body]
    gdp = np.array([float(r[1]) for r in body])
    unemp_raw = np.array([float(r[2]) for r in body])
    growth = 100.0 * np.diff(np.log(gdp))
    unemp = unemp_raw[1:].astype(float)
    quarter = quarter[1:]
    years = np.array([qnum(q) for q in quarter])
    post = years >= qnum("1974Q1")
    growth = growth - np.where(post, growth[post].mean(), growth[~post].mean())
    t = np.arange(len(unemp), dtype=float)
    Z = np.column_stack([np.ones_like(t), t])
    unemp = unemp - Z @ np.linalg.lstsq(Z, unemp, rcond=None)[0]
    return quarter, np.column_stack([growth, unemp])


def ma_weights(coefs, horizon):
    n = coefs[0].shape[0]
    p = len(coefs)
    psi = [np.eye(n)]
    for h in range(1, horizon + 1):
        acc = np.zeros((n, n))
        for i in range(1, min(h, p) + 1):
            acc = acc + psi[h - i] @ coefs[i - 1]
        psi.append(acc)
    return psi


def baseline_path(coefs, intercept, data, p):
    n = data.shape[1]
    t_eff = data.shape[0] - p
    base = np.zeros((t_eff, n))
    for t in range(t_eff):
        val = intercept.copy()
        for l in range(1, p + 1):
            lag_eff = t - l
            yhat = data[p + lag_eff] if lag_eff < 0 else base[lag_eff]
            val = val + coefs[l - 1] @ yhat
        base[t] = val
    return base


def hd_impact(data, p, a0):
    """The documented closed form on a statsmodels VAR(p) with a constant."""
    res = VAR(data).fit(p, trend="c")
    params = np.asarray(res.params)                 # (1 + n p) x n
    coefs = [np.asarray(a) for a in res.coefs]       # A_1..A_p, A_l[i, v]
    intercept = np.asarray(res.intercept)
    n = data.shape[1]
    t_eff = data.shape[0] - p
    X = np.column_stack(
        [np.ones(t_eff)] + [data[p - l:data.shape[0] - l] for l in range(1, p + 1)]
    )
    U = data[p:] - X @ params
    assert np.allclose(U, np.asarray(res.resid), atol=1e-12)
    E = np.linalg.solve(a0, U.T).T
    psi = ma_weights(coefs, t_eff - 1)
    theta = [ps @ a0 for ps in psi]
    hd = np.zeros((t_eff, n, n))
    for t in range(t_eff):
        for s in range(t + 1):
            hd[t] += theta[s] * E[t - s][None, :]
    base = baseline_path(coefs, intercept, data, p)
    resid = data[p:] - base - hd.sum(axis=2)
    adding_up = float(np.abs(resid).max())
    assert adding_up < 1e-10, adding_up
    return {
        "res": res, "params": params, "U": U, "E": E, "theta": theta, "hd": hd, "baseline": base,
        "shock_covariance": E.T @ E / t_eff, "adding_up_residual": adding_up,
    }


def rows(m):
    m = np.asarray(m)
    return [[float(v) for v in r] for r in m]


def case_bq():
    quarter, X = bq_data()
    p = 8
    res = VAR(X).fit(p, trend="c")
    k = X.shape[1]
    sig = np.asarray(res.sigma_u)                    # df-adjusted U'U/(T - m)
    D = np.eye(k) - np.asarray(res.coefs).sum(axis=0)
    C1 = np.linalg.inv(D)
    LR = np.linalg.cholesky(C1 @ sig @ C1.T)
    B = D @ LR
    # impact_signs=[(0, 1)]: demand raises output on impact (strict < 0 flips).
    assert abs(B[0, 1]) > 1e-8, "anchor decision is knife-edge"
    if B[0, 1] < 0:
        B[:, 1] = -B[:, 1]
        LR[:, 1] = -LR[:, 1]
    out = hd_impact(X, p, B)
    t_eff = X.shape[0] - p
    # (T - m)/T * I: the BQ impact orthogonalizes these residuals exactly.
    m = 1 + k * p
    assert np.allclose(out["shock_covariance"], (t_eff - m) / t_eff * np.eye(k), atol=1e-10)
    # NBER recession rows (unemployment, peak-to-trough), as on the page.
    first = p  # times are data rows lags.. in `quarter` (growth sample) indexing
    recess = []
    actual_u = X[p:, 1] - out["baseline"][:, 1]
    for peak, trough in NBER_RECESSIONS:
        if peak not in quarter or trough not in quarter:
            continue
        a, b = quarter.index(peak) - first, quarter.index(trough) - first
        if a < 0:
            continue
        recess.append({
            "peak": peak, "trough": trough, "peak_t": a, "trough_t": b,
            "actual": float(actual_u[b] - actual_u[a]),
            "demand": float(out["hd"][b, 1, 1] - out["hd"][a, 1, 1]),
            "supply": float(out["hd"][b, 1, 0] - out["hd"][a, 1, 0]),
        })
    assert len(recess) == 7
    return {
        "p": p, "n": k, "t_eff": t_eff,
        "data": rows(X),
        "b": rows(out["params"]),
        "impact": rows(B),
        "long_run": rows(LR),
        "impact_signs": [[0, 1]],
        "hd": [rows(h) for h in out["hd"]],
        "baseline": rows(out["baseline"]),
        "shocks": rows(out["E"]),
        "theta_head": [rows(t) for t in out["theta"][:4]],
        "shock_covariance": rows(out["shock_covariance"]),
        "adding_up_residual": out["adding_up_residual"],
        "recessions": recess,
    }


def case_random():
    rng = np.random.default_rng(20260912)
    n, p, T = 3, 2, 160
    A1 = np.array([[0.45, 0.10, -0.05], [0.05, 0.35, 0.10], [-0.08, 0.04, 0.30]])
    A2 = np.array([[0.10, 0.00, 0.03], [0.00, 0.08, -0.02], [0.04, 0.00, 0.06]])
    L = np.array([[1.0, 0.0, 0.0], [0.3, 0.8, 0.0], [-0.2, 0.25, 0.6]])
    y = np.zeros((T + 200 + p, n))
    for t in range(p, len(y)):
        y[t] = 0.2 + A1 @ y[t - 1] + A2 @ y[t - 2] + L @ rng.standard_normal(n)
    data = y[200:]
    # A random, non-orthogonal, non-covariance-consistent impact matrix.
    a0 = rng.standard_normal((n, n)) + 1.5 * np.eye(n)
    sv = np.linalg.svd(a0, compute_uv=False)
    assert sv.min() > 0.1 * sv.max()                 # comfortably invertible
    out = hd_impact(data, p, a0)
    sig = np.asarray(out["res"].sigma_u)
    assert np.abs(a0 @ a0.T - sig).max() > 0.1       # NOT covariance-consistent
    assert np.abs(out["shock_covariance"] - np.eye(n)).max() > 0.05
    return {
        "p": p, "n": n, "t_eff": data.shape[0] - p,
        "data": rows(data),
        "b": rows(out["params"]),
        "impact": rows(a0),
        "hd": [rows(h) for h in out["hd"]],
        "baseline": rows(out["baseline"]),
        "shocks": rows(out["E"]),
        "theta_head": [rows(t) for t in out["theta"][:4]],
        "shock_covariance": rows(out["shock_covariance"]),
        "adding_up_residual": out["adding_up_residual"],
    }


def main():
    fx = {
        "_meta": META,
        "_doc": (
            "Arbitrary-impact historical decomposition: statsmodels VAR(p, trend='c') "
            "reduced form + NumPy closed form (E = solve(A0, U')', Theta_s = Psi_s A0, "
            "HD = sum_s Theta_s E_{t-s}, identification-invariant baseline). Generated by "
            "fixtures/generate_historical_decomposition_impact_fixtures.py; never imports "
            "tsecon. Rust/Python goldens compare at rtol 1e-8 / atol 1e-10."
        ),
        "bq_var8": case_bq(),
        "random_3var_p2": case_random(),
    }
    OUT.write_text(json.dumps(fx, indent=1) + "\n")
    print("wrote", OUT)
    bq = fx["bq_var8"]
    print("  bq_var8 impact", bq["impact"])
    print("  hd[100][1][1] =", bq["hd"][100][1][1], " hd[193][0][0] =", bq["hd"][193][0][0])
    print("  shocks[0] =", bq["shocks"][0])
    print("  adding-up", bq["adding_up_residual"], fx["random_3var_p2"]["adding_up_residual"])


if __name__ == "__main__":
    main()
