"""Golden fixtures for the VAR forecast-error covariance that `var_forecast`
returns as `cov`, with and without Lütkepohl's parameter-uncertainty term
(`param_uncertainty=True`; crate `tsecon-var`, `forecast.rs`,
`VarResults::forecast_cov_with` / `forecast_interval_with`).

WHAT IS PINNED (Lütkepohl 2005, section 3.5, pp. 96-97)
-------------------------------------------------------
    MSE(h)  = sum_{i<h} Psi_i Sigma_u Psi_i'          Sigma_u = U'U / (T - m)
    cov(h)  = MSE(h) + Omega(h) / T                   T = nobs (NOT T - m)
    Omega(h)= sum_{i,j<h} tr[(B')^{h-1-i} G^{-1} B^{h-1-j} G] Psi_i Sigma_u Psi_j'

with m = n_trend + k p regressors per equation, G = Z'Z (Z rows
[1, y_{t-1}', ..., y_{t-p}'], no 1 for trend "n"), and the m x m matrix B whose
rows are e_1' (constant only), then params' (k rows), then
[0 | I_{k(p-1)} | 0]. Omega(1) = m Sigma_u exactly, so cov(1) = (1 + m/T) Sigma_u.

Reference legs, both from this venv:

  * statsmodels 0.15.0 `VAR(data).fit(p, trend)`: `res.mse(steps)` (the default
    `cov`) and, for trend "c", the public `res.forecast_cov(steps,
    method="auto")` (the `param_uncertainty=True` `cov`), its OutputWarning
    suppressed. statsmodels' `forecast_interval` is asserted to equal
    point -/+ z sqrt(diag mse) on every case.
  * trend "n": statsmodels' public `method="auto"` SILENTLY RETURNS `mse` — its
    `k_exog == 1` gate is false without a constant — so the golden there is
    `res.mse(steps) + res._omega_forc_cov(steps) / res.nobs` (a private
    method; version pinned in `_meta`). A QUIRK CANARY asserts statsmodels
    still drops the term, so a future fix there stops this generator rather
    than slipping past it.
  * Every Omega(h) — both trends — is re-derived by an independent NumPy
    transcription of the formula above (`omega_numpy`, using
    `res.endog_lagged`, `res.params`, `res.ma_rep`, `res.sigma_u`) and
    asserted to agree with statsmodels to 1e-12 relative before anything is
    stored. The NumPy leg takes the h = 1 term from the formula itself, not
    from statsmodels' `df_model * sigma_u` shortcut.

Data: the var.json system — 100 x dlog of real GDP, consumption and investment
built from statsmodels' bundled `macrodata` (transformations only; the series
is NOT stored again: the tests read `var.json`'s
`data_100dlog_gdp_cons_inv`, asserted bit-identical here). Growth rates are
used deliberately: cond(Z'Z) on these designs is 337-657, so a 1e-10 gate is
safe. A levels VAR (largest root ~0.998, cond(Z'Z) ~5e10) is stationary only
in name and is kept out of the golden; the model card states what the term
does there.

This generator NEVER imports tsecon. Doubles are written with json's shortest
round-trip repr, which the Rust golden tests parse to identical bits
(serde_json `float_roundtrip`).

Run:  .venv/bin/python fixtures/generate_var_forecast_cov_fixtures.py
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
from scipy import stats
from statsmodels.tsa.api import VAR

HERE = Path(__file__).resolve().parent
OUT = HERE / "var_forecast_cov.json"
STEPS = 12
ALPHAS = (0.05, 0.32)
CASES = [(2, "c"), (2, "n"), (1, "c"), (4, "c")]


def data() -> np.ndarray:
    mac = sm.datasets.macrodata.load_pandas().data
    y = 100.0 * np.diff(np.log(mac[["realgdp", "realcons", "realinv"]].to_numpy(dtype=float)), axis=0)
    stored = np.asarray(json.loads((HERE / "var.json").read_text(encoding="utf-8"))["data_100dlog_gdp_cons_inv"])
    assert np.array_equal(stored, y), "var.json data drifted from the macrodata transformation"
    return y


def omega_numpy(res, steps: int) -> np.ndarray:
    """Lütkepohl (2005) eq. 3.5.13 transcribed: independent of statsmodels'
    `_bmat_forc_cov` / `_omega_forc_cov` (its own B, its own G, h = 1 from the
    formula, not the df_model * sigma_u shortcut)."""
    Z = np.asarray(res.endog_lagged)
    T, m = Z.shape
    k = res.neqs
    p = res.k_ar
    n_trend = m - k * p
    G = Z.T @ Z
    Gi = np.linalg.inv(G)
    B = np.zeros((m, m))
    if n_trend == 1:
        B[0, 0] = 1.0
    B[n_trend:n_trend + k, :] = np.asarray(res.params).T
    for i in range(k * (p - 1)):
        B[n_trend + k + i, n_trend + i] = 1.0
    psi = np.asarray(res.ma_rep(steps))
    sig = np.asarray(res.sigma_u)
    powers = [np.linalg.matrix_power(B, a) for a in range(steps)]
    out = np.zeros((steps, k, k))
    for h in range(1, steps + 1):
        acc = np.zeros((k, k))
        for i in range(h):
            for j in range(h):
                tau = np.trace(powers[h - 1 - i].T @ Gi @ powers[h - 1 - j] @ G)
                acc += tau * psi[i] @ sig @ psi[j].T
        out[h - 1] = acc
    return out


def rel_dev(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


def build_case(y: np.ndarray, p: int, trend: str) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = VAR(y).fit(p, trend=trend)
        mse = np.asarray(res.mse(STEPS))
        auto = np.asarray(res.forecast_cov(STEPS, method="auto"))
        om_sm = np.asarray(res._omega_forc_cov(STEPS))
        point, lo_sm, up_sm = res.forecast_interval(y[-p:], STEPS, alpha=0.05)
    nobs = res.nobs
    m = res.df_model
    om_np = omega_numpy(res, STEPS)
    dev_omega = rel_dev(om_np, om_sm)
    assert dev_omega < 1e-12, (p, trend, dev_omega)
    if trend == "c":
        cov = auto
        dev_auto = rel_dev(auto, mse + om_sm / nobs)
        assert dev_auto < 1e-14, (p, trend, dev_auto)
        source = "statsmodels VARResults.forecast_cov(steps, method='auto')"
    else:
        # QUIRK CANARY: statsmodels' public 'auto' drops Omega without a constant.
        assert np.array_equal(auto, mse), (
            "statsmodels forecast_cov(method='auto') now adds the parameter-uncertainty "
            "term for trend='n' — the quirk this fixture documents is fixed; switch the "
            "trend='n' golden to the public method and update the model card"
        )
        cov = mse + om_sm / nobs
        dev_auto = None
        source = ("statsmodels VARResults.mse(steps) + VARResults._omega_forc_cov(steps) / nobs "
                  "(public method='auto' drops the term for trend='n')")
    sigma_u = np.asarray(res.sigma_u)
    # Exact h = 1 identity, checked here on the reference side too.
    dev_h1 = rel_dev(cov[0], (1.0 + m / nobs) * sigma_u)
    assert dev_h1 < 1e-13, (p, trend, dev_h1)
    point = np.asarray(point)
    se_mse = np.sqrt(np.einsum("hjj->hj", mse))
    se_cov = np.sqrt(np.einsum("hjj->hj", cov))
    z95 = stats.norm.ppf(0.975)
    assert np.max(np.abs(np.asarray(lo_sm) - (point - z95 * se_mse))) < 1e-12
    assert np.max(np.abs(np.asarray(up_sm) - (point + z95 * se_mse))) < 1e-12
    bands = {}
    for alpha in ALPHAS:
        z = stats.norm.ppf(1.0 - alpha / 2.0)
        bands[str(alpha)] = {
            "z": float(z),
            "lower_mse": (point - z * se_mse).tolist(),
            "upper_mse": (point + z * se_mse).tolist(),
            "lower_param": (point - z * se_cov).tolist(),
            "upper_param": (point + z * se_cov).tolist(),
        }
    return {
        "name": f"var{p}{trend}",
        "lags": p,
        "trend": trend,
        "steps": STEPS,
        "nobs": int(nobs),
        "df_model": int(m),
        "cond_zz": float(np.linalg.cond(np.asarray(res.endog_lagged).T @ np.asarray(res.endog_lagged))),
        "sigma_u": sigma_u.tolist(),
        "point": point.tolist(),
        "mse": mse.tolist(),
        "omega": om_sm.tolist(),
        "cov_param": cov.tolist(),
        "bands": bands,
        "cov_param_source": source,
        "max_rel_dev": {
            "omega_numpy_vs_statsmodels": dev_omega,
            "auto_vs_mse_plus_omega_over_T": dev_auto,
            "h1_vs_(1+m/T)sigma_u": dev_h1,
        },
    }


def main() -> None:
    y = data()
    cases = [build_case(y, p, tr) for p, tr in CASES]
    fixture = {
        "_meta": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "statsmodels": statsmodels.__version__,
            "statsmodels_quirk_trend_n_auto_drops_omega": True,
            "note": (
                "VAR forecast-error covariance: mse = statsmodels VARResults.mse; "
                "cov_param = mse + Omega(h)/nobs (Lutkepohl 2005 pp. 96-97) — statsmodels "
                "forecast_cov(method='auto') for trend 'c', mse + _omega_forc_cov/nobs for "
                "trend 'n' (where the public method drops the term). Omega re-derived in "
                "NumPy and asserted equal to statsmodels at 1e-12 relative. Data: var.json's "
                "data_100dlog_gdp_cons_inv (macrodata transformation), not stored again."
            ),
        },
        "data_ref": "var.json:data_100dlog_gdp_cons_inv",
        "cases": cases,
    }
    OUT.write_text(json.dumps(fixture, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")
    for c in cases:
        d = c["max_rel_dev"]
        ratio = np.diag(np.asarray(c["cov_param"])[-1]) / np.diag(np.asarray(c["mse"])[-1])
        print(f"  {c['name']}: nobs={c['nobs']} m={c['df_model']} cond(Z'Z)={c['cond_zz']:.0f} "
              f"omega np-vs-sm={d['omega_numpy_vs_statsmodels']:.1e} "
              f"h1={d['h1_vs_(1+m/T)sigma_u']:.1e} cov/mse diag h12={np.round(ratio, 4)}")
    c0 = cases[0]
    print("  var2c mse[7][0][0] =", c0["mse"][7][0][0], " cov[7][0][0] =", c0["cov_param"][7][0][0],
          " cov[0][2][2] =", c0["cov_param"][0][2][2])
    print("  var2n cov h8 diag =", np.diag(np.asarray(cases[1]["cov_param"])[7]))


if __name__ == "__main__":
    main()
