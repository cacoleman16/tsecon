"""Golden fixtures for the VAR residual diagnostics `var_diagnostics` and
the lag-order selection binding `var_select_order` (crate `tsecon-var`,
module `diagnostics.rs` / `select.rs`).

Reference: statsmodels `VARResults` (independent package) —

  * `test_whiteness(nlags, adjusted=False)` and `adjusted=True`: the
    multivariate Portmanteau statistics Q_h / adjusted Q_h, their
    k^2 (nlags - p) degrees of freedom and chi-squared p-values
    (Lütkepohl 2005, section 4.4.3);
  * `test_normality()`: the multivariate Jarque-Bera omnibus statistic,
    its 2k degrees of freedom and p-value (statsmodels orthogonalises the
    centred residuals with the LOWER CHOLESKY factor of resid'resid / T);
  * `roots` (moduli, sorted descending — the reciprocal characteristic
    roots) and `is_stable()`;
  * `VAR.select_order(maxlags)`: the AIC / BIC / HQIC / FPE table on the
    common sample and the selected orders.

The skewness and kurtosis COMPONENTS of the normality test (`lam_skew`,
`lam_kurt`, and the per-component `b1`, `b2`) are not returned by
statsmodels; they are recomputed here by transcribing the lines of
statsmodels' own `test_normality` (documented formula, same
orthogonalisation), and `lam_skew + lam_kurt` is asserted equal to the
returned omnibus statistic before anything is stored.

Data: the seeded simulated VAR(2) of var_cf.json (k = 3; the same
generator, the same seed, stored again so each fixture is self-contained),
a second simulated VAR(2) driven by Student-t(4) innovations so the
normality test has something to reject, and the macro system of
var_cf.json (100 dlog realgdp, 400 dlog cpi, diff tbilrate —
transformations of statsmodels' bundled macrodata, not the dataset).

This generator NEVER imports tsecon. Doubles are written with json's
shortest round-trip repr, which the Rust golden tests parse to identical
bits (serde_json `float_roundtrip`).

Run:  .venv/bin/python fixtures/generate_var_diag_fixtures.py
"""

from __future__ import annotations

import json
import platform
import sys
import warnings
from pathlib import Path

import numpy as np
import scipy
import statsmodels
from scipy import stats
from statsmodels.tsa.api import VAR

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_var_cf_fixtures import macro_system, sim_var  # noqa: E402

OUT = Path(__file__).resolve().parent / "var_diag.json"


def sim_var_t(seed, T, c, a_list, chol, df, burn=100):
    """VAR(p) with Student-t(df) innovations (scaled to unit variance)."""
    rng = np.random.default_rng(seed)
    k = len(c)
    p = len(a_list)
    n = T + burn + p
    y = np.zeros((n, k))
    c = np.asarray(c)
    a_list = [np.asarray(a) for a in a_list]
    chol = np.asarray(chol)
    scale = np.sqrt((df - 2.0) / df)
    for t in range(p, n):
        v = c.copy()
        for i, a in enumerate(a_list, start=1):
            v = v + a @ y[t - i]
        y[t] = v + chol @ (scale * rng.standard_t(df, size=k))
    return y[burn + p:]


def normality_components(res):
    """Transcription of statsmodels.tsa.vector_ar.var_model.test_normality."""
    resid_c = np.asarray(res.resid) - np.asarray(res.resid).mean(0)
    sig = resid_c.T @ resid_c / res.nobs
    Pinv = np.linalg.inv(np.linalg.cholesky(sig))
    w = Pinv @ resid_c.T
    b1 = (w ** 3).sum(1) / res.nobs
    b2 = (w ** 4).sum(1) / res.nobs - 3
    lam_skew = float(res.nobs * (b1 @ b1) / 6)
    lam_kurt = float(res.nobs * (b2 @ b2) / 24)
    return b1, b2, lam_skew, lam_kurt


def diag_case(name, series_name, y, p, trend, nlags_list):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = VAR(y).fit(p, trend=trend)
    k = res.neqs
    port = []
    for nlags in nlags_list:
        un = res.test_whiteness(nlags=nlags, adjusted=False)
        ad = res.test_whiteness(nlags=nlags, adjusted=True)
        assert un.df == ad.df == k * k * (nlags - p)
        port.append({
            "nlags": nlags,
            "statistic": float(un.test_statistic),
            "pvalue": float(un.pvalue),
            "adjusted": float(ad.test_statistic),
            "adjusted_pvalue": float(ad.pvalue),
            "df": int(un.df),
        })
    norm = res.test_normality()
    b1, b2, lam_skew, lam_kurt = normality_components(res)
    assert abs(lam_skew + lam_kurt - float(norm.test_statistic)) < 1e-10 * max(1.0, float(norm.test_statistic))
    roots = np.abs(np.asarray(res.roots))
    assert np.all(np.diff(roots) <= 1e-12)  # statsmodels sorts descending
    return {
        "name": name,
        "series": series_name,
        "p": p,
        "trend": trend,
        "nobs": int(res.nobs),
        "portmanteau": port,
        "normality": {
            "statistic": float(norm.test_statistic),
            "pvalue": float(norm.pvalue),
            "df": int(norm.df),
            "skewness": lam_skew,
            "skewness_pvalue": float(stats.chi2.sf(lam_skew, k)),
            "kurtosis": lam_kurt,
            "kurtosis_pvalue": float(stats.chi2.sf(lam_kurt, k)),
            "skewness_components": b1.tolist(),
            "kurtosis_components": b2.tolist(),
        },
        "roots": roots.tolist(),
        "eigenvalue_moduli": np.sort(1.0 / roots)[::-1].tolist(),
        "is_stable": bool(res.is_stable()),
    }


def select_case(series_name, y, maxlags, trend):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sel = VAR(y).select_order(maxlags=maxlags, trend=trend)
    p_min = 1 if trend == "n" else 0
    return {
        "series": series_name,
        "max_lags": maxlags,
        "trend": trend,
        "candidates": list(range(p_min, maxlags + 1)),
        "aic_values": [float(v) for v in sel.ics["aic"]],
        "bic_values": [float(v) for v in sel.ics["bic"]],
        "hqic_values": [float(v) for v in sel.ics["hqic"]],
        "fpe_values": [float(v) for v in sel.ics["fpe"]],
        "selected": {kk: int(v) for kk, v in sel.selected_orders.items()},
    }


def main():
    chol = [[0.8, 0.0, 0.0], [0.3, 0.6, 0.0], [-0.2, 0.25, 0.5]]
    a_list = [[[0.5, 0.1, 0.0], [0.0, 0.4, 0.1], [0.1, 0.0, 0.3]],
              [[0.1, 0.0, 0.05], [0.0, 0.1, 0.0], [0.05, 0.0, 0.1]]]
    y = sim_var(20260910, 300, c=[0.5, -0.2, 0.1], a_list=a_list, chol=chol)
    y_t = sim_var_t(20260912, 300, c=[0.5, -0.2, 0.1], a_list=a_list, chol=chol, df=4)
    macro = macro_system()

    cases = [
        diag_case("var2c_sim", "sim_var2_k3", y, 2, "c", [10, 6, 3]),
        diag_case("var3n_sim", "sim_var2_k3", y, 3, "n", [10, 5]),
        diag_case("var1c_sim_underfit", "sim_var2_k3", y, 1, "c", [10, 4]),
        diag_case("var2c_sim_t4", "sim_var2_k3_t4", y_t, 2, "c", [10]),
        diag_case("macro_var2c", "macro_growth_infl_dtbill", macro, 2, "c", [12, 8]),
        diag_case("macro_var4c", "macro_growth_infl_dtbill", macro, 4, "c", [12]),
    ]
    selects = [
        select_case("sim_var2_k3", y, 8, "c"),
        select_case("sim_var2_k3", y, 6, "n"),
        select_case("macro_growth_infl_dtbill", macro, 8, "c"),
    ]
    fixture = {
        "_meta": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "statsmodels": statsmodels.__version__,
            "python": platform.python_version(),
            "note": (
                "VAR residual diagnostics from statsmodels VARResults: "
                "test_whiteness (adjusted False/True), test_normality (omnibus; "
                "the skewness/kurtosis components transcribed from the same "
                "code), roots moduli, is_stable; VAR.select_order tables. "
                "Macro series are transformations of statsmodels' macrodata."
            ),
        },
        "series": {
            "sim_var2_k3": y.tolist(),
            "sim_var2_k3_t4": y_t.tolist(),
            "macro_growth_infl_dtbill": macro.tolist(),
        },
        "cases": cases,
        "select_order": selects,
    }
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(fixture, fh, indent=1)
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes); {len(cases)} diagnostic cases, "
          f"{len(selects)} select_order cases")
    for c in cases:
        q = c["portmanteau"][0]
        n = c["normality"]
        print(f"  {c['name']:22s} p={c['p']} {c['trend']} T={c['nobs']} "
              f"Q({q['nlags']})={q['statistic']:.3f} p={q['pvalue']:.4f} "
              f"adj={q['adjusted']:.3f} p={q['adjusted_pvalue']:.4f} | "
              f"JB={n['statistic']:.3f} p={n['pvalue']:.4f} "
              f"(skew {n['skewness']:.3f}, kurt {n['kurtosis']:.3f}) | "
              f"min root={c['roots'][-1]:.4f} stable={c['is_stable']}")
    for s in selects:
        print(f"  select {s['series']:24s} maxlags={s['max_lags']} {s['trend']} -> {s['selected']}")


if __name__ == "__main__":
    main()
