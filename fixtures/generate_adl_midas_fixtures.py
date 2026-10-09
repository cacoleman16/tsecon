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

Data: seeded NumPy `default_rng` draws (no dataset).

This generator NEVER imports tsecon. Doubles are written with json's shortest
round-trip repr (serde_json `float_roundtrip` on the Rust side).

Run:  .venv/bin/python fixtures/generate_adl_midas_fixtures.py
"""

from __future__ import annotations

import json
import platform
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


def main() -> None:
    rng = np.random.default_rng(SEED)
    y1, x1 = simulate(rng, 160, 6, [0.5, -0.2], 0.7, 0.0)
    y2, x2 = simulate(rng, 275, 6, [0.4, -0.1, 0.15, -0.05], 0.3, 0.8)
    cases = [fit_case("n160_p2", y1, x1, 2), fit_case("n275_p4", y2, x2, 4)]
    assert cases[1]["default_maxlags"] == 4 and cases[1]["maxlags_rule_on_raw_n"] == 5
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
                "frequency lag 0 (the period's last observation)."
            ),
        },
        "cases": cases,
    }
    OUT.write_text(json.dumps(fixture, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")
    for c in cases:
        print(f"  {c['name']}: nobs={c['nobs']} default maxlags={c['default_maxlags']} "
              f"(raw-n rule {c['maxlags_rule_on_raw_n']}) params[0]={c['params'][0]!r} "
              f"R2={c['rsquared']!r} hac bse[1]={c['fits']['hac_default']['bse'][1]!r}")


if __name__ == "__main__":
    main()
