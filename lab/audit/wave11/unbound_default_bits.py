"""Bit-identity check behind the 0.11 claim that binding `dfm_smooth`,
`adl_midas` and `arima_fit(method="css")` moved no pre-existing number.

Dumps, as `float.hex()`, every number `arima_fit` (exact MLE: ARMA, ARIMA with
drift and drift_uncertainty bands, the airline SARIMA), `auto_arima`,
`dfm_nowcast` (both routes, ragged edge), `dfm_news` and `umidas` return at
their pre-0.11 arguments, keyed by the pre-0.11 key set (the two new
`arima_fit` keys, `method` and `nobs`, are excluded on purpose — they are
additions). Run it against a build of the parent and against this branch, and
compare byte for byte:

    PYTHONPATH=<parent build> python lab/audit/wave11/unbound_default_bits.py > base.json
    PYTHONPATH=<this build>   python lab/audit/wave11/unbound_default_bits.py > new.json
    cmp base.json new.json

Result recorded in out/unbound_default_bits.txt.
"""
import json
from pathlib import Path

import numpy as np

import tsecon

REPO = Path(__file__).resolve().parents[3]
NEW_KEYS = {"method", "nobs"}


def hexify(v):
    if isinstance(v, dict):
        return {str(k): hexify(x) for k, x in sorted(v.items()) if k not in NEW_KEYS}
    if isinstance(v, (list, tuple)):
        return [hexify(x) for x in v]
    if isinstance(v, np.ndarray):
        return [hexify(x) for x in v.tolist()]
    if isinstance(v, (bool, np.bool_)) or v is None or isinstance(v, str):
        return v if not isinstance(v, np.bool_) else bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    return float(v).hex()


def main() -> None:
    out = {}
    css = json.loads((REPO / "fixtures" / "arima_css.json").read_text())
    for c in css["cases"]:
        y = np.asarray(c["y"])
        p, d, q = c["order"]
        kw = dict(p=p, d=d, q=q, constant=c["constant"], forecast_steps=6, conf_alpha=0.1)
        if c["seasonal"]:
            kw["seasonal"] = tuple(c["seasonal"])
        out[f"arima_{c['name']}"] = hexify(tsecon.arima_fit(y, **kw))
    rw = np.cumsum(0.2 + np.random.default_rng(1).standard_normal(90))
    out["arima_drift"] = hexify(tsecon.arima_fit(rw, p=0, d=1, q=0, forecast_steps=8, conf_alpha=0.05,
                                                 drift_uncertainty=True))
    out["auto_arima"] = hexify(tsecon.auto_arima(np.asarray(css["cases"][0]["y"])[:200], max_p=2, max_q=2))

    rng = np.random.default_rng(11)
    T, N = 160, 7
    f = np.zeros(T)
    for t in range(2, T):
        f[t] = 0.6 * f[t - 1] - 0.15 * f[t - 2] + rng.standard_normal()
    x = np.outer(f, rng.uniform(0.5, 1.5, N)) + 0.5 * rng.standard_normal((T, N)) + rng.uniform(-3, 3, N)
    x[-1, :4] = np.nan
    x[-2, :2] = np.nan
    out["dfm_two_step"] = hexify(tsecon.dfm_nowcast(x, n_factors=2, factor_order=2))
    out["dfm_mle"] = hexify(tsecon.dfm_nowcast(x, n_factors=1, factor_order=2, method="mle"))
    new = x.copy()
    new[-1, :2] = x[-3, :2]
    out["dfm_news"] = hexify(tsecon.dfm_news(x, new, target_series=5))

    midas = json.loads((REPO / "fixtures" / "adl_midas.json").read_text())["cases"][0]
    y, hf = np.asarray(midas["y"]), np.asarray(midas["hf_lags"])
    out["umidas_hac"] = hexify(tsecon.umidas(y, hf))
    out["umidas_nonrobust"] = hexify(tsecon.umidas(y, hf, se_type="nonrobust"))
    print(json.dumps(out, sort_keys=True))


if __name__ == "__main__":
    main()
