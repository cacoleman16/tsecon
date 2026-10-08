"""Bit-identity check behind the 0.11 claim that adding `cov` /
`param_uncertainty` to `var_forecast` and `se_type="ewc"` / `ewc_b` to `ols`
moved no pre-existing number.

Dumps, as `float.hex()`, every number of `var_forecast` (VAR(1)c, VAR(2)c,
VAR(2)n, VAR(4)c on the var.json system; steps 1/8/12; alpha 0.05/0.32; and a
sup-t band) and of `ols` under every pre-0.11 `se_type`, at their pre-0.11
arguments. Run it once against a build of the parent commit and once against
this branch, and compare the two outputs byte for byte:

    PYTHONPATH=<parent build> python lab/audit/wave11/default_path_bits.py > base.json
    PYTHONPATH=<this build>   python lab/audit/wave11/default_path_bits.py > new.json
    cmp base.json new.json

Result for the 0.11 EWC / VAR-covariance branch against its parent
(e38965e): identical (`out/default_path_bits.txt`).
"""
import json
from pathlib import Path

import numpy as np

import tsecon

REPO = Path(__file__).resolve().parents[3]


def main() -> None:
    data = np.asarray(json.loads((REPO / "fixtures" / "var.json").read_text())["data_100dlog_gdp_cons_inv"])
    rng = np.random.default_rng(3)
    out = {}
    for p, tr in [(1, "c"), (2, "c"), (2, "n"), (4, "c")]:
        for steps in (1, 8, 12):
            for alpha in (0.05, 0.32):
                r = tsecon.var_forecast(data, lags=p, steps=steps, alpha=alpha, trend=tr)
                out[f"{p}{tr}{steps}{alpha}"] = [
                    float(v).hex() for k in ("point", "lower", "upper") for v in np.ravel(r[k])
                ]
        r = tsecon.var_forecast(data, lags=p, steps=6, trend=tr, band="sup-t", band_n_sim=5000)
        out[f"{p}{tr}supt"] = [
            float(v).hex()
            for k in ("point", "lower", "upper", "se", "sim_lower", "sim_upper")
            for v in np.ravel(r[k])
        ]
    X = np.column_stack([np.ones(200), rng.standard_normal((200, 2))])
    y = X @ [1, 2, 3] + rng.standard_normal(200)
    for se in ("nonrobust", "hc0", "hc1", "hc2", "hc3", "hac"):
        r = tsecon.ols(y, X, se_type=se)
        out["ols" + se] = [float(v).hex() for k in ("params", "bse", "tvalues") for v in r[k]]
    print(json.dumps(out, sort_keys=True))


if __name__ == "__main__":
    main()
