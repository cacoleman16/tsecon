"""Every number the ARDL bounds-test model card, the validation matrix and
the CHANGELOG quote, reproduced from fixtures/ardl.json (statsmodels'
numbers) and, for the runtime and the runnable example, from the built
tsecon wheel. Run from the repo root:

    PYTHONPATH=<site-dir-with-the-wheel> python lab/ardl/card_numbers.py
"""
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[2]
FX = json.loads((ROOT / "fixtures" / "ardl.json").read_text())


def case(system, case_no, lags, order, selected=False):
    for c in FX["cases"]:
        if (c["system"], c["case"], c["lags_arg"], c["order_arg"], c["selected"]) == (
            system, case_no, lags, order, selected
        ):
            return c
    raise KeyError((system, case_no, lags, order))


s = FX["stories"]["macrodata_bic_case3"]
print("macrodata BIC case 3:", s["lags"], s["order"], "F = %.4f" % s["statistic"],
      "t = %.4f" % s["t_statistic"], "p_upper = %.4f" % s["p_value_upper"],
      "p_lower = %.4f" % s["p_value_lower"], "crit95 =", [round(v, 3) for v in s["crit95"]],
      "| statsmodels key: p_upper = %.4f" % s["sm_p_value_upper"], "crit95 =",
      [round(v, 3) for v in s["sm_crit95"]])
d = FX["stories"]["danish_bic_case3"]
print("danish BIC case 3: unrestricted", d["unrestricted"], "|", d["from_ardl"], "| admissible",
      d["admissible"], "BIC = %.3f" % d["ic_value"], "margin %.3f" % d["margin"])
c = case("danish", 3, 3, [3, 3, 3])
print("danish UECM(3,3,3,3) case 3: F = %.4f t = %.4f p_upper = %.5f p_lower = %.6f verdict %s"
      % (c["statistic"], c["t_statistic"], c["p_value_upper"], c["p_value_lower"], c["verdict"]),
      "| statsmodels key p_upper = %.5f" % c["canary_key"]["sm_p_value_upper"])
c = case("danish", 3, 2, [1, 3, 2])
print("danish (2; 1,3,2) case 3: nobs %d vs statsmodels %d; llf %.4f vs %.4f; df_resid %d vs %d"
      % (c["nobs"], c["sm_reported"]["nobs"], c["llf"], c["sm_reported"]["llf"], c["df_resid"],
         c["sm_reported"]["df_resid"]))
for sim in FX["canary_sim"]:
    print("statsmodels _pss_simulate at k = %d (n_x = %d), 95%%: %.3f / %.3f | table n_x row %.3f / %.3f"
          " | table n_x + 1 row %.3f / %.3f" % (sim["sim_key_k"], sim["n_x"], sim["sim_lower"][1],
                                              sim["sim_upper"][1], sim["table_k_lower95"],
                                              sim["table_k_upper95"], sim["table_k1_lower95"],
                                              sim["table_k1_upper95"]))
ex = max(abs(c["statistic"] - c["statistic_exact"]) / c["statistic_exact"] for c in FX["cases"])
lv = max(abs(c["statistic_levels"] - c["statistic_exact"]) / c["statistic_exact"] for c in FX["cases"])
print("statsmodels UECM Wald vs exact rationals: max rel %.2e; levels f_test: max rel %.2e" % (ex, lv))
print("smallest winner/runner-up IC margin over the searched cases: %.4f"
      % min(c["ic_margin"] for c in FX["cases"] if c["selected"]))
n_def = sum(c["lags"] < max(c["order"]) for c in FX["cases"])
print("cases:", len(FX["cases"]), "searched:", sum(c["selected"] for c in FX["cases"]),
      "with p < max q (nobs defect):", n_def)

try:
    import tsecon
except ImportError:  # pragma: no cover
    raise SystemExit("tsecon not importable: the runtime and example numbers need the wheel")

# the selection runtime at the guard: 4 x 4^7 = 65 536 candidate fits, and
# its (absent) growth in T: one factorisation of the largest design, then a
# T-free solve per candidate
rng = np.random.default_rng(7)
for k, T in ((5, 300), (6, 300), (7, 300), (7, 30000)):
    x = np.cumsum(rng.standard_normal((T, k)), axis=0)
    y = np.cumsum(rng.standard_normal(T))
    t0 = time.perf_counter()
    r = tsecon.ardl_bounds_test(y, x)
    print("default search, k_x = %d, T = %d: %d models in %.1f s (largest design %d x %d)"
          % (k, T, r["n_models"], time.perf_counter() - t0, T - 4, 1 + 4 + 5 * k))

# the card's runnable example
y = FX["systems"]["macrodata"]["y"]
x = FX["systems"]["macrodata"]["x"]
r = tsecon.ardl_bounds_test(np.asarray(y), np.asarray(x))
print("example:", r["lags"], r["order"], round(r["statistic"], 3), round(r["t_statistic"], 3),
      round(r["p_value_lower"], 4), round(r["p_value_upper"], 4), r["verdict"],
      dict(zip(r["long_run_names"], np.round(r["long_run"], 4))),
      dict(zip(r["long_run_names"], np.round(r["long_run_se"], 4))), round(r["adjustment"], 4))
