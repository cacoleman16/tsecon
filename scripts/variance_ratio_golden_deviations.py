"""Achieved deviations of tsecon.variance_ratio from its golden fixture.

The golden tests (crates/tsecon-diag/tests/variance_ratio_golden.rs and
bindings/python/tests/test_variance_ratio.py) assert the tolerances; this
script reports how far inside them the shipped build actually lands, for the
"Validated against" paragraph of the variance-ratio model card, the
validation-matrix row and the CHANGELOG, which quote its output verbatim.

    python scripts/variance_ratio_golden_deviations.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import tsecon

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "variance_ratio.json"


def main() -> None:
    fx = json.loads(FIXTURE.read_text(encoding="utf-8"))
    series = {k: np.asarray(v, dtype=float) for k, v in fx["series"].items()}
    worst = {"vr": 0.0, "stat": 0.0, "pvalue": 0.0, "stat_variance": 0.0, "cd_pvalue": 0.0}
    zmax, pmin, n_arch_zero = 0.0, 1.0, 0
    for c in fx["cases"]:
        r = tsecon.variance_ratio(series[c["series"]], c["lags"], trend=c["trend"],
                                  overlap=c["overlap"], debiased=c["debiased"],
                                  robust=c["robust"])
        worst["vr"] = max(worst["vr"], abs(r["vr"] / c["vr"] - 1))
        worst["stat"] = max(worst["stat"],
                            abs(r["stat"] - c["stat"]) / (1e-12 + abs(c["stat"])))
        worst["pvalue"] = max(worst["pvalue"], abs(r["pvalue"] / c["pvalue_sf"] - 1))
        worst["stat_variance"] = max(worst["stat_variance"],
                                     abs(r["stat_variance"] / c["stat_variance"] - 1))
        zmax, pmin = max(zmax, abs(r["stat"])), min(pmin, r["pvalue"])
        n_arch_zero += c["pvalue_arch"] == 0.0
    for c in fx["chow_denning"]:
        r = tsecon.variance_ratio(series[c["series"]], c["lags"], trend=c["trend"],
                                  overlap=c["overlap"], debiased=c["debiased"],
                                  robust=c["robust"])
        worst["cd_pvalue"] = max(worst["cd_pvalue"], abs(r["cd_pvalue"] / c["cd_pvalue"] - 1))
        zmax, pmin = max(zmax, r["cd_stat"]), min(pmin, r["cd_pvalue"])
    print(f"tsecon {tsecon.__version__}: {len(fx['cases'])} single-horizon cases vs arch "
          f"{fx['_meta']['arch']}, {len(fx['chow_denning'])} Chow-Denning cases")
    print(f"  max relative deviation  vr {worst['vr']:.2e}  "
          f"stat {worst['stat']:.2e} (1e-12-guarded)  "
          f"stat_variance {worst['stat_variance']:.2e}")
    print(f"  max relative deviation  pvalue vs 2*norm.sf {worst['pvalue']:.2e}  "
          f"cd_pvalue {worst['cd_pvalue']:.2e}")
    print(f"  largest |stat| {zmax:.2f}, smallest p-value {pmin:.3e}; "
          f"arch's own p is 0.0 in {n_arch_zero} single-horizon cases")


if __name__ == "__main__":
    main()
