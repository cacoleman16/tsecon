"""Distributed-lag temperature regressions on the Dell-Jones-Olken panel.

Runs `tsecon.panel_distributed_lag` on the replication data of Dell, Jones &
Olken (2012, AEJ:Macro 4(3)), `climate_panel.dta` (population-weighted
temperature `wtem`, WDI GDP in constant local currency `gdpLCU`, 1950-2006).
The AEA data archive and the authors' Stanford/MIT hosts are unreachable
through the build container's proxy; the file was fetched from a public
GitHub course mirror (ewibbels/ps750, Week5_Geography) whose variable list
is byte-for-byte DJO's. Pass a local path as the first argument to use your
own copy; nothing is committed to the repository.

    .venv/bin/python docs/examples/panel_distributed_lag_djo.py [climate_panel.dta]

What is estimated, and how it differs from DJO's published tables:

  * growth g_it = 100 * (log gdpLCU_it - log gdpLCU_{i,t-1});
  * countries observed in EVERY year of the window 1971-2003 (growth needs
    1970) are kept -- a BALANCED SUBSAMPLE, because the panel crate refuses
    unbalanced panels. DJO use the full unbalanced panel;
  * country and year effects, standard errors clustered by country. DJO
    use region x year and poor x year effects and interact temperature
    with an initial-poverty dummy; none of that is reproduced here, so the
    numbers below are an illustration of the estimator on real data, not a
    replication of DJO's coefficients.

Specifications printed: the linear DJO form at L = 0 and L = 3 (DJO's
Table 3 uses up to 10 lags), and the BHM quadratic form at L = 0 and L = 3
with the marginal effect at 10 / 20 / 30 degrees and the turning point.
"""
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import tsecon

URL = "https://raw.githubusercontent.com/ewibbels/ps750/master/Week5_Geography/climate_panel.dta"
WINDOW = (1971, 2003)


def load(path: Path | None) -> pd.DataFrame:
    if path is None:
        path = Path(__file__).with_name("climate_panel.dta")
        if not path.exists():
            urllib.request.urlretrieve(URL, path)
    d = pd.read_stata(path)[["fips60_06", "year", "wtem", "gdpLCU"]].copy()
    d = d.sort_values(["fips60_06", "year"])
    d["lgdp"] = np.log(d["gdpLCU"])
    d["g"] = 100.0 * d.groupby("fips60_06")["lgdp"].diff()
    return d


def balanced_panel(d: pd.DataFrame):
    lo, hi = WINDOW
    w = d[(d["year"] >= lo) & (d["year"] <= hi)].dropna(subset=["g", "wtem"])
    counts = w.groupby("fips60_06")["year"].nunique()
    keep = counts[counts == hi - lo + 1].index
    w = w[w["fips60_06"].isin(keep)]
    g = w.pivot(index="fips60_06", columns="year", values="g").to_numpy(dtype=float)
    temp = w.pivot(index="fips60_06", columns="year", values="wtem").to_numpy(dtype=float)
    return g, temp, list(keep), len(d.dropna(subset=["g", "wtem"]))


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    d = load(path)
    g, temp, countries, n_available = balanced_panel(d)
    N, T = g.shape
    print(f"balanced subsample: N = {N} countries, T = {T} years ({WINDOW[0]}-{WINDOW[1]}), "
          f"{N * T} country-years of {n_available} with growth and temperature in the file")
    print(f"temperature: mean {temp.mean():.2f} C, sd {temp.std():.2f}; growth: mean {g.mean():.2f}, sd {g.std():.2f}")
    for L in (0, 3):
        r = tsecon.panel_distributed_lag(g, temp[None], lags=L, se_type="cluster")
        b0, s0 = r["lag_effects"][0][0][0], r["lag_se"][0][0][0]
        B, S = r["cumulative_effect"][0][0], r["cumulative_se"][0][0]
        lo, hi = r["cumulative_ci_low"][0][0], r["cumulative_ci_high"][0][0]
        print(f"linear  L={L}: beta_0 = {b0:+.3f} ({s0:.3f})  cumulative = {B:+.3f} ({S:.3f})  "
              f"95% [{lo:+.3f}, {hi:+.3f}]  nobs = {r['nobs']}")
    for L in (0, 3):
        r = tsecon.panel_distributed_lag(g, temp[None], lags=L, powers=2, se_type="cluster",
                                         eval_points=[10.0, 20.0, 30.0])
        B1, B2 = r["cumulative_effect"][0]
        S1, S2 = r["cumulative_se"][0]
        tp, tps = r["turning_point"][0], r["turning_point_se"][0]
        me = ", ".join(f"{x:.0f}C {m:+.3f} ({s:.3f})" for x, m, s in
                       zip(r["eval_points"][0], r["marginal_effect"][0], r["marginal_se"][0]))
        print(f"quadratic L={L}: B1 = {B1:+.3f} ({S1:.3f})  B2 = {B2:+.4f} ({S2:.4f})  "
              f"turning point = {tp:.1f} C ({tps:.1f})  marginal: {me}")


if __name__ == "__main__":
    main()
