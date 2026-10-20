"""F1: dfm_smooth's documented achieved agreement ("states and covariances within
6.0e-11", CHANGELOG line 39; card nowcasting-midas.md:272; validation-matrix.md:137;
docstring "achieved 5e-13 and 6e-11") is measured over the two RAW cases only.
The third pinned case, standardized_r1_p3 (named in the same sentences), deviates
by 2.6e-10 (states) and 1.2e-10 (covariances) -- ~4x the quoted figure."""
import json, numpy as np, tsecon
FX = "./fixtures/dfm_smooth.json"
worst_s = worst_c = worst_l = 0.0
for c in json.load(open(FX))["cases"]:
    kw = {k: c[k] for k in ("center", "scale") if k in c}
    r = tsecon.dfm_smooth(np.asarray(c["panel"], float), c["loadings"], c["factor_ar"],
                          c["idiosyncratic"], factor_cov=c["factor_cov"], **kw)
    dl = abs(r["loglik"] - c["loglik"]) / abs(c["loglik"])
    ds = np.max(np.abs(r["smoothed_state"] - np.asarray(c["smoothed_state"])))
    dc = np.max(np.abs(r["smoothed_state_cov"] - np.asarray(c["smoothed_state_cov"])))
    print(f"{c['name']:22s} loglik rel {dl:.1e}  state abs {ds:.1e}  cov abs {dc:.1e}")
    worst_l, worst_s, worst_c = max(worst_l, dl), max(worst_s, ds), max(worst_c, dc)
print(f"documented: loglik 5.2e-13, states and covariances 6.0e-11")
print(f"measured over all three pinned cases: loglik {worst_l:.1e}, states {worst_s:.1e}, covariances {worst_c:.1e}")
