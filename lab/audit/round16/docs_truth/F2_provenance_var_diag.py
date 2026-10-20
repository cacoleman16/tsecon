"""F2: validation-matrix.md:246-257 ("The 0.11 generation") lists 13 fixtures as the
NumPy 2.5.3 / SciPy 1.18.1 / Python 3.13.16 generation and names var_diag among the
PRE-wave fixtures 'beside NumPy 2.4.6'. Batch 2 regenerated var_diag.json (and
spa.json) under NumPy 2.5.3; the table above the prose counts 15 such fixtures."""
import json, glob, os, re
W = "."
hits = []
for f in sorted(glob.glob(W + "/fixtures/*.json")):
    m = json.load(open(f)).get("_meta")
    if isinstance(m, dict) and any(re.search("numpy", k, re.I) and isinstance(v, str) and v.startswith("2.5.3") for k, v in m.items()):
        hits.append(os.path.basename(f))
print(len(hits), "fixtures record NumPy 2.5.3:", hits)
print("var_diag _meta:", {k: v for k, v in json.load(open(W + "/fixtures/var_diag.json"))["_meta"].items() if isinstance(v, str) and len(v) < 20})
prose = open(W + "/docs/reference/validation-matrix.md").read()
para = prose[prose.index("**The 0.11 generation.**"):].split("\n\n")[0]
listed = sorted({x for x in re.findall(r"`([a-z0-9_]+)\.json`", para.split("record NumPy 2.5.3")[0])})
print(len(listed), "named in the prose as the 0.11 generation:", listed)
print("in table count but not in prose:", sorted(set(hits) - {x + '.json' for x in listed}))
print("prose says pre-wave 'var_cf and var_diag, beside NumPy 2.4.6':", "`var_cf` and `var_diag`, beside NumPy 2.4.6" in para)
