"""F4 (cosmetic): (a) help(tsecon.long_run_svar) says impact_signs indices outside `0..k`
are refused, the stub says `0..k-1`; (b) multivariate_normality's NaN refusal blames
'the VAR estimator ... would corrupt every coefficient' although it estimates nothing;
(c) lab/README.md's directory map lists the batch-2 lab/bq_bands/ but not the batch-2
lab/stepm_dh/ or lab/audit/wave11/ directories that the docs cite."""
import ast, re, numpy as np, tsecon, os
W = "."
stub = {n.name: ast.get_docstring(n) for n in ast.parse(open(W + "/bindings/python/python/tsecon/__init__.pyi").read()).body if isinstance(n, ast.FunctionDef)}
print("(a) runtime:", re.search(r"an index outside[^,]*", tsecon.long_run_svar.__doc__).group(0))
print("    stub   :", re.search(r"an index outside[^,]*", stub["long_run_svar"]).group(0))
x = np.random.default_rng(0).standard_normal((30, 2)); x[3, 1] = np.nan
try:
    tsecon.multivariate_normality(x)
except ValueError as e:
    print("(b)", e)
readme = open(W + "/lab/README.md").read()
for d in ("bq_bands/", "stepm_dh/", "wave11/"):
    print(f"(c) lab/{d:10s} exists={os.path.isdir(W + '/lab/' + d) or os.path.isdir(W + '/lab/audit/' + d)}  in lab/README map={d in readme}")
