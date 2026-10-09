"""specification-tests.md, DW 'Failure modes' paragraph (line ~697) says a long
sample 'is refused by the cap', but since 03de810 the binding returns the
statistic with pvalue=None above T = 4000 (as the COST paragraph, the
docstring and the CHANGELOG say). The card contradicts itself."""
import pathlib, re, numpy as np, tsecon
W = pathlib.Path(__file__).resolve().parents[4]
card = (W / "docs/reference/model-cards/specification-tests.md").read_text().splitlines()
for i, line in enumerate(card, 1):
    if "refused by the cap" in line or "A long sample is" in line:
        print(f"card line {i}: {line.strip()}")
rng = np.random.default_rng(0)
n = 4001
X = np.column_stack([np.ones(n), rng.normal(size=n)])
y = X @ [1.0, 1.0] + rng.normal(size=n)
r = tsecon.durbin_watson_test(y, X)          # expected per the failure-mode line: an exception
print("actual: no refusal ->", {k: r[k] for k in ("statistic", "pvalue", "nobs")})
