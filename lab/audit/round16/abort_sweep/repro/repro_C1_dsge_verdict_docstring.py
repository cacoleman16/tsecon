"""dsge_solve's runtime docstring/stub promise a returned `verdict` of
"unique / indeterminate / no stable solution", but only the unique case
returns: an indeterminate or explosive model raises ValueError (the model card
dsge.md says the solve raises, so only help(tsecon.dsge_solve) misleads)."""
import numpy as np, tsecon
doc = " ".join(tsecon.dsge_solve.__doc__.split())
print("docstring:", doc[doc.index("Returns"):doc.index("solution).") + 10])
A = np.eye(2); B = np.array([[0.9, 0.1], [0.2, 1.5]]); C = np.array([[1.0], [0.0]])
for n_pre, regime in [(1, "unique"), (0, "indeterminate"), (2, "no stable solution")]:
    try:
        r = tsecon.dsge_solve(A, B, C, n_pre)
        print(f"n_predetermined={n_pre} ({regime}): returned verdict={r['verdict']!r}")
    except Exception as e:
        print(f"n_predetermined={n_pre} ({regime}): EXPECTED a returned verdict per the docstring; ACTUAL raised {type(e).__name__}: {str(e)[:100]}")
