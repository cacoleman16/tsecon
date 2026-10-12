"""ols: with an unknown se_type AND ewc_b, the inert-ewc_b check runs before se_type is
validated, so a typo is reported as if it were a real covariance choice and the message never
says the se_type is unknown. Without ewc_b the same typo gets 'unknown se_type'.
"""
import numpy as np, tsecon
rng = np.random.default_rng(1)
T = 100
X = np.column_stack([np.ones(T), rng.standard_normal(T)])
y = X @ [1.0, 2.0] + rng.standard_normal(T)
for kw in [dict(se_type="ewx"), dict(se_type="ewx", ewc_b=5), dict(se_type="EWC", ewc_b=5)]:
    try:
        tsecon.ols(y, X, **kw)
    except ValueError as e:
        print(kw, "->", str(e)[:170])
