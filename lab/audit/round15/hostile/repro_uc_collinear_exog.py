"""unobserved_components(mle_regression=False) accepts collinear regressors and returns a
diffuse period that never ends (nobs_diffuse == nobs) with finite "GLS" coefficients and a
finite regression_coefficients_cov for coefficients that are not identified.

The CHANGELOG/docstring promise a refusal naming `exog` "instead of returning a diffuse period
that never ends and roundoff coefficients" — but the guard only compares against the same model
on rescaled columns, which never ends its diffuse period either, so collinearity passes. An
all-zero column IS refused ("its coefficient is not identified"); a duplicated column, a scaled
copy, or a column of ones beside the (diffuse) local level is not. statsmodels returns the same
numbers (it has no guard either).

Expected: ValueError naming exog (collinear / not identified), as for the zero column.
"""
import numpy as np, tsecon
rng = np.random.default_rng(12345)
n = 80
y = np.cumsum(rng.standard_normal(n)) + rng.standard_normal(n)
x = rng.standard_normal((n, 1))
for lab, X in [("one column (control)", x), ("all-zero column", np.zeros((n, 1))),
               ("duplicated column [x, x]", np.column_stack([x, x])),
               ("scaled copy [x, 3x]", np.column_stack([x, 3 * x])),
               ("column of ones (+ diffuse level)", np.ones((n, 1)))]:
    try:
        r = tsecon.unobserved_components(y, exog=X, mle_regression=False, fixed_params=[1.0, 0.5])
        b = np.asarray(r["regression_coefficients"])[-1]
        print(f"{lab:34s} accepted: nobs={r['nobs']} nobs_diffuse={r['nobs_diffuse']} beta={np.round(b, 4)} "
              f"cov diag={np.round(np.diag(np.asarray(r['regression_coefficients_cov'])), 5)}")
    except ValueError as e:
        print(f"{lab:34s} refused: {str(e)[:110]}")
