"""Wrong-type message blames an innocent int argument whose default is None.

CHANGELOG [Unreleased] Fixed: the input layer's wrong-type message "no longer also blames an
integer count that sits beside an integer handed to a None-default flag". The fix only drops
an int whose DEFAULT is itself an int; the new count keywords of this wave (autoregressive=,
ewc_b=, nlags=, max_lag=) default to None, so they are still blamed.
"""
import numpy as np, tsecon
rng = np.random.default_rng(0)
y = np.cumsum(rng.standard_normal(80)) + rng.standard_normal(80)
X = rng.standard_normal((80, 1))
def show(lab, f):
    try:
        f(); print(f"{lab}: accepted")
    except TypeError as e:
        print(f"{lab}:\n   {str(e)[:220]}")
show("UC autoregressive=1 (valid) + mle_regression=1 (wrong type)",
     lambda: tsecon.unobserved_components(y, exog=X, autoregressive=1, mle_regression=1))
show("UC seasonal=4 (valid) + mle_regression=1 (wrong type)",
     lambda: tsecon.unobserved_components(y, exog=X, seasonal=4, mle_regression=1))
show("UC forecast_steps=4 (int default) + mle_regression=1 [the fixed case]",
     lambda: tsecon.unobserved_components(y, exog=X, forecast_steps=4, forecast_exog=np.ones((4,1)), mle_regression=1))
show("UC autoregressive=1 (valid) + cycle_init=1 (wrong type, str expected)",
     lambda: tsecon.unobserved_components(y, cycle=True, damped_cycle=True, stochastic_cycle=True, autoregressive=1, cycle_init=1))
show("variance_ratio lags=4 + debiased=1 [the fixed case]",
     lambda: tsecon.variance_ratio(y, lags=4, debiased=1))
ay = np.cumsum(rng.standard_normal(120)); ax = np.cumsum(rng.standard_normal((120, 1)), axis=0)
show("ardl lags=2 (valid) + ic=1 (wrong type, str expected)",
     lambda: tsecon.ardl_bounds_test(ay, ax, lags=2, ic=1))
show("ardl max_lag=3 (valid) + max_order=2 (valid) + ic=1 (wrong type)",
     lambda: tsecon.ardl_bounds_test(ay, ax, max_lag=3, max_order=2, ic=1))
T = 100; XX = np.column_stack([np.ones(T), rng.standard_normal(T)]); yy = XX @ [1.0, 2.0] + rng.standard_normal(T)
show("ols ewc_b=5 (valid) + se_type=1 (wrong type)",
     lambda: tsecon.ols(yy, XX, se_type=1, ewc_b=5))
show("breusch_godfrey_test nlags=2 positional ok; durbin alternative=1",
     lambda: tsecon.durbin_watson_test(yy, XX, alternative=1))
