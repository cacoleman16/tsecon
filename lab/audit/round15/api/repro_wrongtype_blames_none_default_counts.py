"""The shared coercion layer's wrong-type message still blames innocent integer
counts whose default is None -- every integer count of ardl_bounds_test
(lags, max_lag, max_order) and the new unobserved_components(autoregressive=).
CHANGELOG [Unreleased] / Fixed: 'no longer also blames an integer count that
sits beside an integer handed to a None-default flag'. The fix only exempts
counts whose DEFAULT is an int (n_starts=3, horizon=5), so the 0.11 surface,
whose counts default to None, still gets the old misattribution."""
import numpy as np, tsecon
rng = np.random.default_rng(0)
x = np.cumsum(rng.standard_normal((200, 2)), axis=0)
y = 0.5 * x[:, 0] + rng.standard_normal(200)
yuc = np.cumsum(rng.standard_normal(120)) + rng.standard_normal(120)
ex = rng.standard_normal((120, 1))
cases = [
    ("ardl_bounds_test(max_lag=3, max_order=2, ic=1)", "ic", lambda: tsecon.ardl_bounds_test(y, x, max_lag=3, max_order=2, ic=1)),
    ("ardl_bounds_test(lags=2, ic=3)", "ic", lambda: tsecon.ardl_bounds_test(y, x, lags=2, ic=3)),
    ("unobserved_components(exog=ex, autoregressive=1, mle_regression=1)", "mle_regression", lambda: tsecon.unobserved_components(yuc, exog=ex, autoregressive=1, mle_regression=1)),
    ("unobserved_components(autoregressive=2, cycle=True, ..., cycle_init=1)", "cycle_init", lambda: tsecon.unobserved_components(yuc, autoregressive=2, cycle=True, damped_cycle=True, stochastic_cycle=True, cycle_init=1)),
    ("control (int-default count): unobserved_components(exog=ex, n_starts=2, mle_regression=1)", "mle_regression", lambda: tsecon.unobserved_components(yuc, exog=ex, n_starts=2, mle_regression=1)),
]
for label, offender, f in cases:
    try:
        f(); print(label, "-> ACCEPTED (unexpected)")
    except TypeError as e:
        head = str(e).split(" is of type")[0].split(" are of type")[0]
        named = [tok.split("=")[0] for tok in head.split(": ", 1)[1].split(" and ")]
        print(f"{label}\n    named: {named}   expected only: ['{offender}']   {'OK' if named == [offender] else 'WRONG'}")
        print("    message:", str(e)[:140])
