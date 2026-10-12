"""var_forecast(param_uncertainty=True): the work-budget refusal names `steps` and states the
cost as "about steps^3 (k^2 + 8) / 3 operations: CUBIC in steps", and the docstring says the
budget is CUBIC in steps. The budget actually checked (forecast.rs param_uncertainty_work) also
adds 2*steps*m^3 + steps^2*m^2 (m = 1 + k*p regressors per equation), which is what trips it
for a wide VAR. The quoted formula then sits far BELOW the 2^34 budget it claims to exceed, and
the advice "Forecast horizons are tens of periods" is given for a horizon of tens of periods.
"""
import numpy as np, tsecon
rng = np.random.default_rng(0)
BUDGET = 2 ** 34
for (k, p, steps, T) in [(30, 10, 200, 3000), (60, 20, 50, 3000), (60, 20, 10, 3000)]:
    data = rng.standard_normal((T, k))
    m = 1 + k * p
    quoted = steps ** 3 * (k * k + 8) / 3
    actual = quoted + steps ** 2 * k ** 3 / 2 + 2 * steps * m ** 3 + steps ** 2 * m ** 2
    try:
        tsecon.var_forecast(data, lags=p, steps=steps, param_uncertainty=True)
        res = "accepted"
    except ValueError as e:
        res = "REFUSED: " + str(e)[:150]
    print(f"k={k} p={p} steps={steps}: quoted steps^3(k^2+8)/3 = {quoted:.2e} ({quoted/BUDGET:.3f} x budget), "
          f"full work = {actual:.2e} ({actual/BUDGET:.2f} x budget)\n   -> {res}")
