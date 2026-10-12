"""alpha (or conf_alpha) in (0, ~1.1e-16) passes validation but the quantile
z_{1-alpha/2} is evaluated at p = 1 - alpha/2 == 1.0 in double precision, and
the call fails with 'special-function failure: argument `p` = 1 outside
domain: requires 0 < p < 1' -- naming neither alpha nor the function. On
long_run_svar_bands' Sidak band the failure starts at alpha = 1e-15. The
same unnamed text comes from var_irf_bands and arima_fit (house-wide), but
it reaches the new surfaces: long_run_svar_bands (both arms) and the soft
var_conditional_forecast path. Expected: refuse naming alpha (or work)."""
import numpy as np, tsecon
rng = np.random.default_rng(1); X = np.zeros((200, 2))
for t in range(1, 200): X[t] = 0.5 * X[t-1] + rng.standard_normal(2)
y = X[:, 0]
cases = [
    ("long_run_svar_bands asymptotic alpha=1e-17", lambda: tsecon.long_run_svar_bands(X, method="asymptotic", alpha=1e-17)),
    ("long_run_svar_bands bootstrap sidak alpha=1e-15", lambda: tsecon.long_run_svar_bands(X, n_boot=50, band="sidak", alpha=1e-15)),
    ("var_conditional_forecast soft alpha=1e-17", lambda: tsecon.var_conditional_forecast(X, [[1.0, None]], condition_sd=[[0.3, None]], soft="signal", alpha=1e-17)),
    ("arima_fit css conf_alpha=1e-17", lambda: tsecon.arima_fit(y, method="css", forecast_steps=2, conf_alpha=1e-17)),
]
for name, f in cases:
    try:
        f(); print(name, "-> ok")
    except ValueError as e:
        print(f"{name} -> ValueError naming alpha: {'alpha' in str(e)} :: {e}")
