"""long_run_svar(lags=0) is refused with 'invalid argument: coefs must contain
at least one VAR lag matrix' -- `coefs` is not an argument of the function and
`lags` is not named. long_run_svar_bands names lags for the same input, and
this batch fixed the same refusal on historical_decomposition."""
import numpy as np, tsecon
rng = np.random.default_rng(1); X = np.zeros((200, 2))
for t in range(1, 200): X[t] = 0.5 * X[t-1] + rng.standard_normal(2)
for name, f in [("long_run_svar(lags=0)", lambda: tsecon.long_run_svar(X, lags=0)),
                ("long_run_svar(lags=0, impact_signs=[(0,1)])", lambda: tsecon.long_run_svar(X, lags=0, impact_signs=[(0, 1)])),
                ("long_run_svar_bands(lags=0)", lambda: tsecon.long_run_svar_bands(X, lags=0, n_boot=10)),
                ("historical_decomposition(lags=0)", lambda: tsecon.historical_decomposition(X, lags=0))]:
    try:
        f(); print(name, "ACCEPTED")
    except ValueError as e:
        print(f"{name}: names 'lags'={'lags' in str(e)} :: {str(e)[:120]}")
