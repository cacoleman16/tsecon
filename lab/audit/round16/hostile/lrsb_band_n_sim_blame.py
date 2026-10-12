"""long_run_svar_bands(method='asymptotic', band='sup-t', band_n_sim=2**31)
with horizon and band_scope at their defaults is refused with a message that
LEADS with 'horizon and band_scope:' -- the two arguments the caller did not
touch -- and mentions band_n_sim only in its remedy clause."""
import numpy as np, tsecon
rng = np.random.default_rng(1); X = np.zeros((200, 2))
for t in range(1, 200): X[t] = 0.5 * X[t-1] + rng.standard_normal(2)
try:
    tsecon.long_run_svar_bands(X, method="asymptotic", band="sup-t", band_n_sim=2**31)
except ValueError as e:
    print(str(e))
