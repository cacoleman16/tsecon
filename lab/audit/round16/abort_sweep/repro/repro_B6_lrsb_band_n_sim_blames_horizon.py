"""long_run_svar_bands(method="asymptotic", band="sup-t", band_n_sim=2**31)
at horizon=6, k=2 is refused, but the message's subject blames the two
arguments that are fine -- "horizon and band_scope: the delta-method working
set ... exceeds" -- and only mentions band_n_sim in the remedy list. Expected:
lead with band_n_sim (the only out-of-range value; the sup-t simulation work
n_families * band_n_sim * K^2 is what overflows the 2^36 budget)."""
import numpy as np, tsecon
r = np.random.default_rng(7); A = np.array([[0.5, 0.1], [0.1, 0.5]]); X = np.zeros((200, 2))
for t in range(1, 200):
    X[t] = X[t - 1] @ A.T + r.standard_normal(2)
for n_sim in [100000, 2**31]:
    try:
        tsecon.long_run_svar_bands(X, lags=1, horizon=6, method="asymptotic", band="sup-t", band_n_sim=n_sim)
        print(f"band_n_sim={n_sim}: returned")
    except Exception as e:
        msg = str(e)
        print(f"band_n_sim={n_sim}: EXPECTED refusal led by `band_n_sim`; ACTUAL {type(e).__name__} whose subject is {msg.split(':')[0]!r}: {msg[:200]}...")
