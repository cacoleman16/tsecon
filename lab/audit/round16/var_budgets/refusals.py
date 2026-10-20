"""The size refusals of the VAR-family callables after audit round 16, on
seeded data: each call is refused before anything is allocated, naming the
keyword and the largest value that fits. Prints the message of each."""
import numpy as np
import tsecon


def var_data(t, k, seed):
    rng = np.random.default_rng(seed)
    a = 0.5 * np.eye(k) + 0.1 * np.ones((k, k)) / k
    y = np.zeros((t, k))
    for i in range(1, t):
        y[i] = a @ y[i - 1] + rng.standard_normal(k)
    return y


X2 = var_data(200, 2, 1)
X3 = var_data(240, 3, 2)
BQ_LIKE = var_data(300, 2, 3)
rng = np.random.default_rng(7)
y_lp = rng.standard_normal(300)
s_lp = rng.standard_normal(300)
C = rng.standard_normal((150, 12)).cumsum(axis=1)
Y = rng.standard_normal(150)
D = np.linspace(0.1, 0.3, 12)
E = [var_data(120, 2, 10 + i) for i in range(4)]

CASES = [
    ("var_irf k=2", lambda: tsecon.var_irf(X2, horizon=2**31)),
    ("var_irf k=3", lambda: tsecon.var_irf(X3, horizon=2**31)),
    ("var_fevd k=2", lambda: tsecon.var_fevd(X2, horizon=2**31)),
    ("connectedness k=3", lambda: tsecon.connectedness(X3, horizon=2**40)),
    ("fvar_scenario K=2", lambda: tsecon.fvar_scenario(Y, C, D, n_factors=2, lags=1, horizon=2**31)),
    ("mean_group_var N=4 k=2", lambda: tsecon.mean_group_var(E, horizon=2**31)),
    ("long_run_svar k=2", lambda: tsecon.long_run_svar(X2, horizon=2**31)),
    ("var_irf_bands asym k=2 lags=2", lambda: tsecon.var_irf_bands(X2, horizon=2**31)),
    ("var_irf_bands asym k=2 lags=8", lambda: tsecon.var_irf_bands(BQ_LIKE, lags=8, horizon=10**4)),
    ("var_irf_bands asym k=20 lags=4", lambda: tsecon.var_irf_bands(var_data(800, 20, 4), lags=4, horizon=29)),
    ("var_irf_bands asym sup-t band_n_sim", lambda: tsecon.var_irf_bands(X2, lags=1, horizon=6, band="sup-t", band_n_sim=2**31)),
    ("var_irf_bands asym sidak all", lambda: tsecon.var_irf_bands(X2, lags=1, horizon=3000, band="sidak", band_scope="all")),
    ("var_irf_bands boot n_boot", lambda: tsecon.var_irf_bands(X2, method="bootstrap", n_boot=10**7)),
    ("var_irf_bands boot horizon", lambda: tsecon.var_irf_bands(X2, method="bootstrap", n_boot=50, horizon=2**31)),
    ("long_run_svar_bands boot n_boot", lambda: tsecon.long_run_svar_bands(X2, n_boot=10**7)),
    ("long_run_svar_bands boot n_boot h0", lambda: tsecon.long_run_svar_bands(X2, n_boot=10**7, horizon=0)),
    ("long_run_svar_bands asym pointwise", lambda: tsecon.long_run_svar_bands(X2, method="asymptotic", horizon=10**6)),
    ("long_run_svar_bands asym sup-t band_n_sim", lambda: tsecon.long_run_svar_bands(X2, method="asymptotic", band="sup-t", band_n_sim=2**31)),
    ("lp sup-t band_n_sim", lambda: tsecon.lp(y_lp, s_lp, band="sup-t", band_n_sim=2**31)),
    ("smooth_lp sup-t band_n_sim", lambda: tsecon.smooth_lp(y_lp, s_lp, band="sup-t", band_n_sim=2**31)),
    ("var_forecast sup-t band_n_sim", lambda: tsecon.var_forecast(X2, band="sup-t", band_n_sim=2**31)),
    ("var_forecast sidak steps", lambda: tsecon.var_forecast(X2, steps=10**5, band="sidak")),
    ("var_forecast steps", lambda: tsecon.var_forecast(X2, steps=2**31)),
]

for name, call in CASES:
    try:
        call()
        print(f"{name}: RETURNED")
    except Exception as e:  # noqa: BLE001
        print(f"{name}: {type(e).__name__}: {e}\n")
