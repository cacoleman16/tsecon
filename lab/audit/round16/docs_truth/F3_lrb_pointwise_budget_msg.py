"""F3: long_run_svar_bands(method="asymptotic", band="pointwise") past the working-set
budget is refused naming "horizon and band_scope" and advising "narrow band_scope, or
lower band_n_sim" -- both of which are refused as inert under band="pointwise"
(and the card/docstring say band_scope is named only for a simultaneous band's family)."""
import numpy as np, tsecon
data = np.random.default_rng(0).standard_normal((300, 2))
try:
    tsecon.long_run_svar_bands(data, lags=2, horizon=10**6, method="asymptotic")
except ValueError as e:
    print("refusal:", e)
for kw in (dict(band_scope="all"), dict(band_n_sim=10)):
    try:
        tsecon.long_run_svar_bands(data, lags=2, horizon=10**6, method="asymptotic", **kw)
    except ValueError as e:
        print(f"following the advice with {kw}:", str(e)[:150])
