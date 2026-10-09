"""Where historical_decomposition(identification="impact") stops working as
the impact matrix is scaled (Blanchard-Quah impact of a 2-variable VAR(1))."""
import numpy as np, tsecon
r = np.random.default_rng(1)
X = np.zeros((200, 2))
for t in range(1, 200):
    X[t] = (0.4 * np.eye(2) + 0.05) @ X[t - 1] + r.standard_normal(2)
A0 = np.asarray(tsecon.long_run_svar(X, lags=1)["impact"])
ref = np.asarray(tsecon.historical_decomposition(X, lags=1, identification="impact", impact=A0)["hd"])
for e in (-160, -156, -155, -154, -153, -150, 150, 155, 160, 161, 162, 163, 165):
    s = 10.0 ** e
    try:
        out = tsecon.historical_decomposition(X, lags=1, identification="impact", impact=A0 * s)
        err = np.max(np.abs(np.asarray(out["hd"]) - ref))
        print(f"1e{e:+d}: works, max|hd - hd(scale 1)| = {err:.1e}")
    except ValueError as ex:
        print(f"1e{e:+d}: refused: {str(ex)[:60]}")
