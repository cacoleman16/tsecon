import numpy as np, warnings
warnings.simplefilter("ignore")
from statsmodels.tsa.statespace.structural import UnobservedComponents as UC
from augkf import sys_at, at
pat = np.array([1.0, 0.5, -0.5, -1.0, 0.2, 0.0, 0.3, -0.2, 0.8, -0.6, 0.1, -0.6])
rng = np.random.default_rng(1); T = 120
y = np.cumsum(rng.normal(0, 0.3, T)) + np.tile(pat, 10) + rng.normal(0, 1, T); y[[2, 14]] = np.nan
mod = UC(y, "local level", seasonal=12, stochastic_seasonal=False, use_exact_diffuse=True)
Z, d, Tm, c, R, Q, H, a1, Pinf, Pstar = sys_at(mod, [1.0, 0.1])
Pi = Pinf.copy(); Ps = Pstar.copy(); Pm = np.abs(Pinf).copy()
for t in range(30):
    Tt = at(Tm, t); RQR = at(R, t) @ at(Q, t) @ at(R, t).T
    if np.isnan(y[t]):
        Pi = Tt @ Pi @ Tt.T; Ps = Tt @ Ps @ Tt.T + RQR; Pm = np.abs(Tt) @ Pm @ np.abs(Tt).T; continue
    z = at(Z, t)[0]; Hh = at(H, t)[0, 0]
    mi = Pi @ z; fi = max(z @ mi, 0.0); sc = np.abs(z) @ np.abs(Pi) @ np.abs(z); mag = np.abs(z) @ Pm @ np.abs(z)
    ms = Ps @ z; fs = z @ ms + Hh
    br = fi > 1e-10 * sc and fi > 1e-14 * mag
    if t >= 20: print(t, f"F_inf {fi:.3e} 1e-14*P_mag-scale {1e-14*mag:.3e} diffuse branch {br}")
    if br:
        k0 = mi / fi
        Ps = Ps + np.outer(mi, mi) * fs / fi**2 - np.outer(ms, mi) / fi - np.outer(mi, ms) / fi
        Pi = Pi - np.outer(mi, mi) / fi; Pm = Pm + np.outer(np.abs(mi), np.abs(k0))
    else:
        Ps = Ps - np.outer(ms, ms) / fs
    Pi = Tt @ Pi @ Tt.T; Ps = Tt @ Ps @ Tt.T + RQR; Pm = np.abs(Tt) @ Pm @ np.abs(Tt).T
