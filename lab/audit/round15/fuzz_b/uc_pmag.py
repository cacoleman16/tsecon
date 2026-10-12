"""Replay tsecon's documented exact-diffuse element tests (filter.rs docs) on seed 2261:
diffuse branch iff F_inf > 1e-10 |Z|'|P_inf||Z| and F_inf > 1e-14 |Z|' P_mag |Z|,
P_mag the cancellation-free upper-bound recursion (adds |M_inf||K0|', transported by |T|P_mag|T|')."""
import json, numpy as np, warnings
warnings.simplefilter("ignore")
from uc_fuzz import gen, sm_model
from augkf import sys_at, at
o = {d["seed"]: d for d in json.load(open('uc_1000.json'))}
seed = 2261
rng = np.random.default_rng(seed)
y, X, Xf, h, spec, scale = gen(rng)
mod = sm_model(y, X, spec)
p = np.array(o[seed]['params'])
Z, d, Tm, c, R, Q, H, a1, Pinf, Pstar = sys_at(mod, p)
Pi = Pinf.copy(); Ps = Pstar.copy(); Pm = np.abs(Pinf).copy()
for t in range(22):
    Tt = at(Tm, t); Rt = at(R, t); Qt = at(Q, t); RQR = Rt @ Qt @ Rt.T
    if np.isnan(y[t]):
        Pi = Tt @ Pi @ Tt.T; Ps = Tt @ Ps @ Tt.T + RQR; Pm = np.abs(Tt) @ Pm @ np.abs(Tt).T
        print(t, "missing"); continue
    z = at(Z, t)[0]; Hh = at(H, t)[0, 0]
    mi = Pi @ z; fi = max(z @ mi, 0.0); scale_ = np.abs(z) @ np.abs(Pi) @ np.abs(z); mag = np.abs(z) @ Pm @ np.abs(z)
    ms = Ps @ z; fs = z @ ms + Hh
    diffuse_branch = fi > 1e-10 * scale_ and fi > 1e-14 * mag
    print(t, f"F_inf {fi:.3e}  1e-10*scale {1e-10*scale_:.2e}  1e-14*mag {1e-14*mag:.2e}  diffuse branch: {diffuse_branch}  ||P_inf||_F^2 {np.sum(Pi**2):.2e}")
    if diffuse_branch:
        k0 = mi / fi; k1 = ms / fi - mi * fs / fi**2
        Ps_new = Ps + np.outer(mi, mi) * fs / fi**2 - np.outer(ms, mi) / fi - np.outer(mi, ms) / fi
        Pi_new = Pi - np.outer(mi, mi) / fi
        Pm = Pm + np.outer(np.abs(mi), np.abs(k0))
        Ps, Pi = Ps_new, Pi_new
    else:
        Ps = Ps - np.outer(ms, ms) / fs
    Pi = Tt @ Pi @ Tt.T; Ps = Tt @ Ps @ Tt.T + RQR; Pm = np.abs(Tt) @ Pm @ np.abs(Tt).T
