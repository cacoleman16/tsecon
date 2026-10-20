"""Independent reference: de Jong (1991) augmented Kalman filter diffuse
log-likelihood (Durbin & Koopman 2012, sec. 5.7 / 7.2.3), in extended precision,
on the system matrices statsmodels builds at the given parameters. No thresholds
are involved (the diffuse elements are integrated out through S = sum V'F^-1 V)."""
import numpy as np

LD = np.longdouble


def sys_at(mod, params):
    mod.update(np.asarray(params, float))
    ssm = mod.ssm
    Z = np.asarray(ssm["design"], float)        # 1 x m x (1 or T)
    d = np.asarray(ssm["obs_intercept"], float)  # 1 x (1 or T)
    Tm = np.asarray(ssm["transition"], float)   # m x m x (1 or T)
    c = np.asarray(ssm["state_intercept"], float)
    R = np.asarray(ssm["selection"], float)
    Q = np.asarray(ssm["state_cov"], float)
    H = np.asarray(ssm["obs_cov"], float)
    fr = mod.filter(np.asarray(params, float)).filter_results
    a1 = np.asarray(fr.initial_state, float)
    Pstar = np.asarray(fr.initial_state_cov, float)
    Pinf = fr.initial_diffuse_state_cov
    Pinf = np.zeros_like(Pstar) if Pinf is None else np.asarray(Pinf, float)
    return Z, d, Tm, c, R, Q, H, np.asarray(a1, float), np.asarray(Pinf, float), np.asarray(Pstar, float)


def at(M, t, nd=2):
    if M.ndim == nd:
        return M
    return M[..., t] if M.shape[-1] > 1 else M[..., 0]


def augmented_loglike(mod, params, y):
    Z, d, Tm, c, R, Q, H, a1, Pinf, Pstar = sys_at(mod, params)
    m = len(a1)
    diff_idx = [i for i in range(m) if Pinf[i, i] > 0]
    assert np.allclose(Pinf, np.diag(np.diag(Pinf)))
    B = np.zeros((m, len(diff_idx)), dtype=LD)
    for j, i in enumerate(diff_idx):
        B[i, j] = LD(np.sqrt(Pinf[i, i]))
    a = a1.astype(LD); A = B; P = Pstar.astype(LD)
    q = len(diff_idx)
    S = np.zeros((q, q), dtype=LD); s = np.zeros(q, dtype=LD)
    slogF = LD(0); se2 = LD(0); nobs = 0
    for t in range(len(y)):
        Tt = at(Tm, t).astype(LD); Rt = at(R, t).astype(LD); Qt = at(Q, t).astype(LD)
        RQR = Rt @ Qt @ Rt.T
        ct = at(c, t, 1).astype(LD)
        if np.isnan(y[t]):
            a = Tt @ a + ct; A = Tt @ A; P = Tt @ P @ Tt.T + RQR
            continue
        Zt = at(Z, t)[0].astype(LD); dt = LD(np.ravel(at(d, t, 1))[0]); Ht = LD(at(H, t)[0, 0])
        e = LD(y[t]) - Zt @ a - dt
        V = Zt @ A
        F = Zt @ P @ Zt + Ht
        K = Tt @ P @ Zt / F
        slogF += np.log(F); se2 += e * e / F
        S += np.outer(V, V) / F; s += V * e / F
        nobs += 1
        a = Tt @ a + K * e + ct
        A = Tt @ A - np.outer(K, V)
        P = Tt @ P @ Tt.T - np.outer(K, K) * F + RQR
        P = (P + P.T) / 2
    if q:
        Sf = np.asarray(S, float)
        # solve in long double via float factor + refinement
        sinv_s = np.linalg.solve(Sf, np.asarray(s, float)).astype(LD)
        for _ in range(3):
            r = s - S @ sinv_s
            sinv_s = sinv_s + np.linalg.solve(Sf, np.asarray(r, float)).astype(LD)
        quad = s @ sinv_s
        sign, ld = np.linalg.slogdet(Sf)
        logdetS = LD(ld)
    else:
        quad = LD(0); logdetS = LD(0)
    ll = -0.5 * (nobs * np.log(2 * np.pi) + slogF + se2 - quad + logdetS)
    return float(ll), q, np.linalg.cond(np.asarray(S, float)) if q else 1.0
