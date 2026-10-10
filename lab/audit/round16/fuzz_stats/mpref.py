import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "pylib"))
import mpmath as mp
import numpy as np
mp.mp.dps = 60

def _mat(X):
    return mp.matrix([[mp.mpf(float(v)) for v in row] for row in X])

def mp_all(X):
    X = np.asarray(X, float)
    if X.ndim == 1: X = X[:, None]
    n, k = X.shape
    M = _mat(X)
    mean = [mp.fsum(M[i, j] for i in range(n)) / n for j in range(k)]
    Xc = mp.matrix(n, k)
    for i in range(n):
        for j in range(k):
            Xc[i, j] = M[i, j] - mean[j]
    S = (Xc.T * Xc) / n
    out = {}
    # cholesky
    L = mp.cholesky(S)
    def jbw(Pinv):
        W = Pinv * Xc.T  # k x n
        b1 = [mp.fsum(W[j, i] ** 3 for i in range(n)) / n for j in range(k)]
        b2 = [mp.fsum(W[j, i] ** 4 for i in range(n)) / n - 3 for j in range(k)]
        ls = n * mp.fsum(b ** 2 for b in b1) / 6
        lk = n * mp.fsum(b ** 2 for b in b2) / 24
        return float(ls), float(lk), float(ls + lk), [float(b) for b in b1], [float(b) for b in b2]
    out["chol"] = jbw(mp.inverse(L))
    E, Q = mp.eigsy(S)
    Dm = mp.diag([1 / mp.sqrt(e) for e in E])
    out["sym"] = jbw(Q * Dm * Q.T)
    # DH
    V = mp.diag([1 / mp.sqrt(S[j, j]) for j in range(k)])
    C = V * S * V
    E2, H = mp.eigsy(C)
    Y = Xc * V * H * mp.diag([1 / mp.sqrt(e) for e in E2]) * H.T
    z1s = []; z2s = []
    nn = mp.mpf(n)
    for j in range(k):
        m2 = mp.fsum(Y[i, j] ** 2 for i in range(n)) / n
        m3 = mp.fsum(Y[i, j] ** 3 for i in range(n)) / n
        m4 = mp.fsum(Y[i, j] ** 4 for i in range(n)) / n
        sb1 = m3 / m2 ** mp.mpf(1.5); b2 = m4 / m2 ** 2; b1 = sb1 ** 2
        beta = 3 * (nn ** 2 + 27 * nn - 70) * (nn + 1) * (nn + 3) / ((nn - 2) * (nn + 5) * (nn + 7) * (nn + 9))
        w2 = -1 + mp.sqrt(2 * (beta - 1))
        delta = 1 / mp.sqrt(mp.log(mp.sqrt(w2)))
        y = sb1 * mp.sqrt((w2 - 1) / 2 * (nn + 1) * (nn + 3) / (6 * (nn - 2)))
        z1 = delta * mp.log(y + mp.sqrt(y * y + 1))
        d = (nn - 3) * (nn + 1) * (nn ** 2 + 15 * nn - 4)
        a = (nn - 2) * (nn + 5) * (nn + 7) * (nn ** 2 + 27 * nn - 70) / (6 * d)
        c = (nn - 7) * (nn + 5) * (nn + 7) * (nn ** 2 + 2 * nn - 5) / (6 * d)
        kk = (nn + 5) * (nn + 7) * (nn ** 3 + 37 * nn ** 2 + 11 * nn - 313) / (12 * d)
        alpha = a + b1 * c
        chi = (b2 - 1 - b1) * 2 * kk
        t = chi / (2 * alpha)
        cb = mp.cbrt(t) if t >= 0 else -mp.cbrt(-t)
        z2 = (cb - 1 + 1 / (9 * alpha)) * mp.sqrt(9 * alpha)
        z1s.append(z1); z2s.append(z2)
    out["dh"] = (float(mp.fsum(z ** 2 for z in z1s) + mp.fsum(z ** 2 for z in z2s)), [float(z) for z in z1s], [float(z) for z in z2s])
    return out
