"""Re-measure the batch-2 normality size claims with fresh seeds.

Claims (var-svar card, CHANGELOG, docstrings):
  univariate n = 50, 20 000 reps: Cholesky/sym JB 3.73%, DH 4.89%;
    null z-moments: mean z1 -0.0026, var z1 0.9987, mean z2 0.0087, var z2 0.9138
  VAR(1) k = 3, T = 40, 2000 reps: Cholesky 4.65%, sym 4.25%, DH 5.30%
  VAR(1) k = 2, T = 200, 1000 reps: Cholesky 4.20%, sym 4.50%, DH 4.80%
Also an independent NumPy transcription of DH (2008) to cross-check the
statistic on a subset of the draws.
"""
import sys
import time
import numpy as np
import tsecon
from scipy import stats

MODE = sys.argv[1] if len(sys.argv) > 1 else "all"


def dh_numpy(u):
    u = np.asarray(u, float)
    if u.ndim == 1:
        u = u[:, None]
    n, k = u.shape
    x = u - u.mean(0)
    S = x.T @ x / n
    v = np.sqrt(np.diag(S))
    C = S / np.outer(v, v)
    lam, H = np.linalg.eigh(C)
    R = H @ np.diag(lam ** -0.5) @ H.T
    r = (x / v) @ R.T  # rows r_t = H L^-1/2 H' V^-1/2 x_t
    sb1 = (r ** 3).mean(0)
    b2 = (r ** 4).mean(0)
    b1 = sb1 ** 2
    beta = 3 * (n * n + 27 * n - 70) * (n + 1) * (n + 3) / ((n - 2) * (n + 5) * (n + 7) * (n + 9))
    w2 = -1 + np.sqrt(2 * (beta - 1))
    delta = 1 / np.sqrt(np.log(np.sqrt(w2)))
    y = sb1 * np.sqrt((w2 - 1) * (n + 1) * (n + 3) / (12 * (n - 2)))
    z1 = delta * np.log(y + np.sqrt(y * y + 1))
    d = (n - 3) * (n + 1) * (n * n + 15 * n - 4)
    a = (n - 2) * (n + 5) * (n + 7) * (n * n + 27 * n - 70) / (6 * d)
    c = (n - 7) * (n + 5) * (n + 7) * (n * n + 2 * n - 5) / (6 * d)
    kk = (n + 5) * (n + 7) * (n ** 3 + 37 * n * n + 11 * n - 313) / (12 * d)
    alpha = a + b1 * c
    chi = (b2 - 1 - b1) * 2 * kk
    z2 = (np.cbrt(chi / (2 * alpha)) - 1 + 1 / (9 * alpha)) * np.sqrt(9 * alpha)
    E = z1 @ z1 + z2 @ z2
    return E, stats.chi2.sf(E, 2 * k), z1, z2


def mcse(p, n):
    return np.sqrt(p * (1 - p) / n)


def report(label, rej, n, claim):
    p = rej / n
    se = mcse(p, n)
    z = (claim - p) / se
    print(f"  {label:28s} {p*100:6.3f}%  (MC se {se*100:.3f} pts)  claim {claim*100:.2f}%  "
          f"claim-measured = {z:+.2f} se")


def univariate(reps=100_000, seed=0xA0D1_6001):
    rng = np.random.default_rng(seed)
    rej_dh = rej_jb = rej_np = 0
    z1s, z2s = np.empty(reps), np.empty(reps)
    maxgap = 0.0
    t0 = time.time()
    for i in range(reps):
        x = rng.standard_normal(50)
        r = tsecon.multivariate_normality(x)
        rej_dh += r["doornik_hansen_pvalue"] < 0.05
        rej_jb += r["jarque_bera_pvalue"] < 0.05
        z1s[i] = r["doornik_hansen_z_skewness"][0]
        z2s[i] = r["doornik_hansen_z_kurtosis"][0]
        if i < 2000:
            E, p, _, _ = dh_numpy(x)
            maxgap = max(maxgap, abs(E - r["doornik_hansen"]) / max(1.0, E))
            rej_np += p < 0.05
    print(f"univariate n = 50, {reps} fresh Gaussian samples (seed {seed:#x}), {time.time()-t0:.0f}s")
    report("asymptotic JB (Cholesky=sym)", rej_jb, reps, 0.0373)
    report("Doornik-Hansen", rej_dh, reps, 0.0489)
    print(f"  null z-moments: mean z1 {z1s.mean():+.4f} (se {z1s.std()/np.sqrt(reps):.4f}; claim -0.0026), "
          f"var z1 {z1s.var():.4f} (claim 0.9987), mean z2 {z2s.mean():+.4f} (claim 0.0087), "
          f"var z2 {z2s.var():.4f} (se ~{np.sqrt((np.mean((z2s-z2s.mean())**4)-z2s.var()**2)/reps):.4f}; claim 0.9138)")
    print(f"  NumPy DH transcription vs tsecon on the first 2000 draws: max rel gap {maxgap:.2e}")


def simulate_var(rng, n, c, A, chol, burn=100):
    k = len(c)
    y = np.zeros((n + burn + 1, k))
    for t in range(1, n + burn + 1):
        y[t] = c + A @ y[t - 1] + chol @ rng.standard_normal(k)
    return y[burn + 1:]


def var_design(label, reps, n, c, A, chol, nlags, seed, claims):
    rng = np.random.default_rng(seed)
    rej = {"jb": 0, "sym": 0, "dh": 0, "dh_np": 0}
    maxgap = 0.0
    t0 = time.time()
    for i in range(reps):
        y = simulate_var(rng, n, c, A, chol)
        d = tsecon.var_diagnostics(y, lags=1, nlags=nlags)
        rej["jb"] += d["jarque_bera_pvalue"] < 0.05
        rej["sym"] += d["jarque_bera_symmetric_pvalue"] < 0.05
        rej["dh"] += d["doornik_hansen_pvalue"] < 0.05
        if i < 1000:
            u = np.asarray(tsecon.var_fit(y, lags=1)["resid"])
            E, p, _, _ = dh_numpy(u)
            maxgap = max(maxgap, abs(E - d["doornik_hansen"]) / max(1.0, E))
    print(f"{label}, {reps} fresh replications (seed {seed:#x}), nobs {d['nobs']}, {time.time()-t0:.0f}s")
    report("Cholesky JB", rej["jb"], reps, claims[0])
    report("symmetric-root JB", rej["sym"], reps, claims[1])
    report("Doornik-Hansen (2008)", rej["dh"], reps, claims[2])
    print(f"  NumPy DH transcription vs tsecon on the first 1000 fits: max rel gap {maxgap:.2e}")


if __name__ == "__main__":
    if MODE in ("all", "uni"):
        univariate()
    if MODE in ("all", "k3"):
        var_design("VAR(1) k = 3, T = 40", 40_000, 40, np.array([0.2, -0.1, 0.05]),
                   np.array([[0.5, 0.1, 0.0], [0.0, 0.4, 0.1], [0.1, 0.0, 0.3]]), np.eye(3), 6,
                   0xA0D1_6002, (0.0465, 0.0425, 0.0530))
    if MODE in ("all", "k2"):
        var_design("VAR(1) k = 2, T = 200", 20_000, 200, np.array([0.2, -0.1]),
                   np.array([[0.5, 0.2], [0.1, 0.4]]), np.array([[1.0, 0.0], [0.5, 0.8]]), 8,
                   0xA0D1_6003, (0.0420, 0.0450, 0.0480))
