"""min_root / n_unstable_draws misclassify stability on badly scaled data.

Same VAR(2), three variables; the third is re-expressed in units 1e14 times
smaller than the first (e.g. GDP in dollars next to a rate in decimals).  The
VAR's roots are invariant to rescaling the variables, and tsecon's own fitted
coefficients are fine (their exact companion roots, computed by numpy after
undoing the scaling, reproduce the unscaled value).  But
long_run_svar_bands(...)["min_root"] (documented: "stable iff > 1") comes out on
the wrong side of 1, and n_unstable_draws disagrees with a replay of the
documented bootstrap stream that measures each refit's companion radius with
numpy.
"""
import numpy as np
import tsecon

k, p, T = 3, 2, 400


def comp(coefs):
    C = np.zeros((k * p, k * p))
    C[:k] = np.hstack(coefs)
    C[k:, :-k] = np.eye(k * (p - 1))
    return C


def radius(coefs):
    return float(np.abs(np.linalg.eigvals(comp(coefs))).max())


def ols(Z, Y):
    """Column-equilibrated least squares (robust to badly scaled regressors)."""
    cn = np.linalg.norm(Z, axis=0)
    return np.linalg.lstsq(Z / cn, Y, rcond=None)[0] / cn[:, None]


def random_var(rng, tgt):
    while True:
        A = [rng.standard_normal((k, k)) * (0.5 / (i + 1)) / np.sqrt(k) for i in range(p)]
        r = radius(A)
        if r < 1e-6:
            continue
        s = tgt / r
        A = [a * s ** (i + 1) for i, a in enumerate(A)]
        if radius(A) < 0.9999:
            return A


rng = np.random.default_rng(21)
A = random_var(rng, 0.9995)
n = T + 200
u = rng.standard_normal((n, k))
y = np.zeros((n, k))
for t in range(p, n):
    y[t] = u[t] + sum(A[i] @ y[t - 1 - i] for i in range(p))
Y0 = y[200:]

for scale in [np.ones(3), np.array([1e7, 1.0, 1e-7])]:
    Y = Y0 * scale
    vf = tsecon.var_fit(Y, lags=p)
    P = np.asarray(vf["params"])
    coefs = [P[1 + i * k:1 + (i + 1) * k].T for i in range(p)]
    Dg = np.diag(np.tile(scale, p))
    exact = 1.0 / np.abs(np.linalg.eigvals(np.linalg.inv(Dg) @ comp(coefs) @ Dg)).max()
    r = tsecon.long_run_svar_bands(Y, lags=p, horizon=4, n_boot=200, seed=3)
    print(f"scales {scale}:")
    print(f"   exact min root of tsecon's own coefficients {exact:.12f}  -> stable={exact > 1}")
    print(f"   long_run_svar_bands min_root                {r['min_root']:.12f}  -> stable={r['min_root'] > 1}")
    print(f"   var_fit min_root {vf['min_root']:.12f} is_stable={vf['is_stable']}")
    # replay the documented stream and count raw refits with radius >= 1 (numpy)
    Z = np.hstack([np.ones((T - p, 1))] + [Y[p - i:T - i] for i in range(1, p + 1)])
    Bm = ols(Z, Y[p:])
    U = Y[p:] - Z @ Bm
    Uc = U - U.mean(0)
    c0 = Bm[0]
    cf = [Bm[1 + i * k:1 + (i + 1) * k].T for i in range(p)]
    cnt = 0
    for child in np.random.SeedSequence(3).spawn(200):
        bg = np.random.Philox(child)
        nn = T - p
        mask = (1 << (nn - 1).bit_length()) - 1
        idx = []
        while len(idx) < nn:
            v = int(bg.random_raw()) & mask
            if v < nn:
                idx.append(v)
        ys = np.zeros((T, k))
        ys[:p] = Y[:p]
        us = Uc[idx]
        for t in range(p, T):
            ys[t] = c0 + us[t - p] + sum(cf[i] @ ys[t - 1 - i] for i in range(p))
        Zb = np.hstack([np.ones((T - p, 1))] + [ys[p - i:T - i] for i in range(1, p + 1)])
        Bb = ols(Zb, ys[p:])
        cb = [Bb[1 + i * k:1 + (i + 1) * k].T for i in range(p)]
        Dg = np.diag(np.tile(scale, p))
        rb = np.abs(np.linalg.eigvals(np.linalg.inv(Dg) @ comp(cb) @ Dg)).max()
        cnt += rb >= 1.0
    print(f"   n_unstable_draws: tsecon {r['n_unstable_draws']}  replay(numpy radius) {cnt}")
