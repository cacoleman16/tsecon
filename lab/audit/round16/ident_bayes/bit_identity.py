"""Run a battery of VALID ident/bayes calls and pickle the outputs, so the
baseline wheel and the fixed wheel can be compared bit for bit:
    python bitident.py dump OUT.pkl ; python bitident.py compare A.pkl B.pkl"""
import pickle
import sys

import numpy as np


def battery():
    import tsecon

    def var(n=200, k=3, seed=0):
        rng = np.random.default_rng(seed)
        a = 0.4 * np.eye(k) + 0.05
        y = np.zeros((n, k))
        for t in range(1, n):
            y[t] = a @ y[t - 1] + rng.standard_normal(k)
        return y

    X3, X2 = var(), var(k=2, seed=1)
    SR = [(0, 0, 0, "+"), (1, 0, 0, "+")]
    NR = [{"type": "shock_sign", "shock": 0, "period": 10, "sign": "+"}]
    P = X3[:, 0] + np.random.default_rng(3).standard_normal(200)
    REG = np.where(np.arange(200) < 100, 0, 1)
    XH = X2 * np.where(REG[:, None] == 1, 2.0, 1.0) * np.array([1.0, 0.5])
    rng = np.random.default_rng(5)
    e = np.column_stack([rng.laplace(size=200), rng.standard_t(4, size=200)])
    NG = e + 0.3 * e[:, ::-1]
    rng = np.random.default_rng(6)
    f = np.cumsum(rng.standard_normal(80)) * 0.1
    PAN = np.column_stack([f + rng.standard_normal(80) for _ in range(6)])
    A0 = np.asarray(tsecon.long_run_svar(X2, lags=1)["impact"])
    calls = {
        "bvar_fit": lambda: tsecon.bvar_fit(X3, lags=2),
        "bvar_fit_ar1": lambda: tsecon.bvar_fit(X3, lags=2, scale_ar=1),
        "bvar_irf_draws": lambda: tsecon.bvar_irf_draws(X3, lags=2, n_draws=200, seed=7),
        "bvar_irf_draws_cum": lambda: tsecon.bvar_irf_draws(X3, lags=1, n_draws=100, seed=1, cumulative=True, scale_ar=1),
        "bvar_hierarchical": lambda: tsecon.bvar_hierarchical(X3, lags=2),
        "bvar_hierarchical_none_ar1": lambda: tsecon.bvar_hierarchical(X3, lags=2, hyperprior="none", scale_ar=1),
        "bvar_ssvs": lambda: tsecon.bvar_ssvs(X2, lags=1, n_draws=600, burn=100, seed=3, n_chains=2, thin=2),
        "sign": lambda: tsecon.sign_restricted_svar(X3, SR, lags=1, n_draws=300, seed=4),
        "zero_sign": lambda: tsecon.zero_sign_svar(X3, SR, [(2, 0, 0)], lags=1, n_draws=200, seed=5),
        "narrative": lambda: tsecon.narrative_svar(X3, SR, NR, lags=1, horizon=6, n_draws=200, max_tries=300, seed=6, n_weight_draws=60),
        "narrative_plain": lambda: tsecon.narrative_svar(X3, SR, lags=1, n_draws=100, seed=6),
        "fry_pagan": lambda: tsecon.fry_pagan_svar(X3, SR, lags=1, n_draws=200, seed=8),
        "robust": lambda: tsecon.robust_svar_bounds(X3, SR, lags=1, n_draws=200, seed=9),
        "hd_chol": lambda: tsecon.historical_decomposition(X3, lags=2),
        "hd_impact": lambda: tsecon.historical_decomposition(X2, lags=1, identification="impact", impact=A0),
        "hd_sign": lambda: tsecon.historical_decomposition(X3, SR, lags=1, identification="sign", n_draws=100, seed=2),
        "hd_sign_narr": lambda: tsecon.historical_decomposition(X3, SR, lags=1, identification="sign", narrative_restrictions=NR, n_draws=100, max_tries=300, seed=2, n_weight_draws=40),
        "sfevd": lambda: tsecon.structural_fevd(X3, lags=2, horizon=20),
        "sfevd_mle": lambda: tsecon.structural_fevd(X3, lags=2, horizon=20, sigma="mle"),
        "sfevd_impact": lambda: tsecon.structural_fevd(X2, lags=1, horizon=20, impact=A0),
        "proxy_svar": lambda: tsecon.proxy_svar(X3, P, lags=2),
        "proxy_bands": lambda: tsecon.proxy_svar_bands(X3, P, lags=2, n_boot=200, seed=3),
        "proxy_ar_delta": lambda: tsecon.proxy_ar_sets(X3, P, lags=2),
        "proxy_ar_so": lambda: tsecon.proxy_ar_sets(X3, P, lags=2, rf_method="second_order", rf_draws=128, rf_seed=5),
        "max_share": lambda: tsecon.max_share_svar(X3, lags=1, h1=8, horizon=12),
        "hetero": lambda: tsecon.hetero_svar(XH, REG, lags=1),
        "nongauss": lambda: tsecon.nongaussian_svar(NG, lags=1),
        "favar": lambda: tsecon.favar(PAN, var(80, 1, 9)[:, 0], n_factors=1, lags=1),
    }
    return {k: v() for k, v in calls.items()}


def same(a, b, path=""):
    if isinstance(a, dict):
        assert isinstance(b, dict) and a.keys() == b.keys(), path
        return all(same(a[k], b[k], f"{path}.{k}") for k in a)
    if isinstance(a, (list, tuple, np.ndarray)) and not isinstance(a, str):
        A, B = np.asarray(a, dtype=object), np.asarray(b, dtype=object)
        try:
            A, B = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
            ok = A.shape == B.shape and np.array_equal(A, B, equal_nan=True)
        except (TypeError, ValueError):
            ok = len(a) == len(b) and all(same(x, y, path) for x, y in zip(a, b))
        if not ok:
            print("DIFF", path)
        return ok
    if isinstance(a, float):
        ok = (a == b) or (np.isnan(a) and np.isnan(b))
    else:
        ok = a == b
    if not ok:
        print("DIFF", path, a, b)
    return ok


if __name__ == "__main__":
    if sys.argv[1] == "dump":
        with open(sys.argv[2], "wb") as fh:
            pickle.dump(battery(), fh)
    else:
        with open(sys.argv[2], "rb") as fh:
            a = pickle.load(fh)
        with open(sys.argv[3], "rb") as fh:
            b = pickle.load(fh)
        bad = [k for k in a if not same(a[k], b[k], k)]
        print(f"{len(a) - len(bad)} of {len(a)} valid calls bit-identical; differing: {bad}")
