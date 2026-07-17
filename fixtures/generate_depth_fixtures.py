"""Golden fixtures for the depth crates: realized volatility/HAR (statsmodels
OLS + documented measures), Diebold-Yilmaz connectedness (self-authored GFEVD),
and FAVAR factor extraction (numpy SVD / PCA).

Run with the project venv: .venv/bin/python fixtures/generate_depth_fixtures.py
"""
import json
import platform
from pathlib import Path

import numpy as np

OUT = Path(__file__).parent
full = lambda a: [float(x) for x in np.asarray(a).ravel()]


# ------------------------------------------- realized volatility / HAR
def gen_realized():
    import statsmodels
    import statsmodels.api as sm

    rng = np.random.default_rng(41)
    n_days, m = 600, 78  # 78 five-minute returns per day
    # Simulate daily integrated variance with a jump component so RV != BV.
    iv = np.empty(n_days)
    iv[0] = 1.0
    for t in range(1, n_days):
        iv[t] = 0.02 + 0.95 * iv[t - 1] + 0.1 * rng.standard_normal() ** 2
    intraday = np.sqrt(iv[:, None] / m) * rng.standard_normal((n_days, m))
    # Inject occasional jumps.
    jump_days = rng.random(n_days) < 0.05
    intraday[jump_days, 0] += rng.standard_normal(jump_days.sum()) * 1.5
    rv = np.sum(intraday ** 2, axis=1)  # realized variance

    # HAR-RV (Corsi 2009): RV_t on [1, RV_{t-1}, RV_week, RV_month].
    rv_d = rv[:-1]
    rv_w = np.array([rv[t - 5:t].mean() for t in range(len(rv))])[:-1]
    rv_m = np.array([rv[t - 22:t].mean() for t in range(len(rv))])[:-1]
    start = 22
    y = rv[start + 1:]
    X = np.column_stack([np.ones(len(y)), rv_d[start:], rv_w[start:], rv_m[start:]])
    ols = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})

    # A documented realized-measure golden: RV and bipower variation on a small
    # fixed return vector (Barndorff-Nielsen & Shephard 2004): BV = (pi/2) *
    # sum_{i=2}^n |r_i||r_{i-1}|.
    small = np.array([0.5, -0.3, 0.8, -1.2, 0.1, 0.4, -0.6])
    rv_small = float(np.sum(small ** 2))
    bv_small = float((np.pi / 2) * np.sum(np.abs(small[1:]) * np.abs(small[:-1])))

    out = {
        "_meta": {"statsmodels": statsmodels.__version__, "numpy": np.__version__,
                  "python": platform.python_version(),
                  "note": "HAR-RV = OLS(RV_t on [const, RV_{t-1}, RV_week(5), RV_month(22)]) "
                          "with HAC(maxlags=5); measures per BNS 2004."},
        "rv_series": full(rv),
        "har": {"start": start, "params": full(ols.params), "bse": full(ols.bse),
                "rsquared": float(ols.rsquared)},
        "measures_small": {"returns": full(small), "rv": rv_small, "bipower": bv_small},
    }
    (OUT / "realized.json").write_text(json.dumps(out, separators=(",", ":")))
    print("wrote realized.json")


# --------------------------------------------- Diebold-Yilmaz connectedness
def gen_connect():
    from statsmodels.tsa.api import VAR

    var_fx = json.loads((OUT / "var.json").read_text())
    data = np.array(var_fx["data_100dlog_gdp_cons_inv"])
    p, H = 2, 10
    res = VAR(data).fit(p, trend="c")
    sigma = np.asarray(res.sigma_u)
    coefs = [np.asarray(res.coefs[i]) for i in range(p)]
    k = data.shape[1]

    # MA(inf) Psi weights.
    psi = [np.eye(k)]
    for h in range(1, H + 1):
        acc = np.zeros((k, k))
        for i in range(1, min(h, p) + 1):
            acc += coefs[i - 1] @ psi[h - i]
        psi.append(acc)

    # Generalized FEVD (Pesaran-Shin 1998): theta_ij = sigma_jj^-1 *
    # sum_h (e_i' Psi_h Sigma e_j)^2 / sum_h (e_i' Psi_h Sigma Psi_h' e_i),
    # then row-normalized (Diebold-Yilmaz 2012).
    gfevd = np.zeros((k, k))
    for i in range(k):
        denom = sum(psi[h][i] @ sigma @ psi[h][i] for h in range(H + 1))
        for j in range(k):
            num = sum((psi[h][i] @ sigma[:, j]) ** 2 for h in range(H + 1)) / sigma[j, j]
            gfevd[i, j] = num / denom
    gfevd_norm = gfevd / gfevd.sum(axis=1, keepdims=True)

    total = 100.0 * (gfevd_norm.sum() - np.trace(gfevd_norm)) / k
    to_others = 100.0 * (gfevd_norm.sum(axis=0) - np.diag(gfevd_norm)) / k
    from_others = 100.0 * (gfevd_norm.sum(axis=1) - np.diag(gfevd_norm)) / k

    out = {
        "_meta": {"numpy": np.__version__,
                  "note": "Pesaran-Shin GFEVD, row-normalized; Diebold-Yilmaz 2012 "
                          "connectedness (percent) on VAR(2, const) of macrodata."},
        "data": [full(data[:, j]) for j in range(k)],
        "lags": p, "horizon": H,
        "gfevd_normalized": [full(gfevd_norm[i]) for i in range(k)],
        "total_connectedness": float(total),
        "to_others": full(to_others),
        "from_others": full(from_others),
    }
    (OUT / "connect.json").write_text(json.dumps(out, separators=(",", ":")))
    print("wrote connect.json")


# ------------------------------------------------------------ FAVAR / PCA
def gen_favar():
    rng = np.random.default_rng(53)
    n, big_n, r = 300, 24, 2
    F = np.zeros((n, r))
    for t in range(1, n):
        F[t] = np.array([0.7, 0.5]) * F[t - 1] + rng.standard_normal(r)
    loadings = rng.standard_normal((big_n, r))
    X = F @ loadings.T + rng.standard_normal((n, big_n)) * 0.5
    # Standardize (PCA convention): columns mean 0, sd 1 (ddof=0).
    Xs = (X - X.mean(0)) / X.std(0)
    # PCA via SVD; principal components = U*S, loadings = V.
    U, S, Vt = np.linalg.svd(Xs, full_matrices=False)
    pcs = (U * S)[:, :r]
    pc_loadings = Vt[:r].T
    eigvals = (S ** 2 / n)

    # Bai-Ng ICp2 criterion inputs: the eigenvalues drive factor-number choice.
    out = {
        "_meta": {"numpy": np.__version__,
                  "note": "PCA on standardized (ddof=0) X via SVD; PCs = U*S, "
                          "loadings = V columns. Factors identified up to sign."},
        "X_standardized": [full(Xs[:, j]) for j in range(big_n)],
        "n": n, "big_n": big_n, "true_r": r,
        "eigenvalues": full(eigvals),
        "pc1_abs": full(np.abs(pcs[:, 0])),  # sign-free comparison
        "pc2_abs": full(np.abs(pcs[:, 1])),
        "loadings_pc1_abs": full(np.abs(pc_loadings[:, 0])),
    }
    (OUT / "favar.json").write_text(json.dumps(out, separators=(",", ":")))
    print("wrote favar.json")


if __name__ == "__main__":
    gen_realized()
    gen_connect()
    gen_favar()
