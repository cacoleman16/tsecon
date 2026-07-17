"""Golden fixtures for the Phase 3/4 crates: cointegration (statsmodels VECM +
Johansen), Markov-switching (statsmodels), MIDAS (U-MIDAS = OLS + documented
weight formulas), and multivariate GARCH (simulated DCC data for recovery).

Run with the project venv: .venv/bin/python fixtures/generate_phase34_fixtures.py
"""
import json
import platform
import warnings
from pathlib import Path

import numpy as np

OUT = Path(__file__).parent
r10 = lambda a: [round(float(x), 10) for x in np.asarray(a).ravel()]
full = lambda a: [float(x) for x in np.asarray(a).ravel()]


# ------------------------------------------------------ cointegration
def gen_coint():
    import statsmodels
    import statsmodels.api as sm
    from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen

    # Two cointegrated series (share a common stochastic trend) + a stationary
    # third, so Johansen finds rank 2 vs the I(1) common trend.
    rng = np.random.default_rng(7)
    n = 400
    trend = np.cumsum(rng.standard_normal(n))
    y1 = trend + rng.standard_normal(n) * 0.5
    y2 = 0.8 * trend + rng.standard_normal(n) * 0.5
    y3 = np.empty(n)
    y3[0] = 0.0
    for t in range(1, n):
        y3[t] = 0.5 * y3[t - 1] + rng.standard_normal()  # stationary AR(1)
    data = np.column_stack([y1, y2, y3])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        joh = coint_johansen(data, det_order=0, k_ar_diff=2)
        vecm = VECM(data, k_ar_diff=2, coint_rank=1, deterministic="n").fit()

    out = {
        "_meta": {"statsmodels": statsmodels.__version__, "numpy": np.__version__,
                  "python": platform.python_version(),
                  "note": "coint_johansen(det_order=0, k_ar_diff=2); VECM(k_ar_diff=2, "
                          "coint_rank=1, deterministic='n')."},
        "data": [full(data[:, j]) for j in range(3)],
        "johansen": {
            "det_order": 0, "k_ar_diff": 2,
            "trace_stat": full(joh.lr1),
            "max_eig_stat": full(joh.lr2),
            "trace_crit_90_95_99": [full(row) for row in joh.cvt],
            "max_eig_crit_90_95_99": [full(row) for row in joh.cvm],
            "eig": full(joh.eig),
        },
        "vecm_rank1": {
            "alpha": [full(vecm.alpha[i]) for i in range(3)],
            "beta": [full(vecm.beta[i]) for i in range(vecm.beta.shape[0])],
            "gamma": [full(vecm.gamma[i]) for i in range(3)],
            "llf": float(vecm.llf),
        },
    }
    (OUT / "coint.json").write_text(json.dumps(out, separators=(",", ":")))
    print("wrote coint.json")


# --------------------------------------------------- Markov switching
def gen_regime():
    import statsmodels
    from statsmodels.tsa.regime_switching.markov_autoregression import (
        MarkovAutoregression,
    )

    # Simulate a 2-regime AR(1): regime 0 low-mean, regime 1 high-mean.
    rng = np.random.default_rng(19)
    n = 500
    p00, p11 = 0.95, 0.90
    mu = [-1.0, 1.5]
    phi = 0.5
    sigma = [0.8, 1.2]
    state = np.empty(n, dtype=int)
    state[0] = 0
    for t in range(1, n):
        stay = p00 if state[t - 1] == 0 else p11
        state[t] = state[t - 1] if rng.random() < stay else 1 - state[t - 1]
    y = np.empty(n)
    y[0] = mu[state[0]]
    for t in range(1, n):
        m = mu[state[t]]
        y[t] = m + phi * (y[t - 1] - mu[state[t - 1]]) + rng.standard_normal() * sigma[state[t]]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mod = MarkovAutoregression(y, k_regimes=2, order=1, switching_ar=False,
                                   switching_variance=True)
        # Fixed-parameter loglik + smoothed probabilities are deterministic and
        # the stable validation target (EM fits wander into local optima).
        # statsmodels param order: p00, p10, const[0], const[1], sigma2[0],
        # sigma2[1], ar1.  (transition given as p_{00}, p_{10} = 1 - p11.)
        start = np.array([p00, 1 - p11, mu[0], mu[1], sigma[0] ** 2, sigma[1] ** 2, phi])
        res_fixed = mod.smooth(start)

    out = {
        "_meta": {"statsmodels": statsmodels.__version__, "numpy": np.__version__,
                  "note": "MarkovAutoregression(k_regimes=2, order=1, "
                          "switching_ar=False, switching_variance=True); params "
                          "[p00, p10, const0, const1, sigma2_0, sigma2_1, ar1]."},
        "y": full(y),
        "fixed_params": {"p00": p00, "p10": 1 - p11, "const": mu,
                         "sigma2": [sigma[0] ** 2, sigma[1] ** 2], "ar1": phi},
        "loglike_fixed": float(res_fixed.llf),
        "smoothed_prob_regime1": full(res_fixed.smoothed_marginal_probabilities[:, 1]),
        "filtered_prob_regime1": full(res_fixed.filtered_marginal_probabilities[:, 1]),
    }
    (OUT / "regime.json").write_text(json.dumps(out, separators=(",", ":")))
    print("wrote regime.json")


# ----------------------------------------------------------- MIDAS
def gen_midas():
    import statsmodels
    import statsmodels.api as sm

    # Mixed frequency: a quarterly target y driven by 3 monthly lags of a
    # monthly indicator x. U-MIDAS (unrestricted) is exactly OLS of y on the
    # stacked high-frequency lags + const, so statsmodels OLS is the golden.
    rng = np.random.default_rng(23)
    n_q = 160
    m = 3  # months per quarter
    x_monthly = rng.standard_normal(n_q * m)
    K = 6  # high-frequency lags used
    # Build the stacked design: for quarter t, the K most recent monthly obs.
    X = np.empty((n_q - 2, K))
    for t in range(2, n_q):
        end = t * m  # first month of quarter t (0-indexed high-freq)
        X[t - 2] = x_monthly[end - K:end][::-1]  # most-recent-first
    true_w = 0.9 ** np.arange(K)
    y = 1.0 + X @ true_w + rng.standard_normal(n_q - 2) * 0.5

    Xc = sm.add_constant(X)
    ols = sm.OLS(y, Xc).fit()

    # Documented weight-function golden values (self-authored formulas the crate
    # must reproduce): normalized exponential Almon and Beta weights.
    def exp_almon(theta1, theta2, k):
        j = np.arange(1, k + 1)
        w = np.exp(theta1 * j + theta2 * j ** 2)
        return w / w.sum()

    def beta_weights(t1, t2, k):
        x = (np.arange(1, k + 1) - 1) / (k - 1) if k > 1 else np.array([0.0])
        x = np.clip(x, 1e-8, 1 - 1e-8)
        w = x ** (t1 - 1) * (1 - x) ** (t2 - 1)
        return w / w.sum()

    out = {
        "_meta": {"statsmodels": statsmodels.__version__, "numpy": np.__version__,
                  "note": "U-MIDAS = OLS of y on [const, stacked K monthly lags "
                          "most-recent-first]; weight fns normalized to sum 1."},
        "y": full(y),
        "X_stacked": [full(X[:, j]) for j in range(K)],
        "K": K,
        "umidas_ols": {"params": full(ols.params), "bse": full(ols.bse),
                       "rsquared": float(ols.rsquared)},
        "weight_goldens": {
            "exp_almon_0.1_-0.05_K6": full(exp_almon(0.1, -0.05, 6)),
            "beta_2_3_K10": full(beta_weights(2.0, 3.0, 10)),
            "beta_1_5_K8": full(beta_weights(1.0, 5.0, 8)),
        },
    }
    (OUT / "midas.json").write_text(json.dumps(out, separators=(",", ":")))
    print("wrote midas.json")


# ------------------------------------------- multivariate GARCH (DCC)
def gen_mgarch():
    # No available Python/R DCC reference in this venv, so the fixture provides
    # SIMULATED data from a known DCC(a, b) with CCC base for recovery testing,
    # plus the true parameters. The crate validates by (1) the CCC special case
    # (a=b=0 -> constant correlation), (2) simulation recovery within MC bounds,
    # (3) analytic properties (PD correlation matrices, targeting).
    rng = np.random.default_rng(2027)
    n, k = 2500, 3
    # Per-series GARCH(1,1).
    omega = np.array([0.05, 0.03, 0.04])
    alpha = np.array([0.08, 0.06, 0.10])
    beta = np.array([0.90, 0.92, 0.88])
    a_dcc, b_dcc = 0.03, 0.95
    # Unconditional correlation Qbar.
    Qbar = np.array([[1.0, 0.5, 0.3], [0.5, 1.0, 0.4], [0.3, 0.4, 1.0]])
    L = np.linalg.cholesky(Qbar)

    h = np.zeros((n, k))
    eps = np.zeros((n, k))
    Q = Qbar.copy()
    std = np.zeros((n, k))
    h[0] = omega / (1 - alpha - beta)
    for t in range(1, n):
        h[t] = omega + alpha * eps[t - 1] ** 2 + beta * h[t - 1]
        Q = (1 - a_dcc - b_dcc) * Qbar + a_dcc * np.outer(std[t - 1], std[t - 1]) + b_dcc * Q
        d = np.sqrt(np.diag(Q))
        R = Q / np.outer(d, d)
        Lr = np.linalg.cholesky(R)
        z = Lr @ rng.standard_normal(k)
        std[t] = z
        eps[t] = np.sqrt(h[t]) * z

    out = {
        "_meta": {"numpy": np.__version__,
                  "note": "Simulated DCC-GARCH(1,1); no external reference in venv, "
                          "so the crate validates via CCC special case, simulation "
                          "recovery, and PD/targeting properties. Params below are truth."},
        "returns": [full(eps[100:, j]) for j in range(k)],  # drop burn-in
        "true": {"omega": full(omega), "alpha": full(alpha), "beta": full(beta),
                 "a_dcc": a_dcc, "b_dcc": b_dcc, "Qbar": [full(Qbar[i]) for i in range(k)]},
    }
    (OUT / "mgarch.json").write_text(json.dumps(out, separators=(",", ":")))
    print("wrote mgarch.json")


if __name__ == "__main__":
    gen_coint()
    gen_regime()
    gen_midas()
    gen_mgarch()
