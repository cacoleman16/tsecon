"""Canonical valid inputs for every public tsecon callable (audit round 11).

One small, seeded input per function, parameterised by a sample size ``T`` so
the same registry drives the result-contract sweep (E), the signature-drift
sweep (F), the complexity-cliff sweep (G) and the seed-contract sweep (H).

    from registry import build, NAMES
    args, kwargs = build("adf", T=200, seed=0)
    tsecon.adf(*args, **kwargs)

Every builder returns ``(args, kwargs)``; ``kwargs`` may include draw counts
deliberately set SMALL (``n_draws``/``n_boot``) so the sweeps finish. The
``DEFAULT_ONLY`` set lists functions whose canonical call passes nothing but
the required arguments, which is what sweep G's "default arguments" rule
needs; ``SIZE_AXIS`` says what ``T`` scales for the non-series functions.
"""
from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------- #
# base data-generating processes (all seeded, all tiny)
# --------------------------------------------------------------------------- #


def _rng(seed):
    return np.random.default_rng(seed)


def ar1(T, seed=0, phi=0.5, scale=1.0):
    rng = _rng(seed)
    e = rng.standard_normal(T) * scale
    y = np.empty(T)
    prev = 0.0
    for t in range(T):
        prev = phi * prev + e[t]
        y[t] = prev
    return y


def rw(T, seed=0):
    return np.cumsum(_rng(seed).standard_normal(T))


def stable_var(T, k=3, seed=7):
    rng = _rng(seed)
    a = np.array([[0.5, 0.1, 0.0], [0.0, 0.4, 0.1], [0.1, 0.0, 0.3]])[:k, :k]
    y = np.zeros((T, k))
    for t in range(1, T):
        y[t] = a @ y[t - 1] + rng.standard_normal(k)
    return y


def garch_returns(T, seed=3, k=1):
    rng = _rng(seed)
    z = rng.standard_normal((T, k))
    y = np.empty((T, k))
    s2 = np.ones(k)
    for t in range(T):
        y[t] = np.sqrt(s2) * z[t]
        s2 = 0.05 + 0.08 * y[t] ** 2 + 0.88 * s2
    return y[:, 0] if k == 1 else y


def coint_pair(T, seed=11):
    rng = _rng(seed)
    x = np.cumsum(rng.standard_normal(T))
    y = 1.5 * x + rng.standard_normal(T)
    return np.column_stack([y, x])


def seasonal(T, period=4, seed=5):
    rng = _rng(seed)
    t = np.arange(T)
    return 2.0 * np.sin(2 * np.pi * t / period) + 0.02 * t + 0.5 * rng.standard_normal(T)


def design(T, k=2, seed=4, const=True):
    rng = _rng(seed)
    x = rng.standard_normal((T, k))
    return np.column_stack([np.ones(T), x]) if const else x


def yx(T, seed=4, k=2):
    X = design(T, k, seed)
    rng = _rng(seed + 100)
    beta = np.arange(1, X.shape[1] + 1) * 0.5
    y = X @ beta + rng.standard_normal(T)
    return y, X


def factor_panel(T, N=8, seed=3):
    rng = _rng(seed)
    f = np.zeros(T)
    for t in range(1, T):
        f[t] = 0.6 * f[t - 1] + rng.standard_normal()
    load = rng.uniform(0.6, 1.4, N)
    return np.outer(f, load) + 0.5 * rng.standard_normal((T, N))


def yield_panel(T, seed=9):
    """NS-factor yield panel in DECIMALS on integer month maturities 1..12."""
    rng = _rng(seed)
    mats = np.arange(1, 13, dtype=float)
    lam = 0.0609 * 12
    g = (1 - np.exp(-lam * mats)) / (lam * mats)
    h = g - np.exp(-lam * mats)
    L = np.column_stack([np.ones_like(mats), g, h])
    f = np.zeros((T, 3))
    mu = np.array([0.05, -0.02, 0.01])
    f[0] = mu
    for t in range(1, T):
        f[t] = mu + 0.9 * (f[t - 1] - mu) + rng.standard_normal(3) * np.array([0.003, 0.003, 0.004])
    return f @ L.T + 0.0005 * rng.standard_normal((T, len(mats))), mats


def unit_panel(N, T, seed=8, k=2):
    rng = _rng(seed)
    ys, xs = [], []
    for i in range(N):
        x = rng.standard_normal((T, k))
        y = 0.5 + x @ np.array([1.0, -0.5][:k]) + rng.standard_normal(T)
        ys.append(y)
        xs.append(x)
    return ys, xs


def pmg_units(N, T, seed=8):
    """Stationary ARDL(1,1) units sharing one long-run coefficient."""
    rng = _rng(seed)
    ys, xs = [], []
    for i in range(N):
        x = np.cumsum(rng.standard_normal(T)) * 0.3
        y = np.zeros(T)
        for t in range(1, T):
            y[t] = 0.5 * y[t - 1] + 0.5 * 1.0 * x[t] + rng.standard_normal() * 0.5
        ys.append(y)
        xs.append(x.reshape(-1, 1))
    return ys, xs


def balanced_panel(N, T, seed=8):
    rng = _rng(seed)
    return np.array([ar1(T, seed + i, phi=0.5) for i in range(N)]) + rng.standard_normal((N, 1))


def lpdid_panel(N, T, seed=13):
    rng = _rng(seed)
    y = np.zeros((N, T))
    d = np.zeros((N, T))
    fe = rng.standard_normal(N)
    for i in range(N):
        if i < N * 2 // 3:
            start = rng.integers(T // 4, 3 * T // 4)
            d[i, start:] = 1.0
        for t in range(T):
            y[i, t] = fe[i] + 0.3 * t / T + 1.0 * d[i, t] + rng.standard_normal() * 0.5
    return y, d


def curves(T, M=6, seed=2):
    rng = _rng(seed)
    return rng.standard_normal((T, M)) @ np.diag(np.linspace(1.5, 0.3, M))


def mean_var_moments(y):
    def f(theta):
        mu, s2 = theta
        return np.column_stack([y - mu, (y - mu) ** 2 - s2])

    return f


# --------------------------------------------------------------------------- #
# the registry: name -> builder(T, seed) -> (args, kwargs)
# --------------------------------------------------------------------------- #
R = {}


def reg(name):
    def deco(fn):
        R[name] = fn
        return fn

    return deco


def _h(T):
    """A horizon that stays small at every T (keeps IRF-shaped outputs cheap)."""
    return 6


# ----- diagnostics
reg("acf")(lambda T, s: ((ar1(T, s),), {}))
reg("pacf")(lambda T, s: ((ar1(T, s),), {}))
reg("ljung_box")(lambda T, s: ((ar1(T, s),), {}))
reg("jarque_bera")(lambda T, s: ((ar1(T, s),), {}))
reg("arch_lm")(lambda T, s: ((garch_returns(T, s),), {}))
# ----- unit roots
reg("adf")(lambda T, s: ((ar1(T, s),), {}))
reg("kpss")(lambda T, s: ((ar1(T, s),), {}))
reg("check_stationarity")(lambda T, s: ((ar1(T, s),), {}))
reg("phillips_perron")(lambda T, s: ((ar1(T, s),), {}))
reg("dfgls")(lambda T, s: ((ar1(T, s),), {}))
reg("ng_perron")(lambda T, s: ((ar1(T, s),), {}))
reg("phillips_ouliaris")(lambda T, s: ((coint_pair(T, s)[:, 0], coint_pair(T, s)[:, 1:2]), {}))  # x is (T, k)
reg("zivot_andrews")(lambda T, s: ((ar1(T, s),), {}))
reg("ndiffs")(lambda T, s: ((rw(T, s),), {}))
reg("nsdiffs")(lambda T, s: ((seasonal(T, 4, s), 4), {}))
reg("box_cox_lambda")(lambda T, s: ((np.exp(ar1(T, s) * 0.3) + 1.0,), {}))
reg("check_series")(lambda T, s: ((ar1(T, s),), {}))
reg("summarize")(lambda T, s: (({"a": 1.0, "b": np.arange(3.0)},), {}))
# ----- robust inference
reg("long_run_variance")(lambda T, s: ((ar1(T, s),), {}))
reg("ols")(lambda T, s: (yx(T, s), {}))
# ----- bootstrap
reg("bootstrap_indices")(lambda T, s: ((T,), {"scheme": "stationary", "seed": s, "p": 0.1}))
reg("optimal_block_length")(lambda T, s: ((ar1(T, s),), {}))
reg("philox_uniforms")(lambda T, s: ((s, T), {}))
# ----- state space
reg("local_level_smooth")(lambda T, s: ((rw(T, s) + _rng(s).standard_normal(T), 1.0, 0.5), {}))
reg("ar_loglik")(lambda T, s: ((ar1(T, s), [0.5], 1.0), {}))
# ----- ARIMA / GARCH
reg("arima_fit")(lambda T, s: ((ar1(T, s),), {"p": 1, "forecast_steps": 3, "conf_alpha": 0.1}))
reg("auto_arima")(lambda T, s: ((ar1(T, s),), {"max_p": 2, "max_q": 2}))
reg("garch_fit")(lambda T, s: ((garch_returns(T, s),), {"forecast_horizon": 3}))
# ----- VAR
reg("var_fit")(lambda T, s: ((stable_var(T, 3, s),), {}))
reg("var_irf")(lambda T, s: ((stable_var(T, 3, s),), {"horizon": _h(T)}))
reg("var_irf_bands")(lambda T, s: ((stable_var(T, 3, s),), {"horizon": _h(T)}))
reg("var_fevd")(lambda T, s: ((stable_var(T, 3, s),), {"horizon": _h(T)}))
reg("var_forecast")(lambda T, s: ((stable_var(T, 3, s),), {"steps": 4}))
reg("var_granger")(lambda T, s: ((stable_var(T, 3, s), [0], [1]), {}))
# ----- Bayesian VAR
reg("bvar_fit")(lambda T, s: ((stable_var(T, 3, s),), {}))
reg("bvar_irf_draws")(lambda T, s: ((stable_var(T, 3, s),), {"horizon": _h(T), "n_draws": 30, "seed": s}))
reg("bvar_hierarchical")(lambda T, s: ((stable_var(T, 3, s),), {}))
reg("bvar_ssvs")(lambda T, s: ((stable_var(T, 3, s),), {"n_draws": 200, "burn": 50, "seed": s, "horizon": _h(T)}))
reg("mcmc_diagnostics")(lambda T, s: ((np.array([ar1(T, s + i, 0.3) for i in range(4)]),), {}))
# ----- filters
reg("hp_filter")(lambda T, s: ((rw(T, s),), {}))
reg("bk_filter")(lambda T, s: ((rw(T, s),), {}))
reg("cf_filter")(lambda T, s: ((rw(T, s),), {}))
reg("hamilton_filter")(lambda T, s: ((rw(T, s),), {}))
reg("bn_filter")(lambda T, s: ((rw(T, s),), {}))
reg("bn_decomposition")(lambda T, s: ((rw(T, s),), {}))
reg("stl")(lambda T, s: ((seasonal(T, 4, s), 4), {}))
reg("mstl")(lambda T, s: ((seasonal(T, 4, s), [4, 12]), {}))
reg("seasonal_strength")(lambda T, s: ((seasonal(T, 4, s), 4), {}))
# ----- forecasting / evaluation
reg("dm_test")(lambda T, s: ((ar1(T, s), ar1(T, s + 1)), {}))
reg("accuracy")(lambda T, s: ((ar1(T, s)[T // 2:], ar1(T, s + 1)[T // 2:], ar1(T, s)[: T // 2]), {}))
reg("theta_forecast")(lambda T, s: ((ar1(T, s) + 10.0, 4), {}))
# ----- local projections
reg("lp")(lambda T, s: ((ar1(T, s), _rng(s + 1).standard_normal(T)), {"horizons": _h(T)}))
reg("lp_iv")(lambda T, s: ((ar1(T, s), _rng(s + 1).standard_normal(T) + 0.5 * _rng(s + 2).standard_normal(T), _rng(s + 2).standard_normal(T)), {"horizons": _h(T)}))
reg("lp_multiplier")(lambda T, s: ((ar1(T, s), _rng(s + 1).standard_normal(T) + 0.5 * _rng(s + 2).standard_normal(T), _rng(s + 2).standard_normal(T)), {"horizons": _h(T)}))
# ----- penalized
reg("ridge")(lambda T, s: ((design(T, 3, s, const=False), yx(T, s, 3)[0], 1.0), {}))
reg("elastic_net")(lambda T, s: ((design(T, 3, s, const=False), yx(T, s, 3)[0], 0.1), {}))
reg("lasso")(lambda T, s: ((design(T, 3, s, const=False), yx(T, s, 3)[0], 0.1), {}))
# ----- structural identification
POS = [(0, 0, 0, "+")]
reg("sign_restricted_svar")(lambda T, s: ((stable_var(T, 3, s), POS), {"horizon": _h(T), "n_draws": 40, "seed": s}))
reg("zero_sign_svar")(lambda T, s: ((stable_var(T, 3, s), [], [(0, 1, 0)]), {"horizon": _h(T), "n_draws": 40, "seed": s}))
reg("structural_fevd")(lambda T, s: ((stable_var(T, 3, s),), {"horizon": _h(T)}))
reg("historical_decomposition")(lambda T, s: ((stable_var(T, 3, s),), {}))
reg("narrative_svar")(lambda T, s: ((stable_var(T, 3, s), POS), {"horizon": _h(T), "n_draws": 40, "seed": s}))
reg("fry_pagan_svar")(lambda T, s: ((stable_var(T, 3, s), POS), {"horizon": _h(T), "n_draws": 40, "seed": s}))
reg("robust_svar_bounds")(lambda T, s: ((stable_var(T, 3, s), POS), {"horizon": _h(T), "n_draws": 40, "seed": s}))
reg("long_run_svar")(lambda T, s: ((stable_var(T, 3, s),), {"horizon": _h(T)}))
reg("max_share_svar")(lambda T, s: ((stable_var(T, 3, s),), {"h1": 8, "horizon": 8}))


def _proxy(T, s):
    d = stable_var(T, 3, s)
    rng = _rng(s + 50)
    u = np.diff(d, axis=0)[:, 0]
    proxy = np.r_[np.nan, 0.8 * u + 0.5 * rng.standard_normal(T - 1)]
    return d, proxy


reg("proxy_svar_bands")(lambda T, s: (_proxy(T, s), {"horizon": _h(T), "n_boot": 49, "seed": s}))
reg("proxy_ar_sets")(lambda T, s: (_proxy(T, s), {"horizon": _h(T)}))
reg("proxy_svar")(lambda T, s: (_proxy(T, s), {"horizon": _h(T)}))
reg("proxy_first_stage")(lambda T, s: (_proxy(T, s), {}))


def _nongauss(T, s):
    rng = _rng(s)
    a = np.array([[0.5, 0.1, 0.0], [0.0, 0.4, 0.1], [0.1, 0.0, 0.3]])
    B = np.array([[1.0, 0.3, 0.0], [0.2, 1.0, 0.4], [0.0, 0.1, 1.0]])
    e = rng.laplace(size=(T, 3)) * np.array([1.0, 0.7, 1.3])
    y = np.zeros((T, 3))
    for t in range(1, T):
        y[t] = a @ y[t - 1] + B @ e[t]
    return y


reg("nongaussian_svar")(lambda T, s: ((_nongauss(T, s),), {"horizon": _h(T)}))
reg("hetero_svar")(lambda T, s: ((np.vstack([stable_var(T // 2, 3, s), 2.0 * stable_var(T - T // 2, 3, s + 1)]), (np.arange(T) >= T // 2).astype(np.int64)), {"horizon": _h(T)}))
# ----- panel
reg("panel_fe")(lambda T, s: ((balanced_panel(6, T, s), np.array([balanced_panel(6, T, s + 1), balanced_panel(6, T, s + 2)])), {}))
reg("panel_lp")(lambda T, s: ((balanced_panel(6, T, s), _rng(s + 9).standard_normal(T)), {"horizon": _h(T)}))
reg("lp_did")(lambda T, s: (lpdid_panel(12, T, s), {"pre_window": 2, "post_window": 3}))
# ----- forecast comparison
reg("cw_test")(lambda T, s: ((ar1(T, s), ar1(T, s + 1), ar1(T, s + 2), ar1(T, s + 3)), {}))
reg("gw_test")(lambda T, s: ((ar1(T, s) ** 2, ar1(T, s + 1) ** 2), {}))
reg("var_backtest")(lambda T, s: ((garch_returns(T, s), np.full(T, -1.7)), {"alpha": 0.05}))
# ----- spectral
reg("periodogram")(lambda T, s: ((ar1(T, s),), {}))
reg("welch")(lambda T, s: ((ar1(T, s),), {"nperseg": 64}))
reg("coherence")(lambda T, s: ((ar1(T, s), ar1(T, s + 1)), {"nperseg": 64}))
# ----- cointegration
reg("johansen")(lambda T, s: ((coint_pair(T, s),), {}))
reg("engle_granger")(lambda T, s: ((coint_pair(T, s),), {}))
reg("vecm")(lambda T, s: ((coint_pair(T, s),), {}))
reg("threshold_vecm")(lambda T, s: ((coint_pair(T, s),), {"n_grid_gamma": 30, "n_grid_beta": 10}))
reg("hansen_seo_test")(lambda T, s: ((coint_pair(T, s),), {"n_grid": 30, "n_boot": 49, "seed": s}))
reg("ou_fit")(lambda T, s: ((ar1(T, s, 0.8),), {}))
reg("spread_zscore")(lambda T, s: ((ar1(T, s, 0.8),), {}))
# ----- regime switching


def _ms(T, s):
    rng = _rng(s)
    y = np.empty(T)
    state, prev = 0, 0.0
    for t in range(T):
        if rng.random() < 0.05:
            state = 1 - state
        mu = (-1.0, 1.5)[state]
        prev = mu + 0.3 * (prev - mu) + 0.5 * rng.standard_normal()
        y[t] = prev
    return y


def _setar(T, s):
    rng = _rng(s)
    y = np.zeros(T)
    for t in range(1, T):
        y[t] = (0.6 if y[t - 1] <= 0 else -0.4) * y[t - 1] + rng.standard_normal()
    return y


reg("markov_switching_ar")(lambda T, s: ((_ms(T, s),), {}))
reg("setar")(lambda T, s: ((_setar(T, s), 1), {}))
reg("setar_test")(lambda T, s: ((_setar(T, s), 1), {"n_boot": 49, "seed": s}))
reg("star")(lambda T, s: ((_setar(T, s), 1), {"n_gamma": 8, "n_c": 8}))
reg("star_eval")(lambda T, s: ((_setar(T, s), 1, 2.0, 0.0), {}))
reg("star_test")(lambda T, s: ((_setar(T, s), 1), {}))
reg("threshold_var")(lambda T, s: ((stable_var(T, 2, s), 1), {}))
reg("threshold_var_test")(lambda T, s: ((stable_var(T, 2, s), 1), {"n_grid": 30, "n_boot": 49, "seed": s}))
# ----- MIDAS
reg("midas_weights")(lambda T, s: (("exp_almon", 0.1, -0.05, 6), {}))


def _midas(T, s):
    rng = _rng(s)
    X = rng.standard_normal((T, 6))
    w = np.exp(-0.5 * np.arange(6))
    w /= w.sum()
    y = 1.0 + 2.0 * (X @ w) + 0.1 * rng.standard_normal(T)
    return y, X


reg("umidas")(lambda T, s: (_midas(T, s), {}))
reg("weighted_midas")(lambda T, s: (_midas(T, s), {}))
# ----- multivariate GARCH
reg("ccc_garch")(lambda T, s: ((garch_returns(T, s, 2),), {"forecast_horizon": 2}))
reg("dcc_garch")(lambda T, s: ((garch_returns(T, s, 2),), {"forecast_horizon": 2}))
reg("dcc_test")(lambda T, s: ((garch_returns(T, s, 2),), {}))
# ----- realized vol
reg("realized_measures")(lambda T, s: ((garch_returns(T, s) * 0.01,), {}))
reg("har_rv")(lambda T, s: ((np.exp(ar1(T, s, 0.7) * 0.3),), {}))
reg("connectedness")(lambda T, s: ((stable_var(T, 3, s),), {"horizon": _h(T)}))
reg("factor_model")(lambda T, s: ((factor_panel(T, 8, s),), {"n_factors": 2, "kmax": 4}))
# ----- term structure
_MATS = np.array([3.0, 6.0, 12.0, 24.0, 36.0, 60.0, 84.0, 120.0])


def _ylds(s, n=8):
    m = np.linspace(3.0, 120.0, n)
    lam = 0.0609
    g = (1 - np.exp(-lam * m)) / (lam * m)
    h = g - np.exp(-lam * m)
    return m, 5.0 - 2.0 * g + 1.5 * h + 0.02 * _rng(s).standard_normal(n)


reg("nelson_siegel")(lambda T, s: (_ylds(s, max(8, T // 25)), {}))
reg("svensson")(lambda T, s: ((*_ylds(s, max(8, T // 25)), 0.0609, 0.2), {}))
reg("afns_adjustment")(lambda T, s: ((np.linspace(3.0, 120.0, max(8, T // 25)), np.array([0.01, 0.01, 0.01])), {}))
# ----- GMM / IV


def _iv(T, s):
    rng = _rng(s)
    z = rng.standard_normal((T, 2))
    v = rng.standard_normal(T)
    x_end = z @ np.array([1.0, 0.5]) + v
    x = np.column_stack([np.ones(T), x_end])
    y = 1.0 + 0.5 * x_end + 0.5 * v + rng.standard_normal(T)
    Z = np.column_stack([np.ones(T), z])
    return x, Z, y


reg("iv_gmm")(lambda T, s: (_iv(T, s), {}))
reg("cv_splits")(lambda T, s: ((T,), {"train": T // 2}))
reg("adaptive_lasso")(lambda T, s: ((design(T, 3, s, const=False), yx(T, s, 3)[0], 0.1), {}))
reg("lasso_path")(lambda T, s: ((design(T, 3, s, const=False), yx(T, s, 3)[0]), {"n_lambdas": 20}))
# ----- backtest / conformal
reg("backtest")(lambda T, s: ((ar1(T, s) + 5.0,), {"train": T // 2, "horizon": 2}))
reg("conformal_forecast")(lambda T, s: ((ar1(T, s) + 5.0,), {"horizon": 2}))
reg("conformal_backtest")(lambda T, s: ((ar1(T, s) + 5.0,), {"horizon": 2, "n_eval": 10}))
# ----- nonlinear GMM
reg("gmm_nonlinear")(lambda T, s: ((mean_var_moments(ar1(T, s)), [0.0, 1.0]), {}))
# ----- state LP
reg("lp_state")(lambda T, s: ((ar1(T, s), _rng(s + 1).standard_normal(T), (_rng(s + 2).random(T) > 0.5).astype(float)), {"horizons": _h(T)}))
# ----- MG panel VAR


def _entities(T, s, n_units=4):
    rng = _rng(s)
    base = np.array([[0.5, 0.1], [0.05, 0.4]])
    out = []
    for _ in range(n_units):
        A = base + 0.05 * rng.standard_normal((2, 2))
        Y = np.zeros((T, 2))
        y = np.zeros(2)
        for t in range(T):
            y = A @ y + 0.5 * rng.standard_normal(2)
            Y[t] = y
        out.append(Y)
    return out


reg("mean_group_var")(lambda T, s: ((_entities(T, s),), {"horizon": _h(T)}))
reg("dynamic_ns")(lambda T, s: ((yield_panel(T, s)[0] * 100.0, yield_panel(T, s)[1]), {}))
reg("favar")(lambda T, s: ((factor_panel(T, 8, s), ar1(T, s + 1)), {"horizon": _h(T)}))
# ----- realized extras
reg("realized_quarticity")(lambda T, s: ((garch_returns(T, s) * 0.01,), {}))
reg("tripower_quarticity")(lambda T, s: ((garch_returns(T, s) * 0.01,), {}))
reg("bns_jump_test")(lambda T, s: ((garch_returns(T, s) * 0.01,), {}))


def _ohlc(T, s):
    rng = _rng(s)
    c = 100.0 * np.exp(np.cumsum(0.01 * rng.standard_normal(T)))  # geometric: stays positive
    o = np.r_[100.0, c[:-1]]
    hi = np.maximum(o, c) * (1 + 0.005 * np.abs(rng.standard_normal(T)))
    lo = np.minimum(o, c) * (1 - 0.005 * np.abs(rng.standard_normal(T)))
    return hi, lo


reg("realized_range")(lambda T, s: (_ohlc(T, s), {}))
# ----- GAS / DCS
reg("gas_volatility")(lambda T, s: ((garch_returns(T, s),), {"horizon": 3}))
reg("dcs_local_level")(lambda T, s: ((rw(T, s) + _rng(s).standard_normal(T),), {}))
# ----- heterogeneous panel
reg("panel_mean_group")(lambda T, s: (unit_panel(5, T, s), {}))
reg("panel_pmg")(lambda T, s: (pmg_units(5, T, s), {}))
reg("panel_unit_root")(lambda T, s: ((balanced_panel(5, T, s),), {"lags": 1}))
# ----- DFM


def _dfm(T, s):
    x = factor_panel(T, 8, s)
    x[-1, :3] = np.nan
    return x


reg("dfm_nowcast")(lambda T, s: ((_dfm(T, s),), {}))
reg("dfm_news")(lambda T, s: ((_dfm(T, s), factor_panel(T, 8, s)), {}))
# ----- predictive regressions


def _predreg(T, s, k=1):
    rng = _rng(s)
    x = np.zeros((T, k))
    for t in range(1, T):
        x[t] = 0.95 * x[t - 1] + rng.standard_normal(k)
    r = 0.05 * x[:, 0] + rng.standard_normal(T)
    return r[1:], x[:-1]


reg("predictive_regression")(lambda T, s: ((_predreg(T, s)[0], _predreg(T, s)[1][:, 0]), {}))
reg("ivx_test")(lambda T, s: (_predreg(T, s, 2), {}))


def _probit(T, s):
    rng = _rng(s)
    x = ar1(T, s, 0.8)
    p = 1 / (1 + np.exp(-(-0.5 + 1.0 * x)))
    y = (rng.random(T) < p).astype(float)
    return y, np.column_stack([np.ones(T), x])


reg("recession_probit")(lambda T, s: (_probit(T, s), {}))
# ----- survey expectations
reg("cg_regression")(lambda T, s: ((ar1(T, s), ar1(T, s + 1)), {"maxlags": 4}))
reg("forecast_efficiency")(lambda T, s: ((ar1(T, s), design(T, 1, s + 1, const=False)), {"maxlags": 4}))  # adds its own constant
reg("forecast_disagreement")(lambda T, s: (([_rng(s + t).standard_normal(5) for t in range(max(4, T // 20))],), {}))
# ----- long memory
reg("frac_diff")(lambda T, s: ((ar1(T, s), 0.4), {}))
reg("frac_integrate")(lambda T, s: ((ar1(T, s), 0.4), {}))
reg("long_memory_d")(lambda T, s: ((ar1(T, s),), {}))
# ----- specification tests
reg("heteroskedasticity_test")(lambda T, s: (yx(T, s), {}))
reg("reset_test")(lambda T, s: (yx(T, s), {}))
reg("chow_test")(lambda T, s: ((*yx(T, s), T // 2), {}))
reg("cusum_test")(lambda T, s: (yx(T, s), {}))
# ----- ACM
reg("acm_term_premium")(lambda T, s: ((yield_panel(T, s)[0], list(range(1, 13))), {"n_factors": 3}))
# ----- DSGE
reg("dsge_solve")(lambda T, s: ((np.array([[1.0, 0.0], [0.0, 0.5]]), np.array([[0.9, 0.0], [-1.0, 1.0]]), np.array([[1.0], [0.0]]), 1), {}))
# ----- quantile
reg("quantile_regression")(lambda T, s: (yx(T, s), {"taus": [0.25, 0.5, 0.75]}))
reg("quantile_lp")(lambda T, s: ((ar1(T, s), _rng(s + 1).standard_normal(T)), {"horizons": 3, "taus": [0.25, 0.5, 0.75]}))
reg("growth_at_risk")(lambda T, s: ((ar1(T, s), ar1(T, s + 1).reshape(-1, 1)), {"horizon": 2, "taus": [0.1, 0.5, 0.9]}))
# ----- functional shocks
reg("functional_pca")(lambda T, s: ((curves(T, 6, s),), {"n_factors": 2}))


def _flp(T, s):
    rng = _rng(s)
    c = curves(T, 6, s)
    sc = c[:, :2]
    y = np.zeros(T)
    for t in range(1, T):
        y[t] = 0.4 * y[t - 1] + 0.7 * sc[t, 0] - 0.3 * sc[t, 1] + rng.standard_normal()
    return y, c


reg("flp")(lambda T, s: ((_flp(T, s)[0], _flp(T, s)[1][:, :2]), {"horizons": 4, "n_lag_controls": 1}))
reg("flp_scenario")(lambda T, s: ((*_flp(T, s), np.linspace(1.0, 0.0, 6)), {"n_factors": 2, "horizons": 4, "n_lag_controls": 1}))
reg("fvar_scenario")(lambda T, s: ((*_flp(T, s), np.linspace(1.0, 0.0, 6)), {"n_factors": 2, "lags": 1, "horizon": 4}))
# ----- structural breaks


def _break(T, s):
    rng = _rng(s)
    y = rng.standard_normal(T) + np.where(np.arange(T) < T // 2, 0.0, 2.0)
    return y, np.ones((T, 1))


reg("bai_perron")(lambda T, s: (_break(T, s), {"max_breaks": 2}))
reg("sup_f_test")(lambda T, s: (_break(T, s), {}))
# ----- smooth LP
reg("smooth_lp")(lambda T, s: ((ar1(T, s), _rng(s + 1).standard_normal(T)), {"horizons": _h(T), "lam": 1.0}))
# ----- EVT
reg("gpd_fit")(lambda T, s: ((np.abs(_rng(s).standard_t(4, T)),), {}))
reg("gev_fit")(lambda T, s: ((np.abs(_rng(s).standard_t(4, T)),), {"block_size": max(2, T // 40)}))
# ----- copulas


def _u(T, s):
    rng = _rng(s)
    z = rng.multivariate_normal([0, 0], [[1, 0.6], [0.6, 1]], size=T)
    from scipy.stats import rankdata

    return np.column_stack([rankdata(z[:, j]) / (T + 1) for j in range(2)])


reg("pseudo_obs")(lambda T, s: ((_rng(s).standard_normal((T, 2)),), {}))
reg("copula_fit")(lambda T, s: ((_u(T, s),), {}))
reg("copula_select")(lambda T, s: ((_u(T, s),), {"families": ["gaussian", "frank"]}))

NAMES = sorted(R)


def build(name, T=200, seed=0):
    args, kwargs = R[name](T, seed)
    return list(args), dict(kwargs)


def reseed(name, T, seed, kwargs):
    """Return kwargs with the function's seed kwarg (if any) set to ``seed``."""
    for key in ("seed", "band_seed", "rf_seed"):
        if key in kwargs:
            kwargs = {**kwargs, key: seed}
    return kwargs


if __name__ == "__main__":
    import tsecon

    public = sorted(n for n in dir(tsecon) if not n.startswith("_") and callable(getattr(tsecon, n)))
    missing = sorted(set(public) - set(NAMES))
    extra = sorted(set(NAMES) - set(public))
    print(f"public={len(public)} registry={len(NAMES)} missing={missing} extra={extra}")
