"""Type stubs for tsecon — a high-performance time series econometrics library.

The runtime module is a compiled Rust extension (PyO3); this stub describes
its public surface for static type checkers and IDEs. Array outputs are
NumPy float64 arrays; dict outputs use the documented keys. Kept in sync
with `bindings/python/src/lib.rs` (a CI check asserts they agree).
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import numpy.typing as npt

__version__: str

_F64 = npt.NDArray[np.float64]
_ArrayLike = npt.ArrayLike

# ----------------------------------------------------------- diagnostics
def acf(y: _ArrayLike, nlags: int = ..., adjusted: bool = ...) -> dict[str, _F64]:
    """Autocorrelation function with Bartlett standard errors."""

def pacf(y: _ArrayLike, nlags: int = ..., method: str = ...) -> _F64:
    """Partial autocorrelation function; `method` is "yw" or "ols"."""

def ljung_box(y: _ArrayLike, nlags: int = ...) -> dict[str, _F64]:
    """Ljung-Box and Box-Pierce portmanteau tests for lags 1..=nlags."""

def jarque_bera(x: _ArrayLike) -> dict[str, float]:
    """Jarque-Bera normality test (statistic, p_value, skewness, kurtosis, n)."""

def arch_lm(resid: _ArrayLike, nlags: int = ...) -> dict[str, float]:
    """Engle's ARCH-LM test for conditional heteroskedasticity."""

# --------------------------------------------------- unit roots / workflow
def adf(
    y: _ArrayLike,
    regression: str = ...,
    autolag: str | None = ...,
    maxlag: int | None = ...,
) -> dict[str, Any]:
    """Augmented Dickey-Fuller test with MacKinnon p-values."""

def kpss(
    y: _ArrayLike, regression: str = ..., nlags: str | int | None = ...
) -> dict[str, Any]:
    """KPSS stationarity test (null: stationary)."""

def check_stationarity(y: _ArrayLike, alpha: float = ...) -> dict[str, Any]:
    """The ADF + KPSS confirmatory-quadrant workflow with a recommendation."""

# ------------------------------------------------------- robust inference
def long_run_variance(
    x: _ArrayLike, kernel: str = ..., bandwidth: float | None = ...
) -> float:
    """Kernel long-run variance of a series (demeaned internally)."""

def ols(
    y: _ArrayLike,
    x: _ArrayLike,
    se_type: str = ...,
    maxlags: int | None = ...,
    use_correction: bool = ...,
) -> dict[str, Any]:
    """OLS with nonrobust / HC0 / HC1 / HAC standard errors."""

# -------------------------------------------------------------- bootstrap
def bootstrap_indices(
    n: int,
    scheme: str = ...,
    seed: int = ...,
    block_length: int | None = ...,
    p: float | None = ...,
) -> npt.NDArray[np.uint64]:
    """Bootstrap resampling indices (iid/moving/circular/stationary)."""

def optimal_block_length(y: _ArrayLike) -> dict[str, float]:
    """Politis-White (2004) automatic block length (stationary, circular)."""

def philox_uniforms(seed: int, n: int) -> _F64:
    """Uniform draws from the Philox stream; bit-identical to NumPy."""

# ------------------------------------------------------------ state space
def local_level_smooth(
    y: _ArrayLike, sigma2_eps: float, sigma2_eta: float
) -> dict[str, Any]:
    """Exact-diffuse local-level Kalman filter + smoother (NaN = missing)."""

def ar_loglik(
    y: _ArrayLike, coeffs: Sequence[float], sigma2: float, intercept: float = ...
) -> float:
    """Exact Gaussian log-likelihood of an AR(p) at fixed parameters."""

# ---------------------------------------------------------------- ARIMA
def arima_fit(
    y: _ArrayLike,
    p: int = ...,
    d: int = ...,
    q: int = ...,
    constant: bool = ...,
    forecast_steps: int = ...,
    conf_alpha: float | None = ...,
) -> dict[str, Any]:
    """Exact-MLE ARIMA(p,d,q) fit, with optional forecast + conf_alpha bands."""

# -------------------------------------------------------------- GARCH
def garch_fit(
    y: _ArrayLike,
    vol: str = ...,
    mean: str = ...,
    dist: str = ...,
    p: int = ...,
    o: int = ...,
    q: int = ...,
    forecast_horizon: int = ...,
) -> dict[str, Any]:
    """GARCH/GJR/EGARCH QMLE with MLE and Bollerslev-Wooldridge robust SEs."""

# --------------------------------------------------------------- VAR
def var_fit(data: _ArrayLike, lags: int = ..., trend: str = ...) -> dict[str, Any]:
    """Fit a VAR(p) by OLS; params, sigma_u, ICs, and max stability root."""

def var_irf(
    data: _ArrayLike,
    lags: int = ...,
    horizon: int = ...,
    orth: bool = ...,
    trend: str = ...,
    cumulative: bool = ...,
) -> list[list[list[float]]]:
    """Impulse responses [h][response][shock]; `cumulative` gives running sums."""

def var_fevd(
    data: _ArrayLike, lags: int = ..., horizon: int = ..., trend: str = ...
) -> list[list[list[float]]]:
    """Forecast-error variance decomposition [h][variable][shock]."""

def var_forecast(
    data: _ArrayLike,
    lags: int = ...,
    steps: int = ...,
    alpha: float = ...,
    trend: str = ...,
) -> dict[str, Any]:
    """Iterated VAR point forecasts with (1-alpha) intervals."""

def var_granger(
    data: _ArrayLike,
    caused: Sequence[int],
    causing: Sequence[int],
    lags: int = ...,
    trend: str = ...,
) -> dict[str, Any]:
    """Granger-causality F test (matches statsmodels test_causality)."""

# --------------------------------------------------------- Bayesian VAR
def bvar_fit(
    data: _ArrayLike,
    lags: int = ...,
    lambda0: float = ...,
    lambda1: float = ...,
    lambda3: float = ...,
    delta: float = ...,
) -> dict[str, Any]:
    """Minnesota-NIW conjugate BVAR posterior + log marginal likelihood."""

def bvar_irf_draws(
    data: _ArrayLike,
    lags: int = ...,
    horizon: int = ...,
    n_draws: int = ...,
    seed: int = ...,
    lambda0: float = ...,
    lambda1: float = ...,
    lambda3: float = ...,
    delta: float = ...,
    cumulative: bool = ...,
) -> list[list[list[list[float]]]]:
    """Posterior Cholesky-IRF draws [draw][h][variable][shock] for credible bands."""

def mcmc_diagnostics(chains: _ArrayLike) -> dict[str, float]:
    """Rank-normalized split R-hat and bulk/tail ESS (ArviZ-exact)."""

# ------------------------------------------------------------- filters
def hp_filter(y: _ArrayLike, lamb: float = ..., one_sided: bool = ...) -> dict[str, Any]:
    """Hodrick-Prescott filter (O(n)); `one_sided=True` for the real-time variant."""

def bk_filter(
    y: _ArrayLike, low: float = ..., high: float = ..., k: int = ...
) -> dict[str, Any]:
    """Baxter-King band-pass filter (loses k observations at each end)."""

def cf_filter(
    y: _ArrayLike, low: float = ..., high: float = ..., drift: bool = ...
) -> dict[str, Any]:
    """Christiano-Fitzgerald asymmetric band-pass filter."""

def hamilton_filter(y: _ArrayLike, h: int = ..., p: int = ...) -> dict[str, Any]:
    """Hamilton (2018) regression filter — the modern HP alternative."""

# ------------------------------------------------- forecasting / evaluation
def dm_test(
    e1: _ArrayLike, e2: _ArrayLike, h: int = ..., loss: str = ...
) -> dict[str, float]:
    """Diebold-Mariano test with the Harvey-Leybourne-Newbold correction."""

def accuracy(
    actual: _ArrayLike,
    forecast: _ArrayLike,
    insample: _ArrayLike | None = ...,
    period: int = ...,
) -> dict[str, float]:
    """Forecast accuracy measures (ME/RMSE/MAE/MAPE/sMAPE/MASE/RMSSE)."""

def theta_forecast(y: _ArrayLike, steps: int, period: int = ...) -> _F64:
    """The Theta method (Assimakopoulos-Nikolopoulos 2000)."""

# ------------------------------------------------------- local projections
def lp(
    y: _ArrayLike,
    shock: _ArrayLike,
    horizons: int = ...,
    n_lag_controls: int = ...,
    se: str = ...,
    maxlags: int | None = ...,
    cumulative: bool = ...,
) -> dict[str, Any]:
    """Local projection IRFs; `se` is "lag_augmented" (default) or "hac"."""

def lp_iv(
    y: _ArrayLike,
    impulse: _ArrayLike,
    instrument: _ArrayLike,
    horizons: int = ...,
    n_lag_controls: int = ...,
    cumulative: bool = ...,
) -> dict[str, Any]:
    """LP-IV: instrumented local projections with a first-stage F diagnostic."""

# -------------------------------------------------- penalized regression
def ridge(x: _ArrayLike, y: _ArrayLike, alpha: float) -> _F64:
    """Ridge regression (closed form); scikit-learn `Ridge` objective."""

def elastic_net(
    x: _ArrayLike,
    y: _ArrayLike,
    alpha: float,
    l1_ratio: float = ...,
    tol: float = ...,
    max_iter: int = ...,
) -> dict[str, Any]:
    """Elastic-net via coordinate descent; scikit-learn objective."""

def lasso(
    x: _ArrayLike,
    y: _ArrayLike,
    alpha: float,
    tol: float = ...,
    max_iter: int = ...,
) -> dict[str, Any]:
    """Lasso (elastic net with l1_ratio = 1.0)."""

# --------------------------------------------------- structural identification
def sign_restricted_svar(
    data: _ArrayLike,
    restrictions: Sequence[tuple[int, int, int, str]],
    lags: int = ...,
    horizon: int = ...,
    n_draws: int = ...,
    max_tries: int = ...,
    seed: int = ...,
    lambda1: float = ...,
) -> dict[str, Any]:
    """Sign-restricted Bayesian SVAR: identified-set bands + acceptance diagnostics.

    `restrictions` are (variable, shock, horizon, sign) tuples with sign in
    {"+", "-"}. Returns per-(horizon, variable, shock) `quantiles` at
    `probs=[0.05,0.16,0.50,0.84,0.95]`, the identified-set envelope
    (`set_min`/`set_max`), and `diagnostics`.
    """

# ------------------------------------------------------------------ panel
def panel_fe(
    outcome: _ArrayLike,
    regressors: _ArrayLike,
    se_type: str = ...,
    bandwidth: float = ...,
) -> dict[str, Any]:
    """Fixed-effects panel OLS; `outcome` is N x T, `regressors` is k x N x T.

    `se_type`: "nonrobust", "cluster" (by entity), or "driscoll_kraay".
    """

def panel_lp(
    outcome: _ArrayLike,
    shock: _ArrayLike,
    horizon: int = ...,
    n_lag_controls: int = ...,
    se_type: str = ...,
    bandwidth: float = ...,
    cumulative: bool = ...,
    jackknife: bool = ...,
) -> dict[str, Any]:
    """Panel local projection of a common shock with fixed effects."""

# --------------------------------------------------- forecast comparison
def cw_test(
    e_small: _ArrayLike,
    e_large: _ArrayLike,
    yhat_small: _ArrayLike,
    yhat_large: _ArrayLike,
    lrv_lags: int = ...,
) -> dict[str, float]:
    """Clark-West test for nested-model equal predictive accuracy."""

def gw_test(loss1: _ArrayLike, loss2: _ArrayLike, lrv_lags: int = ...) -> dict[str, Any]:
    """Giacomini-White unconditional test of equal predictive ability."""

# ------------------------------------------------------ spectral analysis
def periodogram(
    x: _ArrayLike, fs: float = ..., window: str = ..., detrend: str = ...
) -> dict[str, _F64]:
    """Periodogram PSD (freqs, psd); matches scipy.signal.periodogram."""

def welch(
    x: _ArrayLike,
    nperseg: int = ...,
    fs: float = ...,
    noverlap: int | None = ...,
    window: str = ...,
    detrend: str = ...,
) -> dict[str, _F64]:
    """Welch averaged-periodogram PSD; matches scipy.signal.welch."""

def coherence(
    x: _ArrayLike,
    y: _ArrayLike,
    nperseg: int = ...,
    fs: float = ...,
    noverlap: int | None = ...,
    window: str = ...,
    detrend: str = ...,
) -> dict[str, _F64]:
    """Magnitude-squared coherence in [0,1]; matches scipy.signal.coherence."""
