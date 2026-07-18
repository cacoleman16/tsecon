# API reference

The complete callable surface of `tsecon`, generated from the type stub (`bindings/python/python/tsecon/__init__.pyi`). Every function returns plain NumPy arrays and dictionaries — no framework objects. For the *why* and *when* of each method, see the [model cards](README.md) and the [guide](../guide/README.md).

**94 functions.**

## diagnostics

### `acf`

```python
def acf(y: _ArrayLike, nlags: int = ..., adjusted: bool = ...) -> dict[str, _F64]:
```

Autocorrelation function with Bartlett standard errors.

### `pacf`

```python
def pacf(y: _ArrayLike, nlags: int = ..., method: str = ...) -> _F64:
```

Partial autocorrelation function; `method` is "yw" or "ols".

### `ljung_box`

```python
def ljung_box(y: _ArrayLike, nlags: int = ...) -> dict[str, _F64]:
```

Ljung-Box and Box-Pierce portmanteau tests for lags 1..=nlags.

### `jarque_bera`

```python
def jarque_bera(x: _ArrayLike) -> dict[str, float]:
```

Jarque-Bera normality test (statistic, p_value, skewness, kurtosis, n).

### `arch_lm`

```python
def arch_lm(resid: _ArrayLike, nlags: int = ...) -> dict[str, float]:
```

Engle's ARCH-LM test for conditional heteroskedasticity.

## unit roots / workflow

### `adf`

```python
def adf(
    y: _ArrayLike,
    regression: str = ...,
    autolag: str | None = ...,
    maxlag: int | None = ...,
) -> dict[str, Any]:
```

Augmented Dickey-Fuller test with MacKinnon p-values.

### `kpss`

```python
def kpss(
    y: _ArrayLike, regression: str = ..., nlags: str | int | None = ...
) -> dict[str, Any]:
```

KPSS stationarity test (null: stationary).

### `check_stationarity`

```python
def check_stationarity(y: _ArrayLike, alpha: float = ...) -> dict[str, Any]:
```

The ADF + KPSS confirmatory-quadrant workflow with a recommendation.

## robust inference

### `long_run_variance`

```python
def long_run_variance(
    x: _ArrayLike, kernel: str = ..., bandwidth: float | None = ...
) -> float:
```

Kernel long-run variance of a series (demeaned internally).

### `ols`

```python
def ols(
    y: _ArrayLike,
    x: _ArrayLike,
    se_type: str = ...,
    maxlags: int | None = ...,
    use_correction: bool = ...,
) -> dict[str, Any]:
```

OLS with nonrobust / HC0 / HC1 / HAC standard errors.

## bootstrap

### `bootstrap_indices`

```python
def bootstrap_indices(
    n: int,
    scheme: str = ...,
    seed: int = ...,
    block_length: int | None = ...,
    p: float | None = ...,
) -> npt.NDArray[np.uint64]:
```

Bootstrap resampling indices (iid/moving/circular/stationary).

### `optimal_block_length`

```python
def optimal_block_length(y: _ArrayLike) -> dict[str, float]:
```

Politis-White (2004) automatic block length (stationary, circular).

### `philox_uniforms`

```python
def philox_uniforms(seed: int, n: int) -> _F64:
```

Uniform draws from the Philox stream; bit-identical to NumPy.

## state space

### `local_level_smooth`

```python
def local_level_smooth(
    y: _ArrayLike, sigma2_eps: float, sigma2_eta: float
) -> dict[str, Any]:
```

Exact-diffuse local-level Kalman filter + smoother (NaN = missing).

### `ar_loglik`

```python
def ar_loglik(
    y: _ArrayLike, coeffs: Sequence[float], sigma2: float, intercept: float = ...
) -> float:
```

Exact Gaussian log-likelihood of an AR(p) at fixed parameters.

## ARIMA

### `arima_fit`

```python
def arima_fit(
    y: _ArrayLike,
    p: int = ...,
    d: int = ...,
    q: int = ...,
    constant: bool = ...,
    forecast_steps: int = ...,
    conf_alpha: float | None = ...,
) -> dict[str, Any]:
```

Exact-MLE ARIMA(p,d,q) fit, with optional forecast + conf_alpha bands.

## GARCH

### `garch_fit`

```python
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
```

GARCH/GJR/EGARCH QMLE with MLE and Bollerslev-Wooldridge robust SEs.

## VAR

### `var_fit`

```python
def var_fit(data: _ArrayLike, lags: int = ..., trend: str = ...) -> dict[str, Any]:
```

Fit a VAR(p) by OLS; params, sigma_u, ICs, and stability.

    Read `is_stable` for the stability verdict. `min_root`/`max_root` are the
    smallest/largest moduli of the reciprocal characteristic roots — stable iff
    `min_root > 1`, so `max_root` alone is not a verdict.

### `var_irf`

```python
def var_irf(
    data: _ArrayLike,
    lags: int = ...,
    horizon: int = ...,
    orth: bool = ...,
    trend: str = ...,
    cumulative: bool = ...,
) -> list[list[list[float]]]:
```

Impulse responses [h][response][shock]; `cumulative` gives running sums.

### `var_fevd`

```python
def var_fevd(
    data: _ArrayLike, lags: int = ..., horizon: int = ..., trend: str = ...
) -> list[list[list[float]]]:
```

Forecast-error variance decomposition [h][variable][shock].

### `var_forecast`

```python
def var_forecast(
    data: _ArrayLike,
    lags: int = ...,
    steps: int = ...,
    alpha: float = ...,
    trend: str = ...,
) -> dict[str, Any]:
```

Iterated VAR point forecasts with (1-alpha) intervals.

### `var_granger`

```python
def var_granger(
    data: _ArrayLike,
    caused: Sequence[int],
    causing: Sequence[int],
    lags: int = ...,
    trend: str = ...,
) -> dict[str, Any]:
```

Granger-causality F test (matches statsmodels test_causality).

## Bayesian VAR

### `bvar_fit`

```python
def bvar_fit(
    data: _ArrayLike,
    lags: int = ...,
    lambda0: float = ...,
    lambda1: float = ...,
    lambda3: float = ...,
    delta: float = ...,
) -> dict[str, Any]:
```

Minnesota-NIW conjugate BVAR posterior + log marginal likelihood.

### `bvar_irf_draws`

```python
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
```

Posterior Cholesky-IRF draws [draw][h][variable][shock] for credible bands.

### `mcmc_diagnostics`

```python
def mcmc_diagnostics(chains: _ArrayLike) -> dict[str, float]:
```

Rank-normalized split R-hat and bulk/tail ESS (ArviZ-exact).

## filters

### `hp_filter`

```python
def hp_filter(y: _ArrayLike, lamb: float = ..., one_sided: bool = ...) -> dict[str, Any]:
```

Hodrick-Prescott filter (O(n)); `one_sided=True` for the real-time variant.

### `bk_filter`

```python
def bk_filter(
    y: _ArrayLike, low: float = ..., high: float = ..., k: int = ...
) -> dict[str, Any]:
```

Baxter-King band-pass filter (loses k observations at each end).

### `cf_filter`

```python
def cf_filter(
    y: _ArrayLike, low: float = ..., high: float = ..., drift: bool = ...
) -> dict[str, Any]:
```

Christiano-Fitzgerald asymmetric band-pass filter.

### `hamilton_filter`

```python
def hamilton_filter(y: _ArrayLike, h: int = ..., p: int = ...) -> dict[str, Any]:
```

Hamilton (2018) regression filter — the modern HP alternative.

## forecasting / evaluation

### `dm_test`

```python
def dm_test(
    e1: _ArrayLike, e2: _ArrayLike, h: int = ..., loss: str = ...
) -> dict[str, float]:
```

Diebold-Mariano test with the Harvey-Leybourne-Newbold correction.

### `accuracy`

```python
def accuracy(
    actual: _ArrayLike,
    forecast: _ArrayLike,
    insample: _ArrayLike | None = ...,
    period: int = ...,
) -> dict[str, float]:
```

Forecast accuracy measures (ME/RMSE/MAE/MAPE/sMAPE/MASE/RMSSE).

### `theta_forecast`

```python
def theta_forecast(y: _ArrayLike, steps: int, period: int = ...) -> _F64:
```

The Theta method (Assimakopoulos-Nikolopoulos 2000).

## local projections

### `lp`

```python
def lp(
    y: _ArrayLike,
    shock: _ArrayLike,
    horizons: int = ...,
    n_lag_controls: int = ...,
    se: str = ...,
    maxlags: int | None = ...,
    cumulative: bool | str | None = ...,
) -> dict[str, Any]:
```

Local projection IRFs; `se` is "lag_augmented" (default) or "hac".

    `cumulative`: False/"none" (level), True/"outcome" (cumulated outcome on
    the contemporaneous impulse — a cumulative IRF, NOT a multiplier), or
    "both" (cumulated outcome on cumulated impulse). For an identified
    multiplier use `lp_multiplier`.

### `lp_iv`

```python
def lp_iv(
    y: _ArrayLike,
    impulse: _ArrayLike,
    instrument: _ArrayLike,
    horizons: int = ...,
    n_lag_controls: int = ...,
    cumulative: bool | str | None = ...,
) -> dict[str, Any]:
```

LP-IV: instrumented local projections with a first-stage F diagnostic.

    `cumulative` takes False/"none", True/"outcome" or "both". True/"outcome"
    cumulates only the OUTCOME, giving cumulated response per unit of
    *contemporaneous* impulse — that grows without bound in the horizon and is
    not a multiplier. Use `lp_multiplier` for the Ramey-Zubairy integral
    multiplier.

### `lp_multiplier`

```python
def lp_multiplier(
    y: _ArrayLike,
    impulse: _ArrayLike,
    instrument: _ArrayLike,
    horizons: int = ...,
    n_lag_controls: int = ...,
    maxlags: int | None = ...,
) -> dict[str, Any]:
```

Ramey-Zubairy (2018) integral multiplier by one-step LP-IV.

    Regresses the cumulated outcome on the cumulated impulse, instrumented by
    the contemporaneous instrument, controlling for lags of both series. Both
    sides accumulate over the same window, so the coefficient is a multiplier
    rather than a cumulative impulse response. `se` is the kernel-HAC standard
    error of that single 2SLS coefficient — inference on the multiplier
    itself, not a delta-method ratio and not a leg's SE relabelled.

## penalized regression

### `ridge`

```python
def ridge(x: _ArrayLike, y: _ArrayLike, alpha: float) -> _F64:
```

Ridge regression (closed form); scikit-learn `Ridge` objective.

### `elastic_net`

```python
def elastic_net(
    x: _ArrayLike,
    y: _ArrayLike,
    alpha: float,
    l1_ratio: float = ...,
    tol: float = ...,
    max_iter: int = ...,
) -> dict[str, Any]:
```

Elastic-net via coordinate descent; scikit-learn objective.

### `lasso`

```python
def lasso(
    x: _ArrayLike,
    y: _ArrayLike,
    alpha: float,
    tol: float = ...,
    max_iter: int = ...,
) -> dict[str, Any]:
```

Lasso (elastic net with l1_ratio = 1.0).

## structural identification

### `sign_restricted_svar`

```python
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
```

Sign-restricted Bayesian SVAR: identified-set bands + acceptance diagnostics.

    `restrictions` are (variable, shock, horizon, sign) tuples with sign in
    {"+", "-"}. Returns per-(horizon, variable, shock) `quantiles` at
    `probs=[0.05,0.16,0.50,0.84,0.95]`, the identified-set envelope
    (`set_min`/`set_max`), and `diagnostics`.

## panel

### `panel_fe`

```python
def panel_fe(
    outcome: _ArrayLike,
    regressors: _ArrayLike,
    se_type: str = ...,
    bandwidth: float = ...,
) -> dict[str, Any]:
```

Fixed-effects panel OLS; `outcome` is N x T, `regressors` is k x N x T.

    `se_type`: "nonrobust", "cluster" (by entity), or "driscoll_kraay".

### `panel_lp`

```python
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
```

Panel local projection of a common shock with fixed effects.

## forecast comparison

### `cw_test`

```python
def cw_test(
    e_small: _ArrayLike,
    e_large: _ArrayLike,
    yhat_small: _ArrayLike,
    yhat_large: _ArrayLike,
    lrv_lags: int = ...,
) -> dict[str, float]:
```

Clark-West test for nested-model equal predictive accuracy.

### `gw_test`

```python
def gw_test(loss1: _ArrayLike, loss2: _ArrayLike, lrv_lags: int = ...) -> dict[str, Any]:
```

Giacomini-White unconditional test of equal predictive ability.

## spectral analysis

### `periodogram`

```python
def periodogram(
    x: _ArrayLike, fs: float = ..., window: str = ..., detrend: str = ...
) -> dict[str, _F64]:
```

Periodogram PSD (freqs, psd); matches scipy.signal.periodogram.

### `welch`

```python
def welch(
    x: _ArrayLike,
    nperseg: int = ...,
    fs: float = ...,
    noverlap: int | None = ...,
    window: str = ...,
    detrend: str = ...,
) -> dict[str, _F64]:
```

Welch averaged-periodogram PSD; matches scipy.signal.welch.

### `coherence`

```python
def coherence(
    x: _ArrayLike,
    y: _ArrayLike,
    nperseg: int = ...,
    fs: float = ...,
    noverlap: int | None = ...,
    window: str = ...,
    detrend: str = ...,
) -> dict[str, _F64]:
```

Magnitude-squared coherence in [0,1]; matches scipy.signal.coherence.

## cointegration

### `johansen`

```python
def johansen(data: _ArrayLike, k_ar_diff: int = ...) -> dict[str, Any]:
```

Johansen cointegration test (data is T x k); trace + max-eig + rank.

### `vecm`

```python
def vecm(data: _ArrayLike, k_ar_diff: int = ..., coint_rank: int = ...) -> dict[str, Any]:
```

VECM ML estimation: alpha, beta, gamma, sigma_u, llf (statsmodels-exact).

## regime switching

### `markov_switching_ar`

```python
def markov_switching_ar(
    y: _ArrayLike,
    k_regimes: int = ...,
    order: int = ...,
    switching_variance: bool = ...,
    max_iter: int = ...,
    tol: float = ...,
) -> dict[str, Any]:
```

Markov-switching AR fitted by EM (Hamilton 1989); regimes + durations.

## MIDAS

### `midas_weights`

```python
def midas_weights(scheme: str, theta1: float, theta2: float, k: int) -> _F64:
```

MIDAS weights (sum to 1); scheme "exp_almon" or "beta".

### `umidas`

```python
def umidas(
    y: _ArrayLike, hf_lags: _ArrayLike, se_type: str = ..., maxlags: int | None = ...
) -> dict[str, Any]:
```

U-MIDAS: unrestricted mixed-frequency regression (hf_lags is nobs x K).

## multivariate GARCH

### `ccc_garch`

```python
def ccc_garch(returns: _ArrayLike) -> dict[str, Any]:
```

CCC-GARCH (Bollerslev 1990); returns is T x k. Correlation + loglik.

### `dcc_garch`

```python
def dcc_garch(returns: _ArrayLike) -> dict[str, Any]:
```

DCC-GARCH (Engle 2002); a, b, qbar, loglik, last correlation matrix.

## realized volatility / HAR

### `realized_measures`

```python
def realized_measures(returns: _ArrayLike) -> dict[str, float]:
```

Realized variance, bipower variation, and jump component (BNS 2004).

### `har_rv`

```python
def har_rv(
    rv: _ArrayLike,
    start: int = ...,
    variant: str = ...,
    hac_maxlags: int = ...,
    use_correction: bool = ...,
) -> dict[str, Any]:
```

HAR-RV (Corsi 2009): RV_t on [const, daily, weekly, monthly], HAC SEs.

    variant is "level", "log", or "sqrt".

## connectedness

### `connectedness`

```python
def connectedness(
    data: _ArrayLike, lags: int = ..., horizon: int = ..., trend: str = ...
) -> dict[str, Any]:
```

Diebold-Yilmaz connectedness (percent) from a VAR's GFEVD.

    total, to_others, from_others, net, gfevd, pairwise_net (data is T x k).

## factor model

### `factor_model`

```python
def factor_model(
    data: _ArrayLike, n_factors: int = ..., kmax: int = ...
) -> dict[str, Any]:
```

PCA factor model (T x N) + Bai-Ng (2002) factor selection.

    factors, loadings, eigenvalues, icp1/icp2/pcp1/pcp2 and Ahn-Horenstein
    eigenvalue-ratio (er) factor counts.

## term structure

### `nelson_siegel`

```python
def nelson_siegel(
    maturities: _ArrayLike,
    yields: _ArrayLike,
    decay: float = ...,
    optimal_lambda: bool = ...,
) -> dict[str, Any]:
```

Nelson-Siegel yield-curve fit (Diebold-Li 2006).

    level/slope/curvature factors, lambda, residuals, rsquared.
    optimal_lambda=True estimates the decay by NLS.

### `svensson`

```python
def svensson(
    maturities: _ArrayLike, yields: _ArrayLike, lambda1: float, lambda2: float
) -> dict[str, Any]:
```

Svensson (1994) four-factor yield-curve fit; nests Nelson-Siegel.

## GMM / IV-GMM

### `iv_gmm`

```python
def iv_gmm(
    x: _ArrayLike,
    z: _ArrayLike,
    y: _ArrayLike,
    method: str = ...,
    weight: str = ...,
    bandwidth: float = ...,
    tol: float = ...,
    max_iter: int = ...,
) -> dict[str, Any]:
```

Linear IV-GMM (Hansen 1982) with robust or HAC weighting.

    method is "2sls", "2step", or "iterated"; weight is "robust" or "hac".
    Z must include the exogenous regressor columns. Returns params, bse,
    residuals, and (over-identified) the Hansen j_stat/j_dof/j_pval.

## leakage-safe time-series CV

### `cv_splits`

```python
def cv_splits(
    n: int,
    scheme: str = ...,
    train: int = ...,
    horizon: int = ...,
    step: int = ...,
    k: int = ...,
    purge: int = ...,
    embargo: int = ...,
) -> list[dict[str, list[int]]]:
```

Leakage-safe CV split indices for sequential data.

    scheme is "expanding", "rolling", or "purged_kfold". Returns a list of
    {"train": [...], "test": [...]} index dicts.

## penalized ML (paths)

### `adaptive_lasso`

```python
def adaptive_lasso(
    x: _ArrayLike,
    y: _ArrayLike,
    alpha: float,
    l1_ratio: float = ...,
    gamma: float = ...,
    tol: float = ...,
    max_iter: int = ...,
) -> dict[str, Any]:
```

Adaptive LASSO (Zou 2006): oracle-property weighted-L1 penalty.

    coef, n_iter, max_change.

### `lasso_path`

```python
def lasso_path(
    x: _ArrayLike,
    y: _ArrayLike,
    l1_ratio: float = ...,
    n_lambdas: int = ...,
    eps: float = ...,
    tol: float = ...,
    max_iter: int = ...,
) -> dict[str, Any]:
```

Elastic-net regularization path with AIC/BIC selection.

    lambdas, coefs, rss, df, aic, bic, aic_best, bic_best.

## forecast backtest

### `backtest`

```python
def backtest(
    y: _ArrayLike,
    window: str = ...,
    train: int = ...,
    horizon: int = ...,
    refit_every: int = ...,
    forecaster: str = ...,
    period: int = ...,
    insample_period: int = ...,
) -> dict[str, Any]:
```

Rolling/expanding pseudo-out-of-sample backtest.

    window is "expanding" or "rolling"; forecaster is one of naive, drift,
    mean, seasonal_naive, theta. Returns origins, per-horizon forecasts and
    targets, and a per-horizon accuracy table.

## nonlinear GMM (callback)

### `gmm_nonlinear`

```python
def gmm_nonlinear(
    moments_fn: Callable[[_F64], _ArrayLike],
    initial: _ArrayLike,
    weight: _ArrayLike | None = ...,
) -> dict[str, Any]:
```

Nonlinear GMM (Hansen 1982) via Nelder-Mead over a Python moment function.

    moments_fn maps a parameter vector (a 1-D float64 array) to an n-by-m matrix
    of per-observation moment contributions (rows = observations, cols = moments),
    returned as a NumPy array or list of lists. weight is the flattened m*m
    weighting matrix (row-major) or None for the identity. Returns params,
    objective, gbar, converged, iterations, fevals, nmoments, nparams.

## weighted MIDAS

### `weighted_midas`

```python
def weighted_midas(
    y: _ArrayLike,
    hf_lags: _ArrayLike,
    scheme: str = ...,
    weight_start: tuple[float, float] | None = ...,
) -> dict[str, Any]:
```

Weighted MIDAS by NLS (Ghysels et al. 2007); exp_almon/beta weights, hf_lags is nobs x K.

## state-dependent LP

### `lp_state`

```python
def lp_state(
    y: _ArrayLike,
    shock: _ArrayLike,
    state_indicator: _ArrayLike,
    horizons: int = ...,
    n_lag_controls: int = ...,
    se: str = ...,
    maxlags: int | None = ...,
    cumulative: bool | str | None = ...,
) -> dict[str, Any]:
```

State-dependent (interacted) local projections (Ramey-Zubairy 2018); per-regime IRFs and SEs.

    `cumulative` takes False/"none", True/"outcome" or "both", as in `lp`.

## mean-group panel VAR

### `mean_group_var`

```python
def mean_group_var(
    entities: Sequence[_ArrayLike],
    lags: int = ...,
    trend: str = ...,
    horizon: int = ...,
    response: int = ...,
    impulse: int = ...,
) -> dict[str, Any]:
```

Pesaran-Smith (1995) mean-group panel VAR over per-entity T_i x k matrices.

## dynamic Nelson-Siegel

### `dynamic_ns`

```python
def dynamic_ns(
    panel: _ArrayLike, maturities: _ArrayLike, decay: float = ...
) -> dict[str, Any]:
```

Dynamic Nelson-Siegel factors + one-step forecast (Diebold-Li 2006).

    panel is T x n_maturities. Returns maturities, lambda, factors (T x 3),
    rsquared, level/slope/curvature series, and a forecast dict.

## FAVAR

### `favar`

```python
def favar(
    panel: _ArrayLike,
    policy: _ArrayLike,
    n_factors: int = ...,
    lags: int = ...,
    trend: str = ...,
    slow_indices: list[int] | None = ...,
    horizon: int = ...,
    orth: bool = ...,
) -> dict[str, Any]:
```

Two-step factor-augmented VAR (Bernanke-Boivin-Eliasz 2005).

    factors (T x r), params, sigma_u, n_factors, n_endog, policy_index, and
    the recursive policy-shock IRFs irf_panel (N x horizon+1) / irf_policy.

## realized-volatility extras

### `realized_quarticity`

```python
def realized_quarticity(returns: _ArrayLike) -> float:
```

Realized quarticity RQ = (n/3) sum r^4 (BNS 2002).

### `tripower_quarticity`

```python
def tripower_quarticity(returns: _ArrayLike) -> float:
```

Jump-robust tripower quarticity of integrated quarticity (BNS 2004).

### `bns_jump_test`

```python
def bns_jump_test(returns: _ArrayLike) -> dict[str, float]:
```

BNS ratio jump test (BNS 2004; Huang & Tauchen 2005); dict with 'ratio'.

### `realized_range`

```python
def realized_range(
    high: _ArrayLike,
    low: _ArrayLike,
    method: str = ...,
    open: _ArrayLike | None = ...,
    close: _ArrayLike | None = ...,
) -> float:
```

Range variance from OHLC bars; method is "parkinson" or "garman_klass".

## score-driven volatility

### `gas_volatility`

```python
def gas_volatility(
    y: _ArrayLike, density: str = ..., horizon: int = ...
) -> dict[str, Any]:
```

GAS(1,1) score-driven volatility (Creal-Koopman-Lucas 2013).

    density is "gaussian" or "student_t". Returns omega/a/b (+ nu),
    variance, std_resid, loglik, aic, bic, next_variance, and (horizon>0) a
    forecast.

## heterogeneous panel (MG)

### `panel_mean_group`

```python
def panel_mean_group(
    ys: Sequence[_ArrayLike], xs: Sequence[_ArrayLike], method: str = ...
) -> dict[str, Any]:
```

Mean-group (Pesaran-Smith 1995) / CCE-MG (Pesaran 2006) panel estimator.

    method is "mg" or "cce". ys/xs are per-unit response vectors and T_i x k
    regressor matrices. Returns coef, se, tstat, coef_per_unit, n_units, k.

### `panel_pmg`

```python
def panel_pmg(
    ys: Sequence[_ArrayLike], xs: Sequence[_ArrayLike]
) -> dict[str, Any]:
```

Pooled Mean Group ARDL(1,1) panel estimator (Pesaran-Shin-Smith 1999).

    Pools the long-run coefficient across units by ML; error-correction speed
    and short-run dynamics stay unit-specific. Returns theta, theta_se,
    phi_bar, phi, sigma2, loglik, iterations, n_units, k.

## DFM nowcasting

### `dfm_nowcast`

```python
def dfm_nowcast(
    data: _ArrayLike,
    n_factors: int = ...,
    factor_order: int = ...,
    method: str = ...,
) -> dict[str, Any]:
```

Dynamic-factor-model nowcast; data is T x N with an optional NaN edge.

    method is "two_step" (Doz-Giannone-Reichlin 2011) or "mle" (exact
    one-step Gaussian MLE, single factor). Returns nowcast, edge_factor,
    loglik, fit_loglik, smoothed_factors, n_factors, factor_order.

### `dfm_news`

```python
def dfm_news(
    old_vintage: _ArrayLike,
    new_vintage: _ArrayLike,
    target_series: int = ...,
    target_period: int | None = ...,
    n_factors: int = ...,
    factor_order: int = ...,
) -> dict[str, Any]:
```

News/update decomposition of a DFM nowcast revision (Banbura-Modugno 2014).

    Splits the target-series nowcast revision between two data vintages into
    per-datapoint contributions (weight*news). Returns old_nowcast,
    new_nowcast, total_revision, and contributions (a list of dicts).

## predictive regressions / IVX

### `predictive_regression`

```python
def predictive_regression(
    r: _ArrayLike, x: _ArrayLike, cz: float = ..., alpha: float = ...
) -> dict[str, Any]:
```

Predictive regression with a persistent regressor.

    Returns ols, stambaugh (bias-corrected), and ivx (Kostakis-Magdalinos-
    Stamatogiannis 2015, Wald test valid uniformly over persistence).

### `ivx_test`

```python
def ivx_test(
    r: _ArrayLike, xs: _ArrayLike, cz: float = ..., alpha: float = ...
) -> dict[str, Any]:
```

Joint IVX predictability test for several persistent predictors (xs is T x k).

    Returns beta_ivx, the joint wald/pvalue, rz, nregressors, nobs.

## recession probability

### `recession_probit`

```python
def recession_probit(
    y: _ArrayLike, x: _ArrayLike, link: str = ..., dynamic: bool = ...
) -> dict[str, Any]:
```

Probit/logit of a binary recession indicator (Kauppi-Saikkonen dynamic option).

    link is "probit" or "logit". Returns params, bse, zstats, probabilities,
    loglik, pseudo_r2, converged (and rho for dynamic=True).

## survey expectations

### `cg_regression`

```python
def cg_regression(
    errors: _ArrayLike,
    revisions: _ArrayLike,
    maxlags: int | None = ...,
    use_correction: bool = ...,
) -> dict[str, Any]:
```

Coibion-Gorodnichenko (2015) information-rigidity regression (OLS-HAC).

    Returns intercept/slope with HAC se/t/p, r_squared, implied_rigidity.

### `forecast_efficiency`

```python
def forecast_efficiency(
    errors: _ArrayLike,
    regressors: _ArrayLike,
    maxlags: int | None = ...,
    use_correction: bool = ...,
) -> dict[str, Any]:
```

Mincer-Zarnowitz forecast-efficiency Wald test (OLS-HAC); regressors is T x k.

### `forecast_disagreement`

```python
def forecast_disagreement(
    panel: Sequence[_ArrayLike], ddof: int = ...
) -> dict[str, Any]:
```

Forecast-disagreement measures (per-period std/quartiles/iqr) from a forecaster panel.

## long memory

### `frac_diff`

```python
def frac_diff(x: _ArrayLike, d: float) -> _F64:
```

Fractional differencing (1-L)^d via the binomial expansion.

### `frac_integrate`

```python
def frac_integrate(x: _ArrayLike, d: float) -> _F64:
```

Fractional integration (1-L)^-d, the inverse of frac_diff.

### `long_memory_d`

```python
def long_memory_d(
    x: _ArrayLike, m: int | None = ..., method: str = ...
) -> dict[str, float]:
```

Estimate the memory parameter d; method is "gph" or "local_whittle". Returns d, se, m.

## specification tests

### `heteroskedasticity_test`

```python
def heteroskedasticity_test(
    y: _ArrayLike, x: _ArrayLike, test: str = ...
) -> dict[str, Any]:
```

Heteroskedasticity test (test="white" or "breusch_pagan"); x is T x k with a constant.

### `reset_test`

```python
def reset_test(y: _ArrayLike, x: _ArrayLike, max_power: int = ...) -> dict[str, Any]:
```

Ramsey RESET functional-form F-test; x is T x k.

### `chow_test`

```python
def chow_test(y: _ArrayLike, x: _ArrayLike, split: int) -> dict[str, Any]:
```

Chow structural-break F-test at a known 0-indexed split; x is T x k.

### `cusum_test`

```python
def cusum_test(y: _ArrayLike, x: _ArrayLike) -> dict[str, Any]:
```

CUSUM parameter-stability test (Brown-Durbin-Evans); returns the path and 5% bounds.

## arbitrage-free NS

### `afns_adjustment`

```python
def afns_adjustment(
    maturities: _ArrayLike, sigma: _ArrayLike, decay: float = ...
) -> _F64:
```

Arbitrage-free Nelson-Siegel yield adjustment (Christensen-Diebold-Rudebusch 2011); sigma has 3 elements.

## DSGE-lite

### `dsge_solve`

```python
def dsge_solve(
    a: _ArrayLike, b: _ArrayLike, c: _ArrayLike, n_predetermined: int
) -> dict[str, Any]:
```

Blanchard-Kahn solution of a linear RE model A E[y_{t+1}] = B y_t + C z.

    Returns the decision rule g, the law of motion p/q, eigenvalue_moduli, and verdict.

