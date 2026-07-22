# API reference

The complete callable surface of `tsecon`, generated from the type stub (`bindings/python/python/tsecon/__init__.pyi`). Array arguments are float64 NumPy arrays (`_ArrayLike = npt.NDArray[np.float64]`; strided views are fine, plain lists and other dtypes are rejected at the boundary). Every function returns plain NumPy arrays and dictionaries — no framework objects. For the *why* and *when* of each method, see the [model cards](README.md) and the [guide](../guide/README.md).

**122 functions.**

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

### `phillips_perron`

```python
def phillips_perron(
    y: _ArrayLike,
    regression: str = ...,
    test_type: str = ...,
    lags: int | None = ...,
) -> dict[str, Any]:
```

Phillips-Perron unit-root test (Z-tau/Z-alpha) with MacKinnon p-values.

### `phillips_ouliaris`

```python
def phillips_ouliaris(
    y: _ArrayLike,
    x: _ArrayLike,
    trend: str = ...,
    test_type: str = ...,
    bandwidth: int | None = ...,
) -> dict[str, Any]:
```

Phillips-Ouliaris residual cointegration test (Zt/Za) with MacKinnon N-surfaces.

### `check_series`

```python
def check_series(
    data: npt.ArrayLike,
    seasonal_period: int | None = ...,
    lags: int | None = ...,
    alpha: float = ...,
    max_breaks: int = ...,
    trim: float = ...,
) -> dict[str, Any]:
```

One-call diagnostic battery with model recommendations (the Module 01 flagship).

    Pure Python over the compiled tests, so plain lists are coerced. 1D input
    runs descriptives, outliers, the ADF+KPSS quadrant, Ljung-Box/ACF/PACF,
    ARCH-LM, Jarque-Bera, a sup-F/Bai-Perron mean-shift scan, GPH long memory,
    and seasonality evidence; 2D (n, k) input runs per-series integration,
    Johansen, and VAR lag selection with a stability check. Evidence is
    reported in families with the multiple-testing arithmetic shown — never
    silently corrected — and the report ends in an ordered `recommendations`
    list routing to concrete tsecon calls. JSON-serializable throughout.
    `lags` is shape-dependent: the Ljung-Box horizon for 1D input (default
    min(10, n//5)), the VAR lag-search cap for 2D input (default 8). `alpha`
    must lie in (0.01, 0.10] — the compiled KPSS p-value is clamped to that
    range. `seasonal_period` must be an integer >= 2 with at least two full
    cycles in sample.

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

    Point path only. For frequentist confidence bands use `var_irf_bands`.

### `var_irf_bands`

```python
def var_irf_bands(
    data: _ArrayLike,
    lags: int = ...,
    horizon: int = ...,
    orth: bool = ...,
    method: str = ...,
    alpha: float = ...,
    cumulative: bool = ...,
    n_boot: int = ...,
    seed: int = ...,
    trend: str = ...,
    bias_correct: bool = ...,
) -> dict[str, Any]:
```

Frequentist confidence bands on VAR impulse responses — the banded companion to `var_irf`.

    Returns a dict with `point`/`se`/`lower`/`upper`, each `[h][response][shock]`
    (same layout as `var_irf`), plus echoed `method`/`alpha`/`n_boot`. `method`:
    "asymptotic" (Lütkepohl 1990 delta-method SEs, Wald bands
    `point ± z_{1-alpha/2}·se`; `n_boot` is `None`) or "bootstrap" (residual
    Efron/Kilian bootstrap, percentile bands, optional Kilian 1998
    `bias_correct`). `orth` and `cumulative` behave exactly as in `var_irf`.

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

### `bvar_hierarchical`

```python
def bvar_hierarchical(
    data: _ArrayLike,
    lags: int = ...,
    delta: float = ...,
    lambda0: float = ...,
    lambda3: float = ...,
    lambda1_init: float = ...,
    lambda1_lo: float = ...,
    lambda1_hi: float = ...,
    optimize: str = ...,
    hyperprior: str = ...,
    n_grid: int = ...,
    max_iter: int = ...,
    tol: float = ...,
) -> dict[str, Any]:
```

Empirical-Bayes Minnesota-BVAR: pick lambda1 by maximizing the marginal likelihood (Giannone-Lenza-Primiceri 2015).

### `bvar_ssvs`

```python
def bvar_ssvs(
    data: _ArrayLike,
    lags: int = ...,
    n_draws: int = ...,
    burn: int = ...,
    seed: int = ...,
    c0: float = ...,
    c1: float = ...,
    prior_inclusion: float = ...,
    ssvs_cov: bool = ...,
    kappa0: float = ...,
    kappa1: float = ...,
    prior_inclusion_cov: float = ...,
    gamma_a: float = ...,
    gamma_b: float = ...,
    horizon: int = ...,
    thin: int = ...,
    n_chains: int = ...,
) -> dict[str, Any]:
```

SSVS-BVAR (George-Sun-Ni 2008): spike-and-slab stochastic-search selection of VAR (and error-precision) restrictions by Gibbs; posterior inclusion probabilities, coef/Sigma means, and orthogonalized IRF draws.

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

### `zero_sign_svar`

```python
def zero_sign_svar(
    data: _ArrayLike,
    sign_restrictions: Sequence[tuple[int, int, int, str]],
    zero_restrictions: Sequence[tuple[int, int, int]],
    lags: int = ...,
    horizon: int = ...,
    n_draws: int = ...,
    max_tries: int = ...,
    seed: int = ...,
    lambda1: float = ...,
    weighted: bool = ...,
) -> dict[str, Any]:
```

Zero + sign restricted Bayesian SVAR: exact zeros by construction + sign rejection.

    `sign_restrictions` are (variable, shock, horizon, sign) tuples with sign in
    {"+", "-"} (may be empty); `zero_restrictions` are (variable, shock, horizon)
    tuples imposing an exact zero on `Theta_h[variable, shock]` (horizon 0 =
    impact). At least one list must be non-empty. Returns per-(horizon, variable,
    shock) `quantiles` at `probs=[0.05,0.16,0.50,0.84,0.95]` (ARW-2018 importance-
    weighted when `weighted=True`), the weight-invariant identified-set envelope
    (`set_min`/`set_max`), per-accepted-draw `weights` (normalized to sum to 1) and
        their effective sample size `ess`, and the acceptance `diagnostics`. With
    strict-upper-triangle impact zeros and no sign restrictions the rotation at
    every draw is pinned to Q=I, so each posterior draw's structural IRF equals
    that draw's recursive Cholesky IRF (a per-draw identity checked to ~1e-10 in
    the crate golden); the posterior of the bands therefore coincides with the
    recursive-Cholesky posterior, and the reported `set_min`/`set_max` span
    reflects posterior (not identified-set) uncertainty since the rotation is
    fixed. The ARW weight is exactly 1 for impact-only zero patterns.

### `structural_fevd`

```python
def structural_fevd(
    data: _ArrayLike,
    lags: int = ...,
    horizon: int = ...,
    trend: str = ...,
    impact: _ArrayLike | None = ...,
    sigma: str = ...,
) -> dict[str, Any]:
```

Structural FEVD for an arbitrary structural impact matrix A0 (the gap
    var_fevd, recursive-Cholesky only, leaves).

    `impact` is an optional (n, n) structural impact A0 (columns = one-SD
    structural shocks, A0 A0' = Sigma; from any identification scheme). If None,
    A0 is the lower Cholesky of the innovation covariance and the result equals
    `var_fevd` exactly. `sigma` ("dfadj"|"mle") sets the default Cholesky's df
    scaling; the FEVD shares are invariant to it (it only rescales the reported
    `impact`). Returns `fevd` [horizon+1][variable][shock] (each row sums to 1)
    and `impact` [n][n] (the A0 used).

### `historical_decomposition`

```python
def historical_decomposition(
    data: _ArrayLike,
    restrictions: Sequence[tuple[int, int, int, str]] = ...,
    lags: int = ...,
    horizon: int | None = ...,
    identification: str = ...,
    n_draws: int = ...,
    max_tries: int = ...,
    seed: int = ...,
    lambda1: float = ...,
    narrative_restrictions: list[dict] | None = ...,
    n_weight_draws: int = ...,
) -> dict[str, Any]:
```

Historical decomposition: per-(time, variable, shock) structural-shock contributions.

    Splits each variable into a deterministic/initial-condition `baseline` plus the
    cumulated contribution `hd[time][variable][shock]` of each structural shock,
    obeying the exact adding-up identity y = baseline + sum_j hd (validated to ~1e-10
    against a NumPy reference). `times` are 0-based effective-sample indices
    (= data_row - lags).

    identification="cholesky" (default): a point decomposition at the OLS VAR with
    Q=I; returns `times`, `baseline` [T_eff][n], `hd` [T_eff][n][n] indexed
    [time][variable][shock], and the structural `shocks` [T_eff][n].
    identification="sign": the importance-weighted SET decomposition over sign- (and
    optionally narrative-) restricted rotations; returns `times`, `baseline`
    (posterior-mean), `probs`, `hd_quantiles` [T_eff][n][n][len(probs)] (weighted
    type-7), the weight-free identified-set envelope `hd_set_min`/`hd_set_max`,
    per-draw `weights`, and `diagnostics`.

    `narrative_restrictions` (sign mode) is a list of dicts with 0-based effective
    indices:
      {"type":"shock_sign","shock":int,"period":int,"sign":"+"|"-"}
      {"type":"contribution","variable":int,"shock":int,"start":int,"end":int,
       "rule":"most"|"least","strong":bool}
      {"type":"contribution_sign","variable":int,"shock":int,"start":int,"end":int,
       "sign":"+"|"-"}

### `narrative_svar`

```python
def narrative_svar(
    data: _ArrayLike,
    sign_restrictions: Sequence[tuple[int, int, int, str]] = ...,
    narrative_restrictions: list[dict] | None = ...,
    lags: int = ...,
    horizon: int = ...,
    n_draws: int = ...,
    max_tries: int = ...,
    seed: int = ...,
    lambda1: float = ...,
    n_weight_draws: int = ...,
) -> dict[str, Any]:
```

Narrative sign-restricted Bayesian SVAR (Antolín-Díaz & Rubio-Ramírez 2018).

    Augments traditional sign restrictions with restrictions on named historical
    episodes — shock signs and "most/least important contributor" statements (see
    `historical_decomposition` for the `narrative_restrictions` dict schema) —
    imposed by importance-reweighting the accepted rotations with weight = 1/P̂(N|S).
    Returns per-(horizon, variable, shock) `quantiles` (weighted type-7) at
    `probs=[0.05,0.16,0.50,0.84,0.95]`, the weight-free identified-set envelope
    `set_min`/`set_max`, per-draw `weights` (mean 1), and `diagnostics` (with `ess`,
    `narrative_acceptance_rate`, `min_ptilde`). With no narrative restrictions every
    weight is 1 and it reproduces `sign_restricted_svar` bit-for-bit.

### `fry_pagan_svar`

```python
def fry_pagan_svar(
    data: _ArrayLike,
    restrictions: Sequence[tuple[int, int, int, str]],
    lags: int = ...,
    horizon: int = ...,
    n_draws: int = ...,
    max_tries: int = ...,
    seed: int = ...,
    lambda1: float = ...,
    target: str = ...,
) -> dict[str, Any]:
```

Fry-Pagan (2011) median-target SVAR: the single coherent draw closest to the median band.

    Sign restrictions set-identify a *set* of structural models; the pointwise
    median band mixes responses from mutually inconsistent draws and is not
    itself a model. This returns instead the single accepted, sign-normalized
    draw whose structural IRFs jointly minimize the Fry-Pagan criterion -- the
    sum, over the target cells, of squared deviations from the pointwise median,
    each standardized by that cell's across-draw dispersion. `restrictions` are
    (variable, shock, horizon, sign) tuples with sign in {"+", "-"}; `target` is
    "restricted" (response cells of the sign-restricted shocks; default) or
    "all". Returns the coherent `median_target_irf` [horizon+1][n][n], the
    incoherent pointwise `median_irf` (for comparison), the selected `mt_index`
    (0-based into the accepted set), its `mt_statistic`, `n_accepted`, and the
    acceptance `diagnostics`. Reproducible at a fixed `seed` (substream
    contract). The selected draw is a descriptive summary -- one interior point
    of the identified set, dependent on the informative Haar prior -- not a
    prior-free point estimate.

### `robust_svar_bounds`

```python
def robust_svar_bounds(
    data: _ArrayLike,
    restrictions: Sequence[tuple[int, int, int, str]],
    lags: int = ...,
    horizon: int = ...,
    n_draws: int = ...,
    seed: int = ...,
    lambda1: float = ...,
    alpha: float = ...,
) -> dict[str, Any]:
```

Giacomini-Kitagawa prior-robust identified-set bounds for a sign-restricted SVAR.

    `restrictions` are (variable, shock, horizon, sign) tuples with sign in
    {"+", "-"}. For each restricted shock, the per-draw identified set of the
    structural IRF is computed exactly over the admissible rotation set and
    summarized over the reduced-form posterior, removing the informative-Haar-
    prior artifact that pointwise `sign_restricted_svar` bands carry. Returns
    per (horizon, variable, shock): `set_lower_mean`/`set_upper_mean` (posterior-
    mean identified-set edges), `robust_ci_lower`/`robust_ci_upper` (the level-
    `alpha` robust credible region), and `lower_quantiles`/`upper_quantiles` at
    `probs=[0.05,0.16,0.50,0.84,0.95]`. Unrestricted shocks are NaN;
    `restricted_shocks` lists the valid shock indices; `diagnostics` reports
    `empty_set_rate` (the share of draws whose restrictions were mutually
    infeasible). Exact for a single restricted shock (Gafarov-Meier-Montiel-Olea
    2018 closed form); with multiple jointly-restricted shocks each bound is that
    shock's marginal identified set — a conservative outer approximation of the
    joint set, since the cross-shock orthogonality coupling is not imposed.

### `long_run_svar`

```python
def long_run_svar(
    data: _ArrayLike,
    lags: int = ...,
    horizon: int = ...,
    trend: str = ...,
    restrictions: Sequence[tuple[int, int]] | None = ...,
    normalize: str = ...,
) -> dict[str, Any]:
```

Blanchard-Quah long-run SVAR: closed-form structural IRFs under frequency-zero restrictions.

    `restrictions` is a list of (variable, shock) long-run zero pairs (None =>
    classic recursive BQ); `normalize` is "long_run" (positive LR diagonal;
    default) or "impact" (positive B diagonal). Returns `impact` (B),
    `long_run` (LR = C(1) B), `long_run_multiplier` (C(1)), `irf`
    [horizon+1][i][j], `cumulative_irf`, and `fevd`. Point estimate, no RNG.

### `max_share_svar`

```python
def max_share_svar(
    data: _ArrayLike,
    lags: int = ...,
    target: int = ...,
    h0: int = ...,
    h1: int = ...,
    horizon: int = ...,
    trend: str = ...,
    exclude_impact: bool = ...,
    weighting: str = ...,
    sign: str = ...,
) -> dict[str, Any]:
```

Max-share / maximum-FEV structural shock (Uhlig 2004; Francis et al 2014; Barsky-Sims 2011 news).

    Identifies the single UNIT-VARIANCE structural shock maximizing the `target`
    variable's forecast-error variance accumulated over the window `[h0, h1]`.
    `weighting="window"` selects the Uhlig/Francis objective (incremental
    windowed FEV; `share_window` is an exact accumulated-FEV fraction),
    `"cumulative"` the Barsky-Sims objective (window-mean cumulative FEV share).
    `exclude_impact=True` imposes zero impact on the target (Barsky-Sims news
    shock). `sign` pins the identified sign ("cumsum"|"impact"|"none").
    Returns `irf` [horizon+1][k], `impact` [k], `q` [k], `share_window` (float),
    `fev_share` [horizon+1], and `eigenvalues` (ascending; length k, or k-1 when
    `exclude_impact`).

### `proxy_svar`

```python
def proxy_svar(
    data: _ArrayLike,
    proxy: _ArrayLike,
    lags: int = ...,
    horizon: int = ...,
    norm_var: int = ...,
    unit: float = ...,
    trend: str = ...,
    robust_f: bool = ...,
) -> dict[str, Any]:
```

Proxy SVAR (external-instrument SVAR-IV): one shock from one instrument.

    The residual-instrument covariance identifies the target shock's impact
    column up to scale; the unit-effect normalization sets its impact on
    `norm_var` to `unit` (sign pinned). `proxy` aligns to `data` rows (NaN
    outside the instrument window is dropped). Returns `irf` (horizon+1, n),
    `impact`, `relative_impact`, `cov_um`, `first_stage_f` (weak below 10),
    `reliability` = Corr(m, u_norm)^2, `n_proxy`, and the estimated `shock`
    (length T). Point estimate only -- no bands (v2: Jentsch-Lunsford MBB).

### `nongaussian_svar`

```python
def nongaussian_svar(
    data: _ArrayLike,
    lags: int = ...,
    horizon: int = ...,
    trend: str = ...,
    contrast: str = ...,
    max_iter: int = ...,
    tol: float = ...,
    order_by: str = ...,
) -> dict[str, Any]:
```

Non-Gaussian / independent-component SVAR identification (Lanne-Meitz-Saikkonen 2017; Gourieroux-Monfort-Renne 2017; FastICA).

    Point-identifies the structural impact matrix B in u_t = B eps_t from the
    reduced-form residuals ALONE -- no sign, zero, long-run, or proxy
    restriction -- by exploiting the statistical INDEPENDENCE and NON-GAUSSIANITY
    of the structural shocks (at most one Gaussian). Whitens by Sigma_u^{-1/2},
    finds the orthogonal rotation maximizing non-Gaussianity via a deterministic
    symmetric FastICA fixed point (log-cosh contrast, identity init -- bit-
    reproducible), then B = Sigma_u^{1/2} Q. Columns are ordered by `order_by`
    ("kurtosis" = descending |excess kurtosis|, or "colnorm") and signed max-abs-
    positive; both are CONVENTIONS, not economics. This is STATISTICAL
    identification: it FAILS if the shocks are Gaussian, and a `shock_kurtosis`
    near zero flags a weakly identified (near-Gaussian) column. Returns `impact`
    (B, [var][shock]), `irf` ([horizon+1][var][shock], Theta_h = Psi_h B),
    `rotation` (Q, [whitened][shock]), `shock_kurtosis` [k] (identified order),
    `converged` (bool), `n_iter` (int), and `order` [k] (raw FastICA index per
    identified position).

### `hetero_svar`

```python
def hetero_svar(
    data: _ArrayLike,
    regime_labels: npt.NDArray[np.integer] | Sequence[int],
    lags: int = ...,
    horizon: int = ...,
    trend: str = ...,
    base_regime: int | None = ...,
    sign_normalization: str = ...,
) -> dict[str, Any]:
```

SVAR identification through heteroskedasticity (Rigobon 2003; Lanne-Lutkepohl 2008), two known variance regimes.

    `data` is (T, n); `regime_labels` is an array-like of length T with EXACTLY
    two distinct integer values (labels align to observations; the first `lags`
    are dropped to match residuals). `base_regime` is the label normalized to
    Lambda=I (default: the smaller label); the other regime's shock-variance
    ratios are reported. `sign_normalization`: "max" (largest-|entry| per B
    column made positive; default) or "diag" (B[j,j] >= 0).

    Returns a dict with `B` (n x n impact matrix = Theta_0, columns in
    ascending variance-ratio order), `variance_ratios` (the n generalized
    eigenvalues, ascending), `structural_irf` ([h][i][j] = Theta_h = Psi_h B),
    `min_ratio_gap` and `ratio_dist_from_unity` (identification margins),
    `identified` (bool heuristic), `covariance_equality` (Bartlett-corrected
    Box's M: statistic/dof/pvalue/distinct_regimes), `sigma_regime1`,
    `sigma_regime2`, `regime1_label`, `regime2_label`, `regime_sizes`,
    `n_vars`, `horizon`, `lags`, `sign_convention`.

    Point-identified IF AND ONLY IF the variance ratios are pairwise distinct
    (min_ratio_gap > 0); the shocks come out ordered by variance ratio and
    carry no economic labels. Standard errors on B/Theta_h are not provided in
    this closed-form build. The >2-regime and Markov-switching/GARCH variants
    are deferred.

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

### `panel_unit_root`

```python
def panel_unit_root(
    data: _ArrayLike | Sequence[_ArrayLike],
    test: str = ...,
    lags: str | int | None = ...,
    regression: str = ...,
    max_lags: int | None = ...,
    lrv_kernel: str = ...,
    lrv_bandwidth: float | None = ...,
) -> dict[str, Any]:
```

First-generation panel unit-root tests (LLC, IPS, Fisher/Maddala-Wu-Choi).

    data is a balanced N x T array (rows = units) or a list of 1-D per-unit
    series (unbalanced OK for "ips"/"fisher"; "llc" needs a common T). test is
    "ips" (default), "llc", or "fisher"; regression is "c"/"ct"/"n" ("n" is
    invalid for "ips"); lags is None (per-unit auto AIC), an int (fixed common
    lag), or "aic"/"bic"/"t-stat". Returns statistic, p_value,
    per_unit_tstat/pvalue/lags/nobs, n_units, regression, plus test-specific
    extras: ips -> t_bar; llc -> delta_hat, t_delta, s_n, t_bar_periods;
    fisher -> maddala_wu, choi_z, choi_z_pvalue.

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

## quantile & growth-at-risk

### `quantile_regression`

```python
def quantile_regression(
    y: _ArrayLike,
    x: _ArrayLike,
    taus: Sequence[float] | None = ...,
    se: str = ...,
) -> dict[str, Any]:
```

Linear quantile regression (statsmodels QuantReg, all defaults).

    IRLS check-loss coefficients with Powell kernel-sandwich standard errors
    (Epanechnikov kernel, Hall-Sheather bandwidth; `se="robust"` is the only
    flavor). Include the constant column in `x`. Returns per-tau `params`,
    `bse`, `tvalues`, `iterations`, `bandwidth`, `sparsity`, plus a single
    `converged` bool over all taus.

### `quantile_lp`

```python
def quantile_lp(
    y: _ArrayLike,
    shock: _ArrayLike,
    taus: Sequence[float] | None = ...,
    horizons: int = ...,
    n_lag_controls: int = ...,
) -> dict[str, Any]:
```

Quantile local projections: `irf[tau][h]` with Powell-sandwich `se[tau][h]`.

    Per horizon, `y_{t+h}` on `[shock_t, const, p lags of y and shock]` at
    each tau (tsecon-lp design conventions); matches statsmodels QuantReg on
    the identical design.

### `growth_at_risk`

```python
def growth_at_risk(
    y: _ArrayLike,
    conditions: _ArrayLike,
    horizon: int = ...,
    taus: Sequence[float] | None = ...,
    rearrange: bool = ...,
) -> dict[str, Any]:
```

Growth-at-risk (Adrian-Boyarchenko-Giannone 2019).

    Conditional quantiles of the h-ahead outcome on `[const, conditions,
    y_t]`, evaluated at every t — `current` is the latest risk read. `taus`
    must be strictly increasing and `horizon >= 1`. `rearrange` applies the
    Chernozhukov-Fernandez-Val-Galichon monotone sort across tau; `crossing`
    reports whether the raw fitted quantile paths crossed either way.

## functional shocks (FVAR / FLP)

### `functional_pca`

```python
def functional_pca(curves: _ArrayLike, n_factors: int = ...) -> dict[str, Any]:
```

Functional PCA of a T x M curve panel (Inoue-Rossi 2021).

    Returns mean_curve, eigenfunctions (K x M), scores (T x K), eigenvalues,
    explained, total_variance. Sign: each eigenfunction's largest-|.| entry
    is positive.

### `flp`

```python
def flp(
    y: _ArrayLike,
    scores: _ArrayLike,
    horizons: int = ...,
    n_lag_controls: int = ...,
    hac_maxlags: int | None = ...,
) -> dict[str, Any]:
```

Functional local projection: y_{t+h} on ALL K scores jointly + const +
    lags of y, Newey-West HAC (maxlags = h + n_lag_controls default).

    Returns horizons, n_factors, betas ((H+1) x K), covs (joint (H+1) x K x K),
    se, nobs.

### `flp_scenario`

```python
def flp_scenario(
    y: _ArrayLike,
    curves: _ArrayLike,
    delta: _ArrayLike,
    n_factors: int = ...,
    horizons: int = ...,
    n_lag_controls: int = ...,
    hac_maxlags: int | None = ...,
) -> dict[str, Any]:
```

IRF of y to a whole-curve scenario delta (length M): FPCA, joint FLP,
    then response w'beta_h with se sqrt(w' Cov_h w).

    Returns horizons, weights, response, se, betas, explained.

### `fvar_scenario`

```python
def fvar_scenario(
    y: _ArrayLike,
    curves: _ArrayLike,
    delta: _ArrayLike,
    n_factors: int = ...,
    lags: int = ...,
    horizon: int = ...,
) -> dict[str, Any]:
```

FVAR scenario: VAR([scores, y], scores FIRST) with Cholesky
    identification; score innovation set to w = phi'delta, outcome's own
    structural shock zero (impact response of y is a modeling assumption).

    Returns horizons, weights, response_outcome, responses ((H+1) x (K+1),
    scores first then outcome), implied_outcome_innovation.

## structural breaks

### `bai_perron`

```python
def bai_perron(
    y: _ArrayLike, x: _ArrayLike, max_breaks: int = ..., trim: float = ...
) -> dict[str, Any]:
```

Bai-Perron multiple breaks: DP global partitions, sequential supF(l+1|l) selection at 5%, per-regime OLS, and Bai (1997) break-date confidence intervals; x is T x q with all coefficients switching (include your constant).

### `sup_f_test`

```python
def sup_f_test(y: _ArrayLike, x: _ArrayLike, trim: float = ...) -> dict[str, Any]:
```

Andrews sup-F (Quandt) unknown-break test with Hansen (1997) approximate p-value; returns stat, p_value, break_date, and the full f_path over the trimmed dates.

## smooth local projections

### `smooth_lp`

```python
def smooth_lp(
    y: _ArrayLike,
    shock: _ArrayLike,
    horizons: int = ...,
    n_lag_controls: int = ...,
    lam: float | str | None = ...,
    degree: int = ...,
    n_basis: int | None = ...,
    penalty_order: int = ...,
    lambda_grid: Sequence[float] | None = ...,
    n_folds: int = ...,
    hac_maxlags: int | None = ...,
) -> dict[str, Any]:
```

Smooth local projections (Barnichon-Brownlees 2019): the IRF as a
    penalized B-spline in the horizon, estimated jointly across horizons.

    `lam`: a float fixes the smoothing parameter (0.0 reproduces the
    per-horizon `lp(se="hac")` point estimates with the default basis);
    "cv"/None cross-validates it by leave-h-block-out CV over `lambda_grid`
    (or a default log-spaced grid). `penalty_order=2` shrinks the IRF toward
    a straight line as `lam` grows. `se` conditions on `lam` and does not
    account for shrinkage bias; `irf_raw`/`se_raw` are the unsmoothed
    per-horizon HAC LP for comparison. Keys: horizons, irf, se, lambda_used,
    cv_grid, cv_scores, theta, irf_raw, se_raw.

