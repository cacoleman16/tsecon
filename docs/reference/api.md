# API reference

The complete callable surface of `tsecon`, generated from the type stub (`bindings/python/python/tsecon/__init__.pyi`). Array arguments are float64 NumPy arrays (`_ArrayLike = npt.NDArray[np.float64]`; strided views are fine, plain lists and other dtypes are rejected at the boundary). Every function returns plain NumPy arrays and dictionaries — no framework objects. For the *why* and *when* of each method, see the [model cards](README.md) and the [guide](../guide/README.md).

**151 functions.**

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

### `dfgls`

```python
def dfgls(
    y: _ArrayLike,
    regression: str = ...,
    lags: int | None = ...,
    max_lags: int | None = ...,
    method: str = ...,
) -> dict[str, Any]:
```

DF-GLS unit-root test (Elliott-Rothenberg-Stock 1996; null: unit root).

    The ADF test run on a GLS-detrended series (quasi-differenced at the ERS
    local alternative, cbar = -7.0 for "c", -13.5 for "ct") with no
    deterministics in the test regression — near-optimal local power, the
    recommended default over plain ADF. `regression`: "c" (constant, default)
    or "ct" (constant + trend). `lags`: fixed lag count; None selects it by
    `method` ("aic" default, "bic", "t-stat") on the OLS-detrended series
    (Perron-Qu 2007) searching 0..=`max_lags` (default: Schwert's
    ceil(12*(n/100)^(1/4)), capped at (n-1)/2 - 1). When `lags` is given,
    `method`/`max_lags` are ignored (arch behavior). Returns `statistic`,
    `p_value`, `used_lag`, `nobs` (= n - 1 - used_lag), `crit`
    ({"1%","5%","10%"}), `trend`. Statistic and selected lag match
    arch.unitroot.DFGLS (< 1e-10); p-values/critical values are arch's DF-GLS
    response surfaces (Sheppard's MacKinnon-style simulations, transcribed).

### `ng_perron`

```python
def ng_perron(
    y: _ArrayLike,
    trend: str = ...,
    lags: int | str | None = ...,
    max_lags: int | None = ...,
) -> dict[str, Any]:
```

Ng-Perron (2001) M unit-root tests (MZa, MZt, MSB, MPT; null: unit root).

    GLS-detrends `y` through the same engine as `dfgls` (cbar = -7.0 for
    "c", -13.5 for "ct"), selects the ADF lag by the paper's MAIC on the
    detrended series (`lags=None` or `"maic"`, searching 0..=`max_lags`;
    default Schwert's ceil(12*(n/100)^(1/4)) capped at (n-1)/2 - 1) or uses
    a fixed integer `lags`, estimates the autoregressive spectral density at
    frequency zero `s2_ar = sigma2_e / (1 - b(1))^2`, and forms the four M
    statistics. All four reject the unit-root null when SMALL (below the
    critical value); `mzt == mza * msb` exactly. No p-values: no published
    response surface exists for the M tests, so compare each statistic
    against its own critical values (Ng-Perron 2001 Table 1, asymptotic,
    transcribed). Returns dict keys: `mza`, `mzt`, `msb`, `mpt`,
    `used_lag`, `nobs` (= n - 1 - used_lag), `s2_ar`, `crit`
    ({"mza","mzt","msb","mpt"} each {"1%","5%","10%"}), `trend`. Prefer
    this battery over `dfgls` under a suspected large negative MA root;
    caveat (Perron-Qu 2007): on data far from the null MAIC drives the lag
    to its maximum and power collapses — cap `max_lags` or fix `lags`
    there.

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

### `zivot_andrews`

```python
def zivot_andrews(
    y: _ArrayLike,
    regression: str = ...,
    trim: float = ...,
    max_lags: int | None = ...,
    autolag: str | None = ...,
    lags: int | None = ...,
) -> dict[str, Any]:
```

Zivot-Andrews unit-root test with one endogenous break.

    Null: unit root with no break; alternative: stationary around one broken
    deterministic component — `regression` "c" (intercept shift, default),
    "t" (trend-slope shift), "ct" (both); the regression itself always has a
    constant and a trend. The statistic is the minimum t on the lagged level
    over candidate break dates inside the `trim` window (default 0.15, must
    be in (0, 1/3] — 0 itself is unreachable); `break_index` is the last pre-break observation (the
    shift begins at `break_index + 1`). Lag selection follows the
    statsmodels/Baum single up-front convention on the "ct" base ADF:
    `autolag` "aic" (default) / "bic" / "t-stat" capped at `max_lags`, or
    `autolag=None` with `lags` fixed (both None: int(12*(n/100)**0.25)).
    Pass either `lags` or `autolag`, not both. P-values and critical values
    interpolate the statsmodels-simulated null table. Returns dict keys:
    `stat`, `pvalue`, `crit` {"1%","5%","10%"}, `break_index`, `lags`,
    `nobs`, `trim`, `regression`. Matches statsmodels `zivot_andrews`.

### `ndiffs`

```python
def ndiffs(
    y: _ArrayLike, test: str = ..., alpha: float = ..., max_d: int = ...
) -> dict[str, Any]:
```

How many differences a series needs, with the per-order test evidence.

### `nsdiffs`

```python
def nsdiffs(
    y: _ArrayLike, period: int, alpha: float = ..., max_d: int = ...
) -> dict[str, Any]:
```

How many SEASONAL differences a series needs (Hyndman-Khandakar rule).

    D += 1 while the STL seasonal strength is >= 0.64, capped at max_d
    (the forecast::nsdiffs test="seas" rule; alpha is validated but unused
    by this threshold rule, as in forecast). Returns `d`, `period`,
    `threshold`, `alpha`, `max_d`, `stop`, per-order `steps`, and an
    `interpretation`.

### `box_cox_lambda`

```python
def box_cox_lambda(
    y: _ArrayLike,
    method: str = ...,
    bounds: tuple[float, float] = ...,
    period: int | None = ...,
) -> dict[str, Any]:
```

Variance-stabilising Box-Cox lambda (MLE or Guerrero) with its objective.

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

### `summarize`

```python
def summarize(obj: Any, *, title: str | None = ..., wrap: str = ...) -> Any:
```

Render any tsecon output as a readable results object (opt-in).

    `print(tsecon.summarize(tsecon.adf(y)))` works for every function: a plain
    dict becomes a generic `tsecon.results.Result` with an aligned `.summary()`,
    while a bespoke `tsecon.results.*` object is returned unchanged. Additive —
    the returned object is a `dict` subclass, so the plain-dict contract holds.
    `wrap="generic"` forces the structural dump even on a bespoke object.

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

OLS with nonrobust / HC0 / HC1 / HC2 / HC3 / HAC standard errors.

    The leverage-corrected hc2/hc3 are what matter in small samples with
    influential points; hc1's n/(n-k) factor barely moves. HC is
    heteroskedasticity-robust only -- under serial correlation use "hac".

    HAC matches statsmodels cov_type="HAC" when use_correction is matched;
    the DEFAULTS differ deliberately (tsecon True, statsmodels False), so
    pass use_correction=False to reproduce a default statsmodels call.

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
    seasonal: tuple[int, int, int, int] | None = ...,
    constant: bool = ...,
    forecast_steps: int = ...,
    conf_alpha: float | None = ...,
    drift_uncertainty: bool = ...,
) -> dict[str, Any]:
```

Exact-MLE ARIMA(p,d,q) fit, with optional forecast + conf_alpha bands.

    seasonal=(P, D, Q, s) fits the multiplicative SARIMA(p,d,q)(P,D,Q)_s —
    the airline model is seasonal=(0, 1, 1, 12) on the logged series with
    p=0, d=1, q=1, constant=False. Seasonal parameters are named
    statsmodels-style (ar.S.L12, ma.S.L12); differencing (regular and
    seasonal) is simple differencing, losing d + D*s observations.

    Forecast standard errors treat parameters as known by default (the
    statsmodels get_forecast convention). With d >= 1 and constant=True that
    omits the estimated drift's own uncertainty, which grows like h^2 and
    measurably under-covers: 90.2% at h=24, T=60 against a nominal 95%. Pass
    drift_uncertainty=True to add it (94.5% on the same design).

    Also returns bse / param_cov from the observed information, or None with
    cov_ok=False when that matrix is too ill-conditioned to invert honestly.

### `auto_arima`

```python
def auto_arima(
    y: _ArrayLike,
    seasonal_period: int = ...,
    ic: str = ...,
    stepwise: bool = ...,
    max_p: int = ...,
    max_q: int = ...,
    max_P: int = ...,
    max_Q: int = ...,
    max_order: int = ...,
    max_d: int = ...,
    max_D: int = ...,
    d: int | None = ...,
    D: int | None = ...,
    alpha: float = ...,
    forecast_steps: int = ...,
    conf_alpha: float | None = ...,
) -> dict[str, Any]:
```

Automatic ARIMA order selection (Hyndman-Khandakar 2008 stepwise).

    D from the STL seasonal-strength rule (nsdiffs, when
    seasonal_period >= 2), d from successive KPSS tests (ndiffs) on the
    seasonally differenced series, then a stepwise search over
    (p, q, P, Q, constant) minimizing `ic` ("aicc" default, "aic",
    "bic") at those fixed differencing orders; stepwise=False fits the
    exhaustive grid subject to max_order instead (like R, max_order
    binds only the grid). Near-unit-root fits are recorded but never
    selected; failed fits steer the search rather than aborting it.
    Every candidate is fit by the exact-MLE engine behind arima_fit, so
    the search is deterministic. No exogenous regressors in this slice.

    Returns the arima_fit result dict for the selected model plus:
    `order`, `seasonal_order`, `constant`, `converged`, `ic`,
    `ic_value`, `aicc`, `stepwise`, `n_models`, `budget_exhausted`,
    `trace` (every candidate tried, with its criterion and status),
    `d_test` / `D_test` (the full ndiffs / nsdiffs evidence, None when
    fixed or not applicable), and `interpretation`. Honest grading:
    candidate fits are statsmodels-pinned; the selection loop itself is
    graded by Monte-Carlo order recovery (rates in the model card), not
    R/pmdarima parity.

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

    Filter timing (matches `arch`): conditional_volatility[t] is the
    one-step-ahead volatility FOR period t, formed from information through
    t-1 (sigma2_t is built from eps_{t-1} and sigma2_{t-1}); the post-sample
    continuation of that step is `variance_forecast`.

    Boundary fits (a coefficient at its sign constraint, persistence at 1)
    carry per-parameter `se_valid`/`boundary` flags and a `boundary_note`:
    boundary parameters have NaN standard errors (no classical asymptotics
    exist there), interior parameters keep finite ones. `converged` reports
    the optimizer's own verdict.

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
    band: str = ...,
    band_scope: str = ...,
    band_seed: int = ...,
    band_n_sim: int = ...,
) -> dict[str, Any]:
```

Frequentist confidence bands on VAR impulse responses — the banded companion to `var_irf`.

    Returns a dict with `point`/`se`/`lower`/`upper`, each `[h][response][shock]`
    (same layout as `var_irf`), plus echoed `method`/`alpha`/`n_boot`/`band`.
    `method`: "asymptotic" (Lütkepohl 1990 delta-method SEs, Wald bands
    `point ± z_{1-alpha/2}·se`; `n_boot` is `None`) or "bootstrap" (residual
    Efron/Kilian bootstrap, percentile bands, optional Kilian 1998
    `bias_correct`). `orth` and `cumulative` behave exactly as in `var_irf`.

    **Simultaneous bands.** `lower`/`upper` are POINTWISE whatever you pass:
    each covers one `(horizon, response, shock)` cell and promises nothing
    about the path as a whole. Set `band` to `"sup-t"`, `"sidak"` or
    `"bonferroni"` to also get `sim_lower`/`sim_upper` — the same `point` and
    the same `se` with a larger multiplier — plus `critical_value` (a k x k
    grid), `pointwise_critical_value`, `band_scope`, `n_cells` (K) and
    `n_cells_used`. `band="pointwise"` is the default and adds nothing.

    Simultaneous **over what** is your choice and is always reported back:
    `band_scope="horizon"` (default; `K = horizon+1`, one family per
    response-shock pair — the object the coverage audit measured),
    `"shock"` (`K = k(horizon+1)`) or `"all"` (`K = k²(horizon+1)`). Every cell
    added to a family widens the band for every other cell in it.

    **What it fixes and what it does not.** Audit design, nominal 90%, T=500,
    h=0..12, 3000 replications, asymptotic branch: the pointwise band contained
    the whole path in 70.4% ± 0.8 of samples, the sup-t band in 84.8% ± 0.7
    (crate Monte Carlo, T=500, 3000 reps). The published harness measures the
    same shape on its own BASE design: 71.7% ± 1.4 pointwise, 85.2% ± 1.1
    sup-t. The sup-t rate **does not reach nominal**, and the residual is not
    multiplicity — the pointwise band it is built from covers only about 91%
    marginally at h=0 falling to 85.3% at h=12 against nominal 90%. sup-t fixes
    multiplicity exactly and inherits everything else, so what is left needs a
    better standard error, not a bigger multiplier.

    **Shape, on the bootstrap branch.** `lower`/`upper` there are Efron
    *percentile* bounds and pick up bootstrap skewness; the simultaneous band is
    the symmetric `point ± c·se`. These are different shapes of interval, so
    `sim_lower` is **not** guaranteed to sit below `lower` cell by cell. What it
    is guaranteed to contain is the symmetric pointwise band
    `point ± pointwise_critical_value·se` — the like-for-like comparator, in
    which only the multiplier differs.

    `band_seed`/`band_n_sim` drive the Gaussian simulation behind `"sup-t"` on
    the asymptotic branch only, where the band is a pure function of
    `band_seed`. On the bootstrap branch sup-t reads its quantile off the
    bootstrap replications, so `seed` alone reproduces it (use `n_boot` ≥ 999);
    Šidák and Bonferroni are closed forms in K and need neither. Method:
    Montiel Olea and Plagborg-Møller.

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
    band: str = ...,
    band_scope: str = ...,
    band_seed: int = ...,
    band_n_sim: int = ...,
) -> dict[str, Any]:
```

Iterated VAR point forecasts with (1-alpha) intervals.

    **Simultaneous bands.** `lower`/`upper` are MARGINAL whatever you pass: each
    covers one `(horizon, series)` cell. Read as a statement about a whole fan
    chart they are the worst offender in the library — the interval-coverage
    audit, nominal 95% at T=100 over 12 horizons x 2 series, 6000 replications,
    measured the marginal bands containing every cell at once in 41.2% ± 0.6 of
    samples, and still only 48.1% at T=800. That is multiplicity, not a small
    sample.

    Set `band` to `"sup-t"`, `"sidak"` or `"bonferroni"` to also get `se`,
    `sim_lower`/`sim_upper` (same `point`, same `se`, larger multiplier),
    `critical_value` (one per series), `pointwise_critical_value`, `band_scope`,
    `n_cells` (K) and `n_cells_used`. `band="pointwise"` is the default and adds
    nothing. `band_scope="all"` (default) is `K = steps*k`, every horizon of
    every series as one statement — the object the audit measured; `"horizon"`
    is `K = steps`, one family per series.

    **What it fixes and what it does not.** On that design the sup-t joint rate
    was 90.5% ± 0.4 against a nominal 95%. It **does not reach nominal**, and
    the residual is not multiplicity: these intervals are a plug-in treating the
    coefficients as known, so their measured *marginal* rate is 93.3%, not 95%.
    sup-t fixes multiplicity exactly and inherits that approximation unchanged.

    `band_seed`/`band_n_sim` drive the Gaussian simulation behind `"sup-t"`, so
    that band is a pure function of `band_seed`; the closed forms use neither.
    Method: Montiel Olea and Plagborg-Møller.

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
    scale_ar: int = ...,
) -> dict[str, Any]:
```

Minnesota-NIW conjugate BVAR posterior + log marginal likelihood. scale_ar sets the lag order of the AR residual-variance scale regressions (4 = default; 1 = the GLP 2015 convention).

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
    scale_ar: int = ...,
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
    scale_ar: int = ...,
) -> dict[str, Any]:
```

Empirical-Bayes Minnesota-BVAR: pick lambda1 by maximizing the marginal likelihood (Giannone-Lenza-Primiceri 2015). Default hyperprior="glp" (MAP-II under the GLP Gamma hyperprior) — pure ML-II (hyperprior="none") collapses lambda1 to the search-box floor on ~a fifth to a quarter of in-model datasets (audit round 6); a lambda1_opt at the box bottom is a red flag, not a selection. scale_ar=1 switches the prior's residual-scale regressions to GLP's own AR(1) convention (default 4).

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
    kappa0: float | None = ...,
    kappa1: float | None = ...,
    prior_inclusion_cov: float = ...,
    gamma_a: float = ...,
    gamma_b: float | None = ...,
    horizon: int = ...,
    thin: int = ...,
    n_chains: int = ...,
) -> dict[str, Any]:
```

SSVS-BVAR (George-Sun-Ni 2008): spike-and-slab stochastic-search selection of VAR (and error-precision) restrictions by Gibbs; posterior inclusion probabilities, coef/Sigma means, and orthogonalized IRF draws. Default hyperpriors are unit-adaptive (None = scale by the per-equation OLS residual variance); explicit gamma_b/kappa0/kappa1 floats pin absolute prior scales.

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

### `stl`

```python
def stl(
    y: _ArrayLike,
    period: int,
    seasonal: int = ...,
    trend: int | None = ...,
    low_pass: int | None = ...,
    seasonal_deg: int = ...,
    trend_deg: int = ...,
    low_pass_deg: int = ...,
    robust: bool = ...,
    seasonal_jump: int = ...,
    trend_jump: int = ...,
    low_pass_jump: int = ...,
    inner_iter: int | None = ...,
    outer_iter: int | None = ...,
) -> dict[str, Any]:
```

STL seasonal-trend decomposition using LOESS (Cleveland et al. 1990).

    Mirrors statsmodels.tsa.seasonal.STL parameter semantics and defaults
    exactly (matched elementwise at 1e-8; observed ~1e-12); requires
    n >= 2*period. Returns `seasonal`, `trend`, `resid` (y = seasonal +
    trend + resid), `weights` (bisquare robustness weights; all 1 unless
    the outer loop runs), `period`, and `config` (the resolved windows,
    degrees, jumps, and inner/outer iteration counts).

### `mstl`

```python
def mstl(
    y: _ArrayLike,
    periods: Sequence[int],
    windows: Sequence[int] | None = ...,
    iterate: int = ...,
    trend: int | None = ...,
    low_pass: int | None = ...,
    seasonal_deg: int = ...,
    trend_deg: int = ...,
    low_pass_deg: int = ...,
    robust: bool = ...,
    seasonal_jump: int = ...,
    trend_jump: int = ...,
    low_pass_jump: int = ...,
    inner_iter: int | None = ...,
    outer_iter: int | None = ...,
) -> dict[str, Any]:
```

MSTL — STL iterated over multiple seasonal periods
    (Bandara-Hyndman-Bergmeir 2021), e.g. `periods=[24, 168]` for hourly
    data with daily and weekly cycles.

    Matches statsmodels.tsa.seasonal.MSTL elementwise at 1e-8: periods
    sorted ascending, any period >= n/2 dropped (reported in
    `dropped_periods`), per-period seasonal windows from `windows` (None:
    the 7 + 4*k rule -> 11, 15, 19, ...), `iterate` refinement rounds
    (default 2; 1 for a single period), remaining STL keywords forwarded
    to every pass. statsmodels' Box-Cox `lmbda` option is not implemented
    (pre-transform y instead); duplicate periods and iterate=0 are
    refused. Returns `seasonal` (dict of per-period arrays keyed
    "seasonal_<period>"), `trend`, `resid`, `weights` (from the final
    pass), the resolved `periods`/`windows`, `iterate`,
    `dropped_periods`, and per-period `seasonal_strength` (None for a
    constant series).

### `seasonal_strength`

```python
def seasonal_strength(y: _ArrayLike, period: int) -> dict[str, Any]:
```

Wang-Smith-Hyndman seasonal/trend strength from a default STL fit.

    strength = max(0, 1 - var(resid)/var(component + resid)), sample
    variances; near 1 means the component dominates. Returns
    `seasonal_strength`, `trend_strength`, `period`.

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

    Matches statsmodels `ThetaModel(deseasonalize=True, use_test=False)`;
    statsmodels' default additionally pre-tests seasonality and skips
    deseasonalization when the test fails, so the two defaults diverge on
    weakly-seasonal data declared with `period > 1`.

## local projections

### `lp`

```python
def lp(
    y: _ArrayLike,
    shock: _ArrayLike,
    horizons: int = ...,
    n_lag_controls: int = ...,
    se: str | None = ...,
    maxlags: int | None = ...,
    cumulative: bool | str | None = ...,
    band: str | None = ...,
    band_alpha: float = ...,
    band_seed: int = ...,
    band_n_sim: int = ...,
) -> dict[str, Any]:
```

Local projection IRFs; `se` is None (auto), "lag_augmented" or "hac".

    `se=None` (the default) resolves to "lag_augmented" — except under
    `cumulative="both"`, where it resolves to "hac": the cumulated impulse
    `sum_(j=0..h) shock_(t+j)` shares FUTURE shocks across base times up to h
    apart, which past-lag augmentation cannot project out, so lag-augmented
    HC1 standard errors are inconsistent there (audit: 0.507 coverage at a
    nominal 95%, h=12, flat in T) and `se="lag_augmented"` with
    `cumulative="both"` raises. The method actually used is returned as
    `se_method`.

    `cumulative`: False/"none" (level), True/"outcome" (cumulated outcome on
    the contemporaneous impulse — a cumulative IRF, NOT a multiplier), or
    "both" (cumulated outcome on cumulated impulse). For an identified
    multiplier use `lp_multiplier`.

    **Bands.** `band=None` (default) returns the point path and its standard
    errors only, exactly as before. `"pointwise"`, `"sup-t"`, `"sidak"` or
    `"bonferroni"` add `lower`/`upper`, `critical_value`,
    `pointwise_critical_value`, `band_scope`, `n_cells` (K), `n_cells_used`
    and `cov_se_max_rel_diff` (largest relative gap between the band
    covariance's sqrt(diag) and the reported `se`; ~machine epsilon on the
    lag-augmented sup-t path, up to a few percent on the HAC sup-t path,
    None where no covariance is built).
    The family is **the horizons of this one response**, `K = horizons + 1`
    (`band_scope` reports `"horizon"`). A pointwise band covers one horizon at a
    time; the other three cover every horizon at once at `1 - band_alpha`.

    LP is the clean case for this. Audit, nominal 90% over 13 horizons, 400
    replications: pointwise contained the whole path in 36.5% of samples at
    T=240 and sup-t in 81.8%; at T=720, where the per-horizon marginals sit on
    nominal, sup-t lands on nominal too (89.5%) while pointwise reached only
    42.7%. Tripling the sample moved pointwise 36.5% → 42.7% — it is not
    converging, because the problem is multiplicity, not consistency.

    `"sup-t"` builds the cross-horizon covariance and simulates `band_n_sim`
    Gaussian draws from `band_seed`, so the band is a **pure function** of that
    seed; the closed forms use neither. Measured at K=13, alpha=0.10: pointwise
    1.6449, sup-t 2.20–2.65 depending on persistence, Šidák 2.6490, Bonferroni
    2.6653. Method: Montiel Olea and Plagborg-Møller.

### `lp_iv`

```python
def lp_iv(
    y: _ArrayLike,
    impulse: _ArrayLike,
    instrument: _ArrayLike,
    horizons: int = ...,
    n_lag_controls: int = ...,
    cumulative: bool | str | None = ...,
    band: str | None = ...,
    band_alpha: float = ...,
) -> dict[str, Any]:
```

LP-IV: instrumented local projections with a first-stage F diagnostic.

    `cumulative` takes False/"none", True/"outcome" or "both". True/"outcome"
    cumulates only the OUTCOME, giving cumulated response per unit of
    *contemporaneous* impulse — that grows without bound in the horizon and is
    not a multiplier. Use `lp_multiplier` for the Ramey-Zubairy integral
    multiplier.

    **Bands.** `band=None` (default) returns no band. `"pointwise"`, `"sidak"`
    and `"bonferroni"` add `lower`/`upper` over the horizons of this response
    (`K = horizons + 1`, `band_scope="horizon"`) with `critical_value`,
    `pointwise_critical_value`, `n_cells`, `n_cells_used` and
    `cov_se_max_rel_diff` (always None here: no covariance is built).

    `band="sup-t"` is **refused** here with an error saying why: sup-t needs the
    covariance ACROSS horizons and tsecon estimates none for LP-IV, so `lp_iv`,
    `lp_multiplier` and `lp_state` get the **closed-form** simultaneous routes
    only. Šidák and Bonferroni need nothing but K, are valid under arbitrary
    dependence, and are simply wider than a sup-t band would be — never describe
    a band from this function as sup-t. For sup-t use `lp` or `smooth_lp`.

### `lp_multiplier`

```python
def lp_multiplier(
    y: _ArrayLike,
    impulse: _ArrayLike,
    instrument: _ArrayLike,
    horizons: int = ...,
    n_lag_controls: int = ...,
    maxlags: int | None = ...,
    band: str | None = ...,
    band_alpha: float = ...,
) -> dict[str, Any]:
```

Ramey-Zubairy (2018) integral multiplier by one-step LP-IV.

    Regresses the cumulated outcome on the cumulated impulse, instrumented by
    the contemporaneous instrument, controlling for lags of both series. Both
    sides accumulate over the same window, so the coefficient is a multiplier
    rather than a cumulative impulse response. `se` is the kernel-HAC standard
    error of that single 2SLS coefficient — inference on the multiplier
    itself, not a delta-method ratio and not a leg's SE relabelled.

    **Bands.** `band=None` (default) returns no band. `"pointwise"`, `"sidak"`
    and `"bonferroni"` add `lower`/`upper` around `multiplier` over the horizons
    of this path (`K = horizons + 1`, `band_scope="horizon"`) with
    `critical_value`, `pointwise_critical_value`, `n_cells`, `n_cells_used` and
    `cov_se_max_rel_diff` (always None here: no covariance is built).
    `band="sup-t"` is **refused**: no cross-horizon covariance is estimated for
    the multiplier path, so this function (like `lp_iv` and `lp_state`) gets the
    closed-form routes only. Do not call such a band sup-t.

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

### `proxy_svar_bands`

```python
def proxy_svar_bands(
    data: _ArrayLike,
    proxy: _ArrayLike,
    lags: int = ...,
    horizon: int = ...,
    norm_var: int = ...,
    unit: float = ...,
    trend: str = ...,
    alpha: float = ...,
    n_boot: int = ...,
    seed: int = ...,
    bands: str = ...,
    block_length: int | None = ...,
    robust_f: bool = ...,
) -> dict[str, Any]:
```

Confidence bands for a proxy-SVAR impulse response.

    bands="moving_block" (default) is the Jentsch-Lunsford moving-block
    bootstrap: (u_t, m_t) resampled jointly, the VAR reconstructed and
    re-estimated per draw, the unit-effect normalization re-imposed per draw.
    bands="wild" reproduces Mertens-Ravn / Gertler-Karadi but is NOT
    asymptotically valid here -- a common Rademacher draw leaves the
    identifying moment bit-identical across draws, so it carries no bootstrap
    variability. Check asymptotically_valid / validity_note.

    Returns lower/upper (Hall, recommended) and lower_efron/upper_efron.
    The h=0 entry for norm_var is degenerate at `unit` by construction.
    Bands are pointwise, not joint. Failed draws are counted by reason in
    `failures`, never dropped; a nonzero n_failed means the instrument may be
    too weak for a Wald band -- see proxy_ar_sets.

### `proxy_ar_sets`

```python
def proxy_ar_sets(
    data: _ArrayLike,
    proxy: _ArrayLike,
    lags: int = ...,
    horizon: int = ...,
    norm_var: int = ...,
    unit: float = ...,
    trend: str = ...,
    alpha: float = ...,
    variance: str = ...,
    hac_lags: int | None = ...,
    reduced_form_uncertainty: bool = ...,
    rf_method: str = ...,
    rf_draws: int | None = ...,
    rf_seed: int | None = ...,
) -> dict[str, Any]:
```

Weak-instrument-robust (Anderson-Rubin) confidence SETS for a proxy SVAR.

    Under weak identification no bounded set can be honest (Dufour 1997), so a
    cell may be a bounded interval, the COMPLEMENT of an interval (kind
    "exterior", two rays), a single ray ("ray_below"/"ray_above"), the whole
    line, empty, or a point. That shape is the answer.
    Do not read an "exterior" set as an interval -- `lower`/`upper` are the
    set's own bounds (+/-inf there) and `excluded_lower`/`excluded_upper` are
    the rejected middle. `excludes_zero` on an unbounded set does NOT establish
    a sign: both signs can be members.

    Reduced-form uncertainty is propagated by default. Omitting it is
    catastrophic on an estimated VAR -- measured coverage 0.952 at h=0 falling
    to 0.119 by h=8 against nominal 0.95, versus 0.952 to 0.913 with it. When
    reduced_form_uncertainty=False the returned `level` is None, because a set
    conditional on the reduced form has no honest 1-alpha label.

    rf_method="second_order" (with rf_draws/rf_seed) replaces the first-order
    delta propagation with seeded exact simulation of the coefficient
    uncertainty through the nonlinear MA map -- the measured long-horizon
    repair (h=12 coverage 0.889 -> 0.964 on the card's VAR(2) at T=300, 0.830
    -> 0.932 on a routine VAR(1) at T=250; median width ~1.15x at h=8, ~1.45x
    at h=12; weak-instrument boundedness bit-identical). Default "delta" is
    unchanged.

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
    `impact`, `relative_impact`, `cov_um`, `first_stage_f` (HC1-robust when
    `robust_f`), `reliability` = Corr(m, u_norm)^2, `n_proxy`, the estimated
    `shock` (length T), and `first_stage`: the proxy_first_stage diagnostics
    dict (the MOP effective F with its tau-based critical values --
    mop_cv_tau10 = 23.11 is the conventional bar, not the folklore 10).
    Point estimate only; see proxy_svar_bands for moving-block bands
    (strong instrument) and proxy_ar_sets for weak-IV-robust sets (use when
    first_stage["weak_mop_tau10"] is True).

### `proxy_first_stage`

```python
def proxy_first_stage(
    data: _ArrayLike,
    proxy: _ArrayLike,
    lags: int = ...,
    norm_var: int = ...,
    trend: str = ...,
    variance: str = ...,
    hac_lags: int | None = ...,
) -> dict[str, Any]:
```

First-stage strength diagnostics: the Montiel Olea-Pflueger effective F.

    With one instrument the MOP effective F equals the robust F (the squared
    robust t of the first-stage slope; Windmeijer 2025), reported under
    variance="hc1" (default), "hac" (Bartlett/Newey-West, hac_lags defaulting
    to the Newey-West rule -- for serially correlated proxies), or
    "classical" (for comparison with published homoskedastic tables).

    Returns `beta`, `se`, `effective_f`, `f_classical`, `f_hc1`,
    `reliability`, `n_proxy`, `hac_lags`, the MOP critical values at 5% test
    level (`mop_cv_tau5/10/20/30` = 37.42 / 23.11 / 15.06 / 12.05 -- the
    null "worst-case relative bias > tau"), `tau_bound` (the smallest tau the
    observed effective F rejects; +inf when even zero relevance cannot be
    rejected), and the verdicts `weak_mop_tau10` (the honest bar) and
    `weak_folklore` (F < 10, kept only because the literature reports it).
    When weak_mop_tau10 is True do not trust Wald-type bands
    (proxy_svar_bands); use proxy_ar_sets.

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
    bias_correction: str = ...,
    band: str | None = ...,
    band_alpha: float = ...,
) -> dict[str, Any]:
```

Panel local projection of a common shock with fixed effects.

    `outcome` is N x T; `shock` is length T. Fixed effects + lagged outcomes
    + short T carry Nickell bias (horizon-amplified); two half-panel
    corrections are offered. `jackknife=True` (equivalently
    `bias_correction="dj"`) is the Dhaene-Jochmans half-panel jackknife:
    corrected point estimates, full-sample plug-in standard errors
    (measured cost: the estimator's variance inflates at short T while se
    is unchanged — coverage 0.88 -> 0.80 at T=60, equivalence by T ~ 240).
    `bias_correction="spj"` is the Mei-Sheng-Shi (2026, J. Int. Economics)
    split-panel jackknife for panel LPs: leads/lags stay full-panel, the
    regression rows split at the median usable period, and the standard
    errors are recomputed for the corrected estimator (adjusted-score
    cluster or Driscoll-Kraay sandwich, matching their pLP reference
    implementation; `se_type="nonrobust"` is refused under "spj").
    Combining `jackknife=True` with `bias_correction="spj"` raises.

    Returns a dict with `irf`, `se`, `nobs` (each length horizon+1) and the
    stamped `se_type`, `cumulative`, `jackknife`, `bias_correction`.

    **Bands.** `band=None` (default) returns no band. `"pointwise"`, `"sidak"`
    and `"bonferroni"` add `lower`/`upper` over the horizons of this response
    (`K = horizon + 1`, `band_scope="horizon"`) at level `band_alpha`, with
    `critical_value`, `pointwise_critical_value`, `n_cells`, `n_cells_used`
    and `cov_se_max_rel_diff` (always None here: no covariance is built;
    `band_n_sim`/`band_seed` come back 0 — no simulation ran). A pointwise
    band covers one horizon at a time; the closed-form simultaneous routes
    cover every horizon at once at `1 - band_alpha` (Montiel Olea and
    Plagborg-Møller's simultaneous-bands framework; joint coverage measured
    in `test_simultaneous_bands.py` — see the panel model card).

    `band="sup-t"` is **refused** with an error saying why: sup-t needs the
    covariance ACROSS horizons and tsecon estimates none for the panel LP
    (a cross-horizon panel covariance is a documented follow-up), so
    `panel_lp` gets the closed-form routes only, like `lp_iv`,
    `lp_multiplier` and `lp_state`. Never describe such a band as sup-t.

### `lp_did`

```python
def lp_did(
    outcome: _ArrayLike,
    treatment: _ArrayLike,
    pre_window: int = ...,
    post_window: int = ...,
    absorbing: bool = ...,
    nonabsorbing_lag: int = ...,
    reweight: bool = ...,
    pooled: bool = ...,
    never_treated_only: bool = ...,
) -> dict[str, Any]:
```

LP-DiD event-study difference-in-differences (Dube-Girardi-Jordà-Taylor).

    `outcome` and `treatment` are N x T (treatment binary 0/1). Per horizon,
    regresses `y[i, t+h] - y[i, t-1]` on the treatment switch with period
    effects, using only clean controls (not-yet-treated; stabilized units
    under `absorbing=False` with `nonabsorbing_lag`; never-treated when
    `never_treated_only=True`) — avoiding the negative-weight comparisons of
    TWFE event studies. `reweight=True` gives the equally-weighted ATT;
    `pooled=True` adds pooled post/pre estimates. Entity-clustered SEs in
    the authors' fixest/reghdfe convention.

    Returns a dict with `horizons` (-pre_window..post_window; -1 is the
    omitted baseline, stored as zeros), `coef`, `se`, `nobs`, `n_switchers`
    (clean samples shrink with |h| — read them), pooled keys
    (`pooled_post_att`, `pooled_post_se`, `pooled_post_nobs`,
    `pooled_post_n_switchers`, and `pooled_pre_*` when `pre_window >= 2`)
    only when `pooled=True`, and the stamped `absorbing`,
    `nonabsorbing_lag`, `reweight`, `pooled`, `never_treated_only`,
    `se_type`.

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

### `var_backtest`

```python
def var_backtest(
    returns_or_hits: _ArrayLike,
    var_forecasts: _ArrayLike | None = ...,
    alpha: float = ...,
    dq_lags: int = ...,
    input: str = ...,
) -> dict[str, Any]:
```

VaR backtest battery: Kupiec unconditional coverage, Christoffersen
    independence/conditional coverage, and the Engle-Manganelli DQ test.

    Sign convention: returns and VaR forecasts on the same (return) scale,
    `var_forecasts[t]` the alpha-quantile of the conditional return
    distribution (negative for small alpha); a violation is return < VaR.
    `alpha` is the VaR coverage level (0.05 for a 95% VaR), not a test
    size. With `var_forecasts` the first argument is a return series;
    without, a pre-computed 0/1 violation sequence (`input="hits"`
    combines pre-computed hits WITH VaR forecasts so the DQ regression
    keeps its VaR regressor). Returns the three statistics with p-values,
    the violation counts/transition cells, and a teaching `verdict`.

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

Johansen cointegration test (data is T x k); trace + max-eig + rank + evec.

    Matches statsmodels ``coint_johansen(det_order=0)`` — the *unrestricted
    constant* convention. Warning: ``vecm``'s default is ``deterministic="n"``
    (no deterministic terms), a different case; fit the VECM this test ranks
    with ``vecm(..., deterministic="co")``.

### `engle_granger`

```python
def engle_granger(
    data: _ArrayLike,
    trend: str = ...,
    autolag: str | None = ...,
    maxlag: int | None = ...,
) -> dict[str, Any]:
```

Engle-Granger two-step cointegration test: stat + MacKinnon p-value/crit (statsmodels `coint`).

### `vecm`

```python
def vecm(
    data: _ArrayLike,
    k_ar_diff: int = ...,
    coint_rank: int = ...,
    deterministic: str = ...,
) -> dict[str, Any]:
```

VECM ML estimation: alpha, beta, gamma, det_coef, sigma_u, llf (statsmodels-exact).

    ``deterministic``: ``"n"`` (default) — no deterministic terms, statsmodels
    ``VECM(..., deterministic="n")``; ``"co"`` — unrestricted constant
    (statsmodels ``"co"``, the case ``johansen``'s det_order=0 convention
    assumes; the intercepts land in ``det_coef``). Warning: ``johansen``
    assumes the unrestricted constant, NOT this default — pass
    ``deterministic="co"`` when the rank came from ``johansen``.

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

    smoothed_prob / filtered_prob are the full (n, k_regimes) probability
    matrices, n = len(y) - order; smoothed_prob_last_regime keeps the 0.2.0
    scalar path (= smoothed_prob[:, -1]).

### `setar`

```python
def setar(
    y: _ArrayLike,
    p: int,
    delay: int = ...,
    trim: float = ...,
    delays: Sequence[int] | None = ...,
    ic: str = ...,
    constant: bool = ...,
) -> dict[str, Any]:
```

Two-regime SETAR(p) (Tong-Lim 1980) by concentrated least squares
    (Hansen 1997): grid over the trimmed order statistics of y_{t-delay},
    per-candidate OLS in each regime, pooled-SSR-minimizing threshold (and
    delay, when `delays` is a list — all candidates then share the common
    sample t >= max(p, max(delays)) so SSRs are comparable; `delays`
    overrides `delay`).

    Keys: threshold, delay, params_low/params_high (constant first) with
    classical nonrobust bse_low/bse_high, n_low/n_high/nobs, pooled ssr and
    sigma2 = SSR/(nobs - 2k), sigma2_low/sigma2_high, aic/bic (n ln(SSR/n) +
    penalty * m, m = 2k + 1 counting the threshold), ic/ic_used (`ic`
    selects which criterion is *reported* — with p fixed the SSR ranking and
    the IC ranking coincide), min_regime, k, and the candidate grid
    thresholds with its ssr_path. Validated against an independent NumPy
    transcription of the published algorithm (fixtures/setar.json).

### `setar_test`

```python
def setar_test(
    y: _ArrayLike,
    p: int,
    delay: int = ...,
    trim: float = ...,
    n_boot: int = ...,
    seed: int = ...,
) -> dict[str, Any]:
```

Hansen (1996) sup-F linearity test against a two-regime SETAR(p):
    stat = nobs (ssr_linear - ssr_setar)/ssr_setar over the trimmed
    threshold grid. The threshold is unidentified under the null (Davies
    problem), so NO chi-squared p-value exists; p_value = (1 + #{F* >= F}) /
    (n_boot + 1) from the fixed-regressor wild bootstrap (y* = resid * eta,
    eta iid N(0,1), same fixed regressors, same grid) — seeded, parallel,
    bit-identical at any thread count.

    Keys: stat, p_value, threshold, delay, n_boot, nobs, ssr_linear,
    ssr_setar, thresholds, f_path, boot_stats.

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
def dcc_garch(
    returns: _ArrayLike,
    variant: str = ...,
    dist: str = ...,
    forecast_horizon: int = ...,
) -> dict[str, Any]:
```

DCC-GARCH (Engle 2002); returns is T x k. variant: "dcc" | "cdcc"
    (Aielli 2013 consistent targeting) | "adcc" (Cappiello-Engle-Sheppard
    2006 asymmetric); dist: "normal" | "t" (second-stage likelihood; "t"
    adds nu). Keys: a, b, g, qbar, loglik, converged, variant, dist,
    correlation ((T, k, k) nested list -- the in-sample conditional
    correlation path), correlation_last, and with forecast_horizon > 0
    correlation_forecast / covariance_forecast ((horizon, k, k)) and
    variance_forecast ((horizon, k)).

    Timing: correlation[t] = R_t conditions on information through t-1
    (filter convention; Q_0 = Qbar). correlation_last = correlation[-1] is
    the last IN-SAMPLE matrix, not a forecast; the one-step-ahead R_{T+1}
    also uses the final residual z_T and is correlation_forecast[0].
    h >= 2 forecasts use the Engle-Sheppard (2001) recursion on E[Q]
    normalized each step (an approximation), converging to corr(qbar).
    The default call is bit-identical to earlier releases.

### `dcc_test`

```python
def dcc_test(returns: _ArrayLike, lags: int = ...) -> dict[str, Any]:
```

Engle-Sheppard (2001) test of constant conditional correlation
    (CCC vs DCC); returns is T x k. GARCH(1,1) per series, joint
    standardization by the symmetric inverse square root of the constant
    correlation, pooled AR(lags) on the stacked off-diagonal outer
    products. Keys: stat, df (= lags + 1), p_value (small rejects constant
    correlation), lags, nobs, n_stacked.

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

    The aggregates follow Corsi's definition and INCLUDE the daily lag:
    weekly = mean(RV_{t-1}..RV_{t-5}), monthly = mean(RV_{t-1}..RV_{t-22}).
    (Changed in 0.5: through 0.4.0 the windows mistakenly excluded RV_{t-1};
    coefficients on the same data shift.)

    variant is "level", "log", or "sqrt". use_correction now defaults True
    (False through 0.2.0): bse/tvalues carry the finite-sample sqrt(n/(n-k))
    factor by default. statsmodels cov_type="HAC" defaults the correction
    off -- pass use_correction=False to match it (and the old numbers).

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
    bandwidth: float | None = ...,
    tol: float = ...,
    max_iter: int = ...,
) -> dict[str, Any]:
```

Linear IV-GMM (Hansen 1982) with robust or HAC weighting.

    POSITIONAL ORDER IS (x, z, y): regressors, instruments, outcome. x and z
    are both 2-D float matrices, so swapping them coerces cleanly and returns
    plausible-looking garbage -- prefer keywords: iv_gmm(x=X, z=Z, y=y).

    bandwidth defaults to None, which selects the Newey-West rule of thumb.
    It previously defaulted to 0.0 -- a Bartlett kernel truncated at zero
    lags IS White, so weight="hac" used to be a silent no-op returning
    results bit-identical to weight="robust". An explicit bandwidth=0.0 now
    raises. The truncation actually used comes back as hac_bandwidth.
    Neither setting restores nominal coverage under persistent moments: the
    audit measured 0.868 against nominal 0.95 at bandwidth=10.

    Also returns first_stage, a list of per-regressor weak-instrument F
    diagnostics keyed by "regressor". Entries are omitted where the
    statistic is undefined, so it may be shorter than the regressor count,
    and a missing entry is not a failed fit. With two or more endogenous
    regressors these are NOT a weak-identification test -- all can clear 10
    while the system is under-identified. F > 10 is not a safety threshold
    even with one: coverage was 0.915 at a median F of 10.5.

    method is "2sls", "2step", or "iterated"; weight is "robust" or "hac".
    Z must include the exogenous regressor columns. Returns params, bse, cov,
    residuals, nobs, nmoments, nparams, steps, hac_bandwidth, first_stage,
    and (over-identified) the Hansen j_stat/j_dof/j_pval.

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
    {"train": [...], "test": [...]} index dicts. purge drops the last purge
    indices from the end of every training window (all schemes; set it >=
    horizon - 1 for h-step-ahead labels). embargo excludes training rows
    after the test block, which only exist under "purged_kfold"; nonzero
    embargo raises on "expanding"/"rolling".

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

## conformal forecast intervals

### `conformal_forecast`

```python
def conformal_forecast(
    y: _ArrayLike,
    horizon: int = ...,
    method: str = ...,
    base: str = ...,
    alpha: float = ...,
    calib: int | None = ...,
    mode: str = ...,
    period: int = ...,
    gamma: float = ...,
    n_eval: int | None = ...,
    lags: int = ...,
    n_boot: int = ...,
    seed: int = ...,
    optimize_beta: bool = ...,
    order: tuple[int, int, int] | None = ...,
) -> dict[str, Any]:
```

Distribution-free conformal forecast intervals around a point forecaster.

    method is "split" (finite-sample-corrected residual-quantile calibration
    on held-out origins; mode "symmetric" or "asymmetric"), "enbpi" (Xu-Xie
    2021 bootstrap-ensemble batch prediction intervals; base must be "ar"),
    or "aci" (Gibbs-Candes 2021 adaptive conformal inference,
    alpha_{t+1} = alpha_t + gamma (alpha - err_t), gamma default 0.005 from
    the paper). base wraps "theta", "naive", "drift", "mean",
    "seasonal_naive", "ar", or "arima" (order=(p, d, q)). calib defaults to
    n // 4 residuals per horizon; n_eval (aci) to n // 5. Returns mean,
    lower, upper, level, plus per-method calibration diagnostics (split:
    q_lower/q_upper/scores/finite_sample_level; enbpi: beta/residuals;
    aci: alpha_final/alpha_trajectory/err/realized_coverage).

### `conformal_backtest`

```python
def conformal_backtest(
    y: _ArrayLike,
    horizon: int = ...,
    method: str = ...,
    base: str = ...,
    alpha: float = ...,
    calib: int | None = ...,
    mode: str = ...,
    period: int = ...,
    gamma: float = ...,
    n_eval: int | None = ...,
    lags: int = ...,
    n_boot: int = ...,
    batch: int = ...,
    seed: int = ...,
    optimize_beta: bool = ...,
    order: tuple[int, int, int] | None = ...,
) -> dict[str, Any]:
```

Online out-of-sample evaluation of conformal intervals ("split",
    "aci", or "enbpi") over the last n_eval origins: per-origin intervals
    formed from information available then, miss indicators, and realized
    coverage per horizon. ACI adds its alpha_t trajectory; EnbPI is the
    published one-step online algorithm with the residual window sliding
    by batch.

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
    returned as a NumPy array or list of lists -- the return must be 2-D even
    for a single moment condition (reshape with g.reshape(-1, 1)); a 1-D return
    raises a TypeError naming moments_fn. weight is the flattened m*m
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
    se: str | None = ...,
    maxlags: int | None = ...,
    cumulative: bool | str | None = ...,
    band: str | None = ...,
    band_alpha: float = ...,
) -> dict[str, Any]:
```

State-dependent (interacted) local projections (Ramey-Zubairy 2018); per-regime IRFs and SEs.

    `cumulative` takes False/"none", True/"outcome" or "both", as in `lp`.
    `se=None` (the default) resolves to "lag_augmented" — except under
    `cumulative="both"`, where it resolves to "hac" for the same reason as in
    `lp` (the cumulated impulse shares future shocks across nearby base
    times, so lag-augmented HC1 is inconsistent there; audit: 0.640 coverage
    at a nominal 95%, h=12) — and `se="lag_augmented"` with
    `cumulative="both"` raises. The method actually used is returned as
    `se_method`.

    **Bands.** `band=None` (default) returns no band. `"pointwise"`, `"sidak"`
    and `"bonferroni"` add one band PER REGIME —
    `lower_state1`/`upper_state1` and `lower_state0`/`upper_state0`, with
    `critical_value_state1`/`critical_value_state0`,
    `n_cells_used_state1`/`n_cells_used_state0` and
    `cov_se_max_rel_diff_state1`/`cov_se_max_rel_diff_state0` (always None
    here: no covariance is built) — over the horizons of that
    regime's own response (`K = horizons + 1`, `band_scope="horizon"`). The two
    regimes are banded separately; nothing here is simultaneous *across*
    regimes.

    `band="sup-t"` is **refused**: no cross-horizon covariance is estimated for
    the interacted regressions, so `lp_state` (like `lp_iv` and `lp_multiplier`)
    gets the closed-form simultaneous routes only. Report such a band as Šidák
    or Bonferroni, never as sup-t.

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

## score-driven models (GAS/DCS)

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

### `dcs_local_level`

```python
def dcs_local_level(y: _ArrayLike, density: str = ...) -> dict[str, Any]:
```

DCS robust local level mu_{t+1} = mu_t + kappa*u_t (Harvey-Luati 2014).

    MLE of (kappa, scale[, nu]). density is "t" (default; bounded redescending
    score — robust to additive outliers), "laplace" (sign filter, tracks a
    local median), or "gaussian" (exactly the steady-state Kalman local level;
    kappa = steady-state gain). Returns kappa/scale (+ nu) with
    observed-information *_se, the one-step-predicted level path, resid,
    next_level, loglik, aic, bic, honest converged, iterations, n_obs,
    density.

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
    r: _ArrayLike,
    xs: _ArrayLike,
    cz: float = ...,
    alpha: float = ...,
    joint: str = ...,
) -> dict[str, Any]:
```

Joint IVX predictability test for several persistent predictors (xs is T x k).

    Returns beta_ivx, the joint wald/pvalue, rz, nregressors, nobs. The
    default is joint="bonferroni" (changed in 0.5; through 0.4.0 the default
    was "chi2"): per-predictor scalar IVX tests combined at level/k, whose
    measured size is at or below nominal for every measured k, with power on
    par with a size-corrected chi-square test for sparse alternatives. It
    adds wald_scalar/pvalue_scalar/joint keys, and its `wald` is the LARGEST
    scalar statistic (chi-square(1) scale) with `pvalue` already
    Bonferroni-adjusted. The flip is measured, not stylistic: the
    joint="chi2" chi-square(k) Wald's size degrades in k near a unit root
    (0.28 at k=8, n=250, nominal 0.05) and n does not repair it (alpha=0.5
    restores convergence but still ~0.13 at k=8, n=250); chi2 stays
    available for small k or rho safely below 1 — see the
    predictive-regressions model card.

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
    use_correction defaults True (the n/(n-k) HAC scaling); statsmodels
    cov_type="HAC" defaults it off -- match it when comparing.

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

    use_correction defaults True (the n/(n-k) HAC scaling); statsmodels
    cov_type="HAC" defaults it off.

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

Estimate the memory parameter d; method is "gph" or "local_whittle".

    Returns d, se, se_asymptotic and m for both methods, plus se_regression for
    method="gph". BUILD INTERVALS FROM `se`: it is the standard error at the
    bandwidth actually used. `se_asymptotic` is the textbook large-m closed form
    (pi/sqrt(24m) for GPH, 1/(2*sqrt(m)) for local Whittle), kept for reference
    -- at the default bandwidth it is materially too NARROW, measured about 25%
    at n=512.

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

## ACM term premium

### `acm_term_premium`

```python
def acm_term_premium(
    yields: _ArrayLike,
    maturities: Sequence[int],
    n_factors: int = ...,
    periods_per_year: float = ...,
) -> dict[str, Any]:
```

ACM regression-based term premium (Adrian-Crump-Moench 2013).

    The three-step estimator: PCA factors from the yield panel, a factor
    VAR(1), excess-return regressions on lagged factors and contemporaneous
    innovations, the convexity-adjusted lambda0/lambda1 price-of-risk OLS,
    then affine log-price recursions with and without the prices of risk.
    Decomposes fitted yields into risk-neutral (expected-short-rate) yields
    and the term premium.

    UNITS: `yields` is T x M of ANNUALIZED continuously-compounded zero-coupon
    log yields in DECIMAL (divide percent by 100 — the convexity terms are
    quadratic, so percent input misprices them, it does not just rescale).
    `maturities` are integer PERIODS (months for monthly data), ascending,
    containing 1; excess returns need n - 1 in the grid for each return
    maturity n (contiguous grid or pairs; interpolate the curve first if
    needed). Returns factors, factor_loadings, mu/phi/sigma, rx_maturities,
    a/beta/c, sigma2, lambda0/lambda1, delta0/delta1, A/B, A_rn/B_rn,
    fitted / risk_neutral / term_premium (T x M, fitted = risk_neutral +
    term_premium), var/rx/short_rate/yield R-squareds, and the echoed
    inputs maturities / n_factors / periods_per_year. The premium's
    LEVEL is estimation-sample sensitive; compare only across models fit on
    the same sample.

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
    reports whether the raw fitted quantile paths crossed either way. `bse`
    carries the Newey-West overlap correction at `hac_lags = horizon - 1`
    lags; `bse_powell` is the uncorrected Powell sandwich (the statsmodels
    `QuantReg` number), identical to `bse` at `horizon = 1`.

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
    se, nobs. Per-element se conditions on the scores: inconsistent for
    functional_pca-estimated scores (generated regressors) — flp_scenario's
    w'beta contrasts are immune; see the functional-shocks model card.

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
    band: str | None = ...,
    band_alpha: float = ...,
    band_seed: int = ...,
    band_n_sim: int = ...,
) -> dict[str, Any]:
```

Smooth local projections (Barnichon-Brownlees 2019): the IRF as a
    penalized B-spline in the horizon, estimated jointly across horizons.

    `lam`: a float fixes the smoothing parameter (0.0 reproduces the
    per-horizon `lp(se="hac")` point estimates with the default basis);
    "cv"/None cross-validates it by leave-h-block-out CV over `lambda_grid`.
    `lambda_grid=None` uses the default **scale-relative** grid — a 17-point
    log ladder spanning eight decades, anchored to the mean diagonal of the
    spline block of the stacked X'X, so the selected smoothing (and the
    unit-normalized IRF) is invariant to rescaling `y` and/or `shock`; an
    explicit `lambda_grid` is absolute (in the units of your data) and used
    verbatim, and `cv_grid` always reports the grid actually searched.
    `penalty_order=2` shrinks the IRF toward
    a straight line as `lam` grows. `se` conditions on `lam` and does not
    account for shrinkage bias; `irf_raw`/`se_raw` are the unsmoothed
    per-horizon HAC LP for comparison. Keys: horizons, irf, se, lambda_used,
    cv_grid, cv_scores, theta, irf_raw, se_raw.

    **Bands.** `band=None` (default) returns no band. `"pointwise"`, `"sup-t"`,
    `"sidak"` or `"bonferroni"` add `lower`/`upper` over the horizons of this
    response (`K = horizons + 1`, `band_scope="horizon"`) with
    `critical_value`, `pointwise_critical_value`, `n_cells`, `n_cells_used`
    and `cov_se_max_rel_diff` (~machine epsilon here: the band covariance IS
    the delta-method matrix behind `se`; None where no covariance is built).
    A pointwise band covers one horizon at a time; the other three cover every
    horizon at once at `1 - band_alpha`.

    Smooth LP is the one estimator here that already had the full cross-horizon
    covariance — the path is `irf_h = B_h' theta` for a single jointly-estimated
    coefficient vector — so `"sup-t"` needs no extra estimation and no
    compromise. It simulates `band_n_sim` Gaussian draws from `band_seed`, so
    the band is a **pure function** of that seed. The usual smooth-LP caveat
    still applies and is not a band problem: `se` conditions on `lam` and
    ignores the penalty's shrinkage bias, so any band here is centred on a
    shrunk estimator. Method: Montiel Olea and Plagborg-Møller.

## extreme value theory

### `gpd_fit`

```python
def gpd_fit(
    y: _ArrayLike,
    threshold: float | None = ...,
    quantile: float = ...,
    p_tail: Sequence[float] | None = ...,
) -> dict[str, Any]:
```

Peaks-over-threshold GPD tail fit with McNeil-Frey (2000) VaR/ES.

    Fits a generalized Pareto distribution by MLE to the strict exceedances
    of `y` over `threshold` (default: the empirical `quantile` of `y`,
    numpy-linear convention; both the threshold and its quantile are
    reported). `xi` is the tail index — scipy's `genpareto` `c` is the same
    quantity (matches `scipy.stats.genpareto.fit(z, floc=0)`, polished, at
    1e-6). Standard errors are observed-information; when `xi <= -0.5`
    (Smith 1985 irregularity) they are reported but `se_valid` is False.
    `var`/`es` are the McNeil-Frey POT tail quantiles at each `p_tail` entry
    (default [0.99, 0.995, 0.999]; each must reach beyond the threshold:
    `1 - p < n_exceed / n`) in the units of `y` — fit losses (`-returns` or
    `abs(returns)`) to read them as risk numbers; `es` is NaN where
    `xi >= 1`. At least 10 exceedances are required. Keys: threshold,
    threshold_quantile, n, n_exceed, exceed_rate, xi, beta, se_xi, se_beta,
    se_valid, loglik, converged, p_tail, var, es.

### `gev_fit`

```python
def gev_fit(
    y: _ArrayLike,
    block_size: int | None = ...,
    return_periods: Sequence[float] | None = ...,
) -> dict[str, Any]:
```

GEV block-maxima fit with return levels.

    With `block_size=None`, `y` IS the pre-computed block maxima; otherwise
    `y` is cut into consecutive non-overlapping blocks of that length (a
    trailing partial block is dropped) and each block contributes its
    maximum. Fits GEV(`xi`, `mu`, `sigma`) by MLE — `xi` is the tail index;
    scipy's `genextreme` shape is `c = -xi` (matches
    `scipy.stats.genextreme.fit(maxima)`, polished, at 1e-6). Standard
    errors are observed-information with the same `se_valid` certification
    as `gpd_fit` (`xi <= -0.5` reported, not certified). `return_levels`
    are the `1 - 1/T` GEV quantiles at each `return_periods` entry (default
    [10, 50, 100] blocks; each `T > 1`). At least 10 maxima are required.
    Keys: xi, mu, sigma, se_xi, se_mu, se_sigma, se_valid, loglik,
    converged, n_maxima, block_size, return_periods, return_levels.

## static copulas

### `pseudo_obs`

```python
def pseudo_obs(x: _ArrayLike) -> _F64:
```

Pseudo-observations: the average-rank probability-scale transform.

    `u[i, j] = rank of x[i, j] within column j / (n + 1)`, ties assigned
    their average rank — exactly scipy `rankdata(method="average")/(n+1)`
    (golden-pinned, ties included). The `n + 1` denominator keeps every
    value strictly inside (0, 1), which the copula quantile transforms
    require. Ranks see only order, so any strictly INCREASING transform of
    a margin (logs, standardization, exp) leaves the output — and any
    copula fitted to it — bit-identical (property-tested). A strictly
    decreasing transform instead reverses that margin's ranks (`u -> 1 - u`
    when there are no ties), flipping the sign of the fitted dependence —
    the standard copula invariance is increasing-only. This is the one-line
    companion to `copula_fit`: `copula_fit(pseudo_obs(x))`. Accepts any
    number of columns (the transform is columnwise); `copula_fit` itself
    is bivariate in this slice.

### `copula_fit`

```python
def copula_fit(
    u: _ArrayLike,
    family: str = ...,
    method: str = ...,
) -> dict[str, Any]:
```

Fits a bivariate copula to (n, 2) probability-scale pseudo-observations.

    `u` must lie strictly inside (0, 1): rank/PIT-transform the raw margins
    first — `pseudo_obs(x)` does it in one line, and the whole workflow is
    then invariant to strictly increasing transforms of each margin (the
    point of the copula decomposition; property-tested — a decreasing
    transform flips the sign of the dependence instead). At least 20 pairs
    required.

    `family`: "gaussian" (param `rho`), "t" (`rho`, `nu`), "clayton"
    (`theta` > 0, lower-tail), "gumbel" (`theta` >= 1, upper-tail), "frank"
    (`theta`, either sign). Clayton/Gumbel model positive dependence only
    in this slice (rotations deferred) and raise a teaching error when the
    empirical Kendall tau is <= 0. `method`: "mle" (maximum likelihood,
    observed-information SEs — matches a polished scipy optimum of the
    statsmodels log-density at 1e-6) or "tau" (Kendall-tau inversion, the
    statsmodels `fit_corr_param` route — for "t", tau pins `rho` and `nu`
    is profiled by MLE; SEs are NaN with `se_valid` False, honestly, since
    the moment-based SE is deferred).

    Returns the named dependence parameter(s) (`rho` / `rho` + `nu` /
    `theta`, also stacked in `params` with `param_names`), their SEs
    (`se_rho` / `se_nu` / `se_theta`, stacked in `se`, certified by
    `se_valid`), `loglik`, `aic`, `bic`, the empirical Kendall `tau` and
    the fit-implied `tau_implied`, and the closed-form tail-dependence
    coefficients `tail_lower`/`tail_upper` (Gaussian/Frank 0 — the classic
    reason a Gaussian fit understates joint crashes; t symmetric
    Demarta-McNeil; Clayton lower 2^(-1/theta); Gumbel upper
    2 - 2^(1/theta)). Keys: family, method, n, params, param_names, rho,
    nu, theta, se, se_rho, se_nu, se_theta, se_valid, loglik, aic, bic,
    tau, tau_implied, tail_lower, tail_upper, converged (rho/nu/theta and
    their se_* appear per family).

### `copula_select`

```python
def copula_select(
    u: _ArrayLike,
    families: Sequence[str] | None = ...,
    method: str = ...,
) -> dict[str, Any]:
```

Fits several copula families to the same (n, 2) pseudo-observations
    and ranks them by AIC/BIC, with a teaching verdict.

    `families`: list of names (default all five: gaussian, t, clayton,
    gumbel, frank); `method` as in `copula_fit`. Families whose domain
    excludes the data (Clayton/Gumbel under Kendall tau <= 0) are
    *skipped with a reason* rather than failing the call, so the default
    menu works on any data. Each entry of `fits` is a full `copula_fit`
    dict; `ranking_aic`/`ranking_bic` list family names best-first;
    `best_aic`/`best_bic` name the winners; `verdict` states who wins, by
    how much, whether AIC and BIC agree (they differ exactly when the
    extra parameter is not earning its keep by BIC), what the winner
    implies for tail dependence, and what was skipped and why. Keys:
    fits, skipped, best_aic, best_bic, ranking_aic, ranking_bic, verdict.

