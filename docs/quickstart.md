# tsecon Quickstart — 60 seconds to your first impulse response

This is the on-ramp. In one page you will install **tsecon**, fit a vector
autoregression to a bundled dataset, and read an impulse response off it — the
single most common thing an empirical macroeconomist asks a time-series
library to do. Everything below runs today against the shipped API; the code
blocks are the same ones the test suite exercises.

> **A note on the name.** `tsecon` is a working codename. The final name (and
> its PyPI availability) is settled before the first public release — see
> [ROADMAP.md §9](../ROADMAP.md). Until then, `tsecon` is what you install and
> what you import.

---

## Install

tsecon is a compiled Rust extension with a thin Python API, distributed as a
wheel. Build it from the repository with [maturin](https://www.maturin.rs/) and
install the result:

```sh
maturin build --release                       # writes target/wheels/tsecon-0.0.1-*.whl
pip install target/wheels/tsecon-0.0.1-*.whl  # the codename package: `tsecon`
```

The core wheel depends only on NumPy. Plotting is opt-in (`pip install
'tsecon[plots]'` pulls in matplotlib). Confirm the install and see how much is
on the shelf:

```python
import tsecon
print(tsecon.__version__)                                       # 0.0.1
print(sum(callable(getattr(tsecon, n)) for n in dir(tsecon)     # 78
          if not n.startswith("_")))
```

---

## Hello, impulse response

The repository ships golden fixtures in [`fixtures/`](../fixtures/) — the same
data the library is validated against. One of them, `var.json`, holds 100
quarters of (100×dlog) GDP, consumption, and investment growth. Run this from
the repository root:

```python
import json, numpy as np, tsecon

y = np.array(json.load(open("fixtures/var.json"))["data_100dlog_gdp_cons_inv"])
fit = tsecon.var_fit(y, lags=2)                                  # VAR(2) by OLS
irf = np.array(tsecon.var_irf(y, lags=2, horizon=10, orth=True))  # [horizon][response][shock]

np.set_printoptions(precision=3, suppress=True)
print(irf[:, 1, 0])   # consumption's response to a one-SD GDP shock, h = 0..10
```

```
[0.395 0.107 0.106 0.056 0.035 0.022 0.013 0.008 0.005 0.003 0.002]
```

That is the whole idea: a one-standard-deviation surprise to GDP lifts
consumption by 0.40 on impact, and the effect decays smoothly toward zero over
the following quarters — a stable, sensible dynamic response. `orth=True`
orthogonalizes the shocks through the Cholesky factor of the residual
covariance, using the column ordering of `y` (GDP → consumption → investment);
`irf[h][i][j]` is the response of variable `i` to a shock in variable `j` at
horizon `h`.

The `fit` object carries the rest of the story: `fit["params"]`, `fit["aic"]`
/ `fit["bic"]` / `fit["hqic"]`, the residual covariance `fit["sigma_u"]`, and
and `fit["is_stable"]` — the stability verdict. (The roots are the *reciprocal*
characteristic roots, so a stable VAR keeps them all outside the unit circle;
`fit["min_root"] > 1` is the equivalent numeric check, while `fit["max_root"]`
is the root farthest from the circle and is not a verdict on its own.) From
here, `tsecon.var_fevd`, `tsecon.var_forecast`, and `tsecon.var_granger` take
the same `(data, lags)` arguments.

---

## The API at a glance

The 93 functions, grouped by the task they serve. Every one is a plain
function that takes arrays and returns a NumPy array or a dict of documented
keys — no fit/predict objects to learn. Authoritative signatures, defaults,
and docstrings live in
[`bindings/python/tsecon.pyi`](../bindings/python/tsecon.pyi).

### Diagnostics and data prep

| Function | What it does |
|---|---|
| `acf` | Autocorrelation function with Bartlett standard errors |
| `pacf` | Partial autocorrelations (Yule-Walker or OLS) |
| `ljung_box` | Portmanteau white-noise test |
| `jarque_bera` | Normality test from skewness and kurtosis |
| `arch_lm` | Engle's test for conditional heteroskedasticity |
| `adf` | Augmented Dickey-Fuller unit-root test (MacKinnon p-values) |
| `kpss` | KPSS stationarity test — the ADF complement |
| `check_stationarity` | The ADF + KPSS confirmatory workflow, with a recommendation |
| `ols` | Linear regression with nonrobust / HC / HAC standard errors |
| `long_run_variance` | Kernel long-run variance of a series |
| `periodogram` | Raw spectral density (matches SciPy) |
| `welch` | Welch averaged-periodogram spectral density |
| `coherence` | Magnitude-squared coherence between two series |
| `bootstrap_indices` | iid / moving-block / circular / stationary resampling indices |
| `optimal_block_length` | Politis-White automatic block length |
| `philox_uniforms` | Reproducible uniform draws, bit-identical to NumPy |

### Univariate models and filters

| Function | What it does |
|---|---|
| `arima_fit` | Exact-MLE ARIMA(p,d,q) with optional forecast bands |
| `ar_loglik` | Exact Gaussian AR(p) log-likelihood at fixed parameters |
| `local_level_smooth` | Local-level Kalman filter + smoother (handles missing data) |
| `hp_filter` | Hodrick-Prescott trend/cycle decomposition |
| `bk_filter` | Baxter-King band-pass filter |
| `cf_filter` | Christiano-Fitzgerald band-pass filter |
| `hamilton_filter` | Hamilton's regression-based HP alternative |
| `markov_switching_ar` | Regime-switching AR fitted by EM (Hamilton 1989) |

### Volatility

| Function | What it does |
|---|---|
| `garch_fit` | GARCH / GJR / EGARCH by QMLE with robust standard errors |
| `gas_volatility` | GAS(1,1) score-driven volatility |
| `ccc_garch` | Constant-conditional-correlation multivariate GARCH |
| `dcc_garch` | Dynamic-conditional-correlation multivariate GARCH |
| `realized_measures` | Realized variance, bipower variation, and jump component |
| `har_rv` | Corsi HAR-RV regression with HAC standard errors |
| `realized_quarticity` | Realized quarticity (the sampling variance of RV) |
| `tripower_quarticity` | Jump-robust integrated quarticity |
| `bns_jump_test` | Barndorff-Nielsen-Shephard ratio jump test |
| `realized_range` | Parkinson / Garman-Klass range variance from OHLC bars |

### Multivariate and structural

| Function | What it does |
|---|---|
| `var_fit` | Fit a VAR(p) by OLS: params, covariance, ICs, stability |
| `var_irf` | Orthogonalized or raw impulse responses |
| `var_fevd` | Forecast-error variance decomposition |
| `var_forecast` | Iterated VAR forecasts with intervals |
| `var_granger` | Granger-causality F test |
| `sign_restricted_svar` | Sign-restricted Bayesian SVAR identified-set bands |
| `favar` | Factor-augmented VAR policy-shock IRFs (Bernanke-Boivin-Eliasz) |
| `johansen` | Johansen cointegration rank test |
| `vecm` | VECM maximum-likelihood estimation |
| `connectedness` | Diebold-Yilmaz spillover / connectedness measures |
| `factor_model` | PCA factor model with Bai-Ng factor selection |

### Local projections

| Function | What it does |
|---|---|
| `lp` | Local-projection impulse responses (lag-augmented or HAC SEs) |
| `lp_iv` | Instrumented local projections with a first-stage F diagnostic |
| `lp_state` | State-dependent (interacted) local projections (Ramey-Zubairy) |

### Forecasting and evaluation

| Function | What it does |
|---|---|
| `theta_forecast` | The Theta method (Assimakopoulos-Nikolopoulos) |
| `accuracy` | Forecast accuracy measures (RMSE, MAE, MAPE, MASE, RMSSE…) |
| `backtest` | Rolling / expanding pseudo-out-of-sample backtest |
| `dm_test` | Diebold-Mariano equal-accuracy test (HLN-corrected) |
| `cw_test` | Clark-West test for nested models |
| `gw_test` | Giacomini-White test of equal predictive ability |

### Bayesian

| Function | What it does |
|---|---|
| `bvar_fit` | Minnesota-NIW conjugate BVAR posterior + log marginal likelihood |
| `bvar_irf_draws` | Posterior Cholesky-IRF draws for credible bands |
| `mcmc_diagnostics` | Split R-hat and bulk/tail effective sample size |

### Panel time series

| Function | What it does |
|---|---|
| `panel_fe` | Fixed-effects panel OLS (cluster or Driscoll-Kraay SEs) |
| `panel_lp` | Panel local projection of a common shock |
| `panel_mean_group` | Mean-group / CCE-MG heterogeneous-panel estimator (Pesaran) |
| `panel_pmg` | Pooled Mean Group ARDL estimator (Pesaran-Shin-Smith) |
| `mean_group_var` | Pesaran-Smith mean-group panel VAR |

### Nowcasting and mixed frequency

| Function | What it does |
|---|---|
| `dfm_nowcast` | Dynamic-factor-model nowcast over a ragged data edge |
| `dfm_news` | News / update decomposition of a nowcast revision |
| `midas_weights` | MIDAS weighting kernels (exp-Almon or beta) |
| `umidas` | Unrestricted mixed-frequency (U-MIDAS) regression |
| `weighted_midas` | Weighted MIDAS estimated by nonlinear least squares |

### Regression, machine learning, and GMM

| Function | What it does |
|---|---|
| `ridge` | Ridge regression, closed form (scikit-learn objective) |
| `lasso` | Lasso via coordinate descent |
| `elastic_net` | Elastic net via coordinate descent |
| `adaptive_lasso` | Adaptive LASSO with oracle-property weights (Zou) |
| `lasso_path` | Elastic-net regularization path with AIC/BIC selection |
| `cv_splits` | Leakage-safe CV splits (expanding / rolling / purged k-fold) |
| `iv_gmm` | Linear IV-GMM with robust or HAC weighting and a Hansen J test |
| `gmm_nonlinear` | Nonlinear GMM over a Python moment function |

### Term structure

| Function | What it does |
|---|---|
| `nelson_siegel` | Nelson-Siegel yield-curve fit (Diebold-Li) |
| `svensson` | Svensson four-factor yield-curve fit |
| `dynamic_ns` | Dynamic Nelson-Siegel factors + one-step forecast |

---

## Where to go next

- **Not sure which model your problem calls for?** Start at the
  [which-model-when guide](which-model-when.md) — symptom-driven entry points
  ("my series is persistent and I need an impulse response"; "I have quarterly
  GDP and monthly indicators") that route you to the right estimator.
- **Want to learn the ideas, not just the calls?** The
  [tsecon Guide to Time Series Econometrics](guide/README.md) is a free,
  full-length course — from your first autocorrelation plot to research-grade
  structural identification — with runnable code in every chapter.
- **Want to see each method worked end to end?** The
  [gallery](examples/README.md) shows every function with a use case, code on
  real data, and the figure it produces.
</content>
</invoke>
